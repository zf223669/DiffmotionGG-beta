import logging
import sys

import numpy as np
from torch.nn.modules import loss
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from gluonts.core.component import validated

from src.models.components.TimeGradTrainingNetwork.utils import weighted_average
from src.models.components.TimeGradTrainingNetwork.modules import GaussianDiffusion, DiffusionOutput, MeanScaler, \
    NOPScaler

from .epsilon_theta import EpsilonTheta
from src import utils
from tqdm import tqdm

log = utils.get_pylogger(__name__)


# import logging as log
#
# logger = log.getLogger('test')
# logger.setLevel(level=logging.INFO)
# fh = logging.FileHandler('/home/zf223669/Mount/pytorch-ts-2/test.log', mode='w')
# ch = logging.StreamHandler()
# ch.setLevel(level=logging.INFO)
# logger.addHandler(fh)
# logger.addHandler(ch)


class TimeGradTrainingNetwork(nn.Module):
    @validated()
    def __init__(
            self,
            input_size: int,  # 972
            num_layers: int,  # 2
            num_cells: int,  # 512
            cell_type: str,  # LSTM / GRU
            history_length: int,  # 192 24 + 168
            context_length: int,  # 24
            prediction_length: int,  # 24
            dropout_rate: float,
            # lags_seq: List[int],
            target_dim: int,  # 370
            conditioning_length: int,  # 100
            diff_steps: int,
            loss_type: str,
            beta_end: float,
            beta_schedule: str,
            residual_layers: int,
            residual_channels: int,
            dilation_cycle_length: int,
            # cardinality: List[int] = [1],
            embedding_dimension: int = 1,
            scaling: bool = True,
            **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        log.info(f"-------------------Init TimeGradTrainingNetwork----------------")
        self.target_dim = target_dim
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.history_length = history_length
        self.scaling = scaling

        self.cell_type = cell_type

        rnn_cls = {"LSTM": nn.LSTM, "GRU": nn.GRU}[cell_type]
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=num_cells,
            num_layers=num_layers,
            dropout=dropout_rate,
            batch_first=True,
        )

        self.denoise_fn = EpsilonTheta(  # εΘ
            target_dim=target_dim,
            cond_length=conditioning_length,
            residual_layers=residual_layers,
            residual_channels=residual_channels,
            dilation_cycle_length=dilation_cycle_length,
        )

        self.diffusion = GaussianDiffusion(  # Most Importance
            self.denoise_fn,
            input_size=target_dim,
            diff_steps=diff_steps,
            loss_type=loss_type,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
        )

        self.distr_output = DiffusionOutput(
            self.diffusion, input_size=target_dim, cond_size=conditioning_length
        )

        self.proj_dist_args = self.distr_output.get_args_proj(num_cells)

        self.embed_dim = 1
        self.embed = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.embed_dim
        )

        self.forwardCount = 1
        if self.scaling:
            self.scaler = MeanScaler(keepdim=True)
        else:
            self.scaler = NOPScaler(keepdim=True)

    def distr_args(self, rnn_outputs: torch.Tensor):
        """
        Returns the distribution of DeepVAR with respect to the RNN outputs.

        Parameters
        ----------
        rnn_outputs
            Outputs of the unrolled RNN (batch_size, seq_len, num_cells)
        scale
            Mean scale for each time series (batch_size, 1, target_dim)

        Returns
        -------
        distr
            Distribution instance
        distr_args
            Distribution arguments
        """
        (distr_args,) = self.proj_dist_args(rnn_outputs)

        # # compute likelihood of target given the predicted parameters
        # distr = self.distr_output.distribution(distr_args, scale=scale)

        # return distr, distr_args
        return distr_args

    def forward(
            self,
            x_input: torch.Tensor,
            cond: torch.Tensor, ):
        # log.info(f"-------------------forward----------------")
        rnn_output, state, inputs = self.unroll_rnn(x_input=x_input, cond=cond)
        distr_args = self.distr_args(rnn_outputs=rnn_output)
        # log.info(f'distr_args shape: {distr_args.shape}')
        likelihoods = self.diffusion.log_prob(x_input, distr_args).unsqueeze(-1)
        # log.info(f'likelihoods shape: {likelihoods}')
        return likelihoods, likelihoods.mean()

    def unroll_rnn(
            self,
            x_input: torch.Tensor,
            cond: torch.Tensor,
            begin_state: Union[List[torch.Tensor], torch.Tensor] = None, ):
        # log.info(f'x_input shape: {x_input.shape}, cond shape: {cond.shape}')
        inputs = torch.cat((x_input, cond), dim=-1)
        # log.info(f'inputs = : {inputs}')
        # log.info(f'inputs shape: {inputs.shape}')
        rnn_output, state = self.rnn(inputs, begin_state)
        # log.info(f'rnn_output shape: {rnn_output}')
        return rnn_output, state, inputs


class TimeGradPredictionNetwork(TimeGradTrainingNetwork):
    def __init__(self, num_parallel_samples: int, bvh_save_path: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_parallel_samples = num_parallel_samples
        self.bvh_save_path = bvh_save_path

        # for decoding the lags are shifted by one,
        # at the first time-step of the decoder a lag of one corresponds to
        # the last target value
        # self.shifted_lags = [l - 1 for l in self.lags_seq]

    def sampling_decoder(
            self,
            autoreg: torch.Tensor,
            cond: torch.Tensor,
            begin_states: Union[List[torch.Tensor], torch.Tensor],
            sampled_all: torch.Tensor,
            seqlen: int,
    ) -> torch.Tensor:
        """
        Computes sample paths by unrolling the RNN starting with a initial
        input and state.

        Parameters
        ----------
        past_target_cdf
            Past marginal CDF transformed target values (batch_size,
            history_length, target_dim)
        target_dimension_indicator
            Indices of the target dimension (batch_size, target_dim)
        time_feat
            Dynamic features of future time series (batch_size, history_length,
            num_features)
        scale
            Mean scale for each time series (batch_size, 1, target_dim)
        begin_states
            List of initial states for the RNN layers (batch_size, num_cells)
        Returns
        --------
        sample_paths : Tensor
            A tensor containing sampled paths. Shape: (1, num_sample_paths,
            prediction_length, target_dim).
            :param seqlen:
            :param autoreg:
            :param cond:
            :param begin_states:
            :param sampled_all:
        """
        # torch.set_printoptions(threshold=sys.maxsize)
        np.set_printoptions(threshold=500)
        future_samples = sampled_all.cpu().numpy()
        # log.info(f'future_samples :{future_samples.shape}')
        autoreg = autoreg
        states = begin_states
        # for each future time-units we draw new samples for this time-unit
        # and update the state

        for k in tqdm(range(self.prediction_length)):
            # log.info(f'prediction_length = {self.prediction_length}')
            rnn_outputs, states, _ = self.unroll_rnn(begin_state=states, x_input=autoreg, cond=cond)
            distr_args = self.distr_args(rnn_outputs=rnn_outputs)
            # log.info(f'rnn_outputs shape: {rnn_outputs.shape}')
            # (batch_size, 1, target_dim)
            new_samples = self.diffusion.sample(cond=distr_args)
            # log.info(f'new_samples : {type(new_samples)}, device: {new_samples.device}')
            new_samples = new_samples.cpu().numpy()[:, 0, :]
            # new_samples = new_samples[:, 0, :]
            future_samples[:, (k + seqlen), :] = new_samples
            # (batch_size, seq_len, target_dim)
            # future_samples.append(new_samples)

            # log.info(f'new_samples: \n{new_samples} \n future_samples: {future_samples}')
            autoreg = autoreg.cpu().numpy()
            # log.info(
            #     f'new_samples shape: {new_samples.shape} , future_samples shape: {future_samples.shape}, autoreg shape: {autoreg.shape}')
            # log.info(f'new_samples[:, None, :] shape : {new_samples[:, None, :].shape}')
            autoreg = np.concatenate((autoreg[:, 1:, :].copy(), new_samples[:, None, :]), axis=1)
            autoreg = torch.from_numpy(autoreg).cuda()
            # print(f'sampled_all shape: \n {autoreg}')
        # (batch_size * num_samples, prediction_length, target_dim)
        # samples = torch.cat(future_samples, dim=1)
        log.info(f'samples length: {future_samples.size}')
        # (batch_size, num_samples, prediction_length, target_dim)
        return future_samples

    def prepare_cond(self, jt_data, ctrl_data):
        log.info(f'type of jt_data {jt_data.device} ,ctrl_data : {ctrl_data.device}')
        jt_data = jt_data.cuda()
        ctrl_data = ctrl_data.cuda()
        log.info(f'to cuda type of jt_data {jt_data.device} ,ctrl_data : {ctrl_data.device}')
        nn, seqlen, n_feats = jt_data.shape
        print('prepare_cond........')
        jt_data = torch.reshape(jt_data, (nn, seqlen * n_feats))
        nn, seqlen, n_feats = ctrl_data.shape
        ctrl_data = torch.reshape(ctrl_data, (nn, seqlen * n_feats))

        cond = torch.cat((jt_data, ctrl_data), 1)
        log.info(f'pre')
        cond = torch.unsqueeze(cond, -1)
        cond = torch.swapaxes(cond, 1, 2)
        print(f'prepare_cond cond shape: {cond.shape}')
        return cond

    def forward(self, autoreg_all, control_all, trainer) -> torch.Tensor:
        datamodule = trainer.datamodule
        seqlen = datamodule.seqlen
        n_lookahead = datamodule.n_lookahead
        autoreg_all = autoreg_all.cuda()
        control_all = control_all.cuda()
        log.info(f'autoreg_all shape: {autoreg_all.shape}')
        log.info(f'control_all shape: {control_all.shape}')
        batch, n_timesteps, n_feats = autoreg_all.shape
        # log.info(f'batch = {batch}, n_timesteps = {n_timesteps}, n_feats = {n_feats}')
        sampled_all = torch.zeros((batch, n_timesteps - n_lookahead, n_feats))
        autoreg = torch.zeros((batch, seqlen, n_feats), dtype=torch.float32)
        begin_autoreg = autoreg_all[:, seqlen:seqlen + 1, :]
        # log.info(f'begin_autoreg shape: {begin_autoreg.shape}')

        sampled_all[:, :seqlen, :] = autoreg  # pass the first seqlen autoreg to the final sequence.
        # log.info(
        #     f'prediction: sampled_all shape :{np.shape(sampled_all)}  autoreg shape: {np.shape(autoreg)} autoreg_all: {np.shape(autoreg_all)}')
        control = control_all[:, 0:(seqlen + 1 + n_lookahead), :]
        cond = self.prepare_cond(autoreg, control)
        # log.info(f'prediction: cond.shape: {cond.shape}')
        rnn_output, begin_state, inputs = self.unroll_rnn(begin_autoreg, cond)
        self.prediction_length = sampled_all.shape[1] - seqlen
        x = self.sampling_decoder(autoreg=begin_autoreg, cond=cond, begin_states=begin_state, sampled_all=sampled_all,
                                  seqlen=seqlen)
        log.info(f'final x shape: {x.shape} x = \n {x}')
        datamodule.save_animation(control_all[:, :(n_timesteps - n_lookahead), :], sampled_all,
                                  self.bvh_save_path)
        return x
