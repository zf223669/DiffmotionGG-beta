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
from src.models.components.TimeGradTrainingNetwork.modules.distribution_output import GaussianDiag

from .epsilon_theta import EpsilonTheta
from src import utils
from tqdm import tqdm
import ipdb
from src.models.components.TimeGradTrainingNetwork.modules.act_norm import ActNorm2d

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

class LinearZeroInit(nn.Linear):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)
        # init
        # print("-----------------LinearZeroInit----------------")
        self.weight.data.zero_()
        self.bias.data.zero_()


class TimeGradTrainingNetwork(nn.Module):
    @validated()
    def __init__(
            self,
            input_size: int,  # 972
            num_layers: int,  # 2
            num_cells: int,  # 512
            cell_type: str,  # LSTM / GRU
            prediction_length: int,  # 24
            dropout_rate: float,
            target_dim: int,  # 370
            conditioning_length: int,  # 100
            diff_steps: int,
            loss_type: str,
            beta_end: float,
            beta_schedule: str,
            residual_layers: int,
            residual_channels: int,
            dilation_cycle_length: int,
            scaling: bool = True,
            **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        log.info(f"-------------------Init TimeGradTrainingNetwork----------------")
        self.target_dim = target_dim
        self.prediction_length = prediction_length
        self.scaling = scaling

        self.cell_type = cell_type
        self.init_rnn = True

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
        self.normal_distribution = GaussianDiag()

        # self.proj_dist_args = self.distr_output.get_args_proj(num_cells)

        # self.embed_dim = 1
        # self.embed = nn.Embedding(
        #     num_embeddings=self.target_dim, embedding_dim=self.embed_dim
        # )
        # self.BatchNorm = nn.BatchNorm1d(num_features=45)

        self.actnorm = ActNorm2d(45, 1.0)

        # self.linear = LinearZeroInit(num_cells, 512)

        self.forwardCount = 1
        # if self.scaling:
        #     self.scaler = MeanScaler(keepdim=True)
        # else:
        #     self.scaler = NOPScaler(keepdim=True)

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

        x_input_scaled, _ = self.actnorm(x_input, None, reverse=False)
        combined_input = torch.cat((x_input_scaled, cond), dim=-1)

        rnn_outputs, state = self.rnn(combined_input)
        distr_args = self.distr_args(rnn_outputs=rnn_outputs)
        likelihoods = self.diffusion.log_prob(x_input_scaled, distr_args).unsqueeze(-1)
        # log.info(f'likelihoods shape: {likelihoods}')
        return likelihoods, likelihoods.mean()


class TimeGradPredictionNetwork(TimeGradTrainingNetwork):
    def __init__(self, bvh_save_path: str, **kwargs) -> None:
        super().__init__(**kwargs)
        # self.init_rnn = True
        self.bvh_save_path = bvh_save_path
        log.info(f"-------------------Init TimeGradPredictionNetwork----------------")

        # for decoding the lags are shifted by one,
        # at the first time-step of the decoder a lag of one corresponds to
        # the last target value
        # self.shifted_lags = [l - 1 for l in self.lags_seq]

    def prepare_cond(self, jt_data, ctrl_data):
        # log.info(f'type of jt_data {jt_data.device} ,ctrl_data : {ctrl_data.device}')
        jt_data = jt_data.cuda()
        ctrl_data = ctrl_data.cuda()
        # log.info(f'to cuda type of jt_data {jt_data.device} ,ctrl_data : {ctrl_data.device}')
        nn, seqlen, n_feats = jt_data.shape
        # log.info('prepare_cond........')
        jt_data = torch.reshape(jt_data, (nn, seqlen * n_feats))  # jt_data [80,225]
        nn, seqlen, n_feats = ctrl_data.shape
        ctrl_data = torch.reshape(ctrl_data, (nn, seqlen * n_feats))  # ctrl_data [80,702]
        # log.info(f'jt_data shape: {jt_data.shape}, ctrl_data shape: {ctrl_data.shape}')
        # #jt_data [80,225]  ctrl_data [80,702]
        cond = torch.cat((jt_data, ctrl_data), 1)  # [80,927]
        # log.info(f'pre')
        # cond1 = torch.unsqueeze(cond, -1)
        # cond1 = torch.swapaxes(cond1, 1, 2) # [80, 1, 927]

        cond = torch.unsqueeze(cond, 1)  # [80,1,927]
        # log.info(cond1.equal(cond2))
        # log.info(f'prepare_cond cond shape: {cond.shape}')
        return cond

    def sampling_decoder(
            self,
            autoreg: torch.Tensor,
            begin_states: Union[List[torch.Tensor], torch.Tensor],
            control_all: torch.Tensor,
            sampled_all: torch.Tensor,
            seqlen: int,
            n_lookahead: int,
            scale: torch.Tensor,
    ) -> torch.Tensor:
        # torch.set_printoptions(threshold=sys.maxsize)
        np.set_printoptions(threshold=500)
        future_samples = sampled_all.cpu().numpy()  # [0,0,0,0,0,,,,,,] shape:[80,380,45]
        log.info(f'future_samples :{future_samples.shape}')
        if self.scaling:
            self.diffusion.scale = scale
        autoreg = autoreg
        states = begin_states

        # for each future time-units we draw new samples for this time-unit
        # and update the state

        for k in tqdm(range(self.prediction_length)):
            # log.info(f'prediction_length = {self.prediction_length}')
            # big bug!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            control = control_all[:, k:(k + seqlen + 1 + n_lookahead), :]
            log.info(f'control shape: {control.shape}')
            log.info(f'autoreg shape: {autoreg.shape}')
            combined_cond = self.prepare_cond(autoreg, control)
            log.info(f'cond shape: {combined_cond.shape}')
            z = self.normal_distribution.sample([80, 1, 45], 1e-08, "cuda:0")
            log.info(f'z shape: {z.shape}')
            # combined_input = torch.cat((z, combined_cond), dim=-1)  # [80,1,972]
            # rnn_output, state, scale, inputs = self.unroll_encoder(x_input=z, cond=control,
            #                                                        combined_input=combined_cond, reverse=True)
            # distr_args = self.distr_args(rnn_outputs=rnn_output)
            # log.info(f'rnn_outputs shape: {rnn_outputs.shape}')
            # (batch_size, 1, target_dim)
            new_samples = self.diffusion.sample(combined_cond)
            new_samples, _ = self.actnorm(new_samples, None, reverse=True)
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
            print(f'--->autoreg shape:{autoreg.shape} \n {autoreg}')
        # (batch_size * num_samples, prediction_length, target_dim)
        # samples = torch.cat(future_samples, dim=1)
        log.info(f'samples length: {future_samples.size}')
        # (batch_size, num_samples, prediction_length, target_dim)
        return future_samples

    def forward(self, autoreg_all, control_all, trainer) -> torch.Tensor:
        datamodule = trainer.datamodule
        seqlen = datamodule.seqlen
        n_lookahead = datamodule.n_lookahead

        batch, n_timesteps, n_feats = autoreg_all.shape
        sampled_all = torch.zeros((batch, n_timesteps - n_lookahead, n_feats))  # [80,380,45]
        autoreg = torch.zeros((batch, seqlen, n_feats), dtype=torch.float32)  # [80,5,45]

        sampled_all[:, :seqlen, :] = autoreg  # start pose [0,0,0,0,0]

        # z = self.normal_distribution.sample(z_shape=(80, 1, 45), eps_std=1,
        #                                     device="cuda:0")  # [80,1,45] # should create in Gaussian_diffusion: p_sample_loop
        # # ipdb.set_trace()
        # control = control_all[:, :seqlen + 1 + n_lookahead, :]  # [80,26,27]
        # combined_cond = self.prepare_cond(torch.clone(autoreg), torch.clone(control))
        # combined_input = torch.cat((z, combined_cond), dim=-1)  # [80,1,972]
        #
        # rnn_output, state, scale, inputs = self.unroll_encoder(x_input=z, cond=control, combined_input=combined_input, reverse=True)

        x = self.sampling_decoder(autoreg=autoreg, control_all=control_all, begin_states=None,
                                  sampled_all=sampled_all,
                                  seqlen=seqlen, n_lookahead=n_lookahead,
                                  )
        # log.info(f'final x shape: {x.shape} x = \n {x}')
        datamodule.save_animation(control_all[:, :(n_timesteps - n_lookahead), :], sampled_all,
                                  self.bvh_save_path)
        return x
