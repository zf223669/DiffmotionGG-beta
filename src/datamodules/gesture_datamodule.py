from typing import Optional
from pytorch_lightning import LightningDataModule
import joblib as jl
import os
from src import utils
import numpy as np
from src.datamodules.components.motion_data import MotionDataset, TestDataset
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import LightningLoggerBase
from src import utils

log = utils.get_pylogger(__name__)


class GestureDataModule(LightningDataModule):
    def __init__(self,
                 data_dir: str = "data/GesturesData",
                 framerate: str = 20,
                 seqlen: int = 5,
                 n_lookahead: int = 20,
                 dropout: float = 0.4,
                 batch_size: int = 80,
                 ):
        super(GestureDataModule, self).__init__()
        self.test_dataset = None
        self.n_test = None
        self.n_cond_channels = None
        self.n_x_channels = None
        # log.info("-------------------Init DiffFlowDataModule----------------")
        self.val_data_loader = None
        self.train_data_loader = None
        self.test_input = None
        self.validation_dataset = None
        self.data_pipe = None
        self.output_scaler = None
        self.input_scaler = None
        self.train_dataset = None

        self.data_root = data_dir
        self.framerate = framerate
        self.seqlen = seqlen
        self.n_lookahead = n_lookahead
        self.dropout = dropout
        self.batch_size = batch_size

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

    # def prepare_data(self) :
    #     pass

    def setup(self, stage: Optional[str] = None) -> None:
        # log.info(f"Diff Flow Datamodule => {sys._getframe().f_code.co_name}()")
        # load scalers
        print(os.path.join(self.data_root, 'input_scaler.sav'))
        self.input_scaler = jl.load(os.path.join(self.data_root, 'input_scaler.sav'))
        self.output_scaler = jl.load(os.path.join(self.data_root, 'output_scaler.sav'))
        # log.info(self.input_scaler)
        # log.info(self.output_scaler)

        # load pipeline for conversion from motion features to BVH.
        self.data_pipe = jl.load(os.path.join(self.data_root, 'data_pipe_' + self.framerate + 'fps.sav'))
        # log.info(self.data_pipe)

        # load the data. This should already be Standardized
        train_input = np.load(os.path.join(self.data_root, 'train_input_' + self.framerate + 'fps.npz'))[
            'clips'].astype(np.float32)  # Train Audio input [8428,120,27] [B,S,F]
        train_output = np.load(os.path.join(self.data_root, 'train_output_' + self.framerate + 'fps.npz'))[
            'clips'].astype(np.float32)  # Train Gesture output [8428,120,45] [B,S,F]
        val_input = np.load(os.path.join(self.data_root, 'val_input_' + self.framerate + 'fps.npz'))[
            'clips'].astype(np.float32)  # Value Audio input[264,120,27] [B,S,F]
        val_output = np.load(os.path.join(self.data_root, 'val_output_' + self.framerate + 'fps.npz'))[
            'clips'].astype(np.float32)  # Value Gesture output[264,120,45] [B,S,F]
        # log.info(np.shape(train_input) + np.shape(train_output) + np.shape(val_input) + np.shape(val_output))
        # Create pytorch data sets
        self.train_dataset = MotionDataset(control_data=train_input, joint_data=train_output, framerate=self.framerate,
                                           seqlen=self.seqlen, n_lookahead=self.n_lookahead,
                                           dropout=self.dropout)
        self.validation_dataset = MotionDataset(control_data=val_input, joint_data=val_output, framerate=self.framerate,
                                                seqlen=self.seqlen, n_lookahead=self.n_lookahead,
                                                dropout=self.dropout)

        # test data for network tuning. It contains the same data as val_input, but sliced into longer 20-sec exerpts
        test_input = np.load(os.path.join(self.data_root, 'dev_input_' + self.framerate + 'fps.npz'))[
            'clips'].astype(np.float32)
        log.info(self.test_input)

        # make sure the test data is at least one batch size
        self.n_test = test_input.shape[0]
        # ---------------------------7.04 -------------------
        # TODO need to pass DiffFlow.yaml parameter to below seqlen:80
        # n_tiles = 1 + hparams.Train.batch_size // self.n_test
        n_tiles = 1 + 80 // self.n_test
        test_input = np.tile(test_input.copy(), (n_tiles, 1, 1))

        # initialise test output with zeros (mean pose)
        self.n_x_channels = self.output_scaler.mean_.shape[0]
        self.n_cond_channels = self.n_x_channels * self.seqlen + test_input.shape[2] * (
                    self.seqlen + 1 + self.n_lookahead)
        test_output = np.zeros((test_input.shape[0], test_input.shape[1], self.n_x_channels)).astype(np.float32)
        self.test_dataset = TestDataset(test_input, test_output)

    def train_dataloader(self):
        train_data_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=1,
            shuffle=True,
            drop_last=True,
        )
        return train_data_loader

    def val_dataloader(self):
        val_data_loader = DataLoader(
            dataset=self.validation_dataset,
            batch_size=self.batch_size,
            num_workers=8,
            shuffle=False,
            drop_last=True
        )
        return val_data_loader

    def test_dataloader(self):
        test_data_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=1,
            shuffle=False,
            drop_last=True
        )
        return test_data_loader

    def predict_dataloader(self):
        predict_data_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=1,
            shuffle=False,
            drop_last=True
        )
        return predict_data_loader
