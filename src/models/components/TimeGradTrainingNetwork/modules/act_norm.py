import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import scipy.linalg
from . import thops
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from src import utils

log = utils.get_pylogger(__name__)


class _ActNorm(nn.Module):
    """
    Activation Normalization
    Initialize the bias and scale with a given minibatch,
    so that the output per-channel have zero mean and unit variance for that.

    After initialization, `bias` and `log_std` will be trained as parameters.
    """

    def __init__(self, num_features, scale=1.):
        super().__init__()
        # register mean and scale
        size = [1, 1, num_features]  # [1,1,45]
        self.register_parameter("bias", nn.Parameter(torch.zeros(*size)))  # [1,1,45] [0,0,0,0,0,0,,,,,,0]
        self.register_parameter("log_std", nn.Parameter(torch.zeros(*size)))  # log_std
        self.num_features = num_features
        self.scale = float(scale)
        self.inited = False
        # self.inited = True

    def _check_input_dim(self, input):
        return NotImplemented

    def initialize_parameters(self, input):
        self._check_input_dim(input)
        if not self.training:
            return
        assert input.device == self.bias.device
        with torch.no_grad():
            bias = thops.mean(input.clone(), dim=[0, 1], keepdim=True) * -1.0  # learnable [1,1,,45] μb
            vars = thops.mean((input.clone() + bias) ** 2, dim=[0, 1], keepdim=True)  # [1,1,,45] sigma ** 2
            log_std = torch.log(self.scale / (torch.sqrt(vars) + 1e-6))  # learnable  BN`s normalize with log
            self.bias.data.copy_(bias.data)
            self.log_std.data.copy_(log_std.data)
            # log.info(f'Act norm initialize_parameters bias: {self.bias}')
            # log.info(f'Act norm initialize_parameters log_std: {self.log_std}')

            self.inited = True

    def _center(self, input, reverse=False):
        if not reverse:
            return input + self.bias
        else:
            # log.info(f'Actnorm _center reverse bias: {self.bias}')
            return input - self.bias

    def _scale(self, input, logdet=None, reverse=False):
        log_std = self.log_std
        if not reverse:
            input = input * torch.exp(log_std)  # BN normalising
        else:
            # log.info(f'Actnorm _center reverse log_std: {self.log_std}')
            input = input * torch.exp(-log_std)
        if logdet is not None:
            """
            log_std is log_std of `mean of channels`
            so we need to multiply timesteps
            """
            dlogdet = thops.sum(log_std) * thops.timesteps(input)  # seq * sum(log|s|) <- h*w*sum(log|s|)
            if reverse:
                dlogdet *= -1
            logdet = logdet + dlogdet
            # print('>>>>>>>logdet = logdet + dlogdet<<<<<<<')
        return input, logdet

    def forward(self, input, logdet=None, reverse=False):
        if not self.inited:
            self.initialize_parameters(input)
        self._check_input_dim(input)
        # no need to permute dims as old version
        if not reverse:
            # center and scale
            input = self._center(input, reverse)  # input + bias or x - μb
            input, logdet = self._scale(input, logdet,
                                        reverse)  # input = input * torch.exp(log_std) = (input + bias) * torch.exp(log_std)
        else:
            # scale and center
            input, logdet = self._scale(input, logdet, reverse)
            input = self._center(input, reverse)

        return input, logdet


class ActNorm2d(_ActNorm):  # inherit from ActNorm
    def __init__(self, num_features, scale=1.):
        super().__init__(num_features, scale)
        # print('---------------ActNorm2d-------------')

    def _check_input_dim(self, input):
        assert len(input.size()) == 3
        # print('---------------ActNorm2d--_check_input_dim------')
        assert input.size(2) == self.num_features, (
            "[ActNorm]: input should be in shape as `BCT`,"
            " channels should be {} rather than {}".format(
                self.num_features, input.size()))
