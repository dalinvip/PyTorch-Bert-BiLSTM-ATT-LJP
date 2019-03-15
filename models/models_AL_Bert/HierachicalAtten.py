# @Author : bamtercelboo
# @Datetime : 2018/7/24 10:26
# @File : HierachicalAtten.py
# @Last Modify Time : 2018/7/24 10:26
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  HierachicalAtten.py
    FUNCTION : None
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import random
import time
from DataUtils.Common import *
from models.models_AL.initialize import *
torch.manual_seed(seed_num)
random.seed(seed_num)


class HierachicalAtten(nn.Module):
    """
        HierachicalAtten
    """
    def __init__(self, in_size, attention_size, config):
        """
        :Function Hierachical Attention initial
        :param in_size: Linear input size
        :param attention_size: attention size
        :param config: config file
        """
        super(HierachicalAtten, self).__init__()
        self.config = config
        self.in_size = in_size
        self.attention_size = attention_size

        self.MLP_linear = nn.Linear(in_features=self.in_size, out_features=self.attention_size, bias=True)
        init_linear_weight_bias(self.MLP_linear)

        self.context_vector = nn.Linear(self.attention_size, 1, bias=False)
        init_linear_weight_bias(self.context_vector)

        self.dropout = nn.Dropout(p=0.0)

        self.SM = nn.Softmax(dim=1)

    def forward(self, input):
        mlp_out = torch.tanh(self.MLP_linear(input))     ##  B * T * H
        score = self.SM(self.context_vector(mlp_out))  ##  B * T * 1
        input_atten = torch.mul(input, score)  ##  B * T * H
        att_out = torch.sum(input_atten, dim=1, keepdim=False)   ##  B * H
        return att_out




