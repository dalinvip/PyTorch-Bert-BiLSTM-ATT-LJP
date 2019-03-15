# @Author : bamtercelboo
# @Datetime : 2018/07/10 16.03
# @File : train.py
# @Last Modify Time : 2018/07/10 16.03
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  Decoder.py
    FUNCTION : Joint model decoder
"""

import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import time
from torch.autograd import Variable
from models.models_AL.initialize import *


from DataUtils.Common import *
torch.manual_seed(seed_num)
random.seed(seed_num)


class Decoder(nn.Module):

    def __init__(self, config):
        super(Decoder, self).__init__()
        self.config = config
        self.mlp_out_size = 600

        self.bert_dim = config.bert_dim

        # nonLiner
        self.nonLinear = nn.Linear(in_features=config.lstm_hiddens * 2 + 600,
                                   out_features=self.mlp_out_size, bias=True)
        init_linear_weight_bias(self.nonLinear)

        self.accu_law_nonlinear = nn.Linear(in_features=200, out_features=50, bias=True)
        init_linear_weight_bias(self.accu_law_nonlinear)

        # accu
        # self.accu_weight = nn.Parameter(torch.randn(config.accu_class_num, 200), requires_grad=True)
        # init.xavier_uniform(self.accu_weight)
        self.accu_embed = nn.Embedding(config.accu_class_num, 200)
        self.accu_weight = self.accu_embed.weight
        # self.accu_weight.requires_grad = False

        # law
        self.law_embed = nn.Embedding(config.law_class_num, 200)
        self.law_weight = self.law_embed.weight
        # self.law_weight.requires_grad = False

        self.accu_for_law_linear = nn.Linear(in_features=200, out_features=50, bias=False)
        init_linear_weight_bias(self.accu_for_law_linear)

        # accu and law
        # self.linear = nn.Linear(in_features=config.lstm_hiddens * 2, out_features=2, bias=True)
        self.linear = nn.Linear(in_features=50, out_features=2, bias=True)
        # init_linear(self.linear)
        init_linear_weight_bias(self.linear)

    def forward(self, x):
        """
        :param x:
        :return:
        """
        # x = F.tanh(self.nonLinear(x))
        x = self.nonLinear(x)
        x = x.unsqueeze(1)
        x = x.view(x.size(0), x.size(1), -1, 3).permute(3, 0, 1, 2)
        accu_x, law_x = x[0], x[1]

        accu = self.accu_forward(accu_x)

        law = self.law_forward(law_x)

        return accu, law

    def accu_forward(self, x):
        """
        :param x:
        :return:
        """
        # x = x.expand(x.size(0), self.config.accu_class_num, x.size(2))
        x = x.repeat(1, self.config.accu_class_num, 1)
        x = torch.mul(x, self.accu_weight)
        x = torch.tanh(self.accu_law_nonlinear(x))
        # x = F.tanh(x)
        accu = self.linear(x)
        return accu

    def law_forward(self, x):
        """
        :param x:
        :return:
        """
        # x = x.expand(x.size(0), self.config.law_class_num, x.size(2))
        x = x.repeat(1, self.config.law_class_num, 1)
        x = torch.mul(x, self.law_weight)
        x = torch.tanh(self.accu_for_law_linear(x))
        law = self.linear(x)
        return law




