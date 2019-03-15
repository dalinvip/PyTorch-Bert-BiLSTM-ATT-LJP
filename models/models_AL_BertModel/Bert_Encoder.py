# @Author : bamtercelboo
# @Datetime : 2019/3/15 9:29
# @File : Bert_Encoder.py
# @Last Modify Time : 2019/3/15 9:29
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  Bert_Encoder.py
    FUNCTION : None
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable as Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random
import torch.nn.init as init
import numpy as np
import time
from .HierachicalAtten import HierachicalAtten
from wheel.signatures import assertTrue
from .initialize import *

from DataUtils.Common import *
torch.manual_seed(seed_num)
random.seed(seed_num)


class Bert_Encoder(nn.Module):
    """
        Bert_Encoder
    """
    def __init__(self, **kwargs):
        super(Bert_Encoder, self).__init__()
        self.bi_input = kwargs["bi_input"]
        self.bi_hidden = kwargs["bi_hidden"]
        self.bi_layers = kwargs["bi_layers"]
        self.bi_directional = kwargs["bi_directional"]
        self.dropout = kwargs["dropout"]

        self.dropout_bert = nn.Dropout(self.dropout)

        self.bert_bilstm = nn.LSTM(input_size=self.bi_input, hidden_size=self.bi_hidden,
                                   num_layers=self.bi_layers, bidirectional=self.bi_directional,
                                   bias=True)

        self.bert_linear = nn.Linear(in_features=self.bi_hidden * 2, out_features=200, bias=True)
        init_linear_weight_bias(self.bert_linear)

    def forward(self, bert_h):
        """
        :param bert_h:
        :return:
        """
        # print(bert_h)
        h, c = self.bert_bilstm(bert_h)
        h = h.permute(0, 2, 1)
        h = F.max_pool1d(h, h.size(2)).squeeze(2)
        bert_fea = self.bert_linear(h)
        return bert_fea


