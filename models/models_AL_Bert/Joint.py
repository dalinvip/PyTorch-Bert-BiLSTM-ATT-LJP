# @Author : bamtercelboo
# @Datetime : 2018/07/10 16.03
# @File : train.py
# @Last Modify Time : 2018/07/10 16.03
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  model.py
    FUNCTION : None
"""

import torch
import torch.nn as nn
import random
import numpy as np
import time
from .Encoder import Encoder
from .Decoder import Decoder
from DataUtils.Common import *
torch.manual_seed(seed_num)
random.seed(seed_num)


class Joint_Bert(nn.Module):
    """
     Joint
    """

    def __init__(self, config):
        super(Joint_Bert, self).__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, x, train=False):
        """
        :param x:
        :param train:
        :return:
        """
        es_time = time.time()
        encoder = self.encoder(x, train)
        e_time = time.time() - es_time
        # print("encoder size {}".format(encoder.size()))
        ed_time = time.time()
        accu, law = self.decoder(encoder)
        d_time = time.time() - ed_time
        return accu, law, e_time, d_time


