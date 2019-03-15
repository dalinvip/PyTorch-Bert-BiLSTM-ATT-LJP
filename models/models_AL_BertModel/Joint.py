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
from .BERT import BERT
from DataUtils.Common import *
torch.manual_seed(seed_num)
random.seed(seed_num)


class Joint_BertModel(nn.Module):
    """
     Joint
    """

    def __init__(self, config):
        super(Joint_BertModel, self).__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.bert_model = BERT(bert_model=config.bert_model_path, vocab=config.bert_model_vocab,
                               max_seq_length=config.bert_model_max_seq_length, batch_size=config.bert_model_batch_size,
                               extract_dim=config.extract_dim, layers=config.layers, local_rank=config.local_rank,
                               no_cuda=config.no_cuda, do_lower_case=config.do_lower_case)

    def forward(self, batch_features, train=False):
        """
        :param batch_features:
        :param train:
        :return:
        """
        bert_hidden = self.bert_model.extract_feature(batch_features)
        es_time = time.time()
        encoder = self.encoder(batch_features, bert_hidden, train)
        e_time = time.time() - es_time
        # print("encoder size {}".format(encoder.size()))
        ed_time = time.time()
        accu, law = self.decoder(encoder)
        d_time = time.time() - ed_time
        return accu, law, e_time, d_time


