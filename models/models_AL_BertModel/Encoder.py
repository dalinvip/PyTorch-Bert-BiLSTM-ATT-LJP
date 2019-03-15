# @Author : bamtercelboo
# @Datetime : 2018/07/10 16.03
# @File : train.py
# @Last Modify Time : 2018/07/10 16.03
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  Encoder.py
    FUNCTION : Joint encoder model
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
from .Bert_Encoder import Bert_Encoder
from wheel.signatures import assertTrue
from .initialize import *

from DataUtils.Common import *
torch.manual_seed(seed_num)
random.seed(seed_num)


class Encoder(nn.Module):
    """
        Encoder
    """

    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        self.word_Att_size = 200

        V = config.embed_num
        D = config.embed_dim
        paddingId = config.paddingId

        self.embed = nn.Embedding(V, D, padding_idx=paddingId)
        if config.pretrained_embed:
            self.embed.weight.data.copy_(config.pretrained_weight)

        self.dropout_embed = nn.Dropout(config.dropout_emb)
        self.dropout = nn.Dropout(config.dropout)

        self.bilstm = nn.LSTM(input_size=D, hidden_size=config.lstm_hiddens, num_layers=config.lstm_layers,
                              bidirectional=True, bias=True)

        self.hierachicalAtt = HierachicalAtten(in_size=config.lstm_hiddens * 2, attention_size=self.word_Att_size,
                                               config=config)

        self.bert_Encoder = Bert_Encoder(bi_input=config.extract_dim, bi_hidden=200, bi_layers=1,
                                         bi_directional=True, dropout=0.5)


    @staticmethod
    def prepare_pack_padded_sequence(inputs_words, seq_lengths, descending=True):
        """
        :param inputs_words:
        :param seq_lengths:
        :param descending:
        :return:
        """
        sorted_seq_lengths, indices = torch.sort(torch.Tensor(seq_lengths).long(), descending=descending)
        # sorted_seq_lengths, indices = torch.sort(torch.LongTensor(seq_lengths), descending=descending)
        _, desorted_indices = torch.sort(indices, descending=False)
        # batch first is True
        sorted_inputs_words = inputs_words[indices]
        # batch first False
        # sorted_inputs_words = inputs_words[:, indices]
        return sorted_inputs_words, sorted_seq_lengths.numpy(), desorted_indices

    def forward(self, batch_features, bert_hidden, train=False):
        """
        :param batch_features:
        :param bert_hidden:
        :param train:
        :return:
        """
        word = batch_features.word_features
        sentence_length = batch_features.sentence_length
        x = self.embed(word)  # (N,W,D)
        x = self.dropout_embed(x)
        # print(x.size())
        # print(sentence_length)
        packed_embed = pack_padded_sequence(x, sentence_length, batch_first=True)
        x, _ = self.bilstm(packed_embed)
        x, _ = pad_packed_sequence(x, batch_first=True)
        x = x[batch_features.desorted_indices]
        # x = x.permute(0, 2, 1)
        # x = F.max_pool1d(x, x.size(2)).squeeze(2)
        x = self.hierachicalAtt(x)

        # bert encoder
        bert_hidden = bert_hidden.to(self.config.device)
        bert_fea = self.bert_Encoder(bert_hidden)
        bert_x = torch.cat((x, bert_fea), 1)

        return bert_x
