# @Author : bamtercelboo
# @Datetime : 2018/1/30 15:55
# @File : Batch_Iterator.py.py
# @Last Modify Time : 2018/1/30 15:55
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  Batch_Iterator.py
    FUNCTION : None
"""

import torch
from torch.autograd import Variable
import random
import time
import numpy as np

from DataUtils.Common import *
torch.manual_seed(seed_num)
random.seed(seed_num)


class Batch_Features:
    """
    Batch_Features
    """
    def __init__(self):
        self.batch_length = 0
        self.sentence_count = None
        self.inst = None
        self.word_features = 0
        self.accu_label_features = 0
        self.law_label_features = 0
        self.sentence_length = []
        self.sentence_word_size = []
        self.desorted_indices = None
        self.context_indices = None
        self.bert_features = None

    @staticmethod
    def cuda(features, device):
        """
        :param features:
        :return:
        """
        # features.word_features = features.word_features.cuda()
        # features.accu_label_features = features.accu_label_features.cuda()
        # features.law_label_features = features.law_label_features.cuda()
        # features.desorted_indices = features.desorted_indices.cuda()

        features.word_features = features.word_features.to(device)
        features.accu_label_features = features.accu_label_features.to(device)
        features.law_label_features = features.law_label_features.to(device)
        features.desorted_indices = features.desorted_indices.to(device)
        features.bert_features = features.bert_features.to(device)


class Iterators:
    """
    Iterators
    """
    def __init__(self, batch_size=None, data=None, alphabet=None, config=None):
        self.config = config
        self.device = config.device
        self.batch_size = batch_size
        self.data = data
        self.alphabet = alphabet
        self.alphabet_static = None
        self.iterator = []
        self.batch = []
        self.features = []
        self.data_iter = []

        # Bert
        self.use_bert = config.use_bert
        self.bert_dim = config.bert_dim

    def createIterator(self):
        """
        :param batch_size:  batch size
        :param data:  data
        :param alphabet:
        :param config:
        :return:
        """
        start_time = time.time()
        assert isinstance(self.data, list), "ERROR: data must be in list [train_data,dev_data]"
        assert isinstance(self.batch_size, list), "ERROR: batch_size must be in list [16,1,1]"
        for id_data in range(len(self.data)):
            print("*****************    create {} iterator    **************".format(id_data + 1))
            self._convert_word2id(self.data[id_data], self.alphabet)
            self.features = self._Create_Each_Iterator(insts=self.data[id_data], batch_size=self.batch_size[id_data],
                                                       alphabet=self.alphabet)
            self.data_iter.append(self.features)
            self.features = []
        end_time = time.time()
        print("BatchIterator Time {:.4f}".format(end_time - start_time))
        if len(self.data_iter) == 2:
            return self.data_iter[0], self.data_iter[1]
        if len(self.data_iter) == 3:
            return self.data_iter[0], self.data_iter[1], self.data_iter[2]

    @staticmethod
    def _convert_word2id(insts, alphabet):
        """
        :param insts:
        :param alphabet:
        :return:
        """
        for inst in insts:
            # copy with the word and label
            for index in range(inst.words_size):
                word = inst.words[index]
                wordId = alphabet.word_alphabet.from_string(word)
                if wordId == -1:
                    wordId = alphabet.word_unkId
                inst.words_index.append(wordId)

            # accu
            for index in range(inst.accu_labels_size):
                label = inst.accu_labels[index]
                label_id = alphabet.accu_label_alphabet.from_string(label)
                inst.accu_label_index.append(label_id)

            # law
            for index in range(inst.law_labels_size):
                label = inst.law_labels[index]
                label_id = alphabet.law_label_alphabet.from_string(label)
                inst.law_label_index.append(label_id)

        print("Convert Finished.")

    def _Create_Each_Iterator(self, insts, batch_size, alphabet):
        """
        :param insts:
        :param batch_size:
        :param alphabet:
        :return:
        """
        batch = []
        count_inst = 0
        for index, inst in enumerate(insts):
            batch.append(inst)
            count_inst += 1
            # print(batch)
            # batch_flag = (inst.sentence_size != inst.next_sentence_size)
            if (len(batch) == batch_size) or (count_inst == len(insts)):
                one_batch = self._Create_Each_Batch(insts=batch, alphabet=alphabet)
                self.features.append(one_batch)
                batch = []
        print("The all data has created iterator.")
        return self.features

    def _Create_Each_Batch(self, insts, alphabet):
        """
        :param insts:
        :param batch_size:
        :param alphabet:
        :return:
        """
        batch_length = len(insts)
        # copy with the max length for padding
        max_word_size = -1
        sentence_length = []
        for inst in insts:
            sentence_length.append(inst.words_size)
            word_size = inst.words_size
            if word_size > max_word_size:
                max_word_size = word_size

        # create with the Tensor/Variable
        # word and label features
        batch_word_features = np.zeros((batch_length, max_word_size))
        batch_accu_label_features = np.zeros((batch_length * alphabet.accu_label_alphabet.vocab_size))
        batch_law_label_features = np.zeros((batch_length * alphabet.law_label_alphabet.vocab_size))
        batch_bert_features = np.zeros((batch_length, self.bert_dim))
        # batch_bert_features[0] = 1
        # print(batch_bert_features)
        # exit()

        for id_inst in range(batch_length):
            inst = insts[id_inst]
            # copy with the word features
            for id_word_index in range(max_word_size):
                if id_word_index < inst.words_size:
                    batch_word_features[id_inst][id_word_index] = inst.words_index[id_word_index]
                else:
                    batch_word_features[id_inst][id_word_index] = alphabet.word_paddingId

            # accu
            for id_label_index in range(len(inst.accu_label_index)):
                batch_accu_label_features[id_inst * alphabet.accu_label_alphabet.vocab_size + inst.accu_label_index[id_label_index]] = 1

            # law
            for id_label_index in range(len(inst.law_label_index)):
                batch_law_label_features[id_inst * alphabet.law_label_alphabet.vocab_size + inst.law_label_index[id_label_index]] = 1

            # bert
            if self.use_bert:
                batch_bert_features[id_inst] = inst.bert_feature

        batch_word_features = torch.from_numpy(batch_word_features).long()
        batch_accu_label_features = torch.from_numpy(batch_accu_label_features).long()
        batch_law_label_features = torch.from_numpy(batch_law_label_features).long()
        batch_bert_features = torch.from_numpy(batch_bert_features).float()

        # prepare for pack_padded_sequence
        sorted_inputs_words, sorted_seq_lengths, desorted_indices = self._prepare_pack_padded_sequence(
            batch_word_features, sentence_length)
        # print(sorted_seq_lengths)

        # batch
        features = Batch_Features()
        features.batch_length = batch_length
        # features.sentence_batch_size = sentence_batch_size
        features.inst = insts
        features.word_features = sorted_inputs_words
        features.accu_label_features = batch_accu_label_features
        features.law_label_features = batch_law_label_features
        features.sentence_length = sorted_seq_lengths
        features.desorted_indices = desorted_indices
        features.context_indices = None
        features.bert_features = batch_bert_features

        if self.device != cpu_device:
            features.cuda(features, self.device)
        return features

    @staticmethod
    def _prepare_pack_padded_sequence(inputs_words, seq_lengths, descending=True):
        """
        :param inputs_words:
        :param seq_lengths:
        :param descending:
        :return:
        """
        sorted_seq_lengths, indices = torch.sort(torch.Tensor(seq_lengths).long(), descending=descending)
        # sorted_seq_lengths, indices = torch.sort(torch.LongTensor(seq_lengths), descending=descending)
        _, desorted_indices = torch.sort(indices, descending=False)
        sorted_inputs_words = inputs_words[indices]
        return sorted_inputs_words, sorted_seq_lengths.numpy(), desorted_indices

