# @Author : bamtercelboo
# @Datetime : 2018/1/30 15:58
# @File : DataLoader.py
# @Last Modify Time : 2018/1/30 15:58
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :
    FUNCTION :
"""
import os
import sys
import re
import time
import random
import json
import torch
import numpy as np
from collections import OrderedDict
from Dataloader.Instance import Instance

from DataUtils.Common import *
torch.manual_seed(seed_num)
random.seed(seed_num)
np.random.seed(seed_num)


class DataLoaderHelp(object):
    """
    DataLoaderHelp
    """

    @staticmethod
    def _clean_str(string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    @staticmethod
    def _clean_punctuation(string):
        """
        :param string:
        :return:
        """
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"，", "", string)
        string = re.sub(r"。", "", string)
        string = re.sub(r"“", "", string)
        string = re.sub(r"”", "", string)
        string = re.sub(r"、", "", string)
        string = re.sub(r"：", "", string)
        string = re.sub(r"；", "", string)
        string = re.sub(r"（", "", string)
        string = re.sub(r"）", "", string)
        string = re.sub(r"《 ", "", string)
        string = re.sub(r"》", "", string)
        # string = re.sub(r"× ×", "", string)
        # string = re.sub(r"x")
        string = re.sub(r"  ", " ", string)
        return string.lower()

    @staticmethod
    def _normalize_word(word):
        """
        :param word:
        :return:
        """
        new_word = ""
        for char in word:
            if char.isdigit():
                new_word += '0'
            else:
                new_word += char
        return new_word

    @staticmethod
    def _sort(insts):
        """
        :param insts:
        :return:
        """
        sorted_insts = []
        sorted_dict = {}
        for id_inst, inst in enumerate(insts):
            sorted_dict[id_inst] = inst.words_size
        dict = sorted(sorted_dict.items(), key=lambda d: d[1], reverse=True)
        for key, value in dict:
            sorted_insts.append(insts[key])
        print("Sort Finished.")
        return sorted_insts

    @staticmethod
    def _sortby_sencount(insts):
        sorted_insts = []
        sorted_dict = {}
        for id_inst, inst in enumerate(insts):
            sorted_dict[id_inst] = inst.sentence_size
        dict = sorted(sorted_dict.items(), key=lambda d: d[1], reverse=False)
        for key, value in dict:
            if len(sorted_insts) > 0:
                sorted_insts[-1].next_sentence_size = insts[key].sentence_size
            sorted_insts.append(insts[key])
        sorted_insts[-1].next_sentence_size = -1
        print("Sort by Doc Sentence Count Finished.")
        return sorted_insts

    @staticmethod
    def _sort2json_file(insts, path):
        path = "_".join([path, "sort.json"])
        print("Sort Result To File {}.".format(path))
        if os.path.exists(path):
            os.remove(path)
        file = open(path, encoding="UTF-8", mode="w")
        for inst in insts:
            dictObj = {'meta': {'accusation': inst.accu_labels, 'predict_accu': []}, 'fact': " ".join(inst.words)}
            jsObj = json.dumps(dictObj, ensure_ascii=False)
            file.write(jsObj + "\n")
        file.close()
        print("Sort Result To File Finished.")


class DataLoader(DataLoaderHelp):
    """
    DataLoader
    """
    def __init__(self, path, shuffle, config):
        """
        :param path: data path list
        :param shuffle:  shuffle bool
        :param config:  config
        """
        #
        print("Loading Data......")
        self.data_list = []
        self.max_count = config.max_count
        self.path = path
        self.shuffle = shuffle
        self.max_train_len = config.max_train_len

        # BERT
        self.bert_path = [config.bert_train_file,
                          config.bert_dev_file,
                          config.bert_test_file]

        self.use_bert = config.use_bert
        self.bert_max_char_length = config.bert_max_char_length

    def dataLoader(self):
        """
        :return:
        """
        start_time = time.time()
        path = self.path
        shuffle = self.shuffle
        assert isinstance(path, list), "Path Must Be In List"
        print("Data Path {}".format(path))
        for id_data in range(len(path)):
            print("Loading Data Form {}".format(path[id_data]))
            insts = self._Load_Each_JsonData(path=path[id_data], path_id=id_data)
            # if shuffle is True and id_data == 0:
            print("shuffle data......")
            random.shuffle(insts)
            # if self.split_doc is True:
            #     sentence_insts = self._from_chapter2default_cut(insts, cutoff=self.word_cut_count)
            insts = self._sort(insts=insts)
            self._sort2json_file(insts, path=path[id_data])
            self.data_list.append(insts)
        end_time = time.time()
        print("DataLoader Time {:.4f}".format(end_time - start_time))
        # return train/dev/test data
        if len(self.data_list) == 3:
            return self.data_list[0], self.data_list[1], self.data_list[2]
        elif len(self.data_list) == 2:
            return self.data_list[0], self.data_list[1]

    def _Load_Each_JsonData(self, path=None, path_id=0, train=False):
        assert path is not None, "The Data Path Is Not Allow Empty."
        insts = []
        now_lines = 0
        # print()
        with open(path, encoding="UTF-8") as f:
            lines = f.readlines()
            for line in lines:
                now_lines += 1
                if now_lines % 2000 == 0:
                    sys.stdout.write("\rreading the {} line\t".format(now_lines))
                if line == "\n":
                    print("empty line")
                inst = Instance()
                line_json = json.loads(line)
                fact = line_json["fact"].split()[:self.max_train_len]
                bert_line = "".join(fact)

                # accu label
                accu = line_json["meta"]["accusation"]
                # print(accu)
                # law label
                law = line_json["meta"]["relevant_articles"]

                inst.words = fact
                inst.bert_line = bert_line[:self.bert_max_char_length]
                inst.accu_labels = accu
                inst.law_labels = law

                inst.words_size = len(inst.words)
                inst.accu_labels_size = len(inst.accu_labels)
                inst.law_labels_size = len(inst.law_labels)
                insts.append(inst)
                if len(insts) == self.max_count:
                    break
            sys.stdout.write("\rreading the {} line\t".format(now_lines))
        if self.use_bert:
            insts = self._read_bert_file(insts, path=self.bert_path[path_id])
        return insts

    def _read_bert_file(self, insts, path):
        """
        :param insts:
        :param path:
        :return:
        """
        print("\nRead BERT File From {}".format(path))
        now_lines = 0
        with open(path, encoding="utf-8") as f:
            for inst, bert_line in zip(insts, f.readlines()):
                now_lines += 1
                if now_lines % 2000 == 0:
                    sys.stdout.write("\rreading the {} line\t".format(now_lines))
                bert_fea = json.loads(bert_line)["values"]
                inst.bert_feature = np.array(bert_fea)
                # print(inst.bert_feature)
            sys.stdout.write("\rReading the {} line\t".format(now_lines))
        return insts

