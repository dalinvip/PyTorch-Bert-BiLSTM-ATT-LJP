# @Author : bamtercelboo
# @Datetime : 2018/8/16 8:50
# @File : Instance_extend.py
# @Last Modify Time : 2018/8/16 8:50
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  Instance_extend.py
    FUNCTION : None
"""
from .Instance_Base import Instance_Base
import torch
import random
import numpy as np

from DataUtils.Common import *
torch.manual_seed(seed_num)
random.seed(seed_num)


class Instance(Instance_Base):
    def __init__(self):
        # super(Instance).__init__()
        # print("Instance extend")
        super().__init__()
        self.fact = []
        self.bert_line = ""

        self.accu_labels = []
        self.law_labels = []

        self.accu_labels_size = 0
        self.law_labels_size = 0
        self.sentence_size = 0
        self.next_sentence_size = 0
        self.sentence_word_size = []

        self.accu_label_index = []
        self.law_label_index = []

        self.bert_feature = None


