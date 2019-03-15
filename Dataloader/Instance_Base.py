# coding=utf-8
# @Author : bamtercelboo
# @Datetime : 2018/07/10 16.03
# @File : train.py
# @Last Modify Time : 2018/07/10 16.03
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  Instance.py
    FUNCTION : Data Instance
"""

import torch
import random

from DataUtils.Common import *
torch.manual_seed(seed_num)
random.seed(seed_num)


class Instance_Base(object):
    def __init__(self):
        self.words = []
        self.labels = []

        self.words_size = 0
        self.labels_size = 0

        self.words_index = []
        self.label_index = []





