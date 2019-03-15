# @Author : bamtercelboo
# @Datetime : 2018/07/10 16.03
# @File : train.py
# @Last Modify Time : 2018/07/10 16.03
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  cail_eval.py
    FUNCTION : None
"""

import torch
import time
from torch.autograd import Variable

from DataUtils.Common import *
from DataUtils.utils import *


class Eval:
    def __init__(self):
        self.predict_num = 0
        self.correct_num = 0
        self.gold_num = 0

        self.precision = 0
        self.recall = 0
        self.fscore = 0

    def clear_PRF(self):
        self.predict_num = 0
        self.correct_num = 0
        self.gold_num = 0

        self.precision = 0
        self.recall = 0
        self.fscore = 0

    def getFscore(self):
        if self.predict_num == 0:
            self.precision = 0
        else:
            self.precision = (self.correct_num / self.predict_num) * 100

        if self.gold_num == 0:
            self.recall = 0
        else:
            self.recall = (self.correct_num / self.gold_num) * 100

        if self.precision + self.recall == 0:
            self.fscore = 0
        else:
            self.fscore = (2 * (self.precision * self.recall)) / (self.precision + self.recall)

        self.precision = np.round(self.precision, 4)
        self.recall = np.round(self.recall, 4)
        self.fscore = np.round(self.fscore, 4)

        return self.precision, self.recall, self.fscore

    def acc(self):
        if self.predict_num == 0:
            return 0.0000
        else:
            return np.round((self.correct_num / self.predict_num) * 100, 4)


def Micro_F1_measure(logit, gold, predict_batch_list, f_micro, cuda):

    correct_num = 0
    # assert len(predict_batch_list) == len(gold)
    # predict_batch_list = Variable(torch.Tensor(predict_batch_list).long())
    predict_batch_list = torch.Tensor(predict_batch_list).long()
    if cuda is True:
        gold = gold.cuda()
        predict_batch_list = predict_batch_list.cuda()
    # print(gold)
    # print(predict_batch_list)

    correct_num += int(torch.sum(predict_batch_list & gold))

    predict_num = int(torch.sum(predict_batch_list))
    gold_num = int(torch.sum(gold))

    f_micro.correct_num += correct_num
    f_micro.predict_num += predict_num
    f_micro.gold_num += gold_num


def Macro_F1_measure(logit, gold, predict_batch_list, f_macro, cuda):

    batch_size = logit.size(0)
    class_num = logit.size(1)

    '''
        add by zenRRan
    '''
    # predict_batch_list = Variable(torch.LongTensor(predict_batch_list)).view(batch_size, class_num)
    predict_batch_list = torch.Tensor(predict_batch_list).long().view(batch_size, class_num)
    gold_list = gold.view(batch_size, class_num)
    if cuda is True:
        predict_batch_list = predict_batch_list.cuda()
        gold_list = gold_list.cuda()
    correct_num = torch.sum((predict_batch_list.data & gold_list.data), 0)
    predict_num = torch.sum(predict_batch_list, 0)
    gold_num = torch.sum(gold_list.data, 0)

    correct_num = correct_num.tolist()
    predict_num = predict_num.data.tolist()
    gold_num = gold_num.tolist()

    for i in range(class_num):
        f_macro[i].correct_num += correct_num[i]
        f_macro[i].predict_num += predict_num[i]
        f_macro[i].gold_num += gold_num[i]


def F1_measure(logit, gold, f_micro, f_macro, cuda=False):
    """
    :param logit:
    :param gold:
    :param f_micro:
    :param f_macro:
    :param cuda:
    :return:
    """
    predict_batch_list = torch_max(logit)
    Micro_F1_measure(logit=logit, gold=gold, predict_batch_list=predict_batch_list, f_micro=f_micro, cuda=cuda)
    Macro_F1_measure(logit=logit, gold=gold, predict_batch_list=predict_batch_list, f_macro=f_macro, cuda=cuda)


def getFscore_Avg(f_micro, f_macro, class_num):
    p_micro, r_micro, f1_micro = f_micro.getFscore()
    # p_micro, r_micro, f1_micro = 0, 0, 0

    p_macro, r_macro, f1_macro = 0, 0, 0
    for i in range(class_num):
        p_macro += f_macro[i].getFscore()[0]
        r_macro += f_macro[i].getFscore()[1]
        f1_macro += f_macro[i].getFscore()[2]

    p_macro_avg, r_macro_avg, f1_macro_avg = p_macro / class_num, r_macro / class_num, f1_macro / class_num

    # print("macro", p_macro_avg, r_macro_avg, f1_macro_avg)

    p_avg, r_avg, f_avg = (p_micro + p_macro_avg) / 2, (r_micro + r_macro_avg) / 2, (f1_micro + f1_macro_avg) / 2
    # p_avg, r_avg, f_avg = p_macro_avg, r_macro_avg, f1_macro_avg

    p_avg = np.round(p_avg, 4)
    r_avg = np.round(r_avg, 4)
    f_avg = np.round(f_avg, 4)

    return (p_avg, r_avg, f_avg), (p_micro, r_micro, f1_micro), (p_macro_avg, r_macro_avg, f1_macro_avg)


def test():
    np.random.seed(233)
    torch.manual_seed(233)
    """
            batch_size = 3
            class_num = 8
            label_size = 2
        """
    logit = Variable(torch.randn(3, 8, 2).type(torch.FloatTensor))
    gold = Variable(torch.from_numpy(np.random.randint(low=0, high=2, size=3 * 8)).type(torch.LongTensor))

    f_micro = Eval()
    f_micro.clear_PRF()

    f_macro = []
    for i in range(8):
        t = Eval()
        f_macro.append(t)
    # print(f_macro)
    for i in range(8):
        f_macro[i].clear_PRF()

    F1_measure(logit, gold, f_micro, f_macro)

    p_avg, r_avg, f_avg = getFscore_Avg(f_micro, f_macro, logit.size(1))
    print(p_avg, r_avg, f_avg)


if __name__ == "__main__":
    print("CAIL2018 F-Score(micro, macro)")
    test()




