# @Author : bamtercelboo
# @Datetime : 2018/8/26 8:30
# @File : trainer.py
# @Last Modify Time : 2018/8/26 8:30
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  trainer.py
    FUNCTION : None
"""

import os
import sys
import time
import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.utils as utils
from DataUtils.Optim import Optimizer
from DataUtils.utils import *
from DataUtils.cail_eval import Eval, F1_measure, getFscore_Avg
from DataUtils.Common import *
torch.manual_seed(seed_num)
random.seed(seed_num)


class Train(object):
    """
        Train
    """
    def __init__(self, **kwargs):
        """
        :param kwargs:
        Args of data:
            train_iter : train batch data iterator
            dev_iter : dev batch data iterator
            test_iter : test batch data iterator
        Args of train:
            model : nn model
            config : config
        """
        print("Training Start......")
        # for k, v in kwargs.items():
        #     self.__setattr__(k, v)
        self.train_iter = kwargs["train_iter"]
        self.dev_iter = kwargs["dev_iter"]
        self.test_iter = kwargs["test_iter"]
        self.model = kwargs["model"]
        self.config = kwargs["config"]
        self.device = self.config.device
        self.cuda = False
        if self.device != cpu_device:
            self.cuda = True
        self.early_max_patience = self.config.early_max_patience
        self.optimizer = Optimizer(name=self.config.learning_algorithm, model=self.model, lr=self.config.learning_rate,
                                   weight_decay=self.config.weight_decay, grad_clip=self.config.clip_max_norm)
        if self.config.learning_algorithm == "SGD":
            self.loss_function = nn.CrossEntropyLoss(reduction="sum")
        else:
            self.loss_function = nn.CrossEntropyLoss(reduction="mean")
            # self.loss_function = nn.MultiLabelSoftMarginLoss(size_average=True)
        print(self.optimizer)
        self.best_score = Best_Result()
        self.train_iter_len = len(self.train_iter)

        # define accu eval
        self.accu_train_eval_micro, self.accu_dev_eval_micro, self.accu_test_eval_micro = Eval(), Eval(), Eval()
        self.accu_train_eval_macro, self.accu_dev_eval_macro, self.accu_test_eval_macro = [], [], []
        for i in range(self.config.accu_class_num):
            self.accu_train_eval_macro.append(Eval())
            self.accu_dev_eval_macro.append(Eval())
            self.accu_test_eval_macro.append(Eval())

        # define law eval
        self.law_train_eval_micro, self.law_dev_eval_micro, self.law_test_eval_micro = Eval(), Eval(), Eval()
        self.law_train_eval_macro, self.law_dev_eval_macro, self.law_test_eval_macro = [], [], []
        for i in range(self.config.law_class_num):
            self.law_train_eval_macro.append(Eval())
            self.law_dev_eval_macro.append(Eval())
            self.law_test_eval_macro.append(Eval())

    def _clip_model_norm(self, clip_max_norm_use, clip_max_norm):
        """
        :param clip_max_norm_use:  whether to use clip max norm for nn model
        :param clip_max_norm: clip max norm max values [float or None]
        :return:
        """
        if clip_max_norm_use is True:
            gclip = None if clip_max_norm == "None" else float(clip_max_norm)
            assert isinstance(gclip, float)
            utils.clip_grad_norm_(self.model.parameters(), max_norm=gclip)

    def _dynamic_lr(self, config, epoch, new_lr):
        """
        :param config:  config
        :param epoch:  epoch
        :param new_lr:  learning rate
        :return:
        """
        if config.use_lr_decay is True and epoch > config.max_patience and (
                epoch - 1) % config.max_patience == 0 and new_lr > config.min_lrate:
            # print("epoch", epoch)
            new_lr = max(new_lr * config.lr_rate_decay, config.min_lrate)
            set_lrate(self.optimizer, new_lr)
        return new_lr

    def _decay_learning_rate(self, epoch, init_lr):
        """
        Args:
            epoch: int, epoch
            init_lr: initial lr
        """
        lr = init_lr / (1 + self.config.lr_rate_decay * epoch)
        # print('learning rate: {0}'.format(lr))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return self.optimizer

    def _optimizer_batch_step(self, config, backward_count):
        """
        :return:
        """
        if backward_count % config.backward_batch_size == 0 or backward_count == self.train_iter_len:
            self.optimizer.step()
            self.optimizer.zero_grad()

    def _early_stop(self, epoch):
        """
        :param epoch:
        :return:
        """
        best_epoch = self.best_score.best_epoch
        if epoch > best_epoch:
            self.best_score.early_current_patience += 1
            print("Dev Has Not Promote {} / {}".format(self.best_score.early_current_patience, self.early_max_patience))
            if self.best_score.early_current_patience >= self.early_max_patience:
                print("Early Stop Train. Best Score Locate on {} Epoch.".format(self.best_score.best_epoch))
                exit()

    def train(self):
        """
        :return:
        """
        epochs = self.config.epochs
        clip_max_norm_use = self.config.clip_max_norm_use
        clip_max_norm = self.config.clip_max_norm
        new_lr = self.config.learning_rate

        for epoch in range(1, epochs + 1):
            print("\n## The {} Epoch, All {} Epochs ! ##".format(epoch, epochs))
            new_lr = self._dynamic_lr(config=self.config, epoch=epoch, new_lr=new_lr)
            # self.optimizer = self._decay_learning_rate(epoch=epoch - 1, init_lr=self.config.learning_rate)
            print("now lr is {}".format(self.optimizer.param_groups[0].get("lr")), end="")
            start_time = time.time()
            random.shuffle(self.train_iter)
            self.model.train()
            steps = 1
            backward_count = 0
            self.optimizer.zero_grad()
            for batch_count, batch_features in enumerate(self.train_iter):
                backward_count += 1
                # self.optimizer.zero_grad()
                accu, law, e_time, d_time = self.model(batch_features)
                accu_logit = accu.view(accu.size(0) * accu.size(1), accu.size(2))
                law_logit = law.view(law.size(0) * law.size(1), law.size(2))
                # print(accu_logit.size())
                # accu_logit = torch_max_one(accu_logit)
                # law_logit = torch_max_one(law_logit)
                # print(batch_features.accu_label_features.size())
                loss_accu = self.loss_function(accu_logit, batch_features.accu_label_features)
                loss_law = self.loss_function(law_logit, batch_features.law_label_features)
                # total_loss = (loss_accu + loss_law)
                total_loss = (loss_accu + loss_law) / 2
                # loss.backward()
                total_loss.backward()
                self._clip_model_norm(clip_max_norm_use, clip_max_norm)
                self._optimizer_batch_step(config=self.config, backward_count=backward_count)
                # self.optimizer.step()
                steps += 1
                if (steps - 1) % self.config.log_interval == 0:
                    self.accu_train_eval_micro.clear_PRF()
                    for i in range(self.config.accu_class_num):  self.accu_train_eval_macro[i].clear_PRF()
                    F1_measure(accu, batch_features.accu_label_features, self.accu_train_eval_micro,
                               self.accu_train_eval_macro, cuda=self.cuda)
                    (accu_p_avg, accu_r_avg, accu_f_avg), (p_micro, r_micro, f1_micro), (
                    p_macro_avg, r_macro_avg, f1_macro_avg) = getFscore_Avg(self.accu_train_eval_micro,
                                                                            self.accu_train_eval_macro, accu.size(1))
                    sys.stdout.write(
                        "\nbatch_count = [{}/{}] , total_loss is {:.6f}, [accu-Micro-F1 is {:.6f}%]".format(
                            batch_count + 1, self.train_iter_len, total_loss.item(), f1_micro))
                end_time = time.time()
            print("\nTrain Time {:.3f}".format(end_time - start_time), end="")
            self.eval(model=self.model, epoch=epoch, config=self.config)
            self._model2file(model=self.model, config=self.config, epoch=epoch)
            self._early_stop(epoch=epoch)

    def eval(self, model, epoch, config):
        """
        :param model: nn model
        :param epoch:  epoch
        :param config:  config
        :return:
        """
        self.accu_dev_eval_micro.clear_PRF()
        for i in range(self.config.accu_class_num): self.accu_dev_eval_macro[i].clear_PRF()
        self.law_dev_eval_micro.clear_PRF()
        for i in range(self.config.law_class_num): self.law_dev_eval_macro[i].clear_PRF()
        eval_start_time = time.time()
        self._eval_batch(self.dev_iter, model, self.accu_dev_eval_micro, self.accu_dev_eval_macro,
                         self.law_dev_eval_micro, self.law_dev_eval_macro, self.best_score, epoch, config, test=False)
        eval_end_time = time.time()
        print("Dev Time {:.3f}".format(eval_end_time - eval_start_time))

        self.accu_test_eval_micro.clear_PRF()
        for i in range(self.config.accu_class_num): self.accu_test_eval_macro[i].clear_PRF()
        self.law_test_eval_micro.clear_PRF()
        for i in range(self.config.law_class_num): self.law_test_eval_macro[i].clear_PRF()
        eval_start_time = time.time()
        self._eval_batch(self.test_iter, model, self.accu_test_eval_micro, self.accu_test_eval_macro,
                         self.law_test_eval_micro, self.law_test_eval_macro, self.best_score, epoch, config, test=True)
        eval_end_time = time.time()
        print("Test Time {:.3f}".format(eval_end_time - eval_start_time))

    def _model2file(self, model, config, epoch):
        """
        :param model:  nn model
        :param config:  config
        :param epoch:  epoch
        :return:
        """
        if config.save_model and config.save_all_model:
            save_model_all(model, config.save_dir, config.model_name, epoch)
        elif config.save_model and config.save_best_model:
            save_best_model(model, config.save_best_model_path, config.model_name, self.best_score)
        else:
            print()

    def _eval_batch(self, data_iter, model, accu_eval_micro, accu_eval_macro, law_eval_micro, law_eval_macro,
                    best_score, epoch, config, test=False):
        """
        :param data_iter:
        :param model:
        :param accu_eval_micro:
        :param accu_eval_macro:
        :param best_score:
        :param epoch:
        :param config:
        :param test:
        :return:
        """
        model.eval()

        for batch_count, batch_features in enumerate(data_iter):
            accu, law, e_time, d_time = model(batch_features)
            F1_measure(accu, batch_features.accu_label_features, accu_eval_micro, accu_eval_macro, cuda=self.cuda)
            F1_measure(law, batch_features.law_label_features, law_eval_micro, law_eval_macro, cuda=self.cuda)

        # get f-score
        accu_macro_micro_avg, accu_micro, accu_macro = getFscore_Avg(accu_eval_micro, accu_eval_macro, accu.size(1))
        law_macro_micro_avg, law_micro, law_macro = getFscore_Avg(law_eval_micro, law_eval_macro, law.size(1))

        accu_p, accu_r, accu_f = accu_macro_micro_avg
        accu_p_ma, accu_r_ma, accu_f_ma = accu_macro
        accu_p_mi, accu_r_mi, accu_f_mi = accu_micro
        law_p, law_r, law_f = law_macro_micro_avg
        law_p_ma, law_r_ma, law_f_ma = law_macro
        law_p_mi, law_r_mi, law_f_mi = law_micro

        p, r, f = accu_p, accu_r, accu_f
        # p, r, f = law_p, law_r, law_f

        test_flag = "Test"
        if test is False:
            print()
            test_flag = "Dev"
            best_score.current_dev_score = f
            if f >= best_score.best_dev_score:
                best_score.best_dev_score = f
                best_score.best_epoch = epoch
                best_score.best_test = True
        if test is True and best_score.best_test is True:
            best_score.p = p
            best_score.r = r
            best_score.f = f
        print("{}:".format(test_flag))
        print("Macro_Micro_Avg ===>>> ")
        print("Eval: accu    --- Precision = {:.6f}%  Recall = {:.6f}% , F-Score = {:.6f}%".format(accu_p, accu_r, accu_f))
        print("Eval:  law    --- Precision = {:.6f}%  Recall = {:.6f}% , F-Score = {:.6f}%".format(law_p, law_r, law_f))
        print("Macro ===>>> ")
        print("Eval: accu    --- Precision = {:.6f}%  Recall = {:.6f}% , F-Score = {:.6f}%".format(accu_p_ma, accu_r_ma, accu_f_ma))
        print("Eval:  law    --- Precision = {:.6f}%  Recall = {:.6f}% , F-Score = {:.6f}%".format(law_p_ma, law_r_ma, law_f_ma))
        print("Micro ===>>> ")
        print("Eval: accu    --- Precision = {:.6f}%  Recall = {:.6f}% , F-Score = {:.6f}%".format(accu_p_mi, accu_r_mi, accu_f_mi))
        print("Eval:  law    --- Precision = {:.6f}%  Recall = {:.6f}% , F-Score = {:.6f}%".format(law_p_mi, law_r_mi, law_f_mi))

        if test is True:
            print("The Current Best accu Dev F-score: {:.6f}, Locate on {} Epoch.".format(best_score.best_dev_score, best_score.best_epoch))
            # print("The Current Best Law Dev F-score: {:.6f}, Locate on {} Epoch.".format(best_score.best_dev_score, best_score.best_epoch))
        if test is True:
            best_score.best_test = False





