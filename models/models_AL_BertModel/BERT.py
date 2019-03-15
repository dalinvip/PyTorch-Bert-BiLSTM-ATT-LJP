# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# @Author : bamtercelboo
# @Datetime : 2019/3/14 15:26
# @File : BERT.py
# @Last Modify Time : 2019/3/14 15:26
# @Contact : bamtercelboo@{gmail.com, 163.com}


"""Extract pre-computed feature vectors from a PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import collections
import logging
import json
import re
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

# logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
#                     datefmt = '%m/%d/%Y %H:%M:%S',
#                     level = logging.INFO)
# logger = logging.getLogger(__name__)


class InputExample(object):
    """
        InputExample
    """
    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


class BERT(object):
    """
        BERT
    """
    def __init__(self, **kwargs):
        self.bert_model = kwargs["bert_model"]
        self.vocab = kwargs["vocab"]
        self.max_seq_length = kwargs["max_seq_length"]
        self.batch_size = kwargs["batch_size"]
        self.extract_dim = kwargs["extract_dim"]
        self.layers = kwargs["layers"]
        self.local_rank = kwargs["local_rank"]
        self.no_cuda = kwargs["no_cuda"]
        self.do_lower_case = kwargs["do_lower_case"]

        # example index
        self.ex_index = -1

        # Now default -1
        # self.layer_indexes = [int(x) for x in self.layers.split(",")]
        self.layer_indexes = -1

        # cpu cuda device
        self.device, self.n_gpu = self._set_device()

        # tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
        self.tokenizer = BertTokenizer.from_pretrained(os.path.join(self.bert_model, self.vocab))

        # Bert Model
        self.model = self._load_bert_model()

    def extract_feature(self, batch_features):
        """
        :param batch_features:
        :return:
        """
        # print("extract bert feature")
        examples, uniqueid_to_line = self._read_examples(batch_features, self.max_seq_length)
        features = self._convert_examples_to_features(
            examples=examples, seq_length=self.max_seq_length, tokenizer=self.tokenizer)

        unique_id_to_feature = {}
        for feature in features:
            unique_id_to_feature[feature.unique_id] = feature

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_example_index)
        if self.local_rank == -1:
            eval_sampler = SequentialSampler(eval_data)
        else:
            eval_sampler = DistributedSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.batch_size)
        bert_feature = self._get_bert_hidden(eval_dataloader, features, uniqueid_to_line)
        return bert_feature

    def _get_bert_hidden(self, eval_dataloader, features, uniqueid_to_line):
        """
        :param eval_dataloader:
        :param features:
        :param uniqueid_to_line:
        :return:
        """
        print(uniqueid_to_line)

        self.model.eval()
        batch_count = len(eval_dataloader)
        batch_num = 0
        line_index_exist = []
        result = []
        for input_ids, input_mask, example_indices in eval_dataloader:
            batch_num += 1
            # sys.stdout.write("\rBert Model For the {} Batch, All {} batch.".format(batch_num, batch_count))
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            all_encoder_layers, _ = self.model(input_ids, token_type_ids=None, attention_mask=input_mask)
            all_encoder_layers = all_encoder_layers
            layer_index = self.layer_indexes
            layer_output_all = all_encoder_layers[layer_index].detach().cpu().numpy()[:, :, :self.extract_dim]

            for b, example_index in enumerate(example_indices):
                feature = features[example_index.item()]
                tokens = feature.tokens
                token_length = len(tokens)
                layer_output = np.round(layer_output_all[b][:token_length].tolist(), 6).tolist()

                out_features = collections.OrderedDict()
                out_features["tokens"] = tokens
                out_features["values"] = layer_output

                unique_id = int(feature.unique_id)
                line_index = uniqueid_to_line[str(unique_id)]
                if line_index in line_index_exist:
                    output_json["features"]["tokens"].extend(tokens)
                    output_json["features"]["values"].extend(layer_output)
                    continue
                else:
                    if len(line_index_exist) != 0:
                        result.append(output_json)
                    line_index_exist.clear()
                    line_index_exist.append(line_index)
                    output_json = collections.OrderedDict()
                    output_json["linex_index"] = line_index
                    output_json["layer_index"] = layer_index
                    output_json["features"] = out_features
                    # continue
        result.append(output_json)
        bert_feature = self._batch(result)
        return bert_feature

    def _batch(self, result):
        """
        :param result:
        :return:
        """
        batch_size = len(result)
        max_char_size = -1
        extract_dim = len(result[0]["features"]["values"][0])
        for line in result:
            char_size = len(line["features"]["tokens"])
            if char_size > max_char_size:
                max_char_size = char_size
        bert_feature = np.zeros((batch_size, max_char_size, extract_dim))

        for b in range(batch_size):
            values = result[b]["features"]["values"]
            length = len(values)
            bert_feature[b][:length] = np.array(result[b]["features"]["values"])

        bert_feature = torch.from_numpy(bert_feature).float()
        bert_feature.to(self.device)
        return bert_feature

    def _load_bert_model(self):
        model = BertModel.from_pretrained(self.bert_model)
        model.to(self.device)
        if self.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                              device_ids=[self.local_rank],
                                                              output_device=self.local_rank)
        elif self.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        return model

    def _set_device(self):
        if self.local_rank == -1 or self.no_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() and not self.no_cuda else "cpu")
            n_gpu = torch.cuda.device_count()
        else:
            device = torch.device("cuda", self.local_rank)
            n_gpu = 1
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend='nccl')
        print("device: {} n_gpu: {} distributed training: {}".format(device, n_gpu, bool(self.local_rank != -1)))
        return device, n_gpu

    def _convert_examples_to_features(self, examples, seq_length, tokenizer):
        """Loads a data file into a list of `InputBatch`s."""

        features = []
        for (ex_index, example) in enumerate(examples):
            # print(example.text_a)
            tokens_a = tokenizer.tokenize(example.text_a)

            tokens_b = None
            if example.text_b:
                tokens_b = tokenizer.tokenize(example.text_b)

            if tokens_b:
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                self._truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > seq_length - 2:
                    tokens_a = tokens_a[0:(seq_length - 2)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids:   0   0  0    0    0     0      0   0    1  1  1   1  1   1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids:   0   0   0   0  0     0   0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambigiously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens = []
            input_type_ids = []
            tokens.append("[CLS]")
            input_type_ids.append(0)
            for token in tokens_a:
                tokens.append(token)
                input_type_ids.append(0)
            tokens.append("[SEP]")
            input_type_ids.append(0)

            if tokens_b:
                for token in tokens_b:
                    tokens.append(token)
                    input_type_ids.append(1)
                tokens.append("[SEP]")
                input_type_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < seq_length:
                input_ids.append(0)
                input_mask.append(0)
                input_type_ids.append(0)

            assert len(input_ids) == seq_length
            assert len(input_mask) == seq_length
            assert len(input_type_ids) == seq_length

            if ex_index < self.ex_index:
                print("*** Example ***")
                print("unique_id: %s" % (example.unique_id))
                print("tokens: %s" % " ".join([str(x) for x in tokens]))
                print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                print("input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

            features.append(
                InputFeatures(
                    unique_id=example.unique_id,
                    tokens=tokens,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    input_type_ids=input_type_ids))
        return features

    @staticmethod
    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    @staticmethod
    def _cut_text_by_len(text, length):
        """
        :param text:
        :param length:
        :return:
        """
        textArr = re.findall('.{' + str(length) + '}', text)
        textArr.append(text[(len(textArr) * length):])
        return textArr

    def _read_examples(self, batch_features, max_seq_length):
        """Read a list of `InputExample`s from an input file."""
        examples = []
        unique_id = 0
        line_index = 0
        uniqueid_to_line = collections.OrderedDict()
        for inst in batch_features.inst:
            line = inst.bert_line
            line = line.strip()
            # print(line)
            # line = "".join(json.loads(line)["fact"].split())
            line_cut = self._cut_text_by_len(line, max_seq_length)
            for l in line_cut:
                uniqueid_to_line[str(unique_id)] = line_index
                text_a = None
                text_b = None
                m = re.match(r"^(.*) \|\|\| (.*)$", l)
                if m is None:
                    text_a = l
                else:
                    text_a = m.group(1)
                    text_b = m.group(2)
                examples.append(
                    InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
                unique_id += 1
            line_index += 1
        # print(uniqueid_to_line)
        return examples, uniqueid_to_line


