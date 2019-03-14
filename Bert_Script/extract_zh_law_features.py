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

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):

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


def convert_examples_to_features(examples, seq_length, tokenizer):
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
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
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

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("unique_id: %s" % (example.unique_id))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features


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


def read_examples(input_file):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    unique_id = 0
    max_len = []
    with open(input_file, "r", encoding='utf-8') as reader:
        while True:
            line = reader.readline()
            if not line:
                break
            line = line.strip()
            # print(line)
            line = "".join(json.loads(line)["fact"].split())
            max_len.append(len(line))
            # line = json.loads(line)["fact"]
            # print(line)
            # exit()
            # print(line["fact"])
            text_a = None
            text_b = None
            m = re.match(r"^(.*) \|\|\| (.*)$", line)
            if m is None:
                text_a = line
            else:
                text_a = m.group(1)
                text_b = m.group(2)
            examples.append(
                InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
            unique_id += 1

    return examples, max(max_len)


def to_json(output_file, model, eval_dataloader, device, features, layer_indexes):
    """
    :param output_file:
    :param model:
    :param eval_dataloader:
    :param device:
    :param features:
    :param layer_indexes:
    :return:
    """
    model.eval()
    batch_count = len(eval_dataloader)
    batch_num = 0
    with open(output_file, "w", encoding='utf-8') as writer:
        for input_ids, input_mask, example_indices in eval_dataloader:
            batch_num += 1
            sys.stdout.write("\rBert Model For the {} Batch, All {} batch.".format(batch_num, batch_count))
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            all_encoder_layers, _ = model(input_ids, token_type_ids=None, attention_mask=input_mask)
            all_encoder_layers = all_encoder_layers

            for b, example_index in enumerate(example_indices):
                feature = features[example_index.item()]
                unique_id = int(feature.unique_id)
                # feature = unique_id_to_feature[unique_id]
                output_json = collections.OrderedDict()
                output_json["linex_index"] = unique_id
                all_out_features = []
                for (i, token) in enumerate(feature.tokens):
                    all_layers = []
                    for (j, layer_index) in enumerate(layer_indexes):
                        layer_output = all_encoder_layers[int(layer_index)].detach().cpu().numpy()
                        layer_output = layer_output[b]
                        layers = collections.OrderedDict()
                        layers["index"] = layer_index
                        layers["values"] = [round(x.item(), 6) for x in layer_output[i]][:args.bert_dim]
                        all_layers.append(layers)
                    out_features = collections.OrderedDict()
                    out_features["token"] = token
                    out_features["layers"] = all_layers
                    all_out_features.append(out_features)
                output_json["features"] = all_out_features
                writer.write(json.dumps(output_json) + "\n")


def to_json_by_maxpooling(args, output_file, model, eval_dataloader, device, features, layer_indexes):
    """
    :param args:
    :param output_file:
    :param model:
    :param eval_dataloader:
    :param device:
    :param features:
    :param layer_indexes:
    :return:
    """
    model.eval()
    batch_count = len(eval_dataloader)
    batch_num = 0
    if os.path.exists(output_file):
        os.remove(output_file)
    # with open(output_file, "w", encoding='utf-8') as writer:
    result = []
    for input_ids, input_mask, example_indices in eval_dataloader:
        batch_num += 1
        sys.stdout.write("\rBert Model For the {} Batch, All {} batch.".format(batch_num, batch_count))
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        all_encoder_layers, _ = model(input_ids, token_type_ids=None, attention_mask=input_mask)
        all_encoder_layers = all_encoder_layers
        feature_layer = all_encoder_layers[layer_indexes[-1]]
        feature_layer = feature_layer.permute(0, 2, 1)
        feature_layer = F.max_pool1d(feature_layer, feature_layer.size(2)).squeeze(2)
        for b, example_index in enumerate(example_indices):
            feature = features[example_index.item()]
            unique_id = int(feature.unique_id)
            output_json = collections.OrderedDict()
            output_json["linex_index"] = unique_id
            output_json["layer_index"] = layer_indexes[-1]
            layer_output = feature_layer[b].detach().cpu().numpy()
            output_json["values"] = [round(x.item(), 6) for x in layer_output][:args.bert_dim]
            result.append(output_json)

    print("\nto json file {}".format(output_file))
    line_num = 0
    file = open(output_file, mode="w", encoding="utf-8")
    for js in result:
        line_num += 1
        if line_num % 1000 == 0 :
            sys.stdout.write("\rBert Model Result For the {} line, All {} lines.".format(line_num, len(result)))
        file.write(json.dumps(js, ensure_ascii=False) + "\n")
    file.close()


def main(args):
    """
    :param args:
    :return:
    """

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {} distributed training: {}".format(device, n_gpu, bool(args.local_rank != -1)))
    # exit()
    layer_indexes = [int(x) for x in args.layers.split(",")]

    # tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    tokenizer = BertTokenizer.from_pretrained(os.path.join(args.bert_model, args.vocab), do_lower_case=args.do_lower_case)
    examples, max_seq_length = read_examples(args.input_file)
    # print(max_seq_length)
    # exit()

    features = convert_examples_to_features(
        examples=examples, seq_length=args.max_seq_length, tokenizer=tokenizer)

    unique_id_to_feature = {}
    for feature in features:
        unique_id_to_feature[feature.unique_id] = feature

    model = BertModel.from_pretrained(args.bert_model)
    model.to(device)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_example_index)
    if args.local_rank == -1:
        eval_sampler = SequentialSampler(eval_data)
    else:
        eval_sampler = DistributedSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)

    # to_json(args.output_file, model, eval_dataloader, device, features, layer_indexes)
    to_json_by_maxpooling(args, args.output_file, model, eval_dataloader, device, features, layer_indexes)


if __name__ == "__main__":

    max_seq_length = 100
    batch_size = 2
    bert_dim = 300
    # input = "./Data/cail_0518_jieba_length/data_valid.json"
    input = "./Data/cail_0530_short/data_valid.json"
    output = "./law_output/data_valid_bert_dim{}.json".format(bert_dim)
    bert_model = "./bert-base-chinese"
    vocab = "bert-base-chinese-vocab.txt"

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--input_file", default=input, type=str)
    parser.add_argument("--output_file", default=output, type=str)
    parser.add_argument("--bert_model", default=bert_model, type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--vocab", default=vocab, type=str)

    # Other parameters
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--layers", default="-1", type=str)
    parser.add_argument("--max_seq_length", default=max_seq_length, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                             "than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--batch_size", default=batch_size, type=int, help="Batch size for predictions.")
    parser.add_argument("--bert_dim", default=bert_dim, type=int, help="bert_dim.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")

    args = parser.parse_args()
    main(args)
