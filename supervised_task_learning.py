# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import sklearn

from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from scipy.stats import entropy
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

if not 'modeling' in sys.path:
    sys.path += ['modeling']
if not 'utils' in sys.path:
    sys.path += ['utils']

from classification_model import CNNBertForSequenceClassification
from logger import Logger

# TODO: if you are using 'pytorch 1.5.0' you cannot run parallel. In case you are using 'pytorch 1.4.0' you can comment
#  the following line
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class SentimentProcessor(DataProcessor):
    """Processor for the Domain Adaptation Sentiment classification data set."""

    def get_train_examples(self, data_dir, use_fold, fold_num):
        """See base class."""
        if use_fold:
            src = data_dir.split('/')[1]
            fold_data_dir = '5-fold_data/' + src + '/fold-' + str(fold_num) + '/'
            train_path = os.path.join(fold_data_dir, "train")
        else:
            train_path = os.path.join(data_dir, "train")
        logger.info("LOOKING AT {}".format(train_path))
        with open(train_path, 'rb') as f:
            (train, labels) = pickle.load(f)
        return self._create_examples(train, labels, 'train')

    def get_dev_examples(self, data_dir, use_fold, fold_num):
        """See base class."""
        if use_fold:
            src = data_dir.split('/')[1]
            fold_data_dir = '5-fold_data/' + src + '/fold-' + str(fold_num) + '/'
            test_path = os.path.join(fold_data_dir, "dev")
        else:
            test_path = os.path.join(data_dir, "dev")
        logger.info("LOOKING AT {}".format(test_path))
        with open(test_path, 'rb') as f:
            (test, labels) = pickle.load(f)
        return self._create_examples(test, labels, 'dev')

    def get_test_examples(self, data_dir):
        """See base class."""
        test_path = os.path.join(data_dir, "test")
        logger.info("LOOKING AT {}".format(test_path))
        with open(test_path, 'rb') as f:
            (test, labels) = pickle.load(f)
        return self._create_examples(test, labels, 'dev_cross')

    def get_labels(self):
        """See base class."""
        return ["negative", "positive"]

    def _create_examples(self, x, label, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, data_point) in enumerate(zip(x, label)):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = data_point[0]
            text_b = None
            label = "positive" if (data_point[1]==1 or data_point[1] == "positive") else "negative"
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # If unlabeled example - mask pivots
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]


        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
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


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return {"acc": simple_accuracy(preds, labels)}


def evaluate(eval_dataloader, model, device, tokenizer, num_labels=2):
    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    preds = []

    for eval_element in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids, input_mask, segment_ids, label_ids = eval_element[:4]
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
             outputs = model(input_ids, segment_ids, input_mask, labels=None)
             logits = outputs

        # create eval loss and other metric required by the task
        loss_fct = CrossEntropyLoss()
        tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
            all_label_ids = label_ids.detach().cpu().numpy()
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)
            all_label_ids = np.append(all_label_ids, label_ids.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = preds[0]
    preds = np.argmax(preds, axis=1)

    acc = simple_accuracy(preds, all_label_ids)
    model.train()
    return acc, eval_loss


def make_DataLoader(data_dir, processor, tokenizer, label_list, max_seq_length, batch_size=6,
                    local_rank=-1, mode="train", N=-1, use_fold=True, fold_num=1):
    if mode == "train":
        examples = processor.get_train_examples(data_dir, use_fold, fold_num)
    elif mode == "dev":
        examples = processor.get_dev_examples(data_dir, use_fold, fold_num)
    elif mode == "dev_cross":
        examples = processor.get_test_examples(data_dir)
    if N > 0:
        examples = examples[:N]
    features = convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer)
    logger.info("***** Running evaluation on {}-set *****".format(mode))
    logger.info("  Num examples = %d", len(examples))
    logger.info("  Batch size = %d", batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    if mode == "train":
        if local_rank == -1:
            sampler = RandomSampler(data)
        else:
            sampler = DistributedSampler(data)
    elif mode == "dev" or mode == "dev_cross":
        # Run prediction for full data
        sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    return dataloader


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--in_domain_data_dir",
                        default='data/kitchen_to_books/split/',
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--cross_domain_data_dir",
                        default='data/kitchen_to_books/split/',
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default='bert-base-uncased', type=str, required=False,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--log_dir",
                        default='../log',
                        type=str,
                        help="The log output dir.")
    parser.add_argument("--load_model",
                        action='store_true',
                        help="Whether to load a fine-tuned model from output directory.")
    parser.add_argument("--model_name",
                        default=None,
                        type=str,
                        help="The name of the model to load, relevant only in case that load_model is positive.")
    parser.add_argument("--load_model_path",
                        default='',
                        type=str,
                        help="Path to directory containing fine-tuned model.")
    parser.add_argument("--save_on_epoch_end",
                        action='store_true',
                        help="Whether to save the weights each time an epoch ends.")
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--N_train",
                        type=int,
                        default=-1,
                        help="number of training examples")
    parser.add_argument("--N_dev",
                        type=int,
                        default=-1,
                        help="number of development examples")
    parser.add_argument("--cnn_window_size",
                        type=int,
                        default=9,
                        help="CNN 1D-Conv window size")
    parser.add_argument("--cnn_out_channels",
                        type=int,
                        default=16,
                        help="CNN 1D-Conv out channels")
    parser.add_argument("--save_best_weights",
                        type=bool,
                        default=False,
                        help="saves model weight each time epoch accuracy is maximum")
    parser.add_argument("--write_log_for_each_epoch",
                        type=bool,
                        default = False,
                        help = "whether to write log file at the end of every epoch or not")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--bert_output_layer_num",
                        default=12,
                        type=int,
                        help="Which BERT's encoder layer to use as output, used to check if it is possible to use "
                             "smaller BERT.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=5,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--use_fold',
                        type=bool,
                        default=False,
                        help="Whether to use 5-fold split data")
    parser.add_argument('--fold_num',
                        type=int,
                        default=1,
                        help="what number of fold to use")
    parser.add_argument('--save_according_to',
                        type=str,
                        default='acc',
                        help="save results according to in domain dev acc or in domain dev loss")                  
    parser.add_argument('--optimizer',
                        type=str,
                        default='adam',
                        help="which optimizer model to use: adam or sgd")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    logger.info("learning rate: {}, batch size: {}".format(
        args.learning_rate, args.train_batch_size))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.load_model:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if os.path.exists(args.output_dir):
        print(("Output directory ({}) already exists and is not empty.".format(args.output_dir)))
    else:
        os.makedirs(args.output_dir)

    logger.info("cnn out channels: {}, cnn window size: {}".format(
        args.cnn_out_channels, args.cnn_window_size))

    processor = SentimentProcessor()

    label_list = processor.get_labels()
    num_labels = len(label_list)

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.in_domain_data_dir, args.use_fold, args.fold_num)
        train_examples = train_examples[:args.N_train] if args.N_train > 0 else train_examples
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Load a trained model and vocabulary that you have fine-tuned
    if args.load_model or args.load_model_path != '':

        # path to directory to load from fine-tuned model
        load_path = args.load_model_path if args.load_model_path != '' else args.output_dir
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                       'distributed_{}'.format(args.local_rank))

        model = CNNBertForSequenceClassification.from_pretrained(args.bert_model,
                                                                 cache_dir=cache_dir,
                                                                 num_labels=num_labels,
                                                                 hidden_size=768,
                                                                 max_seq_length=args.max_seq_length,
                                                                 filter_size=args.cnn_window_size,
                                                                 out_channels=args.cnn_out_channels,
                                                                 output_layer_num=args.bert_output_layer_num)

        # load pre train modek weights
        if args.model_name is not None:
            print("--- Loading model:", args.output_dir + args.model_name)
            model.load_state_dict(torch.load(args.output_dir + args.model_name), strict=False)
        else:
            model.load_state_dict(torch.load(os.path.join(load_path, "pytorch_model.bin")), strict=False)

        tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        if not tokenizer:
            tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
        model.to(device)

    else:
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                       'distributed_{}'.format(args.local_rank))

        model = CNNBertForSequenceClassification.from_pretrained(args.bert_model,
                                                                 cache_dir=cache_dir,
                                                                 num_labels=num_labels,
                                                                 hidden_size=768,
                                                                 max_seq_length=args.max_seq_length,
                                                                 filter_size=args.cnn_window_size,
                                                                 out_channels=args.cnn_out_channels,
                                                                 output_layer_num=args.bert_output_layer_num)

        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
        model.to(device)

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # freeze all bert weights, train only classifier layer
    try:
        for param in model.module.bert.embeddings.parameters():
            param.requires_grad = False
        for param in model.module.bert.encoder.parameters():
            param.requires_grad = False
    except:
        for param in model.bert.embeddings.parameters():
            param.requires_grad = False
        for param in model.bert.encoder.parameters():
            param.requires_grad = False

    # Prepare optimizer
    if args.do_train:
        try:
            param_optimizer = list(model.module.classifier.named_parameters()) + list(model.module.conv1.named_parameters())
        except:
            param_optimizer = list(model.classifier.named_parameters()) + list(model.conv1.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        if args.fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.learning_rate,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            if args.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
            warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
                                                 t_total=num_train_optimization_steps)

        else:
            if args.optimizer == 'adam':
                optimizer = BertAdam(optimizer_grouped_parameters,
                                     lr=args.learning_rate,
                                     warmup=args.warmup_proportion,
                                     t_total=num_train_optimization_steps)
            elif args.optimizer == 'sgd':
                optimizer = torch.optim.sgd(model.parameters(), lr=args.learning_rate, weight_decay=1e-2)

    global_step = 0

    # prepare dev-set evaluation DataLoader
    # if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    eval_dataloader = make_DataLoader(data_dir=args.in_domain_data_dir,
                                      processor=processor,
                                      tokenizer=tokenizer,
                                      label_list=label_list,
                                      max_seq_length=args.max_seq_length,
                                      batch_size=args.eval_batch_size,
                                      local_rank=args.local_rank,
                                      mode="dev",
                                      N=args.N_dev,
                                      use_fold=args.use_fold,
                                      fold_num=args.fold_num)
    # Evaluate on cross domain development set
    eval_cross_dataloader = make_DataLoader(data_dir=args.cross_domain_data_dir,
                                            processor=processor,
                                            tokenizer=tokenizer,
                                            label_list=label_list,
                                            max_seq_length=args.max_seq_length,
                                            batch_size=args.eval_batch_size,
                                            local_rank=args.local_rank,
                                            mode="dev_cross",
                                            N=args.N_dev)

    if args.do_train:

        # creat results logger
        log_dir_path = os.path.join(args.log_dir, os.path.basename(args.output_dir))
        print("\nsaving logs to {}\n".format(log_dir_path))
        os.makedirs(log_dir_path, exist_ok=1)
        results_logger = Logger(log_dir_path)
        os.chmod(log_dir_path, 0o775)
        os.chmod(args.log_dir, 0o775)

        # prepare training DataLoader
        train_dataloader = make_DataLoader(data_dir=args.in_domain_data_dir,
                                           processor=processor,
                                           tokenizer=tokenizer,
                                           label_list=label_list,
                                           max_seq_length=args.max_seq_length,
                                           batch_size=args.train_batch_size,
                                           local_rank=args.local_rank,
                                           mode="train",
                                           N=args.N_train,
                                           use_fold=args.use_fold,
                                           fold_num=args.fold_num)
        model.train()

        # main training loop
        best_dev_acc = 0.0
        best_dev_loss = 100000.0
        best_dev_cross_acc = 0.0
        in_domain_best, cross_domain_best = {}, {}
        in_domain_best['in'] = 0.0
        in_domain_best['cross'] = 0.0
        cross_domain_best['in'] = 0.0
        cross_domain_best['cross'] = 0.0

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):  # (int(args.num_train_epochs), desc="Epoch"):

            tr_loss = 0
            tr_acc = 0

            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch[:4]

                # define a new function to compute loss values for both output_modes
                logits = model(input_ids, segment_ids, input_mask, labels=None)
                loss_fct = CrossEntropyLoss(ignore_index=-1)
                loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
                preds = logits.detach().cpu().numpy()
                preds = np.argmax(preds, axis=1)
                tr_acc += compute_metrics(preds, label_ids.detach().cpu().numpy())["acc"]


                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear.get_lr(global_step, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            # run evaluation on dev set
            # dev-set loss
            eval_results_dev = evaluate(eval_dataloader=eval_dataloader,
                                        model=model,
                                        device=device,
                                        tokenizer=tokenizer,
                                        num_labels=num_labels)

            dev_acc, dev_loss = eval_results_dev[:2]

            # train-set loss
            tr_loss /= nb_tr_steps
            tr_acc /= nb_tr_steps

            # print and save results
            result = {"acc": tr_acc, "loss": tr_loss, "dev_acc":dev_acc, "dev_loss": dev_loss}

            eval_results_test = evaluate(eval_dataloader=eval_cross_dataloader,
                                         model=model,
                                         device=device,
                                         tokenizer=tokenizer,
                                         num_labels=num_labels)
            dev_cross_acc, dev_cross_loss = eval_results_test[:2]
            result["dev_cross_acc"] = dev_cross_acc
            result["dev_cross_loss"] = dev_cross_loss

            results_logger.log_training(tr_loss, tr_acc, epoch)
            results_logger.log_validation(dev_loss, dev_acc, dev_cross_loss, dev_cross_acc, epoch)
            results_logger.close()

            print('Epoch {}'.format(epoch + 1))
            for key, val in result.items():
                print("{}: {}".format(key, val))

            if args.write_log_for_each_epoch:
                output_eval_file = os.path.join(args.output_dir, "eval_results_Epoch_{}.txt".format(epoch + 1))
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Evaluation results *****")
                    for key in sorted(result.keys()):
                        logger.info("  %s = %s", key, str(result[key]))
                        writer.write("%s = %s\n" % (key, str(result[key])))
            else:
                logger.info("***** Evaluation results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))

            # Save model, configuration and tokenizer on the first epoch
            # If we save using the predefined names, we can load using `from_pretrained`
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            if epoch == 0:
                output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
                model_to_save.config.to_json_file(output_config_file)
                tokenizer.save_vocabulary(args.output_dir)

            if args.save_on_epoch_end:
                # Save a trained model
                output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME + '.Epoch_{}'.format(epoch+1))
                torch.save(model_to_save.state_dict(), output_model_file)

            # save model with best performance on dev-set
            if args.save_best_weights and dev_acc > best_dev_acc:
                print("Saving model, accuracy improved from {} to {}".format(best_dev_acc, dev_acc))
                output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
                torch.save(model_to_save.state_dict(), output_model_file)
                best_dev_acc = dev_acc
            
            if args.save_according_to == 'acc':
                if dev_acc > best_dev_acc:
                    best_dev_acc = dev_acc
                    in_domain_best['in'] = best_dev_acc
                    in_domain_best['cross'] = dev_cross_acc

            elif args.save_according_to == 'loss':
                if dev_loss < best_dev_loss:
                    best_dev_loss = dev_loss
                    in_domain_best['in'] = best_dev_loss
                    in_domain_best['cross'] = dev_cross_acc

            if args.save_according_to == 'acc':
                print('Best results in domain: Acc - {}, Cross Acc - {}'.format(in_domain_best['in'], in_domain_best['cross']))
            elif args.save_according_to == 'loss':
                print('Best results in domain: Loss - {}, Cross Acc - {}'.format(in_domain_best['in'], in_domain_best['cross']))
            if args.model_name is not None:
                final_output_eval_file = os.path.join(args.output_dir, args.model_name + "-final_eval_results.txt")
            else:
                final_output_eval_file = os.path.join(args.output_dir, "final_eval_results.txt")

            with open(final_output_eval_file, "w") as writer:
                writer.write("Results:")
                writer.write("%s = %s\n" % ('in', str(in_domain_best['in'])))
                writer.write("%s = %s\n" % ('cross', str(in_domain_best['cross'])))

    elif args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # dev-set loss
        acc, dev_loss = evaluate(eval_dataloader=eval_dataloader,
                                 model=model,
                                 device=device,
                                 tokenizer=tokenizer,
                                 num_labels=num_labels)

        # print results
        print('Accuracy: {}'.format(acc))

    else:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")


if __name__ == "__main__":
    main()
