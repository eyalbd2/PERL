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

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
import os
import random
from io import open
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import pickle
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

if not 'modeling' in sys.path:
    sys.path += ['modeling']
if not 'utils' in sys.path:
    sys.path += ['utils']

from perl_for_finetune import BertForMaskedLM

# TODO: if you are using 'pytorch 1.5.0' you cannot run parallel. In case you are using 'pytorch 1.4.0' you can comment
#  the following line
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class BERTDataset(Dataset):
    def __init__(self, src_domain, trg_domain, tokenizer, seq_len, pivot2id_dict, encoding="utf-8", corpus_lines=None,
                 on_memory=True, pivot_prob=0.5, non_pivot_prob=0.1):
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.on_memory = on_memory
        self.corpus_lines = corpus_lines  # number of non-empty lines in input corpus
        self.src_corpus_path = "data/" + src_domain + "/unlabeled"
        self.src_train_corpus_path = "data/" + src_domain + "/train"
        self.trg_corpus_path = "data/" + trg_domain + "/unlabeled"
        self.encoding = encoding
        self.current_doc = 0  # to avoid random sentence from same doc
        self.pivot2id_dict = pivot2id_dict
        self.pivot_prob = pivot_prob
        self.non_pivot_prob = non_pivot_prob

        # for loading samples directly from file
        self.sample_counter = 0  # used to keep track of full epochs on file
        self.line_buffer = None  # keep second sentence of a pair in memory and use as first sentence in next pair

        # for loading samples in memory
        self.current_random_doc = 0
        self.num_docs = 0
        self.sample_to_doc = []  # map sample index to doc and line

        pickle_in = open(self.src_corpus_path, "rb")
        corpus_src = pickle.load(pickle_in)
        pickle_in = open(self.src_train_corpus_path, "rb")
        corpus_train_src = pickle.load(pickle_in)
        pickle_in = open(self.trg_corpus_path, "rb")
        corpus_trg = pickle.load(pickle_in)
        # load samples into memory
        if on_memory:
            self.all_docs = []
            doc = []
            self.corpus_lines = 0

            for line in corpus_src:
                line = line.strip()
                if line == "":
                    self.all_docs.append(doc)
                    doc = []
                else:
                    # store as one sample
                    sample = {"doc_id": len(self.all_docs),
                              "line": len(doc)}
                    self.sample_to_doc.append(sample)
                    doc.append(line)
                    self.corpus_lines = self.corpus_lines + 1

            # if last row in file is not empty
            if len(self.all_docs) > 0:
                if self.all_docs[-1] != doc:
                    self.all_docs.append(doc)
            else:
                self.all_docs.append(doc)

            doc = []

            for line in corpus_train_src[0]:
                # print("---", line)
                line = line.strip()
                if line == "":
                    self.all_docs.append(doc)
                    doc = []
                else:
                    # store as one sample
                    sample = {"doc_id": len(self.all_docs),
                              "line": len(doc)}
                    self.sample_to_doc.append(sample)
                    doc.append(line)
                    self.corpus_lines = self.corpus_lines + 1

            # if last row in file is not empty
            if len(self.all_docs) > 0:
                if self.all_docs[-1] != doc:
                    self.all_docs.append(doc)
            else:
                self.all_docs.append(doc)

            doc = []

            for line in corpus_trg:
                line = line.strip()
                if line == "":
                    self.all_docs.append(doc)
                    doc = []
                else:
                    # store as one sample
                    sample = {"doc_id": len(self.all_docs),
                              "line": len(doc)}
                    self.sample_to_doc.append(sample)
                    doc.append(line)
                    self.corpus_lines = self.corpus_lines + 1

            # if last row in file is not empty
            if self.all_docs[-1] != doc:
                self.all_docs.append(doc)

            self.num_docs = len(self.all_docs)

        # load samples later lazily from disk
        else:
            raise ValueError("'on_memory' is currently the only implemented load execution option. Please set "
                             "`on_memory`.")

    def __len__(self):
        # We start counting at 0.
        return self.corpus_lines - 1

    def __getitem__(self, item):
        cur_id = self.sample_counter
        self.sample_counter += 1
        if not self.on_memory:
            raise ValueError("'on_memory' is currently the only implemented load execution option. Please set "
                             "`on_memory`.")

        t1, t2 = self.random_sent(item)

        # tokenize
        tokens_a = self.tokenizer.tokenize(t1)
        tokens_b = []

        # combine to one sample
        cur_example = InputExample(guid=cur_id, tokens_a=tokens_a, tokens_b=tokens_b)

        # transform sample to features
        cur_features = convert_example_to_features(cur_example, self.seq_len, self.tokenizer, self.pivot2id_dict,
                                                   self.pivot_prob, self.non_pivot_prob)

        cur_tensors = (torch.tensor(cur_features.input_ids),
                       torch.tensor(cur_features.input_mask),
                       torch.tensor(cur_features.segment_ids),
                       torch.tensor(cur_features.lm_label_ids))

        return cur_tensors

    def random_sent(self, index):
        """
        Get one sample from corpus consisting of one sentence. Sentence_B is always "".
        :param index: int, index of sample.
        :return: (str, str), sentence 1, sentence 2 (empty)
        """
        t1, t2 = self.get_corpus_line(index)

        assert len(t1) > 0
        assert len(t2) == 0
        return t1, t2

    def get_corpus_line(self, item):
        """
        Get one sample from corpus consisting of a pair of two subsequent lines from the same doc.
        :param item: int, index of sample.
        :return: (str, str), two subsequent sentences from corpus
        """
        t1 = ""
        t2 = ""
        assert item < self.corpus_lines
        if self.on_memory:
            sample = self.sample_to_doc[item]
            t1 = self.all_docs[sample["doc_id"]][sample["line"]]
            self.current_doc = sample["doc_id"]
            return t1, t2
        else:
            if self.line_buffer is None:
                # read first non-empty line of file
                while t1 == "":
                    t1 = next(self.file).strip()
            else:
                # use t2 from previous iteration as new t1
                t1 = self.line_buffer
                # skip empty rows that are used for separating documents and keep track of current doc id
                while t1 == "":
                    t1 = next(self.file).strip()
                    self.current_doc = self.current_doc + 1
            self.line_buffer = next(self.file).strip()

        assert t1 != ""
        assert t1 == ""
        return t1, t2

    def get_random_line(self):
        """
        Get random line from another document for nextSentence task.
        :return: str, content of one line
        """
        # Similar to original tf repo: This outer loop should rarely go for more than one iteration for large
        # corpora. However, just to be careful, we try to make sure that
        # the random document is not the same as the document we're processing.
        for _ in range(10):
            if self.on_memory:
                rand_doc_idx = random.randint(0, len(self.all_docs) - 1)
                rand_doc = self.all_docs[rand_doc_idx]
                line = rand_doc[random.randrange(len(rand_doc))]
            else:
                rand_index = random.randint(1, self.corpus_lines if self.corpus_lines < 1000 else 1000)
                # pick random line
                for _ in range(rand_index):
                    line = self.get_next_line()
            # check if our picked random line is really from another doc like we want it to be
            if self.current_random_doc != self.current_doc:
                break
        return line

    def get_next_line(self):
        """ Gets next line of random_file and starts over when reaching end of file"""
        try:
            line = next(self.random_file).strip()
            # keep track of which document we are currently looking at to later avoid having the same doc as t1
            if line == "":
                self.current_random_doc = self.current_random_doc + 1
                line = next(self.random_file).strip()
        except StopIteration:
            self.random_file.close()
            self.random_file = open(self.corpus_path, "r", encoding=self.encoding)
            line = next(self.random_file).strip()
        return line


class InputExample(object):
    """A single training/test example for the language model."""

    def __init__(self, guid, tokens_a, tokens_b=None, lm_labels=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            tokens_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            tokens_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        """
        self.guid = guid
        self.tokens_a = tokens_a
        self.tokens_b = tokens_b
        self.lm_labels = lm_labels  # masked words for language model


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, lm_label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.lm_label_ids = lm_label_ids


def random_word(tokens, tokenizer, pivot2id, pivot_prob=0.5, non_pivot_prob=0.1):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :param pivot2id: Dict, n-gram pivot features, for each pivot gives ID
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    output_label = []
    continue_flag = False
    for i, token in enumerate(tokens):
        if continue_flag:
            continue_flag = False
        else:
            prob = random.random()
            if tokens[-1] != token:
                if token + " " + tokens[i + 1] in pivot2id:
                    # print(token+" "+tokens[i+1] + ":" + str(pivot2id[token+" "+tokens[i+1]]))
                    if prob < pivot_prob:
                        prob /= pivot_prob
                        continue_flag = True
                        # 80% randomly change token to mask token
                        if prob < 0.8:
                            output_label.append(pivot2id[token + " " + tokens[i + 1]])
                            output_label.append(pivot2id[token + " " + tokens[i + 1]])
                            tokens[i] = "[MASK]"
                            tokens[i + 1] = "[MASK]"
                        else:
                            output_label.append(pivot2id[token + " " + tokens[i + 1]])
                            output_label.append(pivot2id[token + " " + tokens[i + 1]])
                        continue
            if token in pivot2id:
                # print(token + ":" + str(pivot2id[token]))
                if prob < pivot_prob:
                    prob /= pivot_prob
                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        tokens[i] = "[MASK]"
                        output_label.append(pivot2id[token])
                        continue
            # mask token with 15% probability
            elif prob < non_pivot_prob:
                prob /= 0.1
                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = "[MASK]"
                # 10% randomly change token to random token
                # elif prob < 0.9:
                #    tokens[i] = random.choice(list(pivot2id.items())[1:])[0]
                # -> rest 10% randomly keep current token

                # append current token to output (we will predict these later)
                output_label.append(0)
                continue
            output_label.append(-1)

    # print(output_label)
    return tokens, output_label


def convert_example_to_features(example, max_seq_length, tokenizer, pivot2id_dict, pivot_prob=0.5, non_pivot_prob=0.1):
    """
    Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
    IDs, LM labels, input_mask, CLS and SEP tokens etc.
    :param example: InputExample, containing sentence input as strings and is_next label
    :param max_seq_length: int, maximum length of sequence.
    :param tokenizer: Tokenizer
    :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
    """
    tokens_a = example.tokens_a
    tokens_b = example.tokens_b
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 2)

    tokens_a, t1_label = random_word(tokens_a, tokenizer, pivot2id_dict, pivot_prob, non_pivot_prob)
    # concatenate lm labels and account for CLS, SEP, SEP
    lm_label_ids = ([-1] + t1_label + [-1])

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
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
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        lm_label_ids.append(-1)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(lm_label_ids) == max_seq_length
    # print(lm_label_ids)

    if example.guid < 2:
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("tokens: %s" % " ".join(
            [str(x) for x in tokens]))
        logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        logger.info(
            "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        logger.info("LM label: %s " % (lm_label_ids))

    features = InputFeatures(input_ids=input_ids,
                             input_mask=input_mask,
                             segment_ids=segment_ids,
                             lm_label_ids=lm_label_ids)
    return features


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--src_domain",
                        default=None,
                        type=str,
                        required=True,
                        help="The src train corpus. One of: books, dvd, elctronics, kitchen.")
    parser.add_argument("--trg_domain",
                        default=None,
                        type=str,
                        required=True,
                        help="The trg corpus. One of: books, dvd, elctronics, kitchen.")
    parser.add_argument("--bert_model", default='bert-base-uncased', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default='models/books_to_electronics',
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--pivot_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The directory where needed pivots are."
                             "(as data/kitchen_to_books/pivots/40_bigram)")
    parser.add_argument("--pivot_prob",
                        default=0.5,
                        type=float,
                        required=True,
                        help="Probability to mask a pivot.")
    parser.add_argument("--non_pivot_prob",
                        default=0.1,
                        type=float,
                        required=True,
                        help="Probability to mask a non-pivot.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=True,
                        type=bool,
                        help="Whether to run training.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=100.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--save_every_num_epochs",
                        default=20.0,
                        type=float,
                        help="After how many epochs to save weights.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--on_memory",
                        default=True,
                        type=bool,
                        help="Whether to load train samples into memory or use disk")
    parser.add_argument("--do_lower_case",
                        default=True,
                        type=bool,
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--num_of_unfrozen_bert_layers',
                        type=int,
                        default=8,
                        help="Number of trainable BERT layers during pretraining.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--init_output_embeds',
                        action='store_true',
                        help="Whether to initialize pivots decoder with BERT embedding or not.")
    parser.add_argument('--train_output_embeds',
                        action='store_true',
                        help="Whether to train pivots decoder or not.")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    print("---- Pivots Path:", args.pivot_path)
    pickle_in = open(args.pivot_path, "rb")
    pivot_list = pickle.load(pickle_in)
    pivot2id_dict = {}
    id2pivot_dict = {}
    pivot2id_dict['NONE'] = 0
    id2pivot_dict[0] = 'NONE'
    for id, feature in enumerate(pivot_list):
        pivot2id_dict[feature] = id+1
        id2pivot_dict[id+1] = feature

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train:
        raise ValueError("Training is currently the only implemented execution option. Please set `do_train`.")

    dir_for_save = args.output_dir

    if os.path.exists(dir_for_save) and os.listdir(dir_for_save):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(dir_for_save))
    if not os.path.exists(dir_for_save):
        os.mkdir(dir_for_save)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        print("Loading Train Dataset from", args.src_domain, "and from", args.trg_domain)
        train_dataset = BERTDataset(args.src_domain, args.trg_domain, tokenizer, seq_len=args.max_seq_length,
                                    pivot2id_dict=pivot2id_dict, corpus_lines=None, on_memory=args.on_memory,
                                    pivot_prob=args.pivot_prob, non_pivot_prob=args.non_pivot_prob)
        num_train_optimization_steps = int(
            len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    model = BertForMaskedLM.from_pretrained(args.bert_model, output_dim=len(pivot2id_dict),
                                            init_embed=args.init_output_embeds, src=args.src_domain,
                                            trg=args.trg_domain)
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

    # freeze all bert weights, train only last encoder layer
    try:
        for param in model.bert.embeddings.parameters():
            param.requires_grad = False
        for id, param in enumerate(model.bert.encoder.layer.parameters()):
            if id < (192 * (12-args.num_of_unfrozen_bert_layers) / 12):
                param.requires_grad = False
        for param in model.cls.predictions.pivots_decoder.parameters():
            param.requires_grad = args.train_output_embeds
    except:
        for param in model.module.bert.embeddings.parameters():
            param.requires_grad = False
        for id, param in enumerate(model.module.bert.encoder.layer.parameters()):
            if id < (192 * (12 - args.num_of_unfrozen_bert_layers) / 12):
                param.requires_grad = False
        for param in model.module.cls.predictions.pivots_decoder.parameters():
            param.requires_grad = args.train_output_embeds

    # Prepare optimizer
    if args.do_train:
        param_optimizer = list(model.named_parameters())
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
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=args.learning_rate,
                                 warmup=args.warmup_proportion,
                                 t_total=num_train_optimization_steps)

    global_step = 0
    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
        else:
            # TODO: check if this works with current data generator from disk that relies on next(file)
            # (it doesn't return item back by index)
            train_sampler = DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        for cnt in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, lm_label_ids = batch
                loss = model(input_ids, segment_ids, input_mask, lm_label_ids)
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
            if (((cnt + 1) % args.save_every_num_epochs) == 0):
                # Save a trained model
                logger.info("** ** * Saving fine - tuned model ** ** * ")
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(dir_for_save, "pytorch_model" + str(cnt + 1) + ".bin")
                if args.do_train:
                    torch.save(model_to_save.state_dict(), output_model_file)
        logger.info("** ** * Saving fine - tuned model ** ** * ")
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(dir_for_save, "pytorch_model" + ".bin")
        if args.do_train:
            torch.save(model_to_save.state_dict(), output_model_file)


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


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


if __name__ == "__main__":
    main()