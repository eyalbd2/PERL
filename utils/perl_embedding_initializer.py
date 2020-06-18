import pickle
import torch
import numpy as np

from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer


def get_emb_weights(src, trg, num_pivots=100, bert_model='bert-base-uncased'):
    pivots_path = "data/pivots/{}_to_{}/{}_bi".format(src, trg, num_pivots)
    pickle_in = open(pivots_path, "rb")
    pivot_list = pickle.load(pickle_in)
    model = BertModel.from_pretrained(bert_model)
    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True if 'uncased' in bert_model else False)
    emb_list = []
    for pivot in pivot_list:
        emb_list.append(torch.mean(model.embeddings.word_embeddings.weight[
                                       tokenizer.convert_tokens_to_ids(tokenizer.tokenize(pivot))].detach(),
                                   axis=0).numpy().tolist())
    return torch.tensor(emb_list)

