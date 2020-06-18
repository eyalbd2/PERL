#####################################################################################
# this code is partially taken from https://github.com/yftah89/PBLM-Domain-Adaptation
#####################################################################################

import argparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mutual_info_score
import pickle
import os
from pytorch_pretrained_bert import BertTokenizer


def GetTopNMI(n, X, target):
    MI = []
    length = X.shape[1]

    for i in range(length):
        temp = mutual_info_score(X[:, i], target)
        MI.append(temp)
    MIs = sorted(range(len(MI)), key=lambda i: MI[i])[-n:]
    return MIs,MI


def getCounts(X,i):
    return (sum(X[:,i]))


def preproc(pivot_num, pivot_min_st, src, dest, tokenizer=None, n_gram=(1,1)):
    """find pivots from source and data domains with mutual information
    Parameters:
    pivot_num (list of int or int): number of pivots to find
    pivot_min_st (int): minimal appearances of the pivot in both source and target domains
    src (str): source domain name
    dest (str): target domain name
    tokenizer: can be None or one of the following:
        'bert-base-uncased'
        'bert-large-uncased'
        'bert-base-cased'
        'bert-large-cased'
        None - default is words separated by whitespaces
    n_gram (tuple of integers): n_grams to include in pivots selection (min, max), default is 1 grams only
    Returns:
    list of pivots
   """

    # Load pre-trained model tokenizer (vocabulary):
    if tokenizer is not None:
        tokenizer = BertTokenizer.from_pretrained(tokenizer).tokenize

    base_path = os.getcwd() + os.sep
    pivotsCounts= []

    src_path = "data/" + src + os.sep
    with open(src_path + "train", 'rb') as f:
        (train, train_labels) = pickle.load(f)
    with open(src_path + "unlabeled", 'rb') as f:
        source = pickle.load(f)

    # gets all the train and test for pivot classification
    dest_path = "data/" + dest + os.sep
    with open(dest_path + "unlabeled", 'rb') as f:
        target = pickle.load(f)

    source = source + train
    unlabeled = source + target
    src_count = 20
    dest_count = 20
    un_count = 40

    # sets x train matrix for classification
    print('starting bigram_vectorizer for train data...')
    bigram_vectorizer = CountVectorizer(ngram_range=n_gram, token_pattern=r'\b\w+\b', min_df=5,binary=True, tokenizer=tokenizer)
    X_2_train = bigram_vectorizer.fit_transform(train).toarray()
    print('Done!')

    print('starting bigram_vectorizer for unlabled data...')
    bigram_vectorizer_unlabeled = CountVectorizer(ngram_range=n_gram, token_pattern=r'\b\w+\b', min_df=un_count, binary=True, tokenizer=tokenizer)
    bigram_vectorizer_unlabeled.fit_transform(unlabeled).toarray()
    # X_2_train_unlabeled = bigram_vectorizer_unlabeled.fit_transform(unlabeled).toarray()

    print('Done!')

    print('starting bigram_vectorizer for source data...')
    bigram_vectorizer_source = CountVectorizer(ngram_range=n_gram, token_pattern=r'\b\w+\b', min_df=src_count,
                                               binary=True, tokenizer=tokenizer)
    X_2_train_source = bigram_vectorizer_source.fit_transform(source).toarray()
    print('Done!')

    print('starting bigram_vectorizer for target data...')
    bigram_vectorizer_labels = CountVectorizer(ngram_range=n_gram, token_pattern=r'\b\w+\b', min_df=dest_count,
                                               binary=True, tokenizer=tokenizer)
    X_2_train_labels = bigram_vectorizer_labels.fit_transform(target).toarray()
    print('Done!')

    # get a sorted list of pivots with respect to the MI with the label
    print('starting calculating MI...')
    MIsorted, RMI = GetTopNMI(2000, X_2_train, train_labels)
    MIsorted.reverse()
    c=0
    if not isinstance(pivot_num, list):
        pivot_num = [pivot_num]
    names = []
    for i, MI_word in enumerate(MIsorted):
        name = bigram_vectorizer.get_feature_names()[MI_word]

        s_count = getCounts(X_2_train_source, bigram_vectorizer_source.get_feature_names().index(
            name)) if name in bigram_vectorizer_source.get_feature_names() else 0
        t_count = getCounts(X_2_train_labels, bigram_vectorizer_labels.get_feature_names().index(
            name)) if name in bigram_vectorizer_labels.get_feature_names() else 0

        # pivot must meet 2 conditions, to have high MI with the label and appear at least pivot_min_st times in the
        # source and target domains
        if s_count>=pivot_min_st and t_count>=pivot_min_st:
            names.append(name)
            pivotsCounts.append(bigram_vectorizer_unlabeled.get_feature_names().index(name))
            c+=1

            print("feature is ",name," it MI is ",RMI[MI_word]," in source ",s_count," in target ",t_count)

        if c>=max(pivot_num):
            break

    sfx = ''
    if n_gram[1] == 2:
        sfx = "_bi"
    filename = base_path + 'data/pivots/' + src + "_to_" + dest
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    for num in pivot_num:
        with open(filename + "/" + str(pivot_num[0]) + sfx, 'wb') as f:
            pickle.dump(names[:num], f)

    # returns the pivot list
    return names


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pivot_num",
                        default=100,
                        type=int,
                        help="The number of selected pivots")
    parser.add_argument("--pivot_min_st",
                        default=20,
                        type=int,
                        help="Minimum counts of pivots in src and in dest")
    parser.add_argument("--src",
                        default='books',
                        type=str,
                        help="Source domain.")
    parser.add_argument("--dest",
                        default='dvd',
                        type=str,
                        help="Destination domain.")
    parser.add_argument("--tokenizer_name",
                        default='bert-base-cased',
                        type=str,
                        help="The name of the tokenizer.")
    parser.add_argument("--n_gram",
                        default='bigram',
                        type=str,
                        help="N_gram length.")

    args = parser.parse_args()

    if args.n_gram == "bigram":
        n_gram = (1, 2)
    elif args.n_gram == "unigram":
        n_gram = (1, 1)
    else:
        print("This code does not soppurt this type of n_gram")
        exit(0)

    _ = preproc(args.pivot_num, args.pivot_min_st, args.src, args.dest, args.tokenizer_name, n_gram)


if __name__ == "__main__":
    main()