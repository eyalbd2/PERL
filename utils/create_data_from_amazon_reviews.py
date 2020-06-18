
import argparse
import xml.etree.ElementTree as ET
import random
import numpy as np
import pickle
import os


def XML2arrayRAW(neg_path, pos_path, num_lables=None):
    negReviews = []
    posReviews = []

    if os.path.isdir(neg_path):
        files = os.listdir(neg_path)
        for file in files:
            insert = False
            try:
                with open(neg_path + os.sep + file, 'r', encoding='utf_8') as f:
                    review = f.read().strip().replace("<br />", " ")
                insert = True
            except:
                pass
            if insert:
                negReviews.append(review)

    else:
        neg_tree = ET.parse(neg_path)
        neg_root = neg_tree.getroot()
        for rev in neg_root.iter('review'):
            negReviews.append(rev.text)

    if os.path.isdir(pos_path):
        files = os.listdir(pos_path)
        for file in files:
            insert = False
            try:
                with open(pos_path + os.sep + file, 'r', encoding='ascii') as f:
                    review = f.read()
                insert = True
            except:
                pass
            if insert:
                posReviews.append(review)
    else:
        pos_tree = ET.parse(pos_path)
        pos_root = pos_tree.getroot()
        for rev in pos_root.iter('review'):
            posReviews.append(rev.text)

    if num_lables is not None:
        random.shuffle(negReviews)
        negReviews = negReviews[:num_lables]
        random.shuffle(posReviews)
        posReviews = posReviews[:num_lables]

    reviews = negReviews + posReviews
    return reviews,negReviews,posReviews


def split_data_balanced(reviews, dataSize, testSize):
    test_data_neg = random.sample(range(0, dataSize), testSize)
    test_data_pos = random.sample(range(dataSize, 2*dataSize), testSize)
    random_array = np.concatenate((test_data_neg, test_data_pos))
    train = []
    test = []
    test_labels = []
    train_labels = []
    for i in range(0, 2*dataSize):
        if i in random_array:
            test.append(reviews[i])
            target = 0 if i < dataSize else 1
            test_labels.append(target)
        else:
            train.append(reviews[i])
            target = 0 if i < dataSize else 1
            train_labels.append(target)
    return train, train_labels, test, test_labels


def extract_and_split(neg_path, pos_path, num_lables=1000):
    reviews, n, p = XML2arrayRAW(neg_path, pos_path, num_lables=num_lables)
    train, train_labels, test, test_labels = split_data_balanced(reviews, 1000, 200)
    return train, train_labels, test, test_labels

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_lables",
                        default=1000,
                        type=int,
                        help="The number of labeled examples from each class.")
    parser.add_argument("--src",
                        default='books',
                        type=str,
                        help="Source domain.")

    args = parser.parse_args()

    neg_path = 'raw_data/' + args.src + '/negative.parsed'
    pos_path = 'raw_data/' + args.src + '/positive.parsed'
    train, train_labels, test, test_labels = extract_and_split(neg_path, pos_path, num_lables=1000)

    filename = base_path + src + "/split/"
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
        with open(filename + 'train', 'wb') as f:
            pickle.dump(train, f)
        with open(filename + 'test', 'wb') as f:
            pickle.dump(test, f)
        with open(filename + 'train_labels', 'wb') as f:
            pickle.dump(train_labels, f)
        with open(filename + 'test_labels', 'wb') as f:
            pickle.dump(test_labels, f)


if __name__ == "__main__":
    main()