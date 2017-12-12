import string
import numpy as np
import pandas as pd
from nltk.tag.stanford import CoreNLPNERTagger
from nltk.parse.corenlp import CoreNLPDependencyParser
from io import open
from nltk.corpus import brown

from nltk import word_tokenize, pos_tag
from nltk.parse import BllipParser
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords, wordnet_ic
from nltk.chunk import ne_chunk
from nltk.data import find
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from nltk.metrics import *
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.graph_objs import Scatter, Figure, Layout

# init_notebook_mode(connected=True)
brown_ic = wordnet_ic.ic('ic-brown.dat')

'''
    A collection of useful functions for nlp
'''


# Removing the punctuation and lowering the case of a string
def remove_punctuation(line):
    return line.translate(str.maketrans('', '', string.punctuation))


# extract the words from the sentence
def words_from_sent(sent):
    # tokenized
    tokenized = word_tokenize(sent)
    # remove stopwords and return
    return [word for word in tokenized if word not in stopwords.words('english')]


# convert words to tokens
def tokens_from_words(words):
    return pos_tag(words)


# Function to get wordnet pos code
def wordnet_pos_code(tag):
    if tag.startswith('NN'):
        return wordnet.NOUN
    elif tag.startswith('VB'):
        return wordnet.VERB
    elif tag.startswith('JJ'):
        return wordnet.ADJ
    elif tag.startswith('RB'):
        return wordnet.ADV
    else:
        return None


# Tokens to lemmas using wordnet lemmatizer
def tokens_to_lemmas(tokens):
    return list(map(token_to_lemmas, tokens))


def token_to_lemmas(token):
    pos = wordnet_pos_code(token[1])
    if pos:
        return WordNetLemmatizer().lemmatize(token[0], pos=pos)
    return token[0]


# Tokens to lemmas and senses (top_synset + pos)
def extract_lemmas_senses(tokens):
    lemmas, top_synsets = [], []
    for token in tokens:
        pos = token[1]
        wn_pos = wordnet_pos_code(pos)
        lemma = WordNetLemmatizer().lemmatize(token[0])
        lemmas.append(lemma)
        if wn_pos:
            synsets = wordnet.synsets(lemma, pos=wn_pos)
            if len(synsets) > 0:
                top_synsets.append([synsets[0], pos])
    return lemmas, top_synsets


# Load the lines of training text as sentences
def text_to_sentences(filename):
    sentence_pair_array = []
    for line in open(filename, encoding="UTF8").readlines():
        sentence_pair_array.append([s.strip() for s in line.split("\t")])
    return sentence_pair_array


def compare_synsets(synset_a, synset_b):
    lcs = synset_a.lowest_common_hypernyms(synset_b)
    similarity = synset_a.path_similarity(synset_b)
    wup_similarity = synset_a.wup_similarity(synset_b)
    lin_similarity = synset_a.lin_similarity(synset_b, brown_ic)
    lch_similarity = synset_a.lch_similarity(synset_b)
    return lcs, similarity, lin_similarity, wup_similarity, lch_similarity


def count(g, s):
    TP = TN = FP = FN = 0
    for i in range(0, len(g)):
        if g[i] == s[i] and s[i] == 1: TP += 1
        if g[i] == s[i] and s[i] == 0: TN += 1
        if g[i] != s[i] and s[i] == 1: FP += 1
        if g[i] != s[i] and s[i] == 0: FN += 1
    return [TP, TN, FP, FN]


def MSRP_eval(gs, sys):
    [TP, TN, FP, FN] = count(gs, sys)
    acc = (TP + TN) / float(TP + TN + FP + FN)  # ACCURACY
    reject = TN / float(TN + FP)  # precision on negative SPECIFICITY
    accept = TP / float(TP + FN)  # precision on positive SENSITIVITY
    print("acc=", acc, " reject=", reject, " accept=", accept)


def word_order_vector(sentence, union, indexes, threshold):
    order_vector = []
    for word in union:
        if word in sentence:
            order_vector.append(indexes[word])
        else:
            sim_word, max_sim = max_similarity(word, sentence)
            if max_sim > threshold:
                order_vector.append(indexes[sim_word])
            else:
                order_vector.append(0)
    return np.array(order_vector)


def max_similarity(synset, sentence):
    max_sim = -1.0
    sim_word = ""
    for w2 in sentence:
        similarity = synset.path_similarity(w2)
        if similarity and similarity > max_sim:
            max_sim = similarity
            sim_word = w2
    return sim_word, max_sim


# Semantic Similarity
def semantic_similarity(sentence1, sentence2):
    # Sum the synset similarity for every word in sentence1
    simS1 = 0
    simS2 = 0
    for ws1 in sentence1:
        simS1 += max_similarity(ws1, sentence2)[1]

    for ws2 in sentence2:
        simS2 += max_similarity(ws2, sentence1)[1]

    return 1 / 2 * ((simS1 / len(sentence1)) + (simS2 / len(sentence2)))


'''
    The paraphrase data that will be analysed
'''
train_input = text_to_sentences('/Users/felipe/Developer/IHLT/project2/train/msr_paraphrase_train_input.txt')
test_input = text_to_sentences('/Users/felipe/Developer/IHLT/project2/test/msr_paraphrase_test_input.txt')
train_classes = open('/Users/felipe/Developer/IHLT/project2/train/msr_paraphrase_train_gs.txt',
                     encoding="utf-8-sig").readlines()
test_classes = open('/Users/felipe/Developer/IHLT/project2/test/msr_paraphrase_test_gs.txt',
                    encoding="UTF8").readlines()

'''
    Word Order
'''


def word_order(sent_0, sent_1):
    print('.', end='')
    sent_0, sent_1 = remove_punctuation(sent_0.lower()), remove_punctuation(sent_1.lower())
    words_0, words_1 = words_from_sent(sent_0), words_from_sent(sent_1)
    tokens_0, tokens_1 = tokens_from_words(words_0), tokens_from_words(words_1)
    lemmas_0, senses_0 = extract_lemmas_senses(tokens_0)
    lemmas_1, senses_1 = extract_lemmas_senses(tokens_1)
    synsets_0 = [sense[0] for sense in senses_0]
    synsets_1 = [sense[0] for sense in senses_1]

    union = list(set(synsets_0).union(set(synsets_1)))
    indexes = {tup[1]: tup[0] for tup in enumerate(union)}
    r1 = word_order_vector(synsets_0, union, indexes, 0.4)
    r2 = word_order_vector(synsets_1, union, indexes, 0.4)
    return [semantic_similarity(synsets_0, synsets_1),
            1.0 - (np.linalg.norm(r1 - r2) / np.linalg.norm(r1 + r2))]


print('Training')
X_train = [word_order(data[0], data[1]) for data in train_input]
y_train = [int(line.strip()) for line in train_classes]
print('Testing')
X_test = [word_order(data[0], data[1]) for data in test_input]
y_test = [int(line.strip()) for line in test_classes]
print('Results')
regression = LogisticRegression()
regression.fit(np.array(X_train), y_train)
prediction = regression.predict(np.array(X_test))
MSRP_eval(prediction, y_test)
