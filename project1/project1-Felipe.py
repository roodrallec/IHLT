import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
import numpy as np
from nltk.tag.stanford import CoreNLPNERTagger
from collections import Counter
from sklearn import datasets, linear_model
lemmatizer = WordNetLemmatizer()

"""

Get the NE for every word in the sentence.
Join the proper nouns as one single word.
Count the appearances by every word in the sentence.

This needs the CORE NLP Tagger to be running in localhost, port number 9000

"""


# Get NEs
def get_count_ne(sentence):
    counter = Counter(word_tokenize(sentence.lower()))
    try:
        tags = CoreNLPNERTagger(url='http://localhost:9000').tag(word_tokenize(sentence))
    except:
        result = []
    else:
        result = []
        pre_token, pre_tag = tags.pop(0)
        while tags:
            c_token, c_tag = tags.pop(0)
            if pre_tag != c_tag or pre_tag == 'O':
                if counter[pre_token] != 0:
                    result.append(((pre_token, pre_tag), counter[pre_token]))
                else:
                    result.append(((pre_token, pre_tag), 1))
                pre_token = c_token
            else:
                pre_token = " ".join([pre_token, c_token])
            pre_tag = c_tag
    return result


"""
Extract the correct lemma for every word in a sentence
"""


# Lemmatize a word
def lemmatize(word):
    if word[1][0] in {'N', 'V'}:
        return lemmatizer.lemmatize(word=word[0], pos=word[1][0].lower())
    return word[0]


# Get the lemmas for every word in a sentence
def get_lemmas(sentence):
    # Tokenize the text
    sentence = sentence.replace('"', '').replace("'", "").replace(",", "").replace("$", "")
    token = nltk.word_tokenize(sentence)

    # Take the token POS
    token_pos = []
    for ts in token:
        tkn = nltk.pos_tag(ts)
        token_pos.append(tkn)

    # Extract the lemmas
    lemmas = []
    for sentence_t_pos in token_pos:
        temp = []
        for t_pos in sentence_t_pos:
            temp.append(lemmatize(t_pos))
        lemmas.append(temp)
    return lemmas


"""

Calculates the word overlap.
This is the words in common across both sentences divided by the length of the first sentence.

"""


# Word Overlap
def word_overlap(sentence1, sentence2):
    if (len(sentence1) > len(sentence2)):
        size = len(sentence2)
    else:
        size = len(sentence1)
    return len(intersection(sentence1, sentence2)) / size


"""

Semantic Similarity

Calculates the similarity by analyzing the words in both sentences that are the closes to each other using 
path similarity.

"""


# Semantic Similarity
def semantic_similarity(sentence1, sentence2):
    # Sum the synset similarity for every word in sentence1
    simS1 = 0
    simS2 = 0
    for ws1 in sentence1:
        simS1 += max_similarity(ws1, sentence2)[0]

    for ws2 in sentence2:
        simS2 += max_similarity(ws2, sentence1)[0]

    return 1 / 2 * ((simS1 / len(sentence1)) + (simS2 / len(sentence2)))


# Get the largest similarity between a sinset and a sentence
def max_similarity(word, sentence):
    maxS = 0
    maxSW = ''
    for w2 in sentence:
        sim = w2.path_similarity(word)
        if sim and sim >= maxS:
            maxS = sim
            maxSW = w2.name()
    return maxS, maxSW


"""
Extracting senses. These functions extract the synsets from every sentence
"""


# Get semantic sense of a word (verb or noun) and it's pos
def get_sense(word):
    synset = []
    if word[1][0] in {'N', 'V'}:
        if word[1][0] in 'N':
            synset = wn.synsets(word[0], wn.NOUN)
        if word[1][0] in 'V':
            synset = wn.synsets(word[0], wn.VERB)
    if synset:
        return synset[0]
    else:
        return None


# Get the sinsets from a sentence
# Remove the punctuation marks
def extract_sense(sentence):
    sentence = sentence.replace('.', '').replace('"', '').replace(',', '')
    # Take the token POS
    sPOS = nltk.pos_tag(nltk.word_tokenize(sentence))
    result = []
    for s in sPOS:
        if get_sense(s):
            result.append(get_sense(s))
    return result


"""
The Jaccard index functions.

The Jaccard index is calculated by dividing the length of the intersection by the length of the union.

"""


# Jaccard Function
def jaccard(list1, list2):
    inters = len(intersection(list1, list2))
    uni = len(union(list1, list2))
    return inters / uni


# Union Function
def union(list1, list2):
    un = []
    for i in list1:
        if not i in un:
            un.append(i)
    for i in list2:
        if not i in un:
            un.append(i)
    return un


# Intersection Function
def intersection(list1, list2):
    inter = []
    for i in list1:
        if i in list2 and (not i in inter):
            inter.append(i)
    return inter


"""
Functions to determine the threshold to use for every statistic

"""


# With the results from the training set and the expected results create a logistic regression model
# Get the threshold from the model
def get_threshold(results, expected):
    pos = []
    neg = []

    res = np.array(results).reshape((-1, 1))

    # Create logistic regression classifier
    regr = linear_model.LogisticRegression(solver='liblinear', max_iter=100)
    regr.fit(res, expected)

    # Get the value of the threshold
    return float(-regr.intercept_[0] / regr.coef_[0][0])


"""

I/O functions

"""


# Read file
def read_file(filename):
    # Read the file line by line
    with open(filename, 'r', encoding="utf-8-sig") as f:
        content = f.readlines()
    # Remove the '/n'
    content = [x.strip() for x in content]
    sentence_pairs = []
    for pair in content:
        sentence_pairs.append(pair.split(' \t '))
    return sentence_pairs


# compute the accuracy, the rejection and the acceptance of a system response
def read_integer_file(name):
    # read lines of a text file

    with open(name) as f:
        lines = f.readlines()
    # remove`\n` at the end of each line
    lines = [x.strip() for x in lines]
    for i in range(len(lines)):
        if lines[i] == '\ufeff1':
            lines[i] = 1
        else:
            lines[i] = int(lines[i])
    return lines


def count(g, s):
    TP = TN = FP = FN = 0
    for i in range(0, len(g)):
        if g[i] == s[i] and s[i] == 1: TP += 1
        if g[i] == s[i] and s[i] == 0: TN += 1
        if g[i] != s[i] and s[i] == 1: FP += 1
        if g[i] != s[i] and s[i] == 0: FN += 1
    return [TP, TN, FP, FN]


def MSRP_eval(name1, name2):
    gs = name1
    # gs = read_integer_file(name1)
    sys = read_integer_file(name2)
    [TP, TN, FP, FN] = count(gs, sys)
    acc = (TP + TN) / float(TP + TN + FP + FN)
    reject = TN / float(TN + FP)  # precision on negative
    accept = TP / float(TP + FN)  # precision on positive
    print([TP, TN, FP, FN])
    print("acc=", acc, " reject=", reject, " accept=", accept)


"""

Run the analysis on a dataset and compare it to the results.

"""


# Train the dataset
# Returns the three statistics
def train_dataset(filename):
    # Read the file line by line
    sentence_pairs = read_file(filename)

    semant, wordOver, jacc = [], [], []
    for pair in sentence_pairs:
        # Get lemmas for every sentences
        lemma1 = get_lemmas(pair[0])
        lemma2 = get_lemmas(pair[1])

        # Get NEs for every pair
        s1, s2 = get_count_ne(pair[0]), get_count_ne(pair[1])

        # Jaccard with nes
        jacc.append(jaccard(s1, s2))

        # Word overlap
        wordOver.append(word_overlap(lemma1, lemma2))

        # Semantic similarity
        semant.append(semantic_similarity(extract_sense(pair[0]), extract_sense(pair[1])))
    return jacc, wordOver, semant


def test_dataset(filename, gs, thresholds):
    # Read the file
    sentence_pairs = read_file(filename)

    semant, wordOver, jacc = [], [], []
    for pair in sentence_pairs:
        # Get lemmas for every sentences
        lemma1 = get_lemmas(pair[0])
        lemma2 = get_lemmas(pair[1])

        # Get NEs for every pair
        s1 = get_count_ne(pair[0])
        s2 = get_count_ne(pair[1])

        # Jaccard with nes
        jacc.append(jaccard(s1, s2) >= thresholds[0])

        # Word overlap
        wordOver.append(word_overlap(lemma1, lemma2) >= thresholds[1])

        # Semantic similarity
        semant.append(semantic_similarity(extract_sense(pair[0]), extract_sense(pair[1])) >= thresholds[2])

    print("Semantic Similarity")
    MSRP_eval(semant, gs)
    print("Word Overlap")
    MSRP_eval(wordOver, gs)
    print("Jaccard")
    MSRP_eval(jacc, gs)


def main(test_data, test_gs, train_data, train_gs):
    (jacc, wordOver, semant) = train_dataset(train_data)
    print("Training Started")
    gs = read_integer_file(train_gs)

    thresholds = get_threshold(jacc, gs), get_threshold(wordOver, gs), get_threshold(semant, gs)
    print("Training Finished")
    print("Testing Started")
    test_dataset(test_data, test_gs, thresholds)
    print("Testing Finished")


main("test/msr_paraphrase_test_input.txt","test/msr_paraphrase_test_gs.txt","train/msr_paraphrase_train_input.txt", "train/msr_paraphrase_train_gs.txt")