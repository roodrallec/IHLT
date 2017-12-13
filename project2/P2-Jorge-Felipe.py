import string
from io import open

import numpy as np
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet, stopwords, wordnet_ic
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression

# init_notebook_mode(connected=True)
brown_ic = wordnet_ic.ic('ic-brown.dat')

'''
    A collection of useful functions for nlp
'''

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


# Tokens to senses (top_synset + pos)
def extract_senses(tokens):
    top_synsets = []
    for token in tokens:
        pos = token[1]
        wn_pos = wordnet_pos_code(pos)
        lemma = WordNetLemmatizer().lemmatize(token[0])
        if wn_pos:
            synsets = wordnet.synsets(lemma, pos=wn_pos)
            if len(synsets) > 0:
                top_synsets.append([synsets[0], token[0]])
            else:
                top_synsets.append([None, token[0]])
        else:
            top_synsets.append([None, token[0]])
    return top_synsets


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
    return lcs, similarity, wup_similarity, lin_similarity, lch_similarity


def count(g, s):
    TP = TN = FP = FN = 0
    for i in range(0, len(g)):
        if (g[i] == s[i] and s[i] == 1): TP += 1
        if (g[i] == s[i] and s[i] == 0): TN += 1
        if (g[i] != s[i] and s[i] == 1): FP += 1
        if (g[i] != s[i] and s[i] == 0): FN += 1
    return [TP, TN, FP, FN]


def MSRP_eval(gs, sys):
    [TP, TN, FP, FN] = count(gs, sys)
    acc = (TP + TN) / float(TP + TN + FP + FN)  # ACCURACY
    reject = TN / float(TN + FP)  # precision on negative SPECIFICITY
    accept = TP / float(TP + FN)  # precision on positive SENSITIVITY
    print("acc=", acc, " reject=", reject, " accept=", accept)


def get_word_order_vector(sentence, union, indexes, threshold):
    """
    Find the word order vector for a sentence
    """
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


def max_similarity(word, sentence):
    """
    Find the word in a sentence most similar to a word
    """
    max_sim = 0
    sim_word = ""
    # Find the first synset for the word
    synset = word[2]
    # Get the maximum similarity
    for w2 in sentence:
        similarity = word_similarity(synset, w2[2])
        if similarity and similarity > max_sim:
            max_sim = similarity
            sim_word = w2
    return sim_word, max_sim


def word_similarity(syns1, syns2):
    """
    Get the similarity between two words
    """
    if syns1 and syns2:
        return syns1.path_similarity(syns2)
    else:
        return 0


# Semantic Similarity
def get_semantic_vector(sentence, union):
    """
    Get the similarity value for every word in a set of two sentences
    """
    semantic_vector = []
    for word in union:
        if word[0] in list(zip(*sentence))[0]:
            semantic_vector.append(1)
        else:
            sim_word, max_sim = max_similarity(word, sentence)
            semantic_vector.append(max_sim)
    return np.array(semantic_vector)


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
    senses_0 = extract_senses(tokens_0)
    senses_1 = extract_senses(tokens_1)
    synsets_0 = [sense[0] for sense in senses_0]
    synsets_1 = [sense[0] for sense in senses_1]

    word_order_s0 = [(tokens_0[i][0], tokens_0[i][1], synsets_0[i]) for i in range(len(tokens_0))]
    word_order_s1 = [(tokens_1[i][0], tokens_1[i][1], synsets_1[i]) for i in range(len(tokens_1))]
    union = list(set(word_order_s0).union(set(word_order_s1)))
    indexes_0 = {tup[1]: tup[0] + 1 for tup in enumerate(word_order_s0)}
    indexes_1 = {tup[1]: tup[0] + 1 for tup in enumerate(word_order_s1)}
    # Calculate the word order vectors
    threshold = 0.4
    word_order_vector_0 = get_word_order_vector(word_order_s0, union, indexes_0, threshold)
    word_order_vector_1 = get_word_order_vector(word_order_s1, union, indexes_1, threshold)

    # Calculate the semantic vectors
    semantic_vector_0 = get_semantic_vector(word_order_s0, union)
    semantic_vector_1 = get_semantic_vector(word_order_s1, union)

    # Calculate the word order similarity
    word_order_similarity = 1.0 - (np.linalg.norm(word_order_vector_0 - word_order_vector_1) / np.linalg.norm(
        word_order_vector_0 + word_order_vector_1))

    # Calculate the semantic similarity
    semantic_similarity = np.dot(semantic_vector_0, semantic_vector_1.T) / (
            np.linalg.norm(semantic_vector_0) * np.linalg.norm(semantic_vector_1))
    return [word_order_similarity, semantic_similarity]


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
