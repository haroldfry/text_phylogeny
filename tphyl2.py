# -*- coding: iso-8859-1 -*-

from __future__ import division
from nltk.stem.wordnet import WordNetLemmatizer
import random
from random import shuffle
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
import os
import numpy as np
np.random.seed(1337)
import scipy.spatial
import en
import copy
import bz2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import hstack
from scipy.sparse import vstack
import time
import ast
from sklearn import svm
from sklearn.externals import joblib
from scipy import sparse
import unicodedata
import string
import pandas as pd
import numpy
from nltk.stem.snowball import EnglishStemmer
import subprocess
from os.path import basename
import rpy2.robjects as robjects
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
from rpy2.robjects.packages import importr

# BASIC FUNCTIONS AND CLASSES:

# Function read_file: Just a neater way to open,
# and then read the file's content


def read_file(filepath):

    filepath = os.path.normpath(filepath)

    file = open(filepath, 'r')
    content = file.read()
    file.close()

    return content

# Function read_file: Organizing the code needed to create a file
# Inputs:
# filepath: self-explanatory
# name: the name of the file (no need to add "\\")
# content: text string, or any other type.
# Output:
# The file will be written in specified path with specified name


def write_file(filepath, name, content):

    filepath = os.path.normpath(filepath)

    if not os.path.exists(filepath):
        os.makedirs(filepath)

    file = open(os.path.join(filepath, name), "wb")
    file.write(content)
    file.close()


def list_files(filepath, end=""):

    filepath = os.path.normpath(filepath)

    file_list = []

    for root, dirs, files in os.walk(filepath):
        file_list.extend(files)

    return [os.path.join(filepath, name) for name in file_list
            if name.endswith(end)]


def list_directories(filepath, end=""):

    filepath = os.path.normpath(filepath)

    # return "name" ?

    return [os.path.join(filepath, name) for name in os.listdir(filepath)
            if os.path.isdir(os.path.join(filepath, name))
            and name.endswith(end)]

# Function find_root: Finds the root of a tree.
# Input:
# topology: a topology vector, e.g.: [0,0,0,1,2]
# Output:
# root: root of the tree. In the previous example: root = 0


def find_root(topology):

    root = 0
    for i in range(len(topology)):
        if i == topology[i]:
            root = i

    return root

 # Function find_descendants: Given a node, find it's descendants.
 # Inputs:
 # topology: the full topology vector of the tree
 # node: the node we are interested in finding it's descendants
 # Output:
 # descendants: a list with the descendants of the selected node


def find_descendants(topology, node):

    descendants = []
    for i in range(len(topology)):
        if topology[i] == node and i != node:
            descendants.append(i)

    return descendants

# Class Tree: Makes it easier to manipulate - an alternative
# form of representing the Phylogeny
#
# ancestors: list with ancestors - every node that has a descendant
# descendants: list with descendants: each position represents the nodes that
# are descendants of the node in the same position in the ancestors list


class phyl_tree:

    def __init__(self, topology):

        root = find_root(topology)
        roots = []

        roots.append(root)

        self.ancestors = []
        self.descendants = {}

        while len(roots) != 0:

            desc = find_descendants(topology, roots[0])
            self.ancestors.append(roots[0])
            self.descendants[roots.pop(0)] = desc

            for d in desc:
                if d in topology:
                    roots.append(d)

# SIMULATION:

# Function write_phylogeny_file: Generate a file containing all the
# transformations and paramenters:
# E.g.: 0 1 0.5: transform from 0 to 1 with parameter 0.5
# Inputs:
# base_folder: folder where all the phylogeny files will be contained
# name: the name of the particular phylogeny file
# topology: full topology vector


def write_phylogeny_file(phyl_folder, name, topology, parameters):

    phyl_str = ""
    tree = phyl_tree(topology)

    for a in tree.ancestors:
        for d in tree.descendants[a]:

            phyl_str += '{0} ; {1} | {2}'.format(a, d,
                                                 parameter_string(parameters))

    write_file(phyl_folder, name, phyl_str)


# parameter_string:
# gets a parameter list and outputs a string
# with their comma-separated values


def parameter_string(parameters):

    param_str = ""
    param_str += "{} ; ".format(parameters["Synonym"])
    param_str += "{} ; ".format(parameters["Typo Insert"])
    param_str += "{} ; ".format(parameters["Typo Remove"])
    param_str += "{} ; ".format(parameters["Modifier Insert"])
    param_str += "{} ; ".format(parameters["Modifier Remove"])
    param_str += "{} ; ".format(parameters["Sentence Insert"])
    param_str += "{} ; ".format(parameters["Sentence Remove"])
    param_str += "{} ; ".format(parameters["Proportion of Change"])

    return param_str + " \n"


def get_parameters(param_str):

    param_str = param_str.split(";")
    parameters = {"Synonym": float(param_str[0]),
                  "typo_list Insert": float(param_str[1]),
                  "typo_list Remove": float(param_str[2]),
                  "Modifier Insert": float(param_str[3]),
                  "Modifier Remove": float(param_str[4]),
                  "Sentence Insert": float(param_str[5]),
                  "Sentence Remove": float(param_str[6]),
                  "Proportion of Change": float(sum(param_str[7]))}

    return parameters


def get_topology(phyl_file, parameters=False):

    phyl_str = read_file(phyl_file).split("\n")[0:-1]

    with open(phyl_file) as myfile:
        n_nodes = sum(1 for _ in myfile) + 1

    topology = [0] * n_nodes

    param_list = [{}] * n_nodes

    first = True

    for line in phyl_str:

        split = line.split("|")
        nodes = split[0].split(";")
        ancestor = int(nodes[0])
        descendant = int(nodes[1])

        if first:
            topology[ancestor] = ancestor
            first = False

        if parameters:
            param_list[descendant] = get_parameters(split[1])

        topology[descendant] = ancestor

    if not parameters:
        return topology
    if parameters:
        return [topology, param_list]


def build_corpus(data_file, phyl_list, hout_prop, transform, *args):

    data_file = os.path.normpath(data_file)

    original = read_file(data_file)

    tk_text = tokenized(original, hout_prop)

    topology = phyl_list[0]

    parameters = phyl_list[1]

    corpus = []

    for i in range(len(topology)):
        corpus.append(copy.deepcopy(tk_text))

    tree = phyl_tree(topology)

    name = "{0}.{1}".format(data_file.split(os.sep)[-1], len(topology))

    for ancestor in tree.ancestors:

        for descendant in tree.descendants[ancestor]:

            corpus[descendant] = transform(corpus[ancestor],
                                           parameters,
                                           name,
                                           *args)

    return corpus


def write_tree(base_folder, corpus, name):

    tree_string = ""

    for i, element in enumerate(corpus):

        tree_string += element.join_all()
        if not i == len(corpus) - 1:
            tree_string += "\n<\\tphyldoc>\n"

    write_file(base_folder, name + ".txt", tree_string)


def clean_tokenize(text, blacklist, tag=False):

    tokens = nltk.word_tokenize(text)
    new_tokens = []

    for token in tokens:

        if token.endswith(".") and not token is ".":
            new_tokens.extend([token[0:-1], "."])
        else:
            new_tokens.append(token)

    filtered_token = [t for t in new_tokens if not t in blacklist]

    if tag:
        filtered_token = nltk.pos_tag(filtered_token)

    return filtered_token


class tokenized:

    def __init__(self, text, hout_prop):

        self.hout_prop = hout_prop
        self.tokens = {}
        self.held_out = {}

        sentences = nltk.sent_tokenize(text)

        blacklist = ["``", "''"]

        for i, sentence in enumerate(sentences):

            tagged_list = [{"word": tk[0], "tag": tk[1]}
                           for tk in clean_tokenize(sentence,
                                                    blacklist, tag=True)]

            self.tokens[i] = tagged_list

        indexes = random.sample(range(len(self.tokens)),
                                int(round(len(self.tokens) * hout_prop)))
        for i in indexes:
            self.held_out[i] = self.tokens.pop(i)

    def join_sentences(self):

        sentences = []

        for element in self.tokens:
            sentences.append(self.tokens[element])

        word_list = [word["word"] for sentence
                     in sentences for word in sentence]

        return word_list

    def join_all(self):

        sentences = []

        for element in self.tokens:
            sentences.append(self.tokens[element])

        word_list = [word["word"] for sentence
                     in sentences for word in sentence]

        text = " ".join(word_list)

        return text


def generate_topology(n_topologies, n_nodes, rd=True):

    topologies = []

    for i in range(n_topologies):
        topologies.append(randomized_tree(n_nodes, rd))

    return topologies


def randomized_tree(n, rd, rn=None):

    tree = range(0, n)
    order = range(0, n)

    if rn is not None and rn > 0 and rn < len(tree):
        root = order.pop(rn)
        if rd:
            random.shuffle(order)
    else:
        if rd:
            random.shuffle(order)
        root = order.pop(0)

    added = [root]

    while order != []:
        node = order.pop(0)
        tree[node] = random.choice(added)
        added.append(node)

    return tree


# FUNCTIONS FOR TEXT TRANSFORMATION ##########################################


def operation_pick(parameters):

    params = copy.deepcopy(parameters)

    params.pop("Proportion of Change")

    return params.keys()[weighted_pick(params.values())]


def token_pick(tokens, indexes):

    weights = []

    for element in indexes.values():

        weights.append(sum(element))

    if sum(weights) == 0:
        return False

    sentence = tokens.tokens.keys()[weighted_pick(weights)]

    eligible = [i for i, element in enumerate(
        indexes[sentence]) if element == 1]

    if len(eligible) < 1:
        return False

    token = random.sample(eligible, 1)[0]

    return [sentence, token]


def transform(input_tokens, parameters, text_id, typos, mods):

    tokens = copy.deepcopy(input_tokens)

    indexes = {}
    typo_indexes = {}
    modins_indexes = {}
    modrem_indexes = {}

    for element in tokens.tokens.keys():

        indexes[element] = [1] * len(tokens.tokens[element])
        typo_indexes[element] = [0] * len(tokens.tokens[element])
        modins_indexes[element] = [0] * len(tokens.tokens[element])
        modrem_indexes[element] = [0] * len(tokens.tokens[element])

        for i, token in enumerate(tokens.tokens[element]):

            if "TYPO" in token["tag"].split("|"):

                indexes[element][i] = 0
                typo_indexes[element][i] = 1

            if token["tag"].startswith("V") or token["tag"].startswith("N"):

                modins_indexes[element][i] = 1

            if token["tag"].startswith("R") or token["tag"].startswith("J"):

                modrem_indexes[element][i] = 1

    edit_count = 0

    size = len(tokens.join_sentences())

    prop = random.sample(parameters["Proportion of Change"], 1)[0]

    edit_limit = int(prop * size)

    iter_limit = 1000

    prev_count = 0

    iter_count = 0

    while(True):

        if edit_count == prev_count:
            iter_count += 1
            if iter_count > iter_limit:
                print "ERROR - Iteration limit reached in tree: {}".format(text_id)
                break
        else:
            iter_count = 0
            prev_count = edit_count

        op = operation_pick(parameters)

        # print "limit: {0} current: {1} op: {2}".format(edit_limit,
        # edit_count, op)

        if edit_count >= edit_limit:
            break

        if op == "Synonym":

            count = 5

            while(count):

                tk = token_pick(tokens, indexes)

                if not tk:
                    break

                syn = get_synonym(tokens.tokens[tk[0]][tk[1]])

                if syn:

                    tokens.tokens[tk[0]][tk[1]]["word"] = syn
                    indexes[tk[0]][tk[1]] = 0
                    edit_count += 1
                    break

                count -= 1

        elif op == "Typo Insert":

            count = 5

            while(count):

                tk = token_pick(tokens, indexes)

                if not tk:
                    break

                orig_word = tokens.tokens[tk[0]][tk[1]]["word"]

                orig_tag = tokens.tokens[tk[0]][tk[1]]["tag"]

                typo = typos.get(orig_word)

                if typo:

                    tokens.tokens[tk[0]][tk[1]]["word"] = typo
                    tokens.tokens[tk[0]][tk[1]]["tag"] = "{0}|{1}|{2}".format(
                        "TYPO", orig_tag, orig_word)
                    indexes[tk[0]][tk[1]] = 0
                    typo_indexes[tk[0]][tk[1]] = 1
                    modins_indexes[tk[0]][tk[1]] = 0
                    modrem_indexes[tk[0]][tk[1]] = 0

                    edit_count += 1
                    break

                count -= 1

        elif op == "Typo Remove":

            count = 5

            while(count):

                tk = token_pick(tokens, typo_indexes)

                if not tk:
                    break

                split = tokens.tokens[tk[0]][tk[1]]["tag"].split("|")

                if "TYPO" in split:

                    tokens.tokens[tk[0]][tk[1]]["word"] = split[2]
                    tokens.tokens[tk[0]][tk[1]]["tag"] = split[1]
                    typo_indexes[tk[0]][tk[1]] = 0
                    edit_count += 1
                    break

                count -= 1

        elif op == "Modifier Insert":

            count = 5

            while(count):

                tk = token_pick(tokens, modins_indexes)

                if not tk:
                    break

                mod = mods.getmod(tokens.tokens[tk[0]][tk[1]])

                if mod:

                    tokens.tokens[tk[0]].insert(tk[1], mod)

                    indexes[tk[0]].insert(tk[1], 0)
                    typo_indexes[tk[0]].insert(tk[1], 0)
                    modins_indexes[tk[0]].insert(tk[1], 0)
                    modrem_indexes[tk[0]].insert(tk[1], 1)
                    edit_count += 1
                    break

                count -= 1

        elif op == "Modifier Remove":

            while(True):

                tk = token_pick(tokens, modrem_indexes)

                if not tk:
                    break

                tokens.tokens[tk[0]].pop(tk[1])

                indexes[tk[0]].pop(tk[1])
                typo_indexes[tk[0]].pop(tk[1])
                modins_indexes[tk[0]].pop(tk[1])
                modrem_indexes[tk[0]].pop(tk[1])
                edit_count += 1
                break

        elif op == "Sentence Insert":

            if len(tokens.held_out) > 0:

                ind = random.sample(tokens.held_out.keys(), 1)[0]

                sentence = tokens.held_out[ind]

                if not edit_count + len(sentence) >= edit_limit:

                    tokens.tokens[ind] = sentence

                    tokens.held_out.pop(ind)

                    edit_count += len(sentence)

                    indexes[ind] = [1] * len(sentence)
                    typo_indexes[ind] = [0] * len(sentence)
                    modins_indexes[ind] = [0] * len(sentence)
                    modrem_indexes[ind] = [0] * len(sentence)

                    for i, token in enumerate(tokens.tokens[ind]):

                        if "TYPO" in token["tag"].split("|"):

                            indexes[ind][i] = 0
                            typo_indexes[ind][i] = 1

                        if token["tag"].startswith("V") or token["tag"].startswith("N"):

                            modins_indexes[ind][i] = 1

                        if token["tag"].startswith("R") or token["tag"].startswith("J"):

                            modrem_indexes[ind][i] = 1

        elif op == "Sentence Remove":

            if len(tokens.tokens) > 1:

                ind = random.sample(tokens.tokens.keys(), 1)[0]

                if not edit_count + len(tokens.tokens[ind]) >= edit_limit:

                    edit_count += len(tokens.tokens[ind])

                    tokens.tokens.pop(ind)

                    indexes.pop(ind)
                    typo_indexes.pop(ind)
                    modins_indexes.pop(ind)
                    modrem_indexes.pop(ind)

    return tokens


def weighted_pick(weights):

    rnd = random.uniform(0, 1) * sum(weights)

    for i, w in enumerate(weights):
        rnd -= w
        if rnd < 0:
            return i


def wordnet_tag(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ

    elif treebank_tag.startswith('V'):
        return wordnet.VERB

    elif treebank_tag.startswith('N'):
        return wordnet.NOUN

    elif treebank_tag.startswith('R'):
        return wordnet.ADV

    else:
        return ''


def preserve_case(word, subst):
    if word.istitle():
        return subst.title()
    else:
        return subst


def format_word(word, subst, pos_tag):

    try:
        if pos_tag.startswith("V") and en.verb.infinitive(word) == '':
            return word

        if pos_tag.startswith("V") and en.verb.infinitive(subst) == '':
            return word

        if pos_tag.startswith("V") and not en.verb.infinitive(subst) == '':

            word_tense = en.verb.tense(word)
            tenses = en.verb.tenses()

            for tense in tenses:
                if tense == word_tense:
                    subst = en.verb.conjugate(subst, tense=tense)

        elif pos_tag.startswith("N"):

            if word == en.noun.singular(word):
                subst = en.noun.singular(subst)

            elif subst == en.noun.singular(subst):
                subst = en.noun.plural(subst)

        elif pos_tag.startswith("J"):

            if word == en.adjective.plural(word):
                subst = en.adjective.plural(subst)

        subst = preserve_case(word, subst)

        if not subst == "":
            return subst
        else:
            return word

    except NameError:
        return subst

    except RuntimeError:
        return subst


def get_synonym(tagged_word):

    word = tagged_word["word"]
    pos_tag = tagged_word["tag"]

    candidates = []
    word_lemma = format_word("pass", word, pos_tag)
    synsets = wordnet.synsets(word_lemma, wordnet_tag(pos_tag))
    lmtzr = WordNetLemmatizer()

    for element in synsets:

        if (lmtzr.lemmatize(word_lemma) in element.name().split(".")[0] or
                element.name().split(".")[0] in lmtzr.lemmatize(word_lemma)):

            for item in element.lemma_names():
                subst = item.lower()

                if (not lmtzr.lemmatize(subst) == lmtzr.lemmatize(word_lemma)
                        and "_" not in subst):

                    candidates.append(format_word(word, subst, pos_tag))

    if len(candidates) > 0:
        return random.sample(list(set(candidates)), 1)[0]
    else:
        return False


class typo_list:

    def __init__(self, filename):

        data = read_file(filename).split(" \n")[0:-1]
        self.typo_list = {}

        for element in data:

            temp = element.split(" | ")
            self.typo_list[temp[0]] = temp[1].split(" ; ")

    def search(self, word):

        if word.lower() in self.typo_list.keys():
            return self.typo_list[word.lower()]
        else:
            return ""

    def get(self, word):

        if word.lower() in self.typo_list.keys():
            r = self.typo_list[word.lower()]
            return random.sample(r, 1)[0]
        else:
            return False


def mod_classify(word_tag):

    if word_tag.startswith("V"):
        return "RR"
    elif word_tag.startswith("N"):
        return "JJ"
    else:
        return word_tag


class modifier_list:

    def __init__(self, filepath):

        data = {}

        data["N"] = read_file(list_files(filepath,
                                         "nouns.txt")[0]).split("\n")[0:-1]
        data["V"] = read_file(list_files(filepath,
                                         "verbs.txt")[0]).split("\n")[0:-1]

        self.mods_list = {}
        self.mods_list["N"] = {}
        self.mods_list["V"] = {}

        types = ["N", "V"]

        for t in types:

            for line in data[t]:

                set1 = line.split(":")
                self.mods_list[t][set1[0]] = []

                for element in set1[1].split("|")[0:-1]:

                    set2 = element.split(";")
                    self.mods_list[t][set1[0]].append([int(set2[0]), set2[1]])

    def getmod(self, tagged_word):

        searchval = tagged_word["word"]
        word_tag = tagged_word["tag"][0]

        mod_tag = mod_classify(tagged_word["tag"])

        if searchval in self.mods_list[word_tag].keys():
            candidates = self.mods_list[word_tag][searchval]
        else:
            candidates = []

        weights = [element[0] for element in candidates]

        if len(candidates) > 0:
            return wobject(candidates[weighted_pick(weights)][1], mod_tag)
        else:
            return False


def wobject(word, tag):

    return {"word": word, "tag": tag}


def random_picks(prob, size):

    return list(np.random.binomial(1, prob, size))

# FUNCTIONS FOR RECONSTRUCTION ###########################################


def editdistance(a, b):

    m = len(a)
    n = len(b)

    if m >= n:
        a, b = b, a
        m, n = n, m

    offset = m + 1
    delta = n - m
    size = m + n + 3
    fp = [-1 for idx in range(size)]
    p = -1

    while (True):

        p = p + 1

        for k in range(-p, delta, 1):
            fp[k + offset] = snake(a, b, m, n, k, fp[k - 1 + offset] + 1,
                                   fp[k + 1 + offset])

        for k in range(delta + p, delta, -1):
            fp[k + offset] = snake(a, b, m, n, k, fp[k - 1 + offset] + 1,
                                   fp[k + 1 + offset])

        fp[delta + offset] = snake(a, b, m, n, delta,
                                   fp[delta - 1 + offset] + 1,
                                   fp[delta + 1 + offset])

        if fp[delta + offset] >= n:
            break

    return delta + 2 * p


def snake(a, b, m, n, k, p, pp):

    y = max(p, pp)
    x = y - k

    while x < m and y < n and a[x] == b[y]:
        x = x + 1
        y = y + 1

    return y


def ncd_bz2(x, y):

    xbytes = " ".join(x)
    ybytes = " ".join(y)
    xybytes = xbytes + ybytes
    cx = bz2.compress(xbytes)
    cy = bz2.compress(ybytes)
    cxy = bz2.compress(xybytes)
    if len(cy) > len(cx):
        n = (len(cxy) - len(cx)) / float(len(cy))
    else:
        n = (len(cxy) - len(cy)) / float(len(cx))
    return n


def my_tokenizer(s):
    return nltk.word_tokenize(s)


def my_preprocessor(s):
    return s


def tf_idf_dismat(tree_file, matrix_folder, method_name, char_ngrams, word_ngrams, char, word):

    # guess: n files
    corpus = read_file(tree_file).split("<\\tphyldoc>")

    # Convert a collection of text documents to a matrix of token counts
    word_vectorizer = CountVectorizer(ngram_range=(1, word_ngrams),
                                      preprocessor = my_preprocessor,
                                      tokenizer = my_tokenizer,
                                      decode_error="replace",
                                      analyzer="word")

    char_vectorizer = CountVectorizer(ngram_range=(1, char_ngrams),
                                      preprocessor = my_preprocessor,
                                      tokenizer = my_tokenizer,
                                      decode_error="replace",
                                      analyzer="char")
    # Transform a count matrix to a normalized tf or tf-idf representation
    tfidf = TfidfTransformer(norm="l2")

    if char and word:
        word_matrix = word_vectorizer.fit_transform(corpus)
        char_matrix = char_vectorizer.fit_transform(corpus)
        # Stack arrays in sequence horizontally (column wise).
        freq_term_matrix = hstack([word_matrix, char_matrix])

    elif char:
        freq_term_matrix = char_vectorizer.fit_transform(corpus)

    elif word:
        freq_term_matrix = word_vectorizer.fit_transform(corpus)

    print "freq_term_matrix's shape is :", freq_term_matrix.shape

    tf_idf_matrix = tfidf.fit_transform(freq_term_matrix)
    print "tfidf matrix's shape is : ", tf_idf_matrix.shape

    # abs(): Return the absolute value of a number
    dis_matrix = abs(
        (tf_idf_matrix * np.transpose(tf_idf_matrix)).todense() - 1)

    dis_matrix[dis_matrix < 1e-10] = 0

    # np.array(): Create an array.
    dis_matrix = np.array(dis_matrix)

    filename = tree_file.split(os.sep)[-1].replace(".txt", "") + ".dismat"

    write_matrix(
        dis_matrix, os.path.join(matrix_folder, method_name), filename)

    print "dis_matrix.shape is :"
    print dis_matrix.shape
    return dis_matrix


def dissimilarity_tf_idf(base_file, method_name="tf-idf",
                         char_ngrams=1, word_ngrams=1, char=False, word=True):

    tree_file = os.path.join(base_file, "Trees")
    mat_file = os.path.join(base_file, "DisMatrices")

    print "Computing dissimilarity matrices:"

    tree_files = list_files(tree_file, "")

    if not os.path.exists(mat_file):
        os.makedirs(mat_file)

    mat_files = list_files(os.path.join(mat_file, method_name), "")

    mat_files = [
        element.split(os.sep)[-1].replace(".dismat", "") for element in mat_files]

    count = 0

    start_time = time.time()

    for tree in tree_files:
        print tree
        name = tree.split(os.sep)[-1].replace(".txt", "")

        if not name in mat_files:
            tf_idf_dismat(tree, mat_file, method_name, char_ngrams, word_ngrams, char, word)

        count += 1
        print "{0} out of {1} matrices completed...".format(count, len(tree_files))

    elapsed_time = time.time() - start_time
    time_string = "{}s | {} matrices".format(elapsed_time, len(tree_files))
    write_file(base_file, "time_{}.txt".format(method_name), time_string)


def dismatrix_41(base_file, method_name="41_features"):
    tree_file = os.path.join(base_file, "Trees")
    mat_file = os.path.join(base_file, "DisMatrices")

    print "Computing dissimilarity matrices using new 41 features:"

    tree_files = list_files(tree_file, "")

    if not os.path.exists(mat_file):
        os.makedirs(mat_file)

    mat_files = list_files(os.path.join(mat_file, method_name), "")

    mat_files = [
        element.split(os.sep)[-1].replace(".dismat", "") for element in mat_files]

    count = 0

    start_time = time.time()

    for tree in tree_files:

        name = tree.split(os.sep)[-1].replace(".txt", "")

        if not name in mat_files:
            dissimilarity_41(tree, mat_file, method_name)

        count += 1
        print "{0} out of {1} matrices completed...".format(count, len(tree_files))

    elapsed_time = time.time() - start_time
    time_string = "{}s | {} matrices".format(elapsed_time, len(tree_files))
    write_file(base_file, "time_{}.txt".format(method_name), time_string)


# given a tree file which contains a set of near-duplicated documents, use 17 features to generate dissimilarity matrix
def dissimilarity_41(tree_file, matrix_folder, method_name):
    # split tree file into individual texts
    corpus = read_file(tree_file).split("<\\tphyldoc>")

    nonstemmed_corpus = []

    print "os.path(tree_file): ", tree_file

    for file in corpus:

        # turn all texts into lowercase
        file = file.lower()

        # remove all punctuations
        file = file.translate(None, string.punctuation)
        file += "\n"
        # get new corpus ready to use: nonstemmed_corpus
        nonstemmed_corpus.append(file)

    corpus_number = len(nonstemmed_corpus)
    dis_matrix = [[0 for x in range(corpus_number)] for y in range(corpus_number)]
    corpus_index_list1 = np.arange(corpus_number)

    path = os.path.dirname(os.path.abspath(__file__))
    path_index = os.path.splitext(os.path.basename(tree_file))[0]
    nonstemmed_corpus_path = os.path.join(path, "generated_corpus_{}".format(path_index))

    if not os.path.exists(nonstemmed_corpus_path):
        os.makedirs(nonstemmed_corpus_path)

    for element in nonstemmed_corpus:
        write_file(nonstemmed_corpus_path, "nonstemmed_{}.txt".format(nonstemmed_corpus.index(element)), element)

    # calling R scripts to load corpus
    outputDir = os.path.join(path, "processed_corpus_{}".format(path_index))
    outputDir += "/"
    inputDir = nonstemmed_corpus_path

    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    # stem the corpus generate processed corpus
    with open('longworkStemBatchEn.R') as fh:
        rcode = os.linesep.join(fh.readlines())
        myfunc_stem = SignatureTranslatedAnonymousPackage(rcode, "myfunc_stem")
        myfunc_stem.longworkStemBatchEn(inputDir, outputDir)

    # load the corpus
    with open('loadCorpus.R') as lh:
        rcode = os.linesep.join(lh.readlines())
        myfunc_load = SignatureTranslatedAnonymousPackage(rcode, "myfunc_load")
        r_corpus = myfunc_load.loadCorpus(outputDir)

    # get two files as sample1 and sample2 to calculate vec using attribution features
    for file1_id in corpus_index_list1:
        file1 = nonstemmed_corpus[file1_id]
        # generate a full list of corpus
        corpus_index_list2 = np.arange(corpus_number)
        corpus_index_list2 = corpus_index_list2.tolist()
        # remove file1's index from the list
        corpus_index_list2.remove(file1_id)
        dis_matrix[file1_id][file1_id] = 0
        for file2_id in corpus_index_list2:
            file2 = nonstemmed_corpus[file2_id]
            # run genFeatures.R
            with open('genFeatures.R') as gh:
                importr("gtools")
                rcode = os.linesep.join(gh.readlines())
                myfunc_genf = SignatureTranslatedAnonymousPackage(rcode, "myfunc_genf")

                # vec is the resulting vector generated by genFeatures.R
                vec = myfunc_genf.genFeatures(file1, file2, r_corpus, 1)

                # manipulating vec to generate wanted value
                file12_array = numpy.array(vec)
                str12 = ''.join(str(e) for e in file12_array)
                testlist = str12.split()
                del testlist[0]
                newlist = []
                for element in testlist:
                    newele = element.split(":", 1)[-1]
                    if newele.find('.'):
                        float(newele)
                        newlist.append(newele)
                    else:
                        int(newele)
                        newlist.append(newele)

                newarray = np.asarray(newlist, dtype=float)
                value = np.dot(newarray, np.transpose(newarray))
                dis_matrix[file1_id][file2_id] = value
                dis_matrix[file2_id][file1_id] = value
                print "dis_matrix", file1_id, file2_id, "=", value

    # get dissimilarity matrix
    # abs(): Return the absolute value of a number
    # dis_matrix = abs(
    #     (tf_idf_matrix * np.transpose(tf_idf_matrix)).todense() - 1)
    #
    # dis_matrix[dis_matrix < 1e-10] = 0
    #
    # np.array(): Create an array.
    # dis_matrix = np.array(dis_matrix)
    #
    filename = tree_file.split(os.sep)[-1].replace(".txt", "") + ".dismat"
    print filename
    write_matrix(
        dis_matrix, os.path.join(matrix_folder, method_name), filename)

    return dis_matrix


# using pre-trained glove embeddings to generate vectors
def glove_vectors(base_file):
    GLOVE_DIR = '/glove'
    embedding_dir = os.path.join(base_file, "Embedding_vectors")
    if not os.path.exists(embedding_dir):
        os.makedirs(embedding_dir)

    MAX_NB_WORDS = 20000
    EMBEDDING_DIM = 300

    # get glove word embeddings
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    # processing texts
    trees_path = os.path.join(base_file, "Trees")

    for tree in os.listdir(trees_path):
        print (tree)
        treepath = os.path.join(trees_path, tree)
        treefiles = open(treepath, 'r').read().split("<\\tphyldoc>")

        matrix_dim = len(treefiles)
        whole_embedding_vec = np.zeros((matrix_dim, EMBEDDING_DIM))

        writingname = tree.split('.txt')[0] + '_embed'
        writingname = writingname + '.txt'
        writingpath = os.path.join(embedding_dir, writingname)

        open(writingpath, 'w').close()
        output = open(writingpath, 'a')

        for treenode in treefiles:
            treenodedic = {}

            treewords = treenode.split(' ')
            for item in treewords:
                treenodedic[treewords.index(item)] = item

            nb_words = min(MAX_NB_WORDS, len(treenode))
            print ('nb_words: ', nb_words)

            embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
            for i, word in treenodedic.items():
                if i > MAX_NB_WORDS:
                    continue
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    # words not found in embedding index will be all-zeros.
                    embedding_matrix[i] = embedding_vector

            print ('embedding_matrix.shape: ', embedding_matrix.shape)

            tree_embedding_vector = np.mean(embedding_matrix, axis=0)
            whole_embedding_vec[treefiles.index(treenode)] = (tree_embedding_vector)
            output.write(str(tree_embedding_vector))
            output.write("\n")
        #
        # output.close()

        print ('whole_embedding_vec.shape: ', whole_embedding_vec.shape)
        whole_embedding_matrix = np.zeros((matrix_dim, matrix_dim))

        for i in range(0, matrix_dim):
            for j in range(0, matrix_dim):
                # print ("i: ", i)
                # print ("j: ", j)
                vector1 = whole_embedding_vec[i]
                vector2 = whole_embedding_vec[j]
                # whole_embedding_matrix[i][j] = scipy.spatial.distance.cosine(vector1, vector2)
                whole_embedding_matrix[i][j] = scipy.spatial.distance.euclidean(vector1, vector2)

        # print (whole_embedding_matrix)

        matrix_dir = os.path.join(base_file, "DisMatrices")
        # glove_dismatrix(embedding_dir, matrix_dir, method_name="glove_embedding")
        filename = tree.split(os.sep)[-1].replace(".txt", "") + ".dismat"
        print filename
        write_matrix(
            whole_embedding_matrix, os.path.join(matrix_dir, 'glove_embed'), filename)


def compute_matrix(tree_file, matrix_folder, method_name, dis_func, editdist_norm=False, filtering=None):

    tree_string = read_file(tree_file)

    corpus = tree_string.decode("utf-8", errors="replace")

    corpus = unicodedata.normalize('NFKD', corpus).encode("ascii", "ignore")

    corpus = tree_string.split("<\\tphyldoc>")

    for i, element in enumerate(corpus):

        corpus[i] = nltk.word_tokenize(element)

        if filtering == "stopwords":
            corpus[i] = [element for element in corpus[i]
                         if element in stopwords.words("english")]
        if filtering == "not_stopwords":
            corpus[i] = [element for element in corpus[i]
                         if not element in stopwords.words("english")]

    matrix = np.zeros((len(corpus), len(corpus)))

    if editdist_norm:
        norm_matrix = np.zeros((len(corpus), len(corpus)))

    fixed_range = range(len(corpus) - 1)
    dyn_range = range(len(corpus) - 1, 0, -1)

    for i1 in fixed_range:

        for i2 in dyn_range:

            dist = dis_func(corpus[i1], corpus[i2])

            matrix[i1][i2] = dist
            matrix[i2][i1] = dist

            if editdist_norm:
                norm_matrix[i1][i2] = dist / \
                    max([len(corpus[i1]), len(corpus[i2])])
                norm_matrix[i2][i1] = dist / \
                    max([len(corpus[i1]), len(corpus[i2])])

        dyn_range.pop(-1)

    filename = tree_file.split(os.sep)[-1].replace(".txt", "") + ".dismat"

    # print matrix

    write_matrix(matrix, os.path.join(matrix_folder, method_name), filename)

    if editdist_norm:
        write_matrix(norm_matrix, os.path.join(
            matrix_folder, method_name + "_normalized"), filename)

    return matrix


def write_matrix(matrix, matrix_folder, filename):

    matrix = list(matrix)

    matrix_string = ""

    for line in matrix:

        for element in line:

            matrix_string += "{}; ".format(element)

        matrix_string += "\n"

    matrix_string = matrix_string.replace("; \n", " \n")

    write_file(matrix_folder, filename, matrix_string)


def kruskal(matrix_file):

    dis = np.loadtxt(matrix_file, delimiter=';')

    n = len(dis)

    tree = []
    edges = []
    for i in range(n):
        for j in range(n):
            if not (i == j or j < i):
                edges.append([dis[i][j], i, j])

    edges.sort()
    trees = [[i] for i in range(n)]

    for edge in edges:
        ok = True
        for tr in trees:
            if (edge[1] in tr and edge[2] in tr):
                ok = False
        if ok:
            count = 0
            e1 = 0
            e2 = 0
            for tr in trees:
                if edge[1] in tr:
                    e1 = count
                if edge[2] in tr:
                    e2 = count
                count += 1

            tree.append([edge[1], edge[2]])
            trees[e1].extend(trees[e2])
            trees.pop(e2)

        if len(tree) == n - 1:
            break
    tree.sort()

    return [tree, dis]


def get_edges(topology):

    edges = []

    for i, element in enumerate(topology):

        edges.append([element, i])

    return edges


def build_topology(root, edg):

    edges = copy.deepcopy(edg)

    topology = [0] * (len(edges) + 1)

    topology[root] = root
    roots = [root]

    while len(roots) != 0:

        desc = []
        p = []
        count = 0

        for element in edges:

            if roots[0] in element:
                element.pop(element.index(roots[0]))
                desc.append(element[0])
                topology[element[0]] = roots[0]
                p.append(count)

            count += 1
        for i in range(len(p)):
            edges.pop(p[i] - i)

        for d in desc:
            for element in edges:

                if (d in element and d not in roots):
                    roots.append(d)

        roots.pop(0)

    return topology


def random_root(tree):

    root = random.sample(range(len(tree) + 1), 1)[0]

    return build_topology(root, tree)


def least_cost_root(tree, dis, cost):

    n = len(dis)

    cost_list = []

    for node in range(n):
        topology = build_topology(node, tree)
        cost_list.append(cost(topology, dis))

    root = cost_list.index(min(cost_list))

    return build_topology(root, tree)


def max_cost_root(tree, dis, cost):

    n = len(dis)

    cost_list = []

    for node in range(n):
        topology = build_topology(node, tree)
        cost_list.append(cost(topology, dis))

    root = cost_list.index(max(cost_list))

    return build_topology(root, tree)


def leaves_cost(topology, dis):

    root = find_root(topology)

    nodes = find_leaves(topology)

    cost = 0

    for node in nodes:
        while node != root:
            cost += dis[node][topology[node]]
            node = topology[node]

    return cost


def tree_cost(topology, dis):

    root = find_root(topology)

    nodes = range(len(topology))
    nodes.pop(root)

    cost = 0

    for node in nodes:
        while node != root:
            cost += dis[node][topology[node]]
            node = topology[node]

    return cost


def get_sizes(base_folder):

    tree_folder = os.path.join(base_folder, "Trees")

    dest_folder = os.path.join(base_folder, "tree-sizes")

    trees = list_files(tree_folder)

    for tree_file in trees:

        tree_contents = read_file(tree_file).split("\n<\\tphyldoc>\n")

        sizes = [len(nltk.word_tokenize(t)) for t in tree_contents]

        name = tree_file.split(os.sep)[-1]

        write_file(dest_folder, name, str(sizes))


def least_size_alg(matrix_file):

    [tree, dis] = kruskal(matrix_file)

    name = matrix_file.split(os.sep)[-1].replace(".dismat", "")

    base_file = "{}".format(os.sep).join(matrix_file.split(os.sep)[0:-3])

    size_file = read_file(os.path.join(base_file, "tree-sizes", name + ".txt"))

    sizes = ast.literal_eval(size_file)

    min_size = min(sizes)

    root = random.sample([i for i, s in enumerate(sizes) if s == min_size], 1)[0]

    return build_topology(root, tree)


def least_size_binary_cost(matrix_file):

    [tree, dis] = kruskal(matrix_file)

    name = matrix_file.split(os.sep)[-1].replace(".dismat", "")

    base_file = "{}".format(os.sep).join(matrix_file.split(os.sep)[0:-3])

    size_file = read_file(os.path.join(base_file, "tree-sizes", name + ".txt"))

    sizes = ast.literal_eval(size_file)

    n = len(dis)

    cost_list = []

    for node in range(n):
        topology = build_topology(node, tree)
        cost_list.append(size_binary_cost(topology, sizes))

    max_cost = max(cost_list)

    root = random.sample([i for i, c in enumerate(cost_list) if c == max_cost], 1)[0]

    return build_topology(root, tree)


def size_binary_cost(topology, sizes):

    root = find_root(topology)

    cost = 0

    for i, node in enumerate(topology):
        if sizes[topology[i]] < sizes[i]:
            cost += 1

    return cost


def least_size_cost(matrix_file):

    [tree, dis] = kruskal(matrix_file)

    name = matrix_file.split(os.sep)[-1].replace(".dismat", "")

    base_file = "{}".format(os.sep).join(matrix_file.split(os.sep)[0:-3])

    size_file = read_file(os.path.join(base_file, "tree-sizes", name + ".txt"))

    sizes = ast.literal_eval(size_file)

    n = len(dis)

    cost_list = []

    for node in range(n):
        topology = build_topology(node, tree)
        cost_list.append(size_cost(topology, sizes))

    max_cost = max(cost_list)

    root = random.sample([i for i, c in enumerate(cost_list) if c == max_cost], 1)[0]

    return build_topology(root, tree)


def max_size_cost(matrix_file):

    [tree, dis] = kruskal(matrix_file)

    name = matrix_file.split(os.sep)[-1].replace(".dismat", "")

    base_file = "{}".format(os.sep).join(matrix_file.split(os.sep)[0:-3])

    size_file = read_file(os.path.join(base_file, "tree-sizes", name + ".txt"))

    sizes = - np.array(ast.literal_eval(size_file))

    n = len(dis)

    cost_list = []

    for node in range(n):
        topology = build_topology(node, tree)
        cost_list.append(size_cost(topology, sizes))

    max_cost = max(cost_list)

    root = random.sample([i for i, c in enumerate(cost_list) if c == max_cost], 1)[0]

    return build_topology(root, tree)


def size_cost(topology, sizes):

    root = find_root(topology)

    cost = 0

    for i, node in enumerate(topology):
            cost += sizes[i] - sizes[topology[i]]

    return cost

# FUNCTIONS FOR EVALUATION ###############################################


def dir_edges(original_topology, reconstructed_topology):

    ok = 0

    for i, element in enumerate(reconstructed_topology):
        if not element == i:
            if element == original_topology[i]:
                ok += 1

    return ok / (len(original_topology) - 1)


def ind_edges(original_topology, reconstructed_topology):

    ok = 0

    for i, element in enumerate(reconstructed_topology):
        if not element == i:
            if any([element == original_topology[i],
                    i == original_topology[element]]):
                ok += 1

    return ok / (len(original_topology) - 1)


def find_leaves(topology):

    size = len(topology)

    nodes = set(range(size))

    leaves = nodes - set(topology)

    return list(leaves)


def leaves(original_topology, reconstructed_topology):

    orig_leaves = find_leaves(original_topology)

    recon_leaves = find_leaves(reconstructed_topology)

    intersection = set(orig_leaves).intersection(set(recon_leaves))
    union = set(orig_leaves + recon_leaves)

    return len(intersection) / len(union)


def n_depth(original_topology, reconstructed_topology, n):

    orig_root = find_root(original_topology)

    nodes = n_depth_set(reconstructed_topology, n)

    if orig_root in nodes:
        return 1
    else:
        return 0


def find_all_descendants(topology, nodes):

    descendants = []
    descendants.extend(nodes)

    for node in nodes:
        for i in range(len(topology)):
            if topology[i] == node and i != node:
                descendants.append(i)

    return descendants


def n_depth_set(topology, N):

    if N == 0:
        return [find_root(topology)]
    else:
        return list(set(find_all_descendants(topology,
                                             n_depth_set(topology, N - 1))))


def depth(original_topology, reconstructed_topology):

    dist = 0

    while n_depth(original_topology, reconstructed_topology, dist) != 1:

        dist += 1

    return dist


def ancestry(original_topology, reconstructed_topology):

    a1 = find_ancestry(original_topology)
    a2 = find_ancestry(reconstructed_topology)

    intersection = set(a1).intersection(set(a2))
    union = set(a1).union(set(a2))

    return len(intersection) / len(union)


def find_ancestry(topology):

    root = find_root(topology)
    ancest = []
    for i, element in enumerate(topology):
        if not i == root:
            search = i
            while True:
                ancest.append((i + 1, topology[search] + 1))
                search = topology[search]
                if search == root:
                    break

    return ancest


def gtruth_root_alg(matrix_file):

    [tree, dis] = kruskal(matrix_file)

    name = matrix_file.split(os.sep)[-1].replace(".dismat", "")

    base_file = "{}".format(os.sep).join(matrix_file.split(os.sep)[0:-3])

    root = find_root(get_topology(os.path.join(base_file, "Phylogenies",
                                               name + ".phyl")))

    return build_topology(root, tree)


def random_root_alg(matrix_file):

    [tree, dis] = kruskal(matrix_file)

    return random_root(tree)


def least_leaves_cost_alg(matrix_file):

    [tree, dis] = kruskal(matrix_file)

    return least_cost_root(tree, dis, leaves_cost)


def least_tree_cost_alg(matrix_file):

    [tree, dis] = kruskal(matrix_file)

    return least_cost_root(tree, dis, tree_cost)


def max_tree_cost_alg(matrix_file):

    [tree, dis] = kruskal(matrix_file)

    return max_cost_root(tree, dis, tree_cost)


def write_results(res_list, filepath, name):

    name_list = ["Tree Size", "Ind. Edges", "Direct Edges",
                 "Leaves", "Roots", "Ancestry", "Depth", "1-Depth",
                 "2-Depth", "3-Depth", "1-Set", "2-Set", "3-Set"]

    res_string = "{0[0]:^15}\t{0[1]:^15}\t{0[2]:^15}\t{0[3]:^15}\t{0[4]:^15}\t{0[5]:^15}\t{0[6]:^15}\t{0[7]:^15}\t{0[8]:^15}\t{0[9]:^15}\t{0[10]:^15}\t{0[11]:^15}\t{0[12]:^15}\n".format(
        name_list)

    keys = sorted(res_list.keys())

    for key in keys:

        if len(res_list[key]) != 0:

            n_elements = len(res_list[key])

            aux_list = [0] * n_elements

            res_string += "{0:^15}\t".format(key)

            aux_list = np.array(res_list[key]).sum(axis=0) / n_elements

            for element in aux_list:

                temp_str = "{0:.3}".format(element)

                res_string += "{0:^15}\t".format(temp_str)

            res_string += "\n"

            res_string = res_string.replace("\t\n", "\n")

            write_file(filepath, name, res_string)
