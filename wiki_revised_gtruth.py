# -*- coding: iso-8859-1 -*-
from tphyl2 import *
import unicodedata


def wr_groundtruth(data_file, length, max_len=False, randomize=True):

    corpus = read_file(data_file)

    corpus = corpus.decode("utf-8")

    corpus = unicodedata.normalize('NFKD', corpus).encode("ascii", "ignore")

    corpus = corpus.split("</doc>\n")

    if len(corpus) < length and not max_len:
        return False

    if corpus[0].startswith("\n"):
        corpus[0] = corpus[0][1:]

    for i in range(len(corpus)):
        corpus[i] = "\n".join(corpus[i].split("\n")[1:])

    topology = [0] * length

    new_corpus = [corpus[0]]

    father = 0
    for i in range(1, len(corpus)):

        if len(new_corpus) == length:
            break

        if (len(corpus) - i) + len(new_corpus) < length and not max_len:
            return False

        if corpus[i] in new_corpus:
            father = new_corpus.index(corpus[i])
        else:
            new_corpus.append(corpus[i])
            topology[len(new_corpus) - 1] = father
            father = len(new_corpus) - 1

    topology = topology[:len(new_corpus)]

    if randomize:
        return randomize_corpus(new_corpus, topology)
    else:
        return {"topology": topology, "corpus": new_corpus}


def randomize_corpus(corpus, topology):

    nodes = random.sample(range(len(topology)), len(topology))

    root = find_root(topology)

    new_topology = [nodes[root]] * len(topology)
    new_corpus = [""] * len(topology)

    for i in range(len(topology)):

        desc = find_descendants(topology, i)
        new_corpus[nodes[i]] = corpus[i]

        for d in desc:

            new_topology[nodes[d]] = nodes[i]

    return {"topology": new_topology, "corpus": new_corpus}


def write_real_tree(base_folder, corpus, name):

    tree_string = "\n<\\tphyldoc>\n".join(corpus)

    write_file(base_folder, name + ".txt", tree_string)
