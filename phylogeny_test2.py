# -*- coding: iso-8859-1 -*-

from __future__ import division
from nltk.stem.wordnet import WordNetLemmatizer
from tphyl2 import *
import random
import nltk
from nltk.corpus import wordnet
import os
import numpy as np
import en
import copy
import time


class phylogeny_test:

    def __init__(self, base_file, data_file, number_of_trees,
                 tree_lenghts, parameters):

        # Variables:
        self.base_file = base_file
        self.data_file = data_file
        self.number_of_trees = number_of_trees
        self.tree_lenghts = tree_lenghts
        self.parameters = parameters

        # Filenames:
        self.phyl_file = os.path.join(self.base_file, "Phylogenies")
        self.mat_file = os.path.join(self.base_file, "DisMatrices")
        self.tree_file = os.path.join(self.base_file, "Trees")
        self.res_file = os.path.join(self.base_file, "Results")

        typo_file = "/Users/bingyushen/Downloads/Archive/Tools/Misspellings/misspellings_final.txt"
        mods_dir = "/Users/bingyushen/Downloads/Archive/Tools/COCA-N-grams/mods"

        # Tools:
        self.typos = typo_list(typo_file)
        self.mods = modifier_list(mods_dir)

    def simulation(self, hout_prop):

        originals = list_files(self.data_file, "")
        # print originals

        n_topologies = self.number_of_trees
        print n_topologies

        for n in self.tree_lenghts:

            print "Generating trees with {} nodes:".format(n)

            topologies = generate_topology(n_topologies, n, rd=True)

            orig_sample = random.sample(originals, self.number_of_trees)

            count = 0

            for original in orig_sample:

                #obj_number = int(original.split(os.sep)[-1].split(".")[0])
                obj_number = original.split(os.sep)[-1].split(".")[0]

                save_name = "{0}.{1}".format(obj_number, n)

                topology = topologies.pop(0)

                write_phylogeny_file(self.phyl_file, save_name + ".phyl",
                                     topology, self.parameters)

                #top = get_topology(phyl_file, parameters=True)

                corpus = build_corpus(original, [topology, self.parameters], hout_prop,
                                      transform, self.typos, self.mods)

                write_tree(self.tree_file, corpus, save_name)

                count += 1
                print "{0} out of {1} trees completed...".format(count, len(orig_sample))

    def dissimilarity(self, method_name="wu_et_al", dis_func=editdistance,
                      normalize=False, filtering=None):

        print "Computing dissimilarity matrices:"

        tree_files = list_files(self.tree_file, "")

        if not os.path.exists(self.mat_file):
            os.makedirs(self.mat_file)

        mat_files = list_files(os.path.join(self.mat_file, method_name), "")

        mat_files = [
            element.split(os.sep)[-1].replace(".dismat", "") for element in mat_files]

        count = 0

        start_time = time.time()

        for tree in tree_files:

            name = tree.split(os.sep)[-1].replace(".txt", "")

            if not name in mat_files:
                compute_matrix(tree, self.mat_file, method_name,
                               dis_func, normalize, filtering)

            count += 1
            print "{0} out of {1} matrices completed...".format(count, len(tree_files))

        elapsed_time = time.time() - start_time
        time_string = "{}s | {} matrices".format(elapsed_time, len(tree_files))
        write_file(self.base_file, "time_{}.txt".format(method_name),
                   time_string)

    def evaluate(self, algorithm, folder, method="wu_et_al"):

        res_list = {}

        for n in self.tree_lenghts:

            res_list[n] = []

            dismat_list = list_files(
                os.path.join(self.mat_file, method), ".{}.dismat".format(n))

            print "dismat_dir", (os.path.join(self.mat_file, method), ".{}.dismat".format(n))

            for matrix in dismat_list:

                name = matrix.split(os.sep)[-1].replace(".dismat", "")

                orig = get_topology(os.path.join(self.phyl_file,
                                                 "{}.phyl".format(name)))

                recon = algorithm(matrix)

                newdir = (os.path.join(self.res_file, method,
                                        folder, "topologies"),
                           "{}.tpres".format(name),
                           "{0}\n{1}\n".format(orig, recon))
                print newdir

                write_file(os.path.join(self.res_file, method,
                                        folder, "topologies"),
                           "{}.tpres".format(name),
                           "{0}\n{1}\n".format(orig, recon))

                vector = [ind_edges(orig, recon),
                          dir_edges(orig, recon),
                          leaves(orig, recon),
                          n_depth(orig, recon, 0),
                          ancestry(orig, recon),
                          depth(orig, recon),
                          n_depth(orig, recon, 1),
                          n_depth(orig, recon, 2),
                          n_depth(orig, recon, 3),
                          len(n_depth_set(recon, 1)),
                          len(n_depth_set(recon, 2)),
                          len(n_depth_set(recon, 3))]

                res_list[n].append(vector)

        write_results(res_list,
                      os.path.join(self.res_file, method, folder),
                      "{}_mean.txt".format(folder))
        print "writing result"
        return res_list
