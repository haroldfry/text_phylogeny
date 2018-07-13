# -*- coding: iso-8859-1 -*-
from tphyl2 import *
# Dividir as �rvores igualmente:

import os
import shutil

source_folder = "/home/gmarmerola/WR_10-50_revised2/Trees"
destination_folder = "/home/gmarmerola/WR_10-50_revised2/parallel"


def parallel_split(tree_folder, destination_folder, n_splits):

    tree_files = list_files(tree_folder, "")

    size = len(tree_files)
    bin_size = size / n_splits + 1

    print tree_files[0]

    print "bin size: {}".format(bin_size)

    for i in range(n_splits):

        count = 0

        while count < bin_size:

            name = tree_files[0].split(os.sep)[-1]

            if not os.path.exists(os.path.join(destination_folder, "{}".format(i))):
                os.makedirs(os.path.join(destination_folder, "{}".format(i)))

            shutil.copyfile(tree_files[0],
                            os.path.join(destination_folder, "{}".format(i),
                                         name))

            tree_files.pop(0)
            count += 1
            print "{0} out of {1} copied...".format(int(count + (i) * bin_size), size)

#parallel_split(source_folder, destination_folder, 5)


def compute_matrix_folder(tree_folder, matrix_folder):

    tree_files = list_files(tree_folder, "")

    count = 0

    for tree in tree_files:

        compute_matrix(tree, matrix_folder)


def compute_matrix_minibatch(tree_file, matrix_folder, minibatch):

    tree_string = read_file(tree_file)

    corpus = tree_string.split("<\\tphyldoc>")

    for i, element in enumerate(corpus):

        corpus[i] = nltk.word_tokenize(element)

    matrix = numpy.zeros((len(corpus), len(corpus)))

    support_matrix = numpy.zeros((len(corpus), len(corpus)))

    count = 0
    fixed_range = range(len(corpus) - 1)
    dyn_range = range(len(corpus) - 1, 0, -1)

    for i1 in fixed_range:

        for i2 in dyn_range:

            if count in minibatch:

                dist = editdistance(corpus[i1], corpus[i2])

                matrix[i1][i2] = dist
                matrix[i2][i1] = dist

                support_matrix[i1][i2] = 1
                support_matrix[i2][i1] = 1

            count += 1

        dyn_range.pop(-1)

    filename = "batch.dismat"

    write_matrix(matrix, matrix_folder, filename)

    write_matrix(support_matrix, matrix_folder, "mask.dismat")

    return matrix


def get_minibatch(tree_folder, n_cores, core):

    tree_size = len(list_files(tree_folder, ""))

    n_calc = int(tree_size * (tree_size - 1) / 2)

    minibatch_size = int(tree_size * (tree_size - 1) / (2 * n_cores))

    if core == n_cores:

        batch_list = range(minibatch_size * (core - 1), n_calc)

    else:

        batch_list = range(minibatch_size * (core - 1), minibatch_size * core)

    return batch_list


def merge_batches(matrix_file, matrix_len):

    directories = list_directories(matrix_file, "")

    mat_batch = {}

    mat_mask = {}

    for directory in directories:

        batch_number = int(directory.split(os.sep)[-1])

        mat_batch[batch_number] = numpy.loadtxt(os.path.join(directory,
                                                             "batch.dismat"),
                                                delimiter=";")

        mat_mask[batch_number] = numpy.loadtxt(os.path.join(directory,
                                                            "mask.dismat"),
                                               delimiter=";")

    matrix = numpy.zeros((matrix_len, matrix_len))

    for key in mat_batch.keys():

        for i1 in range(len(mat_batch[key])):

            for i2 in range(len(mat_batch[key])):

                if mat_mask[key][i1][i2]:

                    matrix[i1][i2] = mat_batch[key][i1][i2]

    return matrix

# testando a paralelização:
import time

tree_folder = "C:\Users\Guilherme\Desktop\Projeto IC\Text Phylogeny\Test Files\Reuters\Trees\\0.50"

n_cores = 20

ptime = time.time()

for core in range(1, n_cores + 1):

    matrix_folder = "C:\Users\Guilherme\Desktop\Projeto IC\Text Phylogeny\Test Files\matrix_test\\{}".format(core)

    minibatch = get_minibatch(tree_folder, n_cores, core)

    compute_matrix_minibatch(tree_folder, matrix_folder, minibatch)

    print time.time() - ptime

matrix_folder = "C:\Users\Guilherme\Desktop\Projeto IC\Text Phylogeny\Test Files\matrix_test"

ptime = time.time()

compute_matrix(tree_folder, matrix_folder)

print time.time() - ptime

matrix1 = merge_batches(matrix_folder, 50)

matrix2 = numpy.loadtxt(matrix_file + "\\0.50.dismat", delimiter=";")

matrix1 is matrix2
matrix1 == matrix2
