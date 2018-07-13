import sys
import os
import shutil
import random
import glob, os
from shutil import copyfile
import re
import random

corpus_dir = "/Users/bingyushen/Documents/phylonegy/text-phylogeny-all-datasets/Reuters_Dataset/C50test"
tree_base_dir = "/Users/bingyushen/Documents/phylonegy/text-phylogeny-all-datasets/Reuters_Dataset/test_tree_base"


def prepare_doc():
    for folder in os.listdir(corpus_dir):
        if not folder.startswith('.') and os.path.isdir(os.path.join(corpus_dir, folder)):
            folder_dir = corpus_dir+"/"+folder
            outputfilename = folder_dir+"/"+folder+".txt"
            newoutputfilename = folder_dir+".txt"
            if os.path.exists(outputfilename):
                open(outputfilename, 'w').close()

            with open(newoutputfilename, 'w') as outputfile:
                for files in os.listdir(folder_dir):

                    files_dir = folder_dir+"/"+files
                    with open(files_dir, 'rb') as readfile:
                        infile = readfile.read()
                        for line in infile:
                            outputfile.write(line)
                        outputfile.write("\n\n")
                    readfile.close()
                    print "files done ", files
            outputfile.close()
            print "done writing file ", newoutputfilename


def prepare_treebasefile():
    for folder in os.listdir(corpus_dir):
        if not folder.startswith('.') and os.path.isdir(os.path.join(corpus_dir, folder)):
            folder_dir = corpus_dir + "/" + folder
            file_a = random.choice(os.listdir(folder_dir))
            old_path = os.path.join(folder_dir, file_a)
            file_a_name = folder+"_"+file_a
            new_path = os.path.join(tree_base_dir, file_a_name)
            copyfile(old_path, new_path)


def prepare_corpus():
    os.chdir("/Users/bingyushen/Documents/phylonegy/text-phylogeny-all-datasets/Reuters_Dataset/C50test/")

    old_dir = "/Users/bingyushen/Documents/phylonegy/text-phylogeny-all-datasets/Reuters_Dataset/C50test"
    corpus_dir = "/Users/bingyushen/Documents/phylonegy/text-phylogeny-all-datasets/Reuters_Dataset/test_corpus"
    for file in glob.glob("*.txt"):
        file_dir = old_dir + "/" + file
        new_dir = corpus_dir + "/" + file
        copyfile(file_dir, new_dir)


def generatetrainingdata():
    trees_dir = "/Users/bingyushen/Downloads/drive-download-20161026T043937Z/1026_test_tree_10/Trees"
    phyl_dir = "/Users/bingyushen/Downloads/drive-download-20161026T043937Z/1026_test_tree_10/Phylogenies"

    for tree in os.listdir(trees_dir):
        print "entering new tree", tree
        doc = tree.split('_')[0]
        print "doc: ", doc
        tree_dir = os.path.join(trees_dir, tree)
        treefiles = open(tree_dir,'r').read().split("<\\tphyldoc>")

        edgefile_dir = os.path.join(phyl_dir, tree)
        edgefile_dir = edgefile_dir.replace(".txt", ".phyl")

        with open(edgefile_dir, 'r') as edgefile:
            edges = edgefile.readlines()

            # prepare edge table
            tree_edge = [[0 for i in range(2)] for j in range(len(edges))]

            # print tree_edge

            for edge in edges:
                # print edge
                nodes = edge.split()
                first_node = nodes[0]
                second_node = nodes[2]
                # print edges.index(edge)
                tree_edge[edges.index(edge)][0] = int(first_node)
                tree_edge[edges.index(edge)][1] = int(second_node)
        labels = []
        for i in range(len(treefiles)):
            for j in range(len(treefiles)):
                if i != j:
                    label = 0
                    sample1 = treefiles[i]
                    sample2 = treefiles[j]
                    for x in range(len(edges)):
                        if i==tree_edge[x][0] and j==tree_edge[x][1]:
                            # print "i: ", i, "j: ", j, "edgetable: " , tree_edge[x][0], tree_edge[x][1]
                            label = 1
                            writepostrainingsamples(sample1, sample2, doc, 1)
                            # print label
                        else:
                            # print "i: ", i, "j: ", j, "edgetable: ", tree_edge[x][0], tree_edge[x][1]
                            label = 0
                            writenegtrainingsamples(sample1, sample2, doc, 0)
                            # print label
                    # labels.append(label)
                    # print label
        # print len(labels)


def writepostrainingsamples(sample1, sample2, doc, label):
    rdscorpus = "/Users/bingyushen/Downloads/drive-download-20161026T043937Z/test_rdscorpus"
    doc += '.rds'
    docpath = os.path.join(rdscorpus, doc)
    # print docpath
    sample1 = sample1.replace('\n', '')
    sample2 = sample2.replace('\n', '')
    posfilepath = "/Users/bingyushen/Downloads/drive-download-20161026T043937Z/test_positive.txt"
    with open(posfilepath, 'a') as postraining:
        postraining.write("\"")
        postraining.write(sample1)
        postraining.write("\"")
        postraining.write(",")
        postraining.write('\"')
        postraining.write(sample2)
        postraining.write("\"")
        postraining.write(",")
        postraining.write(docpath)
        postraining.write(",")
        postraining.write(docpath)
        postraining.write("\n")

    postraining.close()


def writenegtrainingsamples(sample1, sample2, doc, label):
    # a=0
    rdscorpus = "/Users/bingyushen/Downloads/drive-download-20161026T043937Z/test_rdscorpus"
    doc += '.rds'
    docpath = os.path.join(rdscorpus, doc)
    # print docpath
    sample1 = sample1.replace('\n', '')
    sample2 = sample2.replace('\n', '')
    negfilepath = "/Users/bingyushen/Downloads/drive-download-20161026T043937Z/test_negative.txt"
    with open(negfilepath, 'a') as negtraining:
        negtraining.write("\"")
        negtraining.write(sample1)
        negtraining.write("\"")
        negtraining.write(",")
        negtraining.write('\"')
        negtraining.write(sample2)
        negtraining.write("\"")
        negtraining.write(",")
        negtraining.write(docpath)
        negtraining.write(",")
        negtraining.write(docpath)
        negtraining.write("\n")

    negtraining.close()


def cutnegtrainingfile():
    negtrainingfile = "/Users/bingyushen/Documents/phylonegy/text-phylogeny-all-datasets/Reuters_Dataset/" \
                      "myresults/170219_tree_0.1/SVM_training_samples/negative1.txt"
    newnegtrainingfile = "/Users/bingyushen/Documents/phylonegy/text-phylogeny-all-datasets/Reuters_Dataset" \
                         "/myresults/170219_tree_0.1/SVM_training_samples/new_negative1.txt"
    with open(negtrainingfile, 'r') as negtraining:
        negs = negtraining.readlines()
        for x in range(230):
            neg = random.choice(negs)
            with open(newnegtrainingfile, 'a') as newnegtraining:
                newnegtraining.write(neg)


if __name__ == "__main__":
    cutnegtrainingfile()