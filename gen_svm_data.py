import os
import re
import numpy as np
import scipy.spatial


def gen_embed_features(svm_dir):
    possamnum = 0
    negsamnum = 0
    for tree in os.listdir(embed_dir):
        print "entering new tree", tree
        doc = tree.split('_')[0]
        print "doc: ", doc
        tree_dir = os.path.join(embed_dir, tree)

        # treefiles is the 1*100 vectors for each text
        treefiles = open(tree_dir,'r').read().split("]")
        treefiles = treefiles[:-1]
        print "length of treefiles: ", len(treefiles)
        edgefile_dir = os.path.join(phyl_dir, tree)
        edgefile_dir = edgefile_dir.replace("_embed.txt", ".phyl")

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

            print tree_edge

        for i in range(len(treefiles)):
            for j in range(i, len(treefiles)):
                if i != j:
                    label = 0
                    sample1 = treefiles[i]
                    sample2 = treefiles[j]
                    if is_dir_edge(i, j, tree_edge):
                        # print "i: ", i, "j: ", j
                        label = 1
                        sample1 = sample1.replace('[', '')
                        sample1 = sample1.replace('\n', '')
                        sample2 = sample2.replace('[', '')
                        sample2 = sample2.replace('\n', '')

                        numbers1 = sample1.split(' ')
                        newnumbers1 = []
                        for item in numbers1:
                            if item is not '':
                                newnumbers1.append(float(item))
                        feature1 = np.asarray(newnumbers1)

                        numbers2 = sample2.split(' ')
                        newnumbers2 = []
                        for item in numbers2:
                            if item is not '':
                                newnumbers2.append(float(item))
                        feature2 = np.asarray(newnumbers2)

                        # compute the distance between two vectors using just subtraction
                        features = feature1 - feature2
                        # compute the distance between two vectors using euclidean distance
                        # eucfeatures = scipy.spatial.distance.euclidean(feature1, feature2)
                        # print eucfeatures
                        writingfeatures(label, features.tolist())
                        possamnum += 1

                    else:
                        # print "i: ", i, "j: ", j
                        label = 0
                        sample1 = sample1.replace('[', '')
                        sample1 = sample1.replace('\n', '')
                        sample2 = sample2.replace('[', '')
                        sample2 = sample2.replace('\n', '')

                        numbers1 = sample1.split(' ')
                        newnumbers1 = []
                        for item in numbers1:
                            if item is not '':
                                newnumbers1.append(float(item))
                        feature1 = np.asarray(newnumbers1)

                        numbers2 = sample2.split(' ')
                        newnumbers2 = []
                        for item in numbers2:
                            if item is not '':
                                newnumbers2.append(float(item))
                        feature2 = np.asarray(newnumbers2)

                        features = feature1 - feature2
                        # compute the distance between two vectors using euclidean distance
                        # eucfeatures = scipy.spatial.distance.euclidean(feature1, feature2)
                        # print eucfeatures
                        writingfeatures(label, features.tolist())
                        negsamnum += 1

        print "# of positive samples: ", possamnum
        print "# of negative samples: ", negsamnum


def is_dir_edge(node1, node2, tree_edge):
    flag = 0
    for treepair in tree_edge:
        if treepair[0] == node1 and treepair[1] == node2:
            flag = 1
        elif treepair[0] == node2 and treepair[1] == node1:
            flag = 1

    if flag == 1:
        return True
    else:
        return False


def writingfeatures(label, features):
    if label == 1:
        writingvec = "+1 "
        for vec in features:
            indexed_vec = features.index(vec)+1
            writingvec += str(indexed_vec)
            writingvec += ":"
            writingvec += str(vec)
            writingvec += " "

        writingvec += "\n"
        writingpath = os.path.join(svm_dir, 'positive1.txt')
        writingpos = open(writingpath, 'a')
        writingpos.write(writingvec)

    else:
        label = -1
        writingvec = str(label) + " "
        for vec in features:
            indexed_vec = features.index(vec) + 1
            writingvec += str(indexed_vec)
            writingvec += ":"
            writingvec += str(vec)
            writingvec += " "

        writingvec += "\n"
        writingpath = os.path.join(svm_dir, 'negative1.txt')
        writingpos = open(writingpath, 'a')
        writingpos.write(writingvec)


if __name__ == "__main__":
    embed_dir = "/Users/bingyushen/Documents/phylonegy/text-phylogeny-all-datasets/Reuters_Dataset/myresults/170219_tree_0.1/Embedding_vectors"
    phyl_dir = "/Users/bingyushen/Documents/phylonegy/text-phylogeny-all-datasets/Reuters_Dataset/myresults/170219_tree_0.1/Phylogenies"
    base_file = "/Users/bingyushen/Documents/phylonegy/text-phylogeny-all-datasets/Reuters_Dataset/myresults/170219_tree_0.1"
    svm_dir = os.path.join(base_file, "SVM_training_samples")
    if not os.path.exists(svm_dir):
        os.makedirs(svm_dir)

    gen_embed_features(svm_dir)