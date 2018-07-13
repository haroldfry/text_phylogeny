import os
import numpy as np
import scipy

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