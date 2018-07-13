from phylogeny_test2 import *
from tphyl2 import *

# tree_paths = list_files("/Users/bingyushen/Documents/phylonegy/text-phylogeny-all-datasets/Reuters_Dataset/myresults/Trees")
# tree_filepath = tree_paths[0]
# corpus = read_file(tree_filepath).split("<\\tphyldoc>")
# print corpus[0]
# tknzd_corpus = tokenized(corpus[0], 0.2)
#
# tknzd_corpus = tokenized(corpus[0], 0.2)
# print "a single tagged world: ", tknzd_corpus.tokens[0][0]
# print "list of tokens:", tknzd_corpus.join_sentences()[0:10], "(...)"
# print "full text:", tknzd_corpus.join_all()[0:80], "(...)"
#
# phyl_paths = list_files("/Users/bingyushen/Documents/phylonegy/text-phylogeny-all-datasets/Reuters_Dataset/myresults/Phylogenies")
# phyl_filepath = phyl_paths[0]
#
# topology = get_topology(phyl_filepath)
#
# print "topology: ", topology
# print "root: ", find_root(topology)
#
# all paths to the matrices
mat_paths = list_files("/Users/bingyushen/Documents/phylonegy/text-phylogeny-all-datasets/Reuters_Dataset/myresults/reuters_20/DisMatrices/tf-idf_word3")
mat_filepath = mat_paths[0] # using first path

# complete_path = os.path.join(tphyl_dir, mat_filepath) # np.loadtxt admits only complete paths
#
# dissimilarity matrix
dis = np.loadtxt(mat_filepath, delimiter=';')

# equivalent graph, using kruskal's algorithm
[edges, dis] = kruskal(mat_filepath)

# build a directed topology, assiming node number 0 is the root
reconstructed = build_topology(0, edges)

print "list of edges:", edges
print "reconstructed topology:", reconstructed