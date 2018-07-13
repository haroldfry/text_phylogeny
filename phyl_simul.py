import sys
sys.path.insert(1, '/Library/Python/2.7/site-packages')
from phylogeny_test2 import *
from tphyl2 import *
import fusion_tfidf_glove

# for prop in [10, 15, 20]: #, 15, 20 different extents of modification
number_of_trees = 1

# vector with tree sizes
tree_lengths = [10]

# parameters
parameters = {"Synonym": 0.5,
              "Typo Insert": 0.1,
              "Typo Remove": 1,
              "Modifier Insert": 0,
              "Modifier Remove": 0.5,
              "Sentence Insert": 0,
              "Sentence Remove": 1,
              "Proportion of Change": [0.10]}

data_file = "/Users/bingyushen/Documents/phylonegy/text-phylogeny-all-datasets/Reuters_Dataset/Reuters_filtered"
base_file = "/Users/bingyushen/Documents/phylonegy/text-phylogeny-all-datasets/Reuters_Dataset/myresults/170403_tree_{}".format(0.10)
newcase = phylogeny_test(base_file, data_file, number_of_trees,tree_lengths, parameters)
held_out_prop = 0.2
newcase.simulation(held_out_prop)
# dismatrix_41(base_file, method_name="41_features")
dissimilarity_tf_idf(base_file)
# glove_vectors(base_file)
# res = newcase.evaluate(random_root_alg, "random_root", method="glove_embed")  # choosing the root randomly
# res = newcase.evaluate(least_tree_cost_alg, "tree_cost", method="glove_embed")  # using minimum tree cost heuristic
res = newcase.evaluate(random_root_alg, "random_root", method="tf-idf")
res = newcase.evaluate(least_tree_cost_alg, "tree_cost", method="tf-idf")
