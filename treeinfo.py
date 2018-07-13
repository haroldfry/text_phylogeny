import sys
sys.path.insert(1, '/Library/Python/2.7/site-packages')
from tphyl2 import *

# all paths to the phylogenies (defined in tphyl2.py)
# phyl_paths = list_files("/Users/bingyushen/Documents/phylogeny/text-phylogeny-src/Datasets/Reuters_Dataset/myresults/1024tree_10/Phylogenies")
phyl_paths = list_files("/Users/bingyushen/Documents/phylogeny/text-phylogeny-src/Datasets/Reuters_Dataset/myresults/reuters_10/Phylogenies")
# phyl_filepath = phyl_paths[0] # using first path
#
# topology = get_topology(phyl_filepath)
#
# print "topology:", topology # topology, in vector form
# print "root:", find_root(topology) # tree root
sys.path.insert(1, '/Library/Python/2.7/site-packages')
for phyl in phyl_paths:
    topology = get_topology(phyl)
    print "topology:", topology
    print "root:", find_root(topology)