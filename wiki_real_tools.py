# -*- coding: iso-8859-1 -*-
# Criar árvores do Wikipedia Real no formato padrão
from tphyl2 import *
import pydot


def randomize_real_topology(tree_len):

    root = random.sample(range(tree_len), 1)[0]

    nodes = {}

    for i in range(tree_len):

        nodes[i] = i

    topology = [root] * tree_len

    ancestor = nodes.pop(root)

    while len(nodes) > 0:

        descendant = random.sample(nodes, 1)[0]

        topology[descendant] = ancestor

        ancestor = nodes.pop(descendant)

    return topology


def write_real_tree(base_folder, corpus, name):

    tree_string = "\n<\\tphyldoc>\n".join(corpus)

    write_file(base_folder, name + ".txt", tree_string)


def real_testset(base_folder, tree_lenghts, data_folder):

    parameters = {"Synonym": 0,
                  "Typo Insert": 0,
                  "Typo Remove": 0,
                  "Modifier Insert": 0,
                  "Modifier Remove": 0,
                  "Sentence Insert": 0,
                  "Sentence Remove": 0,
                  "Proportion of Change": [0]}

    base_trees = list_files(data_folder, "")

    for filename in base_trees:

        print filename

        base_file = read_file(filename).split("<\\wiki_real_doc>")

        if len(base_file) > max(tree_lenghts):

            for tree_len in tree_lenghts:

                topology = randomize_real_topology(tree_len)

                root = find_root(topology)

                tree = phyl_tree(topology)

                corpus = [""] * tree_len

                corpus[root] = base_file[0]

                for i, ancestor in enumerate(tree.ancestors):

                    corpus[tree.descendants[ancestor][0]] = base_file[i + 1]

                save_name = filename.split(os.sep)[-1]
                save_name = save_name.replace(".txt", "")
                save_name = "{}.{}".format(save_name, tree_len)

                print save_name

                write_phylogeny_file(os.path.join(
                    base_folder, "Phylogenies"),
                    save_name + ".phyl", topology,
                    parameters)

                write_real_tree(
                    os.path.join(base_folder, "Trees"),
                    corpus, save_name)

# Visualização de grafos:


def gen_graphs(res_file, save_file, mat_file, tps, integer=True):

    save_file = os.path.normpath(save_file)

    count = 0

    plots = list_files(save_file)

    for tp in tps:

        name = tp.split(os.sep)[-1].replace(".tpres", "")
        method = mat_file.split(os.sep)[-1]
        dis = np.loadtxt(os.path.join(mat_file,
                                      "{}.{}".format(name, "dismat")),
                         delimiter=';')

        tp = read_file(tp)
        tp = tp.split("\n")
        orig = ast.literal_eval(tp[0])
        recon = ast.literal_eval(tp[1])

        color_lst = color_list(orig)

        if not os.path.join(save_file, "{}_{}.{}".format(name, "orig", "png")) in plots:
            phyl_vis(orig, dis, save_file, "{}_{}.{}".format(
                name, "orig", "png"), color_lst, integer)

        phyl_vis(recon, dis, save_file, "{}_{}_{}.{}".format(
            name, "recon", method, "png"), color_lst, integer)

        count += 1
        print count


def color_list(topology):

    bin_size = len(topology) / 10

    if len(topology) < 10:

        bin_size = 1

    time_ordered = []

    for i in range(len(topology)):

        time_ordered.append(abs(find_levels(topology, i) / bin_size - 10))

    return time_ordered


def find_levels(topology, node):

    n = 0

    while node not in n_depth_set(topology, n):

        n += 1

    return n


def phyl_vis(topology, dis, save_path, name, color_lst, integer, label=True):

    graph = pydot.Dot(graph_type='digraph')
    root = find_root(topology)
    edges = get_edges(topology)

    for element in edges:

        nd1 = pydot.Node(element[0], style="bold")
        nd1.set_colorscheme("rdgy10")
        nd1.set_color(color_lst[element[0]])
        graph.add_node(nd1)

        nd2 = pydot.Node(element[1], style="bold")
        nd2.set_colorscheme("rdgy10")
        nd2.set_color(color_lst[element[1]])
        graph.add_node(nd2)

        edge = pydot.Edge(nd1, nd2)
        edge.set_dir("forward")

        if label:
            if integer:
                edge.set_label(int(dis[element[0], element[1]]))
            else:
                edge.set_label("{0:.4f}".format(dis[element[0], element[1]]))

        edge.set_fontsize(15.0)
        edge.set_colorscheme("rdgy10")
        edge.set_fontcolor(10)

        graph.add_edge(edge)

    save_path = os.path.normpath(save_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    graph.write_png(os.path.join(save_path, name))


def get_names(vis_folder, res_folder):

    files = list_files(vis_folder)

    names = [os.path.join(res_folder, f.split(os.sep)[-1].split("_")[0] + ".tpres")
             for f in files]

    return list(set(names))
