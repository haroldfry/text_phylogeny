
res_file = "C:\Users\Guilherme\Desktop\Projeto IC\Text Phylogeny\Test Files\Wikipedia_real_28-01-2015\Results\\random_root\\random_root_mean.txt"


def latex_format(res_file):

    results = read_file(res_file)
    lines = results.split("\n")
    header = lines[0].split("\t")
    header = ["\\textbf{" + element + "}" for element in header]
    lines[0] = "\t".join(header)
    results = results.replace("\t", "&")
    results = results.replace("\n", "\\\\")

    write_file(
        "\\".join(res_file.split("\\")[0:-1]), "latex_formatted.txt", results)


for i in ["cut", "non", "light", "heavy"]:

    for j in ["ncd_bz2", "wu_et_al", "tf-idf"]:

        base_file = "C:\Users\Guilherme\Desktop\Projeto IC\Text Phylogeny\Test Files\CS11\\{}\\Results\\{}\\leaves_cost".format(i, j)
        try:
            shutil.rmtree(base_file)
        except WindowsError:
            a = 0


# Implementando NCD:
import bz2

xbytes = open(filex, 'r').read()
ybytes = open(filey, 'r').read()


def ncd_bz2(x, y):

    xbytes = " ".join(x)
    print xbytes
    ybytes = " ".join(y)
    print ybytes
    xybytes = xbytes + ybytes
    cx = bz2.compress(xbytes)
    cy = bz2.compress(ybytes)
    cxy = bz2.compress(xybytes)
    if len(cy) > len(cx):
        n = (len(cxy) - len(cx)) / float(len(cy))
    else:
        n = (len(cxy) - len(cy)) / float(len(cx))
    return n

# NCD com compressor gzip

# tf-idf
tree_file = "C:\Users\Guilherme\Desktop\Projeto IC\Text Phylogeny\Test Files\code_val\Trees\\16.10.txt"

matrix_folder = "C:\Users\Guilherme\Desktop\Projeto IC\Text Phylogeny\Test Files\code_val\DisMatrices"

method_name = "tf-idf"

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


def tf_idf_dismat(tree_file, matrix_folder, method_name):

    corpus = read_file(tree_file).split("\n<\\tphyldoc>\n")

    count_vectorizer = CountVectorizer()
    tfidf = TfidfTransformer(norm="l2")

    count_vectorizer.fit_transform(corpus)

    freq_term_matrix = count_vectorizer.transform(corpus)

    tfidf.fit(freq_term_matrix)

    tf_idf_matrix = tfidf.transform(freq_term_matrix)

    dis_matrix = abs(
        (tf_idf_matrix * np.transpose(tf_idf_matrix)).todense() - 1)

    dis_matrix[dis_matrix < 1e-10] = 0

    dis_matrix = np.array(dis_matrix)

    filename = tree_file.split(os.sep)[-1].replace(".txt", "") + ".dismat"

    write_matrix(
        dis_matrix, os.path.join(matrix_folder, method_name), filename)

    return dis_matrix


def dissimilarity_tf_idf(base_file, method_name="tf-idf"):

    tree_file = os.path.join(base_file, "Trees")
    mat_file = os.path.join(base_file, "DisMatrices")

    print "Computing dissimilarity matrices:"

    tree_files = list_files(tree_file, "")

    if not os.path.exists(mat_file):
        os.makedirs(mat_file)

    mat_files = list_files(os.path.join(mat_file, method_name), "")

    mat_files = [
        element.split(os.sep)[-1].replace(".dismat", "") for element in mat_files]

    count = 0

    start_time = time.time()

    for tree in tree_files:

        name = tree.split(os.sep)[-1].replace(".txt", "")

        if not name in mat_files:
            tf_idf_dismat(tree, mat_file, method_name)

        count += 1
        print "{0} out of {1} matrices completed...".format(count, len(tree_files))

    elapsed_time = time.time() - start_time
    time_string = "{} s | {} matrices".format(elapsed_time, len(tree_files))
    write_file(base_file, "time_{}.txt".format(method_name), time_string)

# Normalizar matrizes de wu_et_al

count = 0

for i in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, "mixed"]:

    mat_folder = "C:\Users\Guilherme\Desktop\Text Phylogeny\Test Files\NAACL\\reuters_{}\DisMatrices".format(i)
    tree_folder = "C:\Users\Guilherme\Desktop\Text Phylogeny\Test Files\NAACL\\reuters_{}\Trees".format(i)

    method = "wu_et_al"

    dismat_list = list_files(os.path.join(mat_folder, method))
    tree_list = list_files(tree_folder)

    for j in range(len(dismat_list)):

        dis = np.loadtxt(dismat_list[j], delimiter=';')

        tree_string = read_file(tree_list[j])

        corpus = tree_string.split("<\\tphyldoc>")

        for k, element in enumerate(corpus):

            corpus[k] = nltk.word_tokenize(element)

        norm_matrix = np.zeros((len(corpus), len(corpus)))

        fixed_range = range(len(corpus) - 1)
        dyn_range = range(len(corpus) - 1, 0, -1)

        for i1 in fixed_range:

            for i2 in dyn_range:

                dist = dis[i1][i2]

                norm_matrix[i1][i2] = dist / \
                    max([len(corpus[i1]), len(corpus[i2])])
                norm_matrix[i2][i1] = dist / \
                    max([len(corpus[i1]), len(corpus[i2])])

            dyn_range.pop(-1)

        filename = tree_list[j].split(
            os.sep)[-1].replace(".txt", "") + ".dismat"

        write_matrix(norm_matrix, os.path.join(
            mat_folder, method + "_normalized"), filename)

        count += 1
        print count

# Juntar árvores para usar o cluster:
folder = "C:\Users\Guilherme\Desktop\Text Phylogeny\Test Files\NAACL\\reuters_5\Trees"


def join_trees(folder):

    folder = os.path.normpath(folder)

    target_folder = "{}".format(os.sep).join(folder.split(os.sep)[0:-1])

    name = folder.split(os.sep)[-1] + ".txt"

    trees = list_files(folder)

    target_file = open(os.path.join(target_folder, name), "w")
    target_file.close()

    for tree in trees:

        tree_name = tree.split(os.sep)[-1]

        tree_content = "{}\n".format(
            tree_name) + read_file(tree) + "\n<\\tphyl_tree_split>\n"

        with open(os.path.join(target_folder, name), "a") as myfile:
            myfile.write(tree_content)


def read_joined_tree(joined_file, index):

    content = read_file(joined_file).split("\n<\\tphyl_tree_split>\n")[index]

    name = content.split("\n")[0]
    content = content.split("\n")[1:]

    return {"name": name, "content": content}


# Fazer gráfico discriminado performance dos metodos (summary)


def res_summary(base_folder, methods, heuristic, measure):

    base_folder = os.path.join(base_folder, "Results")

    summary_dict = {}

    for method in methods:

        res_file = os.path.join(base_folder, method, heuristic,
                                "{}_mean.txt".format(heuristic))

        results = read_file(res_file)

        results = results.split("\n")

        results.pop(-1)

        measure_index = 0

        for i, element in enumerate(results[0].split("\t")):

            if element.strip() == measure:
                measure_index = i

        measure_list = []

        for i in range(1, len(results)):

            measure_list.append(float(results[i].split("\t")[measure_index]))

        mean = np.mean(measure_list)

        std_dev = np.std(measure_list)

        summary_dict[method] = {"mean": mean, "std_dev": std_dev}

    return summary_dict


def save_summary(summary_dict, folder, name):

    header = ["Method", "Mean", "Std. Dev"]

    summary_string = "{0[0]:^15}\t{0[1]:^15}\t{0[2]:^15}\n".format(header)

    keys = sorted(summary_dict.keys())

    for key in keys:

        summary_string += "{0:^15}".format(key)
        summary_string += "{0:^15}".format(summary_dict[key]["mean"])
        summary_string += "{0:^15}".format(summary_dict[key]["std_dev"]) + "\n"

    write_file(folder, name + ".txt", summary_string)


base_folder = "C:\Users\Guilherme\Desktop\Text Phylogeny\Test Files\WR_10-50_revised2"

methods = ["tf-idf_word1_raw", "tf-idf_word2_raw",
           "tf-idf_word3_raw", "tf-idf_word4_raw",
           "tf-idf_word5_raw", "tf-idf_char1_raw",
           "tf-idf_char2_raw", "tf-idf_char3_raw",
           "tf-idf_char4_raw", "tf-idf_char5_raw",
           "tf-idf_char6_raw", "tf-idf_char7_raw",
           "tf-idf_combined_1-3", "tf-idf_combined_1-4",
           "tf-idf_combined_1-5", "tf-idf_combined_2-3",
           "tf-idf_combined_2-4", "tf-idf_combined_2-5",
           "tf-idf_combined_3-3", "tf-idf_combined_3-4",
           "tf-idf_combined_3-5", "wu_et_al",
           "wu_et_al_normalized", "ncd_bz2"]

heuristic = "random_root"
measure = "Ind. Edges"

summary_dict = res_summary(base_folder, methods, heuristic, measure)

folder = "C:\Users\Guilherme\Desktop\Text Phylogeny\Test Files\WR_10-50_revised2\\Results"
name = "least_size_summary"
save_summary(summary_dict, folder, name)


# Discriminar tempo dos métodos:

def time_summary(base_folder, methods):

    time_dict = {}

    for method in methods:

        time_file = os.path.join(base_folder, "time_{}.txt".format(method))

        time = float(read_file(time_file).split()[0])

        time_dict[method] = time

    return time_dict


def save_time_summary(time_dict, folder, name):

    header = ["Method", "Running time"]

    summary_string = "{0[0]:^15}\t{0[1]:^15}\n".format(header)

    keys = sorted(time_dict.keys())

    for key in keys:

        summary_string += "{0:^15}".format(key)
        summary_string += "{0:^15}".format(time_dict[key]) + "\n"

    write_file(folder, name + ".txt", summary_string)


base_folder = "C:\Users\Guilherme\Desktop\Text Phylogeny\Test Files\WR_10-50_revised2"
methods = ["tf-idf_word1_raw", "tf-idf_word2_raw",
           "tf-idf_word3_raw", "tf-idf_word4_raw",
           "tf-idf_word5_raw"]

time_dict = time_summary(base_folder, methods)

folder = "C:\Users\Guilherme\Desktop\Text Phylogeny\Test Files\WR_10-50_revised2\\Results"
name = "tf-idf_combined_time_summary"
save_time_summary(time_dict, folder, name)


# Calcular e gravar os tamanhos das árvores:
def get_sizes(base_folder):

    tree_folder = os.path.join(base_folder, "Trees")

    dest_folder = os.path.join(base_folder, "tree-sizes")

    trees = list_files(tree_folder)

    for tree_file in trees:

        tree_contents = read_file(tree_file).split("\n<\\tphyldoc>\n")

        sizes = [len(nltk.word_tokenize(t)) for t in tree_contents]

        name = tree_file.split(os.sep)[-1]

        write_file(dest_folder, name, str(sizes))

path = "D:\Text Phylogeny\Test Files\NAACL"

method = "wu_et_al"

heuristic = "tree_cost"

r = join_results(path, method, heuristic)

write_joined(r, path, method, heuristic)

#média e desvio padrão entre os tamanhos:

avg_dict = {}
sd_dict = {}

for measure in ["Ind.Edges", "DirectEdges", "Leaves", "Roots", "Ancestry", "Depth"]:

    avg_matrix = []

    for key in r[measure].keys():

        avg_matrix.append(r[measure][key])

    avg_dict[measure] = np.asarray(np.mean(np.matrix(avg_matrix), 0))[0].tolist()
    sd_dict[measure] = np.asarray(np.std(np.matrix(avg_matrix), 0))[0].tolist()
    print measure
    avg_string = ""
    sd_string = ""
    for i in range(len(avg_dict[measure])):
        avg_string += "{} ".format(avg_dict[measure][i])
        sd_string += "{} ".format(sd_dict[measure][i])
    print avg_string
    print sd_string


path = "C:\Users\Guilherme\Desktop\Text Phylogeny\Test Files\NAACL"

method = "tf-idf_char"

heuristic = "tree_cost"

fig_path = "C:\Users\Guilherme\Desktop\Text Phylogeny\Test Files\NAACL_Plots\\Plots\\tf-idf_char_comparison_tree_cost"


res_dict = read_joined(path)
#res_dict = get_tf_idf_means(path, method, heuristic)

for key in res_dict.keys():

    if not key == "Proportion":

        v = list(itertools.chain(*res_dict[key].values()))

        sizes = {'2-Depth': [min(v) * 0.95, max(v) * 1.05],
                 '3-Set': [min(v) * 0.95, max(v) * 1.05],
                 'Leaves': [0.5, 1.02],
                 '1-Set': [min(v) * 0.95, max(v) * 1.05],
                 'DirectEdges': [0.2, 1.02],
                 'Depth': [0, 8],
                 'Ancestry': [0, 0.9],
                 'Roots': [0, 0.5],
                 '1-Depth': [min(v) * 0.95, max(v) * 1.05],
                 '2-Set': [min(v) * 0.95, max(v) * 1.05],
                 '3-Depth': [min(v) * 0.95, max(v) * 1.05],
                 'Ind.Edges': [0.5, 1.02]}

        locs = {'2-Depth': 3,
                '3-Set': 1,
                'Leaves': 3,
                '1-Set': 1,
                'DirectEdges': 3,
                'Depth': 1,
                'Ancestry': 1,
                'Roots': 1,
                '1-Depth': 3,
                '2-Set': 1,
                '3-Depth': 3,
                'Ind.Edges': 3}

        plot_joined(res_dict, key, sizes[key], locs[key], save=True, path=fig_path)

# Construir dataset com novo ground truth

data_folder = "D:\Text Phylogeny\Datasets\Wikipedia Real\\full_histories"
base_folder = "D:\Text Phylogeny\Datasets\Wikipedia Real\\processed_trees_20"

# corpus_lenghts = [100, 200, 400]
corpus_lenghts = [20]

parameters = {"Synonym": 0,
              "Typo Insert": 0,
              "Typo Remove": 0,
              "Modifier Insert": 0,
              "Modifier Remove": 0,
              "Sentence Insert": 0,
              "Sentence Remove": 0,
              "Proportion of Change": [0]}

base_corpus = list_files(data_folder)

for filename in base_corpus:

    for length in corpus_lenghts:

        obj = wr_groundtruth(filename, length, randomize=False)

        if not obj:
            continue

        topology = obj["topology"]
        corpus = obj["corpus"]

        save_name = filename.split(os.sep)[-1]
        save_name = save_name.replace(".txt", "")
        save_name = "{}.{}".format(save_name, length)

        print save_name

        write_phylogeny_file(os.path.join(
                             base_folder, "Phylogenies"),
                             save_name + ".phyl", topology,
                             parameters)

        write_real_tree(os.path.join(base_folder, "Trees"),
                        corpus, save_name)


# Visualização das novas topologias:

import pydot
import ast
import os

phyl_folder = "D:\Text Phylogeny\Datasets\Wikipedia Real\\processed_trees_20\\Phylogenies"

save_path = "D:\Text Phylogeny\Datasets\Wikipedia Real\\processed_trees_20\\Visualization"

phylogenies = list_files(phyl_folder)

for phyl_file in phylogenies:

    topology = get_topology(phyl_file)
    save_name = phyl_file.split(os.sep)[-1].replace(".phyl", "") + ".png"
    dis = np.matrix([[0] * len(topology)] * len(topology))
    phyl_vis(topology, dis, save_path, save_name, [10]*len(topology),
             True, [0.5]*len(topology), label=False)

sample_size = 200
length = 4289

if sample_size > length:

    sample_size = length

inds = random.sample(range(length), sample_size)

method = "wu_et_al_normalized"

res_file = "D:\Text Phylogeny\Test Files\WR_10-50_revised2\Results\\{}\\gtruth_root\\topologies".format(method)
save_file = "D:\Text Phylogeny\Test Files\\Visualization\\WR_10-50_revised2_gtruth_root"
mat_file = "D:\Text Phylogeny\Test Files\WR_10-50_revised2\\DisMatrices\\{}".format(method)

res_filenames = list_files(res_file)

#tps = [res_filenames[i] for i in inds]
tps = get_names(save_file, res_file)

gen_graphs(res_file, save_file, mat_file, tps, integer=False)

#Visulaização com tamanhos:
import pydot

heuristic = "size_cost_max"

phyl_folder = "D:\Text Phylogeny\Test Files\paper-example-21-07-2015\Results\wu_et_al\\{}\\topologies".format(heuristic)
mat_folder = "D:\Text Phylogeny\Test Files\paper-example-21-07-2015\\DisMatrices\\wu_et_al"
size_folder = "D:\Text Phylogeny\Test Files\paper-example-21-07-2015\\tree-sizes"
save_path = "D:\Text Phylogeny\Test Files\Visualization\example_heuristics\\{}".format(heuristic)

max_lim = 1
min_lim = 0.2

phylogenies = list_files(phyl_folder)

for phyl_file in phylogenies:

    name = phyl_file.split(os.sep)[-1].replace(".tpres", "")

    topology = ast.literal_eval(read_file(phyl_file).split("\n")[1])
    sizes = np.array(ast.literal_eval(read_file(os.path.join(size_folder,name + ".txt"))))
    sizes = sizes/min(sizes) - 1
    sizes = sizes/max(sizes) *(max_lim - min_lim) + min_lim
    save_name = name + ".png"
    dis = np.loadtxt(os.path.join(mat_folder, name + ".dismat"), delimiter=';')
    phyl_vis(topology, dis, save_path, save_name, [10]*len(topology),
             True, sizes)
