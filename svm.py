# separar conjuntos de teste e treino
from __future__ import division
import shutil
import os
import random
import ast
from tphyl2 import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm
from scipy import sparse


def get_sizes(trees):

    sizes = dict()
    for tree in trees:

        size = tree.split(os.sep)[-1].replace(".txt", "").split(".")[-1]

        try:
            sizes[int(size)] += 1
        except KeyError:
            sizes[int(size)] = 1

    return sizes


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_train_set(base_folder, dest_folder):

    orig_phyl = os.path.join(base_folder, "Phylogenies")
    orig_trees = os.path.join(base_folder, "Trees")

    trees = list_files(orig_trees)
    size_dict = get_sizes(trees)

    for size in size_dict.keys():

        svm_folder = os.path.join(dest_folder, str(size))
        dest_phyl = os.path.join(svm_folder, "Phylogenies")
        dest_trees = os.path.join(svm_folder, "Trees")
        ensure_dir(svm_folder)
        ensure_dir(dest_phyl)
        ensure_dir(dest_trees)
        ensure_dir(os.path.join(svm_folder, "data", "base"))

        trees = list_files(orig_trees, "{}.txt".format(size))
        phylogenies = list_files(orig_phyl, "{}.phyl".format(size))
        phylogenies = random.sample(phylogenies, len(phylogenies))

        sample_size = int(len(phylogenies) / 2)
        training_examples = []

        for j in range(sample_size):

            phyl_name = phylogenies[j].split(os.sep)[-1].replace(".phyl", "")
            tree_path = os.path.join(orig_trees, phyl_name + ".txt")

            if tree_path in trees:

                save_name = phyl_name + ".phyl"
                shutil.copy(phylogenies[j], os.path.join(dest_phyl, save_name))

                save_name = phyl_name + ".txt"
                shutil.copy(tree_path, os.path.join(dest_trees, save_name))

                training_examples.append(phyl_name)

        write_file(svm_folder, "training_trees.txt", str(training_examples))


def sample_lists(list1, list2, sample_size):

    list1_shuf = []
    list2_shuf = []
    index_shuf = range(len(list1))
    index_shuf = random.sample(index_shuf, sample_size)

    for i in index_shuf:
        list1_shuf.append(list1[i])
        list2_shuf.append(list2[i])

    return [list1_shuf, list2_shuf]


def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z


def build_vocabulary(tree_folder, phyl_folder, sample_size="all", instance="single", char=False):

    vocabulary_word = {}
    vocabulary_char = {}
    examples = {}
    trees = list_files(tree_folder)
    phylogenies = list_files(phyl_folder)
    neg_examples = 2
    vocab_save_name = "vocabulary"
    ex_save_name = "examples"

    word_vectorizer = CountVectorizer(preprocessor=my_preprocessor,
                                      tokenizer=my_tokenizer,
                                      decode_error="replace",
                                      analyzer="word")

    char_vectorizer = CountVectorizer(ngram_range=(1, 5),
                                      preprocessor=my_preprocessor,
                                      tokenizer=my_tokenizer,
                                      decode_error="replace",
                                      analyzer="char")

    if sample_size == "all" and instance == "single":

        [trees, phylogenies] = sample_lists(trees, phylogenies, len(trees))

    elif isinstance(sample_size, int) and instance == "single":

        if sample_size > len(trees):
            [trees, phylogenies] = sample_lists(trees, phylogenies, len(trees))
        else:
            [trees, phylogenies] = sample_lists(
                trees, phylogenies, sample_size)
            vocab_save_name += "-{}-single".format(sample_size)
            ex_save_name += "-{}-single".format(sample_size)

    elif sample_size == "all" and isinstance(instance, int):

        [trees, phylogenies] = sample_lists(trees, phylogenies, len(trees))

    elif isinstance(sample_size, int) and isinstance(instance, int):

        if sample_size > len(trees):
            [trees, phylogenies] = sample_lists(trees, phylogenies, len(trees))
        else:
            [trees, phylogenies] = sample_lists(
                trees, phylogenies, sample_size)
            vocab_save_name += "-{}-{}".format(sample_size, instance)
            ex_save_name += "-{}-{}".format(sample_size, instance)

    else:
        raise ValueError(
            "Inputs should be string ('all' and 'single') or ints")

    for i, tree in enumerate(trees):

        tree_name = tree.split(os.sep)[-1]
        top = get_topology(phylogenies[i])
        content = read_file(tree).split("<\\tphyldoc>")

        root = find_root(top)
        negs = random.sample(range(len(top)), neg_examples)

        examples[tree_name] = [root] + negs

        corpus = [content[k] for k in examples[tree_name]]
        word_fitted = word_vectorizer.fit(corpus)
        temp_word_vocab = word_fitted.vocabulary_
        vocabulary_word = merge_two_dicts(temp_word_vocab, vocabulary_word)

        if char:
            char_fitted = char_vectorizer.fit(corpus)
            temp_char_vocab = char_fitted.vocabulary_
            vocabulary_char = merge_two_dicts(temp_char_vocab, vocabulary_char)

        print i, tree_name, len(vocabulary_word)
        print i, tree_name, len(vocabulary_char)

    vocabulary_word = get_correct_inds(vocabulary_word)
    vocabulary_char = get_correct_inds(vocabulary_char)
    svm_folder = os.sep.join(tree_folder.split(os.sep)[0:-1])

    write_file(os.path.join(svm_folder, "data", "base"),
               vocab_save_name + ".txt", str(vocabulary_word))
    write_file(os.path.join(svm_folder, "data"),
               ex_save_name + ".txt", str(examples))
    if char:
        write_file(os.path.join(svm_folder, "data", "base-char"),
                   vocab_save_name + ".txt", str(vocabulary_char))


def get_correct_inds(base_vocab):

    new_vocab = {}
    count = 0
    for term in base_vocab.keys():

        new_vocab[term] = count
        count += 1

    return new_vocab


def filter_stopwords(base_vocab):

    new_vocab = {}
    count = 0
    for term in base_vocab.keys():

        if term not in stopwords.words("english") + list(string.punctuation):

            new_vocab[term] = count
            count += 1
            print count, term

    return new_vocab


def filter_numerals(base_vocab):

    new_vocab = {}
    count = 0
    for term in base_vocab.keys():

        if term not in stopwords.words("english") + list(string.punctuation):

            if not any([digit in term for digit in list(string.digits)]):

                new_vocab[term] = count
                count += 1
                print count, term

    return new_vocab


def stemmed_vocab(base_vocab):

    new_vocab = {}
    count = 0
    st = LancasterStemmer()

    for term in base_vocab.keys():

        stemmed_word = st.stem(term)

        if stemmed_word not in new_vocab.keys():

            new_vocab[stemmed_word] = count
            count += 1
            print count, stemmed_word

    return new_vocab


def stem_corpus(corpus):

    st = LancasterStemmer()

    for i in range(len(corpus)):

        tokens = nltk.word_tokenize(corpus[i])
        new_tokens = []

        for element in tokens:
            new_tokens.append(st.stem(element))

        corpus[i] = " ".join(new_tokens)

    return corpus


def is_lexical_word(tag):

    lex_word_tags = ["NN", "VB", "RB", "JJ", "CD"]

    flag = False
    for target_tag in lex_word_tags:

        if tag.startswith(target_tag):
            flag = True
            break

    return flag


def is_grammatical_word(tag):

    gram_word_tags = ["DT", "WDT", "PDT", "IN", "PR", "CC", "UH"]

    flag = False
    for target_tag in gram_word_tags:

        if tag.startswith(target_tag):
            flag = True
            break

    return flag


def get_stat_features(text):

    sent_tokens = tokenized(text, 0)
    list_tokens = sent_tokens.join_sentences()
    text_string = sent_tokens.join_all()

    # number of sentences
    n_sents = max(sent_tokens.tokens.keys())

    # number of tokens
    n_tokens = len(list_tokens)

    # number of characters
    n_chars = len(text_string)

    if n_sents == 0 or n_tokens == 0:
        return []

    avg_sent_len = 0
    avg_tk_len = 0
    inf_load = 0
    gram_words = 0
    prop_nouns = 0
    prop_prepos = 0
    prop_pronn = 0
    prop_stpwrds = 0
    # loop over sentences
    for key in sent_tokens.tokens.keys():

        # average sentence length
        avg_sent_len += len(sent_tokens.tokens[key])

        # loop over tokens
        for element in sent_tokens.tokens[key]:

            # average token length
            avg_tk_len += len(element['word'])

            # information load
            if is_lexical_word(element['tag']):
                inf_load += 1

            # grammatical words
            if is_grammatical_word(element['tag']):
                gram_words += 1
            # lexical variety

            # lexical richness

            # proportion of sents without finite verbs
            # proportion of simple sents: only 1 finite verb
            # proportion of complex sents: more than 1 finite

            # nouns over tokens
            if element['tag'].startswith("NN"):
                prop_nouns += 1

            # prepositions over tokens
            if element['tag'].startswith("IN"):
                prop_prepos += 1

            # pronouns over tokens
            if element['tag'].startswith("PR"):
                prop_pronn += 1

            # stopwords over tokens
            if element['word'] in stopwords.words("english"):
                prop_stpwrds += 1

    # cohesion rate
    if gram_words != 0:
        cohesion = inf_load / gram_words
    else:
        cohesion = 0.5

    # normalizing and averaging
    avg_sent_len = avg_sent_len / n_sents
    avg_tk_len = avg_tk_len / n_tokens
    inf_load = inf_load / n_tokens
    prop_nouns = prop_nouns / n_tokens
    prop_prepos = prop_prepos / n_tokens
    prop_pronn = prop_pronn / n_tokens
    prop_stpwrds = prop_stpwrds / n_tokens

    features = [n_sents,
                n_tokens,
                n_chars,
                avg_sent_len,
                avg_tk_len,
                inf_load,
                prop_nouns,
                prop_prepos,
                prop_pronn,
                prop_stpwrds,
                cohesion]

    return features


def scale_linear_bycolumn(rawpoints, high=100.0, low=0.0):

    mins = np.min(rawpoints, axis=0)
    maxs = np.max(rawpoints, axis=0)
    rng = maxs - mins

    # avoid 0/0:
    rng = [x if x != 0 else 1 for x in rng]

    return high - (((high - low) * (maxs - rawpoints)) / rng)


def corpus_stat_features(corpus, *args):

    feats = []
    bad_cases = []
    for i in range(len(corpus)):

        ff = get_stat_features(corpus[i])
        feats.append(ff)
        if ff == []:
            bad_cases.append(i)

    if bad_cases != []:
        ok_cases = [feats[i] for i in range(len(feats)) if i not in bad_cases]
        means = np.mean(ok_cases, axis=0)
        for case in bad_cases:
            feats[case] = means

    normalized = scale_linear_bycolumn(feats, 1, -1).tolist()

    return np.matrix(normalized)


def tf_idf_word(corpus, vocabulary):

    word_vectorizer = CountVectorizer(vocabulary=vocabulary,
                                      preprocessor=my_preprocessor,
                                      tokenizer=my_tokenizer,
                                      decode_error="replace",
                                      analyzer="word")

    tfidf = TfidfTransformer(norm="l2")
    freq_term_matrix = word_vectorizer.fit_transform(corpus)

    tf_idf_matrix = tfidf.fit_transform(freq_term_matrix)

    return tf_idf_matrix.todense()


def tf_idf_wordnchar(corpus, vocabulary, char=False):

    word_vectorizer = CountVectorizer(vocabulary=vocabulary,
                                      preprocessor=my_preprocessor,
                                      tokenizer=my_tokenizer,
                                      decode_error="replace",
                                      analyzer="word")

    char_vectorizer = CountVectorizer(vocabulary=vocabulary,
                                      preprocessor=my_preprocessor,
                                      tokenizer=my_tokenizer,
                                      decode_error="replace",
                                      analyzer="char")

    tfidf = TfidfTransformer(norm="l2")
    word_matrix = word_vectorizer.fit_transform(corpus)
    char_matrix = char_vectorizer.fit_transform(corpus)
    #freq_term_matrix = hstack([word_matrix, char_matrix])

    w_inds = word_matrix.indices
    c_inds = char_matrix.indices

    w_ptrs = word_matrix.indptr
    c_ptrs = char_matrix.indptr

    repeats = [[]] * (len(w_ptrs) - 1)
    for i in range(len(w_ptrs) - 1):
        repeats[i] = [ind for ind in w_inds[w_ptrs[i]:w_ptrs[i+1]] if ind in c_inds[c_ptrs[i]:c_ptrs[i+1]]]

    freq_term_matrix = word_matrix + char_matrix
    for line in range(len(repeats)):
        for cell in repeats[line]:
            freq_term_matrix[line, cell] = freq_term_matrix[line, cell] / 2

    tf_idf_matrix = tfidf.fit_transform(freq_term_matrix)

    return tf_idf_matrix.todense()


def feature_extraction(folder, examples, vocabulary, stem=False, funct=tf_idf_word):

    feat_list = []
    label_list = []
    count = 0
    for ex in examples.keys():

        corpus = read_file(os.path.join(folder, ex)).split("<\\tphyldoc>")

        if stem:
            corpus = stem_corpus(corpus)

        feature_matrix = funct(corpus, vocabulary)

        feat_list.append(feature_matrix[examples[ex][0]])
        label_list.append(1)

        for i in range(1, len(examples[ex])):

            feat_list.append(feature_matrix[examples[ex][i]])
            label_list.append(0)

        count += 1
        print count, ex

    return {"labels": label_list, "features": feat_list}


def save_sparse_csr(filename, array):

    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):

    loader = np.load(filename)
    return sparse.csr_matrix((loader['data'], loader['indices'],
                              loader['indptr']), shape=loader['shape'])


def train_svm(labels, features, kernel_type="linear"):

    clf = svm.SVC(kernel=kernel_type)
    clf.fit(features, labels)

    return clf


def train_randforest(labels, features, n_trees=50, criterion="gini"):

    clf = RandomForestClassifier(n_estimators=n_trees,
                                 criterion=criterion)
    clf.fit(features, labels)

    return clf


def svm_root(classifier, corpus, vocabulary, stem=False, funct=tf_idf_word):

    if stem:
        corpus = stem_corpus(corpus)

    features = funct(corpus, vocabulary)

    root = 0
    root_score = -1e12
    for i in range(features.shape[0]):

        if classifier.decision_function(features[i])[0] > root_score:
            root_score = classifier.decision_function(features[i])[0]
            root = i

    return root


def randforest_root(classifier, corpus, vocabulary, stem=False, funct=tf_idf_word):

    if stem:
        corpus = stem_corpus(corpus)

    features = funct(corpus, vocabulary)

    root = 0
    root_score = 0
    for i in range(features.shape[0]):

        if classifier.predict_proba(features[i])[0][1] > root_score:
            root_score = classifier.predict_proba(features[i])[0][1]
            root = i

    return root


def get_model(svm_folder, model):

    clfs = {}
    for directory in list_directories(svm_folder):

        size = int(directory.split(os.sep)[-1])
        model_folder = os.path.join(directory, "models", model)
        clfs[size] = joblib.load(model_folder)

    return clfs


def get_model2(svm_folder, model, sizes):

    clfs = {}
    for size in sizes:

        model_folder = os.path.join(svm_folder, "models", model)
        clfs[size] = joblib.load(model_folder)

    return clfs


def get_vocab(svm_folder, vocab):

    vocabulary = {}
    for directory in list_directories(svm_folder):

        size = int(directory.split(os.sep)[-1])
        vocab_file = os.path.join(directory, "data", vocab, "vocabulary.txt")
        vocabulary[size] = ast.literal_eval(read_file(vocab_file))

    return vocabulary


def get_vocab2(svm_folder, vocab, sizes):

    vocabulary = {}
    for size in sizes:

        vocab_file = os.path.join(svm_folder, "data", vocab, "vocabulary.txt")
        vocabulary[size] = ast.literal_eval(read_file(vocab_file))

    return vocabulary


def get_vocab3(svm_folder, vocab, sizes, sample_size, instance):

    vocabulary = {}
    for size in sizes:

        vocab_file = os.path.join(svm_folder, "data", vocab,
                                  "vocabulary-{}-{}.txt".format(sample_size, instance))
        vocabulary[size] = ast.literal_eval(read_file(vocab_file))

    return vocabulary


def get_blacklist(svm_folder):

    blacklist = []
    for directory in list_directories(svm_folder):

        blk_file = os.path.join(directory, "training_trees.txt")
        blacklist.extend(ast.literal_eval(read_file(blk_file)))

    return blacklist


def get_blacklist2(svm_folder):

    blk_file = os.path.join(svm_folder, "training_trees.txt")
    blacklist = ast.literal_eval(read_file(blk_file))

    return blacklist


def svm_evaluate(base_folder, svm_folder, method, model, vocab_name, sizes, size_split=True, add_name="", stem=False, funct=tf_idf_word, direct_vocab=False, root_funct=svm_root):

    phyl_folder = os.path.join(base_folder, "Phylogenies")
    tree_folder = os.path.join(base_folder, "Trees")
    res_folder = os.path.join(base_folder, "Results")
    mat_folder = os.path.join(base_folder, "DisMatrices")
    ensure_dir(res_folder)

    if size_split:
        clfs = get_model(svm_folder, model + ".pkl")
        blacklist = get_blacklist(svm_folder)
        if direct_vocab:
            vocabulary = vocab_name
        else:
            vocabulary = get_vocab(svm_folder, vocab_name)
    else:
        clfs = get_model2(svm_folder, model + ".pkl", sizes)
        blacklist = get_blacklist2(svm_folder)
        if direct_vocab:
            vocabulary = vocab_name
        else:
            vocabulary = get_vocab2(svm_folder, vocab_name, sizes)

    res_list = {}

    for n in sizes:

        res_list[n] = []

        dismat_list = list_files(
            os.path.join(mat_folder, method), ".{}.dismat".format(n))

        for matrix_file in dismat_list:

            name = matrix_file.split(os.sep)[-1].replace(".dismat", "")

            if name not in blacklist:

                orig = get_topology(os.path.join(phyl_folder,
                                                 "{}.phyl".format(name)))

                [undirected, dis] = kruskal(matrix_file)

                corpus = read_file(
                    os.path.join(tree_folder, name + ".txt")).split("<\\tphyldoc>")

                root = root_funct(
                    clfs[n], corpus, vocabulary[n], stem=stem, funct=funct)

                recon = build_topology(root, undirected)

                write_file(os.path.join(res_folder, method,
                                        model + add_name, "topologies"),
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
                  os.path.join(res_folder, method, model + add_name),
                  "{}_mean.txt".format(model))

    return res_list


def join_svm(path):

    directories = list_directories(path)

    res_dict = {}

    numbs = []

    for directory in directories:

        numbs.append(int(directory.split("-")[-1]))

    numbs = sorted(numbs)

    new_dirs = [""] * len(directories)

    for i, n in enumerate(numbs):
        for d in directories:
            if d.split("-")[-1] == str(n):
                new_dirs[i] = d

    for directory in new_dirs:

        prop = int(directory.split("-")[-1])

        try:
            res_dict["Instance"]
        except KeyError:
            res_dict["Instance"] = []

        res_dict["Instance"].append(prop)

        results_path = directory

        name = directory.split(os.sep)[-1]

        results_file = os.path.join(results_path, name + "_mean.txt")

        res_table = [line.split("\t")
                     for line in read_file(results_file).split("\n")]

        res_table.pop(-1)

        for i, measure in enumerate(res_table[0][1:]):

            measure = measure.replace(" ", "")

            try:

                res_dict[measure]

            except KeyError:

                res_dict[measure] = {}

            for element in res_table[1:]:

                try:

                    res_dict[measure][int(element[0])]

                except KeyError:

                    res_dict[measure][int(element[0])] = []

                res_dict[measure][int(element[0])].append(
                    float(element[i + 1]))

    return res_dict


def get_means(joined_dict):

    means_dict = {}

    for key1 in joined_dict.keys():

        if key1 != "Instance":

            means_dict[key1] = {}

            for key2 in joined_dict[key1].keys():

                means_dict[key1][key2] = np.mean(joined_dict[key1][key2])

    return means_dict


def format_means(means_dict):

    sizes = sorted(means_dict[means_dict.keys()[0]].keys())
    procesed_dict = {}

    for size in sizes:

        procesed_dict[size] = [[means_dict["Ind.Edges"][size],
                                means_dict["DirectEdges"][size],
                                means_dict["Leaves"][size],
                                means_dict["Roots"][size],
                                means_dict["Ancestry"][size],
                                means_dict["Depth"][size],
                                means_dict["1-Depth"][size],
                                means_dict["2-Depth"][size],
                                means_dict["3-Depth"][size],
                                means_dict["1-Set"][size],
                                means_dict["2-Set"][size],
                                means_dict["3-Set"][size]]]

    return procesed_dict


def get_vartrain_series(folder, prefix, sample_sizes):

    res_dict = {}
    res_dict["s_sizes"] = sample_sizes
    res_dict['avg'] = []
    res_dict['sd'] = []
    for sample_size in sample_sizes:

        res_fpath = os.path.join(
            folder, prefix + str(sample_size), "means.txt")
        dframe = pd.DataFrame.from_csv(res_fpath, sep="\t")
        roots = dframe["     Roots     "]
        size_avg = 0
        sd_vector = []
        for key in sorted(roots.keys()):
            try:
                res_dict[key].append(roots[key])
            except KeyError:
                res_dict[key] = [roots[key]]

            size_avg += roots[key]
            sd_vector.append(roots[key])

        res_dict['avg'].append(size_avg / len(roots.keys()))
        res_dict['sd'].append(np.std(sd_vector))
    
    return res_dict


def get_randforest_series(folder, criterion, n_trees):

    res_dict = {}
    res_dict["n_trees"] = n_trees
    res_dict['avg'] = []
    res_dict['sd'] = []
    for n_tree in n_trees:

        res_fpath = os.path.join(folder,
                                 "random-forest-{}-{}-stemmed".format(n_tree, criterion), 
                                 "random-forest-{}-{}-stemmed_mean.txt".format(n_tree, criterion))
        dframe = pd.DataFrame.from_csv(res_fpath, sep="\t")
        roots = dframe["     Roots     "]
        size_avg = 0
        sd_vector = []
        for key in sorted(roots.keys()):
            try:
                res_dict[key].append(roots[key])
            except KeyError:
                res_dict[key] = [roots[key]]

            size_avg += roots[key]
            sd_vector.append(roots[key])

        res_dict['avg'].append(size_avg / len(roots.keys()))
        res_dict['sd'].append(np.std(sd_vector))
    
    return res_dict


def get_svm_scores(base_folder, classifier, vocabulary):

    tree_folder = os.path.join(base_folder, "Trees")

    dest_folder = os.path.join(base_folder, "tree-scores")

    trees = list_files(tree_folder)

    for tree_file in trees:

        corpus = read_file(tree_file).split("\n<\\tphyldoc>\n")

        corpus = stem_corpus(corpus)

        features = tf_idf_word(corpus, vocabulary)

        sizes = [classifier.decision_function(features[i])[0] for i in range(len(corpus))]

        name = tree_file.split(os.sep)[-1]

        write_file(dest_folder, name, str(sizes))


def max_svm_score(matrix_file):

    [tree, dis] = kruskal(matrix_file)

    name = matrix_file.split(os.sep)[-1].replace(".dismat", "")

    base_file = "{}".format(os.sep).join(matrix_file.split(os.sep)[0:-3])

    score_file = read_file(os.path.join(base_file, "tree-scores", name + ".txt"))

    scores = - np.array(ast.literal_eval(score_file))

    n = len(dis)

    cost_list = []

    for node in range(n):
        topology = build_topology(node, tree)
        cost_list.append(size_cost(topology, scores))

    max_cost = max(cost_list)

    root = random.sample([i for i, c in enumerate(cost_list) if c == max_cost], 1)[0]

    return build_topology(root, tree)
