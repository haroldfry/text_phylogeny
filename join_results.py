from tphyl2 import *


def join_results(path, method, heuristic):

    dirs = list_directories(path, "")

    directories = []
    for i, d in enumerate(dirs):
        if not (d.endswith("mixed") or d.endswith("SVM") or d.endswith("SVM2")):
            directories.append(d)

    res_dict = {}

    numbs = []

    for directory in directories:

        numbs.append(int(directory.split("_")[-1]))

    numbs = sorted(numbs)

    new_dirs = [""] * len(directories)

    for i, n in enumerate(numbs):
        for d in directories:
            if d.split("_")[-1] == str(n):
                new_dirs[i] = d

    for directory in new_dirs:

        prop = int(directory.split("_")[-1])

        try:
            res_dict["Proportion"]
        except KeyError:
            res_dict["Proportion"] = []

        res_dict["Proportion"].append(prop)

        results_path = os.path.join(directory, "Results", method, heuristic)

        results_file = os.path.join(results_path, heuristic + "_mean.txt")

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


def write_joined(res_dict, path, method, heuristic):

    res_string = ""

    for measure in res_dict.keys():

        if not measure == "Proportion":

            res_string += measure + "\n"

            res_string += "{0:^15}\t".format("Tree Size")

            for element in res_dict["Proportion"]:

                res_string += "{0:^15}\t".format(str(element))

            res_string += "\n"

            for tree_size in sorted(res_dict[measure].keys()):

                res_string += "{0:^15}\t".format(str(tree_size))

                for record in res_dict[measure][tree_size]:

                    res_string += "{0:^15}\t".format(str(record))

                res_string += "\n"

    print res_string

    write_file(path, method + "_" + heuristic + ".txt", res_string)


def read_joined(path):

    string = read_file(path)

    split = string.split("\n")

    res_dict = {}

    for i, element in enumerate(split):

        split[i] = [s for s in element.split("\t") if s is not ""]

    split = [s for s in split if len(s) > 0]

    for i, element in enumerate(split):

        if len(element) == 1:

            measure = element[0]

            res_dict[measure] = {}

            res_dict["Proportion"] = [int(s) for s in split[i + 1][1:]]

        if not element[0].strip() == "Tree Size" and len(element) > 1:

            res_dict[measure][int(element[0])] = [float(s)
                                                  for s in element[1:]]

    return res_dict


def get_tf_idf_means(path, method, heuristic):

    datasets = list_directories(path)

    experiment_dict = {}

    for dataset in datasets:

        experiments = [p for p in list_directories(os.path.join(dataset, "Results")) if
                       p.split(os.sep)[-1].startswith(method)]

        for experiment in experiments:

            r = join_results(path, experiment.split(os.sep)[-1], heuristic)

            n = int(experiment[-5])

            try:
                experiment_dict["Proportion"]
            except KeyError:
                experiment_dict["Proportion"] = r["Proportion"]

            for key in r.keys():

                if not key == "Proportion":

                    keysum = None

                    for key2 in r[key].keys():

                        if keysum is None:

                            keysum = np.array(r[key][key2])

                        else:

                            keysum += np.array(r[key][key2])

                    mean = list(keysum/len(r[key].keys()))

                    try:
                        experiment_dict[key]
                    except KeyError:
                        experiment_dict[key] = {}

                    experiment_dict[key][n] = mean

    return experiment_dict



