from tphyl2 import *
import matplotlib.pyplot as plt
import itertools


def plot_joined(res_dict, measure, rng, location, llabel="tree size", save=False, path="", dpi_multiplier=1):

    size_dict = res_dict[measure]
    x_values = res_dict["Proportion"]
    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    markers = ['o', 'v', '^', '8', 's', 'x', '^']
    lstyle = ['-', '--', ':', '-.', '-', ':', '--']
    #lstyle = ['','','','','','']

    fig = plt.figure(figsize=(16, 9))

    ax = fig.add_subplot(111)

    for i, key in enumerate(sorted(size_dict.keys())):

        ax.plot(x_values, size_dict[key], marker=markers[i], linestyle=lstyle[
                i], color=color[i], label="{} = {}".format(llabel, key))

    my_dpi = 96
    plt.ylim(rng)
    plt.xlabel('Proportion of Change [%]')
    plt.ylabel('Score')
    plt.title(measure, fontsize=14)
    ax.legend(loc=location, prop={'size': 14})

    if not save:
        fig.show()
    else:
        if not os.path.exists(path):
            os.makedirs(path)
        fig.savefig(os.path.join(path, measure + ".png"), dpi=dpi_multiplier * my_dpi)
    plt.close()
