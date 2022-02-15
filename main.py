import networkx as nx
import numpy as np
import matplotlib.pylab as plt
from os import listdir
from os.path import basename
from collections import Counter
import seaborn as sb
import gc
from class_link_pred import *
from random import choice

PATH = "../fb100/"


def load( name ):
    return nx.read_graphml(PATH + name + ".graphml")


def from_dict_to_adj_matrix( G ):
    N = len(G)
    adj_matrix = np.zeros((N, N))
    for v in G.keys():
        for u in G [v]:
            adj_matrix [eval(u), eval(v)] = 1
    return adj_matrix


def global_clustering( A ):
    # print(A)
    m = 0.5 * np.trace(np.linalg.matrix_power(A, 3))
    B = np.linalg.matrix_power(A, 2)
    n = 0.5 * (np.sum(B) - np.trace(B))
    return m / n


def plot_degree_dist( G, name ):
    degrees = [G.degree(n) for n in G.nodes()]
    degree_count = Counter(degrees)
    x, y = zip(*degree_count.items())
    plt.figure(1)

    # prep axes
    plt.xlabel('degree')
    # plt.xscale('log')
    plt.xlim(0, max(x) + 1)

    plt.ylabel('log-frequency')
    plt.yscale('log')
    plt.ylim(0, max(y) + 1)
    # do plot

    plt.title("Distribution of the degrees inside " + name + "'s students network - Facebook 100")
    plt.scatter(x, y, marker='P')
    plt.show()


def plot_local_clust_degree( G, name ):
    degrees = [G.degree(n) for n in G.nodes()]
    local_clust_nodes = nx.clustering(G).values()
    x, y = (degrees, local_clust_nodes)
    plt.figure(1)

    # prep axes
    plt.xlabel('log - degree')
    plt.xscale('log')
    plt.xlim(0, max(x) + 1)

    plt.ylabel('local clustering')
    # plt.yscale('log')
    plt.ylim(-0.1, 1.1)
    # do plot

    plt.title("Degree versus local clustering in " + name + "'s students network - Facebook 100")
    plt.scatter(x, y, marker='P')
    plt.show()


def plot_assort_size( l, attr ):
    size = []
    assort = []
    for file, i in zip(l, range(100)):
        G = nx.read_graphml(PATH + file)
        size.append(G.number_of_nodes())
        assort.append(nx.attribute_assortativity_coefficient(G, attr))
        del G
        print(i)
        gc.collect()
    x, y = (size, assort)
    plt.figure(1)

    # prep axes
    plt.xlabel('log - size of network')
    plt.xscale('log')
    plt.xlim(min(x) - 1, max(x) + 1)

    plt.ylabel('Assortativity')
    # plt.yscale('log')
    try:
        plt.ylim(-0.1, max(y) + 0.1)
    except:
        plt.ylim(-0.1, 1)
    # do plot

    plt.title("Assortativity according to the attribute " + attr + " in Facebook 100 dataset")
    plt.scatter(x, y, marker='P')
    plt.axhline(y=0, color='r', linestyle='dotted')
    plt.show()
    arr = np.array(y)
    sb.set_style('whitegrid')
    plt.axvline(x=0, color='r', linestyle='dotted')
    sb.kdeplot(arr)
    plt.show()


def plot_assort_degree_size( l ):
    size = []
    assort = []
    for file, i in zip(l, range(100)):
        G = nx.read_graphml(PATH + file)
        size.append(G.number_of_nodes())
        assort.append(nx.degree_assortativity_coefficient(G))
        del G
        print(i)
        gc.collect()
    x, y = (size, assort)
    plt.figure(1)

    # prep axes
    plt.xlabel('log - size of network')
    plt.xscale('log')
    plt.xlim(min(x) - 1, max(x) + 1)

    plt.ylabel('Assortativity')
    # plt.yscale('log')
    try:
        plt.ylim(-0.1, max(y) + 0.1)
    except:
        plt.ylim(-0.1, 1)
    # do plot

    plt.title("Assortativity according to the attribute vertex degree in Facebook 100 dataset")
    plt.scatter(x, y, marker='P')
    plt.axhline(y=0, color='r', linestyle='dotted')
    plt.show()
    arr = np.array(y)
    sb.set_style('whitegrid')
    plt.axvline(x=0, color='r', linestyle='dotted')
    sb.kdeplot(arr)
    plt.show()


def remove_frac_edges( G, frac ):
    n = G.number_of_edges()
    l = []
    for i in range(int(n * frac)):
        random_pick = choice([e for e in G.edges()])
        l.append(random_pick)
        G.remove_edge(*random_pick)
    return G, l


def evaluate_predictors( G_int, n, frac ):
    G = G_int
    H, removed_edges = remove_frac_edges(G, frac)
    CN = CommonNeighbors(H)
    J = Jaccard(H)
    AA = AdamicAdar(H)
    predicted_edgesCN = {}
    predicted_edgesJ = {}
    predicted_edgesAA = {}
    for i in nx.non_edges(H):
        (a, b) = i
        scoreCN = CN.evaluate(a, b)
        predicted_edgesCN [(a, b)] = scoreCN
        scoreJ = J.evaluate(a, b)
        predicted_edgesJ [(a, b)] = scoreJ
        scoreAA = AA.evaluate(a, b)
        predicted_edgesAA [(a, b)] = scoreAA
    predicted_edges_ord_CN = dict(sorted(predicted_edgesCN.items(), key=lambda item: -item [1]))
    predicted_edges_ord_J = dict(sorted(predicted_edgesJ.items(), key=lambda item: -item [1]))
    predicted_edges_ord_AA = dict(sorted(predicted_edgesAA.items(), key=lambda item: -item [1]))

    N_cut = list(predicted_edges_ord_CN) [:n]
    scoreCN = 0
    for (i, j) in N_cut:
        if (i, j) in removed_edges or (j, i) in removed_edges:
            scoreCN += 1

    N_cut = list(predicted_edges_ord_J) [:n]
    scoreJ = 0
    for (i, j) in N_cut:
        if (i, j) in removed_edges or (j, i) in removed_edges:
            scoreJ += 1

    N_cut = list(predicted_edges_ord_AA) [:n]
    scoreAA = 0
    for (i, j) in N_cut:
        if (i, j) in removed_edges or (j, i) in removed_edges:
            scoreAA += 1

    return (scoreCN, scoreJ, scoreAA)


def partial_attribute_wipeout( G, fraction, attribute ):
    g = G.copy()
    nb_nodes_to_change = int(g.number_of_nodes() * fraction)
    changed_nodes = []
    for i in range(nb_nodes_to_change):
        nodes = []
        for j in g.nodes(data=True):
            if j [1] [attribute] != -1:
                nodes.append(j)

        chosen_node = choice(nodes)
        changed_nodes.append((chosen_node [0], chosen_node [1].copy()))
        chosen_node [1] [attribute] = -1
        nx.set_node_attributes(g, {chosen_node [0]: -1}, attribute)

    return g, changed_nodes


def lpa( G, attribute, max_iter=100 ):
    labels = {n: i for n, i in nx.classes.function.get_node_attributes(G, attribute).items()}
    cont = True
    n_iter = 0

    while cont and n_iter < max_iter:
        nodes = list(G.nodes())
        cont = False
        nodes = np.random.permutation(nodes)
        n_iter += 1

        for v in nodes:

            if not list(G.neighbors(v)):
                continue

            label_freq = Counter()
            for u in G.neighbors(v):
                label_freq.update({labels [u]: 1})

            max_freq = max(label_freq.values())

            best_labels = []
            for label, freq in label_freq.items():
                if freq == max_freq:
                    best_labels.append(label)

            if labels [v] not in best_labels:
                c = choice(best_labels)
                if c != -1:
                    labels [v] = c
                    nx.set_node_attributes(G, {v: labels [v]}, attribute)
                cont = True
    return labels


def compute_accuracy_score_lpa( G, attribute, nmax ):
    fractions = [0.1, 0.2, 0.3]
    for f in fractions:
        G_altered, altered_nodes = partial_attribute_wipeout(G, f, attribute)
        lpa(G_altered, attribute, nmax)
        score = 0
        for i in altered_nodes:
            if G_altered.nodes(data=True) [i [0]] [attribute] == i [1] [attribute]:
                score += 1
        score = score / len(altered_nodes)
        print("Accuracy of", attribute, "attribute predictions with", f * 100, "% of removed labels:", score)


def partie2():
    Caltech36 = load("Caltech36")
    MIT8 = load("MIT8")
    John55 = load("Johns Hopkins55")
    question2a(Caltech36, MIT8, John55)
    question2b(Caltech36, MIT8, John55)
    question2c(Caltech36, MIT8, John55)


def question2a( Caltech36, MIT8, John55 ):
    plot_degree_dist(Caltech36, "Caltech")
    plot_degree_dist(MIT8, "MIT")
    plot_degree_dist(John55, "John Hopkins")


def question2b( Caltech36, MIT8, John55 ):
    print("Caltech's GCC: ", global_clustering(from_dict_to_adj_matrix(nx.convert.to_dict_of_lists(Caltech36))))
    print("Caltech's average clustering: ", nx.algorithms.average_clustering(Caltech36))
    print("Caltech's edge density: ", nx.density(Caltech36))
    print("MIT's GCC: ", global_clustering(from_dict_to_adj_matrix(nx.convert.to_dict_of_dicts(MIT8))))
    print("MIT's average clustering: ", nx.algorithms.average_clustering(MIT8))
    print("MIT's edge density: ", nx.density(MIT8))
    print("John Hopkinks' GCC: ", global_clustering(from_dict_to_adj_matrix(nx.convert.to_dict_of_dicts(John55))))
    print("John Hopkinks' average clustering: ", nx.algorithms.average_clustering(John55))
    print("John Hopkinks' edge density: ", nx.density(John55))


def question2c( Caltech36, MIT8, John55 ):
    plot_local_clust_degree(Caltech36, "Caltech")
    plot_local_clust_degree(MIT8, "MIT")
    plot_local_clust_degree(John55, "John Hopkins")


def question3a():
    l = listdir(PATH)
    plot_assort_size(l, 'student_fac')
    plot_assort_size(l, 'major_index')
    plot_assort_degree_size(l)
    plot_assort_size(l, 'dorm')


def question4():
    l = ['Caltech36.graphml', 'Reed98.graphml', 'Haverford76.graphml', 'Simmons81.graphml', 'Swarthmore42.graphml',
         'Amherst41.graphml', 'Bowdoin47.graphml', 'Hamilton46.graphml', 'Trinity100.graphml', 'USFCA72.graphml',
         'Williams40.graphml']
    for frac in [0.05, 0.1, 0.15, 0.2]:
        for f in l:
            G = nx.read_graphml(PATH + f)
            scoreCN, scoreJ, scoreAA = evaluate_predictors(G, 500, frac)
            print(basename(f), " with n = 500 frac = ", frac)
            print("CN: ", scoreCN)
            print("J: ", scoreJ)
            print("AA: ", scoreAA)


def question5():
    G = nx.read_graphml(PATH + "Johns Hopkins55.graphml")
    compute_accuracy_score_lpa(G, "dorm", 100)
    compute_accuracy_score_lpa(G, "major_index", 100)
    compute_accuracy_score_lpa(G, "year", 100)
    compute_accuracy_score_lpa(G, "gender", 100)


if __name__ == '__main__':
    question5()
