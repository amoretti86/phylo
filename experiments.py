import operator as op
from functools import reduce
import random
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from copy import deepcopy
from gusfield import *
import numpy as np
from gusfield import Phylogeny
import pdb
from sir_genealogy import SIR
import matplotlib.pyplot as plt


def lineplt(d,color):
    for key, val in d.items():
        mean = np.mean(val, axis=0)
        std = np.std(val, axis=0)
        plt.plot(mean, label=key, linewidth=1,c=color)
        plt.fill_between(np.arange(len(mean)), mean - std, mean + std, alpha=0.2,color=color)


M_5 = np.array([[1, 1, 0, 0, 0], [0, 0, 1, 0, 0], [1, 1, 0, 0, 1], [0, 0, 1, 1, 0], [0, 1, 0, 0, 0]])

M_10 = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                 [1, 0, 0, 1, 1, 1, 1, 1, 1, 0],
                 [0, 1, 0, 1, 1, 1, 1, 0, 0, 0],
                 [0, 1, 1, 1, 1, 0, 0, 1, 0, 0],
                 [0, 1, 0, 1, 1, 1, 0, 1, 0, 0],
                 [1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                 [1, 0, 1, 1, 0, 0, 1, 0, 1, 0],
                 [0, 0, 0, 0, 1, 0, 1, 1, 1, 1],
                 [1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
                 [0, 0, 1, 1, 1, 0, 1, 1, 0, 0]])

M_20 = np.array([[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0],
                 [0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0],
                 [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                 [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0],
                 [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1],
                 [1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                 [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0],
                 [0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1],
                 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                 [0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                 [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                 [0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0],
                 [0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                 [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1],
                 [0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
                 [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1]])


testcases = [M_5, M_10, M_20]
MC_samples = [100,500,1000,3000,5000,10000,15000]

testcases = [M_20]


for case in testcases:

    sis = []
    sir = []

    phylo = Phylogeny(case)
    print(phylo.M)
    print(phylo.K)
    graph = phylo.main_phylogeny()

    fig = plt.figure()
    for trial in range(20):
        print("trial %i" %trial)
        gprobs_F = []
        gprobs_T = []
        for M in MC_samples:
            sampler = SIR(case)
            jump_chains, coalescent_probs, theGraph, genealogy_probs_F = sampler.main_sampling(graph, K=M, showing=False, resampling=False)
            gprobs_F.append(np.average(1./genealogy_probs_F))
            plt.plot(gprobs_F, alpha=0.1, c='r')
            jump_chains, coalescent_probs, theGraph, genealogy_probs_T = sampler.main_sampling(graph, K=M, showing=False, resampling=True)
            gprobs_T.append(np.average(1./genealogy_probs_T))
            plt.plot(gprobs_T, alpha=0.1, c='b')
        sis.append(gprobs_F)
        sir.append(gprobs_T)
    plt.xticks(np.arange(len(MC_samples)),MC_samples)
    plt.xlabel("N = # MC Samples")
    plt.ylabel("$\hat{|G^{K}|}$")
    plt.title("Convergence of $\hat{|G|}$")
    plt.show()

    sis_results = {'sis' : sis}
    sir_results = {'sir' : sir}

    plt.figure()
    lineplt(sis_results, 'red')
    lineplt(sir_results, 'blue')
    plt.xticks(np.arange(len(MC_samples)),MC_samples)
    plt.xlabel("N = # MC Samples")
    plt.ylabel("$\hat{|G^{K}|}$")
    plt.title("Convergence of $\hat{|G|}$")
    plt.legend(frameon=False, loc='best', fontsize=11)
    plt.savefig("ConvergenceForTwentyTaxa.png")
    plt.show()