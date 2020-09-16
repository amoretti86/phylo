import operator as op
import random
from functools import reduce
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from gusfield import *
from gusfield import Phylogeny
import pdb
from sir_genealogy import SIR
import msprime


def ncr(n, r):
    # Compute combinatorial term n choose r
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer/denom

def generate_M(N, Ne=1000):
    length = 0
    for i in range(2,N+1):
        length += i*np.random.exponential(ncr(i,2))
    mu = 5/4/Ne/length
    tree_sequence = msprime.simulate(sample_size=N, Ne=Ne, 
                                 length=length, mutation_rate=mu, random_seed=1)
    M = tree_sequence.genotype_matrix()
    return M.T

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

M_20 = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
                 [1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
                 [1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0],
                 [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0],
                 [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1],
                 [1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0],
                 [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0],
                 [0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1],
                 [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1],
                 [0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                 [1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1],
                 [0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0],
                 [1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                 [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1],
                 [1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0],
                 [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1]])

#testcases = [M_5, M_10, M_20]

M = generate_M(50)

testcases = [M]
MC_samples = [100,500,1000,3000,5000,10000,15000]

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