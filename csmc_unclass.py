"""
An implementation of the Combinatorial Sequential Monte Carlo algorithm
for Phylogenetic Inference
"""

import numpy as np
import random
from scipy.stats import lognorm
import scipy.linalg as spl
from scipy.linalg import expm
import matplotlib.pyplot as plt
import pandas as pd
import pdb
from copy import deepcopy
from functools import reduce
import operator as op
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout


class Node:
    """
    Defines a Node object for visualizing phylogeny
    """

    def __init__(self, data):
        # Subedges must be a list of edges (or empty list)
        self.data = data
        self.subnodes = []
        self.parent = None
        self.edge_before = ''

    def del_subnodes(self, d):
        for n in self.subnodes:
            if n.data == d:
                self.subnodes.remove(n)

    def print_node(self):
        s = 'Node ' + self.data + ' with subnodes: '
        for n in self.subnodes:
            s += n.data + ' '
        s += '; edge before is ' + self.edge_before
        print(s)

class Graph:
    """
    Defines a Graph object for visualizing phylogeny
    """

    def __init__(self):
        self.node_dict = {}
        self.num_nodes = 0

    def add_node(self, node):
        self.num_nodes += 1
        self.node_dict[node.data] = node
        return node

    def get_node(self, data):
        if data in self.node_dict:
            return self.node_dict[data]
        else:
            return None

    def del_node(self, data):
        try: 
            for d in self.node_dict:
                node = self.get_node(d)
                node.del_subnodes(data)
            del self.node_dict[data]
            self.num_nodes -= 1
            return None
        except KeyError:
            raise Exception("Node %s does not exist" % data)

    def contains(self, data):
        return data in self.node_dict

    def get_nodes(self):
        return list(self.node_dict.values())

    def get_nodes_data(self):
        return list(self.node_dict.keys())

    # For visualizations
    def build_nx_graph(self):
        # Add nodes and edges to build graph
        G_nx = nx.DiGraph()
        for key in self.get_nodes_data():
            G_nx.add_node(key)
        for node in self.get_nodes():
            for subnode in node.subnodes:
                G_nx.add_edge(node.data, subnode.data)
        return G_nx

    def draw(self, prob):
        # plotting tools. Not used here.
        G_nx = self.build_nx_graph()
        plt.figure(figsize=(10,10))
        pos = nx.kamada_kawai_layout(G_nx)
        nx.draw_networkx(G_nx, pos=pos, with_labels=True, fontsize=4,width=3.8,node_color='r',edge_color='brown')
        plt.title("Sampled Geneaology", fontsize=14)
        plt.xlabel("Prob %1.5f " % prob)
        plt.show()

    def __iter__(self):
        return iter(self.node_dict.values())


#@staticmethod
def ncr(n, r):
    # Compute combinatorial term n choose r
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer/denom

#@staticmethod
def sort_string(s):
    lst = s.split('+')
    lst = sorted(lst)
    result = '+'.join(lst)
    return result

class CSMC:
    """
    CSMC takes as input a dictionary (datadict) with two keys:
     taxa: a list of n strings denoting taxa
     genome_NxSxA: a 3 tensor of genomes for the n taxa one hot encoded
    """

    def __init__(self, datadict):
        self.n = len(datadict['taxa'])
        self.taxa = datadict['taxa']
        self.genome_NxSxA = datadict['genome']
        self.s = len(self.genome_NxSxA[0])
        self.Qmatrix = np.array([[-3.,1.,1.,1.],
                                 [1.,-3.,1.,1.],
                                 [1.,1.,-3.,1.],
                                 [1.,1.,1.,-3.]])
        self.Pmatrix = spl.expm(self.Qmatrix)
        self.state_prob = np.ones(self.Qmatrix.shape[0])/self.Qmatrix.shape[0]

    def build_tree(self, particle1, particle2, particle_coalesced, j):
        """
        Updates two lists graph_repn_data_K and graph_repn_nodes_K
        that are needed to plot a phylogeny using the Graph object
        """
        #pdb.set_trace()
        n3 = Node(particle_coalesced)
        if particle1 not in self.graph_repn_data_K[j]:
            n1 = Node(particle1)
            self.graph_repn_data_K[j].append(particle1)
            self.graph_repn_nodes_K[j].append(n1)
        else:
            n1 = self.graph_repn_nodes_K[j][self.graph_repn_data_K[j].index(particle1)]
        n1.parent = n3
        n3.subnodes.append(n1)
        if particle2 not in self.graph_repn_data_K[j]:
            n2 = Node(particle2)
            self.graph_repn_data_K[j].append(particle2)
            self.graph_repn_nodes_K[j].append(n2)
        else:
            n2 = self.graph_repn_nodes_K[j][self.graph_repn_data_K[j].index(particle2)]
        n2.parent = n3
        n3.subnodes.append(n2)
        self.graph_repn_data_K[j].append(particle_coalesced)
        self.graph_repn_nodes_K[j].append(n3)

    def tree_to_graph(self, G, root_node):
        """
        Constructs a graph for visualizing phylogeny
        """
        if root_node.subnodes == []:
            G.add_node(root_node)
            return G

        G.add_node(root_node)

        for n in root_node.subnodes:
            G = self.tree_to_graph(G, n)

        return G

    def resample(self, weights, op_seqs, i):
        """
        Resamples partial states (particles) based on importance weights
        """
        #pdb.set_trace()
        K = weights.shape[0]
        importance_weights = np.exp(weights[:,i])
        norm = np.sum(importance_weights)
        indices = np.random.choice(K, K, p = importance_weights/norm, replace=True)
        op_seqs = np.array(op_seqs)[indices]
        return op_seqs

    def conditional_likelihood(self, left_data, right_data, left_branch, right_branch):
        #pdb.set_trace()
        # Computes the conditional likelihood using the formula above
        # Matrix exponentiation here is probably inefficient
        left_Pmatrix = spl.expm(self.Qmatrix * left_branch)
        right_Pmatrix = spl.expm(self.Qmatrix * right_branch)
        left_prob = np.dot(left_data, left_Pmatrix)
        right_prob = np.dot(right_data, right_Pmatrix)
        likelihood = np.multiply(left_prob, right_prob)
        return likelihood

    def compute_tree_likelihood(self, onenode_data):
        #pdb.set_trace()
        # Assumes that each node's data is already updated before
        tree_likelihood = np.dot(self.state_prob, onenode_data.T)
        loglik = np.sum(np.log(tree_likelihood))
        return loglik

    def overcounting_correct(self, op_seq_cur, t):
        N = self.n
        rho = 0
        for i in range(N+1, N+1+t+1):
            if i not in op_seq_cur:
                rho += 1
        return 1/rho

    def get_tree_prob(self, op_seqs, weights_KxNm1, K):
        tree_probabilities = []
        for i in range(K):
            tree = op_seqs[i]
            tree_probabilities.append(0)
            for k in range(K):
                if tree == op_seqs[k]:
                    tree_probabilities[i] += weights_KxNm1[k,-1]
            tree_probabilities[i] /= K
        tree_probabilities /= 1/K*sum(weights_KxNm1[:,-1])
        tree_probabilities = list(tree_probabilities)
        return tree_probabilities

    def compute_norm(self, weights_KxNm1):
        K = weights_KxNm1.shape[0]
        norm = 1
        for i in range(self.n-1):
            norm *= 1/K*sum(weights_KxNm1[:,i])
        return norm

    def get_op_seq(self):
        # Get a coalesence operation sequence whose length is 2*(N-1)
        N = self.n
        pool_of_idx = list(range(N))
        op_seq = []
        for i in range(N,2*N-1):
            idxs = sorted(random.sample(pool_of_idx,2))
            op_seq.append(idxs[0])
            op_seq.append(idxs[1])
            pool_of_idx.remove(idxs[0])
            pool_of_idx.remove(idxs[1])
            pool_of_idx.append(i)
        return op_seq

    def op_seq_to_particles(self, op_seq):
        N, S, A = self.n, self.s, self.Qmatrix.shape[0]
        particles, left_branches, right_branches = list(range(2*N-1)), list(range(N-1)), list(range(N-1))
        particles[:N] = self.taxa
        for i in range(int(len(op_seq)/2)):
            particles[N+i] = particles[op_seq[2*i]]+'+'+particles[op_seq[2*i]]
            bl1, bl2 = lognorm.rvs(1,1,1,2)
            left_branches[i] = bl1
            right_branches[i] = bl2
        return particles, left_branches, right_branches

    def op_seq_to_data(self, op_seq, left_branches, right_branches):
        N, S, A = self.n, self.s, self.Qmatrix.shape[0]
        node_data = np.zeros((2*N-1,S,A))
        node_data[:N,:,:] = self.genome_NxSxA
        for i in range(int(len(op_seq)/2)):
            idx1, idx2 = op_seq[2*i], op_seq[2*i+1]
            node_data[N+i] = self.conditional_likelihood(node_data[idx1], node_data[idx2], left_branches[i], right_branches[i]) 
        return node_data

    def sample_phylogenies(self, K, resampling=False, showing=True):

        N, S, A = self.n, self.s, self.Qmatrix.shape[0]

        self.graph_repn_data_K = [[] for i in range(K)]
        self.graph_repn_nodes_K = [[] for i in range(K)]

        log_weights_KxNm1 = np.zeros([K,N-1])
        weights_KxNm1 = np.zeros([K,N-1]) + 1
        log_likelihood = np.zeros([K,N-1])
        log_likelihood_tilda = np.zeros(K) + 1

        # Perform coalescent events K times; record everything that has happened
        particles_K = [list(range(2*N-1))]*K 
        left_branches_K, right_branches_K = np.zeros((K,N-1)), np.zeros((K,N-1))
        node_data_K = np.zeros((K,2*N-1,S,A))
        op_seqs = []
        for k in range(K):
            op_seq = self.get_op_seq()
            op_seqs.append(op_seq)
            particles_K[k], left_branches_K[k], right_branches_K[k] = self.op_seq_to_particles(op_seq)
            node_data_K[k] = self.op_seq_to_data(op_seq, left_branches_K[k], right_branches_K[k])

        # Iterate over coalescent events
        for i in range(N-1):
            # Newly coalesced particle at step i corresponds to particles_KxN[k][i+N]
            q = ncr(N-i, 2)

            # Resampling step
            if resampling and i > 0:
                jump_chain_KxN[:,i-1] = self.resample(log_weights_KxNm1, jump_chain_KxN[:,i-1], i-1)

            # Iterate over partial states
            for k in range(K):
                # Sample from last step
                if i > 0:
                    loglik = 0
                    k_idx = random.randint(0,K-1)
                    op_seq_cur = op_seqs[k_idx][:2*i]
                    for node_idx in range(N+i):
                        if node_idx not in op_seq_cur:
                            loglik += self.compute_tree_likelihood(node_data_K[k_idx,node_idx])
                    log_likelihood_tilda[k] = loglik

                # Build tree for visualization
                self.build_tree(particles_K[k][op_seqs[k][2*i]], particles_K[k][op_seqs[k][2*i+1]], \
                    particles_K[k][N+i], k)

            for k in range(K):
                # Compute log conditional likelihood across genome for a partial state
                op_seq_cur = op_seqs[k][:2*i+2]
                for node_idx in range(N+i+1):
                    if node_idx not in op_seq_cur:
                        log_likelihood[k,i] += self.compute_tree_likelihood(node_data_K[k,node_idx])

                # Overcounting correction
                v = self.overcounting_correct(op_seq_cur, i)

                # Compute the importance weights
                if i > 0:
                    log_weights_KxNm1[k, i] = log_likelihood[k,i] - log_likelihood_tilda[k] + np.log(v) - np.log(q)
                    weights_KxNm1[k, i] = np.exp(log_weights_KxNm1[k, i])
                    
            print('Computation in progress: step ' + str(i+1))
        # End of iteration

        #print(log_likelihood)
        #print(log_likelihood_tilda)
        #print(weights_KxNm1)
        tree_probabilities = self.get_tree_prob(op_seqs, weights_KxNm1, K)
        print(tree_probabilities)
        norm = self.compute_norm(weights_KxNm1, K)
        print(norm)
        selected_idx = tree_probabilities.index(max(tree_probabilities))
        print(op_seqs[selected_idx])
        
        G = self.tree_to_graph(Graph(), self.graph_repn_nodes_K[selected_idx][-1])
        if showing:
            G.draw(tree_probabilities[selected_idx])

        return log_weights_KxNm1, tree_probabilities, norm, G


if __name__ == "__main__":

    real_data_corona = False
    real_data_1 = False
    real_data_2 = False
    simulate_data = True
    load_strings = False

    Alphabet_dir = {'A': [1, 0, 0, 0],
                    'C': [0, 1, 0, 0],
                    'G': [0, 0, 1, 0],
                    'T': [0, 0, 0, 1]}
    alphabet_dir = {'a': [1, 0, 0, 0],
                    'c': [0, 1, 0, 0],
                    'g': [0, 0, 1, 0],
                    't': [0, 0, 0, 1]}
                    

    alphabet = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    genome_strings = ['ACTTTGAGAG','ACTTTGACAG','ACTTTGACTG','ACTTTGACTC']
    #genome_strings = ['AAAAAA','CCCCCC','TTTTTT','GGGGGG']
    # a = ''
    # for i in range(150):
    #     a += 'A'
    # genome_strings = [a,a,a,a]

    def simulateDNA(nsamples, seqlength, alphabet):
        genomes_NxSxA = np.zeros([nsamples, seqlength, alphabet.shape[0]])
        for n in range(nsamples):
            genomes_NxSxA[n] = np.array([random.choice(alphabet) for i in range(seqlength)])
        return genomes_NxSxA

    def form_dataset_from_strings(genome_strings, alphabet_dir):
        #pdb.set_trace()
        genomes_NxSxA = np.zeros([len(genome_strings),len(genome_strings[0]),len(alphabet_dir)])
        for i in range(genomes_NxSxA.shape[0]):
            for j in range(genomes_NxSxA.shape[1]):
                genomes_NxSxA[i,j] = alphabet_dir[genome_strings[i][j]]
        taxa = ['S' + str(i) for i in range(genomes_NxSxA.shape[0])]
        datadict = {'taxa': taxa,
                    'genome': genomes_NxSxA}
        return datadict

    dim = 4

    if simulate_data:
        data_NxSxA = simulateDNA(3, 10, alphabet)
        #print("Simulated genomes:\n", data_NxSxA)
        taxa = ['S' + str(i) for i in range(data_NxSxA.shape[0])]
        #print(taxa)
        datadict = {'taxa': taxa,
                    'genome': data_NxSxA}

    if load_strings:
        datadict = form_dataset_from_strings(genome_strings, Alphabet_dir)

    if real_data_corona:
        datadict = pd.read_pickle('tencovid.p')
        dim = 6

    if real_data_1:
        genome_strings = \
           ['aaccctgttatttccacatgccaacaatcccaacag',
            'aactctgttatttccacatgccaacaatcccaacag',
            'aaatctgtgttgtctaaatgtcagttatttcagtta',
            'aaagctattatttaaaaatataaattatctcaatta',
            'aacactgttatttctaaatatcacttttcccaattg']
        datadict = form_dataset_from_strings(genome_strings, alphabet_dir)
        datadict['taxa'] = ['human', 'gibbon','guinea pig', 'aardvark', 'armadillo']
    
    if real_data_2:
        # Primates
        # block 19, 20, 32, 38, 42, 45, 47, 52, 53, 54, 57, 74, 78, 89, 92, 202, 223, 228, 239, 286, 304, 309, 346
        genome_strings = \
           ['taatggaataacacctttgctatgttatccaaacaatattagtcctttttcttctcttgtcgcccagccagagggcaatggtgggatctcggctcactgagacctctgcctcccagttcaagttacaggcacccgccaggctggtctcgaactgctgacctcaggtgatccacccaccttggcctccgaaagtgccgggattataggcgtgagccaccgcaccacctagcttgtatcgaacaaagggaataaaaaatgtatggatcaaggctcatgtacacaagatccaaattatccaccatccaggataatattttttgg',
            'gaatggaataacacctttgctatgttatccaaacaatattagtcctttttcttctcttgtcgcccagccagagggcaatggtgggatctccgctcactgagacctctgcctcccagttcaagttacaggcacccgccaggctggtctcgaactgctgacctcaggtgaaccacccaccttggcctccgaaagtgccgggattataggcgtgagccaccgcaccacctagcttgtatcgaacaaagggaataaaaaatgtatggatcaaggctcatgtacacaagatccaaattatccaccatccaggataatattttttgg',
            'taatggaataacacctttgctatgttatccaaacaatattagtccttttttttctcttgtcacccagccagagggcaatggcgggatctcggctcactgagacctctgcctcccagttcaagctacaggcacccgccaggctggtcttgaactgctgacctcaggtgatccacccaccttggcctccaaaagtgccgggattataggtgtgagccaccgcaccacctagcttgtatcgaactaagggaataaaaaatgtatggatcaaggctcatgtacacaagatccaaattatccaccatccaggataatatttttcgg',
            'taatggaataacacctttgctatgttatccaaacaatattagtcctattttttctcttgtcacccagccagagggcaatggtgggatctcggctcactgcgacctctgcctcccagttcaagctacaggcacccgccaggctgggctccaactgctgacctcaggtgatccacccatcttggcctccgaaagtgccgggattacaggcgtgagccaccgcactgcctagtttgtatcgaacaaagggaatataaaatgtatgattcaaggctcatgtacacaagatccaaattatcccccatccaggatagtattttacgg',
            'taatggaataacacctttgctatgttattcaaacaatattagtcctattttttctcttgttgcccagctggagggcaatggcgggatctcggctcgctgccacctctgcctcccagttcaggctacaggcacctgccatgctgttcctgaactgctgacctcaggtgatccacctaccttggcctccaaaagtgccgggattacaggcgtgagccaccgcactgcctagtttgtattgaacaaagggaatataaaatgtatgaatcaaggctcatgtacacaagatccaaattatccaccatccaggataatattttatgg',
            'taacagaataacacctttactatgttatctaaataatatttgtcctattttttctcttgtcacccagctggaaagcaatggcgggacctcagctcactgcaacctctgcctcccagttcaagctataggcatctgccaggctggtctcgaactgctgacctcaggtgatccacccgccttggcctcccaaagcgctgggattgtaggcatgagccaccccgccacctagtttgtatagaatataggagatacaaaatgtatgaatcaaggctgacgtatacacgatccaaattatcccccacccaggacaatattttctga',
            'taacagaataacacctttgctatgttatctaaataatatttgtcctattttttctcttgtcacccagctggaaagcaatggcgggacctcagctcactgcaacctctgcctcccagttcaagctacaggcatctgccaggctggtctcgaactgctgacctcaggtgatccacccgccttggcctcccaaagcgctgggattgtaggcatgagccaccccgccacctagtttgtatagaatataggagatacaaaatgtatgaatcaaggctgacgtatacacgatccaaattatcccccacccaggacaatattttctga',
            'taacagaataacacttttgctatgttatctaaataatatttgtcctatttcttctcttgtcgcccagctggaaggcaatggcgggacctcagctcactgcaacctctgcctcccagttcaagctacaggcatctgccaggctggtctagaactgctgacctcaggtgatccacccgccttggcctcccaaagtgctggaattgcaggcatgagccaccccgccacctagtttgtatagaatataggagatacaaaatgtatgaatcaaggctgacgtccacacgatccaaattatcccccacccaggacaatattttctga',
            'taacagaataacacctttgctatgttatctaaataatatttgtcctattttttctcttgtcgcccagctggtgggcaatggcggaatctcggctcaatgcaacctctgcctcccagttcaagctacaggcatctgtcaggctggtctcaaactgctgacctcaggtgatccacccgccttggcctcccaaagtgctgggattacaggcacgacccaccccgccacctagtttgtatagaatagaggagatacaaaatgtatgaatccaggctgacgtacacacgatccaaattatcccccacccaggacaatattttctga']
        datadict = form_dataset_from_strings(genome_strings, alphabet_dir)
        datadict['taxa'] = ['human','chimp','gorilla','oranguta','gibbon','rhesus','macaque','baboon','greenmonkey']
        

    csmc = CSMC(datadict)

    if dim==6:
        csmc.Qmatrix = np.array([[-5.,1.,1.,1.,1.,1.],
                                 [1.,-5.,1.,1.,1.,1.],
                                 [1.,1.,-5.,1.,1.,1.],
                                 [1.,1.,1.,-5.,1.,1.],
                                 [1.,1.,1.,1.,-5.,1.],
                                 [1.,1.,1.,1.,1.,-5.]])
        csmc.Pmatrix = spl.expm(csmc.Qmatrix)
        csmc.prior = np.ones(csmc.Qmatrix.shape[0])/csmc.Qmatrix.shape[0]

    log_weights, tree_probs, norm, G = csmc.sample_phylogenies(200, resampling=False, showing=True)










