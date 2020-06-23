"""
An implementation of the Combinatorial Sequential Monte Carlo algorithm
for Phylogenetic Inference
"""

import numpy as np
import random
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.stats import lognorm
import scipy.linalg as spl
from scipy.linalg import expm
import matplotlib.pyplot as plt
import pandas as pd
import pdb
from copy import deepcopy
import operator as op
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from functools import reduce


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


class Vertex:

    def __init__(self, id=None, data=None):
        self.id = id
        self.data = data
        self.left = None
        self.right = None
        self.left_branch = None
        self.right_branch = None
        self.is_root = True
        self.data_done = False


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



class CSMC:
    """
    CSMC takes as input a dictionary (datadict) with two keys:
     taxa: a list of n strings denoting taxa
     genome_NxSxA: a 3 tensor of genomes for the n taxa one hot encoded
    """

    def __init__(self, datadict):
        self.taxa = datadict['taxa']
        self.genome_NxSxA = datadict['genome']
        self.n = len(datadict['taxa'])
        self.s = len(self.genome_NxSxA[0])
        self.pi_c = tf.Variable(0.25, dtype=tf.float64, name="pi_c", constraint=lambda x: tf.clip_by_value(x, 0, 1))
        self.pi_g = tf.Variable(0.25, dtype=tf.float64, name="pi_g", constraint=lambda x: tf.clip_by_value(x, 0, 1))
        self.pi_t = tf.Variable(0.25, dtype=tf.float64, name="pi_t", constraint=lambda x: tf.clip_by_value(x, 0, 1-self.pi_c-self.pi_g))
        self.kappa = tf.Variable(2., dtype=tf.float64, name="kappa")
        self.Qmatrix = self.get_Q(self.pi_c,self.pi_g,self.pi_t,self.kappa)
        # self.Qmatrix = np.array([[-3.,1.,1.,1.],
        #                          [1.,-3.,1.,1.],
        #                          [1.,1.,-3.,1.],
        #                          [1.,1.,1.,-3.]])
        self.Pmatrix = tf.linalg.expm(self.Qmatrix)
        self.state_probs = tf.stack([[1-self.pi_c-self.pi_g-self.pi_t, self.pi_c, self.pi_g, self.pi_t]], axis=0)
        self.leaf_nodes = []

    def get_Q(self, pi_c, pi_g, pi_t, kappa):
        # returns Qmatrix with entries defined by these four tf.Variables
        A1 = tf.stack([-pi_c-kappa*pi_g-pi_t, pi_c, kappa*pi_g, pi_t], axis=0)
        A2 = tf.stack([1-pi_c-pi_g-pi_t, -1+pi_c+pi_t-kappa*pi_t, pi_g, kappa*pi_t], axis=0)
        A3 = tf.stack([kappa*(1-pi_c-pi_g-pi_t), pi_c, -kappa*(1-pi_c-pi_g-pi_t)-pi_c-pi_t, pi_t], axis=0)
        A4 = tf.stack([1-pi_c-pi_g-pi_t, kappa*pi_c, pi_g, -1+pi_c+pi_t-kappa*pi_c], axis=0)
        Q = tf.stack((A1,A2,A3,A4), axis=0)
        return Q

    #@staticmethod
    def ncr(self, n, r):
        # Compute combinatorial term n choose r
        r = min(r, n-r)
        numer = reduce(op.mul, range(n, n-r, -1), 1)
        denom = reduce(op.mul, range(1, r+1), 1)
        return numer/denom

    #@staticmethod
    def sort_string(self, s):
        lst = s.split('+')
        lst = sorted(lst)
        result = '+'.join(lst)
        return result

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

    def build_graph(self, G, master_node):
        """
        Constructs a graph for visualizing phylogeny
        """
        if master_node.subnodes == []:
            G.add_node(master_node)
            return G

        G.add_node(master_node)

        for n in master_node.subnodes:
            G = self.build_graph(G, n)

        return G

    def get_internal_nodes(self,root):
        """
        Collects all internal nodes (ancestral or latent variables) for likelihood computation
        May not be necessary...
        """
        #pdb.set_trace()
        q = []
        q.append(root)
        nodes = []
        while (len(q)):
            curr = q[0]
            q.pop(0)
            isInternal = 0
            if (curr.left):
                isInternal = 1
                if not curr.left.data_done:
                    q.append(curr.left)
            if (curr.right):
                isInternal = 1
                if not curr.left.data_done:
                    q.append(curr.right)
            if (isInternal):
                nodes.append(curr)
                # print(curr.id, end = " ")
        # Make sure that node ordering is such that any child is placed before its parent
        return nodes[::-1]

    def resample(self, weights, jump_chain_K, i):
        """
        Resamples partial states (particles) based on importance weights
        """
        #pdb.set_trace()
        K = weights.shape[0]
        importance_weights = np.exp(weights[:,i])
        norm = np.sum(importance_weights)
        indices = np.random.choice(K, K, p = importance_weights/norm, replace=True)
        jump_chain_K = jump_chain_K[indices]
        return jump_chain_K

    #@staticmethod
    def create_node_sampler(self, jump_chain, i):
        """
        This defines the sample space of nodes at which coalescent events can happen.
        The sample space is defined in the list variable 'result'
        REVISIT ME. This is no longer necessary...
        """
        result = []
        for key in jump_chain[i]:
            if len(jump_chain[i][key]) > 1:
                for p in range(len(jump_chain[i][key])):
                    result.append(key)
        return result

    def extend_partial_state(self, jump_chain_KxN, j, i):
        """
        Forms a partial state by sampling two nodes from the proposal distribution
        """
        # Set c[n-1] = c[n], copying our dictionary representing the particle set
        jump_chain_KxN[j, i + 1] = deepcopy(jump_chain_KxN[j, i])

        # Sample two posets
        sample = random.sample(jump_chain_KxN[j,i][0], 2)
        q2 = 1 / self.ncr(len(jump_chain_KxN[j,i][0]), 2)
        particle1 = sample[0]
        particle2 = sample[1]
        particle_coalesced = self.sort_string(particle1 + '+' + particle2)
        jump_chain_KxN[j,i+1][0].remove(particle1)
        jump_chain_KxN[j,i+1][0].remove(particle2)
        jump_chain_KxN[j,i+1][0].append(particle_coalesced)

        # Branch length as tf.Variable to be optimized
        bl1 = tf.Variable(1, dtype=tf.float64)
        bl2 = tf.Variable(1, dtype=tf.float64)

        return particle1, particle2, particle_coalesced, bl1, bl2, q2, jump_chain_KxN

    def pass_messages(self,internal_nodes):
        #pdb.set_trace()
        # Pass messages from leaf nodes to root
        for node in internal_nodes:
            if not node.data_done:
                node.data = self.conditional_likelihood(node)
                node.data_done = True

    def conditional_likelihood(self, node):
        #pdb.set_trace()
        # Computes the conditional likelihood using the formula above
        left_Pmatrix = tf.linalg.expm(self.Qmatrix * node.left_branch)
        right_Pmatrix = tf.linalg.expm(self.Qmatrix * node.right_branch)
        left_prob = tf.matmul(node.left.data, left_Pmatrix)
        right_prob = tf.matmul(node.right.data, right_Pmatrix)
        likelihood = tf.multiply(left_prob, right_prob)
        return likelihood

    def compute_log_conditional_likelihood(self, v):
        #pdb.set_trace()
        internal_nodes = self.get_internal_nodes(v)
        self.pass_messages(internal_nodes)
        tree_likelihood = tf.matmul(self.state_probs, v.data, transpose_b=True)
        loglik = tf.reduce_sum(tf.log(tree_likelihood))
        return loglik

    def overcounting_correct(self, vertex_dict):
    	rho = 0
    	for key in vertex_dict:
    		if vertex_dict[key].is_root and vertex_dict[key].left is not None:
    			rho += 1
    	return 1/rho

    def get_tree_prob(self, vertex_dicts, weights_KxNm1, K):
        # This method is not currently used
        trees = []
        for dic in vertex_dicts:
            trees.append(dic.keys())
        tree_probabilities = []
        for i in range(len(trees)):
            tree = trees[i]
            tree_probabilities.append(0)
            for k in range(K):
                if tree == trees[k]:
                    tree_probabilities[i] += weights_KxNm1[k,-1]
            tree_probabilities[i] /= K
        tree_probabilities /= 1/K*sum(weights_KxNm1[:,-1])
        tree_probabilities = list(tree_probabilities)
        return tree_probabilities, trees

    def compute_norm(self, weights_KxNm1):
        # Computes norm, which is negative of our final cost
        norm = tf.reduce_prod(1/K*tf.reduce_sum(weights_KxNm1, axis=0))
        return norm

    def sample_phylogenies(self, K, resampling=False, showing=False):

        n = self.n
        s = self.s
        # Represent a single jump_chain as a list of dictionaries
        jump_chain = [{} for i in range(n)]
        jump_chain[0][0] = self.taxa
        jump_chain_KxN = np.array([jump_chain] * K) # KxN matrix of jump_chain dictionaries

        log_weights_KxNm1 = [[0 for i in range(n-1)] for j in range(K)]
        weights_KxNm1 = [[0 for i in range(n-1)] for j in range(K)]
        log_likelihood = [[0 for i in range(n-1)] for j in range(K)]

        # self.graph_repn_data_K = [[] for i in range(K)]
        # self.graph_repn_nodes_K = [[] for i in range(K)]

        vertex_dicts = [{} for k in range(K)]
        for j in range(K):
            for i in range(n):
                vertex_dicts[j][self.taxa[i]] = Vertex(id=self.taxa[i],data=tf.placeholder(dtype=tf.float64, shape=(s,4)))
                if j == 0:
                    self.leaf_nodes.append(vertex_dicts[j][self.taxa[i]])

        # Iterate over coalescent events
        for i in range(n - 1):

            # Resampling step
            if resampling and i > 0:
                jump_chain_KxN[:,i-1] = self.resample(log_weights_KxNm1, jump_chain_KxN[:,i-1], i-1)

            # Iterate over partial states
            for k in range(K):

                # Extend partial states
                particle1, particle2, particle_coalesced, bl1, bl2, q, jump_chain_KxN \
                    = self.extend_partial_state(jump_chain_KxN, k, i)

                # Save partial set and branch length data
                vertex_dicts[k][particle_coalesced] = Vertex(id=particle_coalesced, data=None)
                vertex_dicts[k][particle_coalesced].left = vertex_dicts[k][particle1]
                vertex_dicts[k][particle_coalesced].right = vertex_dicts[k][particle2]
                vertex_dicts[k][particle_coalesced].left_branch = bl1
                vertex_dicts[k][particle_coalesced].right_branch = bl2
                vertex_dicts[k][particle1].is_root = False
                vertex_dicts[k][particle2].is_root = False

                # Build tree
                # self.build_tree(particle1, particle2, particle_coalesced, k)

            for k in range(K):
                # Compute log conditional likelihood across genome for a partial state
                print('Computation of norm in progress: step ', i+1, k)
                log_likelihood[k][i] = 0
                for key in vertex_dicts[k]:
                    if vertex_dicts[k][key].is_root:
                        log_likelihood[k][i] \
                        += self.compute_log_conditional_likelihood(vertex_dicts[k][key])

                # Sample from last step
                log_likelihood_tilda = 1
                if i > 0:
                    log_likelihood_tilda = 0
                    idx = random.randint(0,K-1)
                    for key in vertex_dicts_laststep[idx]:
                        if vertex_dicts_laststep[idx][key].is_root:
                            log_likelihood_tilda \
                            += self.compute_log_conditional_likelihood(vertex_dicts_laststep[idx][key])

                # Overcounting correction
                v = self.overcounting_correct(vertex_dicts[k])

                # Compute the importance weights
                if i > 0:
                    log_weights_KxNm1[k][i] = log_likelihood[k][i] - log_likelihood_tilda + tf.log(v) - tf.log(q)
                    weights_KxNm1[k][i] = tf.exp(log_weights_KxNm1[k][i])

            vertex_dicts_laststep = deepcopy(vertex_dicts)
            # print('Computation of norm in progress: step ' + str(i+1))
        # End of iteration

        norm = self.compute_norm(weights_KxNm1)
        self.cost = -norm
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01).minimize(self.cost)

        # print(weights_KxNm1)

        # tree_probabilities, trees = self.get_tree_prob(vertex_dicts, weights_KxNm1, K)
        # print(tree_probabilities)

        # plt.hist(tree_probabilities)
        # plt.xlabel('Posterior probability')
        # plt.ylabel('Number of generated trees')
        # plt.show()
        # np.savetxt('tree_prob.csv', tree_probabilities, delimiter=',')

        # selected_idx = tree_probabilities.index(max(tree_probabilities))
        # print(trees[selected_idx])
        
        # pdb.set_trace()
        # G = self.build_graph(Graph(), self.graph_repn_nodes_K[selected_idx][-1])
        # if showing:
            # G.draw(tree_probabilities[selected_idx])

        return norm

    def get_feed_dict(self):
        self.feed_dict = {}
        for i in range(len(self.leaf_nodes)):
            idx = self.taxa.index(self.leaf_nodes[i].id)
            self.feed_dict[self.leaf_nodes[i].data] = self.genome_NxSxA[idx]

    def train(self, K):
        """
        Run the train op in a TensorFlow session and evaluate variables
        """
        self.sample_phylogenies(K)
        self.get_feed_dict()
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        costs = []
        for i in range(1000):
            _, cost = sess.run([self.optimizer, self.cost], feed_dict=self.feed_dict)
            costs.append(cost)
            print('Training in progress: step', i)
        
        print(costs)
        for node in self.internal_nodes:
            print('-----------------')
            print(node.id)
            print(sess.run(node.left_branch))
            print(sess.run(node.right_branch))
        #plt.plot(costs)
        #plt.show()

        return


if __name__ == "__main__":

    real_data_corona = False
    real_data_1 = False
    real_data_2 = False
    simulate_data = True
    load_strings = False

    alphabet = 'ACTG'
    Alphabet_dir = {'A': [1, 0, 0, 0],
                    'C': [0, 1, 0, 0],
                    'T': [0, 0, 1, 0],
                    'G': [0, 0, 0, 1]}
    alphabet_dir = {'a': [1, 0, 0, 0],
                    'c': [0, 1, 0, 0],
                    't': [0, 0, 1, 0],
                    'g': [0, 0, 0, 1]}
                    

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
        data_NxSxA = simulateDNA(3, 5, alphabet)
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

    csmc.train(10)

    # log_weights, tree_probs, norm, G = csmc.sample_phylogenies(200, resampling=False, showing=True)










