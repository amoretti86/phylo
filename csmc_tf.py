"""
An implementation of the Combinatorial Sequential Monte Carlo algorithm
for Phylogenetic Inference
"""

# Version 6; based on 5, and corrects gather_nd issue during gradient eval

import numpy as np
import random
import tensorflow.compat.v1 as tf
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


class Vertex:

    def __init__(self, id=None, data=None):
        self.id = id
        self.data = data
        self.left = None
        self.right = None
        self.left_branch = None
        self.right_branch = None
        self.is_subroot = True
        self.is_leaf = False
        self.data_done = False


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
        self.taxa = datadict['taxa']
        self.genome_NxSxA = datadict['genome']
        self.n = len(datadict['taxa'])
        self.s = len(self.genome_NxSxA[0])
        # self.pi_c = tf.Variable(0.25, dtype=tf.float64, name="pi_c", constraint=lambda x: tf.clip_by_value(x, 0, 1))
        # self.pi_g = tf.Variable(0.25, dtype=tf.float64, name="pi_g", constraint=lambda x: tf.clip_by_value(x, 0, 1))
        # self.pi_t = tf.Variable(0.25, dtype=tf.float64, name="pi_t", constraint=lambda x: tf.clip_by_value(x, 0, 1-self.pi_c-self.pi_g))
        self.pi_c = tf.constant(0.25, dtype=tf.float64)
        self.pi_g = tf.constant(0.25, dtype=tf.float64)
        self.pi_t = tf.constant(0.25, dtype=tf.float64)
        self.kappa = tf.Variable(2., dtype=tf.float64, name="kappa")
        self.Qmatrix = self.get_Q(self.pi_c,self.pi_g,self.pi_t,self.kappa)
        self.Pmatrix = tf.linalg.expm(self.Qmatrix)
        self.state_probs = tf.stack([[1-self.pi_c-self.pi_g-self.pi_t, self.pi_c, self.pi_g, self.pi_t]], axis=0)

    def get_Q(self, pi_c, pi_g, pi_t, kappa):
        # returns Qmatrix with entries defined by these four tf.Variables
        A1 = tf.stack([-pi_c-kappa*pi_g-pi_t, pi_c, kappa*pi_g, pi_t], axis=0)
        A2 = tf.stack([1-pi_c-pi_g-pi_t, -1+pi_c+pi_t-kappa*pi_t, pi_g, kappa*pi_t], axis=0)
        A3 = tf.stack([kappa*(1-pi_c-pi_g-pi_t), pi_c, -kappa*(1-pi_c-pi_g-pi_t)-pi_c-pi_t, pi_t], axis=0)
        A4 = tf.stack([1-pi_c-pi_g-pi_t, kappa*pi_c, pi_g, -1+pi_c+pi_t-kappa*pi_c], axis=0)
        Q = tf.stack((A1,A2,A3,A4), axis=0)
        return Q

    def resample(JC_K, weights_KxNm1, i, debug=False):
        """
        Resample partial states by drawing from a categorical distribution whose parameters are normalized importance weights
        JumpChain (JC_K) is a tensor formed from a numpy array of lists of strings, returns a resampled JumpChain tensor
        """
        indices = tf.random.categorical(tf.math.log(tf.expand_dims(weights_KxNm1[:, i] / tf.reduce_sum(weights_KxNm1[:, i], axis=0), axis=0)), self.K)
        resampled_JC_K = tf.gather(JC_K, tf.squeeze(indices))
        return resampled_JC_K

    def extend_partial_state(self, JCK, debug=False):
        """
        Extends partial state by sampling two states to coalesce (Gumbel-max trick to sample without replacement)
        JumpChain (JC_K) is a tensor formed from a numpy array of lists of strings, returns a new JumpChain tensor
        """
        # Compute combinatorial term
        #pdb.set_trace()
        q = 1 / ncr(JCK.shape[1],2)
        #K = JCK.shape[0]
        #n = JCK.shape[1]
        #data = np.arange(0, (JCK.shape[1].value - 1) * JCK.shape[0].value, 1).reshape(JCK.shape[0].value, JCK.shape[1].value - 1) % (JCK.shape[1].value - 1)
        data = np.arange(0, (JCK.shape[1].value) * JCK.shape[0].value, 1).reshape(JCK.shape[0].value,JCK.shape[1].value) % (JCK.shape[1].value)
        data = tf.constant(data, dtype=tf.int32)
        data = tf.cast(data, dtype=tf.float32)
        # Gumbel-max trick to sample without replacement
        z = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(data), 0, 1)))
        top_values, coalesced_indices = tf.nn.top_k(data + z, 2)
        bottom_values, remaining_indices = tf.nn.top_k(tf.negative(data + z), JCK.shape[1].value - 2)
        JC_keep = tf.gather(tf.reshape(JCK, [JCK.shape[0].value*(JCK.shape[1].value)]), remaining_indices)
        particles = tf.gather(tf.reshape(JCK, [JCK.shape[0].value*(JCK.shape[1].value)]), coalesced_indices)
        particle1 = particles[:, 0]
        particle2 = particles[:, 1]
        # Form new state
        particle_coalesced = particle1 + '+' + particle2
        # Form new Jump Chain
        JCK = tf.concat([JC_keep, tf.expand_dims(particle_coalesced, axis=1)], axis=1)

        return particle1, particle2, particle_coalesced, coalesced_indices, remaining_indices, q, JCK

    def conditional_likelihood(self, l_data, r_data, l_branch, r_branch):
        #pdb.set_trace()
        left_Pmatrix = tf.linalg.expm(self.Qmatrix * l_branch)
        right_Pmatrix = tf.linalg.expm(self.Qmatrix * r_branch)
        left_prob = tf.matmul(l_data, left_Pmatrix)
        right_prob = tf.matmul(r_data, right_Pmatrix)
        likelihood = tf.multiply(left_prob, right_prob)
        return likelihood

    def compute_tree_likelihood(self, data):
        #pdb.set_trace()
        tree_likelihood = tf.matmul(self.state_probs, data, transpose_b=True)
        loglik = tf.reduce_sum(tf.log(tree_likelihood))
        return loglik

    def overcounting_correct(self, v, indices, i):
        idx1 = tf.gather(indices, 0)
        idx2 = tf.gather(indices, 1)
        threshold = self.n-i-1
        cond_greater = tf.cond(tf.logical_and(idx1 > threshold, idx2 > threshold), lambda:-1, lambda:0)
        result = tf.cond(tf.logical_and(idx1 < threshold, idx2 < threshold), lambda:1, lambda:cond_greater)

        v = tf.add(v, result)
        return v

    def compute_norm(self, weights_KxNm1):
        # Computes norm, which is negative of our final cost
        K = len(weights_KxNm1)
        norm = tf.reduce_prod(1/K*tf.reduce_sum(weights_KxNm1, axis=0))
        return norm

    def sample_phylogenies(self, K, resampling=True):
        n = self.n
        s = self.s

        log_weights_KxNm1 = [[0. for i in range(n-1)] for j in range(K)]
        weights_KxNm1 = [[1. for i in range(n-1)] for j in range(K)]
        log_likelihood = [[0. for i in range(n-1)] for j in range(K)]
        log_likelihood_tilda = [1. for j in range(K)]

        # Represent a single jump_chain as a list of dictionaries
        jump_chain_tensor = tf.constant([self.taxa] * K, name='JumpChainK')
        # Keep matrices of all vertices, KxNxSxA (the coalesced children vertices will be removed as we go)
        self.core = tf.constant(np.array([self.genome_NxSxA]*K))
        self.left_branches = [[tf.Variable(2, dtype=tf.float64) for i in range(n-1)] for k in range(K)]
        self.right_branches = [[tf.Variable(2, dtype=tf.float64) for i in range(n-1)] for k in range(K)]
        v = [1 for k in range(K)]
        p_for_last_step = [1/K for i in range(K)]

        # Iterate over coalescent events
        for i in range(n-1):
            # Resampling step
            if resampling and i > 0:
                jump_chain_tensor = self.resample(jump_chain_tensor, weights_KxNm1, i-1)

            # Sample from last step
            '''This needs to be changed. Needs a tensorflow way to do random.randint()'''
            if i > 0:
                for k in range(K):
                    loglik = 0
                    idx = tf.squeeze(tf.random.categorical(tf.log([p_for_last_step]), 1))
                    for j in range(n-i):
                        loglik += self.compute_tree_likelihood(tf.gather_nd(self.core, [[idx,j,s_i] for s_i in range(s)]))
                    log_likelihood_tilda[k] = loglik

            # Extend partial states
            particle1, particle2, particle_coalesced, coalesced_indices, remaining_indices, \
              q, jump_chain_tensor = self.extend_partial_state(jump_chain_tensor)

            # Save partial set and branch length data
            new_data = [0 for i in range(K)]
            for k in range(K):
                n1 = tf.gather_nd(coalesced_indices, [k,0])
                n2 = tf.gather_nd(coalesced_indices, [k,1])
                l_data = tf.gather_nd(self.core, [[k,n1,s_i] for s_i in range(s)])
                r_data = tf.gather_nd(self.core, [[k,n2,s_i] for s_i in range(s)])
                l_branch = self.left_branches[k][i]
                r_branch = self.right_branches[k][i]
                mtx = self.conditional_likelihood(l_data, r_data, l_branch, r_branch)
                new_data[k] = tf.expand_dims(mtx, axis=0)
            new_data = tf.concat([new_data], axis=0)
            shape1 = self.core.shape[0].value * self.core.shape[1].value
            self.core = tf.gather(tf.reshape(self.core, [shape1, self.core.shape[2].value, self.core.shape[3].value]), remaining_indices)
            self.core = tf.concat([self.core, new_data], axis=1)
            # (after these, every matrix in self.core[k] (self.core[k] is a 'list' of matrices) will be a subroot's data)
            # (which means that compute_tree_likelihood only needs to sum over these matrices)

            # Iterate over partial states
            for k in range(K):
                # Compute log conditional likelihood across genome for a partial state
                log_likelihood[k][i] = 0
                for j in range(n-1-i):
                    log_likelihood[k][i] += self.compute_tree_likelihood(tf.gather_nd(self.core, [[k,j,s_i] for s_i in range(s)]))

                # Overcounting correction and ompute the importance weights
                if i > 0:
                    v[k] = self.overcounting_correct(v[k], tf.gather(coalesced_indices, k), i)

                    log_weights_KxNm1[k][i] = log_likelihood[k][i] - log_likelihood_tilda[k] + \
                     tf.math.log(tf.cast(v[k],tf.float64)) - tf.math.log(tf.cast(q,tf.float64))
                    weights_KxNm1[k][i] = tf.exp(log_weights_KxNm1[k][i])

                print('Construction of computational graph in progress: step ', i+1, k+1)
        # End of iteration
        norm = self.compute_norm(weights_KxNm1)
        self.cost = -norm
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(self.cost)

        return norm

    # def get_feed_dict(self):
    #     self.feed_dict = {}
    #     for k in range(len(self.lst_placeholders)):
    #         for i in range(len(self.lst_placeholders[0])):
    #             self.feed_dict[self.lst_placeholders[k][i]] = self.genome_NxSxA[i]

    def train(self, K):
        """
        Run the train op in a TensorFlow session and evaluate variables
        """
        self.K = K
        self.sample_phylogenies(K)
        #self.get_feed_dict()
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        print(sess.run(self.cost))
        #print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name))
        costs = []
        for i in range(1000):
            _, cost = sess.run([self.optimizer, self.cost])
            costs.append(cost)
            print(cost)
            print('Training in progress: step', i)
        
        print(sess.run(self.Qmatrix))
        plt.plot(costs)
        plt.show()

        return


if __name__ == "__main__":

    real_data_corona = False
    real_data_1 = False
    real_data_2 = False
    simulate_data = False
    load_strings = True

    Alphabet_dir = {'A': [1, 0, 0, 0],
                    'C': [0, 1, 0, 0],
                    'G': [0, 0, 1, 0],
                    'T': [0, 0, 0, 1]}
    alphabet_dir = {'a': [1, 0, 0, 0],
                    'c': [0, 1, 0, 0],
                    'g': [0, 0, 1, 0],
                    't': [0, 0, 0, 1]}
                    

    alphabet = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])

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

    csmc.train(20)

    # log_weights, tree_probs, norm, G = csmc.sample_phylogenies(200, resampling=False, showing=True)


















    