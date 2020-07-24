import numpy as np
import tensorflow as tf
import tensorflow.math
import pdb
from copy import deepcopy
import random
from functools import reduce
import operator as op
import scipy.linalg as spl


class Vertex:

    def __init__(self, id=None, data=None):
        self.id = id
        self.data = data
        self.left = None
        self.right = None
        self.left_branch = None
        self.right_branch = None
        self.is_root = True


def ncr(n, r):
    # Compute combinatorial term n choose r
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer / denom


def resample_tensor(weights_KxNm1, JC_K, i, K, debug=False):
    """
    Resample partial states by drawing from a categorical distribution whose parameters are normalized importance weights
    JumpChain (JC_K) is a tensor formed from a numpy array of lists of strings, returns a resampled JumpChain tensor
    """
    indices = tf.random.categorical(tf.math.log(tf.expand_dims(weights_KxNm1[:, i] / tf.reduce_sum(weights_KxNm1[:, i], axis=0), axis=0)), K)
    resampled_tensor = tf.gather(JC_K, tf.squeeze(indices))
    if debug:
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print("Jump Chain Tensor:\n", sess.run(resampled_tensor))
            print("OH YEAH!")
    return resampled_tensor

def extend_partial_state(JCK,debug=False):
    """
    Extends partial state by sampling two states to coalesce (Gumbel-max trick to sample without replacement)
    JumpChain (JC_K) is a tensor formed from a numpy array of lists of strings, returns a new JumpChain tensor
    """
    # Compute combinatorial term
    #pdb.set_trace()
    #    pdb.set_trace()
    q2 = 1 / ncr(JCK.shape[1],2)
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
    JC_keep = tf.gather(tf.reshape(JCK, [JCK.shape[0].value * (JCK.shape[1].value)]), remaining_indices)
    particles = tf.gather(tf.reshape(JCK, [JCK.shape[0].value*(JCK.shape[1].value)]), coalesced_indices)
    particle1 = particles[:, 0]
    particle2 = particles[:, 1]
    # Form new state
    particle_coalesced = particle1 + '+' + particle2
    # Form new Jump Chain
    JCK = tf.concat([JC_keep, tf.expand_dims(particle_coalesced, axis=1)], axis=1)
    return particle1, particle2, particle_coalesced, q2, JCK

def form_final_state(JCK):
    """
    Frorm the final state
    """
    q2 = 1 / ncr(JCK.shape[1], 2)
    p1 = JCK[:,0]
    p2 = JCK[:,1]
    pc = JCK[:, 0] + "+" + JCK[:, 1]
    JCK = pc
    return p1, p2, pc, q2, JCK

def sample_phylogenies(datadict, K, resampling=True, showing=True):
    """
    Rewrite sampler
    """
    testExtend = True
    testResample = True

    n = len(datadict['taxa'])
    taxa = datadict['taxa']
    genome_NxSxA = datadict['genome']
    s = len(genome_NxSxA[0])
    Qmatrix = np.array([[-3., 1., 1., 1.],
                        [1., -3., 1., 1.],
                        [1., 1., -3., 1.],
                        [1., 1., 1., -3.]])
    Pmatrix = spl.expm(Qmatrix)
    prior = np.ones(Qmatrix.shape[0]) / Qmatrix.shape[0]

    # Represent a single jump_chain as a list of dictionaries
    #jump_chain = [{} for i in range(n)]
    #jump_chain[0][0] = taxa
    #jump_chain_KxN = np.array([jump_chain] * K)  # KxN matrix of jump_chain dictionaries

    JC_K = [taxa] * K
    print(JC_K)
    JumpChainTensor = tf.Variable(JC_K, name='JumpChainK')

    log_weights_KxNm1 = np.zeros([K, n - 1])
    log_weights_KxNm1[:, 0] = np.log(1)
    # Hack to test code without evaluating the likelihood
    weights = np.random.uniform(0, 1, K * (n - 1)).reshape([K, n - 1])
    # weights = np.zeros([K, n - 1])
    weights_KxNm1 = tf.Variable(weights, dtype=tf.float64, name='weights_KxNm1')
    #weights_KxNm1 = tf.Variable(log_weights_KxNm1, dtype=tf.float64, name='weights_KxNm1')

    log_likelihood = np.zeros([K, n - 1])
    log_likelihood = tf.Variable(log_likelihood, dtype=tf.float64, name='log_likelihood')

    # pdb.set_trace()
    graph_repn_data_K = [[] for i in range(K)]
    graph_repn_nodes_K = [[] for i in range(K)]

    vertex_dicts = [{} for k in range(K)]
    for j in range(K):
        for i in range(n):
            vertex_id = tf.Variable(taxa[i], name=taxa[i])
            vertex_dicts[j][taxa[i]] = Vertex(id=taxa[i], data=genome_NxSxA[i])

    #pdb.set_trace()
    # Iterate over coalescent events
    for i in range(n - 1):

        # Resampling step
        if resampling and i > 0:
            #jump_chain_KxN[:, i - 1] = resample(log_weights_KxNm1, jump_chain_KxN[:, i - 1], i - 1)
            JumpChainTensor = resample_tensor(weights_KxNm1, JumpChainTensor, i - 1, K)
            print("Resampled JumpChainTensor:\n", JumpChainTensor)
            if testResample:
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    print("Jump Chain Tensor:\n", sess.run(JumpChainTensor))
                    print("oh yeah...!")

        #pdb.set_trace()
        r = n-i
        print("rank %i" %r, "iter %i" %i)
        if r > 2:
            p1,p2,pc,q,JumpChainTensor = extend_partial_state(JumpChainTensor,debug=True)
            print("Extended Jump Chain Tensor:\n", JumpChainTensor)

        elif r == 2:
            p1,p2,pc,q,JumpChainTensor = form_final_state(JumpChainTensor)
            print("Extended JumpChainTensor:\n", JumpChainTensor)
            #pdb.set_trace()


        #for k in range(K):
            #jctmp = tf.stack([jctmp, jc])

        if False:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                print("Jump Chain Tensor:\n", sess.run(JumpChainTensor))
                print("oh yeah...")

            #vertex_dicts_laststep = deepcopy(vertex_dicts)
        print('Computation in progress: step ' + str(i + 1))
    # End of for loop

    # Compute importance weights across all ranks:
    # tree_probabilities = np.exp(np.sum(log_weights_KxNm1, axis=1))
    # tree_log_probabilities = np.sum(log_weights_KxNm1, axis=1)
    # print(tree_log_probabilities)
    if False:
        trees = []
        for dic in vertex_dicts:
            trees.append(dic.keys())
        tree_probabilities = []
        for i in range(len(trees)):
            tree = trees[i]
            tree_probabilities.append(0)
            for k in range(K):
                if tree == trees[k]:
                    tree_probabilities[i] += weights_KxNm1[k, -1]
            tree_probabilities[i] /= K
        tree_probabilities /= 1 / K * sum(weights_KxNm1[:, -1])
        tree_probabilities = list(tree_probabilities)

        # print(weights_KxNm1)
        print(tree_probabilities)
        norm = 1
        for i in range(2, n - 1):
            norm *= 1 / K * sum(weights_KxNm1[:, i])
        print(norm)

    # plt.hist(tree_probabilities)
    # plt.xlabel('Posterior probability')
    # plt.ylabel('Number of generated trees')
    # plt.show()
    # np.savetxt('tree_prob.csv', tree_probabilities, delimiter=',')
        selected_idx = tree_probabilities.index(max(tree_probabilities))
        print(trees[selected_idx])



    return JumpChainTensor# log_weights_KxNm1, tree_probabilities, norm, G


if __name__ == "__main__":

    alphabet = 'ACTG'
    alphabet_dir = {'A': [1, 0, 0, 0],
                    'C': [0, 1, 0, 0],
                    'T': [0, 0, 1, 0],
                    'G': [0, 0, 0, 1]}
    alphabet_dir_l = {'a': [1, 0, 0, 0],
                      'c': [0, 1, 0, 0],
                      't': [0, 0, 1, 0],
                      'g': [0, 0, 0, 1]}

    alphabet = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    genome_strings = ['ACTTTGAGAG', 'ACTTTGACAG', 'ACTTTGACTG', 'ACTTTGACTC']


    # genome_strings = ['AAAAAA','CCCCCC','TTTTTT','GGGGGG']
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
        # pdb.set_trace()
        genomes_NxSxA = np.zeros([len(genome_strings), len(genome_strings[0]), len(alphabet_dir)])
        for i in range(genomes_NxSxA.shape[0]):
            for j in range(genomes_NxSxA.shape[1]):
                genomes_NxSxA[i, j] = alphabet_dir[genome_strings[i][j]]

        taxa = ['S' + str(i) for i in range(genomes_NxSxA.shape[0])]

        datadict = {'taxa': taxa,
                    'genome': genomes_NxSxA}
        return datadict


    dd = form_dataset_from_strings(genome_strings, alphabet_dir)
    print(dd['taxa'])
    #pdb.set_trace()

    #log_weights, tree_prob, norm, G = sample_phylogenies(dd, 10)
    jct = sample_phylogenies(dd, 10)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("Jump Chain Tensor:\n", sess.run(jct))
        print("oh yeah...")