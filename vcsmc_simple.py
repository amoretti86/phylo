import numpy as np
import tensorflow as tf
import tensorflow.math
import pdb
from copy import deepcopy
import random
from functools import reduce
import operator as op

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
    """
    Combinatorial term
    """
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1) ,1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer/denom


def resample_tensor(weights_KxNm1, JC_K, i, K):
    """
    Resample partial states by drawing from a categorical distribution whose parameters are normalized importance weights
    JumpChain (JC_K) is a tensor formed from a numpy array of lists of strings, returns a resampled JumpChain tensor
    """
    indices = tf.random.categorical(tf.math.log(tf.expand_dims(weights_KxNm1[:,i]/tf.reduce_sum(weights_KxNm1[:,i], axis=0), axis=0)),K)
    resampled_tensor = tf.gather(JC_K, indices)
    return resampled_tensor

def extend_partial_state(JCK,debug=False):
    """
    Extends partial state by sampling two states to coalesce (Gumbel-max trick to sample without replacement)
    JumpChain (JC_K) is a tensor formed from a numpy array of lists of strings, returns a new JumpChain tensor
    """
    # pdb.set_trace()
    q2 = 1 / ncr(JCK.shape[1],2)
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
    return particle1, particle2, particle_coalesced, q2, JCK, coalesced_indices, remaining_indices

def conditional_likelihood(node, Q):
    """
    Compute the conditional likelihood at a given node by passing messages up from left and right children
    """
    left_Pmatrix = tf.linalg.expm(Q * node.left_branch)
    right_Pmatrix = tf.linalg.expm(Q * node.right_branch)
    left_prob = tf.matmul(node.left.data, left_Pmatrix)
    right_prob = tf.matmul(node.right.data, right_Pmatrix)
    likelihood = tf.multiply(left_prob, right_prob)
    return likelihood




if __name__ == "__main__":
    """
    Run some tests...
    """


    taxa = ['S0', 'S1', 'S2', 'S3', 'S4']
    itaxa = [0,1,2,3,4]
    n = len(taxa)
    K = 10

    alphabet_dir = {'A': [1, 0, 0, 0],
                    'C': [0, 1, 0, 0],
                    'T': [0, 0, 1, 0],
                    'G': [0, 0, 0, 1]}
    alphabet = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    genome_strings = ['ACTTTGAGAG', 'ACTTTGACAG', 'ACTTTGACTG', 'ACTTTGACTC', 'ACTTTGACAC']


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

    Q = np.array([[-3.,1.,1.,1.],[1.,-3.,1.,1.],[1.,1.,-3,1.],[1.,1.,1.,-3]])
    Qmatrix = tf.Variable(Q, dtype=tf.float64, shape=(4, 4), name="Q_matrix")


    weights = np.random.uniform(0,1,K*(n-1)).reshape([K,n-1])
    print(weights)
    weights_KxNm1 = tf.Variable(weights, dtype=tf.float64, name='weights_KxNm1')

    # Generate some test data
    taxa_A = ['S0+S3','S1','S2','S4']
    taxa_B = ['S0','S1+S2','S3','S4']
    taxa_C = ['S0','S1','S2','S3+S4']

    JC_K_test_1 = np.array(np.array(['' for _ in range(40)], dtype=object)).reshape((10,4))
    JC_K_test_1[:3] = taxa_A
    JC_K_test_1[3:7] = taxa_B
    JC_K_test_1[7:] = taxa_C
    JCK = tf.Variable(JC_K_test_1)

    JCK_init = dd['taxa']
    JCK_init = [JCK_init] * K
    JCK = tf.Variable(JCK_init)



    root = Vertex('A')
    root.left = Vertex('B', tf.constant(np.expand_dims(np.array([0., 0., 1., 0.]),axis=0), name='node_b_data'))
    root.right = Vertex('C', tf.constant(np.expand_dims(np.array([0., 1., 0., 0.]),axis=0), name='node_c_data'))
    root.left_branch = tf.constant(1., dtype=tf.float64)
    root.right_branch = tf.constant(1., dtype=tf.float64)

    # For reference this is how we initialized vertex_dicts
    vertex_dicts = [{} for k in range(K)]
    for j in range(K):
        for i in range(n):
            # Key is a string, value is a Vertex
            vertex_dicts[j][dd['taxa'][i]] = Vertex(id=dd['taxa'][i], data=dd['genome'][i])
    # We need to keep track of vertex objects in order to compute the likelihood

    vertex_lists = [[] for k in range(K)]
    taxa_lists = [[] for k in range(K)]
    taxa_idx_lists = [[] for k in range(K)]
    tf_v_lists = tf.Variable(vertex_lists)
    t_idx_lists = tf.Variable(taxa_idx_lists)

    for j in range(K):
        for i in range(n):
            #pdb.set_trace()
            vertex_lists[j].append(Vertex(id=dd['taxa'][i], data=dd['genome'][i]))
            taxa_lists[j].append(dd['taxa'][i])
            taxa_idx_lists[j].append(i)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        pdb.set_trace()
        print("JCK:\n", JCK.eval())

        # Test resample_tensor
        resampled_jump_chain = resample_tensor(weights_KxNm1, JCK, 0, K)
        print("resampled_jump_chain:\n", resampled_jump_chain.eval())

        p1, p2, pc, q2, new_jck, c_idx, r_idx = extend_partial_state(JCK)
        print("state 1: ", p1.eval())
        print("state 2: ", p2.eval())
        print("new state: ", pc.eval())
        print("new jump chain for particle k:\n", new_jck.eval())
        print("coalesced indices:\n", c_idx.eval())
        print("remaining indices:\n", r_idx.eval())
        print("combinatorial term: ", q2)

        # Need to resolve how to pass vertex objects into the conditional_likelihood computation
        test = Vertex(id=pc[0], data=None)
        vertex_lists[0]
        taxa_lists[0]
        taxa_idx_lists[0]

        #test.left  = Vertex(id=p1[0])
        #test.right = Vertex(id=p2[0])

        # Test likelihood computation
        lik = conditional_likelihood(root, Qmatrix)
        print(lik.eval())

