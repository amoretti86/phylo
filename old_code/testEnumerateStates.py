"""
TensorFlow implementation of resample and extend_partial_state functions from CSMC
States are represented using JumpChain (JC_K), a tensor formed from a numpy array of lists of strings
"""

import numpy as np
import tensorflow as tf
import tensorflow.math
import pdb
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
    Combinatorial term n choose r
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
    #JC_K.assign(resampled_tensor)
    return resampled_tensor


def enumerate_topologies(JCK):
    """
    Enumerate all topologies at state s_{r+1}.
    """

    return JCK_enumerated




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
    return particle1, particle2, particle_coalesced, q2, JCK, coalesced_indices, remaining_indices


def conditional_likelihood(node, Q):
    """
    Compute the conditional likelihood at a given node by passing messages up from left and right children
    """
    # As of 6/4, this version only works with data of one site, i.e. self.s = 1
    # In the future, we might only need this function (and not naive... node... ?)
    left  = node.left.data
    right = node.right.data
    left_p_matrix  = tf.linalg.expm(Q * node.left_branch)
    right_p_matrix = tf.linalg.expm(Q * node.right_branch)
    left_lik  = tf.matmul(left, left_p_matrix)
    right_lik = tf.matmul(right, right_p_matrix)
    lik = left_lik * right_lik
    return lik










if __name__ == "__main__":
    """
    Run some tests...
    """

    n = 5
    K = 10
    taxa = ['S0', 'S1', 'S2', 'S3', 'S4']
    itaxa = [0,1,2,3,4]

    jump_chain = [{} for i in range(n)]
    jump_chain[0][0] = taxa
    jump_chain_KxN = np.array([jump_chain]*K)
    JC_KxN = jump_chain_KxN
    print(jump_chain_KxN)

    alphabet_dir = {'A': [1, 0, 0, 0],
                    'C': [0, 1, 0, 0],
                    'T': [0, 0, 1, 0],
                    'G': [0, 0, 0, 1]}
    alphabet = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    genome_strings = ['ACTTTGAGAG', 'ACTTTGACAG', 'ACTTTGACTG', 'ACTTTGACTC','ACTTTGACAC']
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

    #pdb.set_trace()
    #vdicts = update_vertex_dicts(JCK,0,0,dd)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        j = 0
        i = 0

        pdb.set_trace()
        print("JCK:\n", JCK.eval())

        enumerate_topologies(JCK)

        # Test resample_tensor
        resampled_jump_chain = resample_tensor(weights_KxNm1, JCK, 0, K)
        print("resampled_jump_chain:\n", resampled_jump_chain.eval())

        # Test sampling without replacement

        #p1,p2,pc,q2,new_jck = extend_partial_state_tensor(JCK,j,i)
        #p1, p2, pc, q2, new_jck = extend(JCK, j, i)
        p1, p2, pc, q2, new_jck, c_idx, r_idx = extend_partial_state(JCK)
        print("state 1: ", p1.eval())
        print("state 2: ", p2.eval())
        print("new state: ", pc.eval())
        print("new jump chain for particle k:\n", new_jck.eval())
        print("coalesced indices:\n", c_idx.eval())
        print("remaining indices:\n", r_idx.eval())
        print("combinatorial term: ", q2)
        vd = update_vertex_dicts(JCK,0,0,dd)


        # Can also test computation outside of functions within the session...
        """
        # Take this apart
        data = tf.constant(list(range(4)), dtype=tf.int32)
        data = tf.cast(data, dtype=tf.float32)
        data = np.arange(0, (n-1)*K, 1).reshape(K, n-1) % (n-1)
        data = tf.constant(data, dtype=tf.int32)
        data = tf.cast(data, dtype=tf.float32)

        z = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(data), 0, 1)))
        top_values, coalesced_indices = tf.nn.top_k(data+z, 2)
        bottom_values, remaining_indices = tf.nn.top_k(tf.negative(data+z), JCK.shape[1]-2)
        #particles = tf.gather(JCK,top_indices)
        particles = tf.gather(tf.reshape(JCK, [40]), coalesced_indices)
        JCK_keep = tf.gather(tf.reshape(JCK,[40]), remaining_indices)
        q2 = 1 / ncr(JCK.shape[1], 2)
        particle1 = particles[:,0]
        particle2 = particles[:,1]
        # Form new state
        particle_coalesced = particle1 + '+' + particle2

        # JCK_ = tf.concat(0, [JCK_keep, tf.expand_dims(particle_coalesced,axis=0)])
        # tf.concat([JCK_keep, JCK_keep], axis=0).eval()
        JCK_ = tf.concat([JCK_keep, tf.expand_dims(particle_coalesced, axis=0)], axis=0).eval()
        

        #top_values, top_indices = tf.nn.top_k(tf.reshape(data + z, (-1,)), K*2)


        z = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(data), 0, 1)))
        _, indices = tf.nn.top_k(data + z, 2)
        #tf.negative(data + z)
        #tf.nn.top_k(tf.negative(data + z))
    
        _, coalesced_indices = tf.nn.top_k(data + z, 2)
        __, remaining_indices = tf.nn.top_k(tf.negative(data + z), JCK[j].shape[0] - 2)
        # Collect states to coalesce
        particles = tf.gather(JCK[j],coalesced_indices)
        JCK_keep = tf.gather(JCK[j], remaining_indices)
    
        q2 = 1 / ncr(JCK[j].shape[0], 2)
    
        particle1 = particles[0]
        particle2 = particles[1]
        # Form new state
        particle_coalesced = particle1 + '+' + particle2
    
        #JCK_ = tf.concat(0, [JCK_keep, tf.expand_dims(particle_coalesced,axis=0)])
        #tf.concat([JCK_keep, JCK_keep], axis=0).eval()
        JCK_ = tf.concat([JCK_keep, tf.expand_dims(particle_coalesced,axis=0)], axis=0).eval()
    
    
        # Test extend_partial_state_tensor
        p1, p2, pc, q2, jc = extend_partial_state_tensor(JCK, 0, 0)
        # print("p1.eval()", p1.eval())
        """