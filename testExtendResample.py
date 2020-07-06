"""
TensorFlow implementation of resample and extend_partial_state functions from CSMC
States are represented using JumpChain (JC_K), a tensor formed from a numpy array of lists of strings
"""

import numpy as np
import tensorflow as tf
import tensorflow.math
import pdb
from copy import deepcopy
import random
from functools import reduce
import operator as op


def ncr(n_, r_):
    """
    Combinatorial term n choose r
    """
    _r = min(r_, n_-r_)
    numer = reduce(op.mul, range(n_, n_-r_, -1),1)
    denom = reduce(op.mul, range(1, r_+1), 1)
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


def sample_without_replacement(logits, K):
    """
    Courtesy of https://github.com/tensorflow/tensorflow/issues/9260#issuecomment-437875125
    """
    z = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(logits),0,1)))
    _, indices = tf.nn.top_k(logits + z, K)
    return indices


def extend_partial_state_tensor(JC_K, j, i):
    """
    Extends partial state by sampling two states to coalesce (Gumbel-max trick to sample without replacement)
    JumpChain (JC_K) is a tensor formed from a numpy array of lists of strings, returns a new JumpChain tensor
    """
    # Compute combinatorial term
    q2 = 1 / ncr(JCK[j].shape[0],2)
    # Resample without replacement to select two particles
    data = tf.constant(list(range(JC_K.shape[1])), dtype=tf.int32)
    data = tf.cast(data, dtype=tf.float32)
    # Gumbel-max transformation to resample without replacement
    z = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(data), 0, 1)))
    _, coalesced_indices = tf.nn.top_k(data + z, 2)
    __, remaining_indices = tf.nn.top_k(tf.negative(data + z), JCK[j].shape[0] - 2)
    # Collect states to coalesce
    particles = tf.gather(JC_K[j],coalesced_indices)
    JCK_keep = tf.gather(JCK[j], remaining_indices)
    particle1 = particles[0]
    particle2 = particles[1]
    # Form new state
    particle_coalesced = particle1 + '+' + particle2
    # Form new jump chain
    new_JCK = tf.concat([JCK_keep, tf.expand_dims(particle_coalesced, axis=0)], axis=0)

    return particle1, particle2, particle_coalesced, q2, new_JCK


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

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        #pdb.set_trace()
        print("JCK:\n", JCK.eval())

        # Test resample_tensor
        resampled_jump_chain = resample_tensor(weights_KxNm1, JCK, 0, K)
        print("resampled_jump_chain:\n", resampled_jump_chain.eval())

        # Test sampling without replacement
        j = 0
        i = 0
        p1,p2,pc,q2,new_jck = extend_partial_state_tensor(JCK,j,i)
        print("state 1: ", p1.eval())
        print("state 2: ", p2.eval())
        print("new state: ", pc.eval())
        print("new jump chain for particle k:\n", new_jck.eval())
        print("combinatorial term: ", q2)


        # Can also test computation outside of functions within the session...
        """
        # Take this apart
        data = tf.constant(list(range(4)), dtype=tf.int32)
        data = tf.cast(data, dtype=tf.float32)
    
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
        print("p1.eval()", p1.eval())
        """
