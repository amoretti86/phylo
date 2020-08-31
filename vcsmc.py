"""
An implementation of the Variational Combinatorial Sequential Monte Carlo algorithm
  Combinatorial Sequential Monte Carlo is used to form a variational objective
  to simultaneously learn the parameters of the proposal and target distribution
  and perform Bayesian phylogenetic inference.
"""

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import pdb
import operator as op
from functools import reduce
from datestr import addDateTime
import os
import pickle

# @staticmethod
def ncr(n, r):
    # Compute combinatorial term n choose r
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer / denom


# @staticmethod
def sort_string(s):
    lst = s.split('+')
    lst = sorted(lst)
    result = '+'.join(lst)
    return result


class VCSMC:
    """
    VCSMC takes as input a dictionary (datadict) with two keys:
     taxa: a list of n strings denoting taxa
     genome_NxSxA: a 3 tensor of genomes for the n taxa one hot encoded
    """

    def __init__(self, datadict):
        self.taxa = datadict['taxa']
        self.genome_NxSxA = datadict['genome']
        self.n = len(datadict['taxa'])
        self.s = len(self.genome_NxSxA[0])
        self.a = len(self.genome_NxSxA[0, 0])
        # self.pi_c = tf.Variable(0.25, dtype=tf.float64, name="pi_c", constraint=lambda x: tf.clip_by_value(x, 0, 1))
        # self.pi_g = tf.Variable(0.25, dtype=tf.float64, name="pi_g", constraint=lambda x: tf.clip_by_value(x, 0, 1))
        # self.pi_t = tf.Variable(0.25, dtype=tf.float64, name="pi_t", constraint=lambda x: tf.clip_by_value(x, 0, 1-self.pi_c-self.pi_g))
        self.pi_c = tf.constant(0.25, dtype=tf.float64)
        self.pi_g = tf.constant(0.25, dtype=tf.float64)
        self.pi_t = tf.constant(0.25, dtype=tf.float64)
        # Transcription-Transversion factor
        self.kappa = tf.Variable(2., dtype=tf.float64, name="kappa",
                                 constraint=lambda x: tf.clip_by_value(x, 1e-6, 1e6))
        self.Qmatrix = self.get_Q(self.pi_c, self.pi_g, self.pi_t, self.kappa)
        self.Pmatrix = tf.linalg.expm(self.Qmatrix)
        self.state_probs = tf.stack([[1 - self.pi_c - self.pi_g - self.pi_t, self.pi_c, self.pi_g, self.pi_t]], axis=0)
        self.learning_rate = 0.01

    def get_Q(self, pi_c, pi_g, pi_t, kappa):
        # returns Qmatrix with entries defined by these four tf.Variables
        A1 = tf.stack([-pi_c - kappa * pi_g - pi_t, pi_c, kappa * pi_g, pi_t], axis=0)
        A2 = tf.stack([1 - pi_c - pi_g - pi_t, -1 + pi_c + pi_t - kappa * pi_t, pi_g, kappa * pi_t], axis=0)
        A3 = tf.stack([kappa * (1 - pi_c - pi_g - pi_t), pi_c, -kappa * (1 - pi_c - pi_g - pi_t) - pi_c - pi_t, pi_t],
                      axis=0)
        A4 = tf.stack([1 - pi_c - pi_g - pi_t, kappa * pi_c, pi_g, -1 + pi_c + pi_t - kappa * pi_c], axis=0)
        Q = tf.stack((A1, A2, A3, A4), axis=0)
        return Q

    def resample(self, JC_K, log_weights, debug=False):
        """
        Resample partial states by drawing from a categorical distribution whose parameters are normalized importance weights
        JumpChain (JC_K) is a tensor formed from a numpy array of lists of strings, returns a resampled JumpChain tensor
        """
        indices = tf.squeeze(tf.random.categorical(tf.expand_dims(log_weights, axis=0), self.K))
        resampled_JC_K = tf.gather(JC_K, indices)
        return resampled_JC_K, indices

    def extend_partial_state(self, JCK, debug=False):
        """
        Extends partial state by sampling two states to coalesce (Gumbel-max trick to sample without replacement)
        JumpChain (JC_K) is a tensor formed from a numpy array of lists of strings, returns a new JumpChain tensor
        """
        # Compute combinatorial term
        # pdb.set_trace()
        q = 1 / ncr(JCK.shape[1], 2)
        data = np.arange(0, (JCK.shape[1].value) * JCK.shape[0].value, 1).reshape(JCK.shape[0].value,
                                                                                  JCK.shape[1].value) % (
                   JCK.shape[1].value)
        data = tf.constant(data, dtype=tf.int32)
        data = tf.cast(data, dtype=tf.float32)
        # Gumbel-max trick to sample without replacement
        z = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(data), 0, 1)))
        top_values, coalesced_indices = tf.nn.top_k(data + z, 2)
        bottom_values, remaining_indices = tf.nn.top_k(tf.negative(data + z), JCK.shape[1].value - 2)
        JC_keep = tf.gather(tf.reshape(JCK, [JCK.shape[0].value * (JCK.shape[1].value)]), remaining_indices)
        particles = tf.gather(tf.reshape(JCK, [JCK.shape[0].value * (JCK.shape[1].value)]), coalesced_indices)
        particle1 = particles[:, 0]
        particle2 = particles[:, 1]
        # Form new state
        particle_coalesced = particle1 + '+' + particle2
        # Form new Jump Chain
        JCK = tf.concat([JC_keep, tf.expand_dims(particle_coalesced, axis=1)], axis=1)

        return particle1, particle2, particle_coalesced, coalesced_indices, remaining_indices, q, JCK

    def conditional_likelihood(self, l_data, r_data, l_branch, r_branch):
        """
        Computes conditional complete likelihood at an ancestor node
        """
        left_Pmatrix = tf.linalg.expm(self.Qmatrix * l_branch)
        right_Pmatrix = tf.linalg.expm(self.Qmatrix * r_branch)
        left_prob = tf.matmul(l_data, left_Pmatrix)
        right_prob = tf.matmul(r_data, right_Pmatrix)
        likelihood = tf.multiply(left_prob, right_prob)
        return likelihood

    def compute_tree_likelihood(self, data):
        """
        Forms a probability measure by dotting the prior with tree likelihood
        """
        tree_likelihood = tf.matmul(self.state_probs, data, transpose_b=True)
        loglik = tf.reduce_sum(tf.log(tree_likelihood))
        return loglik

    def overcounting_correct(self, v, indices):
        """
        Computes overcounting correction term to the proposal distribution
        """
        idx1 = tf.gather(indices, 0)
        idx2 = tf.gather(indices, 1)
        threshold = self.n - self.i - 1
        cond_greater = tf.where(tf.logical_and(idx1 > threshold, idx2 > threshold), -1., 0.)
        result = tf.where(tf.logical_and(idx1 < threshold, idx2 < threshold), 1., cond_greater)

        v = tf.add(v, tf.cast(result, tf.float64))
        return 1 / v

    def compute_log_ZSMC(self, log_weights):
        """
        Forms the estimator log_ZSMC, a multi sample lower bound to the likelihood
        Z_SMC is formed by averaging over weights and multiplying over coalescent events
        """
        log_Z_SMC = tf.reduce_sum(tf.reduce_logsumexp(log_weights - tf.log(tf.cast(self.K, tf.float64)), axis=1))
        return log_Z_SMC

    def body_update_data(self, new_data, coalesced_indices, k):
        n1 = tf.gather_nd(coalesced_indices, [k, 0])
        n2 = tf.gather_nd(coalesced_indices, [k, 1])
        l_data = tf.squeeze(tf.gather_nd(self.core, [[k, n1]]))
        r_data = tf.squeeze(tf.gather_nd(self.core, [[k, n2]]))
        l_branch = tf.squeeze(tf.gather_nd(self.left_branches, [[k, self.i]]))
        r_branch = tf.squeeze(tf.gather_nd(self.right_branches, [[k, self.i]]))
        mtx = self.conditional_likelihood(l_data, r_data, l_branch, r_branch)
        mtx_ext = tf.expand_dims(tf.expand_dims(mtx, axis=0), axis=0)  # 1x1xSxA
        new_data = tf.concat([new_data, mtx_ext], axis=0)  # kx1xSxA
        k = k + 1
        return new_data, coalesced_indices, k

    def cond_update_data(self, new_data, coalesced_indices, k):
        return k < self.K

    def body_update_weights(self, log_weights_i, log_likelihood_i, log_likelihood_tilda, coalesced_indices, v, q, k):
        log_likelihood_i_k, k, j = tf.while_loop(self.cond_compute_forest, self.body_compute_forest,
                                                 [tf.constant(0, dtype=tf.float64), k, 0])
        log_likelihood_i = tf.concat([log_likelihood_i, [log_likelihood_i_k]], axis=0)
        if self.i > 0:
            v = tf.concat([v, [self.overcounting_correct(tf.gather(v, k), tf.gather(coalesced_indices, k))]], axis=0)
            new_log_weight = tf.gather(log_likelihood_i, k + 1) - tf.gather(log_likelihood_tilda, k) + \
                             tf.math.log(tf.cast(tf.gather(v, k + self.K), tf.float64)) - tf.math.log(
                tf.cast(q, tf.float64))
            log_weights_i = tf.concat([log_weights_i, [new_log_weight]], axis=0)
        else:
            v = tf.concat([v, [tf.constant(1, dtype=tf.float64)]], axis=0)
            log_weights_i = tf.concat([log_weights_i, [tf.constant(0, dtype=tf.float64)]], axis=0)
        k = k + 1
        return log_weights_i, log_likelihood_i, log_likelihood_tilda, coalesced_indices, v, q, k

    def cond_update_weights(self, log_weights_i, log_likelihood_i, log_likelihood_tilda, coalesced_indices, v, q, k):
        return k < self.K

    def body_compute_forest(self, log_likelihood_i_k, k, j):
        log_likelihood_i_k = log_likelihood_i_k + self.compute_tree_likelihood(
            tf.squeeze(tf.gather_nd(self.core, [[k, j]])))
        j = j + 1
        return log_likelihood_i_k, k, j

    def cond_compute_forest(self, log_likelihood_i_k, k, j):
        return j < self.n - 1 - self.i

    def sample_phylogenies(self, K, resampling=True):
        """

        """
        n = self.n;
        s = self.s;
        self.K = K

        log_weights = [tf.constant(0, shape=(K,), dtype=tf.float64) for i in range(n - 1)]
        log_likelihood = [tf.constant(0, shape=(K,), dtype=tf.float64) for i in range(n - 1)]
        log_likelihood_tilda = tf.constant(1, shape=(K,), dtype=tf.float64)

        # Represent a single jump_chain as a list of dictionaries
        jump_chain_tensor = tf.constant([self.taxa] * K, name='JumpChainK')
        # Keep matrices of all vertices, KxNxSxA (the coalesced children nodes will be removed as we go)
        self.core = tf.constant(np.array([self.genome_NxSxA] * K))
        self.left_branches = tf.Variable(np.zeros((K, n - 1)) + 2, dtype=tf.float64,
                                         constraint=lambda x: tf.clip_by_value(x, 1e-6, 1e6))
        self.right_branches = tf.Variable(np.zeros((K, n - 1)) + 2, dtype=tf.float64,
                                          constraint=lambda x: tf.clip_by_value(x, 1e-6, 1e6))
        v = tf.constant(1, shape=(K,), dtype=tf.float64)  # to be used in overcounting_correct

        # Iterate over coalescent events
        for i in range(n - 1):
            self.i = i
            # Resampling step
            if resampling and i > 0:
                jump_chain_tensor, indices = self.resample(jump_chain_tensor, log_weights[i - 1])
                log_likelihood_tilda = tf.gather_nd(tf.gather(tf.transpose(log_likelihood_tf), indices),
                                                    [[k, i - 1] for k in range(K)])

            # Extend partial states
            particle1, particle2, particle_coalesced, coalesced_indices, remaining_indices, \
            q, jump_chain_tensor = self.extend_partial_state(jump_chain_tensor)

            # Save partial set data
            new_data = tf.constant(np.zeros((1, 1, self.s, self.a)))  # to be used in tf.while_loop
            new_data, coalesced_indices, k = tf.while_loop(self.cond_update_data, self.body_update_data,
                                                           loop_vars=[new_data, coalesced_indices, 0],
                                                           shape_invariants=[tf.TensorShape([None, 1, self.s, self.a]),
                                                                             coalesced_indices.get_shape(),
                                                                             tf.TensorShape([])])
            new_data = tf.gather(new_data, list(range(1, K + 1)))  # remove the trivial index 0
            shape1 = self.core.shape[0].value * self.core.shape[1].value
            self.core = tf.gather(tf.reshape(self.core, [shape1, self.core.shape[2].value, self.core.shape[3].value]),
                                  remaining_indices)
            self.core = tf.concat([self.core, new_data], axis=1)

            # Comptue weights
            log_weights_i, log_likelihood_i, log_likelihood_tilda, coalesced_indices, v, q, k = \
                tf.while_loop(self.cond_update_weights, self.body_update_weights,
                              loop_vars=[tf.constant(np.zeros(1)), tf.constant(np.zeros(1)), log_likelihood_tilda,
                                         coalesced_indices, v, q, 0],
                              shape_invariants=[tf.TensorShape([None]), tf.TensorShape([None]),
                                                log_likelihood_tilda.get_shape(), \
                                                coalesced_indices.get_shape(), tf.TensorShape([None]),
                                                tf.TensorShape([]), tf.TensorShape([])])
            log_likelihood[i] = tf.gather(log_likelihood_i, list(range(1, K + 1)))
            log_weights[i] = tf.gather(log_weights_i, list(range(1, K + 1)))
            v = tf.gather(v, list(range(K, 2 * K)))
            log_likelihood_tf = tf.stack([log_likelihood[i] for i in range(n - 1)], axis=0)
        # End of iteration

        log_weights = tf.stack([log_weights[i] for i in range(n - 1)], axis=0)
        elbo = self.compute_log_ZSMC(log_weights)
        self.cost = -elbo
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        return elbo

    def train(self, K, numIters=1000):
        """
        Run the train op in a TensorFlow session and evaluate variables
        """
        self.sample_phylogenies(K)
        print('===================\nFinished constructing computational graph!\n===================')
        # self.get_feed_dict()
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        print('===================\nInitial evaluation of cost:', round(sess.run(self.cost), 3),
              '\n===================')
        print('Training begins --')
        # print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name))
        costs = []
        Qmatrices = []
        left_branches = []
        right_branches = []
        for i in range(numIters):
            _, cost = sess.run([self.optimizer, self.cost])
            # Plot the ELBO, log_ZSMC which is -cost
            costs.append(-cost)
            print('--------------------\nEpoch', i + 1)
            print('Cost\n', round(cost, 3))
            print('Q-matrix\n', sess.run(self.Qmatrix))
            Qmat = sess.run(self.Qmatrix)
            Qmatrices.append(Qmat)
            left_branches.append(sess.run(self.left_branches))
            right_branches.append(sess.run(self.left_branches))
        print("Done training.")
        #plt.plot(costs)
        #plt.show()


        # Create local directory and save experiment results
        local_rlt_root = './rslts/'
        save_dir = local_rlt_root + addDateTime() + '/'
        if not os.path.exists(save_dir): os.makedirs(save_dir)

        plt.imshow(sess.run(self.Qmatrix))
        plt.title("Trained Q matrix")
        plt.savefig(save_dir + "Qmatrix.png")

        plt.figure(figsize=(10,10))
        plt.plot(costs)
        plt.ylabel("log $Z_{SMC}$")
        plt.xlabel("Epochs")
        plt.title("Elbo convergence across epochs")
        plt.savefig(save_dir + "ELBO.png")
        plt.show()

        resultDict = {'Qmatrices': np.asarray(Qmatrices),
                      'cost': np.asarray(costs),
                      'nParticles': self.K,
                      'lr': self.learning_rate,
                      'nTaxa': self.n,
                      'left_branches': np.asarray(left_branches),
                      'right_branches': np.asarray(right_branches)}

        with open(save_dir + 'results.p', 'wb') as f:
            pickle.dump(resultDict, f)

        return

