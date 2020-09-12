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
from datetime import datetime
import os
import pickle

# @staticmethod
def ncr(n, r):
    # Compute combinatorial term n choose r
    numer = tf.reduce_prod(tf.range(n-r+1, n+1))
    denom = tf.reduce_prod(tf.range(1, r+1))
    return numer / denom

# @staticmethod
def double_factorial(n):
    # Compute double factorial: n!!
    return tf.reduce_prod(tf.range(n, 0, -2))


# branches = self.q_branches.sample('A Certain Number')
class VCSMC:
    """
    VCSMC takes as input a dictionary (datadict) with two keys:
     taxa: a list of n strings denoting taxa
     genome_NxSxA: a 3 tensor of genomes for the n taxa one hot encoded
    """

    def __init__(self, datadict, K):
        self.taxa = datadict['taxa']
        self.genome_NxSxA = datadict['genome']
        self.K = K
        self.N = len(self.genome_NxSxA)
        self.S = len(self.genome_NxSxA[0])
        self.A = len(self.genome_NxSxA[0, 0])
        self.y_q = tf.linalg.set_diag(tf.Variable(np.zeros((self.A, self.A)) + 1/self.A, dtype=tf.float64, name='Qmatrix'), [0]*self.A)
        #self.y_q = tf.linalg.set_diag(tf.constant(np.zeros((self.A, self.A)) + 1/self.A, dtype=tf.float64), [0]*self.A)
        #self.y_q = tf.linalg.set_diag(tf.constant([[10,0.1,0.1,10],[10,10,0.1,10],[10,0.1,10,10],[10,0.1,0.1,10]], dtype=tf.float64), [0]*self.A)
        #self.y_station = tf.Variable(np.zeros(self.A)+1/self.A, dtype=tf.float64, name='Stationary_probs')
        self.y_station = tf.constant(np.zeros(self.A)+1/self.A, dtype=tf.float64, name='Stationary_probs')
        self.l = tf.Variable(1., dtype=tf.float64, constraint=lambda x: tf.clip_by_value(x, 1e-6, 1e8), name='l')
        self.q_branches = tfp.distributions.Exponential(rate=self.l)
        self.left_branches = self.q_branches.sample((self.K, self.N - 1))
        self.right_branches = self.q_branches.sample((self.K, self.N - 1))
        #self.left_branches = tf.constant(np.zeros((self.K, self.N - 1)) + 1)
        #self.right_branches = tf.constant(np.zeros((self.K, self.N - 1)) + 1)
        self.stationary_probs = self.get_stationary_probs()
        self.Qmatrix = self.get_Q()
        self.learning_rate = tf.placeholder(dtype=tf.float64, shape=[])

    def get_stationary_probs(self):
        denom = tf.reduce_sum(tf.exp(self.y_station))
        return tf.expand_dims(tf.exp(self.y_station) / denom, axis=0)

    def get_Q(self):
        denom = tf.reduce_sum(tf.linalg.set_diag(tf.exp(self.y_q), [0]*self.A), axis=1)
        denom = tf.stack([denom]*self.A, axis=1)
        q_entry = tf.multiply(tf.linalg.set_diag(tf.exp(self.y_q), [0]*self.A), 1/denom)
        hyphens = tf.reduce_sum(q_entry, axis=1)
        Q = tf.linalg.set_diag(q_entry, -hyphens)
        return Q

    def resample(self, core, leafnode_record, JC_K, log_weights):
        """
        Resample partial states by drawing from a categorical distribution whose parameters are normalized importance weights
        JumpChain (JC_K) is a tensor formed from a numpy array of lists of strings, returns a resampled JumpChain tensor
        """
        log_normalized_weights = log_weights - tf.reduce_logsumexp(log_weights)
        indices = tf.squeeze(tf.random.categorical(tf.expand_dims(log_normalized_weights, axis=0), self.K))
        resampled_core = tf.gather(core, indices)
        resampled_record = tf.gather(leafnode_record, indices)
        resampled_JC_K = tf.gather(JC_K, indices)
        #self.left_branches = tf.gather(self.left_branches, indices)
        #self.right_branches = tf.gather(self.right_branches, indices)
        return resampled_core, resampled_record, resampled_JC_K, indices

    def extend_partial_state(self, JCK, r):
        """
        Extends partial state by sampling two states to coalesce (Gumbel-max trick to sample without replacement)
        JumpChain (JC_K) is a tensor formed from a numpy array of lists of strings, returns a new JumpChain tensor
        """
        # Compute combinatorial term
        # pdb.set_trace()
        q = 1 / ncr(self.N - r, 2)
        data = tf.reshape(tf.range((self.N - r) * self.K), (self.K, self.N - r))
        data = tf.mod(data, (self.N - r))
        data = tf.cast(data, dtype=tf.float32)
        # Gumbel-max trick to sample without replacement
        z = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(data), 0, 1)))
        top_values, coalesced_indices = tf.nn.top_k(data + z, 2)
        bottom_values, remaining_indices = tf.nn.top_k(tf.negative(data + z), self.N - r - 2)
        JC_keep = tf.gather(tf.reshape(JCK, [self.K * (self.N - r)]), remaining_indices)
        particles = tf.gather(tf.reshape(JCK, [self.K * (self.N - r)]), coalesced_indices)
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

    def compute_tree_likelihood(self, data, leafnode_num):
        """
        Forms a log probability measure by dotting the stationary probs with tree likelihood
        And add that to log-prior of tree topology
        NOTE: we add log-prior of branch-lengths in body_update_weights
        """
        tree_likelihood = tf.matmul(self.stationary_probs, data, transpose_b=True)
        data_loglik = tf.reduce_sum(tf.log(tree_likelihood))
        tree_logprior = tf.log(1 / double_factorial(2 * tf.maximum(leafnode_num, 2) - 3))

        return data_loglik + tree_logprior

    def overcounting_correct(self, v, indices, i):
        """
        Computes overcounting correction term to the proposal distribution
        """
        idx1 = tf.gather(indices, 0)
        idx2 = tf.gather(indices, 1)
        threshold = self.N - i - 1
        cond_greater = tf.where(tf.logical_and(idx1 > threshold, idx2 > threshold), -1., 0.)
        result = tf.where(tf.logical_and(idx1 < threshold, idx2 < threshold), 1., cond_greater)
        v = tf.add(v, tf.cast(result, tf.float64))
        return v

    def compute_log_ZSMC(self, log_weights):
        """
        Forms the estimator log_ZSMC, a multi sample lower bound to the likelihood
        Z_SMC is formed by averaging over weights and multiplying over coalescent events
        """
        log_Z_SMC = tf.reduce_sum(tf.reduce_logsumexp(log_weights - tf.log(tf.cast(self.K, tf.float64)), axis=1))
        return log_Z_SMC

    def body_update_data(self, new_data, new_record, core, leafnode_record, coalesced_indices, i, k):
        n1 = tf.gather_nd(coalesced_indices, [k, 0])
        n2 = tf.gather_nd(coalesced_indices, [k, 1])
        l_data = tf.squeeze(tf.gather_nd(core, [[k, n1]]))
        r_data = tf.squeeze(tf.gather_nd(core, [[k, n2]]))
        l_branch = tf.squeeze(tf.gather_nd(self.left_branches, [[k, i]]))
        r_branch = tf.squeeze(tf.gather_nd(self.right_branches, [[k, i]]))
        mtx = self.conditional_likelihood(l_data, r_data, l_branch, r_branch)
        mtx_ext = tf.expand_dims(tf.expand_dims(mtx, axis=0), axis=0)  # 1x1xSxA
        new_data = tf.concat([new_data, mtx_ext], axis=0)  # kx1xSxA

        leafnode_new = tf.squeeze(tf.gather_nd(leafnode_record, [[k, n1]])) + \
                       tf.squeeze(tf.gather_nd(leafnode_record, [[k, n2]]))
        new_record = tf.concat([new_record, [[leafnode_new]]], axis=0)

        k = k + 1
        return new_data, new_record, core, leafnode_record, coalesced_indices, i, k

    def cond_update_data(self, new_data, new_record, core, leafnode_record, coalesced_indices, i, k):
        return k < self.K

    def body_compute_forest(self, log_likelihood_i_k, core, leafnode_record, i, k, j):
        data = tf.squeeze(tf.gather_nd(core, [[k, j]]))
        leafnode_num = tf.squeeze(tf.gather_nd(leafnode_record, [[k, j]]))
        log_likelihood_i_k = log_likelihood_i_k + \
          self.compute_tree_likelihood(data, leafnode_num)
        j = j + 1
        return log_likelihood_i_k, core, leafnode_record, i, k, j

    def cond_compute_forest(self, log_likelihood_i_k, core, leafnode_record, i, k, j):
        return j < self.N - 1 - i

    def cond_true_update_weights(self, log_weights_i, log_likelihood_i, log_likelihood_tilda, coalesced_indices, v, q, i, k):
        v = tf.concat([v, [self.overcounting_correct(tf.gather(v, k), tf.gather(coalesced_indices, k), i)]], axis=0)
        v_minus = tf.math.log(1 / tf.cast(tf.gather(v, k + self.K), tf.float64))
        l_branch = tf.squeeze(tf.gather_nd(self.left_branches, [[k, i]]))
        r_branch = tf.squeeze(tf.gather_nd(self.right_branches, [[k, i]]))
        v_plus = tf.math.log(tf.cast(q, tf.float64)) + tf.log(self.l) - self.l * l_branch + tf.log(self.l) - self.l * r_branch
        new_log_weight = tf.gather(log_likelihood_i, k + 1) - tf.gather(log_likelihood_tilda, k) + v_minus - v_plus 
        log_weights_i = tf.concat([log_weights_i, [new_log_weight]], axis=0)
        return v, log_weights_i

    def cond_false_update_weights(self, log_weights_i, log_likelihood_i, log_likelihood_tilda, coalesced_indices, v, q, i, k):
        v = tf.concat([v, [tf.constant(1, dtype=tf.float64)]], axis=0)
        v_minus = tf.math.log(1 / tf.cast(tf.gather(v, k + self.K), tf.float64))
        l_branch = tf.squeeze(tf.gather_nd(self.left_branches, [[k, i]]))
        r_branch = tf.squeeze(tf.gather_nd(self.right_branches, [[k, i]]))
        v_plus = tf.log(tf.cast(q, tf.float64)) + tf.log(self.l) - self.l * l_branch + tf.log(self.l) - self.l * r_branch
        new_log_weight = tf.gather(log_likelihood_i, k + 1) - tf.gather(log_likelihood_tilda, k) + v_minus - v_plus 
        log_weights_i = tf.concat([log_weights_i, [new_log_weight]], axis=0)
        return v, log_weights_i

    def body_update_weights(self, log_weights_i, log_likelihood_i, log_likelihood_tilda, coalesced_indices, core, leafnode_record, v, q, i, k):
        log_likelihood_i_k, core, leafnode_record, i, k_, j = tf.while_loop(self.cond_compute_forest, self.body_compute_forest,
                                                 loop_vars=[tf.constant(0, dtype=tf.float64), core, leafnode_record, i, k, tf.constant(0)])
        left_branches_select = tf.gather(tf.gather(self.left_branches, k), tf.range(i+1))
        right_branches_select = tf.gather(tf.gather(self.right_branches, k), tf.range(i+1))
        left_branches_logprior = tf.reduce_sum(-self.l * left_branches_select + tf.log(self.l))
        right_branches_logprior = tf.reduce_sum(-self.l * right_branches_select + tf.log(self.l))
        log_likelihood_i_k = log_likelihood_i_k + left_branches_logprior + right_branches_logprior

        log_likelihood_i = tf.concat([log_likelihood_i, [log_likelihood_i_k]], axis=0)
        v, log_weights_i = tf.cond(i > 0, 
            lambda: self.cond_true_update_weights(log_weights_i, log_likelihood_i, log_likelihood_tilda, coalesced_indices, v, q, i, k),
            lambda: self.cond_false_update_weights(log_weights_i, log_likelihood_i, log_likelihood_tilda, coalesced_indices, v, q, i, k))
        k = k + 1
        return log_weights_i, log_likelihood_i, log_likelihood_tilda, coalesced_indices, core, leafnode_record, v, q, i, k

    def cond_update_weights(self, log_weights_i, log_likelihood_i, log_likelihood_tilda, coalesced_indices, core, leafnode_record, v, q, i, k):
        return k < self.K

    def cond_true_resample(self, log_likelihood_tilda, core, leafnode_record, log_weights, log_likelihood, jump_chains, jump_chain_tensor, r):
        core, leafnode_record, jump_chain_tensor, indices = self.resample(core, leafnode_record, jump_chain_tensor, tf.gather(log_weights, r))
        log_likelihood_tilda = tf.gather_nd(tf.gather(tf.transpose(log_likelihood), indices),
                                            [[k, r] for k in range(self.K)])
        jump_chains = tf.concat([jump_chains, jump_chain_tensor], axis=1)
        return log_likelihood_tilda, core, leafnode_record, jump_chains, jump_chain_tensor

    def cond_false_resample(self, log_likelihood_tilda, core, leafnode_record, log_weights, log_likelihood, jump_chains, jump_chain_tensor, r):

        return log_likelihood_tilda, core, leafnode_record, jump_chains, jump_chain_tensor

    def body_main(self, log_weights, log_likelihood, log_likelihood_tilda, jump_chains, jump_chain_tensor, core, leafnode_record, v, r):
        # Resample
        log_likelihood_tilda, core, leafnode_record, jump_chains, jump_chain_tensor = tf.cond(r > 0, 
            lambda: self.cond_true_resample(log_likelihood_tilda, core, leafnode_record, log_weights, log_likelihood, jump_chains, jump_chain_tensor, r),
            lambda: self.cond_false_resample(log_likelihood_tilda, core, leafnode_record, log_weights, log_likelihood, jump_chains, jump_chain_tensor, r))

        # Extend partial states
        particle1, particle2, particle_coalesced, coalesced_indices, remaining_indices, \
        q, jump_chain_tensor = self.extend_partial_state(jump_chain_tensor, r)

        # Update partial set data
        new_data = tf.constant(np.zeros((1, 1, self.S, self.A)))  # to be used in tf.while_loop
        new_record = tf.constant(np.zeros((1, 1)), dtype=tf.int32)  # to be used in tf.while_loop
        new_data, new_record, core_, leafnode_record_, coalesced_indices, i, k = tf.while_loop(self.cond_update_data, self.body_update_data,
                                                       loop_vars=[new_data, new_record, core, leafnode_record, coalesced_indices, r, tf.constant(0)],
                                                       shape_invariants=[tf.TensorShape([None, 1, self.S, self.A]), 
                                                                         tf.TensorShape([None, 1]), core.get_shape(), 
                                                                         leafnode_record.get_shape(), coalesced_indices.get_shape(),
                                                                         tf.TensorShape([]), tf.TensorShape([])])
        new_data = tf.gather(new_data, tf.range(1, self.K + 1)) # remove the trivial index 0
        new_record = tf.gather(new_record, tf.range(1, self.K + 1)) # remove the trivial index 0
        core = tf.gather(tf.reshape(core, [self.K * (self.N - r), self.S, self.A]), remaining_indices)
        core = tf.concat([core, new_data], axis=1)
        leafnode_record = tf.gather(tf.reshape(leafnode_record, [self.K * (self.N - r)]), remaining_indices)
        leafnode_record = tf.concat([leafnode_record, new_record], axis=1)

        # Comptue weights
        log_weights_i, log_likelihood_i, log_likelihood_tilda, coalesced_indices, core_, leafnode_record_, v, q, i, k = \
            tf.while_loop(self.cond_update_weights, self.body_update_weights,
                          loop_vars=[tf.constant(np.zeros(1)), tf.constant(np.zeros(1)), log_likelihood_tilda,
                                     coalesced_indices, core, leafnode_record, v, q, r, tf.constant(0)],
                          shape_invariants=[tf.TensorShape([None]), tf.TensorShape([None]),
                                            log_likelihood_tilda.get_shape(), coalesced_indices.get_shape(), 
                                            core.get_shape(), leafnode_record.get_shape(), tf.TensorShape([None]),
                                            tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([])])
        log_weights = tf.concat([log_weights, [tf.gather(log_weights_i, tf.range(1, self.K + 1))]], axis=0)
        log_likelihood = tf.concat([log_likelihood, [tf.gather(log_likelihood_i, tf.range(1, self.K + 1))]], axis=0) # pi(s) = pi(Y|t, b, theta) * pi(t, b|theta) / pi(Y)
        v = tf.gather(v, tf.range(self.K, 2 * self.K))
        r = r + 1
        return log_weights, log_likelihood, log_likelihood_tilda, jump_chains, jump_chain_tensor, core, leafnode_record, v, r

    def cond_main(self, log_weights, log_likelihood, log_likelihood_tilda, jump_chains, jump_chain_tensor, core, leafnode_record, v, r):
        return r < self.N - 1

    def body_initialize_tilda(self, log_likelihood_tilda, core, leafnode_record, i, k):
        log_likelihood_tilda_k, core, leafnode_record, i, k_, j = tf.while_loop(self.cond_compute_forest, self.body_compute_forest,
                                                 loop_vars=[tf.constant(0, dtype=tf.float64), core, leafnode_record, i, k, tf.constant(0)])
        log_likelihood_tilda = tf.concat([log_likelihood_tilda, [log_likelihood_tilda_k]], axis=0)
        k = k + 1
        return log_likelihood_tilda, core, leafnode_record, i, k

    def cond_initialize_tilda(self, log_likelihood_tilda, core, leafnode_record, i, k):
        return k < self.K

    def sample_phylogenies(self):
        """

        """
        N = self.N
        S = self.S
        A = self.A
        K = self.K

        self.core = tf.placeholder(dtype=tf.float64, shape=(K, N, S, A))
        leafnode_record = tf.constant(1, shape=(K, N), dtype=tf.int32) # Keeps track of self.core

        log_weights = tf.constant(0, shape=(1, K), dtype=tf.float64)
        log_likelihood = tf.constant(0, shape=(1, K), dtype=tf.float64)
        log_likelihood_tilda = tf.constant(np.zeros(1), dtype=tf.float64)
        log_likelihood_tilda, core_, leafnode_record_, i, k = tf.while_loop(self.cond_initialize_tilda, self.body_initialize_tilda,
            loop_vars=[log_likelihood_tilda, self.core, leafnode_record, tf.constant(0), tf.constant(0)],
            shape_invariants=[tf.TensorShape([None]), self.core.get_shape(), leafnode_record.get_shape(),
                              tf.TensorShape([]), tf.TensorShape([])])
        log_likelihood_tilda = tf.gather(log_likelihood_tilda, list(range(1, K + 1)))

        self.jump_chains = tf.constant('', shape=(K, 1))
        self.jump_chain_tensor = tf.constant([self.taxa] * K, name='JumpChainK')
        v = tf.constant(1, shape=(K, ), dtype=tf.float64)  # to be used in overcounting_correct

        log_weights, log_likelihood, log_likelihood_tilda, self.jump_chains, self.jump_chain_tensor, core_final, record_final, v, r = \
          tf.while_loop(self.cond_main, self.body_main, 
            loop_vars=[log_weights, log_likelihood, log_likelihood_tilda, self.jump_chains, 
                       self.jump_chain_tensor, self.core, leafnode_record, v, tf.constant(0)],
            shape_invariants=[tf.TensorShape([None, K]), tf.TensorShape([None, K]), log_likelihood_tilda.get_shape(),
                              tf.TensorShape([K, None]), tf.TensorShape([K, None]), tf.TensorShape([K, None, S, A]), 
                              tf.TensorShape([K, None]), v.get_shape(), tf.TensorShape([])])

        log_weights = tf.gather(log_weights, list(range(1, N))) # remove the trivial index 0
        log_likelihood = tf.gather(log_likelihood, list(range(1, N))) # remove the trivial index 0
        elbo = self.compute_log_ZSMC(log_weights)
        self.cost = -elbo
        self.log_weights = log_weights
        self.log_likelihood = log_likelihood
        self.log_likelihood_tilda = log_likelihood_tilda
        self.v = v

        return elbo

    def train(self, numIters=800, memory_optimization='on'):
        """
        Run the train op in a TensorFlow session and evaluate variables
        """
        K = self.K

        config = tf.ConfigProto()
        if memory_optimization == 'off':
            from tensorflow.core.protobuf import rewriter_config_pb2
            off = rewriter_config_pb2.RewriterConfig.OFF
            config.graph_options.rewrite_options.memory_optimization = off

        bt = datetime.now()
        self.sample_phylogenies()
        at = datetime.now()
        print('===================\nFinished constructing computational graph!', '\nTime spent:', at-bt, '\n===================')

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        feed_data = np.array([self.genome_NxSxA] * K, dtype=np.double)

        sess = tf.Session(config=config)
        init = tf.global_variables_initializer()
        sess.run(init)
        print('===================\nInitial evaluation of ELBO:', round(sess.run(-self.cost, feed_dict={self.core: feed_data}), 3), '\n===================')
        print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name))
        print('Training begins --')
        elbos = []
        Qmatrices = []
        left_branches = []
        right_branches = []
        jump_chain_evolution = []
        log_weights = []
        ll = []
        ll_tilda = []
        for i in range(numIters):
            bt = datetime.now()
            _, cost = sess.run([self.optimizer, self.cost], feed_dict={self.core: feed_data, self.learning_rate: 0.01})
            elbos.append(-cost)
            print('Epoch', i+1)
            print('ELBO\n', round(-cost, 3))
            print('Q-matrix\n', sess.run(self.Qmatrix))
            print('Stationary probabilities\n', sess.run(self.stationary_probs))
            # print('Left branches (for five particles)\n', sess.run(tf.gather(self.left_branches, [0, int(K/4), int(K/2), int(3*K/4), K-1])))
            # print('Right branches (for five particles)\n', sess.run(tf.gather(self.right_branches, [0, int(K/4), int(K/2), int(3*K/4), K-1])))
            print('Left branches\n', sess.run(self.left_branches))
            print('Right branches\n', sess.run(self.right_branches))
            print('Prior for branches\n', sess.run(self.l))
            print('Overcounting\n', sess.run(self.v, feed_dict={self.core: feed_data}))
            print('Log Weights\n', sess.run(self.log_weights, feed_dict={self.core: feed_data}))
            #pdb.set_trace()
            print('Log likelihood\n', sess.run(self.log_likelihood, feed_dict={self.core: feed_data}))
            print('Log likelihood tilda\n', sess.run(self.log_likelihood_tilda, feed_dict={self.core: feed_data}))
            Qmatrices.append(sess.run(self.Qmatrix))
            left_branches.append(sess.run(self.left_branches))
            right_branches.append(sess.run(self.left_branches))
            ll.append(sess.run(self.log_likelihood, feed_dict={self.core: feed_data}))
            ll_tilda.append(sess.run(self.log_likelihood_tilda, feed_dict={self.core: feed_data}))
            #pdb.set_trace()
            log_weights.append(sess.run(self.log_weights, feed_dict={self.core: feed_data}))
            jump_chain_evolution.append(sess.run(self.jump_chains, feed_dict={self.core: feed_data}))
            at = datetime.now()
            print('Time spent\n', at-bt, '\n-----------------------------------------')
        print("Done training.")

        # Create local directory and save experiment results
        tm = str(datetime.now())
        local_rlt_root = './results/'
        save_dir = local_rlt_root + (tm[:10]+'-'+tm[11:13]+tm[14:16]+tm[17:19]) + '/'
        if not os.path.exists(save_dir): os.makedirs(save_dir)

        plt.imshow(sess.run(self.Qmatrix))
        plt.title("Trained Q matrix")
        plt.savefig(save_dir + "Qmatrix.png")

        plt.figure(figsize=(10,10))
        plt.plot(elbos)
        plt.ylabel("log $Z_{SMC}$")
        plt.xlabel("Epochs")
        plt.title("Elbo convergence across epochs")
        plt.savefig(save_dir + "ELBO.png")
        plt.show()

        plt.figure(figsize=(10, 10))
        myll = np.asarray(ll)
        plt.plot(myll[:,-1,:],c='black',alpha=0.2)
        plt.plot(np.average(myll[:,-1,:],axis=1),c='yellow')
        plt.ylabel("log likelihood")
        plt.xlabel("Epochs")
        plt.title("Log likelihood convergence across epochs")
        plt.savefig(save_dir + "ll.png")
        plt.show()

        resultDict = {'cost': np.asarray(elbos),
                      'nParticles': self.K,
                      'nTaxa': self.N,
                      'lr': self.learning_rate,
                      'log_weights': np.asarray(log_weights),
                      'Qmatrices': np.asarray(Qmatrices),
                      'left_branches': np.asarray(left_branches),
                      'right_branches': np.asarray(right_branches),
                      'log_lik': np.asarray(ll),
                      'll_tilde': np.asarray(ll_tilda),
                      'jump_chain_evolution': jump_chain_evolution}

        with open(save_dir + 'results.p', 'wb') as f:
            pickle.dump(resultDict, f)

        return

