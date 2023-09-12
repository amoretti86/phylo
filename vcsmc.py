"""
An implementation of Twisted Variational Combinatorial Sequential Monte Carlo for Bayesian Phylogenetic Inference.
  A variant of Combinatorial Sequential Monte Carlo is used to form a variational objective
  to simultaneously learn the parameters of the proposal and target distribution
  and perform Bayesian phylogenetic inference.
"""

import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import pdb
import random
from datetime import datetime
import pickle
from tqdm import tqdm

# @staticmethod
def ncr(n, r):
    # Compute combinatorial term n choose r
    numer = tf.reduce_prod(tf.range(n-r+1, n+1))
    denom = tf.reduce_prod(tf.range(1, r+1))
    return numer / denom

# @staticmethod
def _double_factorial_loop_body(n, result, two):
    result = tf.where(tf.greater_equal(n, two), result + tf.math.log(n), result)
    return n - two, result, two

# @staticmethod
def _double_factorial_loop_condition(n, result, two):
    del result  # Unused
    return tf.cast(tf.math.count_nonzero(tf.greater_equal(n, two)), tf.bool)

# @staticmethod
def log_double_factorial(n):
    """Computes the double factorial of `n`.
      Note:
        In the following, A1 to An are optional batch dimensions.
      Args:
        n: A tensor of shape `[A1, ..., An]` containing positive integer values.
      Returns:
        A tensor of shape `[A1, ..., An]` containing the double factorial of `n`.
      """
    n = tf.cast(tf.convert_to_tensor(value=n), tf.float64)

    two = tf.ones_like(n) * 2
    result = tf.math.log(tf.ones_like(n))
    _, result, _ = tf.while_loop(
        cond=_double_factorial_loop_condition,
        body=_double_factorial_loop_body,
        loop_vars=[n, result, two])
    return result

# @staticmethod
def gather_across_2d(a, idx, a_shape_1=None, idx_shape_1=None):
    '''
    Gathers as such:
    if a is K-by-N, idx is K-by-M, then it returns a Tensor with structure like
    [tf.gather(a[k], idx[k]) for k in range(K)].
    But it broadcasts and doesn't actually use for-loop.
    '''
    if a_shape_1 is None:
        a_shape_1 = a.shape[1]
    if idx_shape_1 is None:
        idx_shape_1 = idx.shape[1]

    K = a.shape[0]
    a_reshaped = tf.reshape(a, [K * a_shape_1, -1])
    add_to_idx = a_shape_1 * tf.transpose(tf.tile(tf.expand_dims(tf.range(K), axis=0), [idx_shape_1,1]))
    a_gathered = tf.gather(a_reshaped, idx + add_to_idx)
    a_gathered = tf.reshape(a_gathered, [K, -1])
    return a_gathered

# @staticmethod
def gather_across_core(a, idx, a_shape_1=None, idx_shape_1=None, A=4):
    '''
    Gathers from the core as such:
    if a is K-by-N-by-S-by-A, idx is K-by-M, then it returns a Tensor with structure like
    [tf.gather(a[k], idx[k]) for k in range(K)].
    But it broadcasts and doesn't actually use for-loop.
    '''
    if a_shape_1 is None:
        a_shape_1 = a.shape[1]
    if idx_shape_1 is None:
        idx_shape_1 = idx.shape[1]

    K = a.shape[0]
    a_reshaped = tf.reshape(a, [K * a_shape_1, -1, A])
    add_to_idx = a_shape_1 * tf.transpose(tf.tile(tf.expand_dims(tf.range(K), axis=0), [idx_shape_1,1]))
    a_gathered = tf.gather(a_reshaped, idx + add_to_idx)
    a_gathered = tf.reshape(a_gathered, [K, idx_shape_1, -1, A])
    return a_gathered





class VCSMC:
    """
    VCSMC takes as input a dictionary (datadict) with two keys:
     taxa: a list of n strings denoting taxa
     genome_NxSxA: a 3 tensor of genomes for the n taxa one hot encoded
    """

    def __init__(self, datadict, K, args=None):
        self.args = args
        self.taxa = datadict['taxa']
        self.genome_NxSxA = datadict['genome']
        self.K = K
        self.M = args.M
        self.N = len(self.genome_NxSxA)
        self.S = len(self.genome_NxSxA[0])
        self.A = len(self.genome_NxSxA[0, 0])
        self.left_branches_param = tf.exp(tf.Variable(np.zeros(self.N-1)+self.args.branch_prior, dtype=tf.float64, name='left_branches_param'))
        self.right_branches_param = tf.exp(tf.Variable(np.zeros(self.N-1)+self.args.branch_prior, dtype=tf.float64, name='right_branches_param'))
        if not args.jcmodel:
            self.y_q = tf.linalg.set_diag(tf.Variable(np.zeros((self.A, self.A)) + 1/self.A, dtype=tf.float64, name='Qmatrix'), [0]*self.A)
            self.Qmatrix = self.get_Q()
            self.y_station = tf.Variable(np.zeros(self.A) + 1 / self.A, dtype=tf.float64, name='Stationary_probs')
        else:
            self.Qmatrix = tf.linalg.set_diag(
                tf.constant(np.zeros((self.A, self.A)) + 1/self.A, dtype=tf.float64, name='Qmatrix'),
                [-(self.A-1)/self.A] * self.A
            )
            self.y_station = tf.constant(np.zeros(self.A) + 1 / self.A, dtype=tf.float64, name='Stationary_probs')
        self.stationary_probs = self.get_stationary_probs()

    def get_stationary_probs(self):
        """ Compute stationary probabilities of the Q matrix """
        denom = tf.reduce_sum(tf.exp(self.y_station))
        return tf.expand_dims(tf.exp(self.y_station) / denom, axis=0)

    def get_Q(self):
        """
        Forms the transition matrix of the continuous time Markov Chain, constraints
        are satisfied by defining off-diagonal terms using the softmax function
        """
        denom = tf.reduce_sum(tf.linalg.set_diag(tf.exp(self.y_q), [0]*self.A), axis=1)
        denom = tf.stack([denom]*self.A, axis=1)
        q_entry = tf.multiply(tf.linalg.set_diag(tf.exp(self.y_q), [0]*self.A), 1/denom)
        hyphens = tf.reduce_sum(q_entry, axis=1)
        Q = tf.linalg.set_diag(q_entry, -hyphens)
        return Q

    def conditional_likelihood(self, l_data, r_data, l_branch, r_branch):
        """
        Computes conditional complete likelihood at an ancestor node
        by passing messages from left and right children
        """
        #with tf.device('/gpu:0'): 
        left_Pmatrix = tf.linalg.expm(self.Qmatrix * l_branch)
        right_Pmatrix = tf.linalg.expm(self.Qmatrix * r_branch)
        left_prob = tf.matmul(l_data, left_Pmatrix)
        right_prob = tf.matmul(r_data, right_Pmatrix)
        likelihood = tf.multiply(left_prob, right_prob)
        return likelihood
        
    def broadcast_conditional_likelihood_M(self, l_data_SxA, r_data_SxA, l_branch_samples_M, r_branch_samples_M):
        """
        Broadcast conditional complete likelihood computation at ancestor node
        by passing messages from left and right children.
        Messages passed and Pmatrices are now 3-tensors to broadcast across subparticle x alphabet x alphabet (MxAxA)
        """
        left_message_MxAxA   = tf.tensordot( l_branch_samples_M, self.Qmatrix, axes=0)
        right_message_MxAxA  = tf.tensordot( r_branch_samples_M, self.Qmatrix, axes=0)
        left_Pmat_MxAxA      = tf.linalg.expm(left_message_MxAxA)
        right_Pmat_MxAxA     = tf.linalg.expm(right_message_MxAxA)
        left_prob_MxAxS   = tf.matmul(left_Pmat_MxAxA, l_data_SxA, transpose_b=True)  # Confirm dim(l_data): SxA
        right_prob_MxAxS  = tf.matmul(right_Pmat_MxAxA, r_data_SxA, transpose_b=True)
        left_prob_AxSxM = tf.transpose(left_prob_MxAxS, perm=[1,2,0])
        right_prob_AxSxM = tf.transpose(right_prob_MxAxS, perm=[1,2,0])
        likelihood_AxSxM = left_prob_AxSxM * right_prob_AxSxM
        return likelihood_AxSxM

    def broadcast_conditional_likelihood_K(self, l_data_KxSxA, r_data_KxSxA, l_branch_samples_K, r_branch_samples_K):
        left_message_KxAxA   = tf.tensordot( l_branch_samples_K, self.Qmatrix, axes=0)
        right_message_KxAxA  = tf.tensordot( r_branch_samples_K, self.Qmatrix, axes=0)
        left_Pmat_KxAxA      = tf.linalg.expm(left_message_KxAxA)
        right_Pmat_KxAxA     = tf.linalg.expm(right_message_KxAxA)
        left_prob_KxSxA   = tf.matmul(l_data_KxSxA, left_Pmat_KxAxA)
        right_prob_KxSxA   = tf.matmul(r_data_KxSxA, right_Pmat_KxAxA)
        likelihood_KxSxA = left_prob_KxSxA * right_prob_KxSxA
        return likelihood_KxSxA

    def compute_tree_posterior(self, data, leafnode_num):
        """
        Forms a log probability measure by dotting the stationary probs with tree likelihood
        And add that to log-prior of tree topology
        NOTE: we add log-prior of branch-lengths in body_update_weights
        """
        #with tf.device('/gpu:1'): 
        tree_likelihood = tf.matmul(self.stationary_probs, data, transpose_b=True)
        data_loglik = tf.reduce_sum(tf.log(tree_likelihood))
        tree_logprior = -log_double_factorial(2 * tf.maximum(leafnode_num, 2) - 3)

        return data_loglik + tree_logprior
        
    def broadcast_compute_tree_posterior_M(self, likelihood_AxSxM, leafnode_num):
        """
        Forms a log probability measure by dotting the stationary probs with tree likelihood
        And add that to log-prior of tree topology
        NOTE: we add log-prior of branch-lengths in body_update_weights
        """
        #with tf.device('/gpu:1'): 
        tree_likelihood_SxM = tf.einsum('ia,asm->sm',self.stationary_probs, likelihood_AxSxM)
        tree_likelihood_S = tf.reduce_mean(tree_likelihood_SxM, axis=1)
        data_loglik = tf.reduce_sum(tf.log(tree_likelihood_S))
        tree_logprior = -log_double_factorial(2 * tf.maximum(leafnode_num, 2) - 3)

        return data_loglik + tree_logprior

    def broadcast_compute_tree_posterior_K(self, data_KxSxA, leafnode_num_record):
        """
        Forms a log probability measure by dotting the stationary probs with tree likelihood
        And add that to log-prior of tree topology
        NOTE: we add log-prior of branch-lengths in body_update_weights
        """
        #with tf.device('/gpu:1'): 
        stationary_probs = tf.tile(tf.expand_dims(tf.transpose(self.stationary_probs), axis=0), [self.K, 1, 1])
        tree_lik = tf.squeeze(tf.matmul(data_KxSxA, stationary_probs))
        tree_loglik = tf.reduce_sum(tf.log(tree_lik), axis=1)
        tree_logprior = tf.reduce_sum(-log_double_factorial(2 * tf.maximum(leafnode_num_record, 2) - 3), axis=1)

        return tree_loglik + tree_logprior

    def compute_forest_posterior(self, data_KxXxSxA, leafnode_num_record, r):
        """
        Forms a log probability measure by dotting the stationary probs with tree likelihood
        And add that to log-prior of tree topology
        NOTE: we add log-prior of branch-lengths in body_update_weights
        """
        #with tf.device('/gpu:1'): 
        data_reshaped = tf.reshape(data_KxXxSxA, (self.K*(self.N-r-1), -1, self.A))
        stationary_probs = tf.tile(tf.expand_dims(tf.transpose(self.stationary_probs), axis=0), [self.K*(self.N-r-1), 1, 1])
        forest_lik = tf.squeeze(tf.matmul(data_reshaped, stationary_probs))
        forest_lik = tf.reshape(forest_lik, (self.K, self.N-r-1, -1))
        forest_loglik = tf.reduce_sum(tf.log(forest_lik), axis=(1,2))
        forest_logprior = tf.reduce_sum(-log_double_factorial(2 * tf.maximum(leafnode_num_record, 2) - 3), axis=1)

        return forest_loglik + forest_logprior

    def overcounting_correct(self, leafnode_num_record):
        """
        Computes overcounting correction term to the proposal distribution
        """
        v_minus = tf.reduce_sum(leafnode_num_record - tf.cast(tf.equal(leafnode_num_record, 1), tf.int32), axis=1)
        return v_minus

    def get_log_likelihood(self, log_likelihood):
        """
        Computes last rank-event's log_likelihood P(Y|t, theta) by removing prior from
        the already computed log_likelihood, which includes prior
        """
        l_exponent = tf.multiply(tf.transpose(self.left_branches), tf.expand_dims(self.left_branches_param, axis=0))
        r_exponent = tf.multiply(tf.transpose(self.right_branches), tf.expand_dims(self.right_branches_param, axis=0))
        l_multiplier = tf.expand_dims(tf.log(self.left_branches_param), axis=0)
        r_multiplier = tf.expand_dims(tf.log(self.left_branches_param), axis=0)
        left_branches_logprior = tf.reduce_sum(l_multiplier - l_exponent, axis=1)
        right_branches_logprior = tf.reduce_sum(r_multiplier - r_exponent, axis=1)
        log_likelihood_R = tf.gather(log_likelihood, self.N-2) + \
          log_double_factorial(2 * self.N - 3) - \
          left_branches_logprior - right_branches_logprior
        return log_likelihood_R          

    def compute_log_ZSMC(self, log_weights):
        """
        Forms the estimator log_ZSMC, a multi sample lower bound to the likelihood
        Z_SMC is formed by averaging over weights and multiplying over coalescent events
        """
        #with tf.device('/gpu:1'): 
        log_Z_SMC = tf.reduce_sum(tf.reduce_logsumexp(log_weights - tf.log(tf.cast(self.K, tf.float64)), axis=1))
        return log_Z_SMC

    def resample(self, core, leafnode_num_record, JC_K, log_weights):
        """
        Resample partial states by drawing from a categorical distribution whose parameters are normalized importance weights
        JumpChain (JC_K) is a tensor formed from a numpy array of lists of strings, returns a resampled JumpChain tensor
        """
        log_normalized_weights = log_weights - tf.reduce_logsumexp(log_weights)
        indices = tf.squeeze(tf.random.categorical([log_normalized_weights], self.K))
        resampled_core = tf.gather(core, indices)
        resampled_record = tf.gather(leafnode_num_record, indices)
        resampled_JC_K = tf.gather(JC_K, indices)
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
        top_values, coalesced_indices = tf.nn.top_k(z, 2)
        bottom_values, remaining_indices = tf.nn.top_k(tf.negative(z), self.N - r - 2)
        JC_keep = tf.gather(tf.reshape(JCK, [self.K * (self.N - r)]), remaining_indices)
        particles = tf.gather(tf.reshape(JCK, [self.K * (self.N - r)]), coalesced_indices)
        particle1 = particles[:, 0]
        particle2 = particles[:, 1]
        # Form new state
        particle_coalesced = particle1 + '+' + particle2
        # Form new Jump Chain
        JCK = tf.concat([JC_keep, tf.expand_dims(particle_coalesced, axis=1)], axis=1)

        #return particle1, particle2, particle_coalesced, coalesced_indices, remaining_indices, q, JCK
        return coalesced_indices, remaining_indices, q, JCK

    def cond_true_resample(self, log_likelihood_tilde, core, leafnode_num_record, 
        log_weights, log_likelihood, jump_chains, jump_chain_tensor, r):
        core, leafnode_num_record, jump_chain_tensor, indices = self.resample(
            core, leafnode_num_record, jump_chain_tensor, tf.gather(log_weights, r))
        log_likelihood_tilde = tf.gather_nd(
            tf.gather(tf.transpose(log_likelihood), indices),[[k, r] for k in range(self.K)])
        jump_chains = tf.concat([jump_chains, jump_chain_tensor], axis=1)
        return log_likelihood_tilde, core, leafnode_num_record, jump_chains, jump_chain_tensor

    def cond_false_resample(self, log_likelihood_tilde, core, leafnode_num_record, 
        log_weights, log_likelihood, jump_chains, jump_chain_tensor, r):
        jump_chains = tf.concat([jump_chains, jump_chain_tensor], axis=1)
        return log_likelihood_tilde, core, leafnode_num_record, jump_chains, jump_chain_tensor

    def body_rank_update(self, log_weights, log_likelihood, log_likelihood_tilde, jump_chains, jump_chain_tensor, 
        core, leafnode_num_record, left_branches, right_branches, v_minus, r):
        """
        Define tensors for log_weights, log_likelihood, jump_chain_tensor and core (state data for distribution over characters for ancestral taxa)
        by iterating over rank events.
        """

        # Resample
        log_likelihood_tilde, core, leafnode_num_record, jump_chains, jump_chain_tensor = tf.cond(r > 0,
            lambda: self.cond_true_resample(log_likelihood_tilde, core, leafnode_num_record, 
                log_weights, log_likelihood, jump_chains, jump_chain_tensor, r),
            lambda: self.cond_false_resample(log_likelihood_tilde, core, leafnode_num_record, 
                log_weights, log_likelihood, jump_chains, jump_chain_tensor, r))

        # Extend partial states
        coalesced_indices, remaining_indices, q_log_proposal, jump_chain_tensor = \
        self.extend_partial_state(jump_chain_tensor, r)
        
        # Branch lengths
        left_branches_param_r = tf.gather(self.left_branches_param, r)
        right_branches_param_r = tf.gather(self.right_branches_param, r)
        q_l_branch_dist = tfp.distributions.Exponential(rate=left_branches_param_r)
        q_r_branch_dist = tfp.distributions.Exponential(rate=right_branches_param_r)
        q_l_branch_samples = q_l_branch_dist.sample(self.K) 
        q_r_branch_samples = q_r_branch_dist.sample(self.K) 
        left_branches = tf.concat([left_branches, [q_l_branch_samples]], axis=0) 
        right_branches = tf.concat([right_branches, [q_r_branch_samples]], axis=0) 

        # Update partial set data
        remaining_core = gather_across_core(core, remaining_indices, self.N-r, self.N-r-2, self.A) # Kx(N-r-2)xSxA
        l_coalesced_indices = tf.reshape(tf.gather(tf.transpose(coalesced_indices), 0), (self.K, 1))
        r_coalesced_indices = tf.reshape(tf.gather(tf.transpose(coalesced_indices), 1), (self.K, 1))
        l_data_KxSxA = tf.squeeze(gather_across_core(core, l_coalesced_indices, self.N-r, 1, self.A))
        r_data_KxSxA = tf.squeeze(gather_across_core(core, r_coalesced_indices, self.N-r, 1, self.A))
        new_mtx_KxSxA = self.broadcast_conditional_likelihood_K(l_data_KxSxA, r_data_KxSxA, q_l_branch_samples, q_r_branch_samples)
        new_mtx_Kx1xSxA = tf.expand_dims(new_mtx_KxSxA, axis=1)
        core = tf.concat([remaining_core, new_mtx_Kx1xSxA], axis=1) # Kx(N-r-1)xSxA

        reamining_leafnode_num_record = gather_across_2d(leafnode_num_record, remaining_indices, self.N-r, self.N-r-2)
        new_leafnode_num = tf.expand_dims(tf.reduce_sum(gather_across_2d(
            leafnode_num_record, coalesced_indices, self.N-r, 2), axis=1), axis=1)
        leafnode_num_record = tf.concat([reamining_leafnode_num_record, new_leafnode_num], axis=1)

        # Comptue weights
        log_likelihood_r = self.compute_forest_posterior(core, leafnode_num_record, r)

        left_branches_select = tf.gather(left_branches, tf.range(1, r+2)) # (r+1)xK
        right_branches_select = tf.gather(right_branches, tf.range(1, r+2)) # (r+1)xK
        left_branches_logprior = tf.reduce_sum(
            -left_branches_param_r * left_branches_select + tf.log(left_branches_param_r), axis=0)
        right_branches_logprior = tf.reduce_sum(
            -right_branches_param_r * right_branches_select + tf.log(right_branches_param_r), axis=0)
        log_likelihood_r = log_likelihood_r + left_branches_logprior + right_branches_logprior

        v_minus = self.overcounting_correct(leafnode_num_record)
        l_branch = tf.gather(left_branches, r+1)
        r_branch = tf.gather(right_branches, r+1)
        
        log_weights_r = log_likelihood_r - log_likelihood_tilde - \
        (tf.log(left_branches_param_r) - left_branches_param_r * l_branch + tf.log(right_branches_param_r) - \
        right_branches_param_r * r_branch) + tf.log(tf.cast(v_minus, tf.float64)) - q_log_proposal

        log_weights = tf.concat([log_weights, [log_weights_r]], axis=0)
        log_likelihood = tf.concat([log_likelihood, [log_likelihood_r]], axis=0) # pi(t) = pi(Y|t, b, theta) * pi(t, b|theta) / pi(Y)
        
        r = r + 1

        return log_weights, log_likelihood, log_likelihood_tilde, jump_chains, jump_chain_tensor, \
        core, leafnode_num_record, left_branches, right_branches, v_minus, r

    def cond_rank_update(self, log_weights, log_likelihood, log_likelihood_tilde, jump_chains, jump_chain_tensor, 
        core, leafnode_num_record, left_branches, right_branches, v_minus, r):
        return r < self.N - 1

    def sample_phylogenies(self):
        """
        Main sampling routine that performs combinatorial SMC by calling the rank update subroutine
        """
        N = self.N
        A = self.A
        K = self.K

        self.core = tf.placeholder(dtype=tf.float64, shape=(K, N, None, A))
        leafnode_num_record = tf.constant(1, shape=(K, N), dtype=tf.int32) # Keeps track of self.core

        left_branches = tf.constant(0, shape=(1, K), dtype=tf.float64)
        right_branches = tf.constant(0, shape=(1, K), dtype=tf.float64)

        log_weights = tf.constant(0, shape=(1, K), dtype=tf.float64)
        log_likelihood = tf.constant(0, shape=(1, K), dtype=tf.float64)
        log_likelihood_tilde = tf.constant(np.zeros(K) + np.log(1/K), dtype=tf.float64)

        self.jump_chains = tf.constant('', shape=(K, 1))
        self.jump_chain_tensor = tf.constant([self.taxa] * K, name='JumpChainK')
        v_minus = tf.constant(1, shape=(K, ), dtype=tf.int32)  # to be used in overcounting_correct

        # --- MAIN LOOP ----+
        log_weights, log_likelihood, log_likelihood_tilde, self.jump_chains, self.jump_chain_tensor, \
        core_final, record_final, left_branches, right_branches, v_minus, r = tf.while_loop(
            self.cond_rank_update, 
            self.body_rank_update,
            loop_vars=[log_weights, log_likelihood, log_likelihood_tilde, self.jump_chains, self.jump_chain_tensor, 
                       self.core, leafnode_num_record, left_branches, right_branches, v_minus, tf.constant(0)],
            shape_invariants=[tf.TensorShape([None, K]), tf.TensorShape([None, K]), log_likelihood_tilde.get_shape(),
                              tf.TensorShape([K, None]), tf.TensorShape([K, None]), tf.TensorShape([K, None, None, A]),
                              tf.TensorShape([K, None]), tf.TensorShape([None, K]), tf.TensorShape([None, K]), 
                              v_minus.get_shape(), tf.TensorShape([])])
        # ------------------+

        self.log_weights = tf.gather(log_weights, list(range(1, N))) # remove the trivial index 0
        self.log_likelihood = tf.gather(log_likelihood, list(range(1, N))) # remove the trivial index 0
        self.left_branches = tf.gather(left_branches, list(range(1, N))) # remove the trivial index 0
        self.right_branches = tf.gather(right_branches, list(range(1, N))) # remove the trivial index 0
        self.elbo = self.compute_log_ZSMC(log_weights)
        self.log_likelihood_R = self.get_log_likelihood(self.log_likelihood)
        self.cost = - self.elbo
        self.log_likelihood_tilde = log_likelihood_tilde
        self.v_minus = v_minus

        return self.elbo

    def batch_slices(self, data, batch_size):
        sites = data.shape[2]
        sites_list = list(range(sites))
        num_batches = sites // batch_size
        slices = []
        for i in range(num_batches):
            sampled_indices = random.sample(sites_list, batch_size)
            slices.append(sampled_indices)
            sites_list = list(set(sites_list) - set(sampled_indices))
        if len(sites_list) != 0:
            slices.append(sites_list)
        return slices

    def train(self, epochs=100, batch_size=128, learning_rate=0.001, memory_optimization='on'):
        """
        Run the train op in a TensorFlow session and evaluate variables
        """
        K = self.K
        self.lr = learning_rate

        config = tf.ConfigProto()
        if memory_optimization == 'off':
            from tensorflow.core.protobuf import rewriter_config_pb2
            off = rewriter_config_pb2.RewriterConfig.OFF
            config.graph_options.rewrite_options.memory_optimization = off

        data = np.array([self.genome_NxSxA] * K, dtype=np.double) # KxNxSxA
        slices = self.batch_slices(data, batch_size)
        print('================= Dataset shape: KxNxSxA =================')
        print(data.shape)
        print('==========================================================')

        self.sample_phylogenies()
        print('===================\nFinished constructing computational graph!', '\n===================')

        if self.args.optimizer == 'Adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.cost)
        else:
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.cost)

        sess = tf.Session(config=config)
        init = tf.global_variables_initializer()
        sess.run(init)
        initial_list = sess.run([-self.cost, self.jump_chains], feed_dict={self.core: data})
        print('===================\nInitial evaluation of ELBO:', round(initial_list[0], 3))
        print('Initial jump chain:')
        print(initial_list[1][0])
        print('===================')
        print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name))
        
        # Create local directory and save experiment results
        tm = str(datetime.now())
        local_rlt_root = './results/' + str(self.args.dataset) + '/' + str(self.args.nested) + \
          '/' + str(self.args.n_particles) + '/'
        save_dir = local_rlt_root + (tm[:10]+'-'+tm[11:13]+tm[14:16]+tm[17:19]) + '/'
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        rp = open(save_dir + "run_parameters.txt", "w")
        rp.write('Initial evaluation of ELBO : ' + str(initial_list[0]))
        rp.write('\n')
        for k,v in vars(self.args).items():
            rp.write(str(k) + ' : ' + str(v))
            rp.write('\n')
        rp.write(str(self.optimizer))
        rp.close()
        
        print('Training begins --')
        elbos = []
        Qmatrices = []
        left_branches = []
        right_branches = []
        jump_chain_evolution = []
        log_weights = []
        ll = []
        ll_tilde = []
        ll_R = []

        for i in tqdm(range(epochs)):
            bt = datetime.now()
            
            for j in tqdm(range(len(slices)-1)):
                data_batch = np.take(data, slices[j], axis=2)
                _, cost = sess.run([self.optimizer, self.cost], feed_dict={self.core: data_batch})
                #print('\n Minibatch', j)
                #print(sess.run([self.cost], feed_dict={self.core: data_batch}))

            output = sess.run([self.cost,
                               self.stationary_probs,
                               self.Qmatrix,
                               self.left_branches,
                               self.right_branches,
                               self.log_weights,
                               self.log_likelihood,
                               self.log_likelihood_tilde,
                               self.log_likelihood_R,
                               self.v_minus,
                               self.left_branches_param,
                               self.right_branches_param,
                               self.jump_chains],
                               feed_dict={self.core: data})
            cost = output[0]
            stats = output[1]
            Qs = output[2]
            lb = output[3]
            rb = output[4]
            log_Ws = output[5]
            log_liks = output[6]
            log_lik_tilde = output[7]
            log_lik_R = output[8]
            overcount = output[9]
            lb_param = output[10]
            rb_param = output[11]
            jc = output[12]
            print('Epoch', i+1)
            print('ELBO\n', round(-cost, 3))
            print('Stationary probabilities\n', stats)
            print('Q-matrix\n', Qs)
            # print('Left branches\n', lb)
            # print('Right branches\n', rb)
            # print('Log Weights\n', np.round(log_Ws,3))
            # print('Log likelihood\n', np.round(log_liks,3))
            # print('Log likelihood tilde\n', np.round(log_lik_tilde,3))
            print('LB param:\n', lb_param)
            print('RB param:\n', rb_param)
            # print('Log likelihood at R\n', np.round(log_lik_R,3))
            # print('Jump chains')
            # for i in range(len(jc)):
            #     print(jc[i])
            #     break
            elbos.append(-cost)
            Qmatrices.append(Qs)
            left_branches.append(lb)
            right_branches.append(rb)
            ll.append(log_liks)
            ll_tilde.append(log_lik_tilde)
            ll_R.append(log_lik_R)
            log_weights.append(log_Ws)
            jump_chain_evolution.append(jc)
            at = datetime.now()
            print('Time spent\n', at-bt, '\n-----------------------------------------')
        print("Done training.")
        

        plt.imshow(sess.run(self.Qmatrix))
        plt.title("Trained Q matrix")
        plt.savefig(save_dir + "Qmatrix.png")

        plt.figure(figsize=(10,10))
        plt.plot(elbos)
        plt.ylabel("log $Z_{SMC}$")
        plt.xlabel("Epochs")
        plt.title("Elbo convergence across epochs")
        plt.savefig(save_dir + "ELBO.png")
        #plt.show()

        plt.figure(figsize=(10, 10))
        myll = np.asarray(ll_R)
        plt.plot(myll[:,:],c='black',alpha=0.2)
        plt.plot(np.average(myll[:,:],axis=1),c='yellow')
        plt.ylabel("log likelihood")
        plt.xlabel("Epochs")
        plt.title("Log likelihood convergence across epochs")
        plt.savefig(save_dir + "ll.png")
        #plt.show()

        # Save best log-likelihood value and jump chain
        best_log_lik = np.asarray(ll_R)[np.argmax(elbos)]#.shape
        print("Best log likelihood values:\n", best_log_lik)
        best_jump_chain = jump_chain_evolution[np.argmax(elbos)]

        resultDict = {'cost': np.asarray(elbos),
                      'nParticles': self.K,
                      'nTaxa': self.N,
                      'lr': self.lr,
                      'log_weights': np.asarray(log_weights),
                      'Qmatrices': np.asarray(Qmatrices),
                      'left_branches': left_branches,
                      'right_branches': right_branches,
                      'log_lik': np.asarray(ll),
                      'll_tilde': np.asarray(ll_tilde),
                      'log_lik_R': np.asarray(ll_R),
                      'jump_chain_evolution': jump_chain_evolution,
                      'best_epoch' : np.argmax(elbos),
                      'best_log_lik': best_log_lik,
                      'best_jump_chain': best_jump_chain}



        with open(save_dir + 'results.p', 'wb') as f:
            #pdb.set_trace()
            pickle.dump(resultDict, f)

        print("Finished...")
        sess.close()
