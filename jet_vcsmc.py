"""
An implementation of Variational Combinatorial Sequential Monte Carlo for jet reconstruction under the Gingko model.
  A variant of Combinatorial Sequential Monte Carlo is used to form a variational objective
  to simultaneously learn the parameters of the proposal and target distribution
  and perform jet reconstruction.
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
     data_NxSxA: a 3 tensor of genomes for the n taxa one hot encoded
    """

    def __init__(self, datadict, K, args=None):
        self.args = args
        self.sample_names = datadict['samples']
        self.data_NxSxA = datadict['data']
        
        # self.lam = tf.exp(tf.Variable(1.0+self.args.branch_prior, dtype = tf.float64))
        #self.lam = tf.Variable(1.5, dtype = tf.float64)
        self.lam = tf.exp(tf.Variable(0+self.args.lambda_prior, dtype=tf.float64, name='lambda_param'))
        self.t_cut = tf.cast(self.findTCut(), dtype = tf.float64)
        
        self.K = K # number of monte carlo samples
        self.M = args.M  # not needed 
        self.N = len(self.data_NxSxA) # number of leaves
        self.S = 2 # len(self.data_NxSxA[0]) # always 1 hack s.t. less rewrite
        self.A = 4 # len(self.data_NxSxA[0, 0]) # number of sites always 5 due to 4 vec + delta
        #### change ####
        # the left and the right exp parameters (invariant masses)
        #self.decay_param = tf.exp(tf.Variable(self.args.decay_prior, dtype=tf.float64, name='decay_param'))
        self.decay_param = tf.exp(tf.Variable(self.args.decay_prior, dtype=tf.float64, name='decay_param'))
        self.decay_dist = tfp.distributions.Exponential(rate=self.decay_param)
        #self.decay_factors_Nm1xK = tf.exp(self.decay_dist.sample([self.N-1,self.K]))
        #self.left_and_right_decay_rates_2xNm1xK = self.left_and_right_decay_rates_2xNm1xK



        # q_r_branch_dist = tfp.distributions.Exponential(rate=right_branches_param_r)
        # q_l_branch_samples = q_l_branch_dist.sample(self.K) 
        # self.left_decay_param = tf.exp(tf.Variable(np.zeros(self.N-1)+self.args.branch_prior, dtype=tf.float64, name='left_decay_param'))
        # self.right_decay_param = tf.exp(tf.Variable(np.zeros(self.N-1)+self.args.branch_prior, dtype=tf.float64, name='right_decay_param'))
        
        
    def findTCut(self):
        tData = self.data_NxSxA[:, :,0] ** 2 - tf.norm(self.data_NxSxA[:, :,1:], axis = -1) ** 2
        res = tf.reduce_max(tData)
        return res
    

    def llh_bc(self, l_data_Kx1x4, r_data_Kx1x4, t_cut, decay_factor_r): #lam, t_cut, decay_factor_r):
        """
        Accepts 4-vec for left and right nodes, 
        (lam here is self.lam but it should be a Kx1 vector of the current sampled decay rate)
        t_cut does not change. 
        """
        #pdb.set_trace()
        # grab left invariant mass
        tL_Kx1 = l_data_Kx1x4[:, :,0] ** 2 - tf.norm(l_data_Kx1x4[:, :,1:], axis = -1) ** 2
        # grab right invariant mass
        tR_Kx1 = r_data_Kx1x4[:, :,0] ** 2 - tf.norm(r_data_Kx1x4[:, :,1:], axis = -1) ** 2

        p_data_Kx1x4 = l_data_Kx1x4 + r_data_Kx1x4 ## eq (5)
        # add left and right 4-vecs form parent 4-vec
        # note that due to the exponential decay in the generative model, the parent
        # invariant mass must be strictly greater than the sum of the left and right child invariant mass

        # here we define the parent mass, which has not been scaled up to account for exponential decay
        tp_Kx1 = p_data_Kx1x4[:, :,0] ** 2 - tf.norm(p_data_Kx1x4[:, :,1:], axis = -1) ** 2
        tp_Kx1 = tf.math.multiply(tp_Kx1, decay_factor_r) #  1.01) #decay_factor_r


        is_negative_Kx1 = tf.logical_or(tf.logical_or(tf.less(tL_Kx1, 0), tf.less(tR_Kx1, 0)), tf.less_equal(tp_Kx1, 0))

        is_invalid_Kx1 = tf.logical_or(
                            tf.less_equal(tp_Kx1, t_cut),
                            tf.logical_or(
                               tf.logical_or(
                                   tf.greater_equal(tL_Kx1, (1 - 1e-3) * tp_Kx1),
                                   tf.greater_equal(tR_Kx1, (1 - 1e-3) * tp_Kx1)
                               ),
                               tf.greater(
                                   tf.sqrt(tL_Kx1) + tf.sqrt(tR_Kx1),
                                   tf.sqrt(tp_Kx1)
                               )
                            )
                         )


        def valid_calc(tp_Xx1, tL_Xx1, tR_Xx1):

            def get_logp(tP_local_Xx1, t_Xx1, t_cut):#, lam):
                """ Here we call the actual PDFs and CDFs defined in Eq (7) of the paper"""

                def prob_is_leaf(tP_local_Xx1, t_Xx1, t_cut):#, lam):
                    """ The CDF defined in Eq (7) of the paper """
                    # Probability of the shower to stop F_s
                    one_minus_cdf = 1 - tf.math.exp(- (1 - 1e-3)*self.lam)
                    prob = - tf.math.log(one_minus_cdf)\
                           + tf.math.log(self.lam) - tf.math.log(tP_local_Xx1) - self.lam * t_Xx1 / tP_local_Xx1
                    return prob

                def prob_is_not_leaf(tP_local_Xx1, t_Xx1, t_cut):#, lam):
                    """ The PDF defined in Eq (7) of the paper """
                    t_upper_Xx1 = tf.minimum(tP_local_Xx1, t_cut) #There are cases where tp2 < t_cut
                    one_minus_cdf = 1 - tf.math.exp(- (1 - 1e-3) * self. lam)
                    prob = -tf.math.log(one_minus_cdf) + \
                            tf.math.log(1 - tf.math.exp(- self.lam * t_upper_Xx1 / tP_local_Xx1))
                    return prob

                results_Xx1 = prob_is_not_leaf(tP_local_Xx1, t_Xx1, t_cut)#, self.lam)

                indices_Yx1 = tf.where(tf.greater(tf.squeeze(t_Xx1), t_cut))

                tP_local_new_Yx1 = tf.gather(tf.squeeze(tP_local_Xx1), indices_Yx1)
                t_new_Yx1 = tf.gather(tf.squeeze(t_Xx1), indices_Yx1)

                updates_Yx1 = prob_is_leaf(tP_local_new_Yx1, t_new_Yx1, t_cut)#, lam)

                results_Xx1 = tf.tensor_scatter_nd_update(results_Xx1, indices_Yx1, updates_Yx1)

                return results_Xx1

            # We always assign the left node as the node with the greater invariant mass for consistency
            # To do this, we find the invariant mass squared for each node as a function of the parent
            # by defining tpLR and tpRL using Eq (6) of the paper
            # this is something akin to the parent invarinat mass in case left is bigger than right and the
            # parent invariant mass in case right is bigger than left
            tpLR_Xx1 = (tf.sqrt(tp_Xx1) - tf.sqrt(tL_Xx1)) ** 2
            tpRL_Xx1 = (tf.sqrt(tp_Xx1) - tf.sqrt(tR_Xx1)) ** 2

            # Calculate the log propobability using the CDFs and PDFs
            # for each of the two cases where the left/right node is ultimately greater
            logpLR_Xx1 = tf.cast(tf.math.log(1/2), dtype = tf.float64) + get_logp(tp_Xx1, tL_Xx1, t_cut) + get_logp(tpLR_Xx1, tR_Xx1, t_cut) 

            logpRL_Xx1 = tf.cast(tf.math.log(1/2), dtype = tf.float64) + get_logp(tp_Xx1, tR_Xx1, t_cut) + get_logp(tpRL_Xx1, tL_Xx1, t_cut) #, self.lam)

            # take the product of the two rightmost terms in Eq (8) where the one_minus_cdf term is distributed
            logp_split_Xx1 = tf.reduce_logsumexp(tf.stack([logpLR_Xx1, logpRL_Xx1]), axis = 0)

            # Add the term for the likelihood for sampling uniformly over a 2-sphere
            logLH_Xx1 = logp_split_Xx1 + tf.math.log(1 / (4 * tf.constant(np.pi, dtype = tf.float64)))

            return tf.cast(logLH_Xx1, dtype = tf.float64)


        results_Kx1 = -tf.float64.max * tf.cast(tf.ones_like(is_negative_Kx1), dtype=tf.float64)

        indices_Xx1 = tf.where(
                        tf.logical_and(
                            tf.logical_not(tf.squeeze(is_negative_Kx1)),
                            tf.logical_not(tf.squeeze(is_invalid_Kx1))
                        )
                      )

        tp_new_Xx1 = tf.gather(tf.squeeze(tp_Kx1), indices_Xx1)
        tL_new_Xx1 = tf.gather(tf.squeeze(tL_Kx1), indices_Xx1)
        tR_new_Xx1 = tf.gather(tf.squeeze(tR_Kx1), indices_Xx1)

        updates_Xx1 = valid_calc(tp_new_Xx1, tL_new_Xx1, tR_new_Xx1)

        results_Kx1 = tf.tensor_scatter_nd_update(results_Kx1, indices_Xx1, updates_Xx1)
        parent_vec4_Kx1x4 = tf.squeeze(l_data_Kx1x4 + r_data_Kx1x4)


        results_Kx1_copied = tf.tile(
            tf.reshape(
                tf.squeeze(results_Kx1), (-1, 1)
            ), 
            (1, 4)
        )

        vec4_Kx4 = tf.squeeze(parent_vec4_Kx1x4)

        like_stacked_vec4_Kx2x4 = tf.stack(
                    [
                        results_Kx1_copied,
                        vec4_Kx4
                    ],
                    axis=1
                )
        return like_stacked_vec4_Kx2x4

    
    def compute_forest_posterior_ginkgo(self, data_KxXxSxA, leafnode_num_record, r):
        """ 
        Forms a log probability measure by dotting the stationary probs with tree likelihood
        And add that to log-prior of tree topology
        NOTE: we add log-prior of branch-lengths in body_update_weights
        """
        #### change ####
        data_sliced = tf.identity(data_KxXxSxA)
        data_sliced = data_sliced[:, : , 0:1, :]
        data_sliced = data_sliced[:, : , :, 0:1]
        data_added = tf.reduce_sum(data_sliced, axis = 1)

        forest_logprior = tf.reduce_sum(-log_double_factorial(2 * tf.maximum(leafnode_num_record, 2) - 3), axis=1)
        return tf.add(tf.squeeze(data_added), forest_logprior)

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
        # l_exponent = tf.multiply(tf.transpose(self.left_branches), tf.expand_dims(self.left_branches_param, axis=0))
        # r_exponent = tf.multiply(tf.transpose(self.right_branches), tf.expand_dims(self.right_branches_param, axis=0))
        # l_multiplier = tf.expand_dims(tf.log(self.left_branches_param), axis=0)
        # r_multiplier = tf.expand_dims(tf.log(self.left_branches_param), axis=0)
        # left_branches_logprior = tf.reduce_sum(l_multiplier - l_exponent, axis=1)
        # right_branches_logprior = tf.reduce_sum(r_multiplier - r_exponent, axis=1)
        log_likelihood_R = tf.gather(log_likelihood, self.N-2) + \
          log_double_factorial(2 * self.N - 3)  #- tf.log(self.decay_param) #- 
          #self.
          # left_branches_logprior - right_branches_logprior
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
        log_normalized_weights = log_weights - tf.reduce_logsumexp(log_weights) # weights are first normalized
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
    
    # main loop of program, runs once per n - 1 coalescent events. Three steps. Resampling
    def body_rank_update(self, log_weights, log_likelihood, log_likelihood_tilde, jump_chains, jump_chain_tensor, 
        core, leafnode_num_record, decay_factors, v_minus, r):
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
        
        # Proposal
        
        # sample new typology
        # Extend partial states ###
        coalesced_indices, remaining_indices, q_log_proposal, jump_chain_tensor = \
        self.extend_partial_state(jump_chain_tensor, r)
        # indexes that have been coalesced, have not been, a norm factor, and the tree typo
        
        
        # sample branch lengths (sampling invariant masses)
        # Branch lengths
        # left_branches_param_r = tf.gather(self.left_branches_param, r)
        # right_branches_param_r = tf.gather(self.right_branches_param, r)
        # q_l_branch_dist = tfp.distributions.Exponential(rate=left_branches_param_r)
        # q_r_branch_dist = tfp.distributions.Exponential(rate=right_branches_param_r)
        # q_l_branch_samples = q_l_branch_dist.sample(self.K) 
        # q_r_branch_samples = q_r_branch_dist.sample(self.K) 
        # left_branches = tf.concat([left_branches, [q_l_branch_samples]], axis=0) 
        # right_branches = tf.concat([right_branches, [q_r_branch_samples]], axis=0) 
        
        
        # book keeping
        # Update partial set data
        
        # pdb.set_trace()
        
        remaining_core = gather_across_core(core, remaining_indices, self.N-r, self.N-r-2, self.A) # Kx(N-r-2)xSxA
        l_coalesced_indices = tf.reshape(tf.gather(tf.transpose(coalesced_indices), 0), (self.K, 1))
        r_coalesced_indices = tf.reshape(tf.gather(tf.transpose(coalesced_indices), 1), (self.K, 1))
        l_data_KxSxA = tf.squeeze(gather_across_core(core, l_coalesced_indices, self.N-r, 1, self.A))
        r_data_KxSxA = tf.squeeze(gather_across_core(core, r_coalesced_indices, self.N-r, 1, self.A))

        
        
        # compute the likelihood of the newly proposed state
        # instaed of passing in self.lam we need to pass in the decay rate, which is obtained
        # by adding the masses of the child nodes that were selected to coalesce

        # define a function that 
        #decay_factor_r = tf.gather(self.decay_factors_Nm1xK, r)
        #decay_factor_r = tf.exp(self.decay_dist.sample([self.K,1]))
        decay_factor_r = tf.exp(self.decay_dist.sample(self.K))


        decay_factors = tf.concat([decay_factors, [decay_factor_r]], axis=0)

        new_mtx_KxSxA = self.llh_bc(l_data_KxSxA[:,1:2,:], r_data_KxSxA[:,1:2,:], self.t_cut, tf.expand_dims(decay_factor_r,axis=1)) #, self.lam, self.t_cut, decay_factor_r)
        
        # new_mtx_KxSxA = self.broadcast_conditional_likelihood_K(l_data_KxSxA, r_data_KxSxA, q_l_branch_samples, q_r_branch_samples)
        
        
        new_mtx_Kx1xSxA = tf.expand_dims(new_mtx_KxSxA, axis=1)
        core = tf.concat([remaining_core, new_mtx_Kx1xSxA], axis=1) # new core size: Kx(N-r-1)xSxA

        reamining_leafnode_num_record = gather_across_2d(leafnode_num_record, remaining_indices, self.N-r, self.N-r-2)
        new_leafnode_num = tf.expand_dims(tf.reduce_sum(gather_across_2d(
            leafnode_num_record, coalesced_indices, self.N-r, 2), axis=1), axis=1)
        
        # remaining are the nodes that have not been combined yet
        leafnode_num_record = tf.concat([reamining_leafnode_num_record, new_leafnode_num], axis=1)
        # there is another latent random variable (the decay rate) which is sampled from the distribution over lambda
        # but not accounted for in our code. Simply addint the left and right invariant masses doesnt yield the parent since
        # the the mass decreases as we move down the tree. 
        # pdb.set_trace()
        log_likelihood_r = self.compute_forest_posterior_ginkgo(core, leafnode_num_record, r)
        
        # book keeping for tf
        #decay_rates_select = tf.gather(decay_factors, tf.range(1,r+2))
        # left_branches_select = tf.gather(left_branches, tf.range(1, r+2)) # (r+1)xK
        # right_branches_select = tf.gather(right_branches, tf.range(1, r+2)) # (r+1)xK
        # left_branches_logprior = tf.reduce_sum(
        #   -left_branches_param_r * left_branches_select + tf.log(left_branches_param_r), axis=0)
        # right_branches_logprior = tf.reduce_sum(
        #    -right_branches_param_r * right_branches_select + tf.log(right_branches_param_r), axis=0)
        # log_likelihood_r = log_likelihood_r + left_branches_logprior + right_branches_logprior

        v_minus = self.overcounting_correct(leafnode_num_record)
        # l_branch = tf.gather(left_branches, r+1)
        # r_branch = tf.gather(right_branches, r+1)
        decay = tf.gather(decay_factors, r+1)
        
        # JET CODE (DOUBLE CHECK)
        log_weights_r = log_likelihood_r - log_likelihood_tilde + tf.log(tf.cast(v_minus, tf.float64)) - q_log_proposal - tf.log(self.lam) - tf.log(self.decay_param) - (self.decay_param * decay)
        # - \
        # (tf.log(left_branches_param_r) - left_branches_param_r * l_branch + tf.log(right_branches_param_r) - \
        # right_branches_param_r * r_branch) + 


        
        ### ORIGINAL FROM OLD CODE BELOW
        #log_weights_r = log_likelihood_r - log_likelihood_tilde - \
        #(tf.log(left_branches_param_r) - left_branches_param_r * l_branch + tf.log(right_branches_param_r) - \
        #right_branches_param_r * r_branch) + tf.log(tf.cast(v_minus, tf.float64)) - q_log_proposal
        ###

        log_weights = tf.concat([log_weights, [log_weights_r]], axis=0)
        log_likelihood = tf.concat([log_likelihood, [log_likelihood_r]], axis=0) # pi(t) = pi(Y|t, b, theta) * pi(t, b|theta) / pi(Y)
        
        r = r + 1

        return log_weights, log_likelihood, log_likelihood_tilde, jump_chains, jump_chain_tensor, \
        core, leafnode_num_record, decay_factors, v_minus, r

    def cond_rank_update(self, log_weights, log_likelihood, log_likelihood_tilde, jump_chains, jump_chain_tensor, 
        core, leafnode_num_record, decay_factors, v_minus, r):
        return r < self.N - 1

    def sample_phylogenies(self):
        """
        Main sampling routine that performs combinatorial SMC by calling the rank update subroutine
        """
        # outer loop that calls body_rank_update. A for loop of N - 1.
        
        N = self.N
        A = self.A
        K = self.K
        
        # KxNx?xA. A = 4. Self.core can store vec4 values, 
        self.core = tf.placeholder(dtype=tf.float64, shape=(K, N, None, A)) # tf.TensorShape([K, None, None, A])
        leafnode_num_record = tf.constant(1, shape=(K, N), dtype=tf.int32) # Keeps track of self.core

        # left_branches = tf.constant(0, shape=(1, K), dtype=tf.float64)
        # right_branches = tf.constant(0, shape=(1, K), dtype=tf.float64)
        decay_factors = tf.constant(0, shape=(1,K), dtype=tf.float64)

        log_weights = tf.constant(0, shape=(1, K), dtype=tf.float64)
        log_likelihood = tf.constant(0, shape=(1, K), dtype=tf.float64)
        log_likelihood_tilde = tf.constant(np.zeros(K) + np.log(1/K), dtype=tf.float64)

        self.jump_chains = tf.constant('', shape=(K, 1))
        self.jump_chain_tensor = tf.constant([self.sample_names] * K, name='JumpChainK')
        v_minus = tf.constant(1, shape=(K, ), dtype=tf.int32)  # to be used in overcounting_correct
        # pdb.set_trace()
        #### change ####
        self.core_with_llh = self.precompute_llh(self.core, self.lam, self.t_cut)
        
        # --- MAIN LOOP ----+
        log_weights, log_likelihood, log_likelihood_tilde, self.jump_chains, self.jump_chain_tensor, \
        core_final, record_final, decay_factors, v_minus, r = tf.while_loop(
            self.cond_rank_update, 
            self.body_rank_update,
            loop_vars=[log_weights, log_likelihood, log_likelihood_tilde, self.jump_chains, self.jump_chain_tensor, 
                       self.core_with_llh, leafnode_num_record, decay_factors, v_minus, tf.constant(0)],
            shape_invariants=[tf.TensorShape([None, K]), tf.TensorShape([None, K]), log_likelihood_tilde.get_shape(),
                              tf.TensorShape([K, None]), tf.TensorShape([K, None]), tf.TensorShape([K, None, None, A]),
                              tf.TensorShape([K, None]), tf.TensorShape([None,K]),
                              v_minus.get_shape(), tf.TensorShape([])])
        # ------------------+

        self.log_weights = tf.gather(log_weights, list(range(1, N))) # remove the trivial index 0
        self.log_likelihood = tf.gather(log_likelihood, list(range(1, N))) # remove the trivial index 0
        # self.left_branches = tf.gather(left_branches, list(range(1, N))) # remove the trivial index 0
        # self.right_branches = tf.gather(right_branches, list(range(1, N))) # remove the trivial index 0
        self.decay_factors = tf.gather(decay_factors, list(range(1,N))) # remove the trivial index 0
        self.elbo = self.compute_log_ZSMC(log_weights) # cost computed eq(5), computed using eq(8)
        self.log_likelihood_R = self.get_log_likelihood(self.log_likelihood)
        self.cost = - self.elbo
        self.log_likelihood_tilde = log_likelihood_tilde
        self.v_minus = v_minus

        return self.elbo
    
    def precompute_llh(self, data, lam, t_cut): #### change ####
        """ This is just the log llh_bc function """
        data = tf.reshape(data, (-1,1, 4)) # data is KxNx1x4

        tp_KNx1 = data[:, :,0] ** 2 - tf.norm(data[:, :,1:], axis = -1) ** 2

        def get_leaf_llh(tP_local_Xx1, t_cut, lam):
            #t_upper_Xx1 = tf.minimum(tP_local_Xx1, t_cut) #There are cases where tp2 < t_cut
            #return tf.cast(-tf.math.log(1 - tf.math.exp(- (1 - 1e-3) * lam)), dtype = tf.float64) + \
            #    tf.cast(tf.math.log(1 - tf.math.exp(- lam * t_upper_Xx1 / tP_local_Xx1)), dtype = tf.float64)
            ones_tensor = tf.ones(shape=(self.K*self.N, 1))
            return tf.cast(ones_tensor, dtype=tf.float64)

        llh_KNx1 = get_leaf_llh(tp_KNx1, lam, t_cut)

        results_KNx1_copied = tf.tile(
            tf.reshape(
                tf.squeeze(llh_KNx1), (-1, 1)
            ), 
            (1, 4)
        )

        like_stacked_vec4_Kx2x4 = tf.stack(
                [
                    results_KNx1_copied,
                    tf.squeeze(data)
                ],
                axis=1
        )
        return tf.reshape(like_stacked_vec4_Kx2x4,(self.K,self.N,self.S,self.A))

    def train(self, epochs=100, batch_size=128, learning_rate=0.001, memory_optimization='on'):
        """
        Run the train op in a TensorFlow session and evaluate variables
        """
        K = self.K
        self.lr = learning_rate
        #pdb.set_trace()
        config = tf.ConfigProto()
        if memory_optimization == 'off':
            from tensorflow.core.protobuf import rewriter_config_pb2
            off = rewriter_config_pb2.RewriterConfig.OFF
            config.graph_options.rewrite_options.memory_optimization = off
            
        data = np.array([self.data_NxSxA] * K, dtype = np.double) # KxNxSxA
        
        # slices = self.batch_slices(data, batch_size)
        
        print('================= Dataset shape: KxNxSxA =================')
        print(data.shape)
        print('==========================================================')
        # pdb.set_trace()
        self.sample_phylogenies()
        print('===================\nFinished constructing computational graph!', '\n===================')
        print( "LEARNING RATE IS:", self.lr)
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
        # Qmatrices = []
        # left_branches = []
        # right_branches = []
        jump_chain_evolution = []
        log_weights = []
        ll = []
        ll_tilde = []
        ll_R = []
        lams = []

        for i in tqdm(range(epochs)):
            bt = datetime.now()



            # for j in tqdm(range(len(slices)-1)):
            #     data_batch = np.take(data, slices[j], axis=2)
            _, cost = sess.run([self.optimizer, self.cost], feed_dict={self.core: data})
                #print('\n Minibatch', j)
                #print(sess.run([self.cost], feed_dict={self.core: data_batch}))

            output = sess.run([self.cost,
                               # self.stationary_probs,
                               self.lam,
                               # self.Qmatrix,
                               # self.left_branches,
                               # self.right_branches,
                               self.log_weights,
                               self.log_likelihood,
                               self.log_likelihood_tilde,
                               self.log_likelihood_R,
                               self.v_minus,
                               # self.left_branches_param,
                               # self.right_branches_param,
                               self.jump_chains,
                               self.decay_param,
                               self.decay_factors],
                               feed_dict={self.core: data})
            cost = output[0]
            # stats = output[1]
            lam_ = output[1]
            # Qs = output[2]
            # lb = output[3]
            # rb = output[4]
            log_Ws = output[2]
            log_liks = output[3]
            log_lik_tilde = output[4]
            log_lik_R = output[5]
            overcount = output[6]
            # lb_param = output[10]
            # rb_param = output[11]
            jc = output[7]
            print('Epoch', i+1)
            print('ELBO\n', round(-cost, 3))
            # print('Stationary probabilities\n', stats)
            print('Lambda Parameter', lam_)
            # print('Q-matrix\n', Qs)
            # print('Left branches\n', lb)
            # print('Right branches\n', rb)
            print('Log Weights\n', np.round(log_Ws,3))
            print('Log likelihood\n', np.round(log_liks,3))
            print('Log likelihood tilde\n', np.round(log_lik_tilde,3))
            # print('LB param:\n', lb_param)
            # print('RB param:\n', rb_param)
            print('Log likelihood at R\n', np.round(log_lik_R,3))
            #print('Overcount ', np.round(overcount,3))
            print('lambda param ', np.round(lam_,3))
            print(f'decay_param: {np.round(output[8],3)}')
            print(f"Decay factors: {np.round(output[9],3)}")
            # print('Jump chains')
            # for i in range(len(jc)):
            #     print(jc[i])
            #     break
            elbos.append(-cost)
            # Qmatrices.append(Qs)
            # left_branches.append(lb)
            # right_branches.append(rb)
            lams.append(lam_)
            ll.append(log_liks)
            ll_tilde.append(log_lik_tilde)
            ll_R.append(log_lik_R)
            log_weights.append(log_Ws)
            jump_chain_evolution.append(jc)
            at = datetime.now()
            print('Time spent\n', at-bt, '\n-----------------------------------------')
        print("Done training.")
        

        # plt.imshow(sess.run(self.Qmatrix))
        # plt.title("Trained Q matrix")
        # plt.savefig(save_dir + "Qmatrix.png")

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
        print("estimator llh:\n", np.average(best_log_lik))
        best_jump_chain = jump_chain_evolution[np.argmax(elbos)]

        resultDict = {'cost': np.asarray(elbos),
                      'nParticles': self.K,
                      'nTaxa': self.N,
                      'lr': self.lr,
                      'log_weights': np.asarray(log_weights),
                      'lambda' : np.asarray(lams),
                      # 'Qmatrices': np.asarray(Qmatrices),
                      # 'left_branches': left_branches,
                      # 'right_branches': right_branches,
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
