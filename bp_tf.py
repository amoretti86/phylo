"""
Symbolic implementation of sum-product or belief propagation algorithm for computing the likelihood of a phylogenetic tree
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
#import pdb

tf.reset_default_graph()

class Node:

    def __init__(self, id=None, data=None):
        self.id = id
        self.data = data
        self.left = None
        self.right = None
        self.left_branch = None
        self.right_branch = None


class BeliefProp:

    def __init__(self, Q, root, datadict):
        """
        BeliefProp object initializes the Q matrix as a tf variable as well as the distribution over branch lengths
        Root node is passed in as input which implicitly specifies the topology of the graph
        """
        self.Q = tf.Variable(Q, dtype=tf.float64, shape=(3,3), name="Q_matrix")
        self.K = Q.shape[0]
        self.root = root
        self.lognorm = tfp.distributions.LogNormal(loc=1, scale=0.5, validate_args=False, allow_nan_stats=True, name='LogNorm')
        self.prior = tf.Variable(np.expand_dims(np.ones(self.K), axis=0)/self.K, dtype=tf.float64, name='prior')
        self.learning_rate = 0.001
        self.n = len(datadict['taxa'])
        self.taxa = datadict['taxa']
        self.genome_NxSxA = datadict['genome']
        self.s = len(self.genome_NxSxA[0])
        self.internal_nodes = self.get_internal_nodes()
        self.leaf_nodes = self.get_leaf_nodes()

    def get_internal_nodes(self):
        """
        DFS to check whether nodes have children adding them to a stack otherwise forming a list of internal node objects
        """
        q = []
        q.append(self.root)
        internal_nodes = []
        while (len(q)):
            curr = q[0]
            q.pop(0)
            isInternal = 0
            if (curr.left):
                isInternal = 1
                q.append(curr.left)
            if (curr.right):
                isInternal = 1
                q.append(curr.right)
            if (isInternal):
                internal_nodes.append(curr)
                #curr.data = tf.Variable(np.expand_dims(np.ones(self.K),axis=0)/self.K, dtype=tf.float64, name=curr.id)
        # Make sure that node ordering is such that any child is placed before its parent
        return internal_nodes[::-1]

    def get_leaf_nodes(self):
        q = []
        q.append(self.root)
        leaf_nodes = []
        while (len(q)):
            curr = q[0]
            q.pop(0)
            is_leaf = 0
            if not (curr.left):
                is_leaf = 1
            else:
                q.append(curr.left)
            if not (curr.right):
                is_leaf = 1
            else:
                q.append(curr.right)
            if (is_leaf):
                leaf_nodes.append(curr)
        return leaf_nodes

    def conditional_likelihood(self, node):
        """
        Compute the conditional likelihood at a given node by passing messages up from left and right children
        """
        # As of 6/4, this version only works with data of one site, i.e. self.s = 1
        # In the future, we might only need this function (and not naive... node... ?)
        left  = node.left.data
        right = node.right.data
        left_p_matrix  = tf.linalg.expm(self.Q * node.left_branch)
        right_p_matrix = tf.linalg.expm(self.Q * node.right_branch)
        left_lik  = tf.matmul(left, left_p_matrix)
        right_lik = tf.matmul(right, right_p_matrix)
        lik = left_lik * right_lik
        return lik

    def pass_messages(self):
        """
        Iterate over node objects in internal_nodes and compute conditional likelihood
        """
        for node in self.internal_nodes:
            node.data = self.conditional_likelihood(node)

    def create_likelihood_loss(self):
        """
        Form the loss function by passing messages from leaf nodes to root and defined cost as negative tree likelihood
        """
        self.pass_messages()
        self.tree_likelihood = tf.matmul(self.root.data, self.prior, transpose_b=True)
        self.cost = - tf.math.log(self.tree_likelihood)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def train(self):
        """
        Run the train op in a TensorFlow session and evaluate variables
        """
        #pdb.set_trace()
        self.create_likelihood_loss()
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            node = self.internal_nodes[0]


            for i in range(100):
                # As of 6/4, we arbitrarily assign data to leaf_nodes
                # i.e. we do not know whether leaf_nodes[0] is A, B, or,..., F
                # Should be easy to modify
                _, c = sess.run([self.optimizer, self.cost],
                                feed_dict={self.leaf_nodes[0].data:
                                               np.expand_dims(np.array([0., 1., 0.]),axis=0),
                                           self.leaf_nodes[1].data:
                                               np.expand_dims(np.array([0., 0., 1.]),axis=0),
                                           self.leaf_nodes[2].data:
                                               np.expand_dims(np.array([0., 0., 1.]),axis=0),
                                           self.leaf_nodes[3].data:
                                               np.expand_dims(np.array([1., 0., 0.]),axis=0),
                                           self.leaf_nodes[4].data:
                                               np.expand_dims(np.array([1., 0., 0.]),axis=0),
                                           self.leaf_nodes[5].data:
                                               np.expand_dims(np.array([0., 1., 0.]),axis=0)})
            
            print(sess.run(self.Q))


        return


if __name__ == '__main__':

    #genome_NxSxA = np.array([[0.,1.,0.], [0.,0.,1.], [0.,0.,1.],
    #                         [1.,0.,0.], [1.,0.,0.], [0.,1.,0.]])
    genome_NxSxA = np.array([[[0.,1.,0.]], [[0.,0.,1.]], [[0.,0.,1.]],
                             [[1.,0.,0.]], [[1.,0.,0.]], [[0.,1.,0.]]])

    # Tile genome to repeat across 10 sites
    #genome_NxSxA = np.transpose(np.tile(genome_NxSxA, (10,1,1)), (1,0,2))
    taxa = ['F', 'D', 'E', 'C', 'A', 'B']
    data_dict = {'taxa': taxa, 'genome': genome_NxSxA}
    Qmatrix = np.array([[-2.,1.,1.], [1.,-2.,1.], [1.,1.,-2.]])


    root = Node('5')
    root.left = Node('4')
    root.right = Node('F', tf.placeholder(dtype=tf.float64, shape=(1, 3), name='node_F_data'))
    root.left.left = Node('3')
    root.left.right = Node('2')
    root.left.right.left = Node('D', tf.placeholder(dtype=tf.float64, shape=(1, 3), name='node_D_data'))
    root.left.right.right = Node('E', tf.placeholder(dtype=tf.float64, shape=(1, 3), name='node_E_data'))
    root.left.left.left = Node('1')
    root.left.left.right = Node('C', tf.placeholder(dtype=tf.float64, shape=(1, 3), name='node_C_data'))
    root.left.left.left.left = Node('A', tf.placeholder(dtype=tf.float64, shape=(1, 3), name='node_A_data'))
    root.left.left.left.right = Node('B', tf.placeholder(dtype=tf.float64, shape=(1, 3), name='node_B_data'))
    

    root.left_branch = tf.Variable(0.5,dtype=tf.float64)
    root.right_branch = tf.Variable(2.5,dtype=tf.float64)
    root.left.left_branch = tf.Variable(1.0,dtype=tf.float64)
    root.left.right_branch = tf.Variable(2.0,dtype=tf.float64)
    root.left.right.left_branch = tf.Variable(0.5,dtype=tf.float64)
    root.left.right.right_branch = tf.Variable(0.5,dtype=tf.float64)
    root.left.left.left_branch = tf.Variable(0.5,dtype=tf.float64)
    root.left.left.right_branch = tf.Variable(1.5,dtype=tf.float64)
    root.left.left.left.left_branch = tf.Variable(1.0,dtype=tf.float64)
    root.left.left.left.right_branch = tf.Variable(1.0,dtype=tf.float64)

    model = BeliefProp(Qmatrix, root, data_dict)
    model.train()
