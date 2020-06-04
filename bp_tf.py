import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

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
        self.Q = tf.Variable(Q, dtype=tf.float64, shape=(4,4), name="Q_matrix", constraint=self.constrain_Q)
        self.a = len(Q)
        self.n = len(datadict['taxa'])
        self.root = root
        self.lognorm = tfp.distributions.LogNormal(loc=1, scale=0.5, validate_args=False, allow_nan_stats=True, name='LogNorm')
        # self.state_probs = tf.Variable(np.expand_dims(np.ones(self.a), axis=0)/self.a, dtype=tf.float64, name='state_probs')
        self.state_probs = np.zeros((1,self.a)) + 1/self.a
        self.taxa = datadict['taxa']
        self.genome_NxSxA = datadict['genome']
        self.learning_rate = 0.01
        self.internal_nodes = self.get_internal_nodes()
        self.leaf_nodes = self.get_leaf_nodes()

    def constrain_Q(self, Q):
        output = []
        for i in range(self.a):
            output.append([])
            for j in range(self.a):
                output[i].append(0)
        for i in range(self.a):
            for j in range(self.a):
                if i != j:
                    output[i][j] = tf.clip_by_value(Q[i][j], 0, 1e4)
        for i in range(self.a):
            for j in range(self.a):
                if i == j:
                    output[i][j] = tf.clip_by_value(Q[i][j], -sum(output[i]),-sum(output[i]))
        return output

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
        self.tree_likelihood = tf.matmul(self.root.data, self.state_probs, transpose_b=True)
        self.cost = - tf.math.reduce_sum(tf.math.log(self.tree_likelihood))
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def get_feed_dict(self):
        self.feed_dict = {}
        for i in range(len(self.leaf_nodes)):
            idx = self.taxa.index(self.leaf_nodes[i].id)
            self.feed_dict[self.leaf_nodes[i].data] = self.genome_NxSxA[idx]

    def train(self):
        """
        Run the train op in a TensorFlow session and evaluate variables
        """
        self.create_likelihood_loss()
        self.get_feed_dict()
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        costs = []
        for i in range(1000):
            _, cost = sess.run([self.optimizer, self.cost], feed_dict=self.feed_dict)
            costs.append(cost)
        
        print(sess.run(self.Q))
        print(sess.run(tf.linalg.expm(self.Q)))
        print(costs)
        #plt.plot(costs)
        #plt.show()

        return


if __name__ == '__main__':

    Alphabet_dir = {'A': [1., 0., 0., 0.],
                    'C': [0., 1., 0., 0.],
                    'T': [0., 0., 1., 0.],
                    'G': [0., 0., 0., 1.]}
    alphabet_dir = {'a': [1., 0., 0., 0.],
                    'c': [0., 1., 0., 0.],
                    't': [0., 0., 1., 0.],
                    'g': [0., 0., 0., 1.]}

    def strings_to_datadict(genome_strings, alphabet_dir):
        genomes_NxSxA = np.zeros([len(genome_strings),len(genome_strings[0]),len(alphabet_dir)])
        for i in range(genomes_NxSxA.shape[0]):
            for j in range(genomes_NxSxA.shape[1]):
                genomes_NxSxA[i,j] = alphabet_dir[genome_strings[i][j]]
        taxa = ['S' + str(i) for i in range(genomes_NxSxA.shape[0])]
        datadict = {'taxa': taxa,
                    'genome': genomes_NxSxA}
        return datadict

    def strings_to_data(genome_strings, alphabet_dir):
        genomes_NxSxA = np.zeros([len(genome_strings),len(genome_strings[0]),len(alphabet_dir)])
        for i in range(genomes_NxSxA.shape[0]):
            for j in range(genomes_NxSxA.shape[1]):
                genomes_NxSxA[i,j] = alphabet_dir[genome_strings[i][j]]
        return genomes_NxSxA       

    # Tile genome to repeat across 10 sites
    # genome_NxSxA = np.array([[[0.,1.,0.]], [[0.,0.,1.]], [[0.,0.,1.]],
    #                          [[1.,0.,0.]], [[1.,0.,0.]], [[0.,1.,0.]]])
    # genome_NxSxA = np.transpose(np.tile(genome_NxSxA, (10,1,1)), (1,0,2))
    genome_strings = ['AAAAAATTCCGG','AAAAAATTCCGC','AAAAAATTCCCC','AAATTTTTCGGG','AAATTTTTCGCC','CCCCCCAACGGC']
    #genome_strings = ['AA','AC','CC','CT','TT','GG']
    genome_NxSxA = strings_to_data(genome_strings, Alphabet_dir)
    taxa = ['A','B','C','D','E','F']
    data_dict = {'taxa': taxa, 'genome': genome_NxSxA}
    Qmatrix = np.array([[-3.,1.,1.,1.], [1.,-3.,1.,1.], [1.,1.,-3.,1.], [1.,1.,1.,-3.]])
    a = genome_NxSxA.shape[2]
    s = genome_NxSxA.shape[1]

    # Build a tree manually
    root = Node('5')
    root.left = Node('4')
    root.right = Node('F', tf.placeholder(dtype=tf.float64, shape=(s,a), name='node_F_data'))
    root.left.left = Node('3')
    root.left.right = Node('2')
    root.left.right.left = Node('D', tf.placeholder(dtype=tf.float64, shape=(s,a), name='node_D_data'))
    root.left.right.right = Node('E', tf.placeholder(dtype=tf.float64, shape=(s,a), name='node_E_data'))
    root.left.left.left = Node('1')
    root.left.left.right = Node('C', tf.placeholder(dtype=tf.float64, shape=(s,a), name='node_C_data'))
    root.left.left.left.left = Node('A', tf.placeholder(dtype=tf.float64, shape=(s,a), name='node_A_data'))
    root.left.left.left.right = Node('B', tf.placeholder(dtype=tf.float64, shape=(s,a), name='node_B_data'))

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

