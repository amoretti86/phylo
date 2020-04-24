"""
Implementation of Felsenstein's Pruning Algorithm (Belief Propagation)
Compute Marginal Likelihood of Phylogenetic Tree
"""

import numpy as np
import scipy.linalg as spl
from scipy.linalg import expm
from collections import deque


class Node:

    def __init__(self, id=None, data=None):
        self.id = id
        self.data = data
        self.left = None
        self.right = None
        self.left_branch = None
        self.right_branch = None


class BeliefPropagation:

    def __init__(self, Qmatrix, root):
        self.Qmatrix = Qmatrix
        self.root = root
        self.K = Qmatrix.shape[0]
        self.internal_nodes = self.get_internal_nodes()

    def get_internal_nodes(self):
        # Grab all internal nodes (ancestral or latent variables)
        q = []
        q.append(self.root)
        nodes = []

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
                nodes.append(curr)
                # print(curr.id, end = " ")

        return nodes[::-1]

    def pass_messages(self):
        # Pass messages from leaf nodes to root
        for node in self.internal_nodes:
            node.data = self.conditional_likelihood(node.left, node.right, node.left_branch, node.right_branch)

    def conditional_likelihood(self, left, right, left_branch, right_branch):
        # Computes the conditional likelihood using the formula above
        likelihood = np.zeros(self.K)
        # Matrix exponentiation here is probably inefficient
        left_Pmatrix = spl.expm(self.Qmatrix * left_branch)
        right_Pmatrix = spl.expm(self.Qmatrix * right_branch)
        for i in range(self.K):
            left_prob = np.dot(left_Pmatrix[i], left.data)
            right_prob = np.dot(right_Pmatrix[i], right.data)
            likelihood[i] = left_prob * right_prob
        return likelihood

    def compute_tree_likelihood(self, prior):
        tree_likelihood = np.dot(prior, root.data)
        self.tree_likelihood = tree_likelihood


if __name__ == "__main__":
    """
    Run test case to reproduce example from
    https://lukejharmon.github.io/pcm/chapter8_fitdiscrete/
    """

    # Specify topology and data at leaf nodes
    root = Node('5')
    root.left = Node('4')
    root.right = Node('F', np.array([0., 1., 0.]))
    root.left.left = Node('3')
    root.left.right = Node('2')
    root.left.right.left = Node('D', np.array([0., 0., 1.]))
    root.left.right.right = Node('E', np.array([0., 0., 1.]))
    root.left.left.left = Node('1')
    root.left.left.right = Node('C', np.array([1., 0., 0.]))
    root.left.left.left.left = Node('A', np.array([1., 0., 0.]))
    root.left.left.left.right = Node('B', np.array([0., 1., 0.]))

    # Specify branch lengths
    root.left_branch = 0.5
    root.right_branch = 2.5
    root.left.left_branch = 1.0
    root.left.right_branch = 2.0
    root.left.right.left_branch = 0.5
    root.left.right.right_branch = 0.5
    root.left.left.left_branch = 0.5
    root.left.left.right_branch = 1.5
    root.left.left.left.left_branch = 1.0
    root.left.left.left.right_branch = 1.0

    # Likelihood is computed using the Mk model
    Qmatrix = np.array([[-2,1,1],[1,-2,1],[1,1,-2]])
    model = BeliefPropagation(Qmatrix, root)
    model.pass_messages()
    for node in model.internal_nodes:
        print("Node ID: ", node.id, "\n Conditional Likelihoods: \n", node.data)

    # Print tree likelihood
    model.compute_tree_likelihood(np.array([.33, .33, .33]))
    print("Tree likelihood:\n", model.tree_likelihood)
