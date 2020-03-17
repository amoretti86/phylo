"""
Implements the Sequential Importance Sampling algorithm for the Kingman Coalescent
from Cappello and Palacios along with a visualization of the inferred geneaology.
Leo Zhang and Antonio Moretti
"""

import operator as op
from functools import reduce
import random
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from copy import deepcopy
from gusfield import *
import numpy as np
from gusfield import Phylogeny
import pdb

class SIS:

    def __init__(self, M):

        #self.data = data
        #self.phylogeny = phylogeny
        #self.M = phylo.M
        self.M = M

    def ncr(self, n, r):
        # Compute combinatorial term n choose r
        r = min(r, n-r)
        numer = reduce(op.mul, range(n, n-r, -1), 1)
        denom = reduce(op.mul, range(1, r+1), 1)
        return numer/denom

    def get_leafnodes(self, root_node, leafnodes=[]):
        # Return leaf nodes through recursion
        if root_node.subnodes == []:
            leafnodes.append(root_node)
            return leafnodes
        else:
            for n in root_node.subnodes:
                leafnodes = self.get_leafnodes(n, leafnodes)
        return leafnodes

    def merge_singletons(self, dicts, node_list, i):
        """
        Merges singletons into parent nodes.
        dicts is a list of all c_i dictionaries
        dicts[i] is an object, so editing dict_new changes dicts[i]
        deepcopy is a workaround.
        """
        dict_new = deepcopy(dicts[i])
        cannot_delete = []
        for key in dicts[i]:
            if len(dicts[i][key]) == 1 and node_list[key].data != 'root':
                parent = node_list[key].parent
                parent_idx = node_list.index(parent)
                cannot_delete.append(parent_idx)
                # check to see if parent of the key is not in dictionary (c_i)
                if parent_idx not in dict_new:
                    dict_new[parent_idx] = dicts[i][key]
                else:
                    # add particle to the existing list of particles
                    dict_new[parent_idx].append(dicts[i][key][0])
                if key not in cannot_delete:
                    # delete if dict_new[key] does not correspond to an intermediate node
                    del dict_new[key]
                else:
                    # remove string denoting the singleton to be removed
                    dict_new[key].remove(dicts[i][key][0])
        dicts[i] = dict_new
        return dicts

    def no_singletons(self, dicts, node_list, i):
        """
        Check whether tree has any singletons at all
        Step 1 and 4c of Palacios require this
        """
        result = True
        for key in dicts[i]:
            if len(dicts[i][key]) == 1 and node_list[key].data!= 'root':
                result = False
        return result

    def create_node_sampler(self, dicts, node_list, i):
        """
        This defines the sample space of nodes at which coalescent events can happen.
        The sample space is defined in the list variable 'result'
        REVISIT ME: node_list is not used
        """
        result = []
        for key in dicts[i]:
            if len(dicts[i][key]) > 1:
                for p in range(len(dicts[i][key])):
                    result.append(key)
        return result

    def build_graph(self, G, master_node):
        if master_node.subnodes == []:
            G.add_node(master_node)
            return G

        G.add_node(master_node)

        for n in master_node.subnodes:
            G = self.build_graph(G, n)

        return G

    def main_sampling(self, graph, showing=True):
    #def main_sampling(self, showing=True):
        """
        Operates on a phylogeny object (graph) and M (incidence matrix)
        """
        # graph = self.phylogeny.main_phylogeny()
        root_node = graph.node_dict['root'] # Do this K times for each geneaology
        node_list = list(graph.get_nodes()) # Broadcast node_list
        leafnodes = self.get_leafnodes(root_node)

        n = len(self.M) # number of particles
        # create a list of n empty dictionaries (to represent c_n) of the n coalescent events
        dicts = [{} for i in range(n)] # c_n to c_1

        # Initialize c_n
        # Iterate through all nodes in the phylogeny
        for i in range(len(node_list)):
            if node_list[i] in leafnodes:
                # If node is a leafnode
                # node_list[i].data is a string of particles, e.g.: 's3,s4,s5'
                # this splits the string into a list of strings representing particles
                # e.g.: ['s3', 's4', 's5']
                # the zero here in dicts[0] corresponds to c_n or the nth state of the coalescent
                dicts[0][i] = node_list[i].data.split(',')

        # Execution of merge before the for loop
        while self.no_singletons(dicts, node_list, 0) == False:
            dicts = self.merge_singletons(dicts, node_list, 0)

        # variable to store probability of geneaology
        q = 1

        data_added = []
        nodes_added = []

        # Iterate over the n-1 coalescent events
        for i in range(n-1):
            # Set c[n-1] = c[n], copying our dictionary representing the particle set
            # for all nodes at iteration n
            dicts[i+1] = deepcopy(dicts[i])
            # Choose node and particles to coalesce.
            # Define sample space of q defining nodes at which coalescent events can happen
            node_sampler = self.create_node_sampler(dicts, node_list, i)
            # Sample uniformly from the space of possible nodes
            # [v_0, v_0, v_3, v_3, v_3, v_3] or [0,0,3,3,3,3] in palacios example 1
            node_idx = random.choice(node_sampler)
            # compute probability of selected choice (2/6 or 4/6 in the above)
            q1 = node_sampler.count(node_idx)/len(node_sampler)
            # enumerate the set of particles to sample from the given node
            particle_sampler = [i for i in range(len(dicts[i][node_idx]))]
            # sample two particles to coalesce
            particle_idx = random.sample(particle_sampler, 2)
            # compute combinatorial term
            q2 = 1/self.ncr(len(dicts[i][node_idx]),2)
            # dicts[i] is the c_i key:value pair representing the particle set for given node i

            # Grab the index and corresponding first and second particle to coalesce to store
            # in variables (these are two strings)
            particle1 = dicts[i][node_idx][particle_idx[0]]
            particle2 = dicts[i][node_idx][particle_idx[1]]
            particle_coalesced = particle1 + '+' + particle2

            # Coalesce (4c). Remove these strings from the dict of particles
            # and add string for coalescent event
            dicts[i+1][node_idx].remove(particle1)
            dicts[i+1][node_idx].remove(particle2)
            dicts[i+1][node_idx].append(particle_coalesced)

            # Merge singletons after coalescence (4d)
            while self.no_singletons(dicts, node_list, i+1) == False:
                dicts = self.merge_singletons(dicts, node_list, i+1)

            # Update probability of geneaology (4e)
            q = q*q1*q2

            # Build tree
            n3 = Node(particle_coalesced)
            if particle1 not in data_added:
                n1 = Node(particle1)
                data_added.append(particle1)
                nodes_added.append(n1)
            else:
                n1 = nodes_added[data_added.index(particle1)]
            n1.parent = n3
            n3.subnodes.append(n1)
            if particle2 not in data_added:
                n2 = Node(particle2)
                data_added.append(particle2)
                nodes_added.append(n2)
            else:
                n2 = nodes_added[data_added.index(particle2)]
            n2.parent = n3
            n3.subnodes.append(n2)
            data_added.append(particle_coalesced)
            nodes_added.append(n3)

        # Revisit!
        G = self.build_graph(Graph(), nodes_added[-1])
        #pdb.set_trace()
        #G = self.phylogeny.build_graph(graph, nodes_added[-1])
        if showing:
            G.draw()

        return dicts, q, G

if __name__ == "__main__":
    #print("Akwaaba!")
    """
    N = 10
    M1 = np.array([[1, 1, 0, 0, 0], [0, 0, 1, 0, 0], [1, 1, 0, 0, 1], [0, 0, 1, 1, 0], [0, 1, 0, 0, 0]])
    for i in range(N):
        graph = SIS(M1)
    """
    M1 = np.array([[1, 1, 0, 0, 0], [0, 0, 1, 0, 0], [1, 1, 0, 0, 1], [0, 0, 1, 1, 0], [0, 1, 0, 0, 0]])
    print("M1: \n", M1)
    phylo = Phylogeny(M1)
    print(phylo.M)
    print(phylo.K)
    graph = phylo.main_phylogeny()
    sampler = SIS(M1)

    #print("graph: ", graph)
    mydicts, myq, myG = sampler.main_sampling(graph)
    print(mydicts)
    print(myq)
    print(myG)
