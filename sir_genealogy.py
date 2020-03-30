"""
Sequential Importance Resampling for the Kingman Coalescent
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

class SIR:

    def __init__(self, M):

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

    def merge_singletons_KxN(self, dicts, node_list, i, j):
        """
        Merges singletons into parent nodes.
        dicts is a list of all c_i dictionaries
        dicts[i] is an object, so editing dict_new changes dicts[i]
        deepcopy is a workaround.
        """
        #pdb.set_trace()
        dict_new = deepcopy(dicts[j,i])
        cannot_delete = []
        for key in dicts[j,i]:
            if len(dicts[j,i][key]) == 1 and node_list[key].data != 'root':
                parent = node_list[key].parent
                parent_idx = node_list.index(parent)
                cannot_delete.append(parent_idx)
                # check to see if parent of the key is not in dictionary (c_i)
                if parent_idx not in dict_new:
                    dict_new[parent_idx] = dicts[j,i][key]
                else:
                    # add particle to the existing list of particles
                    dict_new[parent_idx].append(dicts[j,i][key][0])
                if key not in cannot_delete:
                    # delete if dict_new[key] does not correspond to an intermediate node
                    del dict_new[key]
                else:
                    # remove string denoting the singleton to be removed
                    dict_new[key].remove(dicts[j, i][key][0])
        dicts[j,i] = dict_new
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
        #pdb.set_trace()
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

    def main_sampling(self, graph, K, showing=True):
    #def main_sampling(self, showing=True):
        """
        Operates on a phylogeny object (graph) and M (incidence matrix)
        """
        #pdb.set_trace()
        root_node = graph.node_dict['root'] # Do this K times for each geneaology
        node_list = list(graph.get_nodes()) # Broadcast node_list
        leafnodes = self.get_leafnodes(root_node)
        # list of gusfield.Node objects. leafnode[0].data returns the species at the leaf of the tree

        n = len(self.M) # number of particles
        # create a list of n empty dictionaries (to represent c_n) of the n coalescent events
        dicts = [{} for i in range(n)] # c_n to c_1
        # Perhaps this should be a numpy array\

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
        #pdb.set_trace()
        dicts_KxN = np.array([dicts] * K)
        q = 1
        qs = np.zeros([K,n-1])

        K_data_added = [[] for i in range(K)]
        K_nodes_added = [[] for i in range(K)]
        #data_added = []
        #nodes_added = []

        # Iterate over the n-1 coalescent events
        for i in range(n-1):

            # for all nodes at iteration n
            q_selection_probs = []
            particle_samplers = []
            particle_idxes = []
            q_coalescent_probs = []
            # Iterate over particles
            for j in range(K):

                #pdb.set_trace()
                # Set c[n-1] = c[n], copying our dictionary representing the particle set
                dicts_KxN[j,i+1] = deepcopy(dicts_KxN[j,i])
                node_sampler = self.create_node_sampler(dicts_KxN[j], node_list, i)
                node_idx = random.choice(node_sampler)
                q1 = node_sampler.count(node_idx)/len(node_sampler)

                q_selection_probs.append(q1)
                # enumerate the set of particles to sample from the given node
                k_particle_sampler = [i for i in range(len(dicts_KxN[j, i][node_idx]))]
                particle_samplers.append(k_particle_sampler)
                # sample two particles to coalesce
                k_particle_idx = random.sample(k_particle_sampler, 2)
                # compute combinatorial term
                q2 = 1/self.ncr(len(dicts_KxN[j,i][node_idx]),2)
                q_coalescent_probs.append(q2)


                # Grab the index and corresponding first and second particle to coalesce to store
                # in variables (these are two strings)
                particle1 = dicts_KxN[j,i][node_idx][k_particle_idx[0]]
                particle2 = dicts_KxN[j,i][node_idx][k_particle_idx[1]]
                particle_coalesced = particle1 + '+' + particle2


                # Coalesce (4c). Remove these strings from the dict of particles
                # and add string for coalescent event
                #print("dicts_KxN[j,i + 1][node_indices[j]]\n", dicts_KxN[j,i + 1][node_idx])
                #print("particle to be removed:", particle1)
                dicts_KxN[j,i + 1][node_idx].remove(particle1)
                dicts_KxN[j,i + 1][node_idx].remove(particle2)
                dicts_KxN[j,i + 1][node_idx].append(particle_coalesced)


                # Merge singletons after coalescence (4d)
                while self.no_singletons(dicts_KxN[j], node_list, i + 1) == False:
                    dicts_KxN = self.merge_singletons_KxN(dicts_KxN, node_list, i + 1, j)

                # Update probability of geneaology (4e)
                #q = q * q1 * q2
                q = q1*q2
                qs[j,i] = q


                # Build tree
                #pdb.set_trace()
                n3 = Node(particle_coalesced)
                if particle1 not in K_data_added[j]:
                    n1 = Node(particle1)
                    K_data_added[j].append(particle1)
                    K_nodes_added[j].append(n1)
                else:
                    n1 = K_nodes_added[j][K_data_added[j].index(particle1)]
                n1.parent = n3
                n3.subnodes.append(n1)
                if particle2 not in K_data_added[j]:
                    n2 = Node(particle2)
                    K_data_added[j].append(particle2)
                    K_nodes_added[j].append(n2)
                else:
                    n2 = K_nodes_added[j][K_data_added[j].index(particle2)]
                n2.parent = n3
                n3.subnodes.append(n2)
                K_data_added[j].append(particle_coalesced)
                K_nodes_added[j].append(n3)
                #print(K_data_added)
            #K_data_added.append(data_added)
            #K_nodes_added.append(nodes_added)

        probs = np.exp(np.einsum('ij->i', np.log((qs))))
        self.K_nodes_added = K_nodes_added

        # Revisit!
        #pdb.set_trace()
        if showing:
            for k in range(K):
                G = self.build_graph(Graph(), K_nodes_added[k][-1])
        #if showing:
                G.draw(probs[k])



        return dicts_KxN, qs, G, probs


    def plot_genealogy(self, graph, i):
        G = self.build_graph(Graph(), self.K_nodes_added[i][-1])
        G.draw()
        return

if __name__ == "__main__":

    print("Akwaaba! Running some tests...")

    M_5 = np.array([[1, 1, 0, 0, 0], [0, 0, 1, 0, 0], [1, 1, 0, 0, 1], [0, 0, 1, 1, 0], [0, 1, 0, 0, 0]])

    M_10 = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                     [1, 0, 0, 1, 1, 1, 1, 1, 1, 0],
                     [0, 1, 0, 1, 1, 1, 1, 0, 0, 0],
                     [0, 1, 1, 1, 1, 0, 0, 1, 0, 0],
                     [0, 1, 0, 1, 1, 1, 0, 1, 0, 0],
                     [1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                     [1, 0, 1, 1, 0, 0, 1, 0, 1, 0],
                     [0, 0, 0, 0, 1, 0, 1, 1, 1, 1],
                     [1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
                     [0, 0, 1, 1, 1, 0, 1, 1, 0, 0]])

    M_20 = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                     [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
                     [1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
                     [1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0],
                     [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0],
                     [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1],
                     [1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0],
                     [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0],
                     [0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1],
                     [0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                     [1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1],
                     [0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0],
                     [1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                     [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1],
                     [1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0],
                     [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1]])


    #print("M1: \n", M1)
    phylo = Phylogeny(M_5)
    print(phylo.M)
    print(phylo.K)
    graph = phylo.main_phylogeny()
    sampler = SIR(M_5)

    #print("graph: ", graph)
    mydicts, myq, myG, myprobs = sampler.main_sampling(graph, K = 3)
    print(mydicts)
    print(myq)
    print(myG)
    print(myprobs)

    #sampler.plot_genealogy(Graph(),0)
