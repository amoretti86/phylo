"""
Implements Gusfield's algorithm for the perfect phylogeny problem
along with a visualization of the multfurcating tree topology.
Leo Zhang and Antonio Moretti
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import random

print("Imported!")

class Edge:

    def __init__(self, data):
        # Subedges must be a list of edges (or empty list)
        self.data = data
        self.subedges = []

    def print_subedges(self):
        # print subedges
        if self.subedges == []:
            s = 'No subedges'
        else:
            s = 'Subedges: '
            for e in self.subedges:
                s += e.data + ' , '
            s = s[:-2]
        print(s)


class Node:

    def __init__(self, data):
        # Subedges must be a list of edges (or empty list)
        self.data = data
        self.subnodes = []
        self.parent = None
        self.edge_before = ''

    def del_subnodes(self, d):
        for n in self.subnodes:
            if n.data == d:
                self.subnodes.remove(n)

    def print_node(self):
        s = 'Node ' + self.data + ' with subnodes: '
        for n in self.subnodes:
            s += n.data + ' '
        s += '; edge before is ' + self.edge_before
        print(s)


class Graph:

    def __init__(self):
        self.node_dict = {}
        self.num_nodes = 0

    def add_node(self, node):
        self.num_nodes += 1
        self.node_dict[node.data] = node
        return node

    def get_node(self, data):
        if data in self.node_dict:
            return self.node_dict[data]
        else:
            return None

    def del_node(self, data):
        try:
            node_to_del = self.get_node(data)
            for d in self.node_dict:
                node = self.get_node(d)
                node.del_subnodes(data)
            del self.node_dict[data]
            self.num_nodes -= 1
            return None
        except KeyError:
            raise Exception("Node %s does not exist" % data)

    def contains(self, data):
        return data in self.node_dict

    def get_nodes(self):
        return list(self.node_dict.values())

    def get_nodes_data(self):
        return list(self.node_dict.keys())

    # For visualizations
    def build_nx_graph(self):
        G_nx = nx.DiGraph()
        for key in self.get_nodes_data():
            G_nx.add_node(key)
        for node in self.get_nodes():
            for subnode in node.subnodes:
                G_nx.add_edge(node.data, subnode.data)
        return G_nx

    def draw(self, prob):
        G_nx = self.build_nx_graph()
        plt.figure(figsize=(10,10))
        pos = nx.kamada_kawai_layout(G_nx)
        nx.draw_networkx(G_nx, pos=pos, with_labels=True, fontsize=4,width=3.8,node_color='r',edge_color='brown')
        plt.title("Sampled Geneaology", fontsize=14)
        plt.xlabel("Prob %1.6f " % prob)
        plt.show()

    def __iter__(self):
        return iter(self.node_dict.values())


class Phylogeny:

    def __init__(self, imatrix):
        #self.m = imatrix
        self.M, self.indices = self.radix_sort(imatrix)
        self.K = self.build_k()

    def radix_sort(self, m):
        a = []
        for i in range(len(m.T)):
            a.append('')
            for j in range(len(m)):
                a[i] += str(m.T[i][j])
        indices = np.flip(np.argsort(a))
        M = np.array([m.T[indices[i]] for i in range(len(indices))])

        # Remove duplicates
        m_tmp, indices_kept = np.unique(M, return_index=True, axis=0)
        indices_removed = list(set(indices) - set(indices_kept))
        M = np.delete(M, indices_removed, axis=0)
        indices = np.delete(indices, indices_removed)
        #self.m = M.T
        #self.indices = indices
        return M.T, indices

    # static function?
    def build_k(self):#, M, indices):
        K = np.zeros(self.M.shape)
        for i in range(len(K)):
            pointer = 0
            for j in range(len(K.T)):
                if self.M[i][j] == 1:
                    K[i][pointer] = self.indices[j] + 1
                    pointer += 1
            if pointer < len(K.T):
                K[i][pointer] = -1
        return K

    def build_phylogeny(self):
        # Revisit me!
        master = Edge(None)
        dict_of_edges = {}
        added_data = set([])
        for i in range(len(self.K)):
            for j in range(len(self.K.T)):
                if self.K[i][j] > 0 and self.K[i][j] not in added_data:
                    added_data.add(self.K[i][j])
                    dict_of_edges[self.K[i][j]] = Edge('c' + str(int(self.K[i][j])))
                    if j == 0:
                        master.subedges.append(dict_of_edges[self.K[i][j]])
                    else:
                        dict_of_edges[self.K[i][j-1]].subedges.append(dict_of_edges[self.K[i][j]])
                elif self.K[i][j] > 0 and self.K[i][j] in added_data:
                    pass
                elif self.K[i][j] == -1:
                    dict_of_edges[self.K[i][j-1]].subedges.append('s' + str(int(i+1)))
        return master


    def build_tree(self, G_nx, master_edge, master_node, o_of_o=0, i=0, edgelist={}):
        """
        Create a node system and return edgelist dictionary with labels
        """
        o = i
        # revisit
        if len(master_edge.subedges) == 0:
            return i

        has_edge = False
        last_str_idx = 0
        count = 0
        for edge in master_edge.subedges:
            if not isinstance(edge, str):
                has_edge = True
            if isinstance(edge, str):
                last_str_idx = count
            count += 1
        data_si = ''

        count = 0
        for edge in master_edge.subedges:

            # This is an edge 'c_i
            if not isinstance(edge, str):
                i += 1
                node = Node(str(i))
                node.parent = master_node
                node.edge_before = edge.data
                master_node.subnodes.append(node)
                G_nx.add_node(str(i))
                G_nx.add_edge(str(o),str(i))
                edgelist[str(o),str(i)] = edge.data
                n, edgelist, i = self.build_tree(G_nx, edge, node, o_of_o=o, i=i, edgelist=edgelist)
            # This is a string s_i
            else:
                if has_edge == False:
                    data_si += edge + ','
                    if count == last_str_idx:
                        data_si = data_si[:-1]
                        nx.relabel.relabel_nodes(G_nx, {str(o):data_si}, copy=False)
                        master_node.data = data_si
                        edgelist[(str(o_of_o), data_si)] = edgelist[(str(o_of_o), str(o))]
                        del edgelist[(str(o_of_o),str(o))]
                else:
                    data_si += edge + ','
                    if count == last_str_idx:
                        i+=1
                        data_si = data_si[:-1]
                        node = Node(data_si)
                        node.parent = master_node
                        node.edge_before = 'no label'
                        master_node.subnodes.append(node)
                        G_nx.add_node(data_si)
                        G_nx.add_edge(str(o), data_si)

            count += 1

        return master_node, edgelist, i

    def build_graph(self, G, master_node):
        if master_node.subnodes == []:
            G.add_node(master_node)
            return G

        G.add_node(master_node)

        for n in master_node.subnodes:
            G = self.build_graph(G, n)

        return G

    def main_phylogeny(self, showing=True):
        G_nx = nx.DiGraph()
        G_nx.add_node('0')
        #M_, indices = self.radix_sort()
        #master_edge = self.build_phylogeny(self.build_k(M_, indices))
        master_edge = self.build_phylogeny()
        master_node, edgelist, i = self.build_tree(G_nx, master_edge, Node('root'), edgelist={})
        graph = Graph()
        graph = self.build_graph(graph, master_node)
        #base = pyplot.gca().transData
        #rot = transforms.Affine2D().rotate_deg(90)
        if showing:
            print('Printing Phylogeny...')
            fig = plt.figure(figsize=(10,10))
            pos = nx.kamada_kawai_layout(G_nx)
            nx.draw_networkx(G_nx, pos=pos, with_labels=True, fontsize=12, node_color='r',alpha=0.8, width=2)
            nx.draw_networkx_edge_labels(G_nx, pos=pos, edge_labels=edgelist, font_size=12,alpha=0.5,width=8,color='brown')
            plt.title("Perfect Phylogeny")
            plt.show()

        return graph


if __name__ == '__main__':

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

    M_7 = M_10[0:7,0:7]

    #N = 10
    #M_10 = np.array([[random.randint(0, 1) for i in range(N)] for j in range(N)])
    #print(M_10)


    testcases = [M_5, M_7, M_10, M_20]

    for case in testcases:
        phylo = Phylogeny(case)
        print(phylo.M)
        print(phylo.K)
        phylo.main_phylogeny()
