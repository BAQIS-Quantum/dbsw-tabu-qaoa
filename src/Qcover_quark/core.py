# This code is part of Qcover.
#
# (C) Copyright BAQIS 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Object to solve QAOA problems

The QAOA problems can be represented as an Ising model, and then be transformed to a DAG.
The directed acyclic graph is decomposed by a specified p value, and these subgraphs then
be transformed as circuits and be executed on simulators, using optimizer to get the
optimal parameters of the original Ising model
"""

import sys
import time
from itertools import permutations

from typing import Optional
from collections import defaultdict
import numpy as np
import networkx as nx
from Qcover_quark.utils import get_graph_weights, generate_weighted_graph
from Qcover_quark.optimizers import Optimizer, COBYLA
from Qcover_quark.backends import Backend, CircuitByQulacs
from Qcover_quark.exceptions import GraphTypeError, UserConfigError
import warnings
warnings.filterwarnings("ignore")


class Qcovermini:
    """
    Qcover is a QAOA solver
    """
    # pylint: disable=invalid-name
    def __init__(self,
                 graph: nx.Graph = None,
                 p: int = 1,
                 optimizer: Optional[Optimizer] = COBYLA(),
                 backend: Optional[Backend] = CircuitByQulacs(),
                 research_obj: str = "QAOA"
                 ) -> None:

        assert graph is not None
        self._simple_graph = graph
        self._p = p
        self._research_obj = research_obj
        self._backend = backend
        self._backend._p = p
        self._backend._origin_graph = graph
        self._backend._research = research_obj
        self._optimizer = optimizer
        self._optimizer._p = p

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, new_p):
        self._p = new_p

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, new_backend):
        self._backend = new_backend

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, new_optimizer):
        self._optimizer = new_optimizer

    # def generate_subgraph(self, dtype: str):
    def graph_decomposition(self, p):
        """
        according to the arguments of dtype and p to generate subgraphs from graph
        Args:
            graph (nx.Graph): graph to be decomposed
            dtype (string): set "node" or "edge", the ways according to which to decompose the graph
            p (int): the p of subgraphs
        Return:
            subg_dict (dict) form as {node_id : subg, ..., (node_id1, node_id2) : subg, ...}
        """
        # if dtype not in ["node", "edge"]:
        #     print("Error: wrong dtype, dtype should be node or edge")
        #     return None
        if p <= 0:
            warnings.warn(" the argument of p should be >= 1 in qaoa problem, "
                          "so p would be set to the default value at 1")
            p = 1

        nodes_weight, edges_weight = get_graph_weights(self._simple_graph)

        subg_dict = defaultdict(list)
        # if dtype == 'node':
        for node in self._simple_graph.nodes:
            node_set = {(node, nodes_weight[node])}
            edge_set = set()
            for i in range(p):
                new_nodes = {(nd2, nodes_weight[nd2]) for nd1 in node_set for nd2 in self._simple_graph[nd1[0]]}
                new_edges = {(nd1[0], nd2, edges_weight[nd1[0], nd2]) for nd1 in node_set for nd2 in self._simple_graph[nd1[0]]}
                node_set |= new_nodes
                edge_set |= new_edges

            subg = generate_weighted_graph(node_set, edge_set)
            subg_dict[node] = subg
        # else:
        for edge in self._simple_graph.edges:
            node_set = {(edge[0], nodes_weight[edge[0]]), (edge[1], nodes_weight[edge[1]])}
            edge_set = {(edge[0], edge[1], edges_weight[edge[0], edge[1]])}

            for i in range(p):
                new_nodes = {(nd2, nodes_weight[nd2]) for nd1 in node_set for nd2 in self._simple_graph[nd1[0]]}
                new_edges = {(nd1[0], nd2, edges_weight[nd1[0], nd2]) for nd1 in node_set for nd2 in
                             self._simple_graph.adj[nd1[0]]}
                node_set |= new_nodes
                edge_set |= new_edges

            subg = generate_weighted_graph(node_set, edge_set)
            subg_dict[edge] = subg
        return subg_dict

    def calculate(self, pargs, p=None):
        """
        The framework function which use the backend to calculate the value of expectation,
        and be used as the object function in the optimization function of the optimizer
        Args:
            pargs: the value of the parameter alpha and beta in the circuit
            p: the integer used to define the number of layers the current circuit needs to be superimposed
        Returns:
            the value of expectation calculated by backends
        """
        p = self._p if p is None else p
        element_to_graph = self.graph_decomposition(p=p)

        # checking graph type of given problem
        if not isinstance(self._backend, CircuitByQulacs):
            for k, v in element_to_graph.items():
                ncnt, ecnt = len(v.nodes), len(v.edges)
                try:
                    nreq1 = ncnt * (ncnt - 1) <= 2 * ecnt and ncnt >= 20
                    nreq2 = isinstance(k, int) and v.degree[k] >= 30
                    if nreq1 or nreq2:
                        raise GraphTypeError("The problem is transformed into a dense graph, " \
                           "which is difficult to be solved effectively by Qcover")
                except GraphTypeError as e:
                    print(e)
                    # if not self._hard_to_calcute:
                    #     print(e)
                    #     self._hard_to_calcute = True
                    # sys.exit()

        self._backend._pargs = pargs
        self._backend._element_to_graph = element_to_graph
        return self._backend.expectation_calculation(p)

    def run(self, is_parallel=False):
        self._backend._is_parallel = is_parallel
        x, fun, nfev = self._optimizer.optimize(objective_function=self.calculate)
        res = {"Optimal parameter value": x, "Expectation of Hamiltonian": fun, "Total iterations": nfev}
        return res
