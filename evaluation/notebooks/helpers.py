import numpy as np

from typing import List

from tag_mapping.evaluation import LatticeNavigationGraph
from tag_mapping.utils import LineMesh


def generate_lattice_graph_shortest_path_linemeshes(
    lattice_graph: LatticeNavigationGraph, node_inds_a: List, node_inds_b: List
):
    shortest_path_linemeshes = []
    for a_ind in node_inds_a:
        spl = np.inf
        matched_l_ind = None
        for b_ind in node_inds_b:

            new_spl = lattice_graph.shortest_path_length(a_ind, b_ind)
            if new_spl == None:
                continue

            if new_spl < spl:
                spl = new_spl
                matched_l_ind = b_ind

        if matched_l_ind != None:
            sp_inds = lattice_graph.shortest_path(a_ind, matched_l_ind)

            sp_lines = np.zeros((len(sp_inds) - 1, 2)).astype(np.int32)
            sp_lines[:, 0] = np.arange(len(sp_inds) - 1)
            sp_lines[:, 1] = 1 + np.arange(len(sp_inds) - 1)

            sp_linemesh = LineMesh(
                points=lattice_graph.nodes_xyz[sp_inds],
                lines=sp_lines,
                colors=(0, 1, 1),
                radius=0.01,
            )

            shortest_path_linemeshes += sp_linemesh.cylinder_segments

    return shortest_path_linemeshes
