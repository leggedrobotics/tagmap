import os
import numpy as np
import networkx as nx


class PoseGraph:
    """
    Class implementing methods for working with a pose graph.

    Currently, this class is a wrapper providing additional functionality on top
    of a pose graph generated elsewhere.
    It does not support generating pose graphs or modification of the stored pose graph!
    """

    def __init__(self, points, edges):
        """
        Args:
            points: np.ndarray (N, m) of m dimensional points
            edges: np.ndarray (E, 2) of indices into points
        """
        self._nodes = points

        self._graph = nx.Graph()
        self._graph.add_nodes_from(np.arange(points.shape[0]))

        edge_lengths = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
        for (i, j), length in zip(edges, edge_lengths):
            self._graph.add_edge(i, j, length=length)

        # make sure that the graph is connected
        if not nx.is_connected(self._graph):
            raise ValueError(
                "Cannot create pose graph with arguments representing a disconnected graph"
            )

    def closest_node_idx(self, point):
        """
        Args:
            point: np.ndarray (m,) of a m dimensional point

        Returns:
            index of the closest node
        """
        dists = np.linalg.norm(self._nodes - point, axis=1)
        return np.argmin(dists)

    def closest_node(self, point):
        """
        Args:
            point: np.ndarray (m,) of a m dimensional point

        Returns:
            coordinates of the closest node
        """
        return self._nodes[self.closest_node_idx(point)]

    def shortest_path(self, start_point, end_point):
        """
        Args:
            start_point: np.ndarray (m,) of a m dimensional point
            end_point: np.ndarray (m,) of a m dimensional point

        Returns:
            list of indices of nodes on the shortest path
        """
        start_idx = self.closest_node_idx(start_point)
        end_idx = self.closest_node_idx(end_point)

        return nx.shortest_path(self._graph, start_idx, end_idx, weight="length")

    def shortest_path_length(self, start_point, end_point):
        """
        Args:
            start_point: np.ndarray (m,) of a m dimensional point
            end_point: np.ndarray (m,) of a m dimensional point

        Returns:
            length of the shortest path
        """
        start_idx = self.closest_node_idx(start_point)
        end_idx = self.closest_node_idx(end_point)

        return nx.shortest_path_length(self._graph, start_idx, end_idx, weight="length")

    def save(self, save_dir):
        """
        Args:
            save_dir: path to save the graph to
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.save(os.path.join(save_dir, "edges.npy"), np.array(self._graph.edges))
        np.save(os.path.join(save_dir, "node_coords.npy"), self._nodes)

    @classmethod
    def load(cls, load_dir):
        """
        Args:
            load_dir: path to load the graph from

        Returns:
            PoseGraph object
        """
        edges = np.load(os.path.join(load_dir, "edges.npy"))
        nodes = np.load(os.path.join(load_dir, "node_coords.npy"))
        return cls(nodes, edges)

    @property
    def nodes(self):
        return self._nodes.copy()
