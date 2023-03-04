import numpy as np


# KDTreeNode
class KDTreeNode:
    """
    The node of the KDTree.
    """

    def __init__(self, data, depth: int = 0):
        self.data = data
        """(np.ndarray) shape (n_features,) the data of the node, i.e. the point"""
        self.depth: int = depth
        """(int) the depth of the node, the root node has depth 0"""
        self.left: KDTreeNode = None
        """(KDTreeNode) the left child of the node"""
        self.right: KDTreeNode = None
        """(KDTreeNode) the right child of the node"""

    def __str__(self):
        return str('KDTreeNode with data: {}'.format(self.data))


class KDTree:
    """
    The KDTree class.

    The KDTree is a binary tree, which is used to search the nearest neighbors
    of a given point. The basic idea is *divide and conquer*, which means we
    divide the space into several regions, and then search the nearest neighbors
    in the region.

    It can retrieve the k nearest neighbors of a given point in $O(log(n))$ time,

    """

    def __init__(self, points):
        """
        Build the KDTree from the given data.

        :param points: (np.ndarray) shape (n_samples, n_features) the data to build the tree
        """
        self.k = points.shape[1]
        """(int) the dimension of the data"""
        self.root = self.build(points)
        """(KDTreeNode) the root of the tree"""
        print('self.root', self.root)

    def build(self, points):
        """
        Build up the KDTree from the given data.

        :param points: (np.ndarray) shape (n_samples, n_features) the data to build the tree
        :return: (KDTreeNode) the root of the tree
        """
        return self._build(points)

    def _build(self, points, depth=0):
        """
        Implementation of the build method using Recursion.

        :param points: (np.ndarray) shape (n_samples, n_features) the data to build the tree
        :param depth: (int) the depth of the tree
        :return: (KDTreeNode) the root of the tree
        """
        if len(points) == 0:
            return None
        # Choose the axis to split, we just iterate over the axis in order,
        # e.g. the first split will be made on axis 0, the second on axis
        # 1, etc...
        axis = depth % self.k
        # Sort the data by the axis, to do this, we have to slice the
        # array and then argsort it, since numpy doesn't support lambda
        # expression for sorting
        sorted_data = points[points[:, axis].argsort()]
        mid = len(sorted_data) // 2
        node = KDTreeNode(sorted_data[mid], depth)
        node.left = self._build(sorted_data[:mid], depth + 1)
        node.right = self._build(sorted_data[mid + 1:], depth + 1)
        return node

    def search_knn(self, point, k):
        """
        Search the k nearest neighbors of the given point using KDTree.

        :param point: (np.ndarray) shape (ndim,) the point to search
        :param k: (int) the number of neighbors to search
        :return: (list) the k nearest neighbors
        """
        return self._search_knn(point, k)

    def _search_knn(self, point, k: int):
        """
        Implementation of the `search_knn method` using Iteration.

        If no enough neighbors are found, the neighbors will be filled
        with (None, np.inf).

        :param point: (np.ndarray) shape (ndim) the point to search
        :param k: (int) the number of neighbors to search
        :param neighbors: (list) the neighbors found so far
        :return: (list) the k nearest neighbors, length is always `k`
        """
        assert k > 0, "k must be a positive integer"

        # Initialize the neighbors
        neighbors = [(None, np.inf) for _ in range(k)]

        # Search the nearest neighbors using stack
        stack = [self.root]
        while len(stack) > 0:
            node = stack.pop()
            if node is None:
                continue
            distance = np.linalg.norm(point - node.data)
            if distance < neighbors[-1][1]:
                neighbors.append((node.data, distance))
                neighbors.sort(key=lambda x: x[1])
                neighbors = neighbors[:k]
            axis = node.depth % self.k
            if point[axis] < node.data[axis]:
                stack.append(node.right)
                stack.append(node.left)
            else:
                stack.append(node.left)
                stack.append(node.right)
        return neighbors


# test
if __name__ == '__main__':
    data = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
    tree = KDTree(data)
    print(tree.search_knn(np.array([3, 4.5]), 7))

