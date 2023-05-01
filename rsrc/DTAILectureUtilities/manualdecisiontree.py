#!/usr/bin/env python3
# encoding: utf-8
"""
Created by Wannes Meer.
Copyright (c) 2019-2023 KU Leuven. All rights reserved.
"""
import numpy as np
from sklearn.tree import export_graphviz


class Node:
    def __init__(self, parent=None, child_left=None, child_right=None, feature=None, threshold=None, value=None):
        self.parent = parent
        self.child_left = child_left
        self.child_right = child_right
        self.feature = feature
        self.threshold = threshold
        self.value = value

    def add_child_left(self, node=None):
        if node is None:
            node = Node()
        self.child_left = node
        node.parent = self
        return node

    def add_child_false(self, node=None):
        return self.add_child_left(node)

    def add_child_right(self, node=None):
        if node is None:
            node = Node()
        self.child_right = node
        node.parent = self
        return node

    def add_child_true(self, node=None):
        return self.add_child_right(node)

    def add_children(self, node_false=None, node_true=None):
        node_false = self.add_child_left(node_false)
        node_true = self.add_child_right(node_true)
        return node_false, node_true

    def add_test(self, feature, threshold, node_false=None, node_true=None):
        self.feature = feature
        self.threshold = threshold
        node_false, node_true = self.add_children(node_false, node_true)
        return node_false, node_true

    def set_class(self, class_index):
        self.value = [0, 0]
        self.value[class_index] = 1

    def update_datastructure(self, nid, nodes):
        next_nid = nid + 1
        left_nid = -1
        right_nid = -1
        if self.child_left is not None:
            left_nid = next_nid
            next_nid = self.child_left.update_datastructure(left_nid, nodes)
        if self.child_right is not None:
            right_nid = next_nid
            next_nid = self.child_right.update_datastructure(right_nid, nodes)
        nodes[nid] = {
            "children_left": left_nid,
            "children_right": right_nid,
            "feature": self.feature,
            "threshold": self.threshold,
            "value": self.value
        }
        return next_nid


class Tree:
    def __init__(self):
        """
        Based on https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
        """
        self.changed = True
        self.root = Node(parent=None)
        self.children_left = None
        self.children_right = None
        self.feature = None
        self.threshold = None
        self.value = None
        self.impurity = None
        self.feature_names = None
        self.n_outputs = 2  # we only support binary classifiers
        self.n_classes = None
        self.n_node_samples = None
        self.weighted_n_node_samples = None

    def update_datastructure(self):
        if not self.changed:
            return
        nodes = {}
        feature_names = {}
        next_nid = self.root.update_datastructure(0, nodes)
        assert next_nid == len(nodes)

        self.children_left = np.full(next_nid, -1)
        self.children_right = np.full(next_nid, -1)
        self.feature = np.full(next_nid, -1)
        self.threshold = np.full(next_nid, -1)
        self.impurity = np.full(next_nid, -1)
        self.n_node_samples = np.full(next_nid, -1)
        self.weighted_n_node_samples = np.full(next_nid, -1)
        self.n_classes = np.full(next_nid, -1)
        self.value = np.full((next_nid, self.n_outputs), -1)

        for nid, ndata in nodes.items():
            self.children_left[nid] = ndata["children_left"]
            self.children_right[nid] = ndata["children_right"]
            if ndata["feature"] is None:
                self.feature[nid] = -1
            else:
                if ndata["feature"] not in feature_names:
                    feature_names[ndata["feature"]] = len(feature_names)
                self.feature[nid] = feature_names[ndata["feature"]]
            self.threshold[nid] = ndata["threshold"] if ndata["threshold"] is not None else -1
            self.value[nid, :] = ndata["value"] if ndata["value"] is not None else -1

        self.feature_names = [f"F{i}" for i in range(len(nodes))]
        for fname, fid in feature_names.items():
            self.feature_names[fid] = fname

        self.changed = False


class DecisionTreeClassifier:
    def __init__(self):
        self.tree_ = Tree()
        self.criterion = "Manual"
        self.n_features_ = None
        self.n_features_in_ = None

    def fit(self, *args, **kwargs):
        print("Fitting is not supported, build tree manually.")

    def predict(self, x):
        self.update_datastructure()
        values = np.empty(x.shape[0])
        for i in range(x.shape[0]):
            values[i] = self.predict_instance(x[i:i+1, :], 0)
        return values

    def predict_instance(self, Xi, i):
        if self.tree_.children_left[i] == -1:
            # It is leaf
            return np.argmax(self.tree_.value[i])
        feat = self.tree_.feature[i]
        thr = self.tree_.threshold[i]
        Xij = Xi[0, feat]
        # print(f"{Xi}[{feat}] = {Xij} <= {thr}")
        if Xij <= thr:
            return self.predict_instance(Xi, self.tree_.children_left[i])
        else:
            return self.predict_instance(Xi, self.tree_.children_right[i])

    @property
    def root(self):
        return self.tree_.root

    def update_datastructure(self):
        self.tree_.update_datastructure()
        self.n_features_ = len(self.tree_.feature_names)
        self.n_features_in_ = self.n_features_

    def feature_names(self):
        self.update_datastructure()
        return self.tree_.feature_names

    def target_names(self):
        return None


def test_or():
    from pathlib import Path
    from itertools import product
    tree = DecisionTreeClassifier()
    x1f, x1t = tree.root.add_test("X1", 0)
    x2f, x2t = x1f.add_test("X2", 0)
    x1t.value = [0, 1]
    x2t.value = [0, 1]
    x2f.value = [1, 0]
    x1x2 = np.array(list(product([0, 1], repeat=2)))
    c = tree.predict(x1x2)
    for instance, prediction in zip(x1x2, c):
        print(f"{instance} -> {prediction}")

    dot_fn = Path(".") / "tree.gv"
    print(f"Write Graphviz dot file to {dot_fn.absolute()}")
    with dot_fn.open("w") as ofile:
        txt = export_graphviz(tree, out_file=None,
                               feature_names=tree.feature_names(),
                               class_names=["false", "true"])
        print(txt, file=ofile)


def main(argv=None):
    test_or()


if __name__ == "__main__":
    import sys
    sys.exit(main())
