import numpy as np
import os

current_folder = os.path.dirname(os.path.abspath(__file__))

# Unicode characters used to print the tree
VERTICAL = "\u2502"
UP_IN = "\u250C"
DOWN_IN = "\u2514"
OUT_UP = "\u2518"
OUT_DOWN = "\u2510"

features = np.array([
    [37, 44000, 1, 0],
    [61, 52000, 1, 0],
    [23, 44000, 0, 0],
    [39, 38000, 0, 1],
    [48, 49000, 0, 0],
    [57, 92000, 0, 1],
    [38, 41000, 0, 1],
    [27, 35000, 1, 0],
    [23, 26000, 1, 0],
    [38, 45000, 0, 0],
    [32, 50000, 0, 0],
    [25, 52000, 1, 0]
])
labels = np.array([1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1])
features_names = ["age", "salary", "resident", "siblings"]


class Tester(object):
    def __init__(self):
        self.questions = {}

    def add_test(self, question, test_function):
        self.questions[question] = test_function

    def run(self):
        for question in self.questions:
            success, comment = self.questions[question]()
            if success:
                print("Question %s: [PASS]" % question)
            else:
                print("Question %s: [FAIL]" % question, comment)


def test_leaf(LeafNode):
    tester = Tester()
    labels = np.array([1, -1, -1, 1, -1, 1, -1, 2, 2, 2])
    ins = "\nlabels:" + str(labels)
    topic = "Testing LeafNode "

    def test_compute_label():
        outs = -1
        comment = topic + "compute_label" + ins + "\n expected output:  \n" + str(outs)
        leaf = LeafNode(labels)
        obtained = leaf.compute_label(labels)
        comment = comment + "\n obtained: " + str(obtained)
        if np.allclose(obtained, outs, atol=1e-5):
            return True, comment
        return False, comment

    tester.add_test("1.2", test_compute_label)
    tester.run()


def test_entropy(entropy_func):
    tester = Tester()
    labels = np.array([1, 1, 2, 2, 3, 3, 3, 3])
    ins = "\nlabels:" + str(labels)
    topic = "Testing Entropy Function "

    def test_entropyfunc():
        outs = 1.0397207708399179
        comment = topic + ins + "\n expected output:  \n" + str(outs)
        obtained = entropy_func(labels)
        comment = comment + "\n obtained: " + str(obtained)
        if np.allclose(obtained, outs, atol=1e-5):
            return True, comment
        return False, comment

    tester.add_test("1.3", test_entropyfunc)
    tester.run()


def test_information_gain(information_gain_func, entropy_func):
    tester = Tester()
    labels = np.array([0, 0, 1, 1, 2, 2])
    left_indices = np.array([0, 1, 3])
    right_indices = np.array([2, 4, 5])

    ins = "\nlabels:" + str(labels) + "\nleft_indices:" + str(left_indices) + "\nright_indices" + str(right_indices)
    topic = "Testing Information Gain Function "

    def test_infogain():
        outs = 0.4620981203732968
        comment = topic + ins + "\n expected output:  \n" + str(outs)
        obtained = information_gain_func(labels, left_indices, right_indices, entropy_func)
        comment = comment + "\n obtained: " + str(obtained)
        if np.allclose(obtained, outs, atol=1e-5):
            return True, comment
        return False, comment

    tester.add_test("1.4", test_infogain)
    tester.run()


def test_best_partition(best_partition_func, entropy_func):
    tester = Tester()
    labels = np.array([0, 0, 1, 1, 2, 2])
    features = np.array([
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 0, 0]
    ])

    ins = "\n X:" + str(features) + ",\nlabels:" + str(labels)
    topic = "Testing Best Partition Function "

    def test_partition():
        outs = (2, 0.5, 0.4620981203732968)
        comment = topic + ins + "\n expected output:  \n" + str(outs)
        obtained = best_partition_func(features, labels, entropy_func)
        comment = comment + "\n obtained: " + str(obtained)
        if np.allclose(obtained, outs, atol=1e-5):
            return True, comment
        return False, comment

    tester.add_test("1.5", test_partition)
    tester.run()

def test_tree_build(DecisionTree, entropy_func):
    tester = Tester()
    correct_tree = np.load(os.path.join(current_folder, "tree_depth3_min2_entropy.npy"), allow_pickle=True)[0]
    topic = "Testing DecisionTree build with depth 3 and min_samples_split 2"
    expected_tree = "\n".join(_node_to_string(correct_tree.root, features_names)[0])
    ins = ", using Problem1 features, and labels"

    def test_build():
        tree = DecisionTree(max_depth=3, min_samples_split=2, impurity_measure=entropy_func).fit(features, labels)
        obtained_tree = "\n".join(_node_to_string(tree.root, features_names)[0])
        comment = topic + ins + "\n expected output: \n" + expected_tree + "\n obtained: " + obtained_tree
        if compare_trees(correct_tree, tree):
            return True, comment
        return False, comment

    tester.add_test("1.6", test_build)
    tester.run()


def _node_to_string(root_node, features_names):
    if root_node is None:
        return "None"
    if 'leaf' in str(type(root_node)).lower():
        return [VERTICAL + "label: %i" % root_node.label], 0
    else:
        string = ["%s" % features_names[root_node.feature_id], "%.2f" % root_node.threshold]
        max_len = max([len(s) for s in string])
        string[0] = "|" + string[0] + " " * (max_len - len(string[0])) + VERTICAL + OUT_UP
        string[1] = "|" + string[1] + " " * (max_len - len(string[1])) + VERTICAL + OUT_DOWN
        left, left_pos = _node_to_string(root_node.left_child, features_names)
        right, right_pos = _node_to_string(root_node.right_child, features_names)

        for i in range(0, left_pos):
            left[i] = " " + left[i]
        left[left_pos] = UP_IN + left[left_pos]
        for i in range(left_pos + 1, len(left)):
            left[i] = VERTICAL + left[i]

        for i in range(0, right_pos):
            right[i] = VERTICAL + right[i]
        right[right_pos] = DOWN_IN + right[right_pos]
        for i in range(right_pos + 1, len(right)):
            right[i] = " " + right[i]
        left = [" " * (max_len + 2) + l for l in left]
        right = [" " * (max_len + 2) + r for r in right]
        return left + string + right, len(left)


def _compare_node(node1, node2):
    # Not equal if one is leaf and another is parent
    if type(node1) != type(node2):
        return False
    # leaf nodes are equal if they have the same label
    if 'leaf' in str(type(node1)).lower():
        return node1.label == node2.label
    # parent nodes are equal if they have the same feature_id, threshold, and equal children
    compare_values = (node1.feature_id == node2.feature_id) and np.isclose(node1.threshold, node2.threshold, atol=1e-5)
    if not compare_values:
        return False
    return _compare_node(node1.left_child, node2.left_child) and _compare_node(node1.right_child, node2.right_child)


def compare_trees(tree1, tree2):
    return _compare_node(tree1.root, tree2.root)


def print_tree(decision_tree, features_names=None):
    if features_names is None:
        features_names= ["feat_%i" % i for i in range(decision_tree.num_features)]
    print("\n".join(_node_to_string(decision_tree.root, features_names)[0]))
