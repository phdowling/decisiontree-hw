__author__ = 'dowling'
from itertools import groupby
from collections import defaultdict


def gini(frequencies):
    total = float(sum(frequencies))
    return sum([freq/total * (1 - freq/total) for freq in frequencies])


def class_dist_to_list(class_dist, length=3):
    return [class_dist.get(x, 0) for x in range(length)]



def calc_gini(points, length=3):
    dist = class_dist_to_list(get_class_distribution(points), length=length)
    return gini(dist)  # by label


def get_class_distribution(points):
    freqs = defaultdict(int)
    for point in points:
        freqs[point[1]] += 1
    return dict(freqs)


def select_most_frequent_label(points):
    class_dist = get_class_distribution(points)
    return sorted(class_dist.items(), key=lambda (k, v): -v)[0][0]


class DecisionTreeNode(object):
    def __init__(self, points, depth=0, dimensions=3):
        # points should be formatted: [((x1, x2, x3, ...), label), ((x1, x2, x3, ...), label), ...]
        self.left_child = None
        self.right_child = None
        self.split_test = None
        self.contained_points = points
        self.label = select_most_frequent_label(self.contained_points)
        self.cost = calc_gini(self.contained_points, length=dimensions)
        self.depth = depth
        self._dimensions = dimensions

    def classify(self, point, details=False):
        # point shoud just be a tuple of n values here, no label in position 1.
        if self.split_test is None:
            # This is a leaf, simply return the
            if details:
                l_fs = get_class_distribution(self.contained_points)
                best_label_freq = l_fs[self.label]
                return self.label, best_label_freq / float(len(self.contained_points))
            else:
                return self.label
        else:
            if self.split_test(point):
                return self.left_child.classify(point, details=details)
            else:
                return self.right_child.classify(point, details=details)

    def build_children(self):
        split_dimension, split_threshold, split_cost = find_best_split(self.contained_points, length=self._dimensions)
        if split_cost < self.cost:  # is this split even worth making? TODO: include a minumum change threshold here
            group1, group2 = split_data(self.contained_points, split_dimension, split_threshold)

            self.split_test = lambda point: point[split_dimension] < split_threshold
            self._split_dimension = split_dimension
            self._split_threshold = split_threshold

            self.left_child = DecisionTreeNode(group1, depth=self.depth+1, dimensions=self._dimensions)
            self.right_child = DecisionTreeNode(group2, depth=self.depth+1, dimensions=self._dimensions)

            return True  # we did a split
        else:
            return False  # this node is already done, no split made

    def to_string(self, level=0):
        test_str = None if self.split_test is None else "x_%s < %s" % (self._split_dimension, self._split_threshold)
        content_str = class_dist_to_list(get_class_distribution(self.contained_points))
        string_rep = "test: %s. class_dist: %s. gini: %s. label: %s" % (test_str, content_str, self.cost, self.label)
        ret = "\t"*level+string_rep+"\n"

        if self.left_child:
            ret += self.left_child.to_string(level+1)
            ret += self.right_child.to_string(level+1)
        return ret

def build_tree(data, max_depth, dimensions=3):
    # do breadth first traversal to build the tree to it's maximum depth
    root = DecisionTreeNode(data, dimensions=3)
    nodes = [root]
    while nodes:
        node = nodes.pop(0)
        if node.depth == max_depth:
            continue
        res = node.build_children()
        if res:
            nodes.append(node.left_child)
            nodes.append(node.right_child)

    return root


def find_best_split(data, length=3):
    total_points = float(len(data))

    best_dimension = None
    best_split_threshold = None
    best_split_cost = 1

    for attribute_dimension in range(len(data[0][0])):  # iterate over all feature dimensions
        # TODO is this a good selection of split candidates?
        split_candidates = [point[0][attribute_dimension] for point in data]  # Try splitting at each possible point
        for possible_split in split_candidates:  # try all our splits on this dimension and track if we beat our min
            first_group, second_group = split_data(data, attribute_dimension, possible_split)

            if not first_group and second_group:
                continue  # this split basically just lumps everything to one side, it's not an actual split

            weight_first = len(first_group) / total_points
            weight_second = len(second_group) / total_points
            split_score = weight_first * calc_gini(first_group, length=length) + weight_second * calc_gini(second_group, length=length)

            if split_score <= best_split_cost:
                best_split_cost = split_score
                best_dimension = attribute_dimension
                best_split_threshold = possible_split

    return best_dimension, best_split_threshold, best_split_cost


def split_data(data, attribute_dimension, split_threshold):
    left = []
    right = []
    for point in data:
        if point[0][attribute_dimension] < split_threshold:
            left.append(point)
        else:
            right.append(point)
    return left, right

if __name__ == "__main__":
    data = []
    with open("homework03.csv", "r") as f:
        for line in list(f.readlines())[1:]:  # skip first line
            x1, x2, x3, label = map(lambda elem: float(elem.strip()), line.split(","))
            data.append(((x1, x2, x3), int(label)))

    classifier = build_tree(data, max_depth=2, dimensions=3)
    print classifier.to_string()

    x_a = (4.1, -0.1, 2.2)
    label_a, probability_l_a = classifier.classify(x_a, details=True)
    print x_a, label_a, probability_l_a

    x_b = (6.1, 0.4, 1.3)
    label_b, probability_l_b = classifier.classify(x_b, details=True)
    print x_b, label_b, probability_l_b