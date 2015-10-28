__author__ = 'dowling'
from math import sqrt
from collections import defaultdict

def euclideanDistance(x1, x2):
    return sqrt(sum([(v1 - v2)**2 for (v1, v2) in zip(x1, x2)]))

def getNeighbors(X, Z, xt, k):
    return sorted(
            [(x, Z[i], euclideanDistance(x, xt)) for (i, x) in enumerate(X)],
            key=lambda ptd: ptd[2]
        )[:k]


def getResponse(neighbors):
    # We first group the neighbours by label and track their distances
    label_dists = defaultdict(list)
    for neigh in neighbors:
        label_dists[neigh[1]].append(neigh[2])
    # now we sort the labels by how frequently they occur
    freqs = sorted(label_dists.items(), key=lambda x: -len(x[1]))
    # next, check the most frequent label, and create a list of all labels that have the same frequency
    max_freq = len(freqs[0][1])
    best_labels = [label for label, dists in freqs if len(dists) == max_freq]
    if len(best_labels) == 1:
        # if this is the only label with this max frequency, there is no tie, and we return the label
        return best_labels[0]
    else:
        # otherwise, we try and figure out the most frquent label with the lowest average distance
        average_dists = sorted([(label, sum(label_dists[label])) for label in best_labels], key=lambda item: item[1])
        # return the label with the lowest average distance - if there's still a tie, we don't really care which label we assign
        return average_dists[0][0]

def predict(X, Z, unseen_point, k):
    neighbors = getNeighbors(X, Z, unseen_point, k)
    return getResponse(neighbors)


if __name__ == "__main__":
    data = []
    with open("homework03.csv", "r") as f:
        for line in list(f.readlines())[1:]:  # skip first line
            x1, x2, x3, label = map(lambda elem: float(elem.strip()), line.split(","))
            data.append(((x1, x2, x3), int(label)))
    X = [point[0] for point in data]
    Z = [point[1] for point in data]

    x_a = (4.1, -0.1, 2.2)
    label_a = predict(X, Z, x_a, 3)
    print x_a, label_a

    x_b = (6.1, 0.4, 1.3)
    label_b = predict(X, Z, x_b, 3)
    print x_b, label_b