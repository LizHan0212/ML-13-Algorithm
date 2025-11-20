# algorithms/supervised_classifier/_8_decision_tree/train.py

from algorithms.supervised_classifier._8_decision_tree.model import DecisionTreeModel

# Simple ID3-like tree for categorical features
def train_decision_tree(X, y):
    features = ["weather", "workload", "candy"]

    # Build a simple 2-way split tree
    # Our features are all two-category values.
    # Format:
    # {
    #   "type": "node",
    #   "feature": "weather",
    #   "values": ["sunny", "rain"],
    #   "children": [node_if_sunny, node_if_rain]
    # }

    # Helper: get most common label
    def majority(arr):
        return max(set(arr), key=arr.count)

    # Build a tiny tree manually (simple educational version)
    def build_tree(X, y, depth=0):
        # If all same label → leaf
        if len(set(y)) == 1:
            return {"type": "leaf", "class": y[0]}

        # If no features left
        if depth >= 3:
            return {"type": "leaf", "class": majority(y)}

        # Choose feature by order
        feat = features[depth]
        idx = features.index(feat)

        # Split by feature value
        v1, v2 = ("sunny","rain") if feat=="weather" else \
                 ("light","heavy") if feat=="workload" else \
                 ("yes","no")

        # Build children
        child1_X = []
        child1_y = []
        child2_X = []
        child2_y = []

        for row, label in zip(X, y):
            if row[idx] == v1:
                child1_X.append(row)
                child1_y.append(label)
            else:
                child2_X.append(row)
                child2_y.append(label)

        # If a split is empty, fallback
        if len(child1_y)==0 or len(child2_y)==0:
            return {"type": "leaf", "class": majority(y)}

        return {
            "type": "node",
            "feature": feat,
            "values": [v1, v2],
            "children": [
                build_tree(child1_X, child1_y, depth+1),
                build_tree(child2_X, child2_y, depth+1),
            ]
        }

    # Build full tree
    tree = build_tree(X, y)

    # Wrap in model class
    return DecisionTreeModel(tree)
