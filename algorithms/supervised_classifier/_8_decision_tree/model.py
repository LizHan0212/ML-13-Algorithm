# algorithms/supervised_classifier/_8_decision_tree/model.py

class DecisionTreeModel:
    def __init__(self, tree):
        self.tree = tree

    # predict for array of samples
    def predict(self, X):
        results = []
        for row in X:
            node = self.tree
            while node["type"] != "leaf":
                feat = node["feature"]
                idx = ["weather","workload","candy"].index(feat)
                v = row[idx]

                if v == node["values"][0]:
                    node = node["children"][0]
                else:
                    node = node["children"][1]
            results.append(node["class"])
        return results
