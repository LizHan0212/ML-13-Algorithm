from .utils import majority


def predict_tree(tree, feats, row):
    # If leaf
    if isinstance(tree, dict) and "leaf" in tree:
        return tree["leaf"]
    if not isinstance(tree, dict):
        return tree

    subset_index = tree["feature"]
    original_index = feats[subset_index]
    value = row[original_index]

    if value not in tree["children"]:
        # fallback: majority of children leaves
        leaves = []
        for child in tree["children"].values():
            if isinstance(child, dict) and "leaf" in child:
                leaves.append(child["leaf"])
            else:
                leaves.append(child)
        return majority(leaves)

    return predict_tree(tree["children"][value], feats, row)


def random_forest_predict(forest, feat_subsets, row):
    preds = []
    for tree, feats in zip(forest, feat_subsets):
        preds.append(predict_tree(tree, feats, row))
    return majority(preds)
