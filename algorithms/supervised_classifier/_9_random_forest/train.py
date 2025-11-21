from .utils import load_dataset, bootstrap_sample, pick_feature_subset, majority


def split_by_feature(X, y, feature_idx):
    buckets = {}
    for xi, yi in zip(X, y):
        val = xi[feature_idx]
        if val not in buckets:
            buckets[val] = ([], [])
        buckets[val][0].append(xi)
        buckets[val][1].append(yi)
    return buckets


def build_tree(X, y, feats, depth=0, max_depth=3):
    # STOP 1: all labels same
    if len(set(y)) == 1:
        return {"leaf": y[0]}

    # STOP 2: depth limit
    if depth >= max_depth:
        return {"leaf": majority(y)}

    # Try each feature
    best_feat = None
    best_buckets = None
    best_reduction = -1

    initial_size = len(X)

    for fi in feats:
        buckets = split_by_feature(X, y, fi)
        # reduction = how many samples changed groups
        reduction = 0
        for _, group in buckets.items():
            if len(group[0]) < initial_size:
                reduction += 1

        if reduction > best_reduction:
            best_reduction = reduction
            best_feat = fi
            best_buckets = buckets

    # STOP 3: no meaningful reduction
    if best_reduction <= 0:
        return {"leaf": majority(y)}

    # Create children
    children = {}
    for val, (Xc, yc) in best_buckets.items():
        if len(Xc) == 0:
            # fallback
            children[val] = {"leaf": majority(y)}
        else:
            children[val] = build_tree(Xc, yc, feats, depth + 1, max_depth)

    # node stores index inside feature subset (0 or 1)
    subset_index = feats.index(best_feat)
    return {"feature": subset_index, "children": children}


def train_random_forest(n_trees):
    X, y = load_dataset()

    forest = []
    feat_subsets = []

    for _ in range(n_trees):
        Xb, yb = bootstrap_sample(X, y)
        feats = pick_feature_subset()
        tree = build_tree(Xb, yb, feats)
        forest.append(tree)
        feat_subsets.append(feats)

    return forest, feat_subsets, X, y
