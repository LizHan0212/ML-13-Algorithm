
def predict_stump(stump, row):
    feature = stump["feature"]
    pred_map = stump["pred_map"]   # dict: value → class

    val = row[feature]
    return pred_map.get(val, 0)    # safe default


def adaboost_predict(stumps, alphas, row):
    total = 0.0

    for stump, a in zip(stumps, alphas):
        pred = predict_stump(stump, row)
        # map {0→-1, 1→+1}
        pred2 = 1 if pred == 1 else -1
        total += a * pred2

    return 1 if total > 0 else 0
