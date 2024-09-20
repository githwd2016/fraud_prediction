# -*- coding: utf-8 -*-  
"""
@Author: winton
@File: evaluation.py
@Time: 2018/6/26 上午10:25
@Description:
"""
import math

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error, \
    log_loss


def evaluation(test_y, pred_proba_y, threshold):
    assert len(test_y) == len(pred_proba_y)
    y = zip(pred_proba_y, test_y)
    y = list(map(lambda x: list(x), y))
    y.sort(key=lambda x: x[0], reverse=True)
    ndcg = 0
    if threshold != 0:
        # ndcg
        dcg = 0
        dcg_ideal = 0
        for i, j in enumerate(y[:int(len(y) * threshold)]):
            dcg += (2 ** int(j[1]) - 1) / math.log2(i + 2)
            dcg_ideal += 1 / math.log2(i + 2)
        ndcg = dcg / dcg_ideal
        for i, j in enumerate(y):
            if i <= int(len(y) * threshold):
                j[0] = 1
            else:
                j[0] = 0
        pred_yt, test_yt = zip(*y)
        pred_yt = list(pred_yt)
        test_yt = list(test_yt)
    else:
        test_yt = test_y
        pred_yt = [0 if x < 0.5 else 1 for x in pred_proba_y]
    # test_y15 = test_y
    # pred_y15 = np.zeros(pred_proba_y.shape)
    # for i, j in enumerate(pred_proba_y):
    #     if j > 0.5:
    #         pred_y15[i] = 1
    #     else:
    #         pred_y15[i] = 0
    acc = accuracy_score(test_yt, pred_yt)
    precision = precision_score(test_yt, pred_yt, pos_label=1)
    recall = recall_score(test_yt, pred_yt, pos_label=1)
    f1 = f1_score(test_yt, pred_yt, pos_label=1)
    auc = roc_auc_score(test_y, pred_proba_y)
    # MSPE
    mspe = mean_squared_error(test_y, pred_proba_y)
    # Logarithmic Loss
    log_l = log_loss(test_y, pred_proba_y)
    return acc, precision, recall, f1, auc, ndcg, log_l, mspe


if __name__ == '__main__':
    test_y = []
    pred_y = []
    for year in range(2013, 2022):
        file = f'result/rnn/ICD_dataset_240918_1700/{year}_2_300_0.9_200.txt'
        # test_y = []
        # pred_y = []
        with open(file, 'r') as f:
            for line in f.readlines():
                i, j, k = map(float, line.split())
                test_y.append(j)
                pred_y.append(k)
            # _, p, r, _, a, n, _, _ = evaluation(test_y, pred_y, threshold=0.3)
            # print(year, p, r, n, a)
    _, p, r, _, a, n, _, _ = evaluation(test_y, pred_y, threshold=0.3)
    print(p, r, n, a)
