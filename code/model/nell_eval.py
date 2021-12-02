from __future__ import division
from collections import defaultdict
import numpy as np
def nell_eval(model_answers, correct_answers):
    test_data_path = correct_answers
    test_prediction_path = model_answers
    f = open(test_data_path , encoding='utf-8')
    test_data = f.readlines()
    f.close()
    preds = {}
    with open(test_prediction_path , encoding='utf-8') as f:
        for line in f:
            score = 0
            if (len(line.split()) == 4):
                e1, e2, c, score = line.strip().split()
            if (len(line.split()) == 3):
                e1, e2, score = line.strip().split()

            score = float(score)
            if (e1, e2) not in preds:
                preds[(e1, e2)] = score
            else:
                if preds[(e1,e2)] < score:
                    preds[(e1,e2)] = score
    def get_pred_score(e1, e2):
        if (e1, e2) in preds:
            return preds[(e1,e2)]
        else:
            return -np.inf
    test_pairs = defaultdict(lambda : defaultdict(int))
    for line in test_data:
        e1 = line.split(',')[0].replace('thing$','')
        e2 = line.split(',')[1].split(':')[0].replace('thing$','')

        label = 1 if line[-2] == '+' else 0
        test_pairs[e1][e2] = label
    aps = []
    score_all = []
    for e1 in test_pairs:
        y_true = []
        y_score = []
        for  e2 in test_pairs[e1]:
            score = get_pred_score(e1, e2)
            score_all.append(score)
            y_score.append(score)
            y_true.append(test_pairs[e1][e2])
        count = list(zip(y_score, y_true))
        count.sort(key=lambda x: x[0], reverse=True)
        ranks = []
        correct = 0
        for idx_, item in enumerate(count):
            if item[1] == 1:
                correct += 1
                ranks.append(correct / (1.0 + idx_))
        if len(ranks) == 0:
            ranks.append(0)
        aps.append(np.mean(ranks))
    mean_ap = np.mean(aps)
    print('MINERVA MAP: {} ({} queries evaluated)'.format( mean_ap, len(aps)))
 




