import numpy as np
import sklearn.metrics
import torch


def macro_recall(pred_y, y, n_grapheme=168, n_vowel=11, n_consonant=7):
    pred_y = torch.split(pred_y, [n_grapheme, n_vowel, n_consonant], dim=1)
    pred_labels = [torch.argmax(py, dim=1).cpu().numpy() for py in pred_y]

    y = y.cpu().numpy()
    print(pred_labels)
    print(y[:, 0])
    recall_grapheme = sklearn.metrics.recall_score(pred_labels[0], y[:, 0], average='macro')
    recall_vowel = sklearn.metrics.recall_score(pred_labels[1], y[:, 1], average='macro')
    recall_consonant = sklearn.metrics.recall_score(pred_labels[2], y[:, 2], average='macro')

    scores = [recall_grapheme, recall_vowel, recall_consonant]
    final_score = {'recall': np.average(scores, weights=[2, 1, 1]),
                    'recall_grapheme': scores[0], 
                    'recall_vowel': scores[1], 
                    'recall_consonant': scores[2]
                    }

    return final_score


def calc_macro_recall(solution, submission):
    # solution df, submission df
    scores = []
    for component in ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']:
        y_true_subset = solution[component].values
        y_pred_subset = submission[component].values
        scores.append(sklearn.metrics.recall_score(
            y_true_subset, y_pred_subset, average='macro'))
    final_score = np.average(scores, weights=[2, 1, 1])
    return final_score