import logging
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)


def prf1(y_true, y_pred, percent=False):
    """
    Computes precision, recall, and F1 score for each punctuation class and overall.
    :param y_true: Array of labels.
    :param y_pred: Array of predictions.
    :param percent: True if output should be in percentages, False otherwise.
    :return: A dictionary with keys `'comma'`, `'fs'` (full stop), `'qm'` (question mark), and `'overall'`, and each
    value being a tuple (precision, recall, f1).
    """
    stats = confusion_stats(confusion_matrix(y_true, y_pred))
    classes = ['np', 'fs', 'comma', 'qm']
    ignore = ['np']
    results = {}
    total_tp, total_fp, total_fn = 0, 0, 0
    for i, c in enumerate(classes):
        if c in ignore:
            continue
        tp = stats[i][0]
        fp = stats[i][1]
        fn = stats[i][3]
        results[c] = (precision(tp, fp), recall(tp, fn), f1(tp, fp, fn))
        total_tp += tp
        total_fp += fp
        total_fn += fn
    results['overall'] = (precision(total_tp, total_fp), recall(total_tp, total_fn), f1(total_tp, total_fp, total_fn))
    if percent:
        for k in results:
            results[k] = tuple([100*x for x in results[k]])
    return results

def confusion_stats(matrix):
    """
    Computes the number of true positives, false positives, true negatives, and false negatives for each class.
    :param matrix: Confusion matrix.
    :return: stats, which is a list of tuples, where stats[i][0] is class i's number of true positives, stats[i][1] is
    class i's number of false positives, stats[i][2] is class i's number of true negatives, and stats[i][3] is class i's
    number of false negatives.
    """
    assert len(matrix.shape) == 2 and matrix.shape[0] == matrix.shape[1]
    stats = []
    # Loop over each class
    for c in range(matrix.shape[0]):
        c_complement = [i for i in range(matrix.shape[0]) if i != c]
        tp = matrix[c][c]
        fp = sum([matrix[i][c] for i in c_complement])
        tn = sum([matrix[i][j] for i in c_complement for j in c_complement])
        fn = sum([matrix[c][i] for i in c_complement])
        stats.append((tp, fp, tn, fn))
    return stats

def f1(tp, fp, fn):
    """
    Computes F1 score.
    :param tp: Number of true positives.
    :param fp: Number of false positives.
    :param fn: Number of false negatives.
    :return: F1 score.
    """
    p = precision(tp, fp)
    r = recall(tp, fn)
    return 2 * p * r / (p + r)

def precision(tp, fp):
    """
    Computes precision.
    :param tp: Number of true positives.
    :param fp: Number of false positives.
    :return: Precision.
    """
    return tp / (tp + fp)

def recall(tp, fn):
    """
    Computes recall.
    :param tp: Number of true positives.
    :param fn: Number of false negatives.
    :return: Recall.
    """
    return tp / (tp + fn)

def display_results(results):
    """
    Displays results using logger.
    :param results: A dictionary with keys `'comma'`, `'fs'` (full stop), `'qm'` (question mark), and `'overall'`, and each
    value being a tuple (precision, recall, f1).
    """
    output = '\n' + '*'*70 + '\n'
    for k in results:
        output += 'P, R, F1 for ' + k + ': ' + str(results[k]) + '\n'
    output += '*'*70
    logging.info(output)
