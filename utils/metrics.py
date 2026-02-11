import numpy as np
from sklearn.metrics import (

    accuracy_score,
    matthews_corrcoef,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


def compute_accuracy(preds, labels):
    return accuracy_score(labels, preds) * 100.0


def compute_mcc(preds, labels):
    return matthews_corrcoef(labels, preds)


def compute_pcc(preds, labels):
    return float(np.corrcoef(labels, preds)[0, 1])


def compute_f1(preds, labels, average='weighted'):
    return f1_score(labels, preds, average=average)


def compute_precision(preds, labels, average='weighted'):
    return precision_score(labels, preds, average=average)


def compute_recall(preds, labels, average='weighted'):
    return recall_score(labels, preds, average=average)


def compute_confusion_matrix(preds, labels):
    return confusion_matrix(labels, preds)


def compute_task_metric(preds, labels, metric_name='accuracy'):
    if metric_name == 'accuracy':
        return compute_accuracy(preds, labels)
    elif metric_name == 'mcc':
        return compute_mcc(preds, labels)
    elif metric_name == 'pcc':
        return compute_pcc(preds, labels)
    elif metric_name == 'f1':
        return compute_f1(preds, labels)
    raise ValueError(f"Unknown metric: {metric_name}")
