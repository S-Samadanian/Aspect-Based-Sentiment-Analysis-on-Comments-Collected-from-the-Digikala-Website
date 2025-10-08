#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, fbeta_score


# In[2]:


def subset_accuracy(y_true, y_pred):
    return np.mean(np.all(y_true == y_pred, axis=1))

#Fraction of labels incorrectly predicted
def hamming_loss(y_true, y_pred):
    return np.mean(np.not_equal(y_true, y_pred))

#Intersection over Union (Jaccard Similarity) averaged over samples
def jaccard_accuracy(y_true, y_pred):
    intersection = np.sum(np.logical_and(y_true, y_pred), axis=1)
    union = np.sum(np.logical_or(y_true, y_pred), axis=1)
    return np.mean(np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union != 0))

def accuracy_multi_tf(y_true, y_pred):
    """
    Custom accuracy metric for multi-label classification.
    This calculates the intersection over the union for each label across all samples.
    """

    y_pred = tf.cast(y_pred > 0.5, tf.float32) 
    
    y_true = tf.cast(y_true, tf.float32)

    intersection = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=1)

    #  Union ( true positives + false positives)
    union = tf.reduce_sum(tf.clip_by_value(y_true + y_pred, 0, 1), axis=1)

    # Jaccard similarity
    accuracy = tf.math.divide_no_nan(intersection, union)
    
    return tf.reduce_mean(accuracy) 

def precision_mean(y_true, y_pred):
    tp = np.sum(np.logical_and(y_true == 1, y_pred == 1), axis=1)
    pred_pos = np.sum(y_pred == 1, axis=1)
    precision = np.divide(tp, pred_pos, out=np.zeros_like(tp, dtype=float), where=pred_pos != 0)
    return np.mean(precision)


def recall_mean(y_true, y_pred):
    tp = np.sum(np.logical_and(y_true == 1, y_pred == 1), axis=1)
    actual_pos = np.sum(y_true == 1, axis=1)
    recall = np.divide(tp, actual_pos, out=np.zeros_like(tp, dtype=float), where=actual_pos != 0)
    return np.mean(recall)


def fbeta_mean(y_true, y_pred, beta=1.0):
    p = precision_mean(y_true, y_pred)
    r = recall_mean(y_true, y_pred)
    denom = (beta**2 * p + r)
    return (1 + beta**2) * p * r / denom if denom != 0 else 0.0

#Compute TP, FP, TN, FN per label
def multilabel_confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1), axis=0)
    fp = np.sum((y_true == 0) & (y_pred == 1), axis=0)
    tn = np.sum((y_true == 0) & (y_pred == 0), axis=0)
    fn = np.sum((y_true == 1) & (y_pred == 0), axis=0)
    return tp, fp, tn, fn

#Macro average of accuracy per label
def accuracy_macro(y_true, y_pred):
    tp, fp, tn, fn = multilabel_confusion_matrix(y_true, y_pred)
    return np.mean((tp + tn) / (tp + fp + tn + fn + 1e-8))

#Micro accuracy (global)
def accuracy_micro(y_true, y_pred):
    tp, fp, tn, fn = multilabel_confusion_matrix(y_true, y_pred)
    total = tp + fp + tn + fn
    return (np.sum(tp) + np.sum(tn)) / np.sum(total)

#Compute multiple evaluation metrics for multi-label classification
def evaluate_model(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = {
	'accuracy':accuracy_multi_tf(y_true,y_pred).numpy(),
        'subset_accuracy': subset_accuracy(y_true, y_pred),
        'hamming_loss': hamming_loss(y_true, y_pred),
        'jaccard_accuracy': jaccard_accuracy(y_true, y_pred),
        'precision_mean': precision_mean(y_true, y_pred),
        'recall_mean': recall_mean(y_true, y_pred),
        'fbeta_mean': fbeta_mean(y_true, y_pred),
        'accuracy_macro': accuracy_macro(y_true, y_pred),
        'accuracy_micro': accuracy_micro(y_true, y_pred),
        'precision_macro_sklearn': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_micro_sklearn': precision_score(y_true, y_pred, average='micro', zero_division=0),
        'recall_macro_sklearn': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_micro_sklearn': recall_score(y_true, y_pred, average='micro', zero_division=0),
        'fbeta_macro_sklearn': fbeta_score(y_true, y_pred, average='macro', beta=1, zero_division=0),
        'fbeta_micro_sklearn': fbeta_score(y_true, y_pred, average='micro', beta=1, zero_division=0)
    }

    return metrics


# In[ ]:




