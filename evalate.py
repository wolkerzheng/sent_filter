#coding=utf8
__author__ = 'Administrator'
from keras import backend as K

def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall))

def getEvaate(predicted,label):
    """

    :param predicted:预测标签
    :param label:正确标签
    :return:（准确率，召回率，f1值）
    """
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    totalLabel = 0
    totalTrue = 0
    totalPredict = 0
    for i in range(len(predicted)):

        for j in range(len(predicted[i])):
            if label[i][j]==1:
                totalLabel+=1
            if label[i][j]==1 and predicted[i][j]>=0.5:
                totalTrue +=1
            if label[i][j]==0 and predicted[i][j]>=0.5:
                totalPredict += 1
    precision = totalTrue/(totalPredict+totalTrue)
    recall = totalTrue/totalLabel
    f1 = 2*precision*recall/(precision+recall)
    return (precision,recall,f1)

def getEvaateRangeSent(predicted,label,sent=1):
    """

    :param predicted:预测标签
    :param label:正确标签
    :return:（准确率，召回率，f1值）
    """
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    totalLabel = 0
    totalTrue = 0
    totalPredict = 0
    for i in range(len(predicted)):

        for j in range(len(predicted[i])):
            if label[i][j]==1:
                totalLabel+=1
                if predicted[i][j]>=0.5 or (j-1>=0 and predicted[i][j-1]>=0.5) or (j+1<len(predicted[i]) and predicted[i][j+1]>=0.5):

                    totalTrue +=1
            if predicted[i][j]>=0.5:
                totalPredict += 1
    precision = totalTrue/(totalPredict)
    recall = totalTrue/totalLabel
    f1 = 2*precision*recall/(precision+recall)
    return (precision,recall,f1)