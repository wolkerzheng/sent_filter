__author__ = 'Administrator'
#coding=utf8
from keras.models import load_model
import WordCover
import time
import numpy as np
import jieba
import sys
import pickle
from sklearn.model_selection import train_test_split
from keras import backend as K
import evalate
"""

"""
MAX_SENTS = 60 #每轮对话最大的句子数目
MAX_WORDS = 50 #每句话中最大的词数量

WORD_EMBED_SIZE = 200
SENT_EMBED_SIZE = 100
DOC_EMBED_SIZE = 50

NUM_CLASSES = 2

BATCH_SIZE = 16
NUM_EPOCHS = 10

max_features = 30000
maxlen = 200  # cut texts after this number of words (among top max_features most common words)



def mcor(y_true, y_pred):
     #matthews_correlation
     y_pred_pos = K.round(K.clip(y_pred, 0, 1))
     y_pred_neg = 1 - y_pred_pos


     y_pos = K.round(K.clip(y_true, 0, 1))
     y_neg = 1 - y_pos


     tp = K.sum(y_pos * y_pred_pos)
     tn = K.sum(y_neg * y_pred_neg)


     fp = K.sum(y_neg * y_pred_pos)
     fn = K.sum(y_pos * y_pred_neg)


     numerator = (tp * tn - fp * fn)
     denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))


     return numerator / (denominator + K.epsilon())




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


def eval(model):
    """
    func:
    评估主函数
    params:
    model:载入的模型
    filepath:评估文件
    positive_val:初始设置阈值，为了过滤多轮对话的多少，值越大，过滤的越少
    """
    # disModel = load_model(model)
    start_time = time.time()
    disModel = load_model(model, custom_objects={"mcor": mcor,"recall": recall,"f1": f1})
    # word2id = load_embedding()
    word2id = WordCover.loadWord2Vec()[0]
    # x = pickle.load(open("./e2e_set/train_data.pckl",'rb'))
    # y = pickle.load(open("./e2e_set/train_label.pckl",'rb'))
    x = pickle.load(open("./e2e_set/testdata.pckl",'rb'))
    y = pickle.load(open("./e2e_set/test_label.pckl",'rb'))
    y = list(map(pady,y))

    print('loaded data...')


    x,y = np.array(x),np.array(y)

    print(y.shape)
    print(type(y))
    # for yy in y:
    # y_binary = to_categorical(y)
    # y_binary = np.reshape(y_binary,(7000,45,2))
    y_binary = np.reshape(y,(len(y),MAX_SENTS,1))

    X = np.zeros((len(x),MAX_SENTS,MAX_WORDS))
    for docid,sents in enumerate(x):

        sents = pad_or_truncate(sents,MAX_SENTS,False)
        # print(docid,sents)
        for sid, sent in enumerate(sents):
            # print(sid,sent)
            words = pad_or_truncate(sent, MAX_WORDS)
            # print(words)
            for wid, word in enumerate(words):
                try:
                    word_id = word2id[word]
                except KeyError:
                    word_id = word2id['<unk>']
                X[docid, sid, wid] = word_id
    # docid2mat[int(rec_id)] = M
    print(X.shape)

    # x_tv, x_test, y_tv, y_test = train_test_split(X, y_binary, test_size=0.1)
    for layer in disModel.layers[:-1]:
        layer.trainable = False
    for layer in disModel.layers[-1:]:
        layer.trainable = True
    # predictions = disModel.predict(X)
    # print("accuracy score: ",accuracy_score(y, p))
    score, mcor_val,recall_val, f1_val  =disModel.evaluate(X,y_binary,batch_size=32)
    print('predic',score)
    print(mcor_val)
    print('recall:',recall_val)

    print('p:r:f1:',f1_val)

    end_time = time.time()
    print('total time is :'+str(end_time-start_time))
    pass

def data_transform(batch_data,word2id,batch_size=32,MAX_SENTS=20,MAX_WORDS=70):
    """
    func:
    将帖子数据转换成模型的输入格式
    params:
    batch_data:list，中文帖子
    word2id:字典，词语和id的对照
    batch_size: 批量的大小
    MAX_SENTS: 每轮对话设置的最大句子数
    MAX_WORDS: 每句话中设置的最大词语数
    """
    evalExamples = []
    labels = []

    for eval_data in batch_data:

        e = eval_data.strip().split('\t')

        evalExamples.append(e[0])
        if 0 in map(int,e[1].strip().split('#')):
            labels.append(1)
        else:
            labels.append(0)

    x = [ conversation.strip().split('#') for conversation in evalExamples ]
    # print(type(x))
    X = np.zeros((len(batch_data),MAX_SENTS,MAX_WORDS))

    for docid,sents in enumerate(x):
        sents = pad_or_truncate(sents,MAX_SENTS)
        for sid, sent in enumerate(sents):
            seg_list = ' '.join(jieba.cut(sent))
            words = pad_or_truncate(seg_list.strip().split(' '), MAX_WORDS)
            for wid, word in enumerate(words):
                try:
                    word_id = word2id[word]
                except KeyError:
                    word_id = word2id['<unk>']
                X[docid, sid, wid] = word_id
    return X,labels

# word2id = {"PAD": 0, "UNK": 1}
# word2id={"PAD": 0}
# with  open('./embedding/wiki.zh.text.vector','rb') as f:
# 	nums, ran = f.readline().strip().split()
# 	embedding_weight = np.zeros((int(nums)+1, int(ran)))	#(125605,200)
#
# 	# word_dic = dict()
# 	for i in range(int(nums)):
# 		line = f.readline().strip().split()
# 		word, vec = line[0], line[1:]
# 		vec = list(map(float, vec))
# 		embedding_weight[i+1, :] = vec
# 		word2id[word] = i+1
# 	# word2id['PAD'] = int(nums)
# 	# embedding_weight[int(nums),:]  = np.zeros(int(ran))
# id2word ={v:k for k,v in word2id.items()}
word2id,embedding_weight = WordCover.loadw2c_200dim()
vocab_size = len(word2id)
print(type(embedding_weight))	#'numpy.ndarray'
# print(type(embedding_weight.items()[0]))

def pad_or_truncate(xs, maxlen,wordLevel=True):
    if len(xs) > maxlen:
        if wordLevel:
            xs = xs[len(xs) - maxlen:]
        else:
            xs = xs[0:maxlen]
    elif len(xs) < maxlen:
        if wordLevel:
            xs = ["PAD"] * (maxlen - len(xs)) + xs
        else:
            xs = xs+[]*(maxlen-len(xs))
    return xs
def generate_arrays_from_memory(data_train, labels_train, batch_size):
    x = data_train
    y=labels_train
    ylen=len(y)
    loopcount=ylen // batch_size
    while True:
        i = np.random.randint(0,loopcount)
        yield x[i*batch_size:(i+1)*batch_size],y[i*batch_size:(i+1)*batch_size]

def datagen(X,Y,batch_size=BATCH_SIZE):
    while True:
        num_recs = X.shape[0]
        indices = np.random.permutation(np.arange(num_recs))
        num_batches = num_recs//batch_size

        for bid in range(num_batches):
            batch_ids = indices[bid*batch_size:(bid+1)*batch_size]
            # xbatfrom sklearn.metrics import accuracy_score, confusion_matrixch = np.zeros((batch_size,MAX_SENTS,MAX_WORDS))
            xbatch = X[batch_ids,:]
            ybatch = Y[batch_ids,:]
            yield xbatch,ybatch
def pady(y, maxlen=MAX_SENTS):
    if len(y)>maxlen:
        y = y[0:maxlen]
        y[-1] = np.array([1])
    else:
        y = y + [0]*(maxlen-len(y))
    return y




# score, acc = model.evaluate(x_test, y_test,batch_size=BATCH_SIZE)
#
#
#
# print('Test score:', score)
# print('Test accuracy:', acc)

if __name__ == '__main__':
    model_file = './e2e_set/topic_segmentation_e2e.h5'
    eval(model_file)