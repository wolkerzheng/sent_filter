#encoding=utf8
from keras import backend as K
from keras.layers import Input,Activation
from keras.layers.core import Dense,Dropout,Reshape,Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU,LSTM
from keras.layers.wrappers import TimeDistributed,Bidirectional
from keras.models import Model
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import pickle
# from pre_build_dictionary import word_dictionary
import WordCover
import sys
import TiebaDataUtil
from evalate import f1,precision,recall
"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 100)               0
_________________________________________________________________
embedding_1 (Embedding)      (None, 100, 200)          49020200
_________________________________________________________________
bidirectional_1 (Bidirection (None, 200)               240800
=================================================================
Total params: 49,261,000
Trainable params: 49,261,000
Non-trainable params: 0
_________________________________________________________________
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_2 (InputLayer)         (None, 45, 100)           0
_________________________________________________________________
time_distributed_1 (TimeDist (None, 45, 200)           49261000
_________________________________________________________________
bidirectional_2 (Bidirection (None, 45, 200)           240800
_________________________________________________________________
time_distributed_2 (TimeDist (None, 45, 1)             201
=================================================================
Total params: 49,502,001
Trainable params: 49,502,001
Non-trainable params: 0

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
            xs = xs[len(xs) - maxlen:]
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

def my_to_categorical(y,MAX_SENTS=MAX_SENTS,num_classes=None):
    """Converts a class vector (integers) to binary class matrix.


    # Returns
        A binary matrix representation of the input.
    """
    # y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n,MAX_SENTS, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical
# x,y = gd.generateNegitivedata()
x = pickle.load(open("./e2e_set/testdata.pckl",'rb'))
y = pickle.load(open("./e2e_set/test_label.pckl",'rb'))

y = list(map(pady,y))

print('loaded data...')


x,y = np.array(x),np.array(y)


y_binary = np.reshape(y,(len(y),MAX_SENTS,1))

print(y.shape)
print(type(y))


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

x_tv, x_test, y_tv, y_test = train_test_split(X, y_binary, test_size=0.1)
x_train,x_val,y_train,y_val = train_test_split(x_tv,y_tv,train_size=0.9)

print('Build model...')
print('data is :',len(x),len(y))	#212399

# docid2mat = {}
# ftext = open(DOCSIM_TEXTS, "rb")

max_features = max(max_features,vocab_size)

sent_in = Input(shape=(MAX_WORDS,), dtype="int32")

sent_emb = Embedding(input_dim=max_features,
                       output_dim=WORD_EMBED_SIZE,
                       weights=[embedding_weight],
                        mask_zero=True,trainable=True)(sent_in)

sent_enc = Bidirectional(LSTM(SENT_EMBED_SIZE,return_sequences=False))(sent_emb)

sent_model = Model(inputs=sent_in, outputs=sent_enc)

sent_model.summary()

doc_in_l = Input(shape=(MAX_SENTS, MAX_WORDS), dtype="int32")

doc_emb_l = TimeDistributed(sent_model)(doc_in_l)

doc_enc_l = Bidirectional(LSTM(DOC_EMBED_SIZE,
                              return_sequences=True))(doc_emb_l)

doc_pred = TimeDistributed(Dense(1,activation='sigmoid'))(doc_enc_l)


model = Model(inputs=doc_in_l, outputs=doc_pred)
model.summary()

model.compile(loss="binary_crossentropy", optimizer="adam",
             metrics=[f1,precision,recall])
early_stopping =EarlyStopping(monitor='val_loss', patience=1)

# hist = model.fit_generator(datagen(x_train,y_train,BATCH_SIZE),
#           steps_per_epoch=int(len(x_train)/BATCH_SIZE),
#            epochs=NUM_EPOCHS,
#           callbacks=[early_stopping],
#           validation_data=[x_val,y_val])

hist = model.fit(x_train,y_train,
          batch_size=BATCH_SIZE,
          epochs=NUM_EPOCHS, shuffle=True,
          callbacks=[early_stopping],
          validation_data=[x_val,y_val])
score, f1_val,precisi,recall_val = model.evaluate(x_test, y_test,batch_size=BATCH_SIZE)


print('history', hist)
print('Test score:', score)
print('Test f1:', f1_val)
print('test pre:',precisi)
print('test recall:',recall_val)
model.save('dialog_segmentation_e2e.h5')
