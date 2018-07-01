__author__ = 'Administrator'
#coding=utf-8
import numpy as np

import pickle
import gensim
def load_embeding(w2vfilepath='./data/glove.6B.300d.txt',dim=300):
    word2id={"PAD": 0}
    with  open(w2vfilepath,'r',encoding='utf-8') as f:
        lines = f.readlines()
        embedding_weight = np.zeros((len(lines)+2, dim))
        for line in lines:

            l = line.strip().split(' ')
            word, vec = l[0], l[1:]
            vec = list(map(float, vec))
            embedding_weight[len(word2id), :] = vec
            word2id[word] = len(word2id)+1
        embedding_weight[len(word2id),:]  = embedding_weight.mean(axis=0)
        word2id['UNK'] = len(embedding_weight)

    print(embedding_weight.shape,len(word2id))  #(400002, 300)
    # id2word ={v:k for k,v in word2id.items()}
    return word2id,embedding_weight

def batch_iter(x_data,y_data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(x_data)
    y_data = np.array(y_data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
            shuffled_y_data = y_data[shuffle_indices]
        else:
            shuffled_data = data
            shuffled_y_data = y_data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index],shuffled_y_data[start_index:end_index]


def load_data():
    x,y = list(),list()
    documents = pickle.load(open("./data/documents_has_stopwords.pckl", 'rb'))
    documents_bnd = pickle.load(open("./data/documents_bnd.pckl", 'rb'))
    for key,val in documents.items():
        x.append(val)
        y.append(documents_bnd[key])

    return np.array(x),np.array(y)  #(2686,) (2686,)

def pady(y, maxlen=60):
    if len(y)>maxlen:
        y = y[0:maxlen]
        y[-1] = np.array([1])
    else:
        y = y + [0]*(maxlen-len(y))
    return y
def pad_or_truncate(xs, maxlen,wordLevel=True):
    if len(xs) > maxlen:
            xs = xs[len(xs) - maxlen:]
    elif len(xs) < maxlen:
        if wordLevel:
            xs = ["PAD"] * (maxlen - len(xs)) + xs
        else:
            xs = xs+[]*(maxlen-len(xs))
    return xs
def data_tranform(x,y,word2id,MAX_SENTS=60,MAX_WORDS=50):
    y = list(map(pady,y))
    x,y = np.array(x),np.array(y)

    print(y.shape)
    print(type(y))

    # for yy in y:
    # y_binary = to_categorical(y)
    # y_binary = np.reshape(y_binary,(7000,45,2))
    y_binary = np.reshape(y,(len(y),MAX_SENTS,1))
    # print(y_binary[-2:])
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

    return X,y_binary

if __name__ == '__main__':
    x,y = load_data()
    print(x.shape,y.shape)
    # load_embeding()

    pass

