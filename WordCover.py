import gensim
import numpy as np
from chineseDataHelp import *
import sys
import pickle
import jieba
import TiebaDataUtil
def loadWord2Vec(w2vfilepath='./embedding/w2v.vector'):

    # model = gensim.models.Word2Vec.load(w2vfilepath)
    word2id={"PAD": 0}
    with  open(w2vfilepath,'rb') as f:
        nums, ran = f.readline().strip().split()
        print(nums)
        embedding_weight = np.zeros((int(nums)+2, int(ran)))	#(240000+,100)
        # word_dic = dict()
        for i in range(int(nums)):
            line = f.readline().strip().split()
            word, vec = line[0], line[1:]
            vec = list(map(float, vec))
            embedding_weight[i+1, :] = vec
            word2id[word] = i+1
        word2id['<unk>'] = int(nums)+1
        embedding_weight[int(nums)+1,:]  = np.zeros(int(ran))
    print(embedding_weight.shape)
    id2word ={v:k for k,v in word2id.items()}
    return word2id,embedding_weight
    # print type(model)
def load_wiki_tieba():
    word2id={"PAD": 0}
    with  open('./embedding/wiki.zh.text.vector','rb') as f:
        nums, ran = f.readline().strip().split()
        embedding_weight = np.zeros((int(nums)+1, int(ran)))	#(125605,200)

        # word_dic = dict()
        for i in range(int(nums)):
            line = f.readline().strip().split()
            word, vec = line[0], line[1:]
            vec = list(map(float, vec))
            embedding_weight[i+1, :] = vec
            word2id[word] = i+1
    # word2id['PAD'] = int(nums)
    # embedding_weight[int(nums),:]  = np.zeros(int(ran))
    id2word ={v:k for k,v in word2id.items()}
    return word2id,embedding_weight
def loadw2c_200dim(filename = './embedding/tieba_word2vector_model_200'):

    model = gensim.models.Word2Vec.load(filename)
    word_vectors = model.wv
    print(word_vectors,type(word_vectors))

    embedding_weight = np.zeros((245099+2, 200))	#(240000+,100)
    word2id={"PAD": 0}
    for word in (model.wv.vocab):

        word2id[word] = len(word2id)
        embedding_weight[len(word2id)-1] = np.array(model[word])

    embedding_weight[len(word2id),:]  = embedding_weight.mean(axis=0)
    word2id['<unk>'] = len(word2id)

    print(embedding_weight.shape)

    return word2id,embedding_weight



def main():
    # corpus = TDTCorpus()
    print("loading corpus...")
    corpus = pickle.load(open("chinese_corpus_annotated_data_nltk_boundaries.pckl",'rb'))
    story_nums = 0
    total_sents = 0
    total_storys = 0
    WKNums = 0
    TBNums = 0
    wk,tb= set(),set()
    WKWord2id = load_wiki_tieba()[0]
    TBword2id = loadWord2Vec()[0]
    total_words = 0
    allwordset = set()
    for k in corpus.text_sent_corpus.keys():

        total_storys+=len(corpus.doc_boundaries[k])
        sents = corpus.text_sent_corpus[k]
        bnds = corpus.sent_boundaries[k]
        total_sents += len(sents)
        story_nums += sum(bnds)
        for sent in sents:

            words = list(jieba.cut(sent))
            # allwordset.union(set(words))
            total_words += len(words)
            for w in words:
                allwordset.add(w)
                if w.encode('utf-8') in WKWord2id.keys():
                    WKNums+=1
                    wk.add(w)
                if w.encode('utf-8') in TBword2id.keys():
                    TBNums+=1
                    tb.add(w)
                # if w not in TBword2id.keys():
                #     print(w)

    print('total stroy nums',total_storys,story_nums)
    print('total sents nums:',total_sents)
    print('total words nums:',total_words)
    print('Wiki Covers:',WKNums,WKNums/total_words)
    print('::::',len(wk)/125605)
    print('tieba Covers:',TBNums,TBNums/total_words)
    print('::::',len(tb)/245099)
    print('wk is :',len(wk)/len(allwordset))
    print('tb is :',len(tb)/len(allwordset))

def countW2V():
    testdata,label = TiebaDataUtil.getHeader()
    WKNums = 0
    TBNums = 0
    wk,tb= set(),set()
    WKWord2id = load_wiki_tieba()[0]
    TBword2id = loadWord2Vec()[0]
    total_words = 0
    allwordset = set()
    for sents in testdata:
        for sent in sents:
            total_words += len(sent)
            for w in sent:
                allwordset.add(w)
                if w.encode('utf-8') in WKWord2id.keys():
                    WKNums+=1
                    wk.add(w)
                if w.encode('utf-8') in TBword2id.keys():
                    TBNums+=1
                    tb.add(w)
                pass

    print('total words nums:',total_words)
    print('Wiki Covers:',WKNums,WKNums/total_words)
    print('::::',len(wk)/125605)
    print('tieba Covers:',TBNums,TBNums/total_words)
    print('::::',len(tb)/245099)
    print('wk is :',len(wk)/len(allwordset))
    print('tb is :',len(tb)/len(allwordset))
if __name__ == '__main__':

    # w2vfilepath = './embedding/w2v_model/model/w2v.vector'
    #
    # loadWord2Vec(w2vfilepath)
    # loadw2c_200dim()
    # main()
    countW2V()