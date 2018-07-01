__author__ = 'Administrator'
from TDTReader import TDTCorpus
from ChoiCorpus import ChoiCorpus
from TDTReader import Segment
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.stem.porter import *
from gensim.models import word2vec
import logging
import numpy
import math
import itertools
import bisect
import sys
from dynamic_seg import segment_dp
from dynamic_seg import  get_best_segmentation_dp,get_defined_segmentation_dp
import pickle
import string

#########
#HELPERS#
#########
def clean_corpus(corpus):

    del corpus.word_ids

    num_samples = 0
    for key in corpus.text_corpus_bnds.keys():
        if key not in corpus.dev_set or num_samples >= 50:
            corpus.text_corpus.pop(key,None)
            corpus.text_corpus_bnds.pop(key,None)
            corpus.doc_boundaries.pop(key,None)
            corpus.sent_boundaries.pop(key,None)
            corpus.char_boundaries.pop(key,None)
        else:
            num_samples += 1
    return corpus

def get_corpus_vocab(corpus):

    vocab = dict()
    for key, doc in corpus.items():
        for sent in doc:
            for word in sent:
                vocab[word] = ""
    return vocab

def clean_oov_embeddings_matrix(embeddings, vocab):
    for key, val in embeddings.items():
        if key not in vocab:
            del embeddings[key]

    return embeddings

def load_embedding_vectors(path):
    embeddings = dict()
    with open(path) as f:
        for line in f.readlines():
            line = line.split()
            word = line.pop(0)
            line = [float(x) for x in line]

            embeddings[word] = numpy.array(line)

    print('loaded embeddings with dim: '+str(len(embeddings[embeddings.keys()[0]])))
    print('vocab size: '+ str(len(embeddings)))
    return embeddings

def load_embedding_vectors_with_vocab(path,vocab):
    embeddings = dict()
    with open(path,'rb') as f:
        for line in f:
            line = line.split()
            word = line.pop(0)
            if word not in vocab:
                continue

            line = [float(x) for x in line]

            embeddings[word] = numpy.array(line)

    # print('loaded embeddings with dim: '+str(len(embeddings[embeddings.keys()[0]])))
    print('vocab size: '+ str(len(embeddings)))
    return embeddings

def count_oov_words(embeddings, corpus):
    oov_words = 0
    stem_table = dict()
    stemmer = PorterStemmer()
    for key,val in corpus.items():
        for line in val:
            for word in line:
                if word in stem_table:
                    continue
                stem_table[word] = stemmer.stem(word)
                if word not in embeddings:
                    if stem_table[word] not in embeddings:
                        oov_words += 1

    return oov_words,stem_table


def preprocess_text(text,removeStop=True):
    stopwordlist = stopwords.words('english')

    text = text.replace('\'',' ')
    # text = text.translate(None,string.punctuation)
    text = text.translate(str.maketrans('','',string.punctuation))
    text = text.lower()

    text = text.split()
    if removeStop:
        filtered_text = [word for word in text if word not in stopwordlist]
        return filtered_text
    return text

def get_sentence_vec_sum(sentence,embeddings):
    dim = embeddings[embeddings.keys()[0]].shape[0]
    sen_vec = numpy.zeros(dim)
    for word in sentence:
        if word in embeddings:
            word_vec = embeddings[word]
            sen_vec = numpy.add(sen_vec,word_vec)

    return sen_vec

def get_sentence_vec(sentence,embeddings):
    sen_vec = []
    for word in sentence:
        if word in embeddings:
            word_vec = embeddings[word]
            sen_vec.append(word_vec)
        #else:
        #    word_vec = [-1]*len(embeddings[embeddings.keys()[0]])
        #    sen_vec.append(word_vec)

    return sen_vec

def convert_docs_2_vec_representations(documents,embeddings):
    vec_representation = dict()
    for key, sentences in documents.items():
        vec_document = []
        for sentence in sentences:
            vec_document.append(get_sentence_vec(sentence,embeddings))

        vec_representation[key] = vec_document

    return vec_representation

def get_segment_embeddings(document, start, end, embeddings):
    i = start

    segment_embeddings = []
    while i <= end:
        for word in document[i]:
            if word in embeddings:
                segment_embeddings.append(embeddings[word])
        i+=1

    return segment_embeddings

############
#SEGMENTERS#
############




############
#EVALUATION#
############
def convert_bounds_2_labels(boundary_seg):
    label = 0

    label_seg = []

    idx = 0
    while idx < len(boundary_seg):
        label_seg.append(label)
        if boundary_seg[idx] == 1:
            label += 1

        idx+=1

    return label_seg

def convert_seg_2_labels(boundary_seg):
    label = 0

    label_seg = []

    idx = 1
    while idx < len(boundary_seg):
        if idx == 1:
            seg_size = boundary_seg[idx] - boundary_seg[idx-1] + 1
        else:
            seg_size = boundary_seg[idx] - boundary_seg[idx-1]
        arr = [label]*seg_size
        label_seg += arr

        label += 1

        idx+=1

    return label_seg

def calculate_pk(ref,hyp):
    num_segs = ref[-1] + 1
    num_elems = len(ref)

    p_k = 0
    k = int((0.5*num_elems/num_segs) - 1 )
    if k == 0:
        k = 2

    for i in range(0,num_elems - k + 1):
        delta_ref = (ref[i] is ref[i+k - 1])
        delta_hyp = (hyp[i] is hyp[i+k - 1])

        if delta_ref != delta_hyp:
            p_k += 1
    p_k = p_k / float(num_elems - k + 1)

    return p_k

def main_choi_corpus():
    corpus = ChoiCorpus()
    documents = dict()
    refs = dict()
    p_k = []
    for s in range(0,50):
        st = str(s)
        document = []
        sen, bnds = corpus.read_document('./ChoiDataset/9-11/'+st+'.ref')
        text = ' '.join(sen)
        text = text.split('<bnd>')
        del text[-1]
        for sent in text:
            sent = preprocess_text(sent)
            document.append(sent)


        documents[st] = document
        refs[st] = bnds

    vocab = get_corpus_vocab(documents)
    embeddings = load_embedding_vectors_with_vocab('/home/mohamed/Desktop/scripts/tdt_reader/glove.840B.300d.txt',
                                                       vocab)
    doc_vec_repr = convert_docs_2_vec_representations(documents,embeddings)
    del documents
    del embeddings

    max_length = 0


    for key, doc in doc_vec_repr.items():
        length = len(doc)

        if length > max_length:
            max_length = length

        score_list, segmentation_list = segment_dp(doc)

        segmentation = get_defined_segmentation_dp(segmentation_list,9)

        ref = convert_bounds_2_labels(refs[key])
        hyp = convert_seg_2_labels(segmentation)

        p_k.append(calculate_pk(ref,hyp))

    print(p_k)
    print(sum(p_k)/float(len(p_k)))

def main():
    corpus = pickle.load(open("corpus_annotated_data_nltk_boundaries.pckl",'rb'))
    #corpus = clean_corpus(corpus)
    #print(len(corpus.text_corpus_bnds.keys())
    #pickle.dump(corpus,open("corpus_dev_50samples_data_dnnhmm_nltk_boundaries.pckl",'w'))


    documents = dict()

    print('loading sentences...')
    for key in corpus.text_corpus_bnds.keys():
        key = key.strip()
        val = corpus.text_corpus_bnds[key]
        document = []


        text = ' '.join(val)
        text = text.split('<bnd>')

        for sent in text:
            sent = preprocess_text(sent)
            document.append(sent)
        documents[key] = document

    print('converting sentences to vector representations...')
    vocab = get_corpus_vocab(documents)
    embeddings = load_embedding_vectors_with_vocab('./data/glove.6B.300d.txt',
                                                   vocab)

    #embeddings = clean_oov_embeddings_matrix(embeddings, documents)

    doc_vec_repr = convert_docs_2_vec_representations(documents,embeddings)
    del documents
    del embeddings

    print('calculating score for test data ...')

    num_segs_est = []
    num_segs_ref = []

    max_length = 0


    for key, doc in doc_vec_repr.items():
        length = len(doc)
        if length > max_length:
            max_length = length


    progress = 0
    total = len(doc_vec_repr.keys())
    p_k_values = []
    for key, doc in doc_vec_repr.items():
        #error_diff, segmentations = segment_greedy(doc)
        score_list, segmentation_list = segment_dp(doc,1,10,1000)

        segmentation = get_best_segmentation_dp(segmentation_list,score_list, doc, max_length)
        progress +=1


        ref = convert_bounds_2_labels(corpus.sent_boundaries[key])
        hyp = convert_seg_2_labels(segmentation)

        p_k_values.append(calculate_pk(ref,hyp))

        sys.stdout.write("\r{0}".format((float(progress)/total)*100))
        sys.stdout.flush()

        print(p_k_values[-1])


    print(numpy.sum(p_k_values)/float(len(p_k_values)))


    for r in num_segs_est:
        print(r)

    print("   ")

    for r in num_segs_ref:
        print(r)



    #print('training word2vec model...')
    #logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    #model = word2vec.Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
    #model.save_word2vec_format('trained_word2vec_test.txt', binary=False)

    #oov,stem_table = count_oov_words(embeddings, documents)
    #print('oov: '+str(oov*100/float(len(stem_table)))+'%')

if __name__ == "__main__":
    main()