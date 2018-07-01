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
    for key, doc in corpus.iteritems():
        for sent in doc:
            for word in sent:
                vocab[word] = ""
    return vocab

def clean_oov_embeddings_matrix(embeddings, vocab):
    for key, val in embeddings.iteritems():
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
    with open(path) as f:
        for line in f:
            line = line.split()
            word = line.pop(0)
            if word not in vocab:
                continue

            line = [float(x) for x in line]

            embeddings[word] = numpy.array(line)

    print('loaded embeddings with dim: '+str(len(embeddings[embeddings.keys()[0]])))
    print('vocab size: '+ str(len(embeddings)))
    return embeddings

def count_oov_words(embeddings, corpus):
    oov_words = 0
    stem_table = dict()
    stemmer = PorterStemmer()
    for key,val in corpus.iteritems():
        for line in val:
            for word in line:
                if word in stem_table:
                    continue
                stem_table[word] = stemmer.stem(word)
                if word not in embeddings:
                    if stem_table[word] not in embeddings:
                        oov_words += 1

    return oov_words,stem_table

def preprocess_text(text):
    stopwordlist = stopwords.words('english')

    text = text.replace('\'',' ')
    text = text.translate(None,string.punctuation)
    text = text.lower()
    text = text.split()

    filtered_text = [word for word in text if word not in stopwordlist]

    #stemmer = PorterStemmer()
    #stemmed_text = [stemmer.stem(word) for word in filtered_text]


    return filtered_text

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
    for key, sentences in documents.iteritems():
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

def segment_dp(document, window, min_seg, max_seg):
    i = 0
    j = 0

    cum_sum = construct_cum_sum(document)
    score_list = []
    segmentation_list = []
    for k in range(len(document)):
        l = [None]*len(document)
        s = [None]*len(document)
        score_list.append(l)
        segmentation_list.append(s)

    for k in range(50):
        j = k
        while j < len(document):
            #print(str(j) + ' ' + str(k))
            if k == 0: #initialization
                if j - k + 1 >= min_seg and i - k + 1 <= max_seg:
                    score = scoring_function_cvs(cum_sum,k,j)
                    score_list[j][k] = score

            else:
                max_score = float("-inf")
                best_l = -1
                for l in range(k-1,j):

                    if (score_list[l][k-1] != None) \
                            and (j - l + 1 + 1 >= min_seg) \
                            and (j - l + 1 + 1 <= max_seg):
                        score = score_list[l][k-1] + scoring_function_cvs(cum_sum,l+1,j)

                    if max_score < score:
                        max_score = score
                        best_l = l

                if max_score != float("-inf"):
                    score_list[j][k] = max_score
                    segmentation_list[j][k] = best_l
            j+=1
    return score_list, segmentation_list


        #print('finished segmenting with '+str(k+1) +' segments')

def segment_greedy(document):
    error_diff = []
    segmentations = [0,len(document) - 1]
    possible_segmentations = []

    segmentation_scores = dict()
    K = 50

    prev_score = 0

    for k in range(K):

        min_score = float(0)
        best_seg = -1

        for i in range(1,len(document)-2):
            if i in segmentations:
                continue

            possible_seg = segmentations[:]
            bisect.insort_left(possible_seg,i)

            cum_sum = construct_cum_sum(document)
            possible_seg_score = calculate_segmentation_score(cum_sum, possible_seg,segmentation_scores)
            if possible_seg_score > min_score:
                min_score = possible_seg_score
                best_seg = i

        error_diff.append(min_score - prev_score)
        prev_score = min_score
        bisect.insort_left(segmentations, best_seg)
        seg = segmentations[:]
        possible_segmentations.append(seg)

    return error_diff, possible_segmentations
def calculate_segmentation_score(document, segmentation, segmentation_scores):
    sum_score = 0
    for i in range(len(segmentation)-1):
        a = -1
        b = -1

        if i != 0:
            a = segmentation[i] + 1
            b = segmentation[i+1]
        else:
            a = segmentation[i]
            b = segmentation[i+1]

        if a in segmentation_scores:
            if b in segmentation_scores[a]:
                sum_score += segmentation_scores[a][b]
            else:
                segmentation_scores[a][b] = scoring_function_cvs(document,a,b)
                sum_score += segmentation_scores[a][b]
        else:
            segmentation_scores[a] = dict()
            segmentation_scores[a][b] = scoring_function_cvs(document,a,b)
            sum_score += segmentation_scores[a][b]

    return sum_score

def get_best_segmentation_greedy(error_diff,doc, max_length):
    #best for greedy: window=2, cutoff=5
    #print '\n'.join(['%i' % (error_diff[n]) for n in xrange(len(error_diff))])
    #print "----------"
    window = 5
    i = window/2
    for i in range(window/2,len(error_diff) - (window/2)):
        arr = numpy.array(error_diff[i-(window/2):i+1+(window/2)])
        if numpy.std(arr) < 7.0 * len(doc)/max_length:
            return i

    return len(error_diff) - (window/2)

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


    for key, doc in doc_vec_repr.iteritems():
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
    corpus = pickle.load(open("corpus_test_10samples_data_dnnhmm_nltk_boundaries.pckl",'r'))
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
    embeddings = load_embedding_vectors_with_vocab('/home/mohamed/Desktop/scripts/tdt_reader/glove.840B.300d.txt',
                                                   vocab)

    #embeddings = clean_oov_embeddings_matrix(embeddings, documents)

    doc_vec_repr = convert_docs_2_vec_representations(documents,embeddings)
    del documents
    del embeddings

    print('calculating score for test data ...')

    num_segs_est = []
    num_segs_ref = []

    max_length = 0


    for key, doc in doc_vec_repr.iteritems():
        length = len(doc)
        if length > max_length:
            max_length = length


    progress = 0
    total = len(doc_vec_repr.keys())
    p_k_values = []
    for key, doc in doc_vec_repr.iteritems():
        #error_diff, segmentations = segment_greedy(doc)
        score_list, segmentation_list = segment_dp(doc)

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