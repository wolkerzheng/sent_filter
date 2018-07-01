__author__ = 'Administrator'
from TDTReader import TDTCorpus
from ChoiCorpus import ChoiCorpus
from TDTReader import Segment
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.stem.porter import *
from gensim.models import word2vec
from CorpusFeatures import CorpusFeatures
import logging
import numpy
import math
import itertools
import bisect
import sys
from scipy import spatial

import pickle
import string

#########
#HELPERS#
#########

def train_corpus_features(path):
    corpus_feats = CorpusFeatures()

    documents, sent_bnds = load_corpus_sentences(path)

    corpus_feats.init_idf(documents, sent_bnds)

    pickle.dump(corpus_feats,open("corpus_train_feats_idf.pckl",'w'))

    return corpus_feats

#Loads corpus into list of sentences#
#Preprocessing is applied to each sentence#
#returns: list of sentences, sentence bit boundaries#
def load_corpus_sentences(path):
    corpus = pickle.load(open(path,'r'))
    documents = dict()

    #key = corpus.text_corpus_bnds.keys()[5]

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
        del document[-1]
        documents[key] = document
    print('finished loading sentences')
    return documents, corpus.sent_boundaries

def convert_sentences_2_word_windows(documents, sent_boundaries, window):
    word_bnds = dict()
    documents_word = dict()
    for key, doc in documents.items():
        bnds = sent_boundaries[key]
        documents_word[key] = []
        word_bnds[key] = []
        word_window = []
        for sent, bnd in zip(doc, bnds):
            for id, word in enumerate(sent):
                if len(word_window) < window:
                    word_window.append(word)
                else:
                    documents_word[key].append(word_window[:])
                    word_window = [word]
                word_bnds[key].append(0)
            if bnd == 1:
                word_bnds[key][-1] = 1
        if len(word_window) > 0:
            documents_word[key].append(word_window[:])

    return documents_word, word_bnds

def load_corpus_as_word_window_vec_rep(path, train_feats, weighting, window=5):
    documents, sent_bnds = load_corpus_sentences(path)


    doc_window, word_bnds = convert_sentences_2_word_windows(documents, sent_bnds, window)

    #Test Cases#
    #print(documents["19980304_1600_1630_CNN_HDL"][-10:]
    #print(sent_bnds["19980304_1600_1630_CNN_HDL"][-10:]
    #print(doc_window["19980304_1600_1630_CNN_HDL"][-30:]
    #print(word_bnds["19980304_1600_1630_CNN_HDL"][-30:]

    #vocab = get_corpus_vocab(documents)
    embeddings = load_embedding_vectors_with_vocab(
        '/home/mohamed/Desktop/scripts/tdt_reader/glove.6B.50d.txt',
        train_feats.vocab)

    doc_vec_repr = convert_docs_2_vec_representations(doc_window,embeddings,train_feats,weighting)
    del documents
    del embeddings


    return doc_vec_repr, word_bnds

def load_corpus_as_sent_vec_rep(path, train_feats, weighting="uniform"):

    documents, sent_boundaries = load_corpus_sentences(path)

    print('converting sentences to vector representations...')
    #vocab = get_corpus_vocab(documents)

    embeddings = load_embedding_vectors_with_vocab('/home/mohamed/Desktop/scripts/tdt_reader/glove.6B.50d.txt',
                                                   train_feats.vocab)

    #embeddings = clean_oov_embeddings_matrix(embeddings, documents)

    doc_vec_repr = convert_docs_2_vec_representations(documents,embeddings,train_feats,weighting)

    del documents
    del embeddings

    return doc_vec_repr, sent_boundaries

def clean_corpus(corpus):
    for i in range(len(corpus.train_set)):
        key = corpus.train_set[i].strip()
        corpus.train_set[i] = key

    num_samples = 0
    for key in corpus.text_corpus_bnds.keys():
        if key not in corpus.train_set:
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
    with open(path) as f:
        for line in f:
            line = line.split()
            word = line.pop(0)
            if word not in vocab:
                continue

            line = [float(x) for x in line]

            embeddings[word] = numpy.array(line)

    print('loaded embeddings with dim: '+str(len(embeddings[embeddings.keys()[0]])))
    print('embedding vocab size: '+ str(len(embeddings)))
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

def get_sentence_vec(sentence,embeddings, train_feats, weighting):
    sen_vec = []
    for word in sentence:
        weight = 1

        if weighting == "log-idf":
            if word in train_feats.idf_feats:
                weight = train_feats.idf_feats[word]

        if word in embeddings:
            word_vec = weight*embeddings[word]
            sen_vec.append(word_vec)
        else:
            word_vec = [0]*len(embeddings[embeddings.keys()[0]])
            sen_vec.append(word_vec)
        #else:
        #    word_vec = [-1]*len(embeddings[embeddings.keys()[0]])
        #    sen_vec.append(word_vec)

    return sen_vec

def convert_docs_2_vec_representations(documents,embeddings, train_feats, weighting):
    vec_representation = dict()
    for key, sentences in documents.items():
        vec_document = []
        for sentence in sentences:
            vec_document.append(get_sentence_vec(sentence,embeddings, train_feats, weighting))

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

###################
#SCORING FUNCTIONS#
###################
def get_content_vector(segment_embeddings):
    sum_w_k = numpy.sum(segment_embeddings, axis=0)
    #sum_w_k = [sum(i) for i in zip(*segment_embeddings)]
    dim = len(segment_embeddings[0])
    weight = 1/float(math.sqrt(dim)) #1/sqrt(D)

    i = 0
    while i < dim:
        if sum_w_k[i] >= 0:
            sum_w_k[i] = weight
        else:
            sum_w_k[i] = weight * -1
        i+=1

    return sum_w_k

def construct_cum_sum(document):
    cum_sum = []
    for sentence in document:
        sen_sum = numpy.sum(sentence,axis=0)
        cum_sum.append(sen_sum)

    cum_sum = numpy.cumsum(cum_sum,0)
    return cum_sum

def create_content_vector(document_vec):
    for key, doc in document_vec.items():
        cv_doc = []
        for sen in doc:
            cv_sen = []
            for sum_w_k in sen:         #sign(sum(w_k))*1/sqrt(D)
                if sum_w_k >= 0:
                    cv_sen.append(1/float(math.sqrt(len(sen))))
                else:
                    cv_sen.append(-1/float(math.sqrt(len(sen))))
            cv_sen = numpy.array(cv_sen)
            cv_doc.append(cv_sen)
        del document_vec[key]
        document_vec[key] = cv_doc

    return document_vec

def scoring_function_cvs(document,start,end):

    dim = len(document[0])
    weight = 1/float(math.sqrt(dim)) #1/sqrt(D)
    if start != 0:
        sum_w_k = document[end] - document[start - 1]
    else:
        sum_w_k = document[end]

    signs = numpy.sign(sum_w_k)

    dot_prod = weight * numpy.dot(sum_w_k,signs)
    #content_vector = get_content_vector(segment_embeddings)


    #segment_embeddings = [numpy.dot(wordvec,content_vector) for wordvec in segment_embeddings]

    #score = numpy.sum(segment_embeddings)

    return numpy.sum(dot_prod)#/float(len(segment_embeddings)) #normalization by segment length

def scoring_function_cvs_old(document,start,end):
    segment_embeddings = list(itertools.chain(*document[start:end+1]))



    if len(segment_embeddings) == 0:
        return 0.0

    dim = len(segment_embeddings[0])
    weight = 1/float(math.sqrt(dim)) #1/sqrt(D)
    sum_w_k = numpy.sum(segment_embeddings, axis = 0)
    signs = numpy.sign(sum_w_k)

    dot_prod = weight * numpy.dot(sum_w_k,signs)
    #content_vector = get_content_vector(segment_embeddings)


    #segment_embeddings = [numpy.dot(wordvec,content_vector) for wordvec in segment_embeddings]

    #score = numpy.sum(segment_embeddings)

    return numpy.sum(dot_prod)#/float(len(segment_embeddings)) #normalization by segment length

def scoring_function_sum(document,start,end):
    segment_embeddings = list(itertools.chain(*document[start:end+1]))

    if len(segment_embeddings) == 0:
        return 0
    score = numpy.sum(numpy.sum(segment_embeddings))
    return score

############
#SEGMENTERS#
############

#####WordEmbedding DynamicProgramming#####

def get_defined_segmentation_dp(segmentation_list, k):
    segmentation = [0,len(segmentation_list)-1]

    curr_j = len(segmentation_list)-1
    while k > 0:
        best_l = segmentation_list[curr_j][k]
        bisect.insort_left(segmentation,best_l)
        curr_j = best_l
        k -= 1
    return segmentation

def get_best_segmentation_dp(segmentation_list, score_list,doc, max_length):
    K = 50
    error_diff = []
    for k in range(1,K):
        error_diff.append(score_list[-1][k] - score_list[-1][k-1])

    idx = get_best_segmentation_greedy(error_diff,doc,max_length)
    idx += 1

    segmentation = [0,len(score_list)-1]

    k = idx
    curr_j = len(score_list)-1
    while k > 0:
        best_l = segmentation_list[curr_j][k]
        bisect.insort_left(segmentation,best_l)
        curr_j = best_l
        k -= 1
    return segmentation

def segment_dp(document, window=5, min_seg=3, max_seg=20):
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
    #print('\n'.join(['%i' % (error_diff[n]) for n in xrange(len(error_diff))])
    #print("----------"
    window = 5
    i = window/2
    for i in range(window/2,len(error_diff) - (window/2)):
        arr = numpy.array(error_diff[i-(window/2):i+1+(window/2)])
        if numpy.std(arr) < 7.0 * len(doc)/max_length:
            return i

    return len(error_diff) - (window/2)

#####WordEmbedding TextTiling#####

#Calculates a vector representation for each window of sentences (average sum)
def _get_block_centroids(document, window, dim):
    if len(document)%window != 0:
        block_centroids = numpy.zeros(((len(document)/window) + 1,dim))
    else:
        block_centroids = numpy.zeros(((len(document)/window),dim))
    blocks = 0

    #GET WINDOW CENTROIDS#
    for i in range(0,len(document),window):
        window_centroid = numpy.zeros(dim)
        word_count = 0
        for k in range(window):
            if i < len(document):
                sen_sum = numpy.sum(document[i], axis=0)
                window_centroid = numpy.add(window_centroid, sen_sum)
                word_count += len(document[i])


        if word_count is not 0:
            window_centroid /= word_count
        else:
            window_centroid = numpy.zeros(dim)

        block_centroids[blocks] = window_centroid
        blocks += 1

    return block_centroids

#Calculates the distance between adjacent centroids (euclidean)#
def _calculate_adj_blocks_distance(block_centroids):
    distance = []
    for i in range(len(block_centroids) - 1):
        val = numpy.linalg.norm(block_centroids[i+1] - block_centroids[i])
        val = 1/(1+val)
        distance.append(val)

    #distance.append(val)

    return distance

#Calculates the distance between adjacent centroids (cosine)#
def _calculate_adj_blocks_cosine_sim(block_centroids):
    distance = []
    for i in range(len(block_centroids) - 1):
        non_zeros_blockA = numpy.count_nonzero(block_centroids[i])
        non_zeros_blockB = numpy.count_nonzero(block_centroids[i+1])

        if non_zeros_blockA > 0 and non_zeros_blockB > 0:
            val = 1 - spatial.distance.cosine(block_centroids[i], block_centroids[i+1])
        else:
            val = 0

        distance.append(val)


    #distance.append(val)

    return distance

#Perform laplacian smoothing with a given window on the input values#
def _calculate_laplacian_smoothing(values, window):

    smoothed = []
    #perform laplacian smoothing
    for i in range(0,len(values),1):
        j = i - (window/2)

        window_centroid = 0
        word_count = 0
        for k in range(window):

            if j >= 0 and j < len(values):
                window_centroid += values[j]
                word_count += 1
            j+=1


        if word_count is not 0:
            window_centroid /= word_count
        else:
            window_centroid = 0

        smoothed.append(window_centroid)
    return smoothed

#Calcualte the depth scores given a list of similarity scores#
def _calculate_depth_scores(values, clip):
    depth_list = []

    for idx, val in enumerate(values):
        l_peak = val
        r_peak = val

        for l in range(clip):
            access = idx - l - 1
            if access >= 0:
                if values[access] >= l_peak:
                    l_peak = values[access]
                else:
                    break

        for r in range(clip):
            access = idx + r + 1
            if access < len(values):
                if values[access] >= r_peak:
                    r_peak = values[access]
                else:
                    break

        depth_val = l_peak + r_peak - 2 * val
        depth_list.append(depth_val)

    return depth_list

#Remove depth scores if they occur near each other within a window
def _remove_duplicate_depth_scores(depth_list, cutoff, duplicate_window):

    for idx, val in enumerate(depth_list):
        if val < cutoff:
            continue

        for i in range(1,duplicate_window+1):
            if (idx - i) >= 0:
                if depth_list[idx - i] >= cutoff:
                    if val >= depth_list[idx - i]:
                        depth_list[idx - i] = 0
                    else:
                        depth_list[idx] = 0


    return depth_list

#Constructs a boundary list
def _insert_boundaries(depth_list, doc_len, window, alpha):
    hyp_bnds = numpy.zeros(doc_len)
    hyp_bnds[-1] = 1
    mean = numpy.mean(depth_list)
    std = numpy.std(depth_list)

    cutoff = mean - (alpha*std)

    depth_list = _remove_duplicate_depth_scores(depth_list,cutoff, 2)

    update_counter = 0

    for idx, i in enumerate(depth_list):
        for l in range(window - 1):
            update_counter += 1

        if i >= cutoff:
            hyp_bnds[update_counter] = 1

        update_counter+=1
    return hyp_bnds

def segment_document_distance_blocks(document,
                                     context_window=3,
                                     smoothing_window=3,
                                     cutoff_weight=0.5,
                                     depth_clip=2,
                                     duplicate_window=2):
    dim = len(document[0][0])

    block_centroids = _get_block_centroids(document, context_window, dim)

    distances = _calculate_adj_blocks_cosine_sim(block_centroids)

    distances_smoothed = _calculate_laplacian_smoothing(distances, smoothing_window)

    depth_list = _calculate_depth_scores(distances_smoothed, depth_clip)

    hyp_bnds = _insert_boundaries(depth_list,len(document), context_window,cutoff_weight)

    return hyp_bnds

def segment_document_distance(document):
    window = 1
    dim = len(document[0][0])
    window_centroids = numpy.zeros((len(document),dim))


    #GET WINDOW CENTROIDS#
    for i in range(0,len(document),3):
        j = i - (window/2)

        window_centroid = numpy.zeros(dim)
        word_count = 0
        for k in range(window):

            if j >= 0 and j < len(document):
                sen_sum = numpy.sum(document[j], axis=0)
                window_centroid = numpy.add(window_centroid, sen_sum)
                word_count += len(document[j])
            j+=1


        if word_count is not 0:
            window_centroid /= word_count
        else:
            window_centroid = numpy.zeros(dim)

        window_centroids[i] = window_centroid


    #CALCULATE DISTANCE BETWEEN ADJACENT CENTROIDS#
    for i in range(len(window_centroids) - 1):
        val = numpy.linalg.norm(window_centroids[i+1] - window_centroids[i])
        print(val)
    print(0)

############
#EVALUATION#
############

#convert a segmentation of format [0001000101] to [0000111122]#
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

def convert_labels_2_bounds(labels_seg):
    bnds = []

    idx = 0
    while idx < len(labels_seg):
        if idx == len(labels_seg) - 1 or labels_seg[idx] != labels_seg[idx + 1]:
            bnds.append(1)
        else:
            bnds.append(0)
        idx+=1

    return bnds

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

def convert_window_labels_2_word_labels(labels_seg, doc):
    word_labels = []
    for label, window in zip(labels_seg, doc):
        word_label = [label]*len(window)
        word_labels.extend(word_label)

    return word_labels

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

def main_segment_window(args, doc_vec_repr, bnds):
    context = args[0]
    smoothing = args[1]
    clip_val = args[2]
    cutoff_alpha = args[3]
    duplicate_range = args[4]

    pk_list = []

    for key, doc in doc_vec_repr.items():
        hyp_bnds = segment_document_distance_blocks(doc,
                                                    context_window=context,
                                                    smoothing_window=smoothing,
                                                    depth_clip=clip_val,
                                                    cutoff_weight=cutoff_alpha,
                                                    duplicate_window=duplicate_range)
        ref_bnds = bnds[key]
        print("----------")
        print('\n'.join(['%i' % (ref_bnds[n]) for n in range(len(ref_bnds))]))
        print("**********")
        print('\n'.join(['%i' % (hyp_bnds[n]) for n in range(len(hyp_bnds))]))
        print("----------")
        ref_bnds = convert_bounds_2_labels(ref_bnds)
        hyp_bnds = convert_bounds_2_labels(hyp_bnds)

        pk_list.append(calculate_pk(ref_bnds, hyp_bnds))

    print("%f\t"%(numpy.sum(pk_list)/float(len(pk_list))),)

    #score_list, segmentation_list = segment_dp(doc_vec_repr[key])
    #segmentation = get_best_segmentation_dp(segmentation_list,score_list, doc_vec_repr[key], len(doc_vec_repr[key]))
    #hyp_bnds = convert_seg_2_labels(segmentation)

    #print(calculate_pk(ref_bnds, hyp_bnds)

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

#set window= -1 for sentence segmentation#
def main_segment_dp(doc_vec_repr, bnds, window=-1, min_seg=1, max_seg=float("inf")):

    pk_list = []
    max_length = 0
    for key, doc in doc_vec_repr.items():
        length = len(doc)
        if length > max_length:
            max_length = length

    for key, doc in doc_vec_repr.items():
        #print(key
        score_list, segmentation_list = segment_dp(doc, window, min_seg, max_seg)
        segmentation = get_best_segmentation_dp(segmentation_list,score_list, doc, max_length)
        hyp_bnds = convert_seg_2_labels(segmentation)

        if window > 0:
            hyp_bnds = convert_window_labels_2_word_labels(hyp_bnds, doc)

        ref_bnds = bnds[key]

        #hyps = convert_labels_2_bounds(hyp_bnds)
        #print("----------"
        #print('\n'.join(['%i' % (ref_bnds[n]) for n in xrange(len(ref_bnds))])
        #print("**********"
        #print('\n'.join(['%i' % (hyps[n]) for n in xrange(len(hyps))])
        #print("----------"

        ref_bnds = convert_bounds_2_labels(ref_bnds)
        #print(len(hyp_bnds)
        #print(len(ref_bnds)
        pk = calculate_pk(ref_bnds, hyp_bnds)

        print("single %f "%pk)
        pk_list.append(pk)


    print(numpy.sum(pk_list)/float(len(pk_list)))

def main_segment_greedy(doc_vec_repr, bnds, window):
    pk_list = []
    max_length = 0
    for key, doc in doc_vec_repr.items():
        length = len(doc)
        if length > max_length:
            max_length = length

    for key, doc in doc_vec_repr.items():
        error_diff, segmentations = segment_greedy(doc)
        segmentation_idx = get_best_segmentation_greedy(error_diff,doc,max_length)
        hyp_bnds = convert_seg_2_labels(segmentations[segmentation_idx])

        if window > 0:
            hyp_bnds = convert_window_labels_2_word_labels(hyp_bnds, doc)

        ref_bnds = bnds[key]


        hyps = convert_labels_2_bounds(hyp_bnds)

        print("----------")
        print('\n'.join(['%i' % (ref_bnds[n]) for n in range(len(ref_bnds))]))
        print("**********")
        print('\n'.join(['%i' % (hyps[n]) for n in range(len(hyps))]))
        print("----------")
        ref_bnds = convert_bounds_2_labels(ref_bnds)
        pk = calculate_pk(ref_bnds, hyp_bnds)

        pk_list.append(pk)

    print(numpy.sum(pk_list)/float(len(pk_list)))

def main():
    corpus = pickle.load(open("corpus_annotated_data_nltk_boundaries.pckl",'r'))
    corpus = clean_corpus(corpus)
    print(len(corpus.text_corpus_bnds.keys()))
    pickle.dump(corpus,open("corpus_train_annotated_data_nltk_boundaries.pckl",'w'))

    return

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


    for key, doc in doc_vec_repr.items():
        length = len(doc)
        if length > max_length:
            max_length = length


    progress = 0
    total = len(doc_vec_repr.keys())
    p_k_values = []
    for key, doc in doc_vec_repr.items():
        #error_diff, segmentations = segment_greedy(doc)
        score_list, segmentation_list = segment_dp(doc)

        segmentation = get_best_segmentation_dp(segmentation_list,score_list, doc, max_length)
        progress +=1


        ref = convert_bounds_2_labels(corpus.sent_boundaries[key])
        hyp = convert_seg_2_labels(segmentation)

        p_k_values.append(calculate_pk(ref,hyp))

        #sys.stdout.write("\r{0}".format((float(progress)/total)*100))
        #sys.stdout.flush()

        #print(p_k_values[-1])


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
    #main()
    train_feats = train_corpus_features("corpus_train_annotated_data_nltk_boundaries.pckl")
    # train_feats = pickle.load(open("corpus_train_feats_idf.pckl",'r'))
    window = 1
    doc_vec_repr, bnds = load_corpus_as_word_window_vec_rep("corpus_dev_10samples_annotated_data_nltk_boundaries.pckl",
                                                            train_feats,
                                                            "uniform",
                                                            window)
    #doc_vec_repr, bnds = load_corpus_as_sent_vec_rep("corpus_dev_10samples_annotated_data_nltk_boundaries.pckl", train_feats, "uniform")
    #for cutoff in [0.1,0.25,0.5,0.75,1]:
    #    print("Cutoff (alpha): %f"%cutoff
    #    print("--------------"
    #    for context in [1,3,5,7,9]:
    #        for smoothing in [1,3,5,7]:
    #            args = [context, smoothing,2 , cutoff, 2]
    #            main_segment_window(args, doc_vec_repr, bnds)
    #        print("\n",
    #    print("\n",

    #args = [5, 3, 2, 0.5, 2]
    #main_segment_window(args, doc_vec_repr, bnds)
    main_segment_dp(doc_vec_repr, bnds, window,min_seg=10, max_seg=1000)
    #main_segment_dp(doc_vec_repr, bnds, -1,min_seg=1,max_seg=100)
    #main_segment_greedy(doc_vec_repr, bnds, window)