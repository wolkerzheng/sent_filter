#coding=utf-8
__author__ = 'Administrator'
import pickle
from TDTReader import TDTCorpus
from TDTReader import Segment
# corpus = TDTCorpus()
import string
from nltk.corpus import stopwords
import nltk
import numpy as np
import re
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def preprocess_text(text,removeStop=True):
    stopwordlist = stopwords.words('english')

    text = text.replace('\'',' ')
    # text = text.translate(None,string.punctuation)
    text = text.translate(str.maketrans('','',string.punctuation))
    text = text.lower()

    text = text.split()
    if removeStop:
        filtered_text = [clean_str(word) for word in text if word not in stopwordlist]

    else:
        filtered_text = [clean_str(word) for word in text]
    return filtered_text

    #stemmer = PorterStemmer()
    #stemmed_text = [stemmer.stem(word) for word in filtered_text]



def load_data_test():

    print("loading corpus...")
    corpus = pickle.load(open("corpus_annotated_data_nltk_boundaries.pckl",'rb'))
    # annotated_words = 0
    documents = dict()
    documents_bnd = dict()
    print('loading sentences ... ')
    for key,val in corpus.text_corpus_bnds.items():
        key = key.strip()
        document = []
        text = ' '.join(val)
        text = text.replace('<bnd>','\n\n\t')
        textdata = text.strip().split('\n\n\t')
        for sent in textdata:
            sent = preprocess_text(sent)

            document.append(sent)
        documents[key] = document
        # documents_bnd[key] = corpus.sent_boundaries[key]
        print(len(document),np.sum(corpus.sent_boundaries[key]),len(corpus.sent_boundaries[key]),corpus.sent_boundaries[key])
        print('text_corpus_bnds',len(corpus.char_boundaries[key]))
        print('text_corpus_bnds',len(corpus.text_corpus_bnds[key]))
        print('doc_boundaries:',len(corpus.doc_boundaries[key]))
        print('')
    print(len(documents))   #2686
    pickle.dump(documents, open("./data/documents_has_stopwords.pckl", 'wb'))
    pickle.dump(corpus.sent_boundaries, open("./data/documents_bnd.pckl", 'wb'))

    # for doc_segs in list(corpus.doc_boundaries.values()):
    #     print("....",doc_segs)
    #     print('...')
    #     for segment in doc_segs:
    #         print('segment is :',segment)
    #         if segment.topic_annot:
    #             annotated_words += (segment.end_id - segment.beg_id)
    #
    #
    # print("Corpus total words: %i"%corpus.total_words)  #20753432
    # print("Corpus annotated words: %i"%annotated_words) #0

# def load_bnd_file():
#
#     corpus.read_bnd_sample_files('./data/tdt2_em_v4_0/tdt2proj/tdt2_em/tkn_bnd')
#     print(corpus.doc_boundaries)
#     print('sentenced boundaries...')
#     print(corpus.sent_boundaries)

    pass

if __name__ == '__main__':
    load_data_test()