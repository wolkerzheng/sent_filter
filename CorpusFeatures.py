__author__ = 'Administrator'
import numpy
import math

class CorpusFeatures:
    def __init__(self):
        self.idf_feats = dict()
        self.tf_feats = dict()
        self.vocab = dict()

    #Takes a corpus of documents and their boundaries#
    #and constructs idf features and vocab list#
    def init_idf(self, documents, sent_bnds):
        word_occ_in_doc = []

        for key, doc in documents.iteritems():
            bnds = sent_bnds[key]
            doc_occur = dict()
            if len(bnds) != len(doc):
                print("ERR")
            for sent, is_bnd in zip(doc, bnds):
                for word in sent:
                    doc_occur[word] = True
                    self.vocab[word] = True

                if is_bnd == 1:
                    word_occ_in_doc.append(doc_occur)
                    doc_occur = dict()

        print("Vocab size: %i"%len(self.vocab.keys()))
        print("Total segments: %i"%len(word_occ_in_doc))


        N = len(word_occ_in_doc)

        for v_i in self.vocab.keys():
            occ = 0
            for segment in word_occ_in_doc:
                if v_i in segment:
                    occ += 1

            self.idf_feats[v_i] = math.log(N/float(occ))


        print("finished calculating idf values")