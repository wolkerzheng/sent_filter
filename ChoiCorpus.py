__author__ = 'Administrator'
class ChoiCorpus:
    def __init__(self):
        self.text_corpus_bnds = dict()
        self.sent_boundaries = dict()
    def read_document(self,doc_path):
        boundaries = []
        sentences = []
        i = 0
        with open(doc_path,'r') as file:
            for line in file.readlines():
                if line.startswith("=="):
                    if i == 0:
                        continue
                    boundaries[-1] = 1
                else:
                    line = line.split()
                    line[-1] = line[-1]+'<bnd>'
                    sentences = sentences + line[:]
                    boundaries.append(0)
                i+=1

        return sentences, boundaries