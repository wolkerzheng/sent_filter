#coding:utf-8
import os
import re
import pickle
import jieba
import numpy as np
class Segment:
    def __init__(self,id,beg_id,end_id,topic_annot,seg_type):
        self.id = id
        self.beg_id = beg_id
        self.end_id = end_id
        self.seg_type = seg_type
        self.topic_annot = topic_annot


    def __str__(self):
        return "ID: " + self.id + "\nBeg: " + str(self.beg_id) + "\nEnd: " + str(self.end_id) + "\n" +str(self.topic_annot)
    def __repr__(self):
        return "ID: " + self.id + "\nBeg: " + str(self.beg_id) + "\nEnd: " + str(self.end_id) + "\n" +str(self.topic_annot)
class TDTCorpus:
    def __init__(self):
        self.text_corpus = dict()
        self.text_sent_corpus = dict()
        self.text_corpus_bnds = dict()
        self.word_ids = dict()
        self.total_words = 0
        self.total_stories = 0
        self.total_annotated_stories = 0
        self.doc_boundaries = dict()
        self.sent_boundaries = dict()
        self.char_boundaries = dict()
        self.story_topics_table = dict()

        self.train_set = []
        self.test_set = []
        self.dev_set = []

    def read_tkn_sample_files(self, token_path):
        MAX_SAMPLES = 0
        for name in os.listdir(token_path):
            if MAX_SAMPLES >= 1000000:
                break
            if "_MAN"  in name:
                sample_path = os.path.join(token_path, name)
                sample_tokens, token_ids = self._read_tkn_file(sample_path)
                # print('samleo tokens is ')
                # print( sample_tokens )
                # print(token_ids)
                name = name.replace('.tkn', '')
                self.text_corpus[name] = sample_tokens
                self.word_ids[name] = token_ids
                # print(self.word_ids)
                self.total_words += len(sample_tokens)
                MAX_SAMPLES += 1
                # break
        print('tkn man total file is :',MAX_SAMPLES)

    def read_bnd_sample_files(self, boundary_path):
        MAX_SAMPLES = 0

        for name in os.listdir(boundary_path):
            if MAX_SAMPLES >= 1000000:
                break
            if "_MAN"  in name:
                sample_path = os.path.join(boundary_path, name)
                boundaries = self._read_bnd_file(sample_path)
                name = name.replace('.tkn_bnd', '')
                self.doc_boundaries[name] = boundaries
                # print('boundaries is :',boundaries)
                MAX_SAMPLES += 1

        print('bnd files has :',MAX_SAMPLES)

    def _read_bnd_file(self, file_path):
        SEG_IDX = 2
        TYPE_IDX = 4
        BEG_IDX = 6
        END_IDX = 8

        bnd_list = []

        with open(file_path) as file:
            lines = file.readlines()
            header = re.split(' |=|\n', lines[0])
            del lines[0]
            del lines[-1]

            for line in lines:
                line_data = self._parse_bnd_xml_line(line)

                if "Brecid" in line_data.keys():
                    if line_data["docno"] in self.story_topics_table:
                        topic_annot = self.story_topics_table[line_data["docno"]]
                        self.total_annotated_stories += 1
                    else:
                        topic_annot = []
                    self.total_stories += 1
                    new_seg = Segment(line_data["docno"], int(line_data["Brecid"]),
                                      int(line_data["Erecid"]), topic_annot, line_data["doctype"])
                else:
                    new_seg = Segment(line_data["docno"], 0, 0, -1, line_data["doctype"])

                if new_seg.end_id - new_seg.beg_id > 0:
                    bnd_list.append(new_seg)
        return bnd_list
    def _parse_bnd_xml_line(self, line):
        line = re.split(' |=|\n',line.replace('<','').replace('>',''))
        del line[0]
        del line[-1]

        data = dict()
        i = 0

        while i < len(line):
            data[line[i]] = line[i+1]
            i += 2

        return data
    def _read_tkn_file(self, file_path):
        TYPE_IDX = 2
        SAMPLE_ID_IDX = 4
        SOURCE_IDX = 8
        LANG_IDX = 10

        WORD_IDX = 2
        WORD_ID_IDX = 1

        token_list = []
        token_ids = []
        with open(file_path,'r',encoding='GB18030') as file:
            lines = file.readlines()
            # print(lines)
            # print(lines[0])
            header = re.split(' |=|\n', lines[0])
            # print(header)
            del lines[0]
            del lines[-1]

            for line in lines:
                # print(line)
                # print(line)
                line = re.split(' |\n', line)
                token_list.append(line[WORD_IDX])
                token_ids.append(int(line[WORD_ID_IDX].split('=')[1].replace('>', '')))

        return token_list, token_ids

    def construct_sen_boundaries(self):

        for key in self.doc_boundaries.keys():
            bnd_list = self._get_sample_file_bnd_nltk_sent(key)
            self.sent_boundaries[key] = bnd_list

        self.text_corpus.clear()

    def cut_sentence_new(self,words):
        # words = (words).decode('utf8')
        start = 0
        i = 0
        sents = []

        punt_list = '.!?~。！？～'
        punt_list_tmp = '“”——，'
        token = words[start+1]
        for word in words:
            if word in punt_list and token not in punt_list_tmp: #检查标点符号下一个字符是否还是标点
                sents.append(words[start:i+1])
                start = i+1
                i += 1
            else:
                i += 1
                token = list(words[start:i+2]).pop() # 取下一个字符
        if start < len(words):
            sents.append(words[start:])
        return sents

    def _get_sample_file_bnd_nltk_sent(self, key):
        bnd_list = []
        tokens = self.text_corpus[key]
        # print(len(tokens),len(tokens[0]))#13869 1
        # print(tokens)
        ##get sentence boundaries using nltk
        bnds = self.doc_boundaries[key]
        bnds_idx = 0
        wrd_cnt = 1
        doc_words=[]
        sent_bnds=[]
        sentss = []
        for word in tokens:
            if wrd_cnt == bnds[bnds_idx].end_id:
                bnds_idx += 1
                tmpsents = self.cut_sentence_new(''.join(doc_words))

                tmpsents = [sen.strip() for sen in tmpsents]
                sentss.extend(tmpsents)
                for i in range(len(tmpsents)-1):
                    sent_bnds.append(0)
                sent_bnds.append(1)
                doc_words = []

            else:
                doc_words.append(word)
            wrd_cnt += 1
        self.text_sent_corpus[key] = sentss
        return sent_bnds

    def convert_char_bounds_to_binary(self, key, predicted):
        char_bnds = self.char_boundaries[key]
        binary_bnds = []

        i = 0
        for boundary in char_bnds:
            if i < len(predicted) and boundary == predicted[i]:
                binary_bnds.append(1)
                i += 1
            else:
                binary_bnds.append(0)
        return binary_bnds

def main(load_from_raw=True):
    corpus = TDTCorpus()
    if load_from_raw:
        print("reading corpus...")

        print("reading boundary files...")
        corpus.read_bnd_sample_files('./data/tdt2_em_v4_0/tdt2proj/tdt2_em/tkn_bnd')

        print("reading token files...")
        corpus.read_tkn_sample_files('./data/tdt2_em_v4_0/tdt2proj/tdt2_em/tkn')



        print("constructing sentence and char boundary vectors...")
        corpus.construct_sen_boundaries()
        # keys = corpus.text_corpus_bnds
        # print(len(keys))    #911
        print("corpus length: ", len(corpus.text_sent_corpus), " ", len(corpus.word_ids))  # ," ",corpus.text_corpus.keys()
        # print("Boundaries: ", corpus.doc_boundaries)
        # print(len(list(corpus.doc_boundaries.values())[0]))
        print("dividing and validating datasets...")


        print("total size (words): ", corpus.total_words)   #9982868
        print("total stories: ", corpus.total_stories)  #19199
        print("total annotated stories: ", corpus.total_annotated_stories)
        print("saving corpus data...")
        pickle.dump(corpus, open("chinese_corpus_annotated_data_nltk_boundaries.pckl", 'wb'))
        print("done corpus reading...")


    else:

        print("loading corpus...")
        corpus = pickle.load(open("chinese_corpus_annotated_data_nltk_boundaries.pckl",'rb'))
        annotated_words = 0

        for doc_segs in list(corpus.doc_boundaries.values()):
            for segment in doc_segs:
                if segment.topic_annot:
                    annotated_words += (segment.end_id - segment.beg_id)

        print("Corpus total words: %i"%corpus.total_words)
        print("Corpus annotated words: %i"%annotated_words)

def process_tdt2():
    corpus = TDTCorpus()
    corpus = pickle.load(open("chinese_corpus_annotated_data_nltk_boundaries.pckl",'rb'))
    train_data = []
    train_label = []
    for k in corpus.text_sent_corpus.keys():
        sents = corpus.text_sent_corpus[k]
        # sents = corpus.text_sent_corpus[k]
        bnds = corpus.sent_boundaries[k]
        tmp_x = []
        tmp_y = []
        idx = 0
        flag = 0
        for sent in sents:
            words=list(jieba.cut(sent))
            tmp_x.append(words)
            tmp_y.append(np.array([bnds[idx]]))
            flag+=bnds[idx]

            if(flag==3):
                print(tmp_x)
                print(tmp_y)
                train_data.append(tmp_x)
                train_label.append(tmp_y)
                tmp_x = []
                tmp_y = []
                flag = 0
            idx+=1
        train_data.append(tmp_x)
        train_label.append(tmp_y)
    # print(len(train_data),len(train_label))
    # print("saving corpus data...")
    #
    # #python2 use protocal:2
    # pickle.dump(train_data, open("./e2e_set/train_data.pckl", 'wb'), protocol=2)
    # pickle.dump(train_label, open("./e2e_set/train_label.pckl", 'wb'), protocol=2)
    # print("done corpus reading...")

    return train_data,train_label
if __name__ == '__main__':
    # main(False)
    process_tdt2()