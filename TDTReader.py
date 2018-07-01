__author__ = 'Administrator'

import os
import re
import nltk
#from readless.Segmentation import texttiling
from nltk.corpus import brown
from nltk.tokenize import sent_tokenize
import math
import pickle
# import segeval
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random

class ModifiedTextTilingTokenizer(nltk.tokenize.texttiling.TextTilingTokenizer):

    def _mark_paragraph_breaks(self, text):
            """Identifies indented text or line breaks as the beginning of
            paragraphs"""

            MIN_PARAGRAPH = 0
            pattern = re.compile("[ \t\r\f\v]*\n[ \t\r\f\v]*\n[ \t\r\f\v]*")
            matches = pattern.finditer(text)

            last_break = 0
            pbreaks = [0]
            for pb in matches:
                if pb.start()-last_break < MIN_PARAGRAPH:
                    continue
                else:
                    pbreaks.append(pb.start())
                    last_break = pb.start()

            return pbreaks

    def _identify_boundaries(self, depth_scores):
        """Identifies boundaries at the peaks of similarity score
        differences"""

        BLOCK_COMPARISON, VOCABULARY_INTRODUCTION = 0, 1
        LC, HC = 0, 1
        DEFAULT_SMOOTHING = [0]

        boundaries = [0 for x in depth_scores]

        avg = sum(depth_scores)/len(depth_scores)
        stdev = np.std(depth_scores)

        #SB: what is the purpose of this conditional?
        if self.cutoff_policy == LC:
            cutoff = avg-stdev/16.0
        else:
            cutoff = avg-stdev/16.0

        depth_tuples = sorted(zip(depth_scores, range(len(depth_scores))))
        depth_tuples.reverse()
        hp = list(filter(lambda x:x[0]>cutoff, depth_tuples))

        for dt in hp:
            boundaries[dt[1]] = 1
            for dt2 in hp: #undo if there is a boundary close already
                if dt[1] != dt2[1] and abs(dt2[1]-dt[1]) < 4 \
                       and boundaries[dt2[1]] == 1:
                    boundaries[dt[1]] = 0
        return boundaries
    def tokenize(self, text):
        """Return a tokenized copy of *text*, where each "token" represents
        a separate topic."""

        BLOCK_COMPARISON, VOCABULARY_INTRODUCTION = 0, 1
        LC, HC = 0, 1
        DEFAULT_SMOOTHING = [0]

        lowercase_text = text.lower()
        paragraph_breaks = self._mark_paragraph_breaks(text)
        text_length = len(lowercase_text)

        # Tokenization step starts here

        # Remove punctuation
        nopunct_text = ''.join(c for c in lowercase_text
                               if re.match("[a-z\-\' \n\t]", c))
        nopunct_par_breaks = self._mark_paragraph_breaks(nopunct_text)

        tokseqs = self._divide_to_tokensequences(nopunct_text)

        # The morphological stemming step mentioned in the TextTile
        # paper is not implemented.  A comment in the original C
        # implementation states that it offers no benefit to the
        # process. It might be interesting to test the existing
        # stemmers though.
        #words = _stem_words(words)

        # Filter stopwords
        for ts in tokseqs:
            ts.wrdindex_list = [wi for wi in ts.wrdindex_list
                                if wi[0] not in self.stopwords]

        token_table = self._create_token_table(tokseqs, nopunct_par_breaks)
        # End of the Tokenization step

        # Lexical score determination
        if self.similarity_method == BLOCK_COMPARISON:
            gap_scores = self._block_comparison(tokseqs, token_table)
        elif self.similarity_method == VOCABULARY_INTRODUCTION:
            raise NotImplementedError("Vocabulary introduction not implemented")

        if self.smoothing_method == DEFAULT_SMOOTHING:
            smooth_scores = self._smooth_scores(gap_scores)
        # End of Lexical score Determination

        # Boundary identification
        depth_scores = self._depth_scores(smooth_scores)
        depth_scores = self._smooth_scores(depth_scores)

        segment_boundaries = self._identify_boundaries(depth_scores)

        normalized_boundaries = self._normalize_boundaries(text,
                                                           segment_boundaries,
                                                           paragraph_breaks)
        # End of Boundary Identification
        segmented_text = []
        prevb = 0

        for b in normalized_boundaries:
            if b == 0:
                continue
            segmented_text.append(text[prevb:b])
            prevb = b

        if prevb < text_length: # append any text that may be remaining
            segmented_text.append(text[prevb:])

        if not segmented_text:
            segmented_text = [text]

        if self.demo_mode:
            return gap_scores, smooth_scores, depth_scores, segment_boundaries,normalized_boundaries
        return segmented_text


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

    def _read_tkn_file(self, file_path):
        """

        :param file_path:
        :return:token_list:文件中的token, token_ids:文件中的recid
        """


        WORD_IDX    = 2
        WORD_ID_IDX = 1

        token_list = []
        token_ids =  []
        with open(file_path) as file:
            lines = file.readlines()
            header = re.split(' |=|\n', lines[0])
            del lines[0]
            del lines[-1]

            for line in lines:
                line = re.split(' |\n',line)
                token_list.append(line[WORD_IDX])
                token_ids.append(int(line[WORD_ID_IDX].split('=')[1].replace('>','')))


        return token_list, token_ids

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

    def _read_bnd_file(self, file_path):

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
                    new_seg = Segment(line_data["docno"],int(line_data["Brecid"]),
                                  int(line_data["Erecid"]),topic_annot,line_data["doctype"])
                else:
                    new_seg = Segment(line_data["docno"],0, 0,-1, line_data["doctype"])

                if new_seg.end_id - new_seg.beg_id > 0:
                    bnd_list.append(new_seg)
        return bnd_list

    def _fix_inserted_sen_bnds(self,nltk_tokens,tokens,key):
        out_tokens = []
        j = 0
        i = 0
        while i < len(nltk_tokens):
            set_bnd = False
            word = nltk_tokens[i]
            if word.endswith('<bnd>'):
                set_bnd = True
                word = word.replace('<bnd>','')

            if word != tokens[j]:
                while word != tokens[j]:
                    i+=1
                    word =word+ nltk_tokens[i].replace('<bnd>','')

            if set_bnd:
                word = word + '<bnd>'

            out_tokens.append(word)
            i+=1
            j+=1
        return out_tokens

    def _get_sample_file_bnd_nltk_sent(self, key):
        bnd_list = []
        tokens = self.text_corpus[key]
        ##get sentence boundaries using nltk
        sentences = sent_tokenize(' '.join(tokens))
        sentences = [sen.strip() for sen in sentences]
        sentences = '<bnd> '.join(sentences)

        sentences = sentences.strip().split(' ')

        tokens = self._fix_inserted_sen_bnds(sentences,tokens,key)
        ##finished adding sentence boundaries

        bnds = self.doc_boundaries[key]


        bnds_idx = 0
        wrd_cnt = 1
        for word in tokens:
            if wrd_cnt == bnds[bnds_idx].end_id:
                bnds_idx += 1
                bnd_list.append(1)
                if not word.endswith('<bnd>'):
                    word = word+"<bnd>"
                    tokens[wrd_cnt-1] = word
            elif word.endswith("<bnd>"):
                bnd_list.append(0)
            wrd_cnt += 1
        self.text_corpus_bnds[key] = tokens
        return bnd_list

    def _get_sample_file_bnd(self, key):
        bnd_list = []
        tokens = self.text_corpus[key]
        bnds = self.doc_boundaries[key]


        bnds_idx = 0
        wrd_cnt = 1
        for word in tokens:
            if wrd_cnt == bnds[bnds_idx].end_id:
                bnds_idx += 1
                bnd_list.append(1)
                word = word+"<bnd>"
                tokens[wrd_cnt-1] = word
            elif word.endswith("."):
                word = word+"<bnd>"
                bnd_list.append(0)
                tokens[wrd_cnt - 1] = word
            wrd_cnt += 1
        self.text_corpus_bnds[key] = tokens
        return bnd_list

    def _build_story_relevance_table(self, table):
        """

        :param table:
        :return:
        """
        rel_table = dict()

        for val in table:
            story_info = (int(val[2]),val[1], val[3])
            if val[0] in rel_table:
                rel_table[val[0]].append(story_info)
            else:
                rel_table[val[0]] = [story_info]
        return rel_table

    def construct_sen_boundaries(self):
        for key in self.doc_boundaries.keys():
            bnd_list = self._get_sample_file_bnd_nltk_sent(key)
            self.sent_boundaries[key] = bnd_list

        self.text_corpus.clear()

    def construct_char_bnds(self):
        for key, text in self.text_corpus_bnds.items():
            text = ' '.join(text)
            text = text.replace('<bnd>','\n\n\t')

            bnds = []

            for i in range(len(text)):
                if i<len(text) - 1 and text[i] == '\n' and text[i+1] == '\n':
                    bnds.append(i)

            self.char_boundaries[key] = bnds

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

    def read_bnd_sample_files(self, boundary_path):

        MAX_SAMPLES = 0

        for name in os.listdir(boundary_path):
            if MAX_SAMPLES >= 1000000:
                break
            if "_MAN" not in name:
                sample_path = os.path.join(boundary_path,name)
                boundaries = self._read_bnd_file(sample_path)
                name = name.replace('.tkn_bnd','')
                self.doc_boundaries[name] = boundaries
                MAX_SAMPLES += 1

    def read_tkn_sample_files(self, token_path):
        MAX_SAMPLES = 0
        for name in os.listdir(token_path):
            if MAX_SAMPLES >= 1000000:
                break
            if "_MAN" not in name:
                sample_path = os.path.join(token_path,name)
                sample_tokens, token_ids = self._read_tkn_file(sample_path)
                name = name.replace('.tkn','')
                self.text_corpus[name] = sample_tokens
                self.word_ids[name] = token_ids
                self.total_words += len(sample_tokens)
                MAX_SAMPLES += 1

    def read_story_topics(self, story_topics_path):
        with open(story_topics_path) as file:
            story_topics_table = file.readlines()
        story_topics_table = [x.strip().split() for x in story_topics_table]

        self.story_topics_table = self._build_story_relevance_table(story_topics_table)

    def read_corpus_datasets(self, dataset_dir):
        train_path = os.path.join(dataset_dir,'train')
        test_path = os.path.join(dataset_dir,'test')
        dev_path = os.path.join(dataset_dir,'dev')

        with open(train_path) as file:
            self.train_set = file.readlines()
            self.train_set = [x.strip() for x in self.train_set]
        with open(dev_path) as file:
            self.dev_set = file.readlines()
            self.dev_set = [x.strip() for x in self.dev_set]
        with open(test_path) as file:
            self.test_set = file.readlines()
            self.test_set = [x.strip() for x in self.test_set]

    def validate_dataset(self):
        for id in self.train_set:
            if id not in self.text_corpus_bnds:
                print('Error: sample '+id+' not found in train')

        for id in self.test_set:
            if id not in self.text_corpus_bnds:
                print('Error: sample '+id+' not found in test')

        for id in self.dev_set:
            if id not in self.text_corpus_bnds:
                print('Error: sample '+id+' not found in dev')

def main_process(load_from_rawcorpus):

    corpus = TDTCorpus()

    if  load_from_rawcorpus:
        print("reading corpus...")
        print("reading token files...")
        corpus.read_tkn_sample_files('./data/tdt2_em_v4_0/tdt2proj/tdt2_em/tkn')

        print("reading story topics...")
        corpus.read_story_topics('/home/mohamed/Desktop/scripts/tdt_reader/story_topics.tbl')

        print("reading boundary files...")
        corpus.read_bnd_sample_files('./data/tdt2_em_v4_0/tdt2proj/tdt2_em/tkn_bnd')
        # print("reading corpus...")

        #print("reading token files...")
        #corpus.read_tkn_sample_files('./data/tdt2_em_v4_0/tdt2proj/tdt2_em/tkn')

        #

        #
        # print("reading boundary files...")
        # corpus.read_bnd_sample_files('./data/tdt2_em_v4_0/tdt2proj/tdt2_em/tkn_bnd')

        print("constructing sentence and char boundary vectors...")
        corpus.construct_sen_boundaries()
        corpus.construct_char_bnds()

        print("corpus length: ",len(corpus.text_corpus)," ",len(corpus.word_ids))#," ",corpus.text_corpus.keys()
        #print("Boundaries: ",corpus.doc_boundaries)
        #print(len(corpus.doc_boundaries.values()[0]))

        # print("dividing and validating datasets...")
        # corpus.read_corpus_datasets('/home/mohamed/Desktop/tdt2_em_v4_0/tdt2_em/dnnhmm_sets')
        # corpus.validate_dataset()

        print("total size (words): ",corpus.total_words)
        print("total stories: ",corpus.total_stories)
        print("total annotated stories: ",corpus.total_annotated_stories)
        print("saving corpus data...")
        pickle.dump(corpus,open("corpus_annotated_data_nltk_boundaries.pckl",'wb'))
        print("done corpus reading...")

    else:
        print("loading corpus...")
        corpus = pickle.load(open("corpus_annotated_data_nltk_boundaries.pckl",'r'))
        annotated_words = 0

        for doc_segs in corpus.doc_boundaries.values():
            for segment in doc_segs:
                if segment.topic_annot:
                    annotated_words += (segment.end_id - segment.beg_id)

        print("Corpus total words: %i"%corpus.total_words)
        print("Corpus annotated words: %i"%annotated_words)


    print('segmenting text using TextTiling...')
    output_bnds = dict()
    gap_scores = dict()
    smooth_scores = dict()
    depth_scores = dict()

    for key,val in corpus.text_corpus_bnds.items():
        key = key.strip()

        text = ' '.join(val)
        text = text.replace('<bnd>','\n\n\t')
        #file = open('/home/mohamed/Desktop/scripts/tdt_reader/'+key,'w')
        #file.write(text)

        #segmentation = texttiling.TextTiling()

        tt = ModifiedTextTilingTokenizer(w=20,k=10,demo_mode=True)
        gap,smooth,depth,tk, norm_bnds = tt.tokenize(text)

        predicted_bnd = corpus.convert_char_bounds_to_binary(key,norm_bnds)



        output_bnds[key] = predicted_bnd
        gap_scores[key] = gap
        smooth_scores[key] = smooth
        depth_scores[key] = depth

    print('finished segmentation...')

    evaluate_seg(corpus, output_bnds, gap_scores, smooth_scores, depth_scores)
    #pickle.dump(output_bnds,open("output_bnds_test.pckl",'w'))
    #pickle.dump(gap_scores,open("gap_scores_test.pckl",'w'))
    #pickle.dump(smooth_scores,open("smooth_scores_test.pckl",'w'))
    #pickle.dump(depth_scores,open("depth_scores_test.pckl",'w'))

        #seg_text = segmentation.segmentFile('/home/mohamed/Desktop/scripts/tdt_reader/'+key)

def convert_bnd_to_segeval_format(ref_bnds, hyp_bnds):
    ref_eval = []
    hyp_eval = []

    count_hyp = 0
    count_ref = 0
    for i, j in zip(hyp_bnds,ref_bnds):
        if i == 1:
            hyp_eval.append(count_hyp+1)
            count_hyp = 0
        else:
            count_hyp += 1

        if j == 1:
            ref_eval.append(count_ref+1)
            count_ref = 0
        else:
            count_ref += 1

    return ref_eval, hyp_eval

def evaluate_seg(corpus, out_bnds, gap_scores=[], smooth_scores=[], depth_scores=[]):
    pk_segeval = dict()
    precision_segeval = dict()
    recall_segeval = dict()

    matplotlib.use('Agg')
    print("calculating evaluation scores...")
    t_seg = []
    p_seg = []
    i=0

    for key, val in  out_bnds.items():

        plot_x = np.arange(0,len(depth_scores[key]),1)

        ref_bnds = corpus.sent_boundaries[key]
        hyp_bnds = out_bnds[key]


        t_seg.append(ref_bnds.count(1))
        p_seg.append(hyp_bnds.count(1))

        hyp_bnds[-1] = 1

        ref_eval, hyp_eval = convert_bnd_to_segeval_format(ref_bnds, hyp_bnds)

        pk_segeval[key] = calculate_pk(ref_bnds, hyp_bnds)
        #conf_matrix = segeval.boundary_confusion_matrix(hyp_eval, ref_eval,boundary_format='mass')
        #precision_segeval[key] = segeval.precision(conf_matrix)
        #recall_segeval[key] = segeval.recall(conf_matrix)

        #depth_scores_np = np.array(depth_scores[key])
        #ref_bnds_np = np.array(ref_bnds)

        #print(depth_scores_np.shape
        #print(plot_x.shape
        #print(ref_bnds_np.shape

        #fig = matplotlib.pyplot.figure()
        #ax = fig.add_subplot(111)
        #ax.plot(plot_x,np.sort(depth_scores_np))
        #ax.plot(plot_x,ref_bnds_np)
        #ax.plot(plot_x,bounds)
        i+=1
        #fig.savefig('temp'+str(i)+'.png')

    #print("precision: ", sum(precision_segeval.values())/float(len(precision_segeval.values())))
    #print("recall: ", sum(recall_segeval.values())/float(len(recall_segeval.values())))
    print("pk: %f"%(sum(pk_segeval.values())/float(len(pk_segeval.values()))))
    print(pk_segeval.values())
    #plot_x = np.arange(0,len(t_seg),1)
    #fig = matplotlib.pyplot.figure()

    #ax = fig.add_subplot(111)
    #ax.plot(plot_x,t_seg)
    #ax.plot(plot_x,p_seg)
    print(t_seg)
    print(p_seg)

    #fig.savefig('num_seg.png')

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

def calculate_pk(ref,hyp):

    ref = convert_bounds_2_labels(ref)
    hyp = convert_bounds_2_labels(hyp)

    num_segs = ref[-1] + 1
    num_elems = len(ref)

    p_k = 0
    k = int((0.5*num_elems/num_segs) - 1 )
    if k == 0:
        k = 2

    print(k)
    print( num_elems)
    print( num_segs)
    print("----------")
    for i in range(0,num_elems - k + 1):
        delta_ref = (ref[i] is ref[i+k - 1])
        delta_hyp = (hyp[i] is hyp[i+k - 1])

        if delta_ref != delta_hyp:
            p_k += 1
    p_k = p_k / float(num_elems - k + 1)

    return p_k

def main():

    print("loading corpus...")
    corpus = pickle.load(open("corpus_data.pckl",'r'))

    print("loading hypothesized segments and stats...")
    out_bnds = pickle.load(open("output_bnds_test.pckl",'r'))
    gap_scores = pickle.load(open("gap_scores_test.pckl",'r'))



if __name__ == "__main__":
    main_process(True)