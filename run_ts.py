#coding-utf-8
import sys
import WordCover
from keras.models import load_model
import pickle
import evalate
import data_util
def main(input_is='tdt2',funs='ts',embd='tieba'):

    if embd=='tieba':
        word2id = WordCover.loadw2c_200dim()[0]
    else:
        word2id = WordCover.load_wiki_tieba()[0]

    if input_is == 'tdt2':
        x = pickle.load(open("./e2e_set/train_data.pckl",'rb'))
        y = pickle.load(open("./e2e_set/train_label.pckl",'rb'))
    else:
        x = pickle.load(open("./e2e_set/testdata.pckl",'rb'))
        y = pickle.load(open("./e2e_set/test_label.pckl",'rb'))

    if funs == 'ts':

        model = load_model('topic_segmentation_e2e.h5')

    else:
        model  = load_model('dialog_segmentation_e2e.h5')

    X,y_binary = data_util.data_tranform(x,y,word2id)

    predictions = model.predict(X)
    # print("accuracy score: ",accuracy_score(y, p))
    print('predic')
    print(evalate.getEvaate(predictions,y_binary))
    print('p:r:f1:')
    print(evalate.getEvaateRangeSent(predictions,y_binary))


if __name__ == '__main__':

    params={1:'tdt2',
            2:'topic_segment',
            3:'wiki'}
    for i in range(1,len(sys.argv)):
        params[i] = sys.argv[i]

    main(input_is=params[1],funs=params[2],embd=params[3])
