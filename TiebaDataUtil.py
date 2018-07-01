__author__ = 'Administrator'
import codecs
import jieba
import pickle
import numpy as np
def main(filenaem='./data/part-00179'):

    with open(filenaem,'r') as f:
        lines = f.readline()
        print(len(lines),lines)


def getHeader(filename = './data/multi_turn_conversation_demo.txt'):

    testnums = 0
    testdata = []

    test_label = []
    f = codecs.open(filename,'r',encoding='utf-8')
    idx=0
    line = f.readline()
    print(line.strip().split('\t'))
    # lines = f.readlines()
    min_conv,max_conv = 100,0
    total_nums_convr = 0
    tmp_x,tmp_y=[],[]
    myflag = 0
    while line :
        # nums_Cov = len(line.strip().split('\t'))
        # min_conv = min(min_conv,nums_Cov)
        # max_conv = max(max_conv,nums_Cov)
        # total_nums_convr += nums_Cov
        idx+=1
        if(idx%5==0):
            for sent in line.strip().split('\t'):
                words = list(jieba.cut(sent))
                tmp_x.append(words)
                tmp_y.append(np.array([0]))
            tmp_y[-1]=np.array([1])
            myflag+=1
            if myflag == 5:
                testdata.append(tmp_x)
                test_label.append(tmp_y)
                # print(len(tmp_x),len(tmp_y))
                tmp_x,tmp_y=[],[]
                myflag = 0
                testnums+=1
            # print(idx,min_conv,max_conv)
        line = f.readline()
        if testnums==10000:
            break

    print('lenght test data',len(testdata),len(test_label))
    print('total number sentences is :',idx)        # 4734193
    print('averge:',total_nums_convr/idx)
    print('max and min is :',max_conv,min_conv)
    print('total convesations :',total_nums_convr)
    pickle.dump(testdata, open("./e2e_set/testdata.pckl", 'wb'), protocol=2)
    pickle.dump(test_label, open("./e2e_set/test_label.pckl", 'wb'), protocol=2)
    return testdata,test_label
if __name__ == '__main__':
    getHeader()