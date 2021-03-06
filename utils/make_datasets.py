import os
import sys
sys.path.append('../')
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.insert(0,parentdir)
from config.config import datasets_args
import numpy as np
import h5py

def getSentenceData(row_data_path):
    filenames = os.listdir(row_data_path)
    sentence = []
    fuhao = ',./?。、！,。？'
    for fname in filenames:
        fname = fname.split("_",1)[0]
        for f in fuhao:
            if f in fname:
                fname.replace(f,'')
        if fname not in sentence:
            sentence.append(fname)
    juzi = open(datasets_args['data_path'],'w')
    for fname in sentence:
        print(fname,file=juzi)
    juzi.close()


def make_sentence_data(data_path, gesture_dic_path, gesture_synonym_list_path, save_data_path, datasets_type):
    ## 打开手势字典文件，读取手势字典
    gesture_dic_file = open(gesture_dic_path, 'r')
    gesture_dic = eval(gesture_dic_file.readline())
    gesture_dic_file.close()

    ## 打开同义词字典list
    gesture_synonym_list_file = open(gesture_synonym_list_path, 'r')
    gesture_synonym_list = eval(gesture_synonym_list_file.readline())
    gesture_synonym_list_file.close()

    ## 生成数据
    sentence_data = []
    fuhao = ',\!?。，？！、 '
    sentence_data_file = open(data_path, 'r')
    for sentence in sentence_data_file.readlines():
        sentence_label = []
        sentence = sentence.replace('\n','')
        ## 去除多余符号
        for x in fuhao:
            if x in sentence:
                sentence = sentence.replace(x,'')
        ## 将句子按照“-”进行划分
        sentence_word = sentence.split('-')
        print(sentence_word)
        ## 寻找手势标签
        for word in sentence_word:
            if datasets_type == 'train':
                sentence_label.append(gesture_dic[word])
            elif datasets_type == 'test':
                for i in range(len(gesture_synonym_list)):
                    if word in gesture_synonym_list[i]:
                        word = gesture_synonym_list[i][0]
                sentence_label.append(gesture_dic[word])
            else:
                print('error!!')
                return 0
        ## 补齐手势标签
        if len(sentence_label) < datasets_args['sentence_max_label']:
            sentence_label.insert(0,gesture_dic['sos'])
            # label.insert(len(label), gesture_dic['eos'])
            if len(sentence_label) < datasets_args['sentence_max_label']:
                for i in range(len(sentence_label),datasets_args['sentence_max_label']):
                    sentence_label.insert(i,gesture_dic['pos'])
        ## 将手势标签加入到手势数据当中
        sentence_data.append(sentence_label)
        print(sentence_label)

    ## 保存数据
    datasets = h5py.File(save_data_path,'w')
    datasets.create_dataset('sentence_data',data = np.array(sentence_data))
    datasets.close()


if __name__ == '__main__':
    row_data_path = datasets_args['row_data_path']
    data_path = datasets_args['data_path']
    gesture_dic_path = datasets_args['gesture_dic_path']
    save_data_train_path = datasets_args['datasets_train_path']
    gesture_synonym_list_path = datasets_args['gesture_synonym_list']
    ## 得到句子数据
    # getSentenceData(row_data_path)
    ## 制作训练集
    make_sentence_data(data_path, gesture_dic_path, gesture_synonym_list_path, save_data_train_path, 'train')
    ## 制作测试集
    save_data_test_path = datasets_args['datasets_test_path']
    make_sentence_data(data_path, gesture_dic_path, gesture_synonym_list_path, save_data_test_path, 'test')
