from numpy.core.numeric import Inf
from numpy.lib.function_base import delete
import torch
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import numpy as np
from config.config import model_args, datasets_args
from word2vec_model.word2vec_layer import  Word2Vec_model
from utils.get_gesture_dic import get_ges_and_syn_dic
from utils.compute_show_acc import *
from collections import OrderedDict
import h5py


class SynonymChange():
    def __init__(self):
        self.gesture_synonym_list_path = datasets_args['gesture_synonym_list']
        self.gesture_dic_path = datasets_args['gesture_dic_path']

    ## 寻找同义词编号字典
    def get_synonym_label(self, gesture_synonym_list, gesture_dic):
        synonym_label = {}
        for i in range(len(gesture_synonym_list)):
            synonym_label[gesture_dic[gesture_synonym_list[i][0]]] = gesture_synonym_list[i]
        return synonym_label
    
    ## 得到句子label的词向量表示
    def get_wordvec(self, sentence, net):
        ## 需要将数据变成tensor便于网络计算
        sentence_ = np.array(sentence)
        sentence_ = torch.from_numpy(sentence_)
        # sentence_ = sentence_.unsqueeze(0)
        sentence_emb = net(sentence_)
        # sentence_emb = sentence_emb.squeeze()
        return sentence_emb.detach().numpy()

    ## 计算词向量到句子的距离,使用ED距离
    def get_word2sentence_distance(self, word_emb, sentence_emb, distence_type = 'ED'):
        ## 将word_emb进行扩展
        # print(sentence_emb)
        word_emb_ = np.tile(word_emb, (len(sentence_emb), 1))
        if distence_type == 'ED':
            ## 计算每个位置元素的差值
            diffMat = word_emb_ - sentence_emb
            sqDiffMat = diffMat ** 2
            sqDistances = sqDiffMat.sum(axis = 1)
            Distance = sqDistances ** 0.5
            W_S_distence = Distance.sum()/len(sentence_emb)
        elif distence_type == 'COS':
            word_emb_norm = np.linalg.norm(word_emb_, axis=1, keepdims=True)
            sentence_emb_norm = np.linalg.norm(sentence_emb, axis=1, keepdims=True)
            similiarity = np.dot(word_emb_, sentence_emb.T)/(word_emb_norm * sentence_emb_norm) 
            W_S_distence = 1. - similiarity
        else:
            print("error!!")
            return 0
        return W_S_distence


    ## 替换同义词
    def synonym_change(self, sentence_data, net, distence_type = 'ED'):
        ## 读取手势字典，和同义词list
        gesture_dic, gesture_synonym_list = get_ges_and_syn_dic(self.gesture_dic_path, self.gesture_synonym_list_path)
        ## 获取同义词label
        synonym_label = self.get_synonym_label(gesture_synonym_list,gesture_dic)
        sentence_data_out = []
        for sentence in sentence_data:
            # print(sentence)
            ## 替换句子中的同义词
            for i in range(len(sentence)):
                ## 判断句子中是否有同义词
                if sentence[i] in synonym_label.keys():
                    # print('句子当中的词：',sentence[i])
                    '''
                    1、需要将这一行的label进行掩码
                    2、计算候选词到句子的距离
                    3、根据距离选择候选词
                    '''
                    ## 得到句子的词向量表示
                    sentence_emb = self.get_wordvec(sentence, net)
                    # print(sentence_emb)
                    ## 将候选词位置删除，方便距离计算
                    delete(sentence_emb,i,axis = 0)
                    ## 获取候选词的embedding,并举算候选词到句子的距离
                    # word_label = sentence[i]  ## 替换的单词编号
                    word_best_distance = Inf
                    for word in synonym_label[sentence[i]]:
                        # print('同义词：',word, gesture_dic[word])
                        label = gesture_dic[word]
                        word_emb = self.get_wordvec(label, net)
                        ## 计算word_emb和sentence_emb的距离
                        word_distance = self.get_word2sentence_distance(word_emb, sentence_emb, distence_type)
                        # print('distance:', word_distance)
                        if word_distance < word_best_distance:
                            # print(gesture_dic[word])
                            word_best_distance = word_distance
                            word_label = gesture_dic[word]
                    ## 替换同义词
                    # print('替换的词：', word_label)
                    sentence[i] = word_label
            sentence_data_out.append(sentence)
        
        return sentence_data_out

def synonym_change():
    return SynonymChange()


if __name__ == '__main__':
    ## 定义模型
    model =  Word2Vec_model(model_args['input_dim'], model_args['emb_dim'], model_args['label_len'], 'test')
    ## 设置测试方式
    print('设置测试参数：1:默认参数，2:自定义参数')
    change = input()
    device_type = 'cpu'
    if change == '2':
        ## 从键盘读入训练参数以及一些方式
        print('############################################################')
        device_type = input('请输入训练方式（GPU或CPU）：')
        distance_type = input('请选择距离公式ED或者COS：')
        print('############################################################')

    if device_type == 'GPU':
        net = torch.nn.DataParallel(model, device_ids=[0])
        net = net.cuda().eval()
    else:
        net = model.eval()
    ## 模型加载参数
    model_path = model_args['model_weight_path']
    checkpoint = torch.load(model_path, map_location='cpu') #
    state_dict = checkpoint 
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module.'  in k:
             k = k.replace('module.','')
            # k = 'module.' + k
        else:
            continue
        new_state_dict[k]=v
    # print(new_state_dict.keys())
    net.load_state_dict(state_dict)     

    sentence_train_path = datasets_args['datasets_train_path']
    sentence_train = h5py.File(sentence_train_path, 'r')  
    sentence_train_datas = sentence_train['sentence_data'][:]
    sentence_train.close()

    sentence_test_path = datasets_args['datasets_test_path']
    sentence_test = h5py.File(sentence_test_path, 'r')  
    sentence_test_datas = sentence_test['sentence_data'][:]
    sentence_test.close()

    ## 进行同义词的替换
    synonymchange = SynonymChange()
    sentence_change = synonymchange.synonym_change(sentence_test_datas, net, distance_type)
    # print(sentence)
    acc = compute_acc(sentence_train_datas, sentence_test_datas, sentence_change)
    print(acc)
    show_result(sentence_train_datas, sentence_test_datas, sentence_change)
