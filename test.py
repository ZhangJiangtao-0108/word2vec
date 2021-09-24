from sys import set_coroutine_origin_tracking_depth
from numpy.core.numeric import Inf
from numpy.lib.function_base import delete
import torch
from utils.load_datasets import loader_data
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import numpy as np
from config.config import model_args, datasets_args
from word2vec_model.word2vec_layer import  Word2Vec_model
from collections import OrderedDict



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
    def get_word2sentence_distance(self, word_emb, sentence_emb):
        ## 将word_emb进行扩展
        # print(sentence_emb)
        word_emb_ = np.tile(word_emb, (len(sentence_emb), 1))
        ## 计算每个位置元素的差值
        diffMat = word_emb_ - sentence_emb
        sqDiffMat = diffMat ** 2
        sqDistances = sqDiffMat.sum(axis = 1)
        Distance = sqDistances ** 0.5
        return Distance.sum()/len(sentence_emb)


    ## 替换同义词
    def synonym_change(self, sentence_data, net):
        ## 读取手势字典，和同义词list
        gesture_dic_file = open(self.gesture_dic_path, 'r')
        gesture_dic = eval(gesture_dic_file.readline())
        gesture_dic_file.close()
        gesture_synonym_list_file = open(self.gesture_synonym_list_path,'r')
        gesture_synonym_list = eval(gesture_synonym_list_file.readline())
        gesture_synonym_list_file.close()

        ## 获取同义词label
        synonym_label = self.get_synonym_label(gesture_synonym_list,gesture_dic)
        sentence_data_out = []
        for sentence in sentence_data:
            ## 替换句子中的同义词
            for i in range(len(sentence)):
                ## 判断句子中是否有同义词
                if sentence[i] in synonym_label.keys():
                    '''
                    1、需要将这一行的label进行掩码
                    2、计算候选词到句子的距离
                    3、根据距离选择候选词
                    '''
                    ## 得到句子的词向量表示
                    sentence_emb = self.get_wordvec(sentence, net)
                    
                    ## 将候选词位置删除，方便距离计算
                    delete(sentence_emb,i,axis = 0)
                    ## 获取候选词的embedding,并举算候选词到句子的距离
                    # word_label = None  ## 替换的单词编号
                    word_best_distance = Inf
                    for word in synonym_label[sentence[i]]:
                        label = gesture_dic[word]
                        word_emb = self.get_wordvec(label, net)
                        ## 计算word_emb和sentence_emb的距离
                        word_distance = self.get_word2sentence_distance(word_emb, sentence_emb)
                        if word_distance < word_best_distance:
                            word_best_distance = word_distance
                            word_label = gesture_dic[word]
                    ## 替换同义词
                    sentence[i] = word_label
            sentence_data_out.append(sentence)
        
        return sentence_data_out



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
    sentence_test = [[9,473,359,470,338,8,8,8,8,8,8,8,8,8,8,8],
                    [9,533,913,312,709,8,8,8,8,8,8,8,8,8,8,8],
                    [9,109,58,8,8,8,8,8,8,8,8,8,8,8,8,8],
                    [9,818,109,246,764,8,8,8,8,8,8,8,8,8,8,8],
                    [9,78,130,93,32,236,292,770,189,231,559,502,8,8,8,8],
                    [9,329,932,33,553,923,175,421,387,101,8,8,8,8,8,8]]
    synonymchange = SynonymChange()
    sentence = synonymchange.synonym_change(sentence_test,net)
    print(sentence)
