'''
功能：生成手势字典并统计手势词出现的个数
作者：张江涛
'''
import os
import sys
sys.path.append('../')
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.insert(0,parentdir)
from config.config import datasets_args



'''
同义词匹配
'''
class Synonym_matching():
    def __init__(self, gesture_dic, synonym_dic_path):
        self.gesture_dic = gesture_dic
        self.synonym_dic_path = synonym_dic_path

    
    def find_gesture_synonym_list(self, gesture_synonym_list_path):
        '''
        寻找词库当中的同义词，并返回同义词list
        '''
        gesture_synonym_file = open(self.synonym_dic_path,'r',encoding='utf-8')
        gesture_synonym = []
        for line in gesture_synonym_file.readlines():
            line = line.replace('\n','').split("\t")
            if '' in line:
                line = list(set(line))
                line.remove('')
            gesture_synonym.append(line)
        little_gesture_synonym = []
        for gesture in self.gesture_dic.keys():
            number = 0
            for i in range(len(gesture_synonym)):
                if gesture in gesture_synonym[i]:
                    number = i
            if gesture in gesture_synonym[number]:
                # print(gesture_synonym)
                gesture_synonym_list = []
                for gest in gesture_synonym[number]:
                    # print(gesture_synonym[i])
                    if gest in self.gesture_dic.keys():
                        gesture_synonym_list.append(gest)
                gesture_synonym_list.sort(reverse=True)
                if (gesture_synonym_list not in little_gesture_synonym) and (len(gesture_synonym_list) != 1) :
                    little_gesture_synonym.append(gesture_synonym_list)
        gesture_synonym_file.close()
        '''
        保存词库当中同义词list
        '''
        gesture_synonym_list_file = open(gesture_synonym_list_path, 'w')
        print(little_gesture_synonym, file = gesture_synonym_list_file)
        gesture_synonym_list_file.close()
        return little_gesture_synonym

def get_gesture_dic(data_path, gesture_dic_path, gesture_count_path, synonym_dic_path, gesture_synonym_list_path):
    gesture_dic = open(gesture_dic_path,'w')
    gesture_count = open(gesture_count_path,'w')
    data_file = open(data_path,'r')
    filenames = data_file.readlines()
    data_file.close()
    gesture = []
    fuhao = ',\!?。，？！、'
    for fname in filenames:
        fname = fname.replace('\n', '')
        for x in fuhao:
            if x in fname:
                fname = fname.replace(x,'')
        words = fname.split('_')[0].split('-')
        gesture += words

    gestures = list(set(gesture))#.sort()
    print(gestures)
    gestures += ['sos','eos','pos']
    gestures.sort()
    print(gestures)
    ## 统计词的个数
    gesture_count_ = {}
    for g in gestures:
        gesture_count_[g] = gesture.count(g)
    print(gesture_count_,file=gesture_count) 
    gesture_count.close()
    # gestures.remove('')
    
    '''
    将字典中的同义词编码统一(尚未实现)
    '''
    ## 对手势进行编号
    gesture_ = dict(zip(gestures,range(len(gestures))))
    
    ## 获取同义词list,并保存
    gesture_synonym_list = Synonym_matching(gesture_, synonym_dic_path).find_gesture_synonym_list(gesture_synonym_list_path)
    
    # ## 将同义词手势的编号统一 
    # for word in gesture_.keys():
    #     for ind in range(len(gesture_synonym_list)):
    #         if word in gesture_synonym_list[ind]:
    #             gesture_[word] = gesture_[gesture_synonym_list[ind][0]]

    ## 保存手势字典
    print(gesture_,file=gesture_dic) 
    gesture_dic.close()
    
## 读取手势字典和同义词list
def get_ges_and_syn_dic(gesture_dic_path, gesture_synonym_list_path):
    ## 读取手势字典，和同义词list
    gesture_dic_file = open(gesture_dic_path, 'r')
    gesture_dic = eval(gesture_dic_file.readline())
    gesture_dic_file.close()
    gesture_synonym_list_file = open(gesture_synonym_list_path,'r')
    gesture_synonym_list = eval(gesture_synonym_list_file.readline())
    gesture_synonym_list_file.close()
    return gesture_dic, gesture_synonym_list


if __name__ == '__main__':
    data_path = datasets_args['data_path']
    gesture_dic_path = datasets_args['gesture_dic_path']
    gesture_count_path =  datasets_args['gesture_count_path'] 
    synonym_dic_path = datasets_args['synonym_dic']
    gesture_synonym_list_path = datasets_args['gesture_synonym_list']
    # print(data_path)
    # print(gesture_count_path)
    # print(gesture_dic_path)
    # print(synonym_dic_path)
    # print(gesture_synonym_list_path)
    get_gesture_dic(data_path, gesture_dic_path, gesture_count_path, synonym_dic_path, gesture_synonym_list_path)