import os
import sys
sys.path.append('../')
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.insert(0,parentdir)
from config.config import datasets_args
from utils.get_gesture_dic import get_ges_and_syn_dic
from test import synonym_change

## 计算同义词转换的准确率
def compute_acc(truth_sentence, test_sentence, change_sentence):
    ## 读取同义词字典、读取手势字典
    gesture_dic, gesture_synonym_list = get_ges_and_syn_dic(datasets_args['gesture_dic_path'], datasets_args['gesture_synonym_list'])
    ## 得到手势同义词字典
    synonymchange = synonym_change()
    synonym_label = synonymchange.get_synonym_label(gesture_synonym_list, gesture_dic)
    ## 统计同义词和替换正确的同义词
    synonym_count = 0
    change_true = 0
    for i in range(len(truth_sentence)):
        for j in range(len(test_sentence[i])):
            if test_sentence[i][j] in synonym_label.keys():
                synonym_count += 1
                if change_sentence[i][j] == truth_sentence[i][j]:
                    change_true += 1
    return change_true/synonym_count

## 可视化结果
def show_result(truth_sentence, test_sentence, change_sentence):
    ## 读取同义词字典、读取手势字典
    gesture_dic, gesture_synonym_list = get_ges_and_syn_dic(datasets_args['gesture_dic_path'], datasets_args['gesture_synonym_list'])
    ## 将手势字典进行键值转化
    gesture_dic_ = dict(zip(gesture_dic.values(), gesture_dic.keys()))
    ## 得到手势同义词字典
    synonymchange = synonym_change()
    synonym_label = synonymchange.get_synonym_label(gesture_synonym_list, gesture_dic)

    result_file = open('result/result.txt','w')
    for i in range(len(truth_sentence)):
        test_gesture = []
        test_synonym_gesture = []
        for j in range(len(truth_sentence[i])):
            if test_sentence[i][j] in synonym_label.keys():
                # print(gesture_dic_[test_sentence[i][j]])
                test_gesture.append('mask')
                test_synonym_gesture.append([gesture_dic_[truth_sentence[i][j]], gesture_dic_[change_sentence[i][j]]])
            else:
                test_gesture.append(gesture_dic_[test_sentence[i][j]])
            ## 保存结果
        print(test_gesture,'==>',test_synonym_gesture, file=result_file)
    result_file.close()
    
    