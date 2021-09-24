from numpy.core.numeric import Inf
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
from utils.load_datasets import loader_data
from tqdm import tqdm
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import copy
import numpy as np
from config.config import train_args, model_args, datasets_args
from word2vec_model.word2vec_layer import  Word2Vec_model
from collections import OrderedDict


def start_net(net, train_data_loader):
    '''
    设置参数
    '''
    epoch_size = train_args['epoch']     ## 训练轮数
    ## 计算训练使用的batch
    batch_num = len(train_data_loader)
    train_batch_num = round(batch_num * train_args['train_rate'])

    ## 复制模型参数
    best_model_wts = copy.deepcopy(net.state_dict())
    train_loss_all = []

    val_loss_all = []
    best_loss = Inf
    ## 模型保存路径
    model_path = model_args['model_weight_path']

    ## 设置损失函数、优化器和学习率
    criterion_L1 = nn.L1Loss(size_average=False)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    ## model训练
    for epoch, i in zip(range(epoch_size),tqdm(range(epoch_size))):
        train_loss = 0.0
        train_loss_num = 0
        train_num = 0
        
        val_loss = 0.0
        val_loss_num = 0
        val_num = 0
        
        for step,(sentence_inputs) in (enumerate(train_data_loader)):
            if train_args['device'] == 'GPU':
                sentence_inputs = sentence_inputs.cuda()
            if step <= train_batch_num:
                net.train()  ## 设置model为训练模式
                sentence_emb, sentence_pre = net(sentence_inputs)
                if train_args['device'] == 'GPU':
                    sentence_pre = sentence_pre.cuda()
                
                loss = criterion_L1(sentence_pre,sentence_inputs)
                # print('all_loss:',loss)
                # loss = train_args['alpha'] * mycriterion(labels, outputs) + (1 - train_args['alpha']) * criterion_L1(pre_label,labels)
                # loss = mycriterion(labels, outputs)
                optimizer.zero_grad()
                # torch.autograd.set_detect_anomaly(True)
                # print(loss)
                loss.backward(retain_graph=True)
                optimizer.step()
                train_loss += loss.item()
                train_loss_num += sentence_pre.shape[0]
                train_num += 1
            
            else:
                # print('11')
                net.eval()  ## 训练model为评估模式
                sentence_emb, sentence_pre = net(sentence_inputs)
                if train_args['device'] == 'GPU':
                    sentence_pre = sentence_pre.cuda()
                
                loss = criterion_L1(sentence_pre,sentence_inputs)
               
                val_loss += loss.item()
                val_loss_num += sentence_pre.shape[0]
                val_num += 1
            
        ## 计算一个epoch下训练集和验证集上的loss
        train_loss_all.append(train_loss / train_loss_num)
        val_loss_all.append(val_loss / val_loss_num)
        
        ## 检测损失，改变学习率
        scheduler.step(train_loss_all[-1])

        print("Epoch {} / {} Train loss {:.8f},Val loss {:.8f}\n".format(epoch, epoch_size, train_loss_all[-1], val_loss_all[-1]))
        
        
        ## 保存最好的模型参数
        if train_loss_all[-1] < best_loss:

            best_loss = train_loss_all[-1]
            best_model_wts = copy.deepcopy(net.state_dict())
            torch.save(net.state_dict(),model_path)
            # print('11')
    
    print("训练结束")


if __name__ == '__main__':
    print('设置模型参数：1:默认参数，2:自定义参数')
    change = input()
    if change == '2':
        ## 从键盘读入训练参数以及一些方式
        print('############################################################')
        emb_dim = input('请输入嵌入词向量维度：')
        print('############################################################')

        ## 配置参数
        model_args['enc_dim'] = int(emb_dim)

    ## 配置模型
    model =  Word2Vec_model(model_args['input_dim'], model_args['emb_dim'], model_args['label_len'])



    print('设置训练参数：1:默认参数，2:自定义参数')
    change = input()
    if change == '2':
        ## 从键盘读入训练参数以及一些方式
        print('############################################################')
        device_type = input('请输入训练方式（GPU或CPU）：')
        train_epoch = input('输入训练轮数（epoch）：')
        train_batch_size = input('请输入训练的batch_size：')
        print('############################################################')

        ## 配置参数
        train_args['device'], train_args['epoch'], train_args['batch_size'] = device_type, int(train_epoch), int(train_batch_size)

    ## 设置训练方式
    if train_args['device'] == 'GPU':
        net = torch.nn.DataParallel(model, device_ids=[0])
        net = net.cuda()
    else:
        net = model
    train_data_loader = loader_data(datasets_args['datasets_path'], train_args['batch_size'])
    
    ## 判断模型路径是否存在
    model_path = model_args['model_weight_path']
    # print(model_path)
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        if  train_args['device'] == 'GPU':
            state_dict = checkpoint 
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if 'module.' not in k:
                    # k = k.replace('module.','')
                    k = 'module.' + k
                else:
                    continue
                new_state_dict[k]=v
            net.load_state_dict(state_dict)
        else:
            state_dict = checkpoint 
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if 'module.'  in k:
                    k = k.replace('module.','')
                    # k = 'module.' + k
                else:
                    continue
                new_state_dict[k]=v
            net.load_state_dict(new_state_dict)
        print('导入模型成功')
    

    start_net(net, train_data_loader)

