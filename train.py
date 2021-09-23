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
from config.config import train_args, model_args, datasets_args, evalue_args, test_datasets_args
import h5py
from collections import OrderedDict


def start_net(net, train_data_loader):
    ## 计算label_to_word的手势字典
    gesture_dic = change_key_value()
    
    '''
    设置参数
    '''
    epoch_size = train_args['epoch']     ## 训练轮数
    ## 计算训练使用的batch
    batch_num = len(train_data_loader)
    train_batch_num = round(batch_num * train_args['train_rate'])

    ## 模型保存路径
    global model_path_root
   
    model_path = model_path_root  + 'model_weight'
    history_path = model_path_root + 'history_seq2seq_model'
    for scale in train_args['data_scale']:
            model_path = model_path + '_' + scale
            history_path = history_path + '_' + scale
    model_path = model_path + '.pth'
    history_path = history_path + '.hdf5'
    model_args_path = model_path_root + 'model_args.txt'
    train_args_path = model_path_root + 'train_args.txt'
    test_args_path = model_path_root + 'test_args.txt'
    
    ## 保存模型，新联，测试的参数
    model_args_file = open(model_args_path,'w')
    print(model_args, file = model_args_file)
    model_args_file.close()
    train_args_file = open(train_args_path,'w')
    print(train_args, file = train_args_file)
    train_args_file.close()
    test_args_file = open(test_args_path,'w')
    print(test_datasets_args, file = test_args_file)
    test_args_file.close()

    ## 复制模型参数
    best_model_wts = copy.deepcopy(net.state_dict())
    best_WER = 1.0
    train_loss_all = []
    train_WER_all = []
    train_SER_all = []

    val_loss_all = []
    val_WER_all = []
    val_SER_all = []
    
    ## 设置损失函数、优化器和学习率
    criterion_CES = nn.CrossEntropyLoss(ignore_index=2)
    criterion_L1 = nn.L1Loss(size_average=False)
    mycriterion = MyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    ## model训练
    # print(train_args['train_rate'])
    # print(batch_num)
    # print(train_batch_num)
    for epoch, i in zip(range(epoch_size),tqdm(range(epoch_size))):
        train_loss = 0.0
        train_loss_num = 0
        train_WER = 0.0
        train_SER = 0.0
        train_num = 0
        
        val_loss = 0.0
        val_loss_num = 0
        val_WER = 0.0
        val_SER = 0.0
        val_num = 0

        
        for step,(imu_inputs, emg_inputs, labels, scales) in (enumerate(train_data_loader)):
            # print('imu_data:',imu_inputs[:,2])
            # print('emg_data:',emg_inputs[:,2])
            if train_args['device'] == 'GPU':
                imu_inputs, emg_inputs, labels = imu_inputs.cuda(), emg_inputs.cuda(), labels.cuda()
            # print(labels.device)
            # print(emg_inputs.shape)
            # print(imu_inputs.shape)
            if step <= train_batch_num:
                net.train()  ## 设置model为训练模式

                pre_label, outputs, _ = net(emg_inputs, imu_inputs, labels)
                if train_args['device'] == 'GPU':
                    outputs = outputs.cuda()
                # print(next(net.parameters()).device)
                # print(outputs.device)
                pre_lab = torch.argmax(outputs,2)
                # print(outputs.shape)
                # print(labels.shape)
                # print(pre_label.shape)
                sentence = []
                pre_sentence = []
                ## 输出评估过程中的句子
                for i in range(len(pre_lab)):
                    words = label_to_words(gesture_dic,labels[i])
                    pre_words = label_to_words(gesture_dic,pre_lab[i])
                    sentence.append(words)
                    pre_sentence.append(pre_words)
                    # print("src:",generate_sentence(words))
                    # print("pre:",generate_sentence(pre_words))
                    # print('\n')

                ser, wer, _ = computer_wer_ser(pre_sentence, sentence, evalue_args['sentence_rate'])
                train_WER += wer
                train_SER += ser
                loss = 0
                for i in range(len(outputs)):
                    # print('cr_loss:',loss)
                    # print(outputs[i])
                    loss = loss + criterion_CES(outputs[i] + 1e-6,labels[i].long())
                    # print('cr_loss:',loss)
                loss = train_args['alpha'] * loss + (1 - train_args['alpha']) * criterion_L1(pre_label,labels)
                # print('all_loss:',loss)
                # loss = train_args['alpha'] * mycriterion(labels, outputs) + (1 - train_args['alpha']) * criterion_L1(pre_label,labels)
                # loss = mycriterion(labels, outputs)
                optimizer.zero_grad()
                # torch.autograd.set_detect_anomaly(True)
                # print(loss)
                loss.backward(retain_graph=True)
                optimizer.step()
                train_loss += loss.item()
                train_loss_num += outputs.shape[0]
                train_num += 1
            
            else:
                net.eval()  ## 训练model为评估模式
                pre_label, outputs, _ = net(emg_inputs, imu_inputs, labels)
                if train_args['device'] == 'GPU':
                    outputs = outputs.cuda()
                pre_lab = torch.argmax(outputs,2)
                
                sentence = []
                pre_sentence = []
                ## 输出评估过程中的句子
                for i in range(len(pre_lab)):
                    words = label_to_words(gesture_dic,labels[i])
                    pre_words = label_to_words(gesture_dic,pre_lab[i])
                    sentence.append(words)
                    pre_sentence.append(pre_words)
                    print("src:",generate_sentence(words))
                    print("pre:",generate_sentence(pre_words))
                    print('\n')

                ser, wer, _ = computer_wer_ser(pre_sentence, sentence, evalue_args['sentence_rate'])
                val_WER += wer
                val_SER += ser

                # loss = 0
                # for i in range(len(outputs)):
                loss += criterion_CES(outputs[i],labels[i].long()) #
                loss += criterion_L1(pre_label,labels)
                val_loss += loss.item()
                val_loss_num += outputs.shape[0]
                val_num += 1
            
        ## 计算一个epoch下载训练集和验证集上的wer，ser，loss
        train_loss_all.append(train_loss / train_loss_num)
        train_WER_all.append(train_WER / train_num)
        train_SER_all.append(train_SER / train_num)

        # val_loss_all.append(val_loss / val_loss_num)
        # val_WER_all.append(val_WER / val_num)
        # val_SER_all.append(val_SER / val_num)
        
        ## 检测损失，改变学习率
        scheduler.step(train_loss_all[-1])

        # print("Epoch {} / {} Train loss {:.8f},Train WER {:.8f},Train SER {:.8f},val loss {:.8f},val WER {:.8f},val SER {:.8f}"
        # .format(epoch, epoch_size, train_loss_all[-1],train_WER_all[-1],train_SER_all[-1],val_loss_all[-1],val_WER_all[-1],val_SER_all[-1]))

        print("Epoch {} / {} Train loss {:.8f},Train WER {:.8f},Train SER {:.8f}\n"
        .format(epoch, epoch_size, train_loss_all[-1],train_WER_all[-1],train_SER_all[-1]))
        
        
        ## 保存最好的模型参数
        if train_WER_all[-1] < best_WER:
            best_WER = train_WER_all[-1]
            best_model_wts = copy.deepcopy(net.state_dict())
            torch.save(net.state_dict(),model_path)
    
    
    history = h5py.File(history_path,'w')
    history.create_dataset('train_loss',data = np.array(train_loss_all))
    history.create_dataset('train_WER',data = np.array(train_WER_all))
    history.create_dataset('train_SER',data = np.array(train_SER_all))
    history.create_dataset('val_loss',data = np.array(val_loss_all))
    history.create_dataset('val_WER',data = np.array(val_WER_all))
    history.create_dataset('val_SER',data = np.array(val_SER_all))
    history.close()

    print("训练结束")


if __name__ == '__main__':
    print('设置模型参数：1:默认参数，2:自定义参数')
    change = input()
    if change == '2':
        ## 从键盘读入训练参数以及一些方式
        print('############################################################')
        block_size = input('请输入分块大小block_size：')
        print('############################################################')

        ## 配置参数
        model_args['block_size'], model_args['enc_dim'] = int(block_size), int(15000/int(block_size))

    ## 配置模型
    model = word2vec_model



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
    train_data_loader = loader_data('datasets_path', train_args['batch_size'])
    
    ## 判断模型路径是否存在
    model_path = model_args['model_weight_path']
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

