datasets_args = {
            'data_path':'data/sentence_data.txt',
            'datasets_path':'data/sentence_datasets.hdf5',
            'sentence_max_label':16,
            'gesture_dic_path':'config/gesture_dic_all.txt',
            'gesture_count_path':'config/gesture_count.txt',
            'synonym_dic':'config/synonym_dic.txt',
            'gesture_synonym_list':'config/gesture_synonym_list.txt'
}

model_args = {
              'model_weight_path':'',
}


train_args = {
              'device':'cpu',
              'epoch':50,
              'batch_size':128
}