import torch
import h5py




# 定义自己的数据集：datatxt中数据记录是以字典记录，包含name，gender和gaze。
class MyDataset(torch.utils.data.Dataset):  # 创类：MyDataset,继承torch.utils.data.Dataset
    def __init__(self, datasets_path):
        super(MyDataset, self).__init__()
        # 打开数据集，读取数据跟标签
        datasets_path = datasets_path
        print(datasets_path)
        datasets = h5py.File(datasets_path, 'r')  
        sentence_datas = datasets['sentence_data'][:]
        datasets.close()

        self.sentence_datas = sentence_datas

    def __getitem__(self, index):  # 按照索引读取每个元素的具体内容
        sentence_data = self.sentence_datas[index]  
        return sentence_data # return回哪些内容，在训练时循环读取每个batch，就能获得哪些内容
             

    def __len__(self):  # 它返回的是数据集的长度，必须有
        return len(self.imu_datas)


## 定义数据集加载函数
def loader_data(datasets_path, batch_size): 
    datasets = MyDataset(datasets_path=datasets_path)

    data_loader = torch.utils.data.DataLoader(dataset=datasets, batch_size=batch_size,
                                                shuffle=True, num_workers = 0)  
    return data_loader


