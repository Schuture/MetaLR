import os
import random
import pickle
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

random.seed(1)


class COVIDDataset(Dataset):
    def __init__(self, data_dir, fold=[0,1,2,3], transform=None):
        """
        肺炎分类任务的Dataset，按照fold分，不论训练、验证
            param data_dir: str, 数据集所在路径
            param fold: int, 0-4
            param mode: 'train', 'valid', 'test'
            param transform: torch.transform，数据预处理方法
        """
        self.label_name = {"covid19": 0, "pneumonia": 1, "regular": 2}
        with open(data_dir, 'rb') as f: # 导入所有数据
            X, y = pickle.load(f)
        
        self.X = []
        self.y = []
        for i in range(5):
            if i in fold:
                if len(self.X) == 0:
                    self.X = X[i]
                    self.y = y[i]
                else:
                    self.X = np.concatenate((self.X, X[i]))
                    self.y = np.concatenate((self.y, y[i]))
                
        self.transform = transform
    
    def __getitem__(self, index): # 根据index返回数据
        img_arr = self.X[index].transpose(1,2,0) # CHW => HWC 
        img = Image.fromarray(img_arr.astype('uint8')).convert('RGB') # 0~255
        label = self.y[index]

        if self.transform is not None:
            img = self.transform(img)   # 在这里做transform，转为tensor等等

        return img, label

    def __len__(self): # 查看样本的数量
        return len(self.y)
        