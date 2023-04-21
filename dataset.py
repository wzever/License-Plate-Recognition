import torch
import glob
import os
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import numpy as np
from torchvision import transforms
import json

class PlateDataset(Dataset):
    def __init__(self):
        self.imgs = glob.glob('VehicleLicense/Data/*/*.jpg')
        self.labels = self.process_labels()
        self.transforms = transforms.Compose([transforms.ToTensor(),
                                              transforms.CenterCrop(40),
                                              transforms.RandomAffine(degrees=(-15,15), translate=(0.1,0.1), scale=(0.8,1.2), fill=0)])
        self.id_to_label = self.process_labels()[1]

    def __getitem__(self, index):   
        img = self.imgs[index]
        label = self.labels[index]
        pil_img = Image.open(img)
        if len(pil_img.split()) == 3:
            pil_img = pil_img.convert('L')
        data = self.transforms(pil_img)
        
        return data, label
    
    def __len__(self):
        return len(self.imgs)
    
    def process_labels(self):
        labels = os.listdir('VehicleLicense/Data')
        all_labels = []
        for img in self.imgs:
            for i, c in enumerate(labels):
                if c == (img.split('\\'))[-1][:-9]:
                    all_labels.append(i)
        return all_labels

#划分测试集和训练集
def split_set(ratio, batch_size):
    plate_dataset = PlateDataset()
    index = np.random.permutation(len(plate_dataset))

    all_imgs_path = np.array(plate_dataset.imgs)[index]

    train_len = int(len(all_imgs_path) * ratio)
    test_len = len(plate_dataset) - train_len

    train_set, test_set = random_split(plate_dataset, 
                                    [train_len, test_len], 
                                    generator=torch.Generator().manual_seed(1))
    
    train_loader = DataLoader(train_set,
                            batch_size=batch_size,
                            shuffle=False)

    test_loader = DataLoader(test_set,
                            batch_size=batch_size,
                            shuffle=False)
    return train_loader, test_loader