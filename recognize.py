import os
import torch
from PIL import Image
from model import CNN
from torchvision import transforms
import json

colors = {'green': '绿牌', 'blue': '蓝牌'}

def recognize(segs, color, pretrain='pretrained/best_96.6182.pt'):
    res = ''
    labels = os.listdir('VehicleLicense/Data')
    with open('VehicleLicense/label_match.json', encoding='utf-8') as f:
        chinese_label = json.load(fp=f)
    id_to_label = {k: chinese_label[v] for k, v in enumerate(labels)}

    net = CNN()
    net.load_state_dict(torch.load(pretrain))
    load = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(40)])
    i = 0
    print(f'识别结果: {colors[color]}, ', end='')
    for img in segs: 
        img = load(Image.fromarray(img)).unsqueeze_(0)
        pred = net(img).max(dim=1)[1]
        pred = id_to_label[pred.item()]
        res += pred
        if i == 1:
            res += '-'
        i += 1
        print(pred, end='')
    print('\n')
    return res