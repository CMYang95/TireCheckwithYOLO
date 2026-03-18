# makeTxt.py
import os
import random


trainval_percent = 0.9
train_percent = 0.9
xmlfilepath = r'D:\TireCheck\Tire_dataset\annotation'
txtsavepath = r'D:\TireCheck\Tire_dataset\ImageSets'
total_xml = os.listdir(xmlfilepath)

num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

ftrainval = open('D:/TireCheck/Tire_dataset/ImageSets/trainval.txt', 'w',encoding='UTF-8')
ftest = open('D:/TireCheck/Tire_dataset/ImageSets/test.txt', 'w',encoding='UTF-8')
ftrain = open('D:/TireCheck/Tire_dataset/ImageSets/train.txt', 'w',encoding='UTF-8')
fval = open('D:/TireCheck/Tire_dataset/ImageSets/val.txt', 'w',encoding='UTF-8')

for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
