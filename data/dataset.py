# 此模块为数据加载模块，主要进行数据加载，预处理（利用transform）
import os
from PIL import Image
from torch.utils import data
from torchvision import transforms as T


# 数据处理要继承Dataset
class DogCat(data.Dataset):
    def __init__(self, root, transforms=None, train=True, test=False):
        self.test = test
        imgs = [os.path.join(root, img) for img in
                os.listdir(root)]  # 内部也可以定义不带self的变量，for循环放在列表中
        if self.test:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1]))
        else:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))
            #lambda,单值输出，多值输入
            # 注意这里imgs是一个可迭代对象，sorted会自动进行迭代，key的输入值来源于迭代的imgs
            # 返回的独立于原序列的新的序列
        imgs_num = len(imgs)

        if self.test:
            self.imgs = imgs
        elif train:
            self.imgs = imgs[:int(0.7 * imgs_num)]
        # 不是训练，不是测试，剩下验证
        else:
            self.imgs = imgs[int(0.7 * imgs_num):]
        # 以上完成对数据集的选取，确定要读取的数据时做何用，并进行相应的地址的提取，得到顺序排列的地址列表
        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            #归一化是必须要做的
            if self.test or not train:
                self.transforms = T.Compose([
                    T.Resize(224),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                ])
            else:
                self.transforms = T.Compose([
                    T.Resize(256),
                    T.RandomResizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                ])

    def __getitem__(self, index):#需要提供一个索引
        img_path = self.imgs[index]
        if self.test:
            label = int(self.imgs[index].split('.')[-2].split('/')[-1])
        else:
            label = 1 if 'dog' in img_path.split('/')[-1] else 0
            # label是通过文件名来得知的
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label  # 通过dataloder加载的数据格式
        # 将文件读取等费时操作放在此函数中，利用多进程加速(通过)

    def __len__(self):
        return len(self.imgs)

# root1='C:Users/28697/Desktop/AI/deep learning/cat.jpg'
# root2=r'C:\Users\28697\Desktop\AI\deep learning\cat.jpg'
# r'C:Users\28697\Desktop\AI\deep learning\cat.jpg'
# train_dataset=DogCat(r'C:\Users\28697\Desktop\AI\deep learning\.idea')
# trainloader
