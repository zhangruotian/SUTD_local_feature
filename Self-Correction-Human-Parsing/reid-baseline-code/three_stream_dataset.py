from torch.utils.data import Dataset
import os
import cv2
from PIL import Image
from torchvision import transforms
import numpy as np

def get_example(dir):
    samples = []
    for root, dirs, files in os.walk(dir, topdown=False):
        for name in files:
            if name.find('upper') != -1:
                upper_name, file_type = os.path.splitext(name)
                original_name = upper_name[0:-6] + file_type
                label = name[0:4]
                lower_name=upper_name[0:-6]+'_lower'+file_type
                upper_name_new=upper_name+file_type
                full_upper_name=os.path.join(root,upper_name_new)
                full_lower_name = os.path.join(root, lower_name)
                full_original_name=os.path.join(root,original_name)
                samples.append((full_original_name, full_upper_name, full_lower_name,label))
    return samples


class ThreeStreamDataset(Dataset):
    def __init__(self,dir,*transforms):
        self.dir=dir
        self.transforms=transforms
        self.examples=get_example(self.dir)
        classes, class_to_idx = self._find_classes(self.dir)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.imgs=self.examples

    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index):
        original_name, upper_name, lower_name,label=self.examples[index]
        original_data=Image.open(original_name)
        upper_data=Image.open(upper_name)
        lower_data=Image.open(lower_name)
        if self.transforms:
            original_data=self.transforms[0](original_data)
            upper_data=self.transforms[1](upper_data)
            lower_data=self.transforms[1](lower_data)
            upper_data[upper_data!=0]=1
            lower_data[lower_data!=0]=1
        return (original_data,upper_data,lower_data),self.class_to_idx[label]


    def __len__(self):
        return len(self.examples)

if __name__ == '__main__':

    transform_train_list = [
        transforms.Resize((384,192), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    transform_parts_list=[
        transforms.Resize((24,12),interpolation=3),
        transforms.ToTensor()
    ]
    transform_val_list = [
        transforms.Resize(size=(384,192),interpolation=3), #Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    data_transforms = {
        'train': transforms.Compose(transform_train_list),
        'parts': transforms.Compose(transform_parts_list),
        'val': transforms.Compose(transform_val_list),
    }
    data = ThreeStreamDataset('../example3/pytorch_ori_upper_lower/train',data_transforms['train'], data_transforms['parts'])
    print(data[0][0][1])
    b=np.array(data[0][0][1]*50000)
    b=b.squeeze()
    c=np.array(data[0][0][2]*50000)
    c=c.squeeze()
    cv2.imwrite('b.png',b)
    cv2.imwrite('c.png',c)