import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image

class isprsDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=0):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.img_ids = [i_id for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        for name in self.img_ids:
            if '\n' in name:
                name = name.replace('\n','')
                
            img_name = name.split(' ')[0]
            img_file = osp.join(self.root, "%s" % img_name)
            self.files.append({
                "img": img_file,
                "name": img_name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        name = datafiles["name"]
        
      
        # random cropping
        crop_height, crop_width = self.crop_size
        width = image.width
        height = image.height

        #assert height <= crop_height and width <= crop_width, "Size of the cropping must be smaller than the image"

        start_height = random.randint(0, height - crop_height)
        start_width = random.randint(0, width - crop_width)
        end_height = start_height + crop_height
        end_width = start_width + crop_width

        image_cropped = image.crop(box = (start_width, start_height, end_width, end_height))

        image_cropped = np.asarray(image_cropped, np.float32)

        size = image_cropped.shape
        image_cropped = image_cropped[:, :, ::-1]  # change to BGR
        image_cropped -= self.mean
        image_cropped = image_cropped.transpose((2, 0, 1))

        return image_cropped.copy(), np.array(size), name
        '''
        
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        name = datafiles["name"]

        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)

        image = np.asarray(image, np.float32)

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), np.array(size), name'''


if __name__ == '__main__':
    dst = isprsDataSet("./data", is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
