'''import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image

class sourceDataSet(data.Dataset):
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

        self.id_to_trainid = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
        #self.id_to_trainid = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17:17, 18 : 18, 19: 19}
        #self.id_to_trainid = {0: 255, 1: 0, 2: 1, 3: 2, 4: 3}

        for name in self.img_ids:
            if '\n' in name:
                name = name.replace('\n','')

            img_name = name.split(' ')[0]
            label_name = name.split(' ')[1]

            img_file = osp.join(self.root, "%s" % img_name)
            label_file = osp.join(self.root, "%s" % label_name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": img_name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        datafiles = self.files[index]
        #print(datafiles)

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]

        #image_cropped = image.resize(self.crop_size, Image.BICUBIC)
        #label_cropped = label.resize(self.crop_size, Image.NEAREST)

        # random cropping

        crop_height, crop_width = self.crop_size
        width = image.width
        height = image.height

        #assert height >= crop_height and width >= crop_width, "Size of the cropping must be smaller than the image"

        start_height = random.randint(0, height - crop_height)
        start_width = random.randint(0, width - crop_width)
        end_height = start_height + crop_height
        end_width = start_width + crop_width

        image_cropped = image.crop(box = (start_width, start_height, end_width, end_height))
        label_cropped = label.crop(box = (start_width, start_height, end_width, end_height))

        image_cropped = np.asarray(image_cropped, np.float32)
        label_cropped = np.asarray(label_cropped, np.float32)

        # re-assign labels to match the format of Cityscapes
        label_copy = self.ignore_label * np.ones(label_cropped.shape, dtype=np.float32)

        for k, v in self.id_to_trainid.items():
            label_copy[label_cropped == k] = v

        size = image_cropped.shape
        image_cropped = image_cropped[:, :, ::-1]  # change to BGR
        image_cropped -= self.mean
        image_cropped = image_cropped.transpose((2, 0, 1))

        # EDITTED by me
        #label_copy = np.transpose(label_copy, (1, 0))

        return image_cropped.copy(), label_copy.copy(), np.array(size), name


        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]

        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        # re-assign labels to match the format of Cityscapes
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), label_copy.copy(), np.array(size), name


if name == 'main':
    dst = sourceDataSet("./data", is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()'''





import os
import os.path as osp
import numpy as np
import random
import cv2 as cv
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image

class sourceDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), random_rotate=True, random_flip=True, random_lighting=True, ignore_label=0):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.random_rotate = random_rotate
        self.random_lighting = random_lighting

        if random_lighting:
            crop_height, crop_width = self.crop_size
            self.gaussian = np.random.random((crop_height, crop_width, 1)).astype(np.float32)
            self.gaussian = np.concatenate((self.gaussian, self.gaussian, self.gaussian), axis = 2)

        self.ignore_label = ignore_label
        self.mean = mean
        self.random_flip = random_flip
        self.img_ids = [i_id for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        self.id_to_trainid = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
        #self.id_to_trainid = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17:17, 18 : 18, 19: 19}
        #self.id_to_trainid = {0: 255, 1: 0, 2: 1, 3: 2, 4: 3}

        for name in self.img_ids:
            if '\n' in name:
                name = name.replace('\n','')

            img_name = name.split(' ')[0]
            label_name = name.split(' ')[1]

            img_file = osp.join(self.root, "%s" % img_name)
            label_file = osp.join(self.root, "%s" % label_name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": img_name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]

        width = image.width
        height = image.height

        # RANDOM CROPPING
        crop_height, crop_width = self.crop_size
        
        start_height = random.randint(0, height - crop_height)
        start_width = random.randint(0, width - crop_width)
        end_height = start_height + crop_height
        end_width = start_width + crop_width

        image_cropped = image.crop(box = (start_width, start_height, end_width, end_height))
        label_cropped = label.crop(box = (start_width, start_height, end_width, end_height))

        # RANDOM ROTATION
        if self.random_rotate:
            random.seed(self.seed)
            rotation = random.randint(0, 3)
            if rotation == 0:
                angle = Image.ROTATE_270
            elif rotation == 1:
                angle = Image.ROTATE_90
            elif rotation == 2:
                angle = Image.ROTATE_180

            if rotation == 0 or rotation == 1 or rotation == 2:
                image_cropped = image_cropped.transpose(angle)
                label_cropped = label_cropped.transpose(angle)

        # RANDOM FLIPPING
        if self.random_flip:
            flip_leftright = random.randint(0, 1)
            flip_updown = random.randint(0, 1)

            if flip_leftright:
                image_cropped = image_cropped.transpose(Image.FLIP_LEFT_RIGHT)
                label_cropped = label_cropped.transpose(Image.FLIP_LEFT_RIGHT)
            if flip_updown:
                image_cropped = image_cropped.transpose(Image.FLIP_TOP_BOTTOM)
                label_cropped = label_cropped.transpose(Image.FLIP_TOP_BOTTOM)

        image_cropped = np.asarray(image_cropped, np.float32)
        label_cropped = np.asarray(label_cropped, np.float32)

        # Random lighting
        if self.random_lighting:
            change_lighting = random.randint(0, 1)

            if change_lighting:
                image_cropped = cv.addWeighted(image_cropped, 0.75, 0.25 * self.gaussian, 0.25, 0)
                image_cropped = image_cropped.astype("uint8")
                image_cropped = image_cropped.astype("float32")

        # re-assign labels to match the format of Cityscapes
        label_copy = self.ignore_label * np.ones(label_cropped.shape, dtype=np.float32)

        for k, v in self.id_to_trainid.items():
            label_copy[label_cropped == k] = v

        size = image_cropped.shape
        image_cropped = image_cropped[:, :, ::-1]  # change to BGR
        image_cropped -= self.mean
        image_cropped = image_cropped.transpose((2, 0, 1))

        return image_cropped.copy(), label_copy.copy(), np.array(size), name

if __name__ == '__main__':
    dst = sourceDataSet("./data", is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()