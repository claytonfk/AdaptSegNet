import argparse
import scipy
from scipy import ndimage
import cv2
import numpy as np
import sys

import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data
from model.deeplab_multi3 import Res_Deeplab
from dataset.val_dataset import valDataSet
from collections import OrderedDict
import os

import matplotlib.pyplot as plt
import torch.nn as nn
IMG_MEAN = np.array((72.3923987619416, 82.90891754262587, 73.15835921071157), dtype=np.float32) # cityscapes BGR

DATA_DIRECTORY = '/cluster/work/riner/users/zaziza/cityscapes/'
DATA_LIST_PATH = '/cluster/work/riner/users/zaziza/cityscapes/listsOfDataDirs/val_processed_20Classes_originalOrder_originalSize.txt'
IGNORE_LABEL = 0
NUM_CLASSES = 20
NUM_STEPS = 500 # Number of images in the validation set.
RESTORE_FROM = '/cluster/work/riner/users/zaziza/snapshots/AdaptSegNet/deeplab3/model_18.pth'

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    return parser.parse_args()


def get_iou(data_list, class_num, save_path=None):
    from multiprocessing import Pool 
    from metric import ConfusionMatrix

    ConfM = ConfusionMatrix(class_num)
    f = ConfM.generateM
    pool = Pool() 
    m_list = pool.map(f, data_list)
    pool.close() 
    pool.join() 
    
    for m in m_list:
        ConfM.addM(m)

    aveJ, j_list, M = ConfM.jaccard()
    print('meanIOU: ' + str(aveJ) + '\n')
    if save_path:
        with open(save_path, 'w') as f:
            f.write('meanIOU: ' + str(aveJ) + '\n')
            f.write(str(j_list)+'\n')
            f.write(str(M)+'\n')

'''def show_all(gt, pred):
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, axes = plt.subplots(1, 2)
    ax1, ax2 = axes

    classes = np.array(('background',  # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor'))
    colormap = [[0, 0, 0], [128, 64,128], [244, 35,232], [ 70, 70, 70], [102,102,156], [190,153,153], [153,153,153], [250,170, 30], [220,220, 0], [107,142, 35], [152,251,152], [ 70,130,180], [220, 20, 60], [255, 0, 0], [0, 0,142], [0, 0, 70], [0, 60,100], [0,80,100], [0, 0,230], [119, 11, 32]]
    cmap = colors.ListedColormap(colormap)
    bounds=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    ax1.set_title('gt')
    ax1.imshow(gt, cmap=cmap, norm=norm)

    ax2.set_title('pred')
    ax2.imshow(pred, cmap=cmap, norm=norm)

    plt.show()'''

def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()

    gpu0 = args.gpu

    model = Res_Deeplab(num_classes=args.num_classes)
    
    saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict(saved_state_dict)

    model.eval()
    model.cuda(gpu0)

    testloader = data.DataLoader(valDataSet(args.data_dir, args.data_list, crop_size=(550, 550), mean=IMG_MEAN, scale=False, mirror=False), 
                                    batch_size=1, shuffle=False, pin_memory=True)

    interp = nn.Upsample(size=(550, 550), mode='bilinear')
    data_list = []

    for index, batch in enumerate(testloader):
        if index % 100 == 0:
            print('%d processd'%(index))
        image, label, size, name = batch
        size = size[0].numpy()
        output = model(Variable(image, volatile=True).cuda(gpu0))
        output = interp(output).cpu().data[0].numpy()

        output = output[:,:size[0],:size[1]]
        gt = np.asarray(label[0].numpy()[:size[0],:size[1]], dtype=np.int)
        
        output = output.transpose(1,2,0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.int)

        # show_all(gt, output)
        data_list.append([gt.flatten(), output.flatten()])

    get_iou(data_list, args.num_classes)


if __name__ == '__main__':
    main()