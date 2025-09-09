import os
import math
from libtiff import TIFF
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

def read_image(image_path):
    images = []
    tif = TIFF.open(image_path, mode='r')
    for image in tif.iter_images():
        images.append(image)
    return np.array(images)

def save_image(image_path, image):
    tif = TIFF.open(image_path, mode='w')
    num = image.shape[0]
    for i in range(num):
        tif.write_image(((image[i]).astype(np.uint8)), compression=None)
    tif.close()
    return

class seg_dataset(Dataset):
    def __init__(self,dir_path,mode="train"):
        self.image_dir = os.path.join(dir_path, 'image')
        self.label_dir = os.path.join(dir_path, 'label')
        self.image_list = os.listdir(self.image_dir)
        self.mode = mode

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        if self.mode == 'train':
            image = read_image(os.path.join(self.image_dir, self.image_list[item]))
            # label_index = np.random.randint(len(self.label_list))
            label = read_image(os.path.join(self.label_dir, self.image_list[item]))
        elif self.mode == 'test':
            if 'BN' in self.image_dir:
                image = read_image(os.path.join(self.image_dir, self.image_list[item]))
            else:
                image = read_image(os.path.join(self.image_dir, self.image_list[item], self.image_list[item]))

            label_name = self.image_list[item] + '_label.tif_soma.tif'
            label = read_image(os.path.join(self.label_dir, label_name)) #_label.tif_soma.tif
        image = image / (image.max()+1e-5)
        label = label / (label.max()+1e-5)
        image = torch.from_numpy(image).float().unsqueeze(0)
        label = torch.from_numpy(label).float().unsqueeze(0)
        return image,label,self.image_list[item]

def cut_image(img, block_size=(32, 32, 32), step=(25, 32, 32), pad_model='reflect'):
    z_size, y_size, x_size = block_size
    z_step, y_step, x_step = step
    z_img, y_img, x_img = img.shape[2:5]

    z_max = math.ceil((z_img - z_size) / z_step) + 1
    y_max = math.ceil((y_img - y_size) / y_step) + 1
    x_max = math.ceil((x_img - x_size) / x_step) + 1

    max_num = [z_max, y_max, x_max]

    z_pad = (z_max - 1) * z_step + z_size - z_img
    y_pad = (y_max - 1) * y_step + y_size - y_img
    x_pad = (x_max - 1) * x_step + x_size - x_img

    if pad_model == 'constant':
        img = F.pad(img, (0, x_pad, 0, y_pad, 0, z_pad), 'constant', value=-1)
    elif pad_model == 'reflect':
        img = F.pad(img, (0, x_pad, 0, y_pad, 0, z_pad), 'reflect')
    print("pad: ",img.shape)
    image_blockes = []
    block_num = 0
    for xx in range(x_max):
        for yy in range(y_max):
            for zz in range(z_max):
                img_block = img[:, :, zz * z_step:zz * z_step + z_size,
                                      yy * y_step:yy * y_step + y_size,
                                      xx * x_step:xx * x_step + x_size]
                image_blockes.append(img_block)
                block_num = block_num + 1

    return image_blockes, block_num, max_num

def splice_image(img_block, block_num, max_num, image_size=(100, 1024, 1024), step=(25, 32, 32)):
    z_img, y_img, x_img = image_size
    z_num, y_num, x_num = max_num
    z_step, y_step, x_step = step
    print(image_size)
    zz = 0
    yy = 0
    xx = 0

    for i in range(block_num):
        img_block[i] = img_block[i][:, :, 0:z_step, 0:y_step, 0:x_step]
        if zz == 0:
            image_z = img_block[i]
        else:
            image_z = torch.cat([image_z, img_block[i]], dim=2)
        zz = zz + 1
        if zz == z_num:
            zz = 0
            if yy == 0:
                image_y = image_z
            else:
                image_y = torch.cat([image_y, image_z], dim=3)
            yy = yy + 1

        if yy == y_num:
            # print(image_x)
            yy = 0
            if xx == 0:
                image_x = image_y
            else:
                image_x = torch.cat([image_x, image_y], dim=4)
            xx = xx + 1

    image = image_x[:, :, 0:z_img, 0:y_img, 0:x_img]
    # print(image.shape)
    return image