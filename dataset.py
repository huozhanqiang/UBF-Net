import os
import cv2
import numpy as np
import torch
import random
import torch.utils.data as data

from option import args


class MEFdataset(data.Dataset):
    def __init__(self, transform):
        super(MEFdataset, self).__init__()
        self.dir_prefix = args.dir_train

        self.under_1 = os.listdir(self.dir_prefix + 'under_1/')
        self.over_1_5 = os.listdir(self.dir_prefix + 'over_1.5/')
        self.over_gc = os.listdir(self.dir_prefix + 'over_gc/')
        self.over_cpahe = os.listdir(self.dir_prefix + 'over_cpahe/')

        self.patch_size = args.patch_size
        self.transform = transform

    def __len__(self):
        assert len(self.under_1) == len(self.over_1_5) == len(self.over_gc) == len(self.over_cpahe)
        return len(self.under_1)

    def __getitem__(self, idx):
        under_1 = cv2.imread(self.dir_prefix + 'under_1/' + self.under_1[idx])
        under_1 = cv2.cvtColor(under_1, cv2.COLOR_BGR2YCrCb)
        under_1 = under_1[:, :, 0:1]

        over_1_5 = cv2.imread(self.dir_prefix + 'over_1.5/' + self.over_1_5[idx])
        over_1_5 = cv2.cvtColor(over_1_5, cv2.COLOR_BGR2YCrCb)
        over_1_5 = over_1_5[:, :, 0:1]

        over_gc = cv2.imread(self.dir_prefix + 'over_gc/' + self.over_gc[idx])
        over_gc = cv2.cvtColor(over_gc, cv2.COLOR_BGR2YCrCb)
        over_gc = over_gc[:, :, 0:1]

        over_cpahe = cv2.imread(self.dir_prefix + 'over_cpahe/' + self.over_cpahe[idx])
        over_cpahe = cv2.cvtColor(over_cpahe, cv2.COLOR_BGR2YCrCb)
        over_cpahe = over_cpahe[:, :, 0:1]

        under_1_p, over_1_5_p, over_gc_p, over_cpahe_p = self.get_patch(under_1, over_1_5, over_gc, over_cpahe)
        if self.transform:
            under_1_p = self.transform(under_1_p)
            over_1_5_p = self.transform(over_1_5_p)
            over_gc_p = self.transform(over_gc_p)
            over_cpahe_p = self.transform(over_cpahe_p)

        return under_1_p, over_1_5_p, over_gc_p, over_cpahe_p

    def get_patch(self, under1, over1_5, overgc, overcpahe):
        h, w = under1.shape[:2]
        stride = self.patch_size

        x = random.randint(0, w - stride)
        y = random.randint(0, h - stride)

        under1 = under1[y:y + stride, x:x + stride, :]
        over1_5 = over1_5[y:y + stride, x:x + stride, :]
        overgc = overgc[y:y + stride, x:x + stride, :]
        overcpahe = overcpahe[y:y + stride, x:x + stride, :]

        return under1, over1_5, overgc, overcpahe


class TestData(data.Dataset):
    def __init__(self, transform):
        super(TestData, self).__init__()
        self.transform = transform
        self.dir_prefix = args.dir_test
        self.under_1_dir = os.listdir(self.dir_prefix + 'under_1/')
        self.over_1_5_dir = os.listdir(self.dir_prefix + 'over_1.5/')
        self.over_gc_dir = os.listdir(self.dir_prefix + 'over_gc/')
        self.over_cpahe_dir = os.listdir(self.dir_prefix + 'over_cpahe/')

    def __getitem__(self, idx):
        under_1 = cv2.imread(self.dir_prefix + 'under_1/' + self.under_1_dir[idx])
        over_1_5 = cv2.imread(self.dir_prefix + 'over_1.5/' + self.over_1_5_dir[idx])
        over_gc = cv2.imread(self.dir_prefix + 'over_gc/' + self.over_gc_dir[idx])
        over_cpahe = cv2.imread(self.dir_prefix + 'over_cpahe/' + self.over_cpahe_dir[idx])

        under_1_img = cv2.cvtColor(under_1, cv2.COLOR_BGR2YCrCb)
        over_1_5_img = cv2.cvtColor(over_1_5, cv2.COLOR_BGR2YCrCb)
        over_gc_img = cv2.cvtColor(over_gc, cv2.COLOR_BGR2YCrCb)
        over_cpahe_img = cv2.cvtColor(over_cpahe, cv2.COLOR_BGR2YCrCb)

        if self.transform:
            under_1_img = self.transform(under_1_img)
            over_1_5_img = self.transform(over_1_5_img)
            over_gc_img = self.transform(over_gc_img)
            over_cpahe_img = self.transform(over_cpahe_img)

        img_stack = torch.stack((under_1_img, over_1_5_img, over_gc_img, over_cpahe_img), 0)
        return img_stack

    def __len__(self):
        assert len(self.under_1_dir) == len(self.over_1_5_dir) == len(self.over_gc_dir) == len(self.over_cpahe_dir)
        return len(self.under_1_dir)


class UNetTrainData(data.Dataset):
    def __init__(self, transform):
        super(UNetTrainData, self).__init__()
        self.unet_dir_prefix = args.unet_dir_train
        self.noise_dir = os.listdir(self.unet_dir_prefix)
        self.patch_size = args.patchsize
        self.transform = transform

    def __len__(self):
        return len(self.noise_dir)

    def __getitem__(self, idx):
        img = cv2.imread(self.unet_dir_prefix + self.noise_dir[idx])
        img = cv2.resize(img, (512, 512))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype('float32')
        H = img.shape[0]
        W = img.shape[1]
        if H - self.patch_size > 0:
            xx = np.random.randint(0, H - self.patch_size)
            img = img[xx:xx + self.patch_size, :, :]
        if W - self.patch_size > 0:
            yy = np.random.randint(0, W - self.patch_size)
            img = img[:, yy:yy + self.patch_size, :]
        if self.transform:
            img = self.transform(img)
        return img

    def get_patch(self, img):
        h, w = img.shape[:2]
        stride = self.patch_size
        x = random.randint(0, w - stride)
        y = random.randint(0, h - stride)
        imgp = img[y:y + stride, x:x + stride, :]
        return imgp


class UNetTestData(data.Dataset):
    def __init__(self, transform):
        super(UNetTestData, self).__init__()
        self.transform = transform
        self.unet_dir_prefix = args.unet_dir_test
        self.noise_dir = os.listdir(self.unet_dir_prefix)

    def __getitem__(self, idx):
        noise = cv2.imread(self.unet_dir_prefix + self.noise_dir[idx])
        # noise = cv2.cvtColor(noise, cv2.COLOR_BGR2RGB)
        noise = noise.astype('float32')
        if self.transform:
            noise = self.transform(noise)
        return noise

    def __len__(self):
        return len(self.noise_dir)



