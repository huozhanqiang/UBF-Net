import os
import cv2
import math
import time
import torch
import random
import matplotlib
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
from vgg import *
from utils import *
from option import args
from UNet import UNet
from pytorch_msssim import ssim
from dataset import UNetTrainData, UNetTestData
from torch.optim import SGD, Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter

import pyiqa

EPS = 1e-8


class UNetTrain(object):
    def __init__(self):
        self.num_epochs = args.epochs
        self.lr = args.learingrate

        self.transform = transforms.Compose([transforms.ToTensor()])
        self.train_set = UNetTrainData(self.transform)
        self.train_loader = data.DataLoader(self.train_set, batch_size=args.batchsize,
                                            shuffle=True, num_workers=0, pin_memory=False)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.Block1 = UNet().cuda(self.device)
        # self.optimizer = Adam(self.Block1.parameters(), lr=self.lr)
        self.optimizer = SGD(self.Block1.parameters(), lr=self.lr)
        self.ratio = self.num_epochs / 100
        self.noise_adder = AugmentNoise(style=args.noisetype)
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer,
                                                  milestones=[
                                                      int(20 * self.ratio) - 1,
                                                      int(40 * self.ratio) - 1,
                                                      int(60 * self.ratio) - 1,
                                                      int(80 * self.ratio) - 1
                                                  ],
                                                  gamma=0.5)
        self.loss_mse = nn.MSELoss(reduction='mean').cuda(self.device)


        self.train_loss_1 = []

        if args.validation:
            self.val_list = []
            self.best_psnr = 0

    def train(self):
        seed = args.seed
        random.seed(seed)
        torch.manual_seed(seed)
        writer = SummaryWriter(log_dir=args.unet_log_dir, filename_suffix='train_loss')

        if os.path.exists(args.unet_model_path + args.unet_model):
            print('===> Loading pre-trained model......')
            state = torch.load(args.unet_model_path + args.unet_model)
            self.Block1.load_state_dict(state['model_noise'])

            self.train_loss_1 = state['train_loss_1']


        for ep in range(self.num_epochs):
            ep_loss_1 = []

            for batch, clean_img in enumerate(self.train_loader):

                clean_img = clean_img / 255.0
                clean_img = clean_img.cuda(self.device)
                noisy_img = self.noise_adder.add_train_noise(clean_img)


                self.optimizer.zero_grad()

                mask1, mask2 = generate_mask_pair(noisy_img)
                noisy_sub1 = generate_subimages(noisy_img, mask1)
                noisy_sub2 = generate_subimages(noisy_img, mask2)
                torch.cuda.synchronize()
                start_time = time.time()
                noisy_denoised = self.Block1(noisy_img)

                torch.cuda.synchronize()
                end_time = time.time()
                Lambda = ep / args.epochs * args.increase_ratio
                noisy_sub1_denoised = generate_subimages(noisy_denoised, mask1)
                noisy_sub2_denoised = generate_subimages(noisy_denoised, mask2)

                noisy_output = self.Block1(noisy_sub1)
                noisy_target = noisy_sub2
                diff = noisy_output - noisy_target
                exp_diff = noisy_sub1_denoised - noisy_sub2_denoised

                loss1 = torch.mean(diff ** 2)
                loss2 = Lambda * torch.mean((diff - exp_diff) ** 2)
                loss_n2n = args.Lambda1 * loss1 + args.Lambda2 * loss2

                loss_all = loss_n2n

                ep_loss_1.append(loss_n2n.item())

                loss_all.backward()
                self.optimizer.step()

                if batch % 50 == 0 and batch != 0:
                    print('Epoch:{}\tcur/all:{}/{}\tLoss_n2n:{:.4f}'
                          'Time:{:.2f}s'.format(ep + 1, batch,
                                                len(self.train_loader),
                                                loss_n2n.item(),
                                                end_time - start_time))

            self.scheduler.step()
            print(np.mean(ep_loss_1))
            self.train_loss_1.append(np.mean(ep_loss_1))

            state = {
                'model_noise': self.Block1.state_dict(),
                'train_loss_1': self.train_loss_1,
                'lr_1': self.optimizer.param_groups[0]['lr'],
            }
            torch.save(state, args.unet_model_path + args.unet_model)
            if ep % 5 == 0:
                torch.save(state, args.unet_model_path + str(ep) + '.pth')
            matplotlib.use('Agg')
            fig1 = plt.figure()
            plot_loss_list_1 = self.train_loss_1
            plt.plot(plot_loss_list_1)
            plt.savefig('n2n_loss_curve_1.png')

            writer.add_scalar('N2N_loss', np.mean(ep_loss_1), ep)

        print('===> Finished Training!')


class UNetTest(object):
    def __init__(self, ep=None):
        self.ep = ep
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.batch_size = 1
        self.test_set = UNetTestData(self.transform)
        self.test_loader = data.DataLoader(self.test_set, batch_size=1, shuffle=False,
                                           num_workers=0, pin_memory=False)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = UNet().cuda(self.device)
        self.state = torch.load(args.unet_model_path + args.unet_model, map_location='cuda:0')
        self.model.load_state_dict(self.state['model_noise'])
        self.iqa_metric = pyiqa.create_metric('nima').to(self.device)

    def test(self):
        self.model.eval()
        print(self.iqa_metric.lower_better)
        niqe_all_result = 0
        count = 0
        with torch.no_grad():
            for batch, noise in enumerate(self.test_loader):
                print('Processing picture No.{}'.format(batch + 1))

                noise = torch.squeeze(noise, dim=0)
                noise = noise.cpu().numpy()
                noise = noise.transpose((1, 2, 0))
                noise = noise.astype('float32') / 255.0

                H = noise.shape[0]
                W = noise.shape[1]
                val_size = (max(H, W) + 31) // 32 * 32
                noisy_im = np.pad(
                    noise,
                    [[0, val_size - H], [0, val_size - W], [0, 0]],
                    'reflect')

                noisy_im = self.transform(noisy_im)
                noisy_im = torch.unsqueeze(noisy_im, 0)
                noisy_im = noisy_im.cuda()
                with torch.no_grad():
                    noisy_denoised = self.model(noisy_im)
                    noisy_denoised = noisy_denoised[:, :, :H, :W]

                niqe_single = self.iqa_metric(noisy_denoised)
                noisy_denoised = noisy_denoised.permute(0, 2, 3, 1)
                noisy_denoised = noisy_denoised.cpu().numpy()
                noisy_denoised = noisy_denoised.squeeze()

                denoised_result = np.clip(noisy_denoised * 255.0 + 0.5, 0, 255).astype(np.uint8)

                if self.ep:
                    save_path = args.denoised_save_dir + str(self.ep) + '_epoch/'
                else:
                    save_path = args.denoised_save_dir

                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                print(niqe_single)
                niqe_all_result = niqe_all_result + niqe_single
                cv2.imwrite((save_path + str(batch + 1) + args.ext), denoised_result)
                count = count+1

            print('Finished testing!')
            print(niqe_all_result / count)


if __name__ == '__main__':
    # t = UNetTrain()
    # t.train()
    t = UNetTest()
    t.test()
