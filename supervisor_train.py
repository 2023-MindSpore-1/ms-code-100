import time
import numpy as np
import mindspore
from mindspore import nn, optim
import copy

from model import *
from dataloader_d import *

import warnings
warnings.filterwarnings("ignore")

class ModelSupervisor_train(nn.Module):
    def __init__(self, channel, channel_n, kernel_size, timeslot, scaler_n, resnet_n, step, feature, poi, device=0):
        super(ModelSupervisor_train, self).__init__()
        self.device= device

        if feature is not None:
            self.feature = feature
            if self.feature.max() > 1:
                self.feature = self.feature / np.max(self.feature)
        self.poi = poi
        if self.poi.max() > 1:
            self.poi = self.poi / np.max(self.poi)

        self.cmpnet = CMPNet(channel, channel, kernel_size, timeslot, self.feature, self.poi, self.device)
        self.sr = SRNet(channel, channel_n, scaler_n, resnet_n, step, poi.shape[-1])

    def train(self, iteration, cmp_lr, sr_lr, best_rmse, pretrain_d, train_d, pretrain_model, pretrain_model_save, train_model_save):
        # restore cmpnet
        net1 = self.cmpnet.to(device=self.device)
        net1.load_state_dict(mindspore.load_checkpoint(pretrain_model))
        net2 = self.sr.to(device=self.device)

        optimizer = optim.Adam([
            {'params': net2.parameters(), 'lr': sr_lr, 'betas': (0.9, 0.999)},
            {'params': net1.parameters(), 'lr': cmp_lr, 'betas': (0.9, 0.999)},
        ])

        for epoch in range(iteration):
            start = time.time()
            loss_RMSE = []
            loss_MAE = []
            # training
            for i, (pretrain, train) in enumerate(zip(pretrain_d[0], train_d[0])):
                # pretrain[3]: X, Y, Feature; train[3]: X, Y, Feature
                mask_train = copy.deepcopy(pretrain[0].to(self.device))
                mask_train[mask_train != 0] = 1
                label_pre = pretrain[1].to(self.device)

                data_tra = train[0].to(self.device)
                label_tra = train[1].to(self.device)
                # 判断是否存在 external features
                if len(train) == 3:
                    ext_tra = train[2].to(self.device)
                else:
                    ext_tra = None

                output1 = net1(data_tra, mask_train)
                output2 = net2(output1, ext_tra)

                loss_mae_pre = self.mae_loss(output1, label_pre)
                loss_mae = self.mae_loss(output2, label_tra)
                loss_rmse = self.rmse_loss(output2, label_tra)
                loss_MAE.append(loss_mae.item())
                loss_RMSE.append(loss_rmse.item())

                # multi
                total_mae = 0.0001 * loss_mae_pre + 0.1 * loss_mae
                optimizer.zero_grad()
                total_mae.backward(retain_graph=True)
                optimizer.step()

            t = time.time() - start
            result = 'epoch: {}, time: {:.4f}, train_mae: {:.4f}, train_rmse: {:.4f} '.format(epoch, t, np.mean(loss_MAE),
                                                                                            np.sqrt(np.mean(loss_RMSE)))
            print(result)

            # evaluating
            if (epoch+1) % 10 == 0:
                jdg, best_rmse = self.evaluate(net1, net2, best_rmse, train_d[1])
                if jdg == True:
                    print("best model!")
                    best_model_2 = net2
                    best_model_1 = net1
                # testing
                self.test(best_model_1, best_model_2, train_d[2])

            if epoch % 20 == 0 and epoch != 0:
               sr_lr /= 2
               optimizer = optim.Adam(net2.parameters(), lr=sr_lr)

        net1.load_state_dict(mindspore.load_checkpoint(pretrain_model_save))
        net2.load_state_dict(mindspore.load_checkpoint(train_model_save))

    def evaluate(self, net1, net2, b_r, train_val):
        jdg = False
        start = time.time()
        loss_RMSE = []
        loss_MAE = []
        for i, sample in enumerate(train_val):
            mask_train = copy.deepcopy(sample[0].to(self.device))
            mask_train[mask_train != 0] = 1

            data = sample[0].to(self.device)
            label = sample[1].to(self.device)
            if len(sample) == 3:
                ext = ext.to(self.device)
            else: 
                ext = None

            output1 = net1(data, mask_train)
            output2 = net2(output1, ext)

            loss_mae = self.mae_loss(output2, label)
            loss_rmse = self.rmse_loss(output2, label)
            loss_MAE.append(loss_mae.item())
            loss_RMSE.append(loss_rmse.item())

        t = time.time() - start
        result = 'evaluating -- time: {:.4f}, val_mae: {:.4f}, val_rmse: {:.4f} '.format(t, np.mean(loss_MAE),
                                                                                           np.sqrt(np.mean(loss_RMSE)))
        print(result)
        if np.sqrt(np.mean(loss_RMSE)) < b_r:
            jdg = True
            b_r = np.sqrt(np.mean(loss_RMSE))
        return jdg, b_r

    def test(self, net1, best_model, train_test):
        start = time.time()
        loss_RMSE = []
        loss_MAE = []
        for i, sample in enumerate(train_test):
            mask_train = copy.deepcopy(sample[0].to(self.device))
            mask_train[mask_train != 0] = 1

            data = sample[1].to(self.device)
            label = sample[1].to(self.device)
            if len(sample) == 3:
                ext = ext.to(self.device)
            else: 
                ext = None

            output1 = net1(data, mask_train)
            output2 = best_model(output1, ext)

            loss_mae = self.mae_loss(output2, label)
            loss_rmse = self.rmse_loss(output2, label)
            loss_MAE.append(loss_mae.item())
            loss_RMSE.append(loss_rmse.item())

        t = time.time() - start
        result = 'testing -- time: {:.4f}, test_mae: {:.4f}, test_rmse: {:.4f} '.format(t, np.mean(loss_MAE),
                                                                                        np.sqrt(np.mean(loss_RMSE)))
        print(result)

    def mae_loss(self, y_pred, y_true):
        mask = (y_true != 0).to(self.device).float()
        mask /= mask.mean()
        loss = mindspore.ops.abs(y_pred - y_true)
        loss = loss * mask
        loss[loss != loss] = 0
        return loss.mean()

    def rmse_loss(self, y_pred, y_true):
        mask = (y_true != 0).to(self.device).float()
        mask /= mask.mean()
        loss = mindspore.ops.Pow((y_pred - y_true), 2)
        loss = loss * mask
        loss[loss != loss] = 0
        return loss.mean()
