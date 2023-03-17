import time
import numpy as np
import mindspore
import copy
from mindspore import nn, optim
from model import *

from dataloader_d import *

class ModelSupervisor_pretrain(nn.Module):
    def __init__(self, channel, kernel_size, timeslot, feature, poi, device=0):
        super(ModelSupervisor_pretrain, self).__init__()
        self.channel = channel
        self.kernel_size = kernel_size
        self.timeslot = timeslot
        self.device = device
        # initialize feature and poi infomation
        if feature is not None:
            self.feature = feature
            if self.feature.max() > 1:
                self.feature = self.feature / np.max(self.feature)
        self.poi = poi
        if self.poi.max() > 1:
            self.poi = self.poi / np.max(self.poi)

        self.cmpnet = CMPNet(channel, channel, kernel_size, timeslot, self.feature, self.poi, self.device)

    def train(self, iteration, lr, best_rmse, pretrain_d, save_model):
        Net = self.cmpnet.to(device = self.device)
        optimizer=optim.Adam(Net.parameters(), lr = lr)
        for epoch in range(iteration):
            start = time.time()
            loss_RMSE = []
            loss_MAE = []
            # training
            for i, sample in enumerate(pretrain_d[0]):
                # sample[3]: X, Y, Feature
                mask_train = copy.deepcopy(sample[0].to(self.device))
                mask_train[mask_train != 0] = 1
                data = sample[0].to(self.device)
                label = sample[1].to(self.device)

                output = Net(data, mask_train)

                loss_mae = self.mae_loss(output, label)
                loss_rmse = self.rmse_loss(output, label)
                loss_MAE.append(loss_mae.item())
                loss_RMSE.append(loss_rmse.item())

                optimizer.zero_grad()
                loss_rmse.backward()
                optimizer.step()
            t = time.time() - start
            result = 'training -- epoch: {}, time: {:.4f}, train_mae: {:.4f}, train_rmse: {:.4f} '.format(epoch, t, np.mean(loss_MAE), np.sqrt(np.mean(loss_RMSE)))
            print(result)

            # evaluating
            if (epoch+1) % 10 == 0:
                jdg, best_rmse = self.evaluate(Net, best_rmse, pretrain_d[1])
                if jdg == True:
                    print("best model!")
                    best_model = Net

            if epoch % 15 == 0 and epoch != 0:
                lr /= 2
                optimizer = optim.Adam(Net.parameters(), lr=lr)

        # testing
        self.test(best_model, pretrain_d[2])
        mindspore.save_checkpoint(best_model.state_dict(), save_model)

    def evaluate(self, Net, b_r, pretrain_val):
        jdg = False
        start = time.time()
        loss_RMSE_val = []
        loss_MAE_val = []
        for i, sample in enumerate(pretrain_val):
            mask_train = copy.deepcopy(sample[0].to(self.device))
            mask_train[mask_train != 0] = 1
            data = sample[0].to(self.device)
            label = sample[1].to(self.device)

            output = Net(data, mask_train)

            loss_mae_val = self.mae_loss(output, label)
            loss_rmse_val = self.rmse_loss(output, label)
            loss_MAE_val.append(loss_mae_val.item())
            loss_RMSE_val.append(loss_rmse_val.item())

        t = time.time() - start
        result = 'evaluating -- time: {:.4f}, val_mae: {:.4f}, val_rmse: {:.4f} '.format(t, np.mean(loss_MAE_val),
                                                                                           np.sqrt(
                                                                                               np.mean(loss_RMSE_val)))
        print(result)

        if np.sqrt(np.mean(loss_RMSE_val)) < b_r:
            jdg = True
            b_r = np.sqrt(np.mean(loss_RMSE_val))
        return jdg, b_r

    def test(self, net, pretrain_test):
        start = time.time()
        loss_RMSE_test = []
        loss_MAE_test = []
        for i, sample in enumerate(pretrain_test):
            mask_train = copy.deepcopy(sample[0].to(self.device))
            mask_train[mask_train != 0] = 1
            data = sample[0].to(self.device)
            label = sample[1].to(self.device)

            output = net(data, mask_train)

            loss_mae_test = self.mae_loss(output, label)
            loss_rmse_test = self.rmse_loss(output, label)
            loss_MAE_test.append(loss_mae_test.item())
            loss_RMSE_test.append(loss_rmse_test.item())

        t = time.time() - start
        result = 'testing -- time: {:.4f}, test_mae: {:.4f}, test_rmse: {:.4f} '.format(t, np.mean(loss_MAE_test),
                                                                                        np.sqrt(np.mean(loss_RMSE_test)))
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
