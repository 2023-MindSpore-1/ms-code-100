import argparse
import os
import numpy as np

from dataloader_d import *
from supervisor_pretrain import *
from supervisor_train import *

def main(args):
    ''' main function for data imputation and super-resolution with pretrain-stage and train-stage

    '''

    # data path
    impbj = args.data_path + "BJtaxi_15year_uncmp(16x16)_40.npy"
    mpbj = args.data_path + "BJtaxi_15year(16x16).npy"
    srmpbj = args.data_path + "BJtaxi_15year(32x32).npy"
    fea = args.data_path + "BJ_feature.npy"
    p = args.data_path + "BJ_POI_normalize(13x16x16).npy"

    # dataloader
    poi = np.load(p, allow_pickle=True).astype(np.float32)
    # 判断weather是否存在
    if os.path.exists(fea):   
        feature = np.load(fea, allow_pickle=True).astype(np.float32)
    else:
        feature = None
    pretrain_d = get_dataloader(args.batch, args.timeslot, args.val_per, args.test_per, mpbj, impbj, fea)     # pretrain[3]: train, val, test; train[3]: x, y, feature(optional)
    train_d = get_dataloader(args.batch, args.timeslot, args.val_per, args.test_per, srmpbj, impbj, fea)      # train[3]: train, val, test; train[3]: x, y, feature(optional)

    # pretrain-stage
    pretrain_model = ModelSupervisor_pretrain(channel=2, kernel_size=args.kernel_size, timeslot=args.timeslot, feature=feature, poi=poi)
    pretrain_model.train(iteration=args.iterations, lr=args.cmp_lr, best_rmse=args.best_rmse, pretrain_d=pretrain_d, save_model=args.pretrain_model)

    # train-stage
    train_model = ModelSupervisor_train(channel=2, channel_n=args.n_channel, kernel_size=args.kernel_size, timeslot=args.timeslot, scaler_n=args.scaler_n, resnet_n=args.resnet_n, step=args.step, feature=feature, poi=poi)
    train_model.train(iteration=args.iterations, cmp_lr=args.cmp_lr, sr_lr=args.sr_lr, best_rmse=args.best_rmse, pretrain_d=pretrain_d, train_d=train_d, pretrain_model=args.pretrain_model, pretrain_model_save=args.pretrain_model_save, train_model_save=args.train_model_save)
    



if __name__=='__main__':
    # inputs for main function
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel_size', default=3, type=int)
    parser.add_argument('--timeslot', help='the number of LFN block in horizontal', default=5, type=int)
    parser.add_argument('--n_channel', help='the number of channel inner the model', default=32, type=int)
    parser.add_argument('--step', help='the layer of sr block', default=3, type=int)
    parser.add_argument('--resnet_n', help='the number of resnet block', default=16, type=int)
    parser.add_argument('--scaler_n', help='the upscaling factor', default=2, type=int)

    parser.add_argument('--horizon', default=0, type=int)
    parser.add_argument('--val_per', help='the ratio of validation dataset', default=0.2, type=float)
    parser.add_argument('--test_per', help='the ratio of test dateset', default=0.1, type=float)

    parser.add_argument('--batch', default=16, type=int)
    parser.add_argument('--iterations', default=10, type=int)
    parser.add_argument('--cmp_lr', default=1e-4, type=float)
    parser.add_argument('--sr_lr', default=5e-3, type=float)
    parser.add_argument('--best_rmse', default=100, type=int)

    parser.add_argument('--imp_ratio', help='the ratio of missing data', default=0.4, type=float)
    parser.add_argument('--whe_weather', help='whether there is weather external feature', default=True)
    parser.add_argument('--dataset', help='the name of dataset', default='BJtaxi_15year', type=str)

    parser.add_argument('--data_path', default='../MTCSR/data/', type=str)
    parser.add_argument('--pretrain_model', default='../MTCSR/model/pretrain_model.pkl', type=str)
    parser.add_argument('--pretrain_model_save', default='../MTCSR/model/train_model_cmp.pkl', type=str)
    parser.add_argument('--train_model_save', default='../MTCSR/model/train_model_sr.pkl', type=str)

    arg = parser.parse_args()

    # calls main function
    main(args=arg)


