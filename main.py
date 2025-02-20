
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
from option import parser
import csv
from torch import optim
from torch.utils.data import DataLoader
from torch.backends import cudnn
import torch.nn as nn
from torch.nn import DataParallel
import torch
from split_combine_mj import SplitComb
import time
import numpy as np
import data_loader as data
from importlib import import_module
import shutil
from trainval_classifier import train_casenet, val_casenet, simle_train, simple_val, simle_train_GD
from utils import Logger, save_itk, weights_init, debug_dataloader
import sys


import os


def main():
    """
    @description  : exploit insp_img or insp_and_exp_img to predict pseudo_map
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """

    global args
    args = parser.parse_args()
    torch.manual_seed(0)
    # torch.cuda.set_device(0)

    print('----------------------1.Load three Models------------------------')

    model = import_module(args.model)  # 相对导入，py文件名是可选的，可选择导入哪种模型文件
    config, net = model.get_model()
    # net = UNETR(in_channels=1, out_channels=1, img_size=(80,192,192), norm_name='instance')

    start_epoch = args.start_epoch
    save_dir = args.save_dir
    save_dir = os.path.join('results', save_dir)
    print("savedir: ", save_dir)
    print("args.lr: ", args.lr)
    args.lr_stage = config['lr_stage']
    args.lr_preset = config['lr']
    # args.ad_stage = config['multi_ad_stage']

    print('----------------------2.Load full or part of parameters of models------------------------')

    if args.resume:
        state_dict = torch.load(args.resume)['state_dict']
        net.load_state_dict(state_dict, strict=True)
    else:
        weights_init(net, init_type='xavier')  # weight initialization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    GpuNum = torch.cuda.device_count()
    print(GpuNum)

    net = DataParallel(net).to(device)

    if args.epochs is None:
        end_epoch = args.lr_stage[-1]  # 如果未给定总的迭代次数，结束时的迭代次数以lr_stage列表中最后一个为准
    else:
        end_epoch = args.epochs

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logfile = os.path.join(save_dir, 'log.txt')
    logger = SummaryWriter(log_dir=os.path.join(save_dir, 'board_log'))
    sys.stdout = Logger(logfile)  # 将命令行终端的输出同步打印到log文件中
    # pyfiles = [f for f in os.listdir('./') if f.endswith('.py')]
    # for f in pyfiles:
    # 	shutil.copy(f, os.path.join(save_dir, f))  #把当前文件夹中的py文件复制到savedir中

    cudnn.benchmark = True

    # 使用何种优化器：作者用的时Adam
    if not args.sgd:
        # 这里的学习率并不是真实的，后面会对其进行更改
        optimizer = optim.Adam(net.parameters(), lr=1e-5)

    else:
        optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

    print('---------------------------------3:Load Dataset--------------------------------')
    margin = args.cubesize
    marginv = args.cubesizev
    print('patch size ', margin)
    print('train stride ', args.stridet)
    # split_comber = SplitComb(args.stridet, margin)
    # modified by zhao:if use the pseudo label to train

    # trainset--online load
    # dataset_train = data.PRMData(
    #     config=config,
    #     split_comber=split_comber,
    #     phase='train')

    # # trainset--preload
    # dataset_train = data.PRM_preload(
    # 	config=config,
    # 	phase='train')
    # dataset_train = data.PRM_DS_Data(
    #     config=config,
    #     phase='train')
    # train_loader = DataLoader(
    #     dataset_train,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=args.workers,
    #     pin_memory=True)
    train_data = data.generator_dataset(config, phase='train')
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)
    # dataset_val = data.generator_dataset(config, phase='val')
    # val_loader = DataLoader(
    #     dataset_val,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=args.workers,
    #     pin_memory=True)
    # #load validation dataset
    # dataset_val = data.PRMData(
    # 	config=config,
    # 	split_comber = split_comber,
    # 	phase='val')

    # valset--preload
    # dataset_val = data.PRM_preload(
    # 	config=config,
    # 	phase='val')
    # val_loader = DataLoader(
    # 	dataset_val,
    # 	batch_size=args.batch_size,
    # 	shuffle=False,
    # 	num_workers=args.workers,
    # 	pin_memory=True)#锁页内存，当计算机的内存充足的时候，可以设置pin_memory=True

    # #load validation dataset
    # dataset_test = data.PRMData(
    # 	config=config,
    # 	split_comber = split_comber,
    # 	phase='test')

    # testset--preload
    # dataset_test = data.PRM_preload(
    # 	config=config,
    # 	phase='test')
    # test_loader = DataLoader(
    # 	dataset_test,
    # 	batch_size=args.batch_size,
    # 	shuffle=False,
    # 	num_workers=args.workers,
    # 	pin_memory=True)#

    print('--------------------------------------')

    total_epoch = []
    train_loss = []
    epoches_loss = []
    val_loss = []
    best_val_loss = 10
    test_loss = []

    train_acc = []
    val_acc = []
    test_acc = []

    train_sensi = []
    val_sensi = []
    test_sensi = []

    dice_train = []
    dice_val = []
    dice_test = []

    fpr_train = []
    fpr_val = []
    fpr_test = []

    logdirpath = os.path.join(save_dir, 'log')
    if not os.path.exists(logdirpath):
        os.mkdir(logdirpath)

    v_loss, mean_acc2, mean_sensiti2, mean_dice2, mean_fpr2 = 0, 0, 0, 0, 0
    te_loss, mean_acc3, mean_sensiti3, mean_dice3, mean_fpr3 = 0, 0, 0, 0, 0

    for epoch in range(start_epoch, end_epoch + 1):
        t_loss, mean_acc, mean_sensiti, mean_dice, mean_fpr = train_casenet(
            logger, epoch, device, net, train_loader, optimizer, args, save_dir)
        # epoches_loss.append(epoch_loss)

        train_loss.append(t_loss)
        train_acc.append(mean_acc)
        train_sensi.append(mean_sensiti)
        dice_train.append(mean_dice)

        fpr_train.append(mean_fpr)
        # # Save the current model
        if args.multigpu:  # 多GPU保存时使用net.module.state_dict()，防止单GPU加载时出现名称不匹配问题
            state_dict = net.module.state_dict()
        else:
            state_dict = net.state_dict()
        # for key in state_dict.keys():
        # 	state_dict[key] = state_dict[key].cpu()
        torch.save({
            'state_dict': state_dict,
            'args': args,
            'epoch': epoch},
            os.path.join(save_dir, 'latest.ckpt'))

        # Save the model frequently
        if epoch > 59 and epoch % args.save_freq == 0:
            if args.multigpu:
                state_dict_G = net.module.state_dict()

            else:
                state_dict_G = net.state_dict()

            # for key in state_dict.keys():
            # 	state_dict[key] = state_dict[key].cpu()
            torch.save({
                'state_dict': state_dict,
                'args': args},
                os.path.join(save_dir, '%03d.ckpt' % epoch))

        # if epoch % args.val_freq == 0 :
        # 	v_loss, mean_acc2, mean_sensiti2, mean_dice2,mean_fpr2 = val_casenet(epoch,device,net, val_loader, args, save_dir)
        # 	if v_loss < best_val_loss:
        # 		shutil.copyfile(os.path.join(save_dir, '%03d.ckpt' % epoch), os.path.join(save_dir, 'best_%03d.ckpt' % epoch))

        # if epoch % 20 == 0:
        # 	te_loss, mean_acc3, mean_sensiti3, mean_dice3, mean_fpr3 = val_casenet(epoch,device, net, test_loader, args, save_dir, test_flag=True)
        v_loss, mean_acc2, mean_sensiti2, mean_dice2, mean_fpr2 = 0, 0, 0, 0, 0
        te_loss, mean_acc3, mean_sensiti3, mean_dice3, mean_fpr3 = 0, 0, 0, 0, 0
        val_loss.append(v_loss)
        val_acc.append(mean_acc2)
        val_sensi.append(mean_sensiti2)
        dice_val.append(mean_dice2)

        fpr_val.append(mean_fpr2)

        test_loss.append(te_loss)
        test_acc.append(mean_acc3)
        test_sensi.append(mean_sensiti3)
        dice_test.append(mean_dice3)
        fpr_test.append(mean_fpr3)

        total_epoch.append(epoch)

        totalinfo = np.array([total_epoch, train_loss, val_loss, test_loss, train_acc, val_acc, test_acc,
                              train_sensi, val_sensi, test_sensi, dice_train, dice_val, dice_test,
                              fpr_train, fpr_val, fpr_test])
        np.save(os.path.join(logdirpath, 'log.npy'), totalinfo)
        np.save(os.path.join(logdirpath, 'log_loss.npy'),
                np.array(epoches_loss))

    logName = os.path.join(logdirpath, 'log.csv')
    with open(logName, 'a') as csvout:
        writer = csv.writer(csvout)
        row = ['train epoch', 'train loss', 'val loss', 'test loss', 'train acc', 'val acc', 'test acc',
               'train sensi', 'val sensi', 'test sensi', 'dice train', 'dice val', 'dice test',
               'fpr train', 'fpr val', 'fpr test']
        writer.writerow(row)

        for i in range(len(total_epoch)):
            row = [total_epoch[i], train_loss[i], val_loss[i], test_loss[i],
                   train_acc[i], val_acc[i], test_acc[i],
                   train_sensi[i], val_sensi[i], test_sensi[i],
                   dice_train[i], dice_val[i], dice_test[i],
                   fpr_train[i], fpr_val[i], fpr_test[i]]
            writer.writerow(row)
        csvout.close()

    print("Done")
    return


def Generator():
    """
    @description  :exploit insp_imge to generate exp_img (without discriminator)
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """

    global args
    args = parser.parse_args()
    torch.manual_seed(0)
    # torch.cuda.set_device(0)

    print('----------------------1.Load three Models------------------------')

    model = import_module(args.model)  # 相对导入，py文件名是可选的，可选择导入哪种模型文件
    config, net = model.get_model()
    # net = UNETR(in_channels=1, out_channels=1, img_size=(80,192,192), norm_name='instance')

    start_epoch = args.start_epoch
    save_dir = args.save_dir
    save_dir = os.path.join('results', save_dir)
    print("savedir: ", save_dir)
    print("args.lr: ", args.lr)
    args.lr_stage = config['lr_stage']
    args.lr_preset = config['lr']
    # args.ad_stage = config['multi_ad_stage']

    print('----------------------2.Load full or part of parameters of models------------------------')

    if args.resume:
        state_dict = torch.load(args.resume)['state_dict']
        net.load_state_dict(state_dict, strict=True)
    else:
        weights_init(net, init_type='xavier')  # weight initialization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # GpuNum = torch.cuda.device_count()
    # print("GPU_number", GpuNum)

    net = DataParallel(net)
    net.to(device)

    if args.epochs is None:
        end_epoch = args.lr_stage[-1]  # 如果未给定总的迭代次数，结束时的迭代次数以lr_stage列表中最后一个为准
    else:
        end_epoch = args.epochs

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logfile = os.path.join(save_dir, 'log.txt')
    logger = SummaryWriter(log_dir=os.path.join(save_dir, 'board_log'))
    sys.stdout = Logger(logfile)  # 将命令行终端的输出同步打印到log文件中
    # pyfiles = [f for f in os.listdir('./') if f.endswith('.py')]
    # for f in pyfiles:
    # 	shutil.copy(f, os.path.join(save_dir, f))  #把当前文件夹中的py文件复制到savedir中

    cudnn.benchmark = True

    # 使用何种优化器：作者用的时Adam
    if not args.sgd:
        # 这里的学习率并不是真实的，后面会对其进行更改
        optimizer = optim.Adam(net.parameters(), lr=1e-5)
    else:
        optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

    train_data = data.generator_dataset(config, phase='train')
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)
    dataset_val = data.generator_dataset(config, phase='val')
    val_loader = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    total_epoch = []
    train_loss = []
    epoches_loss = []
    val_loss = []
    test_loss = []

    logdirpath = os.path.join(save_dir, 'log')
    if not os.path.exists(logdirpath):
        os.mkdir(logdirpath)

    v_loss = 0
    te_loss = 0
    v_best_loss = 100

    logName = os.path.join(logdirpath, 'log.csv')
    with open(logName, 'a') as csvout:
        writer = csv.writer(csvout)
        row = ['train epoch', 'train loss', 'val loss', 'test loss']
        writer.writerow(row)

        for epoch in range(start_epoch, end_epoch + 1):

            t_loss = simle_train(
                logger, epoch, device, net, train_loader, args, optimizer)

            train_loss.append(t_loss)

            if args.multigpu:  # 多GPU保存时使用net.module.state_dict()，防止单GPU加载时出现名称不匹配问题
                state_dict = net.module.state_dict()
            else:
                state_dict = net.state_dict()
            # for key in state_dict.keys():
            # 	state_dict[key] = state_dict[key].cpu()
            torch.save({
                'state_dict': state_dict,
                'args': args},
                os.path.join(save_dir, 'latest.ckpt'))

            # Save the model frequently
            if epoch > 40 and epoch % args.save_freq == 0:
                if args.multigpu:
                    state_dict = net.module.state_dict()
                else:
                    state_dict = net.state_dict()
                # for key in state_dict.keys():
                # 	state_dict[key] = state_dict[key].cpu()
                torch.save({
                    'state_dict': state_dict,
                    'args': args},
                    os.path.join(save_dir, '%03d.ckpt' % epoch))

            if epoch > 40 and epoch % args.val_freq == 0:
                v_loss = simple_val(
                    logger, epoch, device, net, val_loader)
                if v_loss<v_best_loss:
                    v_best_loss = v_loss
                    if args.multigpu:
                        state_dict = net.module.state_dict()
                    else:
                        state_dict = net.state_dict()
                    # for key in state_dict.keys():
                    # 	state_dict[key] = state_dict[key].cpu()
                    torch.save({
                        'state_dict': state_dict,
                        'args': args,
                         "epoch": epoch},
                        os.path.join(save_dir, 'val_best.ckpt'))

            else:
                v_loss = 0
            # if epoch % 40 == 0:
            # 	te_loss, mean_acc3, mean_sensiti3, mean_dice3, mean_fpr3 = val_casenet(epoch,device, net, test_loader, args, save_dir, test_flag=True)

            val_loss.append(v_loss)
            test_loss.append(te_loss)

            total_epoch.append(epoch)

        totalinfo = np.array([total_epoch, train_loss, val_loss, test_loss])
        np.save(os.path.join(logdirpath, 'log.npy'), totalinfo)
        np.save(os.path.join(logdirpath, 'log_loss.npy'),
                np.array(epoches_loss))

        for i in range(len(total_epoch)):
            row = [total_epoch[i], train_loss[i], val_loss[i], test_loss[i]]
            writer.writerow(row)
        csvout.close()

    print("Done")
    val_loss_best_epoch = torch.load(os.path.join(save_dir, "val_best.ckpt"))[
        "epoch"
    ]
    os.rename(
        os.path.join(save_dir, "val_best.ckpt"),
        os.path.join(save_dir, "val_best_%03d.ckpt" % val_loss_best_epoch),
    )
    
    print(f"val_best_epoch:{val_loss_best_epoch}")
    return


def Generator_Discriminator():
    """
    @description  :exploit insp_imge to generate exp_img (with discriminator)
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    global args
    args = parser.parse_args()
    torch.manual_seed(0)
    # torch.cuda.set_device(0)

    print('----------------------1.Load three Models------------------------')

    model = import_module(args.model)  # 相对导入，py文件名是可选的，可选择导入哪种模型文件
    config, G_net, D_net = model.get_model()
    # net = UNETR(in_channels=1, out_channels=1, img_size=(80,192,192), norm_name='instance')

    start_epoch = args.start_epoch
    save_dir = args.save_dir
    save_dir = os.path.join('results', save_dir)
    print("savedir: ", save_dir)
    print("args.lr: ", args.lr)
    args.lr_stage = config['lr_stage']
    args.lr_preset = config['lr']
    # args.lr_stage_D = config['lr_stage']
    # args.lr_preset = config['lr']
    # args.ad_stage = config['multi_ad_stage']

    print('----------------------2.Load full or part of parameters of models------------------------')

    if args.resume:
        state_dict = torch.load(args.resume)['state_dict']
        G_net.load_state_dict(state_dict, strict=True)
    else:
        weights_init(G_net, init_type='xavier')  # weight initialization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # GpuNum = torch.cuda.device_count()
    # print("GPU_number", GpuNum)

    G_net = DataParallel(G_net).to(device)
    D_net = DataParallel(D_net).to(device)

    if args.epochs is None:
        end_epoch = args.lr_stage[-1]  # 如果未给定总的迭代次数，结束时的迭代次数以lr_stage列表中最后一个为准
    else:
        end_epoch = args.epochs

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logfile = os.path.join(save_dir, 'log.txt')
    logger = SummaryWriter(log_dir=os.path.join(save_dir, 'board_log'))
    sys.stdout = Logger(logfile)  # 将命令行终端的输出同步打印到log文件中
    # pyfiles = [f for f in os.listdir('./') if f.endswith('.py')]
    # for f in pyfiles:
    # 	shutil.copy(f, os.path.join(save_dir, f))  #把当前文件夹中的py文件复制到savedir中

    cudnn.benchmark = True

    # 使用何种优化器：作者用的时Adam
    if not args.sgd:
        # 这里的学习率并不是真实的，后面会对其进行更改
        optimizer_G = optim.Adam(G_net.parameters(), lr=1e-5)
        optimizer_D = optim.Adam(D_net.parameters(), lr=0.0001)
    else:
        optimizer_G = optim.SGD(G_net.parameters(), lr=1e-3, momentum=0.9)
        optimizer_D = optim.SGD(D_net.parameters(), lr=1e-3, momentum=0.9)

    train_data = data.generator_dataset(config, phase='train')
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)
    dataset_val = data.generator_dataset(config, phase='val')
    val_loader = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    total_epoch = []
    train_loss = []
    epoches_loss = []
    val_loss = []
    test_loss = []

    logdirpath = os.path.join(save_dir, 'log')
    if not os.path.exists(logdirpath):
        os.mkdir(logdirpath)

    v_loss = 0
    v_best_loss = 100
    te_loss = 0

    logName = os.path.join(logdirpath, 'log.csv')
    with open(logName, 'a') as csvout:
        writer = csv.writer(csvout)
        row = ['train epoch', 'train loss', 'val loss', 'test loss']
        writer.writerow(row)

        for epoch in range(start_epoch, end_epoch + 1):

            t_loss = simle_train_GD(
                logger, epoch, device, G_net, D_net, train_loader, args, optimizer_G, optimizer_D)

            train_loss.append(t_loss)

            if args.multigpu:  # 多GPU保存时使用net.module.state_dict()，防止单GPU加载时出现名称不匹配问题
                state_dict_G = G_net.module.state_dict()
                state_dict_D = D_net.module.state_dict()
            else:
                state_dict_G = G_net.state_dict()
                state_dict_D = D_net.state_dict()
            # for key in state_dict.keys():
            # 	state_dict[key] = state_dict[key].cpu()
            torch.save({
                'state_dict': state_dict_G,
                'args': args,
                'epoch': epoch},
                os.path.join(save_dir, 'G_latest.ckpt'))
            torch.save({
                'state_dict': state_dict_D,
                'args': args,
                'epoch': epoch},
                os.path.join(save_dir, 'D_latest.ckpt'))

            # Save the model frequently
            if epoch > 59 and epoch % args.save_freq == 0:
                if args.multigpu:
                    state_dict_G = G_net.module.state_dict()
                    state_dict_D = D_net.module.state_dict()
                else:
                    state_dict_G = G_net.state_dict()
                    state_dict_D = D_net.state_dict()
                # for key in state_dict.keys():
                # 	state_dict[key] = state_dict[key].cpu()
                torch.save({
                    'state_dict': state_dict_G,
                    'args': args},
                    os.path.join(save_dir, 'G_%03d.ckpt' % epoch))

                torch.save({
                    'state_dict': state_dict_D,
                    'args': args},
                    os.path.join(save_dir, 'D_%03d.ckpt' % epoch))

            if epoch > 40 and epoch % args.val_freq == 0:
                v_loss = simple_val(
                    logger, epoch, device, G_net, val_loader)
                if v_loss<v_best_loss:
                    v_best_loss = v_loss
                    if args.multigpu:
                        state_dict_G = G_net.module.state_dict()
                        state_dict_D = D_net.module.state_dict()
                    else:
                        state_dict_G = G_net.state_dict()
                        state_dict_D = D_net.state_dict()
                    # for key in state_dict.keys():
                    # 	state_dict[key] = state_dict[key].cpu()
                    torch.save({
                        'state_dict': state_dict_G,
                        'args': args,
                         "epoch": epoch},
                        os.path.join(save_dir, 'G_best.ckpt'))

                    torch.save({
                        'state_dict': state_dict_D,
                        'args': args,
                         "epoch": epoch},
                        os.path.join(save_dir, 'D_best.ckpt'))
            else:
                v_loss = 0
            # if epoch % 40 == 0:
            # 	te_loss, mean_acc3, mean_sensiti3, mean_dice3, mean_fpr3 = val_casenet(epoch,device, net, test_loader, args, save_dir, test_flag=True)

            val_loss.append(v_loss)
            test_loss.append(te_loss)

            total_epoch.append(epoch)

        totalinfo = np.array([total_epoch, train_loss, val_loss, test_loss])
        np.save(os.path.join(logdirpath, 'log.npy'), totalinfo)
        np.save(os.path.join(logdirpath, 'log_loss.npy'),
                np.array(epoches_loss))

        for i in range(len(total_epoch)):
            row = [total_epoch[i], train_loss[i], val_loss[i], test_loss[i]]
            writer.writerow(row)
        csvout.close()

    print("Done")
    val_loss_best_epoch = torch.load(os.path.join(save_dir, "G_best.ckpt"))[
        "epoch"
    ]
    os.rename(
        os.path.join(save_dir, "G_best.ckpt"),
        os.path.join(save_dir, "G_best_%03d.ckpt" % val_loss_best_epoch),
    )
    os.rename(
        os.path.join(save_dir, "D_best.ckpt"),
        os.path.join(save_dir, "D_best_%03d.ckpt" % val_loss_best_epoch),
    )
    print(f"val_BD_best_epoch:{val_loss_best_epoch}")
    return


if __name__ == '__main__':
    #main()
    Generator()
    # Generator_Discriminator()
    # print(torch.cuda.is_available())
    # print(torch.__version__)
