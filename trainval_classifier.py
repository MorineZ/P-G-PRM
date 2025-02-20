
import numpy as np
import os
import time
import torch
from tqdm import tqdm
from utils import *
from loss import dice_loss, Adaptive_threshold_loss
from torch.cuda import empty_cache
import csv
from scipy.ndimage.interpolation import zoom
from skimage.transform import resize
from torch.nn import BCELoss
import torch.nn.functional as F
from split_combine_mj import SplitComb
from utils import combine_local_avg
from torch.utils.data import DataLoader
from monai.losses import FocalLoss
import torch.nn as nn
from grid import GatherGridsFromVolumes
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
# from sklearn.metrics import mean_absolute_error
th_bin = 0.5

# 学习率调整策略


def get_lr(epoch, args):
    """
    :param epoch: current epoch number
    :param args: global arguments args
    :return: learning rate of the next epoch
    """
    if args.lr is None:
        assert epoch <= args.lr_stage[-1]
        # 'lr_stage': np.array([10, 20, 40, 60])
        lrstage = np.sum(epoch > args.lr_stage)
        # args.lr_preset='lr': np.array([3e-3, 3e-4, 3e-5, 3e-6])
        lr = args.lr_preset[lrstage]
    else:
        lr = args.lr
    return lr
# add by zhao


def get_ad(epoch, args):
    """
    :param epoch: current epoch number
    :param args: global arguments args
    :return: gammaad of the epoch
    """

    # 'lr_stage': np.array([10, 20, 40, 60])
    adstage = np.sum(epoch > args.ad_stage)
    # args.lr_preset='lr': np.array([3e-3, 3e-4, 3e-5, 3e-6])
    gamma_ad = args.ad_preset[adstage]

    return gamma_ad


def get_pse_loss_lamta(epoch, args):
    """
    :param epoch: current epoch number
    :param args: global arguments args
    :return: gammaad of the epoch
    """

    # 'lr_stage': np.array([10, 20, 40, 60])
    # args.lr_preset='lr': np.array([3e-3, 3e-4, 3e-5, 3e-6])
    pse_lamta = epoch*1.0/args.epoch

    return pse_lamta

# 某一epoch内的训练运行，epoch为当前迭代次数


def train_casenet(logger, epoch, device, model, data_loader, optimizer, args, save_dir):
    """
    :param epoch: current epoch number
    :param model: CNN model
    :param data_loader: training data
    :param optimizer: training optimizer
    :param args: global arguments args
    :param save_dir: save directory
    :return: performance evaluation of the current epoch
    """
    net = model
    net.train()

    starttime = time.time()
    sidelen = args.stridet
    margin = args.cubesize
    # add by zhao
# 获取并更新当前迭代次数内的学习率
    lr = get_lr(epoch, args)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    assert (lr is not None)
    optimizer.zero_grad()
    logger.add_scalar("LR", lr, global_step=epoch)

    acc_total = []
    dice_hard_total = []
    sensitivity_total = []
    dice_total = []
    fpr_total = []
    lossHist = []
    thre_losshist = []
    # train_gather_fn = GatherGridsFromVolumes(
    #     192, grid_noise=0.01, uniform_grid_noise=True, label_interpolation_mode="nearest")

    for i,  (insp_img, exp_img, pseudo_map) in enumerate(tqdm(data_loader)):
        ###### Wrap Tensor##########
        # NameID = NameID[0]
        # SplitID = SplitID[0]
        batchlen = insp_img.size(0)

        # ###############################
        # coord = coord.cuda()
        # print(device)
        insp_img = insp_img.to(device)
        exp_img = exp_img.to(device)
        pseudo_map = pseudo_map.to(device)
        # _, grids, labels = train_gather_fn(pseudo_map, device)
        # casePreds = model(x)
        x = torch.cat([insp_img, exp_img], dim=1)
        pseudo_preds1, pseudo_preds2 = model(x)
        # pseudo_preds = torch.nn.Sigmoid()(pseudo_preds)
        # print(thres.shape)
        # thre_loss = Adaptive_threshold_loss(
        #     insp_img, exp_img, thres, pseudo_map, device)
        # pseudo_preds = nn.Softmax(dim=1)(pseudo_preds)
        loss = dice_loss(pseudo_preds1, pseudo_map) +\
            FocalLoss(to_onehot_y=False)(pseudo_preds1, pseudo_map)+dice_loss(pseudo_preds2, pseudo_map) +\
            FocalLoss(to_onehot_y=False)(pseudo_preds2, pseudo_map)
        # + thre_loss
        # print(f"thres: {thres}.thre_loss:{thre_loss.item()}")
        # pseudo_preds = nn.Softmax(dim=1)(pseudo_preds)
        # loss = dice_loss(pseudo_preds, pseudo_map) + \
        #     FocalLoss(to_onehot_y=False)(pseudo_preds, pseudo_map)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # for evaluation
        lossHist.append(loss.item())
        # thre_losshist.append(thre_loss.item())
        # segmentation calculating metrics#######################
        outdata = pseudo_preds2.cpu().data.numpy()
        segdata = pseudo_map.cpu().data.numpy()
        # segdata = (segdata > th_bin)

        for j in range(batchlen):
            dice = dice_coef_np(outdata[j], segdata[j])  # 概率预测与真值的dice
            sensiti = sensitivity_np(outdata[j], segdata[j])
            fpr = FalsePositiveRate(outdata[j], segdata[j])
            acc = acc_np(outdata[j], segdata[j])

            ##########################################################################
            dice_total.append(dice)

            fpr_total.append(fpr)
            sensitivity_total.append(sensiti)
            acc_total.append(acc)

    ##################################################################################

    endtime = time.time()
    lossHist = np.array(lossHist)  # 单次迭代所有batch下的loss列表
    thre_losshist = np.array(thre_losshist)
    mean_dice = np.mean(np.array(dice_total))  # 单次迭代所有batch下的平均dice

    mean_fpr = np.mean(np.array(fpr_total))
    mean_sensiti = np.mean(np.array(sensitivity_total))  # 单次迭代所有batch下的平均灵敏度
    mean_acc = np.mean(np.array(acc_total))  # 单次迭代所有batch下的平均准确率
    mean_loss = np.mean(lossHist)  # 本次迭代的训练集平均损失
    mean_thre_loss = np.mean(thre_losshist)

    logger.add_scalar("loss", mean_loss, global_step=epoch)
    logger.add_scalar("dice", mean_dice, global_step=epoch)
    logger.add_scalar("fpr", mean_fpr, global_step=epoch)
    logger.add_scalar("sensiti", mean_sensiti, global_step=epoch)
    logger.add_scalar("acc", mean_acc, global_step=epoch)
    print('Train, epoch %d, thre_loss %.4f, loss %.4f, accuracy %.4f, sensitivity %.4f, dice %.4f, fpr%.4f,time %3.2f,lr %.5f '
          % (epoch, mean_thre_loss, mean_loss, mean_acc, mean_sensiti, mean_dice, mean_fpr, endtime-starttime, lr))

    empty_cache()
    return mean_loss, mean_acc, mean_sensiti, mean_dice, mean_fpr


def val_casenet(epoch, device, model, data_loader, args, save_dir, test_flag=False):
    """
    :param epoch: current epoch number
    :param model: CNN model
    :param data_loader: evaluation and testing data
    :param args: global arguments args
    :param save_dir: save directory
    :param test_flag: current mode of validation or testing
    :return: performance evaluation of the current epoch
    """
    model.eval()
    starttime = time.time()

    sidelen = args.stridev
    if args.cubesizev is not None:
        margin = args.cubesizev
    else:
        margin = args.cubesize

    name_total = []
    lossHist = []

    dice_total = []
    dice_hard_total = []
    fpr_total = []
    sensitivity_total = []

    acc_total = []

    if test_flag:
        valdir = os.path.join(save_dir, 'test%03d' % (epoch))
        state_str = 'test'
    else:
        valdir = os.path.join(save_dir, 'val%03d' % (epoch))
        state_str = 'val'
    if not os.path.exists(valdir):
        os.mkdir(valdir)

    p_total = {}
    x_total = {}
    y_total = {}

    with torch.no_grad():

        for i, (x, y, org, spac, NameID, SplitID, nzhw, ShapeOrg) in enumerate(tqdm(data_loader)):
            torch.cuda.empty_cache()
            ###### Wrap Tensor##########
            NameID = NameID[0]
            SplitID = SplitID[0]
            batchlen = x.size(0)
            # print(batchlen)
            # x = x.cuda()
            # y = y.cuda()
            # ####################################################
            # coord = coord.cuda()
            x = x.to(device=device)
            y = y.to(device=device)
            ####################################################
            # coord = coord.to(device)
            # print(x.shape)
            casePreds = model(x)
            # print(y.shape)
            # print(casePreds.shape)
            if args.deepsupervision:

                # ds6, ds7, ds8 = casePreds[1], casePreds[2], casePreds[3]
                loss = dice_loss(casePreds, y)
            else:
                # casePred = casePreds
                loss = dice_loss(casePreds, y) + \
                    FocalLoss(to_onehot_y=False)(casePreds, y)

            # for evaluation
            lossHist.append(loss.item())

            ##################### seg data#######################
            outdata = casePreds.cpu().data.numpy()
            #######################################################################
            segdata = y.cpu().data.numpy()

            xdata = x.cpu().data.numpy()
            origindata = org.numpy()
            spacingdata = spac.numpy()

            #######################################################################
            ################# REARRANGE THE DATA BY SPLIT ID########################
            for j in range(batchlen):
                curxdata = (xdata[j]*255)
                curydata = segdata[j]
                segpred = outdata[j]
                # print(segpred.shape)
                curorigin = origindata[j].tolist()
                curspacing = spacingdata[j].tolist()
                cursplitID = int(SplitID[j])
                assert (cursplitID >= 0)
                curName = NameID[j]
                curnzhw = nzhw[j]
                curshape = ShapeOrg[j]

                if not (curName in x_total.keys()):
                    x_total[curName] = []
                if not (curName in y_total.keys()):
                    y_total[curName] = []
                if not (curName in p_total.keys()):
                    p_total[curName] = []

                # curxinfo = [curxdata, cursplitID, curnzhw, curshape, curorigin, curspacing]
                curyinfo = [curydata, cursplitID, curnzhw,
                            curshape, curorigin, curspacing]
                curpinfo = [segpred, cursplitID, curnzhw,
                            curshape, curorigin, curspacing]
                # x_total[curName].append(curxinfo)
                y_total[curName].append(curyinfo)
                p_total[curName].append(curpinfo)

            # torch.cuda.empty_cache()
    # combine all the cases together
    for curName in x_total.keys():
        curx = x_total[curName]
        cury = y_total[curName]
        curp = p_total[curName]
        # x_combine, xorigin, xspacing = combine_total(curx, sidelen, margin)
        y_combine, curorigin, curspacing = combine_total(cury, sidelen, margin)
        p_combine, porigin, pspacing = combine_total_avg(curp, sidelen, margin)
        # print(y_combine.shape)
        # print(p_combine.shape)
        p_combine_bw1 = np.argmax(p_combine, axis=0)
        y_combine1 = np.argmax(y_combine, axis=0)
        # print(p_combine_bw1.shape)
        # curpath = os.path.join(valdir, '%s-case-org.nii.gz'%(curName))
        curpredorgpath = os.path.join(valdir, '%s-pred_org.npy' % (curName))
        curpred3path = os.path.join(valdir, '%s-pred3.nii.gz' % (curName))

        # save_itk(x_combine.astype(dtype='uint8'), curorigin, curspacing, curpath)
        # save_itk(y_combine.astype(dtype='uint8'), curorigin, curspacing, curypath)
        save_itk(p_combine_bw1.astype(dtype='uint8'),
                 curorigin, curspacing, curpred3path)
        # p_combine_bw2 = p_combine_bw1.copy()
        # p_combine_bw2[np.where(p_combine_bw1==2)]=3
        # p_combine_bw2[np.where(p_combine_bw1==2)]=3
        np.save(curpredorgpath, p_combine)
        ########################################################################
        curdice = dice_coef_np(p_combine, y_combine)
        cur_hard_dice = dice_coef_np(p_combine_bw1, y_combine1, hard=True)

        curfpr = FalsePositiveRate(p_combine, y_combine)
        cursensi = sensitivity_np(p_combine, y_combine)
        curacc = acc_np(p_combine, y_combine)
        ########################################################################
        dice_total.append(curdice)
        dice_hard_total.append(cur_hard_dice)
        fpr_total.append(curfpr)
        acc_total.append(curacc)
        name_total.append(curName)
        sensitivity_total.append(cursensi)
        del cury, curp, y_combine, p_combine, p_combine_bw1, y_combine1

    endtime = time.time()
    lossHist = np.array(lossHist)

    # print(name_total)
    all_results = []
    with open(os.path.join(valdir, 'val_results.csv'), 'w') as csvout:
        writer = csv.writer(csvout)
        row = ['name', 'val acc', 'val sensi', 'val dice', 'val ppv']
        writer.writerow(row)

        for i in range(len(name_total)):

            row = [name_total[i], float(round(acc_total[i]*100, 3)), float(round(sensitivity_total[i]*100, 3)),
                   float(round(dice_total[i]*100, 3)), float(round(dice_hard_total[i]*100, 3)), float(round(fpr_total[i]*100, 3))]
            all_results.append([float(row[1]), float(row[2]), float(
                row[3]), float(row[4]), float(row[5])])
            writer.writerow(row)

        all_res_mean = np.mean(np.array(all_results), axis=0)
        all_res_std = np.std(np.array(all_results), axis=0)

        all_mean = ['all mean', all_res_mean[0], all_res_mean[1],
                    all_res_mean[2], all_res_mean[3], all_res_mean[4]]
        all_std = ['all std', all_res_std[0], all_res_std[1],
                   all_res_std[2], all_res_std[3], all_res_std[4]]

        writer.writerow(all_mean)
        writer.writerow(all_std)
        csvout.close()

    mean_dice = np.mean(np.array(dice_total))
    mean_hard_dice = np.mean(np.array(dice_hard_total))
    mean_fpr = np.mean(np.array(fpr_total))
    mean_sensiti = np.mean(np.array(sensitivity_total))
    mean_acc = np.mean(np.array(acc_total))
    mean_loss = np.mean(lossHist)
    print('%s, epoch %d, loss %.4f, accuracy %.4f, sensitivity %.4f, dice %.4f,dice_hard %.4f,fpr %.4f, time %3.2f'
          % (state_str, epoch, mean_loss, mean_acc, mean_sensiti, mean_dice, mean_hard_dice, mean_fpr, endtime-starttime))
    print()
    # empty_cache()
    return mean_loss, mean_acc, mean_sensiti, mean_dice, mean_fpr


def test_casenet(device, model, data_loader, args, save_dir, CT_dir):
    """
    :param epoch: current epoch number
    :param model: CNN model
    :param data_loader: evaluation and testing data
    :param args: global arguments args
    :param save_dir: save directory
    :param test_flag: current mode of validation or testing
    :return: performance evaluation of the current epoch
    """
    model.eval()
    starttime = time.time()

    sidelen = args.stridev
    if args.cubesizev is not None:
        margin = args.cubesizev
    else:
        margin = args.cubesize

    # name_total = []

    # dice_total = []
    # dice_hard_total = []
    # fpr_hard_total = []
    # fpr_total = []
    # sensitivity_total = []
    # sensitivity_hard_total = []
    # acc_total = []
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    p_total = {}
    # x_total = {}
    # y_total = {}

    with torch.no_grad():

        for i, (x, y, org, spac, NameID, SplitID, nzhw, ShapeOrg) in enumerate(tqdm(data_loader)):
            torch.cuda.empty_cache()
            ###### Wrap Tensor##########
            NameID = NameID[0]
            SplitID = SplitID[0]
            batchlen = x.size(0)
            x = x.to(device=device)
            y = y.to(device=device)
            ####################################################
            casePreds = model(x)
            # for evaluation
            ##################### seg data#######################
            outdata = casePreds.cpu().data.numpy()
            #######################################################################
            segdata = y.cpu().data.numpy()

            # xdata = x.cpu().data.numpy()
            origindata = org.numpy()
            spacingdata = spac.numpy()

            #######################################################################
            ################# REARRANGE THE DATA BY SPLIT ID########################
            for j in range(batchlen):
                # curxdata = (xdata[j]*255)
                curydata = segdata[j]
                segpred = outdata[j]
                # print(segpred.shape)
                curorigin = origindata[j].tolist()
                curspacing = spacingdata[j].tolist()
                cursplitID = int(SplitID[j])
                assert (cursplitID >= 0)
                curName = NameID[j]
                curnzhw = nzhw[j]
                curshape = ShapeOrg[j]

                # if not (curName in x_total.keys()):
                # 	x_total[curName] = []
                # if not (curName in y_total.keys()):
                # 	y_total[curName] = []
                if not (curName in p_total.keys()):
                    p_total[curName] = []

                # curxinfo = [curxdata, cursplitID, curnzhw, curshape, curorigin, curspacing]
                curyinfo = [curydata, cursplitID, curnzhw,
                            curshape, curorigin, curspacing]
                curpinfo = [segpred, cursplitID, curnzhw,
                            curshape, curorigin, curspacing]
                # x_total[curName].append(curxinfo)
                # y_total[curName].append(curyinfo)
                p_total[curName].append(curpinfo)
            # torch.cuda.empty_cache()
    # combine all the cases together
    for curName in p_total.keys():
        print(curName)

        # curx = x_total[curName]
        # cury = y_total[curName]
        curp = p_total[curName]
        # x_combine, xorigin, xspacing = combine_total(curx, sidelen, margin)
        # y_combine, curorigin, curspacing = combine_total(cury, sidelen, margin)
        p_combine, porigin, pspacing = combine_total_avg(curp, sidelen, margin)

        p_combine_bw1 = np.argmax(p_combine, axis=0)
        # y_combine1 =  np.argmax(y_combine,axis = 0)
        # print(p_combine_bw1.shape)
        # curpath = os.path.join(valdir, '%s-case-org.nii.gz'%(curName))
        # 1.->save the original prediction result
        curpredorgpath = os.path.join(save_dir, '%s-pred_org.npy' % (curName))
        # save_itk(p_combine, porigin, pspacing, curpredorgpath)
        np.save(curpredorgpath, p_combine)

        # 2.->save the argmax binary onehot prediction result
        curpred3path = os.path.join(save_dir, '%s-pred3.nii.gz' % (curName))
        save_itk(p_combine_bw1.astype(dtype='uint8'),
                 porigin, pspacing, curpred3path)

        # 3.->save the  prediction result of class 4 (not onehot)
        curpred4path = os.path.join(save_dir, '%s-pred4.nii.gz' % (curName))
        # save_itk(x_combine.astype(dtype='uint8'), curorigin, curspacing, curpath)
        # save_itk(y_combine.astype(dtype='uint8'), curorigin, curspacing, curypath)
        x_data, _, _, _ = load_itk_image(
            os.path.join(CT_dir, curName+"_304.nii.gz"))
        # label,_,_ = load_itk_image(os.path.join(CT_dir,curName+"_label4.nii.gz"))
        p_combine_bw2 = p_combine_bw1.copy()
        p_combine_bw2[np.where(p_combine_bw1 == 2)] = 3
        p_combine_bw2[np.where((p_combine_bw1 == 1) & (x_data > -950))] = 2
        save_itk(p_combine_bw2.astype(dtype='uint8'),
                 porigin, pspacing, curpred4path)
        ########################################################################
        # curdice = dice_coef_np(p_combine, y_combine,hard = False,avg = False)
        # cur_hard_dice =dice_coef_np(p_combine_bw2, label,hard = True,avg = False)

        # curfpr = FalsePositiveRate(p_combine, y_combine,hard = False,avg = False)
        # cur_hard_fpr = FalsePositiveRate(p_combine_bw2, label,hard = True,avg = False)

        # cursensi = sensitivity_np(p_combine, y_combine,hard = False,avg = False)
        # cur_hard_sensi = sensitivity_np(p_combine_bw2, label,hard = True,avg = False)

        # curacc = acc_np(p_combine, y_combine)
        # ########################################################################
        # dice_total.append(curdice)
        # dice_hard_total.append(cur_hard_dice)
        # fpr_total.append(curfpr)
        # fpr_hard_total.append(cur_hard_fpr)
        # acc_total.append(curacc)
        # name_total.append(curName)
        # sensitivity_total.append(cursensi)
        # sensitivity_hard_total.append(cur_hard_sensi)
        # del cury, curp, y_combine, p_combine,p_combine_bw1,y_combine1,p_combine_bw2

    endtime = time.time()

    # #print(name_total)
    # all_results = []
    # with open(os.path.join(save_dir, 'test_results.csv'), 'w') as csvout:
    # 	writer = csv.writer(csvout)
    # 	row = ['name', 'test acc', 'test_sensi_at','test_sensi_nor','test_sensi_em','test_sensi_fsad','test_sensi_nor',\
    # 		'test_dice_at','test_dice_nor','test_dice_em','test_dice_fsad','test_dice_nor',
    # 		'test_fpr_at','test_fpr_nor','test_fpr_em','test_fpr_fsad','test_fpr_nor',
    # 		]
    # 	writer.writerow(row)

    # 	for i in range(len(name_total)):

    # 		row = [name_total[i],float(round(acc_total[i]*100,3)),float(round(sensitivity_total[i][0]*100,3)),\
    # 		float(round(sensitivity_total[i][1]*100,3)),float(round(sensitivity_hard_total[i][0]*100,3)),
    # 		float(round(sensitivity_hard_total[i][1]*100,3)),float(round(sensitivity_hard_total[i][2]*100,3)),\

    # 		float(round(dice_total[i][0]*100,3)),float(round(dice_total[i][1]*100,3)),
    # 		float(round(dice_hard_total[i][0]*100,3)),float(round(dice_hard_total[i][1]*100,3)),float(round(dice_hard_total[i][2]*100,3)),

    # 		float(round(fpr_total[i][0]*100,3)),float(round(fpr_total[i][1]*100,3)),
    # 		float(round(fpr_hard_total[i][0]*100,3)),float(round(fpr_hard_total[i][1]*100,3)),float(round(fpr_hard_total[i][2]*100,3))]

    # 		all_results.append([float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]),
    # 		float(row[6]), float(row[7]), float(row[8]), float(row[9]), float(row[10]),
    # 		float(row[11]), float(row[12]), float(row[13]), float(row[14]), float(row[15]),float(row[16])
    # 		])
    # 		writer.writerow(row)

    # 	all_res_mean = np.mean(np.array(all_results),axis = 0)
    # 	all_res_std = np.std(np.array(all_results),axis = 0)

    # 	all_mean = ['all mean', all_res_mean[0], all_res_mean[1], all_res_mean[2],all_res_mean[3], all_res_mean[4],
    # 	all_res_mean[5], all_res_mean[6], all_res_mean[7],all_res_mean[8], all_res_mean[9],
    # 	all_res_mean[10], all_res_mean[11], all_res_mean[12],all_res_mean[13], all_res_mean[14],all_res_mean[15],
    # 	]
    # 	all_std = ['all std', all_res_std[0], all_res_std[1], all_res_std[2], all_res_std[3], all_res_std[4],
    # 	all_res_std[5], all_res_std[6], all_res_std[7], all_res_std[8], all_res_std[9],
    # 	all_res_std[10], all_res_std[11], all_res_std[12], all_res_std[13], all_res_std[14], all_res_std[15]]

    # 	writer.writerow(all_mean)
    # 	writer.writerow(all_std)
    # 	csvout.close()

    print()
    empty_cache()
    return None


def simle_train(logger, epoch, device, model, data_loader, args, optimizer):
    """
    :param epoch: current epoch number
    :param model: CNN model
    :param data_loader: training data
    :param optimizer: training optimizer
    :param args: global arguments args
    :param save_dir: save directory
    :return: performance evaluation of the current epoch
    """
    net = model
    net.train()
    lr = get_lr(epoch, args)
    # lr = 0.0001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    assert (lr is not None)
    optimizer.zero_grad()
    logger.add_scalar("LR", lr, global_step=epoch)
    crit = torch.nn.SmoothL1Loss()

    lossHist = []
    starttime = time.time()
    for i, (x, y) in enumerate(tqdm(data_loader)):
        # print(x.shape, y.shape)

        x, y = x.to(device), y.to(device)

        batchlen = x.size(0)

        casePreds = model(x)
       # casePred1, casePred2 = casePreds
        # print(f"before:{y.max()}")
        mask = torch.zeros_like(y).to(device)
        mask[torch.where(y != 0)] = 1
        # print(f"after:{y.max()}")
        # print(f"after_mask:{mask.max()}")
        loss = crit(casePreds, y)+crit(casePreds*mask, y)
        # loss = crit(casePreds*mask, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # model3_e= time.time()
        # #print("model_load_time",model_end-model_start)
        # # for evaluation
        # stu_start = time.time()
        lossHist.append(loss.item())
        # # lossHist1.append(loss1.item())
        # # lossHist2.append(loss2.item())
        # # lossHist3.append(loss3.item())
        # # # segmentation calculating metrics#######################
        # outdata = casePreds.cpu().data.numpy()
        # segdata = y.cpu().data.numpy()

    ##################################################################################

    endtime = time.time()
    lossHist = np.array(lossHist)  # 单次迭代所有batch下的loss列表

    mean_loss = np.mean(lossHist)  # 本次迭代的训练集平均损失
    logger.add_scalar("loss", mean_loss, global_step=epoch)
    # print('Train, epoch %d,  total loss %.4f,dice_focal_loss %.4f, deep_dense_loss %.4f,loss3 %.4f,accuracy %.4f, sensitivity %.4f, dice %.4f, dice % .4f,fpr%.4f,time %3.2f,lr %.5f '
    #       % (epoch, mean_loss, mean_loss1, mean_loss2, mean_loss3, mean_acc, mean_sensiti, mean_dice, mean_dice_hard, mean_fpr, endtime-starttime, lr))
    print('Train, epoch %d,  total loss %.4f,time %3.2f,lr %.5f '
          % (epoch, mean_loss, endtime-starttime, lr))
    # print('Train, epoch %d, dice loss %.4f, time %3.2f,lr %.5f '
    # 	  %(epoch, mean_loss,endtime-starttime,lr))
    # empty_cache()
    return mean_loss


def simple_val(logger, epoch, device, model, data_loader):
    """
    :param epoch: current epoch number
    :param model: CNN model
    :param data_loader: evaluation and testing data
    :param args: global arguments args
    :param save_dir: save directory
    :param test_flag: current mode of validation or testing
    :return: performance evaluation of the current epoch
    """
    model.eval()
    starttime = time.time()
    lossHist = []

    dice_total = []
    crit = torch.nn.SmoothL1Loss()

    # valdir = os.path.join(save_dir, 'val%03d' % (epoch))
    state_str = 'val'
    # if not os.path.exists(valdir):
    #     os.mkdir(valdir)
    with torch.no_grad():

        for i, (x, y) in enumerate(tqdm(data_loader)):

            x, y = x.to(device), y.to(device)

            casePreds = model(x)
            # casePred1, casePred2 = casePreds

            loss = crit(casePreds, y)
            lossHist.append(loss.item())

            ##################### seg data######################

    endtime = time.time()
    lossHist = np.array(lossHist)
    mean_loss = np.mean(lossHist)
    logger.add_scalar("val_loss", mean_loss, global_step=epoch)

    print('%s, epoch %d, loss %.7f,  time %3.2f'
          % (state_str, epoch, mean_loss,  endtime-starttime))
    print()
    empty_cache()
    return mean_loss


# def calculate_fid(act1, act2):
#     # calculate mean and covariance statistics
#     mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
#     mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
#     # calculate sum squared difference between means
#     ssdiff = np.sum((mu1 - mu2)**2.0)
#     # calculate sqrt of product between cov
#     covmean = sqrtm(sigma1.dot(sigma2))
#     # check and correct imaginary numbers from sqrt
#     if iscomplexobj(covmean):
#         covmean = covmean.real
#     # calculate score
#     fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
#     return fid


def simple_test(device, model, data_loader, save_dir, diff=False, insp_dir=None):
    """
    :param epoch: current epoch number
    :param model: CNN model
    :param data_loader: evaluation and testing data
    :param args: global arguments args
    :param save_dir: save directory
    :param test_flag: current mode of validation or testing
    :return: performance evaluation of the current epoch
    """
    model.eval()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    psnr_list = []
    ssim_list = []
    mae_list = []
    # fid_list = []
    total = []
    with torch.no_grad():

        for i, (name, origin, spacing, x, y) in enumerate(tqdm(data_loader)):
            # print(name)
            batch = x.shape[0]
            x = x.to(device=device)
            y = y.to(device=device)
            # ---------predict----------
            casePreds = model(x)
            outdata = casePreds.cpu().data.numpy()
            segdata = y.cpu().data.numpy()
            origin = origin.numpy()
            spacing = spacing.numpy()

            for j in range(batch):
                casename = name[j]

                pred = outdata[j, 0]
                target = segdata[j, 0]
                if diff:
                    savepath = os.path.join(
                        save_dir, casename+"_expw_pred_diff.nii.gz")

                    insp_img, _, _, _ = load_itk_image(os.path.join(
                        insp_dir, casename+"_ins_re969.nii.gz"))
                    # if no normalize,do not execute the following code
                    pred = pred*2-1
                    save_itk(pred, [0, 0, 0], [1, 1, 1], savepath)
                    # ---------------------------
                    expw_pred = insp_img+pred
                    # print(np.max(expw_pred))
                else:
                    expw_pred = pred
                savepath = os.path.join(save_dir, casename+"_expw_pred.nii.gz")
                save_itk(expw_pred, [0, 0, 0], [1, 1, 1], savepath)

                # ---------evaluation------
                # print(expw_pred.dtype, target.dtype)
                mae = np.mean(np.abs(expw_pred-target))
                mae_list.append(mae)
                psnr = peak_signal_noise_ratio(expw_pred, target)
                psnr_list.append(psnr)
                ssim = structural_similarity(expw_pred, target)
                ssim_list.append(ssim)
                total.append([casename, mae, psnr, ssim])

    mean_mae = np.mean(np.array(mae_list), axis=0)
    mean_psnr = np.mean(np.array(psnr_list), axis=0)
    mean_ssim = np.mean(np.array(ssim_list), axis=0)
    std_mae = np.std(np.array(mae_list), axis=0)
    std_psnr = np.std(np.array(psnr_list), axis=0)
    std_ssim = np.std(np.array(ssim_list), axis=0)

    with open(os.path.join(save_dir, 'test_results.csv'), 'w') as csvout:
        writer = csv.writer(csvout)
        row = ['name', 'mae', 'psnr', 'ssim']
        writer.writerow(row)
        for j in range(len(total)):
            writer.writerow(total[j])

        all_mean = ['all mean', mean_mae, mean_psnr, mean_ssim]
        all_std = ['all std', std_mae, std_psnr, std_ssim]

        writer.writerow(all_mean)
        writer.writerow(all_std)
        csvout.close()


def D_train(model_D, model_G, x, y, BCELoss, optimizer_D, device):
    """
    训练判别器
    :param D: 判别器
    :param G: 生成器
    :param X: 未分隔的数据
    :param BCELoss: 二分交叉熵损失函数
    :param optimizer_D: 判别器优化器
    :return: 判别器的损失值
    """
    # 标签转实物（右转左）

    xy = torch.cat([x, y], dim=1)  # 在channel维重叠 xy!=X
    # 梯度初始化为0
    optimizer_D.zero_grad()
    # 在真数据上
    D_output_r = model_D(xy)
    D_real_loss = BCELoss()(D_output_r, torch.ones(D_output_r.size()).to(device))
    # 在假数据上
    G_output = model_G(x)
    X_fake = torch.cat([x, G_output], dim=1)
    D_output_f = model_D(X_fake)
    D_fake_loss = BCELoss()(D_output_f, torch.zeros(
        D_output_f.size()).to(device))
    # 反向传播并优化
    D_loss = (D_real_loss + D_fake_loss) * 0.5
    D_loss.backward()
    optimizer_D.step()

    return D_loss.data.item()


def G_train(model_D, model_G, x, y, BCELoss, crit, optimizer_G, device, lamb=100):
    """
    训练生成器
    :param D: 判别器
    :param G: 生成器
    :param X: 未分隔的数据
    :param BCELoss: 二分交叉熵损失函数
    :param L1: L1正则化函数
    :param optimizer_G: 生成器优化器
    :param lamb: L1正则化的权重
    :return: 生成器的损失值
    """

    # 梯度初始化为0
    optimizer_G.zero_grad()
    # 在假数据上
    G_output = model_G(x)
    X_fake = torch.cat([x, G_output], dim=1)
    D_output_f = model_D(X_fake)
    G_BCE_loss = BCELoss()(D_output_f, torch.ones_like(D_output_f).to(device))
    mask = torch.zeros_like(y).to(device)
    mask[torch.where(y != 0)] = 1
    G_smoothL1_Loss = crit(G_output, y)+crit(G_output*mask, y)
    # 反向传播并优化
    G_loss = G_BCE_loss + lamb * G_smoothL1_Loss
    G_loss.backward()
    optimizer_G.step()

    return G_loss.data.item()


def simle_train_GD(logger, epoch, device, model_G, model_D, data_loader, args, optimizer_G, optimizer_D):
    """
    :param epoch: current epoch number
    :param model: CNN model
    :param data_loader: training data
    :param optimizer: training optimizer
    :param args: global arguments args
    :param save_dir: save directory
    :return: performance evaluation of the current epoch
    """

    model_G.train()
    model_D.train()
    lr = get_lr(epoch, args)
    # lr = 0.0002
    for param_group in optimizer_G.param_groups:
        param_group['lr'] = lr
    assert (lr is not None)
    optimizer_G.zero_grad()
    logger.add_scalar("LR", lr, global_step=epoch)
    crit = torch.nn.SmoothL1Loss()

    D_lossHist = []
    G_lossHist = []
    starttime = time.time()
    for i, (x, y) in enumerate(tqdm(data_loader)):
        # print(x.shape, y.shape)

        x, y = x.to(device), y.to(device)
        if epoch < 20:
            D_lossHist.append(D_train(model_D, model_G, x, y,
                                      BCELoss, optimizer_D, device))
            G_lossHist.append(G_train(model_D, model_G, x, y,
                                      BCELoss, crit, optimizer_G, device, lamb=100))
        if (epoch > 20) and (epoch % 20 < 10):
            D_lossHist.append(D_train(model_D, model_G, x, y,
                                      BCELoss, optimizer_D, device))
        if (epoch > 20) and epoch % 20 >= 10:
            G_lossHist.append(G_train(model_D, model_G, x, y,
                                      BCELoss, crit, optimizer_G, device, lamb=100))

    ##################################################################################

    endtime = time.time()
    D_lossHist = np.array(D_lossHist)  # 单次迭代所有batch下的loss列表
    mean_D_lossHist = np.mean(D_lossHist)  # 本次迭代的训练集平均损失
    logger.add_scalar("D_loss", mean_D_lossHist, global_step=epoch)
    G_lossHist = np.array(G_lossHist)  # 单次迭代所有batch下的loss列表
    mean_G_lossHist = np.mean(G_lossHist)  # 本次迭代的训练集平均损失
    logger.add_scalar("G_loss", mean_G_lossHist, global_step=epoch)
    # print('Train, epoch %d,  total loss %.4f,dice_focal_loss %.4f, deep_dense_loss %.4f,loss3 %.4f,accuracy %.4f, sensitivity %.4f, dice %.4f, dice % .4f,fpr%.4f,time %3.2f,lr %.5f '
    #       % (epoch, mean_loss, mean_loss1, mean_loss2, mean_loss3, mean_acc, mean_sensiti, mean_dice, mean_dice_hard, mean_fpr, endtime-starttime, lr))
    print('Train, epoch %d,  Discriminator loss %.4f,Generator loss %.4f,time %3.2f,lr %.5f '
          % (epoch, mean_D_lossHist, mean_G_lossHist, endtime-starttime, lr))
    # print('Train, epoch %d, dice loss %.4f, time %3.2f,lr %.5f '
    # 	  %(epoch, mean_loss,endtime-starttime,lr))
    # empty_cache()
    return mean_D_lossHist, mean_G_lossHist


def simle_train_pseudo(logger, epoch, device, model, data_loader, args, optimizer):
    """
    :param epoch: current epoch number
    :param model: CNN model
    :param data_loader: training data
    :param optimizer: training optimizer
    :param args: global arguments args
    :param save_dir: save directory
    :return: performance evaluation of the current epoch
    """
    net = model
    net.train()
    lr = get_lr(epoch, args)
    # lr = 0.0001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    assert (lr is not None)
    optimizer.zero_grad()
    logger.add_scalar("LR", lr, global_step=epoch)
    crit = torch.nn.SmoothL1Loss()

    lossHist = []
    starttime = time.time()
    for i, (insp_CT, exp_CT, pseudo_color) in enumerate(tqdm(data_loader)):
        # print(x.shape, y.shape)

        insp_CT, exp_CT, pseudo_color = insp_CT.to(device), exp_CT.to(
            device), pseudo_color.to(device)

        batchlen = insp_CT.size(0)
        input = torch.cat([insp_CT, exp_CT], dim=1)
        casePred1, casePred2 = model(input)
        # casePred1: threshold prediction
        # casePred2: pseudo_map prediction
       # casePred1, casePred2 = casePreds
        # print(f"before:{y.max()}")
        # mask = torch.zeros_like(y).to(device)
        # mask[torch.where(y != 0)] = 1
        # print(f"after:{y.max()}")
        # print(f"after_mask:{mask.max()}")
        # loss = crit(casePreds, y)+crit(casePreds*mask, y)
        # loss = crit(casePreds*mask, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # model3_e= time.time()
        # #print("model_load_time",model_end-model_start)
        # # for evaluation
        # stu_start = time.time()
        lossHist.append(loss.item())
        # # lossHist1.append(loss1.item())
        # # lossHist2.append(loss2.item())
        # # lossHist3.append(loss3.item())
        # # # segmentation calculating metrics#######################
        # outdata = casePreds.cpu().data.numpy()
        # segdata = y.cpu().data.numpy()

    ##################################################################################

    endtime = time.time()
    lossHist = np.array(lossHist)  # 单次迭代所有batch下的loss列表

    mean_loss = np.mean(lossHist)  # 本次迭代的训练集平均损失
    logger.add_scalar("loss", mean_loss, global_step=epoch)
    # print('Train, epoch %d,  total loss %.4f,dice_focal_loss %.4f, deep_dense_loss %.4f,loss3 %.4f,accuracy %.4f, sensitivity %.4f, dice %.4f, dice % .4f,fpr%.4f,time %3.2f,lr %.5f '
    #       % (epoch, mean_loss, mean_loss1, mean_loss2, mean_loss3, mean_acc, mean_sensiti, mean_dice, mean_dice_hard, mean_fpr, endtime-starttime, lr))
    print('Train, epoch %d,  total loss %.4f,time %3.2f,lr %.5f '
          % (epoch, mean_loss, endtime-starttime, lr))
    # print('Train, epoch %d, dice loss %.4f, time %3.2f,lr %.5f '
    # 	  %(epoch, mean_loss,endtime-starttime,lr))
    # empty_cache()
    return mean_loss


def pseudo_test_whole(device, model, data_loader, args, save_dir):
    """
    :param epoch: current epoch number
    :param model: CNN model
    :param data_loader: evaluation and testing data
    :param args: global arguments args
    :param save_dir: save directory
    :param test_flag: current mode of validation or testing
    :return: performance evaluation of the current epoch
    """
    model.eval()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():

        for i, (name, origin, spacing, insp_imgs, exp_imgs, pseudo_maps) in enumerate(tqdm(data_loader)):

            batch = insp_imgs.shape[0]
            insp_imgs = insp_imgs.to(device=device)
            exp_imgs = exp_imgs.to(device=device)
            # pseudo_maps = pseudo_maps.to(device)
            # -----------predict---------------
            x = torch.cat([insp_imgs, exp_imgs], dim=1)
            # thres, pseudo_preds = model(insp_imgs)
            _, pseudo_preds = model(x)
            # pseudo_preds = torch.nn.Sigmoid()(pseudo_preds)
            pseudo_preds1 = pseudo_preds.cpu().data.numpy()
            # thres = thres.cpu().data.numpy()
           # pseudo_maps = pseudo_maps.cpu().data.numpy()
            origin = origin.numpy()
            spacing = spacing.numpy()

            #######################################################################
            ################# REARRANGE THE DATA BY SPLIT ID########################
            for j in range(batch):
                casename = name[j]
                insp_img = insp_imgs[j, 0].cpu().data.numpy()
                exp_img = exp_imgs[j, 0].cpu().data.numpy()
                pseudo_pred1 = pseudo_preds1[j]
                pseudo_pred1 = np.argmax(pseudo_pred1, axis=0)
                # pseudo_map = pseudo_maps[j]
                # thre = thres[j]
                curorigin = origin[j]
                curspacing = spacing[j]
                # ----
                # pseudo_pred2 = np.zeros_like(insp_img)
                # # print(pseudo_pred2.shape)
                # pseudo_pred2[np.where((insp_img > thre[0]) &
                #                       (exp_img < thre[1]))] = 2  # fsad
                # pseudo_pred2[np.where((insp_img < thre[0]) & (
                #     exp_img < thre[1]))] = 1  # emphysema
                # pseudo_pred2[np.where((insp_img > thre[0]) & (
                #     exp_img > thre[1]))] = 3  # normal
                # pseudo_pred2[np.where(insp_img == 0)] = 0
                save3path = os.path.join(
                    save_dir, casename+"_pseudo_pred3.nii.gz")
                save_itk(pseudo_pred1.astype(dtype='uint8'),
                         curorigin, curspacing, save3path)
                save4path = os.path.join(
                    save_dir, casename+"_pseudo_pred4.nii.gz")
                p_combine_bw2 = pseudo_pred1.copy()
                p_combine_bw2[np.where(pseudo_pred1 == 2)] = 3
                p_combine_bw2[np.where(
                    (pseudo_pred1 == 1) & (insp_img > (74.0/1524)))] = 2
                save_itk(p_combine_bw2.astype(dtype='uint8'),
                         curorigin, curspacing, save4path)
                # savepath = os.path.join(
                #     save_dir, casename+"_pseudo_pred2.nii.gz")
                # save_itk(pseudo_pred2.astype(dtype='uint8'),
                #  curorigin, curspacing, savepath)

                # ---------evaluation_metrics------


def whole_deploy_external(device, model, data_loader, save_dir, diff=False, insp_dir=None):
    """
    :param epoch: current epoch number
    :param model: CNN model
    :param data_loader: evaluation and testing data
    :param args: global arguments args
    :param save_dir: save directory
    :param test_flag: current mode of validation or testing
    :return: performance evaluation of the current epoch
    """


    #----------------stage1:generate the registed exp_CT---------------
    model.eval()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # with torch.no_grad():

    #     for i, (name, origin, spacing, x,label) in enumerate(tqdm(data_loader)):
    #         # print(name)
    #         batch = x.shape[0]
    #         x = x.to(device=device)
    #         casePreds = model(x)
    #         outdata = casePreds.cpu().data.numpy()
           
    #         origin = origin.numpy()
    #         spacing = spacing.numpy()

    #         for j in range(batch):
    #             casename = name[j]

    #             pred = outdata[j, 0]
               
    #             if diff:
    #                 savepath = os.path.join(
    #                     save_dir, casename+"_expw_pred_diff.nii.gz")

    #                 insp_img, _, _, _ = load_itk_image(os.path.join(
    #                     insp_dir, casename+"_ins_re969.nii.gz"))
    #                 # if no normalize,do not execute the following code
    #                 pred = pred*2-1
    #                 save_itk(pred,[0,0,0], [1,1,1], savepath)
    #                 # ---------------------------
    #                 expw_pred = insp_img+pred
    #                 # print(np.max(expw_pred))
    #             else:
    #                 expw_pred = pred
    #             savepath = os.path.join(save_dir, casename+"_expw_pred.nii.gz")
    #             save_itk(expw_pred, [0,0,0], [1,1,1], savepath)

                # ---------evaluation------
                
    #----------------stage2:calulate and generate the pselabel according to the registed exp_CT/insp_CT---------------
    insp_thre = 94/1524 #原始的hu值为-950Hu，因为归一化区间为[-1024.0, 500.0]，所以-950Hu对应的归一化值为74/1524，
    #Quantification of Emphysema Progression at CT Using Simultaneous Volume, Noise, and Bias Lung Density Correction：
    #根据文献记载，肺气肿的阈值区间在-910Hu，-950Hu之间，所以归一化之后对应的五个间隔值为74/1524, 84/1524, 94/1524, 104/1524, 114/1524
    exp_thre = 168/1524  #更换为-900Hu对应的归一化值  124/1524
    # if "split_1_300" in save_dir:
    #     expw_pred_dir = "/data/mingyuezhao/PRM/test_results/Generation/Unet_relu_wsigmoid_255_mask_second_ds_threshold/insp_950/split_1_300"
    # elif"split_2_300" in save_dir:
    #     expw_pred_dir = "/data/mingyuezhao/PRM/test_results/Generation/Unet_relu_wsigmoid_255_mask_second_ds_threshold/insp_950/split_2_300"
    # elif "split_3_300" in save_dir:
    #     expw_pred_dir = "/data/mingyuezhao/PRM/test_results/Generation/Unet_relu_wsigmoid_255_mask_second_ds_threshold/insp_950/split_3_300"
    # elif "split_4_300" in save_dir:
    #     expw_pred_dir = "/data/mingyuezhao/PRM/test_results/Generation/Unet_relu_wsigmoid_255_mask_second_ds_threshold/insp_950/split_4_300"
    # elif "split_5_297" in save_dir:
    #     expw_pred_dir = "/data/mingyuezhao/PRM/test_results/Generation/Unet_relu_wsigmoid_255_mask_second_ds_threshold/insp_950/split_5_297"
    # else:
    #     assert False
    insp_dir = insp_dir
    expw_pred_dir = save_dir
    filelist = glob(os.path.join(expw_pred_dir,"*expw_pred.nii.gz"))
    for i in filelist:
        name = i.split("/")[-1].split("_expw")[0]
        insp_path = os.path.join(insp_dir,name+"_clean_hu1.nii.gz")
        insp_img,_,origin,spacing= load_itk_image(insp_path)
        insp_img = insp_img/255.0
        exp_img = load_itk_image(i)[0]
        # resized_exp_img = resize(exp_img, insp_img.shape, order=3, mode='reflect', anti_aliasing=False)
        pseudo_map = np.zeros_like(insp_img)
        # pseudo_map[np.where(insp_img>insp_thre)] =1
        pseudo_map[np.where((insp_img<insp_thre) & (exp_img<exp_thre))] = 1
        pseudo_map[np.where((insp_img>insp_thre) & (exp_img<exp_thre))] = 2
        pseudo_map[np.where((insp_img>insp_thre) & (exp_img>exp_thre))] = 3
        pseudo_map[np.where(insp_img==0)] = 0
        if not os.path.exists(save_dir+"/insp_930"):
            os.makedirs(save_dir+"/insp_930")
        savepath = os.path.join(save_dir+"/insp_930",name+"_pseudo_pred4_1.nii.gz")
        # savepath = i.replace("expw_pred","pseudo_pred4_1")
        save_itk(pseudo_map,origin,spacing,savepath)