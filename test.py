

from option import parser
import csv
from torch import optim
from torch.utils.data import DataLoader
from torch.backends import cudnn
import torch.nn as nn
from torch.nn import DataParallel
import torch

from utils import *
import time
import numpy as np
import data_loader as data
from tqdm import tqdm
from importlib import import_module
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from trainval_classifier import simple_test, pseudo_test_whole,whole_deploy_external
from sklearn import metrics 
import os
from sklearn.utils.multiclass import type_of_target


def test(CT_dir=None):
    """
    @description  :test to predict pseudo_map
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

    print('----------------------1.Load Models------------------------')

    model = import_module(args.model)  # 相对导入，py文件名是可选的，可选择导入哪种模型文件
    config, net = model.get_model()
    state_dict = torch.load(args.resume)['state_dict']
    net.load_state_dict(state_dict, strict=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = DataParallel(net)
    net.to(device)

    print('----------------------2.Load testset------------------------')

    # split_comber = SplitComb(args.stridet, args.cubesize)
    # dataset_test = data.PRMData(
    # 	config=config,
    # 	split_comber = split_comber,
    # 	phase='test')
    # test_loader = DataLoader(
    # 	dataset_test,
    # 	batch_size=args.batch_size,
    # 	shuffle=False,
    # 	num_workers=args.workers,
    # 	pin_memory=True)

    # dataset_test = data.PRM_preload(
    #     config=config,
    #     phase='test')
    # test_loader = DataLoader(
    #     dataset_test,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=args.workers,
    #     pin_memory=True)
    test_data = data.generator_dataset(config, phase='test')
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    print('----------------------3.Test------------------------')

    # test_casenet(device, net, test_loader, args, save_dir, CT_dir)
    pseudo_test_whole(device, net, test_loader, args, args.save_dir)
    # print(te_loss, mean_acc3, mean_sensiti3, mean_dice3, mean_fpr3)


def metric(save_dir, label_dir, name_list):
    dice_list = []
    fpr_list = []
    sensi_list = []
    acc_list = []
    total = []
    AUC_list = []
    for i in name_list:
        print(i)
        pred_path = os.path.join(save_dir, i+"_pseudo_pred4_1.nii.gz")
        label_path = os.path.join(label_dir, i+"_prm_label_c.nii.gz")
        pred, _, _, _ = load_itk_image(pred_path)
        label, _, _, _ = load_itk_image(label_path)

        # y_pred = F.one_hot(torch.from_numpy(pred).long(), num_classes=4)
        # pred = y_pred.numpy().transpose(3, 0, 1, 2)
        # y_true = F.one_hot(torch.from_numpy(label).long(), num_classes=4)
        # label = y_true.numpy().transpose(3, 0, 1, 2)

        curdice = dice_coef_np(pred, label.astype(np.int64), hard=True)
        print(f"dice:{curdice}")
        dice_list.append(curdice)
        curfpr = FalsePositiveRate(pred, label.astype(np.int64), hard=True)
        print(f"fpr:{curfpr}")
        fpr_list.append(curfpr)
        cursensi = sensitivity_np(pred, label.astype(np.int64), hard=True)
        print(f"sensi:{cursensi}")
        sensi_list.append(cursensi)
        curacc = acc_np(pred, label)
        print(f"acc:{curacc}")
        acc_list.append(curacc)
        curAUC = []
        # print(np.sum((label==1).astype(np.int64)))
        # print(np.sum((pred==1).astype(np.int64)))
        # assert label.any()==1
        # print(pred)
        # label = np.array(label).astype(np.int64)
        # print(type_of_target(label))
      
        # # for i in range(pred.shape[0]):
        # #     for j in range(pred.shape[1]):
        # #         for k in range(pred.shape[2]):
        # #             if label[i, j, k].dtype!=np.int64:
        # #                 print(label[i][j][k].dtype)
        fpr, tpr, thresholds = metrics.roc_curve(label.flatten().astype(np.int64), pred.flatten().astype(np.int64), pos_label=1)
        curAUC.append(metrics.auc(fpr, tpr))
        fpr, tpr, thresholds = metrics.roc_curve(label.flatten().astype(np.int64), pred.flatten().astype(np.int64), pos_label=2)
        curAUC.append(metrics.auc(fpr, tpr))
        fpr, tpr, thresholds = metrics.roc_curve(label.flatten().astype(np.int64), pred.flatten().astype(np.int64), pos_label=3)
        curAUC.append(metrics.auc(fpr, tpr))
        print(curAUC)
        AUC_list.append(curAUC)
        total.append([i, float(round(curacc, 4)), 
            float(round(cursensi[0], 4)), float(round(cursensi[1], 4)), float(round(cursensi[2], 4)),
            float(round(curdice[0], 4)), float(round(curdice[1], 4)), float(round(curdice[2], 4)),
            float(round(curfpr[0], 4)), float(round(curfpr[1], 4)), float(round(curfpr[2], 4)),
            float(round(curAUC[0], 4)), float(round(curAUC[1], 4)), float(round(curAUC[2], 4))
            ])
        # with open(os.path.join(save_dir, 'test_results.csv'), 'w') as csvout:
        #     writer = csv.writer(csvout)

    mean_dice = np.mean(np.array(dice_list), axis=0)
    mean_auc = np.mean(np.array(AUC_list), axis=0)
    mean_fpr = np.mean(np.array(fpr_list), axis=0)
    mean_sensi = np.mean(np.array(sensi_list), axis=0)
    mean_acc = np.mean(np.array(acc_list))
    std_dice = np.std(np.array(dice_list), axis=0)
    std_fpr = np.std(np.array(fpr_list), axis=0)
    std_auc = np.std(np.array(AUC_list), axis=0)
    std_sensi = np.std(np.array(sensi_list), axis=0)
    std_acc = np.std(np.array(acc_list), axis=0)
    print(mean_dice)
    print(mean_fpr)
    print(mean_sensi)
    print(mean_acc)
    # print(mean_auc)
    with open(os.path.join(save_dir, 'PRM_reresized_results_2.csv'), 'w') as csvout:
        writer = csv.writer(csvout)
        row = ['name', 'test acc', 'test_sensi_em', 'test_sensi_fsad', 'test_sensi_nor',
               'test_dice_em', 'test_dice_fsad', 'test_dice_nor',
               'test_fpr_em', 'test_fpr_fsad', 'test_fpr_nor','test_auc_em', 'test_auc_fsad', 'test_auc_nor']
        writer.writerow(row)
        for j in range(len(total)):
            writer.writerow(total[j])

        all_mean = ['all mean', mean_acc, mean_sensi[0], mean_sensi[1], mean_sensi[2],
                    mean_dice[0], mean_dice[1], mean_dice[2], mean_fpr[0], mean_fpr[1], mean_fpr[2],mean_auc[0], mean_auc[1], mean_auc[2]]
        all_std = ['all std', std_acc, std_sensi[0], std_sensi[1], std_sensi[2],
                   std_dice[0], std_dice[1], std_dice[2], std_fpr[0], std_fpr[1], std_fpr[2],std_auc[0], std_auc[1], std_auc[2]]

        writer.writerow(all_mean)
        writer.writerow(all_std)
        csvout.close()


# ----------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------self-registration + dual threshold = test result-----------------------------------\


def Generator_metrics(save_dir, label_dir, name_list):
    """
    @description  :Calculate the evaluation metrics of Generation task
    ---------
    @param  :
    -------
    @Returns  :psnr->up is better,
    -------
    """
    psnr_list = []
    ssim_list = []
    mae_list = []
    total = []
    for i in name_list:

        print(i)
        pred_path = os.path.join(save_dir, i+"_expw_pred.nii.gz")
        # label_path = os.path.join(
        #     label_dir, i+"_exp_warp_norm_re969.nii.gz")
        label_path = os.path.join(
            label_dir, i+"_exp_warp_norm_re969.nii.gz")
        expw_pred, _, _, _ = load_itk_image(pred_path)

        # print(np.max(expw_pred))
        target, _, _, _ = load_itk_image(label_path)
        print(np.max(expw_pred), np.max(target))
        expw_pred[np.where(target == 0)] = 0
        valid = len(np.argwhere(target == 0))
        mae = np.sum(np.abs(expw_pred-target))/valid
        # print(mae)
        mae_list.append(mae)
        psnr = peak_signal_noise_ratio(expw_pred, target)
        psnr_list.append(psnr)
        ssim = structural_similarity(expw_pred, target)
        ssim_list.append(ssim)

        # total.append([i, float(round(mae, 4)), float(round(cursensi[0], 4)), float(round(cursensi[1], 4)), float(round(cursensi[2], 4)),
        # float(round(curdice[0], 4)), float(
        #     round(curdice[1], 4)), float(round(curdice[2], 4)),
        # float(round(cursensi[0], 4)), float(round(cursensi[1], 4)), float(round(cursensi[2], 4))])
        total.append([i, mae, psnr, ssim])
        # with open(os.path.join(save_dir, 'test_results.csv'), 'w') as csvout:
        #     writer = csv.writer(csvout)

    mean_mae = np.mean(np.array(mae_list), axis=0)
    mean_psnr = np.mean(np.array(psnr_list), axis=0)
    mean_ssim = np.mean(np.array(ssim_list), axis=0)
    std_mae = np.std(np.array(mae_list), axis=0)
    std_psnr = np.std(np.array(psnr_list), axis=0)
    std_ssim = np.std(np.array(ssim_list), axis=0)

    with open(os.path.join(save_dir, 'test_results_nobg.csv'), 'w') as csvout:
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


def test_generator(insp_dir=None, pseudo_colormap=False):
    """
    @description  :test for generation task-from insp_CT to exp_warp_CT
    ---------
    @param  : save_dir->the savepath for generation results,
            GT_dir->GroudTruth path for real registration exp_warp_CT
    -------
    @Returns  :
    -------
    """
    global args
    args = parser.parse_args()
    torch.manual_seed(0)
    # torch.cuda.set_device(0)

    print('----------------------1.Load Models------------------------')

    model = import_module(args.model)  # 相对导入，py文件名是可选的，可选择导入哪种模型文件
    config, G_net = model.get_model()
    state_dict = torch.load(args.checkpointpath)['state_dict']
    G_net.load_state_dict(state_dict, strict=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    G_net = DataParallel(G_net)
    G_net.to(device)

    print('----------------------2.Load Datasets------------------------')
    test_data = data.generator_dataset(config, phase='test')
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)
    print('----------------------3.Test------------------------')

    simple_test(device, G_net, test_loader,
                args.outputpath, diff=False, insp_dir=insp_dir)

def deploy_test(insp_dir = None,save_dir = None):
    """
    @param  : save_dir->the savepath for generation results,
            GT_dir->GroudTruth path for real registration exp_warp_CT
    -------
    @Returns  :
    -------
    """
    global args
    args = parser.parse_args()
    torch.manual_seed(0)
    # torch.cuda.set_device(0)

    print('----------------------1.Load Models------------------------')
    model = import_module(args.model)  # 相对导入，py文件名是可选的，可选择导入哪种模型文件
    config, G_net = model.get_model()
    # if save_dir is None:
    #     save_dir = args.outputpath
    
    # state_dict = torch.load(args.checkpointpath)['state_dict']
    # G_net.load_state_dict(state_dict, strict=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # G_net = DataParallel(G_net)
    # G_net.to(device)

    print('----------------------2.Load Datasets------------------------')
    deploy_data = data.generator_dataset(config, phase='deploy')
    deploy_loader = DataLoader(
        deploy_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)
    print('----------------------3.Test------------------------')

    whole_deploy_external(device, G_net, deploy_loader, save_dir = save_dir, diff=False, insp_dir=insp_dir)

if __name__ == "__main__":
   # test()
    # name_list =[i for i in os.listdir("/home1/mingyuezhao/PRM/debug/test001") if "org" in i]
    # x = metric(name_list)
    # print(x)
    # print(np.array(x).mean())

    # 1.-----------------------inference-------------
    # save_dir = "/data/mingyuezhao/PRM/test_results/Pseudo_prediction/"
   # CT_dir = "/home1/mingyuezhao/PRM/data/processed2_data_fine"

    # test()
    
    # path = ["test_results/Generation/Unet_relu_wsigmoid_255_mask_second_ds/split_1_300","test_results/Generation/Unet_relu_wsigmoid_255_mask_second_ds/split_2_300","test_results/Generation/Unet_relu_wsigmoid_255_mask_second_ds/split_3_300","test_results/Generation/Unet_relu_wsigmoid_255_mask_second_ds/split_4_300","test_results/Generation/Unet_relu_wsigmoid_255_mask_second_ds/split_5_297"]
    # path = ["test_results/Generation/Unet_relu_wsigmoid_255_mask_second_ds/split_1_300"]
    # for i in path:
    #     save_dir = i
    #     label_dir = "/data/mingyuezhao/PRM/data/first/colormap_PRM"
    #     name_list = []
    #     for i in os.listdir(save_dir):
    #         if "_pseudo" in i and i.split("_pseudo")[0] not in name_list:
    #             if i.startswith("test"):
    #                 pass
    #             else:
    #                 name_list.append(i.split("_pseudo")[0])
    #     metric(save_dir, label_dir, name_list)
    # # 2.-----------------------infernece-------------
    # label_dir = "/home1/mingyuezhao/PRM/data/processed2_data_fine"
    # name_list = []
    # for i in os.listdir("/home1/mingyuezhao/PRM/test_results/AttentionUnet_in_2_fine"):
    #     if i.split("-")[0] not in name_list:
    #         name_list.append(i.split("-")[0])
    # metric(save_dir, label_dir, name_list)

    # test_generator(insp_dir="/data/mingyuezhao/PRM/data/first/resize_ins_2")

    # x_data,_,_,_ = load_itk_image("./data/processed2_data_fine/CZSC000184_304.nii.gz")
    # print(x_data.shape)
    # x_data,_,_,_ = load_itk_image("./data/processed2_data_fine/CZSC000184_in.nii.gz")
    # #x_data = np.load("./data/processed2_data_fine/CZSC0001_label.nii.gz")
    # print(x_data.shape)
    # path = "./data/preload_data"
    # for i in os.listdir(path):
    #     if "_in.nii.gz" in i:
    #         print(i,end ="\r")
    #         x_data,_,_,_ = load_itk_image(os.path.join(path,i))
    #         if x_data.shape== (96,96,96):
    #             pass
    #         else:
    #             print(x_data.shape)

    # ------------------------------------

    # inp_CT_list = [os.path.join("./data/processed1_data")]
    # test_on_self_registration(
    #     inp_CT_list, exp_CT_dir, save_dir, label_gene=False, visul=True, metric_cal=False)
    # save_dir = "/data/mingyuezhao/PRM/test_results/Generation/Unet_relu_wsigmoid_255_mask_second_ds/split_1_198"
    # label_dir = "/data/mingyuezhao/PRM/data/first/resize_warp_2"
    # name_list = []
    # for i in os.listdir(save_dir):
    #     if "CZSC" in i:
    #         if i.split("_")[0] not in name_list:
    #             name_list.append(i.split("_")[0])
    # Generator_metrics(save_dir, label_dir, name_list)



    

    #--------------Internl_test----------------------------
    # global args
    # args = parser.parse_args()
    # insp_dir = "/data/mingyuezhao/PRM/data/first/resize_ins_2"
    # label_dir = "/data/mingyuezhao/PRM/data/first/pseudo_map_2/"
    
    # deploy_test(insp_dir = insp_dir,save_dir = None)
    # dataset_split = load_pickle(args.dataset_split)
    
    # name_list = []
    # for i in dataset_split["test"]:
    #     # if i.endswith("color_re969.nii.gz") and i.split("_color")[0] not in name_list:
    #     name_list.append(i.split("_ins_re969")[0])
    # print(len(name_list))
    # metric(save_dir = args.outputpath, label_dir = label_dir, name_list = name_list)

    #-------------deploy:external_validation and metrics evaluation-----

    # insp_dir = "/data/mingyuezhao/PRM/data/External_verification_25_1_7/Old_manCOPD_PRISm_processed"
    # insp_thre = 74/1524
    # exp_thre = 168/1524
    # # test_expw_pred_dir = "/data/mingyuezhao/PRM/test_results/Generation/Unet_expwCT"
    # insp_dir = insp_dir
    # expw_pred_dir = "/data/mingyuezhao/PRM/test_results/external_val_cz_194/split_1_model_reresized"
    # filelist = glob(os.path.join(expw_pred_dir,"*expw_pred.nii.gz"))
    # for i in filelist:
    #     print(i)
    #     name = i.split("/")[-1].split("_")[0]
    #     insp_path = os.path.join(insp_dir,name+"_clean_hu.nii.gz")
    #     insp_img,_,origin,spacing= load_itk_image(insp_path)
    #     exp_img = load_itk_image(i)[0]
    #     pseudo_map = np.zeros_like(insp_img)
    #     # pseudo_map[np.where(insp_img>insp_thre)] =1
    #     pseudo_map[np.where((insp_img<insp_thre) & (exp_img<exp_thre))] = 1
    #     pseudo_map[np.where((insp_img>insp_thre) & (exp_img<exp_thre))] = 2
    #     pseudo_map[np.where((insp_img>insp_thre) & (exp_img>exp_thre))] = 3
    #     pseudo_map[np.where(insp_img==0)] = 0
    #     savepath = i.replace("expw_pred","pseudo_pred4_1")
    #     save_itk(pseudo_map,origin,spacing,savepath)

    #-------------external_validation----------------------------


    #------------------------------------metrics_cal
    ss = "insp_910_exp_900"
    insp_dir = "/data/mingyuezhao/PRM/data/External_verification_CZ/img"
    
    expw_pred_dir = "/data/mingyuezhao/PRM/test_results/external_val_cz_194/split_1_model_reresized/insp_950"
    save_dir = expw_pred_dir.replace("insp_950",str(ss))
    if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    insp_thre = 114/1524 #原始的hu值为-950Hu，因为归一化区间为[-1024.0, 500.0]，所以-950Hu对应的归一化值为74/1524，
    #Quantification of Emphysema Progression at CT Using Simultaneous Volume, Noise, and Bias Lung Density Correction：
    #根据文献记载，肺气肿的阈值区间在-910Hu，-950Hu之间，所以归一化之后对应的五个间隔值为74/1524, 84/1524, 94/1524, 104/1524, 114/1524
    exp_thre = 124/1524  #更换为-900Hu对应的归一化值  124/1524
    insp_dir = insp_dir

    filelist = glob(os.path.join(expw_pred_dir,"*expw_pred.nii.gz"))
    for i in tqdm(filelist):
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
        
        savepath = os.path.join(save_dir,name+"_pseudo_pred4_1.nii.gz")
        # savepath = i.replace("expw_pred","pseudo_pred4_1")
        save_itk(pseudo_map,origin,spacing,savepath)
    label_dir = "/data/mingyuezhao/PRM/data/External_verification_CZ/color"  #图像路径同步修改至baseline文件下的config
    name_list = []
    for i in os.listdir(label_dir):
        if i.endswith("prm_label_c.nii.gz") and i.split("_prm")[0] not in name_list:
            name_list.append(i.split("_prm")[0])
    # print(len(name_list))
    metric(save_dir, label_dir, name_list)
    