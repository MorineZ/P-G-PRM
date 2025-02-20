import numpy as np
import torch
from torch.utils.data import Dataset
import os
import time
import random
import SimpleITK as sitk
from glob import glob
from scipy.ndimage.filters import gaussian_filter
from utils import save_itk, load_itk_image, lumTrans, load_pickle
from importlib import import_module
import pickle
from einops import rearrange
import torch.nn.functional as F

from split_combine_mj import SplitComb
# from skimage.morphology import skeletonize_3d
from PIL import Image
from monai.transforms import (

    EnsureChannelFirstD,
    Compose,
    Resized,
    RandAdjustContrastd,
    RandGaussianSmoothd,
    RandAxisFlipd,
    RandAffined,
    RandZoomd,
    EnsureTyped,
)
train_transforms = Compose(
    [

        EnsureChannelFirstD(keys=["image", "image2"]),
        # RandAdjustContrastd(keys=["image", "image2", "label"], prob=0.1, gamma=(
        #     0.5, 4.5), allow_missing_keys=False),
        # RandGaussianSmoothd(keys=["image", "image2", "label"], sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5),
        #                     sigma_z=(0.25, 1.5), approx='erf', prob=0.1, allow_missing_keys=False),
        RandAxisFlipd(keys=["image", "image2"], prob=0.5,
                      allow_missing_keys=False),
        RandAffined(keys=["image", "image2"],  prob=0.5,
                    rotate_range=0.05, padding_mode="zeros"),
        RandZoomd(keys=["image", "image2"], prob=0.5, min_zoom=0.9,
                  max_zoom=1.1, mode=['area', 'area']),
        EnsureTyped(keys=["image", "image2"]),
    ]
)
val_transforms = Compose(
    [
        EnsureChannelFirstD(keys=["image", "image2"]),
        EnsureTyped(keys=["image", "image2"]),
    ]
)


class generator_dataset(Dataset):
    """
    Generate dataloader
    """

    def __init__(self, config, phase='train'):
        """
        :param config: configuration from model
        :param phase: training or validation or testing
        :param random_select: use partly, randomly chosen data for training
        """
        assert (phase == 'train' or phase ==
                'val' or phase == 'test' or phase == 'deploy')
        # when phase = 'test':return test_image + test_label ; when phase = 'deploy':return test_image (without label)
        self.phase = phase
        self.patch_per_case = 5  # patches used per case if random training
        """
		specify the path and data split
		"""
        if self.phase=="deploy":
            self.datapath = config['deploy_path']
            self.dataset = [os.path.join(self.datapath,i) for i in os.listdir(self.datapath) if i.endswith("insp_norm_rs.nii.gz")]
        else:
            self.datapath = config['dataset_path']
            self.dataset = load_pickle(config['dataset_split'])

        print(
            "-------------------------Load all data into memory---------------------------")
        """
		count the number of cases
		"""

        namelist = []
        # labellist = []
        all_image_memory = {}
        all_label_memory = {}
        # all_color_memory = {}
        # all_centerline_memory = {}
        # all_IDTM_memory = {}
        self.caseNumber = 0

        if self.phase == 'train':
            # train_set = self.dataset["train"]["lidc"] + \
            #     self.dataset["train"]["exact09"]
            # train_set = self.dataset["train"]["exact09"]
            train_set = self.dataset["train"][:-30]
            
            print(f"train_set_number:{len(train_set)}")
            # assert len(train_set) == 40
            file_num = len(train_set)
            self.caseNumber += file_num
            for raw_path in train_set:
                # print(raw_path)
                # raw_path = "/data/mingyuezhao/Airway_Seg/Dataset/LIDC_EXACT09/" + \
                #     raw_path.split("/")[-1]
                # label_name = raw_path.replace(
                #     "ins_re969", "diff_norm")
                label_name = raw_path.replace(
                    "ins_re969", "exp_warp_norm_re969")
                pseudo_name = raw_path.replace(
                    "ins_re969", "label3")
                raw_path = "/data/mingyuezhao/PRM/data/first/resize_ins_2/" + \
                    raw_path
                assert (os.path.exists(raw_path) is True)
                name = raw_path.split("/")[-1].split("_ins_re969")[0]
                label_path = "/data/mingyuezhao/PRM/data/first/resize_warp_2/" + label_name
                # pseudo_path = "/data/mingyuezhao/PRM/data/first/colormap_PRM/" + pseudo_name
                imgs, _, origin, spacing = load_itk_image(raw_path)
                labels, _, _, _ = load_itk_image(label_path)
                # colors, _, _, _ = load_itk_image(pseudo_path)
                all_image_memory[name] = [imgs, origin, spacing]
                all_label_memory[name] = labels
                # all_color_memory[name] = colors
                namelist.append(name)

        elif self.phase == 'val':
            val_set = self.dataset["train"][-30:]

            file_num = len(val_set)
            self.caseNumber += file_num
            for raw_path in val_set:
                # add by zhao
                # print("train",raw_path)
                # label_name = raw_path.replace(
                #     "ins_re969", "diff_norm")
                label_name = raw_path.replace(
                    "ins_re969", "exp_warp_norm_re969")
                pseudo_name = raw_path.replace(
                    "ins_re969", "label3")
                raw_path = "/data/mingyuezhao/PRM/data/first/resize_ins_2/" + \
                    raw_path
                assert (os.path.exists(raw_path) is True)
                name = raw_path.split("/")[-1].split("_ins_re969")[0]
                label_path = "/data/mingyuezhao/PRM/data/first/resize_warp_2/" + label_name
                pseudo_path = "/data/mingyuezhao/PRM/data/first/colormap_PRM/" + pseudo_name
                imgs, _, origin, spacing = load_itk_image(raw_path)
                labels, _, _, _ = load_itk_image(label_path)

                all_image_memory[name] = [imgs, origin, spacing]
                all_label_memory[name] = labels
                # all_color_memory[name] = colors
                namelist.append(name)
        elif self.phase == 'test':
            # test_set = self.dataset["test"]["lidc"] + \
            #     self.dataset["test"]["exact09"]
            # self.dataset["test"]
            test_set = self.dataset["test"]
            # assert len(test_set) == 90
            file_num = len(test_set)
            self.caseNumber += file_num
            for raw_path in test_set:
                # add by zhao
                print("test", raw_path)
                label_name = raw_path.replace(
                    "ins_re969", "exp_warp_norm_re969")
                pseudo_name = raw_path.replace(
                    "ins_re969", "label3")
                raw_path = "/data/mingyuezhao/PRM/data/first/resize_ins_2/" + \
                    raw_path
                assert (os.path.exists(raw_path) is True)
                name = raw_path.split("/")[-1].split("_ins_re969")[0]
                label_path = "/data/mingyuezhao/PRM/data/first/resize_warp_2/" + label_name
                # pseudo_path = "/data/mingyuezhao/PRM/data/first/colormap_PRM/" + pseudo_name
                imgs, _, origin, spacing = load_itk_image(raw_path)
                labels, _, _, _ = load_itk_image(label_path)
                # colors, _, _, _ = load_itk_image(pseudo_path)
                all_image_memory[name] = [imgs, origin, spacing]
                all_label_memory[name] = labels
                # all_color_memory[name] = colors
                namelist.append(name)

        else:  # phase == 'deploy'
            deploy_set = self.dataset

            file_num = len(deploy_set)
            self.caseNumber += file_num
            for raw_path in deploy_set:
                # add by zhao
                print("deploy", raw_path)
                # raw_path = os.path.join(self.datapath, raw_path.split('/')[-1] + '.gz')
                assert (os.path.exists(raw_path) is True)
                name = raw_path.split("/")[-1].split("_insp")[0]
                label_path = raw_path.replace("insp_rs", "pse")
                
                imgs, _, origin, spacing = load_itk_image(raw_path)
                if os.path.exists(label_path):
                    labels, _, _, _ = load_itk_image(label_path)
                else:
                    labels = np.zeros(imgs.shape)
                
                all_image_memory[name] = [imgs, origin, spacing]
                all_label_memory[name] = labels
                namelist.append(name)
        self.allimgdata_memory = all_image_memory
        self.alllabeldata_memory = all_label_memory
        # self.all_color_memory = all_color_memory
        random.shuffle(namelist)
        self.namelist = namelist

        print('---------------------Initialization Done---------------------')
        print('Phase: %s total cubelist number: %d' %
              (self.phase, len(self.namelist)))
        print()

    def __len__(self):
        """
        :return: length of the dataset
        """

        return len(self.namelist)

    def __getitem__(self, idx):
        """
        :param idx: index of the batch
        :return: wrapped data tensor and name, shape, origin, etc.
        """
        t = time.time()
        np.random.seed(int(str(t % 1)[2:7]))  # seed according to time

        name = self.namelist[idx]

        # train: [data_name, cursplit, j, nzhw, orgshape, 'Y']
        # val/test: [data_name, cursplit, j, nzhw, orgshape, 'N']

        if self.phase == 'deploy':
            imginfo = self.allimgdata_memory[name]
            imgs, origin, spacing = imginfo[0], imginfo[1], imginfo[2]

            curcube = (imgs.astype(np.float32))/255.0
            label = self.alllabeldata_memory[name]
            label = label.astype('float')
            curcube = curcube[np.newaxis, ...]
            label = label[np.newaxis, ...]
            return name, origin, spacing, torch.from_numpy(curcube).float(), torch.from_numpy(label).float()
        else:
            # if self.phase == 'train' and curtransFlag == 'Y' and self.augtype['split_jitter'] is True:
            #     # random jittering during the training
            #     cursplit = augment_split_jittering(cursplit, curShapeOrg)

            ####################################################################
            imginfo = self.allimgdata_memory[name]
            imgs, origin, spacing = imginfo[0], imginfo[1], imginfo[2]
            # imgs = imgs.astype(np.float32)
            imgs = (imgs.astype(np.float32))
            # print(np.max(imgs), np.min(imgs))
            ####################################################################

            label = self.alllabeldata_memory[name]
            label = label.astype('float')

            # color = self.all_color_memory[name]
            # x = torch.from_numpy(color).long()
            # 如果不加类别数，会默认使用 输入数据中最大值，作为列别数。一般还是会加的
            # color = F.one_hot(x, num_classes=3).numpy().transpose(3, 0, 1, 2)
            # print(color.shape)
            if self.phase == "train":
                train_data = {"image": imgs, "image2": label, }
                # print(imgs.shape, label.shape)
                train_data = train_transforms(train_data)
                curcube = train_data["image"]
                label = train_data["image2"]
                # color = train_data["label"]
                # color = color.long()
                # color = F.one_hot(color, num_classes=4)
                # print(color.shape)
                # return curcube.float(), label.float(), color.float()
                return curcube.float(), label.float()

            elif self.phase == "val":
                val_data = {"image": imgs, "image2": label}
                # print(imgs.shape, label.shape)
                val_data = val_transforms(val_data)
                curcube = val_data["image"]
                label = val_data["image2"]
                # color = val_data["label"]

                # return curcube.float(), label.float(), color.float()
                return curcube.float(), label.float()
                # centerline = centerline[np.newaxis,...]
                # if self.phase == 'train':
                # weight = weight[np.newaxis,...]
                # return torch.from_numpy(curcube).float(),torch.from_numpy(label).float(),\
                # 	torch.from_numpy(weight).float(),torch.from_numpy(coord).float(),torch.from_numpy(origin),\
                # 	torch.from_numpy(spacing), curNameID, curSplitID,\
                # 	torch.from_numpy(curnzhw),torch.from_numpy(curShapeOrg)
                # else:
                # torch.from_numpy(centerline).float(),
            else:
                curcube = imgs[np.newaxis, ...]
                # 如果不加类别数，会默认使用 输入数据中最大值，作为列别数。一般还是会加的
                label = label[np.newaxis, ...]

                return name, origin, spacing, torch.from_numpy(curcube).float(), torch.from_numpy(label).float()

# data_preloading


def data_preloading(dataset_split_file, split_comber=None, save_dir=None):
    dataset = load_pickle(dataset_split_file)
    # split_info = {"train":[],"val":[],"test":[]}
    split_info_train = []
    # 1.train_set
    # train_set =dataset["train"]
    # assert len(train_set)==200
    # train_cube_list = []

    # for raw_path in train_set:
    # 	#add by zhao
    # 	print("train",raw_path)
    # 	#raw_path = os.path.join(self.datapath, raw_path.split('/')[-1] + '.gz')

    # 	assert(os.path.exists(raw_path) is True)
    # 	name =  raw_path.split("/")[-1].split("_in")[0]
    # 	label_path = raw_path.replace('_in.nii.gz', '_label.nii.gz')
    # 	assert (os.path.exists(label_path) is True)
    # 	imgs, origin, spacing = load_itk_image(raw_path)
    # 	splits, nzhw, orgshape = split_comber.split_id(imgs)
    # 	print("Name: %s, # of splits: %d"%(name, len(splits)))
    # 	labels,_,_ = load_itk_image(label_path)

    # 	cube_train = []

    # 	for j in range(len(splits)):
    # 		"""
    # 		check if this sub-volume cube is suitable
    # 		"""
    # 		cursplit = splits[j]
    # 		curcube = imgs[cursplit[0][0]:cursplit[0][1], cursplit[1][0]:cursplit[1][1], cursplit[2][0]:cursplit[2][1]]

    # 		labelcube = labels[cursplit[0][0]:cursplit[0][1], cursplit[1][0]:cursplit[1][1], cursplit[2][0]:cursplit[2][1]]
    # 		curnumlabel = np.sum(labelcube)
    # 		if curnumlabel > 0:  # filter out those zero-0 labels
    # 			curlist = [name, cursplit, j, nzhw, orgshape, 'Y']
    # 			cube_train.append(curlist)

    # 	random.shuffle(cube_train)
    # 	train_cube_list += cube_train
    # split_info_train = train_cube_list

    # 2.val_set
    # val_set =dataset["val"]
    # assert len(val_set)==28
    # val_cube_list = []
    # for raw_path in val_set:
    # 	#add by zhao
    # 	print("val",raw_path)
    # 	#raw_path = os.path.join(self.datapath, raw_path.split('/')[-1] + '.gz')

    # 	assert(os.path.exists(raw_path) is True)
    # 	name =  raw_path.split("/")[-1].split("_in")[0]
    # 	label_path = raw_path.replace('_in.nii.gz', '_label.nii.gz')
    # 	assert (os.path.exists(label_path) is True)
    # 	imgs, origin, spacing = load_itk_image(raw_path)
    # 	splits, nzhw, orgshape = split_comber.split_id(imgs)
    # 	print("Name: %s, # of splits: %d"%(name, len(splits)))
    # 	labels,_,_ = load_itk_image(label_path)
    # 	cube_val = []
    # 	for j in range(len(splits)):
    # 		"""
    # 		check if this sub-volume cube is suitable
    # 		"""
    # 		cursplit = splits[j]
    # 		labelcube = labels[cursplit[0][0]:cursplit[0][1], cursplit[1][0]:cursplit[1][1], cursplit[2][0]:cursplit[2][1]]
    # 		curnumlabel = np.sum(labelcube)
    # 		if curnumlabel > 0:  # filter out those zero-0 labels
    # 			curlist = [name, cursplit, j, nzhw, orgshape, 'Y']
    # 			cube_val.append(curlist)

    # 	random.shuffle(cube_val)
    # 	val_cube_list += cube_val
    # split_info_train = val_cube_list

    # 3.test_set
    test_set = dataset["test"]
    assert len(test_set) == 56
    test_cube_list = []
    for raw_path in test_set:
        # add by zhao
        print("test", raw_path)
        # raw_path = os.path.join(self.datapath, raw_path.split('/')[-1] + '.gz')

        assert (os.path.exists(raw_path) is True)
        name = raw_path.split("/")[-1].split("_in")[0]
        label_path = raw_path.replace('_in.nii.gz', '_label.nii.gz')
        assert (os.path.exists(label_path) is True)
        imgs, origin, spacing = load_itk_image(raw_path)
        splits, nzhw, orgshape = split_comber.split_id(imgs)
        print("Name: %s, # of splits: %d" % (name, len(splits)))
        labels, _, _ = load_itk_image(label_path)
        cube_test = []
        for j in range(len(splits)):
            """
            check if this sub-volume cube is suitable
            """
            cursplit = splits[j]
            labelcube = labels[cursplit[0][0]:cursplit[0][1], cursplit[1]
                               [0]:cursplit[1][1], cursplit[2][0]:cursplit[2][1]]
            curnumlabel = np.sum(labelcube)
            if curnumlabel > 0:  # filter out those zero-0 labels
                curlist = [name, cursplit, j, nzhw, orgshape, 'Y']
                cube_test.append(curlist)

        random.shuffle(cube_test)
        test_cube_list += cube_test
    split_info_train = test_cube_list
    with open('split_comber_test.pickle', 'wb') as handle:

        pickle.dump(split_info_train, handle)


class data_2D_preloading():
    def __init__(self, dataset_split_file="./split_dataset_in_2.pickle"):
        self.train_val_test = dataset_split_file

    def run(self, save_dir=None, phase='train'):

        dataset = load_pickle(self.train_val_test)
        split_info = []
        if phase == "train":
            # 1.train_set
            data_set = dataset["train"]
            assert len(data_set) == 200
        elif phase == "val":
            # 1.train_set
            data_set = dataset["val"]
            assert len(data_set) == 28
        else:
            data_set = dataset["test"]
            assert len(data_set) == 57
        cube_list = []

        for raw_path in data_set:
            print(phase, raw_path)
            # raw_path = os.path.join(self.datapath, raw_path.split('/')[-1] + '.gz')

            assert (os.path.exists(raw_path) is True)
            name = raw_path.split("/")[-1].split("_in.nii")[0]
            label_path = raw_path.replace('_in', '_label')
            assert (os.path.exists(label_path) is True)
            imgs, _, origin, spacing = load_itk_image(raw_path)
            print(imgs.shape)

            labels, _, _, _ = load_itk_image(label_path)

            cube_train = []

            for j in range(imgs.shape[1]):
                image_slice = imgs[:, j, :]
                label_slice = labels[:, j, :]
                orgshape = [imgs.shape[0], imgs.shape[2]]
                if phase == "train":
                    if label_slice.sum() > 0:
                        im = Image.fromarray(image_slice)
                        im.save(os.path.join(
                            save_dir, name + "_" + str(j) + ".png"))
                        label = Image.fromarray(label_slice)
                        label = label.convert("RGB")
                        label.save(os.path.join(save_dir, name +
                                   "_" + str(j) + "_label.png"))

                        curlist = [name, j, orgshape]
                        cube_train.append(curlist)
                else:
                    curlist = [name, j, orgshape]
                    im = Image.fromarray(image_slice)
                    im.save(os.path.join(save_dir, name + "_" + str(j) + ".png"))
                    label = Image.fromarray(label_slice)
                    label = label.convert("RGB")
                    label.save(os.path.join(save_dir, name +
                               "_" + str(j) + "_label.png"))
                    cube_train.append(curlist)

            random.shuffle(cube_train)
            cube_list += cube_train
        split_info = cube_list

        with open('split_comber_'+phase+'.pickle', 'wb') as handle:

            pickle.dump(split_info, handle)


class data_preloading():
    def __init__(self, dataset_split_file="./split_dataset_in_2.pickle", split_comber=SplitComb([64, 64, 64], [96, 96, 96])):
        self.train_val_test = dataset_split_file
        self.split_comber = split_comber

    def run(self, save_dir=None, phase='train'):

        dataset = load_pickle(self.train_val_test)
        split_info = []
        if phase == "train":
            # 1.train_set
            data_set = dataset["train"]
            assert len(data_set) == 200
        elif phase == "val":
            # 1.train_set
            data_set = dataset["val"]
            assert len(data_set) == 28
        else:
            data_set = dataset["test"]
            assert len(data_set) == 57
        cube_list = []

        for raw_path in data_set:
            # add by zhao
            print(phase, raw_path)
            # raw_path = os.path.join(self.datapath, raw_path.split('/')[-1] + '.gz')

            assert (os.path.exists(raw_path) is True)
            name = raw_path.split("/")[-1].split("_in.nii")[0]
            label_path = raw_path.replace('_in', '_label')
            assert (os.path.exists(label_path) is True)
            imgs, _, origin, spacing = load_itk_image(raw_path)
            print(imgs.shape)
            splits, nzhw, orgshape = self.split_comber.split_id(imgs)
            print("Name: %s, # of splits: %d" % (name, len(splits)))
            if label_path.endswith(".npy"):
                labels = np.load(label_path, allow_pickle=True)
            else:
                labels, _, _, _ = load_itk_image(label_path)

            cube_train = []

            for j in range(len(splits)):
                """
                check if this sub-volume cube is suitable
                """
                cursplit = splits[j]
                curcube = imgs[cursplit[0][0]:cursplit[0][1], cursplit[1]
                               [0]:cursplit[1][1], cursplit[2][0]:cursplit[2][1]]

                labelcube = labels[cursplit[0][0]:cursplit[0][1], cursplit[1]
                                   [0]:cursplit[1][1], cursplit[2][0]:cursplit[2][1]]
                curnumlabel = np.sum(labelcube)
                if phase == "train":
                    if curnumlabel > 0:  # filter out those zero-0 labels
                        curlist = [name, cursplit, j, nzhw, orgshape, 'Y']

                        # image_save_path = os.path.join(save_dir,name+"_"+str(j)+"_in.nii.gz")
                        # label_save_path = os.path.join(save_dir,name+"_"+str(j)+"_label.nii.gz")
                        # save_itk(curcube,origin,spacing,image_save_path)
                        # save_itk(labelcube,origin,spacing,label_save_path)
                        cube_train.append(curlist)
                else:
                    curlist = [name, cursplit, j, nzhw, orgshape, 'Y']
                    # image_save_path = os.path.join(save_dir,name+"_"+str(j)+"_in.nii.gz")
                    # label_save_path = os.path.join(save_dir,name+"_"+str(j)+"_label.nii.gz")
                    # save_itk(curcube,origin,spacing,image_save_path)
                    # save_itk(labelcube,origin,spacing,label_save_path)
                    cube_train.append(curlist)

            random.shuffle(cube_train)
            cube_list += cube_train
        split_info = cube_list

        with open('split_comber_'+phase+'.pickle', 'wb') as handle:

            pickle.dump(split_info, handle)


class PRM_preload(Dataset):
    """
    Generate dataloader
    """

    def __init__(self, config, phase='train'):
        """
        :param config: configuration from model
        :param phase: training or validation or testing
        :param split_comber: split-combination-er
        :param debug: debug mode to check few data
        :param random_select: use partly, randomly chosen data for training
        """
        assert (phase == 'train' or phase == 'val' or phase == 'test')
        self.phase = phase
        self.augtype = config['augtype']
        self.data_save_path = config['cube_savepath']

        self.patch_per_case = 5  # patches used per case if random training

        print(
            "-------------------------Load all data into memory---------------------------")
        """
		count the number of cases
		"""
        self.caseNumber = 0

        if self.phase == 'train':
            self.train_set = load_pickle(config['train_set'])
            self.cubelist = self.train_set

        elif self.phase == 'val':
            self.val_set = load_pickle(config['val_set'])
            self.cubelist = self.val_set

        else:
            self.test_set = load_pickle(config['test_set'])
            self.cubelist = self.test_set
            # self.cubelist = [i for i in self.test_set if i[0] == "CZSC000184"]

    def __len__(self):
        """
        :return: length of the dataset
        """

        return len(self.cubelist)

    def __getitem__(self, idx):
        """
        :param idx: index of the batch
        :return: wrapped data tensor and name, shape, origin, etc.
        """
        t = time.time()
        np.random.seed(int(str(t % 1)[2:7]))  # seed according to time
        curlist = self.cubelist[idx]
        # train: [data_name, cursplit, j, nzhw, orgshape, 'Y']
        # val/test: [data_name, cursplit, j, nzhw, orgshape, 'N']
        # print(curlist)
        curNameID = curlist[0]
        cursplit = curlist[1]
        curSplitID = curlist[2]
        curnzhw = curlist[3]
        curShapeOrg = curlist[4]
        curtransFlag = curlist[5]

        if self.phase == 'train' and curtransFlag == 'Y' and self.augtype['split_jitter'] is True:
            # random jittering during the training
            cursplit = augment_split_jittering(cursplit, curShapeOrg)

        ####################################################################

        curcube, _, origin, spacing = load_itk_image(os.path.join(
            self.data_save_path, curNameID+"_"+str(curSplitID)+"_in.nii.gz"))
        # curcube,origin,spacing = load_itk_image(curlist)
        curcube = (curcube.astype(np.float32))/255.0
        ####################################################################
        # calculate the coordinate for coordinate-aware convolution
        # start = [float(cursplit[0][0]), float(cursplit[1][0]), float(cursplit[2][0])]
        # normstart = ((np.array(start).astype('float')/np.array(curShapeOrg).astype('float'))-0.5)*2.0
        # crop_size = [curcube.shape[0],curcube.shape[1],curcube.shape[2]]
        # normsize = (np.array(crop_size).astype('float')/np.array(curShapeOrg).astype('float'))*2.0
        # xx, yy, zz = np.meshgrid(np.linspace(normstart[0], normstart[0]+normsize[0], int(crop_size[0])),
        # 						 np.linspace(normstart[1], normstart[1]+normsize[1], int(crop_size[1])),
        # 						 np.linspace(normstart[2], normstart[2]+normsize[2], int(crop_size[2])),
        # 						 indexing ='ij')
        # coord = np.concatenate([xx[np.newaxis,...], yy[np.newaxis,...], zz[np.newaxis,...]], 0).astype('float')
        # assert (coord.shape[0] == 3)

        label, _, _, _ = load_itk_image(os.path.join(
            self.data_save_path, curNameID+"_"+str(curSplitID)+"_label.nii.gz"))
        # label,_,_ = load_itk_image(curlist.replace(".nii.gz","_label.nii.gz"))
        label = label.astype('float')
        ####################################################################
        curNameID = [curNameID]
        curSplitID = [curSplitID]
        curnzhw = np.array(curnzhw)
        curShapeOrg = np.array(curShapeOrg)
        # start_coord = np.array(start)
        #######################################################################
        ######################## Data augmentation##############################

        if self.phase == 'train' and curtransFlag == 'Y':

            curcube, label, _ = augment(curcube, label, coord=None,
                                        ifflip=self.augtype['flip'], ifswap=self.augtype['swap'],
                                        ifsmooth=self.augtype['smooth'], ifjitter=self.augtype['jitter'])

        curcube = curcube[np.newaxis, ...]
        # label=label[np.newaxis,...]   # 如果不加类别数，会默认使用 输入数据中最大值，作为列别数。一般还是会加的
        x = torch.from_numpy(label).long()
        label = F.one_hot(x, num_classes=3)

        return torch.from_numpy(curcube).float(), torch.from_numpy(label.numpy().transpose(3, 0, 1, 2)).float(),\
            torch.from_numpy(origin),\
            torch.from_numpy(spacing), curNameID, curSplitID,\
            torch.from_numpy(curnzhw), torch.from_numpy(curShapeOrg)


class PRMData(Dataset):
    """
    Generate dataloader
    """

    def __init__(self, config, split_comber=None, phase='train'):
        """
        :param config: configuration from model
        :param phase: training or validation or testing
        :param split_comber: split-combination-er
        :param debug: debug mode to check few data
        :param random_select: use partly, randomly chosen data for training
        """
        assert (phase == 'train' or phase == 'val' or phase == 'test')
        self.phase = phase
        self.augtype = config['augtype']
        self.split_comber = split_comber
        self.patch_per_case = 5  # patches used per case if random training

        """
		specify the path and data split
		"""

        self.datapath = config['dataset_path']
        self.dataset = load_pickle(config['dataset_split'])

        print(
            "-------------------------Load all data into memory---------------------------")
        """
		count the number of cases
		"""

        cubelist = []
        labellist = []
        all_image_memory = {}
        all_label_memory = {}
        self.caseNumber = 0

        if self.phase == 'train':
            train_set = self.dataset["train"]
            assert len(train_set) == 200
            file_num = len(train_set)
            self.caseNumber += file_num
            for raw_path in train_set:
                # add by zhao
                # print("train",raw_path)
                # raw_path = os.path.join(self.datapath, raw_path.split('/')[-1] + '.gz')
                assert (os.path.exists(raw_path) is True)
                name = raw_path.split("/")[-1].split("_in")[0]
                label_path = raw_path.replace('_in.nii.gz', '_label.nii.gz')
                assert (os.path.exists(label_path) is True)
                imgs, origin, spacing = load_itk_image(raw_path)
                splits, nzhw, orgshape = split_comber.split_id(imgs)
                print("Name: %s, # of splits: %d" % (name, len(splits)))
                labels, _, _ = load_itk_image(label_path)
                all_image_memory[name] = [imgs, origin, spacing]
                all_label_memory[name] = labels

                cube_train = []

                for j in range(len(splits)):
                    """
                    check if this sub-volume cube is suitable
                    """
                    cursplit = splits[j]
                    labelcube = labels[cursplit[0][0]:cursplit[0][1], cursplit[1]
                                       [0]:cursplit[1][1], cursplit[2][0]:cursplit[2][1]]
                    curnumlabel = np.sum(labelcube)
                    if curnumlabel > 0:  # filter out those zero-0 labels
                        curlist = [name, cursplit, j, nzhw, orgshape, 'Y']
                        cube_train.append(curlist)

                random.shuffle(cube_train)
                cubelist += cube_train

        elif self.phase == 'val':
            val_set = self.dataset["val"]
            assert len(val_set) == 28
            file_num = len(val_set)
            self.caseNumber += file_num
            for raw_path in val_set:
                # add by zhao
                # print("train",raw_path)
                # raw_path = os.path.join(self.datapath, raw_path.split('/')[-1] + '.gz')
                assert (os.path.exists(raw_path) is True)
                name = raw_path.split("/")[-1].split("_in")[0]
                label_path = raw_path.replace('_in.nii.gz', '_label.nii.gz')
                assert (os.path.exists(label_path) is True)
                imgs, origin, spacing = load_itk_image(raw_path)
                splits, nzhw, orgshape = split_comber.split_id(imgs)
                print("Name: %s, # of splits: %d" % (name, len(splits)))
                labels, _, _ = load_itk_image(label_path)
                all_image_memory[name] = [imgs, origin, spacing]
                all_label_memory[name] = labels

                cube_train = []

                for j in range(len(splits)):
                    """
                    check if this sub-volume cube is suitable
                    """
                    cursplit = splits[j]

                    curlist = [name, cursplit, j, nzhw, orgshape, 'Y']
                    cube_train.append(curlist)
                cubelist += cube_train

        else:
            test_set = self.dataset["test"]
# assert len(test_set)==56
            file_num = len(test_set)
            self.caseNumber += file_num
            for raw_path in test_set:
                # add by zhao
                # print("train",raw_path)
                # raw_path = os.path.join(self.datapath, raw_path.split('/')[-1] + '.gz')
                assert (os.path.exists(raw_path) is True)
                name = raw_path.split("/")[-1].split("_in")[0]
                label_path = raw_path.replace('_in.nii.gz', '_label.nii.gz')
                assert (os.path.exists(label_path) is True)
                imgs, _, origin, spacing = load_itk_image(raw_path)
                splits, nzhw, orgshape = split_comber.split_id(imgs)
                print("Name: %s, # of splits: %d" % (name, len(splits)))
                labels, _, _, _ = load_itk_image(label_path)
                all_image_memory[name] = [imgs, origin, spacing]
                all_label_memory[name] = labels

                cube_train = []

                for j in range(len(splits)):
                    """
                    check if this sub-volume cube is suitable
                    """
                    cursplit = splits[j]

                    curlist = [name, cursplit, j, nzhw, orgshape, 'N']
                    cube_train.append(curlist)
                cubelist += cube_train

        self.allimgdata_memory = all_image_memory
        self.alllabeldata_memory = all_label_memory
        random.shuffle(cubelist)
        self.cubelist = cubelist

        print('---------------------Initialization Done---------------------')
        print('Phase: %s total cubelist number: %d' %
              (self.phase, len(self.cubelist)))
        print()

    def __len__(self):
        """
        :return: length of the dataset
        """

        return len(self.cubelist)

    def __getitem__(self, idx):
        """
        :param idx: index of the batch
        :return: wrapped data tensor and name, shape, origin, etc.
        """
        t = time.time()
        np.random.seed(int(str(t % 1)[2:7]))  # seed according to time

        curlist = self.cubelist[idx]

        # train: [data_name, cursplit, j, nzhw, orgshape, 'Y']
        # val/test: [data_name, cursplit, j, nzhw, orgshape, 'N']

        curNameID = curlist[0]
        cursplit = curlist[1]
        curSplitID = curlist[2]
        curnzhw = curlist[3]
        curShapeOrg = curlist[4]
        curtransFlag = curlist[5]

        if self.phase == 'train' and curtransFlag == 'Y' and self.augtype['split_jitter'] is True:
            # random jittering during the training
            cursplit = augment_split_jittering(cursplit, curShapeOrg)

        ####################################################################
        imginfo = self.allimgdata_memory[curNameID]
        imgs, origin, spacing = imginfo[0], imginfo[1], imginfo[2]
        curcube = imgs[cursplit[0][0]:cursplit[0][1], cursplit[1]
                       [0]:cursplit[1][1], cursplit[2][0]:cursplit[2][1]]
        curcube = (curcube.astype(np.float32))/255.0
        ####################################################################
        # calculate the coordinate for coordinate-aware convolution
        start = [float(cursplit[0][0]), float(
            cursplit[1][0]), float(cursplit[2][0])]
        normstart = ((np.array(start).astype('float') /
                     np.array(curShapeOrg).astype('float'))-0.5)*2.0
        crop_size = [curcube.shape[0], curcube.shape[1], curcube.shape[2]]
        normsize = (np.array(crop_size).astype('float') /
                    np.array(curShapeOrg).astype('float'))*2.0
        xx, yy, zz = np.meshgrid(np.linspace(normstart[0], normstart[0]+normsize[0], int(crop_size[0])),
                                 np.linspace(
                                     normstart[1], normstart[1]+normsize[1], int(crop_size[1])),
                                 np.linspace(
                                     normstart[2], normstart[2]+normsize[2], int(crop_size[2])),
                                 indexing='ij')
        coord = np.concatenate(
            [xx[np.newaxis, ...], yy[np.newaxis, ...], zz[np.newaxis, ...]], 0).astype('float')
        assert (coord.shape[0] == 3)

        label = self.alllabeldata_memory[curNameID]
        # label = (label > 0)
        label = label.astype('float')
        label = label[cursplit[0][0]:cursplit[0][1], cursplit[1]
                      [0]:cursplit[1][1], cursplit[2][0]:cursplit[2][1]]

        ####################################################################
        curNameID = [curNameID]
        curSplitID = [curSplitID]
        curnzhw = np.array(curnzhw)
        curShapeOrg = np.array(curShapeOrg)
        # start_coord = np.array(start)
        #######################################################################
        ######################## Data augmentation##############################

        if self.phase == 'train' and curtransFlag == 'Y':

            curcube, label, coord = augment(curcube, label, coord,
                                            ifflip=self.augtype['flip'], ifswap=self.augtype['swap'],
                                            ifsmooth=self.augtype['smooth'], ifjitter=self.augtype['jitter'])

        curcube = curcube[np.newaxis, ...]
        x = torch.from_numpy(label).long()
        # 如果不加类别数，会默认使用 输入数据中最大值，作为列别数。一般还是会加的
        label = F.one_hot(x, num_classes=3)
        # print(label.shape)

        # label = label[np.newaxis,...]

        return torch.from_numpy(curcube).float(), torch.from_numpy(label.numpy().transpose(3, 0, 1, 2)).float(),\
            torch.from_numpy(coord).float(), torch.from_numpy(origin),\
            torch.from_numpy(spacing), curNameID, curSplitID,\
            torch.from_numpy(curnzhw), torch.from_numpy(curShapeOrg)


# 随机抖动
def augment_split_jittering(cursplit, curShapeOrg):
    # orgshape [z, h, w]
    zstart, zend = cursplit[0][0], cursplit[0][1]
    hstart, hend = cursplit[1][0], cursplit[1][1]
    wstart, wend = cursplit[2][0], cursplit[2][1]
    curzjitter, curhjitter, curwjitter = 0, 0, 0
    if zend - zstart <= 3:
        jitter_range = (zend - zstart) * 32
    else:
        jitter_range = (zend - zstart) * 2
    # print("jittering range ", jitter_range)
    jitter_range_half = jitter_range//2

    t = 0
    while t < 10:
        if zstart == 0:
            curzjitter = int(np.random.rand() * jitter_range)
        elif zend == curShapeOrg[0]:
            curzjitter = -int(np.random.rand() * jitter_range)
        else:
            curzjitter = int(np.random.rand() * jitter_range) - \
                jitter_range_half
        t += 1
        if (curzjitter + zstart >= 0) and (curzjitter + zend < curShapeOrg[0]):
            break

    t = 0
    while t < 10:
        if hstart == 0:
            curhjitter = int(np.random.rand() * jitter_range)
        elif hend == curShapeOrg[1]:
            curhjitter = -int(np.random.rand() * jitter_range)
        else:
            curhjitter = int(np.random.rand() * jitter_range) - \
                jitter_range_half
        t += 1
        if (curhjitter + hstart >= 0) and (curhjitter + hend < curShapeOrg[1]):
            break

    t = 0
    while t < 10:
        if wstart == 0:
            curwjitter = int(np.random.rand() * jitter_range)
        elif wend == curShapeOrg[2]:
            curwjitter = -int(np.random.rand() * jitter_range)
        else:
            curwjitter = int(np.random.rand() * jitter_range) - \
                jitter_range_half
        t += 1
        if (curwjitter + wstart >= 0) and (curwjitter + wend < curShapeOrg[2]):
            break

    if (curzjitter + zstart >= 0) and (curzjitter + zend < curShapeOrg[0]):
        cursplit[0][0] = curzjitter + zstart
        cursplit[0][1] = curzjitter + zend

    if (curhjitter + hstart >= 0) and (curhjitter + hend < curShapeOrg[1]):
        cursplit[1][0] = curhjitter + hstart
        cursplit[1][1] = curhjitter + hend

    if (curwjitter + wstart >= 0) and (curwjitter + wend < curShapeOrg[2]):
        cursplit[2][0] = curwjitter + wstart
        cursplit[2][1] = curwjitter + wend
    # print ("after ", cursplit)
    return cursplit


def augment(sample, label, coord=None, ifflip=True, ifswap=False, ifsmooth=False, ifjitter=False):
    """
    :param sample, the cropped sample input
    :param label, the corresponding sample ground-truth
    :param coord, the corresponding sample coordinates
    :param ifflip, flag for random flipping
    :param ifswap, flag for random swapping
    :param ifsmooth, flag for Gaussian smoothing on the CT image
    :param ifjitter, flag for intensity jittering on the CT image
    :return: augmented training samples
    """
    if ifswap:
        if sample.shape[0] == sample.shape[1] and sample.shape[0] == sample.shape[2]:
            axisorder = np.random.permutation(3)

            sample = np.transpose(sample, axisorder)
            if label is not None:
                label = np.transpose(label, axisorder)
            if coord is not None:
                coord = np.transpose(coord, np.concatenate([[0], axisorder+1]))

    if ifflip:
        flipid = np.random.randint(2)*2-1
        sample = np.ascontiguousarray(sample[:, :, ::flipid])
        if label is not None:
            label = np.ascontiguousarray(label[:, :, ::flipid])
        if coord is not None:
            coord = np.ascontiguousarray(coord[:, :, :, ::flipid])

    prob_aug = random.random()
    if ifjitter and prob_aug > 0.5:
        ADD_INT = (np.random.rand(
            sample.shape[0], sample.shape[1], sample.shape[2])*2 - 1)*10
        ADD_INT = ADD_INT.astype('float')
        if label is not None:
            cury_roi = label*ADD_INT/255.0  # 对mask区域进行随机抖动
        else:
            cury_roi = np.zeros(
                (sample.shape[0], sample.shape[1], sample.shape[2]))
        sample += cury_roi
        sample[sample < 0] = 0
        sample[sample > 1] = 1

    prob_aug = random.random()
    if ifsmooth and prob_aug > 0.5:
        sigma = np.random.rand()
        if sigma > 0.5:
            sample = gaussian_filter(sample, sigma=1.0)

    return sample, label, coord


class Two_D_PRMData(Dataset):
    """
    Generate dataloader
    """

    def __init__(self, config, split_comber=None, phase='train'):
        """
        :param config: configuration from model
        :param phase: training or validation or testing
        :param split_comber: split-combination-er
        :param debug: debug mode to check few data
        :param random_select: use partly, randomly chosen data for training
        """
        assert (phase == 'train' or phase == 'val' or phase == 'test')
        self.phase = phase
        self.augtype = config['augtype']
        self.split_comber = split_comber
        self.patch_per_case = 5  # patches used per case if random training

        """
		specify the path and data split
		"""

        self.datapath = config['dataset_path']
        self.dataset = load_pickle(config['dataset_split'])

        print(
            "-------------------------Load all data into memory---------------------------")
        """
		count the number of cases
		"""

        cubelist = []
        labellist = []
        all_image_memory = {}
        all_label_memory = {}
        self.caseNumber = 0

        if self.phase == 'train':
            train_set = self.dataset["train"]
            assert len(train_set) == 68
            file_num = len(train_set)
            self.caseNumber += file_num
            for raw_path in train_set:
                # add by zhao
                # print("train",raw_path)
                # raw_path = os.path.join(self.datapath, raw_path.split('/')[-1] + '.gz')
                assert (os.path.exists(raw_path) is True)
                name = raw_path.split("/")[-1].split("_in")[0]
                label_path = raw_path.replace('_in.nii.gz', '_label.nii.gz')
                assert (os.path.exists(label_path) is True)
                imgs, origin, spacing = load_itk_image(raw_path)
                print(imgs)
                print("Name: %s, # of splits: %d" % (name, len(splits)))
                labels, _, _ = load_itk_image(label_path)
                all_image_memory[name] = [imgs, origin, spacing]
                all_label_memory[name] = labels
                cube_train = []

                for j in range(len(splits)):
                    """
                    check if this sub-volume cube is suitable
                    """
                    cursplit = splits[j]
                    labelcube = labels[cursplit[0][0]:cursplit[0][1], cursplit[1]
                                       [0]:cursplit[1][1], cursplit[2][0]:cursplit[2][1]]
                    curnumlabel = np.sum(labelcube)
                    if curnumlabel > 0:  # filter out those zero-0 labels
                        curlist = [name, cursplit, j, nzhw, orgshape, 'Y']
                        cube_train.append(curlist)

                random.shuffle(cube_train)
                cubelist += cube_train

        elif self.phase == 'val':
            val_set = self.dataset["val"]
            assert len(val_set) == 10
            file_num = len(val_set)
            self.caseNumber += file_num
            for raw_path in val_set:
                # add by zhao
                # print("train",raw_path)
                # raw_path = os.path.join(self.datapath, raw_path.split('/')[-1] + '.gz')
                assert (os.path.exists(raw_path) is True)
                name = raw_path.split("/")[-1].split("_in")[0]
                label_path = raw_path.replace('_in.nii.gz', '_label.nii.gz')
                assert (os.path.exists(label_path) is True)
                imgs, origin, spacing = load_itk_image(raw_path)
                splits, nzhw, orgshape = split_comber.split_id(imgs)
                print("Name: %s, # of splits: %d" % (name, len(splits)))
                labels, _, _ = load_itk_image(label_path)
                all_image_memory[name] = [imgs, origin, spacing]
                all_label_memory[name] = labels

                cube_train = []

                for j in range(len(splits)):
                    """
                    check if this sub-volume cube is suitable
                    """
                    cursplit = splits[j]

                    curlist = [name, cursplit, j, nzhw, orgshape, 'Y']
                    cube_train.append(curlist)
                cubelist += cube_train

        else:
            test_set = self.dataset["test"]
            assert len(test_set) == 19
            file_num = len(test_set)
            self.caseNumber += file_num
            for raw_path in test_set:
                # add by zhao
                # print("train",raw_path)
                # raw_path = os.path.join(self.datapath, raw_path.split('/')[-1] + '.gz')
                assert (os.path.exists(raw_path) is True)
                name = raw_path.split("/")[-1].split("_in")[0]
                label_path = raw_path.replace('_in.nii.gz', '_label.nii.gz')
                assert (os.path.exists(label_path) is True)
                imgs, origin, spacing = load_itk_image(raw_path)
                splits, nzhw, orgshape = split_comber.split_id(imgs)
                print("Name: %s, # of splits: %d" % (name, len(splits)))
                labels, _, _ = load_itk_image(label_path)
                all_image_memory[name] = [imgs, origin, spacing]
                all_label_memory[name] = labels

                cube_train = []

                for j in range(len(splits)):
                    """
                    check if this sub-volume cube is suitable
                    """
                    cursplit = splits[j]

                    curlist = [name, cursplit, j, nzhw, orgshape, 'N']
                    cube_train.append(curlist)
                cubelist += cube_train

        self.allimgdata_memory = all_image_memory
        self.alllabeldata_memory = all_label_memory
        random.shuffle(cubelist)
        self.cubelist = cubelist

        print('---------------------Initialization Done---------------------')
        print('Phase: %s total cubelist number: %d' %
              (self.phase, len(self.cubelist)))
        print()

    def __len__(self):
        """
        :return: length of the dataset
        """

        return len(self.cubelist)

    def __getitem__(self, idx):
        """
        :param idx: index of the batch
        :return: wrapped data tensor and name, shape, origin, etc.
        """
        t = time.time()
        np.random.seed(int(str(t % 1)[2:7]))  # seed according to time

        curlist = self.cubelist[idx]

        # train: [data_name, cursplit, j, nzhw, orgshape, 'Y']
        # val/test: [data_name, cursplit, j, nzhw, orgshape, 'N']

        curNameID = curlist[0]
        cursplit = curlist[1]
        curSplitID = curlist[2]
        curnzhw = curlist[3]
        curShapeOrg = curlist[4]
        curtransFlag = curlist[5]

        if self.phase == 'train' and curtransFlag == 'Y' and self.augtype['split_jitter'] is True:
            # random jittering during the training
            cursplit = augment_split_jittering(cursplit, curShapeOrg)

        ####################################################################
        imginfo = self.allimgdata_memory[curNameID]
        imgs, origin, spacing = imginfo[0], imginfo[1], imginfo[2]
        curcube = imgs[cursplit[0][0]:cursplit[0][1], cursplit[1]
                       [0]:cursplit[1][1], cursplit[2][0]:cursplit[2][1]]
        curcube = (curcube.astype(np.float32))/255.0
        ####################################################################
        # calculate the coordinate for coordinate-aware convolution
        start = [float(cursplit[0][0]), float(
            cursplit[1][0]), float(cursplit[2][0])]
        normstart = ((np.array(start).astype('float') /
                     np.array(curShapeOrg).astype('float'))-0.5)*2.0
        crop_size = [curcube.shape[0], curcube.shape[1], curcube.shape[2]]
        normsize = (np.array(crop_size).astype('float') /
                    np.array(curShapeOrg).astype('float'))*2.0
        xx, yy, zz = np.meshgrid(np.linspace(normstart[0], normstart[0]+normsize[0], int(crop_size[0])),
                                 np.linspace(
                                     normstart[1], normstart[1]+normsize[1], int(crop_size[1])),
                                 np.linspace(
                                     normstart[2], normstart[2]+normsize[2], int(crop_size[2])),
                                 indexing='ij')
        coord = np.concatenate(
            [xx[np.newaxis, ...], yy[np.newaxis, ...], zz[np.newaxis, ...]], 0).astype('float')
        assert (coord.shape[0] == 3)

        label = self.alllabeldata_memory[curNameID]
        # label = (label > 0)
        label = label.astype('float')
        label = label[cursplit[0][0]:cursplit[0][1], cursplit[1]
                      [0]:cursplit[1][1], cursplit[2][0]:cursplit[2][1]]

        ####################################################################
        curNameID = [curNameID]
        curSplitID = [curSplitID]
        curnzhw = np.array(curnzhw)
        curShapeOrg = np.array(curShapeOrg)
        # start_coord = np.array(start)
        #######################################################################
        ######################## Data augmentation##############################

        if self.phase == 'train' and curtransFlag == 'Y':

            curcube, label, coord = augment(curcube, label, coord,
                                            ifflip=self.augtype['flip'], ifswap=self.augtype['swap'],
                                            ifsmooth=self.augtype['smooth'], ifjitter=self.augtype['jitter'])

        curcube = curcube[np.newaxis, ...]
        x = torch.from_numpy(label).long()
        # 如果不加类别数，会默认使用 输入数据中最大值，作为列别数。一般还是会加的
        label = F.one_hot(x, num_classes=4)
        # print(y.shape)

        # label = label[np.newaxis,...]

        return torch.from_numpy(curcube).float(), torch.from_numpy(label.numpy().transpose(3, 0, 1, 2)).float(),\
            torch.from_numpy(coord).float(), torch.from_numpy(origin),\
            torch.from_numpy(spacing), curNameID, curSplitID,\
            torch.from_numpy(curnzhw), torch.from_numpy(curShapeOrg)


class PRM_DS_Data(Dataset):
    """
    Generate dataloader
    """

    def __init__(self, config, phase='train'):
        """
        :param config: configuration from model
        :param phase: training or validation or testing
        :param split_comber: split-combination-er
        :param debug: debug mode to check few data
        :param random_select: use partly, randomly chosen data for training
        """
        assert (phase == 'train' or phase == 'val' or phase == 'test')
        self.phase = phase
        self.augtype = config['augtype']

        self.datapath = config['DS_dataset_path']
        self.dataset = load_pickle(config['dataset_split'])
        # sprint(self.dataset )
        print(
            "-------------------------Load all data into memory---------------------------")
        self.caseNumber = 0
        if phase == "train":
            self.data = self.dataset["train"]
            assert len(self.data) == 200
        elif phase == "test":
            self.data = self.dataset["test"]
        file_num = len(self.data)
        self.caseNumber += file_num
        # self.data = random.shuffle(self.data)

    def __len__(self):
        """
        :return: length of the dataset
        """

        return self.caseNumber

    def __getitem__(self, idx):
        """
        :param idx: index of the batch
        :return: wrapped data tensor and name, shape, origin, etc.
        """
        t = time.time()
        np.random.seed(int(str(t % 1)[2:7]))  # seed according to time

        image_path = os.path.join(self.datapath, self.data[idx].split("/")[-1])
        label_path = os.path.join(self.datapath, self.data[idx].split(
            "/")[-1].replace("_in", "_label"))

        ####################################################################
        img, _, origin, spacing = load_itk_image(image_path)
        label, _, origin, spacing = load_itk_image(label_path)

        img = (img.astype(np.float32))/255.0
        label = label.astype('float')

        if self.phase == 'train':

            img, label, coord = augment(img, label, coord=None,
                                        ifflip=self.augtype['flip'], ifswap=self.augtype['swap'],
                                        ifsmooth=self.augtype['smooth'], ifjitter=self.augtype['jitter'])

        img = img[np.newaxis, ...]
        x = torch.from_numpy(label).long()
        # 如果不加类别数，会默认使用 输入数据中最大值，作为列别数。一般还是会加的
        label = F.one_hot(x, num_classes=3)
        # print(label.shape)

        # label = label[np.newaxis,...]

        return torch.from_numpy(img).float(), torch.from_numpy(label.numpy().transpose(3, 0, 1, 2)).float(),\
            torch.from_numpy(origin),\
            torch.from_numpy(spacing)


if __name__ == "__main__":

    # import torch
    # from torch.utils.data import DataLoader
    # model = import_module('backbone')#相对导入，py文件名是可选的，可选择导入哪种模型文件
    # config, _ = model.get_model()
    # dataset_train = PRMData(
    # 	data_path = "/data2/PRM/data/processed2_data",
    # 	split_info_path = "/data2/PRM/split_comber.pickle",
    # 	config=config,
    # 	phase='train')

    # train_loader = DataLoader(
    # 	dataset_train,
    # 	batch_size=args.batch_size,
    # 	shuffle=True,
    # 	num_workers=args.workers,
    # 	pin_memory=True)
    savedir = "./data/preload_data"
    test_preload = data_preloading(
        dataset_split_file="./split_dataset_in_2.pickle", split_comber=SplitComb([64, 64, 64], [96, 96, 96]))
    test_preload.run(save_dir=savedir, phase="train")

    # dataset_split_file="./split_dataset_in_2.pickle"
    # test_set = load_pickle(dataset_split_file)["test"]
    # print(test_set)

    # savedir = "/home1/mingyuezhao/PRM/data/processed2_data_2D"
    # train_loader = data_2D_preloading(dataset_split_file="./split_dataset_in_2.pickle")
    # train_loader.run(save_dir = savedir,phase="train")

    # img = Image.open("/home1/mingyuezhao/PRM/data/processed2_data_2D/CZSC000308_254_label.png")
    # print(np.unique(np.array(img)))
