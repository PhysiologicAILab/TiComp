# import matplotlib.pyplot as plt
from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image

def mask2onehot(mask, num_classes):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector

    """
    _mask = [mask == i for i in range(num_classes)]
    return np.array(_mask).astype(np.uint8)


def onehot2mask(mask):
    """
    Converts a mask (K, H, W) to (H,W)
    """
    _mask = np.argmax(mask, axis=0).astype(np.uint8)
    return _mask


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, augObj, mode='vis', n_classes=1, apply_aug=False, learn_occlusion_maps=False, norm_mode=1, generate_fakes=False):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.augObj = augObj
        self.mode = mode
        self.n_classes = n_classes
        self.apply_aug = apply_aug
        self.learn_occlusion_maps = learn_occlusion_maps
        self.norm_mode = norm_mode
        self.generate_fakes = generate_fakes

        if self.mode == "therm":
            self.ext = ".npy"
        else:
            self.ext = ".png"

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.') and self.ext in file]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, np_matrix, mode, mask_classes=0, norm_mode=1):

        img_nd = np_matrix.copy()
        img_trans = None

        if mask_classes > 0 and mode != 'therm_ae':
            pass
        else:
            if mode in ['vis', 'rgb'] :
                if img_nd.max() > 1.0:
                    img_nd = img_nd / 255.0
            elif mode == 'therm' or 'therm_ae':
                min_T = img_nd.min()
                max_T = img_nd.max()
                if norm_mode == 1:
                    if (max_T - min_T) != 0:
                        img_nd = (img_nd - min_T) / (max_T - min_T)
                    elif max_T != 0:
                        img_nd = img_nd / max_T
                    else:
                        pass
                else:
                    if (max_T - min_T) != 0:
                        img_nd = 2*((img_nd - min_T) / (max_T - min_T)) - 1
                    elif max_T != 0:
                        img_nd = 2*(img_nd / max_T) - 1
                    else:
                        pass

                # plt.imshow(img_nd); plt.show()
                # exit()
            else:
                pass
                print('Warning: No preprocessing executed')

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)
        
        img_trans = img_nd.transpose((2, 0, 1))
        # if mask_classes == 0:
        #     # HWC to CHW
        #     img_trans = img_nd.transpose((2, 0, 1))
        # else:
        #     img_trans = img_nd
        #     # pass

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + self.ext)
        img_file = glob(self.imgs_dir + idx + self.ext)

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'

        if self.mode == 'therm':
            input_img = np.load(img_file[0])/1000.0
            target_mask = np.load(mask_file[0])
        elif self.mode == 'vis':
            input_img = np.load(img_file[0])
            target_mask = np.load(mask_file[0])
        elif self.mode == 'rgb':
            input_img = Image.open(img_file[0])
            input_img = np.asarray(input_img)
            target_mask = Image.open(mask_file[0]).convert('L')
            target_mask = np.asarray(target_mask)
        else:
            f'Invalid mode specified: {self.mode}. It should be therm/ vis/ rgb'
            exit()

        assert input_img.shape == target_mask.shape, \
            f'Image and mask {idx} should be the same size, but are {input_img.shape} and {target_mask.shape}'

        if self.n_classes == 6:
            # Remapping of segmentation classes
            target_mask[target_mask == 4] = 3
            target_mask[target_mask == 5] = 4
            target_mask[target_mask == 6] = 4
            target_mask[target_mask == 7] = 5
        elif self.n_classes == 2:               #Binary Classification
            target_mask[target_mask > 0] = 1


        if self.apply_aug:
            # if self.augObj.apply_thermal_random_noise:
            #     ths = float(self.th_dict[os.path.basename(img_file[0])])/1000.0
            # else:
            #     ths = None
            input_img, target_mask, gan_target, _, occ_mask, _ = self.augObj.transform(input_img, target_mask, target_mask_fake=None, ths=None)
        else:
            gan_target = None
            occ_mask = None

        one_hot_target_mask = mask2onehot(target_mask, num_classes=self.n_classes)
        
        if self.generate_fakes:
            _, h, w = one_hot_target_mask.shape
            if self.apply_aug and self.augObj.apply_aug_occlusion and self.learn_occlusion_maps:
                one_hot_target_mask_temp = np.zeros((self.n_classes+1, h, w))
                one_hot_target_mask_temp[0:self.n_classes, :, :] = one_hot_target_mask
                one_hot_target_mask_temp[-1, :, :] = occ_mask
                one_hot_target_mask = one_hot_target_mask_temp

            one_hot_target_mask_fake = self.fake_gen_obj.fake_seg_generator(target_mask=one_hot_target_mask, mode='one_hot')

        input_img = self.preprocess(input_img, self.mode, mask_classes=0, norm_mode=self.norm_mode)
        input_img = torch.from_numpy(input_img).type(torch.FloatTensor)
        one_hot_target_mask = torch.from_numpy(one_hot_target_mask).type(torch.FloatTensor)

        data_dict = {}
        data_dict['input_img'] = input_img
        data_dict['one_hot_target_mask'] = one_hot_target_mask

        if np.any(gan_target) != None:
            gan_target = self.preprocess(gan_target, self.mode, mask_classes=0, norm_mode=self.norm_mode)
            gan_target = torch.from_numpy(gan_target).type(torch.FloatTensor)
            data_dict['gan_target'] = gan_target
        
        if self.generate_fakes:
            one_hot_target_mask_fake = torch.from_numpy(one_hot_target_mask_fake).type(torch.FloatTensor)
            data_dict['one_hot_target_mask_fake'] = one_hot_target_mask_fake

        if np.any(occ_mask) != None:
            occ_mask = self.preprocess(np.float32(occ_mask), self.mode, mask_classes=self.n_classes, norm_mode=self.norm_mode)
            occ_mask = torch.from_numpy(occ_mask).type(torch.FloatTensor)
            data_dict['occ_mask'] = occ_mask            

        return data_dict

