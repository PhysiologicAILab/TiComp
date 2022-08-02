import time
import json
import copy
import torch
import torch.nn as nn
from skimage.transform import resize
import numpy as np

from seg.utils.utils import load_checkpoint
from seg.utils.thermal_dataset import BasicDataset

activation = nn.LogSoftmax(dim=1)

class ThermSeg():
    def __init__(self, trained_model_path, config_path, mode="therm"):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode

        fp = open(config_path, 'r')
        config_dict = json.loads(fp.read())
        fp.close()

        n_channels = int(config_dict["hp"]["n_channels"])
        n_classes = int(config_dict["hp"]["n_classes"])
        gen_arch = config_dict["hp"]["gen_arch"]

        learn_occlusion_maps = False
        # cls_labels = ['chin', 'mouth', 'eye', 'eyebrow', 'nose']

        if "learn_occlusion_maps" in config_dict["hp"]:
            learn_occlusion_maps = bool(config_dict["hp"]["learn_occlusion_maps"])

        if learn_occlusion_maps:
            n_classes = n_classes+1

        if gen_arch == "GSNET":
            from seg.models.GSNet import Generator
            self.net = Generator(n_channels=n_channels, n_classes=n_classes)

        elif gen_arch == "GSNET_SN":
            from seg.models.GSNet_SN import Generator
            self.net = Generator(n_channels=n_channels, n_classes=n_classes)

        elif gen_arch == "UNET":
            from seg.models.UNET import Generator
            self.net = Generator(n_channels=n_channels, n_classes=n_classes)

        elif gen_arch == "AttUNET":
            from seg.models.AttUNET import Generator
            self.net = Generator(n_channels=n_channels, n_classes=n_classes)

        elif "DeepLab" in gen_arch:
            from seg.models.deeplab import DeepLab as Generator
            if "xception" in gen_arch:
                self.net = Generator(backbone='xception', in_channels=n_channels, output_stride=16, num_classes=n_classes, sync_bn=True, freeze_bn=False, pretrained=False)
            elif "resnet" in gen_arch:
                self.net = Generator(backbone='resnet', in_channels=n_channels, output_stride=16, num_classes=n_classes, sync_bn=True, freeze_bn=False, pretrained=False)
            elif "drn" in gen_arch:
                self.net = Generator(backbone='drn', in_channels=n_channels, output_stride=8, num_classes=n_classes, sync_bn=True, freeze_bn=False, pretrained=False)

        self.net = self.net.to(device=self.device)
        print("=> Loading checkpoint:", trained_model_path)
        try:
            load_checkpoint(trained_model_path, self.net, self.device, strict=True, load_opt=False)
        except:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.to(device=self.device)
            load_checkpoint(trained_model_path, self.net, self.device, strict=True, load_opt=False)

        self.net.eval()


    def run_inference(self, input_img):

        # print("Input Image Shape:", input_img.shape)

        # 60 FPS A35SC camera - 320x256
        x0, x1 = 32, input_img.shape[1]-32
        y0, y1 = 0, input_img.shape[0]
        input_img = input_img[y0:y1, x0:x1]

        # # 30 FPS A65SC camera - 640x512
        # x0, x1 = 64, input_img.shape[1]-64
        # y0, y1 = 0, input_img.shape[0]
        # input_img = input_img[y0:y1, x0:x1]
        # input_img = resize(input_img, (256, 256))
        # # print("New Image Shape:", input_img.shape)

        # # 30 FPS A65SC camera - 640x512
        # x0, x1 = 192, input_img.shape[1]-192
        # y0, y1 = 128, input_img.shape[0]-128
        # input_img = input_img[y0:y1, x0:x1]
        # # input_img = resize(input_img, (256, 256))
        # input_img = np.fliplr(input_img)
        # # print("New Image Shape:", input_img.shape)

        input_img_org = copy.deepcopy(input_img)

        t1 = time.time()
        with torch.no_grad():
            input_img = torch.from_numpy(BasicDataset.preprocess(input_img, self.mode, mask_classes=0, norm_mode=2))
            input_img = input_img.unsqueeze(0)
            input_img = input_img.to(device=self.device, dtype=torch.float32)

            pred_mask = self.net(input_img)
            pred_mask = torch.argmax(activation(pred_mask).exp(), dim=1).squeeze().cpu().numpy()

            time_taken = time.time() - t1

        return input_img_org, pred_mask, time_taken

