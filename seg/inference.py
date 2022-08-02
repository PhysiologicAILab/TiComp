import time
import json
import copy
import torch
import torch.nn as nn
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt

from seg.utils.utils import load_checkpoint
from seg.utils.thermal_dataset import BasicDataset
from seg.utils.module_runner import ModuleRunner
from seg.utils.data_container import DataContainer
from seg.models.model_manager import ModelManager
import seg.utils.transforms as trans

activation = nn.LogSoftmax(dim=1)

def softmax(X, axis=0):
    max_prob = np.max(X, axis=axis, keepdims=True)
    X -= max_prob
    X = np.exp(X)
    sum_prob = np.sum(X, axis=axis, keepdims=True)
    X /= sum_prob
    return X


class ThermSeg():
    def __init__(self, configer): #, trained_model_path, config_path, mode="therm"):

        self.configer = configer
        self.module_runner = ModuleRunner(self.configer)
        self.model_manager = ModelManager(self.configer)
        self.seg_net = self.model_manager.semantic_segmentor()
        self.seg_net = self.module_runner.load_net(self.seg_net)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = self.configer.get('data', 'input_mode')

        self.seg_net.eval()
        self.img_transform = trans.Compose([
            trans.ToTensor(),
            trans.NormalizeThermal(norm_mode=self.configer.get('normalize', 'norm_mode')), ])
        size_mode = self.configer.get('test', 'data_transformer')['size_mode']
        self.is_stack = size_mode != 'diverse_size'

    def run_inference(self, input_img):

        # # print("Input Image Shape:", input_img.shape)

        # # 60 FPS A35SC camera - 320x256
        # x0, x1 = 32, input_img.shape[1]-32
        # y0, y1 = 0, input_img.shape[0]
        # input_img = input_img[y0:y1, x0:x1]

        # # # 30 FPS A65SC camera - 640x512
        # # x0, x1 = 64, input_img.shape[1]-64
        # # y0, y1 = 0, input_img.shape[0]
        # # input_img = input_img[y0:y1, x0:x1]
        # # input_img = resize(input_img, (256, 256))
        # # # print("New Image Shape:", input_img.shape)

        # # # 30 FPS A65SC camera - 640x512
        # # x0, x1 = 192, input_img.shape[1]-192
        # # y0, y1 = 128, input_img.shape[0]-128
        # # input_img = input_img[y0:y1, x0:x1]
        # # # input_img = resize(input_img, (256, 256))
        # # input_img = np.fliplr(input_img)
        # # # print("New Image Shape:", input_img.shape)

        input_img_org = copy.deepcopy(input_img)

        t1 = time.time()
        with torch.no_grad():
            input_img = self.img_transform(input_img)
            input_img = input_img.unsqueeze(0)
            # input_img = DataContainer(input_img, stack=self.is_stack)
            # input_img = input_img.to(device=self.device, dtype=torch.float64)
            input_img = self.module_runner.to_device(input_img)

            logits = self.seg_net.forward(input_img)
            torch.cuda.synchronize()
            # pred_mask = torch.argmax(activation(logits).exp(), dim=1).squeeze().cpu().numpy()
            
            logits = logits.permute(0, 2, 3, 1).cpu().numpy().squeeze()
            pred_mask = np.argmax(softmax(logits, axis=-1), axis=-1)
            time_taken = time.time() - t1

        return input_img_org, pred_mask, time_taken

