import time
import json
import copy
import torch
import torch.nn as nn
from skimage.transform import resize
import numpy as np

from seg.utils.module_runner import ModuleRunner
from seg.models.model_manager import ModelManager
import seg.utils.transforms as trans

# activation = nn.LogSoftmax(dim=1)

def softmax(X, axis=0):
    max_prob = np.max(X, axis=axis, keepdims=True)
    X -= max_prob
    X = np.exp(X)
    sum_prob = np.sum(X, axis=axis, keepdims=True)
    X /= sum_prob
    return X

class ThermSeg():
    def __init__(self, configer):

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

        input_img_org = copy.deepcopy(input_img)

        t1 = time.time()
        with torch.no_grad():
            input_img = self.img_transform(input_img)
            input_img = input_img.unsqueeze(0)
            input_img = self.module_runner.to_device(input_img)

            logits = self.seg_net.forward(input_img)
            torch.cuda.synchronize()
            # pred_mask = torch.argmax(activation(logits).exp(), dim=1).squeeze().cpu().numpy()
            
            logits = logits.permute(0, 2, 3, 1).cpu().numpy().squeeze()
            pred_mask = np.argmax(softmax(logits, axis=-1), axis=-1)
            time_taken = time.time() - t1

        return input_img_org, pred_mask, time_taken

