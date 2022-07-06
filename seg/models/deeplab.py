import torch
import torch.nn as nn
import torch.nn.functional as F
from seg.models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from seg.models.aspp import build_aspp
from seg.models.decoder import build_decoder
from seg.models.backbone import build_backbone

class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', in_channels=3, output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False, pretrained=True, seg_net_out_opt=1):
        super(DeepLab, self).__init__()
        self.n_channels = in_channels
        self.n_classes = num_classes
        self.n_features = 256+48    #304
        self.seg_net_out_opt = seg_net_out_opt
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, in_channels, output_stride, BatchNorm, pretrained=pretrained)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm, out_opt=self.seg_net_out_opt)
        self.freeze_bn = freeze_bn

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)

        if self.seg_net_out_opt == 1:
            x = self.decoder(x, low_level_feat)
        else:
            x, feat_map = self.decoder(x, low_level_feat)

        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        if self.seg_net_out_opt != 1:
            feat_map = F.interpolate(feat_map, size=input.size()[2:], mode='bilinear', align_corners=True)

        if self.seg_net_out_opt == 1:
            return x
        else:
            return x, feat_map

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

if __name__ == "__main__":
    model = DeepLab(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())


