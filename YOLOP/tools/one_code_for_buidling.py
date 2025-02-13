

import math


import torch
import torch.nn as nn

from torch.nn import Upsample
import torch.backends.cudnn as cudnn
import torch.optim

from lib.models.common import Conv, SPP, BottleneckCSP, Focus, Concat, Detect, SharpenConv



# The lane line and the driving area segment branches without share information with each other and without link
YOLOP = [
[24, 33, 42],   #Det_out_idx, Da_Segout_idx, LL_Segout_idx
[ -1, Focus, [3, 32, 3]],   #0
[ -1, Conv, [32, 64, 3, 2]],    #1
[ -1, BottleneckCSP, [64, 64, 1]],  #2
[ -1, Conv, [64, 128, 3, 2]],   #3
[ -1, BottleneckCSP, [128, 128, 3]],    #4
[ -1, Conv, [128, 256, 3, 2]],  #5
[ -1, BottleneckCSP, [256, 256, 3]],    #6
[ -1, Conv, [256, 512, 3, 2]],  #7
[ -1, SPP, [512, 512, [5, 9, 13]]],     #8
[ -1, BottleneckCSP, [512, 512, 1, False]],     #9
[ -1, Conv,[512, 256, 1, 1]],   #10
[ -1, Upsample, [None, 2, 'nearest']],  #11
[ [-1, 6], Concat, [1]],    #12
[ -1, BottleneckCSP, [512, 256, 1, False]], #13
[ -1, Conv, [256, 128, 1, 1]],  #14
[ -1, Upsample, [None, 2, 'nearest']],  #15
[ [-1,4], Concat, [1]],     #16         #Encoder

[ -1, BottleneckCSP, [256, 128, 1, False]],     #17
[ -1, Conv, [128, 128, 3, 2]],      #18
[ [-1, 14], Concat, [1]],       #19
[ -1, BottleneckCSP, [256, 256, 1, False]],     #20
[ -1, Conv, [256, 256, 3, 2]],      #21
[ [-1, 10], Concat, [1]],   #22
[ -1, BottleneckCSP, [512, 512, 1, False]],     #23
[ [17, 20, 23], Detect,  [1, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]], #Detection head 24

[ 16, Conv, [256, 128, 3, 1]],   #25
[ -1, Upsample, [None, 2, 'nearest']],  #26
[ -1, BottleneckCSP, [128, 64, 1, False]],  #27
[ -1, Conv, [64, 32, 3, 1]],    #28
[ -1, Upsample, [None, 2, 'nearest']],  #29
[ -1, Conv, [32, 16, 3, 1]],    #30
[ -1, BottleneckCSP, [16, 8, 1, False]],    #31
[ -1, Upsample, [None, 2, 'nearest']],  #32
[ -1, Conv, [8, 2, 3, 1]], #33 Driving area segmentation head

[ 16, Conv, [256, 128, 3, 1]],   #34
[ -1, Upsample, [None, 2, 'nearest']],  #35
[ -1, BottleneckCSP, [128, 64, 1, False]],  #36
[ -1, Conv, [64, 32, 3, 1]],    #37
[ -1, Upsample, [None, 2, 'nearest']],  #38
[ -1, Conv, [32, 16, 3, 1]],    #39
[ -1, BottleneckCSP, [16, 8, 1, False]],    #40
[ -1, Upsample, [None, 2, 'nearest']],  #41
[ -1, Conv, [8, 2, 3, 1]] #42 Lane line segmentation head
]


def check_anchor_order(m):
    # Check anchor order against stride order for YOLOv5 Detect() module m, and correct if necessary
    a = m.anchor_grid.prod(-1).view(-1)  # anchor area
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da.sign() != ds.sign():  # same order
        print('Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)
        m.anchor_grid[:] = m.anchor_grid.flip(0)

def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
        # elif t in [nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True

class MCnet(nn.Module):
    def __init__(self, block_cfg, **kwargs):
        super(MCnet, self).__init__()
        layers, save = [], []
        self.nc = 1
        self.detector_index = -1
        self.det_out_idx = block_cfg[0][0]
        self.seg_out_idx = block_cfg[0][1:]

        # Build model
        for i, (from_, block, args) in enumerate(block_cfg[1:]):
            block = eval(block) if isinstance(block, str) else block  # eval strings
            if block is Detect:
                self.detector_index = i
            block_ = block(*args)
            block_.index, block_.from_ = i, from_
            layers.append(block_)
            save.extend(x % i for x in ([from_] if isinstance(from_, int) else from_) if x != -1)  # append to savelist
        assert self.detector_index == block_cfg[0][0]

        self.model, self.save = nn.Sequential(*layers), sorted(save)
        self.names = [str(i) for i in range(self.nc)]

        # set stride、anchor for detector
        Detector = self.model[self.detector_index]  # detector
        if isinstance(Detector, Detect):
            s = 128  # 2x min stride
            # for x in self.forward(torch.zeros(1, 3, s, s)):
            #     print (x.shape)
            with torch.no_grad():
                model_out = self.forward(torch.zeros(1, 3, s, s))
                detects, _, _ = model_out
                Detector.stride = torch.tensor([s / x.shape[-2] for x in detects])  # forward
            # print("stride"+str(Detector.stride ))
            Detector.anchors /= Detector.stride.view(-1, 1, 1)  # Set the anchors for the corresponding scale
            check_anchor_order(Detector)
            self.stride = Detector.stride
            self._initialize_biases()

        initialize_weights(self)

    def forward(self, x):
        cache = []
        out = []
        det_out = None
        Da_fmap = []
        LL_fmap = []
        for i, block in enumerate(self.model):
            if block.from_ != -1:
                x = cache[block.from_] if isinstance(block.from_, int) else [x if j == -1 else cache[j] for j in
                                                                             block.from_]  # calculate concat detect
            x = block(x)
            if i in self.seg_out_idx:  # save driving area segment result
                m = nn.Sigmoid()
                out.append(m(x))
            if i == self.detector_index:
                det_out = x
            cache.append(x if block.index in self.save else None)
        out.insert(0, det_out)
        return out

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        # m = self.model[-1]  # Detect() module
        m = self.model[self.detector_index]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

def get_net(cfg, **kwargs):
    m_block_cfg = YOLOP
    model = MCnet(m_block_cfg, **kwargs)
    return model


def make_cfg():
    from yacs.config import CfgNode as CN
    _C = CN()

    dir_root = 'C:\\Users\\lixingyou\\Documents\\project\\MTL-Perception-Research\\'
    _C.LOG_DIR = f'{dir_root}YOLOP\\runs\\'
    _C.CACHE_DIR = f'{dir_root}YOLOP'
    _C.GPUS = (0,)
    _C.WORKERS = 1
    _C.PIN_MEMORY = False
    _C.PRINT_FREQ = 20
    _C.AUTO_RESUME =False       # Resume from the last training interrupt
    # _C.AUTO_RESUME =False       # Resume from the last training interrupt
    _C.NEED_AUTOANCHOR = False      # Re-select the prior anchor(k-means)    When training from scratch (epoch=0), set it to be ture!
    _C.DEBUG = False
    _C.num_seg_class = 3  # 3 is RGB image for seg training, otherwise is Gray image for seg training

    # Cudnn related params
    _C.CUDNN = CN()
    _C.CUDNN.BENCHMARK = True
    _C.CUDNN.DETERMINISTIC = False
    _C.CUDNN.ENABLED = True


    # common params for NETWORK
    _C.MODEL = CN(new_allowed=True)
    _C.MODEL.NAME = ''
    _C.MODEL.STRU_WITHSHARE = False     #add share_block to segbranch
    _C.MODEL.HEADS_NAME = ['']
    _C.MODEL.PRETRAINED = ""
    _C.MODEL.PRETRAINED_DET = ""
    _C.MODEL.IMAGE_SIZE = [640, 640]  # width * height, ex: 192 * 256
    _C.MODEL.EXTRA = CN(new_allowed=True)


    # loss params
    _C.LOSS = CN(new_allowed=True)
    _C.LOSS.LOSS_NAME = ''
    _C.LOSS.MULTI_HEAD_LAMBDA = None
    _C.LOSS.FL_GAMMA = 0.0  # focal loss gamma
    _C.LOSS.CLS_POS_WEIGHT = 1.0  # classification loss positive weights
    _C.LOSS.OBJ_POS_WEIGHT = 1.0  # object loss positive weights
    _C.LOSS.SEG_POS_WEIGHT = 1.0  # segmentation loss positive weights
    _C.LOSS.BOX_GAIN = 0.05  # box loss gain
    _C.LOSS.CLS_GAIN = 0.5  # classification loss gain
    _C.LOSS.OBJ_GAIN = 1.0  # object loss gain
    _C.LOSS.DA_SEG_GAIN = 0.2  # driving area segmentation loss gain
    _C.LOSS.LL_SEG_GAIN = 0.2  # lane line segmentation loss gain
    _C.LOSS.LL_IOU_GAIN = 0.2 # lane line iou loss gain


    # DATASET related params
    _C.DATASET = CN(new_allowed=True)
    # _C.DATASET.DATAROOT = '/workspace/bdd100k/labels/det_20/sample_dataset/images/'       # the path of images folder
    # _C.DATASET.LABELROOT = '/workspace/bdd100k/labels/det_20/sample_dataset/det'      # the path of det_annotations folder
    # _C.DATASET.MASKROOT = '/workspace/bdd100k/labels/det_20/sample_dataset/drivable'                # the path of da_seg_annotations folder
    # _C.DATASET.LANEROOT = '/workspace/bdd100k/labels/det_20/sample_dataset/lane'               # the path of ll_seg_annotations folder

    data_dir_root = f'{dir_root}bdd100k\\'
    _C.DATASET.DATAROOT = f'{data_dir_root}images'       # the path of images folder
    _C.DATASET.LABELROOT = f'{data_dir_root}\\det'      # the path of det_annotations folder
    _C.DATASET.MASKROOT = f'{data_dir_root}\\drivable'                # the path of da_seg_annotations folder
    _C.DATASET.LANEROOT = f'{data_dir_root}\\lane'               # the path of ll_seg_annotations folder
    _C.DATASET.DATASET = 'BddDataset'
    _C.DATASET.TRAIN_SET = 'train'
    _C.DATASET.TEST_SET = 'val'
    _C.DATASET.DATA_FORMAT = 'jpg'
    _C.DATASET.SELECT_DATA = False
    _C.DATASET.ORG_IMG_SIZE = [720, 1280]

    # training data augmentation
    _C.DATASET.FLIP = True
    _C.DATASET.SCALE_FACTOR = 0.25
    _C.DATASET.ROT_FACTOR = 10
    _C.DATASET.TRANSLATE = 0.1
    _C.DATASET.SHEAR = 0.0
    _C.DATASET.COLOR_RGB = True
    _C.DATASET.HSV_H = 0.015  # image HSV-Hue augmentation (fraction)
    _C.DATASET.HSV_S = 0.7  # image HSV-Saturation augmentation (fraction)
    _C.DATASET.HSV_V = 0.4  # image HSV-Value augmentation (fraction)
    # TODO: more augmet params to add


    # train
    _C.TRAIN = CN(new_allowed=True)
    _C.TRAIN.LR0 = 0.001  # initial learning rate (SGD=1E-2, Adam=1E-3)
    _C.TRAIN.LRF = 0.2  # final OneCycleLR learning rate (lr0 * lrf)
    _C.TRAIN.WARMUP_EPOCHS = 3.0
    _C.TRAIN.WARMUP_BIASE_LR = 0.1
    _C.TRAIN.WARMUP_MOMENTUM = 0.8

    _C.TRAIN.OPTIMIZER = 'adam'
    _C.TRAIN.MOMENTUM = 0.937
    _C.TRAIN.WD = 0.0005
    _C.TRAIN.NESTEROV = True
    _C.TRAIN.GAMMA1 = 0.99
    _C.TRAIN.GAMMA2 = 0.0

    _C.TRAIN.BEGIN_EPOCH = 0
    _C.TRAIN.END_EPOCH = 300

    _C.TRAIN.VAL_FREQ = 1
    _C.TRAIN.BATCH_SIZE_PER_GPU = 24
    _C.TRAIN.SHUFFLE = True

    _C.TRAIN.IOU_THRESHOLD = 0.2
    _C.TRAIN.ANCHOR_THRESHOLD = 4.0

    # if training 3 tasks end-to-end, set all parameters as True
    # Alternating optimization
    _C.TRAIN.SEG_ONLY = False           # Only train two segmentation branchs
    _C.TRAIN.DET_ONLY = False           # Only train detection branch
    _C.TRAIN.ENC_SEG_ONLY = False       # Only train encoder and two segmentation branchs
    _C.TRAIN.ENC_DET_ONLY = False       # Only train encoder and detection branch

    # Single task
    _C.TRAIN.DRIVABLE_ONLY = False      # Only train da_segmentation task
    _C.TRAIN.LANE_ONLY = False          # Only train ll_segmentation task
    _C.TRAIN.DET_ONLY = False          # Only train detection task




    _C.TRAIN.PLOT = True                #

    # testing
    _C.TEST = CN(new_allowed=True)
    _C.TEST.BATCH_SIZE_PER_GPU = 24
    _C.TEST.MODEL_FILE = ''
    _C.TEST.SAVE_JSON = False
    _C.TEST.SAVE_TXT = False
    _C.TEST.PLOTS = True
    _C.TEST.NMS_CONF_THRESHOLD  = 0.001
    _C.TEST.NMS_IOU_THRESHOLD  = 0.6
    return _C

def print_3d_title():
    from colorama import init, Fore
    import pyfiglet

    # 3D Text
    title = pyfiglet.figlet_format("XINGYOU", font='3-d')
    border = "=" * 70
    print(f"{Fore.BLACK}")
    print(border)
    print(title)
    print(border)
    print(f"{Fore.RESET}")


if __name__ == "__main__":
    print_3d_title()
    cfg = make_cfg()

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # Set device
    device_pu = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Set model
    print("load model to device")
    model = get_net(cfg).to(device_pu)

    # Set Loss and Optimizer
    # criterion = get_loss(cfg, device=device)
    # optimizer = get_optimizer(cfg, model)

    print(1)

