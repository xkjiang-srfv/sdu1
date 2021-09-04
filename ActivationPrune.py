import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import time
from model import *
from train import *
import random
# from .model import ResNetBasicBlock

from math import sqrt
import copy
from time import time
from Conv2dNew import Execution



class Conv2dTest(nn.Conv2d):
    def __init__(self,
                 ratio,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 ):
        super(Conv2dTest, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                          bias, padding_mode)
        self.ratio = ratio
    def forward(self, input):
        E = Execution(self.ratio)
        output = E.conv2d(input, self.weight, self.bias, self.stride, self.padding)
        return output

class LinearTest(nn.Linear):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 ):
        super(LinearTest, self).__init__(in_features, out_features, bias)

    def forward(self, input):
        output = F.linear(input, self.weight, self.bias)
        return output

def prepare(model, ratio,inplace=False):
    # move intpo prepare
    def addActivationPruneOp(module):
        nonlocal layer_cnt
        for name, child in module.named_children():
            if isinstance(child, nn.Conv2d):
                p_name = str(layer_cnt)
                activationPruneConv = Conv2dTest(
                    ratio,
                    child.in_channels,
                    child.out_channels, child.kernel_size, stride=child.stride, padding=child.padding,
                    dilation=child.dilation, groups=child.groups, bias=(child.bias is not None),
                    padding_mode=child.padding_mode
                )
                if child.bias is not None:
                    activationPruneConv.bias = child.bias
                activationPruneConv.weight = child.weight
                module._modules[name] = activationPruneConv
                layer_cnt += 1
            elif isinstance(child, nn.Linear):
                p_name = str(layer_cnt)
                activationPruneLinear = LinearTest(
                     child.in_features, child.out_features,
                    bias=(child.bias is not None)
                )
                if child.bias is not None:
                    activationPruneLinear.bias = child.bias
                activationPruneLinear.weight = child.weight
                module._modules[name] = activationPruneLinear
                layer_cnt += 1
            else:
                addActivationPruneOp(child)  # 这是用来迭代的，Maxpool层的功能是不变的
    layer_cnt = 0
    if not inplace:
        model = copy.deepcopy(model)
    addActivationPruneOp( model)  # 为每一个卷积层添加输入特征图剪枝操作
    return model

def getModel(modelName):
    if modelName == 'LeNet':
        return getLeNet()  # 加载原始模型框架
    elif modelName == 'AlexNet':
        return getAlexnet()
    elif modelName == 'VGG16':
        return get_vgg16()
    elif modelName == 'SqueezeNet':
        return get_squeezenet()
    elif modelName == 'ResNet':
        return get_resnet18()
    elif modelName == 'InceptionV3':
        return get_inception_v3()
    # if modelName == 'MobileNet':
    #     return mobilenetv3_large()

def getDataSet(modelName,batchSize,imgSize):
    if modelName == 'VGG16' or modelName == 'AlexNet' or modelName == 'ResNet'  or modelName == 'SqueezeNet' or modelName=='InceptionV3':
        dataloaders, dataset_sizes = load_cifar10(batch_size=batchSize, pth_path='./data',
                                                  img_size=imgSize)  # 确定数据集
    elif modelName == 'LeNet':
        dataloaders, dataset_sizes = load_mnist(batch_size=batchSize, path='./data', img_size=imgSize)

    return dataloaders,dataset_sizes

def getPruneModel(model_name, weight_file_path,pattern,ratio):
    model_orign = getModel(model_name)
    if pattern == 'test' or pattern == 'retrain':
        model_orign.load_state_dict(torch.load(weight_file_path))  # 原始模型框架加载模型信息
    activationPruneModel = prepare(model_orign,ratio)

    return activationPruneModel

def activationPruneModelOp(model_name, batch_size, img_size,pattern,ratio,epoch):
    dataloaders, dataset_sizes = getDataSet(model_name, batch_size, img_size)
    criterion = nn.CrossEntropyLoss()

    if pattern == 'retrain' or pattern == 'train':
        weight_file_path = './pth/' + model_name + '/ratio=0'+ '/Activation' + '/best.pth'
        activationPruneModel = getPruneModel(model_name, weight_file_path, pattern, ratio)
        optimizer = optim.SGD(activationPruneModel.parameters(), lr=0.01, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)  # 设置学习率下降策略
        train_model_jiang(activationPruneModel, dataloaders, dataset_sizes, ratio, 'activation',pattern, criterion=criterion,optimizer=optimizer, name=model_name,
                          scheduler=scheduler, num_epochs=epoch, rerun=False)  # 进行模型的训练
    if pattern == 'test':
        weight_file_path = './pth/' + model_name + '/ratio=' + str(ratio) + '/Activation/' + 'best.pth'
        activationPruneModel = getPruneModel(model_name, weight_file_path, pattern, ratio)
        test_model(activationPruneModel, dataloaders, dataset_sizes, criterion=criterion)


