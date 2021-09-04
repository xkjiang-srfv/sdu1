from ActivationPrune import *
from WeightPrune import weightPruneModelOp
import os
from Op import Op
import torch
torch.set_printoptions(threshold=5e3,edgeitems=15)

if __name__ == '__main__':
    model_name = 'AlexNet'  # 确定模型名称
    batch_size = 20  # 确定批训练图片数目
    img_size = 227  # 确定单张图片大小
    ratio = 0.95  # 确定输入特征图剪枝比率
    epochA = 40  # 确定针对输入特征图剪枝重训练轮数或原始模型（不掺杂任何剪枝训练）轮数
    epochAW = 40  # 确定针对卷积核聚类剪枝重训练轮数
    weightParameter = (4/1)
    LinearParameter = (4/2)
    '''
    一共设置有七种针对模型的操作
    1. operation = 'trainInitialModel'，意为训练初始模型，此时不参杂任何剪枝操作，单纯训练初始模型
    2. operation = 'onlyActivationPruneWithRetrain'，意为只针对输入特征图进行剪枝，并进行重训练
    3. operation = 'onlyWeightPruneWithRetrain'，意为只针对权重值进行聚类剪枝，并进行重训练
    4. operation = 'weightRetrainAfterActivationPrune'，意为此时我已经单独完成了输入特征图剪枝的行为，保存了模型，此时我想再进行权重聚类剪枝
    5. operation = 'activationWeightPruneWithRetrain'，意为对输入特征图剪枝并进行重训练，对其生成的模型权重进行聚类剪枝并进行重训练
    6. operation = 'onlyActivationPruneTest'，意为只针对输入特征图剪枝后的模型进行inferernce，测试模型精度
    7. operation = 'activationWeightPruneTest'，意为针对输入特征图与权重聚类剪枝后的模型进行inference，测试模型精度
    '''
    operation = 'weightRetrainAfterActivationPrune'
    Op(operation,model_name,batch_size,img_size,ratio,epochA,epochAW,weightParameter,LinearParameter)
    '''
    目录说明
    -pth
        --modelName
            ---ratio=0
                ----Activation:存放不经过任何剪枝的初始模型
                ----Weight:存放只经过权重聚类剪枝后的初始模型
            ---ratio=0.1
                ----Activation:存放经过输入特征图剪枝后的模型
                ----ActivationWeight:存放经过输入特征图剪枝后又进行权重聚类剪枝后的模型
    '''