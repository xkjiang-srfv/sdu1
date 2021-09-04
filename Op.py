from ActivationPrune import activationPruneModelOp
from WeightPrune import weightPruneModelOp
import os
def makeDir(model_name,ratio):  # 创建文件夹
    if not os.path.exists('./pth/' + model_name + '/ratio=' + str(ratio)):
        os.makedirs('./pth/' + model_name + '/ratio=' + str(ratio) + '/Activation')
        if ratio == 0:  # ratio=0只有两种情况，一是训练初始模型的时候，二是单对权重进行聚类剪枝的时候
            os.makedirs('./pth/' + model_name + '/ratio=0/' + 'Weight')
        else:
            os.makedirs('./pth/' + model_name + '/ratio=' + str(ratio) + '/ActivationWeight')

def Op(operation,model_name,batch_size,img_size,ratio,epochA,epochAW,weightParameter,LinearParameter):
    if operation == 'trainInitialModel':  # 训练初始模型
        patternA = 'train'
        ratio = 0
        makeDir(model_name,ratio)  # 判断是否为训练初始模型并创建相应的文件夹
        activationPruneModelOp(model_name, batch_size, img_size,patternA,ratio,epochA)

    if operation == 'onlyActivationPruneWithRetrain':  # 只进行输入特征图的剪枝，不进行权重的聚类剪枝
        patternA = 'retrain'
        makeDir(model_name,ratio)
        activationPruneModelOp(model_name, batch_size, img_size,patternA,ratio,epochA)

    if operation == 'onlyWeightPruneWithRetrain':   # 这有bug
        patternW = 'train'  # patternW='retrain'是读入初始模型进行权重聚类剪枝压缩再重训练
        ratio = 0
        makeDir(model_name,ratio)
        weightPruneModelOp(model_name, batch_size, img_size, ratio, patternW, epochAW, weightParameter,LinearParameter)

    if operation == 'activationWeightPruneWithRetrain':
        patternA = 'retrain'
        patternW = 'retrain'  # patternW='retrain'是读入已经经过输入特征图剪枝的模型进行压缩在重训练
        makeDir(model_name, ratio)
        activationPruneModelOp(model_name, batch_size, img_size, patternA, ratio, epochA)
        weightPruneModelOp(model_name, batch_size, img_size, ratio, patternW, epochAW, weightParameter, LinearParameter)

    if operation == 'onlyActivationPruneTest':
        patternA = 'test'
        makeDir(model_name, ratio)
        activationPruneModelOp(model_name, batch_size, img_size, patternA, ratio, epochA)

    if operation == 'activationWeightPruneTest':
        patternW = 'test'
        makeDir(model_name, ratio)
        weightPruneModelOp(model_name, batch_size, img_size, ratio, patternW, epochAW, weightParameter, LinearParameter)

    if operation == 'weightRetrainAfterActivationPrune':
        patternW = 'retrain'
        makeDir(model_name, ratio)
        weightPruneModelOp(model_name, batch_size, img_size, ratio, patternW, epochAW, weightParameter, LinearParameter)