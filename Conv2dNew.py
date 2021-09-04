import math
import numpy as np
import copy
import torch
from statistics import mean
def determine_padding(filter_shape, output_shape="same"):
    '''
    :param filter_shape:
    :param output_shape:
    :return:
    '''
    # No padding
    if output_shape == "valid":
        return (0, 0), (0, 0)
    # Pad so that the output shape is the same as input shape (given that stride=1)
    elif output_shape == "same":
        filter_height, filter_width = filter_shape

        # Derived from:
        # output_height = (height + pad_h - filter_height) / stride + 1
        # In this case output_height = height and stride = 1. This gives the
        # expression for the padding below.
        pad_h1 = int(math.floor((filter_height - 1)/2))
        pad_h2 = int(math.ceil((filter_height - 1)/2))
        pad_w1 = int(math.floor((filter_width - 1)/2))
        pad_w2 = int(math.ceil((filter_width - 1)/2))
    else:
        pad_h1 = output_shape[0]
        pad_h2 = output_shape[0]
        pad_w1 = output_shape[1]
        pad_w2 = output_shape[1]

        return (pad_h1, pad_h2), (pad_w1, pad_w2)

def image_to_column(images, filter_shape, stride, output_shape='same'):
    filter_height, filter_width = filter_shape
    pad_h, pad_w = determine_padding(filter_shape, output_shape)# Add padding to the image
    images_padded = torch.nn.functional.pad(images, [pad_w[0],pad_w[1],pad_h[0],pad_h[1]], mode='constant')# Calculate the indices where the dot products are to be applied between weights
    # and the image
    k, i, j = get_im2col_indices(images.shape, filter_shape, (pad_h, pad_w), stride)

    # Get content from image at those indices
    cols = images_padded[:, k, i, j]
    channels = images.shape[1]
    # Reshape content into column shape
    # cols = cols.transpose(1, 2, 0).reshape(filter_height * filter_width * channels, -1)
    cols = cols.permute(1, 2, 0).reshape(filter_height * filter_width * channels, -1)

    return cols

def get_im2col_indices(images_shape, filter_shape, padding, stride=(1,1)):  # stride:(H,W)
    # First figure out what the size of the output should be
    batch_size, channels, height, width = images_shape
    filter_height, filter_width = filter_shape
    pad_h, pad_w = padding
    out_height = int((height + np.sum(pad_h) - filter_height) / stride[0] + 1)
    out_width = int((width + np.sum(pad_w) - filter_width) / stride[1] + 1)

    i0 = np.repeat(np.arange(filter_height), filter_width)
    i0 = np.tile(i0, channels)
    i1 = stride[0] * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(filter_width), filter_height * channels)
    j1 = stride[1] * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(channels), filter_height * filter_width).reshape(-1, 1)
    return (k, i, j)

class Layer(object):

    def set_input_shape(self, shape):
        """ Sets the shape that the layer expects of the input in the forward
        pass method """
        self.input_shape = shape

    def layer_name(self):
        """ The name of the layer. Used in model summary. """
        return self.__class__.__name__

    def parameters(self):
        """ The number of trainable parameters used by the layer """
        return 0

    def forward_pass(self, X, training):
        """ Propogates the signal forward in the network """
        raise NotImplementedError()

    def backward_pass(self, accum_grad):
        """ Propogates the accumulated gradient backwards in the network.
        If the has trainable weights then these weights are also tuned in this method.
        As input (accum_grad) it receives the gradient with respect to the output of the layer and
        returns the gradient with respect to the output of the previous layer. """
        raise NotImplementedError()

    def output_shape(self):
        """ The shape of the output produced by forward_pass """
        raise NotImplementedError()

class Execution(Layer):
    """A 2D Convolution Layer.
    Parameters:
    -----------
    n_filters: int
        The number of filters that will convolve over the input matrix. The number of channels
        of the output shape.
    filter_shape: tuple
        A tuple (filter_height, filter_width).
    input_shape: tuple
        The shape of the expected input of the layer. (batch_size, channels, height, width)
        Only needs to be specified for first layer in the network.
    padding: string
        Either 'same' or 'valid'. 'same' results in padding being added so that the output height and width
        matches the input height and width. For 'valid' no padding is added.
    stride: int
        The stride length of the filters during the convolution over the input.
    """
    def __init__(self,ratio):
        self.ratio = ratio
        pass

    def conv2d(self, input,weight,bias,stride,padding):
        self.input = input
        self.weight = weight
        self.bias = bias
        self.stride = stride
        self.padding = padding

        self.n_filters = self.weight.shape[0]  # 卷积核的个数
        self.filter_shape = (self.weight.shape[2], self.weight.shape[3])
        self.input_shape = [self.input.shape[1],self.input.shape[2],self.input.shape[3]]
        self.trainable = False

        batch_size, channels, height, width = self.input.shape
        # Turn image shape into column shape
        # (enables dot product between input and weights)
        self.X_col = image_to_column(self.input, self.filter_shape, stride=self.stride, output_shape=self.padding)
        # Turn weights into column shape
        if self.ratio != 0:
            # compareRatio = math.ceil(self.ratio * self.X_col.shape[1])
            self.X_col = self.activationSlidePrune(self.X_col,self.ratio)
        self.W_col = self.weight.reshape((self.n_filters, -1))
        # Calculate output
        if self.bias is not None:
            output = torch.einsum('ij,jk->ik',self.W_col,self.X_col) + (torch.unsqueeze(self.bias,1) )
        else:
            output = torch.einsum('ij,jk->ik', self.W_col, self.X_col)
        # Reshape into (n_filters, out_height, out_width, batch_size)
        output = output.reshape(self.output_shape() + (batch_size, ))
        # Redistribute axises so that batch size comes first
        return output.permute(3,0,1,2)


    def output_shape(self):
        channels, height, width = self.input_shape
        pad_h, pad_w = determine_padding(self.filter_shape, output_shape=self.padding)
        output_height = (height + np.sum(pad_h) - self.filter_shape[0]) / self.stride[0] + 1
        output_width = (width + np.sum(pad_w) - self.filter_shape[1]) / self.stride[0] + 1
        return self.n_filters, int(output_height), int(output_width)

    def parameters(self):
        return np.prod(self.W.shape) + np.prod(self.w0.shape)

    def compressionRateStatistics(self,input,andSum,compareRatio):
        pruneNumber = 0
        zerosNumber = 0
        for i in range(input.shape[1]):
            if andSum[i] == 0:
                zerosNumber += 1
            if andSum[i] != 0 and andSum[i] <= compareRatio:
                pruneNumber += 1
        print('pruneNumberRatio=', pruneNumber / (input.shape[1]))
        print('zerosNumberRatio=', zerosNumber / (input.shape[1]))

    def accuracyTest(self,andSum):
        for i in range(len(andSum)):
            print(i,andSum[i])

    def activationSlidePrune(self,input,ratio):

        matrixOne = torch.ones(input.shape,device='cuda:0')  # 设置一个全1矩阵

        x = torch.clone(torch.detach(input))
        andOp = torch.logical_and(matrixOne,x)  # 进行与操作
        andSum = torch.sum(andOp,dim=1)  # 每行的数据进行一个相加

        # self.compressionRateStatistics(input,andSum,compareRatio)
        # self.accuracyTest(andSum)
        p = (sum(andSum) // len(andSum))*ratio
        # zeroTensor = torch.zeros_like(andSum)
        # zeroTensor[(andSum<=p),] = 1
        # pruneRatio = sum(zeroTensor)
        x[(andSum<=p),] = 0

        return x

# image = np.random.randint(0,255,size=(1,3,32,32)).astype(np.uint8)
# input_shape=image.squeeze().shape
# conv2d = Conv2D(16, (3,3), input_shape=input_shape, padding='same', stride=1)
# conv2d.initialize(None)
# output=conv2d.forward_pass(image,training=True)
# print(output.shape)