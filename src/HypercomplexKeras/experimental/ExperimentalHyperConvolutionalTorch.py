#! /usr/bin/env python


import numpy as np


import os
# This guide can only be run with the torch backend.
os.environ["KERAS_BACKEND"] = "torch"

import torch
import torch.nn as nn



import Algebra




class HyperConv2DTorchTF(nn.Module):
    """Dense layer when the weight are calculated based on arbitrary hyperalgebra.
    The usage is similar to standard hyperdense layer. The input of data is 
    batch x (dim_algebra x n), where n is arbitrary nonzero integer, i.e. the last 
    axis has dimension that is an integer multiple of algebra dimension.
    
    WARNING: This class realizes Hypercomplex convolutional layer with the ordering of TensorFlow data alignment.
    
    The class is a generalization of class described in:
    G. Vieira and M. Eduardo Valle, "Acute Lymphoblastic Leukemia Detection Using Hypercomplex-Valued Convolutional Neural Networks," 2022 International Joint Conference on Neural Networks (IJCNN), Padua, Italy, 2022, pp. 1-8, doi: 10.1109/IJCNN55064.2022.9892036.
    The calulations are adjusted to TensorFlow tensor operations, and are enhanced to 
    arbitrary algebras.
    
    
    
    """
    def __init__(self,
                 input_shape,
                 filters,
                 kernel_size,
                 strides=1,
                 padding=0,
                 use_bias=True,
                 activation=None,
                 initializer=nn.init.xavier_uniform_,
                 bias_initializer=nn.init.zeros_,
                 data_format=None,
                 algebra = Algebra.Quaternions,
                ):
        """
        Initializer of Module for hyperdense hypercomplex layer.

        Parameters
        ----------
        units : int
            Number of units/neurons in the layer.
        input_shape : tuple
            The shape of input data
        activation : torch.activation, optional
            Activation function e.g., torch.tanh. The default is None.
        use_bias : bool, optional
            If we use the bias. The default is True.
        kernel_initializer : nn.init.initializer, optional
            Kernel initialization distirbution. The default is nn.init.xavier_uniform_.
        bias_initializer : nn.init.initializer, optional
            Initialization of the bias. The default is nn.init.zeros_.
        algebra : algebra, optional
            The algebra to use. The default is algebra.Quaternions.

        Returns
        -------
        None.

        """
        
        super(HyperConv2DTorchTF, self).__init__()
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.bias_initializer = bias_initializer
        self.initializer = initializer
        self.activation = activation
        self.A = torch.from_numpy(algebra.getA()).to(torch.float)
        self.algebra_dim = algebra.getDim()
        self.input_shape = input_shape
        
        input_channel = input_shape[-1]
        
        
        assert input_shape[-1] % self.algebra_dim == 0        
            
        input_dim = input_channel // self.algebra_dim
        kernel_shape = self.kernel_size + (input_dim, self.filters)
        
        
        self.kernel = nn.Parameter(torch.zeros(self.algebra_dim, *kernel_shape).to(torch.float))
        self.initializer(self.kernel)
        
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(self.algebra_dim*self.filters).to(torch.float))
            self.bias_initializer(self.bias)
        else:
            self.bias = None
            

    def forward(self, inputs):
        """
        

        Parameters
        ----------
        inputs : torch.float tensor
            Input data of size batch x (n x imput_shape), where n is a positive integer.

        Returns
        -------
        outputs : torch.float tensor
            Output vecotr of size batch units x n x input_shape.

        """
                                      
        inputs = inputs.to(torch.float)
        
        W = torch.tensordot(self.A, self.kernel, dims=[[1],[0]])
        W = torch.permute(W, (1,2,3,0,4,5))
        
        ss = W.shape
        W  = torch.reshape(W, [ss[0],ss[1],ss[2],ss[3]*ss[4],ss[5]]).to(torch.float)
        
        
        W = torch.permute(W, (0, -1, -2, 1, 2))
        inputs = torch.permute(inputs, (0,-1, 1, 2))
        
        outputs = []
        for i in range(self.algebra_dim):
            outputs.append(torch.nn.functional.conv2d(inputs, W[i,:], stride=self.strides, padding=self.padding))
        outputs = torch.concat(outputs,dim=1).to(torch.float)
        
        outputs = torch.permute(outputs, (0,2,3,1))
        
        
        if self.use_bias:
            outputs = outputs + self.bias
            
        if self.activation is not None:
            outputs = self.activation(outputs)    
    
        return outputs



class HyperConv2DTorch(nn.Module):
    """Dense layer when the weight are calculated based on arbitrary hyperalgebra.
    The usage is similar to standard hyperdense layer. The input of data is 
    batch x (dim_algebra x n), where n is arbitrary nonzero integer, i.e. the last 
    axis has dimension that is an integer multiple of algebra dimension.
    
    The class is a generalization of class described in:
    G. Vieira and M. Eduardo Valle, "Acute Lymphoblastic Leukemia Detection Using Hypercomplex-Valued Convolutional Neural Networks," 2022 International Joint Conference on Neural Networks (IJCNN), Padua, Italy, 2022, pp. 1-8, doi: 10.1109/IJCNN55064.2022.9892036.
    The calulations are adjusted to TensorFlow tensor operations, and are enhanced to 
    arbitrary algebras.
    """
    def __init__(self,
                 input_shape,
                 filters,
                 kernel_size,
                 strides=1,
                 padding=0,
                 use_bias=True,
                 activation=None,
                 initializer=nn.init.xavier_uniform_,
                 bias_initializer=nn.init.zeros_,
                 data_format=None,
                 algebra = Algebra.Quaternions,
                ):
        """
        Initializer of Module for hyperdense hypercomplex layer.

        Parameters
        ----------
        units : int
            Number of units/neurons in the layer.
        input_shape : tuple
            The shape of input data
        activation : torch.activation, optional
            Activation function e.g., torch.tanh. The default is None.
        use_bias : bool, optional
            If we use the bias. The default is True.
        kernel_initializer : nn.init.initializer, optional
            Kernel initialization distirbution. The default is nn.init.xavier_uniform_.
        bias_initializer : nn.init.initializer, optional
            Initialization of the bias. The default is nn.init.zeros_.
        algebra : algebra, optional
            The algebra to use. The default is algebra.Quaternions.

        Returns
        -------
        None.

        """
        
        super(HyperConv2DTorch, self).__init__()
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.bias_initializer = bias_initializer
        self.initializer = initializer
        self.activation = activation
        self.A = torch.from_numpy(algebra.getA()).to(torch.float)
        self.algebra_dim = algebra.getDim()
        self.input_shape = input_shape
        
        input_channel = input_shape[-1]
        
        
        assert input_shape[-1] % self.algebra_dim == 0        
            
        input_dim = input_channel // self.algebra_dim
        kernel_shape = self.kernel_size + (input_dim, self.filters)
    
        
        self.kernel = nn.Parameter(torch.zeros(self.algebra_dim, *kernel_shape).to(torch.float))
        self.initializer(self.kernel)
        
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(self.algebra_dim*self.filters).to(torch.float))
            self.bias_initializer(self.bias)
        else:
            self.bias = None
            

    def forward(self, inputs):
        """
        

        Parameters
        ----------
        inputs : torch.float tensor
            Input data of size batch x (n x imput_shape), where n is a positive integer.

        Returns
        -------
        outputs : torch.float tensor
            Output vecotr of size batch units x n x input_shape.

        """
                                      
        inputs = inputs.to(torch.float)
        
        W = torch.tensordot(self.A, self.kernel, dims=[[1],[0]])
        W = torch.permute(W, (1,2,3,0,4,5))
        
        ss = W.shape
        W  = torch.reshape(W, [ss[0],ss[1],ss[2],ss[3]*ss[4],ss[5]]).to(torch.float)
        
        
        
        W = torch.permute(W, (0, -1, -2, 1, 2))
        inputs = torch.permute(inputs, (-1, 0, 1))
        
        
        outputs = []
        for i in range(self.algebra_dim):
            outputs.append(torch.nn.functional.conv2d(inputs, W[i,:], stride=self.strides, padding=self.padding))
        outputs = torch.concat(outputs,dim=0).to(torch.float)
        
        outputs = torch.permute(outputs, (1,2,0))
        
        
        if self.use_bias:
            outputs = outputs + self.bias
            
        if self.activation is not None:
            outputs = self.activation(outputs)    
    
        return outputs



def test_OutputSizeConv2dTF():
    import numpy as np
    import keras
    import torch
    import torch.nn as nn
    from torch.autograd import Variable
    from collections import OrderedDict
    import matplotlib.pylab as plt

    model = HyperConv2DTorchTF(input_shape=(None, 10,5,4), filters=3, kernel_size=(2,2))
    inp =  torch.Tensor(np.random.rand(20,10,5,4)).to(torch.float)
    y = model.forward(inp)
    assert y.shape == (20, 9, 4, 12)


def test_OutputSizeConv2d():
    import numpy as np
    import keras
    import torch
    import torch.nn as nn
    from torch.autograd import Variable
    from collections import OrderedDict
    import matplotlib.pylab as plt

    model = HyperConv2DTorch(input_shape=(10,5,4), filters=3, kernel_size=(2,2))

    inp =  torch.Tensor(np.random.rand(20,10,5,4)).to(torch.float)

    for x in inp:
        pred = model.forward(x)
        assert pred.shape == (9, 4, 12)





if __name__=="__main__":
    test_OutputSizeConv2dTF()
    test_OutputSizeConv2d()
    






