#! /usr/bin/env python


import numpy as np


import os
# This guide can only be run with the torch backend.
os.environ["KERAS_BACKEND"] = "torch"

import torch
import torch.nn as nn


from HypercomplexKeras import Algebra
#import Algebra





class HyperDenseTorch(nn.Module):
    """Dense layer when the weight are calculated based on arbitrary hyperalgebra.
    The usage is similar to standard hyperdense layer. The input of data is 
    batch x (dim_algebra x n), where n is arbitrary nonzero integer, i.e. the last 
    axis has dimension that is an integer multiple of algebra dimension.
    
    The class is a generalization of class described in:
    G. Vieira and M. Eduardo Valle, "Acute Lymphoblastic Leukemia Detection Using Hypercomplex-Valued Convolutional Neural Networks," 2022 International Joint Conference on Neural Networks (IJCNN), Padua, Italy, 2022, pp. 1-8, doi: 10.1109/IJCNN55064.2022.9892036.
    The calulations are adjusted to TensorFlow tensor operations, and are enhanced to 
    arbitrary algebras.
    """
    def __init__(self, units, input_shape,
                 activation=None,
                 use_bias=True,
                 kernel_initializer=nn.init.xavier_uniform_,
                 bias_initializer=nn.init.zeros_,
                 algebra = Algebra.Quaternions
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
            The algebra to use. The default is Algebra.Quaternions.

        Returns
        -------
        None.

        """
        
        super(HyperDenseTorch, self).__init__()
        
        self.units = int(units) if not isinstance(units, int) else units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.A = torch.from_numpy(algebra.getA()).to(torch.float)
        self.algebra_dim = algebra.getDim()
        self.input_shape = input_shape
        
        
        assert input_shape[-1] % self.algebra_dim == 0
        input_dim = input_shape[-1] // self.algebra_dim
        
        
        self.kernel = nn.Parameter(torch.zeros(self.algebra_dim, input_dim, self.units).to(torch.float))
        self.kernel_initializer(self.kernel)
        
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(self.algebra_dim*self.units).to(torch.float))
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
        W = torch.permute(W, (0,2,1,3))
        
        ss = W.shape
        W  = torch.reshape(W, [ss[0]*ss[1],ss[2]*ss[3]]).to(torch.float)
        outputs = torch.matmul(inputs, W)
        
        if self.use_bias:
            outputs = torch.add(outputs, self.bias)
            
        if self.activation is not None:
            outputs = self.activation(outputs)    
    
        return outputs
    
    
    
def test_output_size():
    HD = HyperDenseTorch(1, (4,))
    y = HD.forward(torch.Tensor(np.array([[1,0,0,0], [1,0,0,0]])))
    assert y.shape == (2,4)
    

def test_simple_learning_example():
    from collections import OrderedDict

    #data:
    x_train = torch.Tensor(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],  [0, 0, 0, 1]], dtype = np.dtype(float))).to(torch.float)
    y_train = torch.Tensor(np.array([[0], [1], [1],  [0]])[:,0]).to(torch.float)


    #model
    model = nn.Sequential(OrderedDict([
        ("HyperDense", HyperDenseTorch(10, (4,), activation = torch.tanh )),
        ("Dense", nn.Linear(40,1)),
        ('Sigmoid', nn.Sigmoid())
            ]))

    #init
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.015)
    torch.manual_seed(1)

    num_epoch = 200

    #training loop
    for epoch in range(num_epoch):
        pred = model(x_train)[:,0]
        loss = loss_fn(pred, y_train)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    
    #prediction
    pred = model(x_train)[:,0]
    assert (pred.round() == y_train).all()


    
    
    
if __name__ == "__main__":
    test_output_size()
    test_simple_learning_example()
