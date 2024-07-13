#! /usr/bin/env python

import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, activations, initializers
from tensorflow.python.framework import tensor_shape

from HypercomplexKeras import Algebra
#import Algebra



class HyperConv2D(layers.Layer):
    """Convolutional 2D layer with computations based on arbitrary algebra.
    The input data is: batch x dim_x x dim_y x (dim_algebra x n), where n is arbitrary nonzero integer, i.e. the last 
    axis has dimension that is an integer multiple of algebra dimension.
    
    The class is a generalization of class HyperConv2D described in:
    G. Vieira and M. Eduardo Valle, "Acute Lymphoblastic Leukemia Detection Using Hypercomplex-Valued Convolutional Neural Networks," 2022 International Joint Conference on Neural Networks (IJCNN), Padua, Italy, 2022, pp. 1-8, doi: 10.1109/IJCNN55064.2022.9892036.
    The calulations are adjusted to TensorFlow tensor operations, and are enhanced to 
    arbitrary algebras.
    """
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='VALID',
                 use_bias=True,
                 activation=None,
                 initializer=tf.keras.initializers.GlorotNormal(seed=0),
                 bias_initializer='zeros',
                 data_format=None,
                 algebra = Algebra.Quaternions,
                ):
        """
        Parameters
        ----------
        filters : int
            the dimension of the output space (the number of filters in the convolution)
        kernel_size : int or tuple/list of 2 integer
            specifying the size of the convolution window.
        strides : int or tuple/list of 2 integer, optional
            DESCRIPTION. specifying the stride length of the convolution. The default is 1.
        padding : string, either "valid" or "same" (case-insensitive), optional
            DESCRIPTION. The default is 'VALID'.
        use_bias : bool, optional
            if True, bias will be added to the output. The default is True.
        activation : Activation function, optional
            If None, no activation is applied. The default is None.
        initializer : Kernel initializer type, optional
            Kernel initializer type. The default is tf.keras.initializers.GlorotNormal(seed=0).
        bias_initializer : Initializer for the bias vector, optional
            If None, the default initializer ("zeros") will be used. The default is 'zeros'.
        data_format : string, optional
            HyperConv2d is designed only for channels_last. The default is None.
        algebra : StructureConstants type, optional
            The StructureConstants object that describes multiplication table of the algebra. The default is Algebra.Quaternions.

        Returns
        -------
        None.

        """
        super(HyperConv2D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.bias_initializer = bias_initializer
        self.activation = activations.get(activation)
        self.initializer = initializers.get(initializer)
        self.data_format = data_format
        self.A = tf.convert_to_tensor(algebra.getA(), dtype = np.float32)
        self.algebra_dim = algebra.getDim()

    def _get_channel_axis(self):
        if self.data_format == 'channels_first':
            raise ValueError('HyperConv2d is designed only for channels_last. '
                             'The input must be changed to channels last!')
        else:
            return -1

    def _get_input_channel(self, input_shape):
        channel_axis = self._get_channel_axis()
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        return int(input_shape[channel_axis])

    def build(self, input_shape):
        """
        Parameters
        ----------
        input_shape : tuple
            The input data is: batch x dim_x x dim_y x (dim_algebra x n), where n is arbitrary nonzero integer, i.e. the last 
            axis has dimension that is an integer multiple of algebra dimension.

        Returns
        -------
        None.

        """
        input_shape = tensor_shape.TensorShape(input_shape)
        input_channel = self._get_input_channel(input_shape)
        
        assert input_shape[-1] % self.algebra_dim == 0        
            
        input_dim = input_channel // self.algebra_dim
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        
          
        #make all weights
        #self.kernel = tf.concat([[self.add_weight(shape=kernel_shape, \
        #                                          initializer=self.initializer, trainable=True,)] for i in range(self.algebra_dim)],0)
            
        self.kernel = tf.Variable(tf.concat([[self.initializer(shape=kernel_shape, dtype=tf.dtypes.float32)] for i in range(self.algebra_dim)],0), trainable=True, name="HyperKernel")
        
        if self.use_bias:
            self.bias = self.add_weight(
                name='HyperBias',
                shape=(self.algebra_dim*self.filters,),
                initializer=self.bias_initializer,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None

    def call(self, inputs):
        """
        Parameters
        ----------
        inputs : numpy.array
            This is standard numpy.array data alligned to the shape: batch x dim_x x dim_y x (dim_algebra x n), where n is arbitrary nonzero integer, i.e. the last 
            axis has dimension that is an integer multiple of algebra dimension.

        Returns
        -------
        outputs : numpy.array
            This is a standard numpy.array of dimension batch x new_dim_x x new_dim_y x (dim_algebra x n), where n is arbitrary nonzero integer, i.e. the last 
            axis has dimension that is an integer multiple of algebra dimension. new_dim_x and new_dim_y results from applying padding and trides to the shape dim_x x dim_y.
        """
        W = tf.tensordot(self.A, self.kernel, axes=[1,0])    
        W = tf.transpose(W, [1,2,3,0,4,5])
        ss = W.shape
        W = tf.reshape(W, [ss[0],ss[1],ss[2],ss[3]*ss[4],ss[5]])
        
        outputs = []
        for i in range(self.algebra_dim):
            outputs.append(tf.nn.conv2d(inputs, W[i,:], strides=self.strides, padding=self.padding))
        outputs = tf.concat(outputs,axis=3)           
        
        
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs,self.bias)

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs
    
    
    
#######

class HyperConv1D(layers.Layer):
    """Convolutional 1D layer with computations based on arbitrary algebra.
    The input data is: batch x dim_x x (dim_algebra x n), where n is arbitrary nonzero integer, i.e. the last 
    axis has dimension that is an integer multiple of algebra dimension.
    
    The class is based on the ideas from HyperConv2D adjusted to 1D calculations described in:
    G. Vieira and M. Eduardo Valle, "Acute Lymphoblastic Leukemia Detection Using Hypercomplex-Valued Convolutional Neural Networks," 2022 International Joint Conference on Neural Networks (IJCNN), Padua, Italy, 2022, pp. 1-8, doi: 10.1109/IJCNN55064.2022.9892036.
    The calulations are adjusted to TensorFlow tensor operations, and are enhanced to 
    arbitrary algebras.
    """
    def __init__(self,
                 filters,
                 kernel_size,
                 stride=1,
                 padding='VALID',
                 use_bias=True,
                 bias_initializer='zeros',
                 activation=None,
                 initializer=tf.keras.initializers.GlorotNormal(seed=0),
                 data_format=None,
                 algebra = Algebra.Quaternions,
                ):
        """
        

        Parameters
        ----------
        filters : int
            the dimension of the output space (the number of filters in the convolution).
        kernel_size : int or tuple/list of 1 integer
            specifying the size of the convolution window.
        stride : int or tuple/list of 1 integer, optional
            specifying the stride length of the convolution. strides > 1 is incompatible with dilation_rate > 1.. The default is 1.
        padding : string, "valid", "same" or "causal"(case-insensitive), optional
            "valid" means no padding. "same" results in padding evenly to the left/right or up/down of the input. When padding="same" and strides=1, the output has the same size as the input. "causal" results in causal(dilated) convolutions, e.g. output[t] does not depend oninput[t+1:]. Useful when modeling temporal data where the model should not violate the temporal order.. The default is 'VALID'.
        use_bias : bool, optional
            if True, bias will be added to the output. The default is True.
        bias_initializer : Initializer for the convolution kernel, optional
            If None, the default initializer ("zeros") will be used. The default is 'zeros'.
        activation : Activation function, optional
            If None, no activation is applied. The default is None.
        initializer : Kernel initializer type, optional
            Kernel initializer type. The default is tf.keras.initializers.GlorotNormal(seed=0).
        data_format : string, optional
            The ordering of the dimensions in the inputs. HyperConv1d is designed only for channels_last. The default is None.
        algebra : StructureConstants type, optional
            The StructureConstants object that describes multiplication table of the algebra. The default is Algebra.Quaternions.

        Returns
        -------
        None.

        """
        super(HyperConv1D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias
        self.bias_initializer = bias_initializer
        self.activation = activations.get(activation)
        self.initializer = initializers.get(initializer)
        self.data_format = data_format
        self.A = tf.convert_to_tensor(algebra.getA(), dtype = np.float32)
        self.algebra_dim = algebra.getDim()     
        

    def _get_channel_axis(self):
        if self.data_format == 'channels_first':
            raise ValueError('HyperConv1d is designed only for channels_last. '
                             'The input must be changed to channels last!')
        else:
            return -1

    def _get_input_channel(self, input_shape):
        channel_axis = self._get_channel_axis()
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        return int(input_shape[channel_axis])

    def build(self, input_shape):
        """
        Builds HyperConv1D neural network.

        Parameters
        ----------
        input_shape : tuple
            The input data is: batch x dim_x x (dim_algebra x n), where n is arbitrary nonzero integer, i.e. the last 
            axis has dimension that is an integer multiple of algebra dimension.

        Returns
        -------
        None.

        """
        input_shape = tensor_shape.TensorShape(input_shape)
        input_channel = self._get_input_channel(input_shape)
        
        assert input_shape[-1] % self.algebra_dim == 0
          
        input_dim = input_channel // self.algebra_dim
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        
          
        #make all weights
        self.kernel = tf.Variable(tf.concat([[self.initializer(shape=kernel_shape, dtype=tf.dtypes.float32)] for i in range(self.algebra_dim)],0), trainable=True, name="HyperKernel")
        
        if self.use_bias:
            self.bias = self.add_weight(
                name='HyperBias',
                shape=(self.algebra_dim*self.filters,),
                initializer=self.bias_initializer,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None

    def call(self, inputs):
        """
        Parameters
        ----------
        inputs : numpy.array
            This is standard numpy.array data alligned to the shape: batch x dim_x x (dim_algebra x n), where n is arbitrary nonzero integer, i.e. the last 
            axis has dimension that is an integer multiple of algebra dimension.

        Returns
        -------
        outputs : numpy.array
            This is a standard numpy.array of dimension batch x new_dim_x x (dim_algebra x n), where n is arbitrary nonzero integer, i.e. the last 
            axis has dimension that is an integer multiple of algebra dimension. new_dim_x results from applying padding and trides to the shape dim_x.

        """
        W = tf.tensordot(self.A, self.kernel, axes=[1,0])    
        W = tf.transpose(W, [1,2,0,3,4])
        ss = W.shape
        W = tf.reshape(W, [ss[0],ss[1],ss[2]*ss[3],ss[4]])
        
        outputs = []
        for i in range(self.algebra_dim):
            outputs.append(tf.nn.conv1d(inputs, W[i,:], stride=self.stride, padding=self.padding))
        outputs = tf.concat(outputs,axis=2)
        
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs,self.bias)

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs
    

#####

class HyperConv3D(layers.Layer):
    """Convolutional 3D layer with computations based on arbitrary algebra.
    The input data is: batch x dim_x x dim_y x dim_z x (dim_algebra x n), where n is arbitrary nonzero integer, i.e. the last 
    axis has dimension that is an integer multiple of algebra dimension.
    
    The class is based on the ideas from HyperConv2D adjusted to 3D calculations described in:
    G. Vieira and M. Eduardo Valle, "Acute Lymphoblastic Leukemia Detection Using Hypercomplex-Valued Convolutional Neural Networks," 2022 International Joint Conference on Neural Networks (IJCNN), Padua, Italy, 2022, pp. 1-8, doi: 10.1109/IJCNN55064.2022.9892036.
    The calulations are adjusted to TensorFlow tensor operations, and are enhanced to 
    arbitrary algebras.
    """
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=[1,1,1,1,1],
                 padding='VALID',
                 use_bias=True,
                 bias_initializer='zeros',
                 activation=None,
                 initializer=tf.keras.initializers.GlorotNormal(seed=0),
                 data_format=None,
                 algebra = Algebra.Quaternions,  # Quaternion algebra
                ):
        """
        

        Parameters
        ----------
        filters : int
            the dimension of the output space (the number of filters in the convolution).
        kernel_size : int or tuple/list of 3 integer
            specifying the size of the convolution window.
        strides : int or tuple/list of at least 5 integer, optional
            specifying the stride length of the convolution. strides > 1 is incompatible with dilation_rate > 1. The default is [1,1,1,1,1].
        padding : string, optional
            either "valid" or "same" (case-insensitive). "valid" means no padding. "same" results in padding evenly to the left/right or up/down of the input. When padding="same" and strides=1, the output has the same size as the input. The default is 'VALID'.
        use_bias : bool, optional
           if True, bias will be added to the output. The default is True.
        bias_initializer : string, optional
            Initializer for the bias vector. If None, the default initializer ("zeros") will be used. The default is 'zeros'.
        activation : string, optional
            Activation function. If None, no activation is applied. The default is None.
        initializer : Kernel initializer type, optional
            Kernel initializer type. The default is tf.keras.initializers.GlorotNormal(seed=0).
        data_format : string, optional
            HyperConv3D is designed only for channels_last. The default is None.
         algebra : StructureConstants type, optional
             The StructureConstants object that describes multiplication table of the algebra. The default is Algebra.Quaternions.
        Returns
        -------
        None.

        """
        super(HyperConv3D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.bias_initializer = bias_initializer
        self.activation = activations.get(activation)
        self.initializer = initializers.get(initializer)
        self.data_format = data_format
        self.A = tf.convert_to_tensor(algebra.getA(), dtype = np.float32)
        self.algebra_dim = algebra.getDim()
       
        

    def _get_channel_axis(self):
        if self.data_format == 'channels_first':
            raise ValueError('HyperConv3D is designed only for channels_last. '
                             'The input must be changed to channels last!')
        else:
            return -1

    def _get_input_channel(self, input_shape):
        channel_axis = self._get_channel_axis()
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        return int(input_shape[channel_axis])

    def build(self, input_shape):
        """
        

        Parameters
        ----------
        input_shape : tuple
            The input data is: batch x dim_x x dim_y x dim_z x (dim_algebra x n), where n is arbitrary nonzero integer, i.e. the last 
            axis has dimension that is an integer multiple of algebra dimension.

        Returns
        -------
        None.

        """
        input_shape = tensor_shape.TensorShape(input_shape)
        input_channel = self._get_input_channel(input_shape)
        
        assert input_shape[-1] % self.algebra_dim == 0
            
        input_dim = input_channel // self.algebra_dim
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        #make all weights
        self.kernel = tf.Variable(tf.concat([[self.initializer(shape=kernel_shape, dtype=tf.dtypes.float32)] for i in range(self.algebra_dim)],0), trainable=True, name="HyperKernel")
        
            
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.algebra_dim*self.filters,),
                initializer=self.bias_initializer,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None

    def call(self, inputs):
        """
        Parameters
        ----------
        inputs : numpy.array
            This is standard numpy.array data alligned to the shape: batch x dim_x x dim_y x dim_z x (dim_algebra x n), where n is arbitrary nonzero integer, i.e. the last 
            axis has dimension that is an integer multiple of algebra dimension.

        Returns
        -------
        outputs : numpy.array
            This is a standard numpy.array of dimension batch x new_dim_x x new_dim_y x new_dim_z x (dim_algebra x n), where n is arbitrary nonzero integer, i.e. the last 
            axis has dimension that is an integer multiple of algebra dimension. new_dim_x, new_dim_y, and new_dim_z results from applying padding and trides to the shape dim_x, dim_y, and dim_z.

        """
        W = tf.tensordot(self.A, self.kernel, axes=[1,0])    
        W = tf.transpose(W, [1,2,3,4,0,5,6])
        ss = W.shape
        W = tf.reshape(W, [ss[0],ss[1],ss[2],ss[3],ss[4]*ss[5],ss[6]])
    
        outputs = []
        for i in range(self.algebra_dim):
            outputs.append(tf.nn.conv3d(inputs, W[i,:], strides=self.strides, padding=self.padding))
        outputs = tf.concat(outputs,axis=4)
        
        
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs,self.bias)

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs

    

#####

def test_outputShape_Conv1D():
    """
    Test against correct shape of output.

    Returns
    -------
    None.

    """
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input

    x = Input(shape=(None, 4))
    y = HyperConv1D(3, (2,))(x)
    model = Model(x, y)

    inp =  np.random.rand(10,10,4)
    y = model.predict(inp, verbose=0)
    assert y.shape == (10, 9, 12)
 
    
def test_outputShape_Conv2D():
    """
    Test against correct shape of output.

    Returns
    -------
    None.

    """
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input

    x = Input(shape=(None,5, 4))
    y = HyperConv2D(3, (2,2))(x)
    model = Model(x, y)

    inp =  np.random.rand(20,10,5,4)
    y = model.predict(inp, verbose=0)
    assert y.shape == (20, 9, 4, 12)


def test_outputShape_Conv3D():
    """
    Test against correct shape of output.

    Returns
    -------
    None.

    """
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input

    x = Input(shape=(None,5,5,4))
    y = HyperConv3D(3, (2,2,2))(x)
    model = Model(x, y)

    inp =  np.random.rand(20,5,5,5,4)
    y = model.predict(inp, verbose=0)
    assert y.shape == (20, 4, 4, 4, 12)
    
    
if __name__=="__main__":
    test_outputShape_Conv1D()
    test_outputShape_Conv2D()
    test_outputShape_Conv3D()
