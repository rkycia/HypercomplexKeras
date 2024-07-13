#! /usr/bin/env python

import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, activations, initializers


from HypercomplexKeras import Algebra
#import Algebra


class HyperDense(layers.Layer):
    """Dense layer when the weight are calculated based on arbitrary hyperalgebra.
    The usage is similar to standard hyperdense layer. The input of data is 
    batch x (dim_algebra x n), where n is arbitrary nonzero integer, i.e. the last 
    axis has dimension that is an integer multiple of algebra dimension.
    
    The class is a generalization of class described in:
    G. Vieira and M. Eduardo Valle, "Acute Lymphoblastic Leukemia Detection Using Hypercomplex-Valued Convolutional Neural Networks," 2022 International Joint Conference on Neural Networks (IJCNN), Padua, Italy, 2022, pp. 1-8, doi: 10.1109/IJCNN55064.2022.9892036.
    The calulations are adjusted to TensorFlow tensor operations, and are enhanced to 
    arbitrary algebras.
    """
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer=tf.keras.initializers.GlorotNormal(seed=0),
                 bias_initializer='zeros',
                 algebra = Algebra.Quaternions # Quaternion algebra
                ):
        """The input is the same as for Dense layer. The additional parametr is 
        algebra which is set do algebra.Quaternions.
        """
        super(HyperDense, self).__init__()
        self.units = int(units) if not isinstance(units, int) else units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.A = tf.convert_to_tensor(algebra.getA(), dtype = np.float32)
        self.algebra_dim = algebra.getDim()


    def build(self, input_shape):
        """
        Builds a hyperdense layer.

        Parameters
        ----------
        input_shape : tuple
            The tuple with shape: (batch_size, ..., input_dim). 
            The most common situation would be a 2D input with shape (batch_size, input_dim).
            The imput_dim must be divisable by the dimension of algebra.

        Returns
        -------
        None.

        """
        assert input_shape[-1] % self.algebra_dim == 0
        input_dim = input_shape[-1] // self.algebra_dim

        #make all weights
        self.kernel = tf.Variable(tf.concat([[self.kernel_initializer(shape=(input_dim, self.units), dtype=tf.dtypes.float32)] for i in range(self.algebra_dim)],0), trainable=True, name="HyperKernel")
       
        # Quaternion-valued bias
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.algebra_dim*self.units,), initializer="zeros", trainable=True, name="HyperBias")
        else:
            self.bias = tf.Variable(tf.constant(0, shape= (self.algebra_dim*self.units,)))

    def call(self, inputs):
        """
        Make a forward call.

        Parameters
        ----------
        inputs : numpy.array
            This is standard numpy.array data alligned to the shape:  batch x (algebra dimension x n) for
            a positive integer n, i.e., the last axis has dimension that is an positive integer multiple of the
            algebra dimension.

        Returns
        -------
        outputs : numpy.array
            This is a standard numpy.array of dimension batch x (algebra dimension x units).

        """  
        W = tf.tensordot(self.A, self.kernel, axes=[1,0])    
        W = tf.transpose(W, [0,2,1,3])
        
        ss = W.shape
        W = tf.reshape(W, [ss[0]*ss[1],ss[2]*ss[3]])
        outputs = tf.matmul(inputs, W)
        
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)

        if self.activation is not None:
            outputs = self.activation(outputs)
            
        return outputs
    
    
    
def test_outputShape():
    """
    Test against correct shape of output.

    Returns
    -------
    None.

    """
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input
    x = Input(shape=(8,))
    y = HyperDense(7)(x)
    model = Model(x, y)
    y = model.predict(np.array([[1,0,0,0,1,0,0,0], [1,0,0,0, 1,0,0,0]]), verbose=0)
    assert y.shape == (2,28)

    
def test_simpleCase():
    import numpy as np
    from keras.models import Sequential
    from keras.layers import Dense, Activation

    #data:
    x_train = np.array([[1,0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],  [0, 0, 0, 1]], dtype = np.dtype(float))
    y_train = np.array([[0], [1], [1],  [0]])

    #create model:
    model = Sequential()
    num_neurons = 4
    model.add(HyperDense(num_neurons))
    #model.add(Dense(num_neurons))  #for comparision
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    #plugin input shape
    model.predict(x_train, verbose=0)

    #model.summary()
    
    opt = tf.keras.optimizers.legacy.Adam()
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])


    model.fit(x_train, y_train, epochs=500, verbose=0)


    y_predict = model.predict(x_train, verbose=0)
    y_predict_quantized = np.round(y_predict).astype(int)
    #print("predicted = ", y_predict)
    #print("predicted quantized = ", y_predict_quantized)
    assert (y_train == y_predict_quantized).all()

    
if __name__=="__main__":
    test_outputShape()
    test_simpleCase()
