# Hypercomplex Keras
Keras-based Hypercomplex Neural Networks

**Authors**: Radoslaw A. Kycia, Agnieszka Niemczynowicz

**Github repositiory**: [Github](https://github.com/rkycia/HypercomplexKeras)

**Description**: This package realizes general hypercomplex algebras neural networks. Algebras are realized by [Algebra](https://github.com/rkycia/HypercomplexKeras/blob/main/src/HypercomplexKeras/Algebra.py) module. Dense and Convolutional (1D, 2D, 3D) hypercomplex layers are included. The package works with:

- Keras + TensorFlow ([Dense](https://github.com/rkycia/HypercomplexKeras/blob/main/src/HypercomplexKeras/Hyperdense.py), [Convolutional 1D, Convolutional 2D, Convolutional 3D](https://github.com/rkycia/HypercomplexKeras/blob/main/src/HypercomplexKeras/Convolutional.py))

- Keras + PyTorch ([Dense](https://github.com/rkycia/HypercomplexKeras/blob/main/src/HypercomplexKeras/HyperdenseTorch.py), [Convolutional 2D](https://github.com/rkycia/HypercomplexKeras/blob/main/src/HypercomplexKeras/experimental/ExperimentalHyperConvolutionalTorch.py) - experimental implementations with TensorFlow channels alignment and PyTorch data alignment)

Package also contains:

- [examples](https://github.com/rkycia/HypercomplexKeras/tree/main/examples) - directory contains Jupyter Notebooks illustrating some example usage of classes

- [Makefile](https://github.com/rkycia/HypercomplexKeras/blob/main/src/HypercomplexKeras/Makefile) - simple makefile to run some basic tests and to generate documentation


**Usage**: See the [examples directory](https://github.com/rkycia/HypercomplexKeras/tree/main/examples) on Github for Jupyter Notebooks illustrating the usage or the manuscript: [Agnieszka Niemczynowicz, Radosław Antoni Kycia, KHNNs: hypercomplex neural networks computations via Keras using TensorFlow and PyTorch, arXiv:2407.00452 [cs.LG]](https://arxiv.org/abs/2407.00452) for detailed description.

**Documentation**: Additional examples are in [examples directory]((https://github.com/rkycia/HypercomplexKeras/tree/main/examples). You can also see the examples in the manuscript [Agnieszka Niemczynowicz, Radosław Antoni Kycia, KHNNs: hypercomplex neural networks computations via Keras using TensorFlow and PyTorch, arXiv:2407.00452 [cs.LG]](https://arxiv.org/abs/2407.00452). You can create HTML documentation by running 'make generate_doc'. The HTML files are in doc directory that will be automatically created.

**Acknowledgements**:

* This KHNN library (from which we derived HypercomplexKeras) has been supported by the [Polish National Agency for Academic Exchange](http://nawa.gov.pl/) Strategic Partnership Programme under Grant No. BPI/PST/2021/1/00031 [nawa.gov.pl](http://nawa.gov.pl/).

* We would like to thanks Keras Users Community for suggestions about creating this library. Special thanks to François Chollet for encouragement and technical tips.

**Disclaimer**: This library was created with the high standards. However it requires some knowledge of neural networks and advanced mathematics to be used. It is given 'as if'. We try to test it in various situations, however, we are not responsible for all damages that can occur during the usage of the package.

**Literature**:

+ [Agnieszka Niemczynowicz, Radosław Antoni Kycia, *Fully tensorial approach to hypercomplex neural networks*, arXiv:2407.00449 [cs.LG]](https://arxiv.org/abs/2407.00449) - describes theory of Hypercomplex NN

+ [Agnieszka Niemczynowicz, Radosław Antoni Kycia, *KHNNs: hypercomplex neural networks computations via Keras using TensorFlow and PyTorch*, arXiv:2407.00452 [cs.LG]](https://arxiv.org/abs/2407.00452) - examples of usage of HypercomplexKeras (derived form KHNN project)

*If you find this package useful or inspiring, do not hesitate to send us feedback and cite the above manuscripts.*

We plan to develop this software, so if you want to help us, please do not hesitate to contact us.
