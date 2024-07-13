# Hypercomplex Keras
Keras-based Hypercomplex Neural Networks

Authors: Radoslaw A. Kycia, Agnieszka Niemczynowicz

Description: This package realizes general hypercomplex algebras neural networks. Dense and Convolutional (1D, 2D, 3D) hypercomplex layers are included. The package works with:
- Keras + TensorFlow (Dense, Convolutional 1D, Convolutional 2D, Convolutional 3D)
- Keras + PyTorch (Dense, Convolutional 2D - experimental implementations with TensorFlow channels alignment and PyTorch data alignment)
Package contains:
- Examples - directory contains Jupyter Notebooks illustrating some example usage of classes
- Makefile - simple makefile to run some basic tests and to generate documentation


Usage: See the examples directory.

Documentation: you can create HTML documentation by running 'make generate_doc'. The HTML files are in doc directory that will be automatically created. Additional examples are in 'Examples' directory.

Acknowledgements: This KHNN library has been supported by the [Polish National Agency for Academic Exchange](http://nawa.gov.pl/) Strategic Partnership Programme under Grant No. BPI/PST/2021/1/00031 [nawa.gov.pl](http://nawa.gov.pl/).
We would like to thanks Keras users community for suggestions about creating this library. Special thanks to François Chollet, for encouragement and technical review.

Disclaimer: This library was created with the high standards. However it requires some knowledge of neural networks and advanced mathematics to be used. It is given 'as if'. We try to test it in various situations, however, we are not responsible for all damages that can occur during the usage of the package.

We plan to develop this software, so if you want to help us, please do not hesitate to contact us.
