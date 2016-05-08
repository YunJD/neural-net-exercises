Neural Net Exercises
==============

Solutions to select exercises from Stanford's Unsupervised Feature Learning and Deep Learning (UFLDL) Tutorial (starting with neural networks). Written in Python using Numpy, SciPy, and Tensorflow.

## UfLDL Tutorial Sites
http://deeplearning.stanford.edu/tutorial/ (Newer)

http://deeplearning.stanford.edu/wiki/index.php/UFLDL_Tutorial (Older)

Unfortuantely the *newer* site seems privy to errors, so some solutions follow the *older* site's exercises.

## Requirements
Everything is implemented for Python3. The main modules should be in the requirements file.  However:

SciPy requires a fortan compiler.  Best way I found for Linux Mint was to `sudo apt-get install gfortran` before installing SciPy.

Tensorflow requires some steps found in [Tensorflow's main page](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html#pip-installation). If you are on say, Arch Linux and have Python3.5, just rename the 34 parts in the instructions to 35 :).

Resources
==============

MNIST training files are required under the `./res/` directory.  They can be downloaded from:
http://yann.lecun.com/exdb/mnist/

Various .mat files found in the /res folder were downloaded from the *older* site.
