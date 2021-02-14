# graphdepth - support code for computing graph depth using kernels

## Contents:

* KRMVCE,MVCE - code for computing the **minimum volume covering ellipse** in kernel space
* SVDD - code for computing **suport vector data description** (using spheres) called svdd and **extended support vector data description** (using ellipses) called esvdd


## Prerequisites 

* numpy 
> pip install numpy
* scipy
> pip install scipy
* cvxpy
> pip pinstall cvxpy

## Usage

Add the location of this folder to your ``PYTHONPATH`` environment variable, then ``import svdd``.

See ``tests/test_svdd.py`` and ``tests/test_mvce.py`` for complete examples / tests of usage.

Example using krmvce (kernel regularized minimum volume covering ellipse):

```python
from svdd import krmvce

# simple example:
d = 2
n = 4
X = zeros((d,n))
X[:,0] = [-10,0]
X[:,1] = [10,0]
X[:,2] = [0,-1]
X[:,3] = [0,1]
Xtest = array([100,100])

# parameters
gamma = 1e-1
v = 1

##########################
# linear kernel
##########################

K = linearkernel(X,X) # ie dot(X.T,X)
ellipse = krmvce(K,gamma,v)

Ktest = linearkernel(X,Xtest)
Kxx = kernel(Xtest,Xtest)
inlier = ellipse.classify(Ktest,Kxx) # should be False

Ktest = K[:,1]
Kxx = K[1,1]
inlier = ellipse.classify(Ktest,Kxx) # should be True

##########################
# gaussian kernel
##########################

K = gaussiankernel(X,X) # ie K[i,j] = exp(-dot(X[:,i].T,X[:,j])/(2*sigma*sigma))
ellipse = krmvce(K,gamma,v)

Ktest = gaussiankernel(X,Xtest)
Kxx = kernel(Xtest,Xtest)
inlier = ellipse.classify(Ktest,Kxx) # should be False

Ktest = K[:,1]
Kxx = K[1,1]
inlier = ellipse.classify(Ktest,Kxx) # should be True
```