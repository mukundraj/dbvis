from graphdepth.ellipse.mvee import mvee
from numpy import zeros,array,dot,exp,power
from scipy.linalg import norm

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
v = 0 # means use hard boundary

##########################
print "no regularization, gamma = 0"
##########################

gamma = 0
ellipse = mvee(X,gamma)

inlier = ellipse.classify(Xtest) # should be False
print inlier, " == False"

Xtest = X[:,1]
inlier = ellipse.classify(Xtest) # should be True
print inlier, " == True"

gamma = 1e1
##########################
print "with regularization, gamma = {}".format(gamma)
##########################

ellipse = mvee(X,gamma)

Xtest = array([100,100])
inlier = ellipse.classify(Xtest) # should be False
print inlier, " == False"

Xtest = X[:,1]
inlier = ellipse.classify(Xtest) # should be True
print inlier, " == True"

gamma = 1e5
##########################
print "with regularization, gamma = {}".format(gamma)
##########################

ellipse = mvee(X,gamma)

Xtest = array([100,100])
inlier = ellipse.classify(Xtest) # should be False, but will be True due to regularization value being really large
print inlier, " == False"

Xtest = X[:,1]
inlier = ellipse.classify(Xtest) # should be True
print inlier, " == True"