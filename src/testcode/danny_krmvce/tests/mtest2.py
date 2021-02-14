from src.testcode.danny_krmvce.svdd import krmvce
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
Xtest = array([[100,100,1],[100,100,1]])
# Xtest = array([100,100])
# parameters
gamma = 1e-1
v = 1

##########################
# linear kernel
##########################

def linearkernel(x,y):
    """ compute the linear kernel
    """
    return dot(x.T,y)


K = linearkernel(X,X) # ie dot(X.T,X)
ellipse = krmvce(K,gamma,v)

Ktest = linearkernel(X,Xtest)

Kxx = linearkernel(Xtest,Xtest)


inlier = ellipse.classify(Ktest,Kxx) # should be False
print inlier, " == False"

Ktest = K[:,1]
Kxx = K[1,1]
inlier = ellipse.classify(Ktest,Kxx) # should be True
print inlier, " == True"

##########################
# gaussian kernel
##########################

def estimate_sigma(X):
    """ estimate the sigma parameter by computing the average minimum distance between points.
    """
    d,n = X.shape
    sigma = 0
    for i in range(n):
        dist = 1e10
        for j in range(n):
            if i == j:
                continue
            dist = min(dist, norm(X[:,i]-X[:,j]))
        sigma += dist
    sigma /= n
    return sigma

def gaussiankernel(x,y,sigma):
    """ compute the Gaussian kernel where k(x,y) = exp( - ||x-y||^2/(2*sigma^2))
    """
    n1 = 1
    if len(x.shape) > 1:
        n1 = x.shape[1]
    n2 = 1
    if len(y.shape) > 1:
        n2 = y.shape[1]
    K = zeros([n1,n2])
    for i in range(K.shape[0]):
        if n1 > 1:
            xx = x[:,i]
        else:
            xx = x
        for j in range(K.shape[1]):
            if n2 > 1:
                yy = y[:,j]
            else:
                yy = y
            K[i,j] = exp( -(norm(xx-yy)**2) / (2*(sigma**2)) )
    return K

sigma = estimate_sigma(X)

K = gaussiankernel(X,X,sigma)
ellipse = krmvce(K,gamma,v)

print ellipse
Ktest = gaussiankernel(X,Xtest,sigma)
Kxx = gaussiankernel(Xtest,Xtest,sigma)
inlier = ellipse.classify(Ktest,Kxx) # should be False
print inlier, " == False"

Ktest = K[:,1]
Kxx = K[1,1]
inlier = ellipse.classify(Ktest,Kxx) # should be True
print inlier, " == True"