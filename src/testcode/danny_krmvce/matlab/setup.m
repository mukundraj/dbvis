clear; clf; close all;
d = 2;
n = 4;
%X = diag([30,2])*randn(d,n);
X = zeros(d,n);
X(:,1) = [-10,0];
X(:,2) = [10,0];
X(:,3) = [0,-1];
X(:,4) = [0,1];
Xcenter = mean(X,2);
Xc = X - Xcenter*ones(1,n);
K = Xc'*Xc;
C = 1e2;
