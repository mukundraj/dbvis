% kernelized regularized minimum volume covering ellipse
%
% params:
% K - kernel gram matrix
% gammav - regularization parameter
% v - penalty coefficient for slack variable
function ellipse = krmvce(K, gammav, v)

n = size(K,1);

% center the matrix:
onem = ones(n,n)*(1/n);
Kmean1 = onem*K;
Kmean2 = onem*K*onem;
%Kc = K - Kmean1 - Kmean1' + Kmean2;
Kc = K;

% cholesky decomposition:
%C = chol(Kc);
[L,D] = ldl(Kc);
D(D < 0) = 0;
Dsqrt = diag(sqrt(diag(D)));
C = L*Dsqrt;

cvx_begin quiet
	variable alphav(n,1)
	alphav >= 0;
	n*v*alphav <= 1;
	sum(alphav) == 1;
	M = zeros(n,n);
	for i=1:n
		M = M + alphav(i)*C(:,i)*C(:,i)';
	end
	minimize(-log_det(M + eye(n)*gammav));
cvx_end

opt_val = -log_det(M + eye(n)*gammav)

alphav(alphav < 0) = 0; % sometimes these are not all non-negative ?

ellipse.alphav = alphav;
ellipse.Kmean1 = Kmean1;
ellipse.Kmean2 = Kmean2;

A = diag(sqrt(alphav)); % note: diag(A)'*diag(A) == 1
ellipse.A = A;

[V,L,U] = svd(A*Kc*A); % symmetric so U == V
%[V,L] = eig(A*Kc*A); % symmetric so U == V
ellipse.V = V;
ellipse.L = L;
ellipse.Lsqrt = sqrt(L);
ellipse.Linvsqrt = pinv(sqrt(L));

% find the maximum mahal distance for dataset
mahal = krmvce_mahal(ellipse, K, diag(K), gammav);
mu = max(mahal)
ellipse.mu = mu;
