% minimum volume covering ellipse
%
% params:
% K - kernel gram matrix
function ellipse = mvce(X)

[d,n] = size(X);

Xcenter = mean(X,2);
Xc = X - Xcenter*ones(1,n);

cvx_begin quiet
	variable M(d,d)
	variable alphav(n,1)
	alphav >= 0;
	sum(alphav) == 1;

	M1 = zeros(d,d);
	for i=1:n
		M1 = M1 + alphav(i)*Xc(:,i)*Xc(:,i)';
	end
	M == M1;
	maximize(log_det(M));
cvx_end

mu = 0;
for i=1:n
	mah = Xc(:,i)'*(M \ Xc(:,i));
	mu = max(mah,mu);
end


ellipse.alphav = alphav;
ellipse.M = M;
ellipse.Xcenter = Xcenter;
ellipse.mu = mu;
