% compute the enclosing ellipse using cvx
%
% params:
% K - kernel gram matrix
% C - slack penalty weight
function ellipse = esvdd_brute(X,C1,C2)

[d,n] = size(X);

cvx_begin quiet
	variable r
	for i=1:n
		norm(X(:,i)-C1) + norm(X(:,i)-C2) <= r 
	end
	minimize(r)
cvx_end
	
ellipse.C1 = C1;
CC1 = ellipse.C1
ellipse.C2 = C2;
CC2 = ellipse.C2
ellipse.r = r;
rr = ellipse.r

