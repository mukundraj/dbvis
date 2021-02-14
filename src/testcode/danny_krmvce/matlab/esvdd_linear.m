% compute the enclosing ellipse using cvx
%
% params:
% K - kernel gram matrix
% C - slack penalty weight
function ellipse = esvdd_linear(X,C)

[d,n] = size(X);

%C1 = zeros(2,1);
%C1(1) = 1.0*min(X(1,:));
%C2 = zeros(2,1);
%C2(1) = 1.0*max(X(1,:));

cvx_begin quiet
	variable C1(d,1)
	variable C2(d,1)
	variable eta(n,1)
	eta >= 0;
	variable r
	for i=1:n
		norm(X(:,i)-C1) + norm(X(:,i)-C2) <= r + eta(i)
		%norm(X(:,i)-C1) + norm(X(:,i)-C2) <= r 
	end
	minimize(r + C*sum(eta));
	% if we could include the distance between foci in the optimization (maximize distance) then we can solve this.
	% how to we do that in a reasonable way?
	%minimize(r^2)
cvx_end
	
ellipse.C1 = C1;
CC1 = ellipse.C1
ellipse.C2 = C2;
CC2 = ellipse.C2
ellipse.r = r;
rr = ellipse.r
ellipse.eta = eta;
eta = eta

