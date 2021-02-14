% compute the enclosing ellipse using cvx
%
% params:
% K - kernel gram matrix
% C - slack penalty weight
function ellipse = esvdd(K,C)

n = size(K,1);
Kdiag = diag(K);

cvx_begin quiet
	variable alphav(n,1)
	alphav >= 0;
	variable betav(n,1)
	betav >= 0;
	variable eta(n,1)
	eta >= 0;
	variable r
	for i=1:n
		%sqrt(K(i,i) - 2*K(i,:)*alphav + alphav'*K*alphav) + sqrt(K(i,i) - 2*K(i,:)*betav + betav'*K*betav) <= r + eta(i);
		kernelnorm(K,i,alphav) + kernelnorm(K,i,betav) <= r + eta(i);
	end
	minimize(r + C*sum(eta));
cvx_end

ellipse.alphav = alphav;
ellipse.betav = betav;
ellipse.r = r;
ellipse.alphavKalphav = alphav'*K*alphav;
ellipse.betavKbetav = betav'*K*betav;

