% ellipse - the learned ellipse model
% K - (n x m) where n is training size, m is test size
% Kxx (m x 1) the kernel inner products of all test elements
% gammav - regularization parameter
function mahal = krmvce_mahal(ellipse,K,Kxx,gammav)

[n,m] = size(K);

assert(m == length(Kxx));

mahal = zeros(m,1);

onem = ones(m,m)*(1/m);

%Kc = K - ellipse.Kmean1 - K*onem + ellipse.Kmean2;
Kc = K;

V = ellipse.V;
A = ellipse.A;
L = ellipse.L;
Lsqrt = ellipse.Lsqrt;
Linvsqrt = ellipse.Linvsqrt;

onen = ones(n,1)*(1/n);
for i=1:m
	Kxxc = Kxx(i);
	rhs = Kc(:,i)'*A*V*Lsqrt*((L+eye(n)*gammav)\(Linvsqrt*V'*A*Kc(:,i)));
	mahal(i) = (1/gammav) * (Kxxc - rhs);
end

