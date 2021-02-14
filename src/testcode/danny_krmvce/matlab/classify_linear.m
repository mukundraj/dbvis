function pred = classify_linear(ellipse, X)

[d,n] = size(X);

pred = zeros(n,1);

for i=1:n
	pred(i) = norm(X(:,i)-ellipse.C1) + norm(X(:,i)-ellipse.C2) <= ellipse.r;
end

