function pred = mvce_classify(ellipse, X)

[d,n] = size(X);

Xc = X - ellipse.Xcenter*ones(1,n);

pred = zeros(n,1);

for i=1:n
	pred(i) = Xc(:,i)'*(ellipse.M \ Xc(:,i)) <= ellipse.mu;
end
