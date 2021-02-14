ellipse = mvce(X);

[d,n] = size(X)

pred = mvce_classify(ellipse, X);
accuracy = sum(pred) / length(pred)

minv = min(X(:));
maxv = max(X(:));


figure()
inlier = find(pred);
outlier = find(~pred);
xs = linspace(1.4*minv,1.4*maxv,100);
ys = linspace(1.4*minv,1.4*maxv,100);
hold on
for xi=1:length(xs)
	Xg = zeros(d,length(ys));
	for yi=1:length(ys)
		Xg(:,yi) = [xs(xi),ys(yi)];
	end
	pr = mvce_classify(ellipse, Xg);
	scatter(Xg(1,find(pr)),Xg(2,find(pr)),'rs')
	scatter(Xg(1,find(~pr)),Xg(2,find(~pr)),'ms')
end
scatter(X(1,inlier),X(2,inlier),'kx')
hold on
scatter(X(1,outlier),X(2,outlier),'k*')
ylim([minv,maxv])
xlim([minv,maxv])

