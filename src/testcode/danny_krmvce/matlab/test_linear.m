ellipse = esvdd_linear(X,C);

%ellipse.C1(1) = .5*max(X(1,:));
%ellipse.C2(1) = .5*min(X(1,:));

c1 = ellipse.C1
c2 = ellipse.C2
r = ellipse.r

figure()
scatter(X(1,:),X(2,:),'rx')
hold on
scatter(c1(1),c1(2),'bo')
scatter(c2(1),c2(2),'g^')
minv = min(X(:))
maxv = max(X(:))
ylim([minv,maxv])
xlim([minv,maxv])

pred = classify_linear(ellipse, X);
accuracy = sum(pred) / length(pred)
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
	pr = classify_linear(ellipse, Xg);
	scatter(Xg(1,find(pr)),Xg(2,find(pr)),'rs')
	scatter(Xg(1,find(~pr)),Xg(2,find(~pr)),'ms')
end
scatter(X(1,inlier),X(2,inlier),'kx')
hold on
scatter(X(1,outlier),X(2,outlier),'k*')
ylim([minv,maxv])
xlim([minv,maxv])

