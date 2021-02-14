gammav = 1e-1;
v = 1e0;
ellipse = krmvce(K,gammav,v); 
n = size(K,1);

pred = krmvce_classify(ellipse, K, diag(K), gammav);
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
	Xgc = Xg - Xcenter*ones(1,length(ys));
	Kg = Xc'*Xgc;
	Kxx = diag(Xgc'*Xgc);
	pr = krmvce_classify(ellipse, Kg, Kxx, gammav);
	scatter(Xg(1,find(pr)),Xg(2,find(pr)),'rs')
	scatter(Xg(1,find(~pr)),Xg(2,find(~pr)),'ms')
end
scatter(X(1,inlier),X(2,inlier),'kx')
hold on
scatter(X(1,outlier),X(2,outlier),'k*')
ylim([minv,maxv])
xlim([minv,maxv])

