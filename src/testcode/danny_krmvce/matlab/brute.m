maxv = max(X(1,:));
minv = min(X(1,:));

C1 = zeros(2,1);
C2 = zeros(2,1);

pcts = .1:.1:1.0;
rs = zeros(length(pcts),1);
maxrs = zeros(length(pcts),1);
figure()
for pcti = 1:length(pcts)
	pct = pcts(pcti);
	C1(1) = pct*minv;
	C2(1) = pct*maxv;
	ellipse = esvdd_brute(X,C1,C2);
	rs(pcti) = ellipse.r;

	maxrj = 0;
	for j=1:n
		rj = norm(X(:,j)-C1)+norm(X(:,j)-C2);
		maxrj = max(rj, maxrj);
	end
	maxrs(pcti) = maxrj;

	subplot(4,3,pcti)
	pred = classify_linear(ellipse, X);
	accuracy = sum(pred) / length(pred)
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
	scatter(C1(1),C1(2),'bo')
	scatter(C2(1),C2(2),'g^')
	ylim([minv,maxv])
	xlim([minv,maxv])
	title(sprintf('pct=%.1f',pct))
end

% now run optimization over centers as well

ellipse = esvdd_linear(X,C);

maxrj = 0;
for j=1:n
	rj = norm(X(:,j)-ellipse.C1)+norm(X(:,j)-ellipse.C2);
	maxrj = max(rj, maxrj);
end

sprintf('---------- Optimal centers and radius ----------')
subplot(4,3,length(pcts)+1)
pred = classify_linear(ellipse, X);
accuracy = sum(pred) / length(pred)
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
scatter(ellipse.C1(1),ellipse.C1(2),'bo')
scatter(ellipse.C2(1),ellipse.C2(2),'g^')
ylim([minv,maxv])
xlim([minv,maxv])
title('optimal centers')

% now plot results

figure
plot(pcts,rs,'bo--')
hold on
plot(pcts,ones(length(rs),1)*ellipse.r,'g--')
legend('fixed foci','optimal foci')
title('optimized radius')

figure
plot(pcts,maxrs,'bo--')
hold on
plot(pcts,ones(length(rs),1)*maxrj,'g--')
legend('fixed foci','optimal foci')
title('max radius')


