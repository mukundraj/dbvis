ellipse = esvdd(K,C);

c1 = X*ellipse.alphav
c2 = X*ellipse.betav
ellipse.r

clf
scatter(X(1,:),X(2,:),'rx')
hold on
scatter(c1(1),c1(2),'bo')
scatter(c2(1),c2(2),'g^')
minv = min(X(:))
maxv = max(X(:))
ylim([minv,maxv])
xlim([minv,maxv])
