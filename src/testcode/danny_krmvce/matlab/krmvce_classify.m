function pred = krmvce_classify(ellipse, K, Kxx, gammav)

[n,m] = size(K);

mahal = krmvce_mahal(ellipse,K,Kxx,gammav);
pred = mahal <= ellipse.mu;
%pred = mahal <= 1;

