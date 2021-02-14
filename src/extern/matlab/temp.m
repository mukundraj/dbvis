A = years;
[ii,jj,kk]=unique(A);
freq=accumarray(kk,1);
out=[ii' freq]


a = [1,2,3,4,5,6]
a(a<2 | a>5) = 0