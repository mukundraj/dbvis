function val = kernelnorm(K,i,weight)
%val =  K(i,i) - 2*K(i,:)*weight + weight'*K*weight;
val = sqrt( K(i,i) - 2*K(i,:)*weight + weight'*K*weight );
