function val = kernelnorm(K,i,weight)
%val =  K(i,i) - 2*K(i,:)*weight + weight'*K*weight;

%   Disciplined convex programming information:
%       NORM is convex. 
%       NORM is nonmonotonic, so its input must be affine.

%
% Argument map
%

persistent remap1 remap2
if isempty( remap2 ),
    remap1 = cvx_remap( 'log-convex' );
    remap2 = cvx_remap( 'affine', 'log-convex' );
end

%
% Check arguments
%

error( nargchk( 3, 3, nargin ) ); %#ok
if size(K,2) ~= size(weight,1),
    error('size(K,2) and size(weight,1) must be the same size')
elseif i < 1 || size(K,1) < i || size(K,2) < i,
    error( 'i must be within the size of K' );
end

[m,n] = size(K);
    
%
% Vector norm
%

if isempty( K ),
    cvx_optval = cvx( 0 );
    return
end

xc = cvx_classify( K );
if ~all( remap2( xc ) ),
    error( 'Disciplined convex programming error:\n    Cannot perform the operation norm( {%s}, %g )', cvx_class( x ), p );
end

cvx_optval = sqrt( K(i,i) - 2*K(i,:)*weight + weight'*K*weight );
