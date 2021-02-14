

datapath = '/Users/mukundraj/Desktop/work/projects/kerneldepth/kerneldepth/data/2017-01-23/facebook34/';
name_list = what(datapath);
name_list = name_list.mat;

% N = 57;
N = 5;
% N = 100;
A_studs = cell(1,N);
ns = zeros(1,N);

is = zeros(0,1);
js = zeros(0,1);
ss = zeros(0,1);
graph_inds = zeros(0,1);

old_m = 0;

for i=1:N
   filename = strcat(datapath, name_list(i));
   load(filename{1,1});
   
    % also filter out the non students
    sids = local_info(:,1)==1;
    A_studs{i} = A(sids,sids);
   
   [m,m] = size(A_studs{i});
   ns(i) = m;
   [i2,j2,s] = find(A_studs{i});
   is = [is;old_m+i2];
   js = [js;old_m+j2];
   ss = [ss;s];
   
   old_m = old_m+m;
   graph_inds = [graph_inds;i*ones(m,1)];
   

end

total_n = sum(ns);



Afull = sparse(is,js,ss,total_n,total_n);
A = Afull;

degs = full(sum(A,2));
labels = degs;
[~,~,labels] = unique(labels);
num_nodes = size(A, 1);      
distribution_features = accumarray([(1:num_nodes)', labels], 1, [num_nodes, max(labels)]);


size(A)
sum(sum(A))


% row-normalize A
row_sum = sum(A, 2);
row_sum(row_sum==0)=1;                  % avoid dividing by zero => disconnected nodes
A = bsxfun(@times, A, 1 ./ row_sum);


% TRANSFORMATION FOR LABELS
transformation = @(features) label_diffusion(features, A);


% PK PARAMETERS 
num_iter = 10;      % number of kernel iterations to test
w = 1e-5;
w_attr = 1; 
dist = 'tv';        % 'tv' or 'hellinger'         
dist_attr = 'l1';   % 'l1' or 'l2'

K = propagation_kernel(distribution_features, graph_inds, transformation, num_iter, ...
                           'w', w,'distance', dist);

% path_csv = '/Users/mukundraj/Desktop/work/projects/kerneldepth/kerneldepth/data/processed/social_net_kernel/kernel.csv';
csvwrite('kernel.csv',K(:,:,end));

% path_txt = '/Users/mukundraj/Desktop/work/projects/kerneldepth/kerneldepth/data/processed/social_net_kernel/names.txt';
formatSpec = '%s \n';                      
fileID = fopen('names.txt','w');
for row = 1:N
    filename = name_list(row);
    fprintf(fileID,formatSpec,filename{1,1});
end                
fclose(fileID);
% % load the first A
% [i2,j2,s] = find(A_studs{1});
% 
% % for k = 1:length(i2)
% %    Afull(i2(k),j2(k)) = s(k);
% % end
% 
% % load the rest
% for i=2:N
%     
%    filename = name_list(i);
%    load(filename{1,1});
%    
% end



% [~, graph_ind] = graphconncomp(A, 'directed', false);

% copy propagation kernel folder to graph project location
% try out demo files, including the attribute files
% social net with degree distribution
% social net with gender labels
% social net with classyear attributes
% compile http://www.sandia.gov/~smartin/presentations/OpenOrd.pdf for getting layout
