rng('default');
addpath(genpath('..'));

% datapath = '../data/facebook34/';
datapath = '/Users/mukundraj/Desktop/work/projects/kerneldepth/kerneldepth/data/2017-01-23/facebook100/';
outpath = '/Users/mukundraj/Desktop/work/projects/kerneldepth/kerneldepth/data/processed/facebook-graphs/';

name_list = what(datapath);
name_list = name_list(1).mat;

A_studs = cell(1,N);

N = 100;

ns = zeros(1,N);

for i=1:N
   filename = strcat(datapath, name_list(i));
   load(filename{1,1});
   
    % also filter out the non students
    sids = local_info(:,1)==1;
    A_studs{i} = A(sids,sids);
    [m,m] = size(A_studs{i});
    ns(i) = m;
    adj = A_studs{i};
    [r,c] = find(adj);
    edges = [r,c];
    edges = edges - 1;
    
    fid = fopen(strcat(outpath,sprintf('%04d',i),'.txt'),'wt');
    for ii = 1:size(edges,1)
        fprintf(fid,'%g\t',edges(ii,:));
        fprintf(fid,'\n');
    end
    fclose(fid)
end

size(edges)