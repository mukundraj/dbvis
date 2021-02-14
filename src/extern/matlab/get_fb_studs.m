rng('default');
addpath(genpath('..'));

% datapath = '../data/facebook34/';
datapath = '/Users/mukundraj/Desktop/work/projects/kerneldepth/kerneldepth/data/2017-01-23/facebook100/';
outpath = '/Users/mukundraj/Desktop/work/projects/kerneldepth/kerneldepth/data/processed/fb-studs/graphs/';

name_list = what(datapath);
name_list = name_list(1).mat;



N = 100;
A_studs = cell(1,N);
ns = zeros(1,N);

for i=1:N
   filename = strcat(datapath, name_list(i));
   load(filename{1,1});
   
    % also filter out the non students
    sids = local_info(:,1)==1;
    
    gender = local_info(sids,2);
    labels = gender;
    
    
    
    A_studs{i} = A(sids,sids);
    [m,m] = size(A_studs{i});
    ns(i) = m;
    adj = A_studs{i};
    [r,c] = find(adj);
    edges = [r,c];
    edges = edges - 1;
    
    fid = fopen(strcat(outpath,sprintf('%04d',i),'.txt'),'wt');
    fprintf(fid,'v0\tv1\n');
    for ii = 1:size(edges,1)
        fprintf(fid,'%g\t',edges(ii,:));
        fprintf(fid,'\n');
    end
    fclose(fid)
    
    fid = fopen(strcat(outpath,sprintf('%04d',i),'_labels.txt'),'wt');
    fprintf(fid,'gender\n');
    for ii = 1:length(labels)
        fprintf(fid,'%g',labels(ii));
        fprintf(fid,'\n');
    end
    fclose(fid)
end
ns
size(edges)