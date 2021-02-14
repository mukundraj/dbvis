% 2017-11-23 File to get FB net gram from the matlab WL codef or student
% networks.
% Uses matlab bgl by David Gleich for fast computation of sp matrices.

% addpath(genpath('/Users/mukundraj/Downloads/matlab_bgl 2/'));
% UPDATE maxpath accrodingly for any new data set


datapath = '/Users/mukundraj/Desktop/work/projects/kerneldepth/kerneldepth/data/2017-01-23/facebook100/';

ip_file = '/Users/mukundraj/Desktop/work/projects/kerneldepth/kerneldepth/data/processed/fb-studs/name_lists/smallest_80.txt'
path_csv = '/Users/mukundraj/Desktop/work/projects/kerneldepth/kerneldepth/data/processed/fb-studs/kernels/fb80_stud_sp_normalized.csv';
op_home_dir = '/Users/mukundraj/Desktop/work/projects/kerneldepth/kerneldepth/results/2017-06-18/socialnets/fb_studs/smallest80_normalized/';
% sp_mat_store = '/Users/mukundraj/Desktop/work/projects/kerneldepth/kerneldepth/data/processed/fb-studs/sp_matrix/fb5_stud_spmats.mat';

fid = fopen(ip_file);
M = textscan(fid,'%s');
fclose(fid);
name_list = M{1,1};
N = length(name_list);
A_studs = cell(1,N);

%sprintf('%04d', M(i))
years = [0];

% first pass for learning years
for i=1:N    
   
   load(strcat(datapath,name_list{i}));
    
     % also filter out the non students
    sids = local_info(:,1)==1;
    
    gender = local_info(sids,2);
    year = local_info(sids,6);
    year(year<2004 | year >2010)=0;
    years = [years year'];
       
end

unique_years = unique(years);

%
clear G;

for i=1:N    
   
   load(strcat(datapath,name_list{i}));
    
     % also filter out the non students
    sids = local_info(:,1)==1;
    
%     gender = local_info(sids,2);
    year = local_info(sids,6);
    year(year<2004 |year > 2010)=0;
    
    labels = year;
    [temp,labels] = ismember(labels,unique_years);
    
    A_studs{i} = A(sids,sids);
    [m,m] = size(A_studs{i});
    ns(i) = m;
    adj = A_studs{i};
    [r,c] = find(adj);
    edges = [r,c]; % subtract 1 if using outside matlab
    
    sp_mats(i).sp = all_shortest_paths(A_studs{i});
    
    G(i).am = adj;
    G(i).w = adj;

    al=cellfun(@(x) find(x),num2cell(adj,2),'un',0);
%     celldisp(B)
%     [n,junk] = size(edges);
    
    G(i).al = al;
    G(i).nl.values = labels;
    G(i).sp = sp_mats(i).sp;
    
    
end

% save(sp_mat_store,'sp_mats','-v7.3');

[K, runtime, Phi, maxpath] = spkernel(G, 1,1);
[m n] = size(Phi);

% starting the second pass
csvwrite(path_csv,K);

num_labels = length(unique_years);
gid = length(G) + 1;
id = 1;

for l = 0:maxpath
    for ni=1:num_labels
        for nj=ni:num_labels
            
            dic(id).keys = [ni nj l]; % posid -> sp signature
            
            a = [0 1;1 0];
            w = [0 l;l 0];
            G(gid).am = a;
            G(gid).nl.values = [ni nj]';
            al=cellfun(@(x) find(x),num2cell(a,2),'un',0);
            G(gid).al = al;
            G(gid).w = w;
            G(gid).sp = all_shortest_paths(sparse(G(gid).w));

        gid = gid+1;
        id = id+1;
        end
    end
end
    
[K, runtime, Phi, maxpath] = spkernel(G, 1,1);
[m n] = size(Phi);

clear maper;
for id=1:m

    maper(id).list = containers.Map();
  
end

for id=1:m
    [inds,vals] = find(Phi(:,id+N));
    
    

    for j=1:length(inds)
           jid = inds(j);
           maper(id).list(num2str(dic(jid).keys))=1;
%            maper(jid).list(num2str(dic(1).keys))=1;
    end
    
end


fileID = fopen(strcat(op_home_dir, 'histdetails.txt'),'w');
for i=1:m
    
   tmp = keys(maper(i).list);
   fprintf(fileID, strcat(strjoin(tmp,' , '),'\n'));
   [inds,i,vals] = find(Phi(:,1));
    
end
fclose(fileID);

fileID = fopen(strcat(op_home_dir, 'chem_sp_vectors.txt'),'w');
for i=1:N
    
   vec = zeros(1,m);
   [inds,dump,vals] = find(Phi(:,i));
   vec(inds) = vals;
   fprintf(fileID, strcat(num2str(vec),'\n'));
    
end
fclose(fileID);


fileID = fopen(strcat(op_home_dir, 'sparserep_sp_vectors.txt'),'w');
for i=1:N
mixed = ''; 
for j=1:length(inds)
    mixed = strcat(mixed, strcat('(',num2str(inds(j)),',', num2str(vals(j)),') '));
end
fprintf(fileID,strcat(mixed,'\n'));
end
fclose(fileID);
