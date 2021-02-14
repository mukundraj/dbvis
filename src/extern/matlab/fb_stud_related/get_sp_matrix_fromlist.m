% 2017-11-23 read the fb files and output the shortest path length matrix
% this version for computing the matrix for selected smallest graphs from
% a list. Currently for students

% addpath(genpath('/Users/mukundraj/Downloads/matlab_bgl 2/'));

% check following two params before running, also if students of non
% students
ip_file = '/Users/mukundraj/Desktop/work/projects/kerneldepth/kerneldepth/data/processed/fb-studs/name_lists/smallest_5.txt';
op_file = 'fb5_stud_spmat.mat';

op_filepath = strcat(op_path, op_file);
datapath = '/Users/mukundraj/Desktop/work/projects/kerneldepth/kerneldepth/data/2017-01-23/facebook100/';
op_path = '/Users/mukundraj/Desktop/work/projects/kerneldepth/kerneldepth/data/processed/fb-studs/sp_matrix/';


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
    years = [years year'];
       
end

unique_years = unique(years);



for i=1:N    
   
   load(strcat(datapath,name_list{i}));
    
     % also filter out the non students
    sids = local_info(:,1)==1;
    
%     gender = local_info(sids,2);
    year = local_info(sids,6);
    
    labels = year;
    [temp,labels] = ismember(labels,unique_years);
    
    A_studs{i} = A(sids,sids);
    [m,m] = size(A_studs{i});
    
    ns(i) = m;
%     adj = A_studs{i};

    sp_mats(i).sp = all_shortest_paths(A_studs{i});
   
end

save(op_filepath,'sp_mats','-v7.3');