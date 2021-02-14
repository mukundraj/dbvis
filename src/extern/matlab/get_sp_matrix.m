% 2017-11-03 read the fb files and output the shortest path length matrix

% addpath(genpath('/Users/mukundraj/Downloads/matlab_bgl 2/'));

datapath = '/Users/mukundraj/Desktop/work/projects/kerneldepth/kerneldepth/data/2017-01-23/facebook100/';

op_path = '/Users/mukundraj/Desktop/work/projects/kerneldepth/kerneldepth/data/processed/fb-nonstuds/sp_matrix/';

op_file = 'fb100_nonstud_spmat.mat';

op_filepath = strcat(op_path, op_file);

% Prepare the data object
name_list = what(datapath);
name_list = name_list(1).mat;


N = 100;
A_studs = cell(1,N);


ns = zeros(1,N);






for i=1:N
   
    filename = strcat(datapath, name_list(i));
    load(filename{1,1});
    
     % also filter out the non students?
    sids = local_info(:,1)==2;
	
    
    
    gender = local_info(sids,2);
    labels = gender;
    
    A_studs{i} = A(sids,sids);
    [m,m] = size(A_studs{i});
    ns(i) = m;
%     adj = A_studs{i};


    sp_mats(i).sp = all_shortest_paths(A_studs{i});
    
end

save(op_filepath,'sp_mats','-v7.3');
