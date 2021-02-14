% 2017-11-07 File to get FB net gram from the matlab WL codef or student
% networks.
% Uses matlab bgl by David Gleich for fast computation of sp matrices.

rng('default');
addpath(genpath('..'));

datapath = '/Users/mukundraj/Desktop/work/projects/kerneldepth/kerneldepth/data/2017-01-23/facebook100/';

% sp_mat_path = '/Users/mukundraj/Desktop/work/projects/kerneldepth/kerneldepth/data/processed/fb-nonstuds/sp_matrix/fb100_stud_WLspmat.mat';
path_csv = '/Users/mukundraj/Desktop/work/projects/kerneldepth/kerneldepth/data/processed/fb-studs/kernels/fb100_stud_WL.csv';

% Prepare the data object
name_list = what(datapath);
name_list = name_list(1).mat;


N = 100;
A_studs = cell(1,N);


ns = zeros(1,N);

sp_mat_path = '/Users/mukundraj/Desktop/work/projects/kerneldepth/kerneldepth/data/processed/fb-nonstuds/sp_matrix/fb100_nonstud_spmat.mat';

% sp_mat = matfile(sp_mat_path);
% sp_mats = sp_mat.sp_mats;



for i=1:N
    filename = strcat(datapath, name_list(i));
    load(filename{1,1});
   
    % also filter out the students
    sids = local_info(:,1)==1;
% 	sids = sids(1:100);
    
    gender = local_info(sids,2);
    labels = gender;
    
    A_studs{i} = A(sids,sids);
    [m,m] = size(A_studs{i});
    ns(i) = m;
    adj = A_studs{i};
    [r,c] = find(adj);
    edges = [r,c]; % subtract 1 if using outside matlab
    
    G(i).am = adj;

    al=cellfun(@(x) find(x),num2cell(adj,2),'un',0);
%     celldisp(B)
%     [n,junk] = size(edges);
    
    G(i).al = al;
    G(i).nl.values = labels;
%     G(i).sp = sp_mats(i).sp;
    
    
end



% Compute the kernel
iters = 10;

% [K,runtime] = WLspdelta(G,iters,1,1)

[K,runtime] = WL(G,iters,1)

% for i=2:iters
%     K{1} = K{1}+K{i};
% end

% Write the kernel



csvwrite(path_csv,K{iters+1});

% % path_txt = '/Users/mukundraj/Desktop/work/projects/kerneldepth/kerneldepth/data/processed/social_net_kernel/names.txt';
% formatSpec = '%s \n';                      
% fileID = fopen('names.txt','w');
% for row = 1:N
%     filename = name_list(row);
%     fprintf(fileID,formatSpec,filename{1,1});
% end                
% fclose(fileID);