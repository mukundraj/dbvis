% 2017-11-12 File to get wl sp gram for mutag data set.

sp_mat_path = '/Users/mukundraj/Desktop/work/projects/kerneldepth/kerneldepth/data/processed/mutag/wl_sp/';

h = 10;
nl = 1;
load('MUTAG');

[K,runtime] = WLspdelta(MUTAG,h,nl,0);


writepath = strcat(sp_mat_path,'h_',num2str(h),'.txt');
csvwrite(writepath,K{h});