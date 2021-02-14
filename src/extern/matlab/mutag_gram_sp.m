% 2017-11-12 File to get sp gram for mutag data set.




% code to add the extra graphs to identify shortest path positions


% a = [0 1 0 0;1 0 1 0; 0 1 0 1; 0 0 1 0];
% a = [0 1;1 0];
% MUTAG(189).am = a;
% MUTAG(189).nl.values = [1 2]';
% al=cellfun(@(x) find(x),num2cell(a,2),'un',0);
% MUTAG(189).al = al;

load('MUTAG');
gid = length(MUTAG) + 1;
id = 1;


for i=1:188
    MUTAG(i).w = MUTAG(i).am;
    
end

for l = 0:15
    for ni=1:7
        for nj=ni:7
            
            dic(id).keys = [ni nj l]; % posid -> sp signature
            
            a = [0 1;1 0];
            w = [0 l;l 0];
            MUTAG(gid).am = a;
            MUTAG(gid).nl.values = [ni nj]';
            al=cellfun(@(x) find(x),num2cell(a,2),'un',0);
            MUTAG(gid).al = al;
            MUTAG(gid).w = w;

        gid = gid+1;
        id = id+1;
        end
    end
end
    
% a = find(Phi(:,189));

% [i,j] = find(Phi(:,189))

sp_mat_path = '/Users/mukundraj/Desktop/work/projects/kerneldepth/kerneldepth/data/processed/mutag/sp/';

% [K,runtime] = WLspdelta(MUTAG,h,nl,0);

[K, runtime, Phi] = spkernel(MUTAG, 1,0);
size(Phi)

clear maper;
for id=1:448

    maper(id).list = containers.Map();
  
end

for id=1:448
    [inds,vals] = find(Phi(:,id+188));
    
    

    for j=1:length(inds)
           jid = inds(j);
           maper(id).list(num2str(dic(jid).keys))=1;
%            maper(jid).list(num2str(dic(1).keys))=1;
    end
    
end

fileID = fopen('histdetails.txt','w');
for i=1:448
    
   tmp = keys(maper(i).list);
   fprintf(fileID, strcat(strjoin(tmp,' , '),'\n'));
   [inds,i,vals] = find(Phi(:,1));
    
end
fclose(fileID);

fileID = fopen('chem_sp_vectors.txt','w');
for i=1:188
    
   vec = zeros(1,448);
   [inds,dump,vals] = find(Phi(:,i));
   vec(inds) = vals;
   fprintf(fileID, strcat(num2str(vec),'\n'));
    
end
fclose(fileID);

fileID = fopen('sparserep_sp_vectors.txt','w');
for i=1:188
    
   vec = zeros(1,448);
   [inds,dump,vals] = find(Phi(:,i));
   fprintf(fileID, strcat(num2str(inds',' %03.f'),'\n'));
    
end
fclose(fileID);

fileID = fopen('sparserep_sp_vals.txt','w');
for i=1:188
    
   vec = zeros(1,448);
   [inds,dump,vals] = find(Phi(:,i));
   fprintf(fileID, strcat(num2str(vals',' %03.f'),'\n'));
    
end
fclose(fileID);

return

% writing

writepath = strcat(sp_mat_path,'sp','.txt');
csvwrite(writepath,K);