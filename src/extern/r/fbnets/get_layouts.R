# 
# filepath = "/Users/mukundraj/Desktop/work/projects/kerneldepth/kerneldepth/data/processed/fb-nonstuds/graphmls/0004.gml";
# 
# 
# G = read_graph(filepath, format=c("graphml"))
# lay = layout.auto(G);
# 
# plot(G, layout=lay, vertex.color="green")
# 


ip_folder = "/Users/mukundraj/Desktop/work/projects/kerneldepth/kerneldepth/data/processed/fb-studs/graphmls/";

op_folder = "/Users/mukundraj/Desktop/work/projects/kerneldepth/kerneldepth/data/processed/fb-studs/layouts/";


N = 100;

filenames <- list.files(ip_folder)

for (i in 1:N){
  
  cur_file <- paste(c(ip_folder,filenames[i]), collapse="")
  G = read_graph(cur_file, format=c("graphml"))
  lay = layout.auto(G);
  print(i)

  out_file <- paste(c(op_folder,str_pad(i, 4, pad = "0"),".txt"), collapse="")

  lay <- scale(lay);
  scaling_fac <- max(max(apply(lay,2,max)),abs(min(apply(lay,2,min))));
  lay <- scale(lay,center=FALSE,scale=c(scaling_fac,scaling_fac));
  write.table(lay, file = out_file, sep = "\t", col.names = c('x','y'), qmethod = "double", row.names=FALSE);
    
}


