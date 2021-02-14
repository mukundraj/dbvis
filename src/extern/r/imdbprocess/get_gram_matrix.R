setwd("/Users/mukundraj/Desktop/work/projects/kerneldepth/kerneldepth/src/extern/r/imdbprocess");

# input path

ip = "/Users/mukundraj/Desktop/work/projects/kerneldepth/kerneldepth/data/processed/imdb-graphs/"
opfile = "/Users/mukundraj/Desktop/work/projects/kerneldepth/kerneldepth/data/processed/imdb-kernels/gram100.csv"
filenames <- list.files(ip)

# N = length(filenames)


# output path and filename
graphs_list = list()
for (i in 1:100){
  
  cur_file <- paste(c(ip,filenames[i]), collapse="")
  print(cur_file)
  g <- read_graph(cur_file, format = c("edgelist"));
  
  graphs_list[[i]] <- g

}

K <- CalculateConnectedGraphletKernel(graphs_list, 4)
# K <- CalculateExponentialRandomWalkKernel(graphs_list,0.1)

write.table(K, file = opfile, sep = ",", col.names = FALSE, qmethod = "double", row.names=FALSE)


#https://distill.pub/2016/misread-tsne/
#https://github.com/oreillymedia/t-SNE-tutorial