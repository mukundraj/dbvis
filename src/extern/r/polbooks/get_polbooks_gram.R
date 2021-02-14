setwd("/Users/mukundraj/Desktop/work/projects/kerneldepth/kerneldepth/src/extern/r/polbooks");

# input path

ip = "/Users/mukundraj/Desktop/work/projects/kerneldepth/kerneldepth/data/processed/polbooks-graphs/"
ip_labels = "/Users/mukundraj/Desktop/work/projects/kerneldepth/kerneldepth/data/processed/polbooks-graphs-labels/"
opfile = "/Users/mukundraj/Desktop/work/projects/kerneldepth/kerneldepth/data/processed/polbooks-kernels/gram_full_vertexlabelhist.csv"
filenames <- list.files(ip)

filenames_labels <- list.files(ip_labels)

N = length(filenames)


# output path and filename
graphs_list = list()
for (i in 1:N){
  
  cur_file <- paste(c(ip,filenames[i]), collapse="")
  print(cur_file)
  g <- read_graph(cur_file, format = c("edgelist"));
  

  cur_file_labels <- paste(c(ip_labels,filenames_labels[i]), collapse="")
  labels <- readLines(cur_file_labels)
  labels <- as.numeric(labels)
  
  V(g)$label <- labels;
  
  graphs_list[[i]] <- g
}

# K <- CalculateConnectedGraphletKernel(graphs_list, 4)
# K <- CalculateExponentialRandomWalkKernel(graphs_list,0.1)
K <- CalculateVertexHistKernel(graphs_list)



write.table(K, file = opfile, sep = ",", col.names = FALSE, qmethod = "double", row.names=FALSE)

