# 2017-11-26: file to read and plot a table plot for breast data.


in_path <- '/Users/mukundraj/Desktop/work/projects/kerneldepth/kerneldepth/results/2017-06-18/breast/setdepth/output_tsvs/set_codes.txt'
mydata <- read.table(in_path)

mydata <- mydata
tableplot(mydata,nBins=85)

itabplot(mydata)
discparcoord(mydata,k=85)

