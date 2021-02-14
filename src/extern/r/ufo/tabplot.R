# 2017-11-18: file to read and plot a table plot for UFO data.


in_path <- '/Users/mukundraj/Desktop/work/projects/kerneldepth/kerneldepth/data/processed/ufo/ufo.csv'
mydata <- read.table(in_path)

mydata <- mydata[1:100,]
tableplot(mydata)


discparcoord(mydata[1:100,],k=100)




