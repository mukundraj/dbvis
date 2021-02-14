# 2017-11-12: file to read and plot a table plot for mushroom data.


in_path <- '/Users/mukundraj/Desktop/work/projects/kerneldepth/kerneldepth/data/processed/mushroom/mushroom.csv'
mydata <- read.table(in_path)

mydata <- mydata[1:100,]
tableplot(mydata)

itabplot(mydata[1:100,])
discparcoord(mydata[1:100,],k=100)



# for the ufo data
