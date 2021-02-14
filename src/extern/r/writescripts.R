# Script to convert the high dimensional microarray data from R package to csv.

# http://stackoverflow.com/questions/19250104/how-to-convert-r-factor-prediction-into-csv
write.table(alon$y, file = "prediction-1-Decision-Tree-08-Oct-2013.csv", sep = ",", col.names = FALSE, qmethod = "double"
            , row.names=FALSE)

# http://stackoverflow.com/questions/10608526/writing-a-matrix-to-a-file-without-a-header-and-row-numbers
# http://astrostatistics.psu.edu/datasets/2006tutorial/html/base/html/write.table.html
write.table(alon$x, file = "foo.csv", sep = ",", col.names = F, row.names = F,
            qmethod = "double")

