
x = c(1,2,3,4)
y = c(1,2,3,4)
z = c(1,5,3,5)



fit1 <- ksmooth(x, z, "normal", bandwidth = 0.7)
fit2 <- monoproc(fit1, bandwidth = 0.7)
plot(fit2)

n <- seq(1,4, length=4)

x2 = c(1,2,1,2)
y2 = c(1,1,2,2)
z2 = c(1,2,3,4)


x = (sample.int(101,size=100,replace=TRUE)-1)/100
y = (sample.int(101,size=100,replace=TRUE)-1)/100
z = (sample.int(101,size=10,replace=TRUE)-1)/100

dat <- read.csv("mydata.csv",header=F)


fit <- locfit.raw(cbind(dat$V1, dat$V2), dat$V3, alpha = 0.3, deg = 1, kern = "epan",maxk=100)
plot(fit, type = "image", main = "fit")

fitmono <- monoproc(fit, bandwidth = 1, mono1 = "decreasing", mono2 = "decreasing", dir = "x", gridsize = 50)
plot(fitmono, type = "contour", main = "fitmono")

write.csv(fitmono@fit@x, file="datax.csv")
write.csv(fitmono@fit@y, file="datay.csv")
write.csv(fitmono@fit@z, file="dataz.csv")

#predict(fit, cbind(dat$V1, dat$V2))
#http://stackoverflow.com/questions/29479101/how-to-extract-predictions-from-locfit-and-locfit-robust

