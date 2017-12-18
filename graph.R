args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 3)
{
	print("bad arguments")
	q()
}
fname <- args[1]
fname2 <- args[2]
fname3 <- args[3]

alldat <- read.csv(fname, header=FALSE, sep=" ")
alldat2 <- read.csv(fname2, header=FALSE, sep=" ")
alldat3 <- read.csv(fname3, header=FALSE, sep=" ")

dp = as.data.frame(alldat)
dp2 = as.data.frame(alldat2)
dp3 = as.data.frame(alldat3)

ttime5 <- c()
ytime5 <- c()
for (n in seq(from=1, to=500, by=5)) #990
{
	total = dp[n + 1, 1] + dp[n + 2, 1] + dp[n + 3, 1]
	#ttime5 = c(ttime5, dp[n + 1, 1])
	ttime5 = c(ttime5, total)
	ytime5 = c(ytime5, dp[n, 1])
}

ttime4 <- c()
ytime4 <- c()
for (n in seq(from=1, to=500, by=5)) #990
{
	total = dp2[n + 2, 1] + dp2[n + 3, 1]
	#ttime4 = c(ttime4, dp2[n + 2, 1])
	ttime4 = c(ttime4, total)
	ytime4 = c(ytime4, dp2[n, 1])
}

ttime3 <- c()
ytime3 <- c()
for (n in seq(from=1, to=500, by=5)) #990
{
	total = dp3[n + 3, 1]
	#ttime3 = c(ttime3, dp3[n + 3, 1])
	ttime3 = c(ttime3, total)
	ytime3 = c(ytime3, dp3[n, 1])
}

#fit = lm(ttime~log(ytime))
#summary(fit)
#coef(fit)

png("ttime.png", width=1270, height=768)
par(mar=c(5, 6, 4, 2))
allydat = c(ttime5, ttime4, ttime3)
allxdat = c(ytime5, ytime4, ytime3)
max.y = max(allydat)
min.y = min(allydat)
max.x = max(allxdat)
min.x = min(allxdat)
plotcolors=c(rgb(0,0,1), rgb(0.5, 0, 0.5), rgb(0,0,0))
plot(c(min.x, max.x), c(min.y, max.y), ylab = "Turbo Time", xlab = "Rest Time", type="points", xaxs="i", yaxs="i", col="white", cex.main=2, cex.axis=2, cex.lab=2, pch=20, main="Total Turbo Time as Function of Rest Time")
points(ytime5, ttime5, col=plotcolors[1], pch=19, lty=2);
points(ytime4, ttime4, col=plotcolors[2], pch=19, lty=2);
points(ytime3, ttime3, col=plotcolors[3], pch=19, lty=2);
legend("topleft", legend=c("4.5GHz Max", "4.4GHz Max", "4.3GHz Max"), pch=19, col=plotcolors, cex=2)
#lines(1:198, 0.16826324 + 0.06688577*log(ytime))
dev.off()
