args = commandArgs(trailingOnly=TRUE)
if (length(args) != 1)
{
	print("bad arguments")
	q()
}
fname <- args[1]
alldat <- read.csv(fname, header=FALSE, sep=" ")

dp = as.data.frame(alldat)

ttime <- c()
ytime <- c()
for (n in seq(from=1, to=(990 / 4), by=5))
{
	ttime = c(ttime, dp[n + 1, 1])
	ytime = c(ytime, dp[n, 1])
}

#fit = lm(ttime~log(ytime))
#summary(fit)
#coef(fit)

png("ttime.png", width=1270, height=768)
plot(ytime, ttime, ylab = "Turbo Time", xlab = "Rest Time", type="points", xaxs="i", yaxs="i", col="blue")
#lines(1:198, 0.16826324 + 0.06688577*log(ytime))
dev.off()
