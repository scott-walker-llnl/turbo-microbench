args = commandArgs(trailingOnly=TRUE)
if (length(args) != 2)
{
	print("bad arguments")
	q()
}
notstart <- as.numeric(args[1])
notend <- as.numeric(args[2])
lastitr <- 400000

fname <- "core0.msrdat"
alldat <- read.csv(fname, header=TRUE, sep="\t")

dp = as.data.frame(alldat)
avgfrq = c()

tscbefore = dp[notstart, 3]
for (n in seq(from=notstart, to=notend, by=1))
{
	avgfrq = c(dp[n, 1], avgfrq)
}
tscafter = dp[notend, 3]

nottime = (tscafter - tscbefore) / (mean(avgfrq) * 1000000000)
print(paste("no turbo time", nottime, "seconds", sep=" "))

time45 = c()
time44 = c()
time43 = c()
time42 = c()
itr = notend
seq = 0
while(itr < lastitr)
{
	#tscbefore = tscafter
	#avgfrq = c()
	#while (itr < lastitr && dp[itr, 1] == 4.5)
	#{
	#	avgfrq = c(dp[itr, 1], avgfrq)
	#	itr = itr + 1
	#}
	#if (itr >= lastitr)
	#{
	#	break
	#}
	#tscafter = dp[itr, 3]
	#ttime5 = (tscafter - tscbefore) / (mean(avgfrq) * 1000000000)
	#time45 = c(time45, ttime5)

	#print(paste(seq, "4.5time", ttime5, "seconds", sep=" "))

	tscbefore = tscafter
	avgfrq = c()
	while (itr < lastitr && dp[itr, 1] == 4.4)
	{
		avgfrq = c(dp[itr, 1], avgfrq)
		itr = itr + 1
	}
	if (itr >= lastitr)
	{
		break
	}
	tscafter = dp[itr, 3]
	ttime4 = (tscafter - tscbefore) / (mean(avgfrq) * 1000000000)
	time44 = c(time44, ttime4)

	print(paste(seq, "4.4time", ttime4, "seconds", sep=" "))

	tscbefore = tscafter
	avgfrq = c()
	while (itr < lastitr && dp[itr, 1] == 4.3)
	{
		avgfrq = c(dp[itr, 1], avgfrq)
		itr = itr + 1
	}
	if (itr >= lastitr)
	{
		break
	}
	tscafter = dp[itr, 3]
	ttime3 = (tscafter - tscbefore) / (mean(avgfrq) * 1000000000)
	time43 = c(time43, ttime3)

	print(paste(seq, "4.3time", ttime3, "seconds", sep=" "))

	tscbefore = tscafter
	avgfrq = c()
	while (itr < lastitr && dp[itr, 1] == 4.2)
	{
		avgfrq = c(dp[itr, 1], avgfrq)
		itr = itr + 1
	}
	if (itr >= lastitr)
	{
		break
	}
	tscafter = dp[itr, 3]
	ttime2 = (tscafter - tscbefore) / (mean(avgfrq) * 1000000000)
	time42 = c(time42, ttime2)

	print(paste(seq, "4.2time", ttime2, "seconds", sep=" "))
	seq = seq + 1
}

#maxyval = max(c(time45, time44, time43, time42))
maxyval = max(c(time44, time43, time42))
#minyval = min(c(time45, time44, time43, time42))
minyval = min(c(time44, time43, time42))
#maxxval = max(c(length(time45), length(time44), length(time43), length(time42)))
maxxval = max(c(length(time44), length(time43), length(time42)))

colors = c(rgb(0, 0, 1), rgb(0, 0.4, 0), rgb(0.5, 0, 0.8), rgb(0, 0, 0))
png("phasetimes.png", width=1024, height=768)
par(mar=c(5, 6, 4, 2))
plot(c(1, maxxval), c(minyval, maxyval), col=rgb(1, 1, 1), xaxs="i", yaxs="i", xlab="iteration", ylab="phase time (seconds)", main=paste("Phase Times with Rest Time", format(round(nottime, 4), nsmall=4), "seconds", sep=" "), cex.axis=2, cex.lab=2, cex.main=2)
#lines(1:length(time45), time45, col=colors[1], lwd=2)
lines(1:length(time44), time44, col=colors[2], lwd=2)
lines(1:length(time43), time43, col=colors[3], lwd=2)
lines(1:length(time42), time42, col=colors[4], lwd=2)
legend("topleft", legend=c("4.5GHz", "4.4GHz", "4.3GHz", "4.2GHz"), col=colors, lwd=2, cex=2)
dev.off()
