args = commandArgs(trailingOnly=TRUE)
if (length(args) != 2)
{
	print("bad arguments")
	q()
}
notstart <- as.numeric(args[1])
notend <- as.numeric(args[2])

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

tscbefore = tscafter
itr = notend
avgfrq = c()
while (dp[itr, 1] == 4.5)
{
	avgfrq = c(dp[n, 1], avgfrq)
	itr = itr + 1
}
tscafter = dp[itr, 3]
ttime5 = (tscafter - tscbefore) / (mean(avgfrq) * 1000000000)

print(paste("turbo time 4.5GHz", ttime5, "seconds", sep=" "))

tscbefore = tscafter
avgfrq = c()
while (dp[itr, 1] == 4.4)
{
	avgfrq = c(dp[n, 1], avgfrq)
	itr = itr + 1
}
tscafter = dp[itr, 3]
ttime4 = (tscafter - tscbefore) / (mean(avgfrq) * 1000000000)

print(paste("turbo time 4.4GHz", ttime4, "seconds", sep=" "))

tscbefore = tscafter
avgfrq = c()
while (dp[itr, 1] == 4.3)
{
	avgfrq = c(dp[n, 1], avgfrq)
	itr = itr + 1
}
tscafter = dp[itr, 3]
ttime3 = (tscafter - tscbefore) / (mean(avgfrq) * 1000000000)

print(paste("turbo time 4.3GHz", ttime4, "seconds", sep=" "))

tscbefore = tscafter
avgfrq = c()
while (dp[itr, 1] == 4.2)
{
	avgfrq = c(dp[n, 1], avgfrq)
	itr = itr + 1
}
tscafter = dp[itr, 3]
ttime2 = (tscafter - tscbefore) / (mean(avgfrq) * 1000000000)

print(paste("turbo time 4.2GHz", ttime4, "seconds", sep=" "))

print(paste("end freq" , dp[itr, 1], sep=" "))

#tpavg = c()
#avgfrq = c()
#for (n in seq(from=1, to=400000, by=1))
#{
#	avgfrq = c(dp[n, 1], avgfrq)
#}

#favg = mean(avgfrq)
#print(paste("total freq avg", favg, sep=" "))
#for (n in seq(from = 1, to = 399999, by = 1))
#{
#	tpavg = c(dp[n, 3] / favg, tpavg)
#}

#print(length(tpavg))
#print(length(dp[, 1]))

png("fplot.png", width=1270, height=768)
plot(dp[, 3], dp[, 1], ylab = "freq", xlab = "time", type="line", xaxs="i", yaxs="i", col="blue")
dev.off()
