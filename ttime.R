args = commandArgs(trailingOnly=TRUE)
if (length(args) != 3)
{
	print("bad arguments")
	q()
}
notstart <- as.numeric(args[1])
notend <- as.numeric(args[2])
path <- args[3]
fstart = 4.5

fname <- paste(path, "core0.msrdat", sep="/")
alldat <- read.csv(fname, header=TRUE, sep="\t")

dp = as.data.frame(alldat)
avgfrq1 = c()

tscbefore = dp[notstart, 3]
for (n in seq(from=notstart, to=notend, by=1))
{
	avgfrq1 = c(dp[n, 1], avgfrq1)
}
tscafter = dp[notend, 3]

nottime = (tscafter - tscbefore) / (mean(avgfrq1) * 1000000000)
print(paste("no turbo time", nottime, "seconds", sep=" "))

itr = notend
check = 4.5
if (fstart >= check)
{
	tscbefore = tscafter
	avgfrq2 = c()
	while (dp[itr, 1] == 4.5)
	{
		avgfrq2 = c(dp[n, 1], avgfrq2)
		itr = itr + 1
	}
	tscafter = dp[itr, 3]
	tempmean = mean(avgfrq2)
	if (is.na(tempmean))
	{
		ttime5 = 0.0
	} else {
		ttime5 = (tscafter - tscbefore) / (tempmean * 1000000000)
	}
	print(paste("turbo time 4.5GHz", sprintf("%.6f", ttime5), "seconds", sep=" "))
}

check = 4.4
if (fstart >= check)
{
	tscbefore = tscafter
	avgfrq3 = c()
	while (dp[itr, 1] == 4.4)
	{
		avgfrq3 = c(dp[n, 1], avgfrq3)
		itr = itr + 1
	}
	tscafter = dp[itr, 3]
	tempmean = mean(avgfrq3)
	if (is.na(tempmean))
	{
		ttime4 = 0.0
	} else {
		ttime4 = (tscafter - tscbefore) / (tempmean * 1000000000)
	}
	print(paste("turbo time 4.4GHz", sprintf("%.6f", ttime4), "seconds", sep=" "))
}

check = 4.3
if (fstart >= check)
{
	tscbefore = tscafter
	avgfrq4 = c()
	while (dp[itr, 1] == 4.3)
	{
		avgfrq4 = c(dp[n, 1], avgfrq4)
		itr = itr + 1
	}
	tscafter = dp[itr, 3]
	tempmean = mean(avgfrq4)
	if (is.na(tempmean))
	{
		ttime3 = 0.0
	} else {
		ttime3 = (tscafter - tscbefore) / (tempmean * 1000000000)
	}
	print(paste("turbo time 4.3GHz", sprintf("%.6f", ttime3), "seconds", sep=" "))
}

tscbefore = tscafter
avgfrq5 = c()
while (dp[itr, 1] == 4.2)
{
	avgfrq5 = c(dp[n, 1], avgfrq5)
	itr = itr + 1
}
tscafter = dp[itr, 3]
tempmean = mean(avgfrq5)
if (is.na(tempmean))
{
	ttime2 = 0.0
} else {
	ttime2 = (tscafter - tscbefore) / (tempmean * 1000000000)
}

print(paste("turbo time 4.2GHz", sprintf("%.6f", ttime2), "seconds", sep=" "))

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
plot(dp[, 3], dp[, 1], ylab = "freq", xlab = "time", type="l", xaxs="i", yaxs="i", col="blue")
dev.off()
