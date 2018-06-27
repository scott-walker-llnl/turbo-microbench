import sys
import plotlib as plib
import numpy as np

def rawtodata(raw):
    cyclen = []
    fsratio = []
    iters = []
    avgfrq = []
    time = []
    ips = []
    power = []
    fsiter = []

    for i in range(2, len(raw) - 1, 5):
        cycavg = 0.0
        fsravg = 0.0
       	itravg = 0.0
        frqavg = 0.0
        timeavg = 0.0
        ipsavg = 0.0
        powavg = 0.0
        fsiavg = 0.0

        for j in range(0, 4, 1):
            cycavg += float(raw[i + j][0])
            fsravg += float(raw[i + j][1])
            itravg += float(raw[i + j][2])
            frqavg += float(raw[i + j][3])
            timeavg += float(raw[i + j][4])
            ipsavg += float(raw[i + j][5])
            powavg += float(raw[i + j][6])
            fsiavg += float(raw[i + j][7])

        lavg = lambda x: x / 5.0
        cycavg = lavg(cycavg)
        fsravg = lavg(fsravg)
       	itravg = lavg(itravg)
        frqavg = lavg(frqavg)
        timeavg = lavg(timeavg)
        ipsavg = lavg(ipsavg)
        powavg = lavg(powavg)
        fsiavg = lavg(fsiavg)

        #print(raw[i])
        cyclen.append(cycavg)
        fsratio.append(fsravg)
        iters.append(itravg)
        avgfrq.append(frqavg)
        time.append(timeavg)
        ips.append(ipsavg)
        power.append(powavg)
        fsiter.append(fsiavg)

    return cyclen, fsratio, iters, avgfrq, time, ips, power, fsiter

def main():
    print "begin plotting"
    if len(sys.argv) != 4:
        print "Error: perfploy.py <title> <file name 1> <file name 2>"
        return

    title = str(sys.argv[1])
    fname = str(sys.argv[2])
    fname2 = str(sys.argv[3])
    print "using title", title
    raw = plib.csvread(fname, ",")
    raw2 = plib.csvread(fname2, ",")

    cyclen, fsratio, iters, avgfrq, time, ips, power, fsiter = rawtodata(raw)
    cyclen2, fsratio2, iters2, avgfrq2, time2, ips2, power2, fsiter2 = rawtodata(raw2)

    throughput_speedup = []

    for i in range(0, len(ips), 1):
        throughput_speedup.append(ips[i] / ips2[i])
        cyclen[i] = cyclen[i] / 1000000;

    # y axis is ips, x axis is cyclen, color is fsratio
    plib.plotcolormap(cyclen, throughput_speedup, fsratio, "cycle length (seconds)", "instructions per second speedup", title, 5.0, "perf.png");

main()
