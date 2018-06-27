import sys
import plotlib as plib
import numpy as np

def rawtodata(raw):
	freq = []
	instret = []
	power = []

	for i in range(2, len(raw) - 1, 1):
		#print(raw[i])
		freq.append(float(raw[i][0]))
		instret.append(float(raw[i+1][7]) - float(raw[i][7]))
		power.append(float(raw[i][3]))
	
	return freq, instret, power

def main():
	if len(sys.argv) != 3:
		print "Error: need 1 plot title as argument"
		return

	title = str(sys.argv[1])
	fname = str(sys.argv[2])
	raw = plib.csvread(fname)

	freq, instret, power = rawtodata(raw)

	flen = len(freq)
	ilen = len(instret)
	plen = len(power)

	xax = np.arange(0, plen, 1)

	print "len xax ", len(xax)
	print "len flen ", flen
	print "len instret ", ilen
	print "len power ", plen

	instret = plib.normscale(instret, min(power), max(power))
	#power = normscale(power)

	#plotover([xax, xax], [power, instret], ["time * 2ms", ""], ["power", "instructions retired"], "Power and Instructions Retired")
	plib.plotndat(xax, [power], "time * 2ms", "power", ["power"], "Power " + title)
	plib.plotndat(xax, [freq], "time *2ms", "frequency", ["frequency"], "Frequency " + title)
	return

main()
