import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.lines as mlines
import numpy as np
import matplotlib.pyplot as plt

def plotndat(xdat, ydat, xlab, ylab, leg, title, dotsize=5):
	colors = ["black", "red", "blue", "grey", "green", "orange"]
	font = {'family' : 'normal',
			'weight' : 'normal',
			'size'	 : 14}
	numsets = len(ydat)
	if numsets != len(leg):
		print "Error: number of datasets and legend labels not equal"
		return
	
	for i in range(len(ydat)):
		if len(xdat) != len(ydat[i]):
			print "x1 and y1 lengths differ: ", len(xdat), ",", len(ydat[i])
			return
	
	points = []
	legend_lines = []
	catdat = []
	for i in range(numsets):
		points.append(plt.plot(xdat, ydat[i], markeredgewidth=0.0))
		plt.setp(points[i], color=colors[i], ms=dotsize)
		legend_lines.append(mlines.Line2D([], [], color=colors[i], marker='o', markersize=dotsize))
		catdat += ydat[i]
	
	plt.xlabel(xlab, fontsize=14)
	plt.ylabel(ylab, fontsize=14)
	ax1 = plt.gca()
	plt.xlim([min(xdat), max(xdat)])
	plt.ylim([min(catdat), max(catdat)])
	plt.title(title, fontsize=16)
	lob = plt.legend(legend_lines, leg, loc="upper left", markerscale=2, numpoints=1, fancybox=True)
	lob.get_frame().set_alpha(0.2)
	#plt.axis([min(xdat), max(xdat), 0, max(ydat)], fontsize=20)
	plt.rc('font', **font)
	plt.show()
	return

def plot4dat(xdat, ydat, xlab, ylab, leg, title, dotsize):
	font = {'family' : 'normal',
			'weight' : 'normal',
			'size'	 : 14}
	for i in range(len(ydat)):
		if len(xdat) != len(ydat[i]):
			print "x1 and y1 lengths differ: ", len(xdat), ",", len(ydat[i])
			return
	points1 = plt.plot(xdat, ydat[0], markeredgewidth=0.0)
	points2 = plt.plot(xdat, ydat[1], markeredgewidth=0.0)
	points3 = plt.plot(xdat, ydat[2], markeredgewidth=0.0)
	points4 = plt.plot(xdat, ydat[3], markeredgewidth=0.0)
	plt.setp(points1, color="black", ms=dotsize)
	plt.setp(points2, color="red", ms=dotsize)
	plt.setp(points3, color="blue", ms=dotsize)
	plt.setp(points4, color="grey", ms=dotsize)
	plt.xlabel(xlab, fontsize=14)
	plt.ylabel(ylab, fontsize=14)
	ax1 = plt.gca()
	plt.xlim([min(xdat), max(xdat)])
	plt.ylim([min(ydat[0] + ydat[1] + ydat[2] + ydat[3]), max(ydat[0] + ydat[1] + ydat[2] + ydat[3])])
	plt.title(title, fontsize=16)

	l1 = mlines.Line2D([], [], color="black", marker='o', markersize=dotsize)
	l2 = mlines.Line2D([], [], color="red", marker='o', markersize=dotsize)
	l3 = mlines.Line2D([], [], color="blue", marker='o', markersize=dotsize)
	l4 = mlines.Line2D([], [], color="grey", marker='o', markersize=dotsize)
	l1.set_linestyle("None")
	l2.set_linestyle("None")
	l3.set_linestyle("None")
	l4.set_linestyle("None")

	plt.legend([l1, l2, l3, l4], [leg[0], leg[1], leg[2], leg[3]], loc="upper left", markerscale=2, numpoints=1)
	#plt.axis([min(xdat), max(xdat), 0, max(ydat)], fontsize=20)
	plt.rc('font', **font)
	plt.show()
	return

def plot2d(xdat1, ydat1, xlab1, ylab1, title, dotsize):
	font = {'family' : 'normal',
			'weight' : 'normal',
			'size'	 : 14}
	if len(xdat1) != len(ydat1):
		print "x1 and y1 lengths differ: ", len(xdat1), ",", len(ydat1)
		return
	points1 = plt.plot(xdat1, ydat1, 'o', markeredgewidth=0.0)
	plt.setp(points1, color="black", ms=dotsize)
	plt.xlabel(xlab1, fontsize=14)
	plt.ylabel(ylab1, fontsize=14)
	ax1 = plt.gca()
	plt.xlim([min(xdat1), max(xdat1)])
	plt.ylim([min(ydat1), max(ydat1)])
	plt.title(title, fontsize=16)
	#plt.axis([min(xdat1), max(xdat1), 0, max(ydat1)], fontsize=20)
	plt.rc('font', **font)
	plt.show()
	return

def plotover(xdat, ydat, xlab, ylab, title, dotsize):
	if len(xdat) != len(ydat):
		print "number of datasets differ: ", len(xdat), ",", len(ydat)
		return

	for i in range(len(xdat)):
		if len(xdat[i]) != len(ydat[i]):
			print "x and y lengths differ: [", i, "]", len(xdat1), ",", len(ydat1)
			return

	font = {'family' : 'normal',
			'weight' : 'normal',
			'size'	 : 14}

	fig, ax1 = plt.subplots()

	ax2 = ax1.twinx()
	points1 = ax1.plot(xdat[0], ydat[0], 'o', markeredgewidth=0.0, ms=dotsize)
	points2 = ax2.plot(xdat[1], ydat[1], 'ro', markeredgewidth=0.0, ms=dotsize)

	plt.setp(points1, color="black", ms=dotsize)
	plt.setp(points2, color='r', ms=dotsize)

	ax1.set_xlabel(xlab[0], fontsize=14)
	ax1.set_ylabel(ylab[0], fontsize=14)
	ax2.set_ylabel(ylab[1], fontsize=14)
	
	lab2 = ax2.get_yticklabels()
	for i in range(len(lab2)):
		print "ylab ", lab2[i]
	#ax2.set_yticklabels([min(ydat[1]), max(ydat[1])])

	ax1.set_xlim([min(xdat[0]), max(xdat[0])])
	ax1.set_ylim([0, max(ydat[0])])
	#ax2.set_xlim([min(xdat[1]), max(xdat[1])])
	ax2.set_ylim([0, max(ydat[1])])

	l1 = mlines.Line2D([], [], color="black", marker='o', markersize=dotsize)
	l2 = mlines.Line2D([], [], color="red", marker='o', markersize=dotsize)
	l1.set_linestyle("None")
	l2.set_linestyle("None")

	plt.title(title, fontsize=16)
	plt.legend([l1, l2], [ylab[0], ylab[1]], loc="lower right", markerscale=4, numpoints=1)
	plt.rc('font', **font)
	plt.show()
	return

def plotcolormap(xdat1, ydat1, zdat1, xlab1, ylab1, title, dotsize, fname):
    font = {'family' : 'normal',
            'weight' : 'normal',
            'size'	 : 14}
    if len(xdat1) != len(ydat1):
        print "x1 and y1 lengths differ: ", len(xdat1), ",", len(ydat1)
        return
    # points1 = plt.plot(xdat1, ydat1, 'o', markeredgewidth=0.0)
    for i in range(len(xdat1)):
        co = [1 * zdat1[i], 0, 0];
        plt.scatter(xdat1[i], ydat1[i], marker='o', color=co);
    # plt.setp(points1, color="black", ms=dotsize)
    plt.xlabel(xlab1, fontsize=14)
    plt.ylabel(ylab1, fontsize=14)
    ax1 = plt.gca()
    # ax1.xaxis.get_major_formatter().set_powerlimits((-1, 0))
    plt.xlim([min(xdat1), max(xdat1)])
    plt.ylim([min(ydat1), max(ydat1)])
    plt.title(title, fontsize=16)
    #plt.axis([min(xdat1), max(xdat1), 0, max(ydat1)], fontsize=20)
    plt.rc('font', **font)
    # plt.show()
    plt.savefig(fname, bbox_inches='tight');
    return

def csvread(fname, delim):
	raw = []
	with open(fname, 'rb') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=delim)
		for row in spamreader:
			raw.append(row)
	return raw

def normscale(tonorm, lower, upper):
	tmin = min(tonorm)
	tmax = max(tonorm)

	for i in range(0, len(tonorm), 1):
		tonorm[i] = lower + ((tonorm[i] - tmin) * (upper - lower)) / (tmax - tmin)
	return tonorm

