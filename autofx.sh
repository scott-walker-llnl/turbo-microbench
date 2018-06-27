cyclen=(10000 50000 100000 300000 500000 800000 1000000 1500000 2000000 3000000 5000000) 
# cyclen=(10000000, 50000000) 
# cyclen=(500 1000 10000 50000 100000 300000 500000 800000 1000000 1500000 2000000 3000000 5000000) 
# cyclen=(200000)
ratios=(0.125 0.25 0.375 0.5 0.625 0.75 0.875) 
plimit=70
echo "cyclen, fsratio, iter, avgfrq, time, IPS, pow, fsiter" > manual_freq26_report
for cyc in ${cyclen[@]};
do
	for rat in ${ratios[*]};
	do
		for ((i = 0; i < 5; i++))
		do
			echo $cyc $rat $i
			./fx 0x2D 0x2D $cyc 120 $rat 
			# t=$(grep "Actual run time" sreport)
			t=$(grep "Actual run time: " sreport | sed -r "s/Actual run time: (.*)\$/\1/")
			# ips=$(grep "IPS" sreport)
			ips=$(grep "IPS: " sreport | sed -r "s/IPS: (.*) \(ovf 0\)/\1/")
			# pow=$(grep "total power [0-9]\+" sreport)
			pow=$(grep -m 1 "total power " sreport | sed -r "s/total power (.*)\$/\1/")
			# it=$(grep "total iterations" itreport)
			it=$(grep "total iterations: " itreport | sed -r "s/total iterations: (.*)\$/\1/")
			# avf=$(grep "Avg Frq" sreport)
			avf=$(grep "Avg Frq: " sreport | sed -r "s/Avg Frq: (.*)\$/\1/")
			echo "${cyc}, ${rat}, ${i}, ${avf}, ${t}, ${ips}, ${pow}, ${it}" >> manual_freq26_report
			fname=${cyc}_${rat}_run${i}_phases
			mv profiles data/bigrun/control/${fname}
			sleep 1
		done
	done
done
cp manual_freq26_report data/bigrun/manual_freq26_report
