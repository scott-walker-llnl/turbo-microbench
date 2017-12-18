cnt=0
exp="exp3"
for ((end=300050; end < 310000; end += 100, cnt++))
do
	path="data/$exp/run$cnt"
	Rscript ttime.R 300000 $end $path 1> info
	mv info $path
	mv fplot.png $path
done
