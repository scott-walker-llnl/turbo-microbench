cnt=0
exp="exp3"
for ((end=300050; end < 310000; end += 100, cnt++))
do
	cd data/$exp/run$cnt
	Rscript ../../../rep.R 300000 $end 1> temp
	cd ../../..
done
