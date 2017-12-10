cnt=0
exp="exp4"
for ((end=300050; end < 310000; end += 100, cnt++))
do
	./turbo 4 400000 300000 $end
	Rscript ttime.R 300000 $end 1> info
	mkdir -p data/$exp/run$cnt
	mv *.msrdat data/$exp/run$cnt
	mv *.png data/$exp/run$cnt
	mv info data/$exp/run$cnt
done
