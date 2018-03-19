if [ $# -ne 8 ];
then
	echo "bad arguments"
	echo "./auto.sh <threads> <function> <freq high> <freq low> <plim> <tdp> <manual control> <time>"
	exit
fi
THREADS=$1
FUNCTION=$2
FREQ_HIGH=$3
FREQ_LOW=$4
PLIM=$5
TDP=$6
MANUAL=$7
TIME=$8
rm -f core*.msrdat
rm -f sreport
rm -f avgfrq
rm -f itreport
rm -f instret
echo -e "$FREQ_HIGH\n$FREQ_LOW\n$PLIM\n$TDP\n$MANUAL" > fsconfig
sync 
#sudo ./sampler $THREADS $TIME 500 $TDP $PLIM 1> sreport & sudo ../turbo-fs/FIRESTARTER -q --function $FUNCTION -b 0-$THREADS -t $TIME -l 90 -p 5000
#sudo ./sampler $THREADS $TIME 500 $PLIM $TDP & sudo ../turbo-fs/FIRESTARTER --function $FUNCTION -q -b 0-$THREADS -t $TIME
sudo ./sampler $THREADS $TIME 500 $TDP $PLIM 1> sreport & sudo ./mm $THREADS 1024 $TIME
#sudo ./sampler $THREADS $TIME 500 1> avgfrq & mpirun -np $THREADS ~/Projects/LULESH/lulesh2.0 -i 10
./check.sh  1> frqdist
