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
echo -e "$FREQ_HIGH\n$FREQ_LOW\n$PLIM\n$TDP\n$MANUAL" > fsconfig
./sampler $THREADS $TIME 1000 & ../FIRESTARTER/FIRESTARTER --function $FUNCTION -q -b 0-$THREADS -t $TIME -l 99 -p 1000
./check.sh 
