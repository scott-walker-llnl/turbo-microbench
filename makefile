INCLUDES_PATH=/home/walker8/lib/include
LIBFS_INCLUDES=/home/walker8/libfs
LIBS_PATH=/home/walker8/lib/lib
LIBFS_PATH=/home/walker8/libfs

turbo: turbo.c
	gcc turbo.c -g -pthread -Wall -L${LIBS_PATH} -lmsr -lm -I${INCLUDES_PATH} -I../FIRESTARTER/ -o turbo

repro: repro.c
	gcc -O3 -pthread -Wall -L${LIBS_PATH} -lmsr -lm -I${INCLUDES_PATH} repro.c -o repro

cfb: cfb.c
	#gcc cfb.c -O0 -pthread -Wall -lm -L${LIBS_PATH} -lmsr -L${LIBFS_PATH} -lfs -I${INCLUDES_PATH} -I${LIBFS_INCLUDES} -o cfb -std=c99
	gcc cfb.c ../libfs/avx_functions.o ../FIRESTARTER/fma_functions.o ../libfs/fma4_functions.o ../libfs/init_functions.o ../libfs/init.o -g -pthread -Wall -lm -L${LIBS_PATH} -lmsr -I${INCLUDES_PATH} -I${LIBFS_INCLUDES} -o cfb -std=c99

sampler: sampler.c
	gcc sampler.c -O3 -lm -L${LIBS_PATH} -lmsr -I${INCLUDES_PATH} -o sampler -std=c99

msr: msr.c
	gcc -c msr.c -O3 -I${INCLUDES_PATH} -o msr.o

matmult: matmult.c msr
	gcc matmult.c msr.o -O3 -mtune=native -pthread -L${LIBS_PATH} -lmsr -lm -I${INCLUDES_PATH} -o mm -std=c99

mmass:
	gcc matmult.c msr.o -O3 -mtune=native -S -pthread -I${INCLUDES_PATH} -std=c99

clean:
	rm -rf turbo
	rm -rf repro
	rm -rf cfb
	rm -rf sampler
	rm -rf msr.o
	rm -rf mm
