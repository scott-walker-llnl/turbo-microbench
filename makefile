INCLUDES_PATH=/home/walker8/libmsr/install/include
LIBFS_INCLUDES=/home/walker8/libfs
LIBS_PATH=/home/walker8/libmsr/install/lib
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

clean:
	rm -rf turbo
	rm -rf repro
	rm -rf cfb
	rm -rf sampler
