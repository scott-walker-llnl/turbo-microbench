INCLUDES_PATH=/home/walker8/libmsr/install/include
LIBFS_INCLUDES=/home/walker8/libfs
LIBS_PATH=/home/walker8/libmsr/install/lib
LIBFS_PATH=/home/walker8/libfs

turbo: turbo.c
	gcc -O3 -pthread -Wall -L${LIBS_PATH} -lmsr -lm -I${INCLUDES_PATH} turbo.c -o turbo

repro: repro.c
	gcc -O3 -pthread -Wall -L${LIBS_PATH} -lmsr -lm -I${INCLUDES_PATH} repro.c -o repro

cfb: cfb.c
	gcc -g -pthread -Wall -L${LIBS_PATH} -lmsr -L${LIBFS_PATH} -lfs -I${INCLUDES_PATH} -I${LIBFS_INCLUDES} cfb.c -o cfb -std=c99

clean:
	rm -rf turbo
	rm -rf repro
	rm -rf cfb
