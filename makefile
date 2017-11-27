INCLUDES_PATH=/home/walker8/libmsr/install/include
LIBS_PATH=/home/walker8/libmsr/install/lib

turbo: turbo.c
	gcc -O3 -pthread -Wall -L${LIBS_PATH} -lmsr -I${INCLUDES_PATH} turbo.c -o turbo

repro: repro.c
	gcc -O3 -pthread -Wall -L${LIBS_PATH} -lmsr -I${INCLUDES_PATH} repro.c -o repro

clean:
	rm -rf turbo
	rm -rf repro
