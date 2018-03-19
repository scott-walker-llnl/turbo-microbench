#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <pthread.h>
#include <sys/time.h>
#include "msr.h"
#include "msr_core.h"

#define FREQ_HIGH_DEFAULT 0x2D
#define FREQ_LOW_DEFAULT 0x2A
#define POWER_LIMIT_DEFAULT 91.0
#define POWER_TDP_DEFAULT 100.0
#define MANUAL_TURBO_DEFAULT 1

typedef float v8f __attribute__((vector_size(32)));
typedef float v4f __attribute__((vector_size(16)));

struct thread_data
{
	int tid;
	size_t matsize;
	size_t subsize;
	struct timeval begin;
	double duration;
	int numthreads;
	float **matrix_a;
	float **matrix_b;
	float **matrix_out;
	float **matrix_m;
	float **matrix_n;
	uint64_t itrcount;
} typedef thread_data;

unsigned FREQ_HIGH = FREQ_HIGH_DEFAULT;
unsigned FREQ_LOW = FREQ_LOW_DEFAULT;
double POWER_LIMIT = POWER_LIMIT_DEFAULT;
double POWER_TDP = POWER_TDP_DEFAULT;
unsigned MANUAL_TURBO = MANUAL_TURBO_DEFAULT;


float **alloc_matrix(size_t matsize);
void free_matrix(float ** matrix, size_t matsize);
void print_matrix(float ** matrix, size_t matsize);
void init_matrix(float ** matrix, size_t matsize);
void swap_matrix(float **matrix1, float **matrix2, thread_data *tdat) __attribute__((noinline));
void matmult(thread_data *tdat) __attribute__((noinline));
void *thread_work(void *arg);

float **alloc_matrix(size_t matsize)
{
	float **matrix = (float **) malloc(matsize * sizeof(float *));
	if (matrix == NULL)
	{
		printf("ERROR: out of memory\n");
	}
	int itr;
	for (itr = 0; itr < matsize; itr++)
	{
		matrix[itr] = (float *) malloc(matsize * sizeof(float));
		if (matrix[itr] == NULL)
		{
			printf("ERROR: out of memory\n");
		}
	}
	return matrix;
}

void free_matrix(float ** matrix, size_t matsize)
{
	int itr;
	for (itr = 0; itr < matsize; itr++)
	{
		if (matrix[itr] != NULL)
		{
			free(matrix[itr]);
		}
	}
	if (matrix != NULL)
	{
		free(matrix);
	}
}

void print_matrix(float ** matrix, size_t matsize)
{
	printf("\n");
	int i, j;
	for (i = 0; i < matsize; i++)
	{
		for (j = 0; j < matsize; j++)
		{
			printf("%4d, ", matrix[i][j]);
		}
		printf("\n");
	}
}

void init_matrix(float ** matrix, size_t matsize)
{
	int i, j;
	for (i = 0; i < matsize; i++)
	{
		for (j = 0; j < matsize; j++)
		{
			matrix[i][j] = rand() % 128;
		}
	}
}

void swap_matrix(float **matrix1, float **matrix2, thread_data *tdat)
{
	int i, j;
	for (i = tdat->tid * tdat->subsize; i < (tdat->tid * tdat->subsize) + tdat->subsize; i++)
	{
		for (j = 0; j < tdat->matsize; j++)
		{
			float temp = matrix1[i][j];
			matrix1[i][j] = matrix2[i][j];
			matrix2[i][j] = temp;
		}
	}
}

void matmult8(thread_data *tdat)
{
	v8f a, b, d, e, f, g, h, x, y, z;
	v8f *c;
	int i, j, k;
	for (i = tdat->tid * tdat->subsize; i < (tdat->tid * tdat->subsize) + tdat->subsize; i += 8)
	{
		for (j = 0; j < tdat->matsize; j += 8)
		{
			for (k = 0; k < tdat->matsize; k += 8)
			{
				__asm__ __volatile__(
					"#begin vec\n\t"
				);
				//tdat->matrix_out[i][j] += tdat->matrix_a[i][k] * tdat->matrix_b[k][j];
				a = *((v8f *) &tdat->matrix_a[i][k]);
				b = *((v8f *) &tdat->matrix_b[k][j]);
				/*
				d = a + b;
				e = b - a;
				f = a * e;
				g = b + f;
				h = a + d * f + g * e;
				x = a + b + d + e + f + g + h;
				y = a * b * d * e * f * g * g;
				z = x + y;
				*/
				c = (v8f *) &tdat->matrix_out[i][j];
				*c += a * b;// + h + (x * y * z);
				__asm__ __volatile__(
					"#end vec\n\t"
				);
			}
		}
	}
}

void matmult4(thread_data *tdat)
{
	v4f a, b, d, e, f, g, h, x, y, z;
	v4f *c;
	int i, j, k;
	for (i = tdat->tid * tdat->subsize; i < (tdat->tid * tdat->subsize) + tdat->subsize; i += 4)
	{
		for (j = 0; j < tdat->matsize; j += 4)
		{
			for (k = 0; k < tdat->matsize; k += 4)
			{
				__asm__ __volatile__(
					"#begin vec\n\t"
				);
				//tdat->matrix_out[i][j] += tdat->matrix_a[i][k] * tdat->matrix_b[k][j];
				a = *((v4f *) &tdat->matrix_a[i][k]);
				b = *((v4f *) &tdat->matrix_b[k][j]);
				/*
				d = a + b;
				e = b - a;
				f = a * e;
				g = b + f;
				h = a + d * f + g * e;
				x = a + b + d + e + f + g + h;
				y = a * b * d * e * f * g * g;
				z = x + y;
				*/
				c = (v4f *) &tdat->matrix_out[i][j];
				*c += a * b;// + h + (x * y * z);
				__asm__ __volatile__(
					"#end vec\n\t"
				);
			}
		}
	}
}


void *thread_work(void *arg)
{
	thread_data *tdat = (thread_data *) arg;
	struct timeval current;
	double elapsed;
	int itr = 0;

	//set_perf(FREQ_HIGH, tdat->numthreads);
	do
	{
		matmult4(tdat);
		gettimeofday(&current, NULL);
		elapsed = (current.tv_sec - tdat->begin.tv_sec) + (current.tv_usec - tdat->begin.tv_usec) / 1000000;
		if (MANUAL_TURBO)
		{
			set_perf(FREQ_LOW, tdat->numthreads);
		}
		//swap_matrix(tdat->matrix_n, tdat->matrix_m, tdat);
		//swap_matrix(tdat->matrix_n, tdat->matrix_m, tdat);
		//swap_matrix(tdat->matrix_n, tdat->matrix_m, tdat);
		//swap_matrix(tdat->matrix_n, tdat->matrix_m, tdat);
		//swap_matrix(tdat->matrix_n, tdat->matrix_m, tdat);
		//swap_matrix(tdat->matrix_n, tdat->matrix_m, tdat);
		//swap_matrix(tdat->matrix_n, tdat->matrix_m, tdat);
		//swap_matrix(tdat->matrix_n, tdat->matrix_m, tdat);
		//swap_matrix(tdat->matrix_n, tdat->matrix_m, tdat);
		//swap_matrix(tdat->matrix_n, tdat->matrix_m, tdat);
		if (MANUAL_TURBO)
		{
			set_perf(FREQ_HIGH, tdat->numthreads);
		}
		itr++;
	} while (elapsed < tdat->duration);
	tdat->itrcount = itr;
}

int main(int argc, char ** argv)
{
	if (argc != 4)
	{
		printf("Error: invalid arguments\n");
		printf("./mm <threads> <matrix size> <duration>\n");
		return -1;
	}

	if (init_msr() != 0)
	{
		printf("Error: libmsr init failed\n");
		return -1;
	}

	int numthreads = atoi(argv[1]) + 1;
	size_t matsize = (size_t) atoi(argv[2]);
	double duration = (double) atoi(argv[3]);

	FILE *params = fopen("fsconfig", "r");
	fscanf(params, "%x\n%x\n%lf\n%lf\n%u", &FREQ_HIGH, &FREQ_LOW, &POWER_LIMIT, &POWER_TDP, &MANUAL_TURBO);
	uint64_t plim;
	uint64_t aperf, mperf;
	read_msr_by_coord(0, 0, 0, IA32_PERF_CTL, &plim);
	read_msr_by_coord(0, 0, 0, MSR_IA32_APERF, &aperf);
	read_msr_by_coord(0, 0, 0, MSR_IA32_MPERF, &mperf);
	fprintf(stdout, 
		"FSCONFIG:\n\tFREQ_HIGH %x\n\tFREQ_LOW %x\n\tPOWER_LIMIT %lf\n\tTDP %lf\n\tMANUAL_TURBO %u\n", 
		FREQ_HIGH, FREQ_LOW, POWER_LIMIT, POWER_TDP, MANUAL_TURBO);


	uint64_t default_turbo = get_turbo_limit();
	enable_turbo(0);
	//set_perf(FREQ_LOW, numthreads);
	//set_turbo_limit(FREQ_HIGH);
	double sec_unit, pow_unit;
	get_rapl_units(&pow_unit, &sec_unit);
	//set_rapl2(32, POWER_LIMIT, pow_unit, sec_unit, 0);
	//set_rapl(1, POWER_TDP, pow_unit, sec_unit, 0);

	struct timeval before, after;
	gettimeofday(&before, NULL);

	pthread_t *threads = (pthread_t *) malloc(numthreads * sizeof(pthread_t));
	if (threads == NULL)
	{
		printf("ERROR: out of memory\n");
		return -1;
	}

	float **matrix_a = alloc_matrix(matsize);
	float **matrix_b = alloc_matrix(matsize);
	float **matrix_out = alloc_matrix(matsize);
	float **matrix_m = alloc_matrix(matsize);
	float **matrix_n = alloc_matrix(matsize);

	init_matrix(matrix_a, matsize);
	init_matrix(matrix_b, matsize);
	init_matrix(matrix_m, matsize);
	init_matrix(matrix_n, matsize);

	thread_data *tdat = (thread_data *) malloc(numthreads * sizeof(thread_data));
	if (tdat == NULL)
	{
		printf("ERROR: out of memory\n");
		return -1;
	}


	//set_perf(0x1B, numthreads);
	int itr;
	for (itr = 0; itr < numthreads; itr++)
	{
		tdat[itr].tid = itr;
		tdat[itr].matsize = matsize;
		tdat[itr].matrix_a = matrix_a;
		tdat[itr].matrix_b = matrix_b;
		tdat[itr].matrix_out = matrix_out;
		tdat[itr].subsize = matsize / numthreads;
		tdat[itr].begin = before;
		tdat[itr].duration = duration;
		tdat[itr].numthreads = numthreads;
		tdat[itr].itrcount = 0;

		tdat[itr].matrix_m = matrix_m;
		tdat[itr].matrix_n = matrix_n;

		pthread_create(&threads[itr], NULL, (void *) &thread_work, (void *) &tdat[itr]);
	}

	FILE *irp = fopen("itreport", "w");
	uint64_t totalitr = 0;
	for (itr = 0; itr < numthreads; itr++)
	{
		pthread_join(threads[itr], NULL);
		fprintf(irp, "Thread %d iterations %lu\n", itr, tdat[itr].itrcount);
		totalitr += tdat[itr].itrcount;
	}
	fprintf(irp, "Total iterations %lu\n", totalitr);

	set_all_turbo_limit(default_turbo);

	free(threads);
	free_matrix(matrix_a, matsize);
	free_matrix(matrix_b, matsize);
	free_matrix(matrix_out, matsize);
	free_matrix(matrix_m, matsize);
	free_matrix(matrix_n, matsize);
	free(tdat);

	finalize_msr();
	
	return 0;
}
