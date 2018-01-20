#define _BSD_SOURCE
#define _GNU_SOURCE
#include "msr_core.h"
#include "master.h"
#include <unistd.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <sched.h>

#define FNAMESIZE 32
#define MSR_TURBO_RATIO_LIMIT 0x1AD

struct data_sample
{
	uint64_t frq_data;
	uint64_t tsc_data;
};

struct data_sample **thread_samples;
unsigned THREADCOUNT;
unsigned long *SAMPLECTRS;

//int memory_workload(threaddata_t * tdat);

void dump_rapl()
{
	uint64_t rapl;
	read_msr_by_coord(0, 0, 0, MSR_PKG_POWER_LIMIT, &rapl);
	fprintf(stderr, "rapl is %lx\n", (unsigned long) rapl);
}

void set_turbo_limit(unsigned int limit)
{
	uint64_t turbo_limit;
	limit &= 0xFF;
	turbo_limit = 0x0 | (limit) | (limit << 8) | (limit << 16) | (limit << 24);
	//printf("set turbo limit %lx\n", (unsigned long) turbo_limit);
	write_msr_by_coord(0, 0, 0, MSR_TURBO_RATIO_LIMIT, turbo_limit);
}

int sample_data(int core)
{
	//fprintf(stderr, "thread %d sample %lu\n", core, SAMPLECTRS[core]);
	if (core > THREADCOUNT || core < 0)
	{
		return -1;
	}
	uint64_t perf;
	uint64_t tsc;
	read_msr_by_coord(0, core, 0, IA32_PERF_STATUS, &perf);
	read_msr_by_coord(0, core, 0, IA32_TIME_STAMP_COUNTER, &tsc);
	SAMPLECTRS[core]++;
	thread_samples[core][SAMPLECTRS[core]].frq_data = perf;
	thread_samples[core][SAMPLECTRS[core]].tsc_data = tsc;
	//if (core == 0)
	//{
	//	fprintf(stderr, "tsc\t%llu\n", tsc);
	//}
	return 0;
}

void dump_data(FILE **outfile)
{
	int j;
	for (j = 0; j < THREADCOUNT; j++)
	{
		fprintf(outfile[j], "p-state\ttsc\n");
		unsigned long i;
		for (i = 0; i < SAMPLECTRS[j]; i++)
		{
			fprintf(outfile[j], "%f\t%llx\t%llu\n", 
				((thread_samples[j][i].frq_data & 0xFFFFul) >> 8) / 10.0,
				(unsigned long long) (thread_samples[j][i].frq_data & 0xFFFFul),
				(unsigned long long) (thread_samples[j][i].tsc_data));
		}
	}
}

void set_rapl(unsigned sec, double watts, double pu, double su, unsigned affinity)
{
	uint64_t power = (unsigned long) (watts / pu);
	uint64_t seconds;
	uint64_t timeval_y = 0, timeval_x = 0;
	double logremainder = 0;

	timeval_y = (uint64_t) log2(sec / su);
	// store the mantissa of the log2
	logremainder = (double) log2(sec / su) - (double) timeval_y;
	timeval_x = 0;
	// based on the mantissa, we can choose the appropriate multiplier
	if (logremainder > 0.15 && logremainder <= 0.45)
	{
		timeval_x = 1;
	}
	else if (logremainder > 0.45 && logremainder <= 0.7)
	{
		timeval_x = 2;
	}
	else if (logremainder > 0.7)
	{
		timeval_x = 3;
	}
	// store the bits in the Intel specified format
	seconds = (uint64_t) (timeval_y | (timeval_x << 5));
	uint64_t rapl = 0x0 | power | (seconds << 17);

	rapl |= (1LL << 15) | (1LL << 16);
	write_msr_by_coord(0, 0, 0, MSR_PKG_POWER_LIMIT, rapl);
}

int main(int argc, char **argv)
{
	if (argc < 4)
	{
		fprintf(stderr, "ERROR: bad arguments\n");
		fprintf(stderr, "Usage: ./t <threads> <duration in seconds> <samples per second> \n");
		return -1;
	}

	if (init_msr())
	{
		fprintf(stderr, "ERROR: unable to init libmsr\n");
		return -1;
	}

	THREADCOUNT = (unsigned) atoi(argv[1]);
	unsigned duration = (unsigned) atoi(argv[2]);
	unsigned sps = (unsigned) atoi(argv[3]);
	unsigned srate = (1000.0 / sps) * 1000u;

	cpu_set_t cpus;
	CPU_ZERO(&cpus);
	// use the next logical CPU after the number of threads in use
	// AKA this puts the sampler program on an unused logical core
	CPU_SET(THREADCOUNT, &cpus);
	sched_setaffinity(0, sizeof(cpus), &cpus);

	fprintf(stdout, "Using paremeters:\n");
	fprintf(stdout, "\tThreads: %u\n", THREADCOUNT);
	fprintf(stdout, "\tTime: %u\n", duration);
	fprintf(stdout, "\tSamples Per Second: %u\n", sps);

	uint64_t unit;
	read_msr_by_coord(0, 0, 0, MSR_RAPL_POWER_UNIT, &unit);
	uint64_t power_unit = unit & 0xF;
	double pu = 1.0 / (0x1 << power_unit);
	fprintf(stderr, "power unit: %lx\n", power_unit);
	uint64_t seconds_unit = (unit >> 16) & 0x1F;
	double su = 1.0 / (0x1 << seconds_unit);
	fprintf(stderr, "seconds unit: %lx\n", seconds_unit);

	set_rapl(1, 91.0, pu, su, 0);
	dump_rapl();
	set_turbo_limit(0x2D);

	thread_samples = (struct data_sample **) calloc(THREADCOUNT, sizeof(struct data_sample *));
	FILE **output = (FILE **) calloc(THREADCOUNT, sizeof(FILE *));
	SAMPLECTRS = (unsigned long *) calloc(THREADCOUNT, sizeof(unsigned long));
	unsigned long numsamples = duration * sps;
	char fname[FNAMESIZE];
	int i;
	for (i = 0; i < THREADCOUNT; i++)
	{
		fprintf(stdout, "Allocating for %lu samples\n", (numsamples / 4) + 1);
		thread_samples[i] = (struct data_sample *) calloc((numsamples / 4) + 1, sizeof(struct data_sample));
		if (thread_samples[i] == NULL)
		{
			fprintf(stderr, "ERROR: out of memory\n");
			return -1;
		}
		snprintf((char *) fname, FNAMESIZE, "core%d.msrdat", i);
		output[i] = fopen(fname, "w");
	}

	fprintf(stdout, "Initialization complete...\n");
	sleep(0.25);

	double avgrate = 0.0;
	double durctr = 0.0;
	double lasttv = 0.0;
	struct timeval start, current;
	unsigned samplethread = 0;
	fprintf(stdout, "Benchmark begin...\n");
	gettimeofday(&start, NULL);
	gettimeofday(&current, NULL);
	sample_data(0);
	usleep(srate);
	while (durctr < duration)
	{
		sample_data(samplethread);
		usleep(srate);
		// distribute the sampling over the threads
		samplethread = (samplethread + 1) % THREADCOUNT;
		gettimeofday(&current, NULL);
		durctr = ((double) (current.tv_sec - start.tv_sec) + 
			(current.tv_usec - start.tv_usec) / 1000000.0);
		avgrate += durctr - lasttv;
		lasttv = durctr;
	}
	
	fprintf(stdout, "Benchmark complete...\n");
	fprintf(stdout, "Actual run time: %f\n", (float) (current.tv_sec - start.tv_sec) + (current.tv_usec - start.tv_usec) / 1000000.0);
	fprintf(stdout, "Average Sampling Rate: %lf seconds\n", avgrate / SAMPLECTRS[0]);
	fprintf(stdout, "Dumping data file(s)...\n");

	dump_data(output);
	for (i = 0; i < THREADCOUNT; i++)
	{
		fprintf(stdout, "Thread %d collected %lu samples\n", i, SAMPLECTRS[i]);
		free(thread_samples[i]);
	}
	free(thread_samples);
	free(output);
	free(SAMPLECTRS);

	set_rapl(1, 105.0, pu, su, 0);

	finalize_msr();

	fprintf(stdout, "Done...\n");

	return 0;
};

/*int memory_workload(threaddata_t * tdat)
{
	fprintf(stderr, "doing memory workload\n");
	static int j = 0;
	int i = 0;
	j += 32;
	for (i = 0; i < (2 << 14); i++)
	{
		tdat->bufferMem[j] = j + i;
		j = (j + 256) % tdat->buffersizeMem;
	}
	return 0;
}*/
