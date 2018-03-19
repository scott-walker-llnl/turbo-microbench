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
#define MSR_CORE_PERF_LIMIT_REASONS 0x64F

struct data_sample
{
	uint64_t frq_data;
	uint64_t tsc_data;
	uint64_t energy_data;
	uint64_t rapl_throttled;
	uint64_t therm;
	uint64_t perflimit;
	uint64_t instret;
};

struct data_sample **thread_samples;
unsigned THREADCOUNT;
unsigned long *SAMPLECTRS;
double energy_unit;
double avg_sample_rate;

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

void dump_rapl_info(double power_unit)
{
	uint64_t rapl_info;
	read_msr_by_coord(0, 0, 0, MSR_PKG_POWER_INFO, &rapl_info);
	fprintf(stderr, "RAPL INFO:\n\tTDP %lf (raw %lx)\n",
		(rapl_info & 0xEF) * power_unit, rapl_info & 0xEF);
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
	uint64_t energy;
	uint64_t rapl_throttled;
	uint64_t therm;
	uint64_t perflimit;
	uint64_t instret;
	read_msr_by_coord(0, core, 0, IA32_PERF_STATUS, &perf);
	read_msr_by_coord(0, core, 0, IA32_TIME_STAMP_COUNTER, &tsc);
	read_msr_by_coord(0, core, 0, MSR_PKG_ENERGY_STATUS, &energy);
	read_msr_by_coord(0, core, 0, MSR_PKG_PERF_STATUS, &rapl_throttled);
	read_msr_by_coord(0, core, 0, IA32_THERM_STATUS, &therm);
	read_msr_by_coord(0, core, 0, IA32_FIXED_CTR0, &instret);
	//read_msr_by_coord(0, core, 0, MSR_CORE_PERF_LIMIT_REASONS, &perflimit);
	SAMPLECTRS[core]++;
	thread_samples[core][SAMPLECTRS[core]].frq_data = perf;
	thread_samples[core][SAMPLECTRS[core]].tsc_data = tsc;
	thread_samples[core][SAMPLECTRS[core]].energy_data = energy & 0xFFFFFFFF;
	thread_samples[core][SAMPLECTRS[core]].rapl_throttled = rapl_throttled & 0xFFFFFFFF;
	thread_samples[core][SAMPLECTRS[core]].therm = therm;
	thread_samples[core][SAMPLECTRS[core]].perflimit = perflimit;
	thread_samples[core][SAMPLECTRS[core]].instret = instret;
	//if (core == 0)
	//{
	//	fprintf(stderr, "tsc\t%llu\n", tsc);
	//}
	return 0;
}

void set_perf(const unsigned freq, const unsigned tid)
{
	static uint64_t perf_ctl = 0x0ul;
	uint64_t freq_mask = 0x0ul;
	if (perf_ctl == 0x0ul)
	{
		read_msr_by_coord(0, tid, 0, IA32_PERF_CTL, &perf_ctl);
	}
	perf_ctl &= 0xFFFFFFFF7FFF0000ul;
	freq_mask = freq;
	freq_mask <<= 8;
	perf_ctl |= freq_mask;
	write_msr_by_coord(0, tid, 0, IA32_PERF_CTL, perf_ctl);
}

void dump_data(FILE **outfile, double durctr)
{
	unsigned long total_pre = thread_samples[0][2].energy_data;
	unsigned long total_post = 0;
	int powovf = 0;
	int j;
	for (j = 0; j < THREADCOUNT; j++)
	{
		fprintf(outfile[j], "freq\tp-state\ttsc\tpower\trapl-throttle-cycles\tTemp(C)\tCORE_PERF_LIMIT_REASONS\tinstret\n");
		unsigned long i;
		for (i = 0; i < SAMPLECTRS[j]; i++)
		{
			double time = (thread_samples[j][i + 1].tsc_data -
				thread_samples[j][i].tsc_data) / 
				((((thread_samples[j][i].frq_data & 0xFFFFul) >> 8) / 10.0) 
				* 1000000000.0);

			unsigned long diff = (thread_samples[j][i + 1].energy_data -
				thread_samples[j][i].energy_data);


			fprintf(outfile[j], "%f\t%llx\t%llu\t%lf\t%lu\t%u\t%lx\t%lu\n", 
				((thread_samples[j][i].frq_data & 0xFFFFul) >> 8) / 10.0,
				(unsigned long long) (thread_samples[j][i].frq_data & 0xFFFFul),
				(unsigned long long) (thread_samples[j][i].tsc_data),
				diff * energy_unit / time,
				(unsigned long long) (thread_samples[j][i].rapl_throttled),
				80 - ((thread_samples[j][i].therm & 0x7F0000) >> 16),
				(unsigned long) thread_samples[j][i].perflimit,
				thread_samples[j][i + 1].instret - thread_samples[j][i].instret);

			if (thread_samples[j][i + 1].energy_data < thread_samples[j][i].energy_data)
			{
				powovf++;
			}
		}
	}
	total_post = thread_samples[0][SAMPLECTRS[0] - 1].energy_data;

	printf("total power %lf\n", ((total_pre - total_post) * energy_unit) / durctr);
	printf("total power(reverse) %lf\n", ((total_post - total_pre) * energy_unit) / durctr);
	printf("total power ovf %lf\n", (0xFFFFFFFF * powovf - total_pre) / durctr);
	printf("energy ctr overflows: %d\n", powovf);
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

void set_rapl2(unsigned sec, double watts, double pu, double su, unsigned affinity)
{
	uint64_t power = (unsigned long) (watts / pu);
	uint64_t seconds;
	uint64_t timeval_y = 0, timeval_x = 0;
	double logremainder = 0;

	seconds = sec;
	uint64_t rapl = 0x0 | power | (seconds << 17);
	rapl |= (1LL << 15) | (1LL << 16);

	uint64_t oldrapl;
	read_msr_by_coord(0, 0, 0, MSR_PKG_POWER_LIMIT, &oldrapl);

	rapl = (rapl << 32) | (oldrapl & 0x00000000ffffffff);
	write_msr_by_coord(0, 0, 0, MSR_PKG_POWER_LIMIT, rapl);
}

void hwpstuff()
{
	uint64_t hwp = 0x0;
	read_msr_by_coord(0, 0, 0, 0x770, &hwp);
	printf("hwp %lx %s\n", hwp, (hwp == 0 ? "disabled" : "enabled"));
	uint64_t hwp_cap = 0x0;
	read_msr_by_coord(0, 0, 0, 0x771, &hwp_cap);
	printf("hwp cap %lx\n", hwp_cap);
	uint64_t hwp_req = 0x0;
	read_msr_by_coord(0, 0, 0, 0x772, &hwp_req);
	printf("hwp req %lx\n", hwp_req);
	uint64_t hwp_int = 0x0;
	read_msr_by_coord(0, 0, 0, 0x773, &hwp_int);
	printf("hwp int %lx\n", hwp_int);
	uint64_t hwp_log_req = 0x0;
	read_msr_by_coord(0, 0, 0, 0x774, &hwp_log_req);
	printf("hwp log req %lx\n", hwp_log_req);
	uint64_t hwp_stat = 0x0;
	read_msr_by_coord(0, 0, 0, 0x777, &hwp_stat);
	printf("hwp stat %lx\n", hwp_stat);

	hwp_req = 0x14150Alu;
	hwp_log_req = 0x14150Alu;
	//write_msr_by_coord(0, 0, 0, 0x772, hwp_req);
	//write_msr_by_coord(0, 0, 0, 0x772, hwp_req);
	int ctr;
	for (ctr = 0; ctr < 12; ctr++)
	{
		//write_msr_by_coord(0, 1, 0, 0x774, hwp_log_req);
		//write_msr_by_coord(0, 1, 1, 0x774, hwp_log_req);
		//write_msr_by_coord(0, 1, 0, 0x774, hwp_log_req);
		//write_msr_by_coord(0, 1, 1, 0x774, hwp_log_req);
		//set_perf(0x1A, ctr);
	}
	read_msr_by_coord(0, 0, 0, 0x772, &hwp_req);
	printf("hwp req %lx\n", hwp_req);
	read_msr_by_coord(0, 0, 0, 0x774, &hwp_log_req);
	printf("hwp log req %lx\n", hwp_log_req);
	
}

int main(int argc, char **argv)
{
	if (argc < 6)
	{
		fprintf(stderr, "ERROR: bad arguments\n");
		fprintf(stderr, "Usage: ./t <threads> <duration in seconds> <samples per second> <rapl1> <rapl2>\n");
		return -1;
	}

	if (init_msr())
	{
		fprintf(stderr, "ERROR: unable to init libmsr\n");
		return -1;
	}

	THREADCOUNT = (unsigned) atoi(argv[1]) + 1;
	unsigned duration = (unsigned) atoi(argv[2]);
	unsigned sps = (unsigned) atoi(argv[3]);
	unsigned srate = (1000.0 / sps) * 1000u;
	double rapl1 = (double) atoi(argv[4]);
	double rapl2 = (double) atoi(argv[5]);

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

	uint64_t ovf_ctrl;
	int ctr;
	for (ctr = 0; ctr < THREADCOUNT; ctr++)
	{
		uint64_t perf_global_ctrl;
		uint64_t fixed_ctr_ctrl;
		read_msr_by_coord(0, ctr, 0, IA32_PERF_GLOBAL_CTRL, &perf_global_ctrl);
		read_msr_by_coord(0, ctr, 0, IA32_FIXED_CTR_CTRL, &fixed_ctr_ctrl);
		write_msr_by_coord(0, ctr, 0, IA32_PERF_GLOBAL_CTRL, perf_global_ctrl | (0x7ul << 32));
		write_msr_by_coord(0, ctr, 0, IA32_FIXED_CTR_CTRL, fixed_ctr_ctrl | (0x3));
		write_msr_by_coord(0, ctr, 0, IA32_FIXED_CTR0, 0x0ul);
		read_msr_by_coord(0, ctr, 0, IA32_PERF_GLOBAL_OVF_CTRL, &ovf_ctrl);
		write_msr_by_coord(0, ctr, 0, IA32_PERF_GLOBAL_OVF_CTRL, ovf_ctrl & 0xFFFFFFFFFFFFFFFE);
	}

// This is debug info
/*
	uint64_t mod = 0x0;
	read_msr_by_coord(0, 0, 0, 0x19a, &mod);
	printf("mod %lx\n", mod);
	uint64_t therm = 0x0;
	read_msr_by_coord(0, 0, 0, 0x19c, &therm);
	printf("therm %lx\n", therm);
	uint64_t power_ctl = 0x0;
	read_msr_by_coord(0, 0, 0, 0x1FC, &power_ctl);
	printf("powctl %lx\n", power_ctl);
	//power_ctl |= (0x1 << 20);
	//write_msr_by_coord(0, 0, 0, 0x1FC, power_ctl);
*/

	uint64_t unit;
	read_msr_by_coord(0, 0, 0, MSR_RAPL_POWER_UNIT, &unit);
	uint64_t power_unit = unit & 0xF;
	double pu = 1.0 / (0x1 << power_unit);
	fprintf(stderr, "power unit: %lx\n", power_unit);
	uint64_t seconds_unit = (unit >> 16) & 0x1F;
	double su = 1.0 / (0x1 << seconds_unit);
	fprintf(stderr, "seconds unit: %lx\n", seconds_unit);
	unsigned eu = (unit >> 8) & 0x1F;
	energy_unit = 1.0 / (0x1 << eu);
	fprintf(stderr, "energy unit: %lx (%lf)\n", eu, energy_unit);
	dump_rapl_info(pu);

	set_rapl(32, rapl1, pu, su, 0);
	set_rapl2(120, rapl2, pu, su, 0);
	
	//dump_rapl();
	//set_turbo_limit(0x21);

	thread_samples = (struct data_sample **) calloc(THREADCOUNT, sizeof(struct data_sample *));
	FILE **output = (FILE **) calloc(THREADCOUNT, sizeof(FILE *));
	SAMPLECTRS = (unsigned long *) calloc(THREADCOUNT, sizeof(unsigned long));
	unsigned long numsamples = duration * sps;
	char fname[FNAMESIZE];
	int i;
	for (i = 0; i < THREADCOUNT; i++)
	{
		fprintf(stdout, "Allocating for %lu samples\n", (numsamples) + 1);
		thread_samples[i] = (struct data_sample *) calloc((numsamples) + 1, sizeof(struct data_sample));
		if (thread_samples[i] == NULL)
		{
			fprintf(stderr, "ERROR: out of memory\n");
			return -1;
		}
		snprintf((char *) fname, FNAMESIZE, "core%d.msrdat", i);
		output[i] = fopen(fname, "w");
	}

	fprintf(stdout, "Initialization complete...\n");
	//sleep(0.25);
	
	uint64_t inst_before, inst_after;
	read_msr_by_coord(0, 0, 0, IA32_FIXED_CTR0, &inst_before);
	uint64_t aperf_before, mperf_before;
	read_msr_by_coord(0, 0, 0, MSR_IA32_APERF, &aperf_before);
	read_msr_by_coord(0, 0, 0, MSR_IA32_MPERF, &mperf_before);
	uint64_t ovf_stat;
	unsigned ovf_ctr = 0;

	double avgrate = 0.0;
	double durctr = 0.0;
	double lasttv = 0.0;
	struct timeval start, current;
	unsigned samplethread = 0;
	fprintf(stdout, "Benchmark begin...\n");
	gettimeofday(&start, NULL);
	gettimeofday(&current, NULL);
	sample_data(0);
	//sample_data(6);
	usleep(srate);
	while (durctr < duration)
	{
		sample_data(0);
		read_msr_by_coord(0, 0, 0, IA32_PERF_GLOBAL_STATUS, &ovf_stat);
		if (ovf_stat & 0x1)
		{
			read_msr_by_coord(0, 0, 0, IA32_PERF_GLOBAL_OVF_CTRL, &ovf_ctrl);
			write_msr_by_coord(0, 0, 0, IA32_PERF_GLOBAL_OVF_CTRL, ovf_ctrl & 0xFFFFFFFFFFFFFFFE);
			ovf_ctr++;
		}
		//sample_data(6);
		usleep(srate);
		// distribute the sampling over the threads
		//samplethread = (samplethread + 1) % THREADCOUNT;
		gettimeofday(&current, NULL);
		durctr = ((double) (current.tv_sec - start.tv_sec) + 
			(current.tv_usec - start.tv_usec) / 1000000.0);
		avgrate += durctr - lasttv;
		lasttv = durctr;
	}
	
	uint64_t aperf_after, mperf_after;
	read_msr_by_coord(0, 0, 0, MSR_IA32_APERF, &aperf_after);
	read_msr_by_coord(0, 0, 0, MSR_IA32_MPERF, &mperf_after);

	read_msr_by_coord(0, 0, 0, IA32_FIXED_CTR0, &inst_after);
	fprintf(stdout, "Benchmark complete...\n");
	fprintf(stdout, "Actual run time: %f\n", (float) (current.tv_sec - start.tv_sec) + (current.tv_usec - start.tv_usec) / 1000000.0);
	fprintf(stdout, "Average Sampling Rate: %lf seconds\n", avgrate / SAMPLECTRS[0]);
	fprintf(stdout, "Dumping data file(s)...\n");
	fprintf(stdout, "Avg Frq: %f\n", (float) (aperf_after - aperf_before) / (float) (mperf_after - mperf_before) * 2.6);
	fprintf(stdout, "Instructions: %lu (ovf %u)\n", inst_after - inst_before, ovf_ctr);
	FILE *ins = fopen("instret", "w");
	fprintf(ins, "%lu (ovf%u)\n", inst_after - inst_before, ovf_ctr);
	fclose(ins);

	avg_sample_rate = 0.001;
	dump_data(output, durctr);
	for (i = 0; i < 0; i++)
	{
		fprintf(stdout, "Thread %d collected %lu samples\n", i, SAMPLECTRS[i]);
		free(thread_samples[i]);
	}
	free(thread_samples);
	free(output);
	free(SAMPLECTRS);

	//set_rapl(1, 105.0, pu, su, 0);

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
