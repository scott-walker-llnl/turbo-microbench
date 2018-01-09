#define _GNU_SOURCE
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <signal.h>
#include <pthread.h>
#include <unistd.h>
#include <sched.h>
#include <math.h>
#include <sys/time.h>
#include <xmmintrin.h>
#include "msr_core.h"
#include "master.h"

#define MSR_TURBO_RATIO_LIMIT 0x1AD
#define LOAD_HIGH 1
#define INIT_BLOCKSIZE 8192
#define EXIT_SUCCESS 0
#define FREQ_HIGH (0x2D)
#define LOW_FREQ (0x2A)
#define MEMBUFFSIZE (0x1 << 23) // 1 MB

// used by firestarter
typedef struct threaddata_t
{
	char *bufferMem;
	unsigned long long addrMem;
	unsigned long long addrHigh;
	unsigned long long iterations;
	unsigned long long flops;
	unsigned long long bytes;
	unsigned long long buffersizeMem;
	unsigned int alignment;
	unsigned int numthreads;
	volatile char *barrierdata;
} threaddata_t;

struct thread_data
{
	unsigned tid;
	unsigned niter;
	unsigned num_alu;
	unsigned num_nop;
	uint64_t *perf_data;
	uint64_t *tsc_data;
	unsigned long loc;
	threaddata_t *threaddata;
	unsigned num_threads;
	pthread_t *threads;
	char arch;
	char *buf1;
	char *buf2;
};

unsigned int TLIM;
char WORKLOAD; // 0 = fs, 1 = memory

// workloads
void nop_workload(const unsigned iter);
void alu_workload(const unsigned iter);
void fence_workload(const unsigned iter);
void memory_workload(struct thread_data *tdat);

// prototypes
void work(void *data);
void push_data(const uint64_t data, const uint64_t tsc, struct thread_data *tdat);
void dump_data(const FILE *dest, const struct thread_data *tdat);
int perf_sample(struct thread_data *tdat);
void disable_turbo();
void enable_turbo();
void set_perf(const unsigned freq, const unsigned tid);
void load_dummy(const unsigned tid);
void read_turbo_limit();
void set_turbo_limit(unsigned int limit);
void set_rapl(unsigned sec, double watts, double pu, double su, unsigned affinity);
int asm_work_skl_corei_fma_1t(threaddata_t* threaddata) __attribute__((noinline));
int init_skl_corei_fma_1t(threaddata_t* threaddata) __attribute__((noinline));
int asm_work_snb_xeonep_avx_1t(threaddata_t* threaddata) __attribute__((noinline));
int init_snb_xeonep_avx_1t(threaddata_t* threaddata) __attribute__((noinline));
void signal_turbo_change(int signum);
void dump_rapl();

int main(int argc, char **argv)
{
	struct sigaction action;
	memset(&action, 0, sizeof(struct sigaction));
	action.sa_handler = signal_turbo_change;
	sigaction(SIGUSR1, &action, NULL);

	sigset_t sset;
	sigemptyset(&sset);
	sigaddset(&sset, SIGUSR1);
	int hstat = pthread_sigmask(SIG_UNBLOCK, &sset, NULL);
	if (hstat != 0)
	{
		fprintf(stderr, "ERROR: unable to set signal mask\n");
		return -1;
	}
	if (argc != 6)
	{
		fprintf(stderr, "ERROR: run as so './turbo <num threads> <num iterations> <num alu> <num nop> <arch>'\n");
		return -1;
	}
	if (init_msr())
	{
		fprintf(stderr, "ERROR: unable to init libmsr\n");
		return -1;
	}

	unsigned long long LOADVAR = LOAD_HIGH;
	unsigned long long BUFFERSIZEMEM, RAMBUFFERSIZE;	
	unsigned int BUFFERSIZE[3];
	int ALIGNMENT = 64;
	if (argv[5][0] == 's')
	{
		fprintf(stdout, "using arch: sandy bridge\n");
        BUFFERSIZE[0] = 32768;
        BUFFERSIZE[1] = 262144;
        BUFFERSIZE[2] = 2621440;
        RAMBUFFERSIZE = 104857600;
	}
	else if (argv[5][0] == 'l')
	{
		fprintf(stdout, "using arch: sky lake\n");
		BUFFERSIZE[0] = 32768;
		BUFFERSIZE[1] = 262144;
		BUFFERSIZE[2] = 1572864;
		RAMBUFFERSIZE = 104857600;
	}
	else
	{
		fprintf(stderr, "bad arch: %c\n", argv[5][0]);
		return -1;
	}
	BUFFERSIZEMEM = sizeof(char) * (2 * BUFFERSIZE[0] + BUFFERSIZE[1] + BUFFERSIZE[2] + RAMBUFFERSIZE + ALIGNMENT + 2 * sizeof(unsigned long long));

	dump_rapl();

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

	unsigned num_threads = (unsigned) atoi(argv[1]);
	pthread_t *threads = (pthread_t *) calloc(num_threads, sizeof(pthread_t));

	FILE **dump = (FILE **) malloc(num_threads * sizeof(FILE *));

	unsigned num_iterations = (unsigned) atoi(argv[2]);
	unsigned num_alu = (unsigned) atoi(argv[3]);
	unsigned num_nop = (unsigned) atoi(argv[4]);

	read_turbo_limit();
	set_turbo_limit(FREQ_HIGH);

	struct thread_data *tdat = (struct thread_data *) calloc(num_threads, sizeof(struct thread_data));
	if (tdat == NULL)
	{
		fprintf(stderr, "ERROR: not enough memory\n");
		return -1;
	}

	struct threaddata_t *threaddata = (struct threaddata_t *) calloc(num_threads, sizeof(struct threaddata_t));

	char *barrierdat = (char *) calloc(num_threads, sizeof(char));

	unsigned itr;
	for (itr = 0; itr < num_threads; itr++)
	{
		tdat[itr].arch = argv[5][0];
		tdat[itr].buf1 = (char *) malloc(MEMBUFFSIZE * sizeof(char));
		tdat[itr].buf2 = (char *) malloc(MEMBUFFSIZE * sizeof(char));
		threaddata[itr].alignment = ALIGNMENT;
		threaddata[itr].buffersizeMem = BUFFERSIZEMEM;
		threaddata[itr].bufferMem = _mm_malloc(threaddata[itr].buffersizeMem, threaddata[itr].alignment);
		if (threaddata[itr].bufferMem == NULL)
		{
			fprintf(stderr, "ERROR: not enough memory\n");
			return -1;
		}
		threaddata[itr].addrMem = (unsigned long long) threaddata[itr].bufferMem;
		threaddata[itr].addrHigh = (unsigned long long) (&LOADVAR);
		threaddata[itr].numthreads = num_threads;
		threaddata[itr].barrierdata = barrierdat;
		if (tdat[itr].arch == 's')
		{
			init_snb_xeonep_avx_1t(&threaddata[itr]);
		}
		else if (tdat[itr].arch == 'l')
		{
			init_skl_corei_fma_1t(&threaddata[itr]);
		}
		else
		{
			fprintf(stderr, "bad arch\n");
			return -1;
		}
		tdat[itr].threaddata = &threaddata[itr];
		char fname[128];
		snprintf(fname, 128, "core%u.msrdat", itr);
		dump[itr] = fopen(fname, "w");

		tdat[itr].perf_data = (uint64_t *) malloc(num_iterations * 2 * sizeof(uint64_t));
		if (tdat[itr].perf_data == NULL)
		{
			fprintf(stderr, "ERROR: not enough memory\n");
			return -1;
		}
		tdat[itr].tsc_data = (uint64_t *) malloc(num_iterations * 2 * sizeof(uint64_t));
		if (tdat[itr].tsc_data == NULL)
		{
			fprintf(stderr, "ERROR: not enough memory\n");
			return -1;
		}
		tdat[itr].tid = itr;
		tdat[itr].niter = num_iterations;
		tdat[itr].num_alu = num_alu;
		tdat[itr].num_nop = num_nop;
		tdat[itr].num_threads = num_threads;
		tdat[itr].threads = threads;
		pthread_create(&threads[itr], NULL, (void *) &work, (void *) &tdat[itr]);
	}
	for (itr = 0; itr < num_threads; itr++)
	{
		pthread_join(threads[itr], NULL);
	}
	// TODO: free msr data
	for (itr = 0; itr < num_threads; itr++)
	{
		dump_data(dump[itr], &tdat[itr]);
		fclose(dump[itr]);
		free(tdat[itr].perf_data);
		free(tdat[itr].tsc_data);
		free(threaddata[itr].bufferMem);
	}
	free(barrierdat);
	finalize_msr();
	return 0;
}

void signal_turbo_change(int signum)
{
	if (signum != SIGUSR1)
	{
		fprintf(stderr, "child received wrong signal!\n");
		return;
	}
	set_turbo_limit(TLIM);
	return;
}

void read_turbo_limit()
{
	uint64_t turbo_limit;
	read_msr_by_coord(0, 0, 0, MSR_TURBO_RATIO_LIMIT, &turbo_limit);

	printf("1 core: %x\n", (unsigned) (turbo_limit & 0xFF));
	printf("2 core: %x\n", (unsigned) ((turbo_limit >> 8) & 0xFF));
	printf("3 core: %x\n", (unsigned) ((turbo_limit >> 8) & 0xFF));
	printf("4 core: %x\n", (unsigned) ((turbo_limit >> 8) & 0xFF));
}

void set_turbo_limit(unsigned int limit)
{
	uint64_t turbo_limit;
	limit &= 0xFF;
	turbo_limit = 0x0 | (limit) | (limit << 8) | (limit << 16) | (limit << 24);
	//printf("set turbo limit %lx\n", (unsigned long) turbo_limit);
	write_msr_by_coord(0, 0, 0, MSR_TURBO_RATIO_LIMIT, turbo_limit);
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

void push_data(const uint64_t data, const uint64_t tsc, struct thread_data *tdat)
{
	tdat->perf_data[tdat->loc] = data;
	tdat->tsc_data[tdat->loc] = tsc;
	tdat->loc++;
}

void dump_data(const FILE *dest, const struct thread_data *tdat)
{
	int itr;
	for (itr = 0; itr < tdat->loc; itr++)
	{
		fprintf((FILE * __restrict__) dest, "%f\t%lx\t%lu\n", ((float) ((tdat->perf_data[itr] & (0xFFFF)) >> 8)) / 10.0, (unsigned long) (tdat->perf_data[itr]), tdat->tsc_data[itr] - tdat->tsc_data[0]);
	}
}

void set_perf(const unsigned freq, const unsigned tid)
{
	static uint64_t perf_ctl = 0x0ul;
	uint64_t freq_mask = 0x0ul;
	if (perf_ctl == 0x0ul)
	{
		read_msr_by_coord(0, tid, 0, IA32_PERF_CTL, &perf_ctl);
	}
	perf_ctl &= 0xFFFFFFFFFFFF0000ul;
	freq_mask = freq;
	freq_mask <<= 8;
	perf_ctl |= freq_mask;
	write_msr_by_coord(0, tid, 0, IA32_PERF_CTL, perf_ctl);
}

void load_dummy(const unsigned tid)
{
	static uint64_t perf_dummy = 0x0ul;
	if (perf_dummy == 0x0ul)
	{
		read_msr_by_coord(0, tid, 0, IA32_PERF_CTL, &perf_dummy);
	}
	read_msr_by_coord(0, tid, 0, IA32_PERF_CTL, &perf_dummy);
}

void enable_turbo(const unsigned tid)
{
	uint64_t perf_ctl;
	read_msr_by_coord(0, tid, 0, IA32_PERF_CTL, &perf_ctl);
	perf_ctl &= 0xFFFFFFFEFFFFFFFFul;
	write_msr_by_coord(0, tid, 0, IA32_PERF_CTL, perf_ctl);
}

void disable_turbo(const unsigned tid)
{
	uint64_t perf_ctl;
	read_msr_by_coord(0, tid, 0, IA32_PERF_CTL, &perf_ctl);
	perf_ctl |= 0x0000000100000000ul;
	write_msr_by_coord(0, tid, 0, IA32_PERF_CTL, perf_ctl);
}

inline int perf_sample(struct thread_data *tdat)
{
	uint64_t perf_status;
	read_msr_by_coord(0, tdat->tid, 0, IA32_PERF_STATUS, &perf_status);
	uint64_t tsc;
	read_msr_by_coord(0, tdat->tid, 0, 0x10, &tsc);
	push_data(perf_status, tsc, tdat);
	if ((perf_status & 0xFFFF) == 0x2A00)
	{
		return 1;
	}
	return 0;
}

inline void disable_rapl()
{
	uint64_t rapl_disable = 0x0ul;
	read_msr_by_coord(0, 0, 0, MSR_PKG_POWER_LIMIT, &rapl_disable);
	rapl_disable &= 0xFFFFFFFFFFFE7FFFul;
	rapl_disable |= 0x1FFFul;
	write_msr_by_coord(0, 0, 0, MSR_PKG_POWER_LIMIT, rapl_disable);
}

void dump_rapl()
{
	uint64_t rapl;
	read_msr_by_coord(0, 0, 0, MSR_PKG_POWER_LIMIT, &rapl);
	fprintf(stderr, "rapl is %lx\n", (unsigned long) rapl);
}

void barrier(unsigned affinity, threaddata_t *threaddata)
{
    int sibling = -1;

    volatile int itr;
    for (itr = 1; itr <= (int) ceil(log(threaddata->numthreads)); itr++)
    {
        while (threaddata->barrierdata[affinity] != 0);
        threaddata->barrierdata[affinity] = itr;
        sibling = (affinity + (int) (2 << (itr - 1))) % (int) threaddata->numthreads;
        while (threaddata->barrierdata[sibling] != itr);
        threaddata->barrierdata[sibling] = 0;
    }
}

void send_sigusr(unsigned numthreads, pthread_t *threads)
{
	int i;
	for (i = 0; i < numthreads; i++)
	{
		if (threads[i] != 0)
		{
			pthread_kill(threads[i], SIGUSR1);
		}
	}
}

void work(void *data)
{
	struct thread_data *tdat = (struct thread_data *) data;
	cpu_set_t cpus;
	CPU_ZERO(&cpus);
	CPU_SET(tdat->tid, &cpus);
	sched_setaffinity(0, sizeof(cpus), &cpus);
	sleep(0.25);

	uint64_t power_pre;
	struct timeval time1;
	if (tdat->tid == 0)
	{
		read_msr_by_coord(0, 0, 0, MSR_PKG_ENERGY_STATUS, &power_pre);
		power_pre &= 0xFFFFFFFFFFFFul;
		gettimeofday(&time1, NULL);
	}
	//barrier(tdat->tid, tdat->threaddata);
	
	int phase_end = 0;
	int phase_start = -1;
	int has_signaled = 0;
	int i;
	for (i = 0; i < tdat->niter; i++)
	{
		//nop_workload(tdat->num_nop);
		if (i == tdat->num_alu && tdat->tid == 0)
		{
			TLIM = LOW_FREQ;
			WORKLOAD = 1;
			//set_turbo_limit(TLIM);
			send_sigusr(tdat->num_threads, tdat->threads);
		}
		if (i == tdat->num_nop && tdat->tid == 0)
		{
			TLIM = FREQ_HIGH;
			//set_turbo_limit(TLIM);
			WORKLOAD = 0;
			send_sigusr(tdat->num_threads, tdat->threads);
			has_signaled =  1;
			phase_end = 1;
		}
		if (WORKLOAD == 0)
		{
			if (tdat->arch == 'l')
			{
				asm_work_skl_corei_fma_1t(tdat->threaddata);
			}
			else if (tdat->arch == 's')
			{
				asm_work_snb_xeonep_avx_1t(tdat->threaddata);
			}
			else
			{
				fprintf(stderr, "bad arch\n");
			}
		}
		else
		{
			memory_workload(tdat);
		}
		int ret = perf_sample(tdat);
		if (phase_end == 1 && tdat->tid == 0 && ret == 1)
		{
			TLIM = LOW_FREQ;
			WORKLOAD = 1;
			send_sigusr(tdat->num_threads, tdat->threads);
			phase_start = i;
			phase_end = 0;
		}
		if (tdat->tid == 0 && has_signaled &&
			i == phase_start + (tdat->num_nop - tdat->num_alu))
		{
			TLIM = FREQ_HIGH;
			WORKLOAD = 0;
			send_sigusr(tdat->num_threads, tdat->threads);
			phase_end = 1;
		}
	}
	uint64_t power_post;
	struct timeval time2;
	if (tdat->tid == 0)
	{
		read_msr_by_coord(0, 0, 0, MSR_PKG_ENERGY_STATUS, &power_post);
		power_post &= 0xFFFFFFFFFFFFul;
		uint64_t unit;
		read_msr_by_coord(0, 0, 0, MSR_RAPL_POWER_UNIT, &unit);
		double energy_unit = 1.0 / (0x1 << ((unit & 0x1F00) >> 8));
		gettimeofday(&time2, NULL);
		double time = time2.tv_sec - time1.tv_sec + (time2.tv_usec - time1.tv_usec) / 1000000;
		if (power_post < power_pre)
		{
			fprintf(stderr, "power consumed: %lf\n", (double) (((0xFFFFFFFFFFFFul - power_pre) + power_post) * energy_unit) / time);
		}
		else
		{
			fprintf(stderr, "power consumed: %lf\n", (double) ((power_post - power_pre) * energy_unit) / time);
		}
	}

}

void memory_workload(struct thread_data *tdat)
{
	static int j = 0;
	int i = 0;
	j += 32;
	for (i = 0; i < (2 << 14); i++)
	{
		tdat->buf1[j] = j + i;
		tdat->buf2[j] += tdat->buf1[j] + i;
		j = (j + 1024) % MEMBUFFSIZE;
	}
}

void fence_workload(const unsigned iter)
{
	int i;
	for (i = 0; i < iter; i++)
	{
		__asm__ __volatile__(
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
			"lfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\tlfence\n\t"
		);
	}
}

void alu_workload(const unsigned iter)
{
	int a, b, c;
	int i;
	for (i = 0; i < iter; i++)
	{
		__asm__ __volatile__(
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			"addq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rdx, %%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\t addq %%rdx,%%rax\n\taddq %%rax, %%rbx\n\taddq %%rbx, %%rcx\n\taddq %%rbx, %%rcx\n\t"
			: "=a" (a), "=b" (b), "=c" (c)
			: "d" (i)
			:
		);
	}
}

void nop_workload(const unsigned iter)
{
	int i;
	for (i = 0; i < iter; i++)
	{
		__asm__ __volatile__ (
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
			"nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"
		);
	}
}

int init_skl_corei_fma_1t(threaddata_t* threaddata)
{
    unsigned long long addrMem = threaddata->addrMem;
    int i;

    for (i = 0; i < INIT_BLOCKSIZE; i+=8) *((double*)(addrMem+i)) = 0.25 + (double)i * 0.27948995982e-4    ;
    for (i = INIT_BLOCKSIZE; i <= 106725376 - INIT_BLOCKSIZE; i+= INIT_BLOCKSIZE) memcpy((void*)(addrMem+i),(void*)(addrMem+i-INIT_BLOCKSIZE),INIT_BLOCKSIZE);
    for (; i <= 106725376-8; i+=8) *((double*)(addrMem+i)) = 0.25 + (double)i * 0.27948995982e-4;

    threaddata->flops=21200;
    threaddata->bytes=1920;

    return EXIT_SUCCESS;
}

int asm_work_skl_corei_fma_1t(threaddata_t* threaddata)
{
    if (*((unsigned long long*)threaddata->addrHigh) == 0) return EXIT_SUCCESS;
	threaddata->iterations = 0;
        /* input: 
         *   - threaddata->addrMem    -> rax
         *   - threaddata->addrHigh   -> rbx
         *   - threaddata->iterations -> rcx
         * output: 
         *   - rax -> threaddata->iterations
         * register usage:
         *   - rax:         stores original pointer to buffer, used to periodically reset other pointers
         *   - rbx:         pointer to L1 buffer
         *   - rcx:         pointer to L2 buffer
         *   - r8:          pointer to L3 buffer
         *   - r9:          pointer to RAM buffer
         *   - r10:         counter for L2-pointer reset
         *   - r11:         counter for L3-pointer reset
         *   - r12:         counter for RAM-pointer reset
         *   - r13:         register for temporary results
         *   - r14:         stores cacheline width as increment for buffer addresses
         *   - r15:         stores address of shared variable that controls load level
         *   - mm0:         stores iteration counter
         *   - rdi,rsi,rdx: registers for shift operations
         *   - xmm*,ymm*:   data registers for SIMD instructions
         */
        __asm__ __volatile__(
        "mov %%rax, %%rax;" // store start address of buffer in rax
        "mov %%rbx, %%r15;" // store address of shared variable that controls load level in r15
        "movq %%rcx, %%mm0;" // store iteration counter in mm0
        "mov $64, %%r14;" // increment after each cache/memory access
        //Initialize registers for shift operations
        "mov $0xAAAAAAAA, %%edi;"
        "mov $0xAAAAAAAA, %%esi;"
        "mov $0xAAAAAAAA, %%edx;"
        //Initialize AVX-Registers for FMA Operations
        "vmovapd (%%rax), %%ymm0;"
        "vmovapd (%%rax), %%ymm1;"
        "vmovapd 320(%%rax), %%ymm2;"
        "vmovapd 352(%%rax), %%ymm3;"
        "vmovapd 384(%%rax), %%ymm4;"
        "vmovapd 416(%%rax), %%ymm5;"
        "vmovapd 448(%%rax), %%ymm6;"
        "vmovapd 480(%%rax), %%ymm7;"
        "vmovapd 512(%%rax), %%ymm8;"
        "vmovapd 544(%%rax), %%ymm9;"
        "vmovapd 576(%%rax), %%ymm10;"
        "vmovapd 608(%%rax), %%ymm11;"
        "vmovapd 640(%%rax), %%ymm12;"
        "vmovapd 672(%%rax), %%ymm13;"
        "mov %%rax, %%rbx;" // address for L1-buffer
        "mov %%rax, %%rcx;"
        "add $32768, %%rcx;" // address for L2-buffer
        "mov %%rax, %%r8;"
        "add $262144, %%r8;" // address for L3-buffer
        "mov %%rax, %%r9;"
        "add $1572864, %%r9;" // address for RAM-buffer
        "movabs $18, %%r10;" // reset-counter for L2-buffer with 180 cache lines accessed per loop (202.5 KB)
        "movabs $393, %%r11;" // reset-counter for L3-buffer with 50 cache lines accessed per loop (1228.13 KB)
        "movabs $54613, %%r12;" // reset-counter for RAM-buffer with 30 cache lines accessed per loop (102399.38 KB)

        ".align 64;"     // alignment in bytes 
        "_work_loop_skl_corei_fma_1t:"
////////////////////////////////////////////////////////////////////////////////////
        // decode 0                                 decode 1                                 decode 2             decode 3 
        "vfmadd231pd %%ymm4, %%ymm0, %%ymm3;      vfmadd231pd 64(%%r9), %%ymm1, %%ymm15;   shl $1, %%edi;       add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm11;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm12;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm9, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm9;   shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm13;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm11;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vmovapd %%ymm6, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm12;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm2, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm13;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm11;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vmovapd %%ymm8, 96(%%r8);                vfmadd231pd 64(%%r8), %%ymm0, %%ymm8;    shr $1, %%edx;       add %%r14, %%r8;    " // L3 load, L3 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm12;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm4, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm13;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm11;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vmovapd %%ymm10, 96(%%rcx);              vfmadd231pd 64(%%rcx), %%ymm0, %%ymm10;  shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm12;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm6, 96(%%r8);                vfmadd231pd 64(%%r8), %%ymm0, %%ymm6;    shr $1, %%edi;       add %%r14, %%r8;    " // L3 load, L3 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm13;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm11;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vmovapd %%ymm3, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm12;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm8, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm8;   shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm13;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm11;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd 64(%%r9), %%ymm1, %%ymm15;   shl $1, %%edx;       add %%r14, %%r9;    " // RAM load
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm12;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm10, 96(%%rcx);              vfmadd231pd 64(%%rcx), %%ymm0, %%ymm10;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm13;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm11;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm7, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm7;   shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm12;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm3, 96(%%r8);                vfmadd231pd 64(%%r8), %%ymm0, %%ymm3;    shl $1, %%edi;       add %%r14, %%r8;    " // L3 load, L3 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm13;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm11;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm9, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm9;   shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm12;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm13;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vmovapd %%ymm6, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm11;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm2, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm12;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm13;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vmovapd %%ymm8, 96(%%r8);                vfmadd231pd 64(%%r8), %%ymm0, %%ymm8;    shr $1, %%edx;       add %%r14, %%r8;    " // L3 load, L3 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm11;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm4, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm12;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm13;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vmovapd %%ymm10, 96(%%rcx);              vfmadd231pd 64(%%rcx), %%ymm0, %%ymm10;  shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm11;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm6;      vfmadd231pd 64(%%r9), %%ymm1, %%ymm15;   shr $1, %%edi;       add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm12;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm13;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vmovapd %%ymm3, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm11;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm8, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm8;   shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm12;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm13;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vmovapd %%ymm5, 96(%%r8);                vfmadd231pd 64(%%r8), %%ymm0, %%ymm5;    shl $1, %%edx;       add %%r14, %%r8;    " // L3 load, L3 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm11;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm10, 96(%%rcx);              vfmadd231pd 64(%%rcx), %%ymm0, %%ymm10;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm12;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm13;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm7, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm7;   shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm11;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm4, %%ymm0, %%ymm3;      vfmadd231pd 64(%%r9), %%ymm1, %%ymm15;   shl $1, %%edi;       add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm12;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm13;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm9, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm9;   shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm11;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm12;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vmovapd %%ymm6, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm13;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm2, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm11;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm12;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vmovapd %%ymm8, 96(%%r8);                vfmadd231pd 64(%%r8), %%ymm0, %%ymm8;    shr $1, %%edx;       add %%r14, %%r8;    " // L3 load, L3 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm13;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm4, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm11;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm12;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vmovapd %%ymm10, 96(%%rcx);              vfmadd231pd 64(%%rcx), %%ymm0, %%ymm10;  shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm13;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm6, 96(%%r8);                vfmadd231pd 64(%%r8), %%ymm0, %%ymm6;    shr $1, %%edi;       add %%r14, %%r8;    " // L3 load, L3 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm11;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm12;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vmovapd %%ymm3, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm13;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm8, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm8;   shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm11;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm12;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd 64(%%r9), %%ymm1, %%ymm15;   shl $1, %%edx;       add %%r14, %%r9;    " // RAM load
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm13;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm10, 96(%%rcx);              vfmadd231pd 64(%%rcx), %%ymm0, %%ymm10;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm11;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm12;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm7, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm7;   shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm13;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm3, 96(%%r8);                vfmadd231pd 64(%%r8), %%ymm0, %%ymm3;    shl $1, %%edi;       add %%r14, %%r8;    " // L3 load, L3 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm11;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm12;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm9, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm9;   shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm13;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm11;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vmovapd %%ymm6, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm12;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm2, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm13;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm5, 32(%%rbx); mov %%rax, %%rbx;   " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm11;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vmovapd %%ymm8, 96(%%r8);                vfmadd231pd 64(%%r8), %%ymm0, %%ymm8;    shr $1, %%edx;       add %%r14, %%r8;    " // L3 load, L3 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm12;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm4, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm13;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm11;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vmovapd %%ymm10, 96(%%rcx);              vfmadd231pd 64(%%rcx), %%ymm0, %%ymm10;  shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm12;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm6;      vfmadd231pd 64(%%r9), %%ymm1, %%ymm15;   shr $1, %%edi;       add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm13;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm11;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vmovapd %%ymm3, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm12;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm8, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm8;   shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm13;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm11;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vmovapd %%ymm5, 96(%%r8);                vfmadd231pd 64(%%r8), %%ymm0, %%ymm5;    shl $1, %%edx;       add %%r14, %%r8;    " // L3 load, L3 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm12;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm10, 96(%%rcx);              vfmadd231pd 64(%%rcx), %%ymm0, %%ymm10;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm13;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm11;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm7, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm7;   shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm12;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm4, %%ymm0, %%ymm3;      vfmadd231pd 64(%%r9), %%ymm1, %%ymm15;   shl $1, %%edi;       add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm13;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm11;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm9, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm9;   shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm12;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm13;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vmovapd %%ymm6, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm11;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm2, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm12;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm13;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vmovapd %%ymm8, 96(%%r8);                vfmadd231pd 64(%%r8), %%ymm0, %%ymm8;    shr $1, %%edx;       add %%r14, %%r8;    " // L3 load, L3 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm11;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm4, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm12;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm13;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vmovapd %%ymm10, 96(%%rcx);              vfmadd231pd 64(%%rcx), %%ymm0, %%ymm10;  shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm11;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm6, 96(%%r8);                vfmadd231pd 64(%%r8), %%ymm0, %%ymm6;    shr $1, %%edi;       add %%r14, %%r8;    " // L3 load, L3 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm12;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm13;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vmovapd %%ymm3, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm11;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm8, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm8;   shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm12;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm13;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd 64(%%r9), %%ymm1, %%ymm15;   shl $1, %%edx;       add %%r14, %%r9;    " // RAM load
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm11;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm10, 96(%%rcx);              vfmadd231pd 64(%%rcx), %%ymm0, %%ymm10;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm12;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm13;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm7, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm7;   shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm11;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm3, 96(%%r8);                vfmadd231pd 64(%%r8), %%ymm0, %%ymm3;    shl $1, %%edi;       add %%r14, %%r8;    " // L3 load, L3 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm12;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm13;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm9, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm9;   shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm11;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm12;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vmovapd %%ymm6, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm13;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm2, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm11;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm12;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vmovapd %%ymm8, 96(%%r8);                vfmadd231pd 64(%%r8), %%ymm0, %%ymm8;    shr $1, %%edx;       add %%r14, %%r8;    " // L3 load, L3 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm13;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm4, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm11;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm12;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vmovapd %%ymm10, 96(%%rcx);              vfmadd231pd 64(%%rcx), %%ymm0, %%ymm10;  shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm13;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm6;      vfmadd231pd 64(%%r9), %%ymm1, %%ymm15;   shr $1, %%edi;       add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm11;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm12;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vmovapd %%ymm3, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm13;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm8, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm8;   shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm11;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm12;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vmovapd %%ymm5, 96(%%r8);                vfmadd231pd 64(%%r8), %%ymm0, %%ymm5;    shl $1, %%edx;       add %%r14, %%r8;    " // L3 load, L3 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm13;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm10, 96(%%rcx);              vfmadd231pd 64(%%rcx), %%ymm0, %%ymm10;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm11;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm12;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm7, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm7;   shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm13;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm4, %%ymm0, %%ymm3;      vfmadd231pd 64(%%r9), %%ymm1, %%ymm15;   shl $1, %%edi;       add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm11;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm12;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm9, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm9;   shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm13;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm11;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vmovapd %%ymm6, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm12;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm2, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm13;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm11;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vmovapd %%ymm8, 96(%%r8);                vfmadd231pd 64(%%r8), %%ymm0, %%ymm8;    shr $1, %%edx;       add %%r14, %%r8;    " // L3 load, L3 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm12;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm4, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm13;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm11;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vmovapd %%ymm10, 96(%%rcx);              vfmadd231pd 64(%%rcx), %%ymm0, %%ymm10;  shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm12;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm6, 96(%%r8);                vfmadd231pd 64(%%r8), %%ymm0, %%ymm6;    shr $1, %%edi;       add %%r14, %%r8;    " // L3 load, L3 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm13;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm8, 32(%%rbx); mov %%rax, %%rbx;   " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm11;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vmovapd %%ymm3, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm12;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm8, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm8;   shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm13;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm11;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd 64(%%r9), %%ymm1, %%ymm15;   shl $1, %%edx;       add %%r14, %%r9;    " // RAM load
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm12;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm10, 96(%%rcx);              vfmadd231pd 64(%%rcx), %%ymm0, %%ymm10;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm13;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm11;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm7, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm7;   shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm12;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm3, 96(%%r8);                vfmadd231pd 64(%%r8), %%ymm0, %%ymm3;    shl $1, %%edi;       add %%r14, %%r8;    " // L3 load, L3 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm13;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm11;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm9, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm9;   shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm12;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm13;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vmovapd %%ymm6, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm11;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm2, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm12;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm13;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vmovapd %%ymm8, 96(%%r8);                vfmadd231pd 64(%%r8), %%ymm0, %%ymm8;    shr $1, %%edx;       add %%r14, %%r8;    " // L3 load, L3 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm11;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm4, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm12;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm13;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vmovapd %%ymm10, 96(%%rcx);              vfmadd231pd 64(%%rcx), %%ymm0, %%ymm10;  shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm11;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm6;      vfmadd231pd 64(%%r9), %%ymm1, %%ymm15;   shr $1, %%edi;       add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm12;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm13;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vmovapd %%ymm3, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm11;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm8, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm8;   shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm12;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm13;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vmovapd %%ymm5, 96(%%r8);                vfmadd231pd 64(%%r8), %%ymm0, %%ymm5;    shl $1, %%edx;       add %%r14, %%r8;    " // L3 load, L3 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm11;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm10, 96(%%rcx);              vfmadd231pd 64(%%rcx), %%ymm0, %%ymm10;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm12;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm13;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm7, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm7;   shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm11;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm4, %%ymm0, %%ymm3;      vfmadd231pd 64(%%r9), %%ymm1, %%ymm15;   shl $1, %%edi;       add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm12;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm13;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm9, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm9;   shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm11;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm12;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vmovapd %%ymm6, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm13;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm2, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm11;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm12;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vmovapd %%ymm8, 96(%%r8);                vfmadd231pd 64(%%r8), %%ymm0, %%ymm8;    shr $1, %%edx;       add %%r14, %%r8;    " // L3 load, L3 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm13;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm4, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm11;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm12;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vmovapd %%ymm10, 96(%%rcx);              vfmadd231pd 64(%%rcx), %%ymm0, %%ymm10;  shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm13;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm6, 96(%%r8);                vfmadd231pd 64(%%r8), %%ymm0, %%ymm6;    shr $1, %%edi;       add %%r14, %%r8;    " // L3 load, L3 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm11;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm12;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vmovapd %%ymm3, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm13;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm8, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm8;   shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm11;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm12;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd 64(%%r9), %%ymm1, %%ymm15;   shl $1, %%edx;       add %%r14, %%r9;    " // RAM load
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm13;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm10, 96(%%rcx);              vfmadd231pd 64(%%rcx), %%ymm0, %%ymm10;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm11;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm12;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm7, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm7;   shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm13;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm3, 96(%%r8);                vfmadd231pd 64(%%r8), %%ymm0, %%ymm3;    shl $1, %%edi;       add %%r14, %%r8;    " // L3 load, L3 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm11;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm12;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm9, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm9;   shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm13;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm11;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vmovapd %%ymm6, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm12;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm2, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm13;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm11;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vmovapd %%ymm8, 96(%%r8);                vfmadd231pd 64(%%r8), %%ymm0, %%ymm8;    shr $1, %%edx;       add %%r14, %%r8;    " // L3 load, L3 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm12;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm4, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm13;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm11;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vmovapd %%ymm10, 96(%%rcx);              vfmadd231pd 64(%%rcx), %%ymm0, %%ymm10;  shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm12;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm6;      vfmadd231pd 64(%%r9), %%ymm1, %%ymm15;   shr $1, %%edi;       add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm13;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm11;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vmovapd %%ymm3, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm12;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm8, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm8;   shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm13;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm11;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vmovapd %%ymm5, 96(%%r8);                vfmadd231pd 64(%%r8), %%ymm0, %%ymm5;    shl $1, %%edx;       add %%r14, %%r8;    " // L3 load, L3 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm12;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm9, 32(%%rbx); mov %%rax, %%rbx;   " // 2 L1 loads, L1 store
        "vmovapd %%ymm10, 96(%%rcx);              vfmadd231pd 64(%%rcx), %%ymm0, %%ymm10;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm13;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm11;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm7, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm7;   shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm12;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm4, %%ymm0, %%ymm3;      vfmadd231pd 64(%%r9), %%ymm1, %%ymm15;   shl $1, %%edi;       add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm13;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm11;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm9, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm9;   shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm12;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm13;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vmovapd %%ymm6, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm11;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm2, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm12;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm13;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vmovapd %%ymm8, 96(%%r8);                vfmadd231pd 64(%%r8), %%ymm0, %%ymm8;    shr $1, %%edx;       add %%r14, %%r8;    " // L3 load, L3 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm11;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm4, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm12;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm13;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vmovapd %%ymm10, 96(%%rcx);              vfmadd231pd 64(%%rcx), %%ymm0, %%ymm10;  shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm11;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm6, 96(%%r8);                vfmadd231pd 64(%%r8), %%ymm0, %%ymm6;    shr $1, %%edi;       add %%r14, %%r8;    " // L3 load, L3 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm12;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm13;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vmovapd %%ymm3, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm11;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm8, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm8;   shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm12;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm13;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd 64(%%r9), %%ymm1, %%ymm15;   shl $1, %%edx;       add %%r14, %%r9;    " // RAM load
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm11;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm10, 96(%%rcx);              vfmadd231pd 64(%%rcx), %%ymm0, %%ymm10;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm12;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm13;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm7, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm7;   shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm11;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm3, 96(%%r8);                vfmadd231pd 64(%%r8), %%ymm0, %%ymm3;    shl $1, %%edi;       add %%r14, %%r8;    " // L3 load, L3 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm12;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm13;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm9, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm9;   shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm11;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm12;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vmovapd %%ymm6, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm13;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm2, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm11;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm12;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vmovapd %%ymm8, 96(%%r8);                vfmadd231pd 64(%%r8), %%ymm0, %%ymm8;    shr $1, %%edx;       add %%r14, %%r8;    " // L3 load, L3 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm13;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm4, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm11;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm12;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vmovapd %%ymm10, 96(%%rcx);              vfmadd231pd 64(%%rcx), %%ymm0, %%ymm10;  shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm13;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm6;      vfmadd231pd 64(%%r9), %%ymm1, %%ymm15;   shr $1, %%edi;       add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm11;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm12;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vmovapd %%ymm3, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm13;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm8, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm8;   shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm11;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm12;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vmovapd %%ymm5, 96(%%r8);                vfmadd231pd 64(%%r8), %%ymm0, %%ymm5;    shl $1, %%edx;       add %%r14, %%r8;    " // L3 load, L3 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm13;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm10, 96(%%rcx);              vfmadd231pd 64(%%rcx), %%ymm0, %%ymm10;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm11;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm12;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm7, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm7;   shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm13;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm4, %%ymm0, %%ymm3;      vfmadd231pd 64(%%r9), %%ymm1, %%ymm15;   shl $1, %%edi;       add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm11;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm12;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm9, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm9;   shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm13;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm11;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vmovapd %%ymm6, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm12;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm2, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm13;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm11;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vmovapd %%ymm8, 96(%%r8);                vfmadd231pd 64(%%r8), %%ymm0, %%ymm8;    shr $1, %%edx;       add %%r14, %%r8;    " // L3 load, L3 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm12;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm4, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm13;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm11;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vmovapd %%ymm10, 96(%%rcx);              vfmadd231pd 64(%%rcx), %%ymm0, %%ymm10;  shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm12;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm6, 96(%%r8);                vfmadd231pd 64(%%r8), %%ymm0, %%ymm6;    shr $1, %%edi;       add %%r14, %%r8;    " // L3 load, L3 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm13;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm11;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vmovapd %%ymm3, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm12;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm8, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm8;   shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm13;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm11;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd 64(%%r9), %%ymm1, %%ymm15;   shl $1, %%edx;       add %%r14, %%r9;    " // RAM load
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm12;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm10, 96(%%rcx);              vfmadd231pd 64(%%rcx), %%ymm0, %%ymm10;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm13;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm11;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm7, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm7;   shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm12;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm3, 96(%%r8);                vfmadd231pd 64(%%r8), %%ymm0, %%ymm3;    shl $1, %%edi;       add %%r14, %%r8;    " // L3 load, L3 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm13;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm11;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm9, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm9;   shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm12;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm3, 32(%%rbx); mov %%rax, %%rbx;   " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm13;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vmovapd %%ymm6, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm11;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm2, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm12;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm13;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vmovapd %%ymm8, 96(%%r8);                vfmadd231pd 64(%%r8), %%ymm0, %%ymm8;    shr $1, %%edx;       add %%r14, %%r8;    " // L3 load, L3 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm11;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm4, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm12;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm13;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vmovapd %%ymm10, 96(%%rcx);              vfmadd231pd 64(%%rcx), %%ymm0, %%ymm10;  shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm11;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm6;      vfmadd231pd 64(%%r9), %%ymm1, %%ymm15;   shr $1, %%edi;       add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm12;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm13;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vmovapd %%ymm3, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm11;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm8, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm8;   shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm12;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm13;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vmovapd %%ymm5, 96(%%r8);                vfmadd231pd 64(%%r8), %%ymm0, %%ymm5;    shl $1, %%edx;       add %%r14, %%r8;    " // L3 load, L3 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm11;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm10, 96(%%rcx);              vfmadd231pd 64(%%rcx), %%ymm0, %%ymm10;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm12;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm13;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm7, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm7;   shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm11;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm4, %%ymm0, %%ymm3;      vfmadd231pd 64(%%r9), %%ymm1, %%ymm15;   shl $1, %%edi;       add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm12;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm13;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm9, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm9;   shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm11;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm12;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vmovapd %%ymm6, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm13;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm2, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm11;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm12;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vmovapd %%ymm8, 96(%%r8);                vfmadd231pd 64(%%r8), %%ymm0, %%ymm8;    shr $1, %%edx;       add %%r14, %%r8;    " // L3 load, L3 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm13;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm4, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm11;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm12;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vmovapd %%ymm10, 96(%%rcx);              vfmadd231pd 64(%%rcx), %%ymm0, %%ymm10;  shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm13;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm6, 96(%%r8);                vfmadd231pd 64(%%r8), %%ymm0, %%ymm6;    shr $1, %%edi;       add %%r14, %%r8;    " // L3 load, L3 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm11;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm12;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vmovapd %%ymm3, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm13;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm8, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm8;   shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm11;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm12;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd 64(%%r9), %%ymm1, %%ymm15;   shl $1, %%edx;       add %%r14, %%r9;    " // RAM load
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm13;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm10, 96(%%rcx);              vfmadd231pd 64(%%rcx), %%ymm0, %%ymm10;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm11;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm12;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm7, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm7;   shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm13;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm3, 96(%%r8);                vfmadd231pd 64(%%r8), %%ymm0, %%ymm3;    shl $1, %%edi;       add %%r14, %%r8;    " // L3 load, L3 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm11;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm12;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm9, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm9;   shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm13;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm11;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vmovapd %%ymm6, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm12;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm2, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm13;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm11;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vmovapd %%ymm8, 96(%%r8);                vfmadd231pd 64(%%r8), %%ymm0, %%ymm8;    shr $1, %%edx;       add %%r14, %%r8;    " // L3 load, L3 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm12;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm4, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm13;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm11;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vmovapd %%ymm10, 96(%%rcx);              vfmadd231pd 64(%%rcx), %%ymm0, %%ymm10;  shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm12;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm6;      vfmadd231pd 64(%%r9), %%ymm1, %%ymm15;   shr $1, %%edi;       add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm13;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm11;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vmovapd %%ymm3, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm12;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm8, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm8;   shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm13;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm11;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vmovapd %%ymm5, 96(%%r8);                vfmadd231pd 64(%%r8), %%ymm0, %%ymm5;    shl $1, %%edx;       add %%r14, %%r8;    " // L3 load, L3 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm12;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm10, 96(%%rcx);              vfmadd231pd 64(%%rcx), %%ymm0, %%ymm10;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm13;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm11;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm7, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm7;   shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm12;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm4, %%ymm0, %%ymm3;      vfmadd231pd 64(%%r9), %%ymm1, %%ymm15;   shl $1, %%edi;       add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm13;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm11;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm9, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm9;   shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm12;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm13;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vmovapd %%ymm6, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm11;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm2, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm12;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm13;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vmovapd %%ymm8, 96(%%r8);                vfmadd231pd 64(%%r8), %%ymm0, %%ymm8;    shr $1, %%edx;       add %%r14, %%r8;    " // L3 load, L3 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm11;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm4, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm12;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm6, 32(%%rbx); mov %%rax, %%rbx;   " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm13;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vmovapd %%ymm10, 96(%%rcx);              vfmadd231pd 64(%%rcx), %%ymm0, %%ymm10;  shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm11;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm6, 96(%%r8);                vfmadd231pd 64(%%r8), %%ymm0, %%ymm6;    shr $1, %%edi;       add %%r14, %%r8;    " // L3 load, L3 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm12;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm13;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vmovapd %%ymm3, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm11;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm8, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm8;   shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm12;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm13;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd 64(%%r9), %%ymm1, %%ymm15;   shl $1, %%edx;       add %%r14, %%r9;    " // RAM load
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm11;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm10, 96(%%rcx);              vfmadd231pd 64(%%rcx), %%ymm0, %%ymm10;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm12;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm13;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm7, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm7;   shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm11;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm3, 96(%%r8);                vfmadd231pd 64(%%r8), %%ymm0, %%ymm3;    shl $1, %%edi;       add %%r14, %%r8;    " // L3 load, L3 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm12;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm13;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm9, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm9;   shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm11;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm12;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vmovapd %%ymm6, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm13;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm2, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm11;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm12;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vmovapd %%ymm8, 96(%%r8);                vfmadd231pd 64(%%r8), %%ymm0, %%ymm8;    shr $1, %%edx;       add %%r14, %%r8;    " // L3 load, L3 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm13;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm4, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm11;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm12;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vmovapd %%ymm10, 96(%%rcx);              vfmadd231pd 64(%%rcx), %%ymm0, %%ymm10;  shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm13;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm6;      vfmadd231pd 64(%%r9), %%ymm1, %%ymm15;   shr $1, %%edi;       add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm11;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm12;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vmovapd %%ymm3, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm13;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm8, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm8;   shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm11;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm12;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vmovapd %%ymm5, 96(%%r8);                vfmadd231pd 64(%%r8), %%ymm0, %%ymm5;    shl $1, %%edx;       add %%r14, %%r8;    " // L3 load, L3 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm13;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm10, 96(%%rcx);              vfmadd231pd 64(%%rcx), %%ymm0, %%ymm10;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm11;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm12;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm7, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm7;   shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm13;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm4, %%ymm0, %%ymm3;      vfmadd231pd 64(%%r9), %%ymm1, %%ymm15;   shl $1, %%edi;       add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm11;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm12;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm9, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm9;   shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm13;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm11;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vmovapd %%ymm6, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm12;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm2, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm13;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm11;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vmovapd %%ymm8, 96(%%r8);                vfmadd231pd 64(%%r8), %%ymm0, %%ymm8;    shr $1, %%edx;       add %%r14, %%r8;    " // L3 load, L3 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm12;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm4, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm13;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm11;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vmovapd %%ymm10, 96(%%rcx);              vfmadd231pd 64(%%rcx), %%ymm0, %%ymm10;  shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm12;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm6, 96(%%r8);                vfmadd231pd 64(%%r8), %%ymm0, %%ymm6;    shr $1, %%edi;       add %%r14, %%r8;    " // L3 load, L3 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm13;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm11;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vmovapd %%ymm3, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm12;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm8, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm8;   shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm13;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm11;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd 64(%%r9), %%ymm1, %%ymm15;   shl $1, %%edx;       add %%r14, %%r9;    " // RAM load
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm12;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm10, 96(%%rcx);              vfmadd231pd 64(%%rcx), %%ymm0, %%ymm10;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm13;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm11;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm7, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm7;   shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm12;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm3, 96(%%r8);                vfmadd231pd 64(%%r8), %%ymm0, %%ymm3;    shl $1, %%edi;       add %%r14, %%r8;    " // L3 load, L3 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm13;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm11;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm9, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm9;   shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm12;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm13;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vmovapd %%ymm6, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm11;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm2, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm12;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm13;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vmovapd %%ymm8, 96(%%r8);                vfmadd231pd 64(%%r8), %%ymm0, %%ymm8;    shr $1, %%edx;       add %%r14, %%r8;    " // L3 load, L3 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm11;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm4, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm12;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm7, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm13;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vmovapd %%ymm10, 96(%%rcx);              vfmadd231pd 64(%%rcx), %%ymm0, %%ymm10;  shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm11;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm5;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm5, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm6;      vfmadd231pd 64(%%r9), %%ymm1, %%ymm15;   shr $1, %%edi;       add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm12;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm13;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vmovapd %%ymm3, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm11;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm7;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm7, 32(%%rbx); mov %%rax, %%rbx;   " // 2 L1 loads, L1 store
        "vmovapd %%ymm8, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm8;   shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm12;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm5, %%ymm0, %%ymm4;      vfmadd231pd %%ymm6, %%ymm1, %%ymm13;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vmovapd %%ymm5, 96(%%r8);                vfmadd231pd 64(%%r8), %%ymm0, %%ymm5;    shl $1, %%edx;       add %%r14, %%r8;    " // L3 load, L3 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm7;      vfmadd231pd %%ymm9, %%ymm1, %%ymm11;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm9;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm9, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm10, 96(%%rcx);              vfmadd231pd 64(%%rcx), %%ymm0, %%ymm10;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm2;      vfmadd231pd %%ymm4, %%ymm1, %%ymm12;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm3;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm3, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm4;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm13;  vmovapd %%ymm4, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm6, %%ymm0, %%ymm5;      vfmadd231pd %%ymm7, %%ymm1, %%ymm13;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm6;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm6, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vmovapd %%ymm7, 96(%%rcx);               vfmadd231pd 64(%%rcx), %%ymm0, %%ymm7;   shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load, L2 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm8;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm11;  vmovapd %%ymm8, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm9;     vfmadd231pd %%ymm2, %%ymm1, %%ymm11;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm10;  vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm10, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "vfmadd231pd 64(%%rbx), %%ymm0, %%ymm2;   vfmadd231pd 96(%%rbx), %%ymm1, %%ymm12;  vmovapd %%ymm2, 32(%%rbx); add $128, %%rbx;    " // 2 L1 loads, L1 store
        "movq %%mm0, %%r13;" // restore iteration counter
        //reset RAM counter
        "sub $1, %%r12;"
        "jnz _work_no_ram_reset_skl_corei_fma_1t;"
        "movabs $54613, %%r12;"
        "mov %%rax, %%r9;"
        "add $1572864, %%r9;"
        "_work_no_ram_reset_skl_corei_fma_1t:"
        "inc %%r13;" // increment iteration counter
        //reset L2-Cache counter
        "sub $1, %%r10;"
        "jnz _work_no_L2_reset_skl_corei_fma_1t;"
        "movabs $18, %%r10;"
        "mov %%rax, %%rcx;"
        "add $32768, %%rcx;"
        "_work_no_L2_reset_skl_corei_fma_1t:"
        "movq %%r13, %%mm0;" // store iteration counter
        //reset L3-Cache counter
        "sub $1, %%r11;"
        "jnz _work_no_L3_reset_skl_corei_fma_1t;"
        "movabs $393, %%r11;"
        "mov %%rax, %%r8;"
        "add $262144, %%r8;"
        "_work_no_L3_reset_skl_corei_fma_1t:"
        "mov %%rax, %%rbx;"
        "movq %%mm0, %%r13;" // restore iteration counter
	// one minute
	//"subq $124000000, %%r13;"
	// six seconds
	//"subq $12400000, %%r13;"
	// half second
	//"subq $1000000, %%r13;"
	// quarter second
	//"subq $500000, %%r13;"
	//"subq $50000, %%r13;"
	//"subq $1000, %%r13;"
	// 250 us
	"subq $500, %%r13;"
	// 25 us
	//"subq $50, %%r13;"
	//"subq $20, %%r13;"
	// small
	//"subq $5, %%r13;"
	"jns _work_done_skl_corei_fma_1t;"
        "testq $1, (%%r15);"
        "jnz _work_loop_skl_corei_fma_1t;"
	"_work_done_skl_corei_fma_1t:"
        "movq %%mm0, %%rax;" // restore iteration counter
	"movabs $0, %%r13;"
	"movq %%r13, %%mm0;"
        : "=a" (threaddata->iterations)
        : "a"(threaddata->addrMem), "b"(threaddata->addrHigh), "c" (threaddata->iterations)
        : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%rdi", "%rsi", "%rdx", "%mm0", "%mm1", "%mm2", "%mm3", "%mm4", "%mm5", "%mm6", "%mm7", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7", "%xmm8", "%xmm9", "%xmm10", "%xmm11", "%xmm12", "%xmm13", "%xmm14", "%xmm15"
        );
	    return EXIT_SUCCESS;
}

int init_snb_xeonep_avx_1t(threaddata_t* threaddata)
{
    unsigned long long addrMem = threaddata->addrMem;
    int i;

    for (i = 0; i < INIT_BLOCKSIZE; i+=8) *((double*)(addrMem+i)) = i * 1.654738925401e-10;
    for (i = INIT_BLOCKSIZE; i <= 107773952 - INIT_BLOCKSIZE; i+= INIT_BLOCKSIZE) memcpy((void*)(addrMem+i),(void*)(addrMem+i-INIT_BLOCKSIZE),INIT_BLOCKSIZE);
    for (; i <= 107773952-8; i+=8) *((double*)(addrMem+i)) = i * 1.654738925401e-15;

    threaddata->flops=5940;
    threaddata->bytes=2112;

    return EXIT_SUCCESS;
}

int asm_work_snb_xeonep_avx_1t(threaddata_t* threaddata)
{
    if (*((unsigned long long*)threaddata->addrHigh) == 0) return EXIT_SUCCESS;
	threaddata->iterations = 0;
        /* input: 
         *   - threaddata->addrMem    -> rax
         *   - threaddata->addrHigh   -> rbx
         *   - threaddata->iterations -> rcx
         * output: 
         *   - rax -> threaddata->iterations
         * register usage:
         *   - rax:           stores original pointer to buffer, used to periodically reset other pointers
         *   - rbx:           pointer to L1 buffer
         *   - rcx:           pointer to L2 buffer
         *   - rdx:           pointer to L3 buffer
         *   - rdi:           pointer to RAM buffer
         *   - r8:            counter for L2-pointer reset
         *   - r9:            counter for L3-pointer reset
         *   - r10:           counter for RAM-pointer reset
         *   - r11:           register for temporary results
         *   - r12:           stores cacheline width as increment for buffer addresses
         *   - r13:           stores address of shared variable that controls load level
         *   - r14:           stores iteration counter
         *   - mm*,xmm*,ymm*: data registers for SIMD instructions
         */
        __asm__ __volatile__(
        "mov %%rax, %%rax;" // store start address of buffer in rax
        "mov %%rbx, %%r13;" // store address of shared variable that controls load level in r13
        "mov %%rcx, %%r14;" // store iteration counter in r14
        "mov $64, %%r12;" // increment after each cache/memory access
        //Initialize AVX-Registers for Addition
        "vmovapd 0(%%rax), %%ymm0;"
        "vmovapd 32(%%rax), %%ymm1;"
        "vmovapd 64(%%rax), %%ymm2;"
        "vmovapd 96(%%rax), %%ymm3;"
        "vmovapd 128(%%rax), %%ymm4;"
        "vmovapd 160(%%rax), %%ymm5;"
        "vmovapd 192(%%rax), %%ymm6;"
        "vmovapd 224(%%rax), %%ymm7;"
        "vmovapd 256(%%rax), %%ymm8;"
        "vmovapd 288(%%rax), %%ymm9;"
        //Initialize MMX-Registers for shift operations
        "movabs $0x5555555555555555, %%r11;"
        "movq %%r11, %%mm0;"
        "movq %%mm0, %%mm1;"
        "movq %%mm0, %%mm2;"
        "movq %%mm0, %%mm3;"
        "movq %%mm0, %%mm4;"
        "movq %%mm0, %%mm5;"
        //Initialize AVX-Registers for Transfer-Operations
        "movabs $0x0F0F0F0F0F0F0F0F, %%r11;"
        "pinsrq $0, %%r11, %%xmm10;"
        "pinsrq $1, %%r11, %%xmm10;"
        "vinsertf128 $1, %%xmm10, %%ymm10, %%ymm10;"
        "shl $4, %%r11;"
        "pinsrq $0, %%r11, %%xmm11;"
        "pinsrq $1, %%r11, %%xmm11;"
        "vinsertf128 $1, %%xmm11, %%ymm11, %%ymm11;"
        "shr $4, %%r11;"
        "pinsrq $0, %%r11, %%xmm12;"
        "pinsrq $1, %%r11, %%xmm12;"
        "vinsertf128 $1, %%xmm12, %%ymm12, %%ymm12;"
        "shl $4, %%r11;"
        "pinsrq $0, %%r11, %%xmm13;"
        "pinsrq $1, %%r11, %%xmm13;"
        "vinsertf128 $1, %%xmm13, %%ymm13, %%ymm13;"
        "shr $4, %%r11;"
        "pinsrq $0, %%r11, %%xmm14;"
        "pinsrq $1, %%r11, %%xmm14;"
        "vinsertf128 $1, %%xmm14, %%ymm14, %%ymm14;"
        "shl $4, %%r11;"
        "pinsrq $0, %%r11, %%xmm15;"
        "pinsrq $1, %%r11, %%xmm15;"
        "vinsertf128 $1, %%xmm15, %%ymm15, %%ymm15;"
        "mov %%rax, %%rbx;" // address for L1-buffer
        "mov %%rax, %%rcx;"
        "add $32768, %%rcx;" // address for L2-buffer
        "mov %%rax, %%rdx;"
        "add $262144, %%rdx;" // address for L3-buffer
        "mov %%rax, %%rdi;"
        "add $2621440, %%rdi;" // address for RAM-buffer
        "movabs $29, %%r8;" // reset-counter for L2-buffer with 110 cache lines accessed per loop (199.38 KB)
        "movabs $1489, %%r9;" // reset-counter for L3-buffer with 22 cache lines accessed per loop (2047.38 KB)
        "movabs $49648, %%r10;" // reset-counter for RAM-buffer with 33 cache lines accessed per loop (102399.0 KB)

        ".align 64;"     /* alignment in bytes */
        "_work_loop_snb_xeonep_avx_1t:"
        /****************************************************************************************************
         decode 0                            decode 1                            decode 2                            decode 3 */
        "vaddpd 64(%%rdi), %%ymm1, %%ymm1;                                       psllw %%mm0, %%mm3;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rdx);          psllw %%mm3, %%mm0;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm5, %%ymm5;                                       psrlw %%mm0, %%mm3;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rdx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm9, %%ymm9;                                       psllw %%mm0, %%mm3;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm6, %%ymm6;                                       psllw %%mm3, %%mm0;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rdx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm0, %%ymm0;                                       psrlw %%mm3, %%mm0;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rdx);          psllw %%mm0, %%mm3;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm4, %%ymm4;                                       psllw %%mm3, %%mm0;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm1, %%ymm1;                                       psrlw %%mm0, %%mm3;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rdx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm5, %%ymm5;                                       psllw %%mm0, %%mm3;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rdx);          psllw %%mm3, %%mm0;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm9, %%ymm9;                                       psrlw %%mm0, %%mm3;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 mov %%rax, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm6, %%ymm6;                                       psrlw %%mm3, %%mm0;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rdx);          psllw %%mm0, %%mm3;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm0, %%ymm0;                                       psllw %%mm3, %%mm0;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rdx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm4, %%ymm4;                                       psrlw %%mm3, %%mm0;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm1, %%ymm1;                                       psllw %%mm0, %%mm3;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rdx);          psllw %%mm3, %%mm0;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm5, %%ymm5;                                       psrlw %%mm0, %%mm3;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rdx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm9, %%ymm9;                                       psllw %%mm0, %%mm3;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm6, %%ymm6;                                       psllw %%mm3, %%mm0;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rdx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm0, %%ymm0;                                       psrlw %%mm3, %%mm0;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rdx);          psllw %%mm0, %%mm3;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 mov %%rax, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm4, %%ymm4;                                       psllw %%mm3, %%mm0;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm1, %%ymm1;                                       psrlw %%mm0, %%mm3;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rdx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm5, %%ymm5;                                       psllw %%mm0, %%mm3;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rdx);          psllw %%mm3, %%mm0;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm9, %%ymm9;                                       psrlw %%mm0, %%mm3;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm6, %%ymm6;                                       psrlw %%mm3, %%mm0;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rdx);          psllw %%mm0, %%mm3;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm0, %%ymm0;                                       psllw %%mm3, %%mm0;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rdx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm4, %%ymm4;                                       psrlw %%mm3, %%mm0;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm1, %%ymm1;                                       psllw %%mm0, %%mm3;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rdx);          psllw %%mm3, %%mm0;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm5, %%ymm5;                                       psrlw %%mm0, %%mm3;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 mov %%rax, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rdx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm9, %%ymm9;                                       psllw %%mm0, %%mm3;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm6, %%ymm6;                                       psllw %%mm3, %%mm0;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rdx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm0, %%ymm0;                                       psrlw %%mm3, %%mm0;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rdx);          psllw %%mm0, %%mm3;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm4, %%ymm4;                                       psllw %%mm3, %%mm0;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm1, %%ymm1;                                       psrlw %%mm0, %%mm3;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rdx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm5, %%ymm5;                                       psllw %%mm0, %%mm3;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rdx);          psllw %%mm3, %%mm0;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm9, %%ymm9;                                       psrlw %%mm0, %%mm3;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        //reset RAM counter
        "sub $1, %%r10;"
        "jnz _work_no_ram_reset_snb_xeonep_avx_1t;"
        "movabs $49648, %%r10;"
        "mov %%rax, %%rdi;"
        "add $2621440, %%rdi;"
        "_work_no_ram_reset_snb_xeonep_avx_1t:"
        //reset L2-Cache counter
        "sub $1, %%r8;"
        "jnz _work_no_L2_reset_snb_xeonep_avx_1t;"
        "movabs $29, %%r8;"
        "mov %%rax, %%rcx;"
        "add $32768, %%rcx;"
        "_work_no_L2_reset_snb_xeonep_avx_1t:"
        //reset L3-Cache counter
        "sub $1, %%r9;"
        "jnz _work_no_L3_reset_snb_xeonep_avx_1t;"
        "movabs $1489, %%r9;"
        "mov %%rax, %%rdx;"
        "add $262144, %%rdx;"
        "_work_no_L3_reset_snb_xeonep_avx_1t:"
        "inc %%r14;" // increment iteration counter
        "mov %%rax, %%rbx;"
		// This is the FWQ code
		"mov %%r14, %%r11;"
		"subq $100, %%r11;"
		"jns _work_done_snb_xeonep_avx_1t;"
        "testq $1, (%%r13);"
        "jnz _work_loop_snb_xeonep_avx_1t;"
		"_work_done_snb_xeonep_avx_1t:"
        "movq %%r14, %%rax;" // restore iteration counter
		"movabs $0, %%r14;"
        : "=a" (threaddata->iterations)
        : "a"(threaddata->addrMem), "b"(threaddata->addrHigh), "c" (threaddata->iterations)
        : "%rdx", "%rdi", "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%mm0", "%mm1", "%mm2", "%mm3", "%mm4", "%mm5", "%mm6", "%mm7", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7", "%xmm8", "%xmm9", "%xmm10", "%xmm11", "%xmm12", "%xmm13", "%xmm14", "%xmm15"
        );
    return EXIT_SUCCESS;
}
