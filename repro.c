#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>
#include <unistd.h>
#include <sched.h>
#include "msr_core.h"
#include "master.h"

#define PSTATE 20

struct thread_data
{
	unsigned tid;
	unsigned niter;
	unsigned num_alu;
	uint64_t *perf_data;
	uint64_t *tsc_data;
	uint64_t *ctl_data;
	unsigned long loc;
};

// workloads
void alu_workload(const unsigned iter);

// prototypes
void work(void *data);
void push_data(const uint64_t data, const uint64_t ctl, const uint64_t tsc, struct thread_data *tdat);
void dump_data(const FILE *dest, const struct thread_data *tdat);
void perf_sample(struct thread_data *tdat);
void set_perf(const unsigned freq, const unsigned tid);
void check_misc_enable();
void check_hwp();
void check_rapl();
void disable_rapl();

int main(int argc, char **argv)
{
	if (argc != 4)
	{
		fprintf(stderr, "ERROR: run as so './turbo <num threads> <num iterations> <num alu>'\n");
		return -1;
	}
	if (init_msr())
	{
		fprintf(stderr, "ERROR: unable to init libmsr\n");
		return -1;
	}


	unsigned num_threads = (unsigned) atoi(argv[1]);
	pthread_t *threads = (pthread_t *) malloc(num_threads * sizeof(pthread_t));

	FILE **dump = (FILE **) malloc(num_threads * sizeof(FILE *));

	unsigned num_iterations = (unsigned) atoi(argv[2]);
	unsigned num_alu = (unsigned) atoi(argv[3]);

	check_misc_enable();
	check_hwp();
	check_rapl();
	disable_rapl();
	check_rapl();

	struct thread_data *tdat = (struct thread_data *) calloc(num_threads, sizeof(struct thread_data));
	if (tdat == NULL)
	{
		fprintf(stderr, "ERROR: not enough memory\n");
		return -1;
	}

	unsigned itr;
	for (itr = 0; itr < num_threads; itr++)
	{
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

	tdat[itr].ctl_data = (uint64_t *) malloc(num_iterations * 2 * sizeof(uint64_t));
		if (tdat[itr].ctl_data == NULL)
		{
			fprintf(stderr, "ERROR: not enough memory\n");
			return -1;
		}

		tdat[itr].tid = itr;
		tdat[itr].niter = num_iterations;
		tdat[itr].num_alu = num_alu;
		pthread_create(&threads[itr], NULL, (void *) &work, (void *) &tdat[itr]);
	}
	for (itr = 0; itr < num_threads; itr++)
	{
		pthread_join(threads[itr], NULL);
	}
	for (itr = 0; itr < num_threads; itr++)
	{
		dump_data(dump[itr], &tdat[itr]);
		fclose(dump[itr]);
		free(tdat[itr].perf_data);
		free(tdat[itr].tsc_data);
	}
	finalize_msr();
	return 0;
}

void check_rapl()
{
	uint64_t rapl;
	read_msr_by_coord(0, 0, 0, MSR_PKG_POWER_LIMIT, &rapl);
	printf("RAPL register: 0x%lx\n", (unsigned long) rapl);
	if (rapl & (0x1ul << 15))
	{
		printf("RAPL Limit 1 is Enabled (PKG_POWER_LIMIT[15] == 1)\n");
	}
	else
	{
		printf("RAPL Limit 1 is Disabled (PKG_POWER_LIMIT[15] == 0)\n");
	}

	if (rapl & (0x1ul << 16))
	{
		printf("RAPL Limit 1 is Clamping (PKG_POWER_LIMIT[16] == 1)\n");
	}
	else
	{
		printf("RAPL Limit 1 is Not Clamping (PKG_POWER_LIMIT[16] == 0)\n");
	}

	if (rapl & (0x1ul << 47))
	{
		printf("RAPL Limit 2 is Enabled (PKG_POWER_LIMIT[47] == 1)\n");
	}
	else
	{
		printf("RAPL Limit 2 is Disabled (PKG_POWER_LIMIT[47] == 0)\n");
	}

	if (rapl & (0x1ul << 48))
	{
		printf("RAPL Limit 2 is Clamping (PKG_POWER_LIMIT[48] == 1)\n");
	}
	else
	{
		printf("RAPL Limit 2 is Not Clamping (PKG_POWER_LIMIT[48] == 0)\n");
	}
}

void disable_rapl()
{
	uint64_t rapl;
	read_msr_by_coord(0, 0, 0, MSR_PKG_POWER_LIMIT, &rapl);
	uint64_t mask = ~(0x0ul | (0x1ul << 15) | (0x1ul << 16) | (0x1ul << 47) | (0x1ul << 48));
	printf("Disabling RAPL with mask %lx\n", (unsigned long) mask);
	rapl &= mask;
	write_msr_by_coord(0, 0, 0, MSR_PKG_POWER_LIMIT, rapl);
}

void check_misc_enable()
{
	uint64_t misc_enable;
	read_msr_by_coord(0, 0, 0, IA32_MISC_ENABLE, &misc_enable);
	misc_enable &= (0x1ul << 38);
	if (misc_enable)
	{
		printf("Turbo Disabled (MISC_ENABLE[38] == 1)\n");
	}
	else
	{
		printf("Turbo Enabled (MISC_ENABLE[38] == 0)\n");
	}
}

void check_hwp()
{
	uint64_t hwp_enable;
	if (read_msr_by_coord(0, 0, 0, 0x770, &hwp_enable) != 0)
	{
		printf("HWP not supported on this architecture (Failed to read MSR 0x770)\n");
	}
	hwp_enable &= 0x1ul;
	if (hwp_enable)
	{
		printf("HWP Enabled (HWP_ENABLE[1] == 1)\n");
	}
	else
	{
		printf("HWP Disabled (HWP_ENABLE[1] == 0)\n");
	}
}

void push_data(const uint64_t data, const uint64_t ctl, const uint64_t tsc, struct thread_data *tdat)
{
	if (tdat->perf_data == NULL)
	{
		return;
	}
	if (tdat->tsc_data == NULL)
	{
		return;
	}
	tdat->perf_data[tdat->loc] = data;
	tdat->tsc_data[tdat->loc] = tsc;
	tdat->ctl_data[tdat->loc] = ctl;
	tdat->loc++;
}

void dump_data(const FILE *dest, const struct thread_data *tdat)
{
	int itr;
	fprintf((FILE * __restrict__) dest, "Freq\tPERF_STATUS\tTSC\tPERF_CTL\n");
	for (itr = 0; itr < tdat->loc; itr++)
	{
		fprintf((FILE * __restrict__) dest, "%f\t%lx\t%lu\t%lx\n", ((float) ((tdat->perf_data[itr] & (0xFFFF)) >> 8)) / 10.0, (unsigned long) (tdat->perf_data[itr]), tdat->tsc_data[itr] - tdat->tsc_data[0], (unsigned long) tdat->ctl_data[itr]);
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
	printf("Requesting P-State %.1fGHz (PERF_CTL[0:15] = 0x%lx)\n", (float) freq / 10.0, (unsigned long) perf_ctl & 0xffff);
	write_msr_by_coord(0, tid, 0, IA32_PERF_CTL, perf_ctl);
}

inline void perf_sample(struct thread_data *tdat)
{
	uint64_t perf_status;
	uint64_t perf_ctl;
	static float last_freq = PSTATE / 10.0;
	static int first_catch = 1;
	read_msr_by_coord(0, tdat->tid, 0, IA32_PERF_STATUS, &perf_status);
	read_msr_by_coord(0, tdat->tid, 0, IA32_PERF_CTL, &perf_ctl);
	float freq = ((perf_status & 0xffff) >> 8) / 10.0;
	if ((perf_status & 0xffff) >> 8 != PSTATE && first_catch)
	{
		printf("Freq changed from request %.2f to %.2f\n", (PSTATE / 10.0), freq);
		first_catch = 0;
		last_freq = freq;
	}
	else if (last_freq != freq)
	{
		printf("Freq changed again from %.2f to %.2f\n", last_freq, freq);
		last_freq = freq;
	}
	uint64_t tsc;
	read_msr_by_coord(0, tdat->tid, 0, 0x10, &tsc);
	push_data(perf_status, perf_ctl, tsc, tdat);
}

void work(void *data)
{
	struct thread_data *tdat = (struct thread_data *) data;
	cpu_set_t cpus;
	CPU_ZERO(&cpus);
	CPU_SET(tdat->tid, &cpus);
	sched_setaffinity(getpid(), sizeof(cpus), &cpus);
	set_perf(PSTATE, tdat->tid);
	int i;
	for (i = 0; i < tdat->niter; i++)
	{
		if (i < tdat->niter / 8)
		{
			//set_perf(20, tdat->tid);
			alu_workload(tdat->num_alu);
		}
		else
		{
			//set_perf(33, tdat->tid);
			alu_workload(tdat->num_alu);
		}
		perf_sample(tdat);
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
			: "=a" (a), "=b" (b), "=c" (c)
			: "d" (i)
			:
		);
	}
}
