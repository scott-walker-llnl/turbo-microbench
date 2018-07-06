// Copyright 2018, Scott Walker
#define _BSD_SOURCE
#define _GNU_SOURCE
#include "msr_core.h"
#include "master.h"
#include "msr_counters.h"
#include <unistd.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <sched.h>
#include <signal.h>
#include <float.h>

/* #define DEBUG */

#define FNAMESIZE 32
#define MSR_TURBO_RATIO_LIMIT 0x1AD
#define MSR_CORE_PERF_LIMIT_REASONS 0x64F
#define THROTTLE_THRESH 100
#define LIMIT_ON_RAPL 0x0000C00
#define LIMIT_LOG_RAPL 0xC000000
#define LIMIT_LOG_MASK 0xF3FFFFFF
#define MAX_PROFILES 20
//#define DIST_THRESH 0.25
//#define PCT_THRESH 0.01
/* #define PRUNE_THRESH 0.000001 */
#define MAX_HISTORY 8
#define NUM_CLASSES 4
#define FUP_TIMEOUT 3
#define TDP_SORTA 93.0
#define FDELTA 1

#define CLASS_CPU_SLOPE_IPC 0.63396
#define CLASS_CPU_SLOPE_EPC 0.13005
#define CLASS_MEM_SLOPE_IPC 0.07642
#define CLASS_MEM_SLOPE_EPC 0.73337
#define CLASS_MIX_SLOPE_IPC ((CLASS_CPU_SLOPE_IPC + CLASS_MEM_SLOPE_IPC) / 2)
#define CLASS_MIX_SLOPE_EPC ((CLASS_CPU_SLOPE_EPC + CLASS_MEM_SLOPE_EPC) / 2 )
#define CLASS_CPU_INTERCEPT_IPC 0.10806
#define CLASS_CPU_INTERCEPT_EPC -0.12874
#define CLASS_MEM_INTERCEPT_IPC 0.08295
#define CLASS_MEM_INTERCEPT_EPC -0.20863
#define CLASS_MIX_INTERCEPT_IPC ((CLASS_CPU_INTERCEPT_IPC + CLASS_MEM_INTERCEPT_IPC) / 2)
#define CLASS_MIX_INTERCEPT_EPC ((CLASS_CPU_INTERCEPT_EPC + CLASS_MEM_INTERCEPT_EPC) / 2 )

//#define MAN_TEST
enum CLASS_ID
{
	CLASS_CPU,
	CLASS_MEM,
	CLASS_IO,
	CLASS_MIX
};

struct data_sample
{
	uint64_t frq_data;
	uint64_t tsc_data;
	uint64_t energy_data;
	uint64_t rapl_throttled;
	uint64_t therm;
	uint64_t perflimit;
	uint64_t instret;
	uint64_t llcmiss;
	uint64_t restalls;
	uint64_t exstalls;
	uint64_t branchret;
};

struct phase_profile
{
	// these are all AVERAGES
	double ipc; // instructions/cycle measured for this phase
	double mpc; // LLC miss/cycle measured for this phase
	double rpc; // resource stalls/cycle measured for this phase
	double epc; // execution stalls/cycle measured for this phase
	double bpc; // branch instructions/cycle measured for this phase

	double avg_frq; // average frequency measured for this phase
	uint16_t frq_high; // max frequency measured for this phase
	uint16_t frq_low; // min frequency measured for this phase
	double avg_cycle; // average number of cycles it takes to execute this phase
	uint32_t num_throttles; // number of times this phase was throttled last time (aka misprediction)
	uint64_t occurrences; // how many times this phase was detected
	char prev_phases[MAX_HISTORY];
	char lastprev;
	char class;
	char unthrot_count;
};

struct phase_profile profiles[MAX_PROFILES];
struct phase_profile prof_maximums;
struct phase_profile prof_minimums;
struct phase_profile prof_class[NUM_CLASSES];
char CLASS_NAMES[NUM_CLASSES + 1][8] = {"CPU\0", "MEM\0", "IO\0", "MIX\0", "UNK\0"};
int numphases = 0;
int recentphase = -1;
int lastglom = 0;
struct data_sample **thread_samples;
unsigned THREADCOUNT;
unsigned long *SAMPLECTRS;
double DIST_THRESH = 0.25;
double PCT_THRESH = 0.01;
double energy_unit;
double seconds_unit;
double avg_sample_rate;
double powlim = 0.0;
FILE *sreport;
char MAN_CTRL = 0;
double MAX_NON_TURBO = 0.0;
unsigned long MAX_PSTATE = 0x10ul;
int NUM_CPU = -1;

int LOOP_CTRL = 1;

// Prototypes
void set_perf(const unsigned freq);
void dump_phaseinfo(FILE *outfile, double *avgrate);
void frequency_scale_phase(struct phase_profile *unscaled_profile, double frq_source, double frq_target, 
		struct phase_profile *scaled_profile);

double metric_distance(struct phase_profile *old, struct phase_profile *new, struct phase_profile *maximums, 
	struct phase_profile *minimums)
{
	double ipcnorm = ((old->ipc - minimums->ipc) / (maximums->ipc - minimums->ipc)) -
					 ((new->ipc - minimums->ipc) / (maximums->ipc - minimums->ipc));

	double epcnorm = ((old->epc - minimums->epc) / (maximums->epc - minimums->epc)) -
					 ((new->epc - minimums->epc) / (maximums->epc - minimums->epc));
	
	//return sqrt(pow(ipcnorm, 2.0) + pow(mpcnorm, 2.0) + pow(rpcnorm, 2.0) + pow(epcnorm, 2.0) + pow(bpcnorm, 2.0));
	return sqrt(pow(ipcnorm, 2.0) + pow(epcnorm, 2.0));
}

void agglomerate_profiles()
{
	struct phase_profile old_profiles[MAX_PROFILES];
	int newidx = 0;
	memcpy(old_profiles, profiles, numphases * sizeof(struct phase_profile));

	//double dist[MAX_PROFILES];
	//memset(dist, 0, sizeof(double) * MAX_PROFILES);

	char valid[MAX_PROFILES];
	memset(valid, 1, MAX_PROFILES);

	int numcombines = 0;

	//printf("(glom) num phases %d\n", numphases);

	int i;
	for (i = 0; i < numphases; i++)
	{
		if (valid[i] == 0)
		{
			continue;
		}

		char matches[MAX_PROFILES];
		memset(matches, 0, MAX_PROFILES);
		uint64_t occurrence_sum = old_profiles[i].occurrences;

		int j;
		for (j = 0; j < numphases; j++)
		{
			if (i != j && valid[j])
			{
				double dist = metric_distance(&old_profiles[i], &old_profiles[j], &prof_maximums, &prof_minimums);
				if (dist < DIST_THRESH)
				{
					matches[j] = 1;
					matches[i] = 1;
					valid[j] = 0;
					valid[i] = 0;
					occurrence_sum += old_profiles[j].occurrences;
					numcombines++;
				}
			}
		}
		if (matches[i])
		{
			memcpy(&profiles[newidx], &old_profiles[i], sizeof(struct phase_profile));
			profiles[newidx].ipc *= (profiles[newidx].occurrences / (double) occurrence_sum);
			profiles[newidx].mpc *= (profiles[newidx].occurrences / (double) occurrence_sum);
			profiles[newidx].rpc *= (profiles[newidx].occurrences / (double) occurrence_sum);
			profiles[newidx].epc *= (profiles[newidx].occurrences / (double) occurrence_sum);
			profiles[newidx].bpc *= (profiles[newidx].occurrences / (double) occurrence_sum);
			profiles[newidx].num_throttles *= 
					(profiles[newidx].occurrences / (double) occurrence_sum);
			profiles[newidx].avg_cycle *= (profiles[newidx].occurrences / (double) occurrence_sum);
			int numglom = 1;
			for (j = 0; j < numphases; j++)
			{
				if (i != j && matches[j])
				{
					struct phase_profile scaled_profile;
					frequency_scale_phase(&old_profiles[j], old_profiles[j].avg_frq, 
							old_profiles[i].avg_frq, &scaled_profile);
					//printf("(glom) combining %d and %d\n", i, j);
					if (j == recentphase)
					{
						recentphase = newidx;
					}
					profiles[newidx].ipc += scaled_profile.ipc * 
							(scaled_profile.occurrences / (double) occurrence_sum);
					profiles[newidx].mpc += scaled_profile.mpc * 
							(scaled_profile.occurrences / (double) occurrence_sum);
					profiles[newidx].rpc += scaled_profile.rpc * 
							(scaled_profile.occurrences / (double) occurrence_sum);
					profiles[newidx].epc += scaled_profile.epc * 
							(scaled_profile.occurrences / (double) occurrence_sum);
					profiles[newidx].bpc += scaled_profile.bpc * 
							(scaled_profile.occurrences / (double) occurrence_sum);
					profiles[newidx].num_throttles += scaled_profile.num_throttles * 
							(scaled_profile.occurrences / (double) occurrence_sum);
					profiles[newidx].avg_cycle += scaled_profile.avg_cycle * 
							(scaled_profile.occurrences / (double) occurrence_sum);

					if (scaled_profile.frq_high > profiles[newidx].frq_high)
					{
						profiles[newidx].frq_high = scaled_profile.frq_high;
					}
					if (scaled_profile.frq_low < profiles[newidx].frq_low)
					{
						profiles[newidx].frq_low = scaled_profile.frq_low;
					}
					profiles[newidx].occurrences += scaled_profile.occurrences;

					// TODO
					profiles[newidx].avg_frq += scaled_profile.avg_frq;
					numglom++;
				}
			}

			/*
			profiles[newidx].ipc /= numglom;
			profiles[newidx].mpc /= numglom;
			profiles[newidx].rpc /= numglom;
			profiles[newidx].epc /= numglom;
			profiles[newidx].bpc /= numglom;
			profiles[newidx].num_throttles /= numglom;
			*/
			profiles[newidx].avg_frq /= numglom;

			profiles[newidx].lastprev = 0;
			memset(profiles[newidx].prev_phases, -1, MAX_HISTORY);

			newidx++;
		}
	}
	if (numcombines > 0)
	{
		// slide everything over that wasn't combined
		for (i = 0; i < numphases; i++)
		{
			if (valid[i] && i == newidx)
			{
				//printf("(glom) skipping id %d\n", i);
				newidx++;
			}
			else if (valid[i])
			{
				//printf("(glom) sliding %d to %d\n", i, newidx);
				if (i == recentphase)
				{
					recentphase = newidx;
				}
				memcpy(&profiles[newidx], &old_profiles[i], sizeof(struct phase_profile));
				
				profiles[newidx].lastprev = 0;
				memset(profiles[newidx].prev_phases, -1, MAX_HISTORY);

				newidx++;
			}
		}
		numphases = newidx;
	}
	//printf("(glom) recentphase is now %d\n", recentphase);
}

void remove_unused()
{
	if (numphases <= 0)
	{
		return;
	}
	//printf("(remove) numphases %d\n", numphases);

	int i;
	//uint64_t occurrence_sum = 0;
	char valid[MAX_PROFILES];
	memset(valid, 1, MAX_PROFILES);
	int numinvalid = 0;
	int firstinvalid = -1;

	for (i = 0; i < numphases; i++)
	{
		//occurrence_sum += old_profiles[i].occurrences;
		if (profiles[i].occurrences <= 1)
		{
			//printf("(remove) removing id %d with %lu occurrences\n", i, profiles[i].occurrences);
			valid[i] = 0;
			numinvalid++;
			if (firstinvalid < 0)
			{
				firstinvalid = i;
			}
		}
	}
	/*
	for (i = 0; i < numphases; i++)
	{
		if (old_profiles[i].occurrences / (double) occurrence_sum < PRUNE_THRESH)
		{
			valid[i] = 0;
			numinvalid++;
		}
	}
	*/
	if (numinvalid == 0)
	{
		return;
	}

	struct phase_profile old_profiles[MAX_PROFILES];
	int newidx = 0;
	memcpy(old_profiles, profiles, numphases * sizeof(struct phase_profile));

	newidx = firstinvalid;
	for (i = firstinvalid; i < numphases; i++)
	{
		if (i == newidx && valid[i])
		{
			//printf("(remove) skipping id %d\n", i);
			newidx++;
		}
		else if (valid[i])
		{
			//printf("(remove) sliding id %d to %d\n", i, newidx);
			if (i == recentphase)
			{
				recentphase = newidx;
			}
			memcpy(&profiles[newidx], &old_profiles[i], sizeof(struct phase_profile));
			newidx++;
		}
	}
	numphases = newidx;
	//printf("(remove) numphases is now %d\n", numphases);
	//printf("(remove) recentphase is now %d\n", recentphase);
}

void dump_rapl()
{
	uint64_t rapl;
	read_msr_by_coord(0, 0, 0, MSR_PKG_POWER_LIMIT, &rapl);
	fprintf(stderr, "rapl is %lx\n", (unsigned long) rapl);
}

void enable_turbo(const unsigned tid)
{
	uint64_t perf_ctl;
	read_msr_by_coord(0, tid, 0, IA32_PERF_CTL, &perf_ctl);
	perf_ctl &= 0xFFFFFFFEFFFFFFFFul;
	write_msr_by_coord(0, tid, 0, IA32_PERF_CTL, perf_ctl);
	write_msr_by_coord(0, tid, 1, IA32_PERF_CTL, perf_ctl);
}

void disable_turbo(const unsigned tid)
{
	uint64_t perf_ctl;
	read_msr_by_coord(0, tid, 0, IA32_PERF_CTL, &perf_ctl);
	perf_ctl |= 0x0000000100000000ul;
	write_msr_by_coord(0, tid, 0, IA32_PERF_CTL, perf_ctl);
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
	struct pmc *pmcounters;
	pmc_storage(&pmcounters);
	read_msr_by_coord(0, core, 0, IA32_PERF_STATUS, &perf);
	read_msr_by_coord(0, core, 0, IA32_TIME_STAMP_COUNTER, &tsc);
	read_msr_by_coord(0, core, 0, MSR_PKG_ENERGY_STATUS, &energy);
	read_msr_by_coord(0, core, 0, MSR_PKG_PERF_STATUS, &rapl_throttled);
	read_msr_by_coord(0, core, 0, IA32_THERM_STATUS, &therm);
	read_msr_by_coord(0, core, 0, IA32_FIXED_CTR0, &instret);
	read_msr_by_coord(0, core, 0, MSR_CORE_PERF_LIMIT_REASONS, &perflimit);
	read_batch(COUNTERS_DATA);
	SAMPLECTRS[core]++;
	thread_samples[core][SAMPLECTRS[core]].frq_data = perf;
	thread_samples[core][SAMPLECTRS[core]].tsc_data = tsc;
	thread_samples[core][SAMPLECTRS[core]].energy_data = energy & 0xFFFFFFFF;
	thread_samples[core][SAMPLECTRS[core]].rapl_throttled = rapl_throttled & 0xFFFFFFFF;
	thread_samples[core][SAMPLECTRS[core]].therm = therm;
	thread_samples[core][SAMPLECTRS[core]].perflimit = perflimit;
	thread_samples[core][SAMPLECTRS[core]].instret = instret;
	thread_samples[core][SAMPLECTRS[core]].llcmiss = *pmcounters->pmc0[0];
	thread_samples[core][SAMPLECTRS[core]].restalls = *pmcounters->pmc1[0];
	thread_samples[core][SAMPLECTRS[core]].exstalls = *pmcounters->pmc2[0];
	thread_samples[core][SAMPLECTRS[core]].branchret = *pmcounters->pmc3[0];
	return 0;
}

void update_minmax(struct phase_profile *this_profile, struct phase_profile *maximums, struct phase_profile *minimums)
{
	if (this_profile->ipc > maximums->ipc)
	{
		maximums->ipc = this_profile->ipc;
	}
	if (this_profile->mpc > maximums->mpc)
	{
		maximums->mpc = this_profile->mpc;
	}
	if (this_profile->rpc > maximums->rpc)
	{
		maximums->rpc = this_profile->rpc;
	}
	if (this_profile->epc > maximums->epc)
	{
		maximums->epc = this_profile->epc;
	}
	if (this_profile->bpc > maximums->bpc)
	{
		maximums->bpc = this_profile->bpc;
	}

	if (this_profile->ipc < minimums->ipc)
	{
		minimums->ipc = this_profile->ipc;
	}
	if (this_profile->mpc < minimums->mpc)
	{
		minimums->mpc = this_profile->mpc;
	}
	if (this_profile->rpc < minimums->rpc)
	{
		minimums->rpc = this_profile->rpc;
	}
	if (this_profile->epc < minimums->epc)
	{
		minimums->epc = this_profile->epc;
	}
	if (this_profile->bpc < minimums->bpc)
	{
		minimums->bpc = this_profile->bpc;
	}
}

void print_profile(struct phase_profile *prof)
{
	printf("ipc: %lf\nmpc %lf\nrpc %lf\nepc %lf\nbpc %lf\n", prof->ipc, prof->mpc,
			prof->rpc, prof->epc, prof->bpc);
}

void update_profile(struct phase_profile *this_profile, int profidx, uint64_t perf, unsigned this_throttle, 
		double avgfrq, int lastphase)
{
	if (profidx > numphases)
	{
		printf("ERROR: profile does not exist\n");
		return;
	}

	profiles[profidx].ipc = (profiles[profidx].ipc * 
			(profiles[profidx].occurrences / (profiles[profidx].occurrences + 1.0))) +
			(this_profile->ipc * (1.0 / (profiles[profidx].occurrences + 1.0)));

	profiles[profidx].mpc = (profiles[profidx].mpc * 
			(profiles[profidx].occurrences / (profiles[profidx].occurrences + 1.0))) +
			(this_profile->mpc * (1.0 / (profiles[profidx].occurrences + 1.0)));

	profiles[profidx].rpc = (profiles[profidx].rpc * 
			(profiles[profidx].occurrences / (profiles[profidx].occurrences + 1.0))) +
			(this_profile->rpc * (1.0 / (profiles[profidx].occurrences + 1.0)));

	profiles[profidx].epc = (profiles[profidx].epc * 
			(profiles[profidx].occurrences / (profiles[profidx].occurrences + 1.0))) +
			(this_profile->epc * (1.0 / (profiles[profidx].occurrences + 1.0)));

	profiles[profidx].bpc = (profiles[profidx].bpc * 
			(profiles[profidx].occurrences / (profiles[profidx].occurrences + 1.0))) +
			(this_profile->bpc * (1.0 / (profiles[profidx].occurrences + 1.0)));

//	profiles[profidx].num_throttles = (profiles[profidx].num_throttles * 
//			(profiles[profidx].occurrences / (profiles[profidx].occurrences + 1.0))) +
//			(this_profile->num_throttles * (1.0 / (profiles[profidx].occurrences + 1.0)));
	profiles[profidx].num_throttles += this_throttle;

/*
	if (perf > profiles[profidx].frq_high)
	{
		profiles[profidx].frq_high = perf;
	}
*/
	if (perf < profiles[profidx].frq_low)
	{
		profiles[profidx].frq_low = perf;
	}

	// TODO
	profiles[profidx].avg_frq = avgfrq;
	profiles[profidx].occurrences++;

	char last = profiles[profidx].lastprev;
	int i;
	for (i = 0; i <= last && i < MAX_HISTORY; i++)
	{
		if (profiles[profidx].prev_phases[i] == lastphase)
		{
			// it's already there
			return;
		}
	}
	if (last < MAX_HISTORY)
	{
		last++;
		profiles[profidx].prev_phases[last] = lastphase;
		profiles[profidx].lastprev = last;
	}
}

void add_profile(struct phase_profile *this_profile, uint64_t perf, unsigned this_throttle, double avgfrq, 
		int lastphase)
{
	profiles[numphases].ipc = this_profile->ipc;
	profiles[numphases].mpc = this_profile->mpc;
	profiles[numphases].rpc = this_profile->rpc;
	profiles[numphases].epc = this_profile->epc;
	profiles[numphases].bpc = this_profile->bpc;

	profiles[numphases].avg_frq = avgfrq;
	profiles[numphases].frq_high = MAX_PSTATE;
	profiles[numphases].frq_low = perf & 0xFFFF;
	profiles[numphases].avg_cycle = 0;
	profiles[numphases].num_throttles = this_throttle;
	profiles[numphases].occurrences = 0;
	memset(profiles[numphases].prev_phases, -1, MAX_HISTORY);
	profiles[numphases].prev_phases[0] = lastphase;
	profiles[numphases].lastprev = 0;
	profiles[numphases].class = 4;
	profiles[numphases].unthrot_count = 0;

	numphases++;
#ifdef DEBUG
	printf("Added new phase profile %d\n", numphases);
#endif
}

// TODO: need maximums/minimums to take class values into account?
int classify_phase(struct phase_profile *phase, uint64_t perf)
{
	int i = -1;
	int minidx = -1;
	double mindist = DBL_MAX;
	prof_class[0].ipc = perf * CLASS_CPU_SLOPE_IPC + CLASS_CPU_INTERCEPT_IPC;
	prof_class[0].epc = perf * CLASS_CPU_SLOPE_EPC + CLASS_CPU_INTERCEPT_EPC;
	prof_class[1].ipc = perf * CLASS_MEM_SLOPE_IPC + CLASS_MEM_INTERCEPT_IPC;
	prof_class[1].epc = perf * CLASS_MEM_SLOPE_EPC + CLASS_MEM_INTERCEPT_EPC;
	// skip IO class, it doesn't change much
	prof_class[3].ipc = perf * CLASS_MIX_SLOPE_IPC + CLASS_MIX_INTERCEPT_IPC;
	prof_class[3].epc = perf * CLASS_MIX_SLOPE_EPC + CLASS_MIX_INTERCEPT_EPC;
	for (i = 0; i < NUM_CLASSES; i++)
	{
		double dist = metric_distance(phase, &prof_class[i], &prof_maximums, &prof_minimums);
		if (dist < mindist)
		{
			mindist = dist;
			minidx = i;
		}
	}
	phase->class = minidx;
	return minidx;
}

void classify_and_react(int phase, char isthrottled, char wasthrottled, uint64_t perf)
{
	int class = classify_phase(&profiles[phase], perf);
	switch (class)
	{
		case CLASS_CPU:
			// is CPU phase
			if (wasthrottled)
			{
				set_perf(profiles[phase].frq_high - 1);
				if (perf < profiles[phase].frq_high)
				{
					profiles[phase].frq_high--;
				}
				profiles[phase].unthrot_count = 0;
			}
			else
			{
				if (perf > profiles[phase].frq_high)
				{
					profiles[phase].frq_high = perf;
				}
				if (profiles[phase].frq_high < MAX_PSTATE && 
					profiles[phase].unthrot_count >= FUP_TIMEOUT)
				{
					profiles[phase].frq_high++;
				}
				/*
				else
				{
					profiles[phase].frq_high++;
				}
				*/
				/*
				else if ((double) perf < profiles[phase].avg_frq)
				{
					profiles[phase].frq_high++;
				}
				*/
				profiles[phase].unthrot_count++;
				set_perf(profiles[phase].frq_high);
			}
			profiles[phase].class = 0;
			break;
		case CLASS_MEM:
			// is MEM phase
			/*
			if (wasthrottled)
			{
				set_perf((unsigned) (profiles[phase].avg_frq * 10) - FDELTA);
				profiles[phase].unthrot_count = 0;
			}
			else
			{
				set_perf((unsigned long) (4.5 * (powlim / TDP_SORTA) * 10));
			}
			*/
			//set_perf((unsigned long) (4.5 * (powlim / TDP_SORTA) * 10.0));
			set_perf(0x2A);
			profiles[phase].class = 1;
			break;
		case CLASS_IO:
			// is IO phase
			set_perf(0x2A); // seems counterintiutive but, yes, this should be high
			profiles[phase].class = 2;
			break;
		case CLASS_MIX:
			// is MIXED phase
			if (wasthrottled)
			{
				set_perf((unsigned) (profiles[phase].avg_frq * 10) - FDELTA);
				profiles[phase].unthrot_count = 0;
			}
			else
			{
				if (profiles[phase].unthrot_count > FUP_TIMEOUT)
				{
					set_perf((unsigned) (profiles[phase].avg_frq * 10) + FDELTA * 2);
				}
				else
				{
					set_perf((unsigned) (profiles[phase].avg_frq * 10) + FDELTA);
				}
			}
			profiles[phase].class = 3;
			break;
		default:
			printf("ERROR: no phase classification\n");
			profiles[phase].class = 4;
			break;
	}
}

// TODO: scaling is off
void frequency_scale_phase(struct phase_profile *unscaled_profile, double frq_source, double frq_target, 
		struct phase_profile *scaled_profile)
{
	*scaled_profile = *unscaled_profile;
	// if the frequencies are already close then just return the copy, don't scale
	if (frq_target + 0.1 > frq_source && frq_target - 0.1 < frq_source)
	{
		return;
	}
	double cpuipc = frq_source * CLASS_CPU_SLOPE_IPC + CLASS_CPU_INTERCEPT_IPC;
	double memipc = frq_source * CLASS_MEM_SLOPE_IPC + CLASS_MEM_INTERCEPT_IPC;
	double ipc_percent = (unscaled_profile->ipc - memipc) / (cpuipc - memipc);

	double cpuepc = frq_source * CLASS_CPU_SLOPE_EPC + CLASS_CPU_INTERCEPT_EPC;
	double memepc = frq_source * CLASS_MEM_SLOPE_EPC + CLASS_MEM_INTERCEPT_EPC;
	// cpu and mem switched here because CPU has minimum in this case
	double epc_percent = (unscaled_profile->epc - cpuepc) / (memepc - cpuepc);

	// TODO: this threshold should be a variable
	// if the percentages do not add close to 1, then this data point is atypical so don't scale it
	if (ipc_percent + epc_percent < 0.8)
	{
		return;
	}
	scaled_profile->ipc = ((ipc_percent * CLASS_CPU_SLOPE_IPC) + ((1.0 - ipc_percent) * CLASS_MEM_SLOPE_IPC)) * 
		frq_target + (ipc_percent * CLASS_CPU_INTERCEPT_IPC) + ((1.0 - ipc_percent) * CLASS_MEM_SLOPE_IPC);
	scaled_profile->epc = ((epc_percent * CLASS_MEM_SLOPE_EPC) + ((1.0 - epc_percent) * CLASS_CPU_SLOPE_EPC)) * 
		frq_target + (epc_percent * CLASS_MEM_INTERCEPT_EPC) + ((1.0 - epc_percent) * CLASS_CPU_SLOPE_EPC);
	if (scaled_profile->ipc < 0.0)
	{
		scaled_profile->ipc = 0.0;
	}
	if (scaled_profile->epc < 0.0)
	{
		scaled_profile->epc = 0.0;
	}
	if (scaled_profile->ipc > 4.0 || scaled_profile->epc > 4.0)
	{
		printf("\nfrq source %lf, frq target %lf\n", frq_source, frq_target);
		printf("ipc scale from %lf to %lf\nepc scaled from %lf to %lf\n", unscaled_profile->ipc, scaled_profile->ipc,
				unscaled_profile->epc, scaled_profile->epc);
		printf("ipc percent %lf, epc percent %lf\n", ipc_percent, epc_percent);
	}
#ifdef DEBUG
	printf("\nfrq source %lf, frq target %lf\n", frq_source, frq_target);
	printf("ipc scale from %lf to %lf\nepc scaled from %lf to %lf\n", unscaled_profile->ipc, scaled_profile->ipc,
			unscaled_profile->epc, scaled_profile->epc);
	printf("ipc percent %lf, epc percent %lf\n", ipc_percent, epc_percent);
#endif
}

int branch_same_phase(struct phase_profile *this_profile, uint64_t perf, char wasthrottled, 
		char isthrottled, double phase_avgfrq)
{
	if (recentphase < 0)
	{
		return 0;
	}

	struct phase_profile scaled_profile = *this_profile;
	double dist_to_recent;
	frequency_scale_phase(this_profile, phase_avgfrq, profiles[recentphase].avg_frq, &scaled_profile);
	dist_to_recent = metric_distance(&scaled_profile, &profiles[recentphase], &prof_maximums, &prof_minimums);

	if (dist_to_recent < DIST_THRESH)
	{
#ifdef DEBUG
		printf("Phase has not changed, dist to recent %lf\n", dist_to_recent);
#endif
		// we are in the same phase, update it
		update_profile(&scaled_profile, recentphase, perf, wasthrottled, phase_avgfrq, recentphase);
		if (MAN_CTRL)
		{
			classify_and_react(recentphase, isthrottled, wasthrottled, perf);
		}
		return 1;
	}
#ifdef DEBUG
	printf("Phase has changed, dist to recent %lf\n", dist_to_recent);
	printf("\trecent phase id %d: ipc %lf, epc %lf, freq %lf. new phase: ipc %lf, epc %lf, freq %lf\n",
		recentphase, profiles[recentphase].ipc, profiles[recentphase].epc, profiles[recentphase].avg_frq,
		this_profile->ipc, this_profile->epc, phase_avgfrq);
#endif
	return 0;
}

int branch_change_phase(struct phase_profile *this_profile, uint64_t perf, char wasthrottled, 
		char isthrottled, double phase_avgfrq)
{
	// we are in a different phase so search for the best match
	double distances[MAX_PROFILES];
	int k;
	for (k = 0; k < numphases; k++)
	{
		distances[k] = DBL_MAX;
	}

	if (this_profile->ipc == 0.0 || this_profile->epc == 0.0)
	{
		printf("ERROR: bad profile data\n");
	}

#ifdef DEBUG
		printf("Search for most similar phase\n");
#endif

	struct phase_profile min_scaled_profile;
	double min_dist = DBL_MAX;
	int numunder = 0;
	int min_idx = -1;
	int i;
	for (i = 0; i < numphases; i++)
	{
		// measure the distance to every known phase
		if (i == recentphase)
		{
			continue;
		}
		struct phase_profile scaled_profile;
		frequency_scale_phase(this_profile, phase_avgfrq, profiles[i].avg_frq, &scaled_profile);
		distances[i] = metric_distance(&scaled_profile, &profiles[i], &prof_maximums, &prof_minimums);

		if (distances[i] < min_dist)
		{
			min_dist = distances[i];
			min_idx = i;
			min_scaled_profile = scaled_profile;
		}
		if (distances[i] < DIST_THRESH)
		{
			numunder++;
		}
	}

#ifdef DEBUG
	int q;
	for (q = 0; q < numphases; q++)
	{
		printf("distance from %d to %d is %lf\n", numphases, q, distances[q]);
	}
#endif

	if (min_idx >= 0 && min_dist < DIST_THRESH)
	{
		// we found an existing phase which matches the currently executing workload
#ifdef DEBUG
		printf("Found existing phase, with distance %lf\n", min_dist);
#endif
		update_profile(&min_scaled_profile, min_idx, perf, isthrottled, phase_avgfrq, recentphase);
		recentphase = min_idx;
		if (MAN_CTRL)
		{
			classify_and_react(recentphase, isthrottled, wasthrottled, perf);
		}
	}
	else
	{
		// the currently executing workload has never been seen before
		if (numphases >= MAX_PROFILES)
		{
			printf("ERROR: out of profile storage, increase the limit or change the sensitivity\n");
			return -1;
		}
		// add new phase
		add_profile(this_profile, perf, isthrottled, phase_avgfrq, recentphase);
		recentphase = numphases - 1;
	}

	// if there are many matches, we should combine similar phases
	if (numunder > 0)
	{
		agglomerate_profiles();
	}
}

void pow_aware_perf()
{
	static uint64_t begin_aperf = 0, begin_mperf = 0;
	static uint64_t last_aperf = 0, last_mperf = 0;
	static uint64_t tsc_timer = 0;
	static uint64_t phase_start_tsc = 0;

	if (last_aperf == 0)
	{
		read_msr_by_coord(0, 0, 0, MSR_IA32_MPERF, &begin_mperf);
		read_msr_by_coord(0, 0, 0, MSR_IA32_APERF, &begin_aperf);
		read_msr_by_coord(0, 0, 0, IA32_TIME_STAMP_COUNTER, &tsc_timer);
		phase_start_tsc = tsc_timer;
		last_aperf = begin_aperf;
		last_mperf = begin_mperf;
	}

	uint64_t aperf, mperf;
	read_msr_by_coord(0, 0, 0, MSR_IA32_MPERF, &mperf);
	read_msr_by_coord(0, 0, 0, MSR_IA32_APERF, &aperf);

	uint64_t this_instret = thread_samples[0][SAMPLECTRS[0]].instret - thread_samples[0][SAMPLECTRS[0] - 1].instret;
	uint64_t this_cycle = thread_samples[0][SAMPLECTRS[0]].tsc_data - thread_samples[0][SAMPLECTRS[0] - 1].tsc_data;
	unsigned this_throttle = (thread_samples[0][SAMPLECTRS[0]].rapl_throttled & 0xFFFFFFFF) - 
			(thread_samples[0][SAMPLECTRS[0] - 1].rapl_throttled & 0xFFFFFFFF);
	uint64_t this_llcmiss = thread_samples[0][SAMPLECTRS[0]].llcmiss - thread_samples[0][SAMPLECTRS[0] - 1].llcmiss;
	uint64_t this_restalls = thread_samples[0][SAMPLECTRS[0]].restalls - thread_samples[0][SAMPLECTRS[0] - 1].restalls;
	uint64_t this_exstalls = thread_samples[0][SAMPLECTRS[0]].exstalls - thread_samples[0][SAMPLECTRS[0] - 1].exstalls;
	uint64_t this_branchret = thread_samples[0][SAMPLECTRS[0]].branchret - thread_samples[0][SAMPLECTRS[0] - 1].branchret;

	double total_avgfrq = ((double) (aperf - begin_aperf) / (double) (mperf - begin_mperf)) * MAX_NON_TURBO;
	double phase_avgfrq = ((double) (aperf - last_aperf) / (double) (mperf - last_mperf)) * MAX_NON_TURBO;
	uint64_t perf = ((thread_samples[0][SAMPLECTRS[0]].frq_data & 0xFFFFul) >> 8);

	struct phase_profile this_profile;

	this_profile.ipc = ((double) this_instret) / ((double) this_cycle); // instructions per cycle
	this_profile.mpc = ((double) this_llcmiss) / ((double) this_cycle); // cache misses per cycle
	this_profile.rpc = ((double) this_restalls) / ((double) this_cycle); // resource stalls per cycle
	this_profile.epc = ((double) this_exstalls) / ((double) this_cycle); // execution stalls per cycle
	this_profile.bpc = ((double) this_branchret) / ((double) this_cycle); // branch instructions retired per cycle

	update_minmax(&this_profile, &prof_maximums, &prof_minimums);

	if (numphases >= MAX_PROFILES)
	{
		remove_unused();
	}

	if (numphases == 0)
	{
		add_profile(&this_profile, perf, 0, phase_avgfrq, recentphase);
		return;
	}
	// we do phase analysis
	// if current execution is similar to previously seen phase, update that phase
	// check to see if we are in the same phase
	if (recentphase > numphases)
	{
		//printf("recent phase no longer exists\n");
		recentphase = -1;
	}

	uint64_t limreasons = thread_samples[0][SAMPLECTRS[0]].perflimit;
	char isthrottled = 0;
	char wasthrottled = 0;
	if (limreasons & LIMIT_LOG_RAPL)
	{
		write_msr_by_coord(0, 0, 0, MSR_CORE_PERF_LIMIT_REASONS, (limreasons & LIMIT_LOG_MASK));
		wasthrottled = 1;
	}
	if (limreasons & LIMIT_ON_RAPL)
	{
		isthrottled = 1;
	}

	// we may be in the same phase
	if (branch_same_phase(&this_profile, perf, wasthrottled, isthrottled, phase_avgfrq))
	{
		return;
	}
	// the phase changed
	if (profiles[recentphase].avg_cycle < (thread_samples[0][SAMPLECTRS[0]].tsc_data - phase_start_tsc))
	{
		profiles[recentphase].avg_cycle = (thread_samples[0][SAMPLECTRS[0]].tsc_data - phase_start_tsc);
	}
	phase_start_tsc = thread_samples[0][SAMPLECTRS[0]].tsc_data;
	last_aperf = aperf;
	last_mperf = mperf;
	branch_change_phase(&this_profile, perf, wasthrottled, isthrottled, phase_avgfrq);
}

void set_perf(const unsigned freq)
{
	static uint64_t perf_ctl = 0x0ul;
	uint64_t freq_mask = 0x0ul;
	if (perf_ctl == 0x0ul)
	{
		//read_msr_by_coord(0, tid, 0, IA32_PERF_CTL, &perf_ctl);
		read_msr_by_coord(0, 0, 0, IA32_PERF_CTL, &perf_ctl);
	}
	perf_ctl &= 0xFFFFFFFFFFFF0000ul;
	freq_mask = freq;
	freq_mask <<= 8;
	perf_ctl |= freq_mask;
	//write_msr_by_coord(0, tid, 0, IA32_PERF_CTL, perf_ctl);
	//write_msr_by_coord(0, tid, 1, IA32_PERF_CTL, perf_ctl);
	int i;
	for (i = 0; i < NUM_CPU; i++)
	{
		write_msr_by_coord(0, i, 0, IA32_PERF_CTL, perf_ctl);
		write_msr_by_coord(0, i, 1, IA32_PERF_CTL, perf_ctl);
	}
}

void dump_phaseinfo(FILE *outfile, double *avgrate)
{
	int i;
	uint64_t recorded_steps = 0;
	for (i = 0; i < numphases; i++)
	{
		recorded_steps += profiles[i].occurrences;
	}
	double totaltime = 0.0;
	double totalpct = 0.0;
	for (i = 0; i < numphases; i++)
	{
		double pct = (double) profiles[i].occurrences / (double) recorded_steps;
		if (avgrate != NULL)
		{
			fprintf(outfile, "PHASE ID %d\t %.3lf seconds\t(%3.2lf%%)\n", i, *avgrate * 
					profiles[i].occurrences, pct * 100.0);
			totaltime += *avgrate * profiles[i].occurrences;
			//totalpct += pct * 100.0;
		}
	}
	if (avgrate != NULL)
	{
		totalpct = (double) recorded_steps / (double) SAMPLECTRS[0];
		fprintf(outfile, "TOTAL\t\t%.2lf\t(%3.2lf%% accounted)\n", totaltime, totalpct * 100.0);
	}
	fprintf(outfile, "min instructions per cycle        %lf\n", prof_minimums.ipc);
	fprintf(outfile, "min LLC misses per cycle          %lf\n", prof_minimums.mpc);
	fprintf(outfile, "min resource stalls per cycle     %lf\n", prof_minimums.rpc);
	fprintf(outfile, "min execution stalls per cycle    %lf\n", prof_minimums.epc);
	fprintf(outfile, "min branch instructions per cycle %lf\n", prof_minimums.bpc);
	fprintf(outfile, "max instructions per cycle        %lf\n", prof_maximums.ipc);
	fprintf(outfile, "max LLC misses per cycle          %lf\n", prof_maximums.mpc);
	fprintf(outfile, "max resource stalls per cycle     %lf\n", prof_maximums.rpc);
	fprintf(outfile, "max execution stalls per cycle    %lf\n", prof_maximums.epc);
	fprintf(outfile, "max branch instructions per cycle %lf\n", prof_maximums.bpc);
	for (i = 0; i < numphases; i++)
	{
		// ignore phases that are less than x% of program
		double pct = (double) profiles[i].occurrences / (double) recorded_steps;
		if (pct < PCT_THRESH)
		{
			continue;
		}
		if (avgrate != NULL)
		{
			fprintf(outfile, "\nPHASE ID %d\t %.3lf seconds\t(%3.2lf%%)\n", i, *avgrate * 
					profiles[i].occurrences, pct * 100.0);
		}
		else
		{
			fprintf(outfile, "\nPHASE ID %d\t(%3.0lf%%)\n", i, pct * 100);
		}
		fprintf(outfile, "\tinstructions per cycle        %lf\n", profiles[i].ipc);
		fprintf(outfile, "\tLLC misses per cycle          %lf\n", profiles[i].mpc);
		fprintf(outfile, "\tresource stalls per cycle     %lf\n", profiles[i].rpc);
		fprintf(outfile, "\texecution stalls per cycle    %lf\n", profiles[i].epc);
		fprintf(outfile, "\tbranch instructions per cycle %lf\n", profiles[i].bpc);
		fprintf(outfile, "\tphase occurrences             %lu\n\tprev phase id's:", profiles[i].occurrences);
		int k;
		for (k = 0; k < MAX_HISTORY; k++)
		{
			if (profiles[i].prev_phases[k] >= 0)
			{
				fprintf(outfile, " %d,", profiles[i].prev_phases[k]);
			}
		}
		fprintf(outfile, "\n\tavg frq     %lf\n", profiles[i].avg_frq);
		fprintf(outfile, "\tfrq low       %x\n", profiles[i].frq_low);
		fprintf(outfile, "\tfrq high      %x\n", profiles[i].frq_high);
		fprintf(outfile, "\tavg cycles    %lf (%lf seconds)\n", profiles[i].avg_cycle, 
				profiles[i].avg_cycle / (profiles[i].avg_frq * 1000000000.0));
		fprintf(outfile, "\tnum throttles %u\n", profiles[i].num_throttles);
		fprintf(outfile, "\tclass %s\n", CLASS_NAMES[profiles[i].class]);

		int j;
		for (j = 0; j < numphases; j++)
		{
			double lpct = (double) profiles[j].occurrences / (double) recorded_steps;
			if (lpct > PCT_THRESH)
			{
				struct phase_profile scaled_profile;
				frequency_scale_phase(&profiles[j], profiles[j].avg_frq, profiles[i].avg_frq, &scaled_profile);
				double dist = metric_distance(&scaled_profile, &profiles[i], &prof_maximums, &prof_minimums);
				fprintf(outfile, "\tdistance from %d: %lf\n", j, dist);
			}
		}
	}
}

void dump_data(FILE **outfile, double durctr)
{
	unsigned long total_pre = thread_samples[0][2].energy_data;
	unsigned long total_post = 0;
	unsigned long mid_pow = 0;
	int powovf = 0;
	int j;
	for (j = 0; j < THREADCOUNT; j++)
	{
		fprintf(outfile[j], "freq\tp-state\ttsc\tpower\trapl-throttle-cycles\tTemp(C)\tCORE_PERF_LIMIT_REASONS\tinstret\tllcmiss\tresource-stall\texec-stall\tbranch-retired\n");
		unsigned long i;
		for (i = 0; i < SAMPLECTRS[j]; i++)
		{
			double time = (thread_samples[j][i + 1].tsc_data -
				thread_samples[j][i].tsc_data) / 
				((((thread_samples[j][i].frq_data & 0xFFFFul) >> 8) / 10.0) 
				* 1000000000.0);

			unsigned long diff = (thread_samples[j][i + 1].energy_data -
				thread_samples[j][i].energy_data);


			fprintf(outfile[j], "%f\t%llx\t%llu\t%lf\t%lu\t%u\t%lx\t%lu\t%lu\t%lu\t%lu\t%lu\n", 
				((thread_samples[j][i].frq_data & 0xFFFFul) >> 8) / 10.0,
				(unsigned long long) (thread_samples[j][i].frq_data & 0xFFFFul),
				//(unsigned long long) (thread_samples[j][i].tsc_data),
				(unsigned long long) (thread_samples[j][i + 1].tsc_data - 
					thread_samples[j][i].tsc_data),
				diff * energy_unit / time,
				(unsigned long long) ((thread_samples[j][i + 1].rapl_throttled & 0xFFFFFFFF) - (thread_samples[j][i].rapl_throttled & 0xFFFFFFFF)),
				80 - ((thread_samples[j][i].therm & 0x7F0000) >> 16),
				(unsigned long) thread_samples[j][i].perflimit,
				thread_samples[j][i + 1].instret - thread_samples[j][i].instret,
				thread_samples[j][i + 1].llcmiss - thread_samples[j][i].llcmiss,
				thread_samples[j][i + 1].restalls - thread_samples[j][i].restalls,
				thread_samples[j][i + 1].exstalls - thread_samples[j][i].exstalls,
				thread_samples[j][i + 1].branchret - thread_samples[j][i].branchret
				);

			if (thread_samples[j][i + 1].energy_data < thread_samples[j][i].energy_data)
			{
				powovf++;
			}
			if (i == SAMPLECTRS[j] / 2)
			{
				mid_pow = (unsigned long) thread_samples[j][i].energy_data;
			}
		}
	}
	total_post = thread_samples[0][SAMPLECTRS[0] - 1].energy_data;

	fprintf(sreport, "total power %lf\n", ((total_post - total_pre) * energy_unit) / durctr);
	fprintf(sreport, "mid power %lf\n", ((total_post - mid_pow) * energy_unit) / (durctr / 2.0));
	//fprintf(sreport, "hex pre %lx hex post %lx\n", total_pre, total_post);
	fprintf(sreport, "total power ovf %lf\n", (0xEFFFFFFF * powovf - total_pre + total_post) / durctr);
	fprintf(sreport, "energy ctr overflows: %d\n", powovf);
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
	fprintf(sreport, "hwp %lx %s\n", hwp, (hwp == 0 ? "disabled" : "enabled"));
	uint64_t hwp_cap = 0x0;
	read_msr_by_coord(0, 0, 0, 0x771, &hwp_cap);
	fprintf( sreport, "hwp cap %lx\n", hwp_cap);
	uint64_t hwp_req = 0x0;
	read_msr_by_coord(0, 0, 0, 0x772, &hwp_req);
	fprintf( sreport, "hwp req %lx\n", hwp_req);
	uint64_t hwp_int = 0x0;
	read_msr_by_coord(0, 0, 0, 0x773, &hwp_int);
	fprintf( sreport, "hwp int %lx\n", hwp_int);
	uint64_t hwp_log_req = 0x0;
	read_msr_by_coord(0, 0, 0, 0x774, &hwp_log_req);
	fprintf( sreport, "hwp log req %lx\n", hwp_log_req);
	uint64_t hwp_stat = 0x0;
	read_msr_by_coord(0, 0, 0, 0x777, &hwp_stat);
	fprintf( sreport, "hwp stat %lx\n", hwp_stat);

	hwp_req = 0x14150Alu;
	hwp_log_req = 0x14150Alu;
	//write_msr_by_coord(0, 0, 0, 0x772, hwp_req);
	//write_msr_by_coord(0, 0, 0, 0x772, hwp_req);
	read_msr_by_coord(0, 0, 0, 0x772, &hwp_req);
	fprintf( sreport, "hwp req %lx\n", hwp_req);
	read_msr_by_coord(0, 0, 0, 0x774, &hwp_log_req);
	fprintf( sreport, "hwp log req %lx\n", hwp_log_req);
	
}

void signal_exit(int signum)
{
	fprintf(stderr, "Sampler terminating...\n");
	fflush(stderr);
	LOOP_CTRL = 0;
	return;
}

int main(int argc, char **argv)
{
	if (argc < 12)
	{
		fprintf(stderr, "ERROR: bad arguments\n");
		//fprintf(stderr, "Usage: ./t <threads> <duration in seconds> <samples per second> <rapl1> <rapl2>\n");
		return -1;
	}

	if (init_msr())
	{
		fprintf(stderr, "ERROR: unable to init libmsr\n");
		return -1;
	}

	enable_turbo(0);
	enable_turbo(1);
	enable_turbo(2);
	enable_turbo(3);

	set_all_pmc_ctrl(0x0, 0x43, 0x41, 0x2E, 1); // LLC miss
	set_all_pmc_ctrl(0x0, 0x43, 0x01, 0xA2, 2); // resource stalls
	set_all_pmc_ctrl(0x0, 0x43, 0x04, 0xA3, 3); // execution stalls
	set_all_pmc_ctrl(0x0, 0x43, 0x00, 0xC4, 4); // branch instructions retired
	//set_all_pmc_ctrl(0x0, 0x43, 0x02, 0xC7, 3); // SSE/AVX single precision retired
	//set_all_pmc_ctrl(0x0, 0x43, 0x01, 0xC7, 4); // SSE/AVX double precision retired
	//set_all_pmc_ctrl(0x0, 0x43, 0x04, 0xC5, 3); // branch misses retired
	enable_pmc();

	sreport = fopen("sreport", "w");

	struct sigaction sighand;
	memset(&sighand, 0, sizeof(struct sigaction));
	sighand.sa_handler = signal_exit;
	sigaction(SIGUSR1, &sighand, NULL);

	sigset_t sset;
	sigemptyset(&sset);
	sigaddset(&sset, SIGUSR1);
	sigprocmask(SIG_UNBLOCK, &sset, NULL);

	THREADCOUNT = (unsigned) atoi(argv[1]);
	unsigned duration = (unsigned) atoi(argv[2]);
	unsigned sps = (unsigned) atoi(argv[3]);
	unsigned srate = (1000.0 / sps) * 1000u;
	double rapl1 = (double) atof(argv[4]);
	double rapl2 = (double) atof(argv[5]);
	DIST_THRESH = (double) atof(argv[6]);
	PCT_THRESH = (double) atof(argv[7]);
	NUM_CPU = atoi(argv[9]);
	MAX_NON_TURBO = (double) atof(argv[10]);
	MAX_PSTATE = (unsigned long) strtol(argv[11], NULL, 16);

	powlim = rapl1;
	if (argv[8][0] == 'c')
	{
		MAN_CTRL = 1;
	}
	if (MAN_CTRL == 1)
	{
		printf("manual control ON\n");
	}
	else
	{
		printf("manual control OFF\n");
	}
	powlim = rapl2;
	printf("rapl limit 1 %lf, rapl limit 2 %lf\n", rapl1, rapl2);

	// these are the values at 800MHz, linear regression model based on frequency is used
	// cpu phase
	prof_class[0].ipc = 0.576;
    prof_class[0].mpc = 0.005;
    prof_class[0].rpc = 0.017;
    prof_class[0].epc = 0.027;
 	prof_class[0].bpc = 0.0006;

	// memory phase
	prof_class[1].ipc = 0.122;
    prof_class[1].mpc = 0.004;
    prof_class[1].rpc = 0.118;
    prof_class[1].epc = 0.427;
 	prof_class[1].bpc = 0.017;

	// IO/sleep phase (derived)
	prof_class[2].ipc = 0;
    prof_class[2].mpc = 0;
    prof_class[2].rpc = 0;
    prof_class[2].epc = 0;
 	prof_class[2].bpc = 0;

	// mixed phase (derived)
	prof_class[3].ipc = 0.349;
    prof_class[3].mpc = 0.005;
    prof_class[3].rpc = 0.06;
    prof_class[3].epc = 0.25;
 	prof_class[3].bpc = 0.005;

	cpu_set_t cpus;
	CPU_ZERO(&cpus);
	// use the next logical CPU after the number of threads in use
	// AKA this puts the sampler program on an unused logical core
	CPU_SET(THREADCOUNT, &cpus);
	sched_setaffinity(0, sizeof(cpus), &cpus);

	fprintf(sreport, "Using paremeters:\n");
	fprintf(sreport, "\tThreads: %u\n", THREADCOUNT);
	fprintf(sreport, "\tTime: %u\n", duration);
	fprintf(sreport, "\tSamples Per Second: %u\n", sps);

	prof_minimums.ipc = DBL_MAX;
	prof_minimums.mpc = DBL_MAX;
	prof_minimums.rpc = DBL_MAX;
	prof_minimums.epc = DBL_MAX;
	prof_minimums.bpc = DBL_MAX;

	uint64_t ovf_ctrl;
	int ctr;
	for (ctr = 0; ctr < THREADCOUNT; ctr++)
	{
		uint64_t perf_global_ctrl;
		uint64_t fixed_ctr_ctrl;
		read_msr_by_coord(0, ctr, 0, IA32_PERF_GLOBAL_CTRL, &perf_global_ctrl);
		read_msr_by_coord(0, ctr, 0, IA32_FIXED_CTR_CTRL, &fixed_ctr_ctrl);
		write_msr_by_coord(0, ctr, 0, IA32_PERF_GLOBAL_CTRL, 
				perf_global_ctrl | (0x7ul << 32) | 0x3);
		write_msr_by_coord(0, ctr, 0, IA32_FIXED_CTR_CTRL, fixed_ctr_ctrl | (0x3));
		write_msr_by_coord(0, ctr, 0, IA32_FIXED_CTR0, 0x0ul);
		read_msr_by_coord(0, ctr, 0, IA32_PERF_GLOBAL_OVF_CTRL, &ovf_ctrl);
		write_msr_by_coord(0, ctr, 0, IA32_PERF_GLOBAL_OVF_CTRL, ovf_ctrl & 0xFFFFFFFFFFFFFFFE);
		write_msr_by_coord(0, ctr, 0, MSR_PKG_PERF_STATUS, 0);
	}

// This is debug info
/*
	uint64_t mod = 0x0;
	read_msr_by_coord(0, 0, 0, 0x19a, &mod);
 fprintf(sreport, "mod %lx\n", mod);
	uint64_t therm = 0x0;
	read_msr_by_coord(0, 0, 0, 0x19c, &therm);
 fprintf(sreport, "therm %lx\n", therm);
	uint64_t power_ctl = 0x0;
	read_msr_by_coord(0, 0, 0, 0x1FC, &power_ctl);
 fprintf(sreport, "powctl %lx\n", power_ctl);
	//power_ctl |= (0x1 << 20);
	//write_msr_by_coord(0, 0, 0, 0x1FC, power_ctl);
*/

	uint64_t unit;
	read_msr_by_coord(0, 0, 0, MSR_RAPL_POWER_UNIT, &unit);
	uint64_t power_unit = unit & 0xF;
	double pu = 1.0 / (0x1 << power_unit);
	//fprintf(stderr, "power unit: %lx\n", power_unit);
	uint64_t seconds_unit_raw = (unit >> 16) & 0x1F;
	double su = 1.0 / (0x1 << seconds_unit_raw);
	seconds_unit = su;
	fprintf(stderr, "seconds unit: %lx\n", seconds_unit);
	unsigned eu = (unit >> 8) & 0x1F;
	energy_unit = 1.0 / (0x1 << eu);

	set_rapl(1, rapl1, pu, su, 0);
	set_rapl2(16, rapl2, pu, su, 0);
	
	set_perf(MAX_PSTATE);

	thread_samples = (struct data_sample **) calloc(THREADCOUNT, sizeof(struct data_sample *));
	FILE **output = (FILE **) calloc(THREADCOUNT, sizeof(FILE *));
	SAMPLECTRS = (unsigned long *) calloc(THREADCOUNT, sizeof(unsigned long));
	unsigned long numsamples = duration * sps;
	char fname[FNAMESIZE];
	int i;
	for (i = 0; i < THREADCOUNT; i++)
	{
		fprintf(sreport, "Allocating for %lu samples\n", (numsamples) + 1);
		thread_samples[i] = (struct data_sample *) calloc((numsamples) + 1, sizeof(struct data_sample));
		if (thread_samples[i] == NULL)
		{
			fprintf(stderr, "ERROR: out of memory\n");
			return -1;
		}
		snprintf((char *) fname, FNAMESIZE, "core%d.msrdat", i);
		output[i] = fopen(fname, "w");
	}
	set_perf(0x2D);
	/* set_perf(MAX_PSTATE); */

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
	//gettimeofday(&current, NULL);
	sample_data(0);
	//sample_data(6);
	usleep(srate);
	//while (durctr < duration)
	while (LOOP_CTRL)
	{
		sample_data(0);
		read_msr_by_coord(0, 0, 0, IA32_PERF_GLOBAL_STATUS, &ovf_stat);
		if (ovf_stat & 0x1)
		{
			read_msr_by_coord(0, 0, 0, IA32_PERF_GLOBAL_OVF_CTRL, &ovf_ctrl);
			write_msr_by_coord(0, 0, 0, IA32_PERF_GLOBAL_OVF_CTRL, ovf_ctrl & 0xFFFFFFFFFFFFFFFE);
			ovf_ctr++;
		}
		pow_aware_perf();
		usleep(srate);
	}
	
	gettimeofday(&current, NULL);

	uint64_t aperf_after, mperf_after;
	read_msr_by_coord(0, 0, 0, MSR_IA32_APERF, &aperf_after);
	read_msr_by_coord(0, 0, 0, MSR_IA32_MPERF, &mperf_after);

	read_msr_by_coord(0, 0, 0, IA32_FIXED_CTR0, &inst_after);
	fprintf(stdout, "Benchmark complete...\n");
	double aruntime = (double) (current.tv_sec - start.tv_sec) + (current.tv_usec - start.tv_usec) / 1000000.0;
	avgrate = aruntime / SAMPLECTRS[0];
	fprintf(sreport, "Actual run time: %f\n", aruntime);
	fprintf(sreport, "Average Sampling Rate: %lf seconds\n", avgrate);
	fprintf(sreport, "Dumping data file(s)...\n");
	fprintf(sreport, "Avg Frq: %f\n", (float) (aperf_after - aperf_before) / (float) (mperf_after - mperf_before) * MAX_NON_TURBO);
	fprintf(sreport, "Instructions: %lu (ovf %u)\n", inst_after - inst_before, ovf_ctr);
	fprintf(sreport, "IPS: %lf (ovf %u)\n", (inst_after - inst_before) / aruntime, ovf_ctr);
	FILE *ins = fopen("instret", "w");
	fprintf(ins, "%lu (ovf%u)\n", inst_after - inst_before, ovf_ctr);
	fclose(ins);

	avg_sample_rate = 0.001;
	dump_data(output, aruntime);
	for (i = 0; i < 0; i++)
	{
		fprintf(sreport, "Thread %d collected %lu samples\n", i, SAMPLECTRS[i]);
		free(thread_samples[i]);
	}

	snprintf((char *) fname, FNAMESIZE, "profiles", i);
	FILE *profout = fopen(fname, "w");
	//dump_phaseinfo(stdout, NULL);
	// TODO: figure out if miscounts from remove_unused or something else
	remove_unused();
	agglomerate_profiles();
	printf("phase results\n");
	dump_phaseinfo(profout, &avgrate);

	//set_rapl(1, 105.0, pu, su, 0);
	free(thread_samples);
	free(output);
	free(SAMPLECTRS);

	finalize_msr();

	fprintf(sreport, "Done...\n");
	fclose(sreport);

	return 0;
}
