// Copyright (C) 2018 The University of Arizona
// Author: Scott Walker
// License: GPLv3. See LICENSE file for details.
// Power Governor: An advanced performance governor that is power aware.


#define _BSD_SOURCE // required for usleep
#define _GNU_SOURCE // required for sched
#include <unistd.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>
#include <sys/sysinfo.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <sched.h>
#include <signal.h>
#include <float.h>
#include <assert.h>

// Libmsr includes
#include "msr_core.h"
#include "master.h"
#include "msr_counters.h"
#include "msr_misc.h"
#include "cpuid.h"


//#define DEBUG
#define CPUID_DEBUG
#define VERSIONSTRING "0.2"
#define HEADERSTRING "########################################"
#define FNAMESIZE 32
#define FRQ_CHANGE_STEP 0.1 // tens of MHz
#define MSR_TURBO_RATIO_LIMIT 0x1AD
#define MSR_CORE_PERF_LIMIT_REASONS 0x64F
#define LIMIT_ON_RAPL 0x0000C00
#define LIMIT_LOG_RAPL 0xC000000
#define LIMIT_LOG_MASK 0xF3FFFFFF
#define MAX_PROFILES 20
#define MAX_HISTORY 8
#define NUM_CLASSES 4
#define FUP_TIMEOUT 100
#define RECLASSIFY_INTERVAL 20
#define SCALE_OUTLIER_THRESH_LOW 0.8
#define SCALE_OUTLIER_THRESH_HIGH 1.2
#define SCALE_THRESH 1.0 // 100MHz
#define MEM_FN1_LIM 5 
#define MEM_FN2_LIM 10
#define MEM_FN3_LIM 20
#define MEM_FNRESET_LIM 250
#define TURBO_FN1_LIM 10 
#define FRQ_DUTY_LENGTH 10

#define CLASS_CPU_SLOPE_IPC 0.63396
#define CLASS_CPU_SLOPE_EPC 0.13005
#define CLASS_MEM_SLOPE_IPC 0.07642
/* #define CLASS_MEM_SLOPE_EPC 0.73337 */
#define CLASS_MEM_SLOPE_EPC 0.79
#define CLASS_MIX_SLOPE_IPC ((CLASS_CPU_SLOPE_IPC + CLASS_MEM_SLOPE_IPC) / 2)
#define CLASS_MIX_SLOPE_EPC ((CLASS_CPU_SLOPE_EPC + CLASS_MEM_SLOPE_EPC) / 2 )
#define CLASS_CPU_INTERCEPT_IPC 0.10806
#define CLASS_CPU_INTERCEPT_EPC -0.12874
#define CLASS_MEM_INTERCEPT_IPC 0.08295
#define CLASS_MEM_INTERCEPT_EPC -0.20863
#define CLASS_MIX_INTERCEPT_IPC ((CLASS_CPU_INTERCEPT_IPC + CLASS_MEM_INTERCEPT_IPC) / 2)
#define CLASS_MIX_INTERCEPT_EPC ((CLASS_CPU_INTERCEPT_EPC + CLASS_MEM_INTERCEPT_EPC) / 2 )

#define ARG_ERROR {\
	if (j + 1 > argc - 1)\
	{\
		printf("Error: value required for argument %s\n", argv[j]);\
		exit(-1);\
	}\
}

#define DOUBLE_FRQ_TO_UINT(dbl) ((uint16_t) (dbl * 10.0))
#define UINT_FRQ_TO_DOUBLE(uit) (((double) uit) / 10.0)

// Structures
enum CLASS_ID
{
	CLASS_CPU,
	CLASS_MEM,
	CLASS_IO,
	CLASS_MIX,
	CLASS_UNKNOWN,
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
	double frq_target; // what frequency in *100MHz the algorithm thinks phase should run at
	double avg_cycle; // average number of cycles it takes to execute this phase
	uint32_t num_throttles; // number of times this phase was throttled last time (aka misprediction)
	uint64_t occurrences; // how many times this phase was detected
	char prev_phases[MAX_HISTORY];
	char lastprev;
	char class;
	char unthrot_count;
	char reclass_count;
	char frq_duty_count;
};

struct powgov_sysconfig
{
	unsigned num_cpu; // number of cores present
	double rapl_energy_unit;
	double rapl_seconds_unit;
	double rapl_power_unit;
	double max_non_turbo;
	double max_pstate;
	double min_pstate;
	unsigned sockets;
	unsigned coresPerSocket;
	unsigned threadsPerCore;
};

struct powgov_files
{
	FILE *sreport; // main report file generated
	FILE **sampler_dumpfiles; // report files for sampler
	FILE *profout; // report file for profile clusters
};

struct powgov_sampler
{
	unsigned long *samplectrs; // counter for number of samples on each thread
	struct data_sample **thread_samples;
	unsigned long numsamples;
	struct data_sample new_sample;
	struct data_sample prev_sample;
	unsigned sps;
};

struct powgov_power
{
	double rapl1;
	double rapl2;
	double window;
	double proc_tdp;
};

struct powgov_classifier
{
	double dist_thresh; // the threshold for profile cluster identification
	double pct_thresh; // the threshold for displaying dumped profiles as execution percent
	int numphases; // the current number of phases
	struct phase_profile profiles[MAX_PROFILES]; // the cluster centers
	struct phase_profile prof_maximums; // minimum sampled values used for scaling and normalization
	struct phase_profile prof_minimums; // minimum sampled values used for scaling and normalization
	struct phase_profile prof_class[NUM_CLASSES]; // pre-computed values for various classes of workload
	int recentphase;
};

struct powgov_runtime
{
	struct powgov_sysconfig *sys;
	struct powgov_files *files;
	struct powgov_sampler *sampler;
	struct powgov_classifier *classifier;
	struct powgov_power *power;
};

// Globals
int LOOP_CTRL = 1;
int MAN_CTRL;
double THREADCOUNT; // deprecated?
double DURATION; // deprecated?
double VERBOSE;
double EXPERIMENTAL;
double REPORT;
int MEM_POW_SHIFT = 1;
int THROTTLE_AVOID = 1;
double MEM_FRQ_OVERRIDE = 0.0;
char CLASS_NAMES[NUM_CLASSES + 1][8] = {"CPU\0", "MEM\0", "IO\0", "MIX\0", "UNK\0"};


// Prototypes
double metric_distance(struct phase_profile *old, struct phase_profile *new, struct phase_profile *maximums,
	struct phase_profile *minimums);
void agglomerate_profiles(struct powgov_runtime *runtime);
void remove_unused(struct powgov_runtime *runtime);
void dump_rapl();
void enable_turbo(struct powgov_runtime *runtime);
void disable_turbo(struct powgov_runtime *runtime);
void set_turbo_limit(unsigned int limit);
void dump_rapl_info(double power_unit);
int sample_data(struct powgov_runtime *runtime);
void update_minmax(struct phase_profile *this_profile, struct phase_profile *maximums, 
		struct phase_profile *minimums);
void print_profile(struct phase_profile *prof);
void update_profile(struct powgov_runtime *runtime, struct phase_profile *this_profile, int profidx, 
		uint64_t perf, unsigned this_throttle, double avgfrq, int lastphase);
void add_profile(struct powgov_runtime *runtime, struct phase_profile *this_profile, uint64_t perf, 
		unsigned this_throttle, double avgfrq, int lastphase);
int classify_phase(struct powgov_runtime *runtime, struct phase_profile *phase, uint64_t perf);
void classify_and_react(struct powgov_runtime *runtime, int phase, char wasthrottled, 
		uint64_t perf);
double ipc_scale(double ipc_unscaled, double frq_source, double frq_target);
void frequency_scale_phase(struct phase_profile *unscaled_profile, double frq_source, double frq_target,
		struct phase_profile *scaled_profile);
int branch_same_phase(struct powgov_runtime *runtime, struct phase_profile *this_profile, 
		uint64_t perf, char wasthrottled, char isthrottled, double phase_avgfrq);
int branch_change_phase(struct powgov_runtime *runtime, struct phase_profile *this_profile, 
		uint64_t perf, char wasthrottled, char isthrottled, double phase_avgfrq);
void pow_aware_perf(struct powgov_runtime *runtime);
void set_perf(struct powgov_runtime *runtime, const unsigned freq);
void dump_phaseinfo(struct powgov_runtime *runtime, FILE *outfile, double *avgrate);
void dump_data(struct powgov_runtime *runtime, FILE **outfile, double durctr);
void set_rapl(double sec, double watts, double pu, double su, unsigned affinity);
void set_rapl2(unsigned sec, double watts, double pu, double su, unsigned affinity);
void hwpstuff();
void signal_exit(int signum);
void init_sampling(struct powgov_runtime *runtime);
void activate_performance_counters(struct powgov_runtime * runtime);
void dump_help();
void dump_config(struct powgov_runtime *runtime, FILE *out);
void dump_sys(struct powgov_runtime *runtime, FILE *out);


// BEGIN PROGRAM
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

// TODO: weighted averaging should scale first
void agglomerate_profiles(struct powgov_runtime *runtime)
{
	struct phase_profile old_profiles[MAX_PROFILES];
	int newidx = 0;
	struct phase_profile *profiles = runtime->classifier->profiles;
	memcpy(old_profiles, profiles, 
			runtime->classifier->numphases * sizeof(struct phase_profile));

	//double dist[MAX_PROFILES];
	//memset(dist, 0, sizeof(double) * MAX_PROFILES);

	char valid[MAX_PROFILES];
	memset(valid, 1, MAX_PROFILES);

	int numcombines = 0;

	//printf("(glom) num phases %d\n", runtime->classifier->numphases);

	int i;
	for (i = 0; i < runtime->classifier->numphases; i++)
	{
		if (valid[i] == 0)
		{
			continue;
		}

		char matches[MAX_PROFILES];
		memset(matches, 0, MAX_PROFILES);
		uint64_t occurrence_sum = old_profiles[i].occurrences;

		int j;
		for (j = 0; j < runtime->classifier->numphases; j++)
		{
			if (i != j && valid[j])
			{
				double dist = metric_distance(&old_profiles[i], &old_profiles[j], 
						&runtime->classifier->prof_maximums, &runtime->classifier->prof_minimums);
				if (dist < runtime->classifier->dist_thresh)
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
			memcpy(&runtime->classifier->profiles[newidx], &old_profiles[i], sizeof(struct phase_profile));
			profiles[newidx].ipc *= (profiles[newidx].occurrences / (double) occurrence_sum);
			profiles[newidx].mpc *= (profiles[newidx].occurrences / (double) occurrence_sum);
			profiles[newidx].rpc *= (profiles[newidx].occurrences / (double) occurrence_sum);
			profiles[newidx].epc *= (profiles[newidx].occurrences / (double) occurrence_sum);
			profiles[newidx].bpc *= (profiles[newidx].occurrences / (double) occurrence_sum);
			profiles[newidx].num_throttles *=
					(profiles[newidx].occurrences / (double) occurrence_sum);
			profiles[newidx].avg_cycle *= (profiles[newidx].occurrences / (double) occurrence_sum);
			profiles[newidx].avg_frq *= (profiles[newidx].occurrences / (double) occurrence_sum);
			int numglom = 1;
			for (j = 0; j < runtime->classifier->numphases; j++)
			{
				if (i != j && matches[j])
				{
					struct phase_profile scaled_profile;
					frequency_scale_phase(&old_profiles[j], old_profiles[j].avg_frq,
							old_profiles[i].avg_frq, &scaled_profile);
					//printf("(glom) combining %d and %d\n", i, j);
					if (j == runtime->classifier->recentphase)
					{
						runtime->classifier->recentphase = newidx;
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
					profiles[newidx].avg_frq += scaled_profile.avg_frq * (scaled_profile.occurrences / (double) occurrence_sum);
					numglom++;
				}
			}
			// TODO: instead of clearing, update prev_phases
			profiles[newidx].lastprev = 0;
			memset(profiles[newidx].prev_phases, -1, MAX_HISTORY);
			newidx++;
		}
	}
	if (numcombines > 0)
	{
		// slide everything over that wasn't combined
		for (i = 0; i < runtime->classifier->numphases; i++)
		{
			if (valid[i] && i == newidx)
			{
				//printf("(glom) skipping id %d\n", i);
				newidx++;
			}
			else if (valid[i])
			{
				//printf("(glom) sliding %d to %d\n", i, newidx);
				if (i == runtime->classifier->recentphase)
				{
					runtime->classifier->recentphase = newidx;
				}
				memcpy(&profiles[newidx], &old_profiles[i], sizeof(struct phase_profile));
				runtime->classifier->profiles[newidx].lastprev = 0;
				memset(profiles[newidx].prev_phases, -1, MAX_HISTORY);

				newidx++;
			}
		}
		runtime->classifier->numphases = newidx;
	}
	//printf("(glom) runtime->classifier->recentphase is now %d\n", runtime->classifier->recentphase);
}

void remove_unused(struct powgov_runtime *runtime)
{
	if (runtime->classifier->numphases <= 0)
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
	struct phase_profile *profiles = runtime->classifier->profiles;

	for (i = 0; i < runtime->classifier->numphases; i++)
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
	for (i = 0; i < runtime->classifier->numphases; i++)
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
	memcpy(old_profiles, profiles, runtime->classifier->numphases * sizeof(struct phase_profile));

	newidx = firstinvalid;
	for (i = firstinvalid; i < runtime->classifier->numphases; i++)
	{
		if (i == newidx && valid[i])
		{
			//printf("(remove) skipping id %d\n", i);
			newidx++;
		}
		else if (valid[i])
		{
			//printf("(remove) sliding id %d to %d\n", i, newidx);
			if (i == runtime->classifier->recentphase)
			{
				runtime->classifier->recentphase = newidx;
			}
			memcpy(&profiles[newidx], &old_profiles[i], sizeof(struct phase_profile));
			newidx++;
		}
	}
 runtime->classifier->numphases = newidx;
	//printf("(remove) runtime->classifier->numphases is now %d\n", runtime->classifier->numphases);
	//printf("(remove) runtime->classifier->recentphase is now %d\n", runtime->classifier->recentphase);
}

void dump_rapl(FILE *out)
{
	uint64_t rapl;
	read_msr_by_coord(0, 0, 0, MSR_PKG_POWER_LIMIT, &rapl);
	fprintf(out, "\trapl is %lx\n", (unsigned long) rapl);
}

void enable_turbo(struct powgov_runtime *runtime)
{
	uint64_t perf_ctl;
	read_msr_by_coord(0, 0, 0, IA32_PERF_CTL, &perf_ctl);
	perf_ctl &= 0xFFFFFFFEFFFFFFFFul;
	int i;
	for (i = 0; i < runtime->sys->num_cpu; i++)
	{
		write_msr_by_coord(0, i, 0, IA32_PERF_CTL, perf_ctl);
		/* write_msr_by_coord(1, i, 0, IA32_PERF_CTL, perf_ctl); */
	}
}

void disable_turbo(struct powgov_runtime *runtime)
{
	uint64_t perf_ctl;
	read_msr_by_coord(0, 0, 0, IA32_PERF_CTL, &perf_ctl);
	perf_ctl |= 0x0000000100000000ul;
	int i;
	for (i = 0; i < runtime->sys->num_cpu; i++)
	{
		write_msr_by_coord(0, i, 0, IA32_PERF_CTL, perf_ctl);
		/* write_msr_by_coord(1, i, 0, IA32_PERF_CTL, perf_ctl); */
	}
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
	fprintf(stderr, "\tRAPL INFO:\n\tTDP %lf (raw %lx)\n",
		(rapl_info & 0xEF) * power_unit, rapl_info & 0xEF);
}

int sample_data(struct powgov_runtime *runtime)
{
	// TODO: core stuff for multiple thread sampling
	int core = 0;
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
	//read_msr_by_coord(0, core, 0, MSR_PKG_PERF_STATUS, &rapl_throttled);
	read_msr_by_coord(0, core, 0, IA32_THERM_STATUS, &therm);
	read_msr_by_coord(0, core, 0, IA32_FIXED_CTR0, &instret);
	read_msr_by_coord(0, core, 0, MSR_CORE_PERF_LIMIT_REASONS, &perflimit);
	read_batch(COUNTERS_DATA);
	unsigned long idx = runtime->sampler->samplectrs[core]++;
	if (EXPERIMENTAL)
	{
		if (runtime->sampler->samplectrs[core] >= runtime->sampler->numsamples)
		{
			return -1;
		}
		runtime->sampler->prev_sample = runtime->sampler->new_sample;
		runtime->sampler->thread_samples[core][idx].frq_data = perf;
		runtime->sampler->thread_samples[core][idx].tsc_data = tsc;
		runtime->sampler->thread_samples[core][idx].energy_data = energy & 0xFFFFFFFF;
		runtime->sampler->thread_samples[core][idx].rapl_throttled = rapl_throttled & 0xFFFFFFFF;
		runtime->sampler->thread_samples[core][idx].therm = therm;
		runtime->sampler->thread_samples[core][idx].perflimit = perflimit;
		runtime->sampler->thread_samples[core][idx].instret = instret;
		runtime->sampler->thread_samples[core][idx].llcmiss = *pmcounters->pmc0[0];
		runtime->sampler->thread_samples[core][idx].restalls = *pmcounters->pmc1[0];
		runtime->sampler->thread_samples[core][idx].exstalls = *pmcounters->pmc2[0];
		runtime->sampler->thread_samples[core][idx].branchret = *pmcounters->pmc3[0];
		runtime->sampler->new_sample = runtime->sampler->thread_samples[core][idx];
	}
	else
	{
		runtime->sampler->prev_sample = runtime->sampler->new_sample;
		runtime->sampler->new_sample.frq_data = perf;
		runtime->sampler->new_sample.tsc_data = tsc;
		runtime->sampler->new_sample.energy_data = energy & 0xFFFFFFFF;
		runtime->sampler->new_sample.rapl_throttled = rapl_throttled & 0xFFFFFFFF;
		runtime->sampler->new_sample.therm = therm;
		runtime->sampler->new_sample.perflimit = perflimit;
		runtime->sampler->new_sample.instret = instret;
		runtime->sampler->new_sample.llcmiss = *pmcounters->pmc0[0];
		runtime->sampler->new_sample.restalls = *pmcounters->pmc1[0];
		runtime->sampler->new_sample.exstalls = *pmcounters->pmc2[0];
		runtime->sampler->new_sample.branchret = *pmcounters->pmc3[0];
	}
	return 0;
}

void update_minmax(struct phase_profile *this_profile, struct phase_profile *maximums, 
		struct phase_profile *minimums)
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

void update_profile(struct powgov_runtime *runtime, struct phase_profile *this_profile, int profidx, 
		uint64_t perf, unsigned this_throttle, double avgfrq, int lastphase)
{
	if (profidx > runtime->classifier->numphases)
	{
		printf("ERROR: profile does not exist\n");
		return;
	}
	struct phase_profile *profiles = runtime->classifier->profiles;

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

	if (perf > profiles[profidx].frq_high)
	{
		profiles[profidx].frq_high = perf;
	}
	if (perf < profiles[profidx].frq_low)
	{
		profiles[profidx].frq_low = perf;
	}

	profiles[profidx].avg_frq = (profiles[profidx].avg_frq *
			(profiles[profidx].occurrences / (profiles[profidx].occurrences + 1.0))) +
			(avgfrq * (1.0 / (profiles[profidx].occurrences + 1.0)));
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

void add_profile(struct powgov_runtime *runtime, struct phase_profile *this_profile, uint64_t perf, 
		unsigned this_throttle, double avgfrq, int lastphase)
{
	int numphases = runtime->classifier->numphases;
	struct phase_profile *profiles = runtime->classifier->profiles;
	profiles[numphases].ipc = this_profile->ipc;
	profiles[numphases].mpc = this_profile->mpc;
	profiles[numphases].rpc = this_profile->rpc;
	profiles[numphases].epc = this_profile->epc;
	profiles[numphases].bpc = this_profile->bpc;

	profiles[numphases].avg_frq = avgfrq;
	profiles[numphases].frq_high = runtime->sys->min_pstate;
	profiles[numphases].frq_low = runtime->sys->max_pstate;
	profiles[numphases].frq_target = (double) runtime->sys->max_pstate;
	profiles[numphases].avg_cycle = 0;
	profiles[numphases].num_throttles = this_throttle;
	profiles[numphases].occurrences = 0;
	memset(profiles[numphases].prev_phases, -1, MAX_HISTORY);
	profiles[numphases].prev_phases[0] = lastphase;
	profiles[numphases].lastprev = 0;
	profiles[numphases].class = 4;
	profiles[numphases].unthrot_count = 0;
	profiles[numphases].reclass_count = RECLASSIFY_INTERVAL;
	profiles[numphases].frq_duty_count = 0;
	/* profiles[numphases].mem_fn1_ctr = 0; */
	/* profiles[numphases].mem_fn2_ctr = 0; */
	/* profiles[numphases].mem_fn3_ctr = 0; */
	/* profiles[numphases].mem_fnreset_ctr = 0; */

	 runtime->classifier->numphases++;
#ifdef DEBUG
	printf("Added new phase profile %d\n", runtime->classifier->numphases);
#endif
}

int classify_phase(struct powgov_runtime *runtime, struct phase_profile *phase, uint64_t perf)
{
	int i = -1;
	int minidx = -1;
	double mindist = DBL_MAX;
	double freq = ((double) perf) / 10.0;
	struct phase_profile *prof_class = runtime->classifier->prof_class;
	prof_class[0].ipc = freq * CLASS_CPU_SLOPE_IPC + CLASS_CPU_INTERCEPT_IPC;
	prof_class[0].epc = freq * CLASS_CPU_SLOPE_EPC + CLASS_CPU_INTERCEPT_EPC;
	prof_class[1].ipc = freq * CLASS_MEM_SLOPE_IPC + CLASS_MEM_INTERCEPT_IPC;
	prof_class[1].epc = freq * CLASS_MEM_SLOPE_EPC + CLASS_MEM_INTERCEPT_EPC;
	// skip IO class, it doesn't change much
	prof_class[3].ipc = freq * CLASS_MIX_SLOPE_IPC + CLASS_MIX_INTERCEPT_IPC;
	prof_class[3].epc = freq * CLASS_MIX_SLOPE_EPC + CLASS_MIX_INTERCEPT_EPC;
	for (i = 0; i < NUM_CLASSES; i++)
	{
		double dist = metric_distance(phase, &prof_class[i], &runtime->classifier->prof_maximums, &runtime->classifier->prof_minimums);
		if (dist < mindist)
		{
			mindist = dist;
			minidx = i;
		}
	}
	phase->class = minidx;
	phase->reclass_count = 0;
#ifdef DEBUG
	if (phase->epc > 2.5)
	{
		printf("CLASS %d freq %lf\n", phase->class, freq);
		printf("phase ipc %lf epc %lf\n", phase->ipc, phase->epc);
		printf("cpu scale ipc %lf epc %lf\n", prof_class[0].ipc, prof_class[0].epc);
		printf("mem scale ipc %lf epc %lf\n", prof_class[1].ipc, prof_class[1].epc);
		printf("mix scale ipc %lf epc %lf\n", prof_class[2].ipc, prof_class[2].epc);
	}
#endif
	return minidx;
}

void classify_and_react(struct powgov_runtime *runtime, int phase, char wasthrottled, 
		uint64_t perf)
{
	int class;
	int itr;
	double exact_target;
	double ratio;
	// make these not associated with phase, because all we care about is if it is MEM or not
	// these should only reset after a CPU phase
	static unsigned mem_fnreset_ctr = 0;
	static unsigned mem_fn1_ctr = 0;
	static unsigned mem_fn2_ctr = 0;
	static unsigned mem_fn3_ctr = 0;
	struct phase_profile *profiles = runtime->classifier->profiles;
	// avoid reclassifying every timestep
	if (profiles[phase].reclass_count >= RECLASSIFY_INTERVAL)
	{
		class = classify_phase(runtime, &profiles[phase], perf);
	}
	else
	{
		class = profiles[phase].class;
		profiles[phase].reclass_count++;
	}
	if (MAN_CTRL)
	{
		if (class == CLASS_CPU)
		{
			mem_fnreset_ctr = 0;
			mem_fn1_ctr = 0;
			mem_fn2_ctr = 0;
			mem_fn3_ctr = 0;
		}
		if (wasthrottled && THROTTLE_AVOID)
		{
			set_perf(runtime, (uint16_t) (profiles[phase].frq_target - FRQ_CHANGE_STEP * 2.0));
			if (perf < profiles[phase].frq_target)
			{
				profiles[phase].frq_target -= FRQ_CHANGE_STEP * 2.0;
			}
			profiles[phase].unthrot_count = 0;
		}
		else if (class == CLASS_MEM)
		{
			if (MEM_POW_SHIFT)
			{
				// power shifting heuristic
				if (mem_fn1_ctr < MEM_FN1_LIM)
				{
					profiles[phase].frq_target = (double) 0x22;
					mem_fn1_ctr++;
					set_perf(runtime, (uint16_t) profiles[phase].frq_target);
				}
				else if (mem_fn2_ctr < MEM_FN2_LIM)
				{
					profiles[phase].frq_target = (double) 0x28;
					mem_fn2_ctr++;
					set_perf(runtime, (uint16_t) profiles[phase].frq_target);
				}
				else if (mem_fn3_ctr < MEM_FN3_LIM)
				{
					profiles[phase].frq_target = (double) 0x2B;
					mem_fn3_ctr++;
					set_perf(runtime, (uint16_t) profiles[phase].frq_target);
				}
				else
				{
					profiles[phase].frq_target = (double) runtime->sys->max_pstate;
					set_perf(runtime, (uint16_t) profiles[phase].frq_target);
				}
			}
			else if (MEM_FRQ_OVERRIDE != 0.0)
			{
				// power shifting with user override
				profiles[phase].frq_target = MEM_FRQ_OVERRIDE * 10.0;
				set_perf(runtime, (uint16_t) profiles[phase].frq_target);
			}
			else
			{
				// default behavior
				profiles[phase].frq_target = (double) runtime->sys->max_pstate;
				set_perf(runtime, (double) profiles[phase].frq_target);
			}
		}
		else
		{
			if (perf > profiles[phase].frq_target && profiles[phase].frq_target < runtime->sys->max_pstate)
			{
				profiles[phase].frq_target += FRQ_CHANGE_STEP;
			}
			if (profiles[phase].frq_target < runtime->sys->max_pstate &&
				profiles[phase].unthrot_count >= FUP_TIMEOUT)
			{
				profiles[phase].frq_target += FRQ_CHANGE_STEP;
				// TODO may want to have separate counter for this, timeout before each freq increase
				// although not wrong because self throttling...
				profiles[phase].unthrot_count = 0;
			}
			else
			{
				profiles[phase].unthrot_count++;
			}
			double ratio = profiles[phase].frq_target - (double)((uint16_t) profiles[phase].frq_target);
			profiles[phase].frq_duty_count++;
			if (profiles[phase].frq_duty_count < ratio * FRQ_DUTY_LENGTH)
			{
				// set perf to ciel
				set_perf(runtime, (uint16_t) profiles[phase].frq_target + 1);
			}
			else
			{
				// set perf to floor
				set_perf(runtime, (uint16_t) profiles[phase].frq_target);
			}
			if (profiles[phase].frq_duty_count > FRQ_DUTY_LENGTH)
			{
				profiles[phase].frq_duty_count = 0;
			}
			/* set_perf(runtime, profiles[phase].frq_target); */
		}
	}
}

// TODO: unused func
double ipc_scale(double ipc_unscaled, double frq_source, double frq_target)
{
	if (frq_target + SCALE_THRESH > frq_source && frq_target - SCALE_THRESH < frq_source)
	{
		return ipc_unscaled;
	}
	double cpuipc = frq_source * CLASS_CPU_SLOPE_IPC + CLASS_CPU_INTERCEPT_IPC;
	double memipc = frq_source * CLASS_MEM_SLOPE_IPC + CLASS_MEM_INTERCEPT_IPC;
	double ipc_percent = (ipc_unscaled - memipc) / (cpuipc - memipc);
	double result = ((ipc_percent * CLASS_CPU_SLOPE_IPC) + ((1.0 - ipc_percent) * CLASS_MEM_SLOPE_IPC)) *
		frq_target + (ipc_percent * CLASS_CPU_INTERCEPT_IPC) + ((1.0 - ipc_percent) * CLASS_MEM_SLOPE_IPC);

	if (result < 0.0)
	{
		result = 0.0;
	}

	return result;
}

// TODO: keep working on scaling accuracy
void frequency_scale_phase(struct phase_profile *unscaled_profile, double frq_source, double frq_target,
		struct phase_profile *scaled_profile)
{
	*scaled_profile = *unscaled_profile;
	// if the frequencies are already close then just return the copy, don't scale
	if (frq_target + SCALE_THRESH > frq_source && frq_target - SCALE_THRESH < frq_source)
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

	// if the percentages do not add close to 1, then this data point is atypical so don't scale it
	if (ipc_percent + epc_percent < SCALE_OUTLIER_THRESH_LOW || ipc_percent + epc_percent > SCALE_OUTLIER_THRESH_HIGH)
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
#ifdef DEBUG
	printf("\nfrq source %lf, frq target %lf\n", frq_source, frq_target);
	printf("ipc scale from %lf to %lf\nepc scaled from %lf to %lf\n", unscaled_profile->ipc, scaled_profile->ipc,
			unscaled_profile->epc, scaled_profile->epc);
	printf("ipc percent %lf, epc percent %lf\n", ipc_percent, epc_percent);
#endif
}

int branch_same_phase(struct powgov_runtime *runtime, struct phase_profile *this_profile, 
		uint64_t perf, char wasthrottled, char isthrottled, double phase_avgfrq)
{
	if (runtime->classifier->recentphase < 0)
	{
		return 0;
	}

	struct phase_profile scaled_profile = *this_profile;
	double dist_to_recent;
	struct phase_profile *profiles = runtime->classifier->profiles;
	frequency_scale_phase(this_profile, phase_avgfrq, profiles[runtime->classifier->recentphase].avg_frq, &scaled_profile);
	dist_to_recent = metric_distance(&scaled_profile, &profiles[runtime->classifier->recentphase], &runtime->classifier->prof_maximums, &runtime->classifier->prof_minimums);

	if (dist_to_recent < runtime->classifier->dist_thresh)
	{
#ifdef DEBUG
		printf("Phase has not changed, dist to recent %lf\n", dist_to_recent);
#endif
		// we are in the same phase, update it
		update_profile(runtime, &scaled_profile, runtime->classifier->recentphase, perf, wasthrottled, phase_avgfrq, runtime->classifier->recentphase);
		classify_and_react(runtime, runtime->classifier->recentphase, wasthrottled, perf);
		return 1;
	}
#ifdef DEBUG
	printf("Phase has changed, dist to recent %lf\n", dist_to_recent);
	printf("\trecent phase id %d: ipc %lf, epc %lf, freq %lf. new phase: ipc %lf, epc %lf, freq %lf\n",
		runtime->classifier->recentphase, profiles[runtime->classifier->recentphase].ipc, profiles[runtime->classifier->recentphase].epc, profiles[runtime->classifier->recentphase].avg_frq,
		this_profile->ipc, this_profile->epc, phase_avgfrq);
#endif
	return 0;
}

int branch_change_phase(struct powgov_runtime *runtime, struct phase_profile *this_profile, 
		uint64_t perf, char wasthrottled, char isthrottled, double phase_avgfrq)
{
	// we are in a different phase so search for the best match
	double distances[MAX_PROFILES];
	int k;
	struct phase_profile *profiles = runtime->classifier->profiles;
	for (k = 0; k < runtime->classifier->numphases; k++)
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
	for (i = 0; i < runtime->classifier->numphases; i++)
	{
		// measure the distance to every known phase
		if (i == runtime->classifier->recentphase)
		{
			continue;
		}
		struct phase_profile scaled_profile;
		frequency_scale_phase(this_profile, phase_avgfrq, profiles[i].avg_frq, &scaled_profile);
		distances[i] = metric_distance(&scaled_profile, &profiles[i], &runtime->classifier->prof_maximums, &runtime->classifier->prof_minimums);

		if (distances[i] < min_dist)
		{
			min_dist = distances[i];
			min_idx = i;
			min_scaled_profile = scaled_profile;
		}
		if (distances[i] < runtime->classifier->dist_thresh)
		{
			numunder++;
		}
	}

#ifdef DEBUG
	int q;
	for (q = 0; q < runtime->classifier->numphases; q++)
	{
		printf("distance from %d to %d is %lf\n", runtime->classifier->numphases, q, distances[q]);
	}
#endif

	if (min_idx >= 0 && min_dist < runtime->classifier->dist_thresh)
	{
		// we found an existing phase which matches the currently executing workload
#ifdef DEBUG
		printf("Found existing phase, with distance %lf\n", min_dist);
#endif
		update_profile(runtime, &min_scaled_profile, min_idx, perf, isthrottled, phase_avgfrq, runtime->classifier->recentphase);
		runtime->classifier->recentphase = min_idx;
		classify_and_react(runtime, runtime->classifier->recentphase, wasthrottled, perf);
	}
	else
	{
		// the currently executing workload has never been seen before
		if (runtime->classifier->numphases >= MAX_PROFILES)
		{
			printf("ERROR: out of profile storage, increase the limit or change the sensitivity\n");
			return -1;
		}
		// add new phase
		add_profile(runtime, this_profile, perf, isthrottled, phase_avgfrq, runtime->classifier->recentphase);
		runtime->classifier->recentphase = runtime->classifier->numphases - 1;
	}

	// if there are many matches, we should combine similar phases
	if (numunder > 0)
	{
		agglomerate_profiles(runtime);
	}
}

void pow_aware_perf(struct powgov_runtime *runtime)
{
	static uint64_t begin_aperf = 0, begin_mperf = 0;
	static uint64_t last_aperf = 0, last_mperf = 0;
	static uint64_t tsc_timer = 0;
	static uint64_t phase_start_tsc = 0;
	struct phase_profile *profiles = runtime->classifier->profiles;

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

	uint64_t this_instret = runtime->sampler->new_sample.instret - runtime->sampler->prev_sample.instret;
	uint64_t this_cycle = runtime->sampler->new_sample.tsc_data - runtime->sampler->prev_sample.tsc_data;
	unsigned this_throttle = (runtime->sampler->new_sample.rapl_throttled & 0xFFFFFFFF) -
			(runtime->sampler->prev_sample.rapl_throttled & 0xFFFFFFFF);
	uint64_t this_llcmiss = runtime->sampler->new_sample.llcmiss - runtime->sampler->prev_sample.llcmiss;
	uint64_t this_restalls = runtime->sampler->new_sample.restalls - runtime->sampler->prev_sample.restalls;
	uint64_t this_exstalls = runtime->sampler->new_sample.exstalls - runtime->sampler->prev_sample.exstalls;
	uint64_t this_branchret = runtime->sampler->new_sample.branchret - runtime->sampler->prev_sample.branchret;

	double total_avgfrq = ((double) (aperf - begin_aperf) / (double) (mperf - begin_mperf)) * runtime->sys->max_non_turbo;
	double phase_avgfrq = ((double) (aperf - last_aperf) / (double) (mperf - last_mperf)) * runtime->sys->max_non_turbo;
	uint64_t perf = ((runtime->sampler->new_sample.frq_data & 0xFFFFul) >> 8);

	struct phase_profile this_profile;

	this_profile.ipc = ((double) this_instret) / ((double) this_cycle); // instructions per cycle
	this_profile.mpc = ((double) this_llcmiss) / ((double) this_cycle); // cache misses per cycle
	this_profile.rpc = ((double) this_restalls) / ((double) this_cycle); // resource stalls per cycle
	this_profile.epc = ((double) this_exstalls) / ((double) this_cycle); // execution stalls per cycle
	this_profile.bpc = ((double) this_branchret) / ((double) this_cycle); // branch instructions retired per cycle

	update_minmax(&this_profile, &runtime->classifier->prof_maximums, 
			&runtime->classifier->prof_minimums);

	if (runtime->classifier->numphases >= MAX_PROFILES)
	{
		remove_unused(runtime);
	}

	if (runtime->classifier->numphases == 0)
	{
		add_profile(runtime, &this_profile, perf, 0, phase_avgfrq, runtime->classifier->recentphase);
		return;
	}
	// we do phase analysis
	// if current execution is similar to previously seen phase, update that phase
	// check to see if we are in the same phase
	if (runtime->classifier->recentphase > runtime->classifier->numphases)
	{
		//printf("recent phase no longer exists\n");
		runtime->classifier->recentphase = -1;
	}

	uint64_t limreasons = runtime->sampler->new_sample.perflimit;
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
	if (branch_same_phase(runtime, &this_profile, perf, wasthrottled, isthrottled, phase_avgfrq))
	{
		return;
	}
	// the phase changed
	if (profiles[runtime->classifier->recentphase].avg_cycle < (runtime->sampler->new_sample.tsc_data - phase_start_tsc))
	{
		profiles[runtime->classifier->recentphase].avg_cycle = (runtime->sampler->new_sample.tsc_data - phase_start_tsc);
	}
	phase_start_tsc = runtime->sampler->new_sample.tsc_data;
	last_aperf = aperf;
	last_mperf = mperf;
	branch_change_phase(runtime, &this_profile, perf, wasthrottled, isthrottled, phase_avgfrq);
}

void set_perf(struct powgov_runtime *runtime, const unsigned freq)
{
	uint64_t perf_ctl = 0x0ul;
	uint64_t freq_mask = freq;
	read_msr_by_coord(0, 0, 0, IA32_PERF_CTL, &perf_ctl);
	perf_ctl &= 0xFFFFFFFFFFFF0000ul;
	freq_mask <<= 8;
	perf_ctl |= freq_mask;
	//write_msr_by_coord(0, tid, 0, IA32_PERF_CTL, perf_ctl);
	//write_msr_by_coord(0, tid, 1, IA32_PERF_CTL, perf_ctl);
	int i;
	for (i = 0; i < runtime->sys->num_cpu; i++)
	{
		write_msr_by_coord(0, i, 0, IA32_PERF_CTL, perf_ctl);
		//write_msr_by_coord(0, i, 1, IA32_PERF_CTL, perf_ctl);
		write_msr_by_coord(1, i, 0, IA32_PERF_CTL, perf_ctl);
		//write_msr_by_coord(1, i, 1, IA32_PERF_CTL, perf_ctl);
	}
}

void dump_phaseinfo(struct powgov_runtime *runtime, FILE *outfile, double *avgrate)
{
	int i;
	uint64_t recorded_steps = 0;
	struct phase_profile *profiles = runtime->classifier->profiles;
	for (i = 0; i < runtime->classifier->numphases; i++)
	{
		recorded_steps += profiles[i].occurrences;
	}
	double totaltime = 0.0;
	double totalpct = 0.0;
	for (i = 0; i < runtime->classifier->numphases; i++)
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
		totalpct = (double) recorded_steps / (double) runtime->sampler->samplectrs[0];
		fprintf(outfile, "TOTAL\t\t%.2lf\t(%3.2lf%% accounted)\n", totaltime, totalpct * 100.0);
	}
	fprintf(outfile, "min instructions per cycle        %lf\n", runtime->classifier->prof_minimums.ipc);
	fprintf(outfile, "min LLC misses per cycle          %lf\n", runtime->classifier->prof_minimums.mpc);
	fprintf(outfile, "min resource stalls per cycle     %lf\n", runtime->classifier->prof_minimums.rpc);
	fprintf(outfile, "min execution stalls per cycle    %lf\n", runtime->classifier->prof_minimums.epc);
	fprintf(outfile, "min branch instructions per cycle %lf\n", runtime->classifier->prof_minimums.bpc);
	fprintf(outfile, "max instructions per cycle        %lf\n", runtime->classifier->prof_maximums.ipc);
	fprintf(outfile, "max LLC misses per cycle          %lf\n", runtime->classifier->prof_maximums.mpc);
	fprintf(outfile, "max resource stalls per cycle     %lf\n", runtime->classifier->prof_maximums.rpc);
	fprintf(outfile, "max execution stalls per cycle    %lf\n", runtime->classifier->prof_maximums.epc);
	fprintf(outfile, "max branch instructions per cycle %lf\n", runtime->classifier->prof_maximums.bpc);
	for (i = 0; i < runtime->classifier->numphases; i++)
	{
		// ignore phases that are less than x% of program
		double pct = (double) profiles[i].occurrences / (double) recorded_steps;
		if (pct < runtime->classifier->pct_thresh)
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
		fprintf(outfile, "\n\tavg frq     %lf\n", profiles[i].avg_frq / 10.0);
		fprintf(outfile, "\tfrq low       %x\n", profiles[i].frq_low);
		fprintf(outfile, "\tfrq high      %x\n", profiles[i].frq_high);
		fprintf(outfile, "\tfrq target    %lf\n", profiles[i].frq_target / 10.0);
		fprintf(outfile, "\tavg cycles    %lf (%lf seconds)\n", profiles[i].avg_cycle,
				profiles[i].avg_cycle / (profiles[i].avg_frq * 1000000000.0));
		fprintf(outfile, "\tnum throttles %u\n", profiles[i].num_throttles);
		fprintf(outfile, "\tclass %s\n", CLASS_NAMES[profiles[i].class]);

		int j;
		for (j = 0; j < runtime->classifier->numphases; j++)
		{
			double lpct = (double) profiles[j].occurrences / (double) recorded_steps;
			if (lpct > runtime->classifier->pct_thresh)
			{
				struct phase_profile scaled_profile;
				frequency_scale_phase(&profiles[j], profiles[j].avg_frq, profiles[i].avg_frq, &scaled_profile);
				double dist = metric_distance(&scaled_profile, &profiles[i], &runtime->classifier->prof_maximums, &runtime->classifier->prof_minimums);
				fprintf(outfile, "\tdistance from %d: %lf\n", j, dist);
			}
		}
	}
}

void dump_data(struct powgov_runtime *runtime, FILE **outfile, double durctr)
{
	unsigned long total_post = 0;
	unsigned long mid_pow = 0;
	struct data_sample **thread_samples = runtime->sampler->thread_samples;
	unsigned long total_pre = thread_samples[0][2].energy_data;
	int powovf = 0;
	int j;
	for (j = 0; j < THREADCOUNT; j++)
	{
		fprintf(outfile[j], "freq\tp-state\ttsc\tpower\trapl-throttle-cycles\tTemp(C)\tCORE_PERF_LIMIT_REASONS\tinstret\tllcmiss\tresource-stall\texec-stall\tbranch-retired\n");
		unsigned long i;
		for (i = 0; i < runtime->sampler->samplectrs[j]; i++)
		{
			double time = (thread_samples[j][i + 1].tsc_data - thread_samples[j][i].tsc_data) /
				((((thread_samples[j][i].frq_data & 0xFFFFul) >> 8) / 10.0) * 1000000000.0);

			unsigned long diff = (thread_samples[j][i + 1].energy_data -
				thread_samples[j][i].energy_data);


			fprintf(outfile[j], "%f\t%llx\t%llu\t%lf\t%lu\t%u\t%lx\t%llu\t%lu\t%lu\t%lu\t%lu\n",
				((thread_samples[j][i].frq_data & 0xFFFFul) >> 8) / 10.0,
				(unsigned long long) (thread_samples[j][i].frq_data & 0xFFFFul),
				//(unsigned long long) (thread_samples[j][i].tsc_data),
				(unsigned long long) (thread_samples[j][i + 1].tsc_data - 
					thread_samples[j][i].tsc_data),
				diff * runtime->sys->rapl_energy_unit / time,
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
			if (i == runtime->sampler->samplectrs[j] / 2)
			{
				mid_pow = (unsigned long) thread_samples[j][i].energy_data;
			}
		}
	}
	total_post = thread_samples[0][runtime->sampler->samplectrs[0] - 1].energy_data;

	fprintf(runtime->files->sreport, "total power %lf\n", ((total_post - total_pre) * 
				runtime->sys->rapl_energy_unit) / durctr);
	fprintf(runtime->files->sreport, "mid power %lf\n", ((total_post - mid_pow) * 
				runtime->sys->rapl_energy_unit) / (durctr / 2.0));
	//fprintf(runtime->files->sreport, "hex pre %lx hex post %lx\n", total_pre, total_post);
	fprintf(runtime->files->sreport, "total power ovf %lf\n", (0xEFFFFFFF * powovf - total_pre + total_post) / durctr);
	fprintf(runtime->files->sreport, "energy ctr overflows: %d\n", powovf);
}

void set_rapl(double sec, double watts, double pu, double su, unsigned affinity)
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

void hwpstuff(FILE *out)
{
	uint64_t rax, rbx, rcx, rdx;
	rax = rbx = rcx = rdx = 0;
	cpuid(0x6, &rax, &rbx, &rcx, &rdx);
	if (rax & (0x1ul << 7) == 0)
	{
		fprintf(out, "\thwp is not supported or disabled\n");
		return;
	}

	uint64_t hwp = 0x0;
	read_msr_by_coord(0, 0, 0, 0x770, &hwp);
	fprintf(out, "\thwp %lx %s\n", hwp, (hwp == 0 ? "disabled" : "enabled"));
	if (hwp != 0)
	{
		uint64_t hwp_cap = 0x0;
		read_msr_by_coord(0, 0, 0, 0x771, &hwp_cap);
		fprintf( out, "\thwp cap %lx\n", hwp_cap);
		uint64_t hwp_req = 0x0;
		read_msr_by_coord(0, 0, 0, 0x772, &hwp_req);
		fprintf( out, "\thwp req %lx\n", hwp_req);
		uint64_t hwp_int = 0x0;
		read_msr_by_coord(0, 0, 0, 0x773, &hwp_int);
		fprintf( out, "\thwp int %lx\n", hwp_int);
		uint64_t hwp_log_req = 0x0;
		read_msr_by_coord(0, 0, 0, 0x774, &hwp_log_req);
		fprintf( out, "\thwp log req %lx\n", hwp_log_req);
		uint64_t hwp_stat = 0x0;
		read_msr_by_coord(0, 0, 0, 0x777, &hwp_stat);
		fprintf( out, "\thwp stat %lx\n", hwp_stat);

		hwp_req = 0x14150Alu;
		hwp_log_req = 0x14150Alu;
		//write_msr_by_coord(0, 0, 0, 0x772, hwp_req);
		//write_msr_by_coord(0, 0, 0, 0x772, hwp_req);
		read_msr_by_coord(0, 0, 0, 0x772, &hwp_req);
		fprintf( out, "\thwp req %lx\n", hwp_req);
		read_msr_by_coord(0, 0, 0, 0x774, &hwp_log_req);
		fprintf( out, "\thwp log req %lx\n", hwp_log_req);
	}
}

void signal_exit(int signum)
{
	fprintf(stderr, "Sampler terminating...\n");
	fflush(stderr);
	LOOP_CTRL = 0;
	return;
}

void init_sampling(struct powgov_runtime *runtime)
{
	runtime->sampler->thread_samples = (struct data_sample **) calloc(THREADCOUNT, sizeof(struct data_sample *));
	runtime->files->sampler_dumpfiles = (FILE **) calloc(THREADCOUNT, sizeof(FILE *));
	runtime->sampler->numsamples = DURATION * runtime->sampler->sps * 1.1;
	char fname[FNAMESIZE];
	int i;
	for (i = 0; i < THREADCOUNT; i++)
	{
		fprintf(runtime->files->sreport, "Allocating for %lu samples\n", 
				(runtime->sampler->numsamples) + 1);
		runtime->sampler->thread_samples[i] = (struct data_sample *) 
			calloc((runtime->sampler->numsamples) + 1, sizeof(struct data_sample));
		if (runtime->sampler->thread_samples[i] == NULL)
		{
			fprintf(stderr, "ERROR: out of memory\n");
			exit(-1);
		}
		snprintf((char *) fname, FNAMESIZE, "core%d.msrdat", i);
		runtime->files->sampler_dumpfiles[i] = fopen(fname, "w");
	}
}

void activate_performance_counters(struct powgov_runtime * runtime)
{
	set_all_pmc_ctrl(0x0, 0x43, 0x41, 0x2E, 1); // LLC miss
	set_all_pmc_ctrl(0x0, 0x43, 0x01, 0xA2, 2); // resource stalls
	set_all_pmc_ctrl(0x0, 0x43, 0x04, 0xA3, 3); // execution stalls
	set_all_pmc_ctrl(0x0, 0x43, 0x00, 0xC4, 4); // branch instructions retired
	//set_all_pmc_ctrl(0x0, 0x43, 0x02, 0xC7, 3); // SSE/AVX single precision retired
	//set_all_pmc_ctrl(0x0, 0x43, 0x01, 0xC7, 4); // SSE/AVX double precision retired
	//set_all_pmc_ctrl(0x0, 0x43, 0x04, 0xC5, 3); // branch misses retired
	enable_pmc();
	// enable fixed counters
	uint64_t ovf_ctrl;
	int ctr;
	for (ctr = 0; ctr < runtime->sys->num_cpu; ctr++)
	{
		uint64_t perf_global_ctrl;
		uint64_t fixed_ctr_ctrl;
		read_msr_by_coord(0, ctr, 0, IA32_PERF_GLOBAL_CTRL, &perf_global_ctrl);
		read_msr_by_coord(0, ctr, 0, IA32_FIXED_CTR_CTRL, &fixed_ctr_ctrl);
		write_msr_by_coord(0, ctr, 0, IA32_PERF_GLOBAL_CTRL, perf_global_ctrl | (0x7ul << 32) | 0x3);
		write_msr_by_coord(0, ctr, 0, IA32_FIXED_CTR_CTRL, fixed_ctr_ctrl | (0x3));
		write_msr_by_coord(0, ctr, 0, IA32_FIXED_CTR0, 0x0ul);
		read_msr_by_coord(0, ctr, 0, IA32_PERF_GLOBAL_OVF_CTRL, &ovf_ctrl);
		write_msr_by_coord(0, ctr, 0, IA32_PERF_GLOBAL_OVF_CTRL, ovf_ctrl & 0xFFFFFFFFFFFFFFFE);
		//write_msr_by_coord(0, ctr, 0, MSR_PKG_PERF_STATUS, 0);
	}
}

void dump_help()
{
	char validargs[] = "rlwLtdsveRhSAMO";
	printf("Valid options:\n");
	printf("\t-r: polling rate in samples per second\n");
	printf("\t-l: RAPL limit\n");
	printf("\t-L: hard RAPL limit\n");
	printf("\t-w: RAPL time window\n");
	printf("\t-t: clustering sensitivity threshold\n");
	printf("\t-d: display cutoff for profiles, if argument is missing then profiles are not dumped\n");
	printf("\t-s: system control, do sampling and profiling only\n");
	printf("\t-v: verbose\n");
	printf("\t-e: experimental feature enable\n");
	printf("\t-R: enable reporting\n");
	printf("\t-h: display this menu\n");
	printf("\t-S: disable memory power shifting\n");
	printf("\t-A: disable throttle avoidance\n");
	printf("\t-O: disable overpower\n");
	printf("\t-M: manual frequency override for memory phase (requires -S)\n");
}

void dump_config(struct powgov_runtime *runtime, FILE *out)
{
	fprintf(out, "Power Governor Configuration:\n");
	fprintf(out, "\trapl limit 1 %lf\n\trapl limit 2 %lf\n\trapl limit 1 time window %lf\n", 
			runtime->power->rapl1, runtime->power->rapl2, runtime->power->window);
	fprintf(out, "\tsamples per second %u\n", runtime->sampler->sps);
	fprintf(out, "\tgovernor bound to core %d\n", runtime->sys->num_cpu);
	if (MAN_CTRL == 1)
	{
		fprintf(out, "\tgovernor frequency control ON\n");
	}
	else
	{
		fprintf(out, "\tgovernor frequency control OFF\n");
	}
}

void dump_sys(struct powgov_runtime *runtime, FILE *out)
{
	fprintf(out, "System Configuration:\n");
	fprintf(out, "\tMax frequency %f\n\tBase Frequency %f\n\tSockets %d\n\tCores Per Socket %d\n",
			runtime->sys->max_pstate / 10.0, runtime->sys->max_non_turbo / 10.0, runtime->sys->sockets, 
			runtime->sys->coresPerSocket);
	fprintf(out, "\tHyperthreads %s\n\tTotal Processors %d\n\tTDP %lf\n", 
			(runtime->sys->threadsPerCore ? "enabled" : "disabled"), runtime->sys->num_cpu, 
			runtime->power->proc_tdp);
	dump_rapl(out);
	hwpstuff(out);
	uint64_t misc = 0;
	read_msr_by_coord(0, 0, 0, 0x1A0, &misc);
	fprintf(out, "\tmisc enable %lx\n", misc);
	
}

int main(int argc, char **argv)
{
	// using libmsr
	if (init_msr())
	{
		fprintf(stderr, "ERROR: unable to init libmsr\n");
		exit(-1);
	}


	// have this process listen for SIGUSR1 signal
	struct sigaction sighand;
	memset(&sighand, 0, sizeof(struct sigaction));
	sighand.sa_handler = signal_exit;
	sigaction(SIGUSR1, &sighand, NULL);
	sigset_t sset;
	sigemptyset(&sset);
	sigaddset(&sset, SIGUSR1);
	sigprocmask(SIG_UNBLOCK, &sset, NULL);


	struct powgov_runtime *runtime = calloc(1, sizeof(struct powgov_runtime));;
	runtime->sys = calloc(1, sizeof(struct powgov_sysconfig));
	runtime->files = calloc(1, sizeof(struct powgov_files));
	runtime->sampler = calloc(1, sizeof(struct powgov_sampler));
	runtime->sampler->samplectrs = (unsigned long *) calloc(THREADCOUNT, sizeof(unsigned long));
	runtime->classifier = calloc(1, sizeof(struct powgov_classifier));
	runtime->power = calloc(1, sizeof(struct powgov_power));
	// initialize power governor configurization
	runtime->sampler->sps = 500; // -r for "rate"
	runtime->classifier->dist_thresh = 0.25;
	runtime->classifier->pct_thresh = 0.01;
	MAN_CTRL = 1; // -s for "system control"
	THREADCOUNT = 1; // deprecated, no user control
	// TODO: sampling should just dump every x seconds
	DURATION = 3000; // deprecated, no user control
	VERBOSE = 0; // -v for "verbose"
	EXPERIMENTAL = 0; // -e for experimental
	REPORT = 0; // -R for report
	// TODO: these should be read in from a file
	// these are the values at 800MHz, linear regression model based on frequency is used
	// cpu phase
	runtime->classifier->prof_class[0].ipc = 0.576;
	runtime->classifier->prof_class[0].mpc = 0.005;
	runtime->classifier->prof_class[0].rpc = 0.017;
	runtime->classifier->prof_class[0].epc = 0.027;
	runtime->classifier->prof_class[0].bpc = 0.0006;
	// memory phase
	runtime->classifier->prof_class[1].ipc = 0.122;
	runtime->classifier->prof_class[1].mpc = 0.004;
	runtime->classifier->prof_class[1].rpc = 0.118;
	runtime->classifier->prof_class[1].epc = 0.427;
	runtime->classifier->prof_class[1].bpc = 0.017;
	// IO/sleep phase (derived)
	runtime->classifier->prof_class[2].ipc = 0;
	runtime->classifier->prof_class[2].mpc = 0;
	runtime->classifier->prof_class[2].rpc = 0;
	runtime->classifier->prof_class[2].epc = 0;
	runtime->classifier->prof_class[2].bpc = 0;
	// mixed phase (derived)
	runtime->classifier->prof_class[3].ipc = 0.349;
	runtime->classifier->prof_class[3].mpc = 0.005;
	runtime->classifier->prof_class[3].rpc = 0.06;
	runtime->classifier->prof_class[3].epc = 0.25;
	runtime->classifier->prof_class[3].bpc = 0.005;
	// minimum values
	runtime->classifier->prof_minimums.ipc = DBL_MAX;
	runtime->classifier->prof_minimums.mpc = DBL_MAX;
	runtime->classifier->prof_minimums.rpc = DBL_MAX;
	runtime->classifier->prof_minimums.epc = DBL_MAX;
	runtime->classifier->prof_minimums.bpc = DBL_MAX;


	// lookup processor information with CPUID
	uint64_t rax, rbx, rcx, rdx;
	rax = rbx = rcx = rdx = 0;
	cpuid(0x16, &rax, &rbx, &rcx, &rdx);
	runtime->sys->min_pstate = 8;
	runtime->sys->max_pstate = ((rbx & 0xFFFFul) / 100);
	runtime->sys->max_non_turbo = ((rax & 0xFFFFul) / 100);
	uint64_t coresPerSocket, hyperThreads, sockets;
	int HTenabled;
	coresPerSocket = hyperThreads = sockets = HTenabled = 0;
	cpuid_detect_core_conf(&coresPerSocket, &hyperThreads, &sockets, &HTenabled);
	// TODO: currently hyper threads are ignored
	runtime->sys->num_cpu = sockets * coresPerSocket;
	assert(runtime->sys->num_cpu > 0);
	assert(runtime->sys->max_pstate > runtime->sys->max_non_turbo);
	assert(runtime->sys->max_non_turbo > runtime->sys->min_pstate);
	runtime->sys->sockets = sockets;
	runtime->sys->coresPerSocket = coresPerSocket;
	runtime->sys->threadsPerCore = hyperThreads;


	// add control for various features used
	// 1. power shifting (on default/off -S)
	// 2. throttle avoidance (on default/off -A)
	// 3. overpower (TODO) -O
	// 4. memory phase frequency override -M, only works if power shifting is off
	// process command line arguments
	char validargs[] = "rlwLtdsveRhSAMO";
	unsigned char numargs = strlen(validargs);
	int j;
	// TODO this sucks
	for (j = 1; j < argc; j++)
	{
		if (argv[j][0] != '-')
		{
			continue;
		}
		int badarg = 1;
		int i;
		for (i = 0; i < numargs; i++)
		{
			if (argv[j][1] == validargs[i])
			{
				badarg = 0;
				switch (argv[j][1])
				{
					case 'r':
						ARG_ERROR;
						runtime->sampler->sps = (unsigned) atoi(argv[j+1]);
						break;
					case 'l':
						ARG_ERROR;
						runtime->power->rapl1 = (double) atof(argv[j+1]);
						break;
					case 'w':
						ARG_ERROR;
						runtime->power->window = (double) atof(argv[j+1]);
						break;
					case 'L':
						ARG_ERROR;
						runtime->power->rapl2 = (double) atof(argv[j+1]);
						break;
					case 't':
						ARG_ERROR;
						runtime->classifier->dist_thresh = (double) atof(argv[j+1]);
						break;
					case 'd':
						ARG_ERROR;
						runtime->classifier->pct_thresh = (double) atof(argv[j+1]);
						break;
					case 's':
						MAN_CTRL = 0;	
						break;
					case 'v':
						VERBOSE = 1;
						break;
					case 'R':
						REPORT = 1;
						break;
					case 'e':
						EXPERIMENTAL = 1;
						break;
					case 'O':
						printf("Error: this feature %s not implemented yet\n", argv[j]);
						exit(-1);
						break;
					case 'S':
						MEM_POW_SHIFT = 0;
						break;
					case 'A':
						THROTTLE_AVOID = 0;
						break;
					case 'M':
						ARG_ERROR;
						if (MEM_POW_SHIFT != 0)
						{
							printf("Error: %s requires memory power shifting to be disabled (-S)\n",
									argv[j]);
							exit(-1);
						}
						MEM_FRQ_OVERRIDE = (double) atof(argv[j+1]);
						break;
					case 'h':
					default:
						dump_help();
						exit(-1);
						break;
				}
			}
		}
		if (badarg)
		{
			printf("Error: invalid option %s\n", argv[j]);
			exit(-1);
		}
	}


	// finish configuration based on arguments
	unsigned srate = (1000.0 / runtime->sampler->sps) * 1000u;
	if (MAN_CTRL)
	{
		enable_turbo(runtime);
		set_perf(runtime, runtime->sys->max_pstate);
	}
	activate_performance_counters(runtime);


	// bind the power governor to core X
	cpu_set_t cpus;
	CPU_ZERO(&cpus);
	// TODO: make this an option
	CPU_SET(0, &cpus);
	sched_setaffinity(0, sizeof(cpus), &cpus);


	// setup RAPL
	uint64_t unit;
	read_msr_by_coord(0, 0, 0, MSR_RAPL_POWER_UNIT, &unit);
	uint64_t power_unit = unit & 0xF;
	runtime->sys->rapl_power_unit = 1.0 / (0x1 << power_unit);
	uint64_t seconds_unit_raw = (unit >> 16) & 0x1F;
	runtime->sys->rapl_seconds_unit = 1.0 / (0x1 << seconds_unit_raw);
	runtime->sys->rapl_energy_unit = (unit >> 8) & 0x1F;
	uint64_t powinfo = 0;
	read_msr_by_coord(0, 0, 0, MSR_PKG_POWER_INFO, &powinfo);
	runtime->power->proc_tdp = (powinfo & 0x3FFF) * runtime->sys->rapl_power_unit;
	if (runtime->power->rapl1 == 0.0)
	{
		runtime->power->rapl1 = runtime->power->proc_tdp;
	}
	if (runtime->power->rapl2 == 0.0)
	{
		runtime->power->rapl2 = runtime->power->proc_tdp * 1.2;
	}
	set_rapl(runtime->power->window, runtime->power->rapl1, runtime->sys->rapl_power_unit, 
			runtime->sys->rapl_seconds_unit, 0);
	set_rapl2(100, runtime->power->rapl2, runtime->sys->rapl_power_unit, runtime->sys->rapl_seconds_unit, 0);


	// print verbose descriptions to stdout
	if (VERBOSE)
	{
		printf(HEADERSTRING "\n\tPower Governor v%s\n" HEADERSTRING "\n", VERSIONSTRING);
		dump_sys(runtime, stdout);
		dump_config(runtime, stdout);
	}


	// begin sampling if argument present
	if (EXPERIMENTAL)
	{
		init_sampling(runtime);
	}


	if (VERBOSE)
	{
		fprintf(stdout, "Initialization complete...\n");
	}


	// gather initial measurements for report if argument is present
	uint64_t inst_before, inst_after;
	uint64_t aperf_before, mperf_before;
	uint64_t ovf_stat;
	unsigned ovf_ctr = 0;
	double avgrate = 0.0;
	double durctr = 0.0;
	double lasttv = 0.0;
	struct timeval start, current;
	unsigned samplethread = 0;
	uint64_t busy_tsc_pre[runtime->sys->num_cpu * 2]; // this only works because c99
	if (REPORT)
	{
		read_msr_by_coord(0, 0, 0, IA32_FIXED_CTR0, &inst_before);
		read_msr_by_coord(0, 0, 0, MSR_IA32_APERF, &aperf_before);
		read_msr_by_coord(0, 0, 0, MSR_IA32_MPERF, &mperf_before);
		//uint64_t busy_unh_pre[runtime->sys->num_cpu * 2];
		int j;
		for (j = 0; j < runtime->sys->num_cpu; j++)
		{
			read_msr_by_coord(0, j, 0, IA32_TIME_STAMP_COUNTER, &busy_tsc_pre[j]);
			read_msr_by_coord(1, j, 0, IA32_TIME_STAMP_COUNTER, &busy_tsc_pre[j + runtime->sys->num_cpu]);
			//read_msr_by_coord(0, j, 0, IA32_FIXED_CTR1, &busy_unh_pre[j]);
			//read_msr_by_coord(1, j, 0, IA32_FIXED_CTR1, &busy_unh_pre[j + runtime->sys->num_cpu]);
			write_msr_by_coord(0, j , 0, IA32_FIXED_CTR1, 0x0ul);
			write_msr_by_coord(1, j , 0, IA32_FIXED_CTR1, 0x0ul);
		}
	
	}


	// the main loop, continues until a SIGUSR1 is recieved
	uint64_t ovf_ctrl;
	gettimeofday(&start, NULL);
	sample_data(runtime);
	usleep(srate);
	while (LOOP_CTRL)
	{
		sample_data(runtime);
		read_msr_by_coord(0, 0, 0, IA32_PERF_GLOBAL_STATUS, &ovf_stat);
		if (ovf_stat & 0x1)
		{
			read_msr_by_coord(0, 0, 0, IA32_PERF_GLOBAL_OVF_CTRL, &ovf_ctrl);
			write_msr_by_coord(0, 0, 0, IA32_PERF_GLOBAL_OVF_CTRL, ovf_ctrl & 0xFFFFFFFFFFFFFFFE);
			ovf_ctr++;
		}
		pow_aware_perf(runtime);
		usleep(srate);
	}
	gettimeofday(&current, NULL);
	double exec_time = (double) (current.tv_sec - start.tv_sec) +
			(current.tv_usec - start.tv_usec) / 1000000.0;


	if (VERBOSE)
	{
		fprintf(stdout, "Power Governor terminating...\n");
	}


	// dump all report files if argument present
	if (REPORT)
	{
		uint64_t aperf_after, mperf_after;
		read_msr_by_coord(0, 0, 0, MSR_IA32_APERF, &aperf_after);
		read_msr_by_coord(0, 0, 0, MSR_IA32_MPERF, &mperf_after);

		read_msr_by_coord(0, 0, 0, IA32_FIXED_CTR0, &inst_after);
		avgrate = exec_time / runtime->sampler->samplectrs[0];
		runtime->files->sreport = fopen("sreport", "w");
		dump_sys(runtime, runtime->files->sreport);
		dump_config(runtime, runtime->files->sreport);
		fprintf(runtime->files->sreport, "Results:\n");
		fprintf(runtime->files->sreport, "\tActual run time: %f\n", exec_time);
		fprintf(runtime->files->sreport, "\tAverage Sampling Rate: %lf seconds\n", avgrate);
		fprintf(runtime->files->sreport, "\tAvg Frq: %f\n", (float) 
				(aperf_after - aperf_before) / (float) 
				(mperf_after - mperf_before) * runtime->sys->max_non_turbo / 10.0);
		fprintf(runtime->files->sreport, "\tInstructions: %lu (ovf %u)\n", 
				inst_after - inst_before, ovf_ctr);
		fprintf(runtime->files->sreport, "\tIPS: %lf (ovf %u)\n", 
				(inst_after - inst_before) / exec_time, ovf_ctr);
		char fname[FNAMESIZE];
		snprintf((char *) fname, FNAMESIZE, "powgov_profiles");
		runtime->files->profout = fopen(fname, "w");
		// TODO: figure out if miscounts from remove_unused or something else (fixed?)
		remove_unused(runtime);
		agglomerate_profiles(runtime);
		dump_phaseinfo(runtime, runtime->files->profout, &avgrate);
		int j;
		for (j = 0; j < runtime->sys->num_cpu; j++)
		{
			uint64_t busy_tsc_post, busy_unh_post;
			read_msr_by_coord(0, j, 0, IA32_TIME_STAMP_COUNTER, &busy_tsc_post);
			read_msr_by_coord(0, j, 0, IA32_FIXED_CTR1, &busy_unh_post);
			double diff_unh = ((double) busy_unh_post);
			double diff_tsc = ((double) busy_tsc_post - busy_tsc_pre[j]);
			fprintf(runtime->files->sreport, "\ts1c%d pct busy: %lf\% (%lf/%lf)\n", j,  
					diff_unh / diff_tsc * 100.0, diff_unh, diff_tsc);
			//read_msr_by_coord(1, j, 0, IA32_TIME_STAMP_COUNTER, &busy_tsc_post);
			//read_msr_by_coord(1, j, 0, IA32_FIXED_CTR1, &busy_unh_post);
			//diff_unh = ((double) busy_unh_post);
			//diff_tsc = ((double) busy_tsc_post - busy_tsc_pre[j + runtime->sys->num_cpu]);
			//printf("s2c%d pct busy: %lf\% (%lf/%lf)\n", j,  diff_unh / diff_tsc * 100.0, diff_unh, diff_tsc);
		}
		fclose(runtime->files->sreport);
		fclose(runtime->files->profout);
	}


	// dump sampler data if argument present
	if (EXPERIMENTAL)
	{
		dump_data(runtime, runtime->files->sampler_dumpfiles, exec_time);
		int i;
		for (i = 0; i < THREADCOUNT; i++)
		{
			free(runtime->sampler->thread_samples[i]);
			fclose(runtime->files->sampler_dumpfiles[i]);
		}
		free(runtime->sampler->thread_samples);
		free(runtime->sampler->samplectrs);
		free(runtime->files->sampler_dumpfiles);
	}


	// Reset RAPL to defaults
	set_rapl(1, runtime->power->proc_tdp, runtime->sys->rapl_power_unit, runtime->sys->rapl_seconds_unit, 0);
	set_rapl2(100, runtime->power->proc_tdp * 1.2, runtime->sys->rapl_power_unit, 
			runtime->sys->rapl_seconds_unit, 0);
	finalize_msr();
	return 0;
}
