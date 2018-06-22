#define _BSD_SOURCE
#define _GNU_SOURCE
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <signal.h>
#include <sys/wait.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <ctype.h>
#include "msr.h"

//#define DEBUG

#define GREEN "\x1B[32m"
#define RESET "\x1B[0m"

#define MAX_BENCH_NAME 64
#define INIT_TIME 0
#define INIT_FREQ 0x2D

extern char **environ;

void set_perf(const unsigned freq, const unsigned numcores)
{
	static uint64_t perf_ctl = 0x0ul;
	uint64_t freq_mask = 0x0ul;
	if (perf_ctl == 0x0ul)
	{
		read_msr_by_coord(0, 0, 0, IA32_PERF_CTL, &perf_ctl);
	}
	perf_ctl &= 0xFFFFFFFFFFFF0000ul;
	freq_mask = freq;
	freq_mask <<= 8;
	perf_ctl |= freq_mask;
	int i;
	for (i = 0; i < numcores; i++)
	{
		write_msr_by_coord(0, i, 0, IA32_PERF_CTL, perf_ctl);
	}
}

void parse_config(char **name, char ***args, char ***env, FILE **config)
{
	*name = NULL;
	size_t namesize = 0;
	ssize_t namelen = getline(name, &namesize, *config);
	char *nameptr = *name;
	if (namelen <= 0)
	{
		printf(GREEN "[CONTROLLER] bad config file (%d read), %s" RESET "\n", namelen, strerror(errno));
		return;
	}
	nameptr[namelen - 1] = '\0';

#ifdef DEBUG
	printf(GREEN "[CONTROLLER] exec name '%s'" RESET "\n", nameptr);
#endif

	char *args_buffer = NULL;
	size_t argssize = 0;
	ssize_t arglen = getline(&args_buffer, &argssize, *config);

#ifdef DEBUG
	printf(GREEN "[CONTROLLER] exec args '%s'" RESET, args_buffer);
#endif

	if (arglen > 0)
	{
		int numargs = 0;
		int i;
		// get the number of arguments
		for (i = 0; i < arglen; i++)
		{
			if (isspace(args_buffer[i]))
			{
				numargs++;
				// eat whitepsace
				while(isspace(args_buffer[i]))
				{
					args_buffer[i] = '\0';
					i++;
				}
				i--;
			}
		}

#ifdef DEBUG
	printf(GREEN "[CONTROLLER] exec num args %d" RESET "\n", numargs);
#endif

		*args = (char **) malloc((numargs + 1) * sizeof(char *));
		char **argptr = *args;
		int start = 1;
		int j = 0;
		for (i = 0; i < arglen - 1; i++)
		{
			if (args_buffer[i] == '\0')
			{
				start = 1;
				// eat whitepsace
				while(args_buffer[i] == '\0')
				{
					i++;
				}
				i--;
			}
			else if (start)
			{
				argptr[j] = &args_buffer[i];
				start = 0;
#ifdef DEBUG
				printf(GREEN "[CONTROLLER] argument '%s'" RESET "\n", argptr[j]);
#endif
				j++;
			}
		}
		argptr[j] = NULL;
	}
	else
	{
		*args = NULL;
	}


	char *env_buffer = NULL;
	size_t envsize = 0;
	ssize_t envlen = getline(&env_buffer, &envsize, *config);
#ifdef DEBUG
	printf(GREEN "[CONTROLLER] exec env '%s'" RESET, env_buffer);
#endif
	if (envlen > 0)
	{
		int numenv = 0;
		int i;
		// get the number of environment variables
		for (i = 0; i < envlen; i++)
		{
			if (isspace(env_buffer[i]))
			{
				numenv++;
				// eat whitepsace
				while(isspace(env_buffer[i]))
				{
					env_buffer[i] = '\0';
					i++;
				}
				i--;
			}
		}

#ifdef DEBUG
	printf(GREEN "[CONTROLLER] exec num env %d" RESET "\n", numenv);
#endif
		*env = (char **) malloc((numenv + 1) * sizeof(char *));
		char **envptr = *env;
		int start = 1;
		int j = 0;
		for (i = 0; i < envlen - 1; i++)
		{
			if (env_buffer[i] == '\0')
			{
				start = 1;
				// eat whitepsace
				while(env_buffer[i] == '\0')
				{
					i++;
				}
				i--;
			}
			else if (start)
			{
				envptr[j] = &env_buffer[i];
				start = 0;
#ifdef DEBUG
				printf(GREEN "[CONTROLLER] envp '%s'" RESET "\n", envptr[j]);
#endif
				j++;
			}
		}
		envptr[j] = NULL;
	}
	else
	{
		*env = NULL;
	}
}

// run the two programs creating artificial phases
int multi_program(pid_t phase1child, pid_t phase2child, pid_t samchild, unsigned cycle, unsigned long timelim, float ratio)
{
	kill(phase2child, SIGSTOP);
	kill(phase1child, SIGCONT);
	sleep(1);
	kill(phase1child, SIGSTOP);
	
	unsigned phase1len = (unsigned) (cycle * ratio);
	unsigned phase2len = cycle - phase1len;

	// now the benchmarks starts, begin sampling
	kill(samchild, SIGCONT);

	int deviation = 0;
	int status; // status of the benchmark
	unsigned long totalus = 0;
	pid_t result1 = 0; // pid result of the child process 1
	pid_t result2 = 0; // pid result of the child process 2
	pid_t result3 = 0; // pid result of the sampler
	pid_t resmask = 0;
	// while phase 2 still running...
	// and time limit hasn't been hit
	while (resmask == 0 && timelim * 1000000 > totalus)
	{
		//set_perf(freq_ph_1, 4);
		// begin phase1
		kill(phase1child, SIGCONT);
		deviation += usleep(phase1len);
		totalus += phase1len;
		kill(phase1child, SIGSTOP);
		// end phase1
		
		//set_perf(freq_ph_2, 4);
		// begin phase2
		kill(phase2child, SIGCONT);
		deviation += usleep(phase2len);
		totalus += phase2len;
		kill(phase2child, SIGSTOP);
		// end phase 2
		
		result1 = waitpid(phase1child, &status, WNOHANG);
		result2 = waitpid(phase2child, &status, WNOHANG);
		result3 = waitpid(samchild, &status, WNOHANG);
		resmask = result1 | result2 | result3;
	}
	
	if (timelim * 1000000 <= totalus)
	{
		printf(GREEN "[CONTROLLER] time expired" RESET "\n");
	}
	if (result1 != 0)
	{
		printf(GREEN "[CONTROLLER] program 1 exit" RESET "\n");
	}
	if (result2 != 0)
	{
		printf(GREEN "[CONTROLLER] program 2 exit" RESET "\n");
	}
	if (result3 != 0)
	{
		printf(GREEN "[CONTROLLER] ERROR: sampler program exit unexpectedly" RESET "\n");
	}

	kill(samchild, SIGUSR1);
	// wait until the sampler finishes dumping the data
	pid_t result = 0;
	while (result == 0)
	{
		result = waitpid(samchild, &status, WNOHANG);
		usleep(10000);
	}

	// resume phase1child so it can dump the data
	kill(phase1child, SIGCONT);

	kill(phase1child, SIGTERM);
	// wait until program finishes dumping the data
	result = 0;
	while (result == 0)
	{
		result = waitpid(phase1child, &status, WNOHANG);
		usleep(10000);
	}

	printf(GREEN "[CONTROLLER] Failed Sleeps: %d" RESET "\n", deviation * -1);
}

// run the first program only
int single_program(pid_t phase1child, pid_t samchild, unsigned long timelim)
{
	// let the program initialize
	kill(phase1child, SIGCONT);
	sleep(1);
	// now the benchmarks starts, begin sampling
	kill(samchild, SIGCONT);

	unsigned long totalus = 0;
	int status; // status of the benchmark
	pid_t result = 0; // pid result of the child process
	pid_t result3 = 0; // pid result of sampler
	pid_t resmask = 0;
	// while NAS still running...
	while (resmask == 0 && timelim * 1000000 > totalus)
	{
		usleep(10000);
		totalus += 10000;
		result = waitpid(phase1child, &status, WNOHANG);
		result3 = waitpid(samchild, &status, WNOHANG);
		resmask = result | result3;
		//set_perf(0x8, 4);
	}

	if (timelim * 1000000 <= totalus)
	{
		printf(GREEN "[CONTROLLER] time expired" RESET "\n");
	}
	if (result != 0)
	{
		printf(GREEN "[CONTROLLER] program 1 exit" RESET "\n");
	}
	if (result3 != 0)
	{
		printf(GREEN "[CONTROLLER] ERROR: sampler program exit unexpectedly" RESET "\n");
	}

	kill(samchild, SIGUSR1);
	// wait until the sampler finishes dumping the data
	result = 0;
	while (result == 0)
	{
		result = waitpid(samchild, &status, WNOHANG);
		usleep(10000);
	}

	kill(phase1child, SIGTERM);
	// wait until program finishes dumping any data
	result = 0;
	while (result == 0)
	{
		result = waitpid(phase1child, &status, WNOHANG);
		usleep(10000);
	}

	return 0;
}

int main(int argc, char **argv)
{
	if (argc != 6)
	{
		fprintf(stderr, GREEN "[CONTROLLER] useage: ./fx <freq phase 1> <freq phase 2> <cycle len in us> <time limit (phase 1)> <phase ratio (0-1)>" RESET "\n");
		return -1;
	}

	unsigned freq_ph_1 = (unsigned) strtol(argv[1], NULL, 16);
	unsigned freq_ph_2 = (unsigned) strtol(argv[2], NULL, 16);
	unsigned cycle = (unsigned) atoi(argv[3]);
	unsigned long timelim = (unsigned) atoi(argv[4]);
	float ratio = atof(argv[5]);
	printf(GREEN "[CONTROLLER] Phase 1 Frequency: %x" RESET "\n", freq_ph_1);
	printf(GREEN "[CONTROLLER] Phase 2 Frequency: %x" RESET "\n", freq_ph_2);
	printf(GREEN "[CONTROLLER] Phase Ratio: %f" RESET "\n", ratio);
	printf(GREEN "[CONTROLLER] Cycle Length (us): %d" RESET "\n", cycle);
	//printf(GREEN "[CONTROLLER] Power Limit: %s" RESET "\n", argv[4]);
	printf(GREEN "[CONTROLLER] Time Limit (phase 1): %u" RESET "\n", timelim);
	
	int masterpid = getpid(); // the PID of the top level controller
	char *phase1name;
	char **phase1args;
	char **phase1env;
	FILE * phase1config = fopen("phase1.config", "r");
	if (phase1config == NULL)
	{
		printf(GREEN "[CONTROLLER] ERROR: 'phase1.config' not found" RESET "\n");
		return -1;
	}
	else
	{
		parse_config(&phase1name, &phase1args, &phase1env, &phase1config);
	}

	printf(GREEN "[CONTROLLER] Begin Benchmark '%s'" RESET "\n", phase1name);
	pid_t phase1child = fork();
	if (getpid() != masterpid)
	{
		// exec first program
		int err = execve(phase1name, phase1args, phase1env);
		printf(GREEN "[CONTROLLER] ERROR: could not execute '%s'" RESET "\n", phase1name);
		printf(strerror(errno));
		printf("\n");
		return 0;
	}

	free(phase1name);
	if (phase1args != NULL)
	{
		free(phase1args[0]);
	}
	free(phase1args);
	if (phase1env != NULL)
	{
		free(phase1env[0]);
	}
	free(phase1env);
	fclose(phase1config);
	
	// if there is no cycle length, we are not artificially creating phases
	char *phase2name;
	char **phase2args;
	char **phase2env;
	FILE * phase2config;	
	pid_t phase2child;
	if (cycle != 0)
	{
		phase2config = fopen("phase2.config", "r");
		if (phase2config == NULL)
		{
			printf(GREEN "[CONTROLLER] ERROR: 'phase2.config' not found" RESET "\n");
			return -1;
		}
		else
		{
			parse_config(&phase2name, &phase2args, &phase2env, &phase2config);
			printf(GREEN "[CONTROLLER]: Begin Benchmark '%s'" RESET "\n", phase2name);
		}
		phase2child = fork();

		if (getpid() != masterpid)
		{
			// exec second program
			execve(phase2name, phase2args, phase2env);
			printf(GREEN "[CONTROLLER] ERROR: could not execute '%s'" RESET "\n", phase2name);
			printf(strerror(errno));
			return 0;
		}

		free(phase2name);
		if (phase2args != NULL)
		{
			free(phase2args[0]);
		}
		free(phase2args);
		if (phase2env != NULL)
		{
			free(phase2env[0]);
		}
		free(phase2env);
		fclose(phase2config);
	}

	char *samname;
	char **samargs;
	char **samenv;
	FILE * samconfig = fopen("sampler.config", "r");
	if (samconfig == NULL)
	{
		printf(GREEN "[CONTROLLER] ERROR: 'sampler.config' not found" RESET "\n");
		return -1;
	}
	else
	{
		parse_config(&samname, &samargs, &samenv, &samconfig);
		printf(GREEN "[CONTROLLER] Begin sampling with '%s'" RESET "\n", samname);
	}
	pid_t samchild = fork();
	if (getpid() != masterpid)
	{
		// exec the data sampler
		execve(samname, samargs, samenv);
		printf(GREEN "[CONTROLLER] ERROR: could not execute '%s'" RESET "\n", samname);
		printf(strerror(errno));
		return 0;
	}

	free(samname);
	if (samargs != NULL)
	{
		free(samargs[0]);
	}
	free(samargs);
	if (samenv != NULL)
	{
		free(samenv[0]);
	}
	free(samenv);
	fclose(samconfig);

	// signal sampling to stop, wait until parent is ready
	kill(samchild, SIGSTOP);
	// let the NAS benchmark do its initialization
	kill(phase1child, SIGSTOP);

	if (init_msr())
	{
		printf(GREEN "[CONTROLLER] ERROR: unable to init libmsr" RESET "\n");
		kill(phase1child, SIGKILL);
		if (cycle != 0)
		{
			kill(phase2child, SIGKILL);
		}
		kill(samchild, SIGKILL);
		return -1;
	}

	// let the benchmarks get through their initialization phase
	// set the frequency high, in case it was left low
	//set_perf(INIT_FREQ, 4);
	//set_perf(0x8, 4);
	sleep(INIT_TIME);
	
	if (cycle > 0)
	{
		multi_program(phase1child, phase2child, samchild, cycle, timelim, ratio);
	}
	else
	{
		single_program(phase1child, samchild, timelim);
	}

	printf(GREEN "[CONTROLLER] All processes successfully terminated" RESET "\n");
	finalize_msr();

	return 0;
}
