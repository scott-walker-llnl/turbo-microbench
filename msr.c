#include "msr.h"
#include <math.h>
#include <stdio.h>

void read_turbo_limit()
{
	uint64_t turbo_limit;
	read_msr_by_coord(0, 0, 0, MSR_TURBO_RATIO_LIMIT, &turbo_limit);

	printf("1 core: %x\n", (unsigned) (turbo_limit & 0xFF));
	printf("2 core: %x\n", (unsigned) ((turbo_limit >> 8) & 0xFF));
	printf("3 core: %x\n", (unsigned) ((turbo_limit >> 8) & 0xFF));
	printf("4 core: %x\n", (unsigned) ((turbo_limit >> 8) & 0xFF));
}

unsigned long get_turbo_limit()
{
	uint64_t turbo_limit;
	read_msr_by_coord(0, 0, 0, MSR_TURBO_RATIO_LIMIT, &turbo_limit);
	return turbo_limit;// & 0xFF;
}

unsigned long get_turbo_limit1()
{
	uint64_t turbo_limit;
	read_msr_by_coord(0, 0, 0, MSR_TURBO_RATIO_LIMIT_CORES, &turbo_limit);
	return turbo_limit;// & 0xFF;
}

void set_turbo_limit(unsigned long limit)
{
	uint64_t ratio_limit;
	limit &= 0xFF;
	//ratio_limit = 0x0ul | (limit) | ((limit - 1) << 8) | ((limit - 1) << 16) 
	//	| ((limit - 2) << 24) | ((limit - 3) << 32) | ((limit - 4) << 40) |
	//	((limit - 5) << 48) | ((limit - 6) << 54);
	ratio_limit = 0x0ul | (limit) | ((limit) << 8) | ((limit) << 16) 
		| ((limit) << 24) | ((limit) << 32) | ((limit) << 40) |
		((limit) << 48) | ((limit) << 54);
	write_msr_by_coord(0, 0, 0, MSR_TURBO_RATIO_LIMIT, ratio_limit);
	uint64_t core_limit;
	//core_limit = 0x0ul | (2ul) | (4ul << 8) | (8ul << 16) | (12ul << 24);
	//write_msr_by_coord(0, 0, 0, MSR_TURBO_RATIO_LIMIT_CORES, core_limit);
}

void set_all_turbo_limit(uint64_t limit)
{
	write_msr_by_coord(0, 0, 0, MSR_TURBO_RATIO_LIMIT, limit);
}

void set_all_turbo_limit1(uint64_t limit)
{
	write_msr_by_coord(0, 0, 0, MSR_TURBO_RATIO_LIMIT_CORES, limit);
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

void get_rapl_units(double *power_unit, double *seconds_unit)
{
	uint64_t unit;
	read_msr_by_coord(0, 0, 0, MSR_RAPL_POWER_UNIT, &unit);
	uint64_t pu = unit & 0xF;
	*power_unit = 1.0 / (0x1 << pu);
	fprintf(stderr, "power unit: %lx\n", pu);
	uint64_t su = (unit >> 16) & 0x1F;
	*seconds_unit = 1.0 / (0x1 << su);
	fprintf(stderr, "seconds unit: %lx\n", su);
}

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

void dump_platform_rapl()
{
	uint64_t prapl;
	read_msr_by_coord(0, 0, 0, 0x64D, &prapl);
	fprintf(stderr, "platform rapl is %lx\n", prapl);
	read_msr_by_coord(0, 0, 0, 0x65C, &prapl);
	fprintf(stderr, "platform rapl limit is %lx\n", prapl);
}

void dump_hwp_enable()
{
	uint64_t hwpe;
	read_msr_by_coord(0, 0, 0, 0x770, &hwpe);
	fprintf(stderr, "hwp enable is %lx\n", hwpe);
}

void dump_perf_limit()
{
	uint64_t plimit;
	read_msr_by_coord(0, 0, 0, 0x64F, &plimit);
	fprintf(stderr, "perf limit is %lx\n", plimit);
	if (plimit & (0x1ul))
	{
		fprintf(stderr, "\tPROCHOT\n");
	}
	if (plimit & (0x1ul << 1))
	{
		fprintf(stderr, "\tTHERMAL\n");
	}
	if (plimit & (0x1ul << 4))
	{
		fprintf(stderr, "\tRESIDENCY\n");
	}
	if (plimit & (0x1ul << 5))
	{
		fprintf(stderr, "\tRATL\n");
	}
	if (plimit & (0x1ul << 6))
	{
		fprintf(stderr, "\tVR\n");
	}
	if (plimit & (0x1ul << 7))
	{
		fprintf(stderr, "\tVR TDC\n");
	}
	if (plimit & (0x1ul << 8))
	{
		fprintf(stderr, "\tOTHER\n");
	}
	if (plimit & (0x1ul << 10))
	{
		fprintf(stderr, "\tRAPL PL1\n");
	}
	if (plimit & (0x1ul << 11))
	{
		fprintf(stderr, "\tRAPL PL2\n");
	}
	if (plimit & (0x1ul << 12))
	{
		fprintf(stderr, "\tTURBO\n");
	}
	if (plimit & (0x1ul << 13))
	{
		fprintf(stderr, "\tTURBO ATTN\n");
	}
	if (plimit & (0x1ul << 16))
	{
		fprintf(stderr, "\tPROCHOT LOG\n");
	}
	if (plimit & (0x1ul << 17))
	{
		fprintf(stderr, "\tTHERMAL LOG\n");
	}
	if (plimit & (0x1ul << 20))
	{
		fprintf(stderr, "\tRESIDENCY LOG\n");
	}
	if (plimit & (0x1ul << 21))
	{
		fprintf(stderr, "\tRATL LOG\n");
	}
	if (plimit & (0x1ul << 22))
	{
		fprintf(stderr, "\tVR LOG\n");
	}
	if (plimit & (0x1ul << 23))
	{
		fprintf(stderr, "\tVR THERMAL LOG\n");
	}
	if (plimit & (0x1ul << 24))
	{
		fprintf(stderr, "\tOTHER LOG\n");
	}
	if (plimit & (0x1ul << 26))
	{
		fprintf(stderr, "\tRAPL PL1 LOG\n");
	}
	if (plimit & (0x1ul << 27))
	{
		fprintf(stderr, "\tRAPL PL2 LOG\n");
	}
	if (plimit & (0x1ul << 28))
	{
		fprintf(stderr, "\tTURBO LOG\n");
	}
	if (plimit & (0x1ul << 29))
	{
		fprintf(stderr, "\tTURBO ATTN LOG\n");
	}
}
