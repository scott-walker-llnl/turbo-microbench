#include "master.h"
#include "msr_core.h"

#define MSR_TURBO_RATIO_LIMIT 0x1AD
#define MSR_TURBO_RATIO_LIMIT_CORES 0x1AE

void read_turbo_limit();
void set_turbo_limit(unsigned long limit);
void set_all_turbo_limit(uint64_t limit);
unsigned long get_turbo_limit();
void enable_turbo(const unsigned tid);
void disable_turbo(const unsigned tid);

void get_rapl_units(double *power_unit, double *seconds_unit);
void set_rapl(unsigned sec, double watts, double pu, double su, unsigned affinity);
void set_rapl2(unsigned sec, double watts, double pu, double su, unsigned affinity);
void set_perf(const unsigned freq, const unsigned numcores);
void dump_rapl();
//inline void disable_rapl();

void dump_perf_limit();
void dump_platform_rapl();
unsigned long get_turbo_limit1();
void set_all_turbo_limit1(uint64_t limit);
