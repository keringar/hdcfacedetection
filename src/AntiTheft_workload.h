#pragma once

#include <iostream>
#include <cstdlib>

extern "C" {
	void* anti_theft_workload_init (void* unused);

	void* anti_theft_workload_body (void* unused);

	void* anti_theft_workload_exit (void* unused);
}