#pragma once

#include "gemm.h"

using GemmFn = void (*)(const DeviceMatrix &, const DeviceMatrix &,
                        DeviceMatrix &);

void RunSmallCorrectnessDemo(GemmFn gemm);
void RunLargeProfilingDemo(GemmFn gemm);
