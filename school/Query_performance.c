#include <Windows.h>

typedef long long LARGE_INTEGER;

LARGE_INTEGER freq, start, stop;
double diff;
QueryPerformanceFrequency(&freq); // computer frequency
QueryPerformanceCounter(&start); // starting point

//Algorithm
QueryPerformanceCounter(&stop); // stopping point
diff = (double)(stop.QuadPart - start.QuadPart)/ freq.QuadPart;