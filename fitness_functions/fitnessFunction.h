#ifdef __CUDACC__
#define HOST __host__
#define DEVICE __device__
#else
#define HOST
#define DEVICE
#endif

/**
 * header for a fitness function
 */



 //calculates fitness for a given vector.
inline DEVICE float getFitness(float* pos);