/**
 * code for optimizing a linear equation
 */

#include "kernel.h";

//the fitnessfunction

inline __device__ float getFitness(float* pos){
	return pos[1]^2 + pos[0] + 2;
}
