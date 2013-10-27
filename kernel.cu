#include <iostream>
#include <cuda.h>
#include "kernel.cuh"
#include "fitness_function.cu"
#include <curand.h>
#include <cfloat>

#ifdef __CUDACC__
#define HOST __global__
#define DEVICE __device__
#else
#define HOST
#define DEVICE
#endif

using namespace std;

//Forward declerations for this file
__global__ void doIteration(deviceMem data);
__device__ float getRnd(deviceMem data, int timesUsed);
__device__ void updateVelocity(deviceMem data, int particleIdx, int swarmIdx);
__device__ void updateBests(deviceMem data, int particleIdx, int swarmIdx, float fitness);
__device__ void moveParticle(deviceMem data, int particleIdx, int swarmIdx);

//global variables TODO: this won't work for multiple devices.
curandGenerator_t gen = NULL;

/**
 * A schim to call kernel function from ordinary C++ code in other files.
 */
void doIterationShim(deviceMem data, dim3 numBlock, dim3 blockSize){
	doIteration <<<numBlock, blockSize>>> (data);
}

/**
 * populates an are with a given number of random floats betweem 0 and 1;
 */
void generateRandomNumbers(float* where, int num){
	// Create pseudo-random number generator
	if(gen == NULL){
    	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	    // Set seed
	    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
	}
    // Generate n floats on device
    curandGenerateUniform(gen, where, num);
}

/**
 * Kernel to perform iteration og swarm
 */
__global__ void doIteration(deviceMem data){
	float fitness;
	
	//Getting some indexes
	int particleIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int swarmIdx 	= blockIdx.y * blockDim.y + threadIdx.y;

	//updateVelocity(data);
	updateVelocity(data, particleIdx, swarmIdx);

	//move particles
	moveParticle(data, particleIdx, swarmIdx);

	//get fitness
	fitness = getFitness(data, particleIdx, swarmIdx);

	//eventually update some bests
	updateBests(data, particleIdx, swarmIdx, fitness);
}

/**
 * function that updates velocity.
 *
 * this implementation is for a single swarm
 */
__device__ void updateVelocity(deviceMem data, int particleIdx, int swarmIdx){
	//whether the swarm should repel or not.
	int repel = (swarmIdx % 2 == 0) ? 1 : -1;
	
	//calculate new velocity for all dimensions
	for(int dimIdx=0 ; dimIdx < data.numDimensions ; dimIdx++){
		data.swarms[
			VELOCITY_IDX(swarmIdx, dimIdx, particleIdx, data.numDimensions, data.numParticles)
			]
			= (data.iterationMax/data.iterationCur)/data.iterationMax *
			 data.swarms[VELOCITY_IDX(swarmIdx, dimIdx, particleIdx, data.numDimensions, data.numParticles)]
			 
			 + 2 * getRnd(data, 0) * (//attraction to particle best
			 	 data.swarms[PB_POS_IDX(swarmIdx, dimIdx, particleIdx, data.numDimensions, data.numParticles)]
			 	-data.swarms[POSITION_IDX(swarmIdx, dimIdx, particleIdx, data.numDimensions, data.numParticles)] )
			 
			 + 2 * repel * getRnd(data, 1) * ( //attraction to global best
			 	 data.swarms[SB_POS_IDX(swarmIdx, dimIdx, data.numDimensions, data.numParticles)]
			 	-data.swarms[POSITION_IDX(swarmIdx, dimIdx, particleIdx, data.numDimensions, data.numParticles)]);
	}
}

/**
 * add the velocity to a particle
 */
__device__ void moveParticle(deviceMem data, int particleIdx, int swarmIdx){
	for(int dimIdx=0 ; dimIdx < data.numDimensions ; dimIdx++){
		//add vel to position
		data.swarms[POSITION_IDX(swarmIdx, dimIdx, particleIdx, data.numDimensions, data.numParticles)] 
			=
			  data.swarms[POSITION_IDX(swarmIdx, dimIdx, particleIdx, data.numDimensions, data.numParticles)]
			+ data.swarms[VELOCITY_IDX(swarmIdx, dimIdx, particleIdx, data.numDimensions, data.numParticles)];
	}
}

/**
 * update particle and global bests
 */
__device__ void updateBests(deviceMem data, int particleIdx, int swarmIdx, float fitness){
	//update global best
	if(fitness < data.swarms[SB_VAL_IDX(swarmIdx, data.numDimensions, data.numParticles)]){
		data.swarms[SB_VAL_IDX(swarmIdx, data.numDimensions, data.numParticles)] 
			= fitness;

		//Set the coordinates for this solution
		for(int dimIdx=0 ; dimIdx < data.numDimensions ; dimIdx++){
			data.swarms[SB_POS_IDX(swarmIdx, dimIdx, data.numDimensions, data.numParticles)]
				= data.swarms[POSITION_IDX(swarmIdx, dimIdx, particleIdx, data.numDimensions, data.numParticles)];
		}
	}
	//update particle best
	if(fitness < data.swarms[PB_VAL_IDX(swarmIdx, particleIdx, data.numDimensions, data.numParticles)]){
		data.swarms
			[PB_VAL_IDX(swarmIdx, particleIdx, data.numDimensions, data.numParticles)]
			= fitness;
		//Set the coordinates for this solution
		for(int dimIdx=0 ; dimIdx < data.numDimensions ; dimIdx++){
			data.swarms[PB_POS_IDX(swarmIdx, dimIdx, particleIdx, data.numDimensions, data.numParticles)]
				= data.swarms[POSITION_IDX(swarmIdx, dimIdx, particleIdx, data.numDimensions, data.numParticles)];
		}
	}
}

//**** AUX 
/**
 * Return a random number from the poll
 *
 * timesUsed is the number of times, it was used before, so first time this is called
 * by a thread, it is 0, second time 1 etc.
 */
__device__ float getRnd(deviceMem data, int timesUsed){
	//calculate the thread unique
	int unique = threadIdx.x + (blockDim.x * ((gridDim.x * blockIdx.y) + blockIdx.x));

	//we have to calculate them in batch, as it is insanely expensive
	//to calculate a single one.
	return data.randomNumbers[
		  data.rndIdx * data.numDimensions*data.numParticles*data.numSwarms//rndIdx
		+ unique*2 //2 random numbers pr. thread, this is impl. specific
		+ timesUsed];
}