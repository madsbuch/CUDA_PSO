#include <iostream>
#include <cuda.h>

/*__device__ float getFitness(deviceMem data, int particleIdx, int swarmIdx){
	//pos_1*10 + pos_0
	return data.swarms[POSITION_IDX(swarmIdx, 1, particleIdx, data.numDimensions, data.numParticles)]*10
		 + data.swarms[POSITION_IDX(swarmIdx, 0, particleIdx, data.numDimensions, data.numParticles)];
}*/

__device__ float griewank(deviceMem data, int particleIdx, int swarmIdx){
	float res= 1 + (1/4000);
	for(int dimIdx=0 ; dimIdx < data.numDimensions ; dimIdx++){
		res +=  data.swarms[POSITION_IDX(swarmIdx, dimIdx, particleIdx, data.numDimensions, data.numParticles)]
			   *data.swarms[POSITION_IDX(swarmIdx, dimIdx, particleIdx, data.numDimensions, data.numParticles)];
	}
	float im=0;
	for(int dimIdx=0 ; dimIdx < data.numDimensions ; dimIdx++){
		im *=  acosf (data.swarms[
				POSITION_IDX(swarmIdx, dimIdx, particleIdx, data.numDimensions, data.numParticles)]) / sqrtf(dimIdx+1); 
	}
	return res - im;
}

__device__ float sphere(deviceMem data, int particleIdx, int swarmIdx){
	float res=0;
	for(int dimIdx=0 ; dimIdx < data.numDimensions ; dimIdx++){
		res +=  data.swarms[POSITION_IDX(swarmIdx, dimIdx, particleIdx, data.numDimensions, data.numParticles)]
			   *data.swarms[POSITION_IDX(swarmIdx, dimIdx, particleIdx, data.numDimensions, data.numParticles)];
	}
	return res;
}

/**
 * sphere function, should diverge in 0
 */
__device__ float getFitness(deviceMem data, int particleIdx, int swarmIdx){
	return sphere(data, particleIdx, swarmIdx);
}

/*__device__ float getFitness(deviceMem data, int particleIdx, int swarmIdx){
	//pos_1*10 + pos_0
	return 1000/(data.rndIdx == 0 ? 1 : (float) data.rndIdx);
}*/