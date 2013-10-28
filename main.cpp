/**
 * Primary file that initiates everything
 * do we use python on top of this for servercommunication?
 */
#include <boost/thread/thread.hpp>
#include <boost/lockfree/queue.hpp>
#include <string>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cfloat>
#include "kernel.cuh"
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>

#define DEBUG false
#define VERBOSE 0 //0 to 2, 0, very little is printed

using namespace std;


/**
 * the actual optimization will start when this is true
 *
 * global for all devices
 */
bool startOK     = false;

/**
 * number of dimensions this problem are provided with
 * 
 * global, not optimizable
 */
int numDimensions = 10;

/**
 * number of iterations
 *
 * global
 */
int iterations   = 100000;

/**
 * when to publish?
 */
int publishEvery = 100;

/**
 * suggested number of particles
 * 
 * this number is rounded up to a factor of warp size
 */
int numParticles = 64;

/**
 * number of random numbers to hold
 */
int numRandom = 9000000;

/**
 * number of swarms to run
 *
 * adjusted to be devisable by 4
 */
int numSwarms = 32;

/**
 * when to do spillover
 */
int spilloverEvery = 100;

/**
 * number of particles to spillover
 *
 *
 */
int spillOverNum=1;

/**
 * number of devices
 */
int devCount;

/**
 * devices allocated to swarming
 */
deviceMem* devs;

/**
 * true as long as the main command executer should run
 */
bool cont = false;

void printDeviceMem(int dev);
void publishSwarms(int d);
int roundUp(int numToRound, int multiple);
float rndflt();
void printUniversallyBest(int d);

/**
 * Functions that continously listens on stdin for details on other particle
 * swarms running elsewhere, or other instructions, and buffers it in queues
 */
void extListener(){
	cerr << "Listening for external swarm data" << endl;

	string buffer;
	while(true){
		getline(cin, buffer);
		cerr << "recieved: " << buffer << endl;
	}

	//listen for lines on stdin
	//parse the input string
	//inject the message to the correct queue
}
/**
 * makes spillover in the swarms
 */
void doSpillover(int d){
	return; //TODO make everything below work
	//copy data to host
	int swarmSize = SWARM_SIZE(numDimensions, devs[d].numParticles) * devs[d].numSwarms * sizeof(float);
	cerr << "size: " << swarmSize << endl;
	float* swarms = (float*) malloc(swarmSize);
	cudaDeviceSynchronize();//we really can't do anything if device is not finished
	cudaMemcpy(swarms, devs[d].swarms, swarmSize, cudaMemcpyDeviceToHost);

	//spillover ring topology 
	for(int sIdx = 0 ; sIdx < devs[d].numSwarms ; sIdx++){
		//TODO support spillover for more than 1 particle
		int   bestIdx = SB_VAL_IDX(sIdx, numDimensions, devs[d].numParticles), //best in this swarm
			  worstIdx; //worst in (sIdx + 1) % numSwarms
		float bestVal,
			  worstVal;

		int wsIdx = (sIdx+1)%devs[d].numSwarms;
		bestVal = swarms[bestIdx];
		worstVal = 0;

		//find the single worst
		for(int pIdx=0 ; pIdx < devs[d].numParticles ; pIdx++){
			if(swarms[PB_VAL_IDX(wsIdx, pIdx, devs[d].numDimensions, devs[d].numParticles)] > worstVal){
				worstVal = swarms[PB_VAL_IDX((sIdx+1)%devs[d].numSwarms, pIdx, devs[d].numDimensions, devs[d].numParticles)];
				worstIdx = pIdx;
			}
		}
		//swapp particle bests
		swarms[PB_VAL_IDX(wsIdx, worstIdx, devs[d].numDimensions, devs[d].numParticles)] = bestVal;
		swarms[bestIdx] = worstVal;

		//swap positions and velocities
		for(int d=0 ; d<devs[d].numDimensions ; d++){
			float h; //best
			/*
			//swap positions
			h = swarms[bestIdx];
			swarms[bestIdx] = swarms[worstIdx];
			swarms[worstIdx] = h;
			
			//swapp velocities
			h = swarms[VELOCITY_IDX(sIdx, d, bestIdx, devs[d].numDimensions, devs[d].numParticles)];
			swarms[VELOCITY_IDX(sIdx, d, bestIdx, devs[d].numDimensions, devs[d].numParticles)]
				= swarms[VELOCITY_IDX(wsIdx, d, worstIdx, devs[d].numDimensions, devs[d].numParticles)];
			swarms[VELOCITY_IDX(wsIdx, d, worstIdx, devs[d].numDimensions, devs[d].numParticles)] = h;
			
			//swapp particle best positions
			h = swarms[PB_POS_IDX(sIdx, d, bestIdx, devs[d].numDimensions, devs[d].numParticles)];
			swarms[PB_POS_IDX(sIdx, d, bestIdx, devs[d].numDimensions, devs[d].numParticles)]
				= swarms[PB_POS_IDX(wsIdx, d, worstIdx, devs[d].numDimensions, devs[d].numParticles)];
			swarms[PB_POS_IDX(wsIdx, d, worstIdx, devs[d].numDimensions, devs[d].numParticles)] = h;*/
		}
	}

	//Copy back to device
	cudaMemcpy(devs[d].swarms, swarms, swarmSize, cudaMemcpyHostToDevice);

	//remember to clean
	free(swarms);
}

/**
 * This function starts local swarms and listens contiously on buffered
 * details from other swamrs, to inject to the GPU. Otherwise it writes swarm-
 * state to stdout, to be published to other swarms
 *
 * a swarm pr. dev?
 */
void swarm(int dev){
	//set device
	cudaSetDevice(dev);
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, dev);

	//calculate threads pr. block
	int particlesPrBlock = devProp.warpSize;
	int swarmsPrBlock = devProp.maxThreadsPerBlock / particlesPrBlock;
	
	//calculate number of blocks
	int xBlockNum = devs[dev].numParticles / particlesPrBlock;
	int yBlockNum = devs[dev].numSwarms / swarmsPrBlock;

	cerr << "running swarm on dev: " << dev << " ppb: " << particlesPrBlock << " spb: " << swarmsPrBlock;
	cerr << " xBlockNum " << xBlockNum << " yBlockNum: " << yBlockNum << endl;
	while(iterations-- > 0){
		//iteration count incremented (done here to avoid device by 0)
		devs[dev].iterationCur++;

		//make sure the last iteration is done
		cudaDeviceSynchronize();
		
		//run an iteration
		doIterationShim(devs[dev],
			dim3(xBlockNum, yBlockNum), //blockNum
			dim3(particlesPrBlock , swarmsPrBlock));//threadsPrBlcok

		//every x iterations, extract some key values, and write the to external sources (stdout)
		if(VERBOSE > 1 && iterations%publishEvery == 0)
			publishSwarms(dev);
		else if(VERBOSE > 0 && iterations%publishEvery == 0)
			printUniversallyBest(dev);

		//random idx incremented
		devs[dev].rndIdx++;

		//2 numbers pr particle pr dimension
		int rndsPrIt = devs[dev].numDimensions*devs[dev].numParticles*devs[dev].numSwarms*2;
		if(devs[dev].rndIdx % (numRandom/rndsPrIt)  == 0){
			//regenerate rnd poll
			generateRandomNumbers(devs[dev].randomNumbers, numRandom);
			devs[dev].rndIdx = 0;
		}
	}
	printUniversallyBest(dev);
}



/**
 * initiates area of some memory
 */
void memInit(int d){
	int size = SWARM_SIZE(devs[d].numDimensions, devs[d].numParticles) * devs[d].numSwarms;
	float lMem[size];
	for(int swrmIdx = 0; swrmIdx < devs[d].numSwarms; swrmIdx++){
		for(int dimIdx = 0; dimIdx < devs[d].numDimensions; dimIdx++){
			for(int pIdx = 0; pIdx < devs[d].numParticles; pIdx++){
				//random position
				lMem[POSITION_IDX(swrmIdx, dimIdx, pIdx, devs[d].numDimensions, devs[d].numParticles)] 
					= rndflt();
				//initialise velocities random
				lMem[VELOCITY_IDX(swrmIdx, dimIdx, pIdx, devs[d].numDimensions, devs[d].numParticles)] 
					= rndflt();

				//particle best pos (debugging)
				lMem[PB_POS_IDX(swrmIdx, dimIdx, pIdx, devs[d].numDimensions, devs[d].numParticles)] 
					= 42 ;

				//Particle best
				lMem[PB_VAL_IDX(swrmIdx, pIdx, devs[d].numDimensions, devs[d].numParticles)] 
					= FLT_MAX;	
			}
		}
		//set swarm best to FLT_MAX
		lMem[SB_VAL_IDX(swrmIdx, devs[d].numDimensions, devs[d].numParticles)] = FLT_MAX;	
	}

	//copy to device
	cerr << "size " << size << endl;
	cudaMemcpy(devs[d].swarms, lMem, size*sizeof(float), cudaMemcpyHostToDevice);
}

/**
 * reads the config file and initialises device d
 */
void initialise(int d){
	cerr << "initialising: " << d << endl;
	//Read config and populate variables
	//[...]

	//select devie and fetch properties
	cudaSetDevice(d);
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, d);


    //**** VARIABLES ALLOCATION
	devs[d].numRnd = numRandom; //num random numbers
	devs[d].rndIdx = 0;
	devs[d].numDimensions = numDimensions; //number of dimensions
	devs[d].iterationMax = iterations;
	devs[d].iterationCur = 0;
	//We wan't number of particles to be devisable with the warpsize
	//for maximum utilisation
	devs[d].numParticles = roundUp(numParticles, devProp.warpSize);

	//we wan't swarms enough to utlize the whole GPU at least
	// is that more than devProp.multiProcessorCount?
	devs[d].numSwarms = roundUp(numSwarms, devProp.maxThreadsPerBlock / devProp.warpSize);

	//**** GLOBAL MEM ALLOCATION
	cudaMalloc((void**) &devs[d].randomNumbers,
		devs[d].numRnd*sizeof(float)); //allocate room for random numbers


	int swarmSize = (3*(numDimensions * devs[d].numParticles) 
		+ devs[d].numParticles 
		+ numDimensions 
		+ 1) * sizeof(float) * devs[d].numSwarms;
	//allocate the area swarms are kept in
	cudaMalloc((void**) &devs[d].swarms, swarmSize);

	//GPU initialisation
	generateRandomNumbers(devs[d].randomNumbers, devs[d].numRnd);
	//initiateSwarmShim(devs[d],
	//	dim3(xBlockNum, yBlockNum), //blockNum
	//	dim3(particlesPrBlock , devs[d].numSwarms));//threadsPrBlcok
	memInit(d);

	
	cerr << "Initialised device with: " << devs[d].numParticles;
	cerr << " particles, " << devs[d].numSwarms << " swarms, dim: " << numDimensions << " size: " << swarmSize << endl;
	
	if(DEBUG){
		cerr << "Pos: Swarm 0 dim 0 particle 0 idx = " << POSITION_IDX(0, 0, 0, numDimensions, devs[d].numParticles) << endl;
		cerr << "Vel: Swarm 0 dim 1 particle 0 idx = " << VELOCITY_IDX(0, 1, 0, numDimensions, devs[d].numParticles) << endl;
		cerr << "Bst: Swarm 0 dim 1 particle 0 idx = " << PB_POS_IDX  (0, 1, 0, numDimensions, devs[d].numParticles) << endl;
		cerr << "Gbst swarm 0 : " << SB_VAL_IDX(0, numDimensions, devs[d].numParticles) << endl;
	}
	if(VERBOSE > 1)
		printDeviceMem(d);
	if(VERBOSE > 0)
		publishSwarms(d);
}

/**
 * Cleans up and write final information, and a hasFinished to stdout, to be 
 * published to the controller.
 */
void cleanup(){
	cerr << "cleaning up" << endl;
	cout << "{terminated : true}" << endl;
}

/**
 * prints best result of everything
 */
void printUniversallyBest(int d){
	float best = FLT_MAX;
	float pos[numDimensions];

	int swarmSize = SWARM_SIZE(numDimensions, devs[d].numParticles) * devs[d].numSwarms * sizeof(float);
	float* swarms = (float*) malloc(swarmSize);
	cudaDeviceSynchronize();//we really can't print if device is not finished
	cudaMemcpy(swarms, devs[d].swarms, swarmSize, cudaMemcpyDeviceToHost);

	for (int i = 0; i < devs[d].numSwarms ; i++){
		if(swarms[SB_VAL_IDX(i, numDimensions, devs[d].numParticles)] < best){
			best = swarms[SB_VAL_IDX(i, numDimensions, devs[d].numParticles)];

			//save coordinates
			for(int dimIdx=0 ; dimIdx < devs[d].numDimensions ; dimIdx++){
				pos[dimIdx] = swarms[SB_POS_IDX(i, dimIdx, devs[d].numDimensions, devs[d].numParticles)];
			}
		}
	}
	cout << "Universally best: " << best;
	cout << " at (";
	for(int dimIdx=0 ; dimIdx < devs[d].numDimensions ; dimIdx++){
		cout << pos[dimIdx] << ", ";
	}
	cout << ")" << endl;

	free(swarms);
}

/**
 * print all swamrs bests
 */
void publishSwarms(int d){
	int swarmSize = SWARM_SIZE(numDimensions, devs[d].numParticles) * devs[d].numSwarms * sizeof(float);
	float* swarms = (float*) malloc(swarmSize);
	cudaDeviceSynchronize();//we really can't print if device is not finished
	cudaMemcpy(swarms, devs[d].swarms, swarmSize, cudaMemcpyDeviceToHost);

	for (int i = 0; i < devs[d].numSwarms ; i++){
		if(DEBUG){
			cerr << "p1 pos: (";
			for(int dimIdx=0 ; dimIdx < devs[d].numDimensions ; dimIdx++){
				cout << swarms[POSITION_IDX(i, dimIdx, 0, devs[d].numDimensions, devs[d].numParticles)] << ", ";
			}
			cerr << ") ";
			
			cerr << "p1 vel: (";
			for(int dimIdx=0 ; dimIdx < devs[d].numDimensions ; dimIdx++){
				cout << swarms[VELOCITY_IDX(i, dimIdx, 0, devs[d].numDimensions, devs[d].numParticles)] << ", ";
			}
			cerr << ") ";
		}

		cout << "Swam " << i << "'s best: " << swarms[SB_VAL_IDX(i, numDimensions, devs[d].numParticles)];
		cout << " at (";
		for(int dimIdx=0 ; dimIdx < devs[d].numDimensions ; dimIdx++){
			cout << swarms[SB_POS_IDX(i, dimIdx, devs[d].numDimensions, devs[d].numParticles)] << ", ";
		}
		cout << ")" << endl;
	}

	free(swarms);
}

/**
 * for debugging: prints the device information
 */
void printDeviceMem(int d){
	int swarmSize = SWARM_SIZE(numDimensions, devs[d].numParticles) * devs[d].numSwarms * sizeof(float);
	cerr << "copysize: " << swarmSize << endl;
	float* swarms = (float*) malloc(swarmSize);
	cudaDeviceSynchronize();//we really can't print if device is not finished
	cudaMemcpy(swarms, devs[d].swarms, swarmSize, cudaMemcpyDeviceToHost);

	for (int i = 0; i < swarmSize / sizeof(float); i++)
	{
		cout << swarms[i] << endl;
	}
	cout << "Positions:" << endl;
	for (int dim = 0; dim < numDimensions; dim++){
		for (int p = 0; p < devs[d].numParticles; p++)
		{
			cout << swarms[POSITION_IDX(0, dim, p, numDimensions, devs[d].numParticles)] << " ";
		}
		cout << endl;
	}

	cout << "Velocities:" << endl;
	for (int dim = 0; dim < numDimensions; dim++){
		for (int p = 0; p < devs[d].numParticles; p++)
		{
			cout << swarms[VELOCITY_IDX(0, dim, p, numDimensions, devs[d].numParticles)] << " ";
		}
		cout << endl;
	}

	cout << "Best:" << endl;
	for (int dim = 0; dim < numDimensions; dim++){
		for (int p = 0; p < devs[d].numParticles; p++)
		{
			cout << swarms[PB_POS_IDX(0, dim, p, numDimensions, devs[d].numParticles)] << " ";
		}
		cout << endl;
	}
}

int main(){
	//get cuda device count
	cudaGetDeviceCount(&devCount);
	cerr << "DeviceCount: " << devCount << endl;
	devs = (deviceMem*) malloc(devCount * sizeof(deviceMem));

	//create thread pointers
	boost::thread listener;
	boost::thread initThrs[devCount];
	boost::thread swarmThrs[devCount];

	//start external listener
	listener = boost::thread(extListener);

	//initialise everything (in parallel!)
	for (int i = 0; i < devCount; i++)
		initThrs[i] = boost::thread(initialise, i);
	for (int i = 0; i < devCount; i++)
		initThrs[i].join();

	//start swarm pr. device
	for (int i = 0; i < devCount; i++){
		swarmThrs[i] = boost::thread(swarm, i);
	}

	//if we just wanna stop straight awaywithout joining
	bool doJoin = true;
	while(cont){
		
		//wait some time
		boost::this_thread::sleep(boost::posix_time::milliseconds(50));
	}

	if(doJoin){
		for (int i = 0; i < devCount; i++)
		{
			swarmThrs[i].join();
		}
	}

	//ok, we just end the process here
	
	cleanup();
}

//**** AUX

int roundUp(int numToRound, int multiple) { 
	if(multiple == 0) {
		return numToRound; 
	}

	int remainder = numToRound % multiple;
	if (remainder == 0)
		return numToRound;
	return numToRound + multiple - remainder;
}
boost::mt19937 rng;
float rndflt(){
	boost::uniform_real<float> u(FLT_MIN, FLT_MAX);
	boost::variate_generator<boost::mt19937&, boost::uniform_real<float> > gen(rng, u);
	return gen()/10.0e+30;
}