/**
 * memory layout for a swarm
 *
 * to query these structure following are needed:
 *	- SwarmIndex (threadIdx.x * blockIdx.x)
 *	- ParticleIndex (threadIdx.y * blockIdx.y)
 *	- numberOfDimensions 
 */
/*
typedef struct Dim
{
	//an entry for each particle
	float* entry;
} Dim;

typedef struct SwarmMem {
	//information about this swarm
	Dim* positions;	//Particle positions (size: particles)
	Dim* velocity;	//Particle Velocities (size: particles)

	//for following, we need to save position in dimension room
	Dim* pBetsDim;	//Particle best positions (size: particles )
	float* pBestVal;	//Particle best value (size: particles)
	Dim sBest;		//This swarms best position (size 1)
	float sBestVal;		//Swarm best value (size 1)
} SwarmMem;*/

//macros for querying the swarm, macros allows then to be used on both host and device

//Return swarm size:
#define SWARM_SIZE(dimNum, pNum) (3*(dimNum*pNum) + pNum + dimNum + 1)

//swarmOffset + dimOffset + particleOffset
#define POSITION_IDX(swarmIdx, dimIdx, particleIdx, numDim, numParticles) \
	SWARM_SIZE(numDim, numParticles) * swarmIdx + dimIdx * numParticles + particleIdx

//swarmOffset + dimOffset + particleOffset + velocityOffset
#define VELOCITY_IDX(swarmIdx, dimIdx, particleIdx, numDim, numParticles) \
	SWARM_SIZE(numDim, numParticles) * swarmIdx \
	+ dimIdx * numParticles + particleIdx + numDim * numParticles

//swarmOffset + dimOffset + particleOffset + particleBestOffset
#define PB_POS_IDX(swarmIdx, dimIdx, particleIdx, numDim, numParticles) \
	SWARM_SIZE(numDim, numParticles) * swarmIdx \
	+ dimIdx * numParticles + particleIdx + 2 * numDim * numParticles

//swarmOffset + dimOffset + particleOffset + particleBestValOffset
#define PB_VAL_IDX(swarmIdx, particleIdx, numDim, numParticles) \
	SWARM_SIZE(numDim, numParticles) * swarmIdx \
	+ particleIdx + 3 * numDim * numParticles

//swarm best position
#define SB_POS_IDX(swarmIdx, dimIdx, numDim, numParticles) \
	SWARM_SIZE(numDim, numParticles) * swarmIdx \
	+ 3 * numDim * numParticles \
	+ numParticles \
	+ dimIdx

//returns idx of swarm best
#define SB_VAL_IDX(swarmIdx, numDim, numParticles) SWARM_SIZE(numDim, numParticles) * (1+swarmIdx) - 1


/**
 * Array og structs, one pr. device
 */
typedef struct deviceMem{

	/**
	 * Swarm data (now concentrate ;-) ):
	 * dim_0, dim_1, ... , dim_pNum       <- positions  
	 * dim_0, dim_1, ... , dim_pNum       <- velocities
	 * dim_0, dim_1, ... , dim_pNum       <- particle best positions
	 * float_0, float_1, ... , float_pNum <- particle best values
	 * dim                                <- Swarm best position
	 * float                              <- Swarm best value
	 *
	 * where dim = float_0, float_1, ..., float_dimNum
	 *
	 * size: (3*(dimNum*pNum) + pNum + dimNum + 1) * sizeof(float) * numSwarms
	 */
	float* swarms;

	//random numbers are generated in batch at this pointer
	float* randomNumbers;

	//******* READ ONLY DATA

	//positions for those external bests (size:  (dim * numExt)
	float* extPos;

	//******* VARIABLES
	int iterationMax;
	int iterationCur;

	int numDimensions; //number of dimensions
	int rndIdx; //index that should be passed to swarm iteration
	int numRnd; //number of
	int numSwarms;//number of swarms on this device
	int numParticles;//number of particles pr. swarm
} deviceMem;

void doIterationShim(deviceMem data,
	dim3 numBlock,
	dim3 blockSize);

void generateRandomNumbers(float* where, int num);