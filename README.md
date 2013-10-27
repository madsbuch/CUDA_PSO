The GPU code
============

Dependencies
------------
Boost
Cuda


Fitness function
----------------

The folder fitness_functions contains all the functionality needed for the
fitness function alone, a discriptor structure describes the dimensionality and
bounds, then the underlaying system will do the swarm over this function
providing a vector representing a given solution

The dimensions are floats, without min and max, as is, later on following might be supported:
	FLOAT:
		min: FLT_MIN (from float.h)
		max: FLT_MAX (from float.h)
	
	future:
	INT (beware of lower performance)
	DOUBLE (lower performance)
		

implementation of new types is possible, they should just be totally orderes
sets.
