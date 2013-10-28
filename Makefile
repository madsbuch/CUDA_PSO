sa:
	nvcc main.cpp kernel.cu -lboost_thread -lboost_system -lcurand -O2 -o main 

emu:
	nvcc main.cpp kernel.cu -lboost_thread -lboost_system -lcurand -o main

clean:
	-@rm main 2>/dev/null || true