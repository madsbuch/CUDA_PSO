linear:
	nvcc main.cpp kernel.cu -lboost_thread -lboost_system -lcurand -o main

emu:
	g++ main.cpp kernel.cu -lboost_thread -lboost_system -lcurand -lOcelotConfig  -o main

clean:
	-@rm main 2>/dev/null || true