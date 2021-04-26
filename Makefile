all:
	nvcc -lcuda -lcublas --compiler-options -fopenmp layer.cu main.cu -o ResNet  -arch=sm_70 -Wno-deprecated-gpu-targets

run:
	./ResNet
clean:
	rm ResNet
