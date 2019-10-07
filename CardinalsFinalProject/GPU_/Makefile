NVCC=/usr/local/cuda-9.1/bin/nvcc


# GPU
gpu: colorPaletteGPU.cu
	${NVCC} -arch=sm_37 colorPaletteGPU.cu -o colorPaletteGPU `pkg-config --libs opencv` -L/usr/lib64
