#include <stdio.h>
#include <stdexcept>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"


// return GB/sec
float GBPerSec(int bytes, float sec) {
  return static_cast<float>(bytes) / (1024. * 1024. * 1024.) / sec;
}


// This is the CUDA "kernel" function that is run on the GPU.  You
// know this because it is marked as a __global__ function.
__global__ void
saxpy_kernel(int N, float alpha, float* x, float* y, float* result) {

    // compute overall thread index from position of thread in current
    // block, and given the block we are in (in this example only a 1D
    // calculation is needed so the code only looks at the .x terms of
    // blockDim and threadIdx.
    int index = blockIdx.x * blockDim.x + threadIdx.x;


    // this check is necessary to make the code work for values of N
    // that are not a multiple of the thread block size (blockDim.x)
    if (index < N)
       result[index] = alpha * x[index] + y[index];
}


// saxpyCuda --
//
// This function is regular C code running on the CPU.  It allocates
// memory on the GPU using CUDA API functions, uses CUDA API functions
// to transfer data from the CPU's memory address space to GPU memory
// address space, and launches the CUDA kernel function on the GPU.
namespace {
void checked(cudaError_t error, const char* operation) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}
struct DeviceBuffer {
    float* data = nullptr;
    explicit DeviceBuffer(size_t bytes) { checked(cudaMalloc(&data, bytes), "cudaMalloc"); }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void release() {
        checked(cudaFree(data), "cudaFree");
        data = nullptr;
    }
};
}

void saxpyCuda(int N, float alpha, float* xarray, float* yarray, float* resultarray) {
    if (N < 0) throw std::invalid_argument("negative array size");
    if (N == 0) return;
    if (!xarray || !yarray || !resultarray)
        throw std::invalid_argument("null nonempty array");
    const size_t bytes = sizeof(float) * static_cast<size_t>(N);
    const int threadsPerBlock = 512;
    const unsigned blocks = (static_cast<size_t>(N) + threadsPerBlock - 1) / threadsPerBlock;
    DeviceBuffer x(bytes), y(bytes), result(bytes);
    // Match the assignment's overall timer: transfers + kernel, excluding allocation.
    double startTime = CycleTimer::currentSeconds();
    checked(cudaMemcpy(x.data, xarray, bytes, cudaMemcpyHostToDevice), "copy x to device");
    checked(cudaMemcpy(y.data, yarray, bytes, cudaMemcpyHostToDevice), "copy y to device");
    double kernelStart = CycleTimer::currentSeconds();
    saxpy_kernel<<<blocks, threadsPerBlock>>>(N, alpha, x.data, y.data, result.data);
    checked(cudaGetLastError(), "launch saxpy");
    checked(cudaDeviceSynchronize(), "complete saxpy");
    double kernelEnd = CycleTimer::currentSeconds();
    checked(cudaMemcpy(resultarray, result.data, bytes, cudaMemcpyDeviceToHost), "copy result to host");
    double endTime = CycleTimer::currentSeconds();
    const double gibibytes = 3.0 * bytes / (1024.0 * 1024.0 * 1024.0);
    printf("SAXPY kernel: %.3f ms [%.3f GiB/s]\n",
           1000.0 * (kernelEnd-kernelStart), gibibytes/(kernelEnd-kernelStart));
    printf("SAXPY transfers + kernel: %.3f ms [%.3f GiB/s]\n",
           1000.0 * (endTime-startTime), gibibytes/(endTime-startTime));
    x.release(); y.release(); result.release();
}

void printCudaInfo() {

    // print out stats about the GPU in the machine.  Useful if
    // students want to know what GPU they are running on.

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}
