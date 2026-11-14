#include <stdio.h>
#include <stdexcept>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>

#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"

#define THREADS_PER_BLOCK 256


// helper function to round an integer up to the next power of 2
static inline int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

// exclusive_scan --
//
// Implementation of an exclusive scan on global memory array `input`,
// with results placed in global memory `result`.
//
// N is the logical size of the input and output arrays, however
// students can assume that both the start and result arrays we
// allocated with next power-of-two sizes as described by the comments
// in cudaScan().  This is helpful, since your parallel scan
// will likely write to memory locations beyond N, but of course not
// greater than N rounded up to the next power of 2.
//
// Also, as per the comments in cudaScan(), you can implement an
// "in-place" scan, since the timing harness makes a copy of input and
// places it in result
namespace {
void checkCuda(cudaError_t error, const char* action) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(action) + ": " + cudaGetErrorString(error));
}
int paddedSize(int n) {
    if (n < 0 || n > (1 << 30)) throw std::invalid_argument("unsupported scan size");
    return n == 0 ? 0 : nextPow2(n);
}
__global__ void initializeScan(const int* input, int n, int size, int* output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) output[i] = i < n ? input[i] : 0;
}
__global__ void upsweep(int* values, int stride, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        int right = (i + 1) * (2 * stride) - 1;
        values[right] += values[right - stride];
    }
}
__global__ void downsweep(int* values, int stride, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        int right = (i + 1) * (2 * stride) - 1;
        int left = values[right - stride];
        values[right - stride] = values[right];
        values[right] += left;
    }
}
__global__ void markRepeats(const int* input, int length, int* flags) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length) flags[i] = i + 1 < length && input[i] == input[i+1];
}
__global__ void scatterRepeats(const int* input, int length, const int* offsets, int* output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i + 1 < length && input[i] == input[i+1]) output[offsets[i]] = i;
}
struct Scratch {
    int* data = nullptr;
    explicit Scratch(int size) {
        checkCuda(cudaMalloc(&data, sizeof(int) * static_cast<size_t>(size)), "allocate scan scratch");
    }
    ~Scratch() { if (data) cudaFree(data); }
    Scratch(const Scratch&) = delete;
    Scratch& operator=(const Scratch&) = delete;
};
}

void exclusive_scan(int* input, int N, int* result) {
    const int size = paddedSize(N);
    if (size == 0) return;
    // Each index reads only itself, so initialization supports input == result.
    initializeScan<<<(size+255)/256,256>>>(input,N,size,result);
    checkCuda(cudaGetLastError(), "initialize scan");
    for (int stride=1; stride<size; stride*=2) {
        int count=size/(2*stride);
        upsweep<<<(count+255)/256,256>>>(result,stride,count);
        checkCuda(cudaGetLastError(), "scan upsweep");
    }
    checkCuda(cudaMemset(result+size-1,0,sizeof(int)), "clear scan root");
    for (int stride=size/2; stride>0; stride/=2) {
        int count=size/(2*stride);
        downsweep<<<(count+255)/256,256>>>(result,stride,count);
        checkCuda(cudaGetLastError(), "scan downsweep");
    }
}


//
// cudaScan --
//
// This function is a timing wrapper around the student's
// implementation of scan - it copies the input to the GPU
// and times the invocation of the exclusive_scan() function
// above. Students should not modify it.
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int* device_input;
    int N = end - inarray;  

    // This code rounds the arrays provided to exclusive_scan up
    // to a power of 2, but elements after the end of the original
    // input are left uninitialized and not checked for correctness.
    //
    // Student implementations of exclusive_scan may assume an array's
    // allocated length is a power of 2 for simplicity. This will
    // result in extra work on non-power-of-2 inputs, but it's worth
    // the simplicity of a power of two only solution.

    int rounded_length = nextPow2(end - inarray);
    
    cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);

    // For convenience, both the input and output vectors on the
    // device are initialized to the input values. This means that
    // students are free to implement an in-place scan on the result
    // vector if desired.  If you do this, you will need to keep this
    // in mind when calling exclusive_scan from find_repeats.
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_input, N, device_result);

    // Wait for completion
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
       
    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// cudaScanThrust --
//
// Wrapper around the Thrust library's exclusive scan function
// As above in cudaScan(), this function copies the input to the GPU
// and times only the execution of the scan itself.
//
// Students are not expected to produce implementations that achieve
// performance that is competition to the Thrust version, but it is fun to try.
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
   
    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);

    thrust::device_free(d_input);
    thrust::device_free(d_output);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// find_repeats --
//
// Given an array of integers `device_input`, returns an array of all
// indices `i` for which `device_input[i] == device_input[i+1]`.
//
// Returns the total number of pairs found
int find_repeats(int* device_input, int length, int* device_output) {
    const int size = paddedSize(length);
    if (length < 2) return 0;
    Scratch offsets(size);
    markRepeats<<<(length+255)/256,256>>>(device_input,length,offsets.data);
    checkCuda(cudaGetLastError(), "mark repeats");
    exclusive_scan(offsets.data,length,offsets.data);
    // Last flag is always zero, so the last exclusive prefix is the total.
    int count=0;
    checkCuda(cudaMemcpy(&count,offsets.data+length-1,sizeof(int),cudaMemcpyDeviceToHost), "read repeat count");
    scatterRepeats<<<(length+255)/256,256>>>(device_input,length,offsets.data,device_output);
    checkCuda(cudaGetLastError(), "scatter repeats");
    checkCuda(cudaDeviceSynchronize(), "complete repeat scatter");
    return count;
}


//
// cudaFindRepeats --
//
// Timing wrapper around find_repeats. You should not modify this function.
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {

    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    double startTime = CycleTimer::currentSeconds();
    
    int result = find_repeats(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    // set output count and results array
    *output_length = result;
    cudaMemcpy(output, device_output, length * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    float duration = endTime - startTime; 
    return duration;
}



void printCudaInfo()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
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
