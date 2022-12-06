#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include "cuda_fp16.h"


// Tiled shared memory convolution (2 points)

// Tuning with restrict and loop unrolling (considered as one optimization only if you do both) (3 points)
// Sweeping various parameters to find best values (block sizes, amount of thread coarsening) (1 point)
// Multiple kernel implementations for different layer sizes (1 point)

// Fixed point (FP16) arithmetic. (note this can modify model accuracy slightly) (4 points)

// An advanced matrix multiplication algorithm (register-tiled, for example) (5 points)

__constant__ float Mc[16*4*7*7];

// ---Weight matrix in constant memory + fusion kernel---
// __global__ void conv_forward_kernel(float *output, const float *input, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     /*
//     Modify this function to implement the forward pass described in Chapter 16.
//     We have added an additional dimension to the tensors to support an entire mini-batch
//     The goal here is to be correct AND fast.

//     Function paramter definitions:
//     output - output
//     input - input
//     mask - convolution kernel
//     Batch - batch_size (number of images in x)
//     Map_out - number of output feature maps
//     Channel - number of input feature maps
//     Height - input height dimension
//     Width - input width dimension
//     K - kernel height and width (K x K)
//     */

//     const int Height_out = Height - K + 1;
//     const int Width_out = Width - K + 1;
//     // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
//     // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working

//     // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
//     // An example use of these macros:
//     // float a = in_4d(0,0,0,0)
//     // out_4d(0,0,0,0) = a

//     #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
//     #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
//     #define mask_4d(i3, i2, i1, i0) Mc[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

//     // Insert your GPU convolution kernel code here
//     const int BLOCKSIZE = 4;
//     const int BLOCKSIZE_Z = 64;

//     __shared__ float sibTileN[BLOCKSIZE_Z][BLOCKSIZE][BLOCKSIZE];

//     int b = blockIdx.z * blockDim.z + threadIdx.z;
//     int Row = blockIdx.y * blockDim.y + threadIdx.y;
//     int Col = blockIdx.x * blockDim.x + threadIdx.x;
//     float Pvalue = 0;

//     for (int q = 0; q < ceil((float)Channel*K*K/BLOCKSIZE); ++q) {
//         if (q*BLOCKSIZE+threadIdx.y < Channel*K*K) {
//             int h_unroll = q*BLOCKSIZE+threadIdx.y;
//             int w_unroll = Col;
//             sibTileN[threadIdx.z][threadIdx.y][threadIdx.x] = in_4d(b, h_unroll/(K*K), w_unroll/Width_out+(h_unroll%(K*K))/K, w_unroll%Width_out+(h_unroll%(K*K))%K);
//         } else {
//             sibTileN[threadIdx.z][threadIdx.y][threadIdx.x] = 0;
//         }

//         for (int k = 0; k < BLOCKSIZE; ++k) {
//             int h_unroll = Row;
//             int w_unroll = q*BLOCKSIZE+k;
//             Pvalue += mask_4d(h_unroll, w_unroll/(K*K), (w_unroll%(K*K))/K, (w_unroll%(K*K))%K) * sibTileN[threadIdx.z][k][threadIdx.x];
//         }
//         __syncthreads();
//     }

//     if (b < Batch && (Row < Map_out) && (Col < Height_out*Width_out)) {
//         out_4d(b, Row, Col/Width_out, Col%Width_out) = Pvalue;
//     }

//     #undef out_4d
//     #undef in_4d
//     #undef mask_4d
// }

// ---Weight matrix in constant memory + FP16---
// __global__ void conv_forward_kernel(float *output, const float *input, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     /*
//     Modify this function to implement the forward pass described in Chapter 16.
//     We have added an additional dimension to the tensors to support an entire mini-batch
//     The goal here is to be correct AND fast.
//     Function paramter definitions:
//     output - output
//     input - input
//     mask - convolution kernel
//     Batch - batch_size (number of images in x)
//     Map_out - number of output feature maps
//     Channel - number of input feature maps
//     Height - input height dimension
//     Width - input width dimension
//     K - kernel height and width (K x K)
//     */

//     const int Height_out = Height - K + 1;
//     const int Width_out = Width - K + 1;
//     // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
//     // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working

//     // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
//     // An example use of these macros:
//     // float a = in_4d(0,0,0,0)
//     // out_4d(0,0,0,0) = a

//     #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
//     #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
//     #define mask_4d(i3, i2, i1, i0) Mc[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

//     // Insert your GPU convolution kernel code here
//     int m = blockIdx.x;
//     int b = blockIdx.z * blockDim.z + threadIdx.z;

//     int W_grid = ceil((float)Width/8);
//     int h = (blockIdx.y / W_grid) * blockDim.x + threadIdx.y;
//     int w = (blockIdx.y % W_grid) * blockDim.x + threadIdx.x;
    
//     if (m < Map_out && h < Height_out && w < Width_out && b < Batch) {
//         float acc = 0.0f;
//         for (int c = 0; c < Channel; c++) {
//             for (int p = 0; p < K; p++) {
//                 for (int q = 0; q < K; q += 2) {
//                     if (q+1 < K) {
//                         __half2 a_ = __floats2half2_rn(in_4d(b, c, h+p, w+q), in_4d(b, c, h+p, w+q+1));
//                         __half2 b_ = __floats2half2_rn(mask_4d(m, c, p, q), mask_4d(m, c, p, q+1));

//                         __half2 c_ = __hmul2(a_, b_);

//                         acc += __high2float(c_) +  __low2float(c_);
//                     } else {
//                         acc += in_4d(b, c, h+p, w+q) * mask_4d(m, c, p, q);
//                     }
//                 }
//             }
//         }
//         out_4d(b, m, h, w) = acc;
//     }

//     #undef out_4d
//     #undef in_4d
//     #undef mask_4d
// }

// final kernal
__global__ void conv_forward_kernel(float* __restrict__ output, const float* __restrict__ input, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) Mc[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    const int BLOCKSIZE = 8;

    int m = blockIdx.x;
    int b = blockIdx.z * blockDim.z + threadIdx.z;

    int W_grid = ceil((float)Width/BLOCKSIZE);
    int h = (blockIdx.y / W_grid) * blockDim.x + threadIdx.y;
    int w = (blockIdx.y % W_grid) * blockDim.x + threadIdx.x;
    
    if (m < Map_out && h < Height_out && w < Width_out && b < Batch) {
        float acc = 0.0f;
        #pragma unroll
        for (int c = 0; c < Channel; c++) {
            #pragma unroll
            for (int p = 0; p < K; p++) {
                #pragma unroll
                for (int q = 0; q < K; q++) {
                    acc += in_4d(b, c, h+p, w+q) * mask_4d(m, c, p, q);
                }
            }
        }
        out_4d(b, m, h, w) = acc;
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    cudaMalloc((void **) device_input_ptr, Batch*Channel*Height*Width*sizeof(float));
    cudaMalloc((void **) device_output_ptr, Batch*Map_out*Height_out*Width_out*sizeof(float));

    cudaMemcpy(*device_input_ptr, host_input, Batch*Channel*Height*Width*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(Mc, host_mask, Map_out*Channel*K*K*sizeof(float), 0, cudaMemcpyHostToDevice);
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel

    // ---Weight matrix in constant memory + fusion kernel---
    // const int Height_out = Height - K + 1;
    // const int Width_out = Width - K + 1;

    // const int BLOCKSIZE = 4;
    // const int BLOCKSIZE_Z = 64;

    // dim3 DimGrid(ceil((float)Height_out*Width_out/BLOCKSIZE), ceil((float)Map_out/BLOCKSIZE), ceil((float)Batch/BLOCKSIZE_Z));
    // dim3 DimBlock(BLOCKSIZE, BLOCKSIZE, BLOCKSIZE_Z);

    const int BLOCKSIZE = 8;
    const int BLOCKSIZE_Z = 16;

    int W_grid = ceil((float)Width/BLOCKSIZE);
    int H_grid = ceil((float)Height/BLOCKSIZE);
    int Y = W_grid*H_grid;

    dim3 DimGrid(Map_out, Y, ceil((float)Batch/BLOCKSIZE_Z));
    dim3 DimBlock(BLOCKSIZE, BLOCKSIZE, BLOCKSIZE_Z);

    conv_forward_kernel<<<DimGrid,DimBlock>>>(device_output, device_input,
        Batch, Map_out, Channel, Height, Width, K);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    cudaMemcpy(host_output, device_output, Batch*Map_out*Height_out*Width_out*sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_input);
    cudaFree(device_output);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
