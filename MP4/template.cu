#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here

//@@ Define constant memory for device kernel here
__constant__ float Mc[3][3][3];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  int z_o = blockIdx.z * (blockDim.z-2) + tz;
  int y_o = blockIdx.y * (blockDim.y-2) + ty;
  int x_o = blockIdx.x * (blockDim.x-2) + tx;

  int z_i = z_o-1;
  int y_i = y_o-1;
  int x_i = x_o-1;

  __shared__ float tile[8][8][8];

  float Pvalue = 0.0f;

  if ((z_i >= 0) && (z_i < z_size) && (y_i >= 0) && (y_i < y_size) && (x_i >= 0) && (x_i < x_size)) {
    tile[tz][ty][tx] = input[z_i*y_size*x_size + y_i*x_size + x_i];
  } else {
    tile[tz][ty][tx] = 0.0f;
  }

  __syncthreads();

  if ((tz < (blockDim.z-2)) && (ty < (blockDim.y-2)) && (tx < (blockDim.x-2))) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
          Pvalue += Mc[i][j][k] * tile[i+tz][j+ty][k+tx];
        }
      }
    }

    if ((z_o < z_size) && (y_o < y_size) && (x_o < x_size)) {
      output[z_o*y_size*x_size + y_o*x_size + x_o] = Pvalue;
    }
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  cudaMalloc((void **) &deviceInput, (inputLength-3) * sizeof(float));
  cudaMalloc((void **) &deviceOutput, (inputLength-3) * sizeof(float));

  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  cudaMemcpy(deviceInput, hostInput+3, (inputLength-3) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(Mc, hostKernel, kernelLength * sizeof(float), 0, cudaMemcpyHostToDevice);

  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 DimGrid(ceil((float)x_size/6), ceil((float)y_size/6), ceil((float)z_size/6));
  dim3 DimBlock(6+2, 6+2, 6+2);

  //@@ Launch the GPU kernel here
  conv3d<<<DimGrid,DimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  cudaMemcpy(hostOutput+3, deviceOutput, (inputLength-3) * sizeof(float), cudaMemcpyDeviceToHost);
  
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
