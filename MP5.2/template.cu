// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 1024 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void store(float *input, float *aux, int len) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if ((i+1)*2*BLOCK_SIZE-1 < len) {
    aux[i] = input[(i+1)*2*BLOCK_SIZE-1];
  }
}

__global__ void add(float *input, float *aux, int len) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len && blockIdx.x > 1) {
    input[i] += aux[blockIdx.x/2-1];
  }
}

__global__ void scan(float *input, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float T[2*BLOCK_SIZE];

  if ((blockIdx.x * blockDim.x + threadIdx.x)*2+1 < len) {
    T[threadIdx.x*2] = input[(blockIdx.x * blockDim.x + threadIdx.x)*2];
    T[threadIdx.x*2+1] = input[(blockIdx.x * blockDim.x + threadIdx.x)*2+1];
  } else if ((blockIdx.x * blockDim.x + threadIdx.x)*2 < len) {
    T[threadIdx.x*2] = input[(blockIdx.x * blockDim.x + threadIdx.x)*2];
  }

  int stride = 1;
  while (stride < 2*BLOCK_SIZE) {
    __syncthreads();
    int index = (threadIdx.x+1)*stride*2 - 1;
    if (index < 2*BLOCK_SIZE && (index-stride) >= 0) {
      T[index] += T[index-stride];
    }
    stride *= 2;
  }

  stride = BLOCK_SIZE/2;
  while (stride > 0) {
    __syncthreads();
    int index = (threadIdx.x+1)*stride*2 - 1;
    if ((index+stride) < 2*BLOCK_SIZE) {
      T[index+stride] += T[index];
    }
    stride /= 2;
  }

  __syncthreads();

  if ((blockIdx.x * blockDim.x + threadIdx.x)*2+1 < len) {
    output[(blockIdx.x * blockDim.x + threadIdx.x)*2] = T[threadIdx.x*2];
    output[(blockIdx.x * blockDim.x + threadIdx.x)*2+1] = T[threadIdx.x*2+1];
  } else if ((blockIdx.x * blockDim.x + threadIdx.x)*2 < len) {
    output[(blockIdx.x * blockDim.x + threadIdx.x)*2] = T[threadIdx.x*2];
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 DimGridScan(ceil((float)numElements/2/BLOCK_SIZE), 1, 1);
  dim3 DimGridStore(ceil((float)numElements/2/BLOCK_SIZE/BLOCK_SIZE), 1, 1);
  dim3 DimGridAdd(ceil((float)numElements/BLOCK_SIZE), 1, 1);
  dim3 DimGridSingle(1, 1, 1);
  dim3 DimBlock(BLOCK_SIZE, 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  float *deviceAux;
  cudaMalloc((void **)&deviceAux, numElements/2/BLOCK_SIZE * sizeof(float));

  scan<<<DimGridScan,DimBlock>>>(deviceInput, deviceOutput, numElements);
  store<<<DimGridStore,DimBlock>>>(deviceOutput, deviceAux, numElements);
  scan<<<DimGridSingle,DimBlock>>>(deviceAux, deviceAux, numElements/2/BLOCK_SIZE);
  add<<<DimGridAdd,DimBlock>>>(deviceOutput, deviceAux, numElements);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  // if (numElements > 5000) {
  // for (int i = 0; i < numElements; i++) {
  //   printf("%d,%0.f ", i, hostOutput[i]);
  // }
  // printf("\n");
  // }



  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
