// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256

//@@ insert code here

__global__ void castChar(float *inputImage, unsigned char *ucharImage, int imageWidth, int imageHeight, int imageChannels) {
  int ii = blockIdx.x * blockDim.x + threadIdx.x;

  if (ii < imageWidth*imageHeight*imageChannels) {
      ucharImage[ii] = (unsigned char) (255 * inputImage[ii]);

  }    
}

__global__ void convert(unsigned char *ucharImage, unsigned char *grayImage, int imageWidth, int imageHeight) {
  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  int jj = blockIdx.y * blockDim.y + threadIdx.y;

  int idx = jj * imageWidth + ii;

  if (idx < imageWidth*imageHeight) {
    grayImage[idx] = (unsigned char) (0.21*ucharImage[3*idx] + 0.71*ucharImage[3*idx+1] + 0.07*ucharImage[3*idx+2]);
  }
}

__global__ void histo_kernal(unsigned char *buffer, unsigned int *histo, int imageWidth, int imageHeight) {
  __shared__ unsigned int histo_private[256];

  if (threadIdx.x < 256) {  
      histo_private[threadIdx.x] = 0;
  }

  __syncthreads();

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  int stride = blockDim.x;

  while (i < imageWidth*imageHeight) {
      atomicAdd( &(histo_private[buffer[i]]), 1 );
      i += stride;
  }

  __syncthreads();

  if (threadIdx.x < 256) {
      atomicAdd( &(histo[threadIdx.x]), histo_private[threadIdx.x] );
  }    
}

__global__ void scan(unsigned int *input, float *output, int len, int width, int height) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host

  __shared__ float T[2*128];

  if ((blockIdx.x * blockDim.x + threadIdx.x)*2+1 < len) {
    T[threadIdx.x*2] = (float) input[(blockIdx.x * blockDim.x + threadIdx.x)*2]/ (width*height);
    T[threadIdx.x*2+1] = (float) input[(blockIdx.x * blockDim.x + threadIdx.x)*2+1]/ (width*height);
  } else if ((blockIdx.x * blockDim.x + threadIdx.x)*2 < len) {
    T[threadIdx.x*2] = (float) input[(blockIdx.x * blockDim.x + threadIdx.x)*2] / (width*height);
  }

  int stride = 1;
  while (stride < 2*128) {
    __syncthreads();
    int index = (threadIdx.x+1)*stride*2 - 1;
    if (index < 2*128 && (index-stride) >= 0) {
      T[index] += T[index-stride];
    }
    stride *= 2;
  }

  stride = 128/2;
  while (stride > 0) {
    __syncthreads();
    int index = (threadIdx.x+1)*stride*2 - 1;
    if ((index+stride) < 2*128) {
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

__global__ void equalize(unsigned char *ucharImage, float *cdf, int imageWidth, int imageHeight, int imageChannels) {
  int ii = blockIdx.x * blockDim.x + threadIdx.x;

  if (ii < imageWidth*imageHeight*imageChannels) {
    ucharImage[ii] = (unsigned char) min(max(255.0*(cdf[ucharImage[ii]] - cdf[0])/(1.0 - cdf[0]), 0.0), 255.0);
  }
}

__global__ void castFloat(unsigned char *ucharImage, float *outputImage, int imageWidth, int imageHeight, int imageChannels) {
  int ii = blockIdx.x * blockDim.x + threadIdx.x;

  if (ii < imageWidth*imageHeight*imageChannels) {
    outputImage[ii] = (float) (ucharImage[ii]/255.0);
  }    
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float *deviceInputImage;
  unsigned char *deviceUcharImage;
  unsigned char *deviceGrayImage;
  unsigned int *deviceHisto;
  float *deviceCdf;
  float *deviceoutputImage;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  cudaMalloc((void **) &deviceInputImage, imageWidth*imageHeight*imageChannels * sizeof(float));
  cudaMalloc((void **) &deviceUcharImage, imageWidth*imageHeight*imageChannels * sizeof(unsigned char));
  cudaMalloc((void **) &deviceGrayImage, imageWidth*imageHeight* sizeof(unsigned char));
  cudaMalloc((void **) &deviceHisto, 256*sizeof(unsigned int));
  cudaMalloc((void **) &deviceCdf, 256*sizeof(float));
  cudaMalloc((void **) &deviceoutputImage, imageWidth*imageHeight*imageChannels * sizeof(float));

  cudaMemset(deviceHisto, 0, 256*sizeof(unsigned int));

  cudaMemcpy(deviceInputImage, hostInputImageData, imageWidth*imageHeight*imageChannels * sizeof(float), cudaMemcpyHostToDevice);

  dim3 DimGrid1d(ceil((float)imageWidth*imageHeight*imageChannels/1024), 1, 1);
  dim3 DimGrid2d(ceil((float)imageWidth/32), ceil((float)imageHeight/32), 1);
  dim3 DimGridSingle(1, 1, 1);

  dim3 DimBlock1d(1024, 1, 1);
  dim3 DimBlock2d(32, 32, 1);
  dim3 DimBlock128(128, 1, 1);

  castChar<<<DimGrid1d,DimBlock1d>>>(deviceInputImage, deviceUcharImage, imageWidth, imageHeight, imageChannels);
  convert<<<DimGrid2d,DimBlock2d>>>(deviceUcharImage, deviceGrayImage, imageWidth, imageHeight);
  histo_kernal<<<DimGridSingle,DimBlock1d>>>(deviceGrayImage, deviceHisto, imageWidth, imageHeight);
  scan<<<DimGridSingle,DimBlock128>>>(deviceHisto, deviceCdf, 256, imageWidth, imageHeight);
  equalize<<<DimGrid1d,DimBlock1d>>>(deviceUcharImage, deviceCdf, imageWidth, imageHeight, imageChannels);
  castFloat<<<DimGrid1d,DimBlock1d>>>(deviceUcharImage, deviceoutputImage, imageWidth, imageHeight, imageChannels);

  cudaDeviceSynchronize();

  cudaMemcpy(hostOutputImageData, deviceoutputImage, imageWidth*imageHeight*imageChannels * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(deviceInputImage);
  cudaFree(deviceUcharImage);
  cudaFree(deviceGrayImage);
  cudaFree(deviceHisto);
  cudaFree(deviceCdf);
  cudaFree(deviceoutputImage);

  wbSolution(args, outputImage);

  //@@ insert code here
  free(hostInputImageData);
  free(hostOutputImageData);

  return 0;
}
