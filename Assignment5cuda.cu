#include <iostream>
#include <ctime>
#include <cstdlib>
#include <chrono>

using namespace std;

#define GRIDSIZE 16 * 1024
#define BLOCKSIZE 1024
#define TOTALSIZE (GRIDSIZE * BLOCKSIZE)

//random number generator
__host__ void randGen(float *dst, int size){ while(size--)  dst[size] = (rand() % 1000) / 10.0; }

//adjDiff in Host
__host__ void adjDiffHost(float *res, float *src, int size){
    for(int i = 1; i < size; i++)   res[i] = src[i] - src[i-1];
}

//adjDiff in Device(global memory)
__global__ void addKernelGlobalVersion(float* res, float *src){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i > 0)   res[i] = src[i] - src[i-1];
}

//adjDiff in Device(shared memory)
__global__ void addKernelSharedVersion(float* res, float *src){
    __shared__ float s_data[BLOCKSIZE];
    unsigned int tx = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tx;

    //shared memory insertion
    s_data[tx] = src[i];
    __syncthreads();

    if(tx > 0)  res[i] = s_data[tx] - s_data[tx-1];     //normal case
    else if(i > 0) res[i] = s_data[tx] - src[i-1];        //block's first thread case
}

int main(void){
    //host variable initialize
    srand((unsigned int)time(NULL));
    float* pSource = new float[TOTALSIZE];
    float* pResult = new float[TOTALSIZE];
    float* pGlobalResult = new float[TOTALSIZE];
    float* pSharedResult = new float[TOTALSIZE];
    
    pResult[0] = 0.0f;
    randGen(pSource, TOTALSIZE);

    //device variable initialize
    float *pSourceDev, *pGlobalResultDev, *pSharedResultDev;
    cudaMalloc((void **)&pSourceDev, TOTALSIZE * sizeof(float));
    cudaMalloc((void **)&pGlobalResultDev, TOTALSIZE * sizeof(float));
    cudaMalloc((void **)&pSharedResultDev, TOTALSIZE * sizeof(float));
    
    //dimension setting
    dim3 dimGrid(GRIDSIZE, 1, 1);
    dim3 dimBlock(BLOCKSIZE, 1, 1);

    //host time check
    chrono::system_clock::time_point hostStart = chrono::system_clock::now();
    adjDiffHost(pResult, pSource, TOTALSIZE);
    chrono::system_clock::time_point hostEnd = chrono::system_clock::now();
    chrono::nanoseconds hostNano = hostEnd - hostStart;

    //print host time
    cout << "HOST RESULT" << endl;
    cout << "elapsed time(nsec) : " << hostNano.count() << endl;
    cout << "i = 1 : " << pResult[1] << " = " << pSource[1] << " - " << pSource[0] << endl;
    cout << "i = " << TOTALSIZE-1 << " : " << pResult[TOTALSIZE-1] << " = " << pSource[TOTALSIZE-1] << " - " << pSource[TOTALSIZE-2] << endl;
    cout << "i = " << TOTALSIZE/2 << " : " << pResult[TOTALSIZE/2] << " = " << pSource[TOTALSIZE/2] << " - " << pSource[TOTALSIZE/2-1] << endl << endl;

    //device(global memory) time check
    chrono::system_clock::time_point globalStart = chrono::system_clock::now();
    cudaMemcpy(pSourceDev, pSource, TOTALSIZE * sizeof(float), cudaMemcpyHostToDevice);
    addKernelGlobalVersion<<<dimGrid, dimBlock>>>(pGlobalResultDev, pSourceDev);
    cudaMemcpy(pGlobalResult, pGlobalResultDev, TOTALSIZE * sizeof(float), cudaMemcpyDeviceToHost);
    chrono::system_clock::time_point globalEnd = chrono::system_clock::now();
    chrono::nanoseconds globalNano = globalEnd - globalStart;
    cudaDeviceSynchronize();

    //print device(global memory) time
    cout << "GLOBAL DEVICE VERSION RESULT" << endl;
    cout << "elapsed time(nsec) : " << globalNano.count() << endl;
    cout << "i = 1 : " << pGlobalResult[1] << " = " << pSource[1] << " - " << pSource[0] << endl;
    cout << "i = " << TOTALSIZE-1 << " : " << pGlobalResult[TOTALSIZE-1] << " = " << pSource[TOTALSIZE-1] << " - " << pSource[TOTALSIZE-2] << endl;
    cout << "i = " << TOTALSIZE/2 << " : " << pGlobalResult[TOTALSIZE/2] << " = " << pSource[TOTALSIZE/2] << " - " << pSource[TOTALSIZE/2-1] << endl << endl;
    
    //device (shared memory) time check
    chrono::system_clock::time_point sharedStart = chrono::system_clock::now();
    cudaMemcpy(pSourceDev, pSource, TOTALSIZE * sizeof(float), cudaMemcpyHostToDevice);
    addKernelSharedVersion<<<dimGrid, dimBlock>>>(pSharedResultDev, pSourceDev);
    cudaMemcpy(pSharedResult, pSharedResultDev, TOTALSIZE * sizeof(float), cudaMemcpyDeviceToHost);
    chrono::system_clock::time_point sharedEnd = chrono::system_clock::now();
    chrono::nanoseconds sharedNano = sharedEnd - sharedStart;
    cudaDeviceSynchronize();

    //print device(shared memory) time
    cout << "SHARED DEVICE VERSION RESULT" << endl;
    cout << "elapsed time(nsec) : " << sharedNano.count() << endl;
    cout << "i = 1 : " << pSharedResult[1] << " = " << pSource[1] << " - " << pSource[0] << endl;
    cout << "i = " << TOTALSIZE-1 << " : " << pSharedResult[TOTALSIZE-1] << " = " << pSource[TOTALSIZE-1] << " - " << pSource[TOTALSIZE-2] << endl;
    cout << "i = " << TOTALSIZE/2 << " : " << pSharedResult[TOTALSIZE/2] << " = " << pSource[TOTALSIZE/2] << " - " << pSource[TOTALSIZE/2-1] << endl << endl;
    
    //host memory deallocation
    delete[] pSource;
    delete[] pResult;
    delete[] pGlobalResult;
    delete[] pSharedResult;

    //device memory deallocation
    cudaFree(pSourceDev);
    cudaFree(pGlobalResultDev);
    cudaFree(pSharedResultDev);
}

