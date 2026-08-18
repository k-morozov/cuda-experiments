#include <iostream>

__global__ void helloKernel(float *p) {
    *p = 42;
    printf("Hello from CUDA kernel!\n");
}

int main() {
    float * src;
    cudaError_t r;

    r = cudaMalloc(&src, sizeof(float));
    if (r != cudaSuccess) {
        return -1;
    }

    helloKernel<<<1, 1>>>(src);
    cudaDeviceSynchronize();

    float dst;

    r = cudaMemcpy(&dst, src, sizeof(float), cudaMemcpyKind::cudaMemcpyDefault);
    if (r != cudaSuccess) {
        return -2;
    }
    std::cout << "CUDA example ready: " << dst << std::endl;
    return 0;
}
