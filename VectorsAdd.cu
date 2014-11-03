#include <iostream>
#include <math.h>
#include <iomanip>

using namespace std;

__global__ void add(int *a, int *b, int *c, int n){
	int index = threadIdx.x + blockIdx.x*blockDim.x;

	c[index] = a[index] + b[index];
}

__global__ void print(int *a){
 	printf("%d \n", blockIdx.x);
}

#define N (10)
#define M (2)

int main(void){

	time_t timer = time(0);	

	int *a,*b,*c;					// host copies of a,b,c
	int *d_a, *d_b, *d_c;		// device copies of a,b,c
	int size = N * sizeof(int);

	// Allocate space for device copies of a,b,c
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);

	// Alloc space for host copies of a,b,c and setup input

	a = (int *)malloc(size);
	b = (int *)malloc(size);
	c = (int *)malloc(size);

	for(int i=0; i<N; ++i)
	{
		a[i] = i*i;
		b[i] = i*2;
	}

	// Copy inputs to device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);  // Args: Dir. destino, Dir. origen, tamano de dato, sentido del envio
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	// Launch add() kernel on GPU
	add<<<(N+M-1)/M,M>>> (d_a, d_b, d_c, N);
	print<<<N,1>>> (d_a);

	// Copy result back to host
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

	for(int i=0; i<N; ++i)
		std::cout << setw(6) << a[i];

	std::cout << std::endl;

	for(int i=0; i<N; ++i)
		std::cout << setw(6) << b[i];

	std::cout << std::endl;

	for(int i=0; i<N; ++i)
		std::cout << setw(6) << c[i];

	std::cout << std::endl;

	// Cleanup
	free(a);
	free(b);
	free(c);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	time_t timer2 = time(0);
	cout <<"Tiempo total: " << difftime(timer2, timer) << endl;

	return 0;
}