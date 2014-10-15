#include <iostream>
#include <math.h>
#include <iomanip>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <fstream>
#include "random"

#define space 15

#define dim1 50		//X 	 
#define dim2 50 	//Y
#define dim3 50 	//Z

#define N ( 100000 )
#define M ( 512 )


#define incr 0.2
#define rad_circles 4

#define FOR(i,n,inc) for( i = 0; i < n; i=i+inc)

//int idx_radius = 0;

using namespace std;

struct point3D{
	
	double x, y,z;
	point3D( double _x, double _y, double _z){
bn		x = _x;
		y = _y;
		z = _z;
	}
};

vector<point3D> v_point;	//vector of points
vector<point3D> v_circles;	//vector of circles' center
ofstream mb_file("data.obj");

//double thresh = 0.5;

__global__ void add( point3D* points, point3D* circles, point3D* radius, point3D* intersection, int n){
	
	int index = threadIdx.x + blockIdx.x*blockDim.x;

	double dif_x, dif_y, dif_z, value=0.0, thresh = 0.3;

	
	dif_x = (points[index].x - circles[index].x);
	dif_y = (points[index].y - circles[index].y);
	dif_z = (points[index].z - circles[index].z);
	//r_radius = radius[ index ].x;


	/*if( dif_x*dif_x + dif_y*dif_y + dif_z*dif_z < r_radius*r_radius){

		//mb_file << "v" << setw(space) << (double)points[index].x << setw(space) << (double)points[index].y << setw(space) << (double)points[index].z << endl;  				
		intersection[index].x = points[index].x;
		intersection[index].y = points[index].y;
	intersection[index].z = points[index].z;

		//v_point.push_back( point3D( points[index].x, points[index].y, points[index].z ) );
	}
	//printf("%d %d %d %s", points[index].x, points[index].y, points[index].z, "\n");
	*/

	value = sqrt( dif_x*dif_x + dif_y*dif_y + dif_z*dif_z );
	
	if( value < r_radius){

		value = 1 - ( (value*value) / (rad_circles*rad_circles) );



		if( value * value < thresh){
			intersection[index].x = points[index].x;
			intersection[index].y = points[index].y;
			intersection[index].z = points[index].z;
		}
	}

//	printf( "%i%s", index, "\n");
}

void init_ppoint3D( point3D*& points, int size ){
	for( int i = 0; i < size; i++){
		points[i] = point3D( -1, -1, -1);
	}
	cout << "vector initialized \n";
}

int main(void){

	time_t timer = time(0);	

	//a : points , b : circles, c : radius
	point3D *a, *b, *c, *d;					// host copies of a,b,c
	point3D *d_a, *d_b, *d_c, *d_d;		// device copies of a,b,c


	//If index starts with negative number, first line
	//int size_matrix = ( 2 * dim1 ) * ( 2 * dim2 ) * ( 2 * dim3 ) * (
	int size_matrix = ( dim1/incr ) * ( dim2/incr ) * ( dim3/incr ); // number of elements
	cout << "# elements: " << size_matrix << endl;
	
	int size = size_matrix * sizeof(point3D);
	cout << "size of matrix: " << size << endl;

	// Allocate space for device copies of a,b,c
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);
	cudaMalloc((void **)&d_d, size);

	// Alloc space for host copies of a,b,c and setup input

	a = (point3D *)malloc(size);
	b = (point3D *)malloc(size);
	c = (point3D *)malloc(size);
	d = (point3D *)malloc(size);

	init_ppoint3D( d , size_matrix);

	/*for( int m = 0; m < size_matrix; m++ ){
		d[m].x = d[m].y = d[m].z = -1;		
	}
*/
    int index = 0;
    //double x,y,z;
    //x = y = z = 10;

    default_random_engine rng( random_device{}() ); 
    uniform_int_distribution<int> dist(-dim1, dim1); 

    cout << "Filling vectors" << endl;
	
	for( double i = 0; i < dim1; i=i+incr){
		for( double j = 0; j < dim2; j=j+incr){
			for( double k = 0; k < dim3; k=k+incr){
				
			 	a[index].x = i ;
				a[index].y = j ;
				a[index].z = k ;

				b[index].x = dist(rng) ;
				b[index].y = dist(rng) ;
				b[index].z = dist(rng) ;

				c[index].x = rad_circles;
				c[index].y = c[index].z = 0;

				//mb_file << "v" << setw(space) << (double)a[index].x << setw(space) << (double)a[index].y << setw(space) << (double)a[index].z << endl;  				
				//mb_file << "v" << setw(space) << (double)b[index].x << setw(space) << (double)b[index].y << setw(space) << (double)b[index].z << endl;  				
				//cout << index <<"\n";
				index++;
			}		
		}
	}

	cout << index <<"\n";
	cout << "vectors ready" << endl;

	// Copy inputs to device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);  // Args: Dir. destino, Dir. origen, tamano de dato, sentido del envio
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, c, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_d, d, size, cudaMemcpyHostToDevice);


	// Launch add() kernel on GPU
	//add<<<(N+M-1)/M,M>>> (d_a, d_b, d_c, size_matrix);
	add<<< 15259, 1024>>> (d_a, d_b, d_c, d_d, size_matrix);
	//print<<<N,1>>> (d_a);

	// Copy result back to host
	cudaMemcpy(d, d_d, size, cudaMemcpyDeviceToHost);
/*	
	for(int i=0; i < size_matrix; ++i){
		std::cout << setw(space) << a[i].x << setw(space) << a[i].y << setw(space) << a[i].z << endl ;
	}

	std::cout << std::endl;

	for(int i=0; i < size_matrix; ++i){
		std::cout << setw(space) << b[i].x << setw(space) << b[i].y << setw(space) << b[i].z << endl ;
	}

	std::cout << std::endl;

	for(int i=0; i < size_matrix; ++i){
		std::cout << setw(space) << c[i].x << setw(space) << c[i].y << setw(space) << c[i].z << endl ;
	}

	std::cout << std::endl;

	for(int i=0; i < size_matrix; ++i){
		std::cout << setw(space) << d[i].x << setw(space) << d[i].y << setw(space) << d[i].z << endl ;
	}

	std::cout << std::endl;
*/

	for(int i = 0; i < size_matrix; ++i){
		if( d[i].x != -1 && d[i].y != -1 && d[i].z != -1 ){
			mb_file << "v" << setw(space) << d[i].x << setw(space) << d[i].y << setw(space) << d[i].z << endl ;
		}
	}

	std::cout << std::endl;

	// Cleanup
	free(a);
	free(b);
	free(c);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	mb_file.close();

	time_t timer2 = time(0);
	cout <<"Tiempo total: " << difftime(timer2, timer) << endl;

	return 0;
}