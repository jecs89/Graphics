#include <iostream>
#include <math.h>
#include <iomanip>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <fstream>
#include "random"

#define space 15

#define dim1 10		//X 	 
#define dim2 10 	//Y
#define dim3 10 	//Z

#define N ( 100000 )
#define M ( 512 )

#define incr 0.3

#define rad_circles 3
#define num_circles 20

#define FOR(i,n,inc) for( i = -n; i < n; i=i+inc)

//int idx_radius = 0;

using namespace std;

struct point3D{
	
	double x, y,z;
	point3D( double _x, double _y, double _z){
		x = _x;
		y = _y;
		z = _z;
	}
};

vector<point3D> v_point;	//vector of points
vector<point3D> v_circles;	//vector of circles' center
ofstream mb_file("data.obj");

//double thresh = 0.5;

__global__ void add( point3D* points, point3D* circles, point3D* intersection){
	
	int index = threadIdx.x + blockIdx.x*blockDim.x;

	//if( circles[index].x != -1 && circles[index].y != -1 && circles[index].z != -1 ){
	for( int i = 0 ; i < num_circles; i++){

		double dif_x, dif_y, dif_z, value=0.0, thresh = 0.01;
		
		dif_x = (points[index].x - circles[i].x);
		dif_y = (points[index].y - circles[i].y);
		dif_z = (points[index].z - circles[i].z);

		//printf("%f %f %f %c", points[index].x, points[index].y, points[index].z, '\n');
		
		value = sqrt( dif_x*dif_x + dif_y*dif_y + dif_z*dif_z );

		//printf("%d\n", value);
		
		if( value < rad_circles){

			value = 1 - ( (value*value) / (rad_circles*rad_circles) );

			if( value * value < thresh){
				intersection[index].x = points[index].x;
				intersection[index].y = points[index].y;
				intersection[index].z = points[index].z;
			}
		}

	}

	
}

void init_ppoint3D( point3D*& points, int size ){
	for( int i = 0; i < size; i++){
		points[i] = point3D( -1, -1, -1);
	}
	cout << "vector initialized \n";
}

void print_flag( string message, int type){
	if( type == 1) cout << message << endl;
	else if( type == 2) cout << message << "\t";
}

int main(void){

	time_t timer = time(0);	

	point3D *points, *circles, *intersections;				// host copies of a,b,c
	point3D *d_points, *d_circles, *d_intersections;		// device copies of a,b,c


	//If index starts with negative number, first line
	//int size_matrix = ( 2 * dim1 ) * ( 2 * dim2 ) * ( 2 * dim3 ) * (
	//int size_matrix = ( dim1/incr ) * ( dim2/incr ) * ( dim3/incr ); // number of elements
	int size_matrix = 8 * ( dim1/incr ) * ( dim2/incr ) * ( dim3/incr );
	cout << "# elements: " << size_matrix << endl;
	
	int size = size_matrix * sizeof(point3D);
	cout << "size of matrix: " << size << endl;

	// Allocate space for device copies of a,b,c
	cudaMalloc( (void **)&d_points, size);
	cudaMalloc( (void **)&d_circles, size_matrix * sizeof(point3D) );
	cudaMalloc( (void **)&d_intersections, size);	

	// Alloc space for host copies of a,b,c and setup input
	points   	  = (point3D *)malloc(size);
	circles 	  = (point3D *)malloc(size_matrix * sizeof(point3D));
	intersections = (point3D *)malloc(size);	

	init_ppoint3D( intersections, size_matrix);

	/*    default_random_engine rng( random_device{}() ); 
    uniform_int_distribution<int> dist(-dim1, dim1); 
	*/

    init_ppoint3D( circles, size_matrix);

    for( int i = 0 ; i < num_circles ; i++){
		circles[i].x = ( i * i ) / 5;
		circles[i].y = ( i + 1 ) * ( i + 1) / 5;
		circles[i].z = ( i + 2 ) * ( i + 2) / 5;
	}
	
    int index = 0;
    
    cout << "Filling vectors" << endl;


	
	for( double i = -dim1; i < dim1; i=i+incr){
		for( double j = -dim2; j < dim2; j=j+incr){
			for( double k = -dim3; k < dim3; k=k+incr){
				
			 	points[index].x = i ;
				points[index].y = j ;
				points[index].z = k ;

				//printf("%f %f %f %c", points[index].x, points[index].y, points[index].z, '\n');

				//mb_file << "v" << setw(space) << (double)a[index].x << setw(space) << (double)a[index].y << setw(space) << (double)a[index].z << endl;  				
				//mb_file << "v" << setw(space) << (double)b[index].x << setw(space) << (double)b[index].y << setw(space) << (double)b[index].z << endl;  				
				//cout << index <<"\n";
				index++;
			}		
		}
	}

	cout << index <<"\n";
	cout << "vectors ready" << endl;

	int cocient = 1000;
	int num_blocks = index / cocient;
	int num_threadsxblock = cocient; 

	// Copy inputs to device
	cudaMemcpy(d_points, points, size, cudaMemcpyHostToDevice);  // Args: Dir. destino, Dir. origen, tamano de dato, sentido del envio
	cudaMemcpy(d_circles, circles, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_intersections, intersections, size, cudaMemcpyHostToDevice);	

	// Launch add() kernel on GPU
	//for( int i = 0 ; i < num_circles; i++){
	add<<< num_blocks, num_threadsxblock>>> (d_points, d_circles, d_intersections);
	//}
	
	// Copy result back to host
	cudaMemcpy(intersections, d_intersections, size, cudaMemcpyDeviceToHost);
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
		if( intersections[i].x != -1 && intersections[i].y != -1 && intersections[i].z != -1 ){
			mb_file << "v" << setw(space) << intersections[i].x << setw(space) << intersections[i].y << setw(space) << intersections[i].z << endl ;
		}
	}

	std::cout << std::endl;

	// Cleanup
	free(points);
	free(circles);
	free(intersections);
	cudaFree(d_points);
	cudaFree(d_circles);
	cudaFree(d_intersections);

	mb_file.close();

	time_t timer2 = time(0);
	cout <<"Tiempo total: " << difftime(timer2, timer) << endl;

	return 0;
}
