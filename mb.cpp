#include <iostream>
#include <fstream>
#include <iomanip>
#include <math.h>
#include <vector>

#define N 100	//
#define R 100	//

#define dim1 10 	//X 	 
#define dim2 10 	//Y
#define dim3 10 	//Z

#define inc 0.20 	//distance between points of cube
#define spc 8		//space to setw

#define radio 0.8 	//radius of circle
#define n_circles 25

#define FOR(i,n,inc) for( i = -n; i < n; i=i+inc)

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

//basic function of f_r
double f_r( double r){
	return pow( 1 - pow( r/R, 2), 2);
}

//check if point is in the border of circle
bool in_circle( point3D center, double radius, point3D point){
	return ( pow( center.x - point.x, 2) + pow( center.y - point.y, 2) + pow( center.z - point.z, 2) ) == pow( radius, 2) ? true : false;
}


int main(){

	time_t timer = time(0);	

	ofstream mb_file("data.obj");

	double i, j, k;

	FOR( i, n_circles, 1){
		FOR( j, n_circles, 1){
			FOR( k, n_circles, 1){
				v_circles.push_back( point3D( i/dim1, j/dim2, k/dim3) );
			}
		}
	}

	if (mb_file.is_open()){
    	cout << "Successfully opened\n";
    }
 

	/*FOR( i < N ){
		mb_file << "v" << setw(8) << i*1.0 << setw(8) << f_r(i) << setw(8) << i*1.0 << endl;
	}
*/

	FOR( i, dim1, inc){
		FOR( j, dim2, inc){
			FOR( k, dim3, inc){
				for( int c = 0; c < v_circles.size(); c++){
//					mb_file << "v" << setw(spc) << (double)i << setw(spc) << (double)j << setw(spc) << (double)k << endl;
					if( in_circle( v_circles[c], radio, point3D( i, j, k) ) ){
						mb_file << "v" << setw(spc) << (double)i << setw(spc) << (double)j << setw(spc) << (double)k << endl;  				
						v_point.push_back( point3D( i, j, k) );
					}
				}
			}
		}
	}

	cout << v_point.size() << endl;


	mb_file.close();

	time_t timer2 = time(0);
	cout <<"Tiempo total: " << difftime(timer2, timer) << endl;
	
	return 0;
}