#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <algorithm>
#include <sstream>

#include <fstream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

using namespace std;

//Fill a vector with random numbers in the range [lower, upper]
void rnd_fill(thrust::host_vector<double> &V, const double lower, const double upper,  int seed) {

    //Create a unique seed for the random number generator
    srand(time(NULL));
    
    size_t elem = V.size();
    for( size_t i = 0; i < elem; ++i){
        V[i] = (double) rand() / (double) RAND_MAX;
    }
}


template <typename T>
inline void str2num(string str, T& num){
	if ( ! (istringstream(str) >> num) ) num = 0;
}

// template <typename T>
struct order{
__device__
	bool operator()(double x, double y){
		return x < y;
	}
};

int main( int argc, char** argv ){

	int numelem;
	string param = argv[1]; str2num( param, numelem);

	int numexp;
	param = argv[2]; str2num( param, numexp);

	ofstream result("experiment.data");

	vector<time_t> timer(6);
	timer[0] = 0;
	timer[5] = 0; 

	for(int i = 0; i < numelem; ++i){
		int size = pow(2, i);
		cout << "# elems: " << size << endl;
		
		for( int j = 0; j < numexp; ++j){

			

			//Initialization
			thrust::host_vector<double> h_V;
			thrust::device_vector<double> g_V;

			h_V.resize( size );

			int seed = time(0);
			rnd_fill( h_V, 0.1, 1.0, seed);

			g_V = h_V;

			vector<double> c_V;
			c_V.resize( size );

			for( int i = 0; i < h_V.size(); ++i){
				c_V[i] = h_V[i];
			}	

			

				timer[1] = time(0);
		      
			 	thrust::sort( g_V.begin(), g_V.end(), order() );

			 	timer[2] = time(0);

			    // result << size << "," << difftime(timer[2], timer[1]) << endl;

			    timer[0] += difftime(timer[2], timer[1]);

	    	

	    		timer[3] = time(0);

				sort( c_V.begin(), c_V.end());

				timer[4] = time(0);

			    // result << size << "," << difftime(timer[4], timer[3]) << endl;

				timer[5] += difftime(timer[4], timer[3]);			    
		}
		
		
	}

	cout << "GPU SORTING\n";
		cout <<"Tiempo total: " << timer[0] << endl;
	cout << "CPU SORTING\n";
		cout <<"Tiempo total: " << timer[5] << endl;

	result.close();
    
    return 0;
}