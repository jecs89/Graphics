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

	ofstream result("experiment.data");

	for(int i = 0; i < numelem; ++i){

		int size = pow(2, i);

		cout << "# elems: " << size << endl;

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

		cout << "GPU SORTING\n";
	      
		    time_t timer = time(0); 
		
		 	thrust::sort( g_V.begin(), g_V.end(), order() );

			time_t timer2 = time(0); 

		    cout <<"Tiempo total: " << difftime(timer2, timer) << endl;

		    result << size << "," << difftime(timer2, timer) << endl;

	    cout << "CPU SORTING\n";

	    	timer = time(0); 

			sort( c_V.begin(), c_V.end());

			timer2 = time(0); 

		    cout <<"Tiempo total: " << difftime(timer2, timer) << endl;

		    result << size << "," << difftime(timer2, timer) << endl;

	}

	result.close();
    
    return 0;
}