#include <iostream>
#include <algorithm>
#include <random>
#include <iomanip>
#include <string>
#include <sstream>
#include <fstream>
#include <thread>

//./fish 20000 10 200 15 5 


#define PI 3.1416

using namespace std;

template <typename T>
struct point2D{
	T x;
	T y;

    bool operator<(const point2D<T> other) const
    {
        return y < other.y;
    }
};

template <typename T>
struct point3D{
	T x;
	T y;
	T z;
};

typedef point2D<double> p2D;
typedef point3D<double> p3D;

template <typename T>
struct fish
{
	int num;
	T c;	//position
	T v;	//direction
	float s;	//speed
	float theta;	//vision's angle
	float r_r;		//repulsion radius
	float r_p;		//reach radius
	float w_a;		//attraction weight
	float w_o;		//orientation weight

	vector<int> neighborhood_r;	//neighbors in repulsion zone
	vector<int> neighborhood_p; //neighbors in annulus r_r & r_p
};

template <typename T>
void printelem( T elem ){
	// cout << sizeof(elem) << endl;
	// cout << sizeof(elem[0]) << endl;
	if( sizeof(elem) == 1 || sizeof(elem) == 2 || sizeof(elem) == 4 ){
		// cout << "HOLA\n" ;
		// cout << elem << endl;
	}
	else if( sizeof(elem) > 4){
		if( sizeof(elem[0]) == 1 || sizeof(elem[0]) == 2 || sizeof(elem[0]) == 4){
			for( int i = 0; i < elem.size(); ++i){
				cout << elem[i] << "\t";
			}
			cout << endl;
		}
		 else if( sizeof(elem[0]) > 4){
		 	cout << elem.size() << endl;

			// for( int i = 0; i < elem.size(); ++i){
			// 	for( int j = 0; j < elem[0].size(); ++j){
			// 		cout << elem[i][j] << "\t";	
			// 	}
			// 	cout << endl;		
			// }
			// cout << endl;	
		}
	}
}

// template <typename T, typename V>
// double calc_di( vector<T> vec, T i){
// 	for( vector<T>::iterator it = vec.begin(); it != vec.end(); it++ ){

// 	}
// }

double calc_angle( p3D x, p3D y ){
	double den = x.x * y.x + x.y * y.y + x.z * y.z,
		  num = sqrtf( powf(x.x,2) + powf(x.y,2) + powf(x.z,2) ) + sqrtf( powf(y.x,2) + powf(y.y,2) + powf(y.z,2) );

	return acos( den / num );
}

void test_calc_angle(){
	p3D x, y;
	x.x = 2, x.y = 3, x.z = 0, y.x = 2, y.y = 0, y.z = 3;

    double ang = calc_angle( x, y );
    cout << setprecision(4) << ang << "\t" << ang*180/PI << endl;
}

double calc_norm( p3D x, p3D y){
	return sqrtf( powf(x.x-y.x,2) + powf( x.y-y.y,2) + powf( x.z-y.z,2) );
}

template <typename T>
class SchoolFish{
	vector< fish< T > > schoolfish;
	int lim1, lim2;
	double wa, wo;

	int start, end;

	vector<p3D> v_p3D;

	public:
		SchoolFish(int size){
		schoolfish.resize(size);
		v_p3D.resize(size);
		// double w_a, w_o;
	}

		void parallel_init(int start, int end){
			int scale = 10;

			default_random_engine rng(random_device{}()); 		
			uniform_real_distribution<double> dist( lim1, lim2 );

			// cout << "Thread: " << start << "\t" << end << endl;

			for( int i = start; i < end; ++i){
				schoolfish[i].num = i;

				schoolfish[i].c.x = dist(rng);
				schoolfish[i].c.y = dist(rng);
				schoolfish[i].c.z = dist(rng);

				schoolfish[i].v.x = dist(rng)/scale;
				schoolfish[i].v.y = dist(rng)/scale;
				schoolfish[i].v.z = dist(rng)/scale;

				schoolfish[i].s = 0.1;
				schoolfish[i].theta = PI/6;
				schoolfish[i].r_r = 3;
				schoolfish[i].r_p = 5;
				// schoolfish[i].w_a = wa;
				// schoolfish[i].w_o = wo;

				v_p3D[i] = schoolfish[i].c;
			}
		}

		void init( double _wa, double _wo ){
			wa = _wa; wo = _wo;

			lim1 = 50, lim2 = 100;

			//Parallel Process
		    int nThreads = thread::hardware_concurrency();  // # threads
		    vector<thread> ths(nThreads);   // threads vector

		    //dimensions of blocks
		    double incx = double(schoolfish.size()) / nThreads;

		    // int start, end;
		    //Launching threads
		    for ( int i = 0; i < nThreads/2; ++i){
		    	start = floor(i * incx), end = floor((i+1) * incx );
		    	// cout << start << "\t" << end << endl;
		    	//bind(&SchoolFish<T>::parallel_init, this, start, end) );
		        ths[i] = thread( &SchoolFish<T>::parallel_init, this, start, end );
		    }

		    lim1 = -100, lim2 = -50;

		    for ( int i = nThreads/2; i < nThreads; ++i){
		    	start = floor(i * incx), end = floor((i+1) * incx );
		    	// cout << start << "\t" << end << endl;
		    	//bind(&SchoolFish<T>::parallel_init, this, start, end) );
		        ths[i] = thread( &SchoolFish<T>::parallel_init, this, start, end );
		    }
		    
		    //Joining threads
		    for ( int i = 0; i < nThreads; i++ )
		        ths[i].join();
		}


		void print(){
			int dec=4;
			int pos=dec*2;
			cout << setw(pos) << "# Fish" << setw(pos*3) << "C" << setw(pos*3) << "V" << setw(pos) << "S" << setw(pos) << "Theta" << setw(pos) << "R-R" << setw(pos)
				 << setw(pos) << "R-P" << setw(pos) << "W_A" << setw(pos) << "W_O" << endl;

			for( int i = 0; i < schoolfish.size(); ++i){
				cout << setprecision(dec) << setw(pos) << schoolfish[i].num << setw(pos) << schoolfish[i].c.x << setw(pos) << schoolfish[i].c.y << setw(pos) << schoolfish[i].c.z << setw(pos) << schoolfish[i].v.x
					 << setw(pos) << schoolfish[i].v.y << setw(pos) << schoolfish[i].v.z << setw(pos) << schoolfish[i].s << setw(pos) << schoolfish[i].theta 
					 << setw(pos) << schoolfish[i].r_r << setw(pos) << schoolfish[i].r_p << setw(pos) << wa 
					 << setw(pos) << wo << endl;
			}	
		}

		void print2file( ostream& file, int type ){
			int dec=4;
			int pos=dec*2;
			
			//file << setw(pos) << "# Fish" << setw(pos*2) << "C" << endl;
			for( int i = 0; i < schoolfish.size(); ++i){
				if( type == 1){
					file << setprecision(dec) << setw(pos) << schoolfish[i].num << setw(pos) << schoolfish[i].c.x << setw(pos) << schoolfish[i].c.y << setw(pos) << schoolfish[i].c.z << endl;
				}
				else if( type == 2){
					file << setprecision(dec) << schoolfish[i].c.x << "," <<  schoolfish[i].c.y << "," << schoolfish[i].c.z<< endl;	
				}
			}
		}
		vector<float> v_norm(int pos){
			// cout << "Fish" << pos << endl;
			vector<float> ans;
			for( int i = 0; i < schoolfish.size(); ++i){
				if( i != pos ){
					float dist = sqrtf( powf(schoolfish[pos].c.x-schoolfish[i].c.x,2) + powf(schoolfish[pos].c.y-schoolfish[i].c.y,2) + powf(schoolfish[pos].c.z-schoolfish[i].c.z,2) );
					ans.push_back( dist );
				}
				else{
					ans.push_back( 50 );		
				}
			}

			// cout << "ANS Size: " <<  ans.size() << endl;

			return ans;
		}

		void print_distances(){
			for( int i = 0; i < schoolfish.size(); ++i){
				printelem( v_norm(i) );
			}
		}

		void parallel_calc_neighboors(int start, int end, int k){

			for( int i = start; i < end; ++i){
				vector<float> v_dist = ( v_norm(i) );
				v_dist[i] = 10000;

				for( int j = 0; j < v_dist.size(); ++j){
					int act_size = schoolfish[i].neighborhood_r.size()+ schoolfish[i].neighborhood_p.size();
					if( act_size < k ){
						if( v_dist[j] <= schoolfish[i].r_r ){
							// if( schoolfish[i].num != j ){
								schoolfish[i].neighborhood_r.push_back( j );
								// cout << j << "\t";
							// }
						}
						else if( v_dist[j] > schoolfish[i].r_r && v_dist[j] <= schoolfish[i].r_p ){
							// if( schoolfish[i].num != j ){
								schoolfish[i].neighborhood_p.push_back( j );
								// cout << j << "\t";
							// }
						}
					}
					else if( act_size == k ){
						break;
					}
				}

			}
		}

		void calc_neighboors( int k ){

			int nThreads = thread::hardware_concurrency();  // # threads
		    vector<thread> ths(nThreads);   // threads vector

		    //dimensions of blocks
		    double incx = double(schoolfish.size()) / nThreads;

		    int start, end;
		    
		    //Launching threads
		    for ( int i = 0; i < nThreads; ++i){
		    	start = floor(i * incx), end = floor((i+1) * incx );
		    	// cout << start << "\t" << end << endl;

		        ths[i] = thread( &SchoolFish<T>::parallel_calc_neighboors, this, start, end, k );
		    }

		    // for ( int i = nThreads/2; i < nThreads; ++i){
		    // 	start = floor(i * incx), end = floor((i+1) * incx );
		    // 	cout << start << "\t" << end << endl;
		    
		    //     ths[i] = thread( &SchoolFish<T>::parallel_calc_neighboors, this, start, end, k );
		    // }
		    
		    //Joining threads
		    for ( int i = 0; i < nThreads; i++ )
		        ths[i].join();
			
		}

		void print_neighboors(){
			for( int i = 0; i < schoolfish.size(); ++i){
				cout << "Fish " << i << endl;
				
				cout << "Repulsion zone:" << schoolfish[i].neighborhood_r.size() << endl;
				printelem( schoolfish[i].neighborhood_r );

				cout << "P zone:" << schoolfish[i].neighborhood_p.size() << endl;
				printelem( schoolfish[i].neighborhood_p);
			}

		}

		void update_c(){
			for( int i = 0; i < schoolfish.size(); ++i){
				// pair<double,double> d(0,0);
				p3D d; d.x = d.y = d.z = 0;

				if( schoolfish[i].neighborhood_r.size() > 0 ){
					for( int j = 0; j < schoolfish[i].neighborhood_r.size(); ++j){
						int ii = schoolfish[i].num, jj = schoolfish[i].neighborhood_r[j];
						double den = calc_norm( schoolfish[ii].c, schoolfish[jj].c );
						d.x +=  ( schoolfish[jj].c.x - schoolfish[ii].c.x ) / ( den );
						d.y +=  ( schoolfish[jj].c.y - schoolfish[ii].c.y ) / ( den );	
						d.z += ( schoolfish[jj].c.z - schoolfish[ii].c.z ) / ( den );	
					}
					d.x = -1 * d.x; d.y = -1 * d.y; d.z = -1 * d.z;
				}
				else{
					for( int j = 0; j < schoolfish[i].neighborhood_r.size(); ++j){
						int ii = schoolfish[i].num, jj = schoolfish[i].neighborhood_r[j];
						double den = calc_norm( schoolfish[ii].c, schoolfish[jj].c );

						int norm_v = (schoolfish[i].v.x/ sqrtf( powf(schoolfish[i].v.x,2) + powf(schoolfish[i].v.y,2) + powf(schoolfish[i].v.z,2) ) );

						d.x +=  wa * ( schoolfish[jj].c.x - schoolfish[ii].c.x ) / ( den ) + wo * norm_v;
						d.y +=  wa * ( schoolfish[jj].c.y - schoolfish[ii].c.y ) / ( den ) + wo * norm_v;	
						d.z +=  wa * ( schoolfish[jj].c.z - schoolfish[ii].c.z ) / ( den ) + wo * norm_v;	
					}
					d.x = -1 * d.x; d.y = -1 * d.y; d.z = -1 * d.z;

				}

				if( calc_angle( schoolfish[i].c, d ) <= schoolfish[i].theta ){
					 schoolfish[i].c.x = schoolfish[i].c.x + schoolfish[i].s * d.x;
					 schoolfish[i].c.y = schoolfish[i].c.y + schoolfish[i].s * d.x;
					 schoolfish[i].c.z = schoolfish[i].c.z + schoolfish[i].s * d.x;
				}
			}
		}

		void movement( double t ){
			for( int i = 0; i < schoolfish.size(); ++i){
				// double cx = schoolfish[i].c.x, cy = schoolfish[i].c.second;

				check_limits( schoolfish[i].c, schoolfish[i].v ) ;

				schoolfish[i].c.x += schoolfish[i].v.x * schoolfish[i].s * t;
				schoolfish[i].c.y += schoolfish[i].v.y * schoolfish[i].s * t;
				schoolfish[i].c.z += schoolfish[i].v.z * schoolfish[i].s * t;
			}
			// cout << "Good Movement" << endl;
		}

		void check_limits( p3D& p, p3D& v ){

			int lmin = -100, lmax = 100;

			if( p.x + v.x < lmin || p.x + v.x > lmax ){
				v.x= v.x* -1;
			}
			else if( p.y + v.y < lmin || p.y + v.y > lmax){
				v.y = v.y * -1;
			}
			else if( p.z + v.z < lmin || p.z + v.z > lmax){
				v.z = v.z * -1;
			}
		}

};

template <typename T>
inline void str2num(string str, T& num){
	if ( ! (istringstream(str) >> num) ) num = 0;
}

int main( int argc, char** argv ){

	vector<time_t> timer(10);
	timer[0] = time(0); 

	//Params of SchoolFish
	int num_fish, k, iter;
	string par = argv[1]; str2num( par, num_fish);
	par = argv[2]; str2num( par, k);

	float t = 1;
	par = argv[3]; str2num( par, iter);

	float wa, wo;
	par = argv[4]; str2num( par, wa);
	par = argv[5]; str2num( par, wo);

	//Initialization of values
	// SchoolFish<p2D> myschool( num_fish );
	SchoolFish<p3D> myschool( num_fish );
	myschool.init(wa,wo);
	// myschool.print();

    ofstream result("movement.data");

    timer[1] = time(0); 
    cout <<"\nTiempo Inicializacio\'n: " << difftime(timer[1], timer[0]) << endl;

    timer[2] = 0;
    timer[3] = 0;
    timer[4] = 0;

	for( int i = 0; i < iter; ++i){
		// cout << i << endl;
		// myschool.print_distances();
		timer[0] = time(0); 
		myschool.movement(t);
		timer[1] = time(0); 

		// cout <<"\nTiempo Movimiento " << difftime(timer[1], timer[0]) << endl;

		timer[2] += difftime(timer[1], timer[0]);

		timer[0] = time(0);
		myschool.calc_neighboors(k);
		timer[1] = time(0);

		// cout <<"\nTiempo Calc Vecinos " << difftime(timer[1], timer[0]) << endl;

		timer[3] += difftime(timer[1], timer[0]);

		// myschool.print_neighboors();
		timer[0] = time(0); 
		myschool.update_c();
		timer[1] = time(0);

		// cout <<"\nTiempo Act Pos " << difftime(timer[1], timer[0]) << endl;

		timer[4] += difftime(timer[1], timer[0]);

		// myschool.print();
		myschool.print2file( result, 2);
	}

	cout <<"\nTiempo T Movimiento " << timer[2] << endl;
	cout <<"\nTiempo T Calc Vecinos  " << timer[3] << endl;
	cout <<"\nTiempo T Act Pos " << timer[4] << endl;

	result.close();	

	return 0;
}