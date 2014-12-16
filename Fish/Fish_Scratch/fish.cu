#include <iostream>
#include <algorithm>
#include <random>
#include <iomanip>
#include <string>
#include <sstream>
#include <fstream>

//./fish 20000 10 200 15 5 


#define PI 3.1416

using namespace std;

typedef pair<float,float> p2D;
typedef pair< float, pair<float,float> > p3D;

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

__global__ void cuda_norm( fish<p3D>* schoolfish, int pos, float* c){

	int index = threadIdx.x + blockIdx.x*blockDim.x;
	printf( "%s%i%c", "Hola", index, '\n' );

	float dist = sqrtf( powf(schoolfish[pos].c.first-schoolfish[index].c.first,2) + 
						powf(schoolfish[pos].c.second.first-schoolfish[index].c.second.first,2) + 
						powf(schoolfish[pos].c.second.second-schoolfish[index].c.second.second,2) );

	c[index] = dist ;
	printf( "%d%c", c[index], '\n');
}

template <typename Type>
void ptr2vector( Type*& ptr, vector<Type>& vec, int size){
	for( int i = 0; i < size; ++i){
		vec.push_back( ptr[i] );
	}
}

template <typename Type>
void vector2ptr( vector<Type>& vec, Type*& ptr, int size){

	ptr = ( Type* )malloc( size * sizeof(Type) );

	for( int i = 0; i < size; ++i){
		ptr[i] = vec[i];
	}
}


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

float calc_angle( p3D x, p3D y ){
	float den = x.first * y.first + x.second.first * y.second.first + x.second.second * y.second.second,
		  num = sqrtf( powf(x.first,2) + powf(x.second.first,2) + powf(x.second.second,2) ) + sqrtf( powf(y.first,2) + powf(y.second.first,2) + powf(y.second.second,2) );

	return acos( den / num );
}

void test_calc_angle(){
	p3D x ( 2, p2D( 3, 0 )), y( 2, p2D( 0, 3 ));
    float ang = calc_angle( x, y );
    cout << setprecision(4) << ang << "\t" << ang*180/PI << endl;
}

float calc_norm( p3D x, p3D y){
	return sqrtf( powf(x.first-y.first,2) + powf( x.second.first-y.second.first,2) + powf( x.second.second-y.second.second,2) );
}

template <typename T>
class SchoolFish{
	vector< fish< T > > schoolfish;
	float lim1, lim2;

	public:
		SchoolFish(int size){
		schoolfish.resize(size);
		// float w_a, w_o;
	}

		void init( float wa, float wo ){
			default_random_engine rng(random_device{}()); 		
			// uniform_real_distribution<float> dist( -10, 10); 

			lim1 = 50, lim2 = 100;
			uniform_real_distribution<float> dist( lim1, lim2 );

			for( int i = 0; i < schoolfish.size(); ++i){
				schoolfish[i].num = i;
				// schoolfish[i].c = pair<float,float> ( dist(rng),dist(rng) );
				// schoolfish[i].v = pair<float,float> (2,2);
				// schoolfish[i].v = pair<float,float> (dist(rng),dist(rng) );

				schoolfish[i].c = pair< float, pair<float,float> >( dist(rng), pair<float,float> ( dist(rng),dist(rng) ));
				schoolfish[i].v = pair< float, pair<float,float> >( dist(rng), pair<float,float> ( dist(rng),dist(rng) ));

				schoolfish[i].s = 0.1;
				schoolfish[i].theta = PI/6;
				schoolfish[i].r_r = 1.0;
				schoolfish[i].r_p = 1.5;
				schoolfish[i].w_a = wa;
				schoolfish[i].w_o = wo;
			}
		}

		void print(){
			int dec=4;
			int pos=dec*2;
			cout << setw(pos) << "# Fish" << setw(pos*3) << "C" << setw(pos*3) << "V" << setw(pos) << "S" << setw(pos) << "Theta" << setw(pos) << "R-R" << setw(pos)
				 << setw(pos) << "R-P" << setw(pos) << "W_A" << setw(pos) << "W_O" << endl;

			for( int i = 0; i < schoolfish.size(); ++i){
				cout << setprecision(dec) << setw(pos) << schoolfish[i].num << setw(pos) << schoolfish[i].c.first << setw(pos) << schoolfish[i].c.second.first << setw(pos) << schoolfish[i].c.second.second << setw(pos) << schoolfish[i].v.first 
					 << setw(pos) << schoolfish[i].v.second.first << setw(pos) << schoolfish[i].v.second.second << setw(pos) << schoolfish[i].s << setw(pos) << schoolfish[i].theta 
					 << setw(pos) << schoolfish[i].r_r << setw(pos) << schoolfish[i].r_p << setw(pos) << schoolfish[i].w_a 
					 << setw(pos) << schoolfish[i].w_o << endl;
				
				string blank = "      ";
				// printf( "%s%.4f%s%.4f%s%.4f%s%.4f%s%.4f%s%.4f%s%.4f%s%.4f%s%.4f%s%.4f%s", blank.c_str(), schoolfish[i].c.first, blank.c_str(), schoolfish[i].c.second, 
				// 		blank.c_str(), schoolfish[i].v.first, blank.c_str(), schoolfish[i].v.second, blank.c_str(), schoolfish[i].s, blank.c_str(), schoolfish[i].theta,
				// 		blank.c_str(), schoolfish[i].r_r, blank.c_str(), schoolfish[i].r_p, blank.c_str(), schoolfish[i].w_a, blank.c_str(), schoolfish[i].w_o, "\n" );
			}	
		}

		void print2file( ostream& file, int type ){
			int dec=4;
			int pos=dec*2;
			
			//file << setw(pos) << "# Fish" << setw(pos*2) << "C" << endl;

			for( int i = 0; i < schoolfish.size(); ++i){
				if( type == 1){
					file << setprecision(dec) << setw(pos) << schoolfish[i].num << setw(pos) << schoolfish[i].c.first << setw(pos) << schoolfish[i].c.second.first << setw(pos) << schoolfish[i].c.second.second << endl;
				}
				else if( type == 2){
					file << setprecision(dec) << schoolfish[i].c.first << "," <<  schoolfish[i].c.second.first << "," << schoolfish[i].c.second.second<< endl;	
				}
			}
		}

		vector<float> v_norm(int pos){
			
			int N = 10000, M = 1000;
			// cout << "Fish" << pos << endl;
			
			vector<float> ans;
			// ans[0] = 2.0;

			float* ansptr;
			ansptr = (float *)malloc( schoolfish.size() * sizeof(float) );
			
			//Copy info from schoolfish to schoolptr
			fish<T>* schoolptr;
			vector2ptr( schoolfish, schoolptr, schoolfish.size() );

			// cout << "Original \n";
			// this->print();

			// cout << "Copy of schoolfish\n";

			// int pos_ = 8;
			// int dec = 4;
			// for( int i = 0; i < schoolfish.size(); ++i){
			// 	cout << setprecision(dec) << setw(pos_) << schoolptr[i].num << setw(pos_) << schoolptr[i].c.first << setw(pos_) << schoolptr[i].c.second.first << setw(pos_) << schoolptr[i].c.second.second << setw(pos_) << schoolptr[i].v.first 
			// 		 << setw(pos_) << schoolptr[i].v.second.first << setw(pos_) << schoolptr[i].v.second.second << setw(pos_) << schoolptr[i].s << setw(pos_) << schoolptr[i].theta 
			// 		 << setw(pos_) << schoolptr[i].r_r << setw(pos_) << schoolptr[i].r_p << setw(pos_) << schoolptr[i].w_a 
			// 		 << setw(pos_) << schoolptr[i].w_o << endl;
				
			// 	string blank = "      ";
			// 	// printf( "%s%.4f%s%.4f%s%.4f%s%.4f%s%.4f%s%.4f%s%.4f%s%.4f%s%.4f%s%.4f%s", blank.c_str(), schoolfish[i].c.first, blank.c_str(), schoolfish[i].c.second, 
			// 	// 		blank.c_str(), schoolfish[i].v.first, blank.c_str(), schoolfish[i].v.second, blank.c_str(), schoolfish[i].s, blank.c_str(), schoolfish[i].theta,
			// 	// 		blank.c_str(), schoolfish[i].r_r, blank.c_str(), schoolfish[i].r_p, blank.c_str(), schoolfish[i].w_a, blank.c_str(), schoolfish[i].w_o, "\n" );
			// }
			// cout << endl;	



			//Copy for devices
			fish<T>* d_schoolptr;
			float* d_ansptr;

			// // // Allocate space for device copies of schoolptr
			cudaMalloc((void **)&d_schoolptr, schoolfish.size() * sizeof(T) );
			cudaMalloc((void **)&d_ansptr, schoolfish.size()*sizeof(float) );
			
			// // Copy inputs to device
			cudaMemcpy(d_schoolptr, schoolptr, schoolfish.size() * sizeof(T), cudaMemcpyHostToDevice);  // Args: Dir. destino, Dir. origen, tamano de dato, sentido del envio

			// // Launch add() kernel on GPU
			// cout << "Start Cuda Call\n";
			cuda_norm<<<(N+M-1)/M,M>>> ( d_schoolptr, pos, d_ansptr );
			// cout << "End Cuda Call\n";

			// // Copy result back to host
			cudaMemcpy(ansptr, d_ansptr, schoolfish.size()*sizeof(float), cudaMemcpyDeviceToHost);

			ptr2vector( ansptr, ans, schoolfish.size() );

			// printelem( ans );

			// Cleanup
			free(schoolptr);
			free(ansptr);
			cudaFree(d_ansptr);
			cudaFree(d_schoolptr);

			return ans;
		}

		void print_distances(){
			for( int i = 0; i < schoolfish.size(); ++i){
				printelem( v_norm(i) );
			}
		}

		void calc_neighboors( int k ){
			for( int i = 0; i < schoolfish.size(); ++i){
				vector<float> v_dist = ( v_norm(i) );

				for( int j = 0; j < v_dist.size(); ++j){
					int act_size = schoolfish[i].neighborhood_r.size()+ schoolfish[i].neighborhood_p.size();
					if( act_size < k ){
						if( v_dist[j] <= schoolfish[i].r_r ){
							if( schoolfish[i].num != j ){
								schoolfish[i].neighborhood_r.push_back( j );
								// cout << j << "\t";
							}
						}
						else if( v_dist[j] > schoolfish[i].r_r && v_dist[j] <= schoolfish[i].r_p ){
							if( schoolfish[i].num != j ){
								schoolfish[i].neighborhood_p.push_back( j );
								// cout << j << "\t";
							}
						}
					}
					else if( act_size == k ){
						break;
					}
				}
				// cout << endl;
			}
			cout << "Good Calc neighbors" << endl;
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
				// pair<float,float> d(0,0);
				p3D d ( 0, p2D( 0, 0 ));

				if( schoolfish[i].neighborhood_r.size() > 0 ){
					for( int j = 0; j < schoolfish[i].neighborhood_r.size(); ++j){
						int ii = schoolfish[i].num, jj = schoolfish[i].neighborhood_r[j];
						float den = calc_norm( schoolfish[ii].c, schoolfish[jj].c );
						d.first +=  ( schoolfish[jj].c.first - schoolfish[ii].c.first ) / ( den );
						d.second.first +=  ( schoolfish[jj].c.second.first - schoolfish[ii].c.second.first ) / ( den );	
						d.second.second += ( schoolfish[jj].c.second.second - schoolfish[ii].c.second.second ) / ( den );	
					}
					d.first = -1 * d.first; d.second.first = -1 * d.second.first; d.second.second = -1 * d.second.second;
				}
				else{
					for( int j = 0; j < schoolfish[i].neighborhood_r.size(); ++j){
						int ii = schoolfish[i].num, jj = schoolfish[i].neighborhood_r[j];
						float den = calc_norm( schoolfish[ii].c, schoolfish[jj].c );

						int norm_v = (schoolfish[i].v.first / sqrtf( powf(schoolfish[i].v.first,2) + powf(schoolfish[i].v.second.first,2) + powf(schoolfish[i].v.second.second,2) ) );

						d.first +=  schoolfish[0].w_a * ( schoolfish[jj].c.first - schoolfish[ii].c.first ) / ( den ) + schoolfish[0].w_o * norm_v;
						d.second.first +=  schoolfish[0].w_a * ( schoolfish[jj].c.second.first - schoolfish[ii].c.second.first ) / ( den ) + schoolfish[0].w_o * norm_v;	
						d.second.second +=  schoolfish[0].w_a * ( schoolfish[jj].c.second.second - schoolfish[ii].c.second.second ) / ( den ) + schoolfish[0].w_o * norm_v;	
					}
					d.first = -1 * d.first; d.second.first = -1 * d.second.first; d.second.second = -1 * d.second.second;

				}

				

				if( calc_angle( schoolfish[i].c, d ) <= schoolfish[i].theta ){
					 schoolfish[i].c.first = schoolfish[i].c.first + schoolfish[i].s * d.first;
					 schoolfish[i].c.second.first = schoolfish[i].c.second.first + schoolfish[i].s * d.first;
					 schoolfish[i].c.second.second = schoolfish[i].c.second.second + schoolfish[i].s * d.first;
				}
			}
		}

		void movement( float t ){
			for( int i = 0; i < schoolfish.size(); ++i){
				// float cx = schoolfish[i].c.first, cy = schoolfish[i].c.second;

				check_limits( schoolfish[i].c, schoolfish[i].v ) ;

				schoolfish[i].c.first  += schoolfish[i].v.first *schoolfish[i].s * t;
				schoolfish[i].c.second.first += schoolfish[i].v.second.first *schoolfish[i].s * t;
				schoolfish[i].c.second.second += schoolfish[i].v.second.second *schoolfish[i].s * t;
			}
			cout << "Good Movement" << endl;
		}

		void check_limits( p3D& p, p3D& v ){

			int lmin = -100, lmax = 100;

			if( p.first + v.first < lmin ){
				v.first = v.first * -1;
			}
			else if( p.first + v.first > lmax ){
				v.first = v.first * -1;
			}
			else if( p.second.first + v.second.first < lmin ){
				v.second.first = v.second.first * -1;
			}
			else if( p.second.first + v.second.first > lmax ){
				v.second.first = v.second.first * -1;
			}
			else if( p.second.second + v.second.second < lmin ){
				v.second.second = v.second.second * -1;
			}
			else if( p.second.second + v.second.second > lmax ){
				v.second.second = v.second.second * -1;
			}
		}

};

template <typename T>
inline void str2num(string str, T& num){
	if ( ! (istringstream(str) >> num) ) num = 0;
}

void test_vector2ptr(){
	int* a;
	vector<int> b;
	b.push_back(-10);
	vector2ptr(b,a,b.size()); 

	cout << a[0] << endl;
}

void test_ptr2vector(){
	int* ptr = ( int* )malloc( 1 * sizeof(int) );
	vector<int> a;
	ptr[0] = 100;

	ptr2vector(ptr,a,1);
	cout << a[0] << endl;
}

int main( int argc, char** argv ){

	time_t timer = time(0); 

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

    ofstream result("movement_cuda.data");

	for( int i = 0; i < iter; ++i){
		cout << "Iter: " << i << endl;
		// myschool.print_distances();
		myschool.movement(t);
		myschool.calc_neighboors(k);
		// myschool.print_neighboors();
		myschool.update_c();
		// myschool.print();
		myschool.print2file( result, 2);
	}

	result.close();	

	time_t timer2 = time(0); 
    cout <<"\nTiempo total: " << difftime(timer2, timer) << endl;

	return 0;
}