//CPU includes
#include <iostream>
#include <algorithm>
#include <random>
#include <iomanip>
#include <string>
#include <sstream>
#include <fstream>
#include <thread>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#define WINDOW_TITLE_PREFIX "Chapter 2"

#include <unistd.h>   //_getch*/
#include <termios.h>  //_getch*/

typedef struct
{
  float XYZW[4];
  float RGBA[4];
} Vertex;

Vertex data[5000];

int num_fish, k, type;
float t = 1, wa, wo;

int
  CurrentWidth = 800,
  CurrentHeight = 600,
  WindowHandle = 0;

unsigned FrameCount = 0;

GLuint
  VertexShaderId,
  FragmentShaderId,
  ProgramId,
  VaoId,
  VboId;

const GLchar* VertexShader =
{
  "#version 400\n"\

  "layout(location=0) in vec4 in_Position;\n"\
  "layout(location=1) in vec4 in_Color;\n"\
  "out vec4 ex_Color;\n"\

  "void main(void)\n"\
  "{\n"\
  "  gl_Position = in_Position;\n"\
  "  ex_Color = in_Color;\n"\
  "}\n"
};

const GLchar* FragmentShader =
{
  "#version 400\n"\

  "in vec4 ex_Color;\n"\
  "out vec4 out_Color;\n"\

  "void main(void)\n"\
  "{\n"\
  "  out_Color = ex_Color;\n"\
  "}\n"
};

void Initialize(int, char*[]);
void InitWindow(int, char*[]);
void ResizeFunction(int, int);
void RenderFunction(void);
void TimerFunction(int);
void IdleFunction(void);
void Cleanup(void);
void CreateVBO(void);
void DestroyVBO(void);
void CreateShaders(void);
void DestroyShaders(void);

void processNormalKeys(unsigned char key, int x, int y);

//./fish 20000 10 200 15 5 

#define PI 3.1416

using namespace std;

template <typename T>
struct point2D{
	T x;
	T y;

	__host__ __device__
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
struct fish{
	int num;
	T c;	//position
	T v;	//direction
	double s;	//speed
	double theta;	//vision's angle
	double r_r;		//repulsion radius
	double r_p;		//reach radius
	// double w_a;		//attraction weight
	// double w_o;		//orientation weight

	vector<int> neighborhood_r;	//neighbors in repulsion zone
	vector<int> neighborhood_p; //neighbors in annulus r_r & r_p
};

__global__ void cuda_norm( p3D* schoolfish, int pos, double* c){

	int index = threadIdx.x + blockIdx.x*blockDim.x;
	// printf( "%i\n", index );

	double t1 = powf(schoolfish[pos].x - schoolfish[index].x,2);
	double t2 = powf(schoolfish[pos].y - schoolfish[index].y,2);
	double t3 = powf(schoolfish[pos].z - schoolfish[index].z,2);
	double dist = sqrtf( t1 + t2 + t3 );

	// printf( "%i\n%.4f\t%.4f\t%.4f\n%.4f\t%.4f\t%.4f\n%.4f\t%.4f\t%.4f\t%.4f\n\n", index, schoolfish[pos].c.x, schoolfish[pos].c.y, schoolfish[pos].c.z, 
	// schoolfish[index].c.x, schoolfish[index].c.y, schoolfish[index].c.z, t1, t2, t3, dist);

	c[index] = dist ;
	// printf( "%d\n", c[index] );
}

template <typename Type>
void ptr2vector( Type*& ptr, vector< Type >& vec, int size){
	for( int i = 0; i < size; i++){
		// cout << "Start Ptr2Vec\n";
		vec.push_back( ptr[i] );
		// cout << "End Ptr2Vec\n";
	}
}

template <typename Type>
void vector2ptr( vector< Type >& vec, Type*& ptr, int size){
	// cout << "Start Vec2Ptr\n";
	for( int i = 0; i < size; i++){
		ptr[i] = vec[i];
	}
	// cout << "End Vec2Ptr\n";
}

// void ptrd2vectord( double*& ptr, vector< double >& vec, int size){
// 	cout << "Start PtrD2VecD\n";
// 	for( int i = 0; i < size; i++){
// 		cout << i << "\t" ;
// 		vec.push_back( ptr[i] );
// 	}
// 	cout << "End PtrD2VecD\n";
// }

// // template <typename Type>
// void vectord2ptrd( vector<double>& vec, double*& ptr, int size){
// 	cout << "Start VecD2PtrD\n";
// 	for( int i = 0; i < size; i++){
// 		cout << i << "\t" ;
// 		ptr[i] = vec[i];
// 	}
// 	cout << "End VecD2PtrD\n";
// }

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

double calc_norm( p3D x){
	return sqrtf( powf(x.x,2) + powf( x.y,2) + powf( x.z,2) );
}

double calc_norm( p3D x, p3D y){
	return sqrtf( powf(x.x-y.x,2) + powf( x.y-y.y,2) + powf( x.z-y.z,2) );
}

// params = { size, r_r, r_p, k}
__global__ void calc_n(p3D* schoolfish, double* params, int* p_r, int* p_p){
	int index = threadIdx.x + blockIdx.x*blockDim.x;

	double* norm;
	norm = (double*)malloc( params[0] * sizeof(double) );

	for( int i = 0; i < params[0]; ++i){
		norm[i] = sqrtf(powf(schoolfish[i].x - schoolfish[index].x,2) + powf(schoolfish[i].y - schoolfish[index].y,2)
				  + powf(schoolfish[i].z - schoolfish[index].z,2));
	}

	norm[index] = 10000;

	int x = 0, y = 0, act_size = 0;

	for( int i = 0; i < params[0] && act_size < params[3] ; ++i){
		if( norm[i] <= params[1] ){
			p_r[x+index] = i;
			x++; act_size++;
		}
		else if( norm[i] > params[1] && norm[i] <= params[2] ){
			p_p[y+index] = i;
			y++; act_size++;
		}
	}
}



template <typename T>
class SchoolFish{

	public:
	vector< fish< T > > schoolfish;	
	double lim1, lim2;
	double wa, wo;

	int start, end;
	// int start, int end

	vector<p3D> v_p3D; 

	public:
		SchoolFish(int size){
		schoolfish.resize(size);
		v_p3D.resize(size);
		// double w_a, w_o;
	}

		void parallel_init(int start, int end){

			default_random_engine rng(random_device{}()); 		
			uniform_real_distribution<double> dist( lim1, lim2 );

			default_random_engine rng2(random_device{}()); 		
			uniform_real_distribution<double> dist2( lim1/10, lim2/10 );

			// cout << "Thread: " << start << "\t" << end << endl;

			for( int i = start; i < end; ++i){
				schoolfish[i].num = i;  //name

				schoolfish[i].c.x = dist(rng);
				schoolfish[i].c.y = dist(rng);
				schoolfish[i].c.z = dist(rng);

				schoolfish[i].v.x = dist2(rng2);
				schoolfish[i].v.y = dist2(rng2);
				schoolfish[i].v.z = dist2(rng2);

				schoolfish[i].s = 0.02;
				schoolfish[i].theta = PI/4;
				schoolfish[i].r_r = 0.1;
				schoolfish[i].r_p = 0.5;
				// schoolfish[i].w_a = wa;
				// schoolfish[i].w_o = wo;

				v_p3D[i] = schoolfish[i].c;
			}
		}

		void init( double _wa, double _wo ){

			//data = (Vertex *) malloc( schoolfish.size() * sizeof(Vertex) );

			wa = _wa; wo = _wo;

			lim1 = -0.7, lim2 = 0.7;

			//Parallel Process
		    int nThreads = thread::hardware_concurrency();  // # threads
		    vector<thread> ths(nThreads);   // threads vector

		    //dimensions of blocks
		    double incx = double(schoolfish.size()) / nThreads;

		    // int start, end;
		    //Launching threads
		    for ( int i = 0; i < nThreads; ++i){
		    	start = floor(i * incx), end = floor((i+1) * incx );
		    	// cout << start << "\t" << end << endl;
		    	//bind(&SchoolFish<T>::parallel_init, this, start, end) );
		        ths[i] = thread( &SchoolFish<T>::parallel_init, this, start, end );
		    }

		    /*lim2 = -100, lim1 = -50;

		    for ( int i = nThreads/3; i < nThreads*2/3; ++i){
		    	start = floor(i * incx), end = floor((i+1) * incx );
		    	// cout << start << "\t" << end << endl;
		    	//bind(&SchoolFish<T>::parallel_init, this, start, end) );
		        ths[i] = thread( &SchoolFish<T>::parallel_init, this, start, end );
		    }

		    lim2 = 30, lim1 = 50;

		    for ( int i = nThreads*2/3; i < nThreads; ++i){
		    	start = floor(i * incx), end = floor((i+1) * incx );
		    	// cout << start << "\t" << end << endl;
		    	//bind(&SchoolFish<T>::parallel_init, this, start, end) );
		        ths[i] = thread( &SchoolFish<T>::parallel_init, this, start, end );
		    }
		    */
		    //Joining threads
		    for ( int i = 0; i < nThreads; i++ )
		        ths[i].join();

		    //Predators

		    int n_p = 10;
		    for( int p = 0; p < n_p; ++p){
		    	schoolfish[p].c.x = 0.5;
			    schoolfish[p].c.y = 0.5;
			    schoolfish[p].c.z = 0.5;
			    schoolfish[p].r_r = 0.6;
			    schoolfish[p].r_p = 0.6;
			    schoolfish[p].theta = PI/2;
		    }

		    schoolfish[schoolfish.size()-2].c.x = 0.3;
		    schoolfish[schoolfish.size()-2].c.y = 0.3;
		    schoolfish[schoolfish.size()-2].c.z = 0.3;

		    schoolfish[schoolfish.size()-2].v.x = 0;
		    schoolfish[schoolfish.size()-2].v.y = 0;
		    schoolfish[schoolfish.size()-2].v.z = 0;

		    schoolfish[schoolfish.size()-2].r_r = 0.6;
			schoolfish[schoolfish.size()-2].r_p = 0.6;


		    schoolfish[schoolfish.size()-1].c.x = 0;
		    schoolfish[schoolfish.size()-1].c.y = 0;
		    schoolfish[schoolfish.size()-1].c.z = 0;

		    schoolfish[schoolfish.size()-1].v.x = 0;
		    schoolfish[schoolfish.size()-1].v.y = 0;
		    schoolfish[schoolfish.size()-1].v.z = 0;

		    schoolfish[schoolfish.size()-1].r_r = 0.6;
			schoolfish[schoolfish.size()-1].r_p = 0.6;

		}

		void print(){
			int dec=5;
			int pos=dec*2;
			cout << setw(pos) << "# Fish" << setw(pos*3) << "C" << setw(pos*3) << "V" << setw(pos) << "S" << setw(pos) << "Theta" << setw(pos/2) << "R-R" << setw(pos/2)
				 << setw(pos/2) << "R-P" << setw(pos/2) << "W_A" << setw(pos/2) << "W_O" << endl;

			for( int i = 0; i < schoolfish.size(); ++i){
				cout << setprecision(dec) << setw(pos) << schoolfish[i].num << setw(pos) << schoolfish[i].c.x << setw(pos) << schoolfish[i].c.y << setw(pos) << schoolfish[i].c.z << setw(pos) << schoolfish[i].v.x
					 << setw(pos) << schoolfish[i].v.y << setw(pos) << schoolfish[i].v.z << setw(pos) << schoolfish[i].s << setw(pos) << schoolfish[i].theta 
					 << setw(pos/2) << schoolfish[i].r_r << setw(pos/2) << schoolfish[i].r_p << setw(pos/2) << wa << setw(pos/2) << wo << endl;
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
					file << "{ {" << setprecision(dec) << schoolfish[i].c.x << "," <<  schoolfish[i].c.y << "," << schoolfish[i].c.z << ", 1.0f }, {" << "1.0f, 1.0f, 0.0f, 1.0f } },"<< endl; //<< "," << 1 << endl;	
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


		void calc_neighboors( int type, int k ){

			if(type == 1){
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
			else if(type == 2){

				cout << "Start Calc N\n";
		
				int N = schoolfish.size(), M = 1024;
				// cout << "Fish" << pos << endl;
				
				int size_school = v_p3D.size() * sizeof( p3D );

				int* p_r,* p_p; 
				p_r = (int *)malloc( schoolfish.size() * k * sizeof(int) );
				p_p = (int *)malloc( schoolfish.size() * k * sizeof(int) );

				double* params;
				params = (double *)malloc( 4 * sizeof(double) );
				params[0] = N, params[1] = schoolfish[0].r_r, params[2] = schoolfish[0].r_p, params[3] = k;

				//Initialization
				for( int i = 0; i < schoolfish.size() * k; ++i){
					p_r[i] = p_p[i] = -1;
				}

				//Copy info from schoolfish to schoolptr
				p3D* schoolptr;
				schoolptr = ( p3D* )malloc( size_school );

				vector2ptr( v_p3D, schoolptr, schoolfish.size() );

				//Copy for devices
				p3D* d_schoolptr;
				int* d_p_r,* d_p_p;

				// Allocate space for device copies of schoolptr
				cudaMalloc((void **)&d_schoolptr, size_school );
				cudaMalloc((void **)&d_p_r, schoolfish.size() * k * sizeof(int) );
				cudaMalloc((void **)&d_p_p, schoolfish.size() * k * sizeof(int) );
				
				// Copy inputs to device
				cudaMemcpy(d_schoolptr, schoolptr, size_school, cudaMemcpyHostToDevice);  // Args: Dir. destino, Dir. origen, tamano de dato, sentido del envio
				cudaMemcpy(d_p_r, p_r, schoolfish.size() * k * sizeof(int), cudaMemcpyHostToDevice);
				cudaMemcpy(d_p_p, p_p, schoolfish.size() * k * sizeof(int), cudaMemcpyHostToDevice);

				// Launch add() kernel on GPU
				// cout << "Start Cuda Call\n";
				calc_n<<<(N+M-1)/M,M>>> ( d_schoolptr, params, d_p_r, d_p_p );
				// cout << "End Cuda Call\n";

				// Copy result back to host
				cudaMemcpy(p_r, d_p_r, schoolfish.size() * k * sizeof(int), cudaMemcpyDeviceToHost);
				cudaMemcpy(p_p, d_p_p, schoolfish.size() * k * sizeof(int), cudaMemcpyDeviceToHost);

				// cout << "I'm alive\n";

				int n = 0;
				for( int i = 0; i < schoolfish.size() * k; ++i){
					// vector<int> v_r(schoolfish.size() * k), v_p(schoolfish.size() * k);

					// ptr2vector( p_r, v_r, schoolfish.size() * k );
					// ptr2vector( p_p, v_p, schoolfish.size() * k );


					for( int j = 0; j < schoolfish.size() * k && p_r[j] != -1; ++j){
						schoolfish[n].neighborhood_r.push_back( p_r[j] );
					}

					for( int j = 0; j < schoolfish.size() * k && p_p[j] != -1; ++j){
						schoolfish[n].neighborhood_p.push_back( p_p[j] );
					}
					n++;
				}

				// Cleanup
				free(schoolptr);
				free(p_r);
				free(p_p);
				cudaFree(d_schoolptr);
				cudaFree(d_p_r);
				cudaFree(d_p_p);

				cout << "Finish CNorm\n";	
			}
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

		void parallel_update_c(int start, int end, int k){

			for( int i = start; i < end; ++i){
					// pair<double,double> d(0,0);
				p3D d1; d1.x = d1.y = d1.z = 0;
				p3D d2; d2.x = d2.y = d2.z = 0;

				if( schoolfish[i].neighborhood_r.size() > 0 ){
					
					// cout << schoolfish[i].neighborhood_r.size() << "neighbors in r_r" << endl;

					for( int j = 0; j < schoolfish[i].neighborhood_r.size(); ++j){
						
						int ii = schoolfish[i].num, jj = schoolfish[i].neighborhood_r[j];
						
						double den = calc_norm( schoolfish[ii].c, schoolfish[jj].c );
				
						d1.x +=  ( schoolfish[jj].c.x - schoolfish[ii].c.x ) / ( den );
						d1.y +=  ( schoolfish[jj].c.y - schoolfish[ii].c.y ) / ( den );	
						d1.z += ( schoolfish[jj].c.z - schoolfish[ii].c.z ) / ( den );	
				
					}
			
					d1.x = -1 * d1.x; d1.y = -1 * d1.y; d1.z = -1 * d1.z;

					double vecnorm1 = calc_norm( schoolfish[i].v );
					double vecnorm2 = calc_norm( d1 );

					p3D vn1; vn1.x = schoolfish[i].v.x / vecnorm1; vn1.y = schoolfish[i].v.y / vecnorm1; vn1.z = schoolfish[i].v.z / vecnorm1;
					p3D vn2; vn2.x = d1.x / vecnorm2; vn2.y = d1.y / vecnorm2; vn2.z = d1.z / vecnorm2;

					if( calc_angle( vn1, vn2 ) <= schoolfish[i].theta ){

						check_limits(schoolfish[i].c, vn2);

						schoolfish[i].c.x = schoolfish[i].c.x + schoolfish[i].s * vn2.x; //ojo
					 	schoolfish[i].c.y = schoolfish[i].c.y + schoolfish[i].s * vn2.y;
					 	schoolfish[i].c.z = schoolfish[i].c.z + schoolfish[i].s * vn2.z;

					}

					if( i != schoolfish.size()-1 && i != schoolfish.size()-2){

						// check_limits(schoolfish[i].c, vn2);

						schoolfish[i].v.x = vn2.x;
						schoolfish[i].v.y = vn2.y;
						schoolfish[i].v.z = vn2.z;

					}


				}

				else if( schoolfish[i].neighborhood_p.size() > 0){

					// cout << schoolfish[i].neighborhood_p.size() << "neighbors in r_p" << endl;

					for( int j = 0; j < schoolfish[i].neighborhood_p.size(); ++j){
						int ii = schoolfish[i].num, jj = schoolfish[i].neighborhood_p[j];
						double den = calc_norm( schoolfish[ii].c, schoolfish[jj].c );

						//int norm_v = ( schoolfish[i].v.x/ sqrtf( powf(schoolfish[i].v.x,2) + powf(schoolfish[i].v.y,2) + powf(schoolfish[i].v.z,2) ) );

						d2.x +=  wa * ( schoolfish[jj].c.x - schoolfish[ii].c.x ) / ( den ); //+ wo * norm_v;
						d2.y +=  wa * ( schoolfish[jj].c.y - schoolfish[ii].c.y ) / ( den ); //+ wo * norm_v;	
						d2.z +=  wa * ( schoolfish[jj].c.z - schoolfish[ii].c.z ) / ( den ); //+ wo * norm_v;	
					}
					// d.x = -1 * d.x; d.y = -1 * d.y; d.z = -1 * d.z;

					p3D v1; v1.x = v1.y = v1.z = 0;					

					for( int j = 0; j < schoolfish[i].neighborhood_p.size(); ++j){
						int jj = schoolfish[i].neighborhood_p[j];
						double norm_v = calc_norm( schoolfish[jj].v );

						v1.x += (schoolfish[jj].v.x / norm_v);
						v1.y += (schoolfish[jj].v.y / norm_v);
						v1.z += (schoolfish[jj].v.z / norm_v);
					}

					v1.x = wo * v1.x; v1.y = wo * v1.y; v1.z = wo * v1.z;

					d2.x = d2.x + v1.x; d2.y = d2.y + v1.y; d2.z = d2.z + v1.z;

					double vecnorm1 = calc_norm( schoolfish[i].v );
					double vecnorm2 = calc_norm( d2 );

					p3D vn1; vn1.x = schoolfish[i].v.x / vecnorm1; vn1.y = schoolfish[i].v.y / vecnorm1; vn1.z = schoolfish[i].v.z / vecnorm1;
					p3D vn2; vn2.x = d2.x / vecnorm2; vn2.y = d2.y / vecnorm2; vn2.z = d2.z / vecnorm2;

					if( calc_angle( vn1, vn2 ) <= schoolfish[i].theta ){

						check_limits(schoolfish[i].c, vn2);

						schoolfish[i].c.x = schoolfish[i].c.x + schoolfish[i].s * vn2.x;
					 	schoolfish[i].c.y = schoolfish[i].c.y + schoolfish[i].s * vn2.y;
					 	schoolfish[i].c.z = schoolfish[i].c.z + schoolfish[i].s * vn2.z;

					}

					schoolfish[i].v.x = vn2.x;
					schoolfish[i].v.y = vn2.y;
					schoolfish[i].v.z = vn2.z;

					if( i != schoolfish.size()-1 && i != schoolfish.size()-2){

						

						schoolfish[i].v.x = vn2.x;
						schoolfish[i].v.y = vn2.y;
						schoolfish[i].v.z = vn2.z;

					}

				}		

				v_p3D[i].x = schoolfish[i].c.x;
				v_p3D[i].y = schoolfish[i].c.y;
				v_p3D[i].z = schoolfish[i].c.z;

				
			

			}
		}


		void update_c(){

			//cout << "Updating C\n";

			//print_neighboors();

			int nThreads = thread::hardware_concurrency();  // # threads
			    vector<thread> ths(nThreads);   // threads vector

			    //dimensions of blocks
			    double incx = double(schoolfish.size()) / nThreads;

			    int start, end;
			    
			    //Launching threads
			    for ( int i = 0; i < nThreads; ++i){
			    	start = floor(i * incx), end = floor((i+1) * incx );
			    	// cout << start << "\t" << end << endl;

			        ths[i] = thread( &SchoolFish<T>::parallel_update_c, this, start, end, k );
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

		void movement( double t ){
			for( int i = 0; i < schoolfish.size(); ++i){
				// double cx = schoolfish[i].c.x, cy = schoolfish[i].c.second;

				check_limits( schoolfish[i].c, schoolfish[i].v ) ;

				schoolfish[i].c.x += schoolfish[i].v.x * schoolfish[i].s * t;
				schoolfish[i].c.y += schoolfish[i].v.y * schoolfish[i].s * t;
				schoolfish[i].c.z += schoolfish[i].v.z * schoolfish[i].s * t;

				v_p3D[i].x = schoolfish[i].c.x;
				v_p3D[i].y = schoolfish[i].c.y;
				v_p3D[i].z = schoolfish[i].c.z;
			}
			// cout << "Good Movement" << endl;

			// schoolfish[schoolfish.size()-1].c.x = 0;
		 //    schoolfish[schoolfish.size()-1].c.y = 0;
		 //    schoolfish[schoolfish.size()-1].c.z = 0;
		}

		void check_limits( p3D& p, p3D& v ){

			double lmin = -1, lmax = 1;
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

		void passtoptr(){

			for( int i = 0; i < schoolfish.size(); ++i){
				data[i].XYZW[0] = (GLfloat)schoolfish[i].c.x;
				data[i].XYZW[1] = (GLfloat)schoolfish[i].c.y;
				data[i].XYZW[2] = (GLfloat)schoolfish[i].c.z;
				data[i].XYZW[3] = (GLfloat)1.0f;

				data[i].RGBA[0] = (GLfloat)1.0f;
				data[i].RGBA[1] = (GLfloat)0.0f;
				data[i].RGBA[2] = (GLfloat)0.0f;
				data[i].RGBA[3] = (GLfloat)1.0f;
			}


		    int n_p = 10;

		    for( int p = 0; p < n_p; ++p){
		    	data[p].RGBA[0] = (GLfloat)0.0f;
				data[p].RGBA[1] = (GLfloat)1.0f;
				data[p].RGBA[2] = (GLfloat)0.0f;
				data[p].RGBA[3] = (GLfloat)1.0f;
		    }

			data[schoolfish.size()-1].RGBA[0] = 0.0f;
		    data[schoolfish.size()-1].RGBA[1] = 1.0f;
		    data[schoolfish.size()-1].RGBA[2] = 1.0f;

		    data[schoolfish.size()-2].RGBA[0] = 0.0f;
		    data[schoolfish.size()-2].RGBA[1] = 1.0f;
		    data[schoolfish.size()-2].RGBA[2] = 1.0f;
		}
};

template <typename T>
inline void str2num(string str, T& num){
	if ( ! (istringstream(str) >> num) ) num = 0;
}

SchoolFish<p3D> globalschool( 1 );
//float globalt = 1; int globalk = 5;

void Initialize(int argc, char* argv[])
{
  SchoolFish<p3D> fishschool( num_fish );
  fishschool.init(wa,wo);

  globalschool = fishschool;

  //globalschool.print();

  GLenum GlewInitResult;

  InitWindow(argc, argv);
  
  glewExperimental = GL_TRUE;
  GlewInitResult = glewInit();

  if (GLEW_OK != GlewInitResult) {
    fprintf(
      stderr,
      "ERROR: %s\n",
      glewGetErrorString(GlewInitResult)
    );
    exit(EXIT_FAILURE);
  }
  
  fprintf(
    stdout,
    "INFO: OpenGL Version: %s\n",
    glGetString(GL_VERSION)
  );


}

void InitWindow(int argc, char* argv[])
{
  glutInit(&argc, argv);
  
  glutInitContextVersion(3, 0);
  glutInitContextFlags(GLUT_FORWARD_COMPATIBLE);
  glutInitContextProfile(GLUT_CORE_PROFILE);

  glutSetOption(
    GLUT_ACTION_ON_WINDOW_CLOSE,
    GLUT_ACTION_GLUTMAINLOOP_RETURNS
  );
  
  glutInitWindowSize(CurrentWidth, CurrentHeight);

  glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);

  WindowHandle = glutCreateWindow(WINDOW_TITLE_PREFIX);

  if(WindowHandle < 1) {
    fprintf(
      stderr,
      "ERROR: Could not create a new rendering window.\n"
    );
    exit(EXIT_FAILURE);
  }
  
  glutReshapeFunc(ResizeFunction);
  glutDisplayFunc(RenderFunction);
  glutIdleFunc(IdleFunction);


  glutKeyboardFunc(processNormalKeys);

  glutTimerFunc(0, TimerFunction, 0);

  // Cleanup();
  glutCloseFunc(Cleanup);
}

void ResizeFunction(int Width, int Height)
{
  CurrentWidth = Width;
  CurrentHeight = Height;
  glViewport(0, 0, CurrentWidth, CurrentHeight);

  glLoadIdentity();
}

void RenderFunction(void)
{
  ++FrameCount;

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glDrawArrays(GL_POINTS, 0, num_fish);

  glutSwapBuffers();
}

char getch(){
    /*#include <unistd.h>   //_getch*/
    /*#include <termios.h>  //_getch*/
    char buf=0;
    struct termios old={0};
    fflush(stdout);
    if(tcgetattr(0, &old)<0)
        perror("tcsetattr()");
    old.c_lflag&=~ICANON;
    old.c_lflag&=~ECHO;
    old.c_cc[VMIN]=1;
    old.c_cc[VTIME]=0;
    if(tcsetattr(0, TCSANOW, &old)<0)
        perror("tcsetattr ICANON");
    if(read(0,&buf,1)<0)
        perror("read()");
    old.c_lflag|=ICANON;
    old.c_lflag|=ECHO;
    if(tcsetattr(0, TCSADRAIN, &old)<0)
        perror ("tcsetattr ~ICANON");
    printf("%c\n",buf);
    return buf;
 }

void IdleFunction(void)
{
	// int c = getchar();
	// putchar(c);

	//getch();

  	CreateShaders();
	CreateVBO();

	glClearColor(0,0,0,0);
  	//glClearColor(0.071f, 0.741f, 0.843f, 0.0f);

	//cout << "--" << endl;
	
	// globalschool.print();
		
	globalschool.calc_neighboors( type, k);

	globalschool.movement(t);
		
	globalschool.update_c();

	globalschool.passtoptr();


  glutPostRedisplay();

  //Cleanup();

}

void TimerFunction(int Value)
{
  if (0 != Value) {
    char* TempString = (char*)
      malloc(512 + strlen(WINDOW_TITLE_PREFIX));

    sprintf(
      TempString,
      "%s: %d Frames Per Second @ %d x %d",
      WINDOW_TITLE_PREFIX,
      FrameCount * 20,
      CurrentWidth,
      CurrentHeight
    );

    glutSetWindowTitle(TempString);
    free(TempString);
  }
  
  FrameCount = 0;
  glutTimerFunc(50, TimerFunction, 1);
}

void Cleanup(void)
{
  DestroyShaders();
  DestroyVBO();
}

void CreateVBO(void)
{
  /*Vertex Vertices[] =
  {
    { { -0.8f, -0.8f, 0.0f, 1.0f }, { 1.0f, 0.0f, 0.0f, 1.0f } },
    { {  0.0f,  0.8f, 0.0f, 1.0f }, { 0.0f, 1.0f, 0.0f, 1.0f } },
    { {  0.8f, -0.8f, 0.0f, 1.0f }, { 0.0f, 0.0f, 1.0f, 1.0f } }
  };*/

  GLenum ErrorCheckValue = glGetError();
  const size_t BufferSize = sizeof(data);
  const size_t VertexSize = sizeof(data[0]);
  const size_t RgbOffset = sizeof(data[0].XYZW);
  
  glGenVertexArrays(1, &VaoId);
  glBindVertexArray(VaoId);

  glGenBuffers(1, &VboId);
  glBindBuffer(GL_ARRAY_BUFFER, VboId);
  glBufferData(GL_ARRAY_BUFFER, BufferSize, data, GL_STATIC_DRAW);

  glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, VertexSize, 0);
  glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, VertexSize, (GLvoid*)RgbOffset);

  glEnableVertexAttribArray(0);
  glEnableVertexAttribArray(1);

  ErrorCheckValue = glGetError();
  if (ErrorCheckValue != GL_NO_ERROR)
  {
    fprintf(
      stderr,
      "ERROR: Could not create a VBO: %s \n",
      gluErrorString(ErrorCheckValue)
    );

    exit(-1);
  }
}

void DestroyVBO(void)
{
  GLenum ErrorCheckValue = glGetError();

  glDisableVertexAttribArray(1);
  glDisableVertexAttribArray(0);
  
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glDeleteBuffers(1, &VboId);

  glBindVertexArray(0);
  glDeleteVertexArrays(1, &VaoId);

  ErrorCheckValue = glGetError();
  if (ErrorCheckValue != GL_NO_ERROR)
  {
    fprintf(
      stderr,
      "ERROR: Could not destroy the VBO: %s \n",
      gluErrorString(ErrorCheckValue)
    );

    exit(-1);
  }
}

void CreateShaders(void)
{
  GLenum ErrorCheckValue = glGetError();
  
  VertexShaderId = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(VertexShaderId, 1, &VertexShader, NULL);
  glCompileShader(VertexShaderId);

  FragmentShaderId = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(FragmentShaderId, 1, &FragmentShader, NULL);
  glCompileShader(FragmentShaderId);

  ProgramId = glCreateProgram();
    glAttachShader(ProgramId, VertexShaderId);
    glAttachShader(ProgramId, FragmentShaderId);
  glLinkProgram(ProgramId);
  glUseProgram(ProgramId);

  ErrorCheckValue = glGetError();
  if (ErrorCheckValue != GL_NO_ERROR)
  {
    fprintf(
      stderr,
      "ERROR: Could not create the shaders: %s \n",
      gluErrorString(ErrorCheckValue)
    );

    exit(-1);
  }
}

void DestroyShaders(void)
{
  GLenum ErrorCheckValue = glGetError();

  glUseProgram(0);

  glDetachShader(ProgramId, VertexShaderId);
  glDetachShader(ProgramId, FragmentShaderId);

  glDeleteShader(FragmentShaderId);
  glDeleteShader(VertexShaderId);

  glDeleteProgram(ProgramId);

  ErrorCheckValue = glGetError();
  if (ErrorCheckValue != GL_NO_ERROR)
  {
    fprintf(
      stderr,
      "ERROR: Could not destroy the shaders: %s \n",
      gluErrorString(ErrorCheckValue)
    );

    exit(-1);
  }
}

void processNormalKeys(unsigned char key, int x, int y) {

    switch(key) {
	    case 27: // escape
	        exit(0);
	        break;
   		}
}

int main( int argc, char** argv ){

	vector<time_t> timer(10);
	timer[0] = time(0); 

	//Params of SchoolFish
	// int num_fish, k, iter;
	string par = argv[1]; str2num( par, num_fish); //globalsize = num_fish;
	par = argv[2]; str2num( par, k); //globalk = k;

	// float t = 1;float wa, wo;
	par = argv[3]; str2num( par, type); //globalt = t;

	// float wa, wo;
	par = argv[4]; str2num( par, wa);
	par = argv[5]; str2num( par, wo);

    Initialize(argc, argv);

	glutMainLoop();
  
  	exit(EXIT_SUCCESS);

	return 0;
}