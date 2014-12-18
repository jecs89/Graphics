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

template <typename T>
struct point{
	T x;
	T y;
	T z;
};

typedef point<double> p3D;

template <typename T>
struct fish{
	int num;
	T c;	//position
	T v;	//direction
	double s;	//speed
	double theta;	//vision's angle
	double r_r;		//repulsion radius
	double r_p;		//reach radius
	double w_a;		//attraction weight
	double w_o;		//orientation weight

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

double calc_norm( p3D x, p3D y){
	return sqrtf( powf(x.x-y.x,2) + powf( x.y-y.y,2) + powf( x.z-y.z,2) );
}

template <typename T>
class SchoolFish{
	vector< fish< T > > schoolfish;
	double lim1, lim2;

	vector<p3D> v_p3D;

	public:
		SchoolFish(int size){
		schoolfish.resize(size);
		v_p3D.resize(size);
		// double w_a, w_o;
	}

		void init( double wa, double wo ){

			int scale = 10;

			default_random_engine rng(random_device{}()); 		
			// uniform_real_distribution<double> dist( -10, 10); 

			lim1 = 50, lim2 = 100;
			uniform_real_distribution<double> dist( lim1, lim2 );

			for( int i = 0; i < schoolfish.size(); ++i){
				schoolfish[i].num = i;
				// schoolfish[i].c = pair<double,double> ( dist(rng),dist(rng) );
				// schoolfish[i].v = pair<double,double> (2,2);
				// schoolfish[i].v = pair<double,double> (dist(rng),dist(rng) );

				schoolfish[i].c.x = 1+i/2.0;
				schoolfish[i].c.y = 1+i/2.0;
				schoolfish[i].c.z = 1+i/2.0 ;

				schoolfish[i].v.x = dist(rng)/scale;
				schoolfish[i].v.y = dist(rng)/scale;
				schoolfish[i].v.z = dist(rng)/scale;

				schoolfish[i].s = 0.1;
				schoolfish[i].theta = PI/6;
				schoolfish[i].r_r = 0.5;
				schoolfish[i].r_p = 1;
				schoolfish[i].w_a = wa;
				schoolfish[i].w_o = wo;

				v_p3D[i] = schoolfish[i].c;
			}
		}

		void print(){
			int dec=4;
			int pos=dec*2;
			cout << setw(pos) << "# Fish" << setw(pos*3) << "C" << setw(pos*3) << "V" << setw(pos) << "S" << setw(pos) << "Theta" << setw(pos) << "R-R" << setw(pos)
				 << setw(pos) << "R-P" << setw(pos) << "W_A" << setw(pos) << "W_O" << endl;

			for( int i = 0; i < schoolfish.size(); ++i){
				cout << setprecision(dec) << setw(pos) << schoolfish[i].num << setw(pos) << schoolfish[i].c.x << setw(pos) << schoolfish[i].c.y << setw(pos) << schoolfish[i].c.z << setw(pos) << schoolfish[i].v.x
					 << setw(pos) << schoolfish[i].v.y << setw(pos) << schoolfish[i].v.z << setw(pos) << schoolfish[i].s << setw(pos) << schoolfish[i].theta 
					 << setw(pos) << schoolfish[i].r_r << setw(pos) << schoolfish[i].r_p << setw(pos) << schoolfish[i].w_a 
					 << setw(pos) << schoolfish[i].w_o << endl;
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

		vector<double> v_norm(int pos){

			// cout << "Start CNorm\n";

			int N = 10000, M = 1024;
			// cout << "Fish" << pos << endl;
			
			vector<double> ans;

			// ans[0] = 2.0;

			int size_ans = schoolfish.size() * sizeof(double);
			int size_school = v_p3D.size() * sizeof( p3D );

			// printf("%s\t%i\t%s\t%i\t%s\t%i\n", "School", (int)schoolfish.size(), "Size ans", size_ans, "Size school", size_school);

			double* ansptr;
			ansptr = (double *)malloc( size_ans );

			//Copy info from schoolfish to schoolptr
			p3D* schoolptr;
			schoolptr = ( p3D* )malloc( size_school );

			vector2ptr( v_p3D, schoolptr, schoolfish.size() );

			// cout << "Step\n";

			// cout << "Original \n";
			// this->print();

			// cout << "Copy of schoolfish\n";

			// int pos_ = 8;
			// int dec = 4;
			// for( int i = 0; i < schoolfish.size(); ++i){
			// 	cout << setprecision(dec) << setw(pos_) << schoolptr[i].num << setw(pos_) << schoolptr[i].c.x << setw(pos_) << schoolptr[i].c.y << setw(pos_) << schoolptr[i].c.z << setw(pos_) << schoolptr[i].v.x
			// 		 << setw(pos_) << schoolptr[i].v.y << setw(pos_) << schoolptr[i].v.z << setw(pos_) << schoolptr[i].s << setw(pos_) << schoolptr[i].theta 
			// 		 << setw(pos_) << schoolptr[i].r_r << setw(pos_) << schoolptr[i].r_p << setw(pos_) << schoolptr[i].w_a 
			// 		 << setw(pos_) << schoolptr[i].w_o << endl;
				
			// 	string blank = "      ";
			// 	// printf( "%s%.4f%s%.4f%s%.4f%s%.4f%s%.4f%s%.4f%s%.4f%s%.4f%s%.4f%s%.4f%s", blank.c_str(), schoolfish[i].c.x, blank.c_str(), schoolfish[i].c.second, 
			// 	// 		blank.c_str(), schoolfish[i].v.first, blank.c_str(), schoolfish[i].v.second, blank.c_str(), schoolfish[i].s, blank.c_str(), schoolfish[i].theta,
			// 	// 		blank.c_str(), schoolfish[i].r_r, blank.c_str(), schoolfish[i].r_p, blank.c_str(), schoolfish[i].w_a, blank.c_str(), schoolfish[i].w_o, "\n" );
			// }
			// cout << endl;	

			//Copy for devices
			p3D* d_schoolptr;
			double* d_ansptr;

			// // // Allocate space for device copies of schoolptr
			cudaMalloc((void **)&d_schoolptr, size_school );
			cudaMalloc((void **)&d_ansptr, size_ans );
			
			// // Copy inputs to device
			cudaMemcpy(d_schoolptr, schoolptr, size_school, cudaMemcpyHostToDevice);  // Args: Dir. destino, Dir. origen, tamano de dato, sentido del envio

			// // Launch add() kernel on GPU
			// cout << "Start Cuda Call\n";
			cuda_norm<<<(N+M-1)/M,M>>> ( d_schoolptr, pos, d_ansptr );
			// cout << "End Cuda Call\n";

			// // Copy result back to host
			cudaMemcpy(ansptr, d_ansptr, size_ans, cudaMemcpyDeviceToHost);

			ptr2vector( ansptr, ans, schoolfish.size() );

			// printelem( ans );

			// Cleanup
			free(schoolptr);
			free(ansptr);
			cudaFree(d_ansptr);
			cudaFree(d_schoolptr);

			// cout << "Finish CNorm\n";

			return ans;
		}

		void print_distances(){
			for( int i = 0; i < schoolfish.size(); ++i){
				printelem( v_norm(i) );
			}
		}

		void calc_neighboors( int k ){
			// cout << "Star neighbors\n";
			for( int i = 0; i < schoolfish.size(); ++i){
				vector<double> v_dist = ( v_norm(i) );

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
			// cout << "Good Calc neighbors" << endl;
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

						d.x +=  schoolfish[0].w_a * ( schoolfish[jj].c.x - schoolfish[ii].c.x ) / ( den ) + schoolfish[0].w_o * norm_v;
						d.y +=  schoolfish[0].w_a * ( schoolfish[jj].c.y - schoolfish[ii].c.y ) / ( den ) + schoolfish[0].w_o * norm_v;	
						d.z +=  schoolfish[0].w_a * ( schoolfish[jj].c.z - schoolfish[ii].c.z ) / ( den ) + schoolfish[0].w_o * norm_v;	
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

// void test_vector2ptr(){
// 	int* a;
// 	vector<int> b;
// 	b.push_back(-10);
// 	vector2ptr(b,a,b.size()); 

// 	cout << a[0] << endl;
// }

// void test_ptr2vector(){
// 	int* ptr = ( int* )malloc( 1 * sizeof(int) );
// 	vector<int> a;
// 	ptr[0] = 100;

// 	ptr2vector(ptr,a,1);
// 	cout << a[0] << endl;
// }

int main( int argc, char** argv ){

	time_t timer = time(0); 

	//Params of SchoolFish
	int num_fish, k, iter;
	string par = argv[1]; str2num( par, num_fish);
	par = argv[2]; str2num( par, k);

	double t = 1;
	par = argv[3]; str2num( par, iter);

	double wa, wo;
	par = argv[4]; str2num( par, wa);
	par = argv[5]; str2num( par, wo);

	//Initialization of values
	// SchoolFish<p2D> myschool( num_fish );
	SchoolFish<p3D> myschool( num_fish );
	myschool.init(wa,wo);
	// myschool.print();

    ofstream result("movement_cuda.data");

	for( int i = 0; i < iter; ++i){
		// cout << "Iter: " << i << endl;
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