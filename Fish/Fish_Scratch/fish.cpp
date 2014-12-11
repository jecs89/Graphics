#include <iostream>
#include <algorithm>
#include <random>
#include <iomanip>
#include <string>
#include <sstream>
#include <fstream>


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

float calc_angle( p2D x, p2D y ){
	float den = x.first * y.first + x.second * y.second ,
		  num = sqrtf( powf(x.first,2) + powf(x.second,2) ) + sqrtf( powf(y.first,2) + powf(y.second,2) );

	return acos( den / num );
}

void test_calc_angle(){
	p2D x ( 3, 0 ), y( 0, 3 );
    float ang = calc_angle( x, y );
    cout << setprecision(4) << ang << "\t" << ang*180/PI << endl;
}

float calc_norm( p2D x, p2D y){
	return sqrtf( powf(x.first-y.first,2) + powf( x.second-y.second,2) );
}

template <typename T>
class SchoolFish{
	vector< fish< T > > schoolfish;
	float lim1, lim2;

	public:
		SchoolFish(int size){
		schoolfish.resize(size);
	}

		void init(){
			default_random_engine rng(random_device{}()); 		
			// uniform_real_distribution<float> dist( -10, 10); 

			lim1 = -10, lim2 = 10;
			uniform_real_distribution<float> dist( lim1, lim2 );

			for( int i = 0; i < schoolfish.size(); ++i){
				schoolfish[i].num = i;
				schoolfish[i].c = pair<float,float> ( dist(rng),dist(rng) );
				// schoolfish[i].v = pair<float,float> (2,2);
				schoolfish[i].v = pair<float,float> (dist(rng),dist(rng) );
				schoolfish[i].s = 0.1;
				schoolfish[i].theta = PI/6;
				schoolfish[i].r_r = 1.0;
				schoolfish[i].r_p = 1.5;
				schoolfish[i].w_a = 1;
				schoolfish[i].w_o = 5;
			}
		}

		void print(){
			int dec=4;
			int pos=dec*2;
			cout << setw(pos) << "# Fish" << setw(pos*2) << "C" << setw(pos*2) << "V" << setw(pos) << "S" << setw(pos) << "Theta" << setw(pos) << "R-R" << setw(pos)
				 << setw(pos) << "R-P" << setw(pos) << "W_A" << setw(pos) << "W_O" << endl;

			for( int i = 0; i < schoolfish.size(); ++i){
				cout << setprecision(dec) << setw(pos) << schoolfish[i].num << setw(pos) << schoolfish[i].c.first << setw(pos) << schoolfish[i].c.second << setw(pos) << schoolfish[i].v.first 
					 << setw(pos) << schoolfish[i].v.second << setw(pos) << schoolfish[i].s << setw(pos) << schoolfish[i].theta 
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
					file << setprecision(dec) << setw(pos) << schoolfish[i].num << setw(pos) << schoolfish[i].c.first << setw(pos) << schoolfish[i].c.second << endl;
				}
				else if( type == 2){
					file << setprecision(dec) << schoolfish[i].c.first << "," << schoolfish[i].c.second << endl;	
				}
			}
		}

		vector<float> v_norm(int pos){
			// cout << "Fish" << pos << endl;
			vector<float> ans;
			for( int i = 0; i < schoolfish.size(); ++i){
				if( i != pos ){
					float dist = sqrtf( powf(schoolfish[pos].c.first-schoolfish[i].c.first,2) + powf(schoolfish[pos].c.second-schoolfish[i].c.second,2) );
					ans.push_back( dist );
				}
				else{
					ans.push_back( 50 );		
				}
			}

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
				pair<float,float> d(0,0);

				if( schoolfish[i].neighborhood_r.size() > 0 ){
					for( int j = 0; j < schoolfish[i].neighborhood_r.size(); ++j){
						int ii = schoolfish[i].num, jj = schoolfish[i].neighborhood_r[j];
						float den = calc_norm( schoolfish[ii].c, schoolfish[jj].c );
						d.first +=  ( schoolfish[jj].c.first - schoolfish[ii].c.first ) / ( den );
						d.second +=  ( schoolfish[jj].c.second - schoolfish[ii].c.second ) / ( den );	
					}
					d.first = -1 * d.first; d.second = -1 * d.second;
				}
				else{
					for( int j = 0; j < schoolfish[i].neighborhood_r.size(); ++j){
						int ii = schoolfish[i].num, jj = schoolfish[i].neighborhood_r[j];
						float den = calc_norm( schoolfish[ii].c, schoolfish[jj].c );

						int norm_v = (schoolfish[i].v.first / sqrtf( powf(schoolfish[i].v.first,2) + powf(schoolfish[i].v.second,2) ) );

						d.first +=  ( schoolfish[jj].c.first - schoolfish[ii].c.first ) / ( den ) + norm_v;
						d.second +=  ( schoolfish[jj].c.second - schoolfish[ii].c.second ) / ( den ) + norm_v;	
					}
					d.first = -1 * d.first; d.second = -1 * d.second;

				}

				

				if( calc_angle( schoolfish[i].c, d ) <= schoolfish[i].theta ){
					 schoolfish[i].c.first = schoolfish[i].c.first + schoolfish[i].s * d.first;
					 schoolfish[i].c.second = schoolfish[i].c.second + schoolfish[i].s * d.first;
				}
			}
		}

		void movement( float t ){
			for( int i = 0; i < schoolfish.size(); ++i){
				float cx = schoolfish[i].c.first, cy = schoolfish[i].c.second;

				check_limits( schoolfish[i].c, schoolfish[i].v ) ;

				schoolfish[i].c.first  += schoolfish[i].v.first *schoolfish[i].s * t;
				schoolfish[i].c.second += schoolfish[i].v.second *schoolfish[i].s * t;
			}
		}

		void check_limits( p2D& p, p2D& v ){

			int lmin = -20, lmax = 20;

			if( p.first + v.first < lmin ){
				v.first = v.first * -1;
			}
			else if( p.second + v.second < lmin ){
				v.second = v.second * -1;
			}
			else if( p.first + v.first > lmax ){
				v.first = v.first * -1;
			}
			else if( p.second + v.second > lmax ){
				v.second = v.second * -1;
			}
		}

};

inline void str2int(string str, int& num){
	if ( ! (istringstream(str) >> num) ) num = 0;
}

int main( int argc, char** argv ){

	time_t timer = time(0); 

	//Params of SchoolFish
	int num_fish, k, iter;
	string par = argv[1]; str2int( par, num_fish);
	par = argv[2]; str2int( par, k);

	//Initialization of values
	SchoolFish<p2D> myschool( num_fish );
	myschool.init();
	myschool.print();

	float t = 0.5;
	
	par = argv[3]; str2int( par, iter);

    ofstream result("movement.data");

	for( int i = 0; i < iter; ++i){
		// cout << i << endl;
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