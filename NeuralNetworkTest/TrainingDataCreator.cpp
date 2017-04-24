//#include <iostream>
//#include <cmath>
//#include <cstdlib>
//#include <fstream>
//
//using namespace std;
//
//int main()
//{
//	ofstream out_data("trainingData.txt");
//	out_data << "topology: 2 4 1" << endl;
//	
//	for(int i = 200000; i >= 0; --i)
//	{
//		int n1 = (int)(2.0 * rand() / double(RAND_MAX));
//		int n2 = (int)(2.0 * rand() / double(RAND_MAX));
//		int t = n1 ^ n2;
//		out_data << "in: " << n1 << ".0 " << n2 << ".0 " << endl;
//		out_data << "out: " << t << ".0" << endl;
//	}
//}