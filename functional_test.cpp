#include "mtx_alpha.hpp"
int main() {
  using namespace std;
  M m1(3, 4, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  M m2(4, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  M v1(1, 3, {1, 2, 3});
  M v2(3, 1, {1, 2, 3});
  M m3(4, 4, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  M m4(4, 4, {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1});

  cout << "clone a matrix:\n";
  m3.clone().display();
  cout << "\nmatrices addition:\n";
  (m3 + m3).display();
  cout << "\nmatrix & scalar addition:\n";
  (m3 + 5).display();
  cout << "\nmatrices subtraction:\n";
  (m3 - m4).display();
  cout << "\nmatrix & scalar subtraction:\n";
  (m3 - 9).display();
  cout << "\nmatrices multiplication:\n";
  (m1 * m2).display();
  cout << "\nmatrix & scalar multiplication:\n";
  (m1 * 10).display();
  cout << "\nvector dot product (matrix):\n";
  (v1 * v2).display();  
  cout << "\nvector dot product (scalar):\n";
  cout << v1.dot(v2) << endl;
  cout << "\nhadamard product:\n";
  m3.h_p(m4).display();
  cout << "\ntranspose:\n";
  m3.t().display();
  cout << "\nextract row:\n";
  m3.row(2).display();
  cout << "\nextract column:\n";
  m3.col(1).display();
  cout << "\nextract element:\n";
  cout << m3.at(1, 3) << endl;
  cout << "\nL2 item:\n";
  cout << v1.L2() << endl;
  cout << "\nnorm-2:\n";
  cout << v1.norm2() << endl;
  cout << "\nunwrap matrix with single value:\n";
  cout << (v1 * v2).unwrap() << endl; 
  return 0;
}
