#include "mtx.hpp"
int main() {
  using namespace std;
  ios::sync_with_stdio(false);
  cin.tie(0);

  auto show_vec = [](auto &&v) {
    for (auto i : v)
      cout << i << " ";
    cout << endl;
  };

  Mtx m1({{1, 2, 3, 4}, {3, 4, 5, 6}, {5, 6, 7, 8}});
  Mtx m2({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}});
  Mtx m3({{0, 2, 0}, {4, 0, 6}, {0, 8, 0}, {10, 0, 12}});
  Mtx m4{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  Mtx v1{{1, 2, 3, 4, 5, 6}};
  Mtx v2{{1}, {2}, {3}, {4}, {5}, {6}};
// /*
  std::cout << "Matrix multiplies matrix:\n";
  (m1 * m2).display();
  std::cout << "Matrix multiplies constant:\n";
  (m1 * 2).display();
  std::cout << "Matrix adds matrix:\n";
  (m1 + m1).display();
  std::cout << "Matrix adds constant:\n";
  (m1 + 10).display();
  std::cout << "Matrix subtracts matrix:\n";
  (m2 - m3).display();
  std::cout << "Matrix subtracts constant:\n";
  (m2 - 5).display();
  std::cout << "Matrix hadamard product:\n";
  m1.h_product(m1).display();
  std::cout << "Vector hadamard product:\n";
  v1.h_product(v1).display();
  std::cout << "Vector dot product (Matrix multiplication):\n";
  (v1 * v2).display();
  std::cout << "Unwrap matrix (1 by 1):\n";
  cout << (v1 * v2).unwrap()<<endl;;
  std::cout << "Matrix transpose:\n";
  m1.transpose().display();
  cout << "Inspect element:\n";
  cout << m1.get(0, 0) << endl;
  cout << "Extract row:\n";
  show_vec(m1.row(0));
  cout << "Extract col:\n";
  show_vec(m1.col(0));
  cout << "Trace of matrix:\n";
  cout << m4.tr() << endl;
  cout << "l2 norm of matrix:\n";
  cout << m4.norm2() << endl;
  cout << "L2 of matrix:\n";
  cout << m4.L2() << endl;
  cout << endl;

  return 0;
}
