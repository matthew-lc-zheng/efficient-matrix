#ifndef MTX_HPP
#define MTX_HPP
#include <cmath>
#include <iostream>
#include <thread>
#include <vector>
#pragma GCC optimize(3)
#define max_thread std::thread::hardware_concurrency()

class Mtx {
  /* type of elements */
  using T = float;
  /* data */
  std::vector<std::vector<T>> mtx; // container
  /* property */
  int dim0, dim1, dim;
  ///
  /// \brief tools
  ///
  inline bool same_dim(const Mtx &m) const noexcept {
    return dim0 == m.dim0 && dim1 == m.dim1;
  }
  inline unsigned int LOG(unsigned int x) const noexcept {
    unsigned int ret;
    __asm__ __volatile__("bsrl %1, %%eax" : "=a"(ret) : "m"(x));
    return ret;
  }
  inline bool is_vec() const noexcept { return std::min(dim0, dim1) == 1; }
  inline bool vec_dim() const noexcept { return std::max(dim0, dim1); }
  inline bool is_square() const noexcept { return dim0 == dim1; }
  ///
  /// \brief processing through multiple threads
  /// \param dim: stop point
  /// \param foo: content to process
  /// \param dim_begin: default start point is zero
  ///
  inline void JOIN(auto &threads) const noexcept {
    for (auto &th : threads)
      th.join();
  }
  void th(auto dim, auto &foo, int dim_begin = 0) const noexcept {
    std::vector<std::thread> threads(max_thread);
    auto part = dim >> LOG(max_thread);
    for (auto i = threads.begin(); i < prev(threads.end()); ++i) {
      *i = std::thread(foo, dim_begin, dim_begin + part);
      dim_begin += part;
    }
    *prev(threads.end()) = std::thread(foo, dim_begin, dim);
    JOIN(threads);
  }
  ///
  /// \brief helper: dot_product hadamard_vector
  ///
  T _dot(const Mtx &m) const noexcept {
    T ans = 0;
    auto it1 = (*mtx.begin()).begin(), it_end = (*mtx.begin()).end(),
         it2 = (*m.mtx.begin()).begin();
    auto foo = [&ans, &it1, &it2](auto it_end) {
      while (it1 != it_end)
        ans += *it1++ * *it2++;
    };
    std::vector<std::thread> threads(max_thread);
    auto part = it_end - it1 >> LOG(max_thread);
    auto th_end = prev(threads.end());
    for (auto i = threads.begin(); i < th_end; ++i)
      *i = std::thread(foo, it1 + part);
    *th_end = std::thread(foo, it_end);
    JOIN(threads);
    return ans;
  }
  Mtx _h_p_vec(const Mtx &m) const noexcept {
    Mtx ans(1, vec_dim());
    auto it1 = (*mtx.begin()).begin(), it_end = (*mtx.begin()).end(),
         it2 = (*m.mtx.begin()).begin();
    auto it_ans = (*ans.mtx.begin()).begin();
    auto foo = [&it_ans, &it1, &it2](auto it_end) {
      while (it1 != it_end)
        *it_ans++ = *it1++ * *it2++;
    };

    std::vector<std::thread> threads(max_thread);
    auto part = it_end - it1 >> LOG(max_thread);
    auto th_end = prev(threads.end());
    for (auto i = threads.begin(); i < th_end; ++i)
      *i = std::thread(foo, it1 + part);
    *th_end = std::thread(foo, it_end);
    JOIN(threads);

    return ans;
  }
  Mtx _transpose() const noexcept {
    Mtx ans(dim1, dim0);
    auto foo = [this, &ans](auto dim_begin, auto dim_end) {
      auto i = dim_begin / dim1;
      auto j = dim_begin - i * dim1;
      for (auto a = dim_begin; a < dim_end; ++a) {
        if (j > dim1 - 1) {
          j = 0;
          ++i;
        }
        ans.mtx[j][i] = mtx[i][j];
        ++j;
      }
    };
    th(dim, foo);
    return ans;
  }

  //  T _det(auto left_top, auto right_bottom) const noexcept {}

public:
  ///
  /// \brief Mtx constructors
  ///
  Mtx() {}
  Mtx(int dim0, int dim1, T init)
      : dim0(dim0), dim1(dim1), dim(dim1 * dim0),
        mtx(dim1 == 1 ? 1 : dim0,
            std::vector<T>(dim1 == 1 ? dim0 : dim1, init)) {}
  Mtx(int dim0, int dim1) : Mtx(dim0, dim1, 0) {}
  Mtx(std::initializer_list<std::vector<T>> &&mtx)
      : mtx(mtx), dim0(mtx.size()), dim1((*mtx.begin()).size()),
        dim(dim0 * dim1) {
    if (dim1 == 1)
      this->mtx = this->_transpose().mtx;
  }
  Mtx(const std::initializer_list<std::vector<T>> &mtx)
      : mtx(mtx), dim0(mtx.size()), dim1((*mtx.begin()).size()),
        dim(dim0 * dim1) {
    if (dim1 == 1)
      this->mtx = this->_transpose().mtx;
  }
  ///
  /// \brief operator + - *
  ///
  Mtx operator+(const Mtx &m) const noexcept {
    if (same_dim(m)) {
      Mtx ans(dim0, dim1);
      auto foo = [this, &ans, &m](auto dim_begin, auto dim_end) {
        auto i = dim_begin / dim1, j = dim_begin - i * dim1;
        for (auto a = dim_begin; a < dim_end; ++a) {
          if (j > dim1 - 1) {
            j = 0;
            ++i;
          }
          ans.mtx[i][j] = m.mtx[i][j] + mtx[i][j];
          ++j;
        }
      };
      th(dim, foo);
      return ans;
    }
    std::cerr << "Error --> Dimension is mismatched!\n";
    exit(1);
  }
  Mtx operator+(const T n) const noexcept {
    Mtx ans(dim0, dim1);
    auto foo = [this, &ans, n](auto dim_begin, auto dim_end) {
      auto i = dim_begin / dim1, j = dim_begin - i * dim1;
      for (auto a = dim_begin; a < dim_end; ++a) {
        if (j > dim1 - 1) {
          j = 0;
          ++i;
        }
        ans.mtx[i][j] = n + mtx[i][j];
        ++j;
      }
    };
    th(dim, foo);
    return ans;
  }
  Mtx operator-(const Mtx &m) const noexcept {
    if (same_dim(m)) {
      Mtx ans(dim0, dim1);
      auto foo = [this, &ans, &m](auto dim_begin, auto dim_end) {
        auto i = dim_begin / dim1, j = dim_begin - i * dim1;
        for (auto a = dim_begin; a < dim_end; ++a) {
          if (j > dim1 - 1) {
            j = 0;
            ++i;
          }
          ans.mtx[i][j] = mtx[i][j] - m.mtx[i][j];
          ++j;
        }
      };
      th(dim, foo);
      return ans;
    }
    std::cerr << "Error --> Dimension is mismatched!\n";
    exit(1);
  }
  Mtx operator-(const T n) const noexcept {
    Mtx ans(dim0, dim1);
    auto foo = [this, &ans, n](auto dim_begin, auto dim_end) {
      auto i = dim_begin / dim1, j = dim_begin - i * dim1;
      for (auto a = dim_begin; a < dim_end; ++a) {
        if (j > dim1 - 1) {
          j = 0;
          ++i;
        }
        ans.mtx[i][j] = mtx[i][j] - n;
        ++j;
      }
    };
    th(dim, foo);
    return ans;
  }
  Mtx operator*(const Mtx &m) const noexcept {
    if (dim1 == m.dim0) {
      Mtx ans(dim0, m.dim1);
      if (dim0 == 1 && m.dim1 == 1)
        ans.mtx[0][0] = _dot(m);
      else {
        auto foo = [this, &ans, &m](auto dim_begin, auto dim_end) {
          auto i = dim_begin / dim1, k = dim_begin - i * dim1;
          for (auto a = dim_begin; a < dim_end; ++a) {
            if (k > dim1 - 1) {
              k = 0;
              ++i;
            }
            auto &s_a = mtx[i][k];
            auto &s_b = m.mtx[k];
            auto &s_c = ans.mtx[i];
            for (auto j = 0; j < m.dim1; ++j)
              s_c[j] += s_a * s_b[j];
            ++k;
          }
        };
        th(dim, foo);
      }
      return ans;
    }
    std::cerr << "Error --> Dimension is mismatched!\n";
    exit(2);
  }
  Mtx operator*(const T n) const noexcept {
    Mtx ans(dim0, dim1);
    auto foo = [this, &ans, n](auto dim_begin, auto dim_end) {
      auto i = dim_begin / dim1, j = dim_begin - i * dim1;
      for (auto a = dim_begin; a < dim_end; ++a) {
        if (j > dim1 - 1) {
          j = 0;
          ++i;
        }
        ans.mtx[i][j] = n * mtx[i][j];
        ++j;
      }
    };
    th(dim, foo);
    return ans;
  }
  ///
  /// \brief hadamard_product kronecker_product
  ///
  Mtx h_product(const Mtx &m) const noexcept {
    if (same_dim(m)) {
      Mtx ans(dim0, dim1);
      auto foo = [this, &ans, &m](auto dim_begin, auto dim_end) {
        auto i = dim_begin / dim1, j = dim_begin - i * dim1;
        for (auto a = dim_begin; a < dim_end; ++a) {
          if (j > dim1 - 1) {
            j = 0;
            ++i;
          }
          ans.mtx[i][j] = m.mtx[i][j] * mtx[i][j];
          ++j;
        }
      };
      th(dim, foo);
      return ans;
    }
    if (is_vec() && m.is_vec() && vec_dim() == m.vec_dim())
      return _h_p_vec(m);
    std::cerr << "Error --> Dimension is mismatched!\n";
    exit(1);
  }
  //  Mtx k_product(const Mtx &m) const noexcept { return *this; }

  T L2() const noexcept {
    T ans = 0;

    auto foo = [this, &ans](auto dim_begin, auto dim_end) {
      auto i = dim_begin / dim1, j = dim_begin - i * dim1;
      for (auto a = dim_begin; a < dim_end; ++a) {
        if (j > dim1 - 1) {
          j = 0;
          ++i;
        }
        auto val = mtx[i][j];
        ans += val * val;
        ++j;
      }
    };
    th(dim, foo);

    return ans;
  }
  T norm2() const noexcept { return std::sqrt(L2()); }
  ///
  /// \brief transpose inverse adjoint
  ///
  Mtx transpose() const noexcept {
    if (is_vec()) {
      Mtx ans = *this;
      std::swap(ans.dim0, ans.dim1);
      return ans;
    } else
      return _transpose();
  }
  T tr() const noexcept {
    if (is_square()) {
      T ans = 0;
      auto foo = [&ans, this](auto dim_begin, auto dim_end) {
        for (auto i = dim_begin; i < dim_end; ++i)
          ans += mtx[i][i];
      };
      th(dim0, foo);
      return ans;
    }
    std::cerr << "It is not a square matrix.\n";
    exit(5);
  }

  // to do
  // *----------------------------------------------------------------------
  Mtx inv() const noexcept { return *this; }
  Mtx adjoint() const noexcept { return *this; }
  auto eigen_val() const noexcept { return 0; }
  auto eigen_vec() const noexcept { return 0; }
  T det() const noexcept {
    if (is_square()) {
      return 0;
    }
    std::cerr << "It is not a square matrix.\n";
    exit(5);
  }
  T rank() const noexcept { return 0; }
  // *----------------------------------------------------------------------
  ///
  /// \brief Accessment
  ///
  inline auto row(auto r) const noexcept { return mtx[r]; }
  auto col(auto c) const noexcept {
    std::vector<T> ans(dim0);
    auto foo = [this, &ans, c](auto dim_begin, auto dim_end) {
      for (auto r = dim_begin; r < dim_end; ++r)
        ans[r] = mtx[r][c];
    };
    th(dim0, foo);
    return ans;
  }
  inline auto get(auto x, auto y) const noexcept { return mtx[x][y]; }
  inline T unwrap() const noexcept {
    if (dim0 == 1 && dim1 == 1)
      return mtx[0][0];
    std::cerr << "Not single value to unwrap.\n";
    exit(4);
  }
  ///
  /// \brief Debug tools
  ///
  void display(bool dims = false) const noexcept {
    for (auto &i : mtx) {
      for (auto j : i)
        std::cout << j << " ";
      std::cout << "\n";
    }
    if (dims)
      std::cout << "dimension: {" << dim0 << ", " << dim1 << "}\n";
  }
  void test_private() { return; }
};
#endif
