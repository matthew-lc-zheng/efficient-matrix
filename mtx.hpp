#include <bits/stdc++.h>
#pragma GCC optimize(3)
#define max_thread std::thread::hardware_concurrency()

class Mtx {
  /* type of elements */
  using T = float;
  /* data */
  std::vector<std::vector<T>> mtx; // container to store tensor
  /* property */
  int dim0, dim1, dim;
  ///
  /// \brief tools
  ///
  inline bool wrong_dim(const Mtx &m) const noexcept {
    return !(dim0 == m.dim0 && dim1 == m.dim1);
  }
  inline unsigned int LOG(unsigned int x) const noexcept {
    unsigned int ret;
    __asm__ __volatile__("bsrl %1, %%eax" : "=a"(ret) : "m"(x));
    return ret;
  }
  ///
  /// \brief processing through multiple threads
  /// \param dim: stop point
  /// \param foo: content to process
  /// \param dim_begin: default start point is zero
  ///
  void th(auto dim, auto &foo, int dim_begin = 0) const noexcept {
    std::vector<std::thread> threads(max_thread);
    auto part = dim >> LOG(max_thread);
    for (auto i = threads.begin(); i < prev(threads.end()); ++i) {
      *i = std::move(std::thread(foo, dim_begin, dim_begin + part));
      dim_begin += part;
    }
    threads[max_thread - 1] = std::move(std::thread(foo, dim_begin, dim));
    for (auto &th : threads)
      th.join();
  }

public:
  ///
  /// \brief Mtx constructors
  ///
  Mtx() {}
  Mtx(int dim0, int dim1, T init)
      : dim0(dim0), dim1(dim1), dim(dim1 * dim0),
        mtx(dim0, std::vector<T>(dim1, init)) {}
  Mtx(int dim0, int dim1) : Mtx(dim0, dim1, 0) {}
  Mtx(std::vector<std::vector<T>> &&mtx)
      : mtx(mtx), dim0(mtx.size()), dim1(mtx[0].size()), dim(dim0 * dim1) {}
  Mtx(const std::vector<std::vector<T>> &mtx)
      : mtx(mtx), dim0(mtx.size()), dim1(mtx[0].size()), dim(dim0 * dim1) {}
  ///
  /// \brief operator + - *
  ///
  Mtx operator+(const Mtx &m) const noexcept {
    if (wrong_dim(m)) {
      std::cerr << "Error --> Dimension is mismatched!\n";
      exit(1);
    }
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
    return std::move(ans);
  }
  Mtx operator-(const Mtx &m) const noexcept {
    if (wrong_dim(m)) {
      std::cerr << "Error --> Dimension is mismatched!\n";
      exit(1);
    }
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
    return std::move(ans);
  }
  Mtx operator*(const Mtx &m) const noexcept {
    if (dim1 != m.dim0) {
      std::cerr << "Error --> Dimension is mismatched!\n";
      exit(2);
    }
    Mtx ans(dim0, m.dim1);
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
    return std::move(ans);
  }
  ///
  /// \brief h_product k_product
  ///
  Mtx h_product(const Mtx &m) const noexcept {
    if (wrong_dim(m)) {
      std::cerr << "Error --> Dimension is mismatched!\n";
      exit(1);
    }
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
    return std::move(ans);
  }
  //  Mtx k_product(const Mtx &m) const noexcept {}
  ///
  /// \brief transpose inverse adjoint
  ///
  Mtx transpose() const noexcept {
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
    return std::move(ans);
  }
  // to do
  // *----------------------------------------------------------------------
  Mtx inv() const noexcept { return *this; }
  Mtx adjoint() const noexcept { return *this; }

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
    return std::move(ans);
  }
  inline auto get(auto x, auto y) const noexcept { return mtx[x][y]; }
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
};
