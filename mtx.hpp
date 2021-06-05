#ifndef MTX_HPP
#define MTX_HPP
#include <cmath>
#include <iostream>
#include <memory>
#pragma GCC optimize(3)
// const auto max_thread = std::thread::hardware_concurrency();

class M {
  using T = float;        // element type
  std::unique_ptr<T[]> v; // container
  int dim0, dim1, dim;    // dimension
  T *begin, *end;
  /* tools */
  constexpr bool same_dim(const M &m) const noexcept {
    return dim0 == m.dim0 && dim1 == m.dim1;
  }
  constexpr bool is_vec() const noexcept { return std::min(dim0, dim1) == 1; }
  constexpr auto vec_dim() const noexcept {
    if (is_vec())
      return std::max(dim0, dim1);
    else {
      std::cerr << "Not a vector!\n";
      exit(1);
    }
  }
  constexpr bool is_square() const noexcept { return dim0 == dim1; }
  constexpr bool is_single() const noexcept { return dim0 == 1 && dim1 == 1; }
  /* private helper */
  /// \brief transpose
  M _t() const noexcept {
    M ans(dim1, dim0);
    auto it = end;
    auto i = dim0 - 1, tmp = dim - dim0, s = tmp;
    auto v_a = ans.begin;
    do {
      v_a[s + i] = *--it;
      s = s ? s - dim0 : (--i, tmp);
    } while (it > begin);
    return ans;
  }
  //  T _det(auto left_top, auto right_bottom) const noexcept {}

public:
  /* constructors */
  inline M() {}
  M(auto dim0, auto dim1, T init)
      : dim0(dim0), dim1(dim1), v(new T[dim0 * dim1]) {
    if (init != 0) {
      this->dim = dim0 * dim1;
      this->begin = v.get(), this->end = begin + this->dim;
      auto it = this->end;
      do {
        *--it = init;
      } while (it > begin);
    }
  }
  inline M(auto dim0, auto dim1)
      : dim0(dim0), dim1(dim1), v(new T[dim0 * dim1]) {
    this->dim = dim0 * dim1;
    this->begin = v.get(), this->end = begin + this->dim;
  }
  // debug only
  M(auto dim0, auto dim1, const std::initializer_list<T> m)
      : dim0(dim0), dim1(dim1), v(new T[dim0 * dim1]) {
    this->dim = dim0 * dim1;
    this->begin = v.get(), this->end = begin + this->dim;
    auto it = this->begin;
    for (auto &i : m)
      *it++ = i;
  }

  /* deep copy */
  M clone() const noexcept {
    M ans(dim0, dim1);
    auto it = end;
    auto it_a = ans.end;
    do {
      *--it_a = *--it;
    } while (it > begin);
    return ans;
  }

  /* operator "+ - *" */
  M operator+(const M &m) const noexcept {
    if (same_dim(m)) {
      M ans(dim0, dim1);
      auto it = end, it_m = m.end;
      auto it_a = ans.end;
      do {
        *--it_a = *--it + *--it_m;
      } while (it > begin);
      return ans;
    }
    std::cerr << "Dimension is mismatched!\n";
    exit(2);
  }
  M operator+(const T n) const noexcept {
    M ans(dim0, dim1);
    auto it = end;
    auto it_a = ans.end;
    do {
      *--it_a = *--it + n;
    } while (it > begin);
    return ans;
  }
  M operator-(const M &m) const noexcept {
    if (same_dim(m)) {
      M ans(dim0, dim1);
      auto it = end, it_m = m.end;
      auto it_a = ans.end;
      do {
        *--it_a = *--it - *--it_m;
      } while (it > begin);
      return ans;
    }
    std::cerr << "Dimension is mismatched!\n";
    exit(2);
  }
  M operator-(const T n) const noexcept {
    M ans(dim0, dim1);
    auto it = end;
    auto it_a = ans.end;
    do {
      *--it_a = *--it - n;
    } while (it > begin);
    return ans;
  }
  M operator*(const M &m) const noexcept {
    if (dim1 == m.dim0) {
      M ans(dim0, m.dim1);
      auto it = end;
      auto i = dim0 - 1, k = dim1;
      do {
        auto j = m.dim1;
        auto p_m = m.begin + j + (!k ? (--i, k = dim1 - 1) : --k) * m.dim1,
             p_a = ans.begin + j + i * m.dim1;
        do {
          *--p_a += *--it * *--p_m;
        } while (--j > 0);
      } while (it > begin);
      return ans;
    }
    std::cerr << "Dimension is mismatched!\n";
    exit(2);
  }
  M operator*(const T n) const noexcept {
    M ans(dim0, dim1);
    auto it = end;
    auto it_a = ans.end;
    do {
      *--it_a = *--it * n;
    } while (it > begin);
    return ans;
  }

  /* functional opeartor */
  constexpr T dot(const M &m) const noexcept {
    if (is_vec() && m.is_vec()) {
      if (vec_dim() == m.vec_dim()) {
        T ans = 0;
        auto it = end, it_m = m.end;
        do {
          ans += *--it * *--it_m;
        } while (it > begin);
        return ans;
      }
      std::cerr << "Length doesn't match!\n";
      exit(3);
    }
    std::cerr << "Not a vector!\n";
    exit(1);
  }
  M t() const noexcept {
    if (is_vec()) {
      auto ans = this->clone();
      std::swap(ans.dim0, ans.dim1);
      return ans;
    } else
      return _t();
  }
  /// \brief hadamard product
  M h_p(const M &m) const noexcept {
    if (same_dim(m)) {
      M ans(dim0, dim1);
      auto it = end, it_m = m.end;
      auto it_a = ans.end;
      do {
        *--it_a = *--it * *--it_m;
      } while (it > begin);
      return ans;
    }
    std::cerr << "Dimension is mismatched!\n";
    exit(2);
  }
  //  M k_p(const M &m) const noexcept { return *this; }

  /* matrix property */
  constexpr T L2() const noexcept {
    T ans = 0;
    auto it = end;
    do {
      auto n = *--it;
      ans += n * n;
    } while (it > begin);
    return ans;
  }
  constexpr T norm2() const noexcept { return std::sqrt(L2()); }
  constexpr T tr() const noexcept {
    if (is_square()) {
      T ans = 0;
      auto it = end;
      do {
        ans += *--it;
      } while (it -= dim0,it > begin);
      return ans;
    }
    std::cerr << "Not a square matrix!\n";
    exit(4);
  }
  // to do
  // *----------------------------------------------------------------------
  //  M inv() const noexcept { return *this; }
  //  M adjoint() const noexcept { return *this; }
  //  auto eigen_val() const noexcept { return 0; }
  //  auto eigen_vec() const noexcept { return 0; }
  //  constexpr T det() const noexcept {
  //    if (is_square()) {
  //      return 0;
  //    }
  //    std::cerr << "Not a square matrix.\n";
  //    exit(4);
  //  }
  //  constexpr T rank() const noexcept { return 0; }
  // *----------------------------------------------------------------------

  /* extract element */
  auto row(auto r) const noexcept {
    if (r > -1 && r < dim0) {
      M ans(1, dim1);
      auto head = begin + r * dim1, it = head + dim1, it_a = ans.end;
      do {
        *--it_a = *--it;
      } while (it > head);
      return ans;
    }
    std::cerr << "Out of bound!\n";
    exit(5);
  }
  auto col(auto c) const noexcept {
    if (c > -1 && c < dim1) {
      M ans(dim0, 1);
      auto tmp = dim1 - 1;
      auto it = end - tmp + c;
      auto it_a = ans.end;
      do {
        *--it_a = *--it;
      } while (it -= tmp, it > begin);
      return ans;
    }
    std::cerr << "Out of bound!\n";
    exit(5);
  }
  constexpr auto at(auto x, auto y) const noexcept {
    return begin[x * dim1 + y];
  }
  constexpr T unwrap() const noexcept {
    if (is_single())
      return *begin;
    std::cerr << "Not single value.\n";
    exit(6);
  }

  /* debug tools */
  void dims() const noexcept {
    std::cout << "dimension: {" << dim0 << ", " << dim1 << "}\n";
  }
  void display() const noexcept {
    auto it = begin;
    auto i = 0;
    while (it < end) {
      std::cout << *it++ << " ";
      if (++i == dim1) {
        i = 0;
        std::cout << "\n";
      }
    }
  }
};
#endif
