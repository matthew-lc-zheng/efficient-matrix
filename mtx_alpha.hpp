#ifndef MTX_HPP
#define MTX_HPP
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#pragma GCC optimize(3)
class M {
  using T = float;           // element type
  std::unique_ptr<T[]> __v;  // container
  T *__begin, *__end;        // reference pointers
  int __dim0, __dim1, __dim; // dimension

  /* tools */
  constexpr bool _same_dim(const M &__m) const noexcept {
    return __dim0 == __m.__dim0 && __dim1 == __m.__dim1;
  }
  constexpr bool _is_vec() const noexcept {
    return std::min(__dim0, __dim1) == 1;
  }
  constexpr auto _vec_dim() const noexcept {
    if (_is_vec())
      return __dim;
    std::cerr << "Not a vector!\n";
    exit(1);
  }
  constexpr bool _is_square() const noexcept { return __dim0 == __dim1; }
  constexpr bool _is_single() const noexcept {
    return __dim0 == 1 && __dim1 == 1;
  }

  /* private helper */
  /// \brief transpose
  M _t() const noexcept {
    M ans(__dim1, __dim0);
    auto it = __end;
    auto i = __dim0 - 1, tmp = __dim - __dim0, s = tmp;
    auto v_a = ans.__begin;
    do {
      v_a[s + i] = *--it;
      s = s ? s - __dim0 : (--i, tmp);
    } while (it > __begin);
    return ans;
  }

public:
  /* constructors */
  inline M() {}
  M(auto dim0, auto dim1, T init)
      : __dim0(dim0), __dim1(dim1), __v(std::make_unique<T[]>(dim0 * dim1)) {
    __dim = dim0 * dim1;
    __begin = __v.get();
    __end = __begin + __dim;
    auto it = __end;
    do {
      *--it = init;
    } while (it > __begin);
  }
  inline M(auto dim0, auto dim1)
      : __dim0(dim0), __dim1(dim1), __v(std::make_unique<T[]>(dim0 * dim1)) {
    __dim = dim0 * dim1;
    __begin = __v.get();
    __end = __begin + __dim;
  }
  // debug only
  M(auto dim0, auto dim1, std::initializer_list<T> m)
      : __dim0(dim0), __dim1(dim1), __v(std::make_unique<T[]>(dim0 * dim1)) {
    __dim = __dim0 * __dim1;
    __begin = __v.get();
    __end = __begin + __dim;
    auto it = __begin;
    for (auto &i : m)
      *it++ = i;
  }
  M clone() const noexcept {
    M ans(__dim0, __dim1);
    std::memcpy(ans.__begin, __begin, __dim * sizeof(T));
    return ans;
  }
  /* operator "+ - *" */
  M operator+(const M &m) const noexcept {
    if (_same_dim(m)) {
      M ans(__dim0, __dim1);
      auto it = __end, it_m = m.__end;
      auto it_a = ans.__end;
      do {
        *--it_a = *--it + *--it_m;
      } while (it > __begin);
      return ans;
    }
    std::cerr << "Dimension is mismatched!\n";
    exit(2);
  }
  M operator+(const T n) const noexcept {
    M ans(__dim0, __dim1);
    auto it = __end;
    auto it_a = ans.__end;
    do {
      *--it_a = *--it + n;
    } while (it > __begin);
    return ans;
  }
  M operator-(const M &m) const noexcept {
    if (_same_dim(m)) {
      M ans(__dim0, __dim1);
      auto it = __end, it_m = m.__end;
      auto it_a = ans.__end;
      do {
        *--it_a = *--it - *--it_m;
      } while (it > __begin);
      return ans;
    }
    std::cerr << "Dimension is mismatched!\n";
    exit(2);
  }
  M operator-(const T n) const noexcept {
    M ans(__dim0, __dim1);
    auto it = __end;
    auto it_a = ans.__end;
    do {
      *--it_a = *--it - n;
    } while (it > __begin);
    return ans;
  }
  M operator*(const M &m) const noexcept {
    if (__dim1 == m.__dim0) {
      M ans(__dim0, m.__dim1);
      auto it = __end;
      auto i = __dim0 - 1, k = __dim1, j = 0;
      do {
        --it;
        j = m.__dim1;
        auto p_m = m.__begin + j + (!k ? (--i, k = __dim1 - 1) : --k) * j,
             p_a = ans.__begin + j + i * j;
        do {
          *--p_a += *it * *--p_m;
        } while (--j > 0);
      } while (it > __begin);
      return ans;
    }
    std::cerr << "Dimension is mismatched!\n";
    exit(2);
  }
  M operator*(const T n) const noexcept {
    M ans(__dim0, __dim1);
    auto it = __end;
    auto it_a = ans.__end;
    do {
      *--it_a = *--it * n;
    } while (it > __begin);
    return ans;
  }

  /* functional opeartor */
  constexpr T dot(const M &m) const noexcept {
    if (_is_vec() && m._is_vec()) {
      if (_vec_dim() == m._vec_dim()) {
        T ans = 0;
        auto it = __end, it_m = m.__end;
        do {
          ans += *--it * *--it_m;
        } while (it > __begin);
        return ans;
      }
      std::cerr << "Length doesn't match!\n";
      exit(3);
    }
    std::cerr << "Not a __vector!\n";
    exit(1);
  }
  M t() const noexcept {
    if (_is_vec()) {
      auto ans = this->clone();
      std::swap(ans.__dim0, ans.__dim1);
      return ans;
    } else
      return _t();
  }
  //  /// \brief hadamard product
  M h_p(const M &m) const noexcept {
    if (_same_dim(m)) {
      M ans(__dim0, __dim1);
      auto it = __end, it_m = m.__end;
      auto it_a = ans.__end;
      do {
        *--it_a = *--it * *--it_m;
      } while (it > __begin);
      return ans;
    }
    std::cerr << "Dimension is mismatched!\n";
    exit(2);
  }
  //  //  M k_p(const M &m) const noexcept { return *this; }

  /* matrix property */
  constexpr T L2() const noexcept {
    T ans = 0;
    auto it = __end;
    do {
      auto n = *--it;
      ans += n * n;
    } while (it > __begin);
    return ans;
  }
  constexpr T norm2() const noexcept { return std::sqrt(L2()); }
  constexpr T tr() const noexcept {
    if (_is_square()) {
      T ans = 0;
      auto it = __end;
      do {
        ans += *--it;
      } while (it -= __dim0, it > __begin);
      return ans;
    }
    std::cerr << "Not a square matrix!\n";
    exit(4);
  }
  // to do
  // *----------------------------------------------------------------------
  //  M in__v() const noexcept { return *this; }
  //  M adjoint() const noexcept { return *this; }
  //  auto eigen___val() const noexcept { return 0; }
  //  auto eigen___vec() const noexcept { return 0; }
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
    if (r > -1 && r < __dim0) {
      M ans(1, __dim1);
      auto head = __begin + r * __dim1, it = head + __dim1, it_a = ans.__end;
      do {
        *--it_a = *--it;
      } while (it > head);
      return ans;
    }
    std::cerr << "Out of bound!\n";
    exit(5);
  }
  auto col(auto c) const noexcept {
    if (c > -1 && c < __dim1) {
      M ans(__dim0, 1);
      auto tmp = __dim1 - 1;
      auto it = __end - tmp + c;
      auto it_a = ans.__end;
      do {
        *--it_a = *--it;
      } while (it -= tmp, it > __begin);
      return ans;
    }
    std::cerr << "Out of bound!\n";
    exit(5);
  }
  constexpr auto at(auto x, auto y) const noexcept {
    return __begin[x * __dim1 + y];
  }
  T unwrap() const noexcept {
    if (_is_single())
      return *__begin;
    std::cerr << "Not a single value!\n";
    exit(6);
  }

  /* debug tools */
  void dims() const noexcept {
    std::cout << "Dimension: {" << __dim0 << ", " << __dim1 << "}\n";
  }
  void display() const noexcept {
    auto it = __begin;
    auto i = 0;
    do {
      std::cout << *it << " ";
      if (++i == __dim1) {
        i = 0;
        std::cout << "\n";
      }
    } while (++it < __end);
  }
};
#endif
