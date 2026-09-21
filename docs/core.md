# core — containers

The foundation. This layer knows nothing about ML: no losses, no models, no
statistics. Everything else depends on it, so it must be boring and exact.

Depends on: ForgeFP (`fp::Buffer`, `fp::Result`, `fp::Validation`, `fp::grid`,
`fp::linalg`).

Files: `shape.hpp`, `vector.hpp`, `matrix.hpp`, `tensor.hpp`.

**ForgeFP usage at a glance**

| File | ForgeFP functions |
|---|---|
| `shape.hpp` | `fp::Result`, `fp::fail`, `fp::ok` |
| `vector.hpp` | `fp::map`, `fp::fold_left`, `fp::sum`, `fp::simd::dot`, `fp::approx_equal` (tests) |
| `matrix.hpp` | `fp::Validation`, `fp::to_result`, `fp::transpose`, `fp::map2d`, `fp::for_each_index`, `fp::linalg::*` (directly on the grid) |
| `tensor.hpp` | `fp::Buffer`, `fp::Result`, `fp::for_each`, `fp::transform_inplace`, `fp::simd::map_inplace` |

There is no `core/random.hpp`: randomness is `fp::Rng` from
`fp/random.hpp`.

---

## `core/shape.hpp`

**Job:** describe dimensions and answer compatibility questions in one place, so
dimension-checking logic is not duplicated across `model`, `nn`, and the data
layer.

```cpp
namespace forgeml {

struct Shape {
  std::vector<std::size_t> dims;

  std::size_t rank() const { return dims.size(); }
  std::size_t numel() const;                 // product of dims, 0 if any dim is 0
  std::size_t operator[](std::size_t i) const { return dims[i]; }

  bool operator==(Shape const &) const = default;

  bool is_matrix() const { return rank() == 2; }
  bool is_vector() const { return rank() == 1; }
};

// Compatibility helpers (return Result with a descriptive message).
fp::Result<void> require_same_shape(Shape const &a, Shape const &b);
fp::Result<void> require_matmul(Shape const &a, Shape const &b);   // a[1] == b[0]
fp::Result<void> require_square(Shape const &s);
fp::Result<void> require_numel(Shape const &s, std::size_t expected);

inline Shape make_shape(std::size_t rows, std::size_t cols);       // {rows, cols}
}
```

Invariants:

- `numel()` is the product of `dims`; empty shape has `numel() == 0`.
- `require_*` never throw; they return `fp::fail(...)`.

Tests (`test/core_test.cpp`):

- `Shape({2,3}).numel() == 6`, `Shape({}).numel() == 0`
- `require_matmul({2,3}, {3,4})` ok; `{2,3}, {4,3}` fails with a message
  containing both inner sizes.

---

## `core/vector.hpp`

**Job:** a contiguous 1-D numeric container that is a first-class C++ range, so
every ForgeFP algorithm works on it directly. No math beyond storage/access.

```cpp
namespace forgeml {

template <class T = double>
class Vector {
public:
  using value_type = T;
  using iterator = typename std::vector<T>::iterator;
  using const_iterator = typename std::vector<T>::const_iterator;

  Vector() = default;
  explicit Vector(std::size_t n, T fill = T{});
  Vector(std::initializer_list<T> init);
  explicit Vector(std::vector<T> data);      // takes ownership

  std::size_t size() const;
  bool empty() const;
  T *data();
  T const *data() const;

  T &operator[](std::size_t i);              // unchecked
  T const &operator[](std::size_t i) const;

  T &at(std::size_t i);                      // ML_ASSERT on bounds
  T const &at(std::size_t i) const;

  iterator begin(); const_iterator begin() const;
  iterator end();   const_iterator end() const;

  void resize(std::size_t n, T fill = T{});
  void push_back(T value);
  void clear();

  std::span<T> span();                       // for hot, non-owning APIs
  std::span<const T> span() const;

  static Vector zeros(std::size_t n);
  static Vector ones(std::size_t n);
  static fp::Result<Vector> make(std::vector<T> values);

private:
  std::vector<T> data_;
};
}
```

Design decisions:

- Backed by `std::vector<T>`; `begin/end` make it work with `fp::map`,
  `fp::sum`, `fp::zip_with`, `fp::simd`.
- No arithmetic operators on the class; math is `fp::linalg` (or the thin
  `math/` remainder) as free functions returning `fp::Result` when sizes can
  disagree. Operators would hide failure; `Result` makes it explicit.
- `operator[]` unchecked for hot loops, `at` asserts for debugging.
- `span()` is the API for inner loops and `fp::simd` calls.

ForgeFP usage: `fp::map`/`fp::fold_left`/`fp::sum` in tests and utilities;
`fp::simd::dot` for the hot inner product; `fp::approx_equal` for comparisons.

Tests:

- construction, copy/move, `resize`, iterators, `span().size() == size()`
- `fp::sum(v)` and `fp::simd::dot(v, v)` are correct on a known vector
- `at` out of range triggers `ML_ASSERT`

---

## `core/matrix.hpp`

**Job:** the main 2-D numeric type — and deliberately the *same shape* ForgeFP's
`linalg`/`grid` understand, so `fp::matmul`, `fp::transpose`, `fp::map2d`,
`fp::for_each_index`, `fp::row_sums`, … apply with no adapter.

```cpp
namespace forgeml {

template <class T = double>
class Matrix {
public:
  Matrix() = default;
  Matrix(std::size_t rows, std::size_t cols, T fill = T{});
  Matrix(std::initializer_list<Vector<T>> rows);   // validates ragged

  std::size_t rows() const;
  std::size_t cols() const;                        // rows_[0].size()
  Shape shape() const { return {rows(), cols()}; }

  Vector<T> &operator[](std::size_t r);            // row access
  Vector<T> const &operator[](std::size_t r) const;

  T &operator()(std::size_t r, std::size_t c);
  T const &operator()(std::size_t r, std::size_t c) const;

  Vector<T> &row(std::size_t r);
  Vector<T> const &row(std::size_t r) const;
  Vector<T> col(std::size_t c) const;              // copy (columns are strided)

  void fill(T value);

  // Grid interop: the underlying representation, for fp::linalg/fp::grid.
  std::vector<Vector<T>> &grid();
  std::vector<Vector<T>> const &grid() const;

  static Matrix zeros(std::size_t r, std::size_t c);
  static Matrix ones(std::size_t r, std::size_t c);
  static Matrix identity(std::size_t n);
  static fp::Result<Matrix> make(std::vector<Vector<T>> rows);   // ragged -> fail

private:
  std::vector<Vector<T>> rows_;
};

template <class T>
std::ostream &operator<<(std::ostream &os, Matrix<T> const &m);  // debug only
}
```

Design decisions:

- **Grid-backed** (`std::vector<Vector<T>>`) rather than a flat buffer: this is
  exactly `fp::linalg`'s matrix representation, so the hottest kernels
  (`matmul`, `solve`, reductions) come from fp with no conversion. The cost —
  one allocation per row — is a non-issue for the matrix sizes this stack
  targets, and fp's `i-k-j` matmul already gives a 1.75x win over the naive
  order.
- For raw, single-allocation contiguous memory (SIMD scratch, embedding tables,
  checkpoint buffers) use `fp::Buffer<T>` directly — that is what it is for.
- Ragged construction is a `Result` (`make`) or an `ML_ASSERT` (constructor);
  `fp::Validation` is available for collecting *all* bad rows.
- `grid()` exposes the representation so callers can hand it to fp without a
  copy: `fp::matmul(a.grid(), b.grid())`.

ForgeFP usage: `fp::Validation`/`fp::to_result` (ragged checks), `fp::transpose`,
`fp::map2d`, `fp::for_each_index`, and all of `fp::linalg` on `grid()`.

Tests:

- indexing matches the grid layout; `row(r)` is the grid row; `col(c)` matches
  element-wise
- `fp::matmul(m.grid(), identity.grid()) == m`
- ragged initializer triggers `ML_ASSERT`; `make` returns `fp::fail`
- `col_sums`/`col_means` (fp) match hand-computed values

---

## `core/tensor.hpp`

**Job:** N-dimensional dense array for CNNs and sequence models, backed by one
`fp::Buffer<T>` allocation.

```cpp
namespace forgeml {

template <class T = double>
class Tensor {
public:
  Tensor() = default;
  explicit Tensor(Shape shape, T fill = T{});
  Tensor(Shape shape, fp::Buffer<T> data);         // size check -> Result

  Shape const &shape() const;
  std::size_t rank() const;
  std::size_t numel() const;

  T &operator[](std::size_t flat);
  T const &operator[](std::size_t flat) const;

  T &at(std::vector<std::size_t> const &idx);      // row-major strides
  T const &at(std::vector<std::size_t> const &idx) const;

  T *data();
  T const *data() const;

  std::span<T> span();                             // the whole flat buffer
  T *begin();  T *end();                           // range interface

  void reshape(Shape shape);                       // numel must match

  static fp::Result<Tensor<T>> make(Shape shape);  // Buffer::alloc

private:
  Shape shape_;
  std::vector<std::size_t> strides_;               // row-major
  fp::Buffer<T> data_;
};
}
```

Rules:

- Strides are computed once in the constructor: `stride[i] = prod(shape[i+1:])`.
- `at` asserts `idx.size() == rank()` and each index in range.
- The flat buffer is a range, so `fp::for_each`, `fp::transform_inplace`, and
  `fp::simd::map_inplace` work on the tensor directly.
- 4-D convention for images: `(batch, channels, height, width)`. 3-D convention
  for sequences: `(batch, time, features)`. Documented in `conv2d.hpp` and
  never mixed.
- `make` returns `Result` because `fp::Buffer::alloc` does; the constructor
  asserts.

ForgeFP usage: `fp::Buffer`, `fp::Result`, `fp::for_each`,
`fp::transform_inplace`, `fp::simd::map_inplace`.

Tests:

- 3-D `at({i,j,k})` equals flat `data_[(i*J + j)*K + k]`
- `reshape({2,3}) -> {3,2}` preserves element order; mismatched numel asserts
- `fp::sum`/`fp::fold_left` over the tensor matches the hand-computed value
- `make` on a huge shape returns `fp::fail("allocation failed")` (or succeeds;
  the point is it never throws)

---

## Cross-cutting notes

- Every container is move-friendly: pass by value only for sinks, otherwise
  `const&` or `span`.
- No container overrides `new`/`delete`; allocation goes through `std::vector`
  or `fp::Buffer`.
- `utils/assertions.hpp` is a Stage 1 dependency: implement it first, even
  though it lives under `utils/`.
