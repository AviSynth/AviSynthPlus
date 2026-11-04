// AviSynth+.  Copyright 2025 AviSynth+ Project
// https://avs-plus.net
// http://avisynth.nl
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA, or visit
// http://www.gnu.org/copyleft/gpl.html .
//
// Linking Avisynth statically or dynamically with other modules is making a
// combined work based on Avisynth.  Thus, the terms and conditions of the GNU
// General Public License cover the whole combination.
//
// As a special exception, the copyright holders of Avisynth give you
// permission to link Avisynth with independent modules that communicate with
// Avisynth solely through the interfaces defined in avisynth.h, regardless of the license
// terms of these independent modules, and to copy and distribute the
// resulting combined work under terms of your choice, provided that
// every copy of the combined work is accompanied by a complete copy of
// the source code of Avisynth (the version of Avisynth used to produce the
// combined work), being distributed under the terms of the GNU General
// Public License plus this exception.  An independent module is a module
// which is not derived from or based on Avisynth, such as 3rd-party filters,
// import and export plugins, or graphical user interfaces.

// SIMD-like C++ classes (C)2025 Ferenc Pintér

#ifndef __AVS_SIMD_C_H__
#define __AVS_SIMD_C_H__
// Auto-vectorization friendly types for smart compilers
// Some helper static functions, marked with c++ 17 [[maybe_unused]] attribute

#include <avs/config.h> // force inline variants
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <type_traits>
#include <stdexcept>

#if defined(_MSC_VER) && !defined(__clang__) && defined(INTEL_INTRINSICS)
#include <immintrin.h>  // covers SSE2 through AVX2 for MSVC
#endif

// Everything below is in an anonymous namespace: the classes (VectorWrapper, Int32x4, ...), their member
// functions and the free templates are inline code with external linkage otherwise. Once this header is
// included in a TU built with -mavx2 / /arch:AVX2 (or any other ISA flag), those would be compiled with
// AVX2 instructions, and the linker keeps one COMDAT copy for all TUs: possibly the AVX2 one for a generic
// caller. Here every TU gets its own copy, built with its own flags.
// Consequence: each including TU has its own distinct types. They cannot appear in declarations shared
// between TUs: a function declared in a header with an Int32x4 parameter and defined in another TU would
// not link (undefined reference), and GCC warns (-Wsubobject-linkage) for such class members.
// Use them inside one TU (local variables, static or anonymous-namespace functions).
namespace {

// Determine if we can use vector attribute

// As of 2025 clang (llvm) supports vector attributes and also produces fast code
// gcc can use it but may produce slower code with than without vector attributes.
// (gcc 12.2 on aarch64 Raspberry Pi 5 is slow)
// We still keep it here, maybe non-arm platforms can benefit with gcc as well.
// Note: llvm defines __clang__ as well
#if defined(__GNUC__) || defined(__clang__)
#define HAS_VECTOR_ATTRIBUTE 1
#else
#define HAS_VECTOR_ATTRIBUTE 0
#endif

// Vector type definitions
#if HAS_VECTOR_ATTRIBUTE
    // GCC/Clang supports vector attribute
using uint8_vec32_t = uint8_t __attribute__((vector_size(32)));  // 256-bit (32 bytes)
using uint8_vec16_t = uint8_t __attribute__((vector_size(16)));  // 128-bit (16 bytes)
using uint8_vec8_t  = uint8_t __attribute__((vector_size(8)));   // 64-bit  (8 bytes)
using uint8_vec4_t  = uint8_t __attribute__((vector_size(4)));   // 32-bit  (4 bytes)

using int16_vec16_t = int16_t __attribute__((vector_size(32)));  // 256-bit (32 bytes)
using int16_vec8_t  = int16_t __attribute__((vector_size(16)));  // 128-bit (16 bytes)
using int16_vec4_t  = int16_t __attribute__((vector_size(8)));   // 64-bit  (8 bytes)

using uint16_vec16_t = uint16_t __attribute__((vector_size(32))); // 256-bit (32 bytes)
using uint16_vec8_t  = uint16_t __attribute__((vector_size(16))); // 128-bit (16 bytes)
using uint16_vec4_t  = uint16_t __attribute__((vector_size(8)));  // 64-bit  (8 bytes)

using int32_vec8_t  = int32_t __attribute__((vector_size(32)));  // 256-bit (32 bytes)
using int32_vec4_t  = int32_t __attribute__((vector_size(16)));  // 128-bit (16 bytes)
using int32_vec2_t  = int32_t __attribute__((vector_size(8)));   // 64-bit  (8 bytes)

using float_vec8_t  = float __attribute__((vector_size(32)));    // 256-bit (32 bytes)
using float_vec4_t  = float __attribute__((vector_size(16)));    // 128-bit (16 bytes)
using float_vec2_t  = float __attribute__((vector_size(8)));     // 64-bit  (8 bytes)

#else

// For MSVC: use __declspec(align) + plain array in a struct, may help optimizer
template<typename T, size_t N>
struct alignas(N * sizeof(T)) Avs_SimdArray {
  T v[N];

  AVS_FORCEINLINE T& operator[](size_t i) { return v[i]; }
  AVS_FORCEINLINE const T& operator[](size_t i) const { return v[i]; }

  // Trivial fill
  AVS_FORCEINLINE void fill(T val) {
    if constexpr (sizeof(T) == 1) {
      memset(v, val, N);  // intrinsic, single rep stosd or movdqu
    }
    else {
      // Seemingly MSVC (as of in 2026) will not vectorize a loop here, but will inline these
      for (size_t i = 0; i < N; ++i) v[i] = val;
    }
  }

  // Iterators for compatibility
  AVS_FORCEINLINE T* begin() { return v; }
  AVS_FORCEINLINE T* end() { return v + N; }
  AVS_FORCEINLINE const T* begin() const { return v; }
  AVS_FORCEINLINE const T* end()   const { return v + N; }

  static constexpr size_t size() { return N; }
};

using uint8_vec32_t = Avs_SimdArray<uint8_t, 32>;
using uint8_vec16_t = Avs_SimdArray<uint8_t, 16>;
using uint8_vec8_t = Avs_SimdArray<uint8_t, 8>;
using uint8_vec4_t = Avs_SimdArray<uint8_t, 4>;

using int16_vec16_t = Avs_SimdArray<int16_t, 16>;
using int16_vec8_t = Avs_SimdArray<int16_t, 8>;
using int16_vec4_t = Avs_SimdArray<int16_t, 4>;

using uint16_vec16_t = Avs_SimdArray<uint16_t, 16>;
using uint16_vec8_t = Avs_SimdArray<uint16_t, 8>;
using uint16_vec4_t = Avs_SimdArray<uint16_t, 4>;

using int32_vec8_t = Avs_SimdArray<int32_t, 8>;
using int32_vec4_t = Avs_SimdArray<int32_t, 4>;
using int32_vec2_t = Avs_SimdArray<int32_t, 2>;
using float_vec8_t = Avs_SimdArray<float, 8>;
using float_vec4_t = Avs_SimdArray<float, 4>;
using float_vec2_t = Avs_SimdArray<float, 2>;

#endif // HAS_VECTOR_ATTRIBUTE

// Type traits for vector elements
template <typename T>
struct vector_traits {};


#define DEFINE_VECTOR_TRAITS(VecType, ElemType, Size) \
    template <> struct vector_traits<VecType> { \
        using element_type = ElemType; \
        static constexpr size_t size = Size; \
    };

// Covering 8 to 32 bytes long vectors.
// No int64_t or double, they are not relevant in AviSynth
DEFINE_VECTOR_TRAITS(uint8_vec32_t, uint8_t, 32)
DEFINE_VECTOR_TRAITS(uint8_vec16_t, uint8_t, 16)
DEFINE_VECTOR_TRAITS(uint8_vec8_t, uint8_t, 8)
DEFINE_VECTOR_TRAITS(uint8_vec4_t, uint8_t, 4)
DEFINE_VECTOR_TRAITS(int16_vec16_t, int16_t, 16)
DEFINE_VECTOR_TRAITS(int16_vec8_t, int16_t, 8)
DEFINE_VECTOR_TRAITS(int16_vec4_t, int16_t, 4)
DEFINE_VECTOR_TRAITS(uint16_vec16_t, uint16_t, 16)
DEFINE_VECTOR_TRAITS(uint16_vec8_t, uint16_t, 8)
DEFINE_VECTOR_TRAITS(uint16_vec4_t, uint16_t, 4)
DEFINE_VECTOR_TRAITS(int32_vec8_t, int32_t, 8)
DEFINE_VECTOR_TRAITS(int32_vec4_t, int32_t, 4)
DEFINE_VECTOR_TRAITS(int32_vec2_t, int32_t, 2)
DEFINE_VECTOR_TRAITS(float_vec8_t, float, 8)
DEFINE_VECTOR_TRAITS(float_vec4_t, float, 4)
DEFINE_VECTOR_TRAITS(float_vec2_t, float, 2)

// Specialization for array-based vector operations
#if !HAS_VECTOR_ATTRIBUTE
template <typename VecType, typename ScalarType = typename vector_traits<VecType>::element_type>
class VectorOps {
public:
  static AVS_FORCEINLINE VecType add(const VecType& a, const VecType& b) {
    VecType result;
#if defined(_MSC_VER) && !defined(__clang__) && defined(INTEL_INTRINSICS) && defined(__AVX__)
    if constexpr (sizeof(VecType) == 32) {
      // Force the use of YMM
      __m256 va = _mm256_loadu_ps(reinterpret_cast<const float*>(&a));
      __m256 vb = _mm256_loadu_ps(reinterpret_cast<const float*>(&b));
      __m256 vr = _mm256_add_ps(va, vb);
      _mm256_storeu_ps(reinterpret_cast<float*>(&result), vr);
      return result;
    }
#endif
    for (size_t i = 0; i < vector_traits<VecType>::size; ++i) {
      result[i] = a[i] + b[i];
    }
    return result;
  }

  static AVS_FORCEINLINE VecType subtract(const VecType& a, const VecType& b) {
    VecType result;
    for (size_t i = 0; i < vector_traits<VecType>::size; ++i) {
      result[i] = a[i] - b[i];
    }
    return result;
  }

  static AVS_FORCEINLINE VecType multiply(const VecType& a, const VecType& b) {
    VecType result;
    for (size_t i = 0; i < vector_traits<VecType>::size; ++i) {
      result[i] = a[i] * b[i];
    }
    return result;
  }

  static AVS_FORCEINLINE VecType scalar_multiply(const VecType& a, ScalarType scalar) {
    VecType result;
#if defined(_MSC_VER) && !defined(__clang__) && defined(INTEL_INTRINSICS) && defined(__AVX__)
    if constexpr (sizeof(VecType) == 32 && std::is_same_v<ScalarType, float>) {
      if constexpr (sizeof(VecType) == 32) {
        // Use 'set1' which the compiler usually optimizes to a broadcast
        __m256 v_a = _mm256_loadu_ps((const float*)&a);
        __m256 v_res = _mm256_mul_ps(v_a, _mm256_set1_ps(scalar));
        _mm256_storeu_ps((float*)&result, v_res);
        return result;
      }
    }
#endif
    for (size_t i = 0; i < vector_traits<VecType>::size; ++i) {
      result[i] = a[i] * scalar;
    }
    return result;
  }

  static AVS_FORCEINLINE VecType left_shift(const VecType& a, int shift) {
    VecType result;
    for (size_t i = 0; i < vector_traits<VecType>::size; ++i) {
      result[i] = a[i] << shift;
    }
    return result;
  }

  static AVS_FORCEINLINE VecType right_shift(const VecType& a, int shift) {
    VecType result;
    for (size_t i = 0; i < vector_traits<VecType>::size; ++i) {
      result[i] = a[i] >> shift;
    }
    return result;
  }
};
#endif

// Template for vector class wrapper
// Passing "Derived" allows syntax for assigning a wrapper class to a derived class.
template <typename Derived, typename VecType>
class VectorWrapper {
private:
  using ElementType = typename vector_traits<VecType>::element_type;
  static constexpr size_t N = vector_traits<VecType>::size;
  VecType data;

public:

  // Default constructor
  AVS_FORCEINLINE VectorWrapper() {
#if HAS_VECTOR_ATTRIBUTE
    data = VecType{};  // empty initializer
#else
    data.fill(ElementType{});
#endif
  }

  // Constructor from scalar
  AVS_FORCEINLINE explicit VectorWrapper(ElementType val) {
#if HAS_VECTOR_ATTRIBUTE
    data = VecType{};
    for (size_t i = 0; i < N; ++i) {
      data[i] = val;
    }
#else
#if defined(_MSC_VER) && !defined(__clang__) && defined(INTEL_INTRINSICS)
    if constexpr (sizeof(VecType) == 32 && std::is_same_v<ElementType, float>) {
#if defined(__AVX__)
      if (val == 0.0f) {
        __m256 zero = _mm256_setzero_ps();
        _mm256_storeu_ps((float*)&data, zero);
        return;
      }
      __m256 v = _mm256_set1_ps(val);
      _mm256_storeu_ps((float*)&data, v);
      return;
#else
      // __m128 path
      if (val == 0.0f) {
        __m128 zero = _mm_setzero_ps();
        _mm_storeu_ps((float*)&data, zero); // 2x128 = 32 bytes, so we can store 16 bytes at a time with
        _mm_storeu_ps((float*)((char*)&data + 16), zero);
        return;
      }
      __m128 v = _mm_set1_ps(val); // possibly optimized to a broadcast
      _mm_storeu_ps((float*)&data, v);
      _mm_storeu_ps((float*)((char*)&data + 16), v);
      return;
#endif
    } // 32 bytes
    if constexpr (sizeof(VecType) == 16 && std::is_same_v<ElementType, float>) {
      if (val == 0.0f) {
        __m128 zero = _mm_setzero_ps();
        _mm_storeu_ps((float*)&data, zero);
        return;
      }
      __m128 v = _mm_set1_ps(val); // possibly optimized to a broadcast
      _mm_storeu_ps((float*)&data, v);
      return;
    } // 16 bytes
#endif
    data.fill(val);
#endif
  }

  // Constructor from raw vector type
  AVS_FORCEINLINE explicit VectorWrapper(const VecType& vec) {
#if HAS_VECTOR_ATTRIBUTE
    data = vec;  // native vector assign, optimal
#else
    memcpy(&data, &vec, sizeof(data));  // fixed-size, MSVC inlines to movq/movdqu
#endif
  }
  // Conversion operator, allowing implicit conversion to the derived type.
  // Supporting syntax like: Int32x4 = src * coeff; where the right side is a VectorWrapper<Int32x4, int32_vec4_t>
  AVS_FORCEINLINE operator Derived() const {
    return static_cast<const Derived>(data);
  }

  // Copy constructor
  // Note: There is no need to define assignment operators for move or copy, as the underlying 'data' type 
  // (either array or vector attribute) already provides support for these operations.
  // 2026: but yes, we need! For MSVC optimizer's sake at least.
#ifdef HAS_VECTOR_ATTRIBUTE   
  AVS_FORCEINLINE VectorWrapper(const VectorWrapper& other) : data(other.data) {}
#else
  AVS_FORCEINLINE VectorWrapper(const VectorWrapper& other) {
    memcpy(&data, &other.data, sizeof(data));
  }
  AVS_FORCEINLINE VectorWrapper& operator=(const VectorWrapper& other) {
    memcpy(&data, &other.data, sizeof(data));
    return *this;
  }
#endif

  // Constructors from initializer list, e.g. x = {32768, 0, 0, 0};

  // Constexpr variadic template constructor for compile-time initialization.Thx AI :)
  template <typename... Args,
    typename = std::enable_if_t<sizeof...(Args) + 1 == N>>
    AVS_FORCEINLINE constexpr VectorWrapper(ElementType arg, Args... args) {
    static_assert(sizeof...(args) + 1 == N, "Initializer list must have exactly N elements.");
    ElementType values[] = { arg, static_cast<ElementType>(args)... };
    for (size_t i = 0; i < N; ++i) {
      data[i] = values[i];
    }
  }

  // std::initializer_list constructor for runtime initialization
  AVS_FORCEINLINE VectorWrapper(std::initializer_list<ElementType> values) {
    if (values.size() != N) {
      throw std::logic_error("Initializer list must have exactly N elements");
    }
    size_t i = 0;
    for (auto val : values) {
      data[i++] = val;
    }
  }

  // Load only the lower half of the vector (e.g., 8 bytes for a 16-byte vector)
  AVS_FORCEINLINE void load_lo(const ElementType* ptr) {
#if HAS_VECTOR_ATTRIBUTE
    for (size_t i = 0; i < N / 2; ++i) {
      data[i] = ptr[i];
    }
    for (size_t i = N / 2; i < N; ++i) {
      data[i] = ElementType{}; // Zero out the upper half
    }
#else
    memcpy(data.v, ptr, sizeof(data.v) / 2); // avoid std::copy!!!at least for MSVC
    memset(data.v + N / 2, 0, sizeof(data.v) / 2); // zero upper half
#endif
  }

  // Conversions from other types as long as their size matches
  // Full and half size

  // Full: Widening or narrowing without range checks, e.g., no saturation when converting down.
  // This one  must be used as src=Int32x4::convert_from(src16) and NOT as src32.convert_from(src16).
  template <typename OtherDerived, typename OtherVecType>
  AVS_FORCEINLINE static VectorWrapper convert_from(const VectorWrapper<OtherDerived, OtherVecType>& other) {
    static_assert(vector_traits<VecType>::size == vector_traits<OtherVecType>::size,
      "Vector sizes must match for convert_from.");
#if HAS_VECTOR_ATTRIBUTE
    VecType result_vec;
    result_vec = __builtin_convertvector(other.raw(), VecType);
    return VectorWrapper(result_vec);
#else
    VecType result;
    for (size_t i = 0; i < vector_traits<VecType>::size; ++i) {
      result[i] = static_cast<typename vector_traits<VecType>::element_type>(other[i]);
    }
    return VectorWrapper(result);
#endif

  }

  // Static factory: Widening or narrowing the lower half of a larger vector
  template <typename OtherDerived, typename OtherVecType>
  AVS_FORCEINLINE static VectorWrapper convert_from_lo(const VectorWrapper<OtherDerived, OtherVecType>& other) {
    static_assert(vector_traits<VecType>::size == vector_traits<OtherVecType>::size / 2,
      "Source vector must have twice the elements of target for convert_from_lo.");

    VectorWrapper result;
    for (size_t i = 0; i < vector_traits<VecType>::size; ++i) {
      // Access via operator[] which is public
      result.raw()[i] = static_cast<ElementType>(other[i]);
    }
    return result;
  }

  // Static factory: Widening or narrowing the upper half of a larger vector
  template <typename OtherDerived, typename OtherVecType>
  AVS_FORCEINLINE static VectorWrapper convert_from_hi(const VectorWrapper<OtherDerived, OtherVecType>& other) {
    static_assert(vector_traits<VecType>::size == vector_traits<OtherVecType>::size / 2,
      "Source vector must have twice the elements of target for convert_from_hi.");

    constexpr auto half_size = vector_traits<OtherVecType>::size / 2;
    VectorWrapper result;
    for (size_t i = 0; i < vector_traits<VecType>::size; ++i) {
      result.raw()[i] = static_cast<ElementType>(other[half_size + i]);
    }
    return result;
  }

  // Fill an integer vector/array from integer pointer source.
  // Versions for uint8_t, int16_t (short), uint16_t and int32_t

  AVS_FORCEINLINE void load_from_any_intptr(const uint8_t* ptr) {
    static_assert(
      std::is_same_v<typename vector_traits<VecType>::element_type, int32_t> ||
      std::is_same_v<typename vector_traits<VecType>::element_type, int16_t> ||
      std::is_same_v<typename vector_traits<VecType>::element_type, uint16_t> ||
      std::is_same_v<typename vector_traits<VecType>::element_type, uint8_t>,
      "load_from_any_intptr is only valid for int32_t, int16_t, uint16_t or uint8_t target element types."
      );

    for (size_t i = 0; i < vector_traits<VecType>::size; ++i) {
      data[i] = static_cast<typename vector_traits<VecType>::element_type>(ptr[i]);
    }
  }

  AVS_FORCEINLINE void load_from_any_intptr(const short* ptr) {
    static_assert(
      std::is_same_v<typename vector_traits<VecType>::element_type, int32_t> ||
      std::is_same_v<typename vector_traits<VecType>::element_type, int16_t> ||
      std::is_same_v<typename vector_traits<VecType>::element_type, uint16_t> ||
      std::is_same_v<typename vector_traits<VecType>::element_type, uint8_t>,
      "load_from_any_intptr is only valid for int32_t, int16_t, uint16_t or uint8_t target element types."
      );

    for (size_t i = 0; i < vector_traits<VecType>::size; ++i) {
      data[i] = static_cast<typename vector_traits<VecType>::element_type>(ptr[i]);
    }
  }

  AVS_FORCEINLINE void load_from_any_intptr(const uint16_t* ptr) {
    static_assert(
      std::is_same_v<typename vector_traits<VecType>::element_type, int32_t> ||
      std::is_same_v<typename vector_traits<VecType>::element_type, int16_t> ||
      std::is_same_v<typename vector_traits<VecType>::element_type, uint16_t> ||
      std::is_same_v<typename vector_traits<VecType>::element_type, uint8_t>,
      "load_from_any_intptr is only valid for int32_t, int16_t, uint16_t or uint8_t target element types."
      );

    for (size_t i = 0; i < vector_traits<VecType>::size; ++i) {
      data[i] = static_cast<typename vector_traits<VecType>::element_type>(ptr[i]);
    }
  }

  AVS_FORCEINLINE void load_from_any_intptr(const int32_t* ptr) {
    static_assert(
      std::is_same_v<typename vector_traits<VecType>::element_type, int32_t> ||
      std::is_same_v<typename vector_traits<VecType>::element_type, int16_t> ||
      std::is_same_v<typename vector_traits<VecType>::element_type, uint16_t> ||
      std::is_same_v<typename vector_traits<VecType>::element_type, uint8_t>,
      "load_from_any_intptr is only valid for int32_t, int16_t, uint16_t or uint8_t target element types."
      );

    for (size_t i = 0; i < vector_traits<VecType>::size; ++i) {
      data[i] = static_cast<typename vector_traits<VecType>::element_type>(ptr[i]);
    }
  }

  // Syntax: variable.load(myPointer);
  // Load from pointer (potentially unaligned)
  AVS_FORCEINLINE void load(const ElementType* ptr) {
#if HAS_VECTOR_ATTRIBUTE
    for (size_t i = 0; i < N; ++i) {
      data[i] = ptr[i];
    }
#else
    memcpy(data.v, ptr, sizeof(data.v)); // avoid std::copy!!! At least for MSVC
#endif
  }

  // Syntax: variable = VecType::from_ptr(myPointer);
  // Static factory method
  AVS_FORCEINLINE static VectorWrapper from_ptr(const ElementType* ptr) {
    VectorWrapper result;
#if HAS_VECTOR_ATTRIBUTE
    result.load(ptr);  // Reuse the instance method
#else
    constexpr size_t byte_size = sizeof(result.data);

    // Use _MSC_VER (one underscore) and check for actual size
#if defined(_MSC_VER) && !defined(__clang__) && defined(INTEL_INTRINSICS)
    if constexpr (byte_size == 4) {
      __m128i tmp = _mm_setzero_si128();
      memcpy(&tmp, ptr, 4); // Coax MSVC into MOVD
      memcpy(&result.data, &tmp, 4);
    }
    else if constexpr (byte_size == 8) {
      __m128i tmp = _mm_setzero_si128();
      memcpy(&tmp, ptr, 8); // Coax MSVC into MOVQ
      memcpy(&result.data, &tmp, 8);
    }
    else if constexpr (byte_size == 16) {
      __m128i tmp = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
      memcpy(&result.data, &tmp, 16);
    }
    else if constexpr (byte_size == 32) {
#if defined(__AVX__)
      // Use a single 256-bit load. This stops the XMM-to-stack-back-to-YMM MSVC habit.
      _mm256_storeu_ps((float*)&result.raw(), _mm256_loadu_ps(ptr));
#else
      __m128i tmp1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
      __m128i tmp1next16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr + 4));
      memcpy(&result.data, &tmp1, 16);
      memcpy(reinterpret_cast<char*>(&result.data) + 16, &tmp1next16, 16);
#endif
    }
    else {
      // Fallback for 32+ byte or unusual sizes
      memcpy(&result.data, ptr, byte_size);
    }
#else
    memcpy(&result.data, ptr, byte_size);
#endif
#endif
    return result;
  }

  // Syntax: variable = VecType::from_ptr_lo(myPointer);
  // Loads the lower half of the vector and zeros the upper half.
  AVS_FORCEINLINE static VectorWrapper from_ptr_lo(const ElementType* ptr) {
    VectorWrapper result;
#if HAS_VECTOR_ATTRIBUTE
    // Standard attribute-based load
    result.load_lo(ptr);
#else
    constexpr size_t total_bytes = sizeof(result.data);

#if defined(_MSC_VER) && !defined(__clang__) && defined(INTEL_INTRINSICS)
    if constexpr (total_bytes == 16) {
      // Int16x8 or Int32x4 case: Load 8 bytes, zero top 8.
      // _mm_loadl_epi64 generates MOVQ (XMM load 64-bit).
      __m128i tmp = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr));
      memcpy(&result.data, &tmp, 16);
    }
    else if constexpr (total_bytes == 32) {
#if defined(__AVX__)
      __m128i tmp = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr)); // Load 16 bytes into low half, zero top 16.
      __m256i tmp256 = _mm256_setzero_si256();
      tmp256 = _mm256_insertf128_si256(tmp256, tmp, 0);
      memcpy(&result.data, &tmp256, 32);
#else
      // 256-bit case: Load 16 bytes into low half, zero top 16.
      __m128i lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
      memcpy(&result.data, &lo, 16);
      // clear high 16 bytes
      memset(reinterpret_cast<char*>(&result.data) + 16, 0, 16);
#endif
    }
    else {
      result.load_lo(ptr);
    }
#else
    result.load_lo(ptr);
#endif
#endif
    return result;
  }

  // Store to pointer (potentially unaligned)
  AVS_FORCEINLINE void store(ElementType* ptr) const {
#if HAS_VECTOR_ATTRIBUTE
    for (size_t i = 0; i < N; ++i) {
      ptr[i] = data[i];
    }
#else
    memcpy(ptr, data.v, sizeof(data.v));  // with Avs_SimdArray; or &data[0] with std::array
#endif
  }

  // Addition operator (+)
  AVS_FORCEINLINE VectorWrapper operator+(const VectorWrapper& other) const {
#if HAS_VECTOR_ATTRIBUTE
    return VectorWrapper(data + other.data);
#else
    return VectorWrapper(VectorOps<VecType>::add(data, other.data));
#endif
  }

  // Subtraction operator (-)
  AVS_FORCEINLINE VectorWrapper operator-(const VectorWrapper& other) const {
#if HAS_VECTOR_ATTRIBUTE
    return VectorWrapper(data - other.data);
#else
    return VectorWrapper(VectorOps<VecType>::subtract(data, other.data));
#endif
  }

  // Multiplication operator (*)
  AVS_FORCEINLINE VectorWrapper operator*(const VectorWrapper& other) const {
#if HAS_VECTOR_ATTRIBUTE
    return VectorWrapper(data * other.data);
#else
    return VectorWrapper(VectorOps<VecType>::multiply(data, other.data));
#endif
  }

  // Scalar multiplication (*val)
  AVS_FORCEINLINE VectorWrapper operator*(ElementType scalar) const {
#if HAS_VECTOR_ATTRIBUTE
    return VectorWrapper(data * scalar);
#else
    return VectorWrapper(VectorOps<VecType>::scalar_multiply(data, scalar));
#endif
  }

  // Left shift operator (<<)
  VectorWrapper operator<<(int shift) const {
#if HAS_VECTOR_ATTRIBUTE
    return VectorWrapper(data << shift);
#else
    return VectorWrapper(VectorOps<VecType>::left_shift(data, shift));
#endif
  }

  // Right shift operator (arithmetic) (>>)
  AVS_FORCEINLINE VectorWrapper operator>>(int shift) const {
#if HAS_VECTOR_ATTRIBUTE
    return VectorWrapper(data >> shift);
#else
    return VectorWrapper(VectorOps<VecType>::right_shift(data, shift));
#endif
  }

  // We do not define special operators for logical shift, but provide a function.
  AVS_FORCEINLINE VectorWrapper shift_right_logical(int shift) const {
    // Cast to unsigned type for logical shift
    using UnsignedVecType = std::make_unsigned_t<typename vector_traits<VecType>::element_type>;
    VecType result;
    for (size_t i = 0; i < vector_traits<VecType>::size; ++i) {
      result[i] = static_cast<UnsignedVecType>(data[i]) >> shift;
    }
    return VectorWrapper(result);
  }

  // Addition assignment operator (+=)
  AVS_FORCEINLINE VectorWrapper& operator+=(const VectorWrapper& other) {
#if HAS_VECTOR_ATTRIBUTE
    data += other.data;
#else
    data = VectorOps<VecType>::add(data, other.data);
#endif
    return *this;
  }

  // Subtract assignment operator (-=)
  AVS_FORCEINLINE VectorWrapper& operator-=(const VectorWrapper& other) {
#if HAS_VECTOR_ATTRIBUTE
    data -= other.data;
#else
    data = VectorOps<VecType>::sub(data, other.data);
#endif
    return *this;
  }

  // Multiplication assignment operator (*=) 
  AVS_FORCEINLINE VectorWrapper& operator*=(const VectorWrapper& other) {
#if HAS_VECTOR_ATTRIBUTE
    data *= other.data;
#else
    data = VectorOps<VecType>::mul(data, other.data);
#endif
    return *this;
  }

  // right shift assignment operator (>>=)
  AVS_FORCEINLINE VectorWrapper& operator>>=(int shift) {
#if HAS_VECTOR_ATTRIBUTE
    data >>= shift;
#else
    data = VectorOps<VecType>::right_shift(data, shift);
#endif
    return *this;
  }

  // left shift assignment operator (<<=)
  AVS_FORCEINLINE VectorWrapper& operator<<=(int shift) {
#if HAS_VECTOR_ATTRIBUTE
    data <<= shift;
#else
    data = VectorOps<VecType>::left_shift(data, shift);
#endif
    return *this;
  }

  // Minimum operation: z = x.min(scalar)
  AVS_FORCEINLINE VectorWrapper min(ElementType scalar) const {
#if HAS_VECTOR_ATTRIBUTE
    VecType result;
    for (size_t i = 0; i < N; ++i) {
      result[i] = data[i] < scalar ? data[i] : scalar;
    }
    return VectorWrapper(result);
#else
    VecType result;
    for (size_t i = 0; i < N; ++i) {
      result[i] = std::min(data[i], scalar);
    }
    return VectorWrapper(result);
#endif
  }

  // Maximum operation: z = x.max(scalar)
  AVS_FORCEINLINE VectorWrapper max(ElementType scalar) const {
#if HAS_VECTOR_ATTRIBUTE
    VecType result;
    for (size_t i = 0; i < N; ++i) {
      result[i] = data[i] > scalar ? data[i] : scalar;
    }
    return VectorWrapper(result);
#else
    VecType result;
    for (size_t i = 0; i < N; ++i) {
      result[i] = std::max(data[i], scalar);
    }
    return VectorWrapper(result);
#endif
  }

  // Minimum operation (element-wise): z = x.min(y)
  AVS_FORCEINLINE VectorWrapper min(const VectorWrapper& other) const {
#if HAS_VECTOR_ATTRIBUTE
    // Use a ternary operator with a comparison to simulate min operation
    VecType result;
    for (size_t i = 0; i < N; ++i) {
      result[i] = data[i] < other.data[i] ? data[i] : other.data[i];
    }
    return VectorWrapper(result);
#else
    VecType result;
    for (size_t i = 0; i < N; ++i) {
      result[i] = std::min(data[i], other.data[i]);
    }
    return VectorWrapper(result);
#endif
  }

  // Maximum operation (element-wise) : z = x.max(y)
  AVS_FORCEINLINE VectorWrapper max(const VectorWrapper& other) const {
#if HAS_VECTOR_ATTRIBUTE
    // Use a ternary operator with a comparison to simulate max operation
    VecType result;
    for (size_t i = 0; i < N; ++i) {
      result[i] = data[i] > other.data[i] ? data[i] : other.data[i];
    }
    return VectorWrapper(result);
#else
    VecType result;
    for (size_t i = 0; i < N; ++i) {
      result[i] = std::max(data[i], other.data[i]);
    }
    return VectorWrapper(result);
#endif
  }

  // Static min functions similar to SIMD intrinsics: z = min(x, y)
  AVS_FORCEINLINE static VectorWrapper min(const VectorWrapper& a, const VectorWrapper& b) {
    return a.min(b);
  }

  // Static max function similar to SIMD intrinsics: z = max(x, y)
  AVS_FORCEINLINE static VectorWrapper max(const VectorWrapper& a, const VectorWrapper& b) {
    return a.max(b);
  }

  // Accessor for raw data
  AVS_FORCEINLINE const VecType& raw() const { return data; }
  AVS_FORCEINLINE VecType& raw() { return data; }

  // Element access (const and non-const versions)
  AVS_FORCEINLINE ElementType operator[](size_t idx) const {
    return data[idx];
  }

  // a set method instead of non-const operator[]
  // use: x.set(index, value) instead of x[index] = value
  AVS_FORCEINLINE void set(size_t idx, ElementType value) {
    data[idx] = value;
  }

  AVS_FORCEINLINE int32_t horiz_add_int32() const {
#if defined(_MSC_VER) && !defined(__clang__) && defined(INTEL_INTRINSICS)
    static_assert(N * sizeof(ElementType) <= 16, "horiz_add_int32: vector too large for XMM");
    __m128i v = _mm_setzero_si128();  // zero upper lanes for sub-16-byte vectors
    memcpy(&v, &data, N * sizeof(ElementType));

    // SSE2 only, hadd simulation with shuffles and adds
    if constexpr (N * sizeof(ElementType) <= 4) {
      return _mm_cvtsi128_si32(v);
    }
    else if constexpr (N * sizeof(ElementType) <= 8) {
      __m128i hi32 = _mm_shuffle_epi32(v, _MM_SHUFFLE(1, 1, 1, 1));
      return _mm_cvtsi128_si32(_mm_add_epi32(v, hi32));
    }
    else {
      // 4x int32
      // v = [D, C, B, A]
      __m128i hi64 = _mm_unpackhi_epi64(v, v); // [B, A, D, C]
      v = _mm_add_epi32(v, hi64); // [D+B, C+A, B+D, A+C]
      __m128i hi32 = _mm_shuffle_epi32(v, _MM_SHUFFLE(1, 1, 1, 1)); // [C+A, C+A, C+A, C+A]
      v = _mm_add_epi32(v, hi32); // [..., ..., ..., A+C+B+D]
      return _mm_cvtsi128_si32(v);
    }
#else
    int32_t sum = 0;
    for (size_t i = 0; i < N; ++i)
#if HAS_VECTOR_ATTRIBUTE
      sum += data[i];
#else
      sum += data.v[i];
#endif
    return sum;
#endif
  }

  AVS_FORCEINLINE float horiz_add_float() const {
#if defined(_MSC_VER) && !defined(__clang__) && defined(INTEL_INTRINSICS)
    static_assert(N * sizeof(ElementType) <= 16, "horiz_add_float: vector too large for XMM");
    __m128 v = _mm_setzero_ps();
    memcpy(&v, &data, N * sizeof(ElementType));
    // SSE2 only, not haddps, only quick shuffles and adds
    if constexpr (N <= 1) {
      return _mm_cvtss_f32(v);
    }
    else if constexpr (N <= 2) {
      __m128 hi32 = _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1));
      return _mm_cvtss_f32(_mm_add_ss(v, hi32));
    }
    else {
      // v = [D, C, B, A]
      __m128 hi64 = _mm_movehl_ps(v, v); // [B, A, D, C]
      v = _mm_add_ps(v, hi64); // [D+B, C+A, B+D, A+C]
      __m128 hi32 = _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1)); // [C+A, C+A, C+A, C+A]
      v = _mm_add_ss(v, hi32); // [..., ..., ..., A+C+B+D]
      return _mm_cvtss_f32(v);
    }
#else
    float sum = 0.0f;
    for (size_t i = 0; i < N; ++i)
#if HAS_VECTOR_ATTRIBUTE
      sum += static_cast<float>(data[i]);
#else
      sum += static_cast<float>(data.v[i]);
#endif
    return sum;
#endif
  }

  static constexpr size_t size() { return N; }
};

// Scalar multiplication (scalar on left)
template <typename Derived, typename VecType>
AVS_FORCEINLINE VectorWrapper<Derived, VecType> operator*(
  typename vector_traits<VecType>::element_type scalar,
  const VectorWrapper<Derived, VecType>& vec) {
  return vec * scalar;
}

// Stream output operator
template <typename Derived, typename VecType>
std::ostream& operator<<(std::ostream& os, const VectorWrapper<Derived, VecType>& vec) {
  os << "[";
  for (size_t i = 0; i < vector_traits<VecType>::size; ++i) {
    os << vec[i];
    if (i < vector_traits<VecType>::size - 1) os << ", ";
  }
  os << "]";
  return os;
}

// Concrete class definitions for 8, 16 and 32 bytes vectors
class Uint8x32 : public VectorWrapper<Uint8x32, uint8_vec32_t> {
public:
  using Base = VectorWrapper<Uint8x32, uint8_vec32_t>;
  using Base::Base;
  using Base::operator=;
};

class Uint8x16 : public VectorWrapper<Uint8x16, uint8_vec16_t> {
public:
  using Base = VectorWrapper<Uint8x16, uint8_vec16_t>;
  using Base::Base;
  using Base::operator=;
};

class Uint8x8 : public VectorWrapper<Uint8x8, uint8_vec8_t> {
public:
  using Base = VectorWrapper<Uint8x8, uint8_vec8_t>;
  using Base::Base;
  using Base::operator=;
};

class Uint8x4 : public VectorWrapper<Uint8x4, uint8_vec4_t> {
public:
  using Base = VectorWrapper<Uint8x4, uint8_vec4_t>;
  using Base::Base;
  using Base::operator=;
};

class Int16x16 : public VectorWrapper<Int16x16, int16_vec16_t> {
public:
  using Base = VectorWrapper<Int16x16, int16_vec16_t>;
  using Base::Base;
  using Base::operator=;
};

class Int16x8 : public VectorWrapper<Int16x8, int16_vec8_t> {
public:
  using Base = VectorWrapper<Int16x8, int16_vec8_t>;
  using Base::Base;
  using Base::operator=;
};

class Int16x4 : public VectorWrapper<Int16x4, int16_vec4_t> {
public:
  using Base = VectorWrapper<Int16x4, int16_vec4_t>;
  using Base::Base;
  using Base::operator=;
};

class Uint16x16 : public VectorWrapper<Uint16x16, uint16_vec16_t> {
public:
  using Base = VectorWrapper<Uint16x16, uint16_vec16_t>;
  using Base::Base;
  using Base::operator=;
};

class Uint16x8 : public VectorWrapper<Uint16x8, uint16_vec8_t> {
public:
  using Base = VectorWrapper<Uint16x8, uint16_vec8_t>;
  using Base::Base;
  using Base::operator=;
};

class Uint16x4 : public VectorWrapper<Uint16x4, uint16_vec4_t> {
public:
  using Base = VectorWrapper<Uint16x4, uint16_vec4_t>;
  using Base::Base;
  using Base::operator=;
};

class Int32x8 : public VectorWrapper<Int32x8, int32_vec8_t> {
public:
  using Base = VectorWrapper<Int32x8, int32_vec8_t>;
  using Base::Base;
  using Base::operator=;
};

class Int32x4 : public VectorWrapper<Int32x4, int32_vec4_t> {
public:
  using Base = VectorWrapper<Int32x4, int32_vec4_t>;
  using Base::Base;
  using Base::operator=;
};

class Int32x2 : public VectorWrapper<Int32x2, int32_vec2_t> {
public:
  using Base = VectorWrapper<Int32x2, int32_vec2_t>;
  using Base::Base;
  using Base::operator=;
};

class Float8 : public VectorWrapper<Float8, float_vec8_t> {
public:
  using Base = VectorWrapper<Float8, float_vec8_t>;
  using Base::Base;
  using Base::operator=;
};

class Float4 : public VectorWrapper<Float4, float_vec4_t> {
public:
  using Base = VectorWrapper<Float4, float_vec4_t>;
  using Base::Base;
  using Base::operator=;
};

class Float2 : public VectorWrapper<Float2, float_vec2_t> {
public:
  using Base = VectorWrapper<Float2, float_vec2_t>;
  using Base::Base;
  using Base::operator=;
};

// Saturated narrowing integer conversion functions.

// 8x int32_t -> 8x uint8_t
// Guaranteed to stay in XMM/YMM registers
[[maybe_unused]] static AVS_FORCEINLINE void convert_and_saturate_int32x8_to_uint8x8(const Int32x8& src1, Uint8x8& result) {
#if defined(_MSC_VER) && !defined(__clang__) && defined(INTEL_INTRINSICS)
  // Assuming Int32x8 is composed of two __m128i (low/high) or one __m256i
  // If using 128-bit wrappers:
  __m128i src1_lo;
  __m128i src1_hi;
  memcpy(&src1_lo, &src1, 16); // first 4 ints
  memcpy(&src1_hi, reinterpret_cast<const char*>(&src1) + 16, 16); // second 4 ints

  // Step 1: Pack 32-bit to 16-bit (Signed Saturation)
  __m128i v16 = _mm_packs_epi32(src1_lo, src1_hi);

  // Step 2: Pack 16-bit to 8-bit (Unsigned Saturation)
  // We use v16 twice because pack instructions always process 128-bits into 128-bits
  __m128i v8 = _mm_packus_epi16(v16, v16);

  // Step 3: Store only the low 8 bytes (64-bits)
  _mm_storel_epi64(reinterpret_cast<__m128i*>(&result), v8);
#else
  // Fallback for LLVM (which handles the loop perfectly)
  for (size_t i = 0; i < 8; ++i) {
    result.set(i, static_cast<uint8_t>(std::min(std::max(src1[i], 0), 255)));
  }
#endif
}


// 4x int32_t -> 4x uint8_t
// Prevents the "Extract-to-GPR" penalty seen in your assembly
[[maybe_unused]] static AVS_FORCEINLINE void convert_and_saturate_int32x4_to_uint8x4(const Int32x4& src1, Uint8x4& result) {
#if defined(_MSC_VER) && !defined(__clang__) && defined(INTEL_INTRINSICS)
  __m128i a;
  memcpy(&a, &src1, 16);

  // Step 1: 32-bit signed -> 16-bit signed saturation
  // [i32, i32, i32, i32] -> [i16, i16, i16, i16, i16, i16, i16, i16]
  // We duplicate 'a' to fill the 128-bit output
  __m128i v16 = _mm_packs_epi32(a, a);

  // Step 2: 16-bit signed -> 8-bit unsigned saturation
  // This clamps everything to [0, 255]
  __m128i v8 = _mm_packus_epi16(v16, v16);

  // Step 3: Store only the low 32 bits (4 bytes)
  // This replaces those 4 'mov byte ptr' instructions with one 'movd'
  int packed_pixels = _mm_cvtsi128_si32(v8);
  memcpy(&result, &packed_pixels, 4);
#else
  // LLVM/Clang handles the loop fine, as you noted
  for (size_t i = 0; i < 4; ++i) {
    result.set(i, static_cast<uint8_t>(std::min(std::max(src1[i], 0), 255)));
  }
#endif
}

// 8x int32_t -> 8x uint16_t
[[maybe_unused]] static AVS_FORCEINLINE void convert_and_saturate_int32x8_to_uint16x8(const Int32x8& src1, Uint16x8& result) {
  for (size_t i = 0; i < 8; ++i) {
    result.set(i, static_cast<uint16_t>(std::min(std::max(src1[i], 0), 65535)));
  }
}


// 4x int32_t -> 4x uint16_t with 0 <= x <= limit
[[maybe_unused]] static AVS_FORCEINLINE void convert_and_saturate_int32x4_to_uint16x4_limit(const Int32x4& src1, Uint16x4& result, const uint16_t limit) {
#if defined(_MSC_VER) && !defined(__clang__) && defined(INTEL_INTRINSICS)
  __m128i a;
  memcpy(&a, &src1, 16);

#if defined(__AVX__) || defined(__SSE4_1__)
  __m128i packed = _mm_packus_epi32(a, a);
  __m128i vlimit = _mm_set1_epi16(static_cast<short>(limit));
  __m128i clamped = _mm_min_epu16(packed, vlimit);
  _mm_storel_epi64(reinterpret_cast<__m128i*>(&result), clamped);
#else
  // SSE2 fallback, any limit: bias into the signed range, signed pack (= packus - 32768),
  // signed min against the biased limit (order preserving = unsigned min), unbias.
  // Edge case: x < INT_MIN + 32768 wraps in the 32 bit bias subtract and gives limit instead of 0
  // (the C path gives 0). Not reachable from pixel arithmetic (resizer sums stay far from INT_MIN).
  const __m128i bias = _mm_set1_epi32(32768);
  const __m128i bias16 = _mm_set1_epi16(-32768);
  __m128i biased = _mm_sub_epi32(a, bias);
  __m128i packed = _mm_packs_epi32(biased, biased);
  __m128i vlimit_biased = _mm_set1_epi16(static_cast<short>(limit - 32768));
  __m128i clamped = _mm_sub_epi16(_mm_min_epi16(packed, vlimit_biased), bias16);
  _mm_storel_epi64(reinterpret_cast<__m128i*>(&result), clamped);
#endif

#else
  for (size_t i = 0; i < 4; ++i) {
    result.set(i, static_cast<uint16_t>(std::min(std::max(src1[i], 0), (int)limit)));
  }
#endif
}


// 4x int32_t -> 4x uint16_t
[[maybe_unused]] static AVS_FORCEINLINE void convert_and_saturate_int32x4_to_uint16x4(const Int32x4& src1, Uint16x4& result) {
#if defined(_MSC_VER) && !defined(__clang__) && defined(INTEL_INTRINSICS)
  __m128i a;
  memcpy(&a, &src1, 16);

#if defined(__AVX__) || defined(__SSE4_1__)
  // Packs 4+4 lanes into 8 lanes of uint16. We use 'a' twice.
  __m128i packed = _mm_packus_epi32(a, a);
  // Store only the low 64 bits (4x uint16)
  _mm_storel_epi64(reinterpret_cast<__m128i*>(&result), packed);
#else
  // SSE2 Fallback: The Bias Trick
  // Edge case: x < INT_MIN + 32768 wraps in the 32 bit bias subtract and gives 65535 instead of 0
  // (the C path gives 0). Not reachable from pixel arithmetic (resizer sums stay far from INT_MIN).
  const __m128i bias = _mm_set1_epi32(32768);
  const __m128i bias16 = _mm_set1_epi16(-32768);
  __m128i biased = _mm_sub_epi32(a, bias);
  __m128i packed = _mm_packs_epi32(biased, biased);
  __m128i final = _mm_sub_epi16(packed, bias16);
  _mm_storel_epi64(reinterpret_cast<__m128i*>(&result), final);
#endif

#else
  for (size_t i = 0; i < 4; ++i) {
    result.set(i, static_cast<uint16_t>(std::min(std::max(src1[i], 0), 65535)));
  }
#endif
}

// Combined horizontal add of 4 Int32x4 vectors into one Int32x4 of sums
[[maybe_unused]]
AVS_FORCEINLINE Int32x4 make_from_horiz_sums(
  const Int32x4& r0, const Int32x4& r1,
  const Int32x4& r2, const Int32x4& r3) {
#if defined(_MSC_VER) && !defined(__clang__) && defined(INTEL_INTRINSICS)
  __m128i a, b, c, d;
  memcpy(&a, &r0, 16); memcpy(&b, &r1, 16);
  memcpy(&c, &r2, 16); memcpy(&d, &r3, 16);

  // 1. Interleave to pair up elements (SSE2)
  __m128i t0 = _mm_unpacklo_epi32(a, b); // [a0, b0, a1, b1]
  __m128i t1 = _mm_unpackhi_epi32(a, b); // [a2, b2, a3, b3]
  __m128i t2 = _mm_unpacklo_epi32(c, d); // [c0, d0, c1, d1]
  __m128i t3 = _mm_unpackhi_epi32(c, d); // [c2, d2, c3, d3]

  // 2. First vertical add (Partial sums)
  __m128i sum_ab = _mm_add_epi32(t0, t1); // [a0+a2, b0+b2, a1+a3, b1+b3]
  __m128i sum_cd = _mm_add_epi32(t2, t3); // [c0+c2, d0+d2, c1+c3, d1+d3]

  // 3. Final pairing using 64-bit unpacks (SSE2)
  __m128i res_lo = _mm_unpacklo_epi64(sum_ab, sum_cd); // [a0+a2, b0+b2, c0+c2, d0+d2]
  __m128i res_hi = _mm_unpackhi_epi64(sum_ab, sum_cd); // [a1+a3, b1+b3, c1+c3, d1+d3]

  // 4. Final vertical add
  __m128i res = _mm_add_epi32(res_lo, res_hi); // [sumA, sumB, sumC, sumD]

  Int32x4 out;
  memcpy(&out, &res, 16);
  return out;
#else
  Int32x4 out;
  out.set(0, r0.horiz_add_int32());
  out.set(1, r1.horiz_add_int32());
  out.set(2, r2.horiz_add_int32());
  out.set(3, r3.horiz_add_int32());
  return out;
#endif
}

// two 4x int32_t -> 8x uint16_t
// like an _mm_packus_epi32
[[maybe_unused]]
static AVS_FORCEINLINE void convert_and_saturate_int32x4x2_to_uint16x8(const Int32x4& src1, const Int32x4& src2, Uint16x8& result) {
#if defined(_MSC_VER) && !defined(__clang__) && defined(INTEL_INTRINSICS)
  __m128i a, b;
  memcpy(&a, &src1, 16);
  memcpy(&b, &src2, 16);
  // Though MSVC does not have sse4.1 arch granularity
#if defined(__AVX__) || defined(__SSE4_1__)
  // SSE4.1+: direct unsigned pack
  __m128i packed = _mm_packus_epi32(a, b);
  memcpy(&result, &packed, 16);
#else
  // SSE2 fallback: bias trick
  // _mm_packs_epi32 does signed saturation, so bias into signed range first
  // subtract 32768, pack as signed, then the result is offset by 32768
  // Edge case: x < INT_MIN + 32768 wraps in the 32 bit bias subtract and gives 65535 instead of 0
  // (the C path gives 0). Not reachable from pixel arithmetic (resizer sums stay far from INT_MIN).
  const __m128i bias = _mm_set1_epi32(32768);
  const __m128i bias16 = _mm_set1_epi16(-32768); // same bit pattern, adds back
  __m128i a_biased = _mm_sub_epi32(a, bias);
  __m128i b_biased = _mm_sub_epi32(b, bias);
  __m128i packed = _mm_packs_epi32(a_biased, b_biased); // signed sat to int16
  __m128i unbiased = _mm_sub_epi16(packed, bias16);       // restore: wraps correctly
  memcpy(&result, &unbiased, 16);
#endif
#else
  for (size_t i = 0; i < 4; ++i) {
    result.set(i, static_cast<uint16_t>(std::min(std::max(src1[i], 0), 65535)));
    result.set(i + 4, static_cast<uint16_t>(std::min(std::max(src2[i], 0), 65535)));
  }
#endif
}

// two 4x int32_t -> 8x uint16_t
// like a _mm_packus_epi32 followed by a min(x,limit)
[[maybe_unused]]
static AVS_FORCEINLINE void convert_and_saturate_int32x4x2_to_uint16x8_limit(const Int32x4& src1, const Int32x4& src2, Uint16x8& result, const uint16_t limit) {
#if defined(_MSC_VER) && !defined(__clang__) && defined(INTEL_INTRINSICS)
  __m128i a, b;
  memcpy(&a, &src1, 16);
  memcpy(&b, &src2, 16);
#if defined(__AVX__) || defined(__SSE4_1__)
  __m128i packed = _mm_packus_epi32(a, b);
  __m128i vlimit = _mm_set1_epi16(static_cast<short>(limit));
  __m128i clamped = _mm_min_epu16(packed, vlimit);
  memcpy(&result, &clamped, 16);
#else
  // SSE2 fallback, any limit: bias into the signed range, signed pack (= packus - 32768),
  // signed min against the biased limit (order preserving = unsigned min), unbias.
  // Edge case: x < INT_MIN + 32768 wraps in the 32 bit bias subtract and gives limit instead of 0
  // (the C path gives 0). Not reachable from pixel arithmetic (resizer sums stay far from INT_MIN).
  const __m128i bias = _mm_set1_epi32(32768);
  const __m128i bias16 = _mm_set1_epi16(-32768);
  __m128i packed = _mm_packs_epi32(_mm_sub_epi32(a, bias), _mm_sub_epi32(b, bias));
  __m128i vlimit_biased = _mm_set1_epi16(static_cast<short>(limit - 32768));
  __m128i clamped = _mm_sub_epi16(_mm_min_epi16(packed, vlimit_biased), bias16);
  memcpy(&result, &clamped, 16);
#endif
#else
  for (size_t i = 0; i < 4; ++i) {
    result.set(i, static_cast<uint16_t>(std::min(std::max(src1[i], 0), (int)limit)));
    result.set(i + 4, static_cast<uint16_t>(std::min(std::max(src2[i], 0), (int)limit)));
  }
#endif
}

// two 4x int32_t -> 8x uint8_t
// like an _mm_packs_epi32 followed by _mm_packus_epi16
[[maybe_unused]]
static AVS_FORCEINLINE void convert_and_saturate_int32x4x2_to_uint8x8(const Int32x4& src1, const Int32x4& src2, Uint8x8& result) {
#if defined(_MSC_VER) && !defined(__clang__) && defined(INTEL_INTRINSICS)
  __m128i a, b;
  memcpy(&a, &src1, 16);
  memcpy(&b, &src2, 16);
  __m128i packed16 = _mm_packs_epi32(a, b);        // [s1_0..s1_3, s2_0..s2_3] as int16 with signed sat
  __m128i packed8 = _mm_packus_epi16(packed16, packed16); // uint8 with unsigned sat, result in low 8 bytes
  memcpy(&result, &packed8, 8);                    // movq equivalent
#else
  for (size_t i = 0; i < 4; ++i) {
    result.set(i, static_cast<uint8_t>(std::min(std::max(src1[i], 0), 255)));
    result.set(i + 4, static_cast<uint8_t>(std::min(std::max(src2[i], 0), 255)));
  }
  // This one is optimized with LLVM, but not with gcc or MSVC
  /* bravo LLVM x86-64 icx 2025.1.0, using return Int32x4

        movdqu  xmm0, xmmword ptr [rdi]
        packssdw        xmm0, xmm0
        packuswb        xmm0, xmm0
        movdqu  xmm1, xmmword ptr [rsi + 16]
        packssdw        xmm1, xmm1
        packuswb        xmm1, xmm1
        movd    eax, xmm0
        movd    ecx, xmm1
        shl     rcx, 32
        or      rax, rcx
        ret
    */
#endif
}

// int16 * int16->int32 with partial result addition.
// Simulates the _mm_madd_epi16 x86 SIMD intrinsic function.
// returns [a0*b0 + a1*b1, a2*b2 + a3*b3, a4*b4 + a5*b5, a6*b6 + a7*b7]
// When the pre-addition order is not important use the reduce_add version,
// which may be easier to optimize on non-x86 architectures.
// (only LLVM was able to optimize the x86 version and compile into real madd)
[[maybe_unused]]
static AVS_FORCEINLINE Int32x4 simul_madd_epi16(const Int16x8& a, const Int16x8& b) {
  Int32x4 result;

  // pairwise multiplication and horizontal addition
  // Yes, I know c++ integer promotion rules, but maybe the compiler
  // will see our forced static_casting the already short type.
  // Otherwise, yes, it makes no sense.
  for (size_t i = 0; i < 4; ++i) {
    size_t idx = i * 2;
    int32_t sum =
      static_cast<short>(a[idx]) * static_cast<short>(b[idx]) +
      static_cast<short>(a[idx + 1]) * static_cast<short>(b[idx + 1]);

    result.set(i, sum);
  }
  return result;
}


// int16 * int16 -> int32 with partial result addition.
// Unlike the classic x86 SIMD madd_epi16, the sum of multiplications 
// is not horizontally adjacent, instead, the lanes are added together.
// returns [a0*b0 + a4*b4, a1*b1 + a5*b5, a2*b2 + a6*b6, a3*b3 + a7*b7]
// However, since the order is not always important, this approach is
// beneficial for non-x64 processor architectures when the compiler
// recognizes the widening multiplication pattern.
[[maybe_unused]]
static AVS_FORCEINLINE Int32x4 mul16x16_reduce_to_Int32x4(const Int16x8 &a, const Int16x8 &b) {
  Int32x4 result_lo, result_hi;

  for (size_t i = 0; i < 4; ++i) {
    result_lo.set(i, static_cast<short>(a[i]) * static_cast<short>(b[i]));
  }
  for (size_t i = 0; i < 4; ++i) {
    result_hi.set(i, static_cast<short>(a[i + 4]) * static_cast<short>(b[i + 4]));
  }
  Int32x4 result = result_lo + result_hi;
  return result;
}

[[maybe_unused]]
static AVS_FORCEINLINE Int32x4 mul16x16_to_Int32x4(const Int16x4& a, const Int16x4& b) {
  Int32x4 result;

  for (size_t i = 0; i < 4; ++i) {
    result.set(i, static_cast<short>(a[i]) * static_cast<short>(b[i]));
  }
  return result;
}

} // namespace

#endif  // __AVS_SIMD_C_H__
