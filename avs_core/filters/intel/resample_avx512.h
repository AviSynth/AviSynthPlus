// Avisynth v2.5.  Copyright 2002 Ben Rudiak-Gould et al.
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

#ifndef __Resample_AVX512_H__
#define __Resample_AVX512_H__

#include <avisynth.h>
#include "../resample_functions.h"

#include <immintrin.h> // includes AVX, AVX2, FMA3, AVX512F, AVX512BW, etc. for MSVC, Clang, and GCC

// compiler feature checks and error handling
#if defined(__clang__) && !defined(_MSC_VER)
#if !defined(__AVX512F__) || !defined(__AVX512BW__)
#error "This code requires a compiler that supports AVX-512F and AVX-512BW.  Use compiler flags -mavx512f -mavx512bw."
#endif
#elif defined(__GNUC__)
#if !defined(__AVX512F__) || !defined(__AVX512BW__)
#error "This code requires a compiler that supports AVX-512F and AVX-512BW.  Use compiler flags -mavx512f -mavx512bw."
#endif
#elif defined(_MSC_VER)
  #if !defined(_M_X64) && !defined(_M_AMD64) && !defined(_M_ARM64)
  #error "AVX-512 is only supported on x64 and ARM64 architectures."
  #endif
  // MSVC's <immintrin.h> provides AVX-512 support when /arch:AVX512 is used.
  // However, MSVC may not define __AVX512F__ or __AVX512BW__ consistently.
  // We rely on /arch:AVX512 having been set, and assume that if the user is
  // including this header, they intend to use AVX-512.
#else
  #error "Unsupported compiler. This code requires a compiler that supports AVX-512F and AVX-512BW (GCC, Clang, or MSVC)."
#endif

#if !defined(__FMA__)
// Assume that all processors that have AVX2/AVX512 also have FMA3
#if defined (__GNUC__) && ! defined (__INTEL_COMPILER) && ! defined (__clang__)
// Prevent error message in g++ when using FMA intrinsics with avx2:
#pragma message "It is recommended to specify also option -mfma when using -mavx2 or higher"
#else
#define __FMA__  1
#endif
#endif
// FMA3 instruction set
#if defined (__FMA__) && (defined(__GNUC__) || defined(__clang__))  && ! defined (__INTEL_COMPILER)
#include <fmaintrin.h>
#endif // __FMA__

// MSVC Missing Intrinsics (Workaround for older MSVC versions)
#if defined(_MSC_VER) && !defined(__clang__)
#if _MSC_VER < 1922 // Check for MSVC version less than 16.2 (VS 2019 16.2)
  // Define missing AVX-512BW mask intrinsics for older MSVC.
  // inline functions that perform the mask operations directly.
  // Since this is MSVC only, using specific __forceinline.
__forceinline __mmask64 _kand_mask64(__mmask64 a, __mmask64 b) { return a & b; }
__forceinline __mmask64 _kor_mask64(__mmask64 a, __mmask64 b) { return a | b; }
__forceinline __mmask32 _kand_mask32(__mmask32 a, __mmask32 b) { return a & b; }
__forceinline __mmask32 _kor_mask32(__mmask32 a, __mmask32 b) { return a | b; }
#endif
#endif

// useful macros

#define _MM_TRANSPOSE16_LANE4_PS(row0, row1, row2, row3) \
  do { \
    __m512 __t0, __t1, __t2, __t3; \
    __t0 = _mm512_unpacklo_ps(row0, row1); \
    __t1 = _mm512_unpackhi_ps(row0, row1); \
    __t2 = _mm512_unpacklo_ps(row2, row3); \
    __t3 = _mm512_unpackhi_ps(row2, row3); \
    row0 = _mm512_shuffle_ps(__t0, __t2, _MM_SHUFFLE(1, 0, 1, 0)); \
    row1 = _mm512_shuffle_ps(__t0, __t2, _MM_SHUFFLE(3, 2, 3, 2)); \
    row2 = _mm512_shuffle_ps(__t1, __t3, _MM_SHUFFLE(1, 0, 1, 0)); \
    row3 = _mm512_shuffle_ps(__t1, __t3, _MM_SHUFFLE(3, 2, 3, 2)); \
  } while (0)

#ifndef _mm512_loadu_4_m128
#define _mm512_loadu_4_m128(/* __m128 const* */ addr1, \
                            /* __m128 const* */ addr2, \
                            /* __m128 const* */ addr3, \
                            /* __m128 const* */ addr4) \
_mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(_mm512_castps128_ps512(_mm_loadu_ps(addr1)), _mm_loadu_ps(addr2), 1), _mm_loadu_ps(addr3), 2), _mm_loadu_ps(addr4), 3)
#endif

#ifndef _mm512_load_4_m128
#define _mm512_load_4_m128(/* __m128 const* */ addr1, \
                            /* __m128 const* */ addr2, \
                            /* __m128 const* */ addr3, \
                            /* __m128 const* */ addr4) \
_mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(_mm512_castps128_ps512(_mm_load_ps(addr1)), _mm_load_ps(addr2), 1), _mm_load_ps(addr3), 2), _mm_load_ps(addr4), 3)
#endif


template<int filtersizemod4>
void resize_h_planar_float_avx512_transpose_vstripe_ks4(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);

template<int filtersizemod4>
void resize_h_planar_float_avx512_gather_permutex_vstripe_ks4(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);

template<int filtersizemod4>
void resize_h_planar_float_avx512_gather_permutex_vstripe_ks4_2w(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);


void resize_h_planar_float_avx512_permutex_vstripe_ks4(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
void resize_h_planar_float_avx512_permutex_vstripe_ks8(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
void resize_h_planar_float_avx512_permutex_vstripe_ks16(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);

void resize_v_avx512_planar_float(BYTE* dst0, const BYTE* src0, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel);
void resize_v_avx512_planar_float_w_sr(BYTE* dst0, const BYTE* src0, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel);

// uint8_t
void resize_v_avx512_planar_uint8_t_w_sr(BYTE* AVS_RESTRICT dst, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel);

// uint16_t
template<bool lessthan16bit>
void resize_v_avx512_planar_uint16_t_w_sr(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel);

#endif // __Resample_AVX512_H__
