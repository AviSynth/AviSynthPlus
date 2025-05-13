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

//#include "resample_sse.h"
#include <avs/config.h>
#include "../core/internal.h"

#include <avs/alignment.h>
#include <avs/minmax.h>

// experimental simd includes for avx2 compiled files
#if defined (__GNUC__) && ! defined (__INTEL_COMPILER)
#include <x86intrin.h>
// x86intrin.h includes header files for whatever instruction
// sets are specified on the compiler command line, such as: xopintrin.h, fma4intrin.h
#else
#include <immintrin.h> // MS version of immintrin.h covers AVX, AVX2 and FMA3
#endif // __GNUC__

#if !defined(__FMA__)
// Assume that all processors that have AVX2 also have FMA3
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


#include "resample_avx512.h"

//------- 512 bit float Horizontals

// Transpose-based
// process kernel size from up to 4 - BilinearResize, BicubicResize or sinc up to taps=2
void resize_h_planar_float_avx512_transpose_vstripe_ks4(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel) {
    int filter_size = program->filter_size;

    const float* AVS_RESTRICT current_coeff;

    src_pitch = src_pitch / sizeof(float);
    dst_pitch = dst_pitch / sizeof(float);

    float* src = (float*)src8;
    float* dst = (float*)dst8;

    current_coeff = (const float* AVS_RESTRICT)program->pixel_coefficient_float;

    for (int x = 0; x < width; x += 16) // is it safe to read by 16 floats = 64 bytes ?
    {
        __m512 c1_c5_c9_c13 = _mm512_load_4_m128(current_coeff + filter_size * 0, current_coeff + filter_size * 4, current_coeff + filter_size * 8, current_coeff + filter_size * 12);
        __m512 c2_c6_c10_c14 = _mm512_load_4_m128(current_coeff + filter_size * 1, current_coeff + filter_size * 5, current_coeff + filter_size * 9, current_coeff + filter_size * 13);
        __m512 c3_c7_c11_c15 = _mm512_load_4_m128(current_coeff + filter_size * 2, current_coeff + filter_size * 6, current_coeff + filter_size * 10, current_coeff + filter_size * 14);
        __m512 c4_c8_c12_c16 = _mm512_load_4_m128(current_coeff + filter_size * 3, current_coeff + filter_size * 7, current_coeff + filter_size * 11, current_coeff + filter_size * 15);

        _MM_TRANSPOSE16_LANE4_PS(c1_c5_c9_c13, c2_c6_c10_c14, c3_c7_c11_c15, c4_c8_c12_c16);

        float* AVS_RESTRICT dst_ptr = dst + x;
        const float* src_ptr = src;

        for (int y = 0; y < height; y++)
        {
            __m512 d1_d5_d9_d13 = _mm512_loadu_4_m128(src_ptr + program->pixel_offset[x + 0], src_ptr + program->pixel_offset[x + 4], src_ptr + program->pixel_offset[x + 8], src_ptr + program->pixel_offset[x + 12]);
            __m512 d2_d6_d10_d14 = _mm512_loadu_4_m128(src_ptr + program->pixel_offset[x + 1], src_ptr + program->pixel_offset[x + 5], src_ptr + program->pixel_offset[x + 9], src_ptr + program->pixel_offset[x + 13]);
            __m512 d3_d7_d11_d15 = _mm512_loadu_4_m128(src_ptr + program->pixel_offset[x + 2], src_ptr + program->pixel_offset[x + 6], src_ptr + program->pixel_offset[x + 10], src_ptr + program->pixel_offset[x + 14]);
            __m512 d4_d8_d12_d16 = _mm512_loadu_4_m128(src_ptr + program->pixel_offset[x + 3], src_ptr + program->pixel_offset[x + 7], src_ptr + program->pixel_offset[x + 11], src_ptr + program->pixel_offset[x + 15]);

            _MM_TRANSPOSE16_LANE4_PS(d1_d5_d9_d13, d2_d6_d10_d14, d3_d7_d11_d15, d4_d8_d12_d16);

            __m512 result = _mm512_mul_ps(d1_d5_d9_d13, c1_c5_c9_c13);
            result = _mm512_fmadd_ps(d2_d6_d10_d14, c2_c6_c10_c14, result);
            result = _mm512_fmadd_ps(d3_d7_d11_d15, c3_c7_c11_c15, result);
            result = _mm512_fmadd_ps(d4_d8_d12_d16, c4_c8_c12_c16, result);

            _mm512_store_ps(dst_ptr, result);

            dst_ptr += dst_pitch;
            src_ptr += src_pitch;
        }
        current_coeff += filter_size * 16;
    }

}


void resize_h_planar_float_avx512_permutex_vstripe_ks4(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel)
{

  // assert - check if max pixel_offset is not above single load of 16 src floats (or need several loads and more complex permute program)

#ifdef _DEBUG
  for (int x = 0; x < width; x++) // check each pair ?
  {
    int start_off = program->pixel_offset[x + 0];
    int end_off = program->pixel_offset[x + 15];
    assert((end_off - start_off) < 15);
  }
#endif

  int filter_size = program->filter_size;

  const float* AVS_RESTRICT current_coeff;
  __m512i one_epi32 = _mm512_set1_epi32(1); 

  src_pitch = src_pitch / sizeof(float);
  dst_pitch = dst_pitch / sizeof(float);

  float* src = (float*)src8;
  float* dst = (float*)dst8;

  current_coeff = (const float* AVS_RESTRICT)program->pixel_coefficient_float;

/* // 16 output samples in H-direction per vstripe
  for (int x = 0; x < width; x += 16)
  {
    // prepare coefs in transposed V-form
    __m512 coef_r0 = _mm512_load_4_m128(current_coeff + filter_size * 0, current_coeff + filter_size * 4, current_coeff + filter_size * 8, current_coeff + filter_size * 12);
    __m512 coef_r1 = _mm512_load_4_m128(current_coeff + filter_size * 1, current_coeff + filter_size * 5, current_coeff + filter_size * 9, current_coeff + filter_size * 13);
    __m512 coef_r2 = _mm512_load_4_m128(current_coeff + filter_size * 2, current_coeff + filter_size * 6, current_coeff + filter_size * 10, current_coeff + filter_size * 14);
    __m512 coef_r3 = _mm512_load_4_m128(current_coeff + filter_size * 3, current_coeff + filter_size * 7, current_coeff + filter_size * 11, current_coeff + filter_size * 15);

    _MM_TRANSPOSE16_LANE4_PS(coef_r0, coef_r1, coef_r2, coef_r3);

    // convert resampling program in H-form into permuting indexes for src transposition in V-form
    int iStart = program->pixel_offset[x + 0];
    __m512i perm_0 = _mm512_set_epi32(program->pixel_offset[x + 15] - iStart, program->pixel_offset[x + 14] - iStart, program->pixel_offset[x + 13] - iStart, program->pixel_offset[x + 12] - iStart, program->pixel_offset[x + 11] - iStart, program->pixel_offset[x + 10] - iStart, program->pixel_offset[x + 9] - iStart, program->pixel_offset[x + 8] - iStart, \
      program->pixel_offset[x + 7] - iStart, program->pixel_offset[x + 6] - iStart, program->pixel_offset[x + 5] - iStart, program->pixel_offset[x + 4] - iStart, program->pixel_offset[x + 3] - iStart, program->pixel_offset[x + 2] - iStart, program->pixel_offset[x + 1] - iStart, 0);
    __m512i perm_1 = _mm512_add_epi32(perm_0, one_epi32);
    __m512i perm_2 = _mm512_add_epi32(perm_1, one_epi32);
    __m512i perm_3 = _mm512_add_epi32(perm_2, one_epi32);

    float* AVS_RESTRICT dst_ptr = dst + x;
    const float* src_ptr = src + program->pixel_offset[x + 0]; // all permute offsets relative to this start offset

#if 0
    for (int y = 0; y < height; y++) // single row proc
    {
      __m512 data_src = _mm512_loadu_ps(src_ptr);

      __m512 data_0 = _mm512_permutexvar_ps(perm_0, data_src);
      __m512 data_1 = _mm512_permutexvar_ps(perm_1, data_src);
      __m512 data_2 = _mm512_permutexvar_ps(perm_2, data_src);
      __m512 data_3 = _mm512_permutexvar_ps(perm_3, data_src);

      __m512 result0 = _mm512_mul_ps(data_0, coef_r0);
      __m512 result1 = _mm512_mul_ps(data_2, coef_r2);

      result0 = _mm512_fmadd_ps(data_1, coef_r1, result0);
      result1 = _mm512_fmadd_ps(data_3, coef_r3, result1);

      _mm512_store_ps(dst_ptr, _mm512_add_ps(result0, result1));

      dst_ptr += dst_pitch;
      src_ptr += src_pitch;
    }
#endif

    const int height_mod2 = (height / 2) * 2; // Process pairs of rows for better efficiency
    // dual-rows not worst in performance - may be left for the future better memory performance and compute performance hosts
    for (int y = 0; y < height_mod2; y+=2)
    {
      __m512 data_src = _mm512_loadu_ps(src_ptr);
      __m512 data_src_2 = _mm512_loadu_ps(src_ptr + src_pitch);

      __m512 data_0 = _mm512_permutexvar_ps(perm_0, data_src);
      __m512 data_1 = _mm512_permutexvar_ps(perm_1, data_src);
      __m512 data_2 = _mm512_permutexvar_ps(perm_2, data_src);
      __m512 data_3 = _mm512_permutexvar_ps(perm_3, data_src);

      __m512 data_0_2 = _mm512_permutexvar_ps(perm_0, data_src_2);
      __m512 data_1_2 = _mm512_permutexvar_ps(perm_1, data_src_2);
      __m512 data_2_2 = _mm512_permutexvar_ps(perm_2, data_src_2);
      __m512 data_3_2 = _mm512_permutexvar_ps(perm_3, data_src_2);

      __m512 result0 = _mm512_mul_ps(data_0, coef_r0);
      __m512 result1 = _mm512_mul_ps(data_0_2, coef_r0);

      result0 = _mm512_fmadd_ps(data_1, coef_r1, result0);
      result1 = _mm512_fmadd_ps(data_1_2, coef_r1, result1);

      result0 = _mm512_fmadd_ps(data_2, coef_r2, result0);
      result1 = _mm512_fmadd_ps(data_2_2, coef_r2, result1);

      result0 = _mm512_fmadd_ps(data_3, coef_r3, result0);
      result1 = _mm512_fmadd_ps(data_3_2, coef_r3, result1);

      _mm512_store_ps(dst_ptr, result0);
      _mm512_store_ps(dst_ptr + dst_pitch, result1);

      dst_ptr += dst_pitch * 2;
      src_ptr += src_pitch * 2;
    }

    if (height > height_mod2) // last row
    {
      __m512 data_src = _mm512_loadu_ps(src_ptr);

      __m512 data_0 = _mm512_permutexvar_ps(perm_0, data_src);
      __m512 data_1 = _mm512_permutexvar_ps(perm_1, data_src);
      __m512 data_2 = _mm512_permutexvar_ps(perm_2, data_src);
      __m512 data_3 = _mm512_permutexvar_ps(perm_3, data_src);

      __m512 result0 = _mm512_mul_ps(data_0, coef_r0);
      __m512 result1 = _mm512_mul_ps(data_2, coef_r2);

      result0 = _mm512_fmadd_ps(data_1, coef_r1, result0);
      result1 = _mm512_fmadd_ps(data_3, coef_r3, result1);

      _mm512_store_ps(dst_ptr, _mm512_add_ps(result0, result1));
    }

    current_coeff += filter_size * 16;
  }
*/ // 16 output samples per vstripe

/*
  // 32 output samples per vstripe
  for (int x = 0; x < width; x += 32) // processing by 32 sample - it is safe at the end of row ?
  {
    // prepare coefs in transposed V-form
    __m512 coef_r0 = _mm512_load_4_m128(current_coeff + filter_size * 0, current_coeff + filter_size * 4, current_coeff + filter_size * 8, current_coeff + filter_size * 12);
    __m512 coef_r1 = _mm512_load_4_m128(current_coeff + filter_size * 1, current_coeff + filter_size * 5, current_coeff + filter_size * 9, current_coeff + filter_size * 13);
    __m512 coef_r2 = _mm512_load_4_m128(current_coeff + filter_size * 2, current_coeff + filter_size * 6, current_coeff + filter_size * 10, current_coeff + filter_size * 14);
    __m512 coef_r3 = _mm512_load_4_m128(current_coeff + filter_size * 3, current_coeff + filter_size * 7, current_coeff + filter_size * 11, current_coeff + filter_size * 15);

    // prepare coefs in transposed V-form
    __m512 coef_r0_2 = _mm512_load_4_m128(current_coeff + filter_size * 16, current_coeff + filter_size * 20, current_coeff + filter_size * 24, current_coeff + filter_size * 28);
    __m512 coef_r1_2 = _mm512_load_4_m128(current_coeff + filter_size * 17, current_coeff + filter_size * 21, current_coeff + filter_size * 25, current_coeff + filter_size * 29);
    __m512 coef_r2_2 = _mm512_load_4_m128(current_coeff + filter_size * 18, current_coeff + filter_size * 22, current_coeff + filter_size * 26, current_coeff + filter_size * 30);
    __m512 coef_r3_2 = _mm512_load_4_m128(current_coeff + filter_size * 19, current_coeff + filter_size * 23, current_coeff + filter_size * 27, current_coeff + filter_size * 31);

    _MM_TRANSPOSE16_LANE4_PS(coef_r0, coef_r1, coef_r2, coef_r3);
    _MM_TRANSPOSE16_LANE4_PS(coef_r0_2, coef_r1_2, coef_r2_2, coef_r3_2);

    // convert resampling program in H-form into permuting indexes for src transposition in V-form
    int iStart = program->pixel_offset[x + 0];
    __m512i perm_0 = _mm512_set_epi32(program->pixel_offset[x + 15] - iStart, program->pixel_offset[x + 14] - iStart, program->pixel_offset[x + 13] - iStart, program->pixel_offset[x + 12] - iStart, program->pixel_offset[x + 11] - iStart, program->pixel_offset[x + 10] - iStart, program->pixel_offset[x + 9] - iStart, program->pixel_offset[x + 8] - iStart, \
      program->pixel_offset[x + 7] - iStart, program->pixel_offset[x + 6] - iStart, program->pixel_offset[x + 5] - iStart, program->pixel_offset[x + 4] - iStart, program->pixel_offset[x + 3] - iStart, program->pixel_offset[x + 2] - iStart, program->pixel_offset[x + 1] - iStart, 0);
    __m512i perm_1 = _mm512_add_epi32(perm_0, one_epi32);
    __m512i perm_2 = _mm512_add_epi32(perm_1, one_epi32);
    __m512i perm_3 = _mm512_add_epi32(perm_2, one_epi32);

    int iStart_2 = program->pixel_offset[x + 16];
    __m512i perm_0_2 = _mm512_set_epi32(program->pixel_offset[x + 31] - iStart_2, program->pixel_offset[x + 30] - iStart_2, program->pixel_offset[x + 29] - iStart_2, program->pixel_offset[x + 28] - iStart_2, program->pixel_offset[x + 27] - iStart_2, program->pixel_offset[x + 26] - iStart_2, program->pixel_offset[x + 25] - iStart_2, program->pixel_offset[x + 24] - iStart_2, \
      program->pixel_offset[x + 23] - iStart_2, program->pixel_offset[x + 22] - iStart_2, program->pixel_offset[x + 21] - iStart_2, program->pixel_offset[x + 20] - iStart_2, program->pixel_offset[x + 19] - iStart_2, program->pixel_offset[x + 18] - iStart_2, program->pixel_offset[x + 17] - iStart_2, program->pixel_offset[x + 16] - iStart_2);
    __m512i perm_1_2 = _mm512_add_epi32(perm_0, one_epi32);
    __m512i perm_2_2 = _mm512_add_epi32(perm_1, one_epi32);
    __m512i perm_3_2 = _mm512_add_epi32(perm_2, one_epi32);

    float* AVS_RESTRICT dst_ptr = dst + x;
    float* AVS_RESTRICT dst_ptr_2 = dst + x + 16;
    const float* src_ptr = src + program->pixel_offset[x + 0]; // all permute offsets relative to this start offset
    const float* src_ptr_2 = src + program->pixel_offset[x + 16]; // all permute offsets relative to this start offset
#if 0
    for (int y = 0; y < height; y++) // single row proc, 32 output samples
    {
      __m512 data_src = _mm512_loadu_ps(src_ptr);
      __m512 data_src_2 = _mm512_loadu_ps(src_ptr_2);

      __m512 data_0 = _mm512_permutexvar_ps(perm_0, data_src);
      __m512 data_1 = _mm512_permutexvar_ps(perm_1, data_src);
      __m512 data_2 = _mm512_permutexvar_ps(perm_2, data_src);
      __m512 data_3 = _mm512_permutexvar_ps(perm_3, data_src);

      __m512 data_0_2 = _mm512_permutexvar_ps(perm_0_2, data_src_2);
      __m512 data_1_2 = _mm512_permutexvar_ps(perm_1_2, data_src_2);
      __m512 data_2_2 = _mm512_permutexvar_ps(perm_2_2, data_src_2);
      __m512 data_3_2 = _mm512_permutexvar_ps(perm_3_2, data_src_2);

      __m512 result0 = _mm512_mul_ps(data_0, coef_r0);
      __m512 result1 = _mm512_mul_ps(data_2, coef_r2);

      __m512 result0_2 = _mm512_mul_ps(data_0_2, coef_r0_2);
      __m512 result1_2 = _mm512_mul_ps(data_2_2, coef_r2_2);

      result0 = _mm512_fmadd_ps(data_1, coef_r1, result0);
      result1 = _mm512_fmadd_ps(data_3, coef_r3, result1);

      result0_2 = _mm512_fmadd_ps(data_1_2, coef_r1_2, result0_2);
      result1_2 = _mm512_fmadd_ps(data_3_2, coef_r3_2, result1_2);

      _mm512_store_ps(dst_ptr, _mm512_add_ps(result0, result1));
      _mm512_store_ps(dst_ptr_2, _mm512_add_ps(result0_2, result1_2));

      dst_ptr += dst_pitch;
      src_ptr += src_pitch;

      dst_ptr_2 += dst_pitch;
      src_ptr_2 += src_pitch;

    }
#endif // single row
    //dual rows and 32 per vstripe
    const int height_mod2 = (height / 2) * 2; // Process pairs of rows for better efficiency
    // dual-rows not worst in performance - may be left for the future better memory performance and compute performance hosts
    for (int y = 0; y < height_mod2; y += 2)
    {
      __m512 data_src = _mm512_loadu_ps(src_ptr);
      __m512 data_src_2 = _mm512_loadu_ps(src_ptr_2);

      __m512 data_src_2r = _mm512_loadu_ps(src_ptr + src_pitch);
      __m512 data_src_2_2r = _mm512_loadu_ps(src_ptr_2 + src_pitch);

      __m512 data_0 = _mm512_permutexvar_ps(perm_0, data_src);
      __m512 data_1 = _mm512_permutexvar_ps(perm_1, data_src);
      __m512 data_2 = _mm512_permutexvar_ps(perm_2, data_src);
      __m512 data_3 = _mm512_permutexvar_ps(perm_3, data_src);

      __m512 data_0_2 = _mm512_permutexvar_ps(perm_0_2, data_src_2);
      __m512 data_1_2 = _mm512_permutexvar_ps(perm_1_2, data_src_2);
      __m512 data_2_2 = _mm512_permutexvar_ps(perm_2_2, data_src_2);
      __m512 data_3_2 = _mm512_permutexvar_ps(perm_3_2, data_src_2);

      __m512 data_0_2r = _mm512_permutexvar_ps(perm_0, data_src_2r);
      __m512 data_1_2r = _mm512_permutexvar_ps(perm_1, data_src_2r);
      __m512 data_2_2r = _mm512_permutexvar_ps(perm_2, data_src_2r);
      __m512 data_3_2r = _mm512_permutexvar_ps(perm_3, data_src_2r);

      __m512 data_0_2_2r = _mm512_permutexvar_ps(perm_0_2, data_src_2_2r);
      __m512 data_1_2_2r = _mm512_permutexvar_ps(perm_1_2, data_src_2_2r);
      __m512 data_2_2_2r = _mm512_permutexvar_ps(perm_2_2, data_src_2_2r);
      __m512 data_3_2_2r = _mm512_permutexvar_ps(perm_3_2, data_src_2_2r);

      // 1r
      __m512 result0 = _mm512_mul_ps(data_0, coef_r0);
      __m512 result1 = _mm512_mul_ps(data_2, coef_r2);

      __m512 result0_2 = _mm512_mul_ps(data_0_2, coef_r0_2);
      __m512 result1_2 = _mm512_mul_ps(data_2_2, coef_r2_2);

      result0 = _mm512_fmadd_ps(data_1, coef_r1, result0);
      result1 = _mm512_fmadd_ps(data_3, coef_r3, result1);

      result0_2 = _mm512_fmadd_ps(data_1_2, coef_r1_2, result0_2);
      result1_2 = _mm512_fmadd_ps(data_3_2, coef_r3_2, result1_2);

      // 2r
      __m512 result0_2r = _mm512_mul_ps(data_0_2r, coef_r0);
      __m512 result1_2r = _mm512_mul_ps(data_2_2r, coef_r2);

      __m512 result0_2_2r = _mm512_mul_ps(data_0_2_2r, coef_r0_2);
      __m512 result1_2_2r = _mm512_mul_ps(data_2_2_2r, coef_r2_2);

      result0_2r = _mm512_fmadd_ps(data_1_2r, coef_r1, result0_2r);
      result1_2r = _mm512_fmadd_ps(data_3_2r, coef_r3, result1_2r);

      result0_2_2r = _mm512_fmadd_ps(data_1_2_2r, coef_r1_2, result0_2_2r);
      result1_2_2r = _mm512_fmadd_ps(data_3_2_2r, coef_r3_2, result1_2_2r);


      _mm512_store_ps(dst_ptr, _mm512_add_ps(result0, result1));
      _mm512_store_ps(dst_ptr_2, _mm512_add_ps(result0_2, result1_2));

      _mm512_store_ps(dst_ptr + dst_pitch, _mm512_add_ps(result0_2r, result1_2r));
      _mm512_store_ps(dst_ptr_2 + dst_pitch, _mm512_add_ps(result0_2_2r, result1_2_2r));


      dst_ptr += dst_pitch * 2;
      src_ptr += src_pitch * 2;

      dst_ptr_2 += dst_pitch * 2;
      src_ptr_2 += src_pitch * 2;

    }
*/


// some slower than 32 per vstripe with small cacheabe frame sizes and best performance with large frame size and many threads (most SDRAM controller friendly ?)

  for (int x = 0; x < width; x += 64) // processing by 64 output sample - it is safe at the end of row ?
  {
    // prepare coefs in transposed V-form
    __m512 coef_r0 = _mm512_load_4_m128(current_coeff + filter_size * 0, current_coeff + filter_size * 4, current_coeff + filter_size * 8, current_coeff + filter_size * 12);
    __m512 coef_r1 = _mm512_load_4_m128(current_coeff + filter_size * 1, current_coeff + filter_size * 5, current_coeff + filter_size * 9, current_coeff + filter_size * 13);
    __m512 coef_r2 = _mm512_load_4_m128(current_coeff + filter_size * 2, current_coeff + filter_size * 6, current_coeff + filter_size * 10, current_coeff + filter_size * 14);
    __m512 coef_r3 = _mm512_load_4_m128(current_coeff + filter_size * 3, current_coeff + filter_size * 7, current_coeff + filter_size * 11, current_coeff + filter_size * 15);

    // prepare coefs in transposed V-form
    __m512 coef_r0_2 = _mm512_load_4_m128(current_coeff + filter_size * 16, current_coeff + filter_size * 20, current_coeff + filter_size * 24, current_coeff + filter_size * 28);
    __m512 coef_r1_2 = _mm512_load_4_m128(current_coeff + filter_size * 17, current_coeff + filter_size * 21, current_coeff + filter_size * 25, current_coeff + filter_size * 29);
    __m512 coef_r2_2 = _mm512_load_4_m128(current_coeff + filter_size * 18, current_coeff + filter_size * 22, current_coeff + filter_size * 26, current_coeff + filter_size * 30);
    __m512 coef_r3_2 = _mm512_load_4_m128(current_coeff + filter_size * 19, current_coeff + filter_size * 23, current_coeff + filter_size * 27, current_coeff + filter_size * 31);

    // prepare coefs in transposed V-form
    __m512 coef_r0_3 = _mm512_load_4_m128(current_coeff + filter_size * 32, current_coeff + filter_size * 36, current_coeff + filter_size * 40, current_coeff + filter_size * 44);
    __m512 coef_r1_3 = _mm512_load_4_m128(current_coeff + filter_size * 33, current_coeff + filter_size * 37, current_coeff + filter_size * 41, current_coeff + filter_size * 45);
    __m512 coef_r2_3 = _mm512_load_4_m128(current_coeff + filter_size * 34, current_coeff + filter_size * 38, current_coeff + filter_size * 42, current_coeff + filter_size * 46);
    __m512 coef_r3_3 = _mm512_load_4_m128(current_coeff + filter_size * 35, current_coeff + filter_size * 39, current_coeff + filter_size * 43, current_coeff + filter_size * 47);

    // prepare coefs in transposed V-form
    __m512 coef_r0_4 = _mm512_load_4_m128(current_coeff + filter_size * 48, current_coeff + filter_size * 52, current_coeff + filter_size * 56, current_coeff + filter_size * 60);
    __m512 coef_r1_4 = _mm512_load_4_m128(current_coeff + filter_size * 49, current_coeff + filter_size * 53, current_coeff + filter_size * 57, current_coeff + filter_size * 61);
    __m512 coef_r2_4 = _mm512_load_4_m128(current_coeff + filter_size * 50, current_coeff + filter_size * 54, current_coeff + filter_size * 58, current_coeff + filter_size * 62);
    __m512 coef_r3_4 = _mm512_load_4_m128(current_coeff + filter_size * 51, current_coeff + filter_size * 55, current_coeff + filter_size * 59, current_coeff + filter_size * 63);

    _MM_TRANSPOSE16_LANE4_PS(coef_r0, coef_r1, coef_r2, coef_r3);
    _MM_TRANSPOSE16_LANE4_PS(coef_r0_2, coef_r1_2, coef_r2_2, coef_r3_2);
    _MM_TRANSPOSE16_LANE4_PS(coef_r0_3, coef_r1_3, coef_r2_3, coef_r3_3);
    _MM_TRANSPOSE16_LANE4_PS(coef_r0_4, coef_r1_4, coef_r2_4, coef_r3_4);

    // convert resampling program in H-form into permuting indexes for src transposition in V-form
    int iStart = program->pixel_offset[x + 0];
    __m512i perm_0 = _mm512_set_epi32(program->pixel_offset[x + 15] - iStart, program->pixel_offset[x + 14] - iStart, program->pixel_offset[x + 13] - iStart, program->pixel_offset[x + 12] - iStart, program->pixel_offset[x + 11] - iStart, program->pixel_offset[x + 10] - iStart, program->pixel_offset[x + 9] - iStart, program->pixel_offset[x + 8] - iStart, \
      program->pixel_offset[x + 7] - iStart, program->pixel_offset[x + 6] - iStart, program->pixel_offset[x + 5] - iStart, program->pixel_offset[x + 4] - iStart, program->pixel_offset[x + 3] - iStart, program->pixel_offset[x + 2] - iStart, program->pixel_offset[x + 1] - iStart, 0);

    int iStart_2 = program->pixel_offset[x + 16];
    __m512i perm_0_2 = _mm512_set_epi32(program->pixel_offset[x + 31] - iStart_2, program->pixel_offset[x + 30] - iStart_2, program->pixel_offset[x + 29] - iStart_2, program->pixel_offset[x + 28] - iStart_2, program->pixel_offset[x + 27] - iStart_2, program->pixel_offset[x + 26] - iStart_2, program->pixel_offset[x + 25] - iStart_2, program->pixel_offset[x + 24] - iStart_2, \
      program->pixel_offset[x + 23] - iStart_2, program->pixel_offset[x + 22] - iStart_2, program->pixel_offset[x + 21] - iStart_2, program->pixel_offset[x + 20] - iStart_2, program->pixel_offset[x + 19] - iStart_2, program->pixel_offset[x + 18] - iStart_2, program->pixel_offset[x + 17] - iStart_2, program->pixel_offset[x + 16] - iStart_2);

    int iStart_3 = program->pixel_offset[x + 32];
    __m512i perm_0_3 = _mm512_set_epi32(program->pixel_offset[x + 47] - iStart_3, program->pixel_offset[x + 46] - iStart_3, program->pixel_offset[x + 45] - iStart_3, program->pixel_offset[x + 44] - iStart_3, program->pixel_offset[x + 43] - iStart_3, program->pixel_offset[x + 42] - iStart_3, program->pixel_offset[x + 41] - iStart_3, program->pixel_offset[x + 40] - iStart_3, \
      program->pixel_offset[x + 39] - iStart_3, program->pixel_offset[x + 38] - iStart_3, program->pixel_offset[x + 37] - iStart_3, program->pixel_offset[x + 36] - iStart_3, program->pixel_offset[x + 35] - iStart_3, program->pixel_offset[x + 34] - iStart_3, program->pixel_offset[x + 33] - iStart_3, program->pixel_offset[x + 32] - iStart_3);

    int iStart_4 = program->pixel_offset[x + 48];
    __m512i perm_0_4 = _mm512_set_epi32(program->pixel_offset[x + 63] - iStart_4, program->pixel_offset[x + 62] - iStart_4, program->pixel_offset[x + 61] - iStart_4, program->pixel_offset[x + 60] - iStart_4, program->pixel_offset[x + 59] - iStart_4, program->pixel_offset[x + 58] - iStart_4, program->pixel_offset[x + 57] - iStart_4, program->pixel_offset[x + 56] - iStart_4, \
      program->pixel_offset[x + 55] - iStart_4, program->pixel_offset[x + 54] - iStart_4, program->pixel_offset[x + 53] - iStart_4, program->pixel_offset[x + 52] - iStart_4, program->pixel_offset[x + 51] - iStart_4, program->pixel_offset[x + 50] - iStart_4, program->pixel_offset[x + 49] - iStart_4, program->pixel_offset[x + 48] - iStart_4);


    float* AVS_RESTRICT dst_ptr = dst + x;
    const float* src_ptr = src + program->pixel_offset[x + 0]; // all permute offsets relative to this start offset
    const float* src_ptr_2 = src + program->pixel_offset[x + 16]; // all permute offsets relative to this start offset
    const float* src_ptr_3 = src + program->pixel_offset[x + 32]; // all permute offsets relative to this start offset
    const float* src_ptr_4 = src + program->pixel_offset[x + 48]; // all permute offsets relative to this start offset

    for (int y = 0; y < height; y++) // single row proc, 32 output samples
    {
      __m512 data_src = _mm512_loadu_ps(src_ptr);
      __m512 data_src_2 = _mm512_loadu_ps(src_ptr_2);
      __m512 data_src_3 = _mm512_loadu_ps(src_ptr_3);
      __m512 data_src_4 = _mm512_loadu_ps(src_ptr_4);

      // 1st
      __m512 data_0 = _mm512_permutexvar_ps(perm_0, data_src);

      __m512i perm_next = _mm512_add_epi32(perm_0, one_epi32);
      __m512 data_1 = _mm512_permutexvar_ps(perm_next, data_src);

      perm_next = _mm512_add_epi32(perm_next, one_epi32);
      __m512 data_2 = _mm512_permutexvar_ps(perm_next, data_src);

      perm_next = _mm512_add_epi32(perm_next, one_epi32);
      __m512 data_3 = _mm512_permutexvar_ps(perm_next, data_src);

      // 2nd
      __m512 data_0_2 = _mm512_permutexvar_ps(perm_0_2, data_src_2);

      __m512i perm_next_2 = _mm512_add_epi32(perm_0_2, one_epi32);
      __m512 data_1_2 = _mm512_permutexvar_ps(perm_next_2, data_src_2);

      perm_next_2 = _mm512_add_epi32(perm_0_2, one_epi32);
      __m512 data_2_2 = _mm512_permutexvar_ps(perm_next_2, data_src_2);

      perm_next_2 = _mm512_add_epi32(perm_next_2, one_epi32);
      __m512 data_3_2 = _mm512_permutexvar_ps(perm_next_2, data_src_2);

      // 3rd
      __m512 data_0_3 = _mm512_permutexvar_ps(perm_0_3, data_src_3);

      __m512i perm_next_3 = _mm512_add_epi32(perm_0_3, one_epi32);
      __m512 data_1_3 = _mm512_permutexvar_ps(perm_next_3, data_src_3);

      perm_next_3 = _mm512_add_epi32(perm_0_3, one_epi32);
      __m512 data_2_3 = _mm512_permutexvar_ps(perm_next_3, data_src_3);

      perm_next_3 = _mm512_add_epi32(perm_next_3, one_epi32);
      __m512 data_3_3 = _mm512_permutexvar_ps(perm_next_3, data_src_3);

      // 4th
      __m512 data_0_4 = _mm512_permutexvar_ps(perm_0_4, data_src_4);

      __m512i perm_next_4 = _mm512_add_epi32(perm_0_4, one_epi32);
      __m512 data_1_4 = _mm512_permutexvar_ps(perm_next_4, data_src_4);

      perm_next_4 = _mm512_add_epi32(perm_0_4, one_epi32);
      __m512 data_2_4 = _mm512_permutexvar_ps(perm_next_4, data_src_4);

      perm_next_4 = _mm512_add_epi32(perm_next_4, one_epi32);
      __m512 data_3_4 = _mm512_permutexvar_ps(perm_next_4, data_src_3);


      __m512 result0 = _mm512_mul_ps(data_0, coef_r0);
      __m512 result1 = _mm512_mul_ps(data_2, coef_r2);

      __m512 result0_2 = _mm512_mul_ps(data_0_2, coef_r0_2);
      __m512 result1_2 = _mm512_mul_ps(data_2_2, coef_r2_2);

      __m512 result0_3 = _mm512_mul_ps(data_0_3, coef_r0_3);
      __m512 result1_3 = _mm512_mul_ps(data_2_3, coef_r2_3);

      __m512 result0_4 = _mm512_mul_ps(data_0_4, coef_r0_4);
      __m512 result1_4 = _mm512_mul_ps(data_2_4, coef_r2_4);

      result0 = _mm512_fmadd_ps(data_1, coef_r1, result0);
      result1 = _mm512_fmadd_ps(data_3, coef_r3, result1);

      result0_2 = _mm512_fmadd_ps(data_1_2, coef_r1_2, result0_2);
      result1_2 = _mm512_fmadd_ps(data_3_2, coef_r3_2, result1_2);

      result0_3 = _mm512_fmadd_ps(data_1_3, coef_r1_3, result0_3);
      result1_3 = _mm512_fmadd_ps(data_3_3, coef_r3_3, result1_3);

      _mm512_store_ps(dst_ptr, _mm512_add_ps(result0, result1));
      _mm512_store_ps(dst_ptr + 16, _mm512_add_ps(result0_2, result1_2));
      _mm512_store_ps(dst_ptr + 32, _mm512_add_ps(result0_3, result1_3));
      _mm512_store_ps(dst_ptr + 48, _mm512_add_ps(result0_4, result1_4));

      dst_ptr += dst_pitch;

      src_ptr += src_pitch;
      src_ptr_2 += src_pitch;
      src_ptr_3 += src_pitch;
      src_ptr_4 += src_pitch;

    }


/*
    const int height_mod2 = (height / 2) * 2; // Process pairs of rows for better efficiency
    // dual-rows not worst in performance - may be left for the future better memory performance and compute performance hosts
    for (int y = 0; y < height_mod2; y += 2)
    {
      __m512 data_src = _mm512_loadu_ps(src_ptr);
      __m512 data_src_2 = _mm512_loadu_ps(src_ptr + src_pitch);

      __m512 data_0 = _mm512_permutexvar_ps(perm_0, data_src);
      __m512 data_1 = _mm512_permutexvar_ps(perm_1, data_src);
      __m512 data_2 = _mm512_permutexvar_ps(perm_2, data_src);
      __m512 data_3 = _mm512_permutexvar_ps(perm_3, data_src);

      __m512 data_0_2 = _mm512_permutexvar_ps(perm_0, data_src_2);
      __m512 data_1_2 = _mm512_permutexvar_ps(perm_1, data_src_2);
      __m512 data_2_2 = _mm512_permutexvar_ps(perm_2, data_src_2);
      __m512 data_3_2 = _mm512_permutexvar_ps(perm_3, data_src_2);

      __m512 result0 = _mm512_mul_ps(data_0, coef_r0);
      __m512 result1 = _mm512_mul_ps(data_0_2, coef_r0);

      result0 = _mm512_fmadd_ps(data_1, coef_r1, result0);
      result1 = _mm512_fmadd_ps(data_1_2, coef_r1, result1);

      result0 = _mm512_fmadd_ps(data_2, coef_r2, result0);
      result1 = _mm512_fmadd_ps(data_2_2, coef_r2, result1);

      result0 = _mm512_fmadd_ps(data_3, coef_r3, result0);
      result1 = _mm512_fmadd_ps(data_3_2, coef_r3, result1);

      _mm512_store_ps(dst_ptr, result0);
      _mm512_store_ps(dst_ptr + dst_pitch, result1);

      dst_ptr += dst_pitch * 2;
      src_ptr += src_pitch * 2;
    }

    if (height > height_mod2) // last row
    {
      __m512 data_src = _mm512_loadu_ps(src_ptr);

      __m512 data_0 = _mm512_permutexvar_ps(perm_0, data_src);
      __m512 data_1 = _mm512_permutexvar_ps(perm_1, data_src);
      __m512 data_2 = _mm512_permutexvar_ps(perm_2, data_src);
      __m512 data_3 = _mm512_permutexvar_ps(perm_3, data_src);

      __m512 result0 = _mm512_mul_ps(data_0, coef_r0);
      __m512 result1 = _mm512_mul_ps(data_2, coef_r2);

      result0 = _mm512_fmadd_ps(data_1, coef_r1, result0);
      result1 = _mm512_fmadd_ps(data_3, coef_r3, result1);

      _mm512_store_ps(dst_ptr, _mm512_add_ps(result0, result1));
    }
*/ // dual rows

    current_coeff += filter_size * 64;
  }


}

void resize_h_planar_float_avx512_permutex_vstripe_ks8(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel)
{
  // assert - check if max pixel_offset is not above single load of 16 src floats (or need several loads and more complex permute program)

#ifdef _DEBUG
  for (int x = 0; x < width; x += 16)
  {
    int start_off = program->pixel_offset[x + 0];
    int end_off = program->pixel_offset[x + 15];
    assert((end_off - start_off) > 15);
  }
#endif

  int filter_size = program->filter_size;

  const float* AVS_RESTRICT current_coeff;
  __m512i one_epi32 = _mm512_set1_epi32(1); 

  src_pitch = src_pitch / sizeof(float);
  dst_pitch = dst_pitch / sizeof(float);

  float* src = (float*)src8;
  float* dst = (float*)dst8;

  current_coeff = (const float* AVS_RESTRICT)program->pixel_coefficient_float;

  for (int x = 0; x < width; x += 16)
  {
    // prepare coefs in transposed V-form, use gathering - not very slow until TRANSPOSE8_ is designed

    __m512i offsets = _mm512_set_epi32(filter_size * 7, filter_size * 6, filter_size * 5, filter_size * 4, filter_size * 3, filter_size * 2, filter_size * 1, filter_size * 0, \
                                       filter_size * 7, filter_size * 6, filter_size * 5, filter_size * 4, filter_size * 3, filter_size * 2, filter_size * 1, filter_size * 0 );

    __m512 coef_r0 = _mm512_i32gather_ps(offsets, current_coeff, 4);

    offsets = _mm512_add_epi32(offsets, one_epi32);
    __m512 coef_r1 = _mm512_i32gather_ps(offsets, current_coeff, 4);

    offsets = _mm512_add_epi32(offsets, one_epi32);
    __m512 coef_r2 = _mm512_i32gather_ps(offsets, current_coeff, 4);

    offsets = _mm512_add_epi32(offsets, one_epi32);
    __m512 coef_r3 = _mm512_i32gather_ps(offsets, current_coeff, 4);

    offsets = _mm512_add_epi32(offsets, one_epi32);
    __m512 coef_r4 = _mm512_i32gather_ps(offsets, current_coeff, 4);

    offsets = _mm512_add_epi32(offsets, one_epi32);
    __m512 coef_r5 = _mm512_i32gather_ps(offsets, current_coeff, 4);

    offsets = _mm512_add_epi32(offsets, one_epi32);
    __m512 coef_r6 = _mm512_i32gather_ps(offsets, current_coeff, 4);

    offsets = _mm512_add_epi32(offsets, one_epi32);
    __m512 coef_r7 = _mm512_i32gather_ps(offsets, current_coeff, 4);


    // convert resampling program in H-form into permuting indexes for src transposition in V-form
    int iStart = program->pixel_offset[x + 0];
    __m512i perm_0 = _mm512_set_epi32(program->pixel_offset[x + 15] - iStart, program->pixel_offset[x + 14] - iStart, program->pixel_offset[x + 13] - iStart, program->pixel_offset[x + 12] - iStart, program->pixel_offset[x + 11] - iStart, program->pixel_offset[x + 10] - iStart, program->pixel_offset[x + 9] - iStart, program->pixel_offset[x + 8] - iStart, \
      program->pixel_offset[x + 7] - iStart, program->pixel_offset[x + 6] - iStart, program->pixel_offset[x + 5] - iStart, program->pixel_offset[x + 4] - iStart, program->pixel_offset[x + 3] - iStart, program->pixel_offset[x + 2] - iStart, program->pixel_offset[x + 1] - iStart, 0);
    __m512i perm_1 = _mm512_add_epi32(perm_0, one_epi32);
    __m512i perm_2 = _mm512_add_epi32(perm_1, one_epi32);
    __m512i perm_3 = _mm512_add_epi32(perm_2, one_epi32);
    __m512i perm_4 = _mm512_add_epi32(perm_3, one_epi32);
    __m512i perm_5 = _mm512_add_epi32(perm_4, one_epi32);
    __m512i perm_6 = _mm512_add_epi32(perm_5, one_epi32);
    __m512i perm_7 = _mm512_add_epi32(perm_6, one_epi32);

    float* AVS_RESTRICT dst_ptr = dst + x;
    const float* src_ptr = src + program->pixel_offset[x + 0]; // all permute offsets relative to this start offset
#if 0
    for (int y = 0; y < height; y++) // single row proc
    {
      __m512 data_src = _mm512_loadu_ps(src_ptr);

      __m512 data_0 = _mm512_permutexvar_ps(perm_0, data_src);
      __m512 data_1 = _mm512_permutexvar_ps(perm_1, data_src);
      __m512 data_2 = _mm512_permutexvar_ps(perm_2, data_src);
      __m512 data_3 = _mm512_permutexvar_ps(perm_3, data_src);
      __m512 data_4 = _mm512_permutexvar_ps(perm_4, data_src);
      __m512 data_5 = _mm512_permutexvar_ps(perm_5, data_src);
      __m512 data_6 = _mm512_permutexvar_ps(perm_6, data_src);
      __m512 data_7 = _mm512_permutexvar_ps(perm_7, data_src);

      __m512 result0 = _mm512_mul_ps(data_0, coef_r0);
      __m512 result1 = _mm512_mul_ps(data_4, coef_r4);

      result0 = _mm512_fmadd_ps(data_1, coef_r1, result0);
      result1 = _mm512_fmadd_ps(data_5, coef_r5, result1);

      result0 = _mm512_fmadd_ps(data_2, coef_r2, result0);
      result1 = _mm512_fmadd_ps(data_6, coef_r6, result1);

      result0 = _mm512_fmadd_ps(data_3, coef_r3, result0);
      result1 = _mm512_fmadd_ps(data_7, coef_r7, result1);

      _mm512_store_ps(dst_ptr, _mm512_add_ps(result0, result1));

      dst_ptr += dst_pitch;
      src_ptr += src_pitch;
    }
#endif

    const int height_mod2 = (height / 2) * 2; // Process pairs of rows for better efficiency
    // dual-rows not worst in performance - may be left for the future better memory performance and compute performance hosts
    for (int y = 0; y < height_mod2; y += 2)
    {
      __m512 data_src = _mm512_loadu_ps(src_ptr);
      __m512 data_src_2 = _mm512_loadu_ps(src_ptr + src_pitch);

      __m512 data_0 = _mm512_permutexvar_ps(perm_0, data_src);
      __m512 data_1 = _mm512_permutexvar_ps(perm_1, data_src);
      __m512 data_2 = _mm512_permutexvar_ps(perm_2, data_src);
      __m512 data_3 = _mm512_permutexvar_ps(perm_3, data_src);
      __m512 data_4 = _mm512_permutexvar_ps(perm_4, data_src);
      __m512 data_5 = _mm512_permutexvar_ps(perm_5, data_src);
      __m512 data_6 = _mm512_permutexvar_ps(perm_6, data_src);
      __m512 data_7 = _mm512_permutexvar_ps(perm_7, data_src);

      __m512 data_0_2 = _mm512_permutexvar_ps(perm_0, data_src_2);
      __m512 data_1_2 = _mm512_permutexvar_ps(perm_1, data_src_2);
      __m512 data_2_2 = _mm512_permutexvar_ps(perm_2, data_src_2);
      __m512 data_3_2 = _mm512_permutexvar_ps(perm_3, data_src_2);
      __m512 data_4_2 = _mm512_permutexvar_ps(perm_4, data_src_2);
      __m512 data_5_2 = _mm512_permutexvar_ps(perm_5, data_src_2);
      __m512 data_6_2 = _mm512_permutexvar_ps(perm_6, data_src_2);
      __m512 data_7_2 = _mm512_permutexvar_ps(perm_7, data_src_2);

      __m512 result0 = _mm512_mul_ps(data_0, coef_r0);
      __m512 result1 = _mm512_mul_ps(data_0_2, coef_r0);

      result0 = _mm512_fmadd_ps(data_1, coef_r1, result0);
      result1 = _mm512_fmadd_ps(data_1_2, coef_r1, result1);

      result0 = _mm512_fmadd_ps(data_2, coef_r2, result0);
      result1 = _mm512_fmadd_ps(data_2_2, coef_r2, result1);

      result0 = _mm512_fmadd_ps(data_3, coef_r3, result0);
      result1 = _mm512_fmadd_ps(data_3_2, coef_r3, result1);

      result0 = _mm512_fmadd_ps(data_4, coef_r4, result0);
      result1 = _mm512_fmadd_ps(data_4_2, coef_r4, result1);

      result0 = _mm512_fmadd_ps(data_5, coef_r5, result0);
      result1 = _mm512_fmadd_ps(data_5_2, coef_r5, result1);

      result0 = _mm512_fmadd_ps(data_6, coef_r6, result0);
      result1 = _mm512_fmadd_ps(data_6_2, coef_r6, result1);

      result0 = _mm512_fmadd_ps(data_7, coef_r7, result0);
      result1 = _mm512_fmadd_ps(data_7_2, coef_r7, result1);

      _mm512_store_ps(dst_ptr, result0);
      _mm512_store_ps(dst_ptr + dst_pitch, result1);

      dst_ptr += dst_pitch * 2;
      src_ptr += src_pitch * 2;
    }

    if (height > height_mod2) // last row
    {
      __m512 data_src = _mm512_loadu_ps(src_ptr);

      __m512 data_0 = _mm512_permutexvar_ps(perm_0, data_src);
      __m512 data_1 = _mm512_permutexvar_ps(perm_1, data_src);
      __m512 data_2 = _mm512_permutexvar_ps(perm_2, data_src);
      __m512 data_3 = _mm512_permutexvar_ps(perm_3, data_src);
      __m512 data_4 = _mm512_permutexvar_ps(perm_4, data_src);
      __m512 data_5 = _mm512_permutexvar_ps(perm_5, data_src);
      __m512 data_6 = _mm512_permutexvar_ps(perm_6, data_src);
      __m512 data_7 = _mm512_permutexvar_ps(perm_7, data_src);

      __m512 result0 = _mm512_mul_ps(data_0, coef_r0);
      __m512 result1 = _mm512_mul_ps(data_4, coef_r4);

      result0 = _mm512_fmadd_ps(data_1, coef_r1, result0);
      result1 = _mm512_fmadd_ps(data_5, coef_r5, result1);

      result0 = _mm512_fmadd_ps(data_2, coef_r2, result0);
      result1 = _mm512_fmadd_ps(data_6, coef_r6, result1);

      result0 = _mm512_fmadd_ps(data_3, coef_r3, result0);
      result1 = _mm512_fmadd_ps(data_7, coef_r7, result1);

      _mm512_store_ps(dst_ptr, _mm512_add_ps(result0, result1));
    }

    current_coeff += filter_size * 16;
  }
}

void resize_h_planar_float_avx512_permutex_vstripe_ks16(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel)
{
  // assert - check if max pixel_offset is not above single load of 16 src floats (or need several loads and more complex permute program)
#ifdef _DEBUG
  for (int x = 0; x < width; x += 16)
  {
    int start_off = program->pixel_offset[x + 0];
    int end_off = program->pixel_offset[x + 15];
    assert((end_off - start_off) > 15);
  }
#endif

  int filter_size = program->filter_size;

  const float* AVS_RESTRICT current_coeff;
  __m512i one_epi32 = _mm512_set1_epi32(1); 

  src_pitch = src_pitch / sizeof(float);
  dst_pitch = dst_pitch / sizeof(float);

  float* src = (float*)src8;
  float* dst = (float*)dst8;

  current_coeff = (const float* AVS_RESTRICT)program->pixel_coefficient_float;

  for (int x = 0; x < width; x += 16)
  {
    // prepare coefs in transposed V-form, use gathering - not very slow until TRANSPOSE8_ is designed

    __m512i offsets = _mm512_set_epi32(filter_size * 15, filter_size * 14, filter_size * 13, filter_size * 12, filter_size * 11, filter_size * 10, filter_size * 9, filter_size * 8, \
      filter_size * 7, filter_size * 6, filter_size * 5, filter_size * 4, filter_size * 3, filter_size * 2, filter_size * 1, filter_size * 0);

    __m512 coef_r0 = _mm512_i32gather_ps(offsets, current_coeff, 4);

    offsets = _mm512_add_epi32(offsets, one_epi32);
    __m512 coef_r1 = _mm512_i32gather_ps(offsets, current_coeff, 4);

    offsets = _mm512_add_epi32(offsets, one_epi32);
    __m512 coef_r2 = _mm512_i32gather_ps(offsets, current_coeff, 4);

    offsets = _mm512_add_epi32(offsets, one_epi32);
    __m512 coef_r3 = _mm512_i32gather_ps(offsets, current_coeff, 4);

    offsets = _mm512_add_epi32(offsets, one_epi32);
    __m512 coef_r4 = _mm512_i32gather_ps(offsets, current_coeff, 4);

    offsets = _mm512_add_epi32(offsets, one_epi32);
    __m512 coef_r5 = _mm512_i32gather_ps(offsets, current_coeff, 4);

    offsets = _mm512_add_epi32(offsets, one_epi32);
    __m512 coef_r6 = _mm512_i32gather_ps(offsets, current_coeff, 4);

    offsets = _mm512_add_epi32(offsets, one_epi32);
    __m512 coef_r7 = _mm512_i32gather_ps(offsets, current_coeff, 4);

    offsets = _mm512_add_epi32(offsets, one_epi32);
    __m512 coef_r8 = _mm512_i32gather_ps(offsets, current_coeff, 4);

    offsets = _mm512_add_epi32(offsets, one_epi32);
    __m512 coef_r9 = _mm512_i32gather_ps(offsets, current_coeff, 4);

    offsets = _mm512_add_epi32(offsets, one_epi32);
    __m512 coef_r10 = _mm512_i32gather_ps(offsets, current_coeff, 4);

    offsets = _mm512_add_epi32(offsets, one_epi32);
    __m512 coef_r11 = _mm512_i32gather_ps(offsets, current_coeff, 4);

    offsets = _mm512_add_epi32(offsets, one_epi32);
    __m512 coef_r12 = _mm512_i32gather_ps(offsets, current_coeff, 4);

    offsets = _mm512_add_epi32(offsets, one_epi32);
    __m512 coef_r13 = _mm512_i32gather_ps(offsets, current_coeff, 4);

    offsets = _mm512_add_epi32(offsets, one_epi32);
    __m512 coef_r14 = _mm512_i32gather_ps(offsets, current_coeff, 4);

    offsets = _mm512_add_epi32(offsets, one_epi32);
    __m512 coef_r15 = _mm512_i32gather_ps(offsets, current_coeff, 4);


    // convert resampling program in H-form into permuting indexes for src transposition in V-form
    int iStart = program->pixel_offset[x + 0];
    __m512i perm_0 = _mm512_set_epi32(program->pixel_offset[x + 15] - iStart, program->pixel_offset[x + 14] - iStart, program->pixel_offset[x + 13] - iStart, program->pixel_offset[x + 12] - iStart, program->pixel_offset[x + 11] - iStart, program->pixel_offset[x + 10] - iStart, program->pixel_offset[x + 9] - iStart, program->pixel_offset[x + 8] - iStart, \
      program->pixel_offset[x + 7] - iStart, program->pixel_offset[x + 6] - iStart, program->pixel_offset[x + 5] - iStart, program->pixel_offset[x + 4] - iStart, program->pixel_offset[x + 3] - iStart, program->pixel_offset[x + 2] - iStart, program->pixel_offset[x + 1] - iStart, 0);
    __m512i perm_1 = _mm512_add_epi32(perm_0, one_epi32);
    __m512i perm_2 = _mm512_add_epi32(perm_1, one_epi32);
    __m512i perm_3 = _mm512_add_epi32(perm_2, one_epi32);
    __m512i perm_4 = _mm512_add_epi32(perm_3, one_epi32);
    __m512i perm_5 = _mm512_add_epi32(perm_4, one_epi32);
    __m512i perm_6 = _mm512_add_epi32(perm_5, one_epi32);
    __m512i perm_7 = _mm512_add_epi32(perm_6, one_epi32);
    __m512i perm_8 = _mm512_add_epi32(perm_7, one_epi32);
    __m512i perm_9 = _mm512_add_epi32(perm_8, one_epi32);
    __m512i perm_10 = _mm512_add_epi32(perm_9, one_epi32);
    __m512i perm_11 = _mm512_add_epi32(perm_10, one_epi32);
    __m512i perm_12 = _mm512_add_epi32(perm_11, one_epi32);
    __m512i perm_13 = _mm512_add_epi32(perm_12, one_epi32);
    __m512i perm_14 = _mm512_add_epi32(perm_13, one_epi32);
    __m512i perm_15 = _mm512_add_epi32(perm_14, one_epi32); // to do : test if better to add one_epi32 in the loop and only store perm_0 complex to fill dataword

    float* AVS_RESTRICT dst_ptr = dst + x;
    const float* src_ptr = src + program->pixel_offset[x + 0]; // all permute offsets relative to this start offset

    for (int y = 0; y < height; y++) // single row proc
    {
      __m512 data_src = _mm512_loadu_ps(src_ptr);

      __m512 data_0 = _mm512_permutexvar_ps(perm_0, data_src);
      __m512 data_1 = _mm512_permutexvar_ps(perm_1, data_src);
      __m512 data_2 = _mm512_permutexvar_ps(perm_2, data_src);
      __m512 data_3 = _mm512_permutexvar_ps(perm_3, data_src);
      __m512 data_4 = _mm512_permutexvar_ps(perm_4, data_src);
      __m512 data_5 = _mm512_permutexvar_ps(perm_5, data_src);
      __m512 data_6 = _mm512_permutexvar_ps(perm_6, data_src);
      __m512 data_7 = _mm512_permutexvar_ps(perm_7, data_src);
      __m512 data_8 = _mm512_permutexvar_ps(perm_8, data_src);
      __m512 data_9 = _mm512_permutexvar_ps(perm_9, data_src);
      __m512 data_10 = _mm512_permutexvar_ps(perm_10, data_src);
      __m512 data_11 = _mm512_permutexvar_ps(perm_11, data_src);
      __m512 data_12 = _mm512_permutexvar_ps(perm_12, data_src);
      __m512 data_13 = _mm512_permutexvar_ps(perm_13, data_src);
      __m512 data_14 = _mm512_permutexvar_ps(perm_14, data_src);
      __m512 data_15 = _mm512_permutexvar_ps(perm_15, data_src);

      __m512 result0 = _mm512_mul_ps(data_0, coef_r0);
      __m512 result1 = _mm512_mul_ps(data_8, coef_r8);

      result0 = _mm512_fmadd_ps(data_1, coef_r1, result0);
      result1 = _mm512_fmadd_ps(data_9, coef_r9, result1);

      result0 = _mm512_fmadd_ps(data_2, coef_r2, result0);
      result1 = _mm512_fmadd_ps(data_10, coef_r10, result1);

      result0 = _mm512_fmadd_ps(data_3, coef_r3, result0);
      result1 = _mm512_fmadd_ps(data_11, coef_r11, result1);

      result0 = _mm512_fmadd_ps(data_4, coef_r4, result0);
      result1 = _mm512_fmadd_ps(data_12, coef_r12, result1);

      result0 = _mm512_fmadd_ps(data_5, coef_r5, result0);
      result1 = _mm512_fmadd_ps(data_13, coef_r13, result1);

      result0 = _mm512_fmadd_ps(data_6, coef_r6, result0);
      result1 = _mm512_fmadd_ps(data_14, coef_r14, result1);

      result0 = _mm512_fmadd_ps(data_7, coef_r7, result0);
      result1 = _mm512_fmadd_ps(data_15, coef_r15, result1);

      _mm512_store_ps(dst_ptr, _mm512_add_ps(result0, result1));

      dst_ptr += dst_pitch;
      src_ptr += src_pitch;
    }

    current_coeff += filter_size * 16;
  }
}



//-------- 512 bit float Verticals

void resize_v_avx512_planar_float(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel)
{
  AVS_UNUSED(bits_per_pixel);

  const int filter_size = program->filter_size;
  const float* AVS_RESTRICT current_coeff = program->pixel_coefficient_float;

  const float* src = (const float*)src8;
  float* AVS_RESTRICT dst = (float*)dst8;
  dst_pitch = dst_pitch / sizeof(float);
  src_pitch = src_pitch / sizeof(float);

  const int kernel_size = program->filter_size_real; // not the aligned
  const int kernel_size_mod2 = (kernel_size / 2) * 2; // Process pairs of rows for better efficiency
  const bool notMod2 = kernel_size_mod2 < kernel_size;

  for (int y = 0; y < target_height; y++) {
    int offset = program->pixel_offset[y];
    const float* src_ptr = src + offset * src_pitch;

    // 64 byte 16 floats (AVX512 register holds 16 floats)
    // no need for wmod8, alignment is safe 32 bytes at least - is it safe for 64 bytes ?
    for (int x = 0; x < width; x += 16) {
      __m512 result_single = _mm512_setzero_ps();
      __m512 result_single_2 = _mm512_setzero_ps();

      const float* AVS_RESTRICT src2_ptr = src_ptr + x; // __restrict here

      // Process pairs of rows for better efficiency (2 coeffs/cycle)
      // two result variables for potential parallel operation
      int i = 0;
      for (; i < kernel_size_mod2; i += 2) {
        __m512 coeff_even = _mm512_set1_ps(current_coeff[i]);
        __m512 coeff_odd = _mm512_set1_ps(current_coeff[i + 1]);

        __m512 src_even = _mm512_loadu_ps(src2_ptr);
        __m512 src_odd = _mm512_loadu_ps(src2_ptr + src_pitch);

        result_single = _mm512_fmadd_ps(src_even, coeff_even, result_single);
        result_single_2 = _mm512_fmadd_ps(src_odd, coeff_odd, result_single_2);

        src2_ptr += 2 * src_pitch;
      }

      result_single = _mm512_add_ps(result_single, result_single_2);

      // Process the last odd row if needed
      if (notMod2) {
        __m512 coeff = _mm512_set1_ps(current_coeff[i]);
        __m512 src_val = _mm512_loadu_ps(src2_ptr);
        result_single = _mm512_fmadd_ps(src_val, coeff, result_single);
      }

      _mm512_store_ps(dst + x, result_single);
    }

    dst += dst_pitch;
    current_coeff += filter_size;
  }
}

