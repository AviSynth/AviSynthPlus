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

