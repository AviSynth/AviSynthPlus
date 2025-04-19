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


#include <avisynth.h>
#include <avs/config.h>

#include "../resample.h"

// Intrinsics for SSE4.1, SSSE3, SSE3, SSE2, ISSE and MMX
#include <emmintrin.h>
#include <smmintrin.h>
#include "../../core/internal.h"


#ifdef X86_32
void resize_v_mmx_planar(BYTE* dst, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel)
{
  int filter_size = program->filter_size;
  short* current_coeff = program->pixel_coefficient;

  int wMod8 = (width / 8) * 8;

  __m64 zero = _mm_setzero_si64();

  const int kernel_size = program->filter_size_real; // not the aligned
  const int kernel_size_mod2 = (kernel_size / 2) * 2;
  const bool notMod2 = kernel_size_mod2 < kernel_size;

  for (int y = 0; y < target_height; y++) {
    int offset = program->pixel_offset[y];
    const BYTE* src_ptr = src + offset * src_pitch;

    for (int x = 0; x < wMod8; x += 8) {
      __m64 result_1 = _mm_set1_pi32(8192); // Init. with rounder (16384/2 = 8192)
      __m64 result_2 = result_1;
      __m64 result_3 = result_1;
      __m64 result_4 = result_1;

      for (int i = 0; i < kernel_size_mod2; i += 2) {
        __m64 src_p1 = *(reinterpret_cast<const __m64*>(src_ptr + i * src_pitch + x)); // For detailed explanation please see SSE2 version.
        __m64 src_p2 = *(reinterpret_cast<const __m64*>(src_ptr + (i + 1) * src_pitch + x));

        __m64 src_l = _mm_unpacklo_pi8(src_p1, src_p2);
        __m64 src_h = _mm_unpackhi_pi8(src_p1, src_p2);

        __m64 src_1 = _mm_unpacklo_pi8(src_l, zero);
        __m64 src_2 = _mm_unpackhi_pi8(src_l, zero);
        __m64 src_3 = _mm_unpacklo_pi8(src_h, zero);
        __m64 src_4 = _mm_unpackhi_pi8(src_h, zero);

        // two 16 bit short coeffs
        __m64 coeff = _mm_cvtsi32_si64(*reinterpret_cast<const int*>(current_coeff + i));
        coeff = _mm_unpacklo_pi32(coeff, coeff);

        __m64 dst_1 = _mm_madd_pi16(src_1, coeff);
        __m64 dst_2 = _mm_madd_pi16(src_2, coeff);
        __m64 dst_3 = _mm_madd_pi16(src_3, coeff);
        __m64 dst_4 = _mm_madd_pi16(src_4, coeff);

        result_1 = _mm_add_pi32(result_1, dst_1);
        result_2 = _mm_add_pi32(result_2, dst_2);
        result_3 = _mm_add_pi32(result_3, dst_3);
        result_4 = _mm_add_pi32(result_4, dst_4);
      }

      if (notMod2) { // do last odd row
        __m64 src_p = *(reinterpret_cast<const __m64*>(src_ptr + kernel_size_mod2 * src_pitch + x));

        __m64 src_l = _mm_unpacklo_pi8(src_p, zero);
        __m64 src_h = _mm_unpackhi_pi8(src_p, zero);

        __m64 coeff = _mm_set1_pi16(current_coeff[kernel_size_mod2]);

        __m64 dst_ll = _mm_mullo_pi16(src_l, coeff);   // Multiply by coefficient
        __m64 dst_lh = _mm_mulhi_pi16(src_l, coeff);
        __m64 dst_hl = _mm_mullo_pi16(src_h, coeff);
        __m64 dst_hh = _mm_mulhi_pi16(src_h, coeff);

        __m64 dst_1 = _mm_unpacklo_pi16(dst_ll, dst_lh); // Unpack to 32-bit integer
        __m64 dst_2 = _mm_unpackhi_pi16(dst_ll, dst_lh);
        __m64 dst_3 = _mm_unpacklo_pi16(dst_hl, dst_hh);
        __m64 dst_4 = _mm_unpackhi_pi16(dst_hl, dst_hh);

        result_1 = _mm_add_pi32(result_1, dst_1);
        result_2 = _mm_add_pi32(result_2, dst_2);
        result_3 = _mm_add_pi32(result_3, dst_3);
        result_4 = _mm_add_pi32(result_4, dst_4);
      }

      // Divide by 16348 (FPRound)
      result_1 = _mm_srai_pi32(result_1, 14);
      result_2 = _mm_srai_pi32(result_2, 14);
      result_3 = _mm_srai_pi32(result_3, 14);
      result_4 = _mm_srai_pi32(result_4, 14);

      // Pack and store
      __m64 result_l = _mm_packs_pi32(result_1, result_2);
      __m64 result_h = _mm_packs_pi32(result_3, result_4);
      __m64 result = _mm_packs_pu16(result_l, result_h);

      *(reinterpret_cast<__m64*>(dst + x)) = result;
    }

    // Leftover
    for (int x = wMod8; x < width; x++) {
      int result = 0;
      for (int i = 0; i < kernel_size; i++) {
        result += (src_ptr + i * src_pitch)[x] * current_coeff[i];
      }
      result = ((result + 8192) / 16384);
      result = result > 255 ? 255 : result < 0 ? 0 : result;
      dst[x] = (BYTE)result;
    }

    dst += dst_pitch;
    current_coeff += filter_size;
  }

  _mm_empty();
}
#endif


void resize_v_sse2_planar(BYTE* dst, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel)
{
  AVS_UNUSED(bits_per_pixel);

  int filter_size = program->filter_size;
  short* current_coeff = program->pixel_coefficient;

  __m128i zero = _mm_setzero_si128();
  const __m128i rounder = _mm_set1_epi32(1 << (FPScale8bits - 1));

  const int kernel_size = program->filter_size_real; // not the aligned
  const int kernel_size_mod2 = (kernel_size / 2) * 2;
  const bool notMod2 = kernel_size_mod2 < kernel_size;

  for (int y = 0; y < target_height; y++) {
    int offset = program->pixel_offset[y];
    const BYTE* src_ptr = src + offset * src_pitch;

    // no need wmod8, alignment is safe at least 32
    for (int x = 0; x < width; x += 8) {
      __m128i result_single_lo = rounder;
      __m128i result_single_hi = rounder;

      const uint8_t* src2_ptr = src_ptr + x;

      // Process pairs of rows for better efficiency (2 coeffs/cycle)
      for (int i = 0; i < kernel_size_mod2; i += 2) {
        // Load two coefficients as a single packed value and broadcast
        __m128i coeff = _mm_set1_epi32(*reinterpret_cast<const int*>(current_coeff + i)); // CO|co|CO|co|CO|co|CO|co

        __m128i src_even = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src2_ptr + i * src_pitch)); // 8x 8bit pixels
        __m128i src_odd = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src2_ptr + (i + 1) * src_pitch));  // 8x 8bit pixels
        src_even = _mm_unpacklo_epi8(src_even, zero);
        src_odd = _mm_unpacklo_epi8(src_odd, zero);
        __m128i src_lo = _mm_unpacklo_epi16(src_even, src_odd);
        __m128i src_hi = _mm_unpackhi_epi16(src_even, src_odd);
        result_single_lo = _mm_add_epi32(result_single_lo, _mm_madd_epi16(src_lo, coeff)); // a*b + c
        result_single_hi = _mm_add_epi32(result_single_hi, _mm_madd_epi16(src_hi, coeff)); // a*b + c
      }

      // Process the last odd row if needed
      if (notMod2) {
        const int i = kernel_size_mod2;
        // Load a single coefficients as a single packed value and broadcast
        __m128i coeff = _mm_set1_epi16(*reinterpret_cast<const short*>(current_coeff + i)); // 0|co|0|co|0|co|0|co

        __m128i src_even = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src2_ptr + i * src_pitch)); // 8x 8bit pixels
        src_even = _mm_unpacklo_epi8(src_even, zero);
        __m128i src_lo = _mm_unpacklo_epi16(src_even, zero);
        __m128i src_hi = _mm_unpackhi_epi16(src_even, zero);
        result_single_lo = _mm_add_epi32(result_single_lo, _mm_madd_epi16(src_lo, coeff)); // a*b + c
        result_single_hi = _mm_add_epi32(result_single_hi, _mm_madd_epi16(src_hi, coeff)); // a*b + c
      }

      // scale back, store
      __m128i result_lo = result_single_lo;
      __m128i result_hi = result_single_hi;
      // shift back integer arithmetic 14 bits precision
      result_lo = _mm_srai_epi32(result_lo, FPScale8bits);
      result_hi = _mm_srai_epi32(result_hi, FPScale8bits);

      // Note: SSE4.1 simulations for SSE2: _mm_packus_epi32
      __m128i result_8x_uint16 = _MM_PACKUS_EPI32(result_lo, result_hi); // 8*32 => 8*16
      __m128i result_8x_uint8 = _mm_packus_epi16(result_8x_uint16, result_8x_uint16); // 8*16 => 8*8
      _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + x), result_8x_uint8);
    }

    dst += dst_pitch;
    current_coeff += filter_size;
  }
}


/*
void resize_v_sse2_planar(BYTE* dst, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel) // double proc per pass version
{
    AVS_UNUSED(bits_per_pixel);

    int filter_size = program->filter_size;
    short* current_coeff = program->pixel_coefficient;

    __m128i zero = _mm_setzero_si128();
    const __m128i rounder = _mm_set1_epi32(1 << (FPScale8bits - 1));

    const int kernel_size = program->filter_size_real; // not the aligned
    const int kernel_size_mod2 = (kernel_size / 2) * 2;
    const bool notMod2 = kernel_size_mod2 < kernel_size;

    for (int y = 0; y < target_height; y++) {
        int offset = program->pixel_offset[y];
        const BYTE* src_ptr = src + offset * src_pitch;

        // no need wmod8, alignment is safe at least 32
//        for (int x = 0; x < width; x += 8) {
        for (int x = 0; x < width; x += 16) {
            __m128i result_single_lo = rounder;
            __m128i result_single_hi = rounder;

            __m128i result_single_lo_2 = rounder;
            __m128i result_single_hi_2 = rounder;

            const uint8_t* src2_ptr = src_ptr + x;

            // Process pairs of rows for better efficiency (2 coeffs/cycle)
            for (int i = 0; i < kernel_size_mod2; i += 2) {
                // Load two coefficients as a single packed value and broadcast
                __m128i coeff = _mm_set1_epi32(*reinterpret_cast<const int*>(current_coeff + i)); // CO|co|CO|co|CO|co|CO|co

                __m128i src_even = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src2_ptr + i * src_pitch)); // 8x 8bit pixels
                __m128i src_odd = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src2_ptr + (i + 1) * src_pitch));  // 8x 8bit pixels

                __m128i src_even_2 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src2_ptr + 8 + i * src_pitch)); // 8x 8bit pixels
                __m128i src_odd_2 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src2_ptr + 8 + (i + 1) * src_pitch));  // 8x 8bit pixels

                src_even = _mm_unpacklo_epi8(src_even, zero);
                src_odd = _mm_unpacklo_epi8(src_odd, zero);

                src_even_2 = _mm_unpacklo_epi8(src_even_2, zero);
                src_odd_2 = _mm_unpacklo_epi8(src_odd_2, zero);

                __m128i src_lo = _mm_unpacklo_epi16(src_even, src_odd);
                __m128i src_hi = _mm_unpackhi_epi16(src_even, src_odd);

                __m128i src_lo_2 = _mm_unpacklo_epi16(src_even_2, src_odd);
                __m128i src_hi_2 = _mm_unpackhi_epi16(src_even_2, src_odd);

                result_single_lo = _mm_add_epi32(result_single_lo, _mm_madd_epi16(src_lo, coeff)); // a*b + c
                result_single_hi = _mm_add_epi32(result_single_hi, _mm_madd_epi16(src_hi, coeff)); // a*b + c

                result_single_lo_2 = _mm_add_epi32(result_single_lo_2, _mm_madd_epi16(src_lo_2, coeff)); // a*b + c
                result_single_hi_2 = _mm_add_epi32(result_single_hi_2, _mm_madd_epi16(src_hi_2, coeff)); // a*b + c

            }

            // Process the last odd row if needed
            if (notMod2) {
                const int i = kernel_size_mod2;
                // Load a single coefficients as a single packed value and broadcast
                __m128i coeff = _mm_set1_epi16(*reinterpret_cast<const short*>(current_coeff + i)); // 0|co|0|co|0|co|0|co

                __m128i src_even = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src2_ptr + i * src_pitch)); // 8x 8bit pixels

                __m128i src_even_2 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src2_ptr + 8 + i * src_pitch)); // 8x 8bit pixels

                src_even = _mm_unpacklo_epi8(src_even, zero);

                src_even_2 = _mm_unpacklo_epi8(src_even_2, zero);

                __m128i src_lo = _mm_unpacklo_epi16(src_even, zero);
                __m128i src_hi = _mm_unpackhi_epi16(src_even, zero);

                __m128i src_lo_2 = _mm_unpacklo_epi16(src_even_2, zero);
                __m128i src_hi_2 = _mm_unpackhi_epi16(src_even_2, zero);

                result_single_lo = _mm_add_epi32(result_single_lo, _mm_madd_epi16(src_lo, coeff)); // a*b + c
                result_single_hi = _mm_add_epi32(result_single_hi, _mm_madd_epi16(src_hi, coeff)); // a*b + c

                result_single_lo_2 = _mm_add_epi32(result_single_lo_2, _mm_madd_epi16(src_lo_2, coeff)); // a*b + c
                result_single_hi_2 = _mm_add_epi32(result_single_hi_2, _mm_madd_epi16(src_hi_2, coeff)); // a*b + c

            }

            // scale back, store
            __m128i result_lo = result_single_lo;
            __m128i result_hi = result_single_hi;

            __m128i result_lo_2 = result_single_lo_2;
            __m128i result_hi_2 = result_single_hi_2;

            // shift back integer arithmetic 14 bits precision
            result_lo = _mm_srai_epi32(result_lo, FPScale8bits);
            result_hi = _mm_srai_epi32(result_hi, FPScale8bits);

            result_lo_2 = _mm_srai_epi32(result_lo_2, FPScale8bits);
            result_hi_2 = _mm_srai_epi32(result_hi_2, FPScale8bits);


            // Note: SSE4.1 simulations for SSE2: _mm_packus_epi32
            __m128i result_8x_uint16 = _MM_PACKUS_EPI32(result_lo, result_hi); // 8*32 => 8*16
            __m128i result_8x_uint8 = _mm_packus_epi16(result_8x_uint16, result_8x_uint16); // 8*16 => 8*8

            __m128i result_8x_uint16_2 = _MM_PACKUS_EPI32(result_lo_2, result_hi_2); // 8*32 => 8*16
            __m128i result_8x_uint8_2 = _mm_packus_epi16(result_8x_uint16_2, result_8x_uint16_2); // 8*16 => 8*8

            _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + x), result_8x_uint8);

            _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + x + 8), result_8x_uint8_2);

        }

        dst += dst_pitch;
        current_coeff += filter_size;
    }
}
*/

// like the AVX2 version, but only 8 pixels at a time
template<bool lessthan16bit>
void resize_v_sse2_planar_uint16_t(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel)
{
  AVS_UNUSED(bits_per_pixel);

  int filter_size = program->filter_size;
  short* current_coeff = program->pixel_coefficient;
  int wMod8 = (width / 8) * 8;  // Process 8 pixels at a time instead of 16

  const __m128i zero = _mm_setzero_si128();
  
  // for 16 bits only
  const short shifttosigned_short = -32768;
  const int shiftfromsigned_int = 32768 << FPScale16bits;
  const __m128i shifttosigned = _mm_set1_epi16(-32768);
  const __m128i shiftfromsigned = _mm_set1_epi32(32768 << FPScale16bits);

  const __m128i rounder = _mm_set1_epi32(1 << (FPScale16bits - 1));

  const uint16_t* src = (uint16_t*)src8;
  uint16_t* dst = (uint16_t*)dst8;
  dst_pitch = dst_pitch / sizeof(uint16_t);
  src_pitch = src_pitch / sizeof(uint16_t);

  const int limit = (1 << bits_per_pixel) - 1;
  __m128i clamp_limit = _mm_set1_epi16((short)limit); // clamp limit for <16 bits

  const int kernel_size = program->filter_size_real; // not the aligned
  const int kernel_size_mod2 = (kernel_size / 2) * 2;
  const bool notMod2 = kernel_size_mod2 < kernel_size;

  for (int y = 0; y < target_height; y++) {
    int offset = program->pixel_offset[y];
    const uint16_t* src_ptr = src + offset * src_pitch;

    // 16 byte 8 word (half as many as AVX2)
    // no need wmod8, alignment is safe at least 32
    for (int x = 0; x < width; x += 8) {

      __m128i result_single_lo = rounder;
      __m128i result_single_hi = rounder;

      const uint16_t* src2_ptr = src_ptr + x;

      // Process pairs of rows for better efficiency (2 coeffs/cycle)
      for (int i = 0; i < kernel_size_mod2; i += 2) {
        // Load two coefficients as a single packed value and broadcast
        __m128i coeff = _mm_set1_epi32(*reinterpret_cast<const int*>(current_coeff + i)); // CO|co|CO|co|CO|co|CO|co

        __m128i src_even = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src2_ptr + i * src_pitch)); // 8x 16bit pixels
        __m128i src_odd = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src2_ptr + (i + 1) * src_pitch));  // 8x 16bit pixels
        if constexpr (!lessthan16bit) {
          src_even = _mm_add_epi16(src_even, shifttosigned);
          src_odd = _mm_add_epi16(src_odd, shifttosigned);
        }
        __m128i src_lo = _mm_unpacklo_epi16(src_even, src_odd);
        __m128i src_hi = _mm_unpackhi_epi16(src_even, src_odd);
        result_single_lo = _mm_add_epi32(result_single_lo, _mm_madd_epi16(src_lo, coeff)); // a*b + c
        result_single_hi = _mm_add_epi32(result_single_hi, _mm_madd_epi16(src_hi, coeff)); // a*b + c
      }

      // Process the last odd row if needed
      if (notMod2) {
        const int i = kernel_size_mod2;
        // Load a single coefficients as a single packed value and broadcast
        __m128i coeff = _mm_set1_epi16(*reinterpret_cast<const short*>(current_coeff + i)); // 0|co|0|co|0|co|0|co

        __m128i src_even = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src2_ptr + i * src_pitch)); // 8x 16bit pixels
        if constexpr (!lessthan16bit) {
          src_even = _mm_add_epi16(src_even, shifttosigned);
        }
        __m128i src_lo = _mm_unpacklo_epi16(src_even, zero);
        __m128i src_hi = _mm_unpackhi_epi16(src_even, zero);
        result_single_lo = _mm_add_epi32(result_single_lo, _mm_madd_epi16(src_lo, coeff)); // a*b + c
        result_single_hi = _mm_add_epi32(result_single_hi, _mm_madd_epi16(src_hi, coeff)); // a*b + c
      }

      // correct if signed, scale back, store
      __m128i result_lo = result_single_lo;
      __m128i result_hi = result_single_hi;
      if constexpr (!lessthan16bit) {
        result_lo = _mm_add_epi32(result_lo, shiftfromsigned);
        result_hi = _mm_add_epi32(result_hi, shiftfromsigned);
      }
      // shift back integer arithmetic 13 bits precision
      result_lo = _mm_srai_epi32(result_lo, FPScale16bits);
      result_hi = _mm_srai_epi32(result_hi, FPScale16bits);

      // Note: SSE4.1 simulations for SSE2: _mm_packus_epi32, _mm_min_epu16
      __m128i result_8x_uint16 = _MM_PACKUS_EPI32(result_lo, result_hi); // 8*32 => 8*16
      if constexpr (lessthan16bit) {
        result_8x_uint16 = _MM_MIN_EPU16(result_8x_uint16, clamp_limit); // extra clamp for 10-14 bit
      }
      _mm_stream_si128(reinterpret_cast<__m128i*>(dst + x), result_8x_uint16);
    }

    dst += dst_pitch;
    current_coeff += filter_size;
  }
}
//-------- 128 bit float Horizontals

AVS_FORCEINLINE static void process_two_8pixels_h_float(const float* src, int begin1, int begin2, int i, float* current_coeff, int filter_size, __m128& result1, __m128& result2) {
  __m128 data_1_low = _mm_loadu_ps(src + begin1 + i); // Load first 4 floats
  __m128 data_1_high = _mm_loadu_ps(src + begin1 + i + 4); // Load next 4 floats
  __m128 data_2_low = _mm_loadu_ps(src + begin2 + i); // Load first 4 floats
  __m128 data_2_high = _mm_loadu_ps(src + begin2 + i + 4); // Load next 4 floats

  __m128 coeff_1_low = _mm_load_ps(current_coeff); // Load first 4 coefficients
  __m128 coeff_1_high = _mm_load_ps(current_coeff + 4); // Load next 4 coefficients
  __m128 coeff_2_low = _mm_load_ps(current_coeff + filter_size); // Load first 4 coefficients for second pixel
  __m128 coeff_2_high = _mm_load_ps(current_coeff + filter_size + 4); // Load next 4 coefficients for second pixel

  result1 = _mm_add_ps(result1, _mm_mul_ps(data_1_low, coeff_1_low)); // a*b + c for first 4 floats
  result1 = _mm_add_ps(result1, _mm_mul_ps(data_1_high, coeff_1_high)); // a*b + c for next 4 floats
  result2 = _mm_add_ps(result2, _mm_mul_ps(data_2_low, coeff_2_low)); // a*b + c for first 4 floats
  result2 = _mm_add_ps(result2, _mm_mul_ps(data_2_high, coeff_2_high)); // a*b + c for next 4 floats
}

template<bool safe_aligned_mode>
AVS_FORCEINLINE static void process_two_pixels_h_float(const float* src_ptr, int begin1, int begin2, float* current_coeff, int filter_size, __m128& result1, __m128& result2, int kernel_size) {
  int ksmod8;
  // 32 bytes contain 8 floats
  if constexpr (safe_aligned_mode)
    ksmod8 = filter_size / 8 * 8;
  else
    ksmod8 = kernel_size / 8 * 8; // danger zone, scanline overread possible. Use exact unaligned kernel_size
  const float* src_ptr1 = src_ptr + begin1;
  const float* src_ptr2 = src_ptr + begin2;
  int i = 0;

  // Process 8 elements at a time
  for (; i < ksmod8; i += 8) {
    process_two_8pixels_h_float(src_ptr, begin1, begin2, i, current_coeff + i, filter_size, result1, result2);
  }

  if constexpr (!safe_aligned_mode) {
    // working with the original, unaligned kernel_size
    if (i == kernel_size) return;

    float* current_coeff2 = current_coeff + filter_size; // Points to second pixel's coefficients
    const int ksmod4 = kernel_size / 4 * 4;

    // Process 4 elements if needed
    if (i < ksmod4) {
      // Process 4 elements for first pixel
      __m128 data_1 = _mm_loadu_ps(src_ptr1 + i);
      __m128 coeff_1 = _mm_load_ps(current_coeff + i);
      __m128 temp_result1 = _mm_mul_ps(data_1, coeff_1);

      // Process 4 elements for second pixel
      __m128 data_2 = _mm_loadu_ps(src_ptr2 + i);
      __m128 coeff_2 = _mm_load_ps(current_coeff2 + i);
      __m128 temp_result2 = _mm_mul_ps(data_2, coeff_2);

      // update result vectors
      result1 = _mm_add_ps(result1, temp_result1);
      result2 = _mm_add_ps(result2, temp_result2);

      i += 4;
      if (i == kernel_size) return;
    }

    // Process remaining elements with scalar operations
    if (i < kernel_size) {
      float scalar_sum1[4] = { 0, 0, 0, 0 }; // like an __m128
      float scalar_sum2[4] = { 0, 0, 0, 0 };

      for (; i < kernel_size; i++) {
        scalar_sum1[i % 4] += src_ptr1[i] * current_coeff[i];
        scalar_sum2[i % 4] += src_ptr2[i] * current_coeff2[i];
      }

      // Convert scalar results to SIMD and add to result vectors
      __m128 temp_result1 = _mm_loadu_ps(scalar_sum1);
      __m128 temp_result2 = _mm_loadu_ps(scalar_sum2);

      result1 = _mm_add_ps(result1, temp_result1);
      result2 = _mm_add_ps(result2, temp_result2);
    }
  }
}

template<bool is_safe>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("ssse3")))
#endif
AVS_FORCEINLINE static void process_eight_pixels_h_float(const float* src, int x, float* current_coeff_base, int filter_size,
  __m128& zero128,
  float* dst,
  ResamplingProgram* program)
{
  assert(program->filter_size_alignment >= 8); // code assumes this

  float* current_coeff = current_coeff_base + x * filter_size;
  const int unaligned_kernel_size = program->filter_size_real;

  // Unrolled processing of all 8 pixels

  // 0 & 1
  __m128 result0 = zero128;
  __m128 result1 = zero128;
  int begin0 = program->pixel_offset[x + 0];
  int begin1 = program->pixel_offset[x + 1];
  process_two_pixels_h_float<is_safe>(src, begin0, begin1, current_coeff, filter_size, result0, result1, unaligned_kernel_size);
  current_coeff += 2 * filter_size;
  __m128 sumQuad12 = _mm_hadd_ps(result0, result1); // L1L1L1L1L1L1L1L1 + L2L2L2L2L2L2L2L2L2 = L1L1 L2L2 L1L1 L2L2

  // 2 & 3
  result0 = zero128;
  result1 = zero128;
  begin0 = program->pixel_offset[x + 2];
  begin1 = program->pixel_offset[x + 3];
  process_two_pixels_h_float<is_safe>(src, begin0, begin1, current_coeff, filter_size, result0, result1, unaligned_kernel_size);
  current_coeff += 2 * filter_size;
  __m128 sumQuad1234 = _mm_hadd_ps(sumQuad12, _mm_hadd_ps(result0, result1));

  __m128 result_lo = sumQuad1234; // L1 L2 L3 L4

  // 4 & 5
  result0 = zero128;
  result1 = zero128;
  begin0 = program->pixel_offset[x + 4];
  begin1 = program->pixel_offset[x + 5];
  process_two_pixels_h_float<is_safe>(src, begin0, begin1, current_coeff, filter_size, result0, result1, unaligned_kernel_size);
  current_coeff += 2 * filter_size;
  __m128 sumQuad56 = _mm_hadd_ps(result0, result1); // L1L1L1L1L1L1L1L1 + L2L2L2L2L2L2L2L2L2 = L1L1 L2L2 L1L1 L2L2

  // 6 & 7
  result0 = zero128;
  result1 = zero128;
  begin0 = program->pixel_offset[x + 6];
  begin1 = program->pixel_offset[x + 7];
  process_two_pixels_h_float<is_safe>(src, begin0, begin1, current_coeff, filter_size, result0, result1, unaligned_kernel_size);
  //current_coeff += 2 * filter_size;
  __m128 sumQuad5678 = _mm_hadd_ps(sumQuad56, _mm_hadd_ps(result0, result1));

  __m128 result_hi = sumQuad5678; // L1 L2 L3 L4

  _mm_stream_ps(reinterpret_cast<float*>(dst + x), result_lo); // 8 results at a time
  _mm_stream_ps(reinterpret_cast<float*>(dst + x + 4), result_hi); // 8 results at a time

}

#if defined(GCC) || defined(CLANG)
__attribute__((__target__("ssse3")))
#endif
void resizer_h_ssse3_generic_float(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel) {
  AVS_UNUSED(bits_per_pixel);
  int filter_size = program->filter_size;
  __m128 zero128 = _mm_setzero_ps();

  const float* src = (float*)src8;
  float* dst = (float*)dst8;
  dst_pitch = dst_pitch / sizeof(float);
  src_pitch = src_pitch / sizeof(float);

  const int wmod8 = width / 8 * 8;
  const int w_safe_mod8 = (program->overread_possible ? program->source_overread_beyond_targetx : width) / 8 * 8;

  for (int y = 0; y < height; y++) {
    float* current_coeff_base = program->pixel_coefficient_float;

    // Process safe aligned pixels
    for (int x = 0; x < w_safe_mod8; x += 8) {
      process_eight_pixels_h_float<true>(src, x, current_coeff_base, filter_size, zero128, dst, program);
    }

    // Process up to the actual kernel size instead of the aligned filter_size to prevent overreading beyond the last source pixel.
    // We assume extra offset entries were added to the p->pixel_offset array (aligned to 8 during initialization).
    // This may store 1-7 false pixels, but they are ignored since Avisynth will not read beyond the width.
    for (int x = w_safe_mod8; x < width; x += 8) {
      process_eight_pixels_h_float<false>(src, x, current_coeff_base, filter_size, zero128, dst, program);
    }

    dst += dst_pitch;
    src += src_pitch;
  }
}

//-------- 32 bit float Vertical

void resize_v_sse2_planar_float(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel)
{
  AVS_UNUSED(bits_per_pixel);

  int filter_size = program->filter_size;
  float* current_coeff = program->pixel_coefficient_float;

  const float* src = (float*)src8;
  float* dst = (float*)dst8;
  dst_pitch = dst_pitch / sizeof(float);
  src_pitch = src_pitch / sizeof(float);

  const int kernel_size = program->filter_size_real; // not the aligned

  for (int y = 0; y < target_height; y++) {
    int offset = program->pixel_offset[y];
    const float* src_ptr = src + offset * src_pitch;

    // 16 byte 4 floats (SSE2 register holds 4 floats)
    // no need wmod4, alignment is safe at least 32
    for (int x = 0; x < width; x += 4) {
      __m128 result_single = _mm_setzero_ps();
      const float* src2_ptr = src_ptr + x;

      // Process each row with its coefficient
      for (int i = 0; i < kernel_size; i++) {
        // common coefficients
        __m128 coeff = _mm_set1_ps(current_coeff[i]);
        // Load 4 float values from source
        __m128 src_val = _mm_loadu_ps(src2_ptr + i * src_pitch);

        src_val = _mm_mul_ps(src_val, coeff);
        result_single = _mm_add_ps(result_single, src_val);
      }

      // Store result
      _mm_stream_ps(dst + x, result_single);
    }

    dst += dst_pitch;
    current_coeff += filter_size;
  }
}

// -----------------------------------------------
// 8 bit Horizontal.
// Dual line processing, use template until alignment and end conditions allow.

// Based on AVX2 code

AVS_FORCEINLINE static void process_two_16pixels_h_uint8_t(const uint8_t* src, int begin1, int begin2, int i, short* current_coeff, int filter_size, __m128i& result1, __m128i& result2, const __m128i& zero128) {
  __m128i data_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + begin1 + i));
  __m128i data_2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + begin2 + i));

  // Convert 8-bit unsigned integers to 16-bit integers using unpack
  __m128i data_1_lo = _mm_unpacklo_epi8(data_1, zero128);
  __m128i data_1_hi = _mm_unpackhi_epi8(data_1, zero128);
  __m128i data_2_lo = _mm_unpacklo_epi8(data_2, zero128);
  __m128i data_2_hi = _mm_unpackhi_epi8(data_2, zero128);

  __m128i coeff_1_lo = _mm_load_si128(reinterpret_cast<const __m128i*>(current_coeff)); // 8 coeffs
  __m128i coeff_1_hi = _mm_load_si128(reinterpret_cast<const __m128i*>(current_coeff + 8)); // next 8 coeffs
  __m128i coeff_2_lo = _mm_load_si128(reinterpret_cast<const __m128i*>(current_coeff + 1 * filter_size)); // 8x second pixel's coefficients
  __m128i coeff_2_hi = _mm_load_si128(reinterpret_cast<const __m128i*>(current_coeff + 1 * filter_size + 8)); // next 8x second pixel's coefficients

  result1 = _mm_add_epi32(result1, _mm_madd_epi16(data_1_lo, coeff_1_lo));
  result1 = _mm_add_epi32(result1, _mm_madd_epi16(data_1_hi, coeff_1_hi));
  result2 = _mm_add_epi32(result2, _mm_madd_epi16(data_2_lo, coeff_2_lo));
  result2 = _mm_add_epi32(result2, _mm_madd_epi16(data_2_hi, coeff_2_hi));
}

template<bool safe_aligned_mode>
AVS_FORCEINLINE static void process_two_pixels_h_uint8(const uint8_t* src_ptr, int begin1, int begin2, short* current_coeff, int filter_size, __m128i& result1, __m128i& result2, int kernel_size, const __m128i& zero128) {
  int ksmod16;
  if constexpr (safe_aligned_mode)
    ksmod16 = filter_size / 16 * 16;
  else
    ksmod16 = kernel_size / 16 * 16; // danger zone, scanline overread possible. Use exact unaligned kernel_size
  const uint8_t* src_ptr1 = src_ptr + begin1;
  const uint8_t* src_ptr2 = src_ptr + begin2;
  int i = 0;

  // Process 16 elements at a time
  for (; i < ksmod16; i += 16) {
    process_two_16pixels_h_uint8_t(src_ptr, begin1, begin2, i, current_coeff + i, filter_size, result1, result2, zero128);
  }

  if constexpr (!safe_aligned_mode) {
    // working with the original, unaligned kernel_size
    if (i == kernel_size) return;

    short* current_coeff2 = current_coeff + filter_size; // Points to second pixel's coefficients
    const int ksmod8 = kernel_size / 8 * 8;
    const int ksmod4 = kernel_size / 4 * 4;

    // Process 8 elements if needed
    if (i < ksmod8) {
      // Process 8 elements for first pixel
      __m128i data_1 = _mm_unpacklo_epi8(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(src_ptr1 + i)), zero128);
      __m128i coeff_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(current_coeff + i));
      __m128i temp_result1 = _mm_madd_epi16(data_1, coeff_1);

      // Process 8 elements for second pixel
      __m128i data_2 = _mm_unpacklo_epi8(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(src_ptr2 + i)), zero128);
      __m128i coeff_2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(current_coeff2 + i));
      __m128i temp_result2 = _mm_madd_epi16(data_2, coeff_2);

      // update result vectors
      result1 = _mm_add_epi32(result1, temp_result1);
      result2 = _mm_add_epi32(result2, temp_result2);

      i += 8;
      if (i == kernel_size) return;
    }

    // Process 4 elements if needed
    if (i < ksmod4) {
      // Process 4 elements for first pixel
      __m128i data_1 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(*reinterpret_cast<const int*>(src_ptr1 + i)), zero128);
      __m128i coeff_1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(current_coeff + i));
      __m128i temp_result1 = _mm_madd_epi16(data_1, coeff_1);

      // Process 4 elements for second pixel
      __m128i data_2 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(*reinterpret_cast<const int*>(src_ptr2 + i)), zero128);
      __m128i coeff_2 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(current_coeff2 + i));
      __m128i temp_result2 = _mm_madd_epi16(data_2, coeff_2);

      // update result vectors
      result1 = _mm_add_epi32(result1, temp_result1);
      result2 = _mm_add_epi32(result2, temp_result2);

      i += 4;
      if (i == kernel_size) return;
    }

    // Process remaining elements with scalar operations
    if (i < kernel_size) {
      int scalar_sum1[4] = { 0, 0, 0, 0 }; // like an __m128i
      int scalar_sum2[4] = { 0, 0, 0, 0 };

      for (; i < kernel_size; i++) {
        scalar_sum1[i % 4] += src_ptr1[i] * current_coeff[i];
        scalar_sum2[i % 4] += src_ptr2[i] * current_coeff2[i];
      }

      // Convert scalar results to SIMD and add to result vectors
      __m128i temp_result1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(scalar_sum1));
      __m128i temp_result2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(scalar_sum2));

      // update result vectors
      result1 = _mm_add_epi32(result1, temp_result1);
      result2 = _mm_add_epi32(result2, temp_result2);
    }
  }
}

template<bool is_safe>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("ssse3")))
#endif
AVS_FORCEINLINE static void process_eight_pixels_h_uint8(const uint8_t* src, int x, short* current_coeff_base, int filter_size,
  __m128i& rounder128, __m128i& zero128,
  uint8_t* dst,
  ResamplingProgram* program)
{
  assert(program->filter_size_alignment >= 16); // code assumes this

  short* current_coeff = current_coeff_base + x * filter_size;
  const int unaligned_kernel_size = program->filter_size_real;

  // Unrolled processing of all 8 pixels

  // 0 & 1
  __m128i result0 = rounder128;
  __m128i result1 = rounder128;
  int begin0 = program->pixel_offset[x + 0];
  int begin1 = program->pixel_offset[x + 1];
  process_two_pixels_h_uint8<is_safe>(src, begin0, begin1, current_coeff, filter_size, result0, result1, unaligned_kernel_size, zero128);
  current_coeff += 2 * filter_size;
  __m128i sumQuad12 = _mm_hadd_epi32(result0, result1);

  // 2 & 3
  result0 = rounder128;
  result1 = rounder128;
  begin0 = program->pixel_offset[x + 2];
  begin1 = program->pixel_offset[x + 3];
  process_two_pixels_h_uint8<is_safe>(src, begin0, begin1, current_coeff, filter_size, result0, result1, unaligned_kernel_size, zero128);
  current_coeff += 2 * filter_size;
  __m128i sumQuad1234 = _mm_hadd_epi32(sumQuad12, _mm_hadd_epi32(result0, result1));

  // 4 & 5
  result0 = rounder128;
  result1 = rounder128;
  begin0 = program->pixel_offset[x + 4];
  begin1 = program->pixel_offset[x + 5];
  process_two_pixels_h_uint8<is_safe>(src, begin0, begin1, current_coeff, filter_size, result0, result1, unaligned_kernel_size, zero128);
  current_coeff += 2 * filter_size;
  __m128i sumQuad56 = _mm_hadd_epi32(result0, result1);

  // 6 & 7
  result0 = rounder128;
  result1 = rounder128;
  begin0 = program->pixel_offset[x + 6];
  begin1 = program->pixel_offset[x + 7];
  process_two_pixels_h_uint8<is_safe>(src, begin0, begin1, current_coeff, filter_size, result0, result1, unaligned_kernel_size, zero128);
  //current_coeff += 2 * filter_size;
  __m128i sumQuad5678 = _mm_hadd_epi32(sumQuad56, _mm_hadd_epi32(result0, result1));

  __m128i pix1234 = sumQuad1234;
  __m128i pix5678 = sumQuad5678;

  // scale back, shuffle, store
  __m128i result1234 = _mm_srai_epi32(pix1234, FPScale8bits);
  __m128i result5678 = _mm_srai_epi32(pix5678, FPScale8bits);
  __m128i result_2x4x_uint16_128 = _MM_PACKUS_EPI32(result1234, result5678);
  __m128i result_2x4x_uint8 = _mm_packus_epi16(result_2x4x_uint16_128, zero128);
  _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + x), result_2x4x_uint8);
}

//-------- uint8_t Horizontal (8bit)
// 4 pixels at a time. 
// ssse3: _mm_hadd_epi32
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("ssse3")))
#endif
void resizer_h_ssse3_generic(BYTE* dst, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel) {
  AVS_UNUSED(bits_per_pixel);
  int filter_size = program->filter_size;
  __m128i zero128 = _mm_setzero_si128();
  __m128i rounder128 = _mm_setr_epi32(1 << (FPScale8bits - 1), 0, 0, 0);

  const int wmod8 = width / 8 * 8;
  const int w_safe_mod8 = (program->overread_possible ? program->source_overread_beyond_targetx : width) / 8 * 8;

  for (int y = 0; y < height; y++) {
    short* current_coeff_base = program->pixel_coefficient;

    // Process safe aligned pixels
    for (int x = 0; x < w_safe_mod8; x += 8) {
      process_eight_pixels_h_uint8<true>(src, x, current_coeff_base, filter_size, rounder128, zero128, dst, program);
    }

    // Process up to the actual kernel size instead of the aligned filter_size to prevent overreading beyond the last source pixel.
    // We assume extra offset entries were added to the p->pixel_offset array (aligned to 8 during initialization).
    // This may store 1-7 false pixels, but they are ignored since Avisynth will not read beyond the width.
    for (int x = w_safe_mod8; x < width; x += 8) {
      process_eight_pixels_h_uint8<false>(src, x, current_coeff_base, filter_size, rounder128, zero128, dst, program);
    }

    dst += dst_pitch;
    src += src_pitch;
  }
}

// 16 bit Horizontal
// 2x16 pixels of two consecutive offset entries.
template<bool lessthan16bit>
AVS_FORCEINLINE static void process_two_16pixels_h_uint16_t(const uint16_t* src, int begin1, int begin2, int i, short* current_coeff, int filter_size,
  __m128i& result1, __m128i& result2, __m128i& shifttosigned) {
  __m128i data_1_low = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + begin1 + i));
  __m128i data_1_high = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + begin1 + i + 8));
  if constexpr (!lessthan16bit) {
    data_1_low = _mm_add_epi16(data_1_low, shifttosigned); // unsigned -> signed
    data_1_high = _mm_add_epi16(data_1_high, shifttosigned); // unsigned -> signed
  }
  __m128i data_2_low = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + begin2 + i));
  __m128i data_2_high = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + begin2 + i + 8));
  if constexpr (!lessthan16bit) {
    data_2_low = _mm_add_epi16(data_2_low, shifttosigned); // unsigned -> signed
    data_2_high = _mm_add_epi16(data_2_high, shifttosigned); // unsigned -> signed
  }
  __m128i coeff_1_low = _mm_load_si128(reinterpret_cast<const __m128i*>(current_coeff)); // 8 coeffs
  __m128i coeff_1_high = _mm_load_si128(reinterpret_cast<const __m128i*>(current_coeff + 8)); // 8 coeffs
  __m128i coeff_2_low = _mm_load_si128(reinterpret_cast<const __m128i*>(current_coeff + 1 * filter_size)); // 8x second pixel's coefficients
  __m128i coeff_2_high = _mm_load_si128(reinterpret_cast<const __m128i*>(current_coeff + 1 * filter_size + 8)); // 8x second pixel's coefficients

  result1 = _mm_add_epi32(result1, _mm_madd_epi16(data_1_low, coeff_1_low));
  result1 = _mm_add_epi32(result1, _mm_madd_epi16(data_1_high, coeff_1_high));
  result2 = _mm_add_epi32(result2, _mm_madd_epi16(data_2_low, coeff_2_low));
  result2 = _mm_add_epi32(result2, _mm_madd_epi16(data_2_high, coeff_2_high));
}

template<bool safe_aligned_mode, bool lessthan16bit>
AVS_FORCEINLINE static void process_two_pixels_h_uint16(const uint16_t* src_ptr, int begin1, int begin2, short* current_coeff, int filter_size, __m128i& result1, __m128i& result2, int kernel_size) {
  int ksmod16;
  if constexpr (safe_aligned_mode)
    ksmod16 = filter_size / 16 * 16;
  else
    ksmod16 = kernel_size / 16 * 16; // danger zone, scanline overread possible. Use exact unaligned kernel_size
  const uint16_t* src_ptr1 = src_ptr + begin1;
  const uint16_t* src_ptr2 = src_ptr + begin2;

  __m128i shifttosigned = _mm_set1_epi16(-32768); // for 16 bits only

  // Process 16 elements at a time
  int i = 0;
  for (; i < ksmod16; i += 16) {
    process_two_16pixels_h_uint16_t<lessthan16bit>(src_ptr, begin1, begin2, i, current_coeff + i, filter_size, result1, result2, shifttosigned);
  }

  if constexpr (!safe_aligned_mode) {
    // working with the original, unaligned kernel_size
    if (i == kernel_size) return;

    short* current_coeff2 = current_coeff + filter_size; // Points to second pixel's coefficients
    const int ksmod8 = kernel_size / 8 * 8;
    const int ksmod4 = kernel_size / 4 * 4;

    // Process 8 elements if needed
    if (i < ksmod8) {
      // Process 8 elements for first pixel
      __m128i data_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_ptr1 + i));
      if constexpr (!lessthan16bit)
        data_1 = _mm_add_epi16(data_1, shifttosigned); // unsigned -> signed
      __m128i coeff_1 = _mm_load_si128(reinterpret_cast<const __m128i*>(current_coeff + i));
      __m128i temp_result1 = _mm_madd_epi16(data_1, coeff_1);

      // Process 8 elements for second pixel
      __m128i data_2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_ptr2 + i));
      if constexpr (!lessthan16bit)
        data_2 = _mm_add_epi16(data_2, shifttosigned); // unsigned -> signed
      __m128i coeff_2 = _mm_load_si128(reinterpret_cast<const __m128i*>(current_coeff2 + i));
      __m128i temp_result2 = _mm_madd_epi16(data_2, coeff_2);

      // update result vectors
      result1 = _mm_add_epi32(result1, temp_result1);
      result2 = _mm_add_epi32(result2, temp_result2);

      i += 8;
      if (i == kernel_size) return;
    }

    // Process 4 elements if needed
    if (i < ksmod4) {
      // Process 4 elements for first pixel
      __m128i data_1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src_ptr1 + i));
      if constexpr (!lessthan16bit)
        data_1 = _mm_add_epi16(data_1, shifttosigned); // unsigned -> signed
      __m128i coeff_1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(current_coeff + i));
      __m128i temp_result1 = _mm_madd_epi16(data_1, coeff_1);

      // Process 4 elements for second pixel
      __m128i data_2 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src_ptr2 + i));
      if constexpr (!lessthan16bit)
        data_2 = _mm_add_epi16(data_2, shifttosigned); // unsigned -> signed
      __m128i coeff_2 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(current_coeff2 + i));
      __m128i temp_result2 = _mm_madd_epi16(data_2, coeff_2);

      // update result vectors
      result1 = _mm_add_epi32(result1, temp_result1);
      result2 = _mm_add_epi32(result2, temp_result2);

      i += 4;
      if (i == kernel_size) return;
    }

    // Process remaining elements with scalar operations
    if (i < kernel_size) {
      int scalar_sum1[4] = { 0, 0, 0, 0 }; // like an __m128i
      int scalar_sum2[4] = { 0, 0, 0, 0 };

      for (; i < kernel_size; i++) {
        uint16_t pix1 = src_ptr1[i];
        uint16_t pix2 = src_ptr2[i];

        if constexpr (!lessthan16bit) {
          pix1 -= 32768;
          pix2 -= 32768;
        }

        scalar_sum1[i % 4] += (short)pix1 * current_coeff[i];
        scalar_sum2[i % 4] += (short)pix2 * current_coeff2[i];
      }

      // Convert scalar results to SIMD and add to result vectors
      __m128i temp_result1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(scalar_sum1));
      __m128i temp_result2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(scalar_sum2));

      result1 = _mm_add_epi32(result1, temp_result1);
      result2 = _mm_add_epi32(result2, temp_result2);
    }
  }
}

template<bool is_safe, bool lessthan16bit>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("ssse3")))
#endif
AVS_FORCEINLINE static void process_eight_pixels_h_uint16(const uint16_t* src, int x, short* current_coeff_base, int filter_size,
  __m128i& rounder128, __m128i& zero128,
  uint16_t* dst,
  ResamplingProgram* program,
  __m128i& clamp_limit)
{
  assert(program->filter_size_alignment >= 16); // code assumes this

  short* current_coeff = current_coeff_base + x * filter_size;
  const int unaligned_kernel_size = program->filter_size_real;

  const __m128i shiftfromsigned = _mm_set1_epi32(+32768 << FPScale16bits); // yes, 32 bit data. for 16 bits only
  // Unrolled processing of all 8 pixels

  // 0 & 1
  __m128i result0 = rounder128;
  __m128i result1 = rounder128;
  int begin0 = program->pixel_offset[x + 0];
  int begin1 = program->pixel_offset[x + 1];
  process_two_pixels_h_uint16<is_safe, lessthan16bit>(src, begin0, begin1, current_coeff, filter_size, result0, result1, unaligned_kernel_size);
  current_coeff += 2 * filter_size;
  __m128i sumQuad12 = _mm_hadd_epi32(result0, result1);

  // 2 & 3
  result0 = rounder128;
  result1 = rounder128;
  begin0 = program->pixel_offset[x + 2];
  begin1 = program->pixel_offset[x + 3];
  process_two_pixels_h_uint16<is_safe, lessthan16bit>(src, begin0, begin1, current_coeff, filter_size, result0, result1, unaligned_kernel_size);
  current_coeff += 2 * filter_size;
  __m128i sumQuad1234 = _mm_hadd_epi32(sumQuad12, _mm_hadd_epi32(result0, result1));

  // 4 & 5
  result0 = rounder128;
  result1 = rounder128;
  begin0 = program->pixel_offset[x + 4];
  begin1 = program->pixel_offset[x + 5];
  process_two_pixels_h_uint16<is_safe, lessthan16bit>(src, begin0, begin1, current_coeff, filter_size, result0, result1, unaligned_kernel_size);
  current_coeff += 2 * filter_size;
  __m128i sumQuad56 = _mm_hadd_epi32(result0, result1);

  // 6 & 7
  result0 = rounder128;
  result1 = rounder128;
  begin0 = program->pixel_offset[x + 6];
  begin1 = program->pixel_offset[x + 7];
  process_two_pixels_h_uint16<is_safe, lessthan16bit>(src, begin0, begin1, current_coeff, filter_size, result0, result1, unaligned_kernel_size);
  //current_coeff += 2 * filter_size;
  __m128i sumQuad5678 = _mm_hadd_epi32(sumQuad56, _mm_hadd_epi32(result0, result1));

  __m128i pix1234 = sumQuad1234;
  __m128i pix5678 = sumQuad5678;

  // correct if signed, scale back, store
  if constexpr (!lessthan16bit) {
    pix1234 = _mm_add_epi32(pix1234, shiftfromsigned);
    pix5678 = _mm_add_epi32(pix5678, shiftfromsigned);
  }  
  // scale back, shuffle, store
  __m128i result_lo = _mm_srai_epi32(pix1234, FPScale16bits);
  __m128i result_hi = _mm_srai_epi32(pix5678, FPScale16bits);
  __m128i result_2x4x_uint16_128 = _MM_PACKUS_EPI32(result_lo, result_hi);
 
  if constexpr (lessthan16bit)
    result_2x4x_uint16_128 = _MM_MIN_EPU16(result_2x4x_uint16_128, clamp_limit); // extra clamp for 10-14 bits

  _mm_store_si128(reinterpret_cast<__m128i*>(dst + x), result_2x4x_uint16_128);

}

//-------- uint16_t Horizontal (16bit)
// 8 pixels at a time. 
// ssse3: _mm_hadd_epi32
template<bool lessthan16bit>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("ssse3")))
#endif
void resizer_h_ssse3_generic_uint16_t(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel) {
  int filter_size = program->filter_size;
  __m128i zero128 = _mm_setzero_si128();
  __m128i rounder128 = _mm_setr_epi32(1 << (FPScale16bits - 1), 0, 0, 0);

  __m128i clamp_limit = _mm_set1_epi16((short)((1 << bits_per_pixel) - 1)); // clamp limit for <16 bits

  const uint16_t* src = reinterpret_cast<const uint16_t*>(src8);
  uint16_t* dst = reinterpret_cast<uint16_t*>(dst8);
  dst_pitch /= sizeof(uint16_t);
  src_pitch /= sizeof(uint16_t);

  const int wmod8 = width / 8 * 8;
  const int w_safe_mod8 = (program->overread_possible ? program->source_overread_beyond_targetx : width) / 8 * 8;

  for (int y = 0; y < height; y++) {
    short* current_coeff_base = program->pixel_coefficient;

    // Process safe aligned pixels
    for (int x = 0; x < w_safe_mod8; x += 8) {
      process_eight_pixels_h_uint16<true, lessthan16bit>(src, x, current_coeff_base, filter_size, rounder128, zero128, dst, program, clamp_limit);
    }

    // Process up to the actual kernel size instead of the aligned filter_size to prevent overreading beyond the last source pixel.
    // We assume extra offset entries were added to the p->pixel_offset array (aligned to 8 during initialization).
    // This may store 1-7 false pixels, but they are ignored since Avisynth will not read beyond the width.
    for (int x = w_safe_mod8; x < width; x += 8) {
      process_eight_pixels_h_uint16<false, lessthan16bit>(src, x, current_coeff_base, filter_size, rounder128, zero128, dst, program, clamp_limit);
    }

    dst += dst_pitch;
    src += src_pitch;
  }
}

// instantiate
template void resizer_h_ssse3_generic_uint16_t<false>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resizer_h_ssse3_generic_uint16_t<true>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);

template void resize_v_sse2_planar_uint16_t<false>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel);
template void resize_v_sse2_planar_uint16_t<true>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel);

