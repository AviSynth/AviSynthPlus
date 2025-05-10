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

/*
void resize_v_sse2_planar(BYTE* dst8, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel)
{
  AVS_UNUSED(bits_per_pixel);

  const int filter_size = program->filter_size;
  uint8_t* AVS_RESTRICT dst = (uint8_t * AVS_RESTRICT)dst8;
  const short* AVS_RESTRICT current_coeff = program->pixel_coefficient;

  const __m128i zero = _mm_setzero_si128();
  const __m128i rounder = _mm_set1_epi32(1 << (FPScale8bits - 1));

  const int kernel_size = program->filter_size_real; // not the aligned
  const int kernel_size_mod2 = (kernel_size / 2) * 2;
  const bool notMod2 = kernel_size_mod2 < kernel_size;

  for (int y = 0; y < target_height; y++) {
    const int offset = program->pixel_offset[y];
    const BYTE* src_ptr = src + offset * src_pitch;

    // no need wmod8, alignment is safe at least 32
    for (int x = 0; x < width; x += 8) {
      __m128i result_single_lo = rounder;
      __m128i result_single_hi = rounder;

      const uint8_t* AVS_RESTRICT src2_ptr = src_ptr + x;

      // Process pairs of rows for better efficiency (2 coeffs/cycle)
      int i = 0;
      for (; i < kernel_size_mod2; i += 2) {
        // Load _two_ coefficients as a single packed value and broadcast
        __m128i coeff = _mm_set1_epi32(*reinterpret_cast<const int*>(current_coeff + i)); // CO|co|CO|co|CO|co|CO|co

        __m128i src_even = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src2_ptr)); // 8x 8bit pixels
        __m128i src_odd = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src2_ptr + src_pitch));  // 8x 8bit pixels
        src_even = _mm_unpacklo_epi8(src_even, zero);
        src_odd = _mm_unpacklo_epi8(src_odd, zero);
        __m128i src_lo = _mm_unpacklo_epi16(src_even, src_odd);
        __m128i src_hi = _mm_unpackhi_epi16(src_even, src_odd);
        result_single_lo = _mm_add_epi32(result_single_lo, _mm_madd_epi16(src_lo, coeff)); // a*b + c
        result_single_hi = _mm_add_epi32(result_single_hi, _mm_madd_epi16(src_hi, coeff)); // a*b + c
        src2_ptr += 2 * src_pitch; // Move to the next pair of rows
      }

      // Process the last odd row if needed
      if (notMod2) {
        // Load a single coefficients as a single packed value and broadcast
        __m128i coeff = _mm_set1_epi16(*reinterpret_cast<const short*>(current_coeff + i)); // 0|co|0|co|0|co|0|co

        __m128i src_even = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src2_ptr)); // 8x 8bit pixels
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
*/

void resize_v_sse2_planar(BYTE* dst8, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel)
{
    AVS_UNUSED(bits_per_pixel);

    const int filter_size = program->filter_size;
    uint8_t* AVS_RESTRICT dst = (uint8_t * AVS_RESTRICT)dst8;
    const short* AVS_RESTRICT current_coeff = program->pixel_coefficient;

    const __m128i zero = _mm_setzero_si128();
    const __m128i rounder = _mm_set1_epi32(1 << (FPScale8bits - 1));

    const int kernel_size = program->filter_size_real; // not the aligned
    const int kernel_size_mod2 = (kernel_size / 2) * 2;
    const bool notMod2 = kernel_size_mod2 < kernel_size;

    for (int y = 0; y < target_height; y++) {
        const int offset = program->pixel_offset[y];
        const BYTE* src_ptr = src + offset * src_pitch;

        // no need wmod8, alignment is safe at least 32
        for (int x = 0; x < width; x += 16) { // +=8
            __m128i result_single_lo = rounder;
            __m128i result_single_hi = rounder;

            __m128i result_single2_lo = rounder;
            __m128i result_single2_hi = rounder;

            const uint8_t* AVS_RESTRICT src2_ptr = src_ptr + x;

            // Process pairs of rows for better efficiency (2 coeffs/cycle)
            int i = 0;
            for (; i < kernel_size_mod2; i += 2) {
                // Load _two_ coefficients as a single packed value and broadcast
                __m128i coeff = _mm_set1_epi32(*reinterpret_cast<const int*>(current_coeff + i)); // CO|co|CO|co|CO|co|CO|co

                __m128i src_even = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src2_ptr)); // 8x 8bit pixels
                __m128i src_odd = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src2_ptr + src_pitch));  // 8x 8bit pixels

                __m128i src_even2 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src2_ptr + 8)); // 8x 8bit pixels
                __m128i src_odd2 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src2_ptr + src_pitch + 8));  // 8x 8bit pixels

                src_even = _mm_unpacklo_epi8(src_even, zero);
                src_odd = _mm_unpacklo_epi8(src_odd, zero);

                src_even2 = _mm_unpacklo_epi8(src_even2, zero);
                src_odd2 = _mm_unpacklo_epi8(src_odd2, zero);

                __m128i src_lo = _mm_unpacklo_epi16(src_even, src_odd);
                __m128i src_hi = _mm_unpackhi_epi16(src_even, src_odd);

                __m128i src_lo2 = _mm_unpacklo_epi16(src_even2, src_odd2);
                __m128i src_hi2 = _mm_unpackhi_epi16(src_even2, src_odd2);

                result_single_lo = _mm_add_epi32(result_single_lo, _mm_madd_epi16(src_lo, coeff)); // a*b + c
                result_single_hi = _mm_add_epi32(result_single_hi, _mm_madd_epi16(src_hi, coeff)); // a*b + c

                result_single2_lo = _mm_add_epi32(result_single2_lo, _mm_madd_epi16(src_lo2, coeff)); // a*b + c
                result_single2_hi = _mm_add_epi32(result_single2_hi, _mm_madd_epi16(src_hi2, coeff)); // a*b + c

                src2_ptr += 2 * src_pitch; // Move to the next pair of rows
            }

            // Process the last odd row if needed
            if (notMod2) {
                // Load a single coefficients as a single packed value and broadcast
                __m128i coeff = _mm_set1_epi16(*reinterpret_cast<const short*>(current_coeff + i)); // 0|co|0|co|0|co|0|co

                __m128i src_even = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src2_ptr)); // 8x 8bit pixels
                __m128i src_even2 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src2_ptr + 8)); // 8x 8bit pixels

                src_even = _mm_unpacklo_epi8(src_even, zero);
                src_even2 = _mm_unpacklo_epi8(src_even2, zero);

                __m128i src_lo = _mm_unpacklo_epi16(src_even, zero);
                __m128i src_hi = _mm_unpackhi_epi16(src_even, zero);
                __m128i src_lo2 = _mm_unpacklo_epi16(src_even2, zero);
                __m128i src_hi2 = _mm_unpackhi_epi16(src_even2, zero);

                result_single_lo = _mm_add_epi32(result_single_lo, _mm_madd_epi16(src_lo, coeff)); // a*b + c
                result_single_hi = _mm_add_epi32(result_single_hi, _mm_madd_epi16(src_hi, coeff)); // a*b + c
                result_single2_lo = _mm_add_epi32(result_single2_lo, _mm_madd_epi16(src_lo2, coeff)); // a*b + c
                result_single2_hi = _mm_add_epi32(result_single2_hi, _mm_madd_epi16(src_hi2, coeff)); // a*b + c

            }

            // scale back, store
            __m128i result_lo = result_single_lo;
            __m128i result_hi = result_single_hi;
            __m128i result_lo2 = result_single2_lo;
            __m128i result_hi2 = result_single2_hi;

            // shift back integer arithmetic 14 bits precision
            result_lo = _mm_srai_epi32(result_lo, FPScale8bits);
            result_hi = _mm_srai_epi32(result_hi, FPScale8bits);
            result_lo2 = _mm_srai_epi32(result_lo2, FPScale8bits);
            result_hi2 = _mm_srai_epi32(result_hi2, FPScale8bits);

            // Note: SSE4.1 simulations for SSE2: _mm_packus_epi32
            __m128i result_8x_uint16 = _MM_PACKUS_EPI32(result_lo, result_hi); // 8*32 => 8*16
            __m128i result_8x_uint8 = _mm_packus_epi16(result_8x_uint16, result_8x_uint16); // 8*16 => 8*8
            _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + x), result_8x_uint8);

            __m128i result2_8x_uint16 = _MM_PACKUS_EPI32(result_lo2, result_hi2); // 8*32 => 8*16
            __m128i result2_8x_uint8 = _mm_packus_epi16(result2_8x_uint16, result2_8x_uint16); // 8*16 => 8*8
            _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + x + 8), result2_8x_uint8);

        }

        dst += dst_pitch;
        current_coeff += filter_size;
    }
}

// like the AVX2 version, but only 8 pixels at a time
template<bool lessthan16bit>
void resize_v_sse2_planar_uint16_t(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel)
{
  AVS_UNUSED(bits_per_pixel);

  const int filter_size = program->filter_size;
  const short* AVS_RESTRICT current_coeff = program->pixel_coefficient;

  const __m128i zero = _mm_setzero_si128();
  
  // for 16 bits only
  [[maybe_unused]] const __m128i shifttosigned = _mm_set1_epi16(-32768);
  [[maybe_unused]] const __m128i shiftfromsigned = _mm_set1_epi32(32768 << FPScale16bits);

  const __m128i rounder = _mm_set1_epi32(1 << (FPScale16bits - 1));

  const uint16_t* src = (uint16_t*)src8;
  uint16_t* AVS_RESTRICT dst = (uint16_t* AVS_RESTRICT)dst8;
  dst_pitch = dst_pitch / sizeof(uint16_t);
  src_pitch = src_pitch / sizeof(uint16_t);

  const int limit = (1 << bits_per_pixel) - 1;
  __m128i clamp_limit = _mm_set1_epi16((short)limit); // clamp limit for <16 bits

  const int kernel_size = program->filter_size_real; // not the aligned
  const int kernel_size_mod2 = (kernel_size / 2) * 2;
  const bool notMod2 = kernel_size_mod2 < kernel_size;

  for (int y = 0; y < target_height; y++) {
    const int offset = program->pixel_offset[y];
    const uint16_t* src_ptr = src + offset * src_pitch;

    // 16 byte 8 word (half as many as AVX2)
    // no need wmod8, alignment is safe at least 32
    for (int x = 0; x < width; x += 8) {

      __m128i result_single_lo = rounder;
      __m128i result_single_hi = rounder;

      const uint16_t* AVS_RESTRICT src2_ptr = src_ptr + x;

      // Process pairs of rows for better efficiency (2 coeffs/cycle)
      int i = 0;
      for (; i < kernel_size_mod2; i += 2) {
        // Load _two_ coefficients as a single packed value and broadcast
        __m128i coeff = _mm_set1_epi32(*reinterpret_cast<const int*>(current_coeff + i)); // CO|co|CO|co|CO|co|CO|co

        __m128i src_even = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src2_ptr)); // 8x 16bit pixels
        __m128i src_odd = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src2_ptr + src_pitch));  // 8x 16bit pixels
        if constexpr (!lessthan16bit) {
          src_even = _mm_add_epi16(src_even, shifttosigned);
          src_odd = _mm_add_epi16(src_odd, shifttosigned);
        }
        __m128i src_lo = _mm_unpacklo_epi16(src_even, src_odd);
        __m128i src_hi = _mm_unpackhi_epi16(src_even, src_odd);
        result_single_lo = _mm_add_epi32(result_single_lo, _mm_madd_epi16(src_lo, coeff)); // a*b + c
        result_single_hi = _mm_add_epi32(result_single_hi, _mm_madd_epi16(src_hi, coeff)); // a*b + c
        src2_ptr += 2 * src_pitch; // Move to the next pair of rows
      }

      // Process the last odd row if needed
      if (notMod2) {
        // Load a single coefficients as a single packed value and broadcast
        __m128i coeff = _mm_set1_epi16(*reinterpret_cast<const short*>(current_coeff + i)); // 0|co|0|co|0|co|0|co

        __m128i src_even = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src2_ptr)); // 8x 16bit pixels
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

// Process each row with its coefficient
void resize_v_sse2_planar_float(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel)
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

    // use 8 pixels, like AVX2, by utilizing 2x2 ps registers (speed)
    for (int x = 0; x < width; x += 8) {
      __m128 result_single_even = _mm_setzero_ps();
      __m128 result_single_odd = _mm_setzero_ps();
      __m128 result_single_even_b = _mm_setzero_ps();
      __m128 result_single_odd_b = _mm_setzero_ps();

      const float* AVS_RESTRICT src2_ptr = src_ptr + x; // __restrict here

      // Process pairs of rows for better efficiency (2 coeffs/cycle)
      int i = 0;
      for (; i < kernel_size_mod2; i += 2) {
        __m128 coeff_even = _mm_set1_ps(current_coeff[i]);
        __m128 coeff_odd = _mm_set1_ps(current_coeff[i + 1]);

        __m128 src_even = _mm_loadu_ps(src2_ptr);
        __m128 src_odd = _mm_loadu_ps(src2_ptr + src_pitch);

        __m128 mul_even = _mm_mul_ps(src_even, coeff_even);
        __m128 mul_odd = _mm_mul_ps(src_odd, coeff_odd);

        result_single_even = _mm_add_ps(result_single_even, mul_even);
        result_single_odd = _mm_add_ps(result_single_odd, mul_odd);

        __m128 src_even_b = _mm_loadu_ps(src2_ptr + 4);
        __m128 src_odd_b = _mm_loadu_ps(src2_ptr + 4 + src_pitch);

        __m128 mul_even_b = _mm_mul_ps(src_even_b, coeff_even);
        __m128 mul_odd_b = _mm_mul_ps(src_odd_b, coeff_odd);

        result_single_even_b = _mm_add_ps(result_single_even_b, mul_even_b);
        result_single_odd_b = _mm_add_ps(result_single_odd_b, mul_odd_b);

        src2_ptr += 2 * src_pitch;
      }

      result_single_even = _mm_add_ps(result_single_even, result_single_odd);
      result_single_even_b = _mm_add_ps(result_single_even_b, result_single_odd_b);

      // Process the last odd row if needed  
      if (notMod2) {
        __m128 coeff = _mm_set1_ps(current_coeff[i]);
        __m128 src_val = _mm_loadu_ps(src2_ptr);
        __m128 src_val_b = _mm_loadu_ps(src2_ptr + 4);

        result_single_even = _mm_add_ps(result_single_even, _mm_mul_ps(src_val, coeff));
        result_single_even_b = _mm_add_ps(result_single_even_b, _mm_mul_ps(src_val_b, coeff));
      }

      // Store result  
      _mm_stream_ps(dst + x, result_single_even);
      _mm_stream_ps(dst + x + 4, result_single_even_b);
    }

    dst += dst_pitch;
    current_coeff += filter_size;
  }
}

// -----------------------------------------------
// 8 bit Horizontal.
// Dual line processing, use template until alignment and end conditions allow.

// Based on AVX2 code, but without the filter_size alignment template

template<typename pixel_t, bool lessthan16bit>
AVS_FORCEINLINE static void process_two_16pixels_h_uint8_16_core(const pixel_t* AVS_RESTRICT src, int begin1, int begin2, int i, const short* AVS_RESTRICT current_coeff, int filter_size, __m128i& result1, __m128i& result2, 
  const __m128i& shifttosigned_or_zero128) {

  __m128i data_1_lo, data_1_hi, data_2_lo, data_2_hi;

  if constexpr (sizeof(pixel_t) == 1) {
    // pixel_t is uint8_t
  __m128i data_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + begin1 + i));
  __m128i data_2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + begin2 + i));

    data_1_lo = _mm_unpacklo_epi8(data_1, shifttosigned_or_zero128);
    data_1_hi = _mm_unpackhi_epi8(data_1, shifttosigned_or_zero128);
    data_2_lo = _mm_unpacklo_epi8(data_2, shifttosigned_or_zero128);
    data_2_hi = _mm_unpackhi_epi8(data_2, shifttosigned_or_zero128);
  }
  else {
    // pixel_t is uint16_t, at exact 16 bit size an unsigned -> signed 16 bit conversion needed
    data_1_lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + begin1 + i));
    data_1_hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + begin1 + i + 8));
    if constexpr (!lessthan16bit) {
      data_1_lo = _mm_add_epi16(data_1_lo, shifttosigned_or_zero128);
      data_1_hi = _mm_add_epi16(data_1_hi, shifttosigned_or_zero128);
    }
    data_2_lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + begin2 + i));
    data_2_hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + begin2 + i + 8));
    if constexpr (!lessthan16bit) {
      data_2_lo = _mm_add_epi16(data_2_lo, shifttosigned_or_zero128);
      data_2_hi = _mm_add_epi16(data_2_hi, shifttosigned_or_zero128);
    }
  }

  __m128i coeff_1_lo = _mm_load_si128(reinterpret_cast<const __m128i*>(current_coeff)); // 8 coeffs
  __m128i coeff_1_hi = _mm_load_si128(reinterpret_cast<const __m128i*>(current_coeff + 8)); // next 8 coeffs
  __m128i coeff_2_lo = _mm_load_si128(reinterpret_cast<const __m128i*>(current_coeff + 1 * filter_size)); // 8x second pixel's coefficients
  __m128i coeff_2_hi = _mm_load_si128(reinterpret_cast<const __m128i*>(current_coeff + 1 * filter_size + 8)); // next 8x second pixel's coefficients

  result1 = _mm_add_epi32(result1, _mm_madd_epi16(data_1_lo, coeff_1_lo));
  result1 = _mm_add_epi32(result1, _mm_madd_epi16(data_1_hi, coeff_1_hi));
  result2 = _mm_add_epi32(result2, _mm_madd_epi16(data_2_lo, coeff_2_lo));
  result2 = _mm_add_epi32(result2, _mm_madd_epi16(data_2_hi, coeff_2_hi));
}

template<bool safe_aligned_mode, typename pixel_t, bool lessthan16bit>
AVS_FORCEINLINE static void process_two_pixels_h_uint8_16(const pixel_t* AVS_RESTRICT src_ptr, int begin1, int begin2, const short* AVS_RESTRICT current_coeff, int filter_size, __m128i& result1, __m128i& result2, int kernel_size, 
  const __m128i& shifttosigned_or_zero128) {
  int ksmod16;
  if constexpr (safe_aligned_mode)
    ksmod16 = filter_size / 16 * 16;
  else
    ksmod16 = kernel_size / 16 * 16; // danger zone, scanline overread possible. Use exact unaligned kernel_size
  const pixel_t* src_ptr1 = src_ptr + begin1;
  const pixel_t* src_ptr2 = src_ptr + begin2;
  int i = 0;

  // Process 16 elements at a time
  for (; i < ksmod16; i += 16) {
    process_two_16pixels_h_uint8_16_core<pixel_t, lessthan16bit>(src_ptr, begin1, begin2, i, current_coeff + i, filter_size, result1, result2, shifttosigned_or_zero128);
  }

  if constexpr (!safe_aligned_mode) {
    // working with the original, unaligned kernel_size
    if (i == kernel_size) return;

    const short* current_coeff2 = current_coeff + filter_size; // Points to second pixel's coefficients
    const int ksmod8 = kernel_size / 8 * 8;
    const int ksmod4 = kernel_size / 4 * 4;

    // Process 8 elements if needed
    if (i < ksmod8) {
      // Process 8 elements for first pixel
      __m128i data_1;
      if constexpr(sizeof(pixel_t) == 1)
        data_1 = _mm_unpacklo_epi8(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(src_ptr1 + i)), shifttosigned_or_zero128);
      else {
        // uint16_t
        data_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_ptr1 + i));
        if constexpr (!lessthan16bit)
          data_1 = _mm_add_epi16(data_1, shifttosigned_or_zero128); // unsigned -> signed
      }

      __m128i coeff_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(current_coeff + i));
      __m128i temp_result1 = _mm_madd_epi16(data_1, coeff_1);

      // Process 8 elements for second pixel
      __m128i data_2;
      if constexpr (sizeof(pixel_t) == 1)
        data_2 = _mm_unpacklo_epi8(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(src_ptr2 + i)), shifttosigned_or_zero128);
      else {
        // uint16_t
        data_2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_ptr2 + i));
        if constexpr (!lessthan16bit)
          data_2 = _mm_add_epi16(data_2, shifttosigned_or_zero128); // unsigned -> signed
      }

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
      __m128i data_1;
      if constexpr (sizeof(pixel_t) == 1)
        data_1 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(*reinterpret_cast<const int*>(src_ptr1 + i)), shifttosigned_or_zero128);
      else {
        data_1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src_ptr1 + i));
        if constexpr (!lessthan16bit)
          data_1 = _mm_add_epi16(data_1, shifttosigned_or_zero128); // unsigned -> signed
      }

      __m128i coeff_1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(current_coeff + i));
      __m128i temp_result1 = _mm_madd_epi16(data_1, coeff_1);

      // Process 4 elements for second pixel
      __m128i data_2;
      if constexpr (sizeof(pixel_t) == 1)
        data_2 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(*reinterpret_cast<const int*>(src_ptr2 + i)), shifttosigned_or_zero128);
      else {
        data_2 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src_ptr2 + i));
        if constexpr (!lessthan16bit)
          data_2 = _mm_add_epi16(data_2, shifttosigned_or_zero128); // unsigned -> signed
      }
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
        if constexpr (sizeof(pixel_t) == 1) {
        scalar_sum1[i % 4] += src_ptr1[i] * current_coeff[i];
        scalar_sum2[i % 4] += src_ptr2[i] * current_coeff2[i];
        } else {
          uint16_t pix1 = src_ptr1[i];
          uint16_t pix2 = src_ptr2[i];

          if constexpr (!lessthan16bit) {
            pix1 -= 32768;
            pix2 -= 32768;
          }

          scalar_sum1[i % 4] += (short)pix1 * current_coeff[i];
          scalar_sum2[i % 4] += (short)pix2 * current_coeff2[i];
        }
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

template<bool is_safe, typename pixel_t, bool lessthan16bit>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("ssse3")))
#endif
AVS_FORCEINLINE static void process_eight_pixels_h_uint8_16(const pixel_t* AVS_RESTRICT src, int x, const short* current_coeff_base, int filter_size,
  __m128i& rounder128, __m128i& shifttosigned_or_zero128, __m128i& clamp_limit,
  pixel_t* AVS_RESTRICT dst,
  ResamplingProgram* program)
{
  assert(program->filter_size_alignment >= 16); // code assumes this

  const short* AVS_RESTRICT current_coeff = current_coeff_base + x * filter_size;
  const int unaligned_kernel_size = program->filter_size_real;

  // Unrolled processing of all 8 pixels

  // 0 & 1
  __m128i result0 = rounder128;
  __m128i result1 = rounder128;
  int begin0 = program->pixel_offset[x + 0];
  int begin1 = program->pixel_offset[x + 1];
  process_two_pixels_h_uint8_16<is_safe, pixel_t, lessthan16bit>(src, begin0, begin1, current_coeff, filter_size, result0, result1, unaligned_kernel_size, shifttosigned_or_zero128);
  current_coeff += 2 * filter_size;
  __m128i sumQuad12 = _mm_hadd_epi32(result0, result1);

  // 2 & 3
  result0 = rounder128;
  result1 = rounder128;
  begin0 = program->pixel_offset[x + 2];
  begin1 = program->pixel_offset[x + 3];
  process_two_pixels_h_uint8_16<is_safe, pixel_t, lessthan16bit>(src, begin0, begin1, current_coeff, filter_size, result0, result1, unaligned_kernel_size, shifttosigned_or_zero128);
  current_coeff += 2 * filter_size;
  __m128i sumQuad1234 = _mm_hadd_epi32(sumQuad12, _mm_hadd_epi32(result0, result1));

  // 4 & 5
  result0 = rounder128;
  result1 = rounder128;
  begin0 = program->pixel_offset[x + 4];
  begin1 = program->pixel_offset[x + 5];
  process_two_pixels_h_uint8_16<is_safe, pixel_t, lessthan16bit>(src, begin0, begin1, current_coeff, filter_size, result0, result1, unaligned_kernel_size, shifttosigned_or_zero128);
  current_coeff += 2 * filter_size;
  __m128i sumQuad56 = _mm_hadd_epi32(result0, result1);

  // 6 & 7
  result0 = rounder128;
  result1 = rounder128;
  begin0 = program->pixel_offset[x + 6];
  begin1 = program->pixel_offset[x + 7];
  process_two_pixels_h_uint8_16<is_safe, pixel_t, lessthan16bit>(src, begin0, begin1, current_coeff, filter_size, result0, result1, unaligned_kernel_size, shifttosigned_or_zero128);
  //current_coeff += 2 * filter_size;
  __m128i sumQuad5678 = _mm_hadd_epi32(sumQuad56, _mm_hadd_epi32(result0, result1));

  __m128i pix1234 = sumQuad1234;
  __m128i pix5678 = sumQuad5678;

  // correct if signed, scale back, store
  if constexpr (sizeof(pixel_t) == 2 && !lessthan16bit) {
    const __m128i shiftfromsigned = _mm_set1_epi32(+32768 << FPScale16bits); // yes, 32 bit data. for 16 bits only
    pix1234 = _mm_add_epi32(pix1234, shiftfromsigned);
    pix5678 = _mm_add_epi32(pix5678, shiftfromsigned);
  }

  const int current_fp_scale_bits = (sizeof(pixel_t) == 1) ? FPScale8bits : FPScale16bits;
  // scale back, shuffle, store
  __m128i result1234 = _mm_srai_epi32(pix1234, current_fp_scale_bits);
  __m128i result5678 = _mm_srai_epi32(pix5678, current_fp_scale_bits);
  __m128i result_2x4x_uint16_128 = _MM_PACKUS_EPI32(result1234, result5678);
  if constexpr (sizeof(pixel_t) == 1) {
    __m128i result_2x4x_uint8 = _mm_packus_epi16(result_2x4x_uint16_128, shifttosigned_or_zero128);
  _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + x), result_2x4x_uint8);
}
  else {
    // uint16_t
    if constexpr (lessthan16bit)
      result_2x4x_uint16_128 = _MM_MIN_EPU16(result_2x4x_uint16_128, clamp_limit); // extra clamp for 10-14 bits

    _mm_store_si128(reinterpret_cast<__m128i*>(dst + x), result_2x4x_uint16_128);

  }
}

//-------- uint8/16_t Horizontal
// 4 pixels at a time. 
// ssse3: _mm_hadd_epi32
template<typename pixel_t, bool lessthan16bit>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("ssse3")))
#endif
void resizer_h_ssse3_generic_uint8_16(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel) {
  int filter_size = program->filter_size;
  const int current_fp_scale_bits = (sizeof(pixel_t) == 1) ? FPScale8bits : FPScale16bits;
  __m128i rounder128 = _mm_setr_epi32(1 << (current_fp_scale_bits - 1), 0, 0, 0);

  __m128i shifttosigned_or_zero128;
  if constexpr(sizeof(pixel_t) == 1)
    shifttosigned_or_zero128 = _mm_setzero_si128();
  else
    shifttosigned_or_zero128 = _mm_set1_epi16(-32768); // for 16 bits only
  __m128i clamp_limit = _mm_set1_epi16((short)((1 << bits_per_pixel) - 1)); // clamp limit for 8< <16 bits

  const pixel_t* src = reinterpret_cast<const pixel_t* AVS_RESTRICT>(src8);
  pixel_t* dst = reinterpret_cast<pixel_t* AVS_RESTRICT>(dst8);
  dst_pitch /= sizeof(pixel_t);
  src_pitch /= sizeof(pixel_t);

  const int w_safe_mod8 = (program->overread_possible ? program->source_overread_beyond_targetx : width) / 8 * 8;

  for (int y = 0; y < height; y++) {
    const short* AVS_RESTRICT current_coeff_base = program->pixel_coefficient;

    // Process safe aligned pixels
    for (int x = 0; x < w_safe_mod8; x += 8) {
      process_eight_pixels_h_uint8_16<true, pixel_t, lessthan16bit>(src, x, current_coeff_base, filter_size, rounder128, shifttosigned_or_zero128, clamp_limit, dst, program);
    }

    // Process up to the actual kernel size instead of the aligned filter_size to prevent overreading beyond the last source pixel.
    // We assume extra offset entries were added to the p->pixel_offset array (aligned to 8 during initialization).
    // This may store 1-7 false pixels, but they are ignored since Avisynth will not read beyond the width.
    for (int x = w_safe_mod8; x < width; x += 8) {
      process_eight_pixels_h_uint8_16<false, pixel_t, lessthan16bit>(src, x, current_coeff_base, filter_size, rounder128, shifttosigned_or_zero128, clamp_limit, dst, program);
    }

    dst += dst_pitch;
    src += src_pitch;
  }
}

// 16 bit Horizontal

template void resizer_h_ssse3_generic_uint8_16<uint8_t, true>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resizer_h_ssse3_generic_uint8_16<uint16_t, false>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resizer_h_ssse3_generic_uint8_16<uint16_t, true>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);

template void resize_v_sse2_planar_uint16_t<false>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel);
template void resize_v_sse2_planar_uint16_t<true>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel);

// Transpose-based SIMD
void resize_h_planar_float_sse_transpose(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel) {
	int filter_size = program->filter_size;

	const float* AVS_RESTRICT current_coeff;

	src_pitch = src_pitch / sizeof(float);
	dst_pitch = dst_pitch / sizeof(float);

	float* src = (float*)src8;
	float* dst = (float*)dst8;

	const int kernel_size = program->filter_size_real;
	const int ksmod4 = kernel_size / 4 * 4;
	//	const int ksmod8 = kernel_size / 8 * 8;
#if 0
    // single row processing - slower
	for (int y = 0; y < height; y++) {
		current_coeff = (const float* AVS_RESTRICT)program->pixel_coefficient_float;

		float* AVS_RESTRICT dst2_ptr = dst + y * dst_pitch;
		const float* src_ptr = src + y * src_pitch;

		for (int x = 0; x < width; x+=4) {

			__m128 result = _mm_setzero_ps();

			for (int i = 0; i < ksmod4; i += 4) {
				__m128 data_1 = _mm_loadu_ps(src_ptr + program->pixel_offset[x + 0] + i);
				__m128 data_2 = _mm_loadu_ps(src_ptr + program->pixel_offset[x + 1] + i);
				__m128 data_3 = _mm_loadu_ps(src_ptr + program->pixel_offset[x + 2] + i);
				__m128 data_4 = _mm_loadu_ps(src_ptr + program->pixel_offset[x + 3] + i);

				__m128 coeff_1 = _mm_load_ps(current_coeff + i + filter_size * 0); 
				__m128 coeff_2 = _mm_load_ps(current_coeff + i + filter_size * 1);
				__m128 coeff_3 = _mm_load_ps(current_coeff + i + filter_size * 2);
				__m128 coeff_4 = _mm_load_ps(current_coeff + i + filter_size * 3);

				_MM_TRANSPOSE4_PS(data_1, data_2, data_3, data_4);
				_MM_TRANSPOSE4_PS(coeff_1, coeff_2, coeff_3, coeff_4);

                result = _mm_add_ps(_mm_mul_ps(data_1, coeff_1), result);
                result = _mm_add_ps(_mm_mul_ps(data_2, coeff_2), result);
                result = _mm_add_ps(_mm_mul_ps(data_3, coeff_3), result);
                result = _mm_add_ps(_mm_mul_ps(data_4, coeff_4), result);
            }

			_mm_store_ps(dst2_ptr + x, result);
			current_coeff += filter_size * 4;
		}
	}
#endif

    for (int y = 0; y < height; y+=2) {
        current_coeff = (const float* AVS_RESTRICT)program->pixel_coefficient_float;

        float* AVS_RESTRICT dst2_ptr = dst + y * dst_pitch;
        float* AVS_RESTRICT dst2_ptr2 = dst + (y + 1) * dst_pitch;
        const float* src_ptr = src + y * src_pitch;
        const float* src_ptr2 = src + (y + 1) * src_pitch;

        for (int x = 0; x < width; x += 4) {

            __m128 result = _mm_setzero_ps();
            __m128 result2 = _mm_setzero_ps();

            for (int i = 0; i < kernel_size; i += 4) { // it is always mod4 ?

                __m128 data_1 = _mm_loadu_ps(src_ptr + program->pixel_offset[x + 0] + i);
                __m128 data_2 = _mm_loadu_ps(src_ptr + program->pixel_offset[x + 1] + i);
                __m128 data_3 = _mm_loadu_ps(src_ptr + program->pixel_offset[x + 2] + i);
                __m128 data_4 = _mm_loadu_ps(src_ptr + program->pixel_offset[x + 3] + i);

                __m128 data_1_2 = _mm_loadu_ps(src_ptr2 + program->pixel_offset[x + 0] + i);
                __m128 data_2_2 = _mm_loadu_ps(src_ptr2 + program->pixel_offset[x + 1] + i);
                __m128 data_3_2 = _mm_loadu_ps(src_ptr2 + program->pixel_offset[x + 2] + i);
                __m128 data_4_2 = _mm_loadu_ps(src_ptr2 + program->pixel_offset[x + 3] + i);

                __m128 coeff_1 = _mm_load_ps(current_coeff + i + filter_size * 0);
                __m128 coeff_2 = _mm_load_ps(current_coeff + i + filter_size * 1);
                __m128 coeff_3 = _mm_load_ps(current_coeff + i + filter_size * 2);
                __m128 coeff_4 = _mm_load_ps(current_coeff + i + filter_size * 3);

                _MM_TRANSPOSE4_PS(data_1, data_2, data_3, data_4);
                _MM_TRANSPOSE4_PS(data_1_2, data_2_2, data_3_2, data_4_2);
                _MM_TRANSPOSE4_PS(coeff_1, coeff_2, coeff_3, coeff_4);

                result = _mm_add_ps(_mm_mul_ps(data_1, coeff_1), result);
                result = _mm_add_ps(_mm_mul_ps(data_2, coeff_2), result);
                result = _mm_add_ps(_mm_mul_ps(data_3, coeff_3), result);
                result = _mm_add_ps(_mm_mul_ps(data_4, coeff_4), result);

                result2 = _mm_add_ps(_mm_mul_ps(data_1_2, coeff_1), result2);
                result2 = _mm_add_ps(_mm_mul_ps(data_2_2, coeff_2), result2);
                result2 = _mm_add_ps(_mm_mul_ps(data_3_2, coeff_3), result2);
                result2 = _mm_add_ps(_mm_mul_ps(data_4_2, coeff_4), result2);

            }

            _mm_store_ps(dst2_ptr + x, result);
            _mm_store_ps(dst2_ptr2 + x, result2);

            current_coeff += filter_size * 4;
        }
    }

    // to do - need to process last row of not-mod2 heights
}

// process kernel size from up to 4 - BilinearResize, BicubicResize or sinc up to taps=2
void resize_h_planar_float_sse_transpose_vstripe_ks4(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel) 
{
    int filter_size = program->filter_size;

    const float* AVS_RESTRICT current_coeff;

    src_pitch = src_pitch / sizeof(float);
    dst_pitch = dst_pitch / sizeof(float);

    float* src = (float*)src8;
    float* dst = (float*)dst8;

    current_coeff = (const float* AVS_RESTRICT)program->pixel_coefficient_float;

    for (int x = 0; x < width; x += 4)
    {
        __m128 coeff_1 = _mm_load_ps(current_coeff + filter_size * 0);
        __m128 coeff_2 = _mm_load_ps(current_coeff + filter_size * 1);
        __m128 coeff_3 = _mm_load_ps(current_coeff + filter_size * 2);
        __m128 coeff_4 = _mm_load_ps(current_coeff + filter_size * 3);

        _MM_TRANSPOSE4_PS(coeff_1, coeff_2, coeff_3, coeff_4);

        float* AVS_RESTRICT dst_ptr = dst + x;
        const float* src_ptr = src;

        for (int y = 0; y < height; y++)
        {
            __m128 data_1 = _mm_loadu_ps(src_ptr + program->pixel_offset[x + 0]);
            __m128 data_2 = _mm_loadu_ps(src_ptr + program->pixel_offset[x + 1]);
            __m128 data_3 = _mm_loadu_ps(src_ptr + program->pixel_offset[x + 2]);
            __m128 data_4 = _mm_loadu_ps(src_ptr + program->pixel_offset[x + 3]);

            _MM_TRANSPOSE4_PS(data_1, data_2, data_3, data_4);

            __m128 result = _mm_mul_ps(data_1, coeff_1);
            result = _mm_add_ps(_mm_mul_ps(data_2, coeff_2), result);
            result = _mm_add_ps(_mm_mul_ps(data_3, coeff_3), result);
            result = _mm_add_ps(_mm_mul_ps(data_4, coeff_4), result);

            _mm_store_ps(dst_ptr, result);

            dst_ptr += dst_pitch;
            src_ptr += src_pitch;
        }
        current_coeff += filter_size * 4;
    }

}

