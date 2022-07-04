// Avisynth v2.5.  Copyright 2002 Ben Rudiak-Gould et al.
// http://www.avisynth.org

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
void resize_v_mmx_planar(BYTE* dst, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel, const int* pitch_table, const void* storage)
{
  int filter_size = program->filter_size;
  short* current_coeff = program->pixel_coefficient;

  int wMod8 = (width / 8) * 8;
  int sizeMod2 = (filter_size / 2) * 2;
  bool notMod2 = sizeMod2 < filter_size;

  __m64 zero = _mm_setzero_si64();

  for (int y = 0; y < target_height; y++) {
    int offset = program->pixel_offset[y];
    const BYTE* src_ptr = src + pitch_table[offset];

    for (int x = 0; x < wMod8; x += 8) {
      __m64 result_1 = _mm_set1_pi32(8192); // Init. with rounder (16384/2 = 8192)
      __m64 result_2 = result_1;
      __m64 result_3 = result_1;
      __m64 result_4 = result_1;

      for (int i = 0; i < sizeMod2; i += 2) {
        __m64 src_p1 = *(reinterpret_cast<const __m64*>(src_ptr + pitch_table[i] + x));   // For detailed explanation please see SSE2 version.
        __m64 src_p2 = *(reinterpret_cast<const __m64*>(src_ptr + pitch_table[i + 1] + x));

        __m64 src_l = _mm_unpacklo_pi8(src_p1, src_p2);
        __m64 src_h = _mm_unpackhi_pi8(src_p1, src_p2);

        __m64 src_1 = _mm_unpacklo_pi8(src_l, zero);
        __m64 src_2 = _mm_unpackhi_pi8(src_l, zero);
        __m64 src_3 = _mm_unpacklo_pi8(src_h, zero);
        __m64 src_4 = _mm_unpackhi_pi8(src_h, zero);

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
        __m64 src_p = *(reinterpret_cast<const __m64*>(src_ptr + pitch_table[sizeMod2] + x));

        __m64 src_l = _mm_unpacklo_pi8(src_p, zero);
        __m64 src_h = _mm_unpackhi_pi8(src_p, zero);

        __m64 coeff = _mm_set1_pi16(current_coeff[sizeMod2]);

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
      for (int i = 0; i < filter_size; i++) {
        result += (src_ptr + pitch_table[i])[x] * current_coeff[i];
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

void resize_v_sse2_planar(BYTE* dst, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel, const int* pitch_table, const void* storage)
{
  AVS_UNUSED(src_pitch);
  AVS_UNUSED(bits_per_pixel);
  AVS_UNUSED(storage);

  int filter_size = program->filter_size;
  short* current_coeff = program->pixel_coefficient;

  int wMod16 = (width / 16) * 16;
  int sizeMod2 = (filter_size / 2) * 2;
  bool notMod2 = sizeMod2 < filter_size;

  __m128i zero = _mm_setzero_si128();

  for (int y = 0; y < target_height; y++) {
    int offset = program->pixel_offset[y];
    const BYTE* src_ptr = src + pitch_table[offset];

    for (int x = 0; x < wMod16; x += 16) {
      __m128i result_1 = _mm_set1_epi32(8192); // Init. with rounder (16384/2 = 8192)
      __m128i result_2 = result_1;
      __m128i result_3 = result_1;
      __m128i result_4 = result_1;

      for (int i = 0; i < sizeMod2; i += 2) {
        __m128i src_p1 = _mm_load_si128(reinterpret_cast<const __m128i*>(src_ptr + pitch_table[i] + x));   // p|o|n|m|l|k|j|i|h|g|f|e|d|c|b|a
        __m128i src_p2 = _mm_load_si128(reinterpret_cast<const __m128i*>(src_ptr + pitch_table[i + 1] + x)); // P|O|N|M|L|K|J|I|H|G|F|E|D|C|B|A

        __m128i src_l = _mm_unpacklo_epi8(src_p1, src_p2);                                   // Hh|Gg|Ff|Ee|Dd|Cc|Bb|Aa
        __m128i src_h = _mm_unpackhi_epi8(src_p1, src_p2);                                   // Pp|Oo|Nn|Mm|Ll|Kk|Jj|Ii

        __m128i src_1 = _mm_unpacklo_epi8(src_l, zero);                                      // .D|.d|.C|.c|.B|.b|.A|.a
        __m128i src_2 = _mm_unpackhi_epi8(src_l, zero);                                      // .H|.h|.G|.g|.F|.f|.E|.e
        __m128i src_3 = _mm_unpacklo_epi8(src_h, zero);                                      // etc.
        __m128i src_4 = _mm_unpackhi_epi8(src_h, zero);                                      // etc.

        __m128i coeff = _mm_cvtsi32_si128(*reinterpret_cast<const int*>(current_coeff + i));   // XX|XX|XX|XX|XX|XX|CO|co
        coeff = _mm_shuffle_epi32(coeff, 0);                                                 // CO|co|CO|co|CO|co|CO|co

        __m128i dst_1 = _mm_madd_epi16(src_1, coeff);                                         // CO*D+co*d | CO*C+co*c | CO*B+co*b | CO*A+co*a
        __m128i dst_2 = _mm_madd_epi16(src_2, coeff);                                         // etc.
        __m128i dst_3 = _mm_madd_epi16(src_3, coeff);
        __m128i dst_4 = _mm_madd_epi16(src_4, coeff);

        result_1 = _mm_add_epi32(result_1, dst_1);
        result_2 = _mm_add_epi32(result_2, dst_2);
        result_3 = _mm_add_epi32(result_3, dst_3);
        result_4 = _mm_add_epi32(result_4, dst_4);
      }

      if (notMod2) { // do last odd row
        __m128i src_p = _mm_load_si128(reinterpret_cast<const __m128i*>(src_ptr + pitch_table[sizeMod2] + x));

        __m128i src_l = _mm_unpacklo_epi8(src_p, zero);
        __m128i src_h = _mm_unpackhi_epi8(src_p, zero);

        __m128i coeff = _mm_set1_epi16(current_coeff[sizeMod2]);

        __m128i dst_ll = _mm_mullo_epi16(src_l, coeff);   // Multiply by coefficient
        __m128i dst_lh = _mm_mulhi_epi16(src_l, coeff);
        __m128i dst_hl = _mm_mullo_epi16(src_h, coeff);
        __m128i dst_hh = _mm_mulhi_epi16(src_h, coeff);

        __m128i dst_1 = _mm_unpacklo_epi16(dst_ll, dst_lh); // Unpack to 32-bit integer
        __m128i dst_2 = _mm_unpackhi_epi16(dst_ll, dst_lh);
        __m128i dst_3 = _mm_unpacklo_epi16(dst_hl, dst_hh);
        __m128i dst_4 = _mm_unpackhi_epi16(dst_hl, dst_hh);

        result_1 = _mm_add_epi32(result_1, dst_1);
        result_2 = _mm_add_epi32(result_2, dst_2);
        result_3 = _mm_add_epi32(result_3, dst_3);
        result_4 = _mm_add_epi32(result_4, dst_4);
      }

      // Divide by 16348 (FPRound)
      result_1 = _mm_srai_epi32(result_1, 14);
      result_2 = _mm_srai_epi32(result_2, 14);
      result_3 = _mm_srai_epi32(result_3, 14);
      result_4 = _mm_srai_epi32(result_4, 14);

      // Pack and store
      __m128i result_l = _mm_packs_epi32(result_1, result_2);
      __m128i result_h = _mm_packs_epi32(result_3, result_4);
      __m128i result = _mm_packus_epi16(result_l, result_h);

      _mm_store_si128(reinterpret_cast<__m128i*>(dst + x), result);
    }

    // Leftover
    for (int x = wMod16; x < width; x++) {
      int result = 0;
      for (int i = 0; i < filter_size; i++) {
        result += (src_ptr + pitch_table[i])[x] * current_coeff[i];
      }
      result = ((result + 8192) / 16384);
      result = result > 255 ? 255 : result < 0 ? 0 : result;
      dst[x] = (BYTE)result;
    }

    dst += dst_pitch;
    current_coeff += filter_size;
  }
}

// The only difference between resize_v_sse41_planar and resize_v_ssse3_planar is the load operation
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
void resize_v_sse41_planar(BYTE* dst, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel, const int* pitch_table, const void* storage)
{
  AVS_UNUSED(bits_per_pixel);
  AVS_UNUSED(storage);

  int filter_size = program->filter_size;
  short* current_coeff = program->pixel_coefficient;

  int wMod16 = (width / 16) * 16;

  __m128i zero = _mm_setzero_si128();
  __m128i coeff_unpacker = _mm_set_epi8(1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0);

  for (int y = 0; y < target_height; y++) {
    int offset = program->pixel_offset[y];
    const BYTE* src_ptr = src + pitch_table[offset];

    for (int x = 0; x < wMod16; x += 16) {
      __m128i result_l = _mm_set1_epi16(32); // Init. with rounder ((1 << 6)/2 = 32)
      __m128i result_h = result_l;

      const BYTE* src2_ptr = src_ptr + x;

      for (int i = 0; i < filter_size; i++) {
        __m128i src_p = _mm_stream_load_si128(const_cast<__m128i*>(reinterpret_cast<const __m128i*>(src2_ptr)));

        __m128i src_l = _mm_unpacklo_epi8(src_p, zero);
        __m128i src_h = _mm_unpackhi_epi8(src_p, zero);

        src_l = _mm_slli_epi16(src_l, 7);
        src_h = _mm_slli_epi16(src_h, 7);

        __m128i coeff = _mm_cvtsi32_si128(*reinterpret_cast<const int*>(current_coeff + i));
        coeff = _mm_shuffle_epi8(coeff, coeff_unpacker);

        __m128i dst_l = _mm_mulhrs_epi16(src_l, coeff);   // Multiply by coefficient (SSSE3)
        __m128i dst_h = _mm_mulhrs_epi16(src_h, coeff);

        result_l = _mm_add_epi16(result_l, dst_l);
        result_h = _mm_add_epi16(result_h, dst_h);

        src2_ptr += src_pitch;
      }

      // Divide by 64
      result_l = _mm_srai_epi16(result_l, 6);
      result_h = _mm_srai_epi16(result_h, 6);

      // Pack and store
      __m128i result = _mm_packus_epi16(result_l, result_h);

      _mm_store_si128(reinterpret_cast<__m128i*>(dst + x), result);
    }

    // Leftover
    for (int x = wMod16; x < width; x++) {
      int result = 0;
      for (int i = 0; i < filter_size; i++) {
        result += (src_ptr + pitch_table[i])[x] * current_coeff[i];
      }
      result = ((result + 8192) / 16384);
      result = result > 255 ? 255 : result < 0 ? 0 : result;
      dst[x] = (BYTE)result;
    }

    dst += dst_pitch;
    current_coeff += filter_size;
  }
}

#if defined(GCC) || defined(CLANG)
__attribute__((__target__("ssse3")))
#endif
void resize_v_ssse3_planar(BYTE* dst, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel, const int* pitch_table, const void* storage)
{
  AVS_UNUSED(bits_per_pixel);
  AVS_UNUSED(storage);

  int filter_size = program->filter_size;
  short* current_coeff = program->pixel_coefficient;

  int wMod16 = (width / 16) * 16;

  __m128i zero = _mm_setzero_si128();
  __m128i coeff_unpacker = _mm_set_epi8(1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0);

  for (int y = 0; y < target_height; y++) {
    int offset = program->pixel_offset[y];
    const BYTE* src_ptr = src + pitch_table[offset];

    for (int x = 0; x < wMod16; x += 16) {
      __m128i result_l = _mm_set1_epi16(32); // Init. with rounder ((1 << 6)/2 = 32)
      __m128i result_h = result_l;

      const BYTE* src2_ptr = src_ptr + x;

      for (int i = 0; i < filter_size; i++) {
        __m128i src_p = _mm_load_si128(reinterpret_cast<const __m128i*>(src2_ptr));

        __m128i src_l = _mm_unpacklo_epi8(src_p, zero);
        __m128i src_h = _mm_unpackhi_epi8(src_p, zero);

        src_l = _mm_slli_epi16(src_l, 7);
        src_h = _mm_slli_epi16(src_h, 7);

        __m128i coeff = _mm_cvtsi32_si128(*reinterpret_cast<const int*>(current_coeff + i));
        coeff = _mm_shuffle_epi8(coeff, coeff_unpacker);

        __m128i dst_l = _mm_mulhrs_epi16(src_l, coeff);   // Multiply by coefficient (SSSE3)
        __m128i dst_h = _mm_mulhrs_epi16(src_h, coeff);

        result_l = _mm_add_epi16(result_l, dst_l);
        result_h = _mm_add_epi16(result_h, dst_h);

        src2_ptr += src_pitch;
      }

      // Divide by 64
      result_l = _mm_srai_epi16(result_l, 6);
      result_h = _mm_srai_epi16(result_h, 6);

      // Pack and store
      __m128i result = _mm_packus_epi16(result_l, result_h);

      _mm_store_si128(reinterpret_cast<__m128i*>(dst + x), result);
    }

    // Leftover
    for (int x = wMod16; x < width; x++) {
      int result = 0;
      for (int i = 0; i < filter_size; i++) {
        result += (src_ptr + pitch_table[i])[x] * current_coeff[i];
      }
      result = ((result + 8192) / 16384);
      result = result > 255 ? 255 : result < 0 ? 0 : result;
      dst[x] = (BYTE)result;
    }

    dst += dst_pitch;
    current_coeff += filter_size;
  }
}



// make the resampling coefficient array mod8 or mod16 friendly for simd, padding non-used coeffs with zeros
void resize_h_prepare_coeff_8or16(ResamplingProgram* p, IScriptEnvironment* env, int alignFilterSize8or16) {
  p->filter_size_alignment = alignFilterSize8or16;
  int filter_size = AlignNumber(p->filter_size, alignFilterSize8or16);
  // for even non-simd it was aligned/padded as well, keep the same here
  int target_size = AlignNumber(p->target_size, ALIGN_RESIZER_TARGET_SIZE);

  // Copy existing coeff
  if (p->bits_per_pixel == 32) {
    float* new_coeff_float = (float*)env->Allocate(sizeof(float) * target_size * filter_size, 64, AVS_NORMAL_ALLOC);
    if (!new_coeff_float) {
      env->Free(new_coeff_float);
      env->ThrowError("Could not reserve memory in a resampler.");
    }
    std::fill_n(new_coeff_float, target_size * filter_size, 0.0f);
    float* dst_f = new_coeff_float, * src_f = p->pixel_coefficient_float;
    for (int i = 0; i < p->target_size; i++) {
      for (int j = 0; j < p->filter_size; j++) {
        dst_f[j] = src_f[j];
      }

      dst_f += filter_size;
      src_f += p->filter_size;
    }
    env->Free(p->pixel_coefficient_float);
    p->pixel_coefficient_float = new_coeff_float;
  }
  else {
    short* new_coeff = (short*)env->Allocate(sizeof(short) * target_size * filter_size, 64, AVS_NORMAL_ALLOC);
    if (!new_coeff) {
      env->Free(new_coeff);
      env->ThrowError("Could not reserve memory in a resampler.");
    }
    memset(new_coeff, 0, sizeof(short) * target_size * filter_size);
    short* dst = new_coeff, * src = p->pixel_coefficient;
    for (int i = 0; i < p->target_size; i++) {
      for (int j = 0; j < p->filter_size; j++) {
        dst[j] = src[j];
      }

      dst += filter_size;
      src += p->filter_size;

    }
    env->Free(p->pixel_coefficient);
    p->pixel_coefficient = new_coeff;
  }
}

//-------- 128 bit float Horizontals

AVS_FORCEINLINE static void process_one_pixel_h_float(const float* src, int begin, int i, float*& current_coeff, __m128& result) {
  // 2x4 pixels
  __m128 data_l_single = _mm_loadu_ps(reinterpret_cast<const float*>(src + begin + i * 8));
  __m128 data_h_single = _mm_loadu_ps(reinterpret_cast<const float*>(src + begin + i * 8 + 4));
  __m128 coeff_l = _mm_load_ps(reinterpret_cast<const float*>(current_coeff)); // always aligned
  __m128 coeff_h = _mm_load_ps(reinterpret_cast<const float*>(current_coeff + 4));
  __m128 dst_l = _mm_mul_ps(data_l_single, coeff_l); // Multiply by coefficient
  __m128 dst_h = _mm_mul_ps(data_h_single, coeff_h); // 4*(32bit*32bit=32bit)
  result = _mm_add_ps(result, dst_l); // accumulate result.
  result = _mm_add_ps(result, dst_h);
  current_coeff += 8;
}

template<int filtersizemod8>
AVS_FORCEINLINE static void process_one_pixel_h_float_mask(const float* src, int begin, int i, float*& current_coeff, __m128& result, __m128& mask) {
  __m128 data_l_single;
  __m128 data_h_single;
  // 2x4 pixels
  if constexpr (filtersizemod8 > 4) { // keep low, mask high 4 pixels
    data_l_single = _mm_loadu_ps(reinterpret_cast<const float*>(src + begin + i * 8));
    data_h_single = _mm_loadu_ps(reinterpret_cast<const float*>(src + begin + i * 8 + 4));
    data_h_single = _mm_and_ps(data_h_single, mask);
  }
  else if constexpr (filtersizemod8 == 4) { // keep low, zero high 4 pixels
    data_l_single = _mm_loadu_ps(reinterpret_cast<const float*>(src + begin + i * 8));
    data_h_single = _mm_setzero_ps();
  }
  else { // filtersizemod8 1..3
    data_l_single = _mm_loadu_ps(reinterpret_cast<const float*>(src + begin + i * 8));
    data_l_single = _mm_and_ps(data_l_single, mask);
    data_h_single = _mm_setzero_ps();
  }
  __m128 coeff_l = _mm_load_ps(reinterpret_cast<const float*>(current_coeff)); // always aligned
  __m128 coeff_h = _mm_load_ps(reinterpret_cast<const float*>(current_coeff + 4));
  __m128 dst_l = _mm_mul_ps(data_l_single, coeff_l); // Multiply by coefficient
  __m128 dst_h = _mm_mul_ps(data_h_single, coeff_h); // 4*(32bit*32bit=32bit)
  result = _mm_add_ps(result, dst_l); // accumulate result.
  result = _mm_add_ps(result, dst_h);
  current_coeff += 8;
}

// filtersizealigned8: special: 1, 2. Generic: -1
template<int filtersizealigned8, int filtersizemod8>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("ssse3")))
#endif
void resizer_h_ssse3_generic_float(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel)
{
  AVS_UNUSED(bits_per_pixel);
  int filter_size_numOfBlk8 = (AlignNumber(program->filter_size, 8) / 8);
  if constexpr (filtersizealigned8 >= 1)
    filter_size_numOfBlk8 = filtersizealigned8;

  const float* src = reinterpret_cast<const float*>(src8);
  float* dst = reinterpret_cast<float*>(dst8);
  dst_pitch /= sizeof(float);
  src_pitch /= sizeof(float);

  // OMG! 18.01.19
  // Protection against NaN
  // When reading the last 8 consecutive pixels from right side offsets, it would access beyond-last-pixel area.
  // One SIMD cycle reads 8 bytes from (src + begin + i * 8)
  // When program->filter_size mod 8 is 1..7 then some of the last pixels should be masked because there can be NaN garbage.
  // So it's not enough to mask the coefficients by zero. Theory: let's multiply offscreen elements by 0 which works for integer samples.
  // But we are using float, so since NaN * Zero is NaN which propagates further to NaN when hadd is summing up the pixel*coeff series

  __m128 mask;
  switch (filtersizemod8 & 3) {
  case 3: mask = _mm_castsi128_ps(_mm_setr_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0)); break; // keep 0-1-2, drop #3
  case 2: mask = _mm_castsi128_ps(_mm_setr_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0, 0)); break; // keep 0-1, drop #2-3
  case 1: mask = _mm_castsi128_ps(_mm_setr_epi32(0xFFFFFFFF, 0, 0, 0)); break; // keep 0, drop #1-2-3
  default: break; // for mod4 = 0 no masking needed
  }

  const int pixels_per_cycle = 8; // doing 8 is faster than 4
  const int unsafe_limit = (program->overread_possible && filtersizemod8 != 0) ? (program->source_overread_beyond_targetx / pixels_per_cycle) * pixels_per_cycle : width;

  for (int y = 0; y < height; y++) {
    float* current_coeff = program->pixel_coefficient_float;

    // loop for clean, non-offscreen data
    for (int x = 0; x < unsafe_limit; x += pixels_per_cycle) {
      __m128 result1 = _mm_set1_ps(0.0f);
      __m128 result2 = result1;
      __m128 result3 = result1;
      __m128 result4 = result1;

      // 1-4
      int begin1 = program->pixel_offset[x + 0];
      int begin2 = program->pixel_offset[x + 1];
      int begin3 = program->pixel_offset[x + 2];
      int begin4 = program->pixel_offset[x + 3];

      // begin1, result1
      for (int i = 0; i < filter_size_numOfBlk8; i++)
        process_one_pixel_h_float(src, begin1, i, current_coeff, result1);

      // begin2, result2
      for (int i = 0; i < filter_size_numOfBlk8; i++)
        process_one_pixel_h_float(src, begin2, i, current_coeff, result2);

      // begin3, result3
      for (int i = 0; i < filter_size_numOfBlk8; i++)
        process_one_pixel_h_float(src, begin3, i, current_coeff, result3);

      // begin4, result4
      for (int i = 0; i < filter_size_numOfBlk8; i++)
        process_one_pixel_h_float(src, begin4, i, current_coeff, result4);

      // this part needs ssse3
      __m128 result12 = _mm_hadd_ps(result1, result2);
      __m128 result34 = _mm_hadd_ps(result3, result4);
      __m128 result = _mm_hadd_ps(result12, result34);

      _mm_stream_ps(reinterpret_cast<float*>(dst + x), result); // 4 results at a time

      // 5-8
      result1 = _mm_set1_ps(0.0f);
      result2 = result1;
      result3 = result1;
      result4 = result1;

      begin1 = program->pixel_offset[x + 4];
      begin2 = program->pixel_offset[x + 5];
      begin3 = program->pixel_offset[x + 6];
      begin4 = program->pixel_offset[x + 7];

      // begin1, result1
      for (int i = 0; i < filter_size_numOfBlk8; i++)
        process_one_pixel_h_float(src, begin1, i, current_coeff, result1);

      // begin2, result2
      for (int i = 0; i < filter_size_numOfBlk8; i++)
        process_one_pixel_h_float(src, begin2, i, current_coeff, result2);

      // begin3, result3
      for (int i = 0; i < filter_size_numOfBlk8; i++)
        process_one_pixel_h_float(src, begin3, i, current_coeff, result3);

      // begin4, result4
      for (int i = 0; i < filter_size_numOfBlk8; i++)
        process_one_pixel_h_float(src, begin4, i, current_coeff, result4);

      // this part needs ssse3
      result12 = _mm_hadd_ps(result1, result2);
      result34 = _mm_hadd_ps(result3, result4);
      result = _mm_hadd_ps(result12, result34);

      _mm_stream_ps(reinterpret_cast<float*>(dst + x + 4), result); // 4 results at a time
    } // for x

      // possibly right-side offscreen
      // and the same for the rest with masking the last filtersize/8 chunk
    for (int x = unsafe_limit; x < width; x += 4) {
      __m128 result1 = _mm_set1_ps(0.0f);
      __m128 result2 = result1;
      __m128 result3 = result1;
      __m128 result4 = result1;

      int begin1 = program->pixel_offset[x + 0];
      int begin2 = program->pixel_offset[x + 1];
      int begin3 = program->pixel_offset[x + 2];
      int begin4 = program->pixel_offset[x + 3];

      // begin1, result1
      for (int i = 0; i < filter_size_numOfBlk8 - 1; i++)
        process_one_pixel_h_float(src, begin1, i, current_coeff, result1);
      if (begin1 < program->source_overread_offset)
        process_one_pixel_h_float(src, begin1, filter_size_numOfBlk8 - 1, current_coeff, result1);
      else
        process_one_pixel_h_float_mask<filtersizemod8>(src, begin1, filter_size_numOfBlk8 - 1, current_coeff, result1, mask);

      // begin2, result2
      for (int i = 0; i < filter_size_numOfBlk8 - 1; i++)
        process_one_pixel_h_float(src, begin2, i, current_coeff, result2);
      if (begin2 < program->source_overread_offset)
        process_one_pixel_h_float(src, begin2, filter_size_numOfBlk8 - 1, current_coeff, result2);
      else
        process_one_pixel_h_float_mask<filtersizemod8>(src, begin2, filter_size_numOfBlk8 - 1, current_coeff, result2, mask);

      // begin3, result3
      for (int i = 0; i < filter_size_numOfBlk8 - 1; i++)
        process_one_pixel_h_float(src, begin3, i, current_coeff, result3);
      if (begin3 < program->source_overread_offset)
        process_one_pixel_h_float(src, begin3, filter_size_numOfBlk8 - 1, current_coeff, result3);
      else
        process_one_pixel_h_float_mask<filtersizemod8>(src, begin3, filter_size_numOfBlk8 - 1, current_coeff, result3, mask);

      // begin4, result4
      for (int i = 0; i < filter_size_numOfBlk8 - 1; i++)
        process_one_pixel_h_float(src, begin4, i, current_coeff, result4);
      if (begin4 < program->source_overread_offset)
        process_one_pixel_h_float(src, begin4, filter_size_numOfBlk8 - 1, current_coeff, result4);
      else
        process_one_pixel_h_float_mask<filtersizemod8>(src, begin4, filter_size_numOfBlk8 - 1, current_coeff, result4, mask);

      // this part needs ssse3
      __m128 result12 = _mm_hadd_ps(result1, result2);
      __m128 result34 = _mm_hadd_ps(result3, result4);
      __m128 result = _mm_hadd_ps(result12, result34);

      _mm_stream_ps(reinterpret_cast<float*>(dst + x), result); // 4 results at a time

    } // for x

    dst += dst_pitch;
    src += src_pitch;
  }
  /*
  // check Nans
  dst -= dst_pitch * height;
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x += 4) {
      if (std::isnan(dst[x]))
      {
        x = x;
      }
    }
    dst += dst_pitch;
  }
  */
}

// instantiate
template void resizer_h_ssse3_generic_float<1, 0>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resizer_h_ssse3_generic_float<1, 1>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resizer_h_ssse3_generic_float<1, 2>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resizer_h_ssse3_generic_float<1, 3>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resizer_h_ssse3_generic_float<1, 4>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resizer_h_ssse3_generic_float<1, 5>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resizer_h_ssse3_generic_float<1, 6>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resizer_h_ssse3_generic_float<1, 7>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resizer_h_ssse3_generic_float<2, 0>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resizer_h_ssse3_generic_float<2, 1>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resizer_h_ssse3_generic_float<2, 2>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resizer_h_ssse3_generic_float<2, 3>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resizer_h_ssse3_generic_float<2, 4>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resizer_h_ssse3_generic_float<2, 5>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resizer_h_ssse3_generic_float<2, 6>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resizer_h_ssse3_generic_float<2, 7>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resizer_h_ssse3_generic_float<-1, 0>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resizer_h_ssse3_generic_float<-1, 1>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resizer_h_ssse3_generic_float<-1, 2>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resizer_h_ssse3_generic_float<-1, 3>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resizer_h_ssse3_generic_float<-1, 4>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resizer_h_ssse3_generic_float<-1, 5>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resizer_h_ssse3_generic_float<-1, 6>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resizer_h_ssse3_generic_float<-1, 7>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);

//-------- 128 bit uint16_t Horizontals

template<bool lessthan16bit>
AVS_FORCEINLINE static void process_one_pixel_h_uint16_t(const uint16_t* src, int begin, int i, short*& current_coeff, __m128i& result, const __m128i& shifttosigned) {
  __m128i data_single = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + begin + i * 8)); // 8 pixels
  if (!lessthan16bit)
    data_single = _mm_add_epi16(data_single, shifttosigned); // unsigned -> signed
  __m128i coeff = _mm_load_si128(reinterpret_cast<const __m128i*>(current_coeff)); // 8 coeffs
  result = _mm_add_epi32(result, _mm_madd_epi16(data_single, coeff));
  current_coeff += 8;
}

// filter_size <= 8 -> filter_size_align8 == 1 -> no loop, hope it'll be optimized
// filter_size <= 16 -> filter_size_align8 == 2 -> loop 0..1 hope it'll be optimized
// filter_size > 16 -> use parameter AlignNumber(program->filter_size_numOfFullBlk8, 8) / 8;
template<bool lessthan16bit, int filtersizealigned8>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("ssse3")))
#endif
static void internal_resizer_h_ssse3_generic_uint16_t(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel)
{
  // 1 and 2: special case for compiler optimization
  const int filter_size_numOfBlk8 = (filtersizealigned8 >= 1) ? filtersizealigned8 : (AlignNumber(program->filter_size, 8) / 8);

  const __m128i zero = _mm_setzero_si128();
  const __m128i shifttosigned = _mm_set1_epi16(-32768); // for 16 bits only
  const __m128i shiftfromsigned = _mm_set1_epi32(+32768 << FPScale16bits); // for 16 bits only
  const __m128i rounder = _mm_set_epi32(0, 0, 0, 1 << (FPScale16bits - 1)); // only once

  const uint16_t* src = reinterpret_cast<const uint16_t*>(src8);
  uint16_t* dst = reinterpret_cast<uint16_t*>(dst8);
  dst_pitch /= sizeof(uint16_t);
  src_pitch /= sizeof(uint16_t);

  __m128i clamp_limit = _mm_set1_epi16((short)((1 << bits_per_pixel) - 1)); // clamp limit for <16 bits

  for (int y = 0; y < height; y++) {
    short* current_coeff = program->pixel_coefficient;

    for (int x = 0; x < width; x += 4) {
      __m128i result1 = rounder;
      __m128i result2 = result1;
      __m128i result3 = result1;
      __m128i result4 = result1;

      int begin1 = program->pixel_offset[x + 0];
      int begin2 = program->pixel_offset[x + 1];
      int begin3 = program->pixel_offset[x + 2];
      int begin4 = program->pixel_offset[x + 3];

      // this part is repeated 4 times
      // begin1, result1
      for (int i = 0; i < filter_size_numOfBlk8; i++)
        process_one_pixel_h_uint16_t<lessthan16bit>(src, begin1, i, current_coeff, result1, shifttosigned);

      // begin2, result2
      for (int i = 0; i < filter_size_numOfBlk8; i++)
        process_one_pixel_h_uint16_t<lessthan16bit>(src, begin2, i, current_coeff, result2, shifttosigned);

      // begin3, result3
      for (int i = 0; i < filter_size_numOfBlk8; i++)
        process_one_pixel_h_uint16_t<lessthan16bit>(src, begin3, i, current_coeff, result3, shifttosigned);

      // begin4, result4
      for (int i = 0; i < filter_size_numOfBlk8; i++)
        process_one_pixel_h_uint16_t<lessthan16bit>(src, begin4, i, current_coeff, result4, shifttosigned);

      // _mm_hadd_epi32: SSSE3
      const __m128i sumQuad12 = _mm_hadd_epi32(result1, result2); // L1L1L1L1 + L2L2L2L2 = L1L1 L2L2
      const __m128i sumQuad34 = _mm_hadd_epi32(result3, result4); // L3L3L3L3 + L4L4L4L4 = L3L3 L4L4
      __m128i result = _mm_hadd_epi32(sumQuad12, sumQuad34); // L1L1 L2L2 + L3L3 L4L4 = L1 L2 L3 L4

      // correct if signed, scale back, store
      if (!lessthan16bit)
        result = _mm_add_epi32(result, shiftfromsigned);
      result = _mm_srai_epi32(result, FPScale16bits); // shift back integer arithmetic 13 bits precision

      // _MM_PACKUS_EPI32: SSE4.1 simul
      __m128i result_4x_uint16 = _MM_PACKUS_EPI32(result, zero); // 4*32+zeros = lower 4*16 OK
      // extra clamp for 10-14 bit
      if (lessthan16bit)
        result_4x_uint16 = _MM_MIN_EPU16(result_4x_uint16, clamp_limit);
      _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + x), result_4x_uint16);

    }

    dst += dst_pitch;
    src += src_pitch;
  }
}

template<bool lessthan16bit, int filtersizealigned8>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static void internal_resizer_h_sse41_generic_uint16_t(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel)
{
  // 1 and 2: special case for compiler optimization
  const int filter_size_numOfBlk8 = (filtersizealigned8 >= 1) ? filtersizealigned8 : (AlignNumber(program->filter_size, 8) / 8);

  const __m128i zero = _mm_setzero_si128();
  const __m128i shifttosigned = _mm_set1_epi16(-32768); // for 16 bits only
  const __m128i shiftfromsigned = _mm_set1_epi32(+32768 << FPScale16bits); // for 16 bits only
  const __m128i rounder = _mm_set_epi32(0, 0, 0, 1 << (FPScale16bits - 1)); // only once

  const uint16_t* src = reinterpret_cast<const uint16_t*>(src8);
  uint16_t* dst = reinterpret_cast<uint16_t*>(dst8);
  dst_pitch /= sizeof(uint16_t);
  src_pitch /= sizeof(uint16_t);

  __m128i clamp_limit = _mm_set1_epi16((short)((1 << bits_per_pixel) - 1)); // clamp limit for <16 bits

  for (int y = 0; y < height; y++) {
    short* current_coeff = program->pixel_coefficient;

    for (int x = 0; x < width; x += 4) {
      __m128i result1 = rounder;
      __m128i result2 = result1;
      __m128i result3 = result1;
      __m128i result4 = result1;

      int begin1 = program->pixel_offset[x + 0];
      int begin2 = program->pixel_offset[x + 1];
      int begin3 = program->pixel_offset[x + 2];
      int begin4 = program->pixel_offset[x + 3];

      // this part is repeated 4 times
      // begin1, result1
      for (int i = 0; i < filter_size_numOfBlk8; i++)
        process_one_pixel_h_uint16_t<lessthan16bit>(src, begin1, i, current_coeff, result1, shifttosigned);

      // begin2, result2
      for (int i = 0; i < filter_size_numOfBlk8; i++)
        process_one_pixel_h_uint16_t<lessthan16bit>(src, begin2, i, current_coeff, result2, shifttosigned);

      // begin3, result3
      for (int i = 0; i < filter_size_numOfBlk8; i++)
        process_one_pixel_h_uint16_t<lessthan16bit>(src, begin3, i, current_coeff, result3, shifttosigned);

      // begin4, result4
      for (int i = 0; i < filter_size_numOfBlk8; i++)
        process_one_pixel_h_uint16_t<lessthan16bit>(src, begin4, i, current_coeff, result4, shifttosigned);

      const __m128i sumQuad12 = _mm_hadd_epi32(result1, result2); // L1L1L1L1 + L2L2L2L2 = L1L1 L2L2
      const __m128i sumQuad34 = _mm_hadd_epi32(result3, result4); // L3L3L3L3 + L4L4L4L4 = L3L3 L4L4
      __m128i result = _mm_hadd_epi32(sumQuad12, sumQuad34); // L1L1 L2L2 + L3L3 L4L4 = L1 L2 L3 L4

      // correct if signed, scale back, store
      if (!lessthan16bit)
        result = _mm_add_epi32(result, shiftfromsigned);
      result = _mm_srai_epi32(result, FPScale16bits); // shift back integer arithmetic 13 bits precision

      __m128i result_4x_uint16 = _mm_packus_epi32(result, zero); // 4*32+zeros = lower 4*16 OK
      // extra clamp for 10-14 bit
      if (lessthan16bit)
        result_4x_uint16 = _mm_min_epu16(result_4x_uint16, clamp_limit);
      _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + x), result_4x_uint16);

    }

    dst += dst_pitch;
    src += src_pitch;
  }
}

//-------- 128 bit uint16_t Horizontal Dispatcher

template<bool lessthan16bit>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("ssse3")))
#endif
void resizer_h_ssse3_generic_uint16_t(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel)
{
  const int filter_size_numOfBlk8 = AlignNumber(program->filter_size, 8) / 8;

  if (filter_size_numOfBlk8 == 1)
    internal_resizer_h_ssse3_generic_uint16_t<lessthan16bit, 1>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
  else if (filter_size_numOfBlk8 == 2)
    internal_resizer_h_ssse3_generic_uint16_t<lessthan16bit, 2>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
  else // -1: basic method, use program->filter_size
    internal_resizer_h_ssse3_generic_uint16_t<lessthan16bit, -1>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
}

// instantiate
template void resizer_h_ssse3_generic_uint16_t<false>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resizer_h_ssse3_generic_uint16_t<true>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);


template<bool lessthan16bit>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
void resizer_h_sse41_generic_uint16_t(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel)
{
  const int filter_size_numOfBlk8 = AlignNumber(program->filter_size, 8) / 8;

  if (filter_size_numOfBlk8 == 1)
    internal_resizer_h_sse41_generic_uint16_t<lessthan16bit, 1>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
  else if (filter_size_numOfBlk8 == 2)
    internal_resizer_h_sse41_generic_uint16_t<lessthan16bit, 2>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
  else // -1: basic method, use program->filter_size
    internal_resizer_h_sse41_generic_uint16_t<lessthan16bit, -1>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
}

// instantiate
template void resizer_h_sse41_generic_uint16_t<false>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resizer_h_sse41_generic_uint16_t<true>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);

//-------- 128 bit uint16_t Verticals

template<bool lessthan16bit, int index>
AVS_FORCEINLINE static void process_chunk_v_uint16_t(const uint16_t* src2_ptr, int src_pitch, __m128i& coeff01234567, __m128i& result_single_lo, __m128i& result_single_hi, const __m128i& shifttosigned) {
  // offset table generating is what preventing us from overaddressing
  // 0-1
  __m128i src_even = _mm_load_si128(reinterpret_cast<const __m128i*>(src2_ptr + index * src_pitch)); // 4x 16bit pixels
  __m128i src_odd = _mm_load_si128(reinterpret_cast<const __m128i*>(src2_ptr + (index + 1) * src_pitch));  // 4x 16bit pixels
  __m128i src_lo = _mm_unpacklo_epi16(src_even, src_odd);
  __m128i src_hi = _mm_unpackhi_epi16(src_even, src_odd);
  if (!lessthan16bit) {
    src_lo = _mm_add_epi16(src_lo, shifttosigned);
    src_hi = _mm_add_epi16(src_hi, shifttosigned);
  }
  __m128i coeff = _mm_shuffle_epi32(coeff01234567, ((index / 2) << 0) | ((index / 2) << 2) | ((index / 2) << 4) | ((index / 2) << 6)); // spread short pair
  result_single_lo = _mm_add_epi32(result_single_lo, _mm_madd_epi16(src_lo, coeff)); // a*b + c
  result_single_hi = _mm_add_epi32(result_single_hi, _mm_madd_epi16(src_hi, coeff)); // a*b + c
}

// program->filtersize: 1..16 special optimized, >8: normal
template<bool lessthan16bit, int _filter_size_numOfFullBlk8, int filtersizemod8>
void internal_resize_v_sse2_planar_uint16_t(BYTE* dst0, const BYTE* src0, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel, const int* pitch_table, const void* storage)
{
  AVS_UNUSED(storage);
  int filter_size_numOfFullBlk8 = (program->filter_size / 8);
  if constexpr (_filter_size_numOfFullBlk8 >= 0)
    filter_size_numOfFullBlk8 = _filter_size_numOfFullBlk8;
  short* current_coeff = program->pixel_coefficient;

  // #define NON32_BYTES_ALIGNMENT
  // in AVS+ 32 bytes alignment is guaranteed
#ifdef NON32_BYTES_ALIGNMENT
  int wMod8 = (width / 8) * 8; // uint16: 8 at a time (2x128bit)
#endif

  const __m128i zero = _mm_setzero_si128();
  const __m128i shifttosigned = _mm_set1_epi16(-32768); // for 16 bits only
  const __m128i shiftfromsigned = _mm_set1_epi32(32768 << FPScale16bits); // for 16 bits only
  const __m128i rounder = _mm_set1_epi32(1 << (FPScale16bits - 1));

  const uint16_t* src = (uint16_t*)src0;
  uint16_t* dst = (uint16_t*)dst0;
  dst_pitch = dst_pitch / sizeof(uint16_t);
  src_pitch = src_pitch / sizeof(uint16_t);

  const int limit = (1 << bits_per_pixel) - 1;
  __m128i clamp_limit = _mm_set1_epi16((short)limit); // clamp limit for <16 bits

  for (int y = 0; y < target_height; y++) {
    int offset = program->pixel_offset[y];
    const uint16_t* src_ptr = src + pitch_table[offset] / sizeof(uint16_t);

#ifdef NON32_BYTES_ALIGNMENT
    for (int x = 0; x < wMod8; x += 8) { // 2x4 pixels at a time
#else
    for (int x = 0; x < width; x += 8) {
#endif
      __m128i result_single_lo = rounder;
      __m128i result_single_hi = rounder;

      const uint16_t* src2_ptr = src_ptr + x;

      for (int i = 0; i < filter_size_numOfFullBlk8; i++) {
        __m128i coeff01234567 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(current_coeff + i * 8)); // 4x (2x16bit) shorts for even/odd

        // offset table generating is what preventing us from overaddressing
        // 0-1
        process_chunk_v_uint16_t<lessthan16bit, 0>(src2_ptr, src_pitch, coeff01234567, result_single_lo, result_single_hi, shifttosigned);
        // 2-3
        process_chunk_v_uint16_t<lessthan16bit, 2>(src2_ptr, src_pitch, coeff01234567, result_single_lo, result_single_hi, shifttosigned);
        // 4-5
        process_chunk_v_uint16_t<lessthan16bit, 4>(src2_ptr, src_pitch, coeff01234567, result_single_lo, result_single_hi, shifttosigned);
        // 6-7
        process_chunk_v_uint16_t<lessthan16bit, 6>(src2_ptr, src_pitch, coeff01234567, result_single_lo, result_single_hi, shifttosigned);
        src2_ptr += 8 * src_pitch;
      }

      // and the rest non-div8 chunk
      __m128i coeff01234567 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(current_coeff + filter_size_numOfFullBlk8 * 8)); // 4x (2x16bit) shorts for even/odd
      if constexpr (filtersizemod8 >= 2)
        process_chunk_v_uint16_t<lessthan16bit, 0>(src2_ptr, src_pitch, coeff01234567, result_single_lo, result_single_hi, shifttosigned);
      if constexpr (filtersizemod8 >= 4)
        process_chunk_v_uint16_t<lessthan16bit, 2>(src2_ptr, src_pitch, coeff01234567, result_single_lo, result_single_hi, shifttosigned);
      if constexpr (filtersizemod8 >= 6)
        process_chunk_v_uint16_t<lessthan16bit, 4>(src2_ptr, src_pitch, coeff01234567, result_single_lo, result_single_hi, shifttosigned);
      if constexpr (filtersizemod8 % 2) { // remaining odd one
        const int index = filtersizemod8 - 1;
        __m128i src_even = _mm_load_si128(reinterpret_cast<const __m128i*>(src2_ptr + index * src_pitch)); // 8x 16bit pixels
        if (!lessthan16bit)
          src_even = _mm_add_epi16(src_even, shifttosigned);
        __m128i coeff = _mm_shuffle_epi32(coeff01234567, ((index / 2) << 0) | ((index / 2) << 2) | ((index / 2) << 4) | ((index / 2) << 6));
        __m128i src_lo = _mm_unpacklo_epi16(src_even, zero); // insert zero after the unsigned->signed shift!
        __m128i src_hi = _mm_unpackhi_epi16(src_even, zero); // insert zero after the unsigned->signed shift!
        result_single_lo = _mm_add_epi32(result_single_lo, _mm_madd_epi16(src_lo, coeff)); // a*b + c
        result_single_hi = _mm_add_epi32(result_single_hi, _mm_madd_epi16(src_hi, coeff)); // a*b + c
      }

      // correct if signed, scale back, store
      __m128i result_lo = result_single_lo;
      __m128i result_hi = result_single_hi;
      if (!lessthan16bit) {
        result_lo = _mm_add_epi32(result_lo, shiftfromsigned);
        result_hi = _mm_add_epi32(result_hi, shiftfromsigned);
      }
      result_lo = _mm_srai_epi32(result_lo, FPScale16bits); // shift back integer arithmetic 13 bits precision
      result_hi = _mm_srai_epi32(result_hi, FPScale16bits);

      __m128i result_8x_uint16 = _MM_PACKUS_EPI32(result_lo, result_hi); // SSE4.1 simul
      if (lessthan16bit)
        result_8x_uint16 = _MM_MIN_EPU16(result_8x_uint16, clamp_limit); // SSE4.1 simul extra clamp for 10-14 bit
      _mm_store_si128(reinterpret_cast<__m128i*>(dst + x), result_8x_uint16);
    }

#ifdef NON32_BYTES_ALIGNMENT
    // Leftover, slow C
    for (int x = wMod8; x < width; x++) {
      int64_t result64 = 1 << (FPScale16bits - 1); // rounder
      const uint16_t* src2_ptr = src_ptr + x;
      for (int i = 0; i < program->filter_size; i++) {
        //result64 += (src_ptr + pitch_table[i] / sizeof(uint16_t))[x] * (int64_t)current_coeff[i];
        result64 += (int)(*src2_ptr) * (int64_t)current_coeff[i];
        src2_ptr += src_pitch;
      }
      int result = (int)(result64 >> FPScale16bits); // scale back 13 bits
      result = result > limit ? limit : result < 0 ? 0 : result; // clamp 10..16 bits
      dst[x] = (uint16_t)result;
    }
#endif

    dst += dst_pitch;
    current_coeff += program->filter_size;
  }
}

template<bool lessthan16bit, int _filter_size_numOfFullBlk8, int filtersizemod8>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
void internal_resize_v_sse41_planar_uint16_t(BYTE * dst0, const BYTE * src0, int dst_pitch, int src_pitch, ResamplingProgram * program, int width, int target_height, int bits_per_pixel, const int* pitch_table, const void* storage)
{
  AVS_UNUSED(storage);
  int filter_size_numOfFullBlk8 = (program->filter_size / 8);
  if constexpr (_filter_size_numOfFullBlk8 >= 0)
    filter_size_numOfFullBlk8 = _filter_size_numOfFullBlk8;
  //const int filter_size_numOfFullBlk8 = (_filter_size_numOfFullBlk8 >= 0) ? _filter_size_numOfFullBlk8 : (program->filter_size / 8);
  short* current_coeff = program->pixel_coefficient;

  // #define NON32_BYTES_ALIGNMENT
  // in AVS+ 32 bytes alignment is guaranteed
#ifdef NON32_BYTES_ALIGNMENT
  int wMod8 = (width / 8) * 8; // uint16: 8 at a time (2x128bit)
#endif

  const __m128i zero = _mm_setzero_si128();
  const __m128i shifttosigned = _mm_set1_epi16(-32768); // for 16 bits only
  const __m128i shiftfromsigned = _mm_set1_epi32(32768 << FPScale16bits); // for 16 bits only
  const __m128i rounder = _mm_set1_epi32(1 << (FPScale16bits - 1));

  const uint16_t* src = (uint16_t*)src0;
  uint16_t* dst = (uint16_t*)dst0;
  dst_pitch = dst_pitch / sizeof(uint16_t);
  src_pitch = src_pitch / sizeof(uint16_t);

  const int limit = (1 << bits_per_pixel) - 1;
  __m128i clamp_limit = _mm_set1_epi16((short)limit); // clamp limit for <16 bits

  for (int y = 0; y < target_height; y++) {
    int offset = program->pixel_offset[y];
    const uint16_t* src_ptr = src + pitch_table[offset] / sizeof(uint16_t);

#ifdef NON32_BYTES_ALIGNMENT
    for (int x = 0; x < wMod8; x += 8) { // 2x4 pixels at a time
#else
    for (int x = 0; x < width; x += 8) {
#endif
      __m128i result_single_lo = rounder;
      __m128i result_single_hi = rounder;

      const uint16_t* src2_ptr = src_ptr + x;

      for (int i = 0; i < filter_size_numOfFullBlk8; i++) {
        __m128i coeff01234567 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(current_coeff + i * 8)); // 4x (2x16bit) shorts for even/odd

        // offset table generating is what preventing us from overaddressing
        // 0-1
        process_chunk_v_uint16_t<lessthan16bit, 0>(src2_ptr, src_pitch, coeff01234567, result_single_lo, result_single_hi, shifttosigned);
        // 2-3
        process_chunk_v_uint16_t<lessthan16bit, 2>(src2_ptr, src_pitch, coeff01234567, result_single_lo, result_single_hi, shifttosigned);
        // 4-5
        process_chunk_v_uint16_t<lessthan16bit, 4>(src2_ptr, src_pitch, coeff01234567, result_single_lo, result_single_hi, shifttosigned);
        // 6-7
        process_chunk_v_uint16_t<lessthan16bit, 6>(src2_ptr, src_pitch, coeff01234567, result_single_lo, result_single_hi, shifttosigned);
        src2_ptr += 8 * src_pitch;
      }

      // and the rest non-div8 chunk
      __m128i coeff01234567 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(current_coeff + filter_size_numOfFullBlk8 * 8)); // 4x (2x16bit) shorts for even/odd
      if constexpr (filtersizemod8 >= 2)
        process_chunk_v_uint16_t<lessthan16bit, 0>(src2_ptr, src_pitch, coeff01234567, result_single_lo, result_single_hi, shifttosigned);
      if constexpr (filtersizemod8 >= 4)
        process_chunk_v_uint16_t<lessthan16bit, 2>(src2_ptr, src_pitch, coeff01234567, result_single_lo, result_single_hi, shifttosigned);
      if constexpr (filtersizemod8 >= 6)
        process_chunk_v_uint16_t<lessthan16bit, 4>(src2_ptr, src_pitch, coeff01234567, result_single_lo, result_single_hi, shifttosigned);
      if constexpr (filtersizemod8 % 2) { // remaining odd one
        const int index = filtersizemod8 - 1;
        __m128i src_even = _mm_load_si128(reinterpret_cast<const __m128i*>(src2_ptr + index * src_pitch)); // 8x 16bit pixels
        if (!lessthan16bit)
          src_even = _mm_add_epi16(src_even, shifttosigned);
        __m128i coeff = _mm_shuffle_epi32(coeff01234567, ((index / 2) << 0) | ((index / 2) << 2) | ((index / 2) << 4) | ((index / 2) << 6));
        __m128i src_lo = _mm_unpacklo_epi16(src_even, zero); // insert zero after the unsigned->signed shift!
        __m128i src_hi = _mm_unpackhi_epi16(src_even, zero); // insert zero after the unsigned->signed shift!
        result_single_lo = _mm_add_epi32(result_single_lo, _mm_madd_epi16(src_lo, coeff)); // a*b + c
        result_single_hi = _mm_add_epi32(result_single_hi, _mm_madd_epi16(src_hi, coeff)); // a*b + c
      }

      // correct if signed, scale back, store
      __m128i result_lo = result_single_lo;
      __m128i result_hi = result_single_hi;
      if (!lessthan16bit) {
        result_lo = _mm_add_epi32(result_lo, shiftfromsigned);
        result_hi = _mm_add_epi32(result_hi, shiftfromsigned);
      }
      result_lo = _mm_srai_epi32(result_lo, FPScale16bits); // shift back integer arithmetic 13 bits precision
      result_hi = _mm_srai_epi32(result_hi, FPScale16bits);

      __m128i result_8x_uint16 = _mm_packus_epi32(result_lo, result_hi);
      if (lessthan16bit)
        result_8x_uint16 = _mm_min_epu16(result_8x_uint16, clamp_limit); // extra clamp for 10-14 bit
      _mm_store_si128(reinterpret_cast<__m128i*>(dst + x), result_8x_uint16);
    }

#ifdef NON32_BYTES_ALIGNMENT
    // Leftover, slow C
    for (int x = wMod8; x < width; x++) {
      int64_t result64 = 1 << (FPScale16bits - 1); // rounder
      const uint16_t* src2_ptr = src_ptr + x;
      for (int i = 0; i < program->filter_size; i++) {
        //result64 += (src_ptr + pitch_table[i] / sizeof(uint16_t))[x] * (int64_t)current_coeff[i];
        result64 += (int)(*src2_ptr) * (int64_t)current_coeff[i];
        src2_ptr += src_pitch;
      }
      int result = (int)(result64 >> FPScale16bits); // scale back 13 bits
      result = result > limit ? limit : result < 0 ? 0 : result; // clamp 10..16 bits
      dst[x] = (uint16_t)result;
    }
#endif

    dst += dst_pitch;
    current_coeff += program->filter_size;
  }
}

//-------- uint16_t Vertical Dispatcher

template<bool lessthan16bit>
void resize_v_sse2_planar_uint16_t(BYTE * dst0, const BYTE * src0, int dst_pitch, int src_pitch, ResamplingProgram * program, int width, int target_height, int bits_per_pixel, const int* pitch_table, const void* storage)
{
  // template<bool lessthan16bit, int _filter_size_numOfFullBlk8, int filtersizemod8>
  // filtersize 1..16: to template for optimization
  switch (program->filter_size) {
  case 1:
    internal_resize_v_sse2_planar_uint16_t<lessthan16bit, 0, 1>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 2:
    internal_resize_v_sse2_planar_uint16_t<lessthan16bit, 0, 2>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 3:
    internal_resize_v_sse2_planar_uint16_t<lessthan16bit, 0, 3>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 4:
    internal_resize_v_sse2_planar_uint16_t<lessthan16bit, 0, 4>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 5:
    internal_resize_v_sse2_planar_uint16_t<lessthan16bit, 0, 5>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 6:
    internal_resize_v_sse2_planar_uint16_t<lessthan16bit, 0, 6>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 7:
    internal_resize_v_sse2_planar_uint16_t<lessthan16bit, 0, 7>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 8:
    internal_resize_v_sse2_planar_uint16_t<lessthan16bit, 1, 0>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 9:
    internal_resize_v_sse2_planar_uint16_t<lessthan16bit, 1, 1>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 10:
    internal_resize_v_sse2_planar_uint16_t<lessthan16bit, 1, 2>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 11:
    internal_resize_v_sse2_planar_uint16_t<lessthan16bit, 1, 3>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 12:
    internal_resize_v_sse2_planar_uint16_t<lessthan16bit, 1, 4>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 13:
    internal_resize_v_sse2_planar_uint16_t<lessthan16bit, 1, 5>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 14:
    internal_resize_v_sse2_planar_uint16_t<lessthan16bit, 1, 6>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 15:
    internal_resize_v_sse2_planar_uint16_t<lessthan16bit, 1, 7>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  default:
    switch (program->filter_size & 7) {
    case 0:
      internal_resize_v_sse2_planar_uint16_t<lessthan16bit, -1, 0>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
      break;
    case 1:
      internal_resize_v_sse2_planar_uint16_t<lessthan16bit, -1, 1>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
      break;
    case 2:
      internal_resize_v_sse2_planar_uint16_t<lessthan16bit, -1, 2>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
      break;
    case 3:
      internal_resize_v_sse2_planar_uint16_t<lessthan16bit, -1, 3>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
      break;
    case 4:
      internal_resize_v_sse2_planar_uint16_t<lessthan16bit, -1, 4>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
      break;
    case 5:
      internal_resize_v_sse2_planar_uint16_t<lessthan16bit, -1, 5>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
      break;
    case 6:
      internal_resize_v_sse2_planar_uint16_t<lessthan16bit, -1, 6>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
      break;
    case 7:
      internal_resize_v_sse2_planar_uint16_t<lessthan16bit, -1, 7>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
      break;
    }
    break;
  }
}

template void resize_v_sse2_planar_uint16_t<false>(BYTE * dst0, const BYTE * src0, int dst_pitch, int src_pitch, ResamplingProgram * program, int width, int target_height, int bits_per_pixel, const int* pitch_table, const void* storage);
template void resize_v_sse2_planar_uint16_t<true>(BYTE * dst0, const BYTE * src0, int dst_pitch, int src_pitch, ResamplingProgram * program, int width, int target_height, int bits_per_pixel, const int* pitch_table, const void* storage);


template<bool lessthan16bit>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
void resize_v_sse41_planar_uint16_t(BYTE * dst0, const BYTE * src0, int dst_pitch, int src_pitch, ResamplingProgram * program, int width, int target_height, int bits_per_pixel, const int* pitch_table, const void* storage)
{
  // template<bool lessthan16bit, int _filter_size_numOfFullBlk8, int filtersizemod8>
  // filtersize 1..16: to template for optimization
  switch (program->filter_size) {
  case 1:
    internal_resize_v_sse41_planar_uint16_t<lessthan16bit, 0, 1>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 2:
    internal_resize_v_sse41_planar_uint16_t<lessthan16bit, 0, 2>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 3:
    internal_resize_v_sse41_planar_uint16_t<lessthan16bit, 0, 3>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 4:
    internal_resize_v_sse41_planar_uint16_t<lessthan16bit, 0, 4>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 5:
    internal_resize_v_sse41_planar_uint16_t<lessthan16bit, 0, 5>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 6:
    internal_resize_v_sse41_planar_uint16_t<lessthan16bit, 0, 6>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 7:
    internal_resize_v_sse41_planar_uint16_t<lessthan16bit, 0, 7>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 8:
    internal_resize_v_sse41_planar_uint16_t<lessthan16bit, 1, 0>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 9:
    internal_resize_v_sse41_planar_uint16_t<lessthan16bit, 1, 1>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 10:
    internal_resize_v_sse41_planar_uint16_t<lessthan16bit, 1, 2>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 11:
    internal_resize_v_sse41_planar_uint16_t<lessthan16bit, 1, 3>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 12:
    internal_resize_v_sse41_planar_uint16_t<lessthan16bit, 1, 4>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 13:
    internal_resize_v_sse41_planar_uint16_t<lessthan16bit, 1, 5>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 14:
    internal_resize_v_sse41_planar_uint16_t<lessthan16bit, 1, 6>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 15:
    internal_resize_v_sse41_planar_uint16_t<lessthan16bit, 1, 7>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  default:
    switch (program->filter_size & 7) {
    case 0:
      internal_resize_v_sse41_planar_uint16_t<lessthan16bit, -1, 0>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
      break;
    case 1:
      internal_resize_v_sse41_planar_uint16_t<lessthan16bit, -1, 1>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
      break;
    case 2:
      internal_resize_v_sse41_planar_uint16_t<lessthan16bit, -1, 2>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
      break;
    case 3:
      internal_resize_v_sse41_planar_uint16_t<lessthan16bit, -1, 3>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
      break;
    case 4:
      internal_resize_v_sse41_planar_uint16_t<lessthan16bit, -1, 4>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
      break;
    case 5:
      internal_resize_v_sse41_planar_uint16_t<lessthan16bit, -1, 5>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
      break;
    case 6:
      internal_resize_v_sse41_planar_uint16_t<lessthan16bit, -1, 6>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
      break;
    case 7:
      internal_resize_v_sse41_planar_uint16_t<lessthan16bit, -1, 7>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
      break;
    }
    break;
  }
}

template void resize_v_sse41_planar_uint16_t<false>(BYTE * dst0, const BYTE * src0, int dst_pitch, int src_pitch, ResamplingProgram * program, int width, int target_height, int bits_per_pixel, const int* pitch_table, const void* storage);
template void resize_v_sse41_planar_uint16_t<true>(BYTE * dst0, const BYTE * src0, int dst_pitch, int src_pitch, ResamplingProgram * program, int width, int target_height, int bits_per_pixel, const int* pitch_table, const void* storage);


//-------- 128 bit float Verticals

template<int _filtersize>
static void internal_resize_v_sse2_planar_float(BYTE * dst0, const BYTE * src0, int dst_pitch, int src_pitch, ResamplingProgram * program, int width, int target_height, int bits_per_pixel, const int* pitch_table, const void* storage)
{
  AVS_UNUSED(bits_per_pixel);
  AVS_UNUSED(storage);
  // 1..8: special case for compiler optimization
  int filter_size = program->filter_size;
  if constexpr (_filtersize >= 1)
    filter_size = _filtersize;
  float* current_coeff_float = program->pixel_coefficient_float;

  const float* src = (float*)src0;
  float* dst = (float*)dst0;
  dst_pitch = dst_pitch / sizeof(float);
  src_pitch = src_pitch / sizeof(float);

  const int fsmod4 = (filter_size / 4) * 4;
  for (int y = 0; y < target_height; y++) {
    int offset = program->pixel_offset[y];
    const float* src_ptr = src + pitch_table[offset] / sizeof(float);

    // 8 pixels/cycle (32 bytes)
    for (int x = 0; x < width; x += 8) {  // safe to process 8 floats, 32 bytes alignment is OK
      __m128 result_single_lo = _mm_set1_ps(0.0f);
      __m128 result_single_hi = _mm_set1_ps(0.0f);

      const float* src2_ptr = src_ptr + x;

      for (int i = 0; i < fsmod4; i += 4) {
        __m128 src_single_lo;
        __m128 src_single_hi;
        __m128 coeff0123 = _mm_loadu_ps(reinterpret_cast<const float*>(current_coeff_float + i)); // loads 4 floats
        __m128 coeff;

        // unroll 4x
        // #1
        src_single_lo = _mm_load_ps(reinterpret_cast<const float*>(src2_ptr + 0 * src_pitch));
        src_single_hi = _mm_load_ps(reinterpret_cast<const float*>(src2_ptr + 0 * src_pitch + 4));
        coeff = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(coeff0123), (0 << 0) | (0 << 2) | (0 << 4) | (0 << 6))); // spread 0th
        result_single_lo = _mm_add_ps(result_single_lo, _mm_mul_ps(coeff, src_single_lo));
        result_single_hi = _mm_add_ps(result_single_hi, _mm_mul_ps(coeff, src_single_hi));

        // #2
        src_single_lo = _mm_load_ps(reinterpret_cast<const float*>(src2_ptr + 1 * src_pitch));
        src_single_hi = _mm_load_ps(reinterpret_cast<const float*>(src2_ptr + 1 * src_pitch + 4));
        coeff = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(coeff0123), (1 << 0) | (1 << 2) | (1 << 4) | (1 << 6))); // spread 1st
        result_single_lo = _mm_add_ps(result_single_lo, _mm_mul_ps(coeff, src_single_lo));
        result_single_hi = _mm_add_ps(result_single_hi, _mm_mul_ps(coeff, src_single_hi));

        // #3
        src_single_lo = _mm_load_ps(reinterpret_cast<const float*>(src2_ptr + 2 * src_pitch));
        src_single_hi = _mm_load_ps(reinterpret_cast<const float*>(src2_ptr + 2 * src_pitch + 4));
        coeff = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(coeff0123), (2 << 0) | (2 << 2) | (2 << 4) | (2 << 6))); // spread 2nd
        result_single_lo = _mm_add_ps(result_single_lo, _mm_mul_ps(coeff, src_single_lo));
        result_single_hi = _mm_add_ps(result_single_hi, _mm_mul_ps(coeff, src_single_hi));

        // #4
        src_single_lo = _mm_load_ps(reinterpret_cast<const float*>(src2_ptr + 3 * src_pitch));
        src_single_hi = _mm_load_ps(reinterpret_cast<const float*>(src2_ptr + 3 * src_pitch + 4));
        coeff = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(coeff0123), (3 << 0) | (3 << 2) | (3 << 4) | (3 << 6))); // spread 3rd
        result_single_lo = _mm_add_ps(result_single_lo, _mm_mul_ps(coeff, src_single_lo));
        result_single_hi = _mm_add_ps(result_single_hi, _mm_mul_ps(coeff, src_single_hi));

        src2_ptr += 4 * src_pitch;
      }

      // one-by-one
      for (int i = fsmod4; i < filter_size; i++) {
        __m128 src_single_lo = _mm_load_ps(reinterpret_cast<const float*>(src2_ptr + 0 * src_pitch));
        __m128 src_single_hi = _mm_load_ps(reinterpret_cast<const float*>(src2_ptr + 0 * src_pitch + 4));
        __m128 coeff = _mm_load1_ps(reinterpret_cast<const float*>(current_coeff_float + i)); // loads 1, fills all 8 floats
        result_single_lo = _mm_add_ps(result_single_lo, _mm_mul_ps(coeff, src_single_lo)); // _mm_fmadd_ps(src_single, coeff, result_single); // a*b + c
        result_single_hi = _mm_add_ps(result_single_hi, _mm_mul_ps(coeff, src_single_hi)); // _mm_fmadd_ps(src_single, coeff, result_single); // a*b + c

        src2_ptr += src_pitch;
      }

      _mm_stream_ps(reinterpret_cast<float*>(dst + x), result_single_lo);
      _mm_stream_ps(reinterpret_cast<float*>(dst + x + 4), result_single_hi);
    }

#if 0
    // Leftover, Slow C
    for (int x = wMod4; x < width; x++) {
      float result = 0;
      const float* src2_ptr = src_ptr + x;
      for (int i = 0; i < filter_size; i++) {
        result += (*src2_ptr) * current_coeff_float[i];
        src2_ptr += src_pitch;
      }
      dst[x] = result;
    }
#endif
    dst += dst_pitch;
    current_coeff_float += filter_size;
  }
}

//-------- Float Vertical Dispatcher

void resize_v_sse2_planar_float(BYTE * dst0, const BYTE * src0, int dst_pitch, int src_pitch, ResamplingProgram * program, int width, int target_height, int bits_per_pixel, const int* pitch_table, const void* storage)
{
  // 1..8: special case for compiler optimization
  switch (program->filter_size) {
  case 1:
    internal_resize_v_sse2_planar_float<1>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 2:
    internal_resize_v_sse2_planar_float<2>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 3:
    internal_resize_v_sse2_planar_float<3>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 4:
    internal_resize_v_sse2_planar_float<4>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 5:
    internal_resize_v_sse2_planar_float<5>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 6:
    internal_resize_v_sse2_planar_float<6>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 7:
    internal_resize_v_sse2_planar_float<7>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 8:
    internal_resize_v_sse2_planar_float<8>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  default:
    internal_resize_v_sse2_planar_float<-1>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  }
}


//-------- uint8_t Horizontal (8bit)

#if defined(GCC) || defined(CLANG)
__attribute__((__target__("ssse3")))
#endif
void resizer_h_ssse3_generic(BYTE * dst, const BYTE * src, int dst_pitch, int src_pitch, ResamplingProgram * program, int width, int height, int bits_per_pixel)
{
  AVS_UNUSED(bits_per_pixel);

  int filter_size = AlignNumber(program->filter_size, 8) / 8;
  __m128i zero = _mm_setzero_si128();

  for (int y = 0; y < height; y++) {
    short* current_coeff = program->pixel_coefficient;
    for (int x = 0; x < width; x += 4) {
      __m128i result1 = _mm_setr_epi32(8192, 0, 0, 0);
      __m128i result2 = _mm_setr_epi32(8192, 0, 0, 0);
      __m128i result3 = _mm_setr_epi32(8192, 0, 0, 0);
      __m128i result4 = _mm_setr_epi32(8192, 0, 0, 0);

      int begin1 = program->pixel_offset[x + 0];
      int begin2 = program->pixel_offset[x + 1];
      int begin3 = program->pixel_offset[x + 2];
      int begin4 = program->pixel_offset[x + 3];

      for (int i = 0; i < filter_size; i++) {
        __m128i data, coeff, current_result;
        data = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + begin1 + i * 8)); // 8 * 8 bit pixels
        data = _mm_unpacklo_epi8(data, zero); // make 8*16 bit pixels
        coeff = _mm_load_si128(reinterpret_cast<const __m128i*>(current_coeff));  // 8 coeffs 14 bit scaled -> ushort OK
        current_result = _mm_madd_epi16(data, coeff);
        result1 = _mm_add_epi32(result1, current_result);

        current_coeff += 8;
      }

      for (int i = 0; i < filter_size; i++) {
        __m128i data, coeff, current_result;
        data = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + begin2 + i * 8));
        data = _mm_unpacklo_epi8(data, zero);
        coeff = _mm_load_si128(reinterpret_cast<const __m128i*>(current_coeff));
        current_result = _mm_madd_epi16(data, coeff);
        result2 = _mm_add_epi32(result2, current_result);

        current_coeff += 8;
      }

      for (int i = 0; i < filter_size; i++) {
        __m128i data, coeff, current_result;
        data = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + begin3 + i * 8));
        data = _mm_unpacklo_epi8(data, zero);
        coeff = _mm_load_si128(reinterpret_cast<const __m128i*>(current_coeff));
        current_result = _mm_madd_epi16(data, coeff);
        result3 = _mm_add_epi32(result3, current_result);

        current_coeff += 8;
      }

      for (int i = 0; i < filter_size; i++) {
        __m128i data, coeff, current_result;
        data = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + begin4 + i * 8));
        data = _mm_unpacklo_epi8(data, zero);
        coeff = _mm_load_si128(reinterpret_cast<const __m128i*>(current_coeff));
        current_result = _mm_madd_epi16(data, coeff);
        result4 = _mm_add_epi32(result4, current_result);

        current_coeff += 8;
      }

      __m128i result12 = _mm_hadd_epi32(result1, result2);
      __m128i result34 = _mm_hadd_epi32(result3, result4);
      __m128i result = _mm_hadd_epi32(result12, result34);

      result = _mm_srai_epi32(result, 14);

      result = _mm_packs_epi32(result, zero);
      result = _mm_packus_epi16(result, zero);

      *((int*)(dst + x)) = _mm_cvtsi128_si32(result);
    }

    dst += dst_pitch;
    src += src_pitch;
  }
}

#if defined(GCC) || defined(CLANG)
__attribute__((__target__("ssse3")))
#endif
void resizer_h_ssse3_8(BYTE * dst, const BYTE * src, int dst_pitch, int src_pitch, ResamplingProgram * program, int width, int height, int bits_per_pixel)
{
  AVS_UNUSED(bits_per_pixel);

  __m128i zero = _mm_setzero_si128();

  for (int y = 0; y < height; y++) {
    short* current_coeff = program->pixel_coefficient;
    for (int x = 0; x < width; x += 4) {
      __m128i result1 = _mm_setr_epi32(8192, 0, 0, 0);
      __m128i result2 = _mm_setr_epi32(8192, 0, 0, 0);
      __m128i result3 = _mm_setr_epi32(8192, 0, 0, 0);
      __m128i result4 = _mm_setr_epi32(8192, 0, 0, 0);

      int begin1 = program->pixel_offset[x + 0];
      int begin2 = program->pixel_offset[x + 1];
      int begin3 = program->pixel_offset[x + 2];
      int begin4 = program->pixel_offset[x + 3];

      __m128i data, coeff, current_result;

      // Unroll 1
      data = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + begin1));
      data = _mm_unpacklo_epi8(data, zero);
      coeff = _mm_load_si128(reinterpret_cast<const __m128i*>(current_coeff));
      current_result = _mm_madd_epi16(data, coeff);
      result1 = _mm_add_epi32(result1, current_result);

      current_coeff += 8;

      // Unroll 2
      data = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + begin2));
      data = _mm_unpacklo_epi8(data, zero);
      coeff = _mm_load_si128(reinterpret_cast<const __m128i*>(current_coeff));
      current_result = _mm_madd_epi16(data, coeff);
      result2 = _mm_add_epi32(result2, current_result);

      current_coeff += 8;

      // Unroll 3
      data = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + begin3));
      data = _mm_unpacklo_epi8(data, zero);
      coeff = _mm_load_si128(reinterpret_cast<const __m128i*>(current_coeff));
      current_result = _mm_madd_epi16(data, coeff);
      result3 = _mm_add_epi32(result3, current_result);

      current_coeff += 8;

      // Unroll 4
      data = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + begin4));
      data = _mm_unpacklo_epi8(data, zero);
      coeff = _mm_load_si128(reinterpret_cast<const __m128i*>(current_coeff));
      current_result = _mm_madd_epi16(data, coeff);
      result4 = _mm_add_epi32(result4, current_result);

      current_coeff += 8;

      // Combine
      __m128i result12 = _mm_hadd_epi32(result1, result2);
      __m128i result34 = _mm_hadd_epi32(result3, result4);
      __m128i result = _mm_hadd_epi32(result12, result34);

      result = _mm_srai_epi32(result, 14);

      result = _mm_packs_epi32(result, zero);
      result = _mm_packus_epi16(result, zero);

      *((int*)(dst + x)) = _mm_cvtsi128_si32(result);
    }

    dst += dst_pitch;
    src += src_pitch;
  }
}

