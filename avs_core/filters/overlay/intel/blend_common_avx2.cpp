// Avisynth+
// https://avs-plus.net
//
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

#include "avisynth.h"
#include "blend_common_avx2.h"
#include "../blend_common.h"

#include <stdint.h>
#include <vector>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include <immintrin.h>

// simd_magic_div_32_avx2 is in masked_rowprep_avx2_impl.h (included by masked_merge_avx2_impl.hpp).
// blend8_masked_avx2_row, blend16_masked_avx2_row, masked_merge_avx2_impl
// (implementation-include, no guards — each TU gets its own compiled copy)
#include "masked_merge_avx2_impl.hpp"

// ---------------------------------------------------------------------------
// Family 1: weighted_merge — AVX2 (no mask, flat weight, >> 15 shift)
// weight + invweight == 32768; boundary values (0, 32768) are caller early-outs.
// ---------------------------------------------------------------------------
// Important: the two extremes 0 and 32768 (15 bit arith) are handled specially earlier, returning
// exactly data from p1 or p2 respectively. (32768 mask value does not fit into signed int16)
static void weighted_merge_uint8_avx2_impl(
  BYTE* p1, const BYTE* p2, int p1_pitch, int p2_pitch,
  int rowsize, int height, int weight_i, int invweight_i)
{
  // interleaved mask: invweight in lo word, weight in hi word → madd gives p1*inv + p2*w
  const auto mask       = _mm256_set1_epi32(invweight_i | (weight_i << 16));
  const auto round_mask = _mm256_set1_epi32(0x4000);
  const auto zero       = _mm256_setzero_si256();

  const auto mask_128       = _mm_set1_epi32(invweight_i | (weight_i << 16));
  const auto round_mask_128 = _mm_set1_epi32(0x4000);
  const auto zero_128       = _mm_setzero_si128();

  const int wMod32 = (rowsize / 32) * 32;
  const int wMod16 = (rowsize / 16) * 16;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < wMod32; x += 32) {
      auto px1  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p1 + x));
      auto px2  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p2 + x));

      auto p07   = _mm256_unpacklo_epi8(px1, px2);   // interleave p1/p2 bytes
      auto p815  = _mm256_unpackhi_epi8(px1, px2);

      auto p03   = _mm256_unpacklo_epi8(p07, zero);  // zero-extend to 16-bit pairs
      auto p47   = _mm256_unpackhi_epi8(p07, zero);
      auto p811  = _mm256_unpacklo_epi8(p815, zero);
      auto p1215 = _mm256_unpackhi_epi8(p815, zero);

      p03   = _mm256_add_epi32(_mm256_madd_epi16(p03,   mask), round_mask);
      p47   = _mm256_add_epi32(_mm256_madd_epi16(p47,   mask), round_mask);
      p811  = _mm256_add_epi32(_mm256_madd_epi16(p811,  mask), round_mask);
      p1215 = _mm256_add_epi32(_mm256_madd_epi16(p1215, mask), round_mask);

      p03   = _mm256_srli_epi32(p03,   15);
      p47   = _mm256_srli_epi32(p47,   15);
      p811  = _mm256_srli_epi32(p811,  15);
      p1215 = _mm256_srli_epi32(p1215, 15);

      p07  = _mm256_packs_epi32(p03,  p47);
      p815 = _mm256_packs_epi32(p811, p1215);

      _mm256_storeu_si256(reinterpret_cast<__m256i*>(p1 + x),
        _mm256_packus_epi16(p07, p815));
    }

    for (int x = wMod32; x < wMod16; x += 16) {
      auto px1  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p1 + x));
      auto px2  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p2 + x));

      auto p07   = _mm_unpacklo_epi8(px1, px2);
      auto p815  = _mm_unpackhi_epi8(px1, px2);

      auto p03   = _mm_unpacklo_epi8(p07, zero_128);
      auto p47   = _mm_unpackhi_epi8(p07, zero_128);
      auto p811  = _mm_unpacklo_epi8(p815, zero_128);
      auto p1215 = _mm_unpackhi_epi8(p815, zero_128);

      p03   = _mm_add_epi32(_mm_madd_epi16(p03,   mask_128), round_mask_128);
      p47   = _mm_add_epi32(_mm_madd_epi16(p47,   mask_128), round_mask_128);
      p811  = _mm_add_epi32(_mm_madd_epi16(p811,  mask_128), round_mask_128);
      p1215 = _mm_add_epi32(_mm_madd_epi16(p1215, mask_128), round_mask_128);

      p03   = _mm_srli_epi32(p03,   15);
      p47   = _mm_srli_epi32(p47,   15);
      p811  = _mm_srli_epi32(p811,  15);
      p1215 = _mm_srli_epi32(p1215, 15);

      p07  = _mm_packs_epi32(p03,  p47);
      p815 = _mm_packs_epi32(p811, p1215);

      _mm_storeu_si128(reinterpret_cast<__m128i*>(p1 + x),
        _mm_packus_epi16(p07, p815));
    }

    // Scalar tail
    for (int x = wMod16; x < rowsize; ++x)
      p1[x] = (uint8_t)((p1[x] * invweight_i + p2[x] * weight_i + 16384) >> 15);

    p1 += p1_pitch;
    p2 += p2_pitch;
  }
}

// Important: the two extremes 0 and 32768 (15 bit arith) are handled specially earlier, returning
// exactly data from p1 or p2 respectively. (32768 mask value does not fit into signed int16)
// lessthan16bit=true:  10/12/14-bit — no signed pivot needed (pixel values fit positive int16)
// lessthan16bit=false: full 16-bit  — signed pivot to avoid unsigned overflow in madd
template<bool lessthan16bit>
static void weighted_merge_uint16_avx2_impl(
  BYTE* p1, const BYTE* p2, int p1_pitch, int p2_pitch,
  int rowsize, int height, int weight_i, int invweight_i)
{
  // After unpacklo_epi16(px1,px2): pairs are (px1[i], px2[i]).
  // mask lo=invweight, hi=weight → madd gives px1*invweight + px2*weight.
  const __m256i mask         = _mm256_set1_epi32((weight_i << 16) + invweight_i);
  const __m256i round_mask   = _mm256_set1_epi32(0x4000);
  const __m256i signed_shift = _mm256_set1_epi16(-32768);  // add to convert u16→s16

  const __m128i mask_128         = _mm_set1_epi32((weight_i << 16) + invweight_i);
  const __m128i round_mask_128   = _mm_set1_epi32(0x4000);
  const __m128i signed_shift_128 = _mm_set1_epi16(-32768);

  const int wMod32 = (rowsize / 32) * 32;
  const int wMod16 = (rowsize / 16) * 16;

  for (int y = 0; y < height; y++) {
    // 32-byte (16 pixels) AVX2 loop
    for (int x = 0; x < wMod32; x += 32) {
      __m256i px1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p1 + x));
      __m256i px2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p2 + x));

      if constexpr (!lessthan16bit) {
        // Shift unsigned 0..65535 → signed -32768..32767 for madd safety
        px1 = _mm256_add_epi16(px1, signed_shift);
        px2 = _mm256_add_epi16(px2, signed_shift);
      }

      auto p03 = _mm256_unpacklo_epi16(px1, px2);
      auto p47 = _mm256_unpackhi_epi16(px1, px2);

      p03 = _mm256_add_epi32(_mm256_madd_epi16(p03, mask), round_mask);
      p47 = _mm256_add_epi32(_mm256_madd_epi16(p47, mask), round_mask);

      p03 = _mm256_srai_epi32(p03, 15);
      p47 = _mm256_srai_epi32(p47, 15);

      auto p07 = _mm256_packs_epi32(p03, p47);
      if constexpr (!lessthan16bit) {
        // Restore: signed → unsigned by adding 32768 (modular: -32768 + 32768 = 0)
        p07 = _mm256_add_epi16(p07, signed_shift);
      }

      _mm256_storeu_si256(reinterpret_cast<__m256i*>(p1 + x), p07);
    }

    // 16-byte (8 pixels) fallback
    for (int x = wMod32; x < wMod16; x += 16) {
      __m128i px1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p1 + x));
      __m128i px2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p2 + x));

      if constexpr (!lessthan16bit) {
        px1 = _mm_add_epi16(px1, signed_shift_128);
        px2 = _mm_add_epi16(px2, signed_shift_128);
      }

      auto p03 = _mm_unpacklo_epi16(px1, px2);
      auto p47 = _mm_unpackhi_epi16(px1, px2);

      p03 = _mm_add_epi32(_mm_madd_epi16(p03, mask_128), round_mask_128);
      p47 = _mm_add_epi32(_mm_madd_epi16(p47, mask_128), round_mask_128);

      p03 = _mm_srai_epi32(p03, 15);
      p47 = _mm_srai_epi32(p47, 15);

      auto p07 = _mm_packs_epi32(p03, p47);
      if constexpr (!lessthan16bit) {
        p07 = _mm_add_epi16(p07, signed_shift_128);
      }

      _mm_storeu_si128(reinterpret_cast<__m128i*>(p1 + x), p07);
    }

    // Scalar tail (in pixels)
    for (int x = wMod16 / 2; x < rowsize / 2; ++x) {
      reinterpret_cast<uint16_t*>(p1)[x] = (uint16_t)(
        (reinterpret_cast<uint16_t*>(p1)[x] * invweight_i +
         reinterpret_cast<const uint16_t*>(p2)[x] * weight_i + 16384) >> 15);
    }

    p1 += p1_pitch;
    p2 += p2_pitch;
  }
}

void weighted_merge_avx2(BYTE* p1, const BYTE* p2, int p1_pitch, int p2_pitch,
  int width, int height, int weight, int invweight, int bits_per_pixel)
{
  const int pixelsize = bits_per_pixel <= 8 ? 1 : 2;
  const int rowsize   = width * pixelsize;

  if (bits_per_pixel == 8)
    weighted_merge_uint8_avx2_impl(p1, p2, p1_pitch, p2_pitch, rowsize, height, weight, invweight);
  else if (bits_per_pixel == 16)
    weighted_merge_uint16_avx2_impl<false>(p1, p2, p1_pitch, p2_pitch, rowsize, height, weight, invweight);
  else  // 10, 12, 14-bit
    weighted_merge_uint16_avx2_impl<true>(p1, p2, p1_pitch, p2_pitch, rowsize, height, weight, invweight);
}

void weighted_merge_float_avx2(BYTE* p1, const BYTE* p2, int p1_pitch, int p2_pitch,
  int width, int height, float weight_f)
{
  const float invweight_f = 1.0f - weight_f;
  const auto  v_weight    = _mm256_set1_ps(weight_f);
  const auto  v_invweight = _mm256_set1_ps(invweight_f);
  const int   rowsize     = width * 4;
  const int   wMod32      = (rowsize / 32) * 32;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < wMod32; x += 32) {
      auto px1 = _mm256_loadu_ps(reinterpret_cast<const float*>(p1 + x));
      auto px2 = _mm256_loadu_ps(reinterpret_cast<const float*>(p2 + x));
      // p1 + (p2 - p1) * w  ==  p1*(1-w) + p2*w
      // choose the latter to match C ref
      auto res = _mm256_fmadd_ps(px1, v_invweight, _mm256_mul_ps(px2, v_weight));
      // Linear interp: auto res = _mm256_add_ps(px1, _mm256_mul_ps(_mm256_sub_ps(px2, px1), v_weight))
      _mm256_storeu_ps(reinterpret_cast<float*>(p1 + x), res);
    }
    // Scalar tail
    for (int x = wMod32 / 4; x < width; ++x) {
      reinterpret_cast<float*>(p1)[x] =
        reinterpret_cast<float*>(p1)[x] * invweight_f +
        reinterpret_cast<const float*>(p2)[x] * weight_f;
    }
    p1 += p1_pitch;
    p2 += p2_pitch;
  }
}

// ---------------------------------------------------------------------------
// Overlay blend masked getter — returns masked_merge_avx2_impl instantiation.
// is_chroma=false -> always MASK444 (luma).
// is_chroma=true  -> placement-aware maskMode (chroma).
// ---------------------------------------------------------------------------
masked_merge_fn_t* get_overlay_blend_masked_fn_avx2(bool is_chroma, MaskMode maskMode)
{
#define DISPATCH_OVERLAY_BLEND_AVX2(MaskType) \
  return is_chroma ? masked_merge_avx2_impl<MaskType> \
                   : masked_merge_avx2_impl<MASK444>;

  switch (maskMode) {
  case MASK444:          DISPATCH_OVERLAY_BLEND_AVX2(MASK444)
  case MASK420:          DISPATCH_OVERLAY_BLEND_AVX2(MASK420)
  case MASK420_MPEG2:    DISPATCH_OVERLAY_BLEND_AVX2(MASK420_MPEG2)
  case MASK420_TOPLEFT:  DISPATCH_OVERLAY_BLEND_AVX2(MASK420_TOPLEFT)
  case MASK422:          DISPATCH_OVERLAY_BLEND_AVX2(MASK422)
  case MASK422_MPEG2:    DISPATCH_OVERLAY_BLEND_AVX2(MASK422_MPEG2)
  case MASK411:          DISPATCH_OVERLAY_BLEND_AVX2(MASK411)
  case MASK411_TOPLEFT:  DISPATCH_OVERLAY_BLEND_AVX2(MASK411_TOPLEFT)
  case MASK440:          DISPATCH_OVERLAY_BLEND_AVX2(MASK440)
  case MASK440_TOPLEFT:  DISPATCH_OVERLAY_BLEND_AVX2(MASK440_TOPLEFT)
  case MASK410:          DISPATCH_OVERLAY_BLEND_AVX2(MASK410)
  case MASK410_TOPLEFT:  DISPATCH_OVERLAY_BLEND_AVX2(MASK410_TOPLEFT)
  case MASK410_MPEG2  :  DISPATCH_OVERLAY_BLEND_AVX2(MASK410_MPEG2  )
  case MASK_MODE_COUNT: break;
  }
#undef DISPATCH_OVERLAY_BLEND_AVX2
  return masked_merge_avx2_impl<MASK444>; // unreachable
}

// ---------------------------------------------------------------------------
// Per-row chroma mask preparation for the scratch path in OF_blend.cpp.
// Defined here so masked_rowprep_avx2_impl.h is only included in this TU.
// ---------------------------------------------------------------------------
template<typename pixel_t, bool full_opacity>
void do_fill_chroma_row_avx2(
  pixel_t* buf, const pixel_t* luma_row,
  int luma_pitch_pixels, int chroma_w, int rows_above, int rows_below, MaskMode mode,
  int opacity_i, int half, MagicDiv magic)
{
  switch (mode) {
  case MASK411:
    prepare_effective_mask_for_row_avx2<MASK411, pixel_t, full_opacity>(luma_row, luma_pitch_pixels, chroma_w, rows_above, rows_below, buf, opacity_i, half, magic); break;
  case MASK420:
    prepare_effective_mask_for_row_avx2<MASK420, pixel_t, full_opacity>(luma_row, luma_pitch_pixels, chroma_w, rows_above, rows_below, buf, opacity_i, half, magic); break;
  case MASK420_MPEG2:
    prepare_effective_mask_for_row_avx2<MASK420_MPEG2, pixel_t, full_opacity>(luma_row, luma_pitch_pixels, chroma_w, rows_above, rows_below, buf, opacity_i, half, magic); break;
  case MASK420_TOPLEFT:
    prepare_effective_mask_for_row_avx2<MASK420_TOPLEFT, pixel_t, full_opacity>(luma_row, luma_pitch_pixels, chroma_w, rows_above, rows_below, buf, opacity_i, half, magic); break;
  case MASK422:
    prepare_effective_mask_for_row_avx2<MASK422, pixel_t, full_opacity>(luma_row, luma_pitch_pixels, chroma_w, rows_above, rows_below, buf, opacity_i, half, magic); break;
  case MASK422_MPEG2:
    prepare_effective_mask_for_row_avx2<MASK422_MPEG2, pixel_t, full_opacity>(luma_row, luma_pitch_pixels, chroma_w, rows_above, rows_below, buf, opacity_i, half, magic); break;
  case MASK440:
    prepare_effective_mask_for_row_avx2<MASK440, pixel_t, full_opacity>(luma_row, luma_pitch_pixels, chroma_w, rows_above, rows_below, buf, opacity_i, half, magic); break;
  case MASK410:
    prepare_effective_mask_for_row_avx2<MASK410, pixel_t, full_opacity>(luma_row, luma_pitch_pixels, chroma_w, rows_above, rows_below, buf, opacity_i, half, magic); break;
  case MASK411_TOPLEFT:
    prepare_effective_mask_for_row_avx2<MASK411_TOPLEFT, pixel_t, full_opacity>(luma_row, luma_pitch_pixels, chroma_w, rows_above, rows_below, buf, opacity_i, half, magic); break;
  case MASK440_TOPLEFT:
    prepare_effective_mask_for_row_avx2<MASK440_TOPLEFT, pixel_t, full_opacity>(luma_row, luma_pitch_pixels, chroma_w, rows_above, rows_below, buf, opacity_i, half, magic); break;
  case MASK410_TOPLEFT:
    prepare_effective_mask_for_row_avx2<MASK410_TOPLEFT, pixel_t, full_opacity>(luma_row, luma_pitch_pixels, chroma_w, rows_above, rows_below, buf, opacity_i, half, magic); break;
  case MASK410_MPEG2  :
    prepare_effective_mask_for_row_avx2<MASK410_MPEG2  , pixel_t, full_opacity>(luma_row, luma_pitch_pixels, chroma_w, rows_above, rows_below, buf, opacity_i, half, magic); break;
  default: break;
  }
}

template void do_fill_chroma_row_avx2<uint8_t, true>(uint8_t*, const uint8_t*, int, int, int, int, MaskMode, int, int, MagicDiv);
template void do_fill_chroma_row_avx2<uint8_t, false>(uint8_t*, const uint8_t*, int, int, int, int, MaskMode, int, int, MagicDiv);
template void do_fill_chroma_row_avx2<uint16_t, true>(uint16_t*, const uint16_t*, int, int, int, int, MaskMode, int, int, MagicDiv);
template void do_fill_chroma_row_avx2<uint16_t, false>(uint16_t*, const uint16_t*, int, int, int, int, MaskMode, int, int, MagicDiv);


template<bool full_opacity>
void do_fill_chroma_row_float_avx2(
  float* buf, const float* luma_row,
  int luma_pitch_pixels, int chroma_w, int rows_above, int rows_below, MaskMode mode,
  float opacity)
{
  switch (mode) {
  case MASK411:
    prepare_effective_mask_for_row_float_avx2<MASK411,          full_opacity>(luma_row, luma_pitch_pixels, chroma_w, rows_above, rows_below, buf, opacity); break;
  case MASK420:
    prepare_effective_mask_for_row_float_avx2<MASK420,          full_opacity>(luma_row, luma_pitch_pixels, chroma_w, rows_above, rows_below, buf, opacity); break;
  case MASK420_MPEG2:
    prepare_effective_mask_for_row_float_avx2<MASK420_MPEG2,    full_opacity>(luma_row, luma_pitch_pixels, chroma_w, rows_above, rows_below, buf, opacity); break;
  case MASK420_TOPLEFT:
    prepare_effective_mask_for_row_float_avx2<MASK420_TOPLEFT,  full_opacity>(luma_row, luma_pitch_pixels, chroma_w, rows_above, rows_below, buf, opacity); break;
  case MASK422:
    prepare_effective_mask_for_row_float_avx2<MASK422,          full_opacity>(luma_row, luma_pitch_pixels, chroma_w, rows_above, rows_below, buf, opacity); break;
  case MASK422_MPEG2:
    prepare_effective_mask_for_row_float_avx2<MASK422_MPEG2,    full_opacity>(luma_row, luma_pitch_pixels, chroma_w, rows_above, rows_below, buf, opacity); break;
  case MASK440:
    prepare_effective_mask_for_row_float_avx2 < MASK440, full_opacity>(luma_row, luma_pitch_pixels, chroma_w, rows_above, rows_below, buf, opacity); break;
  case MASK410:
    prepare_effective_mask_for_row_float_avx2 < MASK410, full_opacity>(luma_row, luma_pitch_pixels, chroma_w, rows_above, rows_below, buf, opacity); break;
  case MASK411_TOPLEFT:
    prepare_effective_mask_for_row_float_avx2 < MASK411_TOPLEFT, full_opacity>(luma_row, luma_pitch_pixels, chroma_w, rows_above, rows_below, buf, opacity); break;
  case MASK440_TOPLEFT:
    prepare_effective_mask_for_row_float_avx2 < MASK440_TOPLEFT, full_opacity>(luma_row, luma_pitch_pixels, chroma_w, rows_above, rows_below, buf, opacity); break;
  case MASK410_TOPLEFT:
    prepare_effective_mask_for_row_float_avx2 < MASK410_TOPLEFT, full_opacity>(luma_row, luma_pitch_pixels, chroma_w, rows_above, rows_below, buf, opacity); break;
  case MASK410_MPEG2  :
    prepare_effective_mask_for_row_float_avx2 < MASK410_MPEG2  , full_opacity>(luma_row, luma_pitch_pixels, chroma_w, rows_above, rows_below, buf, opacity); break;
  default: break;
  }
}

template void do_fill_chroma_row_float_avx2<true> (float*,  const float*,  int, int, int, int, MaskMode, float);
template void do_fill_chroma_row_float_avx2<false>(float*,  const float*,  int, int, int, int, MaskMode, float);

// and for float:
masked_merge_float_fn_t* get_overlay_blend_masked_float_fn_avx2(bool is_chroma, MaskMode maskMode)
{
#define DISPATCH_OVERLAY_BLEND_FLOAT_AVX2(MaskType) \
  return is_chroma ? masked_merge_float_avx2_impl<MaskType> \
                   : masked_merge_float_avx2_impl<MASK444>;

  switch (maskMode) {
  case MASK444:          DISPATCH_OVERLAY_BLEND_FLOAT_AVX2(MASK444)
  case MASK420:          DISPATCH_OVERLAY_BLEND_FLOAT_AVX2(MASK420)
  case MASK420_MPEG2:    DISPATCH_OVERLAY_BLEND_FLOAT_AVX2(MASK420_MPEG2)
  case MASK420_TOPLEFT:  DISPATCH_OVERLAY_BLEND_FLOAT_AVX2(MASK420_TOPLEFT)
  case MASK422:          DISPATCH_OVERLAY_BLEND_FLOAT_AVX2(MASK422)
  case MASK422_MPEG2:    DISPATCH_OVERLAY_BLEND_FLOAT_AVX2(MASK422_MPEG2)
  case MASK411:          DISPATCH_OVERLAY_BLEND_FLOAT_AVX2(MASK411)
  case MASK411_TOPLEFT:  DISPATCH_OVERLAY_BLEND_FLOAT_AVX2(MASK411_TOPLEFT)
  case MASK440:          DISPATCH_OVERLAY_BLEND_FLOAT_AVX2(MASK440)
  case MASK440_TOPLEFT:  DISPATCH_OVERLAY_BLEND_FLOAT_AVX2(MASK440_TOPLEFT)
  case MASK410:          DISPATCH_OVERLAY_BLEND_FLOAT_AVX2(MASK410)
  case MASK410_TOPLEFT:  DISPATCH_OVERLAY_BLEND_FLOAT_AVX2(MASK410_TOPLEFT)
  case MASK410_MPEG2  :  DISPATCH_OVERLAY_BLEND_FLOAT_AVX2(MASK410_MPEG2  )
  case MASK_MODE_COUNT: break;
  }
#undef DISPATCH_OVERLAY_BLEND_FLOAT_AVX2
  return masked_merge_float_avx2_impl<MASK444>; // unreachable
}

