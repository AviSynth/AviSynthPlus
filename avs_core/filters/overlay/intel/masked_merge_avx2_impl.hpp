// AviSynth+  Copyright 2026- AviSynth+ Project
// SPDX-License-Identifier: GPL-2.0-or-later
//
// AVX2 masked-merge implementation templates.
//
// Implementation-include — NO include guards intentionally.
// Each including TU gets its own compilation with that TU's SIMD flags.
// All definitions are static so each TU has its own private copy;
// the linker never sees them as the same symbol.
//
// Requires the following to be visible in the including TU before this file:
//   AVX2 intrinsics  (<intrin.h> or <x86intrin.h>, <immintrin.h>)
//   blend_common.h   — MaskMode, MagicDiv, get_magic_div, magic_div_rt,
//                      prepare_effective_mask_for_row
//   <vector>

// Provides simd_magic_div_32_avx2 + prepare_effective_mask_for_row_avx2.
#include "masked_rowprep_avx2_impl.h"
#include "../../../core/scratch_buffer.h"

// ---------------------------------------------------------------------------
// 8-bit row — mask already has opacity baked in by rowprep.
// 32-wide 16-bit mulhi arithmetic (fast, overflow-safe).
// ---------------------------------------------------------------------------
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("avx2")))
#endif
static void blend8_masked_avx2_row(
  uint8_t* p1, const uint8_t* p2, const uint8_t* mask,
  int width)
{
  constexpr uint32_t half    = 127u;
  constexpr uint32_t max_val = 255u;

  const __m256i v_half  = _mm256_set1_epi16((short)half);
  const __m256i v_max   = _mm256_set1_epi16((short)max_val);
  const __m256i v_magic = _mm256_set1_epi16((short)0x8081);

  auto magic_byte = [&](__m256i x) {
    return _mm256_srli_epi16(_mm256_mulhi_epu16(x, v_magic), 7);
  };

  int x = 0;
  for (; x <= width - 32; x += 32) {
    __m256i mr = _mm256_loadu_si256((__m256i*)(mask + x));

    // Unpack 32 bytes to 2x 16x16-bit halves
    __m256i m_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(mr));
    __m256i m_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(mr, 1));

    __m256i ar = _mm256_loadu_si256((__m256i*)(p1 + x));
    __m256i br = _mm256_loadu_si256((__m256i*)(p2 + x));

    auto blend16 = [&](__m256i a, __m256i b, __m256i m) {
      __m256i invm = _mm256_sub_epi16(v_max, m);
      return magic_byte(_mm256_add_epi16(
        _mm256_add_epi16(_mm256_mullo_epi16(a, invm), _mm256_mullo_epi16(b, m)), v_half));
    };

    __m256i r_lo = blend16(_mm256_cvtepu8_epi16(_mm256_castsi256_si128(ar)),
                           _mm256_cvtepu8_epi16(_mm256_castsi256_si128(br)), m_lo);
    __m256i r_hi = blend16(_mm256_cvtepu8_epi16(_mm256_extracti128_si256(ar, 1)),
                           _mm256_cvtepu8_epi16(_mm256_extracti128_si256(br, 1)), m_hi);

    __m256i packed = _mm256_packus_epi16(r_lo, r_hi);
    packed = _mm256_permute4x64_epi64(packed, _MM_SHUFFLE(3, 1, 2, 0));
    _mm256_storeu_si256((__m256i*)(p1 + x), packed);
  }
  for (; x < width; ++x) {
    const uint32_t ms = mask[x];
    const uint32_t a  = p1[x], b_v = p2[x];
    const uint32_t tr = a * (max_val - ms) + b_v * ms + half;
    p1[x] = (uint8_t)((tr * 0x8081u) >> 23);
  }
}

// ---------------------------------------------------------------------------
// 16-bit row (10, 12, 14, 16 bits) — mask already has opacity baked in.
// 32-bit arithmetic, 8 pixels per step.
// ---------------------------------------------------------------------------
template<int bits_per_pixel>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("avx2")))
#endif
static void blend16_masked_avx2_row(
  uint16_t* p1, const uint16_t* p2, const uint16_t* mask,
  int width)
{
  constexpr MagicDiv m_div   = get_magic_div(bits_per_pixel);
  constexpr uint32_t max_val = (1u << bits_per_pixel) - 1;
  constexpr uint32_t half    = max_val / 2;

  const __m256i v_half = _mm256_set1_epi32((int)half);
  const __m256i v_max  = _mm256_set1_epi32((int)max_val);

  int x = 0;
  for (; x <= width - 8; x += 8) {
    __m256i m    = _mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i*)(mask + x)));
    __m256i a    = _mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i*)(p1 + x)));
    __m256i b    = _mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i*)(p2 + x)));
    __m256i invm = _mm256_sub_epi32(v_max, m);
    __m256i res  = simd_magic_div_32_avx2(
                     _mm256_add_epi32(
                       _mm256_add_epi32(_mm256_mullo_epi32(a, invm), _mm256_mullo_epi32(b, m)),
                       v_half),
                     m_div.div, m_div.shift);

    // Pack 8x32 → 8x16 with correct order
    __m256i packed = _mm256_packus_epi32(res, _mm256_setzero_si256());
    __m128i lo = _mm256_castsi256_si128(packed);
    __m128i hi = _mm256_extracti128_si256(packed, 1);
    _mm_storeu_si128((__m128i*)(p1 + x), _mm_unpacklo_epi64(lo, hi));
  }
  for (; x < width; ++x) {
    const uint32_t ms = mask[x];
    const uint32_t a  = p1[x];
    p1[x] = (uint16_t)magic_div_rt<uint16_t>(a * (max_val - ms) + (uint32_t)p2[x] * ms + half, m_div);
  }
}

// ---------------------------------------------------------------------------
// float row — mask already has opacity baked in (range 0.0 to 1.0).
// Linear interpolation: p1[x] = p1[x] * (1.0f - mask[x]) + p2[x] * mask[x]
// 8 pixels per step.
// ---------------------------------------------------------------------------
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("avx2")))
#endif
static void blend_masked_float_avx2_row(
  float* p1, const float* p2, const float* mask, int width)
{
  int x = 0;
  // const __m256 v_one = _mm256_set1_ps(1.0f); in case of 1-x implementation

  for (; x <= width - 8; x += 8) {
    __m256 m = _mm256_loadu_ps(mask + x);
    __m256 a = _mm256_loadu_ps(p1 + x);
    __m256 b = _mm256_loadu_ps(p2 + x);

    // Standard lerp: a + m * (b - a)
    // This is generally more accurate and faster (1 mul, 2 adds) than 
    // a * (1-m) + b * m (2 muls, 2 adds).
    __m256 diff = _mm256_sub_ps(b, a);
    __m256 res = _mm256_add_ps(a, _mm256_mul_ps(m, diff));

    _mm256_storeu_ps(p1 + x, res);
  }

  // Scalar tail
  for (; x < width; ++x) {
    const float m = mask[x];
    const float a = p1[x];
    const float b = p2[x];
    p1[x] = a + m * (b - a);
  }
}

// ---------------------------------------------------------------------------
// Inner loop — full_opacity known at compile time.
// Rowprep bakes opacity when !full_opacity; blend row receives pre-scaled mask.
// MASK444 + full_opacity: returns mask ptr directly (no buffer, no copy).
// MASK444 + !full_opacity: copies row with opacity scaling into buffer.
// ---------------------------------------------------------------------------
template<MaskMode maskMode, bool full_opacity>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("avx2")))
#endif
static void masked_merge_avx2_impl_inner(
  BYTE* p1, const BYTE* p2, const BYTE* mask,
  int p1_pitch, int p2_pitch, int mask_pitch,
  int width, int height, int opacity, int bits_per_pixel)
{
  const MagicDiv mag  = get_magic_div(bits_per_pixel);
  const int max_val   = (1 << bits_per_pixel) - 1;
  const int half      = max_val / 2;

  if (bits_per_pixel == 8) {
    const uint8_t* maskp = reinterpret_cast<const uint8_t*>(mask);
    const int mpx      = mask_pitch;
    const int mask_adv = mpx * MaskVSubsample<maskMode>;

    ScratchBuffer<uint8_t> eff_buf;
    if constexpr (maskMode != MASK444 || !full_opacity) eff_buf.resize(width);

    for (int y = 0; y < height; y++) {
      const uint8_t* eff = prepare_effective_mask_for_row_avx2<maskMode, uint8_t, full_opacity>(
        maskp, mpx, width, eff_buf.data(), opacity, half, mag);
      blend8_masked_avx2_row(
        reinterpret_cast<uint8_t*>(p1), reinterpret_cast<const uint8_t*>(p2), eff, width);
      p1 += p1_pitch; p2 += p2_pitch; maskp += mask_adv;
    }
    return;
  }

  const uint16_t* maskp = reinterpret_cast<const uint16_t*>(mask);
  const int mpx      = mask_pitch / 2;
  const int mask_adv = mpx * MaskVSubsample<maskMode>;

  ScratchBuffer<uint16_t> eff_buf;
  if constexpr (maskMode != MASK444 || !full_opacity) eff_buf.resize(width);

#define BLEND16_LOOP_AVX2(bpp) \
  for (int y = 0; y < height; y++) { \
    const uint16_t* eff = prepare_effective_mask_for_row_avx2<maskMode, uint16_t, full_opacity>( \
      maskp, mpx, width, eff_buf.data(), opacity, half, mag); \
    blend16_masked_avx2_row<bpp>( \
      reinterpret_cast<uint16_t*>(p1), reinterpret_cast<const uint16_t*>(p2), eff, width); \
    p1 += p1_pitch; p2 += p2_pitch; maskp += mask_adv; \
  } break;

  switch (bits_per_pixel) {
  case 10: BLEND16_LOOP_AVX2(10)
  case 12: BLEND16_LOOP_AVX2(12)
  case 14: BLEND16_LOOP_AVX2(14)
  case 16: BLEND16_LOOP_AVX2(16)
  }
#undef BLEND16_LOOP_AVX2
}

// ---------------------------------------------------------------------------
// Inner loop — full_opacity known at compile time.
// Rowprep bakes opacity when !full_opacity; blend row receives pre-scaled mask.
// MASK444 + full_opacity: returns mask ptr directly (no buffer, no copy).
// MASK444 + !full_opacity: copies row with opacity scaling into buffer.
// ---------------------------------------------------------------------------
template<MaskMode maskMode, bool full_opacity>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("avx2")))
#endif
static void masked_merge_float_avx2_impl_inner(
  BYTE* p1, const BYTE* p2, const BYTE* mask,
  int p1_pitch, int p2_pitch, int mask_pitch,
  int width, int height, float opacity)
{
  const float* maskp = reinterpret_cast<const float*>(mask);
  const int mpx = mask_pitch / sizeof(float);
  const int mask_adv = mpx * MaskVSubsample<maskMode>;

  ScratchBuffer<float> eff_buf;
  if constexpr (maskMode != MASK444 || !full_opacity) eff_buf.resize(width);

  for (int y = 0; y < height; y++) {
    const float* eff = prepare_effective_mask_for_row_float_avx2<maskMode, full_opacity>(
      maskp, mpx, width, eff_buf.data(), opacity);
    blend_masked_float_avx2_row(
      reinterpret_cast<float*>(p1), reinterpret_cast<const float*>(p2), eff, width);
    p1 += p1_pitch; p2 += p2_pitch; maskp += mask_adv;
  }
}

// ---------------------------------------------------------------------------
// Outer: dispatch on full_opacity (opacity == max_pixel_value) at the call site.
// ---------------------------------------------------------------------------
template<MaskMode maskMode>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("avx2")))
#endif
static void masked_merge_avx2_impl(
  BYTE* p1, const BYTE* p2, const BYTE* mask,
  int p1_pitch, int p2_pitch, int mask_pitch,
  int width, int height, int opacity, int bits_per_pixel)
{
  const int max_val = (1 << bits_per_pixel) - 1;
  if (opacity == max_val)
    masked_merge_avx2_impl_inner<maskMode, true>(
      p1, p2, mask, p1_pitch, p2_pitch, mask_pitch, width, height, opacity, bits_per_pixel);
  else
    masked_merge_avx2_impl_inner<maskMode, false>(
      p1, p2, mask, p1_pitch, p2_pitch, mask_pitch, width, height, opacity, bits_per_pixel);
}

template<MaskMode maskMode>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("avx2")))
#endif
static void masked_merge_float_avx2_impl(
  BYTE* p1, const BYTE* p2, const BYTE* mask,
  int p1_pitch, int p2_pitch, int mask_pitch,
  int width, int height, float opacity)
{
  if (opacity >= 1.0f)
    masked_merge_float_avx2_impl_inner<maskMode, true>(
      p1, p2, mask, p1_pitch, p2_pitch, mask_pitch, width, height, opacity);
  else
    masked_merge_float_avx2_impl_inner<maskMode, false>(
      p1, p2, mask, p1_pitch, p2_pitch, mask_pitch, width, height, opacity);
}

