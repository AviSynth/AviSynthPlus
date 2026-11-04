// AviSynth+  Copyright 2026- AviSynth+ Project
// SPDX-License-Identifier: GPL-2.0-or-later
//
// SSE4.1 masked-merge implementation templates.
//
// Implementation-include — NO include guards intentionally.
// Each including TU gets its own compilation with that TU's SIMD flags.
// All definitions are static so each TU has its own private copy;
// the linker never sees them as the same symbol.
//
// Requires the following to be visible in the including TU before this file:
//   SSE4.1 intrinsics  (<intrin.h> or <x86intrin.h>, <smmintrin.h>)
//   blend_common.h     — MaskMode, MagicDiv, get_magic_div, magic_div_rt
//   <vector>

#include "masked_rowprep_sse41.h"
#include "../../../core/scratch_buffer.h"

// ---------------------------------------------------------------------------
// 8-bit row — mask already has opacity baked in by rowprep.
// 16-wide 16-bit mulhi arithmetic (fast, overflow-safe).
// ---------------------------------------------------------------------------
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static AVS_FORCEINLINE void blend8_masked_sse41_row(
  uint8_t* p1, const uint8_t* p2, const uint8_t* mask,
  int width)
{
  constexpr uint32_t half    = 127u;
  constexpr uint32_t max_val = 255u;

  const __m128i v_half  = _mm_set1_epi16((short)half);
  const __m128i v_max   = _mm_set1_epi16((short)max_val);
  const __m128i v_magic = _mm_set1_epi16((short)0x8081);

  auto magic_byte = [&](__m128i x) {
    return _mm_srli_epi16(_mm_mulhi_epu16(x, v_magic), 7);
  };

  int x = 0;
  for (; x <= width - 16; x += 16) {
    __m128i mr   = _mm_loadu_si128((__m128i*)(mask + x));
    __m128i m_lo = _mm_cvtepu8_epi16(mr);
    __m128i m_hi = _mm_cvtepu8_epi16(_mm_srli_si128(mr, 8));

    __m128i ar = _mm_loadu_si128((__m128i*)(p1 + x));
    __m128i br = _mm_loadu_si128((__m128i*)(p2 + x));

    auto blend16 = [&](__m128i a, __m128i b, __m128i m) {
      __m128i invm = _mm_sub_epi16(v_max, m);
      return magic_byte(_mm_add_epi16(
        _mm_add_epi16(_mm_mullo_epi16(a, invm), _mm_mullo_epi16(b, m)), v_half));
    };

    _mm_storeu_si128((__m128i*)(p1 + x), _mm_packus_epi16(
      blend16(_mm_cvtepu8_epi16(ar),                    _mm_cvtepu8_epi16(br),                    m_lo),
      blend16(_mm_cvtepu8_epi16(_mm_srli_si128(ar, 8)), _mm_cvtepu8_epi16(_mm_srli_si128(br, 8)), m_hi)));
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
// 32-bit arithmetic, 4 pixels per step.
// ---------------------------------------------------------------------------
template<int bits_per_pixel>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static AVS_FORCEINLINE void blend16_masked_sse41_row(
  uint16_t* p1, const uint16_t* p2, const uint16_t* mask,
  int width)
{
  constexpr MagicDiv m_div   = get_magic_div(bits_per_pixel);
  constexpr uint32_t max_val = (1u << bits_per_pixel) - 1;
  constexpr uint32_t half    = max_val / 2;

  const __m128i v_half = _mm_set1_epi32((int)half);
  const __m128i v_max  = _mm_set1_epi32((int)max_val);

  int x = 0;
  for (; x <= width - 4; x += 4) {
    __m128i m    = _mm_cvtepu16_epi32(_mm_loadl_epi64((__m128i*)(mask + x)));
    __m128i a    = _mm_cvtepu16_epi32(_mm_loadl_epi64((__m128i*)(p1 + x)));
    __m128i b    = _mm_cvtepu16_epi32(_mm_loadl_epi64((__m128i*)(p2 + x)));
    __m128i invm = _mm_sub_epi32(v_max, m);
    __m128i res  = simd_magic_div_32(
                     _mm_add_epi32(_mm_add_epi32(_mm_mullo_epi32(a, invm), _mm_mullo_epi32(b, m)), v_half),
                     m_div.div, m_div.shift);
    _mm_storel_epi64((__m128i*)(p1 + x), _mm_packus_epi32(res, res));
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
// 4 pixels per step.
// ---------------------------------------------------------------------------
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static void blend_masked_float_sse41_row(
  float* p1, const float* p2, const float* mask, int width)
{
  int x = 0;
  // const __m128 v_one = _mm_set1_ps(1.0f); in case of 1-x implementation

  for (; x <= width - 4; x += 4) {
    __m128 m = _mm_loadu_ps(mask + x);
    __m128 a = _mm_loadu_ps(p1 + x);
    __m128 b = _mm_loadu_ps(p2 + x);

    // Standard lerp: a + m * (b - a)
    // This is generally more accurate and faster (1 mul, 2 adds) than 
    // a * (1-m) + b * m (2 muls, 2 adds).
    __m128 diff = _mm_sub_ps(b, a);
    __m128 res = _mm_add_ps(a, _mm_mul_ps(m, diff));

    _mm_storeu_ps(p1 + x, res);
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
__attribute__((__target__("sse4.1")))
#endif
static AVS_FORCEINLINE void masked_merge_sse41_impl_inner(
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
      const uint8_t* eff = prepare_effective_mask_for_row_sse41<maskMode, uint8_t, full_opacity>(
        maskp, mpx, width, eff_buf.data(), opacity, half, mag);
      blend8_masked_sse41_row(
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

#define BLEND16_LOOP(bpp) \
  for (int y = 0; y < height; y++) { \
    const uint16_t* eff = prepare_effective_mask_for_row_sse41<maskMode, uint16_t, full_opacity>( \
      maskp, mpx, width, eff_buf.data(), opacity, half, mag); \
    blend16_masked_sse41_row<bpp>( \
      reinterpret_cast<uint16_t*>(p1), reinterpret_cast<const uint16_t*>(p2), eff, width); \
    p1 += p1_pitch; p2 += p2_pitch; maskp += mask_adv; \
  } break;

  switch (bits_per_pixel) {
  case 10: BLEND16_LOOP(10)
  case 12: BLEND16_LOOP(12)
  case 14: BLEND16_LOOP(14)
  case 16: BLEND16_LOOP(16)
  }
#undef BLEND16_LOOP
}

// ---------------------------------------------------------------------------
// Inner loop — full_opacity known at compile time.
// Rowprep bakes opacity when !full_opacity; blend row receives pre-scaled mask.
// MASK444 + full_opacity: returns mask ptr directly (no buffer, no copy).
// MASK444 + !full_opacity: copies row with opacity scaling into buffer.
// ---------------------------------------------------------------------------
template<MaskMode maskMode, bool full_opacity>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static void masked_merge_float_sse41_impl_inner(
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
    const float* eff = prepare_effective_mask_for_row_float_sse41<maskMode, full_opacity>(
      maskp, mpx, width, eff_buf.data(), opacity);
    blend_masked_float_sse41_row(
      reinterpret_cast<float*>(p1), reinterpret_cast<const float*>(p2), eff, width);
    p1 += p1_pitch; p2 += p2_pitch; maskp += mask_adv;
  }
}

// ---------------------------------------------------------------------------
// Outer: dispatch on full_opacity (opacity == max_pixel_value) at the call site.
// Public signature unchanged — existing callers and function-pointer tables work.
// ---------------------------------------------------------------------------
template<MaskMode maskMode>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static void masked_merge_sse41_impl(
  BYTE* p1, const BYTE* p2, const BYTE* mask,
  int p1_pitch, int p2_pitch, int mask_pitch,
  int width, int height, int opacity, int bits_per_pixel)
{
  const int max_val = (1 << bits_per_pixel) - 1;
  if (opacity == max_val)
    masked_merge_sse41_impl_inner<maskMode, true>(
      p1, p2, mask, p1_pitch, p2_pitch, mask_pitch, width, height, opacity, bits_per_pixel);
  else
    masked_merge_sse41_impl_inner<maskMode, false>(
      p1, p2, mask, p1_pitch, p2_pitch, mask_pitch, width, height, opacity, bits_per_pixel);
}

template<MaskMode maskMode>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static void masked_merge_float_sse41_impl(
  BYTE* p1, const BYTE* p2, const BYTE* mask,
  int p1_pitch, int p2_pitch, int mask_pitch,
  int width, int height, float opacity)
{
  if (opacity >= 1.0f)
    masked_merge_float_sse41_impl_inner<maskMode, true>(
      p1, p2, mask, p1_pitch, p2_pitch, mask_pitch, width, height, opacity);
  else
    masked_merge_float_sse41_impl_inner<maskMode, false>(
      p1, p2, mask, p1_pitch, p2_pitch, mask_pitch, width, height, opacity);
}
