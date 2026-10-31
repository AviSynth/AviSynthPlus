// masked_rowprep_sse41.cpp
// SSE4.1 rowprep implementations + explicit template instantiations.
// Compiled with -msse4.1 (GCC/Clang) or /arch:SSE2 (MSVC) via handle_arch_flags(SSE41).
//
// simd_magic_div_32 lives in masked_rowprep_sse41.h (inline, needed by merge impl).
// All fill_mask*_sse41 helpers are static (internal to this TU).

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <smmintrin.h>
#endif

#include "masked_rowprep_sse41.h"   // own declarations + simd_magic_div_32 inline

#include <vector>
#include <cstdint>

// ---------------------------------------------------------------------------
// Internal helper — deinterleave 16 uint8 → two int16 vectors (even/odd bytes).
// Only used within this TU by the fill_mask* functions below.
// ---------------------------------------------------------------------------
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static AVS_FORCEINLINE void deinterleave_u8_epi16(
  __m128i v, __m128i& even_out, __m128i& odd_out)
{
  even_out = _mm_and_si128(v, _mm_set1_epi16(0x00FF));
  odd_out  = _mm_srli_epi16(v, 8);
}

// ---------------------------------------------------------------------------
// MASK422 — horizontal 2-tap average, no inter-row state.
//   avg[x] = (src[x*2] + src[x*2+1] + 1) >> 1
//   dst[x] = full_opacity ? avg : (avg * opacity_i + half) / max
// ---------------------------------------------------------------------------
template<typename pixel_t, bool full_opacity>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static void fill_mask422_sse41(
  pixel_t* dst, const pixel_t* src, int width,
  int opacity_i, int half, MagicDiv magic)
{
  int x = 0;
  if constexpr (sizeof(pixel_t) == 1) {
    // 8-bit: 16 luma bytes → 8 chroma bytes per iteration
    [[maybe_unused]] const __m128i v_opacity = _mm_set1_epi16((short)opacity_i);
    [[maybe_unused]] const __m128i v_half16  = _mm_set1_epi16((short)half);
    [[maybe_unused]] const __m128i v_mdiv    = _mm_set1_epi16((short)magic.div);
    for (; x <= width - 8; x += 8) {
      __m128i v = _mm_loadu_si128((const __m128i*)(src + x * 2));
      __m128i even, odd;
      deinterleave_u8_epi16(v, even, odd);
      __m128i avg = _mm_srli_epi16(_mm_add_epi16(_mm_add_epi16(even, odd), _mm_set1_epi16(1)), 1);
      if constexpr (!full_opacity) {
        // (avg * opacity + half) / max — all fit in uint16
        __m128i scaled = _mm_add_epi16(_mm_mullo_epi16(avg, v_opacity), v_half16);
        avg = _mm_srli_epi16(_mm_mulhi_epu16(scaled, v_mdiv), magic.shift);
      }
      _mm_storel_epi64((__m128i*)(dst + x), _mm_packus_epi16(avg, avg));
    }
  } else {
    // 16-bit: 8 uint16 luma -> 4 uint16 chroma per iteration.
    // Use signed pivot to utilize madd_epi16 safely
    const __m128i v_ones = _mm_set1_epi16(1);
    const __m128i v_pivot16 = _mm_set1_epi16(-32768);
    // 65536 corrects the double-bias subtraction, +1 handles the formula's rounding
    const __m128i v_correct32 = _mm_set1_epi32(65536 + 1);

    [[maybe_unused]] const __m128i v_opacity32 = _mm_set1_epi32(opacity_i);
    [[maybe_unused]] const __m128i v_half32 = _mm_set1_epi32(half);

    for (; x <= width - 4; x += 4) {
      __m128i v = _mm_loadu_si128((const __m128i*)(src + x * 2));

      // unsigned 0..65535 -> signed -32768..32767
      __m128i v_signed = _mm_add_epi16(v, v_pivot16);

      // Pairwise add: (a - 32768) + (b - 32768) = a + b - 65536
      __m128i sum32 = _mm_madd_epi16(v_signed, v_ones);

      // Add 65536 back + 1 for rounding, then shift
      __m128i avg32 = _mm_srli_epi32(_mm_add_epi32(sum32, v_correct32), 1);

      if constexpr (!full_opacity)
        avg32 = simd_magic_div_32(
          _mm_add_epi32(_mm_mullo_epi32(avg32, v_opacity32), v_half32),
          magic.div, magic.shift);
      _mm_storel_epi64((__m128i*)(dst + x), _mm_packus_epi32(avg32, avg32));
    }
  }
  for (; x < width; x++) {
    const int avg = (src[x * 2] + src[x * 2 + 1] + 1) >> 1;
    dst[x] = full_opacity ? (pixel_t)avg
           : (pixel_t)magic_div_rt<pixel_t>((uint32_t)avg * (uint32_t)opacity_i + (uint32_t)half, magic);
  }
}

// ---------------------------------------------------------------------------
// MASK422_MPEG2 — horizontal 3-tap triangle filter with sliding window.
//   avg[x] = (left + 2*src[x*2] + src[x*2+1] + 2) >> 2
//   dst[x] = full_opacity ? avg : (avg * opacity_i + half) / max
// ---------------------------------------------------------------------------
template<typename pixel_t, bool full_opacity>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static void fill_mask422_mpeg2_sse41(
  pixel_t* dst, const pixel_t* src, int width,
  int opacity_i, int half, MagicDiv magic)
{
  int x = 0;
  if constexpr (sizeof(pixel_t) == 1) {
    [[maybe_unused]] const __m128i v_opacity = _mm_set1_epi16((short)opacity_i);
    [[maybe_unused]] const __m128i v_half16  = _mm_set1_epi16((short)half);
    [[maybe_unused]] const __m128i v_mdiv    = _mm_set1_epi16((short)magic.div);
    __m128i prev_carry = _mm_insert_epi16(_mm_setzero_si128(), src[0], 7);
    for (; x <= width - 8; x += 8) {
      __m128i v = _mm_loadu_si128((const __m128i*)(src + x * 2));
      __m128i even, odd;
      deinterleave_u8_epi16(v, even, odd);
      __m128i left = _mm_alignr_epi8(odd, prev_carry, 14);
      __m128i res  = _mm_srli_epi16(
        _mm_add_epi16(
          _mm_add_epi16(_mm_add_epi16(left, _mm_slli_epi16(even, 1)), odd),
          _mm_set1_epi16(2)), 2);
      if constexpr (!full_opacity) {
        __m128i scaled = _mm_add_epi16(_mm_mullo_epi16(res, v_opacity), v_half16);
        res = _mm_srli_epi16(_mm_mulhi_epu16(scaled, v_mdiv), magic.shift);
      }
      _mm_storel_epi64((__m128i*)(dst + x), _mm_packus_epi16(res, res));
      prev_carry = _mm_insert_epi16(_mm_setzero_si128(), _mm_extract_epi16(odd, 7), 7);
    }
    int right_val = _mm_extract_epi16(prev_carry, 7);
    for (; x < width; x++) {
      const int left = right_val;
      const int mid  = src[x * 2];
      right_val      = src[x * 2 + 1];
      const int avg  = (left + 2 * mid + right_val + 2) >> 2;
      dst[x] = full_opacity ? (pixel_t)avg
             : (pixel_t)magic_div_rt<pixel_t>((uint32_t)avg * (uint32_t)opacity_i + (uint32_t)half, magic);
    }
  } else {
    [[maybe_unused]] const __m128i v_opacity32 = _mm_set1_epi32(opacity_i);
    [[maybe_unused]] const __m128i v_half32    = _mm_set1_epi32(half);
    __m128i prev_carry = _mm_insert_epi32(_mm_setzero_si128(), src[0], 3);
    for (; x <= width - 4; x += 4) {
      __m128i v = _mm_loadu_si128((const __m128i*)(src + x * 2));
      __m128i lo32  = _mm_cvtepu16_epi32(v);
      __m128i hi32  = _mm_cvtepu16_epi32(_mm_srli_si128(v, 8));
      __m128i even32 = _mm_unpacklo_epi64(
        _mm_shuffle_epi32(lo32, _MM_SHUFFLE(2, 0, 2, 0)),
        _mm_shuffle_epi32(hi32, _MM_SHUFFLE(2, 0, 2, 0)));
      __m128i odd32 = _mm_unpacklo_epi64(
        _mm_shuffle_epi32(lo32, _MM_SHUFFLE(3, 1, 3, 1)),
        _mm_shuffle_epi32(hi32, _MM_SHUFFLE(3, 1, 3, 1)));
      __m128i left = _mm_alignr_epi8(odd32, prev_carry, 12);
      __m128i res  = _mm_srli_epi32(
        _mm_add_epi32(
          _mm_add_epi32(_mm_add_epi32(left, _mm_slli_epi32(even32, 1)), odd32),
          _mm_set1_epi32(2)), 2);
      if constexpr (!full_opacity)
        res = simd_magic_div_32(
          _mm_add_epi32(_mm_mullo_epi32(res, v_opacity32), v_half32),
          magic.div, magic.shift);
      _mm_storel_epi64((__m128i*)(dst + x), _mm_packus_epi32(res, res));
      prev_carry = _mm_insert_epi32(_mm_setzero_si128(), _mm_extract_epi32(odd32, 3), 3);
    }
    int right_val = _mm_extract_epi32(prev_carry, 3);
    for (; x < width; x++) {
      const int left = right_val;
      const int mid  = src[x * 2];
      right_val      = src[x * 2 + 1];
      const int avg  = (left + 2 * mid + right_val + 2) >> 2;
      dst[x] = full_opacity ? (pixel_t)avg
             : (pixel_t)magic_div_rt<pixel_t>((uint32_t)avg * (uint32_t)opacity_i + (uint32_t)half, magic);
    }
  }
}

// ---------------------------------------------------------------------------
// MASK420 — 2×2 box average (MPEG-1 placement). No inter-row state.
//   avg[x] = (row0[x*2]+row0[x*2+1]+row1[x*2]+row1[x*2+1]+2) >> 2
//   dst[x] = full_opacity ? avg : (avg * opacity_i + half) / max
// ---------------------------------------------------------------------------
template<typename pixel_t, bool full_opacity>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static void fill_mask420_sse41(
  pixel_t* dst, const pixel_t* row0, int mask_pitch, int width,
  int opacity_i, int half, MagicDiv magic)
{
  const pixel_t* row1 = row0 + mask_pitch;
  int x = 0;
  if constexpr (sizeof(pixel_t) == 1) {
    [[maybe_unused]] const __m128i v_opacity = _mm_set1_epi16((short)opacity_i);
    [[maybe_unused]] const __m128i v_half16  = _mm_set1_epi16((short)half);
    [[maybe_unused]] const __m128i v_mdiv    = _mm_set1_epi16((short)magic.div);
    for (; x <= width - 8; x += 8) {
      __m128i r0 = _mm_loadu_si128((const __m128i*)(row0 + x * 2));
      __m128i r1 = _mm_loadu_si128((const __m128i*)(row1 + x * 2));
      __m128i e0, o0, e1, o1;
      deinterleave_u8_epi16(r0, e0, o0);
      deinterleave_u8_epi16(r1, e1, o1);
      __m128i avg = _mm_srli_epi16(
        _mm_add_epi16(_mm_add_epi16(_mm_add_epi16(e0, o0), _mm_add_epi16(e1, o1)), _mm_set1_epi16(2)), 2);
      if constexpr (!full_opacity) {
        __m128i scaled = _mm_add_epi16(_mm_mullo_epi16(avg, v_opacity), v_half16);
        avg = _mm_srli_epi16(_mm_mulhi_epu16(scaled, v_mdiv), magic.shift);
      }
      _mm_storel_epi64((__m128i*)(dst + x), _mm_packus_epi16(avg, avg));
    }
  } else {
    // 16-bit: unsigned widening to avoid signed overflow in madd_epi16
    [[maybe_unused]] const __m128i v_opacity32 = _mm_set1_epi32(opacity_i);
    [[maybe_unused]] const __m128i v_half32    = _mm_set1_epi32(half);
    for (; x <= width - 4; x += 4) {
      __m128i v0     = _mm_loadu_si128((const __m128i*)(row0 + x * 2)); // 8 uint16
      __m128i v1     = _mm_loadu_si128((const __m128i*)(row1 + x * 2));
      __m128i lo0    = _mm_cvtepu16_epi32(v0);
      __m128i hi0    = _mm_cvtepu16_epi32(_mm_srli_si128(v0, 8));
      __m128i lo1    = _mm_cvtepu16_epi32(v1);
      __m128i hi1    = _mm_cvtepu16_epi32(_mm_srli_si128(v1, 8));
      __m128i sum_lo = _mm_add_epi32(lo0, lo1);
      __m128i sum_hi = _mm_add_epi32(hi0, hi1);
      __m128i even32 = _mm_unpacklo_epi64(
        _mm_shuffle_epi32(sum_lo, _MM_SHUFFLE(2, 0, 2, 0)),
        _mm_shuffle_epi32(sum_hi, _MM_SHUFFLE(2, 0, 2, 0)));
      __m128i odd32  = _mm_unpacklo_epi64(
        _mm_shuffle_epi32(sum_lo, _MM_SHUFFLE(3, 1, 3, 1)),
        _mm_shuffle_epi32(sum_hi, _MM_SHUFFLE(3, 1, 3, 1)));
      __m128i avg32  = _mm_srli_epi32(
        _mm_add_epi32(_mm_add_epi32(even32, odd32), _mm_set1_epi32(2)), 2);
      if constexpr (!full_opacity)
        avg32 = simd_magic_div_32(
          _mm_add_epi32(_mm_mullo_epi32(avg32, v_opacity32), v_half32),
          magic.div, magic.shift);
      _mm_storel_epi64((__m128i*)(dst + x), _mm_packus_epi32(avg32, avg32));
    }
  }
  for (; x < width; x++) {
    const int avg = ((int)row0[x*2] + row0[x*2+1] + row1[x*2] + row1[x*2+1] + 2) >> 2;
    dst[x] = full_opacity ? (pixel_t)avg
           : (pixel_t)magic_div_rt<pixel_t>((uint32_t)avg * (uint32_t)opacity_i + (uint32_t)half, magic);
  }
}

// ---------------------------------------------------------------------------
// MASK420_MPEG2 — 2-row vertical sum + horizontal 3-tap triangle filter.
//   avg[x] = (left + 2*P[x*2] + P[x*2+1] + 4) >> 3  where P[k] = row0[k]+row1[k]
//   dst[x] = full_opacity ? avg : (avg * opacity_i + half) / max
// ---------------------------------------------------------------------------
template<typename pixel_t, bool full_opacity>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static void fill_mask420_mpeg2_sse41(
  pixel_t* dst, const pixel_t* row0, int mask_pitch, int width,
  int opacity_i, int half, MagicDiv magic)
{
  const pixel_t* row1 = row0 + mask_pitch;
  int x = 0;
  if constexpr (sizeof(pixel_t) == 1) {
    [[maybe_unused]] const __m128i v_opacity = _mm_set1_epi16((short)opacity_i);
    [[maybe_unused]] const __m128i v_half16  = _mm_set1_epi16((short)half);
    [[maybe_unused]] const __m128i v_mdiv    = _mm_set1_epi16((short)magic.div);
    const int p0 = (int)row0[0] + row1[0];
    __m128i prev_carry = _mm_insert_epi16(_mm_setzero_si128(), p0, 7);

    for (; x <= width - 8; x += 8) {
      __m128i r0 = _mm_loadu_si128((const __m128i*)(row0 + x * 2));
      __m128i r1 = _mm_loadu_si128((const __m128i*)(row1 + x * 2));
      __m128i plo = _mm_add_epi16(_mm_unpacklo_epi8(r0, _mm_setzero_si128()),
                                   _mm_unpacklo_epi8(r1, _mm_setzero_si128()));
      __m128i phi = _mm_add_epi16(_mm_unpackhi_epi8(r0, _mm_setzero_si128()),
                                   _mm_unpackhi_epi8(r1, _mm_setzero_si128()));

      const __m128i shuf_even = _mm_setr_epi8(0,1, 4,5, 8,9, 12,13, -1,-1,-1,-1,-1,-1,-1,-1);
      const __m128i shuf_odd  = _mm_setr_epi8(2,3, 6,7, 10,11, 14,15, -1,-1,-1,-1,-1,-1,-1,-1);
      __m128i pe = _mm_unpacklo_epi64(_mm_shuffle_epi8(plo, shuf_even),
                                      _mm_shuffle_epi8(phi, shuf_even));
      __m128i po = _mm_unpacklo_epi64(_mm_shuffle_epi8(plo, shuf_odd),
                                      _mm_shuffle_epi8(phi, shuf_odd));

      __m128i left = _mm_alignr_epi8(po, prev_carry, 14);
      __m128i res  = _mm_srli_epi16(
        _mm_add_epi16(
          _mm_add_epi16(_mm_add_epi16(left, _mm_slli_epi16(pe, 1)), po),
          _mm_set1_epi16(4)), 3);
      if constexpr (!full_opacity) {
        __m128i scaled = _mm_add_epi16(_mm_mullo_epi16(res, v_opacity), v_half16);
        res = _mm_srli_epi16(_mm_mulhi_epu16(scaled, v_mdiv), magic.shift);
      }
      _mm_storel_epi64((__m128i*)(dst + x), _mm_packus_epi16(res, res));
      prev_carry = _mm_insert_epi16(_mm_setzero_si128(), _mm_extract_epi16(po, 7), 7);
    }
    int right_val = _mm_extract_epi16(prev_carry, 7);
    for (; x < width; x++) {
      const int left = right_val;
      const int mid  = (int)row0[x*2]   + row1[x*2];
      right_val      = (int)row0[x*2+1] + row1[x*2+1];
      const int avg  = (left + 2 * mid + right_val + 4) >> 3;
      dst[x] = full_opacity ? (pixel_t)avg
             : (pixel_t)magic_div_rt<pixel_t>((uint32_t)avg * (uint32_t)opacity_i + (uint32_t)half, magic);
    }
  } else {
    [[maybe_unused]] const __m128i v_opacity32 = _mm_set1_epi32(opacity_i);
    [[maybe_unused]] const __m128i v_half32    = _mm_set1_epi32(half);
    const int p0 = (int)row0[0] + row1[0];
    __m128i prev_carry = _mm_insert_epi32(_mm_setzero_si128(), p0, 3);

    for (; x <= width - 4; x += 4) {
      __m128i v0  = _mm_loadu_si128((const __m128i*)(row0 + x * 2));
      __m128i v1  = _mm_loadu_si128((const __m128i*)(row1 + x * 2));
      __m128i plo = _mm_add_epi32(_mm_cvtepu16_epi32(v0), _mm_cvtepu16_epi32(v1));
      __m128i phi = _mm_add_epi32(_mm_cvtepu16_epi32(_mm_srli_si128(v0, 8)),
                                   _mm_cvtepu16_epi32(_mm_srli_si128(v1, 8)));
      __m128i pe = _mm_unpacklo_epi64(
        _mm_shuffle_epi32(plo, _MM_SHUFFLE(2, 0, 2, 0)),
        _mm_shuffle_epi32(phi, _MM_SHUFFLE(2, 0, 2, 0)));
      __m128i po = _mm_unpacklo_epi64(
        _mm_shuffle_epi32(plo, _MM_SHUFFLE(3, 1, 3, 1)),
        _mm_shuffle_epi32(phi, _MM_SHUFFLE(3, 1, 3, 1)));

      __m128i left = _mm_alignr_epi8(po, prev_carry, 12);
      __m128i res  = _mm_srli_epi32(
        _mm_add_epi32(
          _mm_add_epi32(_mm_add_epi32(left, _mm_slli_epi32(pe, 1)), po),
          _mm_set1_epi32(4)), 3);
      if constexpr (!full_opacity)
        res = simd_magic_div_32(
          _mm_add_epi32(_mm_mullo_epi32(res, v_opacity32), v_half32),
          magic.div, magic.shift);
      _mm_storel_epi64((__m128i*)(dst + x), _mm_packus_epi32(res, res));
      prev_carry = _mm_insert_epi32(_mm_setzero_si128(), _mm_extract_epi32(po, 3), 3);
    }
    int right_val = _mm_extract_epi32(prev_carry, 3);
    for (; x < width; x++) {
      const int left = right_val;
      const int mid  = (int)row0[x*2]   + row1[x*2];
      right_val      = (int)row0[x*2+1] + row1[x*2+1];
      const int avg  = (left + 2 * mid + right_val + 4) >> 3;
      dst[x] = full_opacity ? (pixel_t)avg
             : (pixel_t)magic_div_rt<pixel_t>((uint32_t)avg * (uint32_t)opacity_i + (uint32_t)half, magic);
    }
  }
}

// ---------------------------------------------------------------------------
// MASK422_TOPLEFT — left co-sited point sample (no averaging).
//   dst[x] = src[x*2]
//   dst[x] = full_opacity ? dst[x] : (dst[x] * opacity_i + half) / max
// ---------------------------------------------------------------------------
template<typename pixel_t, bool full_opacity>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static void fill_mask422_topleft_sse41(
  pixel_t* dst, const pixel_t* src, int width,
  int opacity_i, int half, MagicDiv magic)
{
  int x = 0;
  if constexpr (sizeof(pixel_t) == 1) {
    [[maybe_unused]] const __m128i v_opacity = _mm_set1_epi16((short)opacity_i);
    [[maybe_unused]] const __m128i v_half16  = _mm_set1_epi16((short)half);
    [[maybe_unused]] const __m128i v_mdiv    = _mm_set1_epi16((short)magic.div);
    for (; x <= width - 8; x += 8) {
      __m128i v    = _mm_loadu_si128((const __m128i*)(src + x * 2));
      __m128i even = _mm_and_si128(v, _mm_set1_epi16(0x00FF)); // left (even) bytes
      if constexpr (!full_opacity) {
        __m128i scaled = _mm_add_epi16(_mm_mullo_epi16(even, v_opacity), v_half16);
        even = _mm_srli_epi16(_mm_mulhi_epu16(scaled, v_mdiv), magic.shift);
      }
      _mm_storel_epi64((__m128i*)(dst + x), _mm_packus_epi16(even, even));
    }
  } else {
    // 16-bit: grab even-indexed uint16 elements (src[x*2], src[x*2+2], ...)
    // Load 8 uint16: [a,b,c,d,e,f,g,h] → want [a,c,e,g]
    const __m128i shuf = _mm_setr_epi8(0,1, 4,5, 8,9, 12,13, -1,-1,-1,-1,-1,-1,-1,-1);
    [[maybe_unused]] const __m128i v_opacity32 = _mm_set1_epi32(opacity_i);
    [[maybe_unused]] const __m128i v_half32    = _mm_set1_epi32(half);
    for (; x <= width - 4; x += 4) {
      __m128i v    = _mm_loadu_si128((const __m128i*)(src + x * 2));
      __m128i even = _mm_shuffle_epi8(v, shuf); // [a,c,e,g,0,...] as 4 uint16
      if constexpr (!full_opacity) {
        __m128i even32 = _mm_cvtepu16_epi32(even);
        even32 = simd_magic_div_32(
          _mm_add_epi32(_mm_mullo_epi32(even32, v_opacity32), v_half32),
          magic.div, magic.shift);
        even = _mm_packus_epi32(even32, even32);
      }
      _mm_storel_epi64((__m128i*)(dst + x), even);
    }
  }
  for (; x < width; x++) {
    const int val = src[x * 2];
    dst[x] = full_opacity ? (pixel_t)val
           : (pixel_t)magic_div_rt<pixel_t>((uint32_t)val * (uint32_t)opacity_i + (uint32_t)half, magic);
  }
}

// ---------------------------------------------------------------------------
// MASK420_TOPLEFT — top-left co-sited point sample (top row only, no averaging).
//   dst[x] = row0[x*2]
// ---------------------------------------------------------------------------
template<typename pixel_t, bool full_opacity>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static void fill_mask420_topleft_sse41(
  pixel_t* dst, const pixel_t* row0, int /*mask_pitch*/, int width,
  int opacity_i, int half, MagicDiv magic)
{
  // Identical to fill_mask422_topleft_sse41: top row only, left co-sited.
  fill_mask422_topleft_sse41<pixel_t, full_opacity>(dst, row0, width, opacity_i, half, magic);
}

// ---------------------------------------------------------------------------
// MASK411 — horizontal 4-tap box average.
//   avg[x] = (src[x*4]+src[x*4+1]+src[x*4+2]+src[x*4+3]+2) >> 2
//   dst[x] = full_opacity ? avg : (avg * opacity_i + half) / max
// ---------------------------------------------------------------------------
template<typename pixel_t, bool full_opacity>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static void fill_mask411_sse41(
  pixel_t* dst, const pixel_t* src, int width,
  int opacity_i, int half, MagicDiv magic)
{
  int x = 0;
  if constexpr (sizeof(pixel_t) == 1) {
    const __m128i zero = _mm_setzero_si128();
    [[maybe_unused]] const __m128i v_opacity = _mm_set1_epi16((short)opacity_i);
    [[maybe_unused]] const __m128i v_half16  = _mm_set1_epi16((short)half);
    [[maybe_unused]] const __m128i v_mdiv    = _mm_set1_epi16((short)magic.div);
    for (; x <= width - 8; x += 8) {
      __m128i v0 = _mm_loadu_si128((const __m128i*)(src + x * 4));
      __m128i v1 = _mm_loadu_si128((const __m128i*)(src + x * 4 + 16));
      __m128i p0 = _mm_hadd_epi16(_mm_unpacklo_epi8(v0, zero), _mm_unpackhi_epi8(v0, zero));
      __m128i p1 = _mm_hadd_epi16(_mm_unpacklo_epi8(v1, zero), _mm_unpackhi_epi8(v1, zero));
      __m128i avg = _mm_srli_epi16(_mm_add_epi16(_mm_hadd_epi16(p0, p1), _mm_set1_epi16(2)), 2);
      if constexpr (!full_opacity) {
        __m128i scaled = _mm_add_epi16(_mm_mullo_epi16(avg, v_opacity), v_half16);
        avg = _mm_srli_epi16(_mm_mulhi_epu16(scaled, v_mdiv), magic.shift);
      }
      _mm_storel_epi64((__m128i*)(dst + x), _mm_packus_epi16(avg, avg));
    }
  } else {
    // 16-bit: unsigned widening to avoid signed overflow in madd_epi16
    [[maybe_unused]] const __m128i v_opacity32 = _mm_set1_epi32(opacity_i);
    [[maybe_unused]] const __m128i v_half32    = _mm_set1_epi32(half);
    for (; x <= width - 4; x += 4) {
      __m128i v0   = _mm_loadu_si128((const __m128i*)(src + x * 4));     // s0..s7
      __m128i v1   = _mm_loadu_si128((const __m128i*)(src + x * 4 + 8)); // s8..s15
      __m128i e0   = _mm_cvtepu16_epi32(v0);
      __m128i e0h  = _mm_cvtepu16_epi32(_mm_srli_si128(v0, 8));
      __m128i e1   = _mm_cvtepu16_epi32(v1);
      __m128i e1h  = _mm_cvtepu16_epi32(_mm_srli_si128(v1, 8));
      __m128i p01  = _mm_hadd_epi32(e0, e0h);  // [s0+s1, s2+s3, s4+s5, s6+s7]
      __m128i p23  = _mm_hadd_epi32(e1, e1h);  // [s8+s9, s10+s11, s12+s13, s14+s15]
      __m128i avg32 = _mm_srli_epi32(
        _mm_add_epi32(_mm_hadd_epi32(p01, p23), _mm_set1_epi32(2)), 2);
      if constexpr (!full_opacity)
        avg32 = simd_magic_div_32(
          _mm_add_epi32(_mm_mullo_epi32(avg32, v_opacity32), v_half32),
          magic.div, magic.shift);
      _mm_storel_epi64((__m128i*)(dst + x), _mm_packus_epi32(avg32, avg32));
    }
  }
  for (; x < width; x++) {
    const int avg = ((int)src[x*4] + src[x*4+1] + src[x*4+2] + src[x*4+3] + 2) >> 2;
    dst[x] = full_opacity ? (pixel_t)avg
           : (pixel_t)magic_div_rt<pixel_t>((uint32_t)avg * (uint32_t)opacity_i + (uint32_t)half, magic);
  }
}

// ---------------------------
// Start of float mask helpers
// ---------------------------

// MASK422 — horizontal 2-tap box average.
//   avg[x] = (src[x*2] + src[x*2+1]) * 0.5f
// ---------------------------------------------------------------------------
// MASK422 — horizontal 2-tap box average.
// float: 4 output pixels / iteration (8 input floats loaded)
// ---------------------------------------------------------------------------
template<bool full_opacity>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static void fill_mask422_float_sse41(
  float* dst, const float* src, int width, float opacity)
{
  int x = 0;
  const __m128 v_opacity = _mm_set1_ps(opacity);
  const __m128 v_05 = _mm_set1_ps(0.5f);

  for (; x <= width - 4; x += 4) {
    // 1. Load 8 floats into two 128-bit registers
    __m128 r0 = _mm_loadu_ps(src + x * 2);     // [e0, o0, e1, o1]
    __m128 r1 = _mm_loadu_ps(src + x * 2 + 4); // [e2, o2, e3, o3]

    // 2. De-interleave Even and Odd samples
    // _mm_shuffle_ps(a, b, mask) takes 2 from a and 2 from b
    // Mask (2, 0, 2, 0) -> a[0], a[2], b[0], b[2]
    __m128 even = _mm_shuffle_ps(r0, r1, _MM_SHUFFLE(2, 0, 2, 0)); // [e0, e1, e2, e3]
    __m128 odd = _mm_shuffle_ps(r0, r1, _MM_SHUFFLE(3, 1, 3, 1)); // [o0, o1, o2, o3]

    // 3. Average: (even + odd) * 0.5
    __m128 avg = _mm_mul_ps(_mm_add_ps(even, odd), v_05);

    if constexpr (!full_opacity) {
      avg = _mm_mul_ps(avg, v_opacity);
    }

    _mm_storeu_ps(dst + x, avg);
  }

  for (; x < width; x++) {
    const float avg = (src[x * 2] + src[x * 2 + 1]) * 0.5f;
    dst[x] = full_opacity ? avg : avg * opacity;
  }
}

// ---------------------------------------------------------------------------
// MASK422_MPEG2 — horizontal 3-tap triangle filter with sliding window carry.
//   avg[x] = (left + 2*src[x*2] + src[x*2+1]) * 0.25f
// float: 4 output pixels / iteration (8 input floats loaded)
// ---------------------------------------------------------------------------
template<bool full_opacity>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static void fill_mask422_mpeg2_float_sse41(
  float* dst, const float* src, int width, float opacity)
{
  int x = 0;
  float right_val = src[0];

  const __m128 v_opacity = _mm_set1_ps(opacity);
  const __m128 v_025 = _mm_set1_ps(0.25f);

  // v_prev_odd holds the odd samples from the previous iteration
  __m128 v_prev_odd = _mm_set1_ps(right_val);

  for (; x <= width - 4; x += 4) {
    // 1. Load 8 floats
    __m128 r0 = _mm_loadu_ps(src + x * 2);
    __m128 r1 = _mm_loadu_ps(src + x * 2 + 4);

    // 2. De-interleave
    __m128 even = _mm_shuffle_ps(r0, r1, _MM_SHUFFLE(2, 0, 2, 0));
    __m128 odd = _mm_shuffle_ps(r0, r1, _MM_SHUFFLE(3, 1, 3, 1));

    // 3. Sliding Window Carry
    // We want left = [p_o3, o0, o1, o2]
    // alignr(curr, prev, 12 bytes) pulls 1 float from 'prev' and 3 from 'curr'
    __m128 left = _mm_castsi128_ps(_mm_alignr_epi8(
      _mm_castps_si128(odd),
      _mm_castps_si128(v_prev_odd), 12));

    // 4. Filter: avg = (left + 2*even + odd) * 0.25
    __m128 avg = _mm_add_ps(left, _mm_add_ps(_mm_add_ps(even, even), odd));
    avg = _mm_mul_ps(avg, v_025);

    if constexpr (!full_opacity) {
      avg = _mm_mul_ps(avg, v_opacity);
    }

    _mm_storeu_ps(dst + x, avg);

    // 5. Update Carry: Simply store the current odd register
    v_prev_odd = odd;
  }

  // Extraction: Get index 3 (the last odd sample) for the scalar tail
  right_val = _mm_cvtss_f32(_mm_shuffle_ps(v_prev_odd, v_prev_odd, _MM_SHUFFLE(3, 3, 3, 3)));

  for (; x < width; x++) {
    const float left = right_val;
    const float mid = src[x * 2];
    right_val = src[x * 2 + 1];

    const float avg = (left + 2.0f * mid + right_val) * 0.25f;
    dst[x] = full_opacity ? avg : avg * opacity;
  }
}

// ---------------------------------------------------------------------------
// MASK420 — 2x2 box average (MPEG-1 placement).
// SSE4.1 float: 4 output pixels / iteration (8 input floats per row)
// ---------------------------------------------------------------------------
template<bool full_opacity>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static void fill_mask420_float_sse41(
  float* dst, const float* row0, int mask_pitch, int width, float opacity)
{
  const float* row1 = row0 + mask_pitch;
  int x = 0;

  const __m128 v_opacity = _mm_set1_ps(opacity);
  const __m128 v_025 = _mm_set1_ps(0.25f);

  for (; x <= width - 4; x += 4) {
    // 1. Load 8 floats from each row (2x 128-bit loads)
    __m128 r0_0 = _mm_loadu_ps(row0 + x * 2);
    __m128 r0_1 = _mm_loadu_ps(row0 + x * 2 + 4);
    __m128 r1_0 = _mm_loadu_ps(row1 + x * 2);
    __m128 r1_1 = _mm_loadu_ps(row1 + x * 2 + 4);

    // 2. Vertical Sum
    __m128 s0 = _mm_add_ps(r0_0, r1_0); // [e0, o0, e1, o1]
    __m128 s1 = _mm_add_ps(r0_1, r1_1); // [e2, o2, e3, o3]

    // 3. Horizontal Sum via De-interleave
    // Shuffle picks [e0, e1] from s0 and [e2, e3] from s1
    __m128 even = _mm_shuffle_ps(s0, s1, _MM_SHUFFLE(2, 0, 2, 0));
    __m128 odd = _mm_shuffle_ps(s0, s1, _MM_SHUFFLE(3, 1, 3, 1));

    // 4. Final Average: (even + odd) * 0.25
    __m128 avg = _mm_mul_ps(_mm_add_ps(even, odd), v_025);

    if constexpr (!full_opacity) {
      avg = _mm_mul_ps(avg, v_opacity);
    }

    _mm_storeu_ps(dst + x, avg);
  }

  // Scalar Tail
  for (; x < width; x++) {
    const float sum = row0[x * 2] + row0[x * 2 + 1] + row1[x * 2] + row1[x * 2 + 1];
    const float avg = sum * 0.25f;
    dst[x] = full_opacity ? avg : avg * opacity;
  }
}

// ---------------------------------------------------------------------------
// MASK420_MPEG2 — horizontal 3-tap triangle filter with vertical 2-row sum.
// avg[x] = (po[x-1] + 2*pe[x] + po[x]) * 0.125f
// SSE4.1 float: 4 output pixels / iteration (8 input floats per row)
// ---------------------------------------------------------------------------
template<bool full_opacity>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static void fill_mask420_mpeg2_float_sse41(
  float* dst, const float* row0, int mask_pitch, int width, float opacity)
{
  const float* row1 = row0 + mask_pitch;
  int x = 0;

  float right_val = row0[0] + row1[0];
  __m128 v_prev_odd = _mm_set1_ps(right_val);
  const __m128 v_opacity = _mm_set1_ps(opacity);
  const __m128 v_0125 = _mm_set1_ps(0.125f);

  for (; x <= width - 4; x += 4) {
    // 1. Load 8 floats from each row
    __m128 r0_0 = _mm_loadu_ps(row0 + x * 2);
    __m128 r0_1 = _mm_loadu_ps(row0 + x * 2 + 4);
    __m128 r1_0 = _mm_loadu_ps(row1 + x * 2);
    __m128 r1_1 = _mm_loadu_ps(row1 + x * 2 + 4);

    // 2. Vertical Sum (pe and po interleaved)
    __m128 s0 = _mm_add_ps(r0_0, r1_0);
    __m128 s1 = _mm_add_ps(r0_1, r1_1);

    // 3. De-interleave Even and Odd
    __m128 even = _mm_shuffle_ps(s0, s1, _MM_SHUFFLE(2, 0, 2, 0));
    __m128 odd = _mm_shuffle_ps(s0, s1, _MM_SHUFFLE(3, 1, 3, 1));

    // 4. Construct 'left' vector: [prev_o3, o0, o1, o2]
    // alignr 12 bytes = shift right by 3 floats (leaving 1 from v_prev_odd)
    __m128 left = _mm_castsi128_ps(_mm_alignr_epi8(
      _mm_castps_si128(odd),
      _mm_castps_si128(v_prev_odd), 12));

    // 5. Triangle Filter: (left + 2*even + odd) * 0.125
    __m128 avg = _mm_mul_ps(
      _mm_add_ps(_mm_add_ps(left, _mm_add_ps(even, even)), odd),
      v_0125
    );

    if constexpr (!full_opacity) {
      avg = _mm_mul_ps(avg, v_opacity);
    }

    _mm_storeu_ps(dst + x, avg);

    // Update carry: store the current odd register
    v_prev_odd = odd;
  }

  // Bridge to scalar tail: Extract index 3 (last odd sample)
  right_val = _mm_cvtss_f32(_mm_shuffle_ps(v_prev_odd, v_prev_odd, _MM_SHUFFLE(3, 3, 3, 3)));

  for (; x < width; x++) {
    const float left = right_val;
    const float mid = row0[x * 2] + row1[x * 2];
    right_val = row0[x * 2 + 1] + row1[x * 2 + 1];
    const float avg = (left + 2.0f * mid + right_val) * 0.125f;
    dst[x] = full_opacity ? avg : avg * opacity;
  }
}

// ---------------------------------------------------------------------------
// MASK422_TOPLEFT — left co-sited point sample (no averaging).
//   dst[x] = src[x*2]
// SSE4.1 float: 4 output pixels / iteration (8 input floats loaded)
// ---------------------------------------------------------------------------
template<bool full_opacity>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static void fill_mask422_topleft_float_sse41(
  float* dst, const float* src, int width, float opacity)
{
  int x = 0;
  const __m128 v_opacity = _mm_set1_ps(opacity);

  for (; x <= width - 4; x += 4) {
    // 1. Load 8 floats
    __m128 r0 = _mm_loadu_ps(src + x * 2);     // [e0, o0, e1, o1]
    __m128 r1 = _mm_loadu_ps(src + x * 2 + 4); // [e2, o2, e3, o3]

    // 2. Shuffle to pick only the Even samples
    __m128 even = _mm_shuffle_ps(r0, r1, _MM_SHUFFLE(2, 0, 2, 0)); // [e0, e1, e2, e3]

    if constexpr (!full_opacity) {
      even = _mm_mul_ps(even, v_opacity);
    }

    _mm_storeu_ps(dst + x, even);
  }

  // Scalar Tail
  for (; x < width; x++) {
    const float val = src[x * 2];
    dst[x] = full_opacity ? val : val * opacity;
  }
}

// ---------------------------------------------------------------------------
// MASK420_TOPLEFT — top-left co-sited point sample (top row only, no averaging).
//   dst[x] = row0[x*2]
// ---------------------------------------------------------------------------
template<bool full_opacity>
static void fill_mask420_topleft_float_sse41(
  float* dst, const float* row0, int /*mask_pitch*/, int width, float opacity)
{
  // 420_TOPLEFT is identical to 422_TOPLEFT for the top row
  fill_mask422_topleft_float_sse41<full_opacity>(dst, row0, width, opacity);
}

// ---------------------------------------------------------------------------
// MASK411 — horizontal 4-tap box average.
//   avg[x] = (src[x*4]+src[x*4+1]+src[x*4+2]+src[x*4+3]) / 4 (*0.25)
// SSE4.1 float: 4 output pixels / iteration (16 input floats loaded)
// ---------------------------------------------------------------------------
template<bool full_opacity>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static void fill_mask411_float_sse41(
  float* dst, const float* src, int width, float opacity)
{
  int x = 0;
  const __m128 v_opacity = _mm_set1_ps(opacity);
  const __m128 v_025 = _mm_set1_ps(0.25f);

  for (; x <= width - 4; x += 4) {
    // 1. Load 16 floats (4 input blocks)
    __m128 r0 = _mm_loadu_ps(src + x * 4);      // [s0, s1, s2, s3]
    __m128 r1 = _mm_loadu_ps(src + x * 4 + 4);  // [s4, s5, s6, s7]
    __m128 r2 = _mm_loadu_ps(src + x * 4 + 8);  // [s8, s9, s10, s11]
    __m128 r3 = _mm_loadu_ps(src + x * 4 + 12); // [s12, s13, s14, s15]

    // 2. Horizontal sum pass 1: [s0+s1, s2+s3, s4+s5, s6+s7]
    __m128 h01 = _mm_hadd_ps(r0, r1);
    __m128 h23 = _mm_hadd_ps(r2, r3);

    // 3. Horizontal sum pass 2: Sum the pairs to get 4x-pixel sums
    // [(s0..s3), (s4..s7), (s8..s11), (s12..s15)]
    __m128 sum = _mm_hadd_ps(h01, h23);

    __m128 avg = _mm_mul_ps(sum, v_025);

    if constexpr (!full_opacity) {
      avg = _mm_mul_ps(avg, v_opacity);
    }

    _mm_storeu_ps(dst + x, avg);
  }

  // Scalar Tail
  for (; x < width; x++) {
    const float avg = (src[x * 4] + src[x * 4 + 1] + src[x * 4 + 2] + src[x * 4 + 3]) * 0.25f;
    dst[x] = full_opacity ? avg : avg * opacity;
  }
}

// ---------------------------------------------------------------------------
// prepare_effective_mask_for_row_sse41
// full_opacity == true  (default): MASK444 returns maskp; others fill buf with
//   spatial averages only.
// full_opacity == false: opacity baked in for every mode including MASK444.
// opacity_i, half, magic are ignored when full_opacity == true.
// ---------------------------------------------------------------------------
template<MaskMode maskMode, typename pixel_t, bool full_opacity>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
const pixel_t* prepare_effective_mask_for_row_sse41(
  const pixel_t* maskp,
  int mask_pitch,
  int width,
  std::vector<pixel_t>& buf,
  int opacity_i,
  int half,
  MagicDiv magic)
{
  if constexpr (maskMode == MASK444) {
    if constexpr (full_opacity) {
      return maskp;
    } else {
      // Copy row with opacity scaling into buf
      pixel_t* dst = buf.data();
      int x = 0;
      if constexpr (sizeof(pixel_t) == 1) {
        const __m128i v_opacity = _mm_set1_epi16((short)opacity_i);
        const __m128i v_half16  = _mm_set1_epi16((short)half);
        const __m128i v_mdiv    = _mm_set1_epi16((short)magic.div);
        for (; x <= width - 8; x += 8) {
          __m128i v   = _mm_loadl_epi64((const __m128i*)(maskp + x));
          __m128i v16 = _mm_cvtepu8_epi16(v);
          __m128i scaled = _mm_add_epi16(_mm_mullo_epi16(v16, v_opacity), v_half16);
          __m128i res    = _mm_srli_epi16(_mm_mulhi_epu16(scaled, v_mdiv), magic.shift);
          _mm_storel_epi64((__m128i*)(dst + x), _mm_packus_epi16(res, res));
        }
      } else {
        const __m128i v_opacity32 = _mm_set1_epi32(opacity_i);
        const __m128i v_half32    = _mm_set1_epi32(half);
        for (; x <= width - 4; x += 4) {
          __m128i v32 = _mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i*)(maskp + x)));
          __m128i res = simd_magic_div_32(
            _mm_add_epi32(_mm_mullo_epi32(v32, v_opacity32), v_half32),
            magic.div, magic.shift);
          _mm_storel_epi64((__m128i*)(dst + x), _mm_packus_epi32(res, res));
        }
      }
      for (; x < width; x++)
        dst[x] = static_cast<pixel_t>(
          magic_div_rt<pixel_t>((uint32_t)maskp[x] * (uint32_t)opacity_i + (uint32_t)half, magic));
      return dst;
    }
  }
  else {
    pixel_t* dst = buf.data();
    if constexpr (maskMode == MASK422)
      fill_mask422_sse41<pixel_t, full_opacity>(dst, maskp, width, opacity_i, half, magic);
    else if constexpr (maskMode == MASK422_MPEG2)
      fill_mask422_mpeg2_sse41<pixel_t, full_opacity>(dst, maskp, width, opacity_i, half, magic);
    else if constexpr (maskMode == MASK422_TOPLEFT)
      fill_mask422_topleft_sse41<pixel_t, full_opacity>(dst, maskp, width, opacity_i, half, magic);
    else if constexpr (maskMode == MASK420)
      fill_mask420_sse41<pixel_t, full_opacity>(dst, maskp, mask_pitch, width, opacity_i, half, magic);
    else if constexpr (maskMode == MASK420_MPEG2)
      fill_mask420_mpeg2_sse41<pixel_t, full_opacity>(dst, maskp, mask_pitch, width, opacity_i, half, magic);
    else if constexpr (maskMode == MASK420_TOPLEFT)
      fill_mask420_topleft_sse41<pixel_t, full_opacity>(dst, maskp, mask_pitch, width, opacity_i, half, magic);
    else if constexpr (maskMode == MASK411)
      fill_mask411_sse41<pixel_t, full_opacity>(dst, maskp, width, opacity_i, half, magic);
    else if constexpr (maskMode == MASK410 || maskMode == MASK440 ||
                        maskMode == MASK410_TOPLEFT || maskMode == MASK440_TOPLEFT || maskMode == MASK411_TOPLEFT) {
      // No SIMD kernel yet for these rare ratios -> scalar C fallback, see rowprep in blend_common.h.
      int mask_right = 0; // unused (only MPEG2 sliding-window modes need it)
      for (int x = 0; x < width; ++x) {
        const pixel_t avg = (pixel_t)calculate_effective_mask<maskMode>(maskp, x, mask_pitch, mask_right);
        if constexpr (full_opacity)
          dst[x] = avg;
        else
          dst[x] = static_cast<pixel_t>(
            magic_div_rt<pixel_t>((uint32_t)avg * (uint32_t)opacity_i + (uint32_t)half, magic));
      }
    }
    return dst;
  }
}

template<MaskMode maskMode, bool full_opacity>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
const float* prepare_effective_mask_for_row_float_sse41(
  const float* maskp,
  int mask_pitch,
  int width,
  std::vector<float>& buf,
  float opacity)
{
  if constexpr (maskMode == MASK444) {
    if constexpr (full_opacity) {
      return maskp;
    }
    else {
      float* dst = buf.data();
      int x = 0;
      const __m128 v_opacity = _mm_set1_ps(opacity);
      for (; x <= width - 4; x += 4) {
        // just put back opacity * mask
        __m128 v16 = _mm_loadu_ps(maskp + x);
        __m128 scaled = _mm_mul_ps(v16, v_opacity);
        _mm_storeu_ps(dst + x, scaled);
      }
      for (; x < width; x++)
        dst[x] = maskp[x] * opacity;
      return dst;
    }
  }
  else {
    float* dst = buf.data();
    if constexpr (maskMode == MASK422)
      fill_mask422_float_sse41<full_opacity>(dst, maskp, width, opacity);
    else if constexpr (maskMode == MASK422_MPEG2)
      fill_mask422_mpeg2_float_sse41<full_opacity>(dst, maskp, width, opacity);
    else if constexpr (maskMode == MASK422_TOPLEFT)
      fill_mask422_topleft_float_sse41<full_opacity>(dst, maskp, width, opacity);
    else if constexpr (maskMode == MASK420)
      fill_mask420_float_sse41<full_opacity>(dst, maskp, mask_pitch, width, opacity);
    else if constexpr (maskMode == MASK420_MPEG2)
      fill_mask420_mpeg2_float_sse41<full_opacity>(dst, maskp, mask_pitch, width, opacity);
    else if constexpr (maskMode == MASK420_TOPLEFT)
      fill_mask420_topleft_float_sse41<full_opacity>(dst, maskp, mask_pitch, width, opacity);
    else if constexpr (maskMode == MASK411)
      fill_mask411_float_sse41<full_opacity>(dst, maskp, width, opacity);
    else if constexpr (maskMode == MASK410 || maskMode == MASK440 ||
                        maskMode == MASK410_TOPLEFT || maskMode == MASK440_TOPLEFT || maskMode == MASK411_TOPLEFT) {
      // No SIMD kernel yet for these rare ratios -> scalar C fallback, see rowprep in blend_common.h.
      float mask_right = 0.0f; // unused (only MPEG2 sliding-window modes need it)
      for (int x = 0; x < width; ++x) {
        const float avg = calculate_effective_mask_f<maskMode>(maskp, x, mask_pitch, mask_right);
        if constexpr (full_opacity)
          dst[x] = avg;
        else
          dst[x] = avg * opacity;
      }
    }
    return dst;
  }
}


// ---------------------------------------------------------------------------
// Explicit instantiations
// ---------------------------------------------------------------------------

// prepare_effective_mask_for_row_sse41
#define INST_PREP_SSE41(mm, pt) \
  template const pt* prepare_effective_mask_for_row_sse41<mm, pt, true> (const pt*, int, int, std::vector<pt>&, int, int, MagicDiv); \
  template const pt* prepare_effective_mask_for_row_sse41<mm, pt, false>(const pt*, int, int, std::vector<pt>&, int, int, MagicDiv);
INST_PREP_SSE41(MASK444,          uint8_t)   INST_PREP_SSE41(MASK444,          uint16_t)
INST_PREP_SSE41(MASK420,          uint8_t)   INST_PREP_SSE41(MASK420,          uint16_t)
INST_PREP_SSE41(MASK420_MPEG2,    uint8_t)   INST_PREP_SSE41(MASK420_MPEG2,    uint16_t)
INST_PREP_SSE41(MASK420_TOPLEFT,  uint8_t)   INST_PREP_SSE41(MASK420_TOPLEFT,  uint16_t)
INST_PREP_SSE41(MASK422,          uint8_t)   INST_PREP_SSE41(MASK422,          uint16_t)
INST_PREP_SSE41(MASK422_MPEG2,    uint8_t)   INST_PREP_SSE41(MASK422_MPEG2,    uint16_t)
INST_PREP_SSE41(MASK422_TOPLEFT,  uint8_t)   INST_PREP_SSE41(MASK422_TOPLEFT,  uint16_t)
INST_PREP_SSE41(MASK411,          uint8_t)   INST_PREP_SSE41(MASK411,          uint16_t)
INST_PREP_SSE41(MASK411_TOPLEFT,  uint8_t)   INST_PREP_SSE41(MASK411_TOPLEFT,  uint16_t)
INST_PREP_SSE41(MASK440,          uint8_t)   INST_PREP_SSE41(MASK440,          uint16_t)
INST_PREP_SSE41(MASK440_TOPLEFT,  uint8_t)   INST_PREP_SSE41(MASK440_TOPLEFT,  uint16_t)
INST_PREP_SSE41(MASK410,          uint8_t)   INST_PREP_SSE41(MASK410,          uint16_t)
INST_PREP_SSE41(MASK410_TOPLEFT,  uint8_t)   INST_PREP_SSE41(MASK410_TOPLEFT,  uint16_t)
#undef INST_PREP_SSE41

// prepare_effective_mask_for_row_float_sse41
#define INST_PREP_SSE41(mm) \
  template const float* prepare_effective_mask_for_row_float_sse41<mm, true> (const float*, int, int, std::vector<float>&, float); \
  template const float* prepare_effective_mask_for_row_float_sse41<mm, false>(const float*, int, int, std::vector<float>&, float);
INST_PREP_SSE41(MASK444) 
INST_PREP_SSE41(MASK420)
INST_PREP_SSE41(MASK420_MPEG2)
INST_PREP_SSE41(MASK420_TOPLEFT)
INST_PREP_SSE41(MASK422)
INST_PREP_SSE41(MASK422_MPEG2)
INST_PREP_SSE41(MASK422_TOPLEFT)
INST_PREP_SSE41(MASK411)
INST_PREP_SSE41(MASK411_TOPLEFT)
INST_PREP_SSE41(MASK440)
INST_PREP_SSE41(MASK440_TOPLEFT)
INST_PREP_SSE41(MASK410)
INST_PREP_SSE41(MASK410_TOPLEFT)
#undef INST_PREP_SSE41

