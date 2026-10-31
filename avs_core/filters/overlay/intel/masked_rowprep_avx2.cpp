// masked_rowprep_avx2.cpp
// AVX2 rowprep implementations + explicit template instantiations.
// Compiled with -mavx2 -mfma (GCC/Clang) or /arch:AVX2 (MSVC) via handle_arch_flags(AVX2).
//
// avx2_pack_* helpers are static — internal to this TU only.

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <immintrin.h>
#endif

#include "avs/config.h"
#include "../blend_common.h"
#include "masked_rowprep_avx2_impl.h"   // declarations + simd_magic_div_32_avx2 inline
#include <vector>
#include <cstdint>

// ---------------------------------------------------------------------------
// Pack helpers — internal to this TU.
// ---------------------------------------------------------------------------
static AVS_FORCEINLINE __m128i avx2_pack_u16_to_u8(const __m256i& v) {
  return _mm_packus_epi16(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
}
static AVS_FORCEINLINE __m128i avx2_pack_u32_to_u16(const __m256i& v) {
  __m256i packed = _mm256_packus_epi32(v, _mm256_setzero_si256());
  return _mm_unpacklo_epi64(_mm256_castsi256_si128(packed), _mm256_extracti128_si256(packed, 1));
}

// ---------------------------------------------------------------------------
// MASK422 — horizontal 2-tap average.
//   avg[x] = (src[x*2] + src[x*2+1] + 1) >> 1
// ---------------------------------------------------------------------------
template<typename pixel_t, bool full_opacity>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("avx2")))
#endif
static void fill_mask422_avx2(
  pixel_t* dst, const pixel_t* src, int width,
  int opacity_i, int half, MagicDiv magic)
{
  int x = 0;
  if constexpr (sizeof(pixel_t) == 1) {
    [[maybe_unused]] const __m256i v_opacity = _mm256_set1_epi16((short)opacity_i);
    [[maybe_unused]] const __m256i v_half16  = _mm256_set1_epi16((short)half);
    [[maybe_unused]] const __m256i v_mdiv    = _mm256_set1_epi16((short)magic.div);
    const __m256i mask_lo = _mm256_set1_epi16(0x00FF);
    for (; x <= width - 16; x += 16) {
      __m256i v    = _mm256_loadu_si256((const __m256i*)(src + x * 2));
      __m256i even = _mm256_and_si256(v, mask_lo);
      __m256i odd  = _mm256_srli_epi16(v, 8);
      __m256i avg  = _mm256_srli_epi16(
        _mm256_add_epi16(_mm256_add_epi16(even, odd), _mm256_set1_epi16(1)), 1);
      if constexpr (!full_opacity) {
        __m256i scaled = _mm256_add_epi16(_mm256_mullo_epi16(avg, v_opacity), v_half16);
        avg = _mm256_srli_epi16(_mm256_mulhi_epu16(scaled, v_mdiv), magic.shift);
      }
      _mm_storeu_si128((__m128i*)(dst + x), avx2_pack_u16_to_u8(avg));
    }
  } else {
    // 16-bit: 16 uint16 luma -> 8 uint16 chroma per iteration.
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i v_pivot16 = _mm256_set1_epi16(-32768);
    // 65536 corrects the double-bias subtraction from madd, +1 handles the formula's rounding
    const __m256i v_correct32 = _mm256_set1_epi32(65536 + 1);

    [[maybe_unused]] const __m256i v_opacity32 = _mm256_set1_epi32(opacity_i);
    [[maybe_unused]] const __m256i v_half32 = _mm256_set1_epi32(half);

    for (; x <= width - 8; x += 8) {
      // Load 16 pixels (32 bytes) of uint16_t data
      __m256i v = _mm256_loadu_si256((const __m256i*)(src + x * 2));

      // unsigned 0..65535 -> signed -32768..32767 safely
      __m256i v_signed = _mm256_add_epi16(v, v_pivot16);

      // Horizontal pair addition: (a - 32768) + (b - 32768) = a + b - 65536
      __m256i sum32 = _mm256_madd_epi16(v_signed, ones);

      // Add back 65536 to undo the bias, add 1 for rounding, then logical shift right
      __m256i avg32 = _mm256_srli_epi32(_mm256_add_epi32(sum32, v_correct32), 1);

      if constexpr (!full_opacity)
        avg32 = simd_magic_div_32_avx2(
          _mm256_add_epi32(_mm256_mullo_epi32(avg32, v_opacity32), v_half32),
          magic.div, magic.shift);
      _mm_storeu_si128((__m128i*)(dst + x), avx2_pack_u32_to_u16(avg32));
    }
  }
  for (; x < width; x++) {
    const int avg = (src[x * 2] + src[x * 2 + 1] + 1) >> 1;
    dst[x] = full_opacity ? (pixel_t)avg
           : (pixel_t)magic_div_rt<pixel_t>((uint32_t)avg * (uint32_t)opacity_i + (uint32_t)half, magic);
  }
}

// ---------------------------------------------------------------------------
// MASK422_MPEG2 — horizontal 3-tap triangle filter with sliding window carry.
//   avg[x] = (left + 2*src[x*2] + src[x*2+1] + 2) >> 2
// ---------------------------------------------------------------------------
template<typename pixel_t, bool full_opacity>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("avx2")))
#endif
static void fill_mask422_mpeg2_avx2(
  pixel_t* dst, const pixel_t* src, int width,
  int opacity_i, int half, MagicDiv magic)
{
  int x = 0;
  if constexpr (sizeof(pixel_t) == 1) {
    [[maybe_unused]] const __m256i v_opacity = _mm256_set1_epi16((short)opacity_i);
    [[maybe_unused]] const __m256i v_half16  = _mm256_set1_epi16((short)half);
    [[maybe_unused]] const __m256i v_mdiv    = _mm256_set1_epi16((short)magic.div);
    const __m256i mask_lo = _mm256_set1_epi16(0x00FF);
    __m256i prev_carry = _mm256_castsi128_si256(
      _mm_insert_epi16(_mm_setzero_si128(), src[0], 7));

    for (; x <= width - 16; x += 16) {
      __m256i v    = _mm256_loadu_si256((const __m256i*)(src + x * 2));
      __m256i even = _mm256_and_si256(v, mask_lo);
      __m256i odd  = _mm256_srli_epi16(v, 8);

      __m256i shifted_odd = _mm256_permute2x128_si256(odd, prev_carry, 0x02);
      __m256i left = _mm256_alignr_epi8(odd, shifted_odd, 14);

      __m256i res = _mm256_srli_epi16(
        _mm256_add_epi16(
          _mm256_add_epi16(_mm256_add_epi16(left, _mm256_slli_epi16(even, 1)), odd),
          _mm256_set1_epi16(2)), 2);

      if constexpr (!full_opacity) {
        __m256i scaled = _mm256_add_epi16(_mm256_mullo_epi16(res, v_opacity), v_half16);
        res = _mm256_srli_epi16(_mm256_mulhi_epu16(scaled, v_mdiv), magic.shift);
      }
      _mm_storeu_si128((__m128i*)(dst + x), avx2_pack_u16_to_u8(res));

      prev_carry = _mm256_castsi128_si256(
        _mm_insert_epi16(_mm_setzero_si128(),
          _mm_extract_epi16(_mm256_extracti128_si256(odd, 1), 7), 7));
    }
    int right_val = _mm_extract_epi16(_mm256_castsi256_si128(prev_carry), 7);
    for (; x < width; x++) {
      const int left = right_val;
      const int mid  = src[x * 2];
      right_val      = src[x * 2 + 1];
      const int avg  = (left + 2 * mid + right_val + 2) >> 2;
      dst[x] = full_opacity ? (pixel_t)avg
             : (pixel_t)magic_div_rt<pixel_t>((uint32_t)avg * (uint32_t)opacity_i + (uint32_t)half, magic);
    }
  } else {
    [[maybe_unused]] const __m256i v_opacity32 = _mm256_set1_epi32(opacity_i);
    [[maybe_unused]] const __m256i v_half32    = _mm256_set1_epi32(half);
    __m256i prev_carry = _mm256_castsi128_si256(
      _mm_insert_epi32(_mm_setzero_si128(), src[0], 3));

    for (; x <= width - 8; x += 8) {
      __m256i v = _mm256_loadu_si256((const __m256i*)(src + x * 2));
      __m256i lo32 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(v));
      __m256i hi32 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(v, 1));

      __m256i sh_le = _mm256_shuffle_epi32(lo32, 0x88);
      __m256i sh_he = _mm256_shuffle_epi32(hi32, 0x88);
      __m256i even32 = _mm256_permute4x64_epi64(
        _mm256_unpacklo_epi64(sh_le, sh_he), _MM_SHUFFLE(3, 1, 2, 0));

      __m256i sh_lo = _mm256_shuffle_epi32(lo32, 0xDD);
      __m256i sh_ho = _mm256_shuffle_epi32(hi32, 0xDD);
      __m256i odd32 = _mm256_permute4x64_epi64(
        _mm256_unpacklo_epi64(sh_lo, sh_ho), _MM_SHUFFLE(3, 1, 2, 0));

      __m256i shifted_odd32 = _mm256_permute2x128_si256(odd32, prev_carry, 0x02);
      __m256i left = _mm256_alignr_epi8(odd32, shifted_odd32, 12);

      __m256i res = _mm256_srli_epi32(
        _mm256_add_epi32(
          _mm256_add_epi32(_mm256_add_epi32(left, _mm256_slli_epi32(even32, 1)), odd32),
          _mm256_set1_epi32(2)), 2);

      if constexpr (!full_opacity)
        res = simd_magic_div_32_avx2(
          _mm256_add_epi32(_mm256_mullo_epi32(res, v_opacity32), v_half32),
          magic.div, magic.shift);

      _mm_storeu_si128((__m128i*)(dst + x), avx2_pack_u32_to_u16(res));

      prev_carry = _mm256_castsi128_si256(
        _mm_insert_epi32(_mm_setzero_si128(),
          _mm_extract_epi32(_mm256_extracti128_si256(odd32, 1), 3), 3));
    }
    int right_val = _mm_extract_epi32(_mm256_castsi256_si128(prev_carry), 3);
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
// MASK420 — 2x2 box average (MPEG-1 placement).
//   avg[x] = (row0[x*2]+row0[x*2+1]+row1[x*2]+row1[x*2+1]+2) >> 2
// ---------------------------------------------------------------------------
template<typename pixel_t, bool full_opacity>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("avx2")))
#endif
static void fill_mask420_avx2(
  pixel_t* dst, const pixel_t* row0, int mask_pitch, int width,
  int opacity_i, int half, MagicDiv magic)
{
  const pixel_t* row1 = row0 + mask_pitch;
  int x = 0;
  if constexpr (sizeof(pixel_t) == 1) {
    [[maybe_unused]] const __m256i v_opacity = _mm256_set1_epi16((short)opacity_i);
    [[maybe_unused]] const __m256i v_half16  = _mm256_set1_epi16((short)half);
    [[maybe_unused]] const __m256i v_mdiv    = _mm256_set1_epi16((short)magic.div);
    const __m256i mask_lo = _mm256_set1_epi16(0x00FF);
    for (; x <= width - 16; x += 16) {
      __m256i r0 = _mm256_loadu_si256((const __m256i*)(row0 + x * 2));
      __m256i r1 = _mm256_loadu_si256((const __m256i*)(row1 + x * 2));
      __m256i e0 = _mm256_and_si256(r0, mask_lo);
      __m256i o0 = _mm256_srli_epi16(r0, 8);
      __m256i e1 = _mm256_and_si256(r1, mask_lo);
      __m256i o1 = _mm256_srli_epi16(r1, 8);
      __m256i avg = _mm256_srli_epi16(
        _mm256_add_epi16(
          _mm256_add_epi16(_mm256_add_epi16(e0, o0), _mm256_add_epi16(e1, o1)),
          _mm256_set1_epi16(2)), 2);
      if constexpr (!full_opacity) {
        __m256i scaled = _mm256_add_epi16(_mm256_mullo_epi16(avg, v_opacity), v_half16);
        avg = _mm256_srli_epi16(_mm256_mulhi_epu16(scaled, v_mdiv), magic.shift);
      }
      _mm_storeu_si128((__m128i*)(dst + x), avx2_pack_u16_to_u8(avg));
    }
  } else {
    // uint16_t: unsigned widening to avoid signed overflow in madd_epi16
    [[maybe_unused]] const __m256i v_opacity32 = _mm256_set1_epi32(opacity_i);
    [[maybe_unused]] const __m256i v_half32    = _mm256_set1_epi32(half);
    for (; x <= width - 8; x += 8) {
      __m256i v0     = _mm256_loadu_si256((const __m256i*)(row0 + x * 2));
      __m256i v1     = _mm256_loadu_si256((const __m256i*)(row1 + x * 2));
      __m256i r0_lo  = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(v0));
      __m256i r0_hi  = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(v0, 1));
      __m256i r1_lo  = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(v1));
      __m256i r1_hi  = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(v1, 1));
      __m256i sum_lo = _mm256_add_epi32(r0_lo, r1_lo);
      __m256i sum_hi = _mm256_add_epi32(r0_hi, r1_hi);
      __m256i sh_le  = _mm256_shuffle_epi32(sum_lo, 0x88);
      __m256i sh_he  = _mm256_shuffle_epi32(sum_hi, 0x88);
      __m256i even32 = _mm256_permute4x64_epi64(
        _mm256_unpacklo_epi64(sh_le, sh_he), _MM_SHUFFLE(3, 1, 2, 0));
      __m256i sh_lo  = _mm256_shuffle_epi32(sum_lo, 0xDD);
      __m256i sh_ho  = _mm256_shuffle_epi32(sum_hi, 0xDD);
      __m256i odd32  = _mm256_permute4x64_epi64(
        _mm256_unpacklo_epi64(sh_lo, sh_ho), _MM_SHUFFLE(3, 1, 2, 0));
      __m256i avg32  = _mm256_srli_epi32(
        _mm256_add_epi32(_mm256_add_epi32(even32, odd32), _mm256_set1_epi32(2)), 2);
      if constexpr (!full_opacity)
        avg32 = simd_magic_div_32_avx2(
          _mm256_add_epi32(_mm256_mullo_epi32(avg32, v_opacity32), v_half32),
          magic.div, magic.shift);
      _mm_storeu_si128((__m128i*)(dst + x), avx2_pack_u32_to_u16(avg32));
    }
  }
  for (; x < width; x++) {
    const int avg = ((int)row0[x*2] + row0[x*2+1] + row1[x*2] + row1[x*2+1] + 2) >> 2;
    dst[x] = full_opacity ? (pixel_t)avg
           : (pixel_t)magic_div_rt<pixel_t>((uint32_t)avg * (uint32_t)opacity_i + (uint32_t)half, magic);
  }
}

// ---------------------------------------------------------------------------
// MASK420_MPEG2 — horizontal 3-tap triangle filter with vertical 2-row sum and
// sliding-window carry.  Filter:
//   pe[x] = row0[x*2]   + row1[x*2]     (vertical sum of even-indexed pairs)
//   po[x] = row0[x*2+1] + row1[x*2+1]   (vertical sum of odd-indexed pairs)
//   avg[x] = (po[x-1] + 2*pe[x] + po[x] + 4) >> 3
// uint8_t:  16 output pixels / iteration (32-byte load per row → pe/po as uint16)
// uint16_t:  8 output pixels / iteration (32-byte load per row → pe/po as uint32)
// Cross-lane carry: 1 element = 14-byte alignr (uint8_t), 12-byte (uint16_t).
// ---------------------------------------------------------------------------
template<typename pixel_t, bool full_opacity>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("avx2")))
#endif
static void fill_mask420_mpeg2_avx2(
  pixel_t* dst, const pixel_t* row0, int mask_pitch, int width,
  int opacity_i, int half, MagicDiv magic)
{
  const pixel_t* row1 = row0 + mask_pitch;
  int x = 0;
  if constexpr (sizeof(pixel_t) == 1) {
    [[maybe_unused]] const __m256i v_opacity = _mm256_set1_epi16((short)opacity_i);
    [[maybe_unused]] const __m256i v_half16  = _mm256_set1_epi16((short)half);
    [[maybe_unused]] const __m256i v_mdiv    = _mm256_set1_epi16((short)magic.div);
    const __m256i mask_lo = _mm256_set1_epi16(0x00FF);
    const int p0 = (int)row0[0] + row1[0];
    __m256i prev_carry = _mm256_castsi128_si256(
      _mm_insert_epi16(_mm_setzero_si128(), p0, 7));

    for (; x <= width - 16; x += 16) {
      __m256i r0 = _mm256_loadu_si256((const __m256i*)(row0 + x * 2));
      __m256i r1 = _mm256_loadu_si256((const __m256i*)(row1 + x * 2));
      __m256i pe = _mm256_add_epi16(_mm256_and_si256(r0, mask_lo),
                                     _mm256_and_si256(r1, mask_lo));
      __m256i po = _mm256_add_epi16(_mm256_srli_epi16(r0, 8),
                                     _mm256_srli_epi16(r1, 8));

      __m256i shifted_odd = _mm256_permute2x128_si256(po, prev_carry, 0x02);
      __m256i left = _mm256_alignr_epi8(po, shifted_odd, 14);

      __m256i res = _mm256_srli_epi16(
        _mm256_add_epi16(
          _mm256_add_epi16(_mm256_add_epi16(left, _mm256_slli_epi16(pe, 1)), po),
          _mm256_set1_epi16(4)), 3);

      if constexpr (!full_opacity) {
        __m256i scaled = _mm256_add_epi16(_mm256_mullo_epi16(res, v_opacity), v_half16);
        res = _mm256_srli_epi16(_mm256_mulhi_epu16(scaled, v_mdiv), magic.shift);
      }
      _mm_storeu_si128((__m128i*)(dst + x), avx2_pack_u16_to_u8(res));

      prev_carry = _mm256_castsi128_si256(
        _mm_insert_epi16(_mm_setzero_si128(),
          _mm_extract_epi16(_mm256_extracti128_si256(po, 1), 7), 7));
    }
    int right_val = _mm_extract_epi16(_mm256_castsi256_si128(prev_carry), 7);
    for (; x < width; x++) {
      const int left = right_val;
      const int mid  = (int)row0[x*2]   + row1[x*2];
      right_val      = (int)row0[x*2+1] + row1[x*2+1];
      const int avg  = (left + 2 * mid + right_val + 4) >> 3;
      dst[x] = full_opacity ? (pixel_t)avg
             : (pixel_t)magic_div_rt<pixel_t>((uint32_t)avg * (uint32_t)opacity_i + (uint32_t)half, magic);
    }
  } else {
    [[maybe_unused]] const __m256i v_opacity32 = _mm256_set1_epi32(opacity_i);
    [[maybe_unused]] const __m256i v_half32    = _mm256_set1_epi32(half);
    const int p0 = (int)row0[0] + row1[0];
    __m256i prev_carry = _mm256_castsi128_si256(
      _mm_insert_epi32(_mm_setzero_si128(), p0, 3));

    for (; x <= width - 8; x += 8) {
      __m256i r0 = _mm256_loadu_si256((const __m256i*)(row0 + x * 2));
      __m256i r1 = _mm256_loadu_si256((const __m256i*)(row1 + x * 2));
      // Expand each 8-uint16 half to 8 uint32 and sum vertically.
      __m256i plo32 = _mm256_add_epi32(
        _mm256_cvtepu16_epi32(_mm256_castsi256_si128(r0)),
        _mm256_cvtepu16_epi32(_mm256_castsi256_si128(r1)));
      __m256i phi32 = _mm256_add_epi32(
        _mm256_cvtepu16_epi32(_mm256_extracti128_si256(r0, 1)),
        _mm256_cvtepu16_epi32(_mm256_extracti128_si256(r1, 1)));

      // Deinterleave even/odd (same as fill_mask422_mpeg2_avx2 uint16_t path).
      __m256i sh_le  = _mm256_shuffle_epi32(plo32, 0x88);
      __m256i sh_he  = _mm256_shuffle_epi32(phi32, 0x88);
      __m256i even32 = _mm256_permute4x64_epi64(
        _mm256_unpacklo_epi64(sh_le, sh_he), _MM_SHUFFLE(3, 1, 2, 0));

      __m256i sh_lo = _mm256_shuffle_epi32(plo32, 0xDD);
      __m256i sh_ho = _mm256_shuffle_epi32(phi32, 0xDD);
      __m256i odd32 = _mm256_permute4x64_epi64(
        _mm256_unpacklo_epi64(sh_lo, sh_ho), _MM_SHUFFLE(3, 1, 2, 0));

      __m256i shifted_odd32 = _mm256_permute2x128_si256(odd32, prev_carry, 0x02);
      __m256i left = _mm256_alignr_epi8(odd32, shifted_odd32, 12);

      __m256i res = _mm256_srli_epi32(
        _mm256_add_epi32(
          _mm256_add_epi32(_mm256_add_epi32(left, _mm256_slli_epi32(even32, 1)), odd32),
          _mm256_set1_epi32(4)), 3);

      if constexpr (!full_opacity)
        res = simd_magic_div_32_avx2(
          _mm256_add_epi32(_mm256_mullo_epi32(res, v_opacity32), v_half32),
          magic.div, magic.shift);

      _mm_storeu_si128((__m128i*)(dst + x), avx2_pack_u32_to_u16(res));

      prev_carry = _mm256_castsi128_si256(
        _mm_insert_epi32(_mm_setzero_si128(),
          _mm_extract_epi32(_mm256_extracti128_si256(odd32, 1), 3), 3));
    }
    int right_val = _mm_extract_epi32(_mm256_castsi256_si128(prev_carry), 3);
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
// ---------------------------------------------------------------------------
template<typename pixel_t, bool full_opacity>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("avx2")))
#endif
static void fill_mask422_topleft_avx2(
  pixel_t* dst, const pixel_t* src, int width,
  int opacity_i, int half, MagicDiv magic)
{
  int x = 0;
  if constexpr (sizeof(pixel_t) == 1) {
    [[maybe_unused]] const __m256i v_opacity = _mm256_set1_epi16((short)opacity_i);
    [[maybe_unused]] const __m256i v_half16  = _mm256_set1_epi16((short)half);
    [[maybe_unused]] const __m256i v_mdiv    = _mm256_set1_epi16((short)magic.div);
    const __m256i mask_lo = _mm256_set1_epi16(0x00FF);
    for (; x <= width - 16; x += 16) {
      __m256i v    = _mm256_loadu_si256((const __m256i*)(src + x * 2));
      __m256i even = _mm256_and_si256(v, mask_lo); // left (even) bytes as uint16
      if constexpr (!full_opacity) {
        __m256i scaled = _mm256_add_epi16(_mm256_mullo_epi16(even, v_opacity), v_half16);
        even = _mm256_srli_epi16(_mm256_mulhi_epu16(scaled, v_mdiv), magic.shift);
      }
      _mm_storeu_si128((__m128i*)(dst + x), avx2_pack_u16_to_u8(even));
    }
  } else {
    // 16-bit: grab even-indexed elements (src[x*2], src[x*2+2], ...)
    [[maybe_unused]] const __m256i v_opacity32 = _mm256_set1_epi32(opacity_i);
    [[maybe_unused]] const __m256i v_half32    = _mm256_set1_epi32(half);
    for (; x <= width - 8; x += 8) {
      __m256i v = _mm256_loadu_si256((const __m256i*)(src + x * 2));
      // Deinterleave: keep even-indexed uint16 elements
      __m256i lo32 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(v));
      __m256i hi32 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(v, 1));
      __m256i sh_le  = _mm256_shuffle_epi32(lo32, 0x88); // [v0,v2,_,_] in lo64 per lane
      __m256i sh_he  = _mm256_shuffle_epi32(hi32, 0x88);
      __m256i even32 = _mm256_permute4x64_epi64(
        _mm256_unpacklo_epi64(sh_le, sh_he), _MM_SHUFFLE(3, 1, 2, 0));
      if constexpr (!full_opacity)
        even32 = simd_magic_div_32_avx2(
          _mm256_add_epi32(_mm256_mullo_epi32(even32, v_opacity32), v_half32),
          magic.div, magic.shift);
      _mm_storeu_si128((__m128i*)(dst + x), avx2_pack_u32_to_u16(even32));
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
__attribute__((__target__("avx2")))
#endif
static void fill_mask420_topleft_avx2(
  pixel_t* dst, const pixel_t* row0, int /*mask_pitch*/, int width,
  int opacity_i, int half, MagicDiv magic)
{
  // Identical to fill_mask422_topleft_avx2: top row only, left co-sited.
  fill_mask422_topleft_avx2<pixel_t, full_opacity>(dst, row0, width, opacity_i, half, magic);
}

// ---------------------------------------------------------------------------
// MASK411 — horizontal 4-tap box average.
//   avg[x] = (src[x*4]+src[x*4+1]+src[x*4+2]+src[x*4+3]+2) >> 2
// ---------------------------------------------------------------------------
template<typename pixel_t, bool full_opacity>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("avx2")))
#endif
static void fill_mask411_avx2(
  pixel_t* dst, const pixel_t* src, int width,
  int opacity_i, int half, MagicDiv magic)
{
  int x = 0;
  if constexpr (sizeof(pixel_t) == 1) {
    const __m256i zero = _mm256_setzero_si256();
    [[maybe_unused]] const __m256i v_opacity = _mm256_set1_epi16((short)opacity_i);
    [[maybe_unused]] const __m256i v_half16  = _mm256_set1_epi16((short)half);
    [[maybe_unused]] const __m256i v_mdiv    = _mm256_set1_epi16((short)magic.div);
    for (; x <= width - 16; x += 16) {
      __m256i v0 = _mm256_loadu_si256((const __m256i*)(src + x * 4));
      __m256i v1 = _mm256_loadu_si256((const __m256i*)(src + x * 4 + 32));
      __m256i p0 = _mm256_hadd_epi16(
        _mm256_unpacklo_epi8(v0, zero), _mm256_unpackhi_epi8(v0, zero));
      __m256i p1 = _mm256_hadd_epi16(
        _mm256_unpacklo_epi8(v1, zero), _mm256_unpackhi_epi8(v1, zero));
      __m256i avg = _mm256_srli_epi16(
        _mm256_add_epi16(_mm256_hadd_epi16(p0, p1), _mm256_set1_epi16(2)), 2);
      avg = _mm256_permute4x64_epi64(avg, _MM_SHUFFLE(3, 1, 2, 0));
      if constexpr (!full_opacity) {
        __m256i scaled = _mm256_add_epi16(_mm256_mullo_epi16(avg, v_opacity), v_half16);
        avg = _mm256_srli_epi16(_mm256_mulhi_epu16(scaled, v_mdiv), magic.shift);
      }
      _mm_storeu_si128((__m128i*)(dst + x), avx2_pack_u16_to_u8(avg));
    }
  } else {
    // uint16_t: unsigned widening to avoid signed overflow in madd_epi16
    [[maybe_unused]] const __m256i v_opacity32 = _mm256_set1_epi32(opacity_i);
    [[maybe_unused]] const __m256i v_half32    = _mm256_set1_epi32(half);
    for (; x <= width - 8; x += 8) {
      __m256i v0   = _mm256_loadu_si256((const __m256i*)(src + x * 4));       // s0..s15
      __m256i v1   = _mm256_loadu_si256((const __m256i*)(src + x * 4 + 16));  // s16..s31
      __m256i lo0  = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(v0));
      __m256i hi0  = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(v0, 1));
      __m256i lo1  = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(v1));
      __m256i hi1  = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(v1, 1));
      __m256i pair0 = _mm256_hadd_epi32(lo0, hi0); // lo:[s0+s1,s2+s3,s8+s9,s10+s11] hi:[s4+s5,s6+s7,s12+s13,s14+s15]
      __m256i pair1 = _mm256_hadd_epi32(lo1, hi1);
      __m256i quad0 = _mm256_hadd_epi32(pair0, pair0); // lo:[G0,G2,G0,G2] hi:[G1,G3,G1,G3]
      __m256i quad1 = _mm256_hadd_epi32(pair1, pair1); // lo:[G4,G6,G4,G6] hi:[G5,G7,G5,G7]
      __m128i g0123 = _mm_unpacklo_epi32(_mm256_castsi256_si128(quad0),
                                          _mm256_extracti128_si256(quad0, 1)); // [G0,G1,G2,G3]
      __m128i g4567 = _mm_unpacklo_epi32(_mm256_castsi256_si128(quad1),
                                          _mm256_extracti128_si256(quad1, 1)); // [G4,G5,G6,G7]
      __m256i avg32 = _mm256_srli_epi32(
        _mm256_add_epi32(_mm256_set_m128i(g4567, g0123), _mm256_set1_epi32(2)), 2);
      if constexpr (!full_opacity)
        avg32 = simd_magic_div_32_avx2(
          _mm256_add_epi32(_mm256_mullo_epi32(avg32, v_opacity32), v_half32),
          magic.div, magic.shift);
      _mm_storeu_si128((__m128i*)(dst + x), avx2_pack_u32_to_u16(avg32));
    }
  }
  for (; x < width; x++) {
    const int avg = ((int)src[x*4] + src[x*4+1] + src[x*4+2] + src[x*4+3] + 2) >> 2;
    dst[x] = full_opacity ? (pixel_t)avg
           : (pixel_t)magic_div_rt<pixel_t>((uint32_t)avg * (uint32_t)opacity_i + (uint32_t)half, magic);
  }
}
// ---------------------------
// End of integer mask helpers
// ---------------------------

// ---------------------------
// Start of float mask helpers
// ---------------------------

// MASK422 — horizontal 2-tap box average.
//   avg[x] = (src[x*2] + src[x*2+1]) * 0.5f
//
// float: 8 output pixels / iteration (16 input floats loaded)
// ---------------------------------------------------------------------------
template<bool full_opacity>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("avx2")))
#endif
static void fill_mask422_float_avx2(
  float* dst, const float* src, int width, float opacity)
{
  int x = 0;
  const __m256 v_opacity = _mm256_set1_ps(opacity);
  const __m256 v_05 = _mm256_set1_ps(0.5f);

  for (; x <= width - 8; x += 8) {
    // 1. Load 16 floats (two 256-bit registers)
    __m256 r_lo = _mm256_loadu_ps(src + x * 2);
    __m256 r_hi = _mm256_loadu_ps(src + x * 2 + 8);

    // 2. De-interleave Even and Odd samples
    // r_lo: [s0, s1, s2, s3, s4, s5, s6, s7]
    // r_hi: [s8, s9, s10, s11, s12, s13, s14, s15]
    __m256 even_shuf = _mm256_shuffle_ps(r_lo, r_hi, _MM_SHUFFLE(2, 0, 2, 0));
    __m256 odd_shuf = _mm256_shuffle_ps(r_lo, r_hi, _MM_SHUFFLE(3, 1, 3, 1));

    // Fix lane crossing to get contiguous even and odd vectors
    __m256 even = _mm256_castsi256_ps(
      _mm256_permute4x64_epi64(_mm256_castps_si256(even_shuf), _MM_SHUFFLE(3, 1, 2, 0))
    );
    __m256 odd = _mm256_castsi256_ps(
      _mm256_permute4x64_epi64(_mm256_castps_si256(odd_shuf), _MM_SHUFFLE(3, 1, 2, 0))
    );

    // 3. Average: (even + odd) * 0.5
    __m256 avg = _mm256_mul_ps(_mm256_add_ps(even, odd), v_05);

    if constexpr (!full_opacity) {
      avg = _mm256_mul_ps(avg, v_opacity);
    }

    // 4. Store 8 output pixels
    _mm256_storeu_ps(dst + x, avg);
  }

  // Scalar Tail
  for (; x < width; x++) {
    const float avg = (src[x * 2] + src[x * 2 + 1]) * 0.5f;
    dst[x] = full_opacity ? avg : avg * opacity;
  }
}

// ---------------------------------------------------------------------------
// MASK422_MPEG2 — horizontal 3-tap triangle filter with sliding window carry.
//   avg[x] = (left + 2*src[x*2] + src[x*2+1]) * 0.25f
//
// float: 8 output pixels / iteration (16 input floats per iteration)
// Cross-lane carry: 1 element = 12-byte alignr (4 bytes per float).
// ---------------------------------------------------------------------------
template<bool full_opacity>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("avx2")))
#endif
static void fill_mask422_mpeg2_float_avx2(
  float* dst, const float* src, int width, float opacity)
{
  int x = 0;

  // Seed the carry (left neighbor) with the first even sample 
  // or as per your boundary requirements.
  float right_val = src[0];

  const __m256 v_opacity = _mm256_set1_ps(opacity);
  const __m256 v_025 = _mm256_set1_ps(0.25f);

  // v_prev_carry holds the last 'odd' sample from the previous vector
  __m256 v_prev_carry = _mm256_set1_ps(right_val);

  for (; x <= width - 8; x += 8) {
    // 1. Load 16 floats (one full source block for 8 output pixels)
    __m256 v0 = _mm256_loadu_ps(src + x * 2);

    // 2. De-interleave: Even (src[x*2]) and Odd (src[x*2+1])
    // s0, s1, s2, s3, s4, s5, s6, s7 | s8, s9, s10, s11, s12, s13, s14, s15
    __m256 even = _mm256_permutevar8x32_ps(v0, _mm256_set_epi32(14, 12, 10, 8, 6, 4, 2, 0));
    __m256 odd = _mm256_permutevar8x32_ps(v0, _mm256_set_epi32(15, 13, 11, 9, 7, 5, 3, 1));

    // 3. Sliding Window: Construct 'left' neighbor vector
    // shifted_odd: [prev_o4..o7, o0..o3] - brings previous high into current low
    __m256 shifted_odd = _mm256_permute2f128_ps(odd, v_prev_carry, 0x02);

    // left: [prev_o7, o0, o1, o2 | o3, o4, o5, o6]
    // alignr by 12 bytes shifts the 128-bit window by 3 floats (leaving 1 float carry)
    __m256 left = _mm256_castsi256_ps(_mm256_alignr_epi8(
      _mm256_castps_si256(odd),
      _mm256_castps_si256(shifted_odd), 12));

    // 4. Filter: avg = (left + 2*even + odd) * 0.25
    __m256 avg = _mm256_add_ps(left, _mm256_add_ps(_mm256_add_ps(even, even), odd));
    avg = _mm256_mul_ps(avg, v_025);

    if constexpr (!full_opacity) {
      avg = _mm256_mul_ps(avg, v_opacity);
    }

    _mm256_storeu_ps(dst + x, avg);

    // 5. Carry the last element of 'odd' (index 7) for the next iteration
    v_prev_carry = _mm256_permutevar8x32_ps(odd, _mm256_set1_epi32(7));
  }

  // Extraction for scalar tail
  right_val = _mm256_cvtss_f32(v_prev_carry);

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
//   avg[x] = (row0[x*2] + row0[x*2+1] + row1[x*2] + row1[x*2+1]) * 0.25f
//
// float: 8 output pixels / iteration (requires 16 input floats per row)
// ---------------------------------------------------------------------------
template<bool full_opacity>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("avx2")))
#endif
static void fill_mask420_float_avx2(
  float* dst, const float* row0, int mask_pitch, int width, float opacity)
{
  const float* row1 = row0 + mask_pitch;
  int x = 0;

  const __m256 v_opacity = _mm256_set1_ps(opacity);
  const __m256 v_025 = _mm256_set1_ps(0.25f);

  for (; x <= width - 8; x += 8) {
    // 1. Load 16 floats from each row (total 32 floats per iteration)
    __m256 r0_lo = _mm256_loadu_ps(row0 + x * 2);
    __m256 r0_hi = _mm256_loadu_ps(row0 + x * 2 + 8);
    __m256 r1_lo = _mm256_loadu_ps(row1 + x * 2);
    __m256 r1_hi = _mm256_loadu_ps(row1 + x * 2 + 8);

    // 2. Vertical Sum: Interleaved [e0, o0, e1, o1, e2, o2, e3, o3...]
    __m256 s_lo = _mm256_add_ps(r0_lo, r1_lo);
    __m256 s_hi = _mm256_add_ps(r0_hi, r1_hi);

    // 3. Horizontal Sum via De-interleave
    // We shuffle 's_lo' and 's_hi' to separate even and odd indices
    __m256 even_shuf = _mm256_shuffle_ps(s_lo, s_hi, _MM_SHUFFLE(2, 0, 2, 0));
    __m256 odd_shuf = _mm256_shuffle_ps(s_lo, s_hi, _MM_SHUFFLE(3, 1, 3, 1));

    // Fix AVX2 lane crossing: [L_even, H_even] and [L_odd, H_odd]
    __m256 even = _mm256_castsi256_ps(
      _mm256_permute4x64_epi64(_mm256_castps_si256(even_shuf), _MM_SHUFFLE(3, 1, 2, 0))
    );
    __m256 odd = _mm256_castsi256_ps(
      _mm256_permute4x64_epi64(_mm256_castps_si256(odd_shuf), _MM_SHUFFLE(3, 1, 2, 0))
    );

    // 4. Final Average: (even + odd) * 0.25
    __m256 avg = _mm256_mul_ps(_mm256_add_ps(even, odd), v_025);

    if constexpr (!full_opacity) {
      avg = _mm256_mul_ps(avg, v_opacity);
    }

    _mm256_storeu_ps(dst + x, avg);
  }

  // Scalar Tail
  for (; x < width; x++) {
    const float sum = row0[x * 2] + row0[x * 2 + 1] +
      row1[x * 2] + row1[x * 2 + 1];
    const float avg = sum * 0.25f;
    dst[x] = full_opacity ? avg : avg * opacity;
  }
}

// ---------------------------------------------------------------------------
// MASK420_MPEG2 — horizontal 3-tap triangle filter with vertical 2-row sum and
// sliding-window carry. Filter logic:
//   pe[x]  = row0[x*2]   + row1[x*2]     (vertical sum of even-indexed pairs)
//   po[x]  = row0[x*2+1] + row1[x*2+1]   (vertical sum of odd-indexed pairs)
//   avg[x] = (po[x-1] + 2*pe[x] + po[x]) * 0.125f
//
// float: 8 output pixels / iteration (32-byte load per row -> 16 floats).
// Cross-lane carry: 1 element = 12-byte alignr (4 bytes per float).
// ---------------------------------------------------------------------------
template<bool full_opacity>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("avx2")))
#endif
static void fill_mask420_mpeg2_float_avx2(
  float* dst, const float* row0, int mask_pitch, int width, float opacity)
{
  const float* row1 = row0 + mask_pitch;
  int x = 0;

  // We need the "odd" sum from the pixel immediately before the SIMD block.
  // If x=0, this is technically out of bounds or needs a padding strategy.
  // Assuming p0 is the vertical sum at index -1 or starting logic:
  float right_val = row0[0] + row1[0];
  // Broadcast the last 'odd' sum into the carry register
  // We only need the very last element of the previous 'odd' vector.
  __m256 v_prev_odd = _mm256_set1_ps(right_val);
  const __m256 v_opacity = _mm256_set1_ps(opacity);

  for (; x <= width - 8; x += 8) {
    // 1. Load 16 floats from each row (2x 256-bit loads)
    __m256 r0_lo = _mm256_loadu_ps(row0 + x * 2);
    __m256 r0_hi = _mm256_loadu_ps(row0 + x * 2 + 8);
    __m256 r1_lo = _mm256_loadu_ps(row1 + x * 2);
    __m256 r1_hi = _mm256_loadu_ps(row1 + x * 2 + 8);

    // 2. Vertical Sum (pe and po interleaved: [e0, o0, e1, o1...])
    __m256 sum_lo = _mm256_add_ps(r0_lo, r1_lo);
    __m256 sum_hi = _mm256_add_ps(r0_hi, r1_hi);

    // 3. De-interleave Even and Odd
    // Use shuffle to get [e0, e1, e2, e3, e4, e5, e6, e7] and [o0, o1...]
    // Note: _mm256_shuffle_ps works within 128-bit lanes, so we fix it with a permute.
    __m256 even_shuf = _mm256_shuffle_ps(sum_lo, sum_hi, _MM_SHUFFLE(2, 0, 2, 0));
    __m256 odd_shuf = _mm256_shuffle_ps(sum_lo, sum_hi, _MM_SHUFFLE(3, 1, 3, 1));

    // Fix lane crossing: [L0, L1, H0, H1] -> [L0, H0, L1, H1]
    __m256 even = _mm256_castsi256_ps(
      _mm256_permute4x64_epi64(_mm256_castps_si256(even_shuf), _MM_SHUFFLE(3, 1, 2, 0))
    );
    __m256 odd = _mm256_castsi256_ps(
      _mm256_permute4x64_epi64(_mm256_castps_si256(odd_shuf), _MM_SHUFFLE(3, 1, 2, 0))
    );

    // 4. Construct 'left' vector: [prev_o7, o0, o1, o2, o3, o4, o5, o6]
    // Slide 'odd' right by one element, bringing in the last element from the previous iteration.
    __m256 shifted_odd = _mm256_permute2f128_ps(odd, v_prev_odd, 0x02);
    __m256 left = _mm256_castsi256_ps(_mm256_alignr_epi8(
      _mm256_castps_si256(odd),
      _mm256_castps_si256(shifted_odd), 12));

    // 5. Triangle Filter: (left + 2*even + odd) / 8
    __m256 avg = _mm256_mul_ps(
      _mm256_add_ps(_mm256_add_ps(left, _mm256_add_ps(even, even)), odd),
      _mm256_set1_ps(0.125f)
    );

    if constexpr (!full_opacity) {
      avg = _mm256_mul_ps(avg, v_opacity);
    }

    _mm256_storeu_ps(dst + x, avg);

    // Update carry for next iteration: the last 'odd' sum becomes the next 'left' start
    v_prev_odd = _mm256_set1_ps(((float*)&odd)[7]);
  }

  // Bridge to scalar tail: Extract the last odd sum processed by SIMD
  right_val = _mm256_cvtss_f32(_mm256_permutevar8x32_ps(v_prev_odd, _mm256_setzero_si256()));
  for (; x < width; x++) {
    const float left = right_val;
    const float mid = row0[x * 2] + row1[x * 2];
    right_val = row0[x * 2 + 1] + row1[x * 2 + 1];
    const float avg = (left + 2 * mid + right_val + 4) * 0.125f; // / 8.0f
    dst[x] = full_opacity ? avg : avg * opacity;
  }
}


// ---------------------------------------------------------------------------
// MASK422_TOPLEFT — left co-sited point sample (no averaging).
//   dst[x] = src[x*2]
// ---------------------------------------------------------------------------
template<bool full_opacity>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("avx2")))
#endif
static void fill_mask422_topleft_float_avx2(
  float* dst, const float* src, int width,
  float opacity)
{
  int x = 0;
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
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("avx2")))
#endif
static void fill_mask420_topleft_float_avx2(
  float* dst, const float* row0, int /*mask_pitch*/, int width,
  float opacity)
{
  // Identical to fill_mask422_topleft_float_avx2: top row only, left co-sited.
  fill_mask422_topleft_float_avx2<full_opacity>(dst, row0, width, opacity);
}

// ---------------------------------------------------------------------------
// MASK411 — horizontal 4-tap box average.
//   avg[x] = (src[x*4]+src[x*4+1]+src[x*4+2]+src[x*4+3]) / 4 (*0.25)
// ---------------------------------------------------------------------------
template<bool full_opacity>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("avx2")))
#endif
static void fill_mask411_float_avx2(
  float* dst, const float* src, int width,
  float opacity)
{
  int x = 0;
  for (; x < width; x++) {
    const float avg = (src[x * 4] + src[x * 4 + 1] + src[x * 4 + 2] + src[x * 4 + 3]) * 0.25f;
    dst[x] = full_opacity ? avg : avg * opacity;
  }
}

// ---------------------------
// End of float mask helpers
// ---------------------------


// ---------------------------------------------------------------------------
// prepare_effective_mask_for_row_avx2
// ---------------------------------------------------------------------------
template<MaskMode maskMode, typename pixel_t, bool full_opacity>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("avx2")))
#endif
const pixel_t* prepare_effective_mask_for_row_avx2(
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
      pixel_t* dst = buf.data();
      int x = 0;
      if constexpr (sizeof(pixel_t) == 1) {
        const __m256i v_opacity = _mm256_set1_epi16((short)opacity_i);
        const __m256i v_half16  = _mm256_set1_epi16((short)half);
        const __m256i v_mdiv    = _mm256_set1_epi16((short)magic.div);
        for (; x <= width - 16; x += 16) {
          __m256i v16 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(maskp + x)));
          __m256i scaled = _mm256_add_epi16(_mm256_mullo_epi16(v16, v_opacity), v_half16);
          __m256i res    = _mm256_srli_epi16(_mm256_mulhi_epu16(scaled, v_mdiv), magic.shift);
          _mm_storeu_si128((__m128i*)(dst + x), avx2_pack_u16_to_u8(res));
        }
      } else {
        const __m256i v_opacity32 = _mm256_set1_epi32(opacity_i);
        const __m256i v_half32    = _mm256_set1_epi32(half);
        for (; x <= width - 8; x += 8) {
          __m256i v32 = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(maskp + x)));
          __m256i res = simd_magic_div_32_avx2(
            _mm256_add_epi32(_mm256_mullo_epi32(v32, v_opacity32), v_half32),
            magic.div, magic.shift);
          _mm_storeu_si128((__m128i*)(dst + x), avx2_pack_u32_to_u16(res));
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
      fill_mask422_avx2<pixel_t, full_opacity>(dst, maskp, width, opacity_i, half, magic);
    else if constexpr (maskMode == MASK422_MPEG2)
      fill_mask422_mpeg2_avx2<pixel_t, full_opacity>(dst, maskp, width, opacity_i, half, magic);
    else if constexpr (maskMode == MASK422_TOPLEFT)
      fill_mask422_topleft_avx2<pixel_t, full_opacity>(dst, maskp, width, opacity_i, half, magic);
    else if constexpr (maskMode == MASK420)
      fill_mask420_avx2<pixel_t, full_opacity>(dst, maskp, mask_pitch, width, opacity_i, half, magic);
    else if constexpr (maskMode == MASK420_MPEG2)
      fill_mask420_mpeg2_avx2<pixel_t, full_opacity>(dst, maskp, mask_pitch, width, opacity_i, half, magic);
    else if constexpr (maskMode == MASK420_TOPLEFT)
      fill_mask420_topleft_avx2<pixel_t, full_opacity>(dst, maskp, mask_pitch, width, opacity_i, half, magic);
    else if constexpr (maskMode == MASK411)
      fill_mask411_avx2<pixel_t, full_opacity>(dst, maskp, width, opacity_i, half, magic);
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
__attribute__((__target__("avx2")))
#endif
const float* prepare_effective_mask_for_row_float_avx2(
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
      const __m256 v_opacity = _mm256_set1_ps(opacity);
      for (; x <= width - 8; x += 8) {
        // just put back opacity * mask
        __m256 v16 = _mm256_loadu_ps(maskp + x);
        __m256 scaled = _mm256_mul_ps(v16, v_opacity);
        _mm256_storeu_ps(dst + x, scaled);
      }
      for (; x <= width - 4; x += 4) {
        // just put back opacity * mask
        __m128 v16 = _mm_loadu_ps(maskp + x);
        __m128 scaled = _mm_mul_ps(v16, _mm_set1_ps(opacity));
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
      fill_mask422_float_avx2<full_opacity>(dst, maskp, width, opacity);
    else if constexpr (maskMode == MASK422_MPEG2)
      fill_mask422_mpeg2_float_avx2<full_opacity>(dst, maskp, width, opacity);
    else if constexpr (maskMode == MASK422_TOPLEFT)
      fill_mask422_topleft_float_avx2<full_opacity>(dst, maskp, width, opacity);
    else if constexpr (maskMode == MASK420)
      fill_mask420_float_avx2<full_opacity>(dst, maskp, mask_pitch, width, opacity);
    else if constexpr (maskMode == MASK420_MPEG2)
      fill_mask420_mpeg2_float_avx2<full_opacity>(dst, maskp, mask_pitch, width, opacity);
    else if constexpr (maskMode == MASK420_TOPLEFT)
      fill_mask420_topleft_float_avx2<full_opacity>(dst, maskp, mask_pitch, width, opacity);
    else if constexpr (maskMode == MASK411)
      fill_mask411_float_avx2<full_opacity>(dst, maskp, width, opacity);
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

// prepare_effective_mask_for_row_avx2
#define INST_PREP_AVX2(mm, pt) \
  template const pt* prepare_effective_mask_for_row_avx2<mm, pt, true> (const pt*, int, int, std::vector<pt>&, int, int, MagicDiv); \
  template const pt* prepare_effective_mask_for_row_avx2<mm, pt, false>(const pt*, int, int, std::vector<pt>&, int, int, MagicDiv);
INST_PREP_AVX2(MASK444,          uint8_t)   INST_PREP_AVX2(MASK444,          uint16_t)
INST_PREP_AVX2(MASK420,          uint8_t)   INST_PREP_AVX2(MASK420,          uint16_t)
INST_PREP_AVX2(MASK420_MPEG2,    uint8_t)   INST_PREP_AVX2(MASK420_MPEG2,    uint16_t)
INST_PREP_AVX2(MASK420_TOPLEFT,  uint8_t)   INST_PREP_AVX2(MASK420_TOPLEFT,  uint16_t)
INST_PREP_AVX2(MASK422,          uint8_t)   INST_PREP_AVX2(MASK422,          uint16_t)
INST_PREP_AVX2(MASK422_MPEG2,    uint8_t)   INST_PREP_AVX2(MASK422_MPEG2,    uint16_t)
INST_PREP_AVX2(MASK422_TOPLEFT,  uint8_t)   INST_PREP_AVX2(MASK422_TOPLEFT,  uint16_t)
INST_PREP_AVX2(MASK411,          uint8_t)   INST_PREP_AVX2(MASK411,          uint16_t)
INST_PREP_AVX2(MASK411_TOPLEFT,  uint8_t)   INST_PREP_AVX2(MASK411_TOPLEFT,  uint16_t)
INST_PREP_AVX2(MASK440,          uint8_t)   INST_PREP_AVX2(MASK440,          uint16_t)
INST_PREP_AVX2(MASK440_TOPLEFT,  uint8_t)   INST_PREP_AVX2(MASK440_TOPLEFT,  uint16_t)
INST_PREP_AVX2(MASK410,          uint8_t)   INST_PREP_AVX2(MASK410,          uint16_t)
INST_PREP_AVX2(MASK410_TOPLEFT,  uint8_t)   INST_PREP_AVX2(MASK410_TOPLEFT,  uint16_t)
#undef INST_PREP_AVX2

// prepare_effective_mask_for_row_avx2
#define INST_PREP_AVX2(mm) \
  template const float* prepare_effective_mask_for_row_float_avx2<mm, true> (const float*, int, int, std::vector<float>&, float); \
  template const float* prepare_effective_mask_for_row_float_avx2<mm, false>(const float*, int, int, std::vector<float>&, float);
INST_PREP_AVX2(MASK444)
INST_PREP_AVX2(MASK420)
INST_PREP_AVX2(MASK420_MPEG2)
INST_PREP_AVX2(MASK420_TOPLEFT)
INST_PREP_AVX2(MASK422)
INST_PREP_AVX2(MASK422_MPEG2)
INST_PREP_AVX2(MASK422_TOPLEFT)
INST_PREP_AVX2(MASK411)
INST_PREP_AVX2(MASK411_TOPLEFT)
INST_PREP_AVX2(MASK440)
INST_PREP_AVX2(MASK440_TOPLEFT)
INST_PREP_AVX2(MASK410)
INST_PREP_AVX2(MASK410_TOPLEFT)
#undef INST_PREP_AVX2

