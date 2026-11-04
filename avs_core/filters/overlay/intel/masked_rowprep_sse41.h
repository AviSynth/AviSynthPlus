// masked_rowprep_sse41.h
// SSE4.1 rowprep function declarations.
// Implementations and explicit instantiations are in masked_rowprep_sse41.cpp.
//
// simd_magic_div_32 is kept inline here — used inside masked_merge_sse41_impl.hpp
// hot loops and must remain inlineable within those TUs.

#pragma once

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <smmintrin.h>   // SSE4.1 (includes SSSE3/SSE2 transitively)
#endif

#include "../blend_common.h"   // MaskMode, MagicDiv, magic_div_rt, AVS_FORCEINLINE
#include <vector>
#include <cstdint>

// ---------------------------------------------------------------------------
// simd_magic_div_32
// Inline here: used by masked_merge_sse41_impl.hpp in inner blend loops.
// Pure SSE2 arithmetic; __target__("sse4.1") only for _mm_blend_ps.
// ---------------------------------------------------------------------------
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static AVS_FORCEINLINE __m128i simd_magic_div_32(__m128i val, uint32_t magic, int shift) {
  __m128i v_magic = _mm_set1_epi64x(magic);
  __m128i res_02 = _mm_mul_epu32(val, v_magic);
  __m128i res_13 = _mm_mul_epu32(_mm_srli_si128(val, 4), v_magic);
  res_02 = _mm_srli_epi64(res_02, 32 + shift);
  res_13 = _mm_srli_epi64(res_13, 32 + shift);
  return _mm_castps_si128(_mm_blend_ps(
    _mm_castsi128_ps(res_02),
    _mm_castsi128_ps(_mm_slli_epi64(res_13, 32)),
    10 // 0b1010
  ));
}

// ---------------------------------------------------------------------------
// prepare_effective_mask_for_row_sse41
// Declared here; defined + explicitly instantiated in masked_rowprep_sse41.cpp.
// ---------------------------------------------------------------------------
template<MaskMode maskMode, typename pixel_t, bool full_opacity = true>
const pixel_t* prepare_effective_mask_for_row_sse41(
  const pixel_t* maskp,
  int mask_pitch,
  int width,
  pixel_t* buf,
  int opacity_i = 0,
  int half = 0,
  MagicDiv magic = {});

template<MaskMode maskMode, bool full_opacity = true>
const float* prepare_effective_mask_for_row_float_sse41(
  const float* maskp,
  int mask_pitch,
  int width,
  float* buf,
  float opacity = 0);
