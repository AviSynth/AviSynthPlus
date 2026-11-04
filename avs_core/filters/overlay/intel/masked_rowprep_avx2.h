// masked_rowprep_avx2.h
// AVX2 rowprep function declarations.
// Implementations and explicit instantiations are in masked_rowprep_avx2.cpp.
//
// No AVX2 intrinsics here — safe to include from any compilation unit.
// AVX2-compiled TUs that need simd_magic_div_32_avx2 inline should include
// masked_rowprep_avx2_impl.h instead.

#pragma once

#include "../blend_common.h"
#include <vector>
#include <cstdint>

template<MaskMode maskMode, typename pixel_t, bool full_opacity = true>
const pixel_t* prepare_effective_mask_for_row_avx2(
  const pixel_t* maskp,
  int mask_pitch,
  int width,
  pixel_t* buf,
  int opacity_i = 0,
  int half = 0,
  MagicDiv magic = {});

template<MaskMode maskMode, bool full_opacity = true>
const float* prepare_effective_mask_for_row_float_avx2(
  const float* maskp,
  int mask_pitch,
  int width,
  float* buf,
  float opacity = 0.0f);
