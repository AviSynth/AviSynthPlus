// AviSynth+.  Copyright 2026- AviSynth+ Project
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

#ifndef __Masked_Rowprep_Sep_NEON_H__
#define __Masked_Rowprep_Sep_NEON_H__

// Separable mask kernels, NEON: MASK420_TOPLEFT, MASK440(_TOPLEFT), MASK411(_TOPLEFT), MASK410(_MPEG2/_TOPLEFT).
// See MaskMode / mask_sep_sum in blend_common.h. The same box cascades (mask_filter_cascade) on vectors, the
// same additions in the same order as the C version: integer AND float results are bit-identical to C.
// Per chunk: vertical pass into a column-sum buffer (edge columns replicated, = the C column clamping), then
// the horizontal pass, 4 outputs per vector: H=1 direct, H=2 vld2q (even/odd), H=4 vld4q (one phase per
// luma offset). Included by the NEON row preparation TUs; internal linkage.

#include "../blend_common.h"
#include <arm_neon.h>
#include <algorithm>
#include <cstdint>
#include <cstring>

namespace {

struct SepNeonInt {
  using V = uint32x4_t;
  using acc_t = uint32_t;
  template<typename pixel_t>
  static AVS_FORCEINLINE V load4(const pixel_t* p) {
    if constexpr (sizeof(pixel_t) == 1) {
      uint32_t t;
      memcpy(&t, p, 4);
      return vmovl_u16(vget_low_u16(vmovl_u8(vreinterpret_u8_u32(vdup_n_u32(t)))));
    }
    else
      return vmovl_u16(vld1_u16(p));
  }
  static AVS_FORCEINLINE V add(V a, V b) { return vaddq_u32(a, b); }
  static AVS_FORCEINLINE V loadu(const acc_t* p) { return vld1q_u32(p); }
  static AVS_FORCEINLINE void storeu(acc_t* p, V v) { vst1q_u32(p, v); }
  static AVS_FORCEINLINE uint32x4x2_t load2(const acc_t* p) { return vld2q_u32(p); }
  static AVS_FORCEINLINE uint32x4x4_t load4ph(const acc_t* p) { return vld4q_u32(p); }
};

struct SepNeonFloat {
  using V = float32x4_t;
  using acc_t = float;
  template<typename pixel_t>
  static AVS_FORCEINLINE V load4(const pixel_t* p) { return vld1q_f32(p); }
  static AVS_FORCEINLINE V add(V a, V b) { return vaddq_f32(a, b); }
  static AVS_FORCEINLINE V loadu(const acc_t* p) { return vld1q_f32(p); }
  static AVS_FORCEINLINE void storeu(acc_t* p, V v) { vst1q_f32(p, v); }
  static AVS_FORCEINLINE float32x4x2_t load2(const acc_t* p) { return vld2q_f32(p); }
  static AVS_FORCEINLINE float32x4x4_t load4ph(const acc_t* p) { return vld4q_f32(p); }
};

// The weighted sums of output pixels x .. x+3; colsum[i] = column sum of luma column colbase + i.
template<MaskMode maskMode, typename Ops>
static AVS_FORCEINLINE typename Ops::V mask_sep_hsum4_neon(const typename Ops::acc_t* colsum, int colbase, int x) {
  using V = typename Ops::V;
  constexpr MaskFilter HF = MaskHFilter<maskMode>;
  constexpr int H = MaskHSubsample<maskMode>;
  auto add = [](V a, V b) { return Ops::add(a, b); };
  if constexpr (H == 1) {
    return Ops::loadu(colsum + (x - colbase));
  }
  else if constexpr (H == 2) { // MF_121: columns 2x-1, 2x, 2x+1
    const auto cur = Ops::load2(colsum + (2 * x - colbase));      // val[0]: 2x, val[1]: 2x+1
    const auto prev = Ops::load2(colsum + (2 * x - 2 - colbase)); // val[1]: 2x-1
    const V v[3] = { prev.val[1], cur.val[0], cur.val[1] };
    return mask_filter_cascade<HF, V>(v, add);
  }
  else { // H == 4: MF_TENT7 (offsets -3..3) or MF_TRI8 (-2..5): phases of the blocks -1, 0, +1
    decltype(Ops::load4ph(colsum)) ph[3];
    ph[0] = Ops::load4ph(colsum + (4 * x - 4 - colbase));
    ph[1] = Ops::load4ph(colsum + (4 * x - colbase));
    if constexpr (HF == MF_TRI8)
      ph[2] = Ops::load4ph(colsum + (4 * x + 4 - colbase));
    V v[8];
    for (int k = 0; k < MaskFilterTaps<HF>; k++) {
      const int o = MaskFilterFirst<HF> + k;     // -3..3 or -2..5
      const int s = o < 0 ? -1 : (o < 4 ? 0 : 1);
      v[k] = ph[s + 1].val[o - 4 * s];           // lane i: column 4(x+i) + o
    }
    return mask_filter_cascade<HF, V>(v, add);
  }
}

template<MaskMode maskMode, typename sample_t, typename Ops>
static void mask_sep_row_neon_impl(const sample_t* maskp, int mask_pitch, int width, int rows_above, int rows_below,
  typename Ops::acc_t* out_sums /* width+3 elements */)
{
  using V = typename Ops::V;
  using acc_t = typename Ops::acc_t;
  constexpr MaskFilter VF = MaskVFilter<maskMode>;
  constexpr int H = MaskHSubsample<maskMode>;
  constexpr int PADL = H == 4 ? 4 : H == 2 ? 2 : 0;   // leftmost column read: 4x-4 / 2x-2
  constexpr int PADR = H == 4 ? 12 : H == 2 ? 6 : 4;  // last column read beyond the chunk, incl. the 4-lane rounding
  constexpr int CHUNK = 64;                            // output pixels per chunk (multiple of 4)
  auto add = [](V a, V b) { return Ops::add(a, b); };
  auto adds = [](acc_t a, acc_t b) { return a + b; };

  const sample_t* rows[8];
  mask_sep_rows<maskMode, sample_t>(maskp, mask_pitch, rows_above, rows_below, rows);
  const int luma_width = width * H;
  acc_t colsum[CHUNK * H + PADL + PADR];

  for (int x0 = 0; x0 < width; x0 += CHUNK) {
    const int n = std::min(CHUNK, width - x0);
    const int colbase = x0 * H - PADL;
    const int ncols = ((n + 3) & ~3) * H + PADL + PADR;
    const int lo = std::max(colbase, 0);
    const int hi = std::min(colbase + ncols - 1, luma_width - 1);
    // 1. vertical pass over the columns inside the mask row
    int c = lo;
    if (hi - lo + 1 >= 4) {
      for (;; c += 4) {
        if (c > hi - 3) c = hi - 3; // last group overlaps the previous one, same values
        V v[8];
        for (int j = 0; j < MaskFilterTaps<VF>; j++)
          v[j] = Ops::template load4<sample_t>(rows[j] + c);
        Ops::storeu(colsum + (c - colbase), mask_filter_cascade<VF, V>(v, add));
        if (c == hi - 3) break;
      }
    }
    else {
      for (; c <= hi; c++) {
        acc_t v[8];
        for (int j = 0; j < MaskFilterTaps<VF>; j++)
          v[j] = (acc_t)rows[j][c];
        colsum[c - colbase] = mask_filter_cascade<VF, acc_t>(v, adds);
      }
    }
    // columns outside the mask row: replicated edge columns
    for (int i = 0; i < lo - colbase; i++)
      colsum[i] = colsum[lo - colbase];
    for (int i = hi - colbase + 1; i < ncols; i++)
      colsum[i] = colsum[hi - colbase];
    // 2. horizontal pass
    for (int x = x0; x < x0 + n; x += 4)
      Ops::storeu(out_sums + x, mask_sep_hsum4_neon<maskMode, Ops>(colsum, colbase, x));
  }
}

// Integer: weighted sums -> (sum + half) >> shift -> optional opacity (magic_div_rt) -> pixel_t.
template<MaskMode maskMode, typename pixel_t, bool full_opacity>
static void mask_sep_row_neon(const pixel_t* maskp, int mask_pitch, int width, int rows_above, int rows_below,
  pixel_t* dst, int opacity_i, int half, MagicDiv magic)
{
  constexpr int shift = MaskSepShift<maskMode>;
  ScratchBuffer<uint32_t> sums(width + 3);
  mask_sep_row_neon_impl<maskMode, pixel_t, SepNeonInt>(maskp, mask_pitch, width, rows_above, rows_below, sums.data());
  const uint32x4_t rounder = vdupq_n_u32((1u << shift) >> 1);
  const uint32x4_t v_half = vdupq_n_u32((uint32_t)half);
  int x = 0;
  for (; x <= width - 4; x += 4) {
    uint32x4_t avg = vshrq_n_u32(vaddq_u32(vld1q_u32(sums.data() + x), rounder), shift);
    if constexpr (!full_opacity) {
      const uint32x4_t t = vmlaq_n_u32(v_half, avg, (uint32_t)opacity_i); // avg * opacity + half
      if constexpr (sizeof(pixel_t) == 1) {
        // (t * div) >> (16 + shift), the product is below 2^32
        avg = vshlq_u32(vmulq_n_u32(t, magic.div), vdupq_n_s32(-(16 + (int)magic.shift)));
      }
      else {
        // (uint64)(t * div) >> (32 + shift)
        const int64x2_t sh = vdupq_n_s64(-(32 + (int)magic.shift));
        const uint64x2_t lo = vshlq_u64(vmull_n_u32(vget_low_u32(t), magic.div), sh);
        const uint64x2_t hi = vshlq_u64(vmull_high_n_u32(t, magic.div), sh);
        avg = vcombine_u32(vmovn_u64(lo), vmovn_u64(hi));
      }
    }
    const uint16x4_t p16 = vqmovn_u32(avg);
    if constexpr (sizeof(pixel_t) == 1) {
      const uint8x8_t p8 = vqmovn_u16(vcombine_u16(p16, p16));
      vst1_lane_u32(reinterpret_cast<uint32_t*>(dst + x), vreinterpret_u32_u8(p8), 0);
    }
    else
      vst1_u16(dst + x, p16);
  }
  for (; x < width; x++) {
    const int avg = (int)((sums[x] + ((1u << shift) >> 1)) >> shift);
    dst[x] = full_opacity ? (pixel_t)avg
      : (pixel_t)magic_div_rt<pixel_t>((uint32_t)avg * (uint32_t)opacity_i + (uint32_t)half, magic);
  }
}

// Float: weighted sums * 2^-shift -> optional opacity.
template<MaskMode maskMode, bool full_opacity>
static void mask_sep_row_float_neon(const float* maskp, int mask_pitch, int width, int rows_above, int rows_below,
  float* dst, float opacity)
{
  constexpr float scale = 1.0f / (float)(1 << MaskSepShift<maskMode>);
  ScratchBuffer<float> sums(width + 3);
  mask_sep_row_neon_impl<maskMode, float, SepNeonFloat>(maskp, mask_pitch, width, rows_above, rows_below, sums.data());
  int x = 0;
  for (; x <= width - 4; x += 4) {
    float32x4_t avg = vmulq_n_f32(vld1q_f32(sums.data() + x), scale);
    if constexpr (!full_opacity)
      avg = vmulq_n_f32(avg, opacity);
    vst1q_f32(dst + x, avg);
  }
  for (; x < width; x++) {
    const float avg = sums[x] * scale;
    dst[x] = full_opacity ? avg : avg * opacity;
  }
}

} // namespace

#endif // __Masked_Rowprep_Sep_NEON_H__
