// AviSynth+.  Copyright 2026- AviSynth+ Project
// https://avs-plus.net
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

// Integer mask row preparation for subsampled chroma: the NEON pair of prepare_effective_mask_for_row
// (blend_common.h) and prepare_effective_mask_for_row_sse41. Same arithmetic as C, bit-identical:
//   center box averages: (sum + count/2) / count, count = 2, 4 or 16 -> rounding shifts
//     (vrhadd for 2 samples, vrshrn for 4 and 16)
//   MPEG2 triangle filters: (left + 2*mid + right + 2) >> 2 (4:2:2), (... + 4) >> 3 (4:2:0, 2-row sums),
//     left = the previous odd sample (sliding window), vext from the previous vector
//   top-left point samples: the first sample of the block (vld2/vld4 lane 0)
//   !full_opacity: magic_div_rt(avg * opacity_i + half), 32 bit (8 bit pixels) or 64 bit (16 bit) intermediate
// 8 outputs per step for both pixel sizes; the rest of the row is done per pixel with the C kernel.
// Unlike Intel (no SIMD for 4:1:0, 4:4:0 and 4:1:1 top-left), all 13 modes have a NEON kernel.

#include "masked_rowprep_neon.h"
#include <arm_neon.h>
#include <cstdint>

namespace {

// 8 averaged mask values in 16 bit lanes for chroma x .. x+7.
// prev: MPEG2 sliding window, the previous odd sample (4:2:2) or 2-row sum (4:2:0), all lanes; only lane 7 is used.
template<MaskMode maskMode>
static AVS_FORCEINLINE uint16x8_t avg8_u8(const uint8_t* row0, int pitch, int x, uint16x8_t& prev)
{
  if constexpr (maskMode == MASK422) {
    const uint8x8x2_t v = vld2_u8(row0 + x * 2);
    return vmovl_u8(vrhadd_u8(v.val[0], v.val[1]));
  }
  else if constexpr (maskMode == MASK422_TOPLEFT || maskMode == MASK420_TOPLEFT) {
    return vmovl_u8(vld2_u8(row0 + x * 2).val[0]);
  }
  else if constexpr (maskMode == MASK422_MPEG2) {
    const uint8x8x2_t v = vld2_u8(row0 + x * 2);
    const uint16x8_t odd = vmovl_u8(v.val[1]);
    const uint16x8_t left = vextq_u16(prev, odd, 7);
    prev = odd;
    // (left + 2*mid + right + 2) >> 2, max 4*255
    return vrshrq_n_u16(vaddq_u16(vaddq_u16(left, odd), vshll_n_u8(v.val[0], 1)), 2);
  }
  else if constexpr (maskMode == MASK420) {
    const uint16x8_t s = vpadalq_u8(vpaddlq_u8(vld1q_u8(row0 + x * 2)), vld1q_u8(row0 + pitch + x * 2));
    return vrshrq_n_u16(s, 2);
  }
  else if constexpr (maskMode == MASK420_MPEG2) {
    const uint8x8x2_t v0 = vld2_u8(row0 + x * 2);
    const uint8x8x2_t v1 = vld2_u8(row0 + pitch + x * 2);
    const uint16x8_t even = vaddl_u8(v0.val[0], v1.val[0]);
    const uint16x8_t odd = vaddl_u8(v0.val[1], v1.val[1]);
    const uint16x8_t left = vextq_u16(prev, odd, 7);
    prev = odd;
    // (left + 2*mid + right + 4) >> 3, max 4*510
    return vrshrq_n_u16(vaddq_u16(vaddq_u16(left, odd), vshlq_n_u16(even, 1)), 3);
  }
  else if constexpr (maskMode == MASK411) {
    const uint8x8x4_t v = vld4_u8(row0 + x * 4);
    return vrshrq_n_u16(vaddq_u16(vaddl_u8(v.val[0], v.val[1]), vaddl_u8(v.val[2], v.val[3])), 2);
  }
  else if constexpr (maskMode == MASK411_TOPLEFT || maskMode == MASK410_TOPLEFT) {
    return vmovl_u8(vld4_u8(row0 + x * 4).val[0]);
  }
  else if constexpr (maskMode == MASK440) {
    return vmovl_u8(vrhadd_u8(vld1_u8(row0 + x), vld1_u8(row0 + pitch + x)));
  }
  else if constexpr (maskMode == MASK440_TOPLEFT) {
    return vmovl_u8(vld1_u8(row0 + x));
  }
  else { // MASK410: 4x4 box, max 16*255
    uint16x8_t s = vdupq_n_u16(0);
    for (int r = 0; r < 4; r++) {
      const uint8x8x4_t v = vld4_u8(row0 + r * pitch + x * 4);
      s = vaddq_u16(s, vaddq_u16(vaddl_u8(v.val[0], v.val[1]), vaddl_u8(v.val[2], v.val[3])));
    }
    return vrshrq_n_u16(s, 4);
  }
}

// 16 bit pixels: sums in 32 bit lanes where they can exceed 16 bits.
// prev_lo/hi: MPEG2 sliding window (4:2:0: 2-row sums up to 2*65535), only lane 3 of prev_hi is used.
template<MaskMode maskMode>
static AVS_FORCEINLINE uint16x8_t avg8_u16(const uint16_t* row0, int pitch, int x, uint32x4_t& prev_hi)
{
  if constexpr (maskMode == MASK422) {
    const uint16x8x2_t v = vld2q_u16(row0 + x * 2);
    return vrhaddq_u16(v.val[0], v.val[1]);
  }
  else if constexpr (maskMode == MASK422_TOPLEFT || maskMode == MASK420_TOPLEFT) {
    return vld2q_u16(row0 + x * 2).val[0];
  }
  else if constexpr (maskMode == MASK422_MPEG2 || maskMode == MASK420_MPEG2) {
    uint32x4_t even_lo, even_hi, odd_lo, odd_hi;
    const uint16x8x2_t v0 = vld2q_u16(row0 + x * 2);
    if constexpr (maskMode == MASK422_MPEG2) {
      even_lo = vmovl_u16(vget_low_u16(v0.val[0])); even_hi = vmovl_high_u16(v0.val[0]);
      odd_lo = vmovl_u16(vget_low_u16(v0.val[1]));  odd_hi = vmovl_high_u16(v0.val[1]);
    }
    else {
      const uint16x8x2_t v1 = vld2q_u16(row0 + pitch + x * 2);
      even_lo = vaddl_u16(vget_low_u16(v0.val[0]), vget_low_u16(v1.val[0])); even_hi = vaddl_high_u16(v0.val[0], v1.val[0]);
      odd_lo = vaddl_u16(vget_low_u16(v0.val[1]), vget_low_u16(v1.val[1]));  odd_hi = vaddl_high_u16(v0.val[1], v1.val[1]);
    }
    const uint32x4_t left_lo = vextq_u32(prev_hi, odd_lo, 3);
    const uint32x4_t left_hi = vextq_u32(odd_lo, odd_hi, 3);
    prev_hi = odd_hi;
    const uint32x4_t s_lo = vaddq_u32(vaddq_u32(left_lo, odd_lo), vshlq_n_u32(even_lo, 1));
    const uint32x4_t s_hi = vaddq_u32(vaddq_u32(left_hi, odd_hi), vshlq_n_u32(even_hi, 1));
    // 4:2:2: (... + 2) >> 2; 4:2:0: (... + 4) >> 3
    constexpr int sh = maskMode == MASK422_MPEG2 ? 2 : 3;
    return vcombine_u16(vrshrn_n_u32(s_lo, sh), vrshrn_n_u32(s_hi, sh));
  }
  else if constexpr (maskMode == MASK420) {
    const uint32x4_t s_lo = vpadalq_u16(vpaddlq_u16(vld1q_u16(row0 + x * 2)), vld1q_u16(row0 + pitch + x * 2));
    const uint32x4_t s_hi = vpadalq_u16(vpaddlq_u16(vld1q_u16(row0 + x * 2 + 8)), vld1q_u16(row0 + pitch + x * 2 + 8));
    return vcombine_u16(vrshrn_n_u32(s_lo, 2), vrshrn_n_u32(s_hi, 2));
  }
  else if constexpr (maskMode == MASK411 || maskMode == MASK410) {
    uint32x4_t s_lo = vdupq_n_u32(0), s_hi = vdupq_n_u32(0);
    constexpr int rows = maskMode == MASK410 ? 4 : 1;
    for (int r = 0; r < rows; r++) {
      const uint16x8x4_t v = vld4q_u16(row0 + r * pitch + x * 4);
      s_lo = vaddq_u32(s_lo, vaddq_u32(vaddl_u16(vget_low_u16(v.val[0]), vget_low_u16(v.val[1])),
                                       vaddl_u16(vget_low_u16(v.val[2]), vget_low_u16(v.val[3]))));
      s_hi = vaddq_u32(s_hi, vaddq_u32(vaddl_high_u16(v.val[0], v.val[1]), vaddl_high_u16(v.val[2], v.val[3])));
    }
    constexpr int sh = maskMode == MASK410 ? 4 : 2; // 16 or 4 samples
    return vcombine_u16(vrshrn_n_u32(s_lo, sh), vrshrn_n_u32(s_hi, sh));
  }
  else if constexpr (maskMode == MASK411_TOPLEFT || maskMode == MASK410_TOPLEFT) {
    return vld4q_u16(row0 + x * 4).val[0];
  }
  else if constexpr (maskMode == MASK440) {
    return vrhaddq_u16(vld1q_u16(row0 + x), vld1q_u16(row0 + pitch + x));
  }
  else { // MASK440_TOPLEFT
    return vld1q_u16(row0 + x);
  }
}

// magic_div_rt(avg * opacity_i + half), 8 values
template<typename pixel_t>
static AVS_FORCEINLINE uint16x8_t bake_opacity8(uint16x8_t avg, int opacity_i, int half, MagicDiv magic)
{
  if constexpr (sizeof(pixel_t) == 1) {
    // tmp <= 255*255 + 127 fits 16 bits; (tmp * div) >> (16 + shift) in 32 bit
    const uint16x8_t tmp = vmlaq_n_u16(vdupq_n_u16((uint16_t)half), avg, (uint16_t)opacity_i);
    const uint16x4_t div = vdup_n_u16((uint16_t)magic.div);
    const int32x4_t sh = vdupq_n_s32(-(16 + magic.shift));
    const uint32x4_t lo = vshlq_u32(vmull_u16(vget_low_u16(tmp), div), sh);
    const uint32x4_t hi = vshlq_u32(vmull_u16(vget_high_u16(tmp), div), sh);
    return vcombine_u16(vmovn_u32(lo), vmovn_u32(hi));
  }
  else {
    // tmp <= 65535*65535 + 32767 fits 32 bits; (tmp * div) >> (32 + shift) in 64 bit
    const uint32x4_t halfv = vdupq_n_u32((uint32_t)half);
    const uint32x4_t tmp_lo = vmlaq_n_u32(halfv, vmovl_u16(vget_low_u16(avg)), (uint32_t)opacity_i);
    const uint32x4_t tmp_hi = vmlaq_n_u32(halfv, vmovl_high_u16(avg), (uint32_t)opacity_i);
    const uint32x2_t div = vdup_n_u32(magic.div);
    const int64x2_t sh = vdupq_n_s64(-(32 + magic.shift));
    auto md = [&](uint32x4_t t) {
      const uint64x2_t a = vshlq_u64(vmull_u32(vget_low_u32(t), div), sh);
      const uint64x2_t b = vshlq_u64(vmull_high_u32(t, vcombine_u32(div, div)), sh);
      return vcombine_u32(vmovn_u64(a), vmovn_u64(b));
    };
    return vcombine_u16(vmovn_u32(md(tmp_lo)), vmovn_u32(md(tmp_hi)));
  }
}

template<typename pixel_t>
static AVS_FORCEINLINE void store8(pixel_t* dst, uint16x8_t v)
{
  if constexpr (sizeof(pixel_t) == 1)
    vst1_u8(dst, vmovn_u16(v)); // <= 255
  else
    vst1q_u16(dst, v);
}

} // namespace

template<MaskMode maskMode, typename pixel_t, bool full_opacity>
const pixel_t* prepare_effective_mask_for_row_neon(
  const pixel_t* maskp,
  int mask_pitch,
  int width,
  pixel_t* buf,
  int opacity_i,
  int half,
  MagicDiv magic)
{
  if constexpr (maskMode == MASK444 && full_opacity) {
    return maskp;
  }
  else {
    pixel_t* dst = buf;
    int x = 0;
    if constexpr (maskMode == MASK444) {
      for (; x + 8 <= width; x += 8) {
        uint16x8_t v;
        if constexpr (sizeof(pixel_t) == 1) v = vmovl_u8(vld1_u8(maskp + x));
        else v = vld1q_u16(maskp + x);
        store8(dst + x, bake_opacity8<pixel_t>(v, opacity_i, half, magic));
      }
      for (; x < width; ++x)
        dst[x] = static_cast<pixel_t>(magic_div_rt<pixel_t>((uint32_t)maskp[x] * (uint32_t)opacity_i + (uint32_t)half, magic));
      return dst;
    }
    else {
      // MPEG2 sliding window start: the left neighbour of chroma x = 0 is the first sample (like C)
      [[maybe_unused]] uint16x8_t prev8 = vdupq_n_u16(0);
      [[maybe_unused]] uint32x4_t prev16 = vdupq_n_u32(0);
      if constexpr (maskMode == MASK422_MPEG2) {
        prev8 = vdupq_n_u16(maskp[0]);
        prev16 = vdupq_n_u32(maskp[0]);
      }
      else if constexpr (maskMode == MASK420_MPEG2) {
        prev8 = vdupq_n_u16((uint16_t)(maskp[0] + maskp[mask_pitch]));
        prev16 = vdupq_n_u32((uint32_t)maskp[0] + maskp[mask_pitch]);
      }
      for (; x + 8 <= width; x += 8) {
        uint16x8_t avg;
        if constexpr (sizeof(pixel_t) == 1)
          avg = avg8_u8<maskMode>(maskp, mask_pitch, x, prev8);
        else
          avg = avg8_u16<maskMode>(maskp, mask_pitch, x, prev16);
        if constexpr (!full_opacity)
          avg = bake_opacity8<pixel_t>(avg, opacity_i, half, magic);
        store8(dst + x, avg);
      }
      // rest: the C kernel; the MPEG2 window continues from the last odd sample
      int mask_right = 0;
      if constexpr (maskMode == MASK420_MPEG2)
        mask_right = x == 0 ? maskp[0] + maskp[mask_pitch] : maskp[x * 2 - 1] + maskp[x * 2 - 1 + mask_pitch];
      else if constexpr (maskMode == MASK422_MPEG2)
        mask_right = x == 0 ? maskp[0] : maskp[x * 2 - 1];
      for (; x < width; ++x) {
        const pixel_t avg = (pixel_t)calculate_effective_mask<maskMode>(maskp, x, mask_pitch, mask_right);
        if constexpr (full_opacity)
          dst[x] = avg;
        else
          dst[x] = static_cast<pixel_t>(magic_div_rt<pixel_t>((uint32_t)avg * (uint32_t)opacity_i + (uint32_t)half, magic));
      }
      return dst;
    }
  }
}

#define INST_PREP_NEON(mm, pt) \
  template const pt* prepare_effective_mask_for_row_neon<mm, pt, true> (const pt*, int, int, pt*, int, int, MagicDiv); \
  template const pt* prepare_effective_mask_for_row_neon<mm, pt, false>(const pt*, int, int, pt*, int, int, MagicDiv);
INST_PREP_NEON(MASK444,          uint8_t)   INST_PREP_NEON(MASK444,          uint16_t)
INST_PREP_NEON(MASK420,          uint8_t)   INST_PREP_NEON(MASK420,          uint16_t)
INST_PREP_NEON(MASK420_MPEG2,    uint8_t)   INST_PREP_NEON(MASK420_MPEG2,    uint16_t)
INST_PREP_NEON(MASK420_TOPLEFT,  uint8_t)   INST_PREP_NEON(MASK420_TOPLEFT,  uint16_t)
INST_PREP_NEON(MASK422,          uint8_t)   INST_PREP_NEON(MASK422,          uint16_t)
INST_PREP_NEON(MASK422_MPEG2,    uint8_t)   INST_PREP_NEON(MASK422_MPEG2,    uint16_t)
INST_PREP_NEON(MASK422_TOPLEFT,  uint8_t)   INST_PREP_NEON(MASK422_TOPLEFT,  uint16_t)
INST_PREP_NEON(MASK411,          uint8_t)   INST_PREP_NEON(MASK411,          uint16_t)
INST_PREP_NEON(MASK411_TOPLEFT,  uint8_t)   INST_PREP_NEON(MASK411_TOPLEFT,  uint16_t)
INST_PREP_NEON(MASK440,          uint8_t)   INST_PREP_NEON(MASK440,          uint16_t)
INST_PREP_NEON(MASK440_TOPLEFT,  uint8_t)   INST_PREP_NEON(MASK440_TOPLEFT,  uint16_t)
INST_PREP_NEON(MASK410,          uint8_t)   INST_PREP_NEON(MASK410,          uint16_t)
INST_PREP_NEON(MASK410_TOPLEFT,  uint8_t)   INST_PREP_NEON(MASK410_TOPLEFT,  uint16_t)
#undef INST_PREP_NEON
