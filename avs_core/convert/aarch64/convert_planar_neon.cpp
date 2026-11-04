// AviSynth+.  Copyright 2026 AviSynth+ Project
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

// Legacy direct 8 bit YV24 -> packed RGB24/RGB32, the NEON pair of convert_yv24_to_rgb_c.
// Same arithmetic as C, bit-identical:
//   b = (y_b*Y + u_b*U + v_b*V + 4096 + (offset_rgb << 13)) >> 13, then clip to 0..255
//   with Y = y + offset_y, U = u - 128, V = v - 128.
// Coefficients and inputs fit int16, the products are summed in int32 (vmlal by lane).
// >> 13 is arithmetic like the C int shift; vqmovun_s32 + vqmovn_u16 clip like PixelClip.
// Register pressure (16 pixels per step, vst3/vst4 need 3-4 consecutive registers): the 9 coefficients are kept
// in 2 registers and used by lane, and the input offsets are folded into one rounder per channel:
//   c_y*(y + offset_y) + c_u*(u - 128) + c_v*(v - 128) + r == c_y*y + c_u*u + c_v*v + (r + c_y*offset_y - 128*(c_u + c_v))
// exact in int32, so the result stays bit-identical to C.

#include "convert_planar_neon.h"
#include <avs/config.h>
#include <cstdint>

#include <arm_neon.h>

namespace {

// one output channel for 8 pixels; Y, U, V are the raw 0..255 inputs (offsets are in the rounder).
// Coefficient lanes: k8 = { y_b, u_b, v_b, y_g, u_g, v_g, y_r, u_r }, k4 = { v_r, 0, 0, 0 }.
template<int LY, int LU, int LV, bool V_IN_K4>
static AVS_FORCEINLINE uint8x8_t channel8(int16x8_t Y, int16x8_t U, int16x8_t V,
  int16x8_t k8, int16x4_t k4, int32x4_t rounder)
{
  int32x4_t lo = vmlal_laneq_s16(rounder, vget_low_s16(Y), k8, LY);
  int32x4_t hi = vmlal_high_laneq_s16(rounder, Y, k8, LY);
  lo = vmlal_laneq_s16(lo, vget_low_s16(U), k8, LU);
  hi = vmlal_high_laneq_s16(hi, U, k8, LU);
  if constexpr (V_IN_K4) {
    lo = vmlal_lane_s16(lo, vget_low_s16(V), k4, LV);
    hi = vmlal_high_lane_s16(hi, V, k4, LV);
  }
  else {
    lo = vmlal_laneq_s16(lo, vget_low_s16(V), k8, LV);
    hi = vmlal_high_laneq_s16(hi, V, k8, LV);
  }
  lo = vshrq_n_s32(lo, 13);
  hi = vshrq_n_s32(hi, 13);
  const uint16x8_t u16 = vcombine_u16(vqmovun_s32(lo), vqmovun_s32(hi)); // < 0 -> 0
  return vqmovn_u16(u16); // > 255 -> 255
}

static AVS_FORCEINLINE uint8_t clip8(int v) { return static_cast<uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v)); }

// Scalar tail (at most 15 pixels), kept scalar on purpose: the compilers auto-vectorized it into a large
// loop that is never reached (less than 16 pixels) and has register spills. Its own function, because GCC
// before 14 has no loop pragma for this, only the function attribute.
template<int pixel_step, bool source_has_alpha>
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
static void yv24_to_rgb_tail(BYTE* dstp, const BYTE* srcY, const BYTE* srcU, const BYTE* srcV, const BYTE* srcA,
  size_t x, size_t width, const ConversionMatrix& matrix, int rounder_i)
{
#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
  for (; x < width; x++) {
    const int Y = srcY[x] + matrix.offset_y;
    const int U = srcU[x] - 128;
    const int V = srcV[x] - 128;
    dstp[x * pixel_step + 0] = clip8((matrix.y_b * Y + matrix.u_b * U + matrix.v_b * V + rounder_i) >> 13);
    dstp[x * pixel_step + 1] = clip8((matrix.y_g * Y + matrix.u_g * U + matrix.v_g * V + rounder_i) >> 13);
    dstp[x * pixel_step + 2] = clip8((matrix.y_r * Y + matrix.u_r * U + matrix.v_r * V + rounder_i) >> 13);
    if constexpr (pixel_step == 4)
      dstp[x * 4 + 3] = source_has_alpha ? srcA[x] : 255;
  }
}

} // namespace

template<int pixel_step, bool source_has_alpha>
void convert_yv24_to_rgb_neon(BYTE* dstp, const BYTE* srcY, const BYTE* srcU, const BYTE* srcV, const BYTE* srcA, size_t dst_pitch, size_t src_pitch_y, size_t src_pitch_uv, size_t src_pitch_a, size_t width, size_t height, const ConversionMatrix& matrix)
{
  // The 13 bit coefficients (below 2.2 * 8192) and offset_y are generated in range: int16 is enough.
  const int rounder_i = 4096 + (matrix.offset_rgb << 13);
  const int16_t kc[8] = {
    static_cast<int16_t>(matrix.y_b), static_cast<int16_t>(matrix.u_b), static_cast<int16_t>(matrix.v_b),
    static_cast<int16_t>(matrix.y_g), static_cast<int16_t>(matrix.u_g), static_cast<int16_t>(matrix.v_g),
    static_cast<int16_t>(matrix.y_r), static_cast<int16_t>(matrix.u_r) };
  const int16x8_t k8 = vld1q_s16(kc);
  const int16x4_t k4 = vdup_n_s16(static_cast<int16_t>(matrix.v_r));
  // input offsets folded in: + c_y * offset_y - 128 * (c_u + c_v)
  const int32x4_t rounder_b = vdupq_n_s32(rounder_i + matrix.y_b * matrix.offset_y - 128 * (matrix.u_b + matrix.v_b));
  const int32x4_t rounder_g = vdupq_n_s32(rounder_i + matrix.y_g * matrix.offset_y - 128 * (matrix.u_g + matrix.v_g));
  const int32x4_t rounder_r = vdupq_n_s32(rounder_i + matrix.y_r * matrix.offset_y - 128 * (matrix.u_r + matrix.v_r));
  const uint8x16_t alpha_max = vdupq_n_u8(255);

  const size_t wmod16 = width / 16 * 16;
  dstp += dst_pitch * (height - 1); // packed RGB is bottom-up
  for (size_t y = 0; y < height; y++) {
    size_t x = 0;
    for (; x < wmod16; x += 16) {
      const uint8x16_t y8 = vld1q_u8(srcY + x);
      const uint8x16_t u8 = vld1q_u8(srcU + x);
      const uint8x16_t v8 = vld1q_u8(srcV + x);
      uint8x8_t b[2], g[2], r[2];
      for (int h = 0; h < 2; h++) {
        const uint8x8_t yh = h == 0 ? vget_low_u8(y8) : vget_high_u8(y8);
        const uint8x8_t uh = h == 0 ? vget_low_u8(u8) : vget_high_u8(u8);
        const uint8x8_t vh = h == 0 ? vget_low_u8(v8) : vget_high_u8(v8);
        const int16x8_t Y = vreinterpretq_s16_u16(vmovl_u8(yh));
        const int16x8_t U = vreinterpretq_s16_u16(vmovl_u8(uh));
        const int16x8_t V = vreinterpretq_s16_u16(vmovl_u8(vh));
        b[h] = channel8<0, 1, 2, false>(Y, U, V, k8, k4, rounder_b);
        g[h] = channel8<3, 4, 5, false>(Y, U, V, k8, k4, rounder_g);
        r[h] = channel8<6, 7, 0, true>(Y, U, V, k8, k4, rounder_r);
      }
      if constexpr (pixel_step == 4) {
        uint8x16x4_t bgra;
        bgra.val[0] = vcombine_u8(b[0], b[1]);
        bgra.val[1] = vcombine_u8(g[0], g[1]);
        bgra.val[2] = vcombine_u8(r[0], r[1]);
        bgra.val[3] = source_has_alpha ? vld1q_u8(srcA + x) : alpha_max;
        vst4q_u8(dstp + x * 4, bgra);
      }
      else {
        uint8x16x3_t bgr;
        bgr.val[0] = vcombine_u8(b[0], b[1]);
        bgr.val[1] = vcombine_u8(g[0], g[1]);
        bgr.val[2] = vcombine_u8(r[0], r[1]);
        vst3q_u8(dstp + x * 3, bgr);
      }
    }
    yv24_to_rgb_tail<pixel_step, source_has_alpha>(dstp, srcY, srcU, srcV, srcA, x, width, matrix, rounder_i);
    dstp -= dst_pitch;
    srcY += src_pitch_y;
    srcU += src_pitch_uv;
    srcV += src_pitch_uv;
    if constexpr (source_has_alpha)
      srcA += src_pitch_a;
  }
}

template void convert_yv24_to_rgb_neon<3, false>(BYTE* dstp, const BYTE* srcY, const BYTE* srcU, const BYTE* srcV, const BYTE* srcA, size_t dst_pitch, size_t src_pitch_y, size_t src_pitch_uv, size_t src_pitch_a, size_t width, size_t height, const ConversionMatrix& matrix);
template void convert_yv24_to_rgb_neon<4, false>(BYTE* dstp, const BYTE* srcY, const BYTE* srcU, const BYTE* srcV, const BYTE* srcA, size_t dst_pitch, size_t src_pitch_y, size_t src_pitch_uv, size_t src_pitch_a, size_t width, size_t height, const ConversionMatrix& matrix);
template void convert_yv24_to_rgb_neon<4, true>(BYTE* dstp, const BYTE* srcY, const BYTE* srcU, const BYTE* srcV, const BYTE* srcA, size_t dst_pitch, size_t src_pitch_y, size_t src_pitch_uv, size_t src_pitch_a, size_t width, size_t height, const ConversionMatrix& matrix);
