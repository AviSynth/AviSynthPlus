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

// Horizontal resizer, planar 8-16 bit, generic (any kernel size): the NEON pair of
// resizer_h_c_generic_uint8_16_vectorized (resample.cpp) and resizer_h_avx2_generic_uint8_t / _uint16_t.
// Same structure and integer arithmetic, bit-identical:
//   out[x] = (sum(src[offset[x] + i] * coeff[x][i]) + rounder) >> FPScale, saturated (10-14 bit: clamped to max)
//   exact 16 bit: the pixels are pivoted to signed (v - 32768, wrapping int16 add like C/AVX2) and
//   32768 << FPScale16bits is added back after the sum (the coefficients sum to 1 << FPScale16bits).
// 8 output pixels per step, each a dot product over the coefficients:
//   vmlal_s16 (widening multiply-accumulate) instead of madd, 8 taps per step;
//   the 8 sums: two vpaddq_s32 rounds per 4 pixels, no cross-lane permute like at AVX2 hadd.
// Overread rules as in C/AVX2: up to safelimit the padded (aligned) filter_size is read, beyond it only the
// real kernel size (8 taps, then the rest per tap). Like C/AVX2, the last group of 8 may store up to 7 pixels
// beyond width, inside the aligned frame area.
// The C version compiles to scalar loads, multiplies and lane inserts on aarch64 (GCC 12), this is the reason for it.

#include "resample_neon.h"
#include <avs/config.h>
#include <cmath>
#include <cstdint>

#include <arm_neon.h>

// 8 pixels as int16: 8 bit zero-extended, 10-14 bit as is, 16 bit pivoted to signed
template<typename pixel_t, bool lessthan16bit>
static AVS_FORCEINLINE int16x8_t load8_s16(const pixel_t* p)
{
  if constexpr (sizeof(pixel_t) == 1)
    return vreinterpretq_s16_u16(vmovl_u8(vld1_u8(p)));
  else if constexpr (lessthan16bit)
    return vreinterpretq_s16_u16(vld1q_u16(p));
  else
    return vreinterpretq_s16_u16(vaddq_u16(vld1q_u16(p), vdupq_n_u16(0x8000))); // v - 32768 as int16
}

// dot product of one output pixel; 4 partial sums
template<bool safe_aligned_mode, typename pixel_t, bool lessthan16bit>
static AVS_FORCEINLINE int32x4_t dot_one_pixel(const pixel_t* AVS_RESTRICT src, const short* AVS_RESTRICT coeff, int filter_size, int kernel_size)
{
  int32x4_t acc0 = vdupq_n_s32(0);
  int32x4_t acc1 = vdupq_n_s32(0);
  // safe: the coefficients are padded to filter_size (multiple of 16) with zeros, reading that far is allowed
  const int ksmod8 = safe_aligned_mode ? filter_size : kernel_size / 8 * 8;
  int i = 0;
  for (; i < ksmod8; i += 8) {
    const int16x8_t d = load8_s16<pixel_t, lessthan16bit>(src + i);
    const int16x8_t c = vld1q_s16(coeff + i);
    acc0 = vmlal_s16(acc0, vget_low_s16(d), vget_low_s16(c));
    acc1 = vmlal_high_s16(acc1, d, c);
  }
  if constexpr (!safe_aligned_mode) {
    // the rest of the real kernel, per tap: no read beyond the last source pixel
    int rest = 0;
    for (; i < kernel_size; i++) {
      int v = src[i];
      if constexpr (sizeof(pixel_t) == 2 && !lessthan16bit)
        v = (short)(v - 32768);
      rest += v * coeff[i];
    }
    acc0 = vaddq_s32(acc0, vsetq_lane_s32(rest, vdupq_n_s32(0), 0));
  }
  return vaddq_s32(acc0, acc1);
}

template<bool safe_aligned_mode, typename pixel_t, bool lessthan16bit>
static AVS_FORCEINLINE void process_eight_pixels_h_neon(const pixel_t* AVS_RESTRICT src, int x, const short* AVS_RESTRICT coeff_base, int filter_size,
  int kernel_size, const int* AVS_RESTRICT pixel_offset, int32x4_t rounder, uint16x8_t clamp_limit, pixel_t* AVS_RESTRICT dst)
{
  const short* coeff = coeff_base + x * filter_size;
  int32x4_t r[8];
  // unrolled: keeps the 8 partial sums in registers (otherwise GCC 12 spills them to the stack;
  // Clang ignores the GCC pragma and kept r[] on the stack too, hence its own pragma)
#if defined(__clang__)
#pragma clang loop unroll(full)
#elif defined(__GNUC__)
#pragma GCC unroll 8
#endif
  for (int j = 0; j < 8; j++)
    r[j] = dot_one_pixel<safe_aligned_mode, pixel_t, lessthan16bit>(src + pixel_offset[x + j], coeff + j * filter_size, filter_size, kernel_size);
  // horizontal sums: lane j = sum of r[j]
  int32x4_t s0123 = vpaddq_s32(vpaddq_s32(r[0], r[1]), vpaddq_s32(r[2], r[3]));
  int32x4_t s4567 = vpaddq_s32(vpaddq_s32(r[4], r[5]), vpaddq_s32(r[6], r[7]));
  // rounder (+ the pivot correction for exact 16 bit)
  s0123 = vaddq_s32(s0123, rounder);
  s4567 = vaddq_s32(s4567, rounder);
  constexpr int fp_scale_bits = sizeof(pixel_t) == 1 ? FPScale8bits : FPScale16bits;
  s0123 = vshrq_n_s32(s0123, fp_scale_bits);
  s4567 = vshrq_n_s32(s4567, fp_scale_bits);
  uint16x8_t res = vcombine_u16(vqmovun_s32(s0123), vqmovun_s32(s4567)); // < 0 -> 0, > 65535 -> 65535
  if constexpr (sizeof(pixel_t) == 1) {
    vst1_u8(dst + x, vqmovn_u16(res)); // > 255 -> 255
  }
  else {
    if constexpr (lessthan16bit)
      res = vminq_u16(res, clamp_limit);
    vst1q_u16(dst + x, res);
  }
}

template<typename pixel_t, bool lessthan16bit>
void resizer_h_neon_generic_uint8_16(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel)
{
  const int filter_size = program->filter_size;
  const int kernel_size = program->filter_size_real;
  constexpr int fp_scale_bits = sizeof(pixel_t) == 1 ? FPScale8bits : FPScale16bits;
  int rounder_i = 1 << (fp_scale_bits - 1);
  if constexpr (sizeof(pixel_t) == 2 && !lessthan16bit)
    rounder_i += 32768 << FPScale16bits; // back from the signed pivot
  const int32x4_t rounder = vdupq_n_s32(rounder_i);
  const uint16x8_t clamp_limit = vdupq_n_u16((uint16_t)((1 << bits_per_pixel) - 1));

  const pixel_t* AVS_RESTRICT src = reinterpret_cast<const pixel_t*>(src8);
  pixel_t* AVS_RESTRICT dst = reinterpret_cast<pixel_t*>(dst8);
  dst_pitch /= sizeof(pixel_t);
  src_pitch /= sizeof(pixel_t);

  const int w_safe_mod8 = (program->safelimit_filter_size_aligned.overread_possible ? program->safelimit_filter_size_aligned.source_overread_beyond_targetx : width) / 8 * 8;
  const int* pixel_offset = program->pixel_offset.data();
  const short* coeff_base = program->pixel_coefficient;

  for (int y = 0; y < height; y++) {
    int x = 0;
    for (; x < w_safe_mod8; x += 8)
      process_eight_pixels_h_neon<true, pixel_t, lessthan16bit>(src, x, coeff_base, filter_size, kernel_size, pixel_offset, rounder, clamp_limit, dst);
    // beyond the safe limit: real kernel size only; pixel_offset has entries up to the next multiple of 8
    for (; x < width; x += 8)
      process_eight_pixels_h_neon<false, pixel_t, lessthan16bit>(src, x, coeff_base, filter_size, kernel_size, pixel_offset, rounder, clamp_limit, dst);
    dst += dst_pitch;
    src += src_pitch;
  }
}

template void resizer_h_neon_generic_uint8_16<uint8_t, true>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resizer_h_neon_generic_uint8_16<uint16_t, true>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resizer_h_neon_generic_uint8_16<uint16_t, false>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);

// --- float ---
// The NEON pair of resize_h_c_planar<float> (resample.cpp), same arithmetic and order, bit-identical
// (as GCC 11/12 compile the C on aarch64):
//   acc (4 lanes, from +0) = fma over the taps in groups of 4, lane k gets the taps i % 4 == k
//   sum = (((0 + acc[0]) + acc[1]) + acc[2]) + acc[3]
//   the rest taps (kernel_size % 4): sum = fma(src, coeff, sum), in tap order
// The C does one output pixel at a time (one dependent fma chain). Here 4 output pixels are processed
// together: 4 independent chains, and the horizontal sums of the 4 pixels are done at once after a 4x4
// transpose, in the same order per lane. The last 1..3 pixels of the row are done one at a time.

static AVS_FORCEINLINE float sum_lanes_like_c(float32x4_t acc)
{
  float sum = 0.0f;
  sum = sum + vgetq_lane_f32(acc, 0);
  sum = sum + vgetq_lane_f32(acc, 1);
  sum = sum + vgetq_lane_f32(acc, 2);
  sum = sum + vgetq_lane_f32(acc, 3);
  return sum;
}

void resizer_h_neon_generic_float(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel)
{
  AVS_UNUSED(bits_per_pixel);
  const int filter_size = program->filter_size;
  const int kernel_size = program->filter_size_real;
  const int ksmod4 = kernel_size / 4 * 4;
  const int* pixel_offset = program->pixel_offset.data();
  const float* coeff_base = program->pixel_coefficient_float;

  const float* src = reinterpret_cast<const float*>(src8);
  float* dst = reinterpret_cast<float*>(dst8);
  src_pitch /= sizeof(float);
  dst_pitch /= sizeof(float);
  const float32x4_t zero = vdupq_n_f32(0.0f);

  for (int y = 0; y < height; y++) {
    int x = 0;
    for (; x + 4 <= width; x += 4) {
      const float* s0 = src + pixel_offset[x + 0];
      const float* s1 = src + pixel_offset[x + 1];
      const float* s2 = src + pixel_offset[x + 2];
      const float* s3 = src + pixel_offset[x + 3];
      const float* c0 = coeff_base + (x + 0) * filter_size;
      const float* c1 = c0 + filter_size;
      const float* c2 = c1 + filter_size;
      const float* c3 = c2 + filter_size;
      float32x4_t a0 = zero, a1 = zero, a2 = zero, a3 = zero;
      for (int i = 0; i < ksmod4; i += 4) {
        a0 = vfmaq_f32(a0, vld1q_f32(c0 + i), vld1q_f32(s0 + i));
        a1 = vfmaq_f32(a1, vld1q_f32(c1 + i), vld1q_f32(s1 + i));
        a2 = vfmaq_f32(a2, vld1q_f32(c2 + i), vld1q_f32(s2 + i));
        a3 = vfmaq_f32(a3, vld1q_f32(c3 + i), vld1q_f32(s3 + i));
      }
      // 4x4 transpose: Lk = lane k of the 4 pixels' accumulators
      const float32x4x2_t t01 = vtrnq_f32(a0, a1); // [a0_0 a1_0 a0_2 a1_2] [a0_1 a1_1 a0_3 a1_3]
      const float32x4x2_t t23 = vtrnq_f32(a2, a3);
      // 64 bit trn1/trn2 pairs the halves (vcombine of halves compiles to lane moves)
      const float32x4_t L0 = vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(t01.val[0]), vreinterpretq_f64_f32(t23.val[0])));
      const float32x4_t L2 = vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(t01.val[0]), vreinterpretq_f64_f32(t23.val[0])));
      const float32x4_t L1 = vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(t01.val[1]), vreinterpretq_f64_f32(t23.val[1])));
      const float32x4_t L3 = vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(t01.val[1]), vreinterpretq_f64_f32(t23.val[1])));
      float32x4_t sum = vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(zero, L0), L1), L2), L3);
      for (int i = ksmod4; i < kernel_size; i++) {
        float32x4_t sv = vdupq_n_f32(s0[i]);
        sv = vsetq_lane_f32(s1[i], sv, 1);
        sv = vsetq_lane_f32(s2[i], sv, 2);
        sv = vsetq_lane_f32(s3[i], sv, 3);
        float32x4_t cv = vdupq_n_f32(c0[i]);
        cv = vsetq_lane_f32(c1[i], cv, 1);
        cv = vsetq_lane_f32(c2[i], cv, 2);
        cv = vsetq_lane_f32(c3[i], cv, 3);
        sum = vfmaq_f32(sum, sv, cv);
      }
      vst1q_f32(dst + x, sum);
    }
    for (; x < width; x++) {
      const float* s0 = src + pixel_offset[x];
      const float* c0 = coeff_base + x * filter_size;
      float32x4_t a0 = zero;
      for (int i = 0; i < ksmod4; i += 4)
        a0 = vfmaq_f32(a0, vld1q_f32(c0 + i), vld1q_f32(s0 + i));
      float sum = sum_lanes_like_c(a0);
      for (int i = ksmod4; i < kernel_size; i++)
        sum = std::fma(s0[i], c0[i], sum);
      dst[x] = sum;
    }
    dst += dst_pitch;
    src += src_pitch;
  }
}

// Vertical resizers, planar 8-16 bit and 32 bit float: the NEON pairs of resize_v_c_planar_uint8_16_t_auto_vectorized
// and resize_v_c_planar_float_auto_vectorized (resample.cpp). The C versions compile to per-pixel lane loads (8-16 bit)
// and accumulators kept in stack memory (float) on aarch64 (GCC 11/12).
// 16 pixels per step (the row end: 8), the taps in two chains (even and odd rows), added at the end, the odd last tap
// to the even chain. Same overread/overwrite as C at the row end (within FRAME_ALIGN).
// 8-16 bit: the integer sum is exact, bit-identical to C in any order. Pivot and rounding as in the horizontal one.
// float: the C order: fused multiply-add per chain, both starting at +0, then even + odd, then the odd last tap fused.

// N pixels (8 or 16) as int16x8 groups: 8 bit zero-extended, 10-14 bit as is, 16 bit pivoted to signed
template<int N, typename pixel_t, bool lessthan16bit>
static AVS_FORCEINLINE void load_n_s16(const pixel_t* p, int16x8_t* out)
{
  if constexpr (sizeof(pixel_t) == 1 && N == 16) {
    const uint8x16_t v = vld1q_u8(p);
    out[0] = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v)));
    out[1] = vreinterpretq_s16_u16(vmovl_high_u8(v));
  }
  else {
    for (int k = 0; k < N / 8; k++)
      out[k] = load8_s16<pixel_t, lessthan16bit>(p + 8 * k);
  }
}

template<int N, typename pixel_t, bool lessthan16bit>
static AVS_FORCEINLINE void process_n_pixels_v_neon(const pixel_t* AVS_RESTRICT src2_ptr, int src_pitch, const short* AVS_RESTRICT coeff,
  int kernel_size, int32x4_t rounder, uint16x8_t clamp_limit, pixel_t* AVS_RESTRICT dst)
{
  constexpr int G = N / 8; // int16x8 groups
  int32x4_t acc_even[2 * G], acc_odd[2 * G];
  for (int r = 0; r < 2 * G; r++) {
    acc_even[r] = rounder;
    acc_odd[r] = vdupq_n_s32(0);
  }
  const int ksmod2 = kernel_size / 2 * 2;
  int i = 0;
  for (; i < ksmod2; i += 2) {
    int16x8_t s0[G], s1[G];
    load_n_s16<N, pixel_t, lessthan16bit>(src2_ptr, s0);
    load_n_s16<N, pixel_t, lessthan16bit>(src2_ptr + src_pitch, s1);
    const int16_t c0 = coeff[i];
    const int16_t c1 = coeff[i + 1];
    for (int k = 0; k < G; k++) {
      acc_even[2 * k] = vmlal_n_s16(acc_even[2 * k], vget_low_s16(s0[k]), c0);
      acc_even[2 * k + 1] = vmlal_high_n_s16(acc_even[2 * k + 1], s0[k], c0);
      acc_odd[2 * k] = vmlal_n_s16(acc_odd[2 * k], vget_low_s16(s1[k]), c1);
      acc_odd[2 * k + 1] = vmlal_high_n_s16(acc_odd[2 * k + 1], s1[k], c1);
    }
    src2_ptr += 2 * src_pitch;
  }
  if (i < kernel_size) {
    int16x8_t s0[G];
    load_n_s16<N, pixel_t, lessthan16bit>(src2_ptr, s0);
    const int16_t c0 = coeff[i];
    for (int k = 0; k < G; k++) {
      acc_even[2 * k] = vmlal_n_s16(acc_even[2 * k], vget_low_s16(s0[k]), c0);
      acc_even[2 * k + 1] = vmlal_high_n_s16(acc_even[2 * k + 1], s0[k], c0);
    }
  }
  constexpr int fp_scale_bits = sizeof(pixel_t) == 1 ? FPScale8bits : FPScale16bits;
  uint16x8_t res[G];
  for (int k = 0; k < G; k++) {
    const int32x4_t lo = vshrq_n_s32(vaddq_s32(acc_even[2 * k], acc_odd[2 * k]), fp_scale_bits);
    const int32x4_t hi = vshrq_n_s32(vaddq_s32(acc_even[2 * k + 1], acc_odd[2 * k + 1]), fp_scale_bits);
    res[k] = vcombine_u16(vqmovun_s32(lo), vqmovun_s32(hi)); // < 0 -> 0, > 65535 -> 65535
    if constexpr (sizeof(pixel_t) == 2 && lessthan16bit)
      res[k] = vminq_u16(res[k], clamp_limit);
  }
  if constexpr (sizeof(pixel_t) == 1) {
    if constexpr (N == 16)
      vst1q_u8(dst, vcombine_u8(vqmovn_u16(res[0]), vqmovn_u16(res[1]))); // > 255 -> 255
    else
      vst1_u8(dst, vqmovn_u16(res[0]));
  }
  else {
    for (int k = 0; k < G; k++)
      vst1q_u16(dst + 8 * k, res[k]);
  }
}

template<typename pixel_t, bool lessthan16bit>
void resize_v_neon_planar_uint8_16(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel)
{
  const int filter_size = program->filter_size;
  const int kernel_size = program->filter_size_real;
  const short* AVS_RESTRICT current_coeff = program->pixel_coefficient;
  constexpr int fp_scale_bits = sizeof(pixel_t) == 1 ? FPScale8bits : FPScale16bits;
  int rounder_i = 1 << (fp_scale_bits - 1);
  if constexpr (sizeof(pixel_t) == 2 && !lessthan16bit)
    rounder_i += 32768 << FPScale16bits; // back from the signed pivot
  const int32x4_t rounder = vdupq_n_s32(rounder_i);
  const uint16x8_t clamp_limit = vdupq_n_u16((uint16_t)((1 << bits_per_pixel) - 1));

  const pixel_t* src = reinterpret_cast<const pixel_t*>(src8);
  pixel_t* AVS_RESTRICT dst = reinterpret_cast<pixel_t*>(dst8);
  dst_pitch /= sizeof(pixel_t);
  src_pitch /= sizeof(pixel_t);
  const int width_mod16 = width / 16 * 16;

  for (int y = 0; y < target_height; y++) {
    const pixel_t* src_ptr = src + program->pixel_offset[y] * src_pitch;
    int x = 0;
    for (; x < width_mod16; x += 16)
      process_n_pixels_v_neon<16, pixel_t, lessthan16bit>(src_ptr + x, src_pitch, current_coeff, kernel_size, rounder, clamp_limit, dst + x);
    for (; x < width; x += 8)
      process_n_pixels_v_neon<8, pixel_t, lessthan16bit>(src_ptr + x, src_pitch, current_coeff, kernel_size, rounder, clamp_limit, dst + x);
    dst += dst_pitch;
    current_coeff += filter_size;
  }
}

template void resize_v_neon_planar_uint8_16<uint8_t, true>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel);
template void resize_v_neon_planar_uint8_16<uint16_t, true>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel);
template void resize_v_neon_planar_uint8_16<uint16_t, false>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel);

template<int N>
static AVS_FORCEINLINE void process_n_pixels_v_neon_float(const float* AVS_RESTRICT src2_ptr, int src_pitch, const float* AVS_RESTRICT coeff,
  int kernel_size, float* AVS_RESTRICT dst)
{
  constexpr int R = N / 4;
  float32x4_t acc_even[R], acc_odd[R];
  for (int r = 0; r < R; r++) {
    acc_even[r] = vdupq_n_f32(0.0f);
    acc_odd[r] = vdupq_n_f32(0.0f);
  }
  const int ksmod2 = kernel_size / 2 * 2;
  int i = 0;
  for (; i < ksmod2; i += 2) {
    const float c0 = coeff[i];
    const float c1 = coeff[i + 1];
    for (int r = 0; r < R; r++) {
      acc_even[r] = vfmaq_n_f32(acc_even[r], vld1q_f32(src2_ptr + 4 * r), c0);
      acc_odd[r] = vfmaq_n_f32(acc_odd[r], vld1q_f32(src2_ptr + src_pitch + 4 * r), c1);
    }
    src2_ptr += 2 * src_pitch;
  }
  for (int r = 0; r < R; r++)
    acc_even[r] = vaddq_f32(acc_even[r], acc_odd[r]);
  if (i < kernel_size) {
    const float c0 = coeff[i];
    for (int r = 0; r < R; r++)
      acc_even[r] = vfmaq_n_f32(acc_even[r], vld1q_f32(src2_ptr + 4 * r), c0);
  }
  for (int r = 0; r < R; r++)
    vst1q_f32(dst + 4 * r, acc_even[r]);
}

void resize_v_neon_planar_float(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel)
{
  AVS_UNUSED(bits_per_pixel);
  const int filter_size = program->filter_size;
  const int kernel_size = program->filter_size_real;
  const float* AVS_RESTRICT current_coeff = program->pixel_coefficient_float;

  const float* src = reinterpret_cast<const float*>(src8);
  float* AVS_RESTRICT dst = reinterpret_cast<float*>(dst8);
  dst_pitch /= sizeof(float);
  src_pitch /= sizeof(float);
  const int width_mod16 = width / 16 * 16;

  for (int y = 0; y < target_height; y++) {
    const float* src_ptr = src + program->pixel_offset[y] * src_pitch;
    int x = 0;
    for (; x < width_mod16; x += 16)
      process_n_pixels_v_neon_float<16>(src_ptr + x, src_pitch, current_coeff, kernel_size, dst + x);
    for (; x < width; x += 8)
      process_n_pixels_v_neon_float<8>(src_ptr + x, src_pitch, current_coeff, kernel_size, dst + x);
    dst += dst_pitch;
    current_coeff += filter_size;
  }
}
