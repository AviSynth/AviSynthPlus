// Avisynth v2.5.  Copyright 2002-2009 Ben Rudiak-Gould et al.
// http://www.avisynth.org

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

#include "convert_bits.h"
#ifdef INTEL_INTRINSICS
#include "intel/convert_bits_sse.h"
#include "intel/convert_bits_avx2.h"
#endif
#include "convert_helper.h"

#include <avs/alignment.h>
#include <avs/minmax.h>
#include <avs/config.h>
#include <tuple>
#include <map>
#include <algorithm>
#include <vector>

#ifdef AVS_WINDOWS
#include <avs/win.h>
#else
#include <avs/posix.h>
#endif

template<typename pixel_t_s, typename pixel_t_d, bool chroma, bool fulls, bool fulld, bool TEMPLATE_NEED_BACKSCALE, bool TEMPLATE_LOW_DITHER_BITDEPTH>
static void do_convert_ordered_dither_uint_c(const BYTE* srcp8, BYTE* dstp8, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth)
{
  const pixel_t_s* srcp = reinterpret_cast<const pixel_t_s*>(srcp8);
  pixel_t_d* dstp = reinterpret_cast<pixel_t_d*>(dstp8);
  dst_pitch = dst_pitch / sizeof(pixel_t_d);

  src_pitch = src_pitch / sizeof(pixel_t_s);
  const int src_width = src_rowsize / sizeof(pixel_t_s);

  // helps compiler optimization
  if constexpr (sizeof(pixel_t_s) == 1)
    source_bitdepth = 8;
  if constexpr (sizeof(pixel_t_d) == 1) {
    target_bitdepth = 8;
    if (!TEMPLATE_NEED_BACKSCALE) {
      dither_target_bitdepth = 8;
    }
  }

  const int max_pixel_value_target = (1 << target_bitdepth) - 1;
  const int max_pixel_value_dithered = (1 << dither_target_bitdepth) - 1;
  // precheck ensures:
  // target_bitdepth >= dither_target_bitdepth
  // source_bitdepth - dither_target_bitdepth <= 8 (max precalculated table is 16x16)
  const bool odd_diff = (source_bitdepth - dither_target_bitdepth) & 1;
  const int dither_bit_diff = (source_bitdepth - dither_target_bitdepth);
  const int dither_order = (dither_bit_diff + 1) / 2;
  const int dither_mask = (1 << dither_order) - 1; // 9,10=2  11,12=4  13,14=8  15,16=16
  // 10->8: 0x01 (2x2)
  // 11->8: 0x03 (4x4)
  // 12->8: 0x03 (4x4)
  // 14->8: 0x07 (8x8)
  // 16->8: 0x0F (16x16)
  const BYTE* matrix;
  switch (dither_order) {
  case 1: matrix = reinterpret_cast<const BYTE*>(odd_diff ? dither2x2a.data : dither2x2.data); break;
  case 2: matrix = reinterpret_cast<const BYTE*>(odd_diff ? dither4x4a.data : dither4x4.data); break;
  case 3: matrix = reinterpret_cast<const BYTE*>(odd_diff ? dither8x8a.data : dither8x8.data); break;
  case 4: matrix = reinterpret_cast<const BYTE*>(odd_diff ? dither16x16a.data : dither16x16.data); break;
  default: return; // n/a
  }

  const int bitdiff_between_dither_and_target = target_bitdepth - dither_target_bitdepth;
  assert(TEMPLATE_NEED_BACKSCALE == (target_bitdepth != dither_target_bitdepth));  // dither to x, target to y

  assert(TEMPLATE_LOW_DITHER_BITDEPTH == (dither_target_bitdepth < 8));
  // e.g. instead of 0,1 => -0.5,+0.5;  0,1,2,3 => -1.5,-0.5,0.5,1.5
  const float half_maxcorr_value = ((1 << dither_bit_diff) - 1) / 2.0f;

  const int source_max = (1 << source_bitdepth) - 1;
  //-----------------------
  // When calculating src_pixel, src and dst are of the same bit depth
  bits_conv_constants d;
  get_bits_conv_constants(d, chroma, fulls, fulld, source_bitdepth, source_bitdepth);

  auto dst_offset_plus_round = d.dst_offset + 0.5f;
  constexpr auto src_pixel_min = 0;
  const auto src_pixel_max = source_max;
  const float mul_factor_backfromlowdither = (float)max_pixel_value_target / max_pixel_value_dithered;
  //-----------------------

  for (int y = 0; y < src_height; y++)
  {
    int _y = (y & dither_mask) << dither_order; // ordered dither
    for (int x = 0; x < src_width; x++)
    {
      const int corr = matrix[_y | (x & dither_mask)];

      int src_pixel = srcp[x];

      if constexpr(fulls != fulld) {
        const float val = (srcp[x] - d.src_offset_i) * d.mul_factor + dst_offset_plus_round;
        src_pixel = clamp((int)val, src_pixel_min, src_pixel_max);
      }

      int new_pixel;
      if (TEMPLATE_LOW_DITHER_BITDEPTH) {
        // accurate dither: +/-
        // accurately positioned to the center
        const float corr_f = corr - half_maxcorr_value;
        new_pixel = (int)(src_pixel + corr_f) >> dither_bit_diff;
      }
      else
        new_pixel = ((src_pixel + corr) >> dither_bit_diff);

      // scale back to the required bit depth
      if constexpr (TEMPLATE_NEED_BACKSCALE) { // dither to x, target to y
        new_pixel = min(new_pixel, max_pixel_value_dithered);
        // Interesting problem of dither_bits==1 (or in general at small dither_bits)
        // After simple slli 0,1 becomes 0,128, we'd expect 0,255 instead. So we make cosmetics.
        if (TEMPLATE_LOW_DITHER_BITDEPTH) {
          new_pixel = (int)(new_pixel * mul_factor_backfromlowdither + 0.5f);
        }
        else {
          new_pixel = new_pixel << bitdiff_between_dither_and_target;
        }
        // dither_bits
        // 1            0,1     => 0,128        => 0,255
        // 2            0,1,2,3 => 0,64,128,192 => 0,?,?,255
        // 3            0,...,7 => 0,32,...,224 => 0,?,?,255
        // 4            0,..,15 => 0,16,...,240 => 0,?,?,255
        // 5            0,..,31 => 0,8,....,248 => 0,?,?,255
        // 6            0,..,63 => 0,4,....,252 => 0,?,?,255
        // 7            0,.,127 => 0,2.  ..,254 => 0,?,?,255
      }
      dstp[x] = (pixel_t_d)(max(min((int)new_pixel, max_pixel_value_target),0));
    }
    dstp += dst_pitch;
    srcp += src_pitch;
  }
}

template<typename pixel_t_s, typename pixel_t_d, bool chroma, bool fulls, bool fulld>
static void convert_ordered_dither_uint_c(const BYTE* srcp8, BYTE* dstp8, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth)
{
  const bool need_backscale = target_bitdepth != dither_target_bitdepth; // dither to x, target to y
  const bool low_dither_bitdepth = dither_target_bitdepth < 8; // 1-7 bits dither targets, need_backscale is always true, since 8 bit format is the minimum
  if (need_backscale) {
    if (low_dither_bitdepth)
      do_convert_ordered_dither_uint_c<pixel_t_s, pixel_t_d, chroma, fulls, fulld, true, true>(srcp8, dstp8, src_rowsize, src_height, src_pitch, dst_pitch, source_bitdepth, target_bitdepth, dither_target_bitdepth);
    else
      do_convert_ordered_dither_uint_c<pixel_t_s, pixel_t_d, chroma, fulls, fulld, true, false>(srcp8, dstp8, src_rowsize, src_height, src_pitch, dst_pitch, source_bitdepth, target_bitdepth, dither_target_bitdepth);
  }
  else {
    do_convert_ordered_dither_uint_c<pixel_t_s, pixel_t_d, chroma, fulls, fulld, false, false>(srcp8, dstp8, src_rowsize, src_height, src_pitch, dst_pitch, source_bitdepth, target_bitdepth, dither_target_bitdepth);
  }
}

// idea borrowed from fmtConv
#define FS_OPTIMIZED_SERPENTINE_COEF

template<int direction>
static AVS_FORCEINLINE void diffuse_floyd(int err, int &nextError, int *error_ptr)
{
#if defined (FS_OPTIMIZED_SERPENTINE_COEF)
  const int      e1 = 0;
  const int      e3 = (err * 4 + 8) >> 4;
#else
  const int      e1 = (err + 8) >> 4;
  const int      e3 = (err * 3 + 8) >> 4;
#endif
  const int      e5 = (err * 5 + 8) >> 4;
  const int      e7 = err - e1 - e3 - e5;

  nextError = error_ptr[direction];
  error_ptr[-direction] += e3;
  error_ptr[0] += e5;
  error_ptr[direction] = e1;
  nextError += e7;
}

#if 0
template<int direction>
static AVS_FORCEINLINE void diffuse_floyd_f(float err, float& nextError, float* error_ptr)
{
#if defined (FS_OPTIMIZED_SERPENTINE_COEF)
  const float    e1 = 0;
  const float    e3 = err * (4.0f / 16);
#else
  const float    e1 = err * (1.0f / 16);
  const float    e3 = err * (3.0f / 16);
#endif
  const float    e5 = err * (5.0f / 16);
  const float    e7 = err * (7.0f / 16);

  nextError = error_ptr[direction];
  error_ptr[-direction] += e3;
  error_ptr[0] += e5;
  error_ptr[direction] = e1;
  nextError += e7;
}
#endif

// fixme: use bool chroma, bool fulls, bool fulld
// optimization helper: TEMPLATE_DITHER_BIT_DIFF if not <0 then hold value for frequently used differences from 16->8
// 1nd helper: TEMPLATE_LOW_DITHER_BITDEPTH
template<typename pixel_t_s, typename pixel_t_d, bool chroma, bool fulls, bool fulld, int TEMPLATE_DITHER_BIT_DIFF, bool TEMPLATE_LOW_DITHER_BITDEPTH>
static void do_convert_uint_floyd_c(const BYTE* srcp8, BYTE* dstp8, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth)
{
  if constexpr (TEMPLATE_DITHER_BIT_DIFF > 0) {
    assert(TEMPLATE_DITHER_BIT_DIFF == (source_bitdepth - dither_target_bitdepth));
    assert(target_bitdepth == dither_target_bitdepth);
    dither_target_bitdepth = source_bitdepth - TEMPLATE_DITHER_BIT_DIFF;
    target_bitdepth = dither_target_bitdepth;
    // seems that direct shift with known constant bits is really quicker than shift by 'cl'. 
    // Dithering from 16 to 10 bit: 109 vs 101 fps
    // PF note: experienced the same with SIMD _mm_srl and _mm_srli in the other bit converter
  }
  const int DITHER_BIT_DIFF = TEMPLATE_DITHER_BIT_DIFF > 0 ? TEMPLATE_DITHER_BIT_DIFF : (source_bitdepth - dither_target_bitdepth);
  assert(TEMPLATE_LOW_DITHER_BITDEPTH == (dither_target_bitdepth < 8)); // must match with dispatcher

  const pixel_t_s* srcp = reinterpret_cast<const pixel_t_s*>(srcp8);
  src_pitch = src_pitch / sizeof(pixel_t_s);
  const int src_width = src_rowsize / sizeof(pixel_t_s);

  pixel_t_d* dstp = reinterpret_cast<pixel_t_d*>(dstp8);
  dst_pitch = dst_pitch / sizeof(pixel_t_d);

  // perhaps helps compiler to optimize
  if constexpr (sizeof(pixel_t_s) == 1)
    source_bitdepth = 8;
  if constexpr (sizeof(pixel_t_d) == 1)
    target_bitdepth = 8;

  const int max_pixel_value_target = (1 << target_bitdepth) - 1;
  const int BITDIFF_BETWEEN_DITHER_AND_TARGET = DITHER_BIT_DIFF - (source_bitdepth - target_bitdepth);
  const int max_pixel_value_dithered = (1 << dither_target_bitdepth) - 1;

  auto error_ptr_safe = new int[1 + src_width + 1]; // accumulated errors
  std::fill_n(error_ptr_safe, 1 + src_width + 1, 0);

  int error_ptr = 1;

  const int ROUNDER = 1 << (DITHER_BIT_DIFF - 1); // rounding
  const int source_max = (1 << source_bitdepth) - 1;
  //-----------------------
  bits_conv_constants d;
  get_bits_conv_constants(d, chroma, fulls, fulld, source_bitdepth, source_bitdepth);

  auto dst_offset_plus_round = d.dst_offset + 0.5f;
  constexpr auto src_pixel_min = 0;
  const auto src_pixel_max = source_max;
  const float mul_factor_backfromlowdither = (float)max_pixel_value_target / max_pixel_value_dithered;

  for (int y = 0; y < src_height; y++)
  {
    int nextError = error_ptr_safe[error_ptr];
    // serpentine forward
    if ((y & 1) == 0)
    {
      for (int x = 0; x < src_width; x++)
      {
        int err = nextError;
        int src_pixel = srcp[x];

        if constexpr (fulls != fulld) {
          const float val = (srcp[x] - d.src_offset_i) * d.mul_factor + dst_offset_plus_round;
          src_pixel = clamp((int)val, src_pixel_min, src_pixel_max);
        }

        if (TEMPLATE_LOW_DITHER_BITDEPTH) {
          // accurate dither: +/-
          // accurately positioned to the center
          err = err - (1 << (DITHER_BIT_DIFF - 1)); // signed
        }
        int sum = src_pixel + err;

        int quantized = (sum + ROUNDER) >> (DITHER_BIT_DIFF);
        err = sum - (quantized << DITHER_BIT_DIFF);
        // Interesting problem of dither_bits==1 (or in general at small dither_bits)
        // After simple slli 0,1 becomes 0,128, we'd expect 0,255 instead. So we make cosmetics.
        if (TEMPLATE_LOW_DITHER_BITDEPTH) {
          quantized = min(quantized, max_pixel_value_dithered);
          quantized = (int)(quantized * mul_factor_backfromlowdither + 0.5f);
        }
        else {
          quantized <<= BITDIFF_BETWEEN_DITHER_AND_TARGET;
        }
        int pix = max(min(max_pixel_value_target, quantized), 0); // clamp to target bit
        dstp[x] = (pixel_t_d)pix;
        diffuse_floyd<1>(err, nextError, &error_ptr_safe[error_ptr + x]);
      }
    }
    else {
      // serpentine backward
      for (int x = src_width - 1; x >= 0; --x)
      {
        int err = nextError;
        int src_pixel = srcp[x];

        if constexpr (fulls != fulld) {
          const float val = (srcp[x] - d.src_offset_i) * d.mul_factor + dst_offset_plus_round;
          src_pixel = clamp((int)val, src_pixel_min, src_pixel_max);
        }

        if (TEMPLATE_LOW_DITHER_BITDEPTH) {
          // accurate dither: +/-
          // accurately positioned to the center
          err = err - (1 << (DITHER_BIT_DIFF - 1)); // signed
        }
        int sum = src_pixel + err;

        int quantized = (sum + ROUNDER) >> (DITHER_BIT_DIFF);
        err = sum - (quantized << DITHER_BIT_DIFF);
        // Interesting problem of dither_bits==1 (or in general at small dither_bits)
        // After simple slli 0,1 becomes 0,128, we'd expect 0,255 instead. So we make cosmetics.
        if (TEMPLATE_LOW_DITHER_BITDEPTH) {
          quantized = min(quantized, max_pixel_value_dithered);
          quantized = (int)(quantized * mul_factor_backfromlowdither + 0.5f);
        }
        else {
          quantized <<= BITDIFF_BETWEEN_DITHER_AND_TARGET;
        }
        int pix = max(min(max_pixel_value_target, quantized), 0); // clamp to target bit
        dstp[x] = (pixel_t_d)pix;
        diffuse_floyd<-1>(err, nextError, &error_ptr_safe[error_ptr + x]);
      }
    }
    error_ptr_safe[error_ptr] = nextError;
    dstp += dst_pitch;
    srcp += src_pitch;
  }
  delete[]error_ptr_safe;
}

template<typename pixel_t_s, typename pixel_t_d, bool chroma, bool fulls, bool fulld>
static void convert_uint_floyd_c(const BYTE* srcp8, BYTE* dstp8, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth)
{
  const int dither_bit_diff = source_bitdepth - dither_target_bitdepth;
  const bool low_dither_bitdepth = dither_target_bitdepth < 8;
  // extra internal template makes it quicker for ordinary non-artistic cases
  // do not make templates for all 1-16 target bit combinations
  if (low_dither_bitdepth) {
    do_convert_uint_floyd_c<pixel_t_s, pixel_t_d, chroma, fulls, fulld, -1, true>(srcp8, dstp8, src_rowsize, src_height, src_pitch, dst_pitch, source_bitdepth, target_bitdepth, dither_target_bitdepth);
  }
  else {
    if (target_bitdepth == dither_target_bitdepth) {
      switch (dither_bit_diff) {
      case 2: // e.g. 10->8
        do_convert_uint_floyd_c<pixel_t_s, pixel_t_d, chroma, fulls, fulld, 2, false>(srcp8, dstp8, src_rowsize, src_height, src_pitch, dst_pitch, source_bitdepth, target_bitdepth, dither_target_bitdepth);
        break;
      case 4: // e.g. 12->8
        do_convert_uint_floyd_c<pixel_t_s, pixel_t_d, chroma, fulls, fulld, 4, false>(srcp8, dstp8, src_rowsize, src_height, src_pitch, dst_pitch, source_bitdepth, target_bitdepth, dither_target_bitdepth);
        break;
      case 6: // e.g. 16->10
        do_convert_uint_floyd_c<pixel_t_s, pixel_t_d, chroma, fulls, fulld, 6, false>(srcp8, dstp8, src_rowsize, src_height, src_pitch, dst_pitch, source_bitdepth, target_bitdepth, dither_target_bitdepth);
        break;
      case 8: // e.g. 16->8
        do_convert_uint_floyd_c<pixel_t_s, pixel_t_d, chroma, fulls, fulld, 8, false>(srcp8, dstp8, src_rowsize, src_height, src_pitch, dst_pitch, source_bitdepth, target_bitdepth, dither_target_bitdepth);
        break;
      default: // difference is more than 8 or exotic dither to less than 8 bits, we accept 10-15% speed minus
        do_convert_uint_floyd_c<pixel_t_s, pixel_t_d, chroma, fulls, fulld, -1, false>(srcp8, dstp8, src_rowsize, src_height, src_pitch, dst_pitch, source_bitdepth, target_bitdepth, dither_target_bitdepth);
      }
    }
    else {
      do_convert_uint_floyd_c<pixel_t_s, pixel_t_d, chroma, fulls, fulld, -1, false>(srcp8, dstp8, src_rowsize, src_height, src_pitch, dst_pitch, source_bitdepth, target_bitdepth, dither_target_bitdepth);
    }
  }
}

// float to 8-16 bits
template<typename pixel_t, bool chroma, bool fulls, bool fulld>
static void convert_32_to_uintN_c(const BYTE *srcp, BYTE *dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth)
{
  const float *srcp0 = reinterpret_cast<const float *>(srcp);
  pixel_t *dstp0 = reinterpret_cast<pixel_t *>(dstp);

  src_pitch = src_pitch / sizeof(float);
  dst_pitch = dst_pitch / sizeof(pixel_t);

  const int src_width = src_rowsize / sizeof(float);

  //-----------------------

  bits_conv_constants d;
  get_bits_conv_constants(d, chroma, fulls, fulld, source_bitdepth, target_bitdepth);

  auto dst_offset_plus_round = d.dst_offset + 0.5f;
  constexpr auto dst_pixel_min = 0;
  const auto dst_pixel_max = (1 << target_bitdepth) - 1;

  for(int y=0; y<src_height; y++)
  {
    for (int x = 0; x < src_width; x++)
    {
      const int pixel = (int)((srcp0[x] - d.src_offset) * d.mul_factor + dst_offset_plus_round);
      dstp0[x] = pixel_t(clamp(pixel, dst_pixel_min, dst_pixel_max));
    }
    dstp0 += dst_pitch;
    srcp0 += src_pitch;
  }
}

// YUV: bit shift 8-16 <=> 8-16 bits
// shift right or left, depending on expandrange
template<typename pixel_t_s, typename pixel_t_d>
static void convert_uint_limited_c(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth)
{
  const pixel_t_s* srcp0 = reinterpret_cast<const pixel_t_s*>(srcp);
  pixel_t_d* dstp0 = reinterpret_cast<pixel_t_d*>(dstp);

  src_pitch = src_pitch / sizeof(pixel_t_s);
  dst_pitch = dst_pitch / sizeof(pixel_t_d);

  const int src_width = src_rowsize / sizeof(pixel_t_s);

  if (target_bitdepth > source_bitdepth) // expandrange
  {
    const int shift_bits = target_bitdepth - source_bitdepth;
    for (int y = 0; y < src_height; y++)
    {
      for (int x = 0; x < src_width; x++) {
        dstp0[x] = (pixel_t_d)(srcp0[x]) << shift_bits;  // expand range. No clamp before, source is assumed to have valid range
      }
      dstp0 += dst_pitch;
      srcp0 += src_pitch;
    }
  }
  else
  {
    // reduce range
    const int shift_bits = source_bitdepth - target_bitdepth;
    const int round = 1 << (shift_bits - 1);
    for (int y = 0; y < src_height; y++)
    {
      for (int x = 0; x < src_width; x++) {
        dstp0[x] = (srcp0[x] + round) >> shift_bits;  // reduce range
      }
      dstp0 += dst_pitch;
      srcp0 += src_pitch;
    }
  }
}

// chroma is special in full range
template<typename pixel_t_s, typename pixel_t_d, bool chroma, bool fulls, bool fulld>
static void convert_uint_c(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth)
{
  // limited to limited is bitshift, see in other function
  if constexpr (!fulls && !fulld) {
    convert_uint_limited_c< pixel_t_s, pixel_t_d>(srcp, dstp, src_rowsize, src_height, src_pitch, dst_pitch, source_bitdepth, target_bitdepth, dither_target_bitdepth);
    return;
  }

  const pixel_t_s* srcp0 = reinterpret_cast<const pixel_t_s*>(srcp);
  pixel_t_d* dstp0 = reinterpret_cast<pixel_t_d*>(dstp);

  src_pitch = src_pitch / sizeof(pixel_t_s);
  dst_pitch = dst_pitch / sizeof(pixel_t_d);

  const int src_width = src_rowsize / sizeof(pixel_t_s);

  if constexpr (sizeof(pixel_t_s) == 1 && sizeof(pixel_t_d) == 2) {
    if (fulls && fulld && !chroma && source_bitdepth == 8 && target_bitdepth == 16) {
      // special case * 65535 / 255 = *257 exactly
      for (int y = 0; y < src_height; y++)
      {
        for (int x = 0; x < src_width; x++)
        {
          dstp0[x] = (pixel_t_d)(srcp0[x] * 257);
        }
        dstp0 += dst_pitch;
        srcp0 += src_pitch;
      }
      return;
    }
  }

  const int target_max = (1 << target_bitdepth) - 1;

  bits_conv_constants d;
  get_bits_conv_constants(d, chroma, fulls, fulld, source_bitdepth, target_bitdepth);

  auto dst_offset_plus_round = d.dst_offset + 0.5f;
  constexpr auto target_min = 0;

  for (int y = 0; y < src_height; y++) {
    for (int x = 0; x < src_width; x++)
    {
      const float val = (srcp0[x] - d.src_offset_i) * d.mul_factor + dst_offset_plus_round;
      dstp0[x] = clamp((int)val, target_min, target_max);
    }

    dstp0 += dst_pitch;
    srcp0 += src_pitch;
  }
}



// 8 bit to float, 16/14/12/10 bits to float
template<typename pixel_t, bool chroma, bool fulls, bool fulld>
static void convert_uintN_to_float_c(const BYTE *srcp, BYTE *dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth)
{
  const pixel_t *srcp0 = reinterpret_cast<const pixel_t *>(srcp);
  float *dstp0 = reinterpret_cast<float *>(dstp);

  src_pitch = src_pitch / sizeof(pixel_t);
  dst_pitch = dst_pitch / sizeof(float);

  const int src_width = src_rowsize / sizeof(pixel_t);

  //-----------------------
  bits_conv_constants d;
  get_bits_conv_constants(d, chroma, fulls, fulld, source_bitdepth, target_bitdepth);

  for (int y = 0; y < src_height; y++)
  {
    for (int x = 0; x < src_width; x++)
    {
      const float pixel = (srcp0[x] - d.src_offset_i) * d.mul_factor + d.dst_offset;
      dstp0[x] = pixel; // no clamp
    }
    dstp0 += dst_pitch;
    srcp0 += src_pitch;
  }
}

// float to float
template<bool chroma, bool fulls, bool fulld>
static void convert_float_to_float_c(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth)
{
  // float is good if always full range. For historical reasons Avisynth has "limited" range float,
  // which is simply a /255.0 reduction of original byte pixels
  const float* srcp0 = reinterpret_cast<const float*>(srcp);
  float* dstp0 = reinterpret_cast<float*>(dstp);

  src_pitch = src_pitch / sizeof(float);
  dst_pitch = dst_pitch / sizeof(float);

  const int src_width = src_rowsize / sizeof(float);

  //-----------------------
  bits_conv_constants d;
  get_bits_conv_constants(d, chroma, fulls, fulld, source_bitdepth, target_bitdepth);

#if 0
  float src_offset = 0;

  if constexpr (/*chroma ||*/ !fulls) {
    src_offset = d.src_offset;
  }
  else {
    assert(d.src_offset == 0);
    src_offset = 0; // helps optimizer
  }
#endif

  for (int y = 0; y < src_height; y++)
  {
    for (int x = 0; x < src_width; x++)
    {
      const float pixel = (srcp0[x] - d.src_offset) * d.mul_factor + d.dst_offset;
      dstp0[x] = pixel; // no clamp
    }
    dstp0 += dst_pitch;
    srcp0 += src_pitch;
  }
}

static void get_convert_32_to_uintN_functions(int target_bitdepth, bool fulls, bool fulld, 
#ifdef INTEL_INTRINSICS
  bool sse2, bool sse4, bool avx2,
#endif
  BitDepthConvFuncPtr& conv_function, BitDepthConvFuncPtr& conv_function_chroma, BitDepthConvFuncPtr& conv_function_a)
{
  // 32bit->8-16bits support fulls fulld
      // pure C
#define convert_32_to_uintN_functions(uint_X_t) \
      conv_function_a = convert_32_to_uintN_c<uint_X_t, false, true, true>; /* full-full */ \
      if (fulls && fulld) { \
        conv_function = convert_32_to_uintN_c<uint_X_t, false, true, true>; \
        conv_function_chroma = convert_32_to_uintN_c<uint_X_t, true, true, true>; \
      } \
      else if (fulls && !fulld) { \
        conv_function = convert_32_to_uintN_c<uint_X_t, false, true, false>; \
        conv_function_chroma = convert_32_to_uintN_c<uint_X_t, true, true, false>; \
      } \
      else if (!fulls && fulld) { \
        conv_function = convert_32_to_uintN_c<uint_X_t, false, false, true>; \
        conv_function_chroma = convert_32_to_uintN_c<uint_X_t, true, false, true>; \
      } \
      else if (!fulls && !fulld) { \
        conv_function = convert_32_to_uintN_c<uint_X_t, false, false, false>; \
        conv_function_chroma = convert_32_to_uintN_c<uint_X_t, true, false, false>; \
      }

#ifdef INTEL_INTRINSICS
#undef convert_32_to_uintN_functions

#define convert_32_to_uintN_functions(uint_X_t) \
      conv_function_a = avx2 ? convert_32_to_uintN_avx2<uint_X_t, false, true, true> : sse4 ? convert_32_to_uintN_sse41<uint_X_t, false, true, true> : convert_32_to_uintN_c<uint_X_t, false, true, true>; /* full-full */ \
      if (fulls && fulld) { \
        conv_function = avx2 ? convert_32_to_uintN_avx2<uint_X_t, false, true, true> : sse4 ? convert_32_to_uintN_sse41<uint_X_t, false, true, true> : convert_32_to_uintN_c<uint_X_t, false, true, true>; \
        conv_function_chroma = avx2 ? convert_32_to_uintN_avx2<uint_X_t, true, true, true> : sse4 ? convert_32_to_uintN_sse41<uint_X_t, true, true, true> : convert_32_to_uintN_c<uint_X_t, true, true, true>; \
      } \
      else if (fulls && !fulld) { \
        conv_function = avx2 ? convert_32_to_uintN_avx2<uint_X_t, false, true, false> : sse4 ? convert_32_to_uintN_sse41<uint_X_t, false, true, false> : convert_32_to_uintN_c<uint_X_t, false, true, false>; \
        conv_function_chroma = avx2 ? convert_32_to_uintN_avx2<uint_X_t, true, true, false> : sse4 ? convert_32_to_uintN_sse41<uint_X_t, true, true, false> : convert_32_to_uintN_c<uint_X_t, true, true, false>; \
      } \
      else if (!fulls && fulld) { \
        conv_function = avx2 ? convert_32_to_uintN_avx2<uint_X_t, false, false, true> : sse4 ? convert_32_to_uintN_sse41<uint_X_t, false, false, true> : convert_32_to_uintN_c<uint_X_t, false, false, true>; \
        conv_function_chroma = avx2 ? convert_32_to_uintN_avx2<uint_X_t, true, false, true> : sse4 ? convert_32_to_uintN_sse41<uint_X_t, true, false, true> : convert_32_to_uintN_c<uint_X_t, true, false, true>; \
      } \
      else if (!fulls && !fulld) { \
        conv_function = avx2 ? convert_32_to_uintN_avx2<uint_X_t, false, false, false> : sse4 ? convert_32_to_uintN_sse41<uint_X_t, false, false, false> : convert_32_to_uintN_c<uint_X_t, false, false, false>; \
        conv_function_chroma = avx2 ? convert_32_to_uintN_avx2<uint_X_t, true, false, false> : sse4 ? convert_32_to_uintN_sse41<uint_X_t, true, false, false> : convert_32_to_uintN_c<uint_X_t, true, false, false>; \
      }
#endif

  switch (target_bitdepth)
  {
  case 8:
    convert_32_to_uintN_functions(uint8_t); // all variations of fulls fulld
    break;
  default:
    // 10-16 bits
    convert_32_to_uintN_functions(uint16_t); // all variations of fulls fulld
    break;
  }

#undef convert_32_to_uintN_functions
}

static void get_convert_float_to_float_functions(bool fulls, bool fulld,
  BitDepthConvFuncPtr& conv_function, BitDepthConvFuncPtr& conv_function_chroma, BitDepthConvFuncPtr& conv_function_a)
{
  // 32bit->32bits support fulls fulld, alpha is always full-full
  conv_function_a = convert_float_to_float_c<false, true, true>; /* full-full */
  if (fulls && fulld) {
    conv_function = convert_float_to_float_c<false, true, true>;
    conv_function_chroma = convert_float_to_float_c<true, true, true>;
  }
  else if (fulls && !fulld) {
    conv_function = convert_float_to_float_c<false, true, false>;
    conv_function_chroma = convert_float_to_float_c<true, true, false>;
  }
  else if (!fulls && fulld) { \
    conv_function = convert_float_to_float_c<false, false, true>;
    conv_function_chroma = convert_float_to_float_c<true, false, true>;
  }
  else if (!fulls && !fulld) {
    conv_function = convert_float_to_float_c<false, false, false>;
    conv_function_chroma = convert_float_to_float_c<true, false, false>;
  }
}

static void get_convert_uintN_to_float_functions(int bits_per_pixel, bool fulls, bool fulld,
  BitDepthConvFuncPtr& conv_function, BitDepthConvFuncPtr& conv_function_chroma, BitDepthConvFuncPtr& conv_function_a)
{
  // 8-16bit->32bits support fulls fulld, alpha is always full-full
#define convert_uintN_to_float_functions(uint_X_t) \
      conv_function_a = convert_uintN_to_float_c<uint_X_t, false, true, true>; /* full-full */ \
      if (fulls && fulld) { \
        conv_function = convert_uintN_to_float_c<uint_X_t, false, true, true>; \
        conv_function_chroma = convert_uintN_to_float_c<uint_X_t, true, true, true>; \
      } \
      else if (fulls && !fulld) { \
        conv_function = convert_uintN_to_float_c<uint_X_t, false, true, false>; \
        conv_function_chroma = convert_uintN_to_float_c<uint_X_t, true, true, false>; \
      } \
      else if (!fulls && fulld) { \
        conv_function = convert_uintN_to_float_c<uint_X_t, false, false, true>; \
        conv_function_chroma = convert_uintN_to_float_c<uint_X_t, true, false, true>; \
      } \
      else if (!fulls && !fulld) { \
        conv_function = convert_uintN_to_float_c<uint_X_t, false, false, false>; \
        conv_function_chroma = convert_uintN_to_float_c<uint_X_t, true, false, false>; \
      }
  switch (bits_per_pixel) {
  case 8: 
    convert_uintN_to_float_functions(uint8_t);
    break;
  default: // 10-16 bits
    convert_uintN_to_float_functions(uint16_t);
    break;
  }
#undef convert_uintN_to_float_functions
}

static void get_convert_uintN_to_uintN_ordered_dither_functions(int source_bitdepth, int target_bitdepth, bool fulls, bool fulld,
#ifdef INTEL_INTRINSICS
  bool sse2, bool sse4, bool avx2,
#endif
  BitDepthConvFuncPtr& conv_function, BitDepthConvFuncPtr& conv_function_chroma)
{
// 8-16->8-16 bits support any fulls fulld combination
// dither has no "conv_function_a"
// pure C
#define convert_uintN_to_uintN_ordered_dither_functions(uint_X_t, uint_X_dest_t) \
      if (fulls && fulld) { \
        conv_function = convert_ordered_dither_uint_c<uint_X_t, uint_X_dest_t, false, true, true>; \
        conv_function_chroma = convert_ordered_dither_uint_c<uint_X_t, uint_X_dest_t, true, true, true>; \
      } \
      else if (fulls && !fulld) { \
        conv_function = convert_ordered_dither_uint_c<uint_X_t, uint_X_dest_t, false, true, false>; \
        conv_function_chroma = convert_ordered_dither_uint_c<uint_X_t, uint_X_dest_t, true, true, false>; \
      } \
      else if (!fulls && fulld) { \
        conv_function = convert_ordered_dither_uint_c<uint_X_t, uint_X_dest_t, false, false, true>; \
        conv_function_chroma = convert_ordered_dither_uint_c<uint_X_t, uint_X_dest_t, true, false, true>; \
      } \
      else if (!fulls && !fulld) { \
        conv_function = convert_ordered_dither_uint_c<uint_X_t, uint_X_dest_t, false, false, false>; \
        conv_function_chroma = convert_ordered_dither_uint_c<uint_X_t, uint_X_dest_t, true, false, false>; \
      }

#ifdef INTEL_INTRINSICS
#undef convert_uintN_to_uintN_ordered_dither_functions
// dither has no "conv_function_a"
#define convert_uintN_to_uintN_ordered_dither_functions(uint_X_t, uint_X_dest_t) \
      if (fulls && fulld) { \
        conv_function = avx2 ? convert_ordered_dither_uint_avx2<uint_X_t, uint_X_dest_t, false, true, true> : sse4 ? convert_ordered_dither_uint_sse41<uint_X_t, uint_X_dest_t, false, true, true> : convert_ordered_dither_uint_c<uint_X_t, uint_X_dest_t, false, true, true>; \
        conv_function_chroma = avx2 ? convert_ordered_dither_uint_avx2<uint_X_t, uint_X_dest_t, true, true, true> : sse4 ? convert_ordered_dither_uint_sse41<uint_X_t, uint_X_dest_t, true, true, true> : convert_ordered_dither_uint_c<uint_X_t, uint_X_dest_t, true, true, true>; \
      } \
      else if (fulls && !fulld) { \
        conv_function = avx2 ? convert_ordered_dither_uint_avx2<uint_X_t, uint_X_dest_t, false, true, false> : sse4 ? convert_ordered_dither_uint_sse41<uint_X_t, uint_X_dest_t, false, true, false> : convert_ordered_dither_uint_c<uint_X_t, uint_X_dest_t, false, true, false>; \
        conv_function_chroma = avx2 ? convert_ordered_dither_uint_avx2<uint_X_t, uint_X_dest_t, true, true, false> : sse4 ? convert_ordered_dither_uint_sse41<uint_X_t, uint_X_dest_t, true, true, false> : convert_ordered_dither_uint_c<uint_X_t, uint_X_dest_t, true, true, false>; \
      } \
      else if (!fulls && fulld) { \
        conv_function = avx2 ? convert_ordered_dither_uint_avx2<uint_X_t, uint_X_dest_t, false, false, true> : sse4 ? convert_ordered_dither_uint_sse41<uint_X_t, uint_X_dest_t, false, false, true> : convert_ordered_dither_uint_c<uint_X_t, uint_X_dest_t, false, false, true>; \
        conv_function_chroma = avx2 ? convert_ordered_dither_uint_avx2<uint_X_t, uint_X_dest_t, true, false, true> : sse4 ? convert_ordered_dither_uint_sse41<uint_X_t, uint_X_dest_t, true, false, true> : convert_ordered_dither_uint_c<uint_X_t, uint_X_dest_t, true, false, true>; \
      } \
      else if (!fulls && !fulld) { \
       conv_function = avx2 ? convert_ordered_dither_uint_avx2<uint_X_t, uint_X_dest_t, false, false, false> : sse4 ? convert_ordered_dither_uint_sse41<uint_X_t, uint_X_dest_t, false, false, false> : convert_ordered_dither_uint_c<uint_X_t, uint_X_dest_t, false, false, false>; \
       conv_function_chroma = avx2 ? convert_ordered_dither_uint_avx2<uint_X_t, uint_X_dest_t, true, false, false> : sse4 ? convert_ordered_dither_uint_sse41<uint_X_t, uint_X_dest_t, true, false, false> : convert_ordered_dither_uint_c<uint_X_t, uint_X_dest_t, true, false, false>; \
      }
#endif
  // all variations of fulls fulld byte/word source/target
  switch (target_bitdepth)
  {
  case 8:
    if (source_bitdepth == 8) {
      convert_uintN_to_uintN_ordered_dither_functions(uint8_t, uint8_t);
    }
    else {
      convert_uintN_to_uintN_ordered_dither_functions(uint16_t, uint8_t);
    }
    break;
  default:
    // uint16_t target is always uint16_t source
    convert_uintN_to_uintN_ordered_dither_functions(uint16_t, uint16_t);
    break;
  }

#undef convert_uintN_to_uintN_ordered_dither_functions
}

static void get_convert_uintN_to_uintN_floyd_dither_functions(int source_bitdepth, int target_bitdepth, bool fulls, bool fulld,
  BitDepthConvFuncPtr& conv_function, BitDepthConvFuncPtr& conv_function_chroma)
{
  // 8-16->8-16 bits support any fulls fulld combination
  // dither has no "conv_function_a"
  // pure C
#define convert_uintN_to_uintN_floyd_dither_functions(uint_X_t, uint_X_dest_t) \
      if (fulls && fulld) { \
        conv_function = convert_uint_floyd_c<uint_X_t, uint_X_dest_t, false, true, true>; \
        conv_function_chroma = convert_uint_floyd_c<uint_X_t, uint_X_dest_t, true, true, true>; \
      } \
      else if (fulls && !fulld) { \
        conv_function = convert_uint_floyd_c<uint_X_t, uint_X_dest_t, false, true, false>; \
        conv_function_chroma = convert_uint_floyd_c<uint_X_t, uint_X_dest_t, true, true, false>; \
      } \
      else if (!fulls && fulld) { \
        conv_function = convert_uint_floyd_c<uint_X_t, uint_X_dest_t, false, false, true>; \
        conv_function_chroma = convert_uint_floyd_c<uint_X_t, uint_X_dest_t, true, false, true>; \
      } \
      else if (!fulls && !fulld) { \
        conv_function = convert_uint_floyd_c<uint_X_t, uint_X_dest_t, false, false, false>; \
        conv_function_chroma = convert_uint_floyd_c<uint_X_t, uint_X_dest_t, true, false, false>; \
      }

  // all variations of fulls fulld byte/word source/target
  switch (target_bitdepth)
  {
  case 8:
    if (source_bitdepth == 8) {
      convert_uintN_to_uintN_floyd_dither_functions(uint8_t, uint8_t);
    }
    else {
      convert_uintN_to_uintN_floyd_dither_functions(uint16_t, uint8_t);
    }
    break;
  default:
    // uint16_t target is always uint16_t source
    convert_uintN_to_uintN_floyd_dither_functions(uint16_t, uint16_t);
    break;
  }

#undef convert_uintN_to_uintN_ordered_dither_functions
}


static void get_convert_uintN_to_uintN_functions(int source_bitdepth, int target_bitdepth, bool fulls, bool fulld,
#ifdef INTEL_INTRINSICS
  bool sse2, bool sse4, bool avx2,
#endif
  BitDepthConvFuncPtr& conv_function, BitDepthConvFuncPtr& conv_function_chroma, BitDepthConvFuncPtr& conv_function_a)
{
  // 8-16->8-16 bits support any fulls fulld combination
  // pure C
#define convert_uintN_to_uintN_functions(uint_X_t, uint_X_dest_t) \
      conv_function_a = convert_uint_c<uint_X_t, uint_X_dest_t, false, true, true>; /* full-full */ \
      if (fulls && fulld) { \
        conv_function = convert_uint_c<uint_X_t, uint_X_dest_t, false, true, true>; \
        conv_function_chroma = convert_uint_c<uint_X_t, uint_X_dest_t, true, true, true>; \
      } \
      else if (fulls && !fulld) { \
        conv_function = convert_uint_c<uint_X_t, uint_X_dest_t, false, true, false>; \
        conv_function_chroma = convert_uint_c<uint_X_t, uint_X_dest_t, true, true, false>; \
      } \
      else if (!fulls && fulld) { \
        conv_function = convert_uint_c<uint_X_t, uint_X_dest_t, false, false, true>; \
        conv_function_chroma = convert_uint_c<uint_X_t, uint_X_dest_t, true, false, true>; \
      } \
      else if (!fulls && !fulld) { \
        conv_function = convert_uint_c<uint_X_t, uint_X_dest_t, false, false, false>; \
        conv_function_chroma = convert_uint_c<uint_X_t, uint_X_dest_t, true, false, false>; \
      }

#ifdef INTEL_INTRINSICS
#undef convert_uintN_to_uintN_functions

#define convert_uintN_to_uintN_functions(uint_X_t, uint_X_dest_t) \
      conv_function_a = avx2 ? convert_uint_avx2<uint_X_t, uint_X_dest_t, false, true, true> : sse4 ? convert_uint_sse41<uint_X_t, uint_X_dest_t, false, true, true> : convert_uint_c<uint_X_t, uint_X_dest_t, false, true, true>; /* full-full */ \
      if (fulls && fulld) { \
        conv_function = avx2 ? convert_uint_avx2<uint_X_t, uint_X_dest_t, false, true, true> : sse4 ? convert_uint_sse41<uint_X_t, uint_X_dest_t, false, true, true> : convert_uint_c<uint_X_t, uint_X_dest_t, false, true, true>; \
        conv_function_chroma = avx2 ? convert_uint_avx2<uint_X_t, uint_X_dest_t, true, true, true> : sse4 ? convert_uint_sse41<uint_X_t, uint_X_dest_t, true, true, true> : convert_uint_c<uint_X_t, uint_X_dest_t, true, true, true>; \
      } \
      else if (fulls && !fulld) { \
        conv_function = avx2 ? convert_uint_avx2<uint_X_t, uint_X_dest_t, false, true, false> : sse4 ? convert_uint_sse41<uint_X_t, uint_X_dest_t, false, true, false> : convert_uint_c<uint_X_t, uint_X_dest_t, false, true, false>; \
        conv_function_chroma = avx2 ? convert_uint_avx2<uint_X_t, uint_X_dest_t, true, true, false> : sse4 ? convert_uint_sse41<uint_X_t, uint_X_dest_t, true, true, false> : convert_uint_c<uint_X_t, uint_X_dest_t, true, true, false>; \
      } \
      else if (!fulls && fulld) { \
        conv_function = avx2 ? convert_uint_avx2<uint_X_t, uint_X_dest_t, false, false, true> : sse4 ? convert_uint_sse41<uint_X_t, uint_X_dest_t, false, false, true> : convert_uint_c<uint_X_t, uint_X_dest_t, false, false, true>; \
        conv_function_chroma = avx2 ? convert_uint_avx2<uint_X_t, uint_X_dest_t, true, false, true> : sse4 ? convert_uint_sse41<uint_X_t, uint_X_dest_t, true, false, true> : convert_uint_c<uint_X_t, uint_X_dest_t, true, false, true>; \
      } \
      else if (!fulls && !fulld) { \
        conv_function = avx2 ? convert_uint_avx2<uint_X_t, uint_X_dest_t, false, false, false> : sse4 ? convert_uint_sse41<uint_X_t, uint_X_dest_t, false, false, false> : convert_uint_c<uint_X_t, uint_X_dest_t, false, false, false>; \
        conv_function_chroma = avx2 ? convert_uint_avx2<uint_X_t, uint_X_dest_t, true, false, false> : sse4 ? convert_uint_sse41<uint_X_t, uint_X_dest_t, true, false, false> : convert_uint_c<uint_X_t, uint_X_dest_t, true, false, false>; \
      }
#endif

   // all variations of fulls fulld byte/word source/target
  switch (target_bitdepth)
  {
  case 8:
    if (source_bitdepth == 8) {
      convert_uintN_to_uintN_functions(uint8_t, uint8_t);
    }
    else {
      convert_uintN_to_uintN_functions(uint16_t, uint8_t);
    }
    break;
  default:
    // 10-16 bits
    if (source_bitdepth == 8)
    {
      convert_uintN_to_uintN_functions(uint8_t, uint16_t);
    }
    else {
      convert_uintN_to_uintN_functions(uint16_t, uint16_t);
    }
    break;
  }

#undef convert_uintN_to_uintN_functions
}

ConvertBits::ConvertBits(PClip _child, const int _dither_mode, const int _target_bitdepth, bool _truerange,
  int _ColorRange_src, int _ColorRange_dest,
  int _dither_bitdepth, IScriptEnvironment* env) :
  GenericVideoFilter(_child),
  conv_function(nullptr), conv_function_chroma(nullptr), conv_function_a(nullptr),
  target_bitdepth(_target_bitdepth), dither_mode(_dither_mode), dither_bitdepth(_dither_bitdepth),
  fulls(false), fulld(false), truerange(_truerange)
{

  pixelsize = vi.ComponentSize();
  bits_per_pixel = vi.BitsPerComponent();
  format_change_only = false;

#ifdef INTEL_INTRINSICS
  const bool sse2 = !!(env->GetCPUFlags() & CPUF_SSE2);
  const bool sse4 = !!(env->GetCPUFlags() & CPUF_SSE4_1);
  const bool avx2 = !!(env->GetCPUFlags() & CPUF_AVX2);
#endif

  // full or limited decision
  // dest: if undefined, use src
  if (_ColorRange_dest != ColorRange_e::AVS_RANGE_LIMITED && _ColorRange_dest != ColorRange_e::AVS_RANGE_FULL) {
    _ColorRange_dest = _ColorRange_src;
  }
  //
  fulls = _ColorRange_src == ColorRange_e::AVS_RANGE_FULL;
  fulld = _ColorRange_dest == ColorRange_e::AVS_RANGE_FULL;

  if (!truerange) {
    if ((target_bitdepth == 8 || target_bitdepth == 32) && pixelsize == 2)
      bits_per_pixel = 16;
    if (target_bitdepth > 8 && target_bitdepth <= 16 && (bits_per_pixel == 8 || bits_per_pixel == 32))
      target_bitdepth = 16;
    if (target_bitdepth > 8 && target_bitdepth <= 16 && bits_per_pixel > 8 && bits_per_pixel <= 16)
      format_change_only = true;
  }

  if (bits_per_pixel <= 16 && target_bitdepth <= 16)
  {
    // get basic non-dithered versions
#ifdef INTEL_INTRINSICS
    get_convert_uintN_to_uintN_functions(bits_per_pixel, target_bitdepth, fulls, fulld, sse2, sse4, avx2, conv_function, conv_function_chroma, conv_function_a);
#else
    get_convert_uintN_to_uintN_functions(bits_per_pixel, target_bitdepth, fulls, fulld, conv_function, conv_function_chroma, conv_function_a);
#endif
    if (target_bitdepth <= bits_per_pixel) {
      // dither is only down
      if (dither_mode == 0) {
        // ordered dither
#ifdef INTEL_INTRINSICS
        get_convert_uintN_to_uintN_ordered_dither_functions(bits_per_pixel, target_bitdepth, fulls, fulld, sse2, sse4, avx2, conv_function, conv_function_chroma);
#else
        get_convert_uintN_to_uintN_ordered_dither_functions(bits_per_pixel, target_bitdepth, fulls, fulld, conv_function, conv_function_chroma);
#endif
      }
      else if (dither_mode == 1) {
        // Floyd, no SIMD there
        get_convert_uintN_to_uintN_floyd_dither_functions(bits_per_pixel, target_bitdepth, fulls, fulld, conv_function, conv_function_chroma);
      }
    }
  }

  // 32->8-16 bit
  if (bits_per_pixel == 32 && target_bitdepth <= 16) {
#ifdef INTEL_INTRINSICS
    get_convert_32_to_uintN_functions(target_bitdepth, fulls, fulld, sse2, sse4, avx2, conv_function, conv_function_chroma, conv_function_a);
#else
    get_convert_32_to_uintN_functions(target_bitdepth, fulls, fulld, conv_function, conv_function_chroma, conv_function_a);
#endif
  }

  // 8-32->32
  if (target_bitdepth == 32) {
    if (bits_per_pixel <= 16) // 8-16->32 bit
      get_convert_uintN_to_float_functions(bits_per_pixel, fulls, fulld, conv_function, conv_function_chroma, conv_function_a);
    else
      get_convert_float_to_float_functions(fulls, fulld, conv_function, conv_function_chroma, conv_function_a);
  }

  // Set VideoInfo
  if (target_bitdepth == 8) {
    if (vi.NumComponents() == 1)
      vi.pixel_type = VideoInfo::CS_Y8;
    else if (vi.IsYV411())
      vi.pixel_type = VideoInfo::CS_YV411;
    else if (vi.Is420() || vi.IsYV12())
      vi.pixel_type = vi.IsYUVA() ? VideoInfo::CS_YUVA420 : VideoInfo::CS_YV12;
    else if (vi.Is422())
      vi.pixel_type = vi.IsYUVA() ? VideoInfo::CS_YUVA422 : VideoInfo::CS_YV16;
    else if (vi.Is444())
      vi.pixel_type = vi.IsYUVA() ? VideoInfo::CS_YUVA444 : VideoInfo::CS_YV24;
    else if (vi.IsRGB48() || vi.IsRGB24())
      vi.pixel_type = VideoInfo::CS_BGR24;
    else if (vi.IsRGB64() || vi.IsRGB32())
      vi.pixel_type = VideoInfo::CS_BGR32;
    else if (vi.IsPlanarRGB())
      vi.pixel_type = VideoInfo::CS_RGBP;
    else if (vi.IsPlanarRGBA())
      vi.pixel_type = VideoInfo::CS_RGBAP;
    else
      env->ThrowError("ConvertTo8bit: unsupported color space");

    return;
  }
  else if (target_bitdepth > 8 && target_bitdepth <= 16) {
    // set output vi format
    if (vi.IsRGB24() || vi.IsRGB48()) {
      vi.pixel_type = VideoInfo::CS_BGR48;
    }
    else if (vi.IsRGB32() || vi.IsRGB64()) {
      vi.pixel_type = VideoInfo::CS_BGR64;
    }
    else {
      // Y or YUV(A) or PlanarRGB(A)
      if (vi.IsYV12()) // YV12 can have an exotic compatibility constant
        vi.pixel_type = VideoInfo::CS_YV12; // override for known
      int new_bitdepth_bits;
      switch (target_bitdepth) {
      case 8: new_bitdepth_bits = VideoInfo::CS_Sample_Bits_8; break;
      case 10: new_bitdepth_bits = VideoInfo::CS_Sample_Bits_10; break;
      case 12: new_bitdepth_bits = VideoInfo::CS_Sample_Bits_12; break;
      case 14: new_bitdepth_bits = VideoInfo::CS_Sample_Bits_14; break;
      case 16: new_bitdepth_bits = VideoInfo::CS_Sample_Bits_16; break;
      case 32: new_bitdepth_bits = VideoInfo::CS_Sample_Bits_32; break;
      }
      vi.pixel_type = (vi.pixel_type & ~VideoInfo::CS_Sample_Bits_Mask) | new_bitdepth_bits;
    }
    return;
  }
  else if (target_bitdepth == 32) {
    if (vi.NumComponents() == 1)
      vi.pixel_type = VideoInfo::CS_Y32;
    else if (vi.Is420())
      vi.pixel_type = vi.IsYUVA() ? VideoInfo::CS_YUVA420PS : VideoInfo::CS_YUV420PS;
    else if (vi.Is422())
      vi.pixel_type = vi.IsYUVA() ? VideoInfo::CS_YUVA422PS : VideoInfo::CS_YUV422PS;
    else if (vi.Is444())
      vi.pixel_type = vi.IsYUVA() ? VideoInfo::CS_YUVA444PS : VideoInfo::CS_YUV444PS;
    else if (vi.IsPlanarRGB())
      vi.pixel_type = VideoInfo::CS_RGBPS;
    else if (vi.IsPlanarRGBA())
      vi.pixel_type = VideoInfo::CS_RGBAPS;
    else
      env->ThrowError("ConvertToFloat: unsupported color space");

    return;
  }

  env->ThrowError("ConvertBits: unsupported target bit-depth (%d)", target_bitdepth);

}

AVSValue __cdecl ConvertBits::Create(AVSValue args, void* user_data, IScriptEnvironment* env) {
  PClip clip = args[0].AsClip();
  //0   1        2        3         4         5           6
  //c[bits]i[truerange]b[dither]i[dither_bits]i[fulls]b[fulld]b

  const VideoInfo &vi = clip->GetVideoInfo();

  int create_param = (int)reinterpret_cast<intptr_t>(user_data);

  // when converting from/true 10-16 bit formats, truerange=false indicates bitdepth of 16 bits regardless of the 10-12-14 bit format
  // FIXME: stop supporting this parameter (a workaround in the dawn of hbd?)
  bool assume_truerange = args[2].AsBool(true); // n/a for non planar formats

  int source_bitdepth = vi.BitsPerComponent();
  // default comes from old legacy To8,To16,ToFloat functions
  // or the clip's actual bit depth
  const int default_target_bitdepth = create_param == 0 ? source_bitdepth : create_param;
  int target_bitdepth = args[1].AsInt(default_target_bitdepth); // "bits" parameter
  int dither_bitdepth = args[4].AsInt(target_bitdepth); // "dither_bits" parameter

  if(target_bitdepth!=8 && target_bitdepth!=10 && target_bitdepth!=12 && target_bitdepth!=14 && target_bitdepth!=16 && target_bitdepth!=32)
    env->ThrowError("ConvertBits: invalid bit depth: %d", target_bitdepth);

  if(create_param == 8 && target_bitdepth !=8)
    env->ThrowError("ConvertTo8Bit: invalid bit depth: %d", target_bitdepth);
  if(create_param == 32 && target_bitdepth !=32)
    env->ThrowError("ConvertToFloat: invalid bit depth: %d", target_bitdepth);
  if(create_param == 16 && (target_bitdepth == 8 || target_bitdepth ==32))
    env->ThrowError("ConvertTo16bit: invalid bit depth: %d", target_bitdepth);

  if (args[2].Defined()) {
    if (!vi.IsPlanar())
      env->ThrowError("ConvertBits: truerange specified for non-planar source");
  }

  // retrieve full/limited
  int ColorRange_src;
  int ColorRange_dest;
  if (args[5].Defined())
    ColorRange_src = args[5].AsBool() ? ColorRange_e::AVS_RANGE_FULL : ColorRange_e::AVS_RANGE_LIMITED;
  else
    ColorRange_src = -1; // undefined. A frame property may override
  if (args[6].Defined())
    ColorRange_dest = args[6].AsBool() ? ColorRange_e::AVS_RANGE_FULL : ColorRange_e::AVS_RANGE_LIMITED;
  else
    ColorRange_dest = -1; // undefined. A frame property or ColorRange_src may override
  if (ColorRange_src != ColorRange_e::AVS_RANGE_LIMITED && ColorRange_src != ColorRange_e::AVS_RANGE_FULL) {
    // try getting frame props if parameter is not specified
    auto frame0 = clip->GetFrame(0, env);
    const AVSMap* props = env->getFramePropsRO(frame0);
    if (env->propNumElements(props, "_ColorRange") > 0) {
      ColorRange_src = env->propGetInt(props, "_ColorRange", 0, nullptr); // fixme: range check
    }
    else {
      // no param, no frame property -> rgb is full others are limited
      ColorRange_src = vi.IsRGB() ? ColorRange_e::AVS_RANGE_FULL : ColorRange_e::AVS_RANGE_LIMITED;
    }
  }
  // cr_dest = cr_source if not specified
  if (ColorRange_dest != ColorRange_e::AVS_RANGE_LIMITED && ColorRange_dest != ColorRange_e::AVS_RANGE_FULL) {
    ColorRange_dest = ColorRange_src;
  }
  bool fulls = ColorRange_src == ColorRange_e::AVS_RANGE_FULL;
  bool fulld = ColorRange_dest == ColorRange_e::AVS_RANGE_FULL;


  int dither_type = args[3].AsInt(-1);
  bool dither_defined = args[3].Defined();
  if(dither_defined && dither_type != 1 && dither_type != 0 && dither_type != -1)
    env->ThrowError("ConvertBits: invalid dither type parameter. Only -1 (disabled), 0 (ordered dither) or 1 (Floyd-S) is allowed");

  if (dither_type >= 0) {
    if (source_bitdepth < target_bitdepth)
      env->ThrowError("ConvertBits: dithering is allowed only for scale down");
    if (dither_bitdepth > target_bitdepth)
      env->ThrowError("ConvertBits: dither_bits must be <= target bitdepth");
    if (target_bitdepth == 32)
      env->ThrowError("ConvertBits: dithering is not allowed into 32 bit float target");
  }

  // 3.7.1 t25
  // Unfortunately 32 bit float dithering is not implemented, thus we convert to 16 bit 
  // intermediate clip
  if (source_bitdepth == 32 && (dither_type == 0 || dither_type == 1)) {
    // c[bits]i[truerange]b[dither]i[dither_bits]i[fulls]b[fulld]b

    source_bitdepth = 16;
    // solving ordered dither maximum bit depth difference of 8 problem
    // by automatic preconversion
    if (dither_type == 0 && source_bitdepth - dither_bitdepth > 8) {
      source_bitdepth = dither_bitdepth + 8;
      if (source_bitdepth % 2)
        source_bitdepth--; // must be even
    }
    
    AVSValue new_args[7] = { clip, source_bitdepth, true, -1 /* no dither */, AVSValue() /*dither_bits*/, fulls, fulld };
    clip = env->Invoke("ConvertBits", AVSValue(new_args, 7)).AsClip();

    clip = env->Invoke("Cache", AVSValue(clip)).AsClip();
    // and now the source range becomes the previous target
    fulls = fulld;
    ColorRange_src = ColorRange_dest;
  }

  // solving ordered dither maximum bit depth difference of 8 problem
  // by automatic preconversion
  if (source_bitdepth <= 16 && dither_type == 0 && source_bitdepth - dither_bitdepth > 8) {
    // c[bits]i[truerange]b[dither]i[dither_bits]i[fulls]b[fulld]b
    source_bitdepth = dither_bitdepth + 8;
    if (source_bitdepth % 2)
      source_bitdepth--; // must be even

    AVSValue new_args[7] = { clip, source_bitdepth, true, -1 /* no dither */, AVSValue() /*dither_bits*/, fulls, fulld };
    clip = env->Invoke("ConvertBits", AVSValue(new_args, 7)).AsClip();

    clip = env->Invoke("Cache", AVSValue(clip)).AsClip();
    // and now the source range becomes the previous target
    fulls = fulld;
    ColorRange_src = ColorRange_dest;
  }

  if (source_bitdepth == dither_bitdepth)
    dither_type = -1; // ignore dithering

  if(dither_type == 0) {
    if (dither_bitdepth < 1 || dither_bitdepth > 16)
      env->ThrowError("ConvertBits: ordered dither: invalid dither_bits specified (1-16 allowed)");

    // this error message cannot appear if the automatic bit depth reducing conversions above are done
    if(source_bitdepth - dither_bitdepth > 8)
      env->ThrowError("ConvertBits: dither_bits cannot differ with more than 8 bits from source");
  }

  // floyd
  if (dither_type == 1) {
    if (dither_bitdepth < 1 || dither_bitdepth > 16)
      env->ThrowError("ConvertBits: Floyd-S: invalid dither_bits specified (1-16 allowed)");
  }

  // no change -> return unmodified if no transform required
  if (source_bitdepth == target_bitdepth) { // 10->10 .. 16->16
    if((dither_type < 0 || dither_bitdepth == target_bitdepth) && fulls == fulld)
      return clip;
  }

  // YUY2 conversion is limited
  if (vi.IsYUY2()) {
    if (target_bitdepth != 8)
      env->ThrowError("ConvertBits: YUY2 input must stay in 8 bits");
  }

  if (vi.IsYV411()) {
    if (target_bitdepth != 8)
      env->ThrowError("ConvertBits: YV411 input must stay in 8 bits");
  }

  // packed RGB conversion is limited
  if (vi.IsRGB() && !vi.IsPlanar()) {
    if (target_bitdepth != 8 && target_bitdepth != 16)
      env->ThrowError("ConvertBits: invalid bit-depth for packed RGB formats, only 8 or 16 possible");
  }

    // remark
    // source_10_bit.ConvertTo16bit(truerange=true)  : upscale range
    // source_10_bit.ConvertTo16bit(truerange=false) : leaves data, only format conversion
    // source_10_bit.ConvertTo16bit(bits=12,truerange=true)  : upscale range from 10 to 12
    // source_10_bit.ConvertTo16bit(bits=12,truerange=false) : leaves data, only format conversion
    // source_16_bit.ConvertTo16bit(bits=10, truerange=true)  : downscale range
    // source_16_bit.ConvertTo16bit(bits=10, truerange=false) : leaves data, only format conversion

  // yuy2 is autoconverted to/from YV16. fulls-fulld and dither to lower bit depths are supported
  bool need_convert_yuy2 = vi.IsYUY2();
  // for dither, planar rgb conversion happens
  bool need_convert_24 = vi.IsRGB24() && dither_type >= 0;
  bool need_convert_32 = vi.IsRGB32() && dither_type >= 0;
  bool need_convert_48 = vi.IsRGB48() && dither_type >= 0;
  bool need_convert_64 = vi.IsRGB64() && dither_type >= 0;

  // convert to planar on the fly if dither was asked
  if (need_convert_24 || need_convert_48) {
    AVSValue new_args[1] = { clip };
    clip = env->Invoke("ConvertToPlanarRGB", AVSValue(new_args, 1)).AsClip();
  }
  else if (need_convert_32 || need_convert_64) {
    AVSValue new_args[1] = { clip };
    clip = env->Invoke("ConvertToPlanarRGBA", AVSValue(new_args, 1)).AsClip();
  }
  else if (need_convert_yuy2) {
    AVSValue new_args[1] = { clip };
    clip = env->Invoke("ConvertToYV16", AVSValue(new_args, 1)).AsClip();
  }

  AVSValue result = new ConvertBits(clip, dither_type, target_bitdepth, assume_truerange, ColorRange_src, ColorRange_dest, dither_bitdepth, env);

  // convert back to packed rgb from planar on the fly
  if (need_convert_24 || need_convert_48) {
    AVSValue new_args[1] = { result };
    if(target_bitdepth == 8)
      result = env->Invoke("ConvertToRGB24", AVSValue(new_args, 1)).AsClip();
    else
      result = env->Invoke("ConvertToRGB48", AVSValue(new_args, 1)).AsClip();
  } else if (need_convert_32 || need_convert_64) {
    AVSValue new_args[1] = { result };
    if (target_bitdepth == 8)
      result = env->Invoke("ConvertToRGB32", AVSValue(new_args, 1)).AsClip();
    else
      result = env->Invoke("ConvertToRGB64", AVSValue(new_args, 1)).AsClip();
  }
  else if (need_convert_yuy2) {
    AVSValue new_args[1] = { result };
    result = env->Invoke("ConvertToYUY2", AVSValue(new_args, 1)).AsClip();
  }

  return result;
}


PVideoFrame __stdcall ConvertBits::GetFrame(int n, IScriptEnvironment* env) {
  PVideoFrame src = child->GetFrame(n, env);

  if (format_change_only)
  {
    // for 10-16 bit: simple format override in constructor
    return src;
  }

  PVideoFrame dst = env->NewVideoFrameP(vi, &src);

  auto props = env->getFramePropsRW(dst);
  update_ColorRange(props, fulld ? ColorRange_e::AVS_RANGE_FULL : ColorRange_e::AVS_RANGE_LIMITED, env);

  if(vi.IsPlanar())
  {
    int planes_y[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
    int planes_r[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
    int *planes = (vi.IsYUV() || vi.IsYUVA()) ? planes_y : planes_r;
    for (int p = 0; p < vi.NumComponents(); ++p) {
      const int plane = planes[p];
      if (plane == PLANAR_A) {
        if (conv_function_a == nullptr)
          env->BitBlt(dst->GetWritePtr(plane), dst->GetPitch(plane), src->GetReadPtr(plane), src->GetPitch(plane), src->GetRowSize(plane), src->GetHeight(plane));
        else
          conv_function_a(src->GetReadPtr(plane), dst->GetWritePtr(plane),
            src->GetRowSize(plane), src->GetHeight(plane),
            src->GetPitch(plane), dst->GetPitch(plane),
            bits_per_pixel, target_bitdepth, dither_bitdepth
          );
      }
      else if (conv_function == nullptr)
        env->BitBlt(dst->GetWritePtr(plane), dst->GetPitch(plane), src->GetReadPtr(plane), src->GetPitch(plane), src->GetRowSize(plane), src->GetHeight(plane));
      else {
        const bool chroma = (plane == PLANAR_U || plane == PLANAR_V);
        if (chroma && conv_function_chroma != nullptr)
          // 32bit float and 8-16 when full-range involved needs separate signed-aware conversion
          conv_function_chroma(src->GetReadPtr(plane), dst->GetWritePtr(plane),
            src->GetRowSize(plane), src->GetHeight(plane),
            src->GetPitch(plane), dst->GetPitch(plane),
            bits_per_pixel, target_bitdepth, dither_bitdepth);
        else
          conv_function(src->GetReadPtr(plane), dst->GetWritePtr(plane),
            src->GetRowSize(plane), src->GetHeight(plane),
            src->GetPitch(plane), dst->GetPitch(plane),
            bits_per_pixel, target_bitdepth, dither_bitdepth);
      }
    }
  }
  else {
    // packed RGBs
    conv_function(src->GetReadPtr(), dst->GetWritePtr(),
      src->GetRowSize(), src->GetHeight(),
      src->GetPitch(), dst->GetPitch(),
      bits_per_pixel, target_bitdepth, dither_bitdepth);
  }
  return dst;
}
