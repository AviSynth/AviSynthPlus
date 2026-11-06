// Avisynth v2.5.  Copyright 2002-2009 Ben Rudiak-Gould et al.
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

#include "convert_bits.h"
#ifdef INTEL_INTRINSICS
#include "intel/convert_bits_sse.h"
#include "intel/convert_bits_avx2.h"
#elif defined(NEON_INTRINSICS)
#include "aarch64/convert_bits_neon.h"
#endif
#include "convert_helper.h"

#include <avs/alignment.h>
#include <avs/minmax.h>
#include <avs/config.h>
#include <tuple>
#include <map>
#include <algorithm>
#include <vector>
#include <cstring>

#ifdef AVS_WINDOWS
#include <avs/win.h>
#else
#include <avs/posix.h>
#endif

// for odd dither bit differences, we still take even size but
// correction values are halved (shifted by 1)

// repeated 8x for sse size 16
const BYTE dither2x2a_data[4] = {
  0, 1,
  1, 0,
};
// cycle: 2
alignas(16) const BYTE dither2x2a_data_sse2[2 * 16] = {
  0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
  1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0
};

// e.g. 10->8 bits
// repeated 8x for sse size 16
const BYTE dither2x2_data[4] = {
  0, 2,
  3, 1,
};
// cycle: 2
alignas(16) const BYTE dither2x2_data_sse2[2 * 16] = {
  0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2,
  3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1
};

// e.g. 8->5 bits
const BYTE dither4x4a_data[16] = {
    0,  4,  1,  5,
    6,  2,  7,  3,
    1,  5,  0,  4,
    7,  3,  6,  2
};
// cycle: 4
alignas(16) const BYTE dither4x4a_data_sse2[4 * 16] = {
    0,  4,  1,  5,  0,  4,  1,  5,  0,  4,  1,  5,  0,  4,  1,  5,
    6,  2,  7,  3,  6,  2,  7,  3,  6,  2,  7,  3,  6,  2,  7,  3,
    1,  5,  0,  4,  1,  5,  0,  4,  1,  5,  0,  4,  1,  5,  0,  4,
    7,  3,  6,  2,  7,  3,  6,  2,  7,  3,  6,  2,  7,  3,  6,  2,
};

// e.g. 12->8 bits
const BYTE dither4x4_data[16] = {
    0,  8,  2, 10,
  12,  4, 14,  6,
    3, 11,  1,  9,
  15,  7, 13,  5
};
// cycle: 4
alignas(16) const BYTE dither4x4_data_sse2[4 * 16] = {
    0,  8,  2, 10,  0,  8,  2, 10,  0,  8,  2, 10,  0,  8,  2, 10,
  12,  4, 14,  6, 12,  4, 14,  6, 12,  4, 14,  6, 12,  4, 14,  6,
    3, 11,  1,  9,  3, 11,  1,  9,  3, 11,  1,  9,  3, 11,  1,  9,
  15,  7, 13,  5, 15,  7, 13,  5, 15,  7, 13,  5, 15,  7, 13,  5
};

// e.g. 14->9 bits
extern const BYTE dither8x8a_data[8][8] = {
  { 0, 16,  4, 20,  1, 17,  5, 21}, /* 8x8 Bayer ordered dithering pattern */
  {24,  8, 28, 12, 25,  9, 29, 13},
  { 6, 22,  2, 18,  7, 23,  3, 19},
  {30, 14, 26, 10, 31, 15, 27, 11},
  { 1, 17,  5, 21,  0, 16,  4, 20},
  {25,  9, 29, 13, 24,  8, 28, 12},
  { 7, 23,  3, 19,  6, 22,  2, 18},
  {31, 15, 27, 11, 30, 14, 26, 10}
};
// cycle: 8
alignas(16) const BYTE dither8x8a_data_sse2[8][16] = {
  {  0, 16,  4, 20,  1, 17,  5, 21,  0, 16,  4, 20,  1, 17,  5, 21 },
  { 24,  8, 28, 12, 25,  9, 29, 13, 24,  8, 28, 12, 25,  9, 29, 13 },
  {  6, 22,  2, 18,  7, 23,  3, 19,  6, 22,  2, 18,  7, 23,  3, 19 },
  { 30, 14, 26, 10, 31, 15, 27, 11, 30, 14, 26, 10, 31, 15, 27, 11 },
  {  1, 17,  5, 21,  0, 16,  4, 20,  1, 17,  5, 21,  0, 16,  4, 20 },
  { 25,  9, 29, 13, 24,  8, 28, 12, 25,  9, 29, 13, 24,  8, 28, 12 },
  {  7, 23,  3, 19,  6, 22,  2, 18,  7, 23,  3, 19,  6, 22,  2, 18 },
  { 31, 15, 27, 11, 30, 14, 26, 10, 31, 15, 27, 11, 30, 14, 26, 10 }
};

// e.g. 14->8 bits
extern const BYTE dither8x8_data[8][8] = {
  { 0, 32,  8, 40,  2, 34, 10, 42},
  {48, 16, 56, 24, 50, 18, 58, 26},
  {12, 44,  4, 36, 14, 46,  6, 38},
  {60, 28, 52, 20, 62, 30, 54, 22},
  { 3, 35, 11, 43,  1, 33,  9, 41},
  {51, 19, 59, 27, 49, 17, 57, 25},
  {15, 47,  7, 39, 13, 45,  5, 37},
  {63, 31, 55, 23, 61, 29, 53, 21}
};
// cycle: 8
alignas(16) const BYTE dither8x8_data_sse2[8][16] = {
  {  0, 32,  8, 40,  2, 34, 10, 42,  0, 32,  8, 40,  2, 34, 10, 42 },
  { 48, 16, 56, 24, 50, 18, 58, 26, 48, 16, 56, 24, 50, 18, 58, 26 },
  { 12, 44,  4, 36, 14, 46,  6, 38, 12, 44,  4, 36, 14, 46,  6, 38 },
  { 60, 28, 52, 20, 62, 30, 54, 22, 60, 28, 52, 20, 62, 30, 54, 22 },
  {  3, 35, 11, 43,  1, 33,  9, 41,  3, 35, 11, 43,  1, 33,  9, 41 },
  { 51, 19, 59, 27, 49, 17, 57, 25, 51, 19, 59, 27, 49, 17, 57, 25 },
  { 15, 47,  7, 39, 13, 45,  5, 37, 15, 47,  7, 39, 13, 45,  5, 37 },
  { 63, 31, 55, 23, 61, 29, 53, 21, 63, 31, 55, 23, 61, 29, 53, 21 }
};

// e.g. 16->9 or 8->1 bits
// cycle: 16x. No special 16 byte sse2
alignas(16) const BYTE dither16x16a_data[16][16] = {
  {   0, 96, 24,120,  6,102, 30,126,  1, 97, 25,121,  7,103, 31,127 },
  {  64, 32, 88, 56, 70, 38, 94, 62, 65, 33, 89, 57, 71, 39, 95, 63 },
  {  16,112,  8,104, 22,118, 14,110, 17,113,  9,105, 23,119, 15,111 },
  {  80, 48, 72, 40, 86, 54, 78, 46, 81, 49, 73, 41, 87, 55, 79, 47 },
  {   4,100, 28,124,  2, 98, 26,122,  5,101, 29,125,  3, 99, 27,123 },
  {  68, 36, 92, 60, 66, 34, 90, 58, 69, 37, 93, 61, 67, 35, 91, 59 },
  {  20,116, 12,108, 18,114, 10,106, 21,117, 13,109, 19,115, 11,107 },
  {  84, 52, 76, 44, 82, 50, 74, 42, 85, 53, 77, 45, 83, 51, 75, 43 },
  {   1, 97, 25,121,  7,103, 31,127,  0, 96, 24,120,  6,102, 30,126 },
  {  75, 33, 89, 57, 71, 39, 95, 63, 64, 32, 88, 56, 70, 38, 94, 62 },
  {  17,113,  9,105, 23,119, 15,111, 16,112,  8,104, 22,118, 14,110 },
  {  81, 49, 73, 41, 87, 55, 79, 47, 80, 48, 72, 40, 86, 54, 78, 46 },
  {   5,101, 29,125,  3, 99, 27,123,  4,100, 28,124,  2, 98, 26,122 },
  {  69, 37, 93, 61, 67, 35, 91, 59, 68, 36, 92, 60, 66, 34, 90, 58 },
  {  21,117, 13,109, 19,115, 11,107, 20,116, 12,108, 18,114, 10,106 },
  {  85, 53, 77, 45, 83, 51, 75, 43, 84, 52, 76, 44, 82, 50, 74, 42 }
};

// 16->8
// cycle: 16x. No special 16 byte sse2
alignas(16) const BYTE dither16x16_data[16][16] = {
  {   0,192, 48,240, 12,204, 60,252,  3,195, 51,243, 15,207, 63,255 },
  { 128, 64,176,112,140, 76,188,124,131, 67,179,115,143, 79,191,127 },
  {  32,224, 16,208, 44,236, 28,220, 35,227, 19,211, 47,239, 31,223 },
  { 160, 96,144, 80,172,108,156, 92,163, 99,147, 83,175,111,159, 95 },
  {   8,200, 56,248,  4,196, 52,244, 11,203, 59,251,  7,199, 55,247 },
  { 136, 72,184,120,132, 68,180,116,139, 75,187,123,135, 71,183,119 },
  {  40,232, 24,216, 36,228, 20,212, 43,235, 27,219, 39,231, 23,215 },
  { 168,104,152, 88,164,100,148, 84,171,107,155, 91,167,103,151, 87 },
  {   2,194, 50,242, 14,206, 62,254,  1,193, 49,241, 13,205, 61,253 },
  { 130, 66,178,114,142, 78,190,126,129, 65,177,113,141, 77,189,125 },
  {  34,226, 18,210, 46,238, 30,222, 33,225, 17,209, 45,237, 29,221 },
  { 162, 98,146, 82,174,110,158, 94,161, 97,145, 81,173,109,157, 93 },
  {  10,202, 58,250,  6,198, 54,246,  9,201, 57,249,  5,197, 53,245 },
  { 138, 74,186,122,134, 70,182,118,137, 73,185,121,133, 69,181,117 },
  {  42,234, 26,218, 38,230, 22,214, 41,233, 25,217, 37,229, 21,213 },
  { 170,106,154, 90,166,102,150, 86,169,105,153, 89,165,101,149, 85 }
};


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
  case 1: matrix = reinterpret_cast<const BYTE*>(odd_diff ? dither2x2a_data : dither2x2_data); break;
  case 2: matrix = reinterpret_cast<const BYTE*>(odd_diff ? dither4x4a_data : dither4x4_data); break;
  case 3: matrix = reinterpret_cast<const BYTE*>(odd_diff ? dither8x8a_data : dither8x8_data); break;
  case 4: matrix = reinterpret_cast<const BYTE*>(odd_diff ? dither16x16a_data : dither16x16_data); break;
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

// optimization helper: TEMPLATE_DITHER_BIT_DIFF if not <0 then hold value for frequently used differences from 16->8
// 2nd helper: TEMPLATE_LOW_DITHER_BITDEPTH
// 3rd helper: source_bitdepth_special
template<typename pixel_t_s, typename pixel_t_d, bool chroma, bool fulls, bool fulld, int TEMPLATE_DITHER_BIT_DIFF, bool TEMPLATE_LOW_DITHER_BITDEPTH, int SOURCE_BITDEPTH_SPECIAL>
static void do_convert_uint_floyd_c(const BYTE* srcp8, BYTE* dstp8, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth)
{
  if constexpr (SOURCE_BITDEPTH_SPECIAL > 0) {
    // called with >0 values only for special cases like 16 to 8, 16 to 10 and 10 to 8
    // Hugely helps the optimizers
    source_bitdepth = SOURCE_BITDEPTH_SPECIAL;
  }
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

  std::vector<int> error_ptr_safe(1 + src_width + 1); // accumulated errors
  int *error_ptr = &error_ptr_safe[1];

  const int ROUNDER = 1 << (DITHER_BIT_DIFF - 1); // rounding
  const int source_max = (1 << source_bitdepth) - 1;
  //-----------------------
  bits_conv_constants d;
  get_bits_conv_constants(d, chroma, fulls, fulld, source_bitdepth, source_bitdepth);

  auto dst_offset_plus_round = d.dst_offset + 0.5f;
  constexpr auto src_pixel_min = 0;
  const auto src_pixel_max = source_max;
  const float mul_factor_backfromlowdither = (float)max_pixel_value_target / max_pixel_value_dithered;

  int nextError = 0; // zero

  for (int y = 0; y < src_height; y++)
  {
    // serpentine forward
    if ((y & 1) == 0)
    {
      for (int x = 0; x < src_width; x++)
      {
        int err = nextError;
        int src_pixel = srcp[x];

        if constexpr (fulls != fulld) {
          const float val = (src_pixel - d.src_offset_i) * d.mul_factor + dst_offset_plus_round;
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
        diffuse_floyd<1>(err, nextError, &error_ptr[x]);
      }
    }
    else {
      // serpentine backward
      for (int x = src_width - 1; x >= 0; --x)
      {
        int err = nextError;
        int src_pixel = srcp[x];

        if constexpr (fulls != fulld) {
          const float val = (src_pixel - d.src_offset_i) * d.mul_factor + dst_offset_plus_round;
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
        diffuse_floyd<-1>(err, nextError, &error_ptr[x]);
      }
    }
    dstp += dst_pitch;
    srcp += src_pitch;
  }
}


template<typename pixel_t_s, typename pixel_t_d, bool chroma, bool fulls, bool fulld>
static void convert_uint_floyd_c(const BYTE* srcp8, BYTE* dstp8, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth)
{
  const int dither_bit_diff = source_bitdepth - dither_target_bitdepth;
  const bool low_dither_bitdepth = dither_target_bitdepth < 8;
  // extra internal template makes it quicker for ordinary non-artistic cases
  // do not make templates for all 1-16 target bit combinations
  if (low_dither_bitdepth) {
    do_convert_uint_floyd_c<pixel_t_s, pixel_t_d, chroma, fulls, fulld, -1, true, -1>(srcp8, dstp8, src_rowsize, src_height, src_pitch, dst_pitch, source_bitdepth, target_bitdepth, dither_target_bitdepth);
  }
  else {
    if (target_bitdepth == dither_target_bitdepth) {
      // Specifically when source_bitdepth is known as well, it hugely helps optimization
      // We treat special use cases 10->8, 16->10 and 16->8
      switch (dither_bit_diff) {
      case 2: // e.g. 10->8
        if (source_bitdepth == 10)
          do_convert_uint_floyd_c<pixel_t_s, pixel_t_d, chroma, fulls, fulld, 2, false, 10>(srcp8, dstp8, src_rowsize, src_height, src_pitch, dst_pitch, source_bitdepth, target_bitdepth, dither_target_bitdepth);
        else
          do_convert_uint_floyd_c<pixel_t_s, pixel_t_d, chroma, fulls, fulld, 2, false, -1>(srcp8, dstp8, src_rowsize, src_height, src_pitch, dst_pitch, source_bitdepth, target_bitdepth, dither_target_bitdepth);
        break;
      case 4: // e.g. 12->8
        do_convert_uint_floyd_c<pixel_t_s, pixel_t_d, chroma, fulls, fulld, 4, false, -1>(srcp8, dstp8, src_rowsize, src_height, src_pitch, dst_pitch, source_bitdepth, target_bitdepth, dither_target_bitdepth);
        break;
      case 6: // e.g. 16->10, 14->8
        // prevent invalid templates to generate
        // like do_convert_uint_floyd_c<unsigned short,unsigned char,0,0,1,6,0,16> which would do 16->8 but dither to 10 bit.
        if constexpr (sizeof(pixel_t_s) == 2 && sizeof(pixel_t_d) == 2) {
          if (source_bitdepth == 16)
            do_convert_uint_floyd_c<pixel_t_s, pixel_t_d, chroma, fulls, fulld, 6, false, 16>(srcp8, dstp8, src_rowsize, src_height, src_pitch, dst_pitch, source_bitdepth, target_bitdepth, dither_target_bitdepth);
          else
            do_convert_uint_floyd_c<pixel_t_s, pixel_t_d, chroma, fulls, fulld, 6, false, -1>(srcp8, dstp8, src_rowsize, src_height, src_pitch, dst_pitch, source_bitdepth, target_bitdepth, dither_target_bitdepth);
        } else
          do_convert_uint_floyd_c<pixel_t_s, pixel_t_d, chroma, fulls, fulld, 6, false, -1>(srcp8, dstp8, src_rowsize, src_height, src_pitch, dst_pitch, source_bitdepth, target_bitdepth, dither_target_bitdepth);
        break;
      case 8: // e.g. 16->8
        if (sizeof(pixel_t_s) == 2 && source_bitdepth == 16)
          do_convert_uint_floyd_c<pixel_t_s, pixel_t_d, chroma, fulls, fulld, 8, false, 16>(srcp8, dstp8, src_rowsize, src_height, src_pitch, dst_pitch, source_bitdepth, target_bitdepth, dither_target_bitdepth);
        else
          do_convert_uint_floyd_c<pixel_t_s, pixel_t_d, chroma, fulls, fulld, 8, false, -1>(srcp8, dstp8, src_rowsize, src_height, src_pitch, dst_pitch, source_bitdepth, target_bitdepth, dither_target_bitdepth);
        break;
      default: // difference is more than 8 or exotic dither to less than 8 bits, we accept 10-15% speed minus
        do_convert_uint_floyd_c<pixel_t_s, pixel_t_d, chroma, fulls, fulld, -1, false, -1>(srcp8, dstp8, src_rowsize, src_height, src_pitch, dst_pitch, source_bitdepth, target_bitdepth, dither_target_bitdepth);
      }
    }
    else {
      do_convert_uint_floyd_c<pixel_t_s, pixel_t_d, chroma, fulls, fulld, -1, false, -1>(srcp8, dstp8, src_rowsize, src_height, src_pitch, dst_pitch, source_bitdepth, target_bitdepth, dither_target_bitdepth);
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

  if (target_bitdepth == source_bitdepth) {
    // Shortcut: same bit depth, limited to limited: plain copy
    // The reduce-range path would compute 1 << (0 - 1) and for 8 bits, read bytes as words.
    for (int y = 0; y < src_height; y++) {
      std::memcpy(dstp0, srcp0, src_width * sizeof(pixel_t_d));
      dstp0 += dst_pitch;
      srcp0 += src_pitch;
    }
    return;
  }

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

    const pixel_t_s source_max = (1 << source_bitdepth) - 1;

    for (int y = 0; y < src_height; y++)
    {
      const pixel_t_s* AVS_RESTRICT s = srcp0;
      pixel_t_d* AVS_RESTRICT d = dstp0;

      for (int x = 0; x < src_width; x++) {
        int val = static_cast<int>(s[x]) + round;
        if (val > source_max) val = source_max; // no saturated add in C
        d[x] = static_cast<pixel_t_d>(val >> shift_bits);
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

  if constexpr (fulls == fulld) {
    // identity: copy, shortcut (x - offset) * 1.0f + offset
    for (int y = 0; y < src_height; y++) {
      std::memcpy(dstp0, srcp0, src_width * sizeof(float));
      dstp0 += dst_pitch;
      srcp0 += src_pitch;
    }
    return;
  }

  //-----------------------
  bits_conv_constants d;
  get_bits_conv_constants(d, chroma, fulls, fulld, source_bitdepth, target_bitdepth);

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
#ifdef INTEL_INTRINSICS
  bool sse2, bool sse4, bool avx2,
#endif
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

#define convert_uintN_to_float_functions_avx2(uint_X_t) \
      conv_function_a = convert_uintN_to_float_avx2<uint_X_t, false, true, true>; /* full-full */ \
      if (fulls && fulld) { \
        conv_function = convert_uintN_to_float_avx2<uint_X_t, false, true, true>; \
        conv_function_chroma = convert_uintN_to_float_avx2<uint_X_t, true, true, true>; \
      } \
      else if (fulls && !fulld) { \
        conv_function = convert_uintN_to_float_avx2<uint_X_t, false, true, false>; \
        conv_function_chroma = convert_uintN_to_float_avx2<uint_X_t, true, true, false>; \
      } \
      else if (!fulls && fulld) { \
        conv_function = convert_uintN_to_float_avx2<uint_X_t, false, false, true>; \
        conv_function_chroma = convert_uintN_to_float_avx2<uint_X_t, true, false, true>; \
      } \
      else if (!fulls && !fulld) { \
        conv_function = convert_uintN_to_float_avx2<uint_X_t, false, false, false>; \
        conv_function_chroma = convert_uintN_to_float_avx2<uint_X_t, true, false, false>; \
      }

  switch (bits_per_pixel) {
  case 8: 
    convert_uintN_to_float_functions(uint8_t);
    break;
  default: // 10-16 bits
    convert_uintN_to_float_functions(uint16_t);
    break;
  }

#ifdef INTEL_INTRINSICS
  if (avx2) {
    switch (bits_per_pixel) {
    case 8:
      convert_uintN_to_float_functions_avx2(uint8_t);
      break;
    default: // 10-16 bits
      convert_uintN_to_float_functions_avx2(uint16_t);
      break;
    }
  }
#endif 


#undef convert_uintN_to_float_functions
}

static void get_convert_uintN_to_uintN_ordered_dither_functions(int source_bitdepth, int target_bitdepth, bool fulls, bool fulld,
#ifdef INTEL_INTRINSICS
  bool sse2, bool sse4, bool avx2,
#elif defined(NEON_INTRINSICS)
  bool neon,
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
#elif defined(NEON_INTRINSICS)
#undef convert_uintN_to_uintN_ordered_dither_functions
// dither has no "conv_function_a"
#define convert_uintN_to_uintN_ordered_dither_functions(uint_X_t, uint_X_dest_t) \
      if (fulls && fulld) { \
        conv_function = neon ? convert_ordered_dither_uint_neon<uint_X_t, uint_X_dest_t, false, true, true> : convert_ordered_dither_uint_c<uint_X_t, uint_X_dest_t, false, true, true>; \
        conv_function_chroma = neon ? convert_ordered_dither_uint_neon<uint_X_t, uint_X_dest_t, true, true, true> : convert_ordered_dither_uint_c<uint_X_t, uint_X_dest_t, true, true, true>; \
      } \
      else if (fulls && !fulld) { \
        conv_function = neon ? convert_ordered_dither_uint_neon<uint_X_t, uint_X_dest_t, false, true, false> : convert_ordered_dither_uint_c<uint_X_t, uint_X_dest_t, false, true, false>; \
        conv_function_chroma = neon ? convert_ordered_dither_uint_neon<uint_X_t, uint_X_dest_t, true, true, false> : convert_ordered_dither_uint_c<uint_X_t, uint_X_dest_t, true, true, false>; \
      } \
      else if (!fulls && fulld) { \
        conv_function = neon ? convert_ordered_dither_uint_neon<uint_X_t, uint_X_dest_t, false, false, true> : convert_ordered_dither_uint_c<uint_X_t, uint_X_dest_t, false, false, true>; \
        conv_function_chroma = neon ? convert_ordered_dither_uint_neon<uint_X_t, uint_X_dest_t, true, false, true> : convert_ordered_dither_uint_c<uint_X_t, uint_X_dest_t, true, false, true>; \
      } \
      else if (!fulls && !fulld) { \
        conv_function = neon ? convert_ordered_dither_uint_neon<uint_X_t, uint_X_dest_t, false, false, false> : convert_ordered_dither_uint_c<uint_X_t, uint_X_dest_t, false, false, false>; \
        conv_function_chroma = neon ? convert_ordered_dither_uint_neon<uint_X_t, uint_X_dest_t, true, false, false> : convert_ordered_dither_uint_c<uint_X_t, uint_X_dest_t, true, false, false>; \
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

#undef convert_uintN_to_uintN_floyd_dither_functions
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

void get_convert_any_bits_functions(int dither_mode, int source_bitdepth, int target_bitdepth, bool fulls, bool fulld,
#ifdef INTEL_INTRINSICS
  bool sse2, bool sse4, bool avx2,
#elif defined(NEON_INTRINSICS)
  bool neon,
#endif
  BitDepthConvFuncPtr& conv_function, BitDepthConvFuncPtr& conv_function_chroma, BitDepthConvFuncPtr& conv_function_a)
{

  if (source_bitdepth <= 16 && target_bitdepth <= 16)
  {
    // get basic non-dithered versions
#ifdef INTEL_INTRINSICS
    get_convert_uintN_to_uintN_functions(source_bitdepth, target_bitdepth, fulls, fulld, sse2, sse4, avx2, conv_function, conv_function_chroma, conv_function_a);
#else
    get_convert_uintN_to_uintN_functions(source_bitdepth, target_bitdepth, fulls, fulld, conv_function, conv_function_chroma, conv_function_a);
#endif
    if (target_bitdepth <= source_bitdepth) {
      // dither is only down
      if (dither_mode == 0) {
        // ordered dither
#ifdef INTEL_INTRINSICS
        get_convert_uintN_to_uintN_ordered_dither_functions(source_bitdepth, target_bitdepth, fulls, fulld, sse2, sse4, avx2, conv_function, conv_function_chroma);
#elif defined(NEON_INTRINSICS)
        get_convert_uintN_to_uintN_ordered_dither_functions(source_bitdepth, target_bitdepth, fulls, fulld, neon, conv_function, conv_function_chroma);
#else
        get_convert_uintN_to_uintN_ordered_dither_functions(source_bitdepth, target_bitdepth, fulls, fulld, conv_function, conv_function_chroma);
#endif
      }
      else if (dither_mode == 1) {
        // Floyd, no SIMD there
        get_convert_uintN_to_uintN_floyd_dither_functions(source_bitdepth, target_bitdepth, fulls, fulld, conv_function, conv_function_chroma);
      }
    }
  }

  // 32->8-16 bit
  if (source_bitdepth == 32 && target_bitdepth <= 16) {
#ifdef INTEL_INTRINSICS
    get_convert_32_to_uintN_functions(target_bitdepth, fulls, fulld, sse2, sse4, avx2, conv_function, conv_function_chroma, conv_function_a);
#else
    get_convert_32_to_uintN_functions(target_bitdepth, fulls, fulld, conv_function, conv_function_chroma, conv_function_a);
#endif
  }

  // 8-32->32
  if (target_bitdepth == 32) {
    if (source_bitdepth <= 16) // 8-16->32 bit
#ifdef INTEL_INTRINSICS
      get_convert_uintN_to_float_functions(source_bitdepth, fulls, fulld, sse2, sse4, avx2, conv_function, conv_function_chroma, conv_function_a);
#else
      get_convert_uintN_to_float_functions(source_bitdepth, fulls, fulld, conv_function, conv_function_chroma, conv_function_a);
#endif
    else
      get_convert_float_to_float_functions(fulls, fulld, conv_function, conv_function_chroma, conv_function_a);
  }
}


ConvertBits::ConvertBits(PClip _child, const int _dither_mode, const int _target_bitdepth, bool _truerange,
  int _ColorRange_src, int _ColorRange_dest,
  int _dither_bitdepth, IScriptEnvironment* env) :
  GenericVideoFilter(_child),
  conv_function(nullptr), conv_function_chroma(nullptr), conv_function_a(nullptr),
  target_bitdepth(_target_bitdepth), dither_mode(_dither_mode), dither_bitdepth(_dither_bitdepth),
  fulls(false), fulld(false),
  dynamic_range(_ColorRange_src == -2),
  dest_fulld_pinned(false), dest_fulld_pinned_value(false),
  truerange(_truerange)
{

  pixelsize = vi.ComponentSize();
  bits_per_pixel = vi.BitsPerComponent();
  format_change_only = false;

#ifdef INTEL_INTRINSICS
  const bool sse2 = !!(env->GetCPUFlags() & CPUF_SSE2);
  const bool sse4 = !!(env->GetCPUFlags() & CPUF_SSE4_1);
  const bool avx2 = !!(env->GetCPUFlags() & CPUF_AVX2);
#elif defined(NEON_INTRINSICS)
  const bool neon = !!(env->GetCPUFlags() & CPUF_ARM_NEON);
#endif

  if (!dynamic_range) {
    // static across the whole clip
    // full or limited decision
    // _ColorRange_src is never -1 (undefined) here: static mode means fulls was given explicitly.
    // _ColorRange_dest is already resolved to src in Create when fulld is omitted.
    // dest: if undefined, use src
    if (_ColorRange_dest != ColorRange_Compat_e::AVS_COLORRANGE_LIMITED && _ColorRange_dest != ColorRange_Compat_e::AVS_COLORRANGE_FULL) {
      _ColorRange_dest = _ColorRange_src;
    }
    //
    fulls = _ColorRange_src == ColorRange_Compat_e::AVS_COLORRANGE_FULL;
    fulld = _ColorRange_dest == ColorRange_Compat_e::AVS_COLORRANGE_FULL;
  }
  else {
    // fulls is resolved per-frame in GetFrame.
    // fulld either
    // - mirrors fulls per-frame (auto, when _ColorRange_dest is undefined)
    // - stays pinned to a fixed value given explicitly
    if (_ColorRange_dest == ColorRange_Compat_e::AVS_COLORRANGE_LIMITED || _ColorRange_dest == ColorRange_Compat_e::AVS_COLORRANGE_FULL) {
      dest_fulld_pinned = true;
      dest_fulld_pinned_value = _ColorRange_dest == ColorRange_Compat_e::AVS_COLORRANGE_FULL;
    }
  }

  if (!truerange) {
    if ((target_bitdepth == 8 || target_bitdepth == 32) && pixelsize == 2)
      bits_per_pixel = 16;
    if (target_bitdepth > 8 && target_bitdepth <= 16 && (bits_per_pixel == 8 || bits_per_pixel == 32))
      target_bitdepth = 16;
    if (target_bitdepth > 8 && target_bitdepth <= 16 && bits_per_pixel > 8 && bits_per_pixel <= 16)
      format_change_only = true;
  }

  if (!dynamic_range) {
#ifdef INTEL_INTRINSICS
    get_convert_any_bits_functions(dither_mode, bits_per_pixel, target_bitdepth, fulls, fulld, sse2, sse4, avx2, conv_function, conv_function_chroma, conv_function_a);
#elif defined(NEON_INTRINSICS)
    get_convert_any_bits_functions(dither_mode, bits_per_pixel, target_bitdepth, fulls, fulld, neon, conv_function, conv_function_chroma, conv_function_a);
#else
    get_convert_any_bits_functions(dither_mode, bits_per_pixel, target_bitdepth, fulls, fulld, conv_function, conv_function_chroma, conv_function_a);
#endif
  }
  else {
    // per-frame range support, pre-fill fulls-fulld combination dispatch table
    for (int fs = 0; fs < 2; ++fs) {
      for (int fd = 0; fd < 2; ++fd) {
#ifdef INTEL_INTRINSICS
        get_convert_any_bits_functions(dither_mode, bits_per_pixel, target_bitdepth, fs != 0, fd != 0, sse2, sse4, avx2,
          conv_function_table[fs][fd], conv_function_chroma_table[fs][fd], conv_function_a_table[fs][fd]);
#elif defined(NEON_INTRINSICS)
        get_convert_any_bits_functions(dither_mode, bits_per_pixel, target_bitdepth, fs != 0, fd != 0, neon,
          conv_function_table[fs][fd], conv_function_chroma_table[fs][fd], conv_function_a_table[fs][fd]);
#else
        get_convert_any_bits_functions(dither_mode, bits_per_pixel, target_bitdepth, fs != 0, fd != 0,
          conv_function_table[fs][fd], conv_function_chroma_table[fs][fd], conv_function_a_table[fs][fd]);
#endif
      }
    }
  }

  // Set VideoInfo
  if (vi.IsPlanar()) {
    // Y, YUV(A) or PlanarRGB(A): apply bitmask arithmetic CS_GENERIC_xxx | CS_Sample_Bits_N
    if (vi.IsYV12()) // YV12 can have an exotic compatibility constant (I420/IYUV, U-plane-first)
      vi.pixel_type = VideoInfo::CS_YV12; // override for uniform format handling
    int new_bitdepth_bits;
    switch (target_bitdepth) {
    case 8: new_bitdepth_bits = VideoInfo::CS_Sample_Bits_8; break;
    case 10: new_bitdepth_bits = VideoInfo::CS_Sample_Bits_10; break;
    case 12: new_bitdepth_bits = VideoInfo::CS_Sample_Bits_12; break;
    case 14: new_bitdepth_bits = VideoInfo::CS_Sample_Bits_14; break;
    case 16: new_bitdepth_bits = VideoInfo::CS_Sample_Bits_16; break;
    case 32: new_bitdepth_bits = VideoInfo::CS_Sample_Bits_32; break;
    default:
      env->ThrowError("ConvertBits: unsupported target bit-depth (%d)", target_bitdepth);
      return;
    }
    vi.pixel_type = (vi.pixel_type & ~VideoInfo::CS_Sample_Bits_Mask) | new_bitdepth_bits;
    return;
  }

  // Packed RGB (24/32/48/64): only 8/16-bit targets are valid, checked in Create
  if (vi.IsRGB24() || vi.IsRGB48()) {
    vi.pixel_type = target_bitdepth == 8 ? VideoInfo::CS_BGR24 : VideoInfo::CS_BGR48;
    return;
  }
  if (vi.IsRGB32() || vi.IsRGB64()) {
    vi.pixel_type = target_bitdepth == 8 ? VideoInfo::CS_BGR32 : VideoInfo::CS_BGR64;
    return;
  }

  env->ThrowError("ConvertBits: unsupported color space");

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
  // fulls omitted => source-range detection is deferred to per-frame (ConvertBits::GetFrame).
  // fulld
  // - may be pinned to a fixed value given explicitly
  // - left to automatically mirror the per-frame source range (also omitted).
  // No GetFrame(0) happens in Create: fulls is either given or resolved per-frame.
  // Dithering (ordered/Floyd) may build chained intermediate conversions (float source, or
  // bit-depth gap > 8 for ordered dither, expand back after low dither_bits). Their structure depends
  // only on bit depths, so in deferred mode they are deferred as well: each stage reads the
  // _ColorRange written by the previous one.
  const bool defer_range = !args[5].Defined(); // fulls omitted => deferred range detection

  int ColorRange_src;
  int ColorRange_dest;
  // ColorRange_src is either given by fulls, or -2 (deferred).
  // The former -1 (undefined) state, which triggered a one-time frame#0 _ColorRange lookup here,
  // no longer exists: an omitted fulls always means deferred per-frame detection, thus GetFrame(0) was removed.
  if (defer_range)
    ColorRange_src = -2; // per-frame in ConvertBits::GetFrame
  else
    ColorRange_src = args[5].AsBool() ? ColorRange_Compat_e::AVS_COLORRANGE_FULL : ColorRange_Compat_e::AVS_COLORRANGE_LIMITED;
  if (args[6].Defined()) // fulld
    ColorRange_dest = args[6].AsBool() ? ColorRange_Compat_e::AVS_COLORRANGE_FULL : ColorRange_Compat_e::AVS_COLORRANGE_LIMITED;
  else
    ColorRange_dest = -1; // undefined: mirrors ColorRange_src (static: right below; deferred: per-frame in GetFrame)
  // cr_dest = cr_source if not specified
  // n/a when deferred range (-2), GetFrame mirrors the per-frame detected source property
  if (ColorRange_src != -2 && ColorRange_dest != ColorRange_Compat_e::AVS_COLORRANGE_LIMITED && ColorRange_dest != ColorRange_Compat_e::AVS_COLORRANGE_FULL) {
    ColorRange_dest = ColorRange_src;
  }
  bool fulls = ColorRange_src == ColorRange_Compat_e::AVS_COLORRANGE_FULL;
  bool fulld = ColorRange_dest == ColorRange_Compat_e::AVS_COLORRANGE_FULL;


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

  // Range arguments for the chained intermediate ConvertBits stages below.
  // fulls: deferred => omitted/Undefined AVSValue() (will use per-frame from the previous stage's _ColorRange)
  //        else the resolved source range of that stage (fulls for the pre-steps, fulld for the expand-back)
  // fulld: deferred => passed as given in parameter (undefined/fixed)
  //        else based on pre-established ColorRange_dest
  auto stage_fulls_arg = [&](bool resolved_fulls) { return defer_range ? AVSValue() : AVSValue(resolved_fulls); };
  const AVSValue stage_fulld_arg = defer_range ? args[6] : AVSValue(fulld);

  // Dither works on planar data only: packed RGB is converted to planar RGB(A) on the fly and
  // back at the end. This must happen before the dither pre-steps below: they invoke ConvertBits with
  // intermediate bit depths (e.g. 16->10 for dither_bits=2), which do not exist for packed RGB.
  // Only when dithering will really happen: packed RGB is 8 or 16 bit, dither is ignored when
  // dither_bits equals the source bit depth.
  const bool packed_rgb_dither = dither_type >= 0 && source_bitdepth != dither_bitdepth;
  const bool need_convert_24 = vi.IsRGB24() && packed_rgb_dither;
  const bool need_convert_32 = vi.IsRGB32() && packed_rgb_dither;
  const bool need_convert_48 = vi.IsRGB48() && packed_rgb_dither;
  const bool need_convert_64 = vi.IsRGB64() && packed_rgb_dither;

  if (need_convert_24 || need_convert_48) {
    AVSValue new_args[1] = { clip };
    clip = env->Invoke("ConvertToPlanarRGB", AVSValue(new_args, 1)).AsClip();
  }
  else if (need_convert_32 || need_convert_64) {
    AVSValue new_args[1] = { clip };
    clip = env->Invoke("ConvertToPlanarRGBA", AVSValue(new_args, 1)).AsClip();
  }

  // 32 bit float dithering is not implemented, thus we convert to 16 bit intermediate clip
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
    
    AVSValue new_args[7] = { clip, source_bitdepth, true, -1 /* no dither */, AVSValue() /*dither_bits*/, stage_fulls_arg(fulls), stage_fulld_arg };
    clip = env->Invoke("ConvertBits", AVSValue(new_args, 7)).AsClip();

    clip = env->Invoke("Cache", AVSValue(clip)).AsClip();
    // and now the source range becomes the previous target
    // (deferred: stays -2, the next stage reads the _ColorRange written by this one)
    fulls = fulld;
    if (!defer_range)
      ColorRange_src = ColorRange_dest;
  }

  // solving ordered dither maximum bit depth difference of 8 problem
  // by automatic preconversion
  if (source_bitdepth <= 16 && dither_type == 0 && source_bitdepth - dither_bitdepth > 8) {
    // c[bits]i[truerange]b[dither]i[dither_bits]i[fulls]b[fulld]b
    source_bitdepth = dither_bitdepth + 8;
    if (source_bitdepth % 2)
      source_bitdepth--; // must be even

    AVSValue new_args[7] = { clip, source_bitdepth, true, -1 /* no dither */, AVSValue() /*dither_bits*/, stage_fulls_arg(fulls), stage_fulld_arg };
    clip = env->Invoke("ConvertBits", AVSValue(new_args, 7)).AsClip();

    clip = env->Invoke("Cache", AVSValue(clip)).AsClip();
    // and now the source range becomes the previous target
    // (deferred: stays -2, the next stage reads the _ColorRange written by this one)
    fulls = fulld;
    if (!defer_range)
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
    if (defer_range) {
      // When fulls/fulld are unresolved, but an undefined parameter (unpinned) fulld mirrors the
      // per-frame fulls => no range change
      // When fulld is given (pinned), it needs the per-frame decision, so we don't return here. param 6: fulld
      if ((dither_type < 0 || dither_bitdepth == target_bitdepth) && !args[6].Defined())
        return clip;
    }
    else if ((dither_type < 0 || dither_bitdepth == target_bitdepth) && fulls == fulld)
      return clip;
  }

  // YUY2 conversion is limited
  if (vi.IsYUY2()) {
    if (target_bitdepth != 8)
      env->ThrowError("ConvertBits: YUY2 input must stay in 8 bits");
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
  // (8 bit only: no dither pre-step can happen, so it is fine to convert here, after the no-op check)
  bool need_convert_yuy2 = vi.IsYUY2();
  if (need_convert_yuy2) {
    AVSValue new_args[1] = { clip };
    clip = env->Invoke("ConvertToYV16", AVSValue(new_args, 1)).AsClip();
  }

  // For large gaps originated from a larger dither_bits, we created an intermediate clip,
  // because the algorithm can handle only a maximum of 8 bits difference.
  // In the callee get_convert_any_bits_functions only selects a dither implementation when
  // target_bitdepth <= source_bitdepth ("dither is only down"). The automatic
  // preconversion would result in a (possibly) reduced source_bitdepth that has been
  // changed above, and is _lower_ than the still original target_bitdepth.
  // Case: 16->16 with dither_bits=1 gets preconverted to an 8-bit intermediate, but still
  // has to reach 16-bit output. So we do dither in-place at source_bitdepth first, then simply expand
  // it back up.
  const bool need_expand_back_after_dither = (dither_type >= 0 && target_bitdepth > source_bitdepth);
  const int final_target_bitdepth = target_bitdepth;
  if (need_expand_back_after_dither)
    target_bitdepth = source_bitdepth; // temporarily change in order to do the dithering effectively

  AVSValue result = new ConvertBits(clip, dither_type, target_bitdepth, assume_truerange, ColorRange_src, ColorRange_dest, dither_bitdepth, env);

  if (need_expand_back_after_dither) {
    // scale back w/o dithering
    AVSValue new_args[7] = { result, final_target_bitdepth, true, -1 /* no dither */, AVSValue() /*dither_bits*/, stage_fulls_arg(fulld), stage_fulld_arg };
    result = env->Invoke("ConvertBits", AVSValue(new_args, 7)).AsClip();
    target_bitdepth = final_target_bitdepth; // back to the originally requested bit-depth
  }

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
    env->MakeWritable(&src);
    src->AmendPixelType(vi.pixel_type);
    return src;
  }

  // Static range (fulls given explicitly): fulls/fulld and the conversion functions were fixed in the constructor.
  // (There is no frame#0 based range detection in ConvertBits anymore.
  // An omitted fulls always means dynamic_range.
  bool actual_fulld = fulld;
  BitDepthConvFuncPtr actual_conv_function = conv_function;
  BitDepthConvFuncPtr actual_conv_function_chroma = conv_function_chroma;
  BitDepthConvFuncPtr actual_conv_function_a = conv_function_a;

  if (dynamic_range) {
    // per-frame range detection based on actual source frame props
    const AVSMap* src_props = env->getFramePropsRO(src);
    int established_ColorRange_src;
    if (env->propNumElements(src_props, "_Range") > 0)
      established_ColorRange_src = range_value_flip((int)env->propGetIntSaturated(src_props, "_Range", 0, nullptr));
    else
      // no frame property: RGB full, others limited (usual default)
      established_ColorRange_src = vi.IsRGB() ? ColorRange_Compat_e::AVS_COLORRANGE_FULL : ColorRange_Compat_e::AVS_COLORRANGE_LIMITED;

    const int actual_fulls = established_ColorRange_src == ColorRange_Compat_e::AVS_COLORRANGE_FULL ? 1 : 0;
    // dest mirrors src (e.g. simple bitdepth-only conversion keeping the range) unless pinned
    // to a fixed value. Use case: we can normalize output range regardless of per-frame source range
    const int actual_fulld_idx = dest_fulld_pinned ? (dest_fulld_pinned_value ? 1 : 0) : actual_fulls;
    actual_fulld = actual_fulld_idx != 0;

    // dispatch table set in the constructor
    actual_conv_function = conv_function_table[actual_fulls][actual_fulld_idx];
    actual_conv_function_chroma = conv_function_chroma_table[actual_fulls][actual_fulld_idx];
    actual_conv_function_a = conv_function_a_table[actual_fulls][actual_fulld_idx];
  }

  PVideoFrame dst = env->NewVideoFrameP(vi, &src);

  auto props = env->getFramePropsRW(dst);
  update_ColorRange(props, actual_fulld ? ColorRange_Compat_e::AVS_COLORRANGE_FULL : ColorRange_Compat_e::AVS_COLORRANGE_LIMITED, env);

  if(vi.IsPlanar())
  {
    int planes_yuva[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
    int planes_ya[2]   = { PLANAR_Y, PLANAR_A };
    int planes_r[4]    = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
    int *planes = vi.IsYA() ? planes_ya : (vi.IsYUV() || vi.IsYUVA()) ? planes_yuva : planes_r;
    for (int p = 0; p < vi.NumComponents(); ++p) {
      const int plane = planes[p];
      if (plane == PLANAR_A) {
        if (actual_conv_function_a == nullptr)
          env->BitBlt(dst->GetWritePtr(plane), dst->GetPitch(plane), src->GetReadPtr(plane), src->GetPitch(plane), src->GetRowSize(plane), src->GetHeight(plane));
        else
          actual_conv_function_a(src->GetReadPtr(plane), dst->GetWritePtr(plane),
            src->GetRowSize(plane), src->GetHeight(plane),
            src->GetPitch(plane), dst->GetPitch(plane),
            bits_per_pixel, target_bitdepth, dither_bitdepth
          );
      }
      else if (actual_conv_function == nullptr)
        env->BitBlt(dst->GetWritePtr(plane), dst->GetPitch(plane), src->GetReadPtr(plane), src->GetPitch(plane), src->GetRowSize(plane), src->GetHeight(plane));
      else {
        const bool chroma = (plane == PLANAR_U || plane == PLANAR_V);
        if (chroma && actual_conv_function_chroma != nullptr)
          // 32bit float and 8-16 when full-range involved needs separate signed-aware conversion
          actual_conv_function_chroma(src->GetReadPtr(plane), dst->GetWritePtr(plane),
            src->GetRowSize(plane), src->GetHeight(plane),
            src->GetPitch(plane), dst->GetPitch(plane),
            bits_per_pixel, target_bitdepth, dither_bitdepth);
        else
          actual_conv_function(src->GetReadPtr(plane), dst->GetWritePtr(plane),
            src->GetRowSize(plane), src->GetHeight(plane),
            src->GetPitch(plane), dst->GetPitch(plane),
            bits_per_pixel, target_bitdepth, dither_bitdepth);
      }
    }
  }
  else {
    // packed RGBs
    actual_conv_function(src->GetReadPtr(), dst->GetWritePtr(),
      src->GetRowSize(), src->GetHeight(),
      src->GetPitch(), dst->GetPitch(),
      bits_per_pixel, target_bitdepth, dither_bitdepth);
  }
  return dst;
}
