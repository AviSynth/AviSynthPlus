// Avisynth v2.5.  Copyright 2002 Ben Rudiak-Gould et al.
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


#include <cstdint>
#include <cstdlib>
#include <cmath>

// Avisynth filter: general convolution
// by Richard Berg (avisynth-dev@richardberg.net)
// adapted from General Convolution 3D for VDub by Gunnar Thalin (guth@home.se)
// avs+: all color spaces, 8-32 bits, 7x7, 9x9, luma/chroma/alpha option

#include "convolution.h"
#include "../core/internal.h"

#include <regex>
#include <iostream>
#include <iterator>
#include <string>

/********************************************************************
***** Declare index of new filters for Avisynth's filter engine *****
********************************************************************/

extern const AVSFunction Convolution_filters[] = {
// Please when adding parameters try not to break the legacy order - IanB July 2004
  { "GeneralConvolution", BUILTIN_FUNC_PREFIX, "c[bias]f[matrix]s[divisor]f[auto]b[luma]b[chroma]b[alpha]b", GeneralConvolution::Create },
    /**
      * GeneralConvolution(PClip clip, int divisor=1, int bias=0, string matrix)
      * clip     =  input video
      * bias     =  additive bias to adjust the total output intensity
      * matrix   =  the kernel (3x3 or 5x5).  any kind of whitespace is ok, see example
      * divisor  =  divides the output of the convolution (calculated before adding bias)
      * auto     =  automaticly scale the result based on the sum of the matrix elements
      * luma     =  apply on luma (if applicable) avs+
      * chroma   =  apply on chroma (if applicable) avs+
      * alpha    =  apply on alpha (if applicable e.g. RGB32 converted to planar RGBA) avs+
      *
      * clip.GeneralConvolution(matrix = "1 2 3
      *                                   4 5 6
      *                                   7 8 9")
     **/

  { 0 }
};

template<int mi, int ma>
AVS_FORCEINLINE int static_clip(int value) {
  if (value < mi) {
    return mi;
  }
  if (value > ma) {
    return ma;
  }
  return value;
}

/*****************************************
****** General Convolution 2D filter *****
*****************************************/

// safe_int_t is int or int64_t. Int is faster but for larger bitdepth int32 overflows
template<typename pixel_t, int bits_per_pixel, int matrix_size, typename safe_int_t>
static void do_conv_integer(BYTE* dstp8, int dst_pitch, const BYTE *srcp8, int src_pitch, int width, int height, const int *matrix, int iCountDiv, int iBias)
{
  pixel_t *dstp = reinterpret_cast<pixel_t *>(dstp8);
  const pixel_t *srcp = reinterpret_cast<const pixel_t *>(srcp8);
  dst_pitch /= sizeof(pixel_t);
  src_pitch /= sizeof(pixel_t);

  constexpr int limit = (matrix_size - 1) / 2; // +-1, +-2, +-3
  constexpr int max_pixel_value = (1 << bits_per_pixel) - 1;

  std::vector<const pixel_t*> src_lineptrs;
  src_lineptrs.resize(limit + height + limit);

  // prefill line pointers
  for (int y = -limit; y < height + limit; y++)
  {
    if (y < 0)
      src_lineptrs[y + limit] = srcp + src_pitch * 0;
    else if (y < height)
      src_lineptrs[y + limit] = srcp + src_pitch * y;
    else
      src_lineptrs[y + limit] = srcp + src_pitch * (height - 1);
  }

  std::vector<const pixel_t *> src_current_lineptrs(matrix_size); // +/-limit => 2*limit + 1

  constexpr safe_int_t rounder = 1 << (20 - 1);

  for (int y = 0; y < height; ++y) {
    // prefill current vertical line pointers
    for (int yy = -limit; yy <= limit; yy++) {
      src_current_lineptrs[yy + limit] = src_lineptrs[limit + (y + yy)];
    }
    int x = 0;
    // left area: check valid x
    for (; x < limit; x++)
    {
      safe_int_t sum = 0; // int or int64_t!!!
      const int *current_matrix = matrix + limit; // center of the matrix line
      for (int yy = 0; yy < matrix_size; yy++) { // 0..limit * 2 + 1
        const pixel_t *current_line = src_current_lineptrs[yy];
        for (int xx = -limit; xx <= limit; xx++) {
          int current_x = x + xx;
          if (current_x < 0) current_x = 0;
          else if (current_x >= width) current_x = width - 1;
          const int current_pixel = current_line[current_x];
          const int current_weight = current_matrix[xx];
          sum += current_pixel * current_weight;
        }
        current_matrix += matrix_size; // next matrix line
      }
      int result = (int)((sum * iCountDiv + rounder) >> 20) + iBias;
      dstp[x] = static_clip<0, max_pixel_value>(result);
    }
    // middle area: no x check: fast!
    for (; x < width - limit; x++) {
      safe_int_t sum = 0;
      const int *current_matrix = matrix + limit; // center of the matrix line
      for (int yy = 0; yy < matrix_size; yy++) { // 0..limit * 2 + 1
        const pixel_t *current_line = src_current_lineptrs[yy];
        // compilers are smart nowadays but let's help them
        if constexpr (matrix_size == 3) {
          sum +=
            current_line[x - 1] * current_matrix[-1] +
            current_line[x + 0] * current_matrix[0] +
            current_line[x + 1] * current_matrix[1];
        }
        else if constexpr (matrix_size == 5) {
          sum +=
            current_line[x - 2] * current_matrix[-2] +
            current_line[x - 1] * current_matrix[-1] +
            current_line[x + 0] * current_matrix[0] +
            current_line[x + 1] * current_matrix[1] +
            current_line[x + 2] * current_matrix[2];
        }
        else {
          for (int xx = -limit; xx <= limit; xx++) {
            int current_x = x + xx;
            // no checking!
            /*if (current_x < 0) current_x = 0;
            else if (current_x >= width) current_x = width - 1;*/
            const int current_pixel = current_line[current_x];
            const int current_weight = current_matrix[xx];
            sum += current_pixel * current_weight;
          }
        }
        current_matrix += matrix_size; // next matrix line
      }
      int result = (int)((sum * iCountDiv + rounder) >> 20) + iBias;
      dstp[x] = static_clip<0, max_pixel_value>(result);
    }
    // right area: check valid x
    for (; x < width; x++)
    {
      safe_int_t sum = 0;
      const int *current_matrix = matrix + limit; // center of the matrix line
      for (int yy = 0; yy < matrix_size; yy++) { // 0..limit * 2 + 1
        const pixel_t *current_line = src_current_lineptrs[yy];
        for (int xx = -limit; xx <= limit; xx++) {
          int current_x = x + xx;
          if (current_x < 0) current_x = 0;
          else if (current_x >= width) current_x = width - 1;
          const int current_pixel = current_line[current_x];
          const int current_weight = current_matrix[xx];
          sum += current_pixel * current_weight;
        }
        current_matrix += matrix_size; // next matrix line
      }
      int result = (int)((sum * iCountDiv + rounder) >> 20) + iBias;
      dstp[x] = static_clip<0, max_pixel_value>(result);
    }
    dstp += dst_pitch;
    srcp += src_pitch;
  }
}

// instantiate for 8 bit 3,5,7 and 9
template void do_conv_integer<uint8_t, 8, 3, int>(BYTE* dstp8, int dst_pitch, const BYTE *srcp8, int src_pitch, int width, int height, const int *matrix, int iCountDiv, int iBias);
template void do_conv_integer<uint8_t, 8, 5, int>(BYTE* dstp8, int dst_pitch, const BYTE *srcp8, int src_pitch, int width, int height, const int *matrix, int iCountDiv, int iBias);
template void do_conv_integer<uint8_t, 8, 7, int>(BYTE* dstp8, int dst_pitch, const BYTE *srcp8, int src_pitch, int width, int height, const int *matrix, int iCountDiv, int iBias);
template void do_conv_integer<uint8_t, 8, 9, int>(BYTE* dstp8, int dst_pitch, const BYTE *srcp8, int src_pitch, int width, int height, const int *matrix, int iCountDiv, int iBias);
template void do_conv_integer<uint8_t, 8, 3, int64_t>(BYTE* dstp8, int dst_pitch, const BYTE *srcp8, int src_pitch, int width, int height, const int *matrix, int iCountDiv, int iBias);
template void do_conv_integer<uint8_t, 8, 5, int64_t>(BYTE* dstp8, int dst_pitch, const BYTE *srcp8, int src_pitch, int width, int height, const int *matrix, int iCountDiv, int iBias);
template void do_conv_integer<uint8_t, 8, 7, int64_t>(BYTE* dstp8, int dst_pitch, const BYTE *srcp8, int src_pitch, int width, int height, const int *matrix, int iCountDiv, int iBias);
template void do_conv_integer<uint8_t, 8, 9, int64_t>(BYTE* dstp8, int dst_pitch, const BYTE *srcp8, int src_pitch, int width, int height, const int *matrix, int iCountDiv, int iBias);

template void do_conv_integer<uint16_t, 10, 3, int>(BYTE* dstp8, int dst_pitch, const BYTE *srcp8, int src_pitch, int width, int height, const int *matrix, int iCountDiv, int iBias);
template void do_conv_integer<uint16_t, 10, 5, int>(BYTE* dstp8, int dst_pitch, const BYTE *srcp8, int src_pitch, int width, int height, const int *matrix, int iCountDiv, int iBias);
template void do_conv_integer<uint16_t, 10, 7, int>(BYTE* dstp8, int dst_pitch, const BYTE *srcp8, int src_pitch, int width, int height, const int *matrix, int iCountDiv, int iBias);
template void do_conv_integer<uint16_t, 10, 9, int>(BYTE* dstp8, int dst_pitch, const BYTE *srcp8, int src_pitch, int width, int height, const int *matrix, int iCountDiv, int iBias);
template void do_conv_integer<uint16_t, 10, 3, int64_t>(BYTE* dstp8, int dst_pitch, const BYTE *srcp8, int src_pitch, int width, int height, const int *matrix, int iCountDiv, int iBias);
template void do_conv_integer<uint16_t, 10, 5, int64_t>(BYTE* dstp8, int dst_pitch, const BYTE *srcp8, int src_pitch, int width, int height, const int *matrix, int iCountDiv, int iBias);
template void do_conv_integer<uint16_t, 10, 7, int64_t>(BYTE* dstp8, int dst_pitch, const BYTE *srcp8, int src_pitch, int width, int height, const int *matrix, int iCountDiv, int iBias);
template void do_conv_integer<uint16_t, 10, 9, int64_t>(BYTE* dstp8, int dst_pitch, const BYTE *srcp8, int src_pitch, int width, int height, const int *matrix, int iCountDiv, int iBias);

template void do_conv_integer<uint16_t, 12, 3, int>(BYTE* dstp8, int dst_pitch, const BYTE *srcp8, int src_pitch, int width, int height, const int *matrix, int iCountDiv, int iBias);
template void do_conv_integer<uint16_t, 12, 5, int>(BYTE* dstp8, int dst_pitch, const BYTE *srcp8, int src_pitch, int width, int height, const int *matrix, int iCountDiv, int iBias);
template void do_conv_integer<uint16_t, 12, 7, int>(BYTE* dstp8, int dst_pitch, const BYTE *srcp8, int src_pitch, int width, int height, const int *matrix, int iCountDiv, int iBias);
template void do_conv_integer<uint16_t, 12, 9, int>(BYTE* dstp8, int dst_pitch, const BYTE *srcp8, int src_pitch, int width, int height, const int *matrix, int iCountDiv, int iBias);
template void do_conv_integer<uint16_t, 12, 3, int64_t>(BYTE* dstp8, int dst_pitch, const BYTE *srcp8, int src_pitch, int width, int height, const int *matrix, int iCountDiv, int iBias);
template void do_conv_integer<uint16_t, 12, 5, int64_t>(BYTE* dstp8, int dst_pitch, const BYTE *srcp8, int src_pitch, int width, int height, const int *matrix, int iCountDiv, int iBias);
template void do_conv_integer<uint16_t, 12, 7, int64_t>(BYTE* dstp8, int dst_pitch, const BYTE *srcp8, int src_pitch, int width, int height, const int *matrix, int iCountDiv, int iBias);
template void do_conv_integer<uint16_t, 12, 9, int64_t>(BYTE* dstp8, int dst_pitch, const BYTE *srcp8, int src_pitch, int width, int height, const int *matrix, int iCountDiv, int iBias);

template void do_conv_integer<uint16_t, 14, 3, int>(BYTE* dstp8, int dst_pitch, const BYTE *srcp8, int src_pitch, int width, int height, const int *matrix, int iCountDiv, int iBias);
template void do_conv_integer<uint16_t, 14, 5, int>(BYTE* dstp8, int dst_pitch, const BYTE *srcp8, int src_pitch, int width, int height, const int *matrix, int iCountDiv, int iBias);
template void do_conv_integer<uint16_t, 14, 7, int>(BYTE* dstp8, int dst_pitch, const BYTE *srcp8, int src_pitch, int width, int height, const int *matrix, int iCountDiv, int iBias);
template void do_conv_integer<uint16_t, 14, 9, int>(BYTE* dstp8, int dst_pitch, const BYTE *srcp8, int src_pitch, int width, int height, const int *matrix, int iCountDiv, int iBias);
template void do_conv_integer<uint16_t, 14, 3, int64_t>(BYTE* dstp8, int dst_pitch, const BYTE *srcp8, int src_pitch, int width, int height, const int *matrix, int iCountDiv, int iBias);
template void do_conv_integer<uint16_t, 14, 5, int64_t>(BYTE* dstp8, int dst_pitch, const BYTE *srcp8, int src_pitch, int width, int height, const int *matrix, int iCountDiv, int iBias);
template void do_conv_integer<uint16_t, 14, 7, int64_t>(BYTE* dstp8, int dst_pitch, const BYTE *srcp8, int src_pitch, int width, int height, const int *matrix, int iCountDiv, int iBias);
template void do_conv_integer<uint16_t, 14, 9, int64_t>(BYTE* dstp8, int dst_pitch, const BYTE *srcp8, int src_pitch, int width, int height, const int *matrix, int iCountDiv, int iBias);

template void do_conv_integer<uint16_t, 16, 3, int>(BYTE* dstp8, int dst_pitch, const BYTE *srcp8, int src_pitch, int width, int height, const int *matrix, int iCountDiv, int iBias);
template void do_conv_integer<uint16_t, 16, 5, int>(BYTE* dstp8, int dst_pitch, const BYTE *srcp8, int src_pitch, int width, int height, const int *matrix, int iCountDiv, int iBias);
template void do_conv_integer<uint16_t, 16, 7, int>(BYTE* dstp8, int dst_pitch, const BYTE *srcp8, int src_pitch, int width, int height, const int *matrix, int iCountDiv, int iBias);
template void do_conv_integer<uint16_t, 16, 9, int>(BYTE* dstp8, int dst_pitch, const BYTE *srcp8, int src_pitch, int width, int height, const int *matrix, int iCountDiv, int iBias);
template void do_conv_integer<uint16_t, 16, 3, int64_t>(BYTE* dstp8, int dst_pitch, const BYTE *srcp8, int src_pitch, int width, int height, const int *matrix, int iCountDiv, int iBias);
template void do_conv_integer<uint16_t, 16, 5, int64_t>(BYTE* dstp8, int dst_pitch, const BYTE *srcp8, int src_pitch, int width, int height, const int *matrix, int iCountDiv, int iBias);
template void do_conv_integer<uint16_t, 16, 7, int64_t>(BYTE* dstp8, int dst_pitch, const BYTE *srcp8, int src_pitch, int width, int height, const int *matrix, int iCountDiv, int iBias);
template void do_conv_integer<uint16_t, 16, 9, int64_t>(BYTE* dstp8, int dst_pitch, const BYTE *srcp8, int src_pitch, int width, int height, const int *matrix, int iCountDiv, int iBias);

template<int matrix_size>
static void do_conv_float(BYTE* dstp8, int dst_pitch, const BYTE *srcp8, int src_pitch, int width, int height, const float *matrix, float fCountDiv, float fBias)
{
  float *dstp = reinterpret_cast<float *>(dstp8);
  const float *srcp = reinterpret_cast<const float *>(srcp8);
  dst_pitch /= sizeof(float);
  src_pitch /= sizeof(float);

  constexpr int limit = (matrix_size - 1) / 2; // +-1, +-2, +-3

  std::vector<const float*> src_lineptrs;
  src_lineptrs.resize(limit + height + limit);

  // prefill line pointers
  for (int y = -limit; y < height + limit; y++)
  {
    if (y < 0)
      src_lineptrs[y + limit] = srcp + src_pitch * 0;
    else if (y < height)
      src_lineptrs[y + limit] = srcp + src_pitch * y;
    else
      src_lineptrs[y + limit] = srcp + src_pitch * (height - 1);
  }

  std::vector<const float *> src_current_lineptrs(matrix_size); // +/-limit => 2*limit + 1

  for (int y = 0; y < height; ++y) {
    // prefill current vertical line pointers
    for (int yy = -limit; yy <= limit; yy++) {
      src_current_lineptrs[yy + limit] = src_lineptrs[limit + (y + yy)];
    }
    int x = 0;
    // left area: check valid x
    for (; x < limit; x++)
    {
      float sum = 0.0f;
      const float *current_matrix = matrix + limit; // center of the matrix line
      for (int yy = 0; yy < matrix_size; yy++) { // 0..limit * 2 + 1
        const float *current_line = src_current_lineptrs[yy];
        for (int xx = -limit; xx <= limit; xx++) {
          int current_x = x + xx;
          if (current_x < 0) current_x = 0;
          else if (current_x >= width) current_x = width - 1;
          const float current_pixel = current_line[current_x];
          const float current_weight = current_matrix[xx];
          sum += current_pixel * current_weight;
        }
        current_matrix += matrix_size; // next matrix line
      }
      float result = sum * fCountDiv + fBias;
      dstp[x] = result; // no clipping for float
    }
    // middle area: no x check: fast!
    for (; x < width - limit; x++) {
      float sum = 0.0f;
      const float *current_matrix = matrix + limit; // center of the matrix line
      for (int yy = 0; yy < matrix_size; yy++) { // 0..limit * 2 + 1
        const float *current_line = src_current_lineptrs[yy];
        // compilers are smart nowadays but let's help them
        if constexpr (matrix_size == 3) {
          sum +=
            current_line[x - 1] * current_matrix[-1] +
            current_line[x + 0] * current_matrix[0] +
            current_line[x + 1] * current_matrix[1];
        }
        else if constexpr (matrix_size == 5) {
          sum +=
            current_line[x - 2] * current_matrix[-2] +
            current_line[x - 1] * current_matrix[-1] +
            current_line[x + 0] * current_matrix[0] +
            current_line[x + 1] * current_matrix[1] +
            current_line[x + 2] * current_matrix[2];
        }
        else {
          for (int xx = -limit; xx <= limit; xx++) {
            int current_x = x + xx;
            // no checking!
            /*if (current_x < 0) current_x = 0;
            else if (current_x >= width) current_x = width - 1;*/
            const float current_pixel = current_line[current_x];
            const float current_weight = current_matrix[xx];
            sum += current_pixel * current_weight;
          }
        }
        current_matrix += matrix_size; // next matrix line
      }
      float result = sum * fCountDiv + fBias;
      dstp[x] = result; // no clipping for float
    }
    // right area: check valid x
    for (; x < width; x++)
    {
      float sum = 0.0f;
      const float *current_matrix = matrix + limit; // center of the matrix line
      for (int yy = 0; yy < matrix_size; yy++) { // 0..limit * 2 + 1
        const float *current_line = src_current_lineptrs[yy];
        for (int xx = -limit; xx <= limit; xx++) {
          int current_x = x + xx;
          if (current_x < 0) current_x = 0;
          else if (current_x >= width) current_x = width - 1;
          const float current_pixel = current_line[current_x];
          const float current_weight = current_matrix[xx];
          sum += current_pixel * current_weight;
        }
        current_matrix += matrix_size; // next matrix line
      }
      float result = sum * fCountDiv + fBias;
      dstp[x] = result; // no clipping for float
    }
    dstp += dst_pitch;
    srcp += src_pitch;
  }
}

template void do_conv_float<3>(BYTE* dstp8, int dst_pitch, const BYTE *srcp8, int src_pitch, int width, int height, const float *matrix, float fCountDiv, float fBias);
template void do_conv_float<5>(BYTE* dstp8, int dst_pitch, const BYTE *srcp8, int src_pitch, int width, int height, const float *matrix, float fCountDiv, float fBias);
template void do_conv_float<7>(BYTE* dstp8, int dst_pitch, const BYTE *srcp8, int src_pitch, int width, int height, const float *matrix, float fCountDiv, float fBias);
template void do_conv_float<9>(BYTE* dstp8, int dst_pitch, const BYTE *srcp8, int src_pitch, int width, int height, const float *matrix, float fCountDiv, float fBias);

/***** Setup stuff ****/

GeneralConvolution::GeneralConvolution(PClip _child, double _divisor, float _nBias, const char * _matrix,
  bool _autoscale, bool _luma, bool _chroma, bool _alpha, IScriptEnvironment* _env)
  : GenericVideoFilter(_child), divisor(_divisor), nBias((int)_nBias), fBias(_nBias), autoscale(_autoscale), luma(_luma), chroma(_chroma), alpha(_alpha)
{
  if (vi.Is420() || vi.Is422() || vi.IsYV411()) {
    if (luma && chroma)
      _env->ThrowError("GeneralConvolution: both luma and chroma cannot be set for subsampled video formats");
  }
  if (!vi.IsRGB() && !vi.IsYUV() && !vi.IsYUVA())
    _env->ThrowError("GeneralConvolution requires RGB (planar or packed), greyscale or YUV(A) input");
  if (divisor == 0.0)
    _env->ThrowError("GeneralConvolution: divisor cannot be zero");
  setMatrix(_matrix, vi.BitsPerComponent() < 32, _env); // float: todo

  if (vi.BitsPerComponent() <= 16) {
    // precompute divisor
    int iCountT;
    if (autoscale) {
      iCountT = iNormalizeSum;
    }
    else {
      iCountT = 0;
    }

    // Truncate instead of round - keep in the spirit of the original code
    // 3.6.3: we do introduce rounding before scaling back from +20 bit range
    // 0x100000: 20 bit precision integer arithmetic
    // todo: check overflow for matrices larger than 5x5
    iCountDiv = (int)(0x100000 / (iCountT == 0 ? divisor : iCountT * divisor));
    if (iCountDiv == 0)
      _env->ThrowError("GeneralConvolution: normalizing factor is zero, check for too large elements or divisor value");

    // safety check
    int64needed = false;

    // check possible overflow:
    // Worst case: maximum pixel values, matrix multiplications will overflow either on positive or negative direction
    // When max_pixel_value * sum(max(weight_pos, -weight_neg)) * iCountDiv > 1 << 31 = > int64_t needed(safe_int_t is int64_t)
    const int max_pixel_value = (1 << vi.BitsPerComponent()) - 1;
    const int maxWeightSum = max(iWeightSumPositives, -iWeightSumNegatives);
    if ((int64_t)max_pixel_value * maxWeightSum * iCountDiv >= std::numeric_limits<int>::max())
      int64needed = true;

    switch (vi.BitsPerComponent()) {

    case 8:
      if (nSize == 3 * 3) {
        if (int64needed) conversionFnPtr = do_conv_integer<uint8_t, 8, 3, int64_t>;
        else conversionFnPtr = do_conv_integer<uint8_t, 8, 3, int>;
      }
      else if (nSize == 5 * 5) {
        if (int64needed) conversionFnPtr = do_conv_integer<uint8_t, 8, 5, int64_t>;
        else conversionFnPtr = do_conv_integer<uint8_t, 8, 5, int>;
      }
      else if (nSize == 7 * 7) {
        if (int64needed) conversionFnPtr = do_conv_integer<uint8_t, 8, 7, int64_t>;
        else conversionFnPtr = do_conv_integer<uint8_t, 8, 7, int>;
      }
      else if (nSize == 9 * 9) {
        if (int64needed) conversionFnPtr = do_conv_integer<uint8_t, 8, 9, int64_t>;
        else conversionFnPtr = do_conv_integer<uint8_t, 8, 9, int>;
      }
      break;

    case 10:
      if (nSize == 3 * 3) {
        if (int64needed) conversionFnPtr = do_conv_integer<uint16_t, 10, 3, int64_t>;
        else conversionFnPtr = do_conv_integer<uint16_t, 10, 3, int>;
      }
      else if (nSize == 5 * 5) {
        if (int64needed) conversionFnPtr = do_conv_integer<uint16_t, 10, 5, int64_t>;
        else conversionFnPtr = do_conv_integer<uint16_t, 10, 5, int>;
      }
      else if (nSize == 7 * 7) {
        if (int64needed) conversionFnPtr = do_conv_integer<uint16_t, 10, 7, int64_t>;
        else conversionFnPtr = do_conv_integer<uint16_t, 10, 7, int>;
      }
      else if (nSize == 9 * 9) {
        if (int64needed) conversionFnPtr = do_conv_integer<uint16_t, 10, 9, int64_t>;
        else conversionFnPtr = do_conv_integer<uint16_t, 10, 9, int>;
      }
      break;

    case 12:
      if (nSize == 3 * 3) {
        if (int64needed) conversionFnPtr = do_conv_integer<uint16_t, 12, 3, int64_t>;
        else conversionFnPtr = do_conv_integer<uint16_t, 12, 3, int>;
      }
      else if (nSize == 5 * 5) {
        if (int64needed) conversionFnPtr = do_conv_integer<uint16_t, 12, 5, int64_t>;
        else conversionFnPtr = do_conv_integer<uint16_t, 12, 5, int>;
      }
      else if (nSize == 7 * 7) {
        if (int64needed) conversionFnPtr = do_conv_integer<uint16_t, 12, 7, int64_t>;
        else conversionFnPtr = do_conv_integer<uint16_t, 12, 7, int>;
      }
      else if (nSize == 9 * 9) {
        if (int64needed) conversionFnPtr = do_conv_integer<uint16_t, 12, 9, int64_t>;
        else conversionFnPtr = do_conv_integer<uint16_t, 12, 9, int>;
      }
      break;

    case 14:
      if (nSize == 3 * 3) {
        if (int64needed) conversionFnPtr = do_conv_integer<uint16_t, 14, 3, int64_t>;
        else conversionFnPtr = do_conv_integer<uint16_t, 14, 3, int>;
      }
      else if (nSize == 5 * 5) {
        if (int64needed) conversionFnPtr = do_conv_integer<uint16_t, 14, 5, int64_t>;
        else conversionFnPtr = do_conv_integer<uint16_t, 14, 5, int>;
      }
      else if (nSize == 7 * 7) {
        if (int64needed) conversionFnPtr = do_conv_integer<uint16_t, 14, 7, int64_t>;
        else conversionFnPtr = do_conv_integer<uint16_t, 14, 7, int>;
      }
      else if (nSize == 9 * 9) {
        if (int64needed) conversionFnPtr = do_conv_integer<uint16_t, 14, 9, int64_t>;
        else conversionFnPtr = do_conv_integer<uint16_t, 14, 9, int>;
      }
      break;

    case 16:
      if (nSize == 3 * 3) {
        if (int64needed) conversionFnPtr = do_conv_integer<uint16_t, 16, 3, int64_t>;
        else conversionFnPtr = do_conv_integer<uint16_t, 16, 3, int>;
      }
      else if (nSize == 5 * 5) {
        if (int64needed) conversionFnPtr = do_conv_integer<uint16_t, 16, 5, int64_t>;
        else conversionFnPtr = do_conv_integer<uint16_t, 16, 5, int>;
      }
      else if (nSize == 7 * 7) {
        if (int64needed) conversionFnPtr = do_conv_integer<uint16_t, 16, 7, int64_t>;
        else conversionFnPtr = do_conv_integer<uint16_t, 16, 7, int>;
      }
      else if (nSize == 9 * 9) {
        if (int64needed) conversionFnPtr = do_conv_integer<uint16_t, 16, 9, int64_t>;
        else conversionFnPtr = do_conv_integer<uint16_t, 16, 9, int>;
      }
      break;
    }
  }
  else {
    // 32 bit float clip
    // precompute divisor
    float fCountT;
    if (autoscale) {
      fCountT = fNormalizeSum;
    }
    else {
      fCountT = 0.0f;
    }

    fCountDiv = (float)(1.0f / (fCountT == 0 ? divisor : fCountT * divisor));

    if (nSize == 3 * 3)
      FconversionFnPtr = do_conv_float<3>;
    else if (nSize == 5 * 5)
      FconversionFnPtr = do_conv_float<5>;
    else if (nSize == 7 * 7)
      FconversionFnPtr = do_conv_float<7>;
    else if (nSize == 9 * 9)
      FconversionFnPtr = do_conv_float<9>;
  }
}

AVSValue __cdecl GeneralConvolution::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  const VideoInfo& vi_orig = args[0].AsClip()->GetVideoInfo();

  // convert old RGB format to planar RGB
  AVSValue new_args[1] = { args[0].AsClip() };
  PClip clip;
  if (vi_orig.IsRGB24() || vi_orig.IsRGB48()) {
    clip = env->Invoke("ConvertToPlanarRGB", AVSValue(new_args, 1)).AsClip();
  }
  else if (vi_orig.IsRGB32() || vi_orig.IsRGB64()) {
    clip = env->Invoke("ConvertToPlanarRGBA", AVSValue(new_args, 1)).AsClip();
  }
  else if (vi_orig.IsYUY2()) {
    clip = env->Invoke("ConvertToYV16", AVSValue(new_args, 1)).AsClip();
  }
  else {
    clip = args[0].AsClip();
  }

  GeneralConvolution* Result = new GeneralConvolution(clip, args[3].AsFloat(1.0f), args[1].AsFloatf(0.0f),
    args[2].AsString("0 0 0 0 1 0 0 0 0"), args[4].AsBool(true),
    args[5].AsBool(true), args[6].AsBool(true), args[7].AsBool(true), // luma, chroma, alpha, when n/a then ignored
    env);

  AVSValue new_args2[1] = { Result };
  if (vi_orig.IsRGB24()) {
    return env->Invoke("ConvertToRGB24", AVSValue(new_args2, 1)).AsClip();
  }
  else if (vi_orig.IsRGB48()) {
    return env->Invoke("ConvertToRGB48", AVSValue(new_args2, 1)).AsClip();
  }
  else if (vi_orig.IsRGB32()) {
    return env->Invoke("ConvertToRGB32", AVSValue(new_args2, 1)).AsClip();
  }
  else if (vi_orig.IsRGB64()) {
    return env->Invoke("ConvertToRGB64", AVSValue(new_args2, 1)).AsClip();
  }
  else if (vi_orig.IsYUY2()) {
    return env->Invoke("ConvertToYUY2", AVSValue(new_args2, 1)).AsClip();
  }

  return Result;
}


void GeneralConvolution::setMatrix(const char * _matrix, bool _isInteger, IScriptEnvironment* env)
{
  char delimiter[] = "([ \t\n\r]+)";
  std::regex regex(delimiter);
  std::string str(_matrix);
  std::vector<std::string> out(
    std::sregex_token_iterator(str.begin(), str.end(), regex, -1),
    std::sregex_token_iterator()
  );

  fNormalizeSum = 0.0f;
  iNormalizeSum = 0;
  iWeightSumPositives = 0; // for int32 overflow decision
  iWeightSumNegatives = 0;
  const int MAX_DIMENSION = 9; // 9 is the max matrix size which is templated

  nSize = 0;
  int dim = 3;
  int maxsize = dim * dim;
  if (_isInteger)
    iMatrix.resize(maxsize);
  else
    fMatrix.resize(maxsize);
  for (auto &s : out) {
    if (s.length() == 0) continue; // first string can be empty is matrix string is starting with separators

    if (nSize == (size_t)maxsize) {
      if (dim == MAX_DIMENSION) {
        env->ThrowError("GeneralConvolution: matrix too big, maximum %dx%d elements allowed", MAX_DIMENSION, MAX_DIMENSION);
      }
      dim += 2;
      maxsize = dim * dim;
      if (_isInteger)
        iMatrix.resize(maxsize);
      else
        fMatrix.resize(maxsize);
    }
    if (_isInteger) {
      const double val = atof(s.c_str());
      const int ival = (int)std::lround(val);
      iNormalizeSum += ival;
      iMatrix[nSize++] = ival;
      if (ival >= 0)
        iWeightSumPositives += ival;
      else
        iWeightSumNegatives += ival;
    }
    else {
      const float val = (float)atof(s.c_str());
      fNormalizeSum += val;
      fMatrix[nSize++] = val;
    }

  }

  if (nSize < 9)
    env->ThrowError("GeneralConvolution: matrix too small, need at least 3x3 elements");
  else if (nSize != (size_t)maxsize)
    env->ThrowError("GeneralConvolution: matrix incomplete, possible size %dx%d but element count %d", dim, dim, nSize);
}

PVideoFrame __stdcall GeneralConvolution::GetFrame(int n, IScriptEnvironment* env)
{
  int h = vi.height;
  int w = vi.width;

  PVideoFrame src = child->GetFrame(n, env);
  PVideoFrame dst = env->NewVideoFrameP(vi, &src);

  const int *matrix = iMatrix.data();
  const float *matrixf = fMatrix.data();

  int planes_y[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
  int planes_r[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
  int *planes = (vi.IsYUV() || vi.IsYUVA()) ? planes_y : planes_r;
  for (int p = 0; p < vi.NumComponents(); ++p) {
    const int plane = planes[p];
    if ((plane == PLANAR_Y && !luma) ||
        ((plane == PLANAR_U || plane == PLANAR_V) && !chroma) ||
        (plane == PLANAR_A && !alpha))
    {
      env->BitBlt(dst->GetWritePtr(plane), dst->GetPitch(plane), src->GetReadPtr(plane), src->GetPitch(plane), src->GetRowSize(plane), src->GetHeight(plane));
      continue;
    }

    int width = w;
    int height = h;
    if (plane == PLANAR_U || plane == PLANAR_V) {
      width >>= vi.GetPlaneWidthSubsampling(plane);
      height >>= vi.GetPlaneHeightSubsampling(plane);
    }

    if(vi.BitsPerComponent() <= 16)
      conversionFnPtr(dst->GetWritePtr(plane), dst->GetPitch(plane), src->GetReadPtr(plane), src->GetPitch(plane), width, height, matrix, iCountDiv, nBias);
    else
      FconversionFnPtr(dst->GetWritePtr(plane), dst->GetPitch(plane), src->GetReadPtr(plane), src->GetPitch(plane), width, height, matrixf, fCountDiv, fBias);
  }
  return dst;
  // really, not other case left... packed RGB was converted to planar RGB, YUY2 to YV16
}
