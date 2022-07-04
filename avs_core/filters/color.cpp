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

#include "color.h"

#include <math.h>
#include <float.h>
#if defined(AVS_BSD) || defined(AVS_MACOS)
    #include <stdlib.h>
#else
    #include <malloc.h>
#endif
#include <stdio.h>

#ifdef AVS_WINDOWS
    #include <avs/win.h>
#else
    #include <avs/posix.h>
#endif

#include <avs/minmax.h>
#include "../core/internal.h"
#include <algorithm>
#include <sstream> // stringstream
#include <iomanip> // setprecision
#include <string>
#include "../convert/convert_helper.h"

static void coloryuv_showyuv(BYTE* pY, BYTE* pU, BYTE* pV, int y_pitch, int u_pitch, int v_pitch, int framenumber, bool full_range, int bits_per_pixel)
{
  int internal_bitdepth = bits_per_pixel == 8 ? 8 : 10;

  const int luma_min = full_range ? 0 : (16 << (internal_bitdepth - 8));
  const int luma_max = full_range ? (1 << internal_bitdepth) - 1 : (235 << (internal_bitdepth - 8));

  const int chroma_center = 128 << (internal_bitdepth - 8);
  const int chroma_span = full_range ? (1 << (internal_bitdepth - 1)) - 1 : (112 << (internal_bitdepth - 8)); // +-127/+-112
  const int chroma_min = chroma_center - chroma_span; // +-112 (16-240) -> +-127 (1-255)
  const int chroma_max = chroma_center + chroma_span;

  const int luma_range = luma_max - luma_min + 1; // 256/220 ,1024/880
  const int chroma_range = chroma_max - chroma_min + 1;

  const int luma_size = chroma_range * 2; // YUV output is always 4:2:0. Horizontal subsampling 2.

  int luma;
  // Calculate luma cycle
  // 0,1..255,254,..1 = 2x256-2
  // 0,1..1023,1022,..1 = 2*1024-2
  luma = framenumber % (luma_range * 2 - 2);
  if (luma > luma_range - 1)
    luma = (luma_range * 2 - 2) - luma;
  luma += luma_min;

  // Set luma value
  if (bits_per_pixel == 8) {
    for (int y = 0; y < luma_size; y++) {
      memset(pY, luma, luma_size);
      pY += y_pitch;
    }
  }
  else if (bits_per_pixel <= 16) {
    // display size (thus luma range) covers the same as at 10 bits.
    if (full_range) {
      // stretch
      const float factor = (float)((1 << bits_per_pixel) - 1) / ((1 << internal_bitdepth) - 1);
      const int luma_target = (int)(luma * factor + 0.5f);
      for (int y = 0; y < luma_size; y++) {
        std::fill_n((uint16_t*)pY, luma_size, luma_target);
        pY += y_pitch;
      }
    }
    else {
      // shift
      const int luma_target = luma << (bits_per_pixel - internal_bitdepth);
      for (int y = 0; y < luma_size; y++) {
        std::fill_n((uint16_t*)pY, luma_size, luma_target);
        pY += y_pitch;
      }
    }
  }
  else {
    // 32 bit float
    const float factor = 1.0f / ((1 << internal_bitdepth) - 1);
    const float luma_target = luma * factor;
    for (int y = 0; y < luma_size; y++) {
      std::fill_n((float*)pY, luma_size, luma_target);
      pY += y_pitch;
    }
  }

  // Set chroma
  if (full_range) {
    if (bits_per_pixel != 32) {
      // 8-16 bit full
      const int chroma_center_target = 128 << (bits_per_pixel - 8);
      const int chroma_span_target = full_range ? (1 << (bits_per_pixel - 1)) - 1 : (112 << (bits_per_pixel - 8)); // +-127/+-112
      const float factor = (float)chroma_span_target / chroma_span;
      const float chroma_center_target_plus_round = chroma_center_target + 0.5f;

      if (bits_per_pixel == 8) {
        for (int y = 0; y < chroma_range; y++)
        {
          for (int x = 0; x < chroma_range; x++) {
            int pixel_u = chroma_min + x;
            pixel_u = (int)((pixel_u - chroma_center) * factor + chroma_center_target_plus_round);
            pU[x] = pixel_u;
          }
          int pixel_v = chroma_min + y;
          pixel_v = (int)((pixel_v - chroma_center) * factor + chroma_center_target_plus_round);
          std::fill_n((uint8_t*)pV, chroma_range, pixel_v);

          pU += u_pitch;
          pV += v_pitch;
        }
      }
      else if (bits_per_pixel <= 16) {
        for (int y = 0; y < chroma_range; y++)
        {
          for (int x = 0; x < chroma_range; x++) {
            int pixel_u = chroma_min + x;
            pixel_u = (int)((pixel_u - chroma_center) * factor + chroma_center_target_plus_round);
            reinterpret_cast<uint16_t*>(pU)[x] = pixel_u;
          }
          int pixel_v = chroma_min + y;
          pixel_v = (int)((pixel_v - chroma_center) * factor + chroma_center_target_plus_round);
          std::fill_n((uint16_t*)pV, chroma_range, pixel_v);

          pU += u_pitch;
          pV += v_pitch;
        }
      }
    }
    else {
      // 32 bit float, full
      const float chroma_span_target = 0.5f; // +-0.5/+-112
      const float factor = chroma_span_target / chroma_span;

      for (int y = 0; y < chroma_range; y++)
      {
        for (int x = 0; x < chroma_range; x++) {
          const int pixel_u = chroma_min + x;
          const float pixel_u_f = (pixel_u - chroma_center) * factor;
          reinterpret_cast<float*>(pU)[x] = pixel_u_f;
        }

        const int pixel_v = chroma_min + y;
        const float pixel_v_f = (pixel_v - chroma_center) * factor;
        std::fill_n((float*)pV, chroma_range, pixel_v_f);

        pU += u_pitch;
        pV += v_pitch;
      }
    }
    // full end
  }
  else {
    if (bits_per_pixel == 8) {
      // 8 bit limited range
      for (int y = 0; y < chroma_range; y++)
      {
        for (int x = 0; x < chroma_range; x++)
          pU[x] = chroma_min + x;
        const int pixel_v = chroma_min + y;
        std::fill_n((uint8_t*)pV, chroma_range, pixel_v);
        pU += u_pitch;
        pV += v_pitch;
      }
    }
    else if (bits_per_pixel <= 16) {
      // 16 bit limited range
      const int bitdiff = (bits_per_pixel - internal_bitdepth);
      for (int y = 0; y < chroma_range; y++)
      {
        for (int x = 0; x < chroma_range; x++) {
          reinterpret_cast<uint16_t*>(pU)[x] = (chroma_min + x) << bitdiff;
        }
        const int pixel_v = (chroma_min + y) << bitdiff;
        std::fill_n((uint16_t*)pV, chroma_range, pixel_v);
        pU += u_pitch;
        pV += v_pitch;
      }
    }
    else {
      // 32 bit float limited
      const float factor = 1.0f / ((1 << internal_bitdepth) - 1);
      for (int y = 0; y < chroma_range; y++)
      {
        for (int x = 0; x < chroma_range; x++) {
          reinterpret_cast<float*>(pU)[x] = (float)(chroma_min + x - chroma_center) * factor;
        }
        const float pixel_v = (float)(chroma_min + y - chroma_center) * factor;
        std::fill_n((float*)pV, chroma_range, pixel_v);
        pU += u_pitch;
        pV += v_pitch;
      }
    }
  }
}

// luts are only for integer bits 8/10/12/14/16. float will be realtime
template<typename pixel_t>
static void coloryuv_create_lut(BYTE* lut8, const ColorYUVPlaneConfig* config, int bits_per_pixel, bool tweaklike_params)
{
  pixel_t* lut = reinterpret_cast<pixel_t*>(lut8);

  // scale is 256/1024/4096/16384/65536
  // normalize to [0..1) working range
  const double value_normalization_scale = (double)((int64_t)1 << bits_per_pixel);
  // For gamma pre-post correction. 
  const double tv_range_lo_normalized = 16.0 / 256.0;

  const int lookup_size = 1 << bits_per_pixel; // 256, 1024, 4096, 16384, 65536
  const int source_max = (1 << bits_per_pixel) - 1;
  const bool chroma = config->plane == PLANAR_U || config->plane == PLANAR_V;
  const bool fulls = config->range == COLORYUV_RANGE_PC_TV;
  const bool fulld = config->range == COLORYUV_RANGE_TV_PC;
  // when COLORYUV_RANGE_NONE both are the same false

  //-----------------------
  bits_conv_constants d;
  // When calculating src_pixel, src and dst are of the same bit depth
  get_bits_conv_constants(d, chroma, fulls, fulld, bits_per_pixel, bits_per_pixel);

  auto dst_offset_plus_round = d.dst_offset + 0.5;
  const int src_pixel_min = 0;
  const int src_pixel_max = source_max;

  // parameters are not scaled by bitdepth (legacy 8 bit behaviour)
  double gain = tweaklike_params ? config->gain : (config->gain / 256 + 1.0);
  double contrast = tweaklike_params ? config->contrast : (config->contrast / 256 + 1.0);
  double gamma = tweaklike_params ? config->gamma : (config->gamma / 256 + 1.0);
  double offset = config->offset / 256;

  // for correct Y gamma
  double range_factor_tv_to_pc = 255.0 / 219.0; // 16-235 ==> 0..255
  double range_factor_pc_to_tv = 219.0 / 255.0; // 0..255 ==> 16-235 (219)

  // for coring
  const int tv_range_lo_luma_chroma = (16 << (bits_per_pixel - 8));
  const int tv_range_hi_luma = (235 << (bits_per_pixel - 8));
  const int tv_range_hi_chroma = (240 << (bits_per_pixel - 8));

  // We know that the input is TV range for sure: (coring=true and !PC->TV), levels="TV->PC" or (new!) levels="TV"
  const bool source_is_limited = (config->clip_tv && config->range != COLORYUV_RANGE_PC_TV) || config->range == COLORYUV_RANGE_TV_PC || config->force_tv_range;

  for (int i = 0; i < lookup_size; i++) {
    double value = double(i);
    value /= value_normalization_scale; // normalize to [0..1). For chroma this makes the center to 0.5.
    value *= gain; // Applying gain
    // Applying contrast. For chroma, this sets saturation
    value = (value - 0.5) * contrast + 0.5; // integer chroma center is transformed to 0, apply contrast
    // in Classic AVS: constract is applied on the original value and not on the already gained value
    // value = (value * gain) + ((value - 0.5) * contrast + 0.5) - value + (bright - 1);
    value += offset; // Applying offset

    // Applying gamma. Only on Y
    if (gamma != 0) {
      if (source_is_limited) {
        // avs+ 180301- use gamma on the proper 0.0 based value
        if (value > tv_range_lo_normalized)
        {
          // tv->pc
          value = (value - tv_range_lo_normalized) * range_factor_tv_to_pc; // (v-16)*range
          value = pow(value, 1.0 / gamma);
          // pc->tv
          value = value * range_factor_pc_to_tv + tv_range_lo_normalized; // v*range - 16
        }
      }
      else { // full (PC) range
        if (value > 0)
          value = pow(value, 1.0 / gamma);
      }
    }

    value *= value_normalization_scale; // back from [0..1) range

    if (fulls != fulld)
      // Range conversion
      value = (value - d.src_offset) * d.mul_factor + dst_offset_plus_round;
    else
      value = value + 0.5; // rounder

    // back to the integer world
    int iValue = clamp((int)value, src_pixel_min, src_pixel_max);

    if (config->clip_tv) // set when coring
    {
      iValue = clamp(iValue, tv_range_lo_luma_chroma, config->plane == PLANAR_Y ? tv_range_hi_luma : tv_range_hi_chroma);
    }

    lut[i] = iValue;
  }
}

// works with <= 16 bits, but used only for float
static std::string coloryuv_create_lut_expr(const ColorYUVPlaneConfig* config, int bits_per_pixel, bool tweaklike_params)
{
  const bool f32 = bits_per_pixel == 32;
  const bool chroma = config->plane == PLANAR_U || config->plane == PLANAR_V;
  // 32 bit float is already in [0..1] range but we have to make it similar to integer [0..1): needs /256 and not /255.
  // Reason: match to the integer behaviour
  // integer: normalize to [0..1) working range
  const double value_normalization_scale = f32 ? (256.0 / 255.0) : (double)((int64_t)1 << bits_per_pixel);
  // For gamma pre-post correction. we are in [0..1) working range
  const double tv_range_lo_normalized = 16.0 / 256.0;

  const double source_max = f32 ? 1.0 : (double)((1 << bits_per_pixel) - 1);
  const bool fulls = config->range == COLORYUV_RANGE_PC_TV;
  const bool fulld = config->range == COLORYUV_RANGE_TV_PC;
  // when COLORYUV_RANGE_NONE both are the same false

  bits_conv_constants d;
  // When calculating src_pixel, src and dst are of the same bit depth
  get_bits_conv_constants(d, chroma, fulls, fulld, bits_per_pixel, bits_per_pixel);

  auto dst_offset_no_round = d.dst_offset;
  //const auto src_pixel_max = source_max;

  // parameters are not scaled by bitdepth (legacy 8 bit behaviour)
  double gain = tweaklike_params ? config->gain : (config->gain / 256 + 1.0);
  double contrast = tweaklike_params ? config->contrast : (config->contrast / 256 + 1.0);
  double gamma = tweaklike_params ? config->gamma : (config->gamma / 256 + 1.0);
  double offset = config->offset / 256; // always in the 256 range

  // for correct Y gamma
  double range_factor_tv_to_pc = 255.0 / 219.0; // 16-235 ==> 0..255
  double range_factor_pc_to_tv = 219.0 / 255.0; // 0..255 ==> 16-235 (219)

  // We know that the input is TV range for sure: (coring=true and !PC->TV), levels="TV->PC" or (new!) levels="TV"
  const bool source_is_limited = (config->clip_tv && config->range != COLORYUV_RANGE_PC_TV) || config->range == COLORYUV_RANGE_TV_PC || config->force_tv_range;

  std::stringstream ss;

  // value = double(i), we are using floats here
  ss << "x";

  // value = value / value_normalization_scale;
  if (chroma && f32)
    ss << " 255 * 256 / 0.5 + "; // from float chroma +/- to match with legacy integer behaviour
  else
    ss << " " << value_normalization_scale << " /";

  if (gain != 1.0)
    ss << " " << gain << " *"; // Applying gain // value *= gain;

  // Applying contrast. For chroma, this sets saturation
  //  value = (value - 0.5) * contrast + 0.5;
  // earlier we made measures that float chroma center is kept at 0.5
  ss << " 0.5 - " << contrast << " * 0.5 +";

  // value += offset; // Applying offset
  ss << " " << offset << " +";

  // Applying gamma. Only on Y
  if (gamma != 0) {
    if (source_is_limited) {
      // avs+ 180301- use gamma on the proper 0.0 based value
      // value = value > 16scaled ? (pow((value - 16scl)*range_tv_pc,(1/gamma))*range_pc_tv+16d : value
      ss << " A@ " << tv_range_lo_normalized << " > " // condition: value > tv_range_lo_normalized
        // case: true. tv->pc, power, pc->tv
        << "A " << tv_range_lo_normalized << " - " << range_factor_tv_to_pc << " * " << (1.0 / gamma) << " pow " << range_factor_pc_to_tv << " * " << tv_range_lo_normalized << " + " 
        // case: false. Original value
        << " A ? ";
    }
    else { // full (PC) range
      //if (value > 0)
      //  value = pow(value, 1.0 / gamma);
      // value = value > 0 ? pow(value,(1/gamma)) : value
      ss << " A@ 0 > A " << (1.0 / gamma) << " pow A ? ";
    }
  }

  if (chroma && f32)
    ss << " 0.5 - 256 * 255 / ";
  else
    ss << " " << value_normalization_scale << " * "; // value *= value_normalization_scale; // back from [0..1) range

  if (fulls != fulld) {
    ss << d.src_offset << " - " << d.mul_factor << " * " << dst_offset_no_round << " + ";
    // value = (value - d.src_offset) * d.mul_factor + dst_offset_no_round; // no rounder, Expr will round automatically before return
  }
  else {
    // value = value + 0.5; // no rounder, Expr rounds
  }

  // clamp to the original valid bitdepth is done by Expr
  // int iValue = clamp((int)value, src_pixel_min, src_pixel_max);
  // ss << " " << src_pixel_min << " max " << src_pixel_max << " min ";

  if (config->clip_tv) // set when coring
  {
    double tv_range_lo_luma = f32 ? (16.0 / 255) : ((int64_t)16 << (bits_per_pixel - 8));
    double tv_range_hi_luma = f32 ? (235.0 / 255) : ((int64_t)235 << (bits_per_pixel - 8));
    double tv_range_lo_chroma = f32 ? (-112.0 / 255.0) : ((int64_t)16 << (bits_per_pixel - 8)); // 112/255.0 consistent with get_bits_conv_constants()
    double tv_range_hi_chroma = f32 ? (+112.0 / 255.0) : ((int64_t)240 << (bits_per_pixel - 8));
    ss << (config->plane == PLANAR_Y ? tv_range_lo_luma : tv_range_lo_chroma) << " max ";
    ss << (config->plane == PLANAR_Y ? tv_range_hi_luma : tv_range_hi_chroma) << " min ";
    //iValue = clamp(iValue, tv_range_lo_luma, config->plane == PLANAR_Y ? tv_range_hi_luma : tv_range_hi_chroma);
  }

  std::string exprString = ss.str();
  return exprString;

}

// for float, only loose_min and loose_max is counted
template<bool forFloat>
static void coloryuv_analyse_core(const int* freq, const int pixel_num, ColorYUVPlaneData* data, int bits_per_pixel_for_stat)
{
  // 32 bit float reached here as split into ranges like a 16bit clip
  int pixel_value_count = 1 << bits_per_pixel_for_stat; // size of freq table

  const int pixel_256th = pixel_num / 256; // For loose max/min yes, still 1/256!

  double avg = 0.0;
  int real_min, real_max;

  if (!forFloat) {
    real_min = -1;
    real_max = -1;
  }
  data->loose_max = -1;
  data->loose_min = -1;

  int px_min_c = 0, px_max_c = 0;

  for (int i = 0; i < pixel_value_count; i++)
  {
    if (!forFloat) {
      avg += freq[i] * double(i);

      if (freq[i] > 0 && real_min == -1)
      {
        real_min = i;
      }
    }

    if (data->loose_min == -1)
    {
      px_min_c += freq[i];

      if (px_min_c > pixel_256th)
      {
        data->loose_min = i;
      }
    }

    if (!forFloat) {
      if (freq[pixel_value_count - 1 - i] > 0 && real_max == -1)
      {
        real_max = pixel_value_count - 1 - i;
      }
    }

    if (data->loose_max == -1)
    {
      px_max_c += freq[pixel_value_count - 1 - i];

      if (px_max_c > pixel_256th)
      {
        data->loose_max = pixel_value_count - 1 - i;
      }
    }
  }

  if (!forFloat) {
    avg /= pixel_num;
    data->average = avg;
    data->real_min = (float)real_min;
    data->real_max = (float)real_max;
  }
}


static void coloryuv_analyse_planar(const BYTE* pSrc, int src_pitch, int width, int height, ColorYUVPlaneData* data, int bits_per_pixel, bool chroma)
{
    // We can gather statistics from float, but for population count we have to
    // split the range. We decide to split it into 2^16 ranges.
    int bits_per_pixel_for_freq = bits_per_pixel <= 16 ? bits_per_pixel : 16;
    int statistics_size = 1 << bits_per_pixel_for_freq; // float: 65536
    int *freq = new int[statistics_size];
    std::fill_n(freq, statistics_size, 0);

    double sum = 0.0; // for float
    float real_min;
    float real_max;

    if(bits_per_pixel==8) {
      for (int y = 0; y < height; y++)
      {
          for (int x = 0; x < width; x++)
          {
              freq[pSrc[x]]++;
          }

          pSrc += src_pitch;
      }
    }
    else if (bits_per_pixel >= 10 && bits_per_pixel <= 14) {
      uint16_t mask = statistics_size - 1; // e.g. 0x3FF for 10 bit
      for (int y = 0; y < height; y++)
      {
        for (int x = 0; x < width; x++)
        {
          freq[clamp(reinterpret_cast<const uint16_t *>(pSrc)[x],(uint16_t)0,mask)]++;
        }

        pSrc += src_pitch;
      }
    }
    else if (bits_per_pixel == 16) {
      // no clamp, faster
      for (int y = 0; y < height; y++)
      {
        for (int x = 0; x < width; x++)
        {
          freq[reinterpret_cast<const uint16_t *>(pSrc)[x]]++;
        }

        pSrc += src_pitch;
      }
    } else if(bits_per_pixel==32) {
      // 32 bits: we populate pixels only for loose_min and loose_max
      // real_min and real_max, average (sum) is computed differently
      real_min = reinterpret_cast<const float *>(pSrc)[0];
      real_max = real_min;
      if (chroma) {
        const float shift = 32768.0f;
        for (int y = 0; y < height; y++)
        {
          for (int x = 0; x < width; x++)
          {
            // -0.5..0.5 (0..1.0 when FLOAT_CHROMA_IS_HALF_CENTERED) to 0..65535
            // see also: ConditionalFunctions MinMax
            const float pixel = reinterpret_cast<const float *>(pSrc)[x];
            freq[clamp((int)(65535.0f*pixel + shift + 0.5f), 0, 65535)]++;
            // todo: SSE2
            real_min = min(real_min, pixel);
            real_max = max(real_max, pixel);
            sum += pixel;
          }
          pSrc += src_pitch;
        }
      }
      else {
        for (int y = 0; y < height; y++)
        {
          for (int x = 0; x < width; x++)
          {
            // 0..1 -> 0..65535
            const float pixel = reinterpret_cast<const float *>(pSrc)[x];
            freq[clamp((int)(65535.0f*pixel + 0.5f), 0, 65535)]++;
            // todo: SSE2
            real_min = min(real_min, pixel);
            real_max = max(real_max, pixel);
            sum += pixel;
          }

          pSrc += src_pitch;
        }
      }
    }


    if (bits_per_pixel == 32) {
      coloryuv_analyse_core<true>(freq, width*height, data, bits_per_pixel_for_freq);
      data->average = sum / (height * width);
      data->real_max = real_max;
      data->real_min = real_min;
      // loose min and max was shifted by half of 16bit range. We still keep here the range
      if (chroma) {
        data->loose_max = data->loose_max - 32768;
        data->loose_min = data->loose_min - 32768;
      }
      // autogain treats it as a value of 16bit magnitude, show=true as well
      //data->loose_min = data->loose_min / 65535.0f; not now.
      //data->loose_max = data->loose_max / 65535.0f;
    }
    else {
      coloryuv_analyse_core<false>(freq, width*height, data, bits_per_pixel_for_freq);
    }

    delete[] freq;
}

static void coloryuv_analyse_yuy2(const BYTE* pSrc, int src_pitch, int width, int height, ColorYUVPlaneData* dataY, ColorYUVPlaneData* dataU, ColorYUVPlaneData* dataV)
{
    int freqY[256], freqU[256], freqV[256];
    memset(freqY, 0, sizeof(freqY));
    memset(freqU, 0, sizeof(freqU));
    memset(freqV, 0, sizeof(freqV));

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width*2; x+=4)
        {
            freqY[pSrc[x+0]]++;
            freqU[pSrc[x+1]]++;
            freqY[pSrc[x+2]]++;
            freqV[pSrc[x+3]]++;
        }

        pSrc += src_pitch;
    }

    coloryuv_analyse_core<false>(freqY, width*height, dataY, 1);
    coloryuv_analyse_core<false>(freqU, width*height/2, dataU, 1);
    coloryuv_analyse_core<false>(freqV, width*height/2, dataV, 1);
}

static void coloryuv_autogain(const ColorYUVPlaneData* dY, const ColorYUVPlaneData* dU, const ColorYUVPlaneData* dV, ColorYUVPlaneConfig* cY, ColorYUVPlaneConfig* cU, ColorYUVPlaneConfig* cV,
  int bits_per_pixel, bool tweaklike_params)
{
  int bits_per_pixel_for_freq = bits_per_pixel <= 16 ? bits_per_pixel : 16; // for float: "loose" statistics like uint16_t
  // always 16..235
  int loose_max_limit = (235 + 1) << (bits_per_pixel_for_freq - 8);
  int loose_min_limit = 16 << (bits_per_pixel_for_freq - 8);
  int maxY = min(dY->loose_max, loose_max_limit);
  int minY = max(dY->loose_min, loose_min_limit);

  int range = maxY - minY;

  if (range > 0) {
    double scale = double(loose_max_limit - loose_min_limit) / range;
    cY->offset = (loose_min_limit - scale * minY) / (1 << (bits_per_pixel_for_freq - 8)); // good for float also, 0..256 range
    cY->gain = tweaklike_params ? scale : (256 * (scale - 1.0));
    cY->changed = true;
  }
}

static void coloryuv_autowhite(const ColorYUVPlaneData* dY, const ColorYUVPlaneData* dU, const ColorYUVPlaneData* dV, ColorYUVPlaneConfig* cY, ColorYUVPlaneConfig* cU, ColorYUVPlaneConfig* cV,
  int bits_per_pixel)
{
  if (bits_per_pixel == 32) {
#ifdef FLOAT_CHROMA_IS_HALF_CENTERED
    double middle = 0.5;
#else
    double middle = 0.0;
#endif
    cU->offset = (middle - dU->average) * 256; // parameter is in 256 range
    cV->offset = (middle - dV->average) * 256;
  }
  else {
    double middle = (1 << (bits_per_pixel - 1)) - 1; // 128-1, 2048-1 ...
    cU->offset = (middle - dU->average) / (1 << (bits_per_pixel - 8)); // parameter is in 256 range
    cV->offset = (middle - dV->average) / (1 << (bits_per_pixel - 8));
  }
  cU->changed = true;
  cV->changed = true;
}

// only for integer samples
static void coloryuv_apply_lut_planar(BYTE* pDst, const BYTE* pSrc, int dst_pitch, int src_pitch, int width, int height, const BYTE* lut, int bits_per_pixel)
{
    if(bits_per_pixel==8)
    {
      for (int y = 0; y < height; y++)
      {
          for (int x = 0; x < width; x++)
          {
              pDst[x] = lut[pSrc[x]];
          }

          pSrc += src_pitch;
          pDst += dst_pitch;
      }
    }
    else if (bits_per_pixel >= 10 && bits_per_pixel <= 14) {
      uint16_t max_pixel_value = (1 << bits_per_pixel) - 1;
      // protection needed for lut
      for (int y = 0; y < height; y++)
      {
        for (int x = 0; x < width; x++)
        {
          uint16_t pixel = reinterpret_cast<const uint16_t *>(pSrc)[x];
          pixel = pixel <= max_pixel_value ? pixel : max_pixel_value;
          reinterpret_cast<uint16_t *>(pDst)[x] = reinterpret_cast<const uint16_t *>(lut)[pixel];
        }

        pSrc += src_pitch;
        pDst += dst_pitch;
      }
    }
    else if (bits_per_pixel == 16) {
      // no protection, faster
      for (int y = 0; y < height; y++)
      {
        for (int x = 0; x < width; x++)
        {
          reinterpret_cast<uint16_t *>(pDst)[x] = reinterpret_cast<const uint16_t *>(lut)[reinterpret_cast<const uint16_t *>(pSrc)[x]];
        }

        pSrc += src_pitch;
        pDst += dst_pitch;
      }
    }
}

static void coloryuv_apply_lut_yuy2(BYTE* pDst, const BYTE* pSrc, int dst_pitch, int src_pitch, int width, int height, const BYTE* lutY, const BYTE* lutU, const BYTE* lutV)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width * 2; x += 4)
        {
            pDst[x+0] = lutY[pSrc[x + 0]];
            pDst[x+1] = lutU[pSrc[x + 1]];
            pDst[x+2] = lutY[pSrc[x + 2]];
            pDst[x+3] = lutV[pSrc[x + 3]];
        }

        pSrc += src_pitch;
        pDst += dst_pitch;
    }
}

#define READ_CONDITIONAL(plane, var_name, internal_name, condVarSuffix)  \
    {            \
        std::string s = "coloryuv_" #var_name "_" #plane;\
        s = s + condVarSuffix; \
        const double t = env->GetVarDouble(s.c_str(), DBL_MIN); \
        if (t != DBL_MIN) {                               \
            c_##plane->internal_name = t;               \
            c_##plane->changed = true;                  \
        }                                                 \
    }

// extra: add extra at the end of variable names: different variables for multiple instances of coloryuv
static void coloryuv_read_conditional(IScriptEnvironment* env, ColorYUVPlaneConfig* c_y, ColorYUVPlaneConfig* c_u, ColorYUVPlaneConfig* c_v, const char *condVarSuffix)
{
    READ_CONDITIONAL(y, gain, gain, condVarSuffix);
    READ_CONDITIONAL(y, off, offset, condVarSuffix);
    READ_CONDITIONAL(y, gamma, gamma, condVarSuffix);
    READ_CONDITIONAL(y, cont, contrast, condVarSuffix);

    READ_CONDITIONAL(u, gain, gain, condVarSuffix);
    READ_CONDITIONAL(u, off, offset, condVarSuffix);
    READ_CONDITIONAL(u, gamma, gamma, condVarSuffix);
    READ_CONDITIONAL(u, cont, contrast, condVarSuffix);

    READ_CONDITIONAL(v, gain, gain, condVarSuffix);
    READ_CONDITIONAL(v, off, offset, condVarSuffix);
    READ_CONDITIONAL(v, gamma, gamma, condVarSuffix);
    READ_CONDITIONAL(v, cont, contrast, condVarSuffix);
}

#undef READ_CONDITIONAL

ColorYUV::ColorYUV(PClip child,
  double gain_y, double offset_y, double gamma_y, double contrast_y,
  double gain_u, double offset_u, double gamma_u, double contrast_u,
  double gain_v, double offset_v, double gamma_v, double contrast_v,
  const char* level, const char* opt,
  bool showyuv, bool analyse, bool autowhite, bool autogain, bool conditional,
  int bits, bool showyuv_fullrange,
  bool tweaklike_params, // ColorYUV2: 0.0/0.5/1.0/2.0/3.0 instead of -256/-128/0/256/512
  const char* condVarSuffix,
  bool optForceUseExpr,
  IScriptEnvironment* env)
  : GenericVideoFilter(child),
  colorbar_bits(showyuv ? bits : 0), colorbar_fullrange(showyuv_fullrange),
  analyse(analyse), autowhite(autowhite), autogain(autogain), conditional(conditional),
  tweaklike_params(tweaklike_params), condVarSuffix(condVarSuffix), optForceUseExpr(optForceUseExpr)
{
  luts[0] = luts[1] = luts[2] = nullptr;

  if (!vi.IsYUV() && !vi.IsYUVA())
    env->ThrowError("ColorYUV: Only work with YUV colorspace.");

  bool ColorRangeCanBeGuessed = false;
  if (gamma_y != 0) {
    ColorRangeCanBeGuessed = true;
    // when gamma is used then we must know the color range full/limited
    // and the default is full_scale if gamma is not zero
  }

  configY.gain = gain_y;
  configY.offset = offset_y;
  configY.gamma = gamma_y;
  configY.contrast = contrast_y;
  configY.changed = false;
  configY.clip_tv = false;
  configY.force_tv_range = false;
  configY.plane = PLANAR_Y;

  configU.gain = gain_u;
  configU.offset = offset_u;
  configU.gamma = gamma_u;
  configU.contrast = contrast_u;
  configU.changed = false;
  configU.clip_tv = false;
  configU.force_tv_range = false; // n/a. in chroma. For gamma
  configU.plane = PLANAR_U;

  configV.gain = gain_v;
  configV.offset = offset_v;
  configV.gamma = gamma_v;
  configV.contrast = contrast_v;
  configV.changed = false;
  configV.clip_tv = false;
  configV.force_tv_range = false; // n/a. in chroma. For gamma
  configV.plane = PLANAR_V;

  // Range
  if (lstrcmpi(level, "TV->PC") == 0)
  {
    ColorRangeCanBeGuessed = true; // will be full
    configV.range = configU.range = configY.range = COLORYUV_RANGE_TV_PC;
  }
  else if (lstrcmpi(level, "PC->TV") == 0)
  {
    ColorRangeCanBeGuessed = true;// will be limited
    configV.range = configU.range = configY.range = COLORYUV_RANGE_PC_TV;
  }
  else if (lstrcmpi(level, "PC->TV.Y") == 0)
  {   // ?
    ColorRangeCanBeGuessed = true;// will be limited
    configV.range = configU.range = COLORYUV_RANGE_NONE;
    configY.range = COLORYUV_RANGE_PC_TV;
  }
  else if (lstrcmpi(level, "TV") == 0)
  {
    // When no range conversion occurs only gamma correction
    // By this parameter we know it will be limited, this info is used for gamma adjustment
    ColorRangeCanBeGuessed = true;
    configV.force_tv_range = configU.force_tv_range = configY.force_tv_range = true;
  }
  else if (lstrcmpi(level, "") != 0)
  {
    env->ThrowError("ColorYUV: invalid parameter : levels");
  }
  else {
    configV.range = configU.range = configY.range = COLORYUV_RANGE_NONE;
  }

  // Option
  if (lstrcmpi(opt, "coring") == 0)
  {
    // note: this setting can conflict with e.g. TV->PC but we do not report an error
    ColorRangeCanBeGuessed = true; // used to set _ColorRange=limited only if not conversion mode is specified
    configY.clip_tv = true;
    configU.clip_tv = true;
    configV.clip_tv = true;
  }
  else if (lstrcmpi(opt, "") != 0)
  {
    env->ThrowError("ColorYUV: invalid parameter : opt");
  }

  if (showyuv) {
    if (colorbar_bits != 8 && colorbar_bits != 10 && colorbar_bits != 12 && colorbar_bits != 14 && colorbar_bits != 16 && colorbar_bits != 32)
      env->ThrowError("ColorYUV: bits parameter for showyuv must be 8, 10, 12, 14, 16 or 32");

    switch (colorbar_bits) {
    case 8: vi.pixel_type = VideoInfo::CS_YV12; break;
    case 10: vi.pixel_type = VideoInfo::CS_YUV420P10; break;
    case 12: vi.pixel_type = VideoInfo::CS_YUV420P12; break;
    case 14: vi.pixel_type = VideoInfo::CS_YUV420P14; break;
    case 16: vi.pixel_type = VideoInfo::CS_YUV420P16; break;
    case 32: vi.pixel_type = VideoInfo::CS_YUV420PS; break;
    }
    // pre-avs+: coloryuv_showyuv is always called with full_range false
    const int internal_bitdepth = colorbar_bits == 8 ? 8 : 10;
    const int chroma_span = colorbar_fullrange ? (1 << (internal_bitdepth - 1)) - 1 : (112 << (internal_bitdepth - 8)); // +-127/+-112
    const int chroma_range = 2 * chroma_span + 1; // 1..255 (+-127), 16..240 (+-112)
    // size limited to either 8 or 10 bits, independenly of 12/14/16 or 32bit float bit-depth
    vi.width = chroma_range << vi.GetPlaneWidthSubsampling(PLANAR_U);
    vi.height = vi.width;
    theColorRange = colorbar_fullrange ? ColorRange_e::AVS_RANGE_FULL : ColorRange_e::AVS_RANGE_LIMITED;
    theMatrix = Matrix_e::AVS_MATRIX_BT470_BG; // all the same
    theChromaLocation = ChromaLocation_e::AVS_CHROMA_CENTER; // default "mpeg1" for 4:2:0
    return;
  }

  // !showyuv, real filter

  auto frame0 = child->GetFrame(0, env);
  const AVSMap* props = env->getFramePropsRO(frame0);
  matrix_parse_merge_with_props_def(vi, nullptr, props, theMatrix, theColorRange,
    Matrix_e::AVS_MATRIX_UNSPECIFIED, // default matrix n/a
    configY.force_tv_range ? 
    ColorRange_e::AVS_RANGE_LIMITED :  
    ColorRangeCanBeGuessed ? ColorRange_e::AVS_RANGE_FULL : -1 /* n/a invalid */, env);
  // although we read _ColorRange full/limited, nothing stops us to feed with full-range clip a "TV->PC" conversion
  // Anyway: a frame property can set theColorRange from the default "FULL" to the actual one.
  switch (configY.range) {
  case COLORYUV_RANGE_PC_TV:
  case COLORYUV_RANGE_PC_TVY:
    theColorRange = ColorRange_e::AVS_RANGE_LIMITED;
    break;
  case COLORYUV_RANGE_TV_PC:
    theColorRange = ColorRange_e::AVS_RANGE_FULL;
    break;
  default:
    if (configY.force_tv_range) // "TV" overrides default "PC". Info is needed for gamma correction
      theColorRange = ColorRange_e::AVS_RANGE_LIMITED;
    else if (configY.clip_tv) // coring is also sets this frame property
      theColorRange = ColorRange_e::AVS_RANGE_LIMITED;
    break;
    // leave color range as is
  }
  // theMatrix and theColorRange will set frame properties in GetFrame

  // prepare basic LUT
  int pixelsize = vi.ComponentSize();
  int bits_per_pixel = vi.BitsPerComponent();

  if (pixelsize == 1 || pixelsize == 2) {
    // no float lut. float will be realtime
    int lut_size = pixelsize * (1 << bits_per_pixel); // 256*1 / 1024*2 .. 65536*2
    luts[0] = new BYTE[lut_size];
    if (!vi.IsY()) {
      luts[1] = new BYTE[lut_size];
      luts[2] = new BYTE[lut_size];
    }

    if (pixelsize == 1) {
      coloryuv_create_lut<uint8_t>(luts[0], &configY, bits_per_pixel, tweaklike_params);
      if (!vi.IsY())
      {
        coloryuv_create_lut<uint8_t>(luts[1], &configU, bits_per_pixel, tweaklike_params);
        coloryuv_create_lut<uint8_t>(luts[2], &configV, bits_per_pixel, tweaklike_params);
      }
    }
    else if (pixelsize == 2) { // pixelsize==2
      coloryuv_create_lut<uint16_t>(luts[0], &configY, bits_per_pixel, tweaklike_params);
      if (!vi.IsY())
      {
        coloryuv_create_lut<uint16_t>(luts[1], &configU, bits_per_pixel, tweaklike_params);
        coloryuv_create_lut<uint16_t>(luts[2], &configV, bits_per_pixel, tweaklike_params);
      }
    }
  }
}

ColorYUV::~ColorYUV() {
  if (luts[0]) delete[] luts[0];
  if (luts[1]) delete[] luts[1];
  if (luts[2]) delete[] luts[2];
}


PVideoFrame __stdcall ColorYUV::GetFrame(int n, IScriptEnvironment* env)
{
    if (colorbar_bits>0)
    {
        PVideoFrame dst = env->NewVideoFrame(vi);
        // no frame property source. It's like a source filter

        auto props = env->getFramePropsRW(dst);
        update_Matrix_and_ColorRange(props, theMatrix, theColorRange, env);
        update_ChromaLocation(props, theChromaLocation, env);

        // pre AVS+: full_range is always false
        // AVS+: showyuv_fullrange bool parameter
        // AVS+: bits parameter
        coloryuv_showyuv(dst->GetWritePtr(), dst->GetWritePtr(PLANAR_U), dst->GetWritePtr(PLANAR_V), dst->GetPitch(), dst->GetPitch(PLANAR_U), dst->GetPitch(PLANAR_V), n, colorbar_fullrange, colorbar_bits);
        return dst;
    }

    PVideoFrame src = child->GetFrame(n, env);
    PVideoFrame dst;

    int pixelsize = vi.ComponentSize();
    int bits_per_pixel = vi.BitsPerComponent();

    ColorYUVPlaneConfig // Yes, we copy these struct
        cY = configY,
        cU = configU,
        cV = configV;

    // for analysing data
    char text[512];

    if (analyse || autowhite || autogain)
    {
        ColorYUVPlaneData dY, dU, dV;

        if (vi.IsYUY2())
        {
            coloryuv_analyse_yuy2(src->GetReadPtr(), src->GetPitch(), vi.width, vi.height, &dY, &dU, &dV);
        }
        else
        {
            coloryuv_analyse_planar(src->GetReadPtr(), src->GetPitch(), vi.width, vi.height, &dY, bits_per_pixel, false); // false: not chroma
            if (!vi.IsY())
            {
                const int width = vi.width >> vi.GetPlaneWidthSubsampling(PLANAR_U);
                const int height = vi.height >> vi.GetPlaneHeightSubsampling(PLANAR_U);

                coloryuv_analyse_planar(src->GetReadPtr(PLANAR_U), src->GetPitch(PLANAR_U), width, height, &dU, bits_per_pixel, true); // true: chroma
                coloryuv_analyse_planar(src->GetReadPtr(PLANAR_V), src->GetPitch(PLANAR_V), width, height, &dV, bits_per_pixel, true);
            }
        }

        if (analyse)
        {
          if (!vi.IsY())
          {
            if (bits_per_pixel == 32)
              sprintf(text,
                "        Frame: %-8u ( Luma Y / ChromaU / ChromaV )\n"
                "      Average:      ( %7.5f / %7.5f / %7.5f )\n"
                "      Minimum:      ( %7.5f / %7.5f / %7.5f )\n"
                "      Maximum:      ( %7.5f / %7.5f / %7.5f )\n"
                "Loose Minimum:      ( %7.5f / %7.5f / %7.5f )\n"
                "Loose Maximum:      ( %7.5f / %7.5f / %7.5f )\n",
                n,
                dY.average, dU.average, dV.average,
                dY.real_min, dU.real_min, dV.real_min,
                dY.real_max, dU.real_max, dV.real_max,
                (float)dY.loose_min / 65535.0f, (float)dU.loose_min / 65535.0f, (float)dV.loose_min / 65535.0f,
                (float)dY.loose_max / 65535.0f, (float)dU.loose_max / 65535.0f, (float)dV.loose_max / 65535.0f
              );
            else
              sprintf(text,
                "        Frame: %-8u ( Luma Y / ChromaU / ChromaV )\n"
                "      Average:      ( %7.2f / %7.2f / %7.2f )\n"
                "      Minimum:      (  %5d  /  %5d  /  %5d   )\n"
                "      Maximum:      (  %5d  /  %5d  /  %5d   )\n"
                "Loose Minimum:      (  %5d  /  %5d  /  %5d   )\n"
                "Loose Maximum:      (  %5d  /  %5d  /  %5d   )\n",
                n,
                dY.average, dU.average, dV.average,
                (int)dY.real_min, (int)dU.real_min, (int)dV.real_min,
                (int)dY.real_max, (int)dU.real_max, (int)dV.real_max,
                dY.loose_min, dU.loose_min, dV.loose_min,
                dY.loose_max, dU.loose_max, dV.loose_max
              );
          }
          else
          {
            if (bits_per_pixel == 32)
              sprintf(text,
                "        Frame: %-8u\n"
                "      Average: %7.5f\n"
                "      Minimum: %7.5f\n"
                "      Maximum: %7.5f\n"
                "Loose Minimum: %7.5f\n"
                "Loose Maximum: %7.5f\n",
                n,
                dY.average,
                dY.real_min,
                dY.real_max,
                (float)dY.loose_min / 65535.0f,
                (float)dY.loose_max / 65535.0f
              );
            else
              sprintf(text,
                "        Frame: %-8u\n"
                "      Average: %7.2f\n"
                "      Minimum: %5d\n"
                "      Maximum: %5d\n"
                "Loose Minimum: %5d\n"
                "Loose Maximum: %5d\n",
                n,
                dY.average,
                (int)dY.real_min,
                (int)dY.real_max,
                dY.loose_min,
                dY.loose_max
              );
          }
        }

        if (autowhite && !vi.IsY())
        {
            coloryuv_autowhite(&dY, &dU, &dV, &cY, &cU, &cV, bits_per_pixel);
        }

        if (autogain)
        {
            coloryuv_autogain(&dY, &dU, &dV, &cY, &cU, &cV, bits_per_pixel, tweaklike_params);
        }
    }

    // Read conditional variables
    if(conditional)
      coloryuv_read_conditional(env, &cY, &cU, &cV, condVarSuffix);

    // no float lut. float will be realtime
    if ((pixelsize == 1 || pixelsize == 2) && !optForceUseExpr) {

      BYTE *luts_live[3] = { nullptr };

      BYTE *lutY = luts[0];
      BYTE *lutU = luts[1];
      BYTE *lutV = luts[2];

      // recalculate plane LUT only if changed
      int lut_size = pixelsize * (1 << bits_per_pixel); // 256*1 / 1024*2 .. 65536*2
      if (cY.changed) {
        luts_live[0] = new BYTE[lut_size];
        lutY = luts_live[0];
        if (pixelsize == 1)
          coloryuv_create_lut<uint8_t>(lutY, &cY, bits_per_pixel, tweaklike_params);
        else if (pixelsize == 2)
          coloryuv_create_lut<uint16_t>(lutY, &cY, bits_per_pixel, tweaklike_params);
      }

      if (!vi.IsY())
      {
        if (cU.changed) {
          luts_live[1] = new BYTE[lut_size];
          lutU = luts_live[1];
          if (pixelsize == 1)
            coloryuv_create_lut<uint8_t>(lutU, &cU, bits_per_pixel, tweaklike_params);
          else if (pixelsize == 2)
            coloryuv_create_lut<uint16_t>(lutU, &cU, bits_per_pixel, tweaklike_params);
        }
        if (cV.changed) {
          luts_live[2] = new BYTE[lut_size];
          lutV = luts_live[2];
          if (pixelsize == 1)
            coloryuv_create_lut<uint8_t>(lutV, &cV, bits_per_pixel, tweaklike_params);
          else if (pixelsize == 2)
            coloryuv_create_lut<uint16_t>(lutV, &cV, bits_per_pixel, tweaklike_params);
        }
      }
      dst = env->NewVideoFrameP(vi, &src);

      if (vi.IsYUY2())
      {
        coloryuv_apply_lut_yuy2(dst->GetWritePtr(), src->GetReadPtr(), dst->GetPitch(), src->GetPitch(), vi.width, vi.height, lutY, lutU, lutV);
      }
      else
      {
        coloryuv_apply_lut_planar(dst->GetWritePtr(), src->GetReadPtr(), dst->GetPitch(), src->GetPitch(), vi.width, vi.height, lutY, bits_per_pixel);
        if (!vi.IsY())
        {
          const int width = vi.width >> vi.GetPlaneWidthSubsampling(PLANAR_U);
          const int height = vi.height >> vi.GetPlaneHeightSubsampling(PLANAR_U);

          coloryuv_apply_lut_planar(dst->GetWritePtr(PLANAR_U), src->GetReadPtr(PLANAR_U), dst->GetPitch(PLANAR_U), src->GetPitch(PLANAR_U), width, height, lutU, bits_per_pixel);
          coloryuv_apply_lut_planar(dst->GetWritePtr(PLANAR_V), src->GetReadPtr(PLANAR_V), dst->GetPitch(PLANAR_V), src->GetPitch(PLANAR_V), width, height, lutV, bits_per_pixel);
        }
        if (vi.IsYUVA()) {
          env->BitBlt(dst->GetWritePtr(PLANAR_A), dst->GetPitch(PLANAR_A), src->GetReadPtr(PLANAR_A), src->GetPitch(PLANAR_A), src->GetRowSize(PLANAR_A), src->GetHeight(PLANAR_A));
        }
      }

      if (luts_live[0]) delete[] luts_live[0];
      if (luts_live[1]) delete[] luts_live[1];
      if (luts_live[2]) delete[] luts_live[2];
    } // lut create and use
    else {
      // 32 bit float: expr
      std::string exprY = coloryuv_create_lut_expr(&cY, bits_per_pixel, tweaklike_params);
      std::string exprU = !vi.IsY() ? coloryuv_create_lut_expr(&cU, bits_per_pixel, tweaklike_params) : "";
      std::string exprV = !vi.IsY() ? coloryuv_create_lut_expr(&cV, bits_per_pixel, tweaklike_params) : "";
      std::string exprA = ""; // copy
      // Invoke Expr
      AVSValue child2;
      if (vi.IsY()) {
        AVSValue new_args[2] = { child, exprY.c_str() };
        child2 = env->Invoke("Expr", AVSValue(new_args, 2)).AsClip();
      }
      else if(vi.IsYUV()) {
        AVSValue new_args[4] = { child, exprY.c_str(), exprU.c_str(), exprV.c_str() };
        child2 = env->Invoke("Expr", AVSValue(new_args, 4)).AsClip();
      }
      else if (vi.IsYUVA()) {
        AVSValue new_args[5] = { child, exprY.c_str(), exprU.c_str(), exprV.c_str(), exprA.c_str() };
        child2 = env->Invoke("Expr", AVSValue(new_args, 5)).AsClip();
      }
      dst = child2.AsClip()->GetFrame(n, env);
    }

    if (analyse)
    {
        env->ApplyMessage(&dst, vi, text, vi.width / 4, 0xa0a0a0, 0, 0);
    }

    // when there was no such property from constructor and it could not be guessed then we do not put one
    if (theColorRange == ColorRange_e::AVS_RANGE_FULL || theColorRange == ColorRange_e::AVS_RANGE_LIMITED) {
      auto props = env->getFramePropsRW(dst);
      update_ColorRange(props, theColorRange, env);
    }

    return dst;
}

AVSValue __cdecl ColorYUV::Create(AVSValue args, void* , IScriptEnvironment* env)
{
    const bool tweaklike_params = args[23].AsBool(false); // f2c = true: for tweak-like parameter interpretation
    const float def = tweaklike_params ? 1.0f : 0.0f;
    return new ColorYUV(args[0].AsClip(),
                        args[1].AsFloat(def),                // gain_y
                        args[2].AsFloat(0.0f),                // off_y      bright
                        args[3].AsFloat(def),                // gamma_y
                        args[4].AsFloat(def),                // cont_y
                        args[5].AsFloat(def),                // gain_u
                        args[6].AsFloat(0.0f),                // off_u      bright
                        args[7].AsFloat(def),                // gamma_u
                        args[8].AsFloat(def),                // cont_u
                        args[9].AsFloat(def),                // gain_v
                        args[10].AsFloat(0.0f),                // off_v
                        args[11].AsFloat(def),                // gamma_v
                        args[12].AsFloat(def),                // cont_v
                        args[13].AsString(""),                // levels = "", "TV->PC", "PC->TV"
                        args[14].AsString(""),                // opt = "", "coring"
//                      args[15].AsString(""),                // matrix = "", "rec.709"
                        args[16].AsBool(false),                // colorbars
                        args[17].AsBool(false),                // analyze
                        args[18].AsBool(false),                // autowhite
                        args[19].AsBool(false),                // autogain
                        args[20].AsBool(false),                // conditional
                        args[21].AsInt(8),                     // bits avs+
                        args[22].AsBool(false),                // showyuv_fullrange avs+
                        tweaklike_params,                      // for gain, gamma, cont: 0.0/0.5/1.0/2.0/3.0 instead of -256/-128/0/256/512
                        args[24].AsString(""),                // condvarsuffix avs+
                        args[25].AsBool(false),               // optForceUseExpr debug parameter
      env);
}

extern const AVSFunction Color_filters[] = {
    { "ColorYUV", BUILTIN_FUNC_PREFIX,
                  "c[gain_y]f[off_y]f[gamma_y]f[cont_y]f" \
                  "[gain_u]f[off_u]f[gamma_u]f[cont_u]f" \
                  "[gain_v]f[off_v]f[gamma_v]f[cont_v]f" \
                  "[levels]s[opt]s[matrix]s[showyuv]b" \
                  "[analyze]b[autowhite]b[autogain]b[conditional]b" \
                  "[bits]i[showyuv_fullrange]b[f2c]b[condvarsuffix]s[optForceUseExpr]b", // avs+ 20180226- f2c-like parameters
                  ColorYUV::Create},
    { 0 }
};

