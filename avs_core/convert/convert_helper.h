// Avisynth v2.5.  Copyright 2002 Ben Rudiak-Gould et al.
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

#ifndef __Convert_helper_H__
#define __Convert_helper_H__

#include <avisynth.h>
#include <string>
#include <cstring>
#include "frame_prop_enums.h"

void matrix_parse_merge_with_props(bool rgb_in, bool rgb_out, const char* matrix_name, const AVSMap* props, int& _Matrix, int& _ColorRange, int& ColorRange_Out, IScriptEnvironment* env);
void matrix_parse_merge_with_props_def(bool rgb_in, bool rgb_out, const char* matrix_name, const AVSMap* props, int& _Matrix, int& _ColorRange, int& ColorRange_Out, int _Matrix_Default, int _ColorRange_Default, IScriptEnvironment* env);
void chromaloc_parse_merge_with_props(VideoInfo& vi, const char* chromaloc_name, const AVSMap* props, int& _ChromaLocation, int _ChromaLocation_Default, IScriptEnvironment* env);

void update_Matrix_and_ColorRange(AVSMap* props, int theMatrix, int theColorRange, IScriptEnvironment* env);
void update_Transfer_and_Primaries(AVSMap* props, int theTransfer, int thePrimaries, IScriptEnvironment* env);
void update_ColorRange(AVSMap* props, int theColorRange, IScriptEnvironment* env);
void update_ChromaLocation(AVSMap* props, int theChromaLocation, IScriptEnvironment* env);

typedef struct bits_conv_constants {
  float src_offset = 0.0f;
  int src_offset_i = 0;
  float mul_factor = 1.0f;
  float dst_offset = 0.0f;
  float src_span = 1.0f;
  float dst_span = 1.0f;
} bits_conv_constants;

// universal transform values for any full-limited bit-depth mix
[[maybe_unused]] static AVS_FORCEINLINE void get_bits_conv_constants(bits_conv_constants& d, bool use_chroma, bool fulls, bool fulld, int srcBitDepth, int dstBitDepth)
{
  d.src_offset = 0.0f;
  d.mul_factor = 1.0f;
  d.dst_offset = 0.0f;
  d.src_span = 1.0f;
  d.dst_span = 1.0f;

  // possible usage places
  // Expr (autoscale, scalef, scaleb)
  // convert_bits
  // ColorYUV
  // Histogram
  // YUV->RGB conversions (matrix coefficients depend on full-limited, but also the scaling of the actual pixel values, so this is needed for both matrix and pixel value conversion)

  if (use_chroma) {
    // decision: 'limited' range float +/-112 is +/-112/255.0. Must be consistent with Expr, ColorYUV etc
    // 3.7.2: full range chroma: as per ITU Rec H.273 eq.37 p.10: Cb=Clip1_C(Round(((1<<BitDepth_C)-1)*E'_PB)+(1<<(BitDepth_C-1)))
    // For 8 bit this results in 128 +/-127.5 instead of 128 +/-127
    // In general the span is ((1 << srcBitDepth) - 1) / 2.0 instead of (1 << (srcBitDepth - 1)) - 1

    // a bit asymmetric but meets mpeg, jpeg, Rec.2020 industry standards:
    // full range chroma span intentionally is not 128 but 255/2.0, which is virtually 127.5.
    // This also gives identical results to e.g. zimg when used in full range YUV-RGB conversions, which is good for consistency.
    d.src_span = (srcBitDepth == 32) ?
      (fulls ? 0.5f : 112 / 255.0f) :
      (fulls ? (float)((1 << srcBitDepth) - 1) / 2.0f : (float)(112 << (srcBitDepth - 8)));
    d.dst_span = (dstBitDepth == 32) ?
      (fulld ? 0.5f : 112 / 255.0f) :
      (fulld ? (float)((1 << dstBitDepth) - 1) / 2.0f : (float)(112 << (dstBitDepth - 8)));
    // chroma use case: go into signed world, factor, then go back to biased range
    d.src_offset = (srcBitDepth == 32) ? 0.0f : (1 << (srcBitDepth - 1));
    d.dst_offset = (dstBitDepth == 32) ? 0.0f : (1 << (dstBitDepth - 1));

    /*
    fulls=fulld=false case:
    // mul/div by 2^N when between 8-16 bit integer formats
    // 255*: for consistent float conversion. scale (shift) to/from 8 bit. int-float is always 0..1.0 <-> 0..255
    // int-float: 10+ bits first downshift to 8 bits then /255.
    // float-int: 10+ bits first *255 to have 0..255 range, then shift from 8 bits to actual bit depth
    // Since first we convert to 8 bit 0-255 range and do bit-shift after then to >8 depths, the 1.0 number will show up as 255*256 in 16 bits.
    // This is normal. Like float32.ConvertBits(8).ConvertBits(16)
    (srcBitDepth == 32) -> (255 * (float)(1 << (dstBitDepth - 8)) / 1.0
    (dstBitDepth == 32) -> 1.0 / (255 * (float)(1 << (srcBitDepth - 8)))

    (srcBitDepth == 32) -> ((float)(1 << (dstBitDepth - 8)))) / (1 / 255.0f)
    (dstBitDepth == 32) -> 1.0 / 255.0f / (float)(1 << (srcBitDepth - 8))))
    */
  }
  else {
    // luma
    d.src_span = (srcBitDepth == 32) ?
      (fulls ? 1.0f : 219 / 255.0f) :
      (fulls ? ((1 << srcBitDepth) - 1) : (219 << (srcBitDepth - 8))); // 0..255, 16..235
    d.dst_span = (dstBitDepth == 32) ?
      (fulld ? 1.0f : 219 / 255.0f) :
      (fulld ? ((1 << dstBitDepth) - 1) : (219 << (dstBitDepth - 8)));
    // luma
    d.src_offset = (srcBitDepth == 32) ?
      (fulls ? 0 : 16.0f / 255) :
      (fulls ? 0 : (16 << (srcBitDepth - 8)));
    d.dst_offset = (dstBitDepth == 32) ?
      (fulld ? 0 : 16.0f / 255) :
      (fulld ? 0 : (16 << (dstBitDepth - 8)));
  }

  d.mul_factor = d.dst_span / d.src_span;
  d.src_offset_i = (int)d.src_offset; // no rounding, when used, it is integer
}

enum class ConversionDirection {
  YUV_TO_RGB,
  RGB_TO_YUV,
  YUV_TO_YUV,
  RGB_TO_RGB,
  RGB_TO_Y
};

#endif  // __Convert_helper_H__
