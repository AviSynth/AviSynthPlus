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

#ifndef __Convert_helper_H__
#define __Convert_helper_H__

#include <avisynth.h>
#include <string>
#include <cstring>

typedef enum ColorRange_e {
  AVS_RANGE_FULL = 0,
  AVS_RANGE_LIMITED = 1
} ColorRange_e;

typedef enum ChromaLocation_e {
  AVS_CHROMA_LEFT = 0,
  AVS_CHROMA_CENTER = 1,
  AVS_CHROMA_TOP_LEFT = 2,
  AVS_CHROMA_TOP = 3,
  AVS_CHROMA_BOTTOM_LEFT = 4,
  AVS_CHROMA_BOTTOM = 5,
  AVS_CHROMA_DV = 6 // Special to Avisynth
} ChromaLocation_e;

typedef enum FieldBased_e {
  AVS_FIELD_PROGRESSIVE = 0,
  AVS_FIELD_BOTTOM = 1,
  AVS_FIELD_TOP = 2
} FieldBased_e;

// https://www.itu.int/rec/T-REC-H.265-202108-I
/* ITU-T H.265 Table E.5 */
typedef enum Matrix_e {
  AVS_MATRIX_RGB = 0, /* The identity matrix. Typically used for RGB, may also be used for XYZ */
  AVS_MATRIX_BT709 = 1, /* ITU-R Rec. BT.709-5 */
  AVS_MATRIX_UNSPECIFIED = 2, /* Image characteristics are unknown or are determined by the application */
  AVS_MATRIX_BT470_M = 4, // instead of AVS_MATRIX_FCC
  // FCC Title 47 Code of Federal Regulations (2003) 73.682 (a) (20)
  // Rec. ITU-R BT.470-6 System M (historical)
  AVS_MATRIX_BT470_BG = 5, /* Equivalent to 6. */
  // ITU-R Rec. BT.470-6 System B, G (historical)
  // Rec. ITU-R BT.601-7 625
  // Rec. ITU-R BT.1358-0 625 (historical)
  // Rec. ITU-R BT.1700-0 625 PAL and 625 SECAM
  AVS_MATRIX_ST170_M = 6,  /* Equivalent to 5. */
  // Rec. ITU-R BT.601-7 525
  // Rec. ITU-R BT.1358-1 525 or 625 (historical)
  // Rec. ITU-R BT.1700-0 NTSC
  // SMPTE ST 170 (2004)
  // SMPTE 170M (2004)
  AVS_MATRIX_ST240_M = 7, // SMPTE ST 240 (1999, historical)
  AVS_MATRIX_YCGCO = 8,
  AVS_MATRIX_BT2020_NCL = 9, 
  // Rec. ITU-R BT.2020 non-constant luminance system
  // Rec. ITU-R BT.2100-2 Y'CbCr
  AVS_MATRIX_BT2020_CL = 10, /* Rec. ITU-R BT.2020 constant luminance system */
  AVS_MATRIX_CHROMATICITY_DERIVED_NCL = 12, /* Chromaticity derived non-constant luminance system */
  AVS_MATRIX_CHROMATICITY_DERIVED_CL = 13, /* Chromaticity derived constant luminance system */
  AVS_MATRIX_ICTCP = 14, // REC_2100_ICTCP, Rec. ITU-R BT.2100-2 ICTCP
  AVS_MATRIX_AVERAGE = 9999, // Avisynth compatibility
} Matrix_e;

// Pre-Avisynth 3.7.1 matrix constants, with implicite PC/limited range
typedef enum Old_Avs_Matrix_e {
  AVS_OLD_MATRIX_Rec601 = 0, 
  AVS_OLD_MATRIX_Rec709 = 1,
  AVS_OLD_MATRIX_PC_601 = 2,
  AVS_OLD_MATRIX_PC_709 = 3,
  AVS_OLD_MATRIX_AVERAGE = 4,
  AVS_OLD_MATRIX_Rec2020 = 5,
  AVS_OLD_MATRIX_PC_2020 = 6
} Old_Avs_Matrix_e;

// transfer characteristics ITU-T H.265 Table E.4
typedef enum Transfer_e {
  AVS_TRANSFER_BT709 = 1,
  AVS_TRANSFER_UNSPECIFIED = 2,
  AVS_TRANSFER_BT470_M = 4,
  AVS_TRANSFER_BT470_BG = 5,
  AVS_TRANSFER_BT601 = 6,  /* Equivalent to 1. */
  AVS_TRANSFER_ST240_M = 7,
  AVS_TRANSFER_LINEAR = 8,
  AVS_TRANSFER_LOG_100 = 9,
  AVS_TRANSFER_LOG_316 = 10,
  AVS_TRANSFER_IEC_61966_2_4 = 11,
  AVS_TRANSFER_IEC_61966_2_1 = 13,
  AVS_TRANSFER_BT2020_10 = 14, /* Equivalent to 1. */
  AVS_TRANSFER_BT2020_12 = 15, /* Equivalent to 1. */
  AVS_TRANSFER_ST2084 = 16,
  AVS_TRANSFER_ARIB_B67 = 18
} Transfer_e;

// color primaries ITU-T H.265 Table E.3
typedef enum Primaries_e {
  AVS_PRIMARIES_BT709 = 1,
  AVS_PRIMARIES_UNSPECIFIED = 2,
  AVS_PRIMARIES_BT470_M = 4,
  AVS_PRIMARIES_BT470_BG = 5,
  AVS_PRIMARIES_ST170_M = 6,
  AVS_PRIMARIES_ST240_M = 7,  /* Equivalent to 6. */
  AVS_PRIMARIES_FILM = 8,
  AVS_PRIMARIES_BT2020 = 9,
  AVS_PRIMARIES_ST428 = 10,
  AVS_PRIMARIES_ST431_2 = 11,
  AVS_PRIMARIES_ST432_1 = 12,
  AVS_PRIMARIES_EBU3213_E = 22
} Primaries_e;

void matrix_parse_merge_with_props(VideoInfo &vi, const char* matrix_name, const AVSMap* props, int& _Matrix, int& _ColorRange, IScriptEnvironment* env);
void matrix_parse_merge_with_props_def(VideoInfo& vi, const char* matrix_name, const AVSMap* props, int& _Matrix, int& _ColorRange, int _Matrix_Default, int _ColorRange_Default, IScriptEnvironment* env);
void chromaloc_parse_merge_with_props(VideoInfo& vi, const char* chromaloc_name, const AVSMap* props, int& _ChromaLocation, int _ChromaLocation_Default, IScriptEnvironment* env);

void update_Matrix_and_ColorRange(AVSMap* props, int theMatrix, int theColorRange, IScriptEnvironment* env);
void update_ColorRange(AVSMap* props, int theColorRange, IScriptEnvironment* env);
void update_ChromaLocation(AVSMap* props, int theChromaLocation, IScriptEnvironment* env);

typedef struct bits_conv_constants {
  float src_offset = 0.0f;
  int src_offset_i = 0;
  float mul_factor = 1.0f;
  float dst_offset = 0.0f;
} bits_conv_constants;

// universal transform values for any full-limited bit-depth mix
[[maybe_unused]] static AVS_FORCEINLINE void get_bits_conv_constants(bits_conv_constants& d, bool use_chroma, bool fulls, bool fulld, int srcBitDepth, int dstBitDepth)
{
  d.src_offset = 0.0f;
  d.mul_factor = 1.0f;
  d.dst_offset = 0.0f;
  float src_span = 1.0f;
  float dst_span = 1.0f;

  // possible usage places
  // Expr (autoscale, scalef, scaleb)
  // convert_bits
  // ColorYUV
  // Histogram

  if (srcBitDepth != dstBitDepth || fulls != fulld) {
    // fulls != fulld
    if (use_chroma) {
      // chroma: go into signed world, factor, then go back to biased range
      d.src_offset = (srcBitDepth == 32) ? 0.0f : (1 << (srcBitDepth - 1));
      d.dst_offset = (dstBitDepth == 32) ? 0.0f : (1 << (dstBitDepth - 1));
      // decision: 'limited' range float +/-112 is +/-112/255.0. Must be consistent with Expr, ColorYUV etc
      // 3.7.2: full range chroma: as per ITU Rec H.273 eq.30 p.10: Cb=Clip1_C(Round(((1<<BitDepth_C)-1)*E'_PB)+(1<<(BitDepth_C-1)))
      // For 8 bit this results in 128 +/-127.5 instead of 128 +/-127
      // In general the span is ((1 << srcBitDepth) - 1) / 2.0 instead of (1 << (srcBitDepth - 1)) - 1
      src_span = (srcBitDepth == 32) ?
        (fulls ? 0.5f : 112 / 255.0f) :
        (fulls ? (float)((1 << srcBitDepth) - 1) / 2.0f : (float)(112 << (srcBitDepth - 8)));
      dst_span = (dstBitDepth == 32) ?
        (fulld ? 0.5f : 112 / 255.0f) :
        (fulld ? (float)((1 << dstBitDepth) - 1) / 2.0f : (float)(112 << (dstBitDepth - 8)));

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
    }
    */

    }
    else {
      // luma
      d.src_offset = (srcBitDepth == 32) ?
        (fulls ? 0 : 16.0f / 255) :
        (fulls ? 0 : (16 << (srcBitDepth - 8)));
      d.dst_offset = (dstBitDepth == 32) ?
        (fulld ? 0 : 16.0f / 255) :
        (fulld ? 0 : (16 << (dstBitDepth - 8)));

      // false-false
      //src_span = (srcBitDepth == 32) ? 1.0f : (255 * (float)(1 << (srcBitDepth - 8)));
      //dst_span = (dstBitDepth == 32) ? 1.0f : (255 * (float)(1 << (dstBitDepth - 8)));

      src_span = (srcBitDepth == 32) ?
        (fulls ? 1.0f : 219 / 255.0f) :
        (fulls ? ((1 << srcBitDepth) - 1) : (219 << (srcBitDepth - 8))); // 0..255, 16..235
      dst_span = (dstBitDepth == 32) ?
        (fulld ? 1.0f : 219 / 255.0f) :
        (fulld ? ((1 << dstBitDepth) - 1) : (219 << (dstBitDepth - 8)));
    }
  }

  d.mul_factor = dst_span / src_span;
  d.src_offset_i = (int)d.src_offset; // no rounding, when used, it is integer
}


#endif  // __Convert_helper_H__
