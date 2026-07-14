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


#include "convert_matrix.h"
#include "convert_helper.h"
#include <avisynth.h>
#ifdef AVS_WINDOWS
#include <avs/win.h>
#else
#include <avs/posix.h>
#endif 
#include <cmath>

/**
 * SIMD scaling logic for PMADDWD (16-bit signed multipliers, 32-bit accumulators).
 * RGB->YUV uses 15-bit scale as all coeffs are < 1.0 (max ~0.678, BT.2020 G weight),
 *   fully utilizing 16-bit coefficient precision without overflow.
 * YUV->RGB requires 13-bit scale to accommodate chroma expansion coeffs up to ~1.881
 *   (BT.2020 Cb->B), keeping scaled coefficients within int16 range.
 * Fused YUV->YUV (e.g., 601->2020) needs 14-bit scale; diagonal gain slightly > 1.0
 *   (~1.03 max) rules out 15-bit, but coeffs are well within 13-bit headroom.
 * At 16-bit depth, errors partially average out during 32-bit accumulation before rounding.
 * Avoid unifying all paths to 13-bit: RGB->YUV would degrade from ~±1 to ~±12 LSB worst-case.
 *
 * | Conversion  | Scale  | Max Coeff | LSB Error (16-bit) | Worst-Case Matrix   |
 * |-------------|--------|-----------|--------------------|---------------------|
 * | RGB -> YUV  | 15-bit | ~0.678    | ~±1  (near-exact)  | BT.2020 (G weight)  |
 * | YUV -> RGB  | 13-bit | ~1.881    | ~±8  (practical)   | BT.2020 (Cb -> B)   |
 * | YUV -> YUV  | 14-bit | ~1.030    | ~±2  (low noise)   | 601 <-> 2020 fused  |
 *
 * matrix="RGB" (IDENTITY) exception: If Kr=Kb=0 (Identity), coeffs reach 1.0. (1.0 << 15) = 32768,
 * which overflows int16 (-32768 to 32767). Caller will check it and may diable int16 based SIMD optimization paths.
 */

static void BuildMatrix_Rgb2Yuv_core(double Kr, double Kb, int int_arith_shift, bool full_scale_s, bool full_scale_d, int bits_per_pixel, ConversionMatrix& matrix)
{
  bits_conv_constants luma, chroma;
  bits_conv_constants luma_to_32bit;

  // helpers for post-matrix conversion to 32-bit float (for high bit depth sources or targets, e.g. 16-bit to 32-bit float)
  get_bits_conv_constants(luma_to_32bit, false, full_scale_s, full_scale_d, bits_per_pixel, 32);
  matrix.target_span_f_32 = luma_to_32bit.dst_span;
  matrix.offset_out_f_32 = luma_to_32bit.dst_offset;

  // RGB is source, YUV is destination
  // For RGB source / Y destination (both luma-like):
  get_bits_conv_constants(luma, false, full_scale_s, full_scale_d, bits_per_pixel, bits_per_pixel);
  // For UV destination (chroma behavior):
  // Note: we only need dst_span for UV, so we use full_scale_d for both params
  get_bits_conv_constants(chroma, true, full_scale_d, full_scale_d, bits_per_pixel, bits_per_pixel);

  double Srgb_f = luma.src_span;      // RGB input range
  double Sy_f = luma.dst_span;        // Y output range  
  double Suv_f = chroma.dst_span;     // UV output range
  double Orgb_f = luma.src_offset;    // RGB input offset
  double Oy_f = luma.dst_offset;      // Y output offset

  // Derive integer versions (for <= 16 bit paths)
  int Orgb = (int)Orgb_f;
  int Oy = (int)Oy_f;

  /*
    Kr   = {0.299, 0.2126}
    Kb   = {0.114, 0.0722}
    Kg   = 1 - Kr - Kb // {0.587, 0.7152}
    Srgb = 255
    Sy   = {219, 255}   // { 235-16, 255-0 }
    Suv  = {112, 127}   // { (240-16)/2, (255-0)/2 }
    Oy   = {16, 0}
    Ouv  = 128
    R = r/Srgb                     // 0..1
    G = g/Srgb
    B = b/Srgb
    Y = Kr*R + Kg*G + Kb*B         // 0..1
    U = B - (Kr*R + Kg*G)/(1-Kb)   //-1..1
    V = R - (Kg*G + Kb*B)/(1-Kr)
    y = Y*Sy  + Oy                 // 16..235, 0..255
    u = U*Suv + Ouv                // 16..240, 1..255
    v = V*Suv + Ouv
  */

  const int mulfac_int = 1 << int_arith_shift;
  const double mulfac = double(mulfac_int);
  const double Kg = 1. - Kr - Kb;

  // Symmetric rounding for both positive and negative coefficients
  auto round_coeff = [](double v) {
    return (int)(v >= 0 ? (v + 0.5) : (v - 0.5));
    };

  // Calculate double-precision coefficients
  double y_b_f = Sy_f * Kb / Srgb_f;
  double y_g_f = Sy_f * Kg / Srgb_f;
  double y_r_f = Sy_f * Kr / Srgb_f;

  double u_b_f = Suv_f / Srgb_f;
  double u_g_f = Suv_f * Kg / (Kb - 1) / Srgb_f;
  double u_r_f = Suv_f * Kr / (Kb - 1) / Srgb_f;

  double v_b_f = Suv_f * Kb / (Kr - 1) / Srgb_f;
  double v_g_f = Suv_f * Kg / (Kr - 1) / Srgb_f;
  double v_r_f = Suv_f / Srgb_f;

  double offset_y_f = Oy_f;
  double offset_rgb_f = -Orgb_f;  // Negative because addition is used

  // Store float versions
  matrix.y_b_f = (float)y_b_f;
  matrix.y_g_f = (float)y_g_f;
  matrix.y_r_f = (float)y_r_f;
  matrix.u_b_f = (float)u_b_f;
  matrix.u_g_f = (float)u_g_f;
  matrix.u_r_f = (float)u_r_f;
  matrix.v_b_f = (float)v_b_f;
  matrix.v_g_f = (float)v_g_f;
  matrix.v_r_f = (float)v_r_f;
  matrix.offset_y_f = (float)offset_y_f;
  matrix.offset_rgb_f = (float)offset_rgb_f;

  if (bits_per_pixel <= 16) {
    // Derive integer versions from doubles with proper rounding
    matrix.y_b = round_coeff(mulfac * y_b_f);
    matrix.y_g = round_coeff(mulfac * y_g_f);
    matrix.y_r = round_coeff(mulfac * y_r_f);

    matrix.u_b = round_coeff(mulfac * u_b_f);
    matrix.u_g = round_coeff(mulfac * u_g_f);
    matrix.u_r = round_coeff(mulfac * u_r_f);

    matrix.v_b = round_coeff(mulfac * v_b_f);
    matrix.v_g = round_coeff(mulfac * v_g_f);
    matrix.v_r = round_coeff(mulfac * v_r_f);

    matrix.offset_y = Oy;
    matrix.offset_rgb = -Orgb;  // negative because addition is used

    // Luma gain check: ensure Y captures 100% of RGB energy
    // Only applies when destination is full-range (no offset)
    // and the source RGB is similarly full_range
    if (full_scale_s && full_scale_d) {
      int y_sum = matrix.y_b + matrix.y_g + matrix.y_r;
      if (y_sum != mulfac_int) {
        matrix.y_g += (mulfac_int - y_sum);
      }
    }

    // U neutrality check: ensure R=G=B results in U = 0 (before offset)
    int u_sum = matrix.u_b + matrix.u_g + matrix.u_r;
    if (u_sum != 0) {
      matrix.u_g -= u_sum;
    }

    // V neutrality check: ensure R=G=B results in V = 0 (before offset)
    int v_sum = matrix.v_b + matrix.v_g + matrix.v_r;
    if (v_sum != 0) {
      matrix.v_g -= v_sum;
    }

  }

  // in: rgb. out: yuv
  matrix.offset_in = matrix.offset_rgb;
  matrix.offset_in_f = matrix.offset_rgb_f;
  matrix.offset_out = matrix.offset_y;
  matrix.offset_out_f = matrix.offset_y_f;
}

/*
 * WARNING: int_arith_shift MUST NOT exceed 13 for YUV -> RGB expansion.
 * Example: BT.709 Limited -> Full Range
 * The Blue expansion factor (u_b_f) is ~2.112.
 * - At 14-bit shift: 2.112 * 16384 = 34603 (OVERFLOWS int16_t)
 * - At 13-bit shift: 2.112 * 8192  = 17302 (SAFE)
 * Use 13-bit shift to ensure coefficients fit in int16 for SIMD paths.
 * Additionally, summing up to 3 components (Y, U, V) plus rounding constant must
 * not overflow int32 accumulators in SIMD. (another 2 bits headroom needed)
*/
static void BuildMatrix_Yuv2Rgb_core(double Kr, double Kb, int int_arith_shift, bool full_scale_s, bool full_scale_d, int bits_per_pixel, ConversionMatrix& matrix)
{
  float Sy_f, Suv_f, Oy_f, Orgb_f;

  bits_conv_constants luma, chroma;
  bits_conv_constants luma_to_32bit;

  // helpers for post-matrix conversion to 32-bit float (for high bit depth sources or targets, e.g. 16-bit to 32-bit float)
  get_bits_conv_constants(luma_to_32bit, false, full_scale_s, full_scale_d, bits_per_pixel, 32);
  matrix.target_span_f_32 = luma_to_32bit.dst_span;
  matrix.offset_out_f_32 = luma_to_32bit.dst_offset;

  // We use dstBitDepth = srcBitDepth because the matrix handles the magnitude 
  // via the mulfac and Srgb calculations. We just need the standardized spans.
  get_bits_conv_constants(luma, false, full_scale_s, full_scale_d, bits_per_pixel, bits_per_pixel);
  get_bits_conv_constants(chroma, true, full_scale_s, full_scale_d, bits_per_pixel, bits_per_pixel);

  matrix.target_span_f = luma.dst_span;

  Sy_f = luma.src_span;
  Suv_f = chroma.src_span;
  Oy_f = luma.src_offset;
  Orgb_f = luma.dst_offset;

  /*
    Kr   = {0.299, 0.2126}
    Kb   = {0.114, 0.0722}
    Kg   = 1 - Kr - Kb // {0.587, 0.7152}
    Srgb = 255
    Sy   = {219, 255}   // { 235-16, 255-0 }
    Suv  = {112, 127}   // { (240-16)/2, (255-0)/2 }
    Oy   = {16, 0}
    Ouv  = 128

    Y =(y-Oy)  / Sy                         // 0..1
    U =(u-Ouv) / Suv                        //-1..1
    V =(v-Ouv) / Suv

    R = Y                  + V*(1-Kr)       // 0..1
    G = Y - U*(1-Kb)*Kb/Kg - V*(1-Kr)*Kr/Kg
    B = Y + U*(1-Kb)

    r = R*Srgb                              // 0..255   0..65535
    g = G*Srgb
    b = B*Srgb
  */


  const double mulfac = (double)(1 << int_arith_shift); // integer aritmetic precision scale

  const double Kg = 1. - Kr - Kb;

  // The Srgb (destination span) is also just the dst_span (RGB is luma-like)!
  const float Srgb_f = (float)luma.dst_span;

  // symmetric rounding for both positive and negative coefficients
  auto round_coeff = [](double v) {
    return (int)(v >= 0 ? (v + 0.5) : (v - 0.5));
    };

  double y_b_f = (Srgb_f * 1.000 / Sy_f); //Y
  double u_b_f = (Srgb_f * (1 - Kb) / Suv_f); //U
  double v_b_f = (Srgb_f * 0.000 / Suv_f); //V
  double y_g_f = (Srgb_f * 1.000 / Sy_f);
  double u_g_f = (Srgb_f * (Kb - 1) * Kb / Kg / Suv_f);
  double v_g_f = (Srgb_f * (Kr - 1) * Kr / Kg / Suv_f);
  double y_r_f = (Srgb_f * 1.000 / Sy_f);
  double u_r_f = (Srgb_f * 0.000 / Suv_f);
  double v_r_f = (Srgb_f * (1 - Kr) / Suv_f);
  double offset_y_f = -Oy_f; // negative, it will be added in the conversion, so we store the negative here
  double offset_rgb_f = Orgb_f;

  matrix.y_b_f = (float)(y_b_f);
  matrix.u_b_f = (float)(u_b_f);
  matrix.v_b_f = (float)(v_b_f);
  matrix.y_g_f = (float)(y_g_f);
  matrix.u_g_f = (float)(u_g_f);
  matrix.v_g_f = (float)(v_g_f);
  matrix.y_r_f = (float)(y_r_f);
  matrix.u_r_f = (float)(u_r_f);
  matrix.v_r_f = (float)(v_r_f);
  matrix.offset_y_f = (float)offset_y_f;
  matrix.offset_rgb_f = (float)offset_rgb_f;

  matrix.y_b = round_coeff(mulfac * y_b_f);
  matrix.u_b = round_coeff(mulfac * u_b_f);
  matrix.v_b = round_coeff(mulfac * v_b_f);
  matrix.y_g = round_coeff(mulfac * y_g_f);
  matrix.u_g = round_coeff(mulfac * u_g_f);
  matrix.v_g = round_coeff(mulfac * v_g_f);
  matrix.y_r = round_coeff(mulfac * y_r_f);
  matrix.u_r = round_coeff(mulfac * u_r_f);
  matrix.v_r = round_coeff(mulfac * v_r_f);
  matrix.offset_y = round_coeff(offset_y_f);
  matrix.offset_rgb = round_coeff(offset_rgb_f);

  // in: yuv. out: rgb
  matrix.offset_in = matrix.offset_y;
  matrix.offset_in_f = matrix.offset_y_f;
  matrix.offset_out = matrix.offset_rgb;
  matrix.offset_out_f = matrix.offset_rgb_f;

}

bool GetKrKb(int matrix, double& Kr, double& Kb)
{
  switch (matrix) {
  case AVS_MATRIX_BT470_BG:
  case AVS_MATRIX_ST170_M:    Kr = 0.299;  Kb = 0.114;  return true;
  case AVS_MATRIX_BT709:      Kr = 0.2126; Kb = 0.0722; return true;
  case AVS_MATRIX_BT2020_NCL:
  case AVS_MATRIX_BT2020_CL:  Kr = 0.2627; Kb = 0.0593; return true;
  case AVS_MATRIX_BT470_M:    Kr = 0.3;    Kb = 0.11;   return true;
  case AVS_MATRIX_ST240_M:    Kr = 0.212;  Kb = 0.087;  return true;
  case AVS_MATRIX_AVERAGE:    Kr = 1.0 / 3;  Kb = 1.0 / 3;  return true;
  default: return false;
  }
}

bool do_BuildMatrix_Rgb2Yuv(int _Matrix, int _ColorRange, int _ColorRange_Out, int int_arith_shift, int bits_per_pixel, ConversionMatrix& matrix)
{
  if (_ColorRange != ColorRange_Compat_e::AVS_COLORRANGE_FULL && _ColorRange != ColorRange_Compat_e::AVS_COLORRANGE_LIMITED)
    return false;
  if (_ColorRange_Out != ColorRange_Compat_e::AVS_COLORRANGE_FULL && _ColorRange_Out != ColorRange_Compat_e::AVS_COLORRANGE_LIMITED)
    return false;

  const bool is_full_s = _ColorRange == ColorRange_Compat_e::AVS_COLORRANGE_FULL;
  const bool is_full_d = _ColorRange_Out == ColorRange_Compat_e::AVS_COLORRANGE_FULL;

  // Special cases not handled by GetKrKb
  if (_Matrix == Matrix_e::AVS_MATRIX_RGB) {
    // copies Green to Y and sets UV to 0
    BuildMatrix_Rgb2Yuv_core(0.0, 0.0, int_arith_shift, is_full_s, is_full_d, bits_per_pixel, matrix);
    return true;
  }
  if (_Matrix == Matrix_e::AVS_MATRIX_ICTCP || _Matrix == Matrix_e::AVS_MATRIX_YCGCO)
    return false; // not supported

  double Kr, Kb;
  if (!GetKrKb(_Matrix, Kr, Kb))
    return false;

  BuildMatrix_Rgb2Yuv_core(Kr, Kb, int_arith_shift, is_full_s, is_full_d, bits_per_pixel, matrix);
  return true;
}

bool do_BuildMatrix_Yuv2Rgb(int _Matrix, int _ColorRange, int _ColorRange_Out, int int_arith_shift, int bits_per_pixel, ConversionMatrix& matrix)
{
  if (_ColorRange != ColorRange_Compat_e::AVS_COLORRANGE_FULL && _ColorRange != ColorRange_Compat_e::AVS_COLORRANGE_LIMITED)
    return false;
  if (_ColorRange_Out != ColorRange_Compat_e::AVS_COLORRANGE_FULL && _ColorRange_Out != ColorRange_Compat_e::AVS_COLORRANGE_LIMITED)
    return false;

  const bool is_full_s = _ColorRange == ColorRange_Compat_e::AVS_COLORRANGE_FULL;
  const bool is_full_d = _ColorRange_Out == ColorRange_Compat_e::AVS_COLORRANGE_FULL;

  // Special cases not handled by GetKrKb
  if (_Matrix == Matrix_e::AVS_MATRIX_RGB) {
    BuildMatrix_Yuv2Rgb_core(0.0, 0.0, int_arith_shift, is_full_s, is_full_d, bits_per_pixel, matrix);
    return true;
  }
  if (_Matrix == Matrix_e::AVS_MATRIX_ICTCP || _Matrix == Matrix_e::AVS_MATRIX_YCGCO)
    return false; // not supported

  double Kr, Kb;
  if (!GetKrKb(_Matrix, Kr, Kb))
    return false;

  BuildMatrix_Yuv2Rgb_core(Kr, Kb, int_arith_shift, is_full_s, is_full_d, bits_per_pixel, matrix);
  return true;
}
