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


#include "convert_matrix.h"
#include "convert_helper.h"
#include <avisynth.h>
#ifdef AVS_WINDOWS
#include <avs/win.h>
#else
#include <avs/posix.h>
#endif 

static void BuildMatrix_Rgb2Yuv_core(double Kr, double Kb, int int_arith_shift, bool full_scale, int bits_per_pixel, ConversionMatrix& matrix)
{
  int Sy, Suv, Oy;
  float Sy_f, Suv_f, Oy_f;

  if (bits_per_pixel <= 16) {
    Oy = full_scale ? 0 : (16 << (bits_per_pixel - 8));
    Oy_f = (float)Oy; // for 16 bits

    int ymin = (full_scale ? 0 : 16) << (bits_per_pixel - 8);
    int max_pixel_value = (1 << bits_per_pixel) - 1;
    int ymax = full_scale ? max_pixel_value : (235 << (bits_per_pixel - 8));
    Sy = ymax - ymin;
    Sy_f = (float)Sy;

    int cmin = full_scale ? 0 : (16 << (bits_per_pixel - 8));
    int cmax = full_scale ? max_pixel_value : (240 << (bits_per_pixel - 8));
    Suv = (cmax - cmin) / 2;
    Suv_f = (cmax - cmin) / 2.0f;

  }
  else {
    Oy_f = full_scale ? 0.0f : (16.0f / 255.0f);
    Oy = full_scale ? 0 : 16; // n/a

    Sy_f = full_scale ? c8tof(255) : (c8tof(235) - c8tof(16));
    Suv_f = full_scale ? (0.5f - -0.5f) / 2 : (uv8tof(240) - uv8tof(16)) / 2;
  }


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
    B = b*Srgb

    Y = Kr*R + Kg*G + Kb*B         // 0..1
    U = B - (Kr*R + Kg*G)/(1-Kb)   //-1..1
    V = R - (Kg*G + Kb*B)/(1-Kr)

    y = Y*Sy  + Oy                 // 16..235, 0..255
    u = U*Suv + Ouv                // 16..240, 1..255
    v = V*Suv + Ouv
  */
  const int mulfac_int = 1 << int_arith_shift;
  const double mulfac = double(mulfac_int); // integer aritmetic precision scale

  const double Kg = 1. - Kr - Kb;

  if (bits_per_pixel <= 16) {
    const int Srgb = (1 << bits_per_pixel) - 1;  // 255;
    matrix.y_b = (int)(Sy * Kb * mulfac / Srgb + 0.5); //B
    matrix.y_g = (int)(Sy * Kg * mulfac / Srgb + 0.5); //G
    matrix.y_r = (int)(Sy * Kr * mulfac / Srgb + 0.5); //R
    matrix.u_b = (int)(Suv * mulfac / Srgb + 0.5);
    matrix.u_g = (int)(Suv * Kg / (Kb - 1) * mulfac / Srgb + 0.5);
    matrix.u_r = (int)(Suv * Kr / (Kb - 1) * mulfac / Srgb + 0.5);
    matrix.v_b = (int)(Suv * Kb / (Kr - 1) * mulfac / Srgb + 0.5);
    matrix.v_g = (int)(Suv * Kg / (Kr - 1) * mulfac / Srgb + 0.5);
    matrix.v_r = (int)(Suv * mulfac / Srgb + 0.5);

    matrix.offset_y = Oy;

    // FIXME: do we need it to expand for the other u and v constants?
    // anti overflow e.g. for 15 bits the sum must not exceed 32768
    if (matrix.offset_y == 0 && matrix.y_g + matrix.y_r + matrix.y_b != mulfac_int)
      matrix.y_g = mulfac_int - (matrix.y_r + matrix.y_b);

    // special precalculations for direct RGB to YUY2
    double dku = Suv / (Srgb * (1.0 - Kb)) * mulfac;
    double dkv = Suv / (Srgb * (1.0 - Kr)) * mulfac;
    matrix.ku = (int)(dku + 0.5);
    matrix.kv = (int)(dkv + 0.5);
    matrix.ku_luma = -(int)(dku * Srgb / Sy + 0.5);
    matrix.kv_luma = -(int)(dkv * Srgb / Sy + 0.5);
  }

  // for 16 bits, float is used, no unsigned 16 bit arithmetic
  double Srgb_f = bits_per_pixel == 32 ? 1.0 : ((1 << bits_per_pixel) - 1);
  matrix.y_b_f = (float)(Sy_f * Kb / Srgb_f); //B
  matrix.y_g_f = (float)(Sy_f * Kg / Srgb_f); //G
  matrix.y_r_f = (float)(Sy_f * Kr / Srgb_f); //R
  matrix.u_b_f = (float)(Suv_f / Srgb_f);
  matrix.u_g_f = (float)(Suv_f * Kg / (Kb - 1) / Srgb_f);
  matrix.u_r_f = (float)(Suv_f * Kr / (Kb - 1) / Srgb_f);
  matrix.v_b_f = (float)(Suv_f * Kb / (Kr - 1) / Srgb_f);
  matrix.v_g_f = (float)(Suv_f * Kg / (Kr - 1) / Srgb_f);
  matrix.v_r_f = (float)(Suv_f / Srgb_f);
  matrix.offset_y_f = Oy_f;
}

static void BuildMatrix_Yuv2Rgb_core(double Kr, double Kb, int int_arith_shift, bool full_scale, int bits_per_pixel, ConversionMatrix& matrix)
{
  int Sy, Suv, Oy;
  float Sy_f, Suv_f, Oy_f;

  if (bits_per_pixel <= 16) {
    Oy = full_scale ? 0 : (16 << (bits_per_pixel - 8));
    Oy_f = (float)Oy; // for 16 bits

    int ymin = (full_scale ? 0 : 16) << (bits_per_pixel - 8);
    int max_pixel_value = (1 << bits_per_pixel) - 1;
    int ymax = full_scale ? max_pixel_value : (235 << (bits_per_pixel - 8));
    Sy = ymax - ymin;
    Sy_f = (float)Sy;

    int cmin = full_scale ? 0 : (16 << (bits_per_pixel - 8));
    int cmax = full_scale ? max_pixel_value : (240 << (bits_per_pixel - 8));
    Suv = (cmax - cmin) / 2;
    Suv_f = (cmax - cmin) / 2.0f;
  }
  else {
    Oy_f = full_scale ? 0.0f : (16.0f / 255.0f);
    Oy = full_scale ? 0 : 16; // n/a

    Sy_f = full_scale ? c8tof(255) : (c8tof(235) - c8tof(16));
    Suv_f = full_scale ? (0.5f - -0.5f) / 2 : (uv8tof(240) - uv8tof(16)) / 2;
  }


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

  const double mulfac = double(1 << int_arith_shift); // integer aritmetic precision scale

  const double Kg = 1. - Kr - Kb;

  if (bits_per_pixel <= 16) {
    const int Srgb = (1 << bits_per_pixel) - 1;  // 255;
    matrix.y_b = (int)(Srgb * 1.000 * mulfac / Sy + 0.5); //Y
    matrix.u_b = (int)(Srgb * (1 - Kb) * mulfac / Suv + 0.5); //U
    matrix.v_b = (int)(Srgb * 0.000 * mulfac / Suv + 0.5); //V
    matrix.y_g = (int)(Srgb * 1.000 * mulfac / Sy + 0.5);
    matrix.u_g = (int)(Srgb * (Kb - 1) * Kb / Kg * mulfac / Suv + 0.5);
    matrix.v_g = (int)(Srgb * (Kr - 1) * Kr / Kg * mulfac / Suv + 0.5);
    matrix.y_r = (int)(Srgb * 1.000 * mulfac / Sy + 0.5);
    matrix.u_r = (int)(Srgb * 0.000 * mulfac / Suv + 0.5);
    matrix.v_r = (int)(Srgb * (1 - Kr) * mulfac / Suv + 0.5);
    matrix.offset_y = -Oy;
  }

  double Srgb_f = bits_per_pixel == 32 ? 1.0 : ((1 << bits_per_pixel) - 1);
  matrix.y_b_f = (float)(Srgb_f * 1.000 / Sy_f); //Y
  matrix.u_b_f = (float)(Srgb_f * (1 - Kb) / Suv_f); //U
  matrix.v_b_f = (float)(Srgb_f * 0.000 / Suv_f); //V
  matrix.y_g_f = (float)(Srgb_f * 1.000 / Sy_f);
  matrix.u_g_f = (float)(Srgb_f * (Kb - 1) * Kb / Kg / Suv_f);
  matrix.v_g_f = (float)(Srgb_f * (Kr - 1) * Kr / Kg / Suv_f);
  matrix.y_r_f = (float)(Srgb_f * 1.000 / Sy_f);
  matrix.u_r_f = (float)(Srgb_f * 0.000 / Suv_f);
  matrix.v_r_f = (float)(Srgb_f * (1 - Kr) / Suv_f);
  matrix.offset_y_f = -Oy_f;
}

bool do_BuildMatrix_Rgb2Yuv(int _Matrix, int _ColorRange, int int_arith_shift, int bits_per_pixel, ConversionMatrix& matrix)
{
  if (_ColorRange != ColorRange_e::AVS_RANGE_FULL && _ColorRange != ColorRange_e::AVS_RANGE_LIMITED)
    return false;
  const bool is_full = _ColorRange == ColorRange_e::AVS_RANGE_FULL;
  if (_Matrix == Matrix_e::AVS_MATRIX_BT470_BG || _Matrix == Matrix_e::AVS_MATRIX_ST170_M) { // 601
    /*
    Y'= 0.299*R' + 0.587*G' + 0.114*B'
    Cb=-0.169*R' - 0.331*G' + 0.500*B'
    Cr= 0.500*R' - 0.419*G' - 0.081*B'
    */
    BuildMatrix_Rgb2Yuv_core(0.299,  /* 0.587  */ 0.114, int_arith_shift, is_full, bits_per_pixel, matrix);
  }
  else if (_Matrix == Matrix_e::AVS_MATRIX_BT709) {
    /*
    Y'= 0.2126*R' + 0.7152*G' + 0.0722*B'
    Cb=-0.1145*R' - 0.3855*G' + 0.5000*B'
    Cr= 0.5000*R' - 0.4542*G' - 0.0458*B'
    */
    BuildMatrix_Rgb2Yuv_core(0.2126, /* 0.7152 */ 0.0722, int_arith_shift, is_full, bits_per_pixel, matrix);
  }
  else if (_Matrix == Matrix_e::AVS_MATRIX_AVERAGE) { // non-standard!
    BuildMatrix_Rgb2Yuv_core(1.0 / 3, /* 1.0/3 */ 1.0 / 3, int_arith_shift, is_full, bits_per_pixel, matrix);
  }
  else if (_Matrix == Matrix_e::AVS_MATRIX_BT2020_CL || _Matrix == Matrix_e::AVS_MATRIX_BT2020_NCL) {
    BuildMatrix_Rgb2Yuv_core(0.2627, /* 0.6780 */ 0.0593, int_arith_shift, is_full, bits_per_pixel, matrix);
  }
  else if (_Matrix == Matrix_e::AVS_MATRIX_BT470_M) {
    BuildMatrix_Rgb2Yuv_core(0.3, /* 0.59 */ 0.11, int_arith_shift, is_full, bits_per_pixel, matrix);
  }
  else if (_Matrix == Matrix_e::AVS_MATRIX_ST240_M) {
    BuildMatrix_Rgb2Yuv_core(0.212, /* 0.701 */ 0.087, int_arith_shift, is_full, bits_per_pixel, matrix);
  }
  else if (_Matrix == Matrix_e::AVS_MATRIX_RGB) {
    BuildMatrix_Rgb2Yuv_core(0.0, /*  */ 0.0, int_arith_shift, is_full, bits_per_pixel, matrix);
  }
  else if (_Matrix == Matrix_e::AVS_MATRIX_ICTCP) {
    // not supported REC_2100_LMS
    return false;
  }
  else if (_Matrix == Matrix_e::AVS_MATRIX_YCGCO) {
    // not supported
    return false;
  }
  else {
    return false;
  }
  return true;
}

bool do_BuildMatrix_Yuv2Rgb(int _Matrix, int _ColorRange, int int_arith_shift, int bits_per_pixel, ConversionMatrix& matrix)
{
  if (_ColorRange != ColorRange_e::AVS_RANGE_FULL && _ColorRange != ColorRange_e::AVS_RANGE_LIMITED)
    return false;
  const bool is_full = _ColorRange == ColorRange_e::AVS_RANGE_FULL;
  
  if (_Matrix == Matrix_e::AVS_MATRIX_BT470_BG || _Matrix == Matrix_e::AVS_MATRIX_ST170_M) { // 601
    BuildMatrix_Yuv2Rgb_core(0.299,  /* 0.587  */ 0.114, int_arith_shift, is_full, bits_per_pixel, matrix);
  }
  else if (_Matrix == Matrix_e::AVS_MATRIX_BT709) {
    BuildMatrix_Yuv2Rgb_core(0.2126, /* 0.7152 */ 0.0722, int_arith_shift, is_full, bits_per_pixel, matrix);
  }
  else if (_Matrix == Matrix_e::AVS_MATRIX_AVERAGE) { // non-standard!
    BuildMatrix_Yuv2Rgb_core(1.0 / 3, /* 1.0/3 */ 1.0 / 3, int_arith_shift, is_full, bits_per_pixel, matrix);
  }
  else if (_Matrix == Matrix_e::AVS_MATRIX_BT2020_CL || _Matrix == Matrix_e::AVS_MATRIX_BT2020_NCL) {
    BuildMatrix_Yuv2Rgb_core(0.2627, /* 0.6780 */ 0.0593, int_arith_shift, is_full, bits_per_pixel, matrix);
  }
  else if (_Matrix == Matrix_e::AVS_MATRIX_BT470_M) {
    BuildMatrix_Yuv2Rgb_core(0.3, /* 0.59 */ 0.11, int_arith_shift, is_full, bits_per_pixel, matrix);
  }
  else if (_Matrix == Matrix_e::AVS_MATRIX_ST240_M) {
    BuildMatrix_Yuv2Rgb_core(0.212, /* 0.701 */ 0.087, int_arith_shift, is_full, bits_per_pixel, matrix);
  }
  else if (_Matrix == Matrix_e::AVS_MATRIX_RGB) {
    BuildMatrix_Yuv2Rgb_core(0.0, /*  */ 0.0, int_arith_shift, is_full, bits_per_pixel, matrix);
  }
  else if (_Matrix == Matrix_e::AVS_MATRIX_ICTCP) {
    // not supported REC_2100_LMS
    return false;
  }
  else if (_Matrix == Matrix_e::AVS_MATRIX_YCGCO) {
    // not supported
    return false;
  }
  else {
    return false;
  }
  return true;
}
