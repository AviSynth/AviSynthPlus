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

#ifndef __Convert_matrix_H__
#define __Convert_matrix_H__

#include "../core/internal.h"

struct ConversionMatrix {
  int y_r, y_g, y_b;
  // for grayscale conversion these may not needed
  int u_r, u_g, u_b;
  int v_r, v_g, v_b;

  // used in YUY2 RGB->YUY2 asm
  int ku, ku_luma;
  int kv, kv_luma;

  float y_r_f, y_g_f, y_b_f;
  float u_r_f, u_g_f, u_b_f;
  float v_r_f, v_g_f, v_b_f;

  int offset_y;
  float offset_y_f;
};

bool do_BuildMatrix_Rgb2Yuv(int _Matrix, int _ColorRange, int int_arith_shift, int bits_per_pixel, ConversionMatrix& matrix);
bool do_BuildMatrix_Yuv2Rgb(int _Matrix, int _ColorRange, int int_arith_shift, int bits_per_pixel, ConversionMatrix& matrix);

/*****************************************************
 *******   Colorspace Single-Byte Conversions   ******
 ****************************************************/

// useful to other filters
inline int RGB2YUV_Rec601(int rgb) // limited range
{
  const int cyb = int(0.114*219/255*65536+0.5);
  const int cyg = int(0.587*219/255*65536+0.5);
  const int cyr = int(0.299*219/255*65536+0.5);

  // y can't overflow
  int y = (cyb*(rgb&255) + cyg*((rgb>>8)&255) + cyr*((rgb>>16)&255) + 0x108000) >> 16;
  int scaled_y = (y - 16) * int(255.0/219.0*65536+0.5);
  int b_y = ((rgb&255) << 16) - scaled_y;
  int u = ScaledPixelClip((b_y >> 10) * int(1/2.018*1024+0.5) + 0x800000);
  int r_y = (rgb & 0xFF0000) - scaled_y;
  int v = ScaledPixelClip((r_y >> 10) * int(1/1.596*1024+0.5) + 0x800000);
  return ((y*256+u)*256+v) | (rgb & 0xff000000);
}

#endif  // __Convert_matrix_H__
