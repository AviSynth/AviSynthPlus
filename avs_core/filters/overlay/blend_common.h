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

// Overlay (c) 2003, 2004 by Klaus Post

#ifndef __blend_common_h
#define __blend_common_h

#include <avs/types.h>
#include <avs/config.h>

using overlay_blend_plane_masked_opacity_t = void(BYTE* p1, const BYTE* p2, const BYTE* mask,
  const int p1_pitch, const int p2_pitch, const int mask_pitch,
  const int width, const int height, const int opacity, const float opacity_f);

/*******************************
 ********* Masked Blend ********
 *******************************/

AVS_FORCEINLINE static BYTE overlay_blend_c_core_8(const BYTE p1, const BYTE p2, const int mask) {
  if (mask == 0)
    return p1;
  if (mask == 0xFF)
    return p2;
  //  p1*(1-mask_f) + p2*mask_f -> p1 + (p2-p1)*mask_f
  return (BYTE)(((p1 << 8) + (p2 - p1)*mask + 128) >> 8);
}

template<int bits_per_pixel>
AVS_FORCEINLINE static uint16_t overlay_blend_c_core_16(const uint16_t p1, const uint16_t p2, const int mask) {
  if (mask == 0)
    return p1;
  if (mask >= (1 << bits_per_pixel) -1)
    return p2;
  //  p1*(1-mask_f) + p2*mask_f -> p1 + (p2-p1)*mask_f
  const int half_rounder = 1 << (bits_per_pixel - 1);
  if constexpr(bits_per_pixel == 16) // int32 intermediate overflows
    return (uint16_t)((((int64_t)(p1) << bits_per_pixel) + (p2 - p1)*(int64_t)mask + half_rounder) >> bits_per_pixel);
  else // signed int arithmetic is enough
    return (uint16_t)(((p1 << bits_per_pixel) + (p2 - p1)*mask + half_rounder) >> bits_per_pixel);
}

AVS_FORCEINLINE static float overlay_blend_c_core_simple(const int p1, const int p2, const float factor) {
  //  p1*(1-mask_f) + p2*mask_f -> p1 + (p2-p1)*mask_f
  const float res = p1 + (p2 - p1) * factor;
  return res;
}

AVS_FORCEINLINE static float overlay_blend_c_core_f(const float p1, const float p2, const float mask) {
  return p1 + (p2-p1)*mask; // p1*(1-mask) + p2*mask
}

/*******************************************
 ********* Merge Two Masks Function ********
 *******************************************/
template<typename pixel_t, typename intermediate_result_t, int bits_per_pixel>
AVS_FORCEINLINE static pixel_t overlay_merge_mask_c(const pixel_t p1, const pixel_t p2) {
  return ((intermediate_result_t)p1*p2) >> bits_per_pixel;
}

/********************************
 ********* Blend Opaque *********
 ** Use for Lighten and Darken **
 ********************************/
template<typename pixel_t>
AVS_FORCEINLINE pixel_t overlay_blend_opaque_c_core(const pixel_t p1, const pixel_t p2, const pixel_t mask) {
  return (mask) ? p2 : p1;
}  

// Mode: Overlay
void overlay_blend_c_plane_masked_f(BYTE *p1, const BYTE *p2, const BYTE *mask,
  const int p1_pitch, const int p2_pitch, const int mask_pitch,
  const int width, const int height, const int opacity, const float opacity_f);

template<typename pixel_t, int bits_per_pixel>
void overlay_blend_c_plane_masked(BYTE *p1, const BYTE *p2, const BYTE *mask,
                                  const int p1_pitch, const int p2_pitch, const int mask_pitch,
                                  const int width, const int height, const int opacity, const float opacity_f);

template<bool has_mask, typename pixel_t, int bits_per_pixel>
void overlay_blend_c_uint(BYTE* p1, const BYTE* p2, const BYTE* mask,
  const int p1_pitch, const int p2_pitch, const int mask_pitch,
  const int width, const int height, const int opacity, const float opacity_f);

template<bool has_mask>
void overlay_blend_c_float(BYTE* p1, const BYTE* p2, const BYTE* mask,
  const int p1_pitch, const int p2_pitch, const int mask_pitch,
  const int width, const int height, const int opacity, const float opacity_f);

template<typename pixel_t, int bits_per_pixel>
void overlay_blend_c_plane_opacity(BYTE *p1, const BYTE *p2, const BYTE* mask,
                                   const int p1_pitch, const int p2_pitch, const int mask_pitch,
                                   const int width, const int height, const int opacity, const float opacity_f);
void overlay_blend_c_plane_opacity_f(BYTE *p1, const BYTE *p2, const BYTE* mask,
  const int p1_pitch, const int p2_pitch, const int mask_pitch,
  const int width, const int height, const int opacity, const float opacity_f);

template<typename pixel_t, int bits_per_pixel>
void overlay_blend_c_plane_masked_opacity(BYTE *p1, const BYTE *p2, const BYTE *mask,
                                  const int p1_pitch, const int p2_pitch, const int mask_pitch,
                                  const int width, const int height, const int opacity, const float opacity_f);
void overlay_blend_c_plane_masked_opacity_f(BYTE *p1, const BYTE *p2, const BYTE *mask,
  const int p1_pitch, const int p2_pitch, const int mask_pitch,
  const int width, const int height, const int opacity, const float opacity_f);

// Mode: Darken/lighten
template<typename pixel_t>
void overlay_darken_c(BYTE *p1Y, BYTE *p1U, BYTE *p1V, const BYTE *p2Y, const BYTE *p2U, const BYTE *p2V, int p1_pitch, int p2_pitch, int width, int height);
template<typename pixel_t>
void overlay_lighten_c(BYTE *p1Y, BYTE *p1U, BYTE *p1V, const BYTE *p2Y, const BYTE *p2U, const BYTE *p2V, int p1_pitch, int p2_pitch, int width, int height);

#endif // __blend_common_h
