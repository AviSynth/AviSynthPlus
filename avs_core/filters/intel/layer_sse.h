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

#ifndef __Layer_SSE_H__
#define __Layer_SSE_H__

#include <avisynth.h>
#include <stdint.h>

void mask_sse2(BYTE* srcp, const BYTE* alphap, int src_pitch, int alpha_pitch, size_t width, size_t height);
void colorkeymask_sse2(BYTE* pf, int pitch, int color, int height, int width, int tolB, int tolG, int tolR);
void invert_frame_sse2(BYTE* frame, int pitch, int width, int height, int mask);
void invert_frame_uint16_sse2(BYTE* frame, int pitch, int width, int height, uint64_t mask64);
template<bool use_chroma>
void layer_yuy2_mul_sse2(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level);
template<bool use_chroma>
void layer_yuy2_add_sse2(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level);
void layer_yuy2_fast_sse2(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level);
template<typename pixel_t>
void layer_genericplane_fast_sse2(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level);
template<bool use_chroma>
void layer_yuy2_subtract_sse2(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level);
template<int mode>
void layer_yuy2_lighten_darken_sse2(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level, int thresh);
template<bool use_chroma>
void layer_rgb32_mul_sse2(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level);
template<bool use_chroma>
void layer_rgb32_add_sse2(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level);
void layer_rgb32_fast_sse2(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level);
template<bool use_chroma>
void layer_rgb32_subtract_sse2(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level);
template<int mode>
void layer_rgb32_lighten_darken_sse2(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level, int thresh);

#ifdef X86_32 
void mask_mmx(BYTE* srcp, const BYTE* alphap, int src_pitch, int alpha_pitch, size_t width, size_t height);
void colorkeymask_mmx(BYTE* srcp, int pitch, int color, int height, int width, int tolB, int tolG, int tolR);
void invert_frame_mmx(BYTE* frame, int pitch, int width, int height, int mask);
void invert_plane_mmx(BYTE* frame, int pitch, int width, int height);
template<bool use_chroma>
void layer_yuy2_mul_mmx(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level);
template<bool use_chroma>
void layer_yuy2_add_mmx(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level);
template<bool use_chroma>
void layer_yuy2_subtract_mmx(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level);

void layer_yuy2_fast_isse(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level);
template<int mode>
void layer_yuy2_lighten_darken_isse(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level, int thresh);
template<bool use_chroma>
void layer_rgb32_mul_isse(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level);
template<bool use_chroma>
void layer_rgb32_add_isse(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level);
void layer_rgb32_fast_isse(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level);
template<bool use_chroma>
void layer_rgb32_subtract_isse(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level);
template<int mode>
void layer_rgb32_lighten_darken_isse(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level, int thresh);

#endif

#endif  // __Layer_SSE_H__
