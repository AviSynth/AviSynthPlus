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

#ifndef __Convert_YUY2_sse_H__
#define __Convert_YUY2_sse_H__

#include <avisynth.h>
#include "../convert_matrix.h"

template<int rgb_bytes>
void convert_rgb_to_yuy2_sse2(const BYTE *src, BYTE *dst, int src_pitch, int dst_pitch, int width, int height, const ConversionMatrix &matrix);

#ifdef X86_32
template<int rgb_bytes>
void convert_rgb_to_yuy2_mmx(const BYTE *src, BYTE *dst, int src_pitch, int dst_pitch, int width, int height, const ConversionMatrix& matrix);
#endif

void convert_yv24_back_to_yuy2_sse2(const BYTE* srcY, const BYTE* srcU, const BYTE* srcV, BYTE* dstp, int pitchY, int pitchUV, int dpitch, int height, int width);

#ifdef X86_32
void convert_yv24_back_to_yuy2_mmx(const BYTE* srcY, const BYTE* srcU, const BYTE* srcV, BYTE* dstp, int pitchY, int pitchUV, int dpitch, int height, int width);
#endif

template<int rgb_bytes>
void convert_rgb_back_to_yuy2_sse2(const BYTE *src, BYTE *dst, int src_pitch, int dst_pitch, int width, int height, ConversionMatrix& matrix);

#ifdef X86_32
template<int rgb_bytes>
void convert_rgb_back_to_yuy2_mmx(const BYTE *src, BYTE *dst, int src_pitch, int dst_pitch, int width, int height, const ConversionMatrix &matrix);
#endif

// ex convert_yv12
void convert_yv12_to_yuy2_interlaced_isse(const BYTE* srcY, const BYTE* srcU, const BYTE* srcV, int src_width, int src_pitch_y, int src_pitch_uv, BYTE *dstp, int dst_pitch, int height);
void convert_yv12_to_yuy2_interlaced_sse2(const BYTE* srcY, const BYTE* srcU, const BYTE* srcV, int src_width, int src_pitch_y, int src_pitch_uv, BYTE *dstp, int dst_pitch, int height);
void convert_yv12_to_yuy2_progressive_isse(const BYTE* srcY, const BYTE* srcU, const BYTE* srcV, int src_width, int src_pitch_y, int src_pitch_uv, BYTE *dstp, int dst_pitch, int height);
void convert_yv12_to_yuy2_progressive_sse2(const BYTE* srcY, const BYTE* srcU, const BYTE* srcV, int src_width, int src_pitch_y, int src_pitch_uv, BYTE *dstp, int dst_pitch, int height);
void convert_yuy2_to_yv12_progressive_isse(const BYTE* src, int src_width, int src_pitch, BYTE* dstY, BYTE* dstU, BYTE* dstV, int dst_pitchY, int dst_pitchUV, int height);
void convert_yuy2_to_yv12_progressive_sse2(const BYTE* src, int src_width, int src_pitch, BYTE* dstY, BYTE* dstU, BYTE* dstV, int dst_pitchY, int dst_pitchUV, int height);
void convert_yuy2_to_yv12_interlaced_isse(const BYTE* src, int src_width, int src_pitch, BYTE* dstY, BYTE* dstU, BYTE* dstV, int dst_pitchY, int dst_pitchUV, int height);
void convert_yuy2_to_yv12_interlaced_sse2(const BYTE* src, int src_width, int src_pitch, BYTE* dstY, BYTE* dstU, BYTE* dstV, int dst_pitchY, int dst_pitchUV, int height); 

#endif // __Convert_YUY2_sse_H__
