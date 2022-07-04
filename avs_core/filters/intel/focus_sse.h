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

#ifndef __Focus_SSE_H__
#define __Focus_SSE_H__

#include <avisynth.h>

#ifdef X86_32
void af_vertical_mmx(BYTE* line_buf, BYTE* dstp, int height, int pitch, int width, int amount);
#endif
void af_vertical_sse2(BYTE* line_buf, BYTE* dstp, int height, int pitch, int width, int amount);
void af_vertical_uint16_t_sse2(BYTE* line_buf, BYTE* dstp, int height, int pitch, int row_size, int amount);
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
void af_vertical_uint16_t_sse41(BYTE* line_buf, BYTE* dstp, int height, int pitch, int row_size, int amount);
void af_vertical_sse2_float(BYTE * line_buf, BYTE * dstp, const int height, const int pitch, const int row_size, const float amount);


#ifdef X86_32
void af_horizontal_planar_mmx(BYTE* dstp, size_t height, size_t pitch, size_t width, size_t amount);
#endif
void af_horizontal_planar_sse2(BYTE* dstp, size_t height, size_t pitch, size_t width, size_t amount);
void af_horizontal_planar_uint16_t_sse2(BYTE* dstp, size_t height, size_t pitch, size_t row_size, size_t amount, int bits_per_pixel);
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
void af_horizontal_planar_uint16_t_sse41(BYTE* dstp, size_t height, size_t pitch, size_t row_size, size_t amount, int bits_per_pixel);
void af_horizontal_planar_float_sse2(BYTE* dstp, size_t height, size_t pitch, size_t row_size, float amount);


#ifdef X86_32
void af_horizontal_yuy2_mmx(BYTE* dstp, const BYTE* srcp, size_t dst_pitch, size_t src_pitch, size_t height, size_t width, size_t amount);
#endif
void af_horizontal_yuy2_sse2(BYTE* dstp, const BYTE* srcp, size_t dst_pitch, size_t src_pitch, size_t height, size_t width, size_t amount);

#ifdef X86_32
void af_horizontal_rgb32_mmx(BYTE* dstp, const BYTE* srcp, size_t dst_pitch, size_t src_pitch, size_t height, size_t width, size_t amount);
#endif
void af_horizontal_rgb32_sse2(BYTE* dstp, const BYTE* srcp, size_t dst_pitch, size_t src_pitch, size_t height, size_t width, size_t amount);

void af_horizontal_rgb64_sse2(BYTE* dstp, const BYTE* srcp, size_t dst_pitch, size_t src_pitch, size_t height, size_t width, size_t amount);
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
void af_horizontal_rgb64_sse41(BYTE* dstp, const BYTE* srcp, size_t dst_pitch, size_t src_pitch, size_t height, size_t width, size_t amount);



#ifdef X86_32
void accumulate_line_mmx(BYTE* c_plane, const BYTE** planeP, int planes, size_t width, int threshold, int div);
#endif
template<bool maxThreshold>
void accumulate_line_sse2(BYTE* c_plane, const BYTE** planeP, int planes, size_t width, int threshold, int div);
template<bool maxThreshold>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("ssse3")))
#endif
void accumulate_line_ssse3(BYTE* c_plane, const BYTE** planeP, int planes, size_t width, int threshold, int div);
template<bool maxThreshold, bool lessThan16bit>
void accumulate_line_16_sse2(BYTE* c_plane, const BYTE** planeP, int planes, size_t rowsize, int threshold, int div, int bits_per_pixel);
template<bool maxThreshold, bool lessThan16bit>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
void accumulate_line_16_sse41(BYTE* c_plane, const BYTE** planeP, int planes, size_t rowsize, int threshold, int div, int bits_per_pixel);

#ifdef X86_32
int calculate_sad_isse(const BYTE* cur_ptr, const BYTE* other_ptr, int cur_pitch, int other_pitch, size_t rowsize, size_t height);
#endif
template<bool packedRGB3264>
int calculate_sad_sse2(const BYTE* cur_ptr, const BYTE* other_ptr, int cur_pitch, int other_pitch, size_t rowsize, size_t height);
template<typename pixel_t, bool packedRGB3264>
int64_t calculate_sad_8_or_16_sse2(const BYTE* cur_ptr, const BYTE* other_ptr, int cur_pitch, int other_pitch, size_t rowsize, size_t height);

#endif  // __Focus_SSE_H__
