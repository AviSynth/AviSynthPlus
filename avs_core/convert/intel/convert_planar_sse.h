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

#ifndef __Convert_PLANAR_sse_H__
#define __Convert_PLANAR_sse_H__

#include <avs/types.h>
#include "../convert_matrix.h"

void convert_yuy2_to_y8_sse2(const BYTE *srcp, BYTE *dstp, size_t src_pitch, size_t dst_pitch, size_t width, size_t height);
#ifdef X86_32
void convert_yuy2_to_y8_mmx(const BYTE *srcp, BYTE *dstp, size_t src_pitch, size_t dst_pitch, size_t width, size_t height);
#endif

void convert_rgb32_to_y8_sse2(const BYTE *srcp, BYTE *dstp, size_t src_pitch, size_t dst_pitch, size_t width, size_t height, const ConversionMatrix &matrix);
void convert_rgb24_to_y8_sse2(const BYTE *srcp, BYTE *dstp, size_t src_pitch, size_t dst_pitch, size_t width, size_t height, const ConversionMatrix &matrix);
#ifdef X86_32
void convert_rgb32_to_y8_mmx(const BYTE *srcp, BYTE *dstp, size_t src_pitch, size_t dst_pitch, size_t width, size_t height, const ConversionMatrix &matrix);
void convert_rgb24_to_y8_mmx(const BYTE *srcp, BYTE *dstp, size_t src_pitch, size_t dst_pitch, size_t width, size_t height, const ConversionMatrix &matrix);
#endif

void convert_rgb32_to_yv24_sse2(BYTE* dstY, BYTE* dstU, BYTE* dstV, const BYTE*srcp, size_t dst_pitch_y, size_t UVpitch, size_t src_pitch, size_t width, size_t height, const ConversionMatrix &matrix);
void convert_rgb24_to_yv24_sse2(BYTE* dstY, BYTE* dstU, BYTE* dstV, const BYTE*srcp, size_t dst_pitch_y, size_t UVpitch, size_t src_pitch, size_t width, size_t height, const ConversionMatrix &matrix);

#ifdef X86_32
void convert_rgb32_to_yv24_mmx(BYTE* dstY, BYTE* dstU, BYTE* dstV, const BYTE*srcp, size_t dst_pitch_y, size_t UVpitch, size_t src_pitch, size_t width, size_t height, const ConversionMatrix& matrix);
void convert_rgb24_to_yv24_mmx(BYTE* dstY, BYTE* dstU, BYTE* dstV, const BYTE*srcp, size_t dst_pitch_y, size_t UVpitch, size_t src_pitch, size_t width, size_t height, const ConversionMatrix &matrix);
#endif

template<typename pixel_t, int bits_per_pixel>
void convert_planarrgb_to_yuv_uint8_14_sse2(BYTE *(&dstp)[3], int (&dstPitch)[3], const BYTE *(&srcp)[3], const int (&srcPitch)[3], int width, int height, const ConversionMatrix &m);

void convert_planarrgb_to_yuv_float_sse2(BYTE *(&dstp)[3], int(&dstPitch)[3], const BYTE *(&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix &m);

template<int bits_per_pixel>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
void convert_planarrgb_to_yuv_uint16_sse41(BYTE *(&dstp)[3], int (&dstPitch)[3], const BYTE *(&srcp)[3], const int (&srcPitch)[3], int width, int height, const ConversionMatrix &m);

template<int bits_per_pixel>
void convert_planarrgb_to_yuv_uint16_sse2(BYTE *(&dstp)[3], int (&dstPitch)[3], const BYTE *(&srcp)[3], const int (&srcPitch)[3], int width, int height, const ConversionMatrix &m);

template<typename pixel_t, int bits_per_pixel>
void convert_yuv_to_planarrgb_uint8_14_sse2(BYTE *(&dstp)[3], int (&dstPitch)[3], const BYTE *(&srcp)[3], const int (&srcPitch)[3], int width, int height, const ConversionMatrix &m);

void convert_yuv_to_planarrgb_float_sse2(BYTE *(&dstp)[3], int(&dstPitch)[3], const BYTE *(&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix &m);

template<int bits_per_pixel>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
void convert_yuv_to_planarrgb_uint16_sse41(BYTE *(&dstp)[3], int (&dstPitch)[3], const BYTE *(&srcp)[3], const int (&srcPitch)[3], int width, int height, const ConversionMatrix &m);

template<int bits_per_pixel>
void convert_yuv_to_planarrgb_uint16_sse2(BYTE *(&dstp)[3], int(&dstPitch)[3], const BYTE *(&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix &m);

template<int rgb_pixel_step, bool hasAlpha>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("ssse3")))
#endif
void convert_yv24_to_rgb_ssse3(BYTE* dstp, const BYTE* srcY, const BYTE* srcU, const BYTE*srcV, const BYTE*srcA, size_t dst_pitch, size_t src_pitch_y, size_t src_pitch_uv, size_t src_pitch_a, size_t width, size_t height, const ConversionMatrix &matrix);

template<int rgb_pixel_step, bool hasAlpha>
void convert_yv24_to_rgb_sse2(BYTE* dstp, const BYTE* srcY, const BYTE* srcU, const BYTE*srcV, const BYTE*srcA, size_t dst_pitch, size_t src_pitch_y, size_t src_pitch_uv, size_t src_pitch_a, size_t width, size_t height, const ConversionMatrix &matrix);

#ifdef X86_32
template<int rgb_pixel_step>
void convert_yv24_to_rgb_mmx(BYTE* dstp, const BYTE* srcY, const BYTE* srcU, const BYTE*srcV, size_t dst_pitch, size_t src_pitch_y, size_t src_pitch_uv, size_t width, size_t height, const ConversionMatrix &matrix);
#endif

void convert_yuy2_to_yv16_sse2(const BYTE *srcp, BYTE *dstp_y, BYTE *dstp_u, BYTE *dstp_v, size_t src_pitch, size_t dst_pitch_y, size_t dst_pitch_uv, size_t width, size_t height);
#ifdef X86_32
void convert_yuy2_to_yv16_mmx(const BYTE *srcp, BYTE *dstp_y, BYTE *dstp_u, BYTE *dstp_v, size_t src_pitch, size_t dst_pitch_y, size_t dst_pitch_uv, size_t width, size_t height);
#endif

void convert_yv16_to_yuy2_sse2(const BYTE *srcp_y, const BYTE *srcp_u, const BYTE *srcp_v, BYTE *dstp, size_t src_pitch_y, size_t src_pitch_uv, size_t dst_pitch, size_t width, size_t height);
#ifdef X86_32
void convert_yv16_to_yuy2_mmx(const BYTE *srcp_y, const BYTE *srcp_u, const BYTE *srcp_v, BYTE *dstp, size_t src_pitch_y, size_t src_pitch_uv, size_t dst_pitch, size_t width, size_t height);
#endif

#endif // __Convert_PLANAR_sse_H__
