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

#ifndef __Convert_RGB_sse_H__
#define __Convert_RGB_sse_H__

#include <avs/types.h>

#if defined(GCC) || defined(CLANG)
__attribute__((__target__("ssse3")))
#endif
void convert_rgb48_to_rgb64_ssse3(const BYTE *srcp, BYTE *dstp, size_t src_pitch, size_t dst_pitch, size_t width, size_t height);

#if defined(GCC) || defined(CLANG)
__attribute__((__target__("ssse3")))
#endif
void convert_rgb24_to_rgb32_ssse3(const BYTE *srcp, BYTE *dstp, size_t src_pitch, size_t dst_pitch, size_t width, size_t height);

#ifdef X86_32
void convert_rgb24_to_rgb32_mmx(const BYTE *srcp, BYTE *dstp, size_t src_pitch, size_t dst_pitch, size_t width, size_t height);
#endif

#if defined(GCC) || defined(CLANG)
__attribute__((__target__("ssse3")))
#endif
void convert_rgb64_to_rgb48_ssse3(const BYTE *srcp, BYTE *dstp, size_t src_pitch, size_t dst_pitch, size_t width, size_t height);

#if defined(GCC) || defined(CLANG)
__attribute__((__target__("ssse3")))
#endif
void convert_rgb32_to_rgb24_ssse3(const BYTE *srcp, BYTE *dstp, size_t src_pitch, size_t dst_pitch, size_t width, size_t height);

#ifdef X86_32
void convert_rgb32_to_rgb24_mmx(const BYTE *srcp, BYTE *dstp, size_t src_pitch, size_t dst_pitch, size_t width, size_t height);
#endif

template<typename pixel_t, bool targetHasAlpha>
void convert_rgb_to_rgbp_ssse3(const BYTE *srcp, BYTE * (&dstp)[4], int src_pitch, int(&dst_pitch)[4], int width, int height, int bits_per_pixel);

template<typename pixel_t, bool targetHasAlpha>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("ssse3")))
#endif
void convert_rgba_to_rgbp_ssse3(const BYTE *srcp, BYTE * (&dstp)[4], int src_pitch, int (&dst_pitch)[4], int width, int height);

template<typename pixel_t, bool hasSrcAlpha>
void convert_rgbp_to_rgba_sse2(const BYTE *(&srcp)[4], BYTE * dstp, int (&src_pitch)[4], int dst_pitch, int width, int height);

#endif  // __Convert_RGB_sse_H__
