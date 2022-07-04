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

#ifndef _AVS_TURN_SSE_H
#define _AVS_TURN_SSE_H

#include <avisynth.h>
#include "../turn.h"

// Other filters (e.g. resampler) might also use these functions
void turn_left_plane_8_sse2(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch);
void turn_left_plane_16_sse2(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch);
void turn_left_plane_32_sse2(const BYTE* srcp, BYTE* dstp, int src_rowsize, int srcHeight, int src_pitch, int dst_pitch);
void turn_left_rgb32_sse2(const BYTE *srcp, BYTE *dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch);
void turn_left_rgb64_sse2(const BYTE *srcp, BYTE *dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch);
void turn_right_plane_8_sse2(const BYTE* srcp, BYTE* dstp, int src_rowsize, int srcHeight, int src_pitch, int dst_pitch);
void turn_right_plane_16_sse2(const BYTE* srcp, BYTE* dstp, int src_rowsize, int srcHeight, int src_pitch, int dst_pitch);
void turn_right_plane_32_sse2(const BYTE* srcp, BYTE* dstp, int src_rowsize, int srcHeight, int src_pitch, int dst_pitch);
void turn_right_rgb32_sse2(const BYTE *srcp, BYTE *dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch);
void turn_right_rgb64_sse2(const BYTE *srcp, BYTE *dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch);

template <typename T>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("ssse3")))
#endif
void turn_180_plane_ssse3(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch);

template <typename T>
void turn_180_plane_sse2(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch);

#endif  // _AVS_TURN_SSE_H
