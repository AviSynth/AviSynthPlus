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

#ifndef __Convert_bits_sse_H__
#define __Convert_bits_sse_H__

#include <avs/types.h>

template<typename pixel_t, bool chroma, bool fulls, bool fulld>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
void convert_32_to_uintN_sse41(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth);

template<typename pixel_t_s, typename pixel_t_d, bool chroma, bool fulls, bool fulld>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
void convert_uint_sse41(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth);

template<typename pixel_t_s, typename pixel_t_d, bool chroma, bool fulls, bool fulld>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
void convert_ordered_dither_uint_sse41(const BYTE* srcp8, BYTE* dstp8, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth);

#endif  // __Convert_bits_sse_H__
