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

#ifndef __Resample_SSE_H__
#define __Resample_SSE_H__

#include <avisynth.h>
#include <avs/config.h>
#include "../resample.h"

void resize_h_prepare_coeff_8or16(ResamplingProgram* p, IScriptEnvironment* env, int alignFilterSize8or16);

#ifdef X86_32
void resize_v_mmx_planar(BYTE* dst, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel, const int* pitch_table, const void* storage);
#endif
void resize_v_sse2_planar(BYTE* dst, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel, const int* pitch_table, const void* storage);

#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
void resize_v_sse41_planar(BYTE* dst, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel, const int* pitch_table, const void* storage);

#if defined(GCC) || defined(CLANG)
__attribute__((__target__("ssse3")))
#endif
void resize_v_ssse3_planar(BYTE* dst, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel, const int* pitch_table, const void* storage);

template<bool lessthan16bit>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("ssse3")))
#endif
void resizer_h_ssse3_generic_uint16_t(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);

template<bool lessthan16bit>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
void resizer_h_sse41_generic_uint16_t(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);

template<int filtersizealigned8, int filtersizemod8>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("ssse3")))
#endif
void resizer_h_ssse3_generic_float(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);

template<bool lessthan16bit>
void resize_v_sse2_planar_uint16_t(BYTE* dst0, const BYTE* src0, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel, const int* pitch_table, const void* storage);

template<bool lessthan16bit>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
void resize_v_sse41_planar_uint16_t(BYTE* dst0, const BYTE* src0, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel, const int* pitch_table, const void* storage);

void resize_v_sse2_planar_float(BYTE* dst0, const BYTE* src0, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel, const int* pitch_table, const void* storage);

#if defined(GCC) || defined(CLANG)
__attribute__((__target__("ssse3")))
#endif
void resizer_h_ssse3_generic(BYTE* dst, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);

#if defined(GCC) || defined(CLANG)
__attribute__((__target__("ssse3")))
#endif
void resizer_h_ssse3_8(BYTE* dst, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);


#endif // __Resample_SSE_H__
