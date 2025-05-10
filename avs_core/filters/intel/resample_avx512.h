// Avisynth v2.5.  Copyright 2002 Ben Rudiak-Gould et al.
// http://avisynth.nl

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

#ifndef __Resample_AVX512_H__
#define __Resample_AVX512_H__

#include <avisynth.h>
#include "../resample_functions.h"

void resize_h_planar_float_avx512_transpose_vstripe_ks4(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
void resize_h_planar_float_avx512_permutex_vstripe_ks4(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
void resize_h_planar_float_avx512_permutex_vstripe_ks8(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
void resize_h_planar_float_avx512_permutex_vstripe_ks16(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);

void resize_v_avx512_planar_float(BYTE* dst0, const BYTE* src0, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel);


#define _MM_TRANSPOSE16_LANE4_PS(row0, row1, row2, row3) \
	do { \
		__m512 __t0, __t1, __t2, __t3; \
		__t0 = _mm512_unpacklo_ps(row0, row1); \
		__t1 = _mm512_unpackhi_ps(row0, row1); \
		__t2 = _mm512_unpacklo_ps(row2, row3); \
		__t3 = _mm512_unpackhi_ps(row2, row3); \
		row0 = _mm512_shuffle_ps(__t0, __t2, _MM_SHUFFLE(1, 0, 1, 0)); \
		row1 = _mm512_shuffle_ps(__t0, __t2, _MM_SHUFFLE(3, 2, 3, 2)); \
		row2 = _mm512_shuffle_ps(__t1, __t3, _MM_SHUFFLE(1, 0, 1, 0)); \
		row3 = _mm512_shuffle_ps(__t1, __t3, _MM_SHUFFLE(3, 2, 3, 2)); \
	} while (0)

#ifndef _mm512_loadu_4_m128
#define _mm512_loadu_4_m128(/* __m128 const* */ addr1, \
                            /* __m128 const* */ addr2, \
                            /* __m128 const* */ addr3, \
                            /* __m128 const* */ addr4) \
_mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(_mm512_castps128_ps512(_mm_loadu_ps(addr1)), _mm_loadu_ps(addr2), 1), _mm_loadu_ps(addr3), 2), _mm_loadu_ps(addr4), 3)
#endif

#ifndef _mm512_load_4_m128
#define _mm512_load_4_m128(/* __m128 const* */ addr1, \
                            /* __m128 const* */ addr2, \
                            /* __m128 const* */ addr3, \
                            /* __m128 const* */ addr4) \
_mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(_mm512_castps128_ps512(_mm_load_ps(addr1)), _mm_load_ps(addr2), 1), _mm_load_ps(addr3), 2), _mm_load_ps(addr4), 3)
#endif

#endif // __Resample_AVX512_H__
