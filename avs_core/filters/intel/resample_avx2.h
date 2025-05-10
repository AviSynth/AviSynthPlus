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

#ifndef __Resample_AVX2_H__
#define __Resample_AVX2_H__

#include <avisynth.h>
#include "../resample_functions.h"

void resizer_h_avx2_generic_uint8_t(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);

template<bool lessthan16bit>
void resizer_h_avx2_generic_uint16_t(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);

void resizer_h_avx2_generic_float(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);

void resize_v_avx2_planar_uint8_t(BYTE* dst, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel);

template<bool lessthan16bit>
void resize_v_avx2_planar_uint16_t(BYTE* dst0, const BYTE* src0, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel);

void resize_v_avx2_planar_float(BYTE* dst0, const BYTE* src0, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel);

void resize_h_planar_float_avx_transpose(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);

void resize_h_planar_float_avx_transpose_vstripe_ks4(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);

void resize_h_planar_float_avx_gather_vstripe_ks8(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);

void resize_h_planar_float_avx2_permutex_vstripe_ks4(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
void resize_h_planar_float_avx2_permutex_vstripe_ks8(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);

// Transpose 4x4 blocks within each lane
#define _MM_TRANSPOSE8_LANE4_PS(row0, row1, row2, row3) \
	do { \
		__m256 __t0, __t1, __t2, __t3; \
		__t0 = _mm256_unpacklo_ps(row0, row1); \
		__t1 = _mm256_unpackhi_ps(row0, row1); \
		__t2 = _mm256_unpacklo_ps(row2, row3); \
		__t3 = _mm256_unpackhi_ps(row2, row3); \
		row0 = _mm256_shuffle_ps(__t0, __t2, _MM_SHUFFLE(1, 0, 1, 0)); \
		row1 = _mm256_shuffle_ps(__t0, __t2, _MM_SHUFFLE(3, 2, 3, 2)); \
		row2 = _mm256_shuffle_ps(__t1, __t3, _MM_SHUFFLE(1, 0, 1, 0)); \
		row3 = _mm256_shuffle_ps(__t1, __t3, _MM_SHUFFLE(3, 2, 3, 2)); \
	} while (0)

#define _MM_TRANSPOSE8_PS(row0, row1, row2, row3, row4, row5, row6, row7) \
	do { \
		__m256 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7; \
		__m256 __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7; \
		__t0 = _mm256_unpacklo_ps(row0, row1); \
		__t1 = _mm256_unpackhi_ps(row0, row1); \
		__t2 = _mm256_unpacklo_ps(row2, row3); \
		__t3 = _mm256_unpackhi_ps(row2, row3); \
		__t4 = _mm256_unpacklo_ps(row4, row5); \
		__t5 = _mm256_unpackhi_ps(row4, row5); \
		__t6 = _mm256_unpacklo_ps(row6, row7); \
		__t7 = _mm256_unpackhi_ps(row6, row7); \
		__tt0 = _mm256_shuffle_ps(__t0, __t2, _MM_SHUFFLE(1, 0, 1, 0)); \
		__tt1 = _mm256_shuffle_ps(__t0, __t2, _MM_SHUFFLE(3, 2, 3, 2)); \
		__tt2 = _mm256_shuffle_ps(__t1, __t3, _MM_SHUFFLE(1, 0, 1, 0)); \
		__tt3 = _mm256_shuffle_ps(__t1, __t3, _MM_SHUFFLE(3, 2, 3, 2)); \
		__tt4 = _mm256_shuffle_ps(__t4, __t6, _MM_SHUFFLE(1, 0, 1, 0)); \
		__tt5 = _mm256_shuffle_ps(__t4, __t6, _MM_SHUFFLE(3, 2, 3, 2)); \
		__tt6 = _mm256_shuffle_ps(__t5, __t7, _MM_SHUFFLE(1, 0, 1, 0)); \
		__tt7 = _mm256_shuffle_ps(__t5, __t7, _MM_SHUFFLE(3, 2, 3, 2)); \
		row0 = _mm256_permute2f128_ps(__tt0, __tt4, 0x20); \
		row1 = _mm256_permute2f128_ps(__tt1, __tt5, 0x20); \
		row2 = _mm256_permute2f128_ps(__tt2, __tt6, 0x20); \
		row3 = _mm256_permute2f128_ps(__tt3, __tt7, 0x20); \
		row4 = _mm256_permute2f128_ps(__tt0, __tt4, 0x31); \
		row5 = _mm256_permute2f128_ps(__tt1, __tt5, 0x31); \
		row6 = _mm256_permute2f128_ps(__tt2, __tt6, 0x31); \
		row7 = _mm256_permute2f128_ps(__tt3, __tt7, 0x31); \
	} while (0)



#ifndef _mm256_loadu_2_m128
#define _mm256_loadu_2_m128(/* __m128 const* */ loaddr, \
                            /* __m128 const* */ hiaddr) \
    _mm256_set_m128(_mm_loadu_ps(hiaddr), _mm_loadu_ps(loaddr))
#endif

#ifndef _mm256_load_2_m128
#define _mm256_load_2_m128(/* __m128 const* */ loaddr, \
                            /* __m128 const* */ hiaddr) \
    _mm256_set_m128(_mm_load_ps(hiaddr), _mm_load_ps(loaddr))
#endif


#endif // __Resample_AVX2_H__
