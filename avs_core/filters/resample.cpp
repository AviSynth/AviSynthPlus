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

#include "resample.h"
#ifdef INTEL_INTRINSICS
#include "intel/resample_sse.h"
#include "intel/resample_avx2.h"
#ifdef INTEL_INTRINSICS_AVX512
#include "intel/resample_avx512.h"
#endif
#include "intel/turn_sse.h"
#endif
#include <avs/config.h>

#include "transform.h"
#include "turn.h"
#include <avs/alignment.h>
#include <avs/minmax.h>
#include "../convert/convert_planar.h"
#include "../convert/convert_yuy2.h"
#include "../convert/convert_helper.h"

#include <type_traits>
#include <algorithm>

#include "../core/avs_simd_c.h"
#include <cassert>

// Prepares resampling coefficients for end conditions and/or SIMD processing by:
// 1. Sets a "real-life" size for the filter, which at small dimensions can be less than the original
// 2. Aligning filter_size to 8 or 16 boundary for SIMD efficiency
// 3. Right-aligning coefficients within padded arrays to ensure valid access at boundaries
//
// Before:                After right-alignment (filter_size=4, kernel_size=2):
//
// offset->|            offset-2 ->|         
//        [x][y][  ][  ]          [0][0][x][y]
//         ^ ^   ^   ^             ^         ^
//         | |   Off-boundary      |         |
//     Values used                 Values used
//
// This ensures SIMD instructions can safely load full vectors even at image boundaries
// while maintaining correct coefficient positioning and proper zero padding.


static void checkAndSetOverread(int end_pos, SafeLimit& safelimit, int start_pos, int i, int source_size) {
  if (end_pos > source_size) {
    if (!safelimit.overread_possible) {
      safelimit.overread_possible = true;
      safelimit.source_overread_offset = start_pos;
      safelimit.source_overread_beyond_targetx = i;
    }
  }
}


void resize_prepare_coeffs(ResamplingProgram* p, IScriptEnvironment* env, int filter_size_alignment) {
  p->filter_size_alignment = filter_size_alignment;
  p->safelimit_filter_size_aligned.overread_possible = false;
  p->safelimit_4_pixels.overread_possible = false;
  p->safelimit_8_pixels.overread_possible = false;
  p->safelimit_16_pixels.overread_possible = false;
  p->safelimit_32_pixels.overread_possible = false;

  // note: filter_size_real was the max(kernel_sizes[])
  int filter_size_aligned = AlignNumber(p->filter_size_real, p->filter_size_alignment);

  int target_size_aligned = AlignNumber(p->target_size, ALIGN_RESIZER_TARGET_SIZE);

  // align target_size to 8 units to allow safe up to 8 pixels/cycle in H resizers. modded later.
  p->target_size_alignment = ALIGN_RESIZER_TARGET_SIZE;


  // Common variables for both float and integer paths
  void* new_coeff = nullptr;
  void* src_coeff = nullptr;
  size_t element_size = 0;

  // allocate for a larger target_size area and nullify the coeffs.
  // Even between target_size and target_size_aligned.
  if (p->bits_per_pixel == 32) {
    element_size = sizeof(float);
    src_coeff = p->pixel_coefficient_float;
    new_coeff = env->Allocate(element_size * target_size_aligned * filter_size_aligned, 64, AVS_NORMAL_ALLOC);
    if (!new_coeff) {
      env->Free(new_coeff);
      env->ThrowError("Could not reserve memory in a resampler.");
    }
    std::fill_n((float*)new_coeff, target_size_aligned * filter_size_aligned, 0.0f);
  }
  else {
    element_size = sizeof(short);
    src_coeff = p->pixel_coefficient;
    new_coeff = env->Allocate(element_size * target_size_aligned * filter_size_aligned, 64, AVS_NORMAL_ALLOC);
    if (!new_coeff) {
      env->Free(new_coeff);
      env->ThrowError("Could not reserve memory in a resampler.");
    }
    memset(new_coeff, 0, element_size * target_size_aligned * filter_size_aligned);
  }

  const int last_line = p->source_size - 1;

  // Process coefficients - common code for both types
  for (int i = 0; i < p->target_size; i++) {
    const int kernel_size = p->kernel_sizes[i];
    const int offset = p->pixel_offset[i];
    const int last_coeff_index = offset + p->filter_size_real - 1;
    const int shift_needed = last_coeff_index > last_line ? p->filter_size_real - kernel_size : 0;

    // Copy coefficients with appropriate shift
    if (p->bits_per_pixel == 32) {
      float* dst = (float*)new_coeff + i * filter_size_aligned;
      float* src = (float*)src_coeff + i * p->filter_size;
      for (int j = 0; j < kernel_size; j++) {
        dst[j + shift_needed] = src[j];
      }
    }
    else {
      short* dst = (short*)new_coeff + i * filter_size_aligned;
      short* src = (short*)src_coeff + i * p->filter_size;
      for (int j = 0; j < kernel_size; j++) {
        dst[j + shift_needed] = src[j];
      }
    }

    // Update offsets and kernel sizes
    p->pixel_offset[i] -= shift_needed;
    p->kernel_sizes[i] += shift_needed;

    // left side, already right padded with zero coeffs, we can
    // change to actual width to the common one
    if(p->kernel_sizes[i] < p->filter_size_real)
      p->kernel_sizes[i] = p->filter_size_real;

    // In a horizontal resizer, when reading filter_size_alignment pixels,
    // we must protect against source scanline overread.
    // Using this not in only 32-bit float resizers is new in 3.7.4.
    const int start_pos = p->pixel_offset[i];
    const int end_pos = start_pos + p->filter_size_real - 1;
    if (end_pos >= p->source_size) {
      // This issue has already been fixed, so it cannot occur.
    }

    // Check for SIMD optimization limits
    // a.) when filter_size_aligned pixels are read (e.g. 16 byte SIMD load: 4 float pixels must be safely read)
    // b.-e.) same for exacly 4, 8, 16 and 32 pixels
    // We register only the first occurrence, because we are entering the danger zone from there.
    // Up to this point, it is safe to read 4/8/... pixels from "start_pos" in the actual line.
    // e.g. reading 4 floats will not read beyond the last pixel in line. Used in modified H resizers.

    checkAndSetOverread(start_pos + filter_size_aligned - 1, p->safelimit_filter_size_aligned, start_pos, i, p->source_size);
    checkAndSetOverread(start_pos + 4 - 1, p->safelimit_4_pixels, start_pos, i, p->source_size);
    checkAndSetOverread(start_pos + 8 - 1, p->safelimit_8_pixels, start_pos, i, p->source_size);
    checkAndSetOverread(start_pos + 16 - 1, p->safelimit_16_pixels, start_pos, i, p->source_size);
    checkAndSetOverread(start_pos + 32 - 1, p->safelimit_32_pixels, start_pos, i, p->source_size);
  }

  // Fill the extra offset after target_size with fake values.
  // Our aim is to have a safe, up to 8 pixels/cycle simd loop for V and specific H resizers.
  // Their coeffs will be 0, so they don't count if such coeffs
  // are multiplied with invalid pixels.
  if (p->target_size < target_size_aligned) {
    p->kernel_sizes.resize(target_size_aligned);
    p->pixel_offset.resize(target_size_aligned);
    for (int i = p->target_size; i < target_size_aligned; ++i) {
      p->kernel_sizes[i] = p->filter_size_real;
      p->pixel_offset[i] = 0; // 0th pixel offset makes no harm
      // even if this ensures the in-line safety, alternative H resizer implementations must
      // not read beyond last line, where y>=height.
    }
  }

  // Free old coefficients and assign new ones
  if (p->bits_per_pixel == 32) {
    env->Free(p->pixel_coefficient_float);
    p->pixel_coefficient_float = (float*)new_coeff;
  }
  else {
    env->Free(p->pixel_coefficient);
    p->pixel_coefficient = (short*)new_coeff;
  }

  p->filter_size = filter_size_aligned;
  // by now coeffs[old_filter_size][target_size] was copied and padded into coeffs[new_filter_size][target_size]
}

/***************************************
 ***** Vertical Resizer Assembly *******
 ***************************************/

template<typename pixel_t>
static void resize_v_planar_pointresize(BYTE* dst, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel)
{
  AVS_UNUSED(bits_per_pixel);

  pixel_t* src0 = (pixel_t*)src;
  pixel_t* dst0 = (pixel_t*)dst;
  src_pitch = src_pitch / sizeof(pixel_t);
  dst_pitch = dst_pitch / sizeof(pixel_t);

  for (int y = 0; y < target_height; y++) {
    int offset = program->pixel_offset[y];
    const pixel_t* src_ptr = src0 + src_pitch * offset;

    memcpy(dst0, src_ptr, width * sizeof(pixel_t));

    dst0 += dst_pitch;
  }
}

// This C implementation isn't optimized for auto-vectorization,
// But on x86 MSVC SSE2 settings for 8 bit pixel types it performs better than our 
// vectorizer-friendly version.
// Other compilers benefit from vectorizing by a huge margin for every case.
// The vector code which replaced this is already 2x faster for 10-16 bits processing
// and achieves a 6x speedup for 32-bit operations. LLVM has even an additional 1.5x-4x speedup 
// compared to MSVC.
// Kept for reference, supports 8, 10-16 and 32 bit pixel types.
template<typename pixel_t, bool lessthan16bit>
static void resize_v_c_planar(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel)
{

  int filter_size = program->filter_size;

  typedef typename std::conditional < std::is_floating_point<pixel_t>::value, float, short>::type coeff_t;
  coeff_t* current_coeff;

  if (!std::is_floating_point<pixel_t>::value)
    current_coeff = (coeff_t*)program->pixel_coefficient;
  else
    current_coeff = (coeff_t*)program->pixel_coefficient_float;

  pixel_t* src = (pixel_t*)src8;
  pixel_t* dst = (pixel_t*)dst8;
  src_pitch = src_pitch / sizeof(pixel_t);
  dst_pitch = dst_pitch / sizeof(pixel_t);

  pixel_t limit = 0;
  if (!std::is_floating_point<pixel_t>::value) {  // floats are unscaled and uncapped
    if constexpr (sizeof(pixel_t) == 1) limit = 255;
    else if constexpr (sizeof(pixel_t) == 2) limit = pixel_t((1 << bits_per_pixel) - 1);
  }

  // for 16 bits only
  const short shifttosigned_short = -32768;
  const int shiftfromsigned_int = 32768 << FPScale16bits;

  for (int y = 0; y < target_height; y++) {
    int offset = program->pixel_offset[y];
    const int kernel_size = program->kernel_sizes[y];
    const pixel_t* src_ptr = src + src_pitch * offset;

    // perhaps helps vectorizing decision
    const int ksmod4 = kernel_size / 4 * 4;

    for (int x = 0; x < width; x++) {
      if constexpr (std::is_floating_point<pixel_t>::value) {
        const float* src2_ptr = src_ptr + x;

        float result = 0;
        for (int i = 0; i < ksmod4; i += 4) {
          result += *(src2_ptr + 0 * src_pitch) * current_coeff[i + 0];
          result += *(src2_ptr + 1 * src_pitch) * current_coeff[i + 1];
          result += *(src2_ptr + 2 * src_pitch) * current_coeff[i + 2];
          result += *(src2_ptr + 3 * src_pitch) * current_coeff[i + 3];
          src2_ptr += 4 * src_pitch;
        }
        for (int i = ksmod4; i < kernel_size; i++) {
          result += *src2_ptr * current_coeff[i];
          src2_ptr += src_pitch;
        }
        dst[x] = result;
      }
      else if constexpr (sizeof(pixel_t) == 2) {
        // theoretically, no need for int64 accumulator,
        // sum of coeffs is 1.0 that is (1 << FPScale16bits) in integer arithmetic
        const uint16_t* src2_ptr = src_ptr + x;
        int result = 1 << (FPScale16bits - 1); // rounder;
        for (int i = 0; i < ksmod4; i += 4) {
          int val;
          val = *(src2_ptr + 0 * src_pitch);
          if constexpr (!lessthan16bit)
            val = val + shifttosigned_short;
          result += val * current_coeff[i + 0];

          val = *(src2_ptr + 1 * src_pitch);
          if constexpr (!lessthan16bit)
            val = val + shifttosigned_short;
          result += val * current_coeff[i + 1];

          val = *(src2_ptr + 2 * src_pitch);
          if constexpr (!lessthan16bit)
            val = val + shifttosigned_short;
          result += val * current_coeff[i + 2];

          val = *(src2_ptr + 3 * src_pitch);
          if constexpr (!lessthan16bit)
            val = val + shifttosigned_short;
          result += val * current_coeff[i + 3];

          src2_ptr += 4 * src_pitch;
        }
        for (int i = ksmod4; i < kernel_size; i++) {
          int val = *src2_ptr;
          if constexpr (!lessthan16bit)
            val = val + shifttosigned_short;
          result += val * current_coeff[i];
          src2_ptr += src_pitch;
        }
        if constexpr (!lessthan16bit)
          result = result + shiftfromsigned_int;
        result = result >> FPScale16bits;
        result = result > limit ? limit : result < 0 ? 0 : result; // clamp 10..16 bits
        dst[x] = (uint16_t)result;
      }
      else if constexpr (sizeof(pixel_t) == 1) {
        const uint8_t* src2_ptr = src_ptr + x;
        int result = 1 << (FPScale8bits - 1); // rounder;
        for (int i = 0; i < ksmod4; i += 4) {
          short val;
          val = *(src2_ptr + 0 * src_pitch);
          result += val * current_coeff[i + 0];

          val = *(src2_ptr + 1 * src_pitch);
          result += val * current_coeff[i + 1];

          val = *(src2_ptr + 2 * src_pitch);
          result += val * current_coeff[i + 2];

          val = *(src2_ptr + 3 * src_pitch);
          result += val * current_coeff[i + 3];

          src2_ptr += 4 * src_pitch;
        }
        for (int i = ksmod4; i < kernel_size; i++) {
          short val = *src2_ptr;
          result += val * current_coeff[i];
          src2_ptr += src_pitch;
        }
        result = result >> FPScale8bits;
        result = result > limit ? limit : result < 0 ? 0 : result; // clamp 8 bits
        dst[x] = (uint8_t)result;
      }
    }

    dst += dst_pitch;
    current_coeff += filter_size;
  }
}


/*

Benchmarks.

  SetMaxCPU("none")
  #SetMaxCPU("SSSE3")
  #SetMaxCPU("AVX2")
  ColorbarsHD(200,200, pixel_type="YUV444P8") # P8, P10, P16, PS

  wmul=2 # >1: horizontal resizer test
  hmul=1 # >1: vertical resizer test

  avs=LanczosResize(width*wmul, height*hmul, taps=16)
  avsr=z_ConvertFormat(width=width*wmul, height=height*hmul, resample_filter="lanczos", filter_param_a=16)
  fmtconv=fmtc_resample(w=width*wmul, h=height*hmul, kernel="lanczos", taps=16)
  avs

Figures in general has no place in source code but it is useful to have a clue 
and see the huge differences between compilers and code variants.
Also, only mazochists want to test integer optimization with MSVC (x86). Wasted time.
Use llvm on Arm, clangcl/llvm on Windows x86.

Horizontals                Hor+Vert    Verticals
 [8]  [10-14]  [16]  [32]  [8]  [10]   [8]  [10-14] [16] [32]
 177     170    101  208               262   486    417   217    C-RaspberryPi5 gcc 12.2 (code variant)
 137     469     88  208                                         C-RaspberryPi5 gcc 12.2 (code variant)
 227     218    147  178                                         C-RaspberryPi5 gcc 12.2 no vector attrib ??! Quicker than vector attrib version gcc not recommended
 416     611    579  404    128  186   362   609    556   624    C-RaspberryPi5 llvm 14 vector attrib
  90     185     94  403                                         C-RaspberryPi5 llvm 14 vector attrib + integer madd@H
 180     182    162          67        213   212    206   639    C-RaspberryPi5 llvm 14 no vector attrib
1230    1168   1129                                              C-ClangCl in VS2022 SSE2
1270    1238   1186                   1550  1555   1560  3670    C-ClangCl in VS2022 AVX2
1051    1126   1102                                              C-Intel ICX 2025 SSE2
1513    2355   1560 1128                                 1969    C-Intel ICX 2025 SSE4.2  smart madd!
1938    2413   1775 1061    442  453  1136  1126   1037  3511    C-Intel ICX 2025 AVX2
 212     188    187  264     73   64   223   195    198   268    C-MSVC SSE2 3.7.3
 417     463    360                    449   424    384   352    C-MSVC SSE2 3.7.4 (some unrolling vs. 3.7.3)
 215     215     97  928     79   79   220   744     96  1951    C-MSVC SSE2 (zero optim on 8-16, did not tolerate vector-friendly code)
 201     206     99                                              C-MSVC AVX2 (zero optim also on 8-16, not even using SSE2 xmm registers)
 597     631    651                                              C-Intel SSE4.2  3.7.4 code
1183    1193    889                                              C-Intel AVX2    3.7.4 code
5600                       2140                                  SIMD-avsresize (AVX2 or AVX512?)
2260                        840                                  SIMD-fmtconv (16 bit output for 8 bits)
4578    2614   2560 2250   1490       4220  4534   3887  3570    SIMD-MSVC AVX2 3.7.3 (horizontal was memory-boundary unsafe)
3631    3505   3221 2344   1291 1354  3804  4466   3855  2260    SIMD-MSVC AVX2 3.7.4 Float vertical regression - no time to finish
3720    3478   3130 2385   1566 1480  5014  5288   5077  3810    SIMD-MSVC AVX2 + incrementing offsets in V, 20-25% gain in integer verticals
4730    4612   4233 2487   1390 1471  3792  4476   4380  3942    SIMD-ClangCl AVX2, verticals behind MSVC by surprise
2373    2181   1893 1306    868  723  2660  2137   2670  1886    SIMD-MSVC SSSE3 + incrementing offsets in V, 20-25% gain in integer verticals
2294    2979   2595 1460    859  976  2623  2865   2625  1962    SIMD-Intel ICX 2025 SSSE3
4395    4616   4160 2570   1664 1720  5110  5870   5085  2999    SIMD-Intel ICX 2025 AVX2 Surprisingly slow at vertical float FIXME, slower than C :)

* float has different optimization: 8 pixels 2 coeffs
** on aarch64 the float benchmarks included a ConvertBits(8) at the end.

Non-x86 platforms are the focus of this comparison, as they lack SIMD optimized paths.
x86 platforms have their own SIMD optimized paths and do not use C.

Compilers can give totally different results even after some reordering of the code,
See gcc aarch64: the 8 bit case dropped to 2/3 speed while 16 bits increased by a factor of 250%
depending on where I put a line.

Considerations for aarch64 (Armv8a, includes neon, e.g. RPi5 with gcc 12.2, llvm 14.0.6)
- gcc is not recommended, their vectorizer is either broken or not yet ready.
  Using vector attribute gave slower code.
- llvm gave consistent fast results.
- Vertical resampler: aarch64 + GCC, (8-16 bits) the 4-pixel 4-coefficient version performs best.

On x86 platforms MSVC is unusable regarding integer vectorization, the more the code helps to
recognize vectorization patterns the slower assembly it produces, e.g. horizontal resizer AVX2 16 bit: 
MSVC 99 fps, MSVC+clangcl: 1186 fps, Intel+LLVM: 1775 fps. Joke.

MSVC on x86 is only capable of vectorizing 32 bit float code, and even then it is not the fastest.
There is no reason to not use clangcl for x86 Visual Studio builds. It comes officially with VS2022 for free.

*/

// This is a vectorizer-friendly version of the above function.
template<typename pixel_t, bool lessthan16bit>
void resize_v_c_planar_uint8_16_t_auto_vectorized(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel) {

  const short* AVS_RESTRICT current_coeff = program->pixel_coefficient;

  auto src = reinterpret_cast<const pixel_t*>(src8);
  auto dst = reinterpret_cast<pixel_t * AVS_RESTRICT>(dst8);
  src_pitch = src_pitch / sizeof(pixel_t);
  dst_pitch = dst_pitch / sizeof(pixel_t);

  int limit = 0;
  if constexpr (sizeof(pixel_t) == 1) limit = 255;
  else if constexpr (sizeof(pixel_t) == 2) limit = pixel_t((1 << bits_per_pixel) - 1);

  // for 16 bits only
  [[maybe_unused]] Int16x4 shifttosigned_short(-32768);
  [[maybe_unused]] const Int32x4 shiftfromsigned_int(32768 << FPScale16bits);

  const int filter_size = program->filter_size;
  constexpr int fixpoint_scaler_bits = sizeof(pixel_t) == 2 ? FPScale16bits : FPScale8bits;
  const int rounder = 1 << (fixpoint_scaler_bits - 1);

  // 8-16 SIMD padding of coeffs is not usable here, work with single coeffs

  const int kernel_size = program->filter_size_real;

  const int ksmod4 = kernel_size / 4 * 4;

  for (int y = 0; y < target_height; y++) {
    int offset = program->pixel_offset[y];
    const pixel_t* src_ptr = src + offset * src_pitch;

    // 4 pixels at a time
    for (int x = 0; x < width; x += 4) {
      Int32x4 result(rounder); // master accumulator and the initial first coeff part
      Int32x4 result_2(rounder);
      Int32x4 result_3(rounder);
      Int32x4 result_4(rounder);
      const pixel_t* AVS_RESTRICT src2_ptr = src_ptr + x; // __restrict here
      int i = 0;
      // Process coefficients in pairs or quads for better instruction parallelism.
      // Depending on platform and compiler, results may vary.
      // Since on Intel we have SIMD optimized paths, we decide on the
      // results of a not-yet-optimized aarch64 platform: takeout is 4 pix 4 coeff
      for (; i < ksmod4; i += 4) {

        Int32x4 src_1, src_2, src_3, src_4;

        const int coeff_1 = current_coeff[i];
        const int coeff_2 = current_coeff[i + 1];
        const int coeff_3 = current_coeff[i + 2];
        const int coeff_4 = current_coeff[i + 3];

        if constexpr (sizeof(pixel_t) == 1) {
          // uint8_t
          auto src8_1 = Uint8x4::from_ptr(src2_ptr);
          auto src8_2 = Uint8x4::from_ptr(src2_ptr + src_pitch);
          auto src8_3 = Uint8x4::from_ptr(src2_ptr + 2 * src_pitch);
          auto src8_4 = Uint8x4::from_ptr(src2_ptr + 3 * src_pitch);

          src_1 = Int32x4::convert_from(src8_1);
          src_2 = Int32x4::convert_from(src8_2);
          src_3 = Int32x4::convert_from(src8_3);
          src_4 = Int32x4::convert_from(src8_4);
        }
        else {
          // uint16_t
          auto src16_1 = Int16x4::from_ptr(reinterpret_cast<const short* AVS_RESTRICT>(src2_ptr));
          auto src16_2 = Int16x4::from_ptr(reinterpret_cast<const short* AVS_RESTRICT>(src2_ptr + src_pitch));
          auto src16_3 = Int16x4::from_ptr(reinterpret_cast<const short* AVS_RESTRICT>(src2_ptr + 2 * src_pitch));
          auto src16_4 = Int16x4::from_ptr(reinterpret_cast<const short* AVS_RESTRICT>(src2_ptr + 3 * src_pitch));

          // Turn unsigned to signed 16 bit, will be adjusted back before scaling back and storing.
          // Since there's a little hope that short*short pattern is recognized by the compiler, though 
          // this is different from the horizontal case where hadd can be used.
          if constexpr (!lessthan16bit) {
            src16_1 += shifttosigned_short;
            src16_2 += shifttosigned_short;
            src16_3 += shifttosigned_short;
            src16_4 += shifttosigned_short;
          }
          // widen short->int
          src_1 = Int32x4::convert_from(src16_1);
          src_2 = Int32x4::convert_from(src16_2);
          src_3 = Int32x4::convert_from(src16_3);
          src_4 = Int32x4::convert_from(src16_4);
        }

        result += src_1 * coeff_1;
        result_2 += src_2 * coeff_2;
        result_3 += src_3 * coeff_3;
        result_4 += src_4 * coeff_4;

        src2_ptr += 4 * src_pitch;
      }

      result += result_2;
      result_3 += result_4;
      result += result_3;

      // rest zero or one
      for (; i < kernel_size; ++i) {
        Int32x4 src;
        const int a_coeff = current_coeff[i];
        if constexpr (sizeof(pixel_t) == 1) {
          // uint8_t
          src.load_from_any_intptr(src2_ptr);
        }
        else {
          // uint16_t
          auto src16 = Int16x4::from_ptr(reinterpret_cast<const short* AVS_RESTRICT>(src2_ptr));
          if constexpr (!lessthan16bit) {
            src16 += shifttosigned_short;
          }
          src.convert_from(src16); // widen short->int
        }

        result += src * a_coeff;

        src2_ptr += src_pitch;
      }

      // back to unsigned 16 bit for exact 16 bit source
      if constexpr (sizeof(pixel_t) == 2 && !lessthan16bit) {
        result += shiftfromsigned_int;
      }

      result >>= fixpoint_scaler_bits;

      if constexpr (sizeof(pixel_t) == 1) {
        Uint8x4 result8;
        convert_and_saturate_int32x4_to_uint8x4(result, result8);
        result8.store(dst + x);
      }
      else {
        Uint16x4 result16;
        if constexpr (lessthan16bit) {
          convert_and_saturate_int32x4_to_uint16x4_limit(result, result16, limit);
        }
        else {
          convert_and_saturate_int32x4_to_uint16x4(result, result16);
        }
        result16.store(dst + x);
      }
    }

    dst += dst_pitch;
    current_coeff += filter_size;
  }
}


static void resize_v_c_planar_float_auto_vectorized(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel) {

  const float* AVS_RESTRICT current_coeff = program->pixel_coefficient_float;

  auto src = reinterpret_cast<const float*>(src8);
  auto dst = reinterpret_cast<float* AVS_RESTRICT>(dst8);
  src_pitch = src_pitch / sizeof(float);
  dst_pitch = dst_pitch / sizeof(float);

  const int filter_size = program->filter_size;

  // 8-16 SIMD padding of coeffs is not usable here, work with single coeffs
  const int kernel_size = program->filter_size_real;
  const int ksmod2 = kernel_size / 2 * 2; // Ensure kernel_size is processed in multiples of 2

  for (int y = 0; y < target_height; y++) {
    int offset = program->pixel_offset[y];
    const float* src_ptr = src + offset * src_pitch;

    for (int x = 0; x < width; x += 8) {
      // This function written in vectorizer-friendly C unexpectedly outperformed 
      // our hand-written SSE2 version.
      // The reason: Float8 class is probably using two 128-bit XMM registers
      // when compiled with SSE2 enabled by MSVC, processing 8 floats per iteration.
      // Our original SSE2 function only processed 4 pixels at once, explaining the
      // performance difference. We've since updated the SSE2 version to also process
      // 8 pixels simultaneously using two XMM registers.
      // Processing two coefficients further increases the throughput.
      Float8 result(0.f);
      Float8 result_2(0.f);
      const float* AVS_RESTRICT src2_ptr = src_ptr + x;

      for (int i = 0; i < ksmod2; i += 2) {
        const float coeff_1 = current_coeff[i];
        const float coeff_2 = current_coeff[i + 1];
        Float8 src_1 = Float8::from_ptr(src2_ptr);
        Float8 src_2 = Float8::from_ptr(src2_ptr + src_pitch);
        result += src_1 * coeff_1;
        result_2 += src_2 * coeff_2;

        src2_ptr += 2 * src_pitch;
      }

      result += result_2;

      // Process remaining coefficients if kernel_size is odd
      if (ksmod2 < kernel_size) {
        const float a_coeff = current_coeff[ksmod2];
        Float8 src = Float8::from_ptr(src2_ptr);
        result += src * a_coeff;
      }

      // no rounding, no clamp
      result.store(dst + x);
    }

    dst += dst_pitch;
    current_coeff += filter_size;
  }
}


/***************************************
 ********* Horizontal Resizer** ********
 ***************************************/

// Only <float> is used, which got some SIMD-C optimizations.
// On x86 MSVC the 8-16 bit versions may perform better due to their extremely poor integer 
// vectorization capabilities.
// But gcc and llvm (clangcl) compilers perform very good, their code from the other,
// vector-fiendly versions are excellent, as expected in 2025.
// Luckily, on x86 there are handcrafted SIMD versions for all 8, 10-16 and 32 bit pixel types.

template<typename pixel_t, bool lessthan16bit>
static void resize_h_c_planar(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel) {
  int filter_size = program->filter_size;

  typedef typename std::conditional < std::is_floating_point<pixel_t>::value, float, short>::type coeff_t;
  const coeff_t* AVS_RESTRICT current_coeff;

  pixel_t limit = 0;
  if (!std::is_floating_point<pixel_t>::value) {  // floats are unscaled and uncapped
    if constexpr (sizeof(pixel_t) == 1) limit = 255;
    else if constexpr (sizeof(pixel_t) == 2) limit = pixel_t((1 << bits_per_pixel) - 1);
  }

  src_pitch = src_pitch / sizeof(pixel_t);
  dst_pitch = dst_pitch / sizeof(pixel_t);

  pixel_t* src = (pixel_t*)src8;
  pixel_t* dst = (pixel_t*)dst8;

  // for 16 bits only
  const short shifttosigned_short = -32768;
  const int shiftfromsigned_int = 32768 << FPScale16bits;

  // perhaps helps vectorizing decision
  const int kernel_size = program->filter_size_real;
  const int ksmod4 = kernel_size / 4 * 4;
  const int ksmod8 = kernel_size / 8 * 8;

  // external loop y is much faster
  for (int y = 0; y < height; y++) {
    if (!std::is_floating_point<pixel_t>::value)
      current_coeff = (const coeff_t* AVS_RESTRICT)program->pixel_coefficient;
    else
      current_coeff = (const coeff_t* AVS_RESTRICT)program->pixel_coefficient_float;

    pixel_t* AVS_RESTRICT dst2_ptr = dst + y * dst_pitch;
    const pixel_t* src_ptr = src + y * src_pitch;

    for (int x = 0; x < width; x++) {
      int begin = program->pixel_offset[x];
      const pixel_t* AVS_RESTRICT src2_ptr = src_ptr + begin;

      if constexpr (std::is_floating_point<pixel_t>::value) {
        Float4 result(0.f);
        for (int i = 0; i < ksmod4; i += 4) {
          Float4 src = Float4::from_ptr(src2_ptr + i);
          Float4 coeff = Float4::from_ptr(current_coeff + i);
          result += src * coeff;
        }
        float result_single = result.horiz_add_float();
        for (int i = ksmod4; i < kernel_size; i++) {
          result_single += src2_ptr[i] * current_coeff[i];
        }
        dst2_ptr[x] = result_single;
      }
      else if constexpr (sizeof(pixel_t) == 2) {
        // theoretically, no need for int64 accumulator,
        // sum of coeffs is 1.0 that is (1 << FPScale16bits) in integer arithmetic
        int result = 1 << (FPScale16bits - 1); // rounder;
        for (int i = 0; i < ksmod4; i += 4) {
          int val;
          val = src2_ptr[i+0];
          if constexpr (!lessthan16bit)
            val = val + shifttosigned_short;
          result += val * current_coeff[i+0];

          val = src2_ptr[i+1];
          if constexpr (!lessthan16bit)
            val = val + shifttosigned_short;
          result += val * current_coeff[i+1];

          val = src2_ptr[i + 2];
          if constexpr (!lessthan16bit)
            val = val + shifttosigned_short;
          result += val * current_coeff[i + 2];

          val = src2_ptr[i + 3];
          if constexpr (!lessthan16bit)
            val = val + shifttosigned_short;
          result += val * current_coeff[i + 3];
        }
        for (int i = ksmod4; i < kernel_size; i++) {
          int val = src2_ptr[i];
          if constexpr (!lessthan16bit)
            val = val + shifttosigned_short;
          result += val * current_coeff[i];
        }
        if constexpr (!lessthan16bit)
          result = result + shiftfromsigned_int;
        result = result >> FPScale16bits;
        result = result > limit ? limit : result < 0 ? 0 : result; // clamp 10..16 bits
        dst2_ptr[x] = (uint16_t)result;
      }
      else if constexpr (sizeof(pixel_t) == 1) {
        int result = 1 << (FPScale8bits - 1); // rounder;
        for (int i = 0; i < ksmod8; i += 8) {
          short val;
          val = src2_ptr[i + 0];
          result += val * current_coeff[i + 0];
          val = src2_ptr[i + 1];
          result += val * current_coeff[i + 1];
          val = src2_ptr[i + 2];
          result += val * current_coeff[i + 2];
          val = src2_ptr[i + 3];
          result += val * current_coeff[i + 3];
          val = src2_ptr[i + 4];
          result += val * current_coeff[i + 4];
          val = src2_ptr[i + 5];
          result += val * current_coeff[i + 5];
          val = src2_ptr[i + 6];
          result += val * current_coeff[i + 6];
          val = src2_ptr[i + 7];
          result += val * current_coeff[i + 7];
        }
        for (int i = ksmod8; i < ksmod4; i += 4) {
          short val;
          val = src2_ptr[i + 0];
          result += val * current_coeff[i + 0];
          val = src2_ptr[i + 1];
          result += val * current_coeff[i + 1];
          val = src2_ptr[i + 2];
          result += val * current_coeff[i + 2];
          val = src2_ptr[i + 3];
          result += val * current_coeff[i + 3];
          val = src2_ptr[i + 4];
        }
        for (int i = ksmod4; i < kernel_size; i++) {
          short val = src2_ptr[i];
          result += val * current_coeff[i];
        }
        result = result >> FPScale8bits;
        result = result > limit ? limit : result < 0 ? 0 : result; // clamp 8 bits
        dst2_ptr[x] = (uint8_t)result;
      }
      current_coeff += filter_size;
    }
  }
}

// Vectorizer-friendly 8-16 bit horizontal resampler

// 8 pixel instead of real-SIMD 16 to ease register pressure 
// and the make the compiler task easier
template<typename pixel_t, bool lessthan16bit>
AVS_FORCEINLINE static void process_two_8pixels_h_uint8_16_core(const pixel_t* AVS_RESTRICT src_ptr, const short* AVS_RESTRICT current_coeff, Int32x4& result, const Int16x8& shifttosigned_8xshort) {

  Int16x8 data;
  // using signed int16 (short) until the coeff multiplication
  if constexpr (sizeof(pixel_t) == 1) {
    // pixel_t is uint8_t
    Uint8x8 src = Uint8x8::from_ptr(src_ptr);
    data = Int16x8::convert_from(src); // 8 pixels uint8_t->int16
  }
  else {
    // pixel_t is uint16_t, at exact 16 bit bit depth unsigned -> signed 16 bit conversion needed
    // this mimics the behavior of the SIMD version, which processes signed 16 bit input
    // data.load(reinterpret_cast<const short * AVS_RESTRICT>(src_ptr));
    Uint16x8 src = Uint16x8::from_ptr(src_ptr);
    data = Int16x8::convert_from(src); // 8 pixels uint8_t->int16

    if constexpr (!lessthan16bit) {
      data += shifttosigned_8xshort; // 16 bit addition is invariant regarding overflow
    }
  }

  Int16x8 coeff = Int16x8::from_ptr(current_coeff); // 8 coeffs

  // no need real intel-like madd, but intel compiler can use it (experienced)
#if defined(__INTEL_LLVM_COMPILER)
  // Intel proc + LLVM: only this compiler combo is compiling into madd,which is much faster.
  Int32x4 madd_result = simul_madd_epi16(data, coeff);
#else
  Int32x4 madd_result = mul16x16_reduce_to_Int32x4(data, coeff);
#endif
  result += madd_result;
  // later, the four 32 bit results will be further reduced and added together (hadd)
}


template<typename pixel_t, bool lessthan16bit>
AVS_FORCEINLINE static void process_two_4pixels_h_uint8_16_core(const pixel_t* AVS_RESTRICT src_ptr, const short* AVS_RESTRICT current_coeff, Int32x4& result, const Int32x4& shifttosigned_4xint) {
  Int32x4 data;

  // this one is using full int32 internally

  if constexpr (sizeof(pixel_t) == 1) {
    data.load_from_any_intptr(src_ptr); // 4 pixels uint8_t->int32
  }
  else {
    // pixel_t is uint16_t, at exact 16 bit bit depth an unsigned -> signed 16 bit conversion needed
    data.load_from_any_intptr(src_ptr); // 4 pixels uint16_t->int32

    if constexpr (!lessthan16bit) {
      data += shifttosigned_4xint;
    }
  }

  // 4x signed 16 bit coeffs to int32
  Int32x4 coeff_int;
  coeff_int.load_from_any_intptr(current_coeff);

  result += data * coeff_int;
  // later, the four 32 bit results will be added together (hadd)
}

template<bool safe_aligned_mode, typename pixel_t, bool lessthan16bit>
AVS_FORCEINLINE static void process_two_pixels_h_uint8_16(const pixel_t* src_ptr, const int begin1, const int begin2, const short* AVS_RESTRICT current_coeff, const int filter_size,
  Int32x4& result1, Int32x4& result2, const int kernel_size,
  const Int16x8& shifttosigned) {

  int ksmod8;
  // 16 is too much for C optimizer
  if constexpr (safe_aligned_mode)
    ksmod8 = filter_size / 8 * 8;
  else
    ksmod8 = kernel_size / 8 * 8; // danger zone, scanline overread possible. Use exact unaligned kernel_size
  const pixel_t* src_ptr1 = src_ptr + begin1;
  const pixel_t* src_ptr2 = src_ptr + begin2;
  int i = 0;

  // Process 16 elements at a time
  for (; i < ksmod8; i += 8) {
    process_two_8pixels_h_uint8_16_core<pixel_t, lessthan16bit>(src_ptr1 + i, current_coeff + i, result1, shifttosigned);
    process_two_8pixels_h_uint8_16_core<pixel_t, lessthan16bit>(src_ptr2 + i, current_coeff + filter_size + i , result2, shifttosigned);
  }

  if constexpr (!safe_aligned_mode) {
    // working with the original, unaligned kernel_size
    if (i == kernel_size) return;

    const int ksmod4 = kernel_size / 4 * 4;
    // Process 4 elements if needed
    if (i < ksmod4) {
      Int32x4 shifttosigned_4;
      shifttosigned_4.convert_from_lo(shifttosigned); // copy lower half of 8xshort, 4xshort to 4xint
      process_two_4pixels_h_uint8_16_core<pixel_t, lessthan16bit>(src_ptr1 + i, current_coeff + i, result1, shifttosigned_4);
      process_two_4pixels_h_uint8_16_core<pixel_t, lessthan16bit>(src_ptr2 + i, current_coeff + filter_size + i, result2, shifttosigned_4);
      i += 4;
      if (i == kernel_size) return;
    }

    // Process remaining 1-3 elements with scalar operations
    if (i < kernel_size) {
      Int32x4 scalar_sum1(0); // like an __m128i
      Int32x4 scalar_sum2(0);

      int index = 0;
      for (; i < kernel_size; i++, index++) {

        if constexpr (sizeof(pixel_t) == 1) {
          scalar_sum1.set(index, scalar_sum1[index] + src_ptr1[i] * current_coeff[i]);
          scalar_sum2.set(index, scalar_sum2[index] + src_ptr2[i] * current_coeff[filter_size + i]);
        }
        else {
          // pixel_t is uint16_t
          short val = src_ptr1[i];
          if constexpr (!lessthan16bit)
            val = val + shifttosigned[0]; // still short
          scalar_sum1.set(index, scalar_sum1[index] + val * current_coeff[i]);

          val = src_ptr2[i];
          if constexpr (!lessthan16bit)
            val = val + shifttosigned[0]; // still short
          scalar_sum2.set(index, scalar_sum2[index] + val * current_coeff[filter_size + i]);
        }
      }

      // update result vectors
      result1 += scalar_sum1;
      result2 += scalar_sum2;

    }
  }
}


template<bool is_safe, typename pixel_t, bool lessthan16bit>
AVS_FORCEINLINE static void process_eight_pixels_h_uint8_16(const pixel_t * AVS_RESTRICT src, int x, const short* current_coeff_base, int filter_size,
  const Int32x4& rounder128, const Int16x8& shifttosigned, const uint16_t clamp_limit,
  pixel_t* AVS_RESTRICT dst,
  ResamplingProgram* program)
{
  assert(program->filter_size_alignment >= 16); // code assumes this

  const short* AVS_RESTRICT current_coeff = current_coeff_base + x * filter_size;
  const int unaligned_kernel_size = program->filter_size_real;

  // Unrolled processing of all 8 pixels

  // hadd_epi32 or other SIMD pattern is not recognized, do it differently

  // 0 & 1
  Int32x4 result0 = rounder128;
  Int32x4 result1 = rounder128;
  int begin0 = program->pixel_offset[x + 0];
  int begin1 = program->pixel_offset[x + 1];
  process_two_pixels_h_uint8_16<is_safe, pixel_t, lessthan16bit>(src, begin0, begin1, current_coeff, filter_size, result0, result1, unaligned_kernel_size, shifttosigned);
  current_coeff += 2 * filter_size;

  // 2 & 3
  Int32x4 result2 = rounder128;
  Int32x4 result3 = rounder128;
  begin0 = program->pixel_offset[x + 2];
  begin1 = program->pixel_offset[x + 3];
  process_two_pixels_h_uint8_16<is_safe, pixel_t, lessthan16bit>(src, begin0, begin1, current_coeff, filter_size, result2, result3, unaligned_kernel_size, shifttosigned);
  current_coeff += 2 * filter_size;

  Int32x4 sumQuad1234;
  sumQuad1234.set(0, result0.horiz_add_int32());
  sumQuad1234.set(1, result1.horiz_add_int32());
  sumQuad1234.set(2, result2.horiz_add_int32());
  sumQuad1234.set(3, result3.horiz_add_int32());

  // 4 & 5
  result0 = rounder128;
  result1 = rounder128;
  begin0 = program->pixel_offset[x + 4];
  begin1 = program->pixel_offset[x + 5];
  process_two_pixels_h_uint8_16<is_safe, pixel_t, lessthan16bit>(src, begin0, begin1, current_coeff, filter_size, result0, result1, unaligned_kernel_size, shifttosigned);
  current_coeff += 2 * filter_size;

  // 6 & 7
  result2 = rounder128;
  result3 = rounder128;
  begin0 = program->pixel_offset[x + 6];
  begin1 = program->pixel_offset[x + 7];
  process_two_pixels_h_uint8_16<is_safe, pixel_t, lessthan16bit>(src, begin0, begin1, current_coeff, filter_size, result2, result3, unaligned_kernel_size, shifttosigned);
  // current_coeff += 2 * filter_size; // not needed anymore

  Int32x4 sumQuad5678;
  sumQuad5678.set(0, result0.horiz_add_int32());
  sumQuad5678.set(1, result1.horiz_add_int32());
  sumQuad5678.set(2, result2.horiz_add_int32());
  sumQuad5678.set(3, result3.horiz_add_int32());

  // correct if signed, scale back, store
  if constexpr (sizeof(pixel_t) == 2 && !lessthan16bit) {
    const Int32x4 shiftfromsigned(32768 << FPScale16bits);
    sumQuad1234 += shiftfromsigned;
    sumQuad5678 += shiftfromsigned;
  }

  const int current_fp_scale_bits = (sizeof(pixel_t) == 1) ? FPScale8bits : FPScale16bits;
  // scale back, store
  sumQuad1234 >>= current_fp_scale_bits;
  sumQuad5678 >>= current_fp_scale_bits;
 
  if constexpr (sizeof(pixel_t) == 1) {
    Uint8x8 result_2x4x_uint8;
    convert_and_saturate_int32x4x2_to_uint8x8(sumQuad1234, sumQuad5678, result_2x4x_uint8);
    result_2x4x_uint8.store(&dst[x]);
  }
  else {
    // uint16_t 10-16 bit
    Uint16x8 result_2x4x_uint16_128;
    if constexpr (lessthan16bit) {
      convert_and_saturate_int32x4x2_to_uint16x8_limit(sumQuad1234, sumQuad5678, result_2x4x_uint16_128, clamp_limit);
    }
    else {
      convert_and_saturate_int32x4x2_to_uint16x8(sumQuad1234, sumQuad5678, result_2x4x_uint16_128);
    }
    result_2x4x_uint16_128.store(&dst[x]);
  
  }
}

//-------- uint8/16_t Horizontal
// 4 pixels at a time. 
template<typename pixel_t, bool lessthan16bit>
void resizer_h_c_generic_uint8_16_vectorized(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel) {

  const int filter_size = program->filter_size;
  const int current_fp_scale_bits = (sizeof(pixel_t) == 1) ? FPScale8bits : FPScale16bits;
  const Int32x4 rounder128 = { 1 << (current_fp_scale_bits - 1), 0, 0, 0 };

  const Int16x8 shifttosigned_or_zero128(sizeof(pixel_t) == 1 ? 0 : -32768);

  const uint16_t clamp_limit = (1 << bits_per_pixel) - 1;

  const pixel_t* AVS_RESTRICT src = reinterpret_cast<const pixel_t*>(src8);
  pixel_t* AVS_RESTRICT dst = reinterpret_cast<pixel_t*>(dst8);
  dst_pitch /= sizeof(pixel_t);
  src_pitch /= sizeof(pixel_t);

  const int w_safe_mod8 = (program->safelimit_filter_size_aligned.overread_possible ? program->safelimit_filter_size_aligned.source_overread_beyond_targetx : width) / 8 * 8;

  for (int y = 0; y < height; y++) {
    const short* current_coeff_base = program->pixel_coefficient;

    // Process safe aligned pixels
    for (int x = 0; x < w_safe_mod8; x += 8) {
      process_eight_pixels_h_uint8_16<true, pixel_t, lessthan16bit>(src, x, current_coeff_base, filter_size, rounder128, shifttosigned_or_zero128, clamp_limit, dst, program);
    }

    // Process up to the actual kernel size instead of the aligned filter_size to prevent overreading beyond the last source pixel.
    // We assume extra offset entries were added to the p->pixel_offset array (aligned to 8 during initialization).
    // This may store 1-7 false pixels but it still remain in alignment-safe area.
    for (int x = w_safe_mod8; x < width; x += 8) {
      process_eight_pixels_h_uint8_16<false, pixel_t, lessthan16bit>(src, x, current_coeff_base, filter_size, rounder128, shifttosigned_or_zero128, clamp_limit, dst, program);
    }

    dst += dst_pitch;
    src += src_pitch;
  }
}

// 16 bit Horizontal

template void resizer_h_c_generic_uint8_16_vectorized<uint8_t, true>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resizer_h_c_generic_uint8_16_vectorized<uint16_t, false>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resizer_h_c_generic_uint8_16_vectorized<uint16_t, true>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);

/********************************************************************
***** Declare index of new filters for Avisynth's filter engine *****
********************************************************************/

extern const AVSFunction Resample_filters[] = {
  { "PointResize",    BUILTIN_FUNC_PREFIX, "cii[src_left]f[src_top]f[src_width]f[src_height]f[force]i[keep_center]b[placement]s", FilteredResize::Create_PointResize },
  { "BilinearResize", BUILTIN_FUNC_PREFIX, "cii[src_left]f[src_top]f[src_width]f[src_height]f[force]i[keep_center]b[placement]s", FilteredResize::Create_BilinearResize },
  { "BicubicResize",  BUILTIN_FUNC_PREFIX, "cii[b]f[c]f[src_left]f[src_top]f[src_width]f[src_height]f[force]i[keep_center]b[placement]s", FilteredResize::Create_BicubicResize },
  { "LanczosResize",  BUILTIN_FUNC_PREFIX, "cii[src_left]f[src_top]f[src_width]f[src_height]f[taps]i[force]i[keep_center]b[placement]s", FilteredResize::Create_LanczosResize},
  { "Lanczos4Resize", BUILTIN_FUNC_PREFIX, "cii[src_left]f[src_top]f[src_width]f[src_height]f[force]i[keep_center]b[placement]s", FilteredResize::Create_Lanczos4Resize},
  { "BlackmanResize", BUILTIN_FUNC_PREFIX, "cii[src_left]f[src_top]f[src_width]f[src_height]f[taps]i[force]i[keep_center]b[placement]s", FilteredResize::Create_BlackmanResize},
  { "Spline16Resize", BUILTIN_FUNC_PREFIX, "cii[src_left]f[src_top]f[src_width]f[src_height]f[force]i[keep_center]b[placement]s", FilteredResize::Create_Spline16Resize},
  { "Spline36Resize", BUILTIN_FUNC_PREFIX, "cii[src_left]f[src_top]f[src_width]f[src_height]f[force]i[keep_center]b[placement]s", FilteredResize::Create_Spline36Resize},
  { "Spline64Resize", BUILTIN_FUNC_PREFIX, "cii[src_left]f[src_top]f[src_width]f[src_height]f[force]i[keep_center]b[placement]s", FilteredResize::Create_Spline64Resize},
  { "GaussResize",    BUILTIN_FUNC_PREFIX, "cii[src_left]f[src_top]f[src_width]f[src_height]f[p]f[b]f[s]f[force]i[keep_center]b[placement]s", FilteredResize::Create_GaussianResize},
  { "SincResize",     BUILTIN_FUNC_PREFIX, "cii[src_left]f[src_top]f[src_width]f[src_height]f[taps]i[force]i[keep_center]b[placement]s", FilteredResize::Create_SincResize},
  { "SinPowerResize", BUILTIN_FUNC_PREFIX, "cii[src_left]f[src_top]f[src_width]f[src_height]f[p]f[force]i[keep_center]b[placement]s", FilteredResize::Create_SinPowerResize},
  { "SincLin2Resize", BUILTIN_FUNC_PREFIX, "cii[src_left]f[src_top]f[src_width]f[src_height]f[taps]i[force]i[keep_center]b[placement]s", FilteredResize::Create_SincLin2Resize},
  { "UserDefined2Resize", BUILTIN_FUNC_PREFIX, "cii[b]f[c]f[s]f[src_left]f[src_top]f[src_width]f[src_height]f[force]i[keep_center]b[placement]s", FilteredResize::Create_UserDefined2Resize},
  /**
    * Resize(PClip clip, dst_width, dst_height [src_left, src_top, src_width, int src_height,] )
    *
    * src_left et al.   =  when these optional arguments are given, the filter acts just like
    *                      a Crop was performed with those parameters before resizing, only faster
   **/

  { 0 }
};

// Borrowed from fmtconv
// ChromaPlacement.cpp
// Author : Laurent de Soras, 2015

// Fixes the vertical chroma placement when the picture is interlaced.
// ofs = ordinate to skip between TFF and BFF, relative to the chroma grid. A
// single line of full-res picture is 0.25.
static inline void ChromaPlacement_fix_itl(double& cp_v, bool interlaced_flag, bool top_flag, double ofs = 0.5)
{
  assert(cp_v >= 0);

  if (interlaced_flag)
  {
    cp_v *= 0.5;
    if (!top_flag)
    {
      cp_v += ofs;
    }
  }
}
/*
ss_h and ss_v are log2(subsampling)
rgb_flag actually means that chroma subsampling doesn't apply.

http://www.mir.com/DMG/chroma.html

cp_* is the position of the sampling point relative to the frame
top/left border, in the plane coordinates. For reference, the border
of the frame is at 0.5 units of luma from the first luma sampling point.
I. e., the luma sampling point is at the pixel's center.
*/

// PF added BOTTOM, BOTTOM_LEFT, TOP
// Pass ChromaLocation_e::AVS_CHROMA_UNUSED for defaults
// plane index 0:Y, 1:U, 2:V
// cplace is a ChromaLocation_e constant
static void ChromaPlacement_compute_cplace(double& cp_h, double& cp_v, int cplace, int plane_index, int ss_h, int ss_v, bool rgb_flag, bool interlaced_flag, bool top_flag)
{
  assert(cplace >= 0 || cplace == ChromaLocation_e::AVS_CHROMA_UNUSED);
  assert(cplace < ChromaLocation_e::AVS_CHROMA_DV);
  assert(ss_h >= 0);
  assert(ss_v >= 0);
  assert(plane_index >= 0);

  // Generic case for luma, non-subsampled chroma and center (MPEG-1) chroma.
  cp_h = 0.5;
  cp_v = 0.5;
  ChromaPlacement_fix_itl(cp_v, interlaced_flag, top_flag);

  // Subsampled chroma
  if (!rgb_flag && plane_index > 0)
  {
    if (ss_h > 0) // horizontal subsampling 420 411
    {
      if (cplace == ChromaLocation_e::AVS_CHROMA_LEFT // mpeg2
        || cplace == ChromaLocation_e::AVS_CHROMA_DV
        || cplace == ChromaLocation_e::AVS_CHROMA_TOP_LEFT
        || cplace == ChromaLocation_e::AVS_CHROMA_BOTTOM_LEFT
        )
      {
        cp_h = 0.5 / (1 << ss_h);
      }
    }

    if (ss_v == 1) // vertical subsampling 420, 422
    {
      if (cplace == ChromaLocation_e::AVS_CHROMA_LEFT)
      {
        cp_v = 0.5;
        ChromaPlacement_fix_itl(cp_v, interlaced_flag, top_flag);
      }
      else if (cplace == ChromaLocation_e::AVS_CHROMA_DV
        || cplace == ChromaLocation_e::AVS_CHROMA_TOP_LEFT
        || cplace == ChromaLocation_e::AVS_CHROMA_TOP
        )
      {
        cp_v = 0.25;
        ChromaPlacement_fix_itl(cp_v, interlaced_flag, top_flag, 0.25);

        if (cplace == ChromaLocation_e::AVS_CHROMA_DV && plane_index == 2) // V
        {
          cp_v += 0.5;
        }
      }
      else if (cplace == ChromaLocation_e::AVS_CHROMA_BOTTOM_LEFT
        || cplace == ChromaLocation_e::AVS_CHROMA_BOTTOM
        )
      {
        cp_v = 0.75;
        ChromaPlacement_fix_itl(cp_v, interlaced_flag, top_flag, 0.25);
      }
    }  // ss_v == 1
  }
}


// returns the requested horizontal or vertical pixel center position
static void GetCenterShiftForResizers(double& center_pos_luma, double& center_pos_chroma, bool preserve_center, int chroma_placement, VideoInfo &vi, bool for_horizontal) {
  double center_pos_h_luma = 0.0;
  double center_pos_v_luma = 0.0;
  // if not needed, these won't be used
  double center_pos_h_chroma = 0.0;
  double center_pos_v_chroma = 0.0;

  // chroma, only if applicable
  if (vi.IsPlanar() && vi.NumComponents() > 1 && !vi.IsRGB()) {
    double cp_s_h = 0;
    double cp_s_v = 0;

    if (preserve_center) {
      // same for source and destination
      int plane_index = 1; // U
      int src_ss_h = vi.GetPlaneWidthSubsampling(PLANAR_U);
      int src_ss_v = vi.GetPlaneHeightSubsampling(PLANAR_U);

      // int chromaplace = ChromaLocation_e::AVS_CHROMA_CENTER; // MPEG1

      ChromaPlacement_compute_cplace(
        cp_s_h, cp_s_v, chroma_placement, plane_index, src_ss_h, src_ss_v,
        vi.IsRGB(),
        false, // interlacing flag, we don't handle it here
        false  // top_flag, we don't handle it here
      );
    }

    center_pos_h_chroma = cp_s_h;
    center_pos_v_chroma = cp_s_v;
  }

  // luma/rgb planes
  if (preserve_center) {
    center_pos_h_luma = 0.5;
    center_pos_v_luma = 0.5;
  }
  else {
    center_pos_h_luma = 0.0;
    center_pos_v_luma = 0.0;
  }

  // fill return ref values
  if (for_horizontal) {
    center_pos_luma = center_pos_h_luma;
    center_pos_chroma = center_pos_h_chroma;
  }
  else {
    // vertical
    center_pos_luma = center_pos_v_luma;
    center_pos_chroma = center_pos_v_chroma;
  }

}

FilteredResizeH::FilteredResizeH(PClip _child, double subrange_left, double subrange_width,
  int target_width, ResamplingFunction* func, bool preserve_center, int chroma_placement, IScriptEnvironment* env)
  : GenericVideoFilter(_child),
  resampling_program_luma(nullptr), resampling_program_chroma(nullptr), 
  resampler_h_luma(nullptr), resampler_h_chroma(nullptr),
  resampler_luma(nullptr), resampler_chroma(nullptr)

{
  src_width = vi.width;
  src_height = vi.height;
  dst_width = target_width;
  dst_height = vi.height;

  pixelsize = vi.ComponentSize(); // AVS16
  bits_per_pixel = vi.BitsPerComponent();
  grey = vi.IsY();

  bool isRGBPfamily = vi.IsPlanarRGB() || vi.IsPlanarRGBA();

  if (target_width <= 0) {
    env->ThrowError("Resize: Width must be greater than 0.");
  }

  if (vi.IsPlanar() && !grey && !isRGBPfamily) {
    const int mask = (1 << vi.GetPlaneWidthSubsampling(PLANAR_U)) - 1;

    if (target_width & mask)
      env->ThrowError("Resize: Planar destination height must be a multiple of %d.", mask + 1);
  }

  double center_pos_h_luma;
  double center_pos_h_chroma;
  GetCenterShiftForResizers(center_pos_h_luma, center_pos_h_chroma, preserve_center, chroma_placement, vi, true /* for horizontal */);
  // 3.7.4- parameter, old Avisynth behavior: 0.5, 0.5

  // Main resampling program
  resampling_program_luma = func->GetResamplingProgram(vi.width, subrange_left, subrange_width, target_width, bits_per_pixel, 
    center_pos_h_luma, center_pos_h_luma, // for resizing it's the same for source and dest
    env);
  if (vi.IsPlanar() && !grey && !isRGBPfamily) {
    const int shift = vi.GetPlaneWidthSubsampling(PLANAR_U);
    const int div = 1 << shift;


    resampling_program_chroma = func->GetResamplingProgram(
      vi.width >> shift,
      subrange_left / div,
      subrange_width / div,
      target_width >> shift,
      bits_per_pixel,
      center_pos_h_chroma, center_pos_h_chroma, // horizontal
      env);
  }

// when not fast_resize, then we use vertical resizers between turnleft/turnright
#ifdef INTEL_INTRINSICS
  int cpu = env->GetCPUFlags();
  bool has_sse2 = (cpu & CPUF_SSE2) != 0;
#else
  int cpu = 0;
#endif

  fast_resize = vi.IsPlanar();
  // PF 2025: H is not slower than V in C implementation.
  // Still, H resizers are incompatible with packed RGB formats

    if (!fast_resize) {

      // nonfast-resize: using V resizer for horizontal resizing between a turnleft/right

      resampler_luma = FilteredResizeV::GetResampler(cpu, pixelsize, bits_per_pixel, resampling_program_luma, env);

      if (vi.IsPlanar() && !grey && !isRGBPfamily) {
        resampler_chroma = FilteredResizeV::GetResampler(cpu, pixelsize, bits_per_pixel, resampling_program_chroma, env);
      }

      // Temporary buffer size for turns
      temp_1_pitch = AlignNumber(vi.BytesFromPixels(src_height), FRAME_ALIGN);
      temp_2_pitch = AlignNumber(vi.BytesFromPixels(dst_height), FRAME_ALIGN);

      // Initialize Turn function
      // see turn.cpp
      if (vi.IsRGB24()) {
#ifdef INTEL_INTRINSICS
        // no intel intentionally
#endif
        turn_left = turn_left_rgb24;
        turn_right = turn_right_rgb24;
      }
      else if (vi.IsRGB32()) {
#ifdef INTEL_INTRINSICS
        if (has_sse2) {
          turn_left = turn_left_rgb32_sse2;
          turn_right = turn_right_rgb32_sse2;
        }
        else
#endif
        {
          turn_left = turn_left_rgb32_c;
          turn_right = turn_right_rgb32_c;
        }
      }
      else if (vi.IsRGB48()) {
#ifdef INTEL_INTRINSICS
        // no intel intentionally
#endif
        turn_left = turn_left_rgb48_c;
        turn_right = turn_right_rgb48_c;
      }
      else if (vi.IsRGB64()) {
#ifdef INTEL_INTRINSICS
        if (has_sse2) {
          turn_left = turn_left_rgb64_sse2;
          turn_right = turn_right_rgb64_sse2;
        }
        else

#endif
        {
          turn_left = turn_left_rgb64_c;
          turn_right = turn_right_rgb64_c;
        }
      }
      else {
        switch (vi.ComponentSize()) {// AVS16
        case 1: // 8 bit
#ifdef INTEL_INTRINSICS
          if (has_sse2) {
            turn_left = turn_left_plane_8_sse2;
            turn_right = turn_right_plane_8_sse2;
          }
          else
#endif
          {
            turn_left = turn_left_plane_8_c;
            turn_right = turn_right_plane_8_c;
          }
          break;
        case 2: // 16 bit
#ifdef INTEL_INTRINSICS
          if (has_sse2) {
            turn_left = turn_left_plane_16_sse2;
            turn_right = turn_right_plane_16_sse2;
          }
          else
#endif
          {
            turn_left = turn_left_plane_16_c;
            turn_right = turn_right_plane_16_c;
          }
          break;
        default: // 32 bit
#ifdef INTEL_INTRINSICS
          if (has_sse2) {
            turn_left = turn_left_plane_32_sse2;
            turn_right = turn_right_plane_32_sse2;
          }
          else
#endif
          {
            turn_left = turn_left_plane_32_c;
            turn_right = turn_right_plane_32_c;
          }
        }
      }
    }
    else {
      // planar format (or Y)
      resampler_h_luma = GetResampler(cpu, pixelsize, bits_per_pixel, resampling_program_luma, env);

      if (!grey && !isRGBPfamily) {
        resampler_h_chroma = GetResampler(cpu, pixelsize, bits_per_pixel, resampling_program_chroma, env);
      }
    }
  // Change target video info size
  vi.width = target_width;
}

PVideoFrame __stdcall FilteredResizeH::GetFrame(int n, IScriptEnvironment* env)
{
  PVideoFrame src = child->GetFrame(n, env);
  PVideoFrame dst = env->NewVideoFrameP(vi, &src);

  bool isRGBPfamily = vi.IsPlanarRGB() || vi.IsPlanarRGBA();

  if (!fast_resize) {
    // e.g. not aligned, not mod4
    // temp_1_pitch and temp_2_pitch is pixelsize-aware
    BYTE* temp_1 = static_cast<BYTE*>(env->Allocate(temp_1_pitch * src_width, FRAME_ALIGN, AVS_POOLED_ALLOC));
    BYTE* temp_2 = static_cast<BYTE*>(env->Allocate(temp_2_pitch * dst_width, FRAME_ALIGN, AVS_POOLED_ALLOC));
    if (!temp_1 || !temp_2) {
      env->Free(temp_1);
      env->Free(temp_2);
      env->ThrowError("Could not reserve memory in a resampler.");
    }

    if (!vi.IsRGB() || isRGBPfamily) {
      // Y/G Plane
      turn_right(src->GetReadPtr(), temp_1, src_width * pixelsize, src_height, src->GetPitch(), temp_1_pitch); // * pixelsize: turn_right needs GetPlaneWidth full size
      resampler_luma(temp_2, temp_1, temp_2_pitch, temp_1_pitch, resampling_program_luma, src_height, dst_width, bits_per_pixel);
      turn_left(temp_2, dst->GetWritePtr(), dst_height * pixelsize, dst_width, temp_2_pitch, dst->GetPitch());

      if (isRGBPfamily)
      {
        turn_right(src->GetReadPtr(PLANAR_B), temp_1, src_width * pixelsize, src_height, src->GetPitch(PLANAR_B), temp_1_pitch); // * pixelsize: turn_right needs GetPlaneWidth full size
        resampler_luma(temp_2, temp_1, temp_2_pitch, temp_1_pitch, resampling_program_luma, src_height, dst_width, bits_per_pixel);
        turn_left(temp_2, dst->GetWritePtr(PLANAR_B), dst_height * pixelsize, dst_width, temp_2_pitch, dst->GetPitch(PLANAR_B));

        turn_right(src->GetReadPtr(PLANAR_R), temp_1, src_width * pixelsize, src_height, src->GetPitch(PLANAR_R), temp_1_pitch); // * pixelsize: turn_right needs GetPlaneWidth full size
        resampler_luma(temp_2, temp_1, temp_2_pitch, temp_1_pitch, resampling_program_luma, src_height, dst_width, bits_per_pixel);
        turn_left(temp_2, dst->GetWritePtr(PLANAR_R), dst_height * pixelsize, dst_width, temp_2_pitch, dst->GetPitch(PLANAR_R));
      }
      else if (!grey) {
        const int shift = vi.GetPlaneWidthSubsampling(PLANAR_U);
        const int shift_h = vi.GetPlaneHeightSubsampling(PLANAR_U);

        const int src_chroma_width = src_width >> shift;
        const int dst_chroma_width = dst_width >> shift;
        const int src_chroma_height = src_height >> shift_h;
        const int dst_chroma_height = dst_height >> shift_h;

        // turn_xxx: width * pixelsize: needs GetPlaneWidth-like full size
        // U Plane
        turn_right(src->GetReadPtr(PLANAR_U), temp_1, src_chroma_width * pixelsize, src_chroma_height, src->GetPitch(PLANAR_U), temp_1_pitch);
        resampler_luma(temp_2, temp_1, temp_2_pitch, temp_1_pitch, resampling_program_chroma, src_chroma_height, dst_chroma_width, bits_per_pixel);
        turn_left(temp_2, dst->GetWritePtr(PLANAR_U), dst_chroma_height * pixelsize, dst_chroma_width, temp_2_pitch, dst->GetPitch(PLANAR_U));

        // V Plane
        turn_right(src->GetReadPtr(PLANAR_V), temp_1, src_chroma_width * pixelsize, src_chroma_height, src->GetPitch(PLANAR_V), temp_1_pitch);
        resampler_luma(temp_2, temp_1, temp_2_pitch, temp_1_pitch, resampling_program_chroma, src_chroma_height, dst_chroma_width, bits_per_pixel);
        turn_left(temp_2, dst->GetWritePtr(PLANAR_V), dst_chroma_height * pixelsize, dst_chroma_width, temp_2_pitch, dst->GetPitch(PLANAR_V));
      }
      if (vi.IsYUVA() || vi.IsPlanarRGBA())
      {
        turn_right(src->GetReadPtr(PLANAR_A), temp_1, src_width * pixelsize, src_height, src->GetPitch(PLANAR_A), temp_1_pitch); // * pixelsize: turn_right needs GetPlaneWidth full size
        resampler_luma(temp_2, temp_1, temp_2_pitch, temp_1_pitch, resampling_program_luma, src_height, dst_width, bits_per_pixel);
        turn_left(temp_2, dst->GetWritePtr(PLANAR_A), dst_height * pixelsize, dst_width, temp_2_pitch, dst->GetPitch(PLANAR_A));
      }

    }
    else {
      // packed RGB
      // First left, then right. Reason: packed RGB bottom to top. Right+left shifts RGB24/RGB32 image to the opposite horizontal direction
      turn_left(src->GetReadPtr(), temp_1, vi.BytesFromPixels(src_width), src_height, src->GetPitch(), temp_1_pitch);
      resampler_luma(temp_2, temp_1, temp_2_pitch, temp_1_pitch, resampling_program_luma, vi.BytesFromPixels(src_height) / pixelsize, dst_width, bits_per_pixel);
      turn_right(temp_2, dst->GetWritePtr(), vi.BytesFromPixels(dst_height), dst_width, temp_2_pitch, dst->GetPitch());
    }

    env->Free(temp_1);
    env->Free(temp_2);
  }
  else {

    // Y Plane
    resampler_h_luma(dst->GetWritePtr(), src->GetReadPtr(), dst->GetPitch(), src->GetPitch(), resampling_program_luma, dst_width, dst_height, bits_per_pixel);

    if (isRGBPfamily) {
      resampler_h_luma(dst->GetWritePtr(PLANAR_B), src->GetReadPtr(PLANAR_B), dst->GetPitch(PLANAR_B), src->GetPitch(PLANAR_B), resampling_program_luma, dst_width, dst_height, bits_per_pixel);
      resampler_h_luma(dst->GetWritePtr(PLANAR_R), src->GetReadPtr(PLANAR_R), dst->GetPitch(PLANAR_R), src->GetPitch(PLANAR_R), resampling_program_luma, dst_width, dst_height, bits_per_pixel);
    }
    else if (!grey) {
      const int dst_chroma_width = dst_width >> vi.GetPlaneWidthSubsampling(PLANAR_U);
      const int dst_chroma_height = dst_height >> vi.GetPlaneHeightSubsampling(PLANAR_U);

      // U Plane
      resampler_h_chroma(dst->GetWritePtr(PLANAR_U), src->GetReadPtr(PLANAR_U), dst->GetPitch(PLANAR_U), src->GetPitch(PLANAR_U), resampling_program_chroma, dst_chroma_width, dst_chroma_height, bits_per_pixel);

      // V Plane
      resampler_h_chroma(dst->GetWritePtr(PLANAR_V), src->GetReadPtr(PLANAR_V), dst->GetPitch(PLANAR_V), src->GetPitch(PLANAR_V), resampling_program_chroma, dst_chroma_width, dst_chroma_height, bits_per_pixel);
    }
    if (vi.IsYUVA() || vi.IsPlanarRGBA())
    {
      resampler_h_luma(dst->GetWritePtr(PLANAR_A), src->GetReadPtr(PLANAR_A), dst->GetPitch(PLANAR_A), src->GetPitch(PLANAR_A), resampling_program_luma, dst_width, dst_height, bits_per_pixel);
    }

  }

  return dst;
}

ResamplerH FilteredResizeH::GetResampler(int CPU, int pixelsize, int bits_per_pixel, ResamplingProgram* program, IScriptEnvironment* env)
{
  int simd_coeff_count_padding = 8;

  // Both 8-bit and 16-bit SSSE3 and AVX2 horizontal resizers benefit from processing 16 pixels per cycle.
  // Floats also use 32 bytes, but since 32/sizeof(float) = 8, processing 16 pixels is unnecessary.
  // Even in C, the code is optimized to be vector-friendly.
  if (pixelsize == 1 || pixelsize == 2)
    simd_coeff_count_padding = 16;

  // Not only does it prepare and pad for SIMD/vector code, but it also corrects, reorders, and equalizes coefficients 
  // at the right and bottom ends, since we may have variable kernel sizes due to boundary conditions.
  resize_prepare_coeffs(program, env, simd_coeff_count_padding);

  if (pixelsize == 1)
  {
#ifdef INTEL_INTRINSICS
    if (CPU & CPUF_AVX2) {
      return resizer_h_avx2_generic_uint8_t;
    }
    if (CPU & CPUF_SSSE3) {
      return resizer_h_ssse3_generic_uint8_16<uint8_t, true>;
    }
#endif
    return resizer_h_c_generic_uint8_16_vectorized<uint8_t, true>;
    //return resize_h_c_planar<uint8_t, 1>;
  }
  else if (pixelsize == 2) {
#ifdef INTEL_INTRINSICS
    if (CPU & CPUF_AVX2) {
      if (bits_per_pixel < 16)
        return resizer_h_avx2_generic_uint16_t<true>;
      else
        return resizer_h_avx2_generic_uint16_t<false>;
    }
    if (CPU & CPUF_SSSE3) {
      if (bits_per_pixel < 16)
        return resizer_h_ssse3_generic_uint8_16<uint16_t, true>;
      else
        return resizer_h_ssse3_generic_uint8_16<uint16_t, false>;
    }
#endif
    if (bits_per_pixel == 16)
      return resizer_h_c_generic_uint8_16_vectorized<uint16_t, false>;
      // return resize_h_c_planar<uint16_t, 0>;
    else
      return resizer_h_c_generic_uint8_16_vectorized<uint16_t, true>;
      // return resize_h_c_planar<uint16_t, 1>;
  }
  else { //if (pixelsize == 4)
#ifdef INTEL_INTRINSICS
#ifdef INTEL_INTRINSICS_AVX512
    if ((CPU & CPUF_AVX512F) && program->filter_size_real <= 4) {
      //return resize_h_planar_float_avx2_permutex_vstripe_ks4;
      switch (program->filter_size_real) {
/*      case 1: return resize_h_planar_float_avx512_transpose_vstripe_ks4<1>; break;
      case 2: return resize_h_planar_float_avx512_transpose_vstripe_ks4<2>; break;
      case 3: return resize_h_planar_float_avx512_transpose_vstripe_ks4<3>; break;
      case 4: return resize_h_planar_float_avx512_transpose_vstripe_ks4<0>; break;*/
      case 1: return resize_h_planar_float_avx512_gather_permutex_vstripe_ks4<1>; break;
      case 2: return resize_h_planar_float_avx512_gather_permutex_vstripe_ks4<2>; break;
      case 3: return resize_h_planar_float_avx512_gather_permutex_vstripe_ks4<3>; break;
      case 4: return resize_h_planar_float_avx512_gather_permutex_vstripe_ks4<0>; break;
      }
    }
#endif
    if (CPU & CPUF_AVX2) {
      //return resize_h_planar_float_avx2_permutex_vstripe_ks4;
      
      switch (program->filter_size_real) {
/*      case 1: return resize_h_planar_float_avx_transpose_vstripe_ks4<1>; break;
      case 2: return resize_h_planar_float_avx_transpose_vstripe_ks4<2>; break;
      case 3: return resize_h_planar_float_avx_transpose_vstripe_ks4<3>; break;
      case 4: return resize_h_planar_float_avx_transpose_vstripe_ks4<0>; break;*/
      case 1: return resize_h_planar_float_avx2_gather_permutex_vstripe_ks4<1>; break;
      case 2: return resize_h_planar_float_avx2_gather_permutex_vstripe_ks4<2>; break;
      case 3: return resize_h_planar_float_avx2_gather_permutex_vstripe_ks4<3>; break;
      case 4: return resize_h_planar_float_avx2_gather_permutex_vstripe_ks4<0>; break;
      default: return resizer_h_avx2_generic_float;
      }
      
    }
    if (CPU & CPUF_SSSE3) {
      //      return resizer_h_ssse3_generic_float;
      switch (program->filter_size_real) {
      case 1: return resize_h_planar_float_sse_transpose_vstripe_ks4<1>; break;
      case 2: return resize_h_planar_float_sse_transpose_vstripe_ks4<2>; break;
      case 3: return resize_h_planar_float_sse_transpose_vstripe_ks4<3>; break;
      case 4: return resize_h_planar_float_sse_transpose_vstripe_ks4<0>; break;
      default: return resizer_h_ssse3_generic_float;
      }
    }
#endif
    return resize_h_c_planar<float, 0>;
  }
}

FilteredResizeH::~FilteredResizeH(void)
{
  if (resampling_program_luma) { delete resampling_program_luma; }
  if (resampling_program_chroma) { delete resampling_program_chroma; }
}

/***************************************
 ***** Filtered Resize - Vertical ******
 ***************************************/

FilteredResizeV::FilteredResizeV(PClip _child, double subrange_top, double subrange_height,
  int target_height, ResamplingFunction* func, 
  bool preserve_center, int chroma_placement,
  IScriptEnvironment* env)
  : GenericVideoFilter(_child),
  resampling_program_luma(0), resampling_program_chroma(0)
{
  if (target_height <= 0)
    env->ThrowError("Resize: Height must be greater than 0.");

  pixelsize = vi.ComponentSize(); // AVS16
  bits_per_pixel = vi.BitsPerComponent();
  grey = vi.IsY();
  bool isRGBPfamily = vi.IsPlanarRGB() || vi.IsPlanarRGBA();

  if (vi.IsPlanar() && !grey && !isRGBPfamily) {
    const int mask = (1 << vi.GetPlaneHeightSubsampling(PLANAR_U)) - 1;

    if (target_height & mask)
      env->ThrowError("Resize: Planar destination height must be a multiple of %d.", mask + 1);
  }

  if (vi.IsRGB() && !isRGBPfamily)
    subrange_top = vi.height - subrange_top - subrange_height; // packed RGB upside down

#ifdef INTEL_INTRINSICS
  int cpu = env->GetCPUFlags();
#else
  int cpu = 0;
#endif

  double center_pos_v_luma;
  double center_pos_v_chroma;
  GetCenterShiftForResizers(center_pos_v_luma, center_pos_v_chroma, preserve_center, chroma_placement, vi, false /* for vertical */);
  // 3.7.4- parameter, old Avisynth behavior: 0.5, 0.5

  // Create resampling program and pitch table
  resampling_program_luma = func->GetResamplingProgram(vi.height, subrange_top, subrange_height, target_height, bits_per_pixel, 
    center_pos_v_luma, center_pos_v_luma, // for resizing it's the same for source and dest
    env);
  resampler_luma = GetResampler(cpu, pixelsize, bits_per_pixel, resampling_program_luma, env);

  if (vi.IsPlanar() && !grey && !isRGBPfamily) {
    const int shift = vi.GetPlaneHeightSubsampling(PLANAR_U);
    const int div = 1 << shift;

    resampling_program_chroma = func->GetResamplingProgram(
      vi.height >> shift,
      subrange_top / div,
      subrange_height / div,
      target_height >> shift,
      bits_per_pixel,
      center_pos_v_chroma, center_pos_v_chroma, // for resizing it's the same for source and dest
      env);

    resampler_chroma = GetResampler(cpu, pixelsize, bits_per_pixel, resampling_program_chroma, env);
  }

  // Change target video info size
  vi.height = target_height;
}

PVideoFrame __stdcall FilteredResizeV::GetFrame(int n, IScriptEnvironment* env)
{
  PVideoFrame src = child->GetFrame(n, env);
  PVideoFrame dst = env->NewVideoFrameP(vi, &src);
  int src_pitch = src->GetPitch();
  int dst_pitch = dst->GetPitch();
  const BYTE* srcp = src->GetReadPtr();
  BYTE* dstp = dst->GetWritePtr();

  bool isRGBPfamily = vi.IsPlanarRGB() || vi.IsPlanarRGBA();

  // Do resizing
  int work_width = vi.IsPlanar() ? vi.width : vi.BytesFromPixels(vi.width) / pixelsize; // packed RGB: or vi.width * vi.NumComponent()
  resampler_luma(dstp, srcp, dst_pitch, src_pitch, resampling_program_luma, work_width, vi.height, bits_per_pixel);
  if (isRGBPfamily)
  {
    src_pitch = src->GetPitch(PLANAR_B);
    dst_pitch = dst->GetPitch(PLANAR_B);
    srcp = src->GetReadPtr(PLANAR_B);
    dstp = dst->GetWritePtr(PLANAR_B);
    
    resampler_luma(dstp, srcp, dst_pitch, src_pitch, resampling_program_luma, work_width, vi.height, bits_per_pixel);
    
    src_pitch = src->GetPitch(PLANAR_R);
    dst_pitch = dst->GetPitch(PLANAR_R);
    srcp = src->GetReadPtr(PLANAR_R);
    dstp = dst->GetWritePtr(PLANAR_R);

    resampler_luma(dstp, srcp, dst_pitch, src_pitch, resampling_program_luma, work_width, vi.height, bits_per_pixel);
  }
  else if (!grey && vi.IsPlanar()) {
    int width = vi.width >> vi.GetPlaneWidthSubsampling(PLANAR_U);
    int height = vi.height >> vi.GetPlaneHeightSubsampling(PLANAR_U);

    // Plane U resizing
    src_pitch = src->GetPitch(PLANAR_U);
    dst_pitch = dst->GetPitch(PLANAR_U);
    srcp = src->GetReadPtr(PLANAR_U);
    dstp = dst->GetWritePtr(PLANAR_U);

    resampler_chroma(dstp, srcp, dst_pitch, src_pitch, resampling_program_chroma, width, height, bits_per_pixel);

    // Plane V resizing
    src_pitch = src->GetPitch(PLANAR_V);
    dst_pitch = dst->GetPitch(PLANAR_V);
    srcp = src->GetReadPtr(PLANAR_V);
    dstp = dst->GetWritePtr(PLANAR_V);

    resampler_chroma(dstp, srcp, dst_pitch, src_pitch, resampling_program_chroma, width, height, bits_per_pixel);
  }

  if (vi.IsYUVA() || vi.IsPlanarRGBA()) {
    src_pitch = src->GetPitch(PLANAR_A);
    dst_pitch = dst->GetPitch(PLANAR_A);
    srcp = src->GetReadPtr(PLANAR_A);
    dstp = dst->GetWritePtr(PLANAR_A);
    resampler_luma(dstp, srcp, dst_pitch, src_pitch, resampling_program_luma, work_width, vi.height, bits_per_pixel);
  }

  return dst;
}

ResamplerV FilteredResizeV::GetResampler(int CPU, int pixelsize, int bits_per_pixel, ResamplingProgram* program, IScriptEnvironment* env)
{

  resize_prepare_coeffs(program, env, 8); 
  // for SIMD friendliness and more: consolidate the kernel_size vs filter_size at the end.
  // See comments at FilteredResizeH::GetResampler

  if (program->filter_size == 1) {
    // Fast pointresize
    switch (pixelsize) // AVS16
    {
    case 1: return resize_v_planar_pointresize<uint8_t>;
    case 2: return resize_v_planar_pointresize<uint16_t>;
    default: // case 4:
      return resize_v_planar_pointresize<float>;
    }
  }
  else {
    // Other resizers
    if (pixelsize == 1)
    {
#ifdef INTEL_INTRINSICS
      if (CPU & CPUF_AVX2)
        return resize_v_avx2_planar_uint8_t;
      if (CPU & CPUF_SSE2)
        return resize_v_sse2_planar;
#ifdef X86_32
      if (CPU & CPUF_MMX)
        return resize_v_mmx_planar;
#endif
#endif
      // C version
      return resize_v_c_planar_uint8_16_t_auto_vectorized<uint8_t, true>;
    }
    else if (pixelsize == 2)
    {
#ifdef INTEL_INTRINSICS
      if (CPU & CPUF_AVX2) {
        if (bits_per_pixel < 16)
          return resize_v_avx2_planar_uint16_t<true>;
        else
          return resize_v_avx2_planar_uint16_t<false>;
      }
      if (CPU & CPUF_SSE2) {
        if (bits_per_pixel < 16)
          return resize_v_sse2_planar_uint16_t<true>;
        else
          return resize_v_sse2_planar_uint16_t<false>;
      }
#endif
      // C version
      if(bits_per_pixel == 16)
        return resize_v_c_planar_uint8_16_t_auto_vectorized<uint16_t, false>;
      else
        return resize_v_c_planar_uint8_16_t_auto_vectorized<uint16_t, true>;
    }
    else // pixelsize== 4
    {
#ifdef INTEL_INTRINSICS
      if (CPU & CPUF_AVX2) {
        return resize_v_avx2_planar_float;
      }
      if (CPU & CPUF_SSE2) {
        return resize_v_sse2_planar_float;
      }
#endif
      return resize_v_c_planar_float_auto_vectorized;
    }
  }
}

FilteredResizeV::~FilteredResizeV(void)
{
  if (resampling_program_luma) { delete resampling_program_luma; }
  if (resampling_program_chroma) { delete resampling_program_chroma; }
}


/**********************************************
 *******   Resampling Factory Methods   *******
 **********************************************/

PClip FilteredResize::CreateResizeH(PClip clip, double subrange_left, double subrange_width, int target_width, bool force,
  ResamplingFunction* func, bool preserve_center, int chroma_placement, IScriptEnvironment* env)
{
  const VideoInfo& vi = clip->GetVideoInfo();
  if (!force && subrange_left == 0 && subrange_width == target_width && subrange_width == vi.width) {
    return clip;
  }
  /*
  // intentionally left here: don't use crop at special edge cases to avoid inconsistent results across params/color spaces
  if (subrange_left == int(subrange_left) && subrange_width == target_width
   && subrange_left >= 0 && subrange_left + subrange_width <= vi.width) {
    const int mask = ((vi.IsYUV() || vi.IsYUVA()) && !vi.IsY()) ? (1 << vi.GetPlaneWidthSubsampling(PLANAR_U)) - 1 : 0;

    if (((int(subrange_left) | int(subrange_width)) & mask) == 0)
      return new Crop(int(subrange_left), 0, int(subrange_width), vi.height, 0, clip, env);
  }
  */
  // Convert interleaved yuv to planar yuv
  PClip result = clip;
  if (vi.IsYUY2()) {
    result = new ConvertYUY2ToYV16(result, env);
  }

  result = new FilteredResizeH(result, subrange_left, subrange_width, target_width, func, preserve_center, chroma_placement, env);
  
  if (vi.IsYUY2()) {
    result = new ConvertYV16ToYUY2(result, env);
  }

  return result;
}


PClip FilteredResize::CreateResizeV(PClip clip, double subrange_top, double subrange_height, int target_height, bool force,
  ResamplingFunction* func, bool preserve_center, int chroma_placement, IScriptEnvironment* env)
{
  const VideoInfo& vi = clip->GetVideoInfo();
  if (!force && subrange_top == 0 && subrange_height == target_height && subrange_height == vi.height) {
    return clip;
  }
  /*
  // intentionally left here: don't use crop at special edge cases to avoid inconsistent results across params/color spaces
  if (subrange_top == int(subrange_top) && subrange_height == target_height
   && subrange_top >= 0 && subrange_top + subrange_height <= vi.height) {
    const int mask = ((vi.IsYUV() || vi.IsYUVA()) && !vi.IsY()) ? (1 << vi.GetPlaneHeightSubsampling(PLANAR_U)) - 1 : 0;

    if (((int(subrange_top) | int(subrange_height)) & mask) == 0)
      return new Crop(0, int(subrange_top), vi.width, int(subrange_height), 0, clip, env);
  }
  */
  return new FilteredResizeV(clip, subrange_top, subrange_height, target_height, func, preserve_center, chroma_placement, env);
}


PClip FilteredResize::CreateResize(PClip clip, int target_width, int target_height, const AVSValue* args, int force,
  ResamplingFunction* f, 
  bool preserve_center, const char* placement_name, const int forced_chroma_placement,
  IScriptEnvironment* env)
{
  // args 0-1-2-3: left-top-width-height
  VideoInfo vi = clip->GetVideoInfo();
  const double subrange_left = args[0].AsFloat(0), subrange_top = args[1].AsFloat(0);

  double subrange_width = args[2].AsDblDef(vi.width), subrange_height = args[3].AsDblDef(vi.height);
  // Crop style syntax
  if (subrange_width <= 0.0) subrange_width = vi.width - subrange_left + subrange_width;
  if (subrange_height <= 0.0) subrange_height = vi.height - subrange_top + subrange_height;

  PClip result;
  // ensure that the intermediate area is maximal

  const double area_FirstH = subrange_height * target_width;
  const double area_FirstV = subrange_width * target_height;

  // "minimal area" logic is not necessarily faster because H and V resizers are not the same speed.
  // so we keep the traditional max area logic, which is for quality

  // use forced_chroma_placement >= 0 and placement_name == nullptr together
  int chroma_placement = forced_chroma_placement >= 0 ? forced_chroma_placement : ChromaLocation_e::AVS_CHROMA_UNUSED;
  if (placement_name) {
    // no format-oriented defaults
    if (vi.IsYV411() || vi.Is420() || vi.Is422()) {
      // placement explicite parameter like in ConvertToXXX or Text
      // input frame properties, if "auto"
      // When called from ConvertToXXX, chroma is not involved.
      auto frame0 = clip->GetFrame(0, env);
      const AVSMap* props = env->getFramePropsRO(frame0);
      chromaloc_parse_merge_with_props(vi, placement_name, props, /* ref*/chroma_placement, ChromaLocation_e::AVS_CHROMA_UNUSED /*default*/, env);
    }

  }

  // 0 - return unchanged if no resize needed
  // 1 - force H
  // 2 - force V
  // 3 - force H and V
  const bool force_H = force == 1 || force == 3;
  const bool force_V = force == 2 || force == 3;
  if (area_FirstH < area_FirstV)
  {
    result = CreateResizeV(clip, subrange_top, subrange_height, target_height, force_V, f, preserve_center, chroma_placement, env);
    result = CreateResizeH(result, subrange_left, subrange_width, target_width, force_H, f, preserve_center, chroma_placement, env);
  }
  else
  {
    result = CreateResizeH(clip, subrange_left, subrange_width, target_width, force_H, f, preserve_center, chroma_placement, env);
    result = CreateResizeV(result, subrange_top, subrange_height, target_height, force_V, f, preserve_center, chroma_placement, env);
  }
  return result;
}

AVSValue __cdecl FilteredResize::Create_PointResize(AVSValue args, void*, IScriptEnvironment* env)
{
  auto f = PointFilter();
  const int force = args[7].AsInt(0);

  bool preserve_center = args[8].AsBool(true); // [keep_center] default Avisynth
  const char* placement_name = args[9].AsString("auto"); // [placement]s
  const int forced_chroma_placement = -1; // no force, used internally

  return CreateResize(args[0].AsClip(), args[1].AsInt(), args[2].AsInt(), &args[3], force, &f, preserve_center, placement_name, forced_chroma_placement, env);
}


AVSValue __cdecl FilteredResize::Create_BilinearResize(AVSValue args, void*, IScriptEnvironment* env)
{
  auto f = TriangleFilter();
  const int force = args[7].AsInt(0);

  bool preserve_center = args[8].AsBool(true); // [keep_center] default Avisynth
  const char* placement_name = args[9].AsString("auto"); // [placement]s
  const int forced_chroma_placement = -1; // no force, used internally

  return CreateResize(args[0].AsClip(), args[1].AsInt(), args[2].AsInt(), &args[3], force, &f, preserve_center, placement_name, forced_chroma_placement, env);
}


AVSValue __cdecl FilteredResize::Create_BicubicResize(AVSValue args, void*, IScriptEnvironment* env)
{
  auto f = MitchellNetravaliFilter(args[3].AsDblDef(1. / 3.), args[4].AsDblDef(1. / 3.));
  const int force = args[9].AsInt(0);

  bool preserve_center = args[10].AsBool(true); // [keep_center] default Avisynth
  const char* placement_name = args[11].AsString("auto"); // [placement]s
  const int forced_chroma_placement = -1; // no force, used internally

  return CreateResize(args[0].AsClip(), args[1].AsInt(), args[2].AsInt(), &args[5], force, &f, preserve_center, placement_name, forced_chroma_placement, env);
}

AVSValue __cdecl FilteredResize::Create_LanczosResize(AVSValue args, void*, IScriptEnvironment* env)
{
  auto f = LanczosFilter(args[7].AsInt(3));
  const int force = args[8].AsInt(0);

  bool preserve_center = args[9].AsBool(true); // [keep_center] default Avisynth
  const char* placement_name = args[10].AsString("auto"); // [placement]s
  const int forced_chroma_placement = -1; // no force, used internally

  return CreateResize(args[0].AsClip(), args[1].AsInt(), args[2].AsInt(), &args[3], force, &f, preserve_center, placement_name, forced_chroma_placement, env);
}

AVSValue __cdecl FilteredResize::Create_Lanczos4Resize(AVSValue args, void*, IScriptEnvironment* env)
{
  auto f = LanczosFilter(4);
  const int force = args[7].AsInt(0);

  bool preserve_center = args[8].AsBool(true); // [keep_center] default Avisynth
  const char* placement_name = args[9].AsString("auto"); // [placement]s
  const int forced_chroma_placement = -1; // no force, used internally

  return CreateResize(args[0].AsClip(), args[1].AsInt(), args[2].AsInt(), &args[3], force, &f, preserve_center, placement_name, forced_chroma_placement, env);
}

AVSValue __cdecl FilteredResize::Create_BlackmanResize(AVSValue args, void*, IScriptEnvironment* env)
{
  auto f = BlackmanFilter(args[7].AsInt(4));
  const int force = args[8].AsInt(0);

  bool preserve_center = args[8].AsBool(true); // [keep_center] default Avisynth
  const char* placement_name = args[9].AsString("auto"); // [placement]s
  const int forced_chroma_placement = -1; // no force, used internally

  return CreateResize(args[0].AsClip(), args[1].AsInt(), args[2].AsInt(), &args[3], force, &f, preserve_center, placement_name, forced_chroma_placement, env);
}

AVSValue __cdecl FilteredResize::Create_Spline16Resize(AVSValue args, void*, IScriptEnvironment* env)
{
  auto f = Spline16Filter();
  const int force = args[7].AsInt(0);

  bool preserve_center = args[8].AsBool(true); // [keep_center] default Avisynth
  const char* placement_name = args[9].AsString("auto"); // [placement]s
  const int forced_chroma_placement = -1; // no force, used internally

  return CreateResize(args[0].AsClip(), args[1].AsInt(), args[2].AsInt(), &args[3], force, &f, preserve_center, placement_name, forced_chroma_placement, env);
}

AVSValue __cdecl FilteredResize::Create_Spline36Resize(AVSValue args, void*, IScriptEnvironment* env)
{
  auto f = Spline36Filter();
  const int force = args[7].AsInt(0);

  bool preserve_center = args[8].AsBool(true); // [keep_center] default Avisynth
  const char* placement_name = args[9].AsString("auto"); // [placement]s
  const int forced_chroma_placement = -1; // no force, used internally

  return CreateResize(args[0].AsClip(), args[1].AsInt(), args[2].AsInt(), &args[3], force, &f, preserve_center, placement_name, forced_chroma_placement, env);
}

AVSValue __cdecl FilteredResize::Create_Spline64Resize(AVSValue args, void*, IScriptEnvironment* env)
{
  auto f = Spline64Filter();
  const int force = args[7].AsInt(0);

  bool preserve_center = args[8].AsBool(true); // [keep_center] default Avisynth
  const char* placement_name = args[9].AsString("auto"); // [placement]s
  const int forced_chroma_placement = -1; // no force, used internally

  return CreateResize(args[0].AsClip(), args[1].AsInt(), args[2].AsInt(), &args[3], force, &f, preserve_center, placement_name, forced_chroma_placement, env);
}

AVSValue __cdecl FilteredResize::Create_GaussianResize(AVSValue args, void*, IScriptEnvironment* env)
{
  auto f = GaussianFilter(args[7].AsFloat(30.0f), args[8].AsFloat(2.0f), args[9].AsFloat(4.0f)); // defaults at two more places
  const int force = args[10].AsInt(0);

  bool preserve_center = args[11].AsBool(true); // [keep_center] default Avisynth
  const char* placement_name = args[12].AsString("auto"); // [placement]s
  const int forced_chroma_placement = -1; // no force, used internally

  return CreateResize(args[0].AsClip(), args[1].AsInt(), args[2].AsInt(), &args[3], force, &f, preserve_center, placement_name, forced_chroma_placement, env);
}

AVSValue __cdecl FilteredResize::Create_SincResize(AVSValue args, void*, IScriptEnvironment* env)
{
  auto f = SincFilter(args[7].AsInt(4));
  const int force = args[8].AsInt(0);
 
  bool preserve_center = args[9].AsBool(true); // [keep_center] default Avisynth
  const char * placement_name = args[10].AsString("auto"); // [placement]s
  const int forced_chroma_placement = -1; // no force, used internally

  return CreateResize(args[0].AsClip(), args[1].AsInt(), args[2].AsInt(), &args[3], force, &f, preserve_center, placement_name, forced_chroma_placement, env);
}

// like GaussianFilter(); optional P
AVSValue __cdecl FilteredResize::Create_SinPowerResize(AVSValue args, void*, IScriptEnvironment* env)
{
  auto f = SinPowerFilter(args[7].AsFloat(2.5f));
  const int force = args[8].AsInt(0);

  bool preserve_center = args[9].AsBool(true); // [keep_center] default Avisynth
  const char* placement_name = args[10].AsString("auto"); // [placement]s
  const int forced_chroma_placement = -1; // no force, used internally

  return CreateResize(args[0].AsClip(), args[1].AsInt(), args[2].AsInt(), &args[3], force, &f, preserve_center, placement_name, forced_chroma_placement, env);
}

// like SincFilter or LanczosFilter: optional Taps
AVSValue __cdecl FilteredResize::Create_SincLin2Resize(AVSValue args, void*, IScriptEnvironment* env)
{
  auto f = SincLin2Filter(args[7].AsInt(15));
  const int force = args[8].AsInt(0);

  bool preserve_center = args[9].AsBool(true); // [keep_center] default Avisynth
  const char* placement_name = args[10].AsString("auto"); // [placement]s
  const int forced_chroma_placement = -1; // no force, used internally

  return CreateResize(args[0].AsClip(), args[1].AsInt(), args[2].AsInt(), &args[3], force, &f, preserve_center, placement_name, forced_chroma_placement, env);
}

// like bicubic, plus 's'upport: optional B and C and S
AVSValue __cdecl FilteredResize::Create_UserDefined2Resize(AVSValue args, void*, IScriptEnvironment* env)
{
  auto f = UserDefined2Filter(args[3].AsFloat(121.0f), args[4].AsFloat(19.0f), args[5].AsFloat(2.3f));
  const int force = args[10].AsInt(0);

  bool preserve_center = args[11].AsBool(true); // [keep_center] default Avisynth
  const char* placement_name = args[12].AsString("auto"); // [placement]s
  const int forced_chroma_placement = -1; // no force, used internally

  return CreateResize(args[0].AsClip(), args[1].AsInt(), args[2].AsInt(), &args[6], force, &f, preserve_center, placement_name, forced_chroma_placement, env);
}

