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

// Overlay (c) 2003, 2004 by Klaus Post

#include "overlayfunctions.h"
#include "blend_common.h"
#ifdef INTEL_INTRINSICS
#include "intel/blend_common_sse.h"
#include "intel/blend_common_avx2.h"
#endif
#ifdef NEON_INTRINSICS
#include "aarch64/blend_common_neon.h"
#endif


#include <stdint.h>
#include <type_traits>

// ---------------------------------------------------------------------------
// Getters: select the best masked_merge_fn_t* / masked_merge_float_fn_t*
// for the current CPU and plane type (luma vs chroma).
// ---------------------------------------------------------------------------

// Float masked blend — placement-aware, parallel structure to the integer getter.
// is_lumamask_based_chroma=false → always MASK444 (luma).  is_lumamask_based_chroma=true → placement-aware.
// Note that when we use an exactly calculated and sized chroma mask, which is actually
// in 1:1 relation with the chroma plane, then is_lumamask_based_chroma=false must be given here.
static masked_merge_float_fn_t* get_overlay_blend_masked_float_fn(
  int cpuFlags, bool is_lumamask_based_chroma, int placement, const VideoInfo& vi_internal)
{
  MaskMode maskMode = is_lumamask_based_chroma ? resolveChromaMaskMode(placement, vi_internal) : MASK444;

#ifdef INTEL_INTRINSICS
  if (cpuFlags & CPUF_AVX2)   return get_overlay_blend_masked_float_fn_avx2(is_lumamask_based_chroma, maskMode);
  if (cpuFlags & CPUF_SSE4_1) return get_overlay_blend_masked_float_fn_sse41(is_lumamask_based_chroma, maskMode);
  if ((cpuFlags & CPUF_SSE2) && maskMode == MASK444) return &masked_merge_float_sse2;
#endif
#ifdef NEON_INTRINSICS
  if (cpuFlags & CPUF_ARM_NEON) return get_overlay_blend_masked_float_fn_neon(is_lumamask_based_chroma, maskMode);
#endif
  return get_overlay_blend_masked_float_fn_c(is_lumamask_based_chroma, maskMode);
}

// Integer masked blend: compute MaskMode from the internal VI + placement,
// then dispatch to the appropriate SIMD getter.
// is_lumamask_based_chroma=false → always MASK444 (luma plane is never sub-sampled).
// Note that when we use an exactly calculated and sized chroma mask, which is actually
// in 1:1 relation with the chroma plane, then is_lumamask_based_chroma=false must be given here.
static masked_merge_fn_t* get_overlay_blend_masked_fn(
  int cpuFlags, bool is_lumamask_based_chroma, int placement, const VideoInfo& vi_internal)
{
  MaskMode maskMode = is_lumamask_based_chroma ? resolveChromaMaskMode(placement, vi_internal) : MASK444;

#ifdef INTEL_INTRINSICS
  if (cpuFlags & CPUF_AVX2)   return get_overlay_blend_masked_fn_avx2(is_lumamask_based_chroma, maskMode);
  if (cpuFlags & CPUF_SSE4_1) return get_overlay_blend_masked_fn_sse41(is_lumamask_based_chroma, maskMode);
#endif
#ifdef NEON_INTRINSICS
  if (cpuFlags & CPUF_ARM_NEON) return get_overlay_blend_masked_fn_neon(is_lumamask_based_chroma, maskMode);
#endif
  return get_overlay_blend_masked_fn_c(is_lumamask_based_chroma, maskMode);
}

// Wrappers: adapt overlay_blend_plane_masked_opacity_t to weighted_merge SIMD functions.
// weight = round(opacity_f * 32768); invweight = 32768 - weight.  mask ptr is ignored.
#ifdef NEON_INTRINSICS
static void overlay_blend_neon_weighted_wrap(
  BYTE* p1, const BYTE* p2, const BYTE* /*mask*/,
  const int p1_pitch, const int p2_pitch, const int /*mask_pitch*/,
  const int width, const int height, const float opacity_f,
  const int bits_per_pixel)
{
  const int weight = static_cast<int>(opacity_f * 32768 + 0.5f);
  if (weight == 0 || weight == 32768) {
    // Avoid calling the full weighted_merge when one of the weights is zero, in case of no early out
    weighted_merge_return_a_or_b(p1, p2, p1_pitch, p2_pitch, width, height, weight, 32768 - weight, bits_per_pixel);
    return;
  }
  weighted_merge_neon(p1, p2, p1_pitch, p2_pitch, width, height, weight, 32768 - weight, bits_per_pixel);
}

static void overlay_blend_neon_weighted_float_wrap(
  BYTE* p1, const BYTE* p2, const BYTE* /*mask*/,
  const int p1_pitch, const int p2_pitch, const int /*mask_pitch*/,
  const int width, const int height, const float opacity_f,
  const int /*bits_per_pixel*/)
{
  weighted_merge_float_neon(p1, p2, p1_pitch, p2_pitch, width, height, opacity_f);
}
#endif // NEON_INTRINSICS

#ifdef INTEL_INTRINSICS
static void overlay_blend_weighted_avx2(
  BYTE* p1, const BYTE* p2, const BYTE* /*mask*/,
  const int p1_pitch, const int p2_pitch, const int /*mask_pitch*/,
  const int width, const int height, const float opacity_f,
  const int bits_per_pixel)
{
  const int weight = static_cast<int>(opacity_f * 32768 + 0.5f);
  if (weight == 0 || weight == 32768) {
    // Avoid calling the full weighted_merge when one of the weights is zero, in case of no early out
    weighted_merge_return_a_or_b(p1, p2, p1_pitch, p2_pitch, width, height, weight, 32768 - weight, bits_per_pixel);
    return;
  }
  weighted_merge_avx2(p1, p2, p1_pitch, p2_pitch, width, height, weight, 32768 - weight, bits_per_pixel);
}

static void overlay_blend_weighted_sse2(
  BYTE* p1, const BYTE* p2, const BYTE* /*mask*/,
  const int p1_pitch, const int p2_pitch, const int /*mask_pitch*/,
  const int width, const int height, const float opacity_f,
  const int bits_per_pixel)
{
  const int weight = static_cast<int>(opacity_f * 32768 + 0.5f);
  if (weight == 0 || weight == 32768) {
    // Avoid calling the full weighted_merge when one of the weights is zero, in case of no early out
    weighted_merge_return_a_or_b(p1, p2, p1_pitch, p2_pitch, width, height, weight, 32768 - weight, bits_per_pixel);
    return;
  }
  weighted_merge_sse2(p1, p2, p1_pitch, p2_pitch, width, height, weight, 32768 - weight, bits_per_pixel);
}

static void overlay_blend_weighted_float_avx2(
  BYTE* p1, const BYTE* p2, const BYTE* /*mask*/,
  const int p1_pitch, const int p2_pitch, const int /*mask_pitch*/,
  const int width, const int height, const float opacity_f,
  const int /*bits_per_pixel*/)
{
  weighted_merge_float_avx2(p1, p2, p1_pitch, p2_pitch, width, height, opacity_f);
}

static void overlay_blend_weighted_float_sse2(
  BYTE* p1, const BYTE* p2, const BYTE* /*mask*/,
  const int p1_pitch, const int p2_pitch, const int /*mask_pitch*/,
  const int width, const int height, const float opacity_f,
  const int /*bits_per_pixel*/)
{
  weighted_merge_float_sse2(p1, p2, p1_pitch, p2_pitch, width, height, opacity_f);
}
#endif

// Wrapper: adapt overlay_blend_plane_masked_opacity_t to weighted_merge_c.
// weight = round(opacity_f * 32768); invweight = 32768 - weight.
static void overlay_blend_weighted_c(
  BYTE* p1, const BYTE* p2, const BYTE* /*mask*/,
  const int p1_pitch, const int p2_pitch, const int /*mask_pitch*/,
  const int width, const int height, const float opacity_f,
  const int bits_per_pixel)
{
  const int weight = static_cast<int>(opacity_f * 32768 + 0.5f);
  if (weight == 0 || weight == 32768) {
    // Avoid calling the full weighted_merge_c when one of the weights is zero, in case of no early out
    weighted_merge_return_a_or_b(p1, p2, p1_pitch, p2_pitch, width, height, weight, 32768 - weight, bits_per_pixel);
    return;
  }
  weighted_merge_c(p1, p2, p1_pitch, p2_pitch, width, height, weight, 32768 - weight, bits_per_pixel);
}

static void overlay_blend_weighted_float_c(
  BYTE* p1, const BYTE* p2, const BYTE* /*mask*/,
  const int p1_pitch, const int p2_pitch, const int /*mask_pitch*/,
  const int width, const int height, const float opacity_f,
  const int /*bits_per_pixel*/)
{
  weighted_merge_float_c(p1, p2, p1_pitch, p2_pitch, width, height, opacity_f);
}

// ---------------------------------------------------------------------------
// Per-row chroma scratch helpers — runtime-dispatch on MaskMode to call the
// compile-time prepare_effective_mask_for_row.
// The caller pre-sizes `buf` to chroma_w; mask_pitch is in elements, not bytes.
// ---------------------------------------------------------------------------
template<typename pixel_t, bool full_opacity = true>
static void do_fill_chroma_row(
  pixel_t* buf, const pixel_t* luma_row,
  int luma_pitch_pixels, int chroma_w, MaskMode mode,
  int opacity_i = 0, int half = 0, MagicDiv magic = {})
{
  switch (mode) {
  case MASK411:
    prepare_effective_mask_for_row<MASK411,          pixel_t, full_opacity>(luma_row, luma_pitch_pixels, chroma_w, buf, opacity_i, half, magic); break;
  case MASK420:
    prepare_effective_mask_for_row<MASK420,          pixel_t, full_opacity>(luma_row, luma_pitch_pixels, chroma_w, buf, opacity_i, half, magic); break;
  case MASK420_MPEG2:
    prepare_effective_mask_for_row<MASK420_MPEG2,    pixel_t, full_opacity>(luma_row, luma_pitch_pixels, chroma_w, buf, opacity_i, half, magic); break;
  case MASK420_TOPLEFT:
    prepare_effective_mask_for_row<MASK420_TOPLEFT,  pixel_t, full_opacity>(luma_row, luma_pitch_pixels, chroma_w, buf, opacity_i, half, magic); break;
  case MASK422:
    prepare_effective_mask_for_row<MASK422,          pixel_t, full_opacity>(luma_row, luma_pitch_pixels, chroma_w, buf, opacity_i, half, magic); break;
  case MASK422_MPEG2:
    prepare_effective_mask_for_row<MASK422_MPEG2,    pixel_t, full_opacity>(luma_row, luma_pitch_pixels, chroma_w, buf, opacity_i, half, magic); break;
  case MASK422_TOPLEFT:
    prepare_effective_mask_for_row<MASK422_TOPLEFT,  pixel_t, full_opacity>(luma_row, luma_pitch_pixels, chroma_w, buf, opacity_i, half, magic); break;
  case MASK440:
    prepare_effective_mask_for_row<MASK440,          pixel_t, full_opacity>(luma_row, luma_pitch_pixels, chroma_w, buf, opacity_i, half, magic); break;
  case MASK410:
    prepare_effective_mask_for_row<MASK410,          pixel_t, full_opacity>(luma_row, luma_pitch_pixels, chroma_w, buf, opacity_i, half, magic); break;
  case MASK411_TOPLEFT:
    prepare_effective_mask_for_row<MASK411_TOPLEFT,  pixel_t, full_opacity>(luma_row, luma_pitch_pixels, chroma_w, buf, opacity_i, half, magic); break;
  case MASK440_TOPLEFT:
    prepare_effective_mask_for_row<MASK440_TOPLEFT,  pixel_t, full_opacity>(luma_row, luma_pitch_pixels, chroma_w, buf, opacity_i, half, magic); break;
  case MASK410_TOPLEFT:
    prepare_effective_mask_for_row<MASK410_TOPLEFT,  pixel_t, full_opacity>(luma_row, luma_pitch_pixels, chroma_w, buf, opacity_i, half, magic); break;
  default: break;
  }
}

template<bool full_opacity = true>
static void do_fill_chroma_row_f(
  float* buf, const float* luma_row,
  int luma_pitch_floats, int chroma_w, MaskMode mode,
  float opacity_f = 1.0f)
{
  switch (mode) {
  case MASK411:
    prepare_effective_mask_for_row_float_c<MASK411,          full_opacity>(luma_row, luma_pitch_floats, chroma_w, buf, opacity_f); break;
  case MASK420:
    prepare_effective_mask_for_row_float_c<MASK420,          full_opacity>(luma_row, luma_pitch_floats, chroma_w, buf, opacity_f); break;
  case MASK420_MPEG2:
    prepare_effective_mask_for_row_float_c<MASK420_MPEG2,    full_opacity>(luma_row, luma_pitch_floats, chroma_w, buf, opacity_f); break;
  case MASK420_TOPLEFT:
    prepare_effective_mask_for_row_float_c<MASK420_TOPLEFT,  full_opacity>(luma_row, luma_pitch_floats, chroma_w, buf, opacity_f); break;
  case MASK422:
    prepare_effective_mask_for_row_float_c<MASK422,          full_opacity>(luma_row, luma_pitch_floats, chroma_w, buf, opacity_f); break;
  case MASK422_MPEG2:
    prepare_effective_mask_for_row_float_c<MASK422_MPEG2,    full_opacity>(luma_row, luma_pitch_floats, chroma_w, buf, opacity_f); break;
  case MASK422_TOPLEFT:
    prepare_effective_mask_for_row_float_c<MASK422_TOPLEFT,  full_opacity>(luma_row, luma_pitch_floats, chroma_w, buf, opacity_f); break;
  case MASK440:
    prepare_effective_mask_for_row_float_c<MASK440,          full_opacity>(luma_row, luma_pitch_floats, chroma_w, buf, opacity_f); break;
  case MASK410:
    prepare_effective_mask_for_row_float_c<MASK410,          full_opacity>(luma_row, luma_pitch_floats, chroma_w, buf, opacity_f); break;
  case MASK411_TOPLEFT:
    prepare_effective_mask_for_row_float_c<MASK411_TOPLEFT,  full_opacity>(luma_row, luma_pitch_floats, chroma_w, buf, opacity_f); break;
  case MASK440_TOPLEFT:
    prepare_effective_mask_for_row_float_c<MASK440_TOPLEFT,  full_opacity>(luma_row, luma_pitch_floats, chroma_w, buf, opacity_f); break;
  case MASK410_TOPLEFT:
    prepare_effective_mask_for_row_float_c<MASK410_TOPLEFT,  full_opacity>(luma_row, luma_pitch_floats, chroma_w, buf, opacity_f); break;
  default: break;
  }
}

// do_fill_chroma_row_sse41 and do_fill_chroma_row_avx2 are declared in
// intel/blend_common_sse.h and intel/blend_common_avx2.h respectively,
// and defined (with explicit instantiations) in their matching .cpp files.

void OL_BlendImage::DoBlendImageMask(ImageOverlayInternal* base, ImageOverlayInternal* overlay, ImageOverlayInternal* mask) {
  if (bits_per_pixel == 8)
    BlendImageMask<uint8_t>(base, overlay, mask);
  else if (bits_per_pixel <= 16)
    BlendImageMask<uint16_t>(base, overlay, mask);
  else if (bits_per_pixel == 32)
    BlendImageMask<float>(base, overlay, mask);
}

void OL_BlendImage::DoBlendImage(ImageOverlayInternal* base, ImageOverlayInternal* overlay) {
  if (bits_per_pixel == 8)
    BlendImage<uint8_t>(base, overlay);
  else if (bits_per_pixel <= 16)
    BlendImage<uint16_t>(base, overlay);
  else if (bits_per_pixel == 32)
    BlendImage<float>(base, overlay);
}


template<typename pixel_t>
void OL_BlendImage::BlendImageMask(ImageOverlayInternal* base, ImageOverlayInternal* overlay, ImageOverlayInternal* mask) {
  int w = base->w();
  int h = base->h();

  int planeindex_from = 0;
  int planeindex_to = 0;

  if (of_mode == OF_Blend) {
    planeindex_from = 0;
    planeindex_to = greyscale ? 0 : 2;
  }
  else if (of_mode == OF_Luma) {
    planeindex_from = 0;
    planeindex_to = 0;
  }
  else if (of_mode == OF_Chroma) {
    if (greyscale)
      return;
    planeindex_from = 1;
    planeindex_to = 2;
  }

  const int cpuFlags = env->GetCPUFlags();

  // For RGB all planes are full-resolution (no subsampling), so G/B planes use the luma path.
  // For YUV, planes 1/2 are chroma and use the placement-aware path.
  const bool use_chroma_fn = !rgb;

  // Per-row scratch path: imghelpers stores the luma mask pointer in planes 1/2 for
  // subsampled greymask; BlendImageMask fills a per-row scratch buffer from luma rows,
  // shared for both U and V.
  // Active when use444=false keeps vi_internal at 420/422/411/440/410 AND greymask=true.
  // Not active when use444=true or input is natively 444/RGB: vi_internal is then at full
  // chroma resolution, Is420/Is422/Is411/Is440/Is410 == false, so is_subsampled = false.
  const bool is_subsampled = vi_internal.Is420() || vi_internal.Is422() || vi_internal.Is411() || vi_internal.Is440() || vi_internal.Is410();
  const bool use_scratch_path = greymask_mask && use_chroma_fn && is_subsampled;

  if constexpr (std::is_same_v<pixel_t, float>) {
    // Float path: separate luma/chroma function pointers.
    // blend_fn_chroma uses is_lumamask_based_chroma=false (→ MASK444) in the normal path
    // because the mask plane is already at chroma resolution (greymask=false or non-subsampled).
    // The scratch path pre-fills chroma-resolution scratch and calls blend_fn_luma on it.
    masked_merge_float_fn_t* blend_fn_luma   = get_overlay_blend_masked_float_fn(cpuFlags, false, placement, vi_internal);
    masked_merge_float_fn_t* blend_fn_chroma =
      use_scratch_path ? blend_fn_luma
      : get_overlay_blend_masked_float_fn(cpuFlags, false, placement, vi_internal);

    if (use_scratch_path) {
      // Luma plane (if applicable)
      if (planeindex_from == 0) {
        blend_fn_luma(
          base->GetPtrByIndex(0), overlay->GetPtrByIndex(0), mask->GetPtrByIndex(0),
          base->GetPitchByIndex(0), overlay->GetPitchByIndex(0), mask->GetPitchByIndex(0),
          w, h, opacity_f);
      }
      // Chroma planes: outer loop = rows, inner = planes.
      // Scratch filled once per row, reused for all active chroma planes (U and V).
      if (planeindex_to >= 1) {
        MaskMode chroma_maskMode = resolveChromaMaskMode(placement, vi_internal);

        const int xws = base->xSubSamplingShifts[1];
        const int yhs = base->ySubSamplingShifts[1];
        const int chroma_w = (xws > 0) ? ((base->xAccum() & ((1 << xws) - 1)) + w + (1 << xws) - 1) >> xws : w;
        const int chroma_h = (yhs > 0) ? ((base->yAccum() & ((1 << yhs) - 1)) + h + (1 << yhs) - 1) >> yhs : h;
        const int luma_mask_pitch_floats = mask->GetPitchByIndex(0) / (int)sizeof(float);
        // Vertical row advance per chroma output row, one luma-mask row group per
        // chroma_maskMode's actual MaskVSubsample (blend_common.h)
        // 4 for 4:1:0
        // 2 for 4:2:0 and 4:4:0
        // 1 for others (4:1:1/4:2:2, no vertical subsampling).
        const int luma_mask_advance =
          (chroma_maskMode == MASK410 || chroma_maskMode == MASK410_TOPLEFT) ? luma_mask_pitch_floats * 4 :
          (chroma_maskMode == MASK420 || chroma_maskMode == MASK420_MPEG2 || chroma_maskMode == MASK420_TOPLEFT ||
           chroma_maskMode == MASK440 || chroma_maskMode == MASK440_TOPLEFT) ? luma_mask_pitch_floats * 2 :
          luma_mask_pitch_floats;
        const int scratch_pitch = chroma_w * (int)sizeof(float);
        std::vector<float> scratch(chroma_w);

        // Mutable row pointers for each active chroma plane (p=1, p=2)
        BYTE*       basep[2] = { base->GetPtrByIndex(1),    base->GetPtrByIndex(2)    };
        const BYTE* ovlp [2] = { overlay->GetPtrByIndex(1), overlay->GetPtrByIndex(2) };
        const int base_pitch[2] = { base->GetPitchByIndex(1),    base->GetPitchByIndex(2)    };
        const int ovl_pitch [2] = { overlay->GetPitchByIndex(1), overlay->GetPitchByIndex(2) };
        const int n_planes = planeindex_to; // 1 or 2 active chroma planes (indices 0..n_planes-1)

        const float* lmask = reinterpret_cast<const float*>(mask->GetPtrByIndex(0));
        const bool chroma_full_opacity = (opacity_f == 1.0f);
#ifdef INTEL_INTRINSICS
        const bool use_avx2_rowprep_f  = (cpuFlags & CPUF_AVX2)   != 0;
        const bool use_sse41_rowprep_f = (cpuFlags & CPUF_SSE4_1) != 0;
#endif
#ifdef NEON_INTRINSICS
        const bool use_neon_rowprep_f = (cpuFlags & CPUF_ARM_NEON) != 0;
#endif
        for (int y = 0; y < chroma_h; y++) {
          // Fill scratch once for this luma-mask row pair, baking opacity in.
#ifdef INTEL_INTRINSICS
          if (use_avx2_rowprep_f) {
            if (chroma_full_opacity)
              do_fill_chroma_row_float_avx2<true> (scratch.data(), lmask, luma_mask_pitch_floats, chroma_w, chroma_maskMode);
            else
              do_fill_chroma_row_float_avx2<false>(scratch.data(), lmask, luma_mask_pitch_floats, chroma_w, chroma_maskMode, opacity_f);
          } else if (use_sse41_rowprep_f) {
            if (chroma_full_opacity)
              do_fill_chroma_row_float_sse41<true> (scratch.data(), lmask, luma_mask_pitch_floats, chroma_w, chroma_maskMode);
            else
              do_fill_chroma_row_float_sse41<false>(scratch.data(), lmask, luma_mask_pitch_floats, chroma_w, chroma_maskMode, opacity_f);
          } else
#elif defined(NEON_INTRINSICS)
          if (use_neon_rowprep_f) {
            if (chroma_full_opacity)
              do_fill_chroma_row_float_neon<true>(scratch.data(), lmask, luma_mask_pitch_floats, chroma_w, chroma_maskMode);
            else
              do_fill_chroma_row_float_neon<false>(scratch.data(), lmask, luma_mask_pitch_floats, chroma_w, chroma_maskMode, opacity_f);
          } else
#endif
          {
            if (chroma_full_opacity)
              do_fill_chroma_row_f<true> (scratch.data(), lmask, luma_mask_pitch_floats, chroma_w, chroma_maskMode);
            else
              do_fill_chroma_row_f<false>(scratch.data(), lmask, luma_mask_pitch_floats, chroma_w, chroma_maskMode, opacity_f);
          }
          // Apply to each active chroma plane; opacity already baked — pass 1.0f.
          for (int i = 0; i < n_planes; i++) {
            blend_fn_luma(basep[i], ovlp[i], (const BYTE*)scratch.data(),
              base_pitch[i], ovl_pitch[i], scratch_pitch, chroma_w, 1, 1.0f);
            basep[i] += base_pitch[i];
            ovlp [i] += ovl_pitch[i];
          }
          lmask += luma_mask_advance;
        }
      }
      return;
    }

    // Normal path: greymask=false or non-subsampled — mask already at plane resolution.
    for (int p = planeindex_from; p <= planeindex_to; p++) {
      masked_merge_float_fn_t* fn = (p == 0 || !use_chroma_fn) ? blend_fn_luma : blend_fn_chroma;
      const int xws_p = base->xSubSamplingShifts[p];
      const int yhs_p = base->ySubSamplingShifts[p];
      const int plane_w = (xws_p > 0) ? ((base->xAccum() & ((1 << xws_p) - 1)) + w + (1 << xws_p) - 1) >> xws_p : w;
      const int plane_h = (yhs_p > 0) ? ((base->yAccum() & ((1 << yhs_p) - 1)) + h + (1 << yhs_p) - 1) >> yhs_p : h;
      fn(base->GetPtrByIndex(p), overlay->GetPtrByIndex(p), mask->GetPtrByIndex(p),
        base->GetPitchByIndex(p), overlay->GetPitchByIndex(p), mask->GetPitchByIndex(p),
        plane_w, plane_h, opacity_f);
    }
  }
  else {
    // Integer path: opacity pre-scaled to [0, max_pixel_value].
    const int opacity_i = static_cast<int>(opacity_f * ((1 << bits_per_pixel) - 1) + 0.5f);
    masked_merge_fn_t* blend_fn_luma   = get_overlay_blend_masked_fn(cpuFlags, false, placement, vi_internal);
    masked_merge_fn_t* blend_fn_chroma =
      use_scratch_path ? blend_fn_luma
      : get_overlay_blend_masked_fn(cpuFlags, false, placement, vi_internal);

    if (use_scratch_path) {
      // Luma plane (if applicable)
      if (planeindex_from == 0) {
        blend_fn_luma(
          base->GetPtrByIndex(0), overlay->GetPtrByIndex(0), mask->GetPtrByIndex(0),
          base->GetPitchByIndex(0), overlay->GetPitchByIndex(0), mask->GetPitchByIndex(0),
          w, h, opacity_i, bits_per_pixel);
      }
      // Chroma planes: outer loop = rows, inner = planes.
      // Scratch filled once per row, reused for all active chroma planes (U and V).
      if (planeindex_to >= 1) {
        MaskMode chroma_maskMode = resolveChromaMaskMode(placement, vi_internal);

        const int xws = base->xSubSamplingShifts[1];
        const int yhs = base->ySubSamplingShifts[1];
        const int chroma_w = (xws > 0) ? ((base->xAccum() & ((1 << xws) - 1)) + w + (1 << xws) - 1) >> xws : w;
        const int chroma_h = (yhs > 0) ? ((base->yAccum() & ((1 << yhs) - 1)) + h + (1 << yhs) - 1) >> yhs : h;
        const int luma_mask_pitch_pixels = mask->GetPitchByIndex(0) / (int)sizeof(pixel_t);
        // Vertical row advance per chroma output row, one luma-mask row group per
        // chroma_maskMode's actual MaskVSubsample (blend_common.h):
        // 4 for 4:1:0
        // 2 for 4:2:0 and 4:4:0
        // 1 for others (4:1:1/4:2:2, no vertical subsampling)
        const int luma_mask_advance =
          (chroma_maskMode == MASK410 || chroma_maskMode == MASK410_TOPLEFT) ? luma_mask_pitch_pixels * 4 :
          (chroma_maskMode == MASK420 || chroma_maskMode == MASK420_MPEG2 || chroma_maskMode == MASK420_TOPLEFT ||
           chroma_maskMode == MASK440 || chroma_maskMode == MASK440_TOPLEFT) ? luma_mask_pitch_pixels * 2 :
          luma_mask_pitch_pixels;
        const int scratch_pitch = chroma_w * (int)sizeof(pixel_t);
        std::vector<pixel_t> scratch(chroma_w);

        // Mutable row pointers for each active chroma plane (p=1, p=2)
        BYTE*       basep[2] = { base->GetPtrByIndex(1),    base->GetPtrByIndex(2)    };
        const BYTE* ovlp [2] = { overlay->GetPtrByIndex(1), overlay->GetPtrByIndex(2) };
        const int base_pitch[2] = { base->GetPitchByIndex(1),    base->GetPitchByIndex(2)    };
        const int ovl_pitch [2] = { overlay->GetPitchByIndex(1), overlay->GetPitchByIndex(2) };
        const int n_planes = planeindex_to; // 1 or 2 active chroma planes (indices 0..n_planes-1)

        const pixel_t* lmask = reinterpret_cast<const pixel_t*>(mask->GetPtrByIndex(0));
        const int max_pixel_value = (1 << bits_per_pixel) - 1;
        const MagicDiv mag        = get_magic_div(bits_per_pixel);
        const int half_val        = max_pixel_value / 2;
        const bool chroma_full_opacity = (opacity_i == max_pixel_value);
#ifdef INTEL_INTRINSICS
        const bool use_avx2_rowprep  = (cpuFlags & CPUF_AVX2)   != 0;
        const bool use_sse41_rowprep = (cpuFlags & CPUF_SSE4_1) != 0;
#endif
#ifdef NEON_INTRINSICS
        const bool use_neon_rowprep = (cpuFlags & CPUF_ARM_NEON) != 0;
#endif

        for (int y = 0; y < chroma_h; y++) {
          // Fill scratch once for this luma-mask row pair, baking opacity in.
#ifdef INTEL_INTRINSICS
          if (use_avx2_rowprep) {
            if (chroma_full_opacity)
              do_fill_chroma_row_avx2<pixel_t, true> (scratch.data(), lmask, luma_mask_pitch_pixels, chroma_w, chroma_maskMode);
            else
              do_fill_chroma_row_avx2<pixel_t, false>(scratch.data(), lmask, luma_mask_pitch_pixels, chroma_w, chroma_maskMode, opacity_i, half_val, mag);
          } else if (use_sse41_rowprep) {
            if (chroma_full_opacity)
              do_fill_chroma_row_sse41<pixel_t, true> (scratch.data(), lmask, luma_mask_pitch_pixels, chroma_w, chroma_maskMode);
            else
              do_fill_chroma_row_sse41<pixel_t, false>(scratch.data(), lmask, luma_mask_pitch_pixels, chroma_w, chroma_maskMode, opacity_i, half_val, mag);
          } else
#elif defined(NEON_INTRINSICS)
          if (use_neon_rowprep) {
            if (chroma_full_opacity)
              do_fill_chroma_row_neon<pixel_t, true>(scratch.data(), lmask, luma_mask_pitch_pixels, chroma_w, chroma_maskMode);
            else
              do_fill_chroma_row_neon<pixel_t, false>(scratch.data(), lmask, luma_mask_pitch_pixels, chroma_w, chroma_maskMode, opacity_i, half_val, mag);
          }
          else
#endif
          {
            if (chroma_full_opacity)
              do_fill_chroma_row<pixel_t, true> (scratch.data(), lmask, luma_mask_pitch_pixels, chroma_w, chroma_maskMode);
            else
              do_fill_chroma_row<pixel_t, false>(scratch.data(), lmask, luma_mask_pitch_pixels, chroma_w, chroma_maskMode, opacity_i, half_val, mag);
          }
          // Apply to each active chroma plane; opacity already baked — pass max_pixel_value.
          for (int i = 0; i < n_planes; i++) {
            blend_fn_luma(basep[i], ovlp[i], (const BYTE*)scratch.data(),
              base_pitch[i], ovl_pitch[i], scratch_pitch, chroma_w, 1, max_pixel_value, bits_per_pixel);
            basep[i] += base_pitch[i];
            ovlp [i] += ovl_pitch[i];
          }
          lmask += luma_mask_advance;
        }
      }
      return;
    }

    // Normal path: greymask=false or non-subsampled — mask already at plane resolution.
    for (int p = planeindex_from; p <= planeindex_to; p++) {
      masked_merge_fn_t* fn = (p == 0 || !use_chroma_fn) ? blend_fn_luma : blend_fn_chroma;
      const int xws_p = base->xSubSamplingShifts[p];
      const int yhs_p = base->ySubSamplingShifts[p];
      const int plane_w = (xws_p > 0) ? ((base->xAccum() & ((1 << xws_p) - 1)) + w + (1 << xws_p) - 1) >> xws_p : w;
      const int plane_h = (yhs_p > 0) ? ((base->yAccum() & ((1 << yhs_p) - 1)) + h + (1 << yhs_p) - 1) >> yhs_p : h;
      fn(base->GetPtrByIndex(p), overlay->GetPtrByIndex(p), mask->GetPtrByIndex(p),
        base->GetPitchByIndex(p), overlay->GetPitchByIndex(p), mask->GetPitchByIndex(p),
        plane_w, plane_h, opacity_i, bits_per_pixel);
    }
  }
}

template<typename pixel_t>
void OL_BlendImage::BlendImage(ImageOverlayInternal* base, ImageOverlayInternal* overlay) {
  int w = base->w();
  int h = base->h();

  const int pixelsize = sizeof(pixel_t);

  int planeindex_from = 0;
  int planeindex_to = 0;

  if (of_mode == OF_Blend) {
    planeindex_from = 0;
    planeindex_to = greyscale ? 0 : 2;
  }
  else if (of_mode == OF_Luma) {
    planeindex_from = 0;
    planeindex_to = 0;
  }
  else if (of_mode == OF_Chroma) {
    if (greyscale)
      return;
    planeindex_from = 1;
    planeindex_to = 2;
  }

  // opacity == 0 was also an early-out case
  if (opacity == 256) {
    for (int p = planeindex_from; p <= planeindex_to; p++) {
      const int xws_p = base->xSubSamplingShifts[p];
      const int yhs_p = base->ySubSamplingShifts[p];
      const int plane_w = (xws_p > 0) ? ((base->xAccum() & ((1 << xws_p) - 1)) + w + (1 << xws_p) - 1) >> xws_p : w;
      const int plane_h = (yhs_p > 0) ? ((base->yAccum() & ((1 << yhs_p) - 1)) + h + (1 << yhs_p) - 1) >> yhs_p : h;
      env->BitBlt(base->GetPtrByIndex(p), base->GetPitchByIndex(p), overlay->GetPtrByIndex(p), overlay->GetPitchByIndex(p), plane_w * pixelsize, plane_h);
    }
  }
  else {
    overlay_blend_plane_masked_opacity_t* blend_fn = nullptr;

#ifdef INTEL_INTRINSICS
  if (pixelsize == 4 && (env->GetCPUFlags() & CPUF_AVX2)) {
    blend_fn = overlay_blend_weighted_float_avx2;
  }
  else if (pixelsize == 4 && (env->GetCPUFlags() & CPUF_SSE2)) {
    blend_fn = overlay_blend_weighted_float_sse2;
  }
  else if (env->GetCPUFlags() & CPUF_AVX2) {
    blend_fn = overlay_blend_weighted_avx2;  // handles all integer depths, width-exact
  }
  else if (env->GetCPUFlags() & CPUF_SSE2) {
    blend_fn = overlay_blend_weighted_sse2;  // handles all integer depths
  }
  else
#endif // INTEL_INTRINSICS
#ifdef NEON_INTRINSICS
    if (pixelsize == 4 && (env->GetCPUFlags() & CPUF_ARM_NEON)) {
      blend_fn = overlay_blend_neon_weighted_float_wrap;
    }
    else if (env->GetCPUFlags() & CPUF_ARM_NEON) {
      blend_fn = overlay_blend_neon_weighted_wrap;
    }
    else
#endif // NEON_INTRINSICS
    {
    // pure C
    if (bits_per_pixel == 32) blend_fn = overlay_blend_weighted_float_c;
    else                      blend_fn = overlay_blend_weighted_c;
  }
  // end of new, float precision inside masked overlays

    if (blend_fn == nullptr)
      env->ThrowError("Blend: no valid internal function");

    for (int p = planeindex_from; p <= planeindex_to; p++) {
      const int xws_p = base->xSubSamplingShifts[p];
      const int yhs_p = base->ySubSamplingShifts[p];
      const int plane_w = (xws_p > 0) ? ((base->xAccum() & ((1 << xws_p) - 1)) + w + (1 << xws_p) - 1) >> xws_p : w;
      const int plane_h = (yhs_p > 0) ? ((base->yAccum() & ((1 << yhs_p) - 1)) + h + (1 << yhs_p) - 1) >> yhs_p : h;
      // no mask ptr
      blend_fn(
        base->GetPtrByIndex(p), overlay->GetPtrByIndex(p), nullptr,
        base->GetPitchByIndex(p), overlay->GetPitchByIndex(p), 0,
        plane_w, plane_h, opacity_f,
        bits_per_pixel);
    }

  }
}
