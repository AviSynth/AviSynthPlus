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

#ifndef __blend_common_h
#define __blend_common_h

#include <avs/types.h>
#include "../../core/scratch_buffer.h"
#include <avs/config.h>
#include <vector>
#include "../../convert/frame_prop_enums.h"

// Magic integer dividers for exact division by max_pixel_value (e.g. 255, 1023, …).
//
// Why /max_val instead of a power-of-two shift (÷256, ÷1024, …)?
//
// The per-pixel blend formula is:  result = (a*(max-m) + b*m + half) / max
// where m ∈ [0..max] is the mask weight and max = (1<<bpp)-1.
//
// With the simpler power-of-two shift approach: (a*(range-m) + b*m) >> bpp,
// range = 1<<bpp = max+1 = 256 for 8-bit - the divisor is 256, not 255.
// That means m=255 maps to the fraction 255/256 ≈ 0.996, never to 1.0:
//
//   m=255, b=255, a=0:  (0*1 + 255*255) >> 8 = 65025 >> 8 = 254   is wrong
//
// The full-range endpoint is unreachable.  Old Layer code worked around this
// by inflating the level: level = round(opacity * (max+1)) for the non-alpha
// path and level = round(opacity * (max+2)) for the alpha-aware path, so that
// a combined alpha*level product could reach the next power-of-two.  This fixed
// the two extreme values (0 and max) but introduced asymmetry: intermediate
// mask values are unevenly distributed over the output range because 255 steps
// are mapped through a 256-wide divisor, leaving one "missing" step.
//
// Using /max (this approach) is exact at both endpoints and uniformly linear
// throughout: m=0 -> a, m=max -> b, every intermediate step is one equal part.
// The "magic" multiply+shift (compiler style constant division optimisation)
// makes the division as cheap as a shift at runtime.
struct MagicDiv {
  uint32_t div;
  uint8_t  shift;
};

constexpr MagicDiv get_magic_div(int bits_per_pixel) {
  switch (bits_per_pixel) {
  case  8: return { 0x8081,     7 };  // mulhi_epu16 path, different from word path
  case 10: return { 0x80200803, 9 };
  case 12: return { 0x80080081, 11 };
  case 14: return { 0x80020009, 13 };
  case 16: return { 0x80008001, 15 };
  default: return { 0, 0 };            // unreachable
  }
}

// Apply prefetched MagicDiv to value tmp.
// 8-bit (sizeof(pixel_t)==1) uses the mulhi_epu16 path (32-bit intermediate);
// wider uses the mul_epu32 path (64-bit intermediate)
// tmp is 0-255 for 8-bit pixels, and up to 65535 at 16-bit pixels.
template<typename pixel_t>
AVS_FORCEINLINE static int magic_div_rt(uint32_t tmp, const MagicDiv& magic) {
  if constexpr (sizeof(pixel_t) == 1)
    return (int)(((uint32_t)tmp * magic.div) >> (16 + magic.shift));
  else
    return (int)(((uint64_t)tmp * magic.div) >> (32 + magic.shift));
}

// ============================================================
// Chroma placement mask support
// Shared by Overlay and Layer blend paths.
// MASK444: mask is already at the plane's resolution (1:1).
// Other modes: mask is at luma resolution; values are averaged per placement
//   before blending chroma planes.
// ============================================================

// Mask downsampling kernels per chroma siting. Each axis uses one of these filters (weights are
// symmetric; offsets are luma positions relative to the first luma sample of the chroma block):
//   ratio 1                : [1]                              offset  0
//   ratio 2, centred       : [1 1]          /2  (box)          offsets 0..1
//   ratio 2, co-sited      : [1 2 1]        /4                 offsets -1..1
//   ratio 4, centred       : [1 3 5 7 7 5 3 1] /32 (triangle)  offsets -2..5
//   ratio 4, co-sited      : [1 2 3 4 3 2 1]   /16 (tent)      offsets -3..3
// The ratio 4 filters are the bilinear (triangle) downsampling filters, so 4:1:1 and 4:1:0 masks are
// low-pass filtered instead of point-sampled; the ratio 2 centred box is kept (area average).
// Mask edges are clamped (the first/last luma row and column are replicated).
enum MaskMode {
  MASK410,          // 4:1:0, centred H+V: 8x8 triangle, /1024
  MASK410_MPEG2,    // 4:1:0, left-sited H, centred V: 7x8 (tent x triangle), /512
  MASK410_TOPLEFT,  // 4:1:0, co-sited H+V: 7x7 tent, /256
  MASK440,          // 4:4:0, centred V: 2-row box, /2
  MASK440_TOPLEFT,  // 4:4:0, top-sited V: [1 2 1] over rows -1..1, /4
  MASK411,          // 4:1:1, centred H: 8-tap triangle, /32
  MASK411_TOPLEFT,  // 4:1:1, left-sited H (left, top_left, bottom_left): 7-tap tent, /16
  MASK420,          // 4:2:0, centred H+V: 2x2 box
  MASK420_MPEG2,    // 4:2:0, left-sited H, centred V: [1 2 1] x 2-row box
  MASK420_TOPLEFT,  // 4:2:0, co-sited H+V: [1 2 1] x [1 2 1] (3x3), /16
  MASK422,          // 4:2:2, centred H: 2-tap box
  MASK422_MPEG2,    // 4:2:2, left-sited H: [1 2 1]; also used for top_left (same horizontal siting)
  MASK444,
  MASK_MODE_COUNT // array size sentinel, not a mode
};

// Vertical subsampling factor of a mask mode
// number of luma/alpha (fullres) rows correspond to one chroma row.
template<MaskMode maskMode>
constexpr int MaskVSubsample =
  (maskMode == MASK410 || maskMode == MASK410_MPEG2 || maskMode == MASK410_TOPLEFT) ? 4 :
  (maskMode == MASK420 || maskMode == MASK420_MPEG2 || maskMode == MASK420_TOPLEFT ||
   maskMode == MASK440 || maskMode == MASK440_TOPLEFT) ? 2 : 1;

// Horizontal subsampling factor of a mask mode
// number of luma/alpha (fullres) columns correspond to one chroma column.
template<MaskMode maskMode>
constexpr int MaskHSubsample =
  (maskMode == MASK411 || maskMode == MASK411_TOPLEFT ||
   maskMode == MASK410 || maskMode == MASK410_MPEG2 || maskMode == MASK410_TOPLEFT) ? 4 :
  (maskMode == MASK444 || maskMode == MASK440 || maskMode == MASK440_TOPLEFT) ? 1 : 2;

// Runtime version of MaskVSubsample, for callers that select the mask mode at runtime.
AVS_FORCEINLINE static int MaskVSubsampleRt(MaskMode m) {
  switch (m) {
  case MASK410: case MASK410_MPEG2: case MASK410_TOPLEFT: return 4;
  case MASK420: case MASK420_MPEG2: case MASK420_TOPLEFT: case MASK440: case MASK440_TOPLEFT: return 2;
  default: return 1;
  }
}

// Row context for the row preparation of chroma row y (height: number of chroma rows):
// how many mask (luma) rows exist above the block's first row, and below it. Vertical filters clamp to them.
AVS_FORCEINLINE static int mask_rows_above(int y, int vsubsample) { return y * vsubsample; }
AVS_FORCEINLINE static int mask_rows_below(int y, int height, int vsubsample) { return (height - y) * vsubsample - 1; }

// ------------------------------------------------------------
// Separable mask kernels (4:2:0 top_left, 4:4:0, 4:1:1, 4:1:0)
// ------------------------------------------------------------
// Each filter is a cascade of box sums, evaluated with additions only, in a fixed order:
//   [1 2 1] = box2 * box2, [1 2 3 4 3 2 1] = box4 * box4, [1 3 5 7 7 5 3 1] = box2 * box4 * box4.
// Vertical pass first (per column), then horizontal (per output pixel); the weight sum is a power of two,
// so the integer result is (sum + half) >> shift and the float result sum * 2^-shift.
// The SIMD versions evaluate the same cascade on vectors (same additions, same order), so integer AND float
// results are bit-identical to C.
// Overflow: the largest weight sum is 1024 (8x8): 65535 * 1024 < 2^31, int32 is enough.
enum MaskFilter { MF_ID, MF_BOX2, MF_121, MF_TENT7, MF_TRI8 };

template<MaskFilter F> constexpr int MaskFilterFirst = F == MF_121 ? -1 : F == MF_TENT7 ? -3 : F == MF_TRI8 ? -2 : 0;
template<MaskFilter F> constexpr int MaskFilterTaps = F == MF_ID ? 1 : F == MF_BOX2 ? 2 : F == MF_121 ? 3 : F == MF_TENT7 ? 7 : 8;
template<MaskFilter F> constexpr int MaskFilterLog2 = F == MF_ID ? 0 : F == MF_BOX2 ? 1 : F == MF_121 ? 2 : F == MF_TENT7 ? 4 : 5;

template<MaskMode maskMode>
constexpr bool MaskIsSeparable =
  maskMode == MASK410 || maskMode == MASK410_MPEG2 || maskMode == MASK410_TOPLEFT ||
  maskMode == MASK440 || maskMode == MASK440_TOPLEFT || maskMode == MASK411 || maskMode == MASK411_TOPLEFT ||
  maskMode == MASK420_TOPLEFT;

// Horizontal and vertical filter of every mode. The 4:2:0 / 4:2:2 box and mpeg2 modes are listed as well
// (BitmapText's chroma renderer uses them), but their row preparation keeps the dedicated
// calculate_effective_mask kernels; MaskIsSeparable selects the modes of the separable engine.
template<MaskMode maskMode>
constexpr MaskFilter MaskHFilter =
  (maskMode == MASK410 || maskMode == MASK411) ? MF_TRI8 :
  (maskMode == MASK410_MPEG2 || maskMode == MASK410_TOPLEFT || maskMode == MASK411_TOPLEFT) ? MF_TENT7 :
  (maskMode == MASK420_TOPLEFT || maskMode == MASK420_MPEG2 || maskMode == MASK422_MPEG2) ? MF_121 :
  (maskMode == MASK420 || maskMode == MASK422) ? MF_BOX2 : MF_ID;

template<MaskMode maskMode>
constexpr MaskFilter MaskVFilter =
  (maskMode == MASK410 || maskMode == MASK410_MPEG2) ? MF_TRI8 :
  (maskMode == MASK410_TOPLEFT) ? MF_TENT7 :
  (maskMode == MASK440 || maskMode == MASK420 || maskMode == MASK420_MPEG2) ? MF_BOX2 :
  (maskMode == MASK440_TOPLEFT || maskMode == MASK420_TOPLEFT) ? MF_121 : MF_ID;

template<MaskMode maskMode>
constexpr int MaskSepShift = MaskFilterLog2<MaskHFilter<maskMode>> + MaskFilterLog2<MaskVFilter<maskMode>>;

// Box-cascade evaluation of filter F over v[0 .. taps-1]; add(a, b) is the addition (scalar or SIMD).
template<MaskFilter F, typename T, typename Add>
AVS_FORCEINLINE static T mask_filter_cascade(const T* v, Add add) {
  if constexpr (F == MF_ID) {
    return v[0];
  }
  else if constexpr (F == MF_BOX2) {
    return add(v[0], v[1]);
  }
  else if constexpr (F == MF_121) {
    return add(add(v[0], v[1]), add(v[1], v[2]));
  }
  else if constexpr (F == MF_TENT7) {
    const T s0 = add(add(add(v[0], v[1]), v[2]), v[3]);
    const T s1 = add(add(add(v[1], v[2]), v[3]), v[4]);
    const T s2 = add(add(add(v[2], v[3]), v[4]), v[5]);
    const T s3 = add(add(add(v[3], v[4]), v[5]), v[6]);
    return add(add(add(s0, s1), s2), s3);
  }
  else { // MF_TRI8 = tent over v[0..6] + tent over v[1..7]
    return add(mask_filter_cascade<MF_TENT7, T>(v, add), mask_filter_cascade<MF_TENT7, T>(v + 1, add));
  }
}

// The clamped row pointers of the vertical filter (rows_above/rows_below: see mask_rows_above/below).
template<MaskMode maskMode, typename sample_t>
AVS_FORCEINLINE static void mask_sep_rows(const sample_t* maskp, int mask_pitch, int rows_above, int rows_below,
  const sample_t* (&rows)[8])
{
  constexpr MaskFilter VF = MaskVFilter<maskMode>;
  for (int j = 0; j < MaskFilterTaps<VF>; j++) {
    int dy = MaskFilterFirst<VF> + j;
    dy = dy < -rows_above ? -rows_above : (dy > rows_below ? rows_below : dy);
    rows[j] = maskp + (ptrdiff_t)dy * mask_pitch;
  }
}

// Weighted (not yet normalized) sum for chroma pixel x. acc_t: int32_t for integer, float for float samples.
// luma_width: mask row width, columns are clamped to 0 .. luma_width-1.
template<MaskMode maskMode, typename sample_t, typename acc_t>
AVS_FORCEINLINE static acc_t mask_sep_sum(const sample_t* const (&rows)[8], int x, int luma_width) {
  constexpr MaskFilter HF = MaskHFilter<maskMode>;
  constexpr MaskFilter VF = MaskVFilter<maskMode>;
  constexpr int H = MaskHSubsample<maskMode>;
  auto add = [](acc_t a, acc_t b) { return a + b; };
  acc_t colsum[8];
  for (int k = 0; k < MaskFilterTaps<HF>; k++) {
    int c = x * H + MaskFilterFirst<HF> + k;
    c = c < 0 ? 0 : (c > luma_width - 1 ? luma_width - 1 : c);
    acc_t v[8];
    for (int j = 0; j < MaskFilterTaps<VF>; j++)
      v[j] = (acc_t)rows[j][c];
    colsum[k] = mask_filter_cascade<VF, acc_t>(v, add);
  }
  return mask_filter_cascade<HF, acc_t>(colsum, add);
}

// ------------------------------------------------------------
// Box / [1 2 1] kernels of 4:2:0 (centre, mpeg2) and 4:2:2 (centre, mpeg2)
// ------------------------------------------------------------

// Generic center-placement box average over an HxV luma block,
// where H = MaskHSubsample<maskMode> and V = MaskVSubsample<maskMode>.
template<MaskMode maskMode, typename pixel_t>
AVS_FORCEINLINE static pixel_t calculate_effective_mask_center(const pixel_t* ptr, int x, int pitch) {
  constexpr int H = MaskHSubsample<maskMode>;
  constexpr int V = MaskVSubsample<maskMode>;
  constexpr int count = H * V;
  int sum = 0;
  for (int h = 0; h < V; h++)
    for (int w = 0; w < H; w++)
      sum += ptr[x * H + w + h * pitch];
  return (pixel_t)((sum + count / 2 /* rounder */) / count);
}

// Float version of calculate_effective_mask_center.
template<MaskMode maskMode>
AVS_FORCEINLINE static float calculate_effective_mask_center_f(const float* ptr, int x, int pitch) {
  constexpr int H = MaskHSubsample<maskMode>;
  constexpr int V = MaskVSubsample<maskMode>;
  constexpr float inv_count = 1.0f / (H * V);
  float sum = 0.0f;
  for (int h = 0; h < V; h++)
    for (int w = 0; w < H; w++)
      sum += ptr[x * H + w + h * pitch];
  return sum * inv_count;
}

// Mask downsampling kernel variants — shared by Overlay, Layer and Subtitle (GDI).
// Selected from the clip's chroma location by chromaLocationToMaskPlacement below.
// PLACEMENT_MPEG2   (0): left-sited H, centred V  (H.262/MPEG-2, H.264 default)
// PLACEMENT_MPEG1   (1): centred H+V              (MPEG-1 / JPEG)
// PLACEMENT_TOPLEFT (2): co-sited H+V             (UHD/HDR HEVC, AV1 'colocated')
// The siting sets the position of the filter, the filter itself is always a low-pass (see MaskMode).
enum { PLACEMENT_MPEG2 = 0, PLACEMENT_MPEG1 = 1, PLACEMENT_TOPLEFT = 2 };

// Maps a chroma location (ChromaLocation_e) to the implemented mask kernel variant. When the exact
// kernel is not implemented, the nearest one is taken:
// - 4:2:0, 4:2:2, 4:1:1, 4:1:0: horizontally centered (center, top, bottom) -> MPEG1;
//   top_left -> TOPLEFT; left, bottom_left, dv (U/V average is left) -> MPEG2.
// - 4:4:0 (vertical only): top_left, top -> TOPLEFT; left, center, bottom_left, bottom, dv -> MPEG1.
// resolveChromaMaskMode then picks the format's actual kernel for the variant; formats
// without a separate one use the nearest they have (4:2:2 top_left -> its MPEG2 kernel: same siting;
// 4:1:1 MPEG2 and TOPLEFT -> its left-sited kernel).
template<typename TVideoInfo>
AVS_FORCEINLINE static int chromaLocationToMaskPlacement(int chromaloc, const TVideoInfo& vi) {
  if (vi.Is440())
    return (chromaloc == AVS_CHROMA_TOP_LEFT || chromaloc == AVS_CHROMA_TOP) ? PLACEMENT_TOPLEFT : PLACEMENT_MPEG1;
  switch (chromaloc) {
  case AVS_CHROMA_CENTER:
  case AVS_CHROMA_TOP:
  case AVS_CHROMA_BOTTOM:
    return PLACEMENT_MPEG1;
  case AVS_CHROMA_TOP_LEFT:
    return PLACEMENT_TOPLEFT;
  default: // left, bottom_left, dv
    return PLACEMENT_MPEG2;
  }
}

template<typename TVideoInfo>
AVS_FORCEINLINE static MaskMode resolveChromaMaskMode(int placement, const TVideoInfo& vi) {
  if (vi.Is411())
    return (placement == PLACEMENT_MPEG1) ? MASK411 : MASK411_TOPLEFT;
  if (vi.Is440())
    return (placement == PLACEMENT_MPEG1) ? MASK440 : MASK440_TOPLEFT;
  if (vi.Is410())
    return (placement == PLACEMENT_MPEG1) ? MASK410 : (placement == PLACEMENT_TOPLEFT) ? MASK410_TOPLEFT : MASK410_MPEG2;
  if (vi.Is420())
    return (placement == PLACEMENT_MPEG1) ? MASK420 : (placement == PLACEMENT_TOPLEFT) ? MASK420_TOPLEFT : MASK420_MPEG2;
  if (vi.Is422() || vi.IsYUY2()) // YUY2: packed 4:2:2
    return (placement == PLACEMENT_MPEG1) ? MASK422 : MASK422_MPEG2;
  return MASK444; // Is444() / IsY() / RGB
}

// Effective mask value at chroma position x for the 4:2:0 / 4:2:2 box and mpeg2 modes
// (the separable modes are computed by mask_sep_sum).
// right_value: in/out sliding-window state, used only for MPEG2 modes.
template<MaskMode maskMode, typename pixel_t>
AVS_FORCEINLINE static pixel_t calculate_effective_mask(
  const pixel_t* ptr,
  int x,
  int pitch,
  int& right_value  // in/out for MPEG2 sliding window modes; int to avoid truncation of 2-pixel sums
) {
  static_assert(!MaskIsSeparable<maskMode>, "separable modes: use mask_sep_sum");
  if constexpr (maskMode == MASK444) {
    // +------+
    // | 1.0  |
    // +------+
    return ptr[x];
  }
  else if constexpr (maskMode == MASK420 || maskMode == MASK422) {
    // Generic CENTER box average (see calculate_effective_mask_center above): 4:2:0 (2x2), 4:2:2 (2x1).
    return calculate_effective_mask_center<maskMode, pixel_t>(ptr, x, pitch);
  }
  else if constexpr (maskMode == MASK420_MPEG2) {
    // ------+------+-------+
    // 0.125 | 0.25 | 0.125 |
    // ------|------+-------|
    // 0.125 | 0.25 | 0.125 |
    // ------+------+-------+
    int left = right_value;
    const int mid = ptr[x * 2] + ptr[x * 2 + pitch];
    right_value = ptr[x * 2 + 1] + ptr[x * 2 + 1 + pitch];
    return (left + 2 * mid + right_value + 4) >> 3;
  }
  else if constexpr (maskMode == MASK422_MPEG2) {
    // ------+------+-------+
    // 0.25  | 0.5  | 0.25  |
    // ------+------+-------+
    int left = right_value;
    const int mid = ptr[x * 2];
    right_value = ptr[x * 2 + 1];
    return (left + 2 * mid + right_value + 2) >> 2;
  }
}

// Float version of calculate_effective_mask.
template<MaskMode maskMode>
AVS_FORCEINLINE static float calculate_effective_mask_f(
  const float* ptr,
  int x,
  int pitch,
  float& right_value  // in/out for MPEG2 sliding window modes
) {
  static_assert(!MaskIsSeparable<maskMode>, "separable modes: use mask_sep_sum");
  if constexpr (maskMode == MASK444) {
    return ptr[x];
  }
  else if constexpr (maskMode == MASK420 || maskMode == MASK422) {
    // Generic CENTER box average — see calculate_effective_mask_center_f above.
    return calculate_effective_mask_center_f<maskMode>(ptr, x, pitch);
  }
  else if constexpr (maskMode == MASK420_MPEG2) {
    float left = right_value;
    const float mid = ptr[x * 2] + ptr[x * 2 + pitch];
    right_value = ptr[x * 2 + 1] + ptr[x * 2 + 1 + pitch];
    return (left + 2.0f * mid + right_value) * 0.125f;
  }
  else if constexpr (maskMode == MASK422_MPEG2) {
    float left = right_value;
    const float mid = ptr[x * 2];
    right_value = ptr[x * 2 + 1];
    return (left + 2.0f * mid + right_value) * 0.25f;
  }
}

// Separable modes, a complete row: integer (C reference; the SIMD versions match it bit by bit).
template<MaskMode maskMode, typename pixel_t, bool full_opacity>
static void mask_sep_row_c(const pixel_t* maskp, int mask_pitch, int width, int rows_above, int rows_below,
  pixel_t* dst, int opacity_i, int half, MagicDiv magic)
{
  constexpr int shift = MaskSepShift<maskMode>;
  const pixel_t* rows[8];
  mask_sep_rows<maskMode, pixel_t>(maskp, mask_pitch, rows_above, rows_below, rows);
  const int luma_width = width * MaskHSubsample<maskMode>;
  for (int x = 0; x < width; ++x) {
    const int32_t sum = mask_sep_sum<maskMode, pixel_t, int32_t>(rows, x, luma_width);
    const int avg = (sum + ((1 << shift) >> 1)) >> shift;
    if constexpr (full_opacity)
      dst[x] = (pixel_t)avg;
    else
      dst[x] = static_cast<pixel_t>(magic_div_rt<pixel_t>((uint32_t)avg * (uint32_t)opacity_i + (uint32_t)half, magic));
  }
}

// Separable modes, a complete row: float.
template<MaskMode maskMode, bool full_opacity>
static void mask_sep_row_float_c(const float* maskp, int mask_pitch, int width, int rows_above, int rows_below,
  float* dst, float opacity_f)
{
  constexpr float scale = 1.0f / (float)(1 << MaskSepShift<maskMode>);
  const float* rows[8];
  mask_sep_rows<maskMode, float>(maskp, mask_pitch, rows_above, rows_below, rows);
  const int luma_width = width * MaskHSubsample<maskMode>;
  for (int x = 0; x < width; ++x) {
    const float avg = mask_sep_sum<maskMode, float, float>(rows, x, luma_width) * scale;
    if constexpr (full_opacity)
      dst[x] = avg;
    else
      dst[x] = avg * opacity_f;
  }
}

// Precalculate a complete row of effective mask values.
// integer 8-16 bits version.
//
// full_opacity == true  (default): opacity is 1.0 — mask used as-is.
//   MASK444: returns original maskp pointer directly (no copy, no buffer needed).
//   Others:  fills buffer with spatial averages and returns buffer.data().
//
// full_opacity == false: opacity < 1.0 — bake (avg * opacity_i + half) / max into the buffer.
//   MASK444: copies maskp row scaled by opacity into buffer, returns buffer.data().
//   Others:  fills buffer with spatial-averaged-and-opacity-scaled values.
//   Caller must have allocated effective_mask_buffer (even for MASK444).
//
// mask_pitch is in pixels (caller has divided by sizeof(pixel_t)).
// rows_above / rows_below: mask rows available above / below maskp (mask_rows_above/below), for the
// vertical filters' edge clamping; ignored by the modes without vertical neighbours.
// opacity_i, half, magic are ignored when full_opacity == true.
template<MaskMode maskMode, typename pixel_t, bool full_opacity = true>
AVS_FORCEINLINE static const pixel_t* prepare_effective_mask_for_row(
  const pixel_t* maskp,
  int mask_pitch,
  int width,
  int rows_above,
  int rows_below,
  pixel_t* effective_mask_buffer,
  int opacity_i = 0,
  int half = 0,
  MagicDiv magic = {}
) {
  if constexpr (maskMode == MASK444) {
    if constexpr (full_opacity) {
      return maskp;
    } else {
      for (int x = 0; x < width; ++x)
        effective_mask_buffer[x] = static_cast<pixel_t>(
          magic_div_rt<pixel_t>((uint32_t)maskp[x] * (uint32_t)opacity_i + (uint32_t)half, magic));
      return effective_mask_buffer;
    }
  }
  else if constexpr (MaskIsSeparable<maskMode>) {
    mask_sep_row_c<maskMode, pixel_t, full_opacity>(maskp, mask_pitch, width, rows_above, rows_below,
      effective_mask_buffer, opacity_i, half, magic);
    return effective_mask_buffer;
  }
  else {
    int mask_right = 0;  // int: MASK420_MPEG2 stores a 2-pixel sum (up to 510 for 8-bit), pixel_t would truncate
    if constexpr (maskMode == MASK420_MPEG2)
      mask_right = maskp[0] + maskp[0 + mask_pitch];
    else if constexpr (maskMode == MASK422_MPEG2)
      mask_right = maskp[0];

    for (int x = 0; x < width; ++x) {
      const pixel_t avg = (pixel_t)calculate_effective_mask<maskMode>(maskp, x, mask_pitch, mask_right);
      if constexpr (full_opacity)
        effective_mask_buffer[x] = avg;
      else
        effective_mask_buffer[x] = static_cast<pixel_t>(
          magic_div_rt<pixel_t>((uint32_t)avg * (uint32_t)opacity_i + (uint32_t)half, magic));
    }
    return effective_mask_buffer;
  }
}


// Float version of prepare_effective_mask_for_row.
// full_opacity == true  (default): returns maskp for MASK444; spatial avg for others.
// full_opacity == false: bakes opacity_f into every output element.
//   MASK444: copies row scaled by opacity_f; buffer must be allocated by caller.
// mask_pitch is in floats (caller has divided by sizeof(float)).
// rows_above / rows_below: as in prepare_effective_mask_for_row.
// opacity_f is ignored when full_opacity == true.
template<MaskMode maskMode, bool full_opacity = true>
AVS_FORCEINLINE static const float* prepare_effective_mask_for_row_float_c(
  const float* maskp,
  int mask_pitch,
  int width,
  int rows_above,
  int rows_below,
  float* effective_mask_buffer,
  float opacity_f = 1.0f
) {
  if constexpr (maskMode == MASK444) {
    if constexpr (full_opacity) {
      return maskp;
    } else {
      for (int x = 0; x < width; ++x)
        effective_mask_buffer[x] = maskp[x] * opacity_f;
      return effective_mask_buffer;
    }
  }
  else if constexpr (MaskIsSeparable<maskMode>) {
    mask_sep_row_float_c<maskMode, full_opacity>(maskp, mask_pitch, width, rows_above, rows_below,
      effective_mask_buffer, opacity_f);
    return effective_mask_buffer;
  }
  else {
    float mask_right = 0.0f;
    if constexpr (maskMode == MASK420_MPEG2)
      mask_right = maskp[0] + maskp[0 + mask_pitch];
    else if constexpr (maskMode == MASK422_MPEG2)
      mask_right = maskp[0];

    for (int x = 0; x < width; ++x) {
      const float avg = calculate_effective_mask_f<maskMode>(maskp, x, mask_pitch, mask_right);
      if constexpr (full_opacity)
        effective_mask_buffer[x] = avg;
      else
        effective_mask_buffer[x] = avg * opacity_f;
    }
    return effective_mask_buffer;
  }
}

// ============================================================
// masked_merge inline template implementations
// Templated on maskMode so Layer (luma-res mask, placement-aware) and
// Overlay (mask already at plane resolution, MASK444) share the same code.
//
// For MASK420/MASK420_MPEG2/MASK420_TOPLEFT the mask pointer advances 2 luma rows per output row.
// ============================================================
// ---------------------------------------------------------------------------
// 8-bit row — mask already has opacity baked in by rowprep.
// 32-wide 16-bit mulhi arithmetic (fast, overflow-safe).
// ---------------------------------------------------------------------------
static void blend8_masked_c_row(
  uint8_t* p1, const uint8_t* p2, const uint8_t* mask,
  int width)
{
  constexpr uint32_t half = 127u;
  constexpr uint32_t max_val = 255u;

  int x = 0;
  for (; x < width; ++x) {
    const uint32_t ms = mask[x];
    const uint32_t a = p1[x], b_v = p2[x];
    const uint32_t tr = a * (max_val - ms) + b_v * ms + half;
    p1[x] = (uint8_t)((tr * 0x8081u) >> 23);
  }
}

// ---------------------------------------------------------------------------
// 16-bit row (10, 12, 14, 16 bits) — mask already has opacity baked in.
// 32-bit arithmetic, 8 pixels per step.
// ---------------------------------------------------------------------------
template<int bits_per_pixel>
static void blend16_masked_c_row(
  uint16_t* p1, const uint16_t* p2, const uint16_t* mask,
  int width)
{
  constexpr MagicDiv m_div = get_magic_div(bits_per_pixel);
  constexpr uint32_t max_val = (1u << bits_per_pixel) - 1;
  constexpr uint32_t half = max_val / 2;

  int x = 0;
  for (; x < width; ++x) {
    const uint32_t ms = mask[x];
    const uint32_t a = p1[x];
    p1[x] = (uint16_t)magic_div_rt<uint16_t>(a * (max_val - ms) + (uint32_t)p2[x] * ms + half, m_div);
  }
}

// ---------------------------------------------------------------------------
// float row — mask already has opacity baked in (range 0.0 to 1.0).
// Linear interpolation: p1[x] = p1[x] * (1.0f - mask[x]) + p2[x] * mask[x]
// 8 pixels per step.
// ---------------------------------------------------------------------------
static void blend_masked_float_c_row(
  float* p1, const float* p2, const float* mask, int width)
{
  int x = 0;

  // Scalar tail
  for (; x < width; ++x) {
    const float m = mask[x];
    const float a = p1[x];
    const float b = p2[x];
    p1[x] = a + m * (b - a);
  }
}


// Inner loop: opacity already baked into effective_mask_ptr by rowprep.
template<MaskMode maskMode, bool full_opacity>
static void masked_merge_impl_c_inner(
  BYTE* p1, const BYTE* p2, const BYTE* mask,
  int p1_pitch, int p2_pitch, int mask_pitch,
  int width, int height, int opacity, int bits_per_pixel)
{
  // bits_per_pixel == 8 -> pixel_t is uint8_t, max_pixel_value is 255, magic_div_rt uses mulhi_epu16 path.
  // bits_per_pixel >8 and <= 16 -> pixel_t is uint16_t, max_pixel_value

  const MagicDiv mag = get_magic_div(bits_per_pixel);
  const int max_val = (1 << bits_per_pixel) - 1;
  const int half = max_val / 2;

  if (bits_per_pixel == 8) {
    const uint8_t* maskp = reinterpret_cast<const uint8_t*>(mask);
    const int mpx = mask_pitch;
    const int mask_adv = mpx * MaskVSubsample<maskMode>;

    ScratchBuffer<uint8_t> eff_buf;
    if constexpr (maskMode != MASK444 || !full_opacity) eff_buf.resize(width);

    for (int y = 0; y < height; y++) {
      const uint8_t* eff = prepare_effective_mask_for_row<maskMode, uint8_t, full_opacity>(
        maskp, mpx, width, mask_rows_above(y, MaskVSubsample<maskMode>), mask_rows_below(y, height, MaskVSubsample<maskMode>),
        eff_buf.data(), opacity, half, mag);
      blend8_masked_c_row(
        reinterpret_cast<uint8_t*>(p1), reinterpret_cast<const uint8_t*>(p2), eff, width);
      p1 += p1_pitch; p2 += p2_pitch; maskp += mask_adv;
    }
    return;
  }

  const uint16_t* maskp = reinterpret_cast<const uint16_t*>(mask);
  const int mpx = mask_pitch / 2;
  const int mask_adv = mpx * MaskVSubsample<maskMode>;

  ScratchBuffer<uint16_t> eff_buf;
  if constexpr (maskMode != MASK444 || !full_opacity) eff_buf.resize(width);

#define BLEND16_LOOP_C(bpp) \
  for (int y = 0; y < height; y++) { \
    const uint16_t* eff = prepare_effective_mask_for_row<maskMode, uint16_t, full_opacity>( \
      maskp, mpx, width, mask_rows_above(y, MaskVSubsample<maskMode>), mask_rows_below(y, height, MaskVSubsample<maskMode>), \
      eff_buf.data(), opacity, half, mag); \
    blend16_masked_c_row<bpp>( \
      reinterpret_cast<uint16_t*>(p1), reinterpret_cast<const uint16_t*>(p2), eff, width); \
    p1 += p1_pitch; p2 += p2_pitch; maskp += mask_adv; \
  } break;

  switch (bits_per_pixel) {
  case 10: BLEND16_LOOP_C(10)
  case 12: BLEND16_LOOP_C(12)
  case 14: BLEND16_LOOP_C(14)
  case 16: BLEND16_LOOP_C(16)
  }
#undef BLEND16_LOOP_C
}

// Float version — opacity baked into rowprep, inner loop has no * opacity_f.
template<MaskMode maskMode, bool full_opacity>
AVS_FORCEINLINE static void masked_merge_impl_float_c_inner(
  BYTE* p1, const BYTE* p2, const BYTE* mask,
  int p1_pitch, int p2_pitch, int mask_pitch,
  int width, int height, float opacity)
{
  const float* maskp = reinterpret_cast<const float*>(mask);
  const int mpx = mask_pitch / sizeof(float);
  const int mask_adv = mpx * MaskVSubsample<maskMode>;

  ScratchBuffer<float> eff_buf;
  if constexpr (maskMode != MASK444 || !full_opacity) eff_buf.resize(width);

  for (int y = 0; y < height; y++) {
    const float* eff = prepare_effective_mask_for_row_float_c<maskMode, full_opacity>(
      maskp, mpx, width, mask_rows_above(y, MaskVSubsample<maskMode>), mask_rows_below(y, height, MaskVSubsample<maskMode>),
      eff_buf.data(), opacity);
    blend_masked_float_c_row(
      reinterpret_cast<float*>(p1), reinterpret_cast<const float*>(p2), eff, width);
    p1 += p1_pitch; p2 += p2_pitch; maskp += mask_adv;
  }
}

// ---------------------------------------------------------------------------
// Outer: dispatch on full_opacity (opacity == max_pixel_value) at the call site.
// ---------------------------------------------------------------------------
template<MaskMode maskMode>
AVS_FORCEINLINE static void masked_merge_c_impl(
  BYTE* p1, const BYTE* p2, const BYTE* mask,
  int p1_pitch, int p2_pitch, int mask_pitch,
  int width, int height, int opacity, int bits_per_pixel)
{
  const int max_pixel_value = (1 << bits_per_pixel) - 1;
  if (opacity == max_pixel_value)
    masked_merge_impl_c_inner<maskMode, true>(
      p1, p2, mask, p1_pitch, p2_pitch, mask_pitch, width, height, opacity, bits_per_pixel);
  else
    masked_merge_impl_c_inner<maskMode, false>(
      p1, p2, mask, p1_pitch, p2_pitch, mask_pitch, width, height, opacity, bits_per_pixel);
}

template<MaskMode maskMode>
AVS_FORCEINLINE static void masked_merge_float_c_impl(
  BYTE* p1, const BYTE* p2, const BYTE* mask,
  int p1_pitch, int p2_pitch, int mask_pitch,
  int width, int height, float opacity_f)
{
  if (opacity_f == 1.0f)
    masked_merge_impl_float_c_inner<maskMode, true>(
      p1, p2, mask, p1_pitch, p2_pitch, mask_pitch, width, height, opacity_f);
  else
    masked_merge_impl_float_c_inner<maskMode, false>(
      p1, p2, mask, p1_pitch, p2_pitch, mask_pitch, width, height, opacity_f);
}

/*
 * Family 1: weighted_merge — no mask, flat weight, >> 15 shift
 *   weight + invweight == 32768
 */
using weighted_merge_fn_t = void(
  BYTE* p1, const BYTE* p2,
  int p1_pitch, int p2_pitch,
  int width, int height,
  int weight, int invweight,
  int bits_per_pixel);

using weighted_merge_float_fn_t = void(
  BYTE* p1, const BYTE* p2,
  int p1_pitch, int p2_pitch,
  int width, int height,
  float weight_f);

/*
 * Family 2: masked_merge — mask * opacity via magic_div
 *   opacity is pre-scaled: round(opacity_f * max_pixel_value)
 */
using masked_merge_fn_t = void(
  BYTE* p1, const BYTE* p2, const BYTE* mask,
  int p1_pitch, int p2_pitch, int mask_pitch,
  int width, int height,
  int opacity,
  int bits_per_pixel);

using masked_merge_float_fn_t = void(
  BYTE* p1, const BYTE* p2, const BYTE* mask,
  int p1_pitch, int p2_pitch, int mask_pitch,
  int width, int height,
  float opacity_f);

// Family 1 — C reference

void weighted_merge_return_a_or_b(BYTE* p1, const BYTE* p2,
  int p1_pitch, int p2_pitch,
  int width, int height,
  int weight, int invweight,
  int bits_per_pixel);

void weighted_merge_c(BYTE* p1, const BYTE* p2,
  int p1_pitch, int p2_pitch,
  int width, int height,
  int weight, int invweight,
  int bits_per_pixel);

void weighted_merge_float_c(BYTE* p1, const BYTE* p2,
  int p1_pitch, int p2_pitch,
  int width, int height,
  float weight_f);

// Overlay blend C getters — dispatch on is_chroma × maskMode
masked_merge_fn_t*       get_overlay_blend_masked_fn_c(bool is_chroma, MaskMode maskMode);
masked_merge_float_fn_t* get_overlay_blend_masked_float_fn_c(bool is_chroma, MaskMode maskMode);

using overlay_blend_plane_masked_opacity_t = void(BYTE* p1, const BYTE* p2, const BYTE* mask,
  const int p1_pitch, const int p2_pitch, const int mask_pitch,
  const int width, const int height, const float opacity_f,
  const int bits_per_pixel);

/********************************
 ********* Blend Opaque *********
 ** Use for Lighten and Darken **
 ********************************/
template<typename pixel_t>
AVS_FORCEINLINE pixel_t overlay_blend_opaque_c_core(const pixel_t p1, const pixel_t p2, const pixel_t mask) {
  return (mask) ? p2 : p1;
}

// Mode: Darken/lighten

template<typename pixel_t>
void overlay_darken_c(BYTE *p1Y, BYTE *p1U, BYTE *p1V, const BYTE *p2Y, const BYTE *p2U, const BYTE *p2V, int p1_pitch, int p2_pitch, int width, int height);
template<typename pixel_t>
void overlay_lighten_c(BYTE *p1Y, BYTE *p1U, BYTE *p1V, const BYTE *p2Y, const BYTE *p2U, const BYTE *p2V, int p1_pitch, int p2_pitch, int width, int height);

// ---------------------------------------------------------------------------
// Packed RGBA (RGB32 / RGB64) blend — magic-div arithmetic.
//
// Per-pixel blend weight source:
//   maskp8 == nullptr → alpha is ovrp[x*4+3]  (Add: overlay's own alpha)
//   maskp8 != nullptr → alpha is maskp[x]      (Subtract: original alpha extracted
//                        before pre-inverting overlay in Layer::Create)
//
//   alpha_eff = (alpha_src * opacity_i + half) / max_val
//   result_ch = (dst_ch * (max - alpha_eff) + ovr_ch * alpha_eff + half) / max_val
//
// For Subtract, the overlay is pre-inverted in Layer::Create, so ovr_ch is already
// (max_val - original_ch) — no subtract logic is needed inside the kernel.
//
// opacity_i is in [0..max_pixel_value], the same convention as masked_merge.
// This is the interleaved analogue of masked_merge_impl for MASK444 planar
// data, using the correct ÷max_val formula so that all mask values [0..max]
// map linearly to the blend range without endpoint asymmetry.
// ---------------------------------------------------------------------------
template<typename pixel_t, bool has_separate_mask>
static void masked_blend_packedrgba_c(
  BYTE* dstp8, const BYTE* ovrp8, const BYTE* maskp8,
  int dst_pitch, int ovr_pitch, int mask_pitch,
  int width, int height, int opacity_i)
{
  pixel_t* dstp        = reinterpret_cast<pixel_t*>(dstp8);
  const pixel_t* ovrp  = reinterpret_cast<const pixel_t*>(ovrp8);
  const pixel_t* maskp = has_separate_mask ? reinterpret_cast<const pixel_t*>(maskp8) : nullptr;
  dst_pitch  /= sizeof(pixel_t);
  ovr_pitch  /= sizeof(pixel_t);
  if constexpr (has_separate_mask)
    mask_pitch /= sizeof(pixel_t);

  // For packed RGB32/64, only 8 and 16 bpc exist; bpp derives from pixel_t.
  constexpr uint32_t max_val = sizeof(pixel_t) == 1 ? 255u : 65535u;
  constexpr uint32_t half    = max_val / 2u;
  constexpr int bpp          = sizeof(pixel_t) == 1 ? 8 : 16;
  const MagicDiv magic       = get_magic_div(bpp);

  // uint32_t arithmetic is safe: the maximum sum a*inv + b*alpha is bounded
  // by max_val^2 = 65535^2 = 4,294,836,225 < UINT32_MAX; adding half keeps
  // it within range.  The 8-bit case is well within 32-bit even before that.

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const uint32_t alpha_src = has_separate_mask ? (uint32_t)maskp[x] : (uint32_t)ovrp[x * 4 + 3];
      const uint32_t alpha_eff = (uint32_t)magic_div_rt<pixel_t>(
        alpha_src * (uint32_t)opacity_i + half, magic);
      const uint32_t inv_alpha = max_val - alpha_eff;

      for (int ch = 0; ch < 4; ++ch) {
        dstp[x * 4 + ch] = (pixel_t)magic_div_rt<pixel_t>(
          (uint32_t)dstp[x * 4 + ch] * inv_alpha + (uint32_t)ovrp[x * 4 + ch] * alpha_eff + half, magic);
      }
    }
    dstp += dst_pitch;
    ovrp += ovr_pitch;
    if constexpr (has_separate_mask) maskp += mask_pitch;
  }
}

#endif // __blend_common_h
