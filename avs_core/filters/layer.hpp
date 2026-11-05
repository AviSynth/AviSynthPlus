
// Basic utils in C++ language, to be included in both base and processor specific (e.g. avx2) source modules, where they can
// be optimized for the specific instruction set.
// yuv add, subtract, mul, lighten, darken
// Chroma placement helpers (calculate_effective_mask*, prepare_effective_mask_for_row*)
// are now in overlay/blend_common.h, pulled in via layer.h.

// ---------------------------------------------------------------------------
// Rowprep function selectors — defined by the including TU to inject SIMD variants.
// Default: C scalar functions from overlay/blend_common.h.
//
// LAYER_ROWPREP_FN      — spatial mask averaging (+ opacity baking when
//                         full_opacity=false.
//                         Used for the full_opacity=true path: returns only spatial
//                         averages (or maskp directly for MASK444).
//
// Including TU (e.g. layer_avx2.cpp) defines them before this #include:
//   #define LAYER_ROWPREP_FN       prepare_effective_mask_for_row_avx2
// to allow the same C template to compile with arch-spcific different rowprep variants 
// ---------------------------------------------------------------------------
#include "../core/scratch_buffer.h" // ScratchBuffer for the row buffers

#ifndef LAYER_ROWPREP_FN
#  define LAYER_ROWPREP_FN  prepare_effective_mask_for_row
#endif

// ---------------------------------------------------------------------------
// LAYER_ROW_KERNELS — SIMD row kernels for the pixel loops below, also defined by the including TU
// (e.g. layer_neon.cpp). Each kernel processes the first n pixels of a row (n <= width, the same
// result as the C loop) and returns n; the C loop then does the rest of the row.
// Pointers are in pixels, the same ones the C loop uses. Default: no kernels, the C loop does the whole row.
// ---------------------------------------------------------------------------
#ifndef LAYER_ROW_KERNELS
struct LayerRowKernelsC {
  template<typename pixel_t, bool is_chroma, bool has_alpha>
  static int yuv_mul(pixel_t*, const pixel_t*, const pixel_t*, int, int, int, int, MagicDiv) { return 0; }
  template<int mode, typename pixel_t, bool lumaonly, bool has_alpha, bool update_luma>
  static int yuv_lighten_darken(pixel_t*, pixel_t*, pixel_t*, const pixel_t*, const pixel_t*, const pixel_t*,
    const pixel_t*, const pixel_t*, const pixel_t*, int, int, int, int, int, MagicDiv) { return 0; }
  template<int mode, bool lumaonly, bool has_alpha, bool update_luma>
  static int yuv_lighten_darken_f(float*, float*, float*, const float*, const float*, const float*,
    const float*, const float*, const float*, int, float, float) { return 0; }
  template<int mode, typename pixel_t, bool has_alpha, bool blend_alpha>
  static int planarrgb_lighten_darken(pixel_t*, pixel_t*, pixel_t*, pixel_t*, const pixel_t*, const pixel_t*, const pixel_t*,
    const pixel_t*, const pixel_t*, int, int, int, int, int, MagicDiv) { return 0; }
  template<int mode, bool has_alpha, bool blend_alpha>
  static int planarrgb_lighten_darken_f(float*, float*, float*, float*, const float*, const float*, const float*,
    const float*, const float*, int, float, float) { return 0; }
  template<typename pixel_t, bool chroma, bool has_alpha, bool blend_alpha>
  static int planarrgb_mul(pixel_t*, pixel_t*, pixel_t*, pixel_t*, const pixel_t*, const pixel_t*, const pixel_t*,
    const pixel_t*, const pixel_t*, int, int, int, int, MagicDiv) { return 0; }
  template<typename pixel_t, bool lumaonly, bool has_alpha, bool update_luma>
  static int yuv_mulovr(pixel_t*, pixel_t*, pixel_t*, const pixel_t*, const pixel_t*, int, int, int, int, MagicDiv) { return 0; }
};
#  define LAYER_ROW_KERNELS  LayerRowKernelsC
#endif

DISABLE_WARNING_PUSH
DISABLE_WARNING_UNREFERENCED_LOCAL_VARIABLE

// YUV(A) mul 8-16 bits
// full_opacity == true : rowprep returns mask-average; subsample-aware avg mask is put into effective_mask_buffer[x]. 444 case: returns maskp directly.
// full_opacity == false: rowprep pre-calculates with 'opacity' in the formulae, effective_mask_buffer is filled even for 444.
template<MaskMode maskMode, typename pixel_t, bool is_chroma, bool has_alpha, bool full_opacity>
static void layer_yuv_mul_c_inner(BYTE* dstp8, const BYTE* ovrp8, const BYTE* maskp8, int dst_pitch, int overlay_pitch, int mask_pitch, int width, int height, int opacity_i, int bits_per_pixel) {
  pixel_t* dstp = reinterpret_cast<pixel_t*>(dstp8);
  const pixel_t* ovrp = reinterpret_cast<const pixel_t*>(ovrp8);
  const pixel_t* maskp = has_alpha ? reinterpret_cast<const pixel_t*>(maskp8) : nullptr;
  dst_pitch /= sizeof(pixel_t);
  overlay_pitch /= sizeof(pixel_t);
  if constexpr (has_alpha)
    mask_pitch /= sizeof(pixel_t);

  if constexpr (sizeof(pixel_t) == 1)
    bits_per_pixel = 8; // make quasi constexpr

  const MagicDiv magic = get_magic_div(bits_per_pixel);
  const int max_pixel_value = (1 << bits_per_pixel) - 1;
  const int half = max_pixel_value / 2;

  // Buffer: needed for subsampled spatial averaging, or MASK444+!full_opacity baking.
  ScratchBuffer<pixel_t> effective_mask_buffer;
  if constexpr (has_alpha && (maskMode != MASK444 || !full_opacity)) {
    effective_mask_buffer.resize(width);
  }

  for (int y = 0; y < height; ++y) {
    const pixel_t* effective_mask_ptr = nullptr;

    // Rowprep: full_opacity path returns spatial avg (or maskp for MASK444).
    //          !full_opacity path bakes (avg * opacity_i) / div
    if constexpr (has_alpha) {
      if constexpr (full_opacity)
        effective_mask_ptr = LAYER_ROWPREP_FN<maskMode, pixel_t, true>(maskp, mask_pitch, width, mask_rows_above(y, MaskVSubsample<maskMode>), mask_rows_below(y, height, MaskVSubsample<maskMode>), effective_mask_buffer.data());
      else
        effective_mask_ptr = LAYER_ROWPREP_FN<maskMode, pixel_t, false>(maskp, mask_pitch, width, mask_rows_above(y, MaskVSubsample<maskMode>), mask_rows_below(y, height, MaskVSubsample<maskMode>), effective_mask_buffer.data(), opacity_i, half, magic);
    }

    // Main blending loop — alpha_mask is fully prepared by rowprep.
    // SIMD row kernel: does pixels [0, x) and returns x; 0 when there is none (C build). The C loop does the rest.
    int x = LAYER_ROW_KERNELS::yuv_mul<pixel_t, is_chroma, has_alpha>(dstp, ovrp, effective_mask_ptr, width, opacity_i, max_pixel_value, half, magic);
    for (; x < width; ++x) {
      // has_alpha=true: rowprep baked level in (!full_opacity) or returned avg (full_opacity).
      // has_alpha=false: flat level, no mask.
      const uint32_t alpha_mask = has_alpha ? effective_mask_ptr[x] : opacity_i;
      const uint32_t inv_alpha = max_pixel_value - alpha_mask;
      uint32_t target_pixel;

      if constexpr (!is_chroma) {
        // luma: blend towards the "Multiplied" product. No explicit "+half" round-to-nearest
        // is added here (unlike the alpha blend below). This calculates as floor(A*B/max), not a
        // round(A*B/max). magic_div_rt (see blend_common.h) divides exactly like a true integer
        // division by max_pixel_value, unlike the old >> bits_per_pixel shift it replaced.
        // This keeps the max*max=max rule (e.g. 255*255 -> 255 instead of 254).
        //
        // No Intel SIMD implementation of "Mul" (get_layer_yuv_mul_functions_avx2 in layer_avx2.cpp
        // just includes this scalar path, compiled possibly into a more optimized code). NEON: 10-16 bit
        // row kernel (LAYER_ROW_KERNELS::yuv_mul, layer_row_kernels_neon.h).
        const pixel_t prod = (pixel_t)magic_div_rt<pixel_t>((uint32_t)ovrp[x] * (uint32_t)dstp[x], magic);
        target_pixel = prod;
        // was: dstp[x] = (pixel_t)(dstp[x] + ((((((calc_t)ovrp[x] * dstp[x]) >> bits_per_pixel) - dstp[x]) * alpha_mask) >> bits_per_pixel));
      }
      else {
        // Note: when use_chrome=false, ovrp is prefilled with neutral, so the same formula applies for both modes
        // CHROMA (Normal): Blend towards the overlay chroma (use_chroma=true) or towards neutral (use_chroma=false).
        // Instead of dst + ((ovr - dst) * alpha) >> bits we use ((dst * (max - alpha)) + (ovr * alpha)) / max
        // U = U + ( ((Uovr - U)*level) >> 8 )
        // V = V + ( ((Vovr - V)*level) >> 8 )
        target_pixel = (uint32_t)ovrp[x];
        // for use_chroma = false, ovrp is prefilled with neutral
        // was: dstp[x] = (pixel_t)(dstp[x] + (((calc_t)(ovrp[x] - dstp[x]) * alpha_mask) >> bits_per_pixel));
        // The YUY2 MMX heritage (cargo cult programming) of halving the strength is finally eliminated
        /*
        const int half = 1 << (bits_per_pixel - 1);
        dstp[x] = (pixel_t)(dstp[x] + (((calc_t)(half - dstp[x]) * (alpha_mask / 2)) >> bits_per_pixel));
        // U = U + ( ((128 - U)*(level/2)) >> 8 )
        // V = V + ( ((128 - V)*(level/2)) >> 8 )
        */
      }

      dstp[x] = (pixel_t)magic_div_rt<pixel_t>((uint32_t)dstp[x] * inv_alpha + target_pixel * alpha_mask + half, magic);
    }
    dstp += dst_pitch;
    ovrp += overlay_pitch;
    if constexpr (has_alpha) {
      maskp += mask_pitch * MaskVSubsample<maskMode>;
    }
  }
}

// Outer dispatcher: full_opacity branch.
template<MaskMode maskMode, typename pixel_t, bool is_chroma, bool has_alpha>
static void layer_yuv_mul_c(BYTE* dstp8, const BYTE* ovrp8, const BYTE* maskp8, int dst_pitch, int overlay_pitch, int mask_pitch, int width, int height, int opacity_i, int bits_per_pixel) {
  if constexpr (!has_alpha) {
    // No mask; full_opacity choice is irrelevant; use true to skip dead buffer allocation.
    layer_yuv_mul_c_inner<maskMode, pixel_t, is_chroma, false, true>(
      dstp8, ovrp8, maskp8, dst_pitch, overlay_pitch, mask_pitch, width, height, opacity_i, bits_per_pixel);
  } else {
    const int max_pixel_value = (1 << bits_per_pixel) - 1;
    if (opacity_i >= max_pixel_value) // full opacity
      layer_yuv_mul_c_inner<maskMode, pixel_t, is_chroma, true, true>(
        dstp8, ovrp8, maskp8, dst_pitch, overlay_pitch, mask_pitch, width, height, opacity_i, bits_per_pixel);
    else
      layer_yuv_mul_c_inner<maskMode, pixel_t, is_chroma, true, false>(
        dstp8, ovrp8, maskp8, dst_pitch, overlay_pitch, mask_pitch, width, height, opacity_i, bits_per_pixel);
  }
}

// YUV(A) mul 32 bits
template<MaskMode maskMode, bool is_chroma, bool has_alpha>
static void layer_yuv_mul_f_c(BYTE* dstp8, const BYTE* ovrp8, const BYTE* maskp8, int dst_pitch, int overlay_pitch, int mask_pitch, int width, int height, float opacity) {
  float* dstp = reinterpret_cast<float*>(dstp8);
  const float* ovrp = reinterpret_cast<const float*>(ovrp8);
  const float* maskp = reinterpret_cast<const float*>(maskp8);
  dst_pitch /= sizeof(float);
  overlay_pitch /= sizeof(float);
  mask_pitch /= sizeof(float);

  // precalculate mask buffer
  ScratchBuffer<float> effective_mask_buffer;
  if constexpr (has_alpha && maskMode != MASK444) {
    effective_mask_buffer.resize(width);
  }

  for (int y = 0; y < height; ++y) {
    const float* effective_mask_ptr = nullptr;

    // precalculate effective mask for this row
    if constexpr (has_alpha) {
      effective_mask_ptr = prepare_effective_mask_for_row_float_c<maskMode>(maskp, mask_pitch, width, mask_rows_above(y, MaskVSubsample<maskMode>), mask_rows_below(y, height, MaskVSubsample<maskMode>), effective_mask_buffer.data());
    }

    // blending loop
    for (int x = 0; x < width; ++x) {
      float effective_mask = has_alpha ? effective_mask_ptr[x] : 0.0f;
      float alpha_mask = has_alpha ? effective_mask * opacity : opacity;

      if constexpr (!is_chroma)
        dstp[x] = dstp[x] + (ovrp[x] * dstp[x] - dstp[x]) * alpha_mask;
      else {
        // Note: when use_chrome=false, ovrp is prefilled with neutral, so the same formula applies for both modes
        dstp[x] = dstp[x] + (ovrp[x] - dstp[x]) * alpha_mask;
      }
    }
    dstp += dst_pitch;
    ovrp += overlay_pitch;
    if constexpr (has_alpha) {
      maskp += mask_pitch * MaskVSubsample<maskMode>;
    }
  }
}

/* Comparison of Multiply (Mul) blend mode: Overlay vs. Layer
 * Luma (Y): Compatible. Both use standard (Base * Overlay) / Max logic.
 * Chroma (UV): Incompatible. They utilize different "intent" for color:
 * - Overlay (Legacy): Dynamic desaturation. Base chroma is pulled toward
 * neutral (128/0.0) scaled by the specific Overlay Luma intensity (O_Y).
 * Formula: Result_UV = (Base_UV * O_Y + Neutral * (Max - O_Y)) / Max
 * - Layer (use_chroma=false): Static desaturation. Ignores overlay pixels
 * entirely; blends toward a fixed neutral point via global opacity/mask.
 * Formula: Result_UV = Base_UV + (Neutral - Base_UV) * Mask_Opacity
 * Summary: Overlay treats darkness as a "color vacuum" (luma-dependent),
 * while Layer treats Y and UV as independent channels.
 * The function would need all plane parameters and work like lighten/darken instead of
 * template<MaskMode maskMode, bool is_chroma, bool has_alpha>
 * static void layer_yuv_mul_f_c(BYTE* dstp8, const BYTE* ovrp8, const BYTE* maskp8, int dst_pitch, int overlay_pitch, int mask_pitch, int width, int height, float opacity) {
 * like this (?):
 * template<int mode, MaskMode maskMode, typename pixel_t, bool lumaonly, bool has_alpha>
 * static void layer_yuv_mul_overlaystyle_f_c(
 */

// Unlike RGBA version, YUVA does not update destination alpha
// so no "blend_alpha" template parameter like at planar rgb lighten_darken

// Note, that we always call one instance of the filter: the lumaonly==false version,
// since the filter handles U and V first, then recursively calls itself with lumaonly==true;
// the chroma part needs unaltered luma for decision, so we delay Y processing.
// Except Y-only greayscale input, where only luma plane exists.
template<int mode, MaskMode maskMode, typename pixel_t, bool lumaonly, bool has_alpha>
static void layer_yuv_lighten_darken_c(
  BYTE* dstp8, BYTE* dstp8_u, BYTE* dstp8_v,/* BYTE* dstp8_a,*/
  const BYTE* ovrp8, const BYTE* ovrp8_u, const BYTE* ovrp8_v, const BYTE* maskp8,
  int dst_pitch, int dst_pitchUV,
  int overlay_pitch, int overlay_pitchUV,
  int mask_pitch,
  int width, int height, int opacity_i, int thresh,
  int bits_per_pixel) {

  pixel_t* dstp = reinterpret_cast<pixel_t*>(dstp8);
  pixel_t* dstp_u = reinterpret_cast<pixel_t*>(dstp8_u);
  pixel_t* dstp_v = reinterpret_cast<pixel_t*>(dstp8_v);
  // pixel_t* dstp_a = reinterpret_cast<pixel_t *>(dstp8_a); // not destination alpha update

  const pixel_t* ovrp = reinterpret_cast<const pixel_t*>(ovrp8);
  const pixel_t* ovrp_u = reinterpret_cast<const pixel_t*>(ovrp8_u);
  const pixel_t* ovrp_v = reinterpret_cast<const pixel_t*>(ovrp8_v);
  const pixel_t* maskp = has_alpha ? reinterpret_cast<const pixel_t*>(maskp8) : nullptr;

  dst_pitch /= sizeof(pixel_t);
  dst_pitchUV /= sizeof(pixel_t);
  overlay_pitch /= sizeof(pixel_t);
  overlay_pitchUV /= sizeof(pixel_t);
  if constexpr (has_alpha)
    mask_pitch /= sizeof(pixel_t);

  const int cwidth = width / MaskHSubsample<maskMode>;
  const int cheight = height / MaskVSubsample<maskMode>;

  // In lighten/darken we need 3 buffers:
  ScratchBuffer<pixel_t> ovr_buffer;
  ScratchBuffer<pixel_t> src_buffer;
  ScratchBuffer<pixel_t> mask_buffer;

  // precalculate mask buffer et al.
  if constexpr (maskMode != MASK444) {
    ovr_buffer.resize(cwidth);
    src_buffer.resize(cwidth);
    if constexpr (has_alpha)
      mask_buffer.resize(cwidth);
  }

  if constexpr (sizeof(pixel_t) == 1)
    bits_per_pixel = 8; // make quasi constexpr

  const MagicDiv magic = get_magic_div(bits_per_pixel);
  const int max_pixel_value = (1 << bits_per_pixel) - 1;
  const int half = max_pixel_value / 2;

  // for subsampled color spaces first do chroma, luma is only used for decision
  // second pass will do luma only
  for (int y = 0; y < cheight; ++y) {

    // Prepare all three pointers using the helper, full-opacity mode, no opacity_i-magicdiv baking
    const pixel_t* ovr_ptr = LAYER_ROWPREP_FN<maskMode, pixel_t>(ovrp, overlay_pitch, cwidth, mask_rows_above(y, MaskVSubsample<maskMode>), mask_rows_below(y, cheight, MaskVSubsample<maskMode>), ovr_buffer.data());
    const pixel_t* src_ptr = LAYER_ROWPREP_FN<maskMode, pixel_t>(dstp, dst_pitch, cwidth, mask_rows_above(y, MaskVSubsample<maskMode>), mask_rows_below(y, cheight, MaskVSubsample<maskMode>), src_buffer.data());
    const pixel_t* effective_mask_ptr = nullptr;
    if constexpr (has_alpha) {
      effective_mask_ptr = LAYER_ROWPREP_FN<maskMode, pixel_t>(maskp, mask_pitch, cwidth, mask_rows_above(y, MaskVSubsample<maskMode>), mask_rows_below(y, cheight, MaskVSubsample<maskMode>), mask_buffer.data());
    }

    // SIMD row kernel: does pixels [0, x) and returns x; 0 when there is none (C build). The C loop does the rest.
    int x = LAYER_ROW_KERNELS::yuv_lighten_darken<mode, pixel_t, lumaonly, has_alpha, maskMode == MASK444>(
      dstp, dstp_u, dstp_v, ovrp, ovrp_u, ovrp_v, ovr_ptr, src_ptr, effective_mask_ptr,
      cwidth, opacity_i, thresh, max_pixel_value, half, magic);
    // ovr/src/mask row pointers + 2 destination and 2 overlay chroma planes (+ luma for 444): too many pointer
    // pairs for GCC's runtime alias checks. Safe: x reads and writes element x of each row only.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC ivdep
#endif
    for (; x < cwidth; ++x) {
      // opacity_i is in [0..max_pixel_value], the same convention as masked_merge.
      uint32_t alpha_eff;
      if constexpr (has_alpha) {
        const uint32_t alpha_src = (uint32_t)effective_mask_ptr[x];
        alpha_eff = (uint32_t)magic_div_rt<pixel_t>(alpha_src * (uint32_t)opacity_i + half, magic);
      }
      else {
        alpha_eff = opacity_i;
      }

      int ovr = ovr_ptr[x];
      int src = src_ptr[x];

      // If the threshold isn't met, the overlay weight becomes 0 (no change to dst)
      if constexpr (mode == LIGHTEN) {
        if (!(ovr > (src + thresh))) alpha_eff = 0;
      }
      else { // DARKEN
        if (!(ovr < (src - thresh))) alpha_eff = 0;
      }

      const uint32_t inv_alpha = max_pixel_value - alpha_eff;

      // formula: dst = (dst * inv_alpha + ovr * alpha_eff) / max_val
      if constexpr (!lumaonly)
      {
        // chroma u,v
        dstp_u[x] = (pixel_t)magic_div_rt<pixel_t>((uint32_t)dstp_u[x] * inv_alpha + (uint32_t)ovrp_u[x] * alpha_eff + half, magic);
        dstp_v[x] = (pixel_t)magic_div_rt<pixel_t>((uint32_t)dstp_v[x] * inv_alpha + (uint32_t)ovrp_v[x] * alpha_eff + half, magic);
      }

      // for 444: update here, width/height is the same as for chroma
      if constexpr (maskMode == MASK444)
        dstp[x] = (pixel_t)magic_div_rt<pixel_t>((uint32_t)dstp[x] * inv_alpha + (uint32_t)ovrp[x] * alpha_eff + half, magic);
    }
    dstp += dst_pitch * MaskVSubsample<maskMode>;
    ovrp += overlay_pitch * MaskVSubsample<maskMode>;
    if constexpr (has_alpha) {
      //dstp_a += dst_pitch * MaskVSubsample<maskMode>;
      maskp += mask_pitch * MaskVSubsample<maskMode>;
    }

    if constexpr (!lumaonly) {
      dstp_u += dst_pitchUV;
      dstp_v += dst_pitchUV;
      ovrp_u += overlay_pitchUV;
      ovrp_v += overlay_pitchUV;
    }

  }

  dst_pitch *= sizeof(pixel_t);
  dst_pitchUV *= sizeof(pixel_t);
  overlay_pitch *= sizeof(pixel_t);
  overlay_pitchUV *= sizeof(pixel_t);
  mask_pitch *= sizeof(pixel_t);

  // Phase #2 recursively call the same function, but with lumaonly=true to update luma plane using the original chroma planes for decision
  // make luma
  if constexpr (!lumaonly && maskMode != MASK444)
    layer_yuv_lighten_darken_c<mode, MASK444, pixel_t, true /* lumaonly*/, has_alpha>(
      dstp8, dstp8_u, dstp8_v, //dstp8_a,
      ovrp8, ovrp8_u, ovrp8_v, maskp8,
      dst_pitch, dst_pitchUV, overlay_pitch, overlay_pitchUV, mask_pitch,
      width, height, opacity_i, thresh, bits_per_pixel);
}

template<int mode, MaskMode maskMode, bool lumaonly, bool has_alpha>
static void layer_yuv_lighten_darken_f_c(
  BYTE* dstp8, BYTE* dstp8_u, BYTE* dstp8_v /*, BYTE* dstp8_a*/,
  const BYTE* ovrp8, const BYTE* ovrp8_u, const BYTE* ovrp8_v, const BYTE* maskp8,
  int dst_pitch, int dst_pitchUV,
  int overlay_pitch, int overlay_pitchUV,
  int mask_pitch,
  int width, int height, float opacity, float thresh) {

  float* dstp = reinterpret_cast<float*>(dstp8);
  float* dstp_u = reinterpret_cast<float*>(dstp8_u);
  float* dstp_v = reinterpret_cast<float*>(dstp8_v);
  //float* dstp_a = reinterpret_cast<float *>(dstp8_a);

  const float* ovrp = reinterpret_cast<const float*>(ovrp8);
  const float* ovrp_u = reinterpret_cast<const float*>(ovrp8_u);
  const float* ovrp_v = reinterpret_cast<const float*>(ovrp8_v);
  const float* maskp =  has_alpha ? reinterpret_cast<const float*>(maskp8) : nullptr;

  dst_pitch /= sizeof(float);
  dst_pitchUV /= sizeof(float);
  overlay_pitch /= sizeof(float);
  overlay_pitchUV /= sizeof(float);
  if constexpr (has_alpha)
    mask_pitch /= sizeof(float);

  const int cwidth = width / MaskHSubsample<maskMode>;
  const int cheight = height / MaskVSubsample<maskMode>;

  // In lighten/darken we need 3 buffers:
  ScratchBuffer<float> ovr_buffer;
  ScratchBuffer<float> src_buffer;
  ScratchBuffer<float> mask_buffer;

  // precalculate mask buffer et al.
  if constexpr (maskMode != MASK444) {
    ovr_buffer.resize(cwidth);
    src_buffer.resize(cwidth);
    if constexpr (has_alpha)
      mask_buffer.resize(cwidth);
  }

  // for subsampled color spaces first do chroma, because luma is used for decision
  // second pass will do luma only
  for (int y = 0; y < cheight; ++y) {
    const float* ovr_ptr = prepare_effective_mask_for_row_float_c<maskMode>(ovrp, overlay_pitch, cwidth, mask_rows_above(y, MaskVSubsample<maskMode>), mask_rows_below(y, cheight, MaskVSubsample<maskMode>), ovr_buffer.data());
    const float* src_ptr = prepare_effective_mask_for_row_float_c<maskMode>(dstp, dst_pitch, cwidth, mask_rows_above(y, MaskVSubsample<maskMode>), mask_rows_below(y, cheight, MaskVSubsample<maskMode>), src_buffer.data());
    const float* effective_mask_ptr = nullptr;
    if constexpr (has_alpha) {
      effective_mask_ptr = prepare_effective_mask_for_row_float_c<maskMode>(maskp, mask_pitch, cwidth, mask_rows_above(y, MaskVSubsample<maskMode>), mask_rows_below(y, cheight, MaskVSubsample<maskMode>), mask_buffer.data());
    }

    // SIMD row kernel: does pixels [0, x) and returns x; 0 when there is none (C build). The C loop does the rest.
    int x = LAYER_ROW_KERNELS::yuv_lighten_darken_f<mode, lumaonly, has_alpha, maskMode == MASK444>(
      dstp, dstp_u, dstp_v, ovrp, ovrp_u, ovrp_v, ovr_ptr, src_ptr, effective_mask_ptr, cwidth, opacity, thresh);
    // see layer_yuv_lighten_darken_c
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC ivdep
#endif
    for (; x < cwidth; ++x) {
      float alpha_eff = has_alpha ? effective_mask_ptr[x] * opacity : opacity;

      float ovr = ovr_ptr[x];
      float src = src_ptr[x];
      if constexpr (mode == LIGHTEN) {
        if (!(ovr > (src + thresh))) alpha_eff = 0;
      }
      else {// DARKEN
        if (!(ovr < (src - thresh))) alpha_eff = 0;
      }

      if constexpr (!lumaonly)
      {
        // chroma u,v
        dstp_u[x] = dstp_u[x] + (ovrp_u[x] - dstp_u[x]) * alpha_eff;
        dstp_v[x] = dstp_v[x] + (ovrp_v[x] - dstp_v[x]) * alpha_eff;
        //dstp_a[x] = dstp_a[x] + (maskp[x] - dstp_a[x]) * alpha_eff;
      }

      // for 444: update here, width/height is the same as for chroma
      if constexpr (maskMode == MASK444)
        dstp[x] = dstp[x] + (ovrp[x] - dstp[x]) * alpha_eff;
    }
    dstp += dst_pitch * MaskVSubsample<maskMode>;
    ovrp += overlay_pitch * MaskVSubsample<maskMode>;
    if constexpr (has_alpha) {
      //dstp_a += dst_pitch * MaskVSubsample<maskMode>;
      maskp += mask_pitch * MaskVSubsample<maskMode>;
    }

    if constexpr (!lumaonly) {
      dstp_u += dst_pitchUV;
      dstp_v += dst_pitchUV;
      ovrp_u += overlay_pitchUV;
      ovrp_v += overlay_pitchUV;
    }
  }

  dst_pitch *= sizeof(float);
  dst_pitchUV *= sizeof(float);
  overlay_pitch *= sizeof(float);
  overlay_pitchUV *= sizeof(float);
  if constexpr (has_alpha)
    mask_pitch *= sizeof(float);

  // make luma
  // recursively call the same function, but with lumaonly=true to update luma plane using the original chroma planes for decision
  if constexpr (!lumaonly && maskMode != MASK444)
    layer_yuv_lighten_darken_f_c<mode, MASK444, true /* lumaonly*/, has_alpha>(
      dstp8, dstp8_u, dstp8_v, //dstp8_a,
      ovrp8, ovrp8_u, ovrp8_v, maskp8,
      dst_pitch, dst_pitchUV, overlay_pitch, overlay_pitchUV, mask_pitch,
      width, height, opacity, thresh);
}

DISABLE_WARNING_POP

// Dispatchers

static void get_layer_yuv_lighten_darken_functions(bool isLighten, int placement, VideoInfo& vi, int bits_per_pixel,
  /*out*/layer_yuv_lighten_darken_c_t** layer_fn,
  /*out*/layer_yuv_lighten_darken_f_c_t** layer_f_fn)
{

#define YUV_LIGHTEN_DARKEN_DISPATCH(L_or_D, MaskType, lumaonly, has_alpha) \
      { if (bits_per_pixel == 8) \
        *layer_fn = layer_yuv_lighten_darken_c<L_or_D, MaskType, uint8_t, lumaonly /*lumaonly*/, has_alpha /*has_alpha*/>; \
      else if (bits_per_pixel <= 16) \
        *layer_fn = layer_yuv_lighten_darken_c<L_or_D, MaskType, uint16_t, lumaonly /*lumaonly*/, has_alpha /*has_alpha*/>; \
      else /* float */ \
        *layer_f_fn = layer_yuv_lighten_darken_f_c<L_or_D, MaskType, lumaonly /*lumaonly*/, has_alpha /*has_alpha*/>; \
}

  // Note, that we always call one instance of the filter: the lumaonly==false version,
  // since the filter handles U and V first, then recursively calls itself with lumaonly==true;
  // the chroma part needs unaltered luma for decision, so we delay Y processing.
  // Except Y-only greayscale input, where only luma plane exists.

  if (isLighten) {

    if (vi.Is411())
    {
      // 4:1:1 has horizontal subsampling only: centred (MPEG1) or left-sited (left, top_left).
      if (placement == PLACEMENT_MPEG1)
        YUV_LIGHTEN_DARKEN_DISPATCH(LIGHTEN, MASK411, false, false)
      else
        YUV_LIGHTEN_DARKEN_DISPATCH(LIGHTEN, MASK411_TOPLEFT, false, false)
    }
    else if (vi.Is440())
    {
      // 4:4:0 has vertical subsampling only: top-sited (top_left, top) or centred (all others).
      if (placement == PLACEMENT_MPEG1)
        YUV_LIGHTEN_DARKEN_DISPATCH(LIGHTEN, MASK440, false, false)
      else
        YUV_LIGHTEN_DARKEN_DISPATCH(LIGHTEN, MASK440_TOPLEFT, false, false)
    }
    else if (vi.Is410())
    {
      if (placement == PLACEMENT_MPEG1)
        YUV_LIGHTEN_DARKEN_DISPATCH(LIGHTEN, MASK410, false, false)
      else if (placement == PLACEMENT_TOPLEFT)
        YUV_LIGHTEN_DARKEN_DISPATCH(LIGHTEN, MASK410_TOPLEFT, false, false)
      else // PLACEMENT_MPEG2
        YUV_LIGHTEN_DARKEN_DISPATCH(LIGHTEN, MASK410_MPEG2, false, false)
    }
    else if (vi.Is420())
    {
      if (placement == PLACEMENT_MPEG1)
        YUV_LIGHTEN_DARKEN_DISPATCH(LIGHTEN, MASK420, false, false)
      else if (placement == PLACEMENT_TOPLEFT)
        YUV_LIGHTEN_DARKEN_DISPATCH(LIGHTEN, MASK420_TOPLEFT, false, false)
      else
        YUV_LIGHTEN_DARKEN_DISPATCH(LIGHTEN, MASK420_MPEG2, false, false)
        // PLACEMENT_MPEG2
    }
    else if (vi.Is422())
    {
      if (placement == PLACEMENT_MPEG1)
        YUV_LIGHTEN_DARKEN_DISPATCH(LIGHTEN, MASK422, false, false)
      else
        YUV_LIGHTEN_DARKEN_DISPATCH(LIGHTEN, MASK422_MPEG2, false, false)
        // PLACEMENT_MPEG2
    }
    else if (vi.Is444())
      YUV_LIGHTEN_DARKEN_DISPATCH(LIGHTEN, MASK444, false, false)
    else if (vi.IsY())
      YUV_LIGHTEN_DARKEN_DISPATCH(LIGHTEN, MASK444, true, false)
  }
  else {
    // darken
    if (vi.Is411()) {
      if (placement == PLACEMENT_MPEG1)
        YUV_LIGHTEN_DARKEN_DISPATCH(DARKEN, MASK411, false, false)
      else
        YUV_LIGHTEN_DARKEN_DISPATCH(DARKEN, MASK411_TOPLEFT, false, false)
    }
    else if (vi.Is440()) {
      // see LIGHTEN's 4:4:0 case above
      if (placement == PLACEMENT_MPEG1)
        YUV_LIGHTEN_DARKEN_DISPATCH(DARKEN, MASK440, false, false)
      else
        YUV_LIGHTEN_DARKEN_DISPATCH(DARKEN, MASK440_TOPLEFT, false, false)
    }
    else if (vi.Is410()) {
      if (placement == PLACEMENT_MPEG1)
        YUV_LIGHTEN_DARKEN_DISPATCH(DARKEN, MASK410, false, false)
      else if (placement == PLACEMENT_TOPLEFT)
        YUV_LIGHTEN_DARKEN_DISPATCH(DARKEN, MASK410_TOPLEFT, false, false)
      else // PLACEMENT_MPEG2
        YUV_LIGHTEN_DARKEN_DISPATCH(DARKEN, MASK410_MPEG2, false, false)
    }
    else if (vi.Is420())
    {
      if (placement == PLACEMENT_MPEG1)
        YUV_LIGHTEN_DARKEN_DISPATCH(DARKEN, MASK420, false, false)
      else if (placement == PLACEMENT_TOPLEFT)
        YUV_LIGHTEN_DARKEN_DISPATCH(DARKEN, MASK420_TOPLEFT, false, false)
      else // PLACEMENT_MPEG2
        YUV_LIGHTEN_DARKEN_DISPATCH(DARKEN, MASK420_MPEG2, false, false)
    }
    else if (vi.Is422())
    {
      if (placement == PLACEMENT_MPEG1)
        YUV_LIGHTEN_DARKEN_DISPATCH(DARKEN, MASK422, false, false)
      else // PLACEMENT_MPEG2
        YUV_LIGHTEN_DARKEN_DISPATCH(DARKEN, MASK422_MPEG2, false, false)
    }
    else if (vi.Is444())
      YUV_LIGHTEN_DARKEN_DISPATCH(DARKEN, MASK444, false, false)
    else if (vi.IsY())
      YUV_LIGHTEN_DARKEN_DISPATCH(DARKEN, MASK444, true, false)
  }
#undef YUV_LIGHTEN_DARKEN_DISPATCH
}


static void get_layer_yuv_mul_functions(
  bool is_chroma, bool hasAlpha,
  int placement, VideoInfo& vi, int bits_per_pixel,
  /*out*/layer_yuv_mul_c_t** layer_fn,
  /*out*/layer_yuv_mul_f_c_t** layer_f_fn)
{
#define YUV_MUL_DISPATCH(MaskType, is_chroma, has_alpha) \
  { if (bits_per_pixel == 8) \
    *layer_fn = layer_yuv_mul_c<MaskType, uint8_t, is_chroma, has_alpha>; \
  else if (bits_per_pixel < 16) \
    *layer_fn = layer_yuv_mul_c<MaskType, uint16_t, is_chroma, has_alpha>; \
  else if (bits_per_pixel == 16) \
    *layer_fn = layer_yuv_mul_c<MaskType, uint16_t, is_chroma, has_alpha>; \
  else /* float */ \
    *layer_f_fn = layer_yuv_mul_f_c<MaskType, is_chroma, has_alpha>; \
  }

  if (is_chroma) // not luma channel
  {
    if (vi.Is411())
    {
      // 4:1:1 has horizontal subsampling only: centred (MPEG1) or left-sited (left, top_left).
      if (placement == PLACEMENT_MPEG1) {
        if (hasAlpha) { YUV_MUL_DISPATCH(MASK411, true, true) }
        else          { YUV_MUL_DISPATCH(MASK411, true, false) }
      }
      else {
        if (hasAlpha) { YUV_MUL_DISPATCH(MASK411_TOPLEFT, true, true) }
        else          { YUV_MUL_DISPATCH(MASK411_TOPLEFT, true, false) }
      }
    }
    else if (vi.Is440())
    {
      // vertical subsampling only: centred (MPEG1) or top-sited (TOPLEFT)
      if (placement == PLACEMENT_MPEG1) {
        if (hasAlpha) { YUV_MUL_DISPATCH(MASK440, true, true) }
        else          { YUV_MUL_DISPATCH(MASK440, true, false) }
      }
      else {
        if (hasAlpha) { YUV_MUL_DISPATCH(MASK440_TOPLEFT, true, true) }
        else          { YUV_MUL_DISPATCH(MASK440_TOPLEFT, true, false) }
      }
    }
    else if (vi.Is410())
    {
      if (placement == PLACEMENT_MPEG1) {
        if (hasAlpha) { YUV_MUL_DISPATCH(MASK410, true, true) }
        else          { YUV_MUL_DISPATCH(MASK410, true, false) }
      }
      else if (placement == PLACEMENT_TOPLEFT) {
        if (hasAlpha) { YUV_MUL_DISPATCH(MASK410_TOPLEFT, true, true) }
        else          { YUV_MUL_DISPATCH(MASK410_TOPLEFT, true, false) }
      }
      else { // PLACEMENT_MPEG2
        if (hasAlpha) { YUV_MUL_DISPATCH(MASK410_MPEG2, true, true) }
        else          { YUV_MUL_DISPATCH(MASK410_MPEG2, true, false) }
      }
    }
    else if (vi.Is420())
    {
      if (placement == PLACEMENT_MPEG1) {
        if (hasAlpha) {
          YUV_MUL_DISPATCH(MASK420, true, true)
        }
        else {
          YUV_MUL_DISPATCH(MASK420, true, false)
        }
      }
      else if (placement == PLACEMENT_TOPLEFT) {
        if (hasAlpha) {
          YUV_MUL_DISPATCH(MASK420_TOPLEFT, true, true)
        }
        else {
          YUV_MUL_DISPATCH(MASK420_TOPLEFT, true, false)
        }
      }
      else {
        if (hasAlpha) {
          YUV_MUL_DISPATCH(MASK420_MPEG2, true, true)
        }
        else {
          YUV_MUL_DISPATCH(MASK420_MPEG2, true, false)
        }
      }
    }
    else if (vi.Is422())
    {
      if (placement == PLACEMENT_MPEG1) {
        if (hasAlpha) {
          YUV_MUL_DISPATCH(MASK422, true, true)
        }
        else {
          YUV_MUL_DISPATCH(MASK422, true, false)
        }
      }
      else {
        if (hasAlpha) {
          YUV_MUL_DISPATCH(MASK422_MPEG2, true, true)
        }
        else {
          YUV_MUL_DISPATCH(MASK422_MPEG2, true, false)
        }
      }
    }
    else if (vi.Is444())
    {
      if (hasAlpha) {
        YUV_MUL_DISPATCH(MASK444, true, true)
      }
      else {
        YUV_MUL_DISPATCH(MASK444, true, false)
      }
    }
  }
  else // luma channel
  {
    if (hasAlpha)
      YUV_MUL_DISPATCH(MASK444, false, true)
    else
      YUV_MUL_DISPATCH(MASK444, false, false)
  }
#undef YUV_MUL_DISPATCH
}

// ---------------------------------------------------------------------------
// mulovr: Overlay-style multiply.
// Overlay luma (Y) drives the effect on all planes:
//   darken_factor = alpha_eff * (max - ovr_Y) / max
//   result_Y  = base_Y  * (max - darken_factor) / max
//   result_UV = base_UV * (max - darken_factor) / max  +  half_pix * darken_factor / max
// For float (neutral UV = 0.0): result = base * (1 - alpha_eff * (1 - ovr_Y)) for all planes.
// Two-pass for subsampled formats (mirrors lighten/darken):
//   Pass 1 (lumaonly=false): UV update using spatially-averaged overlay Y; MASK444 also does Y.
//   Pass 2 (lumaonly=true,  MASK444): Y-only update at full luma resolution.
// greyscale (vi.IsY()): dispatched directly as lumaonly=true — identical to Layer "Mul" luma.
// ---------------------------------------------------------------------------

template<MaskMode maskMode, typename pixel_t, bool lumaonly, bool has_alpha>
static void layer_yuv_mulovr_c(
  BYTE* dstp8, BYTE* dstp8_u, BYTE* dstp8_v,
  const BYTE* ovrp8,
  const BYTE* maskp8,
  int dst_pitch, int dst_pitchUV,
  int overlay_pitch,
  int mask_pitch,
  int width, int height, int opacity_i, int bits_per_pixel)
{
  pixel_t*       dstp   = reinterpret_cast<pixel_t*>(dstp8);
  pixel_t*       dstp_u = reinterpret_cast<pixel_t*>(dstp8_u); // nullptr when lumaonly
  pixel_t*       dstp_v = reinterpret_cast<pixel_t*>(dstp8_v); // nullptr when lumaonly
  const pixel_t* ovrp   = reinterpret_cast<const pixel_t*>(ovrp8);
  const pixel_t* maskp  = has_alpha ? reinterpret_cast<const pixel_t*>(maskp8) : nullptr;

  dst_pitch     /= sizeof(pixel_t);
  dst_pitchUV   /= sizeof(pixel_t);
  overlay_pitch /= sizeof(pixel_t);
  if constexpr (has_alpha)
    mask_pitch /= sizeof(pixel_t);

  if constexpr (sizeof(pixel_t) == 1)
    bits_per_pixel = 8;

  const MagicDiv magic          = get_magic_div(bits_per_pixel);
  const int      max_pixel_value = (1 << bits_per_pixel) - 1;
  const int      half            = max_pixel_value / 2; // rounding bias for magic_div
  const uint32_t half_pix        = (uint32_t)(max_pixel_value / 2); // integer chroma neutral

  const int cwidth  = width / MaskHSubsample<maskMode>;
  const int cheight = height / MaskVSubsample<maskMode>;

  ScratchBuffer<pixel_t> ovr_y_buffer;
  ScratchBuffer<pixel_t> mask_buffer;
  if constexpr (maskMode != MASK444) {
    ovr_y_buffer.resize(cwidth);
    if constexpr (has_alpha)
      mask_buffer.resize(cwidth);
  }

  for (int y = 0; y < cheight; ++y) {
    // Spatially average overlay Y to chroma resolution (full_opacity=true: no opacity baking).
    const pixel_t* ovr_y_ptr = LAYER_ROWPREP_FN<maskMode, pixel_t, true>(ovrp, overlay_pitch, cwidth, mask_rows_above(y, MaskVSubsample<maskMode>), mask_rows_below(y, cheight, MaskVSubsample<maskMode>), ovr_y_buffer.data());
    const pixel_t* mask_ptr  = nullptr;
    if constexpr (has_alpha)
      mask_ptr = LAYER_ROWPREP_FN<maskMode, pixel_t, true>(maskp, mask_pitch, cwidth, mask_rows_above(y, MaskVSubsample<maskMode>), mask_rows_below(y, cheight, MaskVSubsample<maskMode>), mask_buffer.data());

    // SIMD row kernel: does pixels [0, x) and returns x; 0 when there is none (C build). The C loop does the rest.
    int x = LAYER_ROW_KERNELS::yuv_mulovr<pixel_t, lumaonly, has_alpha, maskMode == MASK444>(
      dstp, dstp_u, dstp_v, ovr_y_ptr, mask_ptr, cwidth, opacity_i, max_pixel_value, half, magic);
    for (; x < cwidth; ++x) {
      const uint32_t alpha_eff = has_alpha
        ? magic_div_rt<pixel_t>((uint32_t)mask_ptr[x] * opacity_i + half, magic)
        : (uint32_t)opacity_i;
      // dark overlay Y → large darken_factor → strong pull toward neutral/black.
      const uint32_t darken_factor = magic_div_rt<pixel_t>(alpha_eff * ((uint32_t)max_pixel_value - (uint32_t)ovr_y_ptr[x]) + half, magic);
      const uint32_t inv_keep      = (uint32_t)max_pixel_value - darken_factor;

      if constexpr (!lumaonly) {
        // UV: pull toward half_pix (chroma neutral). Overflow-safe in uint32 (see plan).
        dstp_u[x] = (pixel_t)magic_div_rt<pixel_t>((uint32_t)dstp_u[x] * inv_keep + half_pix * darken_factor + half, magic);
        dstp_v[x] = (pixel_t)magic_div_rt<pixel_t>((uint32_t)dstp_v[x] * inv_keep + half_pix * darken_factor + half, magic);
      }
      // For MASK444, luma and chroma share resolution — update Y here.
      if constexpr (maskMode == MASK444)
        dstp[x] = (pixel_t)magic_div_rt<pixel_t>((uint32_t)dstp[x] * inv_keep + half, magic);
    }

    dstp  += dst_pitch * MaskVSubsample<maskMode>;
    ovrp  += overlay_pitch * MaskVSubsample<maskMode>;
    if constexpr (has_alpha)
      maskp += mask_pitch * MaskVSubsample<maskMode>;

    if constexpr (!lumaonly) {
      dstp_u += dst_pitchUV;
      dstp_v += dst_pitchUV;
    }
  }

  dst_pitch     *= sizeof(pixel_t);
  dst_pitchUV   *= sizeof(pixel_t);
  overlay_pitch *= sizeof(pixel_t);
  if constexpr (has_alpha)
    mask_pitch *= sizeof(pixel_t);

  // Phase 2: luma-only pass at full luma resolution.
  if constexpr (!lumaonly && maskMode != MASK444)
    layer_yuv_mulovr_c<MASK444, pixel_t, true, has_alpha>(
      dstp8, nullptr, nullptr,
      ovrp8, maskp8,
      dst_pitch, 0, overlay_pitch, mask_pitch,
      width, height, opacity_i, bits_per_pixel);
}

// Float version — neutral UV = 0.0f in AviSynth+ float YUV, so both luma and chroma
// collapse to the same formula: result = base * (1 - alpha_eff * (1 - ovr_Y)).
template<MaskMode maskMode, bool lumaonly, bool has_alpha>
static void layer_yuv_mulovr_f_c(
  BYTE* dstp8, BYTE* dstp8_u, BYTE* dstp8_v,
  const BYTE* ovrp8,
  const BYTE* maskp8,
  int dst_pitch, int dst_pitchUV,
  int overlay_pitch,
  int mask_pitch,
  int width, int height, float opacity)
{
  float*       dstp   = reinterpret_cast<float*>(dstp8);
  float*       dstp_u = reinterpret_cast<float*>(dstp8_u); // nullptr when lumaonly
  float*       dstp_v = reinterpret_cast<float*>(dstp8_v); // nullptr when lumaonly
  const float* ovrp   = reinterpret_cast<const float*>(ovrp8);
  const float* maskp  = has_alpha ? reinterpret_cast<const float*>(maskp8) : nullptr;

  dst_pitch     /= sizeof(float);
  dst_pitchUV   /= sizeof(float);
  overlay_pitch /= sizeof(float);
  if constexpr (has_alpha)
    mask_pitch /= sizeof(float);

  const int cwidth  = width / MaskHSubsample<maskMode>;
  const int cheight = height / MaskVSubsample<maskMode>;

  ScratchBuffer<float> ovr_y_buffer;
  ScratchBuffer<float> mask_buffer;
  if constexpr (maskMode != MASK444) {
    ovr_y_buffer.resize(cwidth);
    if constexpr (has_alpha)
      mask_buffer.resize(cwidth);
  }

  for (int y = 0; y < cheight; ++y) {
    const float* ovr_y_ptr = prepare_effective_mask_for_row_float_c<maskMode>(ovrp, overlay_pitch, cwidth, mask_rows_above(y, MaskVSubsample<maskMode>), mask_rows_below(y, cheight, MaskVSubsample<maskMode>), ovr_y_buffer.data());
    const float* mask_ptr  = nullptr;
    if constexpr (has_alpha)
      mask_ptr = prepare_effective_mask_for_row_float_c<maskMode>(maskp, mask_pitch, cwidth, mask_rows_above(y, MaskVSubsample<maskMode>), mask_rows_below(y, cheight, MaskVSubsample<maskMode>), mask_buffer.data());

    for (int x = 0; x < cwidth; ++x) {
      const float alpha_eff = has_alpha ? mask_ptr[x] * opacity : opacity;
      const float inv_keep  = 1.0f - alpha_eff * (1.0f - ovr_y_ptr[x]);

      if constexpr (!lumaonly) {
        dstp_u[x] *= inv_keep; // neutral UV = 0.0f → same formula as luma
        dstp_v[x] *= inv_keep;
      }
      if constexpr (maskMode == MASK444)
        dstp[x] *= inv_keep;
    }

    dstp  += dst_pitch * MaskVSubsample<maskMode>;
    ovrp  += overlay_pitch * MaskVSubsample<maskMode>;
    if constexpr (has_alpha)
      maskp += mask_pitch * MaskVSubsample<maskMode>;

    if constexpr (!lumaonly) {
      dstp_u += dst_pitchUV;
      dstp_v += dst_pitchUV;
    }
  }

  dst_pitch     *= sizeof(float);
  dst_pitchUV   *= sizeof(float);
  overlay_pitch *= sizeof(float);
  if constexpr (has_alpha)
    mask_pitch *= sizeof(float);

  if constexpr (!lumaonly && maskMode != MASK444)
    layer_yuv_mulovr_f_c<MASK444, true, has_alpha>(
      dstp8, nullptr, nullptr,
      ovrp8, maskp8,
      dst_pitch, 0, overlay_pitch, mask_pitch,
      width, height, opacity);
}

static void get_layer_yuv_mulovr_functions(
  bool has_alpha, int placement, VideoInfo& vi, int bits_per_pixel,
  layer_yuv_mulovr_c_t**   layer_fn,
  layer_yuv_mulovr_f_c_t** layer_f_fn)
{
#define MULOVR_DISPATCH(MaskType, lumaonly, ha) \
  { if (bits_per_pixel == 8) \
    *layer_fn   = layer_yuv_mulovr_c  <MaskType, uint8_t,  lumaonly, ha>; \
  else if (bits_per_pixel <= 16) \
    *layer_fn   = layer_yuv_mulovr_c  <MaskType, uint16_t, lumaonly, ha>; \
  else \
    *layer_f_fn = layer_yuv_mulovr_f_c<MaskType,           lumaonly, ha>; \
  }

#define MULOVR_HA(MaskType, lumaonly) \
  { if (has_alpha) { MULOVR_DISPATCH(MaskType, lumaonly, true) } \
    else           { MULOVR_DISPATCH(MaskType, lumaonly, false) } }

  if (vi.IsY()) {
    // Greyscale: no UV planes — luma-only pass, identical to Layer "Mul" luma.
    MULOVR_HA(MASK444, true)
  }
  else if (vi.Is411()) {
    if (placement == PLACEMENT_MPEG1) { MULOVR_HA(MASK411, false) }
    else                              { MULOVR_HA(MASK411_TOPLEFT, false) }
  }
  else if (vi.Is440()) {
    // vertical subsampling only: centred (MPEG1) or top-sited (TOPLEFT)
    if (placement == PLACEMENT_MPEG1) { MULOVR_HA(MASK440, false) }
    else                              { MULOVR_HA(MASK440_TOPLEFT, false) }
  }
  else if (vi.Is410()) {
    if      (placement == PLACEMENT_MPEG1)    { MULOVR_HA(MASK410,         false) }
    else if (placement == PLACEMENT_TOPLEFT)  { MULOVR_HA(MASK410_TOPLEFT, false) }
    else                                      { MULOVR_HA(MASK410_MPEG2,   false) }
  }
  else if (vi.Is420()) {
    if      (placement == PLACEMENT_MPEG1)    { MULOVR_HA(MASK420,         false) }
    else if (placement == PLACEMENT_TOPLEFT)  { MULOVR_HA(MASK420_TOPLEFT, false) }
    else                                      { MULOVR_HA(MASK420_MPEG2,   false) }
  }
  else if (vi.Is422()) {
    if      (placement == PLACEMENT_MPEG1)    { MULOVR_HA(MASK422,         false) }
    else                                      { MULOVR_HA(MASK422_MPEG2,   false) }
  }
  else { // Is444() — cwidth == width, cheight == height; same-res, single pass
    MULOVR_HA(MASK444, false)
  }

#undef MULOVR_HA
#undef MULOVR_DISPATCH
}

// both masked merge (HasAlpha) and plain add (no alpha)
// Though we use it only for masked merge, kept for reference.
static void get_layer_yuv_add_masked_functions(
  bool is_chroma, bool hasAlpha,
  int placement, VideoInfo& vi, int bits_per_pixel,
  /*out*/masked_merge_fn_t** layer_fn,
  /*out*/masked_merge_float_fn_t** layer_f_fn)
{
  // Use the unified (Layer,Overlay) masked merge functions
  MaskMode maskMode = is_chroma ? resolveChromaMaskMode(placement, vi) : MASK444;
  *layer_fn = get_overlay_blend_masked_fn_c(is_chroma, maskMode);
  *layer_f_fn = get_overlay_blend_masked_float_fn_c(is_chroma, maskMode);

}

/* planar rgb */
template<int mode, typename pixel_t, bool has_alpha, bool blend_alpha>
static void layer_planarrgb_lighten_darken_c(BYTE** dstp8, const BYTE** ovrp8, const BYTE* maskp8, int dst_pitch, int overlay_pitch, int mask_pitch, int width, int height, int opacity_i, int thresh, int bits_per_pixel) {
  pixel_t* dstp_g = reinterpret_cast<pixel_t*>(dstp8[0]);
  pixel_t* dstp_b = reinterpret_cast<pixel_t*>(dstp8[1]);
  pixel_t* dstp_r = reinterpret_cast<pixel_t*>(dstp8[2]);
  // dstp8[3]: written only when blend_alpha=true (both clips have alpha).
  pixel_t* dstp_a = nullptr;
  if constexpr (blend_alpha)
    dstp_a = reinterpret_cast<pixel_t*>(dstp8[3]);
  const pixel_t* ovrp_g = reinterpret_cast<const pixel_t*>(ovrp8[0]);
  const pixel_t* ovrp_b = reinterpret_cast<const pixel_t*>(ovrp8[1]);
  const pixel_t* ovrp_r = reinterpret_cast<const pixel_t*>(ovrp8[2]);
  // maskp: per-pixel blend weight — decoupled from ovrp8[3] so a future mask clip can be wired in.
  const pixel_t* maskp = reinterpret_cast<const pixel_t*>(maskp8);
  // alpha_target: value blended into dstp_a (only used when blend_alpha=true).
  // For plain Add this is the same plane as maskp; for Subtract it is the inverted A plane.
  const pixel_t* alpha_target = nullptr;
  if constexpr (blend_alpha)
    alpha_target = reinterpret_cast<const pixel_t*>(ovrp8[3]);

  dst_pitch /= sizeof(pixel_t);
  overlay_pitch /= sizeof(pixel_t);
  mask_pitch /= sizeof(pixel_t);

  if constexpr (sizeof(pixel_t) == 1)
    bits_per_pixel = 8; // make quasi constexpr

  const MagicDiv magic = get_magic_div(bits_per_pixel);
  const int max_pixel_value = (1 << bits_per_pixel) - 1;
  const int half = max_pixel_value / 2;

  for (int y = 0; y < height; ++y) {
    // SIMD row kernel: does pixels [0, x) and returns x; 0 when there is none (C build). The C loop does the rest.
    int x = LAYER_ROW_KERNELS::planarrgb_lighten_darken<mode, pixel_t, has_alpha, blend_alpha>(
      dstp_g, dstp_b, dstp_r, dstp_a, ovrp_g, ovrp_b, ovrp_r, maskp, alpha_target,
      width, opacity_i, thresh, max_pixel_value, half, magic);
    // 3-4 destination + 3-5 source planes: too many pointer pairs for GCC's runtime alias checks, the loop
    // stayed scalar on aarch64. Safe: x reads and writes element x of each plane only, no loop-carried dependency.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC ivdep
#endif
    for (; x < width; ++x) {
      // opacity_i is in [0..max_pixel_value], the same convention as masked_merge.
      uint32_t alpha_eff;
      if constexpr (has_alpha) {
        const uint32_t alpha_src = (uint32_t)maskp[x];
        alpha_eff = (uint32_t)magic_div_rt<pixel_t>(alpha_src * (uint32_t)opacity_i + half, magic);
      } else {
        alpha_eff = opacity_i;
      }
      // standard rgb->luma coefficients, scaled up by 15 bits
      // no rounding, not really needed here
      const int luma_ovr = (cyb * ovrp_b[x] + cyg * ovrp_g[x] + cyr * ovrp_r[x]) >> 15;
      const int luma_src = (cyb * dstp_b[x] + cyg * dstp_g[x] + cyr * dstp_r[x]) >> 15;


      if constexpr (mode == LIGHTEN)
        alpha_eff = luma_ovr > luma_src + thresh ? alpha_eff : 0;
      else // DARKEN
        alpha_eff = luma_ovr < luma_src - thresh ? alpha_eff : 0;

      const uint32_t inv_alpha = max_pixel_value - alpha_eff;

      // move to magicdiv
      //dstp_r[x] = (pixel_t)(dstp_r[x] + (((ovrp_r[x] - dstp_r[x]) * alpha + rounder) >> bits_per_pixel));
      //dstp_g[x] = (pixel_t)(dstp_g[x] + (((ovrp_g[x] - dstp_g[x]) * alpha + rounder) >> bits_per_pixel));
      //dstp_b[x] = (pixel_t)(dstp_b[x] + (((ovrp_b[x] - dstp_b[x]) * alpha + rounder) >> bits_per_pixel));
      dstp_r[x] = (pixel_t)magic_div_rt<pixel_t>((uint32_t)dstp_r[x] * inv_alpha + (uint32_t)ovrp_r[x] * alpha_eff + half, magic);
      dstp_g[x] = (pixel_t)magic_div_rt<pixel_t>((uint32_t)dstp_g[x] * inv_alpha + (uint32_t)ovrp_g[x] * alpha_eff + half, magic);
      dstp_b[x] = (pixel_t)magic_div_rt<pixel_t>((uint32_t)dstp_b[x] * inv_alpha + (uint32_t)ovrp_b[x] * alpha_eff + half, magic);
      if constexpr (blend_alpha)
        dstp_a[x] = (pixel_t)magic_div_rt<pixel_t>((uint32_t)dstp_a[x] * inv_alpha + (uint32_t)alpha_target[x] * alpha_eff + half, magic);
    }
    dstp_g += dst_pitch;
    dstp_b += dst_pitch;
    dstp_r += dst_pitch;
    if constexpr (blend_alpha)
      dstp_a += dst_pitch;
    ovrp_g += overlay_pitch;
    ovrp_b += overlay_pitch;
    ovrp_r += overlay_pitch;
    if constexpr (has_alpha)
      maskp += mask_pitch;
    if constexpr (blend_alpha)
      alpha_target += overlay_pitch;
  }
}

template<int mode, bool has_alpha, bool blend_alpha>
static void layer_planarrgb_lighten_darken_f_c(BYTE** dstp8, const BYTE** ovrp8, const BYTE* maskp8, int dst_pitch, int overlay_pitch, int mask_pitch, int width, int height, float opacity, float thresh) {
  float* dstp_g = reinterpret_cast<float*>(dstp8[0]);
  float* dstp_b = reinterpret_cast<float*>(dstp8[1]);
  float* dstp_r = reinterpret_cast<float*>(dstp8[2]);
  // dstp8[3]: written only when blend_alpha=true (both clips have alpha).
  float* dstp_a = nullptr;
  if constexpr (blend_alpha)
    dstp_a = reinterpret_cast<float*>(dstp8[3]);
  const float* ovrp_g = reinterpret_cast<const float*>(ovrp8[0]);
  const float* ovrp_b = reinterpret_cast<const float*>(ovrp8[1]);
  const float* ovrp_r = reinterpret_cast<const float*>(ovrp8[2]);
  // maskp: per-pixel blend weight — decoupled from ovrp8[3].
  const float* maskp = reinterpret_cast<const float*>(maskp8);
  const float* alpha_target = nullptr;
  if constexpr (blend_alpha)
    alpha_target = reinterpret_cast<const float*>(ovrp8[3]);

  dst_pitch /= sizeof(float);
  overlay_pitch /= sizeof(float);
  mask_pitch /= sizeof(float);

  for (int y = 0; y < height; ++y) {
    // SIMD row kernel: does pixels [0, x) and returns x; 0 when there is none (C build). The C loop does the rest.
    int x = LAYER_ROW_KERNELS::planarrgb_lighten_darken_f<mode, has_alpha, blend_alpha>(
      dstp_g, dstp_b, dstp_r, dstp_a, ovrp_g, ovrp_b, ovrp_r, maskp, alpha_target,
      width, opacity, thresh);
    // see layer_planarrgb_lighten_darken_c
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC ivdep
#endif
    for (; x < width; ++x) {
      float alpha = has_alpha ? maskp[x] * opacity : opacity;

      float luma_ovr = cyb_f * ovrp_b[x] + cyg_f * ovrp_g[x] + cyr_f * ovrp_r[x];
      float luma_src = cyb_f * dstp_b[x] + cyg_f * dstp_g[x] + cyr_f * dstp_r[x];

      if constexpr (mode == LIGHTEN)
        alpha = luma_ovr > luma_src + thresh ? alpha : 0;
      else // DARKEN
        alpha = luma_ovr < luma_src - thresh ? alpha : 0;

      dstp_r[x] = dstp_r[x] + (ovrp_r[x] - dstp_r[x]) * alpha;
      dstp_g[x] = dstp_g[x] + (ovrp_g[x] - dstp_g[x]) * alpha;
      dstp_b[x] = dstp_b[x] + (ovrp_b[x] - dstp_b[x]) * alpha;
      if constexpr (blend_alpha)
        dstp_a[x] = dstp_a[x] + (alpha_target[x] - dstp_a[x]) * alpha;
    }
    dstp_g += dst_pitch;
    dstp_b += dst_pitch;
    dstp_r += dst_pitch;
    if constexpr (blend_alpha)
      dstp_a += dst_pitch;
    ovrp_g += overlay_pitch;
    ovrp_b += overlay_pitch;
    ovrp_r += overlay_pitch;
    if constexpr (has_alpha)
      maskp += mask_pitch;
    if constexpr (blend_alpha)
      alpha_target += overlay_pitch;
  }
}


static void get_layer_planarrgb_lighten_darken_functions(bool isLighten, bool hasAlpha, bool blendAlpha, int bits_per_pixel, /*out*/layer_planarrgb_lighten_darken_c_t** layer_fn, /*out*/layer_planarrgb_lighten_darken_f_c_t** layer_f_fn) {

#define PLANARRGB_LD_DISPATCH(LorD, has_alpha, blend_alpha) \
      { if (bits_per_pixel == 8) \
        *layer_fn = layer_planarrgb_lighten_darken_c<LorD, uint8_t, has_alpha, blend_alpha>; \
      else if (bits_per_pixel <= 16) \
        *layer_fn = layer_planarrgb_lighten_darken_c<LorD, uint16_t, has_alpha, blend_alpha>; \
      else /* float */ \
        *layer_f_fn = layer_planarrgb_lighten_darken_f_c<LorD, has_alpha, blend_alpha>; \
      }

  if (isLighten) {
    if (hasAlpha) {
      if (blendAlpha) { PLANARRGB_LD_DISPATCH(LIGHTEN, true, true)  }
      else            { PLANARRGB_LD_DISPATCH(LIGHTEN, true, false) }
    }
    else {
      PLANARRGB_LD_DISPATCH(LIGHTEN, false, false)
    }
  } // lighten end
  else {
    if (hasAlpha) {
      if (blendAlpha) { PLANARRGB_LD_DISPATCH(DARKEN, true, true)  }
      else            { PLANARRGB_LD_DISPATCH(DARKEN, true, false) }
    }
    else {
      PLANARRGB_LD_DISPATCH(DARKEN, false, false)
    }
  }
#undef PLANARRGB_LD_DISPATCH
}

// subtract=false only: overlay clip is pre-inverted by Layer::Create when op="Subtract".
template<typename pixel_t,bool chroma, bool has_alpha, bool blend_alpha>
static void layer_planarrgb_add_c(BYTE** dstp8, const BYTE** ovrp8, const BYTE* maskp8, int dst_pitch, int overlay_pitch, int mask_pitch, int width, int height, int opacity_i, int bits_per_pixel) {
  pixel_t* dstp_g = reinterpret_cast<pixel_t*>(dstp8[0]);
  pixel_t* dstp_b = reinterpret_cast<pixel_t*>(dstp8[1]);
  pixel_t* dstp_r = reinterpret_cast<pixel_t*>(dstp8[2]);
  // dstp8[3]: written only when blend_alpha=true (both clips have alpha).
  pixel_t* dstp_a = nullptr;
  if constexpr (blend_alpha)
    dstp_a = reinterpret_cast<pixel_t*>(dstp8[3]);
  const pixel_t* ovrp_g = reinterpret_cast<const pixel_t*>(ovrp8[0]);
  const pixel_t* ovrp_b = reinterpret_cast<const pixel_t*>(ovrp8[1]);
  const pixel_t* ovrp_r = reinterpret_cast<const pixel_t*>(ovrp8[2]);
  // maskp: per-pixel blend weight — decoupled from ovrp8[3] to support future mask clip param.
  const pixel_t* maskp = reinterpret_cast<const pixel_t*>(maskp8);
  // alpha_target: the value blended into dstp_a.
  // For plain Add: same as maskp (original overlay A).
  // For Subtract: ovrp8[3] is the inverted A; maskp is the saved pre-invert A.
  const pixel_t* alpha_target = nullptr;
  if constexpr (blend_alpha)
    alpha_target = reinterpret_cast<const pixel_t*>(ovrp8[3]);

  dst_pitch /= sizeof(pixel_t);
  overlay_pitch /= sizeof(pixel_t);
  mask_pitch /= sizeof(pixel_t);

  if constexpr (sizeof(pixel_t) == 1)
    bits_per_pixel = 8; // make quasi constexpr

  // use magic-div
  const MagicDiv magic = get_magic_div(bits_per_pixel);
  const int max_pixel_value = (1 << bits_per_pixel) - 1;
  const int half = max_pixel_value / 2;

  for (int y = 0; y < height; ++y) {
    // 3-4 destination + 3-4 overlay planes + mask: too many pointer pairs for GCC's runtime alias checks, the loop
    // stayed scalar on aarch64. Safe: x reads and writes element x of each plane only.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC ivdep
#endif
    for (int x = 0; x < width; ++x) {

      uint32_t alpha_eff;
      if constexpr (has_alpha) {
        const uint32_t alpha_src = (uint32_t)maskp[x];
        alpha_eff = (uint32_t)magic_div_rt<pixel_t>(alpha_src * (uint32_t)opacity_i + half, magic);
      }
      else {
        alpha_eff = opacity_i;
      }
      const uint32_t inv_alpha = max_pixel_value - alpha_eff;

      if constexpr (chroma) {
        //dstp_r[x] = (pixel_t)(dstp_r[x] + (((ovrp_r[x] - dstp_r[x]) * alpha + rounder) >> bits_per_pixel));
        //dstp_g[x] = (pixel_t)(dstp_g[x] + (((ovrp_g[x] - dstp_g[x]) * alpha + rounder) >> bits_per_pixel));
        //dstp_b[x] = (pixel_t)(dstp_b[x] + (((ovrp_b[x] - dstp_b[x]) * alpha + rounder) >> bits_per_pixel));
        dstp_r[x] = (pixel_t)magic_div_rt<pixel_t>((uint32_t)dstp_r[x] * inv_alpha + (uint32_t)ovrp_r[x] * alpha_eff + half, magic);
        dstp_g[x] = (pixel_t)magic_div_rt<pixel_t>((uint32_t)dstp_g[x] * inv_alpha + (uint32_t)ovrp_g[x] * alpha_eff + half, magic);
        dstp_b[x] = (pixel_t)magic_div_rt<pixel_t>((uint32_t)dstp_b[x] * inv_alpha + (uint32_t)ovrp_b[x] * alpha_eff + half, magic);
      }
      else { // use luma instead of overlay
        const int luma = (cyb * ovrp_b[x] + cyg * ovrp_g[x] + cyr * ovrp_r[x]) >> 15; // no rounding not really needed here

        //dstp_r[x] = (pixel_t)(dstp_r[x] + (((luma - dstp_r[x]) * alpha + rounder) >> bits_per_pixel));
        //dstp_g[x] = (pixel_t)(dstp_g[x] + (((luma - dstp_g[x]) * alpha + rounder) >> bits_per_pixel));
        //dstp_b[x] = (pixel_t)(dstp_b[x] + (((luma - dstp_b[x]) * alpha + rounder) >> bits_per_pixel));
        const uint32_t luma_scaled = (uint32_t)luma * alpha_eff;
        dstp_r[x] = (pixel_t)magic_div_rt<pixel_t>((uint32_t)dstp_r[x] * inv_alpha + luma_scaled + half, magic);
        dstp_g[x] = (pixel_t)magic_div_rt<pixel_t>((uint32_t)dstp_g[x] * inv_alpha + luma_scaled + half, magic);
        dstp_b[x] = (pixel_t)magic_div_rt<pixel_t>((uint32_t)dstp_b[x] * inv_alpha + luma_scaled + half, magic);
      }
      if constexpr (blend_alpha)
        dstp_a[x] = (pixel_t)magic_div_rt<pixel_t>((uint32_t)dstp_a[x] * inv_alpha + (uint32_t)alpha_target[x] * alpha_eff + half, magic);
    }
    dstp_g += dst_pitch;
    dstp_b += dst_pitch;
    dstp_r += dst_pitch;
    if constexpr (blend_alpha)
      dstp_a += dst_pitch;
    ovrp_g += overlay_pitch;
    ovrp_b += overlay_pitch;
    ovrp_r += overlay_pitch;
    if constexpr (has_alpha)
      maskp += mask_pitch;
    if constexpr (blend_alpha)
      alpha_target += overlay_pitch;
  }
}

// subtract=false only: overlay clip is pre-inverted by Layer::Create when op="Subtract".
template<bool chroma, bool has_alpha, bool blend_alpha>
static void layer_planarrgb_add_f_c(BYTE** dstp8, const BYTE** ovrp8, const BYTE* maskp8, int dst_pitch, int overlay_pitch, int mask_pitch, int width, int height, float opacity) {
  float* dstp_g = reinterpret_cast<float*>(dstp8[0]);
  float* dstp_b = reinterpret_cast<float*>(dstp8[1]);
  float* dstp_r = reinterpret_cast<float*>(dstp8[2]);
  // dstp8[3]: written only when blend_alpha=true (both clips have alpha).
  float* dstp_a = nullptr;
  if constexpr (blend_alpha)
    dstp_a = reinterpret_cast<float*>(dstp8[3]);
  const float* ovrp_g = reinterpret_cast<const float*>(ovrp8[0]);
  const float* ovrp_b = reinterpret_cast<const float*>(ovrp8[1]);
  const float* ovrp_r = reinterpret_cast<const float*>(ovrp8[2]);
  // maskp: per-pixel blend weight — decoupled from ovrp8[3].
  const float* maskp = reinterpret_cast<const float*>(maskp8);
  const float* alpha_target = nullptr;
  if constexpr (blend_alpha)
    alpha_target = reinterpret_cast<const float*>(ovrp8[3]);

  dst_pitch /= sizeof(float);
  overlay_pitch /= sizeof(float);
  mask_pitch /= sizeof(float);

  for (int y = 0; y < height; ++y) {
    // see layer_planarrgb_add_c
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC ivdep
#endif
    for (int x = 0; x < width; ++x) {
      float alpha = has_alpha ? maskp[x] * opacity : opacity;

      if constexpr (chroma) {
        dstp_r[x] = dstp_r[x] + (ovrp_r[x] - dstp_r[x]) * alpha;
        dstp_g[x] = dstp_g[x] + (ovrp_g[x] - dstp_g[x]) * alpha;
        dstp_b[x] = dstp_b[x] + (ovrp_b[x] - dstp_b[x]) * alpha;
      }
      else { // use luma instead of overlay
        float luma = cyb_f * ovrp_b[x] + cyg_f * ovrp_g[x] + cyr_f * ovrp_r[x];
        dstp_r[x] = dstp_r[x] + (luma - dstp_r[x]) * alpha;
        dstp_g[x] = dstp_g[x] + (luma - dstp_g[x]) * alpha;
        dstp_b[x] = dstp_b[x] + (luma - dstp_b[x]) * alpha;
      }
      if constexpr (blend_alpha)
        dstp_a[x] = dstp_a[x] + (alpha_target[x] - dstp_a[x]) * alpha;
    }
    dstp_g += dst_pitch;
    dstp_b += dst_pitch;
    dstp_r += dst_pitch;
    if constexpr (blend_alpha)
      dstp_a += dst_pitch;
    ovrp_g += overlay_pitch;
    ovrp_b += overlay_pitch;
    ovrp_r += overlay_pitch;
    if constexpr (has_alpha)
      maskp += mask_pitch;
    if constexpr (blend_alpha)
      alpha_target += overlay_pitch;
  }
}

// ---------------------------------------------------------------------------
// Planar RGB add — SSE4.1 per-plane wrappers (mirrors AVX2 counterparts).
// All planar RGB planes are MASK444. maskp8 is the per-pixel weight.
// chroma=false and float fall back to C templates.

static void layer_planarrgb_add_c_3plane(
  BYTE** dstp8, const BYTE** ovrp8, const BYTE* maskp8,
  int dst_pitch, int overlay_pitch, int mask_pitch,
  int width, int height, int opacity_i, int bits_per_pixel)
{
  for (int i = 0; i < 3; i++)
    masked_merge_c_impl<MASK444>(
      dstp8[i], ovrp8[i], maskp8,
      dst_pitch, overlay_pitch, mask_pitch,
      width, height, opacity_i, bits_per_pixel);
}

static void layer_planarrgb_add_c_4plane(
  BYTE** dstp8, const BYTE** ovrp8, const BYTE* maskp8,
  int dst_pitch, int overlay_pitch, int mask_pitch,
  int width, int height, int opacity_i, int bits_per_pixel)
{
  for (int i = 0; i < 4; i++)
    masked_merge_c_impl<MASK444>(
      dstp8[i], ovrp8[i], maskp8,
      dst_pitch, overlay_pitch, mask_pitch,
      width, height, opacity_i, bits_per_pixel);
}


static void get_layer_planarrgb_add_functions(
  bool chroma, bool hasAlpha, bool blendAlpha, int bits_per_pixel,
  /*out*/layer_planarrgb_add_c_t** layer_fn,
  /*out*/layer_planarrgb_add_f_c_t** layer_f_fn)
{

  // chroma is true: Layer can use the unified masked and weighted blend routines
  // chroma is false: Layer-specific extension
  // Integer + hasAlpha + chroma=true: dispatch per-plane to masked_merge_avx2_impl (MASK444).
  // chroma=false (blend toward luma) has a different formula — keep C template.
  // float: keep C template (float perf is usually fine; could add later).
  if (chroma && bits_per_pixel != 32) {
    if (hasAlpha) {
      *layer_fn = blendAlpha ? layer_planarrgb_add_c_4plane : layer_planarrgb_add_c_3plane;
      return;
    }
    // no alpha: standard weighted merge, to be added later.
  }

#define PLANARRGB_ADD_DISPATCH(chroma, has_alpha, blend_alpha) \
  { if (bits_per_pixel == 8) \
    *layer_fn = layer_planarrgb_add_c<uint8_t, chroma, has_alpha, blend_alpha>; \
  else if (bits_per_pixel <= 16) \
    *layer_fn = layer_planarrgb_add_c<uint16_t, chroma, has_alpha, blend_alpha>; \
  else /* float */ \
    *layer_f_fn = layer_planarrgb_add_f_c<chroma, has_alpha, blend_alpha>; \
  }

  if (hasAlpha) {
    if (chroma) {
      if (blendAlpha) { PLANARRGB_ADD_DISPATCH(true, true, true)  }
      else            { PLANARRGB_ADD_DISPATCH(true, true, false) }
    }
    else {
      if (blendAlpha) { PLANARRGB_ADD_DISPATCH(false, true, true)  }
      else            { PLANARRGB_ADD_DISPATCH(false, true, false) }
    }
  }
  else {
    if (chroma) {
      PLANARRGB_ADD_DISPATCH(true, false, false)
    }
    else {
      PLANARRGB_ADD_DISPATCH(false, false, false)
    }
  }
#undef PLANARRGB_ADD_DISPATCH
}

template<typename pixel_t, bool chroma, bool has_alpha, bool blend_alpha>
static void layer_planarrgb_mul_c(BYTE** dstp8, const BYTE** ovrp8, const BYTE* maskp8, int dst_pitch, int overlay_pitch, int mask_pitch, int width, int height, int opacity_i, int bits_per_pixel) {
  pixel_t* dstp_g = reinterpret_cast<pixel_t*>(dstp8[0]);
  pixel_t* dstp_b = reinterpret_cast<pixel_t*>(dstp8[1]);
  pixel_t* dstp_r = reinterpret_cast<pixel_t*>(dstp8[2]);
  // dstp8[3]: written only when blend_alpha=true (both clips have alpha).
  pixel_t* dstp_a = nullptr;
  if constexpr (blend_alpha)
    dstp_a = reinterpret_cast<pixel_t*>(dstp8[3]);
  const pixel_t* ovrp_g = reinterpret_cast<const pixel_t*>(ovrp8[0]);
  const pixel_t* ovrp_b = reinterpret_cast<const pixel_t*>(ovrp8[1]);
  const pixel_t* ovrp_r = reinterpret_cast<const pixel_t*>(ovrp8[2]);
  // maskp: per-pixel blend weight — decoupled from ovrp8[3].
  const pixel_t* maskp = reinterpret_cast<const pixel_t*>(maskp8);
  // alpha_target: the value multiplied into dstp_a (only when blend_alpha=true).
  const pixel_t* alpha_target = nullptr;
  if constexpr (blend_alpha)
    alpha_target = reinterpret_cast<const pixel_t*>(ovrp8[3]);

  dst_pitch /= sizeof(pixel_t);
  overlay_pitch /= sizeof(pixel_t);
  mask_pitch /= sizeof(pixel_t);

  if constexpr (sizeof(pixel_t) == 1)
    bits_per_pixel = 8; // make quasi constexpr

  // use magic-div
  const MagicDiv magic = get_magic_div(bits_per_pixel);
  const int max_pixel_value = (1 << bits_per_pixel) - 1;
  const int half = max_pixel_value / 2;

  for (int y = 0; y < height; ++y) {
    // SIMD row kernel: does pixels [0, x) and returns x; 0 when there is none (C build). The C loop does the rest.
    int x = LAYER_ROW_KERNELS::planarrgb_mul<pixel_t, chroma, has_alpha, blend_alpha>(
      dstp_g, dstp_b, dstp_r, dstp_a, ovrp_g, ovrp_b, ovrp_r, maskp, alpha_target,
      width, opacity_i, max_pixel_value, half, magic);
    // see layer_planarrgb_lighten_darken_c
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC ivdep
#endif
    for (; x < width; ++x) {

      uint32_t alpha_eff;
      if constexpr (has_alpha) {
        const uint32_t alpha_src = (uint32_t)maskp[x];
        alpha_eff = (uint32_t)magic_div_rt<pixel_t>(alpha_src * (uint32_t)opacity_i + half, magic);
      }
      else {
        alpha_eff = opacity_i;
      }
      const uint32_t inv_alpha = max_pixel_value - alpha_eff;

      if constexpr (chroma) {
        // Calculate the multiplied product (A*B)/max_pixel_value
        // Use the MagicDiv instead of bit-shift to keep max*max=max.
        const pixel_t prod_r = (pixel_t)magic_div_rt<pixel_t>((uint32_t)ovrp_r[x] * (uint32_t)dstp_r[x], magic);
        const pixel_t prod_g = (pixel_t)magic_div_rt<pixel_t>((uint32_t)ovrp_g[x] * (uint32_t)dstp_g[x], magic);
        const pixel_t prod_b = (pixel_t)magic_div_rt<pixel_t>((uint32_t)ovrp_b[x] * (uint32_t)dstp_b[x], magic);

        dstp_r[x] = (pixel_t)magic_div_rt<pixel_t>((uint32_t)dstp_r[x] * inv_alpha + (uint32_t)prod_r * alpha_eff + half, magic);
        dstp_g[x] = (pixel_t)magic_div_rt<pixel_t>((uint32_t)dstp_g[x] * inv_alpha + (uint32_t)prod_g * alpha_eff + half, magic);
        dstp_b[x] = (pixel_t)magic_div_rt<pixel_t>((uint32_t)dstp_b[x] * inv_alpha + (uint32_t)prod_b * alpha_eff + half, magic);
      }
      else {
        // use luma instead of overlay
        const int luma = (cyb * ovrp_b[x] + cyg * ovrp_g[x] + cyr * ovrp_r[x]) >> 15;

        const pixel_t prod_r = (pixel_t)magic_div_rt<pixel_t>((uint32_t)luma * (uint32_t)dstp_r[x], magic);
        const pixel_t prod_g = (pixel_t)magic_div_rt<pixel_t>((uint32_t)luma * (uint32_t)dstp_g[x], magic);
        const pixel_t prod_b = (pixel_t)magic_div_rt<pixel_t>((uint32_t)luma * (uint32_t)dstp_b[x], magic);

        dstp_r[x] = (pixel_t)magic_div_rt<pixel_t>((uint32_t)dstp_r[x] * inv_alpha + (uint32_t)prod_r * alpha_eff + half, magic);
        dstp_g[x] = (pixel_t)magic_div_rt<pixel_t>((uint32_t)dstp_g[x] * inv_alpha + (uint32_t)prod_g * alpha_eff + half, magic);
        dstp_b[x] = (pixel_t)magic_div_rt<pixel_t>((uint32_t)dstp_b[x] * inv_alpha + (uint32_t)prod_b * alpha_eff + half, magic);
      }

      if constexpr (blend_alpha) {
        const pixel_t prod_a = (pixel_t)magic_div_rt<pixel_t>((uint32_t)alpha_target[x] * (uint32_t)dstp_a[x], magic);
        dstp_a[x] = (pixel_t)magic_div_rt<pixel_t>((uint32_t)dstp_a[x] * inv_alpha + (uint32_t)prod_a * alpha_eff + half, magic);
      }
    }
    dstp_g += dst_pitch;
    dstp_b += dst_pitch;
    dstp_r += dst_pitch;
    if constexpr (blend_alpha)
      dstp_a += dst_pitch;
    ovrp_g += overlay_pitch;
    ovrp_b += overlay_pitch;
    ovrp_r += overlay_pitch;
    if constexpr (has_alpha)
      maskp += mask_pitch;
    if constexpr (blend_alpha)
      alpha_target += overlay_pitch;
  }
}

template<bool chroma, bool has_alpha, bool blend_alpha>
static void layer_planarrgb_mul_f_c(BYTE** dstp8, const BYTE** ovrp8, const BYTE* maskp8, int dst_pitch, int overlay_pitch, int mask_pitch, int width, int height, float opacity) {
  float* dstp_g = reinterpret_cast<float*>(dstp8[0]);
  float* dstp_b = reinterpret_cast<float*>(dstp8[1]);
  float* dstp_r = reinterpret_cast<float*>(dstp8[2]);
  // dstp8[3]: written only when blend_alpha=true (both clips have alpha).
  float* dstp_a = nullptr;
  if constexpr (blend_alpha)
    dstp_a = reinterpret_cast<float*>(dstp8[3]);
  const float* ovrp_g = reinterpret_cast<const float*>(ovrp8[0]);
  const float* ovrp_b = reinterpret_cast<const float*>(ovrp8[1]);
  const float* ovrp_r = reinterpret_cast<const float*>(ovrp8[2]);
  // maskp: per-pixel blend weight — decoupled from ovrp8[3].
  const float* maskp = reinterpret_cast<const float*>(maskp8);
  const float* alpha_target = nullptr;
  if constexpr (blend_alpha)
    alpha_target = reinterpret_cast<const float*>(ovrp8[3]);

  dst_pitch /= sizeof(float);
  overlay_pitch /= sizeof(float);
  mask_pitch /= sizeof(float);

  for (int y = 0; y < height; ++y) {
    // see layer_planarrgb_lighten_darken_c
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC ivdep
#endif
    for (int x = 0; x < width; ++x) {
      float alpha = has_alpha ? maskp[x] * opacity : opacity;

      if constexpr (chroma) {
        dstp_r[x] = dstp_r[x] + (ovrp_r[x] * dstp_r[x] - dstp_r[x]) * alpha;
        dstp_g[x] = dstp_g[x] + (ovrp_g[x] * dstp_g[x] - dstp_g[x]) * alpha;
        dstp_b[x] = dstp_b[x] + (ovrp_b[x] * dstp_b[x] - dstp_b[x]) * alpha;
      }
      else { // use luma instead of overlay
        float luma = cyb_f * ovrp_b[x] + cyg_f * ovrp_g[x] + cyr_f * ovrp_r[x];
        dstp_r[x] = dstp_r[x] + (luma * dstp_r[x] - dstp_r[x]) * alpha;
        dstp_g[x] = dstp_g[x] + (luma * dstp_g[x] - dstp_g[x]) * alpha;
        dstp_b[x] = dstp_b[x] + (luma * dstp_b[x] - dstp_b[x]) * alpha;
      }
      if constexpr (blend_alpha)
        dstp_a[x] = dstp_a[x] + (alpha_target[x] * dstp_a[x] - dstp_a[x]) * alpha;
    }
    dstp_g += dst_pitch;
    dstp_b += dst_pitch;
    dstp_r += dst_pitch;
    if constexpr (blend_alpha)
      dstp_a += dst_pitch;
    ovrp_g += overlay_pitch;
    ovrp_b += overlay_pitch;
    ovrp_r += overlay_pitch;
    if constexpr (has_alpha)
      maskp += mask_pitch;
    if constexpr (blend_alpha)
      alpha_target += overlay_pitch;
  }
}

static void get_layer_planarrgb_mul_functions(
  bool chroma, bool hasAlpha, bool blendAlpha, int bits_per_pixel,
  /*out*/layer_planarrgb_mul_c_t** layer_fn,
  /*out*/layer_planarrgb_mul_f_c_t** layer_f_fn)
{
#define PLANARRGB_MUL_DISPATCH(chroma, has_alpha, blend_alpha) \
  { if (bits_per_pixel == 8) \
    *layer_fn = layer_planarrgb_mul_c<uint8_t, chroma, has_alpha, blend_alpha>; \
  else if (bits_per_pixel <= 16) \
    *layer_fn = layer_planarrgb_mul_c<uint16_t, chroma, has_alpha, blend_alpha>; \
  else /* float */ \
    *layer_f_fn = layer_planarrgb_mul_f_c<chroma, has_alpha, blend_alpha>; \
  }

  if (hasAlpha) {
    if (chroma) {
      if (blendAlpha) { PLANARRGB_MUL_DISPATCH(true, true, true)  }
      else            { PLANARRGB_MUL_DISPATCH(true, true, false) }
    }
    else {
      if (blendAlpha) { PLANARRGB_MUL_DISPATCH(false, true, true)  }
      else            { PLANARRGB_MUL_DISPATCH(false, true, false) }
    }
  }
  else {
    if (chroma) {
      PLANARRGB_MUL_DISPATCH(true, false, false)
    }
    else {
      PLANARRGB_MUL_DISPATCH(false, false, false)
    }
  }
#undef PLANARRGB_MUL_DISPATCH
}

// ---------------------------------------------------------------------------
// Packed RGBA (RGB32 / RGB64) blend dispatcher — C reference.
// Returns the appropriate masked_blend_packedrgba_c instantiation for the
// given bit depth.  Subtract is handled by pre-inverting the overlay and
// passing a separate maskp8 pointer in the caller (Layer::Create/GetFrame).
// The C reference function selects the alpha source at runtime via the
// maskp8 null-check, so a single function covers both Add and Subtract.
// ---------------------------------------------------------------------------
static void get_layer_packedrgb_blend_functions(
  bool has_separate_mask,
  int bits_per_pixel,
  layer_packedrgb_blend_c_t** fn)
{
  if (bits_per_pixel == 8)
    *fn = has_separate_mask ?  masked_blend_packedrgba_c<uint8_t, true> : masked_blend_packedrgba_c<uint8_t, false>;
  else  // 16-bit (RGB64)
    *fn = has_separate_mask ?  masked_blend_packedrgba_c<uint16_t, true> : masked_blend_packedrgba_c<uint16_t, false>;
}

// Clean up rowprep macros (defined near top of this file or by including TU).
#undef LAYER_ROWPREP_FN

