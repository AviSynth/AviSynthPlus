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

#include <avs/config.h>

#include "blend_common.h"
#include "overlayfunctions.h"

#include <stdint.h>
#include <cstring>
#include <type_traits>


/******************************
 ********* Mode: Blend ********
 ******************************/
// ---------------------------------------------------------------------------
// Overlay blend masked getter — returns masked_merge_avx2_impl instantiation.
// is_chroma=false -> always MASK444 (luma).
// is_chroma=true  -> placement-aware maskMode (chroma).
// ---------------------------------------------------------------------------

masked_merge_fn_t* get_overlay_blend_masked_fn_c(bool is_chroma, MaskMode maskMode)
{
#define DISPATCH_OVERLAY_BLEND_C(MaskType) \
  return is_chroma ? masked_merge_c_impl<MaskType> \
                   : masked_merge_c_impl<MASK444>;

  switch (maskMode) {
  case MASK444:          DISPATCH_OVERLAY_BLEND_C(MASK444)
  case MASK420:          DISPATCH_OVERLAY_BLEND_C(MASK420)
  case MASK420_MPEG2:    DISPATCH_OVERLAY_BLEND_C(MASK420_MPEG2)
  case MASK420_TOPLEFT:  DISPATCH_OVERLAY_BLEND_C(MASK420_TOPLEFT)
  case MASK422:          DISPATCH_OVERLAY_BLEND_C(MASK422)
  case MASK422_MPEG2:    DISPATCH_OVERLAY_BLEND_C(MASK422_MPEG2)
  case MASK411:          DISPATCH_OVERLAY_BLEND_C(MASK411)
  case MASK411_TOPLEFT:  DISPATCH_OVERLAY_BLEND_C(MASK411_TOPLEFT)
  case MASK440:          DISPATCH_OVERLAY_BLEND_C(MASK440)
  case MASK440_TOPLEFT:  DISPATCH_OVERLAY_BLEND_C(MASK440_TOPLEFT)
  case MASK410:          DISPATCH_OVERLAY_BLEND_C(MASK410)
  case MASK410_TOPLEFT:  DISPATCH_OVERLAY_BLEND_C(MASK410_TOPLEFT)
  case MASK410_MPEG2  :  DISPATCH_OVERLAY_BLEND_C(MASK410_MPEG2  )
  case MASK_MODE_COUNT: break;
  }
#undef DISPATCH_OVERLAY_BLEND_C
  return masked_merge_c_impl<MASK444>; // unreachable
}

// and for float:
masked_merge_float_fn_t* get_overlay_blend_masked_float_fn_c(bool is_chroma, MaskMode maskMode)
{
#define DISPATCH_OVERLAY_BLEND_FLOAT_C(MaskType) \
  return is_chroma ? masked_merge_float_c_impl<MaskType> \
                   : masked_merge_float_c_impl<MASK444>;

  switch (maskMode) {
  case MASK444:          DISPATCH_OVERLAY_BLEND_FLOAT_C(MASK444)
  case MASK420:          DISPATCH_OVERLAY_BLEND_FLOAT_C(MASK420)
  case MASK420_MPEG2:    DISPATCH_OVERLAY_BLEND_FLOAT_C(MASK420_MPEG2)
  case MASK420_TOPLEFT:  DISPATCH_OVERLAY_BLEND_FLOAT_C(MASK420_TOPLEFT)
  case MASK422:          DISPATCH_OVERLAY_BLEND_FLOAT_C(MASK422)
  case MASK422_MPEG2:    DISPATCH_OVERLAY_BLEND_FLOAT_C(MASK422_MPEG2)
  case MASK411:          DISPATCH_OVERLAY_BLEND_FLOAT_C(MASK411)
  case MASK411_TOPLEFT:  DISPATCH_OVERLAY_BLEND_FLOAT_C(MASK411_TOPLEFT)
  case MASK440:          DISPATCH_OVERLAY_BLEND_FLOAT_C(MASK440)
  case MASK440_TOPLEFT:  DISPATCH_OVERLAY_BLEND_FLOAT_C(MASK440_TOPLEFT)
  case MASK410:          DISPATCH_OVERLAY_BLEND_FLOAT_C(MASK410)
  case MASK410_TOPLEFT:  DISPATCH_OVERLAY_BLEND_FLOAT_C(MASK410_TOPLEFT)
  case MASK410_MPEG2  :  DISPATCH_OVERLAY_BLEND_FLOAT_C(MASK410_MPEG2  )
  case MASK_MODE_COUNT: break;
  }
#undef DISPATCH_OVERLAY_BLEND_FLOAT_C
  return masked_merge_float_c_impl<MASK444>; // unreachable
}

/*
// Scalar division — used in reference C, constexpr-friendly
template<int bits_per_pixel>
inline int magic_div(uint32_t tmp) {
  constexpr MagicDiv magic = get_magic_div(bits_per_pixel);
  if constexpr (bits_per_pixel == 8)
    // mimics: mulhi_epu16(x, 0x8081) >> 7
    return (int)(((uint32_t)tmp * magic.div) >> (16 + magic.shift));
  else
    // mimics: mul_epu32(x, div) >> (32 + shift)
    return (int)(((uint64_t)tmp * magic.div) >> (32 + magic.shift));
}
*/

/*****************************************************
 ********* Family 1: weighted_merge (no mask) ********
 *****************************************************/

// weight + invweight == 32768; kernel: (p1*inv + p2*w + 16384) >> 15
// Intentionally matches the SIMD >> 15 path (old C used >> 16 with weights summing to 32767).
template<typename pixel_t>
static void weighted_merge_impl_c(BYTE* p1, const BYTE* p2,
  int p1_pitch, int p2_pitch,
  int width, int height,
  int weight, int invweight)
{
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      reinterpret_cast<pixel_t*>(p1)[x] = (pixel_t)(
        (reinterpret_cast<const pixel_t*>(p1)[x] * invweight +
         reinterpret_cast<const pixel_t*>(p2)[x] * weight + 16384) >> 15);
    }
    p1 += p1_pitch;
    p2 += p2_pitch;
  }
}

// merge, handle the two extrems at plane level if it wasn't at clip level
void weighted_merge_return_a_or_b(BYTE* p1, const BYTE* p2,
  int p1_pitch, int p2_pitch,
  int width, int height,
  int weight, int invweight,
  int bits_per_pixel)
{
  // called either with weight==0, invweight==32768 or the opposite
  if (weight == 0) return; // keep p1 as-is

  const int pixelsize = bits_per_pixel == 8 ? 1 : bits_per_pixel <= 16 ? 2 : 4;
  const int rowsize = width * pixelsize;

  if (invweight == 0) {
    // like bitblt
    for (int y = 0; y < height; y++) {
      memcpy(p1, p2, rowsize);
      p1 += p1_pitch;
      p2 += p2_pitch;
    }
  }
}

void weighted_merge_c(BYTE* p1, const BYTE* p2,
  int p1_pitch, int p2_pitch,
  int width, int height,
  int weight, int invweight,
  int bits_per_pixel)
{
  if (bits_per_pixel == 8)
    weighted_merge_impl_c<uint8_t>(p1, p2, p1_pitch, p2_pitch, width, height, weight, invweight);
  else
    weighted_merge_impl_c<uint16_t>(p1, p2, p1_pitch, p2_pitch, width, height, weight, invweight);
}

void weighted_merge_float_c(BYTE* p1, const BYTE* p2,
  int p1_pitch, int p2_pitch,
  int width, int height,
  float weight_f)
{
  const float invweight_f = 1.0f - weight_f;
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      reinterpret_cast<float*>(p1)[x] =
        reinterpret_cast<const float*>(p1)[x] * invweight_f +
        reinterpret_cast<const float*>(p2)[x] * weight_f;
    }
    p1 += p1_pitch;
    p2 += p2_pitch;
  }
}

/***************************************
 ********* Mode: Lighten/Darken ********
 ***************************************/

typedef int (OverlayCCompare)(BYTE, BYTE);

template<typename pixel_t, bool darken /* OverlayCCompare<pixel_t> compare*/>
static void overlay_darklighten_c(BYTE *p1Y_8, BYTE *p1U_8, BYTE *p1V_8, const BYTE *p2Y_8, const BYTE *p2U_8, const BYTE *p2V_8, int p1_pitch, int p2_pitch, int width, int height) {
  pixel_t* p1Y = reinterpret_cast<pixel_t *>(p1Y_8);
  pixel_t* p1U = reinterpret_cast<pixel_t *>(p1U_8);
  pixel_t* p1V = reinterpret_cast<pixel_t *>(p1V_8);

  const pixel_t* p2Y = reinterpret_cast<const pixel_t *>(p2Y_8);
  const pixel_t* p2U = reinterpret_cast<const pixel_t *>(p2U_8);
  const pixel_t* p2V = reinterpret_cast<const pixel_t *>(p2V_8);

  // pitches are already scaled
  //p1_pitch /= sizeof(pixel_t);
  //p2_pitch /= sizeof(pixel_t);

  for (int y = 0; y < height; y++) {
    // All loads before the stores, then plain selects: GCC can if-convert and vectorize the loop
    // ("control flow in loop" when the p1Y store came before the U/V loads).
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC ivdep
#endif
    for (int x = 0; x < width; x++) {
      const pixel_t y1 = p1Y[x], u1 = p1U[x], v1 = p1V[x];
      const pixel_t y2 = p2Y[x], u2 = p2U[x], v2 = p2V[x];
      const bool mask = darken ? (y2 <= y1) : (y2 >= y1);
      p1Y[x] = mask ? y2 : y1;
      p1U[x] = mask ? u2 : u1;
      p1V[x] = mask ? v2 : v1;
    }

    p1Y += p1_pitch;
    p1U += p1_pitch;
    p1V += p1_pitch;

    p2Y += p2_pitch;
    p2U += p2_pitch;
    p2V += p2_pitch;
  }
}

// Exported function
template<typename pixel_t>
void overlay_darken_c(BYTE *p1Y_8, BYTE *p1U_8, BYTE *p1V_8, const BYTE *p2Y_8, const BYTE *p2U_8, const BYTE *p2V_8, int p1_pitch, int p2_pitch, int width, int height) {
  overlay_darklighten_c<pixel_t, true /*overlay_darken_c_cmp */>(p1Y_8, p1U_8, p1V_8, p2Y_8, p2U_8, p2V_8, p1_pitch, p2_pitch, width, height);
}
// instantiate
template void overlay_darken_c<uint8_t>(BYTE *p1Y_8, BYTE *p1U_8, BYTE *p1V_8, const BYTE *p2Y_8, const BYTE *p2U_8, const BYTE *p2V_8, int p1_pitch, int p2_pitch, int width, int height);
template void overlay_darken_c<uint16_t>(BYTE *p1Y_8, BYTE *p1U_8, BYTE *p1V_8, const BYTE *p2Y_8, const BYTE *p2U_8, const BYTE *p2V_8, int p1_pitch, int p2_pitch, int width, int height);

template<typename pixel_t>
void overlay_lighten_c(BYTE *p1Y_8, BYTE *p1U_8, BYTE *p1V_8, const BYTE *p2Y_8, const BYTE *p2U_8, const BYTE *p2V_8, int p1_pitch, int p2_pitch, int width, int height) {
  overlay_darklighten_c<pixel_t, false /*overlay_lighten_c_cmp*/>(p1Y_8, p1U_8, p1V_8, p2Y_8, p2U_8, p2V_8, p1_pitch, p2_pitch, width, height);
}

// instantiate
template void overlay_lighten_c<uint8_t>(BYTE *p1Y_8, BYTE *p1U_8, BYTE *p1V_8, const BYTE *p2Y_8, const BYTE *p2U_8, const BYTE *p2V_8, int p1_pitch, int p2_pitch, int width, int height);
template void overlay_lighten_c<uint16_t>(BYTE *p1Y_8, BYTE *p1U_8, BYTE *p1V_8, const BYTE *p2Y_8, const BYTE *p2U_8, const BYTE *p2V_8, int p1_pitch, int p2_pitch, int width, int height);
