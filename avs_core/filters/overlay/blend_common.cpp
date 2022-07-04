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

// Overlay (c) 2003, 2004 by Klaus Post

#include <avs/config.h>

#include "blend_common.h"
#include "overlayfunctions.h"

#include <stdint.h>
#include <type_traits>


/******************************
 ********* Mode: Blend ********
 ******************************/

// 32 bit float mask calculation inside
template<bool has_mask, typename pixel_t, int bits_per_pixel>
void overlay_blend_c_uint(BYTE* p1, const BYTE* p2, const BYTE* mask,
  const int p1_pitch, const int p2_pitch, const int mask_pitch,
  const int width, const int height, const int opacity, const float opacity_f)
{
  const int max_pixel_value = (1 << bits_per_pixel) - 1;
  auto factor = has_mask ? opacity_f / max_pixel_value : opacity_f;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      const float new_mask = has_mask ? (float)reinterpret_cast<const pixel_t*>(mask)[x] * factor : factor;
      auto result = overlay_blend_c_core_simple(
        reinterpret_cast<pixel_t*>(p1)[x],
        reinterpret_cast<const pixel_t*>(p2)[x],
        new_mask);
      reinterpret_cast<pixel_t*>(p1)[x] = (pixel_t)(result + 0.5f);
    }

    p1 += p1_pitch;
    p2 += p2_pitch;
    if(has_mask)
      mask += mask_pitch;
  }
}

// instantiate
// w/o mask
template void overlay_blend_c_uint<false, uint8_t, 8>(BYTE* p1, const BYTE* p2, const BYTE* mask,
  const int p1_pitch, const int p2_pitch, const int mask_pitch, const int width, const int height, const int opacity, const float opacity_f);
template void overlay_blend_c_uint<false, uint16_t, 10>(BYTE* p1, const BYTE* p2, const BYTE* mask,
  const int p1_pitch, const int p2_pitch, const int mask_pitch, const int width, const int height, const int opacity, const float opacity_f);
template void overlay_blend_c_uint<false, uint16_t, 12>(BYTE* p1, const BYTE* p2, const BYTE* mask,
  const int p1_pitch, const int p2_pitch, const int mask_pitch, const int width, const int height, const int opacity, const float opacity_f);
template void overlay_blend_c_uint<false, uint16_t, 14>(BYTE* p1, const BYTE* p2, const BYTE* mask,
  const int p1_pitch, const int p2_pitch, const int mask_pitch, const int width, const int height, const int opacity, const float opacity_f);
template void overlay_blend_c_uint<false, uint16_t, 16>(BYTE* p1, const BYTE* p2, const BYTE* mask,
  const int p1_pitch, const int p2_pitch, const int mask_pitch, const int width, const int height, const int opacity, const float opacity_f);
// w/ mask
template void overlay_blend_c_uint<true, uint8_t, 8>(BYTE* p1, const BYTE* p2, const BYTE* mask,
  const int p1_pitch, const int p2_pitch, const int mask_pitch, const int width, const int height, const int opacity, const float opacity_f);
template void overlay_blend_c_uint<true, uint16_t, 10>(BYTE* p1, const BYTE* p2, const BYTE* mask,
  const int p1_pitch, const int p2_pitch, const int mask_pitch, const int width, const int height, const int opacity, const float opacity_f);
template void overlay_blend_c_uint<true, uint16_t, 12>(BYTE* p1, const BYTE* p2, const BYTE* mask,
  const int p1_pitch, const int p2_pitch, const int mask_pitch, const int width, const int height, const int opacity, const float opacity_f);
template void overlay_blend_c_uint<true, uint16_t, 14>(BYTE* p1, const BYTE* p2, const BYTE* mask,
  const int p1_pitch, const int p2_pitch, const int mask_pitch, const int width, const int height, const int opacity, const float opacity_f);
template void overlay_blend_c_uint<true, uint16_t, 16>(BYTE* p1, const BYTE* p2, const BYTE* mask,
  const int p1_pitch, const int p2_pitch, const int mask_pitch, const int width, const int height, const int opacity, const float opacity_f);

void overlay_blend_c_plane_masked_f(BYTE* p1, const BYTE* p2, const BYTE* mask,
  const int p1_pitch, const int p2_pitch, const int mask_pitch,
  const int width, const int height, const int /*opacity*/, const float /*opacity_f*/) {

  typedef float pixel_t;
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      pixel_t new_mask = reinterpret_cast<const pixel_t*>(mask)[x];
      pixel_t p1x = reinterpret_cast<pixel_t*>(p1)[x];
      pixel_t p2x = reinterpret_cast<const pixel_t*>(p2)[x];
      pixel_t result = p1x + (p2x - p1x) * new_mask; // p1x*(1-new_mask) + p2x*mask

      //pixel_t result = overlay_blend_c_core(reinterpret_cast<pixel_t *>(p1)[x], reinterpret_cast<pixel_t *>(p2)[x], static_cast<int>(reinterpret_cast<pixel_t *>(mask)[x]));
      reinterpret_cast<pixel_t*>(p1)[x] = result;
    }

    p1 += p1_pitch;
    p2 += p2_pitch;
    mask += mask_pitch;
  }
}

template<bool has_mask>
void overlay_blend_c_float(BYTE* p1, const BYTE* p2, const BYTE* mask,
  const int p1_pitch, const int p2_pitch, const int mask_pitch,
  const int width, const int height, const int /*opacity*/, const float opacity_f) {

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      auto new_mask = has_mask ? reinterpret_cast<const float*>(mask)[x] * opacity_f : opacity_f;
      auto p1x = reinterpret_cast<float*>(p1)[x];
      auto p2x = reinterpret_cast<const float*>(p2)[x];
      auto result = p1x + (p2x - p1x) * new_mask; // p1x*(1-new_mask) + p2x*mask
      reinterpret_cast<float*>(p1)[x] = result;
    }

    p1 += p1_pitch;
    p2 += p2_pitch;
    if constexpr (has_mask)
      mask += mask_pitch;
  }
}

// instantiate
template void overlay_blend_c_float<false>(BYTE* p1, const BYTE* p2, const BYTE* mask,
  const int p1_pitch, const int p2_pitch, const int mask_pitch, const int width, const int height, const int /*opacity*/, const float opacity_f);
template void overlay_blend_c_float<true>(BYTE* p1, const BYTE* p2, const BYTE* mask,
  const int p1_pitch, const int p2_pitch, const int mask_pitch, const int width, const int height, const int /*opacity*/, const float opacity_f);



template<typename pixel_t, int bits_per_pixel>
void overlay_blend_c_plane_masked(BYTE *p1, const BYTE *p2, const BYTE *mask,
                                  const int p1_pitch, const int p2_pitch, const int mask_pitch,
                                  const int width, const int height, const int /*opacity*/, const float /*opacity_f*/)
{
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int new_mask = reinterpret_cast<const pixel_t *>(mask)[x];
      pixel_t p1x = reinterpret_cast<pixel_t *>(p1)[x];
      pixel_t p2x = reinterpret_cast<const pixel_t *>(p2)[x];
      pixel_t result;
      if constexpr(bits_per_pixel == 8)
        result = (pixel_t)overlay_blend_c_core_8((BYTE)p1x, (BYTE)p2x, new_mask);
      else
        result = (pixel_t)overlay_blend_c_core_16<bits_per_pixel>((uint16_t)p1x, (uint16_t)p2x, new_mask);
      reinterpret_cast<pixel_t *>(p1)[x] = result;
    }

    p1   += p1_pitch;
    p2   += p2_pitch;
    mask += mask_pitch;
  }
}


// instantiate
template void overlay_blend_c_plane_masked<uint8_t, 8>(BYTE *p1, const BYTE *p2, const BYTE *mask,
  const int p1_pitch, const int p2_pitch, const int mask_pitch,
  const int width, const int height, const int opacity, const float opacity_f);
template void overlay_blend_c_plane_masked<uint16_t,10>(BYTE *p1, const BYTE *p2, const BYTE *mask,
  const int p1_pitch, const int p2_pitch, const int mask_pitch,
  const int width, const int heigh, const int opacity, const float opacity_ft);
template void overlay_blend_c_plane_masked<uint16_t,12>(BYTE *p1, const BYTE *p2, const BYTE *mask,
  const int p1_pitch, const int p2_pitch, const int mask_pitch,
  const int width, const int height, const int opacity, const float opacity_f);
template void overlay_blend_c_plane_masked<uint16_t,14>(BYTE *p1, const BYTE *p2, const BYTE *mask,
  const int p1_pitch, const int p2_pitch, const int mask_pitch,
  const int width, const int height, const int opacity, const float opacity_f);
template void overlay_blend_c_plane_masked<uint16_t,16>(BYTE *p1, const BYTE *p2, const BYTE *mask,
  const int p1_pitch, const int p2_pitch, const int mask_pitch,
  const int width, const int height, const int opacity, const float opacity_f);


template<typename pixel_t, int bits_per_pixel>
void overlay_blend_c_plane_opacity(BYTE *p1, const BYTE *p2, const BYTE* /*mask*/,
                                   const int p1_pitch, const int p2_pitch, const int /*mask_pitch*/,
                                   const int width, const int height, const int opacity, const float opacity_f) {

  AVS_UNUSED(opacity_f);

  const int OPACITY_SHIFT  = 8; // opacity always max 0..256
  const int MASK_CORR_SHIFT = OPACITY_SHIFT; // no mask, mask = opacity, 8 bits always
  const int half_pixel_value_rounding = (1 << (MASK_CORR_SHIFT - 1));

  // avoid "uint16*uint16 can't get into int32" overflows
  // no need here, opacity as mask is always 8 bit
  // typedef std::conditional < sizeof(pixel_t) == 1, int, typename std::conditional < sizeof(pixel_t) == 2, int64_t, float>::type >::type result_t;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      pixel_t p1x = reinterpret_cast<pixel_t *>(p1)[x];
      pixel_t p2x = reinterpret_cast<const pixel_t *>(p2)[x];
      pixel_t result = (pixel_t)((((p1x << MASK_CORR_SHIFT) | half_pixel_value_rounding) + (p2x-p1x)*opacity) >> MASK_CORR_SHIFT);
      //BYTE result = overlay_blend_c_core_8(p1[x], p2[x], opacity);
      reinterpret_cast<pixel_t *>(p1)[x] = result;
    }

    p1   += p1_pitch;
    p2   += p2_pitch;
  }
}

void overlay_blend_c_plane_opacity_f(BYTE *p1, const BYTE *p2, const BYTE* /*mask*/,
  const int p1_pitch, const int p2_pitch, const int /*mask_pitch*/,
  const int width, const int height,const int opacity, const float opacity_f) {

  AVS_UNUSED(opacity);

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      float p1x = reinterpret_cast<float *>(p1)[x];
      float p2x = reinterpret_cast<const float *>(p2)[x];
      float result = p1x + (p2x-p1x)*opacity_f;
      reinterpret_cast<float *>(p1)[x] = result;
    }

    p1   += p1_pitch;
    p2   += p2_pitch;
  }
}

// instantiate
template void overlay_blend_c_plane_opacity<uint8_t, 8>(BYTE *p1, const BYTE *p2, const BYTE* mask,
  const int p1_pitch, const int p2_pitch, const int mask_pitch,
  const int width, const int height, const int opacity, const float opacity_f);
template void overlay_blend_c_plane_opacity<uint16_t,10>(BYTE *p1, const BYTE *p2, const BYTE* mask,
  const int p1_pitch, const int p2_pitch, const int mask_pitch,
  const int width, const int height, const int opacity, const float opacity_f);
template void overlay_blend_c_plane_opacity<uint16_t,12>(BYTE *p1, const BYTE *p2, const BYTE* mask,
  const int p1_pitch, const int p2_pitch, const int mask_pitch,
  const int width, const int height, const int opacity, const float opacity_f);
template void overlay_blend_c_plane_opacity<uint16_t,14>(BYTE *p1, const BYTE *p2, const BYTE* mask,
  const int p1_pitch, const int p2_pitch, const int mask_pitch,
  const int width, const int height, const int opacity, const float opacity_f);
template void overlay_blend_c_plane_opacity<uint16_t,16>(BYTE *p1, const BYTE *p2, const BYTE* mask,
  const int p1_pitch, const int p2_pitch, const int mask_pitch,
  const int width, const int height, const int opacity, const float opacity_f);


template<typename pixel_t, int bits_per_pixel>
void overlay_blend_c_plane_masked_opacity(BYTE *p1, const BYTE *p2, const BYTE *mask,
                                  const int p1_pitch, const int p2_pitch, const int mask_pitch,
                                  const int width, const int height, const int opacity, const float opacity_f) {

  AVS_UNUSED(opacity_f);

  const int MASK_CORR_SHIFT = (sizeof(pixel_t) == 1) ? 8 : bits_per_pixel;
  const int OPACITY_SHIFT  = 8; // opacity always max 0..256
  const int half_pixel_value_rounding = (1 << (MASK_CORR_SHIFT - 1));

  // avoid "uint16*uint16 can't get into int32" overflows
  typedef typename std::conditional < sizeof(pixel_t) == 1, int, typename std::conditional < sizeof(pixel_t) == 2, int64_t, float>::type >::type result_t;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int new_mask = (reinterpret_cast<const pixel_t *>(mask)[x] * opacity) >> OPACITY_SHIFT; // int is enough, opacity is 8 bits
      result_t p1x = reinterpret_cast<pixel_t *>(p1)[x];
      pixel_t p2x = reinterpret_cast<const pixel_t *>(p2)[x];

      pixel_t result = (pixel_t)((((p1x << MASK_CORR_SHIFT) | half_pixel_value_rounding) + (p2x-p1x)*new_mask) >> MASK_CORR_SHIFT);
      //int new_mask = overlay_merge_mask_c<pixel_t, intermediate_result_t, bits_per_pixel>(mask[x], opacity);
      //BYTE result = overlay_blend_c_core_8(p1[x], p2[x], static_cast<int>(new_mask));
      reinterpret_cast<pixel_t *>(p1)[x] = result;
    }

    p1   += p1_pitch;
    p2   += p2_pitch;
    mask += mask_pitch;
  }
}

// instantiate
template void overlay_blend_c_plane_masked_opacity<uint8_t, 8>(BYTE *p1, const BYTE *p2, const BYTE *mask,
  const int p1_pitch, const int p2_pitch, const int mask_pitch,
  const int width, const int height, const int opacity, const float opacity_f);
template void overlay_blend_c_plane_masked_opacity<uint16_t,10>(BYTE *p1, const BYTE *p2, const BYTE *mask,
  const int p1_pitch, const int p2_pitch, const int mask_pitch,
  const int width, const int height, const int opacity, const float opacity_f);
template void overlay_blend_c_plane_masked_opacity<uint16_t,12>(BYTE *p1, const BYTE *p2, const BYTE *mask,
  const int p1_pitch, const int p2_pitch, const int mask_pitch,
  const int width, const int height, const int opacity, const float opacity_f);
template void overlay_blend_c_plane_masked_opacity<uint16_t,14>(BYTE *p1, const BYTE *p2, const BYTE *mask,
  const int p1_pitch, const int p2_pitch, const int mask_pitch,
  const int width, const int height, const int opacity, const float opacity_f);
template void overlay_blend_c_plane_masked_opacity<uint16_t,16>(BYTE *p1, const BYTE *p2, const BYTE *mask,
  const int p1_pitch, const int p2_pitch, const int mask_pitch,
  const int width, const int height, const int opacity, const float opacity_f);

void overlay_blend_c_plane_masked_opacity_f(BYTE *p1, const BYTE *p2, const BYTE *mask,
  const int p1_pitch, const int p2_pitch, const int mask_pitch,
  const int width, const int height, const int opacity, const float opacity_f) {

  AVS_UNUSED(opacity);

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      float new_mask = (reinterpret_cast<const float *>(mask)[x] * opacity_f);
      float p1x = reinterpret_cast<float *>(p1)[x];
      float p2x = reinterpret_cast<const float *>(p2)[x];

      float result = p1x + (p2x-p1x)*new_mask;
      reinterpret_cast<float *>(p1)[x] = result;
    }

    p1   += p1_pitch;
    p2   += p2_pitch;
    mask += mask_pitch;
  }
}



/***************************************
 ********* Mode: Lighten/Darken ********
 ***************************************/

typedef int (OverlayCCompare)(BYTE, BYTE);

template<typename pixel_t, bool darken /* OverlayCCompare<pixel_t> compare*/>
AVS_FORCEINLINE void overlay_darklighten_c(BYTE *p1Y_8, BYTE *p1U_8, BYTE *p1V_8, const BYTE *p2Y_8, const BYTE *p2U_8, const BYTE *p2V_8, int p1_pitch, int p2_pitch, int width, int height) {
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
    for (int x = 0; x < width; x++) {
      int mask = darken ? (p2Y[x] <= p1Y[x]) : (p2Y[x] >= p1Y[x]); // compare(p1Y[x], p2Y[x]);
      p1Y[x] = overlay_blend_opaque_c_core<pixel_t>(p1Y[x], p2Y[x], mask);
      p1U[x] = overlay_blend_opaque_c_core<pixel_t>(p1U[x], p2U[x], mask);
      p1V[x] = overlay_blend_opaque_c_core<pixel_t>(p1V[x], p2V[x], mask);
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
