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

#include "overlayfunctions.h"

#include "blend_common.h"
#ifdef INTEL_INTRINSICS
#include "intel/blend_common_sse.h"
#endif

#include <stdint.h>
#include <type_traits>

void OL_DarkenImage::DoBlendImageMask(ImageOverlayInternal* base, ImageOverlayInternal* overlay, ImageOverlayInternal* mask) {
  if(of_mode == OF_Darken) {
    if (bits_per_pixel == 8)
      BlendImageMask<uint8_t, true, true>(base, overlay, mask);
    else if(bits_per_pixel <= 16)
      BlendImageMask<uint16_t, true, true>(base, overlay, mask);
    //else if(bits_per_pixel == 32)
    //  BlendImageMask<float>(base, overlay, mask);
  }
  else {
    // OF_Lighten
    if (bits_per_pixel == 8)
      BlendImageMask<uint8_t, true, false>(base, overlay, mask);
    else if(bits_per_pixel <= 16)
      BlendImageMask<uint16_t, true, false>(base, overlay, mask);
    //else if(bits_per_pixel == 32)
    //  BlendImageMask<float>(base, overlay, mask);
  }
}

void OL_DarkenImage::DoBlendImage(ImageOverlayInternal* base, ImageOverlayInternal* overlay) {
  if(of_mode == OF_Darken) {
    if (bits_per_pixel == 8)
      BlendImageMask<uint8_t, false, true>(base, overlay, nullptr);
    else if(bits_per_pixel <= 16)
      BlendImageMask<uint16_t, false, true>(base, overlay, nullptr);
    //else if(bits_per_pixel == 32)
    //  BlendImageMask<float>(base, overlay, mask);
  }
  else {
    // OF_Lighten
    if (bits_per_pixel == 8)
      BlendImageMask<uint8_t, false, false>(base, overlay, nullptr);
    else if(bits_per_pixel <= 16)
      BlendImageMask<uint16_t, false, false>(base, overlay, nullptr);
    //else if(bits_per_pixel == 32)
    //  BlendImageMask<float>(base, overlay, mask);
  }
}


template<typename pixel_t, bool maskMode, bool of_darken>
void OL_DarkenImage::BlendImageMask(ImageOverlayInternal* base, ImageOverlayInternal* overlay, ImageOverlayInternal* mask) {

  pixel_t* baseY = reinterpret_cast<pixel_t *>(base->GetPtr(PLANAR_Y));
  pixel_t* baseU = reinterpret_cast<pixel_t *>(base->GetPtr(PLANAR_U));
  pixel_t* baseV = reinterpret_cast<pixel_t *>(base->GetPtr(PLANAR_V));

  pixel_t* ovY = reinterpret_cast<pixel_t *>(overlay->GetPtr(PLANAR_Y));
  pixel_t* ovU = reinterpret_cast<pixel_t *>(overlay->GetPtr(PLANAR_U));
  pixel_t* ovV = reinterpret_cast<pixel_t *>(overlay->GetPtr(PLANAR_V));

  pixel_t* maskY = maskMode ? reinterpret_cast<pixel_t *>(mask->GetPtr(PLANAR_Y)) : nullptr;
  pixel_t* maskU = maskMode ? reinterpret_cast<pixel_t *>(mask->GetPtr(PLANAR_U)) : nullptr;
  pixel_t* maskV = maskMode ? reinterpret_cast<pixel_t *>(mask->GetPtr(PLANAR_V)) : nullptr;

  const int half_pixel_value = (sizeof(pixel_t) == 1) ? 128 : (1 << (bits_per_pixel - 1));
  const int max_pixel_value = (sizeof(pixel_t) == 1) ? 255 : (1 << bits_per_pixel) - 1;
  const int pixel_range = max_pixel_value + 1;
  const int MASK_CORR_SHIFT = (sizeof(pixel_t) == 1) ? 8 : bits_per_pixel;
  const int OPACITY_SHIFT  = 8; // opacity always max 0..256
  const int basepitch = (base->pitch) / sizeof(pixel_t);
  const int overlaypitch = (overlay->pitch) / sizeof(pixel_t);
  const int maskpitch = maskMode ? (mask->pitch) / sizeof(pixel_t) : 0;

  // avoid "uint16*uint16 can't get into int32" overflows
  typedef typename std::conditional < sizeof(pixel_t) == 1, int, typename std::conditional < sizeof(pixel_t) == 2, int64_t, float>::type >::type result_t;

  int w = base->w();
  int h = base->h();
  if (opacity == 256) {
    if(maskMode) {
      // opacity == 256 && maskMode
      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          bool cmp;
          if (of_darken)
            cmp = ovY[x] < baseY[x];
          else
            cmp = ovY[x] > baseY[x];
          if (cmp) {
            result_t maskYx = maskY[x];
            result_t maskUx = maskU[x];
            result_t maskVx = maskV[x];
            baseY[x] = (pixel_t)((((pixel_range - maskYx)*baseY[x]) + (maskYx * ovY[x] + half_pixel_value)) >> MASK_CORR_SHIFT);
            baseU[x] = (pixel_t)((((pixel_range - maskUx)*baseU[x]) + (maskUx * ovU[x] + half_pixel_value)) >> MASK_CORR_SHIFT);
            baseV[x] = (pixel_t)((((pixel_range - maskVx)*baseV[x]) + (maskVx * ovV[x] + half_pixel_value)) >> MASK_CORR_SHIFT);
          }
        }
        maskY += maskpitch;
        maskU += maskpitch;
        maskV += maskpitch;

        baseY += basepitch;
        baseU += basepitch;
        baseV += basepitch;

        ovY += overlaypitch;
        ovU += overlaypitch;
        ovV += overlaypitch;
      }
    } else {
      // opacity == 256 && !maskMode
      if(of_darken) {
#ifdef INTEL_INTRINSICS
        if (sizeof(pixel_t)==1 && (env->GetCPUFlags() & CPUF_SSE4_1)) {
          overlay_darken_sse41((BYTE *)baseY, (BYTE *)baseU, (BYTE *)baseV, (BYTE *)ovY, (BYTE *)ovU, (BYTE *)ovV, basepitch, overlaypitch, w, h);
        } else if (sizeof(pixel_t)==1 && (env->GetCPUFlags() & CPUF_SSE2)) {
          overlay_darken_sse2((BYTE *)baseY, (BYTE *)baseU, (BYTE *)baseV, (BYTE *)ovY, (BYTE *)ovU, (BYTE *)ovV, basepitch, overlaypitch, w, h);
        } else
  #ifdef X86_32
          if (sizeof(pixel_t)==1 && (env->GetCPUFlags() & CPUF_MMX)) {
            overlay_darken_mmx((BYTE *)baseY, (BYTE *)baseU, (BYTE *)baseV, (BYTE *)ovY, (BYTE *)ovU, (BYTE *)ovV, basepitch, overlaypitch, w, h);
          } else
  #endif
#endif
          {

            overlay_darken_c<pixel_t>((BYTE *)baseY, (BYTE *)baseU, (BYTE *)baseV, (BYTE *)ovY, (BYTE *)ovU, (BYTE *)ovV, basepitch, overlaypitch, w, h);
          }
      } else {
        // OF_Lighten
#ifdef INTEL_INTRINSICS
        if (sizeof(pixel_t)==1 && (env->GetCPUFlags() & CPUF_SSE4_1)) {
          overlay_lighten_sse41((BYTE *)baseY, (BYTE *)baseU, (BYTE *)baseV, (BYTE *)ovY, (BYTE *)ovU, (BYTE *)ovV, basepitch, overlaypitch, w, h);
        } else if (sizeof(pixel_t)==1 && (env->GetCPUFlags() & CPUF_SSE2)) {
          overlay_lighten_sse2((BYTE *)baseY, (BYTE *)baseU, (BYTE *)baseV, (BYTE *)ovY, (BYTE *)ovU, (BYTE *)ovV, basepitch, overlaypitch, w, h);
        } else
#ifdef X86_32
          if (sizeof(pixel_t)==1 && (env->GetCPUFlags() & CPUF_MMX)) {
            overlay_lighten_mmx((BYTE *)baseY, (BYTE *)baseU, (BYTE *)baseV, (BYTE *)ovY, (BYTE *)ovU, (BYTE *)ovV, basepitch, overlaypitch, w, h);
          } else
#endif
#endif
          {
            overlay_lighten_c<pixel_t>((BYTE *)baseY, (BYTE *)baseU, (BYTE *)baseV, (BYTE *)ovY, (BYTE *)ovU, (BYTE *)ovV, basepitch, overlaypitch, w, h);
          }
      }

    }
  } else {
    // opacity != 256 && maskMode
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        bool cmp;
        if (of_darken)
          cmp = ovY[x] < baseY[x];
        else
          cmp = ovY[x] > baseY[x];
        if (cmp)  {
          if (maskMode) {
            result_t mY = (maskY[x] * opacity) >> OPACITY_SHIFT;
            result_t mU = (maskU[x] * opacity) >> OPACITY_SHIFT;
            result_t mV = (maskV[x] * opacity) >> OPACITY_SHIFT;
            baseY[x] = (pixel_t)((((pixel_range - mY)*baseY[x]) + (mY*ovY[x] + half_pixel_value)) >> MASK_CORR_SHIFT);
            baseU[x] = (pixel_t)((((pixel_range - mU)*baseU[x]) + (mU*ovU[x] + half_pixel_value)) >> MASK_CORR_SHIFT);
            baseV[x] = (pixel_t)((((pixel_range - mV)*baseV[x]) + (mV*ovV[x] + half_pixel_value)) >> MASK_CORR_SHIFT);
          }
          else {
            baseY[x] = (pixel_t)(((inv_opacity*baseY[x]) + (opacity*ovY[x] + 128)) >> OPACITY_SHIFT); // 128: half 256 opacity mul rounding
            baseU[x] = (pixel_t)(((inv_opacity*baseU[x]) + (opacity*ovU[x] + 128)) >> OPACITY_SHIFT);
            baseV[x] = (pixel_t)(((inv_opacity*baseV[x]) + (opacity*ovV[x] + 128)) >> OPACITY_SHIFT);
          }
        }
      }
      baseY += basepitch;
      baseU += basepitch;
      baseV += basepitch;

      ovY += overlaypitch;
      ovU += overlaypitch;
      ovV += overlaypitch;

      if(maskMode) {
        maskY += maskpitch;
        maskU += maskpitch;
        maskV += maskpitch;
      }
    }
  }
}

