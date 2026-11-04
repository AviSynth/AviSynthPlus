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

#ifndef __blend_common_sse_h
#define __blend_common_sse_h

#include <avs/types.h>
#include "../blend_common.h"  // MaskMode, masked_merge_fn_t

// SSE2 masked float merge — opacity_f in 0..1; mask values in 0..1.
// result = p1*(1-mask*opacity) + p2*(mask*opacity)
void masked_merge_float_sse2(BYTE* p1, const BYTE* p2, const BYTE* mask,
  int p1_pitch, int p2_pitch, int mask_pitch,
  int width, int height, float opacity_f);

#ifdef X86_32
void overlay_darken_mmx(BYTE* p1Y, BYTE* p1U, BYTE* p1V, const BYTE* p2Y, const BYTE* p2U, const BYTE* p2V, int p1_pitch, int p2_pitch, int width, int height);
void overlay_lighten_mmx(BYTE* p1Y, BYTE* p1U, BYTE* p1V, const BYTE* p2Y, const BYTE* p2U, const BYTE* p2V, int p1_pitch, int p2_pitch, int width, int height);
#endif

void overlay_darken_sse2(BYTE* p1Y, BYTE* p1U, BYTE* p1V, const BYTE* p2Y, const BYTE* p2U, const BYTE* p2V, int p1_pitch, int p2_pitch, int width, int height);
void overlay_lighten_sse2(BYTE* p1Y, BYTE* p1U, BYTE* p1V, const BYTE* p2Y, const BYTE* p2U, const BYTE* p2V, int p1_pitch, int p2_pitch, int width, int height);

#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
void overlay_darken_sse41(BYTE* p1Y, BYTE* p1U, BYTE* p1V, const BYTE* p2Y, const BYTE* p2U, const BYTE* p2V, int p1_pitch, int p2_pitch, int width, int height);
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
void overlay_lighten_sse41(BYTE* p1Y, BYTE* p1U, BYTE* p1V, const BYTE* p2Y, const BYTE* p2U, const BYTE* p2V, int p1_pitch, int p2_pitch, int width, int height);

// SSE2 weighted merge — Family 1 (no mask, flat weight, >> 15 shift).
// weight + invweight == 32768; boundary values (0, 32768) are caller early-outs.
// Integer: bits_per_pixel in {8, 10, 12, 14, 16}; width in pixels.
// Float:   weight_f in 0..1; width in pixels.
void weighted_merge_sse2(BYTE* p1, const BYTE* p2, int p1_pitch, int p2_pitch,
  int width, int height, int weight, int invweight, int bits_per_pixel);

void weighted_merge_float_sse2(BYTE* p1, const BYTE* p2, int p1_pitch, int p2_pitch,
  int width, int height, float weight_f);

// Overlay blend masked getter — returns masked_merge_sse41_impl instantiation.
// is_chroma=false → always MASK444 (luma).
// is_chroma=true  → placement-aware maskMode (chroma).
masked_merge_fn_t* get_overlay_blend_masked_fn_sse41(bool is_chroma, MaskMode maskMode);
masked_merge_float_fn_t* get_overlay_blend_masked_float_fn_sse41(bool is_chroma, MaskMode maskMode);

// ---------------------------------------------------------------------------
// Per-row chroma mask preparation (scratch path). Defined in blend_common_sse.cpp.
// full_opacity=true: spatial averaging only.
// full_opacity=false: opacity baked in (result = (avg * opacity_i + half) / max).
// Only uint8_t and uint16_t instantiations are provided.
// ---------------------------------------------------------------------------
template<typename pixel_t, bool full_opacity = true>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
void do_fill_chroma_row_sse41(
  pixel_t* buf, const pixel_t* luma_row,
  int luma_pitch_pixels, int chroma_w, MaskMode mode,
  int opacity_i = 0, int half = 0, MagicDiv magic = {});

template<bool full_opacity = true>
void do_fill_chroma_row_float_sse41(
  float* buf, const float* luma_row,
  int luma_pitch_pixels, int chroma_w, MaskMode mode,
  float opacity = 0.0f);

#endif // __blend_common_sse_h
