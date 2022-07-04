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


// Avisynth filter: YUV merge / Swap planes
// by Klaus Post (kp@interact.dk)
// adapted by Richard Berg (avisynth-dev@richardberg.net)
// iSSE code by Ian Brabham

#include <avisynth.h>
#include "merge.h"
#ifdef INTEL_INTRINSICS
#include "intel/merge_sse.h"
#include "intel/merge_avx2.h"
#endif
#include "../core/internal.h"
#include "avs/alignment.h"
#include <cstdint>


/* -----------------------------------
 *     weighted_merge_chroma_yuy2
 * -----------------------------------
 */
static void weighted_merge_chroma_yuy2_c(BYTE *src, const BYTE *chroma, int pitch, int chroma_pitch,int width, int height, int weight, int invweight) {
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; x+=2) {
      src[x+1] = (chroma[x+1] * weight + src[x+1] * invweight + 16384) >> 15;
    }
    src+=pitch;
    chroma+=chroma_pitch;
  }
}


/* -----------------------------------
 *      weighted_merge_luma_yuy2
 * -----------------------------------
 */
static void weighted_merge_luma_yuy2_c(BYTE *src, const BYTE *luma, int pitch, int luma_pitch,int width, int height, int weight, int invweight) {
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; x+=2) {
      src[x] = (luma[x] * weight + src[x] * invweight + 16384) >> 15;
    }
    src+=pitch;
    luma+=luma_pitch;
  }
}


/* -----------------------------------
 *          replace_luma_yuy2
 * -----------------------------------
 */
static void replace_luma_yuy2_c(BYTE *src, const BYTE *luma, int pitch, int luma_pitch,int width, int height ) {
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; x+=2) {
      src[x] = luma[x];
    }
    src+=pitch;
    luma+=luma_pitch;
  }
}


/* -----------------------------------
 *            average_plane
 * -----------------------------------
 */
// for uint8_t and uint16_t
template<typename pixel_t>
static void average_plane_c(BYTE *p1, const BYTE *p2, int p1_pitch, int p2_pitch, int rowsize, int height) {
  for (int y = 0; y < height; ++y) {
    for (size_t x = 0; x < rowsize / sizeof(pixel_t); ++x) {
      reinterpret_cast<pixel_t *>(p1)[x] = (int(reinterpret_cast<pixel_t *>(p1)[x]) + reinterpret_cast<const pixel_t *>(p2)[x] + 1) >> 1;
    }
    p1 += p1_pitch;
    p2 += p2_pitch;
  }
}
// for float
static void average_plane_c_float(BYTE *p1, const BYTE *p2, int p1_pitch, int p2_pitch, int rowsize, int height) {

  size_t rs = rowsize / sizeof(float);

  for (int y = 0; y < height; ++y) {
    for (size_t x = 0; x < rs; ++x) {
      reinterpret_cast<float *>(p1)[x] = (reinterpret_cast<float *>(p1)[x] + reinterpret_cast<const float *>(p2)[x]) / 2.0f;
    }
    p1 += p1_pitch;
    p2 += p2_pitch;
  }
}

/* -----------------------------------
 *       weighted_merge_planar
 * -----------------------------------
 */

template<typename pixel_t>
void weighted_merge_planar_c(BYTE *p1, const BYTE *p2, int p1_pitch, int p2_pitch,int rowsize, int height, float weight_f, int weight_i, int invweight_i) {
  AVS_UNUSED(weight_f);
  for (int y=0;y<height;y++) {
    for (size_t x=0;x<rowsize / sizeof(pixel_t);x++) {
      (reinterpret_cast<pixel_t *>(p1))[x] = ((reinterpret_cast<pixel_t *>(p1))[x]*invweight_i + (reinterpret_cast<const pixel_t *>(p2))[x]*weight_i + 32768) >> 16;
    }
    p2+=p2_pitch;
    p1+=p1_pitch;
  }
}

void weighted_merge_planar_c_float(BYTE *p1, const BYTE *p2, int p1_pitch, int p2_pitch, int rowsize, int height, float weight_f, int weight_i, int invweight_i) {
  AVS_UNUSED(weight_i);
  AVS_UNUSED(invweight_i);
  float invweight_f = 1.0f - weight_f;
  size_t rs = rowsize / sizeof(float);

  for (int y = 0; y < height; ++y) {
    for (size_t x = 0; x < rs; ++x) {
      reinterpret_cast<float *>(p1)[x] = (reinterpret_cast<float *>(p1)[x] * invweight_f + reinterpret_cast<const float *>(p2)[x] * weight_f);
    }
    p1 += p1_pitch;
    p2 += p2_pitch;
  }
}


/********************************************************************
***** Declare index of new filters for Avisynth's filter engine *****
********************************************************************/
extern const AVSFunction Merge_filters[] = {
  { "Merge",       BUILTIN_FUNC_PREFIX, "cc[weight]f", MergeAll::Create },  // src, src2, weight
  { "MergeChroma", BUILTIN_FUNC_PREFIX, "cc[weight]f", MergeChroma::Create },  // src, chroma src, weight
  { "MergeChroma", BUILTIN_FUNC_PREFIX, "cc[chromaweight]f", MergeChroma::Create },  // Legacy!
  { "MergeLuma",   BUILTIN_FUNC_PREFIX, "cc[weight]f", MergeLuma::Create },      // src, luma src, weight
  { "MergeLuma",   BUILTIN_FUNC_PREFIX, "cc[lumaweight]f", MergeLuma::Create },      // Legacy!
  { 0 }
};

// also returns the proper integer weight/inverse weight for 8-16 bits
#ifdef INTEL_INTRINSICS
MergeFuncPtr getMergeFunc(int bits_per_pixel, int cpuFlags, BYTE *srcp, const BYTE *otherp, float weight_f, int &weight_i, int &invweight_i)
#else
MergeFuncPtr getMergeFunc(int bits_per_pixel, BYTE *srcp, const BYTE *otherp, float weight_f, int &weight_i, int &invweight_i)
#endif
{
  const int pixelsize = bits_per_pixel == 8 ? 1 : (bits_per_pixel == 32 ? 4 : 2);

  // SIMD 8-16 bit: bitshift 15 integer arithmetic
  // C    8-16 bit: bitshift 16 integer arithmetic
  // SIMD/C Float: original float weight

  // set basic 8-16bit SIMD
  weight_i = (int)(weight_f * 32767.0f + 0.5f);
  invweight_i = 32767 - weight_i;

  if (pixelsize == 1) {
#ifdef INTEL_INTRINSICS
    if (cpuFlags & CPUF_AVX2)
      return &weighted_merge_planar_avx2;
    if (cpuFlags & CPUF_SSE2)
      return &weighted_merge_planar_sse2;
#ifdef X86_32
    if (cpuFlags & CPUF_MMX)
      return &weighted_merge_planar_mmx;
#endif
#endif
    // C: different scale!
    weight_i = (int)(weight_f * 65535.0f + 0.5f);
    invweight_i = 65535 - weight_i;
    return &weighted_merge_planar_c<uint8_t>;
  }
  if (pixelsize == 2) {
#ifdef INTEL_INTRINSICS
    if (cpuFlags & CPUF_AVX2)
    {
      if (bits_per_pixel == 16)
        return &weighted_merge_planar_uint16_avx2<false>;
      return &weighted_merge_planar_uint16_avx2<true>;
    }
    if (cpuFlags & CPUF_SSE2)
    {
      if (bits_per_pixel == 16)
        return &weighted_merge_planar_uint16_sse2<false>;
      return &weighted_merge_planar_uint16_sse2<true>;
    }
#endif
    // C: different scale!
    weight_i = (int)(weight_f * 65535.0f + 0.5f);
    invweight_i = 65535 - weight_i;
    return &weighted_merge_planar_c<uint16_t>;
  }

  // pixelsize == 4
#ifdef INTEL_INTRINSICS
  if (cpuFlags & CPUF_SSE2)
    return &weighted_merge_planar_sse2_float;
#endif
  return &weighted_merge_planar_c_float;
}

static void merge_plane(BYTE* srcp, const BYTE* otherp, int src_pitch, int other_pitch, int src_rowsize, int src_height, float weight, int pixelsize, int bits_per_pixel, IScriptEnvironment* env) {
  if ((weight > 0.4961f) && (weight < 0.5039f))
  {
    //average of two planes
    if (pixelsize != 4) // 1 or 2
    {
#ifdef INTEL_INTRINSICS
      if (env->GetCPUFlags() & CPUF_AVX2) {
        if (pixelsize == 1)
          average_plane_avx2<uint8_t>(srcp, otherp, src_pitch, other_pitch, src_rowsize, src_height);
        else // pixel_size==2
          average_plane_avx2<uint16_t>(srcp, otherp, src_pitch, other_pitch, src_rowsize, src_height);
      }
      else if (env->GetCPUFlags() & CPUF_SSE2) {
        if (pixelsize == 1)
          average_plane_sse2<uint8_t>(srcp, otherp, src_pitch, other_pitch, src_rowsize, src_height);
        else // pixel_size==2
          average_plane_sse2<uint16_t>(srcp, otherp, src_pitch, other_pitch, src_rowsize, src_height);
      }
      else
#ifdef X86_32
        if (env->GetCPUFlags() & CPUF_INTEGER_SSE) {
          if (pixelsize == 1)
            average_plane_isse<uint8_t>(srcp, otherp, src_pitch, other_pitch, src_rowsize, src_height);
          else // pixel_size==2
            average_plane_isse<uint16_t>(srcp, otherp, src_pitch, other_pitch, src_rowsize, src_height);
        }
        else
#endif
#endif
        {
          if (pixelsize == 1)
            average_plane_c<uint8_t>(srcp, otherp, src_pitch, other_pitch, src_rowsize, src_height);
          else // pixel_size==2
            average_plane_c<uint16_t>(srcp, otherp, src_pitch, other_pitch, src_rowsize, src_height);
        }
    }
    else { // if (pixelsize == 4)
#ifdef INTEL_INTRINSICS
      if (env->GetCPUFlags() & CPUF_SSE2)
        average_plane_sse2_float(srcp, otherp, src_pitch, other_pitch, src_rowsize, src_height);
      else
#endif
        average_plane_c_float(srcp, otherp, src_pitch, other_pitch, src_rowsize, src_height);
    }
  }
  else
  {
    int weight_i;
    int invweight_i;
#ifdef INTEL_INTRINSICS
    MergeFuncPtr weighted_merge_planar = getMergeFunc(bits_per_pixel, env->GetCPUFlags(), srcp, otherp, weight, /*out*/weight_i, /*out*/invweight_i);
#else
    MergeFuncPtr weighted_merge_planar = getMergeFunc(bits_per_pixel, srcp, otherp, weight, /*out*/weight_i, /*out*/invweight_i);
#endif
    weighted_merge_planar(srcp, otherp, src_pitch, other_pitch, src_rowsize, src_height, weight, weight_i, invweight_i);
  }
}

/****************************
******   Merge Chroma   *****
****************************/

MergeChroma::MergeChroma(PClip _child, PClip _clip, float _weight, IScriptEnvironment* env)
  : GenericVideoFilter(_child), clip(_clip), weight(_weight)
{
  const VideoInfo& vi2 = clip->GetVideoInfo();

  if (!(vi.IsYUV() || vi.IsYUVA()) || !(vi2.IsYUV() || vi2.IsYUVA()))
    env->ThrowError("MergeChroma: YUV data only (no RGB); use ConvertToYUY2, ConvertToYV12/16/24 or ConvertToYUVxxx");

  if (!(vi.IsSameColorspace(vi2)))
    env->ThrowError("MergeChroma: YUV images must have same data type.");

  if (vi.width!=vi2.width || vi.height!=vi2.height)
    env->ThrowError("MergeChroma: Images must have same width and height!");

  if (weight<0.0f) weight=0.0f;
  if (weight>1.0f) weight=1.0f;

  pixelsize = vi.ComponentSize();
  bits_per_pixel = vi.BitsPerComponent();
}


PVideoFrame __stdcall MergeChroma::GetFrame(int n, IScriptEnvironment* env)
{
  PVideoFrame src = child->GetFrame(n, env);

  if (weight < 0.0039f) return src;

  PVideoFrame chroma = clip->GetFrame(n, env);

  int h = src->GetHeight();
  int w = src->GetRowSize(); // width in pixels

  if (weight < 0.9961f) {
    if (vi.IsYUY2()) {
      env->MakeWritable(&src);
      BYTE* srcp = src->GetWritePtr();
      const BYTE* chromap = chroma->GetReadPtr();

      int src_pitch = src->GetPitch();
      int chroma_pitch = chroma->GetPitch();
#ifdef INTEL_INTRINSICS
      if (env->GetCPUFlags() & CPUF_SSE2)
      {
        weighted_merge_chroma_yuy2_sse2(srcp, chromap, src_pitch, chroma_pitch, w, h, (int)(weight * 32768.0f), 32768 - (int)(weight * 32768.0f));
      }
      else
#ifdef X86_32
        if (env->GetCPUFlags() & CPUF_MMX)
        {
          weighted_merge_chroma_yuy2_mmx(srcp, chromap, src_pitch, chroma_pitch, w, h, (int)(weight * 32768.0f), 32768 - (int)(weight * 32768.0f));
        }
        else
#endif
#endif
        {
          weighted_merge_chroma_yuy2_c(srcp, chromap, src_pitch, chroma_pitch, w, h, (int)(weight * 32768.0f), 32768 - (int)(weight * 32768.0f));
        }
    }
    else {  // Planar YUV
      env->MakeWritable(&src);
      src->GetWritePtr(PLANAR_Y); //Must be requested

      BYTE* srcpU = (BYTE*)src->GetWritePtr(PLANAR_U);
      BYTE* chromapU = (BYTE*)chroma->GetReadPtr(PLANAR_U);
      BYTE* srcpV = (BYTE*)src->GetWritePtr(PLANAR_V);
      BYTE* chromapV = (BYTE*)chroma->GetReadPtr(PLANAR_V);
      int src_pitch_uv = src->GetPitch(PLANAR_U);
      int chroma_pitch_uv = chroma->GetPitch(PLANAR_U);
      int src_rowsize_u = src->GetRowSize(PLANAR_U_ALIGNED);
      int src_rowsize_v = src->GetRowSize(PLANAR_V_ALIGNED);
      int src_height_uv = src->GetHeight(PLANAR_U);

      merge_plane(srcpU, chromapU, src_pitch_uv, chroma_pitch_uv, src_rowsize_u, src_height_uv, weight, pixelsize, bits_per_pixel, env);
      merge_plane(srcpV, chromapV, src_pitch_uv, chroma_pitch_uv, src_rowsize_v, src_height_uv, weight, pixelsize, bits_per_pixel, env);

      if (vi.IsYUVA())
        merge_plane(src->GetWritePtr(PLANAR_A), chroma->GetReadPtr(PLANAR_A), src->GetPitch(PLANAR_A), chroma->GetPitch(PLANAR_A),
          src->GetRowSize(PLANAR_A_ALIGNED), src->GetHeight(PLANAR_A), weight, pixelsize, bits_per_pixel, env);
    }
  }
  else { // weight == 1.0
    if (vi.IsYUY2()) {
      const BYTE* srcp = src->GetReadPtr();
      env->MakeWritable(&chroma);
      BYTE* chromap = chroma->GetWritePtr();

      int src_pitch = src->GetPitch();
      int chroma_pitch = chroma->GetPitch();
#ifdef INTEL_INTRINSICS
      if (env->GetCPUFlags() & CPUF_SSE2)
      {
        replace_luma_yuy2_sse2(chromap, srcp, chroma_pitch, src_pitch, w, h);  // Just swap luma/chroma
      }
      else
#ifdef X86_32
        if (env->GetCPUFlags() & CPUF_MMX)
        {
          replace_luma_yuy2_mmx(chromap, srcp, chroma_pitch, src_pitch, w, h);  // Just swap luma/chroma
        }
        else
#endif
#endif
        {
          replace_luma_yuy2_c(chromap, srcp, chroma_pitch, src_pitch, w, h);  // Just swap luma/chroma
        }

      return chroma;
    }
    else {
      if (src->IsWritable()) {
        src->GetWritePtr(PLANAR_Y); //Must be requested
        env->BitBlt(src->GetWritePtr(PLANAR_U), src->GetPitch(PLANAR_U), chroma->GetReadPtr(PLANAR_U), chroma->GetPitch(PLANAR_U), chroma->GetRowSize(PLANAR_U), chroma->GetHeight(PLANAR_U));
        env->BitBlt(src->GetWritePtr(PLANAR_V), src->GetPitch(PLANAR_V), chroma->GetReadPtr(PLANAR_V), chroma->GetPitch(PLANAR_V), chroma->GetRowSize(PLANAR_V), chroma->GetHeight(PLANAR_V));
        if (vi.IsYUVA())
          env->BitBlt(src->GetWritePtr(PLANAR_A), src->GetPitch(PLANAR_A), chroma->GetReadPtr(PLANAR_A), chroma->GetPitch(PLANAR_A), chroma->GetRowSize(PLANAR_A), chroma->GetHeight(PLANAR_A));
      }
      else { // avoid the cost of 2 chroma blits
        PVideoFrame dst = env->NewVideoFrameP(vi, &src);

        env->BitBlt(dst->GetWritePtr(PLANAR_Y), dst->GetPitch(PLANAR_Y), src->GetReadPtr(PLANAR_Y), src->GetPitch(PLANAR_Y), src->GetRowSize(PLANAR_Y), src->GetHeight(PLANAR_Y));
        env->BitBlt(dst->GetWritePtr(PLANAR_U), dst->GetPitch(PLANAR_U), chroma->GetReadPtr(PLANAR_U), chroma->GetPitch(PLANAR_U), chroma->GetRowSize(PLANAR_U), chroma->GetHeight(PLANAR_U));
        env->BitBlt(dst->GetWritePtr(PLANAR_V), dst->GetPitch(PLANAR_V), chroma->GetReadPtr(PLANAR_V), chroma->GetPitch(PLANAR_V), chroma->GetRowSize(PLANAR_V), chroma->GetHeight(PLANAR_V));
        if (vi.IsYUVA())
          env->BitBlt(dst->GetWritePtr(PLANAR_A), dst->GetPitch(PLANAR_A), chroma->GetReadPtr(PLANAR_A), chroma->GetPitch(PLANAR_A), chroma->GetRowSize(PLANAR_A), chroma->GetHeight(PLANAR_A));

        return dst;
      }
    }
  }
  return src;
}


AVSValue __cdecl MergeChroma::Create(AVSValue args, void* , IScriptEnvironment* env)
{
  return new MergeChroma(args[0].AsClip(), args[1].AsClip(), (float)args[2].AsFloat(1.0f), env);
}


/**************************
******   Merge Luma   *****
**************************/


MergeLuma::MergeLuma(PClip _child, PClip _clip, float _weight, IScriptEnvironment* env)
  : GenericVideoFilter(_child), clip(_clip), weight(_weight)
{
  const VideoInfo& vi2 = clip->GetVideoInfo();

  if (!(vi.IsYUV() || vi.IsYUVA()) || !(vi2.IsYUV() || vi2.IsYUVA()))
    env->ThrowError("MergeLuma: YUV data only (no RGB); use ConvertToYUY2, ConvertToYV12/16/24 or ConvertToYUVxxx");

  pixelsize = vi.ComponentSize();
  bits_per_pixel = vi.BitsPerComponent();

  if (!vi.IsSameColorspace(vi2)) {  // Since this is luma we allow all planar formats to be merged.
    if (!(vi.IsPlanar() && vi2.IsPlanar())) {
      env->ThrowError("MergeLuma: YUV data is not same type. YUY2 and planar images doesn't mix.");
    }
    if (pixelsize != vi2.ComponentSize()) {
      env->ThrowError("MergeLuma: YUV data bit depth is not same.");
    }
  }

  if (vi.width!=vi2.width || vi.height!=vi2.height)
    env->ThrowError("MergeLuma: Images must have same width and height!");

  if (weight<0.0f) weight=0.0f;
  if (weight>1.0f) weight=1.0f;

}


PVideoFrame __stdcall MergeLuma::GetFrame(int n, IScriptEnvironment* env)
{
  PVideoFrame src = child->GetFrame(n, env);

  if (weight < 0.0039f) return src;

  PVideoFrame luma = clip->GetFrame(n, env);

  if (vi.IsYUY2()) {
    env->MakeWritable(&src);
    BYTE* srcp = src->GetWritePtr();
    const BYTE* lumap = luma->GetReadPtr();

    int isrc_pitch = src->GetPitch();
    int iluma_pitch = luma->GetPitch();

    int h = src->GetHeight();
    int w = src->GetRowSize();

    if (weight < 0.9961f) {
#ifdef INTEL_INTRINSICS
      if (env->GetCPUFlags() & CPUF_SSE2)
      {
        weighted_merge_luma_yuy2_sse2(srcp, lumap, isrc_pitch, iluma_pitch, w, h, (int)(weight * 32768.0f), 32768 - (int)(weight * 32768.0f));
      }
      else
#ifdef X86_32
        if (env->GetCPUFlags() & CPUF_MMX)
        {
          weighted_merge_luma_yuy2_mmx(srcp, lumap, isrc_pitch, iluma_pitch, w, h, (int)(weight * 32768.0f), 32768 - (int)(weight * 32768.0f));
        }
        else
#endif
#endif
        {
          weighted_merge_luma_yuy2_c(srcp, lumap, isrc_pitch, iluma_pitch, w, h, (int)(weight * 32768.0f), 32768 - (int)(weight * 32768.0f));
        }
    }
    else {
#ifdef INTEL_INTRINSICS
      if (env->GetCPUFlags() & CPUF_SSE2)
      {
        replace_luma_yuy2_sse2(srcp, lumap, isrc_pitch, iluma_pitch, w, h);
      }
      else
#ifdef X86_32
        if (env->GetCPUFlags() & CPUF_MMX)
        {
          replace_luma_yuy2_mmx(srcp, lumap, isrc_pitch, iluma_pitch, w, h);
        }
        else
#endif
#endif
        {
          replace_luma_yuy2_c(srcp, lumap, isrc_pitch, iluma_pitch, w, h);
        }
    }
    return src;
  }  // Planar
  if (weight > 0.9961f) {
    // 2nd clip weight is almost 100%: no merge, just copy
    const VideoInfo& vi2 = clip->GetVideoInfo();
    if (luma->IsWritable() && vi.IsSameColorspace(vi2)) {
      if (luma->GetRowSize(PLANAR_U)) {
        luma->GetWritePtr(PLANAR_Y); //Must be requested BUT only if we actually do something
        env->BitBlt(luma->GetWritePtr(PLANAR_U), luma->GetPitch(PLANAR_U), src->GetReadPtr(PLANAR_U), src->GetPitch(PLANAR_U), src->GetRowSize(PLANAR_U), src->GetHeight(PLANAR_U));
        env->BitBlt(luma->GetWritePtr(PLANAR_V), luma->GetPitch(PLANAR_V), src->GetReadPtr(PLANAR_V), src->GetPitch(PLANAR_V), src->GetRowSize(PLANAR_V), src->GetHeight(PLANAR_V));
      }
      if (luma->GetPitch(PLANAR_A)) // copy Alpha if exists
        env->BitBlt(luma->GetWritePtr(PLANAR_A), luma->GetPitch(PLANAR_A), src->GetReadPtr(PLANAR_A), src->GetPitch(PLANAR_A), src->GetRowSize(PLANAR_A), src->GetHeight(PLANAR_A));

      return luma;
    }
    else { // avoid the cost of 2 chroma blits
      PVideoFrame dst = env->NewVideoFrameP(vi, &luma);

      env->BitBlt(dst->GetWritePtr(PLANAR_Y), dst->GetPitch(PLANAR_Y), luma->GetReadPtr(PLANAR_Y), luma->GetPitch(PLANAR_Y), luma->GetRowSize(PLANAR_Y), luma->GetHeight(PLANAR_Y));
      if (src->GetRowSize(PLANAR_U) && dst->GetRowSize(PLANAR_U)) {
        env->BitBlt(dst->GetWritePtr(PLANAR_U), dst->GetPitch(PLANAR_U), src->GetReadPtr(PLANAR_U), src->GetPitch(PLANAR_U), src->GetRowSize(PLANAR_U), src->GetHeight(PLANAR_U));
        env->BitBlt(dst->GetWritePtr(PLANAR_V), dst->GetPitch(PLANAR_V), src->GetReadPtr(PLANAR_V), src->GetPitch(PLANAR_V), src->GetRowSize(PLANAR_V), src->GetHeight(PLANAR_V));
      }
      if (dst->GetPitch(PLANAR_A) && src->GetPitch(PLANAR_A)) // copy Alpha if in both clip exists
        env->BitBlt(dst->GetWritePtr(PLANAR_A), dst->GetPitch(PLANAR_A), src->GetReadPtr(PLANAR_A), src->GetPitch(PLANAR_A), src->GetRowSize(PLANAR_A), src->GetHeight(PLANAR_A));

      return dst;
    }
  }
  else { // weight <= 0.9961f
    env->MakeWritable(&src);
    BYTE* srcpY = (BYTE*)src->GetWritePtr(PLANAR_Y);
    BYTE* lumapY = (BYTE*)luma->GetReadPtr(PLANAR_Y);
    int src_pitch = src->GetPitch(PLANAR_Y);
    int luma_pitch = luma->GetPitch(PLANAR_Y);
    int src_rowsize = src->GetRowSize(PLANAR_Y);
    int src_height = src->GetHeight(PLANAR_Y);

    merge_plane(srcpY, lumapY, src_pitch, luma_pitch, src_rowsize, src_height, weight, pixelsize, bits_per_pixel, env);
  }

  return src;
}


AVSValue __cdecl MergeLuma::Create(AVSValue args, void* , IScriptEnvironment* env)
{
  return new MergeLuma(args[0].AsClip(), args[1].AsClip(), (float)args[2].AsFloat(1.0f), env);
}


/*************************
******   Merge All   *****
*************************/


MergeAll::MergeAll(PClip _child, PClip _clip, float _weight, IScriptEnvironment* env)
  : GenericVideoFilter(_child), clip(_clip), weight(_weight)
{
  const VideoInfo& vi2 = clip->GetVideoInfo();

  if (!vi.IsSameColorspace(vi2))
    env->ThrowError("Merge: Pixel types are not the same. Both must be the same.");

  if (vi.width!=vi2.width || vi.height!=vi2.height)
    env->ThrowError("Merge: Images must have same width and height!");

  pixelsize = vi.ComponentSize();
  bits_per_pixel = vi.BitsPerComponent();

  if (weight<0.0f) weight=0.0f;
  if (weight>1.0f) weight=1.0f;
}


PVideoFrame __stdcall MergeAll::GetFrame(int n, IScriptEnvironment* env)
{
  if (weight<0.0039f) return child->GetFrame(n, env);
  if (weight>0.9961f) return clip->GetFrame(n, env);

  PVideoFrame src  = child->GetFrame(n, env);
  PVideoFrame src2 =  clip->GetFrame(n, env);

  env->MakeWritable(&src);
  BYTE* srcp  = src->GetWritePtr();
  const BYTE* srcp2 = src2->GetReadPtr();

  const int src_pitch = src->GetPitch();
  const int src_rowsize = src->GetRowSize();

  merge_plane(srcp, srcp2, src_pitch, src2->GetPitch(), src_rowsize, src->GetHeight(), weight, pixelsize, bits_per_pixel, env);

  if (vi.IsPlanar()) {
    const int planesYUV[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A};
    const int planesRGB[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A};
    const int *planes = (vi.IsYUV() || vi.IsYUVA()) ? planesYUV : planesRGB;
    // first plane is already processed
    for (int p = 1; p < vi.NumComponents(); p++) {
      const int plane = planes[p];
      merge_plane(src->GetWritePtr(plane), src2->GetReadPtr(plane), src->GetPitch(plane), src2->GetPitch(plane), src->GetRowSize(plane), src->GetHeight(plane), weight, pixelsize, bits_per_pixel, env);
    }
  }

  return src;
}


AVSValue __cdecl MergeAll::Create(AVSValue args, void* , IScriptEnvironment* env)
{
  return new MergeAll(args[0].AsClip(), args[1].AsClip(), (float)args[2].AsFloat(0.5f), env);
}
