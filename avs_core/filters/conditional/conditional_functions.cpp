
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

#include "conditional_functions.h"
#include "../focus.h" // sad
#ifdef INTEL_INTRINSICS
#include "intel/conditional_functions_sse.h"
#include "../intel/focus_sse.h" // sad
#endif
#include "../../core/internal.h"
#include <avs/config.h>
#include <avs/minmax.h>
#include <avs/alignment.h>
#include <limits>
#include <algorithm>
#include <cmath>
#include "../core/AVSMap.h"

extern const AVSFunction Conditional_funtions_filters[] = {
  {  "AverageLuma",    BUILTIN_FUNC_PREFIX, "c[offset]i", AveragePlane::Create, (void *)PLANAR_Y },
  {  "AverageChromaU", BUILTIN_FUNC_PREFIX, "c[offset]i", AveragePlane::Create, (void *)PLANAR_U },
  {  "AverageChromaV", BUILTIN_FUNC_PREFIX, "c[offset]i", AveragePlane::Create, (void *)PLANAR_V },
  {  "AverageR", BUILTIN_FUNC_PREFIX, "c[offset]i", AveragePlane::Create, (void *)PLANAR_R },
  {  "AverageG", BUILTIN_FUNC_PREFIX, "c[offset]i", AveragePlane::Create, (void *)PLANAR_G },
  {  "AverageB", BUILTIN_FUNC_PREFIX, "c[offset]i", AveragePlane::Create, (void *)PLANAR_B },
  {  "AverageA", BUILTIN_FUNC_PREFIX, "c[offset]i", AveragePlane::Create, (void *)PLANAR_A },
  //{  "AverageSat","c[offset]i", AverageSat::Create }, Sum(SatLookup[U,V])/N, SatLookup[U,V]=1.4087*sqrt((U-128)**2+(V-128)**2)
//{  "AverageHue","c[offset]i", AverageHue::Create }, Sum(HueLookup[U,V])/N, HueLookup[U,V]=40.5845*Atan2(U-128,V-128)

  {  "RGBDifference",     BUILTIN_FUNC_PREFIX, "cc", ComparePlane::Create, (void *)-1 },
  {  "LumaDifference",    BUILTIN_FUNC_PREFIX, "cc", ComparePlane::Create, (void *)PLANAR_Y },
  {  "ChromaUDifference", BUILTIN_FUNC_PREFIX, "cc", ComparePlane::Create, (void *)PLANAR_U },
  {  "ChromaVDifference", BUILTIN_FUNC_PREFIX, "cc", ComparePlane::Create, (void *)PLANAR_V },
  {  "RDifference", BUILTIN_FUNC_PREFIX, "cc", ComparePlane::Create, (void *)PLANAR_R },
  {  "GDifference", BUILTIN_FUNC_PREFIX, "cc", ComparePlane::Create, (void *)PLANAR_G },
  {  "BDifference", BUILTIN_FUNC_PREFIX, "cc", ComparePlane::Create, (void *)PLANAR_B },
  //{  "SatDifference","cc", CompareSat::Create }, Sum(Abs(SatLookup[U1,V1]-SatLookup[U2,V2]))/N
//{  "HueDifference","cc", CompareHue::Create }, Sum(Abs(HueLookup[U1,V1]-HueLookup[U2,V2]))/N

  {  "YDifferenceFromPrevious",   BUILTIN_FUNC_PREFIX, "c", ComparePlane::Create_prev, (void *)PLANAR_Y },
  {  "UDifferenceFromPrevious",   BUILTIN_FUNC_PREFIX, "c", ComparePlane::Create_prev, (void *)PLANAR_U },
  {  "VDifferenceFromPrevious",   BUILTIN_FUNC_PREFIX, "c", ComparePlane::Create_prev, (void *)PLANAR_V },
  {  "RGBDifferenceFromPrevious", BUILTIN_FUNC_PREFIX, "c", ComparePlane::Create_prev, (void *)-1 },
  {  "RDifferenceFromPrevious",   BUILTIN_FUNC_PREFIX, "c", ComparePlane::Create_prev, (void *)PLANAR_R },
  {  "GDifferenceFromPrevious",   BUILTIN_FUNC_PREFIX, "c", ComparePlane::Create_prev, (void *)PLANAR_G },
  {  "BDifferenceFromPrevious",   BUILTIN_FUNC_PREFIX, "c", ComparePlane::Create_prev, (void *)PLANAR_B },
  //{  "SatDifferenceFromPrevious","c", CompareSat::Create_prev },
//{  "HueDifferenceFromPrevious","c", CompareHue::Create_prev },

  {  "YDifferenceToNext",   BUILTIN_FUNC_PREFIX, "c[offset]i", ComparePlane::Create_next, (void *)PLANAR_Y },
  {  "UDifferenceToNext",   BUILTIN_FUNC_PREFIX, "c[offset]i", ComparePlane::Create_next, (void *)PLANAR_U },
  {  "VDifferenceToNext",   BUILTIN_FUNC_PREFIX, "c[offset]i", ComparePlane::Create_next, (void *)PLANAR_V },
  {  "RGBDifferenceToNext", BUILTIN_FUNC_PREFIX, "c[offset]i", ComparePlane::Create_next, (void *)-1 },
  {  "RDifferenceToNext",   BUILTIN_FUNC_PREFIX, "c[offset]i", ComparePlane::Create_next, (void *)PLANAR_R },
  {  "GDifferenceToNext",   BUILTIN_FUNC_PREFIX, "c[offset]i", ComparePlane::Create_next, (void *)PLANAR_G },
  {  "BDifferenceToNext",   BUILTIN_FUNC_PREFIX, "c[offset]i", ComparePlane::Create_next, (void *)PLANAR_B },
  //{  "SatDifferenceFromNext","c[offset]i", CompareSat::Create_next },
//{  "HueDifferenceFromNext","c[offset]i", CompareHue::Create_next },
  {  "PlaneMinMaxStats", BUILTIN_FUNC_PREFIX, "c[threshold]f[offset]i[plane]i[setvar]b", MinMaxPlane::Create_minmax_stats, (void*)-1 },
  {  "YPlaneMax",    BUILTIN_FUNC_PREFIX, "c[threshold]f[offset]i", MinMaxPlane::Create_max, (void *)PLANAR_Y },
  {  "YPlaneMin",    BUILTIN_FUNC_PREFIX, "c[threshold]f[offset]i", MinMaxPlane::Create_min, (void *)PLANAR_Y },
  {  "YPlaneMedian", BUILTIN_FUNC_PREFIX, "c[offset]i", MinMaxPlane::Create_median, (void *)PLANAR_Y },
  {  "UPlaneMax",    BUILTIN_FUNC_PREFIX, "c[threshold]f[offset]i", MinMaxPlane::Create_max, (void *)PLANAR_U },
  {  "UPlaneMin",    BUILTIN_FUNC_PREFIX, "c[threshold]f[offset]i", MinMaxPlane::Create_min, (void *)PLANAR_U },
  {  "UPlaneMedian", BUILTIN_FUNC_PREFIX, "c[offset]i", MinMaxPlane::Create_median, (void *)PLANAR_U },
  {  "VPlaneMax",    BUILTIN_FUNC_PREFIX, "c[threshold]f[offset]i", MinMaxPlane::Create_max, (void *)PLANAR_V }, // AVS+! was before: missing offset parameter
  {  "VPlaneMin",    BUILTIN_FUNC_PREFIX, "c[threshold]f[offset]i", MinMaxPlane::Create_min, (void *)PLANAR_V }, // AVS+! was before: missing offset parameter
  {  "VPlaneMedian", BUILTIN_FUNC_PREFIX, "c[offset]i", MinMaxPlane::Create_median, (void *)PLANAR_V },
  {  "RPlaneMax",    BUILTIN_FUNC_PREFIX, "c[threshold]f[offset]i", MinMaxPlane::Create_max, (void *)PLANAR_R },
  {  "RPlaneMin",    BUILTIN_FUNC_PREFIX, "c[threshold]f[offset]i", MinMaxPlane::Create_min, (void *)PLANAR_R },
  {  "RPlaneMedian", BUILTIN_FUNC_PREFIX, "c[offset]i", MinMaxPlane::Create_median, (void *)PLANAR_R },
  {  "GPlaneMax",    BUILTIN_FUNC_PREFIX, "c[threshold]f[offset]i", MinMaxPlane::Create_max, (void *)PLANAR_G },
  {  "GPlaneMin",    BUILTIN_FUNC_PREFIX, "c[threshold]f[offset]i", MinMaxPlane::Create_min, (void *)PLANAR_G },
  {  "GPlaneMedian", BUILTIN_FUNC_PREFIX, "c[offset]i", MinMaxPlane::Create_median, (void *)PLANAR_G },
  {  "BPlaneMax",    BUILTIN_FUNC_PREFIX, "c[threshold]f[offset]i", MinMaxPlane::Create_max, (void *)PLANAR_B },
  {  "BPlaneMin",    BUILTIN_FUNC_PREFIX, "c[threshold]f[offset]i", MinMaxPlane::Create_min, (void *)PLANAR_B },
  {  "BPlaneMedian", BUILTIN_FUNC_PREFIX, "c[offset]i", MinMaxPlane::Create_median, (void *)PLANAR_B },
  {  "YPlaneMinMaxDifference", BUILTIN_FUNC_PREFIX, "c[threshold]f[offset]i", MinMaxPlane::Create_minmax, (void *)PLANAR_Y },
  {  "UPlaneMinMaxDifference", BUILTIN_FUNC_PREFIX, "c[threshold]f[offset]i", MinMaxPlane::Create_minmax, (void *)PLANAR_U }, // AVS+! was before: missing offset parameter
  {  "VPlaneMinMaxDifference", BUILTIN_FUNC_PREFIX, "c[threshold]f[offset]i", MinMaxPlane::Create_minmax, (void *)PLANAR_V }, // AVS+! was before: missing offset parameter
  {  "RPlaneMinMaxDifference", BUILTIN_FUNC_PREFIX, "c[threshold]f[offset]i", MinMaxPlane::Create_minmax, (void *)PLANAR_R },
  {  "GPlaneMinMaxDifference", BUILTIN_FUNC_PREFIX, "c[threshold]f[offset]i", MinMaxPlane::Create_minmax, (void *)PLANAR_G },
  {  "BPlaneMinMaxDifference", BUILTIN_FUNC_PREFIX, "c[threshold]f[offset]i", MinMaxPlane::Create_minmax, (void *)PLANAR_B },

//{  "SatMax","c[threshold]f[offset]i", MinMaxPlane::Create_maxsat },  ++accum[SatLookup[U,V]]
//{  "SatMin","c[threshold]f[offset]i", MinMaxPlane::Create_minsat },
//{  "SatMedian","c[offset]i", MinMaxPlane::Create_mediansat },
//{  "SatMinMaxDifference","c[threshold]f[offset]i", MinMaxPlane::Create_minmaxsat },

//{  "HueMax","c[threshold]f[offset]i", MinMaxPlane::Create_maxhue },  ++accum[HueLookup[U,V]]
//{  "HueMin","c[threshold]f[offset]i", MinMaxPlane::Create_minhue },
//{  "HueMedian","c[offset]i", MinMaxPlane::Create_medianhue },
//{  "HueMinMaxDifference","c[threshold]f[offset]i", MinMaxPlane::Create_minmaxhue },

  // frame property setters in conditional_reader
  { "propGetAny", BUILTIN_FUNC_PREFIX, "cs[index]i[offset]i", GetProperty::Create, (void*)0 },
  { "propGetInt", BUILTIN_FUNC_PREFIX, "cs[index]i[offset]i", GetProperty::Create, (void *)1 },
  { "propGetFloat", BUILTIN_FUNC_PREFIX, "cs[index]i[offset]i", GetProperty::Create, (void*)2 },
  { "propGetString", BUILTIN_FUNC_PREFIX, "cs[index]i[offset]i", GetProperty::Create, (void*)3 },
  { "propGetClip", BUILTIN_FUNC_PREFIX, "cs[index]i[offset]i", GetProperty::Create, (void*)4 },
  { "propGetDataSize", BUILTIN_FUNC_PREFIX, "cs[index]i[offset]i", GetPropertyDataSize::Create },
  { "propNumElements", BUILTIN_FUNC_PREFIX, "cs[offset]i", GetPropertyNumElements::Create},
  { "propNumKeys", BUILTIN_FUNC_PREFIX, "c[offset]i", GetPropertyNumKeys::Create},
  { "propGetKeyByIndex", BUILTIN_FUNC_PREFIX, "c[index]i[offset]i", GetPropertyKeyByIndex::Create},
  { "propGetType", BUILTIN_FUNC_PREFIX, "cs[offset]i", GetPropertyType::Create},
  { "propGetAsArray", BUILTIN_FUNC_PREFIX, "cs[offset]i", GetPropertyAsArray::Create},
  { "propGetAll", BUILTIN_FUNC_PREFIX, "c[offset]i", GetAllProperties::Create},

  { 0 }
};


AVSValue AveragePlane::Create(AVSValue args, void* user_data, IScriptEnvironment* env) {
  int plane = (int)reinterpret_cast<intptr_t>(user_data);
  return AvgPlane(args[0], user_data, plane, args[1].AsInt(0), env);
}

// Average plane
template<typename pixel_t>
static double get_sum_of_pixels_c(const BYTE* srcp8, size_t height, size_t width, size_t pitch) {
  typedef typename std::conditional < sizeof(pixel_t) == 4, double, int64_t>::type sum_t;
  sum_t accum = 0; // int32 holds sum of maximum 16 Mpixels for 8 bit, and 65536 pixels for uint16_t pixels
  const pixel_t *srcp = reinterpret_cast<const pixel_t *>(srcp8);
  pitch /= sizeof(pixel_t);
  for (size_t y = 0; y < height; y++) {
    for (size_t x = 0; x < width; x++) {
      accum += srcp[x];
    }
    srcp += pitch;
  }
  return (double)accum;
}

AVSValue AveragePlane::AvgPlane(AVSValue clip, void* , int plane, int offset, IScriptEnvironment* env)
{
  if (!clip.IsClip())
    env->ThrowError("Average Plane: No clip supplied!");

  PClip child = clip.AsClip();
  VideoInfo vi = child->GetVideoInfo();

  // input clip to always planar
  if (vi.IsRGB() && !vi.IsPlanar()) {
    AVSValue new_args[1] = { child };
    if (vi.IsRGB24() || vi.IsRGB48())
      child = env->Invoke("ConvertToPlanarRGB", AVSValue(new_args, 1)).AsClip();
    else // RGB32, RGB64
      child = env->Invoke("ConvertToPlanarRGBA", AVSValue(new_args, 1)).AsClip();
    vi = child->GetVideoInfo();
  }
  else if (vi.IsYUY2()) {
    AVSValue new_args[2] = { child, false };
    child = env->Invoke("ConvertToYUV422", AVSValue(new_args, 2)).AsClip();
    vi = child->GetVideoInfo();
  }

  if (!vi.IsPlanar())
    env->ThrowError("Average Plane: Only planar YUV or planar RGB images supported!");

  if (plane == PLANAR_A)
  {
    if (!vi.IsPlanarRGBA() && !vi.IsYUVA())
      env->ThrowError("Average Plane: clip has no Alpha plane!");
  }
  else if(vi.IsRGB())
  { 
    if (plane != PLANAR_R && plane != PLANAR_G && plane != PLANAR_B)
      env->ThrowError("Average Plane: not a valid plane for an RGB clip!");
  }
  else if (vi.IsY()){
    if (plane != PLANAR_Y)
      env->ThrowError("Average Plane: not a valid plane for an greyscale clip!");
  }
  else {
    if (plane != PLANAR_Y && plane != PLANAR_U && plane != PLANAR_V)
      env->ThrowError("Average Plane: not a valid plane for a YUV clip!");
  }

  AVSValue cn = env->GetVarDef("current_frame");
  if (!cn.IsInt())
    env->ThrowError("Average Plane: This filter can only be used within run-time filters");

  int n = cn.AsInt();
  n = min(max(n+offset,0), vi.num_frames-1);

  PVideoFrame src = child->GetFrame(n,env);

  int pixelsize = vi.ComponentSize();

  const BYTE* srcp = src->GetReadPtr(plane);
  int height = src->GetHeight(plane);
  int width = src->GetRowSize(plane) / pixelsize;
  int pitch = src->GetPitch(plane);

  if (width == 0 || height == 0)
    env->ThrowError("Average Plane: plane does not exist!");

  double sum = 0.0;


  int total_pixels = width*height;
  bool sum_in_32bits;
  if (pixelsize == 4)
    sum_in_32bits = false;
  else // worst case
    sum_in_32bits = ((int64_t)total_pixels * (pixelsize == 1 ? 255 : 65535)) <= std::numeric_limits<int>::max();

#ifdef INTEL_INTRINSICS
  if ((pixelsize==1) && sum_in_32bits && (env->GetCPUFlags() & CPUF_SSE2) && width >= 16) {
    sum = get_sum_of_pixels_sse2(srcp, height, width, pitch);
  } else
#ifdef X86_32
  if ((pixelsize==1) && sum_in_32bits && (env->GetCPUFlags() & CPUF_INTEGER_SSE) && width >= 8) {
    sum = get_sum_of_pixels_isse(srcp, height, width, pitch);
  } else
#endif
#endif
  {
    if(pixelsize==1)
      sum = get_sum_of_pixels_c<uint8_t>(srcp, height, width, pitch);
    else if(pixelsize==2)
      sum = get_sum_of_pixels_c<uint16_t>(srcp, height, width, pitch);
    else // pixelsize==4
      sum = get_sum_of_pixels_c<float>(srcp, height, width, pitch);
  }

  float f = (float)(sum / (height * width));

  return (AVSValue)f;
}

AVSValue ComparePlane::Create(AVSValue args, void* user_data, IScriptEnvironment* env) {
  int plane = (int)reinterpret_cast<intptr_t>(user_data);
  return CmpPlane(args[0],args[1], user_data, plane, env);
}

AVSValue ComparePlane::Create_prev(AVSValue args, void* user_data, IScriptEnvironment* env) {
  int plane = (int)reinterpret_cast<intptr_t>(user_data);
  return CmpPlaneSame(args[0], user_data, -1, plane, env);
}

AVSValue ComparePlane::Create_next(AVSValue args, void* user_data, IScriptEnvironment* env) {
  int plane = (int)reinterpret_cast<intptr_t>(user_data);
  return CmpPlaneSame(args[0], user_data, args[1].AsInt(1), plane, env);
}


template<typename pixel_t>
static double get_sad_c(const BYTE* c_plane8, const BYTE* t_plane8, size_t height, size_t width, size_t c_pitch, size_t t_pitch) {
  const pixel_t *c_plane = reinterpret_cast<const pixel_t *>(c_plane8);
  const pixel_t *t_plane = reinterpret_cast<const pixel_t *>(t_plane8);
  c_pitch /= sizeof(pixel_t);
  t_pitch /= sizeof(pixel_t);
  typedef typename std::conditional < sizeof(pixel_t) == 4, double, int64_t>::type sum_t;
  sum_t accum = 0; // int32 holds sum of maximum 16 Mpixels for 8 bit, and 65536 pixels for uint16_t pixels

  for (size_t y = 0; y < height; y++) {
    for (size_t x = 0; x < width; x++) {
      accum += std::abs(t_plane[x] - c_plane[x]);
    }
    c_plane += c_pitch;
    t_plane += t_pitch;
  }
  return (double)accum;

}

template<typename pixel_t>
static double get_sad_rgb_c(const BYTE* c_plane8, const BYTE* t_plane8, size_t height, size_t width, size_t c_pitch, size_t t_pitch) {
  const pixel_t *c_plane = reinterpret_cast<const pixel_t *>(c_plane8);
  const pixel_t *t_plane = reinterpret_cast<const pixel_t *>(t_plane8);
  c_pitch /= sizeof(pixel_t);
  t_pitch /= sizeof(pixel_t);
  int64_t accum = 0; // packed rgb: integer type only
  for (size_t y = 0; y < height; y++) {
    for (size_t x = 0; x < width; x+=4) {
      accum += std::abs(t_plane[x] - c_plane[x]);
      accum += std::abs(t_plane[x+1] - c_plane[x+1]);
      accum += std::abs(t_plane[x+2] - c_plane[x+2]);
    }
    c_plane += c_pitch;
    t_plane += t_pitch;
  }
  return (double)accum;

}

AVSValue ComparePlane::CmpPlane(AVSValue clip, AVSValue clip2, void* , int plane, IScriptEnvironment* env)
{
  if (!clip.IsClip())
    env->ThrowError("Plane Difference: No clip supplied!");
  if (!clip2.IsClip())
    env->ThrowError("Plane Difference: Second parameter is not a clip!");

  PClip child = clip.AsClip();
  VideoInfo vi = child->GetVideoInfo();
  PClip child2 = clip2.AsClip();
  VideoInfo vi2 = child2->GetVideoInfo();
  if (plane !=-1 ) {
    if (!vi.IsPlanar() || !vi2.IsPlanar())
      env->ThrowError("Plane Difference: Only planar YUV or planar RGB images supported!");
  } else {
    if(vi.IsPlanarRGB() || vi.IsPlanarRGBA())
      env->ThrowError("RGB Difference: Planar RGB is not supported here (clip 1)");
    if(vi2.IsPlanarRGB() || vi2.IsPlanarRGBA())
      env->ThrowError("RGB Difference: Planar RGB is not supported here (clip 2)");
    if (!vi.IsRGB())
      env->ThrowError("RGB Difference: RGB difference can only be tested on RGB images! (clip 1)");
    if (!vi2.IsRGB())
      env->ThrowError("RGB Difference: RGB difference can only be tested on RGB images! (clip 2)");
    plane = 0;
  }

  AVSValue cn = env->GetVarDef("current_frame");
  if (!cn.IsInt())
    env->ThrowError("Plane Difference: This filter can only be used within run-time filters");

  int n = cn.AsInt();
  n = clamp(n,0,vi.num_frames-1);

  PVideoFrame src = child->GetFrame(n,env);
  PVideoFrame src2 = child2->GetFrame(n,env);

  int pixelsize = vi.ComponentSize();

  const BYTE* srcp = src->GetReadPtr(plane);
  const BYTE* srcp2 = src2->GetReadPtr(plane);
  const int height = src->GetHeight(plane);
  const int rowsize = src->GetRowSize(plane);
  const int width = rowsize / pixelsize;
  const int pitch = src->GetPitch(plane);
  const int height2 = src2->GetHeight(plane);
  const int rowsize2 = src2->GetRowSize(plane);
  const int width2 = rowsize2 / pixelsize;
  const int pitch2 = src2->GetPitch(plane);

  if(vi.ComponentSize() != vi2.ComponentSize())
    env->ThrowError("Plane Difference: Bit-depth are not the same!");

  if (width == 0 || height == 0)
    env->ThrowError("Plane Difference: plane does not exist!");

  if (height != height2 || width != width2)
    env->ThrowError("Plane Difference: Images are not the same size!");

#ifdef X86_32
  int bits_per_pixel = vi.BitsPerComponent();
  int total_pixels = width*height;
  bool sum_in_32bits;
  if (pixelsize == 4)
    sum_in_32bits = false;
  else // worst case check
    sum_in_32bits = ((int64_t)total_pixels * ((1 << bits_per_pixel) - 1)) <= std::numeric_limits<int>::max();
#endif

  double sad = 0.0;

  // for c: width, for sse: rowsize
  if (vi.IsRGB32() || vi.IsRGB64()) {
#ifdef INTEL_INTRINSICS
    if ((pixelsize == 2) && (env->GetCPUFlags() & CPUF_SSE2) && rowsize >= 16) {
      // int64 internally, no sum_in_32bits
      sad = (double)calculate_sad_8_or_16_sse2<uint16_t,true>(srcp, srcp2, pitch, pitch2, width*pixelsize, height); // in focus. 21.68/21.39
    } else if ((pixelsize == 1) && (env->GetCPUFlags() & CPUF_SSE2) && rowsize >= 16) {
      sad = (double)calculate_sad_8_or_16_sse2<uint8_t,true>(srcp, srcp2, pitch, pitch2, rowsize, height); // in focus, no overflow
    } else
#ifdef X86_32
      if ((pixelsize==1) && sum_in_32bits && (env->GetCPUFlags() & CPUF_INTEGER_SSE) && width >= 8) {
        sad = get_sad_rgb_isse(srcp, srcp2, height, rowsize, pitch, pitch2);
      } else
#endif
#endif
      {
        if (pixelsize == 1)
          sad = (double)get_sad_rgb_c<uint8_t>(srcp, srcp2, height, width, pitch, pitch2);
        else // pixelsize==2
          sad = (double)get_sad_rgb_c<uint16_t>(srcp, srcp2, height, width, pitch, pitch2);
      }
  } else {
#ifdef INTEL_INTRINSICS
    if ((pixelsize==2) && (env->GetCPUFlags() & CPUF_SSE2) && rowsize >= 16) {
      sad = (double)calculate_sad_8_or_16_sse2<uint16_t,false>(srcp, srcp2, pitch, pitch2, rowsize, height); // in focus, no overflow
    } else
      if ((pixelsize==1) && (env->GetCPUFlags() & CPUF_SSE2) && rowsize >= 16) {
      sad = (double)calculate_sad_8_or_16_sse2<uint8_t,false>(srcp, srcp2, pitch, pitch2, rowsize, height); // in focus, no overflow
    } else
#ifdef X86_32
      if ((pixelsize==1) && sum_in_32bits && (env->GetCPUFlags() & CPUF_INTEGER_SSE) && width >= 8) {
        sad = get_sad_isse(srcp, srcp2, height, rowsize, pitch, pitch2);
      } else
#endif
#endif
      {
        if(pixelsize==1)
          sad = get_sad_c<uint8_t>(srcp, srcp2, height, width, pitch, pitch2);
        else if(pixelsize==2)
          sad = get_sad_c<uint16_t>(srcp, srcp2, height, width, pitch, pitch2);
        else // pixelsize==4
          sad = get_sad_c<float>(srcp, srcp2, height, width, pitch, pitch2);
      }
  }

  float f;

  if (vi.IsRGB32() || vi.IsRGB64())
    f = (float)((sad * 4) / (height * width * 3)); // why * 4/3? alpha plane was masked out, anyway
  else
    f = (float)(sad / (height * width));

  return (AVSValue)f;
}


AVSValue ComparePlane::CmpPlaneSame(AVSValue clip, void* , int offset, int plane, IScriptEnvironment* env)
{
  if (!clip.IsClip())
    env->ThrowError("Plane Difference: No clip supplied!");

  PClip child = clip.AsClip();
  VideoInfo vi = child->GetVideoInfo();
  if (plane ==-1 ) {
    if (!vi.IsRGB() || vi.IsPlanarRGB() || vi.IsPlanarRGBA())
      env->ThrowError("RGB Difference: RGB difference can only be calculated on packed RGB images");
    plane = 0;
  } else {
    if (!vi.IsPlanar())
      env->ThrowError("Plane Difference: Only planar YUV or planar RGB images images supported!");
  }

  AVSValue cn = env->GetVarDef("current_frame");
  if (!cn.IsInt())
    env->ThrowError("Plane Difference: This filter can only be used within run-time filters");

  int n = cn.AsInt();
  n = clamp(n,0,vi.num_frames-1);
  int n2 = clamp(n+offset,0,vi.num_frames-1);

  PVideoFrame src = child->GetFrame(n,env);
  PVideoFrame src2 = child->GetFrame(n2,env);

  int pixelsize = vi.ComponentSize();

  const BYTE* srcp = src->GetReadPtr(plane);
  const BYTE* srcp2 = src2->GetReadPtr(plane);
  int height = src->GetHeight(plane);
  int rowsize = src->GetRowSize(plane);
  int width = rowsize / pixelsize;
  int pitch = src->GetPitch(plane);
  int pitch2 = src2->GetPitch(plane);

  if (width == 0 || height == 0)
    env->ThrowError("Plane Difference: No chroma planes in greyscale clip!");

#ifdef X86_32
  int bits_per_pixel = vi.BitsPerComponent();
  int total_pixels = width * height;
  bool sum_in_32bits;
  if (pixelsize == 4)
    sum_in_32bits = false;
  else // worst case check
    sum_in_32bits = ((int64_t)total_pixels * ((1 << bits_per_pixel) - 1)) <= std::numeric_limits<int>::max();
#endif

  double sad = 0;
  // for c: width, for sse: rowsize
  if (vi.IsRGB32() || vi.IsRGB64()) {
#ifdef INTEL_INTRINSICS
    if ((pixelsize == 2) && (env->GetCPUFlags() & CPUF_SSE2) && rowsize >= 16) {
      // int64 internally, no sum_in_32bits
      sad = (double)calculate_sad_8_or_16_sse2<uint16_t,true>(srcp, srcp2, pitch, pitch2, rowsize, height); // in focus. 21.68/21.39
    } else if ((pixelsize == 1) && (env->GetCPUFlags() & CPUF_SSE2) && rowsize >= 16) {
      sad = (double)calculate_sad_8_or_16_sse2<uint8_t,true>(srcp, srcp2, pitch, pitch2, rowsize, height); // in focus, no overflow
    } else
#ifdef X86_32
      if ((pixelsize==1) && sum_in_32bits && (env->GetCPUFlags() & CPUF_INTEGER_SSE) && width >= 8) {
        sad = get_sad_rgb_isse(srcp, srcp2, height, rowsize, pitch, pitch2);
      } else
#endif
#endif
      {
        if(pixelsize==1)
          sad = get_sad_rgb_c<uint8_t>(srcp, srcp2, height, width, pitch, pitch2);
        else
          sad = get_sad_rgb_c<uint16_t>(srcp, srcp2, height, width, pitch, pitch2);
      }
  } else {
#ifdef INTEL_INTRINSICS
    if ((pixelsize==2) && (env->GetCPUFlags() & CPUF_SSE2) && rowsize >= 16) {
      sad = (double)calculate_sad_8_or_16_sse2<uint16_t,false>(srcp, srcp2, pitch, pitch2, rowsize, height); // in focus, no overflow
    } else if ((pixelsize==1) && (env->GetCPUFlags() & CPUF_SSE2) && rowsize >= 16) {
      sad = (double)calculate_sad_8_or_16_sse2<uint8_t,false>(srcp, srcp2, pitch, pitch2, rowsize, height); // in focus, no overflow
    } else
#ifdef X86_32
      if ((pixelsize==1) && sum_in_32bits && (env->GetCPUFlags() & CPUF_INTEGER_SSE) && width >= 8) {
        sad = get_sad_isse(srcp, srcp2, height, width, pitch, pitch2);
      } else
#endif
#endif
      {
        if(pixelsize==1)
          sad = get_sad_c<uint8_t>(srcp, srcp2, height, width, pitch, pitch2);
        else if (pixelsize==2)
          sad = get_sad_c<uint16_t>(srcp, srcp2, height, width, pitch, pitch2);
        else // pixelsize==4
          sad = get_sad_c<float>(srcp, srcp2, height, width, pitch, pitch2);
      }
  }

  float f;

  if (vi.IsRGB32() || vi.IsRGB64())
    f = (float)((sad * 4) / (height * width * 3));
  else
    f = (float)(sad / (height * width));

  return (AVSValue)f;
}

AVSValue MinMaxPlane::Create_minmax_stats(AVSValue args, void* user_data, IScriptEnvironment* env) {
  const int plane = args[3].AsInt(0); // 0:Y or R  1:U or G  2: V or B  3:A
  if (plane < 0 || plane>3)
    env->ThrowError("MinMax Stats: plane index must be between 0-3!");

  if (!args[0].IsClip())
    env->ThrowError("MinMax Stats: No clip supplied!");

  VideoInfo vi = args[0].AsClip()->GetVideoInfo();
  const int planes_y[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
  const int planes_r[4] = { PLANAR_R, PLANAR_G, PLANAR_B, PLANAR_A }; // in 'logical' RGB order
  const int* planes = (vi.IsYUV() || vi.IsYUVA()) ? planes_y : planes_r;

  const int current_plane = planes[plane];

  const int setvar = args[4].AsBool(false);

  return MinMax(args[0], user_data, args[1].AsDblDef(0.0), args[2].AsInt(0), current_plane, MinMaxPlane::STATS, setvar, env);
}


AVSValue MinMaxPlane::Create_max(AVSValue args, void* user_data, IScriptEnvironment* env) {
  int plane = (int)reinterpret_cast<intptr_t>(user_data);
  return MinMax(args[0], user_data, args[1].AsDblDef(0.0), args[2].AsInt(0), plane, MinMaxPlane::MAX, false, env);
}

AVSValue MinMaxPlane::Create_min(AVSValue args, void* user_data, IScriptEnvironment* env) {
  int plane = (int)reinterpret_cast<intptr_t>(user_data);
  return MinMax(args[0], user_data, args[1].AsDblDef(0.0), args[2].AsInt(0), plane, MinMaxPlane::MIN, false, env);
}

AVSValue MinMaxPlane::Create_median(AVSValue args, void* user_data, IScriptEnvironment* env) {
  int plane = (int)reinterpret_cast<intptr_t>(user_data);
  return MinMax(args[0], user_data, 50.0, args[1].AsInt(0), plane, MinMaxPlane::MIN, false, env);
}

AVSValue MinMaxPlane::Create_minmax(AVSValue args, void* user_data, IScriptEnvironment* env) {
  int plane = (int)reinterpret_cast<intptr_t>(user_data);
  return MinMax(args[0], user_data, args[1].AsDblDef(0.0), args[2].AsInt(0), plane, MinMaxPlane::MINMAX_DIFFERENCE, false, env);
}

void get_minmax_float_c(const BYTE* srcp, int pitch, int w, int h, float& min, float& max)
{
  min = *reinterpret_cast<const float*>(srcp);
  max = min;

  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      const float pix = reinterpret_cast<const float*>(srcp)[x];
      if (pix < min) min = pix;
      if (pix > max) max = pix;
    }
    srcp += pitch;
  }
}

template<typename pixel_t>
void get_minmax_int_c(const BYTE* srcp, int pitch, int w, int h, int& min, int& max)
{
  min = *reinterpret_cast<const pixel_t*>(srcp);
  max = min;

  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      const int pix = reinterpret_cast<const pixel_t*>(srcp)[x];
      if (pix < min) min = pix;
      if (pix > max) max = pix;
    }
    srcp += pitch;
  }
}

AVSValue MinMaxPlane::MinMax(AVSValue clip, void* , double threshold, int offset, int plane, int mode, bool setvar, IScriptEnvironment* env) {

  if (!clip.IsClip())
    env->ThrowError("MinMax: No clip supplied!");

  PClip child = clip.AsClip();
  VideoInfo vi = child->GetVideoInfo();

  if (vi.IsRGB()) {
    if (plane == PLANAR_Y || plane == PLANAR_U || plane == PLANAR_V)
      env->ThrowError("MinMax: no such plane in RGB format");
  }
  else if (vi.IsY()) {
    if (plane != PLANAR_Y)
      env->ThrowError("MinMax: no such plane in Y (greyscale) format");
  }
  else {
    if (plane == PLANAR_R || plane == PLANAR_G || plane == PLANAR_B)
      env->ThrowError("MinMax: no such plane in YUV format");
  }
  if (vi.NumComponents() < 4 && plane == PLANAR_A)
    env->ThrowError("MinMax: no Alpha plane in this format");

  // input clip to always planar
  if (vi.IsRGB() && !vi.IsPlanar()) {
    AVSValue new_args[1] = { child };
    if (vi.IsRGB24() || vi.IsRGB48())
      child = env->Invoke("ConvertToPlanarRGB", AVSValue(new_args, 1)).AsClip();
    else // RGB32, RGB64
      child = env->Invoke("ConvertToPlanarRGBA", AVSValue(new_args, 1)).AsClip();
    vi = child->GetVideoInfo();
  }
  else if (vi.IsYUY2()) {
    AVSValue new_args[2] = { child, false };
    child = env->Invoke("ConvertToYUV422", AVSValue(new_args, 2)).AsClip();
    vi = child->GetVideoInfo();
  }

  if (!vi.IsPlanar())
    env->ThrowError("MinMax: Image must be planar");

  // Get current frame number
  AVSValue cn = env->GetVarDef("current_frame");
  if (!cn.IsInt())
    env->ThrowError("MinMax: This filter can only be used within run-time filters");

  int n = cn.AsInt();
  n = min(max(n + offset, 0), vi.num_frames - 1);

  // Prepare the source
  PVideoFrame src = child->GetFrame(n, env);

  const BYTE* srcp = src->GetReadPtr(plane);
  int pitch = src->GetPitch(plane);
  int pixelsize = vi.ComponentSize();
  int w = src->GetRowSize(plane) / pixelsize; // good for packed rgb as well
  int h = src->GetHeight(plane);

  if (w == 0 || h == 0)
    env->ThrowError("MinMax: plane does not exist!");

  float stats_min;
  float stats_max;
  float stats_median;
  float stats_thresholded_min;
  float stats_thresholded_max;

  if (threshold == 0 || mode == MinMaxPlane::STATS) {
    // special case, no histogram needed

    if (pixelsize == 4) // 32 bit float
    {
      get_minmax_float_c(srcp, pitch, w, h, stats_min, stats_max);
      
      if (mode == MinMaxPlane::MIN) return stats_min;
      else if (mode == MinMaxPlane::MAX) return stats_max;
      else if (mode == MinMaxPlane::MINMAX_DIFFERENCE) return stats_max - stats_min;
      // STATS: go on
    }
    else
    {
      int min, max;
      // sse4.1 is not any faster
      if(pixelsize == 1)
        get_minmax_int_c<uint8_t>(srcp, pitch, w, h, min, max);
      else
        get_minmax_int_c<uint16_t>(srcp, pitch, w, h, min, max);

      if (mode == MinMaxPlane::MIN) return min;
      else if (mode == MinMaxPlane::MAX) return max;
      else if (mode == MinMaxPlane::MINMAX_DIFFERENCE) return max - min;

      // STATS: go on
      stats_min = (float)min;
      stats_max = (float)max;
    }
  }

  const int bits_per_pixel = vi.BitsPerComponent();
  const int max_pixel_value = (1 << bits_per_pixel) - 1;

  const int buffersize = pixelsize == 4 ? 65536 : (1 << bits_per_pixel); // 65536 for float, too, reason for 10-14 bits: avoid overflow
  uint32_t* accum_buf = new uint32_t[buffersize];

  // Reset accumulators
  std::fill_n(accum_buf, buffersize, 0);

  // Count each component
  if (pixelsize == 1) {
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        accum_buf[srcp[x]]++; // safe
      }
      srcp += pitch;
    }
  }
  else if (pixelsize == 2) {
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        accum_buf[min((int)(reinterpret_cast<const uint16_t *>(srcp)[x]), max_pixel_value)]++;
      }
      srcp += pitch;
    }
  }
  else { //pixelsize==4 float
 // for float results are always checked with 16 bit precision only
 // or else we cannot populate non-digital steps with this standard method
    // See similar in colors, ColorYUV analyze
    const bool chroma = (plane == PLANAR_U) || (plane == PLANAR_V);
    if (chroma) {
      const float shift = 32768.0f;
      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          // -0.5..0.5 to 0..65535
          const float pixel = reinterpret_cast<const float *>(srcp)[x];
          accum_buf[clamp((int)(65535.0f*pixel + shift + 0.5f), 0, 65535)]++;
        }
        srcp += pitch;
      }
    }
    else {
      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          const float pixel = reinterpret_cast<const float *>(srcp)[x];
          accum_buf[clamp((int)(65535.0f * pixel + 0.5f), 0, 65535)]++;
        }
        srcp += pitch;
      }
    }
  }

  int pixels = w*h;
  threshold /=100.0;  // Thresh now 0-1
  threshold = clamp(threshold, 0.0, 1.0);

  unsigned int tpixels = (unsigned int)(pixels*threshold);

  int retval;

    // Find the value we need.
  if (mode == MinMaxPlane::MIN) {
    unsigned int counted=0;
    retval = buffersize - 1;
    for (int i = 0; i< buffersize;i++) {
      counted += accum_buf[i];
      if (counted>tpixels) {
        retval = i;
        break;
      }
    }
  }
  
  if (mode == MinMaxPlane::MAX) {
    unsigned int counted=0;
    retval = 0;
    for (int i = buffersize-1; i>=0;i--) {
      counted += accum_buf[i];
      if (counted>tpixels) {
        retval = i;
        break;
      }
    }
  }
  
  if (mode == MinMaxPlane::MINMAX_DIFFERENCE || mode == MinMaxPlane::STATS) {
    unsigned int counted=0;
    int i, t_min = 0;
    // Find min
    for (i = 0; i < buffersize;i++) {
      counted += accum_buf[i];
      if (counted>tpixels) {
        t_min=i;
        break;
      }
    }

    // Find max
    counted=0;
    int t_max = buffersize-1;
    for (i = buffersize-1; i>=0;i--) {
      counted += accum_buf[i];
      if (counted>tpixels) {
        t_max=i;
        break;
      }
    }

    retval = t_max - t_min; // results <0 will be returned if threshold > 50
    stats_thresholded_min = (float)t_min;
    stats_thresholded_max = (float)t_max;
  }
  
  if (mode == MinMaxPlane::STATS) {
    // for STATS we already have min and max, thresholded min and max
    // let's gather median, which is equal to 50% MIN
    unsigned int tpixels = (unsigned int)(pixels * 0.5f);
    unsigned int counted = 0;
    retval = buffersize - 1;
    for (int i = 0; i < buffersize; i++) {
      counted += accum_buf[i];
      if (counted > tpixels) {
        retval = i;
        break;
      }
    }
    stats_median = (float)retval;
  }
  else {
    retval = -1;
  }

  delete[] accum_buf;
  //_RPT2(0, "End of MinMax cn=%d n=%d\r", cn.AsInt(), n);

  if (mode == MinMaxPlane::STATS) {
  if (pixelsize == 4) {
    const bool chroma = (plane == PLANAR_U) || (plane == PLANAR_V);
      const float shift = chroma ? 32768.0f : 0;
      stats_thresholded_min = (float)((double)(stats_thresholded_min - shift) / (buffersize - 1)); // convert back to float, /65535
      stats_thresholded_max = (float)((double)(stats_thresholded_max - shift) / (buffersize - 1)); // convert back to float, /65535
      stats_median = (float)((double)(stats_median - shift) / (buffersize - 1)); // convert back to float, /65535
    }

    AVSValue result[] = { stats_min, stats_max, stats_thresholded_min, stats_thresholded_max, stats_median };
    AVSValue ret = AVSValue(result, 5);
    if (setvar) {
      env->SetGlobalVar("PlaneStats_min", stats_min);
      env->SetGlobalVar("PlaneStats_max", stats_max);
      env->SetGlobalVar("PlaneStats_thmin", stats_thresholded_min);
      env->SetGlobalVar("PlaneStats_thmax", stats_thresholded_max);
      env->SetGlobalVar("PlaneStats_median", stats_median);
    }

    return ret;
  }

  if (pixelsize == 4) {
    const bool chroma = (plane == PLANAR_U) || (plane == PLANAR_V);
    if (chroma && (mode == MIN || mode == MAX)) {
      const float shift = 32768.0f;
      return AVSValue((double)(retval - shift) / (buffersize - 1)); // convert back to float, /65535
    }
    else {
      return AVSValue((double)retval / (buffersize - 1)); // convert back to float, /65535
    }
  }
  else
    return AVSValue(retval);
}


AVSValue GetProperty::Create(AVSValue args, void* user_data, IScriptEnvironment* env) {
  AVSValue clip = args[0];
  if (!clip.IsClip())
    env->ThrowError("propGetxxxxx: No clip supplied!");

  PClip child = clip.AsClip();
  VideoInfo vi = child->GetVideoInfo();

  AVSValue cn = env->GetVarDef("current_frame");
  const bool calledFromRunTime = cn.IsInt();

  int propType = (int)(intptr_t)user_data;
  // vUnset, vInt, vFloat, vData, vNode/Clip, vFrame/*, vMethod*/ }
  // 0: auto
  // 1: integer
  // 2: float
  // 3: char (null terminated data)
  // 4: Clip

  const char* propName = args[1].AsString();
  const int index = args[2].AsInt(0);
  const int offset = args[3].AsInt(0);

  int n = calledFromRunTime ? cn.AsInt() : 0;
  n = min(max(n + offset, 0), vi.num_frames - 1);

  PVideoFrame src = child->GetFrame(n, env);

  const AVSMap* avsmap = env->getFramePropsRO(src);

  int error = 0;

  // check auto
  if (propType == 0) {
    char res = env->propGetType(avsmap, propName);
    // 'u'nset, 'i'nteger, 'f'loat, 's'string, 'c'lip, 'v'ideoframe, 'm'ethod };
    switch (res) {
    case 'u': return AVSValue(); // unSet = AVS undefined
    case 'v': env->ThrowError("Error getting frame property \"%s\": video frames not supported as function return values", propName);
      break;
    case 'i': propType = 1; break;
    case 'f': propType = 2; break;
    case 's': propType = 3; break;
    case 'c': propType = 4; break;
    default:
      env->ThrowError("Error getting frame property \"%s\": type '%c' not supported", propName, res);
    }
  }

  if (propType == 1) {
    int64_t result = env->propGetInt(avsmap, propName, index, &error);
    if(!error)
      return AVSValue((int)result);
  }
  else if (propType == 2) {
    double result = env->propGetFloat(avsmap, propName, index, &error);
    if (!error)
      return AVSValue(result);
  }
  else if (propType == 3) {
    const char *result = env->propGetData(avsmap, propName, index, &error);
    if (!error) {
      result = env->SaveString(result); // property had its own storage
      return AVSValue(result);
    }
  }
  else if (propType == 4) {
    PClip result = env->propGetClip(avsmap, propName, index, &error);
    if (!error)
      return AVSValue(result);
  }
  else {
    error = AVSGetPropErrors::GETPROPERROR_TYPE;
  }

  const char* error_msg = nullptr;

  // really, errors are bits
  if (error & AVSGetPropErrors::GETPROPERROR_UNSET)
    error_msg = "property is not set";
  else if (error & AVSGetPropErrors::GETPROPERROR_TYPE)
    error_msg = "wrong type";
  else if (error & AVSGetPropErrors::GETPROPERROR_INDEX)
    error_msg = "index error"; // arrays

  if (error)
    env->ThrowError("Error getting frame property \"%s\": %s ", propName, error_msg);

  return AVSValue();
}

AVSValue GetPropertyAsArray::Create(AVSValue args, void* , IScriptEnvironment* env)
{
  AVSValue clip = args[0];
  if (!clip.IsClip())
    env->ThrowError("propGetAsArray: No clip supplied!");

  PClip child = clip.AsClip();
  VideoInfo vi = child->GetVideoInfo();

  AVSValue cn = env->GetVarDef("current_frame");
  const bool calledFromRunTime = cn.IsInt();

  const char* propName = args[1].AsString();
  const int offset = args[2].AsInt(0);

  int n = calledFromRunTime ? cn.AsInt() : 0;
  n = min(max(n + offset, 0), vi.num_frames - 1);

  PVideoFrame src = child->GetFrame(n, env);

  const AVSMap* avsmap = env->getFramePropsRO(src);

  int error = 0;

  char propType = env->propGetType(avsmap, propName);
  if (propType == 'u') {
    // special: zero array
    return AVSValue(nullptr, 0);
  }
  // check auto
  int size = env->propNumElements(avsmap, propName);

  std::vector<AVSValue> result(size);

  // propGetIntArray or propGetFloatArray is available
  // note: AVSValue is int and float, prop arrays are int64_t and double
  if (propType == 'i') {
    const int64_t* arr = env->propGetIntArray(avsmap, propName, &error);
    for (int i = 0; i < size; ++i)
      result[i] = (int)arr[i];
  }
  else if (propType == 'f') {
    const double* arr = env->propGetFloatArray(avsmap, propName, &error);
    for (int i = 0; i < size; ++i)
      result[i] = (float)arr[i];
  }
  else
  {
    // generic
    for (int i = 0; i < size; ++i) {
      AVSValue elem;
      switch (propType) {
      case 'i': elem = AVSValue((int)env->propGetInt(avsmap, propName, i, &error)); break; // though handled earlier
      case 'f': elem = AVSValue((float)env->propGetFloat(avsmap, propName, i, &error)); break; // though handled earlier
      case 's': {
        const char* s = env->propGetData(avsmap, propName, i, &error);
        if (!error)
          elem = AVSValue(env->SaveString(s));
      }
        break;
      case 'v': elem = AVSValue(env->propGetFrame(avsmap, propName, i, &error)); break;
      case 'c': elem = AVSValue(env->propGetClip(avsmap, propName, i, &error)); break;
      default:
        elem = AVSValue();
      }
      if (error)
        break;
      result[i] = elem;
    }
  }

  const char* error_msg = nullptr;

  // really, errors are bits
  if (error & AVSGetPropErrors::GETPROPERROR_UNSET)
    error_msg = "property is not set";
  else if (error & AVSGetPropErrors::GETPROPERROR_TYPE)
    error_msg = "wrong type";
  else if (error & AVSGetPropErrors::GETPROPERROR_INDEX)
    error_msg = "index error"; // arrays

  if (error)
    env->ThrowError("propGetAsArray: Error getting frame property \"%s\": %s ", propName, error_msg);

  return AVSValue(result.data(), size); // array deep copy
}

// propGetAll
// fills an AVSValue array with key-value pairs
// array size will be numProps
// each array element is a two dimensional array [key, value]
// value can be an array as well, depending on the property
// only int, float and string data extracted
AVSValue GetAllProperties::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  AVSValue clip = args[0];
  if (!clip.IsClip())
    env->ThrowError("propGetAll: No clip supplied!");

  PClip child = clip.AsClip();
  VideoInfo vi = child->GetVideoInfo();

  AVSValue cn = env->GetVarDef("current_frame");
  const bool calledFromRunTime = cn.IsInt();

  const int offset = args[1].AsInt(0);

  int n = calledFromRunTime ? cn.AsInt() : 0;
  n = min(max(n + offset, 0), vi.num_frames - 1);

  PVideoFrame src = child->GetFrame(n, env);

  const AVSMap* avsmap = env->getFramePropsRO(src);

  const int propNum = env->propNumKeys(avsmap);

  if (0 == propNum)
    return AVSValue(nullptr, 0); // zero sized array

  std::vector<AVSValue> result(propNum);

  for (int index = 0; index < propNum; index++) {
    std::vector<AVSValue> pair(2);
    const char* propName = env->propGetKey(avsmap, index);
    pair[0] = env->SaveString(propName);
    const char propType = env->propGetType(avsmap, propName);
    const int propNumElements = env->propNumElements(avsmap, propName);

    int error;

    AVSValue elem;
    if (propType == 'u') {
      // unSet: undefined value
    }
    else if (propType == 'i') {
      if(propNumElements == 1)
        elem = AVSValue((int)env->propGetInt(avsmap, propName, 0, &error));
      else {
        std::vector<AVSValue> avsarr(propNumElements);
        const int64_t* arr = env->propGetIntArray(avsmap, propName, &error);
        for (int i = 0; i < propNumElements; ++i)
          avsarr[i] = (int)arr[i];
        elem = AVSValue(avsarr.data(), propNumElements); // array deep copy
      }
    }
    else if (propType == 'f') {
      if(propNumElements == 1)
        elem = AVSValue((float)env->propGetFloat(avsmap, propName, 0, &error));
      else {
        std::vector<AVSValue> avsarr(propNumElements);
        const double* arr = env->propGetFloatArray(avsmap, propName, &error);
        for (int i = 0; i < propNumElements; ++i)
          avsarr[i] = (float)arr[i];
        elem = AVSValue(avsarr.data(), propNumElements); // array deep copy
      }
    }
    else if (propType == 's') {
      // no string arrays
      const char* s = env->propGetData(avsmap, propName, 0, &error);
      if (!error)
        elem = AVSValue(env->SaveString(s));
    }
    else if (propType == 'c') {
      if (propNumElements == 1)
        elem = AVSValue(env->propGetClip(avsmap, propName, 0, &error));
      else {
        std::vector<AVSValue> avsarr(propNumElements);
        for (int i = 0; i < propNumElements; ++i)
          avsarr[i] = AVSValue(env->propGetClip(avsmap, propName, i, &error));
        elem = AVSValue(avsarr.data(), propNumElements); // array deep copy
      }
    }
    else {
      // 'v': ignore no such AVSValue in Avisynth 
    }

    pair[1] = elem;
    result[index] = AVSValue(pair.data(), 2); // arrays in arrays, automatic deep copy
  }

  return AVSValue(result.data(), propNum); // array deep copy
}

// e.g. string length
AVSValue GetPropertyDataSize::Create(AVSValue args, void* , IScriptEnvironment* env) {
  AVSValue clip = args[0];
  if (!clip.IsClip())
    env->ThrowError("propGetDataSize: No clip supplied!");

  PClip child = clip.AsClip();
  VideoInfo vi = child->GetVideoInfo();

  AVSValue cn = env->GetVarDef("current_frame");
  const bool calledFromRunTime = cn.IsInt();

  const char* propName = args[1].AsString();
  const int index = args[2].AsInt(0);
  const int offset = args[3].AsInt(0);

  int n = calledFromRunTime ? cn.AsInt() : 0;
  n = min(max(n + offset, 0), vi.num_frames - 1);

  PVideoFrame src = child->GetFrame(n, env);

  const AVSMap* avsmap = env->getFramePropsRO(src);

  int error = 0;

  int size = env->propGetDataSize(avsmap, propName, index, &error);
  if (!error)
    return AVSValue(size);

  const char* error_msg = nullptr;

  // really, errors are bits
  if (error & AVSGetPropErrors::GETPROPERROR_UNSET)
    error_msg = "property is not set";
  else if (error & AVSGetPropErrors::GETPROPERROR_TYPE)
    error_msg = "wrong type";
  else if (error & AVSGetPropErrors::GETPROPERROR_INDEX)
    error_msg = "index error"; // arrays

  if (error)
    env->ThrowError("Error getting frame property data size \"%s\": %s ", propName, error_msg);

  return AVSValue();
}

// array size of a given property, (by setProp append mode arrays can be constructed)
AVSValue GetPropertyNumElements::Create(AVSValue args, void*, IScriptEnvironment* env) {
  AVSValue clip = args[0];
  if (!clip.IsClip())
    env->ThrowError("propNumElements: No clip supplied!");

  PClip child = clip.AsClip();
  VideoInfo vi = child->GetVideoInfo();

  AVSValue cn = env->GetVarDef("current_frame");
  const bool calledFromRunTime = cn.IsInt();

  const char* propName = args[1].AsString();
  const int offset = args[2].AsInt(0);

  int n = calledFromRunTime ? cn.AsInt() : 0;
  n = min(max(n + offset, 0), vi.num_frames - 1);

  PVideoFrame src = child->GetFrame(n, env);

  const AVSMap* avsmap = env->getFramePropsRO(src);

  try {
    int size = env->propNumElements(avsmap, propName);
    return AVSValue(size);
  }
  catch (const AvisynthError& error) {
    env->ThrowError("propNumElements: %s", error.msg);
  }

  return AVSValue();
}

// returns integer enums instead of char (string)
AVSValue GetPropertyType::Create(AVSValue args, void*, IScriptEnvironment* env) {
  AVSValue clip = args[0];
  if (!clip.IsClip())
    env->ThrowError("propGetType: No clip supplied!");

  PClip child = clip.AsClip();
  VideoInfo vi = child->GetVideoInfo();

  AVSValue cn = env->GetVarDef("current_frame");
  const bool calledFromRunTime = cn.IsInt();

  const char* propName = args[1].AsString();
  const int offset = args[2].AsInt(0);

  int n = calledFromRunTime ? cn.AsInt() : 0;
  n = min(max(n + offset, 0), vi.num_frames - 1);

  PVideoFrame src = child->GetFrame(n, env);

  const AVSMap* avsmap = env->getFramePropsRO(src);

  try {
    char prop_type = env->propGetType(avsmap, propName);
    // 'u'nset, 'i'nteger, 'f'loat, 's'string, 'c'lip, 'v'ideoframe, 'm'ethod };
    switch (prop_type) {
    case 'u': return 0;
    case 'i': return 1;
    case 'f': return 2;
    case 's': return 3;
    case 'c': return 4;
    case 'v': return 5;
      //case 'm': return 6;
    default:
      return -1;
    }
  }
  catch (const AvisynthError& error) {
    env->ThrowError("propGetType: %s", error.msg);
  }

  return AVSValue();
}


// Number of properties (keys) for a frame
AVSValue GetPropertyNumKeys::Create(AVSValue args, void*, IScriptEnvironment* env) {
  AVSValue clip = args[0];
  if (!clip.IsClip())
    env->ThrowError("propNumKeys: No clip supplied!");

  PClip child = clip.AsClip();
  VideoInfo vi = child->GetVideoInfo();

  AVSValue cn = env->GetVarDef("current_frame");
  const bool calledFromRunTime = cn.IsInt();

  int n = calledFromRunTime ? cn.AsInt() : 0;
  int offset = args[1].AsInt(0);
  n = min(max(n + offset, 0), vi.num_frames - 1);

  PVideoFrame src = child->GetFrame(n, env);

  const AVSMap* avsmap = env->getFramePropsRO(src);

  try {
    int size = env->propNumKeys(avsmap);
    return AVSValue(size);
  }
  catch (const AvisynthError& error) {
    env->ThrowError("propNumKeys: %s", error.msg);
  }

  return AVSValue();
}

// Get the name (key) of the Nth frame property
AVSValue GetPropertyKeyByIndex::Create(AVSValue args, void*, IScriptEnvironment* env) {
  AVSValue clip = args[0];
  if (!clip.IsClip())
    env->ThrowError("propNumKeys: No clip supplied!");

  PClip child = clip.AsClip();
  VideoInfo vi = child->GetVideoInfo();

  AVSValue cn = env->GetVarDef("current_frame");
  const bool calledFromRunTime = cn.IsInt();

  const int index = args[1].AsInt(0);
  const int offset = args[2].AsInt(0);

  int n = calledFromRunTime ? cn.AsInt() : 0;
  n = min(max(n + offset, 0), vi.num_frames - 1);

  PVideoFrame src = child->GetFrame(n, env);

  const AVSMap* avsmap = env->getFramePropsRO(src);

  try {
    const char* prop_name = env->propGetKey(avsmap, index);
    return AVSValue(env->SaveString(prop_name));
  }
  catch (const AvisynthError& error) {
    env->ThrowError("propGetKeyByIndex: %s", error.msg);
  }

  return AVSValue();
}
