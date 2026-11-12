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

#include "combine.h"
#include "../core/internal.h"

#ifdef AVS_WINDOWS
    #include <avs/win.h>
#else
    #include <avs/posix.h>
#endif

#include <avs/minmax.h>
#include <cmath>
#include <cassert>
#include <algorithm>




/********************************************************************
***** Declare index of new filters for Avisynth's filter engine *****
********************************************************************/

extern const AVSFunction Combine_filters[] = {
  { "StackVertical", BUILTIN_FUNC_PREFIX, "cc+", StackVertical::Create },
  { "StackHorizontal", BUILTIN_FUNC_PREFIX, "cc+", StackHorizontal::Create },
  { "MultiOverlay", BUILTIN_FUNC_PREFIX, "cc+i+", MultiOverlay::Create },
  { "ShowFiveVersions", BUILTIN_FUNC_PREFIX, "ccccc", ShowFiveVersions::Create },
  // first come the custom function versions, bacause of array_of_anything finish
  { "Animate", BUILTIN_FUNC_PREFIX, "iisn.*", Animate::Create, (void*)1 },  // start frame, end frame, filter, function, start-args, end-args
  { "Animate", BUILTIN_FUNC_PREFIX, "ciisn.*", Animate::Create, (void*)1 },
  // then the legacy linear version
  { "Animate", BUILTIN_FUNC_PREFIX, "iis.*", Animate::Create, (void *)0 },  // start frame, end frame, filter, start-args, end-args
  { "Animate", BUILTIN_FUNC_PREFIX, "ciis.*", Animate::Create, (void*)0 },
  { "ApplyRange", BUILTIN_FUNC_PREFIX, "ciis.*", Animate::Create_Range, (void*)0 },
  { 0 }
};






/********************************
 *******   StackVertical   ******
 ********************************/

StackVertical::StackVertical(const std::vector<PClip>& child_array, IScriptEnvironment* env) :
  children(child_array)
{
  vi = children[0]->GetVideoInfo();

  for (size_t i = 1; i < children.size(); ++i) {
    const VideoInfo& vin = children[i]->GetVideoInfo();

    if (vi.width != vin.width)
      env->ThrowError("StackVertical: image widths don't match");

    if (!vi.IsSameColorspace(vin))
      env->ThrowError("StackVertical: image formats don't match");

    if (vi.num_frames < vin.num_frames) // Max of all clips
      vi.num_frames = vin.num_frames;

    vi.height += vin.height;
  }

  // reverse the order of the clips in RGB mode because it's upside-down
  if (vi.IsRGB() && !vi.IsPlanarRGB() && !vi.IsPlanarRGBA()) {
    std::reverse(children.begin(), children.end());
    // get audio and parity from the first in the original list
    firstchildindex = (int)children.size() - 1;
  }
  else {
    firstchildindex = 0;
  }
}


PVideoFrame __stdcall StackVertical::GetFrame(int n, IScriptEnvironment* env)
{
  std::vector<PVideoFrame> frames;
  frames.reserve(children.size());

  for (const auto& child: children)
    frames.emplace_back(child->GetFrame(n, env));

  PVideoFrame dst = env->NewVideoFrameP(vi, &frames[0]);

  const int dst_pitch = dst->GetPitch();
  const int row_size = dst->GetRowSize();
  BYTE* dstp = dst->GetWritePtr();

  for (const auto& src: frames)
  {
    const int src_height = src->GetHeight();
    env->BitBlt(dstp, dst_pitch, src->GetReadPtr(), src->GetPitch(), row_size, src_height);
    dstp += dst_pitch * src_height;
  }

  if (vi.IsPlanar() && (vi.NumComponents() > 1))
  {
    // Copy Planar
    const int planesYUVA[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A};
    const int planesYA[2]   = { PLANAR_Y, PLANAR_A};
    const int planesRGB[4]  = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A};
    const int *planes = vi.IsYA() ? planesYA : (vi.IsYUV() || vi.IsYUVA()) ? planesYUVA : planesRGB;

    // first plane is already processed
    for (int p = 1; p < vi.NumComponents(); p++) {
      const int plane = planes[p];
      dstp = dst->GetWritePtr(plane);
      const int dst_pitch = dst->GetPitch(plane);
      const int row_size = dst->GetRowSize(plane);

      for (const auto& src: frames)
      {
        const int src_height = src->GetHeight(plane);
        env->BitBlt(dstp, dst_pitch, src->GetReadPtr(plane), src->GetPitch(plane), row_size, src_height);
        dstp += dst_pitch * src_height;
      }
    }
  }

  return dst;
}

AVSValue __cdecl StackVertical::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  if (args[1].IsArray()) {
    std::vector<PClip> children(1+args[1].ArraySize());

    children[0] = args[0].AsClip();
    for (int i = 1; i < (int)children.size(); ++i) // Copy clips
      children[i] = args[1][i-1].AsClip();

    return new StackVertical(children, env);
  }
  else if (args[1].IsClip()) { // Make easy to call with trivial 2 clips
    std::vector<PClip> children(2);

    children[0] = args[0].AsClip();
    children[1] = args[1].AsClip();

    return new StackVertical(children, env);
  }
  else {
    env->ThrowError("StackVertical: clip array not recognized!");
    return 0;
  }
}

/**********************************
 *******   MultiOverlay   ******
 **********************************/

// Quick and dirty copy paste of N input clips onto the original (x,y)
// optionally specifying offsets in source and dimensions
MultiOverlay::MultiOverlay(const std::vector<PClip>& child_array, const std::vector<int>& position_array, IScriptEnvironment* env) :
  children(child_array), positions(position_array)
{
  vi = children[0]->GetVideoInfo();

  // 0th is the base clip
  // 1.. the clips to overlay
  for (size_t i = 1; i < children.size(); ++i) {
    const VideoInfo& vin = children[i]->GetVideoInfo();
    if (!vi.IsSameColorspace(vin))
      env->ThrowError("MultiOverlay: image format must match with the base");
  }

  const size_t copy_paste_count = children.size() - 1;
  const size_t param_count = position_array.size();

  // either 2 values per clip
  //   target_x, target_y 
  // or 6 values per clip
  //   target_x, target_y, src_x, src_y, src_width, src_height
  if(copy_paste_count * 2 != param_count && copy_paste_count * 6 != param_count)
    env->ThrowError("MultiOverlay: expected 2 or 6 values per source clip.");

  const bool grey = vi.IsY();
  const bool isRGBPfamily = vi.IsPlanarRGB() || vi.IsPlanarRGBA();

  const int shift_w = (vi.IsPlanar() && !grey && !isRGBPfamily) ? vi.GetPlaneWidthSubsampling(PLANAR_U) : 0;
  const int shift_h = (vi.IsPlanar() && !grey && !isRGBPfamily) ? vi.GetPlaneHeightSubsampling(PLANAR_U) : 0;
  const int mask_w = (1 << shift_w) - 1;
  const int mask_h = (1 << shift_h) - 1;

  auto params_per_clip = param_count / copy_paste_count;
  for (size_t i = 0; i < copy_paste_count; i++) {
    const VideoInfo& vin = children[i + 1]->GetVideoInfo();

    const int target_x = positions[i * params_per_clip + 0];
    const int target_y = positions[i * params_per_clip + 1];
    if (target_x & mask_w)
      env->ThrowError("MultiOverlay: target x must be mod %d for this video format", (1 << shift_w));
    if (target_y & mask_h)
      env->ThrowError("MultiOverlay: target y must be mod %d for this video format", (1 << shift_h));

    if (params_per_clip == 6) {
      // additional source position and dimensions
      const int src_x = positions[i * params_per_clip + 2];
      const int src_y = positions[i * params_per_clip + 3];
      if (src_x < 0 || src_y < 0)
        env->ThrowError("MultiOverlay: source coordinate cannot be negative.");

      const int src_w = positions[i * params_per_clip + 4];
      const int src_h = positions[i * params_per_clip + 5];
      if (src_w <= 0 || src_h <= 0)
        env->ThrowError("MultiOverlay: source width and height must be positive.");

      if (src_x & mask_w)
        env->ThrowError("MultiOverlay: source x must be mod %d for this video format", (1 << shift_w));
      if (src_y & mask_h)
        env->ThrowError("MultiOverlay: source y must be mod %d for this video format", (1 << shift_h));
      if (src_w & mask_w)
        env->ThrowError("MultiOverlay: source width must be mod %d for this video format", (1 << shift_w));
      if (src_h & mask_h)
        env->ThrowError("MultiOverlay: source height must be mod %d for this video format", (1 << shift_h));

      if (src_x + src_w > vin.width)
        env->ThrowError("MultiOverlay: copy exceeds clip width. x=%d, size=%d, clip width=%d", src_x, src_w, vin.width);
      if (src_y + src_h > vin.height)
        env->ThrowError("MultiOverlay: copy exceeds clip height. y=%d, size=%d, clip height=%d", src_y, src_h, vin.height);
    }
  }
}

PVideoFrame __stdcall MultiOverlay::GetFrame(int n, IScriptEnvironment* env)
{
  std::vector<PVideoFrame> frames;
  frames.reserve(children.size());

  for (const auto& child : children)
    frames.emplace_back(child->GetFrame(n, env));

  PVideoFrame dst = frames[0]; // multioverlay target clip
  env->MakeWritable(&dst);

  const size_t copy_paste_count = children.size() - 1;
  const size_t param_count = positions.size();
  const size_t params_per_clip = param_count / copy_paste_count; // 2 or 6
  const bool source_extra = params_per_clip == 6;

  // packed RGB is upside down
  const bool is_packed_rgb = vi.IsRGB24() || vi.IsRGB32() || vi.IsRGB48() || vi.IsRGB64();
  const int bfp = is_packed_rgb || vi.IsYUY2() ? vi.BytesFromPixels(1) : vi.ComponentSize();
  // also for packed RGBs, pixelsize is not enough

  const int planesPacked[] = { DEFAULT_PLANE, DEFAULT_PLANE , DEFAULT_PLANE , DEFAULT_PLANE };
  const int planesYUVA[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
  const int planesYA[2]   = { PLANAR_Y, PLANAR_A };
  const int planesRGB[4]  = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
  const int* planes = is_packed_rgb ? planesPacked : vi.IsYA() ? planesYA : vi.IsY() || vi.IsYUV() || vi.IsYUVA() ? planesYUVA : planesRGB;

  const int planecount = is_packed_rgb || vi.IsYUY2() ? 1 : vi.NumComponents();

  for (int p = 0; p < planecount; p++) {
    const int plane = planes[p];
    BYTE* dstp = dst->GetWritePtr(plane);
    const int dst_pitch = dst->GetPitch(plane);
    const int dst_height = dst->GetHeight(plane);

    // 0th was the target
    for (size_t i = 1; i < frames.size(); i++)
    {
      const auto& src = frames[i];
      const int src_rowsize = src->GetRowSize(plane);

      // these may be safely overridden
      int src_width = src_rowsize / bfp;
      int src_height = src->GetHeight(plane);

      const BYTE* srcp = src->GetReadPtr(plane);
      const int src_pitch = src->GetPitch(plane);

      // packed RGB a Y and all 0th plane is not subsampled.
      const int shift_w = p == 0 ? 0 : vi.GetPlaneWidthSubsampling(plane);
      const int shift_h = p == 0 ? 0 : vi.GetPlaneHeightSubsampling(plane);

      int target_x = positions[(i - 1) * params_per_clip + 0] >> shift_w;
      int target_y = positions[(i - 1) * params_per_clip + 1] >> shift_h;

      int dst_w = dst->GetRowSize(plane) / bfp;

      int src_x = 0;
      int src_y = 0;
      int src_w = src_width;
      int src_h = src_height;

      if (source_extra) {
        src_x = positions[(i - 1) * params_per_clip + 2] >> shift_w;;
        src_y = positions[(i - 1) * params_per_clip + 3] >> shift_h;;
        src_w = positions[(i - 1) * params_per_clip + 4] >> shift_w;;
        src_h = positions[(i - 1) * params_per_clip + 5] >> shift_h;;
      }
      // bring negative target coordinates to 0, 
      // adjust source position and dimension accordingly
      if (target_x < 0) {
        src_x += -target_x;
        src_w -= -target_x;
        target_x = 0;
      }
      if (target_y < 0) {
        src_y += -target_y;
        src_h -= -target_y;
        target_y = 0;
      }

      // check and limit dimension parameters
      
      // Already checked at filter creation:
      // o coordinates and dimensions are subsampling friendly
      // o target x and y >= 0
      // o source width and height > 0
      // o src_x + src_width and src_y + src_height fit in source clip dimensions

      if (src_h > src_height)
        src_h = src_height;
      if (target_y + src_h > dst_height)
        src_h -= (target_y + src_h - dst_height);

      if (src_w > src_width)
        src_w = src_width;
      if (target_x + src_w > dst_w)
        src_w -= (target_x + src_w - dst_w);

      if (src_w > 0 && src_h > 0) {
        src_height = src_h;
        src_width = src_w * bfp;

        if (is_packed_rgb) {
          // start from bottom: packed RGB is upside down
          src_y = (src->GetHeight(plane) - src_y - 1) - (src_height - 1);
          target_y = (dst_height - target_y - 1) - (src_height - 1);
        }

        env->BitBlt(
          dstp + target_x * bfp + target_y * dst_pitch, dst_pitch,
          srcp + src_x * bfp + src_y * src_pitch, src_pitch, src_width, src_height);
      }
    }
  }

  return dst;
}

AVSValue __cdecl MultiOverlay::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  std::vector<PClip> children;
  if (args[1].IsArray()) {
    children.resize(1 + args[1].ArraySize());

    children[0] = args[0].AsClip();
    for (int i = 1; i < (int)children.size(); ++i) // Copy clips
      children[i] = args[1][i - 1].AsClip();
  }
  else if (args[1].IsClip()) { // Make easy to call with trivial 2 clips
    children.resize(2);

    children[0] = args[0].AsClip();
    children[1] = args[1].AsClip();

  }
  else {
    env->ThrowError("MultiOverlay: clip array not recognized!");
    return 0;
  }

  std::vector<int> positions;
  if (!args[2].IsArray()) {
    env->ThrowError("MultiOverlay: position array not recognized!");
    return 0;
  }
  const int pos_count = args[2].ArraySize();
  if (pos_count != 2 * (children.size() - 1) && pos_count != 6 * (children.size() - 1))
    env->ThrowError("MultiOverlay: position array must contain 2 or 6 entries for each clip to overlay!");
  positions.resize(pos_count); // x,y for each overlay clip

  for (int i = 0; i < pos_count; ++i) // Copy positions, x, y, or x, y, src_x, src_y, src_width, src_height
    positions[i] = args[2][i].AsInt();

  return new MultiOverlay(children, positions, env);

}


/**********************************
 *******   StackHorizontal   ******
 **********************************/

StackHorizontal::StackHorizontal(const std::vector<PClip>& child_array, IScriptEnvironment* env) :
  children(child_array)
{
  vi = children[0]->GetVideoInfo();

  for (size_t i = 1; i < children.size(); ++i) {
    const VideoInfo& vin = children[i]->GetVideoInfo();

    if (vi.height != vin.height)
      env->ThrowError("StackHorizontal: image heights don't match");

    if (!vi.IsSameColorspace(vin))
      env->ThrowError("StackHorizontal: image formats don't match");

    if (vi.num_frames < vin.num_frames) // Max of all clips
      vi.num_frames = vin.num_frames;

    vi.width += vin.width;
  }
}

PVideoFrame __stdcall StackHorizontal::GetFrame(int n, IScriptEnvironment* env)
{
  std::vector<PVideoFrame> frames;
  frames.reserve(children.size());

  for (const auto& child : children)
    frames.emplace_back(child->GetFrame(n, env));

  PVideoFrame dst = env->NewVideoFrameP(vi, &frames[0]);
  const int dst_pitch = dst->GetPitch();
  const int height = dst->GetHeight();

  BYTE* dstp = dst->GetWritePtr();
  for (const auto& src: frames)
  {
    const int src_rowsize = src->GetRowSize();
    env->BitBlt(dstp, dst_pitch, src->GetReadPtr(), src->GetPitch(), src_rowsize, height);
    dstp += src_rowsize;
  }

  if (vi.IsPlanar() && (vi.NumComponents() > 1)) {
    // Copy Planar

    const int planesYUVA[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A};
    const int planesYA[2]   = { PLANAR_Y, PLANAR_A};
    const int planesRGB[4]  = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A};
    const int *planes = vi.IsYA() ? planesYA : (vi.IsYUV() || vi.IsYUVA()) ? planesYUVA : planesRGB;

    // first plane is already processed
    for (int p = 1; p < vi.NumComponents(); p++) {
      const int plane = planes[p];
      dstp = dst->GetWritePtr(plane);
      const int dst_pitch = dst->GetPitch(plane);
      const int height = dst->GetHeight(plane);

      for (const auto& src: frames)
      {
        const int src_rowsize = src->GetRowSize(plane);
        env->BitBlt(dstp, dst_pitch, src->GetReadPtr(plane), src->GetPitch(plane), src_rowsize, height);
        dstp += src_rowsize;
      }
    }
  }

  return dst;
}

AVSValue __cdecl StackHorizontal::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  if (args[1].IsArray()) {
    std::vector<PClip> children(1+args[1].ArraySize());

    children[0] = args[0].AsClip();
    for (int i = 1; i < (int)children.size(); ++i) // Copy clips
      children[i] = args[1][i-1].AsClip();

    return new StackHorizontal(children, env);
  }
  else if (args[1].IsClip()) { // Make easy to call with trivial 2 clips
    std::vector<PClip> children(2);

    children[0] = args[0].AsClip();
    children[1] = args[1].AsClip();

    return new StackHorizontal(children, env);
  }
  else {
    env->ThrowError("StackHorizontal: clip array not recognized!");
    return 0;
  }
}



/********************************
 *******   Five Versions   ******
 ********************************/

ShowFiveVersions::ShowFiveVersions(PClip* children, IScriptEnvironment* env)
{
  for (int b=0; b<5; ++b)
    child[b] = children[b];

  vi = child[0]->GetVideoInfo();

  for (int c=1; c<5; ++c)
  {
    const VideoInfo& viprime = child[c]->GetVideoInfo();
    vi.num_frames = max(vi.num_frames, viprime.num_frames);
    if (vi.width != viprime.width || vi.height != viprime.height || vi.pixel_type != viprime.pixel_type)
      env->ThrowError("ShowFiveVersions: video attributes of all clips must match");
  }

  vi.width  *= 3;
  vi.height *= 2;
}

PVideoFrame __stdcall ShowFiveVersions::GetFrame(int n, IScriptEnvironment* env)
{
  PVideoFrame dst = env->NewVideoFrame(vi);
  // frame property source is set later
  BYTE* dstp = dst->GetWritePtr();
  BYTE* dstpU = dst->GetWritePtr(PLANAR_U);
  BYTE* dstpV = dst->GetWritePtr(PLANAR_V);
  const int dst_pitch = dst->GetPitch();
  const int dst_pitchUV = dst->GetPitch(PLANAR_U);
  const int height = dst->GetHeight()/2;
  const int heightUV = dst->GetHeight(PLANAR_U)/2;
  // todo: >8 bits, planar RGB
  if (vi.IsYUV()) {
    const int wg = dst->GetRowSize()/6;
    for (int i=0; i<height; i++){
      memset(dstp + ((height+i)*dst_pitch),        128, wg);
      memset(dstp + ((height+i)*dst_pitch) + wg*5, 128, wg);
    }
    if (dst_pitchUV) {
      const int wgUV = dst->GetRowSize(PLANAR_U)/6;
      for (int i=0; i<heightUV; i++) {
        memset(dstpU + ((heightUV+i)*dst_pitchUV),          128, wgUV);
        memset(dstpU + ((heightUV+i)*dst_pitchUV) + wgUV*5, 128, wgUV);
        memset(dstpV + ((heightUV+i)*dst_pitchUV),          128, wgUV);
        memset(dstpV + ((heightUV+i)*dst_pitchUV) + wgUV*5, 128, wgUV);
      }
    }
  }
  else { // vi.IsRGB()
    const int wg = dst->GetRowSize()/6;
    for (int i=0; i<height; i++){
      memset(dstp + i*dst_pitch,        128, wg);
      memset(dstp + i*dst_pitch + wg*5, 128, wg);
    }
  }

  for (int c=0; c<5; ++c)
  {
    PVideoFrame src = child[c]->GetFrame(n, env);

    if(c == 0) // copy frame properties from the very first
      env->copyFrameProps(src, dst);

    if (vi.IsPlanar()) {
      const BYTE* srcpY = src->GetReadPtr(PLANAR_Y);
      const BYTE* srcpU = src->GetReadPtr(PLANAR_U);
      const BYTE* srcpV = src->GetReadPtr(PLANAR_V);
      const int src_pitchY  = src->GetPitch(PLANAR_Y);
      const int src_pitchUV = src->GetPitch(PLANAR_U);
      const int src_row_sizeY  = src->GetRowSize(PLANAR_Y);
      const int src_row_sizeUV = src->GetRowSize(PLANAR_U);

      // staggered arrangement
      BYTE* dstp2  = dstp  + (c>>1) * src_row_sizeY;
      BYTE* dstp2U = dstpU + (c>>1) * src_row_sizeUV;
      BYTE* dstp2V = dstpV + (c>>1) * src_row_sizeUV;
      if (c&1) {
        dstp2  += (height   * dst_pitch)   + src_row_sizeY /2;
        dstp2U += (heightUV * dst_pitchUV) + src_row_sizeUV/2;
        dstp2V += (heightUV * dst_pitchUV) + src_row_sizeUV/2;
      }

      env->BitBlt(dstp2,  dst_pitch,   srcpY, src_pitchY,  src_row_sizeY,  height);
      env->BitBlt(dstp2U, dst_pitchUV, srcpU, src_pitchUV, src_row_sizeUV, heightUV);
      env->BitBlt(dstp2V, dst_pitchUV, srcpV, src_pitchUV, src_row_sizeUV, heightUV);
    }
    else {
      const BYTE* srcp = src->GetReadPtr();
      const int src_pitch = src->GetPitch();
      const int src_row_size = src->GetRowSize();

      // staggered arrangement
      BYTE* dstp2 = dstp + (c>>1) * src_row_size;
      if ((c&1)^vi.IsRGB())
        dstp2 += (height * dst_pitch);
      if (c&1)
        dstp2 += vi.BytesFromPixels(vi.width/6);

      env->BitBlt(dstp2, dst_pitch, srcp, src_pitch, src_row_size, height);
    }
  }

  return dst;
}


AVSValue __cdecl ShowFiveVersions::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  PClip children[5];
  for (int i=0; i<5; ++i)
    children[i] = args[i].AsClip();
  return new ShowFiveVersions(children, env);
}










/**************************************
 *******   Animate (Recursive)   ******
 **************************************/

Animate::Animate( PClip context, int _first, int _last, const char* _name, const AVSValue* _args_before,
                  const AVSValue* _args_after, int _num_args, bool _range_limit, const AVSValue& _custom_fn, IScriptEnvironment* env )
   : first(_first), last(_last), num_args(_num_args), name(_name), range_limit(_range_limit), custom_fn(_custom_fn)
{
  if (first > last)
    env->ThrowError("Animate: final frame number must be greater than initial.");

  if (first == last && (!range_limit))
    env->ThrowError("Animate: final frame cannot be the same as initial frame.");

  // check that argument types match
  for (int arg=0; arg<num_args; ++arg) {
    const AVSValue& a = _args_before[arg];
    const AVSValue& b = _args_after[arg];
    if (a.IsString() && b.IsString()) {
      if (lstrcmp(a.AsString(), b.AsString()))
        env->ThrowError("Animate: string arguments must match before and after");
    }
    else if (a.IsBool() && b.IsBool()) {
      if (a.AsBool() != b.AsBool())
        env->ThrowError("Animate: boolean arguments must match before and after");
    }
    else if (a.IsFloat() && b.IsFloat()) {
      // ok; also catches other numeric types
    }
    else if (a.IsClip() && b.IsClip()) {
      // ok
    }
    else {
      env->ThrowError("Animate: must have two argument lists with matching types");
    }
  }

  if (custom_fn.IsFunction()) {
    // pre-check validity

    // We could even check if function returns f(0.0) = 0.0 and f(1.0) = 1.0
    // But we are not that serious. Who implements custom function, will take care of it if needed
    constexpr bool check_first = false;
    constexpr bool check_last = false;

    double result_first = 0.0;
    double result_last = 1.0;
    PFunction func = custom_fn.AsFunction();
    try {

      // The Animate callback is a plain scalar function, no child
      // one compulsory double "stage" parameter
      const char* argnames[1] = { "stage" };

      // for syntax check this is always called
      AVSValue test1_args[1] = { 0.0 };
      const AVSValue test1_args_array = AVSValue(test1_args, 1);
      result_first = env->Invoke3(AVSValue(), func, test1_args_array, argnames).AsFloat();

      if (check_last) {
        if (first != last) {
          AVSValue test2_args[1] = { 1.0 };
          const AVSValue test2_args_array = AVSValue(test2_args, 1);
          result_last = env->Invoke3(AVSValue(), func, test2_args_array, argnames).AsFloat();
        }
      }
    }
    catch (IScriptEnvironment::NotFound) {
      // GetDefinition() is nullptr for a function passed by name: Func(myfunc)
      const Function* def = func->GetDefinition();
      env->ThrowError("Animate: cannot call the custom function (%s)\n"
        "It must have one optional float parameter named \"stage\": function f(float \"stage\")%s%s",
        func->ToString(env), def ? "\nParameter types given: " : "", def ? def->param_types : "");
    }
    catch (const AvisynthError& error) {
      env->ThrowError("Animate: Error in custom function: %s\n%s",
        func->ToString(env), error.msg);
    }

    if (check_first && result_first != 0.0)
      env->ThrowError("Animate: Error in custom function, for stage 0.0: 0.0 must be returned.\n%s\n",
        func->ToString(env));
    if (check_last && result_last != 1.0)
      env->ThrowError("Animate: Error in custom function, for stage 1.0: 1.0 must be returned.\n%s\n",
        func->ToString(env));
  }

  // copy args, and add initial clip arg for OOP notation

  if (context)
    num_args++;

  args_before = std::vector<AVSValue>(num_args*3);
  args_after = args_before.data() + num_args;
  args_now = args_after + num_args;

  if (context) {
    args_after[0] = args_before[0] = context;
    for (int i=1; i<num_args; ++i) {
      args_before[i] = _args_before[i-1];
      args_after[i] = _args_after[i-1];
    }
  }
  else {
    for (int i=0; i<num_args; ++i) {
      args_before[i] = _args_before[i];
      args_after[i] = _args_after[i];
    }
  }

  memset(cache_stage, -1, sizeof(cache_stage));

  // first clip with starting parameter values
  cache[0] = env->Invoke(name, AVSValue(args_before.data(), num_args)).AsClip();
  cache_stage[0] = 0;
  VideoInfo vi1 = cache[0]->GetVideoInfo();

  if (range_limit) {
    VideoInfo vi = context->GetVideoInfo();

    if (vi.width != vi1.width || vi.height != vi1.height)
      env->ThrowError("ApplyRange: Filtered and unfiltered video frame sizes must match");

    if (!vi.IsSameColorspace(vi1))
      env->ThrowError("ApplyRange: Filtered and unfiltered video colorspace must match");
  }
  else {
    // last clip with ending parameter values
    cache[1] = env->Invoke(name, AVSValue(args_after, num_args)).AsClip();
    cache_stage[1] = (int64_t)last - first;
    VideoInfo vi2 = cache[1]->GetVideoInfo();

    if (vi1.width != vi2.width || vi1.height != vi2.height)
      env->ThrowError("Animate: initial and final video frame sizes must match");
  }
}


bool __stdcall Animate::GetParity(int n)
{
  if (range_limit) {
    if ((n<first) || (n>last)) {
      return args_after[0].AsClip()->GetParity(n);
    }
  }
  // We could go crazy here and replicate the GetFrame
  // logic and share the cache_stage but it is not
  // really worth it. Although clips that change parity
  // are supported they are very confusing.
  return cache[0]->GetParity(n);
}

// Integer arithmetic helpers for Animate, 128 bit intermediates (portable, no __int128)

// 128-bit integer, two's complement
struct my_int128_t {
  uint64_t H; // high 64 bits (sign in the top bit)
  uint64_t L; // low 64 bits
};

static inline my_int128_t neg_128(my_int128_t a) {
  my_int128_t r;
  r.L = ~a.L + 1;
  r.H = ~a.H + (r.L == 0 ? 1 : 0);
  return r;
}

// unsigned 64 x 64 -> 128
static inline my_int128_t umul_64_64_to_128(uint64_t a, uint64_t b) {
  const uint64_t a_lo = a & 0xFFFFFFFF, a_hi = a >> 32;
  const uint64_t b_lo = b & 0xFFFFFFFF, b_hi = b >> 32;
  const uint64_t lo_lo = a_lo * b_lo;
  const uint64_t hi_lo = a_hi * b_lo;
  const uint64_t lo_hi = a_lo * b_hi;
  const uint64_t hi_hi = a_hi * b_hi;
  const uint64_t mid = (lo_lo >> 32) + (hi_lo & 0xFFFFFFFF) + (lo_hi & 0xFFFFFFFF); // < 3 * 2^32
  my_int128_t r;
  r.L = (lo_lo & 0xFFFFFFFF) | (mid << 32);
  r.H = (hi_lo >> 32) + (lo_hi >> 32) + hi_hi + (mid >> 32);
  return r;
}

// signed 64 x 64 -> 128
static inline my_int128_t mul_64_64_to_128(int64_t a, int64_t b) {
  // magnitudes as unsigned: no overflow for INT64_MIN
  const uint64_t abs_a = a < 0 ? 0 - (uint64_t)a : (uint64_t)a;
  const uint64_t abs_b = b < 0 ? 0 - (uint64_t)b : (uint64_t)b;
  const my_int128_t r = umul_64_64_to_128(abs_a, abs_b);
  return ((a < 0) != (b < 0)) ? neg_128(r) : r;
}

static inline my_int128_t add_128(my_int128_t a, my_int128_t b) {
  my_int128_t r;
  r.L = a.L + b.L;
  r.H = a.H + b.H + (r.L < a.L ? 1 : 0);
  return r;
}

// arithmetic shift right by 32, saturated to the int64 range
static inline int64_t sar32_128_saturate(my_int128_t a) {
  const uint64_t H = (uint64_t)((int64_t)a.H >> 32); // sign extended
  const uint64_t L = (a.H << 32) | (a.L >> 32);
  if (H == 0 && L <= (uint64_t)INT64_MAX)
    return (int64_t)L;
  if (H == ~0ULL && L >= (uint64_t)INT64_MIN)
    return (int64_t)L; // two's complement
  return ((int64_t)H < 0) ? INT64_MIN : INT64_MAX;
}

// n / d, the quotient must fit into 64 bits
static inline uint64_t udiv_128_32(my_int128_t n, uint32_t d) {
  const uint32_t limbs[4] = { (uint32_t)(n.H >> 32), (uint32_t)n.H, (uint32_t)(n.L >> 32), (uint32_t)n.L };
  uint64_t rem = 0, q = 0;
  for (int i = 0; i < 4; i++) {
    const uint64_t cur = (rem << 32) | limbs[i];
    q = (q << 32) | (cur / d); // the upper limbs of the quotient are zero by the precondition
    rem = cur % d;
  }
  return q;
}

// Linear interpolation, exact: a + (b - a) * stage / scale, rounded half up (toward +infinity).
// 0 < stage < scale < 2^32. The result is always between a and b.
static int64_t Interpolate_linear_exact(int64_t a, int64_t b, uint32_t stage, uint32_t scale) {
  const bool up = b >= a;
  const uint64_t d = up ? (uint64_t)b - (uint64_t)a : (uint64_t)a - (uint64_t)b; // |b-a| < 2^64
  // N = d * stage < 2^96. Rounding offset: half up (toward +infinity) in both directions
  my_int128_t N = umul_64_64_to_128(d, stage);
  const my_int128_t h = { 0, up ? scale / 2 : (scale - 1) / 2 };
  N = add_128(N, h);
  const uint64_t q = udiv_128_32(N, scale); // q <= d
  return (int64_t)(up ? (uint64_t)a + q : (uint64_t)a - q);
}

// Interpolation with a factor from a user function, 96.32 bit fixed point: factor 2^32 means 1.0.
// The factor may be outside 0..1 (over- and undershoot), the result saturates to the int64 range.
constexpr int ANIMATE_INT_ARITH_SCALEBITS = 32;
constexpr double ANIMATE_FULL_SCALE_D = 4294967296.0; // 2^32

static int64_t Interpolate_factor(int64_t a, int64_t b, double stage_mod) {
  const double f = stage_mod * ANIMATE_FULL_SCALE_D;
  if (std::fabs(f) >= 9007199254740992.0) {
    // |stage_mod| >= 2^21: the fixed point factor would not be exact anyway, compute in double
    const double r = (double)a + ((double)b - (double)a) * stage_mod;
    if (r >= 9223372036854775807.0) return INT64_MAX;
    if (r <= -9223372036854775808.0) return INT64_MIN;
    return (int64_t)std::llround(r);
  }
  const int64_t factor = (int64_t)std::llround(f); // |factor| < 2^53
  // a*(2^32 - factor) + b*factor + 2^31, then >> 32. |each term| < 2^63 * 2^54: fits into 128 bits
  my_int128_t temp = add_128(mul_64_64_to_128(a, ((int64_t)1 << ANIMATE_INT_ARITH_SCALEBITS) - factor),
                             mul_64_64_to_128(b, factor));
  const my_int128_t round = { 0, (uint64_t)1 << (ANIMATE_INT_ARITH_SCALEBITS - 1) };
  temp = add_128(temp, round);
  return sar32_128_saturate(temp);
}

// Fills args_now for 0 < stage < scale. The ends are never computed: they are the given start and end arguments.
void Animate::InterpolateArgs(int64_t stage, int64_t scale, IScriptEnvironment* env)
{
  const double stage_lin = (double)stage / (double)scale; // linear 0.0 .. 1.0
  double stage_mod = stage_lin;
  const bool has_fn = custom_fn.IsFunction();

  if (has_fn) {
    // custom function maps x -> fn(x). It is not called at the ends, f(0)=0 and f(1)=1 need no check.
    // In between the result may be outside 0..1: over- and undershoot, no clamping.
    PFunction func = custom_fn.AsFunction();
    const char* argnames[1] = { "stage" };
    AVSValue actual_args[1] = { stage_lin };
    const AVSValue actual_args_array = AVSValue(actual_args, 1);
    AVSValue result = env->Invoke3(AVSValue(), func, actual_args_array, argnames);
    if (!result.IsFloat()) // int and long are accepted as well
      env->ThrowError("Animate: custom function must return a number (stage %f)", stage_lin);
    stage_mod = result.AsFloat();
    if (!std::isfinite(stage_mod))
      env->ThrowError("Animate: custom function returned NaN or infinity (stage %f)", stage_lin);
  }

  // A function returning its stage (linear) gives the same result as no function.
  const bool linear = !has_fn || stage_mod == stage_lin;

  for (int a = 0; a < num_args; ++a) {
    if (args_before[a].IsInt() && args_after[a].IsInt()) {
      const int64_t start = args_before[a].AsLong();
      const int64_t end = args_after[a].AsLong();
      args_now[a] = linear ? Interpolate_linear_exact(start, end, (uint32_t)stage, (uint32_t)scale)
                           : Interpolate_factor(start, end, stage_mod);
    }
    else if (args_before[a].IsFloat() && args_after[a].IsFloat()) {
      // note: AsFloat() returns double
      const double start = args_before[a].AsFloat();
      const double end = args_after[a].AsFloat();
      // start + (end - start) * s: exactly start when the two are equal
      double interpolated_value = start + (end - start) * stage_mod;
      if (linear) {
        // linear: rounding error can put the result slightly outside start..end
        interpolated_value = clamp(interpolated_value, std::min(start, end), std::max(start, end));
      }
      else if (!std::isfinite(interpolated_value))
        env->ThrowError("Animate: interpolated value is out of the float range (stage %f)", stage_lin);
      args_now[a] = interpolated_value;
    }
    else {
      args_now[a] = args_before[a]; // bool, string, clip: no transition
    }
  }
}

PVideoFrame __stdcall Animate::GetFrame(int n, IScriptEnvironment* env)
{
  if (range_limit) {
    if ((n<first) || (n>last)) {
      return args_after[0].AsClip()->GetFrame(n, env);
    }
    return cache[0]->GetFrame(n, env);
  }
  // int64: last - first can exceed the int range
  const int64_t scale = (int64_t)last - first;
  const int64_t stage = (int64_t)clamp(n, first, last) - first;
  for (int i=0; i<cache_size; ++i)
    if (cache_stage[i] == stage)
      return cache[i]->GetFrame(n, env);

  // filter not found in cache--create it
  int furthest = 0;
  for (int j=1; j<cache_size; ++j)
    if (std::llabs(stage-cache_stage[j]) > std::llabs(stage-cache_stage[furthest]))
      furthest = j;

  const AVSValue* args_use;
  if (stage == 0)
    args_use = args_before.data(); // the ends are exact, no interpolation, the custom function is not called
  else if (stage == scale)
    args_use = args_after;
  else {
    InterpolateArgs(stage, scale, env);
    args_use = args_now;
  }
  cache_stage[furthest] = stage;
  cache[furthest] = env->Invoke(name, AVSValue(args_use, num_args)).AsClip();
  return cache[furthest]->GetFrame(n, env);
}

void __stdcall Animate::GetAudio(void* buf, int64_t start, int64_t count, IScriptEnvironment* env)  {
  if (range_limit) {  // Applyrange - hard switch between streams.

    const VideoInfo& vi1 = cache[0]->GetVideoInfo();
    const int64_t start_switch =  vi1.AudioSamplesFromFrames(first);
    const int64_t end_switch   =  vi1.AudioSamplesFromFrames(last+1);

    if ( (start+count <= start_switch) || (start >= end_switch) ) {
      // Everything unfiltered
      args_after[0].AsClip()->GetAudio(buf, start, count, env);
      return;
    }
    else if ( (start < start_switch) || (start+count > end_switch) ) {
      // We are at one or both switchover points

      // The bit before
      if (start_switch > start) {
    const int64_t pre_count = start_switch - start;
        args_after[0].AsClip()->GetAudio(buf, start, pre_count, env);  // UnFiltered
    start += pre_count;
    count -= pre_count;
    buf = (void*)( (BYTE*)buf + vi1.BytesFromAudioSamples(pre_count) );
      }

      // The bit in the middle
      const int64_t filt_count = (end_switch < start+count) ? (end_switch - start) : count;
      cache[0]->GetAudio(buf, start, filt_count, env);  // Filtered
      start += filt_count;
      count -= filt_count;
      buf = (void*)( (BYTE*)buf + vi1.BytesFromAudioSamples(filt_count) );

      // The bit after
      if (count > 0)
        args_after[0].AsClip()->GetAudio(buf, start, count, env);  // UnFiltered

      return;
    }
    // Everything filtered
  }
  cache[0]->GetAudio(buf, start, count, env);  // Filtered
}


AVSValue __cdecl Animate::Create(AVSValue args, void* user_data, IScriptEnvironment* env)
{
  auto anim_kind = reinterpret_cast<intptr_t>(user_data);
  // 0: legacy 1: extra callback function parameter

  PClip context;
  // When function parameter exists, it shifts the parameter array by one
  const int param_index = (anim_kind == 0) ? 3 : 4;
  // Convert the clip-at-zeroth-param version to the other signature.
  if (args[0].IsClip()) {
    // ciis.* -> iis.*
    // ciisn.* -> iisn.*
    context = args[0].AsClip();
    args = AVSValue(&args[1], param_index + 1);
  }
  const int first = args[0].AsInt();
  const int last = args[1].AsInt();
  const char* const name = args[2].AsString();

  int n = args[param_index].ArraySize();
  if (n&1)
    env->ThrowError("Animate: must have two argument lists of the same length"); // two sets
  // no element access on an empty array (operator[] asserts)
  const AVSValue* list = n > 0 ? &args[param_index][0] : nullptr;
  return new Animate(context, first, last, name, list, n > 0 ? &args[param_index][n>>1] : nullptr, n>>1, false, 
    anim_kind == 0 ? AVSValue() : args[3], env);
}


AVSValue __cdecl Animate::Create_Range(AVSValue args, void*, IScriptEnvironment* env)
{
  PClip context = args[0].AsClip();

  const int first = args[1].AsInt();
  const int last = args[2].AsInt();
  const char* const name = args[3].AsString();
  int n = args[4].ArraySize();
  // ApplyRange(0, 10, "Invert"): no extra arguments, no element access on the empty array (operator[] asserts)
  const AVSValue* list = n > 0 ? &args[4][0] : nullptr;
  return new Animate(context, first, last, name, list, list, n, true, AVSValue(), env);
}
