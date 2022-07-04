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


#include "field.h"
#ifdef INTEL_INTRINSICS
#include "intel/resample_sse.h"
#else
#include "resample.h"
#endif
#include <avs/minmax.h>
#include "../core/internal.h"


/**** Factory methods ****/

static AVSValue __cdecl Create_DoubleWeave(AVSValue args, void*, IScriptEnvironment* env);
static AVSValue __cdecl Create_Weave(AVSValue args, void*, IScriptEnvironment* env);
static AVSValue __cdecl Create_Pulldown(AVSValue args, void*, IScriptEnvironment* env);
static AVSValue __cdecl Create_SwapFields(AVSValue args, void*, IScriptEnvironment* env);
static AVSValue __cdecl Create_Bob(AVSValue args, void*, IScriptEnvironment* env);


/********************************************************************
***** Declare index of new filters for Avisynth's filter engine *****
********************************************************************/

extern const AVSFunction Field_filters[] = {
  { "ComplementParity", BUILTIN_FUNC_PREFIX, "c", ComplementParity::Create },
  { "AssumeTFF",        BUILTIN_FUNC_PREFIX, "c", AssumeParity::Create, (void*)true },
  { "AssumeBFF",        BUILTIN_FUNC_PREFIX, "c", AssumeParity::Create, (void*)false },
  { "AssumeFieldBased", BUILTIN_FUNC_PREFIX, "c", AssumeFieldBased::Create },
  { "AssumeFrameBased", BUILTIN_FUNC_PREFIX, "c", AssumeFrameBased::Create },
  { "SeparateColumns",  BUILTIN_FUNC_PREFIX, "ci", SeparateColumns::Create },
  { "WeaveColumns",     BUILTIN_FUNC_PREFIX, "ci", WeaveColumns::Create },
  { "SeparateRows",     BUILTIN_FUNC_PREFIX, "ci", SeparateRows::Create },
  { "WeaveRows",        BUILTIN_FUNC_PREFIX, "ci", WeaveRows::Create },
  { "SeparateFields",   BUILTIN_FUNC_PREFIX, "c", SeparateFields::Create },
  { "Weave",            BUILTIN_FUNC_PREFIX, "c", Create_Weave },
  { "DoubleWeave",      BUILTIN_FUNC_PREFIX, "c", Create_DoubleWeave },
  { "Pulldown",         BUILTIN_FUNC_PREFIX, "cii", Create_Pulldown },
  { "SelectEvery",      BUILTIN_FUNC_PREFIX, "cii*", SelectEvery::Create },
  { "SelectEven",       BUILTIN_FUNC_PREFIX, "c", SelectEvery::Create_SelectEven },
  { "SelectOdd",        BUILTIN_FUNC_PREFIX, "c", SelectEvery::Create_SelectOdd },
  { "Interleave",       BUILTIN_FUNC_PREFIX, "c+", Interleave::Create },
  { "SwapFields",       BUILTIN_FUNC_PREFIX, "c", Create_SwapFields },
  { "Bob",              BUILTIN_FUNC_PREFIX, "c[b]f[c]f[height]i", Create_Bob },
  { "SelectRangeEvery", BUILTIN_FUNC_PREFIX, "c[every]i[length]i[offset]i[audio]b", SelectRangeEvery::Create},
  { NULL }
};





/*********************************
 *******   SeparateColumns  ******
 *********************************/

SeparateColumns::SeparateColumns(PClip _child, int _interval, IScriptEnvironment* env)
 : GenericVideoFilter(_child), interval(_interval)
{
  if (_interval <= 0)
    env->ThrowError("SeparateColumns: interval must be greater than zero.");

  if (_interval > vi.width)
    env->ThrowError("SeparateColumns: interval must be less than or equal width.");

  if (vi.width % _interval)
    env->ThrowError("SeparateColumns: width must be mod %d.", _interval);

  vi.width /= _interval;
  vi.MulDivFPS(_interval, 1);
  vi.num_frames *= _interval;

  if (vi.num_frames < 0)
    env->ThrowError("SeparateColumns: Maximum number of frames exceeded.");


  if (vi.IsYUY2() && vi.width & 1)
    env->ThrowError("SeparateColumns: YUY2 output width must be even.");
  if (vi.Is420() && vi.width & 1)
    env->ThrowError("SeparateColumns: YUV420 output width must be even.");
  if (vi.Is422() && vi.width & 1)
    env->ThrowError("SeparateColumns: YUV422 output width must be even.");
  if (vi.IsYV411() && vi.width & 3)
    env->ThrowError("SeparateColumns: YV411 output width must be mod 4.");
}


PVideoFrame SeparateColumns::GetFrame(int n, IScriptEnvironment* env)
{
  const int m = n%interval;
  const int f = n/interval;

  PVideoFrame src = child->GetFrame(f, env);
  PVideoFrame dst = env->NewVideoFrameP(vi, &src);

  if (vi.IsPlanar()) {
    int planes_y[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
    int planes_r[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
    int *planes = (vi.IsYUV() || vi.IsYUVA()) ? planes_y : planes_r;
    for (int p = 0; p < vi.NumComponents(); ++p) {
      int plane = planes[p];
      const int srcpitch = src->GetPitch(plane);
      const int dstpitch = dst->GetPitch(plane);
      const int height = dst->GetHeight(plane);
      const int rowsize_pixels = dst->GetRowSize(plane) / vi.ComponentSize();

      const BYTE* srcp = src->GetReadPtr(plane);
      BYTE* dstp = dst->GetWritePtr(plane);

      switch (vi.ComponentSize()) {
      case 1:
        for (int y = 0; y < height; y++) {
          for (int i = m, j = 0; j < rowsize_pixels; i += interval, j += 1) {
            dstp[j] = srcp[i];
          }
          srcp += srcpitch;
          dstp += dstpitch;
        }
        break;
      case 2:
        for (int y = 0; y < height; y++) {
          for (int i = m, j = 0; j < rowsize_pixels; i += interval, j += 1) {
            reinterpret_cast<uint16_t *>(dstp)[j] = reinterpret_cast<const uint16_t *>(srcp)[i];
          }
          srcp += srcpitch;
          dstp += dstpitch;
        }
        break;
      case 4:
        for (int y = 0; y < height; y++) {
          for (int i = m, j = 0; j < rowsize_pixels; i += interval, j += 1) {
            reinterpret_cast<float *>(dstp)[j] = reinterpret_cast<const float *>(srcp)[i];
          }
          srcp += srcpitch;
          dstp += dstpitch;
        }
        break;
      }
    }
  }
  else if (vi.IsYUY2()) {
    const int srcpitch = src->GetPitch();
    const int dstpitch = dst->GetPitch();
    const int height = dst->GetHeight();
    const int rowsize = dst->GetRowSize();

    const BYTE* srcp = src->GetReadPtr();
    BYTE* dstp = dst->GetWritePtr();

    const int m2 = m*2;
    const int interval2 = interval*2;
    const int interval4 = interval*4;

    for (int y=0; y<height; y+=1) {
      for (int i=m2, j=0; j<rowsize; i+=interval4, j+=4) {
        // Luma
        dstp[j+0] = srcp[i+0];
        dstp[j+2] = srcp[i+interval2];
        // Chroma
        dstp[j+1] = srcp[i+m2+1];
        dstp[j+3] = srcp[i+m2+3];
      }
      srcp += srcpitch;
      dstp += dstpitch;
    }
  }
  else if (vi.IsRGB24() || vi.IsRGB48()) {
    const int srcpitch = src->GetPitch();
    const int dstpitch = dst->GetPitch();
    const int height = dst->GetHeight();
    const int rowsize_pixels = dst->GetRowSize() / vi.ComponentSize();

    const BYTE* srcp = src->GetReadPtr();
    BYTE* dstp = dst->GetWritePtr();

    const int m3 = m * 3;
    const int interval3 = interval * 3;

    if (vi.IsRGB24())
    {
      for (int y = 0; y < height; y += 1) {
        for (int i = m3, j = 0; j < rowsize_pixels; i += interval3, j += 3) {
          dstp[j + 0] = srcp[i + 0];
          dstp[j + 1] = srcp[i + 1];
          dstp[j + 2] = srcp[i + 2];
        }
        srcp += srcpitch;
        dstp += dstpitch;
      }
    }
    else {
      // RGB48
      for (int y = 0; y < height; y += 1) {
        for (int i = m3, j = 0; j < rowsize_pixels; i += interval3, j += 3) {
          reinterpret_cast<uint16_t *>(dstp)[j + 0] = reinterpret_cast<const uint16_t *>(srcp)[i + 0];
          reinterpret_cast<uint16_t *>(dstp)[j + 1] = reinterpret_cast<const uint16_t *>(srcp)[i + 1];
          reinterpret_cast<uint16_t *>(dstp)[j + 2] = reinterpret_cast<const uint16_t *>(srcp)[i + 2];
        }
        srcp += srcpitch;
        dstp += dstpitch;
      }
    }
  }
  else if (vi.IsRGB32()) {
    const int srcpitch4 = src->GetPitch()>>2;
    const int dstpitch4 = dst->GetPitch()>>2;
    const int height = dst->GetHeight();
    const int rowsize4 = dst->GetRowSize()>>2;

    const int* srcp4 = (const int*)src->GetReadPtr();
    int* dstp4 = (int*)dst->GetWritePtr();

    for (int y=0; y<height; y+=1) {
      for (int i=m, j=0; j<rowsize4; i+=interval, j+=1) {
        dstp4[j] = srcp4[i];
      }
      srcp4 += srcpitch4;
      dstp4 += dstpitch4;
    }
  }
  else if (vi.IsRGB64()) {
    const int srcpitch8 = src->GetPitch()>>3;
    const int dstpitch8 = dst->GetPitch()>>3;
    const int height = dst->GetHeight();
    const int rowsize8 = dst->GetRowSize()>>3;

    const uint64_t* srcp8 = (const uint64_t*)src->GetReadPtr();
    uint64_t* dstp8 = (uint64_t*)dst->GetWritePtr();

    for (int y=0; y<height; y+=1) {
      for (int i=m, j=0; j<rowsize8; i+=interval, j+=1) {
        dstp8[j] = srcp8[i];
      }
      srcp8 += srcpitch8;
      dstp8 += dstpitch8;
    }
  }
  return dst;
}


AVSValue __cdecl SeparateColumns::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  if (args[1].AsInt() == 1)
    return args[0];

  return new SeparateColumns(args[0].AsClip(), args[1].AsInt(), env);
}







/*******************************
 *******   WeaveColumns   ******
 *******************************/

WeaveColumns::WeaveColumns(PClip _child, int _period, IScriptEnvironment* env)
 : GenericVideoFilter(_child), period(_period), inframes(vi.num_frames)
{
  if (_period <= 0)
    env->ThrowError("WeaveColumns: period must be greater than zero.");

  vi.width *= _period;
  vi.MulDivFPS(1, _period);
  vi.num_frames += _period-1; // Ceil!
  vi.num_frames /= _period;
}


PVideoFrame WeaveColumns::GetFrame(int n, IScriptEnvironment* env)
{
  const int b = n * period;

  PVideoFrame dst = env->NewVideoFrame(vi);

  for (int m=0; m<period; m++) {
    const int f = b+m < inframes ? b+m : inframes-1;
    PVideoFrame src = child->GetFrame(f, env);
    if(0 == m)
     env->copyFrameProps(src, dst);

    if (vi.IsPlanar()) {
      int planes_y[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
      int planes_r[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
      int *planes = (vi.IsYUV() || vi.IsYUVA()) ? planes_y : planes_r;
      for (int p = 0; p < vi.NumComponents(); ++p) {
        int plane = planes[p];
        BYTE *_dstp = dst->GetWritePtr(plane);
        const int srcpitch = src->GetPitch(plane);
        const int dstpitch = dst->GetPitch(plane);
        const int height = src->GetHeight(plane);
        const int rowsize_pixels = src->GetRowSize(plane) / vi.ComponentSize();

        const BYTE* srcp = src->GetReadPtr(plane);
        BYTE* dstp = _dstp;
        switch (vi.ComponentSize()) {
        case 1:
          for (int y = 0; y < height; y++) {
            for (int i = m, j = 0; j < rowsize_pixels; i += period, j += 1) {
              dstp[i] = srcp[j];
            }
            srcp += srcpitch;
            dstp += dstpitch;
          }
          break;
        case 2:
          for (int y = 0; y < height; y++) {
            for (int i = m, j = 0; j < rowsize_pixels; i += period, j += 1) {
              reinterpret_cast<uint16_t *>(dstp)[i] = reinterpret_cast<const uint16_t *>(srcp)[j];
            }
            srcp += srcpitch;
            dstp += dstpitch;
          }
          break;
        case 4:
          for (int y = 0; y < height; y++) {
            for (int i = m, j = 0; j < rowsize_pixels; i += period, j += 1) {
              reinterpret_cast<float *>(dstp)[i] = reinterpret_cast<const float *>(srcp)[j];
            }
            srcp += srcpitch;
            dstp += dstpitch;
          }
          break;
        }
      }
    }
    else if (vi.IsYUY2()) {
      BYTE *_dstp = dst->GetWritePtr();
      const int dstpitch = dst->GetPitch();

      const int srcpitch = src->GetPitch();
      const int height = src->GetHeight();
      const int rowsize = src->GetRowSize();
      const BYTE* srcp = src->GetReadPtr();
      BYTE* dstp = _dstp;
      const int m2 = m*2;
      const int period2 = period*2;
      const int period4 = period*4;

      for (int y=0; y<height; y+=1) {
        for (int i=m2, j=0; j<rowsize; i+=period4, j+=4) {
          // Luma
          dstp[i+0]       = srcp[j+0];
          dstp[i+period2] = srcp[j+2];
          // Chroma
          dstp[i+m2+1]    = srcp[j+1];
          dstp[i+m2+3]    = srcp[j+3];
        }
        srcp += srcpitch;
        dstp += dstpitch;
      }
    }
    else if (vi.IsRGB24() || vi.IsRGB48()) {
      BYTE *_dstp = dst->GetWritePtr();
      const int dstpitch = dst->GetPitch();

      const int srcpitch = src->GetPitch();
      const int height = src->GetHeight();
      const int rowsize_pixels = src->GetRowSize() / vi.ComponentSize();
      const BYTE* srcp = src->GetReadPtr();
      BYTE* dstp = _dstp;
      const int m3 = m*3;
      const int period3 = period*3;
      if (vi.IsRGB24()) {
        for (int y = 0; y < height; y += 1) {
          for (int i = m3, j = 0; j < rowsize_pixels; i += period3, j += 3) {
            dstp[i + 0] = srcp[j + 0];
            dstp[i + 1] = srcp[j + 1];
            dstp[i + 2] = srcp[j + 2];
          }
          srcp += srcpitch;
          dstp += dstpitch;
        }
      }
      else {
        // RGB48
        for (int y = 0; y < height; y += 1) {
          for (int i = m3, j = 0; j < rowsize_pixels; i += period3, j += 3) {
            reinterpret_cast<uint16_t *>(dstp)[i + 0] = reinterpret_cast<const uint16_t *>(srcp)[j + 0];
            reinterpret_cast<uint16_t *>(dstp)[i + 1] = reinterpret_cast<const uint16_t *>(srcp)[j + 1];
            reinterpret_cast<uint16_t *>(dstp)[i + 2] = reinterpret_cast<const uint16_t *>(srcp)[j + 2];
          }
          srcp += srcpitch;
          dstp += dstpitch;
        }
      }
    }
    else if (vi.IsRGB32()) {
      BYTE *_dstp = dst->GetWritePtr();
      const int dstpitch = dst->GetPitch();

      const int srcpitch4 = src->GetPitch()>>2;
      const int dstpitch4 = dstpitch>>2;
      const int height = src->GetHeight();
      const int rowsize4 = src->GetRowSize()>>2;
      const int* srcp4 = (const int*)src->GetReadPtr();
      int* dstp4 = (int*)_dstp;

      for (int y=0; y<height; y+=1) {
        for (int i=m, j=0; j<rowsize4; i+=period, j+=1) {
          dstp4[i] = srcp4[j];
        }
        srcp4 += srcpitch4;
        dstp4 += dstpitch4;
      }
    }
    else if (vi.IsRGB64()) {
      BYTE *_dstp = dst->GetWritePtr();
      const int dstpitch = dst->GetPitch();

      const int srcpitch8 = src->GetPitch()>>3;
      const int dstpitch8 = dstpitch>>3;
      const int height = src->GetHeight();
      const int rowsize8 = src->GetRowSize()>>3;
      const uint64_t* srcp8 = (const uint64_t*)src->GetReadPtr();
      uint64_t* dstp8 = (uint64_t*)_dstp;

      for (int y=0; y<height; y+=1) {
        for (int i=m, j=0; j<rowsize8; i+=period, j+=1) {
          dstp8[i] = srcp8[j];
        }
        srcp8 += srcpitch8;
        dstp8 += dstpitch8;
      }
    }
  }
  return dst;
}


AVSValue __cdecl WeaveColumns::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  if (args[1].AsInt() == 1)
    return args[0];

  return new WeaveColumns(args[0].AsClip(), args[1].AsInt(), env);
}







/*********************************
 *******   SeparateRows   ******
 *********************************/

SeparateRows::SeparateRows(PClip _child, int _interval, IScriptEnvironment* env)
 : NonCachedGenericVideoFilter(_child), interval(_interval)
{
  if (_interval <= 0)
    env->ThrowError("SeparateRows: interval must be greater than zero.");

  if (_interval > vi.height)
    env->ThrowError("SeparateRows: interval must be less than or equal height.");

  if (vi.height % _interval)
    env->ThrowError("SeparateRows: height must be mod %d.", _interval);

  vi.height /= _interval;
  vi.MulDivFPS(_interval, 1);
  vi.num_frames *= _interval;

  if (vi.num_frames < 0)
    env->ThrowError("SeparateRows: Maximum number of frames exceeded.");

  if (vi.Is420() && vi.height & 1)
    env->ThrowError("SeparateRows: YUV420 output height must be even.");
}


PVideoFrame SeparateRows::GetFrame(int n, IScriptEnvironment* env)
{
  const int m = (vi.IsRGB() && !vi.IsPlanar())? interval-1 - n%interval : n%interval; // RGB upside-down
  const int f = n/interval;

  PVideoFrame frame = child->GetFrame(f, env);

  if (vi.IsPlanar() && !vi.IsY()) {
    int plane0 = vi.IsRGB() ? PLANAR_G : PLANAR_Y;
    int plane1 = vi.IsRGB() ? PLANAR_B : PLANAR_U;
    const int Ypitch   = frame->GetPitch(plane0);
    const int UVpitch  = frame->GetPitch(plane1);
    const int Yoffset  = Ypitch  * m;
    const int UVoffset = UVpitch * m;

    if (vi.NumComponents() == 4) {
      int Aoffset = frame->GetPitch(PLANAR_A) * m;

      return env->SubframePlanarA(frame, Yoffset, Ypitch * interval,
        frame->GetRowSize(plane0), vi.height,
        UVoffset, UVoffset, UVpitch * interval, Aoffset);
    }
    else {
      return env->SubframePlanar(frame, Yoffset, Ypitch * interval,
        frame->GetRowSize(plane0), vi.height,
        UVoffset, UVoffset, UVpitch * interval);
    }
  }
  const int pitch = frame->GetPitch();
  return env->Subframe(frame, pitch * m, pitch * interval, frame->GetRowSize(), vi.height);
}


AVSValue __cdecl SeparateRows::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  if (args[1].AsInt() == 1)
    return args[0];

  return new SeparateRows(args[0].AsClip(), args[1].AsInt(), env);
}







/****************************
 *******   WeaveRows   ******
 ****************************/

WeaveRows::WeaveRows(PClip _child, int _period, IScriptEnvironment* env)
 : GenericVideoFilter(_child), period(_period), inframes(vi.num_frames)
{
  if (_period <= 0)
    env->ThrowError("WeaveRows: period must be greater than zero.");

  vi.height *= _period;
  vi.MulDivFPS(1, _period);
  vi.num_frames += _period-1; // Ceil!
  vi.num_frames /= _period;
}


PVideoFrame WeaveRows::GetFrame(int n, IScriptEnvironment* env)
{
  const int b = n * period;
  const int e = b + period;

  PVideoFrame dst = env->NewVideoFrame(vi);
  BYTE *dstp = dst->GetWritePtr();
  const int dstpitch = dst->GetPitch();

  if (vi.IsRGB() && !vi.IsPlanar()) { // RGB upsidedown
    dstp += dstpitch * period;
    for (int i=b; i<e; i++) {
      dstp -= dstpitch;
      const int j = i < inframes ? i : inframes-1;
      PVideoFrame src = child->GetFrame(j, env);
      if(i==b) // very first
        env->copyFrameProps(src, dst);

      env->BitBlt( dstp,              dstpitch * period,
              src->GetReadPtr(), src->GetPitch(),
              src->GetRowSize(), src->GetHeight() );
    }
  }
  else {
    int planes_y[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
    int planes_r[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
    int *planes = (vi.IsYUV() || vi.IsYUVA()) ? planes_y : planes_r;
    bool isYUY2 = vi.IsYUY2();
    int dstpitch[4];
    BYTE *dstp[4];
    for (int p = 0; p < (isYUY2 ? 1 : vi.NumComponents()); ++p) {
      int plane = planes[p];
      dstpitch[p] = dst->GetPitch(plane);
      dstp[p] = dst->GetWritePtr(plane);
    }

    for (int i=b; i<e; i++) {
      const int j = i < inframes ? i : inframes-1;
      PVideoFrame src = child->GetFrame(j, env);
      if (i == b)  // very first
        env->copyFrameProps(src, dst);
      for (int p = 0; p < (isYUY2 ? 1 : vi.NumComponents()); ++p) {
        int plane = planes[p];
        env->BitBlt(dstp[p], dstpitch[p] * period,
          src->GetReadPtr(plane), src->GetPitch(plane),
          src->GetRowSize(plane), src->GetHeight(plane) );
        dstp[p] += dstpitch[p];
      }
    }
  }
  return dst;
}


AVSValue __cdecl WeaveRows::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  if (args[1].AsInt() == 1)
    return args[0];

  return new WeaveRows(args[0].AsClip(), args[1].AsInt(), env);
}







/*********************************
 *******   SeparateFields   ******
 *********************************/

SeparateFields::SeparateFields(PClip _child, IScriptEnvironment* env)
 : NonCachedGenericVideoFilter(_child)
{
  if (vi.height & 1)
    env->ThrowError("SeparateFields: height must be even");
  if (vi.Is420() && vi.height & 3)
    env->ThrowError("SeparateFields: YUV420 height must be multiple of 4");
  vi.height >>= 1;
  vi.MulDivFPS(2, 1);
  vi.num_frames *= 2;

  if (vi.num_frames < 0)
    env->ThrowError("SeparateFields: Maximum number of frames exceeded.");

  vi.SetFieldBased(true);
}


PVideoFrame SeparateFields::GetFrame(int n, IScriptEnvironment* env)
{
  PVideoFrame frame = child->GetFrame(n>>1, env);
  if (vi.IsPlanar()) {
    const bool topfield = GetParity(n);

    int plane0 = vi.IsRGB() ? PLANAR_G : PLANAR_Y;
    int plane1 = vi.IsRGB() ? PLANAR_B : PLANAR_U;
    const int Ypitch   = frame->GetPitch(plane0);
    const int UVpitch  = frame->GetPitch(plane1);
    const int UVoffset = !topfield ? UVpitch : 0;
    const int Yoffset = !topfield ? Ypitch : 0;

    if (vi.NumComponents() == 4) {
      int Aoffset = !topfield ? frame->GetPitch(PLANAR_A) : 0;

      return env->SubframePlanarA(frame, Yoffset, frame->GetPitch() * 2, frame->GetRowSize(), frame->GetHeight() >> 1,
        UVoffset, UVoffset, frame->GetPitch(PLANAR_U) * 2, Aoffset);
    }
    else {
      return env->SubframePlanar(frame, Yoffset, frame->GetPitch() * 2, frame->GetRowSize(), frame->GetHeight() >> 1,
        UVoffset, UVoffset, frame->GetPitch(PLANAR_U) * 2);
    }
  }
  return env->Subframe(frame,(GetParity(n) ^ vi.IsYUY2()) ? frame->GetPitch() : 0,
                         frame->GetPitch()*2, frame->GetRowSize(), frame->GetHeight()>>1);
}


AVSValue __cdecl SeparateFields::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  PClip clip = args[0].AsClip();
  if (clip->GetVideoInfo().IsFieldBased())
    env->ThrowError("SeparateFields: SeparateFields should be applied on frame-based material: use AssumeFrameBased() beforehand");

  return new SeparateFields(clip, env);
}







/******************************
 *******   Interleave   *******
 ******************************/

Interleave::Interleave(int _num_children, const PClip* _child_array, IScriptEnvironment* env)
  : num_children(_num_children), child_array(_child_array)
{
  vi = child_array[0]->GetVideoInfo();
  vi.MulDivFPS(num_children, 1);
  vi.num_frames = (vi.num_frames - 1) * num_children + 1;
  child_devs = GetDeviceTypes(child_array[0]);
  for (int i=1; i<num_children; ++i)
  {
    const VideoInfo& vi2 = child_array[i]->GetVideoInfo();
    if (vi.width != vi2.width || vi.height != vi2.height)
      env->ThrowError("Interleave: videos must be of the same size.");
    if (!vi.IsSameColorspace(vi2))
      env->ThrowError("Interleave: video formats don't match");

    vi.num_frames = max(vi.num_frames, (vi2.num_frames - 1) * num_children + i + 1);

    child_devs &= GetDeviceTypes(child_array[i]);
    if (child_devs == 0)
      env->ThrowError("Interleave: device types don't match");
  }
  if (vi.num_frames < 0)
    env->ThrowError("Interleave: Maximum number of frames exceeded.");

}

int __stdcall Interleave::SetCacheHints(int cachehints,int frame_range)
{
  AVS_UNUSED(frame_range);
  switch (cachehints)
  {
  case CACHE_DONT_CACHE_ME:
    return 1;
  case CACHE_GET_MTMODE:
    return MT_NICE_FILTER;
  case CACHE_GET_DEV_TYPE:
    return child_devs;
  default:
    return 0;
  }
}

AVSValue __cdecl Interleave::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  args = args[0];
  const int num_args = args.ArraySize();
  if (num_args == 1)
    return args[0];
  PClip* child_array = new PClip[num_args];
  for (int i=0; i<num_args; ++i)
    child_array[i] = args[i].AsClip();
  return new Interleave(num_args, child_array, env);
}






/*********************************
 ********   SelectEvery    *******
 *********************************/


SelectEvery::SelectEvery(PClip _child, int _every, int _from, IScriptEnvironment* env)
: NonCachedGenericVideoFilter(_child), every(_every), from(_from)
{
  if (_every == 0)
    env->ThrowError("Parameter 'every' of SelectEvery cannot be zero.");

  vi.MulDivFPS(1, every);
  vi.num_frames = (vi.num_frames-1-from) / every + 1;
}


AVSValue __cdecl SelectEvery::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  const int num_vals = args[2].ArraySize();
  if (num_vals <= 1)
	return new SelectEvery(args[0].AsClip(), args[1].AsInt(), num_vals>0 ? args[2][0].AsInt() : 0, env);
  else {
    PClip* child_array = new PClip[num_vals];
    for (int i=0; i<num_vals; ++i)
      child_array[i] = new SelectEvery(args[0].AsClip(), args[1].AsInt(), args[2][i].AsInt(), env);
    return new Interleave(num_vals, child_array, env);
  }
}








/**************************************
 ********   DoubleWeaveFields   *******
 *************************************/

DoubleWeaveFields::DoubleWeaveFields(PClip _child)
  : GenericVideoFilter(_child)
{
  vi.height *= 2;
  vi.SetFieldBased(false);
}


void copy_field(const PVideoFrame& dst, const PVideoFrame& src, bool yuv, bool planarRGB, bool parity, IScriptEnvironment* env)
{
  bool noTopBottom = yuv || planarRGB;

  int plane1 = planarRGB ? PLANAR_B : PLANAR_U;
  int plane2 = planarRGB ? PLANAR_R : PLANAR_V;

  const int add_pitch = dst->GetPitch() * (parity ^ noTopBottom);
  const int add_pitchUV = dst->GetPitch(plane1) * (parity ^ noTopBottom);
  const int add_pitchA = dst->GetPitch(PLANAR_A) * (parity ^ noTopBottom);

  env->BitBlt(dst->GetWritePtr()         + add_pitch, dst->GetPitch()*2,
    src->GetReadPtr(), src->GetPitch(),
    src->GetRowSize(), src->GetHeight());

  env->BitBlt(dst->GetWritePtr(plane1) + add_pitchUV, dst->GetPitch(plane1)*2,
    src->GetReadPtr(plane1), src->GetPitch(plane1),
    src->GetRowSize(plane1), src->GetHeight(plane1));

  env->BitBlt(dst->GetWritePtr(plane2) + add_pitchUV, dst->GetPitch(plane2)*2,
    src->GetReadPtr(plane2), src->GetPitch(plane2),
    src->GetRowSize(plane2), src->GetHeight(plane2));

  env->BitBlt(dst->GetWritePtr(PLANAR_A) + add_pitchA, dst->GetPitch(PLANAR_A)*2,
    src->GetReadPtr(PLANAR_A), src->GetPitch(PLANAR_A),
    src->GetRowSize(PLANAR_A), src->GetHeight(PLANAR_A));
}


PVideoFrame DoubleWeaveFields::GetFrame(int n, IScriptEnvironment* env)
{
  PVideoFrame a = child->GetFrame(n, env);
  PVideoFrame b = child->GetFrame(n+1, env);

  PVideoFrame result = env->NewVideoFrameP(vi, &a);

  const bool parity = child->GetParity(n);

  copy_field(result, a, vi.IsYUV() || vi.IsYUVA(), vi.IsPlanarRGB() || vi.IsPlanarRGBA(), parity, env);
  copy_field(result, b, vi.IsYUV() || vi.IsYUVA(), vi.IsPlanarRGB() || vi.IsPlanarRGBA(), !parity, env);

  return result;
}



/**************************************
 ********   DoubleWeaveFrames   *******
 *************************************/

DoubleWeaveFrames::DoubleWeaveFrames(PClip _child)
  : GenericVideoFilter(_child)
{
  vi.num_frames *= 2;
  if (vi.num_frames < 0)
    vi.num_frames = 0x7FFFFFFF; // MAXINT

  vi.MulDivFPS(2, 1);
}

void copy_alternate_lines(const PVideoFrame& dst, const PVideoFrame& src, bool yuv, bool planarRGB, bool parity, IScriptEnvironment* env)
{
  bool noTopBottom = yuv || planarRGB;

  int plane1 = planarRGB ? PLANAR_B : PLANAR_U;
  int plane2 = planarRGB ? PLANAR_R : PLANAR_V;

  const int src_add_pitch = src->GetPitch()         * (parity ^ noTopBottom);
  const int src_add_pitchUV = src->GetPitch(plane1) * (parity ^ noTopBottom);
  const int src_add_pitchA = src->GetPitch(PLANAR_A) * (parity ^ noTopBottom);

  const int dst_add_pitch = dst->GetPitch()         * (parity ^ noTopBottom);
  const int dst_add_pitchUV = dst->GetPitch(plane1) * (parity ^ noTopBottom);
  const int dst_add_pitchA = dst->GetPitch(PLANAR_A) * (parity ^ noTopBottom);

  env->BitBlt(dst->GetWritePtr()         + dst_add_pitch, dst->GetPitch()*2,
    src->GetReadPtr()          + src_add_pitch, src->GetPitch()*2,
    src->GetRowSize(), src->GetHeight()>>1);

  env->BitBlt(dst->GetWritePtr(plane1) + dst_add_pitchUV, dst->GetPitch(plane1)*2,
    src->GetReadPtr(plane1)  + src_add_pitchUV, src->GetPitch(plane1)*2,
    src->GetRowSize(plane1), src->GetHeight(plane1)>>1);

  env->BitBlt(dst->GetWritePtr(plane2) + dst_add_pitchUV, dst->GetPitch(plane2)*2,
    src->GetReadPtr(plane2)  + src_add_pitchUV, src->GetPitch(plane2)*2,
    src->GetRowSize(plane2), src->GetHeight(plane2)>>1);

  env->BitBlt(dst->GetWritePtr(PLANAR_A) + dst_add_pitchA, dst->GetPitch(PLANAR_A)*2,
    src->GetReadPtr(PLANAR_A)  + src_add_pitchA, src->GetPitch(PLANAR_A)*2,
    src->GetRowSize(PLANAR_A), src->GetHeight(PLANAR_A)>>1);
}


PVideoFrame DoubleWeaveFrames::GetFrame(int n, IScriptEnvironment* env)
{
  if (!(n&1))
  {
    return child->GetFrame(n>>1, env);
  }
  else {
    PVideoFrame a = child->GetFrame(n>>1, env);
    PVideoFrame b = child->GetFrame((n+1)>>1, env);
    bool parity = this->GetParity(n);

    if (a->IsWritable()) {
      copy_alternate_lines(a, b,  vi.IsYUV() || vi.IsYUVA(), vi.IsPlanarRGB() || vi.IsPlanarRGBA(), !parity, env);
      return a;
    }
    else if (b->IsWritable()) {
      copy_alternate_lines(b, a,  vi.IsYUV() || vi.IsYUVA(), vi.IsPlanarRGB() || vi.IsPlanarRGBA(), parity, env);
      return b;
    }
    else {
      PVideoFrame result = env->NewVideoFrameP(vi, &a);
      copy_alternate_lines(result, a, vi.IsYUV() || vi.IsYUVA(), vi.IsPlanarRGB() || vi.IsPlanarRGBA(), parity, env);
      copy_alternate_lines(result, b, vi.IsYUV() || vi.IsYUVA(), vi.IsPlanarRGB() || vi.IsPlanarRGBA(), !parity, env);
      return result;
    }
  }
}





/*******************************
 ********   Bob Filter   *******
 *******************************/

Fieldwise::Fieldwise(PClip _child1, PClip _child2)
: NonCachedGenericVideoFilter(_child1), child2(_child2)
  { vi.SetFieldBased(false); } // Make FrameBased, leave IT_BFF and IT_TFF alone


PVideoFrame __stdcall Fieldwise::GetFrame(int n, IScriptEnvironment* env)
{
  return (child->GetParity(n) ? child2 : child)->GetFrame(n, env);
}


bool __stdcall Fieldwise::GetParity(int n)
{
  return child->GetParity(n) ^ (n&1); // ^ = XOR
}







/************************************
 ********   Factory Methods   *******
 ***********************************/

static AVSValue __cdecl Create_DoubleWeave(AVSValue args, void*, IScriptEnvironment* env)
{
  AVS_UNUSED(env);
  PClip clip = args[0].AsClip();
  if (clip->GetVideoInfo().IsFieldBased())
    return new DoubleWeaveFields(clip);
  else
    return new DoubleWeaveFrames(clip);
}


static AVSValue __cdecl Create_Weave(AVSValue args, void*, IScriptEnvironment* env)
{
  PClip clip = args[0].AsClip();
  if (!clip->GetVideoInfo().IsFieldBased())
    env->ThrowError("Weave: Weave should be applied on field-based material: use AssumeFieldBased() beforehand");
  return new SelectEvery(Create_DoubleWeave(args, 0, env).AsClip(), 2, 0, env);
}


static AVSValue __cdecl Create_Pulldown(AVSValue args, void*, IScriptEnvironment* env)
{
  PClip clip = args[0].AsClip();
  PClip* child_array = new PClip[2];
  child_array[0] = new SelectEvery(clip, 5, args[1].AsInt() % 5, env);
  child_array[1] = new SelectEvery(clip, 5, args[2].AsInt() % 5, env);
  return new AssumeFrameBased(new Interleave(2, child_array, env));
}


static AVSValue __cdecl Create_SwapFields(AVSValue args, void*, IScriptEnvironment* env)
{
  return new SelectEvery(new DoubleWeaveFields(new ComplementParity(
	  new SeparateFields(args[0].AsClip(), env))), 2, 0, env);
}


static AVSValue __cdecl Create_Bob(AVSValue args, void*, IScriptEnvironment* env)
{
  PClip clip = args[0].AsClip();
  if (!clip->GetVideoInfo().IsFieldBased())
    clip = new SeparateFields(clip, env);

  const VideoInfo& vi = clip->GetVideoInfo();

  const double b = args[1].AsDblDef(1./3.);
  const double c = args[2].AsDblDef(1./3.);
  const int new_height = args[3].AsInt(vi.height*2);
  MitchellNetravaliFilter filter(b, c);
  return new Fieldwise(new FilteredResizeV(clip, -0.25, vi.height,
                                           new_height, &filter, env),
                       new FilteredResizeV(clip, +0.25, vi.height,
                                           new_height, &filter, env));
}


SelectRangeEvery::SelectRangeEvery(PClip _child, int _every, int _length, int _offset, bool _audio, IScriptEnvironment* env)
: NonCachedGenericVideoFilter(_child), audio(_audio), achild(_child)
{
  const int64_t num_audio_samples = vi.num_audio_samples;

  AVSValue trimargs[3] = { _child, _offset, 0};
  PClip c = env->Invoke("Trim",AVSValue(trimargs,3)).AsClip();
  child = c;
  vi = c->GetVideoInfo();

  every = clamp(_every,1,vi.num_frames);
  length = clamp(_length,1,every);

  const int n = vi.num_frames;
  vi.num_frames = (n/every)*length+(n%every<length?n%every:length);

  if (audio && vi.HasAudio()) {
    vi.num_audio_samples = vi.AudioSamplesFromFrames(vi.num_frames);
  } else {
    vi.num_audio_samples = num_audio_samples; // Undo Trim's work!
  }
}


PVideoFrame __stdcall SelectRangeEvery::GetFrame(int n, IScriptEnvironment* env)
{
  return child->GetFrame((n/length)*every+(n%length), env);
}


bool __stdcall SelectRangeEvery::GetParity(int n)
{
  return child->GetParity((n/length)*every+(n%length));
}


void __stdcall SelectRangeEvery::GetAudio(void* buf, int64_t start, int64_t count, IScriptEnvironment* env)
{
  if (!audio) {
	// Use original unTrim'd child
    achild->GetAudio(buf, start, count, env);
    return;
  }

  int64_t samples_filled = 0;
  BYTE* samples = (BYTE*)buf;
  const int bps = vi.BytesPerAudioSample();
  int startframe = vi.FramesFromAudioSamples(start);
  int64_t general_offset = start - vi.AudioSamplesFromFrames(startframe);  // General compensation for startframe rounding.

  while (samples_filled < count) {
    const int iteration = startframe / length;                    // Which iteration is this.
    const int iteration_into = startframe % length;               // How far, in frames are we into this iteration.
    const int iteration_left = length - iteration_into;           // How many frames is left of this iteration.

    const int64_t iteration_left_samples = vi.AudioSamplesFromFrames(iteration_left);
    // This is the number of samples we can get without either having to skip, or being finished.
    const int64_t getsamples = min(iteration_left_samples, count-samples_filled);
    const int64_t start_offset = vi.AudioSamplesFromFrames(iteration * every + iteration_into) + general_offset;

    child->GetAudio(&samples[samples_filled*bps], start_offset, getsamples, env);
    samples_filled += getsamples;
    startframe = (iteration+1) * length;
    general_offset = 0; // On the following loops, general offset should be 0, as we are either skipping.
  }
}

AVSValue __cdecl SelectRangeEvery::Create(AVSValue args, void* user_data, IScriptEnvironment* env) {
  AVS_UNUSED(user_data);
  return new SelectRangeEvery(args[0].AsClip(), args[1].AsInt(1500), args[2].AsInt(50), args[3].AsInt(0), args[4].AsBool(true), env);
}
