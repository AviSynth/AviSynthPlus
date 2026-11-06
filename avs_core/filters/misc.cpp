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


#include "misc.h"
#include <avs/minmax.h>
#include "../core/bitblt.h"
#include "../core/internal.h"




/********************************************************************
***** Declare index of new filters for Avisynth's filter engine *****
********************************************************************/

extern const AVSFunction Misc_filters[] = {
  { "FixLuminance",  BUILTIN_FUNC_PREFIX, "cif", FixLuminance::Create },    // clip, intercept, slope
  { "PeculiarBlend", BUILTIN_FUNC_PREFIX, "ci", PeculiarBlend::Create },   // clip, cutoff
  { "SkewRows",      BUILTIN_FUNC_PREFIX, "ci", SkewRows::Create },   // clip, skew
  { "FixBrokenChromaUpsampling", BUILTIN_FUNC_PREFIX, "c", FixBrokenChromaUpsampling::Create },
  { NULL }
};







/********************************
 *******   Fix Luminance   ******
 ********************************/

FixLuminance::FixLuminance(PClip _child, int _vertex, int _slope, IScriptEnvironment* env)
 : GenericVideoFilter(_child), vertex(_vertex), slope(_slope)
{
  if (!vi.IsYUY2())
    env->ThrowError("FixLuminance: requires YUY2 input");
}


PVideoFrame FixLuminance::GetFrame(int n, IScriptEnvironment* env)
{
  PVideoFrame frame = child->GetFrame(n, env);
  env->MakeWritable(&frame);
  BYTE* p = frame->GetWritePtr();
  const int pitch = frame->GetPitch();
  for (int y=0; y<=vertex-slope/16; ++y)
  {
    const int subtract = (vertex-y)*16/slope;
    for (int x=0; x<vi.width; ++x)
      p[x*2] = (BYTE)max(0, p[x*2]-subtract);
    p += pitch;
  }
  return frame;
}


AVSValue __cdecl FixLuminance::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  check_float_params(args, "FixLuminance", { { 2, "slope" } }, env);
  return new FixLuminance(args[0].AsClip(), args[1].AsInt(), int(args[2].AsFloat()*16), env);
}






/***********************************************
 *******   Fix Broken Chroma Upsampling   ******
 ***********************************************/

FixBrokenChromaUpsampling::FixBrokenChromaUpsampling(PClip _clip, IScriptEnvironment* env)
  : GenericVideoFilter(_clip)
{
  if (!vi.IsYUY2())
    env->ThrowError("FixBrokenChromaUpsampling: requires YUY2 input");
}


PVideoFrame __stdcall FixBrokenChromaUpsampling::GetFrame(int n, IScriptEnvironment* env)
{
  PVideoFrame frame = child->GetFrame(n, env);
  env->MakeWritable(&frame);
  const int pitch = frame->GetPitch();
  BYTE* p = frame->GetWritePtr() + pitch;
  for (int y = (frame->GetHeight()+1)/4; y > 0; --y) {
    for (int x = 0; x < frame->GetRowSize(); x += 4) {
      BYTE t1 = p[x+1], t3 = p[x+3];
      p[x+1] = p[pitch+x+1]; p[x+3] = p[pitch+x+3];
      p[pitch+x+1] = t1; p[pitch+x+3] = t3;
    }
    p += pitch*4;
  }
  return frame;
}


AVSValue __cdecl FixBrokenChromaUpsampling::Create( AVSValue args, void*,
                                                    IScriptEnvironment* env )
{
  return new FixBrokenChromaUpsampling(args[0].AsClip(), env);
}






/*********************************
 *******   Peculiar Blend   ******
 *********************************/

PeculiarBlend::PeculiarBlend(PClip _child, int _cutoff, IScriptEnvironment* env)
 : GenericVideoFilter(_child), cutoff(_cutoff)
{
  if (!vi.IsYUY2())
    env->ThrowError("PeculiarBlend: requires YUY2 input");
}


PVideoFrame PeculiarBlend::GetFrame(int n, IScriptEnvironment* env) {
  PVideoFrame a = child->GetFrame(n, env);
  PVideoFrame b = child->GetFrame(n+1, env);
  env->MakeWritable(&a);
  BYTE* main = a->GetWritePtr();
  const BYTE* other = b->GetReadPtr();
  const int main_pitch = a->GetPitch();
  const int other_pitch = b->GetPitch();
  const int row_size = a->GetRowSize();

  if (cutoff-31 > 0) {
    int copy_top = min(cutoff-31, vi.height);
    env->BitBlt(main, main_pitch, other, other_pitch, row_size, copy_top);
    main += main_pitch * copy_top;
    other += other_pitch * copy_top;
  }
  for (int y = max(0, cutoff-31); y < min(cutoff, vi.height-1); ++y) {
    int scale = cutoff - y;
    for (int x = 0; x < row_size; ++x)
      main[x] = main[x] + BYTE(((other[x] - main[x]) * scale + 16) >> 5);
    main += main_pitch;
    other += other_pitch;
  }

  return a;
}


AVSValue __cdecl PeculiarBlend::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  return new PeculiarBlend(args[0].AsClip(), args[1].AsInt(), env);
}






/********************************
 *******     SkewRows      ******
 ********************************/

SkewRows::SkewRows(PClip _child, int skew, IScriptEnvironment* env)
 : GenericVideoFilter(_child)
{
  if ((vi.NumComponents() > 1) && vi.IsPlanar())
    env->ThrowError("SkewRows: requires non-planar or greyscale input");

  if (vi.IsYUY2() && skew&1)
    env->ThrowError("SkewRows: For YUY2 skew must be even");

  vi.height *= vi.width;
  vi.width  += skew;
  vi.height += vi.width-1; // Ceiling
  vi.height /= vi.width;
}


PVideoFrame SkewRows::GetFrame(int n, IScriptEnvironment* env) {

  PVideoFrame src = child->GetFrame(n, env);
  PVideoFrame dst = env->NewVideoFrameP(vi, &src);

  const int srowsize = src->GetRowSize();
  const int spitch   = src->GetPitch();
  const BYTE *sptr   = src->GetReadPtr();

  const int drowsize = dst->GetRowSize();
  const int dpitch   = dst->GetPitch();
  BYTE *dptr         = dst->GetWritePtr();

  const int ssize = src->GetHeight()*srowsize;

  int s=0, d=0;
  for (int i=0; i < ssize; i++) {
    if (s >= srowsize) {
      s = 0;
      sptr += spitch;
    }
    if (d >= drowsize) {
      d = 0;
      dptr += dpitch;
    }
    dptr[d++] = sptr[s++];
  }

  while (d < drowsize)
    dptr[d++] = 128;

  return dst;

}


AVSValue __cdecl SkewRows::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  return new SkewRows(args[0].AsClip(), args[1].AsInt(), env);
}
