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


#include "debug.h"

#ifdef AVS_WINDOWS
    #include <avs/win.h>
#else
    #include <avs/posix.h>
#endif

#include "../core/bitblt.h"
#include "../core/internal.h"



/**********************************************
 *******  PlanarLegacyAlignment Filter  *******
 *******  for pitch challenged plugins  *******
 **********************************************/

class PlanarLegacyAlignment : public GenericVideoFilter
{
private:
  const IScriptEnvironment::PlanarChromaAlignmentMode mode;

public:
  PlanarLegacyAlignment( PClip _child, const bool _mode, IScriptEnvironment* env )
    : GenericVideoFilter(_child),
    mode(_mode ?  IScriptEnvironment::PlanarChromaAlignmentOff  // Legacy PLANAR_Y alignment
            : IScriptEnvironment::PlanarChromaAlignmentOn)  // New PLANAR_UV priority alignment
  {
    AVS_UNUSED(env);
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env)
  {
  const IScriptEnvironment::PlanarChromaAlignmentMode
    oldmode = env->PlanarChromaAlignment(mode)            // Set the PLANAR alignement mode
            ? IScriptEnvironment::PlanarChromaAlignmentOn
            : IScriptEnvironment::PlanarChromaAlignmentOff;

  PVideoFrame src = child->GetFrame(n, env);              // run the GetFrame chain

  env->PlanarChromaAlignment(oldmode);                    // reset the PLANAR alignement mode

  return src;
  }

  static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env)
  {
  return new PlanarLegacyAlignment(args[0].AsClip(), args[1].AsBool(), env);
  }

};



/***********************************************
 *******  Echo Filter  *******
 *******  Call GetFrame on all source clips*****
 ***********************************************/

class Echo : public GenericVideoFilter
{
private:
  const int ClipCount;
  PClip *clips;

public:
  Echo( PClip _child, const AVSValue _clips, IScriptEnvironment* env )
     : GenericVideoFilter(_child), ClipCount(_clips.ArraySize()) {
    AVS_UNUSED(env);

    clips = new PClip[ClipCount];
    for (int i=0; i < ClipCount; i++)
      clips[i] = _clips[i].AsClip();
  }

  ~Echo() {
    delete[] clips;
    clips = 0;
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) {

  PVideoFrame src = child->GetFrame(n, env);

    for (int i=0; i < ClipCount; i++)
      clips[i]->GetFrame(n, env);           // run the GetFrame chains

    return src;
  }

  static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env) {
    return new Echo(args[0].AsClip(), args[1], env);
  }

};



/***********************************************
 ******  Preroll GetFrame/GetAudio Filter ******
 ***********************************************/

class Preroll : public GenericVideoFilter
{
private:
  const int videopr;
  const int64_t audiopr;

  int videonext;
  int64_t audionext;

public:
  Preroll( PClip _child, const int _videopr, const double _audiopr, IScriptEnvironment* env )
    : GenericVideoFilter(_child),
      videopr(_videopr),
      audiopr(int64_t(_audiopr*vi.audio_samples_per_second+0.5)),
      videonext(0),
      audionext(0) {

    AVS_UNUSED(env);
    // Force source filter to honour preroll
    child->SetCacheHints(CACHE_NOTHING, 0);       // Disable Video cache
    child->SetCacheHints(CACHE_AUDIO_NOTHING, 0); // Disable Audio cache
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) {
    if (n != videonext) {
      int i = n - videopr;
      if (i < 0) i = 0;

      // Optimise if n is within pr frames of last
      if (i < videonext && n > videonext) i = videonext;

      while (i < n) {
        child->GetFrame(i, env);
        i += 1;
      }
    }

    videonext = n+1;
    return child->GetFrame(n, env);
  }

  void __stdcall GetAudio(void* buf, int64_t start, int64_t count, IScriptEnvironment* env) {
    if (start != audionext) {
      int64_t s = start - audiopr;
      if (s < 0) s = 0;

      // Optimise if start is within pr sample of last
      if (s < audionext && start > audionext) s = audionext;

      while (s < start) {
        int64_t c = start - s;
        if (c > count) c = count;

        child->GetAudio(buf, s, c, env);
        s += c;
      }
    }

    audionext = start+count;
    child->GetAudio(buf, start, count, env);
  }

  static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env) {
    check_float_params(args, "Preroll", { { 2, "audio" } }, env);
    return new Preroll(args[0].AsClip(), args[1].AsInt(0), (float)args[2].AsDblDef(0.0), env);
  }

};



/********************************************************************
***** Declare index of new filters for Avisynth's filter engine *****
********************************************************************/

extern const AVSFunction Debug_filters[] = {
  { "Null", BUILTIN_FUNC_PREFIX, "c[copy]s", Null::Create },     // clip, copy
  { "SetPlanarLegacyAlignment", BUILTIN_FUNC_PREFIX, "cb", PlanarLegacyAlignment::Create },     // clip, legacy alignment
  { "Echo", BUILTIN_FUNC_PREFIX, "cc+", Echo::Create },
  { "Preroll", BUILTIN_FUNC_PREFIX, "c[video]i[audio]f", Preroll::Create },
  { NULL }
};



/*******************************
 *******   Null Filter   *******
 *******  for debugging  *******
 ******************************/

Null::Null(PClip _child, const char * _copy, IScriptEnvironment* env)
  : GenericVideoFilter(_child), copy(_copy)
{
  AVS_UNUSED(env);
}

Null::~Null()
{
}


PVideoFrame __stdcall Null::GetFrame(int n, IScriptEnvironment* env)
{
  PVideoFrame src = child->GetFrame(n, env);

  BYTE * foo = new BYTE[256];
  BYTE * bar = new BYTE[256];
  MemDebug md;

  md.randomFill(foo, 8, 8, 8);
  env->BitBlt(bar, 8, foo, 8, 8, 8);

  md.reset();
  int i = md.randomCheck(bar, 9, 8, 8);

  if (i)
    env->ThrowError("bug found");

  delete [] foo;
  delete [] bar;

  if (!lstrcmpi(copy, "makewritable"))
  {
    env->MakeWritable(&src);
    return src;
  }

  // TODO: no support for planar formats!
  if (!lstrcmpi(copy, "memcopy"))
  {
    PVideoFrame dst = env->NewVideoFrame(child->GetVideoInfo(), 16);
    if (dst->IsWritable() == false)
      env->ThrowError("new frame not writable"); // honestly don't know whether to expect this condition

    memcpy( dst->GetWritePtr(), src->GetReadPtr(), src->GetPitch() * src->GetHeight() );
    return dst;
  }

  if (!lstrcmpi(copy, "bitblt"))
  {
    PVideoFrame dst = env->NewVideoFrame(child->GetVideoInfo(), 16);
    if (dst->IsWritable() == false)
      env->ThrowError("new frame not writable"); // honestly don't know whether to expect this condition

    env->BitBlt( dst->GetWritePtr(), src->GetPitch(), src->GetReadPtr(), src->GetPitch(),
            src->GetRowSize(), src->GetHeight() );
    return dst;
  }

  //if (!lstrcmpi(copy, "none"))
    // do nothing
  return src;
}

AVSValue __cdecl Null::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  return new Null(args[0].AsClip(), args[1].AsString("none"), env);
}




MemDebug::MemDebug() : mask(0xff)
{
  reset();
}


MemDebug::~MemDebug()
{
}


// fills a buffer with "random" (the same every time) data we can
// test for after some manipulation (e.g. BitBlt)
void MemDebug::randomFill(BYTE* const buf, const int pitch, const int row_size, const int height)
{
  for(int x=0; x<height; ++x)
  {
    for(int y=0; y<row_size; ++y)
    {
      buf[x*pitch + y] = nextNum();
    }
  }
}


// checks a buffer to see if "random" data is intact or copied correctly
// return value == offset of first discrepancy, or 0 if all's well
int MemDebug::randomCheck(BYTE* const buf, const int pitch, const int row_size, const int height)
{
  for(int x=0; x<height; ++x)
  {
    for(int y=0; y<row_size; ++y)
    {
      int n = nextNum();
      if (buf[x*pitch + y] != n) return x*pitch + y;
    }
  }

  return 0;
}


char MemDebug::nextNum()
{
  ++whichByte;
  whichByte &= 3;
  if (whichByte == 0)
  {
    randNum *= 0x41c64e6d;
    randNum += 0x3039;
  }

  return char((randNum & (mask << (whichByte * 8))) >> (whichByte * 8));
}


void MemDebug::reset()
{
  randNum = -591377182;  // pulled out of my ass
  whichByte = 3;
}
