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

#ifndef __Limiter_H__
#define __Limiter_H__

#include <avisynth.h>

class Limiter : public GenericVideoFilter
{
public:
    Limiter(PClip _child, float _min_luma, float _max_luma, float _min_chroma, float _max_chroma, int _show, bool paramscale, IScriptEnvironment* env);
    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;

    int __stdcall SetCacheHints(int cachehints, int frame_range) override {
      AVS_UNUSED(frame_range);
      return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
    }

    static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env);
private:

  int max_luma;
  int min_luma;
  int max_chroma;
  int min_chroma;

  float max_luma_f;
  float min_luma_f;
  float max_chroma_f;
  float min_chroma_f;

  enum show_e{show_none, show_luma, show_luma_grey, show_chroma, show_chroma_grey};
  const show_e show;

  // avs+
  int pixelsize;
  int bits_per_pixel; // 8,10..16
};

#endif  // __Limiter_H__

