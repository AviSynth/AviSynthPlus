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
#ifndef __Histogram_H__
#define __Histogram_H__

#include <avisynth.h>
#include <vector>
#include "stdint.h"

/********************************************************************
********************************************************************/


class Histogram : public GenericVideoFilter
/**
  * Class to display histogram based on video input
 **/
{
public:
  enum Mode {
    ModeClassic=0,
	ModeLevels,
	ModeColor,
	ModeColor2,
	ModeLuma,
	ModeStereoY8,
	ModeStereo,
	ModeOverlay,
	ModeAudioLevels
  };

  Histogram(PClip _child, Mode _mode, AVSValue _option, int _show_bits, bool _keepsource, bool _markers, IScriptEnvironment* env);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;
  PVideoFrame DrawModeClassic    (int n, IScriptEnvironment* env);
  PVideoFrame DrawModeLevels     (int n, IScriptEnvironment* env);
  PVideoFrame DrawModeColor      (int n, IScriptEnvironment* env);
  PVideoFrame DrawModeColor2     (int n, IScriptEnvironment* env);
  PVideoFrame DrawModeLuma       (int n, IScriptEnvironment* env);
  PVideoFrame DrawModeStereo     (int n, IScriptEnvironment* env);
  PVideoFrame DrawModeOverlay    (int n, IScriptEnvironment* env);
  PVideoFrame DrawModeAudioLevels(int n, IScriptEnvironment* env);

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

  static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env);
private:
  Mode mode;
  int deg15c[24], deg15s[24];
  PClip aud_clip;
  AVSValue option;
  int pixelsize;
  int bits_per_pixel;
  int show_bits; // e.g. levels for 10 bits
  bool keepsource; // return only the Histogram drawing
  bool markers; // paint hazardous YUV area
  int origwidth;
  int origheight;

  int E167;
  std::vector<uint16_t> exptab;
  void ClassicLUTInit();
};


#endif  // __Histogram_H__
