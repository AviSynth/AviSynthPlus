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

#ifndef __Overlay_funcs_h
#define __Overlay_funcs_h

#include <avisynth.h>
#include <avs/minmax.h>
#include "imghelpers.h"
#include "blend_common.h"

enum {
  OF_Blend = 0,
  OF_Add,
  OF_Subtract,
  OF_Multiply,
  OF_Chroma,
  OF_Luma,
  OF_Lighten,
  OF_Darken,
  OF_SoftLight,
  OF_HardLight,
  OF_Difference,
  OF_Exclusion,
  OF_Blend_Compat
};

class OverlayFunction {
public:
  OverlayFunction() {
  }
  virtual ~OverlayFunction() {};
  void setOpacity(int _opacity, float _opacity_f) { opacity = clamp(_opacity, 0, 256); inv_opacity = 256 - opacity; opacity_f = _opacity_f; inv_opacity_f = 1.0f - _opacity_f; }
  void setEnv(IScriptEnvironment *_env) { env = _env;}
  void setBitsPerPixel(int _bits_per_pixel) { bits_per_pixel = _bits_per_pixel; }
  void setMode(int _of_mode) { of_mode = _of_mode; }
  void setColorSpaceInfo(bool _rgb, bool _greyscale) { rgb = _rgb, greyscale = _greyscale; }

  virtual void DoBlendImage(ImageOverlayInternal* base, ImageOverlayInternal* overlay) = 0;
  virtual void DoBlendImageMask(ImageOverlayInternal* base, ImageOverlayInternal* overlay, ImageOverlayInternal* mask) = 0;
protected:
  int opacity;
  int inv_opacity;
  float opacity_f;
  float inv_opacity_f;
  int bits_per_pixel;
  int of_mode; // add/subtract, etc
  bool rgb; // add/subtract... overshoot mode is different
  bool greyscale; // having only one plane
  IScriptEnvironment *env;
};

class OL_BlendImage : public OverlayFunction {
  void DoBlendImage(ImageOverlayInternal* base, ImageOverlayInternal* overlay);
  void DoBlendImageMask(ImageOverlayInternal* base, ImageOverlayInternal* overlay, ImageOverlayInternal* mask);
  template<typename pixel_t>
  void BlendImage(ImageOverlayInternal* base, ImageOverlayInternal* overlay);
  template<typename pixel_t>
  void BlendImageMask(ImageOverlayInternal* base, ImageOverlayInternal* overlay, ImageOverlayInternal* mask);
private:
};

// common add/subtract
class OL_AddImage : public OverlayFunction {
  void DoBlendImage(ImageOverlayInternal* base, ImageOverlayInternal* overlay);
  void DoBlendImageMask(ImageOverlayInternal* base, ImageOverlayInternal* overlay, ImageOverlayInternal* mask);
  //template<typename pixel_t>
  //void BlendImage(Image444* base, Image444* overlay);
  template<typename pixel_t, bool maskMode, bool of_add>
  void BlendImageMask(ImageOverlayInternal* base, ImageOverlayInternal* overlay, ImageOverlayInternal* mask);
};

#if 0
// common with Add
class OL_SubtractImage : public OverlayFunction {
  void DoBlendImage(ImageOverlayInternal* base, ImageOverlayInternal* overlay);
  void DoBlendImageMask(ImageOverlayInternal* base, ImageOverlayInternal* overlay, ImageOverlayInternal* mask);
  template<typename pixel_t>
  void BlendImage(ImageOverlayInternal* base, ImageOverlayInternal* overlay);
  template<typename pixel_t>
  void BlendImageMask(ImageOverlayInternal* base, ImageOverlayInternal* overlay, ImageOverlayInternal* mask);
};
#endif

class OL_MultiplyImage : public OverlayFunction {
  void DoBlendImage(ImageOverlayInternal* base, ImageOverlayInternal* overlay);
  void DoBlendImageMask(ImageOverlayInternal* base, ImageOverlayInternal* overlay, ImageOverlayInternal* mask);
  template<typename pixel_t>
  void BlendImage(ImageOverlayInternal* base, ImageOverlayInternal* overlay);
  template<typename pixel_t>
  void BlendImageMask(ImageOverlayInternal* base, ImageOverlayInternal* overlay, ImageOverlayInternal* mask);
};

#if 0
// made common with OF_Blend
class OL_BlendLumaImage : public OverlayFunction {
  void DoBlendImage(ImageOverlayInternal* base, ImageOverlayInternal* overlay);
  void DoBlendImageMask(ImageOverlayInternal* base, ImageOverlayInternal* overlay, ImageOverlayInternal* mask);
  template<typename pixel_t>
  void BlendImage(ImageOverlayInternal* base, ImageOverlayInternal* overlay);
  template<typename pixel_t>
  void BlendImageMask(ImageOverlayInternal* base, ImageOverlayInternal* overlay, ImageOverlayInternal* mask);
private:
};
#endif

#if 0
// made common with OF_Blend
class OL_BlendChromaImage : public OverlayFunction {
  void DoBlendImage(ImageOverlayInternal* base, ImageOverlayInternal* overlay);
  void DoBlendImageMask(ImageOverlayInternal* base, ImageOverlayInternal* overlay, ImageOverlayInternal* mask);
  template<typename pixel_t>
  void BlendImage(ImageOverlayInternal* base, ImageOverlayInternal* overlay);
  template<typename pixel_t>
  void BlendImageMask(ImageOverlayInternal* base, ImageOverlayInternal* overlay, ImageOverlayInternal* mask);
private:
};
#endif

#if 0
class OL_LightenImage : public OverlayFunction {
  void DoBlendImage(ImageOverlayInternal* base, ImageOverlayInternal* overlay);
  void DoBlendImageMask(ImageOverlayInternal* base, ImageOverlayInternal* overlay, ImageOverlayInternal* mask);
  template<typename pixel_t>
  void BlendImage(ImageOverlayInternal* base, ImageOverlayInternal* overlay);
  template<typename pixel_t>
  void BlendImageMask(ImageOverlayInternal* base, ImageOverlayInternal* overlay, ImageOverlayInternal* mask);
};
#endif

// common darken/lighten
class OL_DarkenImage : public OverlayFunction {
  void DoBlendImage(ImageOverlayInternal* base, ImageOverlayInternal* overlay);
  void DoBlendImageMask(ImageOverlayInternal* base, ImageOverlayInternal* overlay, ImageOverlayInternal* mask);
  //template<typename pixel_t>
  //void BlendImage(Image444* base, Image444* overlay);
  template<typename pixel_t, bool maskMode, bool of_darken>
  void BlendImageMask(ImageOverlayInternal* base, ImageOverlayInternal* overlay, ImageOverlayInternal* mask);
};

class OL_SoftLightImage : public OverlayFunction {
  void DoBlendImage(ImageOverlayInternal* base, ImageOverlayInternal* overlay);
  void DoBlendImageMask(ImageOverlayInternal* base, ImageOverlayInternal* overlay, ImageOverlayInternal* mask);
  template<typename pixel_t>
  void BlendImage(ImageOverlayInternal* base, ImageOverlayInternal* overlay);
  template<typename pixel_t, bool maskMode, bool hardLight>
  void BlendImageMask(ImageOverlayInternal* base, ImageOverlayInternal* overlay, ImageOverlayInternal* mask);
};

#if 0
// common with OL_HardLightImage
class OL_HardLightImage : public OverlayFunction {
  void DoBlendImage(ImageOverlayInternal* base, ImageOverlayInternal* overlay);
  void DoBlendImageMask(ImageOverlayInternal* base, ImageOverlayInternal* overlay, ImageOverlayInternal* mask);
  template<typename pixel_t>
  void BlendImage(ImageOverlayInternal* base, ImageOverlayInternal* overlay);
  template<typename pixel_t, bool maskMode>
  void BlendImageMask(ImageOverlayInternal* base, ImageOverlayInternal* overlay, ImageOverlayInternal* mask);
};
#endif

class OL_DifferenceImage : public OverlayFunction {
  void DoBlendImage(ImageOverlayInternal* base, ImageOverlayInternal* overlay);
  void DoBlendImageMask(ImageOverlayInternal* base, ImageOverlayInternal* overlay, ImageOverlayInternal* mask);
  //template<typename pixel_t>
  //void BlendImage(Image444* base, Image444* overlay);
  template<typename pixel_t, bool maskMode>
  void BlendImageMask(ImageOverlayInternal* base, ImageOverlayInternal* overlay, ImageOverlayInternal* mask);
};

class OL_ExclusionImage : public OverlayFunction {
  void DoBlendImage(ImageOverlayInternal* base, ImageOverlayInternal* overlay);
  void DoBlendImageMask(ImageOverlayInternal* base, ImageOverlayInternal* overlay, ImageOverlayInternal* mask);
  // template<typename pixel_t>
  // void BlendImage(Image444* base, Image444* overlay);
  template<typename pixel_t, bool maskMode>
  void BlendImageMask(ImageOverlayInternal* base, ImageOverlayInternal* overlay, ImageOverlayInternal* mask);
};

#endif  // Overlay_funcs_h