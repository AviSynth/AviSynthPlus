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

#include <avs/config.h>
#ifdef AVS_WINDOWS
#include <avs/win.h>
#else
#include <avs/posix.h>
#endif

#include "text-overlay.h"
#ifdef INTEL_INTRINSICS
#include "intel/text-overlay_sse.h"
#endif
#include "../convert/convert_matrix.h"  // for RGB2YUV_Rec601

#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <sstream>
#include <cstdint>
#include <cmath>
#include <avs/minmax.h>
#include "../core/internal.h"
#include "../core/info.h"
#include "../core/strings.h"


#if defined(AVS_WINDOWS) && !defined(NO_WIN_GDI)
static HFONT LoadFont(const char name[], int size, bool bold, bool italic, int width=0, int angle=0)
{
  return CreateFont( size, width, angle, angle, bold ? FW_BOLD : FW_NORMAL,
                     italic, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS,
                     CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY, FF_DONTCARE | FIXED_PITCH /*FF_DONTCARE | DEFAULT_PITCH*/, name );
  // avs+: force fixed pitch when font is not found by name
}
#endif

/********************************************************************
***** Declare index of new filters for Avisynth's filter engine *****
********************************************************************/

extern const AVSFunction Text_filters[] = {
  { "ShowFrameNumber",BUILTIN_FUNC_PREFIX,
	"c[scroll]b[offset]i[x]f[y]f[font]s[size]f[text_color]i[halo_color]i[font_width]f[font_angle]f",
	ShowFrameNumber::Create },

  { "ShowCRC32",BUILTIN_FUNC_PREFIX,
        "c[scroll]b[offset]i[x]f[y]f[font]s[size]f[text_color]i[halo_color]i[font_width]f[font_angle]f",
        ShowCRC32::Create },

  { "ShowSMPTE",BUILTIN_FUNC_PREFIX,
	"c[fps]f[offset]s[offset_f]i[x]f[y]f[font]s[size]f[text_color]i[halo_color]i[font_width]f[font_angle]f",
	ShowSMPTE::CreateSMTPE },

  { "ShowTime",BUILTIN_FUNC_PREFIX,
	"c[offset_f]i[x]f[y]f[font]s[size]f[text_color]i[halo_color]i[font_width]f[font_angle]f",
	ShowSMPTE::CreateTime },

  { "Info", BUILTIN_FUNC_PREFIX, "c[font]s[size]f[text_color]i[halo_color]i", FilterInfo::Create },  // clip

  { "Subtitle",BUILTIN_FUNC_PREFIX,
	"cs[x]f[y]f[first_frame]i[last_frame]i[font]s[size]f[text_color]i[halo_color]i"
	"[align]i[spc]i[lsp]i[font_width]f[font_angle]f[interlaced]b[font_filename]s[utf8]b",
#if defined(AVS_WINDOWS) && !defined(NO_WIN_GDI)
    Subtitle::Create
#else
    SimpleText::Create // poor man's SubTitle, it's simulated with SimpleText
#endif
},       // see docs!

  { "Compare",BUILTIN_FUNC_PREFIX,
	"cc[channels]s[logfile]s[show_graph]b",
	Compare::Create },

  { "Text",BUILTIN_FUNC_PREFIX,
  "cs[x]f[y]f[first_frame]i[last_frame]i[font]s[size]f[text_color]i[halo_color]i"
  "[align]i[spc]i[lsp]i[font_width]f[font_angle]f[interlaced]b[font_filename]s[utf8]b[bold]b",
    SimpleText::Create },

  { 0 }
};





#if defined(AVS_WINDOWS) && !defined(NO_WIN_GDI)
/******************************
 *******   Anti-alias    ******
 *****************************/

Antialiaser::Antialiaser(int width, int height, const char fontname[], int size, int _textcolor, int _halocolor, int font_width, int font_angle, bool _interlaced) :
  alpha_calcs(0), w(width), h(height), textcolor(_textcolor), halocolor(_halocolor),
  dirty(true), interlaced(_interlaced)
{
  struct {
    BITMAPINFOHEADER bih;
    RGBQUAD clr[2];
  } b;

  b.bih.biSize                    = sizeof(BITMAPINFOHEADER);
  b.bih.biWidth                   = width * 8 + 32;
  b.bih.biHeight                  = height * 8 + 32;
  b.bih.biBitCount                = 1;
  b.bih.biPlanes                  = 1;
  b.bih.biCompression             = BI_RGB;
  b.bih.biXPelsPerMeter   = 0;
  b.bih.biYPelsPerMeter   = 0;
  b.bih.biClrUsed                 = 2;
  b.bih.biClrImportant    = 2;
  b.clr[0].rgbBlue = b.clr[0].rgbGreen = b.clr[0].rgbRed = 0;
  b.clr[1].rgbBlue = b.clr[1].rgbGreen = b.clr[1].rgbRed = 255;

  hdcAntialias = CreateCompatibleDC(NULL);
  if (hdcAntialias) {
	hbmAntialias = CreateDIBSection
	  ( hdcAntialias,
		(BITMAPINFO *)&b,
		DIB_RGB_COLORS,
		&lpAntialiasBits,
		NULL,
		0 );
	if (hbmAntialias) {
	  hbmDefault = (HBITMAP)SelectObject(hdcAntialias, hbmAntialias);
	  HFONT newfont = LoadFont(fontname, size, true, false, font_width, font_angle);
	  hfontDefault = newfont ? (HFONT)SelectObject(hdcAntialias, newfont) : 0;

	  SetMapMode(hdcAntialias, MM_TEXT);
	  SetTextColor(hdcAntialias, 0xffffff);
	  SetBkColor(hdcAntialias, 0);

	  alpha_calcs = new(std::nothrow) unsigned short[width*height*4];
	  if (!alpha_calcs) FreeDC();
	}
  }
}


Antialiaser::~Antialiaser() {
  FreeDC();
  delete[] alpha_calcs;
}


HDC Antialiaser::GetDC() {
  dirty = true;
  return hdcAntialias;
}


void Antialiaser::FreeDC() {
  if (hdcAntialias) { // :FIXME: Interlocked
    if (hbmDefault) {
	  DeleteObject(SelectObject(hdcAntialias, hbmDefault));
	  hbmDefault = 0;
	}
    if (hfontDefault) {
	  DeleteObject(SelectObject(hdcAntialias, hfontDefault));
	  hfontDefault = 0;
	}
    DeleteDC(hdcAntialias);
    hdcAntialias = 0;
  }
}


void Antialiaser::Apply( const VideoInfo& vi, PVideoFrame* frame, int pitch)
{
  if (!alpha_calcs) return;

  if (vi.IsRGB32() || vi.IsRGB64())
    ApplyRGB32_64((*frame)->GetWritePtr(), pitch, vi.ComponentSize());
  else if (vi.IsRGB24() || vi.IsRGB48())
    ApplyRGB24_48((*frame)->GetWritePtr(), pitch, vi.ComponentSize());
  else if (vi.IsYUY2())
    ApplyYUY2((*frame)->GetWritePtr(), pitch);
  else if (vi.IsYV12()) // YUV420 16/32 bit goes to generic path
    ApplyYV12((*frame)->GetWritePtr(), pitch,
              (*frame)->GetPitch(PLANAR_U),
              (*frame)->GetWritePtr(PLANAR_U),
              (*frame)->GetWritePtr(PLANAR_V) );
  else if (vi.NumComponents() == 1) // Y8, Y16, Y32
    ApplyPlanar((*frame)->GetWritePtr(), pitch, 0, 0, 0, 0, 0, vi.BitsPerComponent(), vi.IsRGB());
  else if (vi.IsPlanar()) {
      if(vi.IsPlanarRGB() || vi.IsPlanarRGBA())
          // internal buffer: Y-R, U-G, V-B
        ApplyPlanar((*frame)->GetWritePtr(PLANAR_R), pitch,
            (*frame)->GetPitch(PLANAR_G),
            (*frame)->GetWritePtr(PLANAR_G),
            (*frame)->GetWritePtr(PLANAR_B),
            vi.GetPlaneWidthSubsampling(PLANAR_G),  // no subsampling
            vi.GetPlaneHeightSubsampling(PLANAR_G),
            vi.BitsPerComponent(), vi.IsRGB() );
      else
        ApplyPlanar((*frame)->GetWritePtr(), pitch,
            (*frame)->GetPitch(PLANAR_U),
            (*frame)->GetWritePtr(PLANAR_U),
            (*frame)->GetWritePtr(PLANAR_V),
            vi.GetPlaneWidthSubsampling(PLANAR_U),
            vi.GetPlaneHeightSubsampling(PLANAR_U),
            vi.BitsPerComponent(), vi.IsRGB());
  }
}


void Antialiaser::ApplyYV12(BYTE* buf, int pitch, int pitchUV, BYTE* bufU, BYTE* bufV) {
  if (dirty) {
    GetAlphaRect();
	xl &= -2; xr |= 1;
	yb &= -2; yt |= 1;
  }
  const int w4 = w*4;
  unsigned short* alpha = alpha_calcs + yb*w4;
  buf  += pitch*yb;
  bufU += (pitchUV*yb)>>1;
  bufV += (pitchUV*yb)>>1;

  for (int y=yb; y<=yt; y+=2) {
    for (int x=xl; x<=xr; x+=2) {
      const int x4 = x<<2;
      const int basealpha00 = alpha[x4+0];
      const int basealpha10 = alpha[x4+4];
      const int basealpha01 = alpha[x4+0+w4];
      const int basealpha11 = alpha[x4+4+w4];
      const int basealphaUV = basealpha00 + basealpha10 + basealpha01 + basealpha11;

      if (basealphaUV != 1024) {
        buf[x+0]       = BYTE((buf[x+0]       * basealpha00 + alpha[x4+3]   ) >> 8);
        buf[x+1]       = BYTE((buf[x+1]       * basealpha10 + alpha[x4+7]   ) >> 8);
        buf[x+0+pitch] = BYTE((buf[x+0+pitch] * basealpha01 + alpha[x4+3+w4]) >> 8);
        buf[x+1+pitch] = BYTE((buf[x+1+pitch] * basealpha11 + alpha[x4+7+w4]) >> 8);

        const int au  = alpha[x4+2] + alpha[x4+6] + alpha[x4+2+w4] + alpha[x4+6+w4];
        bufU[x>>1] = BYTE((bufU[x>>1] * basealphaUV + au) >> 10);

        const int av  = alpha[x4+1] + alpha[x4+5] + alpha[x4+1+w4] + alpha[x4+5+w4];
        bufV[x>>1] = BYTE((bufV[x>>1] * basealphaUV + av) >> 10);
      }
    }
    buf += pitch<<1;
    bufU += pitchUV;
    bufV += pitchUV;
    alpha += w<<3;
  }
}


template<int shiftX, int shiftY, int bits_per_pixel>
void Antialiaser::ApplyPlanar_core(BYTE* buf, int pitch, int pitchUV, BYTE* bufU, BYTE* bufV, bool isRGB)
{
  const int stepX = 1<<shiftX;
  const int stepY = 1<<shiftY;

  if (dirty) {
    GetAlphaRect();
    xl &= -stepX; xr |= stepX-1;
    yb &= -stepY; yt |= stepY-1;
  }
  const int w4 = w*4;
  unsigned short* alpha = alpha_calcs + yb*w4;
  buf += pitch*yb;

  // Apply Y
  // different paths for different bitdepth
  // todo PF 161208 shiftX shiftY bits_per_pixel templates
  // perpaps int->byte, short (faster??)
  if constexpr(bits_per_pixel == 8) {
    for (int y=yb; y<=yt; y+=1) {
      for (int x=xl; x<=xr; x+=1) {
        const int x4 = x<<2;
        const int basealpha = alpha[x4+0];
        if (basealpha != 256) {
          buf[x] = BYTE((buf[x] * basealpha + alpha[x4 + 3]) >> 8);
        }
      }
      buf += pitch;
      alpha += w4;
    }
  }
  else if constexpr(bits_per_pixel >= 10 && bits_per_pixel <= 16) { // uint16_t
    for (int y=yb; y<=yt; y+=1) {
      for (int x=xl; x<=xr; x+=1) {
        const int x4 = x<<2;
        const int basealpha = alpha[x4+0];
        if (basealpha != 256) {
          reinterpret_cast<uint16_t *>(buf)[x] = (uint16_t)((reinterpret_cast<uint16_t *>(buf)[x] * basealpha + ((int)alpha[x4 + 3] << (bits_per_pixel-8))) >> 8);
        }
      }
      buf += pitch;
      alpha += w4;
    }
  }
  else if constexpr(bits_per_pixel == 32) { // float assume 0..1.0 scale
    for (int y=yb; y<=yt; y+=1) {
      for (int x=xl; x<=xr; x+=1) {
        const int x4 = x<<2;
        const int basealpha = alpha[x4+0];
        if (basealpha != 256) {
          reinterpret_cast<float *>(buf)[x] = reinterpret_cast<float *>(buf)[x] * basealpha / 256.0f + alpha[x4 + 3] / 65536.0f;
        }
      }
      buf += pitch;
      alpha += w4;
    }
  }

  if (!bufU) return;

  // This will not be fast, but it will be generic.
  const int skipThresh = 256 << (shiftX+shiftY);
  const int shifter = 8+shiftX+shiftY;
  const int UVw4 = w<<(2+shiftY);
  const int xlshiftX = xl>>shiftX;

  alpha = alpha_calcs + yb*w4;
  bufU += (pitchUV*yb)>>shiftY;
  bufV += (pitchUV*yb)>>shiftY;

  // different paths for different bitdepth
  if constexpr(bits_per_pixel == 8) {
    for (int y=yb; y<=yt; y+=stepY) {
      for (int x=xl, xs=xlshiftX; x<=xr; x+=stepX, xs+=1) {
        unsigned short* UValpha = alpha + x*4;
        int basealphaUV = 0;
        int au = 0;
        int av = 0;
        for (int i = 0; i<stepY; i++) {
          for (int j = 0; j<stepX; j++) {
            basealphaUV += UValpha[0 + j*4];
            av          += UValpha[1 + j*4];
            au          += UValpha[2 + j*4];
          }
          UValpha += w4;
        }
        if (basealphaUV != skipThresh) {
          bufU[xs] = BYTE((bufU[xs] * basealphaUV + au) >> shifter);
          bufV[xs] = BYTE((bufV[xs] * basealphaUV + av) >> shifter);
        }
      }// end for x
      bufU  += pitchUV;
      bufV  += pitchUV;
      alpha += UVw4;
    }//end for y
  }
  else if constexpr(bits_per_pixel >= 10 && bits_per_pixel <= 16) { // uint16_t
    for (int y=yb; y<=yt; y+=stepY) {
      for (int x=xl, xs=xlshiftX; x<=xr; x+=stepX, xs+=1) {
        unsigned short* UValpha = alpha + x*4;
        int basealphaUV = 0;
        int au = 0;
        int av = 0;
        for (int i = 0; i<stepY; i++) {
          for (int j = 0; j<stepX; j++) {
            basealphaUV += UValpha[0 + j*4];
            av          += UValpha[1 + j*4];
            au          += UValpha[2 + j*4];
          }
          UValpha += w4;
        }
        if (basealphaUV != skipThresh) {
          reinterpret_cast<uint16_t *>(bufU)[xs] = (uint16_t)((reinterpret_cast<uint16_t *>(bufU)[xs] * basealphaUV + (au << (bits_per_pixel-8))) >> shifter);
          reinterpret_cast<uint16_t *>(bufV)[xs] = (uint16_t)((reinterpret_cast<uint16_t *>(bufV)[xs] * basealphaUV + (av << (bits_per_pixel-8))) >> shifter);
        }
      }// end for x
      bufU  += pitchUV;
      bufV  += pitchUV;
      alpha += UVw4;
    }//end for y
  }
  else if constexpr(bits_per_pixel == 32) { // float. assume 0..1.0 scale
    const float shifter_inv_f = 1.0f / (1 << shifter);
    const float a_factor = shifter_inv_f / 256.0f;
#ifdef FLOAT_CHROMA_IS_HALF_CENTERED
    const float middle_shift_f = 0.0f;
#else
    const float middle_shift_f = isRGB ? 0.0f : 0.5f; // yes, correction needed in and out
#endif
    for (int y=yb; y<=yt; y+=stepY) {
      for (int x=xl, xs=xlshiftX; x<=xr; x+=stepX, xs+=1) {
        unsigned short* UValpha = alpha + x*4;
        int basealphaUV = 0;
        int au = 0;
        int av = 0;
        for (int i = 0; i<stepY; i++) {
          for (int j = 0; j<stepX; j++) {
            basealphaUV += UValpha[0 + j*4];
            av          += UValpha[1 + j*4];
            au          += UValpha[2 + j*4];
          }
          UValpha += w4;
        }
        if (basealphaUV != skipThresh) {
          const float basealphaUV_f = (float)basealphaUV * shifter_inv_f;
          reinterpret_cast<float *>(bufU)[xs] = (reinterpret_cast<float *>(bufU)[xs] + middle_shift_f) * basealphaUV_f + au * a_factor - middle_shift_f;
          reinterpret_cast<float *>(bufV)[xs] = (reinterpret_cast<float *>(bufV)[xs] + middle_shift_f) * basealphaUV_f + av * a_factor - middle_shift_f;
        }
      }// end for x
      bufU  += pitchUV;
      bufV  += pitchUV;
      alpha += UVw4;
    }//end for y
  }
}

void Antialiaser::ApplyPlanar(BYTE* buf, int pitch, int pitchUV, BYTE* bufU, BYTE* bufV, int shiftX, int shiftY, int bits_per_pixel, bool isRGB) {
  const int stepX = 1 << shiftX;
  const int stepY = 1 << shiftY;
  switch (bits_per_pixel) {
  case 8:
    if (shiftX == 0 && shiftY == 0) {
      ApplyPlanar_core<0, 0, 8>(buf, pitch, pitchUV, bufU, bufV, isRGB); // 4:4:4
      return;
    }
    else if (shiftX == 1 && shiftY == 0) {
      ApplyPlanar_core<1, 0, 8>(buf, pitch, pitchUV, bufU, bufV, isRGB); // 4:2:2
      return;
    }
    else if (shiftX == 1 && shiftY == 1) {
      ApplyPlanar_core<1, 1, 8>(buf, pitch, pitchUV, bufU, bufV, isRGB); // 4:2:0
      return;
    }
    break;
  case 10:
    if (shiftX == 0 && shiftY == 0) {
      ApplyPlanar_core<0, 0, 10>(buf, pitch, pitchUV, bufU, bufV, isRGB); // 4:4:4
      return;
    }
    else if (shiftX == 1 && shiftY == 0) {
      ApplyPlanar_core<1, 0, 10>(buf, pitch, pitchUV, bufU, bufV, isRGB); // 4:2:2
      return;
    }
    else if (shiftX == 1 && shiftY == 1) {
      ApplyPlanar_core<1, 1, 10>(buf, pitch, pitchUV, bufU, bufV, isRGB); // 4:2:0
      return;
    }
    break;
  case 12:
    if (shiftX == 0 && shiftY == 0) {
      ApplyPlanar_core<0, 0, 12>(buf, pitch, pitchUV, bufU, bufV, isRGB); // 4:4:4
      return;
    }
    else if (shiftX == 1 && shiftY == 0) {
      ApplyPlanar_core<1, 0, 12>(buf, pitch, pitchUV, bufU, bufV, isRGB); // 4:2:2
      return;
    }
    else if (shiftX == 1 && shiftY == 1) {
      ApplyPlanar_core<1, 1, 12>(buf, pitch, pitchUV, bufU, bufV, isRGB); // 4:2:0
      return;
    }
    break;
  case 14:
    if (shiftX == 0 && shiftY == 0) {
      ApplyPlanar_core<0, 0, 14>(buf, pitch, pitchUV, bufU, bufV, isRGB); // 4:4:4
      return;
    }
    else if (shiftX == 1 && shiftY == 0) {
      ApplyPlanar_core<1, 0, 14>(buf, pitch, pitchUV, bufU, bufV, isRGB); // 4:2:2
      return;
    }
    else if (shiftX == 1 && shiftY == 1) {
      ApplyPlanar_core<1, 1, 14>(buf, pitch, pitchUV, bufU, bufV, isRGB); // 4:2:0
      return;
    }
    break;
  case 16:
    if (shiftX == 0 && shiftY == 0) {
      ApplyPlanar_core<0, 0, 16>(buf, pitch, pitchUV, bufU, bufV, isRGB); // 4:4:4
      return;
    }
    else if (shiftX == 1 && shiftY == 0) {
      ApplyPlanar_core<1, 0, 16>(buf, pitch, pitchUV, bufU, bufV, isRGB); // 4:2:2
      return;
    }
    else if (shiftX == 1 && shiftY == 1) {
      ApplyPlanar_core<1, 1, 16>(buf, pitch, pitchUV, bufU, bufV, isRGB); // 4:2:0
      return;
    }
    break;
  case 32:
    if (shiftX == 0 && shiftY == 0) {
      ApplyPlanar_core<0, 0, 32>(buf, pitch, pitchUV, bufU, bufV, isRGB); // 4:4:4
      return;
    }
    else if (shiftX == 1 && shiftY == 0) {
      ApplyPlanar_core<1, 0, 32>(buf, pitch, pitchUV, bufU, bufV, isRGB); // 4:2:2
      return;
    }
    else if (shiftX == 1 && shiftY == 1) {
      ApplyPlanar_core<1, 1, 32>(buf, pitch, pitchUV, bufU, bufV, isRGB); // 4:2:0
      return;
    }
    break;
  }
  // keep old path for for any nonstandard surprise

  if (dirty) {
    GetAlphaRect();
    xl &= -stepX; xr |= stepX-1;
    yb &= -stepY; yt |= stepY-1;
  }
  const int w4 = w*4;
  unsigned short* alpha = alpha_calcs + yb*w4;
  buf += pitch*yb;

  // Apply Y
  // different paths for different bitdepth
  // todo PF 161208 shiftX shiftY bits_per_pixel templates
  // perpaps int->byte, short (faster??)
  if(bits_per_pixel == 8) {
      for (int y=yb; y<=yt; y+=1) {
          for (int x=xl; x<=xr; x+=1) {
              const int x4 = x<<2;
              const int basealpha = alpha[x4+0];
              if (basealpha != 256) {
                  buf[x] = BYTE((buf[x] * basealpha + alpha[x4 + 3]) >> 8);
              }
          }
          buf += pitch;
          alpha += w4;
      }
  }
  else if (bits_per_pixel >= 10 && bits_per_pixel <= 16) { // uint16_t
      for (int y=yb; y<=yt; y+=1) {
          for (int x=xl; x<=xr; x+=1) {
              const int x4 = x<<2;
              const int basealpha = alpha[x4+0];
              if (basealpha != 256) {
                  reinterpret_cast<uint16_t *>(buf)[x] = (uint16_t)((reinterpret_cast<uint16_t *>(buf)[x] * basealpha + ((int)alpha[x4 + 3] << (bits_per_pixel-8))) >> 8);
              }
          }
          buf += pitch;
          alpha += w4;
      }
  }
  else { // float assume 0..1.0 scale
      for (int y=yb; y<=yt; y+=1) {
          for (int x=xl; x<=xr; x+=1) {
              const int x4 = x<<2;
              const int basealpha = alpha[x4+0];
              if (basealpha != 256) {
                  reinterpret_cast<float *>(buf)[x] = reinterpret_cast<float *>(buf)[x] * basealpha / 256.0f + alpha[x4 + 3] / 65536.0f;
              }
          }
          buf += pitch;
          alpha += w4;
      }
  }

  if (!bufU) return;

  // This will not be fast, but it will be generic.
  const int skipThresh = 256 << (shiftX+shiftY);
  const int shifter = 8+shiftX+shiftY;
  const int UVw4 = w<<(2+shiftY);
  const int xlshiftX = xl>>shiftX;

  alpha = alpha_calcs + yb*w4;
  bufU += (pitchUV*yb)>>shiftY;
  bufV += (pitchUV*yb)>>shiftY;

  // different paths for different bitdepth
  if(bits_per_pixel == 8) {
      for (int y=yb; y<=yt; y+=stepY) {
          for (int x=xl, xs=xlshiftX; x<=xr; x+=stepX, xs+=1) {
              unsigned short* UValpha = alpha + x*4;
              int basealphaUV = 0;
              int au = 0;
              int av = 0;
              for (int i = 0; i<stepY; i++) {
                  for (int j = 0; j<stepX; j++) {
                      basealphaUV += UValpha[0 + j*4];
                      av          += UValpha[1 + j*4];
                      au          += UValpha[2 + j*4];
                  }
                  UValpha += w4;
              }
              if (basealphaUV != skipThresh) {
                  bufU[xs] = BYTE((bufU[xs] * basealphaUV + au) >> shifter);
                  bufV[xs] = BYTE((bufV[xs] * basealphaUV + av) >> shifter);
              }
          }// end for x
          bufU  += pitchUV;
          bufV  += pitchUV;
          alpha += UVw4;
      }//end for y
  }
  else if (bits_per_pixel >= 10 && bits_per_pixel <= 16) { // uint16_t
      for (int y=yb; y<=yt; y+=stepY) {
          for (int x=xl, xs=xlshiftX; x<=xr; x+=stepX, xs+=1) {
              unsigned short* UValpha = alpha + x*4;
              int basealphaUV = 0;
              int au = 0;
              int av = 0;
              for (int i = 0; i<stepY; i++) {
                  for (int j = 0; j<stepX; j++) {
                      basealphaUV += UValpha[0 + j*4];
                      av          += UValpha[1 + j*4];
                      au          += UValpha[2 + j*4];
                  }
                  UValpha += w4;
              }
              if (basealphaUV != skipThresh) {
                  reinterpret_cast<uint16_t *>(bufU)[xs] = (uint16_t)((reinterpret_cast<uint16_t *>(bufU)[xs] * basealphaUV + (au << (bits_per_pixel-8))) >> shifter);
                  reinterpret_cast<uint16_t *>(bufV)[xs] = (uint16_t)((reinterpret_cast<uint16_t *>(bufV)[xs] * basealphaUV + (av << (bits_per_pixel-8))) >> shifter);
              }
          }// end for x
          bufU  += pitchUV;
          bufV  += pitchUV;
          alpha += UVw4;
      }//end for y
  }
  else { // float. assume 0..1.0 scale
#ifdef FLOAT_CHROMA_IS_HALF_CENTERED
      const float middle_shift_f = 0.0f;
#else
      const float middle_shift_f = isRGB ? 0.0f : 0.5f; // yes, correction needed in and out
#endif
      const float shifter_inv_f = 1.0f / (1 << shifter);
      const float a_factor = shifter_inv_f / 256.0f;
      for (int y=yb; y<=yt; y+=stepY) {
          for (int x=xl, xs=xlshiftX; x<=xr; x+=stepX, xs+=1) {
              unsigned short* UValpha = alpha + x*4;
              int basealphaUV = 0;
              int au = 0;
              int av = 0;
              for (int i = 0; i<stepY; i++) {
                  for (int j = 0; j<stepX; j++) {
                      basealphaUV += UValpha[0 + j*4];
                      av          += UValpha[1 + j*4];
                      au          += UValpha[2 + j*4];
                  }
                  UValpha += w4;
              }
              if (basealphaUV != skipThresh) {
                  const float basealphaUV_f = (float)basealphaUV * shifter_inv_f;
                  reinterpret_cast<float *>(bufU)[xs] = (reinterpret_cast<float *>(bufU)[xs] + middle_shift_f) * basealphaUV_f + au * a_factor - middle_shift_f;
                  reinterpret_cast<float *>(bufV)[xs] = (reinterpret_cast<float *>(bufV)[xs] + middle_shift_f) * basealphaUV_f + av * a_factor - middle_shift_f;
              }
          }// end for x
          bufU  += pitchUV;
          bufV  += pitchUV;
          alpha += UVw4;
      }//end for y
  }
}


void Antialiaser::ApplyYUY2(BYTE* buf, int pitch) {
  if (dirty) {
    GetAlphaRect();
	xl &= -2; xr |= 1;
  }
  unsigned short* alpha = alpha_calcs + yb*w*4;
  buf += pitch*yb;

  for (int y=yb; y<=yt; ++y) {
    for (int x=xl; x<=xr; x+=2) {
      const int basealpha0  = alpha[x*4+0];
      const int basealpha1  = alpha[x*4+4];
      const int basealphaUV = basealpha0 + basealpha1;

      if (basealphaUV != 512) {
        buf[x*2+0] = BYTE((buf[x*2+0] * basealpha0 + alpha[x*4+3]) >> 8);
        buf[x*2+2] = BYTE((buf[x*2+2] * basealpha1 + alpha[x*4+7]) >> 8);

        const int au  = alpha[x*4+2] + alpha[x*4+6];
        buf[x*2+1] = BYTE((buf[x*2+1] * basealphaUV + au) >> 9);

        const int av  = alpha[x*4+1] + alpha[x*4+5];
        buf[x*2+3] = BYTE((buf[x*2+3] * basealphaUV + av) >> 9);
      }
    }
    buf += pitch;
    alpha += w*4;
  }
}


void Antialiaser::ApplyRGB24_48(BYTE* buf, int pitch, int pixelsize) {
  if (dirty) GetAlphaRect();
  unsigned short* alpha = alpha_calcs + yb*w*4;
  buf  += pitch*(h-yb-1);

  if(pixelsize==1)
  {
      for (int y=yb; y<=yt; ++y) {
        for (int x=xl; x<=xr; ++x) {
          const int basealpha = alpha[x*4+0];
          if (basealpha != 256) {
            buf[x*3+0] = BYTE((buf[x*3+0] * basealpha + alpha[x*4+1]) >> 8);
            buf[x*3+1] = BYTE((buf[x*3+1] * basealpha + alpha[x*4+2]) >> 8);
            buf[x*3+2] = BYTE((buf[x*3+2] * basealpha + alpha[x*4+3]) >> 8);
          }
        }
        buf -= pitch;
        alpha += w*4;
      }
  }
  else {
      // pixelsize == 2
      for (int y=yb; y<=yt; ++y) {
          for (int x=xl; x<=xr; ++x) {
              const int basealpha = alpha[x*4+0];
              if (basealpha != 256) {
                  reinterpret_cast<uint16_t *>(buf)[x*3+0] = (uint16_t)((reinterpret_cast<uint16_t *>(buf)[x*3+0] * basealpha + ((int)alpha[x*4+1] << 8)) >> 8);
                  reinterpret_cast<uint16_t *>(buf)[x*3+1] = (uint16_t)((reinterpret_cast<uint16_t *>(buf)[x*3+1] * basealpha + ((int)alpha[x*4+2] << 8)) >> 8);
                  reinterpret_cast<uint16_t *>(buf)[x*3+2] = (uint16_t)((reinterpret_cast<uint16_t *>(buf)[x*3+2] * basealpha + ((int)alpha[x*4+3] << 8)) >> 8);
              }
          }
          buf -= pitch;
          alpha += w*4;
      }
  }
}


void Antialiaser::ApplyRGB32_64(BYTE* buf, int pitch, int pixelsize) {
  if (dirty) GetAlphaRect();
  unsigned short* alpha = alpha_calcs + yb*w*4;
  buf  += pitch*(h-yb-1);

  if(pixelsize==1)
  {
      for (int y=yb; y<=yt; ++y) {
        for (int x=xl; x<=xr; ++x) {
          const int basealpha = alpha[x*4+0];
          if (basealpha != 256) {
            buf[x*4+0] = BYTE((buf[x*4+0] * basealpha + alpha[x*4+1]) >> 8);
            buf[x*4+1] = BYTE((buf[x*4+1] * basealpha + alpha[x*4+2]) >> 8);
            buf[x*4+2] = BYTE((buf[x*4+2] * basealpha + alpha[x*4+3]) >> 8);
          }
        }
        buf -= pitch;
        alpha += w*4;
      }
  }
  else {
      // pixelsize == 2
      for (int y=yb; y<=yt; ++y) {
          for (int x=xl; x<=xr; ++x) {
              const int basealpha = alpha[x*4+0];
              if (basealpha != 256) {
                  reinterpret_cast<uint16_t *>(buf)[x*4+0] = (uint16_t)((reinterpret_cast<uint16_t *>(buf)[x*4+0] * basealpha + ((int)alpha[x*4+1] << 8)) >> 8);
                  reinterpret_cast<uint16_t *>(buf)[x*4+1] = (uint16_t)((reinterpret_cast<uint16_t *>(buf)[x*4+1] * basealpha + ((int)alpha[x*4+2] << 8)) >> 8);
                  reinterpret_cast<uint16_t *>(buf)[x*4+2] = (uint16_t)((reinterpret_cast<uint16_t *>(buf)[x*4+2] * basealpha + ((int)alpha[x*4+3] << 8)) >> 8);
              }
          }
          buf -= pitch;
          alpha += w*4;
      }

  }
}


void Antialiaser::GetAlphaRect()
{
  dirty = false;

  static BYTE bitcnt[256],    // bit count
              bitexl[256],    // expand to left bit
              bitexr[256];    // expand to right bit
  static bool fInited = false;
  static unsigned short gamma[129]; // Gamma lookups

  if (!fInited) {
    fInited = true;

    const double scale = 516*64/sqrt(128.0);
    {for(int i=0; i<=128; i++)
      gamma[i]=uint16_t(sqrt((double)i) * scale + 0.5); // Gamma = 2.0
    }

	{for(int i=0; i<256; i++) {
      BYTE b=0, l=0, r=0;

      if (i&  1) { b=1; l|=0x01; r|=0xFF; }
      if (i&  2) { ++b; l|=0x03; r|=0xFE; }
      if (i&  4) { ++b; l|=0x07; r|=0xFC; }
      if (i&  8) { ++b; l|=0x0F; r|=0xF8; }
      if (i& 16) { ++b; l|=0x1F; r|=0xF0; }
      if (i& 32) { ++b; l|=0x3F; r|=0xE0; }
      if (i& 64) { ++b; l|=0x7F; r|=0xC0; }
      if (i&128) { ++b; l|=0xFF; r|=0x80; }

      bitcnt[i] = b;
      bitexl[i] = l;
      bitexr[i] = r;
    }}
  }

  const int RYtext = ((textcolor>>16)&255), GUtext = ((textcolor>>8)&255), BVtext = (textcolor&255);
  const int RYhalo = ((halocolor>>16)&255), GUhalo = ((halocolor>>8)&255), BVhalo = (halocolor&255);

  // Scaled Alpha
  const int Atext = 255 - ((textcolor >> 24) & 0xFF);
  const int Ahalo = 255 - ((halocolor >> 24) & 0xFF);

  const int srcpitch = (w+4+3) & -4;

  xl=0;
  xr=w+1;
  yt=-1;
  yb=h;

  unsigned short* dest = alpha_calcs;
  for (int y=0; y<h; ++y) {
    BYTE* src = (BYTE*)lpAntialiasBits + ((h-y-1)*8 + 20) * srcpitch + 2;
    int wt = w;
    do {
      int i;

#pragma warning(push)
#pragma warning(disable: 4068)
      DWORD tmp = 0;

      if (interlaced) {
#pragma unroll
        for (int ii = -8; ii < 16; ++ii) {
          tmp |= *reinterpret_cast<int*>(src + srcpitch*ii - 1);
        }
      } else {
#if 0
#pragma unroll

        for (int i = -12; i < 20; ++i) {
          tmp |= *reinterpret_cast<int*>(src + srcpitch*i - 1);
        }
#else
        BYTE *tmpsrc = src + srcpitch*(-12) - 1;
#pragma unroll
        // PF 161208 speedup test manual unroll, no pragma in VS
        for (int ii = -12; ii < 20; ii+=4) { // 0..31
          tmp |= *reinterpret_cast<int*>(tmpsrc) |
            *reinterpret_cast<int*>(tmpsrc+srcpitch*1) |
            *reinterpret_cast<int*>(tmpsrc+srcpitch*2) |
            *reinterpret_cast<int*>(tmpsrc+srcpitch*3)/* |
            *reinterpret_cast<int*>(tmpsrc+srcpitch*4) |
            *reinterpret_cast<int*>(tmpsrc+srcpitch*5) |
            *reinterpret_cast<int*>(tmpsrc+srcpitch*6) |
            *reinterpret_cast<int*>(tmpsrc+srcpitch*7)*/
            ;
          tmpsrc += srcpitch*4;
        }
#endif
      }

      tmp &= 0x00FFFFFF;
#pragma warning(pop)


      if (tmp != 0) {     // quick exit in a common case
        if (wt >= xl) xl=wt;
        if (wt <= xr) xr=wt;
        if (y  >= yt) yt=y;
        if (y  <= yb) yb=y;

        int alpha1, alpha2;

        alpha1 = alpha2 = 0;

        if (interlaced) {
          BYTE topmask=0, cenmask=0, botmask=0;
          BYTE hmasks[16], mask;

          for(i=-4; i<12; i++) {// For interlaced include extra half cells above and below
            mask = src[srcpitch*i];
            // How many lit pixels in the centre cell?
            alpha1 += bitcnt[mask];
            // turn on all halo bits if cell has any lit pixels
            mask = - !! mask;
            // Check left and right neighbours, extend the halo
            // mask 8 pixels in from the nearest lit pixels.
            mask |= bitexr[src[srcpitch*i-1]];
            mask |= bitexl[src[srcpitch*i+1]];
            hmasks[i+4] = mask;
          }

          // Extend halo vertically to 8x8 blocks
          for(i=-4; i<4;  i++) topmask |= hmasks[i+4];
          for(i=0;  i<8;  i++) cenmask |= hmasks[i+4];
          for(i=4;  i<12; i++) botmask |= hmasks[i+4];
          // Check the 3x1.5 cells above
          for(mask = topmask, i=-4; i<4; i++) {
            mask |= bitexr[ src[srcpitch*(i+8)-1] ];
            mask |=    - !! src[srcpitch*(i+8)  ];
            mask |= bitexl[ src[srcpitch*(i+8)+1] ];
            hmasks[i+4] |= mask;
          }
          for(mask = cenmask, i=0; i<8; i++) {
            mask |= bitexr[ src[srcpitch*(i+8)-1] ];
            mask |=    - !! src[srcpitch*(i+8)  ];
            mask |= bitexl[ src[srcpitch*(i+8)+1] ];
            hmasks[i+4] |= mask;
          }
          for(mask = botmask, i=4; i<12; i++) {
            mask |= bitexr[ src[srcpitch*(i+8)-1] ];
            mask |=    - !! src[srcpitch*(i+8)  ];
            mask |= bitexl[ src[srcpitch*(i+8)+1] ];
            hmasks[i+4] |= mask;
          }
          // Check the 3x1.5 cells below
          for(mask = botmask, i=11; i>=4; i--) {
            mask |= bitexr[ src[srcpitch*(i-8)-1] ];
            mask |=    - !! src[srcpitch*(i-8)  ];
            mask |= bitexl[ src[srcpitch*(i-8)+1] ];
            hmasks[i+4] |= mask;
          }
          for(mask = cenmask,i=7; i>=0; i--) {
            mask |= bitexr[ src[srcpitch*(i-8)-1] ];
            mask |=    - !! src[srcpitch*(i-8)  ];
            mask |= bitexl[ src[srcpitch*(i-8)+1] ];
            hmasks[i+4] |= mask;
          }
          for(mask = topmask, i=3; i>=-4; i--) {
            mask |= bitexr[ src[srcpitch*(i-8)-1] ];
            mask |=    - !! src[srcpitch*(i-8)  ];
            mask |= bitexl[ src[srcpitch*(i-8)+1] ];
            hmasks[i+4] |= mask;
          }
          // count the halo pixels
          for(i=0; i<16; i++)
            alpha2 += bitcnt[hmasks[i]];
        }
        else {
          // How many lit pixels in the centre cell?
          for(i=0; i<8; i++)
            alpha1 += bitcnt[src[srcpitch*i]];
          alpha1 *=2;

          if (alpha1) {
            // If we have any lit pixels we fully occupy the cell.
            alpha2 = 128;
          }
          else {
            // No lit pixels here so build the halo mask from the neighbours
            BYTE cenmask = 0;

            // Check left and right neighbours, extend the halo
            // mask 8 pixels in from the nearest lit pixels.
            for(i=0; i<8; i++) {
              cenmask |= bitexr[src[srcpitch*i-1]];
              cenmask |= bitexl[src[srcpitch*i+1]];
            }

            if (cenmask == 0xFF) {
              // If we have hard adjacent lit pixels we fully occupy this cell.
              alpha2 = 128;
            }
            else {
              BYTE hmasks[8], mask;

              mask = cenmask;
#if 1
              { // PF 161208 speedup test get first two bytes as word
                int index = srcpitch*(0 + 8);
                for (int ii = 0; ii < 8; ii++) {
                  // Check the 3 cells above
                  const uint16_t ab = *reinterpret_cast<uint16_t *>(src + index - 1);
                  mask |= bitexr[ab & 0xFF];
                  mask |= -!!(ab >> 8);
                  mask |= bitexl[src[index + 1]];
                  hmasks[ii] = mask;
                  index += srcpitch;
                }
              }
#else
              for(i=0; i<8; i++) {
                // Check the 3 cells above
                mask |= bitexr[ src[srcpitch*(i+8)-1] ];
                mask |=    - !! src[srcpitch*(i+8)  ];
                mask |= bitexl[ src[srcpitch*(i+8)+1] ];
                hmasks[i] = mask;
              }
#endif

              mask = cenmask;
#if 1
              { // PF 161208 speedup test get first two bytes as word
                int index = srcpitch*(7 - 8);
                for (int ii = 7; ii >= 0; ii--) {
                  // Check the 3 cells below
                  const uint16_t ab = *reinterpret_cast<uint16_t *>(src + index - 1);
                  mask |= bitexr[ab & 0xFF];
                  mask |= -!!(ab >> 8);
                  mask |= bitexl[src[index + 1]];
                  alpha2 += bitcnt[hmasks[ii] | mask];
                  index -= srcpitch;
                }
              }
#else
              for (i = 7; i >= 0; i--) {
                // Check the 3 cells below
                mask |= bitexr[src[srcpitch*(i - 8) - 1]];
                mask |= -!!src[srcpitch*(i - 8)];
                mask |= bitexl[src[srcpitch*(i - 8) + 1]];

                alpha2 += bitcnt[hmasks[i] | mask];
              }
            }
#endif
              alpha2 *=2;
            }
          }
        }
        alpha2  = gamma[alpha2];
        alpha1  = gamma[alpha1];

        alpha2 -= alpha1;
        alpha2 *= Ahalo;
        alpha1 *= Atext;
        // Pre calulate table for quick use  --  Pc = (Pc * dest[0] + dest[c]) >> 8;

		dest[0] = (unsigned short)((64*516*255 - alpha1 -          alpha2)>>15);
		dest[1] = (unsigned short)((    BVtext * alpha1 + BVhalo * alpha2)>>15);
		dest[2] = (unsigned short)((    GUtext * alpha1 + GUhalo * alpha2)>>15);
		dest[3] = (unsigned short)((    RYtext * alpha1 + RYhalo * alpha2)>>15);
      }
      else {
        dest[0] = 256;
        dest[1] = 0;
        dest[2] = 0;
        dest[3] = 0;
      }

      dest += 4;
      ++src;
    } while(--wt);
  }

  xl=w-xl;
  xr=w-xr;
}

#endif // #if defined(AVS_WINDOWS) && !defined(NO_WIN_GDI)



/*************************************
 *******   Show Frame Number    ******
 ************************************/

ShowFrameNumber::ShowFrameNumber(PClip _child, bool _scroll, int _offset, int _x, int _y, const char _fontname[],
                                 int _size, int _textcolor, int _halocolor, int font_width, int font_angle, IScriptEnvironment* env)
 : GenericVideoFilter(_child),
#if defined(AVS_WINDOWS) && !defined(NO_WIN_GDI)
  antialiaser(vi.width, vi.height, _fontname, _size,
    vi.IsYUV() || vi.IsYUVA() ? RGB2YUV_Rec601(_textcolor) : _textcolor,
    vi.IsYUV() || vi.IsYUVA() ? RGB2YUV_Rec601(_halocolor) : _halocolor,
    font_width, font_angle),
#endif
  scroll(_scroll), offset(_offset), size(_size), x(_x), y(_y),
  textcolor(vi.IsYUV() || vi.IsYUVA() ? RGB2YUV_Rec601(_textcolor) : _textcolor),
  halocolor(vi.IsYUV() || vi.IsYUVA() ? RGB2YUV_Rec601(_halocolor) : _halocolor)
{
  AVS_UNUSED(env);

#if defined(AVS_WINDOWS) && !defined(NO_WIN_GDI)
#else
  // internal font
  const bool bold = false;
  current_font = GetBitmapFont(size, "Terminus", bold, false);
  if (current_font == nullptr)
  {
    // size
    current_font = GetBitmapFont(size, "", bold, false);
    if (current_font == nullptr)
      current_font = GetBitmapFont(size, "", !bold, false);
  }
#endif
}

enum { DefXY = (int)0x80000000 };

PVideoFrame ShowFrameNumber::GetFrame(int n, IScriptEnvironment* env) {
  PVideoFrame frame = child->GetFrame(n, env);
  n+=offset;
  if (n < 0) return frame;

#if defined(AVS_WINDOWS) && !defined(NO_WIN_GDI)
  HDC hdc = antialiaser.GetDC();
  if (!hdc) return frame;
#else
  if (current_font == nullptr)
    return frame;
#endif
  env->MakeWritable(&frame);

#if defined(AVS_WINDOWS) && !defined(NO_WIN_GDI)
  RECT r = { 0, 0, 32767, 32767 };
  FillRect(hdc, &r, (HBRUSH)GetStockObject(BLACK_BRUSH));
#endif
  char text[16];
  snprintf(text, sizeof(text), "%05d", n);
  text[15] = 0;
  if (x!=DefXY || y!=DefXY) {
#if defined(AVS_WINDOWS) && !defined(NO_WIN_GDI)
    SetTextAlign(hdc, TA_BASELINE|TA_LEFT);
    TextOut(hdc, x+16, y+16, text, (int)strlen(text));
#else
    std::wstring ws = charToWstring(text, true);
    SimpleTextOutW(current_font.get(), vi, frame, x, y, ws, false, textcolor, halocolor, true, 1);
#endif
  } else if (scroll) {
    int n1 = vi.IsFieldBased() ? (n/2) : n;
#if defined(AVS_WINDOWS) && !defined(NO_WIN_GDI)
    int y2 = size + size * (n1 % (vi.height * 8 / size));
    SetTextAlign(hdc, TA_BASELINE | (child->GetParity(n) ? TA_LEFT : TA_RIGHT));
    TextOut(hdc, child->GetParity(n) ? 32 : vi.width*8+8, y2, text, (int)strlen(text));
#else
    int y2 = size + size * (n1 % (vi.height / size));
    std::wstring ws = charToWstring(text, true);
    if(child->GetParity(n))
      SimpleTextOutW(current_font.get(), vi, frame, 4, y2, ws, false, textcolor, halocolor, true, 1); // left
    else
      SimpleTextOutW(current_font.get(), vi, frame, vi.width - 1, y2, ws, false, textcolor, halocolor, true, 3); // right
#endif

  }
  else {
#if defined(AVS_WINDOWS) && !defined(NO_WIN_GDI)
    SetTextAlign(hdc, TA_BASELINE | (child->GetParity(n) ? TA_LEFT : TA_RIGHT));
    int text_len = (int)strlen(text);
    for (int y2 = size; y2 < vi.height * 8; y2 += size)
      TextOut(hdc, child->GetParity(n) ? 32 : vi.width * 8 + 8, y2, text, text_len);
#else
    std::wstring ws = charToWstring(text, true);
    // size-1 because of bottom alignment
    for (int y2 = size - 1; y2 < vi.height; y2 += size) {
      if (child->GetParity(n))
        SimpleTextOutW(current_font.get(), vi, frame, 4, y2, ws, false, textcolor, halocolor, true, 1); // bottom-left
      else
        SimpleTextOutW(current_font.get(), vi, frame, vi.width - 1, y2, ws, false, textcolor, halocolor, true, 3); // bottom-right
    }
#endif
  }
#if defined(AVS_WINDOWS) && !defined(NO_WIN_GDI)
  GdiFlush();

  antialiaser.Apply(vi, &frame, frame->GetPitch());
#endif
  return frame;
}


AVSValue __cdecl ShowFrameNumber::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  PClip clip = args[0].AsClip();
  bool scroll = args[1].AsBool(false);
  const int offset = args[2].AsInt(0);
#if defined(AVS_WINDOWS) && !defined(NO_WIN_GDI)
  const int x = args[3].IsFloat() ? int(args[3].AsFloat() * 8 + 0.5) : DefXY;
  const int y = args[4].IsFloat() ? int(args[4].AsFloat() * 8 + 0.5) : DefXY;
  const char* font = args[5].AsString("Arial");
  const int size = int(args[6].AsFloat(24)*8+0.5);
  const int font_width = int(args[9].AsFloat(0) * 8 + 0.5);
#else
  const int x = args[3].IsFloat() ? int(args[3].AsFloat() + 0.5) : DefXY;
  const int y = args[4].IsFloat() ? int(args[4].AsFloat() + 0.5) : DefXY;
  const char* font = args[5].AsString("Terminus");
  const int size = int(args[6].AsFloat(24) + 0.5);
  const int font_width = int(args[9].AsFloat(0) + 0.5);
#endif
  const int text_color = args[7].AsInt(0xFFFF00);
  const int halo_color = args[8].AsInt(0);
  const int font_angle = int(args[10].AsFloat(0) * 10 + 0.5);

  if ((x==DefXY) ^ (y==DefXY))
	env->ThrowError("ShowFrameNumber: both x and y position must be specified");

  return new ShowFrameNumber(clip, scroll, offset, x, y, font, size, text_color, halo_color, font_width, font_angle, env);
}





/*************************************
 *******   Show CRC32 Number    ******
 ************************************/

ShowCRC32::ShowCRC32(PClip _child, bool _scroll, int _offset, int _x, int _y, const char _fontname[],
  int _size, int _textcolor, int _halocolor, int font_width, int font_angle, IScriptEnvironment* env)
  : GenericVideoFilter(_child),
#if defined(AVS_WINDOWS) && !defined(NO_WIN_GDI)
  antialiaser(vi.width, vi.height, _fontname, _size,
    vi.IsYUV() || vi.IsYUVA() ? RGB2YUV_Rec601(_textcolor) : _textcolor,
    vi.IsYUV() || vi.IsYUVA() ? RGB2YUV_Rec601(_halocolor) : _halocolor,
    font_width, font_angle),
#endif
  scroll(_scroll), offset(_offset), size(_size), x(_x), y(_y),
  textcolor(vi.IsYUV() || vi.IsYUVA() ? RGB2YUV_Rec601(_textcolor) : _textcolor),
  halocolor(vi.IsYUV() || vi.IsYUVA() ? RGB2YUV_Rec601(_halocolor) : _halocolor)
{
  AVS_UNUSED(env);
  
  build_crc32_table();

#if defined(AVS_WINDOWS) && !defined(NO_WIN_GDI)
#else
  // internal font
  const bool bold = false;
  current_font = GetBitmapFont(size, "Terminus", bold, false);
  if (current_font == nullptr)
  {
    // size
    current_font = GetBitmapFont(size, "", bold, false);
    if (current_font == nullptr)
      current_font = GetBitmapFont(size, "", !bold, false);
  }
#endif
}

void ShowCRC32::build_crc32_table(void) {
  for (uint32_t i = 0; i < 256; i++) {
    uint32_t ch = i;
    uint32_t crc = 0;
    for (size_t j = 0; j < 8; j++) {
      uint32_t b = (ch ^ crc) & 1;
      crc >>= 1;
      if (b) crc = crc ^ 0xEDB88320;
      ch >>= 1;
    }
    crc32_table[i] = crc;
  }
}


PVideoFrame ShowCRC32::GetFrame(int n, IScriptEnvironment* env) {
  PVideoFrame frame = child->GetFrame(n, env);
  n += offset;
  if (n < 0) return frame;

#if defined(AVS_WINDOWS) && !defined(NO_WIN_GDI)
  HDC hdc = antialiaser.GetDC();
  if (!hdc) return frame;
#else
  if (current_font == nullptr)
    return frame;
#endif
  env->MakeWritable(&frame);

#if defined(AVS_WINDOWS) && !defined(NO_WIN_GDI)
  RECT r = { 0, 0, 32767, 32767 };
  FillRect(hdc, &r, (HBRUSH)GetStockObject(BLACK_BRUSH));
#endif

  auto ptr = frame->GetReadPtr();
  auto pitch = frame->GetPitch();
  auto width = frame->GetRowSize();
  auto height = frame->GetHeight();
  uint32_t crc = 0xFFFFFFFF;;

  for (int yy = 0; yy < height; yy++) {
    for (int xx = 0; xx < width; xx++)
    {
      uint8_t b = ptr[xx];
      uint32_t t = (b ^ crc) & 0xFF;
      crc = (crc >> 8) ^ crc32_table[t];
    }
    ptr += pitch;
  }
  crc = ~crc;

  char text[16];
  snprintf(text, sizeof(text), "%08X", (int)crc);
  text[15] = 0;

  if (x != DefXY || y != DefXY) {
#if defined(AVS_WINDOWS) && !defined(NO_WIN_GDI)
    SetTextAlign(hdc, TA_BASELINE | TA_LEFT);
    TextOut(hdc, x + 16, y + 16, text, (int)strlen(text));
#else
    std::wstring ws = charToWstring(text, true);
    SimpleTextOutW(current_font.get(), vi, frame, x, y, ws, false, textcolor, halocolor, true, 1);
#endif
  }
  else if (scroll) {
    int n1 = vi.IsFieldBased() ? (n / 2) : n;
#if defined(AVS_WINDOWS) && !defined(NO_WIN_GDI)
    int y2 = size + size * (n1 % (vi.height * 8 / size));
    SetTextAlign(hdc, TA_BASELINE | (child->GetParity(n) ? TA_LEFT : TA_RIGHT));
    TextOut(hdc, child->GetParity(n) ? 32 : vi.width * 8 + 8, y2, text, (int)strlen(text));
#else
    int y2 = size + size * (n1 % (vi.height / size));
    std::wstring ws = charToWstring(text, true);
    if (child->GetParity(n))
      SimpleTextOutW(current_font.get(), vi, frame, 4, y2, ws, false, textcolor, halocolor, true, 1); // left
    else
      SimpleTextOutW(current_font.get(), vi, frame, vi.width - 1, y2, ws, false, textcolor, halocolor, true, 3); // right
#endif

  }
  else {
#if defined(AVS_WINDOWS) && !defined(NO_WIN_GDI)
    SetTextAlign(hdc, TA_BASELINE | (child->GetParity(n) ? TA_LEFT : TA_RIGHT));
    int text_len = (int)strlen(text);
    for (int y2 = size; y2 < vi.height * 8; y2 += size)
      TextOut(hdc, child->GetParity(n) ? 32 : vi.width * 8 + 8, y2, text, text_len);
#else
    std::wstring ws = charToWstring(text, true);
    for (int y2 = size; y2 < vi.height; y2 += size) {
      if (child->GetParity(n))
        SimpleTextOutW(current_font.get(), vi, frame, 4, y2, ws, false, textcolor, halocolor, true, 1); // left
      else
        SimpleTextOutW(current_font.get(), vi, frame, vi.width - 1, y2, ws, false, textcolor, halocolor, true, 3); // right
    }
#endif
  }
#if defined(AVS_WINDOWS) && !defined(NO_WIN_GDI)
  GdiFlush();

  antialiaser.Apply(vi, &frame, frame->GetPitch());
#endif
  return frame;
}


AVSValue __cdecl ShowCRC32::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  PClip clip = args[0].AsClip();
  bool scroll = args[1].AsBool(false);
  const int offset = args[2].AsInt(0);
#if defined(AVS_WINDOWS) && !defined(NO_WIN_GDI)
  const int x = args[3].IsFloat() ? int(args[3].AsFloat() * 8 + 0.5) : DefXY;
  const int y = args[4].IsFloat() ? int(args[4].AsFloat() * 8 + 0.5) : DefXY;
  const char* font = args[5].AsString("Arial");
  const int size = int(args[6].AsFloat(24) * 8 + 0.5);
  const int font_width = int(args[9].AsFloat(0) * 8 + 0.5);
#else
  const int x = args[3].IsFloat() ? int(args[3].AsFloat() + 0.5) : DefXY;
  const int y = args[4].IsFloat() ? int(args[4].AsFloat() + 0.5) : DefXY;
  const char* font = args[5].AsString("Terminus");
  const int size = int(args[6].AsFloat(24) + 0.5);
  const int font_width = int(args[9].AsFloat(0) + 0.5);
#endif
  const int text_color = args[7].AsInt(0xFFFF00);
  const int halo_color = args[8].AsInt(0);
  const int font_angle = int(args[10].AsFloat(0) * 10 + 0.5);

  if ((x == DefXY) ^ (y == DefXY))
    env->ThrowError("ShowCRC32: both x and y position must be specified");

  return new ShowCRC32(clip, scroll, offset, x, y, font, size, text_color, halo_color, font_width, font_angle, env);
}







/***********************************
 *******   Show SMPTE code    ******
 **********************************/

ShowSMPTE::ShowSMPTE(PClip _child, double _rate, const char* offset, int _offset_f, int _x, int _y, const char _fontname[],
                     int _size, int _textcolor, int _halocolor, int font_width, int font_angle, IScriptEnvironment* env)
  : GenericVideoFilter(_child),
#if defined(AVS_WINDOWS) && !defined(NO_WIN_GDI)
  antialiaser(vi.width, vi.height, _fontname, _size,
      vi.IsYUV() || vi.IsYUVA() ? RGB2YUV_Rec601(_textcolor) : _textcolor,
      vi.IsYUV() || vi.IsYUVA() ? RGB2YUV_Rec601(_halocolor) : _halocolor,
      font_width, font_angle),
#endif
  x(_x), y(_y),
  textcolor(vi.IsYUV() || vi.IsYUVA() ? RGB2YUV_Rec601(_textcolor) : _textcolor),
  halocolor(vi.IsYUV() || vi.IsYUVA() ? RGB2YUV_Rec601(_halocolor) : _halocolor)
{
#if defined(AVS_WINDOWS) && !defined(NO_WIN_GDI)
#else
  const bool bold = false;
  current_font = GetBitmapFont(_size, "Terminus", bold, false);
  if (current_font == nullptr)
  {
    // size
    current_font = GetBitmapFont(_size, "", bold, false);
    if (current_font == nullptr)
      current_font = GetBitmapFont(_size, "", !bold, false);
  }
#endif
  int off_f, off_sec, off_min, off_hour;

  rate = int(_rate + 0.5);
  dropframe = false;
  if (_rate > 23.975 && _rate < 23.977) { // Pulldown drop frame rate
    rate = 24;
    dropframe = true;
  }
  else if (_rate > 29.969 && _rate < 29.971) {
    rate = 30;
    dropframe = true;
  }
  else if (_rate > 47.951 && _rate < 47.953) {
    rate = 48;
    dropframe = true;
  }
  else if (_rate > 59.939 && _rate < 59.941) {
    rate = 60;
    dropframe = true;
  }
  else if (_rate > 119.879 && _rate < 119.881) {
    rate = 120;
    dropframe = true;
  }
  else if (fabs(_rate - rate) > 0.001) {
    env->ThrowError("ShowSMPTE: rate argument must be 23.976, 29.97 or an integer");
  }

  if (offset) {
	if (strlen(offset)!=11 || offset[2] != ':' || offset[5] != ':' || offset[8] != ':')
	  env->ThrowError("ShowSMPTE:  offset should be of the form \"00:00:00:00\" ");
	if (!isdigit(offset[0]) || !isdigit(offset[1]) || !isdigit(offset[3]) || !isdigit(offset[4])
	 || !isdigit(offset[6]) || !isdigit(offset[7]) || !isdigit(offset[9]) || !isdigit(offset[10]))
	  env->ThrowError("ShowSMPTE:  offset should be of the form \"00:00:00:00\" ");

	off_hour = atoi(offset);

	off_min = atoi(offset+3);
	if (off_min > 59)
	  env->ThrowError("ShowSMPTE:  make sure that the number of minutes in the offset is in the range 0..59");

	off_sec = atoi(offset+6);
	if (off_sec > 59)
	  env->ThrowError("ShowSMPTE:  make sure that the number of seconds in the offset is in the range 0..59");

	off_f = atoi(offset+9);
	if (off_f >= rate)
	  env->ThrowError("ShowSMPTE:  make sure that the number of frames in the offset is in the range 0..%d", rate-1);

	offset_f = off_f + rate*(off_sec + 60*off_min + 3600*off_hour);
	if (dropframe) {
	  if (rate == 30) {
		int c = 0;
		c = off_min + 60*off_hour;  // number of drop events
		c -= c/10; // less non-drop events on 10 minutes
		c *=2; // drop 2 frames per drop event
		offset_f -= c;
	  }
	  else {
//  Need to cogitate with the guys about this
//  gotta drop 86.3 counts per hour. So until
//  a proper formula is found, just wing it!
		offset_f -= 2 * ((offset_f+1001)/2002);
	  }
	}
  }
  else {
	offset_f = _offset_f;
  }
}


PVideoFrame __stdcall ShowSMPTE::GetFrame(int n, IScriptEnvironment* env)
{
  PVideoFrame frame = child->GetFrame(n, env);
  n+=offset_f;
  if (n < 0) return frame;

#if defined(AVS_WINDOWS) && !defined(NO_WIN_GDI)
  HDC hdc = antialiaser.GetDC();
  if (!hdc) return frame;
#else
  if (current_font == nullptr)
    return frame;
#endif

  env->MakeWritable(&frame);

  if (dropframe) {
    if ((rate == 30) || (rate == 60) || (rate == 120)) {
	// at 10:00, 20:00, 30:00, etc. nothing should happen if offset=0
	  const int f = rate/30;
	  const int r = n % f;
	  n /= f;

	  const int high = n / 17982;
	  int low = n % 17982;
	  if (low>=2)
		low += 2 * ((low-2) / 1798);
	  n = high * 18000 + low;

	  n = f*n + r;
	}
	else {
//  Needs some cogitating
	  n += 2 * ((n+1001)/2002);
	}
  }

  char text[16];

  if (rate > 0) {
    int frames = n % rate;
    int sec = n/rate;
    int min = sec/60;
    int hour = sec/3600;

    snprintf(text, sizeof(text),
              rate>99 ? "%02d:%02d:%02d:%03d" : "%02d:%02d:%02d:%02d",
              hour, min%60, sec%60, frames);
  }
  else {
    int ms = (int)(((int64_t)n * vi.fps_denominator * 1000 / vi.fps_numerator)%1000);
    int sec = (int)((int64_t)n * vi.fps_denominator / vi.fps_numerator);
    int min = sec/60;
    int hour = sec/3600;

    snprintf(text, sizeof(text), "%02d:%02d:%02d.%03d", hour, min%60, sec%60, ms);
  }
  text[15] = 0;

#if defined(AVS_WINDOWS) && !defined(NO_WIN_GDI)
  SetTextAlign(hdc, TA_BASELINE|TA_CENTER);
  TextOut(hdc, x+16, y+16, text, (int)strlen(text));
  GdiFlush();

  antialiaser.Apply(vi, &frame, frame->GetPitch());
#else
  const bool utf8 = true;
  auto ws = charToWstring(text, utf8);
  SimpleTextOutW(current_font.get(), vi, frame, x + 2, y + 2, ws, true, textcolor, halocolor, false, 5);
#endif

  return frame;
}

AVSValue __cdecl ShowSMPTE::CreateSMTPE(AVSValue args, void*, IScriptEnvironment* env)
{
  PClip clip = args[0].AsClip();
  const VideoInfo& arg0vi = args[0].AsClip()->GetVideoInfo();
  double def_rate = (double)arg0vi.fps_numerator / arg0vi.fps_denominator;
  double dfrate = args[1].AsDblDef(def_rate);
  const char* offset = args[2].AsString(0);
  const int offset_f = args[3].AsInt(0);
  const int xreal = arg0vi.width/2;
  const int yreal = arg0vi.height-8;
#if defined(AVS_WINDOWS) && !defined(NO_WIN_GDI)
  const int x = int(args[4].AsDblDef(xreal)*8+0.5);
  const int y = int(args[5].AsDblDef(yreal)*8+0.5);
  const char* font = args[6].AsString("Arial");
  const int size = int(args[7].AsFloat(24)*8+0.5);
  const int font_width = int(args[10].AsFloat(0) * 8 + 0.5);
#else
  const int x = int(args[4].AsDblDef(xreal) + 0.5);
  const int y = int(args[5].AsDblDef(yreal) + 0.5);
  const char* font = args[6].AsString("Terminus");
  const int size = int(args[7].AsFloat(24) + 0.5);
  const int font_width = int(args[10].AsFloat(0) + 0.5);
#endif
  const int text_color = args[8].AsInt(0xFFFF00);
  const int halo_color = args[9].AsInt(0);
  const int font_angle = int(args[11].AsFloat(0)*10+0.5);
  return new ShowSMPTE(clip, dfrate, offset, offset_f, x, y, font, size, text_color, halo_color, font_width, font_angle, env);
}

AVSValue __cdecl ShowSMPTE::CreateTime(AVSValue args, void*, IScriptEnvironment* env)
{
  PClip clip = args[0].AsClip();
  const int offset_f = args[1].AsInt(0);
  const int xreal = args[0].AsClip()->GetVideoInfo().width/2;
  const int yreal = args[0].AsClip()->GetVideoInfo().height-8;
#if defined(AVS_WINDOWS) && !defined(NO_WIN_GDI)
  const int x = int(args[2].AsDblDef(xreal)*8+0.5);
  const int y = int(args[3].AsDblDef(yreal)*8+0.5);
  const char* font = args[4].AsString("Arial");
  const int size = int(args[5].AsFloat(24)*8+0.5);
  const int font_width = int(args[8].AsFloat(0) * 8 + 0.5);
#else
  const int x = int(args[2].AsDblDef(xreal) + 0.5);
  const int y = int(args[3].AsDblDef(yreal) + 0.5);
  const char* font = args[4].AsString("Terminus");
  const int size = int(args[5].AsFloat(24) + 0.5);
  const int font_width = int(args[8].AsFloat(0) + 0.5);
#endif
  const int text_color = args[6].AsInt(0xFFFF00);
  const int halo_color = args[7].AsInt(0);
  const int font_angle = int(args[9].AsFloat(0)*10+0.5);
  return new ShowSMPTE(clip, 0.0, NULL, offset_f, x, y, font, size, text_color, halo_color, font_width, font_angle, env);
}





#if defined(AVS_WINDOWS) && !defined(NO_WIN_GDI)

/***********************************
 *******   Subtitle Filter    ******
 **********************************/

Subtitle::Subtitle( PClip _child, const char _text[], int _x, int _y, int _firstframe,
                    int _lastframe, const char _fontname[], int _size, int _textcolor,
                    int _halocolor, int _align, int _spc, bool _multiline, int _lsp,
                    int _font_width, int _font_angle, bool _interlaced, const char _font_filename[], const bool _utf8, IScriptEnvironment* env)
 : GenericVideoFilter(_child),
  x(_x), y(_y),
  firstframe(_firstframe), lastframe(_lastframe), size(_size),
  lsp(_lsp), font_width(_font_width), font_angle(_font_angle), multiline(_multiline), interlaced(_interlaced),
  textcolor(vi.IsYUV() || vi.IsYUVA() ? RGB2YUV_Rec601(_textcolor) : _textcolor),
  halocolor(vi.IsYUV() || vi.IsYUVA() ? RGB2YUV_Rec601(_halocolor) : _halocolor),
  align(_align), spc(_spc),
  fontname(_fontname), text(_text), font_filename(_font_filename), utf8(_utf8),
  antialiaser(nullptr)
{
  if (*font_filename) {
    int added_font_count = AddFontResourceEx(
      font_filename, // font file name
      FR_PRIVATE,    // font characteristics
      NULL);
    // If the function succeeds, the return value specifies the number of fonts added.
    if (added_font_count == 0) {
      env->ThrowError("SubTitle: font %s not found", font_filename);
    }
  }
}



Subtitle::~Subtitle(void)
{
  delete antialiaser;
  if (font_filename) {
    // same as in AddFontResourceEx
    BOOL b = RemoveFontResourceEx(
      font_filename, // name of font file
      FR_PRIVATE,    // font characteristics
      NULL           // Reserved.
    );
    if (!b) {
      // we can't do anything
    }
  }
}



PVideoFrame Subtitle::GetFrame(int n, IScriptEnvironment* env)
{
  PVideoFrame frame = child->GetFrame(n, env);

  if (n >= firstframe && n <= lastframe) {
    env->MakeWritable(&frame);
    if (!antialiaser) // :FIXME: CriticalSection
	  InitAntialiaser(env);
    if (antialiaser) {
	  antialiaser->Apply(vi, &frame, frame->GetPitch());
	  // Release all the windows drawing stuff
	  // and just keep the alpha calcs
	  antialiaser->FreeDC();
	}
  }
  // if we get far enough away from the frames we're supposed to
  // subtitle, then junk the buffered drawing information
  if (antialiaser && (n < firstframe-10 || n > lastframe+10 || n == vi.num_frames-1)) {
	delete antialiaser;
	antialiaser = 0; // :FIXME: CriticalSection
  }

  return frame;
}

AVSValue __cdecl Subtitle::Create(AVSValue args, void*, IScriptEnvironment* env)
{
    PClip clip = args[0].AsClip();
    const char* text = args[1].AsString();
    const int first_frame = args[4].AsInt(0);
    const int last_frame = args[5].AsInt(clip->GetVideoInfo().num_frames-1);
    const char* font = args[6].AsString("Arial");
    const int size = int(args[7].AsFloat(18)*8+0.5);
    const int text_color = args[8].AsInt(0xFFFF00);
    const int halo_color = args[9].AsInt(0);
    const int align = args[10].AsInt(args[2].AsFloat(0)==-1?2:7);
    const int spc = args[11].AsInt(0);
    const bool multiline = args[12].Defined();
    const int lsp = args[12].AsInt(0);
    const int font_width = int(args[13].AsFloat(0)*8+0.5);
    const int font_angle = int(args[14].AsFloat(0)*10+0.5);
    const bool interlaced = args[15].AsBool(false);
    const char* font_filename = args[16].AsString("");
    const bool utf8 = args[17].AsBool(false);

    if ((align < 1) || (align > 9))
     env->ThrowError("Subtitle: Align values are 1 - 9 mapped to your numeric pad");

    int defx, defy;
    switch (align) {
	 case 1: case 4: case 7: defx = 8; break;
     case 2: case 5: case 8: defx = -1; break;
     case 3: case 6: case 9: defx = clip->GetVideoInfo().width-8; break;
     default: defx = 8; break; }
    switch (align) {
     case 1: case 2: case 3: defy = clip->GetVideoInfo().height-2; break;
     case 4: case 5: case 6: defy = -1; break;
	 case 7: case 8: case 9: defy = 0; break;
     default: defy = (size+4)/8; break; }

    const int x = int(args[2].AsDblDef(defx)*8+0.5);
    const int y = int(args[3].AsDblDef(defy)*8+0.5);

    return new Subtitle(clip, text, x, y, first_frame, last_frame, font, size, text_color,
	                    halo_color, align, spc, multiline, lsp, font_width, font_angle, interlaced, font_filename, utf8, env);
}

void Subtitle::InitAntialiaser(IScriptEnvironment* env)
{
  antialiaser = new Antialiaser(vi.width, vi.height, fontname, size, textcolor, halocolor,
                                font_width, font_angle, interlaced);

  int real_x = x;
  int real_y = y;
  unsigned int al = 0;
  char *_text = 0;
  wchar_t *_textw = 0;

  HDC hdcAntialias = antialiaser->GetDC();
  if (!hdcAntialias) goto GDIError;

  switch (align) // This spec where [X, Y] is relative to the text (inverted logic)
  { case 1: al = TA_BOTTOM   | TA_LEFT; break;		// .----
    case 2: al = TA_BOTTOM   | TA_CENTER; break;	// --.--
    case 3: al = TA_BOTTOM   | TA_RIGHT; break;		// ----.
    case 4: al = TA_BASELINE | TA_LEFT; break;		// .____
    case 5: al = TA_BASELINE | TA_CENTER; break;	// __.__
    case 6: al = TA_BASELINE | TA_RIGHT; break;		// ____.
    case 7: al = TA_TOP      | TA_LEFT; break;		// `----
    case 8: al = TA_TOP      | TA_CENTER; break;	// --`--
    case 9: al = TA_TOP      | TA_RIGHT; break;		// ----`
    default: al= TA_BASELINE | TA_LEFT; break;		// .____
  }
  if (SetTextCharacterExtra(hdcAntialias, spc) == 0x80000000) goto GDIError;
  if (SetTextAlign(hdcAntialias, al) == GDI_ERROR) goto GDIError;

  if (x==-7) real_x = (vi.width>>1)*8;
  if (y==-7) real_y = (vi.height>>1)*8;

  if (utf8) {
    // Test:
    // Title="Cherry blossom "+CHR($E6)+CHR($A1)+CHR($9C)+CHR($E3)+CHR($81)+CHR($AE)+CHR($E8)+CHR($8A)+CHR($B1)
    // SubTitle(Title, utf8 = true)
    auto textw = Utf8ToWideChar(text);

    if (!multiline) {
      if (!TextOutW(hdcAntialias, real_x + 16, real_y + 16, textw.get(), (int)wcslen(textw.get())))
      {
        goto GDIError;
      }
    }
    else {
      // multiline patch -- tateu
      wchar_t *pdest, *psrc;
      int result, y_inc = real_y + 16;
      wchar_t search[] = L"\\n";
      psrc = _textw = _wcsdup(textw.get()); // don't mangle the string constant -- Gavino
      if (!_textw) goto GDIError;
      int length = (int)wcslen(psrc);

      do {
        pdest = wcsstr(psrc, search); // strstr
        while (pdest != NULL && pdest != psrc && *(pdest - 1) == L'\\') { // \n escape -- foxyshadis
          for (size_t i = pdest - psrc; i > 0; i--) psrc[i] = psrc[i - 1];
          psrc++;
          --length;
          pdest = wcsstr(pdest + 1, search); // strstr
        }
        result = pdest == NULL ? length : (int)size_t(pdest - psrc);
        if (!TextOutW(hdcAntialias, real_x + 16, y_inc, psrc, result)) goto GDIError;
        y_inc += size + lsp;
        psrc = pdest + 2;
        length -= result + 2;
      } while (pdest != NULL && length > 0);
      free(_textw);
      _textw = NULL;
    }
  }
  else {
    if (!multiline) {
      if (!TextOut(hdcAntialias, real_x + 16, real_y + 16, text, (int)strlen(text))) goto GDIError;
    }
    else {
    // multiline patch -- tateu
      char *pdest, *psrc;
      int result, y_inc = real_y + 16;
      char search[] = "\\n";
      psrc = _text = _strdup(text); // don't mangle the string constant -- Gavino
      if (!_text) goto GDIError;
      int length = (int)strlen(psrc);

      do {
        pdest = strstr(psrc, search);
        while (pdest != NULL && pdest != psrc && *(pdest - 1) == '\\') { // \n escape -- foxyshadis
          for (size_t i = pdest - psrc; i > 0; i--) psrc[i] = psrc[i - 1];
          psrc++;
          --length;
          pdest = strstr(pdest + 1, search);
        }
        result = pdest == NULL ? length : (int)size_t(pdest - psrc);
        if (!TextOut(hdcAntialias, real_x + 16, y_inc, psrc, result)) goto GDIError;
        y_inc += size + lsp;
        psrc = pdest + 2;
        length -= result + 2;
      } while (pdest != NULL && length > 0);
      free(_text);
      _text = NULL;
    }
  }
  if (!GdiFlush()) goto GDIError;
  return;

GDIError:
  delete antialiaser;
  antialiaser = 0;
  free(_text);
  free(_textw);

  env->ThrowError("Subtitle: GDI or Insufficient Memory Error");
}



#endif

static int CalcFontSizeForInfo(int w, int h, bool autolarge, int upper_limit)
{
  // if frame is smaller than minimum then font size will be decreased proportionally
  // normalized to 640x480 @fontsize=15:
  // 14 lines @height=15 takes ~480/2 pixels (half vertical screens)
  // Info screen takes horizontally ~0.6 * 640 pixels at font size 15

#if defined(AVS_WINDOWS) && !defined(NO_WIN_GDI)
  const int min_font_size = 1;
  const int max_font_size = 255; // n/a
  const int reference_font_size = 15;
  const int reference_font_width = 8;
#else
  const int min_font_size = 12;
  const int max_font_size = 32;
  const int reference_font_size = 18; // 16x8 or 18x10, latter is more visible
  const int reference_font_width = 10;
#endif
  // knowing that a usual Info() is 45x15 characters, plus a margin
  const int reference_min_height = 15 * reference_font_size;
  const int reference_min_width = 48 * reference_font_width; // (45 + margin) characters

  // 0..reference_min_width      :decrease (as always in Avs)
  // reference_min_width ... 640 : constant size
  // 640+                        : enlarge when autolarge is enabled or else constant size

  int w_restricted_font_size;
  if (w < reference_min_width)
    w_restricted_font_size = (reference_font_size * w) / reference_min_width;
  else if (w >= 640 && autolarge)
    w_restricted_font_size = reference_font_size * w / 640;
  else
    w_restricted_font_size = reference_font_size;

  int h_restricted_font_size;
  if (h < reference_min_height)
    h_restricted_font_size = (reference_font_size * h) / reference_min_height;
  else if (h >= 480 && autolarge)
    h_restricted_font_size = reference_font_size * h / 480;
  else
    h_restricted_font_size = reference_font_size;

  int effective_font_size = (w_restricted_font_size < h_restricted_font_size) ? w_restricted_font_size : h_restricted_font_size;

  effective_font_size = std::max(effective_font_size, min_font_size);
  if (upper_limit > 0)
    effective_font_size = std::min(effective_font_size, max_font_size);

#if defined(AVS_WINDOWS) && !defined(NO_WIN_GDI)
  return effective_font_size * 8; // GDI takes *8
#else
  return effective_font_size / 2 * 2; // no odd sized fixed fonts atm.
#endif

}

/***********************************
 *******   SimpleText Filter   *****
 ***********************************/

SimpleText::SimpleText(PClip _child, const char _text[], int _x, int _y, int _firstframe,
  int _lastframe, const char _fontname[], int _size, int _textcolor,
  int _halocolor, int _align, int _spc, bool _multiline, int _lsp,
  int _font_width, int _font_angle, bool _interlaced, const char _font_filename[], const bool _utf8, const bool _bold, IScriptEnvironment* env)
  : GenericVideoFilter(_child),
  x(_x), y(_y),
  firstframe(_firstframe), lastframe(_lastframe), size(_size), lsp(_lsp),
  multiline(_multiline),
  textcolor(vi.IsYUV() || vi.IsYUVA() ? RGB2YUV_Rec601(_textcolor) : _textcolor),
  halocolor(vi.IsYUV() || vi.IsYUVA() ? RGB2YUV_Rec601(_halocolor) : _halocolor), // not supported
  align(_align),
  halocolor_orig(_halocolor),
  fontname(_fontname),
  text(_text),
  // spc(_spc), font_width(_font_width), font_angle(_font_angle), interlaced(_interlaced),
  font_filename(_font_filename), utf8(_utf8),
  bold(_bold)
{

  if (*font_filename) {
    // external font file
    bool debugSave = size < 0;
    current_font = GetBitmapFont(0, font_filename, false, debugSave); // 0: size n/a
    if (current_font == nullptr)
      env->ThrowError("SimpleText: file %s not found or unknown file format", font_filename);
  }
  else
  {
    // internal font
    if (fontname) {
      current_font = GetBitmapFont(size, fontname, bold, false); // 12, "Terminus"
      if (current_font == nullptr)
        env->ThrowError("SimpleText: internal font name %s in size %d not found", fontname, size);
    }
    else {
      // size
      current_font = GetBitmapFont(size, "", bold, false); // 12, ""
      if (current_font == nullptr)
        env->ThrowError("SimpleText: fixed font size %d not found", size);
    }
  }
}



SimpleText::~SimpleText(void)
{
  // nothing here yet
}


PVideoFrame SimpleText::GetFrame(int n, IScriptEnvironment* env)
{
  PVideoFrame frame = child->GetFrame(n, env);

  if (n >= firstframe && n <= lastframe) {
    env->MakeWritable(&frame);

    int real_x = x;
    int real_y = y;

    // Test:
    // Title="Cherry blossom "+CHR($E6)+CHR($A1)+CHR($9C)+CHR($E3)+CHR($81)+CHR($AE)+CHR($E8)+CHR($8A)+CHR($B1)
    std::string s(text);

    if (multiline) { // filter parameter, true when lsp is given
      // multiline case: string contains '\' and 'n' characters explicitely
      // SubTitle compatibility: literal "\n" means line break, but "\\n" means that literal "\n" will be printed
      // Thus we replace two-character literal "\n" to \n (0x0A) then when "\" and \n found, we change it back to literal "\n"
      size_t index = 0;
      while (true) {
        index = s.find("\\n", index);
        if (index == std::string::npos) break;
        s.replace(index, 1, "\n");
        s.erase(index + 1, 1); // two characters replaced by a single one, erase at the second position
        index += 1; // length of the string to replace
      }
      // '\' and '\n' back to literal "\n"
      index = 0;
      while (true) {
        index = s.find("\\\n", index); // yes, \ and \n
        if (index == std::string::npos) break;
        s.replace(index, 2, "\\n"); // back to "\" + "n"
        index += 2; // length of the string to replace
      }
    }

    std::wstring ws = charToWstring(s.c_str(), utf8); // to wchar_t either from utf8 (win/linux) or ansi (win)

    // halocolor MSB
    // FF: fadeIt
    // 01-FE: no halo
    // 00: use halocolor
    SimpleTextOutW_multi(current_font.get(), vi, frame, real_x, real_y, ws,
      halocolor_orig == 0xFF000000, // fadeIt, special halocolor, when MSB byte is FF
      textcolor, halocolor,
      (halocolor_orig & 0xFF000000) == 0, // use halocolor when MSB byte is zero
      align, lsp);
  }

  return frame;
}

AVSValue __cdecl SimpleText::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  PClip clip = args[0].AsClip();
  const char* text = args[1].AsString();
  const int first_frame = args[4].AsInt(0);
  const int last_frame = args[5].AsInt(clip->GetVideoInfo().num_frames - 1);
  const char* font = args[6].AsString("Terminus"); // Terminus, info_h are embedded
  const int size = int(args[7].AsFloat(18)); // height 12, 14, 16, 18, 20, 24, 28, 32
  const int text_color = args[8].AsInt(0xFFFF00);
  const int halo_color = args[9].AsInt(0);
  const int align = args[10].AsInt(args[2].AsFloat(0) == -1 ? 2 : 7);
  const int spc = args[11].AsInt(0);
  const bool multiline = args[12].Defined();
  const int lsp = args[12].AsInt(0); // line spacing if multiline
  const int font_width = int(args[13].AsFloat(0) * 8 + 0.5); // n/a
  const int font_angle = int(args[14].AsFloat(0) * 10 + 0.5); // n/a
  const bool interlaced = args[15].AsBool(false); // n/a
  const char* font_filename = args[16].AsString("");
  const bool utf8 = args[17].AsBool(false); // linux: n/a
  const bool bold = args.ArraySize() >= 19 ? args[18].AsBool(false) : false; // different from SubTitle, guard array access of extra parameter

  // parameters marked with n/a are ignored; parameter list is currently the same as SubTitle

  if ((align < 1) || (align > 9))
    env->ThrowError("SimpleText: Align values are 1 - 9 mapped to your numeric pad");

  int defx, defy;
  switch (align) {
  case 1: case 4: case 7: defx = 8; break;
  case 2: case 5: case 8: defx = -1; break;
  case 3: case 6: case 9: defx = clip->GetVideoInfo().width - 8; break;
  default: defx = 8; break;
  }
  switch (align) {
  case 1: case 2: case 3: defy = clip->GetVideoInfo().height - 2; break;
  case 4: case 5: case 6: defy = -1; break;
  case 7: case 8: case 9: defy = 0; break;
  default: defy = /*(size + 4) / 8;*/ (size + 1) / 2; break; // no mul 8
  }

  const bool isXdefined = args[2].Defined();
  const bool isYdefined = args[3].Defined();

  const int x = int(args[2].AsDblDef(defx) /* * 8 */ + 0.5); // no mul 8 like in SubTitle
  const int y = int(args[3].AsDblDef(defy) /* * 8 */ + 0.5);

  int real_x = x;
  int real_y = y;
  // center check
  if (defx == -1 && !isXdefined)
    real_x = (clip->GetVideoInfo().width >> 1) /* * 8 */; // no mul 8 like in SubTitle

  if (defy == -1 && !isYdefined)
    real_y = (clip->GetVideoInfo().height >> 1) /* * 8 */; // no mul 8 like in SubTitle

  return new SimpleText(clip, text, real_x, real_y, first_frame, last_frame, font, size, text_color,
    halo_color, align, spc, multiline, lsp, font_width, font_angle, interlaced, font_filename, utf8, bold, env);
}



/***********************************
 *******   FilterInfo Filter    ******
 **********************************/

FilterInfo::FilterInfo( PClip _child, const char _fontname[], int _size, int _textcolor, int _halocolor, IScriptEnvironment* env)
  : GenericVideoFilter(_child), vii(AdjustVi()), size(_size),
  text_color(vi.IsYUV() || vi.IsYUVA() ? RGB2YUV_Rec601(_textcolor) : _textcolor),
  halo_color(vi.IsYUV() || vi.IsYUVA() ? RGB2YUV_Rec601(_halocolor) : _halocolor)
#if defined(AVS_WINDOWS) && !defined(NO_WIN_GDI)
  ,antialiaser(vi.width, vi.height, _fontname, size,
    vi.IsYUV() || vi.IsYUVA() ? RGB2YUV_Rec601(_textcolor) : _textcolor,
    vi.IsYUV() || vi.IsYUVA() ? RGB2YUV_Rec601(_halocolor) : _halocolor)
#endif
{
  AVS_UNUSED(env);
#if defined(AVS_WINDOWS) && !defined(NO_WIN_GDI)
#else
  // internal font
  const bool bold = true; // bold looks better
  current_font = GetBitmapFont(size, "Terminus", bold, false);
  if (current_font == nullptr)
  {
    current_font = GetBitmapFont(size, "", bold, false);
    if (current_font == nullptr)
      current_font = GetBitmapFont(size, "", !bold, false);
  }
#endif

}


FilterInfo::~FilterInfo(void)
{
}


const VideoInfo& FilterInfo::AdjustVi()
{
  if ( !vi.HasVideo() ) {
    vi.fps_denominator=1;
    vi.fps_numerator=24;
    vi.height=480;
    vi.num_frames=240;
    vi.pixel_type=VideoInfo::CS_BGR32;
    vi.width=640;
    vi.SetFieldBased(false);
  }
  return child->GetVideoInfo();
}


const char* const t_INT8="Integer 8 bit";
const char* const t_INT16="Integer 16 bit";
const char* const t_INT24="Integer 24 bit";
const char* const t_INT32="Integer 32 bit";
const char* const t_FLOAT32="Float 32 bit";
const char* const t_YES="YES";
const char* const t_NO="NO";
const char* const t_NONE="NONE";
const char* const t_TFF ="Top Field First            ";
const char* const t_BFF ="Bottom Field First         ";
const char* const t_ATFF="Assumed Top Field First    ";
const char* const t_ABFF="Assumed Bottom Field First ";
const char* const t_STFF="Top Field (Separated)      ";
const char* const t_SBFF="Bottom Field (Separated)   ";

#ifdef INTEL_INTRINSICS
std::string GetCpuMsg(IScriptEnvironment * env, bool avx512)
{
  int flags = env->GetCPUFlags();
  std::stringstream ss;

  if (!avx512) {
#ifndef _M_X64
    // don't display old capabilities when at least AVX is used
    if (!(flags & CPUF_AVX)) {
    //if (flags & CPUF_FPU)
    //  ss << "x87 ";
      if (flags & CPUF_MMX)
        ss << "MMX ";
      if (flags & CPUF_INTEGER_SSE)
        ss << "ISSE ";

      if (flags & CPUF_3DNOW_EXT)
        ss << "3DNOW_EXT";
      else if (flags & CPUF_3DNOW)
        ss << "3DNOW ";
    }

    if (flags & CPUF_SSE)
      ss << "SSE ";
#endif
    if (flags & CPUF_SSE2)
      ss << "SSE2 ";
    if (flags & CPUF_SSE3)
      ss << "SSE3 ";
    if (flags & CPUF_SSSE3)
      ss << "SSSE3 ";
    if (flags & CPUF_SSE4_1)
      ss << "SSE4.1 ";
    if (flags & CPUF_SSE4_2)
      ss << "SSE4.2 ";

    if (flags & CPUF_AVX)
      ss << "AVX ";
    if (flags & CPUF_AVX2)
      ss << "AVX2 ";
    if (flags & CPUF_FMA3)
      ss << "FMA3 ";
    if (flags & CPUF_FMA4)
      ss << "FMA4 ";
    if (flags & CPUF_F16C)
      ss << "F16C ";
  }
  else {
    if (flags & CPUF_AVX512F)
      ss << "AVX512F ";
    if (flags & CPUF_AVX512DQ)
      ss << "AVX512DQ ";
    if (flags & CPUF_AVX512PF)
      ss << "AVX512PF ";
    if (flags & CPUF_AVX512ER)
      ss << "AVX512ER ";
    if (flags & CPUF_AVX512CD)
      ss << "AVX512CD ";
    if (flags & CPUF_AVX512BW)
      ss << "AVX512BW ";
    if (flags & CPUF_AVX512VL)
      ss << "AVX512VL ";
    if (flags & CPUF_AVX512IFMA)
      ss << "AVX512IFMA ";
    if (flags & CPUF_AVX512VBMI)
      ss << "AVX512VBMI ";
  }
  return ss.str();
}
#else
std::string GetCpuMsg(IScriptEnvironment * env)
{
  std::stringstream ss;

  return ss.str();
}
#endif

bool FilterInfo::GetParity(int n)
{
  return vii.HasVideo() ? child->GetParity(n) : false;
}


PVideoFrame FilterInfo::GetFrame(int n, IScriptEnvironment* env)
{
  PVideoFrame frame = vii.HasVideo() ? child->GetFrame(n, env) : env->NewVideoFrame(vi);

  if ( !vii.HasVideo() ) {
    memset(frame->GetWritePtr(), 0, frame->GetPitch()*frame->GetHeight()); // Blank frame
  }

#if defined(AVS_WINDOWS) && !defined(NO_WIN_GDI)
  HDC hdcAntialias = antialiaser.GetDC();
  if (!hdcAntialias)
    return frame;
#else
  if (current_font == nullptr)
    return frame;
#endif
    const char* c_space = "Unknown";
    const char* s_type = t_NONE;
    const char* s_parity;
    char text[1024];
    int tlen;

    if (vii.HasVideo()) {
      c_space = GetPixelTypeName(vii.pixel_type);
      if (*c_space == '\0')
        c_space = "Unknown";
      if (vii.IsFieldBased()) {
        if (child->GetParity(n)) {
          s_parity = t_STFF;
        }
        else {
          s_parity = t_SBFF;
        }
      }
      else {
        if (child->GetParity(n)) {
          s_parity = vii.IsTFF() ? t_ATFF : t_TFF;
        }
        else {
          s_parity = vii.IsBFF() ? t_ABFF : t_BFF;
        }
      }
      int vLenInMsecs = (int)(1000.0 * (double)vii.num_frames * (double)vii.fps_denominator / (double)vii.fps_numerator);
      int cPosInMsecs = (int)(1000.0 * (double)n * (double)vii.fps_denominator / (double)vii.fps_numerator);

      tlen = snprintf(text, sizeof(text),
        "Frame: %8u of %-8u\n"                                //  28
        "Time: %02d:%02d:%02d.%03d of %02d:%02d:%02d.%03d\n"  //  35
        "ColorSpace: %s, BitsPerComponent: %u\n"              //  18=13+5
//        "Bits per component: %2u\n"                           //  22
        "Width:%4u pixels, Height:%4u pixels\n"              //  39
        "Frames per second: %7.4f (%u/%u)\n"                  //  51=31+20
        "FieldBased (Separated) Video: %s\n"                  //  35=32+3
        "Parity: %s\n"                                        //  35=9+26
        "Video Pitch: %5u bytes.\n"                           //  25
        "Has Audio: %s\n"                                     //  15=12+3
//        "123456789012345678901234567890123456789012345678901234567890\n"         // test
, n, vii.num_frames
, (cPosInMsecs / (60 * 60 * 1000)), (cPosInMsecs / (60 * 1000)) % 60, (cPosInMsecs / 1000) % 60, cPosInMsecs % 1000,
(vLenInMsecs / (60 * 60 * 1000)), (vLenInMsecs / (60 * 1000)) % 60, (vLenInMsecs / 1000) % 60, vLenInMsecs % 1000
, c_space
, vii.BitsPerComponent()
, vii.width, vii.height
, (float)vii.fps_numerator / (float)vii.fps_denominator, vii.fps_numerator, vii.fps_denominator
, vii.IsFieldBased() ? t_YES : t_NO
, s_parity
, frame->GetPitch()
, vii.HasAudio() ? t_YES : t_NO
);
    }
    else {
      tlen = snprintf(text, sizeof(text),
        "Frame: %8u of %-8u\n"
        "Has Video: NO\n"
        "Has Audio: %s\n"
        , n, vi.num_frames
        , vii.HasAudio() ? t_YES : t_NO
      );
    }
    if (vii.HasAudio()) {
      if (vii.SampleType() == SAMPLE_INT8)  s_type = t_INT8;
      else if (vii.SampleType() == SAMPLE_INT16) s_type = t_INT16;
      else if (vii.SampleType() == SAMPLE_INT24) s_type = t_INT24;
      else if (vii.SampleType() == SAMPLE_INT32) s_type = t_INT32;
      else if (vii.SampleType() == SAMPLE_FLOAT) s_type = t_FLOAT32;

      int aLenInMsecs = (int)(1000.0 * (double)vii.num_audio_samples / (double)vii.audio_samples_per_second);
      tlen += snprintf(text + tlen, sizeof(text) - tlen,
        "Audio Channels: %-8u\n"                              //  25
        "Sample Type: %s\n"                                   //  28=14+14
        "Samples Per Second: %5d\n"                           //  26
        "Audio length: %" PRIu64 " samples. %02d:%02d:%02d.%03d\n"  //  57=37+20
        , vii.AudioChannels()
        , s_type
        , vii.audio_samples_per_second
        , vii.num_audio_samples,
        (aLenInMsecs / (60 * 60 * 1000)), (aLenInMsecs / (60 * 1000)) % 60, (aLenInMsecs / 1000) % 60, aLenInMsecs % 1000
      );
    }
    else {
      strcpy(text + tlen, "\n");
      tlen += 1;
    }
    // CPU capabilities
    tlen += snprintf(text + tlen, sizeof(text) - tlen,
      "CPU: %s\n"
#ifdef INTEL_INTRINSICS
      , GetCpuMsg(env, false).c_str()
#else
      , GetCpuMsg(env).c_str()
#endif
    );
#ifdef INTEL_INTRINSICS
    // AVX512 flags in new line (too long)
    std::string avx512 = GetCpuMsg(env, true);
    if (avx512.length() > 0) {
      tlen += snprintf(text + tlen, sizeof(text) - tlen,
        "     %s\n"
        , avx512.c_str()
      );
    }
#endif


#if defined(AVS_WINDOWS) && !defined(NO_WIN_GDI)
    // So far RECT dimensions were hardcoded: RECT r = { 32, 16, min(3440,vi.width * 8), 900*2 };
    // More flexible way: get text extent
    RECT r;

#if 0
    if(false && !font_override)
    {
        // To prevent slowish full MxN rendering, we calculate a dummy
        // 1xN sized vertical and a Mx1 sized horizontal line extent
        // Assuming that we are using fixed font (e.g. default Courier New)
        std::string s = text;
        size_t n = std::count(s.begin(), s.end(), '\n');
        // create dummy vertical string
        std::string s_vert;
        for (size_t i=0; i<n; i++) s_vert += " \n";
        RECT r0_v = { 0, 0, 100, 100 };
        DrawText(hdcAntialias, s_vert.c_str(), -1, &r0_v, DT_CALCRECT);
        // create dummy horizontal
        int counter = 0; int max_line = -1;
        int len = s.length();
        for (int i = 0; i < len; i++) // get length of longest line
        {
            if(s[i] != '\n') counter++;
            if(s[i] == '\n' || i == len - 1) {
                if(counter > max_line) max_line = counter;
                counter = 0;
            }
        }
        std::string s_horiz = std::string(max_line > 0 ? max_line : 1, ' '); // M*spaces
        RECT r0_h = { 0, 0, 100, 100 }; // for output
        DrawText(hdcAntialias, s_horiz.c_str(), -1, &r0_h, DT_CALCRECT);
        // and use the width and height dimensions from the two results
        r = { 32, 16, min(32+(int)r0_h.right,vi.width * 8-1), min(16+int(r0_v.bottom), vi.height*8-1) }; // do not crop if larger font is used
    } else
#endif
    {
        // font was overridden, may not be fixed type
        RECT r0 = { 0, 0, 100, 100 }; // do not crop if larger font is used
        DrawText(hdcAntialias, text, -1, &r0, DT_CALCRECT);
        r = { 32, 16, min(32+(int)r0.right,vi.width * 8 -1), min(16+int(r0.bottom), vi.height*8-1) };
    }

    // RECT r = { 32, 16, min(3440,vi.width * 8), 900*2 };
    // original code. Values possibly experimented Courier New size 18 + knowing max. text length/line count

    DrawText(hdcAntialias, text, -1, &r, 0);
    GdiFlush();

    env->MakeWritable(&frame);
    frame->GetWritePtr(); // Bump sequence_number
    int dst_pitch = frame->GetPitch();
    antialiaser.Apply(vi, &frame, dst_pitch);
#else
    env->MakeWritable(&frame);
    frame->GetWritePtr(); // Bump sequence_number

    // AVS_POSIX: utf8 is always true
    bool utf8 = false;
    // converting to wchar_t either from utf8 (win/linux) or ansi (win)
    std::wstring ws = charToWstring(text, utf8);

    int align = 7;
    int lsp = 0;
    int x = 4;
    int y = 2;

    SimpleTextOutW_multi(current_font.get(), vi, frame, x, y, ws, false, text_color, halo_color, true, align, lsp);
#endif
  return frame;
}

AVSValue __cdecl FilterInfo::Create(AVSValue args, void*, IScriptEnvironment* env)
{
    // 0   1      2       3             4
    // c[font]s[size]f[text_color]i[halo_color]i
    PClip clip = args[0].AsClip();
    // new parameters 20160823
#if defined(AVS_WINDOWS) && !defined(NO_WIN_GDI)
    const char* font = args[1].AsString("Courier New");
    int size = int(args[2].AsFloat(0) * 8 + 0.5);
    const int upper_limit = -1; // no limit
#else
    const char* font = args[1].AsString("Terminus");
    int size = int(args[2].AsFloat(0));
    const int upper_limit = 32; // max terminus font size is 32
#endif
    if (!args[2].Defined() || size < 0)
      size = CalcFontSizeForInfo(clip->GetVideoInfo().width, clip->GetVideoInfo().height, size < 0, upper_limit);
    // size <= 0 means autolarge over 640x480. Fixed font: up to size 32
    const int text_color = args[3].AsInt(0xFFFF00);
    const int halo_color = args[4].AsInt(0);

    return new FilterInfo(clip, font, size, text_color, halo_color, env);
}



/************************************
 *******    Compare Filter    *******
 ***********************************/

Compare::Compare(PClip _child1, PClip _child2, const char* channels, const char *fname, bool _show_graph, IScriptEnvironment* env)
  : GenericVideoFilter(_child1),
#if defined(AVS_WINDOWS) && !defined(NO_WIN_GDI)
  antialiaser(vi.width, vi.height, "Courier New", 16 * 8,
    (vi.IsYUV() || vi.IsYUVA()) ? 0xD21092 : 0xFFFF00,
    (vi.IsYUV() || vi.IsYUVA()) ? 0x108080 : 0),
#endif
  child2(_child2),
  log(nullptr),
  show_graph(_show_graph),
  framecount(0),
  text_color((vi.IsYUV() || vi.IsYUVA()) ? 0xD21092 : 0xFFFF00),
  halo_color((vi.IsYUV() || vi.IsYUVA()) ? 0x108080 : 0)
{
  const VideoInfo& vi2 = child2->GetVideoInfo();
  psnrs = 0;

  if (!vi.IsSameColorspace(vi2))
    env->ThrowError("Compare: Clips are not same colorspace.");

  if (vi.width != vi2.width || vi.height != vi2.height)
    env->ThrowError("Compare: Clips must have same size.");

  if (!(vi.IsRGB24() || vi.IsYUY2() || vi.IsRGB32() || vi.IsPlanar() || vi.IsRGB48() || vi.IsRGB64()))
    env->ThrowError("Compare: Clips have unknown pixel format. RGB24/32/48/64, YUY2 and YUV/RGB Planar supported.");

  pixelsize = vi.ComponentSize();
  bits_per_pixel = vi.BitsPerComponent();

  if (pixelsize == 4)
      env->ThrowError("Compare: Float pixel format not supported.");

  if (channels[0] == 0) {
    if (vi.IsRGB())
      channels = "RGB";
    else if (vi.IsY())
      channels = "Y";
    else if (vi.IsYUV() || vi.IsYUVA())
      channels = "YUV";
    else env->ThrowError("Compare: Clips have unknown colorspace. RGB and YUV supported.");
  }

  planar_plane = 0;
  mask = 0;
  const size_t length = strlen(channels);
  for (size_t i = 0; i < length; i++) {
    if (vi.IsRGB() && !vi.IsPlanar()) {
      switch (channels[i]) {
      case 'b':
      case 'B': mask |= 0x000000ff; mask64 |= 0x000000000000ffffull; break;
      case 'g':
      case 'G': mask |= 0x0000ff00; mask64 |= 0x00000000ffff0000ull; break;
      case 'r':
      case 'R': mask |= 0x00ff0000; mask64 |= 0x0000ffff00000000ull; break;
      case 'a':
      case 'A': mask |= 0xff000000; mask64 |= 0xffff000000000000ull;  if (vi.IsRGB32() || vi.IsRGB64()) break; // else no alpha -> fall thru
      default: env->ThrowError("Compare: invalid channel: %c", channels[i]);
      }
      if (vi.IsRGB24() || vi.IsRGB48()) mask &= 0x00ffffff;   // no alpha channel in RGB24
    } else if (vi.IsPlanar()) {
        if(vi.IsYUV() || vi.IsYUVA()) {
          switch (channels[i]) {
          case 'y':
          case 'Y': mask |= 0xffffffff; planar_plane |= PLANAR_Y; break;
          case 'u':
          case 'U': mask |= 0xffffffff; planar_plane |= PLANAR_U; break;
          case 'v':
          case 'V': mask |= 0xffffffff; planar_plane |= PLANAR_V; break;
          case 'a':
          case 'A': mask |= 0xffffffff; planar_plane |= PLANAR_A; if (!vi.IsYUVA()) break;  // else no alpha -> fall thru
          default: env->ThrowError("Compare: invalid channel: %c", channels[i]);
          }
          if (vi.IsY() && ((planar_plane & PLANAR_U) || (planar_plane & PLANAR_V))) {
              env->ThrowError("Compare: invalid channel: %c for greyscale clip", channels[i]);
          }
        } else {
            // planar RGB, planar RGBA
            switch (channels[i]) {
            case 'r':
            case 'R': mask |= 0xffffffff; planar_plane |= PLANAR_R; break;
            case 'g':
            case 'G': mask |= 0xffffffff; planar_plane |= PLANAR_G; break;
            case 'b':
            case 'B': mask |= 0xffffffff; planar_plane |= PLANAR_B; break;
            case 'a':
            case 'A': mask |= 0xffffffff; planar_plane |= PLANAR_A; if (!vi.IsPlanarRGBA()) break;  // else no alpha -> fall thru
            default: env->ThrowError("Compare: invalid channel: %c", channels[i]);
            }
        }
    } else {  // YUY2
      switch (channels[i]) {
      case 'y':
      case 'Y': mask |= 0x00ff00ff; break;
      case 'u':
      case 'U': mask |= 0x0000ff00; break;
      case 'v':
      case 'V': mask |= 0xff000000; break;
      default: env->ThrowError("Compare: invalid channel: %c", channels[i]);
      }
    }
  }

  masked_bytes = 0;
  for (uint32_t temp = mask; temp != 0; temp >>=8)
    masked_bytes += (temp & 1);

  if (fname[0] != 0) {
    log = fopen(fname, "wt");
    if (log) {
      fprintf(log,"Comparing channel(s) %s\n\n",channels);
      fprintf(log,"           Mean               Max    Max             \n");
      fprintf(log,"         Absolute     Mean    Pos.   Neg.            \n");
      fprintf(log," Frame     Dev.       Dev.    Dev.   Dev.  PSNR (dB) \n");
      fprintf(log,"-----------------------------------------------------\n");
    } else
      env->ThrowError("Compare: unable to create file %s", fname);
  } else {
	  psnrs = new(std::nothrow) int[vi.num_frames];
    if (psnrs)
      for (int i = 0; i < vi.num_frames; i++)
        psnrs[i] = 0;
  }
#if defined(AVS_WINDOWS) && !defined(NO_WIN_GDI)
#else
  // internal font
  const bool bold = false;
  const int size = 16;
  current_font = GetBitmapFont(size, "Terminus", bold, false);
  if (current_font == nullptr)
  {
    // size
    current_font = GetBitmapFont(size, "", bold, false);
    if (current_font == nullptr)
      current_font = GetBitmapFont(size, "", !bold, false);
  }
#endif
}


Compare::~Compare()
{
  if (log) {
    fprintf(log,"\n\n\nTotal frames processed: %d\n\n", framecount);
    fprintf(log,"                           Minimum   Average   Maximum\n");
    fprintf(log,"Mean Absolute Deviation: %9.4f %9.4f %9.4f\n", MAD_min, MAD_tot/framecount, MAD_max);
    fprintf(log,"         Mean Deviation: %+9.4f %+9.4f %+9.4f\n", MD_min, MD_tot/framecount, MD_max);
    fprintf(log,"                   PSNR: %9.4f %9.4f %9.4f\n", PSNR_min, PSNR_tot/framecount, PSNR_max);
    double factor = (1 << bits_per_pixel) - 1;
    double PSNR_overall = 10.0 * log10(bytecount_overall * factor * factor / SSD_overall);
    fprintf(log,"           Overall PSNR: %9.4f\n", PSNR_overall);
    fclose(log);
  }
  delete[] psnrs;
}


AVSValue __cdecl Compare::Create(AVSValue args, void*, IScriptEnvironment *env)
{
  return new Compare( args[0].AsClip(),     // clip
            args[1].AsClip(),     // base clip
            args[2].AsString(""),   // channels
            args[3].AsString(""),   // logfile
            args[4].AsBool(true),   // show_graph
            env);
}

static void compare_planar_c(
    const BYTE * f1ptr, int pitch1,
    const BYTE * f2ptr, int pitch2,
    int rowsize, int height,
    int &SAD_sum, int &SD_sum, int &pos_D,  int &neg_D, double &SSD_sum)
{
    int row_SSD;

    for (int y = 0; y < height; y++) {
        row_SSD = 0;
        for (int x = 0; x < rowsize; x += 1) {
            int p1 = *(f1ptr + x);
            int p2 = *(f2ptr + x);
            int d0 = p1 - p2;
            SD_sum += d0;
            SAD_sum += abs(d0);
            row_SSD += d0 * d0;
            pos_D = max(pos_D, d0);
            neg_D = min(neg_D, d0);
        }
        SSD_sum += row_SSD;
        f1ptr += pitch1;
        f2ptr += pitch2;
    }
}

static void compare_planar_uint16_t_c(
    const BYTE * f1ptr8, int pitch1,
    const BYTE * f2ptr8, int pitch2,
    int rowsize, int height,
    int64_t&SAD_sum, int64_t &SD_sum, int &pos_D,  int &neg_D, double &SSD_sum)
{
    int64_t row_SSD;

    const uint16_t *f1ptr = reinterpret_cast<const uint16_t *>(f1ptr8);
    const uint16_t *f2ptr = reinterpret_cast<const uint16_t *>(f2ptr8);
    pitch1 /= sizeof(uint16_t);
    pitch2 /= sizeof(uint16_t);
    rowsize /= sizeof(uint16_t);


    for (int y = 0; y < height; y++) {
        row_SSD = 0;
        for (int x = 0; x < rowsize; x += 1) {
            int p1 = *(f1ptr + x);
            int p2 = *(f2ptr + x);
            int d0 = p1 - p2;
            SD_sum += d0;
            SAD_sum += abs(d0);
            row_SSD += d0 * d0;
            pos_D = max(pos_D, d0);
            neg_D = min(neg_D, d0);
        }
        SSD_sum += row_SSD;
        f1ptr += pitch1;
        f2ptr += pitch2;
    }
}


static void compare_c(uint32_t mask, int increment,
    const BYTE * f1ptr, int pitch1,
    const BYTE * f2ptr, int pitch2,
    int rowsize, int height,
    int &SAD_sum, int &SD_sum, int &pos_D,  int &neg_D, double &SSD_sum)
{
    int row_SSD;

    for (int y = 0; y < height; y++) {
        row_SSD = 0;
        for (int x = 0; x < rowsize; x += increment) {
            uint32_t p1 = *(uint32_t*)(f1ptr + x) & mask;
            uint32_t p2 = *(uint32_t*)(f2ptr + x) & mask;
            int d0 = (p1 & 0xff) - (p2 & 0xff);
            int d1 = ((p1 >> 8) & 0xff) - ((p2 & 0xff00) >> 8); // ?PF why not (p2 >> 8) & 0xff as for p1?
            int d2 = ((p1 >> 16) & 0xff) - ((p2 & 0xff0000) >> 16);
            int d3 = (p1 >> 24) - (p2 >> 24);
            SD_sum += d0 + d1 + d2 + d3;
            SAD_sum += abs(d0) + abs(d1) + abs(d2) + abs(d3);
            row_SSD += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
            pos_D = max(max(max(max(pos_D, d0), d1), d2), d3);
            neg_D = min(min(min(min(neg_D, d0), d1), d2), d3);
        }
        SSD_sum += row_SSD;
        f1ptr += pitch1;
        f2ptr += pitch2;
    }
}

static void compare_uint16_t_c(uint64_t mask64, int increment,
    const BYTE * f1ptr8, int pitch1,
    const BYTE * f2ptr8, int pitch2,
    int rowsize, int height,
    int64_t& SAD_sum, int64_t &SD_sum, int &pos_D, int &neg_D, double &SSD_sum)
{
    int64_t row_SSD;

    const uint16_t *f1ptr = reinterpret_cast<const uint16_t *>(f1ptr8);
    const uint16_t *f2ptr = reinterpret_cast<const uint16_t *>(f2ptr8);
    pitch1 /= sizeof(uint16_t);
    pitch2 /= sizeof(uint16_t);
    rowsize /= sizeof(uint16_t);

    for (int y = 0; y < height; y++) {
        row_SSD = 0;
        for (int x = 0; x < rowsize; x += increment) {
            uint64_t p1 = *(uint64_t *)(f1ptr + x) & mask64;
            uint64_t p2 = *(uint64_t *)(f2ptr + x) & mask64;
            int d0 = (p1 & 0xffff) - (p2 & 0xffff);
            int d1 = ((p1 >> 16) & 0xffff) - ((p2 & 0xffff0000) >> 16);     // ?PF why not (p2 >> 16) & 0xffff as for p1?
            int d2 = ((p1 >> 32) & 0xffff) - ((p2 & 0xffff00000000ull) >> 32);
            int d3 = (p1 >> 48) - (p2 >> 48);
            SD_sum += d0 + d1 + d2 + d3;
            SAD_sum += abs(d0) + abs(d1) + abs(d2) + abs(d3);
            row_SSD += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
            pos_D = max(max(max(max(pos_D, d0), d1), d2), d3);
            neg_D = min(min(min(min(neg_D, d0), d1), d2), d3);
        }
        SSD_sum += row_SSD;
        f1ptr += pitch1;
        f2ptr += pitch2;
    }
}



PVideoFrame __stdcall Compare::GetFrame(int n, IScriptEnvironment* env)
{
  PVideoFrame f1 = child->GetFrame(n, env);
  PVideoFrame f2 = child2->GetFrame(n, env);

  int SD = 0;
  int64_t SD_64 = 0;
  int SAD = 0;
  int64_t SAD_64 = 0;
  int pos_D = 0;
  int neg_D = 0;
  double SSD = 0;

  int bytecount = 0;

  const int incr = (vi.IsRGB24() || vi.IsRGB48()) ? 3 : 4;

  if (vi.IsRGB24() || vi.IsYUY2() || vi.IsRGB32() || vi.IsRGB48() || vi.IsRGB64()) {

    const BYTE* f1ptr = f1->GetReadPtr();
    const BYTE* f2ptr = f2->GetReadPtr();
    const int pitch1 = f1->GetPitch();
    const int pitch2 = f2->GetPitch();
    const int rowsize = f1->GetRowSize();
    const int height = f1->GetHeight();

    bytecount = (rowsize / pixelsize) * height * masked_bytes / 4;
#ifdef INTEL_INTRINSICS

    if (((vi.IsRGB32() && (rowsize % 16 == 0)) || (vi.IsRGB24() && (rowsize % 12 == 0)) || (vi.IsYUY2() && (rowsize % 16 == 0))) &&
      (pixelsize == 1) && (env->GetCPUFlags() & CPUF_SSE2)) // only for uint8_t (pixelsize==1), todo
    {

      compare_sse2(mask, incr, f1ptr, pitch1, f2ptr, pitch2, rowsize, height, SAD, SD, pos_D, neg_D, SSD);
    }
    else
#ifdef X86_32
      if (((vi.IsRGB32() && (rowsize % 8 == 0)) || (vi.IsRGB24() && (rowsize % 6 == 0)) || (vi.IsYUY2() && (rowsize % 8 == 0))) &&
        (pixelsize == 1) && (env->GetCPUFlags() & CPUF_INTEGER_SSE)) // only for uint8_t (pixelsize==1), todo
      {
        compare_isse(mask, incr, f1ptr, pitch1, f2ptr, pitch2, rowsize, height, SAD, SD, pos_D, neg_D, SSD);
      }
      else
#endif
#endif
      {

        if (pixelsize == 1)
          compare_c(mask, incr, f1ptr, pitch1, f2ptr, pitch2, rowsize, height, SAD, SD, pos_D, neg_D, SSD);
        else
          compare_uint16_t_c(mask64, incr, f1ptr, pitch1, f2ptr, pitch2, rowsize, height, SAD_64, SD_64, pos_D, neg_D, SSD);
      }
  }
  else { // Planar

    int planes_y[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
    int planes_r[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
    int* planes = (vi.IsYUV() || vi.IsYUVA()) ? planes_y : planes_r;
    for (int p = 0; p < 3; p++) {
      const int plane = planes[p];

      if (planar_plane & plane) {

        const BYTE* f1ptr = f1->GetReadPtr(plane);
        const BYTE* f2ptr = f2->GetReadPtr(plane);
        const int pitch1 = f1->GetPitch(plane);
        const int pitch2 = f2->GetPitch(plane);
        const int rowsize = f1->GetRowSize(plane);
        const int height = f1->GetHeight(plane);

        bytecount += (rowsize / pixelsize) * height;
#ifdef INTEL_INTRINSICS

        if ((pixelsize == 1) && (rowsize % 16 == 0) && (env->GetCPUFlags() & CPUF_SSE2))
        {
          compare_sse2(mask, incr, f1ptr, pitch1, f2ptr, pitch2, rowsize, height, SAD, SD, pos_D, neg_D, SSD);
        }
        else
#ifdef X86_32
          if ((pixelsize == 1) && (rowsize % 8 == 0) && (env->GetCPUFlags() & CPUF_INTEGER_SSE))
          {
            compare_isse(mask, incr, f1ptr, pitch1, f2ptr, pitch2, rowsize, height, SAD, SD, pos_D, neg_D, SSD);
          }
          else
#endif
#endif
          {

            if (pixelsize == 1)
              compare_planar_c(f1ptr, pitch1, f2ptr, pitch2, rowsize, height, SAD, SD, pos_D, neg_D, SSD);
            else
              compare_planar_uint16_t_c(f1ptr, pitch1, f2ptr, pitch2, rowsize, height, SAD_64, SD_64, pos_D, neg_D, SSD);
          }
      }
    }
  }

  double MAD = ((pixelsize==1) ? (double)SAD : (double)SAD_64) / bytecount;
  double MD = ((pixelsize==1) ? (double)SD : (double)SD_64) / bytecount;
  if (SSD == 0.0) SSD = 1.0;
  const int max_pixel_value = (1 << bits_per_pixel) - 1;
  double factor = (double)(max_pixel_value);
  double PSNR = 10.0 * log10(bytecount * factor * factor / SSD);

  framecount++;
  if (framecount == 1) {
    MAD_min = MAD_tot = MAD_max = MAD;
    MD_min = MD_tot = MD_max = MD;
    PSNR_min = PSNR_tot = PSNR_max = PSNR;
    bytecount_overall = double(bytecount);
    SSD_overall = SSD;
  } else {
    MAD_min = min(MAD_min, MAD);
    MAD_tot += MAD;
    MAD_max = max(MAD_max, MAD);
    MD_min = min(MD_min, MD);
    MD_tot += MD;
    MD_max = max(MD_max, MD);
    PSNR_min = min(PSNR_min, PSNR);
    PSNR_tot += PSNR;
    PSNR_max = max(PSNR_max, PSNR);
    bytecount_overall += double(bytecount);
    SSD_overall += SSD;
  }

  if (log) {
    if (pixelsize == 1)
      fprintf(log,"%6u  %8.4f  %+9.4f  %3d    %3d    %8.4f\n", (unsigned int)n, MAD, MD, pos_D, neg_D, PSNR);
    else
      fprintf(log,"%6u  %11.4f  %+12.4f  %7d    %7d    %8.4f\n", (unsigned int)n, MAD, MD, pos_D, neg_D, PSNR);
  } else {
    env->MakeWritable(&f1);
    BYTE* dstp = f1->GetWritePtr();
    int dst_pitch = f1->GetPitch();

#if defined(AVS_WINDOWS) && !defined(NO_WIN_GDI)
    HDC hdc = antialiaser.GetDC();
    if (hdc)
#else
    if(current_font != nullptr)
#endif
    {
        char text[600];
#if defined(AVS_WINDOWS) && !defined(NO_WIN_GDI)
        RECT r = { 32, 16, min((51+(pixelsize==1 ? 0: 12))*67,vi.width * 8), 768 + 128 }; // orig: 3440: 51*67, not enough for 16 bit data
#endif
        double PSNR_overall = 10.0 * log10(bytecount_overall * factor * factor / SSD_overall);
        if (pixelsize == 1)
            snprintf(text, sizeof(text),
                "       Frame:  %-8u(   min  /   avg  /   max  )\n"
                "Mean Abs Dev:%8.4f  (%7.3f /%7.3f /%7.3f )\n"
                "    Mean Dev:%+8.4f  (%+7.3f /%+7.3f /%+7.3f )\n"
                " Max Pos Dev:%4d  \n"
                " Max Neg Dev:%4d  \n"
                "        PSNR:%6.2f dB ( %6.2f / %6.2f / %6.2f )\n"
                "Overall PSNR:%6.2f dB\n",
                n,
                MAD, MAD_min, MAD_tot / framecount, MD_max,
                MD, MD_min, MD_tot / framecount, MD_max,
                pos_D,
                neg_D,
                PSNR, PSNR_min, PSNR_tot / framecount, PSNR_max,
                PSNR_overall
            );
        else
            snprintf(text, sizeof(text),
                "       Frame:  %-8u   (     min   /     avg   /     max   )\n"
                "Mean Abs Dev:%11.4f  (%10.3f /%10.3f /%10.3f )\n"
                "    Mean Dev:%+11.4f  (%+10.3f /%+10.3f /%+10.3f )\n"
                " Max Pos Dev:%7d  \n"
                " Max Neg Dev:%7d  \n"
                "        PSNR:%6.2f dB    (   %6.2f  /   %6.2f  /   %6.2f  )\n"
                "Overall PSNR:%6.2f dB\n",
                n,
                MAD, MAD_min, MAD_tot / framecount, MD_max,
                MD, MD_min, MD_tot / framecount, MD_max,
                pos_D,
                neg_D,
                PSNR, PSNR_min, PSNR_tot / framecount, PSNR_max,
                PSNR_overall
            );
#if defined(AVS_WINDOWS) && !defined(NO_WIN_GDI)
        DrawText(hdc, text, -1, &r, 0);
        GdiFlush();

        antialiaser.Apply(vi, &f1, dst_pitch);
#else
        bool utf8 = true;
        auto ws = charToWstring(text, utf8);
        SimpleTextOutW_multi(current_font.get(), vi, f1, 2, 1, ws, true, text_color, halo_color, false, 0 /* no align */, 0 /*lsp*/);
#endif
    }

    if (show_graph) {
      // original idea by Marc_FD
      // PF remark: show-graph (and file logging) is not for multitask
      // psnrs array is instance specific
      psnrs[n] = min((int)(PSNR + 0.5), 100);
      if (vi.height > 196) {
        if (vi.IsYUY2()) {
          dstp += (vi.height - 1) * dst_pitch;
          for (int y = 0; y <= 100; y++) {
            for (int x = max(0, vi.width - n - 1); x < vi.width; x++) {
              if (y <= psnrs[n - vi.width + 1 + x]) {
                if (y <= psnrs[n - vi.width + 1 + x] - 2) {
                  dstp[x << 1] = 16;                // Y
                  dstp[((x & -1) << 1) + 1] = 0x80; // U
                  dstp[((x & -1) << 1) + 3] = 0x80; // V
                } else {
                  dstp[x << 1] = 235;               // Y
                  dstp[((x & -1) << 1) + 1] = 0x80; // U
                  dstp[((x & -1) << 1) + 3] = 0x80; // V
                }
              }
            } // for x
            dstp -= dst_pitch;
          } // for y
        }
		else if (vi.IsPlanar()) {
            if (vi.IsPlanarRGB() || vi.IsPlanarRGBA())
            {
                BYTE* dstp_RGBP[3] = { f1->GetWritePtr(PLANAR_G), f1->GetWritePtr(PLANAR_B),f1->GetWritePtr(PLANAR_R) };
                int dst_pitch_RGBP[3] = { f1->GetPitch(PLANAR_G), f1->GetPitch(PLANAR_B), f1->GetPitch(PLANAR_R) };

                dstp_RGBP[0] += (vi.height - 1) * dst_pitch_RGBP[0];
                dstp_RGBP[1] += (vi.height - 1) * dst_pitch_RGBP[1];
                dstp_RGBP[2] += (vi.height - 1) * dst_pitch_RGBP[2];
                for (int y = 0; y <= 100; y++) {
                    for (int x = max(0, vi.width - n - 1); x < vi.width; x++) {
                        if (y <= psnrs[n - vi.width + 1 + x]) {
                            if (y <= psnrs[n - vi.width + 1 + x] - 2) {
                                if(pixelsize==1) {
                                    dstp_RGBP[0][x] = 0;
                                    dstp_RGBP[1][x] = 0;
                                    dstp_RGBP[2][x] = 0;
                                } else {
                                    reinterpret_cast<uint16_t *>(dstp_RGBP[0])[x] = 0;
                                    reinterpret_cast<uint16_t *>(dstp_RGBP[1])[x] = 0;
                                    reinterpret_cast<uint16_t *>(dstp_RGBP[2])[x] = 0;
                                }
                            } else {
                                if(pixelsize==1) {
                                    dstp_RGBP[0][x] = 0xFF;
                                    dstp_RGBP[1][x] = 0xFF;
                                    dstp_RGBP[2][x] = 0xFF;
                                } else {
                                    reinterpret_cast<uint16_t *>(dstp_RGBP[0])[x] = max_pixel_value;
                                    reinterpret_cast<uint16_t *>(dstp_RGBP[1])[x] = max_pixel_value;
                                    reinterpret_cast<uint16_t *>(dstp_RGBP[2])[x] = max_pixel_value;
                                }
                            }
                        }
                    } // for x
                    dstp_RGBP[0] -= dst_pitch_RGBP[0];
                    dstp_RGBP[1] -= dst_pitch_RGBP[1];
                    dstp_RGBP[2] -= dst_pitch_RGBP[2];
                }
            } else {
                // planar YUV
                dstp += (vi.height - 1) * dst_pitch;
                const int black = 16 << (bits_per_pixel - 8);
                const int white = 235 << (bits_per_pixel - 8);
                for (int y = 0; y <= 100; y++) {
                    for (int x = max(0, vi.width - n - 1); x < vi.width; x++) {
                        if (y <= psnrs[n - vi.width + 1 + x]) {
                            if (y <= psnrs[n - vi.width + 1 + x] - 2) {
                                if(pixelsize==1)
                                    dstp[x] = 16; // Y
                                else
                                    reinterpret_cast<uint16_t *>(dstp)[x] = black; // Y
                            } else {
                                if(pixelsize==1)
                                    dstp[x] = 235; // Y
                                else
                                    reinterpret_cast<uint16_t *>(dstp)[x] = white; // Y
                            }
                        }
                    } // for x
                    dstp -= dst_pitch;
            }
          } // for y
        } else {  // packed RGB 8 or 16 bits
          for (int y = 0; y <= 100; y++) {
            for (int x = max(0, vi.width - n - 1); x < vi.width; x++) {
              if (y <= psnrs[n - vi.width + 1 + x]) {
                const int xx = x * incr;
                if (y <= psnrs[n - vi.width + 1 + x] -2) {
                    if(pixelsize==1) {
                      dstp[xx] = 0x00;        // B
                      dstp[xx + 1] = 0x00;    // G
                      dstp[xx + 2] = 0x00;    // R
                    }
                    else {
                      reinterpret_cast<uint16_t *>(dstp)[xx] = 0x00;        // B
                      reinterpret_cast<uint16_t *>(dstp)[xx + 1] = 0x00;    // G
                      reinterpret_cast<uint16_t *>(dstp)[xx + 2] = 0x00;    // R
                    }
                } else {
                    if(pixelsize==1) {
                        dstp[xx] = 0xFF;        // B
                        dstp[xx + 1] = 0xFF;    // G
                        dstp[xx + 2] = 0xFF;    // R
                    }
                    else {
                        reinterpret_cast<uint16_t *>(dstp)[xx] = 0xFFFF;        // B
                        reinterpret_cast<uint16_t *>(dstp)[xx + 1] = 0xFFFF;    // G
                        reinterpret_cast<uint16_t *>(dstp)[xx + 2] = 0xFFFF;    // R
                    }
                }
              }
            } // for x
            dstp += dst_pitch;
          } // for y
        } // RGB
      } // height > 100
    } // show_graph
  } // no logfile

  return f1;
}











/************************************
 *******   Helper Functions    ******
 ***********************************/
#if defined(AVS_WINDOWS) && !defined(NO_WIN_GDI)
bool GetTextBoundingBox( const char* text, const char* fontname, int size, bool bold,
                         bool italic, int align, int* width, int* height )
{
  HFONT hfont = LoadFont(fontname, size, bold, italic);
  if (hfont == NULL)
    return false;
  HDC hdc = GetDC(NULL);
  if (hdc == NULL)
	return false;
  HFONT hfontDefault = (HFONT)SelectObject(hdc, hfont);
  int old_map_mode = SetMapMode(hdc, MM_TEXT);
  UINT old_text_align = SetTextAlign(hdc, align);

  *height = *width = 8;
  bool success = true;
  RECT r = { 0, 0, 0, 0 };
  for (;;) {
    const char* nl = strchr(text, '\n');
    if (nl-text) {
      success &= !!DrawText(hdc, text, nl ? int(nl-text) : (int)lstrlen(text), &r, DT_CALCRECT | DT_NOPREFIX);
      *width = max(*width, int(r.right+8));
    }
    *height += r.bottom;
    if (nl) {
      text = nl+1;
      if (*text)
        continue;
      else
        break;
    } else {
      break;
    }
  }

  SetTextAlign(hdc, old_text_align);
  SetMapMode(hdc, old_map_mode);
  SelectObject(hdc, hfontDefault);
  DeleteObject(hfont);
  ReleaseDC(NULL, hdc);

  return success;
}
#endif

bool GetTextBoundingBoxFixed(const char* text, const char* fontname, int size, bool bold,
  bool italic, int align, int& width, int& height, bool utf8)
{
  std::unique_ptr<BitmapFont> current_font;
  /*
  if (*font_filename) {
    // external font file
    const bool debugSave = false;
    current_font = GetBitmapFont(0, font_filename, false, debugSave); // 0: size n/a
    if (current_font == nullptr)
      env->ThrowError("SimpleText: file %s not found or unknown file format", font_filename);
  }
  else
  */
  {
    // internal font
    if (fontname) {
      current_font = GetBitmapFont(size, fontname, bold, false); // 12, "Terminus"
      if (current_font == nullptr)
        return false; // ("internal font name %s in size %d not found", fontname, size
    }
    else {
      // size
      current_font = GetBitmapFont(size, "", bold, false); // 12, ""
      if (current_font == nullptr)
        return false; // "fixed font size %d not found", size
    }
  }

  auto s16 = charToWstring(text, utf8);
  size_t max_width = 1;
  height = 1;

  // make list governed by LF separator
  using wstringstream = std::basic_stringstream<wchar_t>;
  std::wstring temp;
  wstringstream wss(s16);
  while (std::getline(wss, temp, L'\n')) {
    max_width = std::max(max_width, temp.size() * current_font->width);
    height += current_font->height;
  }

  width = (int)max_width;

  return true;
}


void ApplyMessage( PVideoFrame* frame, const VideoInfo& vi, const char* message, int size,
                   int textcolor, int halocolor, int bgcolor, IScriptEnvironment* env )
{
  AVS_UNUSED(bgcolor);
  AVS_UNUSED(env);
  if (vi.IsYUV() || vi.IsYUVA()) {
    textcolor = RGB2YUV_Rec601(textcolor);
    halocolor = RGB2YUV_Rec601(halocolor);
  }

#if defined(AVS_WINDOWS) && !defined(NO_WIN_GDI)
  Antialiaser antialiaser(vi.width, vi.height, "Arial", size, textcolor, halocolor);
  HDC hdcAntialias = antialiaser.GetDC();
  if  (hdcAntialias)
  {
	RECT r = { 4*8, 4*8, vi.width*8, vi.height*8 };
	DrawText(hdcAntialias, message, lstrlen(message), &r, DT_NOPREFIX|DT_CENTER);
	GdiFlush();
	antialiaser.Apply(vi, frame, (*frame)->GetPitch());
  }
#else
  std::unique_ptr<BitmapFont> current_font;

  size = size / 8; // size comes in GDI units (*8)

  // internal font
  const bool bold = true; // bold terminus has better visibility
  current_font = GetBitmapFont(size, "Terminus", bold, false);
  if (current_font == nullptr)
  {
    // size
    current_font = GetBitmapFont(size, "", bold, false);
    if (current_font == nullptr)
      current_font = GetBitmapFont(size, "", !bold, false);
    if (current_font == nullptr)
      return;
  }

  //env->MakeWritable(&frame);
  //frame->GetWritePtr(); // Bump sequence_number

  // AVS_POSIX: utf8 is always true
  bool utf8 = false; // fixme: true for new utf8-avs+
  // converting to wchar_t either from utf8 (win/linux) or ansi (win)
  std::wstring ws = charToWstring(message, utf8);

  int align = 7;
  int lsp = 0;
  int x = 4;
  int y = 4;

  SimpleTextOutW_multi(current_font.get(), vi, *frame, x, y, ws, false, textcolor, halocolor, true, align, lsp);

#endif
}
