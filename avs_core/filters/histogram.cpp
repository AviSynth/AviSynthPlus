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

// FIXME: in general: how to display 32 bit floats?
// Do we have to assume it as if it is used after a limited -> full scale conversion? (preferred - everywhere!)
// Or with values simply: pixel_8bit / 255.0?
// Latter logic converts U=240 to (240-128)/255 instead of (240-128)/226 (=0.5)
// and Y=235 to 235/255 instead of (235-16)/219 (=1.0)

#include "histogram.h"
#include "../core/info.h"
#include "../core/internal.h"
#include "../convert/convert_planar.h"
#include "../convert/convert_audio.h"
#include "../convert/convert_helper.h"
#include "../convert/convert_matrix.h"
#include "colorbars_const.h"

#ifdef AVS_WINDOWS
    #include <avs/win.h>
#else
    #include <avs/posix.h>
#endif

#include <memory>
#include <avs/minmax.h>
#include <cstdio>
#include <cmath>
#include <stdint.h>
#include <vector>


constexpr double PI = 3.14159265358979323846; 
// until c++20 <numbers> std::numbers::pi

/********************************************************************
***** Declare index of new filters for Avisynth's filter engine *****
********************************************************************/
extern const AVSFunction Histogram_filters[] = {
  { "Histogram", BUILTIN_FUNC_PREFIX, "c[mode]s[factor]f[bits]i[keepsource]b[markers]b[matrix]s[graticule]s[targets]b[axes]b[iq]b[iq_lines]b[circle]b[targets100]b", Histogram::Create },
  { 0 }
};

/***********************************
 *******   Histogram Filter   ******
 **********************************/

Histogram::Histogram(PClip _child, Mode _mode, AVSValue _option, int _show_bits, bool _keepsource, bool _markers, const char* _matrix_name, histogram_color2_params _color2_params, IScriptEnvironment* env)
  : GenericVideoFilter(_child), mode(_mode), option(_option), show_bits(_show_bits), keepsource(_keepsource), markers(_markers), color2_params(_color2_params)
{
  bool optionValid = false;

  pixelsize = vi.ComponentSize();
  bits_per_pixel = vi.BitsPerComponent();

  if(show_bits < 8 || show_bits>12)
    env->ThrowError("Histogram: bits parameter can only be 8, 9 .. 12");

  // until all histogram is ported
  bool non8bit = show_bits != 8 || bits_per_pixel != 8;

  if (non8bit && mode != ModeClassic && mode != ModeLevels && mode != ModeColor && mode != ModeColor2 && mode != ModeLuma)
  {
    env->ThrowError("Histogram: this histogram type is available only for 8 bit formats and parameters");
  }

  // Obtain matrix (for YUV) and video range
  auto frame0 = child->GetFrame(0, env);
  const AVSMap* props = env->getFramePropsRO(frame0);
  // default YUV matrix AVS_MATRIX_ST170_M (BT 601)
  const bool rgb_in = vi.IsRGB();
  matrix_parse_merge_with_props(rgb_in, rgb_in /* rgb_out n/a */ , _matrix_name, props, theMatrix, theColorRange, theOutColorRange, env);

  // theColorRange is either inherited
  // - from defaults (RGB full YUV limited) or
  // - _ColorRange frame property or
  // - manual override from "matrix" param ":f" or ":l" designator
  // theOutColorRange is only used in yuv-rgb converters
  full_range = theColorRange == ColorRange_Compat_e::AVS_COLORRANGE_FULL;
  // Full range or limited?
  // VectorScope modes (color, color2) require full range to
  // - properly place the color boxes
  // - When the histogram's virtual bit-size is different from the real video bit-depth
  //   then video-bit depth must be converted to the histogram's bit-depth either by
  //   limited (shift) or full range (stretch) method.

  const int shift = 13; // not used here, we only get the floating point matrix
  // use matrix coeffs in color/color2 for positioning RGB color boxes for the actual YUV matrix.
  // Detected input color range is theColorRange, detected matrix is theMatrix.
  if (!do_BuildMatrix_Rgb2Yuv(theMatrix, theColorRange, theOutColorRange, shift, 32, /*ref*/matrix))
    env->ThrowError("Histogram: Unsupported matrix.");

  origwidth = vi.width;
  origheight = vi.height;

  if (mode == ModeClassic) {
    if (!vi.IsYUV() && !vi.IsYUVA())
      env->ThrowError("Histogram: YUV(A) data only");
    if(keepsource)
      vi.width += (1 << show_bits);
    else
      vi.width = (1 << show_bits);
    ClassicLUTInit();
  }

  if (mode == ModeLevels) {
    if (!vi.IsPlanar()) {
      env->ThrowError("Histogram: Levels mode only available in PLANAR.");
    }
    optionValid = option.IsFloat();
    const double factor = option.AsDblDef(100.0); // Population limit % factor
    if (factor < 0.0 || factor > 100.0) {
      env->ThrowError("Histogram: Levels population clamping must be between 0 and 100%");
    }
    // put diagram on the right side
    if (keepsource) {
      vi.width += (1 << show_bits); // 256 for 8 bit
      vi.height = max(256, vi.height);
    }
    else { // or keep it alone
      vi.width = (1 << show_bits);
      vi.height = 256; // only 224+1 (3*64 + 2*16 + 1) is used
    }
  }

  if (mode == ModeColor || mode == ModeColor2) {
    if (vi.IsRGB()) {
      env->ThrowError("Histogram: VectorScope modes (color, color2) are not available in RGB.");
    }
    if (!vi.IsPlanar()) {
      env->ThrowError("Histogram: VectorScope modes (color, color2) only available in PLANAR.");
    }
    if (vi.IsY()) {
      env->ThrowError("Histogram: VectorScope modes (color, color2) are not available in greyscale.");
    }
    // put diagram on the right side
    if (keepsource) {
      vi.width += (1 << show_bits); // 256 for 8 bit
      vi.height = max(1 << show_bits, vi.height);
    }
    else {
      vi.width = (1 << show_bits); // 256 for 8 bit
      vi.height = 1 << show_bits;
    }

    // for params.circle == true
    // precalculate 15 degree marker dots
    if (color2_params.circle) {
      const int half = (1 << (show_bits - 1)) - 1; // 127
      // dots are placed somewhat inside to the colorful circle, which is thicker for higher show_bits
      color2_innerF = 124.9;  // .9 is for better visuals in subsampled mode
      int R = (int)(1 + color2_innerF * (1 << (show_bits - 8)) + 0.5); // 126 for 8 bits

      for (int y = 0; y < 24; y++) { // just inside the big circle
        deg15c[y] = (int)(R * cos(y * PI / 12.) + 0.5) + half;
        deg15s[y] = (int)(-R * sin(y * PI / 12.) + 0.5) + half;
      }
    }
  }

  if (mode == ModeLuma && !vi.IsYUV() && !vi.IsYUVA()) {
      env->ThrowError("Histogram: Luma mode only available in YUV(A).");
  }

  if ((mode == ModeStereoY8)||(mode == ModeStereo)||(mode == ModeOverlay)) {

    child->SetCacheHints(CACHE_AUDIO,4096*1024);

    if (!vi.HasVideo()) {
      mode = ModeStereo; // force mode to ModeStereo.
      vi.fps_numerator = 25;
      vi.fps_denominator = 1;
      vi.num_frames = vi.FramesFromAudioSamples(vi.num_audio_samples);
    }
    if (mode == ModeOverlay)  {
      if (keepsource) {
        vi.height = max(512, vi.height);
        vi.width = max(512, vi.width);
      }
      else {
        vi.height = 512;
        vi.width = 512;
      }
      if (vi.IsRGB()) {
        env->ThrowError("Histogram: StereoOverlay mode is not available in RGB.");
      }
      if (!vi.IsPlanar()) {
        env->ThrowError("Histogram: StereoOverlay only available in Y or YUV(A).");
      }
    } else if (mode == ModeStereoY8) {
      vi.pixel_type = VideoInfo::CS_Y8;
      vi.height = 512;
      vi.width = 512;
    } else {
      vi.pixel_type = VideoInfo::CS_YV12;
      vi.height = 512;
      vi.width = 512;
    }
    if (!vi.HasAudio()) {
      env->ThrowError("Histogram: Stereo mode requires samples!");
    }
    if (vi.AudioChannels() != 2) {
      env->ThrowError("Histogram: Stereo mode only works on two audio channels.");
    }

     aud_clip = ConvertAudio::Create(child,SAMPLE_INT16,SAMPLE_INT16);
  }

  if (mode == ModeAudioLevels) {
    child->SetCacheHints(CACHE_AUDIO, 4096*1024);
    if (vi.IsRGB()) {
      env->ThrowError("Histogram: Audiolevels mode is not available in RGB.");
    }
    if (!vi.IsPlanar()) {
      env->ThrowError("Histogram: Audiolevels mode only available in planar YUV.");
    }
    if (vi.IsY8()) {
      env->ThrowError("Histogram: AudioLevels mode not available in Y8.");
    }

    aud_clip = ConvertAudio::Create(child, SAMPLE_INT16, SAMPLE_INT16);
  }

  if (!optionValid && option.Defined())
    env->ThrowError("Histogram: Unknown optional value.");
}

PVideoFrame __stdcall Histogram::GetFrame(int n, IScriptEnvironment* env)
{
  switch (mode) {
  case ModeClassic:
    return DrawModeClassic(n, env);
  case ModeLevels:
    return DrawModeLevels(n, env);
  case ModeColor:
    return DrawModeColor(n, env);
  case ModeColor2:
    return DrawModeColor2(n, env);
  case ModeLuma:
    return DrawModeLuma(n, env);
  case ModeStereoY8:
  case ModeStereo:
    return DrawModeStereo(n, env);
  case ModeOverlay:
    return DrawModeOverlay(n, env);
  case ModeAudioLevels:
    return DrawModeAudioLevels(n, env);
  }
  return DrawModeClassic(n, env);
}

inline void MixLuma(BYTE &src, int value, int alpha) {
  src = src + BYTE(((value - (int)src) * alpha) >> 8);
}

PVideoFrame Histogram::DrawModeAudioLevels(int n, IScriptEnvironment* env) {
  PVideoFrame src = child->GetFrame(n, env);
  env->MakeWritable(&src);
  const int w = src->GetRowSize();
  const int channels = vi.AudioChannels();

  constexpr int TEXT_HEIGHT = 20; // for DrawString
  int bar_w = 60;  // Must be divideable by 4 (for subsampling)
  int total_width = (1+channels*2)*bar_w; // Total width in pixels.

  if (total_width > w) {
    bar_w = ((w / (1+channels*2)) / 4)* 4;
  }
  total_width = (1+channels*2)*bar_w; // Total width in pixels.
  int bar_h = vi.height;

  // Get audio for current frame.
  const int64_t start = vi.AudioSamplesFromFrames(n);
  const int count = (int)(vi.AudioSamplesFromFrames(1));
  signed short* samples = static_cast<signed short*>(_alloca(sizeof(signed short)* count * channels));

  aud_clip->GetAudio(samples, max((int64_t)0ll,start), count, env);

  // Find maximum volume and rms.
  int*     channel_max = static_cast<int*>(_alloca(channels * sizeof(int)));
  int64_t* channel_rms = static_cast<int64_t*>(_alloca(channels * sizeof(int64_t)));;

  const int c = count*channels;
  for (int ch = 0; ch<channels; ch++) {
    int max_vol = 0;
    int64_t rms_vol = 0;

    for (int i = ch; i < c; i += channels) {
      int sample = samples[i];
      sample *= sample;
      rms_vol += sample;
      max_vol = max(max_vol, sample);
    }
    channel_max[ch] = max_vol;
    channel_rms[ch] = rms_vol;
  }

  // Draw bars
  BYTE* srcpY = src->GetWritePtr(PLANAR_Y);
  int Ypitch = src->GetPitch(PLANAR_Y);
  BYTE* srcpU = src->GetWritePtr(PLANAR_U);
  BYTE* srcpV = src->GetWritePtr(PLANAR_V);
  int UVpitch = src->GetPitch(PLANAR_U);
  int xSubS = vi.GetPlaneWidthSubsampling(PLANAR_U);
  int ySubS = vi.GetPlaneHeightSubsampling(PLANAR_U);

  // Draw Dotted lines
  const int lines = 16;  // Line every 6dB  (96/6)
  int lines_y[lines];
  float line_every = (float)bar_h / (float)lines;
  char text[32];
  for (int i=0; i<lines; i++) {
    lines_y[i] = (int)(line_every*i);
    if (!(i&1)) {
      snprintf(text, sizeof(text), "%3ddB", -i*6);
      DrawStringPlanar(vi, src, 0, i ? lines_y[i] : TEXT_HEIGHT / 2, text);
    }
  }
  for (int x=bar_w-16; x<total_width-bar_w+16; x++) {
    if (!(x&12)) {
      for (int i=0; i<lines; i++) {
        srcpY[x+lines_y[i]*Ypitch] = 200;
      }
    }
  }

  for (int ch = 0; ch<channels; ch++) {
    int max = channel_max[ch];
    double ch_db = 96;
    if (max > 0) {
      ch_db = -8.685889638/2.0 * log((double)max/(32768.0*32768.0));
    }

    int64_t rms = channel_rms[ch] / count;
    double ch_rms = 96;
    if (rms > 0) {
      ch_rms = -8.685889638/2.0 * log((double)rms/(32768.0*32768.0));
    }

    int x_pos = ((ch*2)+1)*bar_w+8;
    int x_end = x_pos+bar_w-8;
    int y_pos = (int)(((double)bar_h*ch_db) / 96.0);
    int y_mid = (int)(((double)bar_h*ch_rms) / 96.0);
    int y_end = src->GetHeight(PLANAR_Y);
    // Luma                          Red   Blue
    int y_val = (max>=32767*32767) ? 78 : 90;
    int a_val = (max>=32767*32767) ? 96 : 128;
    for (int y = y_pos; y<y_mid; y++) {
      for (int x = x_pos; x < x_end; x++) {
        MixLuma(srcpY[x+y*Ypitch], y_val, a_val);
      }
    } //                      Yellow Green
    y_val = (max>=32767*32767) ? 216 : 137;
    a_val = (max>=32767*32767) ? 160 : 128;
    for (int y = y_mid; y<y_end; y++) {
      for (int x = x_pos; x < x_end; x++) {
        MixLuma(srcpY[x+y*Ypitch], y_val, a_val);
      }
    }
    // Chroma
    x_pos >>= xSubS;
    x_end >>= xSubS;
    y_pos >>= ySubS;
    y_mid >>= ySubS;
    y_end = src->GetHeight(PLANAR_U);//Red  Blue
    BYTE u_val = (max>=32767*32767) ? 92 : 212;
    BYTE v_val = (max>=32767*32767) ? 233 : 114;
    for (int y = y_pos; y<y_mid; y++) {
      for (int x = x_pos; x < x_end; x++) {
        srcpU[x+y*UVpitch] = u_val;
        srcpV[x+y*UVpitch] = v_val;
      }
    } //                      Yellow Green
    u_val = (max>=32767*32767) ? 44 : 58;
    v_val = (max>=32767*32767) ? 142 : 40;
    for (int y = y_mid; y<y_end; y++) {
      for (int x = x_pos; x < x_end; x++) {
        srcpU[x+y*UVpitch] = u_val;
        srcpV[x+y*UVpitch] = v_val;
      }
    }
    // Draw text
    snprintf(text, sizeof(text), "%6.2fdB", (float)-ch_db);
    DrawStringPlanar(vi, src, ((ch*2)+1)*bar_w, vi.height- 2 * TEXT_HEIGHT + TEXT_HEIGHT / 2, text);
    snprintf(text, sizeof(text), "%6.2fdB", (float)-ch_rms);
    DrawStringPlanar(vi, src, ((ch*2)+1)*bar_w, vi.height- 1 * TEXT_HEIGHT + TEXT_HEIGHT / 2, text);

  }

  return src;
}

PVideoFrame Histogram::DrawModeOverlay(int n, IScriptEnvironment* env) {
  PVideoFrame src = child->GetFrame(n, env);
  PVideoFrame dst = env->NewVideoFrameP(vi, &src);

  int64_t start = vi.AudioSamplesFromFrames(n);
  int64_t end = vi.AudioSamplesFromFrames(n+1);
  int64_t count = end-start;
  signed short* samples = static_cast<signed short*>(
    env->Allocate((int)count * vi.AudioChannels() * sizeof(unsigned short), 8, AVS_POOLED_ALLOC)
  );
  if (!samples) {
    env->ThrowError("Histogram: Could not reserve memory.");
  }

  int h = dst->GetHeight();
  int imgSize = h*dst->GetPitch();
  BYTE* dstp = dst->GetWritePtr();
  int p = dst->GetPitch(PLANAR_Y);

  if ((src->GetHeight()<dst->GetHeight()) || (src->GetRowSize() < dst->GetRowSize())) {
    memset(dstp, 16, imgSize);
    int imgSizeU = dst->GetHeight(PLANAR_U) * dst->GetPitch(PLANAR_U);
    if (imgSizeU) {
      memset(dst->GetWritePtr(PLANAR_U), 128, imgSizeU);
      memset(dst->GetWritePtr(PLANAR_V), 128, imgSizeU);
    }
  }

  env->BitBlt(dstp, dst->GetPitch(), src->GetReadPtr(), src->GetPitch(), src->GetRowSize(), src->GetHeight());
  env->BitBlt(dst->GetWritePtr(PLANAR_U), dst->GetPitch(PLANAR_U), src->GetReadPtr(PLANAR_U), src->GetPitch(PLANAR_U), src->GetRowSize(PLANAR_U), src->GetHeight(PLANAR_U));
  env->BitBlt(dst->GetWritePtr(PLANAR_V), dst->GetPitch(PLANAR_V), src->GetReadPtr(PLANAR_V), src->GetPitch(PLANAR_V), src->GetRowSize(PLANAR_V), src->GetHeight(PLANAR_V));

  BYTE* _dstp = dstp;
  for (int iY = 0; iY<512; iY++) {
    for (int iX = 0; iX<512; iX++) {
      _dstp[iX] >>= 1;
    }
    _dstp+=p;
  }

  aud_clip->GetAudio(samples, max((int64_t)0ll,start), count, env);

  int c = (int)count;
  for (int i=1; i < c;i++) {
    int l1 = samples[i*2-2];
    int r1 = samples[i*2-1];
    int l2 = samples[i*2];
    int r2 = samples[i*2+1];
    for (int s = 0 ; s < 8; s++) {  // 8 times supersampling (linear)
      int l = (l1*s) + (l2*(8-s));
      int r = (r1*s) + (r2*(8-s));
      int y = 256+((l+r)>>11);
      int x = 256+((l-r)>>11);
      BYTE v = dstp[x+y*p]+48;
      dstp[x+y*p] = min(v,(BYTE)235);
    }
  }

  int y_off = p*256;
  for (int x = 0; x < 512; x+=16)
    dstp[y_off + x] = (dstp[y_off + x] > 127) ? 16 : 235;

  for (int y = 0; y < 512;y+=16)
    dstp[y*p+256] = (dstp[y*p+256]>127) ? 16 : 235 ;

  env->Free(samples);
  return dst;
}


PVideoFrame Histogram::DrawModeStereo(int n, IScriptEnvironment* env) {
  PVideoFrame src = env->NewVideoFrame(vi);
  int64_t start = vi.AudioSamplesFromFrames(n);
  int64_t end = vi.AudioSamplesFromFrames(n+1);
  int64_t count = end-start;
  signed short* samples = static_cast<signed short*>(
    env->Allocate((int)count * vi.AudioChannels() * sizeof(unsigned short), 8, AVS_POOLED_ALLOC)
  );
  if (!samples) {
    env->ThrowError("Histogram: Could not reserve memory.");
  }

  int h = src->GetHeight();
  int imgSize = h*src->GetPitch();
  BYTE* srcp = src->GetWritePtr();
  memset(srcp, 16, imgSize);
  int p = src->GetPitch();

  aud_clip->GetAudio(samples, max((int64_t)0ll,start), count, env);

  int c = (int)count;
  for (int i=1; i < c;i++) {
    int l1 = samples[i*2-2];
    int r1 = samples[i*2-1];
    int l2 = samples[i*2];
    int r2 = samples[i*2+1];
    for (int s = 0 ; s < 8; s++) {  // 8 times supersampling (linear)
      int l = (l1*s) + (l2*(8-s));
      int r = (r1*s) + (r2*(8-s));
      int y = 256+((l+r)>>11);
      int x = 256+((l-r)>>11);
      BYTE v = srcp[x+y*512]+48;
      srcp[x+y*512] = min(v, (BYTE)235);
    }
  }

  int y_off = p*256;
  for (int x = 0; x < 512; x+=16)
    srcp[y_off + x] = (srcp[y_off + x] > 127) ? 16 : 235;

  for (int y = 0; y < 512;y+=16)
    srcp[y*p+256] = (srcp[y*p+256]>127) ? 16 : 235 ;

  if (vi.IsYV12()) {
    srcp = src->GetWritePtr(PLANAR_U);
    imgSize = src->GetHeight(PLANAR_U) * src->GetPitch(PLANAR_U);
    memset(srcp, 128, imgSize);
    srcp = src->GetWritePtr(PLANAR_V);
    memset(srcp, 128, imgSize);
  }

  env->Free(samples);
  return src;
}


PVideoFrame Histogram::DrawModeLuma(int n, IScriptEnvironment* env) {
  // amplify luminance.
  // In this mode a 1 pixel luminance difference (8 bits world) will show as 
  // a 16 pixel luminance, thus seriously enhancing small flaws
  PVideoFrame src = child->GetFrame(n, env);
  env->MakeWritable(&src);
  const int h = src->GetHeight();
  BYTE* srcp = src->GetWritePtr();
  if (vi.IsYUY2()) {
    int imgsize = h * src->GetPitch();
    for (int i=0; i<imgsize; i+=2) {
      int p = srcp[i];
      p<<=4;
      srcp[i+1] = 128;
      srcp[i] = BYTE((p&256) ? (255-(p&0xff)) : p&0xff);
    }
  } else {
    const int w = vi.width;
    const int pitch = src->GetPitch();
    if (bits_per_pixel == 8) {
      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          const int pixel = srcp[x] << 4; // *16
          srcp[x] = BYTE((pixel & 256) ? (255 - (pixel & 0xff)) : pixel & 0xff);
        }
        srcp += pitch;
      }
    }
    else if (bits_per_pixel <= 16) {
      const int overlimit = (1 << bits_per_pixel);
      const int max_pixel_value = (1 << bits_per_pixel) - 1;
      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          const int pixel = reinterpret_cast<uint16_t*>(srcp)[x] << 4;
          reinterpret_cast<uint16_t*>(srcp)[x] = (uint16_t)((pixel & overlimit) ? (max_pixel_value - (pixel & max_pixel_value)) : pixel & max_pixel_value);
        }
        srcp += pitch;
      }
    }
    else {
      // 32 bit float
      // just simulated by converting to 8 bits
      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          const float pixel_f = reinterpret_cast<float*>(srcp)[x];
          const int pixel_i = (int)(pixel_f * 255.0f + 0.5f);
          const int pixel = pixel_i << 4; // *16
          int pixel_out = BYTE((pixel & 256) ? (255 - (pixel & 0xff)) : pixel & 0xff);
          const float pixel_out_f = pixel_out / 255.0f;
          reinterpret_cast<float*>(srcp)[x] = pixel_out_f;
        }
        srcp += pitch;
      }
    }
    
    if (!vi.IsY() && !vi.IsYA()) { // has real chroma
      auto dstp_u = src->GetWritePtr(PLANAR_U);
      auto dstp_v = src->GetWritePtr(PLANAR_V);
      auto height_uv = src->GetHeight(PLANAR_U);
      auto rowsize_uv = src->GetRowSize(PLANAR_U);
      auto pitch_uv = src->GetPitch(PLANAR_U);
      if (bits_per_pixel == 8)
        fill_chroma<uint8_t>(dstp_u, dstp_v, height_uv, rowsize_uv, pitch_uv, 128);
      else if (bits_per_pixel <= 16)
        fill_chroma<uint16_t>(dstp_u, dstp_v, height_uv, rowsize_uv, pitch_uv, 128 << (bits_per_pixel - 8));
      else // 32)
        fill_chroma<float>(dstp_u, dstp_v, height_uv, rowsize_uv, pitch_uv, 0.0f);
    }

    // alpha, IsYUVA includes IsYA
    if (vi.IsYUVA() || vi.IsPlanarRGBA()) {
      auto dstp_a = src->GetWritePtr(PLANAR_A);
      auto height_a = src->GetHeight(PLANAR_A);
      auto rowsize_a = src->GetRowSize(PLANAR_A);
      auto pitch_a = src->GetPitch(PLANAR_A);
      if (bits_per_pixel == 8)
        fill_plane<uint8_t>(dstp_a, height_a, rowsize_a, pitch_a, 255);
      else if (bits_per_pixel <= 16)
        fill_plane<uint16_t>(dstp_a, height_a, rowsize_a, pitch_a, (1 << bits_per_pixel) -  1);
      else // 32)
        fill_plane<float>(dstp_a, height_a, rowsize_a, pitch_a, 1.0f);
    }
  }
  return src;
}

template<typename pixel_t>
static void DrawModeColor2_DrawRect(
  pixel_t* dstp, int pitch,
  pixel_t* dstpU, pixel_t* dstpV, int pitchUV,
  int cx, int cy,          // center in luma coords
  int half_w, int half_h,  // half-size in luma coords
  int swidth, int sheight,
  pixel_t luma_val,
  pixel_t u_val, pixel_t v_val,
  int limit_showwidth // bounds check
)
{
  // Draw horizontal top/bottom edges (luma)
  for (int x = cx - half_w; x <= cx + half_w; x++) {
    if (x < 0 || x >= limit_showwidth) continue;
    int y_top = cy - half_h;
    int y_bot = cy + half_h;
    if (y_top >= 0 && y_top < limit_showwidth)
      dstp[x + y_top * pitch] = luma_val;
    if (y_bot >= 0 && y_bot < limit_showwidth)
      dstp[x + y_bot * pitch] = luma_val;
  }

  // Draw vertical left/right edges (luma)
  for (int y = cy - half_h; y <= cy + half_h; y++) {
    if (y < 0 || y >= limit_showwidth) continue;
    int x_l = cx - half_w;
    int x_r = cx + half_w;
    if (x_l >= 0 && x_l < limit_showwidth)
      dstp[x_l + y * pitch] = luma_val;
    if (x_r >= 0 && x_r < limit_showwidth)
      dstp[x_r + y * pitch] = luma_val;
  }

  // Chroma planes (subsampled)
  const int limit_showwidth_uv = limit_showwidth >> swidth;
  const int limit_showheight_uv = limit_showwidth >> sheight;

  // Convert luma coordinates to chroma coordinates
  const int cx_uv = cx >> swidth;
  const int cy_uv = cy >> sheight;
  const int half_w_uv = half_w >> swidth;
  const int half_h_uv = half_h >> sheight;

  // Draw horizontal top/bottom edges (chroma)
  for (int x = cx_uv - half_w_uv; x <= cx_uv + half_w_uv; x++) {
    if (x < 0 || x >= limit_showwidth_uv) continue;
    int y_top = cy_uv - half_h_uv;
    int y_bot = cy_uv + half_h_uv;
    if (y_top >= 0 && y_top < limit_showheight_uv) {
      dstpU[x + y_top * pitchUV] = u_val;
      dstpV[x + y_top * pitchUV] = v_val;
    }
    if (y_bot >= 0 && y_bot < limit_showheight_uv) {
      dstpU[x + y_bot * pitchUV] = u_val;
      dstpV[x + y_bot * pitchUV] = v_val;
    }
  }

  // Draw vertical left/right edges (chroma)
  for (int y = cy_uv - half_h_uv; y <= cy_uv + half_h_uv; y++) {
    if (y < 0 || y >= limit_showheight_uv) continue;
    int x_l = cx_uv - half_w_uv;
    int x_r = cx_uv + half_w_uv;
    if (x_l >= 0 && x_l < limit_showwidth_uv) {
      dstpU[x_l + y * pitchUV] = u_val;
      dstpV[x_l + y * pitchUV] = v_val;
    }
    if (x_r >= 0 && x_r < limit_showwidth_uv) {
      dstpU[x_r + y * pitchUV] = u_val;
      dstpV[x_r + y * pitchUV] = v_val;
    }
  }
}


// This draws only on luma
template<typename pixel_t>
static void DrawRadialLine(pixel_t* dstp, int pitch, int limit, int show_bit_shift,
  double angle_deg, pixel_t val)
{
  double angle_rad = angle_deg * M_PI / 180.0;
  double R = 124.9 * (1 << show_bit_shift); // same as innerF * scale
  // Step along the radius
  int steps = (int)(R * 1.5);
  for (int s = 0; s < steps; s++) {
    double t = (double)s / steps;
    int x = (int)(limit + t * R * cos(angle_rad) + 0.5);
    int y = (int)(limit - t * R * sin(angle_rad) + 0.5); // V is flipped
    if (x >= 0 && x <= 2 * limit && y >= 0 && y <= 2 * limit)
      dstp[x + y * pitch] = val;
  }
}

// Set to black, alpha to 0.
template<typename pixel_t>
static void ClearArea(
  uint8_t* dstp, uint8_t* dstp_u, uint8_t* dstp_v, uint8_t* dstp_a,
  int width, int widthUV,
  int pitch, int pitchUV, int pitchA,
  int height, int heightUV, int heightA,
  int bits_per_pixel, bool full_range)
{
  pixel_t black, middle_chroma, alpha;
  if constexpr (std::is_integral<pixel_t>::value) {
    black = full_range ? 0 : (pixel_t)(16 << (bits_per_pixel - 8));
    middle_chroma = (pixel_t)(128 << (bits_per_pixel - 8));
    alpha = 0;
  }
  else {
    black = full_range ? 0.0f : 16.0f / 255.0f;
    middle_chroma = 0.0f;
    alpha = 0.0f;
  }

  // Y
  for (int y = 0; y < height; y++) {
    std::fill_n((pixel_t*)(dstp + y * pitch), width, black);
  }
  // UV
  for (int y = 0; y < heightUV; y++) {
    std::fill_n((pixel_t*)(dstp_u + y * pitchUV), widthUV, middle_chroma);
    std::fill_n((pixel_t*)(dstp_v + y * pitchUV), widthUV, middle_chroma);
  }
  // Alpha, dimensions same as luma
  // heightA is zero if no alpha plane
  for (int y = 0; y < heightA; y++) {
    std::fill_n((pixel_t*)(dstp_a + y * pitchA), width, alpha);
  }
}

// Common prelude for Color and Color2 modes.
// Allocates dst, optionally clears the below-source area, copies source planes.
// Returns the allocated dst frame; populates panel pointer offsets.ű
// Full_range flag is filled in constructor, not per frame
PVideoFrame Histogram::VectorscopePrelude(
  int n, IScriptEnvironment* env,
  PVideoFrame& src,
  // outputs:
  int& dst_pitch, int& dst_height,
  int& dst_pitchUV, int& dst_heightUV,
  int& dst_pitchA, int& dst_heightA,
  BYTE*& panel, BYTE*& panelU, BYTE*& panelV, BYTE*& panelA)
{
  src = child->GetFrame(n, env);
  PVideoFrame dst = env->NewVideoFrameP(vi, &src);

  BYTE *pdst = dst->GetWritePtr();
  BYTE *pdstU = dst->GetWritePtr(PLANAR_U);
  BYTE *pdstV = dst->GetWritePtr(PLANAR_V);
  BYTE *pdstA = dst->GetWritePtr(PLANAR_A);

  dst_pitch = dst->GetPitch();
  dst_pitchUV = dst->GetPitch(PLANAR_U);
  dst_pitchA = dst->GetPitch(PLANAR_A);
  dst_height = dst->GetHeight();
  dst_heightUV = dst->GetHeight(PLANAR_U);
  dst_heightA = dst->GetHeight(PLANAR_A);

  const bool has_alpha = dst_heightA > 0;

  const int src_width = src->GetRowSize() / pixelsize;
  const int src_widthUV = src->GetRowSize(PLANAR_U) / pixelsize;
  const int src_height = src->GetHeight();
  const int src_heightUV = src->GetHeight(PLANAR_U);

  const int show_size = 1 << show_bits;
  const int show_size_w_uv = show_size >> vi.GetPlaneWidthSubsampling(PLANAR_U);
  const int show_size_h_uv = show_size >> vi.GetPlaneHeightSubsampling(PLANAR_U);

  if (keepsource) {

    if (src_height < dst_height) {
      // in case of Histogram area is higher than source, clear the area below source copy to black + A=0
      auto pdst_start = pdst + src_height * dst_pitch;
      auto pdstU_start = pdstU + src_heightUV * dst_pitchUV;
      auto pdstV_start = pdstV + src_heightUV * dst_pitchUV;
      auto pdstA_start = pdstA + src_height * dst_pitchA;
      const int new_height = dst_height - src_height;
      const int new_height_uv = dst_heightUV - src_heightUV;
      const int new_height_a = has_alpha ? new_height : 0;

      if (bits_per_pixel == 8)
        ClearArea<uint8_t>(pdst_start, pdstU_start, pdstV_start, pdstA_start,
          src_width, src_widthUV,
          dst_pitch, dst_pitchUV, dst_pitchA,
          new_height, new_height_uv, new_height_a,
          bits_per_pixel, full_range);
      else if (bits_per_pixel <= 16)
        ClearArea<uint16_t>(pdst_start, pdstU_start, pdstV_start, pdstA_start,
          src_width, src_widthUV,
          dst_pitch, dst_pitchUV, dst_pitchA,
          new_height, new_height_uv, new_height_a,
          bits_per_pixel, full_range);
      else
        ClearArea<float>(pdst_start, pdstU_start, pdstV_start, pdstA_start,
          src_width, src_widthUV,
          dst_pitch, dst_pitchUV, dst_pitchA,
          new_height, new_height_uv, new_height_a,
          bits_per_pixel, full_range);
    }
    // copy source
    env->BitBlt(pdst, dst_pitch, src->GetReadPtr(), src->GetPitch(), src->GetRowSize(), src->GetHeight());
    env->BitBlt(pdstU, dst_pitchUV, src->GetReadPtr(PLANAR_U), src->GetPitch(PLANAR_U), src->GetRowSize(PLANAR_U), src->GetHeight(PLANAR_U));
    env->BitBlt(pdstV, dst_pitchUV, src->GetReadPtr(PLANAR_V), src->GetPitch(PLANAR_V), src->GetRowSize(PLANAR_V), src->GetHeight(PLANAR_V));
    if (has_alpha)
      env->BitBlt(pdstA, dst_pitchA, src->GetReadPtr(PLANAR_A), src->GetPitch(PLANAR_A), src->GetRowSize(PLANAR_A), src->GetHeight(PLANAR_A));
  }

  // panel is with the offset into the histogram panel area (past the source copy)
  panel = pdst + (keepsource ? src->GetRowSize() : 0);
  panelU = pdstU + (keepsource ? src->GetRowSize(PLANAR_U) : 0);
  panelV = pdstV + (keepsource ? src->GetRowSize(PLANAR_V) : 0);
  panelA = (pdstA && src->GetReadPtr(PLANAR_A)) ? (pdstA + (keepsource ? src->GetRowSize(PLANAR_A) : 0)) : nullptr;

  // Clear below the histogram area (bottom right), if source is higher than histogram area.
  if (src_height > show_size) {
    auto pdst_start = panel + show_size * dst_pitch;
    auto pdstU_start = panelU + show_size_h_uv * dst_pitchUV;
    auto pdstV_start = panelV + show_size_h_uv * dst_pitchUV;
    auto pdstA_start = has_alpha ? panelA + show_size * dst_pitchA : nullptr;
    const int new_height = dst_height - show_size;
    const int new_height_uv = dst_heightUV - show_size_h_uv;
    const int new_height_a = has_alpha ? new_height : 0;

    if (bits_per_pixel == 8)
      ClearArea<uint8_t>(pdst_start, pdstU_start, pdstV_start, pdstA_start,
        show_size, show_size_w_uv,
        dst_pitch, dst_pitchUV, dst_pitchA,
        new_height, new_height_uv, new_height_a,
        bits_per_pixel, full_range);
    else if (bits_per_pixel <= 16)
      ClearArea<uint16_t>(pdst_start, pdstU_start, pdstV_start, pdstA_start,
        show_size, show_size_w_uv,
        dst_pitch, dst_pitchUV, dst_pitchA,
        new_height, new_height_uv, new_height_a,
        bits_per_pixel, full_range);
    else
      ClearArea<float>(pdst_start, pdstU_start, pdstV_start, pdstA_start,
        show_size, show_size_w_uv,
        dst_pitch, dst_pitchUV, dst_pitchA,
        new_height, new_height_uv, new_height_a,
        bits_per_pixel, full_range);
  }

  return dst;
}

static void GetYUVFromMatrix(
  int matrix,                          // AVS_MATRIX_* enum value
  double R, double G, double B,
  double& dY, double& dU, double& dV)
{
  double Kr, Kb;
  if (!GetKrKb(matrix, Kr, Kb)) {
    // fallback to BT.601
    Kr = 0.299; Kb = 0.114;
  }
  double Kg = 1.0 - Kr - Kb;
  dY = Kr * R + Kg * G + Kb * B;
  dU = (B - dY) / (2.0 * (1.0 - Kb));
  dV = (R - dY) / (2.0 * (1.0 - Kr));
}

template<typename pixel_t>
static void DrawModeColor2_ClearVectorscopeArea(int bits_per_pixel,
  uint8_t* dstp8, uint8_t* dstp8_u, uint8_t* dstp8_v, uint8_t* dstp8_a,
  int pitch, int pitchUV, int pitchA,
  int height, int heightUV, int heightA,
  int show_bits, int swidth, int sheight,
  bool full_range
)
{
  const int show_size = (1 << show_bits);
  const int show_size_w_uv = show_size >> swidth;
  const int show_size_h_uv = show_size >> sheight;
  const bool has_alpha = heightA > 0;
  const int show_size_a = has_alpha ? show_size : 0;

  // Clear vectorscope area

  if (bits_per_pixel == 8)
    ClearArea<uint8_t>(dstp8, dstp8_u, dstp8_v, dstp8_a,
      show_size, show_size_w_uv, // width
      pitch, pitchUV, pitchA,
      show_size, show_size_h_uv, show_size_a, // height
      bits_per_pixel, full_range);
  else if (bits_per_pixel <= 16)
    ClearArea<uint16_t>(dstp8, dstp8_u, dstp8_v, dstp8_a,
      show_size, show_size_w_uv, // width
      pitch, pitchUV, pitchA,
      show_size, show_size_h_uv, show_size_a, // height
      bits_per_pixel, full_range);
  else
    ClearArea<float>(dstp8, dstp8_u, dstp8_v, dstp8_a,
      show_size, show_size_w_uv, // width
      pitch, pitchUV, pitchA,
      show_size, show_size_h_uv, show_size_a, // height
      bits_per_pixel, full_range);
}

template<typename pixel_t>
static void DrawModeColor2_draw_graticule(int bits_per_pixel, uint8_t* dstp8, int pitch, int height, int show_bits) {

  pitch /= sizeof(pixel_t);
  pixel_t* dstp = reinterpret_cast<pixel_t*>(dstp8);

  pixel_t luma128;

  if constexpr (std::is_integral<pixel_t>::value) {
    luma128 = (pixel_t)(128 << (bits_per_pixel - 8));
  }
  else {
    luma128 = c8tof(128);
  }

  const int show_bit_shift = show_bits - 8;

  //    16       240
  // 16  +---------+
  // 17  |         |
  // ..  |         |
  // 239 |         |
  // 240 +---------+

  // plot valid grey ccir601 square
  const int size = 1 + ((240 - 16) << show_bit_shift); // original 8 bit: 225 0+/-112
  std::fill_n(&dstp[(16 << show_bit_shift) * pitch + (16 << show_bit_shift)], size, luma128);
  std::fill_n(&dstp[(240 << show_bit_shift) * pitch + (16 << show_bit_shift)], size, luma128);

  // vertical lines left and right side
  for (int y = 1 + (16 << show_bit_shift); y < 240 << show_bit_shift; y++) {
    dstp[(16 << show_bit_shift) + y * pitch] = luma128;
    dstp[(240 << show_bit_shift) + y * pitch] = luma128;
  }
}

template<typename pixel_t>
static void DrawModeColor2_draw_circle(int bits_per_pixel, uint8_t* dstp8, uint8_t* dstp8_u, uint8_t* dstp8_v,
  int pitch, int pitchUV, int height, int heightUV, double innerF,
  int show_bits, int swidth, int sheight, int* deg15c, int* deg15s // precalculated array
)
{

  pitch /= sizeof(pixel_t);
  pitchUV /= sizeof(pixel_t);
  pixel_t* dstp = reinterpret_cast<pixel_t*>(dstp8);
  pixel_t* dstp_u = reinterpret_cast<pixel_t*>(dstp8_u);
  pixel_t* dstp_v = reinterpret_cast<pixel_t*>(dstp8_v);

  // possible to display >8 bit data stuffed into 8 bit size
  const int show_bit_shift = show_bits - 8;

  // plot circles

  // here we do not bother with matrix-correct color representation.

  // six hues in the color-wheel:
  // LC[3j,3j+1,3j+2], RC[3j,3j+1,3j+2] in YRange[j]+1 and YRange[j+1]
  int YRange[8] = { -1, 26, 104, 127, 191, 197, 248, 256 };
  // 2x green, 2x yellow, 3x red
  int LC[21] = {
    145, 54, 34,
    145, 54, 34,
    210, 16, 146,
    210, 16, 146,
    81, 90, 240,
    81, 90, 240,
    81, 90, 240
  };
  // cyan, 4x blue, magenta, red:
  int RC[21] = {
    170, 166, 16,
    41, 240, 110,
    41, 240, 110,
    41, 240, 110,
    41, 240, 110,
    106, 202, 222,
    81, 90, 240
  };
  float LC_f[21] = {
    c8tof(145), uv8tof(54), uv8tof(34),
    c8tof(145), uv8tof(54), uv8tof(34),
    c8tof(210), uv8tof(16), uv8tof(146),
    c8tof(210), uv8tof(16), uv8tof(146),
    c8tof(81), uv8tof(90), uv8tof(240),
    c8tof(81), uv8tof(90), uv8tof(240),
    c8tof(81), uv8tof(90), uv8tof(240)
  };
  // cyan, 4x blue, magenta, red:
  float RC_f[21] = {
    c8tof(170), uv8tof(166), uv8tof(16),
    c8tof(41), uv8tof(240), uv8tof(110),
    c8tof(41), uv8tof(240), uv8tof(110),
    c8tof(41), uv8tof(240), uv8tof(110),
    c8tof(41), uv8tof(240), uv8tof(110),
    c8tof(106), uv8tof(202), uv8tof(222),
    c8tof(81), uv8tof(90), uv8tof(240)
  };

  // example boundary of cyan and blue:
  // red = min(r,g,b), blue if g < 2/3 b, green if b < 2/3 g.
  // cyan between green and blue.
  // thus boundary of cyan and blue at (r,g,b) = (0,170,255), since 2/3*255 = 170.
  // => yuv = (127,190,47); hue = -52 degr; sat = 103
  // => u'v' = (207,27) (same hue, sat=128)
  // similar for the other hues.
  // luma

  // innerF is used for 15 degree markers as well
  double thicknessF = 1.5;
  double oneOverThicknessF = 1.0 / thicknessF;
  double outerF = innerF + thicknessF * 2.0;
  double centerF = innerF + thicknessF;
  int64_t innerSq = (int64_t)(innerF * innerF * (1 << (show_bit_shift * 2)));
  int64_t outerSq = (int64_t)(outerF * outerF * (1 << (show_bit_shift * 2)));
  int activeY = 0;
  int xRounder = (1 << swidth) / 2;
  int yRounder = (1 << sheight) / 2;

  const int limit = (1 << (show_bits - 1)) - 1;
  const int limit_showwidth = (1 << show_bits) - 1;
  for (int y = -limit; y < limit + 1; y++) {
    if (y + limit > YRange[activeY + 1] << show_bit_shift)
      activeY++;
    for (int x = -limit; x <= 0; x++) {
      int64_t distSq = x * x + y * y;
      if (distSq <= outerSq && distSq >= innerSq) {

        if constexpr (std::is_integral<pixel_t>::value) {
          const int factorshift = (bits_per_pixel - 8);
          const int MAXINTERP = 256;
          double dist = fabs(sqrt((double)distSq * (1.0 / (1 << (2 * show_bit_shift)))) - centerF);
          int interp = (int)(256.0f - (255.9f * (oneOverThicknessF * dist)));
          // 255.9 is to account for float inprecision, which could cause underflow.

          int xP = limit + x;
          int yP = limit + y;

          dstp[xP + yP * pitch] =
            (pixel_t)((interp * (LC[3 * activeY] << factorshift)) >> 8); // left upper half

          dstp[limit_showwidth - xP + yP * pitch] =
            (pixel_t)((interp * (RC[3 * activeY] << factorshift)) >> 8); // right upper half

          xP = (xP + xRounder) >> swidth;
          yP = (yP + yRounder) >> sheight;

          interp = min(MAXINTERP, interp);
          int invInt = (MAXINTERP - interp);

          int p_uv;
          p_uv = xP + yP * pitchUV;
          dstp_u[p_uv] = (pixel_t)((dstp_u[p_uv] * invInt + interp * (LC[3 * activeY + 1] << factorshift)) >> 8); // left half
          dstp_v[p_uv] = (pixel_t)((dstp_v[p_uv] * invInt + interp * (LC[3 * activeY + 2] << factorshift)) >> 8); // left half

          xP = ((limit_showwidth) >> swidth) - xP;
          p_uv = xP + yP * pitchUV;

          dstp_u[p_uv] = (pixel_t)((dstp_u[p_uv] * invInt + interp * (RC[3 * activeY + 1] << factorshift)) >> 8); // right half
          dstp_v[p_uv] = (pixel_t)((dstp_v[p_uv] * invInt + interp * (RC[3 * activeY + 2] << factorshift)) >> 8); // right half
        }
        else {
          // 32 bit float
          const float MAXINTERP = 1.0f;
          double dist = fabs(sqrt((double)distSq * (1.0 / (1 << (2 * show_bit_shift)))) - centerF);
          float interp = (float)(1.0 - 0.9999 * (oneOverThicknessF * dist));
          // 255.9 is to account for float inprecision, which could cause underflow.

          int xP = limit + x;
          int yP = limit + y;

          dstp[xP + yP * pitch] =
            (pixel_t)(interp * LC_f[3 * activeY]); // left upper half

          dstp[limit_showwidth - xP + yP * pitch] =
            (pixel_t)(interp * RC_f[3 * activeY]); // right upper half

          xP = (xP + xRounder) >> swidth;
          yP = (yP + yRounder) >> sheight;

          interp = min(MAXINTERP, interp);
          float invInt = (MAXINTERP - interp);

          int p_uv;
          p_uv = xP + yP * pitchUV;
          dstp_u[p_uv] = (pixel_t)(dstp_u[p_uv] * invInt + interp * LC_f[3 * activeY + 1]); // left half
          dstp_v[p_uv] = (pixel_t)(dstp_v[p_uv] * invInt + interp * LC_f[3 * activeY + 2]); // left half

          xP = ((limit_showwidth) >> swidth) - xP;

          p_uv = xP + yP * pitchUV;
          dstp_u[p_uv] = (pixel_t)(dstp_u[p_uv] * invInt + interp * RC_f[3 * activeY + 1]); // right half
          dstp_v[p_uv] = (pixel_t)(dstp_v[p_uv] * invInt + interp * RC_f[3 * activeY + 2]); // right half
        }
      }
    }
  }

  // and the 15 degree markers (same as innerF)
  // plot white 15 degree marks
  for (int i = 0; i < 24; i++) {
    if constexpr (sizeof(pixel_t) == 1)
      dstp[deg15c[i] + deg15s[i] * pitch] = 235; // 235: Y-maxluma
    else if constexpr (sizeof(pixel_t) == 2)
      dstp[deg15c[i] + deg15s[i] * pitch] = 235 << (bits_per_pixel - 8); // 235: Y-maxluma
    else // if constexpr (sizeof(pixel_t) == 4)
      dstp[deg15c[i] + deg15s[i] * pitch] = c8tof(235); // 235: Y-maxluma
  }

}


template<typename pixel_t>
static void do_vectorscope_color2(
  pixel_t* pdstb, pixel_t* pdstbU, pixel_t* pdstbV,
  const pixel_t* pY, const pixel_t* pU, const pixel_t* pV,
  int dst_pitch, int dst_pitchUV,
  int src_pitch, int src_pitchUV,
  int src_widthUV, int src_heightUV,
  int swidth, int sheight,
  int show_bits,
  int bits_per_pixel,
  bits_conv_constants& d_chroma,
  bool full_range
)
{
  // 32 bit float Vectorscope is simulated after a 16 bit conversion
  const int shift = bits_per_pixel == 32 ? (16 - show_bits) /*n/a*/ : (bits_per_pixel - show_bits);
  const int max_pos = (1 << show_bits) - 1;
  auto src_pitchY_chromacorr = (src_pitch << sheight);

  if constexpr (sizeof(pixel_t) == 4) {
    // ===== FLOAT PATH =====
    const float mul_factor = d_chroma.mul_factor;
    const float dst_offset = d_chroma.dst_offset;

    for (int y = 0; y < src_heightUV; y++) {
      for (int x = 0; x < src_widthUV; x++) {
        const float uval_f = max(-0.5f, min(0.5f, pU[x]));
        const float vval_f = max(-0.5f, min(0.5f, pV[x]));

        int uval_posindex = (int)(uval_f * mul_factor + dst_offset + 0.5f);
        int vval_posindex = (int)(vval_f * mul_factor + dst_offset + 0.5f);
        uval_posindex = max(0, min(max_pos, uval_posindex));
        vval_posindex = max(0, min(max_pos, vval_posindex));

        pdstb[uval_posindex + vval_posindex * dst_pitch] = pY[x << swidth];
        pdstbU[(uval_posindex >> swidth) + (vval_posindex >> sheight) * dst_pitchUV] = uval_f;
        pdstbV[(uval_posindex >> swidth) + (vval_posindex >> sheight) * dst_pitchUV] = vval_f;
      }
      pY += src_pitchY_chromacorr;
      pU += src_pitchUV;
      pV += src_pitchUV;
    }
  }
  else {
    // ===== INTEGER PATH =====
    const int max_value = (1 << bits_per_pixel) - 1;

    // Split into 3 optimized paths based on shift and range
    // When shift != 0, that is the dispay bit depth is different from source bit depth, we do conversion
    // - full range: use conversion constants
    // - limited range: either by downshifting or upshifting.
    // E.g. displaying 10 bit data on a 8 bit grid, or displaying 8 bit data on a 10 bit grid.
    // We need to convert the chroma values to the display bit depth, taking into account the limited/full range and the shift.

    if (shift == 0) {
      // ===== FAST PATH: No scaling needed =====
      for (int y = 0; y < src_heightUV; y++) {
        for (int x = 0; x < src_widthUV; x++) {
          const int uval = max(0, min(max_value, (int)pU[x]));
          const int vval = max(0, min(max_value, (int)pV[x]));

          pdstb[uval + vval * dst_pitch] = pY[x << swidth];
          pdstbU[(uval >> swidth) + (vval >> sheight) * dst_pitchUV] = (pixel_t)uval;
          pdstbV[(uval >> swidth) + (vval >> sheight) * dst_pitchUV] = (pixel_t)vval;
        }
        pY += src_pitchY_chromacorr;
        pU += src_pitchUV;
        pV += src_pitchUV;
      }
    }
    else if (full_range) {
      // ===== FULL RANGE PATH: Use conversion constants =====
      const float mul_factor = d_chroma.mul_factor;
      const int src_offset_i = d_chroma.src_offset_i;
      const float dst_offset = d_chroma.dst_offset;

      for (int y = 0; y < src_heightUV; y++) {
        for (int x = 0; x < src_widthUV; x++) {
          const int uval = max(0, min(max_value, (int)pU[x]));
          const int vval = max(0, min(max_value, (int)pV[x]));

          int uval_posindex = (int)((uval - src_offset_i) * mul_factor + dst_offset + 0.5f);
          int vval_posindex = (int)((vval - src_offset_i) * mul_factor + dst_offset + 0.5f);
          uval_posindex = max(0, min(max_pos, uval_posindex));
          vval_posindex = max(0, min(max_pos, vval_posindex));

          pdstb[uval_posindex + vval_posindex * dst_pitch] = pY[x << swidth];
          pdstbU[(uval_posindex >> swidth) + (vval_posindex >> sheight) * dst_pitchUV] = (pixel_t)uval;
          pdstbV[(uval_posindex >> swidth) + (vval_posindex >> sheight) * dst_pitchUV] = (pixel_t)vval;
        }
        pY += src_pitchY_chromacorr;
        pU += src_pitchUV;
        pV += src_pitchUV;
      }
    }
    else {
      // ===== LIMITED RANGE PATH: Simple bit shift =====
      if (shift > 0) {
        // Downshift (e.g., 10-bit -> 8-bit display)
        const int rounder = 1 << (shift - 1);
        for (int y = 0; y < src_heightUV; y++) {
          for (int x = 0; x < src_widthUV; x++) {
            const int uval = max(0, min(max_value, (int)pU[x]));
            const int vval = max(0, min(max_value, (int)pV[x]));

            const int uval_posindex = (uval + rounder) >> shift;
            const int vval_posindex = (vval + rounder) >> shift;

            pdstb[uval_posindex + vval_posindex * dst_pitch] = pY[x << swidth];
            pdstbU[(uval_posindex >> swidth) + (vval_posindex >> sheight) * dst_pitchUV] = (pixel_t)uval;
            pdstbV[(uval_posindex >> swidth) + (vval_posindex >> sheight) * dst_pitchUV] = (pixel_t)vval;
          }
          pY += src_pitchY_chromacorr;
          pU += src_pitchUV;
          pV += src_pitchUV;
        }
      }
      else {
        // Upshift (e.g., 8-bit -> 10-bit display)
        const int neg_shift = -shift;
        for (int y = 0; y < src_heightUV; y++) {
          for (int x = 0; x < src_widthUV; x++) {
            const int uval = max(0, min(max_value, (int)pU[x]));
            const int vval = max(0, min(max_value, (int)pV[x]));

            const int uval_posindex = uval << neg_shift;
            const int vval_posindex = vval << neg_shift;

            pdstb[uval_posindex + vval_posindex * dst_pitch] = pY[x << swidth];
            pdstbU[(uval_posindex >> swidth) + (vval_posindex >> sheight) * dst_pitchUV] = (pixel_t)uval;
            pdstbV[(uval_posindex >> swidth) + (vval_posindex >> sheight) * dst_pitchUV] = (pixel_t)vval;
          }
          pY += src_pitchY_chromacorr;
          pU += src_pitchUV;
          pV += src_pitchUV;
        }
      }
    }
  }
}

template<typename pixel_t>
static void Draw_VectorScope_circle_targets_axes(int bits_per_pixel,
  uint8_t* dstp8, uint8_t* dstp8_u, uint8_t* dstp8_v, int pitch, int pitchUV, int height, int heightUV,
  double innerF,
  int show_bits, int swidth, int sheight,
  histogram_color2_params& params,
  ConversionMatrix& conv_matrix,
  int matrix, // AVS_MATRIX_* enum value
  bits_conv_constants& conv_consts,
  bool full_range, int* deg15c, int* deg15s)
{

  if (params.circle) {
    // also the 15 deg marks points
    DrawModeColor2_draw_circle<pixel_t>(bits_per_pixel,
      dstp8, dstp8_u, dstp8_v,
      pitch, pitchUV,
      height, heightUV,
      innerF,
      show_bits, swidth, sheight,
      deg15c, deg15s
    );
  }

  // Used for the colorUV square positions, calculated from linear RGB
  bits_conv_constants conv_consts_fullrange_float_origin;
  get_bits_conv_constants(conv_consts_fullrange_float_origin, /*use_chroma=*/true,
    /*fulls=*/true, full_range,
    /*srcBitDepth=*/32, /*dstBitDepth=*/show_bits);

  const int limit_showwidth = (1 << show_bits) - 1;

  pitch /= sizeof(pixel_t);
  pitchUV /= sizeof(pixel_t);
  pixel_t* dstp = reinterpret_cast<pixel_t*>(dstp8);
  pixel_t* dstp_u = reinterpret_cast<pixel_t*>(dstp8_u);
  pixel_t* dstp_v = reinterpret_cast<pixel_t*>(dstp8_v);

  // ===== -I AND +Q SIGNAL RGB VALUES FOR VECTORSCOPE GRATICULE =====
  // These feed through the same matrix-aware make_yuv path as the color bar patches,
  // giving correct UV pixel coordinates for any matrix and any bit depth.
  // 
  // Note: -I and +Q are analog NTSC test signals that don't map cleanly to digital RGB/YUV.
  // We maintain two separate specifications for backward compatibility:
  //
  // RGB-NATIVE: Lifted to studio black (code 16), broadcast-safe
  //   Used when vectorscope processes RGB sources or when drawing RGB graticules
  //   -I: RGB(16, 90, 130) at 8-bit → produces Y≈77 after matrix conversion
  //   +Q: RGB(92, 16, 143) at 8-bit → produces Y≈63 after matrix conversion
  //
  // YUV-TARGETED: Zero-luma pure chroma definition  
  //   Used when vectorscope processes YUV sources
  //   -I: Converts to Y=16, Cb=158, Cr=95 (legacy ColorBars YUV output)
  //   +Q: Converts to Y=16, Cb=174, Cr=149 (legacy ColorBars YUV output)
  //   Contains out-of-range RGB components (super-blacks)
  //
  // Note 2: These will not lay exactly on UV's 33° and 123° degree lines, since those 
  // angles are defined in the original NTSC YIQ color space, which differs from the YUV 
  // space produced by BT.601/BT.709 matrix conversions.

  // this one is not used, this is YUV vectorscope, we cannot determine that the YUV was converted from
  // an RGB ColorBars.
  /*
  static const RGBEntry iq_rgb_native[] = {
      { "-I", MINUS_I_R,     MINUS_I_G,     MINUS_I_B     }, // RGB-native (studio black lift)
      { "+Q", PLUS_Q_R,      PLUS_Q_G,      PLUS_Q_B      }, // RGB-native (studio black lift)
      { "+I", PLUS_I_R,      PLUS_I_G,      PLUS_I_B      }, // n/a ColorBarsHD only, which is YUV-only
  };
  */

  // ColorBars ground truth linear RGB (0.75 amplitude, Rec. BT.801-1)
  // Top 2/3 bars: LtGrey, Yellow, Cyan, Green, Magenta, Red, Blue
  // (White/LtGrey sits at UV center, not a useful vectorscope target)
  struct RGBEntry { const char* name; double r, g, b; };
  static const RGBEntry bar_rgb[] = {
      { "Yellow",  0.75, 0.75, 0.0  },
      { "Cyan",    0.0,  0.75, 0.75 },
      { "Green",   0.0,  0.75, 0.0  },
      { "Magenta", 0.75, 0.0,  0.75 },
      { "Red",     0.75, 0.0,  0.0  },
      { "Blue",    0.0,  0.0,  0.75 },
  };
  static const RGBEntry bar100_rgb[] = {
      { "Yellow",  1.0, 1.0, 0.0  },
      { "Cyan",    0.0, 1.0, 1.0 },
      { "Green",   0.0, 1.0, 0.0  },
      { "Magenta", 1.0, 0.0, 1.0 },
      { "Red",     1.0, 0.0, 0.0  },
      { "Blue",    0.0, 0.0, 1.0 },
  };

  static const RGBEntry iq_rgb_yuv_targeted[] = {
      { "-I", MINUS_I_R_YUV, MINUS_I_G_YUV, MINUS_I_B_YUV }, // YUV-targeted (zero-luma)
      { "+Q", PLUS_Q_R_YUV,  PLUS_Q_G_YUV,  PLUS_Q_B_YUV  }, // YUV-targeted (zero-luma)
      { "+I", PLUS_I_R_YUV,  PLUS_I_G_YUV,  PLUS_I_B_YUV }, // ColorBarsHD only (same for both)
  };

  const int half_box = 4 * (1 << (show_bits - 8));

  // ---- Pixel position helper ----
  // Converts normalised chroma [-0.5..0.5] to vectorscope pixel coordinate.
  // Always from full range float, since the YUV squares are obtained from the linear RGB values.
  // Target is limited/full range aware.
  auto uv_to_px_float = [&](double uv_norm) -> int {
    return clamp(
      (int)((float)uv_norm * conv_consts_fullrange_float_origin.mul_factor + conv_consts_fullrange_float_origin.dst_offset + 0.5f),
      0, (1 << show_bits) - 1);
    };

  // Box color (brownish, visible on the wheel background) ----
  pixel_t box_luma, box_u, box_v;
  if constexpr (std::is_integral<pixel_t>::value) {
    // Tan/beige: Y=80, slightly orange/brown tint
    box_luma = (pixel_t)(80 << (bits_per_pixel - 8));
    box_u = (pixel_t)(140 << (bits_per_pixel - 8));
    box_v = (pixel_t)(120 << (bits_per_pixel - 8));
  }
  else {
    box_luma = c8tof(80);
    box_u = uv8tof(140);
    box_v = uv8tof(120);
  }

  if (params.targets) {

    // ---- Draw colorbar target boxes ----
    for (auto& e : bar_rgb) {
      double dY, dU, dV;
      GetYUVFromMatrix(matrix, e.r, e.g, e.b, dY, dU, dV);
      int cx;
      int cy;
      if constexpr (std::is_integral<pixel_t>::value) {
        cx = uv_to_px_float(dU);
        cy = uv_to_px_float(dV);
      }
      else {
        cx = uv_to_px_float(dU);
        cy = uv_to_px_float(dV);
      }

      DrawModeColor2_DrawRect<pixel_t>(
        dstp, pitch, dstp_u, dstp_v, pitchUV,
        cx, cy, half_box, half_box,
        swidth, sheight,
        box_luma, box_u, box_v,
        limit_showwidth);

    }
  }

  if (params.targets100) {

    // ---- Draw colorbar target boxes ----
    for (auto& e : bar100_rgb) {
      double dY, dU, dV;
      GetYUVFromMatrix(matrix, e.r, e.g, e.b, dY, dU, dV);
      int cx;
      int cy;
      if constexpr (std::is_integral<pixel_t>::value) {
        cx = uv_to_px_float(dU);
        cy = uv_to_px_float(dV);
      }
      else {
        cx = uv_to_px_float(dU);
        cy = uv_to_px_float(dV);
      }

      DrawModeColor2_DrawRect<pixel_t>(
        dstp, pitch, dstp_u, dstp_v, pitchUV,
        cx, cy, half_box, half_box,
        swidth, sheight,
        box_luma, box_u, box_v,
        limit_showwidth);

    }
  }

  if (params.iq) {
    // ---- Draw +Q and -I target boxes ----
    // These are matrix-independent (they are phase references, not primaries),
    // but still need the same pixel-coordinate conversion.
    for (auto& e : iq_rgb_yuv_targeted) {
      double dY, dU, dV;
      GetYUVFromMatrix(matrix, e.r, e.g, e.b, dY, dU, dV);
      int cx = uv_to_px_float(dU);
      int cy = uv_to_px_float(dV);
      // yuv.u and yuv.v are already the correct pixel_t values at the virtual target
      // (show) bit depth.

      DrawModeColor2_DrawRect<pixel_t>(
        dstp, pitch, dstp_u, dstp_v, pitchUV,
        cx, cy, half_box, half_box,
        swidth, sheight,
        box_luma, box_u, box_v,
        limit_showwidth);

    }
  }

  // ---- Radial lines ----
  pixel_t line_luma;
  if constexpr (std::is_integral<pixel_t>::value)
    line_luma = (pixel_t)(120 << (bits_per_pixel - 8));
  else
    line_luma = c8tof(120);

  // Angles to draw(matching PPro vectorscope) :
  // 0°, 90°, 180°, 270° — the axis cross
  // 33°, 123°, 213°, 303° — the ±I/±Q diagonals
  // knowing that in UV space these are not exactly at 33° and 123°

  if (params.axes) {
    const double angles90[] = {
        0.0, 90.0, 180.0, 270.0
    };
    const int limit = (1 << (show_bits - 1)) - 1;
    const int show_bit_shift = show_bits - 8;
    for (double ang : angles90)
      DrawRadialLine<pixel_t>(dstp, pitch,
        limit, show_bit_shift, ang, line_luma);
  }

  if (params.iq_lines) {
    const double anglesIQ[] = {
        33.0, 123.0, 213.0, 303.0
    };
    const int limit = (1 << (show_bits - 1)) - 1;
    const int show_bit_shift = show_bits - 8;
    for (double ang : anglesIQ)
      DrawRadialLine<pixel_t>(dstp, pitch,
        limit, show_bit_shift, ang, line_luma);
  }
}


PVideoFrame Histogram::DrawModeColor2(int n, IScriptEnvironment* env) {

  // all these will get filled by VectorscopePrelude, common for color and color2 modes
  PVideoFrame src;
  BYTE* panel, * panelU, * panelV, * panelA;
  int dst_pitch, dst_pitchUV, dst_pitchA, dst_height, dst_heightUV, dst_heightA;

  PVideoFrame dst = VectorscopePrelude(n, env,
    // outputs
    src,
    dst_pitch, dst_height,
    dst_pitchUV, dst_heightUV,
    dst_pitchA, dst_heightA,
    panel, panelU, panelV, panelA);

  bits_conv_constants d_chroma;
  get_bits_conv_constants(d_chroma, /*use_chroma=*/true,
    full_range, full_range,
    bits_per_pixel, /*dstBitDepth=*/show_bits);
  // d_chroma.mul_factor and d_chroma.dst_offset are now correct for
  // limited-limited or full-full depending on _ColorRange frame property.

  int swidth = vi.GetPlaneWidthSubsampling(PLANAR_U);
  int sheight = vi.GetPlaneHeightSubsampling(PLANAR_U);

  // Clear Vectorscope area (black background, init Alpha to 0 if present)
  if (bits_per_pixel == 8)
    DrawModeColor2_ClearVectorscopeArea<uint8_t>(bits_per_pixel,
      panel, panelU, panelV, panelA,
      dst_pitch, dst_pitchUV, dst_pitchA,
      dst_height, dst_heightUV, dst_heightA,
      show_bits, swidth, sheight,
      full_range
    );
  else if (bits_per_pixel <= 16)
    DrawModeColor2_ClearVectorscopeArea<uint16_t>(bits_per_pixel,
      panel, panelU, panelV, panelA,
      dst_pitch, dst_pitchUV, dst_pitchA,
      dst_height, dst_heightUV, dst_heightA,
      show_bits, swidth, sheight,
      full_range
    );
  else
    DrawModeColor2_ClearVectorscopeArea<float>(bits_per_pixel,
      panel, panelU, panelV, panelA,
      dst_pitch, dst_pitchUV, dst_pitchA,
      dst_height, dst_heightUV, dst_heightA,
      show_bits, swidth, sheight,
      full_range
    );

  // graticule
  // Valid chroma boundary square
  // always/never/only-when-limited
  // "color" mode's graticule is used for highlight danger zone.
  // "color2" mode's graticule is used for drawing chroma boundary rectangle.
  // "on" or "auto"+limited range
  const bool need_graticule = color2_params.graticule_type == histogram_color2_params::GRATICULE_ON ||
    (color2_params.graticule_type == histogram_color2_params::GRATICULE_AUTO && !full_range);

  if (need_graticule) {
    if (bits_per_pixel == 8)
      DrawModeColor2_draw_graticule<uint8_t>(bits_per_pixel, panel, dst_pitch, dst_height, show_bits);
    else if (bits_per_pixel <= 16)
      DrawModeColor2_draw_graticule<uint16_t>(bits_per_pixel, panel, dst_pitch, dst_height, show_bits);
    else
      DrawModeColor2_draw_graticule<float>(bits_per_pixel, panel, dst_pitch, dst_height, show_bits);
  }

  // others: 75% RGB and IQ targets, 15 degree marks, etc.
  if (bits_per_pixel == 8)
    Draw_VectorScope_circle_targets_axes<uint8_t>(bits_per_pixel,
      panel, panelU, panelV,
      dst_pitch, dst_pitchUV,
      dst_height, dst_heightUV,
      color2_innerF,
      show_bits, swidth, sheight,
      color2_params, matrix, theMatrix, d_chroma, full_range,
      deg15c, deg15s
      );
  else if (bits_per_pixel <= 16)
    Draw_VectorScope_circle_targets_axes<uint16_t>(bits_per_pixel,
      panel, panelU, panelV,
      dst_pitch, dst_pitchUV,
      dst_height, dst_heightUV,
      color2_innerF,
      show_bits, swidth, sheight,
      color2_params, matrix, theMatrix, d_chroma, full_range,
      deg15c, deg15s
      );
  else
    Draw_VectorScope_circle_targets_axes<float>(bits_per_pixel,
      panel, panelU, panelV,
      dst_pitch, dst_pitchUV,
      dst_height, dst_heightUV,
      color2_innerF,
      show_bits, swidth, sheight,
      color2_params, matrix, theMatrix, d_chroma, full_range,
      deg15c, deg15s
      );

  // plot vectorscope
  const int src_pitch = src->GetPitch(PLANAR_Y) / pixelsize;
  const int src_pitchUV = src->GetPitch(PLANAR_U) / pixelsize;
  const int src_widthUV = src->GetRowSize(PLANAR_U) / pixelsize;
  const int src_heightUV = src->GetHeight(PLANAR_U);

  const BYTE* pY = src->GetReadPtr(PLANAR_Y);
  const BYTE* pU = src->GetReadPtr(PLANAR_U);
  const BYTE* pV = src->GetReadPtr(PLANAR_V);

  dst_pitch /= pixelsize;
  dst_pitchUV /= pixelsize;

  if(bits_per_pixel == 8)
    do_vectorscope_color2<uint8_t>(
      panel, panelU, panelV,
      pY, pU, pV,
      dst_pitch, dst_pitchUV,
      src_pitch, src_pitchUV,
      src_widthUV, src_heightUV,
      swidth, sheight,
      show_bits, bits_per_pixel, d_chroma, full_range);
  else if (bits_per_pixel <= 16)
    do_vectorscope_color2<uint16_t>(
      (uint16_t *)panel, (uint16_t*)panelU, (uint16_t*)panelV,
      (uint16_t*)pY, (uint16_t*)pU, (uint16_t*)pV,
      dst_pitch, dst_pitchUV,
      src_pitch, src_pitchUV,
      src_widthUV, src_heightUV,
      swidth, sheight,
      show_bits, bits_per_pixel, d_chroma, full_range);
  else
    do_vectorscope_color2<float>(
      (float*)panel, (float*)panelU, (float*)panelV,
      (float*)pY, (float*)pU, (float*)pV,
      dst_pitch, dst_pitchUV,
      src_pitch, src_pitchUV,
      src_widthUV, src_heightUV,
      swidth, sheight,
      show_bits, bits_per_pixel, d_chroma, full_range);

  return dst;
}

template<typename pixel_t, bool chroma_danger>
static void DrawModeColor_PlotHistogram_inner(
  uint8_t* panel_y, int dstpitch,
  const int* histUV,
  int show_size, int show_bits,
  int bits_per_pixel,
  int maxval)
{
  const int limit16 = 16 << (show_bits - 8);
  const int limit240 = 240 << (show_bits - 8);
  int limit16_pixel, luma235;
  if constexpr (std::is_same_v<pixel_t, float>) {
    limit16_pixel = 16;
    luma235 = 235;
  }
  else {
    limit16_pixel = 16 << (bits_per_pixel - 8);
    luma235 = 235 << (bits_per_pixel - 8);
  }
  const int scale = std::is_same_v<pixel_t, float> ? 1 : (1 << (bits_per_pixel - 8));

  for (int y = 0; y < show_size; y++) {
    // ylimited: compile-time eliminated when chroma_danger==false
    const bool ylimited = chroma_danger && (y < limit16 || y > limit240);
    auto row = reinterpret_cast<pixel_t*>(panel_y);
    const int* histRow = histUV + y * show_size;
    for (int x = 0; x < show_size; x++) {
      // 64 bit: count * scale overflows int when one color fills a large area at high bit depth.
      // Capped at luma235, the limit of 'out' anyway
      int disp_val = (int)std::min<int64_t>((int64_t)histRow[x] * scale / maxval, luma235);
      // x danger check: compile-time eliminated when chroma_danger==false
      if constexpr (chroma_danger) {
        if (ylimited || x < limit16 || x > limit240)
          disp_val -= limit16_pixel;
      }
      int out = min(luma235, limit16_pixel + disp_val);
      if constexpr (std::is_same_v<pixel_t, float>)
        row[x] = (float)out / 255.0f;
      else
        row[x] = (pixel_t)out;
    }
    panel_y += dstpitch;
  }
}

template<typename pixel_t>
static void DrawModeColor_PlotHistogram_dispatch(
  bool show_chroma_danger_zone,
  uint8_t* panel_y, int dstpitch,
  const int* histUV,
  int show_size, int show_bits,
  int bits_per_pixel,
  int maxval)
{
  if (show_chroma_danger_zone)
    DrawModeColor_PlotHistogram_inner<pixel_t, true>(panel_y, dstpitch, histUV, show_size, show_bits, bits_per_pixel, maxval);
  else
    DrawModeColor_PlotHistogram_inner<pixel_t, false>(panel_y, dstpitch, histUV, show_size, show_bits, bits_per_pixel, maxval);
}



PVideoFrame Histogram::DrawModeColor(int n, IScriptEnvironment* env) {

  // all these will get filled by VectorscopePrelude, common for color and color2 modes
  PVideoFrame src;
  BYTE* panel, * panelU, * panelV, * panelA;
  int dst_pitch, dst_pitchUV, dst_pitchA, dst_height, dst_heightUV, dst_heightA;

  PVideoFrame dst = VectorscopePrelude(n, env,
    // outputs
    src,
    dst_pitch, dst_height,
    dst_pitchUV, dst_heightUV,
    dst_pitchA, dst_heightA,
    panel, panelU, panelV, panelA);

  // always planar

  int show_size = 1 << show_bits; // 256 for 8 bits, max 1024x1024 (10 bit resolution) found

  bits_conv_constants d_chroma;
  get_bits_conv_constants(d_chroma, /*use_chroma=*/true,
    full_range, full_range,
    bits_per_pixel, /*dstBitDepth=*/show_bits);
  // d_chroma.mul_factor and d_chroma.dst_offset are now correct for
  // limited-limited or full-full depending on _ColorRange frame property.

  // Allocate histogram array, can be huge for 12 bits (4096x4096 = 16 million ints), so use nothrow and check for failure.
  int* histUV = new(std::nothrow) int[show_size * show_size];
  if (!histUV)
    env->ThrowError("Histogram: malloc failure!");

  memset(histUV, 0, sizeof(int) * show_size * show_size);

  // Create histogram from the original src frame's U and V planes.
  // We need to take into account the bit depth and the show_bits, as well as the limited/full range
  // for scaling to show_bits correctly.
  const BYTE* pU = src->GetReadPtr(PLANAR_U);
  const BYTE* pV = src->GetReadPtr(PLANAR_V);

  int w = origwidth >> vi.GetPlaneWidthSubsampling(PLANAR_U);
  int h = src->GetHeight(PLANAR_U);
  int p = src->GetPitch(PLANAR_U) / pixelsize;

  const int max_pos = (1 << show_bits) - 1;
  const float mul_factor = d_chroma.mul_factor;
  const float dst_offset = d_chroma.dst_offset;
  const int src_offset_i = d_chroma.src_offset_i;

  if (pixelsize == 1) {
    if (show_bits == bits_per_pixel) {
      // quick case: no bit depth conversion needed
      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          int u = pU[y * p + x];
          int v = pV[y * p + x];
          histUV[(v << 8) + u]++;
        }
      }
    }
    else {
      // 8 bit data on 10 bit sized screen
      // works for both limited and full range, since the conversion is the same for 8 bit source
      // (10 bit limited case: mul_factor = 4, dst_offset = 0 for full, dst_offset = 16 for limited)
      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          const int uval = reinterpret_cast<const uint8_t*>(pU)[y * p + x];
          const int vval = reinterpret_cast<const uint8_t*>(pV)[y * p + x];
          int uval_posindex = (int)((uval - src_offset_i) * mul_factor + dst_offset + 0.5f);
          int vval_posindex = (int)((vval - src_offset_i) * mul_factor + dst_offset + 0.5f);
          uval_posindex = max(0, min(max_pos, uval_posindex));
          vval_posindex = max(0, min(max_pos, vval_posindex));
          histUV[(vval_posindex << show_bits) + uval_posindex]++;
        }
      }
    }
  }
  else if (pixelsize == 2) {
    if (show_bits == bits_per_pixel) {
      // quick case: no bit depth conversion needed
      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          int u = reinterpret_cast<const uint16_t*>(pU)[y * p + x];
          int v = reinterpret_cast<const uint16_t*>(pV)[y * p + x];
          histUV[(v << show_bits) + u]++;
        }
      }
    }
    else {
      // works for both limited and full range, since the conversion is the same for 16 bit source (mul_factor = 4, dst_offset = 0 for full, dst_offset = 16 for limited)
      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          const int uval = reinterpret_cast<const uint16_t*>(pU)[y * p + x];
          const int vval = reinterpret_cast<const uint16_t*>(pV)[y * p + x];
          int uval_posindex = (int)((uval - src_offset_i) * mul_factor + dst_offset + 0.5f);
          int vval_posindex = (int)((vval - src_offset_i) * mul_factor + dst_offset + 0.5f);
          uval_posindex = max(0, min(max_pos, uval_posindex));
          vval_posindex = max(0, min(max_pos, vval_posindex));
          histUV[(vval_posindex << show_bits) + uval_posindex]++;
        }
      }
    }
  }
  else { // float
    // 32 bit data on show_bits bit sized screen
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        const float uval_f = max(-0.5f, min(0.5f, reinterpret_cast<const float*>(pU)[y * p + x]));
        const float vval_f = max(-0.5f, min(0.5f, reinterpret_cast<const float*>(pV)[y * p + x]));
        int uval_posindex = (int)(uval_f * mul_factor + dst_offset + 0.5f);
        int vval_posindex = (int)(vval_f * mul_factor + dst_offset + 0.5f);
        uval_posindex = max(0, min(max_pos, uval_posindex));
        vval_posindex = max(0, min(max_pos, vval_posindex));
        histUV[(vval_posindex << show_bits) + uval_posindex]++;
      }
    }
  }
  // End of histogram creation.

  const bool show_chroma_danger_zone = color2_params.graticule_type == histogram_color2_params::GRATICULE_ON ||
    (color2_params.graticule_type == histogram_color2_params::GRATICULE_AUTO && !full_range);

  // Plot Histogram on Y.
  int maxval = 1;
  // Should we adjust the divisor (maxval)??
  if (pixelsize == 1)
    DrawModeColor_PlotHistogram_dispatch<uint8_t>(show_chroma_danger_zone, panel, dst_pitch, histUV, show_size, show_bits, bits_per_pixel, maxval);
  else if (pixelsize == 2)
    DrawModeColor_PlotHistogram_dispatch<uint16_t>(show_chroma_danger_zone, panel, dst_pitch, histUV, show_size, show_bits, bits_per_pixel, maxval);
  else
    DrawModeColor_PlotHistogram_dispatch<float>(show_chroma_danger_zone, panel, dst_pitch, histUV, show_size, show_bits, bits_per_pixel, maxval);

  // Draw colors - U gradient (x-driven) and V gradient (y-driven)
  const int swidth = vi.GetPlaneWidthSubsampling(PLANAR_U);
  const int sheight = vi.GetPlaneHeightSubsampling(PLANAR_U);
  const int uvWidth = show_size >> swidth;
  const int uvHeight = show_size >> sheight;
  const int shiftCount = show_bits - bits_per_pixel; // for int types
  const int rounder = shiftCount > 0 ? (1 << (shiftCount - 1)) : 0; // for int types when downshifting
  const int shiftCount_f = show_bits - 8; // ? Why

  // U: value depends only on x -> precompute one row, memcpy for all y
  // V: value depends only on y -> compute once per row, fill entire row

  if (pixelsize == 1) {
    // --- U ---
    // Precompute one row
    std::vector<uint8_t> uRow(uvWidth);
    for (int x = 0; x < uvWidth; x++)
      uRow[x] = (uint8_t)((x << swidth) >> shiftCount);
    auto p = panelU;
    for (int y = 0; y < uvHeight; y++, p += dst_pitchUV)
      memcpy(p, uRow.data(), uvWidth);
    // --- V ---
    p = panelV;
    for (int y = 0; y < uvHeight; y++, p += dst_pitchUV) {
      uint8_t vval = (uint8_t)(((y << sheight) + rounder) >> shiftCount);
      memset(p, vval, uvWidth);
    }
  }
  else if (pixelsize == 2) {
    // --- U ---
    std::vector<uint16_t> uRow(uvWidth);
    if (shiftCount >= 0) {
      for (int x = 0; x < uvWidth; x++)
        uRow[x] = (uint16_t)(((x << swidth) + rounder) >> shiftCount);
    }
    else {
      for (int x = 0; x < uvWidth; x++)
        uRow[x] = (uint16_t)((x << swidth) << -shiftCount);
    }
    auto p = panelU;
    for (int y = 0; y < uvHeight; y++, p += dst_pitchUV)
      memcpy(p, uRow.data(), uvWidth * sizeof(uint16_t));

    // --- V ---
    p = panelV;
    if (shiftCount >= 0) {
      for (int y = 0; y < uvHeight; y++, p += dst_pitchUV) {
        uint16_t vval = (uint16_t)(((y << sheight) + rounder) >> shiftCount);
        std::fill_n(reinterpret_cast<uint16_t*>(p), uvWidth, vval);
      }
    }
    else {
      for (int y = 0; y < uvHeight; y++, p += dst_pitchUV) {
        uint16_t vval = (uint16_t)((y << sheight) << -shiftCount);
        std::fill_n(reinterpret_cast<uint16_t*>(p), uvWidth, vval);
      }
    }
  }
  else { // float
    // --- U ---
    std::vector<float> uRow(uvWidth);
    for (int x = 0; x < uvWidth; x++)
      uRow[x] = uv8tof((x << swidth) >> shiftCount_f);
    auto p = panelU;
    for (int y = 0; y < uvHeight; y++, p += dst_pitchUV)
      memcpy(p, uRow.data(), uvWidth * sizeof(float));
    // --- V ---
    p = panelV;
    for (int y = 0; y < uvHeight; y++, p += dst_pitchUV) {
      float vval = uv8tof((y << sheight) >> shiftCount_f);
      std::fill_n(reinterpret_cast<float*>(p), uvWidth, vval);
    }
  }

  delete[] histUV;

  // others: RGB and IQ targets, 15 degree marks, etc
  if (bits_per_pixel == 8)
    Draw_VectorScope_circle_targets_axes<uint8_t>(bits_per_pixel,
      panel, panelU, panelV,
      dst_pitch, dst_pitchUV,
      dst_height, dst_heightUV,
      color2_innerF,
      show_bits, swidth, sheight,
      color2_params, matrix, theMatrix, d_chroma, full_range,
      deg15c, deg15s
    );
  else if (bits_per_pixel <= 16)
    Draw_VectorScope_circle_targets_axes<uint16_t>(bits_per_pixel,
      panel, panelU, panelV,
      dst_pitch, dst_pitchUV,
      dst_height, dst_heightUV,
      color2_innerF,
      show_bits, swidth, sheight,
      color2_params, matrix, theMatrix, d_chroma, full_range,
      deg15c, deg15s
    );
  else
    Draw_VectorScope_circle_targets_axes<float>(bits_per_pixel,
      panel, panelU, panelV,
      dst_pitch, dst_pitchUV,
      dst_height, dst_heightUV,
      color2_innerF,
      show_bits, swidth, sheight,
      color2_params, matrix, theMatrix, d_chroma, full_range,
      deg15c, deg15s
    );

  return dst;
}


PVideoFrame Histogram::DrawModeLevels(int n, IScriptEnvironment* env) {
  PVideoFrame src = child->GetFrame(n, env);
  PVideoFrame dst = env->NewVideoFrameP(vi, &src);
  BYTE* dstp = dst->GetWritePtr();

  // Note: drawing colors are of limited range independently from clip range
  // bool full_range is filled in ctor.

  int show_size = 1 << show_bits;

  const bool isGrey = vi.IsY();
  const int planecount = isGrey ? 1 : 3;
  const bool RGB = vi.IsRGB();
  const bool isFloat = (bits_per_pixel == 32);
  const int color_shift =  isFloat ? 0 : (bits_per_pixel - 8);
  const int max_pixel_value = bits_per_pixel <= 16 ? (1 << bits_per_pixel) - 1 : 0; // float: n/a
  const float color_shift_factor = isFloat ? 1.0f / 255.0f : max_pixel_value / 255.0f;

  int plane_default_black[3] = {
    RGB ? 0 : (16 << color_shift),
    RGB ? 0 : (128 << color_shift),
    RGB ? 0 : (128 << color_shift)
  };
  float plane_default_black_f[3] = {
    RGB ? 0 : 16 / 255.0f,
    RGB ? 0 : 0.0f,
    RGB ? 0 : 0.0f
  };

  const int planesYUV[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A};
  const int planesRGB[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A};
  const int *planes = vi.IsYUV() || vi.IsYUVA() ? planesYUV : planesRGB;

  if (keepsource) {
    if (src->GetHeight() < dst->GetHeight()) {
      // fill empty area in the right bottom part
      const int fillSize = (dst->GetHeight() - src->GetHeight()) * dst->GetPitch();
      const int fillStart = src->GetHeight() * dst->GetPitch();

      switch (pixelsize) {
      case 1: memset(dstp + fillStart, plane_default_black[0], fillSize); break;
      case 2: std::fill_n((uint16_t *)(dstp + fillStart), fillSize / sizeof(uint16_t), plane_default_black[0]); break;
      case 4: std::fill_n((float *)(dstp + fillStart), fillSize / sizeof(float), plane_default_black_f[0]); break;
      }

      // first plane is already processed
      // dont't touch Alpha
      for (int p = 1; p < planecount; p++) {
        const int plane = planes[p];
        BYTE* ptr = dst->GetWritePtr(plane);

        const int fillSize = (dst->GetHeight(plane) - src->GetHeight(plane)) * dst->GetPitch(plane);
        const int fillStart = src->GetHeight(plane) * dst->GetPitch(plane);
        switch (pixelsize) {
        case 1: memset(ptr + fillStart, RGB ? 0 : plane_default_black[p], fillSize); break;
        case 2: std::fill_n((uint16_t*)(ptr + fillStart), fillSize / sizeof(uint16_t), plane_default_black[p]); break;
        case 4: std::fill_n((float*)(ptr + fillStart), fillSize / sizeof(float), plane_default_black_f[p]); break;
        }
      }
    }
  }

  // counters
  int bufsize = sizeof(uint32_t)*show_size;
  uint32_t *histPlane1 = static_cast<uint32_t*>(env->Allocate(bufsize * planecount, 16, AVS_NORMAL_ALLOC));
  if (!histPlane1)
    env->ThrowError("Histogram: Could not reserve memory.");
  uint32_t *histPlanes[3] = {
    histPlane1, 
    planecount == 1 ? nullptr : histPlane1 + show_size, 
    planecount == 1 ? nullptr : histPlane1 + 2 * show_size
  };
  std::fill_n(histPlane1, show_size * planecount, 0);

  const int xstart = keepsource ? origwidth : 0; // drawing starts at this column

  // copy planes
  // luma or G
  if (keepsource) {
    env->BitBlt(dstp, dst->GetPitch(), src->GetReadPtr(), src->GetPitch(), src->GetRowSize(), src->GetHeight());
  }
  if (vi.IsPlanar()) {
    // copy rest planes
    if (keepsource) {
      for (int p = 1; p < vi.NumComponents(); p++) {
        const int plane = planes[p];
        env->BitBlt(dst->GetWritePtr(plane), dst->GetPitch(plane), src->GetReadPtr(plane), src->GetPitch(plane), src->GetRowSize(plane), src->GetHeight(plane));
      }
    }

    const int max_show_pixel_value = show_size - 1;

    // ---------------------------------------------------
    // accumulate population: fill histogram array from pixels
    for (int p = 0; p < planecount; p++) {
      const int plane = planes[p];
      const bool chroma = (plane == PLANAR_U || plane == PLANAR_V);
      const BYTE* srcp = src->GetReadPtr(plane);

      const int w = src->GetRowSize(plane) / pixelsize;
      const int h = src->GetHeight(plane);
      const int pitch = src->GetPitch(plane) / pixelsize;

      // accumulator of current plane
      // size: show_size (256 or 1024)
      uint32_t *hist = histPlanes[p];

      bits_conv_constants d;
      get_bits_conv_constants(d, chroma, full_range, full_range, bits_per_pixel/*source_bitdepth*/, show_bits /*target_bitdepth*/);

      if(pixelsize==1) {
        const uint8_t *srcp8 = reinterpret_cast<const uint8_t *>(srcp);
        if (show_bits == bits_per_pixel) {
          // Exact. 8 bit clip into 8 bit histogram
          for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
              hist[srcp8[x]]++;
            }
            srcp8 += pitch;
          }
        }
        else {
          // 8 bit clip into 8,9,... bit histogram
          if (full_range) {
            // full range: stretch
            if (chroma) {
              const float dst_offset_plus_round = d.dst_offset + 0.5f;
              for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                  hist[clamp((int)((srcp8[x] - d.src_offset_i) * d.mul_factor + dst_offset_plus_round), 0, max_show_pixel_value)]++;
                }
                srcp8 += pitch;
              }
            }
            else {
              // src_offset and dst_offset is 0
              for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                  hist[min((int)(srcp8[x] * d.mul_factor + 0.5f), max_show_pixel_value)]++;
                }
                srcp8 += pitch;
              }
            }
          }
          else {
            const int invshift = show_bits - bits_per_pixel;
            // limited range: shift
            for (int y = 0; y < h; y++) {
              for (int x = 0; x < w; x++) {
                hist[(int)srcp8[x] << invshift]++;
              }
              srcp8 += pitch;
            }
          }
        }
      }
      else if (pixelsize == 2) {
        const uint16_t* srcp16 = reinterpret_cast<const uint16_t*>(srcp);
        if (show_bits == bits_per_pixel) {
          // Exact. e.g. 10 bit clip into 10 bit histogram
          for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
              hist[min((int)srcp16[x], max_show_pixel_value)]++;
            }
            srcp16 += pitch;
          }
        }
        else {
          // 10 bit clip into 8,9,10,11,12,... bit histogram
          if (full_range) {
            // full range: stretch
            if (chroma) {
              const float dst_offset_plus_round = d.dst_offset + 0.5f;
              for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                  hist[clamp((int)((srcp16[x] - d.src_offset_i) * d.mul_factor + dst_offset_plus_round), 0, max_show_pixel_value)]++;
                }
                srcp16 += pitch;
              }
            }
            else {
              // src_offset and dst_offset is 0
              for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                  hist[min((int)(srcp16[x] * d.mul_factor + 0.5f), max_show_pixel_value)]++;
                }
                srcp16 += pitch;
              }
            }
          }
          else {
            // limited range: shift. Up or down
            const int shift = bits_per_pixel - show_bits;
            if (shift < 0) {
              // 10 bit clip into 11 bit histogram
              int invshift = -shift;
              for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                  hist[min(srcp16[x] << invshift, max_show_pixel_value)]++;
                }
                srcp16 += pitch;
              }
            }
            else {
              // e.g.10 bit clip into 8-9 bit histogram
              const int round = 1 << (shift - 1);
              for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                  hist[min((srcp16[x] + round) >> shift, max_show_pixel_value)]++;
                }
                srcp16 += pitch;
              }
            }
          }
        }
      }
      else {
        // create fake Histogram 32 bit float
        const float *srcp32 = reinterpret_cast<const float *>(srcp);
        const float dst_offset_plus_round = d.dst_offset + 0.5f;
        for (int y = 0; y < h; y++) {
          for (int x = 0; x < w; x++) {
            hist[clamp((int)(srcp32[x] * d.mul_factor + dst_offset_plus_round), 0, max_show_pixel_value)]++;
          }
          srcp32 += pitch;
        }
      }
    }
    // accumulate end
    // ---------------------------------------------------

    int pos_shift = (show_bits - 8);
    int show_middle_pos = (128 << pos_shift);
    // draw planes
    for (int p = 0; p < planecount; p++) {
      const int plane = planes[p];

      int swidth = vi.GetPlaneWidthSubsampling(plane);
      int sheight = vi.GetPlaneHeightSubsampling(plane);

      // Draw Unsafe zone (UV-graph)

      unsigned char* pdstb = dst->GetWritePtr(plane);
      pdstb += (xstart*pixelsize) >> swidth; // next to the source image if kept

      const int dstPitch = dst->GetPitch(plane);

      // Clear Y/U/V or B, R G
      BYTE *ptr = pdstb;
      for (int y = 0; y < dst->GetHeight() >> sheight; y++) {
        switch (pixelsize) {
        case 1: memset(ptr, plane_default_black[p], show_size >> swidth); break;
        case 2: std::fill_n((uint16_t *)(ptr), show_size >> swidth, plane_default_black[p]); break;
        case 4: std::fill_n((float *)(ptr), show_size >> swidth, plane_default_black_f[p]); break;
        }
        ptr += dstPitch;
      }

      if (!RGB && markers) {
        // Draw Unsafe zone (Y-graph)
        const int color_unsafeZones[3] = { 32, 16, 160 };
        const float color_unsafeZones_f[3] = { c8tof(32), uv8tof(16), uv8tof(160) };
        int color_usz = color_unsafeZones[p];
        int color_i = color_usz << color_shift;
        float color_f = color_unsafeZones_f[p];
        ptr = pdstb + 0 * dstPitch;;

        for (int y = 0; y <= 64 >> sheight; y++) {
          int x = 0;
          for (; x < (16 << pos_shift) >> swidth; x++) {
            if (pixelsize == 1)
              ptr[x] = color_i;
            else if (pixelsize == 2)
              reinterpret_cast<uint16_t *>(ptr)[x] = color_i;
            else
              reinterpret_cast<float *>(ptr)[x] = color_f;
          }
          for (x = (236 << pos_shift) >> swidth; x < (show_size >> swidth); x++) { // or (235 << pos_shift) + 1?
            if (pixelsize == 1)
              ptr[x] = color_i;
            else if (pixelsize == 2)
              reinterpret_cast<uint16_t *>(ptr)[x] = color_i;
            else
              reinterpret_cast<float *>(ptr)[x] = color_f;
          }
          ptr += dstPitch;
        }
      }

      if (markers) {
        if (RGB) {
          // nice gradients
          int StartY;
          switch (plane) {
          case PLANAR_R: StartY = 0 + 0; break;
          case PLANAR_G: StartY = 64 + 16; break;
          case PLANAR_B: StartY = 64 + 16 + 64 + 16; break;
          }
          ptr = pdstb + ((StartY) >> sheight) * dstPitch;
          for (int y = (StartY) >> sheight; y <= (StartY + 64) >> sheight; y++) {
            if (pixelsize == 1) {
              for (int x = 0; x < (show_size >> swidth); x++) {
                int color = x >> pos_shift;
                int color_i = color << color_shift;
                ptr[x] = color_i;
              }
            }
            else if (pixelsize == 2) {
              for (int x = 0; x < (show_size >> swidth); x++) {
                int color = x >> pos_shift;
                int color_i = color << color_shift;
                reinterpret_cast<uint16_t *>(ptr)[x] = color_i;
              }
            }
            else { // pixelsize == 4 float
              for (int x = 0; x < (show_size >> swidth); x++) {
                int color = x >> pos_shift;
                float color_f = color / 255.0f;
                reinterpret_cast<float *>(ptr)[x] = color_f;
              }
            }
            ptr += dstPitch;
          }
        }
        else {
          if (!isGrey) {
            // UV gradients plus danger zones
            for (int gradient_upper_lower = 0; gradient_upper_lower < 2; gradient_upper_lower++)
            {
              // Draw upper and lower gradient
            // upper: x=0-16, R=G=255, B=0; x=128, R=G=B=0; x=240-255, R=G=0, B=255
            // lower: x=0-16, R=0, G=B=255; x=128, R=G=B=0; x=240-255, R=255, G=B=0
              const int color1_upper_lower_gradient[2][3] = { { 210 / 2, 16 + 112 / 2, 128 },{ 170 / 2, 128, 16 + 112 / 2 } };
              const int color2_upper_lower_gradient[2][3] = { { 41 / 2, 240 - 112 / 2, 128 },{ 81 / 2, 128, 240 - 112 / 2 } };
              float color1_upper_lower_gradient_f[2][3] = { { c8tof(210 / 2), uv8tof(16 + 112 / 2), uv8tof(128) },{ c8tof(170 / 2), uv8tof(128), uv8tof(16 + 112 / 2) } };
              float color2_upper_lower_gradient_f[2][3] = { { c8tof(41 / 2), uv8tof(240 - 112 / 2), uv8tof(128) },{ c8tof(81 / 2), uv8tof(128), uv8tof(240 - 112 / 2) } };
              int color = color1_upper_lower_gradient[gradient_upper_lower][p];
              int color_i = color << color_shift;
              float color_f = color1_upper_lower_gradient_f[gradient_upper_lower][p];

              int color2 = color2_upper_lower_gradient[gradient_upper_lower][p];
              int color2_i = color2 << color_shift;
              float color2_f = color2_upper_lower_gradient_f[gradient_upper_lower][p];;

              // upper only for planar U and Y
              if (plane == PLANAR_V && gradient_upper_lower == 0)
                continue;
              // lower only for planar V and Y
              if (plane == PLANAR_U && gradient_upper_lower == 1)
                continue;
              int StartY = gradient_upper_lower == 0 ? 64 + 16 : 128 + 32;
              ptr = pdstb + ((StartY) >> sheight) * dstPitch;
              // we are drawing at the 64th relative line as well.
              for (int y = (StartY) >> sheight; y <= (StartY + 64) >> sheight; y++) {
                int x = 0;
                // 0..15, (scaled) left danger area
                const int left_limit = ((16 << pos_shift) >> swidth) - 1;
                if (pixelsize == 1) {
                  for (; x < left_limit; x++)
                    ptr[x] = color_i;
                }
                else if (pixelsize == 2) {
                  for (; x < left_limit; x++)
                    reinterpret_cast<uint16_t*>(ptr)[x] = color_i;
                }
                else { // float
                  for (; x < left_limit; x++)
                    reinterpret_cast<float*>(ptr)[x] = color_f;
                }

                if (plane == PLANAR_Y) {
                  // from 16 to middle point
                  if (pixelsize == 1) {
                    for (; x <= show_middle_pos; x++) {
                      int color3 =
                        (gradient_upper_lower == 0) ?
                        (((show_middle_pos - x) * 15) >> 3) >> pos_shift : // *1.875
                        ((show_middle_pos - x) * 99515) >> 16 >> pos_shift; // *1.518
                      int color3_i = color3 << color_shift;
                      ptr[x] = color3_i;
                    }
                  }
                  else if (pixelsize == 2) {
                    for (; x <= show_middle_pos; x++) {
                      int color3 =
                        (gradient_upper_lower == 0) ?
                        (((show_middle_pos - x) * 15) >> 3) >> pos_shift : // *1.875
                        ((show_middle_pos - x) * 99515) >> 16 >> pos_shift; // *1.518
                      int color3_i = color3 << color_shift;
                      reinterpret_cast<uint16_t*>(ptr)[x] = color3_i;
                    }
                  }
                  else { // float
                    for (; x <= show_middle_pos; x++) {
                      int color3 =
                        (gradient_upper_lower == 0) ?
                        (((show_middle_pos - x) * 15) >> 3) >> pos_shift : // *1.875
                        ((show_middle_pos - x) * 99515) >> 16 >> pos_shift; // *1.518
                      float color3_f = color3 / 255.0f; // no shift this is Y
                      reinterpret_cast<float*>(ptr)[x] = color3_f;
                    }
                  }
                }

                // Y: from middle point to white point
                // other plane: gradient
                if (plane == PLANAR_Y) {
                  if (pixelsize == 1) {
                    for (; x <= (240 << pos_shift) >> swidth; x++) {
                      int color4 =
                        (gradient_upper_lower == 0) ?
                        ((x - show_middle_pos) * 24001) >> 16 >> pos_shift :  // *0.366
                        ((x - show_middle_pos) * 47397) >> 16 >> pos_shift; // *0.723
                      int color4_i = color4 << color_shift;
                      ptr[x] = color4_i;
                    }
                  }
                  else if (pixelsize == 2) {
                    for (; x <= (240 << pos_shift) >> swidth; x++) {
                      int color4 =
                        (gradient_upper_lower == 0) ?
                        ((x - show_middle_pos) * 24001) >> 16 >> pos_shift :  // *0.366
                        ((x - show_middle_pos) * 47397) >> 16 >> pos_shift; // *0.723
                      int color4_i = color4 << color_shift;
                      reinterpret_cast<uint16_t*>(ptr)[x] = color4_i;
                    }
                  }
                  else { // float
                    for (; x <= (240 << pos_shift) >> swidth; x++) {
                      int color4 =
                        (gradient_upper_lower == 0) ?
                        ((x - show_middle_pos) * 24001) >> 16 >> pos_shift :  // *0.366
                        ((x - show_middle_pos) * 47397) >> 16 >> pos_shift; // *0.723
                      float color4_f = color4 / 255.0f; // no shift, this is Y
                      reinterpret_cast<float*>(ptr)[x] = color4_f;
                    }
                  }
                }
                else {
                  // plane == U or V
                  if (pixelsize == 1) {
                    for (; x <= (240 << pos_shift) >> swidth; x++) {
                      int color4 = (x << swidth) >> pos_shift;
                      int color4_i = color4 << color_shift;
                      ptr[x] = color4_i;
                    }
                  }
                  else if (pixelsize == 2) {
                    for (; x <= (240 << pos_shift) >> swidth; x++) {
                      int color4 = (x << swidth) >> pos_shift;
                      int color4_i = color4 << color_shift;
                      reinterpret_cast<uint16_t*>(ptr)[x] = color4_i;
                    }
                  }
                  else { // float
                    for (; x <= (240 << pos_shift) >> swidth; x++) {
                      int color4 = (x << swidth) >> pos_shift;
                      reinterpret_cast<float*>(ptr)[x] = uv8tof(color4);
                    }
                  }
                }

                if (pixelsize == 1) {
                  for (; x < (show_size >> swidth); x++)
                    ptr[x] = color2_i;
                }
                else if (pixelsize == 2) {
                  for (; x < (show_size >> swidth); x++)
                    reinterpret_cast<uint16_t*>(ptr)[x] = color2_i;
                }
                else { // float
                  for (; x < (show_size >> swidth); x++)
                    reinterpret_cast<float*>(ptr)[x] = color2_f;
                }

                ptr += dstPitch;
              } // for y gradient draw
            } // gradient for upper lower
          }
        } // gradients for RGB/UV
      } // if markers
    } // planes for

    // Draw dotted centerline
    // YUV: only 1 plane (PLANAR_Y)
    for (int p = 0; p < (RGB ? 3 : 1); p++) {
      const int plane = planes[p];

      int color = 128; // also good for RGB
      int color_i = color << color_shift;
      float color_f = 0.5f;

      const int dstPitch = dst->GetPitch(plane);

      unsigned char* pdstb = dst->GetWritePtr(plane);
      pdstb += (xstart * pixelsize); // next to the original clip (if kept), working only on Y plane: no ">> swidth" needed
      BYTE* ptr = pdstb;

      if (markers) {
        // omit centerline if markers == false
        // height of 64, 16 pixels between
        // we are drawing at the 64th/224th relative line as well.
        for (int y = 0; y <= 64 + (planecount - 1) * (16 + 64); y++) {
          if ((y & 3) > 1) {
            if (pixelsize == 1)       ptr[show_middle_pos] = color_i;
            else if (pixelsize == 2)  reinterpret_cast<uint16_t*>(ptr)[show_middle_pos] = color_i;
            else                   reinterpret_cast<float*>(ptr)[show_middle_pos] = color_f;

          }
          ptr += dstPitch;
        }
      }
    }

    for (int p = 0; p < (RGB ? 3 : 1); p++) {
      const int plane = planes[p];

      const int dstPitch = dst->GetPitch(plane);

      unsigned char* pdstb = dst->GetWritePtr(plane);
      pdstb += (xstart * pixelsize); // next to the original clip (if kept), working only on Y plane: no ">> swidth" needed

      // have to draw up to three parts: Y or Y,U,V or R,G,B
      for (int n = 0; n < planecount; n++) {

        // Draw histograms

        // total pixel count must be divided by subsampling factors for U and V
        int src_width = src->GetRowSize(planes[n]) / pixelsize;
        int src_height = src->GetHeight(planes[n]);

        const uint32_t clampval = (int)((src_width*src_height)*option.AsDblDef(100.0) / 100.0); // Population limit % factor
        uint32_t maxval = 0;
        uint32_t *hist;

        hist = histPlanes[n];
        for (int i = 0; i < show_size; i++) {
          if (hist[i] > clampval) hist[i] = clampval;
          maxval = max(hist[i], maxval);
        }

        // factor 0%: maxval may stay at 0.
        float scale = maxval == 0 ? 64.0f : float(64.0 / maxval);

        int color = RGB ? 255 : 235; // also good for RGB
        int color_i = RGB ? (int)(color * color_shift_factor + 0.5f) :  color << color_shift;
        float color_f = color / 255.0f;

        int Y_pos;
        switch (n) { // n: YUV 012, GBR 012
        case 0: Y_pos = RGB ? 128 + 16 :  64 + 0; break;  // Y or G
        case 1: Y_pos = RGB ? 192 + 32 : 128 + 16; break; // U or B
        case 2: Y_pos = RGB ?  64 +  0 : 192 + 32; break; // V or R
        }

        for (int x = 0; x < show_size; x++) {
          float scaled_h = (float)hist[x] * scale;
          int h = Y_pos - min((int)scaled_h, 64) + 1;

          uint8_t *ptr = pdstb + (Y_pos + 1) * dstPitch;

          // Start from below the baseline
          for (int y = Y_pos + 1; y > h; y--) {
            if (pixelsize == 1)       ptr[x] = color_i;
            else if (pixelsize == 2)  reinterpret_cast<uint16_t *>(ptr)[x] = color_i;
            else                   reinterpret_cast<float *>(ptr)[x] = color_f;
            ptr -= dstPitch;
          }

        }
      }
    }
  }

  env->Free(histPlane1);

  return dst;
}


void Histogram::ClassicLUTInit()
{
  int internal_bits_per_pixel = (pixelsize == 4) ? 16 : bits_per_pixel; // 16bit histogram simulation for float

  int tv_range_low = 16 << (internal_bits_per_pixel - 8); // 16
  int tv_range_hi_luma = 235 << (internal_bits_per_pixel - 8); // 16-235
  int range_luma = tv_range_hi_luma - tv_range_low; // 219
  // exptab: population count within a line -> brigtness mapping
  // exptab index (population) is maximized at 255 during the actual drawing
  exptab.resize(256);
  const double K = log(0.5 / 219) / 255.0; // approx -1/42
  const int limit68 = 68 << (internal_bits_per_pixel - 8);
  // exptab: pixel values for final drawing
  exptab[0] = tv_range_low;
  for (int i = 1; i < 255; i++) {
    exptab[i] = uint16_t(tv_range_low + 0.5 + range_luma * (1 - exp(i*K))); // 16.5 + 219*
    if (exptab[i] <= tv_range_hi_luma - limit68)
      E167 = i; // index of last value less than...  for drawing lower extremes
  }
  exptab[255] = tv_range_hi_luma;
}

PVideoFrame Histogram::DrawModeClassic(int n, IScriptEnvironment* env)
{

  int show_size = 1 << show_bits;

  int hist_tv_range_low = 16 << (show_bits - 8); // 16
  int hist_tv_range_hi_luma = 235 << (show_bits - 8); // 16-235
  int hist_range_luma = hist_tv_range_hi_luma - hist_tv_range_low; // 219
  int hist_mid_range_luma = (hist_range_luma + 1) / 2 + hist_tv_range_low - 1; // in Classic Avisynth somehow 124 was fixed for this
  // 235-16 = 219 / 2 => 110; 110 + 16 - 1 = 125.0

  int internal_bits_per_pixel = (pixelsize == 4) ? 16 : bits_per_pixel; // 16bit histogram simulation for float

  int middle_chroma = 1 << (internal_bits_per_pixel - 1); // 128

  const int source_width = origwidth;
  const int xstart = keepsource ? origwidth : 0; // drawing starts at this column

  PVideoFrame src = child->GetFrame(n, env);
  const BYTE* srcp = src->GetReadPtr();
  const int srcpitch = src->GetPitch();

  PVideoFrame dst = env->NewVideoFrameP(vi, &src);
  BYTE* pdst = dst->GetWritePtr();
  const int dstpitch = dst->GetPitch();

  if (keepsource) {
    env->BitBlt(pdst, dst->GetPitch(), src->GetReadPtr(), src->GetPitch(), src->GetRowSize(), src->GetHeight());
  }
  if (vi.IsPlanar()) {
    if (keepsource) {
      env->BitBlt(dst->GetWritePtr(PLANAR_U), dst->GetPitch(PLANAR_U), src->GetReadPtr(PLANAR_U), src->GetPitch(PLANAR_U), src->GetRowSize(PLANAR_U), src->GetHeight(PLANAR_U));
      env->BitBlt(dst->GetWritePtr(PLANAR_V), dst->GetPitch(PLANAR_V), src->GetReadPtr(PLANAR_V), src->GetPitch(PLANAR_V), src->GetRowSize(PLANAR_V), src->GetHeight(PLANAR_V));
      // preserve alpha. no-op if no alpha (0 pitch/rowsize/height)
      env->BitBlt(dst->GetWritePtr(PLANAR_A), dst->GetPitch(PLANAR_A), src->GetReadPtr(PLANAR_A), src->GetPitch(PLANAR_A), src->GetRowSize(PLANAR_A), src->GetHeight(PLANAR_A));
    }

    // luma
    const int height = src->GetHeight(PLANAR_Y);

    std::vector<int> hist;
    hist.resize(1ULL << show_bits);

    for (int y = 0; y<height; ++y) {
      std::fill(hist.begin(), hist.end(), 0);
      // accumulate line population
      if(pixelsize==1) {
        // 8 bit clip into 8,9,... bit histogram
        int invshift = show_bits - 8;
        for (int x = 0; x<source_width; ++x) {
          hist[srcp[x] << invshift]++;
        }
      }
      else if (pixelsize == 2) {
        const uint16_t *srcp16 = reinterpret_cast<const uint16_t *>(srcp);
        int shift = bits_per_pixel - show_bits;
        int max_show_pixel_value = show_size - 1;
        if (shift < 0) {
          // 10 bit clip into 11 bit histogram
          int invshift = -shift;
          for (int x = 0; x < source_width; x++) {
            hist[min(srcp16[x] << invshift, max_show_pixel_value)]++;
          }
        } else {
          // e.g.10 bit clip into 8-9-10 bit histogram
          for (int x = 0; x < source_width; x++) {
            hist[min(srcp16[x] >> shift, max_show_pixel_value)]++;
          }
        }
      }
      else // pixelsize == 4
      {
        // float
        const float *srcp32 = reinterpret_cast<const float *>(srcp);
        const float multiplier = (float)(show_size - 1);
        for (int x = 0; x < source_width; x++) {
          hist[(intptr_t)(clamp(srcp32[x], 0.0f, 1.0f)*multiplier + 0.5f)]++; // luma, no shift needed
        }
      }
      // accumulate end
      BYTE* const q = pdst + xstart * pixelsize; // write to frame
      if (markers) {
        if (pixelsize == 1) {
          for (int x = 0; x < show_size; ++x) {
            if (x<hist_tv_range_low || x == hist_mid_range_luma || x>hist_tv_range_hi_luma) {
              q[x] = (BYTE)exptab[min(E167, hist[x])] + 68; // brighter danger zone
            }
            else {
              q[x] = (BYTE)exptab[min(255, hist[x])];
            }
          }
        }
        else if (pixelsize == 2) {
          uint16_t *dstp16 = reinterpret_cast<uint16_t *>(q);
          for (int x = 0; x < show_size; ++x) {
            int h = hist[x];
            if (x<hist_tv_range_low || x == hist_mid_range_luma || x>hist_tv_range_hi_luma) {
              dstp16[x] = exptab[min(E167, h)] + (68 << (bits_per_pixel - 8));
            }
            else {
              dstp16[x] = exptab[min(255, h)];
            }
          }
        }
        else { // pixelsize == 4
          float *dstp32 = reinterpret_cast<float *>(q);
          for (int x = 0; x < show_size; ++x) {
            int h = hist[x];
            if (x<hist_tv_range_low || x == hist_mid_range_luma || x>hist_tv_range_hi_luma) {
              dstp32[x] = (exptab[min(E167, h)] + (68 << (internal_bits_per_pixel - 8))) / 65535.0f; // fake 0..65535 to 0..1.0
            }
            else {
              dstp32[x] = exptab[min(255, h)] / 65535.0f;
            }
          }
        }
      }
      else {
        if (pixelsize == 1) {
          for (int x = 0; x < show_size; ++x)
            q[x] = (BYTE)exptab[min(255, hist[x])];
        }
        else if (pixelsize == 2) {
          uint16_t *dstp16 = reinterpret_cast<uint16_t *>(q);
          for (int x = 0; x < show_size; ++x) {
            int h = hist[x];
            dstp16[x] = exptab[min(255, h)];
          }
        }
        else { // pixelsize == 4
          float *dstp32 = reinterpret_cast<float *>(q);
          for (int x = 0; x < show_size; ++x) {
            int h = hist[x];
            dstp32[x] = exptab[min(255, h)] / 65535.0f; // fake 0..65535 to 0..1.0
          }
        }
      }
      srcp += srcpitch;
      pdst += dstpitch;
    } // end of pixel accumulation + luma

    // chroma
    const int pitchUV = dst->GetPitch(PLANAR_U);

    if (pitchUV != 0) {
      const int subs = vi.GetPlaneWidthSubsampling(PLANAR_U);
      const int fact = 1<<subs;

      BYTE* p2 = dst->GetWritePtr(PLANAR_U) + ((xstart*pixelsize) >> subs); // put it on the right
      BYTE* p3 = dst->GetWritePtr(PLANAR_V) + ((xstart*pixelsize) >> subs); // put it on the right

      // if markers==false parameter, keep neutral coloring
      const uint16_t color_u_offlimit8 = markers ? 16 : 128;
      const uint16_t color_v_offlimit8 = markers ? 160 : 128;
      const uint16_t color_u_centermark8 = markers ? 160 : 128;
      const uint16_t color_v_centermark8 = markers ? 16 : 128;

      const uint16_t color_u_offlimit = color_u_offlimit8 << (internal_bits_per_pixel - 8);
      const uint16_t color_v_offlimit = color_v_offlimit8 << (internal_bits_per_pixel - 8);
      const uint16_t color_u_centermark = color_u_centermark8 << (internal_bits_per_pixel - 8);
      const uint16_t color_v_centermark = color_v_centermark8 << (internal_bits_per_pixel - 8);

      const float color_u_offlimit_f = uv8tof(color_u_offlimit8);
      const float color_v_offlimit_f = uv8tof(color_v_offlimit8);
      const float color_u_centermark_f = uv8tof(color_u_centermark8);
      const float color_v_centermark_f = uv8tof(color_v_centermark8);
      const float middle_chroma_f = uv8tof(128);

      const int height = src->GetHeight(PLANAR_U);
      for (int y2 = 0; y2<height; ++y2) {
        if(pixelsize==1) {
          for (int x = 0; x<show_size; x += fact) {
            if (x<hist_tv_range_low || x>hist_tv_range_hi_luma) {
              p2[x >> subs] = (BYTE)color_u_offlimit8;
              p3[x >> subs] = (BYTE)color_v_offlimit8;
            } else if (x==hist_mid_range_luma) {
              p2[x >> subs] = (BYTE)color_u_centermark8;
              p3[x >> subs] = (BYTE)color_v_centermark8;
            } else {
              p2[x >> subs] = 128;
              p3[x >> subs] = 128;
            }
          }
        }
        else if (pixelsize == 2) {
          for (int x = 0; x<show_size; x += fact) {
            if (x<hist_tv_range_low || x>hist_tv_range_hi_luma) {
              reinterpret_cast<uint16_t *>(p2)[x >> subs] = color_u_offlimit;
              reinterpret_cast<uint16_t *>(p3)[x >> subs] = color_v_offlimit;
            } else if (x==hist_mid_range_luma) {
              reinterpret_cast<uint16_t *>(p2)[x >> subs] = color_u_centermark;
              reinterpret_cast<uint16_t *>(p3)[x >> subs] = color_v_centermark;
            } else {
              reinterpret_cast<uint16_t *>(p2)[x >> subs] = middle_chroma;
              reinterpret_cast<uint16_t *>(p3)[x >> subs] = middle_chroma;
            }
          }
        } else { // pixelsize==4
          for (int x = 0; x<show_size; x += fact) {
            if (x<hist_tv_range_low || x>hist_tv_range_hi_luma) {
              reinterpret_cast<float *>(p2)[x >> subs] = color_u_offlimit_f;
              reinterpret_cast<float *>(p3)[x >> subs] = color_v_offlimit_f;
            } else if (x==hist_mid_range_luma) {
              reinterpret_cast<float *>(p2)[x >> subs] = color_u_centermark_f;
              reinterpret_cast<float *>(p3)[x >> subs] = color_v_centermark_f;
            } else {
              reinterpret_cast<float *>(p2)[x >> subs] = middle_chroma_f;
              reinterpret_cast<float *>(p3)[x >> subs] = middle_chroma_f;
            }
          }

        }
        p2 += pitchUV;
        p3 += pitchUV;
      }
    }
  } else {
    const int pitch = dst->GetPitch();
    for (int y = 0; y<src->GetHeight(); ++y) { // YUY2
      int hist[256] = { 0 };
      for (int x = 0; x<source_width; ++x) {
        hist[srcp[x*2]]++;
      }
      BYTE* const q = pdst + xstart*2;
      if (markers) {
        for (int x = 0; x < 256; x += 2) {
          if (x < 16 || x>235) {
            q[x * 2 + 0] = (BYTE)exptab[min(E167, hist[x])] + 68;
            q[x * 2 + 1] = 16;
            q[x * 2 + 2] = (BYTE)exptab[min(E167, hist[x + 1])] + 68;
            q[x * 2 + 3] = 160;
          }
          else if (x == 124) {
            q[x * 2 + 0] = (BYTE)exptab[min(E167, hist[x])] + 68;
            q[x * 2 + 1] = 160;
            q[x * 2 + 2] = (BYTE)exptab[min(255, hist[x + 1])];
            q[x * 2 + 3] = 16;
          }
          else {
            q[x * 2 + 0] = (BYTE)exptab[min(255, hist[x])];
            q[x * 2 + 1] = 128;
            q[x * 2 + 2] = (BYTE)exptab[min(255, hist[x + 1])];
            q[x * 2 + 3] = 128;
          }
        }
      }
      else {
        for (int x = 0; x < 256; x += 2) {
          q[x * 2 + 0] = (BYTE)exptab[min(255, hist[x])];
          q[x * 2 + 1] = 128;
          q[x * 2 + 2] = (BYTE)exptab[min(255, hist[x + 1])];
          q[x * 2 + 3] = 128;
        }
      }
      pdst += pitch;
      srcp += srcpitch;
    }
  }
  return dst;
}


AVSValue __cdecl Histogram::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  const char* st_m = args[1].AsString("classic");

  Mode mode = ModeClassic;

  if (!lstrcmpi(st_m, "classic"))
    mode = ModeClassic;

  if (!lstrcmpi(st_m, "levels"))
    mode = ModeLevels;

  if (!lstrcmpi(st_m, "color"))
    mode = ModeColor;

  if (!lstrcmpi(st_m, "color2"))
    mode = ModeColor2;

  if (!lstrcmpi(st_m, "luma"))
    mode = ModeLuma;

  if (!lstrcmpi(st_m, "stereoY8"))
    mode = ModeStereoY8;

  if (!lstrcmpi(st_m, "stereo"))
    mode = ModeStereo;

  if (!lstrcmpi(st_m, "stereooverlay"))
    mode = ModeOverlay;

  if (!lstrcmpi(st_m, "audiolevels"))
    mode = ModeAudioLevels;

  const VideoInfo& vi_orig = args[0].AsClip()->GetVideoInfo();

  histogram_color2_params color2_params;
  const char *graticule_str = args[7].AsString("on"); // ON: old vectorscopes always drew limits/danger zones

  // Use the lstrcmpi macro from your compatibility header
  if (lstrcmpi(graticule_str, "auto") == 0) {
    color2_params.graticule_type = histogram_color2_params::GRATICULE_AUTO;
  }
  else if (lstrcmpi(graticule_str, "on") == 0) {
    color2_params.graticule_type = histogram_color2_params::GRATICULE_ON;
  }
  else if (lstrcmpi(graticule_str, "off") == 0) {
    color2_params.graticule_type = histogram_color2_params::GRATICULE_OFF;
  }
  else {
    env->ThrowError("Histogram: 'graticule' must be \"on\", \"off\", or \"auto\".");
  }

  color2_params.targets = args[8].AsBool(false); // The 6 75% RGB squares
  color2_params.axes = args[9].AsBool(false); // horizontal and vertical axes
  color2_params.iq = args[10].AsBool(false); // +/-I and +Q targets
  color2_params.iq_lines = args[11].AsBool(false); // 33 and 123 degree lines in the color2 mode
  color2_params.circle = args[12].AsBool(mode == ModeColor2 ? true : false); // circle in the color2 mode default
  color2_params.targets100 = args[13].AsBool(false); // The 6 100% RGB squares

  if (mode == ModeLevels && vi_orig.IsRGB() && !vi_orig.IsPlanar()) {
    // as Levels can work for PlanarRGB, convert packed RGB to planar, then back
    // better that nothing
    AVSValue new_args[1] = { args[0].AsClip() };
    PClip clip;
    if (vi_orig.IsRGB24() || vi_orig.IsRGB48()) {
      clip = env->Invoke("ConvertToPlanarRGB", AVSValue(new_args, 1)).AsClip();
    }
    else if (vi_orig.IsRGB32() || vi_orig.IsRGB64()) {
      clip = env->Invoke("ConvertToPlanarRGBA", AVSValue(new_args, 1)).AsClip();
    }

    Histogram* Result = new Histogram(clip, mode, args[2], args[3].AsInt(8), args[4].AsBool(true),
      args[5].AsBool(true),
      args[6].AsString(""), // matrix_name
      color2_params,
      env);

    AVSValue new_args2[1] = { Result };
    if (vi_orig.IsRGB24()) {
      return env->Invoke("ConvertToRGB24", AVSValue(new_args2, 1)).AsClip();
    }
    else if (vi_orig.IsRGB48()) {
      return env->Invoke("ConvertToRGB48", AVSValue(new_args2, 1)).AsClip();
    }
    else if (vi_orig.IsRGB32()) {
      return env->Invoke("ConvertToRGB32", AVSValue(new_args2, 1)).AsClip();
    }
    else { // if (vi_orig.IsRGB64())
      return env->Invoke("ConvertToRGB64", AVSValue(new_args2, 1)).AsClip();
    }
  }
  else {
    return new Histogram(args[0].AsClip(), mode, args[2], args[3].AsInt(8), args[4].AsBool(true),
      args[5].AsBool(true), // markers
      args[6].AsString(""), // matrix_name
      color2_params,
      env);
  }
}
