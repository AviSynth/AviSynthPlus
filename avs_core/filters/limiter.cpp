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


#include "limiter.h"
#ifdef INTEL_INTRINSICS
#include "intel/limiter_sse.h"
#endif
#include <avs/alignment.h>

#ifdef AVS_WINDOWS
    #include <avs/win.h>
#else
    #include <avs/posix.h>
#endif

#include <../core/internal.h>


Limiter::Limiter(PClip _child, float _min_luma, float _max_luma, float _min_chroma, float _max_chroma, int _show, bool paramscale, IScriptEnvironment* env) :
  GenericVideoFilter(_child),
  max_luma((int)_max_luma),
  min_luma((int)_min_luma),
  max_chroma((int)_max_chroma),
  min_chroma((int)_min_chroma),
  max_luma_f(_max_luma),
  min_luma_f(_min_luma),
  max_chroma_f(_max_chroma),
  min_chroma_f(_min_chroma),
  show(show_e(_show))
{
  if (!vi.IsYUV() && !vi.IsYUVA())
      env->ThrowError("Limiter: Source must be YUV or YUVA");

  if(show != show_none && !vi.IsYUY2() && !vi.Is444() && !vi.Is420())
      env->ThrowError("Limiter: Source must be YUV(A) 4:4:4, 4:2:0 or YUY2 with show option.");

  pixelsize = vi.ComponentSize();
  bits_per_pixel = vi.BitsPerComponent(); // 8,10..16
  int pixel_max = (1 << bits_per_pixel) - 1; // 255, 1023, 4095, 16383, 65535

  const bool isFloat = bits_per_pixel == 32;
  int tv_range_low   = isFloat ? 16 : (16 << (bits_per_pixel - 8)); // 16
  int tv_range_hi_luma   = isFloat ? 235 : (235 << (bits_per_pixel - 8)); // 16-235
  int tv_range_hi_chroma = isFloat ? 240 : (240 << (bits_per_pixel - 8)); // 16-240,64-960, ...

  float tv_range_low_luma_f = 16 / 255.0f;
  float tv_range_hi_luma_f = 235 / 255.0f;
#ifdef FLOAT_CHROMA_IS_HALF_CENTERED
  float tv_range_low_chroma = (16-128) / 255.0f + 0.5f;
  float tv_range_hi_chroma_f = (240 - 128) / 255.0f + 0.5f;
#else
  float tv_range_low_chroma_f = (16 - 128) / 255.0f;
  float tv_range_hi_chroma_f = (240 - 128) / 255.0f;
#endif

  if (paramscale) {
    // int versions, scale from the original float accuracy (e.g. 127.5)
    min_luma = (int)(_min_luma * (1 << (bits_per_pixel - 8)) + 0.5f);
    max_luma = (int)(_max_luma * (1 << (bits_per_pixel - 8)) + 0.5f);
    min_chroma = (int)(_min_chroma * (1 << (bits_per_pixel - 8)) + 0.5f);
    max_chroma = (int)(_max_chroma * (1 << (bits_per_pixel - 8)) + 0.5f);

    // float versions
    min_luma_f = min_luma_f / 255.0f;
    max_luma_f = max_luma_f / 255.0f;
#ifdef FLOAT_CHROMA_IS_HALF_CENTERED
    min_chroma_f = (min_chroma_f - 128) / 255.0f + 0.5f;
    max_chroma_f = (max_chroma_f - 128) / 255.0f + 0.5f;
#else
    min_chroma_f = (min_chroma_f - 128) / 255.0f;
    max_chroma_f = (max_chroma_f - 128) / 255.0f;
#endif
  }


  // default min and max values by bitdepths
  if (_min_luma == -9999.0f) {
    min_luma = tv_range_low;
    min_luma_f = tv_range_low_luma_f;
  }
  if (_max_luma == -9999.0f) {
    max_luma = tv_range_hi_luma;
    max_luma_f = tv_range_hi_luma_f;
  }
  if (_min_chroma == -9999.0f) {
    min_chroma = tv_range_low;
    min_chroma_f = tv_range_low_chroma_f;
  }
  if (_max_chroma == -9999.0f) {
    max_chroma = tv_range_hi_chroma;
    max_chroma_f = tv_range_hi_chroma_f;
  }

  if (pixelsize != 4) {
    // no check for float
    if ((min_luma < 0) || (min_luma > pixel_max))
      env->ThrowError("Limiter: Invalid minimum luma");
    if ((max_luma < 0) || (max_luma > pixel_max))
      env->ThrowError("Limiter: Invalid maximum luma");
    if ((min_chroma < 0) || (min_chroma > pixel_max))
      env->ThrowError("Limiter: Invalid minimum chroma");
    if ((max_chroma < 0) || (max_chroma > pixel_max))
      env->ThrowError("Limiter: Invalid maximum chroma");
  }

}

template<typename pixel_t>
static void limit_plane_c(BYTE *srcp8, int pitch, int min, int max, int width, int height) {
  pixel_t *srcp = reinterpret_cast<pixel_t *>(srcp8);
  pitch /= sizeof(pixel_t);
  for(int y = 0; y < height; y++) {
    for(int x = 0; x < width; x++) {
      if(srcp[x] < min )
        srcp[x] = (pixel_t)min;
      else if(srcp[x] > max)
        srcp[x] = (pixel_t)max;
    }
    srcp += pitch;
  }
}

static void limit_plane_f_c(BYTE *srcp8, int pitch, float min, float max, int width, int height) {
  float *srcp = reinterpret_cast<float *>(srcp8);
  pitch /= sizeof(float);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      if (srcp[x] < min)
        srcp[x] = min;
      else if (srcp[x] > max)
        srcp[x] = max;
    }
    srcp += pitch;
  }
}

template<typename pixel_t, bool show_luma_grey>
static void show_luma_with_grey_opt_yuv444(BYTE *srcp8, BYTE *srcpU8, BYTE *srcpV8, int pitch, int pitchUV, int width, int height, int min_luma, int max_luma, int bits_per_pixel)
{
  // show_luma       Mark clamped pixels red/green over a colour image
  // show_luma_grey  Mark clamped pixels red/green over a greyscaled image
    const int shift = sizeof(pixel_t) == 1 ? 0 : (bits_per_pixel - 8);
    pixel_t *srcp = reinterpret_cast<pixel_t *>(srcp8);
    pixel_t *srcpU = reinterpret_cast<pixel_t *>(srcpU8);
    pixel_t *srcpV = reinterpret_cast<pixel_t *>(srcpV8);
    pitch /= sizeof(pixel_t);
    pitchUV /= sizeof(pixel_t);

    for (int h=0; h < height; h+=1) {
      for (int x = 0; x < width; x+=1) {
        if      (srcp[x] < min_luma) { srcp[x] =  81 << shift; srcpU[x] = 91 << shift; srcpV[x] = 240 << shift; }       // red:   Y=81, U=91 and V=240
        else if (srcp[x] > max_luma) { srcp[x] = 145 << shift; srcpU[x] = 54 << shift; srcpV[x] =  34 << shift; }       // green: Y=145, U=54 and V=34
        // this differs from show_luma
        else if(show_luma_grey)      {                srcpU[x] =     srcpV[x] = 128 << shift; }       // grey
      }
      srcp  += pitch;
      srcpV += pitchUV;
      srcpU += pitchUV;
    }
}

template<typename pixel_t, bool show_luma_grey>
static void show_luma_with_grey_opt_yuv420(BYTE *srcp8, BYTE *srcpU8, BYTE *srcpV8, int pitch, int pitchUV, int width, int height, int min_luma, int max_luma, int bits_per_pixel)
{
  // show_luma       Mark clamped pixels red/green over a colour image
  // show_luma_grey  Mark clamped pixels red/green over a greyscaled image
  const int shift = sizeof(pixel_t) == 1 ? 0 : (bits_per_pixel - 8);
  pixel_t *srcp = reinterpret_cast<pixel_t *>(srcp8);
  pixel_t *srcn = reinterpret_cast<pixel_t *>(srcp8 + pitch); // next line
  pixel_t *srcpU = reinterpret_cast<pixel_t *>(srcpU8);
  pixel_t *srcpV = reinterpret_cast<pixel_t *>(srcpV8);
  pitch /= sizeof(pixel_t);
  pitchUV /= sizeof(pixel_t);

  for (int h=0; h < height; h+=2) {
    for (int x = 0; x < width; x+=2) {
      int uv = 0;
      if      (srcp[x  ] < min_luma) { srcp[x  ] =  81 << shift; uv |= 1;}
      else if (srcp[x  ] > max_luma) { srcp[x  ] = 145 << shift; uv |= 2;}
      if      (srcp[x+1] < min_luma) { srcp[x+1] =  81 << shift; uv |= 1;}
      else if (srcp[x+1] > max_luma) { srcp[x+1] = 145 << shift; uv |= 2;}
      if      (srcn[x  ] < min_luma) { srcn[x  ] =  81 << shift; uv |= 1;}
      else if (srcn[x  ] > max_luma) { srcn[x  ] = 145 << shift; uv |= 2;}
      if      (srcn[x+1] < min_luma) { srcn[x+1] =  81 << shift; uv |= 1;}
      else if (srcn[x+1] > max_luma) { srcn[x+1] = 145 << shift; uv |= 2;}
      switch (uv) {
      case 1: srcpU[x/2] = 91 << shift; srcpV[x/2] = 240 << shift; break;       // red:   Y=81, U=91 and V=240
      case 2: srcpU[x/2] = 54 << shift; srcpV[x/2] =  34 << shift; break;       // green: Y=145, U=54 and V=34
      // this differs from show_luma_grey
      case 3:
        if(show_luma_grey) {
          srcpU[x/2] = 90 << shift;
          srcpV[x/2] = 134 << shift; break;       // puke:  Y=81, U=90 and V=134 olive: Y=145, U=90 and V=134
        } else {
        srcp[x]=srcp[x+2]=srcn[x]=srcn[x+2]=210 << shift; // yellow:Y=210, U=16 and V=146
        srcpU[x/2] = 16 << shift;
        srcpV[x/2] = 146 << shift;
        }
        break;
      default:
        if(show_luma_grey) {
          srcpU[x/2] = srcpV[x/2] = 128 << shift; // olive: Y=145, U=90 and V=134
        }
        break;
      }
    }
    srcp += pitch*2; // 2x2 pixels at a time (4:2:0 subsampling)
    srcn += pitch*2;
    srcpV += pitchUV;
    srcpU += pitchUV;
  }
}

template<bool show_luma_grey>
static void show_luma_with_grey_opt_yuv444_f(BYTE *srcp8, BYTE *srcpU8, BYTE *srcpV8, int pitch, int pitchUV, int width, int height, float min_luma, float max_luma)
{
  // show_luma       Mark clamped pixels red/green over a colour image
  // show_luma_grey  Mark clamped pixels red/green over a greyscaled image
  float *srcp = reinterpret_cast<float *>(srcp8);
  float *srcpU = reinterpret_cast<float *>(srcpU8);
  float *srcpV = reinterpret_cast<float *>(srcpV8);
  pitch /= sizeof(float);
  pitchUV /= sizeof(float);

  for (int h = 0; h < height; h += 1) {
    for (int x = 0; x < width; x += 1) {
      if (srcp[x] < min_luma) { srcp[x] =  c8tof(81); srcpU[x] = uv8tof(91); srcpV[x] = uv8tof(240); }       // red:   Y=81, U=91 and V=240
      else if (srcp[x] > max_luma) { srcp[x] = c8tof(145); srcpU[x] = uv8tof(54); srcpV[x] = uv8tof(34); }       // green: Y=145, U=54 and V=34
                                                                                                                     // this differs from show_luma
      else if (show_luma_grey) { srcpU[x] = srcpV[x] = uv8tof(128); }       // grey
    }
    srcp += pitch;
    srcpV += pitchUV;
    srcpU += pitchUV;
  }
}

template<bool show_luma_grey>
static void show_luma_with_grey_opt_yuv420_f(BYTE *srcp8, BYTE *srcpU8, BYTE *srcpV8, int pitch, int pitchUV, int width, int height, float min_luma, float max_luma)
{
  // show_luma       Mark clamped pixels red/green over a colour image
  // show_luma_grey  Mark clamped pixels red/green over a greyscaled image
  float *srcp = reinterpret_cast<float *>(srcp8);
  float *srcn = reinterpret_cast<float *>(srcp8 + pitch); // next line
  float *srcpU = reinterpret_cast<float *>(srcpU8);
  float *srcpV = reinterpret_cast<float *>(srcpV8);
  pitch /= sizeof(float);
  pitchUV /= sizeof(float);

  for (int h = 0; h < height; h += 2) {
    for (int x = 0; x < width; x += 2) {
      int uv = 0;
      if (srcp[x] < min_luma) { srcp[x] = c8tof(81); uv |= 1; }
      else if (srcp[x] > max_luma) { srcp[x] = c8tof(145); uv |= 2; }
      if (srcp[x + 1] < min_luma) { srcp[x + 1] = c8tof(81); uv |= 1; }
      else if (srcp[x + 1] > max_luma) { srcp[x + 1] = c8tof(145); uv |= 2; }
      if (srcn[x] < min_luma) { srcn[x] = c8tof(81); uv |= 1; }
      else if (srcn[x] > max_luma) { srcn[x] = c8tof(145); uv |= 2; }
      if (srcn[x + 1] < min_luma) { srcn[x + 1] = c8tof(81); uv |= 1; }
      else if (srcn[x + 1] > max_luma) { srcn[x + 1] = c8tof(145); uv |= 2; }
      switch (uv) {
      case 1: srcpU[x / 2] = uv8tof(91); srcpV[x / 2] = uv8tof(240); break;       // red:   Y=81, U=91 and V=240
      case 2: srcpU[x / 2] = uv8tof(54); srcpV[x / 2] = uv8tof(34); break;       // green: Y=145, U=54 and V=34
                                                                                   // this differs from show_luma_grey
      case 3:
        if (show_luma_grey) {
          srcpU[x / 2] = uv8tof(90);
          srcpV[x / 2] = uv8tof(134); break;       // puke:  Y=81, U=90 and V=134 olive: Y=145, U=90 and V=134
        }
        else {
          srcp[x] = srcp[x + 2] = srcn[x] = srcn[x + 2] = c8tof(210); // yellow:Y=210, U=16 and V=146
          srcpU[x / 2] = uv8tof(16);
          srcpV[x / 2] = uv8tof(146);
        }
        break;
      default:
        if (show_luma_grey) {
          srcpU[x / 2] = srcpV[x / 2] = uv8tof(128); // olive: Y=145, U=90 and V=134
        }
        break;
      }
    }
    srcp += pitch * 2; // 2x2 pixels at a time (4:2:0 subsampling)
    srcn += pitch * 2;
    srcpV += pitchUV;
    srcpU += pitchUV;
  }
}

template<typename pixel_t>
static void show_chroma_yuv444(BYTE *srcp8, BYTE *srcpU8, BYTE *srcpV8, int pitch, int pitchUV, int width, int height, int min_chroma, int max_chroma, int bits_per_pixel)
{
  const int shift = sizeof(pixel_t) == 1 ? 0 : (bits_per_pixel - 8);
  pixel_t *srcp = reinterpret_cast<pixel_t *>(srcp8);
  pixel_t *srcpU = reinterpret_cast<pixel_t *>(srcpU8);
  pixel_t *srcpV = reinterpret_cast<pixel_t *>(srcpV8);
  pitch /= sizeof(pixel_t);
  pitchUV /= sizeof(pixel_t);

  for (int h=0; h < height; h+=1) {
    for (int x = 0; x < width; x+=1) {
      if ( (srcpU[x] < min_chroma)  // U-
        || (srcpU[x] > max_chroma)  // U+
        || (srcpV[x] < min_chroma)  // V-
        || (srcpV[x] > max_chroma) )// V+
      { srcp[x]=210 << shift; srcpU[x]= 16 << shift; srcpV[x]=146 << shift; }   // yellow:Y=210, U=16 and V=146
    }
    srcp  += pitch;
    srcpV += pitchUV;
    srcpU += pitchUV;
  }
}

template<typename pixel_t>
static void show_chroma_yuv420(BYTE *srcp8, BYTE *srcpU8, BYTE *srcpV8, int pitch, int pitchUV, int width, int height, int min_chroma, int max_chroma, int bits_per_pixel)
{
  const int shift = sizeof(pixel_t) == 1 ? 0 : (bits_per_pixel - 8);
  pixel_t *srcp = reinterpret_cast<pixel_t *>(srcp8);
  pixel_t *srcn = reinterpret_cast<pixel_t *>(srcp8 + pitch); // next line
  pixel_t *srcpU = reinterpret_cast<pixel_t *>(srcpU8);
  pixel_t *srcpV = reinterpret_cast<pixel_t *>(srcpV8);
  pitch /= sizeof(pixel_t);
  pitchUV /= sizeof(pixel_t);

  for (int h=0; h < height; h+=2) {
    for (int x = 0; x < width; x+=2) {
      if ( (srcpU[x/2] < min_chroma)  // U-
        || (srcpU[x/2] > max_chroma)  // U+
        || (srcpV[x/2] < min_chroma)  // V-
        || (srcpV[x/2] > max_chroma) )// V+
      { srcp[x]=srcp[x+1]=srcn[x]=srcn[x+1]=210 << shift; srcpU[x/2]= 16 << shift; srcpV[x/2]=146 << shift; }   // yellow:Y=210, U=16 and V=146
    }
    srcp += pitch*2; // 2x2 pixels at a time (4:2:0 subsampling)
    srcn += pitch*2;
    srcpV += pitchUV;
    srcpU += pitchUV;
  }
}

template<typename pixel_t>
static void show_chroma_grey_yuv444(BYTE *srcp8, BYTE *srcpU8, BYTE *srcpV8, int pitch, int pitchUV, int width, int height, int min_chroma, int max_chroma, int bits_per_pixel)
{
  const int shift = sizeof(pixel_t) == 1 ? 0 : (bits_per_pixel - 8);
  pixel_t *srcp = reinterpret_cast<pixel_t *>(srcp8);
  pixel_t *srcpU = reinterpret_cast<pixel_t *>(srcpU8);
  pixel_t *srcpV = reinterpret_cast<pixel_t *>(srcpV8);
  pitch /= sizeof(pixel_t);
  pitchUV /= sizeof(pixel_t);

  for (int h=0; h < height; h+=1) {
    for (int x = 0; x < width; x+=1) {
      int uv = 0;
      if      (srcpU[x] < min_chroma) uv |= 1; // U-
      else if (srcpU[x] > max_chroma) uv |= 2; // U+
      if      (srcpV[x] < min_chroma) uv |= 4; // V-
      else if (srcpV[x] > max_chroma) uv |= 8; // V+
      switch (uv) {
      case  8: srcp[x]= 81 << shift; srcpU[x]= 91 << shift; srcpV[x]=240 << shift; break;   //   +V Red
      case  9: srcp[x]=146 << shift; srcpU[x]= 53 << shift; srcpV[x]=193 << shift; break;   // -U+V Orange
      case  1: srcp[x]=210 << shift; srcpU[x]= 16 << shift; srcpV[x]=146 << shift; break;   // -U   Yellow
      case  5: srcp[x]=153 << shift; srcpU[x]= 49 << shift; srcpV[x]= 49 << shift; break;   // -U-V Green
      case  4: srcp[x]=170 << shift; srcpU[x]=165 << shift; srcpV[x]= 16 << shift; break;   //   -V Cyan
      case  6: srcp[x]=105 << shift; srcpU[x]=203 << shift; srcpV[x]= 63 << shift; break;   // +U-V Teal
      case  2: srcp[x]= 41 << shift; srcpU[x]=240 << shift; srcpV[x]=110 << shift; break;   // +U   Blue
      case 10: srcp[x]=106 << shift; srcpU[x]=202 << shift; srcpV[x]=222 << shift; break;   // +U+V Magenta
      default:              srcpU[x]=     srcpV[x]=128 << shift; break;
      }
    }
    srcp  += pitch;
    srcpV += pitchUV;
    srcpU += pitchUV;
  }
}

template<typename pixel_t>
static void show_chroma_grey_yuv420(BYTE *srcp8, BYTE *srcpU8, BYTE *srcpV8, int pitch, int pitchUV, int width, int height, int min_chroma, int max_chroma, int bits_per_pixel)
{
  const int shift = sizeof(pixel_t) == 1 ? 0 : (bits_per_pixel - 8);
  pixel_t *srcp = reinterpret_cast<pixel_t *>(srcp8);
  pixel_t *srcn = reinterpret_cast<pixel_t *>(srcp8 + pitch); // next line
  pixel_t *srcpU = reinterpret_cast<pixel_t *>(srcpU8);
  pixel_t *srcpV = reinterpret_cast<pixel_t *>(srcpV8);
  pitch /= sizeof(pixel_t);
  pitchUV /= sizeof(pixel_t);

  for (int h=0; h < height; h+=2) {
    for (int x = 0; x < width; x+=2) {
      int uv = 0;
      if      (srcpU[x/2] < min_chroma) uv |= 1; // U-
      else if (srcpU[x/2] > max_chroma) uv |= 2; // U+
      if      (srcpV[x/2] < min_chroma) uv |= 4; // V-
      else if (srcpV[x/2] > max_chroma) uv |= 8; // V+
      switch (uv) {
      case  8: srcp[x]=srcp[x+1]=srcn[x]=srcn[x+1]= 81 << shift; srcpU[x/2]= 91 << shift; srcpV[x/2]=240 << shift; break;   //   +V Red
      case  9: srcp[x]=srcp[x+1]=srcn[x]=srcn[x+1]=146 << shift; srcpU[x/2]= 53 << shift; srcpV[x/2]=193 << shift; break;   // -U+V Orange
      case  1: srcp[x]=srcp[x+1]=srcn[x]=srcn[x+1]=210 << shift; srcpU[x/2]= 16 << shift; srcpV[x/2]=146 << shift; break;   // -U   Yellow
      case  5: srcp[x]=srcp[x+1]=srcn[x]=srcn[x+1]=153 << shift; srcpU[x/2]= 49 << shift; srcpV[x/2]= 49 << shift; break;   // -U-V Green
      case  4: srcp[x]=srcp[x+1]=srcn[x]=srcn[x+1]=170 << shift; srcpU[x/2]=165 << shift; srcpV[x/2]= 16 << shift; break;   //   -V Cyan
      case  6: srcp[x]=srcp[x+1]=srcn[x]=srcn[x+1]=105 << shift; srcpU[x/2]=203 << shift; srcpV[x/2]= 63 << shift; break;   // +U-V Teal
      case  2: srcp[x]=srcp[x+1]=srcn[x]=srcn[x+1]= 41 << shift; srcpU[x/2]=240 << shift; srcpV[x/2]=110 << shift; break;   // +U   Blue
      case 10: srcp[x]=srcp[x+1]=srcn[x]=srcn[x+1]=106 << shift; srcpU[x/2]=202 << shift; srcpV[x/2]=222 << shift; break;   // +U+V Magenta
      default: srcpU[x/2] = srcpV[x/2] = 128 << shift; break;
      }
    }
    srcp += pitch*2; // 2x2 pixels at a time (4:2:0 subsampling)
    srcn += pitch*2;
    srcpV += pitchUV;
    srcpU += pitchUV;
  }
}

static void show_chroma_yuv444_f(BYTE *srcp8, BYTE *srcpU8, BYTE *srcpV8, int pitch, int pitchUV, int width, int height, float min_chroma, float max_chroma)
{
  float *srcp = reinterpret_cast<float *>(srcp8);
  float *srcpU = reinterpret_cast<float *>(srcpU8);
  float *srcpV = reinterpret_cast<float *>(srcpV8);
  pitch /= sizeof(float);
  pitchUV /= sizeof(float);

  for (int h = 0; h < height; h += 1) {
    for (int x = 0; x < width; x += 1) {
      if ((srcpU[x] < min_chroma)  // U-
        || (srcpU[x] > max_chroma)  // U+
        || (srcpV[x] < min_chroma)  // V-
        || (srcpV[x] > max_chroma))// V+
      {
        srcp[x] = c8tof(210); srcpU[x] = uv8tof(16); srcpV[x] = uv8tof(146);
      }   // yellow:Y=210, U=16 and V=146
    }
    srcp += pitch;
    srcpV += pitchUV;
    srcpU += pitchUV;
  }
}

static void show_chroma_yuv420_f(BYTE *srcp8, BYTE *srcpU8, BYTE *srcpV8, int pitch, int pitchUV, int width, int height, float min_chroma, float max_chroma)
{
  float *srcp = reinterpret_cast<float *>(srcp8);
  float *srcn = reinterpret_cast<float *>(srcp8 + pitch); // next line
  float *srcpU = reinterpret_cast<float *>(srcpU8);
  float *srcpV = reinterpret_cast<float *>(srcpV8);
  pitch /= sizeof(float);
  pitchUV /= sizeof(float);

  for (int h = 0; h < height; h += 2) {
    for (int x = 0; x < width; x += 2) {
      if ((srcpU[x / 2] < min_chroma)  // U-
        || (srcpU[x / 2] > max_chroma)  // U+
        || (srcpV[x / 2] < min_chroma)  // V-
        || (srcpV[x / 2] > max_chroma))// V+
      {
        srcp[x] = srcp[x + 1] = srcn[x] = srcn[x + 1] = c8tof(210); srcpU[x / 2] = uv8tof(16); srcpV[x / 2] = uv8tof(146);
      }   // yellow:Y=210, U=16 and V=146
    }
    srcp += pitch * 2; // 2x2 pixels at a time (4:2:0 subsampling)
    srcn += pitch * 2;
    srcpV += pitchUV;
    srcpU += pitchUV;
  }
}

static void show_chroma_grey_yuv444_f(BYTE *srcp8, BYTE *srcpU8, BYTE *srcpV8, int pitch, int pitchUV, int width, int height, float min_chroma, float max_chroma)
{
  float *srcp = reinterpret_cast<float *>(srcp8);
  float *srcpU = reinterpret_cast<float *>(srcpU8);
  float *srcpV = reinterpret_cast<float *>(srcpV8);
  pitch /= sizeof(float);
  pitchUV /= sizeof(float);

  for (int h = 0; h < height; h += 1) {
    for (int x = 0; x < width; x += 1) {
      int uv = 0;
      if (srcpU[x] < min_chroma) uv |= 1; // U-
      else if (srcpU[x] > max_chroma) uv |= 2; // U+
      if (srcpV[x] < min_chroma) uv |= 4; // V-
      else if (srcpV[x] > max_chroma) uv |= 8; // V+
      switch (uv) {
      case  8: srcp[x] = c8tof(81); srcpU[x] = uv8tof(91); srcpV[x] = uv8tof(240); break;   //   +V Red
      case  9: srcp[x] = c8tof(146); srcpU[x] = uv8tof(53); srcpV[x] = uv8tof(193); break;   // -U+V Orange
      case  1: srcp[x] = c8tof(210); srcpU[x] = uv8tof(16); srcpV[x] = uv8tof(146); break;   // -U   Yellow
      case  5: srcp[x] = c8tof(153); srcpU[x] = uv8tof(49); srcpV[x] = uv8tof(49); break;   // -U-V Green
      case  4: srcp[x] = c8tof(170); srcpU[x] = uv8tof(165); srcpV[x] = uv8tof(16); break;   //   -V Cyan
      case  6: srcp[x] = c8tof(105); srcpU[x] = uv8tof(203); srcpV[x] = uv8tof(63); break;   // +U-V Teal
      case  2: srcp[x] = c8tof(41); srcpU[x] = uv8tof(240); srcpV[x] = uv8tof(110); break;   // +U   Blue
      case 10: srcp[x] = c8tof(106); srcpU[x] = uv8tof(202); srcpV[x] = uv8tof(222); break;   // +U+V Magenta
      default:              srcpU[x] = srcpV[x] = uv8tof(128); break;
      }
    }
    srcp += pitch;
    srcpV += pitchUV;
    srcpU += pitchUV;
  }
}

static void show_chroma_grey_yuv420_f(BYTE *srcp8, BYTE *srcpU8, BYTE *srcpV8, int pitch, int pitchUV, int width, int height, float min_chroma, float max_chroma)
{
  float *srcp = reinterpret_cast<float *>(srcp8);
  float *srcn = reinterpret_cast<float *>(srcp8 + pitch); // next line
  float *srcpU = reinterpret_cast<float *>(srcpU8);
  float *srcpV = reinterpret_cast<float *>(srcpV8);
  pitch /= sizeof(float);
  pitchUV /= sizeof(float);

  for (int h = 0; h < height; h += 2) {
    for (int x = 0; x < width; x += 2) {
      int uv = 0;
      if (srcpU[x / 2] < min_chroma) uv |= 1; // U-
      else if (srcpU[x / 2] > max_chroma) uv |= 2; // U+
      if (srcpV[x / 2] < min_chroma) uv |= 4; // V-
      else if (srcpV[x / 2] > max_chroma) uv |= 8; // V+
      switch (uv) {
      case  8: srcp[x] = srcp[x + 1] = srcn[x] = srcn[x + 1] = c8tof(81); srcpU[x / 2] = uv8tof(91); srcpV[x / 2] = uv8tof(240); break;   //   +V Red
      case  9: srcp[x] = srcp[x + 1] = srcn[x] = srcn[x + 1] = c8tof(146); srcpU[x / 2] = uv8tof(53); srcpV[x / 2] = uv8tof(193); break;   // -U+V Orange
      case  1: srcp[x] = srcp[x + 1] = srcn[x] = srcn[x + 1] = c8tof(210); srcpU[x / 2] = uv8tof(16); srcpV[x / 2] = uv8tof(146); break;   // -U   Yellow
      case  5: srcp[x] = srcp[x + 1] = srcn[x] = srcn[x + 1] = c8tof(153); srcpU[x / 2] = uv8tof(49); srcpV[x / 2] = uv8tof(49); break;   // -U-V Green
      case  4: srcp[x] = srcp[x + 1] = srcn[x] = srcn[x + 1] = c8tof(170); srcpU[x / 2] = uv8tof(165); srcpV[x / 2] = uv8tof(16); break;   //   -V Cyan
      case  6: srcp[x] = srcp[x + 1] = srcn[x] = srcn[x + 1] = c8tof(105); srcpU[x / 2] = uv8tof(203); srcpV[x / 2] = uv8tof(63); break;   // +U-V Teal
      case  2: srcp[x] = srcp[x + 1] = srcn[x] = srcn[x + 1] = c8tof(41); srcpU[x / 2] = uv8tof(240); srcpV[x / 2] = uv8tof(110); break;   // +U   Blue
      case 10: srcp[x] = srcp[x + 1] = srcn[x] = srcn[x + 1] = c8tof(106); srcpU[x / 2] = uv8tof(202); srcpV[x / 2] = uv8tof(222); break;   // +U+V Magenta
      default: srcpU[x / 2] = srcpV[x / 2] = uv8tof(128); break;
      }
    }
    srcp += pitch * 2; // 2x2 pixels at a time (4:2:0 subsampling)
    srcn += pitch * 2;
    srcpV += pitchUV;
    srcpU += pitchUV;
  }
}

PVideoFrame __stdcall Limiter::GetFrame(int n, IScriptEnvironment* env) {
  PVideoFrame frame = child->GetFrame(n, env);
  env->MakeWritable(&frame);
  unsigned char* srcp = frame->GetWritePtr();
  int pitch = frame->GetPitch();
  int row_size = frame->GetRowSize();
  int width = vi.width;
  int height = vi.height;

  if (vi.IsYUY2()) {

		if (show == show_luma) {  // Mark clamped pixels red/yellow/green over a colour image
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < row_size; x+=4) {
					int uv = 0;
					if      (srcp[x  ] < min_luma) { srcp[x  ] =  81; uv |= 1;}
					else if (srcp[x  ] > max_luma) { srcp[x  ] = 145; uv |= 2;}
					if      (srcp[x+2] < min_luma) { srcp[x+2] =  81; uv |= 1;}
					else if (srcp[x+2] > max_luma) { srcp[x+2] = 145; uv |= 2;}
					switch (uv) {
						case 1: srcp[x+1] = 91; srcp[x+3] = 240; break;     // red:   Y= 81, U=91 and V=240
						case 2: srcp[x+1] = 54; srcp[x+3] =  34; break;     // green: Y=145, U=54 and V=34
						case 3: srcp[x  ] =     srcp[x+2] = 210;
						        srcp[x+1] = 16; srcp[x+3] = 146; break;     // yellow:Y=210, U=16 and V=146
						default: break;
					}
				}
				srcp += pitch;
			}
			return frame;
		}
		else if (show == show_luma_grey) {    // Mark clamped pixels coloured over a greyscaled image
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < row_size; x+=4) {
					int uv = 0;
					if      (srcp[x  ] < min_luma) { srcp[x  ] =  81; uv |= 1;}
					else if (srcp[x  ] > max_luma) { srcp[x  ] = 145; uv |= 2;}
					if      (srcp[x+2] < min_luma) { srcp[x+2] =  81; uv |= 1;}
					else if (srcp[x+2] > max_luma) { srcp[x+2] = 145; uv |= 2;}
					switch (uv) {
						case 1: srcp[x+1] = 91; srcp[x+3] = 240; break;     // red:   Y=81, U=91 and V=240
						case 2: srcp[x+1] = 54; srcp[x+3] =  34; break;     // green: Y=145, U=54 and V=34
						case 3: srcp[x+1] = 90; srcp[x+3] = 134; break;     // puke:  Y=81, U=90 and V=134
						default: srcp[x+1] = srcp[x+3] = 128; break;        // olive: Y=145, U=90 and V=134
					}
				}
				srcp += pitch;
			}
			return frame;
		}
		else if (show == show_chroma) {    // Mark clamped pixels yellow over a colour image
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < row_size; x+=4) {
					if ( (srcp[x+1] < min_chroma)  // U-
					  || (srcp[x+1] > max_chroma)  // U+
					  || (srcp[x+3] < min_chroma)  // V-
					  || (srcp[x+3] > max_chroma) )// V+
					 { srcp[x]=srcp[x+2]=210; srcp[x+1]=16; srcp[x+3]=146; }    // yellow:Y=210, U=16 and V=146
				}
				srcp += pitch;
			}
			return frame;
		}
		else if (show == show_chroma_grey) {    // Mark clamped pixels coloured over a greyscaled image
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < row_size; x+=4) {
					int uv = 0;
					if      (srcp[x+1] < min_chroma) uv |= 1; // U-
					else if (srcp[x+1] > max_chroma) uv |= 2; // U+
					if      (srcp[x+3] < min_chroma) uv |= 4; // V-
					else if (srcp[x+3] > max_chroma) uv |= 8; // V+
					switch (uv) {
						case  8: srcp[x] = srcp[x+2] =  81; srcp[x+1] =  91; srcp[x+3] = 240; break;    //   +V Red
						case  9: srcp[x] = srcp[x+2] = 146; srcp[x+1] =  53; srcp[x+3] = 193; break;    // -U+V Orange
						case  1: srcp[x] = srcp[x+2] = 210; srcp[x+1] =  16; srcp[x+3] = 146; break;    // -U   Yellow
						case  5: srcp[x] = srcp[x+2] = 153; srcp[x+1] =  49; srcp[x+3] =  49; break;    // -U-V Green
						case  4: srcp[x] = srcp[x+2] = 170; srcp[x+1] = 165; srcp[x+3] =  16; break;    //   -V Cyan
						case  6: srcp[x] = srcp[x+2] = 105; srcp[x+1] = 203; srcp[x+3] =  63; break;    // +U-V Teal
						case  2: srcp[x] = srcp[x+2] =  41; srcp[x+1] = 240; srcp[x+3] = 110; break;    // +U   Blue
						case 10: srcp[x] = srcp[x+2] = 106; srcp[x+1] = 202; srcp[x+3] = 222; break;    // +U+V Magenta
						default: srcp[x+1] = srcp[x+3] = 128; break;
					}
				}
				srcp += pitch;
			}
			return frame;
		}
#ifdef INTEL_INTRINSICS
    if (env->GetCPUFlags() & CPUF_SSE2) {
      limit_plane_sse2(srcp, min_luma | (min_chroma << 8), max_luma | (max_chroma << 8), pitch, row_size, height);
      return frame;
    }

    /** Run emulator if CPU supports it**/
#ifdef X86_32
    if (env->GetCPUFlags() & CPUF_INTEGER_SSE)
    {
      //limit_plane_mmx(srcp, min_luma, max_luma, pitch, row_size, height);
      limit_plane_isse(srcp, min_luma | (min_chroma << 8), max_luma | (max_chroma << 8), pitch, row_size, height);
      return frame;
    }
#endif
    // If not ISSE
#endif
    for(int y = 0; y < height; y++) {
      for(int x = 0; x < row_size; x++) {
        if(srcp[x] < min_luma )
          srcp[x++] = (unsigned char)min_luma;
        else if(srcp[x] > max_luma)
          srcp[x++] = (unsigned char)max_luma;
        else
          x++;
        if(srcp[x] < min_chroma)
          srcp[x] = (unsigned char)min_chroma;
        else if(srcp[x] > max_chroma)
          srcp[x] = (unsigned char)max_chroma;
      }
      srcp += pitch;
    }
    return frame;
    // YUY end
  } else if(vi.Is420() && show != show_none) {
    const int pitchUV = frame->GetPitch(PLANAR_U);
    unsigned char* srcpV = frame->GetWritePtr(PLANAR_V);
    unsigned char* srcpU = frame->GetWritePtr(PLANAR_U);

    if (show == show_luma || show == show_luma_grey) {    // Mark clamped pixels red/yellow/green over a colour image
      if (pixelsize == 1) {
        if (show == show_luma)
          show_luma_with_grey_opt_yuv420<uint8_t, false>(srcp, srcpU, srcpV, pitch, pitchUV, width, height, min_luma, max_luma, bits_per_pixel);
        else // show_luma_grey
          show_luma_with_grey_opt_yuv420<uint8_t, true>(srcp, srcpU, srcpV, pitch, pitchUV, width, height, min_luma, max_luma, bits_per_pixel);
      } else if (pixelsize == 2) { // pixelsize == 2
        if (show == show_luma)
          show_luma_with_grey_opt_yuv420<uint16_t, false>(srcp, srcpU, srcpV, pitch, pitchUV, width, height, min_luma, max_luma, bits_per_pixel);
        else // show_luma_grey
          show_luma_with_grey_opt_yuv420<uint16_t, true>(srcp, srcpU, srcpV, pitch, pitchUV, width, height, min_luma, max_luma, bits_per_pixel);
      }
      else { // pixelsize == 4
        if (show == show_luma)
          show_luma_with_grey_opt_yuv420_f<false>(srcp, srcpU, srcpV, pitch, pitchUV, width, height, min_luma_f, max_luma_f);
        else // show_luma_grey
          show_luma_with_grey_opt_yuv420_f<true>(srcp, srcpU, srcpV, pitch, pitchUV, width, height, min_luma_f, max_luma_f);
      }
      return frame;
    }
		else if (show == show_chroma) {   // Mark clamped pixels yellow over a colour image
      if (pixelsize == 1)
        show_chroma_yuv420<uint8_t>(srcp, srcpU, srcpV, pitch, pitchUV, width, height, min_chroma, max_chroma, bits_per_pixel);
      else if (pixelsize == 2)
        show_chroma_yuv420<uint16_t>(srcp, srcpU, srcpV, pitch, pitchUV, width, height, min_chroma, max_chroma, bits_per_pixel);
      else // float
        show_chroma_yuv420_f(srcp, srcpU, srcpV, pitch, pitchUV, width, height, min_chroma_f, max_chroma_f);
			return frame;
		}
		else if (show == show_chroma_grey) {   // Mark clamped pixels coloured over a greyscaled image
      if (pixelsize == 1)
        show_chroma_grey_yuv420<uint8_t>(srcp, srcpU, srcpV, pitch, pitchUV, width, height, min_chroma, max_chroma, bits_per_pixel);
      else if (pixelsize == 2)
        show_chroma_grey_yuv420<uint16_t>(srcp, srcpU, srcpV, pitch, pitchUV, width, height, min_chroma, max_chroma, bits_per_pixel);
      else // float
        show_chroma_grey_yuv420_f(srcp, srcpU, srcpV, pitch, pitchUV, width, height, min_chroma_f, max_chroma_f);
			return frame;
		}
    // YV12 (4:2:0) end
  } else if(vi.Is444() && show != show_none) {

    const int pitchUV = frame->GetPitch(PLANAR_U);
    unsigned char* srcpV = frame->GetWritePtr(PLANAR_V);
    unsigned char* srcpU = frame->GetWritePtr(PLANAR_U);

		if (show == show_luma || show == show_luma_grey) {
      if (pixelsize == 1) {
        if (show == show_luma)
          show_luma_with_grey_opt_yuv444<uint8_t, false>(srcp, srcpU, srcpV, pitch, pitchUV, width, height, min_luma, max_luma, bits_per_pixel);
        else // show_luma_grey
          show_luma_with_grey_opt_yuv444<uint8_t, true>(srcp, srcpU, srcpV, pitch, pitchUV, width, height, min_luma, max_luma, bits_per_pixel);
      } else if (pixelsize == 2) {
          if (show == show_luma)
            show_luma_with_grey_opt_yuv444<uint16_t, false>(srcp, srcpU, srcpV, pitch, pitchUV, width, height, min_luma, max_luma, bits_per_pixel);
          else // show_luma_grey
            show_luma_with_grey_opt_yuv444<uint16_t, true>(srcp, srcpU, srcpV, pitch, pitchUV, width, height, min_luma, max_luma, bits_per_pixel);
      }
      else { // pixelsize == 4
        if (show == show_luma)
          show_luma_with_grey_opt_yuv444_f<false>(srcp, srcpU, srcpV, pitch, pitchUV, width, height, min_luma_f, max_luma_f);
        else // show_luma_grey
          show_luma_with_grey_opt_yuv444_f<true>(srcp, srcpU, srcpV, pitch, pitchUV, width, height, min_luma_f, max_luma_f);
      }
      return frame;
    }
		else if (show == show_chroma) {   // Mark clamped pixels yellow over a colour image
      if (pixelsize == 1)
        show_chroma_yuv444<uint8_t>(srcp, srcpU, srcpV, pitch, pitchUV, width, height, min_chroma, max_chroma, bits_per_pixel);
      else if (pixelsize == 2)
        show_chroma_yuv444<uint16_t>(srcp, srcpU, srcpV, pitch, pitchUV, width, height, min_chroma, max_chroma, bits_per_pixel);
      else // float
        show_chroma_yuv444_f(srcp, srcpU, srcpV, pitch, pitchUV, width, height, min_chroma_f, max_chroma_f);
			return frame;
		}
		else if (show == show_chroma_grey) {   // Mark clamped pixels coloured over a greyscaled image
      if (pixelsize == 1)
        show_chroma_grey_yuv444<uint8_t>(srcp, srcpU, srcpV, pitch, pitchUV, width, height, min_chroma, max_chroma, bits_per_pixel);
      else if (pixelsize == 2)
        show_chroma_grey_yuv444<uint16_t>(srcp, srcpU, srcpV, pitch, pitchUV, width, height, min_chroma, max_chroma, bits_per_pixel);
      else // float
        show_chroma_grey_yuv444_f(srcp, srcpU, srcpV, pitch, pitchUV, width, height, min_chroma_f, max_chroma_f);
			return frame;
		}
    // YV24 (4:4:4) end
  }
  if (vi.IsPlanar())
  {
#ifdef INTEL_INTRINSICS
    //todo: separate to functions and use sse2 for aligned planes even if some are unaligned
    if ((pixelsize==1) && (env->GetCPUFlags() & CPUF_SSE2)) {
        limit_plane_sse2(srcp, min_luma | (min_luma << 8), max_luma | (max_luma << 8), pitch, row_size, height);

        limit_plane_sse2(frame->GetWritePtr(PLANAR_U), min_chroma | (min_chroma << 8), max_chroma | (max_chroma << 8),
          frame->GetPitch(PLANAR_U), frame->GetRowSize(PLANAR_U), frame->GetHeight(PLANAR_U));

        limit_plane_sse2(frame->GetWritePtr(PLANAR_V), min_chroma | (min_chroma << 8), max_chroma | (max_chroma << 8),
          frame->GetPitch(PLANAR_V), frame->GetRowSize(PLANAR_V), frame->GetHeight(PLANAR_V));

        return frame;
    }

#ifdef X86_32
    if ((pixelsize==1) && (env->GetCPUFlags() & CPUF_INTEGER_SSE))
    {
      limit_plane_isse(srcp, min_luma | (min_luma << 8), max_luma | (max_luma << 8), pitch, row_size, height);
      limit_plane_isse(frame->GetWritePtr(PLANAR_U), min_chroma | (min_chroma << 8), max_chroma | (max_chroma << 8),
        frame->GetPitch(PLANAR_U), frame->GetRowSize(PLANAR_U), frame->GetHeight(PLANAR_U));
      limit_plane_isse(frame->GetWritePtr(PLANAR_V), min_chroma | (min_chroma << 8), max_chroma | (max_chroma << 8),
        frame->GetPitch(PLANAR_V), frame->GetRowSize(PLANAR_V), frame->GetHeight(PLANAR_V));

      return frame;
    }
#endif

    if (pixelsize == 2 )
    {
      if (env->GetCPUFlags() & CPUF_SSE4_1) {
        limit_plane_uint16_sse4(srcp, min_luma, max_luma, pitch, height);

        limit_plane_uint16_sse4(frame->GetWritePtr(PLANAR_U), min_chroma, max_chroma,
          frame->GetPitch(PLANAR_U), frame->GetHeight(PLANAR_U));

        limit_plane_uint16_sse4(frame->GetWritePtr(PLANAR_V), min_chroma, max_chroma,
          frame->GetPitch(PLANAR_V), frame->GetHeight(PLANAR_V));

        return frame;
      }
      if (env->GetCPUFlags() & CPUF_SSE2) {
        limit_plane_uint16_sse2(srcp, min_luma, max_luma, pitch, height);

        limit_plane_uint16_sse2(frame->GetWritePtr(PLANAR_U), min_chroma, max_chroma,
          frame->GetPitch(PLANAR_U), frame->GetHeight(PLANAR_U));

        limit_plane_uint16_sse2(frame->GetWritePtr(PLANAR_V), min_chroma, max_chroma,
          frame->GetPitch(PLANAR_V), frame->GetHeight(PLANAR_V));

        return frame;
      }
    }
#endif
    // C
    // luma
    if(pixelsize == 1)
      limit_plane_c<uint8_t>(srcp, pitch, min_luma, max_luma, width, height);
    else if(pixelsize == 2)
      limit_plane_c<uint16_t>(srcp, pitch, min_luma, max_luma, width, height);
    else // pixelsize == 4: 32 bit float
      limit_plane_f_c(srcp, pitch, min_luma_f, max_luma_f, width, height);

    // chroma if exists
    srcp = frame->GetWritePtr(PLANAR_U);
    width = frame->GetRowSize(PLANAR_U) / pixelsize;
    height = frame->GetHeight(PLANAR_U);
    pitch = frame->GetPitch(PLANAR_U);
    if (!pitch)
      return frame;
    BYTE* srcpV = frame->GetWritePtr(PLANAR_V);
    if(pixelsize == 1) {
      limit_plane_c<uint8_t>(srcp, pitch, min_chroma, max_chroma, width, height);
      limit_plane_c<uint8_t>(srcpV, pitch, min_chroma, max_chroma, width, height);
    } else if(pixelsize == 2) {
      limit_plane_c<uint16_t>(srcp, pitch, min_chroma, max_chroma, width, height);
      limit_plane_c<uint16_t>(srcpV, pitch, min_chroma, max_chroma, width, height);
    }
    else {
      // pixelsize == 4: 32 bit float
      limit_plane_f_c(srcp, pitch, min_chroma_f, max_chroma_f, width, height);
      limit_plane_f_c(srcpV, pitch, min_chroma_f, max_chroma_f, width, height);
    }
  }
  return frame;
}

AVSValue __cdecl Limiter::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  const char* option = args[5].AsString(nullptr);
  show_e show = show_none;

  if (option) {
    if (lstrcmpi(option, "luma") == 0)
      show = show_luma;
    else if (lstrcmpi(option, "luma_grey") == 0)
      show = show_luma_grey;
    else if (lstrcmpi(option, "chroma") == 0)
      show = show_chroma;
    else if (lstrcmpi(option, "chroma_grey") == 0)
      show = show_chroma_grey;
    else
      env->ThrowError("Limiter: show must be \"luma\", \"luma_grey\", \"chroma\" or \"chroma_grey\"");
  }

  const bool paramscale = args[6].AsBool(false);
  return new Limiter(args[0].AsClip(), args[1].AsFloatf(-9999.0f), args[2].AsFloatf(-9999.0f), args[3].AsFloatf(-9999.0f), args[4].AsFloatf(-9999.0f), show, paramscale, env);
}
