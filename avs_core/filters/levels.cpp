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


#include "levels.h"
#include <float.h>
#include "limiter.h"
#include <cstdio>
#include <cmath>
#include <avs/minmax.h>
#include <avs/alignment.h>
#include "../core/internal.h"
#include <algorithm>
#include <string>

#define PI 3.141592653589793


/********************************************************************
***** Declare index of new filters for Avisynth's filter engine *****
********************************************************************/

extern const AVSFunction Levels_filters[] = {
  { "Levels",    BUILTIN_FUNC_PREFIX, "cfffff[coring]b[dither]b", Levels::Create },        // src_low, gamma, src_high, dst_low, dst_high cifiii->ffffff
  { "RGBAdjust", BUILTIN_FUNC_PREFIX, "c[r]f[g]f[b]f[a]f[rb]f[gb]f[bb]f[ab]f[rg]f[gg]f[bg]f[ag]f[analyze]b[dither]b[conditional]b[condvarsuffix]s", RGBAdjust::Create },
  // 'sse': unused even on INTEL, just for script compatibility
  { "Tweak",     BUILTIN_FUNC_PREFIX, "c[hue]f[sat]f[bright]f[cont]f[coring]b[sse]b[startHue]f[endHue]f[maxSat]f[minSat]f[interp]f[dither]b[realcalc]b[dither_strength]f", Tweak::Create },
  { "MaskHS",    BUILTIN_FUNC_PREFIX, "c[startHue]f[endHue]f[maxSat]f[minSat]f[coring]b[realcalc]b", MaskHS::Create },
  { "Limiter",   BUILTIN_FUNC_PREFIX, "c[min_luma]f[max_luma]f[min_chroma]f[max_chroma]f[show]s[paramscale]b", Limiter::Create },
  { 0 }
};


avs_alignas(64) static const BYTE ditherMap[256] = {
#if 0
  // default 0231 recursed table
  0x00, 0x80, 0x20, 0xA0,  0x08, 0x88, 0x28, 0xA8,  0x02, 0x82, 0x22, 0xA2,  0x0A, 0x8A, 0x2A, 0xAA,
  0xC0, 0x40, 0xE0, 0x60,  0xC8, 0x48, 0xE8, 0x68,  0xC2, 0x42, 0xE2, 0x62,  0xCA, 0x4A, 0xEA, 0x6A,
  0x30, 0xB0, 0x10, 0x90,  0x38, 0xB8, 0x18, 0x98,  0x32, 0xB2, 0x12, 0x92,  0x3A, 0xBA, 0x1A, 0x9A,
  0xF0, 0x70, 0xD0, 0x50,  0xF8, 0x78, 0xD8, 0x58,  0xF2, 0x72, 0xD2, 0x52,  0xFA, 0x7A, 0xDA, 0x5A,

  0x0C, 0x8C, 0x2C, 0xAC,  0x04, 0x84, 0x24, 0xA4,  0x0E, 0x8E, 0x2E, 0xAE,  0x06, 0x86, 0x26, 0xA6,
  0xCC, 0x4C, 0xEC, 0x6C,  0xC4, 0x44, 0xE4, 0x64,  0xCE, 0x4E, 0xEE, 0x6E,  0xC6, 0x46, 0xE6, 0x66,
  0x3C, 0xBC, 0x1C, 0x9C,  0x34, 0xB4, 0x14, 0x94,  0x3E, 0xBE, 0x1E, 0x9E,  0x36, 0xB6, 0x16, 0x96,
  0xFC, 0x7C, 0xDC, 0x5C,  0xF4, 0x74, 0xD4, 0x54,  0xFE, 0x7E, 0xDE, 0x5E,  0xF6, 0x76, 0xD6, 0x56,

  0x03, 0x83, 0x23, 0xA3,  0x0B, 0x8B, 0x2B, 0xAB,  0x01, 0x81, 0x21, 0xA1,  0x09, 0x89, 0x29, 0xA9,
  0xC3, 0x43, 0xE3, 0x63,  0xCB, 0x4B, 0xEB, 0x6B,  0xC1, 0x41, 0xE1, 0x61,  0xC9, 0x49, 0xE9, 0x69,
  0x33, 0xB3, 0x13, 0x93,  0x3B, 0xBB, 0x1B, 0x9B,  0x31, 0xB1, 0x11, 0x91,  0x39, 0xB9, 0x19, 0x99,
  0xF3, 0x73, 0xD3, 0x53,  0xFB, 0x7B, 0xDB, 0x5B,  0xF1, 0x71, 0xD1, 0x51,  0xF9, 0x79, 0xD9, 0x59,

  0x0F, 0x8F, 0x2F, 0xAF,  0x07, 0x87, 0x27, 0xA7,  0x0D, 0x8D, 0x2D, 0xAD,  0x05, 0x85, 0x25, 0xA5,
  0xCF, 0x4F, 0xEF, 0x6F,  0xC7, 0x47, 0xE7, 0x67,  0xCD, 0x4D, 0xED, 0x6D,  0xC5, 0x45, 0xE5, 0x65,
  0x3F, 0xBF, 0x1F, 0x9F,  0x37, 0xB7, 0x17, 0x97,  0x3D, 0xBD, 0x1D, 0x9D,  0x35, 0xB5, 0x15, 0x95,
  0xFF, 0x7F, 0xDF, 0x5F,  0xF7, 0x77, 0xD7, 0x57,  0xFD, 0x7D, 0xDD, 0x5D,  0xF5, 0x75, 0xD5, 0x55,
#else
  // improved "equal sum" modified table
  0x00, 0xB0, 0x60, 0xD0,  0x0B, 0xBB, 0x6B, 0xDB,  0x06, 0xB6, 0x66, 0xD6,  0x0D, 0xBD, 0x6D, 0xDD,
  0xC0, 0x70, 0x90, 0x20,  0xCB, 0x7B, 0x9B, 0x2B,  0xC6, 0x76, 0x96, 0x26,  0xCD, 0x7D, 0x9D, 0x2D,
  0x30, 0x80, 0x50, 0xE0,  0x3B, 0x8B, 0x5B, 0xEB,  0x36, 0x86, 0x56, 0xE6,  0x3D, 0x8D, 0x5D, 0xED,
  0xF0, 0x40, 0xA0, 0x10,  0xFB, 0x4B, 0xAB, 0x1B,  0xF6, 0x46, 0xA6, 0x16,  0xFD, 0x4D, 0xAD, 0x1D,

  0x0C, 0xBC, 0x6C, 0xDC,  0x07, 0xB7, 0x67, 0xD7,  0x09, 0xB9, 0x69, 0xD9,  0x02, 0xB2, 0x62, 0xD2,
  0xCC, 0x7C, 0x9C, 0x2C,  0xC7, 0x77, 0x97, 0x27,  0xC9, 0x79, 0x99, 0x29,  0xC2, 0x72, 0x92, 0x22,
  0x3C, 0x8C, 0x5C, 0xEC,  0x37, 0x87, 0x57, 0xE7,  0x39, 0x89, 0x59, 0xE9,  0x32, 0x82, 0x52, 0xE2,
  0xFC, 0x4C, 0xAC, 0x1C,  0xF7, 0x47, 0xA7, 0x17,  0xF9, 0x49, 0xA9, 0x19,  0xF2, 0x42, 0xA2, 0x12,

  0x03, 0xB3, 0x63, 0xD3,  0x08, 0xB8, 0x68, 0xD8,  0x05, 0xB5, 0x65, 0xD5,  0x0E, 0xBE, 0x6E, 0xDE,
  0xC3, 0x73, 0x93, 0x23,  0xC8, 0x78, 0x98, 0x28,  0xC5, 0x75, 0x95, 0x25,  0xCE, 0x7E, 0x9E, 0x2E,
  0x33, 0x83, 0x53, 0xE3,  0x38, 0x88, 0x58, 0xE8,  0x35, 0x85, 0x55, 0xE5,  0x3E, 0x8E, 0x5E, 0xEE,
  0xF3, 0x43, 0xA3, 0x13,  0xF8, 0x48, 0xA8, 0x18,  0xF5, 0x45, 0xA5, 0x15,  0xFE, 0x4E, 0xAE, 0x1E,

  0x0F, 0xBF, 0x6F, 0xDF,  0x04, 0xB4, 0x64, 0xD4,  0x0A, 0xBA, 0x6A, 0xDA,  0x01, 0xB1, 0x61, 0xD1,
  0xCF, 0x7F, 0x9F, 0x2F,  0xC4, 0x74, 0x94, 0x24,  0xCA, 0x7A, 0x9A, 0x2A,  0xC1, 0x71, 0x91, 0x21,
  0x3F, 0x8F, 0x5F, 0xEF,  0x34, 0x84, 0x54, 0xE4,  0x3A, 0x8A, 0x5A, 0xEA,  0x31, 0x81, 0x51, 0xE1,
  0xFF, 0x4F, 0xAF, 0x1F,  0xF4, 0x44, 0xA4, 0x14,  0xFA, 0x4A, 0xAA, 0x1A,  0xF1, 0x41, 0xA1, 0x11,
#endif
};


avs_alignas(16) static const BYTE ditherMap4[16] = {
  0x0, 0xB, 0x6, 0xD,
  0xC, 0x7, 0x9, 0x2,
  0x3, 0x8, 0x5, 0xE,
  0xF, 0x4, 0xA, 0x1,
};

static void __cdecl free_buffer(void* buff, IScriptEnvironment* env)
{
    if (buff) {
        static_cast<IScriptEnvironment2*>(env)->Free(buff);
    }
}

// Helper for Limits, MaskHS, Tweak
static void get_limits(luma_chroma_limits_t &d, int bits_per_pixel) {
  int tv_range_lo_luma_8 = 16;
  int tv_range_hi_luma_8 = 235;
  int tv_range_lo_chroma_8 = tv_range_lo_luma_8;
  int tv_range_hi_chroma_8 = 240;

  if (bits_per_pixel == 32) {
    d.tv_range_low_luma_f = tv_range_lo_luma_8 / 255.0f;
    d.tv_range_hi_luma_f = tv_range_hi_luma_8 / 255.0f;
    d.full_range_low_luma_f = 0.0f;
    d.full_range_hi_luma_f = 1.0f;
#ifdef FLOAT_CHROMA_IS_HALF_CENTERED
    d.middle_chroma_f = 0.5f;
#else
    d.middle_chroma_f = 0.0f;
#endif
    d.tv_range_low_chroma_f = (tv_range_lo_chroma_8 - 128) / 255.0f + d.middle_chroma_f; // -112
    d.tv_range_hi_chroma_f = (tv_range_hi_chroma_8 - 128) / 255.0f + d.middle_chroma_f; // 112
    d.full_range_low_chroma_f = d.middle_chroma_f - 0.5f; // -0.5..0.5 or 0..1.0
    d.full_range_hi_chroma_f = d.middle_chroma_f + 0.5f;

    d.range_luma_f = d.tv_range_hi_luma_f - d.tv_range_low_luma_f;
    d.range_chroma_f = d.tv_range_hi_chroma_f - d.tv_range_low_chroma_f;
  }
  else {
    d.tv_range_low = tv_range_lo_luma_8 << (bits_per_pixel - 8); // 16-240,64-960, 256-3852,... 4096-61692
    d.tv_range_hi_luma = tv_range_hi_luma_8 << (bits_per_pixel - 8);
    d.tv_range_hi_chroma = tv_range_hi_chroma_8 << (bits_per_pixel - 8);
    d.middle_chroma = 1 << (bits_per_pixel - 1); // 128
    d.range_luma = d.tv_range_hi_luma - d.tv_range_low; // 219
    d.range_chroma = d.tv_range_hi_chroma - d.tv_range_low; // 224
  }
}

/********************************
 *******   Levels Filter   ******
 ********************************/

template<bool chroma, bool use_gamma>
AVS_FORCEINLINE float Levels::calcPixel(const float pixel)
{
    float result;
    if (!chroma) {
      float p;
      if (coring)
        p = ((pixel - limits.tv_range_low_luma_f)*(1.0f / limits.range_luma_f) - in_min_f) / divisor_f;
      else
        p = (pixel - in_min_f) / divisor_f;

      if(use_gamma)
        p = (float)pow((double)clamp(p, 0.0f, 1.0f), gamma);

      p = p * out_diff_f + out_min_f; // out_diff_f = out_max_f - out_min_f;
      // luma
      if (coring) {
        result = clamp(p* limits.range_luma_f / 1.0f + limits.tv_range_low_luma_f, limits.tv_range_low_luma_f, limits.tv_range_hi_luma_f);
      }
      else
        result = clamp(p, 0.0f, 1.0f); // todo: theoretical question, should we clamp in Levels function?
    }
    else {
      /*
      int q = (int)(((bias_dither + ii - middle_chroma * scale) * (out_max - out_min)) / divisor + middle_chroma + 0.5);
      int chroma;
      if (coring)
        chroma = clamp(q, tv_range_low, tv_range_hi_chroma); // e.g. clamp(q, 16, 240)
      else
        chroma = clamp(q, 0, max_pixel_value); // e.g. clamp(q, 0, 255)
      */
      float q = ((pixel - limits.middle_chroma_f) * out_diff_f) / divisor_f + limits.middle_chroma_f;
      if (coring)
        result = clamp(q, limits.tv_range_low_chroma_f, limits.tv_range_hi_chroma_f); // e.g. clamp(q, 16, 240)
      else
        result = clamp(q, limits.full_range_low_chroma_f, limits.full_range_hi_chroma_f); // e.g. clamp(q, 0, 255) todo: theoretical question, should we clamp in Levels function?
    }
    return result;
}


Levels::Levels(PClip _child, float _in_min, double _gamma, float _in_max, float _out_min, float _out_max, bool _coring, bool _dither,
  IScriptEnvironment* env)
  : GenericVideoFilter(_child), coring(_coring), dither(_dither), gamma(_gamma), in_min_f(_in_min), in_max_f(_in_max), out_min_f(_out_min), out_max_f(_out_max)
{
  if (gamma <= 0.0)
    env->ThrowError("Levels: gamma must be positive");

  gamma = 1/gamma;
  use_gamma = (gamma != 1.0);

  int in_min = (int)in_min_f;
  int in_max = (int)in_max_f;
  int out_min = (int)out_min_f;
  int out_max = (int)out_max_f;
  out_diff_f = out_max_f - out_min_f;

  int divisor;
  if (in_min == in_max)
    divisor = 1;
  else
    divisor = in_max - in_min;

  if (in_min_f == in_max_f)
    divisor_f = 1;
  else
    divisor_f = in_max_f - in_min_f;

  int scale = 1;
  //double bias = 0.0;

  dither_strength = 1.0f; // later: from parameter as Tweak

  pixelsize = vi.ComponentSize();
  bits_per_pixel = vi.BitsPerComponent(); // 8,10..16

  // No lookup for float. Only slow pixel-by-pixel realtime calculation

  int lookup_size = 1 << bits_per_pixel; // 256, 1024, 4096, 16384, 65536
  real_lookup_size = (pixelsize == 1) ? 256 : 65536; // avoids lut overflow in case of non-standard content of a 10 bit clip
  int max_pixel_value = (1 << bits_per_pixel) - 1;

  use_lut = bits_per_pixel != 32; // for float: realtime only

  get_limits(limits, bits_per_pixel); // tv range limits

  if (pixelsize == 4)
    dither_strength /= 65536.0f; // same dither range as for a 16 bit clip

  if (dither) {
    // lut scale settings
    // same 256*dither for chroma and luma
    scale = 256; // lower 256 is dither value
    divisor *= 256;
    in_min *= 256;
    bias_dither = -(256.0f * dither_strength - 1) / 2; // -127.5 for 8 bit, scaling because of dithershift
  }
  else {
    scale = 1;
    bias_dither = 0.0f;
  }

  need_chroma = (vi.IsYUV() || vi.IsYUVA()) && !vi.IsY8();
  if (vi.IsRGB())
    coring = false; // no coring option for packed and planar RGBs

  // one buffer for map and mapchroma
  map = nullptr;
  if (use_lut) {
    int number_of_maps = need_chroma ? 2 : 1;
    int bufsize = pixelsize * real_lookup_size * scale * number_of_maps;
    map = static_cast<uint8_t*>(env->Allocate(bufsize, 16, AVS_NORMAL_ALLOC));
    if (!map)
      env->ThrowError("Levels: Could not reserve memory.");
    env->AtExit(free_buffer, map);
    if (bits_per_pixel > 8 && bits_per_pixel < 16) // make lut table safe for 10-14 bit garbage
      std::fill_n(map, bufsize, 0); // 8 and 16 bit is safe

    if(need_chroma)
      mapchroma = map + pixelsize * real_lookup_size * scale; // pointer offset

    for (int i = 0; i < lookup_size*scale; ++i) {
      double p;

      int ii;
      if (dither)
        ii = (i & 0xFFFFFF00) + (int)((i & 0xFF)*dither_strength);
      else
        ii = i;

      if (coring)
        p = ((bias_dither + ii - limits.tv_range_low *scale)*((double)max_pixel_value / limits.range_luma) - in_min) / divisor;
      else
        p = (bias_dither + ii - in_min) / divisor;

      p = pow(clamp(p, 0.0, 1.0), gamma);
      p = p * (out_max - out_min) + out_min;
      int luma;
      if (coring)
        luma = clamp(int(p*((double)limits.range_luma / max_pixel_value) + limits.tv_range_low + 0.5), limits.tv_range_low, limits.tv_range_hi_luma);
      else
        luma = clamp(int(p + 0.5), 0, max_pixel_value);

      if (pixelsize == 1)
        map[i] = (BYTE)luma;
      else // pixelsize==2
        reinterpret_cast<uint16_t *>(map)[i] = (uint16_t)luma;

      if (need_chroma) {
        int q = (int)(((bias_dither + ii - limits.middle_chroma*scale) * (out_max - out_min)) / divisor + limits.middle_chroma + 0.5f);
        int chroma;
        if (coring)
          chroma = clamp(q, limits.tv_range_low, limits.tv_range_hi_chroma); // e.g. clamp(q, 16, 240)
        else
          chroma = clamp(q, 0, max_pixel_value); // e.g. clamp(q, 0, 255)
        if (pixelsize == 1)
          mapchroma[i] = (BYTE)chroma;
        else // pixelsize==2
          reinterpret_cast<uint16_t *>(mapchroma)[i] = (uint16_t)chroma;
      }
    }
  }
  else {
    // precalc float dither table from integer one
    if (dither) {
      for (int y = 0; y <= 15; y++)
        for (int x = 0; x <= 15; x++) {
          int index = (y << 4) | x; // 0..255
          ditherMap_f[index] = (ditherMap[index] / 255.0f - 0.5f) * dither_strength;
          // float dithering is 16 bit granularity, dither_strength is 1/65536.0 and not a parameter yet
        }
    }
  }
}


PVideoFrame __stdcall Levels::GetFrame(int n, IScriptEnvironment* env)
{
  PVideoFrame frame = child->GetFrame(n, env);
  env->MakeWritable(&frame);
  BYTE* p = frame->GetWritePtr();
  const int pitch = frame->GetPitch();

  if (use_lut) {
    if (dither) {
      if (vi.IsYUY2()) {
        const int UVwidth = vi.width / 2;
        for (int y = 0; y < vi.height; ++y) {
          const int _y = (y << 4) & 0xf0;
          for (int x = 0; x < vi.width; ++x) {
            p[x * 2] = map[p[x * 2] << 8 | ditherMap[(x & 0x0f) | _y]];
          }
          for (int z = 0; z < UVwidth; ++z) {
            const int _dither = ditherMap[(z & 0x0f) | _y];
            p[z * 4 + 1] = mapchroma[p[z * 4 + 1] << 8 | _dither];
            p[z * 4 + 3] = mapchroma[p[z * 4 + 3] << 8 | _dither];
          }
          p += pitch;
        }
      }
      else if (vi.IsPlanar()) {
        if (vi.IsYUV() || vi.IsYUVA()) {
          // planar YUV
          if (pixelsize == 1) {
            for (int y = 0; y < vi.height; ++y) {
              const int _y = (y << 4) & 0xf0;
              for (int x = 0; x < vi.width; ++x) {
                p[x] = map[p[x] << 8 | ditherMap[(x & 0x0f) | _y]];
              }
              p += pitch;
            }
          }
          else { // pixelsize==2
            for (int y = 0; y < vi.height; ++y) {
              const int _y = (y << 4) & 0xf0;
              for (int x = 0; x < vi.width; ++x) {
                reinterpret_cast<uint16_t *>(p)[x] = reinterpret_cast<uint16_t *>(map)[reinterpret_cast<uint16_t *>(p)[x] << 8 | ditherMap[(x & 0x0f) | _y]];
              }
              p += pitch;
            }
          }
          const int UVpitch = frame->GetPitch(PLANAR_U);
          const int w = frame->GetRowSize(PLANAR_U) / pixelsize;
          const int h = frame->GetHeight(PLANAR_U);
          p = frame->GetWritePtr(PLANAR_U);
          BYTE* q = frame->GetWritePtr(PLANAR_V);
          if (pixelsize == 1) {
            for (int y = 0; y < h; ++y) {
              const int _y = (y << 4) & 0xf0;
              for (int x = 0; x < w; ++x) {
                const int _dither = ditherMap[(x & 0x0f) | _y];
                p[x] = mapchroma[p[x] << 8 | _dither];
                q[x] = mapchroma[q[x] << 8 | _dither];
              }
              p += UVpitch;
              q += UVpitch;
            }
          }
          else { // pixelsize==2
            for (int y = 0; y < h; ++y) {
              const int _y = (y << 4) & 0xf0;
              for (int x = 0; x < w; ++x) {
                const int _dither = ditherMap[(x & 0x0f) | _y];
                reinterpret_cast<uint16_t *>(p)[x] = reinterpret_cast<uint16_t *>(mapchroma)[reinterpret_cast<uint16_t *>(p)[x] << 8 | _dither];
                reinterpret_cast<uint16_t *>(q)[x] = reinterpret_cast<uint16_t *>(mapchroma)[reinterpret_cast<uint16_t *>(q)[x] << 8 | _dither];
              }
              p += UVpitch;
              q += UVpitch;
            }
          }
        }
        else {
          // planar RGB
          BYTE* b = frame->GetWritePtr(PLANAR_B);
          BYTE* r = frame->GetWritePtr(PLANAR_R);
          const int pitch_b = frame->GetPitch(PLANAR_B);
          const int pitch_r = frame->GetPitch(PLANAR_R);
          if (pixelsize == 1) {
            for (int y = 0; y < vi.height; ++y) {
              const int _y = (y << 4) & 0xf0;
              for (int x = 0; x < vi.width; ++x) {
                p[x] = map[p[x] << 8 | ditherMap[(x & 0x0f) | _y]];
                b[x] = map[b[x] << 8 | ditherMap[(x & 0x0f) | _y]];
                r[x] = map[r[x] << 8 | ditherMap[(x & 0x0f) | _y]];
              }
              p += pitch;
              b += pitch_b;
              r += pitch_r;
            }
          }
          else { // pixelsize==2
            for (int y = 0; y < vi.height; ++y) {
              const int _y = (y << 4) & 0xf0;
              for (int x = 0; x < vi.width; ++x) {
                reinterpret_cast<uint16_t *>(p)[x] = reinterpret_cast<uint16_t *>(map)[reinterpret_cast<uint16_t *>(p)[x] << 8 | ditherMap[(x & 0x0f) | _y]];
                reinterpret_cast<uint16_t *>(b)[x] = reinterpret_cast<uint16_t *>(map)[reinterpret_cast<uint16_t *>(b)[x] << 8 | ditherMap[(x & 0x0f) | _y]];
                reinterpret_cast<uint16_t *>(r)[x] = reinterpret_cast<uint16_t *>(map)[reinterpret_cast<uint16_t *>(r)[x] << 8 | ditherMap[(x & 0x0f) | _y]];
              }
              p += pitch;
              b += pitch_b;
              r += pitch_r;
            }
          }
        }
      }
      else if (vi.IsRGB32()) {
        for (int y = 0; y < vi.height; ++y) {
          const int _y = (y << 4) & 0xf0;
          for (int x = 0; x < vi.width; ++x) {
            const int _dither = ditherMap[(x & 0x0f) | _y];
            p[x * 4 + 0] = map[p[x * 4 + 0] << 8 | _dither];
            p[x * 4 + 1] = map[p[x * 4 + 1] << 8 | _dither];
            p[x * 4 + 2] = map[p[x * 4 + 2] << 8 | _dither];
            p[x * 4 + 3] = map[p[x * 4 + 3] << 8 | _dither];
          }
          p += pitch;
        }
      }
      else if (vi.IsRGB24()) {
        for (int y = 0; y < vi.height; ++y) {
          const int _y = (y << 4) & 0xf0;
          for (int x = 0; x < vi.width; ++x) {
            const int _dither = ditherMap[(x & 0x0f) | _y];
            p[x * 3 + 0] = map[p[x * 3 + 0] << 8 | _dither];
            p[x * 3 + 1] = map[p[x * 3 + 1] << 8 | _dither];
            p[x * 3 + 2] = map[p[x * 3 + 2] << 8 | _dither];
          }
          p += pitch;
        }
      }
      else if (vi.IsRGB64()) {
        for (int y = 0; y < vi.height; ++y) {
          const int _y = (y << 4) & 0xf0;
          for (int x = 0; x < vi.width; ++x) {
            const int _dither = ditherMap[(x & 0x0f) | _y];
            reinterpret_cast<uint16_t *>(p)[x * 4 + 0] = reinterpret_cast<uint16_t *>(map)[reinterpret_cast<uint16_t *>(p)[x * 4 + 0] << 8 | _dither];
            reinterpret_cast<uint16_t *>(p)[x * 4 + 1] = reinterpret_cast<uint16_t *>(map)[reinterpret_cast<uint16_t *>(p)[x * 4 + 1] << 8 | _dither];
            reinterpret_cast<uint16_t *>(p)[x * 4 + 2] = reinterpret_cast<uint16_t *>(map)[reinterpret_cast<uint16_t *>(p)[x * 4 + 2] << 8 | _dither];
            reinterpret_cast<uint16_t *>(p)[x * 4 + 3] = reinterpret_cast<uint16_t *>(map)[reinterpret_cast<uint16_t *>(p)[x * 4 + 3] << 8 | _dither];
          }
          p += pitch;
        }
      }
      else if (vi.IsRGB48()) {
        for (int y = 0; y < vi.height; ++y) {
          const int _y = (y << 4) & 0xf0;
          for (int x = 0; x < vi.width; ++x) {
            const int _dither = ditherMap[(x & 0x0f) | _y];
            reinterpret_cast<uint16_t *>(p)[x * 3 + 0] = reinterpret_cast<uint16_t *>(map)[reinterpret_cast<uint16_t *>(p)[x * 3 + 0] << 8 | _dither];
            reinterpret_cast<uint16_t *>(p)[x * 3 + 1] = reinterpret_cast<uint16_t *>(map)[reinterpret_cast<uint16_t *>(p)[x * 3 + 1] << 8 | _dither];
            reinterpret_cast<uint16_t *>(p)[x * 3 + 2] = reinterpret_cast<uint16_t *>(map)[reinterpret_cast<uint16_t *>(p)[x * 3 + 2] << 8 | _dither];
          }
          p += pitch;
        }
      }
    }
    else { // no dithering
      if (vi.IsYUY2()) {
        for (int y = 0; y < vi.height; ++y) {
          for (int x = 0; x < vi.width; ++x) {
            p[x * 2 + 0] = map[p[x * 2 + 0]];
            p[x * 2 + 1] = mapchroma[p[x * 2 + 1]];
          }
          p += pitch;
        }
      }
      else if (vi.IsPlanar()) {
        if (vi.IsYUV() || vi.IsYUVA()) {
          // planar YUV
          if (pixelsize == 1) {
            for (int y = 0; y < vi.height; ++y) {
              for (int x = 0; x < vi.width; ++x) {
                p[x] = map[p[x]];
              }
              p += pitch;
            }
          }
          else { // pixelsize==2
            for (int y = 0; y < vi.height; ++y) {
              for (int x = 0; x < vi.width; ++x) {
                reinterpret_cast<uint16_t *>(p)[x] = reinterpret_cast<uint16_t *>(map)[reinterpret_cast<uint16_t *>(p)[x]];
              }
              p += pitch;
            }
          }
          const int UVpitch = frame->GetPitch(PLANAR_U);
          p = frame->GetWritePtr(PLANAR_U);
          const int w = frame->GetRowSize(PLANAR_U) / pixelsize;
          const int h = frame->GetHeight(PLANAR_U);
          if (pixelsize == 1) {
            for (int y = 0; y < h; ++y) {
              for (int x = 0; x < w; ++x) {
                p[x] = mapchroma[p[x]];
              }
              p += UVpitch;
            }
            p = frame->GetWritePtr(PLANAR_V);
            for (int y = 0; y < h; ++y) {
              for (int x = 0; x < w; ++x) {
                p[x] = mapchroma[p[x]];
              }
              p += UVpitch;
            }
          }
          else { // pixelsize==2
            for (int y = 0; y < h; ++y) {
              for (int x = 0; x < w; ++x) {
                reinterpret_cast<uint16_t *>(p)[x] = reinterpret_cast<uint16_t *>(mapchroma)[reinterpret_cast<uint16_t *>(p)[x]];
              }
              p += UVpitch;
            }
            p = frame->GetWritePtr(PLANAR_V);
            for (int y = 0; y < h; ++y) {
              for (int x = 0; x < w; ++x) {
                reinterpret_cast<uint16_t *>(p)[x] = reinterpret_cast<uint16_t *>(mapchroma)[reinterpret_cast<uint16_t *>(p)[x]];
              }
              p += UVpitch;
            }
          }
        }
        else {
          // Planar RGB
          BYTE* b = frame->GetWritePtr(PLANAR_B);
          BYTE* r = frame->GetWritePtr(PLANAR_R);
          const int pitch_b = frame->GetPitch(PLANAR_B);
          const int pitch_r = frame->GetPitch(PLANAR_R);
          if (pixelsize == 1) {
            for (int y = 0; y < vi.height; ++y) {
              for (int x = 0; x < vi.width; ++x) {
                p[x] = map[p[x]];
                b[x] = map[b[x]];
                r[x] = map[r[x]];
              }
              p += pitch;
              b += pitch_b;
              r += pitch_r;
            }
          }
          else { // pixelsize==2
            for (int y = 0; y < vi.height; ++y) {
              for (int x = 0; x < vi.width; ++x) {
                reinterpret_cast<uint16_t *>(p)[x] = reinterpret_cast<uint16_t *>(map)[reinterpret_cast<uint16_t *>(p)[x]];
                reinterpret_cast<uint16_t *>(b)[x] = reinterpret_cast<uint16_t *>(map)[reinterpret_cast<uint16_t *>(b)[x]];
                reinterpret_cast<uint16_t *>(r)[x] = reinterpret_cast<uint16_t *>(map)[reinterpret_cast<uint16_t *>(r)[x]];
              }
              p += pitch;
              b += pitch_b;
              r += pitch_r;
            }
          }
        }
      }
      else if (vi.IsRGB()) {
     // packed RGB
        const int work_width = frame->GetRowSize() / pixelsize;
        if (pixelsize == 1) {
          for (int y = 0; y < vi.height; ++y) {
            for (int x = 0; x < work_width; ++x) {
              p[x] = map[p[x]];
            }
            p += pitch;
          }
        }
        else { // pixelsize==2
          for (int y = 0; y < vi.height; ++y) {
            for (int x = 0; x < work_width; ++x) {
              reinterpret_cast<uint16_t *>(p)[x] = reinterpret_cast<uint16_t *>(map)[reinterpret_cast<uint16_t *>(p)[x]];
            }
            p += pitch;
          }
        }
      }
    }
  }
  else {
    // float 32 bit. only planars here
    // no lut, realtime calculation only
    if (dither) {
      if (vi.IsYUV() || vi.IsYUVA()) { // planar YUV (incl Y only)
        // luma. with or w/o gamma
        if (use_gamma) {
          for (int y = 0; y < vi.height; ++y) {
            const int _y = (y << 4) & 0xf0;
            for (int x = 0; x < vi.width; ++x) {
              float _dither = ditherMap_f[(x & 0x0f) | _y];
              const float pixel = reinterpret_cast<float *>(p)[x] + _dither;
              reinterpret_cast<float *>(p)[x] = calcPixel<false, true>(pixel);
            }
            p += pitch;
          }
        }
        else {
          // don't use gamma (faster)
          for (int y = 0; y < vi.height; ++y) {
            const int _y = (y << 4) & 0xf0;
            for (int x = 0; x < vi.width; ++x) {
              float _dither = ditherMap_f[(x & 0x0f) | _y];
              const float pixel = reinterpret_cast<float *>(p)[x] + _dither;
              reinterpret_cast<float *>(p)[x] = calcPixel<false, false>(pixel); // w/o gamma
            }
            p += pitch;
          }
        }
        // chroma
        if (need_chroma) {
          const int UVpitch = frame->GetPitch(PLANAR_U);
          const int w = frame->GetRowSize(PLANAR_U) / pixelsize;
          const int h = frame->GetHeight(PLANAR_U);
          p = frame->GetWritePtr(PLANAR_U);
          BYTE* q = frame->GetWritePtr(PLANAR_V);
          for (int y = 0; y < h; ++y) {
            const int _y = (y << 4) & 0xf0;
            for (int x = 0; x < w; ++x) {
              float _dither = ditherMap_f[(x & 0x0f) | _y];
              const float pixel_u = reinterpret_cast<float *>(p)[x] + _dither;
              reinterpret_cast<float *>(p)[x] = calcPixel<true, false>(pixel_u);
              const float pixel_v = reinterpret_cast<float *>(q)[x] + _dither;
              reinterpret_cast<float *>(q)[x] = calcPixel<true, false>(pixel_v);
            }
            p += UVpitch;
            q += UVpitch;
          }
        }
      }
      else if (vi.IsPlanarRGB() || vi.IsPlanarRGBA()) {
        // planar RGB
        BYTE* b = frame->GetWritePtr(PLANAR_B);
        BYTE* r = frame->GetWritePtr(PLANAR_R);
        const int pitch_b = frame->GetPitch(PLANAR_B);
        const int pitch_r = frame->GetPitch(PLANAR_R);
        if (use_gamma) {
          for (int y = 0; y < vi.height; ++y) {
            const int _y = (y << 4) & 0xf0;
            for (int x = 0; x < vi.width; ++x) {
              float _dither = ditherMap_f[(x & 0x0f) | _y];
              const float pixel_p = reinterpret_cast<float *>(p)[x] + _dither; // g channel
              reinterpret_cast<float *>(p)[x] = calcPixel<false, true>(pixel_p);
              const float pixel_b = reinterpret_cast<float *>(b)[x] + _dither;
              reinterpret_cast<float *>(b)[x] = calcPixel<false, true>(pixel_b);
              const float pixel_r = reinterpret_cast<float *>(r)[x] + _dither;
              reinterpret_cast<float *>(r)[x] = calcPixel<false, true>(pixel_r);
            }
            p += pitch; // g
            b += pitch_b;
            r += pitch_r;
          }
        }
        else {
          for (int y = 0; y < vi.height; ++y) {
            const int _y = (y << 4) & 0xf0;
            for (int x = 0; x < vi.width; ++x) {
              float _dither = ditherMap_f[(x & 0x0f) | _y];
              const float pixel_p = reinterpret_cast<float *>(p)[x] + _dither; // g channel
              reinterpret_cast<float *>(p)[x] = calcPixel<false, false>(pixel_p);
              const float pixel_b = reinterpret_cast<float *>(b)[x] + _dither;
              reinterpret_cast<float *>(b)[x] = calcPixel<false, false>(pixel_b);
              const float pixel_r = reinterpret_cast<float *>(r)[x] + _dither;
              reinterpret_cast<float *>(r)[x] = calcPixel<false, false>(pixel_r);
            }
            p += pitch; // g
            b += pitch_b;
            r += pitch_r;
          }
        }
      }
      else {
        // 32 bit, neither YUV(A), nor RGB(A)
      }
    }
    else {
      // no dither
      if (vi.IsYUV() || vi.IsYUVA()) { // planar YUV (incl Y only)
                                       // luma. with or w/o gamma
        if (use_gamma) {
          for (int y = 0; y < vi.height; ++y) {
            for (int x = 0; x < vi.width; ++x) {
              const float pixel = reinterpret_cast<float *>(p)[x];
              reinterpret_cast<float *>(p)[x] = calcPixel<false, true>(pixel);
            }
            p += pitch;
          }
        }
        else {
          // don't use gamma (faster)
          for (int y = 0; y < vi.height; ++y) {
            for (int x = 0; x < vi.width; ++x) {
              const float pixel = reinterpret_cast<float *>(p)[x];
              reinterpret_cast<float *>(p)[x] = calcPixel<false, false>(pixel); // w/o gamma
            }
            p += pitch;
          }
        }
        // chroma
        if (need_chroma) {
          const int UVpitch = frame->GetPitch(PLANAR_U);
          const int w = frame->GetRowSize(PLANAR_U) / pixelsize;
          const int h = frame->GetHeight(PLANAR_U);
          p = frame->GetWritePtr(PLANAR_U);
          BYTE* q = frame->GetWritePtr(PLANAR_V);
          for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
              const float pixel_u = reinterpret_cast<float *>(p)[x];
              reinterpret_cast<float *>(p)[x] = calcPixel<true, false>(pixel_u);
              const float pixel_v = reinterpret_cast<float *>(q)[x];
              reinterpret_cast<float *>(q)[x] = calcPixel<true, false>(pixel_v);
            }
            p += UVpitch;
            q += UVpitch;
          }
        }
      }
      else if (vi.IsPlanarRGB() || vi.IsPlanarRGBA()) {
        // planar RGB
        BYTE* b = frame->GetWritePtr(PLANAR_B);
        BYTE* r = frame->GetWritePtr(PLANAR_R);
        const int pitch_b = frame->GetPitch(PLANAR_B);
        const int pitch_r = frame->GetPitch(PLANAR_R);
        if (use_gamma) {
          for (int y = 0; y < vi.height; ++y) {
            for (int x = 0; x < vi.width; ++x) {
              const float pixel_p = reinterpret_cast<float *>(p)[x]; // g channel
              reinterpret_cast<float *>(p)[x] = calcPixel<false, true>(pixel_p);
              const float pixel_b = reinterpret_cast<float *>(b)[x];
              reinterpret_cast<float *>(b)[x] = calcPixel<false, true>(pixel_b);
              const float pixel_r = reinterpret_cast<float *>(r)[x];
              reinterpret_cast<float *>(r)[x] = calcPixel<false, true>(pixel_r);
            }
            p += pitch; // g
            b += pitch_b;
            r += pitch_r;
          }
        }
        else {
          for (int y = 0; y < vi.height; ++y) {
            for (int x = 0; x < vi.width; ++x) {
              const float pixel_p = reinterpret_cast<float *>(p)[x]; // g channel
              reinterpret_cast<float *>(p)[x] = calcPixel<false, false>(pixel_p);
              const float pixel_b = reinterpret_cast<float *>(b)[x];
              reinterpret_cast<float *>(b)[x] = calcPixel<false, false>(pixel_b);
              const float pixel_r = reinterpret_cast<float *>(r)[x];
              reinterpret_cast<float *>(r)[x] = calcPixel<false, false>(pixel_r);
            }
            p += pitch; // g
            b += pitch_b;
            r += pitch_r;
          }
        }
      }
      else {
        // 32 bit, neither YUV(A), nor RGB(A)
      }
    }
  }
  return frame;
}

AVSValue __cdecl Levels::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  enum { CHILD, IN_MIN, GAMMA, IN_MAX, OUT_MIN, OUT_MAX, CORING, DITHER };
  return new Levels( args[CHILD].AsClip(), (float)args[IN_MIN].AsFloat(), (float)args[GAMMA].AsFloat(), (float)args[IN_MAX].AsFloat(),
    (float)args[OUT_MIN].AsFloat(), (float)args[OUT_MAX].AsFloat(), args[CORING].AsBool(true), args[DITHER].AsBool(false), env );
}

/********************************
 *******    RGBA Filter    ******
 ********************************/

#define READ_CONDITIONAL(plane_num, var_name, internal_name, condVarSuffix)  \
    {                                                     \
        std::string s = "rgbadjust_" #var_name;\
        s = s + condVarSuffix; \
        const double t = env->GetVarDouble(s.c_str(), DBL_MIN); \
        if (t != DBL_MIN) {                             \
            config->rgba[plane_num].internal_name = t;  \
            config->rgba[plane_num].changed = true;     \
        }                                               \
    }

static void rgbadjust_read_conditional(IScriptEnvironment* env, RGBAdjustConfig* config, const char * condVarSuffix)
{
  READ_CONDITIONAL(0, r, scale, condVarSuffix);
  READ_CONDITIONAL(1, g, scale, condVarSuffix);
  READ_CONDITIONAL(2, b, scale, condVarSuffix);
  READ_CONDITIONAL(3, a, scale, condVarSuffix);

  READ_CONDITIONAL(0, rb, bias, condVarSuffix);
  READ_CONDITIONAL(1, gb, bias, condVarSuffix);
  READ_CONDITIONAL(2, bb, bias, condVarSuffix);
  READ_CONDITIONAL(3, ab, bias, condVarSuffix);

  READ_CONDITIONAL(0, rg, gamma, condVarSuffix);
  READ_CONDITIONAL(1, gg, gamma, condVarSuffix);
  READ_CONDITIONAL(2, bg, gamma, condVarSuffix);
  READ_CONDITIONAL(3, ag, gamma, condVarSuffix);
}

#undef READ_CONDITIONAL



RGBAdjust::~RGBAdjust()
{
  if (map_holder)
    delete[] map_holder;
}

void RGBAdjust::CheckAndConvertParams(RGBAdjustConfig &config, IScriptEnvironment *env)
{
  if ((config.rgba[0].gamma <= 0.0) || (config.rgba[1].gamma <= 0.0) || (config.rgba[2].gamma <= 0.0) || (config.rgba[3].gamma <= 0.0))
    env->ThrowError("RGBAdjust: gammas must be positive");
}

static __inline float RGBAdjust_processPixel(const float val, const double c0, const double c1, const double c2)
{
  const double pixel_max = 1.0;
  return (float)(pow(clamp((c0 + val * c1) / pixel_max, 0.0, 1.0), c2));
}

void RGBAdjust::rgbadjust_create_lut(BYTE *lut_buf, const int plane, RGBAdjustConfig &cfg) {
  if (!use_lut)
    return;

  const int lookup_size = 1 << bits_per_pixel; // 256, 1024, 4096, 16384, 65536

  void(*set_map)(BYTE*, int, int, float, const double, const double, const double);
  if (dither) {
    set_map = [](BYTE* map, int lookup_size, int bits_per_pixel, float dither_strength, const double c0, const double c1, const double c2) {
      double bias_dither = -(256.0f * dither_strength - 1) / 2; // -127.5 for 8 bit, scaling because of dithershift
      double pixel_max = (1 << bits_per_pixel) - 1;
      if (bits_per_pixel == 8) {
        for (int i = 0; i < lookup_size * 256; ++i) {
          int ii = (i & 0xFFFFFF00) + (int)((i & 0xFF)*dither_strength);
          map[i] = BYTE(pow(clamp((c0 * 256 + ii * c1 - bias_dither) / (double(pixel_max) * 256), 0.0, 1.0), c2) * (double)pixel_max + 0.5);
        }
      }
      else {
        for (int i = 0; i < lookup_size * 256; ++i) {
          int ii = (i & 0xFFFFFF00) + (int)((i & 0xFF)*dither_strength);
          reinterpret_cast<uint16_t *>(map)[i] = uint16_t(pow(clamp((c0 * 256 + ii * c1 - bias_dither) / (double(pixel_max) * 256), 0.0, 1.0), c2) * (double)pixel_max + 0.5);
        }
      }
    };
  }
  else {
    set_map = [](BYTE* map, int lookup_size, int bits_per_pixel, float dither_strength, const double c0, const double c1, const double c2) {
      double pixel_max = (1 << bits_per_pixel) - 1;
      if (bits_per_pixel == 8) {
        for (int i = 0; i < lookup_size; ++i) { // fix of bug introduced in an earlier refactor was: i < 256 * 256
          map[i] = BYTE(pow(clamp((c0 + i * c1) / (double)pixel_max, 0.0, 1.0), c2) * double(pixel_max) + 0.5);
        }
      }
      else {
        for (int i = 0; i < lookup_size; ++i) { // fix of bug introduced in an earlier refactor was: i < 256 * 256
          reinterpret_cast<uint16_t *>(map)[i] = uint16_t(pow(clamp((c0 + i * c1) / (double)pixel_max, 0.0, 1.0), c2) * double(pixel_max) + 0.5);
        }
      }
    };
  }

  set_map(lut_buf, lookup_size, bits_per_pixel, dither_strength, cfg.rgba[plane].bias, cfg.rgba[plane].scale, 1 / cfg.rgba[plane].gamma);
}

RGBAdjust::RGBAdjust(PClip _child, double r, double g, double b, double a,
    double rb, double gb, double bb, double ab,
    double rg, double gg, double bg, double ag,
    bool _analyze, bool _dither, bool _conditional, const char *_condVarSuffix, IScriptEnvironment* env)
    : GenericVideoFilter(_child), analyze(_analyze), dither(_dither), condVarSuffix(_condVarSuffix)
{
    // one buffer for all maps
    map_holder = nullptr;

    if (!vi.IsRGB())
        env->ThrowError("RGBAdjust requires RGB input");

    config.rgba[0].scale = r;
    config.rgba[1].scale = g;
    config.rgba[2].scale = b;
    config.rgba[3].scale = a;
    // bias
    config.rgba[0].bias = rb;
    config.rgba[1].bias = gb;
    config.rgba[2].bias = bb;
    config.rgba[3].bias = ab;
    // gammas
    config.rgba[0].gamma = rg;
    config.rgba[1].gamma = gg;
    config.rgba[2].gamma = bg;
    config.rgba[3].gamma = ag;

    config.rgba[0].changed = false;
    config.rgba[1].changed = false;
    config.rgba[2].changed = false;
    config.rgba[3].changed = false;

    CheckAndConvertParams(config, env);

    pixelsize = vi.ComponentSize();
    bits_per_pixel = vi.BitsPerComponent(); // 8,10..16

    if (pixelsize == 4) {
      // dither parameter is silently ignored
      // if (dither) env->ThrowError("RGBAdjust: cannot 'dither' a 32bit float video");
    }
    // No lookup for float. todo: slow on-the-fly realtime calculation

    real_lookup_size = (pixelsize == 1) ? 256 : 65536; // avoids lut overflow in case of non-standard content of a 10 bit clip
    max_pixel_value = (pixelsize == 4) ? 255 : (1 << bits_per_pixel) - 1; // n/a for float formats
    dither_strength = 1.0f; // fixed, not used

    use_lut = bits_per_pixel != 32; // for float: realtime (todo)

    if (!use_lut)
      dither = false;

    if(use_lut) {
      number_of_maps = (vi.IsRGB24() || vi.IsRGB48() || vi.IsPlanarRGB()) ? 3 : 4;
      int one_bufsize = pixelsize * real_lookup_size;
      if (dither) one_bufsize *= 256;

      map_holder = new uint8_t[one_bufsize * number_of_maps];
      /*
      // left here intentionally:
      // for some reason, AtExit does not get called from within ScriptClip, causing no free thus memory leak
      // We are using new here and delete in destructor
      static_cast<uint8_t*>(env->Allocate(one_bufsize * number_of_maps, 16, AVS_NORMAL_ALLOC));
      if (!mapR)
          env->ThrowError("RGBAdjust: Could not reserve memory.");
      env->AtExit(free_buffer, mapR);
      */
      if(bits_per_pixel>8 && bits_per_pixel<16) // make lut table safe for 10-14 bit garbage
        std::fill_n(map_holder, one_bufsize * number_of_maps, 0); // 8 and 16 bit fully overwrites
      maps[0] = map_holder;
      maps[1] = maps[0] + one_bufsize;
      maps[2] = maps[1] + one_bufsize;
      maps[3] = number_of_maps == 4 ? maps[2] + one_bufsize : nullptr;

      for (int plane = 0; plane < number_of_maps; plane++) {
        rgbadjust_create_lut(maps[plane], plane, config);
      }
    }
}

template<typename pixel_t>
static void fill_accum_rgb_planar_c(const BYTE *srcpR, const BYTE* srcpG, const BYTE* srcpB, int pitch,
  unsigned int *accum_r, unsigned int *accum_g, unsigned int *accum_b,
  int width, int height, int max_pixel_value) {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int r = reinterpret_cast<const pixel_t*>(srcpR)[x];
        int g = reinterpret_cast<const pixel_t*>(srcpG)[x];
        int b = reinterpret_cast<const pixel_t*>(srcpB)[x];
        if constexpr (sizeof(pixel_t) != 1) {
          if (r > max_pixel_value) r = max_pixel_value;
          if (g > max_pixel_value) g = max_pixel_value;
          if (b > max_pixel_value) b = max_pixel_value;
        }
        accum_r[r]++;
        accum_g[g]++;
        accum_b[b]++;
      }
      srcpR += pitch;
      srcpG += pitch;
      srcpB += pitch;
    }
}

static void fill_accum_rgb_planar_float_c(const BYTE* srcpR, const BYTE* srcpG, const BYTE* srcpB, int pitch,
  unsigned int* accum_r, unsigned int* accum_g, unsigned int* accum_b,
  int width, int height, RGBStats &rgbplanedata) {

  for (int i = 0; i < 3; i++) {
    rgbplanedata.data[i].real_max = std::numeric_limits<float>::min();
    rgbplanedata.data[i].real_min = std::numeric_limits<float>::max();
    rgbplanedata.data[i].sum = 0.0;
  }

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      // convert range 0..1 to 0..65535 for loose_min/max fake histogram
      float r = reinterpret_cast<const float*>(srcpR)[x];
      float g = reinterpret_cast<const float*>(srcpG)[x];
      float b = reinterpret_cast<const float*>(srcpB)[x];
      int ri = (int)(clamp(r * 65535.0f + 0.5f, 0.0f, 65535.0f));
      int gi = (int)(clamp(g * 65535.0f + 0.5f, 0.0f, 65535.0f));
      int bi = (int)(clamp(b * 65535.0f + 0.5f, 0.0f, 65535.0f));
      if (r > rgbplanedata.data[0].real_max) rgbplanedata.data[0].real_max = r;
      if (r < rgbplanedata.data[0].real_min) rgbplanedata.data[0].real_min = r;
      rgbplanedata.data[0].sum += r;
      if (g > rgbplanedata.data[1].real_max) rgbplanedata.data[1].real_max = g;
      if (g < rgbplanedata.data[1].real_min) rgbplanedata.data[1].real_min = g;
      rgbplanedata.data[1].sum += g;
      if (b > rgbplanedata.data[2].real_max) rgbplanedata.data[2].real_max = b;
      if (b < rgbplanedata.data[2].real_min) rgbplanedata.data[2].real_min = b;
      rgbplanedata.data[2].sum += b;
      accum_r[ri]++;
      accum_g[gi]++;
      accum_b[bi]++;
    }
    srcpR += pitch;
    srcpG += pitch;
    srcpB += pitch;
  }
}

template<typename pixel_t>
static void fill_accum_rgb_packed_c(const BYTE *srcp, int pitch,
  unsigned int *accum_r, unsigned int *accum_g, unsigned int *accum_b,
  int work_width, int height, int pixel_step) {
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < work_width; x += pixel_step) {
      accum_r[reinterpret_cast<const pixel_t *>(srcp)[x + 2]]++;
      accum_g[reinterpret_cast<const pixel_t *>(srcp)[x + 1]]++;
      accum_b[reinterpret_cast<const pixel_t *>(srcp)[x + 0]]++;
    }
    srcp += pitch;
  }
}

template<typename pixel_t, int pixel_step, bool dither>
static void apply_map_rgb_packed_c(BYTE *dstp8, int pitch,
  BYTE *mapR, BYTE *mapG, BYTE *mapB, BYTE *mapA,
  int width, int height)
{
  int _y = 0;
  int _dither = 0;
  pixel_t *dstp = reinterpret_cast<pixel_t *>(dstp8);
  pitch /= sizeof(pixel_t);

  for (int y = 0; y < height; y++) {
    if(dither)
      _y = (y << 4) & 0xf0;
    for (int x = 0; x < width; x++) {
      if (dither)
        _dither = ditherMap[(x & 0x0f) | _y];
      dstp[x * pixel_step + 0] = reinterpret_cast<pixel_t *>(mapB)[dither ? dstp[x * pixel_step + 0] << 8 | _dither : dstp[x * pixel_step + 0]];
      dstp[x * pixel_step + 1] = reinterpret_cast<pixel_t *>(mapG)[dither ? dstp[x * pixel_step + 1] << 8 | _dither : dstp[x * pixel_step + 1]];
      dstp[x * pixel_step + 2] = reinterpret_cast<pixel_t *>(mapR)[dither ? dstp[x * pixel_step + 2] << 8 | _dither : dstp[x * pixel_step + 2]];
      if constexpr(pixel_step == 4)
        dstp[x * pixel_step + 3] = reinterpret_cast<pixel_t *>(mapA)[dither ? dstp[x * pixel_step + 3] << 8 | _dither : dstp[x * pixel_step + 3]];
    }
    dstp += pitch;
  }
}

template<typename pixel_t, bool hasAlpha, bool dither>
static void apply_map_rgb_planar_c(BYTE *dstpR8, BYTE *dstpG8, BYTE *dstpB8, BYTE *dstpA8, int pitch,
  BYTE *mapR, BYTE *mapG, BYTE *mapB, BYTE *mapA,
  int width, int height)
{
  int _y = 0;
  int _dither = 0;
  pixel_t *dstpR = reinterpret_cast<pixel_t *>(dstpR8);
  pixel_t *dstpG = reinterpret_cast<pixel_t *>(dstpG8);
  pixel_t *dstpB = reinterpret_cast<pixel_t *>(dstpB8);
  pixel_t *dstpA = reinterpret_cast<pixel_t *>(dstpA8);
  pitch /= sizeof(pixel_t);

  for (int y = 0; y < height; y++) {
    if(dither)
      _y = (y << 4) & 0xf0;
    for (int x = 0; x < width; x++) {
      if (dither)
        _dither = ditherMap[(x & 0x0f) | _y];
      reinterpret_cast<pixel_t *>(dstpG)[x] = reinterpret_cast<pixel_t *>(mapG)[dither ? dstpG[x] << 8 | _dither : dstpG[x]];
      reinterpret_cast<pixel_t *>(dstpB)[x] = reinterpret_cast<pixel_t *>(mapB)[dither ? dstpB[x] << 8 | _dither : dstpB[x]];
      reinterpret_cast<pixel_t *>(dstpR)[x] = reinterpret_cast<pixel_t *>(mapR)[dither ? dstpR[x] << 8 | _dither : dstpR[x]];
      if(hasAlpha)
        reinterpret_cast<pixel_t *>(dstpA)[x] = reinterpret_cast<pixel_t *>(mapA)[dither ? dstpA[x] << 8 | _dither : dstpA[x]];
    }
    dstpG += pitch; dstpB += pitch; dstpR += pitch;
    if(hasAlpha)
      dstpA += pitch;
  }
}


PVideoFrame __stdcall RGBAdjust::GetFrame(int n, IScriptEnvironment* env)
{
    PVideoFrame frame = child->GetFrame(n, env);
    env->MakeWritable(&frame);
    BYTE* p = frame->GetWritePtr();
    const int pitch = frame->GetPitch();

    int w = vi.width;
    int h = vi.height;

    RGBAdjustConfig local_config = config;

    // Read conditional variables
    local_config.rgba[0].changed = false;
    local_config.rgba[1].changed = false;
    local_config.rgba[2].changed = false;
    local_config.rgba[3].changed = false;
    rgbadjust_read_conditional(env, &local_config, condVarSuffix);

    BYTE *maps_live[4] = { nullptr };
    BYTE *maps_local[4] = { nullptr }; // for local lut table allocation, don't overwrite common buffer
    for (int i = 0; i < 4; i++)
      maps_live[i] = maps[i];

    if (local_config.rgba[0].changed || local_config.rgba[1].changed || local_config.rgba[2].changed || local_config.rgba[3].changed) {
      CheckAndConvertParams(local_config, env);
      if (use_lut) {
        for (int plane = 0; plane < (int)number_of_maps; plane++) {
          // recalculate plane LUT only if changed
          if (local_config.rgba[plane].changed)
          {
            maps_local[plane] = new BYTE[pixelsize * real_lookup_size];
            maps_live[plane] = maps_local[plane]; // use our new local lut
            rgbadjust_create_lut(maps_live[plane], plane, local_config);
          }
        }
      }
    }

    if (dither) {
      if (vi.IsRGB32())
        apply_map_rgb_packed_c<uint8_t, 4, true>(p, pitch, maps_live[0], maps_live[1], maps_live[2], maps_live[3], w, h);
      else if(vi.IsRGB24())
        apply_map_rgb_packed_c<uint8_t, 3, true>(p, pitch, maps_live[0], maps_live[1], maps_live[2], maps_live[3], w, h);
      else if(vi.IsRGB64())
        apply_map_rgb_packed_c<uint16_t, 4, true>(p, pitch, maps_live[0], maps_live[1], maps_live[2], maps_live[3], w, h);
      else if(vi.IsRGB48())
        apply_map_rgb_packed_c<uint16_t, 3, true>(p, pitch, maps_live[0], maps_live[1], maps_live[2], maps_live[3], w, h);
      else {
        // Planar RGB
        bool hasAlpha = vi.IsPlanarRGBA();
        BYTE *p_g = p;
        BYTE *p_b = frame->GetWritePtr(PLANAR_B);
        BYTE *p_r = frame->GetWritePtr(PLANAR_R);
        BYTE *p_a = frame->GetWritePtr(PLANAR_A);
        // no float support
        if(pixelsize==1) {
          if(hasAlpha)
            apply_map_rgb_planar_c<uint8_t, true, true>(p_r, p_g, p_b, p_a, pitch, maps_live[0], maps_live[1], maps_live[2], maps_live[3], w, h);
          else
            apply_map_rgb_planar_c<uint8_t, false, true>(p_r, p_g, p_b, p_a, pitch, maps_live[0], maps_live[1], maps_live[2], maps_live[3], w, h);
        }
        else {
          if(hasAlpha)
            apply_map_rgb_planar_c<uint16_t, true, true>(p_r, p_g, p_b, p_a, pitch, maps_live[0], maps_live[1], maps_live[2], maps_live[3], w, h);
          else
            apply_map_rgb_planar_c<uint16_t, false, true>(p_r, p_g, p_b, p_a, pitch, maps_live[0], maps_live[1], maps_live[2], maps_live[3], w, h);
        }
      }
    }
    else {
      // no dither
      if (vi.IsRGB32())
        apply_map_rgb_packed_c<uint8_t, 4, false>(p, pitch, maps_live[0], maps_live[1], maps_live[2], maps_live[3], w, h);
      else if(vi.IsRGB24())
        apply_map_rgb_packed_c<uint8_t, 3, false>(p, pitch, maps_live[0], maps_live[1], maps_live[2], maps_live[3], w, h);
      else if(vi.IsRGB64())
        apply_map_rgb_packed_c<uint16_t, 4, false>(p, pitch, maps_live[0], maps_live[1], maps_live[2], maps_live[3], w, h);
      else if(vi.IsRGB48())
        apply_map_rgb_packed_c<uint16_t, 3, false>(p, pitch, maps_live[0], maps_live[1], maps_live[2], maps_live[3], w, h);
      else {
          // Planar RGB
        bool hasAlpha = vi.IsPlanarRGBA();
        BYTE *p_g = p;
        BYTE *p_b = frame->GetWritePtr(PLANAR_B);
        BYTE *p_r = frame->GetWritePtr(PLANAR_R);
        BYTE *p_a = frame->GetWritePtr(PLANAR_A);
        // no float support
        if(pixelsize==1) {
          if(hasAlpha)
            apply_map_rgb_planar_c<uint8_t, true, false>(p_r, p_g, p_b, p_a, pitch, maps_live[0], maps_live[1], maps_live[2], maps_live[3], w, h);
          else
            apply_map_rgb_planar_c<uint8_t, false, false>(p_r, p_g, p_b, p_a, pitch, maps_live[0], maps_live[1], maps_live[2], maps_live[3], w, h);
        }
        else if(pixelsize==2) {
          if(hasAlpha)
            apply_map_rgb_planar_c<uint16_t, true, false>(p_r, p_g, p_b, p_a, pitch, maps_live[0], maps_live[1], maps_live[2], maps_live[3], w, h);
          else
            apply_map_rgb_planar_c<uint16_t, false, false>(p_r, p_g, p_b, p_a, pitch, maps_live[0], maps_live[1], maps_live[2], maps_live[3], w, h);
        }
        else {
          // 32 bit float, no dither
          const int planesRGB_RgbaOrder[4] = { PLANAR_R, PLANAR_G, PLANAR_B, PLANAR_A };
          const int *planes = planesRGB_RgbaOrder;

          for (int cplane = 0; cplane < (hasAlpha ? 4 : 3); cplane++) {
            RGBAdjustPlaneConfig x = local_config.rgba[cplane];
            const double scale = x.scale;
            const double bias = x.bias;
            const double gamma = 1 / x.gamma;
            int plane = planes[cplane];
            float* dstp = reinterpret_cast<float *>(frame->GetWritePtr(plane));
            int pitch = frame->GetPitch(plane) / sizeof(float);
            for (int y = 0; y < h; y++) {
              for (int x = 0; x < w; x++) {
                dstp[x] = RGBAdjust_processPixel(dstp[x], bias, scale, gamma);
              }
              dstp += pitch;
            }
          }
        }
      }
    }

    if (use_lut && conditional) {
      for(int i = 0; i<4; i++)
        if (maps_local[i]) delete[] maps_local[i];
    }

    if (analyze) {
        const int w = frame->GetRowSize() / pixelsize;
        const int h = frame->GetHeight();

        const int analyze_lookup_size = pixelsize == 4 ? 1 << 16 : 1 << bits_per_pixel; // 32 bit float is quantized to 16 bits
        const int max_pixel_value_analyze = analyze_lookup_size - 1;

        // allocate 3x bufsize for R. G and B will share it
        auto accum_r = static_cast<uint32_t*>(env->Allocate(analyze_lookup_size * sizeof(uint32_t) * 3, 16, AVS_NORMAL_ALLOC));
        auto accum_g = accum_r + analyze_lookup_size;
        auto accum_b = accum_g + analyze_lookup_size;
        if (!accum_r)
          env->ThrowError("RGBAdjust: Could not reserve memory.");

        for (int i = 0; i < analyze_lookup_size; i++) {
          accum_r[i] = 0;
          accum_g[i] = 0;
          accum_b[i] = 0;
        }

        const int pixels = vi.width * vi.height;

        if (bits_per_pixel == 32) {
          RGBStats rgbplanedata;
          const BYTE* p_g = frame->GetReadPtr(PLANAR_G);;
          const BYTE* p_b = frame->GetReadPtr(PLANAR_B);
          const BYTE* p_r = frame->GetReadPtr(PLANAR_R);
          fill_accum_rgb_planar_float_c(p_r, p_g, p_b, pitch, accum_r, accum_g, accum_b, w, h, rgbplanedata);

          double avg_r = rgbplanedata.data[0].sum / pixels;
          double avg_g = rgbplanedata.data[1].sum / pixels;
          double avg_b = rgbplanedata.data[2].sum / pixels;

          int Amin_r = 0, Amin_g = 0, Amin_b = 0;
          int Amax_r = 0, Amax_g = 0, Amax_b = 0;
          bool Ahit_minr = false, Ahit_ming = false, Ahit_minb = false;
          bool Ahit_maxr = false, Ahit_maxg = false, Ahit_maxb = false;
          int At_256 = (pixels + 128) / 256; // When 1/256th of all pixels have been reached, trigger "Loose min/max"

          double st_r = 0, st_g = 0, st_b = 0;

          for (int i = 0; i < analyze_lookup_size; i++) {
            double i_float = i / 65535.0;
            st_r += accum_r[i] * (i_float - avg_r) * (i_float - avg_r);
            st_g += accum_g[i] * (i_float - avg_g) * (i_float - avg_g);
            st_b += accum_b[i] * (i_float - avg_b) * (i_float - avg_b);

            // loose min
            if (!Ahit_minr) { Amin_r += accum_r[i]; if (Amin_r > At_256) { Ahit_minr = true; Amin_r = i; } }
            if (!Ahit_ming) { Amin_g += accum_g[i]; if (Amin_g > At_256) { Ahit_ming = true; Amin_g = i; } }
            if (!Ahit_minb) { Amin_b += accum_b[i]; if (Amin_b > At_256) { Ahit_minb = true; Amin_b = i; } }
            // loose max
            if (!Ahit_maxr) { Amax_r += accum_r[max_pixel_value_analyze - i]; if (Amax_r > At_256) { Ahit_maxr = true; Amax_r = max_pixel_value_analyze - i; } }
            if (!Ahit_maxg) { Amax_g += accum_g[max_pixel_value_analyze - i]; if (Amax_g > At_256) { Ahit_maxg = true; Amax_g = max_pixel_value_analyze - i; } }
            if (!Ahit_maxb) { Amax_b += accum_b[max_pixel_value_analyze - i]; if (Amax_b > At_256) { Ahit_maxb = true; Amax_b = max_pixel_value_analyze - i; } }
          }

          auto Fst_r = sqrt(st_r / pixels);
          auto Fst_g = sqrt(st_g / pixels);
          auto Fst_b = sqrt(st_b / pixels);

          char text[512];
          const bool StatsAsInteger16 = false;
          if(StatsAsInteger16)
            sprintf(text,
              "At 16 bits.  Frame: %-8u (   Red   /  Green  /  Blue   )\n"
              "           Average:      ( %7.2f / %7.2f / %7.2f )\n"
              "Standard Deviation:      ( %7.2f / %7.2f / %7.2f )\n"
              "           Minimum:      ( %7.2f / %7.2f / %7.2f )\n"
              "           Maximum:      ( %7.2f / %7.2f / %7.2f )\n"
              "     Loose Minimum:      ( %5d    / %5d    / %5d    )\n"
              "     Loose Maximum:      ( %5d    / %5d    / %5d    )\n"
              ,
              (unsigned int)n,
              avg_r * 65535, avg_g * 65535, avg_b * 65535,
              Fst_r * 65535, Fst_g * 65535, Fst_b * 65535,
              rgbplanedata.data[0].real_min * 65535, rgbplanedata.data[1].real_min * 65535, rgbplanedata.data[2].real_min * 65535,
              rgbplanedata.data[0].real_max * 65535, rgbplanedata.data[1].real_max * 65535, rgbplanedata.data[2].real_max * 65535,
              Amin_r, Amin_g, Amin_b,
              Amax_r, Amax_g, Amax_b
            );
          else // stats as Float
            sprintf(text,
              "             Frame: %-8u (   Red   /  Green  /  Blue   )\n"
              "           Average:      ( %7.5f / %7.5f / %7.5f )\n"
              "Standard Deviation:      ( %7.5f / %7.5f / %7.5f )\n"
              "           Minimum:      ( %7.5f / %7.5f / %7.5f )\n"
              "           Maximum:      ( %7.5f / %7.5f / %7.5f )\n"
              "     Loose Minimum:      ( %7.5f / %7.5f / %7.5f )\n"
              "     Loose Maximum:      ( %7.5f / %7.5f / %7.5f )\n"
              ,
              (unsigned int)n,
              avg_r, avg_g, avg_b,
              Fst_r, Fst_g, Fst_b,
              rgbplanedata.data[0].real_min, rgbplanedata.data[1].real_min, rgbplanedata.data[2].real_min,
              rgbplanedata.data[0].real_max, rgbplanedata.data[1].real_max, rgbplanedata.data[2].real_max,
              Amin_r / 65535.0, Amin_g / 65535.0, Amin_b / 65535.0,
              Amax_r / 65535.0, Amax_g / 65535.0, Amax_b / 65535.0
            );
          env->ApplyMessage(&frame, vi, text, vi.width / 4, 0xa0a0a0, 0, 0);
        }
        else {

          if (vi.IsPlanarRGB() || vi.IsPlanarRGBA())
          {
            const BYTE* p_g = frame->GetReadPtr(PLANAR_G);;
            const BYTE* p_b = frame->GetReadPtr(PLANAR_B);
            const BYTE* p_r = frame->GetReadPtr(PLANAR_R);
            if (bits_per_pixel == 8)
              fill_accum_rgb_planar_c<uint8_t>(p_r, p_g, p_b, pitch, accum_r, accum_g, accum_b, w, h, max_pixel_value_analyze);
            else if (bits_per_pixel <= 16)
              fill_accum_rgb_planar_c<uint16_t>(p_r, p_g, p_b, pitch, accum_r, accum_g, accum_b, w, h, max_pixel_value_analyze);
            else // 32 bit float
              ;// handled in other branch;
          }
          else {
            // packed RGB
            const BYTE* srcp = frame->GetReadPtr();
            const int pixel_step = vi.IsRGB24() || vi.IsRGB48() ? 3 : 4;

            if (pixelsize == 1)
              fill_accum_rgb_packed_c<uint8_t>(srcp, pitch, accum_r, accum_g, accum_b, w, h, pixel_step);
            else
              fill_accum_rgb_packed_c<uint16_t>(srcp, pitch, accum_r, accum_g, accum_b, w, h, pixel_step);
          }

          double avg_r = 0, avg_g = 0, avg_b = 0;
          double st_r = 0, st_g = 0, st_b = 0;
          int min_r = 0, min_g = 0, min_b = 0;
          int max_r = 0, max_g = 0, max_b = 0;
          bool hit_r = false, hit_g = false, hit_b = false;
          int Amin_r = 0, Amin_g = 0, Amin_b = 0;
          int Amax_r = 0, Amax_g = 0, Amax_b = 0;
          bool Ahit_minr = false, Ahit_ming = false, Ahit_minb = false;
          bool Ahit_maxr = false, Ahit_maxg = false, Ahit_maxb = false;
          int At_256 = (pixels + 128) / 256; // When 1/256th of all pixels have been reached, trigger "Loose min/max"

          for (int i = 0; i < analyze_lookup_size; i++) {
            avg_r += (double)accum_r[i] * i;
            avg_g += (double)accum_g[i] * i;
            avg_b += (double)accum_b[i] * i;

            if (accum_r[i] != 0) { max_r = i; hit_r = true; }
            else { if (!hit_r) min_r = i + 1; }
            if (accum_g[i] != 0) { max_g = i; hit_g = true; }
            else { if (!hit_g) min_g = i + 1; }
            if (accum_b[i] != 0) { max_b = i; hit_b = true; }
            else { if (!hit_b) min_b = i + 1; }

            if (!Ahit_minr) { Amin_r += accum_r[i]; if (Amin_r > At_256) { Ahit_minr = true; Amin_r = i; } }
            if (!Ahit_ming) { Amin_g += accum_g[i]; if (Amin_g > At_256) { Ahit_ming = true; Amin_g = i; } }
            if (!Ahit_minb) { Amin_b += accum_b[i]; if (Amin_b > At_256) { Ahit_minb = true; Amin_b = i; } }

            if (!Ahit_maxr) { Amax_r += accum_r[max_pixel_value_analyze - i]; if (Amax_r > At_256) { Ahit_maxr = true; Amax_r = max_pixel_value_analyze - i; } }
            if (!Ahit_maxg) { Amax_g += accum_g[max_pixel_value_analyze - i]; if (Amax_g > At_256) { Ahit_maxg = true; Amax_g = max_pixel_value_analyze - i; } }
            if (!Ahit_maxb) { Amax_b += accum_b[max_pixel_value_analyze - i]; if (Amax_b > At_256) { Ahit_maxb = true; Amax_b = max_pixel_value_analyze - i; } }
          }

          float Favg_r = (float)(avg_r / pixels);
          float Favg_g = (float)(avg_g / pixels);
          float Favg_b = (float)(avg_b / pixels);

          for (int i = 0; i < analyze_lookup_size; i++) {
            st_r += (float)accum_r[i] * (float(i - Favg_r) * (i - Favg_r));
            st_g += (float)accum_g[i] * (float(i - Favg_g) * (i - Favg_g));
            st_b += (float)accum_b[i] * (float(i - Favg_b) * (i - Favg_b));
          }

          float Fst_r = (float)sqrt(st_r / pixels);
          float Fst_g = (float)sqrt(st_g / pixels);
          float Fst_b = (float)sqrt(st_b / pixels);

          char text[512];
          if (bits_per_pixel == 8)
            sprintf(text,
              "             Frame: %-8u (  Red  / Green / Blue  )\n"
              "           Average:      ( %5.2f / %5.2f / %5.2f )\n"
              "Standard Deviation:      ( %5.2f / %5.2f / %5.2f )\n"
              "           Minimum:      ( %3d    / %3d    / %3d    )\n"
              "           Maximum:      ( %3d    / %3d    / %3d    )\n"
              "     Loose Minimum:      ( %3d    / %3d    / %3d    )\n"
              "     Loose Maximum:      ( %3d    / %3d    / %3d    )\n"
              ,
              (unsigned int)n,
              Favg_r, Favg_g, Favg_b,
              Fst_r, Fst_g, Fst_b,
              min_r, min_g, min_b,
              max_r, max_g, max_b,
              Amin_r, Amin_g, Amin_b,
              Amax_r, Amax_g, Amax_b
            );
          else // if (bits_per_pixel <= 16)
            sprintf(text,
              "             Frame: %-8u (  Red  / Green / Blue  )\n"
              "           Average:      ( %7.2f / %7.2f / %7.2f )\n"
              "Standard Deviation:      ( %7.2f / %7.2f / %7.2f )\n"
              "           Minimum:      ( %5d    / %5d    / %5d    )\n"
              "           Maximum:      ( %5d    / %5d    / %5d    )\n"
              "     Loose Minimum:      ( %5d    / %5d    / %5d    )\n"
              "     Loose Maximum:      ( %5d    / %5d    / %5d    )\n"
              ,
              (unsigned int)n,
              Favg_r, Favg_g, Favg_b,
              Fst_r, Fst_g, Fst_b,
              min_r, min_g, min_b,
              max_r, max_g, max_b,
              Amin_r, Amin_g, Amin_b,
              Amax_r, Amax_g, Amax_b
            );
          env->ApplyMessage(&frame, vi, text, vi.width / 4, 0xa0a0a0, 0, 0);
        }
        env->Free(accum_r);
    }
    return frame;
}


AVSValue __cdecl RGBAdjust::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  return new RGBAdjust(args[ 0].AsClip(),
                       args[ 1].AsDblDef(1.0), args[ 2].AsDblDef(1.0), args[ 3].AsDblDef(1.0), args[ 4].AsDblDef(1.0),
                       args[ 5].AsDblDef(0.0), args[ 6].AsDblDef(0.0), args[ 7].AsDblDef(0.0), args[ 8].AsDblDef(0.0),
                       args[ 9].AsDblDef(1.0), args[10].AsDblDef(1.0), args[11].AsDblDef(1.0), args[12].AsDblDef(1.0),
                       args[13].AsBool(false), args[14].AsBool(false), args[15].AsBool(false), args[16].AsString(""), env );
}



/* helper function for Tweak and MaskHS filters */
static bool ProcessPixel(double X, double Y, double startHue, double endHue,
    double maxSat, double minSat, double p, int &iSat)
{
    // a hue analog
    double T = atan2(X, Y) * 180.0 / PI;
    if (T < 0.0) T += 360.0;

    // startHue <= hue <= endHue
    if (startHue < endHue) {
        if (T > endHue || T < startHue) return false;
    }
    else {
        if (T<startHue && T>endHue) return false;
    }

    const double W = X*X + Y*Y;

    // In Range, full adjust but no need to interpolate
    if (minSat*minSat <= W && W <= maxSat*maxSat) return true;

    // p == 0 (no interpolation) needed for MaskHS
    if (p == 0.0) return false;

    // Interpolation range is +/-p for p>0
    // 180 is not in degrees!
    // its sqrt(127^2 + 127^2): max overshoot for 8 bits; U and V is 0 +/- 127
    const double max = min(maxSat + p, 180.0);
    const double min = ::max(minSat - p, 0.0);

    // Outside of [min-p, max+p] no adjustment
    // minSat-p <= (U^2 + V^2) <= maxSat+p
    if (W <= min*min || max*max <= W) return false; // don't adjust

    // Interpolate saturation value
    const double holdSat = W < 180.0*180.0 ? sqrt(W) : 180.0;

    if (holdSat < minSat) { // within p of lower range
        iSat += (int)((512 - iSat) * (minSat - holdSat) / p);
    }
    else { // within p of upper range
        iSat += (int)((512 - iSat) * (holdSat - maxSat) / p);
    }

    return true;
}

// for float
static bool ProcessPixelUnscaled(double X, double Y, double startHue, double endHue,
    double maxSat, double minSat, double p, double &dSat)
{
    // a hue analog
    double T = atan2(X, Y) * 180.0 / PI;
    if (T < 0.0) T += 360.0;

    // startHue <= hue <= endHue
    if (startHue < endHue) {
        if (T > endHue || T < startHue) return false;
    }
    else {
        if (T<startHue && T>endHue) return false;
    }

    const double W = X*X + Y*Y;

    // In Range, full adjust but no need to interpolate
    if (minSat*minSat <= W && W <= maxSat*maxSat) return true;

    // p == 0 (no interpolation) needed for MaskHS
    if (p == 0.0) return false;

    // Interpolation range is +/-p for p>0
    const double max = min(maxSat + p, 180.0);
    const double min = ::max(minSat - p, 0.0);

    // Outside of [min-p, max+p] no adjustment
    // minSat-p <= (U^2 + V^2) <= maxSat+p
    if (W <= min*min || max*max <= W) return false; // don't adjust

    // Interpolate saturation value
    const double holdSat = W < 180.0*180.0 ? sqrt(W) : 180.0;

    if (holdSat < minSat) { // within p of lower range
        dSat += ((1 - dSat) * (minSat - holdSat) / p);
    }
    else { // within p of upper range
        dSat += ((1 - dSat) * (holdSat - maxSat) / p);
    }

    return true;
}


/**********************
******   Tweak    *****
**********************/
template<typename pixel_t, bool bpp10_14, bool dither>
void Tweak::tweak_calc_luma(BYTE *srcp, int src_pitch, float minY, float maxY, int width, int height)
{
  float ditherval = 0.0f;
  for (int y = 0; y < height; ++y) {
    const int _y = (y << 4) & 0xf0;
    for (int x = 0; x < width; ++x) {
      if (dither)
        ditherval = (ditherMap[(x & 0x0f) | _y] * dither_strength + bias_dither_luma) / (float)scale_dither_luma; // 0x00..0xFF -> -0.7F .. + 0.7F (+/- maxrange/512)
      float y0 = reinterpret_cast<pixel_t *>(srcp)[x] - minY;
      if(bpp10_14)
        y0 = minY + (y0 + ditherval)*(float)dcont + (float)(1 << (bits_per_pixel - 8))*(float)dbright; // dbright parameter always 0..255. Scale to 0..255*4, 0.. 255*256
      else if(pixelsize == 2)
        y0 = minY + (y0 + ditherval)*(float)dcont + 256.0f*(float)dbright; // dbright parameter always 0..255. Scale to 0..255*4, 0.. 255*256
      else if(pixelsize == 4)
        y0 = minY + (y0 + ditherval)*(float)dcont + (float)dbright / 256.0f; // dbright parameter always 0..255, scale it to 0..1
      else // pixelsize == 1
        y0 = minY + ((y0 + ditherval)*(float)dcont + 1.0f*(float)dbright); // dbright parameter always 0..255. Scale to 0..255*4, 0.. 255*256

      reinterpret_cast<pixel_t *>(srcp)[x] = (pixel_t)clamp(y0, minY, maxY);
      /*
      int y = int(((ii - range_low * scale_dither_luma)*_cont + _bright * scale_dither_luma + bias_dither_luma) / scale_dither_luma + range_low + 0.5); // 256 _cont & _bright param range
      // coring, dither:
      // int y = int(((ii - 16 * 256)*_cont + _bright * 256 - 127.5) / 256 + 16.5); // 256 _cont & _bright param range
      // coring, no dither:
      // int y = int(((ii - 16)*_cont + _bright) + 16.5); // 256 _cont & _bright param range
      // no coring, dither:
      // int y = int((ii *_cont + _bright * 256 - 127.5) / 256 + 0.5 ); // 256 _cont & _bright param range
      // no coring, no dither:
      // int y = int((ii *_cont + _bright) + 0.5 ); // 256 _cont & _bright param range
      */
    }
    srcp += src_pitch;
  }
}

Tweak::Tweak(PClip _child, double _hue, double _sat, double _bright, double _cont, bool _coring,
            double _startHue, double _endHue, double _maxSat, double _minSat, double p,
            bool _dither, bool _realcalc, double _dither_strength, IScriptEnvironment* env)
  : GenericVideoFilter(_child), coring(_coring), 
  dither(_dither), realcalc(_realcalc),
  dhue(_hue), dsat(_sat), dbright(_bright), dcont(_cont), dstartHue(_startHue), dendHue(_endHue),
  dmaxSat(_maxSat), dminSat(_minSat), dinterp(p), dither_strength((float)_dither_strength)
{
  if (vi.IsRGB())
        env->ThrowError("Tweak: YUV data only (no RGB)");

  pixelsize = vi.ComponentSize();
  bits_per_pixel = vi.BitsPerComponent();
  max_pixel_value = (1 << bits_per_pixel) - 1;
  lut_size = 1 << bits_per_pixel;
  int safe_luma_lookup_size = (pixelsize == 1) ? 256 : 65536; // avoids lut overflow in case of non-standard content of a 10 bit clip

  get_limits(limits, bits_per_pixel); // tv range limits

  scale_dither_luma = 1;
  divisor_dither_luma = 1;
  bias_dither_luma = 0.0;

  scale_dither_chroma = 1;
  divisor_dither_chroma = 1;
  bias_dither_chroma = 0.0;

  if (pixelsize == 4)
    dither_strength /= 65536.0f; // same dither range as for a 16 bit clip
  // Set dither_strength = 4.0 for 10 bits or 256.0 for 16 bits in order to have same dither range as for 8 bits
  // Otherwise dithering is always +/- 0.5 at all bit-depth

  if (dither) {
    // lut scale settings
    scale_dither_luma = 256; // lower 256 is dither value
    divisor_dither_luma *= 256;
    bias_dither_luma = -(256.0f * dither_strength - 1) / 2;
    // original bias: -127.5 or -(256.0f * dither_strength - 1) / 2;
    // dither strength =1 = (1 << (8-8))
    // dither min: int( (0*1-127.5)/256+0.5) = -0.498046875 + 0.5 = 0,001953125
    // dither max: int( (255*1-127.5)/256+0.5) = 0,998046875

    // 16 bit: 32767,5
    // dither strength =256 = (1 << (16-8))
    // dither min: int( (0*256-32767,5)/256+0.5)   = -127,498046875
    // dither max: int( (255*256-32767,5)/256+0.5) = 127,501953125


    scale_dither_chroma = 16; // lower 16 is dither value
    divisor_dither_chroma *= 16;
    bias_dither_chroma = -(16.0f * dither_strength - (pixelsize==4 ? 1/256.0f : 1)) / 2; // -7.5
  }

  // Flag to skip special processing if doing all pixels
  // If defaults, don't check for ranges, just do all
  allPixels = (_startHue == 0.0 && _endHue == 360.0 && _maxSat == 150.0 && _minSat == 0.0);

  if (vi.NumComponents() == 1) {
      if (!(_hue == 0.0 && _sat == 1.0 && allPixels))
      env->ThrowError("Tweak: bright and cont are the only options available for greyscale.");
  }

  if (_startHue < 0.0 || _startHue >= 360.0)
        env->ThrowError("Tweak: startHue must be greater than or equal to 0.0 and less than 360.0");

  if (_endHue <= 0.0 || _endHue > 360.0)
        env->ThrowError("Tweak: endHue must be greater than 0.0 and less than or equal to 360.0");

  if (_minSat >= _maxSat)
        env->ThrowError("Tweak: MinSat must be less than MaxSat");

  if (_minSat < 0.0 || _minSat >= 150.0)
        env->ThrowError("Tweak: minSat must be greater than or equal to 0 and less than 150.");

  if (_maxSat <= 0.0 || _maxSat > 150.0)
        env->ThrowError("Tweak: maxSat must be greater than 0 and less than or equal to 150.");

  if (p>=150.0 || p<0.0)
        env->ThrowError("Tweak: Interp must be greater than or equal to 0 and less than 150.");

  Sat = (int) (_sat * 512);    // 9 bits extra precision
  Cont = (int) (_cont * 512);
  Bright = (int) _bright;

  const double Hue = (_hue * PI) / 180.0;
  const double SIN = sin(Hue);
  const double COS = cos(Hue);

  Sin = (int) (SIN * 4096 + 0.5);
  Cos = (int) (COS * 4096 + 0.5);

  realcalc_luma = realcalc; // from parameter
  realcalc_chroma = realcalc;
  if (vi.IsPlanar() && (bits_per_pixel > 10))
    realcalc_chroma = true;
  if (vi.IsPlanar() && (bits_per_pixel == 32))
    realcalc_luma = true;
  // 8/10bit: chroma lut OK. 12+ bits: force no lookup tables.
  // 8-16bit: luma lut OK. float: force no lookup tables.

  // fill brightness/constrast lookup tables
  if(!(realcalc_luma && vi.IsPlanar()))
  {
    size_t map_size = pixelsize * safe_luma_lookup_size * scale_dither_luma;
    // for 10-16 bit with dither: 2 * 65536 * 256 = 33 MByte
    //               w/o  dither: 2 * 65536 = 128 KByte
    map = static_cast<uint8_t*>(env->Allocate(map_size, 8, AVS_NORMAL_ALLOC));
    if (!map)
      env->ThrowError("Tweak: Could not reserve memory.");
    env->AtExit(free_buffer, map);

    if(bits_per_pixel>8 && bits_per_pixel<16) // make lut table safe for 10-14 bit garbage
      std::fill_n((uint16_t *)map, map_size / pixelsize, max_pixel_value);

    int range_low = coring ? limits.tv_range_low : 0;
    int range_high = coring ? limits.tv_range_hi_luma : max_pixel_value;

    // dither_scale_luma = 1 if no dither, 256 if dither
    /* create luma lut for brightness and contrast */
    for (int i = 0; i < lut_size * scale_dither_luma; i++) {
      int ii;
      if(dither) {
        ii = (i & 0xFFFFFF00) + (int)((i & 0xFF)*dither_strength);
      } else {
        ii = i;
      }
      // _bright param range is accepted as 0..256
      int y = (int)(((ii - range_low * scale_dither_luma)*_cont + _bright * (1 << (bits_per_pixel - 8)) * scale_dither_luma + bias_dither_luma) / scale_dither_luma + range_low + 0.5);

      // coring, dither:
      // int y = int(((ii - 16 * 256)*_cont + _bright * 256 - 127.5) / 256 + 16.5); // 256 _cont & _bright param range
      // coring, no dither:
      // int y = int(((ii - 16)*_cont + _bright) + 16.5); // 256 _cont & _bright param range
      // no coring, dither:
      // int y = int((ii *_cont + _bright * 256 - 127.5) / 256 + 0.5 ); // 256 _cont & _bright param range
      // no coring, no dither:
      // int y = int((ii *_cont + _bright) + 0.5 ); // 256 _cont & _bright param range
      if(pixelsize==1)
        map[i] = (BYTE)clamp(y, range_low, range_high);
      else
        reinterpret_cast<uint16_t *>(map)[i] = (uint16_t)clamp(y, range_low, range_high);
    }
  }
  // 100% equals sat=119 (= maximal saturation of valid RGB (R=255,G=B=0)
  // 150% (=180) - 100% (=119) overshoot
  const double minSat = 1.19 * _minSat;
  const double maxSat = 1.19 * _maxSat;

  p *= 1.19; // Same units as minSat/maxSat

  if (!(realcalc_chroma && vi.IsPlanar()))
  { // fill lookup tables for UV
    size_t map_size = pixelsize * lut_size * lut_size * 2 * scale_dither_chroma;
    // for 10 bit with dither: 2 * 1024 * 1024 * 2 * 4 = 4*64 MByte = 256M huh!
    // for 10 bit w/o  dither: 2 * 1024 * 1024 * 2 = 64 MByte

    mapUV = static_cast<uint16_t*>(env->Allocate(map_size, 8, AVS_NORMAL_ALLOC)); // uint16_t for (U+V bytes), casted to uint32_t for (U+V words in non-8 bit)
    if (!mapUV)
      env->ThrowError("Tweak: Could not reserve memory.");
    env->AtExit(free_buffer, mapUV);

    int range_low = coring ? limits.tv_range_low : 0;
    int range_high = coring ? limits.tv_range_hi_chroma : max_pixel_value;

    double uv_range_corr = 1.0 / (1 << (bits_per_pixel - 8));

    if (dither) {
      // lut chroma, dither
      for (int d = 0; d < scale_dither_chroma; d++) { // scale = 4    0..15 mini-dither
        for (int u = 0; u < lut_size; u++) {
          // dither_strength: optional correction for 8+ bit to have the same dither range as in 8 bits
          const double destu = (((u << 4) + d*dither_strength) + bias_dither_chroma) / scale_dither_chroma - limits.middle_chroma; // scale_dither_chroma: 16
          for (int v = 0; v < lut_size; v++) {
            const double destv = (((v << 4) + d*dither_strength) + bias_dither_chroma) / scale_dither_chroma - limits.middle_chroma;
            int iSat = Sat;
            if (allPixels || ProcessPixel(destv * uv_range_corr, destu * uv_range_corr, _startHue, _endHue, maxSat, minSat, p, iSat)) {
              int du = (int)((destu*COS + destv*SIN) * iSat + 0x100) >> 9; // back from the extra 9 bits Sat precision
              int dv = (int)((destv*COS - destu*SIN) * iSat + 0x100) >> 9;
              du = clamp(du + limits.middle_chroma, range_low, range_high);
              dv = clamp(dv + limits.middle_chroma, range_low, range_high);
              if(pixelsize==1)
                mapUV[(u << 12) | (v << 4) | d] = (uint16_t)(du | (dv << 8)); // U and V: two bytes
              else
                reinterpret_cast<uint32_t *>(mapUV)[(u << (4+bits_per_pixel)) | (v << 4) | d] = (uint32_t)(du | (dv << 16)); // U and V: two words
            }
            else {
              if(pixelsize==1)
                mapUV[(u << 12) | (v << 4) | d] = (uint16_t)(clamp(u, range_low, range_high) | (clamp(v, range_low, range_high) << 8)); // U and V: two bytes
              else
                reinterpret_cast<uint32_t *>(mapUV)[(u << (4+bits_per_pixel)) | (v << 4) | d] = (uint32_t)(clamp(u, range_low, range_high) | ((clamp(v, range_low, range_high) << 16))); // U and V: two words
            }
          }
        }
      }
    }
    else {
      // lut chroma, no dither
      for (int u = 0; u < lut_size; u++) {
        const double destu = u - limits.middle_chroma;
        for (int v = 0; v < lut_size; v++) {
          const double destv = v - limits.middle_chroma;
          int iSat = Sat;
          if (allPixels || ProcessPixel(destv * uv_range_corr, destu * uv_range_corr, _startHue, _endHue, maxSat, minSat, p, iSat)) {
            int du = int((destu*COS + destv*SIN) * iSat) >> 9; // back from the extra 9 bits Sat precision
            int dv = int((destv*COS - destu*SIN) * iSat) >> 9;
            du = clamp(du + limits.middle_chroma, range_low, range_high);
            dv = clamp(dv + limits.middle_chroma, range_low, range_high);
            if(pixelsize==1)
              mapUV[(u << 8) | v] = (uint16_t)(du | (dv << 8)); // U and V: two bytes
            else
              reinterpret_cast<uint32_t *>(mapUV)[(u << bits_per_pixel) | v] = (uint32_t)(du | (dv << 16)); // U and V: two words
          }
          else {
            if(pixelsize==1)
              mapUV[(u << 8) | v] = (uint16_t)(clamp(u, range_low, range_high) | (clamp(v, range_low, range_high) << 8));  // U and V: two bytes
            else
              reinterpret_cast<uint32_t *>(mapUV)[(u << bits_per_pixel) | v] = (uint32_t)(clamp(u, range_low, range_high) | (clamp(v, range_low, range_high) << 16));  // U and V: two words
          }
        }
      }
    }
  }
}


template<typename pixel_t, bool dither>
void Tweak::tweak_calc_chroma(BYTE *srcpu, BYTE *srcpv, int src_pitch, int width, int height, float minUV, float maxUV)
{
  // no lookup, alway true for 16/32 bit, optional for 8 bit
  const double Hue = (dhue * PI) / 180.0;
  // 100% equals sat=119 (= maximal saturation of valid RGB (R=255,G=B=0)
  // 150% (=180) - 100% (=119) overshoot
  const double minSat = 1.19 * dminSat;
  const double maxSat = 1.19 * dmaxSat;

  const double p = dinterp * 1.19; // Same units as minSat/maxSat

  const int minUVi = (int)minUV;
  const int maxUVi = (int)maxUV;

  float ditherval = 0.0;
  float u, v;
  const float cosHue = (float)cos(Hue);
  const float sinHue = (float)sin(Hue);
  // no lut, realcalc, float internals
  const float pixel_range = sizeof(pixel_t) == 4 ? 1.0f : (float)(max_pixel_value + 1);

  double uv_range_corr = 255.0;

  const bool isFloat = sizeof(pixel_t) == 4;

  for (int y = 0; y < height; ++y) {
    const int _y = (y << 2) & 0xC;
    for (int x = 0; x < width; ++x) {
      if (dither)
        ditherval = ((float(ditherMap4[(x & 0x3) | _y]) * dither_strength + bias_dither_chroma) / scale_dither_chroma); // +/-0.5 on 0..255 range
      pixel_t orig_u = reinterpret_cast<pixel_t *>(srcpu)[x];
      pixel_t orig_v = reinterpret_cast<pixel_t *>(srcpv)[x] ;
      u = isFloat ? (orig_u - limits.middle_chroma_f) : (orig_u - limits.middle_chroma);
      v = isFloat ? (orig_v - limits.middle_chroma_f) : (orig_v - limits.middle_chroma);

      u = (u + (dither ? ditherval : 0)) / (sizeof(pixel_t) == 4 ? 1.0f : pixel_range); // going from 0..1 to +/-0.5
      v = (v + (dither ? ditherval : 0)) / (sizeof(pixel_t) == 4 ? 1.0f : pixel_range);

      double dWorkSat = dsat; // init from original param
      if(allPixels || ProcessPixelUnscaled(v * uv_range_corr, u * uv_range_corr, dstartHue, dendHue, maxSat, minSat, p, dWorkSat))
      {
        float du = ((u*cosHue + v*sinHue) * (float)dWorkSat);
        float dv = ((v*cosHue - u*sinHue) * (float)dWorkSat);

        if (isFloat) {
          du = du + limits.middle_chroma_f;
          dv = dv + limits.middle_chroma_f;
        }
        else {
          // back to 0..1
          du = du + 0.5f;
          dv = dv + 0.5f;
        }

        if(isFloat) {
          reinterpret_cast<pixel_t *>(srcpu)[x] = (pixel_t)clamp(du, minUV, maxUV);
          reinterpret_cast<pixel_t *>(srcpv)[x] = (pixel_t)clamp(dv, minUV, maxUV);
        } else {
          reinterpret_cast<pixel_t *>(srcpu)[x] = (pixel_t)clamp((int)(du * pixel_range), minUVi, maxUVi);
          reinterpret_cast<pixel_t *>(srcpv)[x] = (pixel_t)clamp((int)(dv * pixel_range), minUVi, maxUVi);
        }
      }
      else {
        if(isFloat) {
          reinterpret_cast<pixel_t *>(srcpu)[x] = (pixel_t)clamp((float)orig_u, minUV, maxUV);
          reinterpret_cast<pixel_t *>(srcpv)[x] = (pixel_t)clamp((float)orig_v, minUV, maxUV);
        } else {
          reinterpret_cast<pixel_t *>(srcpu)[x] = (pixel_t)clamp((int)(orig_u), minUVi, maxUVi);
          reinterpret_cast<pixel_t *>(srcpv)[x] = (pixel_t)clamp((int)(orig_v), minUVi, maxUVi);
        }
      }
    }
    srcpu += src_pitch;
    srcpv += src_pitch;
  }

}


PVideoFrame __stdcall Tweak::GetFrame(int n, IScriptEnvironment* env)
{
    PVideoFrame src = child->GetFrame(n, env);
    env->MakeWritable(&src);

    BYTE* srcp = src->GetWritePtr();

    int src_pitch = src->GetPitch();
    int height = src->GetHeight();
    int row_size = src->GetRowSize();

    if (vi.IsYUY2()) {

        if (dither) {
            const int UVwidth = vi.width / 2;
            for (int y = 0; y < height; y++) {
                {const int _y = (y << 4) & 0xf0;
                for (int x = 0; x < vi.width; ++x) {
                    /* brightness and contrast */
                    srcp[x * 2] = map[srcp[x * 2] << 8 | ditherMap[(x & 0x0f) | _y]];
                }}
                {const int _y = (y << 2) & 0xC;
                for (int x = 0; x < UVwidth; ++x) {
                    const int _dither = ditherMap4[(x & 0x3) | _y];
                    /* hue and saturation */
                    const int u = srcp[x * 4 + 1];
                    const int v = srcp[x * 4 + 3];
                    const int mapped = mapUV[(u << 12) | (v << 4) | _dither];
                    srcp[x * 4 + 1] = (BYTE)(mapped & 0xff);
                    srcp[x * 4 + 3] = (BYTE)(mapped >> 8);
                }}
                srcp += src_pitch;
            }
        }
        else {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < row_size; x += 4)
                {
                    /* brightness and contrast */
                    srcp[x] = map[srcp[x]];
                    srcp[x + 2] = map[srcp[x + 2]];

                    /* hue and saturation */
                    const int u = srcp[x + 1];
                    const int v = srcp[x + 3];
                    const int mapped = mapUV[(u << 8) | v];
                    srcp[x + 1] = (BYTE)(mapped & 0xff);
                    srcp[x + 3] = (BYTE)(mapped >> 8);
                }
                srcp += src_pitch;
            }
        }
        // YUY2 end
    }
    else if (vi.IsPlanar()) {
        // brightness and contrast
        // no_lut and lut
        int width = row_size / pixelsize;
        if (realcalc_luma)
        {
          // no luma lookup! alway true for 32 bit, optional for 8-16 bits
            float maxY;
            float minY;
            // unique for each bit-depth, difference in the innermost loop (speed)
            maxY = (float)(coring ? limits.tv_range_hi_luma : max_pixel_value);
            minY = (float)(coring ? limits.tv_range_low : 0);

            if(pixelsize == 1) {
              if(dither)
                tweak_calc_luma<uint8_t, false, true>(srcp, src_pitch, minY, maxY, width, height);
              else
                tweak_calc_luma<uint8_t, false, false>(srcp, src_pitch, minY, maxY, width, height);
            } else if (bits_per_pixel < 16) {
              if(dither)
                tweak_calc_luma<uint16_t, true, true>(srcp, src_pitch, minY, maxY, width, height);
              else
                tweak_calc_luma<uint16_t, true, false>(srcp, src_pitch, minY, maxY, width, height);
            } else if(bits_per_pixel == 16) {
              if(dither)
                tweak_calc_luma<uint16_t, false, true>(srcp, src_pitch, minY, maxY, width, height);
              else
                tweak_calc_luma<uint16_t, false, false>(srcp, src_pitch, minY, maxY, width, height);
            } else { // float
              maxY = coring ? 235.0f / 256 : 1.0f; // scale into 0..1 range
              minY = coring ? 16.0f / 256 : 0;
              if(dither)
                tweak_calc_luma<float, false, true>(srcp, src_pitch, minY, maxY, width, height);
              else
                tweak_calc_luma<float, false, false>(srcp, src_pitch, minY, maxY, width, height);
            }
        }
        else {
            /* brightness and contrast */
            // use luma lookup for 8-16 bits
            if (dither) {
              if(pixelsize==1) {
                for (int y = 0; y < height; ++y) {
                    const int _y = (y << 4) & 0xf0;
                    for (int x = 0; x < width; ++x) {
                        /* brightness and contrast */
                        srcp[x] = map[srcp[x] << 8 | ditherMap[(x & 0x0f) | _y]];
                    }
                    srcp += src_pitch;
                }
              }
              else { // pixelsize == 2
                for (int y = 0; y < height; ++y) {
                  const int _y = (y << 4) & 0xf0;
                  for (int x = 0; x < width; ++x) {
                    reinterpret_cast<uint16_t *>(srcp)[x] = reinterpret_cast<uint16_t *>(map)[reinterpret_cast<uint16_t *>(srcp)[x] << 8 | ditherMap[(x & 0x0f) | _y]];
                    // no clamp, map is safely sized
                  }
                  srcp += src_pitch;
                }
              }
            }
            else {
              if(pixelsize==1) {
                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x) {
                        srcp[x] = map[srcp[x]];
                    }
                    srcp += src_pitch;
                }
              }
              else { // pixelsize == 2
                for (int y = 0; y < height; ++y) {
                  for (int x = 0; x < width; ++x) {
                    reinterpret_cast<uint16_t *>(srcp)[x] = reinterpret_cast<uint16_t *>(map)[reinterpret_cast<uint16_t *>(srcp)[x]];
                    // no clamp, map is safely sized
                  }
                  srcp += src_pitch;
                }
              }
            }
        }
        // Y: brightness and contrast done

        // UV: hue and saturation start
        src_pitch = src->GetPitch(PLANAR_U);
        BYTE * srcpu = src->GetWritePtr(PLANAR_U);
        BYTE * srcpv = src->GetWritePtr(PLANAR_V);
        row_size = src->GetRowSize(PLANAR_U);
        height = src->GetHeight(PLANAR_U);
        width = row_size / pixelsize;

        if (realcalc_chroma) {
          // no lookup, alway true for > 10 bit, optional for 8/10 bit
          float maxUV = (float)(coring ? limits.tv_range_hi_chroma : max_pixel_value);
          float minUV = (float)(coring ? limits.tv_range_low : 0);
          if(pixelsize == 1) {
            if (dither)
              tweak_calc_chroma<uint8_t, true>(srcpu, srcpv, src_pitch, width, height, minUV, maxUV);
            else
              tweak_calc_chroma<uint8_t, false>(srcpu, srcpv, src_pitch, width, height, minUV, maxUV);
          } else if(pixelsize==2) {
            if (dither)
              tweak_calc_chroma<uint16_t, true>(srcpu, srcpv, src_pitch, width, height, minUV, maxUV);
            else
              tweak_calc_chroma<uint16_t, false>(srcpu, srcpv, src_pitch, width, height, minUV, maxUV);
          } else { // pixelsize == 4
            maxUV = coring ? limits.tv_range_hi_chroma_f : limits.full_range_hi_chroma_f;
            minUV = coring ? limits.tv_range_low_chroma_f : limits.full_range_low_chroma_f;
            if (dither)
              tweak_calc_chroma<float, true>(srcpu, srcpv, src_pitch, width, height, minUV, maxUV);
            else
              tweak_calc_chroma<float, false>(srcpu, srcpv, src_pitch, width, height, minUV, maxUV);
          }
        }
        else { // lookup UV
            if (dither) {
              // lut + dither
              if(pixelsize==1) {
                for (int y = 0; y < height; ++y) {
                    const int _y = (y << 2) & 0xC;
                    for (int x = 0; x < width; ++x) {
                        const int _dither = ditherMap4[(x & 0x3) | _y];
                        /* hue and saturation */
                        const int u = srcpu[x];
                        const int v = srcpv[x];
                        const int mapped = mapUV[(u << 12) | (v << 4) | _dither];
                        srcpu[x] = (BYTE)(mapped & 0xff);
                        srcpv[x] = (BYTE)(mapped >> 8);
                    }
                    srcpu += src_pitch;
                    srcpv += src_pitch;
                }
              }
              else { // pixelsize == 2
                for (int y = 0; y < height; ++y) {
                  const int _y = (y << 2) & 0xC;
                  for (int x = 0; x < width; ++x) {
                    const int _dither = ditherMap4[(x & 0x3) | _y]; // 0..15
                    /* hue and saturation */
                    const int u = clamp(0,(int)reinterpret_cast<uint16_t *>(srcpu)[x], max_pixel_value);
                    const int v = clamp(0,(int)reinterpret_cast<uint16_t *>(srcpv)[x], max_pixel_value);
                    const unsigned int mapped = reinterpret_cast<uint32_t *>(mapUV)[(u << (4+bits_per_pixel)) | (v << 4) | _dither];
                    reinterpret_cast<uint16_t *>(srcpu)[x] = (uint16_t)(mapped & 0xffff);
                    reinterpret_cast<uint16_t *>(srcpv)[x] = (uint16_t)(mapped >> 16);
                  }
                  srcpu += src_pitch;
                  srcpv += src_pitch;
                }
              }
            }
            else {
              // lut + no dither
              if(pixelsize==1) {
                for (int y = 0; y < height; ++y) {
                      for (int x = 0; x < width; ++x) {
                          /* hue and saturation */
                          const int u = srcpu[x];
                          const int v = srcpv[x];
                          const int mapped = mapUV[(u << 8) | v];
                          srcpu[x] = (BYTE)(mapped & 0xff);
                          srcpv[x] = (BYTE)(mapped >> 8);
                      }
                      srcpu += src_pitch;
                      srcpv += src_pitch;
                  }
              }
              else { // pixelsize == 2
                for (int y = 0; y < height; ++y) {
                  for (int x = 0; x < width; ++x) {
                    const int u = clamp(0,(int)reinterpret_cast<uint16_t *>(srcpu)[x], max_pixel_value);
                    const int v = clamp(0,(int)reinterpret_cast<uint16_t *>(srcpv)[x], max_pixel_value);
                    const unsigned int mapped = reinterpret_cast<uint32_t *>(mapUV)[(u << bits_per_pixel) | v];
                    reinterpret_cast<uint16_t *>(srcpu)[x] = (uint16_t)(mapped & 0xffff);
                    reinterpret_cast<uint16_t *>(srcpv)[x] = (uint16_t)(mapped >> 16);
                  }
                  srcpu += src_pitch;
                  srcpv += src_pitch;
                }
              }
            }
        }
    }

    return src;
}

AVSValue __cdecl Tweak::Create(AVSValue args, void* , IScriptEnvironment* env)
{
    return new Tweak(args[0].AsClip(),
        args[1].AsDblDef(0.0),     // hue
        args[2].AsDblDef(1.0),     // sat
        args[3].AsDblDef(0.0),     // bright
        args[4].AsDblDef(1.0),     // cont
        args[5].AsBool(true),      // coring
        // not used even on intel. // args[6].AsBool(false),     // sse
        args[7].AsDblDef(0.0),     // startHue
        args[8].AsDblDef(360.0),   // endHue
        args[9].AsDblDef(150.0),   // maxSat
        args[10].AsDblDef(0.0),    // minSat
        args[11].AsDblDef(16.0 / 1.19),// interp
        args[12].AsBool(false),    // dither
        args[13].AsBool(false),    // realcalc: force no-lookup (pure float calculation pixel)
        args[14].AsDblDef(1.0),    // dither_strength 1.0 = +/-0.5 on the 0.255 range, scaled for others
        env);
}

/**********************
******   MaskHS   *****
**********************/

MaskHS::MaskHS(PClip _child, double _startHue, double _endHue, double _maxSat, double _minSat, bool _coring, bool _realcalc,
    IScriptEnvironment* env)
    : GenericVideoFilter(_child), dstartHue(_startHue), dendHue(_endHue), dmaxSat(_maxSat), dminSat(_minSat), coring(_coring), realcalc(_realcalc)
{
    if (vi.IsRGB())
        env->ThrowError("MaskHS: YUV data only (no RGB)");

    if (vi.NumComponents() == 1) {
        env->ThrowError("MaskHS: clip must contain chroma.");
    }

    if (dstartHue < 0.0 || dstartHue >= 360.0)
        env->ThrowError("MaskHS: startHue must be greater than or equal to 0.0 and less than 360.0");

    if (dendHue <= 0.0 || dendHue > 360.0)
        env->ThrowError("MaskHS: endHue must be greater than 0.0 and less than or equal to 360.0");

    if (dminSat >= dmaxSat)
        env->ThrowError("MaskHS: MinSat must be less than MaxSat");

    if (dminSat < 0.0 || dminSat >= 150.0)
        env->ThrowError("MaskHS: minSat must be greater than or equal to 0 and less than 150.");

    if (dmaxSat <= 0.0 || dmaxSat > 150.0)
        env->ThrowError("MaskHS: maxSat must be greater than 0 and less than or equal to 150.");

    pixelsize = vi.ComponentSize();
    bits_per_pixel = vi.BitsPerComponent();
    max_pixel_value = (1 << bits_per_pixel) - 1;
    lut_size = 1 << bits_per_pixel;

    get_limits(limits, bits_per_pixel);

    mask_low = coring ? limits.tv_range_low : 0;
    mask_high = coring ? limits.tv_range_hi_luma : max_pixel_value;

    if (bits_per_pixel == 32) {
      mask_low_f = coring ? limits.tv_range_low_luma_f : limits.full_range_low_luma_f;
      mask_high_f = coring ? limits.tv_range_hi_luma_f : limits.full_range_hi_luma_f;
    }

    realcalc_chroma = realcalc;
    if (vi.IsPlanar() && (bits_per_pixel > 12)) // max bitdepth is 12 for lut
      realcalc_chroma = true;

    // 100% equals sat=119 (= maximal saturation of valid RGB (R=255,G=B=0)
    // 150% (=180) - 100% (=119) overshoot
    minSat = 1.19 * dminSat;
    maxSat = 1.19 * dmaxSat;

    if (!(realcalc_chroma && vi.IsPlanar()))
    { // fill lookup tables for UV
      size_t map_size = pixelsize * lut_size * lut_size;
      // for  8 bit : 1 * 256 * 256 = 65536 byte
      // for 10 bit : 2 * 1024 * 1024 = 2 MByte
      // for 12 bit : 2 * 4096 * 4096 = 32 MByte
      mapUV = static_cast<uint8_t*>(env->Allocate(map_size, 8, AVS_NORMAL_ALLOC)); // uint16_t for (U+V bytes), casted to uint32_t for (U+V words in non-8 bit)
      if (!mapUV)
        env->ThrowError("Tweak: Could not reserve memory.");
      env->AtExit(free_buffer, mapUV);

      // apply mask
      double uv_range_corr = 1.0 / (1 << (bits_per_pixel - 8)); // no float here
      for (int u = 0; u < lut_size; u++) {
          const double destu = (u - limits.middle_chroma) * uv_range_corr; // processpixel's minSat and maxSat is for 256 range
          int ushift = u << bits_per_pixel;
          for (int v = 0; v < lut_size; v++) {
              const double destv = (v - limits.middle_chroma) * uv_range_corr;
              int iSat = 0; // won't be used in MaskHS; interpolation is skipped since p==0:
              bool ppres = ProcessPixel(destv, destu, dstartHue, dendHue, maxSat, minSat, 0.0, iSat);
              if(pixelsize==1)
                  mapUV[ushift | v] = ppres ? mask_high : mask_low;
              else
                  reinterpret_cast<uint16_t *>(mapUV)[ushift | v] = ppres ? mask_high : mask_low;
          }
      }
    } // end of lut calculation
    // #define MaskPointResizing
#ifndef MaskPointResizing
    vi.width >>= vi.GetPlaneWidthSubsampling(PLANAR_U);
    vi.height >>= vi.GetPlaneHeightSubsampling(PLANAR_U);
#endif
    switch(bits_per_pixel) {
    case 8: vi.pixel_type = VideoInfo::CS_Y8; break;
    case 10: vi.pixel_type = VideoInfo::CS_Y10; break;
    case 12: vi.pixel_type = VideoInfo::CS_Y12; break;
    case 14: vi.pixel_type = VideoInfo::CS_Y14; break;
    case 16: vi.pixel_type = VideoInfo::CS_Y16; break;
    case 32: vi.pixel_type = VideoInfo::CS_Y32; break;
    }
}



PVideoFrame __stdcall MaskHS::GetFrame(int n, IScriptEnvironment* env)
{
    PVideoFrame src = child->GetFrame(n, env);
    PVideoFrame dst = env->NewVideoFrameP(vi, &src);

    uint8_t* dstp = dst->GetWritePtr();
    int dst_pitch = dst->GetPitch();

    // show mask
    if (child->GetVideoInfo().IsYUY2()) {
        const uint8_t* srcp = src->GetReadPtr();
        const int src_pitch = src->GetPitch();
        const int height = src->GetHeight();

#ifndef MaskPointResizing
        const int row_size = src->GetRowSize() >> 2;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < row_size; x++) {
                dstp[x] = mapUV[((srcp[x * 4 + 1]) << 8) | srcp[x * 4 + 3]];
            }
            srcp += src_pitch;
            dstp += dst_pitch;
        }
#else
        const int row_size = src->GetRowSize();

        for (int y = 0; y < height; y++) {
            for (int xs = 0, xd = 0; xs < row_size; xs += 4, xd += 2) {
                const BYTE mapped = mapY[((srcp[xs + 1]) << 8) | srcp[xs + 3]];
                dstp[xd] = mapped;
                dstp[xd + 1] = mapped;
            }
            srcp += src_pitch;
            dstp += dst_pitch;
        }
#endif
    }
    else if (child->GetVideoInfo().IsPlanar()) {
        const int srcu_pitch = src->GetPitch(PLANAR_U);
        const uint8_t* srcpu = src->GetReadPtr(PLANAR_U);
        const uint8_t* srcpv = src->GetReadPtr(PLANAR_V);
        const int width = src->GetRowSize(PLANAR_U) / pixelsize;
        const int heightu = src->GetHeight(PLANAR_U);

#ifndef MaskPointResizing
        if(realcalc_chroma) {
          double uv_range_corr = (pixelsize == 4) ? 255.0 : 1.0 / (1 << (bits_per_pixel - 8));
          if(pixelsize == 1) {
            for (int y = 0; y < heightu; ++y) {
              for (int x = 0; x < width; ++x) {
                const double destu = srcpu[x] - limits.middle_chroma;
                const double destv = srcpv[x] - limits.middle_chroma;
                int iSat = 0; // won't be used in MaskHS; interpolation is skipped since p==0:
                bool ppres = ProcessPixel(destv * uv_range_corr, destu * uv_range_corr, dstartHue, dendHue, maxSat, minSat, 0.0, iSat);
                dstp[x] = ppres ? mask_high : mask_low;
              }
              dstp += dst_pitch;
              srcpu += srcu_pitch;
              srcpv += srcu_pitch;
            }
          }
          else if (pixelsize == 2) {
            for (int y = 0; y < heightu; ++y) {
              for (int x = 0; x < width; ++x) {
                const double destu = (reinterpret_cast<const uint16_t *>(srcpu)[x] - limits.middle_chroma);
                const double destv = (reinterpret_cast<const uint16_t *>(srcpv)[x] - limits.middle_chroma);
                int iSat = 0; // won't be used in MaskHS; interpolation is skipped since p==0:
                bool ppres = ProcessPixel(destv * uv_range_corr, destu * uv_range_corr, dstartHue, dendHue, maxSat, minSat, 0.0, iSat);
                reinterpret_cast<uint16_t *>(dstp)[x] = ppres ? mask_high : mask_low;
              }
              dstp += dst_pitch;
              srcpu += srcu_pitch;
              srcpv += srcu_pitch;
            }
          } else { // pixelsize == 4
            for (int y = 0; y < heightu; ++y) {
              for (int x = 0; x < width; ++x) {
                const double destu = (reinterpret_cast<const float *>(srcpu)[x] - limits.middle_chroma_f);
                const double destv = (reinterpret_cast<const float *>(srcpv)[x] - limits.middle_chroma_f);
                int iSat = 0; // won't be used in MaskHS; interpolation is skipped since p==0:
                bool ppres = ProcessPixel(destv * uv_range_corr, destu * uv_range_corr, dstartHue, dendHue, maxSat, minSat, 0.0, iSat);
                reinterpret_cast<float *>(dstp)[x] = ppres ? mask_high_f : mask_low_f;
              }
              dstp += dst_pitch;
              srcpu += srcu_pitch;
              srcpv += srcu_pitch;
            }
          }

        } else {
          // use LUT
          if(pixelsize==1) {
            for (int y = 0; y < heightu; ++y) {
                for (int x = 0; x < width; ++x) {
                    dstp[x] = mapUV[((srcpu[x]) << 8) | srcpv[x]];
                }
                dstp += dst_pitch;
                srcpu += srcu_pitch;
                srcpv += srcu_pitch;
            }
          }
          else if (pixelsize == 2) {
            for (int y = 0; y < heightu; ++y) {
              for (int x = 0; x < width; ++x) {
                reinterpret_cast<uint16_t *>(dstp)[x] =
                  reinterpret_cast<uint16_t *>(mapUV)[((reinterpret_cast<const uint16_t *>(srcpu)[x]) << bits_per_pixel) | reinterpret_cast<const uint16_t *>(srcpv)[x]];
              }
              dstp += dst_pitch;
              srcpu += srcu_pitch;
              srcpv += srcu_pitch;
            }
          } // no lut for float (and for 14-16 bit)
        }
#else
        const int swidth = child->GetVideoInfo().GetPlaneWidthSubsampling(PLANAR_U);
        const int sheight = child->GetVideoInfo().GetPlaneHeightSubsampling(PLANAR_U);
        const int sw = 1 << swidth;
        const int sh = 1 << sheight;

        const int dpitch = dst_pitch << sheight;
        for (int y = 0; y < heightu; ++y) {
            for (int x = 0; x < row_sizeu; ++x) {
                const BYTE mapped = mapY[((srcpu[x]) << 8) | srcpv[x]];
                const int sx = x << swidth;

                for (int lumv = 0; lumv < sh; ++lumv) {
                    const int sy = lumv*dst_pitch + sx;

                    for (int lumh = 0; lumh < sw; ++lumh) {
                        dstp[sy + lumh] = mapped;
                    }
                }
            }
            dstp += dpitch;
            srcpu += srcu_pitch;
            srcpv += srcu_pitch;
        }
#endif
    }
    return dst;
}



AVSValue __cdecl MaskHS::Create(AVSValue args, void* , IScriptEnvironment* env)
{
    return new MaskHS(args[0].AsClip(),
        args[1].AsDblDef(0.0),    // startHue
        args[2].AsDblDef(360.0),    // endHue
        args[3].AsDblDef(150.0),    // maxSat
        args[4].AsDblDef(0.0),    // minSat
        args[5].AsBool(false),      // coring
        args[6].AsBool(false),      // realcalc
      env);
}

