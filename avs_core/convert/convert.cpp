// Avisynth v2.5.  Copyright 2002-2009 Ben Rudiak-Gould et al.
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


#include "convert.h"
#include "convert_matrix.h"
#include "convert_helper.h"
#include "convert_bits.h"
#include "convert_planar.h"
#include "convert_rgb.h"

#include <cmath>
#include <avs/alignment.h>
#include <avs/minmax.h>
#include <avs/config.h>
#include <tuple>
#include <map>
#include <algorithm>
#include <cassert>

#ifdef AVS_WINDOWS
#include <avs/win.h>
#else
#include <avs/posix.h>
#endif

/********************************************************************
***** Declare index of new filters for Avisynth's filter engine *****
********************************************************************/

extern const AVSFunction Convert_filters[] = {
  { "ConvertToRGB",   BUILTIN_FUNC_PREFIX, "c[matrix]s[interlaced]b[ChromaInPlacement]s[chromaresample]s[param1]f[param2]f[param3]f[bits]i[quality]b", CreateConvertToAdaptivePackedRGB, (void *)0 },
  { "ConvertToRGB24", BUILTIN_FUNC_PREFIX, "c[matrix]s[interlaced]b[ChromaInPlacement]s[chromaresample]s[param1]f[param2]f[param3]f[bits]i[quality]b", CreateConvertToPackedRGB, (void *)24 },
  { "ConvertToRGB32", BUILTIN_FUNC_PREFIX, "c[matrix]s[interlaced]b[ChromaInPlacement]s[chromaresample]s[param1]f[param2]f[param3]f[bits]i[quality]b", CreateConvertToPackedRGB, (void *)32 },
  { "ConvertToRGB48", BUILTIN_FUNC_PREFIX, "c[matrix]s[interlaced]b[ChromaInPlacement]s[chromaresample]s[param1]f[param2]f[param3]f[bits]i[quality]b", CreateConvertToPackedRGB, (void *)48 },
  { "ConvertToRGB64", BUILTIN_FUNC_PREFIX, "c[matrix]s[interlaced]b[ChromaInPlacement]s[chromaresample]s[param1]f[param2]f[param3]f[bits]i[quality]b", CreateConvertToPackedRGB, (void *)64 },
  { "ConvertToPlanarRGB",  BUILTIN_FUNC_PREFIX, "c[matrix]s[interlaced]b[ChromaInPlacement]s[chromaresample]s[param1]f[param2]f[param3]f[bits]i[quality]b", CreateConvertToRGB, (void *)-1 },
  { "ConvertToPlanarRGBA", BUILTIN_FUNC_PREFIX, "c[matrix]s[interlaced]b[ChromaInPlacement]s[chromaresample]s[param1]f[param2]f[param3]f[bits]i[quality]b", CreateConvertToRGB, (void *)-2 },
  { "ConvertToY8",    BUILTIN_FUNC_PREFIX, "c[matrix]s[bits]i[quality]b", ConvertToPlanarGeneric::CreateY, (void*)0 }, // user_data == 0 -> only 8 bit sources
  { "ConvertToYV12",  BUILTIN_FUNC_PREFIX, "c[interlaced]b[matrix]s[ChromaInPlacement]s[chromaresample]s[ChromaOutPlacement]s[param1]f[param2]f[param3]f[bits]i[quality]b", ConvertToPlanarGeneric::CreateYUV420, (void*)0 },
  { "ConvertToYV24",  BUILTIN_FUNC_PREFIX, "c[interlaced]b[matrix]s[ChromaInPlacement]s[chromaresample]s[param1]f[param2]f[param3]f[bits]i[quality]b", ConvertToPlanarGeneric::CreateYUV444, (void*)0},
  { "ConvertToYV16",  BUILTIN_FUNC_PREFIX, "c[interlaced]b[matrix]s[ChromaInPlacement]s[chromaresample]s[ChromaOutPlacement]s[param1]f[param2]f[param3]f[bits]i[quality]b", ConvertToPlanarGeneric::CreateYUV422, (void*)0},
  { "ConvertToYV411", BUILTIN_FUNC_PREFIX, "c[interlaced]b[matrix]s[ChromaInPlacement]s[chromaresample]s[ChromaOutPlacement]s[param1]f[param2]f[param3]f[bits]i[quality]b", ConvertToPlanarGeneric::CreateYUV411, (void*)0},
  { "ConvertToYUY2",  BUILTIN_FUNC_PREFIX, "c[interlaced]b[matrix]s[ChromaInPlacement]s[chromaresample]s[ChromaOutPlacement]s[param1]f[param2]f[param3]f[bits]i[quality]b", ConvertToPlanarGeneric::CreateConvertToYUY2 },
  { "ConvertBackToYUY2", BUILTIN_FUNC_PREFIX, "c[matrix]s", ConvertToPlanarGeneric::CreateConvertBackToYUY2 },
  { "ConvertToY",       BUILTIN_FUNC_PREFIX, "c[matrix]s[bits]i[quality]b", ConvertToPlanarGeneric::CreateY, (void*)1 }, // user_data == 1 -> any bit depth sources
  { "ConvertToYA",      BUILTIN_FUNC_PREFIX, "c[matrix]s[bits]i[quality]b", ConvertToPlanarGeneric::CreateY, (void*)2 }, // user_data == 2 -> any bit depth sources, force/keep alpha
  { "ConvertToYUV411",  BUILTIN_FUNC_PREFIX, "c[interlaced]b[matrix]s[ChromaInPlacement]s[chromaresample]s[ChromaOutPlacement]s[param1]f[param2]f[param3]f[bits]i[quality]b", ConvertToPlanarGeneric::CreateYUV411, (void*)1},
  { "ConvertToYUV440",  BUILTIN_FUNC_PREFIX, "c[interlaced]b[matrix]s[ChromaInPlacement]s[chromaresample]s[ChromaOutPlacement]s[param1]f[param2]f[param3]f[bits]i[quality]b", ConvertToPlanarGeneric::CreateYUV440, (void*)1},
  { "ConvertToYUV410",  BUILTIN_FUNC_PREFIX, "c[interlaced]b[matrix]s[ChromaInPlacement]s[chromaresample]s[ChromaOutPlacement]s[param1]f[param2]f[param3]f[bits]i[quality]b", ConvertToPlanarGeneric::CreateYUV410, (void*)1},
  { "ConvertToYUV420",  BUILTIN_FUNC_PREFIX, "c[interlaced]b[matrix]s[ChromaInPlacement]s[chromaresample]s[ChromaOutPlacement]s[param1]f[param2]f[param3]f[bits]i[quality]b", ConvertToPlanarGeneric::CreateYUV420, (void*)1},
  { "ConvertToYUV422",  BUILTIN_FUNC_PREFIX, "c[interlaced]b[matrix]s[ChromaInPlacement]s[chromaresample]s[ChromaOutPlacement]s[param1]f[param2]f[param3]f[bits]i[quality]b", ConvertToPlanarGeneric::CreateYUV422, (void*)1},
  { "ConvertToYUV444",  BUILTIN_FUNC_PREFIX, "c[interlaced]b[matrix]s[ChromaInPlacement]s[chromaresample]s[param1]f[param2]f[param3]f[bits]i[quality]b", ConvertToPlanarGeneric::CreateYUV444, (void*)1},
  { "ConvertToYUVA411", BUILTIN_FUNC_PREFIX, "c[interlaced]b[matrix]s[ChromaInPlacement]s[chromaresample]s[ChromaOutPlacement]s[param1]f[param2]f[param3]f[bits]i[quality]b", ConvertToPlanarGeneric::CreateYUV411, (void*)2},
  { "ConvertToYUVA440", BUILTIN_FUNC_PREFIX, "c[interlaced]b[matrix]s[ChromaInPlacement]s[chromaresample]s[ChromaOutPlacement]s[param1]f[param2]f[param3]f[bits]i[quality]b", ConvertToPlanarGeneric::CreateYUV440, (void*)2},
  { "ConvertToYUVA410", BUILTIN_FUNC_PREFIX, "c[interlaced]b[matrix]s[ChromaInPlacement]s[chromaresample]s[ChromaOutPlacement]s[param1]f[param2]f[param3]f[bits]i[quality]b", ConvertToPlanarGeneric::CreateYUV410, (void*)2},
  { "ConvertToYUVA420", BUILTIN_FUNC_PREFIX, "c[interlaced]b[matrix]s[ChromaInPlacement]s[chromaresample]s[ChromaOutPlacement]s[param1]f[param2]f[param3]f[bits]i[quality]b", ConvertToPlanarGeneric::CreateYUV420, (void*)2},
  { "ConvertToYUVA422", BUILTIN_FUNC_PREFIX, "c[interlaced]b[matrix]s[ChromaInPlacement]s[chromaresample]s[ChromaOutPlacement]s[param1]f[param2]f[param3]f[bits]i[quality]b", ConvertToPlanarGeneric::CreateYUV422, (void*)2},
  { "ConvertToYUVA444", BUILTIN_FUNC_PREFIX, "c[interlaced]b[matrix]s[ChromaInPlacement]s[chromaresample]s[param1]f[param2]f[param3]f[bits]i[quality]b", ConvertToPlanarGeneric::CreateYUV444, (void*)2},
  { "ConvertTo8bit",  BUILTIN_FUNC_PREFIX, "c[bits]i[truerange]b[dither]i[dither_bits]i[fulls]b[fulld]b", ConvertBits::Create, (void *)8 },
  { "ConvertTo16bit", BUILTIN_FUNC_PREFIX, "c[bits]i[truerange]b[dither]i[dither_bits]i[fulls]b[fulld]b", ConvertBits::Create, (void *)16 },
  { "ConvertToFloat", BUILTIN_FUNC_PREFIX, "c[bits]i[truerange]b[dither]i[dither_bits]i[fulls]b[fulld]b", ConvertBits::Create, (void *)32 },
  { "ConvertBits",    BUILTIN_FUNC_PREFIX, "c[bits]i[truerange]b[dither]i[dither_bits]i[fulls]b[fulld]b", ConvertBits::Create, (void *)0 },
  { "AddAlphaPlane",  BUILTIN_FUNC_PREFIX, "c[mask].[opacity]f", AddAlphaPlane::Create},
  { "RemoveAlphaPlane",  BUILTIN_FUNC_PREFIX, "c", RemoveAlphaPlane::Create},
  { 0 }
};


/****************************************
*******   Convert to RGB / RGBA   ******
***************************************/

// 0    1         2                 3                  4          5        6        7       8       9
// c[matrix]s[interlaced]b[ChromaInPlacement]s[chromaresample]s[param1]f[param2]f[param3]f[bits]i[quality]b

// Special syntax: adaptive packed target format: ConvertToRGB
AVSValue __cdecl CreateConvertToAdaptivePackedRGB(AVSValue args, void* user_data, IScriptEnvironment* env)
{
  const PClip clip = args[0].AsClip();
  const VideoInfo& vi = clip->GetVideoInfo();

  // bits= if given must be 8 or 16 (only valid packed RGB channel depths)
  int target_bits_per_channel;
  if (args[8].Defined()) {
    target_bits_per_channel = args[8].AsInt();
    if (target_bits_per_channel != 8 && target_bits_per_channel != 16)
      env->ThrowError("ConvertToRGB: bits must be 8 or 16 if specified. "
        "Use ConvertToPlanarRGB for other bit depths.");
  }
  else {
    const int src_bits = vi.BitsPerComponent();
    if (src_bits != 8 && src_bits != 16)
      env->ThrowError("ConvertToRGB: source bit depth must be 8 or 16. "
        "Use ConvertToPlanarRGB or ConvertBits(8)/ConvertBits(16) first.");
    target_bits_per_channel = src_bits;
  }

  // Decide target format based on source type and alpha capability (and the optional bits= override):
  // - YUV/YUY2 source: always alpha-capable (RGB32 or RGB64)
  // - RGB-like source: preserve alpha capability of source format
  //     RGB32/RGB64/PlanarRGBA -> RGB32/RGB64 (with alpha)
  //     RGB24/RGB48/PlanarRGB  -> RGB24/RGB48 (without alpha)
  const bool hasAlpha = vi.IsYUV() || vi.IsYUVA() // YUV always gets alpha target
    || vi.IsRGB32() || vi.IsRGB64()
    || vi.IsPlanarRGBA();

  // 8-bit: RGB24 or RGB32, 16-bit: RGB48 or RGB64
  const intptr_t target_rgbtype = (target_bits_per_channel == 8)
    ? (hasAlpha ? 32 : 24)
    : (hasAlpha ? 64 : 48);

  return CreateConvertToPackedRGB(args, reinterpret_cast<void*>(target_rgbtype), env);
}

// Registration names with implied fixed bit depth:
// ConvertToRGB24 -> 24-bit only  (bits must be 24 if specified)
// ConvertToRGB32 -> 32-bit only  (bits must be 32 if specified)
// ConvertToRGB48 -> 48-bit only  (bits must be 48 if specified)
// ConvertToRGB64 -> 64-bit only  (bits must be 64 if specified)
AVSValue __cdecl CreateConvertToPackedRGB(AVSValue args, void* user_data, IScriptEnvironment* env)
{
  const int target_bits = (int)reinterpret_cast<intptr_t>(user_data); // 24, 32, 48, 64
  const int target_bytes = target_bits / 8; // 3, 4, 6, 8
  const int bits_per_channel = (target_bytes <= 4) ? 8 : 16; // RGB24/32=8bit, RGB48/64=16bit

  // bits= must match the implied depth if specified
  if (args[8].Defined() && args[8].AsInt() != bits_per_channel)
    env->ThrowError("ConvertToRGB%d: bits must be %d if specified.",
      target_bits, bits_per_channel);

  // Rebuild args with bits forced to the correct value
  AVSValue new_args[10] = {
    args[0], args[1], args[2], args[3], args[4],
    args[5], args[6], args[7],
    AVSValue(bits_per_channel), // bits forced
    args[9]  // quality
  };
  AVSValue new_args_val(new_args, 10);
  return CreateConvertToRGB(new_args_val, user_data, env);
}

// Dispatch hub for all ConvertToRGB variants.
// Since 3.7.6 YUY2 is pre-converted to YV16 before dispatch, so no class is needed.
// Called from ConvertToRGB24/32/48/64 and ConvertToPlanarRGB(A) registrations,
// and from any internal env->Invoke("ConvertToRGB",...) calls.
// All ToRGB variants has bits and quality parameters, bit the quality=true and on-the-fly bit depth
// conversion logic is only implemented for YUV->PlanarRGB path.

// Note: when you add new parameters, find all places env->Invoke("ConvertToRGB",...)
// and check parameter count!
AVSValue __cdecl CreateConvertToRGB(AVSValue args, void* user_data, IScriptEnvironment* env)
{
  PClip clip = args[0].AsClip();
  VideoInfo vi = clip->GetVideoInfo();

  // Before any conversions, convert YUY2 to YV16
  if (vi.IsYUY2()) {
    clip = new ConvertYUY2ToYV16_or_Y(clip, false /*to_y*/, env);
    vi = clip->GetVideoInfo();
  }

  const char* const matrix_name = args[1].AsString(0);
  const bool haveOpts = args[3].Defined() || args[4].Defined();

  // common Create for all CreateRGB24/32/48/64/Planar(RGBP:-1, RGPAP:-2) using user_data
  int target_rgbtype = (int)reinterpret_cast<intptr_t>(user_data);
  // can be overridden later
  // -1,-2: Planar RGB(A)
  // 24,32,48,64: RGB24/32/48/64
  const bool original_target_is_packed = target_rgbtype > 0; // 24,32,48,64

  const int target_bits_per_pixel = args[8].AsInt(vi.BitsPerComponent());
  // for legacy packed formats it's 8 for target_rgbtype==24,32, and 16 for target_rgbtype==48,64
  // for planar RGB(A) target, it can be any bit depth supported by the source and the conversion logic (8,10,12,14,16,32)

  const bool quality = args[9].AsBool(false); // yuv-planarrgb float workflow

  // planar YUV-like source
  if (vi.IsPlanar() && (vi.IsYUV() || vi.IsYUVA())) {
    // here we keep the bit depth, later, the RGB conversion will take target_bits_per_pixel into account
    AVSValue new_args[10] = { clip, args[2], args[1], args[3], args[4], args[5], args[6], args[7], vi.BitsPerComponent(), quality };
    // conversion to planar or packed RGB is always from 444
    // clip, interlaced, matrix, chromainplacement, chromaresample, param1, param2, param3, bits, quality   Check for ConvertToYUV444 param list!!!! Count must match!
    clip = ConvertToPlanarGeneric::CreateYUV444(AVSValue(new_args, 10), (void*)1, env).AsClip(); // (void *)1: not restricted to 8 bits

    // planar RGB(A) target or packed needing intermediate: set finalBitdepth if conversion needed
    // also: when quality==true, we need to use the float workflow which requires a planar RGB(A) intermediate
    if (target_rgbtype > 0) {
      // packed RGB target: redirect to planar rgb(a) intermediate if bit-depth mismatch or quality mode
      const bool hasAlpha = (target_rgbtype == 32 || target_rgbtype == 64);
      if (vi.BitsPerComponent() != target_bits_per_pixel || quality)
        target_rgbtype = hasAlpha ? -2 : -1;
    }

    if (target_rgbtype == 48 || target_rgbtype == 64) {
      // Unlike parameter-less RGB24/32, 16 bit packed format have no direct conversions at all.
      // 1.) YUV->PlanarRGB first (recursive call),
      // 2.) then fall through to the planar RGB->packed RGB path below for the final repack
      // So instead of unoptimized code of YUV(A)444P16->RGB48/64 we convert to PlanarRGB(A) then to RGB48/64
      // Also: when quality=true or bit-depth conversion is needed, the planar RGB intermediate is required anyway
      AVSValue new_args2[10] = { clip, args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8]/*bits*/, args[9]/*quality*/ };

      const intptr_t target_planar_rgb_type =
        (target_rgbtype == 24 || target_rgbtype == 48) ? -1 : // no-alpha target: skip alpha even if source has it
        vi.IsYUVA() ? -2 :  // alpha target + alpha source: preserve it
        -1;   // alpha target + no alpha source: no alpha to copy
      // Note: target_rgbtype == 24 is currently never reached here (24/32 use direct conversion path),
      // but kept for when 24/32-bit packed is also redirected to the planar intermediate path in the future.

      clip = CreateConvertToRGB(AVSValue(new_args2, 10), (void*)target_planar_rgb_type, env).AsClip();
      vi = clip->GetVideoInfo(); // must update vi for fall-through
      // Fall through to planar RGB source path below to do the final pack
      // reuses the same PlanarRGBtoPackedRGB logic.
    }
    else {
      // Direct YUV444->RGB conversion path
      bool bitdepthConverted = false;
      const int finalBitdepth = (vi.BitsPerComponent() != target_bits_per_pixel) ?
        target_bits_per_pixel : -1; // -1 means: no need to convert

      // Optional bit-depth conversion in PackedRGBtoPlanarRGB.
      // Note: all formats are bit-depth-converted if finalBitdepth != -1
      // (historically there were cases and bit-depths which were not implemented to have in-line bit-depth conversion)
      // pass -1 as finalBitdepth, signing that no bit-depth conversion required

      const int rgbtype_param = (target_rgbtype == -1 || target_rgbtype == -2) ? target_rgbtype : // planar RGB(A)
        target_rgbtype == 24 ? 3 : 4; // RGB24 or RGB32 "pixel_step" parameter for legacy direct YUV444->RGB24/32 conversion

      clip = new ConvertYUV444ToRGB(clip, matrix_name, rgbtype_param,
        finalBitdepth, // -1 if no conversion needed
        quality, /*ref*/bitdepthConverted, env);
      vi = clip->GetVideoInfo();
      // When direct YUV-RGB24/32 conversion happens, the result is not planar!

      const bool needConvertFinalBitdepth = finalBitdepth != -1;

      if (needConvertFinalBitdepth && !bitdepthConverted) {
        AVSValue new_args[2] = { clip, finalBitdepth };
        // plain Invoke instead of "new ConvertBits", this detects and keeps source and target ranges.
        // Since no fulls and fulld is given, actual range detection happens per-frame in GetFrame
        clip = env->Invoke("ConvertBits", AVSValue(new_args, 2)).AsClip();
        vi = clip->GetVideoInfo();
      }

      // We can still have RGB24/32 here, YV24->24/32 is still valid
      if (original_target_is_packed && vi.IsPlanar()) {
        // from any planar rgb(a) -> rgb24/32/48/64
        // source here is always a 8/16bit planar RGB(A), but we used
        // planar intermediate.
        // finally it has to be converted to RGB24/32/48/64
        clip = new PlanarRGBtoPackedRGB(clip, target_rgbtype == -2);
        vi = clip->GetVideoInfo();
      }
      return clip;
    }
  } // end of YUV->RGB path
  else {
    if (haveOpts)
      env->ThrowError("ConvertToRGB: ChromaPlacement and ChromaResample options are not supported.");
  }

  // This part can fallthrough from the YUV path when target is packed RGB

  // RGB->RGB paths (planar or packed) do not support ChromaPlacement and ChromaResample options, as they are meaningless for RGB formats.

  // Planar RGB(A) source
  if (vi.IsPlanarRGB() || vi.IsPlanarRGBA())
  {
    // Handle alpha channel add/remove for planar-to-planar
    if (target_rgbtype < 0) {
      if (vi.IsPlanarRGB() && target_rgbtype == -2)
        clip = new AddAlphaPlane(clip, nullptr, 0.0f, false, env);
      else if (vi.IsPlanarRGBA() && target_rgbtype == -1)
        clip = new RemoveAlphaPlane(clip, env);
    }

    // Convert bit depth if needed
    if (vi.BitsPerComponent() != target_bits_per_pixel) {
      AVSValue args[2] = { clip, target_bits_per_pixel };
      // plain Invoke instead of "new ConvertBits", this detects and keeps source and target ranges.
      // Since no fulls and fulld is given, actual range detection happens per-frame in GetFrame
      clip = env->Invoke("ConvertBits", AVSValue(args, 2)).AsClip();
    }

    // Convert planar to packed
    if (target_rgbtype >= 0) {
      bool hasAlpha = (target_rgbtype == 32 || target_rgbtype == 64);
      clip = new PlanarRGBtoPackedRGB(clip, hasAlpha);
    }

    return clip;
  } // Planar RGB(A) source

  // YUV source is done
  // Planar RGB(A) source is done
  // Now remains packed RGB source
  // Conversions from packed RGB

  // Packed to Packed
  if (target_rgbtype >= 0) {
    // target bit depth (8-bit for 24/32, 16-bit for 48/64)
    // target_bits_per_pixel is same as target_rgbtype

    if (vi.BitsPerComponent() != target_bits_per_pixel) {
      AVSValue args[2] = { clip, target_bits_per_pixel };
      // plain Invoke instead of "new ConvertBits", this detects and keeps source and target ranges.
      // Since no fulls and fulld is given, actual range detection happens per-frame in GetFrame
      clip = env->Invoke("ConvertBits", AVSValue(args, 2)).AsClip();
      vi = clip->GetVideoInfo();
    }

    bool target_has_alpha = (target_rgbtype == 32 || target_rgbtype == 64);
    bool source_has_alpha = (vi.IsRGB32() || vi.IsRGB64());

    // between packed RGB types, alpha add/remove
    if (target_has_alpha && !source_has_alpha)
      return new RGBtoRGBA(clip);
    if (!target_has_alpha && source_has_alpha)
      return new RGBAtoRGB(clip);

    return clip;
  }

  // Packed to Planar
  // RGB24/32/48/64 ->
  const bool isSrcRGBA = vi.IsRGB32() || vi.IsRGB64();
  const bool isTargetRGBA = target_rgbtype == -2; // -2 planar RGBA, -1 planar RGB
  clip = new PackedRGBtoPlanarRGB(clip, isSrcRGBA, isTargetRGBA);
  vi = clip->GetVideoInfo(); // new format
  // no embedded bitdepth conversion in PackedRGBtoPlanarRGB
  if (target_bits_per_pixel != vi.BitsPerComponent()) {
    AVSValue new_args[2] = { clip, target_bits_per_pixel };
    clip = env->Invoke("ConvertBits", AVSValue(new_args, 2)).AsClip();
    vi = clip->GetVideoInfo(); // new format
  }

  return clip;
}


AVSValue AddAlphaPlane::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  bool isMaskDefined = args[1].Defined();
  bool isOpacityDefined = args[2].Defined();
  bool maskIsClip = false;

  // if mask is not defined and videoformat has Alpha then we return
  if (isMaskDefined && !args[1].IsClip() && !args[1].IsFloat())
    env->ThrowError("AddAlphaPlane: mask parameter should be clip or number");

  if (isOpacityDefined && !args[2].IsFloat())
    env->ThrowError("AddAlphaPlane: opacity parameter should be a number");

  if (isMaskDefined && isOpacityDefined)
    env->ThrowError("AddAlphaPlane: cannot specify both mask and opacity parameters");

  const VideoInfo& vi = args[0].AsClip()->GetVideoInfo();
  if (!isMaskDefined && !isOpacityDefined && (vi.IsPlanarRGBA() || vi.IsYUVA() || vi.IsRGB32() || vi.IsRGB64()))
    return args[0].AsClip();

  PClip alphaClip = nullptr;
  if (isMaskDefined && args[1].IsClip()) {
    const VideoInfo& viAlphaClip = args[1].AsClip()->GetVideoInfo();
    maskIsClip = true;
    if (viAlphaClip.BitsPerComponent() != vi.BitsPerComponent())
      env->ThrowError("AddAlphaPlane: alpha clip is of different bit depth");
    if (viAlphaClip.width != vi.width || viAlphaClip.height != vi.height)
      env->ThrowError("AddAlphaPlane: alpha clip is of different size");
    if (viAlphaClip.IsY())
      alphaClip = args[1].AsClip();
    else if (viAlphaClip.IsYUVA() || viAlphaClip.IsPlanarRGBA() || viAlphaClip.IsRGB32() || viAlphaClip.IsRGB64()) {
      // IsYUVA covers IsYA as well
      AVSValue new_args[1] = { args[1].AsClip() };
      alphaClip = env->Invoke("ExtractA", AVSValue(new_args, 1)).AsClip();
    }
    else {
      env->ThrowError("AddAlphaPlane: alpha clip should be greyscale or should have alpha plane");
    }
    // alphaClip is always greyscale here
  }

  float maskAsFloat = -1.0f;
  if (!maskIsClip) {
    if (isOpacityDefined) {
      // Handle opacity parameter (0.0 to 1.0)
      float opacity = args[2].AsFloatf(1.0f);
      if (std::isnan(opacity) || opacity < 0.0f || opacity > 1.0f)
        env->ThrowError("AddAlphaPlane: opacity must be between 0.0 and 1.0");
      if (vi.BitsPerComponent() <= 16) {
        int max_pixel_value = (1 << vi.BitsPerComponent()) - 1;
        maskAsFloat = opacity * max_pixel_value;
      }
      else {
        maskAsFloat = opacity;
      }
    }
    else {
      // Handle mask parameter (direct value)
      maskAsFloat = (float)args[1].AsFloat(-1.0f);
      if (std::isnan(maskAsFloat))
        env->ThrowError("AddAlphaPlane: mask must be a number or a clip, not NaN");
    }
  }

  if (vi.IsRGB24()) {
    AVSValue new_args[1] = { args[0].AsClip() };
    PClip child = env->Invoke("ConvertToRGB32", AVSValue(new_args, 1)).AsClip();
    return new AddAlphaPlane(child, alphaClip, maskAsFloat, isMaskDefined || isOpacityDefined, env);
  }
  else if (vi.IsRGB48()) {
    AVSValue new_args[1] = { args[0].AsClip() };
    PClip child = env->Invoke("ConvertToRGB64", AVSValue(new_args, 1)).AsClip();
    return new AddAlphaPlane(child, alphaClip, maskAsFloat, isMaskDefined || isOpacityDefined, env);
  }
  return new AddAlphaPlane(args[0].AsClip(), alphaClip, maskAsFloat, isMaskDefined || isOpacityDefined, env);
}

AddAlphaPlane::AddAlphaPlane(PClip _child, PClip _alphaClip, float _mask_f, bool isMaskDefined, IScriptEnvironment* env)
  : GenericVideoFilter(_child), alphaClip(_alphaClip)
  , mask(0), mask_f(0.0f), pixelsize(0), bits_per_pixel(0)
{
  if(vi.IsYUY2())
    env->ThrowError("AddAlphaPlane: YUY2 is not allowed");
  if(!vi.IsYUV() && !vi.IsYUVA() && !vi.IsRGB())
    env->ThrowError("AddAlphaPlane: format not supported");

  pixelsize = vi.ComponentSize();
  bits_per_pixel = vi.BitsPerComponent();

  if (vi.IsYUV()) {
    // Y (greyscale) becomes YA: CS_GENERIC_Y and CS_GENERIC_YA differ only in CS_YUV/CS_YUVA
    int pixel_type = vi.pixel_type;
    if (vi.IsYV12())
      pixel_type = VideoInfo::CS_YV12;
    int new_pixel_type = (pixel_type & ~VideoInfo::CS_YUV) | VideoInfo::CS_YUVA;
    vi.pixel_type = new_pixel_type;
  } else if(vi.IsPlanarRGB()) {
    int pixel_type = vi.pixel_type;
    int new_pixel_type = (pixel_type & ~VideoInfo::CS_RGB_TYPE) | VideoInfo::CS_RGBA_TYPE;
    vi.pixel_type = new_pixel_type;
  }
  // RGB24 and RGB48 already converted to 32/64
  // RGB32, RGB64, YUVA and RGBA: no change

  // mask parameter. If none->max opacity

  if (!alphaClip) {
    int max_pixel_value = bits_per_pixel <= 16 ? (1 << bits_per_pixel) - 1 : 0; // n/a for float
    if (!isMaskDefined) {
      mask_f = 1.0f;
      mask = max_pixel_value;
    }
    else {
      mask_f = _mask_f;
      mask = (mask_f < 0) ? 0 : (mask_f > max_pixel_value) ? max_pixel_value : (int)(mask_f + 0.5f);
      mask = clamp(mask, 0, max_pixel_value);
      // no clamp for float
    }
  }
}

PVideoFrame AddAlphaPlane::GetFrame(int n, IScriptEnvironment* env)
{
  PVideoFrame src = child->GetFrame(n, env);
  PVideoFrame dst = env->NewVideoFrameP(vi, &src);
  if(vi.IsPlanar())
  {
    int planes_y[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
    int planes_r[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
    int *planes = (vi.IsYUV() || vi.IsYUVA()) ? planes_y : planes_r;
    // copy existing color planes: 1 for YA (from Y), 3 otherwise
    const int num_color_planes = vi.IsYA() ? 1 : 3;
    for (int p = 0; p < num_color_planes; ++p) {
      const int plane = planes[p];
      env->BitBlt(dst->GetWritePtr(plane), dst->GetPitch(plane), src->GetReadPtr(plane),
           src->GetPitch(plane), src->GetRowSize(plane), src->GetHeight(plane));
    }
  } else {
    // Packed RGB, already converted to RGB32 or RGB64
    env->BitBlt(dst->GetWritePtr(), dst->GetPitch(), src->GetReadPtr(),
      src->GetPitch(), src->GetRowSize(), src->GetHeight());
  }

  if (vi.IsPlanarRGBA() || vi.IsYUVA()) {
    if (alphaClip) {
      PVideoFrame srcAlpha = alphaClip->GetFrame(n, env);
      env->BitBlt(dst->GetWritePtr(PLANAR_A), dst->GetPitch(PLANAR_A), srcAlpha->GetReadPtr(PLANAR_Y),
        srcAlpha->GetPitch(PLANAR_Y), srcAlpha->GetRowSize(PLANAR_Y), srcAlpha->GetHeight(PLANAR_Y));
    }
    else {
      // default constant
      const int rowsizeA = dst->GetRowSize(PLANAR_A);
      const int dst_pitchA = dst->GetPitch(PLANAR_A);
      BYTE* dstp_a = dst->GetWritePtr(PLANAR_A);
      const int heightA = dst->GetHeight(PLANAR_A);

      switch (vi.ComponentSize())
      {
      case 1:
        fill_plane<BYTE>(dstp_a, heightA, rowsizeA, dst_pitchA, mask);
        break;
      case 2:
        fill_plane<uint16_t>(dstp_a, heightA, rowsizeA, dst_pitchA, mask);
        break;
      case 4:
        fill_plane<float>(dstp_a, heightA, rowsizeA, dst_pitchA, mask_f);
        break;
      }
    }
    return dst;
  }
  // RGB32 and RGB64

  BYTE* pf = dst->GetWritePtr();
  int pitch = dst->GetPitch();
  int rowsize = dst->GetRowSize();
  int height = dst->GetHeight();
  int width = vi.width;

  if (alphaClip) {
    // fill by alpha clip already converted to grey-only
    PVideoFrame srcAlpha = alphaClip->GetFrame(n, env);
    const BYTE* srcp_a = srcAlpha->GetReadPtr(PLANAR_Y);
    size_t pitch_a = srcAlpha->GetPitch(PLANAR_Y);

    pf += pitch * (vi.height - 1); // start from bottom: packed RGB is upside down

    if (vi.IsRGB32()) {
      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x ++) {
          pf[x*4+3] = srcp_a[x];
        }
        pf -= pitch; // packed RGB is upside down
        srcp_a += pitch_a;
      }
    }
    else if (vi.IsRGB64()) {
      rowsize /= sizeof(uint16_t);
      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x ++) {
          reinterpret_cast<uint16_t *>(pf)[x*4+3] = reinterpret_cast<const uint16_t *>(srcp_a)[x];
        }
        pf -= pitch; // packed RGB is upside down
        srcp_a += pitch_a;
      }
    }
  }
  else {
    // fill with constant
    if (vi.IsRGB32()) {
      for (int y = 0; y < height; y++) {
        for (int x = 3; x < rowsize; x += 4) {
          pf[x] = mask;
        }
        pf += pitch;
      }
    }
    else if (vi.IsRGB64()) {
      rowsize /= sizeof(uint16_t);
      for (int y = 0; y < height; y++) {
        for (int x = 3; x < rowsize; x += 4) {
          reinterpret_cast<uint16_t *>(pf)[x] = mask;
        }
        pf += pitch;
      }
    }
  }

  return dst;
}

AVSValue RemoveAlphaPlane::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  // if videoformat has no Alpha then we return
  const VideoInfo& vi = args[0].AsClip()->GetVideoInfo();
  if(vi.IsPlanar() && (vi.IsYUV() || vi.IsPlanarRGB())) // planar and no alpha
    return args[0].AsClip();
  if (vi.IsYUY2()) // YUY2: no alpha
    return args[0].AsClip();
  if(vi.IsRGB24() || vi.IsRGB48()) // packed RGB and no alpha
    return args[0].AsClip();
  if (vi.IsRGB32()) {
    AVSValue new_args[1] = { args[0].AsClip() };
    return env->Invoke("ConvertToRGB24", AVSValue(new_args, 1)).AsClip();
  }
  if (vi.IsRGB64()) {
    AVSValue new_args[1] = { args[0].AsClip() };
    return env->Invoke("ConvertToRGB48", AVSValue(new_args, 1)).AsClip();
  }
  return new RemoveAlphaPlane(args[0].AsClip(), env);
}

RemoveAlphaPlane::RemoveAlphaPlane(PClip _child, IScriptEnvironment* env)
  : GenericVideoFilter(_child)
{
  if(vi.IsYUY2())
    env->ThrowError("RemoveAlphaPlane: YUY2 is not allowed");

  if (vi.IsYUVA()) {
    // YA becomes Y: CS_GENERIC_YA and CS_GENERIC_Y differ only in CS_YUVA/CS_YUV
    int pixel_type = vi.pixel_type;
    int new_pixel_type = (pixel_type & ~VideoInfo::CS_YUVA) | VideoInfo::CS_YUV;
    vi.pixel_type = new_pixel_type;
  } else if(vi.IsPlanarRGBA()) {
    int pixel_type = vi.pixel_type;
    int new_pixel_type = (pixel_type & ~VideoInfo::CS_RGBA_TYPE) | VideoInfo::CS_RGB_TYPE;
    vi.pixel_type = new_pixel_type;
  }
}

PVideoFrame RemoveAlphaPlane::GetFrame(int n, IScriptEnvironment* env)
{
  PVideoFrame src = child->GetFrame(n, env);
  // Packed RGB: already handled in ::Create through Invoke 32->24 or 64->48 conversion
  // only planar here
  if (vi.IsY()) // from YA (or Y passthrough): keep the Y plane only
    return env->Subframe(src, 0, src->GetPitch(PLANAR_Y), src->GetRowSize(PLANAR_Y), src->GetHeight(PLANAR_Y));
  int planes_y[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
  int planes_r[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
  int *planes = (vi.IsYUV() || vi.IsYUVA()) ? planes_y : planes_r;
  // Abuse Subframe to snatch the YUV/GBR planes
  return env->SubframePlanar(src, 0, src->GetPitch(planes[0]), src->GetRowSize(planes[0]), src->GetHeight(planes[0]), 0, 0, src->GetPitch(planes[1]));

#if 0
  // BitBlt version. Kept for reference
  PVideoFrame dst = env->NewVideoFrameP(vi, &src);
  // copy 3 planes w/o alpha
  for (int p = 0; p < 3; ++p) {
    const int plane = planes[p];
    env->BitBlt(dst->GetWritePtr(plane), dst->GetPitch(plane), src->GetReadPtr(plane),
      src->GetPitch(plane), src->GetRowSize(plane), src->GetHeight(plane));
  }
return dst;
#endif
}

