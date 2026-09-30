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

// Overlay (c) 2003, 2004 by Klaus Post
#include <avisynth.h>
#ifdef AVS_WINDOWS
    #include <avs/win.h>
#else
    #include <avs/posix.h>
#endif

#include <stdlib.h>
#include "overlay.h"
#include <string>
#include "../core/internal.h"
#include "../../convert/convert_helper.h" // chroma location helpers

// Chroma placement helpers

// Chroma location of an overlay/mask clip
// - _ChromaLocation frame prop (frame 0) if present.
// - Otherwise
//   - having the same chroma subsampling as the base clip, it is assumed to be sited
//     like the base (parameter baseChromaLocation)
//   - if not, it gets its own format's default (similar to ConvertToYUV4xx behavior).
static int getClipChromaLocation(PClip clip, const VideoInfo& baseVi, int baseChromaLocation, IScriptEnvironment* env) {
  int chromaloc;
  if (GetChromaLocationFromProps(clip, env, chromaloc))
    return chromaloc;
  const VideoInfo& vi = clip->GetVideoInfo();
  if (IsSubsampledYUV(vi) && IsSubsampledYUV(baseVi) &&
    vi.GetPlaneWidthSubsampling(PLANAR_U) == baseVi.GetPlaneWidthSubsampling(PLANAR_U) &&
    vi.GetPlaneHeightSubsampling(PLANAR_U) == baseVi.GetPlaneHeightSubsampling(PLANAR_U))
    return baseChromaLocation;
  return GetDefaultChromaLocation(vi);
}

// Obtain ConvertToYUV4xx function names by vi.
// nullptr for non-subsampled formats (Y, 4:4:4, RGB).
static const char* subsampledConvertName(const VideoInfo& vi) {
  if (vi.Is420()) return "ConvertToYUV420";
  if (vi.Is422()) return "ConvertToYUV422";
  if (vi.Is411()) return "ConvertToYUV411";
  if (vi.Is440()) return "ConvertToYUV440";
  if (vi.Is410()) return "ConvertToYUV410";
  return nullptr;
}

// Converts an overlay or color mask clip (Y, YUV(A)) to the subsampled native working
// format. ChromaInPlacement is the clip's own siting (getClipChromaLocation),
// ChromaOutPlacement is the base clip's `placement`. Also converts when the subsampling
// already matches but the effective siting differs, e.g. YV12 'center' -> 'left';
// equivalent sitings are left alone, e.g. YV16 'top' vs 'center' (same horizontal position).
static PClip convertToSubsampledWorkingFormat(PClip clip, const VideoInfo& workingVi, const VideoInfo& baseVi, int chromaLocation, IScriptEnvironment* env) {
  const VideoInfo& vi = clip->GetVideoInfo();
  const char* convertName = subsampledConvertName(workingVi);
  const char* clipConvertName = subsampledConvertName(vi);
  const int clipChromaLocation = getClipChromaLocation(clip, baseVi, chromaLocation, env);
  if (clipConvertName && !strcmp(clipConvertName, convertName) &&
    IsSameChromaSiting(clipChromaLocation, chromaLocation,
      1 << workingVi.GetPlaneWidthSubsampling(PLANAR_U), 1 << workingVi.GetPlaneHeightSubsampling(PLANAR_U)))
    return clip;
  // c[interlaced]b[matrix]s[ChromaInPlacement]s[chromaresample]s[ChromaOutPlacement]s
  AVSValue new_args[6] = { clip, false, AVSValue(),
    clipConvertName ? AVSValue(GetChromaLocationName(clipChromaLocation)) : AVSValue(), // n/a for Y, 4:4:4
    AVSValue(),
    GetChromaLocationName(chromaLocation) };
  return env->Invoke(convertName, AVSValue(new_args, 6)).AsClip();
}

/********************************************************************
***** Declare index of new filters for Avisynth's filter engine *****
********************************************************************/

extern const AVSFunction Overlay_filters[] = {
  { "Overlay", BUILTIN_FUNC_PREFIX, "cc[x]i[y]i[mask]c[opacity]f[mode]s[greymask]b[output]s[ignore_conditional]b[PC_Range]b[use444]b[condvarsuffix]s[placement]s", Overlay::Create },
    // 0, src clip
    // 1, overlay clip
    // 2, x
    // 3, y
    // 4, mask clip
    // 5, overlay opacity.(0.0->1.0)
    // 6, mode string, "blend", "add"
    // 7, greymask bool - true = only use luma information for mask
    // 8, output type, string
    // 9, ignore conditional variabels
    // 10, full YUV range.
    // 11, ignore 4:4:4 conversion
    // 12, conditional variable suffix AVS+
    // 13, chroma placement of the base clip, ConvertToYUV4xx syntax ("left"/"mpeg2", "center"/"mpeg1", "top_left", ...)
  { 0 }
};

enum {
  ARG_SRC = 0,
  ARG_OVERLAY = 1,
  ARG_X = 2,
  ARG_Y = 3,
  ARG_MASK = 4,
  ARG_OPACITY = 5,
  ARG_MODE = 6,
  ARG_GREYMASK = 7,
  ARG_OUTPUT = 8,
  ARG_IGNORE_CONDITIONAL = 9,
  ARG_FULL_RANGE = 10,
  ARG_USE444 = 11, // 170103 possible conversionless option experimental
  ARG_CONDVARSUFFIX = 12, // 190408
  ARG_PLACEMENT = 13
};

static int getPixelTypeWithoutAlpha(VideoInfo& vi)
{
  return (vi.IsYUVA() ? (vi.pixel_type & ~VideoInfo::CS_YUVA) | VideoInfo::CS_YUV :
    (vi.IsPlanarRGBA() ? (vi.pixel_type & ~VideoInfo::CS_RGBA_TYPE) | VideoInfo::CS_RGB_TYPE : vi.pixel_type));
}

Overlay::Overlay(PClip _child, AVSValue args, IScriptEnvironment *env) :
GenericVideoFilter(_child), child444(nullptr) {

  // child here is always planar: no packed RGB or YUY2 allowed

  full_range = args[ARG_FULL_RANGE].AsBool(false);  // Maintain CCIR601 range when converting to/from RGB.
  bool use444_defined = args[ARG_USE444].Defined();
  use444 = args[ARG_USE444].AsBool(true);  // avs+ option to use 444-conversionless mode
  name = args[ARG_MODE].AsString("Blend");
  condVarSuffix = args[ARG_CONDVARSUFFIX].AsString("");
  // `placement`: the chroma location (ChromaLocation_e) of the base clip, and thus of the output.
  // Same treatment like ConvertToYUV4xx's ChromaInPlacement.
  chromaLocation = ResolveChromaLocation(child, args[ARG_PLACEMENT].AsString(nullptr), env, &chromaLocationDefined);

  // Make copy of the VideoInfo
  inputVi = vi;
  // by default outputVi is the same as inputVi
  outputVi = vi;
  viInternalWorkingFormat = vi;

  opacity_f = (float)args[ARG_OPACITY].AsDblDef(1.0); // for float support
  opacity = (int)(256.0*opacity_f + 0.5); // range is converted to 256 for all all bit_depth
  offset_x = args[ARG_X].AsInt(0);
  offset_y = args[ARG_Y].AsInt(0);

  if (!args[ARG_OVERLAY].IsClip())
    env->ThrowError("Overlay: Overlay parameter is not a clip");

  overlay = args[ARG_OVERLAY].AsClip();

  overlayVi = overlay->GetVideoInfo();
  // overlay clip to always planar
  if (overlayVi.IsRGB() && !overlayVi.IsPlanar()) {
    // no packed RGB allowed from now on, autoconvert from packed
    AVSValue new_args[1] = { overlay };
    if (overlayVi.IsRGB24() || overlayVi.IsRGB48())
      overlay = env->Invoke("ConvertToPlanarRGB", AVSValue(new_args, 1)).AsClip();
    else
      overlay = env->Invoke("ConvertToPlanarRGBA", AVSValue(new_args, 1)).AsClip();
    overlayVi = overlay->GetVideoInfo();
  }
  else if (overlayVi.IsYUY2()) {
    // convert YUY2 to 422, keep internals simple
    AVSValue new_args[2] = { overlay, false };
    overlay = env->Invoke("ConvertToYUV422", AVSValue(new_args, 2)).AsClip();
    overlayVi = overlay->GetVideoInfo();
  }

  SetOfModeByName(name, env); // setting of_mode, checks valid mode strings as well

  viInternalOverlayWorkingFormat = overlayVi;

  greymask = args[ARG_GREYMASK].AsBool(true);  // Grey mask, default true
  ignore_conditional = args[ARG_IGNORE_CONDITIONAL].AsBool(false);  // Don't ignore conditionals by default

  mask = nullptr;
  if (args[ARG_MASK].Defined()) {  // Mask defined
    mask = args[ARG_MASK].AsClip();
    maskVi = mask->GetVideoInfo();
    if (maskVi.width!=overlayVi.width) {
      env->ThrowError("Overlay: Mask and overlay must have the same image size! (Width is not the same)");
    }
    if (maskVi.height!=overlayVi.height) {
      env->ThrowError("Overlay: Mask and overlay must have the same image size! (Height is not the same)");
    }
    if (maskVi.BitsPerComponent() != overlayVi.BitsPerComponent()) {
      env->ThrowError("Overlay: Mask and overlay must have the same bit depths!");
    }
  }

  pixelsize = vi.ComponentSize();
  bits_per_pixel = vi.BitsPerComponent();

  if (bits_per_pixel != overlayVi.BitsPerComponent()) {
    env->ThrowError("Overlay: input and overlay clip must have the same bit depths!");
  }

  // already filled vi = child->GetVideoInfo();
  // parse and check output format override vi
  output_pixel_format_override = args[ARG_OUTPUT].AsString(nullptr);
  if(output_pixel_format_override) {
    int output_pixel_type = GetPixelTypeFromName(output_pixel_format_override);
    if(output_pixel_type == VideoInfo::CS_UNKNOWN)
      env->ThrowError("Overlay: invalid pixel_type!");

    outputVi.pixel_type = output_pixel_type; // override output pixel format
    if(outputVi.BitsPerComponent() != inputVi.BitsPerComponent())
      env->ThrowError("Overlay: output bitdepth should be the same as input's!");
  }

  if (bits_per_pixel == 32) {
    if (_stricmp(name, "Blend") != 0 && _stricmp(name, "Luma") != 0 && _stricmp(name, "Chroma") != 0
      && _stricmp(name, "Add") != 0 && _stricmp(name, "Subtract") != 0) {
      env->ThrowError("Overlay: only Blend, Luma, Chroma, Add and Subtract modes are supported for 32-bit float video");
    }
  }

  if (vi.Is444())
    use444 = true; // 444 is conversionless by default

  // let use444=false to go live for subfilters that are ready to use it
  // except: RGB must be converted to 444 for Luma and Chroma operation
  if (vi.IsRGB() && (_stricmp(name, "Luma") == 0 || _stricmp(name, "Chroma") == 0)) {
    if (use444_defined && !use444) {
      env->ThrowError("Overlay: for RGB you cannot specify use444=false for overlay mode: %s", name);
    }
    use444 = true;
  }
  else if (!use444_defined &&
    (vi.IsY() || vi.Is420() || vi.Is422() || vi.Is411() || vi.Is440() || vi.Is410() || vi.IsRGB()) &&
    (_stricmp(name, "Blend") == 0 || _stricmp(name, "Luma") == 0 || _stricmp(name, "Chroma") == 0))
  {
    use444 = false; // default false for modes capable handling of use444==false, and valid formats
  }
  else if (!use444_defined &&
    (vi.IsRGB()) &&
    (_stricmp(name, "Add") == 0 || _stricmp(name, "Subtract") == 0))
  {
    use444 = false; // native RGB support for Add/Subtract
  }

  if (!use444) {
    // check if we can work in conversionless mode
    // 1.) colorspace is greyscale, 4:2:0, 4:2:2, 4:1:1, 4:4:0, 4:1:0 or any RGB
    // 2.) mode is "blend-like" (at the moment)
    // 4:1:1/4:4:0/4:1:0: blend kernels dispatch MASK411/MASK440/MASK410(_TOPLEFT) natively;
    // isInternal411/440/410 + ConvertToYUV411/440/410 (already bit-depth-agnostic) provide the
    // shape-matching that isInternal420/isInternal422 provide for their formats.
    if (!vi.IsY() && !vi.Is420() && !vi.Is422() && !vi.Is411() && !vi.Is440() && !vi.Is410() && !vi.IsRGB())
      env->ThrowError("Overlay: use444=false is allowed only for greyscale, 4:2:0, 4:2:2, 4:1:1, 4:4:0, 4:1:0 or any RGB video formats");
    //if (output_pixel_format_override && outputVi->pixel_type != vi.pixel_type)
    //  env->ThrowError("Overlay: use444=false is allowed only when no output pixel format is specified");
    if (_stricmp(name, "Blend") != 0 && _stricmp(name, "Luma") != 0 && _stricmp(name, "Chroma") != 0 &&
      (_stricmp(name, "Add") != 0 && _stricmp(name, "Subtract") != 0) && !vi.IsRGB())
      env->ThrowError("Overlay: cannot specify use444=false for this overlay mode: %s", name);
  }

  bool hasAlpha = vi.IsYUVA() || vi.IsPlanarRGBA();

  // Bit-mask arithmetic  CS_GENERIC_xxx | CS_Sample_Bits_N
  int new_bitdepth_bits;
  switch (bits_per_pixel) {
  case 8:  new_bitdepth_bits = VideoInfo::CS_Sample_Bits_8;  break;
  case 10: new_bitdepth_bits = VideoInfo::CS_Sample_Bits_10; break;
  case 12: new_bitdepth_bits = VideoInfo::CS_Sample_Bits_12; break;
  case 14: new_bitdepth_bits = VideoInfo::CS_Sample_Bits_14; break;
  case 16: new_bitdepth_bits = VideoInfo::CS_Sample_Bits_16; break;
  case 32: new_bitdepth_bits = VideoInfo::CS_Sample_Bits_32; break;
  default:
    env->ThrowError("Overlay: unsupported bit depth (%d)", bits_per_pixel);
    new_bitdepth_bits = 0; // unreachable
  }

  // set internal working format
  if (use444) {
    // we convert everything to 4:4:4
    viInternalWorkingFormat.pixel_type =
      (hasAlpha ? VideoInfo::CS_GENERIC_YUVA444 : VideoInfo::CS_GENERIC_YUV444) | new_bitdepth_bits;
  }
  else {
    // keep input format for internal format. Always 4:2:0, 4:2:2 (or 4:4:4 / Planar RGB)
    // filters have to prepare to work for these formats
    viInternalWorkingFormat = vi;
  }

  viInternalOverlayWorkingFormat.pixel_type = viInternalWorkingFormat.pixel_type;

  // Set GetFrame's real output format.
  if (outputVi.IsYUY2())
  {
    // on-the-fly fast conversion at the end of GetFrame
    vi.pixel_type = VideoInfo::CS_YUY2;
  } else {
    vi.pixel_type = viInternalWorkingFormat.pixel_type;
    // Y,420,422,444,PlanarRGB (and packed RGB converted to any intermediate)
  }

  // internal working formats:
  // - subsampled planar: 420, 422, 411, 440, 410
  // - full info: 444, planarRGB(A)
  // - full info 1 plane: greyscale
  isInternalRGB = viInternalWorkingFormat.IsRGB(); // must be planar rgb
  isInternalGrey = viInternalWorkingFormat.IsY();
  isInternal444 = viInternalWorkingFormat.Is444();
  isInternal422 = viInternalWorkingFormat.Is422();
  isInternal420 = viInternalWorkingFormat.Is420();
  isInternal411 = viInternalWorkingFormat.Is411();
  isInternal440 = viInternalWorkingFormat.Is440();
  isInternal410 = viInternalWorkingFormat.Is410();

  // narrow down the mask downsampling kernel variant for the subsampled native working formats
  placement = chromaLocationToMaskPlacement(chromaLocation, viInternalWorkingFormat);

  // Base clip conversion to internal 444 working format, done once here at
  // construction, uniformly for _every_ source formats (Y, RGB, 4:2:0, 4:2:2,
  // 4:1:1, 4:4:0, 4:1:0). Always goes through a real resampler (ConvertToYUV444)
  // with ChromaInPlacement pinned to this filter's own `placement`.
  // Pre 3.7.6: 420/422 had special, placement-unaware, Convert444FromYV12/16 and its
  // counterpart converters (point-replication in, box-average out), introducing
  // a chroma shift on round trip.
  if (inputVi.pixel_type != viInternalWorkingFormat.pixel_type &&
    isInternal444)
  {
    if (inputVi.IsRGB()) {
      AVSValue new_args[3] = { child, false, full_range ? "PC.601" : "rec601" }; // no chroma placement for RGB source
      child444 = env->Invoke("ConvertToYUV444", AVSValue(new_args, 3)).AsClip();
    }
    else {
      // Y, 4:2:0, 4:2:2, 4:1:1, 4:4:0, 4:1:0
      AVSValue new_args[4] = { child, false, AVSValue() /*matrix, unused for YUV->YUV444*/, GetChromaLocationName(chromaLocation) };
      child444 = env->Invoke("ConvertToYUV444", AVSValue(new_args, 4)).AsClip();
    }
  }
  // more format match of overlay
  if (overlayVi.IsRGB()) {
    if (isInternalGrey) {
      AVSValue new_args[2] = { overlay, full_range ? "PC.601" : "rec601" };
      overlay = env->Invoke("ConvertToY", AVSValue(new_args, 2)).AsClip();
      overlayVi = overlay->GetVideoInfo();
    }
    else if (!isInternalRGB) {
      AVSValue new_args[3] = { overlay, false, full_range ? "PC.601" : "rec601" };
      overlay = env->Invoke("ConvertToYUV444", AVSValue(new_args, 3)).AsClip();
      overlayVi = overlay->GetVideoInfo();
    }
  }
  if (isInternal444) {
    if (!overlayVi.Is444()) {
      // ChromaInPlacement: overlay's own siting, see getClipChromaLocation (n/a for Y)
      AVSValue new_args[4] = { overlay, false, AVSValue() /*matrix, unused for YUV->YUV444*/,
        GetChromaLocationName(getClipChromaLocation(overlay, inputVi, chromaLocation, env)) };
      overlay = env->Invoke("ConvertToYUV444", AVSValue(new_args, 4)).AsClip();
      overlayVi = overlay->GetVideoInfo();
    }
  }
  else if (isInternalRGB) {
    if (!overlayVi.IsPlanarRGB() && !overlayVi.IsPlanarRGBA()) {
      AVSValue new_args[3] = { overlay, full_range ? "PC.601" : "rec601", false};
      if (overlayVi.IsYUVA())
        overlay = env->Invoke("ConvertToPlanarRGBA", AVSValue(new_args, 3)).AsClip();
      else
        overlay = env->Invoke("ConvertToPlanarRGB", AVSValue(new_args, 3)).AsClip();
      overlayVi = overlay->GetVideoInfo();
    }
  }
  else if (isInternal420 || isInternal422 || isInternal411 || isInternal440 || isInternal410) {
    // in: overlay's own siting, out: `placement`, like the base clip
    overlay = convertToSubsampledWorkingFormat(overlay, viInternalWorkingFormat, inputVi, chromaLocation, env);
    overlayVi = overlay->GetVideoInfo();
  }

  if (mask) {
    if (maskVi.IsY() || isInternalGrey)
      greymask = true;

    // mask to always planar
    if (maskVi.IsRGB() && !maskVi.IsPlanar()) {
      // no packed RGB mask allowed from now on, autoconvert from packed
      AVSValue new_args[1] = { mask };
      if (maskVi.IsRGB24() || maskVi.IsRGB48())
        mask = env->Invoke("ConvertToPlanarRGB", AVSValue(new_args, 1)).AsClip();
      else
        mask = env->Invoke("ConvertToPlanarRGBA", AVSValue(new_args, 1)).AsClip();
      maskVi = mask->GetVideoInfo();
    }
    else if (maskVi.IsYUY2()) {
      // convert YUY2 mask to 422, keep internals simple
      AVSValue new_args[2] = { mask, false };
      mask = env->Invoke("ConvertToYUV422", AVSValue(new_args, 2)).AsClip();
      maskVi = mask->GetVideoInfo();
    }

    if (maskVi.IsRGB()) {
      if (greymask) {
        // Compatibility: ExtractB. See notes above.
        // This is good, because in the old times rgb32clip.ShowAlpha() was
        // used for feeding mask to overlay, that spreads alpha to all channels, so classic avisynth chose
        // B channel for source. (=R=G)
        // So we are not using greyscale conversion here at mask for compatibility reasons
        // Still: recommended usage for mask: rgbclip.ExtractA()
        /*
        AVSValue new_args[2] = { mask, (full_range) ? "PC.601" : "rec601" };
        mask2 = env->Invoke("ConvertToY", AVSValue(new_args, 2)).AsClip();
        */
        AVSValue new_args[1] = { mask };
        mask = env->Invoke("ExtractB", AVSValue(new_args, 1)).AsClip();
        maskVi = mask->GetVideoInfo();
      }
      else {
        if (!isInternalRGB) {
          AVSValue new_args[3] = { mask, false, full_range ? "PC.601" : "rec601" };
          mask = env->Invoke("ConvertToYUV444", AVSValue(new_args, 3)).AsClip();
          maskVi = mask->GetVideoInfo();
        }
      }
    } // RGB mask cases end

    const bool isInternalSubsampled = isInternal420 || isInternal422 || isInternal411 || isInternal440 || isInternal410;
    // A mask of the same subsampled format may still need re-siting, so it is checked too:
    // convertToSubsampledWorkingFormat returns it as is if the siting already matches
    if (getPixelTypeWithoutAlpha(maskVi) != getPixelTypeWithoutAlpha(viInternalWorkingFormat)
      || (!greymask && isInternalSubsampled))
    {
      if (!maskVi.IsRGB()) {
        if (isInternalRGB) {
          if (!greymask) {
            // mask clip was not RGB, but our internal format is.
            // convert from any (Y, YUV) mask format to planar RGB
            AVSValue new_args[3] = { mask, full_range ? "PC.601" : "rec601", false };
            mask = env->Invoke("ConvertToPlanarRGB", AVSValue(new_args, 3)).AsClip();
          }
        }
        else {
          if (greymask) {
            AVSValue new_args[1] = { mask }; // mask is not rgb here, no matrix param
            mask = env->Invoke("ConvertToY", AVSValue(new_args, 1)).AsClip();
          }
          else {
            if (isInternalSubsampled) {
              // in: mask's own siting, out: `placement`, like the base clip
              // returns the mask as is (no conversion) if format and effective siting already match
              mask = convertToSubsampledWorkingFormat(mask, viInternalWorkingFormat, inputVi, chromaLocation, env);
            }
            else if (isInternal444) {
              if (!maskVi.Is444()) {
                // ChromaInPlacement: mask's own siting, see getClipChromaLocation (n/a for Y)
                AVSValue new_args[4] = { mask, false, AVSValue() /*matrix, unused for YUV->YUV444*/,
                  GetChromaLocationName(getClipChromaLocation(mask, inputVi, chromaLocation, env)) };
                mask = env->Invoke("ConvertToYUV444", AVSValue(new_args, 4)).AsClip();
              }
            }
          }
        }
      }
      maskVi = mask->GetVideoInfo();
    }
  }
}


Overlay::~Overlay() {
}

PVideoFrame __stdcall Overlay::GetFrame(int n, IScriptEnvironment *env) {
  // fixme: do all necessary conversions in filter creation, not in GetFrame!
  // 20251122: tried but for some reason it's slower!
  int op_offset;
  float op_offset_f;
  int con_x_offset;
  int con_y_offset;
  FetchConditionals(env, &op_offset, &op_offset_f, &con_x_offset, &con_y_offset, ignore_conditional, condVarSuffix);

  PVideoFrame frame;

  if (inputVi.pixel_type == viInternalWorkingFormat.pixel_type)
  {
    // get frame as is.
    // includes 420, 422, 444, planarRGB(A)
    frame = child->GetFrame(n, env);
  }
  else if (isInternal444) {
    // Y, RGB, 4:2:0, 4:2:2, 4:1:1, 4:4:0, 4:1:0: child444 was already built
    // once in the constructor (see above), via the real resampler at the
    // filter's own `placement`.
    frame = child444->GetFrame(n, env);
  }
  else if (isInternalRGB) {
    if(inputVi.IsYUV()) {
      // Just for the sake of completeness.
      // when input is YUV, internal working format is never RGB
      env->ThrowError("Overlay: internal error; isInternalRGB but input is YUV");
    }
  }

  // Fetch current frame and convert it to internal format
  env->MakeWritable(&frame);

  ImageOverlayInternal* img = new ImageOverlayInternal(frame, vi.width, vi.height, viInternalWorkingFormat, child->GetVideoInfo().IsYUVA() || child->GetVideoInfo().IsPlanarRGBA(), false, env);

  PVideoFrame Oframe;
  AVSValue overlay2;

  PVideoFrame Mframe;
  ImageOverlayInternal* maskImg = NULL;

  // overlay clip should be converted to internal format if different, except for internal Y,
  // for which original planar YUV is OK
  if(overlayVi.pixel_type == viInternalWorkingFormat.pixel_type)
  {
    // don't convert is input and overlay is the same formats
    // so we can work in YUV420 or YUV422 and Planar RGB directly besides YUV444 (use444==false)
    Oframe = overlay->GetFrame(n, env);
  }
  else if (isInternal444) {
    // sanity check
    // optimize: 'multiply' is always using only Y from overlay clip, no need to match chroma
    if (!overlayVi.Is444() && of_mode != OF_Multiply)
      env->ThrowError("Overlay: internal error, overlayVi must be 444 for internal444");
    Oframe = overlay->GetFrame(n, env);
  }
  else if(isInternalGrey) {
    if (!overlayVi.IsY() && !overlayVi.IsYUV() && !overlayVi.IsYUVA())
      env->ThrowError("Overlay: internal error, overlayVi must be Y or YUV(A) for internalGrey");
    // they are good as-is, we'll use only Y
    Oframe = overlay->GetFrame(n, env);
  }
  else if (isInternalRGB) {
    if (!overlayVi.IsPlanarRGB() && !overlayVi.IsPlanarRGBA())
      env->ThrowError("Overlay: internal error, overlayVi must be planar RGB(A) for internalRGB");
    Oframe = overlay->GetFrame(n, env);
  }
  else if (isInternal420) {
    if (!overlayVi.Is420())
      env->ThrowError("Overlay: internal error, overlayVi must be 420 for internal420");
    Oframe = overlay->GetFrame(n, env);
  }
  else if (isInternal422) {
    if (!overlayVi.Is422())
      env->ThrowError("Overlay: internal error, overlayVi must be 422 for internal422");
    Oframe = overlay->GetFrame(n, env);
  }
  else if (isInternal411) {
    if (!overlayVi.Is411())
      env->ThrowError("Overlay: internal error, overlayVi must be 411 for internal411");
    Oframe = overlay->GetFrame(n, env);
  }
  else if (isInternal440) {
    if (!overlayVi.Is440())
      env->ThrowError("Overlay: internal error, overlayVi must be 440 for internal440");
    Oframe = overlay->GetFrame(n, env);
  }
  else if (isInternal410) {
    if (!overlayVi.Is410())
      env->ThrowError("Overlay: internal error, overlayVi must be 410 for internal410");
    Oframe = overlay->GetFrame(n, env);
  }
  else
    env->ThrowError("Overlay: internal error, unhandled internal working format for the overlay clip");
  // Fetch current overlay and convert it to internal format
  VideoInfo actual_viInternalOverlayWorkingFormat = viInternalOverlayWorkingFormat;
  if (of_mode == OF_Multiply) {
    // this mode does not need chroma for Overlay
    switch (bits_per_pixel) {
    case 8: actual_viInternalOverlayWorkingFormat.pixel_type = VideoInfo::CS_Y8; break;
    case 10: actual_viInternalOverlayWorkingFormat.pixel_type = VideoInfo::CS_Y10; break;
    case 12: actual_viInternalOverlayWorkingFormat.pixel_type = VideoInfo::CS_Y12; break;
    case 14: actual_viInternalOverlayWorkingFormat.pixel_type = VideoInfo::CS_Y14; break;
    case 16: actual_viInternalOverlayWorkingFormat.pixel_type = VideoInfo::CS_Y16; break;
    case 32: actual_viInternalOverlayWorkingFormat.pixel_type = VideoInfo::CS_Y32; break;
    }
  }

  ImageOverlayInternal* overlayImg = new ImageOverlayInternal(Oframe, overlayVi.width, overlayVi.height, actual_viInternalOverlayWorkingFormat, overlay->GetVideoInfo().IsYUVA() || overlay->GetVideoInfo().IsPlanarRGBA(), false, env);

  // Clip overlay to original image
  ClipFrames(img, overlayImg, offset_x + con_x_offset, offset_y + con_y_offset);

  if (overlayImg->IsSizeZero()) { // Nothing to overlay
  }
  else {
    // fetch current mask (if given)
    if (mask) {

      AVSValue mask2;
      if(maskVi.IsRGB() && greymask)
        env->ThrowError("Overlay: Internal error, this mask cannot be RGB here, it should be Y or 444 by now.");
      if (maskVi.IsRGB() && isInternal444)
        env->ThrowError("Overlay: Internal error, this mask cannot be RGB here, it should be 444 by now.");
      if (!greymask && isInternalRGB && !maskVi.IsRGB())
        env->ThrowError("Overlay: Internal error, this mask must be RGB here if greymask==false and isInternalRGB.");

      if (greymask
        || getPixelTypeWithoutAlpha(maskVi) == getPixelTypeWithoutAlpha(viInternalWorkingFormat)
        || (!greymask && isInternalRGB)
        )
      {
        // 4:4:4,
        // 4:2:0, 4:2:2 -> greymask uses Y
        // internalworking format: 4:4:4, 4:2:2, 4:2:0
        Mframe = mask->GetFrame(n, env);
      }
      else {
        // greymask == false
        // sanity check
        if (!maskVi.Is444())
          env->ThrowError("Overlay: internal error, maskVi must be 444 here in greymask==false");
        Mframe = mask->GetFrame(n, env);
      }
      // MFrame here is either internalWorkingFormat or Y or 4:4:4
      maskImg = new ImageOverlayInternal(Mframe, maskVi.width, maskVi.height, viInternalOverlayWorkingFormat, mask->GetVideoInfo().IsYUVA() || mask->GetVideoInfo().IsPlanarRGBA(), greymask, env);

      img->ReturnOriginal(true);
      ClipFrames(img, maskImg, offset_x + con_x_offset, offset_y + con_y_offset);


    }

    OverlayFunction* func = SelectFunction();

    // Process the image
    func->setMode(of_mode);
    func->setBitsPerPixel(bits_per_pixel);
    func->setOpacity(opacity + op_offset, opacity_f + op_offset_f);
    func->setColorSpaceInfo(viInternalWorkingFormat.IsRGB(), viInternalWorkingFormat.IsY());

    // overlay and color mask were already re-sited to `placement` in the constructor
    func->setSubsamplingInfo(viInternalWorkingFormat, placement);
    func->setGreyMask(greymask);
    func->setEnv(env);

    if (!mask) {
      func->DoBlendImage(img, overlayImg);
    } else {
      func->DoBlendImageMask(img, overlayImg, maskImg);
    }

    delete func;

    // Reset overlay & image offsets
    img->ReturnOriginal(true);
    overlayImg->ReturnOriginal(true);
    if (mask)
        maskImg->ReturnOriginal(true);
  }

  // Cleanup
  if (mask) {
    delete maskImg;
  }
  delete overlayImg;
  if (img) {
    delete img;
  }

  // here img->frame is 444 whenever use444 is true (isInternal444)
  return frame;
}


/*************************
 *   Helper functions    *
 *************************/

void Overlay::SetOfModeByName(const char* name, IScriptEnvironment* env) {

  if (!lstrcmpi(name, "Blend")) {
    of_mode = OF_Blend;
  }
  else if (!lstrcmpi(name, "Add")) {
    of_mode = OF_Add;
  }
  else if (!lstrcmpi(name, "Subtract")) {
    of_mode = OF_Subtract;
  }
  else if (!lstrcmpi(name, "Multiply")) {
    of_mode = OF_Multiply;
  }
  else if (!lstrcmpi(name, "Chroma")) {
    of_mode = OF_Chroma;
  }
  else if (!lstrcmpi(name, "Luma")) {
    of_mode = OF_Luma;
  }
  else if (!lstrcmpi(name, "Lighten")) {
    of_mode = OF_Lighten;
  }
  else if (!lstrcmpi(name, "Darken")) {
    of_mode = OF_Darken;
  }
  else if (!lstrcmpi(name, "SoftLight")) {
    of_mode = OF_SoftLight;
  }
  else if (!lstrcmpi(name, "HardLight")) {
    of_mode = OF_HardLight;
  }
  else if (!lstrcmpi(name, "Difference")) {
    of_mode = OF_Difference;
  }
  else if (!lstrcmpi(name, "Exclusion")) {
    of_mode = OF_Exclusion;
  }
  else env->ThrowError("Overlay: Invalid 'Mode' specified.");
}

OverlayFunction* Overlay::SelectFunction()
{
  switch (of_mode) {
  case OF_Blend: return new OL_BlendImage();
  case OF_Add: return new OL_AddImage();
  case OF_Subtract: return new OL_AddImage(); // common with Add
  case OF_Multiply: return new OL_MultiplyImage();
  case OF_Chroma: return new OL_BlendImage(); // Common with BlendImage. plane range differs of_mode checked inside
  case OF_Luma: return new OL_BlendImage(); // Common with BlendImage. plane range differs of_mode checked inside
  case OF_Lighten: return new OL_DarkenImage(); // common with Darken
  case OF_Darken: return new OL_DarkenImage();
  case OF_SoftLight: return new OL_SoftLightImage();
  case OF_HardLight: return new OL_SoftLightImage(); // Common with SoftLight
  case OF_Difference: return new OL_DifferenceImage();
  case OF_Exclusion: return new OL_ExclusionImage();
  default: return nullptr; // cannot be
  }
}

void Overlay::ClipFrames(ImageOverlayInternal* input, ImageOverlayInternal* overlay, int x, int y) {

  input->ResetFake();
  overlay->ResetFake();

  input->ReturnOriginal(false);  // We now use cropped space
  overlay->ReturnOriginal(false);

  // Crop negative offset off overlay
  if (x<0) {
    overlay->SubFrame(-x,0,overlay->w()+x, overlay->h());
    x=0;
  }
  if (y<0) {
    overlay->SubFrame(0,-y, overlay->w(), overlay->h()+y);
    y=0;
  }
  // Clip input-frame to topleft overlay:
  input->SubFrame(x,y,input->w()-x, input->h()-y);

  // input and overlay are now topleft aligned

  // Clip overlay that is beyond the right side of the input

  if (overlay->w() > input->w()) {
    overlay->SubFrame(0,0,input->w(), overlay->h());
  }

  if (overlay->h() > input->h()) {
    overlay->SubFrame(0,0,overlay->w(), input->h());
  }

  // Clip right/ bottom of input

  if(input->w() > overlay->w()) {
    input->SubFrame(0,0, overlay->w(), input->h());
  }

  if(input->h() > overlay->h()) {
    input->SubFrame(0,0, input->w(), overlay->h());
  }

}

void Overlay::FetchConditionals(IScriptEnvironment* env, int* op_offset, float* op_offset_f, int* con_x_offset, int* con_y_offset, bool ignore_conditional, const char *condVarSuffix) {
  *op_offset = 0;
  *op_offset_f = 0.0f;
  *con_x_offset = 0;
  *con_y_offset = 0;

  if (!ignore_conditional) {
    {
      std::string s = std::string("OL_opacity_offset") + condVarSuffix;
      *op_offset = (int)(env->GetVarDouble(s.c_str(), 0.0) * 256);
      *op_offset_f = (float)(env->GetVarDouble(s.c_str(), 0.0));
    }
    {
      std::string s = std::string("OL_x_offset") + condVarSuffix;
      *con_x_offset = (int)(env->GetVarDouble(s.c_str(), 0.0));
    }
    {
      std::string s = std::string("OL_y_offset") + condVarSuffix;
      *con_y_offset = (int)(env->GetVarDouble(s.c_str(), 0.0));
    }
  }
}


AVSValue __cdecl Overlay::Create(AVSValue args, void*, IScriptEnvironment* env) {
  // provide planar-only main clip
  PClip input = args[0].AsClip();
  VideoInfo vi_orig = input->GetVideoInfo();
  bool converted = false;

  // input clip to always planar
  if (vi_orig.IsRGB() && !vi_orig.IsPlanar()) {
    // no packed RGB allowed from now on, autoconvert from packed
    AVSValue new_args[1] = { input };
    if(vi_orig.IsRGB24() || vi_orig.IsRGB48())
      input = env->Invoke("ConvertToPlanarRGB", AVSValue(new_args, 1)).AsClip();
    else // RGB32, RGB64
      input = env->Invoke("ConvertToPlanarRGBA", AVSValue(new_args, 1)).AsClip();
    converted = true;
  }
  else if (vi_orig.IsYUY2()) {
    // convert YUY2 to 422, keep internals simple
    AVSValue new_args[2] = { input, false };
    input = env->Invoke("ConvertToYUV422", AVSValue(new_args, 2)).AsClip();
    converted = true;
  }

  // input now is always planar, overlay and mask clip converted later
  Overlay* Result = new Overlay(input, args, env);

  // where there was no output override but old formats were converted
  // then we turn them back
  if (Result->output_pixel_format_override == nullptr && converted) {
    if (vi_orig.IsRGB() && !vi_orig.IsPlanar()) {
      AVSValue new_args[1] = { Result };
      if (vi_orig.IsRGB24())
        return env->Invoke("ConvertToRGB24", AVSValue(new_args, 1)).AsClip();
      else if (vi_orig.IsRGB32())
        return env->Invoke("ConvertToRGB32", AVSValue(new_args, 1)).AsClip();
      else if (vi_orig.IsRGB48())
        return env->Invoke("ConvertToRGB48", AVSValue(new_args, 1)).AsClip();
      else // if (vi_orig.IsRGB64())
        return env->Invoke("ConvertToRGB64", AVSValue(new_args, 1)).AsClip();
    }
    if (vi_orig.IsYUY2()) {
      // convert back to YUY2
      AVSValue new_args[2] = { Result, false };
      return env->Invoke("ConvertToYUY2", AVSValue(new_args, 2)).AsClip();
    }
  }

  if (Result->GetVideoInfo().pixel_type == Result->outputVi.pixel_type)
     return Result;
   // c[interlaced]b[matrix]s[ChromaInPlacement]s[chromaresample]s
   // chromaresample = 'bicubic' default
   // chromaresample = 'point' is faster
   // If output requires, we keep existing/add new alpha, though it is not altered in any overlay subfilter.
   const bool outputIsAlphaYUV = Result->outputVi.IsYUVA(); // false for plain YUVxxx targets
   if(Result->outputVi.Is444()) {
     // if workingFormat is not 444 but output was specified
     // c[interlaced]b[matrix]s[ChromaInPlacement]s
     // Source is subsampled, use ChromaInPlacement to this filter's own `placement`.
     AVSValue new_args[4] = { Result, false, Result->full_range ? "PC.601" : "rec601", GetChromaLocationName(Result->chromaLocation) };
     return env->Invoke(outputIsAlphaYUV ? "ConvertToYUVA444" : "ConvertToYUV444", AVSValue(new_args, 4)).AsClip();
   }
   // c[interlaced]b[matrix]s[ChromaInPlacement]s[chromaresample]s[ChromaOutPlacement]s
   // source (Result) is always 4:4:4 here (isInternal444)
   // Non-subsampled base without explicit placement: the output format's own default siting.
   const char* outputChromaLocationName = GetChromaLocationName(
     Result->chromaLocationDefined ? Result->chromaLocation : GetDefaultChromaLocation(Result->outputVi));
   // ChromaOutPlacement goes to `placement`: reconstructed output siting matches
   // whatever the input side (base/overlay clip conversion in the ctor) assumed.
   if(Result->outputVi.Is422()) {
     AVSValue new_args[6] = { Result, false, Result->full_range ? "PC.601" : "rec601", AVSValue(), AVSValue(), outputChromaLocationName };
     return env->Invoke(outputIsAlphaYUV ? "ConvertToYUVA422" : "ConvertToYUV422", AVSValue(new_args, 6)).AsClip();
   }
   if(Result->outputVi.Is420()) {
     AVSValue new_args[6] = { Result, false, Result->full_range ? "PC.601" : "rec601", AVSValue(), AVSValue(), outputChromaLocationName };
     return env->Invoke(outputIsAlphaYUV ? "ConvertToYUVA420" : "ConvertToYUV420", AVSValue(new_args, 6)).AsClip();
   }
   if (Result->outputVi.Is411()) {
     AVSValue new_args[6] = { Result, false, Result->full_range ? "PC.601" : "rec601", AVSValue(), AVSValue(), outputChromaLocationName };
     return env->Invoke(outputIsAlphaYUV ? "ConvertToYUVA411" : "ConvertToYUV411", AVSValue(new_args, 6)).AsClip();
   }
   if (Result->outputVi.Is440()) {
     AVSValue new_args[6] = { Result, false, Result->full_range ? "PC.601" : "rec601", AVSValue(), AVSValue(), outputChromaLocationName };
     return env->Invoke(outputIsAlphaYUV ? "ConvertToYUVA440" : "ConvertToYUV440", AVSValue(new_args, 6)).AsClip();
   }
   if (Result->outputVi.Is410()) {
     AVSValue new_args[6] = { Result, false, Result->full_range ? "PC.601" : "rec601", AVSValue(), AVSValue(), outputChromaLocationName };
     return env->Invoke(outputIsAlphaYUV ? "ConvertToYUVA410" : "ConvertToYUV410", AVSValue(new_args, 6)).AsClip();
   }
   if(Result->outputVi.IsYUY2()) {
     AVSValue new_args[3] = { Result, false, Result->full_range ? "PC.601" : "rec601" };
     return env->Invoke("ConvertToYUY2", AVSValue(new_args, 3)).AsClip();
   }
   if(Result->outputVi.IsY()) {
     AVSValue new_args[2] = { Result, Result->full_range ? "PC.601" : "rec601" };
     return env->Invoke("ConvertToY", AVSValue(new_args, 2)).AsClip();
   }
   if(Result->outputVi.IsYA()) {
     // ConvertToYA: like ConvertToYUVA: keep existing/add new alpha.
     AVSValue new_args[2] = { Result, Result->full_range ? "PC.601" : "rec601" };
     return env->Invoke("ConvertToYA", AVSValue(new_args, 2)).AsClip();
   }
   if(Result->outputVi.IsRGB()) {
     // c[matrix]s[interlaced]b[ChromaInPlacement]s[chromaresample]s
     AVSValue new_args[3] = { Result, Result->full_range ? "PC.601" : "rec601", false};
     if(Result->outputVi.IsPlanarRGB()) {
       return env->Invoke("ConvertToPlanarRGB", AVSValue(new_args, 3)).AsClip();
     }
     if(Result->outputVi.IsPlanarRGBA()) {
       return env->Invoke("ConvertToPlanarRGBA", AVSValue(new_args, 3)).AsClip();
     }
     if(Result->outputVi.IsRGB24()) {
       return env->Invoke("ConvertToRGB24", AVSValue(new_args, 3)).AsClip();
     }
     if(Result->outputVi.IsRGB32()) {
       return env->Invoke("ConvertToRGB32", AVSValue(new_args, 3)).AsClip();
     }
     if(Result->outputVi.IsRGB48()) {
       return env->Invoke("ConvertToRGB48", AVSValue(new_args, 3)).AsClip();
     }
     if(Result->outputVi.IsRGB64()) {
       return env->Invoke("ConvertToRGB64", AVSValue(new_args, 3)).AsClip();
     }
   }
   env->ThrowError("Overlay: Invalid output format.");
   return Result;
}
