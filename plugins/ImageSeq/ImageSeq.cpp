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



#include <avisynth.h>
#include "ImageSeq.h"

// Since devIL isn't threadsafe, we need to ensure that only one thread at the time requests frames
std::mutex DevIL_mutex;
volatile ILint DevIL_Version = 0;

AVSValue __cdecl Create_ImageWriter(AVSValue args, void* user_data, IScriptEnvironment* env)
{
  PClip clip = args[0].AsClip();
  const VideoInfo &vi = clip->GetVideoInfo();
  if (vi.IsPlanarRGB() || vi.IsPlanarRGBA())
  {
    // silently convert to packed bitmap
    int bits_per_pixel = vi.BitsPerComponent();
    bool hasAlpha = vi.IsPlanarRGBA();
    AVSValue new_args[1] = { clip };
    if (bits_per_pixel == 8) {
      if (hasAlpha)
        clip = env->Invoke("ConvertToRGB32", AVSValue(new_args, 1)).AsClip();
      else
        clip = env->Invoke("ConvertToRGB24", AVSValue(new_args, 1)).AsClip();
    }
    else if (bits_per_pixel == 16) {
      if (hasAlpha)
        clip = env->Invoke("ConvertToRGB64", AVSValue(new_args, 1)).AsClip();
      else
        clip = env->Invoke("ConvertToRGB48", AVSValue(new_args, 1)).AsClip();
    }
    else {
      env->ThrowError("ImageWriter: only 8 or 16 bit planar RGB formats supported");
    }
  }

  return new ImageWriter(clip,
#ifdef AVS_WINDOWS
                         env->SaveString(args[1].AsString("c:\\")),
#else
                         env->SaveString(args[1].AsString("")),
#endif
                         args[2].AsInt(0),
                         args[3].AsInt(0),
                         env->SaveString(args[4].AsString("ebmp")),
                         args[5].AsBool(false), env);
}

AVSValue __cdecl Create_ImageReader(AVSValue args, void*, IScriptEnvironment* env)
{
#ifdef AVS_WINDOWS
  const char * path = args[0].AsString("c:\\%06d.ebmp");
#else
  const char* path = args[0].AsString("./%06d.ebmp");
#endif

  ImageReader *IR = new ImageReader(path, args[1].AsInt(0), args[2].AsInt(1000), (float)args[3].AsDblDef(24.0),
                                    args[4].AsBool(false), args[5].AsBool(false), args[6].AsString("rgb24"),
                                    /*animation*/ false, env);
  // If we are returning a stream of 2 or more copies of the same image
  // then use FreezeFrame and the Cache to minimise any reloading.
  if (IR->framecopies > 1) {
    AVSValue cache = env->Invoke("Cache", AVSValue(IR));
    AVSValue ff_args[4] = { cache, 0, IR->framecopies-1, 0 };
    return env->Invoke("FreezeFrame", AVSValue(ff_args, 4)).AsClip();
  }

  return IR;
}

AVSValue __cdecl Create_Animated(AVSValue args, void*, IScriptEnvironment* env)
{
  if (!args[0].IsString())
    env->ThrowError("ImageSourceAnim: You must specify a filename.");

  return new ImageReader(args[0].AsString(), 0, 0, (float)args[1].AsDblDef(24.0), /*use_DevIL*/ true,
                         args[2].AsBool(false), args[3].AsString("rgb32"), /*animation*/ true, env);
}

const AVS_Linkage * AVS_linkage = 0;
extern "C" AVS_PLUGIN_EXPORT const char* __stdcall AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors)
{
	AVS_linkage = vectors;

  // clip, base filename, start, end, image format/extension, info
  env->AddFunction("ImageWriter", "c[file]s[start]i[end]i[type]s[info]b", Create_ImageWriter, 0);

  // base filename (sprintf-style), start, end, frames per second, default reader to use, info, pixel_type
  env->AddFunction("ImageReader", "[file]s[start]i[end]i[fps]f[use_devil]b[info]b[pixel_type]s", Create_ImageReader, 0);
  env->AddFunction("ImageSource", "[file]s[start]i[end]i[fps]f[use_devil]b[info]b[pixel_type]s", Create_ImageReader, 0);

  env->AddFunction("ImageSourceAnim", "[file]s[fps]f[info]b[pixel_type]s", Create_Animated, 0);

  return "`ImageSeq' Methods for loading and writing still images.";
}

const char *const getErrStr(ILenum err)
{
  if (err == IL_INVALID_ENUM)
    return "Invalid Enum";
  if (err == IL_OUT_OF_MEMORY)
    return "Out of memory";
  if (err == IL_FORMAT_NOT_SUPPORTED)
    return "Format not supported";
  if (err == IL_INTERNAL_ERROR)
    return "Internal error";
  if (err == IL_INVALID_VALUE)
    return "Invalid value";
  if (err == IL_ILLEGAL_OPERATION)
    return "Illegal operation";
  if (err == IL_ILLEGAL_FILE_VALUE)
    return "Illegal file value";
  if (err == IL_INVALID_FILE_HEADER)
    return "Illegal file header";
  if (err == IL_INVALID_PARAM)
    return "Invalid Parameter";
  if (err == IL_COULD_NOT_OPEN_FILE)
    return "Could not open file";
  if (err == IL_INVALID_EXTENSION)
    return "Invalid extension";
  if (err == IL_FILE_ALREADY_EXISTS)
    return "File already exists";
  if (err == IL_OUT_FORMAT_SAME)
    return "Output format same";
  if (err == IL_STACK_OVERFLOW)
    return "Stack overflow";
  if (err == IL_STACK_UNDERFLOW)
    return "Stack underflow";
  if (err == IL_INVALID_CONVERSION)
    return "Invalid conversion";
  if (err == IL_BAD_DIMENSIONS)
    return "Bad dimensions";
  if ((err == IL_FILE_READ_ERROR) || (err == IL_FILE_WRITE_ERROR))
    return "File read/write error";
  if (err == IL_LIB_GIF_ERROR)
    return "LibGif error";
  if (err == IL_LIB_JPEG_ERROR)
    return "LibJpeg error";
  if (err == IL_LIB_PNG_ERROR)
    return "LibPng error";
  if (err == IL_LIB_TIFF_ERROR)
    return "LibTiff error";
  if (err == IL_LIB_MNG_ERROR)
    return "LibMng error";
#if IL_VERSION >= 178
  if (err == IL_LIB_JP2_ERROR)
    return "LibJP2 error";
  if (err == IL_LIB_EXR_ERROR)
    return "LibExr error";
#endif
  if (err == IL_UNKNOWN_ERROR)
    return "Unknown error";

  return "Unknown error";
}

