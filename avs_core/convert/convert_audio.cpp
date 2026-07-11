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

// ConvertAudio classes
// Copyright (c) Klaus Post 2001 - 2004
// Copyright (c) Ian Brabham 2005
// Copyright (c) 2020 Xinyue Lu
// Copyright (c) 2021 pinterf

#include <avisynth.h>
#include <avs/alignment.h>
#include "convert_audio.h"
#if defined(AVS_BSD) || defined(AVS_MACOS)
  #include <stdlib.h>
#else
  #include <malloc.h>
#endif
#include <limits>

// There are two type parameters. Acceptable sample types and a prefered sample type.
// If the current clip is already one of the defined types in sampletype, this will be returned.
// If not, the current clip will be converted to the prefered type.
PClip ConvertAudio::Create(PClip clip, int sample_type, int prefered_type) {
  if ((!clip->GetVideoInfo().HasAudio()) || clip->GetVideoInfo().SampleType() & (sample_type | prefered_type)) {
    // Sample type is already ok!
    return clip;
  } else
    return new ConvertAudio(clip, prefered_type);
}

int __stdcall ConvertAudio::SetCacheHints(int cachehints, int frame_range) {
  // We do pass cache requests upwards, to the next filter.
  return child->SetCacheHints(cachehints, frame_range);
}

ConvertAudio::ConvertAudio(PClip _clip, int _sample_type)
    : GenericVideoFilter(_clip) {
  dst_format = _sample_type;
  src_format = vi.SampleType();
  // Set up convertion matrix
  src_bps = vi.BytesPerChannelSample(); // Store old size
  vi.sample_type = dst_format;
  tempbuffer_size = 0;

  #define PAIR(src, dst) ((src << 16) | dst)
  switch(PAIR(src_format, dst_format)) {
    case PAIR(SAMPLE_INT32, SAMPLE_INT16): convert_c = convert32To16; break;
    case PAIR(SAMPLE_INT16, SAMPLE_INT32): convert_c = convert16To32; break;
    case PAIR(SAMPLE_INT32, SAMPLE_INT8 ): convert_c = convert32To8; break;
    case PAIR(SAMPLE_INT8 , SAMPLE_INT32): convert_c = convert8To32; break;
    case PAIR(SAMPLE_INT16, SAMPLE_INT8 ): convert_c = convert16To8; break;
    case PAIR(SAMPLE_INT8 , SAMPLE_INT16): convert_c = convert8To16; break;
    case PAIR(SAMPLE_FLOAT, SAMPLE_INT24): two_stage = true; // no-break;
    case PAIR(SAMPLE_INT32, SAMPLE_INT24): convert_c = convert32To24; break;
    case PAIR(SAMPLE_INT24, SAMPLE_FLOAT): two_stage = true; // no-break;
    case PAIR(SAMPLE_INT24, SAMPLE_INT32): convert_c = convert24To32; break;
    case PAIR(SAMPLE_INT24, SAMPLE_INT16): convert_c = convert24To16; break;
    case PAIR(SAMPLE_INT16, SAMPLE_INT24): convert_c = convert16To24; break;
    case PAIR(SAMPLE_INT24, SAMPLE_INT8 ): convert_c = convert24To8; break;
    case PAIR(SAMPLE_INT8 , SAMPLE_INT24): convert_c = convert8To24; break;
    case PAIR(SAMPLE_INT8 , SAMPLE_FLOAT): convert_c = convert8ToFLT; break;
    case PAIR(SAMPLE_FLOAT, SAMPLE_INT8): convert_c = convertFLTTo8; break;
    case PAIR(SAMPLE_INT16, SAMPLE_FLOAT): convert_c = convert16ToFLT; break;
    case PAIR(SAMPLE_FLOAT, SAMPLE_INT16): convert_c = convertFLTTo16; break;
    case PAIR(SAMPLE_INT32, SAMPLE_FLOAT): convert_c = convert32ToFLT; break;
    case PAIR(SAMPLE_FLOAT, SAMPLE_INT32): convert_c = convertFLTTo32; break;
  }
  #ifdef INTEL_INTRINSICS
    switch(PAIR(src_format, dst_format)) {
      case PAIR(SAMPLE_INT32, SAMPLE_INT16): convert_sse2  = convert32To16_SSE2;  convert_avx2 = convert32To16_AVX2;  break;
      case PAIR(SAMPLE_INT16, SAMPLE_INT32): convert_sse2  = convert16To32_SSE2;  convert_avx2 = convert16To32_AVX2;  break;
      case PAIR(SAMPLE_INT32, SAMPLE_INT8 ): convert_sse2  = convert32To8_SSE2;   break;
      case PAIR(SAMPLE_INT8 , SAMPLE_INT32): convert_sse2  = convert8To32_SSE2;   break;
      case PAIR(SAMPLE_INT16, SAMPLE_INT8 ): convert_sse2  = convert16To8_SSE2;   break;
      case PAIR(SAMPLE_INT8 , SAMPLE_INT16): convert_sse2  = convert8To16_SSE2;   break;
      case PAIR(SAMPLE_FLOAT, SAMPLE_INT24):
      case PAIR(SAMPLE_INT32, SAMPLE_INT24): convert_ssse3 = convert32To24_SSSE3; break;
      case PAIR(SAMPLE_INT24, SAMPLE_FLOAT):
      case PAIR(SAMPLE_INT24, SAMPLE_INT32): convert_ssse3 = convert24To32_SSSE3; break;
      case PAIR(SAMPLE_INT24, SAMPLE_INT16): convert_ssse3 = convert24To16_SSSE3; break;
      case PAIR(SAMPLE_INT16, SAMPLE_INT24): convert_ssse3 = convert16To24_SSSE3; break;
      case PAIR(SAMPLE_INT24, SAMPLE_INT8 ): convert_ssse3 = convert24To8_SSSE3;  break;
      case PAIR(SAMPLE_INT8 , SAMPLE_INT24): convert_ssse3 = convert8To24_SSSE3;  break;
      case PAIR(SAMPLE_INT8 , SAMPLE_FLOAT): convert_sse41 = convert8ToFLT_SSE41; convert_avx2 = convert8ToFLT_AVX2; break;
      case PAIR(SAMPLE_FLOAT, SAMPLE_INT8) : convert_sse2 = convertFLTTo8_SSE2; convert_avx2 = convertFLTTo8_AVX2; break;
      case PAIR(SAMPLE_INT16, SAMPLE_FLOAT): convert_sse41 = convert16ToFLT_SSE41; convert_avx2 = convert16ToFLT_AVX2; break;
      case PAIR(SAMPLE_FLOAT, SAMPLE_INT16): convert_sse2 = convertFLTTo16_SSE2; convert_avx2 = convertFLTTo16_AVX2; break;
      case PAIR(SAMPLE_INT32, SAMPLE_FLOAT): convert_sse2  = convert32ToFLT_SSE2; convert_avx2 = convert32ToFLT_AVX2; break;
      case PAIR(SAMPLE_FLOAT, SAMPLE_INT32): convert_sse41 = convertFLTTo32_SSE41; convert_avx2 = convertFLTTo32_AVX2; break;
    }
  #endif
  #undef PAIR
}

ConvertAudio::~ConvertAudio() {
  if (tempbuffer_size) {
    avs_free(tempbuffer);
    tempbuffer_size = 0;
  }
}

void __stdcall ConvertAudio::GetAudio(void *buf, int64_t start, int64_t count, IScriptEnvironment *env) {
  if (src_format == dst_format) {
    // Shouldn't happen, but just in case
    child->GetAudio(buf, start, count, env);
    return;
  }

  int channels = vi.AudioChannels();

  if (tempbuffer_size < count) {
    if (count < 0 || count > std::numeric_limits<int>::max())
      env->ThrowError("ConvertAudio: audio count is too large.");

    const size_t sample_count = static_cast<size_t>(count);
    const size_t bytes_per_sample = static_cast<size_t>(src_bps) * static_cast<size_t>(channels);

    if (bytes_per_sample == 0 || sample_count > std::numeric_limits<size_t>::max() / bytes_per_sample)
      env->ThrowError("ConvertAudio: audio buffer size overflow.");

    const size_t buffer_size = sample_count * bytes_per_sample;
    char* new_tempbuffer = static_cast<char*>(avs_malloc(buffer_size, 16));

    if (!new_tempbuffer)
      env->ThrowError("ConvertAudio: insufficient memory.");

    avs_free(tempbuffer);
    tempbuffer = new_tempbuffer;
    tempbuffer_size = static_cast<int>(count);
  }

  child->GetAudio(tempbuffer, start, count, env);

  if (convert == nullptr) {
    convert = convert_c;
    convert_float = src_format == SAMPLE_FLOAT ? convertFLTTo32 : convert32ToFLT; // for two-stage
    #ifdef INTEL_INTRINSICS
      int cpu_flags = env->GetCPUFlags();
      if ((cpu_flags & CPUF_SSE2)) {
        if (convert_sse2)
          convert = convert_sse2;
        if (src_format != SAMPLE_FLOAT)
          convert_float = convert32ToFLT_SSE2;
      }
      if ((cpu_flags & CPUF_SSSE3)) {
        if (convert_ssse3)
          convert = convert_ssse3;
      }
      if ((cpu_flags & CPUF_SSE4_1)) {
        if (convert_sse41)
          convert = convert_sse41;
        if (src_format == SAMPLE_FLOAT)
          convert_float = convertFLTTo32_SSE41;
      }
      if ((cpu_flags & CPUF_AVX2)) {
        if (convert_avx2)
          convert = convert_avx2;
        convert_float = src_format == SAMPLE_FLOAT ? convertFLTTo32_AVX2 : convert32ToFLT_AVX2;
      }
    #endif
  }

  int sample_count = static_cast<int>(count * channels);

  // Direct conversion cases
  if (!two_stage) {
    convert(tempbuffer, buf, sample_count);
    return;
  }

  // Floating point cases
  if (src_format == SAMPLE_FLOAT) {
    convert_float(tempbuffer, tempbuffer, sample_count);
    convert(tempbuffer, buf, sample_count);
    return;
  }

  if (dst_format == SAMPLE_FLOAT) {
    convert(tempbuffer, buf, sample_count);
    convert_float(buf, buf, sample_count);
    return;
  }
}
