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


#include "limiter_sse.h"
#include <emmintrin.h>
#include <smmintrin.h> // for sse41
#include "../core/internal.h"

//min and max values are 16-bit integers either max_plane|max_plane for planar or max_luma|max_chroma for yuy2
void limit_plane_sse2(BYTE *ptr, int min_value, int max_value, int pitch, int width, int height) {
  __m128i min_vector = _mm_set1_epi16(min_value);
  __m128i max_vector = _mm_set1_epi16(max_value);
  BYTE* end_point = ptr + pitch * height;

  while(ptr < end_point) {
    __m128i src = _mm_load_si128(reinterpret_cast<const __m128i*>(ptr));
    src = _mm_max_epu8(src, min_vector);
    src = _mm_min_epu8(src, max_vector);
    _mm_store_si128(reinterpret_cast<__m128i*>(ptr), src);
    ptr += 16;
  }
}

//min and max values are 16-bit unsigned integers
void limit_plane_uint16_sse2(BYTE *ptr, unsigned int min_value, unsigned int max_value, int pitch, int height) {
  __m128i min_vector = _mm_set1_epi16(min_value);
  __m128i max_vector = _mm_set1_epi16(max_value);
  BYTE* end_point = ptr + pitch * height;

  while(ptr < end_point) {
    __m128i src = _mm_load_si128(reinterpret_cast<const __m128i*>(ptr));
    src = _MM_MAX_EPU16(src, min_vector);
    src = _MM_MIN_EPU16(src, max_vector);
    _mm_store_si128(reinterpret_cast<__m128i*>(ptr), src);
    ptr += 16;
  }
}

//min and max values are 16-bit unsigned integers
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
void limit_plane_uint16_sse4(BYTE *ptr, unsigned int min_value, unsigned int max_value, int pitch, int height)
{
  __m128i min_vector = _mm_set1_epi16(min_value);
  __m128i max_vector = _mm_set1_epi16(max_value);
  BYTE* end_point = ptr + pitch * height;

  while(ptr < end_point) {
    __m128i src = _mm_load_si128(reinterpret_cast<const __m128i*>(ptr));
    src = _mm_max_epu16(src, min_vector);
    src = _mm_min_epu16(src, max_vector);
    _mm_store_si128(reinterpret_cast<__m128i*>(ptr), src);
    ptr += 16;
  }
}


#ifdef X86_32

//min and max values are 16-bit integers either max_plane|max_plane for planar or max_luma|max_chroma for yuy2
void limit_plane_isse(BYTE *ptr, int min_value, int max_value, int pitch, int width, int height) {
  __m64 min_vector = _mm_set1_pi16(min_value);
  __m64 max_vector = _mm_set1_pi16(max_value);
  int mod8_width = width / 8 * 8;

  for(int y = 0; y < height; y++) {
    for(int x = 0; x < mod8_width; x+=8) {
      __m64 src = *reinterpret_cast<__m64*>(ptr+x);
      src = _mm_max_pu8(src, min_vector);
      src = _mm_min_pu8(src, max_vector);
      *reinterpret_cast<__m64*>(ptr+x) = src;
    }

    if (mod8_width != width) {
      int x = width - 8;
      __m64 src = *reinterpret_cast<__m64*>(ptr+x);
      src = _mm_max_pu8(src, min_vector);
      src = _mm_min_pu8(src, max_vector);
      *reinterpret_cast<__m64*>(ptr+x) = src;
    }

    ptr += pitch;
  }
  _mm_empty();
}

#endif



