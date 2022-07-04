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

#ifndef __Convert_bits_H__
#define __Convert_bits_H__

#include <avisynth.h>
#include <stdint.h>
#include "convert.h"

typedef void (*BitDepthConvFuncPtr)(const BYTE *srcp, BYTE *dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth);

class ConvertBits : public GenericVideoFilter
{
public:
  ConvertBits(PClip _child, const int _dither_mode, const int _target_bitdepth, bool _truerange, int _ColorRange_src, int _ColorRange_dest, int _dither_bitdepth, IScriptEnvironment* env);
  PVideoFrame __stdcall GetFrame(int n,IScriptEnvironment* env) override;

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    AVS_UNUSED(frame_range);
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

  static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env);
private:
  BitDepthConvFuncPtr conv_function;
  BitDepthConvFuncPtr conv_function_chroma; // 32bit float YUV chroma
  BitDepthConvFuncPtr conv_function_a;
  int target_bitdepth;
  int dither_mode;
  int dither_bitdepth;
  bool fulls; // source is full range (defaults: rgb=true, yuv=false (bit shift))
  bool fulld; // destination is full range (defaults: rgb=true, yuv=false (bit shift))
  bool truerange; // if 16->10 range reducing or e.g. 14->16 bit range expansion needed
  int pixelsize;
  int bits_per_pixel;
  bool format_change_only;
};

/**********************************
******  Bitdepth conversions  *****
**********************************/
// for odd dither bit differences, we still take even size but
// correction values are halved (shifted by 1)

// repeated 8x for sse size 16
static const struct dither2x2a_t
{
  const BYTE data[4] = {
    0, 1,
    1, 0,
  };
  // cycle: 2
  alignas(16) const BYTE data_sse2[2 * 16] = {
    0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0
  };
  dither2x2a_t() {};
} dither2x2a;

// e.g. 10->8 bits
// repeated 8x for sse size 16
static const struct dither2x2_t
{
  const BYTE data[4] = {
    0, 2,
    3, 1,
  };
  // cycle: 2
  alignas(16) const BYTE data_sse2[2 * 16] = {
    0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2,
    3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1
  };
  dither2x2_t() {};
} dither2x2;

// e.g. 8->5 bits
static const struct dither4x4a_t
{
  const BYTE data[16] = {
     0,  4,  1,  5,
     6,  2,  7,  3,
     1,  5,  0,  4,
     7,  3,  6,  2
  };
  // cycle: 4
  alignas(16) const BYTE data_sse2[4 * 16] = {
     0,  4,  1,  5,  0,  4,  1,  5,  0,  4,  1,  5,  0,  4,  1,  5,
     6,  2,  7,  3,  6,  2,  7,  3,  6,  2,  7,  3,  6,  2,  7,  3,
     1,  5,  0,  4,  1,  5,  0,  4,  1,  5,  0,  4,  1,  5,  0,  4,
     7,  3,  6,  2,  7,  3,  6,  2,  7,  3,  6,  2,  7,  3,  6,  2,
  };
  dither4x4a_t() {};
} dither4x4a;

// e.g. 12->8 bits
static const struct dither4x4_t
{
  const BYTE data[16] = {
     0,  8,  2, 10,
    12,  4, 14,  6,
     3, 11,  1,  9,
    15,  7, 13,  5
  };
  // cycle: 4
  alignas(16) const BYTE data_sse2[4 * 16] = {
     0,  8,  2, 10,  0,  8,  2, 10,  0,  8,  2, 10,  0,  8,  2, 10,
    12,  4, 14,  6, 12,  4, 14,  6, 12,  4, 14,  6, 12,  4, 14,  6,
     3, 11,  1,  9,  3, 11,  1,  9,  3, 11,  1,  9,  3, 11,  1,  9,
    15,  7, 13,  5, 15,  7, 13,  5, 15,  7, 13,  5, 15,  7, 13,  5
  };
  dither4x4_t() {};
} dither4x4;

// e.g. 14->9 bits
static const struct dither8x8a_t
{
  const BYTE data[8][8] = {
    { 0, 16,  4, 20,  1, 17,  5, 21}, /* 8x8 Bayer ordered dithering pattern */
    {24,  8, 28, 12, 25,  9, 29, 13},
    { 6, 22,  2, 18,  7, 23,  3, 19},
    {30, 14, 26, 10, 31, 15, 27, 11},
    { 1, 17,  5, 21,  0, 16,  4, 20},
    {25,  9, 29, 13, 24,  8, 28, 12},
    { 7, 23,  3, 19,  6, 22,  2, 18},
    {31, 15, 27, 11, 30, 14, 26, 10}
  };
  // cycle: 8
  alignas(16) const BYTE data_sse2[8][16] = {
    {  0, 16,  4, 20,  1, 17,  5, 21,  0, 16,  4, 20,  1, 17,  5, 21 },
    { 24,  8, 28, 12, 25,  9, 29, 13, 24,  8, 28, 12, 25,  9, 29, 13 },
    {  6, 22,  2, 18,  7, 23,  3, 19,  6, 22,  2, 18,  7, 23,  3, 19 },
    { 30, 14, 26, 10, 31, 15, 27, 11, 30, 14, 26, 10, 31, 15, 27, 11 },
    {  1, 17,  5, 21,  0, 16,  4, 20,  1, 17,  5, 21,  0, 16,  4, 20 },
    { 25,  9, 29, 13, 24,  8, 28, 12, 25,  9, 29, 13, 24,  8, 28, 12 },
    {  7, 23,  3, 19,  6, 22,  2, 18,  7, 23,  3, 19,  6, 22,  2, 18 },
    { 31, 15, 27, 11, 30, 14, 26, 10, 31, 15, 27, 11, 30, 14, 26, 10 }
  };
  dither8x8a_t() {};
} dither8x8a;

// e.g. 14->8 bits
static const struct dither8x8_t
{
  const BYTE data[8][8] = {
    { 0, 32,  8, 40,  2, 34, 10, 42},
    {48, 16, 56, 24, 50, 18, 58, 26},
    {12, 44,  4, 36, 14, 46,  6, 38},
    {60, 28, 52, 20, 62, 30, 54, 22},
    { 3, 35, 11, 43,  1, 33,  9, 41},
    {51, 19, 59, 27, 49, 17, 57, 25},
    {15, 47,  7, 39, 13, 45,  5, 37},
    {63, 31, 55, 23, 61, 29, 53, 21}
  };
  // cycle: 8
  alignas(16) const BYTE data_sse2[8][16] = {
    {  0, 32,  8, 40,  2, 34, 10, 42,  0, 32,  8, 40,  2, 34, 10, 42 },
    { 48, 16, 56, 24, 50, 18, 58, 26, 48, 16, 56, 24, 50, 18, 58, 26 },
    { 12, 44,  4, 36, 14, 46,  6, 38, 12, 44,  4, 36, 14, 46,  6, 38 },
    { 60, 28, 52, 20, 62, 30, 54, 22, 60, 28, 52, 20, 62, 30, 54, 22 },
    {  3, 35, 11, 43,  1, 33,  9, 41,  3, 35, 11, 43,  1, 33,  9, 41 },
    { 51, 19, 59, 27, 49, 17, 57, 25, 51, 19, 59, 27, 49, 17, 57, 25 },
    { 15, 47,  7, 39, 13, 45,  5, 37, 15, 47,  7, 39, 13, 45,  5, 37 },
    { 63, 31, 55, 23, 61, 29, 53, 21, 63, 31, 55, 23, 61, 29, 53, 21 }
  };
  dither8x8_t() {};
} dither8x8;

// e.g. 16->9 or 8->1 bits
static const struct dither16x16a_t
{
  // cycle: 16x. No special 16 byte sse2
  alignas(16) const BYTE data[16][16] = {
    {   0, 96, 24,120,  6,102, 30,126,  1, 97, 25,121,  7,103, 31,127 },
    {  64, 32, 88, 56, 70, 38, 94, 62, 65, 33, 89, 57, 71, 39, 95, 63 },
    {  16,112,  8,104, 22,118, 14,110, 17,113,  9,105, 23,119, 15,111 },
    {  80, 48, 72, 40, 86, 54, 78, 46, 81, 49, 73, 41, 87, 55, 79, 47 },
    {   4,100, 28,124,  2, 98, 26,122,  5,101, 29,125,  3, 99, 27,123 },
    {  68, 36, 92, 60, 66, 34, 90, 58, 69, 37, 93, 61, 67, 35, 91, 59 },
    {  20,116, 12,108, 18,114, 10,106, 21,117, 13,109, 19,115, 11,107 },
    {  84, 52, 76, 44, 82, 50, 74, 42, 85, 53, 77, 45, 83, 51, 75, 43 },
    {   1, 97, 25,121,  7,103, 31,127,  0, 96, 24,120,  6,102, 30,126 },
    {  75, 33, 89, 57, 71, 39, 95, 63, 64, 32, 88, 56, 70, 38, 94, 62 },
    {  17,113,  9,105, 23,119, 15,111, 16,112,  8,104, 22,118, 14,110 },
    {  81, 49, 73, 41, 87, 55, 79, 47, 80, 48, 72, 40, 86, 54, 78, 46 },
    {   5,101, 29,125,  3, 99, 27,123,  4,100, 28,124,  2, 98, 26,122 },
    {  69, 37, 93, 61, 67, 35, 91, 59, 68, 36, 92, 60, 66, 34, 90, 58 },
    {  21,117, 13,109, 19,115, 11,107, 20,116, 12,108, 18,114, 10,106 },
    {  85, 53, 77, 45, 83, 51, 75, 43, 84, 52, 76, 44, 82, 50, 74, 42 }
  };
  dither16x16a_t() {};
} dither16x16a;

// 16->8
static const struct dither16x16_t
{
  // cycle: 16x. No special 16 byte sse2
  alignas(16) const BYTE data[16][16] = {
    {   0,192, 48,240, 12,204, 60,252,  3,195, 51,243, 15,207, 63,255 },
    { 128, 64,176,112,140, 76,188,124,131, 67,179,115,143, 79,191,127 },
    {  32,224, 16,208, 44,236, 28,220, 35,227, 19,211, 47,239, 31,223 },
    { 160, 96,144, 80,172,108,156, 92,163, 99,147, 83,175,111,159, 95 },
    {   8,200, 56,248,  4,196, 52,244, 11,203, 59,251,  7,199, 55,247 },
    { 136, 72,184,120,132, 68,180,116,139, 75,187,123,135, 71,183,119 },
    {  40,232, 24,216, 36,228, 20,212, 43,235, 27,219, 39,231, 23,215 },
    { 168,104,152, 88,164,100,148, 84,171,107,155, 91,167,103,151, 87 },
    {   2,194, 50,242, 14,206, 62,254,  1,193, 49,241, 13,205, 61,253 },
    { 130, 66,178,114,142, 78,190,126,129, 65,177,113,141, 77,189,125 },
    {  34,226, 18,210, 46,238, 30,222, 33,225, 17,209, 45,237, 29,221 },
    { 162, 98,146, 82,174,110,158, 94,161, 97,145, 81,173,109,157, 93 },
    {  10,202, 58,250,  6,198, 54,246,  9,201, 57,249,  5,197, 53,245 },
    { 138, 74,186,122,134, 70,182,118,137, 73,185,121,133, 69,181,117 },
    {  42,234, 26,218, 38,230, 22,214, 41,233, 25,217, 37,229, 21,213 },
    { 170,106,154, 90,166,102,150, 86,169,105,153, 89,165,101,149, 85 }
  };
  dither16x16_t() {};
} dither16x16;

#endif // __Convert_bits_H__
