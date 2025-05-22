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

#ifndef __Resample_Functions_H__
#define __Resample_Functions_H__

#include <avisynth.h>
#include "avs/alignment.h"
#include <vector>
#include <stdint.h>

// Original value: 65536
// 2 bits sacrificed because of 16 bit signed MMX multiplication
// NOTE: Don't change this value. It's hard-coded in SIMD code.
constexpr int FPScale8bits = 14; // fixed point scaler 14 bit
constexpr int FPScale = 1 << FPScale8bits; // fixed point scaler (1<<14)
// for 16 bits: one bit less
constexpr int FPScale16bits = 13;
constexpr int FPScale16 = 1 << FPScale16bits; // fixed point scaler for 10-16 bit SIMD signed operation
constexpr int ALIGN_RESIZER_TARGET_SIZE = 16; // 16: avx512 float Hoprizontal
// 09-14-2002 - Vlad59 - Lanczos3Resize - Constant added
#define M_PI 3.14159265358979323846

struct SafeLimit {
  bool overread_possible;
  int source_overread_offset;
  int source_overread_beyond_targetx;
};

struct ResamplingProgram {
  IScriptEnvironment * Env;
  int source_size, target_size;
  double crop_start, crop_size;
  int filter_size;
  int filter_size_real; // maybe less than filter_size if dimensions are small
  int filter_size_alignment; // for info, 1 (C, nonvector-friendly), 8 (sse or avx2) or 16 (avx2)
  int target_size_alignment; // coeff table exists (and containt zero coeffs) even beyond target_size. Helps alternative H resizers.

  // Array of Integer indicate starting point of sampling
  std::vector<int> pixel_offset;

  int bits_per_pixel;

  // Array of array of coefficient for each pixel
  // {{pixel[0]_coeff}, {pixel[1]_coeff}, ...}
  short* pixel_coefficient;
  float* pixel_coefficient_float;
  // Array of real kernel size, handles edge cases! <= filter_size
  // for SIMD, coefficients are copied over a padded aligned storage
  std::vector<short> kernel_sizes; 
  // 3.7.4- can be different for each line but then they get equalized and aligned.

  // in H resizers danger zone starts from here.
  // When reading multiple (SIMD load) source pixels from (src+offset) and it no
  // longer fits image scanline dimensions (width)
  SafeLimit safelimit_filter_size_aligned = { false, -1, -1 };
  SafeLimit safelimit_4_pixels = { false, -1, -1 };
  SafeLimit safelimit_8_pixels = { false, -1, -1 };
  SafeLimit safelimit_16_pixels = { false, -1, -1 };
  SafeLimit safelimit_32_pixels = { false, -1, -1 };

  ResamplingProgram(int filter_size, int source_size, int target_size, double crop_start, double crop_size, int bits_per_pixel, IScriptEnvironment* env)
    : Env(env), source_size(source_size), target_size(target_size), crop_start(crop_start), crop_size(crop_size), filter_size(filter_size), filter_size_real(filter_size),
    bits_per_pixel(bits_per_pixel), pixel_coefficient(0), pixel_coefficient_float(0)
  {

    
    filter_size_alignment = 1;
    // align target_size to 8 units to allow safe up to 8 pixels/cycle in H resizers. modded later.
    target_size_alignment = 1;
    // resize_prepare_coeff can override and realign the size of coefficient table
    if (bits_per_pixel < 32)
      pixel_coefficient = (short*)Env->Allocate(sizeof(short) * target_size * filter_size, 64, AVS_NORMAL_ALLOC);
    else
      pixel_coefficient_float = (float*)Env->Allocate(sizeof(float) * target_size * filter_size, 64, AVS_NORMAL_ALLOC);

    pixel_offset.resize(target_size);
    kernel_sizes.resize(target_size);

    if ((pixel_coefficient == nullptr && bits_per_pixel < 32) ||
        (pixel_coefficient_float == nullptr && bits_per_pixel == 32)) {
      Env->Free(pixel_coefficient);
      Env->Free(pixel_coefficient_float);
      Env->ThrowError("ResamplingProgram: Could not reserve memory.");
    }

  };

  ~ResamplingProgram() {
    Env->Free(pixel_coefficient);
    Env->Free(pixel_coefficient_float);
  };
};

typedef struct ResamplingProgram ResamplingProgram;


/*******************************************
   ***************************************
   **  Helper classes for resample.cpp  **
   ***************************************
 *******************************************/


class ResamplingFunction
/**
  * Pure virtual base class for resampling functions
  */
{
public:
  virtual double f(double x) = 0;
  virtual double support() = 0;

  virtual ResamplingProgram* GetResamplingProgram(int source_size, double crop_start, double crop_size, int target_size, int bits_per_pixel, 
    double center_pos_src, double center_pos_dst,
    IScriptEnvironment* env);
  virtual ~ResamplingFunction() = default;
  // virtual bool CheckValidity(int source_size, double crop_size, int target_size);
};

class PointFilter : public ResamplingFunction
/**
  * Nearest neighbour (point sampler), used in PointResize
 **/
{
public:
  double f(double x);
  double support() { return 0; }  
  // Pre 3.7.4 : 0.0001. Comment: 0.0 crashes it. 
  // 3.7.4- this 0 is specially handled in GetResamplingProgram
};


class TriangleFilter : public ResamplingFunction
/**
  * Simple triangle filter, used in BilinearResize
 **/
{
public:
  double f(double x);
  double support() { return 1.0; }
};


class MitchellNetravaliFilter : public ResamplingFunction
/**
  * Mitchell-Netraveli filter, used in BicubicResize
 **/
{
public:
  MitchellNetravaliFilter(double b = 1. / 3., double c = 1. / 3.);
  double f(double x);
  double support() { return 2.0; }

private:
  double p0,p2,p3,q0,q1,q2,q3;
};

class LanczosFilter : public ResamplingFunction
/**
  * Lanczos filter, used in LanczosResize
 **/
{
public:
  LanczosFilter(int _taps = 3);
	double f(double x);
	double support() { return taps; };

private:
	double sinc(double value);
  double taps;
};

class BlackmanFilter : public ResamplingFunction
/**
  * Blackman filter, used in BlackmanResize
 **/
{
public:
  BlackmanFilter(int _taps = 4);
	double f(double x);
	double support() { return taps; };

private:
  double taps, rtaps;
};

// Spline16
class Spline16Filter : public ResamplingFunction
/**
  * Spline16 of Panorama Tools is a cubic-spline, with derivative set to 0 at the edges (4x4 pixels).
 **/
{
public:
	double f(double x);
	double support() { return 2.0; };

private:
};

// Spline36
class Spline36Filter : public ResamplingFunction
/**
  * Spline36 is like Spline16,  except that it uses 6x6=36 pixels.
 **/
{
public:
	double f(double x);
	double support() { return 3.0; };

private:
};

// Spline64
class Spline64Filter : public ResamplingFunction
/**
  * Spline64 is like Spline36,  except that it uses 8x8=64 pixels.
 **/
{
public:
	double f(double x);
	double support() { return 4.0; };

private:
};


class GaussianFilter : public ResamplingFunction
/**
  * GaussianFilter, from swscale.
 **/
{
public:
  GaussianFilter(double p = 30.0, double _b = 2.0, double _s = 4.0);
  double f(double x);
  double support() { return s; }; // <3.7.4 was fixed at 4.0

private:
  double param;
  double b; // base value since 3.7.4
  double s; // variable support since 3.7.4
};

class SincFilter : public ResamplingFunction
/**
  * Sinc filter, used in SincResize
 **/
{
public:
  SincFilter(int _taps = 4);
	double f(double x);
	double support() { return taps; };

private:
  double taps;
};

class SinPowerFilter : public ResamplingFunction
  // SinPow kernel, used in SinPowResize
{
public:
  SinPowerFilter(double p = 2.5);
  double f(double x);
  double support() { return 2.0; }; // 2 very important, 4 cause bugs

private:
  double param;
};

class SincLin2Filter : public ResamplingFunction
  /**
  * SincLin2 filter, used in SincLin2Resize
  **/
{
public:
  SincLin2Filter(int _taps = 15);
  double f(double x);
  double support() { return taps; };

private:
  double sinc(double value);
  double taps;
};

class UserDefined2Filter : public ResamplingFunction
	/**
	  * User-defined by 2 samples filter, used in UDef2Resize
	 **/
{
public:
	UserDefined2Filter(double _b, double _c, double _s);
	double f(double x);
	double support() { return s; }

private:
	double sinc(double value);
  double a, b, c;
  double s; // 'support' as a parameter
};

#endif  // __Reample_Functions_H__
