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


#include "fps.h"

#ifdef AVS_WINDOWS
    #include <avs/win.h>
#else
    #include <avs/posix.h>
#endif

#include <avs/minmax.h>
#include "../core/bitblt.h"
#include "../core/internal.h"
#include "merge.h"
#include <cmath>



/********************************************************************
***** Declare index of new filters for Avisynth's filter engine *****
********************************************************************/

extern const AVSFunction Fps_filters[] = {
  { "AssumeScaledFPS", BUILTIN_FUNC_PREFIX, "c[multiplier]i[divisor]i[sync_audio]b", AssumeScaledFPS::Create },

  { "AssumeFPS", BUILTIN_FUNC_PREFIX, "ci[]i[sync_audio]b", AssumeFPS::Create },      // dst framerate, sync audio?
  { "AssumeFPS", BUILTIN_FUNC_PREFIX, "cf[sync_audio]b", AssumeFPS::CreateFloat },    // dst framerate, sync audio?
  { "AssumeFPS", BUILTIN_FUNC_PREFIX, "cs[sync_audio]b", AssumeFPS::CreatePreset }, // dst framerate, sync audio?
  { "AssumeFPS", BUILTIN_FUNC_PREFIX, "cc[sync_audio]b", AssumeFPS::CreateFromClip }, // clip with dst framerate, sync audio?

  { "ChangeFPS", BUILTIN_FUNC_PREFIX, "ci[]i[linear]b", ChangeFPS::Create },     // dst framerate, fetch all frames
  { "ChangeFPS", BUILTIN_FUNC_PREFIX, "cf[linear]b", ChangeFPS::CreateFloat },   // dst framerate, fetch all frames
  { "ChangeFPS", BUILTIN_FUNC_PREFIX, "cs[linear]b", ChangeFPS::CreatePreset }, // dst framerate, fetch all frames
  { "ChangeFPS", BUILTIN_FUNC_PREFIX, "cc[linear]b", ChangeFPS::CreateFromClip },// clip with dst framerate, fetch all frames

  { "ConvertFPS", BUILTIN_FUNC_PREFIX, "ci[]i[zone]i[vbi]i", ConvertFPS::Create },      // dst framerate, zone lines, vbi lines
  { "ConvertFPS", BUILTIN_FUNC_PREFIX, "cf[zone]i[vbi]i", ConvertFPS::CreateFloat },    // dst framerate, zone lines, vbi lines
  { "ConvertFPS", BUILTIN_FUNC_PREFIX, "cs[zone]i[vbi]i", ConvertFPS::CreatePreset }, // dst framerate, zone lines, vbi lines
  { "ConvertFPS", BUILTIN_FUNC_PREFIX, "cc[zone]i[vbi]i", ConvertFPS::CreateFromClip }, // clip with dst framerate, zone lines, vbi lines

  { "ContinuedDenominator", BUILTIN_FUNC_PREFIX, "f[]i[limit]i", ContinuedCreate, (void*)0 },
  { "ContinuedNumerator",   BUILTIN_FUNC_PREFIX, "f[]i[limit]i", ContinuedCreate, (void*)1 },

  { NULL }
};



/*********************************************
 ******    Float and Rational utility   ******
 *********************************************

 *********************************************
 ******  Thanks to RAYMOD2 for his help  *****
 ******  in beating continued fractions  *****
 ******  into a usable form.       IanB  *****
 *********************************************/

// This function converts a 32-bit IEEE float into a fraction.  This
// process is lossless but it may fail for very large or very small
// floats.  It also discards the sign bit.  Since the denominator will
// always be a power of 2 and the numerator will always be odd (except
// when the denominator is 1) the fraction will already be reduced
// to its lowest terms. output range (2^32-2^(32-24))/1 -- (1/2^31)
// i.e. 4294967040/1 -- 1/2147483648 (4.65661287307e-10)
// returns true if input is out of range

static bool float_to_frac(float input, unsigned &num, unsigned &den)
{
  union { float f; unsigned i; } value;
  uint32_t mantissa;
  int exponent;

  // Bit strip the float
  value.f = input;
  mantissa = (value.i & 0x7FFFFF) + 0x800000;  // add implicit bit on the left
  exponent = ((value.i & 0x7F800000) >> 23) - 127 - 23;  // remove decimal pt

  // minimize the mantissa by removing any trailing 0's
  while (!(mantissa & 1)) { mantissa >>= 1; exponent += 1; }

  // if too small try to pull result from the reciprocal
  if (exponent < -31) {
    return float_to_frac((float)(1.0/input), den, num);
  }

  // if exponent is too large try removing leading 0's of mantissa
  while( (exponent > 0) && !(mantissa & 0x80000000) ) {
     mantissa <<= 1; exponent -= 1;
  }
  if (exponent > 0) {  // number too large
    num = 0xffffffff;
    den = 1;
    return true; // Out of range!
  }
  num = mantissa;
  den = 1 << (-exponent);

  return false;
}


// This function uses continued fractions to find the rational
// approximation such that the result truncated to a float is
// equal to value. The semiconvergents for the smallest such
// rational pair is then returned. The algorithm is modified
// from Wikipedia, Continued Fractions.

static bool reduce_float(float value, unsigned &num, unsigned &den)
{
  if (float_to_frac(value, num, den)) {
    return true;
  }

  uint32_t n0 = 0, n1 = 1, n2, nx = num;  // numerators
  uint32_t d0 = 1, d1 = 0, d2, dx = den;  // denominators
  uint32_t a2, ax, amin;  // integer parts of quotients
  uint32_t f1 = 0, f2;    // fractional parts of quotients

  for (;;)  // calculate convergents
  {
    a2 = nx / dx;
    f2 = nx % dx;
    n2 = n0 + n1 * a2;
    d2 = d0 + d1 * a2;

    if (f2 == 0) break;  // no more convergents (n2 / d2 == input)

    // Damn compiler does not correctly cast
    // to intermediate results to float
    float f = (float)((double)n2/d2);
    if (f == value) break;

    n0 = n1; n1 = n2;
    d0 = d1; d1 = d2;
    nx = dx; dx = f1 = f2;
  }
  if (d2 == 1)
  {
    num = n2;
    den = d2;
  }
  else { // we have been through the loop at least twice

    if ((a2 % 2 == 0) && (d0 * f1 > f2 * d1))
       amin = a2 / 2;  // passed 1/2 a_k admissibility test
    else
       amin = a2 / 2 + 1;

// find the sign of the error (actual + error = n2/d2) and then
// set (n2/d2) = (num/den + sign * epsilon) = R2 and solve for ax
//--------------------
//   n2 = n0 + n1 * ax
//   d2 = d0 + d1 * ax
//-----------------------------------------------
//   (n2/d2)       = R2     = (n0 + n1 * ax)/(d0 + d1 * ax)
//   n0 + n1 * ax           = R2 * (d0 + d1 * ax)
//   n0 + n1 * ax           = R2 * d0 + R2 * d1 * ax
//   R2 * d1 * ax - n1 * ax = n0 - R2 * d0
//   (R2 * d1 - n1) * ax    = n0 - R2 * d0
//   ax                     = (n0 - R2 * d0)/(R2 * d1 - n1)

    // bump float to adjacent float value
    union { float f; uint32_t i; } eps; eps.f = value;
    if (UInt32x32To64(n1, den) > UInt32x32To64(num, d1)) {
      eps.i -= 1;
    } else {
      eps.i += 1;
    }
    double r2 = eps.f;
    r2 += value;
    r2 /= 2;

    double yn = n0 - r2*d0;
    double yd = r2*d1 - n1;
    ax = (uint32_t)((yn + yd)/yd); // ceiling value

    if (ax < amin) ax = amin;

    // calculate nicest semiconvergent
    num = n0 + n1 * ax;
    den = d0 + d1 * ax;
  }
  return false;
}


// This function uses continued fractions to find the best rational
// approximation that satisfies (denom <= limit).  The algorithm
// is from Wikipedia, Continued Fractions.
//
static void reduce_frac(uint32_t &num, uint32_t &den, uint32_t limit)
{
  uint32_t n0 = 0, n1 = 1, n2, nx = num;    // numerators
  uint32_t d0 = 1, d1 = 0, d2, dx = den;  // denominators
  uint32_t a2, ax, amin;  // integer parts of quotients
  uint32_t f1 = 0, f2;        // fractional parts of quotients
  int i = 0;  // number of loop iterations

  for (;;) { // calculate convergents
    a2 = nx / dx;
    f2 = nx % dx;
    n2 = n0 + n1 * a2;
    d2 = d0 + d1 * a2;

    if (f2 == 0) break;
    if ((i++) && (d2 >= limit)) break;

    n0 = n1; n1 = n2;
    d0 = d1; d1 = d2;
    nx = dx; dx = f1 = f2;
  }
  if (d2 <= limit)
  {
    num = n2; den = d2;  // use last convergent
  }
  else { // (d2 > limit)
    // d2 = d0 + d1 * ax
    // d1 * ax = d2 - d1
    ax = (limit - d0) / d1;  // set d2 = limit and solve for a2

    if ((a2 % 2 == 0) && (d0 * f1 > f2 * d1))
      amin = a2 / 2;  // passed 1/2 a_k admissibility test
    else
      amin = a2 / 2 + 1;

    if (ax < amin) {
      // use previous convergent
      num = n1;
      den = d1;
    }
    else {
      // calculate best semiconvergent
      num = n0 + n1 * ax;
      den = d0 + d1 * ax;
    }
  }
}


AVSValue __cdecl ContinuedCreate(AVSValue args, void* key, IScriptEnvironment* env)
{
  uint32_t num, den;

  if (args[1].IsInt()) { // num, den[, limit] form
    if (args[0].IsInt()) {
      num = args[0].AsInt();
    } else { // IsFloat
      num = (uint32_t)args[0].AsFloat();
      if ((float)num != args[0].AsFloatf()) {
        env->ThrowError("ContinuedFraction: Numerator must be an integer.\n");
      }
    }
    den = args[1].AsInt();
    reduce_frac(num, den, (uint32_t)args[2].AsInt(1001));
  } else { // float[, limit] form
    if (args[2].IsInt()) {
      if (float_to_frac(args[0].AsFloatf(), num, den)) {
        env->ThrowError("ContinuedFraction: Float value out of range for rational pair.\n");
      }
      reduce_frac(num, den, (uint32_t)args[2].AsInt());
    } else {
      if (reduce_float(args[0].AsFloatf(), num, den)) {
        env->ThrowError("ContinuedFraction: Float value out of range for rational pair.\n");
      }
    }
  }
  return AVSValue((int)(key ? num : den));
}


/***************************************
 *******   Float to FPS utility   ******
 ***************************************/

void FloatToFPS(const char *name, float n, uint32_t &num, uint32_t &den, IScriptEnvironment* env)
{
  if (n <= 0)
    env->ThrowError("%s: FPS must be greater then 0.\n", name);

  float x;
  uint32_t u = (uint32_t)(n*1001+0.5);

  // Check for the 30000/1001 multiples
  x = (float)((u/30000*30000)/1001.0);
  if (x == n) { num = u; den= 1001; return; }

  // Check for the 24000/1001 multiples
  x = (float)((u/24000*24000)/1001.0);
  if (x == n) { num = u; den= 1001; return; }

  if (n < 14.986) {
    // Check for the 30000/1001 factors
    u = (uint32_t)(30000/n+0.5);
    x = (float)(30000.0/(u/1001*1001));
    if (x == n) { num = 30000; den= u; return; }

    // Check for the 24000/1001 factors
    u = (uint32_t)(24000/n+0.5);
    x = (float)(24000.0/(u/1001*1001));
    if (x == n) { num = 24000; den= u; return; }
  }

  // Find the rational pair with the smallest denominator
  // that is equal to n within the accuracy of an IEEE float.
  if (reduce_float(n, num, den))
    env->ThrowError("%s: FPS value is out of range.\n", name);

}


/****************************************
 *******   Preset to FPS utility   ****** -- Tritcal, IanB Jan 2006
 ****************************************/

void PresetToFPS(const char *name, const char *p, uint32_t &num, uint32_t &den, IScriptEnvironment* env)
{
	if (lstrcmpi(p, "ntsc_film"        ) == 0) { num = 24000; den = 1001; }
	else if (lstrcmpi(p, "ntsc_video"       ) == 0) { num = 30000; den = 1001; }
	else if (lstrcmpi(p, "ntsc_double"      ) == 0) { num = 60000; den = 1001; }
	else if (lstrcmpi(p, "ntsc_quad"        ) == 0) { num =120000; den = 1001; }

	else if (lstrcmpi(p, "ntsc_round_film"  ) == 0) { num =  2997; den =  125; }
	else if (lstrcmpi(p, "ntsc_round_video" ) == 0) { num =  2997; den =  100; }
	else if (lstrcmpi(p, "ntsc_round_double") == 0) { num =  2997; den =   50; }
	else if (lstrcmpi(p, "ntsc_round_quad"  ) == 0) { num =  2997; den =   25; }

	else if (lstrcmpi(p, "film"             ) == 0) { num =    24; den =    1; }

	else if (lstrcmpi(p, "pal_film"         ) == 0) { num =    25; den =    1; }
	else if (lstrcmpi(p, "pal_video"        ) == 0) { num =    25; den =    1; }
	else if (lstrcmpi(p, "pal_double"       ) == 0) { num =    50; den =    1; }
	else if (lstrcmpi(p, "pal_quad"         ) == 0) { num =   100; den =    1; }

	else if (lstrcmpi(p, "drop24"           ) == 0) { num = 24000; den = 1001; }
	else if (lstrcmpi(p, "drop30"           ) == 0) { num = 30000; den = 1001; }
	else if (lstrcmpi(p, "drop60"           ) == 0) { num = 60000; den = 1001; }
	else if (lstrcmpi(p, "drop120"          ) == 0) { num =120000; den = 1001; }
/*
	else if (lstrcmpi(p, "drop25"           ) == 0) { num = 25000; den = 1001; }
	else if (lstrcmpi(p, "drop50"           ) == 0) { num = 50000; den = 1001; }
	else if (lstrcmpi(p, "drop100"          ) == 0) { num =100000; den = 1001; }

	else if (lstrcmpi(p, "nondrop24"        ) == 0) { num =    24; den =    1; }
	else if (lstrcmpi(p, "nondrop25"        ) == 0) { num =    25; den =    1; }
	else if (lstrcmpi(p, "nondrop30"        ) == 0) { num =    30; den =    1; }
	else if (lstrcmpi(p, "nondrop50"        ) == 0) { num =    50; den =    1; }
	else if (lstrcmpi(p, "nondrop60"        ) == 0) { num =    60; den =    1; }
	else if (lstrcmpi(p, "nondrop100"       ) == 0) { num =   100; den =    1; }
	else if (lstrcmpi(p, "nondrop120"       ) == 0) { num =   120; den =    1; }

	else if (lstrcmpi(p, "23.976"           ) == 0) { num = 24000; den = 1001; }
	else if (lstrcmpi(p, "23.976!"          ) == 0) { num =  2997; den =  125; }
	else if (lstrcmpi(p, "24.0"             ) == 0) { num =    24; den =    1; }
	else if (lstrcmpi(p, "25.0"             ) == 0) { num =    25; den =    1; }
	else if (lstrcmpi(p, "29.97"            ) == 0) { num = 30000; den = 1001; }
	else if (lstrcmpi(p, "29.97!"           ) == 0) { num =  2997; den =  100; }
	else if (lstrcmpi(p, "30.0"             ) == 0) { num =    30; den =    1; }
	else if (lstrcmpi(p, "59.94"            ) == 0) { num = 60000; den = 1001; }
	else if (lstrcmpi(p, "59.94!"           ) == 0) { num =  2997; den =   50; }
	else if (lstrcmpi(p, "60.0"             ) == 0) { num =    60; den =    1; }
	else if (lstrcmpi(p, "100.0"            ) == 0) { num =   100; den =    1; }
	else if (lstrcmpi(p, "119.88"           ) == 0) { num =120000; den = 1001; }
	else if (lstrcmpi(p, "119.88!"          ) == 0) { num =  2997; den =   25; }
	else if (lstrcmpi(p, "120.0"            ) == 0) { num =   120; den =    1; }
*/
	else {
	  env->ThrowError("%s: invalid preset value used.\n", name);
	}
}



/******************************************
 *******   AssumeScaledFPS Filters   ******
 ******************************************/

AssumeScaledFPS::AssumeScaledFPS(PClip _child, int multiplier, int divisor, bool sync_audio, IScriptEnvironment* env)
 : NonCachedGenericVideoFilter(_child)
{
  if (divisor <= 0)
    env->ThrowError("AssumeScaledFPS: Divisor must be positive.");

  if (multiplier <= 0)
    env->ThrowError("AssumeScaledFPS: Multiplier must be positive.");

  if (sync_audio)
  {
    vi.audio_samples_per_second = MulDiv(vi.audio_samples_per_second, multiplier, divisor);
  }
  vi.MulDivFPS((uint32_t)multiplier, (uint32_t)divisor);
}


AVSValue __cdecl AssumeScaledFPS::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  return new AssumeScaledFPS( args[0].AsClip(), args[1].AsInt(1),
                        args[2].AsInt(1), args[3].AsBool(false), env );
}




/************************************
 *******   AssumeFPS Filters   ******
 ************************************/

AssumeFPS::AssumeFPS(PClip _child, unsigned numerator, unsigned denominator, bool sync_audio, IScriptEnvironment* env)
 : NonCachedGenericVideoFilter(_child)
{
  if (denominator == 0)
    env->ThrowError("AssumeFPS: Denominator cannot be 0 (zero).");

  if (sync_audio)
  {
    int64_t a = int64_t(vi.fps_numerator) * denominator;
    int64_t b = int64_t(vi.fps_denominator) * numerator;
    vi.audio_samples_per_second = int((vi.audio_samples_per_second * b + (a>>1)) / a);
  }
  vi.SetFPS(numerator, denominator);
}


AVSValue __cdecl AssumeFPS::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  return new AssumeFPS( args[0].AsClip(), args[1].AsInt(),
                        args[2].AsInt(1), args[3].AsBool(false), env );
}


AVSValue __cdecl AssumeFPS::CreateFloat(AVSValue args, void*, IScriptEnvironment* env)
{
  uint32_t num, den;

	FloatToFPS("AssumeFPS", args[1].AsFloatf(), num, den, env);
	return new AssumeFPS(args[0].AsClip(), num, den, args[2].AsBool(false), env);
}

// Tritical Jan 2006
AVSValue __cdecl AssumeFPS::CreatePreset(AVSValue args, void*, IScriptEnvironment* env)
{
  uint32_t num, den;

	PresetToFPS("AssumeFPS", args[1].AsString(), num, den, env);
	return new AssumeFPS(args[0].AsClip(), num, den, args[2].AsBool(false), env);
}

AVSValue __cdecl AssumeFPS::CreateFromClip(AVSValue args, void*, IScriptEnvironment* env)
{
  const VideoInfo& vi = args[1].AsClip()->GetVideoInfo();

  if (!vi.HasVideo()) {
    env->ThrowError("AssumeFPS: The clip supplied to get the FPS from must contain video.");
  }

  return new AssumeFPS( args[0].AsClip(), vi.fps_numerator,
                        vi.fps_denominator, args[2].AsBool(false), env );
}





/************************************
 *******   ChangeFPS Filters   ******
 ************************************/


ChangeFPS::ChangeFPS(PClip _child, unsigned new_numerator, unsigned new_denominator, bool _linear, IScriptEnvironment* env)
  : GenericVideoFilter(_child), linear(_linear)
{
  if (new_denominator == 0)
    env->ThrowError("ChangeFPS: Denominator cannot be 0 (zero).");

  a = int64_t(vi.fps_numerator) * new_denominator;
  b = int64_t(vi.fps_denominator) * new_numerator;
  if (linear && (a + (b>>1))/b > 10)
    env->ThrowError("ChangeFPS: Ratio must be less than 10 for linear access. Set LINEAR=False.");

  vi.SetFPS(new_numerator, new_denominator);
  const int64_t num_frames = (vi.num_frames * b + (a >> 1)) / a;
  if (num_frames > 0x7FFFFFFF)  // MAXINT
    env->ThrowError("ChangeFPS: Maximum number of frames exceeded.");

  vi.num_frames = int(num_frames);
  lastframe = -1;
}


PVideoFrame __stdcall ChangeFPS::GetFrame(int n, IScriptEnvironment* env)
{
  int getframe = int((n * a) / b); // Use Floor! - Which frame to get next?

  if (linear) {
    if ((lastframe < (getframe-1)) && (getframe - lastframe < 10)) {  // Do not decode more than 10 frames
      while (lastframe < (getframe-1)) {
        lastframe++;
        PVideoFrame p = child->GetFrame(lastframe, env);  // If MSVC optimizes this I'll kill it ;)
      }
    }
  }

  lastframe = getframe;
  return child->GetFrame(getframe , env );
}


bool __stdcall ChangeFPS::GetParity(int n)
{
  return child->GetParity(int((n * a) / b)); // Use Floor!
}


AVSValue __cdecl ChangeFPS::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  return new ChangeFPS(args[0].AsClip(), args[1].AsInt(), args[2].AsInt(1), args[3].AsBool(true), env);
}


AVSValue __cdecl ChangeFPS::CreateFloat(AVSValue args, void*, IScriptEnvironment* env)
{
  uint32_t num, den;

  FloatToFPS("ChangeFPS", args[1].AsFloatf(), num, den, env);
  return new ChangeFPS(args[0].AsClip(), num, den, args[2].AsBool(true), env);
}

// Tritical Jan 2006
AVSValue __cdecl ChangeFPS::CreatePreset(AVSValue args, void*, IScriptEnvironment* env)
{
  uint32_t num, den;

  PresetToFPS("ChangeFPS", args[1].AsString(), num, den, env);
  return new ChangeFPS(args[0].AsClip(), num, den, args[2].AsBool(true), env);
}

AVSValue __cdecl ChangeFPS::CreateFromClip(AVSValue args, void*, IScriptEnvironment* env)
{
  const VideoInfo& vi = args[1].AsClip()->GetVideoInfo();

  if (!vi.HasVideo()) {
    env->ThrowError("ChangeFPS: The clip supplied to get the FPS from must contain video.");
  }

  return new ChangeFPS(args[0].AsClip(), vi.fps_numerator, vi.fps_denominator,
    args[2].AsBool(true), env);
}







/*************************************
 *******   ConvertFPS Filters   ******
 *************************************/

ConvertFPS::ConvertFPS(PClip _child, unsigned new_numerator, unsigned new_denominator, int _zone,
  int _vbi, IScriptEnvironment* env)
  : GenericVideoFilter(_child), zone(_zone), vbi(_vbi), lps(0)
{
  if (zone >= 0 && !vi.IsYUY2()) // Tritical Jan 2006
    env->ThrowError("ConvertFPS: zone >= 0 requires YUY2 input");

  fa = int64_t(vi.fps_numerator) * new_denominator;
  fb = int64_t(vi.fps_denominator) * new_numerator;
  if (zone >= 0)
  {
    if (vbi < 0) vbi = 0;
    if (vbi > vi.height) vbi = vi.height;
    lps = int((vi.height + vbi) * fb / fa);
    if (zone > lps)
      env->ThrowError("ConvertFPS: 'zone' too large. Maximum allowed %d", lps);
  }
  else if (3 * fb < (fa << 1)) {
    int dec = MulDiv(vi.fps_numerator, 20000, vi.fps_denominator);
    env->ThrowError("ConvertFPS: New frame rate too small. Must be greater than %d.%04d "
      "Increase or use 'zone='", dec / 30000, (dec / 3) % 10000);
  }
  vi.SetFPS(new_numerator, new_denominator);
  const int64_t num_frames = (vi.num_frames * fb + (fa >> 1)) / fa;
  if (num_frames > 0x7FFFFFFF)  // MAXINT
    env->ThrowError("ConvertFPS: Maximum number of frames exceeded.");

  vi.num_frames = int(num_frames);
}


PVideoFrame __stdcall ConvertFPS::GetFrame(int n, IScriptEnvironment* env)
{
  // Using int64 modulo instead of modf, for double holds only 53 bits
  // n*fa worst-like case: (60000/1001 <> 30000/1001)
  // n = 0x7FFFFFFF; // 31 bits
  // fa = 1001 * 60000ULL // 26 bits
  // summa 57 bits, too much, modf((double)n * fa / fb, &nsrc_f); is not enough
  int64_t modulo = (n * fa) % fb;
  double frac_f = (double)modulo / fb;
  int nsrc = int(n * fa / fb);

  if (zone < 0) {

    // Mode 1: Blend full frames

    constexpr double threshold_f = 1.0 / 16.0;
    // was: 1 << (resolution - 4); // 64/1024

    // Don't bother if the blend ratio is small
    if (frac_f < threshold_f)
      return child->GetFrame(nsrc, env);

    if (frac_f > 1.0 - threshold_f)
      return child->GetFrame(nsrc + 1, env);

    PVideoFrame a = child->GetFrame(nsrc, env);
    PVideoFrame b = child->GetFrame(nsrc + 1, env);

    env->MakeWritable(&a);

    const int planes_y[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
    const int planes_r[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
    const int *planes;

    int planeCount;
    planeCount = vi.IsPlanar() ? vi.NumComponents() : 1;
    planes = (!vi.IsPlanar() || vi.IsYUV() || vi.IsYUVA()) ? planes_y : planes_r;

    const int bits_per_pixel = vi.BitsPerComponent();
    for (int j = 0; j < planeCount; ++j)
    {
      const int plane = planes[j];
      const BYTE*  b_data = b->GetReadPtr(plane);
      int          b_pitch = b->GetPitch(plane);
      BYTE*        a_data = a->GetWritePtr(plane);
      int          a_pitch = a->GetPitch(plane);
      int          row_size = a->GetRowSize(plane);
      int          height = a->GetHeight(plane);

      int weight_i;
      int invweight_i;
      float weight = (float)frac_f; // between 0 and 1.0

#ifdef INTEL_INTRINSICS
      MergeFuncPtr weighted_merge_planar = getMergeFunc(bits_per_pixel, env->GetCPUFlags(), a_data, b_data, weight, /*out*/weight_i, /*out*/invweight_i);
#else
      MergeFuncPtr weighted_merge_planar = getMergeFunc(bits_per_pixel, a_data, b_data, weight, /*out*/weight_i, /*out*/invweight_i);
#endif
      weighted_merge_planar(a_data, b_data, a_pitch, b_pitch, row_size, height, weight, weight_i, invweight_i);
    }

    return a;

  }
  else {
	// Mode 2: Switch to next frame at the scan line corresponding to the source frame's timing.
	// If zone > 0, perform a gradual transition, i.e. blend one frame into the next
	// over the given number of lines.

		PVideoFrame a = child->GetFrame(nsrc, env);
		PVideoFrame b = child->GetFrame(nsrc+1, env);
		const BYTE*  b_data   = b->GetReadPtr();
		int          b_pitch  = b->GetPitch();
		const int    row_size = a->GetRowSize();
		const int    height   = a->GetHeight();


		BYTE *pd;
		const BYTE *pa, *pb, *a_data = a->GetReadPtr();
		int   a_pitch = a->GetPitch();

		int switch_line = (int)(lps * (1.0 - frac_f));
		int top = switch_line - (zone>>1);
		int bottom = switch_line + (zone>>1) - lps;
		if( bottom > 0 && nsrc > 0 ) {
		// Finish the transition from the previous frame
			switch_line -= lps;
			top -= lps;
			nsrc--;
			b = a;
			a = child->GetFrame( nsrc, env );
			b_pitch = a_pitch;
			b_data  = a_data;
			a_data  = a->GetReadPtr();
			a_pitch = a->GetPitch();
		} else if( top >= height )
			return a;

		// Result goes into a new buffer since it can be made up of a number of source frames
    PVideoFrame d = env->NewVideoFrameP(vi, &a);
		BYTE* data   = d->GetWritePtr();
		const int      pitch  = d->GetPitch();
		if( top > 0 )
			env->BitBlt( data, pitch, a_data, a_pitch, row_size, top );
loop:
		bottom = min( switch_line + (zone>>1), height );
		int safe_top = max(top,0);
		pd = data   + safe_top * pitch;
		pa = a_data + safe_top * a_pitch;
		pb = b_data + safe_top * b_pitch;
		for( int y = safe_top; y < bottom; y++ ) {
			int scale = y - top;
			for( int x = 0; x < row_size; x++ )
				pd[x] = BYTE(pa[x] + ((pb[x] - pa[x]) * scale + (zone>>1)) / zone);
			pd += pitch;
			pa += a_pitch;
			pb += b_pitch;
		}
		switch_line += lps;
		top = switch_line - (zone>>1);
		int limit = min(height,top);
		if( bottom < limit ) {
			pd = data   + bottom * pitch;
			pb = b_data + bottom * b_pitch;
			env->BitBlt( pd, pitch, pb, b_pitch, row_size, limit-bottom );
		}
		if( top < height ) {
			nsrc++;
			a = b;
			b = child->GetFrame(nsrc+1, env);
			a_pitch = b_pitch;
			b_pitch = b->GetPitch();
			a_data  = b_data;
			b_data  = b->GetReadPtr();
			goto loop;
		}
		return d;
	}
}


bool __stdcall ConvertFPS::GetParity(int n)
{
	if( vi.IsFieldBased())
		return child->GetParity(0) ^ (n&1);
	else
		return child->GetParity(0);
}

AVSValue __cdecl ConvertFPS::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  return new ConvertFPS( args[0].AsClip(), args[1].AsInt(), args[2].AsInt(1),
                         args[3].AsInt(-1), args[4].AsInt(0), env );
}


AVSValue __cdecl ConvertFPS::CreateFloat(AVSValue args, void*, IScriptEnvironment* env)
{
  uint32_t num, den;

	FloatToFPS("ConvertFPS", (float)args[1].AsFloat(), num, den, env);
	return new ConvertFPS( args[0].AsClip(), num, den, args[2].AsInt(-1), args[3].AsInt(0), env );
}

// Tritical Jan 2006
AVSValue __cdecl ConvertFPS::CreatePreset(AVSValue args, void*, IScriptEnvironment* env)
{
  uint32_t num, den;

	PresetToFPS("ConvertFPS", args[1].AsString(), num, den, env);
	return new ConvertFPS(args[0].AsClip(), num, den, args[2].AsInt(-1), args[3].AsInt(0), env);
}

AVSValue __cdecl ConvertFPS::CreateFromClip(AVSValue args, void*, IScriptEnvironment* env)
{
  const VideoInfo& vi = args[1].AsClip()->GetVideoInfo();

  if (!vi.HasVideo()) {
    env->ThrowError("ConvertFPS: The clip supplied to get the FPS from must contain video.");
  }

  return new ConvertFPS( args[0].AsClip(), vi.fps_numerator, vi.fps_denominator,
                         args[2].AsInt(-1), args[3].AsInt(0), env );
}
