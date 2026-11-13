/******************************************************
SuperEQ written by Naoki Shibata  shibatch@users.sourceforge.net

Shibatch Super Equalizer is a graphic and parametric equalizer plugin
for winamp. This plugin uses 16383th order FIR filter with FFT algorithm.
It's equalization is very precise. Equalization setting can be done
for each channel separately.


Homepage : http://shibatch.sourceforge.net/
e-mail   : shibatch@users.sourceforge.net

Some changes are from foobar2000 (www.foobar2000.org):

Copyright (c) 2001-2003, Peter Pawlowski
All rights reserved.

Other changes are:

Copyright (c) 2003, Klaus Post

*******************************************************/

#include <avisynth.h>
#include "ssrc-convert.h"
#include "supereq.h"

const AVS_Linkage * AVS_linkage = 0;
extern "C" AVS_PLUGIN_EXPORT const char* __stdcall AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors) {
  AVS_linkage = vectors;

  env->AddFunction("SSRC", "ci[fast]b", Create_SSRC, 0);

  env->AddFunction("SuperEQ", "cs", Create_SuperEq, 0);
  env->AddFunction("SuperEQ", "ci+", Create_SuperEqCustom, 0);

  return "`Shibatch' Audio equalization and high-quality resampling from Naoki Shibata";
}
