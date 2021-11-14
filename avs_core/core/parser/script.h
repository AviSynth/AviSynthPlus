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

#ifndef __Script_H__
#define __Script_H__

#include <avisynth.h>
#ifdef AVS_WINDOWS
#include <avs/win.h>
#include <tchar.h>
#else
#include <avs/posix.h>
#endif

#include "../internal.h"
#include "expression.h"
#include "scriptparser.h"


/********************************************************************
********************************************************************/

class ScriptFunction
  /**
    * Executes a script
   **/
{
public:
  ScriptFunction(const PExpression& _body, const bool* _param_floats, const char** _param_names, int param_count);
  virtual ~ScriptFunction()
  {
    delete[] param_floats;
    delete[] param_names;
  }

  static AVSValue Execute(AVSValue args, void* user_data, IScriptEnvironment* env);
  static void Delete(void* self, IScriptEnvironment*);

private:
  const PExpression body;
  bool* param_floats;
  const char** param_names;
};


/****    Helper functions   ****/

AVSValue Assert(AVSValue args, void*, IScriptEnvironment* env);
AVSValue AssertEval(AVSValue args, void*, IScriptEnvironment* env);
AVSValue Eval(AVSValue args, void*, IScriptEnvironment* env);
AVSValue EvalOop(AVSValue args, void*, IScriptEnvironment* env);
AVSValue Apply(AVSValue args, void*, IScriptEnvironment* env) ;

AVSValue Import(AVSValue args, void*, IScriptEnvironment* env);

AVSValue SetMemoryMax(AVSValue args, void*, IScriptEnvironment* env);
AVSValue SetWorkingDir(AVSValue args, void*, IScriptEnvironment* env);

/*****   Entry/Factory Methods   ******/

AVSValue Muldiv(AVSValue args, void*, IScriptEnvironment* env);

AVSValue Floor(AVSValue args, void*, IScriptEnvironment* env);
AVSValue Ceil(AVSValue args, void*, IScriptEnvironment* env);
AVSValue Round(AVSValue args, void*, IScriptEnvironment* env);

AVSValue Acos(AVSValue args, void* user_data, IScriptEnvironment* env);
AVSValue Asin(AVSValue args, void* user_data, IScriptEnvironment* env);
AVSValue Atan(AVSValue args, void* user_data, IScriptEnvironment* env);
AVSValue Atan2(AVSValue args, void* user_data, IScriptEnvironment* env);
AVSValue Cos(AVSValue args, void* user_data, IScriptEnvironment* env);
AVSValue Cosh(AVSValue args, void* user_data, IScriptEnvironment* env);
AVSValue Exp(AVSValue args, void* user_data, IScriptEnvironment* env);
AVSValue Fmod(AVSValue args, void* user_data, IScriptEnvironment* env);
AVSValue Log(AVSValue args, void* user_data, IScriptEnvironment* env);
AVSValue Log10(AVSValue args, void* user_data, IScriptEnvironment* env);
AVSValue Pow(AVSValue args, void* user_data, IScriptEnvironment* env);
AVSValue Sin(AVSValue args, void* user_data, IScriptEnvironment* env);
AVSValue Sinh(AVSValue args, void* user_data, IScriptEnvironment* env);
AVSValue Tan(AVSValue args, void* user_data, IScriptEnvironment* env);
AVSValue Tanh(AVSValue args, void* user_data, IScriptEnvironment* env);
AVSValue Sqrt(AVSValue args, void* user_data, IScriptEnvironment* env);

AVSValue Abs(AVSValue args, void* user_data, IScriptEnvironment* env);
AVSValue FAbs(AVSValue args, void* user_data, IScriptEnvironment* env);
AVSValue Pi(AVSValue args, void* user_data, IScriptEnvironment* env);
#ifdef OPT_ScriptFunctionTau
AVSValue Tau(AVSValue args, void* user_data, IScriptEnvironment* env);
#endif
AVSValue Sign(AVSValue args, void*, IScriptEnvironment* env);

AVSValue PixelType (AVSValue args, void*, IScriptEnvironment* env);
AVSValue Width(AVSValue args, void*, IScriptEnvironment* env);
AVSValue Height(AVSValue args, void*, IScriptEnvironment* env);
AVSValue FrameCount(AVSValue args, void*, IScriptEnvironment* env);
AVSValue FrameRate(AVSValue args, void*, IScriptEnvironment* env);
AVSValue FrameRateNumerator(AVSValue args, void*, IScriptEnvironment* env);
AVSValue FrameRateDenominator(AVSValue args, void*, IScriptEnvironment* env);
AVSValue AudioRate(AVSValue args, void*, IScriptEnvironment* env);
AVSValue AudioLength(AVSValue args, void*, IScriptEnvironment* env);
AVSValue AudioLengthLo(AVSValue args, void*, IScriptEnvironment* env);
AVSValue AudioLengthHi(AVSValue args, void*, IScriptEnvironment* env);
AVSValue AudioLengthS(AVSValue args, void*, IScriptEnvironment* env);
AVSValue AudioLengthF(AVSValue args, void*, IScriptEnvironment* env);
AVSValue AudioDuration(AVSValue args, void*, IScriptEnvironment* env);
AVSValue AudioChannels(AVSValue args, void*, IScriptEnvironment* env);
AVSValue AudioBits(AVSValue args, void*, IScriptEnvironment* env);
AVSValue IsAudioFloat(AVSValue args, void*, IScriptEnvironment* env);
AVSValue IsAudioInt(AVSValue args, void*, IScriptEnvironment* env);

AVSValue IsRGB(AVSValue args, void*, IScriptEnvironment* env);
AVSValue IsY8(AVSValue args, void*, IScriptEnvironment* env);
AVSValue IsYV12(AVSValue args, void*, IScriptEnvironment* env);
AVSValue IsYV16(AVSValue args, void*, IScriptEnvironment* env);
AVSValue IsYV24(AVSValue args, void*, IScriptEnvironment* env);
AVSValue IsYV411(AVSValue args, void*, IScriptEnvironment* env);
AVSValue IsYUY2(AVSValue args, void*, IScriptEnvironment* env);
AVSValue IsYUV(AVSValue args, void*, IScriptEnvironment* env);
AVSValue IsRGB24(AVSValue args, void*, IScriptEnvironment* env);
AVSValue IsRGB32(AVSValue args, void*, IScriptEnvironment* env);
AVSValue IsPlanar(AVSValue args, void*, IScriptEnvironment* env);
AVSValue IsInterleaved(AVSValue args, void*, IScriptEnvironment* env);
AVSValue IsFieldBased(AVSValue args, void*, IScriptEnvironment* env);
AVSValue IsFrameBased(AVSValue args, void*, IScriptEnvironment* env);
AVSValue GetParity(AVSValue args, void*, IScriptEnvironment* env);
AVSValue String(AVSValue args, void*, IScriptEnvironment* env);
AVSValue Hex(AVSValue args, void*, IScriptEnvironment* env);
AVSValue Func(AVSValue args, void*, IScriptEnvironment* env);
AVSValue FormatString(AVSValue args, void*, IScriptEnvironment* env);

AVSValue IsBool(AVSValue args, void*, IScriptEnvironment* env);
AVSValue IsInt(AVSValue args, void*, IScriptEnvironment* env);
AVSValue IsFloat(AVSValue args, void*, IScriptEnvironment* env);
AVSValue IsString(AVSValue args, void*, IScriptEnvironment* env);
AVSValue IsClip(AVSValue args, void*, IScriptEnvironment* env);
AVSValue IsFunction(AVSValue args, void*, IScriptEnvironment* env);
AVSValue Defined(AVSValue args, void*, IScriptEnvironment* env);

AVSValue TypeName(AVSValue args, void*, IScriptEnvironment* env);

AVSValue Default(AVSValue args, void*, IScriptEnvironment* env);

AVSValue VersionNumber(AVSValue args, void*, IScriptEnvironment* env);
AVSValue VersionString(AVSValue args, void*, IScriptEnvironment* env);
AVSValue IsVersionOrGreater(AVSValue args, void*, IScriptEnvironment* env);

AVSValue Int(AVSValue args, void*, IScriptEnvironment* env);
AVSValue Frac(AVSValue args, void*, IScriptEnvironment* env);
AVSValue Float(AVSValue args, void*,IScriptEnvironment* env);
AVSValue Value(AVSValue args, void*, IScriptEnvironment* env);
AVSValue HexValue(AVSValue args, void*, IScriptEnvironment* env);

AVSValue BitAnd(AVSValue args, void*, IScriptEnvironment* env);
AVSValue BitNot(AVSValue args, void*, IScriptEnvironment* env);
AVSValue BitOr(AVSValue args, void*, IScriptEnvironment* env);
AVSValue BitXor(AVSValue args, void*, IScriptEnvironment* env);

AVSValue BitLShift(AVSValue args, void*, IScriptEnvironment* env);
AVSValue BitRShiftL(AVSValue args, void*, IScriptEnvironment* env);
AVSValue BitRShiftA(AVSValue args, void*, IScriptEnvironment* env);

AVSValue BitRotateL(AVSValue args, void*, IScriptEnvironment* env);
AVSValue BitRotateR(AVSValue args, void*, IScriptEnvironment* env);

AVSValue BitChg(AVSValue args, void*, IScriptEnvironment* env);
AVSValue BitClr(AVSValue args, void*, IScriptEnvironment* env);
AVSValue BitSet(AVSValue args, void*, IScriptEnvironment* env);
AVSValue BitTst(AVSValue args, void*, IScriptEnvironment* env);
AVSValue BitSetCount(AVSValue args, void*, IScriptEnvironment* env);

AVSValue UCase(AVSValue args, void*, IScriptEnvironment* env);
AVSValue LCase(AVSValue args, void*, IScriptEnvironment* env);
AVSValue StrLen(AVSValue args, void*, IScriptEnvironment* env);
AVSValue RevStr(AVSValue args, void*, IScriptEnvironment* env);
AVSValue LeftStr(AVSValue args, void*, IScriptEnvironment* env);
AVSValue MidStr(AVSValue args, void*, IScriptEnvironment* env);
AVSValue RightStr(AVSValue args, void*, IScriptEnvironment* env);
AVSValue FindStr(AVSValue args, void*, IScriptEnvironment* env);
AVSValue FillStr(AVSValue args, void*, IScriptEnvironment* env);
AVSValue StrCmp(AVSValue args, void*, IScriptEnvironment* env);
AVSValue StrCmpi(AVSValue args, void*, IScriptEnvironment* env);

AVSValue TrimAll(AVSValue args, void*, IScriptEnvironment* env);
AVSValue TrimLeft(AVSValue args, void*, IScriptEnvironment* env);
AVSValue TrimRight(AVSValue args, void*, IScriptEnvironment* env);

AVSValue Rand(AVSValue args, void* user_data, IScriptEnvironment* env);

AVSValue Select(AVSValue args, void*, IScriptEnvironment* env);

AVSValue NOP(AVSValue args, void*, IScriptEnvironment* env);

AVSValue Undefined(AVSValue args, void*, IScriptEnvironment* env);

AVSValue Exist(AVSValue args, void*, IScriptEnvironment* env);

// WE ->
AVSValue AVSChr(AVSValue args, void*, IScriptEnvironment* env);
AVSValue AVSOrd(AVSValue args, void*, IScriptEnvironment* env);
AVSValue AVSTime(AVSValue args, void*, IScriptEnvironment* env);
AVSValue Spline(AVSValue args, void*, IScriptEnvironment* env);
// WE <-

AVSValue HasVideo(AVSValue args, void*, IScriptEnvironment* env);
AVSValue HasAudio(AVSValue args, void*, IScriptEnvironment* env);

AVSValue AvsMin(AVSValue args, void*, IScriptEnvironment* env);
AVSValue AvsMax(AVSValue args, void*, IScriptEnvironment* env);

AVSValue ScriptName(AVSValue args, void*, IScriptEnvironment* env);
AVSValue ScriptFile(AVSValue args, void*, IScriptEnvironment* env);
AVSValue ScriptDir (AVSValue args, void*, IScriptEnvironment* env);
AVSValue ScriptNameUtf8(AVSValue args, void*, IScriptEnvironment* env);
AVSValue ScriptFileUtf8(AVSValue args, void*, IScriptEnvironment* env);
AVSValue ScriptDirUtf8(AVSValue args, void*, IScriptEnvironment* env);

AVSValue AddAutoloadDir (AVSValue args, void*, IScriptEnvironment* env);
AVSValue ClearAutoloadDirs (AVSValue args, void*, IScriptEnvironment* env);
AVSValue AutoloadPlugins (AVSValue args, void*, IScriptEnvironment* env);
AVSValue FunctionExists (AVSValue args, void*, IScriptEnvironment* env);
AVSValue InternalFunctionExists (AVSValue args, void*, IScriptEnvironment* env);

AVSValue SetFilterMTMode (AVSValue args, void*, IScriptEnvironment* env);
AVSValue SetLogParams(AVSValue args, void*, IScriptEnvironment* env);
AVSValue LogMsg(AVSValue args, void*, IScriptEnvironment* env);

AVSValue SetCacheMode(AVSValue args, void*, IScriptEnvironment* env);
AVSValue SetDeviceOpt(AVSValue args, void*, IScriptEnvironment* env);
AVSValue SetMemoryMax(AVSValue args, void*, IScriptEnvironment* env);
AVSValue SetMaxCPU(AVSValue args, void*, IScriptEnvironment* env); // 20200331

AVSValue IsY(AVSValue args, void*, IScriptEnvironment* env);
AVSValue Is420(AVSValue args, void*, IScriptEnvironment* env);
AVSValue Is422(AVSValue args, void*, IScriptEnvironment* env);
AVSValue Is444(AVSValue args, void*, IScriptEnvironment* env);
AVSValue IsRGB48(AVSValue args, void*, IScriptEnvironment* env);
AVSValue IsRGB64(AVSValue args, void*, IScriptEnvironment* env);
AVSValue ComponentSize(AVSValue args, void*, IScriptEnvironment* env);
AVSValue BitsPerComponent(AVSValue args, void*, IScriptEnvironment* env);
AVSValue IsYUVA(AVSValue args, void*, IScriptEnvironment* env);
AVSValue IsPlanarRGB(AVSValue args, void*, IScriptEnvironment* env);
AVSValue IsPlanarRGBA(AVSValue args, void*, IScriptEnvironment* env);
AVSValue ColorSpaceNameToPixelType(AVSValue args, void*, IScriptEnvironment* env);
AVSValue NumComponents(AVSValue args, void*, IScriptEnvironment* env);
AVSValue HasAlpha(AVSValue args, void*, IScriptEnvironment* env);
AVSValue IsPackedRGB(AVSValue args, void*, IScriptEnvironment* env);

AVSValue ReplaceStr(AVSValue args, void*, IScriptEnvironment* env); // avs+ 161230
AVSValue IsVideoFloat(AVSValue args, void*, IScriptEnvironment* env); // avs+ 170309

AVSValue GetProcessInfo(AVSValue args, void*, IScriptEnvironment* env); // avs+ 170526
AVSValue StrToUtf8(AVSValue args, void*, IScriptEnvironment* env); // avs+ 170601
AVSValue StrFromUtf8(AVSValue args, void*, IScriptEnvironment* env); // avs+ 170601

AVSValue IsFloatUvZeroBased(AVSValue args, void*, IScriptEnvironment* env); // avs+ 180516
AVSValue BuildPixelType(AVSValue args, void*, IScriptEnvironment* env); // avs+ 180517
AVSValue VarExist(AVSValue args, void*, IScriptEnvironment* env); // avs+ 180606

AVSValue ArrayCreate(AVSValue args, void*, IScriptEnvironment* env);
AVSValue IsArray(AVSValue args, void*, IScriptEnvironment* env);
AVSValue ArrayGet(AVSValue args, void*, IScriptEnvironment* env);
AVSValue ArraySize(AVSValue args, void*, IScriptEnvironment* env);
AVSValue ArrayIns(AVSValue args, void*, IScriptEnvironment* env);
AVSValue ArrayDel(AVSValue args, void*, IScriptEnvironment* env);

#endif  // __Script_H__
