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


#include "script.h"
#include <time.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <fstream>
#include <memory>

#ifdef AVS_WINDOWS
#include <io.h>
#include <avs/win.h>
#else
#include <avs/posix.h>
#include "os/win32_string_compat.h"
#include <dirent.h>
#endif

#include <avs/filesystem.h>
#include <avs/minmax.h>
#include <new>
#include "../internal.h"
#include "../Prefetcher.h"
#include "../InternalEnvironment.h"
#include "../strings.h"
#include <map>
#include <string>
#include <utility>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>

#ifndef MINGW_HAS_SECURE_API
#define sprintf_s sprintf
#endif



/********************************************************************
***** Declare index of new filters for Avisynth's filter engine *****
********************************************************************/


extern const AVSFunction Script_functions[] = {
  { "muldiv",   BUILTIN_FUNC_PREFIX, "iii", Muldiv },

  { "floor",    BUILTIN_FUNC_PREFIX, "f", Floor },
  { "ceil",     BUILTIN_FUNC_PREFIX, "f", Ceil },
  { "round",    BUILTIN_FUNC_PREFIX, "f", Round },

  { "acos",     BUILTIN_FUNC_PREFIX, "f", Acos },
  { "asin",     BUILTIN_FUNC_PREFIX, "f", Asin },
  { "atan",     BUILTIN_FUNC_PREFIX, "f", Atan },
  { "atan2",    BUILTIN_FUNC_PREFIX, "ff", Atan2 },
  { "cos",      BUILTIN_FUNC_PREFIX, "f", Cos },
  { "cosh",     BUILTIN_FUNC_PREFIX, "f", Cosh },
  { "exp",      BUILTIN_FUNC_PREFIX, "f", Exp },
  { "fmod",     BUILTIN_FUNC_PREFIX, "ff", Fmod },
  { "log",      BUILTIN_FUNC_PREFIX, "f", Log },
  { "log10",    BUILTIN_FUNC_PREFIX, "f", Log10 },
  { "pow",      BUILTIN_FUNC_PREFIX, "ff", Pow },
  { "sin",      BUILTIN_FUNC_PREFIX, "f", Sin },
  { "sinh",     BUILTIN_FUNC_PREFIX, "f", Sinh },
  { "tan",      BUILTIN_FUNC_PREFIX, "f", Tan },
  { "tanh",     BUILTIN_FUNC_PREFIX, "f", Tanh },
  { "sqrt",     BUILTIN_FUNC_PREFIX, "f", Sqrt },


  { "abs",      BUILTIN_FUNC_PREFIX, "i", Abs },
  { "abs",      BUILTIN_FUNC_PREFIX, "f", FAbs },
  { "pi",       BUILTIN_FUNC_PREFIX, "", Pi },
#ifdef OPT_ScriptFunctionTau
  { "tau",      BUILTIN_FUNC_PREFIX, "", Tau },
#endif
  { "sign",     BUILTIN_FUNC_PREFIX, "f",Sign},

  { "bitand",   BUILTIN_FUNC_PREFIX, "ii",BitAnd},
  { "bitnot",   BUILTIN_FUNC_PREFIX, "i",BitNot},
  { "bitor",    BUILTIN_FUNC_PREFIX, "ii",BitOr},
  { "bitxor",   BUILTIN_FUNC_PREFIX, "ii",BitXor},

  { "bitlshift",  BUILTIN_FUNC_PREFIX, "ii",BitLShift},
  { "bitlshiftl", BUILTIN_FUNC_PREFIX, "ii",BitLShift},
  { "bitlshifta", BUILTIN_FUNC_PREFIX, "ii",BitLShift},
  { "bitlshiftu", BUILTIN_FUNC_PREFIX, "ii",BitLShift},
  { "bitlshifts", BUILTIN_FUNC_PREFIX, "ii",BitLShift},
  { "bitshl",     BUILTIN_FUNC_PREFIX, "ii",BitLShift},
  { "bitsal",     BUILTIN_FUNC_PREFIX, "ii",BitLShift},

  { "bitrshiftl", BUILTIN_FUNC_PREFIX, "ii",BitRShiftL},
  { "bitrshifta", BUILTIN_FUNC_PREFIX, "ii",BitRShiftA},
  { "bitrshiftu", BUILTIN_FUNC_PREFIX, "ii",BitRShiftL},
  { "bitrshifts", BUILTIN_FUNC_PREFIX, "ii",BitRShiftA},
  { "bitshr",     BUILTIN_FUNC_PREFIX, "ii",BitRShiftL},
  { "bitsar",     BUILTIN_FUNC_PREFIX, "ii",BitRShiftA},

  { "bitlrotate", BUILTIN_FUNC_PREFIX, "ii",BitRotateL},
  { "bitrrotate", BUILTIN_FUNC_PREFIX, "ii",BitRotateR},
  { "bitrol",     BUILTIN_FUNC_PREFIX, "ii",BitRotateL},
  { "bitror",     BUILTIN_FUNC_PREFIX, "ii",BitRotateR},

  { "bitchg",    BUILTIN_FUNC_PREFIX, "ii",BitChg},
  { "bitchange", BUILTIN_FUNC_PREFIX, "ii",BitChg},
  { "bitclr",    BUILTIN_FUNC_PREFIX, "ii",BitClr},
  { "bitclear",  BUILTIN_FUNC_PREFIX, "ii",BitClr},
  { "bitset",    BUILTIN_FUNC_PREFIX, "ii",BitSet},
  { "bittst",    BUILTIN_FUNC_PREFIX, "ii",BitTst},
  { "bittest",   BUILTIN_FUNC_PREFIX, "ii",BitTst},
  { "bitsetcount", BUILTIN_FUNC_PREFIX, "i+",BitSetCount }, // avs+ 180221

  { "lcase",    BUILTIN_FUNC_PREFIX, "s",LCase},
  { "ucase",    BUILTIN_FUNC_PREFIX, "s",UCase},
  { "strlen",   BUILTIN_FUNC_PREFIX, "s",StrLen},
  { "revstr",   BUILTIN_FUNC_PREFIX, "s",RevStr},
  { "leftstr",  BUILTIN_FUNC_PREFIX, "si",LeftStr},
  { "midstr",   BUILTIN_FUNC_PREFIX, "si[length]i",MidStr},
  { "rightstr", BUILTIN_FUNC_PREFIX, "si",RightStr},
  { "findstr",  BUILTIN_FUNC_PREFIX, "ss",FindStr},
  { "fillstr",  BUILTIN_FUNC_PREFIX, "i[]s",FillStr},
  { "replacestr", BUILTIN_FUNC_PREFIX, "sss[sig]b",ReplaceStr}, // avs+ 161230, case 180222
  { "trimall",  BUILTIN_FUNC_PREFIX, "s",TrimAll }, // avs+ 180225 diff name of clip-function Trim
  { "trimleft", BUILTIN_FUNC_PREFIX, "s",TrimLeft }, // avs+ 180225
  { "trimright", BUILTIN_FUNC_PREFIX, "s",TrimRight }, // avs+ 180225

  { "strcmp",   BUILTIN_FUNC_PREFIX, "ss",StrCmp},
  { "strcmpi",  BUILTIN_FUNC_PREFIX, "ss",StrCmpi},

  { "rand",     BUILTIN_FUNC_PREFIX, "[max]i[scale]b[seed]b", Rand },

  { "Select",   BUILTIN_FUNC_PREFIX, "i.+", Select },

  { "nop",      BUILTIN_FUNC_PREFIX, "", NOP },
  { "undefined",BUILTIN_FUNC_PREFIX, "", Undefined },

  { "width",      BUILTIN_FUNC_PREFIX, "c", Width },
  { "height",     BUILTIN_FUNC_PREFIX, "c", Height },
  { "framecount", BUILTIN_FUNC_PREFIX, "c", FrameCount },
  { "framerate",  BUILTIN_FUNC_PREFIX, "c", FrameRate },
  { "frameratenumerator",   BUILTIN_FUNC_PREFIX, "c", FrameRateNumerator },
  { "frameratedenominator", BUILTIN_FUNC_PREFIX, "c", FrameRateDenominator },
  { "audiorate",     BUILTIN_FUNC_PREFIX, "c", AudioRate },
  { "audiolength",   BUILTIN_FUNC_PREFIX, "c", AudioLength },  // Fixme: Add int64 to script
  { "audiolengthlo", BUILTIN_FUNC_PREFIX, "c[]i", AudioLengthLo }, // audiolength%i
  { "audiolengthhi", BUILTIN_FUNC_PREFIX, "c[]i", AudioLengthHi }, // audiolength/i
  { "audiolengths",  BUILTIN_FUNC_PREFIX, "c", AudioLengthS }, // as a string
  { "audiolengthf",  BUILTIN_FUNC_PREFIX, "c", AudioLengthF }, // at least this will give an order of the size
  { "audioduration", BUILTIN_FUNC_PREFIX, "c", AudioDuration }, // In seconds
  { "audiochannels", BUILTIN_FUNC_PREFIX, "c", AudioChannels },
  { "audiobits",     BUILTIN_FUNC_PREFIX, "c", AudioBits },
  { "IsAudioFloat",  BUILTIN_FUNC_PREFIX, "c", IsAudioFloat },
  { "IsAudioInt",    BUILTIN_FUNC_PREFIX, "c", IsAudioInt },
  { "IsRGB",    BUILTIN_FUNC_PREFIX, "c", IsRGB },
  { "IsYUY2",   BUILTIN_FUNC_PREFIX, "c", IsYUY2 },
  { "IsYUV",    BUILTIN_FUNC_PREFIX, "c", IsYUV },
  { "IsY8",     BUILTIN_FUNC_PREFIX, "c", IsY8 },
  { "IsYV12",   BUILTIN_FUNC_PREFIX, "c", IsYV12 },
  { "IsYV16",   BUILTIN_FUNC_PREFIX, "c", IsYV16 },
  { "IsYV24",   BUILTIN_FUNC_PREFIX, "c", IsYV24 },
  { "IsYV411",  BUILTIN_FUNC_PREFIX, "c", IsYV411 },
  { "IsPlanar", BUILTIN_FUNC_PREFIX, "c", IsPlanar },
  { "IsInterleaved", BUILTIN_FUNC_PREFIX, "c", IsInterleaved },
  { "IsRGB24",       BUILTIN_FUNC_PREFIX, "c", IsRGB24 },
  { "IsRGB32",       BUILTIN_FUNC_PREFIX, "c", IsRGB32 },
  { "IsFieldBased",  BUILTIN_FUNC_PREFIX, "c", IsFieldBased },
  { "IsFrameBased",  BUILTIN_FUNC_PREFIX, "c", IsFrameBased },
  { "GetParity", BUILTIN_FUNC_PREFIX, "c[n]i", GetParity },
  { "String",    BUILTIN_FUNC_PREFIX, ".[]s", String },
  { "Hex",       BUILTIN_FUNC_PREFIX, "i[width]i", Hex }, // avs+ 20180222 new width parameter
  { "Func",    BUILTIN_FUNC_PREFIX, "n", Func },
  { "Format",  BUILTIN_FUNC_PREFIX, "s.*", FormatString },

  { "IsBool",   BUILTIN_FUNC_PREFIX, ".", IsBool },
  { "IsInt",    BUILTIN_FUNC_PREFIX, ".", IsInt },
  { "IsFloat",  BUILTIN_FUNC_PREFIX, ".", IsFloat },
  { "IsString", BUILTIN_FUNC_PREFIX, ".", IsString },
  { "IsClip",   BUILTIN_FUNC_PREFIX, ".", IsClip },
  { "IsFunction", BUILTIN_FUNC_PREFIX, ".", IsFunction },
  { "Defined",  BUILTIN_FUNC_PREFIX, ".", Defined },
  { "TypeName",  BUILTIN_FUNC_PREFIX, ".", TypeName },

  { "Default",  BUILTIN_FUNC_PREFIX, "..", Default },

  { "Eval",   BUILTIN_FUNC_PREFIX, "s[name]s", Eval },
  { "Eval",   BUILTIN_FUNC_PREFIX, "cs[name]s", EvalOop },
  { "Apply",  BUILTIN_FUNC_PREFIX, "s.*", Apply },
  { "Import", BUILTIN_FUNC_PREFIX, "s+[utf8]b", Import },

  { "Assert", BUILTIN_FUNC_PREFIX, "b[message]s", Assert },
  { "Assert", BUILTIN_FUNC_PREFIX, "s", AssertEval },

  { "SetMemoryMax", BUILTIN_FUNC_PREFIX, "[]i[type]i[index]i", SetMemoryMax }, // Neo
  { "SetWorkingDir", BUILTIN_FUNC_PREFIX, "s", SetWorkingDir },
  { "Exist",         BUILTIN_FUNC_PREFIX, "s[utf8]b", Exist },

  { "Chr",    BUILTIN_FUNC_PREFIX, "i", AVSChr },
  { "Ord",    BUILTIN_FUNC_PREFIX, "s", AVSOrd },
  { "Time",   BUILTIN_FUNC_PREFIX, "s", AVSTime },
  { "Spline", BUILTIN_FUNC_PREFIX, "[x]ff+[cubic]b", Spline },

  { "int",   BUILTIN_FUNC_PREFIX, "f", Int },
  { "frac",  BUILTIN_FUNC_PREFIX, "f", Frac},
  { "float", BUILTIN_FUNC_PREFIX, "f",Float},

  { "value",    BUILTIN_FUNC_PREFIX, "s",Value},
  { "hexvalue", BUILTIN_FUNC_PREFIX, "s[pos]i",HexValue}, // avs+ 20180222 new pos parameter

  { "VersionNumber", BUILTIN_FUNC_PREFIX, "", VersionNumber },
  { "VersionString", BUILTIN_FUNC_PREFIX, "", VersionString },
  { "IsVersionOrGreater", BUILTIN_FUNC_PREFIX, "[majorversion]i[minorVersion]i[bugfixVersion]i", IsVersionOrGreater },

  { "HasVideo", BUILTIN_FUNC_PREFIX, "c", HasVideo },
  { "HasAudio", BUILTIN_FUNC_PREFIX, "c", HasAudio },

  { "Min", BUILTIN_FUNC_PREFIX, "f+", AvsMin },
  { "Max", BUILTIN_FUNC_PREFIX, "f+", AvsMax },

  { "ScriptName", BUILTIN_FUNC_PREFIX, "", ScriptName },
  { "ScriptFile", BUILTIN_FUNC_PREFIX, "", ScriptFile },
  { "ScriptDir",  BUILTIN_FUNC_PREFIX, "", ScriptDir  },
  { "ScriptNameUtf8", BUILTIN_FUNC_PREFIX, "", ScriptNameUtf8 },
  { "ScriptFileUtf8", BUILTIN_FUNC_PREFIX, "", ScriptFileUtf8 },
  { "ScriptDirUtf8",  BUILTIN_FUNC_PREFIX, "", ScriptDirUtf8 },

  { "PixelType",  BUILTIN_FUNC_PREFIX, "c", PixelType  },

  { "AddAutoloadDir",     BUILTIN_FUNC_PREFIX, "s[toFront]b", AddAutoloadDir  },
  { "ClearAutoloadDirs",  BUILTIN_FUNC_PREFIX, "", ClearAutoloadDirs  },
  { "AutoloadPlugins",    BUILTIN_FUNC_PREFIX, "", AutoloadPlugins  },
  { "FunctionExists",     BUILTIN_FUNC_PREFIX, "s", FunctionExists  },
  { "InternalFunctionExists", BUILTIN_FUNC_PREFIX, "s", InternalFunctionExists  },

  { "SetFilterMTMode",  BUILTIN_FUNC_PREFIX, "si[force]b", SetFilterMTMode  },
  { "Prefetch",         BUILTIN_FUNC_PREFIX, "c[threads]i[frames]i", Prefetcher::Create },
  { "SetLogParams",     BUILTIN_FUNC_PREFIX, "[target]s[level]i", SetLogParams },
  { "LogMsg",           BUILTIN_FUNC_PREFIX, "si", LogMsg },
  { "SetCacheMode",     BUILTIN_FUNC_PREFIX, "[mode]i", SetCacheMode }, // Neo
  { "SetDeviceOpt",     BUILTIN_FUNC_PREFIX, "[opt]i[val]i", SetDeviceOpt }, // Neo
  { "SetMaxCPU",        BUILTIN_FUNC_PREFIX, "s", SetMaxCPU }, // 20200331

  { "IsY",       BUILTIN_FUNC_PREFIX, "c", IsY },
  { "Is420",     BUILTIN_FUNC_PREFIX, "c", Is420 },
  { "Is422",     BUILTIN_FUNC_PREFIX, "c", Is422 },
  { "Is444",     BUILTIN_FUNC_PREFIX, "c", Is444 },
  { "IsRGB48",       BUILTIN_FUNC_PREFIX, "c", IsRGB48 },
  { "IsRGB64",       BUILTIN_FUNC_PREFIX, "c", IsRGB64 },
  { "ComponentSize", BUILTIN_FUNC_PREFIX, "c", ComponentSize },
  { "BitsPerComponent", BUILTIN_FUNC_PREFIX, "c", BitsPerComponent },
  { "IsYUVA",       BUILTIN_FUNC_PREFIX, "c", IsYUVA },
  { "IsPlanarRGB",  BUILTIN_FUNC_PREFIX, "c", IsPlanarRGB },
  { "IsPlanarRGBA", BUILTIN_FUNC_PREFIX, "c", IsPlanarRGBA },
  { "ColorSpaceNameToPixelType",  BUILTIN_FUNC_PREFIX, "s", ColorSpaceNameToPixelType },
  { "NumComponents", BUILTIN_FUNC_PREFIX, "c", NumComponents }, // r2348+
  { "HasAlpha", BUILTIN_FUNC_PREFIX, "c", HasAlpha }, // r2348+
  { "IsPackedRGB", BUILTIN_FUNC_PREFIX, "c", IsPackedRGB }, // r2348+
  { "IsVideoFloat", BUILTIN_FUNC_PREFIX, "c", IsVideoFloat }, // r2435+

  { "GetProcessInfo", BUILTIN_FUNC_PREFIX, "[type]i", GetProcessInfo }, // 170526-
#ifdef AVS_WINDOWS
  { "StrToUtf8", BUILTIN_FUNC_PREFIX, "s", StrToUtf8 }, // 170601-
  { "StrFromUtf8", BUILTIN_FUNC_PREFIX, "s", StrFromUtf8 }, // 170601-
#endif

  { "IsFloatUvZeroBased", BUILTIN_FUNC_PREFIX, "", IsFloatUvZeroBased }, // 180516-
  { "BuildPixelType", BUILTIN_FUNC_PREFIX, "[family]s[bits]i[chroma]i[compat]b[oldnames]b[sample_clip]c", BuildPixelType }, // 180517-
  { "VarExist", BUILTIN_FUNC_PREFIX, "s", VarExist }, // 180606-


    // Creates script array from zero or more anything.
    // Direct array constant syntax e.g. x = [arg1,arg2,...] is translated to x = Array(arg1,arg2,...)
  { "Array", BUILTIN_FUNC_PREFIX, ".*", ArrayCreate },
  { "IsArray",   BUILTIN_FUNC_PREFIX, ".", IsArray },
    // dictionary type array indexing
  { "ArrayGet",  BUILTIN_FUNC_PREFIX, ".s", ArrayGet },
    // classic array indexing background helper: e.g. a[3,4] -> ArrayGet(a, [2,3])
  { "ArrayGet",  BUILTIN_FUNC_PREFIX, ".i+", ArrayGet }, // .+i+ syntax is not possible.
    // length can be zero
  { "ArraySize", BUILTIN_FUNC_PREFIX, ".", ArraySize },
  { "ArrayIns",  BUILTIN_FUNC_PREFIX, "..i+", ArrayIns, (void*)0 },
  { "ArrayAdd",  BUILTIN_FUNC_PREFIX, "..i*", ArrayIns, (void*)1 },
  { "ArraySet",  BUILTIN_FUNC_PREFIX, "..i+", ArrayIns, (void*)2 },
  { "ArrayDel",  BUILTIN_FUNC_PREFIX, ".i+", ArrayIns, (void*)3 },
  /*
  { "IsArrayOf", BUILTIN_FUNC_PREFIX, ".s", IsArrayOf },
  */
  { 0 }
};


/**********************************
 *******   Script Function   ******
 *********************************/

ScriptFunction::ScriptFunction( const PExpression& _body, const bool* _param_floats,
                                const char** _param_names, int param_count )
  : body(_body)
{
  param_floats = new bool[param_count];
  memcpy(param_floats, _param_floats, param_count * sizeof(const bool));

  param_names = new const char* [param_count];
  memcpy(param_names, _param_names, param_count * sizeof(const char*));
}


AVSValue ScriptFunction::Execute(AVSValue args, void* user_data, IScriptEnvironment* env)
{
  ScriptFunction* self = (ScriptFunction*)user_data;
  env->PushContext();
  for (int i=0; i<args.ArraySize(); ++i)
    env->SetVar(self->param_names[i], // Force float args that are actually int to be float
      (self->param_floats[i] && args[i].IsInt()) ? float(args[i].AsInt()) : args[i]);

  AVSValue result;
  try {
    result = self->body->Evaluate(env);
  }
  catch (...) {
    env->PopContext();
    throw;
  }

  env->PopContext();
  return result;
}

void ScriptFunction::Delete(void* self, IScriptEnvironment*)
{
  delete (ScriptFunction*)self;
}

/***********************************
 *******   Helper Functions   ******
 **********************************/

#ifdef AVS_WINDOWS
void CWDChanger::Init(const wchar_t* new_cwd)
{
  // works in unicode internally
  uint32_t cwdLen = GetCurrentDirectoryW(0, NULL);
  old_working_directory = std::make_unique<wchar_t[]>(cwdLen); // instead of new wchar_t[cwdLen];
  uint32_t save_cwd_success = GetCurrentDirectoryW(cwdLen, old_working_directory.get());
  bool set_cwd_success = SetCurrentDirectoryW(new_cwd);
  restore = (save_cwd_success && set_cwd_success);
}

CWDChanger::CWDChanger(const wchar_t* new_cwd)
{
  Init(new_cwd);
}

CWDChanger::CWDChanger(const char* new_cwd)
{
  auto new_cwd_w = AnsiToWideChar(new_cwd);
  Init(new_cwd_w.get());
}

CWDChanger::~CWDChanger(void)
{
  if (restore)
    SetCurrentDirectoryW(old_working_directory.get());
}

DllDirChanger::DllDirChanger(const char* new_dir)
{
  uint32_t len = GetDllDirectory (0, NULL);
  old_directory = std::make_unique<char[]>(len + 1); // instead of new char[len+1]
  uint32_t save_success = GetDllDirectory (len, old_directory.get());
  bool set_success = SetDllDirectory(new_dir);
  restore = (save_success && set_success);
}

DllDirChanger::~DllDirChanger(void)
{
  if (restore)
    SetDllDirectory(old_directory.get());
}
#else // copied from AvxSynth
CWDChanger::CWDChanger(const char* new_cwd)
{

  char* path = getcwd(old_working_directory, FILENAME_MAX);
  bool save_cwd_success = (NULL != path);
  bool set_cwd_success = (0 == chdir(new_cwd));
  restore = (save_cwd_success && set_cwd_success);
}

CWDChanger::~CWDChanger(void)
{
  if (restore)
    chdir(old_working_directory);
}
#endif


AVSValue Assert(AVSValue args, void*, IScriptEnvironment* env)
{
  if (!args[0].AsBool())
    env->ThrowError("%s", args[1].Defined() ? args[1].AsString() : "Assert: assertion failed");
  return AVSValue();
}

AVSValue AssertEval(AVSValue args, void*, IScriptEnvironment* env)
{
  const char* pred = args[0].AsString();
  AVSValue eval_args[] = { args[0].AsString(), "asserted expression" };
  AVSValue val = env->Invoke("Eval", AVSValue(eval_args, 2));
  if (!val.IsBool())
    env->ThrowError("Assert: expression did not evaluate to true or false: \"%s\"", pred);
  if (!val.AsBool())
    env->ThrowError("Assert: assertion failed: \"%s\"", pred);
  return AVSValue();
}

AVSValue Eval(AVSValue args, void*, IScriptEnvironment* env)
{
  const char *filename = args[1].AsString(0);
  if (filename) filename = env->SaveString(filename);
  ScriptParser parser(env, args[0].AsString(), filename);
  PExpression exp = parser.Parse();
  return exp->Evaluate(env);
}

AVSValue Apply(AVSValue args, void*, IScriptEnvironment* env)
{
  return env->Invoke(args[0].AsString(), args[1]);
}

AVSValue EvalOop(AVSValue args, void*, IScriptEnvironment* env)
{
  AVSValue prev_last = env->GetVarDef("last");  // Store previous last
  env->SetVar("last", args[0]);              // Set implicit last

  AVSValue result;
  try {
    result = Eval(AVSValue(&args[1], 2), 0, env);
  }
  catch(...) {
    env->SetVar("last", prev_last);          // Restore implicit last
	throw;
  }
  env->SetVar("last", prev_last);            // Restore implicit last
  return result;
}

AVSValue Import(AVSValue args, void*, IScriptEnvironment* env)
{
  // called as s+ or s+[Utf8]b
  const bool bHasUTF8param = args.IsArray() && args.ArraySize() == 2 && args[1].IsBool();
  const bool bUtf8 = bHasUTF8param ? args[1].AsBool(false) : false;

  args = args[0];
  AVSValue result;

  InternalEnvironment *envi = static_cast<InternalEnvironment*>(env);
  const bool MainScript = (envi->IncrImportDepth() == 1);

  AVSValue lastScriptName = env->GetVarDef("$ScriptName$");
  AVSValue lastScriptFile = env->GetVarDef("$ScriptFile$");
  AVSValue lastScriptDir  = env->GetVarDef("$ScriptDir$");

  AVSValue lastScriptNameUtf8 = env->GetVarDef("$ScriptNameUtf8$");
  AVSValue lastScriptFileUtf8 = env->GetVarDef("$ScriptFileUtf8$");
  AVSValue lastScriptDirUtf8 = env->GetVarDef("$ScriptDirUtf8$");

  for (int i = 0; i < args.ArraySize(); ++i) {
    const char* script_name = args[i].AsString();

#ifdef AVS_WINDOWS
    /* Linux, macOS, pretty much every OS aside from Windows uses
       UTF-8 pervasively and by default, making all the Ansi<->Unicode
       stuff we have to specially handle on Windows (which uses UTF-16
       when it does 'Unicode', further complicating things if you don't
       force UTF-8) irrelevant. */

      // Handling utf8 and ansi, working in wchar_t internally
      // filename and path can be full unicode
      // unicode input can come from CAVIFileSynth

    std::unique_ptr<wchar_t[]> full_path_w;
    wchar_t *file_part_w;

    // make wchar_t full path strnig from either ansi or utf8
    auto script_name_w = !bUtf8 ? AnsiToWideChar(script_name) : Utf8ToWideChar(script_name);

    // Long (>MAX_PATH) path support starting in Windows 10, version 1607.
    if (wcschr(script_name_w.get(), '\\') || wcschr(script_name_w.get(), '/')) {
      DWORD len = GetFullPathNameW(script_name_w.get(), 0, NULL, NULL); // buffer size for path + terminating zero
      full_path_w = std::make_unique<wchar_t[]>(len);
      len = GetFullPathNameW(script_name_w.get(), len, full_path_w.get(), &file_part_w);
      if (len == 0)
        env->ThrowError("Import: unable to open \"%s\" (path invalid?), error=0x%x", script_name, GetLastError());
    }
    else {
      DWORD len = SearchPathW(NULL, script_name_w.get(), NULL, 0, NULL, NULL); // buffer size for path + terminating zero
      full_path_w = std::make_unique<wchar_t[]>(len);
      len = SearchPathW(NULL, script_name_w.get(), NULL, len, full_path_w.get(), &file_part_w);
      if (len == 0)
        env->ThrowError("Import: unable to locate \"%s\" (try specifying a path), error=0x%x", script_name, GetLastError());
    }

    // back to 8 bit Ansi and Utf8
    auto full_path = WideCharToAnsi(full_path_w.get());
    auto full_path_utf8 = WideCharToUtf8(full_path_w.get());
    auto file_part = WideCharToAnsi(file_part_w);
    auto file_part_utf8 = WideCharToUtf8(file_part_w);
    size_t dir_part_len = wcslen(full_path_w.get()) - wcslen(file_part_w);
    auto dir_part = WideCharToAnsi_maxn(full_path_w.get(), dir_part_len);
    auto dir_part_utf8 = WideCharToUtf8_maxn(full_path_w.get(), dir_part_len);

    // supply L"\\\\?\\" if necessary for long file path support
    std::wstring full_path_ex = std::wstring(full_path_w.get());
    if (full_path_ex.length() > FILENAME_MAX && full_path_ex.substr(0, 4) != L"\\\\?\\")
      full_path_ex = L"\\\\?\\" + full_path_ex;

    HANDLE h = ::CreateFileW(full_path_ex.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, 0, NULL);
    if (h == INVALID_HANDLE_VALUE)
      env->ThrowError("Import: couldn't open \"%s\"", full_path.get());

    env->SetGlobalVar("$ScriptName$", env->SaveString(full_path.get()));
    env->SetGlobalVar("$ScriptFile$", env->SaveString(file_part.get()));
    env->SetGlobalVar("$ScriptDir$", env->SaveString(dir_part.get()));
    env->SetGlobalVar("$ScriptNameUtf8$", env->SaveString(full_path_utf8.get()));
    env->SetGlobalVar("$ScriptFileUtf8$", env->SaveString(file_part_utf8.get()));
    env->SetGlobalVar("$ScriptDirUtf8$", env->SaveString(dir_part_utf8.get()));
    if (MainScript)
    {
      env->SetGlobalVar("$MainScriptName$", env->SaveString(full_path.get()));
      env->SetGlobalVar("$MainScriptFile$", env->SaveString(file_part.get()));
      env->SetGlobalVar("$MainScriptDir$", env->SaveString(dir_part.get()));
      env->SetGlobalVar("$MainScriptNameUtf8$", env->SaveString(full_path_utf8.get()));
      env->SetGlobalVar("$MainScriptFileUtf8$", env->SaveString(file_part_utf8.get()));
      env->SetGlobalVar("$MainScriptDirUtf8$", env->SaveString(dir_part_utf8.get()));
    }

    *file_part_w = 0; // trunc full_path_w to dir-only
    CWDChanger change_cwd(full_path_w.get());
    // end of filename parsing / file open things

    DWORD size = GetFileSize(h, NULL);
    std::vector<char> buf(size + 1, 0);
    bool status = ReadFile(h, buf.data(), size, &size, NULL);
    CloseHandle(h);
    if (!status)
      env->ThrowError("Import: unable to read \"%s\"", script_name);

    // Give poor Unicode users a hint they need to use ANSI encoding import"
    if (size >= 2) {
      unsigned char* q = reinterpret_cast<unsigned char*>(buf.data());

      if ((q[0] == 0xFF && q[1] == 0xFE) || (q[0] == 0xFE && q[1] == 0xFF))
        env->ThrowError("Import: Unicode source files are not supported, "
          "re-save script with ANSI or UTF8 w/o BOM encoding! : \"%s\"", script_name);

      if (q[0] == 0xEF && q[1] == 0xBB && q[2] == 0xBF)
        env->ThrowError("Import: UTF-8 source files with BOM are not supported, "
          "re-save script with ANSI or UTF8 w/o BOM encoding! : \"%s\"", script_name);
    }

#else // adapted from AvxSynth
    std::string file_part = fs::path(script_name).filename().string();
    std::string full_path = fs::path(script_name).remove_filename();
    std::string dir_part = fs::path(script_name).parent_path();

    FILE* h = fopen(script_name, "r");
    if(NULL == h)
      env->ThrowError("Import: couldn't open \"%s\"", script_name );

    env->SetGlobalVar("$ScriptName$", env->SaveString(script_name));
    env->SetGlobalVar("$ScriptFile$", env->SaveString(file_part.c_str()));
    env->SetGlobalVar("$ScriptDir$", env->SaveString(full_path.c_str()));
    env->SetGlobalVar("$ScriptNameUtf8$", env->SaveString(script_name));
    env->SetGlobalVar("$ScriptFileUtf8$", env->SaveString(file_part.c_str()));
    env->SetGlobalVar("$ScriptDirUtf8$", env->SaveString(full_path.c_str()));
    if (MainScript)
    {
      env->SetGlobalVar("$MainScriptName$", env->SaveString(script_name));
      env->SetGlobalVar("$MainScriptFile$", env->SaveString(file_part.c_str()));
      env->SetGlobalVar("$MainScriptDir$", env->SaveString(full_path.c_str()));
      env->SetGlobalVar("$MainScriptNameUtf8$", env->SaveString(script_name));
      env->SetGlobalVar("$MainScriptFileUtf8$", env->SaveString(file_part.c_str()));
      env->SetGlobalVar("$MainScriptDirUtf8$", env->SaveString(full_path.c_str()));
    }

    //*file_part = 0; // trunc full_path to dir-only
    CWDChanger change_cwd(full_path.c_str());
    // end of filename parsing / file open things

    fseek(h, 0, SEEK_END);
    size_t size = ftell(h);
    fseek(h, 0, SEEK_SET);

    std::vector<char> buf(size + 1, 0);
    if(size != fread(buf.data(), 1, size, h))
      env->ThrowError("Import: unable to read \"%s\"", script_name);

    fclose(h);
#endif

    buf[size] = 0;
    AVSValue eval_args[] = { buf.data(), script_name };
    result = env->Invoke("Eval", AVSValue(eval_args, 2));
    //env->ThrowError("Import: test %s size %d\n", buf.data(), (int)size);
  }

  env->SetGlobalVar("$ScriptName$", lastScriptName);
  env->SetGlobalVar("$ScriptFile$", lastScriptFile);
  env->SetGlobalVar("$ScriptDir$",  lastScriptDir);
  env->SetGlobalVar("$ScriptNameUtf8$", lastScriptNameUtf8);
  env->SetGlobalVar("$ScriptFileUtf8$", lastScriptFileUtf8);
  env->SetGlobalVar("$ScriptDirUtf8$", lastScriptDirUtf8);
  envi->DecrImportDepth();

  return result;
}


AVSValue ScriptName(AVSValue args, void*, IScriptEnvironment* env) { return env->GetVarDef("$ScriptName$"); }
AVSValue ScriptFile(AVSValue args, void*, IScriptEnvironment* env) { return env->GetVarDef("$ScriptFile$"); }
AVSValue ScriptDir (AVSValue args, void*, IScriptEnvironment* env) { return env->GetVarDef("$ScriptDir$" ); }
AVSValue ScriptNameUtf8(AVSValue args, void*, IScriptEnvironment* env) { return env->GetVarDef("$ScriptNameUtf8$"); }
AVSValue ScriptFileUtf8(AVSValue args, void*, IScriptEnvironment* env) { return env->GetVarDef("$ScriptFileUtf8$"); }
AVSValue ScriptDirUtf8(AVSValue args, void*, IScriptEnvironment* env) { return env->GetVarDef("$ScriptDirUtf8$"); }
AVSValue SetWorkingDir(AVSValue args, void*, IScriptEnvironment* env) { return env->SetWorkingDir(args[0].AsString()); }

AVSValue Muldiv(AVSValue args, void*, IScriptEnvironment* ) { return int(MulDiv(args[0].AsInt(), args[1].AsInt(), args[2].AsInt())); }

AVSValue Floor(AVSValue args, void*, IScriptEnvironment* ) { return int(floor(args[0].AsFloat())); }
AVSValue Ceil(AVSValue args, void*, IScriptEnvironment* ) { return int(ceil(args[0].AsFloat())); }
AVSValue Round(AVSValue args, void*, IScriptEnvironment* ) { return args[0].AsFloat()<0 ? -int(-args[0].AsFloat()+.5) : int(args[0].AsFloat()+.5); }

AVSValue Acos(AVSValue args, void* , IScriptEnvironment* ) { return acos(args[0].AsFloat()); }
AVSValue Asin(AVSValue args, void* , IScriptEnvironment* ) { return asin(args[0].AsFloat()); }
AVSValue Atan(AVSValue args, void* , IScriptEnvironment* ) { return atan(args[0].AsFloat()); }
AVSValue Atan2(AVSValue args, void* , IScriptEnvironment* ) { return atan2(args[0].AsFloat(), args[1].AsFloat()); }
AVSValue Cos(AVSValue args, void* , IScriptEnvironment* ) { return cos(args[0].AsFloat()); }
AVSValue Cosh(AVSValue args, void* , IScriptEnvironment* ) { return cosh(args[0].AsFloat()); }
AVSValue Exp(AVSValue args, void* , IScriptEnvironment* ) { return exp(args[0].AsFloat()); }
AVSValue Fmod(AVSValue args, void* , IScriptEnvironment* ) { return fmod(args[0].AsFloat(), args[1].AsFloat()); }
AVSValue Log(AVSValue args, void* , IScriptEnvironment* ) { return log(args[0].AsFloat()); }
AVSValue Log10(AVSValue args, void* , IScriptEnvironment* ) { return log10(args[0].AsFloat()); }
AVSValue Pow(AVSValue args, void* , IScriptEnvironment* ) { return pow(args[0].AsFloat(),args[1].AsFloat()); }
AVSValue Sin(AVSValue args, void* , IScriptEnvironment* ) { return sin(args[0].AsFloat()); }
AVSValue Sinh(AVSValue args, void* , IScriptEnvironment* ) { return sinh(args[0].AsFloat()); }
AVSValue Tan(AVSValue args, void* , IScriptEnvironment* ) { return tan(args[0].AsFloat()); }
AVSValue Tanh(AVSValue args, void* , IScriptEnvironment* ) { return tanh(args[0].AsFloat()); }
AVSValue Sqrt(AVSValue args, void* , IScriptEnvironment* ) { return sqrt(args[0].AsFloat()); }

AVSValue Abs(AVSValue args, void* , IScriptEnvironment* ) { return abs(args[0].AsInt()); }
AVSValue FAbs(AVSValue args, void* , IScriptEnvironment* ) { return fabs(args[0].AsFloat()); }
AVSValue Pi(AVSValue args, void* , IScriptEnvironment* )  { return 3.14159265358979324; }
#ifdef OPT_ScriptFunctionTau
AVSValue Tau(AVSValue args, void* , IScriptEnvironment* ) { return 6.28318530717958648; }
#endif
AVSValue Sign(AVSValue args, void*, IScriptEnvironment* ) { return args[0].AsFloat()==0 ? 0 : args[0].AsFloat() > 0 ? 1 : -1; }

AVSValue BitAnd(AVSValue args, void*, IScriptEnvironment* ) { return args[0].AsInt() & args[1].AsInt(); }
AVSValue BitNot(AVSValue args, void*, IScriptEnvironment* ) { return ~args[0].AsInt(); }
AVSValue BitOr(AVSValue args, void*, IScriptEnvironment* )  { return args[0].AsInt() | args[1].AsInt(); }
AVSValue BitXor(AVSValue args, void*, IScriptEnvironment* ) { return args[0].AsInt() ^ args[1].AsInt(); }

AVSValue BitLShift(AVSValue args, void*, IScriptEnvironment* ) { return args[0].AsInt() << args[1].AsInt(); }
AVSValue BitRShiftL(AVSValue args, void*, IScriptEnvironment* ) { return int(unsigned(args[0].AsInt()) >> unsigned(args[1].AsInt())); }
AVSValue BitRShiftA(AVSValue args, void*, IScriptEnvironment* ) { return args[0].AsInt() >> args[1].AsInt(); }

static int a_rol(int value, int shift) {
  if ((shift &= sizeof(value)*8 - 1) == 0)
      return value;
  return (value << shift) | (value >> (sizeof(value)*8 - shift));
}

static int a_ror(int value, int shift) {
  if ((shift &= sizeof(value)*8 - 1) == 0)
      return value;
  return (value >> shift) | (value << (sizeof(value)*8 - shift));
}

static int a_btc(int value, int bit) {
  value ^= 1 << bit;
  return value;
}

static int a_btr(int value, int bit) {
  value &= ~(1 << bit);
  return value;
}

static int a_bts(int value, int bit) {
  value |= (1 << bit);
  return value;
}

static bool a_bt (int value, int bit) {
  return (value & (1 << bit)) ? true : false;
}

AVSValue BitRotateL(AVSValue args, void*, IScriptEnvironment* ) { return a_rol(args[0].AsInt(), args[1].AsInt()); }
AVSValue BitRotateR(AVSValue args, void*, IScriptEnvironment* ) { return a_ror(args[0].AsInt(), args[1].AsInt()); }

AVSValue BitChg(AVSValue args, void*, IScriptEnvironment* ) { return a_btc(args[0].AsInt(), args[1].AsInt()); }
AVSValue BitClr(AVSValue args, void*, IScriptEnvironment* ) { return a_btr(args[0].AsInt(), args[1].AsInt()); }
AVSValue BitSet(AVSValue args, void*, IScriptEnvironment* ) { return a_bts(args[0].AsInt(), args[1].AsInt()); }
AVSValue BitTst(AVSValue args, void*, IScriptEnvironment* ) { return a_bt (args[0].AsInt(), args[1].AsInt()); }

static int numberOfSetBits(uint32_t i)
{
  i = i - ((i >> 1) & 0x55555555);
  i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
  return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}

AVSValue BitSetCount(AVSValue args, void*, IScriptEnvironment*) {
  if (args[0].IsInt())
    return numberOfSetBits(args[0].AsInt());

  int count = 0;
  for (int i = 0; i < args[0].ArraySize(); i++)
    count += numberOfSetBits(args[0][i].AsInt());
  return count;
}

AVSValue UCase(AVSValue args, void*, IScriptEnvironment* env) { return _strupr(env->SaveString(args[0].AsString())); }
AVSValue LCase(AVSValue args, void*, IScriptEnvironment* env) { return _strlwr(env->SaveString(args[0].AsString())); }

AVSValue StrLen(AVSValue args, void*, IScriptEnvironment* ) { return int(strlen(args[0].AsString())); }
AVSValue RevStr(AVSValue args, void*, IScriptEnvironment* env) { return _strrev(env->SaveString(args[0].AsString())); }

AVSValue LeftStr(AVSValue args, void*, IScriptEnvironment* env)
 {
   const int count = args[1].AsInt();
   if (count < 0)
      env->ThrowError("LeftStr: Negative character count not allowed");
   char *result = new(std::nothrow) char[count+1];
   if (!result) env->ThrowError("LeftStr: malloc failure!");
   *result = 0;
   strncat(result, args[0].AsString(), count);
   AVSValue ret = env->SaveString(result);
   delete[] result;
   return ret;
 }

AVSValue MidStr(AVSValue args, void*, IScriptEnvironment* env)
{
  const int maxlen = (int)strlen(args[0].AsString());
  if (args[1].AsInt() < 1)
      env->ThrowError("MidStr: Illegal character location");
  int len = args[2].AsInt(maxlen);
  if (len < 0)
      env->ThrowError("MidStr: Illegal character count");
  int offset = args[1].AsInt() - 1;
  if (maxlen <= offset) { offset = 0; len = 0;}
  char *result = new(std::nothrow) char[len+1];
  if (!result) env->ThrowError("MidStr: malloc failure!");
  *result = 0;
  strncat(result, args[0].AsString()+offset, len);
  AVSValue ret = env->SaveString(result);
  delete[] result;
  return ret;
}

AVSValue RightStr(AVSValue args, void*, IScriptEnvironment* env)
 {
   if (args[1].AsInt() < 0)
      env->ThrowError("RightStr: Negative character count not allowed");

   int offset = (int)strlen(args[0].AsString()) - args[1].AsInt();
   if (offset < 0) offset = 0;
   char *result = new(std::nothrow) char[args[1].AsInt()+1];
   if (!result) env->ThrowError("RightStr: malloc failure!");
   *result = 0;
   strncat(result, args[0].AsString()+offset, args[1].AsInt());
   AVSValue ret = env->SaveString(result);
   delete[] result;
   return ret;
 }

AVSValue ReplaceStr(AVSValue args, void*, IScriptEnvironment* env) {
  char const * const original = args[0].AsString();
  char const * const pattern = args[1].AsString();
  char const * const replacement = args[2].AsString();
  const bool case_insensitive = args[3].AsBool(false);

  const size_t replace_len = strlen(replacement);
  const size_t pattern_len = strlen(pattern);
  const size_t orig_len = strlen(original);

  size_t pattern_count = 0;
  const char * orig_ptr;
  const char * pattern_location;

  if (0 == pattern_len)
    return original;

  if (case_insensitive) {
    char *original_lower = new(std::nothrow) char[sizeof(char) * (orig_len + 1)];
    if (!original_lower) env->ThrowError("ReplaceStr: malloc failure!");
    char *pattern_lower = new(std::nothrow) char[sizeof(char) * (pattern_len + 1)];
    if (!pattern_lower) env->ThrowError("ReplaceStr: malloc failure!");

    // make them lowercase for comparison
    strcpy(original_lower, original);
    strcpy(pattern_lower, pattern);
#ifdef MSVC
    // works fine also for accented ANSI characters
    _locale_t locale = _create_locale(LC_ALL, ".ACP"); // Sets the locale to the ANSI code page obtained from the operating system.
    _strlwr_l(original_lower, locale);
    _strlwr_l(pattern_lower, locale);
    _free_locale(locale);
#else
    _strlwr(original_lower);
    _strlwr(pattern_lower);
#endif

    // find how many times the _lowercased_ pattern occurs in the _lowercased_ original string
    for (orig_ptr = original_lower; (pattern_location = strstr(orig_ptr, pattern_lower)); orig_ptr = pattern_location + pattern_len)
    {
      pattern_count++;
    }

    // allocate memory for the new string
    size_t const retlen = orig_len + pattern_count * (replace_len - pattern_len);
    char *result = new(std::nothrow) char[sizeof(char) * (retlen + 1)];
    if (!result) env->ThrowError("ReplaceStr: malloc failure!");
    *result = 0;

    // copy the original string,
    // replacing all the instances of the pattern
    const char * orig_upper_ptr;
    char * result_ptr = result;
    // handling dual pointer set: orig, uppercase
    for (orig_ptr = original, orig_upper_ptr = original_lower;
      (pattern_location = strstr(orig_upper_ptr, pattern_lower));
      orig_upper_ptr = pattern_location + pattern_len, orig_ptr = original + (orig_upper_ptr - original_lower))
    {
      const size_t skiplen = pattern_location - orig_upper_ptr;
      // copy the section until the occurence of the pattern
      strncpy(result_ptr, orig_ptr, skiplen);
      result_ptr += skiplen;
      // copy the replacement
      strncpy(result_ptr, replacement, replace_len);
      result_ptr += replace_len;
    }
    // copy rest
    strcpy(result_ptr, orig_ptr);
    AVSValue ret = env->SaveString(result);
    delete[] result;
    delete[] original_lower;
    delete[] pattern_lower;
    return ret;
  }

  // old case sensitive version

    // find how many times the pattern occurs in the original string
  for (orig_ptr = original; (pattern_location = strstr(orig_ptr, pattern)); orig_ptr = pattern_location + pattern_len)
  {
    pattern_count++;
  }

  // allocate memory for the new string
  size_t const retlen = orig_len + pattern_count * (replace_len - pattern_len);
  char *result = new(std::nothrow) char[sizeof(char) * (retlen + 1)];
  if (!result) env->ThrowError("ReplaceStr: malloc failure!");
  *result = 0;

  // copy the original string,
  // replacing all the instances of the pattern
  char * result_ptr = result;
  for (orig_ptr = original; (pattern_location = strstr(orig_ptr, pattern)); orig_ptr = pattern_location + pattern_len)
  {
    const size_t skiplen = pattern_location - orig_ptr;
    // copy the section until the occurence of the pattern
    strncpy(result_ptr, orig_ptr, skiplen);
    result_ptr += skiplen;
    // copy the replacement
    strncpy(result_ptr, replacement, replace_len);
    result_ptr += replace_len;
  }
  // copy rest
  strcpy(result_ptr, orig_ptr);
  AVSValue ret = env->SaveString(result);
  delete[] result;
  return ret;
}

AVSValue TrimLeft(AVSValue args, void*, IScriptEnvironment* env)
{
  char const *original = args[0].AsString();
  char const *s = original;
  char ch;
  // space, npsp, tab
  while ((ch = *s) == (char)32 || ch == (char)160 || ch == (char)9)
    s++;

  if (original == s)
    return args[0]; // avoid SaveString if no change

  return env->SaveString(s);
}

AVSValue TrimRight(AVSValue args, void*, IScriptEnvironment* env)
{
  char const *original = args[0].AsString();
  size_t len = strlen(original);
  if (len == 0)
    return args[0]; // avoid SaveString if no change

  size_t orig_len = len;
  char const *s = original + len;

  char ch;
  // space, npsp, tab
  while ((len > 0) && ((ch = *--s) == (char)32 || ch == (char)160 || ch == (char)9)) {
    len--;
  }

  if(orig_len == len)
    return args[0]; // avoid SaveString if no change

  if (len == 0)
    return env->SaveString("");

  size_t retlen = s - original + 1;

  char *result = new(std::nothrow) char[sizeof(char) * (retlen + 1)];
  if (!result) env->ThrowError("TrimRight: malloc failure!");
  strncpy(result, original, retlen);
  result[retlen] = 0;

  AVSValue ret = env->SaveString(result);
  delete[] result;
  return ret;
}

AVSValue TrimAll(AVSValue args, void*, IScriptEnvironment* env)
{
  // not simplify with calling Left/Right, avoid double SaveStrings

  // like TrimLeft
  char const *original = args[0].AsString();
  if (!*original)
    return args[0]; // avoid SaveString if no change

  char ch;
  // space, npsp, tab
  while ((ch = *original) == (char)32 || ch == (char)160 || ch == (char)9)
    original++;

  // almost like TrimRight
  size_t len = strlen(original);
  if (len == 0)
    return env->SaveString("");

  size_t orig_len = len;
  char const *s = original + len;

  // space, npsp, tab
  while ((len > 0) && ((ch = *--s) == (char)32 || ch == (char)160 || ch == (char)9))
    len--;

  if (orig_len == len)
    return env->SaveString(original); // nothing to cut from right

  if (len == 0)
    return env->SaveString(""); // full cut

  size_t retlen = s - original + 1;

  char *result = new(std::nothrow) char[sizeof(char) * (retlen + 1)];
  if (!result) env->ThrowError("TrimAll: malloc failure!");
  strncpy(result, original, retlen);
  result[retlen] = 0;

  AVSValue ret = env->SaveString(result);
  delete[] result;
  return ret;
}


AVSValue StrCmp(AVSValue args, void*, IScriptEnvironment*)
{
  return lstrcmp( args[0].AsString(), args[1].AsString() );
}

AVSValue StrCmpi(AVSValue args, void*, IScriptEnvironment*)
{
  return lstrcmpi( args[0].AsString(), args[1].AsString() );
}

AVSValue FindStr(AVSValue args, void*, IScriptEnvironment*)
{
  const char *pdest = strstr( args[0].AsString(),args[1].AsString() );
  int result = (int)(pdest - args[0].AsString() + 1);
  if (pdest == NULL) result = 0;
  return result;
}

AVSValue Rand(AVSValue args, void*, IScriptEnvironment*)
 { int limit = args[0].AsInt(RAND_MAX);
   bool scale_mode = args[1].AsBool((abs(limit) > RAND_MAX));

   if (args[2].AsBool(false)) srand( (unsigned) time(NULL) ); //seed

   if (scale_mode) {
      double f = 1.0 / (RAND_MAX + 1.0);
      return int(f * rand() * limit);
   }
   else { //modulus mode
      int s = (limit < 0 ? -1 : 1);
      if (limit==0) return 0;
       else return s * rand() % limit;
   }
 }

AVSValue Select(AVSValue args, void*, IScriptEnvironment* env)
{ int i = args[0].AsInt();
  if ((args[1].ArraySize() <= i) || (i < 0))
    env->ThrowError("Select: Index value out of range");
  return args[1][i];
}

AVSValue NOP(AVSValue args, void*, IScriptEnvironment*) { return 0;}

AVSValue Undefined(AVSValue args, void*, IScriptEnvironment*) { return AVSValue();}

AVSValue Exist(AVSValue args, void*, IScriptEnvironment*nv) {
  const char *filename = args[0].AsString();
#ifdef AVS_POSIX
  constexpr bool utf8default = true;
#else
  constexpr bool utf8default = false;
#endif
  const bool utf8 = args[1].AsBool(utf8default);

  if (strchr(filename, '*') || strchr(filename, '?')) // wildcard
    return false;

#ifdef AVS_WINDOWS
  if (utf8) {
    // fixme/enhance me: check win codepage 65001 UTF8 and do like posix native utf8
    // (remark applies to all utf8 in avs+)
    auto wsource = Utf8ToWideChar(filename);
    std::wstring filename_w = wsource.get();
    return fs::exists(filename_w);
  }
#endif
  return fs::exists(filename);
}


//WE ->

// Spline functions to generate and evaluate a natural bicubic spline
void spline(float x[], float y[], int n, float y2[])
{
	int i,k;
	float p, qn, sig, un, *u;

	u = new float[n];

	y2[1]=u[1]=0.0f;

	for (i=2; i<=n-1; i++) {
		sig = (x[i] - x[i-1])/(x[i+1] - x[i-1]);
		p = sig * y2[i-1] + 2.0f;
		y2[i] = (sig - 1.0f) / p;
		u[i] = (y[i+1] - y[i])/(x[i+1] - x[i]) - (y[i] - y[i-1])/(x[i] - x[i-1]);
		u[i] = (6.0f*u[i]/(x[i+1] - x[i-1]) - sig*u[i-1])/p;
	}
	qn=un=0.0f;
	y2[n]=(un - qn*u[n-1])/(qn * y2[n-1] + 1.0f);
	for (k=n-1; k>=1; k--) {
		y2[k] = y2[k] * y2[k+1] + u[k];
	}

	delete[] u;
}

int splint(float xa[], float ya[], float y2a[], int n, float x, float &y, bool cubic)
{
	int klo, khi, k;
	float h,b,a;

	klo=1;
	khi=n;
	while (khi-klo > 1) {
		k=(khi + klo) >> 1;
		if (xa[k] > x ) khi = k;
		else klo = k;
	}
	h = xa[khi] - xa[klo];
	if (h==0.0f) {
		y=0.0f;
		return -1;	// all x's have to be different
	}
	a = (xa[khi] - x)/h;
	b = (x - xa[klo])/h;

	if (cubic) {
		y = a * ya[klo] + b*ya[khi] + ((a*a*a - a)*y2a[klo] + (b*b*b - b)*y2a[khi]) * (h*h) / 6.0f;
	} else {
		y = a * ya[klo] + b*ya[khi];
	}
	return 0;
}

// the script functions
AVSValue AVSChr(AVSValue args, void*, IScriptEnvironment* env )
{
    char s[2];

	s[0]=(char)(args[0].AsInt());
	s[1]=0;
    return env->SaveString(s);
}

AVSValue AVSOrd(AVSValue args, void*, IScriptEnvironment*)
{
    return (int)args[0].AsString()[0] & 0xFF;
}

AVSValue FillStr(AVSValue args, void*, IScriptEnvironment* env )
{
    const int count = args[0].AsInt();
    if (count <= 0)
      env->ThrowError("FillStr: Repeat count must greater than zero!");

    const char *str = args[1].AsString(" ");
    const int len = lstrlen(str);
    const int total = count * len;

    char *buff = new(std::nothrow) char[total];
    if (!buff)
      env->ThrowError("FillStr: malloc failure!");

    for (int i=0; i<total; i+=len)
      memcpy(buff+i, str, len);

    AVSValue ret = env->SaveString(buff, total);
    delete[] buff;
    return ret;
}

AVSValue AVSTime(AVSValue args, void*, IScriptEnvironment* env )
{
	time_t lt_t;
	struct tm * lt;
	time(&lt_t);
	lt = localtime (&lt_t);
    char s[1024];
    strftime(s,1024,args[0].AsString(""),lt);
    s[1023] = 0;
    return env->SaveString(s);
}

AVSValue Spline(AVSValue args, void*, IScriptEnvironment* env )
{
	int n;
	float x,y;
	int i;
	bool cubic;

	AVSValue coordinates;

	x = args[0].AsFloatf(0);
	coordinates = args[1];
	cubic = args[2].AsBool(true);

	n = coordinates.ArraySize() ;

	if (n<4 || n&1) env->ThrowError("To few arguments for Spline");

	n=n/2;

  float *buf = new float[(n+1)*3];
  float *xa  = &(buf[(n+1) * 0]);
  float *ya  = &(buf[(n+1) * 1]);
  float *y2a = &(buf[(n+1) * 2]);

	for (i=1; i<=n; i++) {
		xa[i] = coordinates[(i-1)*2+0].AsFloatf(0);
		ya[i] = coordinates[(i-1)*2+1].AsFloatf(0);
	}

	for (i=1; i<n; i++) {
		if (xa[i] >= xa[i+1]) env->ThrowError("Spline: all x values have to be different and in ascending order!");
	}

	spline(xa, ya, n, y2a);
	splint(xa, ya, y2a, n, x, y, cubic);

  delete[] buf;

	return y;
}

// WE <-

static inline const VideoInfo& VI(const AVSValue& arg) { return arg.AsClip()->GetVideoInfo(); }

static const std::map<int, std::string> pixel_format_table =
{ // names for lookup by pixel_type or name
  {VideoInfo::CS_BGR24, "RGB24"},
  {VideoInfo::CS_BGR32, "RGB32"},
  {VideoInfo::CS_YUY2 , "YUY2"},
  {VideoInfo::CS_YV24 , "YV24"},
  {VideoInfo::CS_YV16 , "YV16"},
  {VideoInfo::CS_YV12 , "YV12"},
  {VideoInfo::CS_I420 , "YV12"},
  {VideoInfo::CS_YUV9 , "YUV9"},
  {VideoInfo::CS_YV411, "YV411"},
  {VideoInfo::CS_Y8   , "Y8"},

  {VideoInfo::CS_YUV420P10, "YUV420P10"},
  {VideoInfo::CS_YUV422P10, "YUV422P10"},
  {VideoInfo::CS_YUV444P10, "YUV444P10"},
  {VideoInfo::CS_Y10      , "Y10"},
  {VideoInfo::CS_YUV420P12, "YUV420P12"},
  {VideoInfo::CS_YUV422P12, "YUV422P12"},
  {VideoInfo::CS_YUV444P12, "YUV444P12"},
  {VideoInfo::CS_Y12      , "Y12"},
  {VideoInfo::CS_YUV420P14, "YUV420P14"},
  {VideoInfo::CS_YUV422P14, "YUV422P14"},
  {VideoInfo::CS_YUV444P14, "YUV444P14"},
  {VideoInfo::CS_Y14      , "Y14"},
  {VideoInfo::CS_YUV420P16, "YUV420P16"},
  {VideoInfo::CS_YUV422P16, "YUV422P16"},
  {VideoInfo::CS_YUV444P16, "YUV444P16"},
  {VideoInfo::CS_Y16      , "Y16"},
  {VideoInfo::CS_YUV420PS , "YUV420PS"},
  {VideoInfo::CS_YUV422PS , "YUV422PS"},
  {VideoInfo::CS_YUV444PS , "YUV444PS"},
  {VideoInfo::CS_Y32      , "Y32"},

  {VideoInfo::CS_BGR48    , "RGB48"},
  {VideoInfo::CS_BGR64    , "RGB64"},

  {VideoInfo::CS_RGBP     , "RGBP"},
  {VideoInfo::CS_RGBP10   , "RGBP10"},
  {VideoInfo::CS_RGBP12   , "RGBP12"},
  {VideoInfo::CS_RGBP14   , "RGBP14"},
  {VideoInfo::CS_RGBP16   , "RGBP16"},
  {VideoInfo::CS_RGBPS    , "RGBPS"},

  {VideoInfo::CS_YUVA420, "YUVA420"},
  {VideoInfo::CS_YUVA422, "YUVA422"},
  {VideoInfo::CS_YUVA444, "YUVA444"},
  {VideoInfo::CS_YUVA420P10, "YUVA420P10"},
  {VideoInfo::CS_YUVA422P10, "YUVA422P10"},
  {VideoInfo::CS_YUVA444P10, "YUVA444P10"},
  {VideoInfo::CS_YUVA420P12, "YUVA420P12"},
  {VideoInfo::CS_YUVA422P12, "YUVA422P12"},
  {VideoInfo::CS_YUVA444P12, "YUVA444P12"},
  {VideoInfo::CS_YUVA420P14, "YUVA420P14"},
  {VideoInfo::CS_YUVA422P14, "YUVA422P14"},
  {VideoInfo::CS_YUVA444P14, "YUVA444P14"},
  {VideoInfo::CS_YUVA420P16, "YUVA420P16"},
  {VideoInfo::CS_YUVA422P16, "YUVA422P16"},
  {VideoInfo::CS_YUVA444P16, "YUVA444P16"},
  {VideoInfo::CS_YUVA420PS , "YUVA420PS"},
  {VideoInfo::CS_YUVA422PS , "YUVA422PS"},
  {VideoInfo::CS_YUVA444PS , "YUVA444PS"},

  {VideoInfo::CS_RGBAP     , "RGBAP"},
  {VideoInfo::CS_RGBAP10   , "RGBAP10"},
  {VideoInfo::CS_RGBAP12   , "RGBAP12"},
  {VideoInfo::CS_RGBAP14   , "RGBAP14"},
  {VideoInfo::CS_RGBAP16   , "RGBAP16"},
  {VideoInfo::CS_RGBAPS    , "RGBAPS"},
};

static const std::multimap<int, std::string> pixel_format_table_ex =
{ // alternative names for lookup by name (multimap!)
  {VideoInfo::CS_YV24 , "YUV444"},
  {VideoInfo::CS_YV16 , "YUV422"},
  {VideoInfo::CS_YV12 , "YUV420"},
  {VideoInfo::CS_YV411, "YUV411"},
  {VideoInfo::CS_RGBP , "RGBP8"},
  {VideoInfo::CS_RGBAP, "RGBAP8"},
  {VideoInfo::CS_YV24 , "YUV444P8"},
  {VideoInfo::CS_YV16 , "YUV422P8"},
  {VideoInfo::CS_YV12 , "YUV420P8"},
  {VideoInfo::CS_YV411, "YUV411P8"},
  {VideoInfo::CS_YUVA420, "YUVA420P8"},
  {VideoInfo::CS_YUVA422, "YUVA422P8"},
  {VideoInfo::CS_YUVA444, "YUVA444P8"},
};

const char *GetPixelTypeName(const int pixel_type)
{
  const std::string name = "";
  auto it = pixel_format_table.find(pixel_type);
  if (it == pixel_format_table.end())
    return "";
  return (it->second).c_str();
}

int GetPixelTypeFromName(const char *pixeltypename)
{
  std::string name_to_find = pixeltypename;
  for (auto & c: name_to_find) c = toupper(c); // uppercase input string
  for (auto it = pixel_format_table.begin(); it != pixel_format_table.end(); it++)
  {
    if ((it->second).compare(name_to_find) == 0)
      return it->first;
  }
  // find by alternative names e.g. YUV420 or YUV420P8 instead of YV12
  for (auto it = pixel_format_table_ex.begin(); it != pixel_format_table_ex.end(); it++)
  {
    if ((it->second).compare(name_to_find) == 0)
      return it->first;
  }
  return VideoInfo::CS_UNKNOWN;
}


AVSValue PixelType (AVSValue args, void*, IScriptEnvironment*) {
  return GetPixelTypeName(VI(args[0]).pixel_type);
}

// AVS+
AVSValue ColorSpaceNameToPixelType (AVSValue args, void*, IScriptEnvironment*) {
  return GetPixelTypeFromName(args[0].AsString());
}

AVSValue Width(AVSValue args, void*, IScriptEnvironment*) { return VI(args[0]).width; }
AVSValue Height(AVSValue args, void*, IScriptEnvironment*) { return VI(args[0]).height; }
AVSValue FrameCount(AVSValue args, void*, IScriptEnvironment*) { return VI(args[0]).num_frames; }
AVSValue FrameRate(AVSValue args, void*, IScriptEnvironment*) { const VideoInfo& vi = VI(args[0]); return (double)vi.fps_numerator / vi.fps_denominator; } // maximise available precision
AVSValue FrameRateNumerator(AVSValue args, void*, IScriptEnvironment*) { return (int)VI(args[0]).fps_numerator; } // unsigned int truncated to int
AVSValue FrameRateDenominator(AVSValue args, void*, IScriptEnvironment*) { return (int)VI(args[0]).fps_denominator; } // unsigned int truncated to int
AVSValue AudioRate(AVSValue args, void*, IScriptEnvironment*) { return VI(args[0]).audio_samples_per_second; }
AVSValue AudioLength(AVSValue args, void*, IScriptEnvironment*) { return (int)VI(args[0]).num_audio_samples; }  // Truncated to int
AVSValue AudioLengthLo(AVSValue args, void*, IScriptEnvironment*) { return (int)(VI(args[0]).num_audio_samples % (unsigned)args[1].AsInt(1000000000)); }
AVSValue AudioLengthHi(AVSValue args, void*, IScriptEnvironment*) { return (int)(VI(args[0]).num_audio_samples / (unsigned)args[1].AsInt(1000000000)); }
AVSValue AudioLengthS(AVSValue args, void*, IScriptEnvironment* env) {
    char s[32];
#ifdef AVS_WINDOWS
    return env->SaveString(_i64toa(VI(args[0]).num_audio_samples, s, 10));
#else
    sprintf(s, "%" PRId64, VI(args[0]).num_audio_samples);
    return env->SaveString(s);
#endif
}
AVSValue AudioLengthF(AVSValue args, void*, IScriptEnvironment*) { return (float)VI(args[0]).num_audio_samples; } // at least this will give an order of the size
AVSValue AudioDuration(AVSValue args, void*, IScriptEnvironment*) {
  const VideoInfo& vi = VI(args[0]);
  return (double)vi.num_audio_samples / vi.audio_samples_per_second;
}

AVSValue AudioChannels(AVSValue args, void*, IScriptEnvironment*) {  return VI(args[0]).HasAudio() ? VI(args[0]).nchannels : 0; }
AVSValue AudioBits(AVSValue args, void*, IScriptEnvironment*) {  return VI(args[0]).BytesPerChannelSample()*8; }
AVSValue IsAudioFloat(AVSValue args, void*, IScriptEnvironment*) {  return VI(args[0]).IsSampleType(SAMPLE_FLOAT); }
AVSValue IsAudioInt(AVSValue args, void*, IScriptEnvironment*) {  return VI(args[0]).IsSampleType(SAMPLE_INT8 | SAMPLE_INT16 | SAMPLE_INT24 | SAMPLE_INT32 ); }

AVSValue IsRGB(AVSValue args, void*, IScriptEnvironment*) {  return VI(args[0]).IsRGB(); }
AVSValue IsRGB24(AVSValue args, void*, IScriptEnvironment*) {  return VI(args[0]).IsRGB24(); }
AVSValue IsRGB32(AVSValue args, void*, IScriptEnvironment*) {  return VI(args[0]).IsRGB32(); }
AVSValue IsYUV(AVSValue args, void*, IScriptEnvironment*) {  return VI(args[0]).IsYUV(); }
AVSValue IsYUY2(AVSValue args, void*, IScriptEnvironment*) {  return VI(args[0]).IsYUY2(); }
AVSValue IsY8(AVSValue args, void*, IScriptEnvironment*) {  return VI(args[0]).IsY8(); }
AVSValue IsYV12(AVSValue args, void*, IScriptEnvironment*) {  return VI(args[0]).IsYV12(); }
AVSValue IsYV16(AVSValue args, void*, IScriptEnvironment*) {  return VI(args[0]).IsYV16(); }
AVSValue IsYV24(AVSValue args, void*, IScriptEnvironment*) {  return VI(args[0]).IsYV24(); }
AVSValue IsYV411(AVSValue args, void*, IScriptEnvironment*) {  return VI(args[0]).IsYV411(); }
AVSValue IsPlanar(AVSValue args, void*, IScriptEnvironment*) {  return VI(args[0]).IsPlanar(); }
AVSValue IsInterleaved(AVSValue args, void*, IScriptEnvironment*) {  return VI(args[0]).IsColorSpace(VideoInfo::CS_INTERLEAVED); }
AVSValue IsFieldBased(AVSValue args, void*, IScriptEnvironment*) {  return VI(args[0]).IsFieldBased(); }
AVSValue IsFrameBased(AVSValue args, void*, IScriptEnvironment*) {  return !VI(args[0]).IsFieldBased(); }
AVSValue GetParity(AVSValue args, void*, IScriptEnvironment*) {  return args[0].AsClip()->GetParity(args[1].AsInt(0)); }

AVSValue HasVideo(AVSValue args, void*, IScriptEnvironment*) {  return VI(args[0]).HasVideo(); }
AVSValue HasAudio(AVSValue args, void*, IScriptEnvironment*) {  return VI(args[0]).HasAudio(); }

AVSValue String(AVSValue args, void*, IScriptEnvironment* env)
{
  if (args[0].IsString()) return args[0];
  if (args[0].IsBool()) return (args[0].AsBool() ? "true" : "false");
  if (args[0].IsFunction()) return args[0].AsFunction()->ToString(env);
  if (args[1].Defined()) {	// WE --> a format parameter is present
    if (args[0].IsFloat()) {	//if it is an Int: IsFloat gives True, also !
      return  env->Sprintf(args[1].AsString("%f"), args[0].AsFloat());
    }
    return "";	// <--WE
  }
  else {	// standard behaviour
    if (args[0].IsInt()) {
      char s[12];
#ifdef AVS_WINDOWS
      return env->SaveString(_itoa(args[0].AsInt(), s, 10));
#else // copied from AvxSynth
      sprintf(s, "%d", args[0].AsInt());
      return env->SaveString(s);
#endif
    }
    if (args[0].IsFloat()) {
      char s[30];
#ifdef MSVC
      _locale_t locale = _create_locale(LC_NUMERIC, "C"); // decimal point: dot
      _sprintf_l(s, "%lf", locale, args[0].AsFloat());
      _free_locale(locale);
#else
      sprintf(s, "%lf", args[0].AsFloat());
#endif
      return env->SaveString(s);
    }
  }
  return "";
}

AVSValue Hex(AVSValue args, void*, IScriptEnvironment* env)
{
  int n = args[0].AsInt();
  int wid = args[1].AsInt(0); // 0..8 is the minimum width of the returned string
  wid = (wid<0) ? 0 : (wid > 8) ? 8 : wid;
  char buf[8 + 1];
  sprintf_s(buf, "%0*X", wid, n); // uppercase, unlike <=r2580
  return env->SaveString(buf);
}

static std::string AVSValue_to_string(AVSValue v, IScriptEnvironment* env) {
  if (v.IsString()) return v.AsString();
  if (v.IsBool()) return v.AsBool() ? "true" : "false";
  if (v.IsFunction()) return v.AsFunction()->ToString(env);
  if (v.IsInt()) return std::to_string(v.AsInt());
  if (v.IsFloat()) {
    char s[30];
#ifdef MSVC
    _locale_t locale = _create_locale(LC_NUMERIC, "C"); // decimal point: dot
    _sprintf_l(s, "%lf", locale, v.AsFloat());
    _free_locale(locale);
#else
    sprintf(s, "%lf", v.AsFloat());
#endif
    return s;
  }
  return "";
}


// Formatting function with parameter list with ordered, indexed or named replacements
AVSValue FormatString(AVSValue args, void*, IScriptEnvironment* env)
{
  // Format("{} {}!", "Hello", "world")
  // max_pixel_value = 255
  // Format("max pixel value = {max_pixel_value}!")
  // Format("Pi={1} x={0} y={0}!", 12, Pi())
  std::string format = args[0].AsString();
  int numargs = args[1].ArraySize();

  // (name), value pairs, name can be empty
  std::vector<std::pair<std::string, std::string>> sv;
  for (int i = 0; i < numargs; i++)
  {
    std::string name; // can be empty
    std::string val_as_s;

    AVSValue v = args[1][i];
    // ["name", value] support
    if (v.IsArray()) {
      if (v.ArraySize() != 2 || !v[0].IsString())
        env->ThrowError("Format: for key-value lookup parameter must be in [\"name\", value] array format");
      name = v[0].AsString();
      v = v[1];
    }

    val_as_s = AVSValue_to_string(v, env);

    sv.push_back(std::make_pair(name, val_as_s));
  }

  size_t supplied_params_count = sv.size();

  size_t len = format.size();
  size_t i = 0;
  bool in_parenthesis = false;
  std::string ss;
  size_t last_pos = 0;
  std::string last_param_section;

  size_t param_counter = 0;

  while (i < len) {
    if (!in_parenthesis) {
      size_t x = format.find_first_of('{', last_pos);
      // }} can appear only when escaped
      size_t cx = format.find_first_of('}', last_pos);
      if (cx != std::string::npos && cx < x)
      {
        if (cx + 1 < len && format[cx + 1] == '}') // }} escaped
        {
          ss += format.substr(last_pos, cx - last_pos + 1);
          last_pos = cx + 2;
          i = last_pos;
          continue;
        }
        env->ThrowError("Format: unbalanced curly bracket at position %zu", cx);
      }

      if (x == std::string::npos) // { not found
      {
        ss += format.substr(last_pos); // copy rest
        break;
      }
      else if (x + 1 < len && format[x + 1] == '{') // {{ escaped
      {
        ss += format.substr(last_pos, x - last_pos + 1);
        last_pos = x + 2;
        i = last_pos;
      }
      else {
        // found {, not escaped
        ss += format.substr(last_pos, x - last_pos);
        last_pos = x + 1; // points to after the {
        i = last_pos;
        in_parenthesis = true;
      }
      continue;
    } // end of not-in-bracket-mode

    // in-curly-bracket: search for the closing bracket

    size_t x = format.find_first_of('}', last_pos);
    if (x == std::string::npos) // not found, will throw error outside
      break;

    last_param_section = format.substr(last_pos, x - last_pos); // name, order number or empty

    if (last_param_section.empty()) {
      // simple {}, insert next parameter, consume one from the list
      // in c++20 you cannot mix
      if (param_counter >= supplied_params_count)
        env->ThrowError("Format: more parameter sections than parameters supplied");
      ss += sv[param_counter++].second;
    }
    else {
      // name or number

      bool validName = true;
      // check for a valid identifier name
      auto ch = last_param_section[0];
      if (ch != '_' && !isalpha(ch))
        validName = false;
      else {
        for (size_t i = 1; i < last_param_section.length(); i++) {
          const char ch = last_param_section[i];
          if (!(ch == '_' || isalnum(ch))) {
            validName = false;
            break;
          }
        }
      }

      if (!validName) {
        // valid number like {2} to index parameters
        int index;
        try {
          // string -> integer
          index = std::stoi(last_param_section);
        }
        catch (...) {
          env->ThrowError("Format: invalid parameter specifier: \"%s\".", last_param_section.c_str());
        }

        if (index < 0 || index >= (int)supplied_params_count)
          env->ThrowError("Format: parameter index is out of range: %d", index);

        ss += sv[index].second;
      }
      else {
        // find among the named parameters
        auto it = std::find_if(sv.begin(), sv.end(),
          [&last_param_section](const std::pair<std::string, std::string>& element) { return element.first == last_param_section; });
        if (it != sv.end())
          ss += it->second; // name was found
        else {
          // last resort: variable with the given name
          AVSValue v;
          if (!env->GetVarTry(last_param_section.c_str(), &v))
            env->ThrowError("Format: no parameter or variable found with name \"%s\".", last_param_section.c_str());

          std::string val_as_s = AVSValue_to_string(v, env);

          ss += val_as_s;
        }
      }
    }

    last_pos = x + 1;
    i = last_pos;
    in_parenthesis = false;
  }

  if(in_parenthesis)
    env->ThrowError("Format: unclosed curly bracket");

  return env->SaveString(ss.c_str());
}

AVSValue Func(AVSValue args, void*, IScriptEnvironment*) { return args[0]; }

AVSValue IsBool(AVSValue args, void*, IScriptEnvironment*) {  return args[0].IsBool(); }
AVSValue IsInt(AVSValue args, void*, IScriptEnvironment*) {  return args[0].IsInt(); }
AVSValue IsFloat(AVSValue args, void*, IScriptEnvironment*) {  return args[0].IsFloat(); }
AVSValue IsString(AVSValue args, void*, IScriptEnvironment*) {  return args[0].IsString(); }
AVSValue IsClip(AVSValue args, void*, IScriptEnvironment*) {  return args[0].IsClip(); }
AVSValue IsFunction(AVSValue args, void*, IScriptEnvironment*) { return args[0].IsFunction(); }
AVSValue Defined(AVSValue args, void*, IScriptEnvironment*) {  return args[0].Defined(); }

const char* GetAVSTypeName(AVSValue value) {
  if (value.IsClip())
    return "clip";
  else if (value.IsBool())
    return "bool";
  else if (value.IsInt())
    return "int";
  else if (value.IsFloat())
    return "float";
  else if (value.IsString())
    return "string";
  else if (value.IsArray())
    return "array";
  else if (value.IsFunction())
    return "function";
  else if (!value.Defined())
    return "undefined value";
  else
    return "unknown type";
}

AVSValue TypeName(AVSValue args, void*, IScriptEnvironment*) { return GetAVSTypeName(args[0]); }

AVSValue Default(AVSValue args, void*, IScriptEnvironment*) {  return args[0].Defined() ? args[0] : args[1]; }
AVSValue VersionNumber(AVSValue args, void*, IScriptEnvironment*) {  return AVS_CLASSIC_VERSION; }
AVSValue VersionString(AVSValue args, void*, IScriptEnvironment*) {  return AVS_FULLVERSION; }
AVSValue IsVersionOrGreater(AVSValue args, void*, IScriptEnvironment* env)
{
  if (!args[0].Defined() || !args[1].Defined()) {
    env->ThrowError("IsVersionOrGreater error: at least two parameters (majorVersion, minorVersion) required!");
  }

    // true when current version is at least the same or greater than the given three-part version
  const int majorVersion = args[0].AsInt(0);
  const int minorVersion = args[1].AsInt(0);
  const int bugfixVersion = args[2].AsInt(0);
  if (majorVersion != AVS_MAJOR_VER) return majorVersion < AVS_MAJOR_VER;
  if (minorVersion != AVS_MINOR_VER) return minorVersion < AVS_MINOR_VER;
  return bugfixVersion <= AVS_BUGFIX_VER;
}

AVSValue Int(AVSValue args, void*, IScriptEnvironment*) {  return int(args[0].AsFloat()); }
AVSValue Frac(AVSValue args, void*, IScriptEnvironment*) {  return args[0].AsFloat() - int64_t(args[0].AsFloat()); }
AVSValue Float(AVSValue args, void*, IScriptEnvironment*) {  return args[0].AsFloat(); }

AVSValue Value(AVSValue args, void*, IScriptEnvironment*) {  char *stopstring; return strtod(args[0].AsString(),&stopstring); }
AVSValue HexValue(AVSValue args, void*, IScriptEnvironment*)
{
  // Added optional pos arg default = 1, start position in string of the HexString, 1 denotes the string beginning.
  // Will return 0 if error in 'pos' ie if pos is less than 1 or greater than string length.
  const char *str = args[0].AsString();
  int pos = args[1].AsInt(1) - 1;
  int sz = static_cast<int>(strlen(str));
  if (pos<0 || pos >= sz)
    return 0;
  str += pos;
  char *stopstring;
  return (int)(strtoul(str, &stopstring, 16));
}

AVSValue AvsMin(AVSValue args, void*, IScriptEnvironment* env )
{
  int i;
  bool isInt = true;

  const int n = args[0].ArraySize();
  if (n < 2) env->ThrowError("To few arguments for Min");

  // If all numbers are Ints return an Int
  for (i=0; i < n; i++)
    if (!args[0][i].IsInt()) {
      isInt = false;
      break;
  }

  if (isInt) {
    int V = args[0][0].AsInt();
    for (i=1; i < n; i++)
      V = min(V, args[0][i].AsInt());
    return V;
  }
  else {
    float V = args[0][0].AsFloatf();
    for (i=1; i < n; i++)
      V = min(V, args[0][i].AsFloatf());
    return V;
  }
}

AVSValue AvsMax(AVSValue args, void*, IScriptEnvironment* env )
{
  int i;
  bool isInt = true;

  const int n = args[0].ArraySize();
  if (n < 2) env->ThrowError("To few arguments for Max");

  // If all numbers are Ints return an Int
  for (i=0; i < n; i++)
    if (!args[0][i].IsInt()) {
      isInt = false;
      break;
  }

  if (isInt) {
    int V = args[0][0].AsInt();
    for (i=1; i < n; i++)
      V = max(V, args[0][i].AsInt());
    return V;
  }
  else {
    float V = args[0][0].AsFloatf();
    for (i=1; i < n; i++)
      V = max(V, args[0][i].AsFloatf());
    return V;
  }
}

AVSValue AddAutoloadDir (AVSValue args, void*, IScriptEnvironment* env)
{
  IScriptEnvironment2 *env2 = static_cast<IScriptEnvironment2*>(env);
  env2->AddAutoloadDir(args[0].AsString(), args[1].AsBool(true));
  return AVSValue();
}

AVSValue ClearAutoloadDirs (AVSValue args, void*, IScriptEnvironment* env)
{
  IScriptEnvironment2 *env2 = static_cast<IScriptEnvironment2*>(env);
  env2->ClearAutoloadDirs();
  return AVSValue();
}

AVSValue AutoloadPlugins (AVSValue args, void*, IScriptEnvironment* env)
{
  IScriptEnvironment2 *env2 = static_cast<IScriptEnvironment2*>(env);
  env2->AutoloadPlugins();
  return AVSValue();
}

AVSValue FunctionExists (AVSValue args, void*, IScriptEnvironment* env)
{
  return env->FunctionExists(args[0].AsString());
}

AVSValue InternalFunctionExists (AVSValue args, void*, IScriptEnvironment* env)
{
  IScriptEnvironment2 *env2 = static_cast<IScriptEnvironment2*>(env);
  return env2->InternalFunctionExists(args[0].AsString());
}

AVSValue SetFilterMTMode (AVSValue args, void*, IScriptEnvironment* env)
{
  IScriptEnvironment2 *env2 = static_cast<IScriptEnvironment2*>(env);
  env2->SetFilterMTMode(args[0].AsString(), (MtMode)args[1].AsInt(), args[2].AsBool(false));
  return AVSValue();
}

AVSValue SetLogParams(AVSValue args, void*, IScriptEnvironment* env)
{
    const char *target = NULL;
    int level = -1;

    if (1 <= args.ArraySize())
    {
        if (args[0].IsString()) {
            target = args[0].AsString();
        }
        else {
            env->ThrowError("1st argument to SetLogParams() must be a string.");
            return AVSValue();
        }
    }

    if (2 <= args.ArraySize())
    {
        if (args[1].IsInt()) {
            level = args[1].AsInt();
        }
        else {
            env->ThrowError("2nd argument to SetLogParams() must be an integer.");
            return AVSValue();
        }
    }

    if (3 <= args.ArraySize())
    {
        env->ThrowError("Too many arguments to SetLogParams().");
        return AVSValue();
    }

    InternalEnvironment *envi = static_cast<InternalEnvironment*>(env);
    envi->SetLogParams(target, level);
    return AVSValue();
}

AVSValue LogMsg(AVSValue args, void*, IScriptEnvironment* env)
{
    if ((args.ArraySize() != 2) || !args[0].IsString() || !args[1].IsInt())
    {
        env->ThrowError("Invalid parameters to Log() function.");
    }
    else
    {
        InternalEnvironment *envi = static_cast<InternalEnvironment*>(env);
        envi->LogMsg(args[1].AsInt(), args[0].AsString());
    }
    return AVSValue();
}

AVSValue SetCacheMode(AVSValue args, void*, IScriptEnvironment* env)
{
	InternalEnvironment *envI = static_cast<InternalEnvironment*>(env);
  envI->SetCacheMode((CacheMode)args[0].AsInt());
	return AVSValue();
}

AVSValue SetDeviceOpt(AVSValue args, void*, IScriptEnvironment* env)
{
    InternalEnvironment *envI = static_cast<InternalEnvironment*>(env);
    envI->SetDeviceOpt((DeviceOpt)args[0].AsInt(), args[1].AsInt(0));
    return AVSValue();
}

// Neo style
AVSValue SetMemoryMax(AVSValue args, void*, IScriptEnvironment* env)
{
  int memMax = args[0].AsInt(0);
  int deviceType = args[1].AsInt(0);
  int deviceIndex = args[2].AsInt(0);

  if (deviceType == 0 || deviceType == DEV_TYPE_CPU) {
    return env->SetMemoryMax(memMax);
  }

  InternalEnvironment *envI = static_cast<InternalEnvironment*>(env);
  return envI->SetMemoryMax((AvsDeviceType)deviceType, deviceIndex, memMax);
}

AVSValue SetMaxCPU(AVSValue args, void*, IScriptEnvironment* env)
{
  InternalEnvironment* envI = static_cast<InternalEnvironment*>(env);
  envI->SetMaxCPU(args[0].AsString());
  return AVSValue();
}

AVSValue IsY(AVSValue args, void*, IScriptEnvironment*) {  return VI(args[0]).IsY(); }
AVSValue Is420(AVSValue args, void*, IScriptEnvironment*) {  return VI(args[0]).Is420(); }
AVSValue Is422(AVSValue args, void*, IScriptEnvironment*) {  return VI(args[0]).Is422(); }
AVSValue Is444(AVSValue args, void*, IScriptEnvironment*) {  return VI(args[0]).Is444(); }
AVSValue IsRGB48(AVSValue args, void*, IScriptEnvironment*) {  return VI(args[0]).IsRGB48(); }
AVSValue IsRGB64(AVSValue args, void*, IScriptEnvironment*) {  return VI(args[0]).IsRGB64(); }
AVSValue ComponentSize(AVSValue args, void*, IScriptEnvironment*) {  return VI(args[0]).ComponentSize(); }
AVSValue BitsPerComponent(AVSValue args, void*, IScriptEnvironment*) {  return VI(args[0]).BitsPerComponent(); }
AVSValue IsYUVA(AVSValue args, void*, IScriptEnvironment*) {  return VI(args[0]).IsYUVA(); }
AVSValue IsPlanarRGB(AVSValue args, void*, IScriptEnvironment*) {  return VI(args[0]).IsPlanarRGB(); }
AVSValue IsPlanarRGBA(AVSValue args, void*, IScriptEnvironment*) {  return VI(args[0]).IsPlanarRGBA(); }
AVSValue NumComponents(AVSValue args, void*, IScriptEnvironment*) {  return VI(args[0]).NumComponents(); }
AVSValue HasAlpha(AVSValue args, void*, IScriptEnvironment*) {  return VI(args[0]).IsPlanarRGBA() || VI(args[0]).IsYUVA() || VI(args[0]).IsRGB32() || VI(args[0]).IsRGB64(); }
AVSValue IsPackedRGB(AVSValue args, void*, IScriptEnvironment*) {  return VI(args[0]).IsRGB24() || VI(args[0]).IsRGB32() || VI(args[0]).IsRGB48() || VI(args[0]).IsRGB64(); }
AVSValue IsVideoFloat(AVSValue args, void*, IScriptEnvironment*) {  return VI(args[0]).BitsPerComponent() == 32; }

// helper for GetProcessInfo
static int ProcessType() {
#define PROCESS_UNKNOWN  -1
#define PROCESS_32_ON_32 0
#define PROCESS_32_ON_64 1
#define PROCESS_64_ON_64 2

  if constexpr(sizeof(void*) == 8)
    return PROCESS_64_ON_64;
#ifdef AVS_WINDOWS
  else {
    // IsWow64Process is not available on all supported versions of Windows.
    // Use GetModuleHandle to get a handle to the DLL that contains the function
    // and GetProcAddress to get a pointer to the function if available.

    BOOL bWoW64Process = FALSE;
    typedef bool(WINAPI *LPFN_ISWOW64PROCESS) (HANDLE, PBOOL);
    LPFN_ISWOW64PROCESS fnIsWow64Process;
    HMODULE hKernel32 = GetModuleHandle("kernel32.dll");
    if (hKernel32 == NULL)
      return PROCESS_UNKNOWN;

    fnIsWow64Process = (LPFN_ISWOW64PROCESS)GetProcAddress(hKernel32, "IsWow64Process");
    if (fnIsWow64Process != NULL)
      fnIsWow64Process(GetCurrentProcess(), &bWoW64Process);
    else
      return PROCESS_UNKNOWN;

    if (bWoW64Process)
      return PROCESS_32_ON_64; //WoW64

    return PROCESS_32_ON_32;
  }
#endif
}

AVSValue GetProcessInfo(AVSValue args, void*, IScriptEnvironment* env)
{
  int infoType = args[0].AsInt(0);
  if (infoType < 0 || infoType > 1)
    env->ThrowError("GetProcessInfo: type must be 0 or 1");
  if (infoType == 0) {
    return sizeof(void *) == 8 ? 64 : 32;
  }
  // infoType == 1
  return ProcessType();
}

#ifdef AVS_WINDOWS
AVSValue StrToUtf8(AVSValue args, void*, IScriptEnvironment* env) {
  const char *source = args[0].AsString();
  // in two steps: Ansi -> WideChar -> Utf8
  auto wsource = AnsiToWideCharACP(source);
  // wide -> utf8
  auto source_utf8 = WideCharToUtf8(wsource.get());
  AVSValue ret = env->SaveString(source_utf8.get());
  return ret;
}

AVSValue StrFromUtf8(AVSValue args, void*, IScriptEnvironment* env) {
  const char *source_utf8 = args[0].AsString();
  // in two steps: Utf8 -> WideChar -> Ansi
  auto wsource = Utf8ToWideChar(source_utf8);
  // wide -> ansi
  auto source_ansi = WideCharToAnsiACP(wsource.get());
  AVSValue ret = env->SaveString(source_ansi.get());
  return ret;
}
#endif


AVSValue IsFloatUvZeroBased(AVSValue args, void*, IScriptEnvironment*)
{
#ifdef FLOAT_CHROMA_IS_HALF_CENTERED
  return false;
#else
  return true;
#endif
}

AVSValue BuildPixelType(AVSValue args, void*, IScriptEnvironment* env)
{
  //  { "BuildPixelType", BUILTIN_FUNC_PREFIX, "[family]s[bits]i[chroma]i[compat]b[oldnames]b[sample_clip]c", BuildPixelType }, // 180517-
  // family: YUV, YUVA, RGB, RGBA, Y
  // bits: 8, 10, 12, 14, 16, 32
  // chroma: for YUV(A) 420,422,444,411. Ignored for RGB(A) and Y
  // compat (default false): returns packed rgb formats for 8/16 bits (RGB default: planar RGB)
  // oldnames (default false): returns YV12/YV16/YV24 instead of YUV420P8/YUV422P8/YUV444P8
  // sample_clip: when supported, its format is overridden by specified parameters (e.g. only change bits=10)

  const bool hasTemplate = args[5].Defined();

  if (!args[0].Defined() && !hasTemplate)
    env->ThrowError("BuildPixelType error: no color space 'family' or template 'sample_clip' specified");
  if (!args[1].Defined() && !hasTemplate)
    env->ThrowError("BuildPixelType error: no 'bits' or  template 'sample_clip' specified");

  std::string family;
  if (!args[0].Defined() && hasTemplate) {
    // no family parameter: use template
    VideoInfo const &vi = args[5].AsClip()->GetVideoInfo();
    if (vi.IsY())
      family = "Y";
    else if (vi.IsPlanar()) {
      if (vi.IsYUV())
        family = "YUV";
      else if (vi.IsYUVA())
        family = "YUVA";
      else if (vi.IsPlanarRGB())
        family = "RGB";
      else if (vi.IsPlanarRGBA())
        family = "RGBA";
      else
        env->ThrowError("BuildPixelType error: invalid sample_clip format");
    }
    else if (vi.IsRGB24() || vi.IsRGB48())
      family = "RGB";
    else if (vi.IsRGB32() || vi.IsRGB64())
      family = "RGBA";
    else
      env->ThrowError("BuildPixelType error: invalid sample_clip format");
  }
  else {
    family = args[0].AsString();
    for (auto & c : family) c = toupper(c); // uppercase input string
  }

  const bool isYUV = family == "YUV";
  const bool isYUVA = family == "YUVA";
  const bool isRGB = family == "RGB";
  const bool isRGBA = family == "RGBA";
  const bool isY = family == "Y";

  if(!isYUV && !isYUVA && !isRGB && !isRGBA && !isY)
    env->ThrowError("BuildPixelType error: wrong 'family'.", family.c_str());

  int bits;
  if (!args[1].Defined() && hasTemplate) {
    // no bits parameter: get it from template sample_clip
    bits = args[5].AsClip()->GetVideoInfo().BitsPerComponent();
  } else {
    bits = args[1].AsInt();
  }

  if (bits != 8 && bits != 10 && bits != 12 && bits != 14 && bits != 16 && bits != 32)
    env->ThrowError("BuildPixelType error: 'bits'=%d is not valid.", bits);

  int chroma;

  if (isYUV || isYUVA) {
    if (!args[2].Defined() && hasTemplate) {
      // no chroma parameter: subsampling from template clip
      VideoInfo const &vi = args[5].AsClip()->GetVideoInfo();
      const int hs = vi.GetPlaneWidthSubsampling(PLANAR_U);
      const int vs = vi.GetPlaneHeightSubsampling(PLANAR_U);
      if (hs == 0 && vs == 0) chroma = 444;
      else if (hs == 1 && vs == 0) chroma = 422;
      else if (hs == 1 && vs == 1) chroma = 420;
      else if (hs == 2 && vs == 0) chroma = 411;
      else
        env->ThrowError("BuildPixelType error: sample_clip has invalid chroma subsampling.");
    }
    else {
      chroma = args[2].AsInt(444);
    }
  }
  else {
    chroma = 444; // n/a
  }

  if(chroma != 444 && chroma != 422 && chroma != 420 && chroma != 411)
    env->ThrowError("BuildPixelType error: 'chroma' must be 444, 422, 420 or 411.");

  // packed RGB compatibility formats only for RGB(A)
  const bool compat = isRGB || isRGBA ? args[3].AsBool(false) : false;

  // e.g. return YV12 instead of YUV420P8
  const bool oldNames = args[4].AsBool(false);

  if(compat && bits != 8 && bits != 16)
    env->ThrowError("BuildPixelType error: 'compat'=true requires bits=8 or 16 for RGB(A).");

  if(chroma == 411 && bits != 8)
    env->ThrowError("BuildPixelType error: 411 is supported only for 8 bits.");

  if (compat) {
    if (isRGB && bits == 8)
      return "RGB24";
    if (isRGB && bits == 16)
      return "RGB48";
    if (isRGBA && bits == 8)
      return "RGB32";
    return "RGB64"; // RGBA, bits==16
  }

  std::string format;

  if (isYUV || isYUVA || isY)
    format = family;
  else if (isRGB)
    format = "RGBP";
  else if (isRGBA)
    format = "RGBAP";

  if (isYUV || isYUVA) {
    if (chroma == 444)
      format += "444";
    else if(chroma == 422)
      format += "422";
    else if (chroma == 420)
      format += "420";
    else if (chroma == 411)
      format += "411";

    format = format + "P";
  }

  if (bits == 32)
      format += (isY ? "32" : "S"); // no "YS", only "Y32"
  else
    format = format + std::to_string(bits);

  if (oldNames) {
    if (format == "YUV420" || format == "YUV420P8") format = "YV12";
    else if (format == "YUV422" || format == "YUV422P8") format = "YV16";
    else if (format == "YUV444" || format == "YUV444P8") format = "YV24";
  }

  // 411 has no alternative naming
  if (format == "YUV411") format = "YV411";

  return env->SaveString(format.c_str());
}

AVSValue VarExist(AVSValue args, void*, IScriptEnvironment* env)
{
  const char *name = args[0].AsString();
  int len = (int)strlen(name);

  bool validName = true;
  // check for a valid identifier name
  if (*name != '_' && !isalpha(*name))
    validName = false;
  else {
    for (int i = 1; i < len; i++) {
      const char ch = name[i];
      if (!(ch == '_' || isalnum(ch))) {
        validName = false;
        break;
      }
    }
  }

  if (!validName)
    env->ThrowError("VarExist: invalid variable name");

  AVSValue result;
  return (env->GetVarTry(name, &result)); // true if exists
}


AVSValue ArrayCreate(AVSValue args, void*, IScriptEnvironment* env)
{
    return args[0];
}

AVSValue IsArray(AVSValue args, void*, IScriptEnvironment* env) { return args[0].IsArray(); }

AVSValue ArrayGet(AVSValue args, void*, IScriptEnvironment* env)
{
  // signature .i+
  // parameters: [0] array to index; [1] one or more integer indexes or a string
  if (!args[0].IsArray())
    env->ThrowError("ArrayGet: array type required.");
  const int size = args[0].ArraySize();
  if (args[1].IsString()) {
    // associative search
    // { {"a", element1}, { "b", element2 }, etc..}
const char* tag = args[1].AsString();
for (int i = 0; i < size; i++)
{
  AVSValue currentTagValue = args[0][i]; // two elements e.g. { "b", element2 }
  if (!currentTagValue.IsArray())
    env->ThrowError("ArrayGet: Array must contain array[string, any] elements for dictionary lookup");
  if (currentTagValue.ArraySize() < 2)
    env->ThrowError("ArrayGet: Internal array must have at least two elements (tag, value)");
  AVSValue currentTag = currentTagValue[0];
  if (currentTag.IsString() && !lstrcmpi(currentTag.AsString(), tag))
  {
    return currentTagValue[1];
  }
}
return AVSValue(); // undefined
  }
  else if (args[1].IsArray()) {
  AVSValue indexes = args[1];
  AVSValue currentValue = args[0];
  int index_count = indexes.ArraySize(); // array of parameters. a[1,2] -> [1,2]
  if (index_count == 0)
    env->ThrowError("ArrayGet: no index specified");
  for (int i = 0; i < index_count; i++)
  {
    if (!currentValue.IsArray())
      env->ThrowError("ArrayGet: not an array. Index=%d", i);
    int currentIndex = indexes[i].AsInt();
    if (currentIndex < 0 || currentIndex >= currentValue.ArraySize())
      env->ThrowError("ArrayGet: Array index out of range. Problematic index count: %d", i + 1);
    currentValue = currentValue[currentIndex];
  }
  return currentValue;
  }
  env->ThrowError("ArrayGet: Invalid array index, must be integer or string, or comma separated integers");
  return AVSValue(); // undefined
}

AVSValue ArraySize(AVSValue args, void*, IScriptEnvironment* env)
{
  // func signature: "."
  if (!args[0].IsArray())
    env->ThrowError("ArraySize: parameter must be an array");
  return args[0].ArraySize();
}

AVSValue ArrayIns(AVSValue args, void* user_data, IScriptEnvironment* env)
{
  int mode = (int)(intptr_t)user_data;
  enum ArrayMode {
    INSERT = 0,
    APPEND = 1,
    REPLACE = 2,
    DEL = 3
  };
  // signature .. and ..i
  // parameters:
  // [0] array to modify;
  // [1] element to insert (ArrayAdd, ArrayIns and ArraySet) [2] inserting index(es) (ArrayIns, ArraySet)
  // or [1] delete index(es) ArrayDel

  const char* funcname = mode == DEL ? "ArrayDel" : mode == REPLACE ? "ArraySet" : mode == APPEND ? "ArrayAdd" : "ArrayIns";

  if (!args[0].IsArray())
    env->ThrowError("%s error: array type required.", funcname);

  const auto orig_size = args[0].ArraySize();

  const int index_param_pos = mode == DEL ? 1 : 2;
  AVSValue indexes = args[index_param_pos];
  int index_count = indexes.ArraySize(); // array of parameters. a[1,2] -> [1,2]

  if (mode == INSERT || mode == REPLACE || mode == DEL) {
    if (index_count == 0)
      env->ThrowError("%s: no index specified", funcname);
  }

  const int new_size =
    mode == DEL && index_count == 1 ? orig_size - 1 :
    mode == APPEND && index_count == 0 ? orig_size + 1 :
    mode == INSERT && index_count == 1 ? orig_size + 1 :
    orig_size; // replace and recurside other cases

  std::vector<AVSValue> new_val(new_size);

  int action_pos;
  if (mode == APPEND)
    action_pos = orig_size; // at the end
  else
    action_pos = indexes[0].AsInt();

  // copy before insertion/replace point
  for (int i = 0; i < action_pos; i++)
    new_val[i] = args[0][i]; // avs+: automatic deep copy

  if (
    ((mode == REPLACE || mode == INSERT || mode == DEL) && index_count > 1) ||
    ((mode == APPEND) && index_count >= 1))
  {
    int current_index = indexes[0].AsInt();
    // for multi-level array recursion is needed because there is no exact reference to an inner element
    if (mode == DEL) {
      AVSValue params[2] = { args[0][current_index], index_count <= 1 ? AVSValue(nullptr, 0) : AVSValue(&indexes[1], index_count - 1) };
      new_val[current_index] = env->Invoke(funcname, AVSValue(params, 2)); // recursively
    }
    else {
      AVSValue params[3] = { args[0][current_index], args[1], index_count <= 1 ? AVSValue(nullptr, 0) : AVSValue(&indexes[1], index_count - 1) };
      new_val[current_index] = env->Invoke(funcname, AVSValue(params, 3)); // recursively
    }
    mode = REPLACE;
  }
  else if (mode != DEL) {
    new_val[action_pos] = args[1];
  }

  // copy from after insertion/replace/delete point
  if (mode == DEL) {
    for (int i = action_pos + 1; i < orig_size; i++)
      new_val[i - 1] = args[0][i]; // avs+: automatic deep copy
  }
  else if (mode == REPLACE) {
    for (int i = action_pos+1; i < orig_size; i++)
      new_val[i] = args[0][i]; // avs+: automatic deep copy
  }
  else {
    for (int i = action_pos; i < orig_size; i++)
      new_val[i + 1] = args[0][i]; // avs+: automatic deep copy
  }

  if(new_size == 0)
    return AVSValue(nullptr, 0); // zero array

  return AVSValue(new_val.data(), new_size);
}

