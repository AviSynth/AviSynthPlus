// Avisynth v2.5.  Copyright 2002-2009 Ben Rudiak-Gould et al.
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

#ifndef _AVS_COMPATENVIRONMENT_H_INCLUDED
#define _AVS_COMPATENVIRONMENT_H_INCLUDED

#include <avisynth.h>

// IScriptEnvironment_Avs25, IScriptEnvironment_AvsPreV11C are used internally.
// Entries 100% match with IScriptEnvironment to allow typecasting to IScriptEnvironment_Avs25/AvsPreV11.
// While IScriptEnvironment_Avs25 ends at interface V6 changes, PreV11C version is full copy of IScriptEnvironment.
// Differences involve three functions and the GetVar family:
// 1. AddFunction25/PreV11C: Marks the added function as originating from 
//    an Avs2.5/PreV11C plugin, see also: arrays-deep-copy, 64-bit params.
// 2. Invoke25/PreV11C: Argument arrays, which must not be deep-freed, 
//    baked AVSValue code incompatible with deep-array concept and 64 bit data.
// 3. ManageCache25/PreV11C: Specially returns 1 for key MC_QueryAvs25/PreV11C 
//    to check if called from AVS2.5/PreV11C interface for cache GetFrame/GetAudio.
// 4. GetVar25/PreV11C, GetVarDef25/PreV11C, GetVarTryPreV11C: the variable's 64 bit
//    'l'ong and 'd'ouble values are returned as 'i'nt and 'f'loat, like Invoke25/PreV11C results.

// See also: InternalEnvironment interface conversion methods comment.

// Compatibility camouflage classes are *passed in place of the original IScriptEnvironment,
// providing the same function orders to match their VMT tables with the original.
// Differently named functions appear at specific indexes to perform different compatibility behaviors.
// *In real they all are static casted from InternalEnvironment, where all these pure virtual methods are implemented.

class IScriptEnvironment_Avs25 {
public:
  virtual ~IScriptEnvironment_Avs25() {}

  virtual /*static*/ int __stdcall GetCPUFlags() = 0;

  virtual char* __stdcall SaveString(const char* s, int length = -1) = 0;
  virtual char* Sprintf(const char* fmt, ...) = 0;
  virtual char* __stdcall VSprintf(const char* fmt, va_list val) = 0;

#ifdef AVS_WINDOWS
  __declspec(noreturn) virtual void ThrowError(const char* fmt, ...) = 0;
#else
  virtual void ThrowError(const char* fmt, ...) = 0;
#endif

  class NotFound /*exception*/ {};  // thrown by Invoke and GetVar

  typedef AVSValue(__cdecl* ApplyFunc)(AVSValue args, void* user_data, IScriptEnvironment* env);

  virtual void __stdcall AddFunction25(const char* name, const char* params, ApplyFunc apply, void* user_data) = 0;
  virtual bool __stdcall FunctionExists(const char* name) = 0;
  virtual AVSValue __stdcall Invoke25(const char* name, const AVSValue args, const char* const* arg_names = 0) = 0;

  virtual AVSValue __stdcall GetVar25(const char* name) = 0;
  virtual bool __stdcall SetVar(const char* name, const AVSValue& val) = 0;
  virtual bool __stdcall SetGlobalVar(const char* name, const AVSValue& val) = 0;

  virtual void __stdcall PushContext(int level = 0) = 0;
  virtual void __stdcall PopContext() = 0;

  virtual PVideoFrame __stdcall NewVideoFrame(const VideoInfo& vi, int align = FRAME_ALIGN) = 0;

  virtual bool __stdcall MakeWritable(PVideoFrame* pvf) = 0;

  virtual void __stdcall BitBlt(BYTE* dstp, int dst_pitch, const BYTE* srcp, int src_pitch, int row_size, int height) = 0;

  typedef void(__cdecl* ShutdownFunc)(void* user_data, IScriptEnvironment* env);
  virtual void __stdcall AtExit(ShutdownFunc function, void* user_data) = 0;

  virtual void __stdcall CheckVersion(int version = AVISYNTH_CLASSIC_INTERFACE_VERSION_25) = 0;

  virtual PVideoFrame __stdcall Subframe(PVideoFrame src, int rel_offset, int new_pitch, int new_row_size, int new_height) = 0;

  virtual int __stdcall SetMemoryMax(int mem) = 0;

  virtual int __stdcall SetWorkingDir(const char* newdir) = 0;

  // specially returns 1 for key MC_QueryAvs25 to check if called from AVS2.5 interface
  virtual void* __stdcall ManageCache25(int key, void* data) = 0;

  enum PlanarChromaAlignmentMode {
    PlanarChromaAlignmentOff,
    PlanarChromaAlignmentOn,
    PlanarChromaAlignmentTest
  };

  virtual bool __stdcall PlanarChromaAlignment(IScriptEnvironment::PlanarChromaAlignmentMode key) = 0;

  virtual PVideoFrame __stdcall SubframePlanar(PVideoFrame src, int rel_offset, int new_pitch, int new_row_size,
    int new_height, int rel_offsetU, int rel_offsetV, int new_pitchUV) = 0;

  // Despite the name, we provide entries up to V6 in case someone requests
  // a V3 interface and still wants to use V5-V6 functions

  // **** AVISYNTH_INTERFACE_VERSION 5 **** defined since classic Avisynth 2.6 beta
  virtual void __stdcall DeleteScriptEnvironment() = 0;

  virtual void __stdcall ApplyMessage(PVideoFrame* frame, const VideoInfo& vi, const char* message, int size,
    int textcolor, int halocolor, int bgcolor) = 0;

  virtual const AVS_Linkage* __stdcall GetAVSLinkage() = 0;

  // **** AVISYNTH_INTERFACE_VERSION 6 **** defined since classic Avisynth 2.6
  // noThrow version of GetVar
  virtual AVSValue __stdcall GetVarDef25(const char* name, const AVSValue& def = AVSValue()) = 0;

}; // end class IScriptEnvironment_Avs25. Order is important.


// Unlike IScriptEnvironment_Avs25 which ends at the V6 changes,
// IScriptEnvironment_AvsPreV11C is a complete copy of IScriptEnvironment, 
// except the name of the above mentioned methods.
class IScriptEnvironment_AvsPreV11C {
public:
  virtual ~IScriptEnvironment_AvsPreV11C() {}

  virtual int __stdcall GetCPUFlags() = 0;

  virtual char* __stdcall SaveString(const char* s, int length = -1) = 0;
  virtual char* Sprintf(const char* fmt, ...) = 0;
  virtual char* __stdcall VSprintf(const char* fmt, va_list val) = 0;

#ifdef AVS_WINDOWS
  __declspec(noreturn) virtual void ThrowError(const char* fmt, ...) = 0;
#else
  virtual void ThrowError(const char* fmt, ...) = 0;
#endif

  class NotFound /*exception*/ {};  // thrown by Invoke and GetVar

  typedef AVSValue(__cdecl* ApplyFunc)(AVSValue args, void* user_data, IScriptEnvironment* env);

  virtual void __stdcall AddFunctionPreV11C(const char* name, const char* params, ApplyFunc apply, void* user_data) = 0;
  virtual bool __stdcall FunctionExists(const char* name) = 0;
  virtual AVSValue __stdcall InvokePreV11C(const char* name, const AVSValue args, const char* const* arg_names = 0) = 0;

  virtual AVSValue __stdcall GetVarPreV11C(const char* name) = 0;
  virtual bool __stdcall SetVar(const char* name, const AVSValue& val) = 0;
  virtual bool __stdcall SetGlobalVar(const char* name, const AVSValue& val) = 0;

  virtual void __stdcall PushContext(int level = 0) = 0;
  virtual void __stdcall PopContext() = 0;

  virtual PVideoFrame __stdcall NewVideoFrame(const VideoInfo& vi, int align = FRAME_ALIGN) = 0;

  virtual bool __stdcall MakeWritable(PVideoFrame* pvf) = 0;

  virtual void __stdcall BitBlt(BYTE* dstp, int dst_pitch, const BYTE* srcp, int src_pitch, int row_size, int height) = 0;

  typedef void(__cdecl* ShutdownFunc)(void* user_data, IScriptEnvironment* env);
  virtual void __stdcall AtExit(ShutdownFunc function, void* user_data) = 0;

  virtual void __stdcall CheckVersion(int version = AVISYNTH_CLASSIC_INTERFACE_VERSION_25) = 0;

  virtual PVideoFrame __stdcall Subframe(PVideoFrame src, int rel_offset, int new_pitch, int new_row_size, int new_height) = 0;

  virtual int __stdcall SetMemoryMax(int mem) = 0;

  virtual int __stdcall SetWorkingDir(const char* newdir) = 0;

  // specially returns 1 for key MC_QueryAvsPreV11C to check if called from C interface
  virtual void* __stdcall ManageCachePreV11C(int key, void* data) = 0;

  enum PlanarChromaAlignmentMode {
    PlanarChromaAlignmentOff,
    PlanarChromaAlignmentOn,
    PlanarChromaAlignmentTest
  };

  virtual bool __stdcall PlanarChromaAlignment(IScriptEnvironment::PlanarChromaAlignmentMode key) = 0;

  virtual PVideoFrame __stdcall SubframePlanar(PVideoFrame src, int rel_offset, int new_pitch, int new_row_size,
    int new_height, int rel_offsetU, int rel_offsetV, int new_pitchUV) = 0;

  // **** AVISYNTH_INTERFACE_VERSION 5 **** defined since classic Avisynth 2.6 beta
  virtual void __stdcall DeleteScriptEnvironment() = 0;

  virtual void __stdcall ApplyMessage(PVideoFrame* frame, const VideoInfo& vi, const char* message, int size,
    int textcolor, int halocolor, int bgcolor) = 0;

  virtual const AVS_Linkage* __stdcall GetAVSLinkage() = 0;

  // **** AVISYNTH_INTERFACE_VERSION 6 **** defined since classic Avisynth 2.6
  // noThrow version of GetVar
  virtual AVSValue __stdcall GetVarDefPreV11C(const char* name, const AVSValue& def = AVSValue()) = 0;


  // **** AVISYNTH_INTERFACE_VERSION 8 **** AviSynth+ 3.6.0-
  virtual PVideoFrame __stdcall SubframePlanarA(PVideoFrame src, int rel_offset, int new_pitch, int new_row_size,
    int new_height, int rel_offsetU, int rel_offsetV, int new_pitchUV, int rel_offsetA) = 0;

  virtual void __stdcall copyFrameProps(const PVideoFrame& src, PVideoFrame& dst) = 0;
  virtual const AVSMap* __stdcall getFramePropsRO(const PVideoFrame& frame) = 0;
  virtual AVSMap* __stdcall getFramePropsRW(PVideoFrame& frame) = 0;

  virtual int __stdcall propNumKeys(const AVSMap* map) = 0;

  virtual const char* __stdcall propGetKey(const AVSMap* map, int index) = 0;
  virtual int __stdcall propNumElements(const AVSMap* map, const char* key) = 0;
  virtual char __stdcall propGetType(const AVSMap* map, const char* key) = 0;

  virtual int64_t __stdcall propGetInt(const AVSMap* map, const char* key, int index, int* error) = 0;
  virtual double __stdcall propGetFloat(const AVSMap* map, const char* key, int index, int* error) = 0;
  virtual const char* __stdcall propGetData(const AVSMap* map, const char* key, int index, int* error) = 0;
  virtual int __stdcall propGetDataSize(const AVSMap* map, const char* key, int index, int* error) = 0;
  virtual PClip __stdcall propGetClip(const AVSMap* map, const char* key, int index, int* error) = 0;
  virtual const PVideoFrame __stdcall propGetFrame(const AVSMap* map, const char* key, int index, int* error) = 0;

  virtual int __stdcall propDeleteKey(AVSMap* map, const char* key) = 0;

  virtual int __stdcall propSetInt(AVSMap* map, const char* key, int64_t i, int append) = 0;
  virtual int __stdcall propSetFloat(AVSMap* map, const char* key, double d, int append) = 0;
  virtual int __stdcall propSetData(AVSMap* map, const char* key, const char* d, int length, int append) = 0;
  virtual int __stdcall propSetClip(AVSMap* map, const char* key, PClip& clip, int append) = 0;
  virtual int __stdcall propSetFrame(AVSMap* map, const char* key, const PVideoFrame& frame, int append) = 0;

  virtual const int64_t* __stdcall propGetIntArray(const AVSMap* map, const char* key, int* error) = 0;
  virtual const double* __stdcall propGetFloatArray(const AVSMap* map, const char* key, int* error) = 0;
  virtual int __stdcall propSetIntArray(AVSMap* map, const char* key, const int64_t* i, int size) = 0;
  virtual int __stdcall propSetFloatArray(AVSMap* map, const char* key, const double* d, int size) = 0;

  virtual AVSMap* __stdcall createMap() = 0;
  virtual void __stdcall freeMap(AVSMap* map) = 0;
  virtual void __stdcall clearMap(AVSMap* map) = 0;

  // NewVideoFrame with frame property source.
  virtual PVideoFrame __stdcall NewVideoFrameP(const VideoInfo& vi, const PVideoFrame* prop_src, int align = FRAME_ALIGN) = 0;

  // Generic query to ask for various system properties
  virtual size_t  __stdcall GetEnvProperty(AvsEnvProperty prop) = 0;

  // Support functions
  virtual void* __stdcall Allocate(size_t nBytes, size_t alignment, AvsAllocType type) = 0;
  virtual void __stdcall Free(void* ptr) = 0;

  // these GetVar versions (renamed differently) were moved from IScriptEnvironment2

  // Returns TRUE and the requested variable. If the method fails, returns FALSE and does not touch 'val'.
  virtual bool  __stdcall GetVarTryPreV11C(const char* name, AVSValue* val) const = 0; // ex virtual bool  __stdcall GetVar(const char* name, AVSValue* val) const = 0;
  // Return the value of the requested variable.
  // If the variable was not found or had the wrong type,
  // return the supplied default value.
  virtual bool __stdcall GetVarBool(const char* name, bool def) const = 0;
  virtual int  __stdcall GetVarInt(const char* name, int def) const = 0;
  virtual double  __stdcall GetVarDouble(const char* name, double def) const = 0;
  virtual const char* __stdcall GetVarString(const char* name, const char* def) const = 0;
  // brand new in v8 - v11: real int64 support
  virtual int64_t __stdcall GetVarLong(const char* name, int64_t def) const = 0;

  // 'Invoke' functions moved here from internal ScriptEnvironments are renamed in order to keep vtable order
  // Invoke functions with 'Try' will return false instead of throwing NotFound().
  // ex-IScriptEnvironment2
  virtual bool __stdcall InvokeTry(AVSValue* result, const char* name, const AVSValue& args, const char* const* arg_names = 0) = 0;
  // Since V8
  virtual AVSValue __stdcall Invoke2(const AVSValue& implicit_last, const char* name, const AVSValue args, const char* const* arg_names = 0) = 0;
  // Ex-INeo
  virtual bool __stdcall Invoke2Try(AVSValue* result, const AVSValue& implicit_last, const char* name, const AVSValue args, const char* const* arg_names = 0) = 0;
  virtual AVSValue __stdcall Invoke3(const AVSValue& implicit_last, const PFunction& func, const AVSValue args, const char* const* arg_names = 0) = 0;
  virtual bool __stdcall Invoke3Try(AVSValue* result, const AVSValue& implicit_last, const PFunction& func, const AVSValue args, const char* const* arg_names = 0) = 0;

  // V9
  virtual bool __stdcall MakePropertyWritable(PVideoFrame* pvf) = 0;

  // V11
  virtual int __stdcall propGetIntSaturated(const AVSMap* map, const char* key, int index, int* error) = 0;
  virtual float __stdcall propGetFloatSaturated(const AVSMap* map, const char* key, int index, int* error) = 0;
  virtual int __stdcall propGetDataTypeHint(const AVSMap* map, const char* key, int index, int* error) = 0; // returns AVSPropDataTypeHint
  virtual int __stdcall propSetDataH(AVSMap* map, const char* key, const char* d, int length, int type, int append) = 0;

  // V12
  // New Global Lock API for cross-plugin synchronization.
  // Plugins must ensure these calls are balanced (acquire followed by release),
  virtual bool __stdcall AcquireGlobalLock(const char* name) = 0;
  virtual void __stdcall ReleaseGlobalLock(const char* name) = 0;
  virtual void __stdcall ApplyMessageEx(PVideoFrame* frame, const VideoInfo& vi, const char* message, int size,
    int textcolor, int halocolor, int bgcolor, bool utf8) = 0;
  virtual int64_t __stdcall GetCPUFlagsEx() = 0;
}; // end class IScriptEnvironment_AvsPreV11C. Order is important.

#endif // _AVS_COMPATENVIRONMENT_H_INCLUDED
