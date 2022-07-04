// Avisynth v2.5.  Copyright 2007 Ben Rudiak-Gould et al.
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


#include <avisynth.h>
#include "internal.h"

#ifdef AVS_WINDOWS // inspect for Linux; COM/VfW stuff not needed?
#ifdef AVS_WINDOWS
    #include <avs/win.h>
    #include <initguid.h>
    #include <vfw.h>
    #include <intrin.h>
#else
    #include <avs/posix.h>
    #include <x86intrin.h>
#endif

#include <avs/minmax.h>
#include "bitblt.h"
#include "exception.h"
#include <cstdio>
#include <new>
#include "AviHelper.h"


#include <float.h>

#ifdef MSVC
constexpr uint32_t FP_STATE = 0x9001f;
#ifdef _M_X64
constexpr uint32_t FP_MASK = 0xffffffff & !(_MCW_PC || _MCW_IC);
// x64 ignores _MCW_PC and _MCW_IC, Debug CRT library actually asserts when these are passed.
// https ://docs.microsoft.com/en-us/cpp/c-runtime-library/reference/control87-controlfp-control87-2?redirectedfrom=MSDN&view=msvc-170
#else
constexpr uint32_t FP_MASK = 0xffffffff;
#endif
#endif

#ifndef _DEBUG
// Release mode logging
// #define OPT_RELS_LOGGING
# ifdef OPT_RELS_LOGGING

#undef _RPT0
#undef _RPT1
#undef _RPT2
#undef _RPT3
#undef _RPT4
#  ifdef _RPT5
#undef _RPT5
#  endif

#define _RPT0(rptno, msg)                     ReportMe(msg)
#define _RPT1(rptno, msg, a1)                 ReportMe(msg, a1)
#define _RPT2(rptno, msg, a1, a2)             ReportMe(msg, a1, a2)
#define _RPT3(rptno, msg, a1, a2, a3)         ReportMe(msg, a1, a2, a3)
#define _RPT4(rptno, msg, a1, a2, a3, a4)     ReportMe(msg, a1, a2, a3, a4)
#define _RPT5(rptno, msg, a1, a2, a3, a4, a5) ReportMe(msg, a1, a2, a3, a4, a5)

void ReportMe(const char * msg, ...) {
  static char buf[256] = "";
  va_list args;
  int l = strlen(buf);

  va_start(args, msg);
  l = _vsnprintf(buf+l, sizeof(buf)-1-l, msg, args);
  buf[sizeof(buf)-1] = 0;
  va_end(args);

  if (l == -1 || strchr(buf, '\n')) {
    OutputDebugString(buf);
    buf[0] = 0;
  }
}

# else
#  ifndef _RPT5
#define _RPT5(rptno, msg, a1, a2, a3, a4, a5)
#  endif
# endif
#else
# ifndef _RPT5
#  ifdef _RPT_BASE
#define _RPT5(rptno, msg, a1, a2, a3, a4, a5) \
  _RPT_BASE((rptno, NULL, 0, NULL, msg, a1, a2, a3, a4, a5))
#  else
#   ifdef _CrtDbgBreak
#define _RPT5(rptno, msg, a1, a2, a3, a4, a5) \
  do { if ((1 == _CrtDbgReport(rptno, NULL, 0, NULL, msg, a1, a2, a3, a4, a5))) \
  _CrtDbgBreak(); } while (0)
#   else
#define _RPT5(rptno, msg, a1, a2, a3, a4, a5)
#   endif
#  endif
# endif
#endif

static long gRefCnt=0;

#ifdef XP_TLS
_TLS _tls;
#endif

extern "C" const GUID CLSID_CAVIFileSynth   // {E6D6B700-124D-11D4-86F3-DB80AFD98778}
= {0xe6d6b700, 0x124d, 0x11d4, {0x86, 0xf3, 0xdb, 0x80, 0xaf, 0xd9, 0x87, 0x78}};

extern "C" const GUID IID_IAvisynthClipInfo   // {E6D6B708-124D-11D4-86F3-DB80AFD98778}
= {0xe6d6b708, 0x124d, 0x11d4, {0x86, 0xf3, 0xdb, 0x80, 0xaf, 0xd9, 0x87, 0x78}};


struct IAvisynthClipInfo : IUnknown {
  virtual int __stdcall GetError(const char** ppszMessage) = 0;
  virtual bool __stdcall GetParity(int n) = 0;
  virtual bool __stdcall IsFieldBased() = 0;
};


class CAVIFileSynth: public IAVIFile, public IPersistFile, public IClassFactory, public IAvisynthClipInfo {
  friend class CAVIStreamSynth;
private:
  long m_refs;

  char* szScriptName;
  char* szScriptNameUTF8;
  IScriptEnvironment2* env;
  PClip filter_graph; // actual result of script evaluation
  const VideoInfo* vi;
  const char* error_msg;

  CRITICAL_SECTION cs_filter_graph;

  bool VDubPlanarHack;
  bool AVIPadScanlines;
  bool Enable_V210;
  bool Enable_b64a;
  bool Enable_Y3_10_10;
  bool Enable_Y3_10_16;
  bool Enable_PlanarToPackedRGB;


  int ImageSize(const VideoInfo *vi);

  bool DelayInit();
  bool DelayInit2();

  void MakeErrorStream(const char* msg);

  void Lock();
  void Unlock();

public:

  CAVIFileSynth(const CLSID& rclsid);
  ~CAVIFileSynth();

  static HRESULT Create(const CLSID& rclsid, const IID& riid, void **ppv);

  //////////// IUnknown

  STDMETHODIMP QueryInterface(const IID& iid, void** ppv);
  STDMETHODIMP_(ULONG) AddRef();
  STDMETHODIMP_(ULONG) Release();

  //////////// IClassFactory

  STDMETHODIMP CreateInstance (LPUNKNOWN pUnkOuter, REFIID riid,  void * * ppvObj) ;
  STDMETHODIMP LockServer (BOOL fLock) ;

  //////////// IPersistFile

  STDMETHODIMP GetClassID(LPCLSID lpClassID);  // IPersist

  STDMETHODIMP IsDirty();
  STDMETHODIMP Load(LPCOLESTR lpszFileName, DWORD grfMode);
  STDMETHODIMP Save(LPCOLESTR lpszFileName, BOOL fRemember);
  STDMETHODIMP SaveCompleted(LPCOLESTR lpszFileName);
  STDMETHODIMP GetCurFile(LPOLESTR *lplpszFileName);

  //////////// IAVIFile

  STDMETHODIMP CreateStream(PAVISTREAM *ppStream, AVISTREAMINFOW *psi);       // 5
  STDMETHODIMP EndRecord();                                                   // 8
  STDMETHODIMP GetStream(PAVISTREAM *ppStream, DWORD fccType, LONG lParam);   // 4
  STDMETHODIMP Info(AVIFILEINFOW *psi, LONG lSize);                           // 3

  STDMETHODIMP Open(LPCSTR szFile, UINT mode, LPCOLESTR lpszFileName);        // ???
  STDMETHODIMP Save(LPCSTR szFile, AVICOMPRESSOPTIONS FAR *lpOptions,         // ???
    AVISAVECALLBACK lpfnCallback);

  STDMETHODIMP ReadData(DWORD fcc, LPVOID lp, LONG *lpcb);                    // 7
  STDMETHODIMP WriteData(DWORD fcc, LPVOID lpBuffer, LONG cbBuffer);          // 6
  STDMETHODIMP DeleteStream(DWORD fccType, LONG lParam);                      // 9

                                                                              //////////// IAvisynthClipInfo

  int __stdcall GetError(const char** ppszMessage);
  bool __stdcall GetParity(int n);
  bool __stdcall IsFieldBased();
};

///////////////////////////////////

class CAVIStreamSynth;

class CAVIStreamSynth: public IAVIStream, public IAVIStreaming {
public:

  //////////// IUnknown

  STDMETHODIMP QueryInterface(const IID& iid, void **ppv);
  STDMETHODIMP_(ULONG) AddRef();
  STDMETHODIMP_(ULONG) Release();

  CAVIStreamSynth(CAVIFileSynth *parentPtr, bool isAudio);
  ~CAVIStreamSynth();

  //////////// IAVIStream

  STDMETHODIMP Create(LPARAM lParam1, LPARAM lParam2);
  STDMETHODIMP Delete(LONG lStart, LONG lSamples);
  STDMETHODIMP_(LONG) Info(AVISTREAMINFOW *psi, LONG lSize);
  STDMETHODIMP_(LONG) FindSample(LONG lPos, LONG lFlags);
  STDMETHODIMP Read(LONG lStart, LONG lSamples, LPVOID lpBuffer, LONG cbBuffer, LONG *plBytes, LONG *plSamples);
  STDMETHODIMP ReadData(DWORD fcc, LPVOID lp, LONG *lpcb);
  STDMETHODIMP ReadFormat(LONG lPos, LPVOID lpFormat, LONG *lpcbFormat);
  STDMETHODIMP SetFormat(LONG lPos, LPVOID lpFormat, LONG cbFormat);
  STDMETHODIMP Write(LONG lStart, LONG lSamples, LPVOID lpBuffer,
    LONG cbBuffer, DWORD dwFlags, LONG FAR *plSampWritten,
    LONG FAR *plBytesWritten);
  STDMETHODIMP WriteData(DWORD fcc, LPVOID lpBuffer, LONG cbBuffer);
  STDMETHODIMP SetInfo(AVISTREAMINFOW *psi, LONG lSize);

  //////////// IAVIStreaming

  STDMETHODIMP Begin(LONG lStart, LONG lEnd, LONG lRate);
  STDMETHODIMP End();

private:
  long m_refs;

  CAVIFileSynth *parent;
  BOOL fAudio;

  const char *sName;

  //////////// internal

  void ReadFrame(void* lpBuffer, int n);

  HRESULT Read2(LONG lStart, LONG lSamples, LPVOID lpBuffer, LONG cbBuffer, LONG *plBytes, LONG *plSamples);
};

#ifndef AVS_STATIC_LIB
BOOL APIENTRY DllMain(HANDLE hModule, ULONG ulReason, LPVOID lpReserved) {

  _RPT4(0,"DllMain: hModule=0x%08x, ulReason=%x, lpReserved=0x%08x, gRefCnt = %ld\n",
    hModule, ulReason, lpReserved, gRefCnt);

  return TRUE;
}

STDAPI DllGetClassObject(IN REFCLSID rclsid, IN REFIID riid, OUT LPVOID FAR* ppv){

  if (rclsid != CLSID_CAVIFileSynth) {
    _RPT0(0,"DllGetClassObject() CLASS_E_CLASSNOTAVAILABLE\n");
    return CLASS_E_CLASSNOTAVAILABLE;
  }
  _RPT0(0,"DllGetClassObject() CLSID: CAVIFileSynth\n");

  HRESULT hresult = CAVIFileSynth::Create(rclsid, riid, ppv);

  _RPT2(0,"DllGetClassObject() result=0x%X, object=%p\n", hresult, *ppv);

  return hresult;
}

STDAPI DllCanUnloadNow() {
  _RPT1(0,"DllCanUnloadNow(): gRefCnt = %ld\n", gRefCnt);

  return gRefCnt ? S_FALSE : S_OK;
}
#endif

///////////////////////////////////////////////////////////////////////////
//
//	CAVIFileSynth
//
///////////////////////////////////////////////////////////////////////////
//////////// IClassFactory

STDMETHODIMP CAVIFileSynth::CreateInstance (LPUNKNOWN pUnkOuter, REFIID riid,  void * * ppvObj) {

  if (pUnkOuter) {
    _RPT1(0,"%p->CAVIFileSynth::CreateInstance() CLASS_E_NOAGGREGATION\n", this);
    return CLASS_E_NOAGGREGATION;
  }
  _RPT1(0,"%p->CAVIFileSynth::CreateInstance()\n", this);

  HRESULT hresult = Create(CLSID_CAVIFileSynth, riid, ppvObj);

  _RPT3(0,"%p->CAVIFileSynth::CreateInstance() result=0x%X, object=%p\n", this, hresult, *ppvObj);

  return hresult;
}

STDMETHODIMP CAVIFileSynth::LockServer (BOOL fLock) {
  _RPT2(0,"%p->CAVIFileSynth::LockServer(%u)\n", this, fLock);
  return S_OK;
}

///////////////////////////////////////////////////
//////////// IPersistFile

STDMETHODIMP CAVIFileSynth::GetClassID(LPCLSID lpClassID) {  // IPersist
  _RPT1(0,"%p->CAVIFileSynth::GetClassID()\n", this);

  if (!lpClassID) return E_POINTER;

  *lpClassID = CLSID_CAVIFileSynth;

  return S_OK;
}

STDMETHODIMP CAVIFileSynth::IsDirty() {
  _RPT1(0,"%p->CAVIFileSynth::IsDirty()\n", this);
  return S_FALSE;
}

STDMETHODIMP CAVIFileSynth::Load(LPCOLESTR lpszFileName, DWORD grfMode) {
  char filename[MAX_PATH];

  WideCharToMultiByte(AreFileApisANSI() ? CP_ACP : CP_OEMCP, 0, lpszFileName, -1, filename, sizeof filename, NULL, NULL);

  _RPT3(0,"%p->CAVIFileSynth::Load(\"%s\", 0x%X)\n", this, filename, grfMode);

  return Open(filename, grfMode, lpszFileName);
}

STDMETHODIMP CAVIFileSynth::Save(LPCOLESTR lpszFileName, BOOL fRemember) {
  _RPT1(0,"%p->CAVIFileSynth::Save()\n", this);
  return E_FAIL;
}

STDMETHODIMP CAVIFileSynth::SaveCompleted(LPCOLESTR lpszFileName) {
  _RPT1(0,"%p->CAVIFileSynth::SaveCompleted()\n", this);
  return S_OK;
}

STDMETHODIMP CAVIFileSynth::GetCurFile(LPOLESTR *lplpszFileName) {
  _RPT1(0,"%p->CAVIFileSynth::GetCurFile()\n", this);

  if (lplpszFileName) *lplpszFileName = NULL;

  return E_FAIL;
}

///////////////////////////////////////////////////
/////// static local

HRESULT CAVIFileSynth::Create(const CLSID& rclsid, const IID& riid, void **ppv) {
  HRESULT hresult;

  //	_RPT0(0,"CAVIFileSynth::Create()\n");

  CAVIFileSynth* pAVIFileSynth = new(std::nothrow) CAVIFileSynth(rclsid);

  if (!pAVIFileSynth) return E_OUTOFMEMORY;

  hresult = pAVIFileSynth->QueryInterface(riid, ppv);
  pAVIFileSynth->Release();

  //	_RPT1(0,"CAVIFileSynth::Create() exit, result=0x%X\n", hresult);

  return hresult;
}

///////////////////////////////////////////////////
//////////// IUnknown

STDMETHODIMP CAVIFileSynth::QueryInterface(const IID& iid, void **ppv) {

  if (!ppv) {
    _RPT1(0,"%p->CAVIFileSynth::QueryInterface() E_POINTER\n", this);
    return E_POINTER;
  }

  _RPT1(0,"%p->CAVIFileSynth::QueryInterface() ", this);
  _RPT3(0,"{%08lx-%04x-%04x-", iid.Data1, iid.Data2, iid.Data3);
  _RPT4(0,"%02x%02x-%02x%02x", iid.Data4[0], iid.Data4[1], iid.Data4[2], iid.Data4[3]);
  _RPT4(0,"%02x%02x%02x%02x} (", iid.Data4[4], iid.Data4[5], iid.Data4[6], iid.Data4[7]);

  if (iid == IID_IUnknown) {
    *ppv = (IUnknown *)(IAVIFile *)this;
    _RPT0(0,"IUnknown)\n");
  } else if (iid == IID_IClassFactory) {
    *ppv = (IClassFactory *)this;
    _RPT0(0,"IClassFactory)\n");
  } else if (iid == IID_IPersist) {
    *ppv = (IPersist *)this;
    _RPT0(0,"IPersist)\n");
  } else if (iid == IID_IPersistFile) {
    *ppv = (IPersistFile *)this;
    _RPT0(0,"IPersistFile)\n");
  } else if (iid == IID_IAVIFile) {
    *ppv = (IAVIFile *)this;
    _RPT0(0,"IAVIFile)\n");
  } else if (iid == IID_IAvisynthClipInfo) {
    *ppv = (IAvisynthClipInfo *)this;
    _RPT0(0,"IAvisynthClipInfo)\n");
  } else {
    _RPT0(0,"unsupported!)\n");
    *ppv = NULL;
    return E_NOINTERFACE;
  }

  AddRef();

  return S_OK;
}

STDMETHODIMP_(ULONG) CAVIFileSynth::AddRef() {

  const int refs = InterlockedIncrement(&m_refs);
  InterlockedIncrement(&gRefCnt);

  _RPT3(0,"%p->CAVIFileSynth::AddRef() gRefCnt=%d, m_refs=%d\n", this, gRefCnt, refs);

  return refs;
}

STDMETHODIMP_(ULONG) CAVIFileSynth::Release() {

  const int refs = InterlockedDecrement(&m_refs);
  InterlockedDecrement(&gRefCnt);

  _RPT3(0,"%p->CAVIFileSynth::Release() gRefCnt=%d, m_refs=%d\n", this, gRefCnt, refs);

  if (!refs) delete this;
  return refs;
}

////////////////////////////////////////////////////////////////////////
//
//		CAVIStreamSynth
//
////////////////////////////////////////////////////////////////////////
//////////// IUnknown

STDMETHODIMP CAVIStreamSynth::QueryInterface(const IID& iid, void **ppv) {

  if (!ppv) {
    _RPT2(0,"%p->CAVIStreamSynth::QueryInterface() (%s) E_POINTER\n", this, sName);
    return E_POINTER;
  }

  _RPT2(0,"%p->CAVIStreamSynth::QueryInterface() (%s) ", this, sName);
  _RPT3(0,"{%08lx-%04x-%04x-", iid.Data1, iid.Data2, iid.Data3);
  _RPT4(0,"%02x%02x-%02x%02x", iid.Data4[0], iid.Data4[1], iid.Data4[2], iid.Data4[3]);
  _RPT4(0,"%02x%02x%02x%02x} (", iid.Data4[4], iid.Data4[5], iid.Data4[6], iid.Data4[7]);

  if (iid == IID_IUnknown) {
    *ppv = (IUnknown *)(IAVIStream *)this;
    _RPT0(0,"IUnknown)\n");
  } else if (iid == IID_IAVIStream) {
    *ppv = (IAVIStream *)this;
    _RPT0(0,"IAVIStream)\n");
  } else if (iid == IID_IAVIStreaming) {
    *ppv = (IAVIStreaming *)this;
    _RPT0(0,"IAVIStreaming)\n");
  } else {
    _RPT0(0,"unsupported!)\n");
    *ppv = NULL;
    return E_NOINTERFACE;
  }

  AddRef();

  return S_OK;
}

STDMETHODIMP_(ULONG) CAVIStreamSynth::AddRef() {

  const int refs = InterlockedIncrement(&m_refs);
  InterlockedIncrement(&gRefCnt);

  _RPT4(0,"%p->CAVIStreamSynth::AddRef() (%s) gRefCnt=%d, m_refs=%d\n", this, sName, gRefCnt, refs);

  return refs;
}

STDMETHODIMP_(ULONG) CAVIStreamSynth::Release() {

  const int refs = InterlockedDecrement(&m_refs);
  InterlockedDecrement(&gRefCnt);

  _RPT4(0,"%p->CAVIStreamSynth::Release() (%s) gRefCnt=%d, m_refs=%d\n", this, sName, gRefCnt, refs);

  if (!refs) delete this;
  return refs;
}

////////////////////////////////////////////////////////////////////////
//
//		CAVIFileSynth
//
////////////////////////////////////////////////////////////////////////
//////////// IAVIFile

STDMETHODIMP CAVIFileSynth::CreateStream(PAVISTREAM *ppStream, AVISTREAMINFOW *psi) {
  _RPT1(0,"%p->CAVIFileSynth::CreateStream()\n", this);
  *ppStream = NULL;
  return S_OK;//AVIERR_READONLY;
}

STDMETHODIMP CAVIFileSynth::EndRecord() {
  _RPT1(0,"%p->CAVIFileSynth::EndRecord()\n", this);
  return AVIERR_READONLY;
}

STDMETHODIMP CAVIFileSynth::Save(LPCSTR szFile, AVICOMPRESSOPTIONS FAR *lpOptions,
  AVISAVECALLBACK lpfnCallback) {
  _RPT1(0,"%p->CAVIFileSynth::Save()\n", this);
  return AVIERR_READONLY;
}

STDMETHODIMP CAVIFileSynth::ReadData(DWORD fcc, LPVOID lp, LONG *lpcb) {
  _RPT1(0,"%p->CAVIFileSynth::ReadData()\n", this);
  return AVIERR_NODATA;
}

STDMETHODIMP CAVIFileSynth::WriteData(DWORD fcc, LPVOID lpBuffer, LONG cbBuffer) {
  _RPT1(0,"%p->CAVIFileSynth::WriteData()\n", this);
  return AVIERR_READONLY;
}

STDMETHODIMP CAVIFileSynth::DeleteStream(DWORD fccType, LONG lParam) {
  _RPT1(0,"%p->CAVIFileSynth::DeleteStream()\n", this);
  return AVIERR_READONLY;
}


///////////////////////////////////////////////////
/////// local

CAVIFileSynth::CAVIFileSynth(const CLSID& rclsid) {
  _RPT1(0,"%p->CAVIFileSynth::CAVIFileSynth()\n", this);

  m_refs = 0;
  AddRef();

  szScriptName = NULL;
  env = NULL;
  error_msg = NULL;

  VDubPlanarHack = false;
  AVIPadScanlines = false;
  Enable_V210 = false;
  Enable_b64a = false;
  Enable_Y3_10_10 = false;
  Enable_Y3_10_16 = false;
  Enable_PlanarToPackedRGB = false;

  InitializeCriticalSection(&cs_filter_graph);
}

CAVIFileSynth::~CAVIFileSynth() {
  _RPT2(0,"%p->CAVIFileSynth::~CAVIFileSynth(), gRefCnt = %d\n", this, gRefCnt);

  Lock();

  delete[] szScriptName;

  filter_graph = 0;

  if (env) env->DeleteScriptEnvironment();
  env = NULL;

  DeleteCriticalSection(&cs_filter_graph);
}


STDMETHODIMP CAVIFileSynth::Open(LPCSTR szFile, UINT mode, LPCOLESTR lpszFileName) {

  //	_RPT3(0,"%p->CAVIFileSynth::Open(\"%s\", 0x%08lx)\n", this, szFile, mode);

  if (mode & (OF_CREATE|OF_WRITE))
    return E_FAIL;

  if (env) env->DeleteScriptEnvironment();   // just in case
  env = NULL;
  filter_graph = 0;
  vi = NULL;

  szScriptName = new(std::nothrow) char[lstrlen(szFile)+1];
  if (!szScriptName)
    return AVIERR_MEMORY;
  lstrcpy(szScriptName, szFile);

  // when unicode file names are given and cannot be converted to 8 bit, szFile has ??? chars
  // szFile: 0x0018f198 "C:\\unicode\\SNH48 - ??????.avs"
  // lpSzFileName 0x02631cc8 L"C:\\unicode\\SNH48 - unicode_chars_here.avs"
  // use utf8 in DelayInit2 instead of szScriptName
  std::wstring ScriptNameW = lpszFileName;

  // wide -> utf8
  int utf8len = WideCharToMultiByte(CP_UTF8, 0, ScriptNameW.c_str(), -1/*null terminated src*/, NULL, 0/*returns the required buffer size*/, 0, 0) + 1; // with \0 terminator
  szScriptNameUTF8 = new(std::nothrow) char[utf8len];
  if (!szScriptNameUTF8)
    return AVIERR_MEMORY;

  WideCharToMultiByte(CP_UTF8, 0, ScriptNameW.c_str(), -1, szScriptNameUTF8, utf8len, 0, 0);
  // conversion to c_str once, here, not in DelayInit2

  return S_OK;
}

bool CAVIFileSynth::DelayInit() {

  Lock();

  bool result = DelayInit2();

  Unlock();

  return result;
}

bool CAVIFileSynth::DelayInit2() {
  // _RPT1(0,"Original: 0x%.4x\n", _control87( 0, 0 ) );
#if defined(MSVC)
  unsigned int fp_state = _controlfp(0, 0);
  _controlfp(FP_STATE, FP_MASK);
#endif
  if (szScriptNameUTF8) // unicode!
  {
#ifndef _DEBUG
    try {
#endif
      try {
        // create a script environment and load the script into it
        env = CreateScriptEnvironment2();
        if (!env) return false;
      }
      catch (const AvisynthError &error) {
        error_msg = error.msg;
        return false;
      }
      try {
        AVSValue new_args[2] = { szScriptNameUTF8, true };
        AVSValue return_val = env->Invoke("Import", AVSValue(new_args, 2));

        // store the script's return value (a video clip)
        if (return_val.IsClip()) {
          filter_graph = return_val.AsClip();

          // Allow WAVE_FORMAT_IEEE_FLOAT audio output
          const bool AllowFloatAudio = env->GetVarBool(VARNAME_AllowFloatAudio, false);

          if (!AllowFloatAudio && filter_graph->GetVideoInfo().IsSampleType(SAMPLE_FLOAT)) // Ensure samples are int
            filter_graph = env->Invoke("ConvertAudioTo16bit", AVSValue(&return_val, 1)).AsClip();

          filter_graph = env->Invoke("Cache", AVSValue(filter_graph)).AsClip();

          // Tune top-level cache for encoding
          filter_graph->SetCacheHints(CACHE_SET_MIN_CAPACITY, 0);
          filter_graph->SetCacheHints(CACHE_SET_MAX_CAPACITY, 0);

          // Alternate sample values for editors:
          //filter_graph->SetCacheHints(CACHE_SET_MIN_CAPACITY, 10);
          //filter_graph->SetCacheHints(CACHE_SET_MAX_CAPACITY, 100);
        }
        else if (return_val.IsBool())
          env->ThrowError("The script's return value was not a video clip, (Is a bool, %s).", return_val.AsBool() ? "True" : "False");
        else if (return_val.IsInt())
          env->ThrowError("The script's return value was not a video clip, (Is an int, %d).", return_val.AsInt());
        else if (return_val.IsFloat())
          env->ThrowError("The script's return value was not a video clip, (Is a float, %f).", return_val.AsFloat());
        else if (return_val.IsString())
          env->ThrowError("The script's return value was not a video clip, (Is a string, %s).", return_val.AsString());
        else if (return_val.IsArray())
          env->ThrowError("The script's return value was not a video clip, (Is an array[%d]).", return_val.ArraySize());
        else if (!return_val.Defined())
          env->ThrowError("The script's return value was not a video clip, (Is the undefined value).");
        else
          env->ThrowError("The script's return value was not a video clip, (The type is unknown).");

        if (!filter_graph)
          env->ThrowError("The returned video clip was nil (this is a bug)");

        // get information about the clip
        vi = &filter_graph->GetVideoInfo();

        // Hack YV16 and YV24 chroma plane order for old VDub's
        VDubPlanarHack = env->GetVarBool(VARNAME_VDubPlanarHack, false);

        // Option to have scanlines mod4 padded in all pixel formats
        AVIPadScanlines = env->GetVarBool(VARNAME_AVIPadScanlines, false);

        // AVS+ Enable_V210 instead of P210
        Enable_V210 = env->GetVarBool(VARNAME_Enable_V210, false);
        // AVS+ y3[10][10] instead of P210
        Enable_Y3_10_10 = env->GetVarBool(VARNAME_Enable_Y3_10_10, false);
        // AVS+ y3[10][16] instead of P216
        Enable_Y3_10_16 = env->GetVarBool(VARNAME_Enable_Y3_10_16, false);
        // AVS+ Enable_V210 instead of BRA[64]
        Enable_b64a = env->GetVarBool(VARNAME_Enable_b64a, false);
        // AVS+ Enable on-the-fly Planar RGB to Packed RGB conversion
        Enable_PlanarToPackedRGB = env->GetVarBool(VARNAME_Enable_PlanarToPackedRGB, false);
      }
      catch (const AvisynthError &error) {
        error_msg = error.msg;
        try {
          AVSValue args[2] = { error.msg, 0xff3333 };
          static const char* const arg_names[2] = { 0, "text_color" };
          filter_graph = env->Invoke("MessageClip", AVSValue(args, 2), arg_names).AsClip();
          vi = &filter_graph->GetVideoInfo();
        }
        catch (const AvisynthError&) {
          filter_graph = 0;
        }
      }

      delete[] szScriptName;
      szScriptName = NULL;
      delete[] szScriptNameUTF8;
      szScriptNameUTF8 = NULL;
#ifdef X86_32
      _mm_empty();
#endif
#if defined(MSVC)
      _clearfp();
      _controlfp( fp_state, FP_MASK);
#endif
      return true;
#ifndef _DEBUG
    }
    catch (...) {
      _RPT0(1,"DelayInit() caught general exception!\n");
#if defined(MSVC)
      _clearfp();
#endif
#ifdef X86_32
      _mm_empty();
#if defined(MSVC)
      _controlfp( fp_state, FP_MASK );
#endif
#endif
      return false;
    }
#endif
  } else {
#ifdef X86_32
    _mm_empty();
#endif
#if defined(MSVC)
    _clearfp();
    _controlfp( fp_state, FP_MASK);
#endif
    return (env && filter_graph && vi);
  }
}


void CAVIFileSynth::MakeErrorStream(const char* msg) {
  error_msg = msg;
  filter_graph = Create_MessageClip(msg, vi->width, vi->height, vi->pixel_type, false, 0xFF3333, 0, 0, -1, -1, -1, env);
}

void CAVIFileSynth::Lock() {

  EnterCriticalSection(&cs_filter_graph);

}

void CAVIFileSynth::Unlock() {

  LeaveCriticalSection(&cs_filter_graph);

}

///////////////////////////////////////////////////
//////////// IAVIFile

STDMETHODIMP CAVIFileSynth::Info(AVIFILEINFOW *pfi, LONG lSize) {

  _RPT2(0,"%p->CAVIFileSynth::Info(pfi, %d)\n", this, lSize);

  if (!pfi) return E_POINTER;

  if (!DelayInit()) return E_FAIL;

  AVIFILEINFOW afi;

  memset(&afi, 0, sizeof(afi));

  afi.dwMaxBytesPerSec	= 0;
  afi.dwFlags				= AVIFILEINFO_HASINDEX | AVIFILEINFO_ISINTERLEAVED;
  afi.dwCaps				= AVIFILECAPS_CANREAD | AVIFILECAPS_ALLKEYFRAMES | AVIFILECAPS_NOCOMPRESSION;

  int nrStreams=0;
  if (vi->HasVideo()==true)	nrStreams=1;
  if (vi->HasAudio()==true)	nrStreams++;

  afi.dwStreams				= nrStreams;
  afi.dwSuggestedBufferSize	= 0;
  afi.dwWidth					= vi->width;
  afi.dwHeight				= vi->height;
  afi.dwEditCount				= 0;

  afi.dwRate					= vi->fps_numerator;
  afi.dwScale					= vi->fps_denominator;
  afi.dwLength				= vi->num_frames;

  wcscpy(afi.szFileType, L"Avisynth");

  // Maybe should return AVIERR_BUFFERTOOSMALL for lSize < sizeof(afi)
  memset(pfi, 0, lSize);
  memcpy(pfi, &afi, min(size_t(lSize), sizeof(afi)));
  return S_OK;
}

static inline char BePrintable(int ch) {
  ch &= 0xff;
  return (char)(isprint(ch) ? ch : '.');
}


STDMETHODIMP CAVIFileSynth::GetStream(PAVISTREAM *ppStream, DWORD fccType, LONG lParam) {
  CAVIStreamSynth *casr;
  char fcc[5];

  fcc[0] = BePrintable(fccType      );
  fcc[1] = BePrintable(fccType >>  8);
  fcc[2] = BePrintable(fccType >> 16);
  fcc[3] = BePrintable(fccType >> 24);
  fcc[4] = 0;

  _RPT4(0,"%p->CAVIFileSynth::GetStream(*, %08x(%s), %ld)\n", this, fccType, fcc, lParam);

  if (!DelayInit()) return E_FAIL;

  *ppStream = NULL;

  if (!fccType)
  {
    // Maybe an Option to set the order of stream discovery
    if ((lParam==0) && (vi->HasVideo()) )
      fccType = streamtypeVIDEO;
    else
      if ( ((lParam==1) && (vi->HasVideo())) ||  ((lParam==0) && vi->HasAudio()) )
      {
        lParam=0;
        fccType = streamtypeAUDIO;
      }
  }

  if (lParam > 0) return AVIERR_NODATA;

  if (fccType == streamtypeVIDEO) {
    if (!vi->HasVideo())
      return AVIERR_NODATA;

    if ((casr = new(std::nothrow) CAVIStreamSynth(this, false)) == 0)
      return AVIERR_MEMORY;

    *ppStream = (IAVIStream *)casr;

  } else if (fccType == streamtypeAUDIO) {
    if (!vi->HasAudio())
      return AVIERR_NODATA;

    if ((casr = new(std::nothrow) CAVIStreamSynth(this, true)) == 0)
      return AVIERR_MEMORY;

    *ppStream = (IAVIStream *)casr;
  } else
    return AVIERR_NODATA;

  return S_OK;
}


////////////////////////////////////////////////////////////////////////
//////////// IAvisynthClipInfo

int __stdcall CAVIFileSynth::GetError(const char** ppszMessage) {
  if (!DelayInit() && !error_msg)
    error_msg = "Avisynth: script open failed!";

  if (ppszMessage)
    *ppszMessage = error_msg;
  return !!error_msg;
}

bool __stdcall CAVIFileSynth::GetParity(int n) {
  if (!DelayInit())
    return false;
  return filter_graph->GetParity(n);
}

bool __stdcall CAVIFileSynth::IsFieldBased() {
  if (!DelayInit())
    return false;
  return vi->IsFieldBased();
}


////////////////////////////////////////////////////////////////////////
//
//		CAVIStreamSynth
//
////////////////////////////////////////////////////////////////////////
//////////// IAVIStreaming

STDMETHODIMP CAVIStreamSynth::Begin(LONG lStart, LONG lEnd, LONG lRate) {
  _RPT5(0,"%p->CAVIStreamSynth::Begin(%ld, %ld, %ld) (%s)\n", this, lStart, lEnd, lRate, sName);
  return S_OK;
}

STDMETHODIMP CAVIStreamSynth::End() {
  _RPT2(0,"%p->CAVIStreamSynth::End() (%s)\n", this, sName);
  return S_OK;
}

//////////// IAVIStream

STDMETHODIMP CAVIStreamSynth::Create(LPARAM lParam1, LPARAM lParam2) {
  _RPT1(0,"%p->CAVIStreamSynth::Create()\n", this);
  return AVIERR_READONLY;
}

STDMETHODIMP CAVIStreamSynth::Delete(LONG lStart, LONG lSamples) {
  _RPT1(0,"%p->CAVIStreamSynth::Delete()\n", this);
  return AVIERR_READONLY;
}

STDMETHODIMP CAVIStreamSynth::ReadData(DWORD fcc, LPVOID lp, LONG *lpcb) {
  _RPT1(0,"%p->CAVIStreamSynth::ReadData()\n", this);
  return AVIERR_NODATA;
}

STDMETHODIMP CAVIStreamSynth::SetFormat(LONG lPos, LPVOID lpFormat, LONG cbFormat) {
  _RPT1(0,"%p->CAVIStreamSynth::SetFormat()\n", this);
  return AVIERR_READONLY;
}

STDMETHODIMP CAVIStreamSynth::WriteData(DWORD fcc, LPVOID lpBuffer, LONG cbBuffer) {
  _RPT1(0,"%p->CAVIStreamSynth::WriteData()\n", this);
  return AVIERR_READONLY;
}

STDMETHODIMP CAVIStreamSynth::SetInfo(AVISTREAMINFOW *psi, LONG lSize) {
  _RPT1(0,"%p->CAVIStreamSynth::SetInfo()\n", this);
  return AVIERR_READONLY;
}

////////////////////////////////////////////////////////////////////////
//////////// local

CAVIStreamSynth::CAVIStreamSynth(CAVIFileSynth *parentPtr, bool isAudio) {

  sName = isAudio ? "audio" : "video";

  _RPT2(0,"%p->CAVIStreamSynth(%s)\n", this, sName);

  m_refs = 0; AddRef();

  parent			= parentPtr;
  fAudio			= isAudio;

  parent->AddRef();
}

CAVIStreamSynth::~CAVIStreamSynth() {
  _RPT3(0,"%p->~CAVIStreamSynth() (%s), gRefCnt = %d\n", this, sName, gRefCnt);

  if (parent)
    parent->Release();
}

////////////////////////////////////////////////////////////////////////
//////////// IAVIStream

STDMETHODIMP_(LONG) CAVIStreamSynth::Info(AVISTREAMINFOW *psi, LONG lSize) {
  _RPT4(0,"%p->CAVIStreamSynth::Info(%p, %ld) (%s)\n", this, psi, lSize, sName);

  if (!psi) return E_POINTER;

  AVISTREAMINFOW asi;

  const VideoInfo* const vi = parent->vi;

  memset(&asi, 0, sizeof(asi));
  asi.fccType = fAudio ? streamtypeAUDIO : streamtypeVIDEO;
  asi.dwQuality = DWORD(-1);
  if (fAudio) {
    asi.fccHandler = 0;
    int bytes_per_sample = vi->BytesPerAudioSample();
    asi.dwScale = bytes_per_sample;
    asi.dwRate = vi->audio_samples_per_second * bytes_per_sample;
    asi.dwLength = (unsigned int)vi->num_audio_samples;
    asi.dwSampleSize = bytes_per_sample;
    wcscpy(asi.szName, L"Avisynth audio #1");
  } else {
    bool targetIsConvertedToPackedRGB = (parent->Enable_PlanarToPackedRGB && (vi->IsPlanarRGB() || vi->IsPlanarRGBA()));

    VideoInfo vi_final = *vi;
    // if basic type is changed, that affects buffer size, we change the format here
    if (targetIsConvertedToPackedRGB) {
      // Enable_PlanarToPackedRGB results in packed RGB64 for bits>8, RGB24/32 for 8 bits
      if (vi->BitsPerComponent() == 8) {
        if (vi->IsPlanarRGB())
          vi_final.pixel_type = VideoInfo::CS_BGR24;
        else // planar RGBA
          vi_final.pixel_type = VideoInfo::CS_BGR32;
      }
      else // all 8+ bit planar RGB(A) is converted to RGB64
        vi_final.pixel_type = VideoInfo::CS_BGR64;
    }
    // silent mapping of 12/14 bit YUV formats to 16 bit
    if (vi->pixel_type == VideoInfo::CS_YUV420P12 || vi->pixel_type == VideoInfo::CS_YUV420P14 || vi->pixel_type == VideoInfo::CS_YUV420PS)
      vi_final.pixel_type = VideoInfo::CS_YUV420P16;
    else if (vi->pixel_type == VideoInfo::CS_YUV422P12 || vi->pixel_type == VideoInfo::CS_YUV422P14 || vi->pixel_type == VideoInfo::CS_YUV422PS)
      vi_final.pixel_type = VideoInfo::CS_YUV422P16;
    else if (vi->pixel_type == VideoInfo::CS_YUV444P10 || vi->pixel_type == VideoInfo::CS_YUV444P12 || vi->pixel_type == VideoInfo::CS_YUV444P14 || vi->pixel_type == VideoInfo::CS_YUV444PS)
      vi_final.pixel_type = VideoInfo::CS_YUV444P16;
    else if (vi->pixel_type == VideoInfo::CS_YUVA444P10 || vi->pixel_type == VideoInfo::CS_YUVA444P12 || vi->pixel_type == VideoInfo::CS_YUVA444P14 || vi->pixel_type == VideoInfo::CS_YUVA444PS)
      vi_final.pixel_type = VideoInfo::CS_YUVA444P16;
    // -- pixel_type change end

    const int image_size = parent->ImageSize(&vi_final);
    asi.fccHandler = MAKEFOURCC('N','K','N','U'); // 'UNKN';

    if (vi_final.IsRGB() && !vi_final.IsPlanar() && vi_final.BitsPerComponent() == 8)
      asi.fccHandler = MAKEFOURCC('D','I','B',' ');
    else if (vi_final.IsYUY2())
      asi.fccHandler = MAKEFOURCC('Y','U','Y','2');
    else if (vi_final.IsYV12())
      asi.fccHandler = MAKEFOURCC('Y','V','1','2');
    else if (vi_final.IsY8())
      asi.fccHandler = MAKEFOURCC('Y','8','0','0');
    else if (vi_final.IsYV24())
      asi.fccHandler = MAKEFOURCC('Y','V','2','4');
    else if (vi_final.IsYV16())
      asi.fccHandler = MAKEFOURCC('Y','V','1','6');
    else if (vi_final.IsYV411())
      asi.fccHandler = MAKEFOURCC('Y','4','1','B');
    // avs+
    else if (vi_final.IsRGB64() && parent->Enable_b64a)
      asi.fccHandler = MAKEFOURCC('b','6','4','a'); // b64a = packed rgba 4*16-bit
    else if (vi_final.IsRGB64())
      asi.fccHandler = MAKEFOURCC('B','R','A',64); // BRA@ ie. BRA[64]
    else if (vi_final.IsRGB48())
      asi.fccHandler = MAKEFOURCC('B','G','R',48); // BGR0 ie. BGR[48]
    else if (vi_final.pixel_type == VideoInfo::CS_YUV420P10)
      asi.fccHandler = MAKEFOURCC('P','0','1','0');
    else if (vi_final.pixel_type == VideoInfo::CS_YUV420P16)
      asi.fccHandler = MAKEFOURCC('P','0','1','6');
    else if (vi_final.pixel_type == VideoInfo::CS_YUV422P10 && parent->Enable_V210)
      asi.fccHandler = MAKEFOURCC('v','2','1','0');
    else if (vi_final.pixel_type == VideoInfo::CS_YUV422P10 && parent->Enable_Y3_10_10)
      asi.fccHandler = MAKEFOURCC('Y', '3', 10, 10); // Y3[10][10] (AV_PIX_FMT_YUV422P10) = planar YUV 422*10-bit
    else if (vi_final.pixel_type == VideoInfo::CS_YUV422P10)
      asi.fccHandler = MAKEFOURCC('P','2','1','0');
    else if (vi_final.pixel_type == VideoInfo::CS_YUV422P16 && parent->Enable_Y3_10_16)
      asi.fccHandler = MAKEFOURCC('Y', '3', 10, 16); // Y3[10][16] (AV_PIX_FMT_YUV422P16) = planar YUV 422*16-bit
    else if (vi_final.pixel_type == VideoInfo::CS_YUV422P16)
      asi.fccHandler = MAKEFOURCC('P','2','1','6');
    else if (vi_final.pixel_type == VideoInfo::CS_YUV444P16 || vi_final.pixel_type == VideoInfo::CS_YUVA444P16)
      asi.fccHandler = MAKEFOURCC('Y','4','1','6');
    else if (vi_final.pixel_type == VideoInfo::CS_RGBP)
      asi.fccHandler = MAKEFOURCC('G','3',0, 8); // similar to 10-16 bits G3[0][8]
    // asi.fccHandler = MAKEFOURCC('8','B','P','S'); // this would be a special RLE encoded format
    // MagicYUV implements these (planar rgb/rgba 10,12,14,16) G3[0][10], G4[0][10], G3[0][12], G4[0][12], G3[0][14], G4[0][14], G3[0][16], G4[0][16]
    else if (vi_final.pixel_type == VideoInfo::CS_RGBP10)
      asi.fccHandler = MAKEFOURCC('G','3',0,10);
    else if (vi_final.pixel_type == VideoInfo::CS_RGBP12)
      asi.fccHandler = MAKEFOURCC('G','3',0,12);
    else if (vi_final.pixel_type == VideoInfo::CS_RGBP14)
      asi.fccHandler = MAKEFOURCC('G','3',0,14);
    else if (vi_final.pixel_type == VideoInfo::CS_RGBP16)
      asi.fccHandler = MAKEFOURCC('G','3',0,16);
    else if (vi_final.pixel_type == VideoInfo::CS_RGBAP)
      asi.fccHandler = MAKEFOURCC('G','4',0, 8); // similar to 10-16 bits G4[0][10]
    else if (vi_final.pixel_type == VideoInfo::CS_RGBAP10)
      asi.fccHandler = MAKEFOURCC('G','4',0,10);
    else if (vi_final.pixel_type == VideoInfo::CS_RGBAP12)
      asi.fccHandler = MAKEFOURCC('G','4',0,12);
    else if (vi_final.pixel_type == VideoInfo::CS_RGBAP14)
      asi.fccHandler = MAKEFOURCC('G','4',0,14);
    else if (vi_final.pixel_type == VideoInfo::CS_RGBAP16)
      asi.fccHandler = MAKEFOURCC('G','4',0,16);
    else {
      // fixme: YUVA420P8, grey 10+ bits such as Y16 are not covered
      _ASSERT(FALSE);
    }

    asi.dwScale = vi_final.fps_denominator;
    asi.dwRate = vi_final.fps_numerator;
    asi.dwLength = vi_final.num_frames;
    asi.rcFrame.right = vi_final.width;
    asi.rcFrame.bottom = vi_final.height;
    asi.dwSampleSize = image_size;
    asi.dwSuggestedBufferSize = image_size;
    wcscpy(asi.szName, L"Avisynth video #1");
  }

  // Maybe should return AVIERR_BUFFERTOOSMALL for lSize < sizeof(asi)
  memset(psi, 0, lSize);
  memcpy(psi, &asi, min(size_t(lSize), sizeof(asi)));
  return S_OK;
}

STDMETHODIMP_(LONG) CAVIStreamSynth::FindSample(LONG lPos, LONG lFlags) {
  //	_RPT3(0,"%p->CAVIStreamSynth::FindSample(%ld, %08lx)\n", this, lPos, lFlags);

  if (lFlags & FIND_FORMAT)
    return -1;

  if (lFlags & FIND_FROM_START)
    return 0;

  return lPos;
}


////////////////////////////////////////////////////////////////////////
//////////// local

int CAVIFileSynth::ImageSize(const VideoInfo *vi) {
  return AviHelper_ImageSize(vi, AVIPadScanlines, Enable_V210, false, false, false, false, false, false);
}

void CAVIStreamSynth::ReadFrame(void* lpBuffer, int n) {
//  _RPT1(0, "CAVIStreamSynth::ReadFrame %d\r\n", n);
  VideoInfo vi = parent->filter_graph->GetVideoInfo();
  PVideoFrame frame;

  if (((vi.Is420() || vi.Is422()) && (vi.BitsPerComponent() == 12 || vi.BitsPerComponent() == 14 || vi.BitsPerComponent() == 32)) ||
    (vi.Is444() && (vi.BitsPerComponent() > 8 && vi.BitsPerComponent() != 16)))
  {
    // silent mapping of 12/14bit/float YUV420/422 formats to 16 bits
    AVSValue new_args[2] = { parent->filter_graph, 16 };
    PClip newClip = parent->env->Invoke("ConvertBits", AVSValue(new_args, 2)).AsClip();
    frame = newClip->GetFrame(n, parent->env);
    vi = newClip->GetVideoInfo();
  }
  else if (parent->Enable_PlanarToPackedRGB && (vi.IsPlanarRGB() || vi.IsPlanarRGBA())) {
    PClip newClip;
    // convert Planar RGB to RGB24/32/RGB64
    if (vi.BitsPerComponent() == 8) // 8 bit: ConvertToRGB24/32
    {
      if (vi.IsPlanarRGB()) {
        AVSValue new_args[1] = { parent->filter_graph };
        newClip = parent->env->Invoke("ConvertToRGB24", AVSValue(new_args, 1)).AsClip();
      }
      else { // IsPlanarRGBA()
        AVSValue new_args[1] = { parent->filter_graph };
        newClip = parent->env->Invoke("ConvertToRGB32", AVSValue(new_args, 1)).AsClip();
      }
    }
    else {
      // 8+ bits, always RGB64
      newClip = parent->filter_graph;
      if (vi.BitsPerComponent() != 16) {
        AVSValue new_args[2] = { newClip, 16 };
        newClip = parent->env->Invoke("ConvertBits", AVSValue(new_args, 2)).AsClip();
      }
      AVSValue new_args[1] = { newClip };
      newClip = parent->env->Invoke("ConvertToRGB64", AVSValue(new_args, 1)).AsClip();
    }
    frame = newClip->GetFrame(n, parent->env);
    vi = newClip->GetVideoInfo();
  }
  else {
    // no on-the-fly conversion
    frame = parent->filter_graph->GetFrame(n, parent->env);
  }
  if (!frame)
    parent->env->ThrowError("Avisynth error: generated video frame was nil (this is a bug)");

  // Y416 packed U,Y,V,A
  if (vi.pixel_type == VideoInfo::CS_YUV444P16 || vi.pixel_type == VideoInfo::CS_YUVA444P16) {
    int width = vi.width;
    int height = vi.height;
    int ppitch_y = frame->GetPitch(PLANAR_Y);
    int ppitch_uv = frame->GetPitch(PLANAR_U);
    int ppitch_a = frame->GetPitch(PLANAR_A);
    bool hasAlpha = (vi.NumComponents() == 4);
    const uint8_t *yptr = frame->GetReadPtr(PLANAR_Y);
    const uint8_t *uptr = frame->GetReadPtr(PLANAR_U);
    const uint8_t *vptr = frame->GetReadPtr(PLANAR_V);
    const uint8_t *aptr = frame->GetReadPtr(PLANAR_A);
    uint8_t *outbuf = (uint8_t *)lpBuffer;
    int out_pitch = width * 4 * sizeof(uint16_t);
#ifdef INTEL_INTRINSICS
    if ((parent->env->GetCPUFlags() & CPUF_SSE2) != 0) {
      if (hasAlpha)
        ToY416_sse2<true>(outbuf, out_pitch, yptr, ppitch_y, uptr, vptr, ppitch_uv, aptr, ppitch_a, width, height);
      else
        ToY416_sse2<false>(outbuf, out_pitch, yptr, ppitch_y, uptr, vptr, ppitch_uv, aptr, ppitch_a, width, height);
    }
    else
#endif
    {
      if (hasAlpha)
        ToY416_c<true>(outbuf, out_pitch, yptr, ppitch_y, uptr, vptr, ppitch_uv, aptr, ppitch_a, width, height);
      else
        ToY416_c<false>(outbuf, out_pitch, yptr, ppitch_y, uptr, vptr, ppitch_uv, aptr, ppitch_a, width, height);
    }
    return;
  }

  const int pitch    = frame->GetPitch();
  const int row_size = frame->GetRowSize();
  const int height   = frame->GetHeight();

  int out_pitch;
  int out_pitchUV;

  // BMP scanlines are dword-aligned
  if ((vi.IsRGB() && !vi.IsPlanar()) || vi.IsYUY2() || vi.IsY() || parent->AVIPadScanlines) {
    out_pitch = (row_size+3) & ~3;
    out_pitchUV = (frame->GetRowSize(PLANAR_U)+3) & ~3; // 0 for packed RGB
  }
  // Planar scanlines are packed
  else {
    out_pitch = row_size;
    if(vi.IsRGB())
      out_pitchUV = frame->GetRowSize(PLANAR_B); // G=B=R
    else
      out_pitchUV = frame->GetRowSize(PLANAR_U);
  }

  int plane1;
  int plane2;

  // Old VDub wants YUV for YV24 and YV16 and YVU for YV12.
  if (parent->VDubPlanarHack && !vi.IsYV12() && !vi.IsRGB() && vi.BitsPerComponent() == 8) {
    plane1 = PLANAR_U;
    plane2 = PLANAR_V;
  }
  else {
    if (vi.IsRGB() && vi.IsPlanar())
    {
      // (PLANAR_G)
      plane1 = PLANAR_B;
      plane2 = PLANAR_R;
    }
    else {
      if (vi.BitsPerComponent() == 8) {
      // Set default VFW output plane order.
        plane1 = PLANAR_V;
        plane2 = PLANAR_U;
      }
      else {
        plane1 = PLANAR_U;
        plane2 = PLANAR_V;
      }
    }
  }

  bool semi_packed_p10 = (vi.pixel_type == VideoInfo::CS_YUV420P10) || (vi.pixel_type == VideoInfo::CS_YUV422P10) ;
  bool semi_packed_p16 = (vi.pixel_type == VideoInfo::CS_YUV420P16) || (vi.pixel_type == VideoInfo::CS_YUV422P16) ;

#ifdef INTEL_INTRINSICS
  const bool ssse3 = (parent->env->GetCPUFlags() & CPUF_SSSE3) != 0;
  const bool sse2 = (parent->env->GetCPUFlags() & CPUF_SSE2) != 0;
#endif

  if ((vi.pixel_type == VideoInfo::CS_YUV420P10) || (vi.pixel_type == VideoInfo::CS_YUV420P16) ||
      ((vi.pixel_type == VideoInfo::CS_YUV422P10) && !parent->Enable_Y3_10_10 && !parent->Enable_V210) ||
      ((vi.pixel_type == VideoInfo::CS_YUV422P16) && !parent->Enable_Y3_10_16))
  {
    yuv42xp10_16_to_Px10_16((uint8_t *)lpBuffer, out_pitch,
      frame->GetReadPtr(), frame->GetPitch(),
      frame->GetReadPtr(PLANAR_U), frame->GetReadPtr(PLANAR_V), frame->GetPitch(PLANAR_U),
      vi.width, vi.height, frame->GetHeight(PLANAR_U), semi_packed_p16, parent->env);

  }
  else {
    if (vi.pixel_type == VideoInfo::CS_YUV422P10 && parent->Enable_V210) {
      int width = frame->GetRowSize(PLANAR_Y) / vi.ComponentSize();
      yuv422p10_to_v210((BYTE *)lpBuffer, frame->GetReadPtr(PLANAR_Y), frame->GetPitch(PLANAR_Y),
        frame->GetReadPtr(PLANAR_U), frame->GetReadPtr(PLANAR_V), frame->GetPitch(PLANAR_U), width, height);
    }
    else if (semi_packed_p10 && !parent->Enable_Y3_10_10 && !parent->Enable_V210) {
      // already handled
    }
    else if (vi.IsRGB64() && parent->Enable_b64a) {
      // BGRA -> big endian ARGB with byte swap
      uint8_t* pdst = (uint8_t *)lpBuffer + out_pitch * (height - 1); // upside down

      int srcpitch = frame->GetPitch();
      const BYTE *src = frame->GetReadPtr();
#ifdef INTEL_INTRINSICS
      if (ssse3)
        bgra_to_argbBE_ssse3(pdst, -out_pitch, src, srcpitch, vi.width, vi.height);
      else if (sse2)
        bgra_to_argbBE_sse2(pdst, -out_pitch, src, srcpitch, vi.width, vi.height);
      else
#endif
        bgra_to_argbBE_c(pdst, -out_pitch, src, srcpitch, vi.width, vi.height);
    }
    else if (vi.IsRGB48() || vi.IsRGB64())
    {
      // avisynth: upside down, output: back to normal
      parent->env->BitBlt((BYTE*)lpBuffer + out_pitch * (height - 1), -out_pitch, frame->GetReadPtr(), pitch, row_size, height);
    }
    else {
      parent->env->BitBlt((BYTE*)lpBuffer, out_pitch, frame->GetReadPtr(), pitch, row_size, height);
    }

    if (vi.pixel_type == VideoInfo::CS_YUV422P10 && parent->Enable_V210) {
      // intentionally empty
    }
    else if ((semi_packed_p10 && !parent->Enable_Y3_10_10 && !parent->Enable_V210) ||
      (semi_packed_p16 && !parent->Enable_Y3_10_16)) {
      // already handled
    }
    else { // for RGB48 and 64 these are zero sized
      parent->env->BitBlt((BYTE*)lpBuffer + (out_pitch*height),
        out_pitchUV, frame->GetReadPtr(plane1),
        frame->GetPitch(plane1), frame->GetRowSize(plane1),
        frame->GetHeight(plane1));

      parent->env->BitBlt((BYTE*)lpBuffer + (out_pitch*height + frame->GetHeight(plane1)*out_pitchUV),
        out_pitchUV, frame->GetReadPtr(plane2),
        frame->GetPitch(plane2), frame->GetRowSize(plane2),
        frame->GetHeight(plane2));

      // fill alpha, not from YUVA, only from RGBAP, because only RGBAP8-16 is mapped to alpha aware fourCCs
      if (vi.IsPlanarRGBA()) {
        parent->env->BitBlt((BYTE*)lpBuffer + (out_pitch*height + 2 * frame->GetHeight(plane1)*out_pitchUV),
          out_pitchUV, frame->GetReadPtr(PLANAR_A),
          frame->GetPitch(PLANAR_A), frame->GetRowSize(PLANAR_A),
          frame->GetHeight(PLANAR_A));
      }
    }
  }
  // no alpha?
//_RPT1(0, "CAVIStreamSynth::ReadFrame %d END\r\n", n);
}


////////////////////////////////////////////////////////////////////////
//////////// IAVIStream

STDMETHODIMP CAVIStreamSynth::Read(LONG lStart, LONG lSamples, LPVOID lpBuffer, LONG cbBuffer, LONG *plBytes, LONG *plSamples) {

#if defined(X86_32) && defined(MSVC)
  // TODO: Does this need 64-bit porting?
  __asm { // Force compiler to protect these registers!
    mov ebx,ebx;
    mov esi,esi;
    mov edi,edi;
  }
#endif

  parent->Lock();

  HRESULT result = Read2(lStart, lSamples, lpBuffer, cbBuffer, plBytes, plSamples);

  parent->Unlock();

  return result;
}

HRESULT CAVIStreamSynth::Read2(LONG lStart, LONG lSamples, LPVOID lpBuffer, LONG cbBuffer, LONG *plBytes, LONG *plSamples) {

  //  _RPT3(0,"%p->CAVIStreamSynth::Read(%ld samples at %ld)\n", this, lSamples, lStart);
  //  _RPT2(0,"\tbuffer: %ld bytes at %p\n", cbBuffer, lpBuffer);
#ifdef MSVC
  unsigned int fp_state = _controlfp( 0, 0 );
  _controlfp( FP_STATE, FP_MASK);
#endif

  const VideoInfo* const vi = parent->vi;

  bool targetIsConvertedToPackedRGB = (parent->Enable_PlanarToPackedRGB && (vi->IsPlanarRGB() || vi->IsPlanarRGBA()));

  VideoInfo vi_final = *vi;
  // if basic type is changed, that affects buffer size, we change the format here
  if (targetIsConvertedToPackedRGB) {
    // Enable_PlanarToPackedRGB results in packed RGB64 for bits>8, RGB24/32 for 8 bits
    if (vi->BitsPerComponent() == 8) {
      if (vi->IsPlanarRGB())
        vi_final.pixel_type = VideoInfo::CS_BGR24;
      else // planar RGBA
        vi_final.pixel_type = VideoInfo::CS_BGR32;
    }
    else // all 8+ bit planar RGB(A) is converted to RGB64
      vi_final.pixel_type = VideoInfo::CS_BGR64;
  }
  // silent mapping of 12/14/float bit YUV formats to 16 bit
  if (vi->pixel_type == VideoInfo::CS_YUV420P12 || vi->pixel_type == VideoInfo::CS_YUV420P14 || vi->pixel_type == VideoInfo::CS_YUV420PS)
    vi_final.pixel_type = VideoInfo::CS_YUV420P16;
  else if (vi->pixel_type == VideoInfo::CS_YUV422P12 || vi->pixel_type == VideoInfo::CS_YUV422P14 || vi->pixel_type == VideoInfo::CS_YUV422PS)
    vi_final.pixel_type = VideoInfo::CS_YUV422P16;
  else if (vi->pixel_type == VideoInfo::CS_YUV444P10 || vi->pixel_type == VideoInfo::CS_YUV444P12 || vi->pixel_type == VideoInfo::CS_YUV444P14 || vi->pixel_type == VideoInfo::CS_YUV444PS)
    vi_final.pixel_type = VideoInfo::CS_YUV444P16;
  else if (vi->pixel_type == VideoInfo::CS_YUVA444P10 || vi->pixel_type == VideoInfo::CS_YUVA444P12 || vi->pixel_type == VideoInfo::CS_YUVA444P14 || vi->pixel_type == VideoInfo::CS_YUVA444PS)
    vi_final.pixel_type = VideoInfo::CS_YUVA444P16;
  // -- pixel_type change end

  if (fAudio) {
    // buffer overflow patch -- Avery Lee - Mar 2006
    if (lSamples == AVISTREAMREAD_CONVENIENT)
      lSamples = (int)vi->AudioSamplesFromFrames(1);

    if (int64_t(lStart)+lSamples > vi->num_audio_samples) {
      lSamples = (int)(vi->num_audio_samples - lStart);
      if (lSamples < 0) lSamples = 0;
    }

    int bytes = (int)vi->BytesFromAudioSamples(lSamples);
    if (lpBuffer && bytes > cbBuffer) {
      lSamples = (int)vi->AudioSamplesFromBytes(cbBuffer);
      bytes = (int)vi->BytesFromAudioSamples(lSamples);
    }
    if (plBytes) *plBytes = bytes;
    if (plSamples) *plSamples = lSamples;
    if (!lpBuffer || !lSamples)
      return S_OK;

  } else {
    if (lStart >= vi->num_frames) {
      if (plSamples) *plSamples = 0;
      if (plBytes) *plBytes = 0;
      return S_OK;
    }

    const int image_size = parent->ImageSize(&vi_final);
    if (plSamples) *plSamples = 1;
    if (plBytes) *plBytes = image_size;

    if (!lpBuffer) {
      return S_OK;
    } else if (cbBuffer < image_size) {
      //      _RPT1(0,"\tBuffer too small; should be %ld samples\n", image_size);
      return AVIERR_BUFFERTOOSMALL;
    }
  }

#ifndef _DEBUG
  try {
    try {
#endif
      SehGuard seh_guard;

      if (fAudio)
        parent->filter_graph->GetAudio(lpBuffer, lStart, lSamples, parent->env);
      else
        ReadFrame(lpBuffer, lStart);

#ifndef _DEBUG
    }
    catch (AvisynthError &error) {
      parent->MakeErrorStream(error.msg);
      throw;
    }
    catch (SehException &seh) {
      char buf[256];
      _snprintf(buf, 255, "CAVIStreamSynth: %s at 0x%p", seh.m_msg, seh.m_addr);
      parent->MakeErrorStream(parent->env->SaveString(buf));
      throw;
    }
    catch (...) {
      parent->MakeErrorStream("CAVIStreamSynth: unhandled C++ exception");
      throw;
    }
  }
  catch (...) {
#ifdef X86_32
    _mm_empty();
#endif
#ifdef MSVC
    _clearfp();
    _controlfp( fp_state, FP_MASK);
#endif
    return E_FAIL;
  }
#endif
#ifdef X86_32
  _mm_empty();
#endif
#ifdef MSVC
  _clearfp();
  _controlfp( fp_state, FP_MASK);
#endif
  return S_OK;
}

STDMETHODIMP CAVIStreamSynth::ReadFormat(LONG lPos, LPVOID lpFormat, LONG *lpcbFormat) {
  _RPT2(0,"%p->CAVIStreamSynth::ReadFormat() (%s)\n", this, sName);

  if (!lpcbFormat) return E_POINTER;

  const bool UseWaveExtensible = parent->env->GetVarBool(VARNAME_UseWaveExtensible, false);

  if (!lpFormat) {
    *lpcbFormat = fAudio ? ( UseWaveExtensible ? sizeof(WAVEFORMATEXTENSIBLE)
      : sizeof(WAVEFORMATEX) )
      : sizeof(BITMAPINFOHEADER);
    return S_OK;
  }

  memset(lpFormat, 0, *lpcbFormat);

  const VideoInfo* const vi = parent->vi;

  if (fAudio) {
    if (UseWaveExtensible) {  // Use WAVE_FORMAT_EXTENSIBLE audio output format
#ifndef MSVC
      const GUID KSDATAFORMAT_SUBTYPE_PCM       = {0x00000001, 0x0000, 0x0010, {0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71}};
#endif
#ifndef MSVC
      const GUID KSDATAFORMAT_SUBTYPE_IEEE_FLOAT= {0x00000003, 0x0000, 0x0010, {0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71}};
#endif
      WAVEFORMATEXTENSIBLE wfxt;

      memset(&wfxt, 0, sizeof(wfxt));
      wfxt.Format.wFormatTag = WAVE_FORMAT_EXTENSIBLE;
      wfxt.Format.nChannels = (WORD)vi->AudioChannels();
      wfxt.Format.nSamplesPerSec = vi->SamplesPerSecond();
      wfxt.Format.wBitsPerSample = (WORD)(vi->BytesPerChannelSample() * 8);
      wfxt.Format.nBlockAlign = (WORD)vi->BytesPerAudioSample();
      wfxt.Format.nAvgBytesPerSec = wfxt.Format.nSamplesPerSec * wfxt.Format.nBlockAlign;
      wfxt.Format.cbSize = sizeof(wfxt) - sizeof(wfxt.Format);
      wfxt.Samples.wValidBitsPerSample = wfxt.Format.wBitsPerSample;

      const int SpeakerMasks[9] = { 0,
        0x00004, // 1   -- -- Cf
        0x00003, // 2   Lf Rf
        0x00007, // 3   Lf Rf Cf
        0x00033, // 4   Lf Rf -- -- Lr Rr
        0x00037, // 5   Lf Rf Cf -- Lr Rr
        0x0003F, // 5.1 Lf Rf Cf Sw Lr Rr
        0x0013F, // 6.1 Lf Rf Cf Sw Lr Rr -- -- Cr
        0x0063F, // 7.1 Lf Rf Cf Sw Lr Rr -- -- -- Ls Rs
      };
      wfxt.dwChannelMask = (unsigned)vi->AudioChannels() <= 8 ? SpeakerMasks[vi->AudioChannels()]
        : (unsigned)vi->AudioChannels() <=18 ? DWORD(-1) >> (32-vi->AudioChannels())
        : SPEAKER_ALL;

      unsigned int userChannelMask = (unsigned)(parent->env->GetVarInt(VARNAME_dwChannelMask, 0));
      if (userChannelMask != 0)
        wfxt.dwChannelMask = userChannelMask;

      wfxt.SubFormat = vi->IsSampleType(SAMPLE_FLOAT) ? KSDATAFORMAT_SUBTYPE_IEEE_FLOAT : KSDATAFORMAT_SUBTYPE_PCM;
      *lpcbFormat = min(*lpcbFormat, (LONG)sizeof(wfxt));
      memcpy(lpFormat, &wfxt, size_t(*lpcbFormat));
    }
    else {
      WAVEFORMATEX wfx;
      memset(&wfx, 0, sizeof(wfx));
      wfx.wFormatTag = vi->IsSampleType(SAMPLE_FLOAT) ? WAVE_FORMAT_IEEE_FLOAT : WAVE_FORMAT_PCM;
      wfx.nChannels = (WORD)vi->AudioChannels();
      wfx.nSamplesPerSec = vi->SamplesPerSecond();
      wfx.wBitsPerSample = (WORD)(vi->BytesPerChannelSample() * 8);
      wfx.nBlockAlign = (WORD)vi->BytesPerAudioSample();
      wfx.nAvgBytesPerSec = wfx.nSamplesPerSec * wfx.nBlockAlign;
      *lpcbFormat = min(*lpcbFormat, (LONG)sizeof(wfx));
      memcpy(lpFormat, &wfx, size_t(*lpcbFormat));
    }
  } else {
    bool targetIsConvertedToPackedRGB = (parent->Enable_PlanarToPackedRGB && (vi->IsPlanarRGB() || vi->IsPlanarRGBA()));

    VideoInfo vi_final = *vi;
    // if basic type is changed, that affects buffer size, we change the format here
    if (targetIsConvertedToPackedRGB) {
      // Enable_PlanarToPackedRGB results in packed RGB64 for bits>8, RGB24/32 for 8 bits
      if (vi->BitsPerComponent() == 8) {
        if (vi->IsPlanarRGB())
          vi_final.pixel_type = VideoInfo::CS_BGR24;
        else // planar RGBA
          vi_final.pixel_type = VideoInfo::CS_BGR32;
      }
      else // all 8+ bit planar RGB(A) is converted to RGB64
        vi_final.pixel_type = VideoInfo::CS_BGR64;
    }
    // silent mapping of 12/14 bit YUV formats to 16 bit
    if (vi->pixel_type == VideoInfo::CS_YUV420P12 || vi->pixel_type == VideoInfo::CS_YUV420P14 || vi->pixel_type == VideoInfo::CS_YUV420PS)
      vi_final.pixel_type = VideoInfo::CS_YUV420P16;
    else if (vi->pixel_type == VideoInfo::CS_YUV422P12 || vi->pixel_type == VideoInfo::CS_YUV422P14 || vi->pixel_type == VideoInfo::CS_YUV422PS)
      vi_final.pixel_type = VideoInfo::CS_YUV422P16;
    else if (vi->pixel_type == VideoInfo::CS_YUV444P10 || vi->pixel_type == VideoInfo::CS_YUV444P12 || vi->pixel_type == VideoInfo::CS_YUV444P14 || vi->pixel_type == VideoInfo::CS_YUV444PS)
      vi_final.pixel_type = VideoInfo::CS_YUV444P16;
    else if (vi->pixel_type == VideoInfo::CS_YUVA444P10 || vi->pixel_type == VideoInfo::CS_YUVA444P12 || vi->pixel_type == VideoInfo::CS_YUVA444P14 || vi->pixel_type == VideoInfo::CS_YUVA444PS)
      vi_final.pixel_type = VideoInfo::CS_YUVA444P16;
    // -- pixel_type change end

    BITMAPINFOHEADER bi;
    memset(&bi, 0, sizeof(bi));
    bi.biSize = sizeof(bi);
    bi.biWidth = vi_final.width;
    bi.biHeight = vi_final.height;
    bi.biPlanes = 1;
    bi.biBitCount = (WORD)vi_final.BitsPerPixel();
    if (vi_final.pixel_type == VideoInfo::CS_YUV422P10 && parent->Enable_V210)
      bi.biBitCount = 20;

    // Enable_PlanarToPackedRGB results in packed RGB64 for bits>8, RGB24/32 for 8 bits
    if (vi_final.IsRGB() && !vi_final.IsPlanar() && vi_final.BitsPerComponent() == 8)
      bi.biCompression = BI_RGB;
    else if (vi_final.IsYUY2())
      bi.biCompression = MAKEFOURCC('Y','U','Y','2');
    else if (vi_final.IsYV12())
      bi.biCompression = MAKEFOURCC('Y','V','1','2');
    else if (vi_final.IsY8())
      bi.biCompression = MAKEFOURCC('Y','8','0','0');
    else if (vi_final.IsYV24())
      bi.biCompression = MAKEFOURCC('Y','V','2','4');
    else if (vi_final.IsYV16())
      bi.biCompression = MAKEFOURCC('Y','V','1','6');
    else if (vi_final.IsYV411())
      bi.biCompression = MAKEFOURCC('Y','4','1','B');
    // avs+
    else if (vi_final.IsRGB64() && parent->Enable_b64a)
      bi.biCompression = MAKEFOURCC('b','6','4','a');
    else if (vi_final.IsRGB64())
      bi.biCompression = MAKEFOURCC('B','R','A',64); // BRA@ ie. BRA[64]
    else if (vi_final.IsRGB48())
      bi.biCompression = MAKEFOURCC('B','G','R',48); // BGR0 ie. BGR[48]
    else if (vi_final.pixel_type == VideoInfo::CS_YUV420P10)
      bi.biCompression = MAKEFOURCC('P','0','1','0');
    else if (vi_final.pixel_type == VideoInfo::CS_YUV420P16)
      bi.biCompression = MAKEFOURCC('P','0','1','6');
    else if (vi_final.pixel_type == VideoInfo::CS_YUV422P10 && parent->Enable_Y3_10_10)
      bi.biCompression = MAKEFOURCC('Y', '3', 10, 10); // Y3[10][10] (AV_PIX_FMT_YUV422P10) = planar YUV 422*10-bit
    else if (vi_final.pixel_type == VideoInfo::CS_YUV422P10 && parent->Enable_V210)
      bi.biCompression = MAKEFOURCC('v','2','1','0');
    else if (vi_final.pixel_type == VideoInfo::CS_YUV422P10)
      bi.biCompression = MAKEFOURCC('P','2','1','0');
    else if (vi_final.pixel_type == VideoInfo::CS_YUV422P16 && parent->Enable_Y3_10_16)
      bi.biCompression = MAKEFOURCC('Y', '3', 10, 16); // Y3[10][16] (AV_PIX_FMT_YUV422P16) = planar YUV 422*16-bit
    else if (vi_final.pixel_type == VideoInfo::CS_YUV422P16)
      bi.biCompression = MAKEFOURCC('P','2','1','6');
    else if (vi_final.pixel_type == VideoInfo::CS_YUV444P16 || vi_final.pixel_type == VideoInfo::CS_YUVA444P16)
      bi.biCompression = MAKEFOURCC('Y','4','1','6');
    else if (vi_final.pixel_type == VideoInfo::CS_RGBP)
      bi.biCompression = MAKEFOURCC('G','3', 0, 8);
    //      bi.biCompression = MAKEFOURCC('8','B','P','S'); special RLA encoded format, support G3[0][8] instead
    // MagicYUV implements these (planar rgb/rgba 10,12,14,16) G3[0][10], G4[0][10], G3[0][12], G4[0][12], G3[0][14], G4[0][14], G3[0][16], G4[0][16]
    else if (vi_final.pixel_type == VideoInfo::CS_RGBP10)
      bi.biCompression = MAKEFOURCC('G','3',0,10);
    else if (vi_final.pixel_type == VideoInfo::CS_RGBP12)
      bi.biCompression = MAKEFOURCC('G','3',0,12);
    else if (vi_final.pixel_type == VideoInfo::CS_RGBP14)
      bi.biCompression = MAKEFOURCC('G','3',0,14);
    else if (vi_final.pixel_type == VideoInfo::CS_RGBP16)
      bi.biCompression = MAKEFOURCC('G','3',0,16);
    else if (vi_final.pixel_type == VideoInfo::CS_RGBAP)
      bi.biCompression = MAKEFOURCC('G','4',0, 8); // similar to 10-16 bits
    else if (vi_final.pixel_type == VideoInfo::CS_RGBAP10)
      bi.biCompression = MAKEFOURCC('G','4',0,10);
    else if (vi_final.pixel_type == VideoInfo::CS_RGBAP12)
      bi.biCompression = MAKEFOURCC('G','4',0,12);
    else if (vi_final.pixel_type == VideoInfo::CS_RGBAP14)
      bi.biCompression = MAKEFOURCC('G','4',0,14);
    else if (vi_final.pixel_type == VideoInfo::CS_RGBAP16)
      bi.biCompression = MAKEFOURCC('G','4',0,16);
    else {
      _ASSERT(FALSE);
    }

    bi.biSizeImage = parent->ImageSize(&vi_final);
    *lpcbFormat = min(*lpcbFormat, (LONG)sizeof(bi));
    memcpy(lpFormat, &bi, size_t(*lpcbFormat));
  }
  return S_OK;
}

STDMETHODIMP CAVIStreamSynth::Write(LONG lStart, LONG lSamples, LPVOID lpBuffer,
  LONG cbBuffer, DWORD dwFlags, LONG FAR *plSampWritten,
  LONG FAR *plBytesWritten) {

  _RPT1(0,"%p->CAVIStreamSynth::Write()\n", this);

  return AVIERR_READONLY;
}
#endif
