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


#include "directshow_source.h"
#include <avs/minmax.h>

#define DSS_VERSION "2.6.1"

/************************************
 *          Logging Utility         *
 ************************************/

/* WARNING - For some stupid reason the compile chokes if I put ()'s around f in the macros
   so that f & log->mask cannot be subverted by the caller. Therefore the brackets have to
   go in the calls to the macros. DAMN!!!!! */

// Based on M$ _RPT<n> macros
//
// We print a timestamp, the objects address, the message
//
#define dssRPT0(f, s)                 _RPT0(0, "DSS " s);                \
      if (log && (f & log->mask)) fprintf(log->file, "%s %03x 0x%p 0x%08X " s, Tick(), f, this, GetCurrentThreadId())
#define dssRPT1(f, s, a1)             _RPT1(0, "DSS " s, a1);            \
      if (log && (f & log->mask)) fprintf(log->file, "%s %03x 0x%p 0x%08X " s, Tick(), f, this, GetCurrentThreadId(), a1)
#define dssRPT2(f, s, a1, a2)         _RPT2(0, "DSS " s, a1, a2);        \
      if (log && (f & log->mask)) fprintf(log->file, "%s %03x 0x%p 0x%08X " s, Tick(), f, this, GetCurrentThreadId(), a1, a2)
#define dssRPT3(f, s, a1, a2, a3)     _RPT3(0, "DSS " s, a1, a2, a3);    \
      if (log && (f & log->mask)) fprintf(log->file, "%s %03x 0x%p 0x%08X " s, Tick(), f, this, GetCurrentThreadId(), a1, a2, a3)
#define dssRPT4(f, s, a1, a2, a3, a4) _RPT4(0, "DSS " s, a1, a2, a3, a4);\
      if (log && (f & log->mask)) fprintf(log->file, "%s %03x 0x%p 0x%08X " s, Tick(), f, this, GetCurrentThreadId(), a1, a2, a3, a4)

// Reporting masks
enum {
  dssNEG   = 1 << 0, //   1    Format Negotiation
  dssSAMP  = 1 << 1, //   2    Received samples
  dssCALL  = 1 << 2, //   4    GetFrame/GetAudio calls
  dssCMD   = 1 << 3, //   8    Directshow callbacks
  dssPROC  = 1 << 4, //  16    Requests to Directshow
  dssERROR = 1 << 5, //  32    Errors
  dssREF   = 1 << 6, //  64    COM object use count
  dssNEW   = 1 << 7, // 128    New objects
  dssINFO  = 1 << 8, // 256    Extra info
  dssWAIT  = 1 << 9  // 512    Wait events
};

// Millisecond timestamp
char* Tick() {
  static unsigned long firstTick;
  static bool init = true;
  static char buf[16];

  unsigned long tick = timeGetTime();

  if (init) {
      init = false;
      firstTick = tick;
  }

  tick -= firstTick;

  unsigned msec = tick % 1000;
  tick /= 1000;
  unsigned sec = tick % 60;
  tick /= 60;
  unsigned min = tick % 60;
  tick /= 60;
  tick %= 99;

  _snprintf(buf, 15, "%02u:%02u:%02u.%03u", tick, min, sec, msec);
  buf[15] = 0;

  return buf;
}

LOG::LOG(const char* fn, int _mask, IScriptEnvironment* env) : mask(_mask), count(0) {
  file = fopen(fn, "a");
  if (!file)
    env->ThrowError("DirectShowSource: Not able to open log file, '%s' for appending.", fn);

  fprintf(file, "%s fff 0x00000000 DirectShowSource " DSS_VERSION " build:" __DATE__ " [" __TIME__ "]\n", Tick());
}

LOG::~LOG() {
  fclose(file);
}

void LOG::DelRef(const char* s) {
  fprintf(file, "%s fff 0x00000000 Close %s log %d.\n", Tick(), s, count);

  if (!(--count)) {
    delete this;
  }
};

// For some reason KSDATAFORMAT_SUBTYPE_IEEE_FLOAT and KSDATAFORMAT_SUBTYPE_PCM doesn't work - we construct the GUID manually!
const GUID SUBTYPE_IEEE_AVSPCM     = {0x00000001, 0x0000, 0x0010, {0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71}};
const GUID SUBTYPE_IEEE_AVSFLOAT   = {0x00000003, 0x0000, 0x0010, {0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71}};

// Cope with missing code in DirectShow "AVI/WAV File Source"
const GUID MEDIASUBTYPE_extensible = {0x0000FFFE, 0x0000, 0x0010, {0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71}};

#define FourCC(ch4) ( (((DWORD)(ch4) &       0xFF) << 24) | \
                      (((DWORD)(ch4) &     0xFF00) <<  8) | \
                      (((DWORD)(ch4) &   0xFF0000) >>  8) | \
                      (((DWORD)(ch4) & 0xFF000000) >> 24) )

const GUID MEDIASUBTYPE_I420 = {FourCC('I420'), 0x0000, 0x0010, {0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71}};
// Causes redefinition error
// Already defined by platform headers: const GUID MEDIASUBTYPE_NV12 = {FourCC('NV12'), 0x0000, 0x0010, {0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71}};
const GUID MEDIASUBTYPE_YV16 = {FourCC('YV16'), 0x0000, 0x0010, {0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71}};
const GUID MEDIASUBTYPE_YV24 = {FourCC('YV24'), 0x0000, 0x0010, {0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71}};
const GUID MEDIASUBTYPE_BRA64 = { FourCC('BRA\100'), 0x0000, 0x0010,{ 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71 } }; // BRA@ ie. BRA[64]
const GUID MEDIASUBTYPE_BGR48 = { FourCC('BGR\060'), 0x0000, 0x0010,{ 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71 } }; // BGR0 ie. BGR[48]


// Format a GUID for printing
char* PrintGUID(const GUID *g) {

  static char buf[41];

  if (g) {
    _snprintf(buf, 40, "{%08x-%04hx-%04hx-%02x%02x-%02x%02x%02x%02x%02x%02x}\0",
              g->Data1,    g->Data2,    g->Data3,    g->Data4[0], g->Data4[1],
              g->Data4[2], g->Data4[3], g->Data4[4], g->Data4[5], g->Data4[6], g->Data4[7]);
    buf[40] = 0;
  }
  else {
    lstrcpy(buf, "<null>");
  }
  return buf;
}

// Format a state for printing
char* PrintState(FILTER_STATE state) {
  switch (state) {
    case State_Stopped: return "State_Stopped";
    case State_Paused:  return "State_Paused";
    case State_Running: return "State_Running";
    default: break;
  }
  return "State Unknown";
}

// Format a FourCC for printing
static inline char BePrintable(int ch) {
  ch &= 0xff;
  return (char)(isprint(ch) ? ch : '.');
}

char* PrintFourCC(const int fccType) {
  static char fcc[5];

  fcc[0] = BePrintable(fccType      );
  fcc[1] = BePrintable(fccType >>  8);
  fcc[2] = BePrintable(fccType >> 16);
  fcc[3] = BePrintable(fccType >> 24);
  fcc[4] = 0;

  return fcc;
}

char* PrintAudioType(const int type) {
  switch (type) {
    case SAMPLE_INT8:  return "Int8";
    case SAMPLE_INT16: return "Int16";
    case SAMPLE_INT24: return "Int24";
    case SAMPLE_INT32: return "Int32";
    case SAMPLE_FLOAT: return "SFloat";
    default: break;
  }

  static char buf[21];
  _snprintf(buf, 20, "unknown 0x%x", type);
  buf[20] = 0;
  return buf;
}

char* PrintPixelType(int pixel_type) {
  switch (pixel_type) {
    case VideoInfo::CS_BGR24:    return "RGB24";
    case VideoInfo::CS_BGR32:    return "RGB32";
    case VideoInfo::CS_BGR48:    return "RGB48";
    case VideoInfo::CS_BGR64:    return "RGB64";
    case VideoInfo::CS_YUY2:     return "YUY2";
//  case VideoInfo::CS_RAW32:    return "Raw32";
    case VideoInfo::CS_YV24:     return "YV24";
    case VideoInfo::CS_YV16:     return "YV16";
    case VideoInfo::CS_YV12:     return "YV12";
    case VideoInfo::CS_I420:     return "I420";
//  case VideoInfo::CS_YUV9:     return "YUV9";
    case VideoInfo::CS_YV411:    return "YV411";
    case VideoInfo::CS_Y8:       return "Y8";
    default: break;
  }

  static char buf[12];
  _snprintf(buf, 11, "0x%08x", pixel_type);
  buf[11] = 0;
  return buf;
}

/************************************
 *             GetSample            *
 ************************************/


inline void InitMediaType(AM_MEDIA_TYPE* &media_type, const GUID &major, const GUID &sub) {
  media_type = new AM_MEDIA_TYPE;
  memset(media_type, 0, sizeof(AM_MEDIA_TYPE));
  media_type->majortype  = major;
  media_type->subtype    = sub;
  media_type->formattype = GUID_NULL;
}


GetSample::GetSample(bool _load_audio, bool _load_video, unsigned _media, LOG* _log)
  : load_audio(_load_audio), load_video(_load_video), media(_media),
    streamName(_load_audio ? "audio" : "video"), log(_log), lockvi(false), pvf(0) {

    if(log) log->AddRef();

    dssRPT1(dssNEW, "New GetSample (%s).\n", streamName);

    am_media_type = 0;
    refcnt = 1;
    Allocator = 0;
    source_pin = 0;
    filter_graph = 0;
    pclock = 0;
    m_pPos =0;
    state = State_Stopped;
    avg_time_per_frame = 0;
    av_sample_bytes = 0;
    av_buffer = 0;
    seeking = flushing = end_of_stream = false;
    memset(&vi, 0, sizeof(vi));
    m_bInvertFrames = false;
    sample_end_time = sample_start_time = segment_start_time = segment_stop_time = 0;
    evtDoneWithSample = ::CreateEvent(NULL, FALSE, FALSE, NULL);
    evtNewSampleReady = ::CreateEvent(NULL, FALSE, FALSE, NULL);

    if (load_audio) {
      unsigned i=0;
      InitMediaType(my_media_types[i++], MEDIATYPE_Audio, MEDIASUBTYPE_IEEE_FLOAT);
      InitMediaType(my_media_types[i++], MEDIATYPE_Audio, MEDIASUBTYPE_PCM);
      no_my_media_types = i;
    }
    else {
      // Make sure my_media_types[16] is long enough!
      unsigned i=0;
//    In the order we want codecs to bid
      if (media & mediaYV24)   InitMediaType(my_media_types[i++], MEDIATYPE_Video, MEDIASUBTYPE_YV24);
      if (media & mediaYV16)   InitMediaType(my_media_types[i++], MEDIATYPE_Video, MEDIASUBTYPE_YV16);
      if (media & mediaYV12)   InitMediaType(my_media_types[i++], MEDIATYPE_Video, MEDIASUBTYPE_YV12);
      if (media & mediaI420)   InitMediaType(my_media_types[i++], MEDIATYPE_Video, MEDIASUBTYPE_I420);
      if (media & mediaNV12)   InitMediaType(my_media_types[i++], MEDIATYPE_Video, MEDIASUBTYPE_NV12); //Needs unpacking
      if (media & mediaYUY2)   InitMediaType(my_media_types[i++], MEDIATYPE_Video, MEDIASUBTYPE_YUY2);
      if (media & mediaAYUV)   InitMediaType(my_media_types[i++], MEDIATYPE_Video, MEDIASUBTYPE_AYUV); //Needs unpacking
      if (media & mediaY41P)   InitMediaType(my_media_types[i++], MEDIATYPE_Video, MEDIASUBTYPE_Y41P); //Needs unpacking
      if (media & mediaY411)   InitMediaType(my_media_types[i++], MEDIATYPE_Video, MEDIASUBTYPE_Y411); //Needs unpacking
//    if (media & mediaYUV9)   InitMediaType(my_media_types[i++], MEDIATYPE_Video, MEDIASUBTYPE_YUV9);
      if (media & mediaARGB)   InitMediaType(my_media_types[i++], MEDIATYPE_Video, MEDIASUBTYPE_ARGB32);
      if (media & mediaRGB32)  InitMediaType(my_media_types[i++], MEDIATYPE_Video, MEDIASUBTYPE_RGB32);
      if (media & mediaRGB24)  InitMediaType(my_media_types[i++], MEDIATYPE_Video, MEDIASUBTYPE_RGB24);
      if (media & mediaRGB64)  InitMediaType(my_media_types[i++], MEDIATYPE_Video, MEDIASUBTYPE_BRA64);
      if (media & mediaRGB48)  InitMediaType(my_media_types[i++], MEDIATYPE_Video, MEDIASUBTYPE_BGR48);
      no_my_media_types = i;
      if (media == mediaNONE) media = mediaAUTO;
    }
  }

  GetSample::~GetSample() {
    dssRPT1(dssNEW, "~GetSample(%s).\n", streamName);

    if (am_media_type)
      DeleteMediaType(am_media_type);
    am_media_type = 0;

    SetEvent(evtDoneWithSample);
    SetEvent(evtNewSampleReady);

    CloseHandle(evtDoneWithSample);
    CloseHandle(evtNewSampleReady);

    for (unsigned i=0; i<no_my_media_types; i++)
      delete my_media_types[i];

    if (Allocator) {
      dssRPT1(dssNEW, "Releasing Allocator 0x%08p.\n", Allocator);
      Allocator->Release();
      Allocator = 0;
    }

    if (log) log->DelRef(streamName);
  }


  const AM_MEDIA_TYPE *GetSample::GetMediaType(unsigned pos) {
    return my_media_types[(pos < no_my_media_types) ? pos : 0];
  }


  bool GetSample::WaitForStart(DWORD &timeout) {

    // Give the graph an opportunity to start before we return empty data

    if (state == State_Stopped) {
      dssRPT1(dssERROR, "WaitForStart() state == State_Stopped (%s)\n", streamName); // Opps should never happen!
      return false;
    }

    if (graphTimeout)
      graphTimeout = (WaitForSingleObject(evtNewSampleReady, timeout) == WAIT_TIMEOUT);
    if (graphTimeout) {
      dssRPT0(dssERROR, "** TIMEOUT ** waiting for Graph to start!\n");
      timeout = 55; // 1 windows tick
    }
    return graphTimeout;
  }


  PVideoFrame GetSample::GetCurrentFrame(IScriptEnvironment* env, int n, bool _TrapTimeouts, DWORD &timeout) {

    if (WaitForStart(timeout))
      if (_TrapTimeouts) {
        dssRPT1(dssERROR, "GetCurrentFrame() Timeout waiting for video frame=%d!\n", n);
        env->ThrowError("DirectShowSource : Timeout waiting for video.");
    }

    if (av_buffer) {

      pvf = env->NewVideoFrame(vi);

      // Put any knowledge of video packing and alignment here and
      // keep it independant of any AviSynth packing and alignment.

      PBYTE buf = av_buffer;

      if (!vi.IsPlanar() || vi.IsY8()) { // All DIB formats have rows 32bit aligned

        const int rowsize = pvf->GetRowSize();
        int height  = pvf->GetHeight();
        int stride;

        // Check for rows not being 32 bit aligned as expected
        if (((rowsize*height + 3)&~3) == ((av_sample_bytes+3)&~3)) {
          // Setup for a naughty packed layout
          stride = rowsize;
        } else  {
          // Setup for the expected padded layout
          stride  = (rowsize+3)&~3;
        }

        // Check input size is adequate
        if (av_sample_bytes < height*stride) {
          // Truncate the height to avoid buffer overrun.
          dssRPT2(dssERROR, "GetCurrentFrame() Input buffer to small, %d expecting %d!\n", av_sample_bytes, height*stride);
          height = av_sample_bytes/stride;
        }

        if (m_bInvertFrames) {
          buf += stride * (height - 1);
          stride = -stride;
        }

        env->BitBlt(pvf->GetWritePtr(), pvf->GetPitch(), buf, stride, rowsize, height);
      }
      else {
        const int rowsize   = pvf->GetRowSize(PLANAR_Y);
        int height          = pvf->GetHeight(PLANAR_Y);

        const int UVrowsize = pvf->GetRowSize(PLANAR_V);
        int UVheight        = pvf->GetHeight(PLANAR_V);

        // Unlike DIB based formats planar data is supposed to be fully packed
        // but some implemetation incorrectly pad both the Y and UV rows out
        // to 32 bit while other implementations pad the Y rows out to 32 bit
        // and pad the UV rows as subsampled Y rows.

        // Start with what is expected, i.e. packed
        int stride   = rowsize;
        int UVstride = UVrowsize;

        if (media & mediaPAD) {
          // Use mod4 Y row with subsampled UV rows
          stride   = (rowsize+3) & ~3;
          UVstride = stride >> vi.GetPlaneWidthSubsampling(PLANAR_U);
        }

        BYTE* dstY = pvf->GetWritePtr(PLANAR_Y);
        BYTE* dstU = pvf->GetWritePtr(PLANAR_U);
        BYTE* dstV = pvf->GetWritePtr(PLANAR_V);
        BYTE* const dstT = dstV;
        BYTE* srcP = buf;

        const int pitchY = pvf->GetPitch(PLANAR_Y);
        const int pitchUV = pvf->GetPitch(PLANAR_U);
        int src_pitch;

        switch (am_media_type->subtype.Data1)
        {
          // Standard planar formats
          case FourCC('I420'):
            // Swap U and V pointers
            dstV = dstU;
            dstU = dstT;
          case FourCC('YV12'):
          case FourCC('YV16'):
          case FourCC('YV24'):
            dssRPT3(dssINFO, "GetCurrentFrame() Input buffer size %d, stride %d, UVstride %d\n",
                             av_sample_bytes, stride, UVstride);

            // Check input size is adequate
            if (av_sample_bytes < height*stride) {
              // Truncate the height to avoid buffer overrun.
              dssRPT2(dssERROR, "GetCurrentFrame() Input buffer to small, %d expecting %d!\n", av_sample_bytes, height*stride);
              height = av_sample_bytes/stride;
            }
            env->BitBlt(dstY, pitchY, srcP, stride, rowsize, height);

            // Check input size is adequate
            if (av_sample_bytes < height*stride+2*UVheight*UVstride) {
              // Truncate the height to avoid buffer overrun.
              dssRPT2(dssERROR, "GetCurrentFrame() Input buffer to small, %d expecting %d!\n",
                                av_sample_bytes, height*stride+2*UVheight*UVstride);
              UVheight = (av_sample_bytes-height*stride)/(2*UVstride);
            }
            // V plane first, after aligned end of Y plane
            srcP += stride * height;
            env->BitBlt(dstV, pitchUV, srcP, UVstride, UVrowsize, UVheight);

            // And U plane last, after aligned end of V plane
            srcP += UVstride * UVheight;
            env->BitBlt(dstU, pitchUV, srcP, UVstride, UVrowsize, UVheight);
            break;

          // 4:2:0 which needs packed chroma -> planar conversion.
          case FourCC('NV12'):

            UVstride *= 2;

            dssRPT3(dssINFO, "GetCurrentFrame() Input buffer size %d, stride %d, UVstride %d\n",
                             av_sample_bytes, stride, UVstride);

            // Check input size is adequate
            if (av_sample_bytes < height*stride) {
              // Truncate the height to avoid buffer overrun.
              dssRPT2(dssERROR, "GetCurrentFrame() Input buffer to small, %d expecting %d!\n",
                                av_sample_bytes, height*stride);
              height = av_sample_bytes/stride;
            }
            env->BitBlt(dstY, pitchY, srcP, stride, rowsize, height);

            // Check input size is adequate
            if (av_sample_bytes < height*stride+UVheight*UVstride) {
              // Truncate the height to avoid buffer overrun.
              dssRPT2(dssERROR, "GetCurrentFrame() Input buffer to small, %d expecting %d!\n",
                                av_sample_bytes, height*stride+UVheight*UVstride);
              UVheight = (av_sample_bytes-height*stride)/UVstride;
            }
            // Packed UV plane after aligned end of Y plane
            srcP += stride * height;

            {for (int y=0; y<UVheight; y++) {
              for(int x=0; x<UVrowsize; x++) {
                dstU[x] = srcP[(x*2)+0];
                dstV[x] = srcP[(x*2)+1];
              }
              dstU += pitchUV;
              dstV += pitchUV;
              srcP += UVstride;
            }}
            break;

          // 4:4:4 which needs packed -> planar conversion.
          case FourCC('AYUV'):
            // V0 U0 Y0 A0
            src_pitch = vi.width * 4;  // Naturally aligned.

            // Check input size is adequate
            if (av_sample_bytes < height*src_pitch) {
              dssRPT2(dssERROR, "GetCurrentFrame() Input buffer to small, %d expecting %d!\n",
                                av_sample_bytes, height*src_pitch);
              height = av_sample_bytes/src_pitch;
            }

            {for (int y=0; y<height; y++) {
              for(int x=0; x<rowsize; x++) {
                dstV[x] = srcP[(x*4)+0];
                dstU[x] = srcP[(x*4)+1];
                dstY[x] = srcP[(x*4)+2];
              }
              dstY += pitchY;
              dstU += pitchUV;
              dstV += pitchUV;
              srcP += src_pitch;
            }}
            break;

          // 4:1:1 which needs packed -> planar conversion.
          case FourCC('Y41P'):
          case FourCC('Y411'):
            // U0 Y0 V0 Y1   U4 Y2 V4 Y3   Y4 Y5 Y6 Y7
            src_pitch = (vi.width / 8) * 12;  // Naturally aligned.

            // Check input size is adequate
            if (av_sample_bytes < height*src_pitch) {
              dssRPT2(dssERROR, "GetCurrentFrame() Input buffer to small, %d expecting %d!\n",
                                av_sample_bytes, height*src_pitch);
              height = av_sample_bytes/src_pitch;
            }

            {for (int y=0; y<height; y++) {
              for(int px=0, dx=0; px<rowsize; px+=12, dx+=2) {
                dstU[dx+0] = srcP[px+0];
                dstU[dx+1] = srcP[px+4];

                dstV[dx+0] = srcP[px+2];
                dstV[dx+1] = srcP[px+6];

                dstY[dx*4+0] = srcP[px+1];
                dstY[dx*4+1] = srcP[px+3];
                dstY[dx*4+2] = srcP[px+5];
                dstY[dx*4+3] = srcP[px+7];

                dstY[dx*4+4] = srcP[px+8];
                dstY[dx*4+5] = srcP[px+9];
                dstY[dx*4+6] = srcP[px+10];
                dstY[dx*4+7] = srcP[px+11];
              }
              dstY += pitchY;
              dstU += pitchUV;
              dstV += pitchUV;
              srcP += src_pitch;
            }}
            break;

          default:
            dssRPT2(dssERROR, "GetCurrentFrame() unknown pixel type '%s' %s\n",
                    PrintFourCC(am_media_type->subtype.Data1), PrintGUID(&am_media_type->subtype));
            break;
        }
      }
    }
    else if (pvf) {
      dssRPT1(dssERROR, "GetCurrentFrame() Returning last good frame %d!\n", n);
    }
    else {
      dssRPT1(dssERROR, "GetCurrentFrame() Returning ** BLANK ** frame %d!\n", n);

      pvf = env->NewVideoFrame(vi);

      // If the graph still hasn't started yet we won't have a current frame
      // so dummy up a grey frame so at least things won't be fatal.

      memset(pvf->GetWritePtr(), 128, pvf->GetPitch()*pvf->GetHeight() +
                                      pvf->GetPitch(PLANAR_U)*pvf->GetHeight(PLANAR_U)*2);
    }
    return pvf;
  }


  HRESULT GetSample::StartGraph(IGraphBuilder* gb) {
    dssRPT1(dssPROC, "StartGraph(%s) enter...\n", streamName);
    try {
      ResetEvent(evtDoneWithSample);  // Nuke any unused SetEvents
      ResetEvent(evtNewSampleReady);  // Nuke any unused SetEvents
      IMediaControl* mc;
      gb->QueryInterface(&mc);
      seeking = flushing = end_of_stream = false;
      pvf = NULL; // Nuke current frame
      HRESULT hr = mc->Run();
      dssRPT2(dssPROC, "StartGraph(%s) mc->Run() = 0x%x\n", streamName, hr);
      // Retry!!
      if (hr == E_FAIL) {
        hr = mc->Run();
        dssRPT2(dssPROC, "Retry(%s) mc->Run() = 0x%x\n", streamName, hr);
      }
      if (hr == S_FALSE) {
        // Damn! graph is stuffing around and has not started (yet?)
        OAFilterState fs = State_Stopped;
        hr = mc->GetState(5000, &fs); // Give it 5 seconds to sort itself out
        dssRPT3(dssPROC, "StartGraph(%s) mc->GetState(%s) = 0x%x\n", streamName, PrintState(FILTER_STATE(fs)), hr);
        if ((fs == State_Running) && ((hr == S_OK) || (hr == VFW_S_STATE_INTERMEDIATE)))
          hr = S_OK; // It is good or still may become good
        else if (SUCCEEDED(hr))
          hr = E_FAIL;  // It's playing possum and about to lock up
//      else
//        it's totally screwed
      }
//    else
//      hr == S_OK or some serious error like the infamous E_FAIL
      mc->Release();
      graphTimeout = true;
      dssRPT2(dssPROC, "StartGraph(%s) ... exit 0x%x\n", streamName, hr);
      return hr;
    }
    catch (...) {
      dssRPT1((dssERROR|dssPROC), "StartGraph(%s) Unknown Exception!\n", streamName);
      return E_FAIL;
    }
  }

  void GetSample::StopGraph(IGraphBuilder* gb) {
    dssRPT1(dssPROC, "StopGraph() indicating done with sample - %s\n", PrintState(state));
    IMediaControl* mc;
    gb->QueryInterface(&mc);
    SetEvent(evtDoneWithSample); // Free task if waiting
    graphTimeout = true;
    mc->Stop();
    mc->Release();
    if (m_pPos)
      m_pPos->Release();
    m_pPos = 0;
  }

  void GetSample::PauseGraph(IGraphBuilder* gb) {
    dssRPT0(dssPROC, "PauseGraph()\n");
    IMediaControl* mc;
    gb->QueryInterface(&mc);
    mc->Pause();
    mc->Release();
  }

  HRESULT GetSample::SeekTo(__int64 pos, IGraphBuilder* gb) {
    HRESULT hr;

    dssRPT1(dssPROC, "SeekTo() seeking to new position %I64d\n", pos);

    // If trying to seek past the known end of stream just exit.
    if (end_of_stream && pos >= GetSampleStartTime()) {
      dssRPT1(dssPROC, "SeekTo() End_of_stream last samples position %I64d\n", GetSampleStartTime());
      return S_OK;
    }

    seeking = true;

    IMediaControl* mc = NULL;
    try {
      gb->QueryInterface(&mc);
      SetEvent(evtDoneWithSample); // Free task if waiting
      graphTimeout = true;
      mc->Stop();
      dssRPT0(dssPROC, "SeekTo() mc->Stop()\n");
      mc->Release();
      if (m_pPos)
        m_pPos->Release();
      m_pPos = 0;
    }
    catch (...) {
      if (mc) mc->Release();
    }

    IMediaSeeking* ms = NULL;
    try {
      gb->QueryInterface(&ms);

//    It looks like GetCapabilities() is a useless liar. So it seems the
//    best thing is to just try the seeks and test the result code
//
//    ms->GetCapabilities(&dwCaps);

      ms->SetTimeFormat(&TIME_FORMAT_MEDIA_TIME); // Setting can give E_NOTIMPL

      LONGLONG pCurrent = -1, pStop;
      GUID time_f;

      ms->GetTimeFormat(&time_f);                 // so check what it currently is
      if (time_f != TIME_FORMAT_MEDIA_TIME) {
        // Probably should implement code for all the time formats  :: FIXME
        dssRPT0(dssERROR, "Could not set time format to media time!\n");
      }

      if (SUCCEEDED(hr = ms->SetPositions(&pos, AM_SEEKING_AbsolutePositioning, NULL, AM_SEEKING_NoPositioning)))
          goto SeekExit;
      dssRPT1(dssERROR, "Absolute seek failed! 0x%x\n", hr); // and now trying relative seek

      if (FAILED(hr = ms->GetPositions(&pCurrent, &pStop)) || (pCurrent == -1)) {
        dssRPT1(dssERROR, "GetPositions failed! 0x%x\n", hr);
        pCurrent = GetSampleStartTime(); // Wing it from last the sample delived
      }

      pCurrent = pos - pCurrent;
      if (FAILED(hr = ms->SetPositions(&pCurrent, AM_SEEKING_RelativePositioning, NULL, AM_SEEKING_NoPositioning))) {
        dssRPT1(dssERROR, "Relative seek failed! 0x%x\n", hr); }

SeekExit:
      ms->GetPositions(&pCurrent, &pStop);
      dssRPT2(dssPROC, "SeekTo() ms->GetPositions(%I64d, %I64d)\n", pCurrent, pStop);
      ms->Release();
    }
    catch (...) {
      dssRPT0((dssERROR|dssPROC), "SeekTo() Unknown Exception!\n");
      if (ms) ms->Release();
      return E_FAIL;
    }

    hr = SUCCEEDED(hr) ? S_OK : S_FALSE;

    HRESULT hr1 = StartGraph(gb);
    if (FAILED(hr1))
      hr = hr1;  // pretty serious the Graph is not running

    return hr;
  }


  bool GetSample::NextSample(DWORD &timeout) {

    if (end_of_stream) {
      dssRPT1(dssPROC, "NextSample() end of stream (%s)\n", streamName);
      return false;
    }

    if (state == State_Stopped) {
      dssRPT1(dssERROR, "NextSample() state == State_Stopped (%s)\n", streamName); // Opps should never happen!
      return false;
    }

    // If the graph didn't start, check if it has yet. We absolutly have to keep in lock
    // step with the evtNewSampleReady and evtDoneWithSample synchronizaton objects. This
    // is a bit kludgy because when it happens blank frames or silence are returned but
    // it is better than returning totally corrupt data.

    if (WaitForStart(timeout)) return false;

    dssRPT1(dssPROC, "NextSample() indicating done with previous sample...(%s)\n", streamName);

    SetEvent(evtDoneWithSample);  // We indicate that Receive can run again. We have now finished using the frame.

    dssRPT1(dssPROC, "...NextSample() waiting for new sample...(%s)\n", streamName);
    graphTimeout = false;
    if (WaitForSingleObject(evtNewSampleReady, timeout) == WAIT_TIMEOUT) {
      dssRPT1(dssERROR, "...NextSample() TIMEOUT waiting for new sample (%s)\n", streamName);
      timeout = 55;
      graphTimeout = true;
      return false;
    }

    dssRPT1(dssPROC, "...NextSample() done waiting for new sample (%s)\n", streamName);
    return !end_of_stream;
  }

  // IUnknown

  ULONG __stdcall GetSample::AddRef() {
    ULONG ref = InterlockedIncrement(&refcnt);
    dssRPT1(dssREF, "GetSample::AddRef() -> %d\n", ref);
    return ref;
  }

  ULONG __stdcall GetSample::Release() {
    ULONG ref = InterlockedDecrement(&refcnt);
    dssRPT1(dssREF, "GetSample::Release() -> %d\n", ref);
    return ref;
  }

  HRESULT __stdcall GetSample::QueryInterface(REFIID iid, void** ppv) {
    if      (iid == IID_IUnknown)     *ppv = static_cast<IUnknown*>(static_cast<IBaseFilter*>(this));
    else if (iid == IID_IPersist)     *ppv = static_cast<IPersist*>(this);
    else if (iid == IID_IMediaFilter) *ppv = static_cast<IMediaFilter*>(this);
    else if (iid == IID_IBaseFilter)  *ppv = static_cast<IBaseFilter*>(this);
    else if (iid == IID_IPin)         *ppv = static_cast<IPin*>(this);
    else if (iid == IID_IMemInputPin) *ppv = static_cast<IMemInputPin*>(this);
    else if (iid == IID_IMediaSeeking || iid == IID_IMediaPosition) {
      if (!source_pin) {
        *ppv = 0;
        dssRPT1(dssERROR, "GetSample::QueryInterface(%s, ppv) ** E_NOINTERFACE **, No Source Pin!\n", PrintGUID(&iid));
        return E_NOINTERFACE;
      }
      if (m_pPos == NULL)  {
        // We have not created the CPosPassThru object yet. Do so now.
        HRESULT hr = CreatePosPassThru(NULL , FALSE, static_cast<IPin*>(this), &m_pPos);

        if (FAILED(hr))  {
          *ppv = 0;
          dssRPT1(dssERROR, "GetSample::QueryInterface(%s, ppv), Failed CreatePosPassThru!\n", PrintGUID(&iid));
          return hr;
        }
      }
      dssRPT1(dssCMD, "GetSample::QueryInterface(%s, ppv) -> m_pPos\n", PrintGUID(&iid));
      return m_pPos->QueryInterface(iid, ppv);
    }
    else {
      *ppv = 0;
      dssRPT1(dssCMD, "GetSample::QueryInterface(%s, ppv) ** E_NOINTERFACE **\n", PrintGUID(&iid));
      return E_NOINTERFACE;
    }
    AddRef();
    dssRPT1(dssCMD, "GetSample::QueryInterface(%s, ppv)\n", PrintGUID(&iid));
    return S_OK;
  }

  // IPersist

  HRESULT __stdcall GetSample::GetClassID(CLSID* pClassID) {
    dssRPT0(dssCMD, "GetSample::GetClassID() E_NOTIMPL\n");
    return E_NOTIMPL;
  }

  // IMediaFilter

  HRESULT __stdcall GetSample::Stop() {
    dssRPT1(dssCMD, "GetSample::Stop(), state was %s\n", PrintState(state));
    state = State_Stopped;
    SetEvent(evtDoneWithSample);
    graphTimeout = true;
/*
    if (Allocator) {
      HRESULT result = Allocator->Decommit();
      dssRPT2(dssNEW, "GetSample::Stop(), %x = 0x%08x->Decommit().\n", result, Allocator);
    }
*/
    return S_OK;
  }

  HRESULT __stdcall GetSample::Pause() {
    dssRPT1(dssCMD, "GetSample::Pause(), state was %s\n", PrintState(state));
    state = State_Paused;
    return S_OK;
  }

  HRESULT __stdcall GetSample::Run(REFERENCE_TIME tStart) {
    dssRPT2(dssCMD, "GetSample::Run(%I64d), state was %s\n", tStart, PrintState(state));
    ResetEvent(evtDoneWithSample);
    state = State_Running;
    return S_OK;
  }

  HRESULT __stdcall GetSample::GetState(DWORD dwMilliSecsTimeout, FILTER_STATE* State) {
    if (!State) {
      dssRPT1(dssERROR, "GetSample::GetState() ** E_POINTER **, state is %s\n", PrintState(state));
      return E_POINTER;
    }
    dssRPT1(dssCMD, "GetSample::GetState(), state is %s\n", PrintState(state));
    *State = state;
    return S_OK;
  }

  HRESULT __stdcall GetSample::SetSyncSource(IReferenceClock* pClock) {
    dssRPT2(dssCMD, "GetSample::SetSyncSource(0x%08p), was 0x%08p\n", pClock, pclock);
    pclock = pClock;
    return S_OK;
  }

  HRESULT __stdcall GetSample::GetSyncSource(IReferenceClock** ppClock) {
    if (!ppClock) {
      dssRPT1(dssERROR, "GetSample::GetSyncSource() ** E_POINTER **, is 0x%08p\n", pclock);
      return E_POINTER;
    }
    dssRPT1(dssCMD, "GetSample::GetSyncSource(), is 0x%08p\n", pclock);
    *ppClock = pclock;
    if (pclock) pclock->AddRef();
    return S_OK;
  }

  // IBaseFilter

  HRESULT __stdcall GetSample::EnumPins(IEnumPins** ppEnum) {
    if (!ppEnum) {
      dssRPT0(dssERROR, "GetSample::EnumPins() ** E_POINTER **\n");
      return E_POINTER;
    }
    dssRPT0(dssCMD, "GetSample::EnumPins()\n");
    *ppEnum = new GetSampleEnumPins(this);
    return *ppEnum ? S_OK : E_OUTOFMEMORY;
  }

  HRESULT __stdcall GetSample::FindPin(LPCWSTR Id, IPin** ppPin) { // See QueryID
    if (!Id) {
      dssRPT0(dssERROR, "GetSample::FindPin(Id, ppPin) ** E_POINTER **\n");
      return E_POINTER;
    }

    if (!ppPin) {
      dssRPT1(dssERROR, "GetSample::FindPin(%ls, ppPin) ** E_POINTER **\n", Id);
      return E_POINTER;
    }

    if (wcscmp(L"GetSample01", Id)) {
      dssRPT1(dssERROR, "GetSample::FindPin(%ls, ppPin) ** VFW_E_NOT_FOUND **\n", Id);
      *ppPin = NULL;
      return VFW_E_NOT_FOUND;
    }

    dssRPT1(dssCMD, "GetSample::FindPin(%ls, ppPin)\n", Id);

    *ppPin = static_cast<IPin*>(this);

    AddRef();

    return S_OK;
  }

  HRESULT __stdcall GetSample::QueryFilterInfo(FILTER_INFO* pInfo) {
    if (!pInfo) {
      dssRPT0(dssERROR, "GetSample::QueryFilterInfo() ** E_POINTER **\n");
      return E_POINTER;
    }
    dssRPT0(dssCMD, "GetSample::QueryFilterInfo()\n");
    wcsncpy(pInfo->achName, L"GetSample", MAX_FILTER_NAME);
    pInfo->pGraph = filter_graph;
    if (filter_graph) filter_graph->AddRef();
    return S_OK;
  }

  HRESULT __stdcall GetSample::JoinFilterGraph(IFilterGraph* pGraph, LPCWSTR pName) {
    dssRPT2(dssCMD, "GetSample::JoinFilterGraph(0x%08p, %ls)\n", pGraph, pName);
    filter_graph = pGraph;
    return S_OK;
  }

  HRESULT __stdcall GetSample::QueryVendorInfo(LPWSTR* pVendorInfo) {
    dssRPT0(dssCMD, "GetSample::QueryVendorInfo() E_NOTIMPL\n");
    return E_NOTIMPL;
  }

  // IPin

  HRESULT __stdcall GetSample::Connect(IPin* pReceivePin, const AM_MEDIA_TYPE* pmt) {
    dssRPT0(dssERROR, "GetSample::Connect() E_UNEXPECTED\n");
    return E_UNEXPECTED;
  }

  HRESULT __stdcall GetSample::ReceiveConnection(IPin* pConnector, const AM_MEDIA_TYPE* pmt) {

    if (!pConnector || !pmt) {
      dssRPT0(dssERROR, "GetSample::ReceiveConnection() ** E_POINTER **\n");
      return E_POINTER;
    }

    if (state != State_Stopped) {
      dssRPT0(dssERROR, "GetSample::ReceiveConnection() ** VFW_E_NOT_STOPPED **\n");
      return VFW_E_NOT_STOPPED;
    }

    if (source_pin) {
      dssRPT0(dssERROR, "GetSample::ReceiveConnection() ** VFW_E_ALREADY_CONNECTED **\n");
      return VFW_E_ALREADY_CONNECTED;
    }

    VideoInfo tmp = vi;
    bool bInvertFrames = false;

    if (GetSample::InternalQueryAccept(pmt, tmp, bInvertFrames) != S_OK) {
      dssRPT0(dssERROR, "GetSample::ReceiveConnection() ** VFW_E_TYPE_NOT_ACCEPTED **\n");
      return VFW_E_TYPE_NOT_ACCEPTED;
    }

    dssRPT1(dssCMD, "GetSample::ReceiveConnection(0x%08p, pmt)\n", pConnector);
    vi = tmp;
    m_bInvertFrames = bInvertFrames;
    source_pin = pConnector;
    if (am_media_type)
      DeleteMediaType(am_media_type);
    am_media_type = CreateMediaType(pmt);
    return S_OK;
  }

  HRESULT __stdcall GetSample::Disconnect() {
    if (state != State_Stopped) {
      dssRPT0(dssERROR, "GetSample::Disconnect() ** VFW_E_NOT_STOPPED **\n");
      return VFW_E_NOT_STOPPED;
    }
    if (!source_pin) {
      dssRPT0(dssCMD, "GetSample::Disconnect() ** S_FALSE **\n");
      return S_FALSE;
    }
    source_pin = 0;
    if (am_media_type)
      DeleteMediaType(am_media_type);
    am_media_type = 0;

    if (Allocator) {
      dssRPT1(dssNEW, "Releasing Allocator 0x%08p.\n", Allocator);
      Allocator->Release();
      Allocator = 0;
    }

    dssRPT0(dssCMD, "GetSample::Disconnect()\n");
    return S_OK;
  }

  HRESULT __stdcall GetSample::ConnectedTo(IPin** ppPin) {
    if (!ppPin) {
      dssRPT0(dssERROR, "GetSample::ConnectedTo() ** E_POINTER **\n");
      return E_POINTER;
    }
    *ppPin = source_pin;
    if (!source_pin) {
      dssRPT0(dssERROR, "GetSample::ConnectedTo() ** VFW_E_NOT_CONNECTED **\n");
      return VFW_E_NOT_CONNECTED;
    }
    source_pin->AddRef();
    dssRPT1(dssCMD, "GetSample::ConnectedTo() is 0x%08p\n", source_pin);
    return S_OK;
  }

  HRESULT __stdcall GetSample::ConnectionMediaType(AM_MEDIA_TYPE* pmt) {
    if (!pmt) {
      dssRPT0(dssERROR, "GetSample::ConnectionMediaType() ** E_POINTER **\n");
      return E_POINTER;
    }
    if (!source_pin || !am_media_type) {
      dssRPT0(dssERROR, "GetSample::ConnectionMediaType() ** VFW_E_NOT_CONNECTED **\n");
      return VFW_E_NOT_CONNECTED;
    }
    FreeMediaType(*pmt);
    CopyMediaType(pmt, am_media_type);
    dssRPT0(dssCMD, "GetSample::ConnectionMediaType()\n");

    return S_OK;
  }

  HRESULT __stdcall GetSample::QueryPinInfo(PIN_INFO* pInfo) {
    if (!pInfo) {
      dssRPT0(dssERROR, "GetSample::QueryPinInfo() ** E_POINTER **\n");
      return E_POINTER;
    }
    pInfo->pFilter = static_cast<IBaseFilter*>(this);
    AddRef();
    pInfo->dir = PINDIR_INPUT;
    wcsncpy(pInfo->achName, L"GetSample", MAX_PIN_NAME);
    dssRPT1(dssCMD, "GetSample::QueryPinInfo() 0x%08p\n", this);
    return S_OK;
  }

  HRESULT __stdcall GetSample::QueryDirection(PIN_DIRECTION* pPinDir) {
    if (!pPinDir) {
      dssRPT0(dssERROR, "GetSample::QueryDirection() ** E_POINTER **\n");
      return E_POINTER;
    }
    *pPinDir = PINDIR_INPUT;
    dssRPT0(dssCMD, "GetSample::QueryDirection()\n");
    return S_OK;
  }

  HRESULT __stdcall GetSample::QueryId(LPWSTR* Id) { // See FindPin
    if (!Id) {
      dssRPT0(dssERROR, "GetSample::QueryId() ** E_POINTER **\n");
      return E_POINTER;
    }
    // CoTaskMemAlloc() fix from Dean Pavlekovic (dpavlekovic) May 2008
    const WCHAR name[] = L"GetSample01";
    const DWORD nameSize = sizeof(name);
    *Id = (LPWSTR) CoTaskMemAlloc(nameSize);
    if (*Id == NULL) {
       dssRPT0(dssERROR, "GetSample::QueryId() ** E_OUTOFMEMORY **\n");
       return E_OUTOFMEMORY;
    }
    memcpy(*Id, name, nameSize);
    dssRPT0(dssCMD, "GetSample::QueryId()\n");
    return S_OK;
  }

  HRESULT __stdcall GetSample::QueryAccept(const AM_MEDIA_TYPE* pmt) {
    VideoInfo tmp = vi;
    bool bInvertFrames = false;

    HRESULT result = InternalQueryAccept(pmt, tmp, bInvertFrames);

    if (result == S_OK) {
      vi = tmp;
      m_bInvertFrames = bInvertFrames;
    }

    return result;
  }

  HRESULT GetSample::InternalQueryAccept(const AM_MEDIA_TYPE* pmt, VideoInfo &vi, bool &bInvertFrames) {
    if (!pmt) {
      dssRPT0(dssERROR, "GetSample::QueryAccept() ** E_POINTER **\n");
      return E_POINTER;
    }

    if      (pmt->majortype == MEDIATYPE_Video) {
      dssRPT1(dssNEG, "GetSample::QueryAccept(%s) MEDIATYPE_Video\n", streamName);
      if (!load_video) {
        dssRPT2(dssNEG,  "*** Video: Subtype - '%s' %s\n", PrintFourCC(pmt->subtype.Data1), PrintGUID(&pmt->subtype));
        dssRPT1(dssNEG,  "*** Video: Format type - %s\n", PrintGUID(&pmt->formattype));
        return S_FALSE;
      }
    }
    else if (pmt->majortype == MEDIATYPE_Audio) {
      dssRPT1(dssNEG, "GetSample::QueryAccept(%s) MEDIATYPE_Audio\n", streamName);
      if (!load_audio) {
        dssRPT1(dssNEG,  "*** Audio: Subtype - %s\n", PrintGUID(&pmt->subtype));
        dssRPT1(dssNEG,  "*** Audio: Format type - %s\n", PrintGUID(&pmt->formattype));
        return S_FALSE;
      }
    }
    else {
      dssRPT3(dssNEG, "GetSample::QueryAccept(%s) reject major type '%s' %s\n",
              streamName, PrintFourCC(pmt->majortype.Data1), PrintGUID(&pmt->majortype));
      dssRPT1(dssNEG,  "*** Subtype - %s\n", PrintGUID(&pmt->subtype));
      dssRPT1(dssNEG,  "*** Format type - %s\n", PrintGUID(&pmt->formattype));
      return S_FALSE;
    }

// Handle audio:
/*
Audio: WAVE_FORMAT_EXTENSIBLE 48000Hz 6ch 6912Kbps

AM_MEDIA_TYPE:
majortype: MEDIATYPE_Audio {73647561-0000-0010-8000-00AA00389B71}
subtype: MEDIASUBTYPE_PCM {00000001-0000-0010-8000-00AA00389B71}
formattype: FORMAT_WaveFormatEx {05589F81-C356-11CE-BF01-00AA0055595A}
bFixedSizeSamples: 1
bTemporalCompression: 0
lSampleSize: 0
cbFormat: 40

WAVEFORMATEX:
wFormatTag: WAVE_FORMAT_EXTENSIBLE = 0xfffe
nChannels: 6
nSamplesPerSec: 48000
nAvgBytesPerSec: 864000
nBlockAlign: 18
wBitsPerSample: 24
cbSize: 22 (extra bytes)

WAVEFORMATEXTENSIBLE:
wValidBitsPerSample: 24
dwChannelMask: 0x0000003f
SubFormat: KSDATAFORMAT_SUBTYPE_PCM {00000001-0000-0010-8000-00AA00389B71}

pbFormat:
0000: fe ff 06 00 80 bb 00 00 00 2f 0d 00 12 00 18 00 ........./......
0010: 16 00 18 00 3f 00 00 00 01 00 00 00 00 00 10 00 ....?...........
0020: 80 00 00 aa 00 38 9b 71
*/
    if (pmt->majortype == MEDIATYPE_Audio) {

      if (pmt->subtype != MEDIASUBTYPE_PCM
       && pmt->subtype != MEDIASUBTYPE_IEEE_FLOAT
       && pmt->subtype != MEDIASUBTYPE_extensible ) {
        dssRPT1(dssNEG,  "*** Audio: Subtype rejected - %s\n", PrintGUID(&pmt->subtype));
        return S_FALSE;
      }

      dssRPT1(dssNEG,  "*** Audio: Subtype accepted - %s\n", PrintGUID(&pmt->subtype));

      if (pmt->formattype != FORMAT_WaveFormatEx) {
        dssRPT1(dssNEG,  "*** Audio: Not FORMAT_WaveFormatEx - %s\n", PrintGUID(&pmt->formattype));
        return S_FALSE;
      }
      if (pmt->cbFormat < sizeof(WAVEFORMATEX)) {
        dssRPT2(dssNEG,  "*** Audio: AM_MEDIA_TYPE.cbFormat to small - %d of %zd\n",
                         pmt->cbFormat, sizeof(WAVEFORMATEX));
        return S_FALSE;
      }
      WAVEFORMATEX* wex = (WAVEFORMATEX*)pmt->pbFormat;

      if ((wex->wFormatTag != WAVE_FORMAT_PCM) &&
          (wex->wFormatTag != WAVE_FORMAT_IEEE_FLOAT) &&
          (wex->wFormatTag != WAVE_FORMAT_EXTENSIBLE)) {
        dssRPT1(dssNEG,  "*** Audio: Unsupported format - WAVEFORMATEX.wFormatTag=0x%04x\n", wex->wFormatTag);
        return S_FALSE;
      }

      int sample_type = 0;

      if (wex->wFormatTag == WAVE_FORMAT_IEEE_FLOAT) {
        sample_type = SAMPLE_FLOAT;
      }
      else {
        switch (wex->wBitsPerSample) {
          case  8: sample_type = SAMPLE_INT8;  break;
          case 16: sample_type = SAMPLE_INT16; break;
          case 24: sample_type = SAMPLE_INT24; break;
          case 32: sample_type = SAMPLE_INT32; break;
          default:
            dssRPT1(dssNEG,  "*** Audio: Unsupported number of bits per sample: %d\n", wex->wBitsPerSample);
            return S_FALSE;
        }

        if (wex->wFormatTag == WAVE_FORMAT_EXTENSIBLE) {
          if (pmt->cbFormat < sizeof(WAVEFORMATEXTENSIBLE)) {
            dssRPT2(dssNEG,  "*** Audio: AM_MEDIA_TYPE.cbFormat to small - %d of %zd\n",
                             pmt->cbFormat, sizeof(WAVEFORMATEXTENSIBLE));
            return S_FALSE;
          }
          if (wex->cbSize < 22) {
            dssRPT1(dssNEG,  "*** Audio: Extended wave format structure wrong size: %d of 22\n", wex->cbSize);
            return S_FALSE;
          }
          // Override settings with extended data (float or >2 ch).
          WAVEFORMATEXTENSIBLE* wext =  (WAVEFORMATEXTENSIBLE*)pmt->pbFormat;

          if (wext->Samples.wValidBitsPerSample != wext->Format.wBitsPerSample) {
            dssRPT2(dssNEG,  "*** Audio: Warning ValidBitsPerSample(%d) != BitsPerSample(%d)!\n",
                     wext->Samples.wValidBitsPerSample, wext->Format.wBitsPerSample);
//          return S_FALSE; // accept the data here - a postprocessing filter can repair it later
          }

          if (wext->SubFormat == SUBTYPE_IEEE_AVSFLOAT) {  // We have float audio.
            sample_type = SAMPLE_FLOAT;
          } else if (wext->SubFormat != SUBTYPE_IEEE_AVSPCM) {
            dssRPT1(dssNEG,  "*** Audio: Extended WAVE format must be Float or PCM. %s\n", PrintGUID(&wext->SubFormat));
            return S_FALSE;
          }
        }
      }

      if (lockvi) {
        if ( (vi.audio_samples_per_second != (int)wex->nSamplesPerSec)
          || (vi.nchannels                != wex->nChannels)
          || (vi.sample_type              != sample_type) ) {
          dssRPT4(dssNEG,  "*** Audio: Reject format change! Channels:%d. Samples/sec:%d. Bits/sample:%d. Type:%s\n",
                wex->nChannels, wex->nSamplesPerSec, wex->wBitsPerSample, PrintAudioType(sample_type));
          return S_FALSE;
        }
      }

      vi.audio_samples_per_second = wex->nSamplesPerSec;
      vi.nchannels = wex->nChannels;
      vi.sample_type = sample_type;

      dssRPT4(dssNEG,  "*** Audio: Accepted! Channels:%d. Samples/sec:%d. Bits/sample:%d. Type:%s\n",
            wex->nChannels, wex->nSamplesPerSec, wex->wBitsPerSample, PrintAudioType(sample_type));
      return S_OK;
    }

// Handle video:

    if (pmt->majortype == MEDIATYPE_Video) {
      int pixel_type = 0;
      bool invert_if_height_is_negative = false; // For RGB24,RGB32,ARGB32.
      bInvertFrames = false;
      if        (pmt->subtype == MEDIASUBTYPE_YV12) {
        if (!(media & mediaYV12)) {
          dssRPT0(dssNEG,  "*** Video: Subtype denied - YV12\n");
          return S_FALSE;
        }
        pixel_type = VideoInfo::CS_YV12;

      } else if (pmt->subtype == MEDIASUBTYPE_I420) {
        if (!(media & mediaI420)) {
          dssRPT0(dssNEG,  "*** Video: Subtype denied - I420\n");
          return S_FALSE;
        }
        pixel_type = VideoInfo::CS_I420;

      } else if (pmt->subtype == MEDIASUBTYPE_NV12) {
        if (!(media & mediaNV12)) {
          dssRPT0(dssNEG,  "*** Video: Subtype denied - NV12\n");
          return S_FALSE;
        }
        pixel_type = VideoInfo::CS_YV12;

      } else if (pmt->subtype == MEDIASUBTYPE_YV16) {
        if (!(media & mediaYV16)) {
          dssRPT0(dssNEG,  "*** Video: Subtype denied - YV16\n");
          return S_FALSE;
        }
        pixel_type = VideoInfo::CS_YV16;

      } else if (pmt->subtype == MEDIASUBTYPE_YV24) {
        if (!(media & mediaYV24)) {
          dssRPT0(dssNEG,  "*** Video: Subtype denied - YV24\n");
          return S_FALSE;
        }
        pixel_type = VideoInfo::CS_YV24;

      } else if (pmt->subtype == MEDIASUBTYPE_Y411) {
        if (!(media & mediaY411)) {
          dssRPT0(dssNEG,  "*** Video: Subtype denied - Y411\n");
          return S_FALSE;
        }
        pixel_type = VideoInfo::CS_YV411;

      } else if (pmt->subtype == MEDIASUBTYPE_Y41P) {
        if (!(media & mediaY41P)) {
          dssRPT0(dssNEG,  "*** Video: Subtype denied - Y41P\n");
          return S_FALSE;
        }
        pixel_type = VideoInfo::CS_YV411;

      } else if (pmt->subtype == MEDIASUBTYPE_AYUV) {
        if (!(media & mediaAYUV)) {
          dssRPT0(dssNEG,  "*** Video: Subtype denied - AYUV\n");
          return S_FALSE;
        }
        pixel_type = VideoInfo::CS_YV24;

      } else if (pmt->subtype == MEDIASUBTYPE_YUY2) {
        if (!(media & mediaYUY2)) {
          dssRPT0(dssNEG,  "*** Video: Subtype denied - YUY2\n");
          return S_FALSE;
        }
        pixel_type = VideoInfo::CS_YUY2;

      } else if (pmt->subtype == MEDIASUBTYPE_RGB24) {
        if (!(media & mediaRGB24)) {
          dssRPT0(dssNEG,  "*** Video: Subtype denied - RGB24\n");
          return S_FALSE;
        }
        pixel_type = VideoInfo::CS_BGR24;
        invert_if_height_is_negative = true;

      } else if (pmt->subtype == MEDIASUBTYPE_RGB32) {
        if (!(media & mediaRGB32)) {
          dssRPT0(dssNEG,  "*** Video: Subtype denied - RGB32\n");
          return S_FALSE;
        }
        pixel_type = VideoInfo::CS_BGR32;
        invert_if_height_is_negative = true;

      } else if (pmt->subtype == MEDIASUBTYPE_ARGB32) {
        if (!(media & mediaARGB)) {
          dssRPT0(dssNEG,  "*** Video: Subtype denied - ARGB32\n");
          return S_FALSE;
        }
        pixel_type = VideoInfo::CS_BGR32;
        invert_if_height_is_negative = true;

      } else if (pmt->subtype == MEDIASUBTYPE_BGR48) {
        if (!(media & mediaRGB48)) {
          dssRPT0(dssNEG,  "*** Video: Subtype denied - BGR[48]\n");
          return S_FALSE;
        }
        pixel_type = VideoInfo::CS_BGR48;
        bInvertFrames = true; // BGR[48] is top-down, CS_BGR48 is bottom-up

      } else if (pmt->subtype == MEDIASUBTYPE_BRA64) {
        if (!(media & mediaRGB64)) {
          dssRPT0(dssNEG,  "*** Video: Subtype denied - BRA[64]\n");
          return S_FALSE;
        }
        pixel_type = VideoInfo::CS_BGR64;
        bInvertFrames = true; // BRA[64] is top-down, CS_BGR64 is bottom-up

//      } else if (pmt->subtype == MEDIASUBTYPE_Y8) {
//        TODO: Y8

      } else {
        dssRPT2(dssNEG,  "*** Video: Subtype rejected - '%s' %s\n", PrintFourCC(pmt->subtype.Data1), PrintGUID(&pmt->subtype));
        dssRPT1(dssNEG,  "*** Video: Format type - %s\n", PrintGUID(&pmt->formattype));
        return S_FALSE;
      }

      dssRPT2(dssNEG,  "*** Video: Subtype accepted - '%s' %s\n", PrintFourCC(pmt->subtype.Data1), PrintGUID(&pmt->subtype));

      BITMAPINFOHEADER* pbi;
      unsigned _avg_time_per_frame;

      if (pmt->formattype == FORMAT_VideoInfo) {
        VIDEOINFOHEADER* vih = (VIDEOINFOHEADER*)pmt->pbFormat;
        _avg_time_per_frame = unsigned(vih->AvgTimePerFrame);
        pbi = &vih->bmiHeader;
      }
      else if (pmt->formattype == FORMAT_VideoInfo2) {
        VIDEOINFOHEADER2* vih2 = (VIDEOINFOHEADER2*)pmt->pbFormat;
        _avg_time_per_frame = unsigned(vih2->AvgTimePerFrame);
        pbi = &vih2->bmiHeader;
//      if (vih2->dwInterlaceFlags & AMINTERLACE_1FieldPerSample) {
//        vi.SetFieldBased(true);
//      }
      }
      else {
        dssRPT1(dssNEG,  "*** Video: Format type rejected - %s\n", PrintGUID(&pmt->formattype));
        return S_FALSE;
      }

      dssRPT1(dssNEG,  "*** Video: Format type accepted - %s\n", PrintGUID(&pmt->formattype));

      if (lockvi) {
        if ( (vi.pixel_type != pixel_type)
          || (vi.width      != pbi->biWidth)
          || (vi.height     != ((pbi->biHeight < 0) ? -pbi->biHeight : pbi->biHeight)) ) {
          dssRPT3(dssNEG,  "*** Video: reject format change: %dx%d, pixel_type %s\n",
                  pbi->biWidth, pbi->biHeight, PrintPixelType(pixel_type));
          return S_FALSE;
        }
      }

      vi.pixel_type = pixel_type;
      vi.width = pbi->biWidth;
      vi.height = (pbi->biHeight < 0) ? -pbi->biHeight : pbi->biHeight;

      if (pbi->biHeight < 0 && invert_if_height_is_negative) {
        bInvertFrames = true;
      }

      if (_avg_time_per_frame) {
        vi.SetFPS(10000000, _avg_time_per_frame);
      } else {
        vi.fps_numerator = 1;
        vi.fps_denominator = 0;
      }

      dssRPT4(dssNEG,  "*** Video: format accepted: %dx%d, pixel_type %s, avg_time_per_frame %dx100ns\n",
              vi.width, vi.height, PrintPixelType(vi.pixel_type), _avg_time_per_frame);
      dssRPT3(dssNEG,  "*** Video: bFixedSizeSamples=%d, bTemporalCompression=%d, lSampleSize=%d\n",
            pmt->bFixedSizeSamples, pmt->bTemporalCompression, pmt->lSampleSize);

      return S_OK;
    }
    return S_FALSE;
  }

  HRESULT __stdcall GetSample::EnumMediaTypes(IEnumMediaTypes** ppEnum) {
    if (!ppEnum) {
      dssRPT0(dssERROR, "GetSample::EnumMediaTypes() ** E_POINTER **\n");
      return E_POINTER;
    }
    if (no_my_media_types == 0) {
      dssRPT0(dssCMD, "GetSample::EnumMediaTypes() ** E_NOTIMPL **\n");
      return E_NOTIMPL;
    }
    dssRPT0(dssCMD, "GetSample::EnumMediaTypes()\n");
    *ppEnum = new GetSampleEnumMediaTypes(this, no_my_media_types);
    return *ppEnum ? S_OK : E_OUTOFMEMORY;
  }

  HRESULT __stdcall GetSample::QueryInternalConnections(IPin** apPin, ULONG* nPin) {
    if (!nPin) {
      dssRPT0(dssERROR, "GetSample::QueryInternalConnections() ** E_POINTER **\n");
      return E_POINTER;
    }
    *nPin = 0;
    dssRPT0(dssCMD, "GetSample::QueryInternalConnections()\n");
    return S_OK;
  }

  HRESULT __stdcall GetSample::EndOfStream() {
    dssRPT1((dssSAMP|dssCMD), "GetSample::EndOfStream() (%s)\n", streamName);
    end_of_stream = true;
    if (filter_graph) {
      IMediaEventSink* mes = NULL;
      try {
        if (SUCCEEDED(filter_graph->QueryInterface(&mes))) {
          mes->Notify(EC_COMPLETE, (LONG_PTR)S_OK, (LONG_PTR)static_cast<IBaseFilter*>(this));
          mes->Release();
        }
      }
      catch (...) {
        dssRPT0((dssERROR|dssCMD), "GetSample::EndOfStream() Unknown Exception!\n");
        if (mes) mes->Release();
      }
    }
    dssRPT0(dssCMD, "EndOfStream() indicating new sample ready\n");
    SetEvent(evtNewSampleReady);
    return S_OK;
  }

  HRESULT __stdcall GetSample::BeginFlush() {
    dssRPT1(dssCMD, "GetSample::BeginFlush() (%s)\n", streamName);
    flushing = true;
    SetEvent(evtDoneWithSample); // Free task if waiting
    graphTimeout = true;
    return S_OK;
  }

  HRESULT __stdcall GetSample::EndFlush() {
    dssRPT1(dssCMD, "GetSample::EndFlush() (%s)\n", streamName);
    ResetEvent(evtDoneWithSample);  // Nuke any unused SetEvents
    flushing = false;
    return S_OK;
  }

  HRESULT __stdcall GetSample::NewSegment(REFERENCE_TIME tStart, REFERENCE_TIME tStop, double dRate) {
    dssRPT4(dssSAMP, "GetSample::NewSegment(%I64d, %I64d, %f) (%s)\n", tStart, tStop, dRate, streamName);
    segment_start_time = tStart;
    segment_stop_time  = tStop;
    time_of_last_frame = int(sample_end_time - sample_start_time);
    sample_end_time = sample_start_time = 0;
    return S_OK;
  }

  // IMemInputPin

  HRESULT __stdcall GetSample::GetAllocator(IMemAllocator** ppAllocator) {
    if (!ppAllocator) {
      dssRPT0(dssCMD, "GetSample::GetAllocator() E_POINTER\n");
      return E_POINTER;
    }
    if (!Allocator) {
      dssRPT0(dssCMD, "GetSample::GetAllocator() VFW_E_NO_ALLOCATOR\n");
      return VFW_E_NO_ALLOCATOR;
    }
    dssRPT1(dssCMD, "GetSample::GetAllocator(0x%08p)\n", Allocator);
    Allocator->AddRef();
    *ppAllocator = Allocator;
    return S_OK;
  }

  HRESULT __stdcall GetSample::NotifyAllocator(IMemAllocator* pAllocator, BOOL bReadOnly) {
    dssRPT4(dssCMD, "GetSample::NotifyAllocator(0x%08p, %x) was 0x%08p (%s)\n", pAllocator, bReadOnly, Allocator, streamName);
    if (!pAllocator) {
      dssRPT0(dssCMD, "GetSample::NotifyAllocator() E_POINTER\n");
      return E_POINTER;
    }
    if (Allocator) {
      Allocator->Release();
    }
    Allocator = pAllocator;
    Allocator->AddRef();
    return S_OK;
  }

  HRESULT __stdcall GetSample::GetAllocatorRequirements(ALLOCATOR_PROPERTIES* pProps) {
    if (!pProps) {
      dssRPT0(dssERROR, "GetSample::GetAllocatorRequirements(*pProps) E_POINTER\n");
      return E_POINTER;
    }
    dssRPT4(dssCMD, "GetSample::GetAllocatorRequirements(%d, %d, %d, %d) E_NOTIMPL\n",
            pProps->cBuffers, pProps->cbBuffer, pProps->cbAlign, pProps->cbPrefix);
    return E_NOTIMPL;
  }

  HRESULT __stdcall GetSample::Receive(IMediaSample* pSamples) {
    if (seeking) {
      dssRPT1(dssSAMP, "Receive: discarding sample (seeking) (%s)\n", streamName);
      return S_OK;
    }
    if (S_OK == pSamples->IsPreroll()) {
      dssRPT1(dssSAMP, "Receive: discarding sample (preroll) (%s)\n", streamName);
      return S_OK;
    }
    if (flushing) {
      dssRPT1(dssSAMP, "Receive: discarding sample (flushing) (%s)\n", streamName);
      return S_FALSE;
    }
    if (state == State_Stopped) {
      dssRPT1(dssSAMP, "Receive: discarding sample (State_Stopped) (%s)\n", streamName);
      return VFW_E_WRONG_STATE;
    }

    pSamples->GetPointer(&av_buffer);
    int deltaT = avg_time_per_frame;

    av_sample_bytes = pSamples->GetActualDataLength();
    if (load_audio) {  // audio
      deltaT = MulDiv(av_sample_bytes, 10000000, vi.BytesPerAudioSample()*vi.SamplesPerSecond());
      dssRPT1(dssSAMP, "Receive: Got %d bytes of audio data.\n",av_sample_bytes);
    }

    HRESULT result = pSamples->GetTime(&sample_start_time, &sample_end_time);
    if (result == VFW_S_NO_STOP_TIME) {
      dssRPT0(dssINFO, "VFW_S_NO_STOP_TIME!\n");
      sample_end_time = sample_start_time + deltaT; // wing it
    }
    else if (FAILED(result)) {
      dssRPT0(dssINFO, "GetTime failed!\n");
      sample_start_time += deltaT; // wing it
      sample_end_time = sample_start_time + deltaT;
    }
    // Yes +1! Some brain dead decoders think an acceptable implementation is
    // STOP_TIME = START_TIME + 1, others get it just plain wrong i.e. Negative etc,
    else if (sample_end_time <= sample_start_time+1) {
      dssRPT1(dssINFO, "Got bogus stop time! %I64d\n", sample_end_time);
      sample_end_time = sample_start_time + deltaT;
    }
    dssRPT4(dssSAMP, "Receive: %s sample time span x100ns %I64d to %I64d (%d)\n", streamName,
            sample_start_time, sample_end_time, DWORD(sample_end_time - sample_start_time));
    HRESULT wait_result;
    SetEvent(evtNewSampleReady);  // New sample is finished - wait releasing it
                                  // until it has been fetched (DoneWithSample).
    do {
      dssRPT1(dssWAIT, "...Receive() waiting for DoneWithSample. (%s)\n", streamName);
      if (log && log->mask && log->file) fflush(log->file);
      wait_result = WaitForSingleObject(evtDoneWithSample, 15000);
    } while ((wait_result == WAIT_TIMEOUT) && (state != State_Stopped));

    av_buffer = 0;
    av_sample_bytes = 0;

    dssRPT1(dssINFO, "Receive() - returning. (%s)\n", streamName);

    return S_OK;
  }

  HRESULT __stdcall GetSample::ReceiveMultiple(IMediaSample** ppSamples, long nSamples, long* nSamplesProcessed) {
    dssRPT0(dssCMD, "GetSample::ReceiveMultiple()\n");
    for (int i=0; i<nSamples; ++i) {
      HRESULT hr = Receive(ppSamples[i]);
      if (FAILED(hr)) {
        *nSamplesProcessed = i;
        return hr;
      }
    }
    *nSamplesProcessed = nSamples;
    return S_OK;
  }

  HRESULT __stdcall GetSample::ReceiveCanBlock() {
    dssRPT0(dssCMD, "GetSample::ReceiveCanBlock()\n");
    return S_OK;
  }



/***********************************************
 *             GetSampleEnumPins               *
 ***********************************************/



GetSampleEnumPins::GetSampleEnumPins(GetSample* _parent, int _pos)
 : parent(_parent), log(_parent->log) {
  dssRPT0(dssNEW, "New GetSampleEnumPins.\n");
  pos=_pos;
  refcnt = 1;
}

GetSampleEnumPins::~GetSampleEnumPins() {
  dssRPT0(dssNEW, "~GetSampleEnumPins.\n");
}

HRESULT __stdcall GetSampleEnumPins::Next(ULONG cPins, IPin** ppPins, ULONG* pcFetched) {
  if (!ppPins || !pcFetched) return E_POINTER;
  int copy = *pcFetched = min(int(cPins), 1-pos);
  if (copy>0) {
    *ppPins = static_cast<IPin*>(parent);
    parent->AddRef();
  }
  pos += copy;
  return int(cPins) > copy ? S_FALSE : S_OK;
}


/***********************************************
 *          GetSampleEnumMediaTypes            *
 ***********************************************/



GetSampleEnumMediaTypes::GetSampleEnumMediaTypes(GetSample* _parent, unsigned _count, unsigned _pos)
 : parent(_parent), log(_parent->log) {

  dssRPT0(dssNEW, "New GetSampleEnumMediaTypes.\n");

  pos = 0;
  count = _count;
  refcnt = 1;
}

GetSampleEnumMediaTypes::~GetSampleEnumMediaTypes() {
  dssRPT0(dssNEW, "~GetSampleEnumMediaTypes.\n");
}

HRESULT __stdcall GetSampleEnumMediaTypes::Next(ULONG cMediaTypes, AM_MEDIA_TYPE** ppMediaTypes, ULONG* pcFetched) {
  if (!ppMediaTypes) {
    dssRPT0(dssERROR, "GetSampleEnumMediaTypes::Next(ppMediaTypes) E_POINTER\n");
    return E_POINTER;
  }
  if (!pcFetched && (cMediaTypes != 1)) {
    dssRPT0(dssERROR, "GetSampleEnumMediaTypes::Next(pcFetched) E_POINTER\n");
    return E_POINTER;
  }
  dssRPT2(dssCMD, "GetSampleEnumMediaTypes::Next(%u) pos=%u\n", cMediaTypes, pos);
  unsigned long copy = min(cMediaTypes, (unsigned long)(count-pos));
  if (pcFetched) *pcFetched = copy;
  while (copy-- > 0) {
    *ppMediaTypes++ = CreateMediaType(parent->GetMediaType(pos++)) ;
  }
  return (pos >= count) ? S_FALSE : S_OK;
}

HRESULT __stdcall GetSampleEnumMediaTypes::Skip(ULONG cMediaTypes) {
  dssRPT2(dssCMD, "GetSampleEnumMediaTypes::Skip(%u) pos=%u\n", cMediaTypes, pos);
  pos += cMediaTypes;
  if (pos >= count) {
    pos = count;
    return S_FALSE;
  }
  return S_OK;
}

HRESULT __stdcall GetSampleEnumMediaTypes::Reset() {
  pos=0;
  return S_OK;
}

HRESULT __stdcall GetSampleEnumMediaTypes::Clone(IEnumMediaTypes** ppEnum) {
  if (!ppEnum) {
    dssRPT0(dssERROR, "GetSampleEnumMediaTypes::Clone() E_POINTER\n");
    return E_POINTER;
  }
  dssRPT0(dssCMD, "GetSampleEnumMediaTypes::Clone()\n");
  *ppEnum = new GetSampleEnumMediaTypes(parent, count, pos);
  return *ppEnum ? S_OK : E_OUTOFMEMORY;
}


/***********************************************
 *    DirectShowSource Helper Functions.       *
 ***********************************************/



static bool HasNoConnectedOutputPins(IBaseFilter* bf) {
  IEnumPins* ep;
  if (FAILED(bf->EnumPins(&ep)))
    return true;
  ULONG fetched=1;
  IPin* pin;
  while (S_OK == ep->Next(1, &pin, &fetched)) {
    PIN_DIRECTION dir;
    pin->QueryDirection(&dir);
    if (dir == PINDIR_OUTPUT) {
      IPin* other;
      pin->ConnectedTo(&other);
      if (other) {
        other->Release();
        pin->Release();
        ep->Release();
        return false;
      }
    }
    pin->Release();
  }
  ep->Release();
  return true;
}



static void DisconnectAllPinsAndRemoveFilter(IGraphBuilder* gb, IBaseFilter* bf) {
  IEnumPins* ep;
  if (SUCCEEDED(bf->EnumPins(&ep))) {
    ULONG fetched=1;
    IPin* pin;
    while (S_OK == ep->Next(1, &pin, &fetched)) {
      IPin* other;
      pin->ConnectedTo(&other);
      if (other) {
        gb->Disconnect(other);
        gb->Disconnect(pin);
        other->Release();
      }
      pin->Release();
    }
    ep->Release();
  }
  gb->RemoveFilter(bf);
}


static void RemoveUselessFilters(IGraphBuilder* gb, IBaseFilter* not_this_one, IBaseFilter* nor_this_one) {
  IEnumFilters* ef;
  if (FAILED(gb->EnumFilters(&ef)))
    return;
  ULONG fetched=1;
  IBaseFilter* bf;
  while (S_OK == ef->Next(1, &bf, &fetched)) {
    if (bf != not_this_one && bf != nor_this_one) {
      if (HasNoConnectedOutputPins(bf)) {
        DisconnectAllPinsAndRemoveFilter(gb, bf);
        ef->Reset();
      }
    }
    bf->Release();
  }
  ef->Release();
}

static HRESULT AttemptConnectFilters(IGraphBuilder* gb, IBaseFilter* connect_filter) {
  IEnumFilters* ef;

  if (FAILED(gb->EnumFilters(&ef)))
    return E_UNEXPECTED;

  HRESULT hr;
  ULONG fetched=1;
  IBaseFilter* bf;
  IEnumPins* ep_conn;

  connect_filter->EnumPins(&ep_conn);
  IPin* p_conn;

  if (FAILED(ep_conn->Next(1, &p_conn, &fetched)))
    return E_UNEXPECTED;

  while (S_OK ==(ef->Next(1, &bf, &fetched))) {
    if (bf != connect_filter) {
      IEnumPins* ep;

      bf->EnumPins(&ep);
      IPin* pPin;
      while (S_OK == (ep->Next(1, &pPin, &fetched)))  {

        PIN_DIRECTION PinDirThis;
        pPin->QueryDirection(&PinDirThis);

        if (PinDirThis == PINDIR_OUTPUT) {
          hr = gb->ConnectDirect(pPin, p_conn, NULL);
          if (SUCCEEDED(hr)) {
            pPin->Release();
            ep->Release();
            bf->Release();
            ep_conn->Release();
            ef->Release();
            return S_OK;
          }
        }
      }
      pPin->Release();
      ep->Release();
    }
    bf->Release();
  }
  ep_conn->Release();
  ef->Release();
  return S_OK;
}

void DirectShowSource::SetMicrosoftDVtoFullResolution(IGraphBuilder* gb) {
  // Microsoft's DV codec defaults to half-resolution, to everyone's
  // great annoyance.  This will set it to full res if possible.
  // Note that IIPDVDec is not declared in older versions of
  // strmif.h; you may need the Win2000 platform SDK.
  IEnumFilters* ef;
  if (FAILED(gb->EnumFilters(&ef)))
    return;
  ULONG fetched=1;
  IBaseFilter* bf;
  while (S_OK == ef->Next(1, &bf, &fetched)) {
    IIPDVDec* pDVDec;
    if (SUCCEEDED(bf->QueryInterface(&pDVDec))) {
      dssRPT1(dssINFO, "DVtoFullResolution() pDVDec=0x%08p\n", pDVDec);
      pDVDec->put_IPDisplay(DVRESOLUTION_FULL); // DVDECODERRESOLUTION_720x480);   // yes, this includes 720x576
      pDVDec->Release();
    }
    bf->Release();
  }
  ef->Release();

}

// The following constant is from "wmcodecconst.h" in the
// "Windows Media Audio and Video Codec Interfaces download package"
// available for download from MSDN.
static const WCHAR *g_wszWMVCDecoderDeinterlacing = L"_DECODERDEINTERLACING";

void DirectShowSource::DisableDeinterlacing(IFilterGraph *pGraph)
{
  IEnumFilters *pEnum = NULL;
  IBaseFilter *pFilter;
  ULONG cFetched;

  HRESULT hr = pGraph->EnumFilters(&pEnum);
  if (FAILED(hr))
    return;

  while(pEnum->Next(1, &pFilter, &cFetched) == S_OK) {
    FILTER_INFO FilterInfo;

    hr = pFilter->QueryFilterInfo(&FilterInfo);
    if (SUCCEEDED(hr)) {
      if (wcscmp(FilterInfo.achName, L"WMVideo Decoder DMO") == 0) {
        IPropertyBag *pPropertyBag = NULL;

        hr = pFilter->QueryInterface(IID_IPropertyBag, (void**)&pPropertyBag);
        if(SUCCEEDED(hr)) {
          dssRPT1(dssINFO, "DisableDeinterlacing() pPropertyBag=0x%08p\n", pPropertyBag);
          VARIANT myVar;

          VariantInit(&myVar);
          // Disable decoder deinterlacing
          myVar.vt   = VT_BOOL;
          myVar.lVal = FALSE;
          pPropertyBag->Write(g_wszWMVCDecoderDeinterlacing, &myVar);

          pPropertyBag->Release();
        }
        else {
          dssRPT2(dssINFO, "DisableDeinterlacing() pFilter=0x%08p code=%X\n", pFilter, hr);
        }
      }
      // The FILTER_INFO structure holds a pointer to the Filter Graph
      // Manager, with a reference count that must be released.
      if (FilterInfo.pGraph != NULL)
        FilterInfo.pGraph->Release();
    }
    pFilter->Release();
  }
  pEnum->Release();
}


static const WCHAR *g_wszWMACHiResOutput = L"_HIRESOUTPUT";

void DirectShowSource::SetWMAudioDecoderDMOtoHiResOutput(IFilterGraph *pGraph)
{
  IEnumFilters *pEnum = NULL;
  IBaseFilter *pFilter;
  ULONG cFetched;

  HRESULT hr = pGraph->EnumFilters(&pEnum);
  if (FAILED(hr))
    return;

  // Search graph for "WMAudio Decoder DMO"
  while(pEnum->Next(1, &pFilter, &cFetched) == S_OK) {
    FILTER_INFO FilterInfo;

    hr = pFilter->QueryFilterInfo(&FilterInfo);
    if (SUCCEEDED(hr)) {
      if (wcscmp(FilterInfo.achName, L"WMAudio Decoder DMO") == 0) {
        IPropertyBag *pPropertyBag = NULL;

        hr = pFilter->QueryInterface(IID_IPropertyBag, (void**)&pPropertyBag);
        if(SUCCEEDED(hr)) {
          dssRPT1(dssINFO, "WMAudioDecoderDMOtoHiRes() pPropertyBag=0x%08p\n", pPropertyBag);
          VARIANT myVar;

          VariantInit(&myVar);
          // Enable full output capabilities
          myVar.vt      = VT_BOOL;
          myVar.boolVal = -1; // True
          pPropertyBag->Write(g_wszWMACHiResOutput, &myVar);

          pPropertyBag->Release();

          IEnumPins* ep;
          if (SUCCEEDED(pFilter->EnumPins(&ep))) {
            ULONG fetched=1;
            IPin* pin;
            // Search for output pin
            while (S_OK == ep->Next(1, &pin, &fetched)) {
              PIN_DIRECTION dir;
              pin->QueryDirection(&dir);
              if (dir == PINDIR_OUTPUT) {
                // Reconnect output pin
                hr = pGraph->Reconnect(pin);
                if(FAILED(hr)) {
                  dssRPT1(dssINFO, "WMAudioDecoderDMOtoHiRes() Reconnect failed, code = %X\n", hr);
                }
              }
              pin->Release();
            }
            ep->Release();
          }
        }
        else {
          dssRPT2(dssINFO, "WMAudioDecoderDMOtoHiRes() pFilter=0x%08p code=%X\n", pFilter, hr);
        }
      }
      // The FILTER_INFO structure holds a pointer to the Filter Graph
      // Manager, with a reference count that must be released.
      if (FilterInfo.pGraph != NULL)
        FilterInfo.pGraph->Release();
    }
    pFilter->Release();
  }
  pEnum->Release();
}


/************************************************
 *               DirectShowSource               *
 ***********************************************/


DirectShowSource::DirectShowSource(const char* filename, int _avg_time_per_frame, int _seekmode,
                                   bool _enable_audio, bool _enable_video, bool _convert_fps, unsigned _media,
                                   int _timeout, int _frames, LOG* _log, IScriptEnvironment* env)
  : get_sample(_enable_audio, _enable_video, _media, _log), seekmode(_seekmode), convert_fps(_convert_fps),
    gb(NULL), currentFrame(0), TrapTimeouts(_timeout < 0), WaitTimeout(abs(_timeout)), log(_log) {

  dssRPT0(dssNEW, "New DirectShowSource.\n");

  IMediaFilter*  mf = 0;
  IMediaSeeking* ms = 0;

  try {
    CheckHresult(env, CoCreateInstance(CLSID_FilterGraphNoThread, 0, CLSCTX_INPROC_SERVER, IID_IGraphBuilder, (void**)&gb),
                 "couldn't create filter graph");

    WCHAR filenameW[MAX_PATH];
    MultiByteToWideChar(CP_ACP, 0, filename, -1, filenameW, MAX_PATH);

    CheckHresult(env, gb->AddFilter(static_cast<IBaseFilter*>(&get_sample), L"GetSample"), "couldn't add GetSample filter");

    int fnlen = lstrlen(filename);
    bool load_grf = (fnlen >= 4) ? !lstrcmpi(filename+fnlen-4,".grf") : false;  // Detect ".GRF" extension and load as graph if so.

    if (load_grf) {
      CheckHresult(env, LoadGraphFile(gb, filenameW), "Couldn't open GRF file.", filename);
      // Try connecting to any open pins.
      AttemptConnectFilters(gb, &get_sample);
      if (!get_sample.IsConnected()) {
        if (_enable_video)
          env->ThrowError("DirectShowSource: GRF file does not have a compatible open video pin.\n"
                          "Graph must have 1 output pin that will bid RGB24, RGB32, ARGB, YUY2,\n"
                          "YV12, I420, NV12, YV16, YV24, Y41P, Y411 or AYUV");
        else
          env->ThrowError("DirectShowSource: GRF file does not have a compatible open audio pin.\n"
                          "Graph must have 1 output pin that will bid 8, 16, 24 or 32 bit PCM or IEEE Float.");
      }
    } else {
      HRESULT RFHresult = gb->RenderFile(filenameW, NULL);
      if (!get_sample.IsConnected()) { // Ignore arbitary errors, run with what we got
        CheckHresult(env, RFHresult, "couldn't open file ", filename);
        env->ThrowError("DirectShowSource: RenderFile, the filter graph manager won't talk to me");
      }
    }

    RemoveUselessFilters(gb, &get_sample, &get_sample);

    if (_enable_video) {
      SetMicrosoftDVtoFullResolution(gb);
      DisableDeinterlacing(gb);
    }

    if (_enable_audio) {
      SetWMAudioDecoderDMOtoHiResOutput(gb);
    }

    // Prevent the graph from trying to run in "real time"
    // ... Disabled because it breaks ASF.  Now I know why
    // Avery swears so much.
    CheckHresult(env, gb->QueryInterface(&mf), "couldn't get IMediaFilter interface");
    CheckHresult(env, mf->SetSyncSource(NULL), "couldn't set null sync source");
    SAFE_RELEASE(mf);

    CheckHresult(env, gb->QueryInterface(&ms), "couldn't get IMediaSeeking interface");

    CheckHresult(env, get_sample.StartGraph(gb), "DirectShowSource : Graph refused to run.");

    if (TrapTimeouts) {
      DWORD timeout = 2000;   // 2 seconds
      get_sample.WaitForStart(timeout);
    }
    else {
      DWORD timeout = clamp(WaitTimeout, 5000ul, 300000ul);   // 5 seconds to 5 minutes
      if (get_sample.WaitForStart(timeout))
        // If returning grey frames, trap on init!
        env->ThrowError("DirectShowSource : Timeout waiting for graph to start.");
    }

    vi = get_sample.GetVideoInfo();

    if (vi.HasVideo()) {
      __int64 frame_count = 0, duration = 0;
      GUID time_fmt = {0, 0, 0, {0, 0, 0, 0, 0, 0, 0, 0}};

      // SetTimeFormat can give E_NOTIMPL
      ms->SetTimeFormat(&TIME_FORMAT_MEDIA_TIME);
      // so check what it currently is
      ms->GetTimeFormat(&time_fmt);

      if (time_fmt == TIME_FORMAT_MEDIA_TIME) ms->GetDuration(&duration);

      if (duration <= 0) duration = get_sample.segment_stop_time-get_sample.segment_start_time;

      // run in frame mode if we can set time format to frame
      frame_units = SUCCEEDED(ms->SetTimeFormat(&TIME_FORMAT_FRAME));

      if (frame_units) ms->GetDuration(&frame_count);

      dssRPT2((dssNEG|dssCALL), "Directshow duration %I64d, frame_count %I64d.\n", duration, frame_count);

      if (convert_fps || frame_count <= 0) frame_units = false;

      if (duration <= 0 && !frame_units && !_frames) {
        env->ThrowError("DirectShowSource: unable to determine the duration of the video.");
      }

      if (_avg_time_per_frame) { // User specified FPS
        get_sample.avg_time_per_frame = _avg_time_per_frame;
        vi.SetFPS(10000000, _avg_time_per_frame);
        vi.num_frames = int(frame_units ? frame_count : (duration + (_avg_time_per_frame-1)) / _avg_time_per_frame); // Ceil()
      }
      else {
        // this is exact (no rounding needed) because of the way the fps is set in GetSample
        get_sample.avg_time_per_frame = 10000000 / vi.fps_numerator * vi.fps_denominator; // Floor()

        if (get_sample.avg_time_per_frame != 0) {
          // We have all the info
          vi.num_frames = int(frame_units ? frame_count :
                              (duration + (get_sample.avg_time_per_frame-1)) / get_sample.avg_time_per_frame); // Ceil()
        }
        else {
          // Try duration divided by frame count
          if (frame_count > 0 && duration > 0) {
            get_sample.avg_time_per_frame = int((duration + (frame_count>>1)) / frame_count); // Round()
            vi.num_frames = int(frame_count);

            unsigned __int64 numerator   = 10000000 * frame_count;
            unsigned __int64 denominator = duration;

            unsigned __int64 x=numerator, y=denominator;
            while (y) {   // find gcd
              unsigned __int64 t = x%y; x = y; y = t;
            }
            numerator   /= x; // normalize
            denominator /= x;

            unsigned __int64 temp = numerator | denominator; // Just looking for top bit
            unsigned u = 0;
            while (temp & 0xffffffff80000000) {
              temp = Int64ShrlMod32(temp, 1);
              u++;
            }
            if (u) { // Scale to fit
              const unsigned round = 1 << (u-1);
              vi.SetFPS( (unsigned)Int64ShrlMod32(numerator   + round, u),
                         (unsigned)Int64ShrlMod32(denominator + round, u) );
            }
            else {
              vi.fps_numerator   = (unsigned)numerator;
              vi.fps_denominator = (unsigned)denominator;
            }
          }
          // Try duration of first frame
          else {
            switch (get_sample.time_of_last_frame) {
              case 160000: case 170000:
                vi.fps_numerator   = 60000;
                vi.fps_denominator = 1001;
                get_sample.avg_time_per_frame = 166833;
                break;

              case 200000:
                vi.fps_numerator   = 50;
                vi.fps_denominator = 1;
                get_sample.avg_time_per_frame = 200000;
                break;

              case 330000: case 340000:
                vi.fps_numerator   = 30000;
                vi.fps_denominator = 1001;
                get_sample.avg_time_per_frame = 333667;
                break;

              case 400000:
                vi.fps_numerator   = 25;
                vi.fps_denominator = 1;
                get_sample.avg_time_per_frame = 400000;
                break;

              case 410000: case 420000:
                vi.fps_numerator   = 24000;
                vi.fps_denominator = 1001;
                get_sample.avg_time_per_frame = 417083;
                break;

              default:
                env->ThrowError("DirectShowSource: I can't determine the frame rate\n"
                                "of the video, you must use the \"fps\" parameter."); // Note must match message below
            }
            vi.num_frames = int(frame_units ? frame_count :
               (duration*vi.fps_numerator + vi.fps_denominator*10000000i64 - 1) / (vi.fps_denominator*10000000i64) ); // Ceil()
          }
        }
      }
      if (_frames) vi.num_frames = _frames;

      dssRPT4((dssNEG|dssCALL), "New Video: %dx%d, frame_count=%d, pixel type=%s.\n",
                                vi.width, vi.height, vi.num_frames, PrintPixelType(vi.pixel_type));
    }

    if (vi.HasAudio()) {
      GUID time_fmt = {0, 0, 0, {0, 0, 0, 0, 0, 0, 0, 0}};
      __int64 audio_dur = 0;

      if (_avg_time_per_frame && _frames) { // User specified FPS + FrameCount
        audio_dur = Int32x32To64(_avg_time_per_frame, _frames);
      }
      else {
        // SetTimeFormat can give E_NOTIMPL
        ms->SetTimeFormat(&TIME_FORMAT_MEDIA_TIME);
        // so check what it currently is
        ms->GetTimeFormat(&time_fmt);

        if (time_fmt == TIME_FORMAT_MEDIA_TIME) ms->GetDuration(&audio_dur);

        if (audio_dur == 0) audio_dur = get_sample.segment_stop_time-get_sample.segment_start_time;

        if (audio_dur == 0) {
            env->ThrowError("DirectShowSource: unable to determine the duration of the audio.\n"
                            "Manually specify FPS and Framecount. Duration = Framecount / FPS");
        }
      }

      vi.num_audio_samples = (audio_dur * vi.audio_samples_per_second + 9999999) / 10000000; // Ceil()

      dssRPT2((dssNEG|dssCALL), "New Audio: audio_dur %I64dx100ns, samples %I64d.\n",
                                audio_dur, vi.num_audio_samples);
    }
    SAFE_RELEASE(ms);

    cur_frame = 0;
    audio_bytes_read = 0;
    next_sample = 0;
  }
  catch (...) {
    SAFE_RELEASE(mf);
    SAFE_RELEASE(ms);
    cleanUp();
    throw;
  }
}


  DirectShowSource::~DirectShowSource() {
    dssRPT0(dssNEW, "~DirectShowSource.\n");

    cleanUp();
  }

  void DirectShowSource::cleanUp() {
    if (gb) {
      IMediaControl* mc = NULL;
      try {
      if (SUCCEEDED(gb->QueryInterface(&mc))) {
        OAFilterState st;
        mc->GetState(1000, &st);
        if (st != State_Stopped) mc->Stop();
        mc->Release();
      }
      get_sample.StopGraph(gb);
      SAFE_RELEASE(gb);
      }
      catch (...) {
        dssRPT0(dssERROR, "cleanup Unknown Exception!\n");
        if (mc) mc->Release();
      }
    }
  }

#if 0
  PVideoFrame __stdcall DirectShowSource::GetFrame(int n, IScriptEnvironment* env) {
    DWORD timeout = WaitTimeout;
    n = max(min(n, vi.num_frames-1), 0);

    // Ask for the frame whose [start_time ->T<- end_time] spans sample_time
    const __int64 sample_time = (n == 0) ? 0 : Int32x32To64(n, get_sample.avg_time_per_frame) + (get_sample.avg_time_per_frame>>1);

    dssRPT2(dssCALL, "GetFrame: Frame %d time %I64dx100ns.\n", n, sample_time);

    if ( (seekmode == 0 && n >= cur_frame) || (seekmode == 2) || (n >= cur_frame && n <= cur_frame+10) ) {
      // seekzero==true+forwards or seek==false or a short hop forwards
      if (convert_fps) {
        // automatic fps conversion: trust only sample time
        while (get_sample.GetSampleEndTime() <= sample_time) {
          if(!get_sample.NextSample(timeout)) break;
        }
        cur_frame = n;
      }
      else {
        while (cur_frame < n) {
          if (!get_sample.NextSample(timeout)) break;
          cur_frame++;
        }
      }
    }
    else {
      HRESULT hr;
      if (seekmode == 0) {
        // Seekzero=true and stepping back
        hr = get_sample.SeekTo(0);
        if (hr == S_OK) hr = S_FALSE;
      }
      else {
        // Seek=true and stepping back or a long hop forwards
        hr = get_sample.SeekTo(sample_time);
      }

      if (hr == S_OK) {
        // seek ok!
        cur_frame = n;
      }
      else if (hr == S_FALSE) {
        // seekzero or seek failed!
        if (!get_sample.WaitForStart(timeout)) {
          // We have stopped and started the graph. Many unseekable streams
          // reset to 0, others don't move. Try to get our position
          cur_frame = int(get_sample.GetSampleStartTime() / get_sample.avg_time_per_frame);
          if (frame_units) {
            while (cur_frame < n) {
              if (!get_sample.NextSample(timeout)) break;
              cur_frame++;
            }
          }
          else {
            while (get_sample.GetSampleEndTime() <= sample_time) {
              if(!get_sample.NextSample(timeout)) break;
            }
            cur_frame = n;
          }
        }
      }
      else {
        env->ThrowError("DirectShowSource : The video Graph failed to restart after seeking. Status = 0x%x", hr);
      }
    }
    dssRPT2(dssCALL, "GetFrame: Frame time span x100ns %I64d to %I64d\n",
            get_sample.GetSampleStartTime(), get_sample.GetSampleEndTime());

    return get_sample.GetCurrentFrame(env, n, TrapTimeouts, timeout);
  }

#else

  PVideoFrame __stdcall DirectShowSource::GetFrame(int n, IScriptEnvironment* env) {
    DWORD timeout = WaitTimeout;

    n = max(min(n, vi.num_frames-1), 0);
    // Ask for the frame whose start_time == T
    const __int64 sample_time = Int32x32To64(n, get_sample.avg_time_per_frame);

    dssRPT2(dssCALL, "GetFrame: Frame %d start time %I64dx100ns.\n", n, sample_time);

    HRESULT hr = S_OK;
    switch (seekmode) {
      case 0: // Seekzero
        if (n < cur_frame) {
          hr = get_sample.SeekTo(0, gb);  // stepping back
          cur_frame = 0;
        }
        break;

      case 1: // Seek
        if (n < cur_frame) {
          hr = get_sample.SeekTo(sample_time, gb);
          cur_frame = n;
        }
        else if (convert_fps) {
          if (sample_time > get_sample.GetSampleStartTime() + get_sample.avg_time_per_frame*30) {
            hr = get_sample.SeekTo(sample_time, gb);
            cur_frame = n;
          }
        }
        else {
          if (n > cur_frame+30) {
            hr = get_sample.SeekTo(sample_time, gb);
            cur_frame = n;
          }
        }
        break;

      case 2: // No_Search
        break;

      default:
        env->ThrowError("DirectShowSource : Invalid seek mode %d", seekmode);
    }
    if (FAILED(hr))
      env->ThrowError("DirectShowSource : The video Graph failed to restart after seeking. Status = 0x%x", hr);

    if (convert_fps) {
      while (get_sample.GetSampleStartTime() <= (sample_time+5000) || !currentFrame) { // Allow 0.5 millisecond roundup
        sampleStartTime = get_sample.GetSampleStartTime();
        cur_frame = int(sampleStartTime / get_sample.avg_time_per_frame); // Floor()
        currentFrame = get_sample.GetCurrentFrame(env, n, TrapTimeouts, timeout);
        if(!get_sample.NextSample(timeout)) break;
      }
      dssRPT3(dssCALL, "GetFrame: VFR Frame %d time span x100ns %I64d to %I64d\n", n,
              sampleStartTime, get_sample.GetSampleStartTime());
    }
    else {
      while (cur_frame < n) {
        if (!get_sample.NextSample(timeout)) break;
        cur_frame++;
      }
      currentFrame = get_sample.GetCurrentFrame(env, n, TrapTimeouts, timeout);
      dssRPT3(dssCALL, "GetFrame: CFR Frame %d time span x100ns %I64d to %I64d\n", n,
              get_sample.GetSampleStartTime(), get_sample.GetSampleEndTime());
    }

    return currentFrame;
  }
#endif


  void __stdcall DirectShowSource::GetAudio(void* buf, __int64 start, __int64 count, IScriptEnvironment* env) {
    DWORD timeout = WaitTimeout;
    int bytes_filled = 0;

    if (next_sample != start) {  // We have been searching!  Skip until sync!

      dssRPT2(dssCALL, "GetAudio: Seeking to %I64d previous was %I64d samples.\n", start, next_sample);

      // Backup to begining of current buffer
      next_sample -= vi.AudioSamplesFromBytes(audio_bytes_read);
      audio_bytes_read = 0;

      const __int64 avail_samples = vi.AudioSamplesFromBytes(get_sample.av_sample_bytes);
      if ( ((seekmode != 2) && (start < next_sample))
        // Seek=true and Seekzero=true and stepping back
        || ((seekmode == 1) && (start >= next_sample+avail_samples+50000))) {
        // Seek=true and a long hop forwards
        const __int64 seekTo = (seekmode == 0) ? 0 : (start*10000000 + (vi.audio_samples_per_second>>1)) / vi.audio_samples_per_second; // Round()
        dssRPT1(dssCALL, "GetAudio: SeekTo %I64dx100ns media time.\n", seekTo);

        HRESULT hr = get_sample.SeekTo(seekTo, gb);

        if (seekmode == 0 && hr == S_OK) hr = S_FALSE;

        if (hr == S_OK) {
          // Seek succeeded!
          next_sample = start;
          audio_bytes_read = 0;
        }
        else if (hr == S_FALSE) {
          // seek failed!
          if (!get_sample.WaitForStart(timeout)) {
            // We have stopped and started the graph many unseekable streams
            // reset to 0, others don't move. Try to get our position
            next_sample = (get_sample.GetSampleStartTime() * vi.audio_samples_per_second + 5000000) / 10000000; // Round()
          }
        }
        else {
          env->ThrowError("DirectShowSource : The audio Graph failed to restart after seeking. Status = 0x%x", hr);
        }
      }

      if (start < next_sample) { // We are behind sync - pad with 0
        const int fill_nsamples  = (int)min(next_sample - start, count);
        dssRPT1(dssCALL, "GetAudio: Padding %d samples.\n", fill_nsamples);

        // We cannot seek.
        if (vi.sample_type == SAMPLE_FLOAT) {
          float* samps = (float*)buf;
          for (int i = 0; i < fill_nsamples; i++)
            samps[i] = 0.0f;
        }
        else {
          memset(buf,0, (unsigned int)vi.BytesFromAudioSamples(fill_nsamples));
        }

        if (fill_nsamples == count)  // Buffer is filled - return
          return;
        start += fill_nsamples;
        count -= fill_nsamples;
        bytes_filled += (int)vi.BytesFromAudioSamples(fill_nsamples);
      }

      if (start > next_sample) {  // Skip forward (decode)
        // Should we search?
        int skip_left = (int)vi.BytesFromAudioSamples(start - next_sample);
        dssRPT1(dssCALL, "GetAudio: Skipping %d bytes.\n", skip_left);

        if (get_sample.WaitForStart(timeout))
          if (TrapTimeouts)
            env->ThrowError("DirectShowSource : Timeout waiting for audio.");

        while (skip_left > 0) {
          if (get_sample.av_sample_bytes-audio_bytes_read >= skip_left) {
            audio_bytes_read += skip_left;
            break;
          }
          skip_left -= get_sample.av_sample_bytes-audio_bytes_read;
          audio_bytes_read = get_sample.av_sample_bytes;

          if (get_sample.NextSample(timeout))
            audio_bytes_read = 0;
          else
             break;  // EndOfStream? Timeout?
        } // end while
        next_sample = start;
      }
    }

    BYTE* samples = (BYTE*)buf;
    int bytes_left = (int)vi.BytesFromAudioSamples(count);
    dssRPT2(dssCALL, "GetAudio: Reading %I64d samples, %d bytes.\n", count, bytes_left);

    if (get_sample.WaitForStart(timeout))
      if (TrapTimeouts)
        env->ThrowError("DirectShowSource : Timeout waiting for audio.");
    while (bytes_left) {
      // Can we read from the Directshow filters buffer?
      if (get_sample.av_sample_bytes - audio_bytes_read > 0) { // Copy as many bytes as needed.

        // This many bytes can be safely read.
        const int available_bytes = min(bytes_left, get_sample.av_sample_bytes - audio_bytes_read);
        dssRPT2(dssCALL, "GetAudio: Memcpy %d offset, %d bytes.\n", bytes_filled, available_bytes);

        memcpy(&samples[bytes_filled], &get_sample.av_buffer[audio_bytes_read], available_bytes);

        bytes_left -= available_bytes;
        bytes_filled += available_bytes;
        audio_bytes_read += available_bytes;

      }
      else { // Read more samples
        if (get_sample.NextSample(timeout)) {
          audio_bytes_read = 0;
        }
        else { // Pad with 0
          if (TrapTimeouts)
            if (get_sample.WaitForStart(timeout))
              env->ThrowError("DirectShowSource : Timeout waiting for audio.");

          dssRPT2(dssCALL, "GetAudio: Memset %d offset, %d bytes.\n", bytes_filled, bytes_left);
          if (vi.sample_type == SAMPLE_FLOAT) {
            float* samps = (float*)(&samples[bytes_filled]); // Aligned just to be sure
            const int samples_left = (bytes_left+sizeof(float)-1)/sizeof(float);
            for (int i = 0; i < samples_left; i++)
              samps[i] = 0.0f;
          } else {
            memset(&samples[bytes_filled],0,bytes_left);
          }
          bytes_left = 0;
          break;
        }
      }
    }
    next_sample +=count;
  }



void DirectShowSource::CheckHresult(IScriptEnvironment* env, HRESULT hr, const char* msg, const char* msg2) {
  if (SUCCEEDED(hr)) return;
//  char buf[1024] = {0};
//  if (!FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM, NULL, hr, 0, buf, 1024, NULL))
  char buf[MAX_ERROR_TEXT_LEN+1] = {0};
  if (!AMGetErrorText(hr, buf, MAX_ERROR_TEXT_LEN))
    wsprintf(buf, "error code 0x%x", hr);
  env->ThrowError("DirectShowSource: %s%s:\n%s", msg, msg2, buf);
}

int __stdcall DirectShowSource::SetCacheHints(int cachehints,int frame_range)
{
  switch(cachehints)
  {
  case CACHE_GET_MTMODE:
    return MT_SERIALIZED;
  default:
    return 0;
  }
}

HRESULT DirectShowSource::LoadGraphFile(IGraphBuilder *pGraph, const WCHAR* wszName)
{
    IStorage *pStorage = 0;
    if (S_OK != StgIsStorageFile(wszName)) return E_FAIL;

    HRESULT hr = StgOpenStorage(wszName, 0,
        STGM_TRANSACTED | STGM_READ | STGM_SHARE_DENY_WRITE,
        0, 0, &pStorage);
    if (FAILED(hr)) return hr;

    IPersistStream *pPersistStream = 0;
    hr = pGraph->QueryInterface(IID_IPersistStream,
             reinterpret_cast<void**>(&pPersistStream));
    if (SUCCEEDED(hr))
    {
        IStream *pStream = 0;
        hr = pStorage->OpenStream(L"ActiveMovieGraph", 0,
            STGM_READ | STGM_SHARE_EXCLUSIVE, 0, &pStream);
        if(SUCCEEDED(hr))
        {
            hr = pPersistStream->Load(pStream);
            pStream->Release();
        }
        pPersistStream->Release();
    }
    pStorage->Release();
    return hr;
}


/* As this is currently implemented we use two separate instance of DSS, one for video and
 * one for audio. This means we create two (2) filter graphs. An alternate implementation
 * would be to have a video GetSample::IPin object and a separate audio GetSample::IPin object
 * in the one filter graph. Possible problems with this idea could be related to independant
 * positioning of the Video and Audio streams within the one filter graph. */


AVSValue __cdecl Create_DirectShowSource(AVSValue args, void*, IScriptEnvironment* env) {

  const char* filename = args[0].AsString();
  const int _avg_time_per_frame = args[1].Defined() ? int(10000000 / args[1].AsFloat() + 0.5) : 0;

  const bool audio    = args[3].AsBool(true);
  const bool video    = args[4].AsBool(true);

  if (!(audio || video))
    env->ThrowError("DirectShowSource: Both video and audio are disabled!");

  const bool seek     = args[2].AsBool(true);
  const bool seekzero = args[6].AsBool(false);
  const int  seekmode = seek ? (seekzero ? 0 : 1) : 2; // 0=seek_zero, 1=seek, 2=no_search

  const int _timeout  = args[7].AsInt(60000); // Default timeout = 1 minute

  unsigned _media = GetSample::mediaNONE;
  if (args[8].Defined()) {
    bool mediaPad = false;
    const char* pixel_type = args[8].AsString();
    if (pixel_type[0] == '+') {
      pixel_type += 1;
      mediaPad = true;
    }
    if      (!lstrcmpi(pixel_type, "YUY2"))  { _media = GetSample::mediaYUY2; }
    else if (!lstrcmpi(pixel_type, "YV12"))  { _media = GetSample::mediaYV12; }
    else if (!lstrcmpi(pixel_type, "I420"))  { _media = GetSample::mediaI420; }
//  else if (!lstrcmpi(pixel_type, "YUV9"))  { _media = GetSample::mediaYUV9; }
    else if (!lstrcmpi(pixel_type, "Y41P"))  { _media = GetSample::mediaY41P; }
    else if (!lstrcmpi(pixel_type, "Y411"))  { _media = GetSample::mediaY411; }
    else if (!lstrcmpi(pixel_type, "AYUV"))  { _media = GetSample::mediaAYUV; }
    else if (!lstrcmpi(pixel_type, "RGB24")) { _media = GetSample::mediaRGB24; }
    else if (!lstrcmpi(pixel_type, "RGB32")) { _media = GetSample::mediaRGB32 | GetSample::mediaARGB; }
    else if (!lstrcmpi(pixel_type, "RGB48")) { _media = GetSample::mediaRGB48; }
    else if (!lstrcmpi(pixel_type, "RGB64")) { _media = GetSample::mediaRGB64; }
    else if (!lstrcmpi(pixel_type, "ARGB"))  { _media = GetSample::mediaARGB; }
    else if (!lstrcmpi(pixel_type, "RGB"))   { _media = GetSample::mediaRGB; }
    else if (!lstrcmpi(pixel_type, "YUV"))   { _media = GetSample::mediaYUV; }
    else if (!lstrcmpi(pixel_type, "AUTO"))  { _media = GetSample::mediaAUTO; }
    else if (!lstrcmpi(pixel_type, "NV12"))  { _media = GetSample::mediaNV12; }
    else if (!lstrcmpi(pixel_type, "YV24"))  { _media = GetSample::mediaYV24; }
    else if (!lstrcmpi(pixel_type, "YV16"))  { _media = GetSample::mediaYV16; }
    else if (!lstrcmpi(pixel_type, "YUVEX")) { _media = GetSample::mediaYUVex; }
    else if (!lstrcmpi(pixel_type, "FULL"))  { _media = GetSample::mediaFULL; }
    else {
      env->ThrowError("DirectShowSource: pixel_type must be \"RGB24\", \"RGB32\", \"ARGB\", \"RGB48\", \"RGB64\", "
                      "\"YUY2\", \"YV12\", \"I420\", \"YV16\", \"YV24\", \"AYUV\", \"Y41P\", "
                      "\"Y411\", \"NV12\", \"RGB\", \"YUV\" , \"YUVex\", \"AUTO\"  or \"FULL\"");
    }
    if (mediaPad) _media |= GetSample::mediaPAD;
  }
  const int _frames = args[9].AsInt(0);

  LOG* log = NULL;

  if (args[10].Defined()) {
    log = new LOG(args[10].AsString(), args[11].AsInt(dssNEG | dssSAMP | dssERROR), env);
    if (!log) env->ThrowError("DirectShowSource: No memory for Log.");
  }

  if (!(audio && video)) { // Hey - simple!!
    if (audio) {
      return new DirectShowSource(filename, _avg_time_per_frame, seekmode, true , false,
                                  args[5].AsBool(false), _media, _timeout, _frames, log, env);
    } else {
      return new DirectShowSource(filename, _avg_time_per_frame, seekmode, false , true,
                                  args[5].AsBool(false), _media, _timeout, _frames, log, env);
    }
  }

  PClip DS_audio;
  PClip DS_video;

  bool audio_success = true;
  bool video_success = true;

  if (log) log->AddRef();
  try {
    int fnlen = lstrlen(filename);
    if ((fnlen >= 4) && !lstrcmpi(filename+fnlen-4,".grf")) {
      env->ThrowError("DirectShowSource: Only 1 stream supported for .GRF files, one of Audio or Video must be disabled.");
    }

    const char *a_e_msg = "";
    const char *v_e_msg = "";

    try {
      DS_audio = new DirectShowSource(filename, _avg_time_per_frame, seekmode, true , false,
                                      args[5].AsBool(false), _media, _timeout, _frames, log, env);
    } catch (const AvisynthError &e) {
      a_e_msg = e.msg;
      audio_success = false;
    }

    try {
      DS_video = new DirectShowSource(filename, _avg_time_per_frame, seekmode, false, true,
                                      args[5].AsBool(false), _media, _timeout, _frames, log, env);
    } catch (const AvisynthError &e) {
      if (!lstrcmpi(e.msg, "DirectShowSource: I can't determine the frame rate\n"
                           "of the video, you must use the \"fps\" parameter.") ) { // Note must match message above
            env->ThrowError(e.msg);
        }
      v_e_msg = e.msg;
      video_success = false;
    }


    if (!(audio_success || video_success)) {
      env->ThrowError("DirectShowSource: Could not open as video or audio.\r\n\r\n"
                                        "Video returned:  \"%s\"\r\n\r\n"
                                        "Audio returned:  \"%s\"\r\n", v_e_msg, a_e_msg);
    }
  }
  catch (...) {
    if (log) log->DelRef("Create_DirectShowSource Cleanup Handler");
    throw;
  }
  if (log) log->DelRef("Create_DirectShowSource");

  if (!audio_success)
    return DS_video;

  if (!video_success)
    return DS_audio;

  AVSValue inv_args[2] = { DS_video, DS_audio };
  PClip ds_all =  env->Invoke("AudioDub",AVSValue(inv_args,2)).AsClip();

  return ds_all;
}

const AVS_Linkage *AVS_linkage = 0;

extern "C" __declspec(dllexport) const char* __stdcall AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors)
{
  AVS_linkage = vectors;

  env->AddFunction("DirectShowSource",
// args   0      1      2       3       4            5          6
       "s[fps]f[seek]b[audio]b[video]b[convertfps]b[seekzero]b"
//                 7            8            9        10        11
       "[timeout]i[pixel_type]s[framecount]i[logfile]s[logmask]i",
       Create_DirectShowSource, 0);

  return "DirectShowSource";
}
