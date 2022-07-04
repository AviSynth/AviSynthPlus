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

#include <avisynth.h>
#include "../../convert/convert.h"
#include "../../filters/transform.h"
#include "../../core/alignplanar.h"
#include "../../core/strings.h"
#include "AudioSource.h"
#include "VD_Audio.h"
#include "AVIReadHandler.h"
#include "avi_source.h"
#include <vfw.h>
#include <avs/minmax.h>
#ifdef INTEL_INTRINSICS
#include <emmintrin.h>
#include <tmmintrin.h>
#endif
#include <string>
#include <sstream>

#include "../../core/AviHelper.h"

static void __cdecl free_buffer(void* buff, IScriptEnvironment* env)
{
  if (buff)
    static_cast<IScriptEnvironment2*>(env)->Free(buff);
}

TemporalBuffer::TemporalBuffer(const VideoInfo& vi, bool bMediaPad,
  AVI_SpecialFormats specf,
  IScriptEnvironment* env)
{
  int heightY = vi.height;
  int heightUV = (vi.pixel_type & VideoInfo::CS_INTERLEAVED) ? 0 : heightY >> vi.GetPlaneHeightSubsampling(PLANAR_U);

  if (specf == AVI_SpecialFormats::Y410)
  { // Y410 packed 4444 U,Y,V,A
    // This format is a packed 10-bit representation that includes 2 bits of alpha.
    // Bits 0 - 9 contain the U sample, bits 10 - 19 contain the Y sample, bits 20 - 29 contain the V sample, 
    // and bits 30 - 31 contain the alpha value.
    // To indicate that a pixel is fully opaque, an application must set the two alpha bits equal to 0x03.
    pitchY = vi.width * sizeof(uint32_t) ; // n/a
    pitchUV = 0;
  }
  else if (specf == AVI_SpecialFormats::Y416)
  { // Y416 packed 4444 U,Y,V,A
    // image_size = vi->width * vi->height * 4 * sizeof(uint16_t);
    pitchY = vi.width * 4 * sizeof(uint16_t); // n/a
    pitchUV = 0;
  }
  else if (specf == AVI_SpecialFormats::v308)
  { // v308 packed 444
    // image_size = vi->width * vi->height * 3 * sizeof(uint8_t);
    pitchY = vi.width * 3; // n/a
    pitchUV = 0;
  }
  else if (specf == AVI_SpecialFormats::v408)
  { // v408 packed 4444
    // image_size = vi->width * vi->height * 4 * sizeof(uint8_t);
    pitchY = vi.width * 4; // n/a
    pitchUV = 0;
  }
  else if (specf == AVI_SpecialFormats::v410)
  { // v410 packed 444 U,Y,V
    // image_size = vi->width * vi->height * 4 (32 bit holds 3x10 bits);
    pitchY = vi.width * 4; // n/a
    pitchUV = 0;
  }
  else if (specf == AVI_SpecialFormats::v210) {
    pitchY = ((16 * ((vi.width + 5) / 6) + 127) & ~127);
    pitchUV = 0;
  }
  else if (specf == AVI_SpecialFormats::r210) {
    pitchY = ((vi.width + 63) / 64) * 256;
    pitchUV = 0;
  }
  else if (specf == AVI_SpecialFormats::R10k) {
    pitchY = vi.width * 4;
    pitchUV = 0;
  }
  else if (vi.IsPlanarRGB() || vi.IsPlanarRGBA()) {
    pitchY = vi.RowSize();
  }
  else {
    if (!bMediaPad) { // do not padding at all.
      pitchY = vi.RowSize();
      pitchUV = vi.RowSize(PLANAR_U);
    }
    else { // align 4bytes with minimum padding.
      pitchY = (vi.RowSize() + 3) & ~3;
      pitchUV = (vi.RowSize(PLANAR_U) + 3) & ~3;
    }
  }

  size_t sizeY = pitchY * heightY;
  size_t sizeUV = pitchUV * heightUV;
  if(specf == AVI_SpecialFormats::r210 || 
    specf == AVI_SpecialFormats::R10k || 
    specf == AVI_SpecialFormats::Y416 || 
    specf == AVI_SpecialFormats::v410 || 
    specf == AVI_SpecialFormats::v210 || 
    specf == AVI_SpecialFormats::v308 || 
    specf == AVI_SpecialFormats::v408 ||
    specf == AVI_SpecialFormats::Y410
  )
    size = sizeY;
  else if (vi.IsPlanarRGB())
    size = sizeY * 3;
  else if (vi.IsPlanarRGBA())
    size = sizeY * 4;
  else
    size = sizeY + 2 * sizeUV;

  // maybe memcpy become fast by aligned start address.
  orig = env->Allocate(size, FRAME_ALIGN, AVS_POOLED_ALLOC);
  if (!orig)
    env->ThrowError("AVISource: couldn't allocate temporal buffer.");
  env->AtExit(free_buffer, orig);

  pY = reinterpret_cast<uint8_t*>(orig);
  pA = nullptr;
  if (vi.IsPlanarRGB() || vi.IsPlanarRGBA())
  {
    // pY: G
    pU = pY + sizeY; // B
    pV = pU + sizeY; // R
    if (vi.IsPlanarRGBA())
      pA = pV + sizeY; // A
  } else if (vi.pixel_type & VideoInfo::CS_UPlaneFirst) {
    pU = pY + sizeY;
    pV = pU + sizeUV;
  } else {
    pV = pY + sizeY;
    pU = pV + sizeUV;
  }
}

// todo:
// // 10-bit+ planar RGB
/*
mmioFOURCC('G', '3', 00, 10); // ffmpeg GBRP10LE
mmioFOURCC('G', '3', 00, 12); // ffmpeg GBRP12LE
mmioFOURCC('G', '3', 00, 14); // ffmpeg GBRP14LE
mmioFOURCC('G', '3', 00, 16); // ffmpeg GBRP16LE
mmioFOURCC('G', '4', 00, 10); // ffmpeg GBRAP10LE
mmioFOURCC('G', '4', 00, 12); // ffmpeg GBRAP12LE
mmioFOURCC('G', '4', 00, 14); // ffmpeg GBRAP14LE
mmioFOURCC('G', '4', 00, 16); // ffmpeg GBRAP16LE
*/
static PVideoFrame AdjustFrameAlignment(TemporalBuffer* frame, const VideoInfo& vi, bool bInvertFrames,
  AVI_SpecialFormats specf,
  IScriptEnvironment* env)
{
    auto result = env->NewVideoFrame(vi);
    // fixme: this is source filter, add frame properties if possible
    BYTE* dstp = result->GetWritePtr();
    int pitch = result->GetPitch();
    int height = result->GetHeight();
    if (bInvertFrames) { // write from bottom to top
      dstp += pitch * (height - 1);
      pitch = -pitch;
    }

    switch(specf) {
    case AVI_SpecialFormats::v210:
      v210_to_yuv422p10(dstp, pitch, result->GetWritePtr(PLANAR_U), result->GetWritePtr(PLANAR_V), result->GetPitch(PLANAR_U),
        frame->GetPtr(), vi.width, vi.height);
      break;
    case AVI_SpecialFormats::P210:
    case AVI_SpecialFormats::P216:
    case AVI_SpecialFormats::P010:
    case AVI_SpecialFormats::P016:
        Px10_16_to_yuv42xp10_16(dstp, pitch, result->GetWritePtr(PLANAR_U), result->GetWritePtr(PLANAR_V), result->GetPitch(PLANAR_U),
          frame->GetPtr(), frame->GetPitch(), vi.width, vi.height, result->GetHeight(PLANAR_U),
          specf == AVI_SpecialFormats::P016 || specf == AVI_SpecialFormats::P216, env);
        break;
    case AVI_SpecialFormats::Y416:
      if(vi.pixel_type == VideoInfo::CS_YUVA444P16)
        FromY416_c<true>(dstp, pitch, result->GetWritePtr(PLANAR_U), result->GetWritePtr(PLANAR_V), result->GetPitch(PLANAR_U),
          result->GetWritePtr(PLANAR_A), result->GetPitch(PLANAR_A),
          frame->GetPtr(), frame->GetPitch(), vi.width, vi.height);
      else
        FromY416_c<false>(dstp, pitch, result->GetWritePtr(PLANAR_U), result->GetWritePtr(PLANAR_V), result->GetPitch(PLANAR_U),
          result->GetWritePtr(PLANAR_A), result->GetPitch(PLANAR_A),
          frame->GetPtr(), frame->GetPitch(), vi.width, vi.height);
      break;
    case AVI_SpecialFormats::r210:
      From_r210_c(result->GetWritePtr(PLANAR_R), result->GetWritePtr(PLANAR_G), result->GetWritePtr(PLANAR_B), result->GetPitch(PLANAR_G),
        frame->GetPtr(), frame->GetPitch(), vi.width, vi.height);
      break;
    case AVI_SpecialFormats::R10k:
      From_R10k_c(result->GetWritePtr(PLANAR_R), result->GetWritePtr(PLANAR_G), result->GetWritePtr(PLANAR_B), result->GetPitch(PLANAR_G),
        frame->GetPtr(), frame->GetPitch(), vi.width, vi.height);
      break;
    case AVI_SpecialFormats::v410:
      v410_to_yuv444p10(dstp, pitch, result->GetWritePtr(PLANAR_U), result->GetWritePtr(PLANAR_V), result->GetPitch(PLANAR_U),
        frame->GetPtr(), vi.width, vi.height);
      break;
    case AVI_SpecialFormats::v308:
      v308_to_yuv444p8(dstp, pitch, result->GetWritePtr(PLANAR_U), result->GetWritePtr(PLANAR_V), result->GetPitch(PLANAR_U),
        frame->GetPtr(), vi.width, vi.height);
      break;
    case AVI_SpecialFormats::v408:
      v408_to_yuva444p8(dstp, pitch, result->GetWritePtr(PLANAR_U), result->GetWritePtr(PLANAR_V), result->GetWritePtr(PLANAR_A), result->GetPitch(PLANAR_U), result->GetPitch(PLANAR_A),
        frame->GetPtr(), vi.width, vi.height);
      break;
    case AVI_SpecialFormats::Y410:
      if (vi.pixel_type == VideoInfo::CS_YUVA444P10)
        FromY410_c<true>(dstp, pitch, result->GetWritePtr(PLANAR_U), result->GetWritePtr(PLANAR_V), result->GetPitch(PLANAR_U),
          result->GetWritePtr(PLANAR_A), result->GetPitch(PLANAR_A),
          frame->GetPtr(), frame->GetPitch(), vi.width, vi.height);
      else
        FromY410_c<false>(dstp, pitch, result->GetWritePtr(PLANAR_U), result->GetWritePtr(PLANAR_V), result->GetPitch(PLANAR_U),
          result->GetWritePtr(PLANAR_A), result->GetPitch(PLANAR_A),
          frame->GetPtr(), frame->GetPitch(), vi.width, vi.height);
      break;
    case AVI_SpecialFormats::b64a:
    {
      // BGRA <-> big endian ARGB with byte swap
      uint8_t* pdst = dstp;

      int srcpitch = frame->GetPitch();
      const BYTE* src = frame->GetPtr();
#ifdef INTEL_INTRINSICS
      const bool ssse3 = (env->GetCPUFlags() & CPUF_SSSE3) != 0;
      const bool sse2 = (env->GetCPUFlags() & CPUF_SSE2) != 0;
      if (ssse3)
        bgra_to_argbBE_ssse3(pdst, pitch, src, srcpitch, vi.width, vi.height);
      else if (sse2)
        bgra_to_argbBE_sse2(pdst, pitch, src, srcpitch, vi.width, vi.height);
      else
#endif // INTEL_INTRINSICS
        bgra_to_argbBE_c(pdst, pitch, src, srcpitch, vi.width, vi.height);
    }
        break;
    case AVI_SpecialFormats::b48r:
    {
      // BGR <-> 16 bits per component big-endian Red, Green and Blue with byte swap
      uint8_t* pdst = dstp;

      int srcpitch = frame->GetPitch();
      const BYTE* src = frame->GetPtr();
      // c only
      bgr_to_rgbBE_c(pdst, pitch, src, srcpitch, vi.width, vi.height);
    }
    break;
    default:
        if (vi.IsPlanarRGB() || vi.IsPlanarRGBA()) {
          env->BitBlt(result->GetWritePtr(PLANAR_G), result->GetPitch(PLANAR_G), frame->GetPtr(PLANAR_G), frame->GetPitch(PLANAR_G), result->GetRowSize(PLANAR_G), result->GetHeight(PLANAR_G));
          env->BitBlt(result->GetWritePtr(PLANAR_B), result->GetPitch(PLANAR_B), frame->GetPtr(PLANAR_B), frame->GetPitch(PLANAR_B), result->GetRowSize(PLANAR_B), result->GetHeight(PLANAR_B));
          env->BitBlt(result->GetWritePtr(PLANAR_R), result->GetPitch(PLANAR_R), frame->GetPtr(PLANAR_R), frame->GetPitch(PLANAR_R), result->GetRowSize(PLANAR_R), result->GetHeight(PLANAR_R));
          if(vi.IsPlanarRGBA())
            env->BitBlt(result->GetWritePtr(PLANAR_A), result->GetPitch(PLANAR_A), frame->GetPtr(PLANAR_A), frame->GetPitch(PLANAR_A), result->GetRowSize(PLANAR_A), result->GetHeight(PLANAR_A));
        }
        else {
          env->BitBlt(dstp, pitch, frame->GetPtr(), frame->GetPitch(), result->GetRowSize(), result->GetHeight());
          // when no U or V plane these do nothing
          env->BitBlt(result->GetWritePtr(PLANAR_V), result->GetPitch(PLANAR_V), frame->GetPtr(PLANAR_V), frame->GetPitch(PLANAR_V), result->GetRowSize(PLANAR_V), result->GetHeight(PLANAR_V));
          env->BitBlt(result->GetWritePtr(PLANAR_U), result->GetPitch(PLANAR_U), frame->GetPtr(PLANAR_U), frame->GetPitch(PLANAR_U), result->GetRowSize(PLANAR_U), result->GetHeight(PLANAR_U));
        }
    } // switch
    return result;
}

#ifndef MSVC
static __inline LRESULT
ICDecompressEx(HIC hic,DWORD dwFlags,LPBITMAPINFOHEADER lpbiSrc,LPVOID lpSrc,INT xSrc,INT ySrc,INT dxSrc,INT dySrc,LPBITMAPINFOHEADER lpbiDst,LPVOID lpDst,INT xDst,INT yDst,INT dxDst,INT dyDst)
{
	ICDECOMPRESSEX ic;
	ic.dwFlags = dwFlags;
	ic.lpbiSrc = lpbiSrc;
	ic.lpSrc = lpSrc;
	ic.xSrc = xSrc;
	ic.ySrc = ySrc;
	ic.dxSrc = dxSrc;
	ic.dySrc = dySrc;
	ic.lpbiDst = lpbiDst;
	ic.lpDst = lpDst;
	ic.xDst = xDst;
	ic.yDst = yDst;
	ic.dxDst = dxDst;
	ic.dyDst = dyDst;
	return ICSendMessage(hic,ICM_DECOMPRESSEX,(DWORD_PTR)&ic,sizeof(ic));
}

static __inline LRESULT
ICDecompressExBegin(HIC hic,DWORD dwFlags,LPBITMAPINFOHEADER lpbiSrc,LPVOID lpSrc,INT xSrc,INT ySrc,INT dxSrc,INT dySrc,LPBITMAPINFOHEADER lpbiDst,LPVOID lpDst,INT xDst,INT yDst,INT dxDst,INT dyDst)
{
	ICDECOMPRESSEX ic;
	ic.dwFlags = dwFlags;
	ic.lpbiSrc = lpbiSrc;
	ic.lpSrc = lpSrc;
	ic.xSrc = xSrc;
	ic.ySrc = ySrc;
	ic.dxSrc = dxSrc;
	ic.dySrc = dySrc;
	ic.lpbiDst = lpbiDst;
	ic.lpDst = lpDst;
	ic.xDst = xDst;
	ic.yDst = yDst;
	ic.dxDst = dxDst;
	ic.dyDst = dyDst;
	return ICSendMessage(hic,ICM_DECOMPRESSEX_BEGIN,(DWORD_PTR)&ic,sizeof(ic));
}
#endif // MSVC

LRESULT AVISource::DecompressBegin(LPBITMAPINFOHEADER lpbiSrc, LPBITMAPINFOHEADER lpbiDst) {
  if (!ex) {
    LRESULT result = ICDecompressBegin(hic, lpbiSrc, lpbiDst);
    if (result != ICERR_UNSUPPORTED)
      return result;
    else
      ex = true;
      // and fall thru
  }
  return ICDecompressExBegin(hic, 0,
    lpbiSrc, 0, 0, 0, lpbiSrc->biWidth, lpbiSrc->biHeight,
    lpbiDst, 0, 0, 0, lpbiDst->biWidth, lpbiDst->biHeight);
}

LRESULT AVISource::DecompressFrame(int n, bool preroll, IScriptEnvironment* env) {
  AVS_UNUSED(env);
  _RPT2(0,"AVISource: Decompressing frame %d%s\n", n, preroll ? " (preroll)" : "");
  BYTE* buf = frame->GetPtr();
  long bytes_read;

  if (!hic) {
    bytes_read = long(frame->GetSize());
    pvideo->Read(n, 1, buf, bytes_read, &bytes_read, NULL);
    dropped_frame = !bytes_read;
    if (dropped_frame) return ICERR_OK;  // If frame is 0 bytes (dropped), return instead of attempt decompressing as Vdub.
  }
  else {
    bytes_read = srcbuffer_size;
    LRESULT err = pvideo->Read(n, 1, srcbuffer, srcbuffer_size, &bytes_read, NULL);
    while (err == AVIERR_BUFFERTOOSMALL || (err == 0 && !srcbuffer)) {
      delete[] srcbuffer;
      pvideo->Read(n, 1, 0, srcbuffer_size, &bytes_read, NULL);
      srcbuffer_size = bytes_read;
      srcbuffer = new BYTE[bytes_read + 16]; // Provide 16 hidden guard bytes for HuffYUV, Xvid, etc bug
      err = pvideo->Read(n, 1, srcbuffer, srcbuffer_size, &bytes_read, NULL);
    }
    dropped_frame = !bytes_read;
    if (dropped_frame) return ICERR_OK;  // If frame is 0 bytes (dropped), return instead of attempt decompressing as Vdub.

    // Fill guard bytes with 0xA5's for Xvid bug
    memset(srcbuffer + bytes_read, 0xA5, 16);
    // and a Null terminator for good measure
    srcbuffer[bytes_read + 15] = 0;

    int flags = preroll ? ICDECOMPRESS_PREROLL : 0;
    flags |= dropped_frame ? ICDECOMPRESS_NULLFRAME : 0;
    flags |= !pvideo->IsKeyFrame(n) ? ICDECOMPRESS_NOTKEYFRAME : 0;
    pbiSrc->biSizeImage = bytes_read;
    LRESULT result = !ex ? ICDecompress(hic, flags, pbiSrc, srcbuffer, &biDst, buf)
                         : ICDecompressEx(hic, flags, pbiSrc, srcbuffer,
                                          0, 0, vi.width, vi.height, &biDst, buf,
                                          0, 0, vi.width, vi.height);
    if (result != ICERR_OK) return result;
  }
  return ICERR_OK;
}


void AVISource::CheckHresult(HRESULT hr, const char* msg, IScriptEnvironment* env) {
  if (SUCCEEDED(hr)) return;
  char buf[1024] = {0};
  if (!FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM, NULL, hr, 0, buf, 1024, NULL))
    wsprintf(buf, "error code 0x%x", hr);
  env->ThrowError("AVISource: %s:\n%s", msg, buf);
}


// taken from VirtualDub
bool AVISource::AttemptCodecNegotiation(DWORD fccHandler, BITMAPINFOHEADER* bmih) {

    // Try the handler specified in the file first.  In some cases, it'll
  // be wrong or missing.

  if (fccHandler)
    hic = ICOpen(ICTYPE_VIDEO, fccHandler, ICMODE_DECOMPRESS);

  if (!hic || ICERR_OK!=ICDecompressQuery(hic, bmih, NULL)) {
    if (hic)
      ICClose(hic);

    // Pick a handler based on the biCompression field instead.

    hic = ICOpen(ICTYPE_VIDEO, bmih->biCompression, ICMODE_DECOMPRESS);

    if (!hic || ICERR_OK!=ICDecompressQuery(hic, bmih, NULL)) {
      if (hic)
        ICClose(hic);

      // This never seems to work...

      hic = ICLocate(ICTYPE_VIDEO, 0, bmih, NULL, ICMODE_DECOMPRESS);
    }
  }

    return !!hic;
}

static std::string fourCCtoString(DWORD fourCC) {
  std::ostringstream oss;
  for (int i = 0; i < 4; i++) {
    uint8_t b = fourCC & 0xFF;
    fourCC >>= 8;
    if (b < 32)
      oss << "[" << std::to_string(b) << "]";
    else
      oss << char(b);
  }
  return oss.str();
}

void AVISource::LocateVideoCodec(const char fourCC[], IScriptEnvironment* env) {
  AVISTREAMINFO asi;
  CheckHresult(pvideo->Info(&asi, sizeof(asi)), "couldn't get video info", env);
  long size = sizeof(BITMAPINFOHEADER);

  // Read video format.  If it's a
  // type-1 DV, we're going to have to fake it.

  if (bIsType1) {
    pbiSrc = (BITMAPINFOHEADER *)malloc(size);
    if (!pbiSrc) env->ThrowError("AviSource: Could not allocate BITMAPINFOHEADER.");

    pbiSrc->biSize      = sizeof(BITMAPINFOHEADER);
    pbiSrc->biWidth     = 720;

    if (asi.dwRate > asi.dwScale*26LL)
      pbiSrc->biHeight      = 480;
    else
      pbiSrc->biHeight      = 576;

    pbiSrc->biPlanes      = 1;
    pbiSrc->biBitCount    = 24;
    pbiSrc->biCompression = MAKEFOURCC('d','v','s','d'); // 'dsvd';
    pbiSrc->biSizeImage   = asi.dwSuggestedBufferSize;
    pbiSrc->biXPelsPerMeter = 0;
    pbiSrc->biYPelsPerMeter = 0;
    pbiSrc->biClrUsed     = 0;
    pbiSrc->biClrImportant  = 0;

  } else {
    CheckHresult(pvideo->ReadFormat(0, 0, &size), "couldn't get video format size", env);
    pbiSrc = (LPBITMAPINFOHEADER)malloc(size);
    CheckHresult(pvideo->ReadFormat(0, pbiSrc, &size), "couldn't get video format", env);
  }

  vi.width = pbiSrc->biWidth;
  vi.height = pbiSrc->biHeight < 0 ? -pbiSrc->biHeight : pbiSrc->biHeight;
  vi.SetFPS(asi.dwRate, asi.dwScale);
  vi.num_frames = asi.dwLength;

  bool overridden = false;

  // try the requested decoder, if specified
  if (fourCC != NULL /*&& strlen(fourCC) == 4*/) { // no strlen! fourCC can contain zeros as well
    DWORD fcc = fourCC[0] | (fourCC[1] << 8) | (fourCC[2] << 16) | (fourCC[3] << 24);
    if (pbiSrc->biCompression != fcc)
      overridden = true;
    asi.fccHandler = pbiSrc->biCompression = fcc;
  }

  // see if we can handle the video format directly
  // when recognizing these formats in input fourCC, use them directly
  // Note: forced overrides may not work

  // check for only basic formats only. Known formats with special unpackers are handled outside.
  if (pbiSrc->biCompression == MAKEFOURCC('Y', 'U', 'Y', '2')) { // :FIXME: Handle UYVY, etc
    vi.pixel_type = VideoInfo::CS_YUY2;
  } else if (pbiSrc->biCompression == MAKEFOURCC('Y', 'V', '1', '2')) {
    vi.pixel_type = VideoInfo::CS_YV12;
  } else if (pbiSrc->biCompression == MAKEFOURCC('I', '4', '2', '0')) {
    vi.pixel_type = VideoInfo::CS_I420;
  } else if (pbiSrc->biCompression == BI_RGB && pbiSrc->biBitCount == 32) {
    vi.pixel_type = VideoInfo::CS_BGR32;
  } else if (pbiSrc->biCompression == BI_RGB && pbiSrc->biBitCount == 24) {
    vi.pixel_type = VideoInfo::CS_BGR24;
  } else if (pbiSrc->biCompression == MAKEFOURCC('G', 'R', 'E', 'Y')) {
    vi.pixel_type = VideoInfo::CS_Y8;
  } else if (pbiSrc->biCompression == MAKEFOURCC('Y', '8', '0', '0')) {
    vi.pixel_type = VideoInfo::CS_Y8;
  } else if (pbiSrc->biCompression == MAKEFOURCC('Y', '8', ' ', ' ')) {
    vi.pixel_type = VideoInfo::CS_Y8;
  } else if (pbiSrc->biCompression == MAKEFOURCC('Y', 'V', '2', '4')) {
    vi.pixel_type = VideoInfo::CS_YV24;
  } else if (pbiSrc->biCompression == MAKEFOURCC('Y', 'V', '1', '6')) {
    vi.pixel_type = VideoInfo::CS_YV16;
  } else if (pbiSrc->biCompression == MAKEFOURCC('Y', '4', '1', 'B')) {
    vi.pixel_type = VideoInfo::CS_YV411;
  }
  else if (pbiSrc->biCompression == MAKEFOURCC('B', 'R', 'A', 64)) { // BRA@ ie. BRA[64]
    vi.pixel_type = VideoInfo::CS_BGR64;
  }
  else if (pbiSrc->biCompression == MAKEFOURCC('b', '6', '4', 'a')) { // b64a
    vi.pixel_type = VideoInfo::CS_BGR64;
  }
  else if (pbiSrc->biCompression == MAKEFOURCC('B', 'G', 'R', 48)) { // BGR0 ie. BGR[48]
    vi.pixel_type = VideoInfo::CS_BGR48;
  }
  else if (pbiSrc->biCompression == MAKEFOURCC('b', '4', '8', 'r')) { // b48r
    vi.pixel_type = VideoInfo::CS_BGR48;
  }
  else if (pbiSrc->biCompression == MAKEFOURCC('v', '2', '1', '0')) { // v210
    vi.pixel_type = VideoInfo::CS_YUV422P10;
  }
  else if (pbiSrc->biCompression == MAKEFOURCC('P', '2', '1', '0')) { // P210
    vi.pixel_type = VideoInfo::CS_YUV422P10;
  }
  else if (pbiSrc->biCompression == MAKEFOURCC('P', '0', '1', '0')) { // P010
    vi.pixel_type = VideoInfo::CS_YUV420P10;
  }
  else if (pbiSrc->biCompression == MAKEFOURCC('P', '0', '1', '6')) { // P016
    vi.pixel_type = VideoInfo::CS_YUV420P16;
  }
  else if (pbiSrc->biCompression == MAKEFOURCC('P', '2', '1', '6')) { // P216
    vi.pixel_type = VideoInfo::CS_YUV422P16;
  }
  else if (pbiSrc->biCompression == MAKEFOURCC('v', '4', '1', '0')) { // v410
    vi.pixel_type = VideoInfo::CS_YUV444P10;
  }
  else if (pbiSrc->biCompression == MAKEFOURCC('Y', '4', '1', '6')) { // Y416
    vi.pixel_type = VideoInfo::CS_YUV444P16;
    // FIXME/Decide: Y416 contains alpha, we ignore it now
  }
  else if (pbiSrc->biCompression == MAKEFOURCC('r', '2', '1', '0')) { // r210
    vi.pixel_type = VideoInfo::CS_RGBP10;
  }
  else if (pbiSrc->biCompression == MAKEFOURCC('R', '1', '0', 'k')) { // R10k
    vi.pixel_type = VideoInfo::CS_RGBP10;
  }
  else if (pbiSrc->biCompression == MAKEFOURCC('v', '3', '0', '8')) { // v308
    vi.pixel_type = VideoInfo::CS_YV24;
  }
  else if (pbiSrc->biCompression == MAKEFOURCC('v', '4', '0', '8')) { // v408
    vi.pixel_type = VideoInfo::CS_YUVA444;
  }
  else if (pbiSrc->biCompression == MAKEFOURCC('Y', '4', '1', '0')) { // Y410
    vi.pixel_type = VideoInfo::CS_YUVA444P10;
    // Y410 contains a 2 bit alpha, so we use YUVA
  } else {
    // otherwise, find someone who will decompress it
    switch(pbiSrc->biCompression) {
    case MAKEFOURCC('M','P','4','3'):    // Microsoft MPEG-4 V3 '34PM'
    case MAKEFOURCC('D','I','V','3'):    // "DivX Low-Motion" (4.10.0.3917) '3VID'
    case MAKEFOURCC('D','I','V','4'):    // "DivX Fast-Motion" (4.10.0.3920) 4VID'
    case MAKEFOURCC('A','P','4','1'):    // "AngelPotion Definitive" (4.0.00.3688) '14PA'
      if (AttemptCodecNegotiation(asi.fccHandler, pbiSrc)) return;
      pbiSrc->biCompression = MAKEFOURCC('M', 'P', '4', '3');
      if (AttemptCodecNegotiation(asi.fccHandler, pbiSrc)) return;
      pbiSrc->biCompression = MAKEFOURCC('D', 'I', 'V', '3');
      if (AttemptCodecNegotiation(asi.fccHandler, pbiSrc)) return;
      pbiSrc->biCompression = MAKEFOURCC('D', 'I', 'V', '4');
      if (AttemptCodecNegotiation(asi.fccHandler, pbiSrc)) return;
      pbiSrc->biCompression = MAKEFOURCC('A', 'P', '4', '1');
    default:
      if (AttemptCodecNegotiation(asi.fccHandler, pbiSrc)) return;
    }
    auto fourCCstr = fourCCtoString(asi.fccHandler);
    const char *s = fourCCstr.c_str();
    env->SaveString(s);
    env->ThrowError("AVISource: couldn't locate a decompressor for fourcc %s", s);
  }
  if (!overridden) return;
  if (AttemptCodecNegotiation(asi.fccHandler, pbiSrc)) return;
  // note: it is possible that fourCC G3[0][10] cannot be negotiated
  // but if e.g. M0RG is negotiated here, the driver can support G3[0][10] later in icDecompressQuery
  auto fourCCstr = fourCCtoString(asi.fccHandler);
  const char* s = fourCCstr.c_str();
  env->SaveString(s);
  env->ThrowError("AVISource: couldn't locate a decompressor for overridden fourcc %s", s);
}

static bool setInternalFormat_b64a(VideoInfo& vi, HIC& hic, BITMAPINFOHEADER* pbiSrc, BITMAPINFOHEADER* biDst, DWORD fourCC, int pixel_type) {
  vi.pixel_type = pixel_type;
  biDst->biSizeImage = vi.BMPSize();
  biDst->biCompression = fourCC; // BRA@ ie. BRA[64]
  biDst->biBitCount = 64;
  if (hic == nullptr) return true;
  return (ICERR_OK == ICDecompressQuery(hic, pbiSrc, biDst));
}

static bool setInternalFormat_BRA64(VideoInfo& vi, HIC& hic, BITMAPINFOHEADER* pbiSrc, BITMAPINFOHEADER* biDst, DWORD fourCC, int pixel_type) {
  vi.pixel_type = pixel_type;
  biDst->biSizeImage = vi.BMPSize();
  biDst->biCompression = fourCC; // BRA@ ie. BRA[64]
  biDst->biBitCount = 64;
  if (hic == nullptr) return true;
  return (ICERR_OK == ICDecompressQuery(hic, pbiSrc, biDst));
}

static bool setInternalFormat_BGR48(VideoInfo& vi, HIC& hic, BITMAPINFOHEADER* pbiSrc, BITMAPINFOHEADER* biDst, DWORD fourCC, int pixel_type) {
  vi.pixel_type = pixel_type;
  biDst->biSizeImage = vi.BMPSize();
  biDst->biCompression = fourCC; // BGR0 ie. BGR48[64]
  biDst->biBitCount = 48;
  if (hic == nullptr) return true;
  return (ICERR_OK == ICDecompressQuery(hic, pbiSrc, biDst));
}

static bool setInternalFormat_b48r(VideoInfo& vi, HIC& hic, BITMAPINFOHEADER* pbiSrc, BITMAPINFOHEADER* biDst, DWORD fourCC, int pixel_type) {
  vi.pixel_type = pixel_type;
  biDst->biSizeImage = vi.BMPSize();
  biDst->biCompression = fourCC; // BGR0 ie. BGR48[64]
  biDst->biBitCount = 48;
  if (hic == nullptr) return true;
  return (ICERR_OK == ICDecompressQuery(hic, pbiSrc, biDst));
}

static bool setInternalFormat_v210(VideoInfo& vi, HIC& hic, BITMAPINFOHEADER* pbiSrc, BITMAPINFOHEADER* biDst, DWORD fourCC, int pixel_type) {
  vi.pixel_type = pixel_type;
  biDst->biSizeImage = AviHelper_ImageSize(&vi, false, true, false, false, false, false, false, false);
  biDst->biCompression = fourCC; // v210 (422P10)
  biDst->biBitCount = 30;
  if (hic == nullptr) return true;
  return (ICERR_OK == ICDecompressQuery(hic, pbiSrc, biDst));
}

static bool setInternalFormat_P210(VideoInfo& vi, HIC& hic, BITMAPINFOHEADER* pbiSrc, BITMAPINFOHEADER* biDst, DWORD fourCC, int pixel_type) {
  vi.pixel_type = pixel_type;
  biDst->biSizeImage = vi.BMPSize();
  biDst->biCompression = fourCC; // P210 (422P10)
  biDst->biBitCount = 30; // should be 30, not 10+(10+10)/2
  if (hic == nullptr) return true;
  return (ICERR_OK == ICDecompressQuery(hic, pbiSrc, biDst));
}

static bool setInternalFormat_P010(VideoInfo& vi, HIC& hic, BITMAPINFOHEADER* pbiSrc, BITMAPINFOHEADER* biDst, DWORD fourCC, int pixel_type) {
  vi.pixel_type = pixel_type;
  biDst->biSizeImage = vi.BMPSize();
  biDst->biCompression = fourCC; // P010 (420P10)
  biDst->biBitCount = 15; // 10+(10+10)/4 or 3x10?
  if (hic == nullptr) return true;
  return (ICERR_OK == ICDecompressQuery(hic, pbiSrc, biDst));
}

static bool setInternalFormat_P016(VideoInfo& vi, HIC& hic, BITMAPINFOHEADER* pbiSrc, BITMAPINFOHEADER* biDst, DWORD fourCC, int pixel_type) {
  vi.pixel_type = pixel_type;
  biDst->biSizeImage = vi.BMPSize();
  biDst->biCompression = fourCC; // P016 (420P16)
  biDst->biBitCount = 24; // 16+(16+16)/4 or 3x16?
  if (hic == nullptr) return true;
  return (ICERR_OK == ICDecompressQuery(hic, pbiSrc, biDst));
}

static bool setInternalFormat_Y416(VideoInfo& vi, HIC& hic, BITMAPINFOHEADER* pbiSrc, BITMAPINFOHEADER* biDst, DWORD fourCC, int pixel_type) {
  vi.pixel_type = pixel_type;
  biDst->biSizeImage = vi.BMPSize();
  biDst->biCompression = fourCC; // Y416 (444P16)
  biDst->biBitCount = 48; // 3x16?
  if (hic == nullptr) return true;
  return (ICERR_OK == ICDecompressQuery(hic, pbiSrc, biDst));
}

static bool setInternalFormat_Y410(VideoInfo& vi, HIC& hic, BITMAPINFOHEADER* pbiSrc, BITMAPINFOHEADER* biDst, DWORD fourCC, int pixel_type) {
  vi.pixel_type = pixel_type;
  biDst->biSizeImage = AviHelper_ImageSize(&vi, false, false, false, false, false, false, false, true);
  biDst->biCompression = fourCC; // Y410 (444P10)
  biDst->biBitCount = 32; // 3x10+2?
  if (hic == nullptr) return true;
  return (ICERR_OK == ICDecompressQuery(hic, pbiSrc, biDst));
}

static bool setInternalFormat_P216(VideoInfo& vi, HIC& hic, BITMAPINFOHEADER* pbiSrc, BITMAPINFOHEADER* biDst, DWORD fourCC, int pixel_type) {
  vi.pixel_type = pixel_type;
  biDst->biSizeImage = vi.BMPSize();
  biDst->biCompression = fourCC; // P216 (422P16)
  biDst->biBitCount = 32; // 16+(16+16)/2 or 3x16?
  if (hic == nullptr) return true;
  return (ICERR_OK == ICDecompressQuery(hic, pbiSrc, biDst));
}

static bool setInternalFormat_v410(VideoInfo& vi, HIC& hic, BITMAPINFOHEADER* pbiSrc, BITMAPINFOHEADER* biDst, DWORD fourCC, int pixel_type) {
  vi.pixel_type = pixel_type;
  biDst->biSizeImage = AviHelper_ImageSize(&vi, false, false, true, false, false, false, false, false);
  biDst->biCompression = fourCC; // v410 (444P10)
  biDst->biBitCount = 30; // 3x10?
  if (hic == nullptr) return true;
  return (ICERR_OK == ICDecompressQuery(hic, pbiSrc, biDst));
}

static bool setInternalFormat_r210(VideoInfo& vi, HIC& hic, BITMAPINFOHEADER* pbiSrc, BITMAPINFOHEADER* biDst, DWORD fourCC, int pixel_type) {
  vi.pixel_type = pixel_type;
  biDst->biSizeImage = AviHelper_ImageSize(&vi, false, false, false, true, false, false, false, false);
  biDst->biCompression = fourCC; // r210 (RGBP10)
  biDst->biBitCount = 30; // 3x10?
  if (hic == nullptr) return true;
  return (ICERR_OK == ICDecompressQuery(hic, pbiSrc, biDst));
}

static bool setInternalFormat_R10k(VideoInfo& vi, HIC& hic, BITMAPINFOHEADER* pbiSrc, BITMAPINFOHEADER* biDst, DWORD fourCC, int pixel_type) {
  vi.pixel_type = pixel_type;
  biDst->biSizeImage = AviHelper_ImageSize(&vi, false, false, false, false, true, false, false, false);
  biDst->biCompression = fourCC; // R10k (RGBP10)
  biDst->biBitCount = 30; // 3x10?
  if (hic == nullptr) return true;
  return (ICERR_OK == ICDecompressQuery(hic, pbiSrc, biDst));
}

static bool setInternalFormat_v308(VideoInfo& vi, HIC& hic, BITMAPINFOHEADER* pbiSrc, BITMAPINFOHEADER* biDst, DWORD fourCC, int pixel_type) {
  vi.pixel_type = pixel_type;
  biDst->biSizeImage = AviHelper_ImageSize(&vi, false, false, false, false, false, true, false, false);
  biDst->biCompression = fourCC; // v308 (YV24 YUV444P8)
  biDst->biBitCount = 24;
  if (hic == nullptr) return true;
  return (ICERR_OK == ICDecompressQuery(hic, pbiSrc, biDst));
}

static bool setInternalFormat_v408(VideoInfo& vi, HIC& hic, BITMAPINFOHEADER* pbiSrc, BITMAPINFOHEADER* biDst, DWORD fourCC, int pixel_type) {
  vi.pixel_type = pixel_type;
  biDst->biSizeImage = AviHelper_ImageSize(&vi, false, false, false, false, false, false, true, false);
  biDst->biCompression = fourCC; // v408 (YUVA444P8)
  biDst->biBitCount = 32;
  if (hic == nullptr) return true;
  return (ICERR_OK == ICDecompressQuery(hic, pbiSrc, biDst));
}

AVISource::AVISource(const char filename[], bool fAudio, const char pixel_type[], const char fourCC[], int vtrack, int atrack, avi_mode_e mode, bool utf8, IScriptEnvironment* env) {
  srcbuffer = 0; srcbuffer_size = 0;
  memset(&vi, 0, sizeof(vi));
  ex = false;
  last_frame_no = -1;
  pbiSrc = 0;
  aSrc = 0;
  audioStreamSource = 0;
  pvideo=0;
  pfile=0;
  bIsType1 = false;
  hic = 0;
  bInvertFrames = false;

  specf = AVI_SpecialFormats::none;
  bMediaPad = false;
  frame = 0;

  auto filename_w = utf8 ? Utf8ToWideChar(filename) : AnsiToWideChar(filename);

  AVIFileInit();
  try {

    if (mode == MODE_NORMAL) {
      // if it looks like an AVI file, open in OpenDML mode; otherwise AVIFile mode
      HANDLE h = CreateFileW(filename_w.get(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, 0, NULL);
      if (h == INVALID_HANDLE_VALUE) {
        env->ThrowError("AVISource autodetect: couldn't open file '%s'\nError code: %d", filename, GetLastError());
      }
      unsigned int buf[3];
      DWORD bytes_read;
      if (ReadFile(h, buf, 12, &bytes_read, NULL) && bytes_read == 12 && buf[0] == MAKEFOURCC('R','I','F','F') && buf[2] == MAKEFOURCC('A','V','I',' '))
        mode = MODE_OPENDML;
      else
        mode = MODE_AVIFILE;
      CloseHandle(h);
    }

    if (mode == MODE_AVIFILE || mode == MODE_WAV) {    // AVIFile mode
      PAVIFILE paf = NULL;
      try { // The damn .WAV clsid handler has only a 48 byte buffer to parse the filename and GPF's
		if (FAILED(AVIFileOpenW(&paf, filename_w.get(), OF_READ, 0)))
		  env->ThrowError("AVIFileSource: couldn't open file '%s'", filename);
      }
	  catch (AvisynthError) {
		throw;
	  }
	  catch (...) {
		env->ThrowError("AVIFileSource: VFW failure, AVIFileOpen(%s), length of filename part must be < 48", filename);
	  }
	  pfile = CreateAVIReadHandler(paf);
    } else {              // OpenDML mode
      pfile = CreateAVIReadHandler(filename_w.get());
    }

    if (mode != MODE_WAV) { // check for video stream
      pvideo = pfile->GetStream(streamtypeVIDEO, vtrack);

      if (!pvideo) { // Attempt DV type 1 video.
        pvideo = pfile->GetStream(MAKEFOURCC('i','a','v','s'), vtrack);
        bIsType1 = true;
      }

      if (pvideo) {
        // accept code-in-the-bracker style like G4[0][16]
        bool inBracket = false;
        char buf[4+1];
        int len = 0;
        char ch;
        int number = 0;
        const char* fc = fourCC;
        while ((ch = *fc++)) {
          if (len == 4)
            env->ThrowError("Avisource: fourCC must be four characters");
          if (ch == '[') {
            if (inBracket)
              env->ThrowError("Avisource: fourCC syntax error: unexpected [");
            inBracket = true;
            number = 0;
            continue;
          }
          if (ch == ']') {
            if (!inBracket)
              env->ThrowError("Avisource: fourCC syntax error: unexpected ]");
            buf[len++] = number;
            inBracket = false;
            continue;
          }
          if (inBracket) {
            if(!isdigit(ch))
              env->ThrowError("Avisource: fourCC parsing: only numbers allowed inside brackets");
            number = number * 10 + (ch - '0');
            if (number>255)
              env->ThrowError("Avisource: fourCC parsing: maximum code cannot exceed 255");
            continue;
          }
          buf[len++] =  ch;
        }
        if (inBracket)
          env->ThrowError("Avisource: fourCC syntax error: unclosed [");
        buf[len] = 0;

        LocateVideoCodec(len == 0 ? nullptr : buf, env);
        if (hic) {
          if (pixel_type[0] == '+') {
            pixel_type += 1;
            bMediaPad = true;
          }
          bool forcedType = !(pixel_type[0] == 0);

          bool fY8    = pixel_type[0] == 0 || lstrcmpi(pixel_type, "Y8"   ) == 0;
          bool fYV12  = pixel_type[0] == 0 || lstrcmpi(pixel_type, "YV12" ) == 0;
          bool fYV16  = pixel_type[0] == 0 || lstrcmpi(pixel_type, "YV16" ) == 0;
          bool fYV24  = pixel_type[0] == 0 || lstrcmpi(pixel_type, "YV24" ) == 0;
          bool fv308  = pixel_type[0] == 0 || lstrcmpi(pixel_type, "v308") == 0;
          bool fv408  = pixel_type[0] == 0 || lstrcmpi(pixel_type, "v408") == 0;

          bool fYV411 = pixel_type[0] == 0 || lstrcmpi(pixel_type, "YV411") == 0;
          bool fYUY2  = pixel_type[0] == 0 || lstrcmpi(pixel_type, "YUY2" ) == 0;
          bool fRGB32 = pixel_type[0] == 0 || lstrcmpi(pixel_type, "RGB32") == 0;
          bool fRGB24 = pixel_type[0] == 0 || lstrcmpi(pixel_type, "RGB24") == 0;
          bool fRGB48 = pixel_type[0] == 0 || lstrcmpi(pixel_type, "RGB48") == 0;
          bool fRGB64 = pixel_type[0] == 0 || lstrcmpi(pixel_type, "RGB64") == 0;

          bool fYUV420P10 = pixel_type[0] == 0 || lstrcmpi(pixel_type, "YUV420P10") == 0;
          bool fP010 = pixel_type[0] == 0 || lstrcmpi(pixel_type, "P010") == 0;

          bool fYUV420P16 = pixel_type[0] == 0 || lstrcmpi(pixel_type, "YUV420P16") == 0;
          bool fP016 = pixel_type[0] == 0 || lstrcmpi(pixel_type, "P016") == 0;

          bool fYUV422P10 = pixel_type[0] == 0 || lstrcmpi(pixel_type, "YUV422P10") == 0;
          // we cannot know it beforehand, don't set it true if not specified, only generic YUV422P10 is valid
          bool fv210 = lstrcmpi(pixel_type, "v210") == 0;
          bool fP210 = lstrcmpi(pixel_type, "P210") == 0;

          bool fYUV422P16 = pixel_type[0] == 0 || lstrcmpi(pixel_type, "YUV422P16") == 0;
          bool fP216 = pixel_type[0] == 0 || lstrcmpi(pixel_type, "P216") == 0;

          bool fYUV444P10 = pixel_type[0] == 0 || lstrcmpi(pixel_type, "YUV444P10") == 0;
          bool fv410 = pixel_type[0] == 0 || lstrcmpi(pixel_type, "v410") == 0;
          bool fY410 = pixel_type[0] == 0 || lstrcmpi(pixel_type, "Y410") == 0;

          bool fYUV444P16 = pixel_type[0] == 0 || lstrcmpi(pixel_type, "YUV444P16") == 0;
          bool fY416 = pixel_type[0] == 0 || lstrcmpi(pixel_type, "Y416") == 0;

          bool fRGBP = pixel_type[0] == 0 || lstrcmpi(pixel_type, "RGBP") == 0; // RGBP means planar RGB 10-16
          // FIXME: 8 bit planar RGB possible?
          // FIXME: hint for RGBPA priority?
          bool fGrayscale = pixel_type[0] == 0 || lstrcmpi(pixel_type, "Y") == 0; // Y8-16

          // special 10 bit RGB, multiple formats supported
          bool fRGBP10 = pixel_type[0] == 0 || lstrcmpi(pixel_type, "RGBP10") == 0;
          bool fr210 = pixel_type[0] == 0 || lstrcmpi(pixel_type, "r210") == 0;
          bool fR10k = pixel_type[0] == 0 || lstrcmpi(pixel_type, "R10k") == 0;

          // we don't set specifically v210, P010, P016, P210, P216, Y410, Y416, v308, v408 for auto.
          // These are set only when negotiated
          if (lstrcmpi(pixel_type, "AUTO") == 0) {
            fY8 = fYV12 = fYUY2 = fRGB32 = fRGB24 = fRGB48 = fRGB64 = true;
            fYUV420P10 = true;
            fYUV420P16 = true;
            fYUV422P10 = true;
            fYUV422P16 = true;
            fYUV444P10 = true;
            fYUV444P16 = true;
            fRGBP = true;
            fRGBP10 = true;
            fr210 = true;
            fR10k = true;
            fGrayscale = true;
            forcedType = false;
          }
          else if (lstrcmpi(pixel_type, "FULL") == 0) {
            fY8 = fYV12 = fYV16 = fYV24 = fYV411 = fYUY2 = fRGB32 = fRGB24 = fRGB48 = fRGB64 = true;
            fYUV420P10 = true;
            fYUV420P16 = true;
            fYUV422P10 = true;
            fYUV422P16 = true;
            fYUV444P10 = true;
            fYUV444P16 = true;
            fRGBP = true;
            fRGBP10 = true;
            fr210 = true;
            fR10k = true;
            fGrayscale = true;
            forcedType = false;
          }

          if (!(fY8 || fYV12 || fYV16 || fYV24 || fYV411 || fYUY2 || fRGB32 || fRGB24 || fRGB48 || fRGB64
            || fv308 || fv408
            || fYUV420P10 || fP010
            || fYUV420P16 || fP016
            || fYUV422P10 || fP210 || fv210
            || fYUV422P16 || fP216
            || fYUV444P10 || fv410 || fY410
            || fYUV444P16 || fY416
            || fRGBP
            || fRGBP10
            || fr210 || fR10k
            || fGrayscale
            ))
            env->ThrowError("AVISource: requested format must be one of YV12/16/24, YV411, YUY2, Y8, Y, RGBP, RGBP10, r210, R10k, RGB24/32/48/64, YUV420P10/16, YUV422P10/16, YUV444P10/16, v210, P010/16, P210/16, v410, Y410, Y416, v308, v408, AUTO or FULL");

          // try to decompress to YV12, YV411, YV16, YV24, YUY2, Y8, RGB32, and RGB24, RGB48, RGB64, YUV422P10 in turn
          memset(&biDst, 0, sizeof(BITMAPINFOHEADER));
          biDst.biSize = sizeof(BITMAPINFOHEADER);
          biDst.biWidth = vi.width;
          biDst.biHeight = vi.height;
          biDst.biPlanes = 1;
          bool bOpen = true;

          // YV24
          if (fYV24 && bOpen) {
            vi.pixel_type = VideoInfo::CS_YV24;
            biDst.biSizeImage = vi.BMPSize();
            biDst.biCompression = MAKEFOURCC('Y', 'V', '2', '4');
            biDst.biBitCount = 24;
            if (ICERR_OK == ICDecompressQuery(hic, pbiSrc, &biDst)) {
              _RPT0(0,"AVISource: Opening as YV24.\n");
              bOpen = false;  // Skip further attempts
            } else if (forcedType) {
               env->ThrowError("AVISource: the video decompressor couldn't produce YV24 output");
            }
          }

          // YV16
          if (fYV16 && bOpen) {
            vi.pixel_type = VideoInfo::CS_YV16;
            biDst.biSizeImage = vi.BMPSize();
            biDst.biCompression = MAKEFOURCC('Y', 'V', '1', '6');
            biDst.biBitCount = 16;
            if (ICERR_OK == ICDecompressQuery(hic, pbiSrc, &biDst)) {
              _RPT0(0,"AVISource: Opening as YV16.\n");
              bOpen = false;  // Skip further attempts
            } else if (forcedType) {
               env->ThrowError("AVISource: the video decompressor couldn't produce YV16 output");
            }
          }

          // YV12
          if (fYV12 && bOpen) {
            vi.pixel_type = VideoInfo::CS_YV12;
            biDst.biSizeImage = vi.BMPSize();
            biDst.biCompression = MAKEFOURCC('Y', 'V', '1', '2');
            biDst.biBitCount = 12;
            if (ICERR_OK == ICDecompressQuery(hic, pbiSrc, &biDst)) {
              _RPT0(0,"AVISource: Opening as YV12.\n");
              bOpen = false;  // Skip further attempts
            } else if (forcedType) {
               env->ThrowError("AVISource: the video decompressor couldn't produce YV12 output");
            }
          }

// ::FIXME:: Is this the most appropriate order.  Not sure about YUY2 vrs YV411

          // YV411
          if (fYV411 && bOpen) {
            vi.pixel_type = VideoInfo::CS_YV411;
            biDst.biSizeImage = vi.BMPSize();
            biDst.biCompression = MAKEFOURCC('Y', '4', '1', 'B');
            biDst.biBitCount = 16;
            if (ICERR_OK == ICDecompressQuery(hic, pbiSrc, &biDst)) {
              _RPT0(0,"AVISource: Opening as YV411.\n");
              bOpen = false;  // Skip further attempts
            } else if (forcedType) {
               env->ThrowError("AVISource: the video decompressor couldn't produce YV411 output");
            }
          }

          // YUY2
          if (fYUY2 && bOpen) {
            vi.pixel_type = VideoInfo::CS_YUY2;
            biDst.biSizeImage = vi.BMPSize();
            biDst.biCompression = MAKEFOURCC('Y', 'U', 'Y', '2'); // :FIXME: Handle UYVY, etc
            biDst.biBitCount = 16;
            if (ICERR_OK == ICDecompressQuery(hic, pbiSrc, &biDst)) {
              _RPT0(0,"AVISource: Opening as YUY2.\n");
              bOpen = false;  // Skip further attempts
            } else if (forcedType) {
               env->ThrowError("AVISource: the video decompressor couldn't produce YUY2 output");
            }
          }

          // RGBP 10 bit special
          if ((fRGBP10 || fr210 || fR10k) && bOpen) {
            if (fRGBP10 && bOpen) {
              vi.pixel_type = VideoInfo::CS_RGBP10;
              biDst.biSizeImage = vi.BMPSize();
              biDst.biCompression = MAKEFOURCC('G', '3', 0, 10); // ffmpeg GBRP10LE
              biDst.biBitCount = 30;
              if (ICERR_OK == ICDecompressQuery(hic, pbiSrc, &biDst)) {
                _RPT0(0, "AVISource: Opening as G3[0][10].\n");
                bOpen = false;  // Skip further attempts
              }
            }
            if ((fRGBP10 || fr210) && bOpen) {
              vi.pixel_type = VideoInfo::CS_RGBP10;
              if (setInternalFormat_r210(vi, hic, pbiSrc, &biDst, MAKEFOURCC('r', '2', '1', '0'), VideoInfo::CS_RGBP10)) {
                _RPT0(0, "AVISource: Opening as r210.\n");
                specf = AVI_SpecialFormats::r210;
                bOpen = false;  // Skip further attempts
              }
            }
            if ((fRGBP10 || fR10k) && bOpen) {
              vi.pixel_type = VideoInfo::CS_RGBP10;
              if (setInternalFormat_R10k(vi, hic, pbiSrc, &biDst, MAKEFOURCC('R', '1', '0', 'k'), VideoInfo::CS_RGBP10)) {
                _RPT0(0, "AVISource: Opening as R10k.\n");
                specf = AVI_SpecialFormats::R10k;
                bOpen = false;  // Skip further attempts
              }
            }
            if (bOpen && forcedType) {
              env->ThrowError("AVISource: the video decompressor couldn't produce Planar RGB 10 output");
            }
          }

          // "RGBP": this is a hint that planar RGB is preferred, no specific bit depth needed
          if (fRGBP && bOpen) {
            // FIXME: consider putting here G3[0][8] and G4[0][8]
            vi.pixel_type = VideoInfo::CS_RGBP10;
            biDst.biSizeImage = vi.BMPSize();
            biDst.biCompression = MAKEFOURCC('G', '3', 0, 10); // ffmpeg GBRP10LE
            biDst.biBitCount = 30;
            if (ICERR_OK == ICDecompressQuery(hic, pbiSrc, &biDst)) {
              _RPT0(0, "AVISource: Opening as G3[0][10].\n");
              bOpen = false;  // Skip further attempts
            }
            else {
              vi.pixel_type = VideoInfo::CS_RGBP10;
              if (setInternalFormat_r210(vi, hic, pbiSrc, &biDst, MAKEFOURCC('r', '2', '1', '0'), VideoInfo::CS_RGBP10)) {
                _RPT0(0, "AVISource: Opening as r210.\n");
                specf = AVI_SpecialFormats::r210;
                bOpen = false;  // Skip further attempts
              }
              else {
                vi.pixel_type = VideoInfo::CS_RGBP10;
                if (setInternalFormat_R10k(vi, hic, pbiSrc, &biDst, MAKEFOURCC('R', '1', '0', 'k'), VideoInfo::CS_RGBP10)) {
                  _RPT0(0, "AVISource: Opening as R10k.\n");
                  specf = AVI_SpecialFormats::R10k;
                  bOpen = false;  // Skip further attempts
                }
                else {
                  vi.pixel_type = VideoInfo::CS_RGBP12;
                  biDst.biCompression = MAKEFOURCC('G', '3', 0, 12); // ffmpeg GBRP12LE
                  biDst.biBitCount = 36;
                  if (ICERR_OK == ICDecompressQuery(hic, pbiSrc, &biDst)) {
                    _RPT0(0, "AVISource: Opening as G3[0][12].\n");
                    bOpen = false;  // Skip further attempts
                  }
                  else {
                    vi.pixel_type = VideoInfo::CS_RGBP14;
                    biDst.biCompression = MAKEFOURCC('G', '3', 0, 14); // ffmpeg GBRP14LE
                    biDst.biBitCount = 42;
                    if (ICERR_OK == ICDecompressQuery(hic, pbiSrc, &biDst)) {
                      _RPT0(0, "AVISource: Opening as G3[0][14].\n");
                      bOpen = false;  // Skip further attempts
                    }
                    else {
                      vi.pixel_type = VideoInfo::CS_RGBP16;
                      biDst.biCompression = MAKEFOURCC('G', '3', 0, 16); // ffmpeg GBRP16LE
                      biDst.biBitCount = 48;
                      if (ICERR_OK == ICDecompressQuery(hic, pbiSrc, &biDst)) {
                        _RPT0(0, "AVISource: Opening as G3[0][16].\n");
                        bOpen = false;  // Skip further attempts
                      }
                      else {
                        vi.pixel_type = VideoInfo::CS_RGBAP10;
                        biDst.biSizeImage = vi.BMPSize();
                        biDst.biCompression = MAKEFOURCC('G', '4', 0, 10); // ffmpeg GBRAP10LE
                        biDst.biBitCount = 40; // FIXME: 4*10 = 40?
                        if (ICERR_OK == ICDecompressQuery(hic, pbiSrc, &biDst)) {
                          _RPT0(0, "AVISource: Opening as G4[0][10].\n");
                          bOpen = false;  // Skip further attempts
                        }
                        else {
                          vi.pixel_type = VideoInfo::CS_RGBAP12;
                          biDst.biCompression = MAKEFOURCC('G', '4', 0, 12); // ffmpeg GBRAP12LE
                          biDst.biBitCount = 48; // FIXME: 4*12 = 48?
                          if (ICERR_OK == ICDecompressQuery(hic, pbiSrc, &biDst)) {
                            _RPT0(0, "AVISource: Opening as G4[0][12].\n");
                            bOpen = false;  // Skip further attempts
                          }
                          else {
                            vi.pixel_type = VideoInfo::CS_RGBAP14;
                            biDst.biCompression = MAKEFOURCC('G', '4', 0, 14); // ffmpeg GBRAP14LE
                            biDst.biBitCount = 56; // FIXME: 4*14 = 56?
                            if (ICERR_OK == ICDecompressQuery(hic, pbiSrc, &biDst)) {
                              _RPT0(0, "AVISource: Opening as G4[0][14].\n");
                              bOpen = false;  // Skip further attempts
                            }
                            else {
                              vi.pixel_type = VideoInfo::CS_RGBAP16;
                              biDst.biCompression = MAKEFOURCC('G', '4', 0, 16); // ffmpeg GBRAP16LE
                              biDst.biBitCount = 64; // FIXME: 4*16 = 64?
                              if (ICERR_OK == ICDecompressQuery(hic, pbiSrc, &biDst)) {
                                _RPT0(0, "AVISource: Opening as G4[0][16].\n");
                                bOpen = false;  // Skip further attempts
                              }
                              else
                                if (forcedType) {
                                  env->ThrowError("AVISource: the video decompressor couldn't produce Planar RGB(A) output");
                                }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }

          // RGB32
          if (fRGB32 && bOpen) {
            vi.pixel_type = VideoInfo::CS_BGR32;
            biDst.biSizeImage = vi.BMPSize();
            biDst.biCompression = BI_RGB;
            biDst.biBitCount = 32;
            if (ICERR_OK == ICDecompressQuery(hic, pbiSrc, &biDst)) {
              _RPT0(0,"AVISource: Opening as RGB32.\n");
              bOpen = false;  // Skip further attempts
            } else if (forcedType) {
               env->ThrowError("AVISource: the video decompressor couldn't produce RGB32 output");
            }
          }

          // RGB24
          if (fRGB24 && bOpen) {
            vi.pixel_type = VideoInfo::CS_BGR24;
            biDst.biSizeImage = vi.BMPSize();
            biDst.biCompression = BI_RGB;
            biDst.biBitCount = 24;
            if (ICERR_OK == ICDecompressQuery(hic, pbiSrc, &biDst)) {
              _RPT0(0,"AVISource: Opening as RGB24.\n");
              bOpen = false;  // Skip further attempts
            } else if (forcedType) {
               env->ThrowError("AVISource: the video decompressor couldn't produce RGB24 output");
            }
          }

          // RGB48
          if (fRGB48 && bOpen) {
            if (setInternalFormat_BGR48(vi, hic, pbiSrc, &biDst, MAKEFOURCC('B', 'G', 'R', 48), VideoInfo::CS_BGR48)) {
              _RPT0(0, "AVISource: Opening as BGR0 (BGR[48]).\n");
              bOpen = false;  // Skip further attempts
            }
            else {
              // e.g. preferred by UT RGB 10 bits : UQRG
              if (setInternalFormat_b48r(vi, hic, pbiSrc, &biDst, MAKEFOURCC('b', '4', '8', 'r'), VideoInfo::CS_BGR48)) {
                _RPT0(0, "AVISource: Opening as b48r.\n");
                specf = AVI_SpecialFormats::b48r;
                bOpen = false;  // Skip further attempts
              }
              else if (forcedType) {
                env->ThrowError("AVISource: the video decompressor couldn't produce RGB48 output");
              }
            }
          }

          // RGB64
          if (fRGB64 && bOpen) {
            if (setInternalFormat_BRA64(vi, hic, pbiSrc, &biDst, MAKEFOURCC('B', 'R', 'A', 64), VideoInfo::CS_BGR64)) {
              _RPT0(0, "AVISource: Opening as BRA@ (BRA[64]).\n");
              bOpen = false;  // Skip further attempts
            }
            else {
              if (setInternalFormat_b64a(vi, hic, pbiSrc, &biDst, MAKEFOURCC('b', '6', '4', 'a'), VideoInfo::CS_BGR64)) {
                _RPT0(0, "AVISource: Opening as b64a.\n");
                bOpen = false;  // Skip further attempts
                specf = AVI_SpecialFormats::b64a;
              }
              else
                if (forcedType) {
                  env->ThrowError("AVISource: the video decompressor couldn't produce RGB64 output");
                }
            }
          }

          // Y8
          if ((fY8 || fGrayscale) && bOpen) {
            vi.pixel_type = VideoInfo::CS_Y8;
            biDst.biSizeImage = vi.BMPSize();
            biDst.biCompression = MAKEFOURCC('Y', '8', '0', '0');
            biDst.biBitCount = 8;
            if (ICERR_OK == ICDecompressQuery(hic, pbiSrc, &biDst)) {
              _RPT0(0,"AVISource: Opening as Y800.\n");
              bOpen = false;  // Skip further attempts
            } else {
              biDst.biCompression = MAKEFOURCC('Y', '8', ' ', ' ');
              if (ICERR_OK == ICDecompressQuery(hic, pbiSrc, &biDst)) {
                _RPT0(0,"AVISource: Opening as Y8.\n");
                bOpen = false;  // Skip further attempts
              } else {
                biDst.biCompression = MAKEFOURCC('G', 'R', 'E', 'Y');
                if (ICERR_OK == ICDecompressQuery(hic, pbiSrc, &biDst)) {
                  _RPT0(0,"AVISource: Opening as GREY.\n");
                  bOpen = false;  // Skip further attempts
                } else if (forcedType) {
                   env->ThrowError("AVISource: the video decompressor couldn't produce Y8 output");
                }
              }
            }
          }

          // Greyscale 'Y' no specific bit depth
          if (fGrayscale && bOpen) {
            vi.pixel_type = VideoInfo::CS_Y10;
            biDst.biSizeImage = vi.BMPSize();
            biDst.biCompression = MAKEFOURCC('Y', '1', 0, 10);
            biDst.biBitCount = 10;
            if (ICERR_OK == ICDecompressQuery(hic, pbiSrc, &biDst)) {
              _RPT0(0, "AVISource: Opening as Y1[0][10].\n");
              bOpen = false;  // Skip further attempts
            }
            else {
              vi.pixel_type = VideoInfo::CS_Y12;
              biDst.biSizeImage = vi.BMPSize();
              biDst.biCompression = MAKEFOURCC('Y', '1', 0, 12);
              biDst.biBitCount = 12;
              if (ICERR_OK == ICDecompressQuery(hic, pbiSrc, &biDst)) {
                _RPT0(0, "AVISource: Opening as Y1[0][12].\n");
                bOpen = false;  // Skip further attempts
              }
              else {
                vi.pixel_type = VideoInfo::CS_Y14;
                biDst.biSizeImage = vi.BMPSize();
                biDst.biCompression = MAKEFOURCC('Y', '1', 0, 14);
                biDst.biBitCount = 14;
                if (ICERR_OK == ICDecompressQuery(hic, pbiSrc, &biDst)) {
                  _RPT0(0, "AVISource: Opening as Y1[0][14].\n");
                  bOpen = false;  // Skip further attempts
                }
                else {
                  vi.pixel_type = VideoInfo::CS_Y16;
                  biDst.biSizeImage = vi.BMPSize();
                  biDst.biCompression = MAKEFOURCC('Y', '1', 0, 16);
                  biDst.biBitCount = 16;
                  if (ICERR_OK == ICDecompressQuery(hic, pbiSrc, &biDst)) {
                    _RPT0(0, "AVISource: Opening as Y1[0][16].\n");
                    bOpen = false;  // Skip further attempts
                  }
                  else
                    if (forcedType) {
                      env->ThrowError("AVISource: the video decompressor couldn't produce grayscale output");
                    }
                }
              }
            }
          }

          // YUV422P10
          if ((fYUV422P10 || fv210 || fP210) && bOpen) {
            vi.pixel_type = VideoInfo::CS_YUV422P10;
            if ((fv210 || fYUV422P10) && bOpen) {
              if (setInternalFormat_v210(vi, hic, pbiSrc, &biDst, MAKEFOURCC('v', '2', '1', '0'), VideoInfo::CS_YUV422P10)) {
                _RPT0(0, "AVISource: Opening as v210.\n");
                specf = AVI_SpecialFormats::v210;
                bOpen = false;  // Skip further attempts
              }
            }
            if ((fP210 || fYUV422P10) && bOpen) {
              if (setInternalFormat_P210(vi, hic, pbiSrc, &biDst, MAKEFOURCC('P', '2', '1', '0'), VideoInfo::CS_YUV422P10)) {
                _RPT0(0, "AVISource: Opening as P210.\n");
                specf = AVI_SpecialFormats::P210;
                bOpen = false;  // Skip further attempts
              }
            }
            else if (forcedType && bOpen) {
              env->ThrowError("AVISource: the video decompressor couldn't produce YUV422P10 output");
            }
          }

          // YUV422P16
          if ((fYUV422P16 || fP216) && bOpen) {
            vi.pixel_type = VideoInfo::CS_YUV422P16;
            if (setInternalFormat_P216(vi, hic, pbiSrc, &biDst, MAKEFOURCC('P', '2', '1', '6'), VideoInfo::CS_YUV422P16)) {
              _RPT0(0, "AVISource: Opening as P216.\n");
              specf = AVI_SpecialFormats::P216;
              bOpen = false;  // Skip further attempts
            }
            else if (forcedType) {
              env->ThrowError("AVISource: the video decompressor couldn't produce YUV422P16 output");
            }
          }

          // YUV420P10
          if ((fYUV420P10 || fP010) && bOpen) {
            vi.pixel_type = VideoInfo::CS_YUV420P10;
            if (setInternalFormat_P010(vi, hic, pbiSrc, &biDst, MAKEFOURCC('P', '0', '1', '0'), VideoInfo::CS_YUV420P10)) {
              _RPT0(0, "AVISource: Opening as P010.\n");
              specf = AVI_SpecialFormats::P010;
              bOpen = false;  // Skip further attempts
            }
            else if (forcedType) {
              env->ThrowError("AVISource: the video decompressor couldn't produce YUV420P10 output");
            }
          }

          // YUV420P16
          if ((fYUV420P16 || fP016) && bOpen) {
            vi.pixel_type = VideoInfo::CS_YUV420P16;
            if (setInternalFormat_P016(vi, hic, pbiSrc, &biDst, MAKEFOURCC('P', '0', '1', '6'), VideoInfo::CS_YUV420P16)) {
              _RPT0(0, "AVISource: Opening as P016.\n");
              specf = AVI_SpecialFormats::P016;
              bOpen = false;  // Skip further attempts
            }
            else if (forcedType) {
              env->ThrowError("AVISource: the video decompressor couldn't produce YUV420P16 output");
            }
          }

          // YUV444P16
          if ((fYUV444P16 || fY416) && bOpen) {
            vi.pixel_type = VideoInfo::CS_YUV444P16;
            if (setInternalFormat_Y416(vi, hic, pbiSrc, &biDst, MAKEFOURCC('Y', '4', '1', '6'), VideoInfo::CS_YUV444P16)) {
              _RPT0(0, "AVISource: Opening as Y416.\n");
              specf = AVI_SpecialFormats::Y416;
              bOpen = false;  // Skip further attempts
            }
            else if (forcedType) {
              env->ThrowError("AVISource: the video decompressor couldn't produce YUV444P16 output");
            }
          }

          // YUV444P10
          if ((fYUV444P10 || fv410 || fY410) && bOpen) {
            vi.pixel_type = VideoInfo::CS_YUV444P10;
            if ((fYUV444P10 || fv410) && setInternalFormat_v410(vi, hic, pbiSrc, &biDst, MAKEFOURCC('v', '4', '1', '0'), VideoInfo::CS_YUV444P10)) {
              _RPT0(0, "AVISource: Opening as v410.\n");
              specf = AVI_SpecialFormats::v410;
              bOpen = false;  // Skip further attempts
            }
            else if ((fYUV444P10 || fY410) && setInternalFormat_Y410(vi, hic, pbiSrc, &biDst, MAKEFOURCC('Y', '4', '1', '0'), VideoInfo::CS_YUVA444P10)) {
                _RPT0(0, "AVISource: Opening as Y410.\n");
                specf = AVI_SpecialFormats::Y410;
                bOpen = false;  // Skip further attempts
            }
            else if (forcedType) {
              env->ThrowError("AVISource: the video decompressor couldn't produce YUV444P10 output");
            }
          }

          // v308 to YUV444P8 aka YV24
          if ((fv308) && bOpen) {
            vi.pixel_type = VideoInfo::CS_YV24;
            if (setInternalFormat_v308(vi, hic, pbiSrc, &biDst, MAKEFOURCC('v', '3', '0', '8'), VideoInfo::CS_YV24)) {
              _RPT0(0, "AVISource: Opening as v308.\n");
              specf = AVI_SpecialFormats::v308;
              bOpen = false;  // Skip further attempts
            }
            else if (forcedType) {
              env->ThrowError("AVISource: the video decompressor couldn't produce v308 output");
            }
          }

          // v408 to YUVA444P8
          if ((fv408) && bOpen) {
            vi.pixel_type = VideoInfo::CS_YUVA444;
            if (setInternalFormat_v408(vi, hic, pbiSrc, &biDst, MAKEFOURCC('v', '4', '0', '8'), VideoInfo::CS_YUVA444)) {
              _RPT0(0, "AVISource: Opening as v408.\n");
              specf = AVI_SpecialFormats::v408;
              bOpen = false;  // Skip further attempts
            }
            else if (forcedType) {
              env->ThrowError("AVISource: the video decompressor couldn't produce v408 output");
            }
          }

          // No takers!
          if (bOpen)
            env->ThrowError("AviSource: Could not open video stream in any supported format.");

          DecompressBegin(pbiSrc, &biDst);
        }
        else {
          // no hic

          // when we reach here, maybe a natively supported fourCC format was reported
          if (pbiSrc->biCompression == MAKEFOURCC('b', '6', '4', 'a')) {
            if (setInternalFormat_b64a(vi, hic, pbiSrc, &biDst, pbiSrc->biCompression, VideoInfo::CS_BGR64)) {
              _RPT0(0, "AVISource: Opening as b64a.\n");
              specf = AVI_SpecialFormats::b64a; // special flag!
            }
          }
          if (pbiSrc->biCompression == MAKEFOURCC('B', 'R', 'A', 64)) {
            if (setInternalFormat_BRA64(vi, hic, pbiSrc, &biDst, pbiSrc->biCompression, VideoInfo::CS_BGR64)) {
              _RPT0(0, "AVISource: Opening as BRA@.\n");
            }
          }
          if (pbiSrc->biCompression == MAKEFOURCC('B', 'G', 'R', 48)) {
            if (setInternalFormat_BGR48(vi, hic, pbiSrc, &biDst, pbiSrc->biCompression, VideoInfo::CS_BGR48)) {
              _RPT0(0, "AVISource: Opening as BGR[48].\n");
            }
          }
          if (pbiSrc->biCompression == MAKEFOURCC('b', '4', '8', 'r')) {
            if (setInternalFormat_b48r(vi, hic, pbiSrc, &biDst, pbiSrc->biCompression, VideoInfo::CS_BGR48)) {
              _RPT0(0, "AVISource: Opening as b48r.\n");
              specf = AVI_SpecialFormats::b48r; // special flag!
            }
          }
          if (pbiSrc->biCompression == MAKEFOURCC('v', '2', '1', '0')) {
            if (setInternalFormat_v210(vi, hic, pbiSrc, &biDst, pbiSrc->biCompression, VideoInfo::CS_YUV422P10)) {
              _RPT0(0, "AVISource: Opening as v210.\n");
              specf = AVI_SpecialFormats::v210; // special flag!
            }
          }
          if (pbiSrc->biCompression == MAKEFOURCC('P', '2', '1', '0')) {
            if (setInternalFormat_P210(vi, hic, pbiSrc, &biDst, pbiSrc->biCompression, VideoInfo::CS_YUV422P10)) {
              _RPT0(0, "AVISource: Opening as P210.\n");
              specf = AVI_SpecialFormats::P210;
            }
          }
          if (pbiSrc->biCompression == MAKEFOURCC('P', '0', '1', '0')) {
            if (setInternalFormat_P010(vi, hic, pbiSrc, &biDst, pbiSrc->biCompression, VideoInfo::CS_YUV420P10)) {
              _RPT0(0, "AVISource: Opening as P010.\n");
              specf = AVI_SpecialFormats::P010;
            }
          }
          if (pbiSrc->biCompression == MAKEFOURCC('P', '0', '1', '6')) {
            if (setInternalFormat_P016(vi, hic, pbiSrc, &biDst, pbiSrc->biCompression, VideoInfo::CS_YUV420P16)) {
              _RPT0(0, "AVISource: Opening as P016.\n");
              specf = AVI_SpecialFormats::P016;
            }
          }
          if (pbiSrc->biCompression == MAKEFOURCC('P', '2', '1', '6')) {
            if (setInternalFormat_P216(vi, hic, pbiSrc, &biDst, pbiSrc->biCompression, VideoInfo::CS_YUV422P16)) {
              _RPT0(0, "AVISource: Opening as P216.\n");
              specf = AVI_SpecialFormats::P216;
            }
          }
          if (pbiSrc->biCompression == MAKEFOURCC('v', '4', '1', '0')) {
            if (setInternalFormat_v410(vi, hic, pbiSrc, &biDst, pbiSrc->biCompression, VideoInfo::CS_YUV444P10)) {
              _RPT0(0, "AVISource: Opening as v410.\n");
              specf = AVI_SpecialFormats::v410;
            }
          }
          if (pbiSrc->biCompression == MAKEFOURCC('Y', '4', '1', '0')) {
            if (setInternalFormat_Y410(vi, hic, pbiSrc, &biDst, pbiSrc->biCompression, VideoInfo::CS_YUVA444P10)) {
              _RPT0(0, "AVISource: Opening as Y410.\n");
              specf = AVI_SpecialFormats::Y410;
            }
          }
          if (pbiSrc->biCompression == MAKEFOURCC('Y', '4', '1', '6')) {
            if (setInternalFormat_Y416(vi, hic, pbiSrc, &biDst, pbiSrc->biCompression, VideoInfo::CS_YUV444P16)) {
              _RPT0(0, "AVISource: Opening as Y416.\n");
              specf = AVI_SpecialFormats::Y416;
            }
          }
          if (pbiSrc->biCompression == MAKEFOURCC('r', '2', '1', '0')) {
            if (setInternalFormat_r210(vi, hic, pbiSrc, &biDst, pbiSrc->biCompression, VideoInfo::CS_RGBP10)) {
              _RPT0(0, "AVISource: Opening as r210.\n");
              specf = AVI_SpecialFormats::r210;
            }
          }
          if (pbiSrc->biCompression == MAKEFOURCC('R', '1', '0', 'k')) {
            if (setInternalFormat_R10k(vi, hic, pbiSrc, &biDst, pbiSrc->biCompression, VideoInfo::CS_RGBP10)) {
              _RPT0(0, "AVISource: Opening as R10k.\n");
              specf = AVI_SpecialFormats::R10k;
            }
          }
          if (pbiSrc->biCompression == MAKEFOURCC('v', '3', '0', '8')) {
            if (setInternalFormat_v308(vi, hic, pbiSrc, &biDst, pbiSrc->biCompression, VideoInfo::CS_YV24)) {
              _RPT0(0, "AVISource: Opening as v308.\n");
              specf = AVI_SpecialFormats::v308;
            }
          }
          if (pbiSrc->biCompression == MAKEFOURCC('v', '4', '0', '8')) {
            if (setInternalFormat_v408(vi, hic, pbiSrc, &biDst, pbiSrc->biCompression, VideoInfo::CS_YUVA444)) {
              _RPT0(0, "AVISource: Opening as v408.\n");
              specf = AVI_SpecialFormats::v408;
            }
          }
        }
        // Flip DIB formats if negative height (FIXME: Y8 too?). Flip RGB48/64 always.
        if ((pbiSrc->biHeight < 0 && (vi.IsRGB24() || vi.IsRGB32() || vi.IsY8())) || vi.IsRGB48() || vi.IsRGB64())
          bInvertFrames = true;
      }
      else {
        env->ThrowError("AviSource: Could not locate video stream.");
      }
    }

    // check for audio stream
    if (fAudio) /*  && pfile->GetStream(streamtypeAUDIO, 0)) */ {
      aSrc = new AudioSourceAVI(pfile, true, atrack);
      if (aSrc->init()) {
          audioStreamSource = new AudioStreamSource(aSrc,
                                                    aSrc->lSampleFirst,
                                                    aSrc->lSampleLast - aSrc->lSampleFirst,
                                                    true);
          WAVEFORMATEX* pwfx;
          pwfx = audioStreamSource->GetFormat();
          vi.audio_samples_per_second = pwfx->nSamplesPerSec;
          vi.nchannels = pwfx->nChannels;
          if (pwfx->wFormatTag == WAVE_FORMAT_IEEE_FLOAT) {
            vi.sample_type = SAMPLE_FLOAT;
          } else if (pwfx->wBitsPerSample == 16) {
            vi.sample_type = SAMPLE_INT16;
          } else if (pwfx->wBitsPerSample == 8) {
            vi.sample_type = SAMPLE_INT8;
          } else if (pwfx->wBitsPerSample == 24) {
            vi.sample_type = SAMPLE_INT24;
          } else if (pwfx->wBitsPerSample == 32) {
            vi.sample_type = SAMPLE_INT32;
          }
          vi.num_audio_samples = audioStreamSource->GetLength();

          audio_stream_pos = 0;
        }
    }

    // try to decompress frame 0 if not audio only.

    dropped_frame=false;

    if (mode != MODE_WAV) {
      bMediaPad = !(!bMediaPad && !vi.IsY8() && vi.IsPlanar());
      int keyframe = pvideo->NearestKeyFrame(0);
      frame = new TemporalBuffer(vi, bMediaPad, specf, env);

      LRESULT error = DecompressFrame(keyframe, false, env);
      if (error != ICERR_OK)   // shutdown, if init not succesful.
        env->ThrowError("AviSource: Could not decompress frame 0");

      // Cope with dud AVI files that start with drop
      // frames, just return the first key frame
      if (dropped_frame) {
        keyframe = pvideo->NextKeyFrame(0);
        error = DecompressFrame(keyframe, false, env);
        if (error != ICERR_OK)   // shutdown, if init not succesful.
          env->ThrowError("AviSource: Could not decompress first keyframe %d", keyframe);
      }
      last_frame_no=0;
      last_frame = AdjustFrameAlignment(frame, vi, bInvertFrames, specf, env);
    }
  }
  catch (...) {
    AVISource::CleanUp();
    throw;
  }
}

AVISource::~AVISource() {
  AVISource::CleanUp();
}

void AVISource::CleanUp() { // Tritical - Jan 2006
  if (hic) {
    !ex ? ICDecompressEnd(hic) : ICDecompressExEnd(hic);
    ICClose(hic);
  }
  if (pvideo) delete pvideo;
  if (aSrc) delete aSrc;
  if (audioStreamSource) delete audioStreamSource;
  if (pfile)
    pfile->Release();
  AVIFileExit();
  if (pbiSrc)
    free(pbiSrc);
  if (srcbuffer)  // Tritical May 2005
    delete[] srcbuffer;
  if (frame)
    delete frame;
}

const VideoInfo& AVISource::GetVideoInfo() { return vi; }

PVideoFrame AVISource::GetFrame(int n, IScriptEnvironment* env) {

  n = clamp(n, 0, vi.num_frames-1);
  dropped_frame=false;
  if (n != last_frame_no) {
    // find the last keyframe
    int keyframe = pvideo->NearestKeyFrame(n);
    // maybe we don't need to go back that far
    if (last_frame_no < n && last_frame_no >= keyframe)
      keyframe = last_frame_no+1;
    if (keyframe < 0) keyframe = 0;

    bool frameok = false;
    //PVideoFrame frame = env->NewVideoFrameP(vi, -4);

    bool not_found_yet;
    do {
      not_found_yet=false;
      for (int i = keyframe; i <= n; ++i) {
        LRESULT error = DecompressFrame(i, i != n, env);
        if ((!dropped_frame) && (error == ICERR_OK)) frameok = true;   // Better safe than sorry
      }
      last_frame_no = n;

      if (!last_frame && !frameok) {  // Last keyframe was not valid.
        const int key_pre=keyframe;
        keyframe = pvideo->NearestKeyFrame(keyframe-1);
        if (keyframe < 0) keyframe = 0;
        if (keyframe == key_pre)
          env->ThrowError("AVISource: could not find valid keyframe for frame %d.", n);

        not_found_yet=true;
      }
    } while(not_found_yet);

    if (frameok) {
      last_frame = AdjustFrameAlignment(frame, vi, bInvertFrames, specf, env);
    }
  }
  return last_frame;
}

void AVISource::GetAudio(void* buf, __int64 start, __int64 count, IScriptEnvironment* env) {
  AVS_UNUSED(env);
  long bytes_read=0;
  __int64 samples_read=0;

  if (start < 0) {
    int bytes = (int)vi.BytesFromAudioSamples(min(-start, count));
    memset(buf, 0, bytes);
    buf = (char*)buf + bytes;
    count -= vi.AudioSamplesFromBytes(bytes);
    start += vi.AudioSamplesFromBytes(bytes);
  }

  if (audioStreamSource) {
    if (start != audio_stream_pos)
        audioStreamSource->Seek((long)start);
    samples_read = audioStreamSource->Read(buf, (long)count, &bytes_read);
    audio_stream_pos = start + samples_read;
  }

  if (samples_read < count)
    memset((char*)buf + bytes_read, 0, (size_t)(vi.BytesFromAudioSamples(count) - bytes_read));
}

bool AVISource::GetParity(int n) {
  AVS_UNUSED(n);
  return false;
}

int __stdcall AVISource::SetCacheHints(int cachehints,int frame_range)
{
  AVS_UNUSED(frame_range);
  switch(cachehints)
  {
  case CACHE_GET_MTMODE:
    return MT_SERIALIZED;
  default:
    return 0;
  }
}
