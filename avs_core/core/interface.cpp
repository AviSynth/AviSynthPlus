// Avisynth v2.5.  Copyright 2002, 2005 Ben Rudiak-Gould et al.
// Avisynth v2.6.  Copyright 2006 Klaus Post.
// Avisynth v2.6.  Copyright 2009 Ian Brabham.
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


/* Maintenance notes :-
 *
 *   All the code here was formally baked into the avisynth.h interface.
 *
 *   Whenever you modify any code here please keep and mark the original
 *   code with a block comment beginning with the word "Baked"
 *
 *   Be mindful with changes you do make that previous 2.5 plugins will
 *   still have the original code active. This may require other defensive
 *   or mitigating code elsewhere.
 */

#include <avisynth.h>

#ifdef AVS_WINDOWS
    #include <avs/win.h>
#else
    #include <avs/posix.h>
#endif
#include "InternalEnvironment.h"
#include "DeviceManager.h"
#include "AVSMap.h"
#include "function.h"
#include "assert.h"

/**********************************************************************/

// struct VideoInfo

// useful functions of the above
bool VideoInfo::HasVideo() const { return (width!=0); }
bool VideoInfo::HasAudio() const { return (audio_samples_per_second!=0); }
bool VideoInfo::IsRGB() const { return !!(pixel_type&CS_BGR); }
bool VideoInfo::IsRGB24() const { return ((pixel_type & CS_BGR24) == CS_BGR24) && ((pixel_type & CS_Sample_Bits_Mask) == CS_Sample_Bits_8); } // Clear out additional properties
bool VideoInfo::IsRGB32() const { return ((pixel_type & CS_BGR32) == CS_BGR32) && ((pixel_type & CS_Sample_Bits_Mask) == CS_Sample_Bits_8); }
bool VideoInfo::IsYUV() const { return !!(pixel_type&CS_YUV ); }
bool VideoInfo::IsYUY2() const { return (pixel_type & CS_YUY2) == CS_YUY2; }

bool VideoInfo::IsYV24()  const { return (pixel_type & CS_PLANAR_MASK) == (CS_YV24  & CS_PLANAR_FILTER); }
bool VideoInfo::IsYV16()  const { return (pixel_type & CS_PLANAR_MASK) == (CS_YV16  & CS_PLANAR_FILTER); }
bool VideoInfo::IsYV12()  const { return (pixel_type & CS_PLANAR_MASK) == (CS_YV12  & CS_PLANAR_FILTER); }
bool VideoInfo::IsY8()    const { return (pixel_type & CS_PLANAR_MASK) == (CS_Y8    & CS_PLANAR_FILTER); }

bool VideoInfo::IsYV411() const { return (pixel_type & CS_PLANAR_MASK) == (CS_YV411 & CS_PLANAR_FILTER); }
//bool VideoInfo::IsYUV9()  const { return (pixel_type & CS_PLANAR_MASK) == (CS_YUV9  & CS_PLANAR_FILTER); }

/* Baked ********************
bool VideoInfo::IsColorSpace(int c_space) const { return ((pixel_type & c_space) == c_space); }
   Baked ********************/
bool VideoInfo::IsColorSpace(int c_space) const {
  // This function will check if the colorspace (VideoInfo.pixel_type) is the same as given c_space (or more general it checks for a Colorspace property (see avisynth.h)).
  // So it's not only for exact colorspace comparison but checking properties also.
  return IsPlanar() ? ((pixel_type & CS_PLANAR_MASK) == (c_space & CS_PLANAR_FILTER)) :
    // support exact individual flag checking
    c_space == CS_YUVA ? ((c_space & CS_YUVA) == CS_YUVA) :
    c_space == CS_BGR ? ((c_space & CS_BGR) == CS_BGR) :
    c_space == CS_YUV ? ((c_space & CS_YUV) == CS_YUV) :
    c_space == CS_INTERLEAVED ? ((c_space & CS_INTERLEAVED) == CS_INTERLEAVED) :
  // RGB got sample bits:
  // In order to work: RGB64.IsColorSpace(RGB32) => false, or else we get true (RGB64 & RGB32 == RGB32)
  // Simple ((pixel_type & c_space) == c_space) would not work, does not take into account bit depth
    ( ((pixel_type & ~CS_Sample_Bits_Mask & c_space) == (c_space & ~CS_Sample_Bits_Mask)) &&
        ((pixel_type & CS_Sample_Bits_Mask) == (c_space & CS_Sample_Bits_Mask)) );
}

bool VideoInfo::Is(int property) const { return ((image_type & property)==property ); }
bool VideoInfo::IsPlanar() const { return !!(pixel_type & CS_PLANAR); }
bool VideoInfo::IsFieldBased() const { return !!(image_type & IT_FIELDBASED); }
bool VideoInfo::IsParityKnown() const { return ((image_type & IT_FIELDBASED)&&(image_type & (IT_BFF|IT_TFF))); }
bool VideoInfo::IsBFF() const { return !!(image_type & IT_BFF); }
bool VideoInfo::IsTFF() const { return !!(image_type & IT_TFF); }

/* Baked ********************
bool VideoInfo::IsVPlaneFirst() const {return ((pixel_type & CS_YV12) == CS_YV12); }  // Don't use this
int VideoInfo::BytesFromPixels(int pixels) const { return pixels * (BitsPerPixel()>>3); }   // Will not work on planar images, but will return only luma planes
int VideoInfo::RowSize() const { return BytesFromPixels(width); }  // Also only returns first plane on planar images
int VideoInfo::BMPSize() const { if (IsPlanar()) {int p = height * ((RowSize()+3) & ~3); p+=p>>1; return p;  } return height * ((RowSize()+3) & ~3); }
int64_t VideoInfo::AudioSamplesFromFrames(int64_t frames) const { return (fps_numerator && HasVideo()) ? ((int64_t)(frames) * audio_samples_per_second * fps_denominator / fps_numerator) : 0; }
   Baked ********************/
int64_t VideoInfo::AudioSamplesFromFrames(int frames) const { return (fps_numerator && HasVideo()) ? ((int64_t)(frames) * audio_samples_per_second * fps_denominator / fps_numerator) : 0; }
int VideoInfo::FramesFromAudioSamples(int64_t samples) const { return (fps_denominator && HasAudio()) ? (int)((samples * fps_numerator)/((int64_t)fps_denominator * audio_samples_per_second)) : 0; }
int64_t VideoInfo::AudioSamplesFromBytes(int64_t bytes) const { return HasAudio() ? bytes / BytesPerAudioSample() : 0; }
int64_t VideoInfo::BytesFromAudioSamples(int64_t samples) const { return samples * BytesPerAudioSample(); }
int VideoInfo::AudioChannels() const { return HasAudio() ? nchannels : 0; }
int VideoInfo::SampleType() const{ return sample_type;}
bool VideoInfo::IsSampleType(int testtype) const{ return !!(sample_type&testtype);}
int VideoInfo::SamplesPerSecond() const { return audio_samples_per_second; }
int VideoInfo::BytesPerAudioSample() const { return nchannels*BytesPerChannelSample();}
void VideoInfo::SetFieldBased(bool isfieldbased)  { if (isfieldbased) image_type|=IT_FIELDBASED; else  image_type&=~IT_FIELDBASED; }
void VideoInfo::Set(int property)  { image_type|=property; }
void VideoInfo::Clear(int property)  { image_type&=~property; }

/* Baked ********************
int VideoInfo::BitsPerPixel() const {
  switch (pixel_type) {
    case CS_BGR24:
      return 24;
    case CS_BGR32:
      return 32;
    case CS_YUY2:
      return 16;
    case CS_YV12:
    case CS_I420:
      return 12;
    default:
      return 0;
  }
}
   Baked ********************/

int VideoInfo::BytesPerChannelSample() const {
  switch (sample_type) {
  case SAMPLE_INT8:
    return sizeof(unsigned char);
  case SAMPLE_INT16:
    return sizeof(signed short);
  case SAMPLE_INT24:
    return 3;
  case SAMPLE_INT32:
    return sizeof(signed int);
  case SAMPLE_FLOAT:
    return sizeof(SFLOAT);
  default:
    _ASSERTE("Audio sample type not recognized!");
    return 0;
  }
}

bool VideoInfo::IsVPlaneFirst() const {
  return (NumComponents() > 1) && IsPlanar() && (pixel_type & (CS_VPlaneFirst | CS_UPlaneFirst)) == CS_VPlaneFirst;   // Shouldn't use this
}

int VideoInfo::BytesFromPixels(int pixels) const {
    const int componentSizes[8] = {1,2,4,0,0,2,2,2};
    return (NumComponents() > 1) && IsPlanar() ? pixels * componentSizes[(pixel_type>>CS_Shift_Sample_Bits) & 7] : pixels * (BitsPerPixel()>>3);
    // For planar images, will return luma plane
}

int VideoInfo::RowSize(int plane) const {
  const int rowsize = BytesFromPixels(width);
  switch (plane) {
    case PLANAR_U: case PLANAR_V:
      return ((NumComponents() > 1) && IsPlanar() && !IsPlanarRGB() && !IsPlanarRGBA()) ? rowsize>>GetPlaneWidthSubsampling(plane) : 0;

    case PLANAR_U_ALIGNED: case PLANAR_V_ALIGNED:
      return ((NumComponents() > 1) && IsPlanar() && !IsPlanarRGB() && !IsPlanarRGBA()) ? ((rowsize>>GetPlaneWidthSubsampling(plane))+FRAME_ALIGN-1)&(~(FRAME_ALIGN-1)) : 0; // Aligned rowsize

    case PLANAR_Y_ALIGNED:
      return (rowsize+FRAME_ALIGN-1)&(~(FRAME_ALIGN-1)); // Aligned rowsize

    case PLANAR_R: case PLANAR_G: case PLANAR_B:
        return ((NumComponents() > 1) && (IsPlanarRGB() || IsPlanarRGBA())) ? rowsize : 0;

    case PLANAR_R_ALIGNED: case PLANAR_G_ALIGNED: case PLANAR_B_ALIGNED:
        return IsPlanarRGB() || IsPlanarRGBA() ? (rowsize+FRAME_ALIGN-1)&(~(FRAME_ALIGN-1)) : 0; // Aligned rowsize

    case PLANAR_A:
        return ((NumComponents() == 4) && IsPlanar()) ? rowsize : 0;

    case PLANAR_A_ALIGNED:
        return ((NumComponents() == 4) && IsPlanar()) ? (rowsize+FRAME_ALIGN-1)&(~(FRAME_ALIGN-1)) : 0; // Aligned rowsize

  }
  return rowsize;
}

int VideoInfo::BMPSize() const {
  if ((NumComponents() > 1) && IsPlanar()) {
    if (IsPlanarRGB() || IsPlanarRGBA()) {
      return ((RowSize(PLANAR_G) + 3) & ~3) * height * NumComponents(); // does it make sense for planar rgb?
    }
    // Y plane
    const int Ybytes  = ((RowSize(PLANAR_Y)+3) & ~3) * height;
    const int UVbytes = Ybytes >> (GetPlaneWidthSubsampling(PLANAR_U)+GetPlaneHeightSubsampling(PLANAR_U));
    return Ybytes + UVbytes*2;
  }
  return height * ((RowSize()+3) & ~3);
}

int VideoInfo::GetPlaneWidthSubsampling(int plane) const {  // Subsampling in bitshifts!
  if (plane == PLANAR_Y || plane == PLANAR_R || plane == PLANAR_G || plane == PLANAR_B || plane == PLANAR_A)  // No subsampling
    return 0;
  if (NumComponents() == 1)
    throw AvisynthError("Filter error: GetPlaneWidthSubsampling not available on greyscale pixel type.");
  if (plane == PLANAR_U || plane == PLANAR_V) {
    if (IsYUY2())
      return 1;
    else if (IsPlanar())
      return ((pixel_type>>CS_Shift_Sub_Width)+1) & 3;
    else
      throw AvisynthError("Filter error: GetPlaneWidthSubsampling called with unsupported pixel type.");
  }
  throw AvisynthError("Filter error: GetPlaneWidthSubsampling called with unsupported plane.");
}

int VideoInfo::GetPlaneHeightSubsampling(int plane) const {  // Subsampling in bitshifts!
  if (plane == PLANAR_Y || plane == PLANAR_R || plane == PLANAR_G || plane == PLANAR_B || plane == PLANAR_A)  // No subsampling
    return 0;
  if (NumComponents() == 1)
    throw AvisynthError("Filter error: GetPlaneHeightSubsampling not available on greyscale pixel type.");
  if (plane == PLANAR_U || plane == PLANAR_V) {
    if (IsYUY2())
      return 0;
    else if (IsPlanar())
      return ((pixel_type>>CS_Shift_Sub_Height)+1) & 3;
    else
      throw AvisynthError("Filter error: GetPlaneHeightSubsampling called with unsupported pixel type.");
  }
  throw AvisynthError("Filter error: GetPlaneHeightSubsampling called with unsupported plane.");
}

int VideoInfo::BitsPerPixel() const {
// Lookup Interleaved, calculate PLANAR's
// Remark:
// - total bitsize for interleaved/packed types, e.g. RGB48 = 48 = 3x16
// - byte size corrected with U-V subsampling factor for Planar YUV/YUVA
// - external softwares may use it for calculating buffer size
// - use BitsPerComponent instead for returning the pixel component format 8/10/12/14/16/32 bits
    switch (pixel_type) {
      case CS_BGR24:
        return 24;
      case CS_BGR32:
        return 32;
      case CS_YUY2:
        return 16;
      case CS_Y8:
        return 8;
      case CS_Y10:
      case CS_Y12:
      case CS_Y14:
      case CS_Y16: // AVS16
        return 16;
      case CS_Y32:
        return 32;
      case CS_BGR48:
        return 48;
      case CS_BGR64:
        return 64;
    }
    if (IsPlanar()) {
      const int componentSizes[8] = {1,2,4,0,0,2,2,2};
      const int S = (IsYUV() || IsYUVA()) ? GetPlaneWidthSubsampling(PLANAR_U) + GetPlaneHeightSubsampling(PLANAR_U) : 0; // planar RGBA: no subsampling
      const int fullSizePlaneCount = IsYUVA() || IsPlanarRGBA() ? 2 : 1; // alpha plane is always full size
      return (((fullSizePlaneCount << S) + 2) * (componentSizes[(pixel_type >> CS_Shift_Sample_Bits) & 7]) * 8) >> S;
    }
    return 0;
}

// useful mutator
void VideoInfo::SetFPS(unsigned numerator, unsigned denominator) {
  if ((numerator == 0) || (denominator == 0)) {
    fps_numerator = 0;
    fps_denominator = 1;
  }
  else {
    unsigned x=numerator, y=denominator;
    while (y) {   // find gcd
      unsigned t = x%y; x = y; y = t;
    }
    fps_numerator = numerator/x;
    fps_denominator = denominator/x;
  }
}

// Range protected multiply-divide of FPS
void VideoInfo::MulDivFPS(unsigned multiplier, unsigned divisor) {
  uint64_t numerator   = UInt32x32To64(fps_numerator,   multiplier);
  uint64_t denominator = UInt32x32To64(fps_denominator, divisor);

  uint64_t x=numerator, y=denominator;
  while (y) {   // find gcd
    uint64_t t = x%y; x = y; y = t;
  }
  numerator   /= x; // normalize
  denominator /= x;

  uint64_t temp = numerator | denominator; // Just looking top bit
  unsigned u = 0;
  while (temp & 0xffffffff80000000) { // or perhaps > 16777216*2
    temp = Int64ShrlMod32(temp, 1);
    u++;
  }
  if (u) { // Scale to fit
    const unsigned round = 1 << (u-1);
    SetFPS( (unsigned)Int64ShrlMod32(numerator   + round, u),
            (unsigned)Int64ShrlMod32(denominator + round, u) );
  }
  else {
    fps_numerator   = (unsigned)numerator;
    fps_denominator = (unsigned)denominator;
  }
}

// Test for same colorspace
bool VideoInfo::IsSameColorspace(const VideoInfo& vi) const {
  if (vi.pixel_type == pixel_type) return TRUE;
  if (IsYV12() && vi.IsYV12()) return TRUE;
  return FALSE;
}

int VideoInfo::NumComponents() const {

  switch (pixel_type) {
  case CS_UNKNOWN:
     return 0;
  case CS_RAW32:
  case CS_Y8:
  case CS_Y10:
  case CS_Y12:
  case CS_Y14:
  case CS_Y16:
  case CS_Y32:
    return 1;
  case CS_BGR32:
  case CS_BGR64:
    return 4; // these are not planar but return the count
  default:
      return (IsYUVA() || IsPlanarRGBA()) ? 4 : 3;
      // 3: YUV, planar RGB, 24/48 bit RGB
      // 4: YUVA and PlanarRGBA
  }
}

int VideoInfo::ComponentSize() const {
// occupied bytes per one pixel component
  if(IsPlanar()) {
    const int componentSizes[8] = {1,2,4,0,0,2,2,2};
    return componentSizes[(pixel_type >> CS_Shift_Sample_Bits) & 7];
  }
  switch (pixel_type) {
  case CS_UNKNOWN:
    return 0;
  case CS_RAW32:
    return 4;
  case CS_BGR48:
  case CS_BGR64:
    return 2;
  default: // YUY2, packed 8 bit BGRs
    return 1;
  }
}

int VideoInfo::BitsPerComponent() const {
// unlike BitsPerPixel, this returns the real
// component size as 8/10/12/14/16/32 bit
    if (pixel_type == CS_YUY2)
        return 8;
    else if (pixel_type == CS_RAW32)
        return 32;
    // planar YUV/RGB and packed RGB
    const int componentBitSizes[8] = {8,16,32,0,0,10,12,14};
    return componentBitSizes[(pixel_type >> CS_Shift_Sample_Bits) & 7];
}

// bit-depth independent helper functions instead of 8 bit specific IsYV24/16/12/8
bool VideoInfo::Is444()  const { return ((pixel_type & CS_PLANAR_MASK & ~CS_Sample_Bits_Mask) == (CS_GENERIC_YUV444 & CS_PLANAR_FILTER)) ||
                                        ((pixel_type & CS_PLANAR_MASK & ~CS_Sample_Bits_Mask) == (CS_GENERIC_YUVA444 & CS_PLANAR_FILTER)) ; }
bool VideoInfo::Is422()  const { return ((pixel_type & CS_PLANAR_MASK & ~CS_Sample_Bits_Mask) == (CS_GENERIC_YUV422 & CS_PLANAR_FILTER)) ||
                                        ((pixel_type & CS_PLANAR_MASK & ~CS_Sample_Bits_Mask) == (CS_GENERIC_YUVA422 & CS_PLANAR_FILTER)); }
bool VideoInfo::Is420()  const { return ((pixel_type & CS_PLANAR_MASK & ~CS_Sample_Bits_Mask) == (CS_GENERIC_YUV420 & CS_PLANAR_FILTER)) ||
                                        ((pixel_type & CS_PLANAR_MASK & ~CS_Sample_Bits_Mask) == (CS_GENERIC_YUVA420 & CS_PLANAR_FILTER)); }
bool VideoInfo::IsY()       const { return (pixel_type & CS_PLANAR_MASK & ~CS_Sample_Bits_Mask) == (CS_GENERIC_Y      & CS_PLANAR_FILTER); }
bool VideoInfo::IsRGB48()   const { return ((pixel_type & CS_BGR24) == CS_BGR24) && ((pixel_type & CS_Sample_Bits_Mask) == CS_Sample_Bits_16); } // Clear out additional properties
bool VideoInfo::IsRGB64()   const { return ((pixel_type & CS_BGR32) == CS_BGR32) && ((pixel_type & CS_Sample_Bits_Mask) == CS_Sample_Bits_16); }
bool VideoInfo::IsYUVA() const { return !!(pixel_type&CS_YUVA ); }
bool VideoInfo::IsPlanarRGB() const { return !!(pixel_type&CS_PLANAR) && !!(pixel_type&CS_BGR) && !!(pixel_type&CS_RGB_TYPE); }
bool VideoInfo::IsPlanarRGBA() const { return !!(pixel_type&CS_PLANAR) && !!(pixel_type&CS_BGR) && !!(pixel_type&CS_RGBA_TYPE); }
bool VideoInfo::IsChannelMaskKnown() const { return !!(image_type & IT_HAS_CHANNELMASK); }
// Re-maps and stores channel mask into image_type, sets the 'has channel mask' flag as well
void VideoInfo::SetChannelMask(bool isChannelMaskKnown, unsigned int dwChannelMask)
{
  if (isChannelMaskKnown) image_type |= IT_HAS_CHANNELMASK; else image_type &= ~IT_HAS_CHANNELMASK;
  if (!isChannelMaskKnown) dwChannelMask = 0;
  if (dwChannelMask & AvsChannelMask::MASK_SPEAKER_ALL) // special mapping due to lack of bits in image_type
    dwChannelMask = AvsImageTypeFlags::IT_SPEAKER_ALL;
  else {
    dwChannelMask &= AvsChannelMask::MASK_SPEAKER_DEFINED;
    dwChannelMask <<= 4;
  }
  image_type &= ~AvsImageTypeFlags::IT_SPEAKER_BITS_MASK;
  image_type |= dwChannelMask;
}

unsigned int VideoInfo::GetChannelMask() const
{
  // returns zero if not defined
  if (!IsChannelMaskKnown()) return 0;
  unsigned int dwChannelMask = image_type & AvsImageTypeFlags::IT_SPEAKER_BITS_MASK;
  // returns the original SPEAKER_ALL constant
  if (dwChannelMask == AvsImageTypeFlags::IT_SPEAKER_ALL)
    return AvsChannelMask::MASK_SPEAKER_ALL;
  // other SPEAKER bits were simply shifted by 4 bits
  return dwChannelMask >> 4;
}

bool VideoInfo::Is440()  const { return ((pixel_type & CS_PLANAR_MASK & ~CS_Sample_Bits_Mask) == (CS_GENERIC_YUV440 & CS_PLANAR_FILTER)) ||
                                        ((pixel_type & CS_PLANAR_MASK & ~CS_Sample_Bits_Mask) == (CS_GENERIC_YUVA440 & CS_PLANAR_FILTER)) ; }
bool VideoInfo::IsYA()   const { return ((pixel_type & CS_PLANAR_MASK & ~CS_Sample_Bits_Mask) == (CS_GENERIC_YA & CS_PLANAR_FILTER)) ||
                                        ((pixel_type & CS_PLANAR_MASK & ~CS_Sample_Bits_Mask) == (CS_GENERIC_YA & CS_PLANAR_FILTER)) ; }
bool VideoInfo::Is410()  const { return ((pixel_type & CS_PLANAR_MASK & ~CS_Sample_Bits_Mask) == (CS_GENERIC_YUV410 & CS_PLANAR_FILTER)) ||
                                        ((pixel_type & CS_PLANAR_MASK & ~CS_Sample_Bits_Mask) == (CS_GENERIC_YUVA410 & CS_PLANAR_FILTER)) ; }
bool VideoInfo::Is411()  const { return ((pixel_type & CS_PLANAR_MASK & ~CS_Sample_Bits_Mask) == (CS_GENERIC_YUV411 & CS_PLANAR_FILTER)) ||
                                        ((pixel_type & CS_PLANAR_MASK & ~CS_Sample_Bits_Mask) == (CS_GENERIC_YUVA411 & CS_PLANAR_FILTER)) ; }

// end struct VideoInfo

/**********************************************************************/

// class VideoFrameBuffer

const BYTE* VideoFrameBuffer::GetReadPtr() const { return data; }
/* Baked ********************
BYTE* VideoFrameBuffer::GetWritePtr() { ++sequence_number; return data; }
   Baked ********************/
BYTE* VideoFrameBuffer::GetWritePtr() { InterlockedIncrement(&sequence_number); return data; }
int VideoFrameBuffer::GetDataSize() const { return data_size; }
int VideoFrameBuffer::GetSequenceNumber() const { return sequence_number; }
int VideoFrameBuffer::GetRefcount() const { return refcount; }

// end class VideoFrameBuffer

/**********************************************************************/

// class VideoFrame

/* Baked ********************
void VideoFrame::AddRef() { InterlockedIncrement((int *)&refcount); }
void VideoFrame::Release() { if (refcount==1) InterlockedDecrement(&vfb->refcount); InterlockedDecrement((int *)&refcount); }

int VideoFrame::GetPitch() const { return pitch; }
int VideoFrame::GetPitch(int plane) const { switch (plane) {case PLANAR_U: case PLANAR_V: return pitchUV;} return pitch; }
int VideoFrame::GetRowSize() const { return row_size; }

int VideoFrame::GetRowSize(int plane) const {
  switch (plane) {
  case PLANAR_U: case PLANAR_V: if (pitchUV) return row_size>>1; else return 0;
  case PLANAR_U_ALIGNED: case PLANAR_V_ALIGNED:
    if (pitchUV) {
      int r = ((row_size+FRAME_ALIGN-1)&(~(FRAME_ALIGN-1)) )>>1; // Aligned rowsize
      if (r<=pitchUV)
        return r;
      return row_size>>1;
    } else return 0;
  case PLANAR_Y_ALIGNED:
    int r = (row_size+FRAME_ALIGN-1)&(~(FRAME_ALIGN-1)); // Aligned rowsize
    if (r<=pitch)
      return r;
    return row_size;
  }
  return row_size; }

int VideoFrame::GetHeight() const { return height; }
int VideoFrame::GetHeight(int plane) const {  switch (plane) {case PLANAR_U: case PLANAR_V: if (pitchUV) return height>>1; return 0;} return height; }

// generally you shouldn't use these three
VideoFrameBuffer* VideoFrame::GetFrameBuffer() const { return vfb; }
int VideoFrame::GetOffset() const { return offset; }
int VideoFrame::GetOffset(int plane) const { switch (plane) {case PLANAR_U: return offsetU;case PLANAR_V: return offsetV;default: return offset;}; }

const BYTE* VideoFrame::GetReadPtr() const { return vfb->GetReadPtr() + offset; }
const BYTE* VideoFrame::GetReadPtr(int plane) const { return vfb->GetReadPtr() + GetOffset(plane); }

bool VideoFrame::IsWritable() const { return (refcount == 1 && vfb->refcount == 1); }

BYTE* VideoFrame::GetWritePtr() const {
  if (vfb->GetRefcount()>1) {
    _ASSERT(FALSE);
    //throw AvisynthError("Internal Error - refcount was more than one!");
  }
  return IsWritable() ? (vfb->GetWritePtr() + offset) : 0;
}
   Baked ********************/

void VideoFrame::AddRef() { InterlockedIncrement(&refcount); }
void VideoFrame::Release() {
  VideoFrameBuffer* _vfb = vfb;

  if (!InterlockedDecrement(&refcount)) {
    // Do NOT touch 'properties' here. The FrameRegistry (under memory_mutex) is the
    // sole owner: it clears the AVSMap contents for reuse (winner frames) or deletes
    // it entirely (discarded frames). Freeing or clearing here without that lock would
    // race with concurrent FrameRegistry access.
#ifdef ALTERNATIVE_VFB_TIMESTAMP
    if (!InterlockedDecrement(&_vfb->refcount)) {
      // all VideoFrameBuffer instances are actually VFBStorage, which has way mode admin fields
      VFBHelper::UpdateVFBFreeTimestamp(_vfb);
    }
#else
    InterlockedDecrement(&_vfb->refcount);
#endif
  }
}

int VideoFrame::GetPitch(int plane) const { switch (plane) { case PLANAR_U: case PLANAR_V: return pitchUV; case PLANAR_A: return pitchA; } return pitch; }

int VideoFrame::GetRowSize(int plane) const {
  switch (plane) {
  case PLANAR_U: case PLANAR_V: if (pitchUV) return row_sizeUV; else return 0;
  case PLANAR_U_ALIGNED: case PLANAR_V_ALIGNED:
    if (pitchUV) {
      const int r = (row_sizeUV+FRAME_ALIGN-1)&(~(FRAME_ALIGN-1)); // Aligned rowsize
      if (r<=pitchUV)
        return r;
      return row_sizeUV;
    }
    else return 0;
  case PLANAR_ALIGNED: case PLANAR_Y_ALIGNED:
  case PLANAR_R_ALIGNED: case PLANAR_G_ALIGNED: case PLANAR_B_ALIGNED:
    {
    const int r = (row_size+FRAME_ALIGN-1)&(~(FRAME_ALIGN-1)); // Aligned rowsize
    if (r<=pitch)
      return r;
    return row_size;
  }
  case PLANAR_A:
    if (pitchA) return row_sizeA; else return 0;
  case PLANAR_A_ALIGNED:
    if(pitchA) {
      const int r = (row_sizeA+FRAME_ALIGN-1)&(~(FRAME_ALIGN-1)); // Aligned rowsize
      if (r<=pitchA)
        return r;
      return row_sizeA;
    }
    else return 0;
  }
  return row_size; // PLANAR_Y, PLANAR_G, PLANAR_B, PLANAR_R
}

int VideoFrame::GetHeight(int plane) const {
  switch (plane) {
  case PLANAR_U: case PLANAR_V: if (pitchUV) return heightUV; return 0;
  case PLANAR_A: if (pitchA) return height; return 0;
  }
  return height;
}

int VideoFrame::GetPixelType() const { return pixel_type; }
void VideoFrame::AmendPixelType(int new_pixel_type) { pixel_type = new_pixel_type; }

// Generally you should not be using these two
VideoFrameBuffer* VideoFrame::GetFrameBuffer() const { return vfb; }
int VideoFrame::GetOffset(int plane) const {
    switch (plane) {
    case PLANAR_U: case PLANAR_B: return offsetU; // G is first. Then B,R order like U,V
    case PLANAR_V: case PLANAR_R: return offsetV;
    case PLANAR_A: return offsetA;
    default: return offset; // PLANAR Y, PLANAR_G
    };
}

const BYTE* VideoFrame::GetReadPtr(int plane) const { return vfb->GetReadPtr() + GetOffset(plane); }

bool VideoFrame::IsWritable() const {
  if (refcount == 1 && vfb->refcount == 1) {
    vfb->GetWritePtr(); // Bump sequence number
    return true;
  }
  return false;
}

bool VideoFrame::IsPropertyWritable() const {
  return (refcount == 1);
}

BYTE* VideoFrame::GetWritePtr(int plane) const {
  if (!plane || plane == PLANAR_Y || plane == PLANAR_G) { // planar RGB order GBR
    if (vfb->GetRefcount()>1) {
      _ASSERT(FALSE);
//        throw AvisynthError("Internal Error - refcount was more than one!");
    }
    return (refcount == 1 && vfb->refcount == 1) ? vfb->GetWritePtr() + GetOffset(plane) : 0;
  }
  return vfb->data + GetOffset(plane);
}

AVSMap& VideoFrame::getProperties() {
  return *properties;
}

const AVSMap& VideoFrame::getConstProperties() {
  return *properties;
}

void VideoFrame::setProperties(const AVSMap& properties) {
  *(this->properties) = properties;
}

PDevice VideoFrame::GetDevice() const {
  return vfb->device;
}

int VideoFrame::CheckMemory() const {
#ifdef _DEBUG
  if (vfb->data && vfb->device->device_type == DEV_TYPE_CPU) {
    // check buffer overrun
    int *pInt = (int *)(vfb->data + vfb->data_size);
    if (pInt[0] != 0xDEADBEEF ||
      pInt[1] != 0xDEADBEEF ||
      pInt[2] != 0xDEADBEEF ||
      pInt[3] != 0xDEADBEEF)
    {
      return 1;
    }
    return 0;
  }
#endif
  return -1;
}

/* Baked ********************
VideoFrame::~VideoFrame() { InterlockedDecrement(&vfb->refcount); }
   Baked ********************/
VideoFrame::~VideoFrame()     { DESTRUCTOR(); }
void VideoFrame::DESTRUCTOR() { 
  Release(); 
  // finally destroyed when removed from FrameRegistry
  if (properties) {
    delete properties;
    properties = nullptr;
  }
}

// end class VideoFrame

/**********************************************************************/

// class IClip

/* Baked ********************
  void IClip::AddRef() { InterlockedIncrement((int *)&refcnt); }
  void IClip::Release() { InterlockedDecrement((int *)&refcnt); if (!refcnt) delete this; }
   Baked ********************/
void IClip::AddRef() { InterlockedIncrement(&refcnt); }
void IClip::Release() { if (!InterlockedDecrement(&refcnt)) delete this; }

// end class IClip

/**********************************************************************/

// class PClip

IClip* PClip::GetPointerWithAddRef() const { if (p) p->AddRef(); return p; }

void PClip::Init(IClip* x) {
  if (x) x->AddRef();
  p=x;
}

void PClip::Set(IClip* x) {
  if (x) x->AddRef();
  if (p) p->Release();
  p=x;
}

PClip::PClip()                               { CONSTRUCTOR0(); }
void PClip::CONSTRUCTOR0()                   { p = 0; }

PClip::PClip(const PClip& x)                 { CONSTRUCTOR1(x); }
void PClip::CONSTRUCTOR1(const PClip& x)     { Init(x.p); }

PClip::PClip(IClip* x)                       { CONSTRUCTOR2(x); }
void PClip::CONSTRUCTOR2(IClip* x)           { Init(x); }

void PClip::operator=(IClip* x)              { OPERATOR_ASSIGN0(x); }
void PClip::OPERATOR_ASSIGN0(IClip* x)       { Set(x); }

void PClip::operator=(const PClip& x)        { OPERATOR_ASSIGN1(x); }
void PClip::OPERATOR_ASSIGN1(const PClip& x) { Set(x.p); }

PClip::~PClip()                              { DESTRUCTOR(); }
void PClip::DESTRUCTOR()                     { if (p) p->Release(); }

// end class PClip

/**********************************************************************/

// class PVideoFrame

void PVideoFrame::Init(VideoFrame* x) {
  if (x) x->AddRef();
  p=x;
}

void PVideoFrame::Set(VideoFrame* x) {
  if (x) x->AddRef();
  if (p) p->Release();
  p=x;
}

PVideoFrame::PVideoFrame()                               { CONSTRUCTOR0(); }
void PVideoFrame::CONSTRUCTOR0()                         { p = 0; }

PVideoFrame::PVideoFrame(const PVideoFrame& x)           { CONSTRUCTOR1(x); }
void PVideoFrame::CONSTRUCTOR1(const PVideoFrame& x)     { Init(x.p); }

PVideoFrame::PVideoFrame(VideoFrame* x)                  { CONSTRUCTOR2(x); }
void PVideoFrame::CONSTRUCTOR2(VideoFrame* x)            { Init(x); }

void PVideoFrame::operator=(VideoFrame* x)               { OPERATOR_ASSIGN0(x); }
void PVideoFrame::OPERATOR_ASSIGN0(VideoFrame* x)        { Set(x); }

void PVideoFrame::operator=(const PVideoFrame& x)        { OPERATOR_ASSIGN1(x); }
void PVideoFrame::OPERATOR_ASSIGN1(const PVideoFrame& x) { Set(x.p); }

PVideoFrame::~PVideoFrame()                              { DESTRUCTOR(); }
void PVideoFrame::DESTRUCTOR()                           { if (p) p->Release(); }


// end class PVideoFrame

/**********************************************************************/

// class AVSValue

AVSValue::AVSValue()                                     { CONSTRUCTOR0(); }
void AVSValue::CONSTRUCTOR0()                            { type = 'v'; array_size = 0; clip = NULL; }

AVSValue::AVSValue(IClip* c)                             { CONSTRUCTOR1(c); }
void AVSValue::CONSTRUCTOR1(IClip* c)                    { type = 'c'; array_size = 0; clip = c; if (c) c->AddRef(); }

AVSValue::AVSValue(const PClip& c)                       { CONSTRUCTOR2(c); }
void AVSValue::CONSTRUCTOR2(const PClip& c)              { type = 'c'; array_size = 0; clip = c.GetPointerWithAddRef(); }

AVSValue::AVSValue(bool b)                               { CONSTRUCTOR3(b); }
void AVSValue::CONSTRUCTOR3(bool b)                      { type = 'b'; array_size = 0; clip = NULL; boolean = b; }

AVSValue::AVSValue(int i)                                { CONSTRUCTOR4(i); }
void AVSValue::CONSTRUCTOR4(int i)                       { type = 'i'; array_size = 0; clip = NULL; integer = i; }

AVSValue::AVSValue(float f)                              { CONSTRUCTOR5(f); }
void AVSValue::CONSTRUCTOR5(float f)                     { type = 'f'; array_size = 0; clip = NULL; floating_pt = f; }

AVSValue::AVSValue(double f)                             { CONSTRUCTOR6(f); }
void AVSValue::CONSTRUCTOR6(double f)
{
  type = 'd'; array_size = 0; clip = NULL;
#if UINTPTR_MAX >= 0xffffffffffffffff
  // pre-v11: floating_pt = float(f);
  double_pt = f; // v11: real 64 bit double!
#else
  double_pt_ptr = new double;
  *double_pt_ptr = f;
#endif
}

AVSValue::AVSValue(const char* s)                        { CONSTRUCTOR7(s); }
void AVSValue::CONSTRUCTOR7(const char* s)               { type = 's'; array_size = 0; string = s; }

/* Baked ********************
AVSValue::AVSValue(const AVSValue* a, int size) { type = 'a'; array = a; array_size = size; }
   Baked ********************/
AVSValue::AVSValue(const AVSValue& a, int size)          { CONSTRUCTOR8(&a, size); }
AVSValue::AVSValue(const AVSValue* a, int size)          { CONSTRUCTOR8(a, size); }
void AVSValue::CONSTRUCTOR8(const AVSValue* a, int size)
{
  type = 'a';
  array_size = (short)size;
  if (a == nullptr)
    size = 0;
  array = new AVSValue[size]; // even for size = 0 a valid ptr is given
  for (int i = 0; i < size; i++) {
    const_cast<AVSValue*>(array)[i].Assign(&a[i], true); // init from source
  }
}

AVSValue::AVSValue(const AVSValue& v)                    { CONSTRUCTOR9(v); }
void AVSValue::CONSTRUCTOR9(const AVSValue& v)           { Assign(&v, true); }

// from compatibility interface
AVSValue::AVSValue(const AVSValue& v, bool no_deep_arrays) { CONSTRUCTOR10(v, no_deep_arrays); }
void AVSValue::CONSTRUCTOR10(const AVSValue& v, bool no_deep_arrays)  { Assign2(&v, true, no_deep_arrays); }

AVSValue::AVSValue(const PFunction& n) { CONSTRUCTOR11(n); }
void AVSValue::CONSTRUCTOR11(const PFunction& n) { type = 'n'; array_size = 0; function = n.GetPointerWithAddRef(); }

AVSValue::AVSValue(int64_t l) { CONSTRUCTOR12(l); }
void AVSValue::CONSTRUCTOR12(int64_t l) { type = 'l'; array_size = 0; clip = NULL; 
#if UINTPTR_MAX >= 0xffffffffffffffff
  longlong = l; 
#else
  longlong_ptr = new int64_t;
  *longlong_ptr = l;
#endif
}

AVSValue::~AVSValue()                                    { DESTRUCTOR(); }
void AVSValue::DESTRUCTOR()
{
  if (IsClip() && clip)
    clip->Release();
  if (IsFunction() && function)
    function->Release();
#if !(UINTPTR_MAX >= 0xffffffffffffffff)
  // 32-bit systems support 64-bit data through dynamic allocation.
  if (type == 'l' && longlong_ptr) {
    delete longlong_ptr;
    longlong_ptr = nullptr;
  }
  else if (type == 'd' && double_pt_ptr) {
    delete double_pt_ptr;
    double_pt_ptr = nullptr;
  }
#endif
  if (IsArray() && array) {
    delete[] array; // calls AVSValue destructors for all elements
    array = nullptr;
  }
}

void AVSValue::MarkArrayAsNonDeepCopy()
{
  // Convert AVSValue content to a neutral 'v' type
  // to prevent destructor from freeing up the allocated array 
  // and its elements.
  // Set just before releasing AVSValue.
  if (type == 'a')
  {
    type = 'v';
    array_size = 0;
    array = nullptr;
  }
}

AVSValue& AVSValue::operator=(const AVSValue& v)         { return OPERATOR_ASSIGN(v); }
AVSValue& AVSValue::OPERATOR_ASSIGN(const AVSValue& v)   { Assign(&v, false); return *this; }

// pre-v11: Transparently allow 'int' to be treated as 'float'.
// No int<->bool conversions.
// v11 64 bit value changes:
// - 'long' is int64_t.
// - IsInt returns true for both 'long' and 'int' 
// - IsLong returns true only for exact 64-bit integer type
// - Transparently allow 'long' and 'int' to be treated as 'float' or 'double'
// - IsFloat returns true for both 'float' and 'double' (old IsFloat + 'double').
// - new IsFloatfStrict returns true only for 32-bit floating-point numbers.
// - new IsLongStrict returns true only for exact 64-bit integer types.

// Note: Exact type check is available GetType() with returns AvsValueType enum
bool AVSValue::Defined() const { return type != 'v'; }
bool AVSValue::IsClip() const { return type == 'c'; }
bool AVSValue::IsBool() const { return type == 'b'; }
bool AVSValue::IsInt() const { return type == 'i' || type == 'l'; }
bool AVSValue::IsLongStrict() const { return type == 'l'; }
bool AVSValue::IsFloat() const { return type == 'd' || type == 'f' || type == 'i' || type == 'l'; }
bool AVSValue::IsFloatfStrict() const { return type == 'f'; }
bool AVSValue::IsString() const { return type == 's'; }
bool AVSValue::IsArray() const { return type == 'a'; }
bool AVSValue::IsFunction() const { return type == 'n'; }

PClip AVSValue::AsClip() const { _ASSERTE(IsClip()); return IsClip()?clip:0; }

bool AVSValue::AsBool1() const { _ASSERTE(IsBool()); return boolean; }
bool AVSValue::AsBool() const { return AsBool1(); }

int AVSValue::AsInt1() const { 
  _ASSERTE(IsInt()); 
  // simple typecast, no saturation
#if UINTPTR_MAX >= 0xffffffffffffffff
  return type == 'i' ? integer : (int)longlong;
#else
  return type == 'i' ? integer : (int)*longlong_ptr;
#endif
}

int AVSValue::AsInt() const { return AsInt1(); }

int64_t AVSValue::AsLong1() const {
  _ASSERTE(IsInt());
#if UINTPTR_MAX >= 0xffffffffffffffff
  return type == 'i' ? integer : longlong;
#else
  return type == 'i' ? integer : *longlong_ptr;
#endif
}
int64_t AVSValue::AsLong() const { return AsLong1(); }

const char* AVSValue::AsString1() const { _ASSERTE(IsString()); return IsString()?string:0; }
const char* AVSValue::AsString() const { return AVSValue::AsString1(); }

double AVSValue::AsFloat1() const { 
#if UINTPTR_MAX >= 0xffffffffffffffff
  _ASSERTE(IsFloat()); return type == 'i' ? (double)integer : type == 'l' ? (double)longlong : type == 'f' ? (double)floating_pt : double_pt;
#else
  _ASSERTE(IsFloat()); return type == 'i' ? (double)integer : type == 'l' ? (double)*longlong_ptr : type == 'f' ? (double)floating_pt : *double_pt_ptr;
#endif
}

double AVSValue::AsFloat() const { return AsFloat1(); }

float AVSValue::AsFloatf() const { 
#if UINTPTR_MAX >= 0xffffffffffffffff
  _ASSERTE(IsFloat()); return type == 'i' ? (float)integer : type == 'l' ? (float)longlong : type == 'f' ? floating_pt : (float)double_pt;
#else
  _ASSERTE(IsFloat()); return type == 'i' ? (float)integer : type == 'l' ? (float)*longlong_ptr : type == 'f' ? floating_pt : (float)*double_pt_ptr;
#endif
}

bool AVSValue::AsBool2(bool def) const { _ASSERTE(IsBool()||!Defined()); return IsBool() ? boolean : def; }
bool AVSValue::AsBool(bool def) const { return AsBool2(def); }

int AVSValue::AsInt2(int def) const { 
  _ASSERTE(IsInt()||!Defined());
#if UINTPTR_MAX >= 0xffffffffffffffff
  return type == 'i' ? integer : type == 'l' ? (int)longlong : def;
#else
  return type == 'i' ? integer : type == 'l' ? (int)*longlong_ptr : def;
#endif
}

int AVSValue::AsInt(int def) const { return AsInt2(def); }
int64_t AVSValue::AsLong2(int64_t def) const {
  _ASSERTE(IsInt() || !Defined());
#if UINTPTR_MAX >= 0xffffffffffffffff
  return type == 'i' ? integer : type == 'l' ? longlong : def;
#else
  return type == 'i' ? integer : type == 'l' ? *longlong_ptr : def;
#endif
}

int64_t AVSValue::AsLong(int64_t def) const {
  return AsLong2(def);
}

double AVSValue::AsDblDef(double def) const { 
  _ASSERTE(IsFloat()||!Defined());
#if UINTPTR_MAX >= 0xffffffffffffffff
  return type == 'i' ? (double)integer : type == 'l' ? (double)longlong : type == 'f' ? (double)floating_pt : type == 'd' ? double_pt : def;
#else
  return type == 'i' ? (double)integer : type == 'l' ? (double)*longlong_ptr : type == 'f' ? (double)floating_pt : type == 'd' ? *double_pt_ptr : def;
#endif
}

double AVSValue::AsFloat2(float def) const { 
  _ASSERTE(IsFloat()||!Defined()); 
#if UINTPTR_MAX >= 0xffffffffffffffff
  return type == 'i' ? integer : type == 'l' ? (double)longlong : type == 'f' ? (double)floating_pt : type == 'd' ? double_pt : (double)def;
#else
  return type == 'i' ? integer : type == 'l' ? (double)*longlong_ptr : type == 'f' ? (double)floating_pt : type == 'd' ? *double_pt_ptr : (double)def;
#endif
}

double AVSValue::AsFloat(float def) const { return AsFloat2(def); }
float AVSValue::AsFloatf(float def) const { return float( AsFloat2(def) ); }

const char* AVSValue::AsString2(const char* def) const { _ASSERTE(IsString()||!Defined()); return IsString() ? string : def; }
const char* AVSValue::AsString(const char* def) const { return AVSValue::AsString2(def); }

PFunction AVSValue::AsFunction() const { _ASSERTE(IsFunction()); return IsFunction() ? function : 0; }

int AVSValue::ArraySize() const { _ASSERTE(IsArray()) ; return IsArray() ? array_size : 1; }

const AVSValue& AVSValue::operator[](int index) const     { return OPERATOR_INDEX(index); }
const AVSValue& AVSValue::OPERATOR_INDEX(int index) const {
  _ASSERTE(IsArray() && index>=0 && index<array_size);
  return (IsArray() && index>=0 && index<array_size) ? array[index] : *this;
}

// This Assign can deep-copy array elements
// For compatibility interface, we use Assign2 through CONSTRUCTOR10
void AVSValue::Assign(const AVSValue* src, bool init) {
  Assign2(src, init, false);
}

void AVSValue::Assign2(const AVSValue* src, bool init, bool no_deep_arrays) {
  if (src->IsClip() && src->clip)
    src->clip->AddRef();
  if (src->IsFunction() && src->function)
    src->function->AddRef();

  // The left side of the assignment ("this") will be overwritten by "src",
  // so it should be released or freed if necessary.
  // The relevant properties must be released/dereferenced at the end:
  // clip, array, function, double_pt_ptr and longlong_ptr
  // Since they share the same pointer, saving (void*)clip would do it;
  const bool shouldReleaseClip = !init && IsClip() && clip;
  const bool shouldReleaseFunction = !init && IsFunction() && function;
  const bool shouldReleaseArray = !init && IsArray() && array && !no_deep_arrays;
#if !(UINTPTR_MAX >= 0xffffffffffffffff)
  const bool shouldReleaseDouble = !init && type == 'd' && double_pt_ptr;
  const bool shouldReleaseLong = !init && type == 'l' && longlong_ptr;
#endif
  void* prev_pointer_to_release = (void*)clip;

  // common fields
  this->type = src->type;
  this->array_size = src->array_size;

  // no_deep_arrays: compatibility for AVS2.5: don't free or deep copy array elements!
  if (src->IsArray() && !no_deep_arrays)
  {
    // Delay releasing the previous array to avoid the args = args[0] case.
    // If an existing array were freed before this point, it would break the
    // value = value[x] case where the source value[x] is not an array.
    // Even for size == 0, this returns a valid pointer.
    AVSValue* tmp = new AVSValue[array_size];
    for (int i = 0; i < array_size; i++)
      tmp[i].Assign(&src->array[i], true); // init from source
    array = tmp;
  }
#if !(UINTPTR_MAX >= 0xffffffffffffffff)
  // 32 bit Avisynth: new 64 bit types are specially treated
  else if (this->type == 'l') {
    const uint64_t l = *src->longlong_ptr;
    this->longlong_ptr = new int64_t;
    *this->longlong_ptr = l;
  }
  else if (this->type == 'd') {
    const double d = *src->double_pt_ptr;
    this->double_pt_ptr = new double;
    *this->double_pt_ptr = d;
  }
#endif
  else {
    this->clip = (IClip*)((void*)src->clip);
    // "clip" is the largest member of the union, making sure we copy everything
  }

  if (shouldReleaseClip)
    ((IClip*)prev_pointer_to_release)->Release();
  if (shouldReleaseFunction)
    ((IFunction*)prev_pointer_to_release)->Release();
  if (shouldReleaseArray)
    delete[](AVSValue*)(prev_pointer_to_release);
  // deallocates former array memory + calls destructor of AVSValue elements
#if !(UINTPTR_MAX >= 0xffffffffffffffff)
  if (shouldReleaseDouble)
    delete (double*)prev_pointer_to_release;
  else if (shouldReleaseLong)
    delete (int64_t*)prev_pointer_to_release;
#endif
}

AvsValueType AVSValue::GetType() const { return (AvsValueType)type; }

// end class AVSValue

/**********************************************************************/

PFunction::PFunction() { CONSTRUCTOR0(); }
void PFunction::CONSTRUCTOR0() { Init(0); }

PFunction::PFunction(IFunction* p) { CONSTRUCTOR1(p); }
void PFunction::CONSTRUCTOR1(IFunction* p) { Init(p); }

PFunction::PFunction(const PFunction& p) { CONSTRUCTOR2(p); }
void PFunction::CONSTRUCTOR2(const PFunction& p) { Init(p.e); }

PFunction& PFunction::operator=(IFunction* p) { return OPERATOR_ASSIGN0(p); }
PFunction& PFunction::OPERATOR_ASSIGN0(IFunction* p) { Set(p); return *this; }

PFunction& PFunction::operator=(const PFunction& p) { return OPERATOR_ASSIGN1(p); }
PFunction& PFunction::OPERATOR_ASSIGN1(const PFunction& p) { Set(p.e); return *this; }

PFunction::~PFunction() { DESTRUCTOR(); }
void PFunction::DESTRUCTOR() { if (e) e->Release(); }

IFunction * PFunction::GetPointerWithAddRef() const { if (e) e->AddRef(); return e; }
void PFunction::Init(IFunction* p) { e = p; if (e) e->AddRef(); }
void PFunction::Set(IFunction* p) { if (p) p->AddRef(); if (e) e->Release(); e = p; }

PDevice::PDevice() { CONSTRUCTOR0(); }
void PDevice::CONSTRUCTOR0() { e = 0; }

PDevice::PDevice(Device* p) { CONSTRUCTOR1(p); }
void PDevice::CONSTRUCTOR1(Device* p) { e = p; }

PDevice::PDevice(const PDevice& p) { CONSTRUCTOR2(p); }
void PDevice::CONSTRUCTOR2(const PDevice& p) { e = p.e; }

PDevice& PDevice::operator=(Device* p) { return OPERATOR_ASSIGN0(p); }
PDevice& PDevice::OPERATOR_ASSIGN0(Device* p) { e = p; return *this; }

PDevice& PDevice::operator=(const PDevice& p) { return OPERATOR_ASSIGN1(p); }
PDevice& PDevice::OPERATOR_ASSIGN1(const PDevice& p) { e = p.e; return *this; }

PDevice::~PDevice() { }
void PDevice::DESTRUCTOR() { }

AvsDeviceType PDevice::GetType() const { return e ? e->device_type : DEV_TYPE_NONE; }
int PDevice::GetId() const { return e ? e->device_id : -1; }
int PDevice::GetIndex() const { return e ? e->device_index : -1; }
const char* PDevice::GetName() const { return e ? e->GetName() : nullptr; }

INeoEnv* __stdcall GetAvsEnv(IScriptEnvironment* env) { return static_cast<InternalEnvironment*>(env); }

PNeoEnv::PNeoEnv(IScriptEnvironment* env) : p(static_cast<InternalEnvironment*>(env)) { }
PNeoEnv::operator IScriptEnvironment2* () { return static_cast<InternalEnvironment*>(p); }

/**********************************************************************/

static const AVS_Linkage avs_linkage = {    // struct AVS_Linkage {

  sizeof(AVS_Linkage),                      //   int Size;

/***************************************************************************************************************/
// struct VideoInfo
  &VideoInfo::HasVideo,                     //   bool    (VideoInfo::*HasVideo)() const;
  &VideoInfo::HasAudio,                     //   bool    (VideoInfo::*HasAudio)() const;
  &VideoInfo::IsRGB,                        //   bool    (VideoInfo::*IsRGB)() const;
  &VideoInfo::IsRGB24,                      //   bool    (VideoInfo::*IsRGB24)() const;
  &VideoInfo::IsRGB32,                      //   bool    (VideoInfo::*IsRGB32)() const;
  &VideoInfo::IsYUV,                        //   bool    (VideoInfo::*IsYUV)() const;
  &VideoInfo::IsYUY2,                       //   bool    (VideoInfo::*IsYUY2)() const;
  &VideoInfo::IsYV24,                       //   bool    (VideoInfo::*IsYV24)()  const;
  &VideoInfo::IsYV16,                       //   bool    (VideoInfo::*IsYV16)()  const;
  &VideoInfo::IsYV12,                       //   bool    (VideoInfo::*IsYV12)()  const;
  &VideoInfo::IsYV411,                      //   bool    (VideoInfo::*IsYV411)() const;
  &VideoInfo::IsY8,                         //   bool    (VideoInfo::*IsY8)()    const;
  &VideoInfo::IsColorSpace,                 //   bool    (VideoInfo::*IsColorSpace)(int c_space) const;
  &VideoInfo::Is,                           //   bool    (VideoInfo::*Is)(int property) const;
  &VideoInfo::IsPlanar,                     //   bool    (VideoInfo::*IsPlanar)() const;
  &VideoInfo::IsFieldBased,                 //   bool    (VideoInfo::*IsFieldBased)() const;
  &VideoInfo::IsParityKnown,                //   bool    (VideoInfo::*IsParityKnown)() const;
  &VideoInfo::IsBFF,                        //   bool    (VideoInfo::*IsBFF)() const;
  &VideoInfo::IsTFF,                        //   bool    (VideoInfo::*IsTFF)() const;
  &VideoInfo::IsVPlaneFirst,                //   bool    (VideoInfo::*IsVPlaneFirst)() const;
  &VideoInfo::BytesFromPixels,              //   int     (VideoInfo::*BytesFromPixels)(int pixels) const;
  &VideoInfo::RowSize,                      //   int     (VideoInfo::*RowSize)(int plane) const;
  &VideoInfo::BMPSize,                      //   int     (VideoInfo::*BMPSize)() const;
  &VideoInfo::AudioSamplesFromFrames,       //   int64_t (VideoInfo::*AudioSamplesFromFrames)(int frames) const;
  &VideoInfo::FramesFromAudioSamples,       //   int     (VideoInfo::*FramesFromAudioSamples)(int64_t samples) const;
  &VideoInfo::AudioSamplesFromBytes,        //   int64_t (VideoInfo::*AudioSamplesFromBytes)(int64_t bytes) const;
  &VideoInfo::BytesFromAudioSamples,        //   int64_t (VideoInfo::*BytesFromAudioSamples)(int64_t samples) const;
  &VideoInfo::AudioChannels,                //   int     (VideoInfo::*AudioChannels)() const;
  &VideoInfo::SampleType,                   //   int     (VideoInfo::*SampleType)() const;
  &VideoInfo::IsSampleType,                 //   bool    (VideoInfo::*IsSampleType)(int testtype) const;
  &VideoInfo::SamplesPerSecond,             //   int     (VideoInfo::*SamplesPerSecond)() const;
  &VideoInfo::BytesPerAudioSample,          //   int     (VideoInfo::*BytesPerAudioSample)() const;
  &VideoInfo::SetFieldBased,                //   void    (VideoInfo::*SetFieldBased)(bool isfieldbased);
  &VideoInfo::Set,                          //   void    (VideoInfo::*Set)(int property);
  &VideoInfo::Clear,                        //   void    (VideoInfo::*Clear)(int property);
  &VideoInfo::GetPlaneWidthSubsampling,     //   int     (VideoInfo::*GetPlaneWidthSubsampling)(int plane) const;
  &VideoInfo::GetPlaneHeightSubsampling,    //   int     (VideoInfo::*GetPlaneHeightSubsampling)(int plane) const;
  &VideoInfo::BitsPerPixel,                 //   int     (VideoInfo::*BitsPerPixel)() const;
  &VideoInfo::BytesPerChannelSample,        //   int     (VideoInfo::*BytesPerChannelSample)() const;
  &VideoInfo::SetFPS,                       //   void    (VideoInfo::*SetFPS)(unsigned numerator, unsigned denominator)
  &VideoInfo::MulDivFPS,                    //   void    (VideoInfo::*MulDivFPS)(unsigned multiplier, unsigned divisor)
  &VideoInfo::IsSameColorspace,             //   bool    (VideoInfo::*IsSameColorspace)(const VideoInfo& vi) const;
// end struct VideoInfo
/***************************************************************************************************************/
// class VideoFrameBuffer
  &VideoFrameBuffer::GetReadPtr,            //   const BYTE* (VideoFrameBuffer::*VFBGetReadPtr)() const;
  &VideoFrameBuffer::GetWritePtr,           //   BYTE*       (VideoFrameBuffer::*VFBGetWritePtr)();
  &VideoFrameBuffer::GetDataSize,           //   int         (VideoFrameBuffer::*GetDataSize)() const;
  &VideoFrameBuffer::GetSequenceNumber,     //   int         (VideoFrameBuffer::*GetSequenceNumber)() const;
  &VideoFrameBuffer::GetRefcount,           //   int         (VideoFrameBuffer::*GetRefcount)() const;
// end class VideoFrameBuffer
/***************************************************************************************************************/
// class VideoFrame
  &VideoFrame::GetPitch,                    //   int               (VideoFrame::*GetPitch)(int plane) const;
  &VideoFrame::GetRowSize,                  //   int               (VideoFrame::*GetRowSize)(int plane) const;
  &VideoFrame::GetHeight,                   //   int               (VideoFrame::*GetHeight)(int plane) const;
  &VideoFrame::GetFrameBuffer,              //   VideoFrameBuffer* (VideoFrame::*GetFrameBuffer)() const;
  &VideoFrame::GetOffset,                   //   int               (VideoFrame::*GetOffset)(int plane) const;
  &VideoFrame::GetReadPtr,                  //   const BYTE*       (VideoFrame::*VFGetReadPtr)(int plane) const;
  &VideoFrame::IsWritable,                  //   bool              (VideoFrame::*IsWritable)() const;
  &VideoFrame::GetWritePtr,                 //   BYTE*             (VideoFrame::*VFGetWritePtr)(int plane) const;
  &VideoFrame::DESTRUCTOR,                  //   void              (VideoFrame::*VideoFrame_DESTRUCTOR)();
// end class VideoFrame
/***************************************************************************************************************/
// class IClip
                                            //   /* nothing */
// end class IClip
/***************************************************************************************************************/
// class PClip
  &PClip::CONSTRUCTOR0,                     //   void (PClip::*PClip_CONSTRUCTOR0)();
  &PClip::CONSTRUCTOR1,                     //   void (PClip::*PClip_CONSTRUCTOR1)(const PClip& x);
  &PClip::CONSTRUCTOR2,                     //   void (PClip::*PClip_CONSTRUCTOR2)(IClip* x);
  &PClip::OPERATOR_ASSIGN0,                 //   void (PClip::*PClip_OPERATOR_ASSIGN0)(IClip* x);
  &PClip::OPERATOR_ASSIGN1,                 //   void (PClip::*PClip_OPERATOR_ASSIGN1)(const PClip& x);
  &PClip::DESTRUCTOR,                       //   void (PClip::*PClip_DESTRUCTOR)();
// end class PClip
/***************************************************************************************************************/
// class PVideoFrame
  &PVideoFrame::CONSTRUCTOR0,               //   void (PVideoFrame::*PVideoFrame_CONSTRUCTOR0)();
  &PVideoFrame::CONSTRUCTOR1,               //   void (PVideoFrame::*PVideoFrame_CONSTRUCTOR1)(const PVideoFrame& x);
  &PVideoFrame::CONSTRUCTOR2,               //   void (PVideoFrame::*PVideoFrame_CONSTRUCTOR2)(VideoFrame* x);
  &PVideoFrame::OPERATOR_ASSIGN0,           //   void (PVideoFrame::*PVideoFrame_OPERATOR_ASSIGN0(VideoFrame* x);
  &PVideoFrame::OPERATOR_ASSIGN1,           //   void (PVideoFrame::*PVideoFrame_OPERATOR_ASSIGN1(const PVideoFrame& x);
  &PVideoFrame::DESTRUCTOR,                 //   void (PVideoFrame::*PVideoFrame_DESTRUCTOR)();
// end class PVideoFrame
/***************************************************************************************************************/
// class AVSValue
  &AVSValue::CONSTRUCTOR0,                  //   void            (AVSValue::*AVSValue_CONSTRUCTOR0)();
  &AVSValue::CONSTRUCTOR1,                  //   void            (AVSValue::*AVSValue_CONSTRUCTOR1)(IClip* c);
  &AVSValue::CONSTRUCTOR2,                  //   void            (AVSValue::*AVSValue_CONSTRUCTOR2)(const PClip& c);
  &AVSValue::CONSTRUCTOR3,                  //   void            (AVSValue::*AVSValue_CONSTRUCTOR3)(bool b);
  &AVSValue::CONSTRUCTOR4,                  //   void            (AVSValue::*AVSValue_CONSTRUCTOR4)(int i);
  &AVSValue::CONSTRUCTOR5,                  //   void            (AVSValue::*AVSValue_CONSTRUCTOR5)(float f);
  &AVSValue::CONSTRUCTOR6,                  //   void            (AVSValue::*AVSValue_CONSTRUCTOR6)(double f);
  &AVSValue::CONSTRUCTOR7,                  //   void            (AVSValue::*AVSValue_CONSTRUCTOR7)(const char* s);
  &AVSValue::CONSTRUCTOR8,                  //   void            (AVSValue::*AVSValue_CONSTRUCTOR8)(const AVSValue* a, int
  &AVSValue::CONSTRUCTOR9,                  //   void            (AVSValue::*AVSValue_CONSTRUCTOR9)(const AVSValue& v);
  &AVSValue::DESTRUCTOR,                    //   void            (AVSValue::*AVSValue_DESTRUCTOR)();
  &AVSValue::OPERATOR_ASSIGN,               //   AVSValue&       (AVSValue::*AVSValue_OPERATOR_ASSIGN)(const AVSValue& v);
  &AVSValue::OPERATOR_INDEX,                //   const AVSValue& (AVSValue::*AVSValue_OPERATOR_INDEX)(int index) const;
  &AVSValue::Defined,                       //   bool            (AVSValue::*Defined)() const;
  &AVSValue::IsClip,                        //   bool            (AVSValue::*IsClip)() const;
  &AVSValue::IsBool,                        //   bool            (AVSValue::*IsBool)() const;
  &AVSValue::IsInt,                         //   bool            (AVSValue::*IsInt)() const;
  &AVSValue::IsFloat,                       //   bool            (AVSValue::*IsFloat)() const;
  &AVSValue::IsString,                      //   bool            (AVSValue::*IsString)() const;
  &AVSValue::IsArray,                       //   bool            (AVSValue::*IsArray)() const;
  &AVSValue::AsClip,                        //   PClip           (AVSValue::*AsClip)() const;
  &AVSValue::AsBool1,                       //   bool            (AVSValue::*AsBool1)() const;
  &AVSValue::AsInt1,                        //   int             (AVSValue::*AsInt1)() const;
  &AVSValue::AsString1,                     //   const char*     (AVSValue::*AsString1)() const;
  &AVSValue::AsFloat1,                      //   double          (AVSValue::*AsFloat1)() const; // AsDouble1
  &AVSValue::AsBool2,                       //   bool            (AVSValue::*AsBool2)(bool def) const;
  &AVSValue::AsInt2,                        //   int             (AVSValue::*AsInt2)(int def) const;
  &AVSValue::AsDblDef,                      //   double          (AVSValue::*AsDblDef)(double def) const; // AsDouble2
  &AVSValue::AsFloat2,                      //   double          (AVSValue::*AsFloat2)(float def) const;
  &AVSValue::AsString2,                     //   const char*     (AVSValue::*AsString2)(const char* def) const;
  &AVSValue::ArraySize,                     //   int             (AVSValue::*ArraySize)() const;
  &AVSValue::IsLongStrict,                  //   bool            (AVSValue::*IsLongStrict)() const; // v11
  &AVSValue::IsFloatfStrict,                //   bool            (AVSValue::*IsFloatfStrict() const; // v11
  &AVSValue::AsLong1,                       //   int64_t         (AVSValue::*AsLong1)() const; // v11
  &AVSValue::AsLong2,                       //   int64_t         (AVSValue::*AsLong2)(int64_t def) const; // v11
  &AVSValue::CONSTRUCTOR12,                 //   void            (AVSValue::*AVSValue_CONSTRUCTOR12)(int64_ l); // v11
  // end class AVSValue
/**********************************************************************/
  // a single { nullptr } initializes the whole placeholder array
  { nullptr },                              // void    (VideoInfo::* reserved[27])(); // reserved for AVS classic
/**********************************************************************/
  // AviSynth+ additions
  &VideoInfo::NumComponents,                //   int     (VideoInfo::*NumChannels)() const;
  &VideoInfo::ComponentSize,                //   int     (VideoInfo::*ComponentSize)() const;
  &VideoInfo::BitsPerComponent,             //   int     (VideoInfo::*BitsPerComponent)() const;
  &VideoInfo::Is444,                        //   bool    (VideoInfo::*Is444)()  const;
  &VideoInfo::Is422,                        //   bool    (VideoInfo::*Is422)()  const;
  &VideoInfo::Is420,                        //   bool    (VideoInfo::*Is420)()  const;
  &VideoInfo::IsY,                          //   bool    (VideoInfo::*IsY)()    const;
  &VideoInfo::IsRGB48,                      //   bool    (VideoInfo::*IsRGB48)()  const;
  &VideoInfo::IsRGB64,                      //   bool    (VideoInfo::*IsRGB64)()  const;
  &VideoInfo::IsYUVA,                       //   bool    (VideoInfo::*IsYUVA)()  const;
  &VideoInfo::IsPlanarRGB,                  //   bool    (VideoInfo::*IsPlanarRGB)()  const;
  &VideoInfo::IsPlanarRGBA,                 //   bool    (VideoInfo::*IsPlanarRGBA)()  const;
/**********************************************************************/
  // Frame properties
  &VideoFrame::getProperties, // AVSMap& (VideoFrame::* getProperties)();
  &VideoFrame::getConstProperties, // const AVSMap& (VideoFrame::* getConstProperties)();
  &VideoFrame::setProperties, // void (VideoFrame::* setProperties)(const AVSMap& properties);

  // PFunction (Neo)
  &AVSValue::CONSTRUCTOR11,
  &AVSValue::IsFunction,
  &PFunction::CONSTRUCTOR0,
  &PFunction::CONSTRUCTOR1,
  &PFunction::CONSTRUCTOR2,
  &PFunction::OPERATOR_ASSIGN0,
  &PFunction::OPERATOR_ASSIGN1,
  &PFunction::DESTRUCTOR,
  // end PFunction

  // VideoFrame extras (Neo)
  &VideoFrame::CheckMemory,
  &VideoFrame::GetDevice,

  // class PDevice (Neo)
  &PDevice::CONSTRUCTOR0,
  &PDevice::CONSTRUCTOR1,
  &PDevice::CONSTRUCTOR2,
  &PDevice::OPERATOR_ASSIGN0,
  &PDevice::OPERATOR_ASSIGN1,
  &PDevice::DESTRUCTOR,
  &PDevice::GetType,
  &PDevice::GetId,
  &PDevice::GetIndex,
  &PDevice::GetName,
  // end class PDevice

  // V9
  &VideoFrame::IsPropertyWritable,

  // V10
  &VideoFrame::GetPixelType,                //   int          (VideoFrame::*VideoFrame_GetPixelType)() const;
  &VideoFrame::AmendPixelType,              //   void         (VideoFrame::*VideoFrame_AmendPixelType)();
  &VideoFrameBuffer::DESTRUCTOR,            //   void         (VideoFrameBuffer::*VideoFrameBuffer_DESTRUCTOR)();
  &AVSValue::GetType,                       //   AvsValueType (AVSValue::*AVSValue_GetType)() const;

  // V10.1
  &VideoInfo::IsChannelMaskKnown,           //   bool    (VideoInfo::*IsChannelMaskKnown)()  const;
  &VideoInfo::SetChannelMask,               //   void    (VideoInfo::*SetChannelMask)();
  &VideoInfo::GetChannelMask,               //   int     (VideoInfo::*GetChannelMask)()  const;

  // V13
  &VideoInfo::Is440,                        //   bool    (VideoInfo::*Is440)()  const;
  &VideoInfo::IsYA,                         //   bool    (VideoInfo::*IsYA)()  const;
  &VideoInfo::Is410,                        //   bool    (VideoInfo::*Is410)()  const;
  &VideoInfo::Is411,                        //   bool    (VideoInfo::*Is411)()  const;

  // a single { nullptr } initializes the whole placeholder array
  { nullptr },                              // void          (VideoInfo::* reserved2[64 - 31])(); // Reserve pointer space for Avisynth+

/**********************************************************************/
  // AviSynth Neo additions
  &GetAvsEnv
  // Most Neo linkage entries are moved to standard avs+ place.
  // frame property logic has been replaced entirely

// this part should be identical with struct AVS_Linkage in avisynth.h

/**********************************************************************/
};                                          // }

extern __declspec(dllexport) const AVS_Linkage* const AVS_linkage = &avs_linkage;


/**********************************************************************/
// in UserPlugin.cpp
#if 0

#include "avisynth.h"

/* New 2.6 requirment!!! */
// Declare and initialise server pointers static storage.
const AVS_Linkage *AVS_linkage = 0;

/* New 2.6 requirment!!! */
// DLL entry point called from LoadPlugin() to setup a user plugin.
extern "C" __declspec(dllexport) const char* __stdcall AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors) {

  /* New 2.6 requirment!!! */
  // Save the server pointers.
  AVS_linkage = vectors;

  // Add the name of our function
  env->AddFunction("Plugin", "c", Create_Plugin, 0);

  // Return plugin text identifier.
  return "Plugin";
}

#endif
/**********************************************************************/

