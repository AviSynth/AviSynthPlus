// Avisynth C Interface
// Based on Copyright 2003 Kevin Atkinson
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//

#include <avisynth.h>
#include <avisynth_c.h>
#include "AVSMap.h"
#include "internal.h"

#ifdef AVS_WINDOWS
#include <avs/win.h>
#else
#include <avs/posix.h>
#endif

#include <algorithm>
#include <cstdarg>


struct AVS_Clip
{
  PClip clip;
  IScriptEnvironment* env;
  const char* error;
  AVS_Clip() : env(0), error(0) {}
};

class C_VideoFilter : public IClip {
public: // but don't use
  AVS_Clip child;
  AVS_ScriptEnvironment env;
  AVS_FilterInfo d;
public:
  C_VideoFilter() { memset(&d, 0, sizeof(d)); }
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
  void __stdcall GetAudio(void* buf, int64_t start, int64_t count, IScriptEnvironment* env);
  const VideoInfo& __stdcall GetVideoInfo();
  bool __stdcall GetParity(int n);
  int __stdcall SetCacheHints(int cachehints, int frame_range);
  AVSC_CC ~C_VideoFilter();
};

/////////////////////////////////////////////////////////////////////
//
//
//

extern "C"
int AVSC_CC avs_is_rgb48(const AVS_VideoInfo * p)
{
  return ((p->pixel_type & AVS_CS_BGR24) == AVS_CS_BGR24) && ((p->pixel_type & AVS_CS_SAMPLE_BITS_MASK) == AVS_CS_SAMPLE_BITS_16);
}

extern "C"
int AVSC_CC avs_is_rgb64(const AVS_VideoInfo * p)
{
  return ((p->pixel_type & AVS_CS_BGR32) == AVS_CS_BGR32) && ((p->pixel_type & AVS_CS_SAMPLE_BITS_MASK) == AVS_CS_SAMPLE_BITS_16);
}

extern "C"
int AVSC_CC avs_is_yv24(const AVS_VideoInfo * p)
{
  return (p->pixel_type & AVS_CS_PLANAR_MASK) == (AVS_CS_YV24 & AVS_CS_PLANAR_FILTER);
}

extern "C"
int AVSC_CC avs_is_yv16(const AVS_VideoInfo * p)
{
  return (p->pixel_type & AVS_CS_PLANAR_MASK) == (AVS_CS_YV16 & AVS_CS_PLANAR_FILTER);
}

extern "C"
int AVSC_CC avs_is_yv12(const AVS_VideoInfo * p)
{
  return (p->pixel_type & AVS_CS_PLANAR_MASK) == (AVS_CS_YV12 & AVS_CS_PLANAR_FILTER);
}

extern "C"
int AVSC_CC avs_is_yv411(const AVS_VideoInfo * p)
{
  return (p->pixel_type & AVS_CS_PLANAR_MASK) == (AVS_CS_YV411 & AVS_CS_PLANAR_FILTER);
}

extern "C"
int AVSC_CC avs_is_y8(const AVS_VideoInfo * p)
{
  return (p->pixel_type & AVS_CS_PLANAR_MASK) == (AVS_CS_Y8 & AVS_CS_PLANAR_FILTER);
}

extern "C"
int AVSC_CC avs_is_yuv444p16(const AVS_VideoInfo * p)
{
  return (p->pixel_type & AVS_CS_PLANAR_MASK) == (AVS_CS_YUV444P16 & AVS_CS_PLANAR_FILTER);
}

extern "C"
int AVSC_CC avs_is_yuv422p16(const AVS_VideoInfo * p)
{
  return (p->pixel_type & AVS_CS_PLANAR_MASK) == (AVS_CS_YUV422P16 & AVS_CS_PLANAR_FILTER);
}

extern "C"
int AVSC_CC avs_is_yuv420p16(const AVS_VideoInfo * p)
{
  return (p->pixel_type & AVS_CS_PLANAR_MASK) == (AVS_CS_YUV420P16 & AVS_CS_PLANAR_FILTER);
}

extern "C"
int AVSC_CC avs_is_y16(const AVS_VideoInfo * p)
{
  return (p->pixel_type & AVS_CS_PLANAR_MASK) == (AVS_CS_Y16 & AVS_CS_PLANAR_FILTER);
}

extern "C"
int AVSC_CC avs_is_yuv444ps(const AVS_VideoInfo * p)
{
  return (p->pixel_type & AVS_CS_PLANAR_MASK) == (AVS_CS_YUV444PS & AVS_CS_PLANAR_FILTER);
}

extern "C"
int AVSC_CC avs_is_yuv422ps(const AVS_VideoInfo * p)
{
  return (p->pixel_type & AVS_CS_PLANAR_MASK) == (AVS_CS_YUV422PS & AVS_CS_PLANAR_FILTER);
}

extern "C"
int AVSC_CC avs_is_yuv420ps(const AVS_VideoInfo * p)
{
  return (p->pixel_type & AVS_CS_PLANAR_MASK) == (AVS_CS_YUV420PS & AVS_CS_PLANAR_FILTER);
}

extern "C"
int AVSC_CC avs_is_y32(const AVS_VideoInfo * p)
{
  return (p->pixel_type & AVS_CS_PLANAR_MASK) == (AVS_CS_Y32 & AVS_CS_PLANAR_FILTER);
}

extern "C"
int AVSC_CC avs_is_444(const AVS_VideoInfo * p)
{
  return ((p->pixel_type & AVS_CS_PLANAR_MASK & ~AVS_CS_SAMPLE_BITS_MASK) == (AVS_CS_GENERIC_YUV444 & AVS_CS_PLANAR_FILTER)) ||
    ((p->pixel_type & AVS_CS_PLANAR_MASK & ~AVS_CS_SAMPLE_BITS_MASK) == (AVS_CS_GENERIC_YUVA444 & AVS_CS_PLANAR_FILTER));
}

extern "C"
int AVSC_CC avs_is_422(const AVS_VideoInfo * p)
{
  return ((p->pixel_type & AVS_CS_PLANAR_MASK & ~AVS_CS_SAMPLE_BITS_MASK) == (AVS_CS_GENERIC_YUV422 & AVS_CS_PLANAR_FILTER)) ||
    ((p->pixel_type & AVS_CS_PLANAR_MASK & ~AVS_CS_SAMPLE_BITS_MASK) == (AVS_CS_GENERIC_YUVA422 & AVS_CS_PLANAR_FILTER));
}

extern "C"
int AVSC_CC avs_is_420(const AVS_VideoInfo * p)
{
  return ((p->pixel_type & AVS_CS_PLANAR_MASK & ~AVS_CS_SAMPLE_BITS_MASK) == (AVS_CS_GENERIC_YUV420 & AVS_CS_PLANAR_FILTER)) ||
    ((p->pixel_type & AVS_CS_PLANAR_MASK & ~AVS_CS_SAMPLE_BITS_MASK) == (AVS_CS_GENERIC_YUVA420 & AVS_CS_PLANAR_FILTER));
}

extern "C"
int AVSC_CC avs_is_y(const AVS_VideoInfo * p)
{
  return (p->pixel_type & AVS_CS_PLANAR_MASK & ~AVS_CS_SAMPLE_BITS_MASK) == (AVS_CS_GENERIC_Y & AVS_CS_PLANAR_FILTER);
}

extern "C"
int AVSC_CC avs_is_color_space(const AVS_VideoInfo * p, int c_space)
{
  return avs_is_planar(p) ?
    ((p->pixel_type & AVS_CS_PLANAR_MASK) == (c_space & AVS_CS_PLANAR_FILTER))
    :
    (((p->pixel_type & ~AVS_CS_SAMPLE_BITS_MASK & c_space) == (c_space & ~AVS_CS_SAMPLE_BITS_MASK)) && // RGB got sample bits
      ((p->pixel_type & AVS_CS_SAMPLE_BITS_MASK) == (c_space & AVS_CS_SAMPLE_BITS_MASK)));
}

extern "C"
int AVSC_CC avs_is_yuva(const AVS_VideoInfo * p)
{
  return !!(p->pixel_type & AVS_CS_YUVA);
}

extern "C"
int AVSC_CC avs_is_planar_rgb(const AVS_VideoInfo * p)
{
  return !!(p->pixel_type & AVS_CS_PLANAR) && !!(p->pixel_type & AVS_CS_BGR) && !!(p->pixel_type & AVS_CS_RGB_TYPE);
}

extern "C"
int AVSC_CC avs_is_planar_rgba(const AVS_VideoInfo * p)
{
  return !!(p->pixel_type & AVS_CS_PLANAR) && !!(p->pixel_type & AVS_CS_BGR) && !!(p->pixel_type & AVS_CS_RGBA_TYPE);
}

extern "C"
int AVSC_CC avs_get_plane_width_subsampling(const AVS_VideoInfo * p, int plane)
{
  try {
    return ((VideoInfo*)p)->GetPlaneWidthSubsampling(plane);
  }
  catch (const AvisynthError& err) {
    (void)err;  // silence warning about unused variable; variable is kept for debugging
    return -1;
  }
}

extern "C"
int AVSC_CC avs_get_plane_height_subsampling(const AVS_VideoInfo * p, int plane)
{
  try {
    return ((VideoInfo*)p)->GetPlaneHeightSubsampling(plane);
  }
  catch (const AvisynthError& err) {
    (void)err;  // silence warning about unused variable; variable is kept for debugging
    return -1;
  }
}

extern "C"
int AVSC_CC avs_bits_per_pixel(const AVS_VideoInfo * p)
{
  return ((VideoInfo*)p)->BitsPerPixel();
}

extern "C"
int AVSC_CC avs_bytes_from_pixels(const AVS_VideoInfo * p, int pixels)
{
  return ((VideoInfo*)p)->BytesFromPixels(pixels);
}

// This method should be called avs_row_size_p,
// but we won't change it anymore to avoid breaking
// the interface.
extern "C"
int AVSC_CC avs_row_size(const AVS_VideoInfo * p, int plane)
{
  return ((VideoInfo*)p)->RowSize(plane);
}

extern "C"
int AVSC_CC avs_bmp_size(const AVS_VideoInfo * vi)
{
  return ((VideoInfo*)vi)->BMPSize();
}


/////////////////////////////////////////////////////////////////////
//
//
//

extern "C"
int AVSC_CC avs_get_pitch_p(const AVS_VideoFrame * p, int plane)
{
  // Memo: the lines in class PVideoFrame:
  //   VideoFrame* p;
  //   VideoFrame* operator->() const { return p; }
  // help when you use the arrow operator on a PVideoFrame object, it will return 
  // the "VideoFrame* p" pointer, allowing you to access members of the 
  // VideoFrame class directly.
  return (*(const PVideoFrame*)&p)->GetPitch(plane);
}

extern "C"
int AVSC_CC avs_get_row_size_p(const AVS_VideoFrame * p, int plane)
{
  return (*(const PVideoFrame*)&p)->GetRowSize(plane);
}

extern "C"
int AVSC_CC avs_get_height_p(const AVS_VideoFrame * p, int plane)
{
  return (*(const PVideoFrame*)&p)->GetHeight(plane);
}

extern "C"
const BYTE * AVSC_CC avs_get_read_ptr_p(const AVS_VideoFrame * p, int plane)
{
  return (*(const PVideoFrame*)&p)->GetReadPtr(plane);
}

extern "C"
int AVSC_CC avs_is_writable(const AVS_VideoFrame * p)
{
  return (*(const PVideoFrame*)&p)->IsWritable() ? 1 : 0;
}

// V9
extern "C"
int AVSC_CC avs_is_property_writable(const AVS_VideoFrame * p)
{
  return (*(const PVideoFrame*)&p)->IsPropertyWritable() ? 1 : 0;
}

// V10
extern "C"
int AVSC_CC avs_video_frame_get_pixel_type(const AVS_VideoFrame * p)
{
  return (*(const PVideoFrame*)&p)->GetPixelType();
}

// V10
void AVSC_CC avs_video_frame_amend_pixel_type(AVS_VideoFrame* p, int new_pixel_type)
{
  (*(PVideoFrame*)&p)->AmendPixelType(new_pixel_type);
}

extern "C"
BYTE * AVSC_CC avs_get_write_ptr_p(const AVS_VideoFrame * p, int plane)
{
  return (*(const PVideoFrame*)&p)->GetWritePtr(plane);
}

extern "C"
void AVSC_CC avs_release_video_frame(AVS_VideoFrame * f)
{
  ((PVideoFrame*)&f)->~PVideoFrame();
}

extern "C"
AVS_VideoFrame * AVSC_CC avs_copy_video_frame(AVS_VideoFrame * f)
{
  AVS_VideoFrame* fnew;
  new ((PVideoFrame*)&fnew) PVideoFrame(*(PVideoFrame*)&f);
  return fnew;
}


extern "C"
int AVSC_CC avs_num_components(const AVS_VideoInfo * p)
{
  return ((VideoInfo*)p)->NumComponents();
}

extern "C"
int AVSC_CC avs_component_size(const AVS_VideoInfo * p)
{
  return ((VideoInfo*)p)->ComponentSize();
}

extern "C"
int AVSC_CC avs_bits_per_component(const AVS_VideoInfo * p)
{
  return ((VideoInfo*)p)->BitsPerComponent();
}

// V10.1
extern "C"
bool AVSC_CC avs_is_channel_mask_known(const AVS_VideoInfo * p)
{
  return ((VideoInfo*)p)->IsChannelMaskKnown();
}

extern "C"
void AVSC_CC avs_set_channel_mask(const AVS_VideoInfo * p, bool isChannelMaskKnown, unsigned int dwChannelMask)
{
  ((VideoInfo*)p)->SetChannelMask(isChannelMaskKnown, dwChannelMask);
}

extern "C"
unsigned int AVSC_CC avs_get_channel_mask(const AVS_VideoInfo * p)
{
  return ((VideoInfo*)p)->GetChannelMask();
}

//////////////////////////////////////////////////////////
//
// frame properties
//
extern "C"
void AVSC_CC avs_copy_frame_props(AVS_ScriptEnvironment * p, const AVS_VideoFrame * src, AVS_VideoFrame * dst)
{
  p->error = 0;
  try {
    p->env->copyFrameProps(*(const PVideoFrame*)&src, *(PVideoFrame*)&dst);
  }
  catch (const AvisynthError& err) {
    p->error = err.msg;
  }
}

extern "C"
const AVS_Map * AVSC_CC avs_get_frame_props_ro(AVS_ScriptEnvironment * p, const AVS_VideoFrame * frame)
{
  p->error = 0;
  try {
    return (const AVS_Map*)(p->env->getFramePropsRO(*(const PVideoFrame*)&frame));
  }
  catch (const AvisynthError& err) {
    p->error = err.msg;
    return 0;
  }
}

extern "C"
AVS_Map * AVSC_CC avs_get_frame_props_rw(AVS_ScriptEnvironment * p, AVS_VideoFrame * frame)
{
  p->error = 0;
  try {
    return (AVS_Map*)(p->env->getFramePropsRW(*(PVideoFrame*)&frame));
  }
  catch (const AvisynthError& err) {
    p->error = err.msg;
    return 0;
  }
}

extern "C"
int AVSC_CC avs_prop_num_keys(AVS_ScriptEnvironment * p, const AVS_Map * map)
{
  p->error = 0;
  try {
    return (p->env->propNumKeys((const AVSMap*)map));
  }
  catch (const AvisynthError& err) {
    p->error = err.msg;
    return 0;
  }
}

extern "C"
const char* AVSC_CC avs_prop_get_key(AVS_ScriptEnvironment * p, const AVS_Map * map, int index)
{
  p->error = 0;
  try {
    const char* key = (p->env->propGetKey((const AVSMap*)map, index));
    return p->env->SaveString(key);
  }
  catch (const AvisynthError& err) {
    p->error = err.msg;
    return 0;
  }
}

extern "C"
int AVSC_CC avs_prop_num_elements(AVS_ScriptEnvironment * p, const AVS_Map * map, const char* key)
{
  p->error = 0;
  try {
    return (p->env->propNumElements((const AVSMap*)map, key));
  }
  catch (const AvisynthError& err) {
    p->error = err.msg;
    return 0;
  }
}

extern "C"
char AVSC_CC avs_prop_get_type(AVS_ScriptEnvironment * p, const AVS_Map * map, const char* key)
{
  p->error = 0;
  try {
    return (p->env->propGetType((const AVSMap*)map, key));
  }
  catch (const AvisynthError& err) {
    p->error = err.msg;
    return 0;
  }
}

extern "C"
int AVSC_CC avs_prop_delete_key(AVS_ScriptEnvironment * p, AVS_Map * map, const char* key)
{
  p->error = 0;
  try {
    return (p->env->propDeleteKey((AVSMap*)map, key));
  }
  catch (const AvisynthError& err) {
    p->error = err.msg;
    return 0;
  }
}

extern "C"
int64_t AVSC_CC avs_prop_get_int(AVS_ScriptEnvironment * p, const AVS_Map * map, const char* key, int index, int* error)
{
  p->error = 0;
  try {
    return (p->env->propGetInt((const AVSMap*)map, key, index, error));
  }
  catch (const AvisynthError& err) {
    p->error = err.msg;
    return 0;
  }
}

extern "C"
int AVSC_CC avs_prop_get_int_saturated(AVS_ScriptEnvironment* p, const AVS_Map* map, const char* key, int index, int* error)
{
  p->error = 0;
  try {
    return (p->env->propGetIntSaturated((const AVSMap*)map, key, index, error));
  }
  catch (const AvisynthError& err) {
    p->error = err.msg;
    return 0;
  }
}

extern "C"
double AVSC_CC avs_prop_get_float(AVS_ScriptEnvironment * p, const AVS_Map * map, const char* key, int index, int* error)
{
  p->error = 0;
  try {
    return (p->env->propGetFloat((const AVSMap*)map, key, index, error));
  }
  catch (const AvisynthError& err) {
    p->error = err.msg;
    return 0;
  }
}

// v11
extern "C"
float AVSC_CC avs_prop_get_float_saturated(AVS_ScriptEnvironment* p, const AVS_Map* map, const char* key, int index, int* error)
{
  p->error = 0;
  try {
    return (p->env->propGetFloatSaturated((const AVSMap*)map, key, index, error));
  }
  catch (const AvisynthError& err) {
    p->error = err.msg;
    return 0;
  }
}

extern "C"
const char* AVSC_CC avs_prop_get_data(AVS_ScriptEnvironment * p, const AVS_Map * map, const char* key, int index, int* error)
{
  p->error = 0;
  try {
    const char* data = p->env->propGetData((const AVSMap*)map, key, index, error);
    if (!data)
      return nullptr;
    else
      return data;
  }
  catch (const AvisynthError& err) {
    p->error = err.msg;
    return 0;
  }
}

extern "C"
int AVSC_CC avs_prop_get_data_size(AVS_ScriptEnvironment * p, const AVS_Map * map, const char* key, int index, int* error)
{
  p->error = 0;
  try {
    return (p->env->propGetDataSize((const AVSMap*)map, key, index, error));
  }
  catch (const AvisynthError& err) {
    p->error = err.msg;
    return 0;
  }
}

extern "C"
int AVSC_CC avs_prop_get_data_type_hint(AVS_ScriptEnvironment* p, const AVS_Map* map, const char* key, int index, int* error)
{
  p->error = 0;
  try {
    return (p->env->propGetDataTypeHint((const AVSMap*)map, key, index, error));
  }
  catch (const AvisynthError& err) {
    p->error = err.msg;
    return 0;
  }
}

extern "C"
AVS_Clip* AVSC_CC avs_prop_get_clip(AVS_ScriptEnvironment* p, const AVS_Map* map, const char* key, int index, int* error)
{
  p->error = 0;
  try {
    PClip c0 = p->env->propGetClip((const AVSMap*)map, key, index, error);
    AVS_Clip* c;
    new((PClip*)&c) PClip(c0);
    return c;
  }
  catch (const AvisynthError& err) {
    p->error = err.msg;
    return 0;
  }
}

extern "C"
const AVS_VideoFrame * AVSC_CC avs_prop_get_frame(AVS_ScriptEnvironment * p, const AVS_Map * map, const char* key, int index, int* error)
{
  p->error = 0;
  try {
    const PVideoFrame f0 = p->env->propGetFrame((const AVSMap*)map, key, index, error);
    AVS_VideoFrame* f;
    new((PVideoFrame*)&f) PVideoFrame(f0);
    return f;
  }
  catch (const AvisynthError& err) {
    p->error = err.msg;
    return 0;
  }
}

extern "C"
int AVSC_CC avs_prop_set_int(AVS_ScriptEnvironment * p, AVS_Map * map, const char* key, int64_t i, int append)
{
  p->error = 0;
  try {
    return (p->env->propSetInt((AVSMap*)map, key, i, append));
  }
  catch (const AvisynthError& err) {
    p->error = err.msg;
    return 0;
  }
}

extern "C"
int AVSC_CC avs_prop_set_float(AVS_ScriptEnvironment * p, AVS_Map * map, const char* key, double d, int append)
{
  p->error = 0;
  try {
    return (p->env->propSetFloat((AVSMap*)map, key, d, append));
  }
  catch (const AvisynthError& err) {
    p->error = err.msg;
    return 0;
  }
}

extern "C"
int AVSC_CC avs_prop_set_data(AVS_ScriptEnvironment * p, AVS_Map * map, const char* key, const char* d, int length, int append)
{
  // length = -1 -> auto strlen
  p->error = 0;
  try {
    return (p->env->propSetData((AVSMap*)map, key, d, length, append));
  }
  catch (const AvisynthError& err) {
    p->error = err.msg;
    return 0;
  }
}

extern "C"
int AVSC_CC avs_prop_set_data_h(AVS_ScriptEnvironment* p, AVS_Map* map, const char* key, const char* d, int length, int type, int append)
{
  // length = -1 -> auto strlen
  p->error = 0;
  try {
    return (p->env->propSetDataH((AVSMap*)map, key, d, length, type, append));
  }
  catch (const AvisynthError& err) {
    p->error = err.msg;
    return 0;
  }
}

extern "C"
int AVSC_CC avs_prop_set_clip(AVS_ScriptEnvironment * p, AVS_Map * map, const char* key, AVS_Clip * clip, int append)
{
   p->error = 0;
  try {
    return (p->env->propSetClip((AVSMap*)map, key, *(PClip*)clip, append));
  }
  catch (const AvisynthError& err) {
    p->error = err.msg;
    return 0;
  }
}

extern "C"
int AVSC_CC avs_prop_set_frame(AVS_ScriptEnvironment * p, AVS_Map * map, const char* key, const AVS_VideoFrame * frame, int append)
{
  p->error = 0;
  try {
    return (p->env->propSetFrame((AVSMap*)map, key, *(PVideoFrame*)&frame, append));
  }
  catch (const AvisynthError& err) {
    p->error = err.msg;
    return 0;
  }
}


extern "C"
const int64_t * AVSC_CC avs_prop_get_int_array(AVS_ScriptEnvironment * p, const AVS_Map * map, const char* key, int* error)
{
  p->error = 0;
  try {
    return p->env->propGetIntArray((const AVSMap*)map, key, error);
  }
  catch (const AvisynthError& err) {
    p->error = err.msg;
    return 0;
  }
}

extern "C"
const double* AVSC_CC avs_prop_get_float_array(AVS_ScriptEnvironment * p, const AVS_Map * map, const char* key, int* error)
{
  p->error = 0;
  try {
    return p->env->propGetFloatArray((const AVSMap*)map, key, error);
  }
  catch (const AvisynthError& err) {
    p->error = err.msg;
    return 0;
  }
}

extern "C"
int AVSC_CC avs_prop_set_int_array(AVS_ScriptEnvironment * p, AVS_Map * map, const char* key, const int64_t * i, int size)
{
  p->error = 0;
  try {
    return (p->env->propSetIntArray((AVSMap*)map, key, i, size));
  }
  catch (const AvisynthError& err) {
    p->error = err.msg;
    return 0;
  }
}

extern "C"
int AVSC_CC avs_prop_set_float_array(AVS_ScriptEnvironment * p, AVS_Map * map, const char* key, const double* d, int size)
{
  p->error = 0;
  try {
    return (p->env->propSetFloatArray((AVSMap*)map, key, d, size));
  }
  catch (const AvisynthError& err) {
    p->error = err.msg;
    return 0;
  }
}

extern "C"
void AVSC_CC avs_clear_map(AVS_ScriptEnvironment * p, AVS_Map * map)
{
  p->error = 0;
  try {
    p->env->clearMap((AVSMap*)map);
  }
  catch (const AvisynthError& err) {
    p->error = err.msg;
  }
}



/////////////////////////////////////////////////////////////////////
//
// C_VideoFilter
//

PVideoFrame C_VideoFilter::GetFrame(int n, IScriptEnvironment* env)
{
  if (d.get_frame) {
    d.error = 0;
    AVS_VideoFrame* f = d.get_frame(&d, n);
    if (d.error)
      throw AvisynthError(d.error);
    if (d.child != NULL && d.child->error)
      throw AvisynthError(d.child->error);
    PVideoFrame fr((VideoFrame*)f);
    ((PVideoFrame*)&f)->~PVideoFrame();
    return fr;
  }
  else {
    return d.child->clip->GetFrame(n, env);
  }
}

void __stdcall C_VideoFilter::GetAudio(void* buf, int64_t start, int64_t count, IScriptEnvironment* env)
{
  if (d.get_audio) {
    d.error = 0;
    d.get_audio(&d, buf, start, count);
    if (d.error)
      throw AvisynthError(d.error);
  }
  else {
    d.child->clip->GetAudio(buf, start, count, env);
  }
}

const VideoInfo& __stdcall C_VideoFilter::GetVideoInfo()
{
  return *(VideoInfo*)&d.vi;
}

bool __stdcall C_VideoFilter::GetParity(int n)
{
  if (d.get_parity) {
    d.error = 0;
    int res = d.get_parity(&d, n);
    if (d.error)
      throw AvisynthError(d.error);
    return !!res;
  }
  else {
    return d.child->clip->GetParity(n);
  }
}

int __stdcall C_VideoFilter::SetCacheHints(int cachehints, int frame_range)
{
  if (d.set_cache_hints) {
    d.error = 0;
    int res = d.set_cache_hints(&d, cachehints, frame_range);
    if (d.error)
      throw AvisynthError(d.error);
    return res;
  }
  // We do not pass cache requests upwards, only to the hosted filter.
  return 0;
}

C_VideoFilter::~C_VideoFilter()
{
  if (d.free_filter)
    d.free_filter(&d);
}

/////////////////////////////////////////////////////////////////////
//
// AVS_Clip
//

extern "C"
void AVSC_CC avs_release_clip(AVS_Clip * p)
{
  delete p;
}

AVS_Clip* AVSC_CC avs_copy_clip(AVS_Clip* p)
{
  return new AVS_Clip(*p);
}

extern "C"
const char* AVSC_CC avs_clip_get_error(AVS_Clip * p) // return 0 if no error
{
  return p->error;
}

extern "C"
int AVSC_CC avs_get_version(AVS_Clip * p)
{
  return p->clip->GetVersion();
}

extern "C"
const AVS_VideoInfo * AVSC_CC avs_get_video_info(AVS_Clip * p)
{
  return  (const AVS_VideoInfo*)&p->clip->GetVideoInfo();
}


extern "C"
AVS_VideoFrame * AVSC_CC avs_get_frame(AVS_Clip * p, int n)
{
  p->error = 0;
  try {
    PVideoFrame f0 = p->clip->GetFrame(n, p->env);
    AVS_VideoFrame* f;
    new((PVideoFrame*)&f) PVideoFrame(f0);
    return f;
  }
  catch (const AvisynthError& err) {
    p->error = err.msg;
    return 0;
  }
}

extern "C"
int AVSC_CC avs_get_parity(AVS_Clip * p, int n) // return field parity if field_based, else parity of first field in frame
{
  try {
    p->error = 0;
    return p->clip->GetParity(n);
  }
  catch (const AvisynthError& err) {
    p->error = err.msg;
    return -1;
  }
}

extern "C"
int AVSC_CC avs_get_audio(AVS_Clip * p, void* buf, int64_t start, int64_t count) // start and count are in samples
{
  try {
    p->error = 0;
    p->clip->GetAudio(buf, start, count, p->env);
    return 0;
  }
  catch (const AvisynthError& err) {
    p->error = err.msg;
    return -1;
  }
}

extern "C"
int AVSC_CC avs_set_cache_hints(AVS_Clip * p, int cachehints, int frame_range)  // We do not pass cache requests upwards, only to the next filter.
{
  try {
    p->error = 0;
    return p->clip->SetCacheHints(cachehints, frame_range);
  }
  catch (const AvisynthError& err) {
    p->error = err.msg;
    return -1;
  }
}

//////////////////////////////////////////////////////////////////
//
//
//
extern "C"
AVS_Clip * AVSC_CC avs_take_clip(AVS_Value v, AVS_ScriptEnvironment * env)
{
  AVS_Clip* c = new AVS_Clip;
  c->env = env->env;
  c->clip = (IClip*)v.d.clip;
  return c;
}

// v11 API AVS_Value type checkers.
extern "C"
int AVSC_CC avs_val_defined(AVS_Value v) { return ((const AVSValue*)(&v))->Defined() ? 1 : 0; }
extern "C"
int AVSC_CC avs_val_is_clip(AVS_Value v) { return ((const AVSValue*)(&v))->IsClip() ? 1 : 0; }
extern "C"
int AVSC_CC avs_val_is_bool(AVS_Value v) { return ((const AVSValue*)(&v))->IsBool() ? 1 : 0; }
extern "C"
int AVSC_CC avs_val_is_int(AVS_Value v) { return ((const AVSValue*)(&v))->IsInt() ? 1 : 0; }
extern "C"
int AVSC_CC avs_val_is_long_strict(AVS_Value v) { return ((const AVSValue*)(&v))->IsLongStrict() ? 1 : 0; }
extern "C"
int AVSC_CC avs_val_is_float(AVS_Value v) { return ((const AVSValue*)(&v))->IsFloat() ? 1 : 0; }
extern "C"
int AVSC_CC avs_val_is_floatf_strict(AVS_Value v) { return ((const AVSValue*)(&v))->IsFloatfStrict() ? 1 : 0; }
extern "C"
int AVSC_CC avs_val_is_string(AVS_Value v) { return ((const AVSValue*)(&v))->IsString() ? 1 : 0; }
extern "C"
int AVSC_CC avs_val_is_array(AVS_Value v) { return ((const AVSValue*)(&v))->IsArray() ? 1 : 0; }
extern "C"
int AVSC_CC avs_val_is_error(AVS_Value v) { return v.type == 'e' ? 1 : 0; }


// v11 API AVS_Value setters
extern "C"
void AVSC_CC avs_set_to_error(AVS_Value* v, const char* v0) { new(v) AVSValue(v0); /*string->error*/ v->type = 'e'; }
extern "C"
void AVSC_CC avs_set_to_bool(AVS_Value* v, int v0) { new(v) AVSValue(v0 != 0); }
extern "C"
void AVSC_CC avs_set_to_int(AVS_Value* v, int v0) { new(v) AVSValue(v0); }
extern "C"
void AVSC_CC avs_set_to_float(AVS_Value* v, float v0) { new(v) AVSValue(v0); }
extern "C"
void AVSC_CC avs_set_to_string(AVS_Value* v, const char* v0) { new(v) AVSValue(v0); }
extern "C"
void AVSC_CC avs_set_to_double(AVS_Value* v, double d) { new(v) AVSValue(d); }
extern "C"
void AVSC_CC avs_set_to_long(AVS_Value* v, int64_t l) { new(v) AVSValue(l); }
extern "C"
void AVSC_CC avs_set_to_array(AVS_Value* v, AVS_Value* src, int size) { new(v) AVSValue((AVSValue *)src, size); }
extern "C"
void AVSC_CC avs_set_to_void(AVS_Value* v) { new(v) AVSValue(); }
// existed pre V11
extern "C"
void AVSC_CC avs_set_to_clip(AVS_Value* v, AVS_Clip* c) { new(v) AVSValue(c->clip); }

extern "C"
void AVSC_CC avs_copy_value(AVS_Value * dest, AVS_Value src)
{
#if 0
  // no need to guard multidim arrays. avs_release_value will release properly.
  // true: don't copy array elements recursively
  new(dest) AVSValue(*(const AVSValue*)&src, true);
#endif
  // CONSTRUCTOR9-->Assign(&v, init=true) ensures that the original content
  // is simply overwritten and will not be freed.
  new(dest) AVSValue(*(const AVSValue*)&src);
}

// Releases/free resources contained in an AVS_Value
// Such types are: 
// - clip
// - function (not supported on C interface)
// - double and long on 32 bit architects
// - Avisynth+ arrays
// Since AVS_Value is just a struct passed by value, nothing else happens.
// If AVS_Value has no extra resource to free up, nothing is done.
// release is mandatory for results of:
// - avs_invoke 
// - avs_copy_value 
// - avs_set_to_double (x86)
// - avs_set_to_long (x86)
// - avs_set_to_array, avs_set_to_clip
// - in general: all avs_set_xxx values

// Note:
// Unlike dynamic arrays, which have a full create-copy-modify-release cycle,
// C plugins/clients can use in-source arrays, typically for assembling function
// parameters ("args"). Do not call avs_release_value on such arrays, as it causes
// a crash since the array itself and the elements are not maintained by the 
// Avisynth core.However, it is good practice to release the array content
// with avs_release_value one-by-one for the resource-allocated types mentioned above.

extern "C"
void AVSC_CC avs_release_value(AVS_Value v)
{
#if 0
  // This would prevent crashes when a C client calls it for an array.
  // However, since Avisynth+ arrays can be returned by avs_invoke, avs_copy_value,
  // and avs_set_to_array, this exemption is no longer valid.
  // This rule was not explicitly written earlier and must be enforced in clients/plugins,
  // see the Python wrapper issue in AvsPmod, which was fixed.
  if (((AVSValue*)&v)->IsArray()) {
    // signing for destructor: don't free array elements
    ((AVSValue*)&v)->MarkArrayAsNonDeepCopy();
  }
#endif  
  ((AVSValue*)&v)->~AVSValue();
}

// API AVS_Value getters, like INLINE versions
extern "C"
int AVSC_CC avs_get_as_bool(AVS_Value v) { return ((AVSValue*)&v)->AsBool() ? 1 : 0; }

extern "C"
AVS_Clip* AVSC_CC avs_get_as_clip(AVS_Value v, AVS_ScriptEnvironment* env) {
  // like the existing avs_take_clip, to fit in the avs_get_as_xxxx line
  AVS_Clip* c = new AVS_Clip;
  c->env = env->env;
  c->clip = (IClip*)(v.d.clip);
  return c;
}

extern "C"
int AVSC_CC avs_get_as_int(AVS_Value v) { return ((AVSValue*)&v)->AsInt(); }
extern "C"
int64_t AVSC_CC avs_get_as_long(AVS_Value v) { return ((AVSValue*)&v)->AsLong(); }
extern "C"
const char* AVSC_CC avs_get_as_string(AVS_Value v) { return ((AVSValue*)&v)->AsString(); }
extern "C"
double AVSC_CC avs_get_as_float(AVS_Value v) { return ((AVSValue*)&v)->AsFloat(); }
extern "C"
const char* AVSC_CC avs_get_as_error(AVS_Value v) {
  if (v.type == 'e') {
    v.type = 's'; // 'e'rror is unknown in c++ api
    return ((AVSValue*)&v)->AsString();
  }
  return nullptr;
}
extern "C"
const AVS_Value * AVSC_CC avs_get_as_array(AVS_Value v) { return v.d.array; }
extern "C"
AVS_Value AVSC_CC avs_get_array_elt(AVS_Value v, int index) 
{ // just a dumb data copy, no ref counting, no resource handling
  return avs_is_array(v) ? v.d.array[index] : v;
}
extern "C"
int AVSC_CC avs_get_array_size(AVS_Value v) { return ((AVSValue*)&v)->ArraySize(); }


//////////////////////////////////////////////////////////////////
//
//
//

extern "C"
AVS_Clip * AVSC_CC avs_new_c_filter(AVS_ScriptEnvironment * e,
  AVS_FilterInfo * *fi,
  AVS_Value child, int store_child)
{
  C_VideoFilter* f = new C_VideoFilter();
  AVS_Clip* ff = new AVS_Clip();
  ff->clip = f; // IClip descendant
  ff->env = e->env;
  f->env.env = e->env;
  f->d.env = &f->env;
  if (store_child) {
    _ASSERTE(child.type == 'c');
    f->child.clip = (IClip*)child.d.clip;
    f->child.env = e->env;
    f->d.child = &f->child;
  }
  *fi = &f->d;
  // (*fi)->free_filter will be set later by the plugin.
  // It is used in ~C_VideoFilter, which is called when the clip's 
  // reference count reaches zero and it is released. This serves 
  // as the filter destructor in C.
  if (child.type == 'c')
    f->d.vi = *(const AVS_VideoInfo*)(&((IClip*)child.d.clip)->GetVideoInfo());
  return ff;
}

/////////////////////////////////////////////////////////////////////
//
// AVS_ScriptEnvironment::add_function
//

struct C_VideoFilter_UserData {
  void* user_data;
  AVS_ApplyFunc func;
  AVS_ApplyFuncR func_r;
};

AVSValue __cdecl create_c_video_filter(AVSValue args, void* user_data,
  IScriptEnvironment* e0)
{
  C_VideoFilter_UserData* d = (C_VideoFilter_UserData*)user_data;
  AVS_ScriptEnvironment env;
  env.env = e0;
  env.error = NULL;

  AVS_Value res;
  if(d->func)
    res = (d->func)(&env, *(AVS_Value*)&args, d->user_data);
  else
    (d->func_r)(&env, (AVS_Value*)&res, *(AVS_Value*)&args, d->user_data); // new in v11: byref
  if (res.type == 'e') {
    throw AvisynthError(res.d.string);
  }
  else {
    AVSValue val;
    val = (*(const AVSValue*)&res);
    ((AVSValue*)&res)->~AVSValue();
    return val;
  }
}

extern "C"
int AVSC_CC
avs_add_function(AVS_ScriptEnvironment * p, const char* name, const char* params,
  AVS_ApplyFunc applyf, void* user_data)
{
  C_VideoFilter_UserData* dd, * d = new C_VideoFilter_UserData;
  p->error = 0;
  d->func = applyf; // return value struct
  d->func_r = nullptr;
  d->user_data = user_data;
  dd = (C_VideoFilter_UserData*)p->env->SaveString((const char*)d, sizeof(C_VideoFilter_UserData));
  delete d;
  try {
    p->env->AddFunction(name, params, create_c_video_filter, dd);
  }
  catch (AvisynthError& err) {
    p->error = err.msg;
    return -1;
  }
  return 0;
}

// v11
extern "C"
int AVSC_CC
avs_add_function_r(AVS_ScriptEnvironment* p, const char* name, const char* params,
  AVS_ApplyFuncR applyf, void* user_data)
{
  C_VideoFilter_UserData* dd, * d = new C_VideoFilter_UserData;
  p->error = 0;
  d->func = nullptr;
  d->func_r = applyf;  // return value among parameters byref struct (Python cref callbacks like it better)
  d->user_data = user_data;
  dd = (C_VideoFilter_UserData*)p->env->SaveString((const char*)d, sizeof(C_VideoFilter_UserData));
  delete d;
  try {
    p->env->AddFunction(name, params, create_c_video_filter, dd);
  }
  catch (AvisynthError& err) {
    p->error = err.msg;
    return -1;
  }
  return 0;
}

/////////////////////////////////////////////////////////////////////
//
// AVS_ScriptEnvironment
//

extern "C"
const char* AVSC_CC avs_get_error(AVS_ScriptEnvironment * p) // return 0 if no error
{
  return p->error;
}

extern "C"
int AVSC_CC avs_get_cpu_flags(AVS_ScriptEnvironment * p)
{
  p->error = 0;
  return p->env->GetCPUFlags();
}

extern "C"
char* AVSC_CC avs_save_string(AVS_ScriptEnvironment * p, const char* s, int length)
{
  p->error = 0;
  return p->env->SaveString(s, length);
}

extern "C"
char* AVSC_CC avs_sprintf(AVS_ScriptEnvironment * p, const char* fmt, ...)
{
  p->error = 0;
  va_list vl;
  va_start(vl, fmt);
  char* v = p->env->VSprintf(fmt, vl);
  va_end(vl);
  return v;
}

// note: val is really a va_list; I hope everyone typedefs va_list to a pointer
extern "C"
char* AVSC_CC avs_vsprintf(AVS_ScriptEnvironment * p, const char* fmt, va_list val)
{
  p->error = 0;
  return p->env->VSprintf(fmt, val);
}

extern "C"
int AVSC_CC avs_function_exists(AVS_ScriptEnvironment * p, const char* name)
{
  p->error = 0;
  return p->env->FunctionExists(name);
}

extern "C"
AVS_Value AVSC_CC avs_invoke(AVS_ScriptEnvironment* p, const char* name, AVS_Value args, const char** arg_names)
{
  AVS_Value v = { 0,0 };
  p->error = 0;
  try {
    // AVSValue(*(AVSValue*)&args) is deep-copying input array

    // Invoke has a special InvokePreV11C variant if pre V11 C interface 
    // detected, which converts a possible 64 bit result to 32 bit one for
    // compatibility. double->float, long->int. Nothing should be done here.
    AVSValue v0 = p->env->Invoke(name, *(AVSValue*)&args, arg_names);
    new ((AVSValue*)&v) AVSValue(v0);
  }
  catch (const IScriptEnvironment::NotFound&) {
    p->error = "Function Not Found";
  }
  catch (const AvisynthError& err) {
    p->error = err.msg;
  }
  if (p->error)
    v = avs_new_value_error(p->error);
  return v;
}

extern "C"
AVS_Value AVSC_CC avs_get_var(AVS_ScriptEnvironment * p, const char* name)
{
  AVS_Value v = { 0,0 };
  p->error = 0;
  try {
    AVSValue v0 = p->env->GetVar(name);
    new ((AVSValue*)&v) AVSValue(v0);
  }
  catch (const IScriptEnvironment::NotFound&) {}
  catch (const AvisynthError& err) {
    p->error = err.msg;
    v = avs_new_value_error(p->error);
  }
  return v;
}

extern "C"
int AVSC_CC avs_set_var(AVS_ScriptEnvironment * p, const char* name, AVS_Value val)
{
  p->error = 0;
  try {
    return p->env->SetVar(p->env->SaveString(name), *(const AVSValue*)(&val));
  }
  catch (const AvisynthError& err) {
    p->error = err.msg;
    return -1;
  }
}

extern "C"
int AVSC_CC avs_set_global_var(AVS_ScriptEnvironment * p, const char* name, AVS_Value val)
{
  p->error = 0;
  try {
    return p->env->SetGlobalVar(p->env->SaveString(name), *(const AVSValue*)(&val));
  }
  catch (const AvisynthError& err) {
    p->error = err.msg;
    return -1;
  }
}

extern "C"
AVS_VideoFrame * AVSC_CC avs_new_video_frame_a(AVS_ScriptEnvironment * p, const AVS_VideoInfo * vi, int align)
{
  p->error = 0;
  try {
    PVideoFrame f0 = p->env->NewVideoFrame(*(const VideoInfo*)vi, align);
    AVS_VideoFrame* f;
    new((PVideoFrame*)&f) PVideoFrame(f0);
    return f;
  }
  catch (const AvisynthError& err) {
    p->error = err.msg;
  }
  return 0;
}

// with frame properties, and alignment
// note: in general there is no need for alignment specificationm use avs_new_video_frame_p
extern "C"
AVS_VideoFrame * AVSC_CC avs_new_video_frame_p_a(AVS_ScriptEnvironment * p, const AVS_VideoInfo * vi, const AVS_VideoFrame * prop_src, int align)
{
  p->error = 0;
  try {
    PVideoFrame f0 = p->env->NewVideoFrameP(*(const VideoInfo*)vi, (const PVideoFrame*)&prop_src, align);
    AVS_VideoFrame* f;
    new((PVideoFrame*)&f) PVideoFrame(f0);
    return f;
  }
  catch (const AvisynthError& err) {
    p->error = err.msg;
  }
  return 0;
}

// with frame properties, no extra alignment requirement
extern "C"
AVS_VideoFrame * AVSC_CC avs_new_video_frame_p(AVS_ScriptEnvironment * p, const AVS_VideoInfo * vi, const AVS_VideoFrame * prop_src)
{
  p->error = 0;
  try {
    PVideoFrame f0 = p->env->NewVideoFrameP(*(const VideoInfo*)vi, (const PVideoFrame*)&prop_src, AVS_FRAME_ALIGN);
    AVS_VideoFrame* f;
    new((PVideoFrame*)&f) PVideoFrame(f0);
    return f;
  }
  catch (const AvisynthError& err) {
    p->error = err.msg;
  }
  return 0;
}


extern "C"
int AVSC_CC avs_make_writable(AVS_ScriptEnvironment * p, AVS_VideoFrame * *pvf)
{
  p->error = 0;
  try {
    return p->env->MakeWritable((PVideoFrame*)(pvf));
  }
  catch (const AvisynthError& err) {
    p->error = err.msg;
  }
  return -1;
}

// Since V9
extern "C"
int AVSC_CC avs_make_property_writable(AVS_ScriptEnvironment * p, AVS_VideoFrame * *pvf)
{
  p->error = 0;
  try {
    return p->env->MakePropertyWritable((PVideoFrame*)(pvf));
  }
  catch (const AvisynthError& err) {
    p->error = err.msg;
  }
  return -1;
}

extern "C"
void AVSC_CC avs_bit_blt(AVS_ScriptEnvironment * p, BYTE * dstp, int dst_pitch, const BYTE * srcp, int src_pitch, int row_size, int height)
{
  p->error = 0;
  try {
    p->env->BitBlt(dstp, dst_pitch, srcp, src_pitch, row_size, height);
  }
  catch (const AvisynthError& err) {
    p->error = err.msg;
  }
}

struct ShutdownFuncData
{
  AVS_ShutdownFunc func;
  void* user_data;
};

void __cdecl shutdown_func_bridge(void* user_data, IScriptEnvironment* env)
{
  ShutdownFuncData* d = (ShutdownFuncData*)user_data;
  AVS_ScriptEnvironment e;
  e.env = env;
  e.error = NULL;
  d->func(d->user_data, &e);
}

extern "C"
void AVSC_CC avs_at_exit(AVS_ScriptEnvironment * p,
  AVS_ShutdownFunc function, void* user_data)
{
  p->error = 0;
  ShutdownFuncData* dd, * d = new ShutdownFuncData;
  d->func = function;
  d->user_data = user_data;
  dd = (ShutdownFuncData*)p->env->SaveString((const char*)d, sizeof(ShutdownFuncData));
  delete d;
  p->env->AtExit(shutdown_func_bridge, dd);
}

extern "C"
int AVSC_CC avs_check_version(AVS_ScriptEnvironment * p, int version)
{
  p->error = 0;
  try {
    p->env->CheckVersion(version);
    return 0;
  }
  catch (const AvisynthError& err) {
    p->error = err.msg;
    return -1;
  }
}

extern "C"
AVS_VideoFrame * AVSC_CC avs_subframe(AVS_ScriptEnvironment * p, AVS_VideoFrame * src0,
  int rel_offset, int new_pitch, int new_row_size, int new_height)
{
  p->error = 0;
  try {
    PVideoFrame f0 = p->env->Subframe((VideoFrame*)src0, rel_offset, new_pitch, new_row_size, new_height);
    AVS_VideoFrame* f;
    new((PVideoFrame*)&f) PVideoFrame(f0);
    return f;
  }
  catch (const AvisynthError& err) {
    p->error = err.msg;
    return 0;
  }
}

extern "C"
AVS_VideoFrame * AVSC_CC avs_subframe_planar(AVS_ScriptEnvironment * p, AVS_VideoFrame * src0,
  int rel_offset, int new_pitch, int new_row_size, int new_height,
  int rel_offsetU, int rel_offsetV, int new_pitchUV)
{
  p->error = 0;
  try {
    PVideoFrame f0 = p->env->SubframePlanar((VideoFrame*)src0, rel_offset, new_pitch, new_row_size,
      new_height, rel_offsetU, rel_offsetV, new_pitchUV);
    AVS_VideoFrame* f;
    new((PVideoFrame*)&f) PVideoFrame(f0);
    return f;
  }
  catch (const AvisynthError& err) {
    p->error = err.msg;
    return 0;
  }
}

// Interface V8
extern "C"
AVS_VideoFrame * AVSC_CC avs_subframe_planar_a(AVS_ScriptEnvironment * p, AVS_VideoFrame * src0,
  int rel_offset, int new_pitch, int new_row_size, int new_height,
  int rel_offsetU, int rel_offsetV, int new_pitchUV, int rel_offsetA)
{
  p->error = 0;
  try {
    PVideoFrame f0 = p->env->SubframePlanarA((VideoFrame*)src0, rel_offset, new_pitch, new_row_size,
      new_height, rel_offsetU, rel_offsetV, new_pitchUV, rel_offsetA);
    AVS_VideoFrame* f;
    new((PVideoFrame*)&f) PVideoFrame(f0);
    return f;
  }
  catch (const AvisynthError& err) {
    p->error = err.msg;
    return 0;
  }
}

extern "C"
int AVSC_CC avs_set_memory_max(AVS_ScriptEnvironment * p, int mem)
{
  p->error = 0;
  try {
    return p->env->SetMemoryMax(mem);
  }
  catch (const AvisynthError& err) {
    p->error = err.msg;
    return -1;
  }
}

extern "C"
int AVSC_CC avs_set_working_dir(AVS_ScriptEnvironment * p, const char* newdir)
{
  p->error = 0;
  try {
    return p->env->SetWorkingDir(newdir);
  }
  catch (const AvisynthError& err) {
    p->error = err.msg;
    return -1;
  }
}

// V12
extern "C"
int AVSC_CC avs_acquire_global_lock(AVS_ScriptEnvironment* p, const char* name)
{
  p->error = 0;
  try {
    return p->env->AcquireGlobalLock(name) ? 1 : 0;
  }
  catch (const AvisynthError& err) {
    p->error = err.msg;
    return 0;
  }
}

// V12
extern "C"
void AVSC_CC avs_release_global_lock(AVS_ScriptEnvironment* p, const char* name)
{
  p->error = 0;
  try {
    p->env->ReleaseGlobalLock(name);
  }
  catch (const AvisynthError& err) {
    p->error = err.msg;
  }
}

// Interface V8. See AVS_AEP_xxx enums
extern "C"
size_t AVSC_CC avs_get_env_property(AVS_ScriptEnvironment * p, int avs_aep_prop)
{
  p->error = 0;
  try {
    return p->env->GetEnvProperty((AvsEnvProperty)avs_aep_prop);
  }
  catch (const AvisynthError& err) {
    p->error = err.msg;
    return 0;
  }
}

// Interface V8, buffer pool, Support functions
// see AVS_ALLOCTYPE_xxx enum
extern "C"
void * AVSC_CC avs_pool_allocate(AVS_ScriptEnvironment * p, size_t nBytes, size_t alignment, int avs_alloc_type)
{
  p->error = 0;
  try {
    return p->env->Allocate(nBytes, alignment, (AvsAllocType)avs_alloc_type);
  }
  catch (const AvisynthError& err) {
    p->error = err.msg;
    return 0;
  }
}

extern "C"
void AVSC_CC avs_pool_free(AVS_ScriptEnvironment * p, void* ptr)
{
  p->error = 0;
  try {
    return p->env->Free(ptr);
  }
  catch (const AvisynthError& err) {
    p->error = err.msg;
  }
}

// Returns TRUE (1) and the requested variable. If the method fails, returns 0 (FALSE) and does not touch 'val'.
// The returned AVS_Value *val value must be be released with avs_release_value only on success
// AVS_Value *val is not caller allocated
extern "C"
int AVSC_CC avs_get_var_try(AVS_ScriptEnvironment * p, const char* name, AVS_Value *val)
{
  p->error = 0;
  AVSValue v0;
  const bool success = p->env->GetVarTry(name, &v0);
  if (success) {
    new ((AVSValue*)val) AVSValue(v0);
    return 1;
  }
  return 0;
}

// Return the value of the requested variable.
// If the variable was not found or had the wrong type,
// return the supplied default value.
extern "C"
int AVSC_CC avs_get_var_bool(AVS_ScriptEnvironment * p, const char* name, int def)
{
  p->error = 0;
  return (int)(p->env->GetVarBool(name, (bool)def));
}

extern "C"
int AVSC_CC avs_get_var_int(AVS_ScriptEnvironment * p, const char* name, int def)
{
  p->error = 0;
  return p->env->GetVarInt(name, def);
}

extern "C"
double AVSC_CC avs_get_var_double(AVS_ScriptEnvironment * p, const char* name, double def)
{
  p->error = 0;
  return p->env->GetVarDouble(name, def);
}

extern "C"
const char * AVSC_CC avs_get_var_string(AVS_ScriptEnvironment * p, const char* name, const char* def)
{
  p->error = 0;
  return p->env->GetVarString(name, def);
}

extern "C"
int64_t AVSC_CC avs_get_var_long(AVS_ScriptEnvironment * p, const char* name, int64_t def)
{
  p->error = 0;
  return p->env->GetVarLong(name, def);
}

/////////////////////////////////////////////////////////////////////
//
//
//

// prototype from avisynth.cpp
IScriptEnvironment2* CreateScriptEnvironment2_internal(int version, bool fromAvs25, bool fromC);

extern "C"
AVS_ScriptEnvironment * AVSC_CC avs_create_script_environment(int version)
{
  AVS_ScriptEnvironment* e = new AVS_ScriptEnvironment;
  try {
    if (version < AVISYNTH_CLASSIC_INTERFACE_VERSION)
      version = AVISYNTH_CLASSIC_INTERFACE_VERSION; // always request a more modern ScriptEnvironment.
    e->env = CreateScriptEnvironment2_internal(version, false, true); // flag: from C
    e->error = NULL;
  }
  catch (const AvisynthError& err) {
    e->error = err.msg;
    e->env = 0;
  }
  return e;
}


/////////////////////////////////////////////////////////////////////
//
//
//

extern "C"
void AVSC_CC avs_delete_script_environment(AVS_ScriptEnvironment * e)
{
  if (e) {
    if (e->env) {
      try {
        e->env->DeleteScriptEnvironment();
      }
      catch (const AvisynthError& err) {
        (void)err;  // silence warning about unused variable; variable is kept for debugging
      }
      e->env = 0;
    }
    delete e;
  }
}
