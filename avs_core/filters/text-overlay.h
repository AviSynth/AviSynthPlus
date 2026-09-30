// Avisynth v2.5.  Copyright 2002 Ben Rudiak-Gould et al.
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

#ifndef __Text_overlay_H__
#define __Text_overlay_H__

#include <avisynth.h>

#ifdef AVS_WINDOWS
    #include <avs/win.h>
#else
    #include <avs/posix.h>
#endif

#include <cstdio>
#include <stdint.h>
#include <string>
#include <vector>
#include "../core/info.h"


/********************************************************************
********************************************************************/


#if defined(AVS_WINDOWS) && !defined(NO_WIN_GDI)
#include "overlay/blend_common.h"  // MaskMode, PLACEMENT_*
#include <vector>
#include "getalpharect_impl.h"

/*
 * Antialiaser — GDI-based anti-aliased text renderer for AviSynth+ video frames.
 *
 * OVERVIEW
 * --------
 * The caller draws text into an internal 8x-supersampled 1-bit GDI DIB via GetDC().
 * Apply() then composites that text onto the video frame by:
 *   1. GetAlphaRect() — scans the DIB, gamma-corrects, and computes per-pixel
 *      blend weights (basealpha) and pre-scaled color addends (Y/R, U/G, V/B),
 *      all as uint16_t values stored in soa_buf.
 *   2. ApplyPlanar_SoA / ApplyYUY2 / ApplyRGB_packed — reads from soa_buf and
 *      blends those values onto the actual video pixels.
 *
 * FORMAT-AGNOSTIC MASK BUFFER
 * ---------------------------
 * soa_buf is always uint16_t, regardless of the video format (8-bit, 10–16-bit
 * integer, or 32-bit float).  GetAlphaRect() produces:
 *   basealpha : 0..256  (256 = fully transparent — pixel untouched)
 *   Y/R, U/G, V/B addends : 0..~65530
 * These are fixed-point values scaled to 8-bit headroom.  The Apply functions
 * rescale on the fly: integer formats left-shift the addend by (bpp-8), float
 * formats divide by 65536.  The mask computation itself never needs to know the
 * output bit depth.
 *
 * CHROMA PLACEMENT SUPPORT
 * ------------------------
 * For subsampled YUV formats the U/G and V/B planes in soa_buf are stored at
 * full luma resolution.  When Apply() composites a subsampled clip, each UV
 * output pixel is derived by spatially downsampling the corresponding luma-
 * resolution mask row(s) using prepare_effective_mask_for_row<MaskMode,...>
 * (from overlay/blend_common.h).  The MaskMode encodes both the subsampling
 * ratio (4:4:4, 4:2:2, 4:2:0, 4:1:1, 4:4:0, 4:1:0) and the chroma siting (CENTER (MPEG1) / LEFT (MPEG2) /
 * TOP_LEFT), giving correct chroma placement rather than a naive box average.
 *
 * rowprep_fns[MASK_MODE_COUNT] holds one function pointer per MaskMode, sized
 * by the MaskMode enum's own sentinel count (not a literal, so it can't silently
 * fall out of sync with the enum again), with the best available SIMD tier (AVX2 /
 * SSE4.1 / scalar) selected once at construction time from cpuFlags.  Apply()
 * computes the actual MaskMode at call time from the VideoInfo subsampling
 * factors and the stored chromaplacement member, so the same Antialiaser
 * instance can be reused if the clip format ever changes.
 *
 * BUFFER LAYOUT  (row-interleaved SoA)
 * -------------------------------------
 * The original implementation stored the four mask components interleaved per
 * pixel (AoS: [ba,bv,gu,ry] packed as four consecutive uint16_t).  That layout
 * is ideal for GetAlphaRect() writes (one sequential stream) but stride-4 for
 * the Apply readers.
 *
 * The current layout is row-interleaved SoA: within each scanline the four
 * sub-planes are stored consecutively, each w_stride elements wide:
 *
 *   row y: [ ba_0..ba_n | ry_0..ry_n | u_0..u_n | v_0..v_n ]
 *             plane 0        plane 1    plane 2    plane 3
 *
 *   row y, plane p: soa_buf + y * 4 * w_stride + p * w_stride
 *
 * w_stride = (w + 31) & ~31, rounding up to a multiple of 32 so every sub-row
 * starts on a 64-byte cache-line boundary.
 *
 * This is a compromise between the two extremes:
 *   - Full AoS (old): GetAlphaRect() writes one stream (optimal), Apply reads
 *     with stride 4 (poor locality per plane).
 *   - Full SoA (four separate plane allocations): Apply reads are stride-1
 *     (optimal per plane), but GetAlphaRect() writes to four streams that are
 *     w*h*2 bytes apart (~600 KB for 640×480), causing heavy cache thrashing.
 *   - Row-interleaved SoA (current): GetAlphaRect() writes four streams that
 *     are at most w_stride*2 bytes apart (<=1280 bytes for 640-wide), all within
 *     one L1-cache-sized window per row.  Apply reads each plane stride-1 within
 *     a row and steps by 4*w_stride between rows — well within prefetcher range.
 *
 * Overall throughput is broadly on par with the original AoS method while
 * enabling correct chroma-placement-aware UV compositing.
 */
class Antialiaser
{
public:
  Antialiaser(int width, int height, const char fontname[], int size,
    int textcolor, int halocolor, bool _bold, bool _italic, bool _noaa,
    int64_t cpuFlags,
    int chromaplacement,
    int font_width=0, int font_angle=0, bool _interlaced=false);
  virtual ~Antialiaser();
  HDC GetDC();
  void FreeDC();

  void Apply(const VideoInfo& vi, PVideoFrame* frame, int pitch);

private:
  template<MaskMode maskMode, int bits_per_pixel>
  void ApplyPlanar_SoA(BYTE* buf, int pitch, int pitchUV, BYTE* bufU, BYTE* bufV, bool isRGB);
  void ApplyYUY2(BYTE* buf, int pitch, MaskMode maskMode); // maskMode: one of the MASK422 variants

  template<typename pixel_t, bool has_alpha>
  void ApplyRGB_packed(BYTE* buf, int pitch);

  using rowprep_u16_fn_t = const uint16_t*(*)(const uint16_t*, int, int, std::vector<uint16_t>&, int, int, MagicDiv);

  void* lpAntialiasBits;
  // Row-interleaved SoA: each scanline holds [ba|ry|u|v], each sub-row w_stride wide.
  // Row y, plane p: soa_buf + y * 4 * w_stride + p * w_stride
  // w_stride = (w + 31) & ~31  — rounded up so each sub-row is 64-byte aligned.
  uint16_t* soa_buf;   // single allocation: w_stride * h * 4 uint16_t
  int w_stride;        // padded row stride (>= w, multiple of 32)
  std::vector<uint16_t> uv_buf_ba, uv_buf_u, uv_buf_v; // scratch for ApplyPlanar_SoA / ApplyYUY2 UV section, sized w
  rowprep_u16_fn_t rowprep_fns[MASK_MODE_COUNT]; // one per MaskMode, SIMD variant selected at construction
  int chromaplacement; // ChromaLocation_e, mapped to the mask kernel in Apply()
  HDC hdcAntialias;
  HBITMAP hbmAntialias;
  HFONT hfontDefault;
  HBITMAP hbmDefault;
  const int w, h;
  const int textcolor, halocolor;
  int xl, yt, xr, yb; // sub-rectangle containing live text
  bool dirty, interlaced;
  bool bold, italic;
  bool noaa;

#ifdef INTEL_INTRINSICS
  getalpharect_fn_t getalpharect_fn;
#endif
  void GetAlphaRect();
};
#endif


class ShowFrameNumber : public GenericVideoFilter
/**
  * Class to display frame number on a video clip
 **/
{
public:
  ShowFrameNumber(PClip _child, bool _scroll, int _offset, int _x, int _y, const char _fontname[], int _size,
      int _textcolor, int _halocolor, int font_width, int font_angle, bool _bold, bool _italic, bool _noaa, bool _gdi, IScriptEnvironment* env);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;

  static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env);

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    AVS_UNUSED(frame_range);
    return cachehints == CACHE_GET_MTMODE ? MT_MULTI_INSTANCE : 0;
      // Antialiaser usage -> MT_MULTI_INSTANCE (with NICE_FILTER rect area conflicts)
  }

private:
  const bool use_gdi;
#if defined(AVS_WINDOWS) && !defined(NO_WIN_GDI)
  std::unique_ptr<Antialiaser> antialiaser;
#endif
  std::unique_ptr<BitmapFont> current_font;
  int chromaplacement;
  const bool scroll;
  const int offset;
  const int size, x, y;
  const int textcolor, halocolor;
  [[maybe_unused]] const bool bold;
  [[maybe_unused]] const bool italic; // n/a in NO_WIN_GDI
  [[maybe_unused]] const bool noaa; // n/a in NO_WIN_GDI
};

class ShowCRC32 : public GenericVideoFilter
  /**
    * Class to display CRC32 checksum of selected planes on a video clip
   **/
{
  uint32_t crc32_table[256];

  void build_crc32_table(void);
  std::string compute_crc_text(PVideoFrame& crc_frame, std::vector<uint32_t>& out_values) const;

public:
  ShowCRC32(PClip _child, PClip _crc_child, bool _scroll, int _offset, int _x, int _y,
            const char _fontname[], int _size,
            int _textcolor, int _halocolor, int font_width, int font_angle,
            bool _bold, bool _italic, bool _noaa, bool _gdi,
            const char* channels, int _mode, bool _compatible_mode, int _showmode, IScriptEnvironment* env);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;

  static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env);

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    AVS_UNUSED(frame_range);
    return cachehints == CACHE_GET_MTMODE ? MT_MULTI_INSTANCE : 0;
  }

private:
  PClip crc_child;
  bool doY, doU, doV, doR, doG, doB, doA;
  const int mode;
  bool compatible_mode;
  const int showmode;
  const bool use_gdi;
#if defined(AVS_WINDOWS) && !defined(NO_WIN_GDI)
  std::unique_ptr<Antialiaser> antialiaser;
#endif
  std::unique_ptr<BitmapFont> current_font;
  int chromaplacement;
  const bool scroll;
  const int offset;
  const int size, x, y;
  const int textcolor, halocolor;
  [[maybe_unused]] const bool bold;
  [[maybe_unused]] const bool italic; // n/a in NO_WIN_GDI
  [[maybe_unused]] const bool noaa;   // n/a in NO_WIN_GDI
};



class ShowSMPTE : public GenericVideoFilter
/**
  * Class to display SMPTE codes on a video clip
 **/
{
public:
  ShowSMPTE(PClip _child, double _rate, const char* _offset, int _offset_f, int _x, int _y, const char _fontname[], int _size,
            int _textcolor, int _halocolor, int font_width, int font_angle, bool _bold, bool _italic, bool _noaa, bool _gdi, IScriptEnvironment* env);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;

  static AVSValue __cdecl CreateSMTPE(AVSValue args, void*, IScriptEnvironment* env);
  static AVSValue __cdecl CreateTime(AVSValue args, void*, IScriptEnvironment* env);

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    AVS_UNUSED(frame_range);
    return cachehints == CACHE_GET_MTMODE ? MT_MULTI_INSTANCE : 0;
      // Antialiaser usage -> MT_MULTI_INSTANCE (with NICE_FILTER rect area conflicts)
  }

private:
  const bool use_gdi;
#if defined(AVS_WINDOWS) && !defined(NO_WIN_GDI)
  std::unique_ptr<Antialiaser> antialiaser;
#endif
  std::unique_ptr<BitmapFont> current_font;
  int chromaplacement;
  int rate;
  int offset_f;
  const int x, y;
  bool dropframe;
  const int textcolor, halocolor;
  [[maybe_unused]] const bool bold;
  [[maybe_unused]] const bool italic; // n/a in NO_WIN_GDI
  [[maybe_unused]] const bool noaa; // n/a in NO_WIN_GDI
};



#if defined(AVS_WINDOWS) && !defined(NO_WIN_GDI)
class Subtitle : public GenericVideoFilter
/**
  * Subtitle creation class
 **/
{
public:
  Subtitle( PClip _child, const char _text[], int _x, int _y, int _firstframe, int _lastframe,
            const char _fontname[], int _size, int _textcolor, int _halocolor, int _align,
            int _spc, bool _multiline, int _lsp, int _font_width, int _font_angle, bool _interlaced, const char _font_filename[], const bool _utf8,
            const bool _bold, const bool _italic, const bool _noaa, int _chromaplacement, IScriptEnvironment* env);
  virtual ~Subtitle(void);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;

  static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env);

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    AVS_UNUSED(frame_range);
    return cachehints == CACHE_GET_MTMODE ? MT_MULTI_INSTANCE : 0;
      // Antialiaser usage -> MT_MULTI_INSTANCE (with NICE_FILTER rect area conflicts)
  }

private:
  void InitAntialiaser(IScriptEnvironment* env);

  const int x, y, firstframe, lastframe, size, lsp, font_width, font_angle;
  const bool multiline, interlaced;
  const int textcolor, halocolor, align, spc;
  const char* const fontname;
  const char* const text;
  const char* const font_filename;
  const bool utf8;
  const bool bold;
  const bool italic;
  const bool noaa;
  const int chromaplacement;
  Antialiaser* antialiaser;
};
#endif

class SimpleText : public GenericVideoFilter
  /**
    * SimpleText creation class
   **/
{
public:
  SimpleText(PClip _child, const char _text[], int _x, int _y, int _firstframe, int _lastframe,
    const char _fontname[], int _size, int _textcolor, int _halocolor, int _align,
    int _spc, bool _multiline, int _lsp, int _font_width, int _font_angle, bool _interlaced, const char _font_filename[],
    const bool _utf8, const bool _bold, const int _chromalocation,
    IScriptEnvironment* env);
  virtual ~SimpleText(void);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;

  static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env);

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    AVS_UNUSED(frame_range);
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

private:
  const int x, y, firstframe, lastframe;
  const int size;
  const int lsp;
  // const int font_width, font_angle; // n/a
  const bool multiline;
  // const bool interlaced; // n/a
  const int textcolor, halocolor, align;
  // const int spc; // n/a
  const int halocolor_orig;
  const char* const fontname; // Terminus or info_h
  const char* const text;
  const char* const font_filename; // .BDF files
  const bool utf8;
  const bool bold;
  const int chromalocation;
  std::unique_ptr<BitmapFont> current_font;
};

class FilterInfo : public GenericVideoFilter
/**
  * FilterInfo creation class
 **/
{
public:
  FilterInfo( PClip _child, const char _fontname[], int _size, int _textcolor, int _halocolor, bool _bold, bool _italic, bool _noaa, bool _cpu, int _x, int _y, int _align, bool _gdi, IScriptEnvironment* env);
  virtual ~FilterInfo(void);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;
  bool __stdcall GetParity(int n) override;

  static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env);

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    AVS_UNUSED(frame_range);
    return cachehints == CACHE_GET_MTMODE ? MT_MULTI_INSTANCE : 0;
      // Antialiaser usage -> MT_MULTI_INSTANCE (with NICE_FILTER rect area conflicts)
  }

private:
  const VideoInfo& AdjustVi();

  const VideoInfo &vii;

  const int size;

  const int text_color, halo_color;
  const bool use_gdi;
#if defined(AVS_WINDOWS) && !defined(NO_WIN_GDI)
  std::unique_ptr<Antialiaser> antialiaser;
#endif
  std::unique_ptr<BitmapFont> current_font;
  int chromaplacement;
  [[maybe_unused]] const bool bold;
  [[maybe_unused]] const bool italic;
  [[maybe_unused]] const bool noaa;
  const bool cpu;
  const int x, y;
  const int align;
};


class Compare : public GenericVideoFilter
/**
  * Compare two clips frame by frame and display fidelity measurements (with optionnal logging to file)
 **/
{
public:
  Compare(PClip _child1, PClip _child2, const char* channels, const char *fname, bool _show_graph, bool _gdi, IScriptEnvironment* env);
  ~Compare();
  static AVSValue __cdecl Create(AVSValue args, void* , IScriptEnvironment* env);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    AVS_UNUSED(frame_range);
    return cachehints == CACHE_GET_MTMODE ? MT_SERIALIZED : 0;
      // Antialiaser usage -> MT_MULTI_INSTANCE (with NICE_FILTER rect area conflicts)
      // show_graph gathers data of last n frames inside class -> conditional MT_SERIALIZED
      // logfile writing: if log -> conditional MT_SERIALIZED
      // display of global counters -> MT_SERIALIZED
      // So least common multiple -> MT_SERIALIZED
  }


private:
  bool use_gdi;
#if defined(AVS_WINDOWS) && !defined(NO_WIN_GDI)
  std::unique_ptr<Antialiaser> antialiaser;
#endif
  std::unique_ptr<BitmapFont> current_font;
  int chromaplacement;
  PClip child2;
  uint32_t mask;
  uint64_t mask64;
  int masked_bytes;
  FILE* log;
  int* psnrs;
  bool show_graph;
  double PSNR_min, PSNR_tot, PSNR_max;
  double MAD_min, MAD_tot, MAD_max;
  double MD_min, MD_tot, MD_max;
  double bytecount_overall, SSD_overall;
  int framecount;
  int planar_plane;
  int pixelsize;
  int bits_per_pixel;
  const int text_color, halo_color;

};



/**** Helper functions ****/

void ApplyMessage( PVideoFrame* frame, const VideoInfo& vi, const char* message, int size,
                   int textcolor, int halocolor, int bgcolor, IScriptEnvironment* env );

bool GetTextBoundingBox( const char* text, const char* fontname, int size, bool bold,
                         bool italic, int align, int* width, int* height );

bool GetTextBoundingBoxFixed(const char* text, const char* fontname, int size, bool bold,
  bool italic, int align, int& width, int& height, bool utf8);

#endif  // __Text_overlay_H__
