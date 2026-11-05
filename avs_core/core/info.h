// Avisynth+
// https://avs-plus.net
//
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

#ifndef __INFO_H__
#define __INFO_H__

#ifdef AVS_LINUX
#include <uchar.h>
#endif
#include <sstream>
#include "internal.h"
#include <unordered_map>
#include <array>
#include <iomanip>
#include <vector>
#include <memory>
#include <cstring>
#include "strings.h"

typedef struct BBX {
  uint8_t width; // e.g. 8
  uint8_t height; // 16
  int8_t offset_x; // 0
  int8_t offset_y; // -4
} BBX;

class PreRendered {
  const bool useHalocolor;
  const int vi_width;
  const int vi_height;

public:
  int x, y;
  int len;
  int xstart;
  int text_width; // draw this amount of pixels starting from horizontal index xstart
  int ystart; // vertical visibility: starting row in stringbitmap
  int yend;   // vertical visibility: ending row in stringbitmap
  int stringbitmap_height; // font height plus optinally added top/bottom line

  std::vector<std::vector<uint8_t>> stringbitmap;
  std::vector<std::vector<uint8_t>> stringbitmap_outline;

  PreRendered(
    const uint8_t* fonts,
    const int fontline_bytes,
    const int _vi_width, const int _vi_height,
    int _x, int _y, // they may change
    std::vector<int>& s, // it may get shortened
    const std::vector<BBX>& bbx_array,
    const int align,
    const bool _useHalocolor,
    const int FONT_WIDTH_notused, const int FONT_HEIGHT
    );

  void make_outline();
  int frame_width() const { return vi_width; }
  int frame_height() const { return vi_height; }
};

class BitmapFont {

  int number_of_chars;
  std::string font_name;
  std::string font_filename;

public:
  const BBX global_bbx;

  const bool bold;

  std::vector<uint8_t> font_bitmaps;
  std::vector<BBX> bbx_array; // FontBoundingBox BBX array, can be individual for each character
  const int fontline_bytes;

  std::unordered_map<std::string, int> charReMapUtf8; // utf8 vs. font image index

  void SaveAsC(const uint32_t* _codepoints);

  BitmapFont(int _number_of_chars,
    const uint16_t* _src_font_bitmaps_internaluint16, 
    const uint8_t* _src_font_bitmaps, 
    const int _fontline_bytes, 
    const uint16_t* _codepoints_internaluint16, // unicode codepoints of an internal, predefined font
    const uint32_t* _codepoints, // unicode codepoints from external BDF, can be above U+FFFF
    const std::vector<BBX>* bbx_array_ptr,
    const BBX& _global_bbx,
    std::string _font_name, std::string _font_filename, bool _bold, bool debugSave) :
    number_of_chars(_number_of_chars),
    font_name(_font_name),
    font_filename(_font_filename),
    global_bbx(_global_bbx),
    bold(_bold),
    fontline_bytes(_fontline_bytes)
    //font_bitmaps(_font_bitmaps),
  {
    //fixme: do not copy data
    const int charline_count = global_bbx.height * number_of_chars;
    font_bitmaps.resize(charline_count * fontline_bytes);
    if (_src_font_bitmaps != nullptr) 
      std::memcpy(font_bitmaps.data(), _src_font_bitmaps, font_bitmaps.size());
    else {
      // this must be an internal, predefined array
      // copy uint16_t array to byte array MSB-LSB order 
      // fontline_bytes is 2
      const uint16_t* src = _src_font_bitmaps_internaluint16;
      uint8_t* dst = font_bitmaps.data();
      for (auto i = 0; i < charline_count; i++) {
        uint16_t one_fontline = src[i];
        dst[i * 2] = (uint8_t)(one_fontline >> 8);
        dst[i * 2 + 1] = (uint8_t)(one_fontline & 0xFF);
      }
    }

    // We always display utf8 strings, so the reverse lookup must be utf8 character based,
    // which returns the index to the font bitmap and bbx array.
    // Either the internal 16 bit or the external (BDF) 32 bit codepoint array is given, the other is nullptr.
    for (int i = 0; i < _number_of_chars; i++) {
      const uint32_t cp = _codepoints != nullptr ? _codepoints[i] : _codepoints_internaluint16[i];
      std::string s_utf8 = U32_to_utf8(cp);
      if (!s_utf8.empty()) // not for an invalid codepoint, e.g. BDF ENCODING -1 (unencoded glyph)
        charReMapUtf8[s_utf8] = i;
    }


    // Many East Asian characters—especially CJK (Chinese, Japanese, Korean) ideographs
    // are typically double-width compared to Latin characters when used in monospaced bitmap
    // fonts like BDF.
    // For Avisynth-embedded internal fonts (e.g., Terminus, Info_H), only a fixed global BBX is used,
    // as these fonts do not include CJK characters, only Latin, Cyrillic, and Greek.
    // The variable BBX array is optional and is intended for use with external fonts,
    // such as unifont-16.0.04.bdf, where CJK characters have different BBX values
    // compared to Latin characters.

    if (bbx_array_ptr != nullptr) {
      // copy BBX array if provided
      bbx_array.insert(bbx_array.end(), bbx_array_ptr->begin(), bbx_array_ptr->end());
    }
    else {
      // When no BBX array was provided (bbx_array_ptr is nullptr)
      // fill bbx_array with constant default global values
      bbx_array.assign(_number_of_chars, global_bbx);
    }

    if (debugSave && _codepoints != nullptr) // only for external BDF
      SaveAsC(_codepoints);
  }

  // true when loaded from an external BDF file (built-in fonts have no file name)
  bool IsFromFile() const { return !font_filename.empty(); }

  // helper function for remapping an utf8 string to font index entry list
  std::vector<int> remap(const std::string& s_utf8) const;
};

// creates a new font instance (uncached)
std::unique_ptr<BitmapFont> CreateBitmapFont(int size, const char* name, bool bold, bool debugSave);
// Returns a shared, immutable font from the per-ScriptEnvironment font cache, created on first use.
// Uncached (new instance) when env is nullptr or for debugSave.
std::shared_ptr<const BitmapFont> GetBitmapFont(int size, const char* name, bool bold, bool debugSave, IScriptEnvironment* env);

void SimpleTextOutW(const BitmapFont* current_font, const VideoInfo& vi, PVideoFrame& frame, int real_x, int real_y, std::string& text_utf8,
  bool fadeBackground, int textcolor, int halocolor, bool useHaloColor, int align, int chromaplacement);
void SimpleTextOutW_multi(const BitmapFont* current_font, const VideoInfo& vi, PVideoFrame& frame, int real_x, int real_y, std::string& text_utf8,
  bool fadeBackground, int textcolor, int halocolor, bool useHaloColor, int align, int lsp, int chromaplacement);

// legacy function w/o outline, originally with ASCII input, background fading
void DrawStringPlanar(VideoInfo& vi, PVideoFrame& dst, int x, int y, const char* s, IScriptEnvironment* env);
void DrawStringYUY2(VideoInfo& vi, PVideoFrame& dst, int x, int y, const char* s, IScriptEnvironment* env);
void DrawStringRGB32(VideoInfo& vi, PVideoFrame& dst, int x, int y, const char* s, IScriptEnvironment* env);
void DrawStringRGB24(VideoInfo& vi, PVideoFrame& dst, int x, int y, const char* s, IScriptEnvironment* env);

#endif  // __INFO_H__
