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

// pinterf:
// high bit depth, planar RGB
// utf8 option, internally unicode
// info_h font definition reorganized, Latin-1 Supplement 00A0-00FF
// Configurable color
// Configurable halocolor (text outline)
// Configurable background fading
// Alignment
// multiline
// multiple size, multiple fonts, "Terminus", "info_h"
// chroma location, overlay-like weighted chroma handling

#include "info.h"
#include "InternalEnvironment.h"
#include <cstring>
#include <sstream>
#include <fstream>
#include <unordered_map>
#include <array>
#include <algorithm>
#include <iomanip>
#include <avs/filesystem.h>

#include <locale>
#include <cstdio>
#include <cassert>
#include <type_traits>
#include "fonts/fixedfonts.h"
#include "strings.h"
#include "../convert/convert_helper.h"
#include "../filters/overlay/blend_common.h" // chromaLocationToMaskPlacement

// helper function for remapping an utf8 string to font index entry list
// Invalid UTF-8 gives one empty character (index 0), then decoding goes on, so that a following valid
// character is not lost (like the Unicode/WHATWG replacement recommendation): one for each stray
// continuation byte or 0xF8..0xFF byte, and one for an incomplete sequence (a lead byte with fewer
// continuation bytes than needed, also at the end of the string: no reading past it).
std::vector<int> BitmapFont::remap(const std::string& s_utf8) const
{
  // new vector with characters remapped to font table indexes
  std::vector<int> s_remapped;
  s_remapped.reserve(str_utf8_size(s_utf8));

  const char* p = s_utf8.data();
  const char* end = p + s_utf8.size();
  while (p < end) {
    // Get the lead byte of the current UTF-8 character
    const unsigned char lb = static_cast<unsigned char>(*p);
    // Determine the number of bytes in the current UTF-8 character
    int n = 0;
    if ((lb & 0x80) == 0) n = 1; // 0xxxxxxx
    else if ((lb & 0xE0) == 0xC0) n = 2; // 110xxxxx
    else if ((lb & 0xF0) == 0xE0) n = 3; // 1110xxxx
    else if ((lb & 0xF8) == 0xF0) n = 4; // 11110xxx
    // else: invalid lead byte (continuation byte 10xxxxxx or 0xF8..0xFF), n = 0
    if (n == 0) {
      s_remapped.push_back(0); // empty neutral character (space) for the bad byte
      ++p;
      continue;
    }
    // the whole sequence must be within the string, with continuation bytes 10xxxxxx
    int k = 1;
    while (k < n && p + k < end && (static_cast<unsigned char>(p[k]) & 0xC0) == 0x80)
      k++;
    if (k < n) {
      // incomplete sequence: one empty character for it (the lead byte and its continuation bytes)
      s_remapped.push_back(0);
      p += k;
      continue;
    }
    // finds by utf8 character sequence in font index table
    auto it = charReMapUtf8.find(std::string(p, n));
    s_remapped.push_back(it != charReMapUtf8.end() ? it->second : 0); // 0: empty neutral character (space)
    // Advance the pointer by the number of bytes
    p += n;
  }
  return s_remapped;
}

// Internal function! For creating source code from a previously LoadBDF'd font file
// see fixedfonts.cpp
// Developer utility to generate the embeddable C++ source of the built-in fonts from the Terminus
// BDF font family (ter-u*.bdf). Triggered by the Text filter (class SimpleText) with an external font file
// and a negative size, e.g. Text("x", font_filename="ter-u16n.bdf", size=-1); output: <font_filename>.cpp_sample
// in the current directory, to be pasted into fixedfonts.cpp.
// The built-in fonts are BMP-only: the generated codepoint array is uint16_t, codepoints above U+FFFF
// would be truncated. Terminus has no such characters.
void BitmapFont::SaveAsC(const uint32_t* _codepoints)
{
  if (font_filename == "") return; // no GUS no sound :)

  std::string fontname;
  if (font_filename.substr(0, 4) == "ter-")
    font_name = "Terminus";
  else
    font_name = font_filename;

  std::ostringstream ss;
  ss << "namespace fixed_font_# {" << std::endl;
  ss << "// -- start of autogenerated text ---" << std::endl;
  ss << "// definition section for font: " << font_filename << std::endl;
  ss << "constexpr int CHARCOUNT = " << std::to_string(number_of_chars) << ";" << std::endl;
  ss << "constexpr int WIDTH = " << std::to_string(global_bbx.width) << ";" << std::endl;
  ss << "constexpr int HEIGHT = " << std::to_string(global_bbx.height) << ";" << std::endl;
  ss << "constexpr int OFFSET_X = " << std::to_string(global_bbx.offset_x) << ";" << std::endl;
  ss << "constexpr int OFFSET_Y = " << std::to_string(global_bbx.offset_y) << ";" << std::endl;
  ss << "constexpr FixedFont_info_t fixedfont_info = {" << std::endl;
  ss << "  \"" << font_filename << "\", // font name" << std::endl;
  ss << "  \"" << font_filename << "\", // font name internal" << std::endl;
  ss << "  CHARCOUNT, // num of chars" << std::endl;
  ss << "  WIDTH, HEIGHT, OFFSET_X, OFFSET_Y," << std::endl;
  ss << "  " << (bold ? "true" : "false") << " // bold" << std::endl;
  ss << "};" << std::endl;
  ss << "// font bitmap definitions" << std::endl;
  ss << "constexpr std::array<uint16_t, CHARCOUNT * HEIGHT> fixedfont_bitmap = {" << std::endl;
  for (int charcount = 0; charcount < number_of_chars; charcount++)
  {
    constexpr int LINES_BY_N = 16; // maximum number of constants per line
    for (int y = 0; y < global_bbx.height; y++) {
      ss << "0x";
      for (int x = 0; x < fontline_bytes; x++) {
        uint8_t charline = font_bitmaps[(charcount * global_bbx.height + y) * fontline_bytes + x];
        ss << std::setfill('0') << std::setw(2) << std::hex << charline;
      }
      const bool last = charcount == number_of_chars - 1 && y == global_bbx.height - 1;
      if (!last) ss << ",";
      if (y == global_bbx.height - 1)
        ss << " // u" << std::setw(4) << std::hex << _codepoints[charcount];
      if (y % LINES_BY_N == LINES_BY_N - 1) ss << std::endl; // last line of character- forced new line
    }
    if (global_bbx.height % LINES_BY_N != 0)
      ss << std::endl;
  };
  // example output:
  // 0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,
  // 0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000, // u0020
  // 0x0000,0x0000,0x0000,0x0000,0x0300,0x0300,0x0300,0x0300,0x0300,0x0300,0x0300,0x0300,0x0300,0x0300,0x0300,0x0300,
  // 0x0000,0x0000,0x0300,0x0300,0x0300,0x0300,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000, // u0021

  ss << "};" << std::endl;
  ss << "// codepoints array" << std::endl;
  ss << "constexpr std::array<uint16_t, CHARCOUNT> fixedfont_codepoints = {" << std::endl;
  for (int charcount = 0; charcount < number_of_chars; charcount++)
  {
    constexpr int LINES_BY_N = 16;
    int val = _codepoints[charcount];
    ss << "0x" << std::setfill('0') << std::setw(4) << std::hex << val;
    const bool last = charcount == number_of_chars - 1;
    if (!last) ss << ",";
    if (charcount % LINES_BY_N == LINES_BY_N - 1) ss << std::endl;
  }
  ss << "};" << std::endl;

  ss << "} // namespace" << std::endl;
  ss << "// -- end of autogenerated text ---" << std::endl;
  //
  std::ofstream outFile(font_filename + ".cpp_sample");
  outFile << ss.str();
  outFile.close();

  /* usage:
  // Predefined fonts main table
  constexpr int PREDEFINED_FONT_COUNT = 2;
  static const uint16_t* font_bitmaps[PREDEFINED_FONT_COUNT] =
  {
    &fixed_font_1::fixedfont_bitmap[0],
    &fixed_font_2::fixedfont_bitmap[0]
    // ...
  };
  static const uint16_t* font_codepoints[PREDEFINED_FONT_COUNT] =
  {
    &fixed_font_1::fixedfont_codepoints[0],
    &fixed_font_2::fixedfont_codepoints[0]
    // ...
  };
  static const FixedFont_info_t* font_infos[PREDEFINED_FONT_COUNT] =
  {
    &fixed_font_1::fixedfont_info,
    &fixed_font_2::fixedfont_info
    // ...
  };
  */
}

typedef struct CharInfo { // STARTCHAR charname
  std::string friendly_name;
  int encoding; // unicode codepoint, can be above U+FFFF; -1: unencoded glyph
  int dwidth_x, dwidth_y;
  // DWIDTH not supported, same as main PIXELSIZE
  // font properties can be overridden.
  BBX bbx;
  int empty_lines_bottom;
  int empty_lines_top;
  int bits_to_shift;
} CharDef;

typedef struct FontProperties {
  std::string Copyright;
  std::string Notice;
  std::string Family_name;
  std::string Weight_name;
  int pixel_size;
  int font_ascent; // 12
  int font_descent; // 4
  uint16_t default_char; // 65533
} FontProperties;

typedef struct FontInfo {
  std::string font; // n/a
  int size_points, size_dpi_x, size_dpi_y; // n/a
  // these may be overridden by individual characters.
  BBX bbx; // FONTBOUNDINGBOX e.g. 8 16 0 -4
  int chars; // number of characters
  int fontline_bytes; // length of a single font line in the byte array
} FontInfo;

class BdfFont {
public:
  std::string font_filename;
  FontInfo font_info;
  FontProperties font_properties;
  std::vector<uint32_t> codepoints_array; // -1 (unencoded) becomes 0xFFFFFFFF, not a valid codepoint
  std::vector<std::string> charnames_array;
  // each character can have different bounding box dimensions from global one
  std::vector<BBX> bbx_array;
  // one character line is not limited to 16/32 pixels anymore:
  std::vector<uint8_t> font_bitmaps;
};

std::string UnQuote(std::string s) {
  if (s.size() >= 2 && s.substr(0, 1) == "\"" && (s.substr(s.size() - 1, 1) == "\""))
    return s.substr(1,s.size()-2); // zero based
  return s;
}

static void vector_shl(uint8_t* buf, const size_t size, const size_t bits) {
  const size_t whole_chunks = bits / 8;

  // more bits that size
  if (whole_chunks >= size) {
    memset(buf, 0, size);
    return;
  }

  if (whole_chunks) {
    memmove(buf, buf + whole_chunks, size - whole_chunks);
    memset(buf + size - whole_chunks, 0, whole_chunks);
  }

  const uint8_t final_shift = bits % 8;

  if (final_shift) {
    const auto right = 8 - final_shift;
    const size_t len = size - whole_chunks - 1;
    for (size_t i = 0; i < len; i++) {
      buf[i] = (buf[i] << final_shift) | (buf[i + 1] >> right);
    }
    buf[len] = buf[len] << final_shift;
  }
}

static void vector_shr(uint8_t* buf, const size_t size, const size_t bits) {
  const size_t whole_chunks = bits / 8;

  // more bits that size
  if (whole_chunks >= size) {
    memset(buf, 0, size);
    return;
  }

  if (whole_chunks) {
    memmove(buf + whole_chunks, buf, size - whole_chunks);
    memset(buf, 0, whole_chunks);
  }

  const uint8_t final_shift = bits % 8;

  if (final_shift) {
    const auto left = 8 - final_shift;
    const size_t len = size - whole_chunks - 1;
    auto index = size - 1;
    for (size_t i = 0; i < len; i++) {
      buf[index] = (buf[index] >> final_shift) | (buf[index - 1] << left);
      index--;
    }
    buf[index] = buf[index] >> final_shift;
  }
}

static BdfFont LoadBMF(std::string name, bool bold) {

  // https://en.wikipedia.org/wiki/Glyph_Bitmap_Distribution_Format
  // also for make cpp source code, see BdfFont methods

  enum loadstate { ls_Init, ls_StartFont, ls_StartProperties, ls_Char };

  BdfFont fnt;
  CharInfo current_char;

  std::string temp;
  //std::ifstream ss("c:\\Download\\terminus-font-4.48\\terminus-font-4.48\\ter-u16n.bdf");
  std::ifstream ss;
  // explicite font file name
  auto fname = fs::path(name);

  if (!fs::exists(name))
    return fnt;

  fnt.font_filename = fname.filename().generic_string();
  ss.open(name);

  loadstate state = ls_Init;

  size_t line_counter = 0;
  int char_counter = 0;

  // make list governed by LF separator
  while (std::getline(ss, temp, '\n')) {
    std::string token;
    std::istringstream ssline(temp);

    std::getline(ssline, token, ' ');

    if (token.size() == 0) continue;

    switch (state) {
    case ls_Init:
      if (token == "STARTFONT")
        state = ls_StartFont;
      else {
        // unexpected token
      }
      break;
    case ls_StartFont:
      if (token == "ENDFONT") {
        state = ls_Init;
      }
      else if (token == "FONT") {
        // FONT-xos4-Terminus-Bold-R-Normal--16-160-72-72-C-80-ISO10646-1
        std::getline(ssline, token, ' ');
        fnt.font_info.font = UnQuote(token);
      }
      else if (token == "SIZE") {
        // SIZE 16 72 72
        // size in points, X and Y-axis resolution
        std::getline(ssline, token, ' ');
        fnt.font_info.size_points = std::stoi(token);
        std::getline(ssline, token, ' ');
        fnt.font_info.size_dpi_x = std::stoi(token);
        std::getline(ssline, token, ' ');
        fnt.font_info.size_dpi_y = std::stoi(token);
      }
      else if (token == "FONTBOUNDINGBOX") {
        // FONTBOUNDINGBOX 8 16 0 -4
        // bounding box of 8 pixels wide and 16 pixels high
        // lower left hand corner starting at x=0, y=-4. Note that although the bounding box is defined to be a 8x16 cell,
        // this can be overridden for individual glyphs.

        // e.g: Terminus: FONTBOUNDINGBOX 8 16 0 -4   Terminus
        // later for "A": BBX 8 16 0 -4
        //      Unifont:  FONTBOUNDINGBOX 16 16 0 -2  Unifont
        // later for "A": BBX 8 16 0 -2  should override the global bounding box

        std::getline(ssline, token, ' ');
        fnt.font_info.bbx.width = std::stoi(token);
        std::getline(ssline, token, ' ');
        fnt.font_info.bbx.height = std::stoi(token);
        std::getline(ssline, token, ' ');
        fnt.font_info.bbx.offset_x = std::stoi(token);
        std::getline(ssline, token, ' ');
        fnt.font_info.bbx.offset_y = std::stoi(token);

        // a single font line can occupy variable amount of bytes depending on font width
        // Note: in this implementation these global font dimensions are _the_
        // dimensions of the fixed size font
        // Note, that this is the maximum theoretical width; some BDF fonts (unifont) contain variable width characters
        // contrary to that it contains monospaced "C" type characters. In this case latin characters are 8 pixels wide,
        // and use DWIDTH and BBOX as 8, less than FONTBOUNDINGBOX value of 16.
        // but CJK characters are 16 pixels wide.
        fnt.font_info.fontline_bytes = (fnt.font_info.bbx.width + 7) / 8;
      }
      else if (token == "STARTPROPERTIES") {
        // STARTPROPERTIES 20
        // do not check line count now
        state = ls_StartProperties;
      }
      else if (token == "CHARS") {
        std::getline(ssline, token);
        fnt.font_info.chars = std::stoi(token);
        // allocate area for "character_count" * "height"
        fnt.font_bitmaps.resize(fnt.font_info.chars * fnt.font_info.bbx.height * fnt.font_info.fontline_bytes);
        fnt.charnames_array.resize(fnt.font_info.chars);
        fnt.codepoints_array.resize(fnt.font_info.chars);
        fnt.bbx_array.resize(fnt.font_info.chars);
      }
      else if (token == "STARTCHAR") {
        std::getline(ssline, token);
        current_char.friendly_name = token;
        current_char.encoding = 0;
        current_char.dwidth_x = 0;
        current_char.dwidth_y = 0;
        current_char.bbx = fnt.font_info.bbx; // inherited from FONTBOUNDINGBOX
        current_char.bits_to_shift = 0;
        current_char.empty_lines_bottom = 0;
        current_char.empty_lines_top = 0;

        state = ls_Char;
      }
      else {
        // unexpected token
      }
      break;
    case ls_StartProperties:
      if (token == "ENDPROPERTIES") {
        state = ls_StartFont;
      }
      /*
      FAMILY_NAME "Terminus"
        FOUNDRY "xos4"
        SETWIDTH_NAME "Normal"
        ADD_STYLE_NAME ""
        COPYRIGHT "Copyright (C) 2019 Dimitar Toshkov Zhekov"
        NOTICE "Licensed under the SIL Open Font License, Version 1.1"
        WEIGHT_NAME "Bold"
        SLANT "R"
        PIXEL_SIZE 16
        POINT_SIZE 160
        RESOLUTION_X 72
        RESOLUTION_Y 72
        SPACING "C"
        AVERAGE_WIDTH 80
        CHARSET_REGISTRY "ISO10646"
        CHARSET_ENCODING "1"
        MIN_SPACE 8
        */
      else if (token == "FAMILY_NAME") {
        //FAMILY_NAME "Terminus"
        std::getline(ssline, token, ' ');
        fnt.font_properties.Family_name = UnQuote(token);
      }
      else if (token == "COPYRIGHT") {
        std::getline(ssline, token, ' ');
        fnt.font_properties.Copyright = UnQuote(token);
      }
      else if (token == "NOTICE") {
        std::getline(ssline, token, ' ');
        fnt.font_properties.Notice = UnQuote(token);
      }
      else if (token == "WEIGHT_NAME") {
        // "Medium", "Bold"
        std::getline(ssline, token, ' ');
        fnt.font_properties.Weight_name = UnQuote(token);
      }
      else if (token == "PIXEL_SIZE") {
        std::getline(ssline, token, ' ');
        fnt.font_properties.pixel_size = std::stoi(token);
      }
      else if (token == "FONT_ASCENT") {
        // FONT_ASCENT 12
        // 12 of the 16 pixels in height are above the baseline.
        std::getline(ssline, token, ' ');
        fnt.font_properties.font_ascent = std::stoi(token);
      }
      else if (token == "FONT_DESCENT") {
        // FONT_DESCENT 4
        // 4 of the 16 pixels in height are below the baseline
        std::getline(ssline, token, ' ');
        fnt.font_properties.font_descent = std::stoi(token);
      }
      else if (token == "DEFAULT_CHAR") {
        // DEFAULT_CHAR 65533
        std::getline(ssline, token, ' ');
        fnt.font_properties.default_char = std::stoi(token);
      }
      break;
    case ls_Char:
      if (token == "ENDCHAR") {
        // add to vector
        state = ls_StartFont;
      }
      else if (token == "DWIDTH") {
        // DWIDTH 9 0
        // declares the Device Width of a glyph. After the glyph is rendered, the start of the next glyph is 
        // offset 9 pixels on the X-axis and 
        // offset 0 pixels on the Y-axis from the current glyph origin.
        // They are not necessarily equal to the width of the glyph. 
        // It is simply the offset on the X-axis to move the current point to the start of the next glyph. 
        // dwidth_y is not used in Avisynth, fonts are of fixed heights
        std::getline(ssline, token, ' ');
        current_char.dwidth_x = std::stoi(token);
        std::getline(ssline, token, ' ');
        current_char.dwidth_y = std::stoi(token);
      }
      else if (token == "ENCODING") {
        // ENCODING 32
        std::getline(ssline, token, ' ');
        current_char.encoding = std::stoi(token);
      }
      else if (token == "BBX") {
        // BBX 8 16 0 -4 bounding box. 8 pixels wide and 16 pixels tall;
        // lower left corner is offset by 0 on the X and -4 pixels on the Y axis.

        // Originally we only assumed Monospaced, uniformly fixed size fonts.
        // But technically we can have variable width characters inside due to per character BBOX.
        // E.g. Latin characters usually have a DWIDTH of 8, while others like CJK Unifont
        // characters have a DWIDTH of 16. (like U + 007F and CJK glyphs). Because of these
        // wider characters, the FONTBOUNDINGBOX must be 16 pixels wide to signal the renderer
        // the maximum possible width for the font type (useful for buffer preallocation). But
        // ordinarly Latin characters are still 8 bits wide.
        // So can we support "fixed width" fonts which contains character bitmaps of different widths.

        std::getline(ssline, token, ' ');
        current_char.bbx.width = std::stoi(token);
        std::getline(ssline, token, ' ');
        current_char.bbx.height = std::stoi(token);
        std::getline(ssline, token, ' ');
        current_char.bbx.offset_x = std::stoi(token);
        std::getline(ssline, token, ' ');
        current_char.bbx.offset_y = std::stoi(token);
        current_char.empty_lines_bottom = current_char.bbx.offset_y - fnt.font_info.bbx.offset_y;
        current_char.empty_lines_top = fnt.font_info.bbx.height - (current_char.empty_lines_bottom + current_char.bbx.height);
        current_char.bits_to_shift = current_char.bbx.width - current_char.dwidth_x; // actual per-characer dwidth_x can differ from font_info.bbx.width
      }
      else if (token == "BITMAP") {
        /* space:
          STARTCHAR space
          ENCODING 32
          SWIDTH 692 0
          DWIDTH 9 0
          BBX 0 0 0 0
          BITMAP
          ENDCHAR

          or

          6x12:
          BITMAP
          00
          00
          70
          88
          08
          30
          08
          08
          88
          70
          00
          00
          ENDCHAR

          10x18:
          BITMAP
          0000
          0000
          0000
          3F00
          6180
          6180
          0180
          0180
          1F00
          0180
          0180
          0180
          6180
          6180
          3F00
          0000
          0000
          0000
          ENDCHAR
        */
        fnt.codepoints_array[char_counter] = static_cast<uint32_t>(current_char.encoding);
        fnt.charnames_array[char_counter] = current_char.friendly_name;
        fnt.bbx_array[char_counter] = current_char.bbx;
        char_counter++;

        // by spec: charlines are left aligned within byte boundaries

        // fill empty top lines
        for (int count = 0; count < current_char.empty_lines_top; count++) {
          // fontline_bytes bytes per line
          for(int i = 0; i < fnt.font_info.fontline_bytes; i++)
            fnt.font_bitmaps[line_counter++] = 0;
        }

        // one character line is of (almost) arbitrary length, not limited to 16 or 32 bits
        std::vector<uint8_t> charline_buffer(fnt.font_info.fontline_bytes);
        const size_t linebuf_len = fnt.font_info.fontline_bytes;

        for(int count = 0; count < current_char.bbx.height; count++)
        {
          std::getline(ss, temp);

          size_t len = temp.length();
          const size_t bytes_defined = len / 2;

          if (len % 2) {
            // weird, one byte must be defined on two hexadecimal characters
            // FIXME: should give an error
          }
          if (bytes_defined > linebuf_len) {
            // weird, more bytes defined that needed for the font width defined in the header
            // FIXME: should give an error
            len = linebuf_len * 2; // until then a safe limit
          }
          size_t buf_ctr = 0;
          // two hex characters by two character then put into byte buffer
          for (size_t i = 0; i < len; i += 2) {
            auto ssss = temp.substr(i, 2); // next two hex chars
            charline_buffer[buf_ctr++] = (uint8_t)std::stoul(ssss, nullptr, 16);
          }
          for (auto i = buf_ctr; i < linebuf_len; i++)
            charline_buffer[i] = 0;

          // shift full buffer, they are msb...lsb for increasing addresses
          if (current_char.bits_to_shift < 0) {
            const int bits_to_shift_left = -current_char.bits_to_shift;
            vector_shl(charline_buffer.data(), fnt.font_info.fontline_bytes, bits_to_shift_left);
          }
          else if (current_char.bits_to_shift > 0) {
            const int bits_to_shift_right = current_char.bits_to_shift;
            vector_shr(charline_buffer.data(), fnt.font_info.fontline_bytes, bits_to_shift_right);
          }

          for(int i = 0; i < fnt.font_info.fontline_bytes; i++) {
            fnt.font_bitmaps[line_counter++] = charline_buffer[i];
          }
        }

        // fill empty bottom lines
        for (int count = 0; count < current_char.empty_lines_bottom; count++) {
          // fontline_bytes bytes per line
          for (int i = 0; i < fnt.font_info.fontline_bytes; i++)
            fnt.font_bitmaps[line_counter++] = 0;
        }

      }
      break;
    }
  }

  return fnt;
}
  /*

  STARTFONT 2.1
    FONT -xos4-Terminus-Bold-R-Normal--16-160-72-72-C-80-ISO10646-1
    SIZE 16 72 72
    FONTBOUNDINGBOX 8 16 0 -4

    STARTPROPERTIES 20
      FAMILY_NAME "Terminus"
      FOUNDRY "xos4"
      SETWIDTH_NAME "Normal"
      ADD_STYLE_NAME ""
      COPYRIGHT "Copyright (C) 2019 Dimitar Toshkov Zhekov"
      NOTICE "Licensed under the SIL Open Font License, Version 1.1"
      WEIGHT_NAME "Bold"
      SLANT "R"
      PIXEL_SIZE 16
      POINT_SIZE 160
      RESOLUTION_X 72
      RESOLUTION_Y 72
      SPACING "C"
      AVERAGE_WIDTH 80
      CHARSET_REGISTRY "ISO10646"
      CHARSET_ENCODING "1"
      MIN_SPACE 8
      FONT_ASCENT 12
      FONT_DESCENT 4
      DEFAULT_CHAR 65533
    ENDPROPERTIES

    CHARS 1354

    STARTCHAR space
      ENCODING 32
      SWIDTH 500 0
      DWIDTH 8 0
      BBX 8 16 0 -4
      BITMAP
      ...
      00
    ENDCHAR

  ENDFONT
  */

static constexpr int ATA_LEFT = 1;
static constexpr int ATA_RIGHT = 2;
static constexpr int ATA_CENTER = 4;

static constexpr int ATA_TOP = 8;
static constexpr int ATA_BOTTOM = 16;
static constexpr int ATA_BASELINE = 32;

static int alignToBitmask(int align_1_to_9)
{
  // alignment 1-9: digit positions on numeric keypad
  int al = 0;
  switch (align_1_to_9) // This spec where [X, Y] is relative to the text (inverted logic)
  {
  case 1: al = ATA_BOTTOM | ATA_LEFT; break;     // .----
  case 2: al = ATA_BOTTOM | ATA_CENTER; break;   // --.--
  case 3: al = ATA_BOTTOM | ATA_RIGHT; break;    // ----.
  case 4: al = ATA_BASELINE | ATA_LEFT; break;   // .____
  case 5: al = ATA_BASELINE | ATA_CENTER; break; // __.__
  case 6: al = ATA_BASELINE | ATA_RIGHT; break;  // ____.
  case 7: al = ATA_TOP | ATA_LEFT; break;        // `----
  case 8: al = ATA_TOP | ATA_CENTER; break;      // --`--
  case 9: al = ATA_TOP | ATA_RIGHT; break;       // ----`
  default: al = ATA_BASELINE | ATA_LEFT; break;  // .____
  }
  return al;
}

static int getColorForPlane(int plane, int color)
{
  switch (plane) {
  case PLANAR_A:
    return (color >> 24) & 0xff; break;
  case PLANAR_R:
  case PLANAR_Y:
    return (color >> 16) & 0xff; break;
  case PLANAR_G:
  case PLANAR_U:
    return (color >> 8) & 0xff; break;
  case PLANAR_B:
  case PLANAR_V:
    return color & 0xff; break;
  }
  return color & 0xFF;
}

template<typename pixel_t, bool fadeBackground>
void AVS_FORCEINLINE LightOnePixelPackedRGB(const bool lightIt, BYTE* _dp, int val_color_R, int val_color_G, int val_color_B)
{
  pixel_t* dp = reinterpret_cast<pixel_t*>(_dp);
  if (lightIt) {
    dp[0] = val_color_B;
    dp[1] = val_color_G;
    dp[2] = val_color_R;
  }
  else {
    if constexpr (fadeBackground) {
      dp[0] = (pixel_t)((dp[0] * 7) >> 3);
      dp[1] = (pixel_t)((dp[1] * 7) >> 3);
      dp[2] = (pixel_t)((dp[2] * 7) >> 3);
    }
  }
}

// isChroma: U or V plane of a non-subsampled (4:4:4) format; its background fades toward
// the chroma center, not toward the luma black level
template<typename pixel_t, bool fadeBackground, bool isRGB, bool isChroma = false>
void AVS_FORCEINLINE LightOnePixel(const bool lightIt, pixel_t* dstp, int j, pixel_t& val_color, int bits_per_pixel)
{
  // some optimization hint
  if constexpr (sizeof(pixel_t) == 1)
    bits_per_pixel = 8;
  else if constexpr (sizeof(pixel_t) == 4)
    bits_per_pixel = 32;

  if (lightIt) { // character definition bits aligned to msb
    dstp[j] = val_color;
  }
  else {
    // 16 = y_min
    // speed optimization: one subtraction less, 5-8% faster
    // (((Y - 16) * 7) >> 3) + 16 = ((Y * 7) >> 3) + 2
    // in general: ((Y * 7) >> 3) + n, where n = range_min - ((range_min * 7) >> 3)
    if constexpr (fadeBackground) {
      // background darkening
      if constexpr (isRGB) {
        if constexpr (sizeof(pixel_t) != 4)
          dstp[j] = (pixel_t)((dstp[j] * 7) >> 3);
        else {
          constexpr float factor = 7.0f / 8;
          dstp[j] = (pixel_t)(dstp[j] * factor);
        }
      }
      else if constexpr (isChroma) {
        // same as LightOneUVPixel for the subsampled formats:
        // (((U - range_half) * 7) >> 3) + range_half = ((U * 7) >> 3) + n
        if constexpr (sizeof(pixel_t) != 4) {
          const int range_half = 1 << (bits_per_pixel - 1);
          const int n = range_half - ((range_half * 7) >> 3);
          dstp[j] = (pixel_t)(((dstp[j] * 7) >> 3) + n);
        }
        else {
          constexpr float chroma_center = 0.0f;
          dstp[j] = (pixel_t)(((dstp[j] - chroma_center) * 7 / 8) + chroma_center);
        }
      }
      else {
        if constexpr (sizeof(pixel_t) != 4) {
          const int range_min = 16 << (bits_per_pixel - 8);
          const int n = range_min - ((range_min * 7) >> 3);
          dstp[j] = (pixel_t)(((dstp[j] * 7) >> 3) + n); // (_dstp[j] - range_min) * 7) >> 3) + range_min);
        }
        else {
          constexpr float range_min_f = y8tof(16);
          dstp[j] = (pixel_t)(((dstp[j] - range_min_f) * 7 / 8) + range_min_f);
        }
      }
    }
  }
}

// One chroma sample (U and V) of the text, from the weighted luma counts of its kernel (see RenderChroma).
// font, halo: weighted count of the font and of the halo (outline) luma bits; rect: weighted count of the luma
// samples inside the text rectangle. The weights sum to 1 << log2total, the rest of it is background. With
// fadeBackground the background inside the rectangle is faded (as luma is there), the part outside is kept.
// Integer range: 65535 * 1024 < 2^31.
template<typename pixel_t, bool fadeBackground, int log2total>
static void LightOneUVPixel(pixel_t& dstU, pixel_t& dstV,
  const pixel_t font_color_u, const pixel_t font_color_v, const pixel_t halo_color_u, const pixel_t halo_color_v,
  const int font, const int halo, const int rect,
  int bits_per_pixel)
{
  constexpr int total = 1 << log2total;
  if constexpr (fadeBackground) {
    if (rect == 0) return; // kernel outside the text rectangle: no change
  }
  else {
    if (font == 0 && halo == 0) return; // background only: no change
  }

  // some optimization hint
  if constexpr (sizeof(pixel_t) == 1)
    bits_per_pixel = 8;
  else if constexpr (sizeof(pixel_t) == 4)
    bits_per_pixel = 32;

  if (font == total) {
    dstU = font_color_u;
    dstV = font_color_v;
    return;
  }
  if (halo == total) {
    dstU = halo_color_u;
    dstV = halo_color_v;
    return;
  }

  const pixel_t actualU = dstU;
  const pixel_t actualV = dstV;
  pixel_t fadedU = actualU;
  pixel_t fadedV = actualV;
  int bg_faded = 0; // background share inside the rectangle (faded)
  int bg_kept = total - font - halo; // background share kept as is
  if constexpr (fadeBackground) {
    bg_faded = rect - font - halo;
    bg_kept = total - rect;
    // fade the existing background color toward the chroma center
    // speed optimization: one subtraction less
    // (((U - 128) * 7) >> 3) + 128 = ((U * 7) >> 3) + 16
    // in general: ((U * 7) >> 3) + n where n = range_half - ((range_half * 7) >> 3)
    if constexpr (sizeof(pixel_t) != 4) {
      const int range_half = 1 << (bits_per_pixel - 1);
      const int n = range_half - ((range_half * 7) >> 3);
      fadedU = (pixel_t)(((actualU * 7) >> 3) + n); // ((((U - range_half) * 7) >> 3) + range_half);
      fadedV = (pixel_t)(((actualV * 7) >> 3) + n);
    }
    else {
      constexpr float chroma_center = 0.0f; // ancient times this was 0.5
      constexpr float factor = 7.0f / 8.0f;
      fadedU = (pixel_t)(((actualU - chroma_center) * factor) + chroma_center);
      fadedV = (pixel_t)(((actualV - chroma_center) * factor) + chroma_center);
    }
  }

  if constexpr (fadeBackground) {
    // Speed: only faded background, the most common case in the fade area: no need for the mix below,
    // which would give the same (faded * total + rounder) >> log2total = faded.
    if (bg_faded == total) {
      dstU = fadedU;
      dstV = fadedV;
      return;
    }
  }

  // resulting color weighted by the pixel kinds
  if constexpr (sizeof(pixel_t) != 4) {
    constexpr int rounder = total >> 1;
    const int u = font_color_u * font + halo_color_u * halo + fadedU * bg_faded + actualU * bg_kept + rounder;
    const int v = font_color_v * font + halo_color_v * halo + fadedV * bg_faded + actualV * bg_kept + rounder;
    dstU = (pixel_t)(u >> log2total);
    dstV = (pixel_t)(v >> log2total);
  }
  else {
    constexpr float scale = 1.0f / total; // exact, power of two
    const float u = font_color_u * font + halo_color_u * halo + fadedU * bg_faded + actualU * bg_kept;
    const float v = font_color_v * font + halo_color_v * halo + fadedV * bg_faded + actualV * bg_kept;
    dstU = u * scale;
    dstV = v * scale;
  }
}

// s (containing the indexes to the fontbitmap array) is modified in-place
static void adjustWriteLimits(std::vector<int>& s, const std::vector<BBX>& bbx_array, const int vi_width, const int vi_height,
  const int FONT_WIDTH_not_used, const int FONT_HEIGHT, 
  int align, 
  const bool useHalocolor,
  int& x, int& y, int& len, int& startindex, int& xstart, int& ystart, int& yend)
{
  const int al = alignToBitmask(align);

  // cannot calculate with len * FONT_WIDTH, because characters can have different widths
  int total_width = 0;
  for (int i = 0; i < len; i++) {
    // bbx_array is the array of bounding boxes for each character
    // s[i] is the index to the fontbitmap array
    total_width += bbx_array[s[i]].width;
  }

  // alignment X
  if (al & ATA_RIGHT)
    x -= (total_width - 1);
  else if (al & ATA_CENTER)
    x -= (total_width / 2);

  // alignment Y
  if (al & ATA_BASELINE)
    y -= FONT_HEIGHT / 2;
  else if (al & ATA_BOTTOM)
    y -= (FONT_HEIGHT - 1);

  const int final_height = useHalocolor ? FONT_HEIGHT + 2 : FONT_HEIGHT;
  if (useHalocolor) y = y - 1; // one more top line, adjust y (we do it after the alignment!)

  // Chop text if exceed right margin
  // keep last character of which at least one pixel can be drawn

  // Fixed FONT_WIDTH calculation logic is replaced with logic which allows variable width characters
  // Starting the truncation from the right side until the last character fits into the right margin, at least one pixel of it.
  // bbx.width is the width of the character, not the whole font, this amount must be subtracted from the right margin.
  while(len > 0 && total_width > vi_width - x + bbx_array[s[len - 1]].width) {
    // remove last character
    len--;
    total_width -= bbx_array[s[len]].width;
  }
  /* // old method left here intentionally:
  if (len * FONT_WIDTH > width - x)
    len = (width - x + FONT_WIDTH - 1) / FONT_WIDTH;
  */

  // FIXME: what if the character not, but its left side halo would be seen?

  startindex = 0;
  xstart = 0;
  // Chop 1st char if exceed left margin
  if (x < 0) {
    // startindex is the first position in the string to be drawn
    // Previously we had a global font-constant FONT_WIDTH, it was the same as x / FONT_WIDTH
    // and the internal start position could be calculated as (-x) % FONT_WIDTH.
    // Now that we allow that BDF originated characters can have different widths, we must
    // calculate the startindex one-by-one from the left side.
    while (startindex < len && x < 0) {
      // s[startindex] is the index to the fontbitmap array
      // bbx_array[s[startindex]].width is the width of the character
      x += bbx_array[s[startindex]].width;
      startindex++;
    }
    // x is the remaining negative offset, which is the start position of the first character
    xstart = x; // this is the start position of the first character
    // OLD method left here intentionally:
    /*
      startindex = (-x) / FONT_WIDTH;
      xstart = (-x) % FONT_WIDTH;
    */
    x = 0;
  }

  ystart = 0;
  yend = final_height;
  // Chop font if exceed bottom margin
  if (y > vi_height - final_height)
    yend = vi_height - y;

  // Chop font if exceed top margin
  if (y < 0) {
    ystart = -y;
    y = 0;
  }

  // Roll in start index
  if (startindex > 0) {
    s.erase(s.begin(), s.begin() + startindex);
    len -= startindex;
  }
}

// Inserts leftmost 'bitcount' bits of fontlinebuf at the end of the byte buffer dst,
// Target is the 'bitposition'th bit
static void insert_from_msb_bit(uint8_t *dst, int bitposition, const uint8_t* fontlinebuf, int fontlinebuf_size, int bitcount)
{
  int pos = bitposition / 8;
  int bitindex = bitposition % 8;
  if (bitindex > 0) {
    const int usable_msb_count = 8 - bitindex;
    //                     mask       FF >> (bitindex)
    // bitindex 1   mask = 10000000   01111111 
    // bitindex 7   mask = 11111110   00000001
    const uint8_t mask = (uint8_t)(0xFF >> bitindex); // leftmost (msb) N bits
    const uint8_t partial_val = (uint8_t)((fontlinebuf[0] >> bitindex) & mask); // shift leftmost bits lsb
    dst[pos++] |= partial_val; // or'd with existing

    bitcount -= usable_msb_count;

    // deal with shifted buffer from now on
    int bufpos = 0;
    while (bitcount > 0) {
      uint8_t val;
      if (bufpos <fontlinebuf_size - 1)
        val = (fontlinebuf[bufpos] << usable_msb_count) | (fontlinebuf[bufpos + 1] >> (8 - usable_msb_count));
      else 
        val = (fontlinebuf[bufpos] << usable_msb_count); // suppose that unused lsb bits were 0 in the end

      // put actual byte into buffer
      dst[pos++] = val;
      bufpos++;
      bitcount -= 8;
    }
    return;
  }

  int bufpos = 0;
  while (bitcount > 0) {
    // put actual byte into buffer
    dst[pos++] = fontlinebuf[bufpos++]; // suppose that unused lsb bits were 0 in the end
    bitcount -= 8;
  }
}

static uint8_t get_bit(const uint8_t* src, const int bitposition)
{
  int pos = bitposition / 8;
  int bitindex = bitposition % 8;
  const uint8_t mask = 1 << (7 - bitindex);
  return src[pos] & mask;
}

// Speed-up of the luma loops, used only without fadeBackground: then only the set bits of the bitmap change the
// frame, and text is mostly empty space (typically over 3/4 of the bits are zero). Testing whole bytes
// avoids the per-bit get_bit and its unpredictable branch there.
// True if bit position tx is at a byte start and that byte is empty in both bitmaps: the caller can skip these
// 8 pixels. Before the first byte boundary (xstart is not always aligned), and in non-empty bytes, the loops go
// on bit by bit. An empty byte reaching over the end of the text is no problem: nothing would be written there.
// outline: nullptr without halo.
AVS_FORCEINLINE static bool is_empty_bitmap_byte(const uint8_t* fontline, const uint8_t* outline, int tx)
{
  return (tx & 7) == 0 && (fontline[tx / 8] | (outline ? outline[tx / 8] : 0)) == 0;
}

void PreRendered::make_outline() {
  auto h = stringbitmap.size();
  auto w = stringbitmap[0].size();

  // circular line buffer for holding precalculated shifted lines
  std::vector<uint8_t> buf1(w);
  std::vector<uint8_t> buf2(w);
  std::vector<uint8_t> buf3(w);

  // it's unnecessary to mask the last byte against valid bitcounter, they won't drawn
  // const uint8_t mask = ~(0xFF >> (8 - (bitcounter % 8)));

  // shift a line left and rights and result is or'd
  auto make_dizzyLR = [](uint8_t* dst, auto src, size_t w) {
    if (w == 1)
    {
      *dst = (src[0] << 1) | (src[0] >> 1);
      return;
    }
    // leftmost
    uint8_t left = (src[0] << 1) | (src[1] >> (8 - 1));
    uint8_t right = 0 | (src[0] >> 1);
    *dst = left | right;
    dst++;
    src++;
    // middle
    for (size_t i = 1; i < w - 1; ++i)
    {
      left = (src[0] << 1) | (src[1] >> (8 - 1));
      right = (src[-1] << (8 - 1)) | (src[0] >> 1);
      *dst++ = left | right;
      src++;
    }
    // rightmost
    left = (src[0] << 1) | 0;
    right = (src[-1] << (8 - 1)) | (src[0] >> 1);
    *dst = left | right;
    };

#if defined(_MSC_VER) && _MSC_VER < 1944
  // MSVC's optimizer fail (as of 17.8.4, release, optimize for speed, SSE2) would
  // Workaround. Bad code generated. Compiler bug.
  // Fixed (said) in 17.9 preview 3 (_MSC_VER 1939)
  // Tested again in 17.14 (_MSC_VER 1944), it's OK.
  // https://developercommunity.visualstudio.com/t/Bad-c-codegen-in-1784-x64-unless-se/10565370
  volatile uint8_t* src_prev;
  volatile uint8_t* src;
#else
  uint8_t* src_prev;
  uint8_t* src;
#endif
  uint8_t* src_next;
  uint8_t* dst;

  uint8_t* tmp_line;
  uint8_t* prev_line_LR = buf1.data();
  uint8_t* curr_line_LR = buf2.data();
  uint8_t* next_line_LR = buf3.data();

  // line 0, no previous line
  size_t y = 0;
  dst = stringbitmap_outline[y].data();
  src = stringbitmap[y].data();
  src_next = stringbitmap[y + 1].data();
  make_dizzyLR(curr_line_LR, src, w);
  make_dizzyLR(next_line_LR, src_next, w);
  for (size_t x = 0; x < w; x++) {
    dst[x] = (curr_line_LR[x] | next_line_LR[x] | src_next[x]) & ~src[x];
  }
  tmp_line = prev_line_LR;
  prev_line_LR = curr_line_LR;
  curr_line_LR = next_line_LR;
  next_line_LR = tmp_line;

  src_prev = src;
  src = src_next;
  y++;

  // middle lines, y runs on 1..(h-2)
  for (; y < h - 1; y++)
  {
    dst = stringbitmap_outline[y].data();
    src_next = stringbitmap[y + 1].data();
    make_dizzyLR(next_line_LR, src_next, w);
    for (size_t x = 0; x < w; x++) {
      dst[x] = (prev_line_LR[x] | curr_line_LR[x] | next_line_LR[x] | src_prev[x] | src_next[x]) & ~src[x];
    }
    tmp_line = prev_line_LR;
    prev_line_LR = curr_line_LR;
    curr_line_LR = next_line_LR;
    next_line_LR = tmp_line;
    src_prev = src;
    src = src_next;
  }
  // last one, no next line
  dst = stringbitmap_outline[y].data();
  for (size_t x = 0; x < w; x++) {
    dst[x] = (prev_line_LR[x] | curr_line_LR[x] | src_prev[x]) & ~src[x];
  }
}

PreRendered::PreRendered(
  const uint8_t* fonts,
  const int fontline_bytes_storage,
  const int _vi_width, const int _vi_height, // from VideoInfo
  int _x, int _y, // they may change
  std::vector<int>& s,
  // each character can have different width, variable bbx.width is used instead of font-level constant FONT_WIDTH
  const std::vector<BBX>& bbx_array,
  int align,
  const bool _useHalocolor,

  // Each character can have a different width; variable bbx.width is used instead of the font-level constant FONT_WIDTH.
  //
  // A BDF font can contain mixed sizes. The next two parameters (FONT_WIDTH, FONT_HEIGHT) represent the global maximum
  // font size. A font file may contain per-character widths and heights in bbx_array.
  //
  // Regarding the wider spacing in Unifont:
  // This is due to how FONTBOUNDINGBOX is interpreted during rendering. In Unifont's BDF header, FONTBOUNDINGBOX is set
  // to 16 pixels wide. In earlier versions of Avisynth, having a BDF header with FONTBOUNDINGBOX set to 16 pixels wide
  // globally caused all glyphs, regardless of their actual per-character width, to be rendered with 16-pixel spacing in
  // Avisynth's Text() function. This resulted in visible gaps for narrower, 8-pixel wide Latin characters.
  //
  // While Unifont is conceptually monospaced, it technically includes variable-width glyphs.
  // For example, Latin characters have DWIDTH 8, but others (like U+007F or CJK characters) use DWIDTH 16.
  //
  // Thus, the global FONTBOUNDINGBOX only specifies a maximum size for buffer allocation and must be 16 pixels wide.
  // In a fully BDF-compliant renderer, the per-character DWIDTH and BBX values can override the global bounding box
  // for spacing and alignment.
  //
  // Since Avisynth 3.7.6, the BDF renderer implements this override behavior.

  const int FONT_WIDTH, const int FONT_HEIGHT
  )
  :
  useHalocolor(_useHalocolor), vi_width(_vi_width), vi_height(_vi_height)
{
  len = (int)s.size();
  x = _x;
  y = _y;
  xstart = 0;
  ystart = 0;
  yend = 0;
  text_width = 0;

  int startindex = 0;

  // optional two additional lines for top and bottom for outline
  stringbitmap_height = useHalocolor ? FONT_HEIGHT + 2 : FONT_HEIGHT;

  adjustWriteLimits(s, // s is modified in-place
    bbx_array,  // bounding boxes for each character
    vi_width, vi_height,
    FONT_WIDTH, // not used, bbx_array contains per-character widths, but must be passed
    FONT_HEIGHT, // extra top and bottom for halo
    align,
    useHalocolor,
    // adjusted parameters
    x, y, len, startindex, xstart, ystart, yend
    );

  if (len <= 0)
    return;

  // prepare font mask and outline mask

  // instead of FONT_WIDTH * len we have to calculate the total width of the string dynamically,
  // because characters can have different widths, so we use bbx_array[s[i]].width
  int total_font_width = 0;
  for (int i = 0; i < len; i++) {
    // bbx_array is the array of bounding boxes for each character
    // s[i] is the index to the fontbitmap array
    // total_font_width += bbx_array[s[i]].width;
    total_font_width += bbx_array[s[i]].width; // this is the total width of the string
  }

  // the chroma renderer (RenderChroma) checks the bitmap bounds, no safety columns are needed
  const int stringbitmap_width = total_font_width + (useHalocolor ? 2 : 0);
  const int stringbitmapline_bytes = (stringbitmap_width + 7) / 8;

  // allocate actual space
  stringbitmap.resize(stringbitmap_height);
  for (auto& subarray : stringbitmap) subarray.resize(stringbitmapline_bytes);
  if (useHalocolor) {
    stringbitmap_outline.resize(stringbitmap_height);
    for (auto& subarray : stringbitmap_outline) subarray.resize(stringbitmapline_bytes);
  }

  // fill matrix with fonts
  // optionally leave spare left-right-top-bottom pixel lines for out-of-matrix halo pixels
  uint8_t zerobyte[1] = { 0 };
  int bitcounter = 0;

  if (useHalocolor) {
    // fill leftmost 1 bitcolumn with 0
    for (int ty = 0; ty < stringbitmap_height; ty++) {
      uint8_t zerobyte[1] = { 0 };
      insert_from_msb_bit(&stringbitmap[ty][0], 0, &zerobyte[0], 1, 1);
    }
    bitcounter++;
  }

  // if outline needed, leave place for top outline, copy actual character to the second row
  const int ypos_of_char_in_bitmap = useHalocolor ? 1 : 0;

  for (int i = 0; i < len; i++) {
    int num = s[i];
    int actual_font_width = bbx_array[num].width; // per character width, instead of font-level constant FONT_WIDTH;
    const uint8_t* fontlinebuf = &fonts[num * FONT_HEIGHT * fontline_bytes_storage];
    for (int ty = 0; ty < FONT_HEIGHT; ty++) {
      // stuff FONT_WIDTH bits from fontline into bitcounter_th bit of target line
      insert_from_msb_bit(&stringbitmap[ty + ypos_of_char_in_bitmap][0], bitcounter, fontlinebuf, fontline_bytes_storage, actual_font_width);
      fontlinebuf += fontline_bytes_storage;
    }
    bitcounter += actual_font_width;
  }
  if (useHalocolor) {
    // fill rightmost 1 bitcolumn with 0 (right halo)
    for (int ty = 0; ty < stringbitmap_height; ty++) {
      insert_from_msb_bit(&stringbitmap[ty][0], bitcounter, &zerobyte[0], 1, 1);
    }
    bitcounter++;
  }

  assert(bitcounter == stringbitmap_width);

  if (useHalocolor)
    make_outline();

  // recalculate the text width with assuming variable character widths

  // actual visible pixel count
  // Instead of (len - 1) * FONT_WIDTH, we must calculate the text width based on the actual character widths
  text_width = total_font_width - xstart;
  if (x + text_width > vi_width) text_width -= (x + text_width - vi_width);

  if (useHalocolor) {
    // x was calculated w/o the extra left-right padding
    xstart += 1; // worst case, skip leftmost extra halo column
    if (x + xstart - 1 >= 0 && x > 0) {
      x -= 1;
      xstart -= 1;
      text_width += 1;
    }
    // still good an increased text_width for last halo column?
    if (x + text_width + 1 < vi_width)
      text_width += 1;
  }
}

template<typename pixel_t>
static auto getHBDColor_UV(int color, int bits_per_pixel)
{
  if (bits_per_pixel < 32)
    return (pixel_t)(color << (bits_per_pixel - 8));
  return (pixel_t)uv8tof(color);
}

template<typename pixel_t>
static auto getHBDColor_Y(int color, int bits_per_pixel)
{
  if (bits_per_pixel < 32)
    return (pixel_t)(color << (bits_per_pixel - 8));
  return (pixel_t)y8tof(color);
}

template<typename pixel_t>
static auto getHBDColor_RGB(int color, int bits_per_pixel)
{
  if (bits_per_pixel <= 16) {
    const int max_pixel_value = (1 << (bits_per_pixel & 31)) - 1;
    return (pixel_t)((float)color * max_pixel_value / 255); // 0..255 --> 0..1023,4095,16383,65535
  }
  return (pixel_t)(color / 255.0f); // 0..255 -> 0..1.0
}


template<typename pixel_t, bool useHalocolor, bool fadeBackground, bool isRGB>
void Render1by1Planes(int bits_per_pixel, int color, int halocolor, int* pitches, BYTE** dstps, PreRendered& pre,
  const int planeCount, const bool is444)
{
  // 1:1 planes, Y or planar RGB or 4:4:4 U/V

  int planes_y[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
  int planes_r[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
  int* planes = isRGB ? planes_r : planes_y;

  for (int p = 0; p < planeCount; p++)
  {
    const int plane = planes[p];

    if (!(isRGB || plane == PLANAR_Y || ((plane == PLANAR_U || plane == PLANAR_V) && is444)))
      continue; // Y, R, G, B is O.K. U, V is OK if 444

    pixel_t val_color;
    pixel_t val_color_outline;

    const int planecolor = getColorForPlane(plane, color);
    const int planecolor_outline = getColorForPlane(plane, halocolor);
    if (isRGB) {
      val_color = getHBDColor_RGB<pixel_t>(planecolor, bits_per_pixel);
      val_color_outline = getHBDColor_RGB<pixel_t>(planecolor_outline, bits_per_pixel);
    }
    else if (plane == PLANAR_U || plane == PLANAR_V) {
      val_color = getHBDColor_UV<pixel_t>(planecolor, bits_per_pixel);
      val_color_outline = getHBDColor_UV<pixel_t>(planecolor_outline, bits_per_pixel);
    }
    else {// Y
      val_color = getHBDColor_Y<pixel_t>(planecolor, bits_per_pixel);
      val_color_outline = getHBDColor_Y<pixel_t>(planecolor_outline, bits_per_pixel);
    }

    const int pitch = pitches[p];
    BYTE* dstp = dstps[p] + pre.x * sizeof(pixel_t) + pre.y * pitch;

    // Start rendering. isChroma (4:4:4 U/V) is decided once per plane: it selects the
    // background fade toward the chroma center instead of the luma black level.
    // Trick: a runtime bool cannot be a template argument, but a type can do the task.
    // Lambda's 'auto' parameter makes it a template; decltype(tag)::value is a
    // compile time constant.
    auto render_plane = [&](auto isChromaTag) {
      constexpr bool isChroma = decltype(isChromaTag)::value;
      for (int ty = pre.ystart; ty < pre.yend; ty++) {
        pixel_t* _dstp = reinterpret_cast<pixel_t*>(dstp);
        uint8_t* fontline_ptr = pre.stringbitmap[ty].data();
        uint8_t* fontoutline_ptr = nullptr;
        if constexpr(useHalocolor)
          fontoutline_ptr = pre.stringbitmap_outline[ty].data();
        int j = 0;
        const int shifted_xstart = pre.xstart;
        for (int tx = shifted_xstart; tx < shifted_xstart + pre.text_width; tx++)
        {
          if constexpr (!fadeBackground) {
            if (is_empty_bitmap_byte(fontline_ptr, fontoutline_ptr, tx)) {
              tx += 7; // +1 in the loop header
              j += 8;
              continue;
            }
          }
          const bool lightIt = 0 != get_bit(fontline_ptr, tx);
          LightOnePixel<pixel_t, fadeBackground, isRGB, isChroma>(lightIt, _dstp, j, val_color, bits_per_pixel);
          if constexpr(useHalocolor) {
            if (!lightIt)
            {
              const bool lightIt_outline = 0 != get_bit(fontoutline_ptr, tx);
              LightOnePixel<pixel_t, fadeBackground, isRGB, isChroma>(lightIt_outline, _dstp, j, val_color_outline, bits_per_pixel);
            }
          }
          j += 1;
        }
        dstp += pitch;
      }
    };
    if (!isRGB && (plane == PLANAR_U || plane == PLANAR_V))
      render_plane(std::true_type{});
    else
      render_plane(std::false_type{});
  }
}

// floor(a / d) for d > 0, also for negative a
static constexpr int floor_div(int a, int d) { return a >= 0 ? a / d : -((d - 1 - a) / d); }

// The filter weights (see MaskFilter in blend_common.h), indexed by MaskFilter.
// Here the counts are integers, so the box cascades of the masks (mask_filter_cascade, needed there for
// C = SIMD on float) are not needed: the same result comes from at most 8 multiply-adds instead of the
// 15 (tent) or 31 (triangle) additions.
static constexpr int text_filter_weights[5][8] = {
  { 1 }, { 1, 1 }, { 1, 2, 1 }, { 1, 2, 3, 4, 3, 2, 1 }, { 1, 3, 5, 7, 7, 5, 3, 1 }
};

// Weighted sum of filter F over v[0 .. taps-1]
template<MaskFilter F>
AVS_FORCEINLINE static int text_filter_sum(const int* v) {
  int sum = 0;
  for (int k = 0; k < MaskFilterTaps<F>; k++)
    sum += text_filter_weights[F][k] * v[k];
  return sum;
}

// Vertical pass of the text chroma, for one bitmap row. For the set bits between bit_first and bit_last
// (inclusive): col[bit - bit_of_col0] += weight. Returns true if there was any, RenderChroma can then skip
// the whole chroma row without fading. Speed:
// - zero bytes, the most common ones, are skipped at once;
// - inside a byte the walk stops after its last set bit (v becomes 0);
// - no branch on the bit: font bits are unpredictable, a mispredicted branch costs more than an addition.
static bool add_set_bits(const uint8_t* row, int bit_first, int bit_last, int weight, int* col, int bit_of_col0)
{
  bool any = false;
  const int byte_first = bit_first / 8;
  const int byte_last = bit_last / 8;
  for (int byte = byte_first; byte <= byte_last; byte++) {
    unsigned v = row[byte];
    if (v == 0)
      continue;
    if (byte == byte_last)
      v &= (0xFF00u >> (bit_last % 8 + 1)) & 0xFFu;
    int i = byte * 8 - bit_of_col0;
    if (byte == byte_first) {
      // bits are msb first. Bits before bit_first may be before col[0]: skip them, not only mask them out.
      const int skip = bit_first % 8;
      v = (v << skip) & 0xFFu;
      i += skip;
    }
    any = any || v != 0;
    // The actual bit is always moved to the 0x80 position. It decides whether weight is added or not, without a
    // branch: v >> 7 is 0 or 1, its negative is a mask of all zeros or all ones, so "weight & mask" is 0 or weight.
    for (; v != 0; v = (v << 1) & 0xFFu, i++)
      col[i] += weight & -(int)(v >> 7);
  }
  return any;
}

// Chroma of the text for the subsampled formats (planar 4:2:0, 4:2:2, 4:1:1, 4:1:0, 4:4:0 and YUY2).
// A chroma sample is the mask kernel of the format and chroma placement (blend_common.h: MaskMode, MaskHFilter,
// MaskVFilter: box, [1 2 1], 4:1 tent or triangle), applied to three luma resolution maps of the text: the font
// bits, the halo (outline) bits and the text rectangle (the luma area which is drawn, or faded with
// fadeBackground). The weighted counts are the shares of the font, halo and background colors
// (LightOneUVPixel). So the chroma follows the same rules as a mask made of the luma text in Overlay or Layer:
// - same kernels (integer counts, so summing with the weights gives the same result as the box cascades of the masks);
// - luma positions outside the frame are clamped to the frame edge;
// - every chroma sample is drawn whose kernel reaches the text rectangle, also those beside the chroma blocks
//   of the text (the 4:1 kernels are 7 or 8 luma samples wide).
// The bitmap is read only inside the text rectangle, so it needs no extra (safety) columns or rows.
// Speed: text rendering happens on every frame, and font bitmaps are sparse, so
// - the vertical pass visits only the set bits (add_set_bits), one bitmap row at a time, not all taps of all columns;
// - a chroma row without any set bit is skipped (no fade), or needs no column work (fade);
// - the rectangle weights are computed once per chroma column and once per chroma row, then just multiplied;
// - the most common sample kinds leave LightOneUVPixel early (background only; font only; halo only).
// dstpU, dstpV: chroma plane start (YUY2: the first U and V byte of the frame); step: distance of
// horizontally neighboring chroma samples, in pixel_t units (planar: 1, YUY2: 4).
template<typename pixel_t, bool useHalocolor, bool fadeBackground, MaskMode maskMode>
static void RenderChroma(int bits_per_pixel, int color, int halocolor, BYTE* dstpU, BYTE* dstpV, int pitchUV, int step,
  const PreRendered& pre)
{
  static_assert(maskMode != MASK444, "not subsampled");
  constexpr MaskFilter HF = MaskHFilter<maskMode>;
  constexpr MaskFilter VF = MaskVFilter<maskMode>;
  constexpr int HS = MaskHSubsample<maskMode>;
  constexpr int VS = MaskVSubsample<maskMode>;
  constexpr int hfirst = MaskFilterFirst<HF>;
  constexpr int htaps = MaskFilterTaps<HF>;
  constexpr int vfirst = MaskFilterFirst<VF>;
  constexpr int vtaps = MaskFilterTaps<VF>;
  constexpr int log2total = MaskFilterLog2<HF> + MaskFilterLog2<VF>; // weight sum: 2 (box) .. 1024 (4:1:0 8x8)

  // some optimization hint
  if constexpr (sizeof(pixel_t) == 1)
    bits_per_pixel = 8;
  else if constexpr (sizeof(pixel_t) == 4)
    bits_per_pixel = 32;

  const pixel_t color_u = getHBDColor_UV<pixel_t>(getColorForPlane(PLANAR_U, color), bits_per_pixel);
  const pixel_t color_v = getHBDColor_UV<pixel_t>(getColorForPlane(PLANAR_V, color), bits_per_pixel);
  const pixel_t color_outline_u = getHBDColor_UV<pixel_t>(getColorForPlane(PLANAR_U, halocolor), bits_per_pixel);
  const pixel_t color_outline_v = getHBDColor_UV<pixel_t>(getColorForPlane(PLANAR_V, halocolor), bits_per_pixel);

  const int width = pre.frame_width();
  const int height = pre.frame_height();
  // Text rectangle in frame coordinates (inclusive). Frame position (x, y) inside it is
  // bit (pre.xstart + x - X0) of bitmap row (pre.ystart + y - Y0).
  const int X0 = pre.x;
  const int X1 = pre.x + pre.text_width - 1;
  const int Y0 = pre.y;
  const int Y1 = pre.y + (pre.yend - pre.ystart) - 1;
  if (X1 < X0 || Y1 < Y0)
    return;

  // The chroma samples c whose kernel window, luma [c * S + first, c * S + first + taps - 1], reaches the rectangle.
  // Near the frame edges the window is clamped into the frame, which cannot add more samples.
  const int cx_first = std::max(0, floor_div(X0 - hfirst - htaps + HS, HS)); // ceil((X0 - hfirst - htaps + 1) / HS)
  const int cx_last = std::min(width / HS - 1, floor_div(X1 - hfirst, HS));
  const int cy_first = std::max(0, floor_div(Y0 - vfirst - vtaps + VS, VS));
  const int cy_last = std::min(height / VS - 1, floor_div(Y1 - vfirst, VS));
  if (cx_last < cx_first || cy_last < cy_first)
    return;

  // The luma columns of all kernel windows of a chroma row: col[i] is frame column lx_first + i, its
  // vertically filtered font / halo count. Only the visible part of the text is read from the bitmap.
  const int lx_first = cx_first * HS + hfirst;
  const int ncols = (cx_last - cx_first) * HS + htaps;
  const int lx_last = lx_first + ncols - 1;
  const int bit_of_col0 = pre.xstart + (lx_first - X0);
  const int bit_first = pre.xstart + (std::max(X0, lx_first) - X0);
  const int bit_last = pre.xstart + (std::min({ X1, lx_last, width - 1 }) - X0);
  std::vector<int> col_font(ncols);
  std::vector<int> col_halo(useHalocolor ? ncols : 0);

  // clamped frame edges: the columns outside the frame repeat the edge column
  auto replicate_edges = [&](std::vector<int>& col) {
    for (int i = 0; i < -lx_first; i++)
      col[i] = col[-lx_first];
    for (int i = width - lx_first; i < ncols; i++)
      col[i] = col[width - 1 - lx_first];
  };

  // Horizontal rectangle weight of each chroma column, the same for all chroma rows, so it is computed only
  // once. The rectangle is separable: the weight of a chroma sample is the product of its H and V weights.
  // Only the fade needs it.
  std::vector<int> hrect(fadeBackground ? cx_last - cx_first + 1 : 0);
  if constexpr (fadeBackground) {
    for (int cx = cx_first; cx <= cx_last; cx++) {
      int sum = 0;
      for (int k = 0; k < htaps; k++) {
        const int x = std::clamp(cx * HS + hfirst + k, 0, width - 1);
        if (x >= X0 && x <= X1)
          sum += text_filter_weights[HF][k];
      }
      hrect[cx - cx_first] = sum;
    }
  }

  for (int cy = cy_first; cy <= cy_last; cy++) {
    // vertical pass: weighted font and halo counts of each luma column
    std::fill(col_font.begin(), col_font.end(), 0);
    if constexpr (useHalocolor)
      std::fill(col_halo.begin(), col_halo.end(), 0);
    bool any = false;
    [[maybe_unused]] int vrect = 0;
    for (int j = 0; j < vtaps; j++) {
      const int y = std::clamp(cy * VS + vfirst + j, 0, height - 1);
      if (y < Y0 || y > Y1)
        continue; // outside the rectangle
      const int r = pre.ystart + (y - Y0);
      const int weight = text_filter_weights[VF][j];
      vrect += weight;
      any |= add_set_bits(pre.stringbitmap[r].data(), bit_first, bit_last, weight, col_font.data(), bit_of_col0);
      if constexpr (useHalocolor)
        any |= add_set_bits(pre.stringbitmap_outline[r].data(), bit_first, bit_last, weight, col_halo.data(), bit_of_col0);
    }
    if constexpr (!fadeBackground) {
      if (!any)
        continue; // only background in this chroma row: no change, e.g. the blank rows of the font
    }
    if (any) { // all counts are zero otherwise, nothing to replicate
      replicate_edges(col_font);
      if constexpr (useHalocolor)
        replicate_edges(col_halo);
    }

    // horizontal pass, per chroma sample
    pixel_t* dstU = reinterpret_cast<pixel_t*>(dstpU + (size_t)cy * pitchUV);
    pixel_t* dstV = reinterpret_cast<pixel_t*>(dstpV + (size_t)cy * pitchUV);
    for (int cx = cx_first; cx <= cx_last; cx++) {
      const int i = (cx - cx_first) * HS;
      const int font = text_filter_sum<HF>(&col_font[i]);
      int halo = 0;
      if constexpr (useHalocolor)
        halo = text_filter_sum<HF>(&col_halo[i]); // halo bits exclude the font bits
      int rect = 0;
      if constexpr (fadeBackground)
        rect = hrect[cx - cx_first] * vrect;
      LightOneUVPixel<pixel_t, fadeBackground, log2total>(dstU[cx * step], dstV[cx * step],
        color_u, color_v, color_outline_u, color_outline_v, font, halo, rect, bits_per_pixel);
    }
  }
}

// Runtime chroma kernel (MaskMode) -> RenderChroma instance.
template<typename pixel_t, bool fadeBackground>
static void RenderChromaForMode(MaskMode chromaMode, bool useHalocolor, int bits_per_pixel, int color, int halocolor,
  BYTE* dstpU, BYTE* dstpV, int pitchUV, int step, const PreRendered& pre)
{
  auto render = [&](auto modeTag) {
    constexpr MaskMode mm = decltype(modeTag)::value;
    if (useHalocolor)
      RenderChroma<pixel_t, true, fadeBackground, mm>(bits_per_pixel, color, halocolor, dstpU, dstpV, pitchUV, step, pre);
    else
      RenderChroma<pixel_t, false, fadeBackground, mm>(bits_per_pixel, color, halocolor, dstpU, dstpV, pitchUV, step, pre);
  };
  switch (chromaMode) {
  case MASK410: render(std::integral_constant<MaskMode, MASK410>{}); break;
  case MASK410_MPEG2: render(std::integral_constant<MaskMode, MASK410_MPEG2>{}); break;
  case MASK410_TOPLEFT: render(std::integral_constant<MaskMode, MASK410_TOPLEFT>{}); break;
  case MASK440: render(std::integral_constant<MaskMode, MASK440>{}); break;
  case MASK440_TOPLEFT: render(std::integral_constant<MaskMode, MASK440_TOPLEFT>{}); break;
  case MASK411: render(std::integral_constant<MaskMode, MASK411>{}); break;
  case MASK411_TOPLEFT: render(std::integral_constant<MaskMode, MASK411_TOPLEFT>{}); break;
  case MASK420: render(std::integral_constant<MaskMode, MASK420>{}); break;
  case MASK420_MPEG2: render(std::integral_constant<MaskMode, MASK420_MPEG2>{}); break;
  case MASK420_TOPLEFT: render(std::integral_constant<MaskMode, MASK420_TOPLEFT>{}); break;
  case MASK422: render(std::integral_constant<MaskMode, MASK422>{}); break;
  case MASK422_MPEG2: render(std::integral_constant<MaskMode, MASK422_MPEG2>{}); break;
  default: assert(0); break; // MASK444: not subsampled, no chroma pass
  }
}

template<typename pixel_t, bool fadeBackground, bool isRGB>
void do_DrawStringPlanar(
  const int width, const int height, BYTE** dstps, int* pitches, const int logXRatioUV, const int logYRatioUV, const int planeCount,
  int bits_per_pixel,
  const BitmapFont* bmfont, int x, int y, std::vector<int>& s, int color, int halocolor, int align, bool useHalocolor, MaskMode chromaMode)
{
  // some optimization hint
  if constexpr (sizeof(pixel_t) == 1)
    bits_per_pixel = 8;
  else if constexpr (sizeof(pixel_t) == 4)
    bits_per_pixel = 32;

  PreRendered pre(bmfont->font_bitmaps.data(), bmfont->fontline_bytes, width, height, x, y, s, bmfont->bbx_array, align, useHalocolor,
    bmfont->global_bbx.width, bmfont->global_bbx.height);

  if (pre.len <= 0)
    return;

  const bool is444 = !isRGB && (planeCount >= 3) && (logXRatioUV == 0) && (logYRatioUV == 0);

  if (useHalocolor)
    Render1by1Planes<pixel_t, true, fadeBackground, isRGB>(bits_per_pixel, color, halocolor, pitches, dstps, pre, planeCount, is444);
  else
    Render1by1Planes<pixel_t, false, fadeBackground, isRGB>(bits_per_pixel, color, halocolor, pitches, dstps, pre, planeCount, is444);

  if constexpr (isRGB)
    return;

  if (is444)
    return;

  if (planeCount < 3)
    return; // Y

  // subsampled chroma: the mask kernel of the format and chroma placement (chromaMode, see DrawString_internal)
  RenderChromaForMode<pixel_t, fadeBackground>(chromaMode, useHalocolor, bits_per_pixel, color, halocolor,
    dstps[1], dstps[2], pitches[1], 1, pre);
}

template<bool useHalocolor, bool fadeBackground>
void RenderYUY2(int color, int halocolor, int pitch, BYTE* _dstp, PreRendered& pre, MaskMode chromaMode)
{
  BYTE* dstp = _dstp + pre.x * 2 + pre.y * pitch;

  typedef uint8_t pixel_t;

  pixel_t val_color = getColorForPlane(PLANAR_Y, color);
  pixel_t val_color_outline = getColorForPlane(PLANAR_Y, halocolor);

  // first Y, like in 4:4:4
  for (int ty = pre.ystart; ty < pre.yend; ty++) {
    BYTE* dp = dstp;
    uint8_t* fontline_ptr = pre.stringbitmap[ty].data();
    uint8_t* fontoutline_ptr = nullptr;
    if constexpr (useHalocolor)
      fontoutline_ptr = pre.stringbitmap_outline[ty].data();

    int j = 0;
    const int shifted_xstart = pre.xstart;
    for (int tx = shifted_xstart; tx < shifted_xstart + pre.text_width; tx++) {
      if constexpr (!fadeBackground) {
        if (is_empty_bitmap_byte(fontline_ptr, fontoutline_ptr, tx)) {
          tx += 7; // +1 in the loop header
          j += 16; // 8 YUYV luma positions
          continue;
        }
      }
      const bool lightIt = 0 != get_bit(fontline_ptr, tx);
      LightOnePixel<uint8_t, fadeBackground, false>(lightIt, dp, j, val_color, 8);
      if constexpr (useHalocolor) {
        if (!lightIt)
        {
          const bool lightIt_outline = 0 != get_bit(fontoutline_ptr, tx);
          LightOnePixel<uint8_t, fadeBackground, false>(lightIt_outline, dp, j, val_color_outline, 8);
        }
      }
      j += 2; // next Y
    }
    dstp += pitch;
  }

  // then chroma, like planar 4:2:2: U and V are the bytes 1 and 3 of each YUYV group
  RenderChromaForMode<uint8_t, fadeBackground>(chromaMode, useHalocolor, 8, color, halocolor,
    _dstp + 1, _dstp + 3, pitch, 4, pre);
}

template<bool fadeBackground>
static void do_DrawStringYUY2(
  const int width, const int height, BYTE* _dstp, int pitch, const BitmapFont* bmfont, int x, int y, std::vector<int>& s,
  int color, int halocolor, int align, bool useHalocolor, MaskMode chromaMode)
{
  PreRendered pre(bmfont->font_bitmaps.data(), bmfont->fontline_bytes, width, height, x, y, s, bmfont->bbx_array, align, useHalocolor,
    bmfont->global_bbx.width, bmfont->global_bbx.height);

  if (pre.len <= 0)
    return;

  if (useHalocolor)
    RenderYUY2<true, fadeBackground>(color, halocolor, pitch, _dstp, pre, chromaMode);
  else
    RenderYUY2<false, fadeBackground>(color, halocolor, pitch, _dstp, pre, chromaMode);
}

template<typename pixel_t, bool useHalocolor, bool fadeBackground, int rgbstep>
static void RenderPackedRGB(int color, int halocolor, BYTE* _dstp, int pitch, int height, PreRendered& pre)
{
  // packed: only 8 and 16 bits
  int bits_per_pixel = 0;
  if constexpr (sizeof(pixel_t) == 1)
    bits_per_pixel = 8;
  else if constexpr (sizeof(pixel_t) == 2)
    bits_per_pixel = 16;

  const int val_color_R = getHBDColor_RGB<pixel_t>(getColorForPlane(PLANAR_R, color), bits_per_pixel);
  const int val_color_R_outline = getHBDColor_RGB<pixel_t>(getColorForPlane(PLANAR_R, halocolor), bits_per_pixel);
  const int val_color_G = getHBDColor_RGB<pixel_t>(getColorForPlane(PLANAR_G, color), bits_per_pixel);
  const int val_color_G_outline = getHBDColor_RGB<pixel_t>(getColorForPlane(PLANAR_G, halocolor), bits_per_pixel);
  const int val_color_B = getHBDColor_RGB<pixel_t>(getColorForPlane(PLANAR_B, color), bits_per_pixel);
  const int val_color_B_outline = getHBDColor_RGB<pixel_t>(getColorForPlane(PLANAR_B, halocolor), bits_per_pixel);

  // upside down
  BYTE* dstp = _dstp + pre.x * rgbstep + (height - 1 - pre.y) * pitch;

  // Start rendering
  for (int ty = pre.ystart; ty < pre.yend; ty++) {
    uint8_t* dp = dstp;
    uint8_t* fontline_ptr = pre.stringbitmap[ty].data();
    uint8_t* fontoutline_ptr = nullptr;
    if constexpr(useHalocolor)
      fontoutline_ptr = pre.stringbitmap_outline[ty].data();

    const int shifted_xstart = pre.xstart;
    for (int tx = shifted_xstart; tx < shifted_xstart + pre.text_width; tx++)
    {
      if constexpr (!fadeBackground) {
        if (is_empty_bitmap_byte(fontline_ptr, fontoutline_ptr, tx)) {
          tx += 7; // +1 in the loop header
          dp += rgbstep * 8;
          continue;
        }
      }
      const bool lightIt = 0 != get_bit(fontline_ptr, tx);
      LightOnePixelPackedRGB<pixel_t, fadeBackground>(lightIt, dp, val_color_R, val_color_G, val_color_B);
      if constexpr(useHalocolor) {
        if (!lightIt) {
          const bool lightIt_outline = 0 != get_bit(fontoutline_ptr, tx);
          LightOnePixelPackedRGB<pixel_t, fadeBackground>(lightIt_outline, dp, val_color_R_outline, val_color_G_outline, val_color_B_outline);
        }
      }
      dp += rgbstep;
    }
    dstp -= pitch;
  }
}

template<typename pixel_t, int rgbstep, bool fadeBackground>
static void do_DrawStringPackedRGB(
  const int width, const int height, BYTE* _dstp, int pitch,
  const BitmapFont* bmfont, int x, int y, std::vector<int>& s, int color, int halocolor, int align, bool useHalocolor)
{
  PreRendered pre(bmfont->font_bitmaps.data(), bmfont->fontline_bytes, width, height, x, y, s, bmfont->bbx_array, align, useHalocolor,
    bmfont->global_bbx.width, bmfont->global_bbx.height);

  if (pre.len <= 0)
    return;

  if (useHalocolor)
    RenderPackedRGB<pixel_t, true, fadeBackground, rgbstep>(color, halocolor, _dstp, pitch, height, pre);
  else
    RenderPackedRGB<pixel_t, false, fadeBackground, rgbstep>(color, halocolor, _dstp, pitch, height, pre);
}


static bool strequals_i(const std::string& a, const std::string& b)
{
  return std::equal(a.begin(), a.end(),
    b.begin(), b.end(),
    [](char a, char b) {
      return asciiToLower(a) == asciiToLower(b);
    });
}

// in fixedfonts.cpp
extern const uint16_t *font_bitmaps[];
extern const uint16_t *font_codepoints[];
extern const FixedFont_info_t *font_infos[];

std::unique_ptr<BitmapFont> CreateBitmapFont(int size, const char *name, bool bold, bool debugSave) {

  BitmapFont* current_font = nullptr;

  // check internal embedded fonts
  bool found = false;

  // find font in internal list
  for (int i = 0; i < PREDEFINED_FONT_COUNT; i++)
  {
    const FixedFont_info_t* fi = font_infos[i];
    if (fi->height == size && fi->bold == bold && strequals_i(fi->fontname, name)) {
      BBX global_bbx = BBX{ fi->width, fi->height, fi->offset_x, fi->offset_y }; // BBX from FONTBOUNDINGBOX
      // max WIDTH, max HEIGHT, offset_x, offset_y
      current_font = new BitmapFont(
        fi->charcount,
        font_bitmaps[i], // internal one
        nullptr, // not a byte array from external BDF
        sizeof(uint16_t), // sizeof(*font_bitmaps) 2: uint16_t
        font_codepoints[i], // internal one
        nullptr, // not a codepoint array from external BDF
        nullptr, // Individual BBX for each character. For our internal characters it comes from global bbx
        global_bbx,
        fi->fontname,
        "",
        fi->bold,
        false);
      found = true;
      break;
    }
  }
  // pass #2 when size does not match exactly, find nearest, but still smaller font.
  if (!found) {
    // find font i internal list
    int last_good_size = 0;
    int found_index = -1;
    for (int i = 0; i < PREDEFINED_FONT_COUNT; i++)
    {
      const FixedFont_info_t* fi = font_infos[i];
      if (fi->bold == bold && strequals_i(fi->fontname, name)) {
        if (last_good_size == 0) {
          found_index = i;
          last_good_size = fi->height;
        }
        else if (std::abs(fi->height - size) < std::abs(last_good_size - size) && fi->height <= size) {
          // has better size match and is not larger
          found_index = i;
          last_good_size = fi->height;
        }
      }
    }
    if (found_index >= 0) {
      const FixedFont_info_t* fi = font_infos[found_index];
      BBX global_bbx = BBX{ fi->width, fi->height, fi->offset_x, fi->offset_y }; // BBX from FONTBOUNDINGBOX
      current_font = new BitmapFont(
        fi->charcount,
        font_bitmaps[found_index], // internal one
        nullptr, // not a byte array from external BDF
        sizeof(uint16_t), // sizeof(*font_bitmaps) 2: uint16_t
        font_codepoints[found_index], // internal one
        nullptr, // not a codepoint array from external BDF
        nullptr,  // Individual BBX for each character. For our internal characters it comes from global bbx
        global_bbx,
        fi->fontname,
        "",
        fi->bold,
        false);
      found = true;
    }
  }

  if (!found) {
    // fixme: make cache
    BdfFont bdf;
    bdf = LoadBMF(name, bold);
    if (bdf.codepoints_array.size() == 0)
      return nullptr;

    current_font = new BitmapFont(
      bdf.font_info.chars,
      nullptr, // not an internal one
      bdf.font_bitmaps.data(), // a byte array from external BDF
      bdf.font_info.fontline_bytes,
      nullptr, // not an internal one
      bdf.codepoints_array.data(),
      &bdf.bbx_array, // Individual BBX for each character, nullptr if not available for our internal fonts
      bdf.font_info.bbx, // from FONTBOUNDINGBOX, max WIDTH, HEIGHT, offset_x, offset_y
      bdf.font_info.font,
      bdf.font_filename,
      strequals_i(bdf.font_properties.Weight_name, "bold"),
      debugSave);
  }
  return std::unique_ptr<BitmapFont>(current_font);
}

std::shared_ptr<const BitmapFont> GetBitmapFont(int size, const char* name, bool bold, bool debugSave, IScriptEnvironment* env)
{
  if (env == nullptr || debugSave)
    return CreateBitmapFont(size, name, bold, debugSave);
  return GetAndRevealCamouflagedEnv(env)->GetCachedBitmapFont(size, name, bold);
}

static void DrawString_internal(const BitmapFont* current_font, const VideoInfo& vi, PVideoFrame& dst, int x, int y, std::string& s_utf8,
  int color, int halocolor, bool useHalocolor, int align, bool fadeBackground, int chromalocation)
{
  //static BitmapFont_10_20 infoFont1020; // constructor runs once, single instance

  // map an utf8 string to a sequence of character map indexes
  auto s_remapped = current_font->remap(s_utf8); // array of font table indexes

  //SaveBitmapSource(); // debug to generate source from original table

  const bool isRGB = vi.IsRGB();
  const int planes_y[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
  const int planes_r[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
  const int* planes = isRGB ? planes_r : planes_y;

  int logXRatioUV = 0;
  int logYRatioUV = 0;
  if (!vi.IsY() && !vi.IsYA() && !vi.IsRGB()) {
    logXRatioUV = vi.IsYUY2() ? 1 : vi.GetPlaneWidthSubsampling(PLANAR_U);
    logYRatioUV = vi.IsYUY2() ? 0 : vi.GetPlaneHeightSubsampling(PLANAR_U);
  }
  // Y+A: treat it like Y (planecount = 1), alpha is ignored
  const int planecount = vi.IsYUY2() ? 1 : (vi.IsYA() ? 1 : std::min(vi.NumComponents(), 3));
  BYTE* dstps[3] = { nullptr };
  int pitches[3] = { 0 };

  for (int i = 0; i < planecount; i++)
  {
    int plane = planes[i];
    dstps[i] = dst->GetWritePtr(plane);
    pitches[i] = dst->GetPitch(plane);
  }

  const int width = vi.width;
  const int height = vi.height;

  const int bits_per_pixel = vi.BitsPerComponent();

  // Chroma kernel: the mask kernel of Overlay/Layer/Subtitle for the format and chroma placement
  // (chromaLocationToMaskPlacement and resolveChromaMaskMode, blend_common.h): box (centred 4:2:x and 4:4:0),
  // [1 2 1] (co-sited 4:2:x and 4:4:0), 4:1 triangle (centred 4:1:x) or tent (co-sited 4:1:x).
  const bool subsampledYUV = (vi.IsYUV() || vi.IsYUVA()) && !vi.IsY() && !vi.IsYA() && !vi.Is444();
  const MaskMode chromaMode = subsampledYUV ? resolveChromaMaskMode(chromaLocationToMaskPlacement(chromalocation, vi), vi) : MASK444;

  // YUY2
  if (vi.IsYUY2()) {
    if (fadeBackground)
      do_DrawStringYUY2<true>(width, height, dstps[0], pitches[0], current_font, x, y, s_remapped, color, halocolor, align, useHalocolor, chromaMode);
    else
      do_DrawStringYUY2<false>(width, height, dstps[0], pitches[0], current_font, x, y, s_remapped, color, halocolor, align, useHalocolor, chromaMode);
    return;
  }

  // Packed RGB24/32/48/64
  if (isRGB && !vi.IsPlanar()) {
    if (fadeBackground) {
      if (vi.IsRGB24())
        do_DrawStringPackedRGB<uint8_t, 3, true>(width, height, dstps[0], pitches[0], current_font, x, y, s_remapped, color, halocolor, align, useHalocolor);
      else if (vi.IsRGB32())
        do_DrawStringPackedRGB<uint8_t, 4, true>(width, height, dstps[0], pitches[0], current_font, x, y, s_remapped, color, halocolor, align, useHalocolor);
      else if (vi.IsRGB48())
        do_DrawStringPackedRGB<uint16_t, 6, true>(width, height, dstps[0], pitches[0], current_font, x, y, s_remapped, color, halocolor, align, useHalocolor);
      else if (vi.IsRGB64())
        do_DrawStringPackedRGB<uint16_t, 8, true>(width, height, dstps[0], pitches[0], current_font, x, y, s_remapped, color, halocolor, align, useHalocolor);
    }
    else {
      if (vi.IsRGB24())
        do_DrawStringPackedRGB<uint8_t, 3, false>(width, height, dstps[0], pitches[0], current_font, x, y, s_remapped, color, halocolor, align, useHalocolor);
      else if (vi.IsRGB32())
        do_DrawStringPackedRGB<uint8_t, 4, false>(width, height, dstps[0], pitches[0], current_font, x, y, s_remapped, color, halocolor, align, useHalocolor);
      else if (vi.IsRGB48())
        do_DrawStringPackedRGB<uint16_t, 6, false>(width, height, dstps[0], pitches[0], current_font, x, y, s_remapped, color, halocolor, align, useHalocolor);
      else if (vi.IsRGB64())
        do_DrawStringPackedRGB<uint16_t, 8, false>(width, height, dstps[0], pitches[0], current_font, x, y, s_remapped, color, halocolor, align, useHalocolor);
    }
    return;
  }

  // planar and Y
  if (fadeBackground) {
    if (isRGB) {
      switch (bits_per_pixel)
      {
      case 8:
        do_DrawStringPlanar<uint8_t, true, true>(width, height, dstps, pitches, logXRatioUV, logYRatioUV, planecount, bits_per_pixel, current_font, x, y, s_remapped, color, halocolor, align, useHalocolor, chromaMode);
        break;
      case 10:
      case 12:
      case 14:
      case 16:
        do_DrawStringPlanar<uint16_t, true, true>(width, height, dstps, pitches, logXRatioUV, logYRatioUV, planecount, bits_per_pixel, current_font, x, y, s_remapped, color, halocolor, align, useHalocolor, chromaMode);
        break;
      case 32:
        do_DrawStringPlanar<float, true, true>(width, height, dstps, pitches, logXRatioUV, logYRatioUV, planecount, bits_per_pixel, current_font, x, y, s_remapped, color, halocolor, align, useHalocolor, chromaMode);
        break;
      }
    }
    else {
      switch (bits_per_pixel)
      {
      case 8:
        do_DrawStringPlanar<uint8_t, true, false>(width, height, dstps, pitches, logXRatioUV, logYRatioUV, planecount, bits_per_pixel, current_font, x, y, s_remapped, color, halocolor, align, useHalocolor, chromaMode);
        break;
      case 10:
      case 12:
      case 14:
      case 16:
        do_DrawStringPlanar<uint16_t, true, false>(width, height, dstps, pitches, logXRatioUV, logYRatioUV, planecount, bits_per_pixel, current_font, x, y, s_remapped, color, halocolor, align, useHalocolor, chromaMode);
        break;
      case 32:
        do_DrawStringPlanar<float, true, false>(width, height, dstps, pitches, logXRatioUV, logYRatioUV, planecount, bits_per_pixel, current_font, x, y, s_remapped, color, halocolor, align, useHalocolor, chromaMode);
        break;
      }
    }
  }
  else {
    if (isRGB) {
      switch (bits_per_pixel)
      {
      case 8:
        do_DrawStringPlanar<uint8_t, false, true>(width, height, dstps, pitches, logXRatioUV, logYRatioUV, planecount, bits_per_pixel, current_font, x, y, s_remapped, color, halocolor, align, useHalocolor, chromaMode);
        break;
      case 10:
      case 12:
      case 14:
      case 16:
        do_DrawStringPlanar<uint16_t, false, true>(width, height, dstps, pitches, logXRatioUV, logYRatioUV, planecount, bits_per_pixel, current_font, x, y, s_remapped, color, halocolor, align, useHalocolor, chromaMode);
        break;
      case 32:
        do_DrawStringPlanar<float, false, true>(width, height, dstps, pitches, logXRatioUV, logYRatioUV, planecount, bits_per_pixel, current_font, x, y, s_remapped, color, halocolor, align, useHalocolor, chromaMode);
        break;
      }
    }
    else {
      switch (bits_per_pixel)
      {
      case 8:
        do_DrawStringPlanar<uint8_t, false, false>(width, height, dstps, pitches, logXRatioUV, logYRatioUV, planecount, bits_per_pixel, current_font, x, y, s_remapped, color, halocolor, align, useHalocolor, chromaMode);
        break;
      case 10:
      case 12:
      case 14:
      case 16:
        do_DrawStringPlanar<uint16_t, false, false>(width, height, dstps, pitches, logXRatioUV, logYRatioUV, planecount, bits_per_pixel, current_font, x, y, s_remapped, color, halocolor, align, useHalocolor, chromaMode);
        break;
      case 32:
        do_DrawStringPlanar<float, false, false>(width, height, dstps, pitches, logXRatioUV, logYRatioUV, planecount, bits_per_pixel, current_font, x, y, s_remapped, color, halocolor, align, useHalocolor, chromaMode);
        break;
      }
    }
  }
}

void SimpleTextOutW(const BitmapFont *current_font, const VideoInfo& vi, PVideoFrame& frame, int real_x, int real_y, std::string& text_utf8,
  bool fadeBackground, int textcolor, int halocolor, bool useHaloColor, int align, int chromalocation)
{
  DrawString_internal(current_font, vi, frame, real_x, real_y, text_utf8, textcolor, halocolor, useHaloColor, align, fadeBackground, chromalocation); // fully transparent background
}

// additional parameter: lsp line spacing
void SimpleTextOutW_multi(const BitmapFont *current_font, const VideoInfo& vi, PVideoFrame& frame, int real_x, int real_y, std::string& text_utf8,
  bool fadeBackground, int textcolor, int halocolor, bool useHaloColor,
  int align, int lsp, int chromalocation)
{

  // make list governed by LF separator
  std::string temp;
  std::vector<std::string> parts;
  std::stringstream ss(text_utf8);
  while (std::getline(ss, temp, '\n'))
    parts.push_back(temp); // still in utf8
  // It doesn't result in a new line if the last character is \n and is followed by nothing.
  // "Line1\nLine2" is the same as "Line1\nLine2\n"
  // Like in SubTitle
  /*
  if(!text.empty())
  {
    if( *text.rbegin() == '\n')
      parts.push_back("");
  }
  */
  const int fontSize = current_font->global_bbx.height;

  // when multiline, bottom and vertically centered cases affect starting y
  // lsp units are in 1/8 pixels by definition
  int al = alignToBitmask(align);
  if (al & ATA_BOTTOM)
    real_y -= (int)((fontSize + lsp / 8.0) * ((int)parts.size() - 1) + 0.5);
  else if (al & ATA_BASELINE)
    real_y -= (int)((fontSize + lsp / 8.0) * (int)(parts.size() - 1) / 2.0 + 0.5);

  const int orig_real_y = real_y;
  int linecount = 0;
  for (auto s_utf8 : parts) {
    real_y = orig_real_y + fontSize * linecount + (int)(lsp / 8.0 * linecount + 0.5);
    SimpleTextOutW(current_font, vi, frame, real_x, real_y, s_utf8, fadeBackground, textcolor, halocolor, useHaloColor, align, chromalocation);
    linecount++;
  }
}

// Old legacy info.h functions, but with utf8 mode
// w/o outline, originally with ASCII input, background fading
// Despite name Planar, it works for all formats
void DrawStringPlanar(VideoInfo& vi, PVideoFrame& dst, int x, int y, const char* s, IScriptEnvironment* env)
{
  int color;
  if (vi.IsRGB())
    color = (250 << 16) + (250 << 8) + (250);
  else
    color = (230 << 16) + (128 << 8) + (128);

  // fadeBackground = true: background letter area is faded instead not being untouched.

  std::string s_utf8 = charToUtf8(s, false);

  int halocolor = 0;

  std::shared_ptr<const BitmapFont> current_font = GetBitmapFont(20, "info_h", false, false, env); // 10x20

  if (current_font == nullptr)
    return;

  DrawString_internal(current_font.get(), vi, dst, x, y, s_utf8,
    color,
    halocolor,
    false, // don't use halocolor
    0 /* no align */,
    true, // fadeBackGround
    ChromaLocation_e::AVS_CHROMA_LEFT
  );

}

void DrawStringYUY2(VideoInfo& vi, PVideoFrame& dst, int x, int y, const char* s, IScriptEnvironment* env)
{
  DrawStringPlanar(vi, dst, x, y, s, env);
}

// legacy function w/o outline, originally with ASCII input, background fading
void DrawStringRGB24(VideoInfo &vi, PVideoFrame& dst, int x, int y, const char* s, IScriptEnvironment* env)
{
  DrawStringPlanar(vi, dst, x, y, s, env);
}

// legacy function w/o outline, originally with ASCII input, background fading
void DrawStringRGB32(VideoInfo& vi, PVideoFrame& dst, int x, int y, const char* s, IScriptEnvironment* env)
{
  DrawStringPlanar(vi, dst, x, y, s, env);
}
