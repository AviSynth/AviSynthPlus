// Avisynth v2.5.  Copyright 2002-2009 Ben Rudiak-Gould et al.
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
#ifdef AVS_WINDOWS
#include <avs/win.h>
#else
#include <avs/posix.h>
#endif 

#include "convert_matrix.h"
#include "convert_helper.h"

#include <unordered_map>
#include <iostream>
#include <vector>
#include <sstream>

const std::vector<std::pair<const char*, ColorRange_e>> g_range_table{
    { "limited", AVS_RANGE_LIMITED },
    { "full",    AVS_RANGE_FULL },
    { "l",       AVS_RANGE_LIMITED },
    { "f",       AVS_RANGE_FULL },
};

const std::vector<std::pair<const char*, ChromaLocation_e>> g_chromaloc_table{
    { "left",        AVS_CHROMA_LEFT },
    { "center",      AVS_CHROMA_CENTER },
    { "top_left",    AVS_CHROMA_TOP_LEFT },
    { "top",         AVS_CHROMA_TOP }, // not used in Avisynth
    { "bottom_left", AVS_CHROMA_BOTTOM_LEFT }, // not used in Avisynth
    { "bottom",      AVS_CHROMA_BOTTOM }, // not used in Avisynth
    { "dv",          AVS_CHROMA_DV }, // Special to Avisynth
    // compatibility
    { "mpeg1",       AVS_CHROMA_CENTER },
    { "mpeg2",       AVS_CHROMA_LEFT },
    { "jpeg",        AVS_CHROMA_CENTER },
};

// unlike Avisynth conventions, the strings do not contain hints on full or limited range
// e.g. PC.709 or Rec709
const std::vector<std::pair<const char*, Matrix_e>> g_matrix_table{
    { "rgb",         AVS_MATRIX_RGB },
    { "709",         AVS_MATRIX_BT709 },
    { "unspec",      AVS_MATRIX_UNSPECIFIED },
    { "170m",        AVS_MATRIX_ST170_M },
    { "240m",        AVS_MATRIX_ST240_M },
    { "470bg",       AVS_MATRIX_BT470_BG },
    { "fcc",         AVS_MATRIX_BT470_M },
    { "470m",        AVS_MATRIX_BT470_M }, // as of 20211111 name is an add-on in Avisynth+
    { "ycgco",       AVS_MATRIX_YCGCO },
    { "2020ncl",     AVS_MATRIX_BT2020_NCL },
    { "2020cl",      AVS_MATRIX_BT2020_CL },
    { "chromacl",    AVS_MATRIX_CHROMATICITY_DERIVED_CL },
    { "chromancl",   AVS_MATRIX_CHROMATICITY_DERIVED_NCL },
    { "ictcp",       AVS_MATRIX_ICTCP },
    // compatibility
    { "601",         AVS_MATRIX_BT470_BG },
    { "2020",        AVS_MATRIX_BT2020_NCL },
};

// old Avisynth "matrix" parameter strings
const std::vector<std::pair<const char*, Old_Avs_Matrix_e>> g_old_avs_matrix_table{
    { "rec601",      AVS_OLD_MATRIX_Rec601 },
    { "rec709",      AVS_OLD_MATRIX_Rec709 },
    { "pc.601",      AVS_OLD_MATRIX_PC_601 },
    { "pc.709",      AVS_OLD_MATRIX_PC_709 },
    { "pc601",       AVS_OLD_MATRIX_PC_601 },
    { "pc709",       AVS_OLD_MATRIX_PC_709 },
    { "average",     AVS_OLD_MATRIX_AVERAGE },
    { "rec2020",     AVS_OLD_MATRIX_Rec2020 },
    { "pc.2020",     AVS_OLD_MATRIX_PC_2020 },
    { "pc2020",      AVS_OLD_MATRIX_PC_2020 }
};

// not used in Avisynth (yet)
const std::vector<std::pair<const char*, Transfer_e>> g_transfer_table{
    { "709",     AVS_TRANSFER_BT709 },
    { "unspec",  AVS_TRANSFER_UNSPECIFIED },
    { "601",     AVS_TRANSFER_BT601 },
    { "linear",  AVS_TRANSFER_LINEAR },
    { "2020_10", AVS_TRANSFER_BT2020_10 },
    { "2020_12", AVS_TRANSFER_BT2020_12 },
    { "240m",    AVS_TRANSFER_ST240_M },
    { "470m",    AVS_TRANSFER_BT470_M },
    { "470bg",   AVS_TRANSFER_BT470_BG },
    { "log100",  AVS_TRANSFER_LOG_100 },
    { "log316",  AVS_TRANSFER_LOG_316 },
    { "st2084",  AVS_TRANSFER_ST2084 },
    { "std-b67", AVS_TRANSFER_ARIB_B67 },
    { "srgb",    AVS_TRANSFER_IEC_61966_2_1 },
    { "xvycc",   AVS_TRANSFER_IEC_61966_2_4 },
};

// not used in Avisynth (yet)
const std::vector<std::pair<const char*, Primaries_e>> g_primaries_table{
    { "709",       AVS_PRIMARIES_BT709 },
    { "unspec",    AVS_PRIMARIES_UNSPECIFIED },
    { "170m",      AVS_PRIMARIES_ST170_M },
    { "240m",      AVS_PRIMARIES_ST240_M },
    { "470m",      AVS_PRIMARIES_BT470_M },
    { "470bg",     AVS_PRIMARIES_BT470_BG },
    { "film",      AVS_PRIMARIES_FILM },
    { "2020",      AVS_PRIMARIES_BT2020 },
    { "st428",     AVS_PRIMARIES_ST428 },
    { "xyz",       AVS_PRIMARIES_ST428 },
    { "st431-2",   AVS_PRIMARIES_ST431_2 },
    { "st432-1",   AVS_PRIMARIES_ST432_1 },
    { "ebu3213-e", AVS_PRIMARIES_EBU3213_E },
};

// -1: null input or empty string
// -2: not found
// table index otherwise
template<class T>
auto lookup_table(const std::vector<std::pair<const char *, T>>& table, const char* key) {
  if (key == nullptr) return -1;
  if (*key == 0) return -1;
  auto it = std::find_if(table.begin(), table.end(),
    [&key](const auto& element) { return !lstrcmpi(element.first, key); });
  return it != std::end(table) ? it->second : -2;
}

bool getPrimaries(const char* primaries_name, IScriptEnvironment* env, int& _Primaries) {
  auto index = lookup_table<Primaries_e>(g_primaries_table, primaries_name);
  if (index >= 0) {
    _Primaries = index;
    return true;
  }

  if (index == -2) // not found
    env->ThrowError("Convert: Unknown Primaries");

  // empty
  return false;
}

static bool getMatrix(const char* matrix_name, IScriptEnvironment* env, int& _Matrix) {
  auto index = lookup_table<Matrix_e>(g_matrix_table, matrix_name);
  if (index >= 0) {
    _Matrix = index;
    return true;
  }

  if (index == -2) // not found
    env->ThrowError("Convert: Unknown matrix");

  // empty
  return false;
}

static bool getChromaLocation(const char* chromaloc_name, IScriptEnvironment* env, int& _ChromaLocation) {
  auto index = lookup_table<ChromaLocation_e>(g_chromaloc_table, chromaloc_name);
  if (index >= 0) {
    _ChromaLocation = index;
    return true;
  }

  if (index == -2) // not found
    env->ThrowError("Unknown chroma placement");

  // empty
  return false;
}

bool getColorRange(const char* color_range_name, IScriptEnvironment* env, int& _ColorRange) {
  auto color_range_enum = lookup_table<ColorRange_e>(g_range_table, color_range_name);
  if (color_range_enum >= 0) {
    _ColorRange = color_range_enum;
    return true;
  }

  if (color_range_enum == -2) // not found
    env->ThrowError("Convert: Unknown color range, must be 'auto', 'full', 'f', 'limited' or 'l'");
  
  // empty
  return false;

}

// Converts old-style Avisynth matrix to separated _Matrix and _ColorRange values
// if not found or empty, returns false, does not throw exception.
static bool getOldMatrix(const char* matrix_name, int &_Matrix, int &_ColorRange) {
  auto old_matrix_enum = lookup_table<Old_Avs_Matrix_e>(g_old_avs_matrix_table, matrix_name);
  if (old_matrix_enum < 0)
    return false; // not found or empty

  switch (old_matrix_enum) {
  case Old_Avs_Matrix_e::AVS_OLD_MATRIX_Rec601:
    _Matrix = Matrix_e::AVS_MATRIX_ST170_M;
    _ColorRange = ColorRange_e::AVS_RANGE_LIMITED;
    break;
  case Old_Avs_Matrix_e::AVS_OLD_MATRIX_PC_601:
    _Matrix = Matrix_e::AVS_MATRIX_ST170_M;
    _ColorRange = ColorRange_e::AVS_RANGE_FULL;
    break;
  case Old_Avs_Matrix_e::AVS_OLD_MATRIX_Rec709:
    _Matrix = Matrix_e::AVS_MATRIX_BT709;
    _ColorRange = ColorRange_e::AVS_RANGE_LIMITED;
    break;
  case Old_Avs_Matrix_e::AVS_OLD_MATRIX_PC_709:
    _Matrix = Matrix_e::AVS_MATRIX_BT709;
    _ColorRange = ColorRange_e::AVS_RANGE_FULL;
    break;
  case Old_Avs_Matrix_e::AVS_OLD_MATRIX_Rec2020:
    _Matrix = Matrix_e::AVS_MATRIX_BT2020_NCL;
    _ColorRange = ColorRange_e::AVS_RANGE_LIMITED;
    break;
  case Old_Avs_Matrix_e::AVS_OLD_MATRIX_PC_2020:
    _Matrix = Matrix_e::AVS_MATRIX_BT2020_NCL;
    _ColorRange = ColorRange_e::AVS_RANGE_FULL;
    break;
  case Old_Avs_Matrix_e::AVS_OLD_MATRIX_AVERAGE:
    _Matrix = Matrix_e::AVS_MATRIX_AVERAGE; // non-standard!
    _ColorRange = ColorRange_e::AVS_RANGE_FULL;
    break;
  }
  return true;
}


static bool is_paramstring_auto(const std::string &param) {
  return !lstrcmpi(param.c_str(), "auto"); // true is match
}

static bool is_paramstring_empty_or_auto(const std::string& param) {
  return param.empty() || !lstrcmpi(param.c_str(), "auto"); // true is match
}

static bool is_paramstring_empty_or_auto(const char* param) {
  if (!param)
    return true;
  return !lstrcmpi(param, "auto"); // true is match
}

// called from yuv <-> rgb and to_greyscale converters
void matrix_parse_merge_with_props_def(VideoInfo& vi, const char* matrix_name, const AVSMap* props, int& _Matrix, int& _ColorRange, int _Matrix_Default, int _ColorRange_Default, IScriptEnvironment* env) {

  // if once we'd like to use input colorrange when input is rgb (e.g. studio rgb of ColorBars is limited)
  int _ColorRange_In = vi.IsRGB() ? ColorRange_e::AVS_RANGE_FULL : ColorRange_e::AVS_RANGE_LIMITED;

  if (props) {
    // theoretically props==nullptr when input is RGB
    if (env->propNumElements(props, "_ColorRange") > 0) {
      _ColorRange_In = env->propGetInt(props, "_ColorRange", 0, nullptr); // fixme: range check
      if (!vi.IsRGB())
        _ColorRange_Default = _ColorRange_In;
    }
    if (!vi.IsRGB()) {
      if (env->propNumElements(props, "_Matrix") > 0) {
        auto tmp_matrix = env->propGetInt(props, "_Matrix", 0, nullptr); // fixme: range check
        if (tmp_matrix != Matrix_e::AVS_MATRIX_UNSPECIFIED)
          _Matrix_Default = tmp_matrix;
      }
    }
  }

  // valid values:
  // old matrix name, optionally followed by 'auto' e.g. Rec709
  // matrix name, optionally followed by color range e.g. 709:limited 2020:f 601:full
  std::vector<std::string> splits;
  std::string split;

  if (matrix_name) {
    std::istringstream ss(matrix_name);
    while (std::getline(ss, split, ':'))
      splits.push_back(split);

    if (splits.size() > 2)
      env->ThrowError("Invalid matrix specifier, too many parts");
  }

  std::string s_matrix_name = splits.size() > 0 ? splits[0] : "";
  std::string s_color_range_name = splits.size() > 1 ? splits[1] : "";

  if (getOldMatrix(s_matrix_name.c_str(), _Matrix, _ColorRange)) {
    if (!s_color_range_name.empty() && !is_paramstring_auto(s_color_range_name))
      env->ThrowError("Error: this 'old-style' matrix string can only be followed by 'auto' color range");
    return;
  }

  if (is_paramstring_empty_or_auto(s_matrix_name) || !getMatrix(s_matrix_name.c_str(), env, _Matrix)) {
    _Matrix = _Matrix_Default;
  }
  if (_Matrix == Matrix_e::AVS_MATRIX_UNSPECIFIED) {
    _Matrix = _Matrix_Default;
  }
  if (is_paramstring_empty_or_auto(s_color_range_name) || !getColorRange(s_color_range_name.c_str(), env, _ColorRange)) {
    _ColorRange = _ColorRange_Default;
  }
}

void matrix_parse_merge_with_props(VideoInfo& vi, const char* matrix_name, const AVSMap* props, int& _Matrix, int& _ColorRange, /* int& ColorRange_In, */ IScriptEnvironment * env) {
  int _Matrix_Default = Matrix_e::AVS_MATRIX_ST170_M; // Rec601 AVS_MATRIX_ST170_M (6-NTSC) and not AVS_MATRIX_BT470_BG (5-PAL)
  int _ColorRange_Default = ColorRange_e::AVS_RANGE_LIMITED;
  matrix_parse_merge_with_props_def(vi, matrix_name, props, _Matrix, _ColorRange, _Matrix_Default, _ColorRange_Default, env);
}

void chromaloc_parse_merge_with_props(VideoInfo& vi, const char* chromaloc_name, const AVSMap* props, int& _ChromaLocation, int _ChromaLocation_Default, IScriptEnvironment* env) {
  if (props) {
    if (vi.Is420() || vi.Is422() || vi.IsYV411()) { // ?? does YV411 has a valid _ChromaLocation?
      if (env->propNumElements(props, "_ChromaLocation") > 0) {
        _ChromaLocation_Default = env->propGetInt(props, "_ChromaLocation", 0, nullptr); // fixme: range check
      }
    }
    else {
      // Theoretically RGB and not subsampled formats must not have chroma location
      if (env->propNumElements(props, "_ChromaLocation") > 0) {
        // Uncommented for a while, just ignore when there is any
        // env->ThrowError("Error: _ChromaLocation property found at a non-subsampled source.");
      }
    }
  }

  if (is_paramstring_empty_or_auto(chromaloc_name) || !getChromaLocation(chromaloc_name, env, _ChromaLocation)) {
    _ChromaLocation = _ChromaLocation_Default;
  }
}

void export_frame_props(VideoInfo& vi, AVSMap* props, int _Matrix, int _ColorRange, IScriptEnvironment* env) {
  // fixme: what to do with the special "AVERAGE" non standard matrix? Solution 1: delete entry
  if (_Matrix == Matrix_e::AVS_MATRIX_AVERAGE)
    env->propDeleteKey(props, "_Matrix");
  else if (_Matrix < 0)
    env->propDeleteKey(props, "_Matrix");
  else
    env->propSetInt(props, "_Matrix", _Matrix, AVSPropAppendMode::PROPAPPENDMODE_REPLACE);

  env->propSetInt(props, "_ColorRange", _ColorRange, AVSPropAppendMode::PROPAPPENDMODE_REPLACE);
}

/*  avsresize:
    The name of the frame properties that are read and set are: _ChromaLocation, _ColorRange, _Matrix, _Transfer, _Primaries
    The frame properties read and set the corresponding numerical index of the parameters. For example: matrix "709" has numerical index `1` and the frame property have value of `1`.
    If colorspace_op is not defined and there are frame properties, they are used for default source values.
    If colorspace_op is not defined and there are no frame properties or they are not supported, default values are used as before (there are default values for matrix, range and chromaloc).
    If colorspace_op is defined and you want to use the frame property for a source value, use "auto".
    If colorspace_op is defined and you use "auto" without frame property, the default value for that argument will be used if exist.
    If you use "auto" for argument with frame property that has value of 2 (unspec) and use anything different than "same" for destination, error will be raised.
    If you use "auto=>same" for matrix/transfer/primaries with frame property 2 (unspec) and you want to make colorspace conversion, error will be raised. For example:
*/

void update_Matrix_and_ColorRange(AVSMap* props, int theMatrix, int theColorRange, IScriptEnvironment* env)
{
  auto set_int_if_positive = [&](const char* key, int x)
  {
    if (x >= 0)
      env->propSetInt(props, key, x, AVSPropAppendMode::PROPAPPENDMODE_REPLACE);
    else
      env->propDeleteKey(props, key);
  };

  if (theColorRange == ColorRange_e::AVS_RANGE_FULL || theColorRange == ColorRange_e::AVS_RANGE_LIMITED)
    env->propSetInt(props, "_ColorRange", theColorRange, AVSPropAppendMode::PROPAPPENDMODE_REPLACE);
  else
    env->propDeleteKey(props, "_ColorRange");

  set_int_if_positive("_Matrix", theMatrix);
  //set_int_if_positive("_Transfer", theTransferCharacteristics);
  //set_int_if_positive("_Primaries", theColorPrimaries);
}

void update_ChromaLocation(AVSMap* props, int theChromaLocation, IScriptEnvironment* env)
{
  auto set_int_if_positive = [&](const char* key, int x)
  {
    if (x >= 0)
      env->propSetInt(props, key, x, AVSPropAppendMode::PROPAPPENDMODE_REPLACE);
    else
      env->propDeleteKey(props, key);
  };

  set_int_if_positive("_ChromaLocation", theChromaLocation);
}

void update_ColorRange(AVSMap* props, int theColorRange, IScriptEnvironment* env)
{
  if (theColorRange == ColorRange_e::AVS_RANGE_FULL || theColorRange == ColorRange_e::AVS_RANGE_LIMITED)
    env->propSetInt(props, "_ColorRange", theColorRange, AVSPropAppendMode::PROPAPPENDMODE_REPLACE);
  else
    env->propDeleteKey(props, "_ColorRange");
}
