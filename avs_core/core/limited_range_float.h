// Avisynth+
// https://avs-plus.net
//
// This file is part of Avisynth+ which is released under GPL2+ with exception.

#ifndef __LIMITED_RANGE_FLOAT_H__
#define __LIMITED_RANGE_FLOAT_H__

#include <avs/config.h>

// The limited range convention of 32 bit float. The one place of these numbers:
// get_bits_conv_constants (convert_helper.h) and the helpers below are built on them.
// Integer limited range to limited range needs no such thing, it is a bit shift: value << (bits - 8).
//
// The values are an arbitrary decision: limited range float approximately keeps the 8 bit range, scaled into
// 0..1 by 1/255 (Y 16..235 -> 0.063..0.922, U/V +/-112 -> +/-0.439).
// External plugins mostly treat float as full range always: Y 0..1, U/V -0.5..0.5, whatever _ColorRange says.
// So we rely on, and hope for, limited range float not existing outside Avisynth.
//
// A possible later decision: limited range float is 0..1 and -0.5..0.5 internally as well, overshoot and undershoot
// being the values outside of them, and _ColorRange still keeping the origin for the conversions back and forth.
// Then the constants here have to be adjusted.
#if 0
// The possible later decision above: 16..235 -> 0..1, +/-112 -> +/-0.5.
constexpr float limited_float_luma_span = 1.0f;           // 16..235
constexpr float limited_float_luma_offset = 0.0f;         // float value of 16
constexpr float limited_float_chroma_span = 0.5f;         // +/-112
#else
constexpr float limited_float_luma_span = 219 / 255.0f;   // 16..235
constexpr float limited_float_luma_offset = 16.0f / 255;  // float value of 16
constexpr float limited_float_chroma_span = 112 / 255.0f; // +/-112
#endif

// 8 bit limited range luma level to float. The same calculation as ConvertBits with get_bits_conv_constants.
// For the Y plane of YUV only: RGB and alpha are full range, for them see c8tof.
[[maybe_unused]] static constexpr AVS_FORCEINLINE float y8tof(float luma) {
  return (luma - 16) * (limited_float_luma_span / 219) + limited_float_luma_offset;
}

// A difference of 8 bit limited range luma levels to float: scale only, no offset (e.g. the bright parameter)
[[maybe_unused]] static constexpr AVS_FORCEINLINE float y8diff_tof(float luma_difference) {
  return luma_difference * (limited_float_luma_span / 219);
}

// 8 bit limited range chroma level to float. 16-128-240 -> -112-0-112.
// float parameter: Limiter(paramscale=true) passes the script's levels here, they can be non-integer
[[maybe_unused]] static constexpr AVS_FORCEINLINE float uv8tof(float chroma) {
  return (chroma - 128) * (limited_float_chroma_span / 112);
}

// 8 bit full range to float: RGB and alpha. Not for limited range luma, see y8tof.
[[maybe_unused]] static constexpr AVS_FORCEINLINE float c8tof(int color) {
  return color / 255.0f;
}

#endif // __LIMITED_RANGE_FLOAT_H__
