// AviSynth + .Copyright 2026 - AviSynth + Project
// https://avs-plus.net
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

#pragma once
// GetAlphaRect shared types, lookup table accessor, and SIMD dispatch declarations.
// Included by text-overlay.cpp (scalar path) and by the _sse41 / _avx2 TUs.

#include <stdint.h>
#include <cmath>
#include <cstring>

// Function signature with noaa and interlaced baked into each instantiation.
// Parameters mirror the state that GetAlphaRect() used to read from `this`.
using getalpharect_fn_t = void (*)(
    const void*  lpAntialiasBits,
    uint16_t*    soa_buf,
    int w, int h, int w_stride,
    int srcpitch,
    int textcolor, int halocolor,
    int& xl, int& yt, int& xr, int& yb);

// Lookup tables shared by all tiers.
struct GARTables {
  uint8_t  bitcnt[256];   // popcount(b), 0..8
  uint8_t  bitexl[256];   // halo from right neighbour into current column
  uint8_t  bitexr[256];   // halo from left neighbour into current column
  uint16_t gamma[129];    // sqrt-gamma: gamma[n] = uint16_t(sqrt(n/128.0)*516*64+0.5)
};

// Defined out of line in text-overlay.cpp (generic, baseline-ISA TU). Not inline on purpose: an inline
// definition here would also be compiled in the _sse41/_avx2/_avx512 TUs, and the linker keeps a single
// COMDAT copy for all callers, possibly an AVX2 one for the generic text-overlay.cpp caller.
const GARTables& gar_tables() noexcept;

// SWAR popcount for a uint64_t — no CPUID extension required.
static inline int swar_popcount64(uint64_t v) {
  v -= (v >> 1) & UINT64_C(0x5555555555555555);
  v  = (v & UINT64_C(0x3333333333333333)) + ((v >> 2) & UINT64_C(0x3333333333333333));
  v  = (v + (v >> 4)) & UINT64_C(0x0F0F0F0F0F0F0F0F);
  return (int)((v * UINT64_C(0x0101010101010101)) >> 56);
}

#ifdef NEON_INTRINSICS
getalpharect_fn_t GetAlphaRect_select_neon(bool noaa, bool interlaced);
#endif

#ifdef INTEL_INTRINSICS
getalpharect_fn_t GetAlphaRect_select_sse41(bool noaa, bool interlaced);
getalpharect_fn_t GetAlphaRect_select_avx2 (bool noaa, bool interlaced);
#ifdef INTEL_INTRINSICS_AVX512
getalpharect_fn_t GetAlphaRect_select_avx512(bool noaa, bool interlaced);
#endif
#endif
