/*
*
* Avisynth+ Expression filter
* SPDX-License-Identifier: GPL-2.0-or-later
* Base version: ported from the VapourSynth project (Expr, VS r39)
*   Copyright (c) 2012-2015 Fredrik Mellbin
* Modified and extended by the AviSynth+ project. Ideas and syntax elements:
*   masktools2 mt_lut family: built-in constants (ymin/ymax/cmin/cmax/range_*),
*     scaleb/scalef, clip, spatial input (mt_lutspa -> sx/sy/sxr/syr), lut modes
*   wonkey_monkey: variables, store (@) and store-and-pop (^),
*     frameno and time; after his RPN filters
*   akarin.Expr (VSAkarin): clip.PropName frame property syntax,
*     SIMD sin/cos (VapourSynth Expr port), :m border mode compatibility
* Full feature list: see the documentation (expr.rst).
*
* Optimizations
*   x^0.5 is sqrt, ^1 +0 -0 *1 /1 to nothing, ^2, ^3, ^4 by multiplication
*   constant plane expression: plane fill instead of the generic SIMD process
*   single clip letter expression: plane copy (BitBlt)
*   no GetFrame for input clips that are not referenced or plane-copied
*
* Version history: see the changelog in the documentation (expr.rst).
*
* Differences from masktools 2.2.15
* ---------------------------------
*   Up to 26 clips are allowed (x,y,z,a,b,...w). Masktools handles only up to 4 clips with its mt_lut, my_lutxy, mt_lutxyz, mt_lutxyza
*   Clips with different bit depths are allowed
*   works with 32 bit floats instead of 64 bit double internally
*   less functions (e.g. no bit shifts)
*   logical 'false' is 0 instead of -1
*   avs+: ymin, ymax, etc built-in constants can have a _X suffix, where X is the corresponding clip designator letter. E.g. cmax_z, range_half_x
*   mt_lutspa-like functionality is available through "sx", "sy", "sxr", "syr"
*/

#include <iostream>
#include <locale>
#include <sstream>
#include <vector>
#include <list>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <memory>
#include <cmath>
#include <unordered_map>

#include <avisynth.h>

#ifdef AVS_WINDOWS
    #include <avs/win.h>
#else
    #include <avs/posix.h>
#endif

#include <stdlib.h>
#include "../../core/internal.h"
#include "../../core/strings.h" // isAlphaUnderscore, isAlphaNumUnderscore
#include "../../convert/convert_planar.h" // fill_plane
#include "../../convert/convert_helper.h"
#include "avs/alignment.h"

#if defined(_MSC_VER) || (defined(__clang__) && defined(_MSC_VER)) || defined(__MINGW32__) || defined(__MINGW64__)
#include <malloc.h>  // For _aligned_malloc and _aligned_free
#endif

#if (defined(_WIN64) && (defined(_M_AMD64) || defined(_M_X64))) || defined(__x86_64__)
#define JITASM64
#endif

#ifdef INTEL_INTRINSICS
#define VS_TARGET_CPU_X86
#endif
#ifdef AVS_WINDOWS
#define VS_TARGET_OS_WINDOWS
#endif
#include "exprfilter.h"

#ifdef VS_TARGET_CPU_X86
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "jitasm.h"
#endif

#ifndef VS_TARGET_OS_WINDOWS
#include <sys/mman.h>
#endif

#ifdef VS_TARGET_CPU_X86
// Common code, called from the SSE2 and AVX2 codegen paths.
// Copies the JIT code into a new executable memory block. nullptr on failure.
static ExprData::ProcessLineProc allocExecCode(const void* code, size_t size)
{
#ifdef VS_TARGET_OS_WINDOWS
  void* p = VirtualAlloc(nullptr, size, MEM_COMMIT, PAGE_EXECUTE_READWRITE);
  if (!p)
    return nullptr;
#else
  // fd must be -1 for MAP_ANON (FreeBSD rejects 0)
  void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_ANON | MAP_PRIVATE, -1, 0);
  if (p == MAP_FAILED)
    return nullptr;
#endif
  memcpy(p, code, size);
  return (ExprData::ProcessLineProc)p;
}
#endif

#ifdef XP_TLS
#ifdef MSVC_PURE
// v141_xp workaround: disabling /O2 global optimizations reduces build time
// from 56 minutes to 48 seconds.
#pragma optimize("g", off) // disables global optimizations (e.g. /O2 max opt,speed)
#pragma optimize("t", on) // favor speed /Ot
#pragma auto_inline(on) // enable aggressive inlining, equivalent of /Ob2
#endif
#endif

// Source index of position s (may be off-screen) in a row/column of n pixels, for border mode bc.
// Any distance: mirror and wrap modes reflect or wrap as many times as needed (periodic,
// period 2n or 2n-2 for mirror, n for wrap).
static inline int borderIndex(int s, int n, int bc)
{
  if (bc == borderMirror) {
    // boundary value is repeated d c b a | a b c d
    const int period = 2 * n;
    int m = s % period;
    if (m < 0) m += period;
    return m < n ? m : period - 1 - m;
  }
  if (bc == borderReflect101) {
    //                                                4 3 2   1 2 3 4
    // boundary value is single point of reflection e d c b | a b c d
    if (n == 1) return 0;
    const int period = 2 * n - 2;
    int m = s % period;
    if (m < 0) m += period;
    return m < n ? m : period - m;
  }
  if (bc == borderWrap) {
    // periodic: f g h | a b c d
    int m = s % n;
    if (m < 0) m += n;
    return m;
  }
  return std::max(0, std::min(s, n - 1)); // borderClamp, borderConstant (n/a)
}

#ifdef VS_TARGET_CPU_X86

//#define TEST_AVX2_CODEGEN_IN_AVX

#include <immintrin.h>

#if defined(GCC) || defined(CLANG)
#include <avxintrin.h>
#endif

// rounder constants
constexpr int FROUND_TO_NEAREST_INT = 0x00;
constexpr int FROUND_TO_NEG_INF = 0x01;
constexpr int FROUND_TO_POS_INF = 0x02;
constexpr int FROUND_TO_ZERO = 0x03;
constexpr int FROUND_NO_EXC = 0x08;

// normal versions work with two xmm or ymm registers (2*4 or 2*8 pixels per cycle)
// _Single suffixed versions work only one xmm or ymm registers at a time (1*4 or 1*8 pixels per cycle)

#define OneArgOp(instr) \
auto &t1 = stack.back(); \
instr(t1.first, t1.first); \
instr(t1.second, t1.second);

#define OneArgOp_Single(instr) \
auto &t1 = stack1.back(); \
instr(t1, t1);

#define TwoArgOp(instr) \
auto t1 = stack.back(); \
stack.pop_back(); \
auto &t2 = stack.back(); \
instr(t2.first, t1.first); \
instr(t2.second, t1.second);

#define TwoArgOp_Single(instr) \
auto t1 = stack1.back(); \
stack1.pop_back(); \
auto &t2 = stack1.back(); \
instr(t2, t1);

#define TwoArgOp_Avx(instr) \
auto t1 = stack.back(); \
stack.pop_back(); \
auto &t2 = stack.back(); \
instr(t2.first, t2.first, t1.first); \
instr(t2.second, t2.second, t1.second);

#define TwoArgOp_Single_Avx(instr) \
auto t1 = stack1.back(); \
stack1.pop_back(); \
auto &t2 = stack1.back(); \
instr(t2, t2, t1);

#define CmpOp(instr) \
auto t1 = stack.back(); \
stack.pop_back(); \
auto t2 = stack.back(); \
stack.pop_back(); \
instr(t1.first, t2.first); \
instr(t1.second, t2.second); \
andps(t1.first, CPTR(elfloat_one)); \
andps(t1.second, CPTR(elfloat_one)); \
stack.push_back(t1);

#define CmpOp_Single(instr) \
auto t1 = stack1.back(); \
stack1.pop_back(); \
auto t2 = stack1.back(); \
stack1.pop_back(); \
instr(t1, t2); \
andps(t1, CPTR(elfloat_one)); \
stack1.push_back(t1);

#define CmpOp_Avx(instr, op) \
auto t1 = stack.back(); \
stack.pop_back(); \
auto t2 = stack.back(); \
stack.pop_back(); \
instr(t1.first, t1.first, t2.first, op); \
instr(t1.second, t1.second, t2.second, op); \
vandps(t1.first, t1.first, CPTR_AVX(elfloat_one)); \
vandps(t1.second, t1.second, CPTR_AVX(elfloat_one)); \
stack.push_back(t1);

#define CmpOp_Single_Avx(instr, op) \
auto t1 = stack1.back(); \
stack1.pop_back(); \
auto t2 = stack1.back(); \
stack1.pop_back(); \
instr(t1, t1, t2, op); \
vandps(t1, t1, CPTR_AVX(elfloat_one)); \
stack1.push_back(t1);

#define LogicOp(instr) \
auto t1 = stack.back(); \
stack.pop_back(); \
auto t2 = stack.back(); \
stack.pop_back(); \
cmpnleps(t1.first, zero); \
cmpnleps(t1.second, zero); \
cmpnleps(t2.first, zero); \
cmpnleps(t2.second, zero); \
instr(t1.first, t2.first); \
instr(t1.second, t2.second); \
andps(t1.first, CPTR(elfloat_one)); \
andps(t1.second, CPTR(elfloat_one)); \
stack.push_back(t1);

#define LogicOp_Single(instr) \
auto t1 = stack1.back(); \
stack1.pop_back(); \
auto t2 = stack1.back(); \
stack1.pop_back(); \
cmpnleps(t1, zero); \
cmpnleps(t2, zero); \
instr(t1, t2); \
andps(t1, CPTR(elfloat_one)); \
stack1.push_back(t1);

#define LogicOp_Avx(instr) \
auto t1 = stack.back(); \
stack.pop_back(); \
auto t2 = stack.back(); \
stack.pop_back(); \
vcmpps(t1.first, t1.first, zero, _CMP_GT_OQ); \
vcmpps(t1.second, t1.second, zero, _CMP_GT_OQ); \
vcmpps(t2.first, t2.first, zero, _CMP_GT_OQ); \
vcmpps(t2.second, t2.second, zero, _CMP_GT_OQ); \
instr(t1.first, t1.first, t2.first); \
instr(t1.second, t1.second, t2.second); \
vandps(t1.first, t1.first, CPTR_AVX(elfloat_one)); \
vandps(t1.second, t1.second, CPTR_AVX(elfloat_one)); \
stack.push_back(t1);

#define LogicOp_Single_Avx(instr) \
auto t1 = stack1.back(); \
stack1.pop_back(); \
auto t2 = stack1.back(); \
stack1.pop_back(); \
vcmpps(t1, t1, zero, _CMP_GT_OQ); \
vcmpps(t2, t2, zero, _CMP_GT_OQ); \
instr(t1, t1, t2); \
vandps(t1, t1, CPTR_AVX(elfloat_one)); \
stack1.push_back(t1);

enum {
    elabsmask, elc7F, elmin_norm_pos, elinv_mant_mask,
    elfloat_one, elfloat_minusone, elfloat_half, elsignmask, elstore8, elstore10, elstore12, elstore14, elstore16,
    spatialX, spatialX2,
    loadmask1000, loadmask1100, loadmask1110,
    elexp_hi, elexp_lo, elcephes_LOG2EF,
    elcephes_exp_C1, elcephes_log_q2 = elcephes_exp_C1, elcephes_exp_C2, elcephes_log_q1 = elcephes_exp_C2, elcephes_exp_p0, elcephes_exp_p1, elcephes_exp_p2, elcephes_exp_p3, elcephes_exp_p4, elcephes_exp_p5, elcephes_SQRTHF,
    elcephes_log_p0, elcephes_log_p1, elcephes_log_p2, elcephes_log_p3, elcephes_log_p4, elcephes_log_p5, elcephes_log_p6, elcephes_log_p7, elcephes_log_p8,
    float_invpi, float_rintf,
    float_pi1, float_pi2, float_pi3, float_pi4,
    float_sinC3, float_sinC5, float_sinC7, float_sinC9,
    float_cosC2, float_cosC4, float_cosC6, float_cosC8,
    float_atan2f_rmul, float_atan2f_radd, float_atan2f_tmul, float_atan2f_tadd, float_atan2f_halfpi, float_atan2f_pi,
    float_tan_2_over_pi, float_tan_c0, float_tan_c1, float_tan_c2, float_tan_c3, float_tan_c4, float_tan_c5
};

// constants for xmm

#define XCONST(x) { x, x, x, x }


static constexpr ExprUnion logexpconst alignas(16)[66][4] = {
    XCONST(0x7FFFFFFF), // absmask
    XCONST(0x7F), // c7F
    XCONST(0x00800000), // min_norm_pos
    XCONST(~0x7f800000), // inv_mant_mask
    XCONST(1.0f), // float_one
    XCONST(-1.0f), // float_minusone
    XCONST(0.5f), // float_half
    XCONST(0x80000000), // elsignmask
    XCONST(255.0f), // store8
    XCONST(1023.0f), // store10 (avs+)
    XCONST(4095.0f), // store12 (avs+)
    XCONST(16383.0f), // store14 (avs+)
    XCONST(65535.0f), // store16
    { 0.0f, 1.0f, 2.0f, 3.0f }, // spatialX
    { 4.0f, 5.0f, 6.0f, 7.0f }, // spatialX2
    { (int)0xFFFFFFFF, 0x00000000, 0x00000000, 0x00000000 }, // loadmask1000
    { (int)0xFFFFFFFF, (int)0xFFFFFFFF, 0x00000000, 0x00000000 }, // loadmask1100
    { (int)0xFFFFFFFF, (int)0xFFFFFFFF, (int)0xFFFFFFFF, 0x00000000 }, // loadmask1110
    XCONST(88.3762626647949f), // exp_hi
    XCONST(-88.3762626647949f), // exp_lo
    XCONST(1.44269504088896341f), // cephes_LOG2EF
    XCONST(0.693359375f), // cephes_exp_C1
    XCONST(-2.12194440e-4f), // cephes_exp_C2
    XCONST(1.9875691500E-4f), // cephes_exp_p0
    XCONST(1.3981999507E-3f), // cephes_exp_p1
    XCONST(8.3334519073E-3f), // cephes_exp_p2
    XCONST(4.1665795894E-2f), // cephes_exp_p3
    XCONST(1.6666665459E-1f), // cephes_exp_p4
    XCONST(5.0000001201E-1f), // cephes_exp_p5
    XCONST(0.707106781186547524f), // cephes_SQRTHF
    XCONST(7.0376836292E-2f), // cephes_log_p0
    XCONST(-1.1514610310E-1f), // cephes_log_p1
    XCONST(1.1676998740E-1f), // cephes_log_p2
    XCONST(-1.2420140846E-1f), // cephes_log_p3
    XCONST(+1.4249322787E-1f), // cephes_log_p4
    XCONST(-1.6668057665E-1f), // cephes_log_p5
    XCONST(+2.0000714765E-1f), // cephes_log_p6
    XCONST(-2.4999993993E-1f), // cephes_log_p7
    XCONST(+3.3333331174E-1f), // cephes_log_p8
    // sin/cos (SINCOS_PS), ported from VapourSynth's Expr.
    // Range reduction and sin polynomial: glibc libmvec sinf (Intel SVML-derived, LGPL-2.1+),
    // sysdeps/x86_64/fpu/svml_s_trig_data.S: __sInvPI, __sRShifter, __sPI1..__sPI4, __sA3..__sA9.
    // cos polynomial (float_cosC2..C8): unknown origin, degree 8 minimax on [-pi/2, pi/2].
    XCONST(0x3ea2f983), // float_invpi, 1/pi = 0.31830988618379067154f
    XCONST(0x4b400000), // float_rintf  Used for rounding
    XCONST(0x40490000), // float_pi1    High precision part of Pi
    XCONST(0x3a7da000), // float_pi2    Second part of Pi for extended precision
    XCONST(0x34222000), // float_pi3    Third part of Pi for extended precision
    XCONST(0x2cb4611a), // float_pi4    Fourth part of Pi for extended precision
    XCONST(0xbe2aaaa6), // float_sinC3
    XCONST(0x3c08876a), // float_sinC5
    XCONST(0xb94fb7ff), // float_sinC7
    XCONST(0x362edef8), // float_sinC9
    XCONST(static_cast<int32_t>(0xBEFFFFE2)), // float_cosC2
    XCONST(0x3D2AA73C), // float_cosC4
    XCONST(static_cast<int32_t>(0XBAB58D50)), // float_cosC6
    XCONST(0x37C1AD76), // float_cosC8
    XCONST(0x3ccb7dda), // float_atan2f_rmul 0.024840285f
    XCONST(0x3e3f4c37), // float_atan2f_radd 0.18681418f
    XCONST(0xbdc0b66d), // float_atan2f_tmul -0.094097948f
    XCONST(0xbeaa0d0a), // float_atan2f_tadd -0.33213072f
    XCONST(0x3fc90fdb), // float_atan2f_halfpi 1.57079637f
    XCONST(0x40490fdb), // float_atan2f_pi 3.14159274f
    // tan: 2/pi for quadrant, minimax polynomial coefficients for tan(r) = r + r^3*P(r^2), abs(r) <= pi/4
    XCONST(0x3f22f983), // float_tan_2_over_pi 0.636619747f
    XCONST(0x3eaaaa6f), // float_tan_c0 3.333315551e-01f
    XCONST(0x3e0896dd), // float_tan_c1 1.333879977e-01f
    XCONST(0x3d5ac5b2), // float_tan_c2 5.341119319e-02f
    XCONST(0x3cc8225c), // float_tan_c3 2.443044633e-02f
    XCONST(0x3b4c6f37), // float_tan_c4 3.119421890e-03f
    XCONST(0x3c19c679), // float_tan_c5 9.385698475e-03f
};


#define CPTR(x) (xmmword_ptr[constptr + (x) * 16])

// AVX2 stuff
// constants for ymm

#undef XCONST
#define XCONST(x) { x, x, x, x, x, x, x, x }

static constexpr ExprUnion logexpconst_avx alignas(32)[66][8] = {
  XCONST(0x7FFFFFFF), // absmask
  XCONST(0x7F), // c7F
  XCONST(0x00800000), // min_norm_pos
  XCONST(~0x7f800000), // inv_mant_mask
  XCONST(1.0f), // float_one
  XCONST(-1.0f), // float_minusone
  XCONST(0.5f), // float_half
  XCONST(0x80000000), // elsignmask
  XCONST(255.0f), // store8
  XCONST(1023.0f), // store10 (avs+)
  XCONST(4095.0f), // store12 (avs+)
  XCONST(16383.0f), // store14 (avs+)
  XCONST(65535.0f), // store16
  { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f }, // spatialX
  { 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f }, // spatialX2
  { (int)0xFFFFFFFF, 0x00000000, 0x00000000, 0x00000000, 0, 0, 0, 0 }, // loadmask1000 not used, avx supports blendps
  { (int)0xFFFFFFFF, (int)0xFFFFFFFF, 0x00000000, 0x00000000, 0, 0, 0, 0 }, // loadmask1100 not used, avx supports blendps
  { (int)0xFFFFFFFF, (int)0xFFFFFFFF, (int)0xFFFFFFFF, 0x00000000, 0, 0, 0, 0 }, // loadmask1110 not used, avx supports blendps
  XCONST(88.3762626647949f), // exp_hi
  XCONST(-88.3762626647949f), // exp_lo
  XCONST(1.44269504088896341f), // cephes_LOG2EF
  XCONST(0.693359375f), // cephes_exp_C1
  XCONST(-2.12194440e-4f), // cephes_exp_C2
  XCONST(1.9875691500E-4f), // cephes_exp_p0
  XCONST(1.3981999507E-3f), // cephes_exp_p1
  XCONST(8.3334519073E-3f), // cephes_exp_p2
  XCONST(4.1665795894E-2f), // cephes_exp_p3
  XCONST(1.6666665459E-1f), // cephes_exp_p4
  XCONST(5.0000001201E-1f), // cephes_exp_p5
  XCONST(0.707106781186547524f), // cephes_SQRTHF
  XCONST(7.0376836292E-2f), // cephes_log_p0
  XCONST(-1.1514610310E-1f), // cephes_log_p1
  XCONST(1.1676998740E-1f), // cephes_log_p2
  XCONST(-1.2420140846E-1f), // cephes_log_p3
  XCONST(+1.4249322787E-1f), // cephes_log_p4
  XCONST(-1.6668057665E-1f), // cephes_log_p5
  XCONST(+2.0000714765E-1f), // cephes_log_p6
  XCONST(-2.4999993993E-1f), // cephes_log_p7
  XCONST(+3.3333331174E-1f), // cephes_log_p8
  // sin/cos: see logexpconst
  XCONST(0x3ea2f983), // float_invpi, 1/pi = 0.31830988618379067154f
  XCONST(0x4b400000), // float_rintf  Used for rounding
  XCONST(0x40490000), // float_pi1    High precision part of Pi
  XCONST(0x3a7da000), // float_pi2    Second part of Pi for extended precision
  XCONST(0x34222000), // float_pi3    Third part of Pi for extended precision
  XCONST(0x2cb4611a), // float_pi4    Fourth part of Pi for extended precision
  XCONST(0xbe2aaaa6), // float_sinC3
  XCONST(0x3c08876a), // float_sinC5
  XCONST(0xb94fb7ff), // float_sinC7
  XCONST(0x362edef8), // float_sinC9
  XCONST(static_cast<int32_t>(0xBEFFFFE2)), // float_cosC2
  XCONST(0x3D2AA73C), // float_cosC4
  XCONST(static_cast<int32_t>(0XBAB58D50)), // float_cosC6
  XCONST(0x37C1AD76), // float_cosC8 
  XCONST(0x3ccb7dda), // float_atan2f_rmul 0.024840285f
  XCONST(0x3e3f4c37), // float_atan2f_radd 0.18681418f
  XCONST(0xbdc0b66d), // float_atan2f_tmul -0.094097948f
  XCONST(0xbeaa0d0a), // float_atan2f_tadd -0.33213072f
  XCONST(0x3fc90fdb), // float_atan2f_halfpi 1.57079637f
  XCONST(0x40490fdb), // float_atan2f_pi 3.14159274f
  // tan: 2/pi for quadrant, minimax polynomial coefficients for tan(r) = r + r^3*P(r^2), |r| <= pi/4
  XCONST(0x3f22f983), // float_tan_2_over_pi 0.636619747f
  XCONST(0x3eaaaa6f), // float_tan_c0 3.333315551e-01f
  XCONST(0x3e0896dd), // float_tan_c1 1.333879977e-01f
  XCONST(0x3d5ac5b2), // float_tan_c2 5.341119319e-02f
  XCONST(0x3cc8225c), // float_tan_c3 2.443044633e-02f
  XCONST(0x3b4c6f37), // float_tan_c4 3.119421890e-03f
  XCONST(0x3c19c679), // float_tan_c5 9.385698475e-03f
};
#undef XCONST

#define CPTR_AVX(x) (ymmword_ptr[constptr + (x) * 32])

#define EXP_PS(x) { \
XmmReg fx, emm0, etmp, y, mask, z; \
minps(x, CPTR(elexp_hi)); \
maxps(x, CPTR(elexp_lo)); \
movaps(fx, x); \
mulps(fx, CPTR(elcephes_LOG2EF)); \
addps(fx, CPTR(elfloat_half)); \
cvttps2dq(emm0, fx); \
cvtdq2ps(etmp, emm0); \
movaps(mask, etmp); \
cmpnleps(mask, fx); \
andps(mask, CPTR(elfloat_one)); \
movaps(fx, etmp); \
subps(fx, mask); \
movaps(etmp, fx); \
mulps(etmp, CPTR(elcephes_exp_C1)); \
movaps(z, fx); \
mulps(z, CPTR(elcephes_exp_C2)); \
subps(x, etmp); \
subps(x, z); \
movaps(z, x); \
mulps(z, z); \
movaps(y, CPTR(elcephes_exp_p0)); \
mulps(y, x); \
addps(y, CPTR(elcephes_exp_p1)); \
mulps(y, x); \
addps(y, CPTR(elcephes_exp_p2)); \
mulps(y, x); \
addps(y, CPTR(elcephes_exp_p3)); \
mulps(y, x); \
addps(y, CPTR(elcephes_exp_p4)); \
mulps(y, x); \
addps(y, CPTR(elcephes_exp_p5)); \
mulps(y, z); \
addps(y, x); \
addps(y, CPTR(elfloat_one)); \
cvttps2dq(emm0, fx); \
paddd(emm0, CPTR(elc7F)); \
pslld(emm0, 23); \
mulps(y, emm0); \
x = y; }

#define LOG_PS(x) { \
XmmReg emm0, invalid_mask, mask, y, etmp, z; \
xorps(invalid_mask, invalid_mask); \
cmpnleps(invalid_mask, x); \
maxps(x, CPTR(elmin_norm_pos)); \
movaps(emm0, x); \
psrld(emm0, 23); \
andps(x, CPTR(elinv_mant_mask)); \
orps(x, CPTR(elfloat_half)); \
psubd(emm0, CPTR(elc7F)); \
cvtdq2ps(emm0, emm0); \
addps(emm0, CPTR(elfloat_one)); \
movaps(mask, x); \
cmpltps(mask, CPTR(elcephes_SQRTHF)); \
movaps(etmp, x); \
andps(etmp, mask); \
subps(x, CPTR(elfloat_one)); \
andps(mask, CPTR(elfloat_one)); \
subps(emm0, mask); \
addps(x, etmp); \
movaps(z, x); \
mulps(z, z); \
movaps(y, CPTR(elcephes_log_p0)); \
mulps(y, x); \
addps(y, CPTR(elcephes_log_p1)); \
mulps(y, x); \
addps(y, CPTR(elcephes_log_p2)); \
mulps(y, x); \
addps(y, CPTR(elcephes_log_p3)); \
mulps(y, x); \
addps(y, CPTR(elcephes_log_p4)); \
mulps(y, x); \
addps(y, CPTR(elcephes_log_p5)); \
mulps(y, x); \
addps(y, CPTR(elcephes_log_p6)); \
mulps(y, x); \
addps(y, CPTR(elcephes_log_p7)); \
mulps(y, x); \
addps(y, CPTR(elcephes_log_p8)); \
mulps(y, x); \
mulps(y, z); \
movaps(etmp, emm0); \
mulps(etmp, CPTR(elcephes_log_q1)); \
addps(y, etmp); \
mulps(z, CPTR(elfloat_half)); \
subps(y, z); \
mulps(emm0, CPTR(elcephes_log_q2)); \
addps(x, y); \
addps(x, emm0); \
orps(x, invalid_mask); }

#define EXP_PS_AVX(x) { \
YmmReg fx, emm0, etmp, y, mask, z; \
vminps(x, x, CPTR_AVX(elexp_hi)); \
vmaxps(x, x, CPTR_AVX(elexp_lo)); \
vmulps(fx, x, CPTR_AVX(elcephes_LOG2EF)); \
vaddps(fx, fx, CPTR_AVX(elfloat_half)); \
vcvttps2dq(emm0, fx); \
vcvtdq2ps(etmp, emm0); \
vcmpps(mask, etmp, fx, _CMP_GT_OQ); /* cmpnleps */ \
vandps(mask, mask, CPTR_AVX(elfloat_one)); \
vsubps(fx, etmp, mask); \
vfnmadd231ps(x, fx, CPTR_AVX(elcephes_exp_C1)); \
vfnmadd231ps(x, fx, CPTR_AVX(elcephes_exp_C2)); \
vmulps(z, x, x); \
vmovaps(y, CPTR_AVX(elcephes_exp_p0)); \
vfmadd213ps(y, x, CPTR_AVX(elcephes_exp_p1)); \
vfmadd213ps(y, x, CPTR_AVX(elcephes_exp_p2)); \
vfmadd213ps(y, x, CPTR_AVX(elcephes_exp_p3)); \
vfmadd213ps(y, x, CPTR_AVX(elcephes_exp_p4)); \
vfmadd213ps(y, x, CPTR_AVX(elcephes_exp_p5)); \
vfmadd213ps(y, z, x); \
vaddps(y, y, CPTR_AVX(elfloat_one)); \
vcvttps2dq(emm0, fx); \
vpaddd(emm0, emm0, CPTR_AVX(elc7F)); \
vpslld(emm0, emm0, 23); \
vmulps(x, y, emm0); \
}

#define LOG_PS_AVX(x) { \
YmmReg emm0, invalid_mask, mask, y, etmp, z; \
vcmpps(invalid_mask, zero, x, _CMP_GT_OQ); /* cmpnleps. or signalling _CMP_NLE_US? */ \
vmaxps(x, x, CPTR_AVX(elmin_norm_pos)); \
vpsrld(emm0, x, 23); \
vandps(x, x, CPTR_AVX(elinv_mant_mask)); \
vorps(x, x, CPTR_AVX(elfloat_half)); \
vpsubd(emm0, emm0, CPTR_AVX(elc7F)); \
vcvtdq2ps(emm0, emm0); \
vaddps(emm0, emm0, CPTR_AVX(elfloat_one)); \
vcmpps(mask, x, CPTR_AVX(elcephes_SQRTHF), _CMP_LT_OQ); /* cmpltps. or signalling _CMP_LT_OS? */ \
vandps(etmp, x, mask); \
vsubps(x, x, CPTR_AVX(elfloat_one)); \
vandps(mask, mask, CPTR_AVX(elfloat_one)); \
vsubps(emm0, emm0, mask); \
vaddps(x, x, etmp); \
vmulps(z, x, x); \
vmovaps(y, CPTR_AVX(elcephes_log_p0)); \
vfmadd213ps(y, x, CPTR_AVX(elcephes_log_p1)); \
vfmadd213ps(y, x, CPTR_AVX(elcephes_log_p2)); \
vfmadd213ps(y, x, CPTR_AVX(elcephes_log_p3)); \
vfmadd213ps(y, x, CPTR_AVX(elcephes_log_p4)); \
vfmadd213ps(y, x, CPTR_AVX(elcephes_log_p5)); \
vfmadd213ps(y, x, CPTR_AVX(elcephes_log_p6)); \
vfmadd213ps(y, x, CPTR_AVX(elcephes_log_p7)); \
vfmadd213ps(y, x, CPTR_AVX(elcephes_log_p8)); \
vmulps(y, y, x); \
vmulps(y, y, z); \
vfmadd231ps(y, emm0, CPTR_AVX(elcephes_log_q1)); \
vfnmadd231ps(y, z, CPTR_AVX(elfloat_half)); \
vaddps(x, x, y); \
vfmadd231ps(x, emm0, CPTR_AVX(elcephes_log_q2)); \
vorps(x, x, invalid_mask); }

// Note: VS Expr changed a lot since ported to Avisynth, 
// Here we are using their VEX macros for easy port of new sin and cos
// however we do not support VEX encoding or FMA3 in xmm register mode
// Note: VEX2 cmpltps -> vcmpltps does not work so we uncomment v##op parts as well (comparisons changed in vex)
#define VEX1(op, arg1, arg2) \
do { \
  if constexpr(false /*cpuFlags & CPUF_AVX*/) \
    /*v##op(arg1, arg2)*/; \
  else \
    op(arg1, arg2); \
} while (0)
#define VEX1IMM(op, arg1, arg2, imm) \
do { \
  if constexpr(false /*cpuFlags & CPUF_AVX*/) { \
    /*v##op(arg1, arg2, imm)*/; \
  } else if (arg1 == arg2) { \
    op(arg2, imm); \
  } else { \
    movdqa(arg1, arg2); \
    op(arg1, imm); \
  } \
} while (0)
#define VEX2(op, arg1, arg2, arg3) \
do { \
  if constexpr(false /*cpuFlags & CPUF_AVX*/) { \
    /*v##op(arg1, arg2, arg3)*/; \
  } else if (arg1 == arg2) { \
    op(arg2, arg3); \
  } else if (arg1 != arg3) { \
    movdqa(arg1, arg2); \
    op(arg1, arg3); \
  } else { \
    XmmReg tmp; \
    movdqa(tmp, arg2); \
    op(tmp, arg3); \
    movdqa(arg1, tmp); \
  } \
} while (0)
#define VEX2IMM(op, arg1, arg2, arg3, imm) \
do { \
  if constexpr(false/*cpuFlags & CPUF_AVX*/) { \
    /*v##op(arg1, arg2, arg3, imm)*/; \
  } else if (arg1 == arg2) { \
    op(arg2, arg3, imm); \
  } else if (arg1 != arg3) { \
    movdqa(arg1, arg2); \
    op(arg1, arg3, imm); \
  } else { \
    XmmReg tmp; \
    movdqa(tmp, arg2); \
    op(tmp, arg3, imm); \
    movdqa(arg1, tmp); \
  } \
} while (0)

#if 0
// Scalar reference of the TAN_PS / TAN_PS_AVX JIT code
// Each step maps to one SIMD instruction.
// Same scheme as e.g. SLEEF's vectorized tan, see https://arxiv.org/pdf/2001.09258 :
// - k = round(x * 2/pi), r = x - k * pi/2 (Cody-Waite, four-part pi), |r| <= pi/4
// - tan(r) = r + r^3 * P(r^2), P: degree 5 (c0..c5), own fit
// - odd k: tan(x) = -1/tan(r)
// Coefficients: minimax fit for the relative error of tan(r) on [0, pi/4] (tan is odd), by Lawson's
// algorithm (iteratively reweighted least squares, 20000 points, 200 iterations) in double
// precision, then rounded to float. Fit error 1.8e-8; float32 polynomial evaluation <= 1.4 ulp.
// Max error vs. double tan (all floats): SSE (no FMA) <= 3.5 ulp for |x| < 1e4, degrades above ~5e4
// (k * pi1 no longer exact, same limit as sin/cos); AVX2 (FMA) <= 3.9 ulp up to 1e5.
float fast_tanf(float x) {
  float kb = x * 0.636619747f + 12582912.0f; // 1.5*2^23 (float_rintf): rounds to integer
  uint32_t kb_bits;
  memcpy(&kb_bits, &kb, sizeof(kb_bits)); // C++17 equivalent of C++20 std::bit_cast<uint32_t>(kb)
  const bool odd = (kb_bits & 1) != 0; // lowest mantissa bit is the parity of k
  const float khalf = (kb - 12582912.0f) * 0.5f; // k/2: k * pi/2 = k/2 * pi, exact
  float r = x - khalf * pi1; // float_pi1..float_pi4
  r = r - khalf * pi2;
  r = r - khalf * pi3;
  r = r - khalf * pi4;
  const float z = r * r;
  float p = c5 * z + c4;
  p = p * z + c3;
  p = p * z + c2;
  p = p * z + c1;
  p = p * z + c0;
  const float t = r + (r * z) * p;
  return odd ? -1.0f / t : t / 1.0f; // one division in SIMD: blended numerator / denominator
}
#endif

// Fast tangent approximation for non-AVX version, see fast_tanf
#define TAN_PS(x0) { \
XmmReg x1, x2, x3, x4; \
/* k = round(x * 2/pi): adding 1.5*2^23 rounds to integer, lowest mantissa bit is the parity of k */ \
VEX2(mulps, x1, x0, CPTR(float_tan_2_over_pi)); \
VEX2(addps, x1, x1, CPTR(float_rintf)); /* biased k */ \
VEX1(movaps, x2, x1); \
pslld(x2, 31); \
psrad(x2, 31); /* odd quadrant mask */ \
VEX2(subps, x1, x1, CPTR(float_rintf)); /* k */ \
VEX2(mulps, x1, x1, CPTR(elfloat_half)); /* k/2: k * pi/2 = k/2 * pi, exact */ \
/* r = x - k * pi/2, four-part pi for precision, |r| <= pi/4 */ \
VEX2(mulps, x4, x1, CPTR(float_pi1)); \
VEX2(subps, x3, x0, x4); \
VEX2(mulps, x4, x1, CPTR(float_pi2)); \
VEX2(subps, x3, x3, x4); \
VEX2(mulps, x4, x1, CPTR(float_pi3)); \
VEX2(subps, x3, x3, x4); \
VEX2(mulps, x4, x1, CPTR(float_pi4)); \
VEX2(subps, x3, x3, x4); /* r */ \
/* tan(r) = r + r*z*P(z), z = r^2 */ \
VEX2(mulps, x4, x3, x3); /* z */ \
VEX1(movaps, x1, CPTR(float_tan_c5)); \
VEX2(mulps, x1, x1, x4); \
VEX2(addps, x1, x1, CPTR(float_tan_c4)); \
VEX2(mulps, x1, x1, x4); \
VEX2(addps, x1, x1, CPTR(float_tan_c3)); \
VEX2(mulps, x1, x1, x4); \
VEX2(addps, x1, x1, CPTR(float_tan_c2)); \
VEX2(mulps, x1, x1, x4); \
VEX2(addps, x1, x1, CPTR(float_tan_c1)); \
VEX2(mulps, x1, x1, x4); \
VEX2(addps, x1, x1, CPTR(float_tan_c0)); /* P(z) */ \
VEX2(mulps, x4, x4, x3); /* r*z */ \
VEX2(mulps, x1, x1, x4); \
VEX2(addps, x1, x1, x3); /* t = tan(r) */ \
/* odd k: -1/t, even k: t/1 */ \
VEX2(andnps, x3, x2, x1); /* even: t */ \
VEX2(andps, x4, x2, CPTR(elfloat_minusone)); /* odd: -1 */ \
VEX2(orps, x3, x3, x4); /* numerator */ \
VEX2(andps, x4, x2, x1); /* odd: t */ \
VEX2(andnps, x2, x2, CPTR(elfloat_one)); /* even: 1 */ \
VEX2(orps, x4, x4, x2); /* denominator */ \
VEX2(divps, x0, x3, x4); \
}

// Fast tangent approximation for AVX version, see fast_tanf
#define TAN_PS_AVX(x0) { \
YmmReg x1, x2, x3, x4; \
/* k = round(x * 2/pi): adding 1.5*2^23 rounds to integer, lowest mantissa bit is the parity of k */ \
vmovaps(x1, CPTR_AVX(float_tan_2_over_pi)); \
vfmadd213ps(x1, x0, CPTR_AVX(float_rintf)); /* biased k */ \
vpslld(x2, x1, 31); /* odd quadrant: sign bit set, enough for vblendvps */ \
vsubps(x1, x1, CPTR_AVX(float_rintf)); /* k */ \
vmulps(x1, x1, CPTR_AVX(elfloat_half)); /* k/2: k * pi/2 = k/2 * pi, exact */ \
/* r = x - k * pi/2, four-part pi for precision, |r| <= pi/4 */ \
vmovaps(x3, x0); \
vfnmadd231ps(x3, x1, CPTR_AVX(float_pi1)); \
vfnmadd231ps(x3, x1, CPTR_AVX(float_pi2)); \
vfnmadd231ps(x3, x1, CPTR_AVX(float_pi3)); \
vfnmadd231ps(x3, x1, CPTR_AVX(float_pi4)); /* r */ \
/* tan(r) = r + r*z*P(z), z = r^2 */ \
vmulps(x4, x3, x3); /* z */ \
vmovaps(x1, CPTR_AVX(float_tan_c5)); \
vfmadd213ps(x1, x4, CPTR_AVX(float_tan_c4)); \
vfmadd213ps(x1, x4, CPTR_AVX(float_tan_c3)); \
vfmadd213ps(x1, x4, CPTR_AVX(float_tan_c2)); \
vfmadd213ps(x1, x4, CPTR_AVX(float_tan_c1)); \
vfmadd213ps(x1, x4, CPTR_AVX(float_tan_c0)); /* P(z) */ \
vmulps(x4, x4, x3); /* r*z */ \
vfmadd213ps(x1, x4, x3); /* t = tan(r) = r*z*P(z) + r */ \
/* odd k: -1/t, even k: t/1 */ \
vmovaps(x3, CPTR_AVX(elfloat_minusone)); \
vblendvps(x3, x1, x3, x2); /* numerator */ \
vmovaps(x4, CPTR_AVX(elfloat_one)); \
vblendvps(x4, x4, x1, x2); /* denominator */ \
vdivps(x0, x3, x4); \
}

// atan2: based on https://stackoverflow.com/questions/46210708/atan2-approximation-with-11bits-in-mantissa-on-x86with-sse2-and-armwith-vfpv4?noredirect=1&lq=1
#if 0
float fast_atan2f(float y, float x)
{
  // max rel err = 3.53486939e-5
  const float atan2f_rmul = 0.024840285f;
  const float atan2f_radd = 0.18681418f;
  const float atan2f_tmul = -0.094097948f;
  const float atan2f_tadd = -0.33213072f;
  const float atan2f_halfpi = 1.57079637f;
  const float atan2f_pi = 3.14159274f;

  float a, r, s, t, c, q, ax, ay, mx, mn;
  ax = fabsf(x);
  ay = fabsf(y);

  mx = fmaxf(ay, ax);
  mn = fminf(ay, ax);
  a = mn / mx;
  // Minimax polynomial approximation to atan(a) on [0,1]
  s = a * a;
  c = s * a;
  q = s * s;
  r = atan2f_rmul * q + atan2f_radd;
  t = atan2f_tmul * q + atan2f_tadd;
  r = r * s + t;
  r = r * c + a;
  // Map to full circle
  if (ay > ax) r = atan2f_halfpi - r;
  if (x < 0) r = atan2f_pi - r;
  if (y < 0) r = -r;
  return r;
}
#endif

// atan2(0, 0) = 0
// ~speed: "y x atan2" C/SSE2/AVX2:52/480/1000 fps
#define ATAN2_PS(x0 /*y*/, /*x*/x1) { \
XmmReg x2, x3, x4, x5, x6, x7, x8; \
/*VEX1(movaps, x1, x);*/ \
/*VEX1(movaps, x0, y);*/ \
/* Remove sign */ \
VEX1(movaps, x3, CPTR(elabsmask)); \
VEX1(movaps, x2, x1); \
VEX2(andps, x2, x2, x3); /* ax = fabsf (x); */ \
VEX2(andps, x3, x3, x0); /* ay = fabsf (y); */ \
VEX1(movaps, x4, x3); \
VEX2(maxps, x4, x4, x2); /* mx = fmaxf (ay, ax); */ \
VEX1(movaps, x5, x3); \
VEX2(minps, x5, x5, x2); /* fminf (ay, ax); */ \
VEX2(divps, x5, x5, x4); /* a = mn / mx; */ \
VEX1(movaps, x4, x5); \
VEX2(mulps, x4, x4, x5); /* s = a * a; */ \
VEX1(movaps, x6, x4); \
VEX2(mulps, x6, x6, x4); /* q = s * s; */ \
VEX1(movaps, x7, CPTR(float_atan2f_rmul)); \
VEX2(mulps, x7, x7, x6); /* r = atan2f_rmul * q (+ atan2f_radd) */ \
VEX2(addps, x7, x7, CPTR(float_atan2f_radd)); /* r = (atan2f_rmul * q) + atan2f_radd; */ \
VEX2(mulps, x7, x7, x4); /* r = r * s (+ t) */ \
VEX2(mulps, x4, x4, x5); /* c = s * a; */ \
VEX2(mulps, x6, x6, CPTR(float_atan2f_tmul)); /* t = atan2f_tmul * q (+ atan2f_tadd) */ \
VEX2(addps, x6, x6, CPTR(float_atan2f_tadd)); /* t = (atan2f_tmul * q) + atan2f_tadd; */ \
VEX2(addps, x7, x7, x6); /* r = (r * s) + t; */ \
VEX2(mulps, x7, x7, x4); /* r = r * c (+ a) */ \
VEX2(addps, x7, x7, x5); /* r = (r * c) + a */ \
/* Map to full circle */ \
/* if (ay > ax) r = atan2f_halfpi - r; */ \
/* if (x < 0) r = atan2f_pi - r; */ \
/* if (y < 0) r = -r; */ \
/* r = atan2f_halfpi - r */ \
VEX1(movaps, x4, CPTR(float_atan2f_halfpi)); \
VEX2(subps, x4, x4, x7); /* r = atan2f_halfpi - r */ \
VEX2(cmpltps, x2, x2, x3); /* if (ay > ax) */ \
/* blend */ \
VEX1(movaps, x3, x2); \
VEX2(andnps, x3, x3, x7); \
VEX2(andps, x2, x2, x4); \
VEX2(orps, x2, x2, x3); \
/* r = atan2f_pi - r; */ \
VEX1(movaps, x3, CPTR(float_atan2f_pi)); \
VEX2(subps, x3, x3, x2); /* r = atan2f_pi - r */ \
/* if (x < 0) */ \
VEX2(xorps, x4, x4, x4); /* zero */ \
VEX2(cmpltps, x1, x1, x4); /* if (x < 0) */ \
/* blend */ \
VEX1(movaps, x5, x1); \
VEX2(andnps, x5, x5, x2); \
VEX2(andps, x1, x1, x3); \
VEX2(orps, x1, x1, x5); \
/* r = -r; */ \
VEX1(movaps, x2, CPTR(elsignmask)); \
VEX2(subps, x2, x2, x1); /* r = -r */ \
/* if (y < 0) */ \
VEX2(cmpltps, x0, x0, x4); /* if (y < 0) */ \
/* blend */ \
VEX1(movaps, x3, x0); \
VEX2(andnps, x3, x3, x1); \
VEX2(andps, x0, x0, x2); \
VEX2(orps, x0, x0, x3); \
/* extra check when 0,0 given -> convert NaN to 0 */ \
VEX1(movaps, x3, x0); \
VEX2(cmpordps, x3, x3, x3);/* find NaNs. 0: NaN in either. FFFF: both non-Nan */ \
/* mask NaN to zero */ \
VEX2(andps, x0, x0, x3); \
/* return value in y */ \
/* input was "x0 for y */ \
}

#define ATAN2_PS_AVX(x0 /*y*/, x1 /*x*/) { \
YmmReg x2, x3, x4, x5, x6, x7, x8; \
/* Remove sign */ \
/* vmovaps(x1, x); */ \
/* vmovaps(x0, y); */ \
vmovaps(x2, CPTR_AVX(elabsmask)); \
vandps(x8, x1, x2); /* ax = fabsf (x); */ \
vandps(x2, x0, x2); /* ay = fabsf (y); */ \
vmaxps(x4, x8, x2); /* mx = fmaxf (ay, ax); */ \
vminps(x5, x8, x2); /* fminf (ay, ax); */ \
vdivps(x4, x5, x4); /* a = mn / mx; */ \
vmulps(x5, x4, x4); /* s = a * a; */ \
vmulps(x6, x5, x5); /* q = s * s; */ \
vmovaps(x7, CPTR_AVX(float_atan2f_rmul)); \
vmovaps(x3, CPTR_AVX(float_atan2f_radd)); \
vfmadd213ps(x7, x6, CPTR_AVX(float_atan2f_radd)); /* r = atan2f_rmul * q + atan2f_radd; */ \
vmovaps(x3, CPTR_AVX(float_atan2f_tmul)); \
vfmadd213ps(x3, x6, CPTR_AVX(float_atan2f_tadd)); /* t = atan2f_tmul * q + atan2f_tadd */ \
vmulps(x6, x5, x4); \
vfmadd231ps(x3, x5, x7); /* r = r * s + t; */ \
vfmadd213ps(x3, x6, x4); /* r = (r * c) + a */ \
/* Map to full circle */ \
/* if (ay > ax) r = atan2f_halfpi - r; */ \
/* if (x < 0) r = atan2f_pi - r; */ \
/* if (y < 0) r = -r; */ \
/* r = atan2f_pi - r */ \
vmovaps(x4, CPTR_AVX(float_atan2f_halfpi)); \
vsubps(x4, x4, x3); /* r = atan2f_halfpi - r */ \
vcmpps(x2, x8, x2, _CMP_LT_OQ); /*vcmpltps(x2, x8, x2);*/ /* if (ay > ax) */ \
vblendvps(x2, x3, x4, x2); \
vmovaps(x3, CPTR_AVX(float_atan2f_pi)); \
vsubps(x3, x3, x2); /* r = atan2f_pi - r */ \
vxorps(x4, x4, x4); \
vcmpps(x1, x1, x4, _CMP_LT_OQ); /* vcmpltps(x1, x1, x4); */ /* if (x < 0) */ \
vblendvps(x1, x2, x3, x1); \
vmovaps(x2, CPTR_AVX(elsignmask)); \
vxorps(x2, x1, x2); /* r = -r */ \
vcmpps(x0, x0, x4, _CMP_LT_OQ); /* vcmpltps(x0, x0, x4); */ /* if (y < 0) */ \
vblendvps(x0, x1, x2, x0); \
/* extra check when 0,0 given -> convert NaN to 0 */ \
vcmpps(x3, x0, x0, _CMP_ORD_Q); /* vcmpordps, x3, x3, x3);*/ /* find NaNs. 0: NaN in either. FFFF: both non-Nan */ \
/* mask NaN to zero */ \
vandps(x0, x0, x3); \
/* return value in y */ \
/* no need. input was "x0" for y */ \
}

// sin/cos: k = round(x/pi) (1.5*2^23 shifter), r = x - k*pi (four-part pi, Cody-Waite),
// sign flipped for odd k; sin(r) = r + r^3*P(r^2) (degree 9), cos(r) = 1 + r^2*Q(r^2) (degree 8),
// |r| <= pi/2. Same scheme and comments as glibc libmvec sinf (see the constant table).
#define SINCOS_PS(issin, y, x) { \
XmmReg t1, sign, t2, t3, t4; \
/* // Remove sign */ \
VEX1(movaps, t1, CPTR(elabsmask)); \
if (issin) { \
  VEX1(movaps, sign, t1); \
  VEX2(andnps, sign, sign, x); \
} \
else { \
  VEX2(pxor, sign, sign, sign); \
} \
VEX2(andps, t1, t1, x); \
/*// Range reduction*/ \
VEX1(movaps, t3, CPTR(float_rintf)); \
VEX2(mulps, t2, t1, CPTR(float_invpi)); \
VEX2(addps, t2, t2, t3); \
VEX1IMM(pslld, t4, t2, 31); \
VEX2(xorps, sign, sign, t4); \
VEX2(subps, t2, t2, t3); \
if constexpr(false /*cpuFlags & CPUF_FMA3*/) { \
  vfnmadd231ps(t1, t2, CPTR(float_pi1)); \
  vfnmadd231ps(t1, t2, CPTR(float_pi2)); \
  vfnmadd231ps(t1, t2, CPTR(float_pi3)); \
  vfnmadd231ps(t1, t2, CPTR(float_pi4)); \
} \
else { \
  VEX2(mulps, t4, t2, CPTR(float_pi1)); \
  VEX2(subps, t1, t1, t4); \
  VEX2(mulps, t4, t2, CPTR(float_pi2)); \
  VEX2(subps, t1, t1, t4); \
  VEX2(mulps, t4, t2, CPTR(float_pi3)); \
  VEX2(subps, t1, t1, t4); \
  VEX2(mulps, t4, t2, CPTR(float_pi4)); \
  VEX2(subps, t1, t1, t4); \
} \
if (issin) { \
  /* // Evaluate minimax polynomial for sin(x) in [-pi/2, pi/2] interval */ \
  /* // Y <- X + X * X^2 * (C3 + X^2 * (C5 + X^2 * (C7 + X^2 * C9))) */ \
  VEX2(mulps, t2, t1, t1); \
  if constexpr(false /*cpuFlags & CPUF_FMA3*/) { \
    vmovaps(t3,  CPTR(float_sinC7)); \
    vfmadd231ps(t3, t2, CPTR(float_sinC9)); \
    vfmadd213ps(t3, t2, CPTR(float_sinC5)); \
    vfmadd213ps(t3, t2, CPTR(float_sinC3)); \
    VEX2(mulps, t3, t3, t2); \
    vfmadd231ps(t1, t1, t3); \
  } \
  else { \
    VEX2(mulps, t3, t2, CPTR(float_sinC9)); \
    VEX2(addps, t3, t3, CPTR(float_sinC7)); \
    VEX2(mulps, t3, t3, t2); \
    VEX2(addps, t3, t3, CPTR(float_sinC5)); \
    VEX2(mulps, t3, t3, t2); \
    VEX2(addps, t3, t3, CPTR(float_sinC3)); \
    VEX2(mulps, t3, t3, t2); \
    VEX2(mulps, t3, t3, t1); \
    VEX2(addps, t1, t1, t3); \
  } \
} \
else { \
/*  // Evaluate minimax polynomial for cos(x) in [-pi/2, pi/2] interval */ \
/*  // Y <- 1 + X^2 * (C2 + X^2 * (C4 + X^2 * (C6 + X^2 * C8))) */ \
  VEX2(mulps, t2, t1, t1); \
  if constexpr(false /*cpuFlags & CPUF_FMA3*/) { \
    vmovaps(t1, CPTR(float_cosC6)); \
    vfmadd231ps(t1, t2, CPTR(float_cosC8)); \
    vfmadd213ps(t1, t2, CPTR(float_cosC4)); \
    vfmadd213ps(t1, t2, CPTR(float_cosC2)); \
    vfmadd213ps(t1, t2, CPTR(elfloat_one)); \
  } \
  else { \
    VEX2(mulps, t1, t2, CPTR(float_cosC8)); \
    VEX2(addps, t1, t1, CPTR(float_cosC6)); \
    VEX2(mulps, t1, t1, t2); \
    VEX2(addps, t1, t1, CPTR(float_cosC4)); \
    VEX2(mulps, t1, t1, t2); \
    VEX2(addps, t1, t1, CPTR(float_cosC2)); \
    VEX2(mulps, t1, t1, t2); \
    VEX2(addps, t1, t1, CPTR(elfloat_one)); \
  } \
} \
/*// Apply sign */ \
VEX2(xorps, y, t1, sign); \
}

// sin/cos, AVX2 version of SINCOS_PS
// y dst x src
#define SINCOS_PS_AVX(issin, y, x) { \
YmmReg t1, sign, t2, t3, t4; \
/* // Remove sign */ \
vmovaps(t1, CPTR_AVX(elabsmask)); \
if (issin) { \
  vmovaps(sign, t1); \
  vandnps(sign, sign, x); \
} \
else { \
  vxorps(sign, sign, sign); \
} \
vandps(t1, t1, x); \
/*// Range reduction*/ \
vmovaps(t3, CPTR_AVX(float_rintf)); \
vmulps(t2, t1, CPTR_AVX(float_invpi)); \
vaddps(t2, t2, t3); \
vpslld(t4, t2, 31); \
vxorps(sign, sign, t4); \
vsubps(t2, t2, t3); \
vfnmadd231ps(t1, t2, CPTR_AVX(float_pi1)); \
vfnmadd231ps(t1, t2, CPTR_AVX(float_pi2)); \
vfnmadd231ps(t1, t2, CPTR_AVX(float_pi3)); \
vfnmadd231ps(t1, t2, CPTR_AVX(float_pi4)); \
if (issin) { \
  /* // Evaluate minimax polynomial for sin(x) in [-pi/2, pi/2] interval */ \
  /* // Y <- X + X * X^2 * (C3 + X^2 * (C5 + X^2 * (C7 + X^2 * C9))) */ \
  vmulps(t2, t1, t1); \
  vmovaps(t3,  CPTR_AVX(float_sinC7)); \
  vfmadd231ps(t3, t2, CPTR_AVX(float_sinC9)); \
  vfmadd213ps(t3, t2, CPTR_AVX(float_sinC5)); \
  vfmadd213ps(t3, t2, CPTR_AVX(float_sinC3)); \
  vmulps(t3, t3, t2); \
  vfmadd231ps(t1, t1, t3); \
} \
else { \
/*  // Evaluate minimax polynomial for cos(x) in [-pi/2, pi/2] interval */ \
/*  // Y <- 1 + X^2 * (C2 + X^2 * (C4 + X^2 * (C6 + X^2 * C8))) */ \
  vmulps(t2, t1, t1); \
  vmovaps(t1, CPTR_AVX(float_cosC6)); \
  vfmadd231ps(t1, t2, CPTR_AVX(float_cosC8)); \
  vfmadd213ps(t1, t2, CPTR_AVX(float_cosC4)); \
  vfmadd213ps(t1, t2, CPTR_AVX(float_cosC2)); \
  vfmadd213ps(t1, t2, CPTR_AVX(elfloat_one)); \
} \
/*// Apply sign */ \
vxorps(y, t1, sign); \
}

// x % d, same result as C fmodf when |x/d| < 2^24:
// q = trunc(|x|/|d|) can be one too big (division rounding), never too small.
// r = |x| - q*|d| is exactly representable; computed exactly (SSE2: in double, AVX2: by FMA)
// r < 0 (q was one too big): r += |d|. Result gets the sign of x (also -0); d == 0: NaN.
// SSE2 intrinsic equivalent of FMOD_PS:
//   __m128 fmod_ps(__m128 x, __m128 d) {
//     const __m128 absmask = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));
//     __m128 ax = _mm_and_ps(x, absmask);
//     __m128 ad = _mm_and_ps(d, absmask);
//     __m128 q = _mm_cvtepi32_ps(_mm_cvttps_epi32(_mm_div_ps(ax, ad)));
//     __m128d lo = _mm_sub_pd(_mm_cvtps_pd(ax), _mm_mul_pd(_mm_cvtps_pd(q), _mm_cvtps_pd(ad)));
//     __m128d hi = _mm_sub_pd(_mm_cvtps_pd(_mm_movehl_ps(ax, ax)),
//                             _mm_mul_pd(_mm_cvtps_pd(_mm_movehl_ps(q, q)), _mm_cvtps_pd(_mm_movehl_ps(ad, ad))));
//     __m128 r = _mm_movelh_ps(_mm_cvtpd_ps(lo), _mm_cvtpd_ps(hi));
//     r = _mm_add_ps(r, _mm_and_ps(_mm_cmplt_ps(r, _mm_setzero_ps()), ad));
//     r = _mm_or_ps(r, _mm_andnot_ps(absmask, x));
//     return _mm_or_ps(r, _mm_cmpeq_ps(ad, _mm_setzero_ps()));
//   }
#define FMOD_PS(x, d) { \
XmmReg aX, aD, aQ, aR, aT0, aT1, aT2; \
movaps(aX, x); \
andps(aX, CPTR(elabsmask)); /* |x| */ \
movaps(aD, d); \
andps(aD, CPTR(elabsmask)); /* |d| */ \
movaps(aQ, aX); \
divps(aQ, aD); \
cvttps2dq(aQ, aQ); \
cvtdq2ps(aQ, aQ); /* q = trunc(|x|/|d|) */ \
/* r = |x| - q*|d| in double: 24x24 bit product is exact, the difference is representable */ \
cvtps2pd(aT0, aQ); \
cvtps2pd(aT1, aD); \
mulpd(aT0, aT1); \
cvtps2pd(aT1, aX); \
subpd(aT1, aT0); \
cvtpd2ps(aR, aT1); /* elements 0,1 */ \
movhlps(aQ, aQ); \
movaps(aT2, aD); \
movhlps(aT2, aT2); \
movhlps(aX, aX); \
cvtps2pd(aT0, aQ); \
cvtps2pd(aT1, aT2); \
mulpd(aT0, aT1); \
cvtps2pd(aT1, aX); \
subpd(aT1, aT0); \
cvtpd2ps(aT1, aT1); \
movlhps(aR, aT1); /* elements 2,3 */ \
xorps(aT0, aT0); \
movaps(aT1, aR); \
cmpltps(aT1, aT0); \
andps(aT1, aD); \
addps(aR, aT1); /* r < 0: r += |d| */ \
cmpeqps(aD, aT0); /* d == 0 */ \
andps(x, CPTR(elsignmask)); \
orps(x, aR); /* sign of x */ \
orps(x, aD); /* d == 0: NaN */ \
}

#define FMOD_PS_AVX(x, d) { \
YmmReg aX, aD, aQ, aZ; \
vandps(aX, x, CPTR_AVX(elabsmask)); /* |x| */ \
vandps(aD, d, CPTR_AVX(elabsmask)); /* |d| */ \
vdivps(aQ, aX, aD); \
vroundps(aQ, aQ, FROUND_TO_ZERO | FROUND_NO_EXC); /* q = trunc(|x|/|d|) */ \
vfnmadd231ps(aX, aQ, aD); /* r = |x| - q*|d|, single rounding: exact */ \
vxorps(aZ, aZ, aZ); \
vcmpps(aQ, aX, aZ, _CMP_LT_OQ); \
vandps(aQ, aQ, aD); \
vaddps(aX, aX, aQ); /* r < 0: r += |d| */ \
vcmpps(aD, aD, aZ, _CMP_EQ_OQ); /* d == 0 */ \
vandps(x, x, CPTR_AVX(elsignmask)); \
vorps(x, x, aX); /* sign of x */ \
vorps(x, x, aD); /* d == 0: NaN */ \
}

// Register types for the code generator functions below. They are not class members,
// so they cannot use the evaluators' inherited Reg / XmmReg names.
using JitReg = jitasm::Reg;
using JitXmmReg = jitasm::XmmReg;


// pshufd immediate: element x gets source element src_element_of(x)
template<typename F>
static constexpr uint8_t elements_imm(F src_element_of) {
  return (uint8_t)(src_element_of(0) | (src_element_of(1) << 2) | (src_element_of(2) << 4) | (src_element_of(3) << 6));
}

// Element masks for emitPshufdWithFiller (bit i set: element i gets the filler)
// first_elements(n): elements 0..n-1     n = 1, 2, 3, 4 -> 0001, 0011, 0111, 1111 (0x1, 0x3, 0x7, 0xF)
// last_elements(n):  elements 4-n..3     n = 1, 2, 3, 4 -> 1000, 1100, 1110, 1111 (0x8, 0xC, 0xE, 0xF)
static constexpr uint8_t first_elements(int n) { return (uint8_t)((1 << n) - 1); }
static constexpr uint8_t last_elements(int n) { return (uint8_t)(0xF & ~((1 << (4 - n)) - 1)); }


// borderConstant: scale_inputs conversion of the loaded pixels in r (and r2, when given),
// before the filler is put in place of the off-screen ones. Same steps, same order as the
// separate operations of other loads (see ExprLoadConv). No code when there is no conversion.
template<bool VEX, class J>
static AVS_FORCEINLINE void emitConvert4(J& j, JitXmmReg& r, JitXmmReg* r2, const ExprLoadConv* conv)
{
  if (conv == nullptr || !conv->active())
    return;
  enum { Sub, Mul, Add };
  // one step: broadcast 'value', then sub/mul/add it to r (and r2)
  auto step = [&](float value, int kind) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits)); // C++17 equivalent of C++20 std::bit_cast<uint32_t>(value)
    jitasm::Reg32 t;
    JitXmmReg c;
    j.mov(t, (int32_t)bits);
    if constexpr (VEX) {
      j.vmovd(c, t);
      j.vpshufd(c, c, 0);
    }
    else {
      j.movd(c, t);
      j.pshufd(c, c, 0);
    }
    for (JitXmmReg* x : { &r, r2 }) {
      if (x == nullptr)
        continue;
      if constexpr (VEX) {
        if (kind == Sub) j.vsubps(*x, *x, c);
        else if (kind == Mul) j.vmulps(*x, *x, c);
        else j.vaddps(*x, *x, c);
      }
      else {
        if (kind == Sub) j.subps(*x, c);
        else if (kind == Mul) j.mulps(*x, c);
        else j.addps(*x, c);
      }
    }
  };
  // and now the real work, optimized to skip no-op steps
  if (conv->sub != 0.0f) step(conv->sub, Sub);
  if (conv->mul != 1.0f) step(conv->mul, Mul);
  if (conv->add != 0.0f) step(conv->add, Add);
  if (conv->add_uv != 0.0f) step(conv->add_uv, Add);
}

// Load 4 pixels from [base + offset] and convert them to 4 floats.
// VEX: VEX-encoded instructions for the AVX2 evaluator; 'zero' for the SSE2 unpack based
//      widening is not needed, direct vpmovzx* zero-extends directly.
// conv: borderConstant only, the scale_inputs conversion (see emitConvert4), nullptr: none.
template<bool VEX, class J>
static AVS_FORCEINLINE void emitLoad4AsFloat(J& j, JitXmmReg& r, JitReg& base, int offset, int op, const JitXmmReg* zero, const ExprLoadConv* conv)
{
  if constexpr (VEX) {
    if (op == opLoadRelSrc8) {
      // 8->32 bits like _mm256_cvtepu8_epi32
      j.vpmovzxbd(r, j.dword_ptr[base + offset]);
      j.vcvtdq2ps(r, r);
    }
    else if (op == opLoadRelSrc16) {
      // 16->32 bit like _mm256_cvtepu16_epi32
      j.vpmovzxwd(r, j.qword_ptr[base + offset]);
      j.vcvtdq2ps(r, r);
    }
    else // opLoadRelSrcF32
      j.vmovdqu(r, j.xmmword_ptr[base + offset]);
  }
  else {
    // SSE2: widen with unpacks
    if (op == opLoadRelSrc8) {
      j.movd(r, j.dword_ptr[base + offset]);
      j.punpcklbw(r, *zero);
      j.punpcklwd(r, *zero);
      j.cvtdq2ps(r, r);
    }
    else if (op == opLoadRelSrc16) {
      j.movq(r, j.mmword_ptr[base + offset]);
      j.punpcklwd(r, *zero);
      j.cvtdq2ps(r, r);
    }
    else // opLoadRelSrcF32
      j.movdqu(r, j.xmmword_ptr[base + offset]);
  }
  emitConvert4<VEX>(j, r, nullptr, conv);
}

template<bool VEX, class J>
static AVS_FORCEINLINE void emitPshufd(J& j, JitXmmReg& r, uint8_t imm)
{
  if constexpr (VEX)
    j.vpshufd(r, r, imm);
  else
    j.pshufd(r, r, imm);
}

// used from 'constant' and 'wrap' border modes
template<bool VEX, class J>
static AVS_FORCEINLINE void emitPshufdWithFiller(
  J& j,
  JitXmmReg& r,
  uint8_t imm, // shuffle
  uint8_t blend_mask, // bit i set: element i gets the filler: first_elements(n) or last_elements(n), n = 1..4 (VEX: vblendps immediate)
  JitXmmReg& filler_value_r,
  JitReg* constptr) // SSE2 only (element mask constants), nullptr for VEX
{
  if (blend_mask == first_elements(4)) {
    // 4: all elements filler: nothing to shuffle or blend
    if constexpr (VEX)
      j.vmovaps(r, filler_value_r);
    else
      j.movaps(r, filler_value_r);
    return;
  }
  // shuffle first
  emitPshufd<VEX, J>(j, r, imm);

  // overwrite with filler_value_r constant
  if (blend_mask != 0)
  {
    if constexpr (VEX)
    {
      // VEX 32-bit blending (0 = r, 1 = filler_value_r)
      j.vblendps(r, r, filler_value_r, blend_mask);
    }
    else
    {
      // SSE2 blend with the loadmask1000/1100/1110 mask constants
      // Unfortunately blend exists only since sse 4.1
      // SSE2: r = (filler & fillmask) | (r & ~fillmask)
      auto element_mask = [&](int c) { return j.xmmword_ptr[*constptr + c * 16]; };

      JitXmmReg t;
      switch (blend_mask) {
      case first_elements(1):
      case first_elements(2):
      case first_elements(3): {
        // first 1,2,3 => mask1000/1100/1110
        // prefix: the filler elements are the mask elements
        const int fillmask = blend_mask == first_elements(1) ? loadmask1000 : blend_mask == first_elements(2) ? loadmask1100 : loadmask1110;
        j.movaps(t, element_mask(fillmask));
        j.andnps(t, r); // t = r & ~fillmask (andnps: dst = ~dst & src)
        j.movaps(r, filler_value_r);
        j.andps(r, element_mask(fillmask)); // r = filler & fillmask
        j.orps(r, t);
        break;
      }
      case last_elements(1):
      case last_elements(2):
      case last_elements(3): {
        // last 1,2,3 => mask1110/1100/1000 (+inverted arguments)
        // suffix: the filler elements are the complement of a prefix, keep the prefix of r
        const int keepmask = blend_mask == last_elements(1) ? loadmask1110 : blend_mask == last_elements(2) ? loadmask1100 : loadmask1000;
        j.movaps(t, element_mask(keepmask));
        j.andnps(t, filler_value_r); // t = filler & ~keepmask
        j.andps(r, element_mask(keepmask)); // r = r & keepmask
        j.orps(r, t);
        break;
      }
      default:
        assert(false); // cannot occur
      }
    }
  }
}


// Generates code that moves 'a' (pointer at xcounter in the current row of clip 'clip') to the
// source row y + dy, handled by border mode bc.
// Mirror and wrap modes: only a single reflection or wrap is handled, |dy| < planeheight
// (checked when the JIT is chosen for the plane).
template<class J>
static void emitRowAdjust(J& j, JitReg& a, JitReg& regptrs, JitReg& SpatialY, int clip, int dy, int bc, int planeheight)
{
  if (dy == 0)
    return;
  JitReg rows; // signed row difference
  if (bc == borderClamp || bc == borderConstant) {
    // borderConstant: pointer clamped to a valid row, the caller replaces the result
    // with the filler when the source row is off-screen (emitRowOffscreenCheck)
    // repeat (clamp) mode
    JitReg sy;
    if (dy < 0) {
      // Read from above: rows = -min(-dy, y)
      j.mov(sy, SpatialY);
      j.mov(rows, -dy);
      j.cmp(rows, sy);
      j.cmovg(rows, sy);
      j.neg(rows);
    }
    else {
      // Read from below: rows = min(dy, planeheight - 1 - y)
      j.mov(sy, planeheight - 1);
      j.sub(sy, SpatialY);
      j.mov(rows, dy);
      j.cmp(rows, sy);
      j.cmovg(rows, sy);
    }
  }
  else if (bc == borderWrap) {
    // source row t = y + dy, wrapped once
    JitReg t, alt;
    j.mov(t, SpatialY);
    j.add(t, dy);
    // alt computed independently from SpatialY (not as a copy of t)
    j.mov(alt, SpatialY);
    if (dy < 0) {
      j.add(alt, dy + planeheight); // t < 0: t + h
      j.cmp(t, 0);
      j.cmovl(t, alt);
    }
    else {
      j.add(alt, dy - planeheight); // t >= h: t - h
      j.cmp(t, planeheight);
      j.cmovge(t, alt);
    }
    j.mov(rows, t);
    j.sub(rows, SpatialY);
  }
  else {
    // two types of mirror modes: borderMirror and borderReflect101
    // source row t = y + dy, reflected once
    JitReg t, alt;
    j.mov(t, SpatialY);
    j.add(t, dy);
    if (dy < 0) {
      j.mov(alt, bc == borderMirror ? -1 : 0); // t < 0: -1 - t (:m) or -t (:r)
      j.sub(alt, t);
      j.cmp(t, 0);
      j.cmovl(t, alt);
    }
    else {
      j.mov(alt, 2 * planeheight - (bc == borderMirror ? 1 : 2)); // t >= h: 2h-1-t (:m) or 2h-2-t (:r)
      j.sub(alt, t);
      j.cmp(t, planeheight);
      j.cmovge(t, alt);
    }
    j.mov(rows, t);
    j.sub(rows, SpatialY);
  }
#ifdef JITASM64
  j.imul(rows, j.qword_ptr[regptrs + sizeof(void *) * (clip + RWPTR_START_OF_STRIDES)]); // rows * stride
#else
  j.imul(rows, j.dword_ptr[regptrs + sizeof(void *) * (clip + RWPTR_START_OF_STRIDES)]); // rows * stride
#endif
  j.add(a, rows);
}

// borderConstant: emits a jump to labelOnScreen when the source row y + dy is inside the plane.
// The code following it (up to the label) is for an off-screen row: fill with the filler.
template<class J>
static void emitRowOffscreenCheck(J& j, JitReg& SpatialY, int dy, int planeheight, const std::string& labelOnScreen)
{
  if (dy < 0) {
    j.cmp(SpatialY, -dy);
    j.jge(labelOnScreen); // y + dy >= 0
  }
  else {
    j.cmp(SpatialY, planeheight - 1 - dy);
    j.jle(labelOnScreen); // y + dy <= planeheight - 1
  }
}

// Generates code (into JIT function j) that loads 4 pixels at x + dxh (x = current xcounter) as floats.
// Why four: 4 float pixels fill one 128 bit XMM register.
// Where the position falls outside the row, border mode bc decides:
// - repeat the first/last pixel (borderClamp, :c)
// - mirror (borderMirror :m, borderReflect101 :r).
// - constant (borderConstant :k): the filler value of filler_r (from the stack, see foldConstants).
//   The loaded pixels get their scale_inputs conversion (conv) right after the load, the filler
//   is already in the expression's domain: it is blended in after the conversion.
// - wrap (borderWrap :w): the pixel from the other end of the row (periodic)
// Result: pixel i of r = pixel borderIndex(xcounter + dxh + i, planewidth, bc) of the row
// pointed by 'a' (a: pointer at xcounter, vertical dy offset already applied).
// For :k: the filler when xcounter + dxh + i is outside 0..planewidth-1. An off-screen
// source row (vertical) is not handled here: 'a' points to a clamped valid row, the caller
// replaces the whole result with the filler (emitRowOffscreenCheck).
//
// Mirror and wrap modes reflect or wrap once: valid for -(planewidth-1) <= dx <= planewidth-3
// (:m, :w) or planewidth-4 (:r). Other offsets cannot be handled by JIT, and handled by scalar C path.
//
// When planewidth is not a multiple of 4, the last group extends past the row end
// (xcounter + i >= planewidth). Those output pixels are never stored, their values
// do not matter.
//
// Cases:
// - dxh == 0: simple load
// - dxh < 0: left edge crossed
//   Partial (group crosses the edge): load pixels 0..3, constant shuffle.
//     constant: elements 0..s-1 get the filler (emitPshufdWithFiller, blend mask 1, 3, 7).
//     wrap: elements 0..s-1 come from the row end: pixels planewidth-4..planewidth-1 are loaded
//     as well, shuffled, and blended in (blend mask 1, 3, 7).
//   Full (group entirely left of the row):
//   - clamp: load pixels 0..3, all = pixel 0;
//   - mirror: the needed pixels are a reversed run of 4 inside the row, load it, reverse.
//   - constant: all = filler, no load.
//   - wrap: a forward run of 4 from newx+planewidth, at a constant offset from 'a'
//     (dxh+planewidth), no shuffle.
// - dxh > 0: right edge crossed.
//   Partial: load pixels planewidth-4..planewidth-1, constant shuffle.
//     constant: elements 4-t..3 get the filler (blend mask 8, C, E).
//     wrap: elements 4-t..3 come from the row start: pixels 0..3 are loaded as well, shuffled,
//     and blended in (blend mask 8, C, E).
//   Full:
//   - clamp: same load, all = pixel planewidth-1
//   - mirror: reversed run of 4, its start limited to >= 0 (only groups without
//     stored pixels would start lower).
//   - constant: all = filler, no load.
//   - wrap: a forward run of 4 from newx-planewidth, at a constant offset from 'a'
//     (dxh-planewidth), no shuffle. For groups without stored pixels the run may extend
//     past the row end, not further than a normal load of the same group.
// - planewidth < 4: only the group at xcounter == 0 is stored, one constant shuffle
//   (any dx, any mode). constant: plus a blend of the off-screen elements; no load when
//   all elements are off-screen.
//
// - Constant shuffles: xcounter is a multiple of 4, so the shift of a partial edge case is
//   known at code generation time (pshufd immediate, see elements_imm).
// - Shuffles are done after conversion to float: one implementation for 8/16/32 bit,
//   only emitLoad4AsFloat depends on the pixel type.
// - Reads stay within [0, planewidth) of the row, except where a normal (non-relative) load of
//   the same pixel group would read as well.
// - Used by the SSE2 (VEX=false) and AVX2 (VEX=true) evaluators; xcounter is a multiple of 4
//   in both (8 or 16 in AVX2).
template<bool VEX, class J>
static void emitRelLoad4(J& j, JitXmmReg& r, JitReg& a, JitReg& regptrs, const JitXmmReg* zero, int op, int dxh, int planewidth, int bc,
  JitXmmReg& filler_r, // borderConstant: the filler value in all 4 elements
  const ExprLoadConv* conv, // borderConstant: scale_inputs conversion of the loaded pixels, done before the filler is blended in
  JitReg* sse_helper_constptr) // SSE2 (VEX=false) only: constant table for emitPshufdWithFiller, nullptr for VEX
{
  const int shift = op == opLoadRelSrc8 ? 0 : op == opLoadRelSrc16 ? 1 : 2; // log2(bytes per pixel)
  if (dxh == 0) {
    // valid elements need no border handling
    emitLoad4AsFloat<VEX>(j, r, a, 0, op, zero, conv);
    return;
  }
  if (planewidth < 4) {
    // only the group at xcounter == 0 has valid elements: everything is known at compile time
    // :k mode: off-screen elements get the filler. Only stored elements (i < planewidth) matter, so only one
    // side is checked: dxh < 0: a prefix of elements, dxh > 0: a suffix (contiguous blend mask)
    uint8_t blendmask = 0;
    if (bc == borderConstant) {
      // dxh < 0: elements i + dxh < 0, the first -dxh elements; dxh > 0: elements i + dxh >= planewidth, the last 4-(planewidth-dxh)
      blendmask = dxh < 0 ? first_elements(std::min(-dxh, 4)) : last_elements(std::max(0, std::min(4 - (planewidth - dxh), 4)));
      if (blendmask == first_elements(4)) {
        emitPshufdWithFiller<VEX>(j, r, 0, first_elements(4), filler_r, sse_helper_constptr); // all filler, no source pixel needed
        return;
      }
    }
    JitReg rowp, xb;
    j.mov(xb, j.ptr[regptrs + sizeof(void*) * (RWPTR_START_OF_XCOUNTER)]);
    j.shl(xb, shift);
    j.mov(rowp, a);
    j.sub(rowp, xb); // start of row
    emitLoad4AsFloat<VEX>(j, r, rowp, 0, op, zero, conv);
    if (bc != borderConstant) {
      emitPshufd<VEX>(j, r, elements_imm([=](int i) { return borderIndex(i + dxh, planewidth, bc); }));
    }
    else {
      emitPshufdWithFiller<VEX>(j, r, elements_imm([=](int i) { return std::max(0, std::min(i + dxh, 3)); }), blendmask, filler_r, sse_helper_constptr);
    }

    return;
  }
  // trivial cases are done

  std::string LabelEdge = "relEdge" + j.getLabelCount();
  std::string LabelPartial = "relPartial" + j.getLabelCount();
  std::string LabelEnd = "relEnd" + j.getLabelCount();

  JitReg newx;
  j.mov(newx, j.ptr[regptrs + sizeof(void*) * (RWPTR_START_OF_XCOUNTER)]);
  j.add(newx, dxh);
  // dxh < 0: only the left edge needs border handling, valid elements never reach the right edge.
  // dxh > 0: only the right edge.
  j.cmp(newx, dxh < 0 ? 0 : planewidth - 4);
  if (dxh < 0)
    j.jl(LabelEdge); // newx < 0
  else
    j.jg(LabelEdge); // newx > planewidth - 4

  emitLoad4AsFloat<VEX>(j, r, a, dxh << shift, op, zero, conv); // normal case: no border handling
  j.jmp(LabelEnd);

  j.L(LabelEdge);
  JitReg rowp;
  j.mov(rowp, j.ptr[regptrs + sizeof(void*) * (RWPTR_START_OF_XCOUNTER)]);
  j.shl(rowp, shift); // index to real ptr 1-2-4x uint8/uint16/float
  j.neg(rowp);
  j.add(rowp, a); // start of row
  const uint8_t reverse = elements_imm([](int i) { return 3 - i; });
  if (dxh < 0) {
    const int s = (-dxh) & 3; // -newx for the partial case, -3 < newx < 0
    if (s != 0) {
      j.cmp(newx, -4);
      j.jg(LabelPartial);
    }
    // newx <= -4: all elements are left of the row start
    if (bc == borderClamp) {
      emitLoad4AsFloat<VEX>(j, r, rowp, 0, op, zero, conv);
      emitPshufd<VEX>(j, r, elements_imm([](int) { return 0; })); // all elements pixel 0
    }
    else if (bc == borderConstant) {
      // :k mode: all elements filler, no source pixel needed
      emitPshufdWithFiller<VEX>(j, r, 0, first_elements(4) /* all elements */, filler_r, sse_helper_constptr);
    }
    else if (bc == borderWrap) {
      // wrapped: pixels newx+i+w, a forward run of 4 inside the row (1 <= newx+w, newx+w+3 <= w-1).
      // Relative to 'a' (xcounter) it is a constant offset: dxh + w
      emitLoad4AsFloat<VEX>(j, r, a, (dxh + planewidth) << shift, op, zero, conv);
    }
    else {
      // two kinds of mirror modes: borderMirror and borderReflect101
      // mirrored: pixels -1-newx-i (:m) or -newx-i (:r), a reversed run of 4 inside the row.
      // Its first pixel (of element 3) from the row start: c - newx, c = -4 (:m), -3 (:r).
      // Note: on x86 this case crashed under register pressure (a jitasm register allocator bug,
      // fixed 20261008 in jitasm.h GenerateInterBlockInstr); it was worked around by reading X
      // from memory until then.
      JitReg b;
      j.mov(b, newx);
      j.neg(b);
      j.sub(b, bc == borderMirror ? 4 : 3);
      j.shl(b, shift); // index to real ptr 1-2-4x uint8/uint16/float
      j.add(b, rowp);
      emitLoad4AsFloat<VEX>(j, r, b, 0, op, zero, conv);
      emitPshufd<VEX>(j, r, reverse);
    }
    if (s != 0) {
      j.jmp(LabelEnd);
      j.L(LabelPartial);
      // all needed pixels are within 0..3 (wrap: also within planewidth-4..planewidth-1)
      emitLoad4AsFloat<VEX>(j, r, rowp, 0, op, zero, conv);
      if (bc == borderWrap) {
        // elements 0..s-1 wrap to the row end: pixels w-s+i, block w-4..w-1 index 4-s+i (blend mask 1, 3, 7);
        // the others pixel i - s from the row start
        JitXmmReg u;
        emitLoad4AsFloat<VEX>(j, u, rowp, (planewidth - 4) << shift, op, zero, conv);
        emitPshufd<VEX>(j, u, elements_imm([=](int i) { return i < s ? 4 - s + i : 3; }));
        const uint8_t blendmask = first_elements(s);
        emitPshufdWithFiller<VEX>(j, r, elements_imm([=](int i) { return std::max(i - s, 0); }), blendmask, u, sse_helper_constptr);
      }
      else if (bc != borderConstant)
        emitPshufd<VEX>(j, r, elements_imm([=](int i) { return borderIndex(i - s, planewidth, bc); }));
      else {
        // elements 0..s-1 are left of the row start: filler (blend mask 1, 3, 7), the others pixel i - s
        const uint8_t blendmask = first_elements(s);
        emitPshufdWithFiller<VEX>(j, r, elements_imm([=](int i) { return std::max(i - s, 0); }), blendmask, filler_r, sse_helper_constptr);
      }
    }
  }
  else {
    const int t = (dxh - planewidth) & 3;
    // newx - (planewidth - 4) for the partial case, 0 < t < 4

    // clamp (edge repeat): with t == 3 only pixel 0 is inside the row, all 4 elements
    // get pixel at (planewidth-1), the same as the full case, where
    // no partial branch needed (t == 0 is the trivial no partial case)
    //
    //   loaded: the last 4 pixels of the row      P Q R S | (row end)
    //   block index                               0 1 2 3 |
    //                        clamp                                  constant (:k), k: filler
    //   t   group starts at  needs        -> indexes                needs        -> indexes  blend mask
    //   1        Q           Q R S + S       1 2 3 3                Q R S + k       1 2 3 k     8
    //   2        R           R S + S S       2 3 3 3                R S + k k       2 3 k k     C
    //   3        S           S + S S S       3 3 3 3 = full case    S + k k k       3 k k k     E
    const bool hasPartial = bc == borderClamp ? (t == 1 || t == 2) : (t != 0);
    if (hasPartial) {
      j.cmp(newx, planewidth);
      j.jl(LabelPartial);
    }
    // newx >= planewidth: all elements are right of the row end
    if (bc == borderClamp) {
      emitLoad4AsFloat<VEX>(j, r, rowp, (planewidth - 4) << shift /* byte index */, op, zero, conv);
      emitPshufd<VEX>(j, r, elements_imm([](int) { return 3; })); // all elements pixel planewidth-1
    }
    else if (bc == borderConstant) {
      // :k mode: all elements filler, no source pixel needed
      emitPshufdWithFiller<VEX>(j, r, 0, first_elements(4) /* all elements */, filler_r, sse_helper_constptr);
    }
    else if (bc == borderWrap) {
      // wrapped: pixels newx-w+i, a forward run of 4 from newx-w (>= 0).
      // Relative to 'a' (xcounter) it is a constant offset: dxh - w.
      // For groups without stored pixels the run may extend past the row end,
      // but not further than a normal load of the same pixel group would read.
      emitLoad4AsFloat<VEX>(j, r, a, (dxh - planewidth) << shift, op, zero, conv);
    }
    else {
      // w: planewidth. :m and :r are the mirror modes (borderMirror, borderReflect101)
      // mirrored: pixels 2w-1-newx-i (:m) or 2w-2-newx-i (:r), a reversed run of 4.
      // Its start is negative only for groups without stored pixels: limited to 0.
      // First pixel (of element 3) from the row start:
      // c - newx, where c = 2w-4 (:m), 2w-5 (:r), limited to >= 0.
      JitReg b, z;
      j.mov(b, 2 * planewidth - (bc == borderMirror ? 4 : 5));
      j.sub(b, newx);
      j.mov(z, 0);
      j.cmp(b, 0);
      j.cmovl(b, z); // limited to 0
      j.shl(b, shift); // index to real ptr 1-2-4x uint8/uint16/float
      j.add(b, rowp);
      emitLoad4AsFloat<VEX>(j, r, b, 0, op, zero, conv);
      emitPshufd<VEX>(j, r, reverse);
    }

    if (hasPartial) {
      j.jmp(LabelEnd);
      j.L(LabelPartial);
      // all needed pixels are within planewidth-4..planewidth-1 (wrap: also within 0..3)
      emitLoad4AsFloat<VEX>(j, r, rowp, (planewidth - 4) << shift, op, zero, conv);
      if (bc == borderWrap) {
        // elements 4-t..3 wrap to the row start: pixels t+i-4 (blend mask 8, C, E);
        // the others block pixel t + i
        JitXmmReg u;
        emitLoad4AsFloat<VEX>(j, u, rowp, 0, op, zero, conv);
        emitPshufd<VEX>(j, u, elements_imm([=](int i) { return i < 4 - t ? 0 : t + i - 4; }));
        const uint8_t blendmask = last_elements(t);
        emitPshufdWithFiller<VEX>(j, r, elements_imm([=](int i) { return std::min(t + i, 3); }), blendmask, u, sse_helper_constptr);
      }
      else if (bc != borderConstant) {
        emitPshufd<VEX>(j, r, elements_imm([=](int i) { return borderIndex(planewidth - 4 + t + i, planewidth, bc) - (planewidth - 4); }));
      }
      else {
        // elements 4-t..3 are right of the row end: filler (blend mask 8, C, E), the others block pixel t + i
        const uint8_t blendmask = last_elements(t);
        emitPshufdWithFiller<VEX>(j, r, elements_imm([=](int i) { return borderIndex(planewidth - 4 + t + i, planewidth, bc) - (planewidth - 4); }), blendmask, filler_r, sse_helper_constptr);
      }
    }
  }
  j.L(LabelEnd);
}


struct ExprEval : public jitasm::function<void, ExprEval, uint8_t *, const intptr_t *, intptr_t, intptr_t> {

  std::vector<ExprOp> ops;
  int numInputs;
  int cpuFlags;
  int planeheight;
  int planewidth;
  bool singleMode;
  int labelCount; // to have unique label strings

  std::string getLabelCount()
  {
    return std::to_string(++labelCount);
  }

  ExprEval(std::vector<ExprOp> &ops, int numInputs, int cpuFlags, int planewidth, int planeheight, bool singleMode) : ops(ops), numInputs(numInputs), cpuFlags(cpuFlags),
    planeheight(planeheight), planewidth(planewidth), singleMode(singleMode), labelCount(0) {}

  AVS_FORCEINLINE void doMask(XmmReg &r, Reg &constptr, int _planewidth)
  {
    switch (_planewidth & 3) {
    case 1: andps(r, CPTR(loadmask1000)); break;
    case 2: andps(r, CPTR(loadmask1100)); break;
    case 3: andps(r, CPTR(loadmask1110)); break;
    }
  }

  // Load 8 pixels from [base + offset] and convert them to 2x4 floats
  AVS_FORCEINLINE void load8AsFloat(XmmReg& r1, XmmReg& r2, Reg& base, int offset, int op, XmmReg& zero)
  {
    // widen by unpacks
    if (op == opLoadRelSrc8) {
      movq(r1, mmword_ptr[base + offset]);
      punpcklbw(r1, zero);
      movdqa(r2, r1);
      punpcklwd(r1, zero);
      punpckhwd(r2, zero);
      cvtdq2ps(r1, r1);
      cvtdq2ps(r2, r2);
    }
    else if (op == opLoadRelSrc16) {
      movdqu(r1, xmmword_ptr[base + offset]);
      movdqa(r2, r1);
      punpcklwd(r1, zero);
      punpckhwd(r2, zero);
      cvtdq2ps(r1, r1);
      cvtdq2ps(r2, r2);
    }
    else { // opLoadRelSrcF32
      movdqu(r1, xmmword_ptr[base + offset]);
      movdqu(r2, xmmword_ptr[base + offset + 16]);
    }
  }

  // borderConstant: broadcast the filler value into all 4 elements
  AVS_FORCEINLINE void loadFiller(XmmReg& r, float filler)
  {
    uint32_t bits;
    memcpy(&bits, &filler, sizeof(bits)); // C++17 equivalent of C++20 std::bit_cast<uint32_t>(filler)
    Reg t;
    mov(t, (int32_t)bits);
#ifdef JITASM64
    movq(r, t);
#else
    movd(r, t);
#endif
    pshufd(r, r, 0);
  }

  AVS_FORCEINLINE void relLoad4(XmmReg& r, Reg& a, Reg& regptrs, XmmReg& zero, int op, int dxh, int bc, XmmReg& filler_r, const ExprLoadConv* conv, Reg& constptr)
  {
    emitRelLoad4<false>(*this, r, a, regptrs, &zero, op, dxh, planewidth, bc, filler_r, conv, &constptr);
  }

  // Generates code that loads 8 pixels at x + dx (x = current xcounter, multiple of 8) as floats.
  // Fast path: when no pixel of the group needs border handling, a single 8-pixel load.
  // Otherwise (row edges, a few groups per row) two relLoad4, at dx and dx + 4: the second
  // half's stored pixels are a subset of what relLoad4 assumes as stored, so its rules hold.
  AVS_FORCEINLINE void relLoad8(XmmReg& r1, XmmReg& r2, Reg& a, Reg& regptrs, XmmReg& zero, int op, int dx, int bc, XmmReg& filler_r, const ExprLoadConv* conv, Reg& constptr)
  {
    const int shift = op == opLoadRelSrc8 ? 0 : op == opLoadRelSrc16 ? 1 : 2; // log2(bytes per pixel)
    if (dx == 0) {
      load8AsFloat(r1, r2, a, 0, op, zero); // stored pixels need no border handling
      emitConvert4<false>(*this, r1, &r2, conv);
      return;
    }
    // trivial case is done
    std::string LabelEdge = "relEdge8_" + getLabelCount();
    std::string LabelEnd = "relEnd8_" + getLabelCount();
    Reg newx;
    mov(newx, ptr[regptrs + sizeof(void*) * (RWPTR_START_OF_XCOUNTER)]);
    add(newx, dx);
    // dx < 0: only the left edge
    // dx > 0: only the right edge can be crossed (similar to relLoad4)
    cmp(newx, dx < 0 ? 0 : planewidth - 8);
    if (dx < 0)
      jl(LabelEdge); // newx < 0
    else
      jg(LabelEdge); // newx > planewidth - 8

    // direct load, no border handling needed
    load8AsFloat(r1, r2, a, dx << shift, op, zero);
    emitConvert4<false>(*this, r1, &r2, conv);
    jmp(LabelEnd);

    L(LabelEdge);
    // with border handling
    relLoad4(r1, a, regptrs, zero, op, dx, bc, filler_r, conv, constptr);
    relLoad4(r2, a, regptrs, zero, op, dx + 4, bc, filler_r, conv, constptr);

    L(LabelEnd);
  }


  template<bool processSingle, bool maskUnused>
  AVS_FORCEINLINE void processingLoop(Reg &regptrs, XmmReg &zero, Reg &constptr, Reg &SpatialY)
  {
    std::list<std::pair<XmmReg, XmmReg>> stack;
    std::list<XmmReg> stack1;


    const bool maskIt = (maskUnused && ((planewidth & 3) != 0));

    for (const auto &iter : ops) {
      if (iter.op == opLoadSpatialX) {
        if (processSingle) {
          XmmReg r1;
          movd(r1, dword_ptr[regptrs + sizeof(void *) * (RWPTR_START_OF_XCOUNTER)]);
          shufps(r1, r1, 0);
          cvtdq2ps(r1, r1);
          addps(r1, CPTR(spatialX));
          stack1.push_back(r1);
        }
        else {
          XmmReg r1, r2;
          movd(r1, dword_ptr[regptrs + sizeof(void *) * (RWPTR_START_OF_XCOUNTER)]);
          shufps(r1, r1, 0);
          cvtdq2ps(r1, r1);
          movaps(r2, r1);
          addps(r1, CPTR(spatialX));
          addps(r2, CPTR(spatialX2));
          stack.push_back(std::make_pair(r1, r2));
        }
      }
      else if (iter.op == opLoadSpatialY) {
        if (processSingle) {
          XmmReg r1;
          movd(r1, SpatialY);
          shufps(r1, r1, 0);
          cvtdq2ps(r1, r1);
          stack1.push_back(r1);
        }
        else {
          XmmReg r1, r2;
          movd(r1, SpatialY);
          shufps(r1, r1, 0);
          cvtdq2ps(r1, r1);
          movaps(r2, r1);
          stack.push_back(std::make_pair(r1, r2));
        }
      }
      else if (iter.op == opLoadInternalVar) {
        if (processSingle) {
          XmmReg r1;
          movd(r1, dword_ptr[regptrs + sizeof(void *) * (iter.e.ival + RWPTR_START_OF_INTERNAL_VARIABLES)]);
          shufps(r1, r1, 0);
          stack1.push_back(r1);
        }
        else {
          XmmReg r1, r2;
          movd(r1, dword_ptr[regptrs + sizeof(void *) * (iter.e.ival + RWPTR_START_OF_INTERNAL_VARIABLES)]);
          shufps(r1, r1, 0);
          movaps(r2, r1);
          stack.push_back(std::make_pair(r1, r2));
        }
      }
      else if (iter.op == opLoadFramePropVar) {
        if (processSingle) {
          XmmReg r1;
          movd(r1, dword_ptr[regptrs + sizeof(void*) * (iter.e.ival + RWPTR_START_OF_INTERNAL_FRAMEPROP_VARIABLES)]);
          shufps(r1, r1, 0);
          stack1.push_back(r1);
        }
        else {
          XmmReg r1, r2;
          movd(r1, dword_ptr[regptrs + sizeof(void*) * (iter.e.ival + RWPTR_START_OF_INTERNAL_FRAMEPROP_VARIABLES)]);
          shufps(r1, r1, 0);
          movaps(r2, r1);
          stack.push_back(std::make_pair(r1, r2));
        }
      }
      else if (iter.op == opLoadRelSrc8 || iter.op == opLoadRelSrc16 || iter.op == opLoadRelSrcF32) {
        // either dx or dy is nonzero
        // common part follows for single 4 pixels/cycle and dual 8 pixels/cycle
        Reg a;
        mov(a, ptr[regptrs + sizeof(void *) * (iter.e.ival + RWPTR_START_OF_INPUTS)]); // current pixel group of current line
        emitRowAdjust(*this, a, regptrs, SpatialY, iter.e.ival, iter.dy, iter.bc, planeheight);
        // 'a' points to xcounter in the source row
        if (processSingle) {
          // LoadRel8/16/32, single register mode 1x4 pixels
          XmmReg r1;
          XmmReg filler_r; // borderConstant: filler value in all elements
          if (iter.bc == borderConstant)
            loadFiller(filler_r, iter.filler);
          relLoad4(r1, a, regptrs, zero, iter.op, iter.dx, iter.bc, filler_r, &iter.conv, constptr);
          if (iter.bc == borderConstant && iter.dy != 0) {
            std::string LabelOnScreen = "relRowOn" + getLabelCount();
            emitRowOffscreenCheck(*this, SpatialY, iter.dy, planeheight, LabelOnScreen);
            // this can be skipped by a conditional jump from the function above
            movaps(r1, filler_r); // source row off-screen: all filler
            L(LabelOnScreen);
          }
          stack1.push_back(r1);
        }
        else {
          // LoadRel8/16/32, dual register mode 2x4 pixels
          XmmReg r1, r2;
          XmmReg filler_r; // borderConstant: filler value in all elements
          if (iter.bc == borderConstant)
            loadFiller(filler_r, iter.filler);
          relLoad8(r1, r2, a, regptrs, zero, iter.op, iter.dx, iter.bc, filler_r, &iter.conv, constptr);
          if (iter.bc == borderConstant && iter.dy != 0) {
            std::string LabelOnScreen = "relRowOn" + getLabelCount();
            emitRowOffscreenCheck(*this, SpatialY, iter.dy, planeheight, LabelOnScreen);
            // next two fills can be skipped by a conditional jump from the function above
            movaps(r1, filler_r); // source row off-screen: all filler
            movaps(r2, filler_r);
            L(LabelOnScreen);
          }
          stack.push_back(std::make_pair(r1, r2));
        }
      } // oploadRel8/16/32
      else if (iter.op == opLoadSrc8) {
        if (processSingle) {
          XmmReg r1;
          Reg a;
          mov(a, ptr[regptrs + sizeof(void *) * (iter.e.ival + RWPTR_START_OF_INPUTS)]);
          movd(r1, dword_ptr[a]); // 4 pixels, 4 bytes
          punpcklbw(r1, zero);
          punpcklwd(r1, zero);
          cvtdq2ps(r1, r1);
          if (maskIt)
            doMask(r1, constptr, planewidth);
          stack1.push_back(r1);
        }
        else {
          XmmReg r1, r2;
          Reg a;
          mov(a, ptr[regptrs + sizeof(void *) * (iter.e.ival + RWPTR_START_OF_INPUTS)]);
          movq(r1, mmword_ptr[a]);
          punpcklbw(r1, zero);
          movdqa(r2, r1);
          punpcklwd(r1, zero);
          punpckhwd(r2, zero);
          cvtdq2ps(r1, r1);
          cvtdq2ps(r2, r2);
          if (maskIt)
            doMask(r2, constptr, planewidth);
          stack.push_back(std::make_pair(r1, r2));
        }
      }
      else if (iter.op == opLoadSrc16) {
        if (processSingle) {
          XmmReg r1;
          Reg a;
          mov(a, ptr[regptrs + sizeof(void *) * (iter.e.ival + RWPTR_START_OF_INPUTS)]);
          movq(r1, mmword_ptr[a]); // 4 pixels, 8 bytes
          punpcklwd(r1, zero);
          cvtdq2ps(r1, r1);
          if (maskIt)
            doMask(r1, constptr, planewidth);
          stack1.push_back(r1);
        }
        else {
          XmmReg r1, r2;
          Reg a;
          mov(a, ptr[regptrs + sizeof(void *) * (iter.e.ival + RWPTR_START_OF_INPUTS)]);
          movdqa(r1, xmmword_ptr[a]);
          movdqa(r2, r1);
          punpcklwd(r1, zero);
          punpckhwd(r2, zero);
          cvtdq2ps(r1, r1);
          cvtdq2ps(r2, r2);
          if (maskIt)
            doMask(r2, constptr, planewidth);
          stack.push_back(std::make_pair(r1, r2));
        }
      }
      else if (iter.op == opLoadSrcF32) {
        if (processSingle) {
          XmmReg r1;
          Reg a;
          mov(a, ptr[regptrs + sizeof(void *) * (iter.e.ival + RWPTR_START_OF_INPUTS)]);
          movdqa(r1, xmmword_ptr[a]);
          if (maskIt)
            doMask(r1, constptr, planewidth);
          stack1.push_back(r1);
        }
        else {
          XmmReg r1, r2;
          Reg a;
          mov(a, ptr[regptrs + sizeof(void *) * (iter.e.ival + RWPTR_START_OF_INPUTS)]);
          movdqa(r1, xmmword_ptr[a]);
          movdqa(r2, xmmword_ptr[a + 16]);
          if (maskIt)
            doMask(r2, constptr, planewidth);
          stack.push_back(std::make_pair(r1, r2));
        }
      }
      else if (iter.op == opLoadSrcF16) { // not supported in avs+
        if (processSingle) {
          XmmReg r1;
          Reg a;
          mov(a, ptr[regptrs + sizeof(void *) * (iter.e.ival + RWPTR_START_OF_INPUTS)]);
          vcvtph2ps(r1, qword_ptr[a]);
          if (maskIt)
            doMask(r1, constptr, planewidth);
          stack1.push_back(r1);
        }
        else {
          XmmReg r1, r2;
          Reg a;
          mov(a, ptr[regptrs + sizeof(void *) * (iter.e.ival + RWPTR_START_OF_INPUTS)]);
          vcvtph2ps(r1, qword_ptr[a]);
          vcvtph2ps(r2, qword_ptr[a + 8]);
          if (maskIt)
            doMask(r2, constptr, planewidth);
          stack.push_back(std::make_pair(r1, r2));
        }
      }
      else if (iter.op == opLoadVar) {
        if (processSingle) {
          XmmReg r1;
          // 16 bytes/variable
          int offset = sizeof(void *) * RWPTR_START_OF_USERVARIABLES + 16 * iter.e.ival;
          movdqa(r1, xmmword_ptr[regptrs + offset]);
          if (maskIt)
            doMask(r1, constptr, planewidth);
          stack1.push_back(r1);
        }
        else {
          XmmReg r1, r2;
          // 32 bytes/variable
          int offset = sizeof(void *) * RWPTR_START_OF_USERVARIABLES + 32 * iter.e.ival;
          movdqa(r1, xmmword_ptr[regptrs + offset]);
          movdqa(r2, xmmword_ptr[regptrs + offset + 16]);
          if (maskIt)
            doMask(r2, constptr, planewidth);
          stack.push_back(std::make_pair(r1, r2));
        }
      }
      else if (iter.op == opLoadConst) {
        if (processSingle) {
          XmmReg r1;
          Reg a;
          mov(a, iter.e.ival);
          movd(r1, a);
          shufps(r1, r1, 0);
          if (maskIt)
            doMask(r1, constptr, planewidth);
          stack1.push_back(r1);
        }
        else {
          XmmReg r1, r2;
          Reg a;
          mov(a, iter.e.ival);
          movd(r1, a);
          shufps(r1, r1, 0);
          movaps(r2, r1);
          if (maskIt)
            doMask(r2, constptr, planewidth);
          stack.push_back(std::make_pair(r1, r2));
        }
      }
      else if (iter.op == opDup) {
        if (processSingle) {
          auto p = std::next(stack1.rbegin(), iter.e.ival);
          XmmReg r1;
          movaps(r1, *p);
          stack1.push_back(r1);
        }
        else {
          auto p = std::next(stack.rbegin(), iter.e.ival);
          XmmReg r1, r2;
          movaps(r1, p->first);
          movaps(r2, p->second);
          stack.push_back(std::make_pair(r1, r2));
        }
      }
      else if (iter.op == opSwap) {
        if (processSingle) {
          std::swap(stack1.back(), *std::next(stack1.rbegin(), iter.e.ival));
        }
        else {
          std::swap(stack.back(), *std::next(stack.rbegin(), iter.e.ival));
        }
      }
      else if (iter.op == opAdd) {
        if (processSingle) {
          TwoArgOp_Single(addps)
        }
        else {
          TwoArgOp(addps)
        }
      }
      else if (iter.op == opSub) {
        if (processSingle) {
          TwoArgOp_Single(subps)
        }
        else {
          TwoArgOp(subps)
        }
      }
      else if (iter.op == opMul) {
        if (processSingle) {
          TwoArgOp_Single(mulps)
        }
        else {
          TwoArgOp(mulps)
        }
      }
      else if (iter.op == opDiv) {
        if (processSingle) {
          TwoArgOp_Single(divps)
        }
        else {
          TwoArgOp(divps)
        }
      }
      else if (iter.op == opFmod) {
        if (processSingle) {
          auto t1 = stack1.back(); // despite the Intel compiler warnings, this is intentionally a copy and not a reference (valid for all such messages as well)
          stack1.pop_back();
          auto &t2 = stack1.back();
          FMOD_PS(t2, t1)
        }
        else {
          auto t1 = stack.back();
          stack.pop_back();
          auto &t2 = stack.back();
          FMOD_PS(t2.first, t1.first)
          FMOD_PS(t2.second, t1.second)
        }
      }
      else if (iter.op == opMax) {
        if (processSingle) {
          TwoArgOp_Single(maxps)
        }
        else {
          TwoArgOp(maxps)
        }
      }
      else if (iter.op == opMin) {
        if (processSingle) {
          TwoArgOp_Single(minps)
        }
        else {
          TwoArgOp(minps)
        }
      }
      else if (iter.op == opSqrt) {
        if (processSingle) {
          auto &t1 = stack1.back();
          maxps(t1, zero);
          sqrtps(t1, t1);
        }
        else {
          auto &t1 = stack.back();
          maxps(t1.first, zero);
          maxps(t1.second, zero);
          sqrtps(t1.first, t1.first);
          sqrtps(t1.second, t1.second);
        }
      }
      // Integer store operations: Why sometimes C version differs from SSE: convert to int from .5 intermediates.
      // Simd version of float -> int32 (cvtps2dq) is using the SSE rounding mode "round to nearest"
      // C version is using: (uint8_t)(f + 0.5f) which turnes into cvttps, typecast to int uses trunc
      // Even for positive numbers they are not the same, when converting occurs exactly from the halfway
      // SSE is using Banker's rounding, which rounds to the nearest _even_ integer value. https://en.wikipedia.org/wiki/IEEE_754#Roundings_to_nearest
      //        C   SSE
      //  0.5   1   0
      //  1.5   2   2
      //  2.5   3   2
      //  3.5   4   4
      // 3.7.1test27: no more banker's rounding. Using cvttps and +0.5f
      else if (iter.op == opStore8) {
        if (processSingle) {
          auto t1 = stack1.back();
          stack1.pop_back();
          XmmReg r1;
          Reg a;
          addps(t1, CPTR(elfloat_half)); // rounder for truncate! no banker's rounding
          maxps(t1, zero);
          minps(t1, CPTR(elstore8));
          mov(a, ptr[regptrs]);
          cvttps2dq(t1, t1);    // 00 w3 00 w2 00 w1 00 w0 -- min/max clamp ensures that high words are zero
          packssdw(t1, zero);  // _mm_packs_epi32: w7 w6 w5 w4 w3 w2 w1 w0
          packuswb(t1, zero);  // _mm_packus_epi16: 0 0 0 0 0 0 0 0 b7 b6 b5 b4 b3 b2 b1 b0
          movd(dword_ptr[a], t1);
        }
        else {
          auto t1 = stack.back();
          stack.pop_back();
          XmmReg r1, r2;
          Reg a;
          addps(t1.first, CPTR(elfloat_half)); // rounder for truncate! no banker's rounding
          maxps(t1.first, zero);
          addps(t1.second, CPTR(elfloat_half)); // rounder for truncate! no banker's rounding
          maxps(t1.second, zero);
          minps(t1.first, CPTR(elstore8));
          minps(t1.second, CPTR(elstore8));
          mov(a, ptr[regptrs]);
          cvttps2dq(t1.first, t1.first);    // 00 w3 00 w2 00 w1 00 w0 -- min/max clamp ensures that high words are zero
          cvttps2dq(t1.second, t1.second);  // 00 w7 00 w6 00 w5 00 w4
          // t1.second is the lo
          packssdw(t1.first, t1.second);   // _mm_packs_epi32: w7 w6 w5 w4 w3 w2 w1 w0
          packuswb(t1.first, zero);       // _mm_packus_epi16: 0 0 0 0 0 0 0 0 b7 b6 b5 b4 b3 b2 b1 b0
          movq(mmword_ptr[a], t1.first);
        }
      }
      else if (iter.op == opStore10 // avs+
        || iter.op == opStore12 // avs+
        || iter.op == opStore14 // avs+
        || iter.op == opStore16
        ) {
        if (processSingle) {
          auto t1 = stack1.back();
          stack1.pop_back();
          XmmReg r1;
          Reg a;
          addps(t1, CPTR(elfloat_half)); // rounder for truncate! no banker's rounding
          maxps(t1, zero);
          switch (iter.op) {
          case opStore10:
            minps(t1, CPTR(elstore10));
            break;
          case opStore12:
            minps(t1, CPTR(elstore12));
            break;
          case opStore14:
            minps(t1, CPTR(elstore14));
            break;
          case opStore16:
            minps(t1, CPTR(elstore16));
            break;
          }
          mov(a, ptr[regptrs]);
          cvttps2dq(t1, t1); // no cvtps, but cvttps
          // new
          switch (iter.op) {
          case opStore10:
          case opStore12:
          case opStore14:
            packssdw(t1, zero);   // _mm_packs_epi32: w7 w6 w5 w4 w3 w2 w1 w0
            break;
          case opStore16:
            if (cpuFlags & CPUF_SSE4_1) {
              packusdw(t1, zero);   // _mm_packus_epi32: w7 w6 w5 w4 w3 w2 w1 w0
            }
            else {
              // old, sse2
              movdqa(r1, t1);   // 00 w3 00 w2 00 w1 00 w0 -- min/max clamp ensures that high words are zero
              psrldq(t1, 6);
              por(t1, r1);
              pshuflw(t1, t1, 0b11011000);
              punpcklqdq(t1, zero);
            }
            break;
          }
          movq(mmword_ptr[a], t1);
        }
        else {
          auto t1 = stack.back();
          stack.pop_back();
          XmmReg r1, r2;
          Reg a;
          addps(t1.first, CPTR(elfloat_half)); // rounder for truncate! no banker's rounding
          maxps(t1.first, zero);
          addps(t1.second, CPTR(elfloat_half)); // rounder for truncate! no banker's rounding
          maxps(t1.second, zero);
          switch (iter.op) {
          case opStore10:
            minps(t1.first, CPTR(elstore10));
            minps(t1.second, CPTR(elstore10));
            break;
          case opStore12:
            minps(t1.first, CPTR(elstore12));
            minps(t1.second, CPTR(elstore12));
            break;
          case opStore14:
            minps(t1.first, CPTR(elstore14));
            minps(t1.second, CPTR(elstore14));
            break;
          case opStore16:
            minps(t1.first, CPTR(elstore16));
            minps(t1.second, CPTR(elstore16));
            break;
          }
          mov(a, ptr[regptrs]);
          cvttps2dq(t1.first, t1.first); // no bankers rounding
          cvttps2dq(t1.second, t1.second);
          // new
          switch (iter.op) {
          case opStore10:
          case opStore12:
          case opStore14:
            packssdw(t1.first, t1.second);   // _mm_packs_epi32: w7 w6 w5 w4 w3 w2 w1 w0
            break;
          case opStore16:
            if (cpuFlags & CPUF_SSE4_1) {
              packusdw(t1.first, t1.second);   // _mm_packus_epi32: w7 w6 w5 w4 w3 w2 w1 w0
            }
            else {
              // old, sse2
              movdqa(r1, t1.first);   // 00 w3 00 w2 00 w1 00 w0 -- min/max clamp ensures that high words are zero
              movdqa(r2, t1.second);  // 00 w7 00 w6 00 w5 00 w4
              psrldq(t1.first, 6);
              psrldq(t1.second, 6);
              por(t1.first, r1);
              por(t1.second, r2);
              pshuflw(t1.first, t1.first, 0b11011000);
              pshuflw(t1.second, t1.second, 0b11011000);
              punpcklqdq(t1.first, t1.second);
            }
            break;
          }
          movdqa(xmmword_ptr[a], t1.first);
        }
      }
      else if (iter.op == opStoreF32) {
        if (processSingle) {
          auto t1 = stack1.back();
          stack1.pop_back();
          Reg a;
          mov(a, ptr[regptrs]);
          movaps(xmmword_ptr[a], t1);
        }
        else {
          auto t1 = stack.back();
          stack.pop_back();
          Reg a;
          mov(a, ptr[regptrs]);
          movaps(xmmword_ptr[a], t1.first);
          movaps(xmmword_ptr[a + 16], t1.second);
        }
      }
      else if (iter.op == opStoreF16) { // not supported in avs+
        if (processSingle) {
          auto t1 = stack1.back();
          stack1.pop_back();
          Reg a;
          mov(a, ptr[regptrs]);
          vcvtps2ph(qword_ptr[a], t1, 0);
        }
        else {
          auto t1 = stack.back();
          stack.pop_back();
          Reg a;
          mov(a, ptr[regptrs]);
          vcvtps2ph(qword_ptr[a], t1.first, 0);
          vcvtps2ph(qword_ptr[a + 8], t1.second, 0);
        }
      }
      else if (iter.op == opStoreVar || iter.op == opStoreVarAndDrop1) {
        if (processSingle) {
          auto t1 = stack1.back();
          // 16 bytes/variable
          int offset = sizeof(void *) * RWPTR_START_OF_USERVARIABLES + 16 * iter.e.ival;
          movaps(xmmword_ptr[regptrs + offset], t1);
          if (iter.op == opStoreVarAndDrop1)
            stack1.pop_back();
        }
        else {
          auto t1 = stack.back();
          // 32 byte/variable
          int offset = sizeof(void *) * RWPTR_START_OF_USERVARIABLES + 32 * iter.e.ival;
          movaps(xmmword_ptr[regptrs + offset], t1.first);
          movaps(xmmword_ptr[regptrs + offset + 16], t1.second);
          if (iter.op == opStoreVarAndDrop1)
            stack.pop_back();
        }
      }
      else if (iter.op == opAbs) {
        if (processSingle) {
          auto &t1 = stack1.back();
          andps(t1, CPTR(elabsmask));
        }
        else {
          auto &t1 = stack.back();
          andps(t1.first, CPTR(elabsmask));
          andps(t1.second, CPTR(elabsmask));
        }
      }
      else if (iter.op == opSgn) {
        // 1, 0, -1
        /*
          __m128 sgn(__m128 value) {
            const __m128 zero = _mm_set_ps1 (0.0f);
            __m128 p = _mm_and_ps(_mm_cmpgt_ps(value, zero), _mm_set_ps1(1.0f));
            __m128 n = _mm_and_ps(_mm_cmplt_ps(value, zero), _mm_set_ps1(-1.0f));
            return _mm_or_ps(p, n);
          }
        */
        if (processSingle) {
          auto &t1 = stack1.back();
          XmmReg r1, r2;
          xorps(r2, r2);
          movaps(r1, t1);
          movaps(t1, r2);
          cmpltps(t1, r1);
          cmpltps(r1, r2);
          andps(t1, CPTR(elfloat_one));
          andps(r1, CPTR(elfloat_minusone));
          orps(t1, r1);
        }
        else {
          auto &t1 = stack.back();
          XmmReg r2, r3, r4, r5;
          xorps(r2, r2);
          xorps(r3, r3);
          cmpltps(r3, t1.first);
          cmpltps(t1.first, r2);
          movaps(r4, t1.first);
          andnps(r4, r3);
          movaps(r3, CPTR(elfloat_one));
          xorps(r5, r5);
          cmpltps(r5, t1.second);
          cmpltps(t1.second, r2);
          movaps(r2, CPTR(elfloat_minusone));
          andps(t1.first, r2);
          andps(r2, t1.second);
          andnps(t1.second, r5);
          andps(r4, r3);
          orps(t1.first, r4);
          andps(t1.second, r3);
          orps(t1.second, r2);
        }
      }
      else if (iter.op == opNeg) {
        if (processSingle) {
          auto &t1 = stack1.back();
          cmpleps(t1, zero);
          andps(t1, CPTR(elfloat_one));
        }
        else {
          auto &t1 = stack.back();
          cmpleps(t1.first, zero);
          cmpleps(t1.second, zero);
          andps(t1.first, CPTR(elfloat_one));
          andps(t1.second, CPTR(elfloat_one));
        }
      }
      else if (iter.op == opNegSign) {
        if (processSingle) {
          auto& t1 = stack1.back();
          xorps(t1, CPTR(elsignmask));
        }
        else {
          auto& t1 = stack.back();
          xorps(t1.first, CPTR(elsignmask));
          xorps(t1.second, CPTR(elsignmask));
        }
      }
      else if (iter.op == opAnd) {
        if (processSingle) {
          LogicOp_Single(andps)
        }
        else {
          LogicOp(andps)
        }
      }
      else if (iter.op == opOr) {
        if (processSingle) {
          LogicOp_Single(orps)
        }
        else {
          LogicOp(orps)
        }
      }
      else if (iter.op == opXor) {
        if (processSingle) {
          LogicOp_Single(xorps)
        }
        else {
          LogicOp(xorps)
        }
      }
      else if (iter.op == opGt) { // a > b (gt) -> b < (lt) a
        if (processSingle) {
          CmpOp_Single(cmpltps)
        }
        else {
          CmpOp(cmpltps)
        }
      }
      else if (iter.op == opLt) { // a < b (lt) -> b > (gt,nle) a
        if (processSingle) {
          CmpOp_Single(cmpnleps)
        }
        else {
          CmpOp(cmpnleps)
        }
      }
      else if (iter.op == opEq) { // a == b -> b == a
        if (processSingle) {
          CmpOp_Single(cmpeqps)
        }
        else {
          CmpOp(cmpeqps)
        }
      }
      else if (iter.op == opNotEq) { // a != b
        if (processSingle) {
          CmpOp_Single(cmpneqps)
        }
        else {
          CmpOp(cmpneqps)
        }
      }
      else if (iter.op == opLE) { // a <= b -> b >= (ge,nlt) a
        if (processSingle) {
          CmpOp_Single(cmpnltps)
        }
        else {
          CmpOp(cmpnltps)
        }
      }
      else if (iter.op == opGE) { // a >= b -> b <= (le) a
        if (processSingle) {
          CmpOp_Single(cmpleps)
        }
        else {
          CmpOp(cmpleps)
        }
      }
      else if (iter.op == opTernary) {
        if (processSingle) {
          auto t1 = stack1.back();
          stack1.pop_back();
          auto t2 = stack1.back();
          stack1.pop_back();
          auto t3 = stack1.back();
          stack1.pop_back();
          XmmReg r1;
          xorps(r1, r1);
          cmpltps(r1, t3);
          andps(t2, r1);
          andnps(r1, t1);
          orps(r1, t2);
          stack1.push_back(r1);
        }
        else {
          auto t1 = stack.back();
          stack.pop_back();
          auto t2 = stack.back();
          stack.pop_back();
          auto t3 = stack.back();
          stack.pop_back();
          XmmReg r1, r2;
          xorps(r1, r1);
          xorps(r2, r2);
          cmpltps(r1, t3.first);
          cmpltps(r2, t3.second);
          andps(t2.first, r1);
          andps(t2.second, r2);
          andnps(r1, t1.first);
          andnps(r2, t1.second);
          orps(r1, t2.first);
          orps(r2, t2.second);
          stack.push_back(std::make_pair(r1, r2));
        }
      }
      else if (iter.op == opExp) {
        if (processSingle) {
          auto &t1 = stack1.back();
          EXP_PS(t1)
        }
        else {
          auto &t1 = stack.back();
          EXP_PS(t1.first)
            EXP_PS(t1.second)
        }
      }
      else if (iter.op == opLog) {
        if (processSingle) {
          auto &t1 = stack1.back();
          LOG_PS(t1)
        }
        else {
          auto &t1 = stack.back();
          LOG_PS(t1.first)
            LOG_PS(t1.second)
        }
      }
      else if (iter.op == opPow) {
        if (processSingle) {
          auto t1 = stack1.back();
          stack1.pop_back();
          auto &t2 = stack1.back();
          LOG_PS(t2)
            mulps(t2, t1);
          EXP_PS(t2)
        }
        else {
          auto t1 = stack.back();
          stack.pop_back();
          auto &t2 = stack.back();
          LOG_PS(t2.first)
            mulps(t2.first, t1.first);
          EXP_PS(t2.first)
            LOG_PS(t2.second)
            mulps(t2.second, t1.second);
          EXP_PS(t2.second)
        }
      }
      else if (iter.op == opSin) {
        if (processSingle) {
          auto& _t1 = stack1.back();
          SINCOS_PS(true, _t1, _t1)
        }
        else {
          auto& _t1 = stack.back();
          SINCOS_PS(true, _t1.first, _t1.first);
          SINCOS_PS(true, _t1.second, _t1.second);
        }
      }
      else if (iter.op == opCos) {
        if (processSingle) {
          auto& _t1 = stack1.back();
          SINCOS_PS(false, _t1, _t1)
        }
        else {
          auto& _t1 = stack.back();
          SINCOS_PS(false, _t1.first, _t1.first);
          SINCOS_PS(false, _t1.second, _t1.second);
        }
      }
      else if (iter.op == opTan) {
        if (processSingle) {
          auto& t1 = stack1.back();
          TAN_PS(t1)
        }
        else {
          auto& t1 = stack.back();
          TAN_PS(t1.first);
          TAN_PS(t1.second);
        }
      }
      else if (iter.op == opAtan2) {
        if (processSingle) {
          auto t1 = stack1.back();
          stack1.pop_back();
          auto& t2 = stack1.back();
          ATAN2_PS(t2, t1);
        }
        else {
          auto t1 = stack.back();
          stack.pop_back();
          auto& t2 = stack.back();
          ATAN2_PS(t2.first, t1.first);
          ATAN2_PS(t2.second, t1.second);
        }
      }
      else if (iter.op == opClip) {
        // clip(a, low, high) = min(max(a, low),high)
        if (processSingle) {
          auto t1 = stack1.back();
          stack1.pop_back();
          auto t2 = stack1.back();
          stack1.pop_back();
          auto &t3 = stack1.back();
          maxps(t3, t2);
          minps(t3, t1);
        }
        else {
          auto t1 = stack.back();
          stack.pop_back();
          auto t2 = stack.back();
          stack.pop_back();
          auto &t3 = stack.back();
          maxps(t3.first, t2.first);
          minps(t3.first, t1.first);
          maxps(t3.second, t2.second);
          minps(t3.second, t1.second);
        }
      }
      else if (iter.op == opRound || iter.op == opFloor || iter.op == opCeil || iter.op == opTrunc) {
         const int rounder_flag =
           (iter.op == opRound) ? (FROUND_TO_NEAREST_INT | FROUND_NO_EXC) :
           (iter.op == opFloor) ? (FROUND_TO_NEG_INF | FROUND_NO_EXC) :
           (iter.op == opCeil) ? (FROUND_TO_POS_INF | FROUND_NO_EXC) :
           (FROUND_TO_ZERO | FROUND_NO_EXC); // opTrunc
        if (processSingle) {
          auto& t1 = stack1.back();
          roundps(t1, t1, rounder_flag);
        }
        else {
          auto& t1 = stack.back();
          roundps(t1.first, t1.first, rounder_flag);
          roundps(t1.second, t1.second, rounder_flag);
        }
      }

    }
  }

    void main(Reg regptrs, Reg regoffs, Reg niter, Reg SpatialY)
    {
        XmmReg zero;
        pxor(zero, zero);
        Reg constptr;
        mov(constptr, (uintptr_t)logexpconst);

        L("wloop");
        cmp(niter, 0);
        je("wend");
        //sub(niter, 1);
        dec(niter);

        // process two sets, no partial input masking
        if (singleMode)
          processingLoop<true, false>(regptrs, zero, constptr, SpatialY);
        else
          processingLoop<false, false>(regptrs, zero, constptr, SpatialY);

        const int EXTRA = 2; // output pointer, xcounter
        if constexpr(sizeof(void *) == 8) {
            int numIter = (numInputs + EXTRA + 1) / 2;
            for (int i = 0; i < numIter; i++) {
                XmmReg r1, r2;
                movdqu(r1, xmmword_ptr[regptrs + 16 * i]);
                movdqu(r2, xmmword_ptr[regoffs + 16 * i]);
                paddq(r1, r2);
                movdqu(xmmword_ptr[regptrs + 16 * i], r1);
            }
        } else {
            int numIter = (numInputs + EXTRA + 3) / 4;
            for (int i = 0; i < numIter; i++) {
                XmmReg r1, r2;
                movdqu(r1, xmmword_ptr[regptrs + 16 * i]);
                movdqu(r2, xmmword_ptr[regoffs + 16 * i]);
                paddd(r1, r2);
                movdqu(xmmword_ptr[regptrs + 16 * i], r1);
            }
        }

        jmp("wloop");
        L("wend");

        int nrestpixels = planewidth & (singleMode ? 3 : 7);
        if (nrestpixels > 4) // dual process with masking
          processingLoop<false, true>(regptrs, zero, constptr, SpatialY);
        else if (nrestpixels == 4) // single process, no masking
          processingLoop<true, false>(regptrs, zero, constptr, SpatialY);
        else if (nrestpixels > 0) // single process, masking
          processingLoop<true, true>(regptrs, zero, constptr, SpatialY);
    }
};

// avx2 evaluator with two ymm registers
struct ExprEvalAvx2 : public jitasm::function<void, ExprEvalAvx2, uint8_t *, const intptr_t *, intptr_t, intptr_t> {

  std::vector<ExprOp> ops;
  int numInputs;
  int cpuFlags;
  int planewidth; // original, lut can overwrite
  int planeheight;
  bool singleMode;
  int labelCount; // to have unique label strings

  std::string getLabelCount()
  {
    return std::to_string(++labelCount);
  }

  ExprEvalAvx2(std::vector<ExprOp> &ops, int numInputs, int cpuFlags, int planewidth, int planeheight, bool singleMode) : ops(ops), numInputs(numInputs), cpuFlags(cpuFlags),
    planewidth(planewidth), planeheight(planeheight), singleMode(singleMode), labelCount(0) {}

  // Load 8 pixels from [base + offset] and convert them to 8 floats
  AVS_FORCEINLINE void load8AsFloat(YmmReg& r, Reg& base, int offset, int op)
  {
    if (op == opLoadRelSrc8) {
      // 8->32 bits like _mm256_cvtepu8_epi32
      vpmovzxbd(r, qword_ptr[base + offset]);
      vcvtdq2ps(r, r);
    }
    else if (op == opLoadRelSrc16) {
      // 16->32 bit like _mm256_cvtepu16_epi32
      vpmovzxwd(r, xmmword_ptr[base + offset]);
      vcvtdq2ps(r, r);
    }
    else // opLoadRelSrcF32
      vmovdqu(r, ymmword_ptr[base + offset]);
  }

  // borderConstant: scale_inputs conversion of the 8 loaded pixels in r (and r2, when given),
  // see emitConvert4. No code when there is no conversion.
  AVS_FORCEINLINE void convert8(YmmReg& r, YmmReg* r2, const ExprLoadConv* conv)
  {
    if (conv == nullptr || !conv->active())
      return;
    enum { Sub, Mul, Add };
    auto step = [&](float value, int kind) {
      uint32_t bits;
      memcpy(&bits, &value, sizeof(bits)); // C++17 equivalent of C++20 std::bit_cast<uint32_t>(value)
      Reg32 t;
      XmmReg cx;
      YmmReg c;
      mov(t, (int32_t)bits);
      vmovd(cx, t);
      vbroadcastss(c, cx);
      for (YmmReg* x : { &r, r2 }) {
        if (x == nullptr)
          continue;
        if (kind == Sub) vsubps(*x, *x, c);
        else if (kind == Mul) vmulps(*x, *x, c);
        else vaddps(*x, *x, c);
      }
    };
    if (conv->sub != 0.0f) step(conv->sub, Sub);
    if (conv->mul != 1.0f) step(conv->mul, Mul);
    if (conv->add != 0.0f) step(conv->add, Add);
    if (conv->add_uv != 0.0f) step(conv->add_uv, Add);
  }

  // borderConstant: broadcast the filler value into all 4 elements (VEX)
  AVS_FORCEINLINE void loadFiller(XmmReg& r, float filler)
  {
    uint32_t bits;
    memcpy(&bits, &filler, sizeof(bits)); // C++17 equivalent of C++20 std::bit_cast<uint32_t>(filler)
    Reg t;
    mov(t, (int32_t)bits);
#ifdef JITASM64
    vmovq(r, t);
#else
    vmovd(r, t);
#endif
    vpshufd(r, r, 0);
  }

  // Generates code that loads 8 pixels at x + dx (x = current xcounter, multiple of 8) as floats,
  // off-screen positions handled by border mode bc (see emitRelLoad4).
  // Fast path: when no pixel of the group is off-screen, a single 8-pixel load.
  // Otherwise (row edges, a few groups per row) emitRelLoad4 (VEX) at dx and dx + 4, combined
  // into r: the second half's stored pixels are a subset of what emitRelLoad4 assumes as stored.
  // borderConstant: conv is the scale_inputs conversion of the loaded pixels (before the filler)
  AVS_FORCEINLINE void relLoad8(YmmReg& r, Reg& a, Reg& regptrs, YmmReg& zero, int op, int dx, int bc, XmmReg& filler_r, const ExprLoadConv* conv)
  {
    const int shift = op == opLoadRelSrc8 ? 0 : op == opLoadRelSrc16 ? 1 : 2; // log2(bytes per pixel)
    if (dx == 0) {
      load8AsFloat(r, a, 0, op); // stored pixels need no border handling
      convert8(r, nullptr, conv);
      return;
    }
    std::string LabelEdge = "relEdge8_" + getLabelCount();
    std::string LabelEnd = "relEnd8_" + getLabelCount();
    Reg newx;
    mov(newx, ptr[regptrs + sizeof(void*) * (RWPTR_START_OF_XCOUNTER)]);
    add(newx, dx);
    // dx < 0: only the left edge
    // dx > 0: only the right edge can be crossed (see emitRelLoad4)
    cmp(newx, dx < 0 ? 0 : planewidth - 8);
    if (dx < 0)
      jl(LabelEdge); // newx < 0
    else
      jg(LabelEdge); // newx > planewidth - 8
    load8AsFloat(r, a, dx << shift, op);
    convert8(r, nullptr, conv);
    jmp(LabelEnd);
    L(LabelEdge);
    XmmReg lo, hi;
    emitRelLoad4<true>(*this, lo, a, regptrs, nullptr, op, dx, planewidth, bc, filler_r, conv, nullptr);
    emitRelLoad4<true>(*this, hi, a, regptrs, nullptr, op, dx + 4, planewidth, bc, filler_r, conv, nullptr);
    vinsertf128(r, zero, lo, 0);
    vinsertf128(r, r, hi, 1);
    L(LabelEnd);
  }

  // Generates code that loads 16 pixels at x + dx (x = current xcounter, multiple of 16) as floats.
  // Fast path: a single check for the whole group; otherwise relLoad8 at dx and dx + 8.
  AVS_FORCEINLINE void relLoad16(YmmReg& r1, YmmReg& r2, Reg& a, Reg& regptrs, YmmReg& zero, int op, int dx, int bc, XmmReg& filler_r, const ExprLoadConv* conv)
  {
    const int shift = op == opLoadRelSrc8 ? 0 : op == opLoadRelSrc16 ? 1 : 2; // log2(bytes per pixel)
    if (dx == 0) {
      load8AsFloat(r1, a, 0, op); // stored pixels need no border handling
      load8AsFloat(r2, a, 8 << shift, op);
      convert8(r1, &r2, conv);
      return;
    }
    // trivial case is done
    std::string LabelEdge = "relEdge16_" + getLabelCount();
    std::string LabelEnd = "relEnd16_" + getLabelCount();
    Reg newx;
    mov(newx, ptr[regptrs + sizeof(void*) * (RWPTR_START_OF_XCOUNTER)]);
    add(newx, dx);
    cmp(newx, dx < 0 ? 0 : planewidth - 16);
    if (dx < 0)
      jl(LabelEdge); // newx < 0
    else
      jg(LabelEdge); // newx > planewidth - 16
    // no edge handling needed
    load8AsFloat(r1, a, dx << shift, op);
    load8AsFloat(r2, a, (dx + 8) << shift, op);
    convert8(r1, &r2, conv);
    jmp(LabelEnd);

    L(LabelEdge);
    // edge handling needed
    relLoad8(r1, a, regptrs, zero, op, dx, bc, filler_r, conv);
    relLoad8(r2, a, regptrs, zero, op, dx + 8, bc, filler_r, conv);
    L(LabelEnd);
  }

  template<bool processSingle, bool maskUnused>
  AVS_FORCEINLINE void processingLoop(Reg &regptrs, YmmReg &zero, Reg &constptr, Reg &SpatialY)
  {
    std::list<std::pair<YmmReg, YmmReg>> stack;
    std::list<YmmReg> stack1;

    // reason of masking: prevent loading 'junk', out of frame pixels, which can be NaN floats for example.
    // If processingLoop works in dual register mode (!processSingle), masking occurs only for the high register,
    // when there is no need for dual registers (width mod 16 is <= 8 pixels), processSingle=true is used
    const bool maskIt = maskUnused && ((planewidth & 7) != 0);
    const int mask = ((1 << (planewidth & 7)) - 1);

    // mask by zero when we have only 1-7 valid pixels in the lower (single register) or upper (dual register)
    // 1: 2-1   = 1   // 00000001
    // 2: 4-1   = 3   // 00000011
    // 7: 128-1 = 127 // 01111111

    for (const auto &iter : ops) {
      if (iter.op == opLoadSpatialX) {
        if (processSingle) {
          YmmReg r1;
          XmmReg r1x;
          vmovd(r1x, dword_ptr[regptrs + sizeof(void *) * (RWPTR_START_OF_XCOUNTER)]);
          vcvtdq2ps(r1x, r1x);
          vbroadcastss(r1, r1x);
          vaddps(r1, r1, CPTR_AVX(spatialX));
          stack1.push_back(r1);
        }
        else {
          YmmReg r1, r2;
          XmmReg r1x;
          vmovd(r1x, dword_ptr[regptrs + sizeof(void *) * (RWPTR_START_OF_XCOUNTER)]);
          vcvtdq2ps(r1x, r1x);
          vbroadcastss(r1, r1x);
          vmovaps(r2, r1);
          vaddps(r1, r1, CPTR_AVX(spatialX));
          vaddps(r2, r2, CPTR_AVX(spatialX2));
          stack.push_back(std::make_pair(r1, r2));
        }
      }
      else if (iter.op == opLoadSpatialY) {
        if (processSingle) {
          YmmReg r1;
          XmmReg r1x;
#ifdef JITASM64
          vmovq(r1x, SpatialY);
#else
          vmovd(r1x, SpatialY);
#endif
          vcvtdq2ps(r1x, r1x);
          vbroadcastss(r1, r1x);
          stack1.push_back(r1);
        }
        else {
          YmmReg r1, r2;
          XmmReg r1x;
#ifdef JITASM64
          vmovq(r1x, SpatialY);
#else
          vmovd(r1x, SpatialY);
#endif
          vcvtdq2ps(r1x, r1x);
          vbroadcastss(r1, r1x);
          vmovaps(r2, r1);
          stack.push_back(std::make_pair(r1, r2));
        }
      }
      else if (iter.op == opLoadInternalVar) {
        if (processSingle) {
          YmmReg r1;
          XmmReg r1x;
          vmovd(r1x, dword_ptr[regptrs + sizeof(void *) * (iter.e.ival + RWPTR_START_OF_INTERNAL_VARIABLES)]);
          vbroadcastss(r1, r1x);
          stack1.push_back(r1);
        }
        else {
          YmmReg r1, r2;
          XmmReg r1x;
          vmovd(r1x, dword_ptr[regptrs + sizeof(void *) * (iter.e.ival + RWPTR_START_OF_INTERNAL_VARIABLES)]);
          vbroadcastss(r1, r1x);
          vmovaps(r2, r1);
          stack.push_back(std::make_pair(r1, r2));
        }
      }
      else if (iter.op == opLoadFramePropVar) {
        if (processSingle) {
          YmmReg r1;
          XmmReg r1x;
          vmovd(r1x, dword_ptr[regptrs + sizeof(void*) * (iter.e.ival + RWPTR_START_OF_INTERNAL_FRAMEPROP_VARIABLES)]);
          vbroadcastss(r1, r1x);
          stack1.push_back(r1);
        }
        else {
          YmmReg r1, r2;
          XmmReg r1x;
          vmovd(r1x, dword_ptr[regptrs + sizeof(void*) * (iter.e.ival + RWPTR_START_OF_INTERNAL_FRAMEPROP_VARIABLES)]);
          vbroadcastss(r1, r1x);
          vmovaps(r2, r1);
          stack.push_back(std::make_pair(r1, r2));
        }
      }
      else if (iter.op == opLoadRelSrc8 || iter.op == opLoadRelSrc16 || iter.op == opLoadRelSrcF32) {
        // either dx or dy is nonzero
        Reg a;
        mov(a, ptr[regptrs + sizeof(void *) * (iter.e.ival + RWPTR_START_OF_INPUTS)]); // current pixel group of current line

        // handles y shift and vertical border repeat/reflextions
        emitRowAdjust(*this, a, regptrs, SpatialY, iter.e.ival, iter.dy, iter.bc, planeheight);
        // 'a' now points to xcounter in the source row
        if (processSingle) {
          // 8 pixels
          YmmReg r1;
          XmmReg filler_r; // borderConstant: filler value in all elements
          if (iter.bc == borderConstant)
            loadFiller(filler_r, iter.filler);
          relLoad8(r1, a, regptrs, zero, iter.op, iter.dx, iter.bc, filler_r, &iter.conv);
          if (iter.bc == borderConstant && iter.dy != 0) {
            std::string LabelOnScreen = "relRowOn" + getLabelCount();
            emitRowOffscreenCheck(*this, SpatialY, iter.dy, planeheight, LabelOnScreen);
            vbroadcastss(r1, filler_r); // source row off-screen: all filler
            L(LabelOnScreen);
          }
          stack1.push_back(r1);
        }
        else {
          // 16 pixels
          YmmReg r1, r2;
          XmmReg filler_r; // borderConstant: filler value in all elements
          if (iter.bc == borderConstant)
            loadFiller(filler_r, iter.filler);
          relLoad16(r1, r2, a, regptrs, zero, iter.op, iter.dx, iter.bc, filler_r, &iter.conv);
          if (iter.bc == borderConstant && iter.dy != 0) {
            std::string LabelOnScreen = "relRowOn" + getLabelCount();
            emitRowOffscreenCheck(*this, SpatialY, iter.dy, planeheight, LabelOnScreen);
            vbroadcastss(r1, filler_r); // source row off-screen: all filler
            vbroadcastss(r2, filler_r);
            L(LabelOnScreen);
          }
          stack.push_back(std::make_pair(r1, r2));
        }
      }
      else if (iter.op == opLoadSrc8) {
        if (processSingle) {
          XmmReg r1x;
          YmmReg r1;
          Reg a;
          mov(a, ptr[regptrs + sizeof(void *) * (iter.e.ival + RWPTR_START_OF_INPUTS)]);
          // 8 bytes, 8 pixels * uint8_t
          vmovq(r1x, mmword_ptr[a]);
          // 8->32 bits like _mm256_cvtepu8_epi32
          vpmovzxbd(r1, r1x);
          // int -> float
          vcvtdq2ps(r1, r1);
          if (maskIt)
            vblendps(r1, zero, r1, mask);
          stack1.push_back(r1);
        }
        else {
          XmmReg r1x;
          YmmReg r1, r2;
          Reg a;
          mov(a, ptr[regptrs + sizeof(void *) * (iter.e.ival + RWPTR_START_OF_INPUTS)]);
          // 16 bytes, 16 pixels * uint8_t
          vmovdqa(r1x, xmmword_ptr[a]);
          // 8->16 bits like _mm256_cvtepu8_epi16
          vpmovzxbw(r1, r1x);
          // 16->32 bit like _mm256_cvtepu16_epi32
          vextracti128(r1x, r1, 1); // upper 128
          vpmovzxwd(r2, r1x);
          vextracti128(r1x, r1, 0); // lower 128
          vpmovzxwd(r1, r1x);
          // int -> float
          vcvtdq2ps(r1, r1);
          vcvtdq2ps(r2, r2);
          if (maskIt)
            vblendps(r2, zero, r2, mask);
          stack.push_back(std::make_pair(r1, r2));
        }
      }
      else if (iter.op == opLoadSrc16) {
        if (processSingle) {
          XmmReg r1x;
          YmmReg r1;
          Reg a;
          mov(a, ptr[regptrs + sizeof(void *) * (iter.e.ival + RWPTR_START_OF_INPUTS)]);
          // 16 bytes, 8 pixels * uint16_t
          vmovdqa(r1x, xmmword_ptr[a]);
          // 16->32 bit like _mm256_cvtepu16_epi32
          vpmovzxwd(r1, r1x);
          // int -> float
          vcvtdq2ps(r1, r1);
          if (maskIt)
            vblendps(r1, zero, r1, mask);
          stack1.push_back(r1);
        }
        else {
          XmmReg r1x;
          YmmReg r1, r2;
          Reg a;
          mov(a, ptr[regptrs + sizeof(void *) * (iter.e.ival + RWPTR_START_OF_INPUTS)]);
          // 32 bytes, 16 pixels * uint16_t
          vmovdqa(r1, ymmword_ptr[a]);
          // 16->32 bit like _mm256_cvtepu16_epi32
          vextracti128(r1x, r1, 1); // upper 128
          vpmovzxwd(r2, r1x);
          vextracti128(r1x, r1, 0); // lower 128
          vpmovzxwd(r1, r1x);
          // int -> float
          vcvtdq2ps(r1, r1);
          vcvtdq2ps(r2, r2);
          if (maskIt)
            vblendps(r2, zero, r2, mask);
          stack.push_back(std::make_pair(r1, r2));
        }
      }
      else if (iter.op == opLoadSrcF32) {
        if (processSingle) {
          YmmReg r1;
          Reg a;
          mov(a, ptr[regptrs + sizeof(void *) * (iter.e.ival + RWPTR_START_OF_INPUTS)]);
          // 32 bytes, 8 * float
          vmovdqa(r1, ymmword_ptr[a]);
          if (maskIt)
            vblendps(r1, zero, r1, mask);
          stack1.push_back(r1);
        }
        else {
          YmmReg r1, r2;
          Reg a;
          mov(a, ptr[regptrs + sizeof(void *) * (iter.e.ival + RWPTR_START_OF_INPUTS)]);
          // 32 bytes, 8 * float
          vmovdqa(r1, ymmword_ptr[a]);
          vmovdqa(r2, ymmword_ptr[a + 32]); // needs 64 byte aligned data to prevent read past valid data!
          if (maskIt)
            vblendps(r2, zero, r2, mask);
          stack.push_back(std::make_pair(r1, r2));
        }
      }
      else if (iter.op == opLoadSrcF16) { // not supported in avs+
        if (processSingle) {
          YmmReg r1;
          Reg a;
          mov(a, ptr[regptrs + sizeof(void *) * (iter.e.ival + RWPTR_START_OF_INPUTS)]);
          vcvtph2ps(r1, xmmword_ptr[a]);
          if (maskIt)
            vblendps(r1, zero, r1, mask);
          stack1.push_back(r1);
        }
        else {
          YmmReg r1, r2;
          Reg a;
          mov(a, ptr[regptrs + sizeof(void *) * (iter.e.ival + RWPTR_START_OF_INPUTS)]);
          vcvtph2ps(r1, xmmword_ptr[a]);
          vcvtph2ps(r2, xmmword_ptr[a + 16]);
          if (maskIt)
            vblendps(r2, zero, r2, mask);
          stack.push_back(std::make_pair(r1, r2));
        }
      }
      else if (iter.op == opLoadVar) {
        if (processSingle) {
          YmmReg r1;
          // 32 bytes/variable
          int offset = sizeof(void *) * RWPTR_START_OF_USERVARIABLES + 32 * iter.e.ival;
          // 32 bytes, 8 * float
          vmovdqa(r1, ymmword_ptr[regptrs + offset]);
          if (maskIt)
            vblendps(r1, zero, r1, mask);
          stack1.push_back(r1);
        }
        else {
          YmmReg r1, r2;
          // 64 bytes/variable
          int offset = sizeof(void *) * RWPTR_START_OF_USERVARIABLES + 64 * iter.e.ival;
          // 32 bytes, 8 * float
          vmovdqa(r1, ymmword_ptr[regptrs + offset]);
          vmovdqa(r2, ymmword_ptr[regptrs + offset + 32]); // needs 64 byte aligned data to prevent read past valid data!
          if (maskIt)
            vblendps(r2, zero, r2, mask);
          stack.push_back(std::make_pair(r1, r2));
        }
      }
      else if (iter.op == opLoadConst) {
        if (processSingle) {
          YmmReg r1;
          Reg32 a;
          XmmReg r1x;
          mov(a, iter.e.ival);
          vmovd(r1x, a);
          vbroadcastss(r1, r1x);
          stack1.push_back(r1);
        }
        else {
          YmmReg r1, r2;
          Reg32 a;
          XmmReg r1x;
          mov(a, iter.e.ival);
          vmovd(r1x, a);
          vbroadcastss(r1, r1x);
          vmovaps(r2, r1);
          stack.push_back(std::make_pair(r1, r2));
        }
      }
      else if (iter.op == opDup) {
        if (processSingle) {
          auto p = std::next(stack1.rbegin(), iter.e.ival);
          YmmReg r1;
          vmovaps(r1, *p);
          stack1.push_back(r1);
        }
        else {
          auto p = std::next(stack.rbegin(), iter.e.ival);
          YmmReg r1, r2;
          vmovaps(r1, p->first);
          vmovaps(r2, p->second);
          stack.push_back(std::make_pair(r1, r2));
        }
      }
      else if (iter.op == opSwap) {
        if(processSingle)
          std::swap(stack1.back(), *std::next(stack1.rbegin(), iter.e.ival));
        else
          std::swap(stack.back(), *std::next(stack.rbegin(), iter.e.ival));
      }
      else if (iter.op == opAdd) {
        if (processSingle) {
          TwoArgOp_Single_Avx(vaddps);
        }
        else {
          TwoArgOp_Avx(vaddps);
        }
      }
      else if (iter.op == opSub) {
        if (processSingle) {
          TwoArgOp_Single_Avx(vsubps);
        }
        else {
          TwoArgOp_Avx(vsubps);
        }
      }
      else if (iter.op == opMul) {
        if (processSingle) {
          TwoArgOp_Single_Avx(vmulps);
        }
        else {
          TwoArgOp_Avx(vmulps);
        }
      }
      else if (iter.op == opDiv) {
        if (processSingle) {
          TwoArgOp_Single_Avx(vdivps);
        }
        else {
          TwoArgOp_Avx(vdivps);
        }
      }
      else if (iter.op == opFmod) {
        if (processSingle) {
          auto t1 = stack1.back();
          stack1.pop_back();
          auto &t2 = stack1.back();
          FMOD_PS_AVX(t2, t1)
        }
        else {
          auto t1 = stack.back();
          stack.pop_back();
          auto &t2 = stack.back();
          FMOD_PS_AVX(t2.first, t1.first)
          FMOD_PS_AVX(t2.second, t1.second)
        }
      }
      else if (iter.op == opMax) {
        if (processSingle) {
          TwoArgOp_Single_Avx(vmaxps);
        }
        else {
          TwoArgOp_Avx(vmaxps);
        }
      }
      else if (iter.op == opMin) {
        if (processSingle) {
          TwoArgOp_Single_Avx(vminps);
        }
        else {
          TwoArgOp_Avx(vminps);
        }
      }
      else if (iter.op == opSqrt) {
        if (processSingle) {
          auto &t1 = stack1.back();
          vmaxps(t1, t1, zero);
          vsqrtps(t1, t1);
        }
        else {
          auto &t1 = stack.back();
          vmaxps(t1.first, t1.first, zero);
          vmaxps(t1.second, t1.second, zero);
          vsqrtps(t1.first, t1.first);
          vsqrtps(t1.second, t1.second);
        }
      }
      else if (iter.op == opStore8) {
        if (processSingle) {
          auto t1 = stack1.back();
          stack1.pop_back();
          Reg a;
          vaddps(t1, t1, CPTR_AVX(elfloat_half)); // rounder for truncate! no banker's rounding
          vmaxps(t1, t1, zero);
          vminps(t1, t1, CPTR_AVX(elstore8));
          mov(a, ptr[regptrs]);
          vcvttps2dq(t1, t1);   // float to int32 no bankers rounding
          XmmReg r1x, r2x;
          // 32 -> 16 bits from ymm 8 integers to xmm 8 words
          // first
          vextracti128(r1x, t1, 0);
          vextracti128(r2x, t1, 1);
          vpackusdw(r1x, r1x, r2x); // _mm_packus_epi32: w7 w6 w5 w4 w3 w2 w1 w0
          // 16 -> 8 bits
          vpackuswb(r1x, r1x, r1x); // _mm_packus_epi16: w3 w2 w1 w0 w3 w2 w1 w0
          vmovq(mmword_ptr[a], r1x); // store 8 bytes
        }
        else {
          auto t1 = stack.back();
          stack.pop_back();
          Reg a;
          vaddps(t1.first, t1.first, CPTR_AVX(elfloat_half)); // rounder for truncate! no banker's rounding
          vmaxps(t1.first, t1.first, zero);
          vaddps(t1.second, t1.second, CPTR_AVX(elfloat_half)); // rounder for truncate! no banker's rounding
          vmaxps(t1.second, t1.second, zero);
          vminps(t1.first, t1.first, CPTR_AVX(elstore8));
          vminps(t1.second, t1.second, CPTR_AVX(elstore8));
          mov(a, ptr[regptrs]);
          vcvttps2dq(t1.first, t1.first);   // float to int32 no bankers rounding
          vcvttps2dq(t1.second, t1.second);
          // we have 8 integers in t.first and another 8 in t.second
          // second                           first
          // d15 d14 d13 d12 d11 d10 d9 d8    d7 d6 d5 d4 d3 d2 d1 d0  // 16x32 bit integers in two ymm registers. not really 256 bits, but 2x128 bits
          XmmReg r1x, r2x, r_lo_x;
          // 32 -> 16 bits from ymm 8 integers to xmm 8 words
          // first
          vextracti128(r1x, t1.first, 0);
          vextracti128(r2x, t1.first, 1);
          vpackusdw(r_lo_x, r1x, r2x); // _mm_packus_epi32: w7 w6 w5 w4 w3 w2 w1 w0
          // second
          vextracti128(r1x, t1.second, 0); // not perfect, lower 128 bits of t1 could be used as xmm in packus. Cannot tell jitasm that xxmN is lower ymmN
          vextracti128(r2x, t1.second, 1);
          vpackusdw(r1x, r1x, r2x); //  _mm_packus_epi32: w7 w6 w5 w4 w3 w2 w1 w0
          // 16 -> 8 bits
          vpackuswb(r1x, r_lo_x, r1x); // _mm_packus_epi16: w3 w2 w1 w0 w3 w2 w1 w0
          vmovdqa(xmmword_ptr[a], r1x); // store 16 bytes
        }
      }
      else if (iter.op == opStore10 // avs+
        || iter.op == opStore12 // avs+
        || iter.op == opStore14 // avs+
        || iter.op == opStore16
        ) {
        if (processSingle) {
          auto t1 = stack1.back();
          stack1.pop_back();
          Reg a;
          vaddps(t1, t1, CPTR_AVX(elfloat_half)); // rounder for truncate! no banker's rounding
          vmaxps(t1, t1, zero);
          switch (iter.op) {
          case opStore10:
            vminps(t1, t1, CPTR_AVX(elstore10));
            break;
          case opStore12:
            vminps(t1, t1, CPTR_AVX(elstore12));
            break;
          case opStore14:
            vminps(t1, t1, CPTR_AVX(elstore14));
            break;
          case opStore16:
            vminps(t1, t1, CPTR_AVX(elstore16));
            break;
          }
          mov(a, ptr[regptrs]);
          vcvttps2dq(t1, t1);   // min / max clamp ensures that high words are zero
          XmmReg r1x, r2x;
          // 32 -> 16 bits from ymm 8 integers to xmm 8 words
          vextracti128(r1x, t1, 0); // not perfect, lower 128 bits of t1 could be used as xmm in packus. Cannot tell jitasm that xxmN is lower ymmN
          vextracti128(r2x, t1, 1);
          vpackusdw(r1x, r1x, r2x); //  _mm_packus_epi32: w7 w6 w5 w4 w3 w2 w1 w0
          vmovdqa(xmmword_ptr[a], r1x);
        }
        else {
          auto t1 = stack.back();
          stack.pop_back();
          Reg a;
          vaddps(t1.first, t1.first, CPTR_AVX(elfloat_half)); // rounder for truncate! no banker's rounding
          vmaxps(t1.first, t1.first, zero);
          vaddps(t1.second, t1.second, CPTR_AVX(elfloat_half)); // rounder for truncate! no banker's rounding
          vmaxps(t1.second, t1.second, zero);
          switch (iter.op) {
          case opStore10:
            vminps(t1.first, t1.first, CPTR_AVX(elstore10));
            vminps(t1.second, t1.second, CPTR_AVX(elstore10));
            break;
          case opStore12:
            vminps(t1.first, t1.first, CPTR_AVX(elstore12));
            vminps(t1.second, t1.second, CPTR_AVX(elstore12));
            break;
          case opStore14:
            vminps(t1.first, t1.first, CPTR_AVX(elstore14));
            vminps(t1.second, t1.second, CPTR_AVX(elstore14));
            break;
          case opStore16:
            vminps(t1.first, t1.first, CPTR_AVX(elstore16));
            vminps(t1.second, t1.second, CPTR_AVX(elstore16));
            break;
          }
          mov(a, ptr[regptrs]);
          vcvttps2dq(t1.first, t1.first);   // min / max clamp ensures that high words are zero
          vcvttps2dq(t1.second, t1.second);
          // we have 8 integers in t.first and another 8 in t.second
          // second                           first
          // d15 d14 d13 d12 d11 d10 d9 d8    d7 d6 d5 d4 d3 d2 d1 d0  // 16x32 bit integers in two ymm registers. not really 256 bits, but 2x128 bits
          XmmReg r1x, r2x;
          // 32 -> 16 bits from ymm 8 integers to xmm 8 words
          // first
          vextracti128(r1x, t1.first, 0); // not perfect, lower 128 bits of t1 could be used as xmm in packus. Cannot tell jitasm that xxmN is lower ymmN
          vextracti128(r2x, t1.first, 1);
          vpackusdw(r1x, r1x, r2x); //  _mm_packus_epi32: w7 w6 w5 w4 w3 w2 w1 w0
          vmovdqa(xmmword_ptr[a], r1x);
          // second
          vextracti128(r1x, t1.second, 0);
          vextracti128(r2x, t1.second, 1);
          vpackusdw(r1x, r1x, r2x); //  _mm_packus_epi32: w7 w6 w5 w4 w3 w2 w1 w0
          vmovdqa(xmmword_ptr[a + 16], r1x);
        }
      }
      else if (iter.op == opStoreF32) {
        if (processSingle) {
          auto t1 = stack1.back();
          stack1.pop_back();
          Reg a;
          mov(a, ptr[regptrs]);
          vmovaps(ymmword_ptr[a], t1);
        } else {
          auto t1 = stack.back();
          stack.pop_back();
          Reg a;
          mov(a, ptr[regptrs]);
          vmovaps(ymmword_ptr[a], t1.first);
          vmovaps(ymmword_ptr[a + 32], t1.second); // this needs 64 byte aligned data to prevent overwrite!
        }
      }
      else if (iter.op == opStoreF16) { // not supported in avs+
        if (processSingle) {
          auto t1 = stack1.back();
          stack1.pop_back();
          Reg a;
          mov(a, ptr[regptrs]);
          vcvtps2ph(xmmword_ptr[a], t1, 0);
        } else {
          auto t1 = stack.back();
          stack.pop_back();
          Reg a;
          mov(a, ptr[regptrs]);
          vcvtps2ph(xmmword_ptr[a], t1.first, 0);
          vcvtps2ph(xmmword_ptr[a + 16], t1.second, 0);
        }
      }
      else if (iter.op == opStoreVar || iter.op == opStoreVarAndDrop1) {
        if (processSingle) {
          auto t1 = stack1.back();
          // 32 bytes/variable
          int offset = sizeof(void *) * RWPTR_START_OF_USERVARIABLES + 32 * iter.e.ival;
          vmovaps(ymmword_ptr[regptrs + offset], t1);
          if (iter.op == opStoreVarAndDrop1)
            stack1.pop_back();
        }
        else {
          auto t1 = stack.back();
          // 64 bytes/variable
          int offset = sizeof(void *) * RWPTR_START_OF_USERVARIABLES + 64 * iter.e.ival;
          vmovaps(ymmword_ptr[regptrs + offset], t1.first);
          vmovaps(ymmword_ptr[regptrs + offset + 32], t1.second); // this needs 64 byte aligned data to prevent overwrite!
          if (iter.op == opStoreVarAndDrop1)
            stack.pop_back();
        }
      }
      else if (iter.op == opAbs) {
        if (processSingle) {
          auto &t1 = stack1.back();
          vandps(t1, t1, CPTR_AVX(elabsmask));
        }
        else {
          auto &t1 = stack.back();
          vandps(t1.first, t1.first, CPTR_AVX(elabsmask));
          vandps(t1.second, t1.second, CPTR_AVX(elabsmask));
        }
      } 
      else if (iter.op == opSgn) {
        // 1, 0, -1
        if (processSingle) {
          auto &t1 = stack1.back();
          YmmReg r1, r2;
          vxorps(r2, r2, r2);
          vcmpps(r1, t1, r2, _CMP_GT_OQ);
          vcmpps(t1, t1, r2, _CMP_LT_OQ);
          vandps(r1, r1, CPTR_AVX(elfloat_one));
          vandps(t1, t1, CPTR_AVX(elfloat_minusone));
          vorps(t1, r1, t1);
        }
        else {
          auto &t1 = stack.back();
          YmmReg r2, r3, r4, r5;
          vxorps(r2, r2, r2);
          vcmpps(r3, t1.first, r2, _CMP_GT_OQ);
          vcmpps(t1.first, t1.first, r2, _CMP_LT_OQ);
          vcmpps(r4, t1.second, r2, _CMP_GT_OQ);
          vcmpps(t1.second, t1.second, r2, _CMP_LT_OQ);
          vmovaps(r2, CPTR_AVX(elfloat_one));
          vandps(r3, r3, r2);
          vmovaps(r5, CPTR_AVX(elfloat_minusone));
          vblendvps(t1.first, r3, r5, t1.first);
          vandps(r2, r4, r2);
          vblendvps(t1.second, r2, r5, t1.second);
        }
      }
      else if (iter.op == opNeg) {
        if (processSingle) {
          auto &t1 = stack1.back();
          vcmpps(t1, t1, zero, _CMP_LE_OQ); // cmpleps
          vandps(t1, t1, CPTR_AVX(elfloat_one));
        }
        else {
          auto &t1 = stack.back();
          vcmpps(t1.first, t1.first, zero, _CMP_LE_OQ); // cmpleps
          vcmpps(t1.second, t1.second, zero, _CMP_LE_OQ);
          vandps(t1.first, t1.first, CPTR_AVX(elfloat_one));
          vandps(t1.second, t1.second, CPTR_AVX(elfloat_one));
        }
      }
      else if (iter.op == opNegSign) {
        if (processSingle) {
          auto& t1 = stack1.back();
          vxorps(t1, t1, CPTR_AVX(elsignmask));
        }
        else {
          auto& t1 = stack.back();
          vxorps(t1.first, t1.first, CPTR_AVX(elsignmask));
          vxorps(t1.second, t1.second, CPTR_AVX(elsignmask));
        }
      }
      else if (iter.op == opAnd) {
        if (processSingle) {
          LogicOp_Single_Avx(vandps);
        }
        else {
          LogicOp_Avx(vandps);
        }
      }
      else if (iter.op == opOr) {
        if (processSingle) {
          LogicOp_Single_Avx(vorps);
        }
        else {
          LogicOp_Avx(vorps);
        }
      }
      else if (iter.op == opXor) {
        if (processSingle) {
          LogicOp_Single_Avx(vxorps);
        }
        else {
          LogicOp_Avx(vxorps);
        }
      }
      else if (iter.op == opGt) { // a > b (gt) -> b < (lt) a
        if (processSingle) {
          CmpOp_Single_Avx(vcmpps, _CMP_LT_OQ); // cmpltps
        }
        else {
          CmpOp_Avx(vcmpps, _CMP_LT_OQ) // cmpltps
        }
      }
      else if (iter.op == opLt) { // a < b (lt) -> b > (gt,nle) a
        if (processSingle) {
          CmpOp_Single_Avx(vcmpps, _CMP_GT_OQ); // cmpnleps
        }
        else {
          CmpOp_Avx(vcmpps, _CMP_GT_OQ); // cmpnleps
        }
      }
      else if (iter.op == opEq) {
        if (processSingle) {
          CmpOp_Single_Avx(vcmpps, _CMP_EQ_OQ);
        }
        else {
          CmpOp_Avx(vcmpps, _CMP_EQ_OQ);
        }
      }
      else if (iter.op == opNotEq) { // avs+
        if (processSingle) {
          CmpOp_Single_Avx(vcmpps, _CMP_NEQ_OQ);
        }
        else {
          CmpOp_Avx(vcmpps, _CMP_NEQ_OQ);
        }
      }
      else if (iter.op == opLE) { // a <= b -> b >= (ge,nlt) a
        if (processSingle) {
          CmpOp_Single_Avx(vcmpps, _CMP_GE_OS); // cmpnltps
        }
        else {
          CmpOp_Avx(vcmpps, _CMP_GE_OS) // cmpnltps
        }
      }
      else if (iter.op == opGE) { // a >= b -> b <= (le) a
        if (processSingle) {
          CmpOp_Single_Avx(vcmpps, _CMP_LE_OS) // cmpleps
        }
        else {
          CmpOp_Avx(vcmpps, _CMP_LE_OS) // cmpleps
        }
      }
      else if (iter.op == opTernary) {
        if (processSingle) {
          auto t1 = stack1.back();
          stack1.pop_back();
          auto t2 = stack1.back();
          stack1.pop_back();
          auto t3 = stack1.back();
          stack1.pop_back();
          YmmReg r1;
          vxorps(r1, r1, r1);
          vcmpps(r1, r1, t3, _CMP_LT_OQ); // cmpltps -> vcmpps ... _CMP_LT_OQ
          vandps(t2, t2, r1);
          vandnps(r1, r1, t1);
          vorps(r1, r1, t2);
          stack1.push_back(r1);
        }
        else {
          auto t1 = stack.back();
          stack.pop_back();
          auto t2 = stack.back();
          stack.pop_back();
          auto t3 = stack.back();
          stack.pop_back();
          YmmReg r1, r2;
          vxorps(r1, r1, r1);
          vxorps(r2, r2, r2);
          vcmpps(r1, r1, t3.first, _CMP_LT_OQ); // cmpltps -> vcmpps ... _CMP_LT_OQ
          vcmpps(r2, r2, t3.second, _CMP_LT_OQ);
          vandps(t2.first, t2.first, r1);
          vandps(t2.second, t2.second, r2);
          vandnps(r1, r1, t1.first);
          vandnps(r2, r2, t1.second);
          vorps(r1, r1, t2.first);
          vorps(r2, r2, t2.second);
          stack.push_back(std::make_pair(r1, r2));
        }
      }
      else if (iter.op == opExp) {
        if (processSingle) {
          auto &t1 = stack1.back();
          EXP_PS_AVX(t1);
        }
        else {
          auto &t1 = stack.back();
          EXP_PS_AVX(t1.first);
          EXP_PS_AVX(t1.second);
        }
      }
      else if (iter.op == opLog) {
        if (processSingle) {
          auto &t1 = stack1.back();
          LOG_PS_AVX(t1);
        } else {
          auto &t1 = stack.back();
          LOG_PS_AVX(t1.first);
          LOG_PS_AVX(t1.second);
        }
      }
      else if (iter.op == opPow) {
        if (processSingle) {
          auto t1 = stack1.back();
          stack1.pop_back();
          auto &t2 = stack1.back();
          LOG_PS_AVX(t2);
          vmulps(t2, t2, t1);
          EXP_PS_AVX(t2);
        } else {
          auto t1 = stack.back();
          stack.pop_back();
          auto &t2 = stack.back();
          LOG_PS_AVX(t2.first);
          vmulps(t2.first, t2.first, t1.first);
          EXP_PS_AVX(t2.first);
          LOG_PS_AVX(t2.second);
          vmulps(t2.second, t2.second, t1.second);
          EXP_PS_AVX(t2.second);
        }
      }
      else if (iter.op == opSin) {
        if (processSingle) {
          auto& _t1 = stack1.back();
          SINCOS_PS_AVX(true, _t1, _t1);
        }
        else {
          auto& _t1 = stack.back();
          SINCOS_PS_AVX(true, _t1.first, _t1.first);
          SINCOS_PS_AVX(true, _t1.second, _t1.second);
        }
      }
      else if (iter.op == opCos) {
        if (processSingle) {
          auto& _t1 = stack1.back();
          SINCOS_PS_AVX(false, _t1, _t1);
        }
        else {
          auto& _t1 = stack.back();
          SINCOS_PS_AVX(false, _t1.first, _t1.first);
          SINCOS_PS_AVX(false, _t1.second, _t1.second);
        }
      }
      else if (iter.op == opTan) {
        if (processSingle) {
          auto& t1 = stack1.back();
          TAN_PS_AVX(t1);
        }
        else {
          auto& t1 = stack.back();
          TAN_PS_AVX(t1.first);
          TAN_PS_AVX(t1.second);
        }
      }
      else if (iter.op == opAtan2) {
        if (processSingle) {
          auto t1 = stack1.back();
          stack1.pop_back();
          auto &t2 = stack1.back();
          ATAN2_PS_AVX(t2, t1);
        } else {
          auto t1 = stack.back();
          stack.pop_back();
          auto &t2 = stack.back();
          ATAN2_PS_AVX(t2.first, t1.first);
          ATAN2_PS_AVX(t2.second, t1.second);
        }
      }
      else if (iter.op == opClip) {
        // clip(a, low, high) = min(max(a, low),high)
        if (processSingle) {
          auto t1 = stack1.back();
          stack1.pop_back();
          auto t2 = stack1.back();
          stack1.pop_back();
          auto &t3 = stack1.back();
          vmaxps(t3, t3, t2);
          vminps(t3, t3, t1);
        }
        else {
          auto t1 = stack.back();
          stack.pop_back();
          auto t2 = stack.back();
          stack.pop_back();
          auto &t3 = stack.back();
          vmaxps(t3.first, t3.first, t2.first);
          vminps(t3.first, t3.first, t1.first);
          vmaxps(t3.second, t3.second, t2.second);
          vminps(t3.second, t3.second, t1.second);
        }
      }
      else if (iter.op == opRound || iter.op == opFloor || iter.op == opCeil || iter.op == opTrunc) {
        const int rounder_flag =
          (iter.op == opRound) ? (FROUND_TO_NEAREST_INT | FROUND_NO_EXC) :
          (iter.op == opFloor) ? (FROUND_TO_NEG_INF | FROUND_NO_EXC) :
          (iter.op == opCeil) ? (FROUND_TO_POS_INF | FROUND_NO_EXC) :
          (FROUND_TO_ZERO | FROUND_NO_EXC); // opTrunc
        if (processSingle) {
          auto& t1 = stack1.back();
          vroundps(t1, t1, rounder_flag);
        }
        else {
          auto& t1 = stack.back();
          vroundps(t1.first, t1.first, rounder_flag);
          vroundps(t1.second, t1.second, rounder_flag);
        }
      }
    }
  }
/*
In brief:
jitasm was modded to accept avx_epilog_=true for code generation

Why: couldn't use vzeroupper because prolog/epilog was saving all xmm6:xmm15 registers even if they were not used at all
Why2: movaps was generated instead of vmovaps for prolog/epilog
Why3: internal register reordering/saving was non-vex encoded
All these issues resulted in AVX->SSE2 penalty

From MSDN:
XMM6:XMM15, YMM6:YMM15 rules for x64:
Nonvolatile (XMM), Volatile (upper half of YMM)
XMM6:XMM15 Must be preserved as needed by callee.
YMM registers must be preserved as needed by caller. (they do not need to be preserved)

Problem:
- Jitasm saves xmm6..xmm15 when vzeroupper is used,even if only an xmm0 is used (Why?)
No problem (looking at the disassembly list):
- when there is no vzeroupper, then the xmm6:xmm11 is properly saved/restored in prolog/epilog but only
  if ymm6:ymm11 (in this example) is used. If no register is used over xmm6/ymm6 then xmm registers are not saved at all.
- question: does it have any penalty when movaps is used w/o vzeroupper?
  The epilog generates movaps
  movaps      xmm11,xmmword ptr [rbx-10h]

0000000002430000  push        rbp
0000000002430001  mov         rbp,rsp
0000000002430004  push        rbx
0000000002430005  lea         rbx,[rsp-8]
000000000243000A  sub         rsp,0A8h
0000000002430011  movaps      xmmword ptr [rbx-0A0h],xmm6
0000000002430018  movaps      xmmword ptr [rbx-90h],xmm7
000000000243001F  movaps      xmmword ptr [rbx-80h],xmm8
0000000002430024  movaps      xmmword ptr [rbx-70h],xmm9
0000000002430029  movaps      xmmword ptr [rbx-60h],xmm10
000000000243002E  movaps      xmmword ptr [rbx-50h],xmm11
0000000002430033  movaps      xmmword ptr [rbx-40h],xmm12
0000000002430038  movaps      xmmword ptr [rbx-30h],xmm13
000000000243003D  movaps      xmmword ptr [rbx-20h],xmm14
0000000002430042  movaps      xmmword ptr [rbx-10h],xmm15
-- end of jitasm generated prolog

-- PF AVX+: passing avx_epilog_ = true for codegen, vmovaps is generated instead of movaps
0000000001E60010  vmovaps     xmmword ptr [rbx-60h],xmm6
0000000001E60015  vmovaps     xmmword ptr [rbx-50h],xmm7
0000000001E6001A  vmovaps     xmmword ptr [rbx-40h],xmm8
0000000001E6001F  vmovaps     xmmword ptr [rbx-30h],xmm9
0000000001E60024  vmovaps     xmmword ptr [rbx-20h],xmm10
0000000001E60029  vmovaps     xmmword ptr [rbx-10h],xmm11

// PF comment: user's code like this:
  YmmReg zero;
  vpxor(zero, zero, zero);
  Reg constptr;
  mov(constptr, (uintptr_t)logexpconst_avx);
  vzeroupper();
And the generated instructions:
0000000002430047  vpxor       ymm0,ymm0,ymm0
000000000243004B  mov         rax,7FECCBD15C0h
0000000002430055  vzeroupper
Note: Don't use vzeroupper manually. When vzeroupper is issued manually, jitasm is not too generous: marks all xmm6:xmm15 registers as used
      and epilog and prolog will save all of them, even if none of those xmm/ymm registers are used in the code.
      Modded jitasm: pass avx_epilog_ = true for codegen, it will issue vzeroupper automatically (and has other benefits)

-- start of jitasm generated epilog (old)
0000000002430058  movaps      xmm15,xmmword ptr [rbx-10h]
000000000243005D  movaps      xmm14,xmmword ptr [rbx-20h]
0000000002430062  movaps      xmm13,xmmword ptr [rbx-30h]
0000000002430067  movaps      xmm12,xmmword ptr [rbx-40h]
000000000243006C  movaps      xmm11,xmmword ptr [rbx-50h]
0000000002430071  movaps      xmm10,xmmword ptr [rbx-60h]
0000000002430076  movaps      xmm9,xmmword ptr [rbx-70h]
000000000243007B  movaps      xmm8,xmmword ptr [rbx-80h]
0000000002430080  movaps      xmm7,xmmword ptr [rbx-90h]
0000000002430087  movaps      xmm6,xmmword ptr [rbx-0A0h]
000000000243008E  add         rsp,0A8h
0000000002430095  pop         rbx
0000000002430096  pop         rbp
0000000002430097  ret
-- end of jitasm generated epilog (old)

PF: modded jitasm (calling codegen with avx_epilog_ = true) generates vmovaps instead of movaps and and automatic vzeroupper before the ret instruction
generated epilog example (new):
0000000001E70613  vmovaps     xmm11,xmmword ptr [rbx-10h]
0000000001E70618  vmovaps     xmm10,xmmword ptr [rbx-20h]
0000000001E7061D  vmovaps     xmm9,xmmword ptr [rbx-30h]
0000000001E70622  vmovaps     xmm8,xmmword ptr [rbx-40h]
0000000001E70627  vmovaps     xmm7,xmmword ptr [rbx-50h]
0000000001E7062C  vmovaps     xmm6,xmmword ptr [rbx-60h]
0000000001E70631  add         rsp,68h
0000000001E70635  pop         rdi
0000000001E70636  pop         rsi
0000000001E70637  pop         rbx
0000000001E70638  pop         rbp
0000000001E70639  vzeroupper
0000000001E7063C  ret

*/
  void main(Reg regptrs, Reg regoffs, Reg niter, Reg SpatialY)
  {
    YmmReg zero;
    vpxor(zero, zero, zero);
    Reg constptr;
    mov(constptr, (uintptr_t)logexpconst_avx);

    L("wloop");
    cmp(niter, 0); // while(niter>0)
    je("wend");
    sub(niter, 1);

    // process two sets, no partial input masking
    if(singleMode)
      processingLoop<true, false>(regptrs, zero, constptr, SpatialY);
    else
      processingLoop<false, false>(regptrs, zero, constptr, SpatialY);

    // increase read and write pointers by 16 pixels
    const int EXTRA = 2; // output pointer, xcounter
    if constexpr(sizeof(void *) == 8) {
      // x64: two 8 byte pointers in an xmm
      int numIter = (numInputs + EXTRA + 1) / 2;

      for (int i = 0; i < numIter; i++) {
        XmmReg r1, r2;
        vmovdqu(r1, xmmword_ptr[regptrs + 16 * i]);
        vmovdqu(r2, xmmword_ptr[regoffs + 16 * i]);
        vpaddq(r1, r1, r2); // pointers are 64 bits
        vmovdqu(xmmword_ptr[regptrs + 16 * i], r1);
      }
    }
    else {
      // x86: four 4 byte pointers in an xmm
      int numIter = (numInputs + EXTRA + 3) / 4;
      for (int i = 0; i < numIter; i++) {
        XmmReg r1, r2;
        vmovdqu(r1, xmmword_ptr[regptrs + 16 * i]);
        vmovdqu(r2, xmmword_ptr[regoffs + 16 * i]);
        vpaddd(r1, r1, r2); // pointers are 32 bits
        vmovdqu(xmmword_ptr[regptrs + 16 * i], r1);
      }
    }

    jmp("wloop");
    L("wend");

    int nrestpixels = planewidth & (singleMode ? 7 : 15);
    if(nrestpixels > 8) // dual process with masking
      processingLoop<false, true>(regptrs, zero, constptr, SpatialY);
    else if (nrestpixels == 8) // single process, no masking
      processingLoop<true, false>(regptrs, zero, constptr, SpatialY);
    else if (nrestpixels > 0) // single process, masking
      processingLoop<true, true>(regptrs, zero, constptr, SpatialY);
    // bug in jitasm?
    // on x64, when this is here, debug throws an assert, that a register save/load has an
    // operand size 8 bit, instead of 128 (XMM) or 256 (YMM)
    // vzeroupper(); // don't use it directly. Generate code with avx_epilog_=true
  }
};

#endif


/********************************************************************
***** Declare index of new filters for Avisynth's filter engine *****
********************************************************************/

extern const AVSFunction Exprfilter_filters[] = {
  { "Expr", BUILTIN_FUNC_PREFIX, "c+s+[format]s[optAvx2]b[optSingleMode]b[optSSE2]b[scale_inputs]s[clamp_float]b[clamp_float_UV]b[lut]i[optVectorC]b", Exprfilter::Create },
  { 0 }
};


AVSValue __cdecl Exprfilter::Create(AVSValue args, void* , IScriptEnvironment* env) {

  std::vector<PClip> children;
  std::vector<std::string> expressions;
  int next_paramindex;

  // one or more clips
  if (args[0].IsArray() && args[0][0].IsClip()) { // c+s+ case
    children.resize(args[0].ArraySize());

    for (int i = 0; i < (int)children.size(); ++i) // Copy all
      children[i] = args[0][i].AsClip();

    next_paramindex = 1;
  }
  else if (args[1].IsArray() && args[1][0].IsClip()) { // cc+s+ case
    children.resize(1 + args[1].ArraySize());

    children[0] = args[0].AsClip(); // Copy 1st
    for (int i = 1; i < (int)children.size(); ++i) // Copy rest
      children[i] = args[1][i - 1].AsClip();

    next_paramindex = 2;
  }
  else if (args[1].IsClip()) { //cc case
    children.resize(2);

    children[0] = args[0].AsClip();
    children[1] = args[1].AsClip();

    next_paramindex = 2;
  }
  else if (args[0].IsClip()) { // single clip, cs+ case
    children.resize(1);
    children[0] = args[0].AsClip();

    next_paramindex = 1;
  }
  else {
    env->ThrowError("Expr: Invalid parameter type");
  }

  // one or more expressions: s+
  if (args[next_paramindex].Defined()) {
    AVSValue exprarg = args[next_paramindex++];
    if (exprarg.IsArray()) {
      int nexpr = exprarg.ArraySize();
      expressions.resize(nexpr);
      for (int i = 0; i < nexpr; i++)
        expressions[i] = exprarg[i].AsString();
    }
    else if (exprarg.IsString()) {
      expressions.resize(1);
      expressions[0] = exprarg.AsString();
    }
    else {
      env->ThrowError("Expr: Invalid parameter type for expression string");
    }
  }

  // optional named argument: format
  const char *newformat = nullptr;
  if (args[next_paramindex].Defined()) {
    // always string
    newformat = args[next_paramindex].AsString();
  }
  next_paramindex++;

#ifdef VS_TARGET_CPU_X86
  // test parameter for avx2-less mode even with avx2 available
#ifdef TEST_AVX2_CODEGEN_IN_AVX
  bool optAvx2 = !!(env->GetCPUFlags() & CPUF_AVX);
#else
  bool optAvx2 = !!(env->GetCPUFlags() & CPUF_AVX2);
#endif
  bool optSSE2 = !!(env->GetCPUFlags() & CPUF_SSE2);
#else
  bool optAvx2 = false;
  bool optSSE2 = false;
#endif

  if (args[next_paramindex].Defined()) {
    if (optAvx2) // disable only
      optAvx2 = args[next_paramindex].AsBool();
  }
  next_paramindex++;

  bool optSingleMode = false;
  if (args[next_paramindex].Defined()) {
    optSingleMode = args[next_paramindex].AsBool();
  }
  next_paramindex++;

  if (args[next_paramindex].Defined()) {
    if (optSSE2) // disable only
      optSSE2 = args[next_paramindex].AsBool();
  }
  next_paramindex++;

  std::string scale_inputs = args[next_paramindex].Defined() ? args[next_paramindex].AsString("none") : "none";
  transform(scale_inputs.begin(), scale_inputs.end(), scale_inputs.begin(), asciiToLower);
  next_paramindex++;

  const bool clamp_float = args[next_paramindex].AsBool(false);
  next_paramindex++;

  const bool clamp_float_UV = args[next_paramindex].AsBool(false);
  next_paramindex++;

  // clamp_float clamp_float_uv -> clamp_float_i   clamp range for Y clamp range for UV
  // false       x                 0               0..1              -0.5..+0.5
  // true        false             1               0..1              -0.5..+0.5
  // true        true              2               0..1              0..1

  int clamp_float_i;
  if (clamp_float)
    clamp_float_i = clamp_float_UV ? 2 : 1;
  else
    clamp_float_i = 0;

  const int lutmode = args[next_paramindex].AsInt(0); // 0, 1, 2
  next_paramindex++;

  const bool optVectorC = args[next_paramindex].AsBool(true);

  return new Exprfilter(children, expressions, newformat, optAvx2, optSingleMode, optSSE2, optVectorC, scale_inputs, clamp_float_i, lutmode, env);

}

// Base SIMD Processor interface
class ISIMDProcessor {
public:
  virtual void processVector(
    std::vector<const uint8_t*>&srcp,
    uint8_t*& dstp,
    int x, int y) = 0;
  virtual ~ISIMDProcessor() = default;
};

/**
 * Custom aligned memory allocator for STL containers like std::vector
 *
 * This allocator ensures that the underlying data pointer returned by container.data()
 * is aligned to the specified alignment boundary, which is critical for:
 *   - SIMD vector operations that require aligned memory access
 *   - Cache-friendly data structures
 *   - Hardware-specific memory alignment requirements
 *
 * Platform-specific implementation:
 *   - Windows (MSVC, ClangCL): Uses _aligned_malloc/_aligned_free
 *   - MinGW: Uses _aligned_malloc/_aligned_free
 *   - POSIX systems (Linux, Unix, macOS): Uses aligned_alloc/free
 *   - C++17 fallback: Uses std::aligned_alloc/free
 *
 * Usage:
 *   std::vector<float, aligned_allocator<float, 32>> aligned_vector;
 */
template <typename T, size_t Alignment>
struct aligned_allocator {
  // Standard allocator typedefs
  typedef T value_type;
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef T& reference;
  typedef const T& const_reference;
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;
  // Rebind allocator to type U
  template <typename U>
  struct rebind {
    typedef aligned_allocator<U, Alignment> other;
  };
  aligned_allocator() noexcept {}
  template <typename U>
  aligned_allocator(const aligned_allocator<U, Alignment>&) noexcept {}
  T* allocate(std::size_t n) {
#if defined(_MSC_VER) || (defined(__clang__) && defined(_MSC_VER))
    // Both MSVC and ClangCL should use _aligned_malloc
    void* ptr = _aligned_malloc(n * sizeof(T), Alignment);
    if (!ptr) throw std::bad_alloc();
#elif defined(__MINGW32__) || defined(__MINGW64__)
    // MinGW/MinGW-w64 specific
    void* ptr = _aligned_malloc(n * sizeof(T), Alignment);
    if (!ptr) throw std::bad_alloc();
#elif defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER) || defined(__APPLE__) || (defined(__GNUC__) && !defined(_WIN32) && !defined(__CYGWIN__))
    // POSIX-compliant systems: Intel compilers, macOS, GCC on non-Windows
    // aligned_alloc requires size to be a multiple of alignment
    size_t size = n * sizeof(T);
    if (size % Alignment != 0) {
      size = (size / Alignment + 1) * Alignment;
    }
    void* ptr = aligned_alloc(Alignment, size);
    if (!ptr) throw std::bad_alloc();
#else
    // Generic fallback for C++17 and later
    // std::aligned_alloc requires size to be a multiple of alignment
    size_t size = n * sizeof(T);
    if (size % Alignment != 0) {
      size = (size / Alignment + 1) * Alignment;
    }
    void* ptr = std::aligned_alloc(Alignment, size);
    if (!ptr) throw std::bad_alloc();
#endif
    return static_cast<T*>(ptr);
  }
  void deallocate(T* p, std::size_t) noexcept {
#if defined(_MSC_VER) || defined(__MINGW32__) || defined(__MINGW64__) || (defined(__clang__) && defined(_MSC_VER))
    _aligned_free(p);
#else
    free(p);
#endif
  }
};

// convenient type def
using aligned_float_vector = std::vector<float, aligned_allocator<float, 32>>;

template<int VectorSize>
class SIMDProcessor : public ISIMDProcessor {
  int w, h;
  size_t maxStackSize;

  // Define the aligned stack with custom allocator
  std::vector<aligned_float_vector> stack;

  aligned_float_vector& variable_area;
  std::vector<float>& internal_vars;
  std::vector<int>& src_stride;
  std::vector<const uint8_t*>& srcp_orig;
  const ExprOp* vops;
public:
  SIMDProcessor(int _w, int _h, size_t _maxStackSize, 
    aligned_float_vector& var_area,
    std::vector<float>& int_vars, std::vector<int>& src_str, std::vector<const uint8_t*>& src_orig, const ExprOp* vops)
    : w(_w), h(_h), maxStackSize(_maxStackSize), 
    stack(_maxStackSize, aligned_float_vector(VectorSize)), 
    variable_area(var_area), internal_vars(int_vars), src_stride(src_str), srcp_orig(src_orig), vops(vops) {
  }

  int stackIndex = 0;
  alignas(32) float stacktop[VectorSize] = {};

  // Broadcast a single value across the vector
  inline void push_and_broadcast(float val) {
    auto& current_stack = stack[stackIndex];

    // First loop: copy stacktop to stack - help vectorization pattern
    for (int i = 0; i < VectorSize; ++i)
      current_stack[i] = stacktop[i];
    for (int i = 0; i < VectorSize; ++i)
      stacktop[i] = val;
    stackIndex++;
  }

  inline void broadcast(float val) {
    for (int i = 0; i < VectorSize; ++i)
      stacktop[i] = val;
  }

  // Load spatial X coordinate
  inline void loadSpatialX(int x) {
    // Push current stacktop and fills x, x+1, x+2, x+3, ...
    auto& current_stack = stack[stackIndex];
    for (int i = 0; i < VectorSize; ++i)
      current_stack[i] = stacktop[i];
    for (int i = 0; i < VectorSize; ++i)
      stacktop[i] = static_cast<float>(x + i);
    stackIndex++;
  }

  // Load vector from source from x
  template<typename T>
  inline void loadSource(const T* src, int x) {
    auto& current_stack = stack[stackIndex];
    for (int i = 0; i < VectorSize; ++i)
      current_stack[i] = stacktop[i];
    for (int i = 0; i < VectorSize; ++i)
      stacktop[i] = static_cast<float>(src[x + i]);
    stackIndex++;
  }

  // Load from source with relative offset
  template<typename T>
  void loadRelSource(const T* src, int x, int y, int dx, int dy, int bc, float filler, const ExprLoadConv& conv, int width, int height, int stride) {
    auto& current_stack = stack[stackIndex];
    for (int i = 0; i < VectorSize; ++i)
      current_stack[i] = stacktop[i];
    // off-screen pixels: border mode bc (clamp, mirror, reflect101, constant, wrap)
    // borderConstant: conv converts the loaded pixels (scale_inputs), the filler is used as is
    // stride is of byte pitch
    if (bc == borderConstant && (y + dy < 0 || y + dy >= height)) {
      for (int i = 0; i < VectorSize; ++i)
        stacktop[i] = filler;
      stackIndex++;
      return;
    }
    int newY = borderIndex(y + dy, height, bc) - y;
    const uint8_t* rowPtr = reinterpret_cast<const uint8_t*>(src) + static_cast<intptr_t>(newY) * stride;
    for (int i = 0; i < VectorSize; ++i) {
      if (bc == borderConstant && (x + dx + i < 0 || x + dx + i >= width)) {
        stacktop[i] = filler;
        continue;
      }
      int newX = borderIndex(x + dx + i, width, bc);
      stacktop[i] = conv.apply(static_cast<float>(reinterpret_cast<const T*>(rowPtr)[newX]));
    }
    stackIndex++;
  }

  // Load from user variable area
  inline void loadVar(int index) {
    auto& current_stack = stack[stackIndex];
    for (int i = 0; i < VectorSize; ++i)
      current_stack[i] = stacktop[i];
    // Push current stacktop and broadcast x
    for (int i = 0; i < VectorSize; ++i)
      stacktop[i] = variable_area[index * MAX_C_VECT + i];
    stackIndex++;
  }

  // Vectorized duplicate
  inline void dup(int offset) {
    auto& current_stack = stack[stackIndex];
    for (int i = 0; i < VectorSize; ++i)
      current_stack[i] = stacktop[i];
    auto& prev = stack[stackIndex - offset];
    for (int i = 0; i < VectorSize; ++i)
      stacktop[i] = prev[i];
    stackIndex++;
  }

  // Vectorized swap
  inline void swap(int offset) {
    auto& prev = stack[stackIndex - offset];
    for (int i = 0; i < VectorSize; ++i)
      std::swap(stacktop[i], prev[i]);
  }

  // Vectorized addition
  inline void add() {
    stackIndex--;
    auto& prev = stack[stackIndex];
    for (int i = 0; i < VectorSize; ++i)
      stacktop[i] += prev[i];
  }

  // Vectorized subtract
  inline void sub() {
    stackIndex--;
    auto& prev = stack[stackIndex];
    for (int i = 0; i < VectorSize; ++i)
      stacktop[i] = prev[i] - stacktop[i];
  }

  // Vectorized multiplication
  inline void multiply() {
    stackIndex--;
    auto& prev = stack[stackIndex];
    for (int i = 0; i < VectorSize; ++i)
      stacktop[i] *= prev[i];
  }

  inline void divide() {
    stackIndex--;
    auto& prev = stack[stackIndex];
    for (int i = 0; i < VectorSize; ++i)
      stacktop[i] = prev[i] / stacktop[i];
  }

  void fmod() {
    stackIndex--;
    auto& prev = stack[stackIndex];
    for (int i = 0; i < VectorSize; ++i)
      stacktop[i] = std::fmod(prev[i], stacktop[i]);
  }

  inline void max() {
    stackIndex--;
    auto& prev = stack[stackIndex];
    for (int i = 0; i < VectorSize; ++i)
      stacktop[i] = std::max(prev[i], stacktop[i]);
  }

  inline void min() {
    stackIndex--;
    auto& prev = stack[stackIndex];
    for (int i = 0; i < VectorSize; ++i)
      stacktop[i] = std::min(prev[i], stacktop[i]);
  }

  void exp() {
    for (int i = 0; i < VectorSize; ++i)
      stacktop[i] = std::exp(stacktop[i]);
  }

  void log() {
    for (int i = 0; i < VectorSize; ++i)
      stacktop[i] = std::log(stacktop[i]);
  }

  void pow() {
    stackIndex--;
    auto& prev = stack[stackIndex];
    for (int i = 0; i < VectorSize; ++i)
      stacktop[i] = std::pow(prev[i], stacktop[i]);
  }

  // Vectorized clip
  inline void clip() {
    stackIndex -= 2;
    auto& prev = stack[stackIndex];
    auto& prev1 = stack[stackIndex + 1];
    for (int i = 0; i < VectorSize; ++i)
      stacktop[i] = std::max(std::min(prev[i], stacktop[i]), prev1[i]);
  }

  // Vectorized round
  inline void round() {
    for (int i = 0; i < VectorSize; ++i)
      stacktop[i] = std::round(stacktop[i]);
  }

  // Vectorized floor
  inline void floor() {
    for (int i = 0; i < VectorSize; ++i)
      stacktop[i] = std::floor(stacktop[i]);
  }

  // Vectorized ceil
  void ceil() {
    for (int i = 0; i < VectorSize; ++i)
      stacktop[i] = std::ceil(stacktop[i]);
  }

  // Vectorized trunc
  inline void trunc() {
    for (int i = 0; i < VectorSize; ++i)
      stacktop[i] = std::trunc(stacktop[i]);
  }

  // Vectorized sqrt
  void sqrt() {
    for (int i = 0; i < VectorSize; ++i)
      stacktop[i] = std::sqrt(stacktop[i]);
  }

  // Vectorized abs
  inline void abs() {
    for (int i = 0; i < VectorSize; ++i)
      stacktop[i] = std::abs(stacktop[i]);
  }

  // Vectorized sgn
  inline void sgn() {
    for (int i = 0; i < VectorSize; ++i)
      stacktop[i] = stacktop[i] < 0 ? -1.0f : stacktop[i] > 0 ? 1.0f : 0.0f;
  }

  // Vectorized sin
  void sin() {
    for (int i = 0; i < VectorSize; ++i)
      stacktop[i] = std::sin(stacktop[i]);
  }

  // Vectorized cos
  void cos() {
    for (int i = 0; i < VectorSize; ++i)
      stacktop[i] = std::cos(stacktop[i]);
  }

  // Vectorized tan
  void tan() {
    for (int i = 0; i < VectorSize; ++i)
      stacktop[i] = std::tan(stacktop[i]);
    // reference of JITAsm code test: fast_tanf(stacktop[i]); 
  }

  // Vectorized asin
  void asin() {
    for (int i = 0; i < VectorSize; ++i)
      stacktop[i] = std::asin(stacktop[i]);
  }

  // Vectorized acos
  void acos() {
    for (int i = 0; i < VectorSize; ++i)
      stacktop[i] = std::acos(stacktop[i]);
  }

  // Vectorized atan
  void atan() {
    for (int i = 0; i < VectorSize; ++i)
      stacktop[i] = std::atan(stacktop[i]);
  }

  // Vectorized atan2
  void atan2() {
    stackIndex--;
    auto& prev = stack[stackIndex];
    for (int i = 0; i < VectorSize; ++i)
      stacktop[i] = std::atan2(prev[i], stacktop[i]); // y, x -> -Pi..+Pi
  }

  // Vectorized greater than
  inline void gt() {
    stackIndex--;
    auto& prev = stack[stackIndex];
    for (int i = 0; i < VectorSize; ++i)
      stacktop[i] = (prev[i] > stacktop[i]) ? 1.0f : 0.0f;
  }

  // Vectorized less than
  inline void lt() {
    stackIndex--;
    auto& prev = stack[stackIndex];
    for (int i = 0; i < VectorSize; ++i)
      stacktop[i] = (prev[i] < stacktop[i]) ? 1.0f : 0.0f;
  }

  // Vectorized equal
  inline void eq() {
    stackIndex--;
    auto& prev = stack[stackIndex];
    for (int i = 0; i < VectorSize; ++i)
      stacktop[i] = (prev[i] == stacktop[i]) ? 1.0f : 0.0f; // consider with not 100% match, use epsilon
  }

  // Vectorized not equal
  inline void notEq() {
    stackIndex--;
    auto& prev = stack[stackIndex];
    for (int i = 0; i < VectorSize; ++i)
      stacktop[i] = (prev[i] != stacktop[i]) ? 1.0f : 0.0f; // consider with not 100% match, use epsilon
  }

  // Vectorized less than or equal
  inline void le() {
    stackIndex--;
    auto& prev = stack[stackIndex];
    for (int i = 0; i < VectorSize; ++i)
      stacktop[i] = (prev[i] <= stacktop[i]) ? 1.0f : 0.0f;
  }

  // Vectorized greater than or equal
  inline void ge() {
    stackIndex--;
    auto& prev = stack[stackIndex];
    for (int i = 0; i < VectorSize; ++i)
      stacktop[i] = (prev[i] >= stacktop[i]) ? 1.0f : 0.0f;
  }

  // Vectorized ternary
  inline void ternary() {
    stackIndex -= 2;
    auto& prev = stack[stackIndex];
    auto& prev1 = stack[stackIndex + 1];
    for (int i = 0; i < VectorSize; ++i)
      stacktop[i] = (prev[i] > 0) ? prev1[i] : stacktop[i];
  }

  // Vectorized logical AND
  inline void logicalAnd() {
    stackIndex--;
    auto& prev = stack[stackIndex];
    for (int i = 0; i < VectorSize; ++i)
      stacktop[i] = (stacktop[i] > 0 && prev[i] > 0) ? 1.0f : 0.0f;
  }

  // Vectorized logical OR
  inline void logicalOr() {
    stackIndex--;
    auto& prev = stack[stackIndex];
    for (int i = 0; i < VectorSize; ++i)
      stacktop[i] = (stacktop[i] > 0 || prev[i] > 0) ? 1.0f : 0.0f;
  }

  // Vectorized logical XOR
  inline  void logicalXor() {
    stackIndex--;
    auto& prev = stack[stackIndex];
    for (int i = 0; i < VectorSize; ++i)
      stacktop[i] = ((stacktop[i] > 0) != (prev[i] > 0)) ? 1.0f : 0.0f;
  }

  // Vectorized logical NOT
  inline void logicalNot() {
    for (int i = 0; i < VectorSize; ++i)
      stacktop[i] = (stacktop[i] > 0) ? 0.0f : 1.0f;
  }

  // Vectorized negation
  inline void negSign() {
    for (int i = 0; i < VectorSize; ++i)
      stacktop[i] = -stacktop[i];
  }

  // Templatized store function
  template<typename T, int MaxValue>
  inline void store(T* dst, int x) {
    for (int i = 0; i < VectorSize; ++i)
      dst[x + i] = static_cast<T>(std::max(0.0f, std::min(stacktop[i], static_cast<float>(MaxValue))) + 0.5f);
  }

  // Specialized store function for float
  inline void store(float* dst, int x) {
    for (int i = 0; i < VectorSize; ++i)
      dst[x + i] = stacktop[i];
  }

  // Store variable
  inline void storeVar(int index) {
    for (int i = 0; i < VectorSize; ++i)
      variable_area[index * MAX_C_VECT + i] = stacktop[i];
  }

  // Store variable and drop one element from the stack
  inline void storeVarAndDrop1(int index) {
    for (int i = 0; i < VectorSize; ++i)
      variable_area[index * MAX_C_VECT + i] = stacktop[i];
    stackIndex--;
    if (stackIndex >= 0) {
      auto& prev = stack[stackIndex];
      for (int i = 0; i < VectorSize; ++i)
        stacktop[i] = prev[i];
    }
  }

  // Process a sequence of operations
  void processVector(std::vector<const uint8_t*>& srcp, uint8_t*& dstp, int x, int y) override {

    // Reset stack
    stackIndex = 0;
    broadcast(0); // stacktop = 0

    const ExprOp* vops_current = vops; // reset instruction pointer

    while (true) {
      // Process instruction sequence
      switch (vops_current->op) {
      case opLoadSrc8:
        loadSource<uint8_t>(reinterpret_cast<const uint8_t*>(srcp[vops_current->e.ival]), x);
        break;
      case opLoadSrc16:
        loadSource<uint16_t>(reinterpret_cast<const uint16_t*>(srcp[vops_current->e.ival]), x);
        break;
      case opLoadSrcF32:
        loadSource<float>(reinterpret_cast<const float*>(srcp[vops_current->e.ival]), x);
        break;
      case opLoadRelSrc8:
        loadRelSource<uint8_t>(reinterpret_cast<const uint8_t*>(srcp[vops_current->e.ival]), x, y, vops_current->dx, vops_current->dy, vops_current->bc, vops_current->filler, vops_current->conv, w, h, src_stride[vops_current->e.ival]);
        break;
      case opLoadRelSrc16:
        loadRelSource<uint16_t>(reinterpret_cast<const uint16_t*>(srcp[vops_current->e.ival]), x, y, vops_current->dx, vops_current->dy, vops_current->bc, vops_current->filler, vops_current->conv, w, h, src_stride[vops_current->e.ival]);
        break;
      case opLoadRelSrcF32:
        loadRelSource<float>(reinterpret_cast<const float*>(srcp[vops_current->e.ival]), x, y, vops_current->dx, vops_current->dy, vops_current->bc, vops_current->filler, vops_current->conv, w, h, src_stride[vops_current->e.ival]);
        break;
      case opLoadConst:
        push_and_broadcast(vops_current->e.fval);
        break;
      case opLoadSpatialX:
        loadSpatialX(x);
        break;
      case opLoadSpatialY:
        push_and_broadcast(static_cast<float>(y));
        break;
      case opLoadInternalVar:
        push_and_broadcast(internal_vars[vops_current->e.ival]);
        break;
      case opStore8:
        store<uint8_t, 255>(reinterpret_cast<uint8_t*>(dstp), x);
        goto loopend;
      case opStore10:
        store<uint16_t, 1023>(reinterpret_cast<uint16_t*>(dstp), x);
        goto loopend;
      case opStore12:
        store<uint16_t, 4095>(reinterpret_cast<uint16_t*>(dstp), x);
        goto loopend;
      case opStore14:
        store<uint16_t, 16383>(reinterpret_cast<uint16_t*>(dstp), x);
        goto loopend;
      case opStore16:
        store<uint16_t, 65535>(reinterpret_cast<uint16_t*>(dstp), x);
        goto loopend;
      case opStoreF32:
        store(reinterpret_cast<float*>(dstp), x);
        goto loopend;

      case opDup: dup(vops_current->e.ival); break;
      case opSwap: swap(vops_current->e.ival); break;
      case opAdd: add(); break;
      case opSub: sub(); break;
      case opMul: multiply(); break;
      case opDiv: divide(); break;
      case opMax: max(); break;
      case opMin: min(); break;
      case opSqrt: sqrt(); break;
      case opAbs: abs(); break;
      case opSgn: sgn(); break;
      case opFmod: fmod(); break;
      case opGt: gt(); break;
      case opLt: lt(); break;
      case opEq: eq(); break;
      case opNotEq: notEq(); break;
      case opLE: le(); break;
      case opGE: ge(); break;
      case opTernary: ternary(); break;
      case opAnd: logicalAnd(); break;
      case opOr: logicalOr(); break;
      case opXor: logicalXor(); break;
      case opNeg: logicalNot(); break;
      case opNegSign: negSign(); break;
      case opExp: exp(); break;
      case opLog: log(); break;
      case opPow: pow(); break;
      case opSin: sin(); break;
      case opCos: cos(); break;
      case opTan: tan(); break;
      case opAsin: asin(); break;
      case opAcos: acos(); break;
      case opAtan: atan(); break;
      case opAtan2: atan2(); break;
      case opClip: clip(); break;
      case opRound: round(); break;
      case opFloor: floor(); break;
      case opCeil: ceil(); break;
      case opTrunc: trunc(); break;

      case opStoreVar:
        storeVar(vops_current->e.ival);
        break;
      case opLoadVar:
        loadVar(vops_current->e.ival);
        break;
      case opLoadFramePropVar:
        push_and_broadcast(internal_vars[INTERNAL_VAR_FRAMEPROP_VARIABLES_START + vops_current->e.ival]);
        break;
      case opStoreVarAndDrop1:
        storeVarAndDrop1(vops_current->e.ival);
        break;
      }
      vops_current++; // next opcode
    }
  // store makes the sequence end, wherever it was
  loopend:;
  }
};


// Factory to create appropriate SIMD processors
class SIMDProcessorFactory {
public:
  template<int MaxVectorSize>
  static std::unique_ptr<ISIMDProcessor> createProcessor(
    int w, int h, size_t maxStackSize, 
    aligned_float_vector& var_area,
    std::vector<float>& int_vars, std::vector<int>& src_str, std::vector<const uint8_t*>& src_orig, const ExprOp* vops) {
    if constexpr (MaxVectorSize >= 16) {
      return std::make_unique<SIMDProcessor<16>>(w, h, maxStackSize, var_area, int_vars, src_str, src_orig, vops);
    }
    if constexpr (MaxVectorSize >= 8) {
      return std::make_unique<SIMDProcessor<8>>(w, h, maxStackSize, var_area, int_vars, src_str, src_orig, vops);
    }
    else if constexpr (MaxVectorSize >= 4) {
      return std::make_unique<SIMDProcessor<4>>(w, h, maxStackSize, var_area, int_vars, src_str, src_orig, vops);
    }
    else {
      return std::make_unique<SIMDProcessor<1>>(w, h, maxStackSize, var_area, int_vars, src_str, src_orig, vops);
    }
  }
};


template<int MaxVectorSize>
void processFrameWithDynamicVectors(int plane, int w, int h, int pixels_per_iter, float framecount, float relative_time, int numInputs,
  uint8_t* &dstp, int dst_stride,
  std::vector<const uint8_t*>& srcp, std::vector<int>& src_stride, std::vector<intptr_t>& ptroffsets, std::vector<const uint8_t*>& srcp_orig, ExprData& d) {

  const ExprOp* vops = d.ops[plane].data();
  aligned_float_vector variable_area(MAX_USER_VARIABLES * MAX_C_VECT); // for C, place for expr variables (each is a vector)

  std::vector<float> internal_vars(INTERNAL_VARIABLES + MAX_FRAMEPROP_VARIABLES);
  // frame dependent internal variables
  internal_vars[INTERNAL_VAR_CURRENT_FRAME] = (float)framecount;
  internal_vars[INTERNAL_VAR_RELTIME] = (float)relative_time;
  // followed by dynamic frame properties
  for (auto& framePropToRead : d.frameprops[plane]) {
    int whereToPut = framePropToRead.var_index;
    internal_vars[INTERNAL_VAR_FRAMEPROP_VARIABLES_START + whereToPut] = framePropToRead.value;
  };

  // Create processors dynamically based on MaxVectorSize
  std::unique_ptr<ISIMDProcessor> processor16 = MaxVectorSize >= 16 ?
    SIMDProcessorFactory::createProcessor<16>(w, h, d.maxStackSize, variable_area, internal_vars, src_stride, srcp_orig, vops) : nullptr;
  std::unique_ptr<ISIMDProcessor> processor8 = MaxVectorSize >= 8 ?
    SIMDProcessorFactory::createProcessor<8>(w, h, d.maxStackSize, variable_area, internal_vars, src_stride, srcp_orig, vops) : nullptr;
  std::unique_ptr<ISIMDProcessor> processor4 = MaxVectorSize >= 4 ?
    SIMDProcessorFactory::createProcessor<4>(w, h, d.maxStackSize, variable_area, internal_vars, src_stride, srcp_orig, vops) : nullptr;
  std::unique_ptr<ISIMDProcessor> processor1 = SIMDProcessorFactory::createProcessor<1>(w, h, d.maxStackSize, variable_area, internal_vars, src_stride, srcp_orig, vops);

  for (int y = 0; y < h; y++) {
    int x = 0;

    // Conditionally process larger vector sizes
    if (MaxVectorSize >= 16 && processor16) {
      for (; x + 16 <= w; x += 16) {
        processor16->processVector(srcp, dstp, x, y);
      }
    }

    if (MaxVectorSize >= 8 && processor8) {
      for (; x + 8 <= w; x += 8) {
        processor8->processVector(srcp, dstp, x, y);
      }
    }

    if (MaxVectorSize >= 4 && processor4) {
      for (; x + 4 <= w; x += 4) {
        processor4->processVector(srcp, dstp, x, y);
      }
    }

    // Always process remaining pixels
    for (; x < w; x++) {
      processor1->processVector(srcp, dstp, x, y);
    }

    // Update destination and source pointers for next row
    dstp += dst_stride;
    if (d.lutmode == 0) {
      for (int i = 0; i < numInputs; i++)
        srcp[i] += src_stride[i];
    }
  }
}

void Exprfilter::processFrame(int plane, int w, int h, int pixels_per_iter, float framecount, float relative_time, int numInputs, 
  uint8_t*& dstp, int dst_stride,
  std::vector<const uint8_t*>& srcp, std::vector<int>& src_stride, std::vector<intptr_t>& ptroffsets, std::vector<const uint8_t*>& srcp_orig)
{
#ifdef VS_TARGET_CPU_X86
  if (optSSE2 && d.planeOptSSE2[plane]) {

    int nfulliterations = w / pixels_per_iter;

    ExprData::ProcessLineProc proc = d.proc[plane];

    alignas(32) intptr_t rwptrs[RWPTR_SIZE]; // should work, gcc 8.3 gives false warning
    
    *reinterpret_cast<float*>(&rwptrs[RWPTR_START_OF_INTERNAL_VARIABLES + INTERNAL_VAR_CURRENT_FRAME]) = (float)framecount;
    *reinterpret_cast<float*>(&rwptrs[RWPTR_START_OF_INTERNAL_VARIABLES + INTERNAL_VAR_RELTIME]) = (float)relative_time;
    // refresh frame properties
    for (auto& framePropToRead : d.frameprops[plane]) {
      int whereToPut = framePropToRead.var_index;
      *reinterpret_cast<float*>(&rwptrs[RWPTR_START_OF_INTERNAL_FRAMEPROP_VARIABLES + whereToPut]) = framePropToRead.value;
    };
    for (int y = 0; y < h; y++) {
      rwptrs[RWPTR_START_OF_OUTPUT] = reinterpret_cast<intptr_t>(dstp + dst_stride * y);
      rwptrs[RWPTR_START_OF_XCOUNTER] = 0; // xcounter internal variable
      for (int i = 0; i < numInputs; i++) {
        rwptrs[i + RWPTR_START_OF_INPUTS] = reinterpret_cast<intptr_t>(srcp[i] + src_stride[i] * y); // input pointers 1..Nth
        rwptrs[i + RWPTR_START_OF_STRIDES] = static_cast<intptr_t>(src_stride[i]);
      }
      // a single line at a time
      proc(rwptrs, ptroffsets.data(), nfulliterations, y); // parameters are put directly in registers
    }
  }
  else
#endif // VS_TARGET_CPU_X86
  if (optVectorC) {
    // SIMD factory, vector friendly C version 16 then 8, 4, 1 floats at a time
    // Even if the compiler does not vectorize, we have less overhead during the opcode flow processing
    // As of 2025: original:1-2.5fps, new MAX_C_VECT=16: MSVC~6fps MSVC AVX2:~6fps, LLVM-14,7fps, LLVM AVX2-19fps

    processFrameWithDynamicVectors<MAX_C_VECT>(
      plane, w, h, pixels_per_iter, framecount, relative_time, numInputs,
      dstp, dst_stride,
      srcp, src_stride, ptroffsets, srcp_orig, d);
  }
  else
  {
    // C version, single pixel/loop reference
    std::vector<float> stackVector(d.maxStackSize);

    const ExprOp* vops = d.ops[plane].data();
    float* stack = stackVector.data();
    float stacktop = 0;

    std::vector<float> variable_area(MAX_USER_VARIABLES); // for C, place for expr variables A..Z
    std::vector<float> internal_vars(INTERNAL_VARIABLES + MAX_FRAMEPROP_VARIABLES);
    internal_vars[INTERNAL_VAR_CURRENT_FRAME] = (float)framecount;
    internal_vars[INTERNAL_VAR_RELTIME] = (float)relative_time;
    // followed by dynamic frame properties
    for (auto& framePropToRead : d.frameprops[plane]) {
      int whereToPut = framePropToRead.var_index;
      internal_vars[INTERNAL_VAR_FRAMEPROP_VARIABLES_START + whereToPut] = framePropToRead.value;
    };

    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        int si = 0;
        int i = -1;
        while (true) {
          i++;
          switch (vops[i].op) {
          case opLoadSpatialX:
            stack[si] = stacktop;
            stacktop = (float)x;
            ++si;
            break;
          case opLoadSpatialY:
            stack[si] = stacktop;
            stacktop = (float)y;
            ++si;
            break;
          case opLoadInternalVar:
            stack[si] = stacktop;
            stacktop = internal_vars[vops[i].e.ival];
            ++si;
            break;
          case opLoadFramePropVar:
            stack[si] = stacktop;
            stacktop = internal_vars[INTERNAL_VAR_FRAMEPROP_VARIABLES_START + vops[i].e.ival];
            ++si;
            break;
          case opLoadSrc8:
            stack[si] = stacktop;
            stacktop = srcp[vops[i].e.ival][x];
            ++si;
            break;
          case opLoadSrc16:
            stack[si] = stacktop;
            stacktop = reinterpret_cast<const uint16_t*>(srcp[vops[i].e.ival])[x];
            ++si;
            break;
          case opLoadSrcF32:
            stack[si] = stacktop;
            stacktop = reinterpret_cast<const float*>(srcp[vops[i].e.ival])[x];
            ++si;
            break;
          case opLoadRelSrc8:
            stack[si] = stacktop;
            {
              const int newx = x + vops[i].dx;
              const int newy = y + vops[i].dy;
              const int clipIndex = vops[i].e.ival;
              if (vops[i].bc == borderConstant && (newx < 0 || newx >= w || newy < 0 || newy >= h))
                stacktop = vops[i].filler;
              else {
                const uint8_t* srcp2 = srcp_orig[clipIndex] + borderIndex(newy, h, vops[i].bc) * src_stride[clipIndex];
                stacktop = vops[i].conv.apply(srcp2[borderIndex(newx, w, vops[i].bc)]); // conv: borderConstant only
              }
            }
            ++si;
            break;
          case opLoadRelSrc16:
            stack[si] = stacktop;
            {
              const int newx = x + vops[i].dx;
              const int newy = y + vops[i].dy;
              const int clipIndex = vops[i].e.ival;
              if (vops[i].bc == borderConstant && (newx < 0 || newx >= w || newy < 0 || newy >= h))
                stacktop = vops[i].filler;
              else {
                const uint16_t* srcp2 = reinterpret_cast<const uint16_t*>(srcp_orig[clipIndex] + borderIndex(newy, h, vops[i].bc) * src_stride[clipIndex]);
                stacktop = vops[i].conv.apply(srcp2[borderIndex(newx, w, vops[i].bc)]); // conv: borderConstant only
              }
            }
            ++si;
            break;
          case opLoadRelSrcF32:
            stack[si] = stacktop;
            {
              const int newx = x + vops[i].dx;
              const int newy = y + vops[i].dy;
              const int clipIndex = vops[i].e.ival;
              if (vops[i].bc == borderConstant && (newx < 0 || newx >= w || newy < 0 || newy >= h))
                stacktop = vops[i].filler;
              else {
                const float* srcp2 = reinterpret_cast<const float*>(srcp_orig[clipIndex] + borderIndex(newy, h, vops[i].bc) * src_stride[clipIndex]);
                stacktop = vops[i].conv.apply(srcp2[borderIndex(newx, w, vops[i].bc)]); // conv: borderConstant only
              }
            }
            ++si;
            break;
          case opLoadConst:
            stack[si] = stacktop;
            stacktop = vops[i].e.fval;
            ++si;
            break;
          case opLoadVar:
            stack[si] = stacktop;
            stacktop = variable_area[vops[i].e.ival];
            ++si;
            break;
          case opDup:
            stack[si] = stacktop;
            stacktop = stack[si - vops[i].e.ival];
            ++si;
            break;
          case opSwap:
            std::swap(stacktop, stack[si - vops[i].e.ival]);
            break;
          case opAdd:
            --si;
            stacktop += stack[si];
            break;
          case opSub:
            --si;
            stacktop = stack[si] - stacktop;
            break;
          case opMul:
            --si;
            stacktop *= stack[si];
            break;
          case opDiv:
            --si;
            stacktop = stack[si] / stacktop;
            break;
          case opFmod:
            --si;
            stacktop = std::fmod(stack[si], stacktop);
            break;
          case opMax:
            --si;
            stacktop = std::max(stacktop, stack[si]);
            break;
          case opMin:
            --si;
            stacktop = std::min(stacktop, stack[si]);
            break;
          case opExp:
            stacktop = std::exp(stacktop);
            break;
          case opLog:
            stacktop = std::log(stacktop);
            break;
          case opPow:
            --si;
            stacktop = std::pow(stack[si], stacktop);
            break;
          case opClip:
            // clip(a, low, high) = min(max(a, low),high)
            si -= 2;
            stacktop = std::max(std::min(stack[si], stacktop), stack[si + 1]);
            break;
          case opRound:
            stacktop = std::round(stacktop);
            break;
          case opFloor:
            stacktop = std::floor(stacktop);
            break;
          case opCeil:
            stacktop = std::ceil(stacktop);
            break;
          case opTrunc:
            stacktop = std::trunc(stacktop);
            break;
          case opSqrt:
            stacktop = std::sqrt(stacktop);
            break;
          case opAbs:
            stacktop = std::abs(stacktop);
            break;
          case opSgn:
            stacktop = stacktop < 0 ? -1.0f : stacktop > 0 ? 1.0f : 0.0f;
            break;
          case opSin:
            stacktop = std::sin(stacktop);
            break;
          case opCos:
            stacktop = std::cos(stacktop);
            break;
          case opTan:
            stacktop = std::tan(stacktop);
            break;
          case opAsin:
            stacktop = std::asin(stacktop);
            break;
          case opAcos:
            stacktop = std::acos(stacktop);
            break;
          case opAtan:
            stacktop = std::atan(stacktop);
            break;
          case opAtan2:
            --si;
            stacktop = std::atan2(stack[si], stacktop); // y, x -> -Pi..+Pi
            break;
          case opGt:
            --si;
            stacktop = (stack[si] > stacktop) ? 1.0f : 0.0f;
            break;
          case opLt:
            --si;
            stacktop = (stack[si] < stacktop) ? 1.0f : 0.0f;
            break;
          case opEq:
            --si;
            stacktop = (stack[si] == stacktop) ? 1.0f : 0.0f;
            break;
          case opNotEq:
            --si;
            stacktop = (stack[si] != stacktop) ? 1.0f : 0.0f;
            break;
          case opLE:
            --si;
            stacktop = (stack[si] <= stacktop) ? 1.0f : 0.0f;
            break;
          case opGE:
            --si;
            stacktop = (stack[si] >= stacktop) ? 1.0f : 0.0f;
            break;
          case opTernary:
            si -= 2;
            stacktop = (stack[si] > 0) ? stack[si + 1] : stacktop;
            break;
          case opAnd:
            --si;
            stacktop = (stacktop > 0 && stack[si] > 0) ? 1.0f : 0.0f;
            break;
          case opOr:
            --si;
            stacktop = (stacktop > 0 || stack[si] > 0) ? 1.0f : 0.0f;
            break;
          case opXor:
            --si;
            stacktop = ((stacktop > 0) != (stack[si] > 0)) ? 1.0f : 0.0f;
            break;
          case opNeg:
            stacktop = (stacktop > 0) ? 0.0f : 1.0f;
            break;
          case opNegSign:
            stacktop = -stacktop;
            break;
          case opStore8:
            dstp[x] = (uint8_t)(std::max(0.0f, std::min(stacktop, 255.0f)) + 0.5f);
            goto loopend;
          case opStore10:
            reinterpret_cast<uint16_t*>(dstp)[x] = (uint16_t)(std::max(0.0f, std::min(stacktop, 1023.0f)) + 0.5f);
            goto loopend;
          case opStore12:
            reinterpret_cast<uint16_t*>(dstp)[x] = (uint16_t)(std::max(0.0f, std::min(stacktop, 4095.0f)) + 0.5f);
            goto loopend;
          case opStore14:
            reinterpret_cast<uint16_t*>(dstp)[x] = (uint16_t)(std::max(0.0f, std::min(stacktop, 16383.0f)) + 0.5f);
            goto loopend;
          case opStore16:
            reinterpret_cast<uint16_t*>(dstp)[x] = (uint16_t)(std::max(0.0f, std::min(stacktop, 65535.0f)) + 0.5f);
            goto loopend;
          case opStoreF32:
            reinterpret_cast<float*>(dstp)[x] = stacktop;
            goto loopend;
          case opStoreVar:
            variable_area[vops[i].e.ival] = stacktop;
            break;
          case opStoreVarAndDrop1:
            variable_area[vops[i].e.ival] = stacktop;
            --si;
            if (si >= 0)
              stacktop = stack[si];
            break;
          }
        }
      loopend:;
      }
      dstp += dst_stride;
      if (d.lutmode == 0) {
        for (int i = 0; i < numInputs; i++)
          srcp[i] += src_stride[i];
      }
    }
  }
}

void Exprfilter::preReadFrameProps(int plane, std::vector<PVideoFrame>& src, IScriptEnvironment* env)
{
  for (auto& framePropToRead : d.frameprops[plane]) {
    int srcIndex = framePropToRead.srcIndex;
    auto fpname = framePropToRead.name;

    const AVSMap* avsmap = env->getFramePropsRO(src[srcIndex]);

    // default is 0f
    float varToStore = 0.0f; //  std::numeric_limits<float>::quiet_NaN();

    char res = env->propGetType(avsmap, fpname.c_str());
    // 'u'nset, 'i'nteger, 'f'loat, 's'string, 'c'lip, 'v'ideoframe, 'm'ethod };

    int error;
    // only float and int are valid
    // cast to float: Expr supports 32 bit float, no double, 
    if (res == 'i') {
      int64_t result = env->propGetInt(avsmap, fpname.c_str(), 0, &error);
      if (!error) varToStore = static_cast<float>(result);
    }
    else if (res == 'f') {
      float result = env->propGetFloatSaturated(avsmap, fpname.c_str(), 0, &error);
      if (!error) varToStore = result;
    }

    framePropToRead.value = varToStore;
  }
}

void Exprfilter::calculate_lut(IScriptEnvironment* env)
{
  // ExprData d class variable already filled

  // Only when there are frame props.
  // frame property set from GetFrame(0) is treated as Clip prop

  std::vector<PVideoFrame> src;

  bool frameprops = false;
  for (int plane = 0; plane < d.vi.NumComponents(); plane++) {
    if (d.frameprops[plane].size() > 0) {
      frameprops = true;
      break;
    }
  }

  if (frameprops) {
    src.reserve(children.size());
    // fetch 0th frame only when needed in lut
    for (size_t i = 0; i < children.size(); i++) {
      const auto& child = children[i];
      src.emplace_back(child->GetFrame(0, env));
    }
  }

  std::vector<const uint8_t*> srcp(MAX_EXPR_INPUTS);
  std::vector<const uint8_t*> srcp_orig(MAX_EXPR_INPUTS);
  std::vector<int> src_stride(MAX_EXPR_INPUTS);

  for (int plane = 0; plane < d.vi.NumComponents(); plane++) {
    // calculate only if plane is processed
    if (d.plane[plane] != poProcess)
      continue;

    // read actually needed frame properties into the variable storage area
    preReadFrameProps(plane, src, env);

    uint8_t* dstp;
    int dst_stride;
    int h, w;

    // no buffer allocated yet, prepare lut target buffer, and fake input frame dimensions
    const int bits_per_pixel = d.vi.BitsPerComponent();
    const int pixelsize = d.vi.ComponentSize();
    const auto lut1d_size = (1 << bits_per_pixel); // 1 or 2 bytes per entry
    const auto lut1d_bytesize = lut1d_size * pixelsize;
    const auto lut_size = d.lutmode == 1 ? lut1d_bytesize : lut1d_bytesize * lut1d_bytesize;
    // buffer start must be aligned to at least 32 bytes for avx2.
    // Size must be mod64 but it is fulfilled always.
    d.luts[plane] = (uint8_t *)avs_malloc(lut_size, 32); // 256 lut_x    65536: lut_xy (8 bit)
    dstp = d.luts[plane];
    dst_stride = lut1d_bytesize;
    h = lutmode == 1 ? 1 : lut1d_size; // 1x256, 256x256. 10 bit: 1024, 1024x1024
    w = lut1d_size;

    // for simd:
    // same as in GetFrame
    const int pixels_per_iter = (optAvx2 && d.planeOptAvx2[plane]) ? (optSingleMode ? 8 : 16) : (optSingleMode ? 4 : 8);
    std::vector<intptr_t> ptroffsets(1 + 1 + MAX_EXPR_INPUTS);
    ptroffsets[RWPTR_START_OF_OUTPUT] = d.vi.ComponentSize() * pixels_per_iter; // stepping for output pointer
    ptroffsets[RWPTR_START_OF_XCOUNTER] = pixels_per_iter; // stepping for xcounter

    // no srcp pointers in lut. Technically only inputless sx,sy relative coordinates are in there
    for (int i = 0; i < d.numInputs; i++) {
      srcp[i] = nullptr;
      srcp_orig[i] = nullptr;
      src_stride[i] = 0;
      ptroffsets[RWPTR_START_OF_INPUTS + i] = 0;
    }

    const int dummy_framecount = 0;
    const int dummy_relative_time = 0;
    processFrame(plane, w, h, pixels_per_iter, dummy_framecount, dummy_relative_time, d.numInputs, dstp, dst_stride, srcp, src_stride, ptroffsets, srcp_orig);
  } // for planes
}

template<typename pixel_t, int bits_per_pixel>
static void do_lut_xy(const uint8_t* lut8, uint8_t* dstp, int dst_stride, const uint8_t** srcp, const int* src_stride, int w, int h)
{
  const int max_pixel_value = (1 << bits_per_pixel) - 1;
  const pixel_t* lut = reinterpret_cast<const pixel_t*>(lut8);
  const uint8_t* src0 = srcp[0];
  const uint8_t* src1 = srcp[1];
  const auto pitch0 = src_stride[0];
  const auto pitch1 = src_stride[1];
  for (auto y = 0; y < h; y++) {
    for (auto x = 0; x < w; x++) {
      if constexpr (bits_per_pixel == 8 || bits_per_pixel == 16) {
        // no limit check
        const int pixel0 = reinterpret_cast<const pixel_t*>(src0)[x];
        const int pixel1 = reinterpret_cast<const pixel_t*>(src1)[x];
        reinterpret_cast<pixel_t*>(dstp)[x] = lut[(pixel1 << bits_per_pixel) + pixel0];
      }
      else {
        const int pixel0 = min((int)reinterpret_cast<const pixel_t*>(src0)[x], max_pixel_value);
        const int pixel1 = min((int)reinterpret_cast<const pixel_t*>(src1)[x], max_pixel_value);
        reinterpret_cast<pixel_t*>(dstp)[x] = lut[(pixel1 << bits_per_pixel) + pixel0];
      }
    }
    src0 += pitch0;
    src1 += pitch1;
    dstp += dst_stride;
  }
}

PVideoFrame __stdcall Exprfilter::GetFrame(int n, IScriptEnvironment *env) {
  // ExprData d class variable already filled

  std::vector<PVideoFrame> src;
  src.reserve(children.size());

  int first_used_clip_index = -1;
  for (size_t i = 0; i < children.size(); i++) {
    const auto &child = children[i];
    src.emplace_back(d.clipsUsed[i] ? child->GetFrame(n, env) : nullptr); // GetFrame only when really referenced
    if (first_used_clip_index < 0) {
      if (d.clipsUsed[i])
        first_used_clip_index = (int)i; // inherit frameprop from
    }
  }

  PVideoFrame dst;
  if (first_used_clip_index >= 0)
    dst = env->NewVideoFrameP(d.vi, &src[first_used_clip_index]);
  else
    dst = env->NewVideoFrame(d.vi);

  std::vector<const uint8_t*> srcp(MAX_EXPR_INPUTS);
  std::vector<const uint8_t*> srcp_orig(MAX_EXPR_INPUTS);
  std::vector<int> src_stride(MAX_EXPR_INPUTS);

  const float framecount = (float)n; // max precision: 2^24 (16M) frames (32 bit float precision)
  const float relative_time = vi.num_frames > 1 ? (float)((double)n / (vi.num_frames - 1)) : 0.0f; // 0 <= time <= 1
  const int planes_yuva[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
  const int planes_ya[2]   = { PLANAR_Y, PLANAR_A };
  const int planes_r[4]    = { PLANAR_R, PLANAR_G, PLANAR_B, PLANAR_A }; // expression string order is R G B unlike internal G B R plane order
  const int *plane_enums_d = d.vi.IsYA() ? planes_ya : (d.vi.IsYUV() || d.vi.IsYUVA()) ? planes_yuva : planes_r;

  for (int plane = 0; plane < d.vi.NumComponents(); plane++) {

    const int plane_enum_d = plane_enums_d[plane];

    if (d.plane[plane] == poProcess) {

      // read actually needed frame properties into the variable storage area
      preReadFrameProps(plane, src, env);

      uint8_t* dstp;
      int dst_stride;
      int h, w;

      dstp = dst->GetWritePtr(plane_enum_d);
      dst_stride = dst->GetPitch(plane_enum_d);
      h = d.vi.height >> d.vi.GetPlaneHeightSubsampling(plane_enum_d);
      w = d.vi.width >> d.vi.GetPlaneWidthSubsampling(plane_enum_d);

      // for simd:
      const int pixels_per_iter = (optAvx2 && d.planeOptAvx2[plane]) ? (optSingleMode ? 8 : 16) : (optSingleMode ? 4 : 8);
      std::vector<intptr_t> ptroffsets(1 + 1 + MAX_EXPR_INPUTS);
      ptroffsets[RWPTR_START_OF_OUTPUT] = d.vi.ComponentSize() * pixels_per_iter; // stepping for output pointer
      ptroffsets[RWPTR_START_OF_XCOUNTER] = pixels_per_iter; // stepping for xcounter

      for (int i = 0; i < d.numInputs; i++) {
        if (d.clips[i]) {
          if (d.clipsUsed[i]) {
            // when input is a single Y, use PLANAR_Y instead of the plane matching to the output
            // Check IsYA() before IsYUVA(): IsYUVA() is also true for YA
            const VideoInfo& vi_src = d.clips[i]->GetVideoInfo();
            const int* plane_enums_s = vi_src.IsYA() ? planes_ya : (vi_src.IsYUV() || vi_src.IsYUVA()) ? planes_yuva : planes_r;
            const int plane_enum_s = vi_src.IsY() ? PLANAR_Y : plane_enums_s[plane];

            srcp[i] = src[i]->GetReadPtr(plane_enum_s);
            // C only:
            srcp_orig[i] = srcp[i];
            src_stride[i] = src[i]->GetPitch(plane_enum_s);
            // SIMD only
            ptroffsets[RWPTR_START_OF_INPUTS + i] = d.clips[i]->GetVideoInfo().ComponentSize() * pixels_per_iter; // 1..Nth: inputs
          }
          else {
            srcp[i] = nullptr;
            srcp_orig[i] = nullptr;
            src_stride[i] = 0;
            ptroffsets[RWPTR_START_OF_INPUTS + i] = 0;
          }
        }
      }

      if (lutmode == 0) {
        processFrame(plane, w, h, pixels_per_iter, framecount, relative_time, d.numInputs, dstp, dst_stride, srcp, src_stride, ptroffsets, srcp_orig);
      } else {
        // lut table for plane is filled, do lookup now
        const int bits_per_pixel = d.vi.BitsPerComponent();

        if (d.lutmode == 1) {
          // lut_x
          if (bits_per_pixel == 8)
          {
            uint8_t* lut = d.luts[plane];
            const uint8_t* src0 = srcp[0];
            const auto pitch0 = src_stride[0];
            for (auto y = 0; y < h; y++) {
              for (auto x = 0; x < w; x++) {
                const int pixel = src0[x];
                dstp[x] = lut[pixel];
              }
              src0 += pitch0;
              dstp += dst_stride;
            }
          }
          else {
            const int max_pixel_value = (1 << bits_per_pixel) - 1;
            uint16_t* lut = reinterpret_cast<uint16_t*>(d.luts[plane]);
            const uint8_t* src0 = srcp[0];
            const auto pitch0 = src_stride[0];
            if (bits_per_pixel == 16) {
              // no limit check
              for (auto y = 0; y < h; y++) {
                for (auto x = 0; x < w; x++) {
                  const int pixel = reinterpret_cast<const uint16_t*>(src0)[x];
                  reinterpret_cast<uint16_t*>(dstp)[x] = lut[pixel];
                }
                src0 += pitch0;
                dstp += dst_stride;
              }
            }
            else {
              for (auto y = 0; y < h; y++) {
                for (auto x = 0; x < w; x++) {
                  const int pixel = reinterpret_cast<const uint16_t*>(src0)[x];
                  reinterpret_cast<uint16_t*>(dstp)[x] = lut[min(pixel, max_pixel_value)]; // e.g. 10 bits in 2 byte safety
                }
                src0 += pitch0;
                dstp += dst_stride;
              }
            }
          }
        }
        else if (d.lutmode == 2) {
          // lut_xy
          // templates for speed: bitshift with immediate constant
          const uint8_t* lut = d.luts[plane];
          if (bits_per_pixel == 8)
            do_lut_xy<uint8_t, 8>(lut, dstp, dst_stride, srcp_orig.data(), src_stride.data(), w, h);
          else if (bits_per_pixel == 10)
            do_lut_xy<uint16_t, 10>(lut, dstp, dst_stride, srcp_orig.data(), src_stride.data(), w, h);
          else if (bits_per_pixel == 12)
            do_lut_xy<uint16_t, 12>(lut, dstp, dst_stride, srcp_orig.data(), src_stride.data(), w, h);
          else if (bits_per_pixel == 14)
            do_lut_xy<uint16_t, 14>(lut, dstp, dst_stride, srcp_orig.data(), src_stride.data(), w, h);
          else if (bits_per_pixel == 16) // well, this is not enabled 16bit lutxy would take a 8GB table
            do_lut_xy<uint16_t, 16>(lut, dstp, dst_stride, srcp_orig.data(), src_stride.data(), w, h);
          else
            assert(0);
        }
        else { // 1d lut, 2d lut
          assert(0); // no lut_xyz
        }
      } // lut branch
    }
    // avs+: copy plane here
    else if (d.plane[plane] == poCopy) {
      // avs+ copy from Nth clip
      const int copySource = d.planeCopySourceClip[plane];
      // when input is a single Y, use PLANAR_Y instead of the plane matching to the output
      const VideoInfo& vi_src = d.clips[copySource]->GetVideoInfo();
      const int plane_enum_s = vi_src.IsY() ? PLANAR_Y : plane_enums_d[plane];

      env->BitBlt(dst->GetWritePtr(plane_enum_d), dst->GetPitch(plane_enum_d),
        src[copySource]->GetReadPtr(plane_enum_s),
        src[copySource]->GetPitch(plane_enum_s),
        src[copySource]->GetRowSize(plane_enum_s),
        src[copySource]->GetHeight(plane_enum_s)
      );
    }
    else if (d.plane[plane] == poFill) { // avs+
      uint8_t *dstp = dst->GetWritePtr(plane_enum_d);
      const int dst_rowsize = dst->GetRowSize(plane_enum_d);
      const int dst_stride = dst->GetPitch(plane_enum_d);
      const int h = dst->GetHeight(plane_enum_d);

      const int bits_per_pixel = vi.BitsPerComponent();

      float val = d.planeFillValue[plane];
      int val_i = 0;
      if (bits_per_pixel <= 16) {
        const int max_pixel_value = (1 << bits_per_pixel) - 1;
        val_i = (int)(std::max(0.0f, std::min(val, (float)max_pixel_value)) + 0.5f);
      }

      if(bits_per_pixel == 8)
        fill_plane<BYTE>(dstp, h, dst_rowsize, dst_stride, val_i);
      else if(bits_per_pixel <= 16)
        fill_plane<uint16_t>(dstp, h, dst_rowsize, dst_stride, val_i);
      else // 32 bit float
        fill_plane<float>(dstp, h, dst_rowsize, dst_stride, val);

    } // plane modes
  } // for planes

  return dst;
}

Exprfilter::~Exprfilter() {
  for (int i = 0; i < MAX_EXPR_INPUTS; i++)
    d.clips[i] = nullptr;
  for (int i = 0; i < 4; i++)
    if(d.luts[i]) avs_free(d.luts[i]); // aligned free
}

static SOperation getLoadOp(const VideoInfo *vi, bool relativeKind) {
  if (!vi)
    return relativeKind ? opLoadRelSrcF32 : opLoadSrcF32;
  if (vi->BitsPerComponent() == 32) // float, avs has no f16c float
    return relativeKind ? opLoadRelSrcF32 : opLoadSrcF32;
  else if (vi->BitsPerComponent() == 8)
    return relativeKind ? opLoadRelSrc8 : opLoadSrc8;
  else
    return relativeKind ? opLoadRelSrc16 : opLoadSrc16; // 10..16 bits common
}

static SOperation getStoreOp(const VideoInfo *vi) {
  // avs+ has no f16c float
  switch (vi->BitsPerComponent()) {
  case 8: return opStore8;
  case 10: return opStore10; // avs+
  case 12: return opStore12; // avs+
  case 14: return opStore14; // avs+
  case 16: return opStore16;
  case 32: return opStoreF32;
  default: return opStoreF32;
  }
}

#define LOAD_OP(op,v,req) do { if (stackSize < req) env->ThrowError("Expr: Not enough elements on stack to perform operation %s", tokens[i].c_str()); ops.push_back(ExprOp(op, (v))); maxStackSize = std::max(++stackSize, maxStackSize); } while(0)
#define LOAD_REL_OP(op,v,req,dx,dy,bc) do { if (stackSize < req) env->ThrowError("Expr: Not enough elements on stack to perform operation %s", tokens[i].c_str()); ops.push_back(ExprOp(op, (v), (dx), (dy), (bc))); maxStackSize = std::max(++stackSize, maxStackSize); } while(0)
#define GENERAL_OP(op, v, req, dec) do { if (stackSize < req) env->ThrowError("Expr: Not enough elements on stack to perform operation %s", tokens[i].c_str()); ops.push_back(ExprOp(op, (v))); stackSize-=(dec); } while(0)
#define ONE_ARG_OP(op) GENERAL_OP(op, 0, 1, 0)
#define VAR_STORE_OP(op,v) GENERAL_OP(op, v, 1, 0)
#define VAR_STORE_SPEC_OP(op,v) GENERAL_OP(op, v, 1, 1)
#define TWO_ARG_OP(op) GENERAL_OP(op, 0, 2, 1)
#define THREE_ARG_OP(op) GENERAL_OP(op, 0, 3, 2)
// defines for special scale-back-before-store where no token is in context:
#define LOAD_OP_NOTOKEN(op,v,req) do { if (stackSize < req) env->ThrowError("Expr: Not enough elements on stack to perform a load operation"); ops.push_back(ExprOp(op, (v))); maxStackSize = std::max(++stackSize, maxStackSize); } while(0)
#define GENERAL_OP_NOTOKEN(op, v, req, dec) do { if (stackSize < req) env->ThrowError("Expr: Not enough elements on stack to perform an operation"); ops.push_back(ExprOp(op, (v))); stackSize-=(dec); } while(0)
#define TWO_ARG_OP_NOTOKEN(op) GENERAL_OP_NOTOKEN(op, 0, 2, 1)

static bool isValidVarName(const std::string& s) {
  size_t len = s.length();
  if (!len)
    return false;

  if (!isAlphaUnderscore(s[0]))
    return false;
  for (size_t i = 1; i < len; i++)
    if (!isAlphaNumUnderscore(s[i]))
      return false;
  return true;
}


// finds _X suffix (clip letter) and returns 0..25 for x,y,z,a,b,...w
// no suffix means 0
static int getSuffix(std::string token, std::string base) {
  size_t len = base.length();

  if (token.substr(0, len) != base)
    return -1; // no match

  if (token.length() == len)
    return 0; // no suffix, treat as _x

    // find _X suffix, where X is x,y,z,a..w
  if (token.length() != len + 2 || token[len] != '_')
    return -2; // no proper suffix

  char srcChar = token[len + 1]; // last char
  int loadIndex;
  if (srcChar >= 'x')
    loadIndex = srcChar - 'x';
  else
    loadIndex = srcChar - 'a' + 3;
  return loadIndex;
}

// if automatic source bit depth conversion takes place, ymax, ymin, range_xxx, etc.. constants are changed accordingly
static int getEffectiveBitsPerComponent(int bitsPerComponent, bool autoconv_conv_float, bool autoconv_conv_int, int autoScaleSourceBitDepth)
{
  if ((autoconv_conv_float && bitsPerComponent == 32) || (autoconv_conv_int && bitsPerComponent != 32))
    return autoScaleSourceBitDepth;
  return bitsPerComponent;
}

static size_t parseExpression(const std::string &expr, std::vector<ExprOp> &ops, std::vector<ExprFramePropData>& fp, const VideoInfo **vi, const VideoInfo *vi_output, const SOperation storeOp, int numInputs, int planewidth, int planeheight, bool chroma,
  const bool autoconv_full_scale, const bool autoconv_conv_int, const bool autoconv_conv_float, const int clamp_float_i, const bool shift_float, const int lutmode,
  IScriptEnvironment *env)
{
    // vi_output is new in avs+, and is not used yet

    // optional scaling, scale_from bit depth, default scale_to bitdepth, used in scaleb and scalef (yscaleb, yscalef)
    int targetBitDepth = vi[0]->BitsPerComponent(); // avs+
    int autoScaleSourceBitDepth = 8; // avs+ scalable constants are in 8 bit range by default

    std::vector<std::string> tokens;
    split(tokens, expr, " \r\n\t", split1::no_empties);

    std::unordered_map<std::string, int> varnames;
    int varindex = 0;
    std::unordered_map<std::string, int> fpnames;
    int fpindex = 0;

    size_t maxStackSize = 0;
    size_t stackSize = 0;

    for (size_t i = 0; i < tokens.size(); i++) {
        const size_t tokenlen = tokens[i].length();
        if (tokens[i] == "+")
            TWO_ARG_OP(opAdd);
        else if (tokens[i] == "-")
            TWO_ARG_OP(opSub);
        else if (tokens[i] == "*")
            TWO_ARG_OP(opMul);
        else if (tokens[i] == "/")
            TWO_ARG_OP(opDiv);
        else if (tokens[i] == "%")
            TWO_ARG_OP(opFmod);
        else if (tokens[i] == "max")
            TWO_ARG_OP(opMax);
        else if (tokens[i] == "min")
            TWO_ARG_OP(opMin);
        else if (tokens[i] == "exp")
            ONE_ARG_OP(opExp);
        else if (tokens[i] == "log")
            ONE_ARG_OP(opLog);
        else if (tokens[i] == "pow" || tokens[i] == "^") // avs+: ^ can be used for power
            TWO_ARG_OP(opPow);
        else if (tokens[i] == "sqrt")
            ONE_ARG_OP(opSqrt);
        else if (tokens[i] == "abs")
            ONE_ARG_OP(opAbs);
        else if (tokens[i] == "sgn")
          ONE_ARG_OP(opSgn);
        else if (tokens[i] == "sin")
          ONE_ARG_OP(opSin);
        else if (tokens[i] == "cos")
          ONE_ARG_OP(opCos);
        else if (tokens[i] == "tan")
          ONE_ARG_OP(opTan);
        else if (tokens[i] == "asin")
          ONE_ARG_OP(opAsin);
        else if (tokens[i] == "acos")
          ONE_ARG_OP(opAcos);
        else if (tokens[i] == "atan")
          ONE_ARG_OP(opAtan);
        else if (tokens[i] == "atan2")
          TWO_ARG_OP(opAtan2);
        else if (tokens[i] == "clip")
          THREE_ARG_OP(opClip);
        else if (tokens[i] == "round")
          ONE_ARG_OP(opRound);
        else if (tokens[i] == "floor")
          ONE_ARG_OP(opFloor);
        else if (tokens[i] == "ceil")
          ONE_ARG_OP(opCeil);
        else if (tokens[i] == "trunc")
          ONE_ARG_OP(opTrunc);
        else if (tokens[i] == ">")
            TWO_ARG_OP(opGt);
        else if (tokens[i] == "<")
            TWO_ARG_OP(opLt);
        else if (tokens[i] == "=" || tokens[i] == "==") // avs+: == can be used to equality check
            TWO_ARG_OP(opEq);
        else if (tokens[i] == "!=") // avs+: not equal
          TWO_ARG_OP(opNotEq);
        else if (tokens[i] == ">=")
            TWO_ARG_OP(opGE);
        else if (tokens[i] == "<=")
            TWO_ARG_OP(opLE);
        else if (tokens[i] == "?")
            THREE_ARG_OP(opTernary);
        else if (tokens[i] == "and" || tokens[i] == "&") // avs+: & alias for and
            TWO_ARG_OP(opAnd);
        else if (tokens[i] == "or" || tokens[i] == "|") // avs+: | alias for or
            TWO_ARG_OP(opOr);
        else if (tokens[i] == "xor")
            TWO_ARG_OP(opXor);
        else if (tokens[i] == "not")
            ONE_ARG_OP(opNeg);
        else if (tokens[i] == "neg")
            ONE_ARG_OP(opNegSign);
        else if (tokens[i].substr(0, 3) == "dup")
            if (tokens[i].size() == 3) {
                LOAD_OP(opDup, 0, 1);
            }
            else {
              try {
                int tmp = std::stoi(tokens[i].substr(3));
                if (tmp < 0)
                  env->ThrowError("Expr: Dup suffix can't be less than 0 '%s'", tokens[i].c_str());
                LOAD_OP(opDup, tmp, (size_t)(tmp + 1));
              }
              catch (std::logic_error &) {
                env->ThrowError("Expr: Failed to convert dup suffix '%s' to valid index", tokens[i].c_str());
              }
            }
        else if (tokens[i].substr(0, 4) == "swap")
            if (tokens[i].size() == 4) {
                GENERAL_OP(opSwap, 1, 2, 0);
            }
            else {
              try {
                int tmp = std::stoi(tokens[i].substr(4));
                if (tmp < 1)
                  env->ThrowError("Expr: Swap suffix can't be less than 1 '%s'", tokens[i].c_str());
                GENERAL_OP(opSwap, tmp, (size_t)(tmp + 1), 0);
              }
              catch (std::logic_error &) {
                env->ThrowError("Expr: Failed to convert swap suffix '%s' to valid index", tokens[i].c_str());
              }
            }
        else if (tokens[i] == "sx") { // avs+
          // spatial
          if (lutmode > 0)
            env->ThrowError("Expr: 'sx' is forbidden in lut mode");
          LOAD_OP(opLoadSpatialX, 0, 0);
        }
        else if (tokens[i] == "sy") { // avs+
          // spatial
          if (lutmode > 0)
            env->ThrowError("Expr: 'sy' is forbidden in lut mode");
          LOAD_OP(opLoadSpatialY, 0, 0);
        }
        else if (tokens[i] == "sxr") { // avs+
          // spatial X relative 0..1
          if (lutmode > 0)
            env->ThrowError("Expr: 'sxr' is forbidden in lut mode");
          LOAD_OP(opLoadSpatialX, 0, 0);
          /* Paranoia: precision at rightmost position? Ensure that sxr == 1.0 there
          Multiply by 1/x is different.
          Test:
          constexpr float x = 1919.0f;
          constexpr float y = 1 / 1919.0f;
          constexpr float zz = x * y;
          constexpr bool b = zz == 1.0f; // false!

          const float p = planewidth > 1 ? 1.0f / ((float)planewidth - 1.0f) : 1.0f;
          LOAD_OP(opLoadConst, p, 0);
          TWO_ARG_OP(opMul);
          */
          const float p = planewidth > 1 ? (float)planewidth - 1.0f : 1.0f;
          LOAD_OP(opLoadConst, p, 0);
          TWO_ARG_OP(opDiv);
        }
        else if (tokens[i] == "syr") { // avs+
          // spatial Y relative 0..1
          if (lutmode > 0)
            env->ThrowError("Expr: 'syr' is forbidden in lut mode");
          LOAD_OP(opLoadSpatialY, 0, 0);
          /* Multiply by 1/x is different
          const float p = planeheight > 1 ? 1.0f / ((float)planeheight - 1.0f) : 1.0f;
          LOAD_OP(opLoadConst, p, 0);
          TWO_ARG_OP(opMul);
          */
          const float p = planeheight > 1 ? (float)planeheight - 1.0f : 1.0f;
          LOAD_OP(opLoadConst, p, 0);
          TWO_ARG_OP(opDiv);
        }
        else if (tokens[i] == "frameno") { // avs+
          if (lutmode > 0)
            env->ThrowError("Expr: 'frameno' is forbidden in lut mode");
          LOAD_OP(opLoadInternalVar, INTERNAL_VAR_CURRENT_FRAME, 0);
        }
        else if (tokens[i] == "time") { // avs+
          if (lutmode > 0)
            env->ThrowError("Expr: 'time' is forbidden in lut mode");
          LOAD_OP(opLoadInternalVar, INTERNAL_VAR_RELTIME, 0);
        }
        else if (tokens[i] == "width") { // avs+
          LOAD_OP(opLoadConst, (float)planewidth, 0);
        }
        else if (tokens[i] == "height") { // avs+
          LOAD_OP(opLoadConst, (float)planeheight, 0);
        }
        else if ((tokens[i].length() == 1 || (tokens[i].length() > 1 && tokens[i][1] == '[')) && tokens[i][0] >= 'a' && tokens[i][0] <= 'z') {
          const bool rel = tokens[i].length() > 1; // relative pixel addressing; indexed clips e.g. x[-1,-2]
          // loading source clip pixels
          char srcChar = tokens[i][0];
          int loadIndex;
          if (srcChar >= 'x')
            loadIndex = srcChar - 'x';
          else
            loadIndex = srcChar - 'a' + 3;
          if (loadIndex >= numInputs)
            env->ThrowError("Expr: Too few input clips supplied to reference '%s'", tokens[i].c_str());

          if (rel) {
            int dx, dy;
            std::string s;
            std::istringstream numStream(tokens[i].substr(2)); // after '['
            numStream.imbue(std::locale::classic());
            // first coord
            if (!(numStream >> dx))
              env->ThrowError("Expr: Failed to convert '%s' to integer, relative index dx", tokens[i].c_str());
            // separator ','
            if (numStream.get() != ',')
              env->ThrowError("Expr: Failed to convert '%s', character ',' expected between the coordinates", tokens[i].c_str());
            // second coord
            if (!(numStream >> dy))
              env->ThrowError("Expr: Failed to convert '%s' to integer, relative index dy", tokens[i].c_str());
            // ending ']'
            if (numStream.get() != ']')
              env->ThrowError("Expr: Failed to convert '%s' to [x,y], closing ']' expected ", tokens[i].c_str());
            // optional border mode suffix :c (clamp, default), :m (mirror), :r (reflect101), :k (constant filler from the stack), :w (wrap)
            int bc = borderClamp;
            if (numStream >> s) {
              if (s == ":c") bc = borderClamp;
              else if (s == ":m") bc = borderMirror;
              else if (s == ":r") bc = borderReflect101;
              else if (s == ":k") bc = borderConstant;
              else if (s == ":w") bc = borderWrap;
              else
                env->ThrowError("Expr: Failed to convert '%s', after ']' only :c, :m, :r, :k or :w (border mode) is allowed", tokens[i].c_str());
            }

            if (lutmode > 0)
              env->ThrowError("Expr: relative pixel addressing is forbidden in lut mode");

            if (dx <= -vi_output->width || dx >= vi_output->width)
              env->ThrowError("Expr: dx must be between +/- (width-1) in '%s'", tokens[i].c_str());
            if (dy <= -vi_output->height || dy >= vi_output->height)
              env->ThrowError("Expr: dy must be between +/- (height-1) in '%s'", tokens[i].c_str());
            if (bc == borderConstant) {
              // the filler is the stack top, must be a constant (checked after constant folding):
              // pops it and pushes the loaded pixel: stack size unchanged
              if (stackSize < 1)
                env->ThrowError("Expr: '%s' needs a constant filler value on the stack", tokens[i].c_str());
              ops.push_back(ExprOp(getLoadOp(vi[loadIndex], true), loadIndex, dx, dy, bc));
              // stack size unchanged: maxStackSize was already updated when the filler was pushed
            }
            else
              LOAD_REL_OP(getLoadOp(vi[loadIndex], true), loadIndex, 0, dx, dy, bc);
          }
          else {
            // not relative, single clip letter
            if (lutmode == 0) {
              LOAD_OP(getLoadOp(vi[loadIndex], false), loadIndex, 0);
            }
            else {
              // for lut we replace x and y to sx and sy to make the initialization
              if (loadIndex >= lutmode) // lutx
                env->ThrowError("Expr: more input clips than lut's dimension. Problematic clip: '%s'", tokens[i].c_str());
              // spatial
              if (loadIndex == 0)
                LOAD_OP(opLoadSpatialX, 0, 0);
              else // if (loadIndex == 1)
                LOAD_OP(opLoadSpatialY, 0, 0);
            }
          }

          // avs+: 'scale_inputs': converts input pixels to a common specified range
          // Apply full or limited conversion to integer and/or float bit depths.
          // For integers bit-shift or full-scale-stretch method can be chosen
          // There is no precision loss, since the multiplication/division occurs when original pixels
          // are already loaded as float
          const int srcBitDepth = vi[loadIndex]->BitsPerComponent();
          const int dstBitDepth = autoScaleSourceBitDepth; // internal precision
          const size_t loadOpIndex = ops.size() - 1; // the load op just added
          ExprLoadConv conv; // the conversion steps after the load

          const bool use_chroma = chroma; // && !forceNonUV;
          const bool isfull = autoconv_full_scale;

          if (autoconv_conv_int || autoconv_conv_float) {

            if ((srcBitDepth != 32 && autoconv_conv_int) ||
              ((srcBitDepth == 32 && autoconv_conv_float))) {

              bits_conv_constants d;
              get_bits_conv_constants(d, use_chroma, isfull, isfull, srcBitDepth, dstBitDepth);
              // chroma is spec: signed. Limited:16-240 is really 128 +/-112. Full:1-255 is really 128+/-127
              conv.sub = (float)d.src_offset;
              conv.mul = d.mul_factor;
              conv.add = (float)d.dst_offset;
            }
          }

          // "floatUV" - shifts -0.5 .. +0.5 chroma range to 0 .. 1.0 only when no other autoscaling is active
          // the effect of this pre-shift is reversed at the storage phase (opSub at the end pixel exit)
          if ((!autoconv_conv_float || dstBitDepth == 32) && srcBitDepth == 32) {
            if (chroma && shift_float)
              conv.add_uv = 0.5f;
          }

          if (ops[loadOpIndex].bc == borderConstant) {
            // :k relative load: the load op converts the loaded pixels itself, before putting
            // the filler (given in the expression's domain, no conversion) in place of the
            // off-screen ones.
            ops[loadOpIndex].conv = conv;
          }
          else {
            // separate operations after the load, steps in the order of ExprLoadConv::apply
            if (conv.sub != 0.0f) {
              LOAD_OP(opLoadConst, conv.sub, 0);
              TWO_ARG_OP(opSub);
            }
            if (conv.mul != 1.0f) {
              LOAD_OP(opLoadConst, conv.mul, 0);
              TWO_ARG_OP(opMul);
            }
            if (conv.add != 0.0f) {
              LOAD_OP(opLoadConst, conv.add, 0);
              TWO_ARG_OP(opAdd);
            }
            if (conv.add_uv != 0.0f) {
              LOAD_OP(opLoadConst, conv.add_uv, 0);
              TWO_ARG_OP(opAdd);
            }
          }
        }
        else if (tokens[i] == "pi") // avs+
        {
          float pi = 3.141592653589793f;
          LOAD_OP(opLoadConst, pi, 0);
        }
        // avs+
        // bitdepth: automatic silent parameter of the lut expression(clip bit depth)
        // clip-specific bitdepths: bitdepth_x, bitdepth_y, bitdepth_z, bitdepth_a, .. bitdepth_w,
        // sbitdepth : automatic silent parameter of the lut expression(bit depth of values to scale)
        //
        // pre-defined, bit depth aware constants
        //   range_half : autoscaled 128 or 0.5 for float, (or 0.0 for chroma with zero-base float chroma version)
        //   range_max  : 255 / 1023 / 4095 / 16383 / 65535 or 1.0 for float
        //                                                     0.5 for float chroma (new zero-based style)
        //   range_min  : 0 for 8-16bits, or 0 for float luma, or -0.5 for float chroma
        //                -0.5 for float chroma (new zero-based style)
        //   range_size : 256 / 1024...65536
        //   ymin, ymax : 16 / 235 autoscaled.
        //   cmin, cmax : 16 / 240 autoscaled. For 32bits zero based chroma: (16-128)/255.0, (240-128)/255.0

        // ymin or ymin_x, ymin_y, ymin_y, ymin_a....
        // similarly: ymax, range_max, cmin, cmax, range_half
        // without clip index specifier, or with '_'+letter suffix
        else if (tokens[i].substr(0, 8) == "bitdepth") // avs+
        {
          int loadIndex = -1;
          std::string toFind = "bitdepth";
          if (tokens[i].substr(0, toFind.length()) == toFind)
            loadIndex = getSuffix(tokens[i], toFind);
          if (loadIndex < 0)
            env->ThrowError("Expr: Error in built-in constant expression '%s'", tokens[i].c_str());
          if (loadIndex >= numInputs)
            env->ThrowError("Expr: Too few input clips supplied for reference '%s'", tokens[i].c_str());

          int bitsPerComponent = getEffectiveBitsPerComponent(vi[loadIndex]->BitsPerComponent(), autoconv_conv_float, autoconv_conv_int, autoScaleSourceBitDepth);
          float q = (float)bitsPerComponent;
          LOAD_OP(opLoadConst, q, 0);
        }
        else if (tokens[i] == "sbitdepth") // avs+
        {
          float q = (float)autoScaleSourceBitDepth;
          LOAD_OP(opLoadConst, q, 0);
        }
        else if (tokens[i].substr(0, 4) == "ymin") // avs+
        {
          int loadIndex = -1;
          std::string toFind = "ymin";
          if (tokens[i].substr(0, toFind.length()) == toFind)
            loadIndex = getSuffix(tokens[i], toFind);
          if (loadIndex < 0)
            env->ThrowError("Expr: Error in built-in constant expression '%s'", tokens[i].c_str());
          if (loadIndex >= numInputs)
            env->ThrowError("Expr: Too few input clips supplied for reference '%s'", tokens[i].c_str());

          int bitsPerComponent = getEffectiveBitsPerComponent(vi[loadIndex]->BitsPerComponent(), autoconv_conv_float, autoconv_conv_int, autoScaleSourceBitDepth);
          float q = bitsPerComponent == 32 ? 16.0f / 255 : (16 << (bitsPerComponent - 8)); // scale luma min 16
          LOAD_OP(opLoadConst, q, 0);
        }
        else if (tokens[i].substr(0, 4) == "ymax") // avs+
        {
          int loadIndex = -1;
          std::string toFind = "ymax";
          if (tokens[i].substr(0, toFind.length()) == toFind)
            loadIndex = getSuffix(tokens[i], toFind);
          if (loadIndex < 0)
            env->ThrowError("Expr: Error in built-in constant expression '%s'", tokens[i].c_str());
          if (loadIndex >= numInputs)
            env->ThrowError("Expr: Too few input clips supplied for reference '%s'", tokens[i].c_str());

          int bitsPerComponent = getEffectiveBitsPerComponent(vi[loadIndex]->BitsPerComponent(), autoconv_conv_float, autoconv_conv_int, autoScaleSourceBitDepth);
          float q = bitsPerComponent == 32 ? 235.0f / 255 : (235 << (bitsPerComponent - 8)); // scale luma max 235
          LOAD_OP(opLoadConst, q, 0);
        }
        else if (tokens[i].substr(0, 4) == "cmin") // avs+
        {
          int loadIndex = -1;
          std::string toFind = "cmin";
          if (tokens[i].substr(0, toFind.length()) == toFind)
            loadIndex = getSuffix(tokens[i], toFind);
          if (loadIndex < 0)
            env->ThrowError("Expr: Error in built-in constant expression '%s'", tokens[i].c_str());
          if (loadIndex >= numInputs)
            env->ThrowError("Expr: Too few input clips supplied for reference '%s'", tokens[i].c_str());

          int bitsPerComponent = getEffectiveBitsPerComponent(vi[loadIndex]->BitsPerComponent(), autoconv_conv_float, autoconv_conv_int, autoScaleSourceBitDepth);

          float q = bitsPerComponent == 32 ? uv8tof(16) : (16 << (bitsPerComponent - 8)); // scale chroma min 16
          if (shift_float && bitsPerComponent) q += 0.5f;
          LOAD_OP(opLoadConst, q, 0);
        }
        else if (tokens[i].substr(0, 4) == "cmax") // avs+
        {
          int loadIndex = -1;
          std::string toFind = "cmax";
          if (tokens[i].substr(0, toFind.length()) == toFind)
            loadIndex = getSuffix(tokens[i], toFind);
          if (loadIndex < 0)
            env->ThrowError("Expr: Error in built-in constant expression '%s'", tokens[i].c_str());
          if (loadIndex >= numInputs)
            env->ThrowError("Expr: Too few input clips supplied for reference '%s'", tokens[i].c_str());

          int bitsPerComponent = getEffectiveBitsPerComponent(vi[loadIndex]->BitsPerComponent(), autoconv_conv_float, autoconv_conv_int, autoScaleSourceBitDepth);
          float q = bitsPerComponent == 32 ? uv8tof(240) : (240 << (bitsPerComponent - 8)); // scale chroma max 240
          if (shift_float && bitsPerComponent == 32) q += 0.5f;
          LOAD_OP(opLoadConst, q, 0);
        }
        else if (tokens[i].substr(0, 10) == "range_size") // avs+
        {
          int loadIndex = -1;
          std::string toFind = "range_size";
          if (tokens[i].substr(0, toFind.length()) == toFind)
            loadIndex = getSuffix(tokens[i], toFind);
          if (loadIndex < 0)
            env->ThrowError("Expr: Error in built-in constant expression '%s'", tokens[i].c_str());
          if (loadIndex >= numInputs)
            env->ThrowError("Expr: Too few input clips supplied for reference '%s'", tokens[i].c_str());

          int bitsPerComponent = getEffectiveBitsPerComponent(vi[loadIndex]->BitsPerComponent(), autoconv_conv_float, autoconv_conv_int, autoScaleSourceBitDepth);
          float q = bitsPerComponent == 32 ? 1.0f : (1 << bitsPerComponent); // 1.0, 256, 1024,... 65536
          LOAD_OP(opLoadConst, q, 0);
        }
        else if (tokens[i].substr(0, 9) == "range_min") // avs+ > r2636
        {
          int loadIndex = -1;
          std::string toFind = "range_min";
          if (tokens[i].substr(0, toFind.length()) == toFind)
            loadIndex = getSuffix(tokens[i], toFind);
          if (loadIndex < 0)
            env->ThrowError("Expr: Error in built-in constant expression '%s'", tokens[i].c_str());
          if (loadIndex >= numInputs)
            env->ThrowError("Expr: Too few input clips supplied for reference '%s'", tokens[i].c_str());

          int bitsPerComponent = getEffectiveBitsPerComponent(vi[loadIndex]->BitsPerComponent(), autoconv_conv_float, autoconv_conv_int, autoScaleSourceBitDepth);
          // 0.0 (or -0.5 for 32bit float chroma)
          float q = bitsPerComponent == 32 ? (chroma ? -0.5f : 0.0f) : 0;
          if (chroma && shift_float && bitsPerComponent == 32) q += 0.5f;
          LOAD_OP(opLoadConst, q, 0);
        }
        else if (tokens[i].substr(0, 10) == "yrange_min")
        {
          int loadIndex = -1;
          std::string toFind = "yrange_min";
          if (tokens[i].substr(0, toFind.length()) == toFind)
            loadIndex = getSuffix(tokens[i], toFind);
          if (loadIndex < 0)
            env->ThrowError("Expr: Error in built-in constant expression '%s'", tokens[i].c_str());
          if (loadIndex >= numInputs)
            env->ThrowError("Expr: Too few input clips supplied for reference '%s'", tokens[i].c_str());

          int bitsPerComponent = getEffectiveBitsPerComponent(vi[loadIndex]->BitsPerComponent(), autoconv_conv_float, autoconv_conv_int, autoScaleSourceBitDepth);
          float q = bitsPerComponent == 32 ? 0.0f : 0;
          LOAD_OP(opLoadConst, q, 0);
        }
        else if (tokens[i].substr(0, 9) == "range_max") // avs+
        {
          int loadIndex = -1;
          std::string toFind = "range_max";
          if (tokens[i].substr(0, toFind.length()) == toFind)
            loadIndex = getSuffix(tokens[i], toFind);
          if (loadIndex < 0)
            env->ThrowError("Expr: Error in built-in constant expression '%s'", tokens[i].c_str());
          if (loadIndex >= numInputs)
            env->ThrowError("Expr: Too few input clips supplied for reference '%s'", tokens[i].c_str());

          int bitsPerComponent = getEffectiveBitsPerComponent(vi[loadIndex]->BitsPerComponent(), autoconv_conv_float, autoconv_conv_int, autoScaleSourceBitDepth);
          // 1.0 (or 0.5 for 32bit float chroma), 255, 1023,... 65535
          float q = bitsPerComponent == 32 ? (chroma ? + 0.5f : 1.0f) : ((1 << bitsPerComponent) - 1);
          if (chroma && shift_float && bitsPerComponent == 32) q += 0.5f;
          LOAD_OP(opLoadConst, q, 0);
        }
        else if (tokens[i].substr(0, 10) == "yrange_max")
        {
          int loadIndex = -1;
          std::string toFind = "yrange_max";
          if (tokens[i].substr(0, toFind.length()) == toFind)
            loadIndex = getSuffix(tokens[i], toFind);
          if (loadIndex < 0)
            env->ThrowError("Expr: Error in built-in constant expression '%s'", tokens[i].c_str());
          if (loadIndex >= numInputs)
            env->ThrowError("Expr: Too few input clips supplied for reference '%s'", tokens[i].c_str());

          int bitsPerComponent = getEffectiveBitsPerComponent(vi[loadIndex]->BitsPerComponent(), autoconv_conv_float, autoconv_conv_int, autoScaleSourceBitDepth);
          // 1.0, 255, 1023,... 65535
          float q = bitsPerComponent == 32 ? 1.0f : ((1 << bitsPerComponent) - 1);
          LOAD_OP(opLoadConst, q, 0);
        }
        else if (tokens[i].substr(0, 10) == "range_half") // avs+
        {
          int loadIndex = -1;
          std::string toFind = "range_half";
          if (tokens[i].substr(0, toFind.length()) == toFind)
            loadIndex = getSuffix(tokens[i], toFind);
          if (loadIndex < 0)
            env->ThrowError("Expr: Error in built-in constant expression '%s'", tokens[i].c_str());
          if (loadIndex >= numInputs)
            env->ThrowError("Expr: Too few input clips supplied for reference '%s'", tokens[i].c_str());

          // for chroma: range_half is 0.0 for 32bit float (or 0.5 for old float chroma representation)
          int bitsPerComponent = getEffectiveBitsPerComponent(vi[loadIndex]->BitsPerComponent(), autoconv_conv_float, autoconv_conv_int, autoScaleSourceBitDepth);
          float q = bitsPerComponent == 32 ? (chroma ? uv8tof(128) : 0.5f) : (1 << (bitsPerComponent - 1)); // 0.5f, 128, 512, ... 32768
          if (chroma && shift_float && bitsPerComponent == 32) q += 0.5f;
          LOAD_OP(opLoadConst, q, 0);
        }
        else if (tokens[i].substr(0, 11) == "yrange_half") // avs+
        {
          int loadIndex = -1;
          std::string toFind = "yrange_half";
          if (tokens[i].substr(0, toFind.length()) == toFind)
            loadIndex = getSuffix(tokens[i], toFind);
          if (loadIndex < 0)
            env->ThrowError("Expr: Error in built-in constant expression '%s'", tokens[i].c_str());
          if (loadIndex >= numInputs)
            env->ThrowError("Expr: Too few input clips supplied for reference '%s'", tokens[i].c_str());

          int bitsPerComponent = getEffectiveBitsPerComponent(vi[loadIndex]->BitsPerComponent(), autoconv_conv_float, autoconv_conv_int, autoScaleSourceBitDepth);
          float q = bitsPerComponent == 32 ? 0.5f : (1 << (bitsPerComponent - 1)); // 0.5f, 128, 512, ... 32768
          LOAD_OP(opLoadConst, q, 0);
        }
        // "scaleb" and "scalef" functions scale their operand from 8 bit to the bit depth of the first clip.
        // "i8", "i10", "i14", "i16" and "f32" (typically at the beginning of the expression) sets the scale-base to 8..16 bits or float, respectively.
        // "i8".."f32" keywords can appear anywhere in the expression, but only the last occurence will be effective for the whole expression.
        else if (tokens[i] == "scaleb" || tokens[i] == "yscaleb" || tokens[i] == "scalef" || tokens[i] == "yscalef") // avs+, scale by bit shift
        {
          // Note: if 'scale_float' is used then all float input is automatically converted to integer
          // in this case the targetBitDepth is not 32 for float clips but the actual autoscaleSourceBitDepth
          int effectivetargetBitDepth = getEffectiveBitsPerComponent(targetBitDepth, autoconv_conv_float, autoconv_conv_int, autoScaleSourceBitDepth);
          // number to scale is not chroma-related one even if we are in chroma (U,V) plane
          const bool forceNonUV = (tokens[i] == "yscaleb") || (tokens[i] == "yscalef");

          const int srcBitDepth = autoScaleSourceBitDepth;
          const int dstBitDepth = effectivetargetBitDepth;
          const bool use_chroma = chroma && !forceNonUV;

          const bool isfull = tokens[i] == "scalef" || tokens[i] == "yscalef";

          bits_conv_constants d;
          get_bits_conv_constants(d, use_chroma, isfull, isfull, srcBitDepth, dstBitDepth);
          // chroma is spec: signed. Limited:16-240 is really 128 +/-112. Full:1-255 is really 128+/-127

          // floatUV
          if (use_chroma && shift_float && dstBitDepth == 32 && srcBitDepth < 32) {
            d.dst_offset = 0.5f; // get out from -0.5..0.5 to 0..1
          }

          if (d.src_offset != 0) {
            LOAD_OP(opLoadConst, (float)d.src_offset, 0);
            TWO_ARG_OP(opSub);
          }
          if (d.mul_factor != 1.0f) {
            LOAD_OP(opLoadConst, d.mul_factor, 0);
            TWO_ARG_OP(opMul);
          }
          if (d.dst_offset != 0) {
            LOAD_OP(opLoadConst, (float)d.dst_offset, 0);
            TWO_ARG_OP(opAdd);
          }
        }
        else if (tokens[i] == "i8") // avs+
        {
          autoScaleSourceBitDepth = 8;
        }
        else if (tokens[i] == "i10") // avs+
        {
          autoScaleSourceBitDepth = 10;
        }
        else if (tokens[i] == "i12") // avs+
        {
          autoScaleSourceBitDepth = 12;
        }
        else if (tokens[i] == "i14") // avs+
        {
          autoScaleSourceBitDepth = 14;
        }
        else if (tokens[i] == "i16") // avs+
        {
          autoScaleSourceBitDepth = 16;
        }
        else if (tokens[i] == "f32") // avs+
        {
          autoScaleSourceBitDepth = 32;
        }
        else if (tokens[i].length() > 1 && tokens[i][0] >= 'a' && tokens[i][0] <= 'z' && tokens[i][1] == '.') {
          // frame property access: x.framePropName syntax
          char srcChar = tokens[i][0];
          int srcIndex;
          if (srcChar >= 'x')
            srcIndex = srcChar - 'x';
          else
            srcIndex = srcChar - 'a' + 3;
          if (srcIndex >= numInputs)
            env->ThrowError("Expr: Too few input clips supplied to reference '%s'", tokens[i].c_str());

          auto fullname = tokens[i];
          auto fpname = tokens[i].substr(2); // after '.'
          if(fpname.length() == 0)
            env->ThrowError("Expr: no frame property name is specified");
          if (!isValidVarName(fpname))
            env->ThrowError("Expr: invalid frame property name '%s'", fpname.c_str());

          // check if frame property already existed in a variable slots
          auto key = fullname;
          auto it = fpnames.find(key);
          int loadIndex;
          if (it == fpnames.end()) {
            // first occurance, insert name and actual index
            if (fpindex >= MAX_FRAMEPROP_VARIABLES)
              env->ThrowError("Expr: too many frame property references, maximum reached (%d)", MAX_FRAMEPROP_VARIABLES);
            loadIndex = fpindex++;
            fpnames[key] = loadIndex;
            // Register into the list of frame properties to read
            // input clip index: srcIndex
            // name of frame property: fpname
            // variable index to fill: loadindex
            ExprFramePropData fpData;
            fpData.name = fpname;
            fpData.srcIndex = srcIndex;
            fpData.var_index = loadIndex;
            fp.push_back(fpData);
          }
          else {
            loadIndex = it->second;
          }
          LOAD_OP(opLoadFramePropVar, loadIndex, 0);
        }
        else if (tokenlen >= 2 && (tokens[i][tokenlen - 1] == '^' || tokens[i][tokenlen - 1] == '@'))
        {
          // storing a variable: A@ .. Z@
          // storing a variable and remove from stack: A^..Z^
          auto key = tokens[i].substr(0, tokenlen - 1);
          auto opchar = tokens[i][tokenlen - 1];
          auto it = varnames.find(key);
          int loadIndex;
          if (it == varnames.end()) {
            if(!isValidVarName(key))
              env->ThrowError("Expr: invalid variable name '%s'", key.c_str());
            // first occurance, insert name and actual index
            if (varindex >= MAX_USER_VARIABLES)
              env->ThrowError("Expr: too many variables, maximum reached (%d)", MAX_USER_VARIABLES);
            loadIndex = varindex++;
            varnames[key] = loadIndex;
          }
          else {
            loadIndex = it->second;
          }
          if (opchar == '^')
            VAR_STORE_SPEC_OP(opStoreVarAndDrop1, loadIndex);
          else // if (opchar == '@')
            VAR_STORE_OP(opStoreVar, loadIndex);
        }
        else if (isValidVarName(tokens[i]))
        {
          // variable names
          // the very end, all reserved expr words were processed
          auto key = tokens[i];
          auto it = varnames.find(key);
          if (it == varnames.end())
            env->ThrowError("Expr: keyword or variable not found: '%s'", key.c_str());
          auto loadIndex = it->second;
          LOAD_OP(opLoadVar, loadIndex, 0);
        }
        else {
          // parse a number
            float f;
            std::string s;
            std::istringstream numStream(tokens[i]);
            numStream.imbue(std::locale::classic());
            if (!(numStream >> f))
              env->ThrowError("Expr: Failed to convert '%s' to float", tokens[i].c_str());
            if (numStream >> s)
              env->ThrowError("Expr: Failed to convert '%s' to float, not the whole token could be converted", tokens[i].c_str());
            LOAD_OP(opLoadConst, f, 0);
        }
    }

    if (tokens.size() > 0) {
        if (stackSize != 1)
            env->ThrowError("Expr: Stack unbalanced at end of expression (size=%zu). Need to have exactly one value on the stack to return.", stackSize);

        // When scale_inputs option was used for scaling input to a common internal range,
        // we have to scale pixels before storing them back
        // need any conversion?
        // or use effectiveTargetBitDepth instead of autoScaleSourceBitDepth
        const int srcBitDepth = autoScaleSourceBitDepth;
        const int dstBitDepth = targetBitDepth;
        const bool use_chroma = chroma; // && !forceNonUV;

        const bool isfull = autoconv_full_scale;

        if (autoconv_conv_int || autoconv_conv_float) {

          if ((targetBitDepth != 32 && autoconv_conv_int) || 
            ((targetBitDepth == 32 && autoconv_conv_float))) {

            bits_conv_constants d;
            get_bits_conv_constants(d, use_chroma, isfull, isfull, srcBitDepth, dstBitDepth);
            // chroma is spec: signed. Limited:16-240 is really 128 +/-112. Full:1-255 is really 128+/-127

            if (d.src_offset != 0) {
              LOAD_OP_NOTOKEN(opLoadConst, (float)d.src_offset, 0);
              TWO_ARG_OP_NOTOKEN(opSub);
            }
            if (d.mul_factor != 1.0f) {
              LOAD_OP_NOTOKEN(opLoadConst, d.mul_factor, 0);
              TWO_ARG_OP_NOTOKEN(opMul);
            }
            if (d.dst_offset != 0) {
              LOAD_OP_NOTOKEN(opLoadConst, (float)d.dst_offset, 0);
              TWO_ARG_OP_NOTOKEN(opAdd);
            }
          }
        }
        // "floatUV" -
        // this reverses the effect of the 0.5 float-type pre-shift is the pixel load phase
        // pre-shifts chroma pixels by +0.5 before applying the expression (see at pixel load),
        // then here the result is shifted back by -0.5
        // Thus expressions, which rely on a working range of 0..1.0 will work transparently
        if ((!autoconv_conv_float || srcBitDepth == 32) && targetBitDepth == 32) {
          if (chroma && shift_float) {
            LOAD_OP_NOTOKEN(opLoadConst, 0.5f, 0);
            TWO_ARG_OP_NOTOKEN(opSub); // at pixel load it was opAdd
          }
        }

        if (clamp_float_i > 0 && targetBitDepth == 32) {
          if (chroma) {
            // clamp_float clamp_float_uv -> clamp_float_i   clamp range for Y clamp range for UV
            // false       x                 0               0..1              -0.5..+0.5
            // true        false             1               0..1              -0.5..+0.5
            // true        true              2               0..1              0..1
            LOAD_OP_NOTOKEN(opLoadConst, clamp_float_i == 2 ? 0.0f : -0.5f, 0);
            TWO_ARG_OP_NOTOKEN(opMax);
            LOAD_OP_NOTOKEN(opLoadConst, clamp_float_i == 2 ? 1.0f : 0.5f, 0);
            TWO_ARG_OP_NOTOKEN(opMin);
          }
          else
          { // luma
            LOAD_OP_NOTOKEN(opLoadConst, 0.0f, 0);
            TWO_ARG_OP_NOTOKEN(opMax);
            LOAD_OP_NOTOKEN(opLoadConst, 1.0f, 0);
            TWO_ARG_OP_NOTOKEN(opMin);
          }
        }

        // and finally store it
        ops.push_back(storeOp);
    }

    return maxStackSize;
}

static float calculateOneOperand(uint32_t op, float a) {
    switch (op) {
        case opSqrt:
            return std::sqrt(a);
        case opAbs:
            return std::abs(a);
        case opSgn:
            return (a < 0) ? -1.0f : (a > 0) ? 1.0f : 0.0f;
        case opNeg: // Expr "boolean": not
            return (a > 0) ? 0.0f : 1.0f;
        case opNegSign:
            return -a;
        case opExp:
            return std::exp(a);
        case opLog:
            return std::log(a);
        case opSin:
          return std::sin(a);
        case opCos:
          return std::cos(a);
        case opTan:
          return std::tan(a);
        case opAsin:
          return std::asin(a);
        case opAcos:
          return std::acos(a);
        case opAtan:
          return std::atan(a);
        case opRound:
          return std::round(a);
        case opFloor:
          return std::floor(a);
        case opCeil:
          return std::ceil(a);
        case opTrunc:
          return std::trunc(a);
    }

    return 0.0f;
}

static float calculateTwoOperands(uint32_t op, float a, float b) {
    switch (op) {
        case opAdd:
            return a + b;
        case opSub:
            return a - b;
        case opMul:
            return a * b;
        case opDiv:
            return a / b;
        case opFmod:
            return std::fmod(a, b);
        case opMax:
            return std::max(a, b);
        case opMin:
            return std::min(a, b);
        case opGt:
            return (a > b) ? 1.0f : 0.0f;
        case opLt:
            return (a < b) ? 1.0f : 0.0f;
        case opEq:
            return (a == b) ? 1.0f : 0.0f;
        case opNotEq:
            return (a != b) ? 1.0f : 0.0f;
        case opLE:
            return (a <= b) ? 1.0f : 0.0f;
        case opGE:
            return (a >= b) ? 1.0f : 0.0f;
        case opAnd:
            return (a > 0 && b > 0) ? 1.0f : 0.0f;
        case opOr:
            return (a > 0 || b > 0) ? 1.0f : 0.0f;
        case opXor:
            return ((a > 0) != (b > 0)) ? 1.0f : 0.0f;
        case opPow:
            return std::pow(a, b);
        case opAtan2:
            return std::atan2(a, b);
    }

    return 0.0f;
}

static int numOperands(uint32_t op) {
    switch (op) {
        case opLoadConst:
        case opLoadSrc8:
        case opLoadSrc16:
        case opLoadSrcF32:
        case opLoadSrcF16:
        case opLoadRelSrc8:
        case opLoadRelSrc16:
        case opLoadRelSrcF32:
        case opDup:
        case opLoadSpatialX:
        case opLoadSpatialY:
        case opLoadVar:
        case opLoadFramePropVar:
        case opLoadInternalVar:
        case opSwap:
        case opStoreVar:
        case opStoreVarAndDrop1:
          return 0;

        case opSqrt:
        case opAbs:
        case opSgn:
        case opNeg:
        case opNegSign:
        case opExp:
        case opLog:
        case opSin:
        case opCos:
        case opTan:
        case opAsin:
        case opAcos:
        case opAtan:
        case opRound:
        case opFloor:
        case opCeil:
        case opTrunc:
          return 1;

        case opAdd:
        case opSub:
        case opMul:
        case opDiv:
        case opFmod:
        case opMax:
        case opMin:
        case opGt:
        case opLt:
        case opEq:
        case opNotEq:
        case opLE:
        case opGE:
        case opAnd:
        case opOr:
        case opXor:
        case opPow:
        case opAtan2:
          return 2;

        case opTernary:
        case opClip:
            return 3;
    }

    return 0;
}

static bool isLoadOp(uint32_t op) {
    switch (op) {
        case opLoadConst:
        case opLoadSrc8:
        case opLoadSrc16:
        case opLoadSrcF32:
        case opLoadSrcF16:
        case opLoadRelSrc8:
        case opLoadRelSrc16:
        case opLoadRelSrcF32:
        case opLoadSpatialX:
        case opLoadSpatialY:
        case opLoadVar:
        case opLoadFramePropVar:
        case opLoadInternalVar:
          return true;
    }

    return false;
}

// true when ops[from..to) has a variable store, dup or swap
// Used from constant folding (ternary part x?a:b )
static bool hasStackOrStoreOps(const std::vector<ExprOp>& ops, size_t from, size_t to) {
  for (size_t k = from; k < to; k++) {
    const uint32_t op = ops[k].op;
    if (op == opStoreVar || op == opStoreVarAndDrop1 || op == opDup || op == opSwap)
      return true;
  }
  return false;
}

static void findBranches(std::vector<ExprOp>& ops, size_t pos, size_t* start1, size_t* start2, size_t* start3) {
  int operands = numOperands(ops[pos].op);

  size_t temp1, temp2, temp3;

  if (operands == 0) {
    // dup loadsrc loadrel loadconst swap, storevar
    if (ops[pos].op == opSwap || ops[pos].op == opStoreVar || ops[pos].op == opStoreVarAndDrop1)
    {
      findBranches(ops, pos - 1, &temp1, &temp2, &temp3);
      *start1 = temp1;
      if (ops[pos].op == opStoreVarAndDrop1) {
        // StoreAndPopVar: opStoreVar + Ignore topmost stack.
        // Branch was calculated only for storing its result into var.
        // Leaves no trace on stack.
        // So we go on with next branch
        pos = *start1;
        findBranches(ops, pos - 1, &temp1, &temp2, &temp3);
        *start1 = temp1;
      }
    }
    else
      *start1 = pos;
  }
  else if (operands == 1) {
    if (isLoadOp(ops[pos - 1].op)) {
      *start1 = pos - 1;
    }
    else {
      findBranches(ops, pos - 1, &temp1, &temp2, &temp3);
      *start1 = temp1;
    }
  }
  else if (operands == 2) {
    if (isLoadOp(ops[pos - 1].op)) {
      *start2 = pos - 1;
    }
    else {
      findBranches(ops, pos - 1, &temp1, &temp2, &temp3);
      *start2 = temp1;
    }

    if (isLoadOp(ops[*start2 - 1].op)) {
      *start1 = *start2 - 1;
    }
    else {
      findBranches(ops, *start2 - 1, &temp1, &temp2, &temp3);
      *start1 = temp1;
    }
  }
  else if (operands == 3) {
    if (isLoadOp(ops[pos - 1].op)) {
      *start3 = pos - 1;
    }
    else {
      findBranches(ops, pos - 1, &temp1, &temp2, &temp3);
      *start3 = temp1;
    }

    if (isLoadOp(ops[*start3 - 1].op)) {
      *start2 = *start3 - 1;
    }
    else {
      findBranches(ops, *start3 - 1, &temp1, &temp2, &temp3);
      *start2 = temp1;
    }

    if (isLoadOp(ops[*start2 - 1].op)) {
      *start1 = *start2 - 1;
    }
    else {
      findBranches(ops, *start2 - 1, &temp1, &temp2, &temp3);
      *start1 = temp1;
    }
  }
}


static void foldConstants(std::vector<ExprOp> &ops, IScriptEnvironment* env) {
    for (size_t i = 0; i < ops.size(); i++) {
        switch (ops[i].op) {
        case opLoadRelSrc8:
        case opLoadRelSrc16:
        case opLoadRelSrcF32:
          // x[dx,dy]:k (constant border mode) takes the filler from the stack top, which
          // must be constant by now.
          // Store it in the op as is (expression's domain, the load op does the scale_inputs
          // conversion of the pixels before putting the filler in), remove the constant:
          // from here on the op is an ordinary load.
          if (ops[i].bc == borderConstant) {
            if (i == 0 || ops[i - 1].op != opLoadConst)
              env->ThrowError("Expr: the filler value before x[dx,dy]:k must be a constant");
            ops[i].filler = ops[i - 1].e.fval;
            ops.erase(ops.begin() + i - 1);
            i--;
          }
          break;
          // optimize pow
        case opPow:
          if (ops[i - 1].op == opLoadConst) {
            if (ops[i - 1].e.fval == 0.5f) {
              // replace pow 0.5 with sqrt
              ops[i].op = opSqrt;
              ops.erase(ops.begin() + i - 1);
              i--;
            }
            else if (ops[i - 1].e.fval == 1.0f) {
              // replace pow 1 with nothing
              ops.erase(ops.begin() + i - 1, ops.begin() + i + 1);
              i -= 2;
            }
            else if (ops[i - 1].e.fval == 2.0f) {
              // replace pow 2 with dup *
              ops[i].op = opMul;
              ops[i - 1].op = opDup; ops[i - 1].e.ival = 0; // dup 0
              i--;
            }
            else if (ops[i - 1].e.fval == 3.0f) {
              // replace pow 3 with dup dup * *
              ops[i].op = opMul;
              ops[i - 1].op = opMul;
              ExprOp extraDup(opDup, 0);
              ops.insert(ops.begin() + i - 1, extraDup);
              ops.insert(ops.begin() + i - 1, extraDup);
              i--;
            }
            else if (ops[i - 1].e.fval == 4.0f) {
              // replace pow 4 with dup * dup *
              ops[i].op = opMul;
              ops[i - 1].op = opDup; ops[i - 1].e.ival = 0; // dup 0
              ExprOp extraMul(opMul);
              ExprOp extraDup(opDup, 0);
              ops.insert(ops.begin() + i - 1, extraMul);
              ops.insert(ops.begin() + i - 1, extraDup);
              i--;
            }
          }
          break;
          // optimize Mul 1 Div 1, Mul -1, Div -1
        case opMul: case opDiv:
          if (ops[i - 1].op == opLoadConst) {
            if (ops[i - 1].e.fval == 1.0f) {
              // replace mul 1 or div 1 with nothing
              ops.erase(ops.begin() + i - 1, ops.begin() + i + 1);
              i -= 2;
            }
            else if (ops[i - 1].e.fval == -1.0f) {
              // replace mul -1 or div -1 with neg
              ops[i].op = opNegSign;
              ops.erase(ops.begin() + i - 1);
              i--;
            }
          }
          break;
          // optimize Add 0 or Sub 0
        case opAdd: case opSub:
          if (ops[i - 1].op == opLoadConst) {
            if (ops[i - 1].e.fval == 0.0f) {
              // replace add 0 or sub 0 with nothing
              ops.erase(ops.begin() + i - 1, ops.begin() + i + 1);
              i -= 2;
            }
          }
          break;

        }

        // fold constant
        switch (ops[i].op) {
            case opDup:
              if (ops[i - 1].op == opLoadConst && ops[i].e.ival == 0) {
                ops[i] = ops[i - 1];
              }
              break;

            case opSqrt:
            case opAbs:
            case opSgn:
            case opNeg:
            case opNegSign:
            case opExp:
            case opLog:
            case opSin:
            case opCos:
            case opTan:
            case opAsin:
            case opAcos:
            case opAtan:
            case opRound:
            case opFloor:
            case opCeil:
            case opTrunc:
              if (ops[i - 1].op == opLoadConst) {
                ops[i].e.fval = calculateOneOperand(ops[i].op, ops[i - 1].e.fval);
                ops[i].op = opLoadConst;
                ops.erase(ops.begin() + i - 1);
                i--;
              }
              break;

            case opSwap:
              if (ops[i - 2].op == opLoadConst && ops[i - 1].op == opLoadConst && ops[i].e.ival == 1) {
                const float temp = ops[i - 2].e.fval;
                ops[i - 2].e.fval = ops[i - 1].e.fval;
                ops[i - 1].e.fval = temp;
                ops.erase(ops.begin() + i);
                i--;
              }
              break;

            case opAdd:
            case opSub:
            case opMul:
            case opDiv:
            case opFmod:
            case opMax:
            case opMin:
            case opGt:
            case opLt:
            case opEq:
            case opNotEq:
            case opLE:
            case opGE:
            case opAnd:
            case opOr:
            case opXor:
            case opPow:
            case opAtan2:
              if (ops[i - 2].op == opLoadConst && ops[i - 1].op == opLoadConst) {
                ops[i].e.fval = calculateTwoOperands(ops[i].op, ops[i - 2].e.fval, ops[i - 1].e.fval);
                ops[i].op = opLoadConst;
                ops.erase(ops.begin() + i - 2, ops.begin() + i);
                i -= 2;
              }
              break;

            case opTernary:
              size_t start1, start2, start3;
              findBranches(ops, i, &start1, &start2, &start3);
              // start1/2/3: condition/true/false branch

              // Folded only when the condition is a single constant and the operand range
              // (condition..ternary) has none of these:
              // - variable store (A@: store the stack top into A, keep it; A^: store and pop):
              //   Expr evaluates both branches, so a store in the erased branch must still
              //   happen. A^ also removes the value it stores, so e.g. "sx A^" leaves nothing
              //   on the stack: it is not an operand, belongs to neither branch, but
              //   findBranches attaches it to a neighbor (in "0 7 sx A^ A ?" to the "7").
              // - dup, swap: they read stack values below their own branch, findBranches
              //   cannot split the range correctly.
              if (ops[start2 - 1].op == opLoadConst && !hasStackOrStoreOps(ops, start1, i)) {
                ops.erase(ops.begin() + i); // erase ternary op
                if (ops[start1].e.fval > 0.0f) { // condition is constant 'true'
                  // start1 is start2 - 1
                  ops.erase(ops.begin() + start3, ops.begin() + i); // erase 'false' branch
                  i = start3;
                } else {
                  ops.erase(ops.begin() + start2, ops.begin() + start3);  // erase 'true' branch
                  i -= start3 - start2;
                }
                ops.erase(ops.begin() + start1); // erase constant
                i -= 2;
              }
              break;
      }
    }
}

Exprfilter::Exprfilter(const std::vector<PClip>& _child_array, const std::vector<std::string>& _expr_array, const char *_newformat, const bool _optAvx2,
  const bool _optSingleMode, const bool _optSSE2, const bool _optVectorC, const std::string _scale_inputs, const int _clamp_float_i, const int _lutmode, IScriptEnvironment *env) :
  children(_child_array), expressions(_expr_array), optAvx2(_optAvx2), optSingleMode(_optSingleMode), optSSE2(_optSSE2),
  optVectorC(_optVectorC), scale_inputs(_scale_inputs), clamp_float_i(_clamp_float_i), lutmode(_lutmode) {

  vi = children[0]->GetVideoInfo();
  d.vi = vi;
  if (lutmode < 0 || lutmode>2)
    env->ThrowError("'Expr: 'lut' can be 0 (no lut), 1 (lut_x) or 2 (lut_xy)");
  if (lutmode == 1 && vi.BitsPerComponent() == 32)
    lutmode = 0; // fallback to realtime
  if (lutmode == 2 && vi.BitsPerComponent() > 14)
    lutmode = 0; // fallback to realtime
  d.lutmode = lutmode;

  for (int i = 0; i < 4; i++)
    d.luts[i] = nullptr;

  // parse "scale_inputs"
  autoconv_full_scale = false;
  autoconv_conv_float = false;
  autoconv_conv_int = false;
  shift_float = false;

  if (scale_inputs == "allf") {
    autoconv_full_scale = true;
    autoconv_conv_int = true;
    autoconv_conv_float = true;
  }
  else if (scale_inputs == "intf") {
    autoconv_full_scale = true;
    autoconv_conv_int = true;
  }
  else if (scale_inputs == "floatf") {
    autoconv_full_scale = true;
    autoconv_conv_float = true;
  }
  else if (scale_inputs == "all") {
    autoconv_conv_int = true;
    autoconv_conv_float = true;
  }
  else if (scale_inputs == "int") {
    autoconv_conv_int = true;
  }
  else if (scale_inputs == "float") {
    autoconv_conv_float = true;
  }
  else if (scale_inputs == "floatuv") {
    autoconv_conv_float = false; // !! really
    // like in masktools2 2.2.20+
    shift_float = true; // !!
  }
  else if (scale_inputs != "none") {
    env->ThrowError("Expr: scale_inputs must be 'all','allf','int','intf','float','floatf','floatUV' or 'none'");
  }

  try {
    d.numInputs = (int)children.size(); // d->numInputs = vsapi->propNumElements(in, "clips");
    if (d.numInputs > 26)
      env->ThrowError("Expr: More than 26 input clips provided");

    for (int i = 0; i < d.numInputs; i++)
      d.clips[i] = children[i];

    if (d.lutmode > 0) {
      if(d.numInputs != d.lutmode)
        env->ThrowError("Expr lut: number of input clips must be the same as LUT's dimension. LUT is %dD. Passed clip(s): %d", d.lutmode, d.numInputs);
    }

    // checking formats
    const VideoInfo* vi_array[MAX_EXPR_INPUTS] = {};
    for (int i = 0; i < d.numInputs; i++)
      if (d.clips[i])
        vi_array[i] = &d.clips[i]->GetVideoInfo();


    int planes_y[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
    int planes_r[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A }; // for checking GBR order is OK
    int *plane_enums = (d.vi.IsYUV() || d.vi.IsYUVA()) ? planes_y : planes_r;
    const int plane_enum = plane_enums[1]; // for subsampling check U only, unused when IsY()/IsYA(), see below

    // check all clips against first one
    for (int i = 0; i < d.numInputs; i++) {

      int *plane_enums_i = (vi_array[i]->IsYUV() || vi_array[i]->IsYUVA()) ? planes_y : planes_r;
      const int plane_enum_i = plane_enums_i[1];

      if (vi_array[0]->NumComponents() != vi_array[i]->NumComponents() // number of planes should match
        ||
        ( !vi_array[0]->IsY() && !vi_array[0]->IsYA() && ( // no subsampling for Y or Y+Alpha (no chroma)
           vi_array[0]->GetPlaneWidthSubsampling(plane_enum) != vi_array[i]->GetPlaneWidthSubsampling(plane_enum_i)
           || vi_array[0]->GetPlaneHeightSubsampling(plane_enum) != vi_array[i]->GetPlaneHeightSubsampling(plane_enum_i)
          )
        )
        || vi_array[0]->width != vi_array[i]->width
        || vi_array[0]->height != vi_array[i]->height)
        env->ThrowError("Expr: All inputs must have the same number of planes and the same dimensions, subsampling included");

      if (vi_array[i]->IsRGB() && !vi_array[i]->IsPlanar())
        env->ThrowError("Expr: No packed RGB format allowed for clip #%d, use planar RGB instead",i+1);
      if (vi_array[i]->IsYUY2())
        env->ThrowError("Expr: YUY2 format not allowed for clip #%d", i+1);
    }

    // format override
    if (_newformat != nullptr) {
      int pixel_type = GetPixelTypeFromName(_newformat);
      if (pixel_type == VideoInfo::CS_UNKNOWN)
        env->ThrowError("Expr: Invalid video format string parameter");
      d.vi.pixel_type = pixel_type;
      if(d.vi.IsRGB() && !d.vi.IsPlanar())
        env->ThrowError("Expr: No packed RGB format allowed");
      if (d.vi.IsYUY2())
        env->ThrowError("Expr: YUY2 format not allowed");

      const bool isSinglePlaneInput = vi_array[0]->IsY();

      // input number of planes >= output planes
      if (!isSinglePlaneInput && vi_array[0]->NumComponents() < d.vi.NumComponents())
        env->ThrowError("Expr: number of planes in input should be greater than or equal than of output");

      // subsampling should match
      int planes_ya[2] = { PLANAR_Y, PLANAR_A };
      int *plane_enums_s = vi_array[0]->IsYA() ? planes_ya : (vi_array[0]->IsYUV() || vi_array[0]->IsYUVA()) ? planes_y : planes_r;
      int *plane_enums_d = d.vi.IsYA() ? planes_ya : (d.vi.IsYUV() || d.vi.IsYUVA()) ? planes_y : planes_r;
      for (int p = 0; p < d.vi.NumComponents(); p++) {
        const int plane_enum_s = isSinglePlaneInput ? plane_enums_s[0] : plane_enums_s[p]; // for Y inputs, reference is Y for each output plane
        const int plane_enum_d = plane_enums_d[p];
        if (vi_array[0]->GetPlaneWidthSubsampling(plane_enum_s) != d.vi.GetPlaneWidthSubsampling(plane_enum_d)
          || vi_array[0]->GetPlaneHeightSubsampling(plane_enum_s) != d.vi.GetPlaneHeightSubsampling(plane_enum_d)) {
          if(isSinglePlaneInput)
            env->ThrowError("Expr: output must not be a subsampled format for Y-only input(s)");
          else
            env->ThrowError("Expr: inputs and output must have the same subsampling");
        }
      }

      vi = d.vi;
    }

    // check expression count, duplicate omitted expressions from previous one
    int nexpr = (int)expressions.size();
    if (nexpr > d.vi.NumComponents()) // ->numPlanes)
      env->ThrowError("Expr: More expressions given than there are planes");

    std::string expr[4]; // 4th: alpha
    for (int i = 0; i < nexpr; i++)
      expr[i] = expressions[i];
    if (nexpr == 1) {
      expr[1] = expr[0];
      expr[2] = expr[0]; // e.g. exprU = exprV = exprY
    }
    else if (nexpr == 2) {
      expr[2] = expr[1]; // e.g. exprV = exprU
    }
    if(nexpr <= 3)
      expr[3] = ""; // do not use previous expression to alpha expr. Default: "" (copy)

    // default: all clips unused
    for (int i = 0; i < MAX_EXPR_INPUTS; i++) {
      d.clipsUsed[i] = false;
    }

    for (int i = 0; i < d.vi.NumComponents(); i++) {
      if (!expr[i].empty()) {
        d.plane[i] = poProcess;
      }
      else {
        if (d.vi.BitsPerComponent() == vi_array[0]->BitsPerComponent()) {
          d.plane[i] = poCopy; // copy only when target clip format bit depth == 1st clip's bit depth
          d.planeCopySourceClip[i] = 0; // default source clip from empty expression: first one
          d.clipsUsed[0] = true; // mark clip to have its GetFrame
        }
        else
          d.plane[i] = poUndefined;
      }
    }

    int planes_ya[2] = { PLANAR_Y, PLANAR_A }; // no chroma: index 1 is alpha, not U
    int* plane_enums_d = d.vi.IsYA() ? planes_ya : (d.vi.IsYUV() || d.vi.IsYUVA()) ? planes_y : planes_r;

    d.maxStackSize = 0;
    for (int i = 0; i < d.vi.NumComponents(); i++) {
      // plane_enums_d is YA-aware array
      const int plane_enum = plane_enums_d[i];
      const int planewidth = d.vi.width >> d.vi.GetPlaneWidthSubsampling(plane_enum);
      const int planeheight = d.vi.height >> d.vi.GetPlaneHeightSubsampling(plane_enum);
      const bool chroma = (plane_enum == PLANAR_U || plane_enum == PLANAR_V);
      d.maxStackSize = std::max(parseExpression(expr[i], d.ops[i], d.frameprops[i], vi_array, &d.vi, getStoreOp(&d.vi), d.numInputs, planewidth, planeheight, chroma,
        autoconv_full_scale, autoconv_conv_int, autoconv_conv_float, clamp_float_i, shift_float, d.lutmode,
        env), d.maxStackSize);
      foldConstants(d.ops[i], env);

      // optimize constant store, change operation to "fill"
      if (d.plane[i] == poProcess && d.ops[i].size() == 2 && d.ops[i][0].op == opLoadConst) {
        uint32_t op = d.ops[i][1].op;
        if (op == opStore8 || op == opStore10 || op == opStore12 || op == opStore14 || op == opStore16 || op == opStoreF32)
        {
          d.plane[i] = poFill;
          d.planeFillValue[i] = d.ops[i][0].e.fval;
        }
      }

      // optimize single clip letter in expression: Load-Store. Change operation to "copy"
      // no relative loads here
      if (d.plane[i] == poProcess && d.ops[i].size() == 2 &&
        (d.ops[i][0].op == opLoadSrc8 || d.ops[i][0].op == opLoadSrc16 || d.ops[i][0].op == opLoadSrcF16 || d.ops[i][0].op == opLoadSrcF32) &&
        (d.ops[i][1].op == opStore8 || d.ops[i][1].op == opStore10 || d.ops[i][1].op == opStore12 || d.ops[i][1].op == opStore14 || d.ops[i][1].op == opStore16 || d.ops[i][1].op == opStoreF16 || d.ops[i][1].op == opStoreF32))
      {
        const int sourceClip = d.ops[i][0].e.ival;
        // check target vs source bit depth
        if(d.vi.BitsPerComponent() == vi_array[sourceClip]->BitsPerComponent()) // no 16bit float in avs+
        {
          d.plane[i] = poCopy;
          d.planeCopySourceClip[i] = sourceClip;
          d.clipsUsed[sourceClip] = true; // mark clip to have its GetFrame
        }
      }

      // optimize: mark referenced input clips in order to not call GetFrame for unused inputs
      if (lutmode == 0) {
        for (size_t j = 0; j < d.ops[i].size(); j++) {
          const uint32_t op = d.ops[i][j].op;
          if (op == opLoadSrc8 || op == opLoadSrc16 || op == opLoadSrcF16 || op == opLoadSrcF32 ||
            op == opLoadRelSrc8 || op == opLoadRelSrc16 || op == opLoadRelSrcF32)
          {
            const int sourceClip = d.ops[i][j].e.ival;
            d.clipsUsed[sourceClip] = true;
          }
        }
        // input clips with frame property access are used as well
        for (auto framePropToRead : d.frameprops[i]) {
          const int sourceClip = framePropToRead.srcIndex;
          d.clipsUsed[sourceClip] = true;
        }
      }

      if (lutmode > 0) {
        for (int i = 0; i < lutmode; i++) // lut: always get. Needed for the init
          d.clipsUsed[i] = true;
        // * bit depth of input clip(s) and output must match
        bool lut_ok = 
          (d.numInputs == 1
            && d.vi.BitsPerComponent() == vi_array[0]->BitsPerComponent())
          ||
          (
            d.numInputs == 2
            && d.vi.BitsPerComponent() == vi_array[0]->BitsPerComponent()
            && d.vi.BitsPerComponent() == vi_array[1]->BitsPerComponent()
            );
        if(!lut_ok)
          env->ThrowError("Expr: error in lut mode: input bit depths and output bit depth must be the same");
      }


#ifdef INTEL_INTRINSICS
      // Check CPU instuction level constraints:
      // round, floor, ceil, trunc: minimun SSE4.1
      // Trig.func: C only
      d.planeOptAvx2[i] = optAvx2;
      d.planeOptSSE2[i] = optSSE2;
      for (size_t j = 0; j < d.ops[i].size(); j++) {
        const uint32_t op = d.ops[i][j].op;
        // relative load with mirror or wrap border modes:
        // the JIT code implements reflects or wraps only once.
        // Offsets needing more will run in C.
        if ((op == opLoadRelSrc8 || op == opLoadRelSrc16 || op == opLoadRelSrcF32) && (d.ops[i][j].bc == borderMirror || d.ops[i][j].bc == borderReflect101 || d.ops[i][j].bc == borderWrap)) {
          const int dx = d.ops[i][j].dx;
          const int dy = d.ops[i][j].dy;
          const bool dx_ok = planewidth < 4 || // small planes: compile-time constant mapping, any dx
            (dx >= -(planewidth - 1) && dx <= planewidth - (d.ops[i][j].bc == borderReflect101 ? 4 : 3));
          const bool dy_ok = dy > -planeheight && dy < planeheight;
          if (!dx_ok || !dy_ok) {
            d.planeOptAvx2[i] = false;
            d.planeOptSSE2[i] = false;
          }
        }
        // some trig.functions C only, except Sin and Cos and Atan2 and tan
        if (op == opAsin || op == opAcos || op == opAtan) {
          d.planeOptAvx2[i] = false;
          d.planeOptSSE2[i] = false;
          break; // we can break here, we limited to the minimum.
        }
        // round, trunc, ceil: minimum of SSE4.1
        if (op == opRound || op == opFloor || op == opCeil || op == opTrunc ) {
          if (!(env->GetCPUFlags() & CPUF_SSE4_1)) // required minimum (_mm_round_ps...)
            d.planeOptSSE2[i] = false;
          // no break: later ops (relative load, asin/acos/atan) must be checked as well
        }
      }
#endif

    }

#ifdef VS_TARGET_CPU_X86
    // optAvx2 can only disable avx2 when available

    for (int i = 0; i < d.vi.NumComponents(); i++) {
      if (d.plane[i] == poProcess) {

        const int plane_enum = plane_enums_d[i];
        int planewidth = d.vi.width >> d.vi.GetPlaneWidthSubsampling(plane_enum);
        int planeheight = d.vi.height >> d.vi.GetPlaneHeightSubsampling(plane_enum);

        const int planewidth_real_or_lut = (lutmode == 0) ? planewidth: (1 << d.vi.BitsPerComponent());
        // to decide if partial chunk is left from the width at the end of the 4/8/16 pixel processing unit big main loops
        // when lut: fake width (x size of lut table) of the lut-init

        if (optAvx2 && d.planeOptAvx2[i]) {

          // avx2
          ExprEvalAvx2 ExprObj(d.ops[i], d.numInputs, env->GetCPUFlags(), planewidth_real_or_lut, planeheight, optSingleMode);
          if (ExprObj.GetCode(true) && ExprObj.GetCodeSize()) { // PF modded jitasm. true: epilog with vmovaps, and vzeroupper
            d.proc[i] = allocExecCode(ExprObj.GetCode(), ExprObj.GetCodeSize());
            if (!d.proc[i])
              env->ThrowError("Expr: cannot allocate executable memory for the JIT code");
            d.procSize[i] = ExprObj.GetCodeSize();
          }
        }
        else if (optSSE2 && d.planeOptSSE2[i]) {
          // sse2, sse4
          ExprEval ExprObj(d.ops[i], d.numInputs, env->GetCPUFlags(), planewidth_real_or_lut, planeheight, optSingleMode);
          if (ExprObj.GetCode() && ExprObj.GetCodeSize()) {
            d.proc[i] = allocExecCode(ExprObj.GetCode(), ExprObj.GetCodeSize());
            if (!d.proc[i])
              env->ThrowError("Expr: cannot allocate executable memory for the JIT code");
            d.procSize[i] = ExprObj.GetCodeSize();
          }
        }

      } // if plane is to be processed
    }
#ifdef VS_TARGET_OS_WINDOWS
    if (optSSE2)
      FlushInstructionCache(GetCurrentProcess(), nullptr, 0);
#endif
#endif

    if (lutmode > 0)
      calculate_lut(env);
  }
  catch (std::runtime_error &e) {
    for (int i = 0; i < MAX_EXPR_INPUTS; i++)
      d.clips[i] = nullptr; //  vsapi->freeNode(d->node[i]);
    std::string s = "Expr: ";
    s += e.what();
    env->ThrowError(s.c_str());
    return;
  }

}

#ifdef XP_TLS
#ifdef MSVC_PURE
// end of v141_xp toolset ultra slow build workaround
#pragma optimize("", on)
#endif
#endif
