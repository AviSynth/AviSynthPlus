// Avisynth+
// https://avs-plus.net
//
// This file is part of Avisynth+ which is released under GPL2+ with exception.

// Convert Audio helper functions (AVX2)
// Copyright (c) 2020 Xinyue Lu, (c) 2021 pinterf

#include <avs/types.h>
#include <avs/config.h>
#include <immintrin.h> // AVX2 at most

// Easy: 32-16, 16-32
// Float: 8/16/32-FLT, FLT-8/16/32

void convert32To16_AVX2(void *inbuf, void *outbuf, int count) {
  auto in = reinterpret_cast<int32_t *>(inbuf);
  auto in16 = reinterpret_cast<int16_t *>(inbuf);
  auto out = reinterpret_cast<int16_t *>(outbuf);

  const int c_loop = count & ~15;

  for (int i = c_loop; i < count; i++)
    out[i] = in16[i * 2 + 1];

  for (int i = 0; i < c_loop; i += 16) {
    __m256i in32a = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(in)); in += 8;
    __m256i in32b = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(in)); in += 8;
    __m256i in16a = _mm256_srai_epi32(in32a, 16);
    __m256i in16b = _mm256_srai_epi32(in32b, 16);
    __m256i out16 = _mm256_packs_epi32(in16a, in16b);
    out16 = _mm256_permute4x64_epi64(out16, 216);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(out), out16); out += 16;
  }
}

void convert16To32_AVX2(void *inbuf, void *outbuf, int count) {
  auto in = reinterpret_cast<int16_t *>(inbuf);
  auto out = reinterpret_cast<int32_t *>(outbuf);
  auto out16 = reinterpret_cast<int16_t *>(outbuf);

  const int c_loop = count & ~15;

  for (int i = c_loop; i < count; i++) {
    out16[i * 2] = 0;
    out16[i * 2 + 1] = in[i];
  }

  __m256i zero = _mm256_set1_epi16(0);
  for (int i = 0; i < c_loop; i += 16) {
    __m256i in16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(in)); in += 16;
    in16 = _mm256_permute4x64_epi64(in16, 216);
    __m256i out32a = _mm256_unpacklo_epi16(zero, in16);
    __m256i out32b = _mm256_unpackhi_epi16(zero, in16);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(out), out32a); out += 8;
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(out), out32b); out += 8;
  }
}

void convert8ToFLT_AVX2(void* inbuf, void* outbuf, int count) {
  auto in = reinterpret_cast<uint8_t*>(inbuf);
  auto out = reinterpret_cast<SFLOAT*>(outbuf);
  constexpr float divisor = 1.0f / 128.f; // 1 << 7

  const int c_loop = count & ~7;

  for (int i = c_loop; i < count; i++)
    out[i] = (in[i] - 128) * divisor;

  __m256 divv = _mm256_set1_ps(divisor);
  for (int i = 0; i < c_loop; i += 8) {
    __m128i src = _mm_loadl_epi64(reinterpret_cast<__m128i*>(in)); in += 8;
    __m256i in32 = _mm256_cvtepu8_epi32(src);
    in32 = _mm256_sub_epi32(in32, _mm256_set1_epi32(128));
    __m256 infl = _mm256_cvtepi32_ps(in32);
    __m256 outfl = _mm256_mul_ps(infl, divv);
    _mm256_storeu_ps(out, outfl); out += 8;
  }
}

void convertFLTTo8_AVX2(void* inbuf, void* outbuf, int count) {
  auto in = reinterpret_cast<SFLOAT*>(inbuf);
  auto out = reinterpret_cast<uint8_t*>(outbuf);
  constexpr float multiplier = 128.f;
  constexpr float max8 = 127.f;
  constexpr float min8 = -128.f;

  const int c_loop = count & ~15;

  for (int i = c_loop; i < count; i++) {
    float val = in[i] * multiplier;
    uint8_t result;
    if (val >= max8) result = 255;
    else if (val <= min8) result = 0;
    else result = static_cast<int8_t>(val) + 128;
    out[i] = result;
  }

  __m256 mulv = _mm256_set1_ps(multiplier);
  __m256 maxv = _mm256_set1_ps(max8);
  __m256 minv = _mm256_set1_ps(min8);
  for (int i = 0; i < c_loop; i += 16) {
    __m256 infl_lo = _mm256_loadu_ps(in); in += 8;
    __m256 infl_hi = _mm256_loadu_ps(in); in += 8;
    __m256 outfl_lo = _mm256_max_ps(minv, _mm256_min_ps(maxv, _mm256_mul_ps(infl_lo, mulv)));
    __m256 outfl_hi = _mm256_max_ps(minv, _mm256_min_ps(maxv, _mm256_mul_ps(infl_hi, mulv)));
    __m256i out32_lo = _mm256_cvttps_epi32(outfl_lo);
    __m256i out32_hi = _mm256_cvttps_epi32(outfl_hi);
    __m256i out16 = _mm256_packs_epi32(out32_lo, out32_hi);

    out16 = _mm256_permute4x64_epi64(out16, (0 << 0) | (2 << 2) | (1 << 4) | (3 << 6));
    __m128i out16_lo = _mm256_castsi256_si128(out16);
    __m128i out16_hi = _mm256_extractf128_si256(out16, 1);
    __m128i out8 = _mm_packs_epi16(out16_lo, out16_hi);
    out8 = _mm_add_epi8(out8, _mm_set1_epi8(-128)); // 128
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out), out8); out += 16;
  }
}

void convert16ToFLT_AVX2(void* inbuf, void* outbuf, int count) {
  auto in = reinterpret_cast<int16_t*>(inbuf);
  auto out = reinterpret_cast<SFLOAT*>(outbuf);
  constexpr float divisor = 1.0f / 32768.f; // 1 << 15

  const int c_loop = count & ~7;

  for (int i = c_loop; i < count; i++)
    out[i] = in[i] * divisor;

  __m256 divv = _mm256_set1_ps(divisor);
  for (int i = 0; i < c_loop; i += 8) {
    __m128i src = _mm_loadu_si128(reinterpret_cast<__m128i*>(in)); in += 8;
    __m256i in32 = _mm256_cvtepi16_epi32(src);
    __m256 infl = _mm256_cvtepi32_ps(in32);
    __m256 outfl = _mm256_mul_ps(infl, divv);
    _mm256_storeu_ps(out, outfl); out += 8;
  }
}

void convertFLTTo16_AVX2(void* inbuf, void* outbuf, int count) {
  auto in = reinterpret_cast<SFLOAT*>(inbuf);
  auto out = reinterpret_cast<int16_t*>(outbuf);
  constexpr float multiplier = 32768.f;
  constexpr float max16 = 32767.f;
  constexpr float min16 = -32768.f;

  const int c_loop = count & ~7;

  for (int i = c_loop; i < count; i++) {
    float val = in[i] * multiplier;
    int16_t result;
    if (val >= max16) result = 32767;
    else if (val <= min16) result = (int16_t)-32768;
    else result = static_cast<int16_t>(val);
    out[i] = result;
  }

  __m256 mulv = _mm256_set1_ps(multiplier);
  __m256 maxv = _mm256_set1_ps(max16);
  __m256 minv = _mm256_set1_ps(min16);
  for (int i = 0; i < c_loop; i += 8) {
    __m256 infl = _mm256_loadu_ps(in); in += 8;
    __m256 outfl = _mm256_max_ps(minv, _mm256_min_ps(maxv, _mm256_mul_ps(infl, mulv)));
    __m256i out32 = _mm256_cvttps_epi32(outfl);
    __m256i out16 = _mm256_packs_epi32(out32, out32);

    out16 = _mm256_permute4x64_epi64(out16, (0 << 0) | (2 << 2) | (1 << 4) | (3 << 6));

    _mm_storeu_si128(reinterpret_cast<__m128i*>(out), _mm256_castsi256_si128(out16)); out += 8;
  }
}



void convert32ToFLT_AVX2(void *inbuf, void *outbuf, int count) {
  auto in = reinterpret_cast<int32_t *>(inbuf);
  auto out = reinterpret_cast<SFLOAT *>(outbuf);
  const float divisor = 1.0f/2147483648.0f;

  const int c_loop = count & ~7;

  for (int i = c_loop; i < count; i++)
    out[i] = in[i] * divisor;

  __m256 divv = _mm256_set1_ps(divisor);
  for (int i = 0; i < c_loop; i += 8) {
    __m256i in32 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(in)); in += 8;
    __m256 infl = _mm256_cvtepi32_ps(in32);
    __m256 outfl = _mm256_mul_ps(infl, divv);
    _mm256_storeu_ps(out, outfl); out += 8;
  }
}

void convertFLTTo32_AVX2(void *inbuf, void *outbuf, int count) {
  auto in = reinterpret_cast<SFLOAT *>(inbuf);
  auto out = reinterpret_cast<int32_t *>(outbuf);
  const float multiplier = 2147483648.0f;
  const float max32 = 2147483647.0f; // 2147483648.0f in reality
  const float min32 = -2147483648.0f;

  const int c_loop = count & ~7;

  for (int i = c_loop; i < count; i++) {
    float val = in[i] * multiplier;
    int32_t result;
    if (val >= max32) result = 0x7FFFFFFF; // 2147483647
    else if (val <= min32) result = 0x80000000; // -2147483648
    else result = static_cast<int32_t>(val);
    out[i] = result;
  }

  __m256 mulv = _mm256_set1_ps(multiplier);
  __m256 maxv = _mm256_set1_ps(max32);
  __m256 minv = _mm256_set1_ps(min32);
  __m256i maxv_i = _mm256_set1_epi32(0x7FFFFFFF); // 2147483647
  __m256i minv_i = _mm256_set1_epi32(0x80000000); // -2147483648
  for (int i = 0; i < c_loop; i += 8) {
    __m256 infl = _mm256_loadu_ps(in); in += 8;
    __m256 outfl = _mm256_mul_ps(infl, mulv);
    __m256i cmphigh = _mm256_castps_si256(_mm256_cmp_ps(outfl, maxv, _CMP_GE_OS));
    __m256i cmplow = _mm256_castps_si256(_mm256_cmp_ps(minv, outfl, _CMP_GE_OS));
    __m256i out32 = _mm256_cvttps_epi32(outfl);
    out32 = _mm256_blendv_epi8(out32, maxv_i, cmphigh);
    out32 = _mm256_blendv_epi8(out32, minv_i, cmplow);

    _mm256_storeu_si256(reinterpret_cast<__m256i *>(out), out32); out += 8;
  }
}
