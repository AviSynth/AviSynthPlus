// Avisynth v2.5.  Copyright 2002 Ben Rudiak-Gould et al.
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

// ConvertNativeToStacked, ConvertStackedToNative 2016 by pinterf

#include <avisynth.h>
#include <avs/alignment.h>
#include <cstdint>
#ifdef INTEL_INTRINSICS
#include <emmintrin.h>
#endif // INTEL_INTRINSICS

class ConvertToStacked : public GenericVideoFilter
{
public:

    ConvertToStacked(PClip src, IScriptEnvironment* env) : GenericVideoFilter(src)
    {

        if (vi.IsColorSpace(VideoInfo::CS_YUV420P16)) vi.pixel_type = VideoInfo::CS_YV12;
        else if (vi.IsColorSpace(VideoInfo::CS_YUV422P16)) vi.pixel_type = VideoInfo::CS_YV16;
        else if (vi.IsColorSpace(VideoInfo::CS_YUV444P16)) vi.pixel_type = VideoInfo::CS_YV24;
        else if (vi.IsColorSpace(VideoInfo::CS_Y16)) vi.pixel_type = VideoInfo::CS_Y8;
        else env->ThrowError("ConvertToStacked: Input clip must be native 16 bit: YUV420P16, YUV422P16, YUV444P16, Y16");

        vi.height = vi.height << 1; // * 2 stacked
                                    // back from native 16 bit to stacked 8 bit

        return;
    }

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override
    {
        PVideoFrame src = child->GetFrame(n, env);
        PVideoFrame dst = env->NewVideoFrameP(vi, &src);

        const int planes[] = { PLANAR_Y, PLANAR_U, PLANAR_V };
        const int plane_count = vi.IsY8() ? 1 : 3; // checking the stacked 8 bit format constants
        for (int p = 0; p < plane_count; ++p) {
            const int plane = planes[p];
            const uint16_t* srcp = reinterpret_cast<const uint16_t*>(src->GetReadPtr(plane));
            uint8_t* msb = dst->GetWritePtr(plane);
            const int src_pitch = src->GetPitch(plane) / sizeof(uint16_t);
            const int dst_pitch = dst->GetPitch(plane);
            const int height = src->GetHeight(plane); // non-stacked real height
            const int width = dst->GetRowSize(plane);
            uint8_t* lsb = msb + dst_pitch*height;

#ifdef INTEL_INTRINSICS
            bool use_sse2 = (env->GetCPUFlags() & CPUF_SSE2) && IsPtrAligned(msb, 16) && IsPtrAligned(srcp, 16);

            if (use_sse2)
            {
                // read 32bytes from src, write 16bytes to msb and lsb.
                // pitch is aligned at least 32 bytes. Thus, an access violation does not happen.
                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; x += 16) {
                        __m128i data16_1, data16_2;
                        __m128i masklo = _mm_set1_epi16(0x00FF);
                        // no gain when sse4.1 _mm_stream_load_si128 is used
                        data16_1 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp + x));       // 16 bytes, 8 words ABCDEFGH
                        data16_2 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp + x + 8));   // 16 bytes, 8 words
                        _mm_stream_si128(reinterpret_cast<__m128i*>(msb + x), _mm_packus_epi16(_mm_srli_epi16(data16_1, 8), _mm_srli_epi16(data16_2, 8))); // ABCDEFGH Hi
                        _mm_stream_si128(reinterpret_cast<__m128i*>(lsb + x), _mm_packus_epi16(_mm_and_si128(data16_1, masklo), _mm_and_si128(data16_2, masklo))); // ABCDEFGH Lo
                    }
                    srcp += src_pitch;
                    msb += dst_pitch;
                    lsb += dst_pitch;
                } // y
            }
            else
#endif // INTEL_INTRINSICS
            {
                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x) {
                        const uint16_t out = srcp[x];
                        msb[x] = out >> 8;
                        lsb[x] = (uint8_t)out;
                    }

                    srcp += src_pitch;
                    msb += dst_pitch;
                    lsb += dst_pitch;
                }
            }
        }
        return dst;
    }

    int __stdcall SetCacheHints(int cachehints, int frame_range) override
    {
        return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
    }

    static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env)
    {
        PClip clip = args[0].AsClip();
        /*if (clip->GetVideoInfo().IsY8())
        return clip;*/
        return new ConvertToStacked(clip, env);
    }
};


class ConvertFromStacked : public GenericVideoFilter
{
public:

    ConvertFromStacked(PClip src, int bits, IScriptEnvironment* env) : GenericVideoFilter(src)
    {
        if (bits == 10 && vi.IsYV12())
            vi.pixel_type = VideoInfo::CS_YUV420P10;
        else if (bits == 10 && vi.IsYV16())
            vi.pixel_type = VideoInfo::CS_YUV422P10;
        else if (bits == 10 && vi.IsYV24())
            vi.pixel_type = VideoInfo::CS_YUV444P10;
        else if (bits == 10 && vi.IsY8())
            vi.pixel_type = VideoInfo::CS_Y10;
        else if (bits == 12 && vi.IsYV12())
            vi.pixel_type = VideoInfo::CS_YUV420P12;
        else if (bits == 12 && vi.IsYV16())
            vi.pixel_type = VideoInfo::CS_YUV422P12;
        else if (bits == 12 && vi.IsYV24())
            vi.pixel_type = VideoInfo::CS_YUV444P12;
        else if (bits == 12 && vi.IsY8())
            vi.pixel_type = VideoInfo::CS_Y12;
        else if (bits == 14 && vi.IsYV12())
            vi.pixel_type = VideoInfo::CS_YUV420P14;
        else if (bits == 14 && vi.IsYV16())
            vi.pixel_type = VideoInfo::CS_YUV422P14;
        else if (bits == 14 && vi.IsYV24())
            vi.pixel_type = VideoInfo::CS_YUV444P14;
        else if (bits == 14 && vi.IsY8())
            vi.pixel_type = VideoInfo::CS_Y14;
        else if (bits == 16 && vi.IsYV12())
            vi.pixel_type = VideoInfo::CS_YUV420P16;
        else if (bits == 16 && vi.IsYV16())
            vi.pixel_type = VideoInfo::CS_YUV422P16;
        else if (bits == 16 && vi.IsYV24())
            vi.pixel_type = VideoInfo::CS_YUV444P16;
        else if (bits == 16 && vi.IsY8())
            vi.pixel_type = VideoInfo::CS_Y16;
        else env->ThrowError("ConvertFromStacked: Input stacked clip must be YV12, YV16, YV24 or Y8");

        vi.height = vi.height >> 1; // div 2 non stacked

        return;
    }

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override
    {
        PVideoFrame src = child->GetFrame(n, env);
        PVideoFrame dst = env->NewVideoFrameP(vi, &src);
        const int planes[] = { PLANAR_Y, PLANAR_U, PLANAR_V };
        const int plane_count = vi.IsY() ? 1 : 3;
        for (int p = 0; p < plane_count; ++p) {
            const int plane = planes[p];
            const uint8_t* msb = src->GetReadPtr(plane);
            uint16_t* dstp = reinterpret_cast<uint16_t*>(dst->GetWritePtr(plane));
            const int src_pitch = src->GetPitch(plane);
            const int dst_pitch = dst->GetPitch(plane) / sizeof(uint16_t);
            const int height = dst->GetHeight(plane); // real non-stacked height
            const int width = src->GetRowSize(plane);
            const uint8_t* lsb = msb + src_pitch*height;

#ifdef INTEL_INTRINSICS
            bool use_sse2 = (env->GetCPUFlags() & CPUF_SSE2) && IsPtrAligned(msb, 16) && IsPtrAligned(dstp, 16);

            if (use_sse2)
            {
                // pf my very first intrinsic, hey
                for (int y = 0; y < height; ++y) {
                    // Read 16 bytes from msb and lsb, write 32bytes to dst.
                    // pitch is aligned at least 32bytes. Thus, we don't have to care about buffer over run.
                    for (int x = 0; x < width; x += 16) {
                        __m128i data_hi, data_lo;
                        data_hi = _mm_load_si128(reinterpret_cast<const __m128i*>(msb + x)); // 16 bytes
                        data_lo = _mm_load_si128(reinterpret_cast<const __m128i*>(lsb + x));
                        // Interleaves the lower 8 signed or unsigned 8-bit integers in a with the lower 8 signed or unsigned 8-bit integers in b.
                        _mm_stream_si128(reinterpret_cast<__m128i*>(dstp + x), _mm_unpacklo_epi8(data_lo, data_hi));
                        // Interleaves the higher 8 signed or unsigned 8-bit integers in a with the lower 8 signed or unsigned 8-bit integers in b.
                        _mm_stream_si128(reinterpret_cast<__m128i*>(dstp + x + 8), _mm_unpackhi_epi8(data_lo, data_hi));
                    }
                    msb += src_pitch;
                    lsb += src_pitch;
                    dstp += dst_pitch;
                }
            }
            else
#endif // INTEL_INTRINSICS
           {
                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x) {
                        dstp[x] = msb[x] << 8 | lsb[x];
                    }
                    msb += src_pitch;
                    lsb += src_pitch;
                    dstp += dst_pitch;
                }
            }
        }
        return dst;
    }

    int __stdcall SetCacheHints(int cachehints, int frame_range) override
    {
        return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
    }


    static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env)
    {
        PClip clip = args[0].AsClip();
        int bits = args[1].AsInt(16);

        return new ConvertFromStacked(clip, bits, env);
    }
};


class ConvertFromDoubleWidth : public GenericVideoFilter
{
public:

    ConvertFromDoubleWidth(PClip src, int bits, IScriptEnvironment* env) : GenericVideoFilter(src)
    {
        if (!vi.IsRGB() && vi.RowSize(PLANAR_U) % 2)
            env->ThrowError("ConvertFromDoubleWidth: Input clip's chroma width must be even.");
        if ((vi.IsRGB24() || vi.IsRGB32()) && bits != 16)
          env->ThrowError("ConvertFromDoubleWidth: only bits=16 allowed for RGB24 or RGB32 input");

        if (bits == 10 && vi.IsYV12())
            vi.pixel_type = VideoInfo::CS_YUV420P10;
        else if (bits == 10 && vi.IsYV16())
            vi.pixel_type = VideoInfo::CS_YUV422P10;
        else if (bits == 10 && vi.IsYV24())
            vi.pixel_type = VideoInfo::CS_YUV444P10;
        else if (bits == 10 && vi.IsY8())
            vi.pixel_type = VideoInfo::CS_Y10;
        else if (bits == 12 && vi.IsYV12())
            vi.pixel_type = VideoInfo::CS_YUV420P12;
        else if (bits == 12 && vi.IsYV16())
            vi.pixel_type = VideoInfo::CS_YUV422P12;
        else if (bits == 12 && vi.IsYV24())
            vi.pixel_type = VideoInfo::CS_YUV444P12;
        else if (bits == 12 && vi.IsY8())
            vi.pixel_type = VideoInfo::CS_Y12;
        else if (bits == 14 && vi.IsYV12())
            vi.pixel_type = VideoInfo::CS_YUV420P14;
        else if (bits == 14 && vi.IsYV16())
            vi.pixel_type = VideoInfo::CS_YUV422P14;
        else if (bits == 14 && vi.IsYV24())
            vi.pixel_type = VideoInfo::CS_YUV444P14;
        else if (bits == 14 && vi.IsY8())
            vi.pixel_type = VideoInfo::CS_Y14;
        else if (bits == 16 && vi.IsYV12())
            vi.pixel_type = VideoInfo::CS_YUV420P16;
        else if (bits == 16 && vi.IsYV16())
            vi.pixel_type = VideoInfo::CS_YUV422P16;
        else if (bits == 16 && vi.IsYV24())
            vi.pixel_type = VideoInfo::CS_YUV444P16;
        else if (bits == 16 && vi.IsY8())
            vi.pixel_type = VideoInfo::CS_Y16;
        else if (bits == 16 && vi.IsRGB24())
          vi.pixel_type = VideoInfo::CS_BGR48;
        else if (bits == 16 && vi.IsRGB32())
          vi.pixel_type = VideoInfo::CS_BGR64;
        else env->ThrowError("ConvertFromDoubleWidth: Input double width clip must be YV12, YV16, YV24, Y8, RGB24 or RGB32");

        vi.width /= 2;
    }

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override
    {
        return child->GetFrame(n, env);
    }

    int __stdcall SetCacheHints(int cachehints, int frame_range) override
    {
        return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
    }


    static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env)
    {
        PClip clip = args[0].AsClip();
        int bits = args[1].AsInt(16);

        return new ConvertFromDoubleWidth(clip, bits, env);
    }
};


class ConvertToDoubleWidth : public GenericVideoFilter
{
public:

    ConvertToDoubleWidth(PClip src, IScriptEnvironment* env) : GenericVideoFilter(src)
    {
        if (vi.IsColorSpace(VideoInfo::CS_YUV420P16)) vi.pixel_type = VideoInfo::CS_YV12;
        else if (vi.IsColorSpace(VideoInfo::CS_YUV422P16)) vi.pixel_type = VideoInfo::CS_YV16;
        else if (vi.IsColorSpace(VideoInfo::CS_YUV444P16)) vi.pixel_type = VideoInfo::CS_YV24;
        else if (vi.IsColorSpace(VideoInfo::CS_Y16)) vi.pixel_type = VideoInfo::CS_Y8;
        else if (vi.IsColorSpace(VideoInfo::CS_BGR48)) vi.pixel_type = VideoInfo::CS_BGR24;
        else if (vi.IsColorSpace(VideoInfo::CS_BGR64)) vi.pixel_type = VideoInfo::CS_BGR32;
        else env->ThrowError("ConvertToDoubleWidth: Input clip must be 16bit YUV format, RGB48 or RGB64");

        vi.width *= 2;
    }

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override
    {
        return child->GetFrame(n, env);
    }

    int __stdcall SetCacheHints(int cachehints, int frame_range) override
    {
        return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
    }

    static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env)
    {
        PClip clip = args[0].AsClip();
        return new ConvertToDoubleWidth(clip, env);
    }
};


const AVS_Linkage* AVS_linkage = 0;
extern "C" __declspec(dllexport) const char* __stdcall AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors) {
    AVS_linkage = vectors;

    env->AddFunction("ConvertFromStacked", "c[bits]i", ConvertFromStacked::Create, 0);
    env->AddFunction("ConvertToStacked", "c", ConvertToStacked::Create, 0);
    env->AddFunction("ConvertFromDoubleWidth", "c[bits]i", ConvertFromDoubleWidth::Create, 0);
    env->AddFunction("ConvertToDoubleWidth", "c", ConvertToDoubleWidth::Create, 0);

    return "`ConvertStacked' Stacked format conversion for 16-bit formats.";
}
