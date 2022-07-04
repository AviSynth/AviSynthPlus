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

/*
** Turn. version 0.1
** (c) 2003 - Ernst Peché
**
*/

#include "turn.h"
#ifdef INTEL_INTRINSICS
#include "intel/turn_sse.h"
#endif
#include "resample.h"
#include "planeswap.h"
#include "../core/internal.h"
#include <stdint.h>


extern const AVSFunction Turn_filters[] = {
    { "TurnLeft",  BUILTIN_FUNC_PREFIX, "c", Turn::create_turnleft },
    { "TurnRight", BUILTIN_FUNC_PREFIX, "c", Turn::create_turnright },
    { "Turn180",   BUILTIN_FUNC_PREFIX, "c", Turn::create_turn180 },
    { 0 }
};

enum TurnDirection
{
  DIRECTION_LEFT = 0,
  DIRECTION_RIGHT = 1,
  DIRECTION_180 = 2
};


// TurnLeft() is FlipVertical().TurnRight().FlipVertical().
// Therefore, we don't have to implement both TurnRight() and TurnLeft().

template <typename T>
void turn_right_plane_c(const BYTE* srcp, BYTE* dstp, int src_rowsize, int height, int src_pitch, int dst_pitch)
{
    const BYTE* s0 = srcp + src_pitch * (height - 1);

    for (int y = 0; y < height; ++y)
    {
        BYTE* d0 = dstp;
        for (int x = 0; x < src_rowsize; x += sizeof(T))
        {
            *reinterpret_cast<T*>(d0) = *reinterpret_cast<const T*>(s0 + x);
            d0 += dst_pitch;
        }
        s0 -= src_pitch;
        dstp += sizeof(T);
    }
}


// Explicit instantiation, can be used from other modules.
template void turn_right_plane_c<uint64_t>(const BYTE*, BYTE*, int, int, int, int);


void turn_right_plane_8_c(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch)
{
    turn_right_plane_c<BYTE>(srcp, dstp, src_rowsize, src_height, src_pitch, dst_pitch);
}


void turn_left_plane_8_c(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch)
{
    turn_right_plane_c<BYTE>(srcp + src_pitch * (src_height - 1), dstp + dst_pitch * (src_rowsize - 1), src_rowsize, src_height, -src_pitch, -dst_pitch);
}


void turn_right_plane_16_c(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch)
{
    turn_right_plane_c<uint16_t>(srcp, dstp, src_rowsize, src_height, src_pitch, dst_pitch);
}


void turn_left_plane_16_c(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch)
{
    turn_right_plane_c<uint16_t>(srcp + src_pitch * (src_height - 1), dstp + dst_pitch * (src_rowsize / 2 - 1), src_rowsize, src_height, -src_pitch, -dst_pitch);
}


void turn_right_plane_32_c(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch)
{
    turn_right_plane_c<uint32_t>(srcp, dstp, src_rowsize, src_height, src_pitch, dst_pitch);
}


void turn_left_plane_32_c(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch)
{
    turn_right_plane_c<uint32_t>(srcp + src_pitch * (src_height - 1), dstp + dst_pitch * (src_rowsize / 4 - 1), src_rowsize, src_height, -src_pitch, -dst_pitch);
}


// on RGB, TurnLeft and TurnRight are reversed.
void turn_left_rgb32_c(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch)
{
    turn_right_plane_32_c(srcp, dstp, src_rowsize, src_height, src_pitch, dst_pitch);
}


void turn_right_rgb32_c(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch)
{
    turn_left_plane_32_c(srcp, dstp, src_rowsize, src_height, src_pitch, dst_pitch);
}


struct Rgb24 {
    BYTE b, g, r;
};


void turn_left_rgb24(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch)
{
    turn_right_plane_c<Rgb24>(srcp, dstp, src_rowsize, src_height, src_pitch, dst_pitch);
}


void turn_right_rgb24(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch)
{
    turn_right_plane_c<Rgb24>(srcp + src_pitch * (src_height - 1), dstp + dst_pitch * (src_rowsize / 3 - 1), src_rowsize, src_height, -src_pitch, -dst_pitch);
}


struct Rgb48 {
    uint16_t b, g, r;
};


void turn_left_rgb48_c(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch)
{
    turn_right_plane_c<Rgb48>(srcp, dstp, src_rowsize, src_height, src_pitch, dst_pitch);
}


void turn_right_rgb48_c(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch)
{
    turn_right_plane_c<Rgb48>(srcp + src_pitch * (src_height - 1), dstp + dst_pitch * (src_rowsize / 6 - 1), src_rowsize, src_height, -src_pitch, -dst_pitch);
}


void turn_left_rgb64_c(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch)
{
    turn_right_plane_c<uint64_t>(srcp, dstp, src_rowsize, src_height, src_pitch, dst_pitch);
}


void turn_right_rgb64_c(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch)
{
    turn_right_plane_c<uint64_t>(srcp + src_pitch * (src_height - 1), dstp + dst_pitch * (src_rowsize / 8 - 1), src_rowsize, src_height, -src_pitch, -dst_pitch);
}


static void turn_right_yuy2(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch)
{
    dstp += (src_height - 2) * 2;

    for (int y = 0; y < src_height; y += 2)
    {
        BYTE* d0 = dstp - y * 2;
        for (int x = 0; x < src_rowsize; x += 4)
        {
            int u = (srcp[x + 1] + srcp[x + 1 + src_pitch] + 1) / 2;
            int v = (srcp[x + 3] + srcp[x + 3 + src_pitch] + 1) / 2;

            d0[0] = srcp[x + src_pitch];
            d0[1] = u;
            d0[2] = srcp[x];
            d0[3] = v;
            d0 += dst_pitch;

            d0[0] = srcp[x + src_pitch + 2];
            d0[1] = u;
            d0[2] = srcp[x + 2];
            d0[3] = v;
            d0 += dst_pitch;
        }
        srcp += src_pitch * 2;
    }
}


static void turn_left_yuy2(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch)
{
    turn_right_yuy2(srcp + src_pitch * (src_height - 1), dstp + dst_pitch * (src_rowsize / 2 - 1), src_rowsize, src_height, -src_pitch, -dst_pitch);
}


template <typename T>
void turn_180_plane_c(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch)
{
    dstp += dst_pitch * (src_height - 1) + src_rowsize - sizeof(T);
    src_rowsize /= sizeof(T);

    for (int y = 0; y < src_height; ++y)
    {
        const T* s0 = reinterpret_cast<const T*>(srcp);
        T* d0 = reinterpret_cast<T*>(dstp);

        for (int x = 0; x < src_rowsize; ++x)
        {
            d0[-x] = s0[x];
        }
        srcp += src_pitch;
        dstp -= dst_pitch;
    }
}

static void turn_180_yuy2(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch)
{
    dstp += dst_pitch * (src_height - 1) + src_rowsize - 4;

    for (int y = 0; y < src_height; ++y)
    {
        for (int x = 0; x < src_rowsize; x += 4)
        {
            dstp[-x + 2] = srcp[x + 0];
            dstp[-x + 1] = srcp[x + 1];
            dstp[-x + 0] = srcp[x + 2];
            dstp[-x + 3] = srcp[x + 3];
        }
        srcp += src_pitch;
        dstp -= dst_pitch;
    }
}


Turn::Turn(PClip c, int direction, IScriptEnvironment* env) : GenericVideoFilter(c), u_or_b_source(nullptr), v_or_r_source(nullptr)
{
    if (vi.pixel_type & VideoInfo::CS_INTERLEAVED) {
        num_planes = 1;
    } else if (vi.IsPlanarRGBA() || vi.IsYUVA()) {
        num_planes = 4;
    } else {
        num_planes = 3;
    }

    splanes[0] = vi.IsRGB() ? PLANAR_G : PLANAR_Y;
    splanes[1] = vi.IsRGB() ? PLANAR_B : PLANAR_U;
    splanes[2] = vi.IsRGB() ? PLANAR_R : PLANAR_V;
    splanes[3] = PLANAR_A;

    if (direction != DIRECTION_180)
    {
        if (vi.IsYUY2() && (vi.height & 1))
        {
            env->ThrowError("Turn: YUY2 data must have mod2 height.");
        }
        if (num_planes > 1) {
            int mod_h = vi.IsRGB() ? 1 : (1 << vi.GetPlaneWidthSubsampling(PLANAR_U));
            int mod_v = vi.IsRGB() ? 1 : (1 << vi.GetPlaneHeightSubsampling(PLANAR_U));
            if (mod_h != mod_v)
            {
                if (vi.width % mod_h)
                {
                    env->ThrowError("Turn: Planar data must have MOD %d height.", mod_h);
                }
                if (vi.height % mod_v)
                {
                    env->ThrowError("Turn: Planar data must have MOD %d width.", mod_v);
                }
                SetUVSource(mod_h, mod_v, env);
            }
        }
        int t = vi.width;
        vi.width = vi.height;
        vi.height = t;
    }

    SetTurnFunction(direction, env);
}


void Turn::SetUVSource(int mod_h, int mod_v, IScriptEnvironment* env)
{
    MitchellNetravaliFilter filter(1.0 / 3, 1.0 / 3);
    AVSValue subs[4] = { 0.0, 0.0, 0.0, 0.0 };

    bool isRGB = vi.IsRGB(); // can be planar

    u_or_b_source = new SwapUVToY(child, isRGB ? SwapUVToY::BToY8 : SwapUVToY::UToY8, env); // Y16 and Y32 capable
    v_or_r_source = new SwapUVToY(child, isRGB ? SwapUVToY::RToY8 : SwapUVToY::VToY8, env);

    const VideoInfo& vi_u = u_or_b_source->GetVideoInfo();

    const int uv_height = vi_u.height * mod_v / mod_h;
    const int uv_width  = vi_u.width  * mod_h / mod_v;

    u_or_b_source = FilteredResize::CreateResize(u_or_b_source, uv_width, uv_height, subs, &filter, env);
    v_or_r_source = FilteredResize::CreateResize(v_or_r_source, uv_width, uv_height, subs, &filter, env);

    splanes[1] = 0;
    splanes[2] = 0;
}


void Turn::SetTurnFunction(int direction, IScriptEnvironment* env)
{
#ifdef INTEL_INTRINSICS
  const int cpu = env->GetCPUFlags();
  const bool sse2 = cpu & CPUF_SSE2;
  const bool ssse3 = cpu & CPUF_SSSE3;
#endif

  TurnFuncPtr funcs[3];
  auto set_funcs = [&funcs](TurnFuncPtr tleft, TurnFuncPtr tright, TurnFuncPtr t180) {
    funcs[0] = tleft;
    funcs[1] = tright;
    funcs[2] = t180;
  };

  if (vi.IsRGB64())
  {
#ifdef INTEL_INTRINSICS
    if (sse2)
      set_funcs(turn_left_rgb64_sse2, turn_right_rgb64_sse2, turn_180_plane_sse2<uint64_t>);
    else
#endif
    {
      set_funcs(turn_left_rgb64_c, turn_right_rgb64_c, turn_180_plane_c<uint64_t>);
    }

  }
  else if (vi.IsRGB48())
  {
    set_funcs(turn_left_rgb48_c, turn_right_rgb48_c, turn_180_plane_c<Rgb48>);
  }
  else if (vi.IsRGB32())
  {
#ifdef INTEL_INTRINSICS
    if (sse2)
      set_funcs(turn_left_rgb32_sse2, turn_right_rgb32_sse2, turn_180_plane_sse2<uint32_t>);
    else
#endif
    {
      set_funcs(turn_left_rgb32_c, turn_right_rgb32_c, turn_180_plane_c<uint32_t>);
    }
  }
  else if (vi.IsRGB24())
  {
    set_funcs(turn_left_rgb24, turn_right_rgb24, turn_180_plane_c<Rgb24>);
  }
  else if (vi.IsYUY2())
  {
    set_funcs(turn_left_yuy2, turn_right_yuy2, turn_180_yuy2);
  }
  else if (vi.ComponentSize() == 1) // 8 bit
  {
#ifdef INTEL_INTRINSICS
    if (sse2)
    {
      set_funcs(turn_left_plane_8_sse2, turn_right_plane_8_sse2,
        ssse3 ? turn_180_plane_ssse3<BYTE> : turn_180_plane_sse2<BYTE>);
    }
    else
#endif
    {
      set_funcs(turn_left_plane_8_c, turn_right_plane_8_c, turn_180_plane_c<BYTE>);
    }
  }
  else if (vi.ComponentSize() == 2) // 16 bit
  {
#ifdef INTEL_INTRINSICS
    if (sse2)
    {
      set_funcs(turn_left_plane_16_sse2, turn_right_plane_16_sse2,
        ssse3 ? turn_180_plane_ssse3<uint16_t> : turn_180_plane_sse2<uint16_t>);
    }
    else

#endif
    {
      set_funcs(turn_left_plane_16_c, turn_right_plane_16_c, turn_180_plane_c<uint16_t>);
    }
  }
  else if (vi.ComponentSize() == 4) // 32 bit
  {
#ifdef INTEL_INTRINSICS
    if (sse2) {
      set_funcs(turn_left_plane_32_sse2, turn_right_plane_32_sse2, turn_180_plane_sse2<uint32_t>);
    }
    else

#endif
    {
      set_funcs(turn_left_plane_32_c, turn_right_plane_32_c, turn_180_plane_c<uint32_t>);
    }
  }
  else env->ThrowError("Turn: Image format not supported!");

  turn_function = funcs[direction];
}


int __stdcall Turn::SetCacheHints(int cachehints, int frame_range)
{
  AVS_UNUSED(frame_range);
  return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
}


PVideoFrame __stdcall Turn::GetFrame(int n, IScriptEnvironment* env)
{
    static const int dplanes[] = {
        0,
        vi.IsRGB() ? PLANAR_B : PLANAR_U,
        vi.IsRGB() ? PLANAR_R : PLANAR_V,
        PLANAR_A,
    };

    auto src = child->GetFrame(n, env);
    auto dst = env->NewVideoFrameP(vi, &src);

    PVideoFrame srcs[4] = {
        src,
        u_or_b_source ? u_or_b_source->GetFrame(n, env) : src,
        v_or_r_source ? v_or_r_source->GetFrame(n, env) : src,
        src,
    };

    for (int p = 0; p < num_planes; ++p) {
        const int splane = splanes[p];
        const int dplane = dplanes[p];
        turn_function(srcs[p]->GetReadPtr(splane), dst->GetWritePtr(dplane),
                      srcs[p]->GetRowSize(splane), srcs[p]->GetHeight(splane),
                      srcs[p]->GetPitch(splane), dst->GetPitch(dplane));
    }

    return dst;
}


AVSValue __cdecl Turn::create_turnleft(AVSValue args, void* , IScriptEnvironment* env)
{
    return new Turn(args[0].AsClip(), DIRECTION_LEFT, env);
}


AVSValue __cdecl Turn::create_turnright(AVSValue args, void* , IScriptEnvironment* env)
{
    return new Turn(args[0].AsClip(), DIRECTION_RIGHT, env);
}


AVSValue __cdecl Turn::create_turn180(AVSValue args, void* , IScriptEnvironment* env)
{
    return new Turn(args[0].AsClip(), DIRECTION_180, env);
}
