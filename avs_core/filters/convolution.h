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



// Avisynth filter: general convolution 3d
// by Richard Berg (avisynth-dev@richardberg.net)
// adapted from General Convolution 3D for VDub by Gunnar Thalin (guth@home.se)

#ifndef __GeneralConvolution_H__
#define __GeneralConvolution_H__

#include <avisynth.h>
#include <vector>

/*****************************************
****** General Convolution 2D filter *****
*****************************************/


class GeneralConvolution : public GenericVideoFilter
/** This class exposes a video filter that applies general convolutions -- up to a 5x5
  * kernel -- to a clip.  Smaller (3x3) kernels have their own code path.  SIMD support forthcoming.
 **/
{
  using do_conv_int_t = void(BYTE* dstp8, int dst_pitch, const BYTE *srcp8, int src_pitch, int width, int height, const int *matrix, int iCountDiv, int iBias);
  using do_conv_float_t = void(BYTE* dstp8, int dst_pitch, const BYTE *srcp8, int src_pitch, int width, int height, const float *matrix, float fCountDiv, float fBias);

public:
    GeneralConvolution(PClip _child, double _divisor, float _nBias, const char * _matrix, bool _autoscale, bool _luma, bool _chroma, bool _alpha, IScriptEnvironment* _env);
    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;
    static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env);

    int __stdcall SetCacheHints(int cachehints, int frame_range) override {
      AVS_UNUSED(frame_range);
      return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
    }

protected:
    void setMatrix(const char * _matrix, bool _isinteger, IScriptEnvironment* env);

private:
    double divisor;
    size_t nSize;
    int nBias;
    float fBias;
    bool autoscale;

    int iCountDiv;
    float fCountDiv;
    bool int64needed;

    bool luma;
    bool chroma;
    bool alpha;

    std::vector<int> iMatrix;
    std::vector<float> fMatrix;
    float fNormalizeSum;
    int iNormalizeSum;
    int iWeightSumPositives;
    int iWeightSumNegatives;

    do_conv_int_t *conversionFnPtr;
    do_conv_float_t *FconversionFnPtr;

};



#endif  // __GeneralConvolution_H__
