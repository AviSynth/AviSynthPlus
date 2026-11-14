
Quick Reference
===============

**Abbreviations:**

Named arguments are written as "arg" and are always optional, non-named
optional arguments are placed into [] brackets.
The types of the non-clip-arguments can be int (integer), float, string or
bool.

*[yuy2]* etc. denotes the color formats the filter can deal with. See
:doc:`colorspace conversion filters <corefilters/convert>` for more details.

*[16 bit, float]* etc, denotes the sample types which are supported by the
audio filter. "float" is the most accurate one.

*[v2.53]* denotes the version of AviSynth (above 2.01) since the filter is
available, or the last changes in the filter.


--------


Alphabetic view
================

:ref:`A` :ref:`B` :ref:`C` :ref:`D` :ref:`E` :ref:`F` :ref:`G` :ref:`H` :ref:`I` [J] :ref:`K`
:ref:`L` :ref:`M` :ref:`N` :ref:`O` :ref:`P` [Q] :ref:`R` :ref:`S` :ref:`T` [U] :ref:`V` :ref:`W`
[X] [Y] [Z]

.. _A:

A
-

:doc:`AddBorders <corefilters/addborders>` *[yv24] [yv16] [yv12] [yv411] [y8] [yuy2] [rgb32] [rgb24]*

-   ``AddBorders`` (clip, int left, int top, int right, int bottom, int
    "color")

:doc:`AudioStats <corefilters/audiostats>` *[all audio sample types]*

-   ``AudioStats`` (clip, float "threshold", bool "per_channel") *[v3.7.7]*

:doc:`AudioTrim <corefilters/trim>`

-   ``AudioTrim`` (clip, float start_time, float end_time) *[v2.60]*
-   ``AudioTrim`` (clip, float start_time, float -duration) *[v2.60]*
-   ``AudioTrim`` (clip, float start_time, float "end") *[v2.60]*
-   ``AudioTrim`` (clip, float start_time, float "length") *[v2.60]*

:doc:`Amplify / AmplifydB <corefilters/amplify>` *[16 bit, float]*

-   ``Amplify`` (clip, float amount1 [, ...])
-   ``AmplifydB`` (clip, float amount1 [, ...])

:doc:`Animate / ApplyRange <corefilters/animate>` *[yv24] [yv16] [yv12] [yv411] [y8] [yuy2] [rgb32]
[rgb24]*

-   ``Animate`` (clip, int start_frame, int end_frame, string filtername,
    start_args, end_args)
-   ``ApplyRange`` (clip, int start_frame, int end_frame, string
    filtername, args)* [v2.51]*

:doc:`AssumeFPS <corefilters/fps>` *[yv24] [yv16] [yv12] [yv411] [y8] [yuy2] [rgb32] [rgb24]*

-   ``AssumeFPS`` (clip, int numerator , int denominator, bool
    "sync_audio")
-   ``AssumeFPS`` (clip, float fps, bool "sync_audio")
-   ``AssumeFPS`` (clip1, clip2, bool "sync_audio") *[v2.55]*
-   ``AssumeFPS`` (clip, string preset) *[v2.57]*

:ref:`AssumeFrameBased / AssumeFieldBased <AssumeFrameField>` *[yv24] [yv16] [yv12] [yv411] [y8]
[yuy2] [rgb32] [rgb24]*

-   ``AssumeFrameBased`` (clip)
-   ``AssumeFieldBased`` (clip)

:ref:`AssumeBFF / AssumeTFF <AssumeFieldFirst>` *[yv24] [yv16] [yv12] [yv411] [y8] [yuy2] [rgb32]
[rgb24]*

-   ``AssumeBFF`` (clip)
-   ``AssumeTFF`` (clip)

:doc:`AssumeSampleRate <corefilters/assumerate>` *[all]*

-   ``AssumeSampleRate`` (clip, int samplerate)

:ref:`AssumeScaledFPS` *[yv24] [yv16] [yv12] [yv411] [y8] [yuy2] [rgb32] [rgb24]*

-   ``AssumeScaledFPS`` (clip, int "multiplier", int "divisor", bool
    "sync_audio") *[v2.56]*

:doc:`AudioDub / AudioDubEx <corefilters/audiodub>` *[all]*

-   ``AudioDub`` (video_clip, audio_clip)
-   ``AudioDubEx`` (video_clip, audio_clip) *[v2.56]*

:doc:`AVISource / OpenDMLSource / AVIFileSource / WAVSource <corefilters/avisource>`

-   ``AVISource`` (string filename [, ...], bool "audio", string
    "pixel_type" [, string fourCC])
-   ``OpenDMLSource`` (string filename [, ...], bool "audio", string
    "pixel_type" [, string fourCC])
-   ``AVIFileSource`` (string filename [, ...], bool "audio", string
    "pixel_type" [, string fourCC])
-   ``WAVSource`` (string filename [, ...])

.. _B:

B
-

:doc:`BlankClip / Blackness <corefilters/blankclip>` *[yv24] [yv16] [yv12] [yv411] [y8] [yuy2] [rgb32]
[rgb24]*

-   ``BlankClip`` (clip clip, int "length", int "width", int "height",
    string "pixel_type", float "fps", int "fps_denominator",
    int "audio_rate", bool "stereo", bool "sixteen_bit", int "color", int
    "color_yuv")
-   ``BlankClip`` (clip clip, int "length", int "width", int "height",
    string "pixel_type", float "fps", int "fps_denominator",
    int "audio_rate", int "channels", string "sample_type", int "color",
    int "color_yuv") *[v2.58]*
-   ``Blackness`` ()

:doc:`Blur / Sharpen <corefilters/blur>` *[yv24] [yv16] [yv12] [yv411] [y8] [yuy2] [rgb32] [rgb24]*

-   ``Blur`` (clip, float amount)
-   ``Blur`` (clip, float amountH, float amountV)
-   ``Sharpen`` (clip, float amount)
-   ``Sharpen`` (clip, float amountH, float amountV)

:doc:`Bob <corefilters/bob>` *[yv24] [yv16] [yv12] [yv411] [y8] [yuy2] [rgb32] [rgb24]*

-   ``Bob`` (clip, float "b", float "c", float "height")

.. _C:

C
-

:ref:`ChangeFPS` *[yv24] [yv16] [yv12] [yv411] [y8] [yuy2] [rgb32] [rgb24]*

-   ``ChangeFPS`` (clip, int numerator , int denominator, bool "linear")
    *[v2.50]*
-   ``ChangeFPS`` (clip, float fps, bool "linear") *[v2.50]*
-   ``ChangeFPS`` (clip1, clip2, bool "linear") *[v2.56]*
-   ``ChangeFPS`` (clip, string preset) *[v2.57]*

:doc:`ColorBars / ColorBarsHD <corefilters/colorbars>` *[rgb32] [yuy2] [yv12] [yv24]*

-   ``ColorBars`` (int "width", int "height", string "pixel_type")
-   ``ColorBarsHD`` (int "width", int "height", string "pixel_type")
    *[v2.60]*

:doc:`ColorYUV <corefilters/coloryuv>` *[yv24] [yv16] [yv12] [yv411] [y8] [yuy2] [rgb32] [rgb24]*

-   ``ColorYUV`` (clip, float "gain_y", float "off_y", float "gamma_y", float
    "cont_y", float "gain_u", float "off_u", float "gamma_u",
    float "cont_u", float "gain_v", float "off_v", float "gamma_v", float
    "cont_v", string "levels", string "opt", bool "showyuv", bool "analyze",
    bool "autowhite", bool "autogain") *[v2.50]*

:ref:`ComplementParity` *[yv24] [yv16] [yv12] [yv411] [y8] [yuy2] [rgb32]
[rgb24]*

-   ``ComplementParity`` (clip)

:doc:`Compare <corefilters/compare>` *[yv12] [yuy2] [rgb32] [rgb24]*

-   ``Compare`` (clip_filtered, clip_original, string "channels", string
    "logfile", bool "show_graph")

:doc:`ConditionalFilter / ConditionalSelect / FrameEvaluate / ScriptClip <corefilters/conditionalfilter>` /
:doc:`ConditionalReader <corefilters/conditionalreader>` *[yv12] [yuy2]*

-   ``ConditionalFilter`` (clip testclip, clip source1, clip source2,
    string expression1, string operator, string expression2, bool "show")
    *[v2.52]*
-   ``ConditionalSelect`` (clip testclip, string expression, clip
    source0, clip source1, clip source2, ..., bool "show") *[v2.60]*
-   ``FrameEvaluate`` (clip clip, script function, bool "after_frame")
    *[v2.52]*
-   ``ScriptClip`` (clip clip, string function, bool "show", bool
    "after_frame") *[v2.52]*
-   ``ConditionalReader`` (clip clip, string filename, string
    variablename, bool "show") *[v2.54]*

:doc:`ConvertAudioTo8bit / ConvertAudioTo16bit / ConvertAudioTo24bit / ConvertAudioTo32bit / ConvertAudioToFloat <corefilters/convertaudio>` *[all]*

-   ``ConvertAudioTo8bit`` (clip) *[v2.50]*
-   ``ConvertAudioTo16bit`` (clip)
-   ``ConvertAudioTo24bit`` (clip) *[v2.53]*
-   ``ConvertAudioTo32bit`` (clip) *[v2.50]*
-   ``ConvertAudioToFloat`` (clip) *[v2.50]*

:doc:`ConvertBackToYUY2 / ConvertToRGB / ConvertToRGB24 / ConvertToRGB32 / ConvertToY8 / ConvertToYUY2 / ConvertToYV12 / ConvertToYV16 / ConvertToYV24 / ConvertToYV411 <corefilters/convert>` *[yv24] [yv16] [yv12] [yv411] [y8] [yuy2]
[rgb32] [rgb24]*

-   ``ConvertToRGB`` (clip, string "matrix", bool "interlaced", string
    "ChromaInPlacement", string "chromaresample")
-   ``ConvertToRGB24`` (clip, string "matrix", bool "interlaced", string
    "ChromaInPlacement", string "chromaresample")
-   ``ConvertToRGB32`` (clip, string "matrix", bool "interlaced", string
    "ChromaInPlacement", string "chromaresample")
-   ``ConvertToYUY2`` (clip, bool "interlaced", string "matrix", string
    "ChromaInPlacement", string "chromaresample")
-   ``ConvertToBackYUY2`` (clip, string "matrix")
-   ``ConvertToY8`` (clip, string "matrix") *[v 2.60]*
-   ``ConvertToYV12`` (clip, bool "interlaced", string "matrix", string
    "ChromaInPlacement", string "chromaresample", string
    "ChromaOutPlacement") *[v 2.50]*
-   ``ConvertToYV16`` (clip, bool "interlaced", string "matrix", string
    "ChromaInPlacement", string "chromaresample") *[v 2.60]*
-   ``ConvertToYV24`` (clip, bool "interlaced", string "matrix", string
    "ChromaInPlacement", string "chromaresample") *[v 2.60]*
-   ``ConvertToYV411`` (clip, bool "interlaced", string "matrix", string
    "ChromaInPlacement", string "chromaresample") *[v 2.60]*

:ref:`ConvertFPS` *[yv24] [yv16] [yv12] [yv411] [y8] [yuy2] [rgb32] [rgb24]*

-   ``ConvertFPS`` (clip, float new_fps, int "zone", int "vbi")
-   ``ConvertFPS`` (clip, int numerator, int denominator, int "zone", int "vbi")
-   ``ConvertFPS`` (clip1, clip2, int "zone", int "vbi") *[v2.56]*
-   ``ConvertFPS`` (clip, string preset) *[v2.57]*

:doc:`ConvertToMono <corefilters/converttomono>` *[16 bit, float]*

-   ``ConvertToMono`` (clip)

:doc:`Crop / CropBottom <corefilters/crop>` *[yv24] [yv16] [yv12] [yv411] [y8] [yuy2] [rgb32] [rgb24]*

-   ``Crop`` (clip, int left, int top, int width, int height, bool "align")
-   ``Crop`` (clip, int left, int top, int -right, int -bottom, bool "align")
-   ``CropBottom`` (clip, int count,  bool "align")

.. _D:

D
-

:doc:`DelayAudio <corefilters/delayaudio>` *[all]*

-   ``DelayAudio`` (clip, float seconds)

:doc:`DeleteFrame <corefilters/deleteframe>` *[yv24] [yv16] [yv12] [yv411] [y8] [yuy2] [rgb32] [rgb24]*

-   ``DeleteFrame`` (clip, int frame [, ...])

:doc:`DirectShowSource <corefilters/directshowsource>`

-   ``DirectShowSource`` (string filename, float "fps", bool "seek", bool
    "audio", bool "video", bool "convertfps", bool "seekzero", int "timeout",
    string "pixel_type", int "framecount", string "logfile", int "logmask")

:doc:`Dissolve <corefilters/dissolve>` *[yv24] [yv16] [yv12] [yv411] [y8] [yuy2] [rgb32] [rgb24]*

-   ``Dissolve`` (clip1, clip2 [, ...], int overlap, float "fps")

:doc:`DoubleWeave <corefilters/doubleweave>` *[yv24] [yv16] [yv12] [yv411] [y8] [yuy2] [rgb32] [rgb24]*

-   ``DoubleWeave`` (clip)

:doc:`DuplicateFrame <corefilters/duplicateframe>` *[yv24] [yv16] [yv12] [yv411] [y8] [yuy2] [rgb32] [rgb24]*

-   ``DuplicateFrame`` (clip, int frame [, ...])

.. _E:

E
-

:doc:`Echo <corefilters/echo>` *[all]*

-   ``Echo`` (clip1, clip2 [, ...])

:doc:`EnsureVBRMP3Sync <corefilters/ensuresync>` *[all]*

-   ``EnsureVBRMP3Sync`` (clip)

.. _F:

F
-

:doc:`FadeIn0 / FadeIO0 / FadeOut0 / FadeIn / FadeIO / FadeOut / FadeIn2 / FadeIO2/ FadeOut2 <corefilters/fade>` *[yv24] [yv16] [yv12] [yv411] [y8] [yuy2]
[rgb32] [rgb24]*

-   ``FadeIn0`` (clip, int frames, int "color", float "fps") *[v2.56]*
-   ``FadeIn`` (clip, int frames, int "color", float "fps")
-   ``FadeIn2`` (clip, int frames, int "color", float "fps")
-   ``FadeIO0`` (clip, int frames, int "color", float "fps") *[v2.56]*
-   ``FadeIO`` (clip, int frames, int "color", float "fps")
-   ``FadeIO2`` (clip, int frames, int "color", float "fps")
-   ``FadeOut0`` (clip, int frames, int "color", float "fps") *[v2.56]*
-   ``FadeOut`` (clip, int frames, int "color", float "fps")
-   ``FadeOut2`` (clip, int frames, int "color", float "fps")

:doc:`FixBrokenChromaUpsampling <corefilters/fixbrokenchromaupsampling>` *[yuy2]*

-   ``FixBrokenChromaUpsampling`` (clip)

:doc:`FixLuminance <corefilters/fixluminance>` *[yuy2]*

-   ``FixLuminance`` (clip, int intercept, int slope)

:doc:`FlipHorizontal / FlipVertical <corefilters/flip>` *[yv24] [yv16] [yv12] [yv411] [y8] [yuy2]
[rgb32] [rgb24]*

-   ``FlipHorizontal`` (clip) *[v2.50]*
-   ``FlipVertical`` (clip)

:doc:`FreezeFrame <corefilters/freezeframe>` *[yv24] [yv16] [yv12] [yv411] [y8] [yuy2] [rgb32] [rgb24]*

-   ``FreezeFrame`` (clip, int first_frame, int last_frame, int
    source_frame)

.. _G:

G
-

:doc:`GeneralConvolution <corefilters/convolution>` *[rgb32]*

-   ``GeneralConvolution`` (clip, int "bias", string "matrix", float
    "divisor", bool "auto") *[v2.55]*

:doc:`GetChannel <corefilters/getchannel>` *[all]*

-   ``GetChannel`` (clip, int ch1 [, int ch2, ...]) *[v2.50]*
-   ``GetChannels`` (clip, int ch1 [, int ch2, ...]) *[v2.50]*

:doc:`Greyscale <corefilters/greyscale>` *[yv24] [yv16] [yv12] [yv411] [y8] [yuy2] [rgb32] [rgb24]*

-   ``Greyscale`` (clip, string "matrix")

.. _H:

H
-

:doc:`Histogram <corefilters/histogram>` *[yv12] [yuy2]*

-   ``Histogram`` (clip, string ''mode'') *[v2.54]*

.. _I:

I
-

:doc:`ImageReader / ImageSource/ ImageSourceAnim <corefilters/imagesource>` / :doc:`ImageWriter <corefilters/imagewriter>` *[yv12] [y8]
[yuy2] [rgb32] [rgb24]*

-   ``ImageReader`` (string "path", int "start", int "end", float "fps",
    bool "use_DevIL", bool "info", string "pixel_type") *[v2.52]*
-   ``ImageSource`` (string "path", int "start", int "end", float "fps",
    bool "use_DevIL", bool "info", string "pixel_type") *[v2.55]*
-   ``ImageSourceAnim`` (string "file", float "fps", bool "info", string
    "pixel_type") *[v2.60]*
-   ``ImageWriter`` (clip, string "path", int "start", int "end", string
    "type", bool "info") *[v2.52]*

:doc:`Import <corefilters/import>`

-   ``Import`` (string [, ...])

:doc:`Info <corefilters/info>` *[yv24] [yv16] [yv12] [yv411] [y8] [yuy2] [rgb32] [rgb24]*

-   ``Info`` (clip) *[v2.50]*

:doc:`Interleave <corefilters/interleave>` *[yv24] [yv16] [yv12] [yv411] [y8] [yuy2] [rgb32] [rgb24]*

-   ``Interleave`` (clip1, clip2 [, ...])

:doc:`Invert <corefilters/invert>` *[yv12, v2.55] [yuy2, v2.55] [rgb32] [rgb24, v2.55]*

-   ``Invert`` (clip, string "channels") *[v2.53]*

.. _K:

K
-

:doc:`KillAudio <corefilters/killaudio>` *[all]*

-   ``KillAudio`` (clip)

:doc:`KillVideo <corefilters/killaudio>` *[all]*

-   ``KillVideo`` (clip) *[v2.57]*

.. _L:

L
-

:doc:`Layer / Mask / ResetMask / ColorKeyMask <corefilters/layer>` *[RGB32]*

-   ``Layer`` (clip, layer_clip, string "op", int "level", int "x", int
    "y", int "threshold", bool "use_chroma") *[yuy2] [rgb32]*
-   ``Mask`` (clip, mask_clip) *[rgb32]*
-   ``ResetMask`` (clip) *[rgb32]*
-   ``ColorKeyMask`` (clip, int "color", int "tolB" [, int "tolG", int
    "tolR"]) *[rgb32]*

:doc:`Letterbox <corefilters/letterbox>` *[yv24] [yv16] [yv12] [yv411] [y8] [yuy2] [rgb32] [rgb24]*

-   ``Letterbox`` (clip, int top, int bottom [, int left, int right])

:doc:`Levels <corefilters/levels>` *[yv24] [yv16] [yv12] [yv411] [y8] [yuy2] [rgb32] [rgb24]*

-   ``Levels`` (clip, int input_low, float gamma, int input_high, int
    output_low, int output_high, bool "coring", bool "dither")

:doc:`Limiter <corefilters/limiter>` *[yv24] [yv16] [yv12] [yv411] [y8] [yuy2]*

-   ``Limiter`` (clip, int "min_luma", int "max_luma", int "min_chroma",
    int "max_chroma" [, string show])* [v2.50]*

:doc:`Loop <corefilters/loop>` *[yv24] [yv16] [yv12] [yv411] [y8] [yuy2] [rgb32] [rgb24]*

-   ``Loop`` (clip, int "times", int "start", int "end")

.. _M:

M
-

:doc:`MaskHS <corefilters/maskhs>` *[yv24] [yv16] [yv12] [yv411] [yuy2] [2.60]*

-   ``MaskHS`` (clip, float "startHue", float "endHue", float "maxSat",
    float "minSat", bool "coring")

:doc:`MergeARGB / MergeRGB <corefilters/mergergb>` *[yv24] [yv16] [yv12] [yv411] [y8] [yuy2] [rgb32]
[rgb24] [v2.56]*

-   ``MergeARGB`` (clipA, clipR, clipG, clipB)
-   ``MergeRGB`` (clipR, clipG, clipB [, string "pixel_type"])

:doc:`MergeChannels <corefilters/mergechannels>` *[all]*

-   ``MergeChannels`` (clip1, clip2 [, ...])* [v2.50]*

:doc:`Merge / MergeChroma / MergeLuma <corefilters/merge>` *[yv24] [yv16] [yv12] [yv411] [y8] [yuy2]*

-   ``Merge`` (clip1, clip2, float "weight") *[yv12, yuy2, rgb32, rgb24] [v2.56]*
-   ``MergeChroma`` (clip1, clip2, float "weight")
-   ``MergeLuma`` (clip1, clip2, float "weight")

:doc:`MessageClip <corefilters/message>` *[rgb32]*

-   ``MessageClip`` (string message, int "width", int "height", bool
    "shrink", int "text_color", int "halo_color", int "bg_color")

:doc:`MixAudio <corefilters/mixaudio>` *[16 bit, float]*

-   ``MixAudio`` (clip1, clip 2, float clip1_factor, float
    "clip2_factor")

.. _N:

N
-

:doc:`Normalize <corefilters/normalize>` *[16 bit, float]*

-   ``Normalize`` (clip, float "volume", bool "show")

.. _O:

O
-

:doc:`Overlay <corefilters/overlay>` *[yv24] [yv16] [yv12] [yv411] [y8] [yuy2] [rgb32] [rgb24]*

-   ``Overlay`` (clip, clip overlay, int "x", int "y", clip "mask", float
    "opacity", string "mode", bool "greymask", string "output",
    bool "ignore_conditional", bool "pc_range") *[v2.54]*

.. _P:

P
-

:doc:`PeculiarBlend <corefilters/peculiar>` *[yuy2]*

-   ``PeculiarBlend`` (clip, int cutoff)

:doc:`Preroll <corefilters/preroll>` *[all]*

-   ``Preroll`` (clip, int "video", float "audio")

:doc:`Pulldown <corefilters/pulldown>` *[yv24] [yv16] [yv12] [yv411] [y8] [yuy2] [rgb32] [rgb24]*

-   ``Pulldown`` (clip, int a , int b)

.. _R:

R
-

:doc:`RGBAdjust <corefilters/adjust>` *[rgb32] [rgb24]*

-   ``RGBAdjust`` (clip, float "r", float "g", float "b", float "a",
    float "rb", float "gb", float "bb", float "ab", float "rg", float "gg",
    float "bg", float "ag", bool "analyze", bool "dither")

:doc:`ReduceBy2 / HorizontalReduceBy2 / VerticalReduceBy2 <corefilters/reduceby2>` *[yv24] [yv16] [yv12]
[yv411] [y8] [yuy2] [rgb32] [rgb24]*

-   ``HorizontalReduceBy2`` (clip)
-   ``VerticalReduceBy2`` (clip)
-   ``ReduceBy2`` (clip)

:doc:`ResampleAudio <corefilters/resampleaudio>` *[16 bit, float]*

-   ``ResampleAudio`` (clip, int new_rate_numberator [, int
    new_rate_denominator])

:doc:`BilinearResize / BicubicResize / BlackmanResize / GaussResize / LanczosResize / Lanczos4Resize / PointResize / SincResize / Spline16Resize / Spline36Resize / Spline64Resize <corefilters/resize>` *[yv24] [yv16] [yv12] [yv411] [y8] [yuy2]
[rgb32] [rgb24]*

-   ``BilinearResize`` (clip, int target_width, int target_height, float
    "src_left", float "src_top", float "src_width", float "src_height")
-   ``BicubicResize`` (clip, int target_width, int target_height, float
    "b=1./3.", float "c=1./3.", float "src_left", float "src_top", float
    "src_width", float "src_height")
-   ``BlackmanResize`` (clip, int target_width, int target_height, float
    "src_left", float "src_top", float "src_width", float "src_height", int
    "taps=4") *[v2.58]*
-   ``GaussResize`` (clip, int target_width, int target_height, float
    "src_left", float "src_top", float "src_width", float "src_height", float
    "p=30.0") *[v2.56]*
-   ``LanczosResize`` (clip, int target_width, int target_height, float
    "src_left", float "src_top", float "src_width", float "src_height", int
    "taps=3")
-   ``Lanczos4Resize`` (clip, int target_width, int target_height, float
    "src_left", float "src_top", float "src_width", float "src_height")
    *[v2.55]*
-   ``PointResize`` (clip, int target_width, int target_height, float
    "src_left", float "src_top", float "src_width", float "src_height")
-   ``SincResize`` (clip, int target_width, int target_height, float
    "src_left", float "src_top", float "src_width", float "src_height", int
    "taps=4") *[v2.6]*
-   ``Spline16Resize`` (clip, int target_width, int target_height, float
    "src_left", float "src_top", float "src_width", float "src_height")
    *[v2.56]*
-   ``Spline36Resize`` (clip, int target_width, int target_height, float
    "src_left", float "src_top", float "src_width", float "src_height")
    *[v2.56]*
-   ``Spline64Resize`` (clip, int target_width, int target_height, float
    "src_left", float "src_top", float "src_width", float "src_height")
    *[v2.58]*
-   all resizers: ``xxxResize`` (clip, int target_width, int
    target_height, float "src_left", float "src_top", float -"src_right",
    float -"src_bottom") *[v2.56]*

:doc:`Reverse <corefilters/reverse>` *[yv24] [yv16] [yv12] [yv411] [y8] [yuy2] [rgb32] [rgb24]*

-   ``Reverse`` (clip)

.. _S:

S
-

:doc:`SegmentedAVISource / SegmentedDirectShowSource <corefilters/segmentedsource>`

-   ``SegmentedAVISource`` (string base_filename [, ...], bool "audio")
-   ``SegmentedDirectShowSource`` (string base_filename [, ...]  [, fps])

:doc:`SelectEven / SelectOdd <corefilters/select>` *[yv24] [yv16] [yv12] [yv411] [y8] [yuy2] [rgb32]
[rgb24]*

-   ``SelectEven`` (clip)
-   ``SelectOdd`` (clip)

:doc:`SelectEvery <corefilters/selectevery>` *[yv24] [yv16] [yv12] [yv411] [y8] [yuy2] [rgb32] [rgb24]*

-   ``SelectEvery`` (clip, int step_size, int offset1 [, int offset2 [,
    ...]])

:doc:`SelectRangeEvery <corefilters/selectrangeevery>` *[yv24] [yv16] [yv12] [yv411] [y8] [yuy2] [rgb32]
[rgb24]*

-   ``SelectRangeEvery`` (clip, int every, int length, int "offset", bool
    "audio'') *[v2.50]*

:doc:`SeparateFields <corefilters/separatefields>` *[yv24] [yv16] [yv12] [yv411] [y8] [yuy2] [rgb32] [rgb24]*

-   ``SeparateFields`` (clip)
-   ``SeparateColumns`` (clip, int interval) *[v2.60]*
-   ``SeparateRows`` (clip, int interval) *[v2.60]*

:doc:`ShowAlpha <corefilters/showalpha>` *[rgb32]*

-   ``ShowAlpha`` (clip, string "pixel_type") *[v2.54]*

:doc:`ShowRed, ShowGreen, ShowBlue <corefilters/showalpha>` *[rgb24] [rgb32] [v2.56]*

-   ``ShowRed`` (clip, string "pixel_type")
-   ``ShowGreen`` (clip, string "pixel_type")
-   ``ShowBlue`` (clip, string "pixel_type")

:doc:`ShowFiveVersions <corefilters/showfive>` *[yv12] [yuy2] [rgb32] [rgb24]*

-   ``ShowFiveVersions`` (clip1, clip2, clip3, clip4, clip5)

:doc:`ShowFrameNumber / ShowSMPTE / ShowTime <corefilters/showframes>` *[yv12] [yuy2] [rgb32] [rgb24]*

-   ``ShowFrameNumber`` (clip, bool "scroll", int "offset", float "x",
    float "y", string "font", int "size", int "text_color", int "halo_color",
    float "font_width", float "font_angle")
-   ``ShowSMPTE`` (clip, float "fps", string "offset", int "offset_f",
    float "x", float "y", string "font", int "size", int "text_color", int
    "halo_color", float "font_width", float "font_angle")
-   ``ShowTime`` (clip, int "offset_f", float "x", float "y", string
    "font", int "size", int "text_color", int "halo_color", float
    "font_width", float "font_angle")* [v2.58]*

:doc:`SkewRows <corefilters/skewrows>` *[y8] [yuy2] [rgb32] [rgb24]*

-   ``SkewRows`` (clip, int skew) *[v2.60]*

:doc:`SoundOut <corefilters/soundout>` *[all] [v2.60]*

-   ``SoundOut`` (string output, string filename, bool "showprogress",
    string overwritefile, bool "autoclose", bool "silentblock", bool
    "addvideo", special parameters)

-   -   (the :doc:`special parameters <corefilters/soundout>` are output dependent and they are explained in the
        documentation itself)

:doc:`SpatialSoften / TemporalSoften <corefilters/soften>` *[yv12] [yuy2] [rgb32, v2.56]*

-   ``SpatialSoften`` (clip, int radius, int luma_threshold, int
    chroma_threshold)
-   ``TemporalSoften`` (clip, int radius, int luma_threshold, int
    chroma_threshold, int "scenechange", int "mode")* [v2.50]*

:doc:`AlignedSplice / UnalignedSplice <corefilters/splice>` *[yv24] [yv16] [yv12] [yv411] [y8] [yuy2]
[rgb32] [rgb24]*

-   ``AlignedSplice`` (clip1, clip2 [, ...])
-   ``UnAlignedSplice`` (clip1, clip2 [, ...])

:doc:`SSRC <corefilters/ssrc>` *[float]*

-   ``SSRC`` (clip, int samplerate, bool "fast") *[v2.54]*

:doc:`StackHorizontal / StackVertical <corefilters/stack>` *[yv24] [yv16] [yv12] [yv411] [y8] [yuy2]
[rgb32] [rgb24]*

-   ``StackHorizontal`` (clip1, clip2 [, ...])
-   ``StackVertical`` (clip1, clip2 [, ...])

:doc:`Subtitle <corefilters/subtitle>` *[yv24] [yv16] [yv12] [yv411] [y8] [yuy2] [rgb32] [rgb24]*

-   ``Subtitle`` (clip, string text, float "x", float "y", int
    "first_frame", int "last_frame", string "font", int "size", int
    "text_color", int "halo_color", int "lsp", float "font_width", float
    "font_angle", bool "interlaced")
-   ``Subtitle`` (clip, string "text")

:doc:`Subtract <corefilters/subtract>` *[yv24] [yv16] [yv12] [yv411] [y8] [yuy2] [rgb32] [rgb24]*

-   ``Subtract`` (clip1, clip2)

:doc:`SuperEQ <corefilters/supereq>` *[float]*

-   ``SuperEQ`` (clip, string filename) *[v2.54]*
-   ``SuperEQ`` (clip, float band1 [, float band1, ..., float band18])
    *[v2.60]*

:doc:`SwapUV / UToY / VToY / YToUV <corefilters/swap>` *[yv24] [yv16] [yv12] [yv411] [y8] [yuy2]*

-   ``SwapUV`` (clip) *[v2.50]*
-   ``UToY`` (clip) *[v2.50]*
-   ``UToY8`` (clip) *[v2.60]*
-   ``VToY`` (clip) *[v2.50]*
-   ``VToY8`` (clip) *[v2.60]*
-   ``YToUV`` (clip clipU, clip clipV [, clip clipY]) *[v2.50, v2.51]*

:doc:`SwapFields <corefilters/swapfields>` *[yv24] [yv16] [yv12] [yv411] [y8] [yuy2] [rgb32] [rgb24]*

-   ``SwapFields`` (clip)

.. _T:

T
-

:doc:`TCPDeliver <corefilters/tcpdeliver>`

-   ``TCPServer`` (clip, int "port") *[v2.55]*
-   ``TCPSource`` (string hostname, int "port", string "compression")
    *[v2.55]*

:doc:`TimeStretch <corefilters/timestretch>` *[float]*

-   ``TimeStretch`` (clip, float "tempo", float "rate", float "pitch",
    int "sequence", int "seekwindow", int "overlap", bool "quickseek", int
    "aa") *[v2.57]*

:doc:`Tone <corefilters/tone>` *[float]*

-   ``Tone`` (float "length", float "frequency", int "samplerate", int
    "channels", string "type", float "level") *[v2.54]*

:doc:`Trim <corefilters/trim>` *[yv24] [yv16] [yv12] [yv411] [y8] [yuy2] [rgb32] [rgb24]*

-   ``Trim`` (clip, int first_frame, int last_frame [, bool "pad"])
    *[v2.56]*
-   ``Trim`` (clip, int first_frame, int -num_frames [, bool "pad"])
    *[v2.56]*
-   ``Trim`` (clip, int start_time, int "end" [, bool "pad"]) *[v2.60]*
-   ``Trim`` (clip, int start_time, int "length" [, bool "pad"])
    *[v2.60]*

:doc:`TurnLeft / TurnRight / Turn180 <corefilters/turn>` *[yv24] [yv16] [yv12] [yv411] [y8] [yuy2]
[rgb32] [rgb24]*

-   ``TurnLeft`` (clip) *[v2.51]*
-   ``TurnRight`` (clip) *[v2.51]*
-   ``Turn180`` (clip) *[v2.55]*

:doc:`Tweak <corefilters/tweak>` *[yv24] [yv16] [yv12] [yv411] [y8] [yuy2]*

-   ``Tweak`` (clip, float "hue", float "sat", float "bright", float
    "cont", bool "coring", bool "sse", float "startHue", float "endHue",
    float "maxSat", float "minSat", float "interp", bool "dither")

.. _V:

V
-

:doc:`Version <corefilters/version>` *[rgb24]*

-   ``Version`` ()

.. _W:

W
-

:doc:`Weave <corefilters/weave>` *[yv24] [yv16] [yv12] [yv411] [y8] [yuy2] [rgb32] [rgb24]*

-   ``Weave`` (clip)
-   ``WeaveColumns`` (clip, int period) *[v2.60]*
-   ``WeaveRows`` (clip, int period) *[v2.60]*

:doc:`WriteFile / WriteFileIf / WriteFileStart / WriteFileEnd <corefilters/write>` *[yv12] [yuy2]
[rgb32] [rgb24]*

-   ``WriteFile`` (clip, string filename, *string expression1 [, string
    expression2 [, ...]], bool "append", bool "flush"*)
-   ``WriteFileIf`` (clip, string filename, *string expression1 [, string
    expression2 [, ...]], bool "append", bool "flush"*)
-   ``WriteFileStart`` (clip, string filename, *string expression1 [,
    string expression2 [, ...]], bool "append"*)
-   ``WriteFileEnd`` (clip, string filename, *string expression1 [,
    string expression2 [, ...]], bool "append"*)

[ :ref:`A` :ref:`B` :ref:`C` :ref:`D` :ref:`E` :ref:`F` :ref:`G` :ref:`H` :ref:`I` [J] :ref:`K`
:ref:`L` :ref:`M` :ref:`N` :ref:`O` :ref:`P` [Q] :ref:`R` :ref:`S` :ref:`T` [U] :ref:`V` :ref:`W`
[X] [Y] [Z] ]

$Date: 2026/11/14 14:15:00 $
