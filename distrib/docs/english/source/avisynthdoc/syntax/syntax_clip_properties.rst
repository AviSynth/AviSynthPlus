
AviSynth Clip properties
========================

You can access clip properties in AVS scripts. For example, if the variable
*clip* holds a video clip, then *clip.height* is its height in pixels,
*clip.framecount* is its length in frames, and so on. Clip properties can be
manipulated just like :doc:`script variables <syntax_script_variables>` (see the :doc:`AviSynth Syntax <syntax_ref>` for
more), except that they cannot be l-values in C-terminology.

The full list of properties:

-   Width (clip)

Returns the width of the clip in pixels (type: int).

-   Height (clip)

Returns the height of the clip in pixels (type: int).

-   FrameCount (clip)

Returns the number of frames of the clip (type: int).

-   FrameRate (clip)

Returns the number of frames per seconds of the clip (type: float). The
framerate is internally stored as a ratio though and more about it can be
read :doc:`here[1] <../corefilters/fps>`.

-   FrameRateNumerator (clip) (v2.55)

Returns the numerator of the number of frames per seconds of the clip
(type: int).

-   FrameRateDenominator (clip) (v2.55)

Returns the denominator of the number of frames per seconds of the clip
(type: int).

-   AudioRate (clip)

Returns the sample rate of the audio of the clip (type: int).

-   AudioLength (clip) (v2.51, v3.7.4)

Returns the number of samples of the audio of the clip.
(type: int, since 3.7.4 exact 64 bit number).
On pre-3.7.4 be aware of possible overflow on very
long clips ( 2^31 samples limit - 32 bit integer).

-   AudioLengthLo (clip [, int]) (v2.60)

Returns the number of samples of the audio of the clip modulo int. int is
1,000,000,000 by default (type: int).

-   AudioLengthHi (clip [, int]) (v2.60)

Returns the number of samples of the audio of the clip divided by int
(truncated to nearest integer). int is 1,000,000,000 by default (type: int).

-   AudioLengthS (clip) (v2.60)

Returns a string formated with the total number of samples of the audio of
the clip (type: string).

-   AudioLengthF (clip) (v2.55, v3.7.4)

Returns the number of samples of the audio of the clip.
(type: float, since 3.7.4 a 64 bit double, still not exact and may overflow, 
use Audiolength) 

-   AudioDuration (clip) (v2.60)

Returns the duration in seconds of the audio of the clip (type: float).

-   AudioChannels (clip)

Returns the number of audio channels of the clip (type: int).

-   AudioBits (clip)

Returns the audio bit depth of the clip (type: int).

-   IsAudioFloat (clip) (v2.55)

Returns true if the audio format of the clip is float (type: bool).

-   IsAudioInt (clip) (v2.55)

Returns true if the audio format of the clip is an integer type (type: bool).

-   IsRGB (clip)

Returns true if the clip is `RGB`_, false otherwise (type: bool).

-   IsPlanarRGB (clip)

Returns true if the clip is planar RGB, false otherwise (type: bool).

-   IsPlanarRGBA (clip)

Returns true if the clip is planar RGBA, false otherwise (type: bool).

-   IsRGB24 (clip) (v2.07)

Returns true if the clip is `RGB24`_, false otherwise (type: bool).

-   IsRGB32 (clip) (v2.07)

Returns true if the clip is `RGB32`_, false otherwise (type: bool).

-   IsRGB48 (clip) (Avisynth+)

Returns true if the clip is RGB48, false otherwise (type: bool).

-   IsRGB64 (clip) (Avisynth+)

Returns true if the clip is RGB64, false otherwise (type: bool).

-   IsYUV (clip) (v2.54)

Returns true if the clip is `YUV`_, false otherwise (type: bool).

-   IsYUVA (clip) (Avisynth+)

Returns true if the clip is YUVA (YUV + alpha channel), false 
otherwise (type: bool).

-   IsY8 (clip) (2.60)

Returns true if the clip is `Y8`_, false otherwise (type: bool).

-   IsY (clip) (Avisynth+)

Returns true if the clip is Y (like Y8, but bit depth independent)
grey scale, false otherwise (type: bool).

-   IsYUY2 (clip)

Returns true if the clip is `YUY2`_, false otherwise (type: bool).

-   IsYV12 (clip) (v2.52)

Returns true if the clip is `YV12`_, false otherwise (type: bool).

-   Is420 (clip) (Avisynth+)

Returns true if the clip is 4:2:0 (like IsYV12, but bit depth 
independently), false otherwise (type: bool).

-   IsYV16 (clip) (v2.60)

Returns true if the clip is `YV16`_, false otherwise (type: bool).

-   Is422 (clip) (Avisynth+)

Returns true if the clip is 4:2:2 (like Is YV16, but bit depth
independently), false otherwise (type: bool).

-   IsYV24 (clip) (v2.60)

Returns true if the clip is `YV24`_, false otherwise (type: bool).

-   Is444 (clip) (Avisynth+)

Returns true if the clip is 4:4:4 (like IsYV24, but bit depth
independently), false otherwise (type: bool).

-   IsYV411 (clip) (v2.60)

Returns true if the clip is `YV411`_, false otherwise (type: bool).

-   PixelType (clip) (v2.60)

Returns the name of the pixel format (type: string).

-   IsFieldBased (clip)

Returns true if the clip is field-based (type: bool). What this means is
explained :doc:`here[2] <../advancedtopics/interlaced_fieldbased>`.

-   IsFrameBased (clip)

Returns true if the clip is frame-based (type: bool). What this means is
explained :doc:`here[2] <../advancedtopics/interlaced_fieldbased>`.

-   IsPlanar (clip) (v2.52)

Returns true if the clip is `planar`_, false otherwise (type: bool).

-   IsInterleaved (clip) (v2.52)

Returns true if the clip color format is Interleaved, false otherwise (type:
bool).

-   GetParity (clip, int n)

Returns true if frame n (default 0) is top field of field-based clip, or it
is full frame with top field first of frame-based clip (type: bool).

-   HasAudio (clip) (v2.56)

Returns true if the clip has audio, false otherwise (type: bool).

-   HasVideo (clip) (v2.56)

Returns true if the clip has video, false otherwise (type: bool).

-   HasVideo (clip) (v2.56)

Returns true if the clip has video, false otherwise (type: bool).

-   ComponentSize (clip) (Avisynth+)

Returns 1 for 8 bit, 2 for 10-16 bits, 4 for 32 bit float pixel 
formats (type: int).

-   BitsPerComponent (clip) (Avisynth+)

Returns 8, 10, 12, 14, 16 or 32: the actual bit depth of the video format
(type: int).

-   NumComponents (clip) (Avisynth+)

Returns 1 for greyscale, 3 for alpha-less RGB and YUV formats, 4 for formats
with alpha channel: YUVA, planar RGBA, RGB32 or RGB64 (type: int).

-   HasAlpha (clip) (Avisynth+)

Returns true if video format has alpha channel (type: bool).

-   IsPackedRGB (clip) (Avisynth+)

Returns true if video format is one of the packed (not planar) RGB formats:
RGB24, RGB32, RGB48 or RGB64 (type: bool).

-   IsChannelMaskKnown (clip) (Avisynth+ v3.7.3)

Returns true if the clip has audio channel mask defined (type: bool).

-   GetChannelMask (clip) (Avisynth+ v3.7.3)

Returns the channel mask (a 32 bit number), if audio channel mask is 
defined, 0 otherwise. Channel mask bits - speaker bits - follow
WAVE_FORMAT_EXTENSIBLE dwChannelMask speaker position definitions, see
https://learn.microsoft.com/en-us/windows/win32/api/mmreg/ns-mmreg-waveformatextensible
(type: int)

-   Is440 (clip) (v3.7.6)

Returns true if the clip is 4:4:0, false otherwise (type: bool).

-   Is410 (clip) (v3.7.6)

Returns true if the clip is 4:1:0, false otherwise (type: bool).

-   Is411 (clip) (v3.7.6)

Returns true if the clip is 4:1:1 (like IsYV411, but bit depth
independently), false otherwise (type: bool).

-   IsYA (clip) (v3.7.6)

Returns true if the clip is 4:0:0:α, false otherwise (type: bool).



--------

Back to :doc:`AviSynth Syntax <syntax_ref>`.

$Date: 2026-08-25 22:50:00-05:00 $

.. _rgb: http://avisynth.nl/index.php/RGB
.. _rgb24: http://avisynth.nl/index.php/RGB24
.. _rgb32: http://avisynth.nl/index.php/RGB32
.. _yuv: http://avisynth.nl/index.php/YUV
.. _y8: http://avisynth.nl/index.php/Y8
.. _yuy2: http://avisynth.nl/index.php/YUY2
.. _yv12: http://avisynth.nl/index.php/YV12
.. _yv16: http://avisynth.nl/index.php/YV16
.. _yv24: http://avisynth.nl/index.php/YV24
.. _yv411: http://avisynth.nl/index.php/YV411
.. _planar: http://avisynth.nl/index.php/Planar
