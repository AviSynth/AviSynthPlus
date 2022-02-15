
BlankClip / Blackness
=====================

The **BlankClip** filter produces a solid color, silent video clip of the
specified length (in frames). The *clip* passed as an argument is used as a
*template* for frame rate, image size, and so on, but you can specify all clip
properties without having to provide a template. Without any arguments specified, 
**BlankClip** will produce a pitch-black 10 seconds clip (RGB32), 640x480, 24 
fps, 16 bit 44100 Hz mono.

When supplying a template, **BlankClip** returns a clip with 
:doc:`properties <../syntax/syntax_clip_properties>` copied from that template. 
If the template is audio-only, you get a blank audio-only clip, and if it's 
video-only you get a blank video-only clip. If you start to add parameters that 
force a video track (i.e. width, height or pixel_type) or audio track 
(i.e. audio_rate, channels or sample_type), the remaining parameters 
for that track will be the defaults.

**Blackness** is an alias for **BlankClip**. The parameters are the same, minus
the *colors* parameter.

Syntax and Parameters
----------------------

::

    BlankClip (clip clip, int "length", int "width", int "height",
               string "pixel_type", float "fps", int "fps_denominator", int "audio_rate",
               int "channels", string "sample_type", int "color", int "color_yuv", float+ "colors")

    BlankClip (clip clip, int "length", int "width", int "height",
               string "pixel_type", float "fps", int "fps_denominator", int "audio_rate",
               bool "stereo", bool "sixteen_bit", int "color", int "color_yuv", float+ "colors")

    Blackness ( ...same parameters as BlankClip )

.. describe:: clip
    
    If present, the resulting clip will have the clip-properties of the
    template, except for the properties you define explicitly.

.. describe:: length 
    
    Length of the resulting clip (in frames).
    
    Default: 240

.. describe:: width, height 
    
    Width and height of the resulting clip.
    
    Default: 640, 480

.. describe:: pixel_type 
    
    Pixel type of the resulting clip. Valid color formats are listed in the 
    following table.

    +--------+---------+------------+-------------+------------+--------+
    | Bits   | RGB(A)  | YUV(A)444  | YUV(A)422   | YUV(A)420  | Y only |
    +========+=========+============+=============+============+========+
    | **8**  | RGB24   | YV24       | YV16 / YUY2 | YV12       | Y8     |
    |        |         |            |             |            |        |
    |        | RGBP    | YUV444     | YUV422      | YUV420     |        |
    |        |         |            |             |            |        |
    |        | RGBP8   | YUV444P8   | YUV422P8    | YUV420P8   |        |
    +--------+---------+------------+-------------+------------+--------+
    | **10** | RGBP10  | YUV444P10  | YUV422P10   | YUV420P10  | Y10    |
    +--------+---------+------------+-------------+------------+--------+
    | **12** | RGBP12  | YUV444P12  | YUV422P12   | YUV420P12  | Y12    |
    +--------+---------+------------+-------------+------------+--------+
    | **14** | RGBP14  | YUV444P14  | YUV422P14   | YUV420P14  | Y14    |
    +--------+---------+------------+-------------+------------+--------+
    | **16** | RGB48   | YUV444P16  | YUV422P16   | YUV420P16  | Y16    |
    |        |         |            |             |            |        |
    |        | RGBP16  |            |             |            |        |
    +--------+---------+------------+-------------+------------+--------+
    | **32** | RGBPS   | YUV444PS   | YUV422PS    | YUV420PS   | Y32    |
    +--------+---------+------------+-------------+------------+--------+
    | `With an alpha channel`                                           |
    +--------+---------+------------+-------------+------------+--------+
    | **8**  | RGB32   |            |             |            |        |
    |        |         |            |             |            |        |
    |        | RGBAP   | YUVA444    | YUVA422     | YUVA420    |        |
    |        |         |            |             |            |        |
    |        | RGBAP8  | YUVA444P8  | YUVA422P8   | YUVA420P8  |        |
    +--------+---------+------------+-------------+------------+--------+
    | **10** | RGBAP10 | YUVA444P10 | YUVA422P10  | YUVA420P10 |        |
    +--------+---------+------------+-------------+------------+--------+
    | **12** | RGBAP12 | YUVA444P12 | YUVA422P12  | YUVA420P12 |        |
    +--------+---------+------------+-------------+------------+--------+
    | **14** | RGBAP14 | YUVA444P14 | YUVA422P14  | YUVA420P14 |        |
    +--------+---------+------------+-------------+------------+--------+
    | **16** | RGB64   | YUVA444P16 | YUVA422P16  | YUVA420P16 |        |
    |        |         |            |             |            |        |
    |        | RGBAP16 |            |             |            |        |
    +--------+---------+------------+-------------+------------+--------+
    | **32** | RGBAPS  | YUVA444PS  | YUVA422PS   | YUVA420PS  |        |
    +--------+---------+------------+-------------+------------+--------+
    | **Note**: 8-bit color formats (``YV411, YUV411, YUV411P8``) were  | 
    | omitted from the table.                                           |
    +--------+---------+------------+-------------+------------+--------+

    Default: "RGB32"

.. describe:: fps
    
    The framerate of the resulting clip.
    
    Default: 24

.. describe:: fps_denominator
    
    | You can use this option if "fps" is not accurate enough. 
    | For example: ``fps = 30000, fps_denominator = 1001`` (ratio = 29.97) or 
      ``fps = 24000, fps_denominator = 1001`` (ratio = 23.976).
    
    *Note* – if ``fps_denominator`` is given (even if it is "1"), ``fps`` is 
    **rounded to the nearest integer**. 
    
    Default: 1

.. describe:: audio_rate
    
    | Sample rate of the (silent) audio.
    | *Note* – ``BlankClip(audio_rate=0)`` produces the same result as 
      ``BlankClip.KillAudio()``. 
    
    Default: 44100 

.. describe:: channels
    
    Specifies the number of audio channels of silent audio added to the blank clip.
    
    Default: 1

.. describe:: stereo
    
    | **Deprecated!** Use should the ``channels`` parameter instead.
    | If true, the (silent) audio is in stereo: ``channels=2``.
    
    Default: false

.. describe:: sample_type
    
    Specifies the audio sample type of the resulting clip. It can be "8bit", 
    "16bit", "24bit", "32bit" or "float".
    
    Default: "16bit"

.. describe:: sixteen_bit
    
    | **Deprecated!** Use the ``sample_type`` parameter instead.
    | True returns 16-bit audio, *false* returns 32-bit float.
    
    Default: true

.. describe:: color
    
    | Specifies the color of the clip. Color is specified as an RGB value in 
      either hexadecimal or decimal notation.
    | Hex numbers must be preceded with a $. See the
      :doc:`colors <../syntax/syntax_colors>` page for more information on 
      specifying colors.
    
    * For YUV clips, colors are converted from full range (0–255) to limited 
      range (16–235) `Rec.601`_.
    * Use ``color_yuv`` to specify full range YUV values or a color with a 
      different matrix.
    
    Default: $000000

.. describe:: color_yuv 
    
    Specifies the color of the clip using YUV values. ``pixel_type`` must be 
    set to one of the YUV formats or a YUV reference clip provided; otherwise 
    an error is raised. See the :ref:`YUV colors <yuv-colors>` for more 
    information.
    
.. describe:: colors
    
    Specify the color of the clip using an array. Use this to pass exact, 
    unscaled color values.
    
    Color order: Y,U,V,A or R,G,B,A


Examples
---------

* Produces a black clip (3000 frames, width 720, height 576, framerate 25),
  with a silent audio track (16-bit 44.1 kHz stereo):

 .. code-block:: c++

    BlankClip(length=3000, width=720, height=576, fps=25, channels=2, color=$000000)

* Produces a black clip (3000 frames) with the remaining clip properties of the 
  reference clip:

 .. code-block:: c++

    video = AviSource("E:\pdwork\DO-Heaven.AVI")
    BlankClip(video, length=3000, color=$000000)

* Adds a silent audio stream (with a samplerate of 48 kHz) to a video clip:

 .. code-block:: c++

    video = AviSource("E:\pdwork\DO-Heaven.AVI")
    audio = BlankClip(video, audio_rate=48000)
    AudioDub(video, audio)
    
* Create an RGB64 clip and specify the colors using an array:

 .. code-block:: c++

    BlankClip(pixel_type="RGB64", colors=[64000,32768,1231,65535])

* Create a full range black YUV clip using the ``color_yuv`` parameter:

 .. code-block:: c++

    BlankClip(pixel_type="YUV420P8", color_yuv=$008080)

* Create a full range white YUV clip using the ``colors`` parameter:

 .. code-block:: c++

    BlankClip(pixel_type="YUV420P8", colors=[255,128,128])


Changelog
----------

+-----------------+--------------------------------------------------------------+
| Version         | Changes                                                      |
+=================+==============================================================+
| AviSynth+ r2487 || BlankClip: new ``colors`` parameter.                        |
|                 || Added support for the remaining 10-12-14-bit color formats. |
+-----------------+--------------------------------------------------------------+
| AviSynth+ r2290 | Added support for RGB48/64 and Planar RGB(A)/YUV(A) color    |
|                 | formats (16-bit and Float).                                  |
+-----------------+--------------------------------------------------------------+
| AviSynth 2.6.0  || Added pixel_type="YV24"/"YV16"/"YV411"/"Y8".                |
|                 || Supply useful defaults for new Audio/Video when using a     |
|                 |  Video/Audio only template clip.                             |
+-----------------+--------------------------------------------------------------+
| AviSynth 2.5.8  | Added ``channels`` and ``sample_type`` parameters.           |
+-----------------+--------------------------------------------------------------+
| AviSynth 2.5.5  | Added ``color_yuv`` parameter.                               |
+-----------------+--------------------------------------------------------------+

$Date: 2022/02/14 20:09:50 $

.. _Rec.601:
    https://en.wikipedia.org/wiki/Rec._601
