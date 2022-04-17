============
Fade Filters
============

| **FadeIn** blends progressively *from* ``color`` at the beginning.
| **FadeOut** blends progressively *to* ``color`` at the end.
| **FadeIO** blends progressively *from/to* ``color`` at both ends.

* The sound track (if present) also fades linearly to and/or from silence.

* The fading affects only the first/last ``num_frames`` frames of the video.

* The first/last frame of the video becomes almost-but-not-quite color. |br|
  An additional ``color`` frame is added at the start/end, thus increasing the
  total frame count by one (or for **FadeIO**, by two).


| **FadeIn0** / **FadeOut0** / **FadeIO0** do not add the extra ``color`` frames,
  leaving the end(s) almost-but-not-quite color.
| **FadeIn2** / **FadeOut2** / **FadeIO2** add two extra ``color`` frames at the
  start/end instead of one.


Syntax and Parameters
----------------------

::

    FadeIn (clip clip, int num_frames, int "color", float "fps", int "color_yuv", float+ "colors")
    FadeIO (clip clip, int num_frames, int "color", float "fps", int "color_yuv", float+ "colors")
    FadeOut (clip clip, int num_frames, int "color", float "fps", int "color_yuv", float+ "colors")

    FadeIn0 (clip clip, int num_frames, int "color", float "fps", int "color_yuv", float+ "colors")
    FadeIO0 (clip clip, int num_frames, int "color", float "fps", int "color_yuv", float+ "colors")
    FadeOut0 (clip clip, int num_frames, int "color", float "fps", int "color_yuv", float+ "colors")

    FadeIn2 (clip clip, int num_frames, int "color", float "fps", int "color_yuv", float+ "colors")
    FadeIO2 (clip clip, int num_frames, int "color", float "fps", int "color_yuv", float+ "colors")
    FadeOut2 (clip clip, int num_frames, int "color", float "fps", int "color_yuv", float+ "colors")


.. describe:: clip

    Source clip; all color formats supported.

.. describe:: num_frames

    Fade duration, in frames.

.. describe:: color

    | Specifies the start/end color; black by default.
    | Color is specified as an RGB value in either hexadecimal or decimal notation.
    | Hex numbers must be preceded with a $. See the
      :doc:`colors <../syntax/syntax_colors>` page for more information on
      specifying colors.

    * For YUV clips, colors are converted from full range to limited range
      `Rec.601`_.

    * Use ``color_yuv`` or ``colors`` to specify full range YUV values or a
      color with a different matrix.

    Default: $000000

.. describe:: fps

    Provides a reference for num_frames in audio only clips. It is ignored if a
    video stream is present.

    * Set ``fps=AudioRate`` if sample exact audio positioning is required.

    Default: 24.0

.. describe:: color_yuv

    Specifies the start/end color using YUV values. Input clip must be YUV;
    otherwise an error is raised. See the :ref:`YUV colors <yuv-colors>` for
    more information.

.. describe:: colors

    Specifies the start/end color using an array. Use this to pass exact,
    unscaled color values. If the array is larger, further values are simply
    ignored.

    Color order: Y,U,V,A or R,G,B,A


Notes
-----

::

    FadeOut(clip, num_frames)

is just a shorthand for ::

    Dissolve(clip, BlankClip(clip, num_frames+1, color=color), num_frames)

(with num_frames+2 instead of num_frames+1 for ``FadeOut2``, and num_frames+0
for ``FadeOut0``).


Examples
--------

Fade in the first 15 frames from black on a 8-bit clip (either RGB or YUV)::

    # RGB
    Fade(15, color=$000000)
    Fade(15, colors=[0,0,0]

    # YUV
    Fade(15, color=$000000)       # limited range
    Fade(15, color_yuv=$108080)   # limited range
    Fade(15, colors=[16,128,128]  # limited range
    Fade(15, colors=[0,128,128]   # full range
    Fade(15, color_yuv=$008080)   # full range

Fade out the last 15 frames to white on a 8-bit clip (either RGB or YUV)::

    # RGB
    Fade(15, color=$FFFFFF)
    Fade(15, colors=[255,255,255]

    # YUV
    Fade(15, color=$FFFFFF)       # limited range
    Fade(15, color_yuv=$EB8080)   # limited range
    Fade(15, colors=[235,128,128] # limited range
    Fade(15, colors=[255,128,128) # full range
    Fade(15, color_yuv=$FF8080)   # full range


Changelog
---------

+-----------------+---------------------------------------------------------------------+
| Version         | Changes                                                             |
+=================+=====================================================================+
| AviSynth+ 3.7.2 | Added parameters ``color_yuv`` and ``colors`` for all fade filters. |
+-----------------+---------------------------------------------------------------------+
| AviSynth 2.5.6  || Added FadeIn0, FadeOut0, and FadeIO0 filters.                      |
|                 || Added optional ``fps`` parameter for processing audio only clips.  |
+-----------------+---------------------------------------------------------------------+
| AviSynth 2.0.7  || Added FadeIn, FadeIn2, FadeIO and FadeIO2 filters.                 |
|                 || Added the ``color`` parameter to all fade functions.               |
+-----------------+---------------------------------------------------------------------+

$Date: 2022/04/17 11:43:32 $

.. _Rec.601:
    https://en.wikipedia.org/wiki/Rec._601

.. |br| raw:: html

      <br>
