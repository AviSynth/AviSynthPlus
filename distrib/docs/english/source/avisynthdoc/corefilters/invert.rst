======
Invert
======

Inverts one or several color channels of a clip.


Syntax and Parameters
----------------------

::

    Invert (clip, string "channels")

.. describe:: clip

    Source clip; all color formats supported.

.. describe:: channels

    | Defines which channels should be inverted by their initial letters, e.g.
      "R" (=red).
    | Any letters that don't correspond to a channel in the current colorspace
      are ignored.
    | Valid channel letters are:

    * R, G, B, A for RGB(A) clips.
    * Y, U, V, A for YUV(A) clips.

    | Letters are not case sensitive and may be given in any order.
    | By default, all channels of the current colorspace are inverted.

    Default: "RGBA" if input clip is RGB, "YUVA" if input clip is YUV.


Examples
---------

Invert the blue and green channels::

    AviSource("clip.avi")
    ConvertToRGB32()
    Invert(channels="BG") # can also be written as channels="g, b"

Examples were Invert has no effect::

    AviSource("clip.avi")
    ConvertToRGB24()
    Invert(channels="A")     # no effect (no current A channel)
    Invert(channels="VUY")   # no effect (no current Y, U or V channels)


Changelog
---------

+-----------------+--------------------------------------------------------+
| Version         | Changes                                                |
+=================+========================================================+
| AviSynth+ r2487 | Added support for YUV(A)/PlanarRGB(A) 8,10-16,32 bit,  |
|                 | RGB48/64 color formats, with SSE2.                     |
+-----------------+--------------------------------------------------------+
| AviSynth 2.6.0  | Added support for YV24, YV16, YV411, Y8 color formats. |
+-----------------+--------------------------------------------------------+
| Avisynth 2.5.5  | Added support for RGB24, YUY2 and YV12 color formats.  |
+-----------------+--------------------------------------------------------+
| AviSynth 2.5.3  | Initial Release.                                       |
+-----------------+--------------------------------------------------------+

$Date: 2022/03/20 16:58:20 $
