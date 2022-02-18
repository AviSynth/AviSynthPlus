
Compare
=======

Compares two clips (the "filtered" clip and the "original" clip) and returns the 
following data for each frame: 

    | **Mean Absolute Deviation** `[1]`_
    |     *(minimum, average, maximum)*
    | **Mean Deviation**
    |     *(minimum, average, maximum)*
    | **Max. Positive Deviation**
    | **Max. Negative Deviation**
    | **Peak Signal-to-Noise Ratio (PSNR)** `[2]`_
    |     (minimum, average, maximum)
    | **PSNR**
    |     *(minimum, average, maximum)*
    | **Overall PSNR**

This data is displayed on screen by default (see `Examples`_), or can 
optionally be written to a *logfile*.

By default a *PSNR graph* is displayed; this graph shows overall PSNR visually 
as a timeline that grows left-to-right along the bottom of the screen as the 
clip plays. 


Syntax and Parameters
----------------------

::

    Compare (clip filtered, clip original, string "channels", string "logfile", bool "show_graph")

.. describe:: filtered, original

    | Source clips; the "filtered" clip and the "original" clip. Size and color 
      formats must match. 
    | All color formats supported except 32-bit float.
    | The "filtered" clip is returned. 

.. describe:: channels

    | Define which color channels to be compared by their initial letters, e.g. 
      "R" (=red).
    | Valid channel letters are:

    * R, G, B, A for RGB(A) clips.
    * Y, U, V, A for YUV(A) clips.
    * Y for single channel greyscale clips.

    | Letters are not case sensitive and may be given in any order. 
    | By default, all channels are compared (except for the alpha (A) channel).

    Default: ""

.. describe:: logfile

    | If specified, the results will be written to a file, and not drawn on the 
      clip. 
    | Using a logfile is much faster if you need to compare a lot of frames. 

    Default: ""

.. describe:: show_graph

    If true, the PSNR graph is drawn on the clip.

    Default: true


Examples
--------

* Basic usage – display differences on screen::

    LSMASHVideoSource("sintel-2048-surround.mp4")
    A = BicubicResize(640, 272).Crop(80, 0, -80, 0)
    B = A.Sharpen(1.0) ## Sharpen is our filter under test
    return Compare(B, A)

 .. figure:: pictures/compare-sintel-9507.jpg
    :align: left


* Create a log file::

    Compare(clip1, clip2, log="compare.log")

* Compare chroma channels only::

    Compare(clip1, clip2, channels="UV")

* See also the `Doom9 discussion`_.


Changelog
---------

+-----------------+-------------------------------------------------------------+
| Version         | Changes                                                     |
+=================+=============================================================+
| AviSynth+ 3.7.2 || Fix: ``channels`` now defaults to "Y" instead of "YUV" for |
|                 |  greyscale input.                                           |
|                 || Compare: fix 10-14 bit support (graph, PSNR).              |
+-----------------+-------------------------------------------------------------+
| AviSynth+ r2150 || Compare: port to 16 bits (RGB48/64, Planar YUV(A)/RGB(A)). |
|                 || Fix:  Negative SAD in 8 bit SSE branches.                  |
+-----------------+-------------------------------------------------------------+
| AviSynth 2.5.8  | YV12 support.                                               |
+-----------------+-------------------------------------------------------------+

$Date: 2022/02/18 19:42:53 $

.. _[1]:
    https://en.wikipedia.org/wiki/Statistical_dispersion
.. _[2]:
    http://avisynth.nl/index.php/PSNR
.. _Sintel:
    https://durian.blender.org/download/
.. _Doom9 discussion:
    https://forum.doom9.org/showthread.php?t=29538
