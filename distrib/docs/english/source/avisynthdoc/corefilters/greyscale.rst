=========
Greyscale
=========

Converts the input clip to greyscale (no color), without changing the color format.
For RGB, the resulting clip will consist of 3 identical channels. For YUV, the
luma channel is left as is and the chroma channels are set to grey (neutral).

UK (Greyscale) and US (Grayscale) spellings are both allowed.


Syntax and Parameters
----------------------

::

    Greyscale (clip  string "matrix")
    Grayscale (clip, string "matrix")

.. describe:: clip

    Source clip; all color formats supported.

.. describe:: matrix

    In RGB based formats, the conversion produces the luma using the coefficients
    given in the ``matrix`` parameter.

    * "Rec601" : use `Rec.601`_ coefficients (and keeping luma range unchanged)
    * "Rec709" : use `Rec.709`_ (HD) coefficients (and keeping luma range unchanged)
    * "Rec2020" : use `Rec.2020`_ (UHD) coefficients (and keeping luma range unchanged)
    * "AVERAGE" : use averaged coefficients (Y = (R + G + B) / 3)

    Default: "Rec601"

    See :doc:`Color conversions <../advancedtopics/color_conversions>` for an
    explanation of the coefficients. Broadly speaking though,

    * using ``"Rec709"``, **green** contributes more to the output, compared to
      ``"Rec2020"`` or the default ``"Rec601"``;
    * using ``"Average"``, **blue** contributes more.

    If the source clip is `YUV`_, the chroma channels are simply set to neutral
    (e.g. 128 for 8-bit clips) – ``matrix`` is not used and **must not be**
    specified otherwise an error will be thrown.

    In all cases, luma range is not changed. In other words, **Greyscale** does
    not do any range conversion, meaning if the source is full range, the output
    will also be full range. The same applies to limited range.


Examples
--------

For YUV clips, ``Greyscale()`` is identical to:

.. code-block:: swift

    /* assume a YUV(A) clip as the source */
    src = last
    csp = BuildPixelType(sample_clip=src)
    ShowY(pixel_type=csp)

For RGB clips, ``Greyscale(matrix="Rec601")`` is identical to:

.. code-block:: swift

    /* assume an RGB(A) clip as the source */
    src = last
    compat = IsInterleaved(src) ? true : false
    csp = BuildPixelType(compat=compat, sample_clip=src)
    ConvertToY(matrix="PC.601") /* or "PC.709" "PC.2020", "Average" */
    ShowY(pixel_type=csp)
    HasAlpha(src) ? AddAlphaPlane(last, src) : last


Changelog
----------

+------------------+--------------------------------------------------------------------+
| Version          | Changes                                                            |
+==================+====================================================================+
| AviSynth+ 3.7.2  | Greyscale to not convert to limited range when RGB. Regression in  |
|                  | 3.7.1. Issue `#257`_.                                              |
+------------------+--------------------------------------------------------------------+
| AviSynth+ r2728  | Greyscale: zero-centered 32bit chroma support.                     |
+------------------+--------------------------------------------------------------------+
| AviSynth+ r2487  || Add new matrix: "Rec2020".                                        |
|                  || Add support for RGB64, PlanarRGB(A), and remaining 10-12-14 bit   |
|                  |  YUV(A) formats.                                                   |
+------------------+--------------------------------------------------------------------+
| AviSynth+ r2003  | Added 16 and 32 bit YUV(A) support.                                |
+------------------+--------------------------------------------------------------------+
| AviSynth 2.6.0   | Added support for Y8, YV16, YV24 and YV411 formats.                |
+------------------+--------------------------------------------------------------------+
| AviSynth 2.5.7   | Greyscale RGB now accepts "Rec601" as a valid matrix.              |
+------------------+--------------------------------------------------------------------+
| AviSynth 2.5.6   | Added ``matrix`` parameter, Greyscale RGB now supports "Rec709"    |
|                  | and "Average" matrices.                                            |
+------------------+--------------------------------------------------------------------+

$Date: 2022/04/29 13:57:17 $

.. _Rec.601:
    https://en.wikipedia.org/wiki/Rec._601
.. _Rec.709:
    https://en.wikipedia.org/wiki/Rec._709
.. _Rec.2020:
    https://en.wikipedia.org/wiki/Rec._2020
.. _YUV:
    http://avisynth.nl/index.php/YUV
.. _#257:
    https://github.com/AviSynth/AviSynthPlus/issues/257
