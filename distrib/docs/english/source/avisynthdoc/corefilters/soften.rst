Soften Filters
==============

`SpatialSoften`_ and `TemporalSoften`_ remove noise from a video clip by
selectively blending pixels.

.. _SpatialSoften:

SpatialSoften
-------------

Like :doc:`Blur <blur>`, **SpatialSoften** blends neighboring pixels in the
frame – but with a wider possible ``radius``, and only if neighboring pixels
are within ``luma_threshold`` and ``chroma_threshold``, as explained below.

.. rubric:: Syntax and Parameters

::

    SpatialSoften (clip clip, int radius, int luma_threshold, int chroma_threshold)

.. describe:: clip

    Source clip. Must be `YUY2`_ color format.

.. describe:: radius

    | Filter radius – defines which pixels are processed.
    | Range 0-32; ``radius=0`` results in no smoothing. Values > 32 may cause
      AviSynth to crash.

.. describe:: luma_threshold, chroma_threshold

    When smoothing a given pixel P, ``SpatialSoften`` ignores any neighbor pixel
    P\ :sub:`n` where:

    * P\ :sub:`n` luma differs from P luma by more than ``luma_threshold``, OR
    * P\ :sub:`n` chroma differs from P chroma by more than ``chroma_threshold``.

.. _TemporalSoften:

TemporalSoften
--------------

Blends corresponding pixels in neighboring frames. All frames no more than radius
away are examined. Blending occurs only if corresponding pixels are within
``luma_threshold`` or ``chroma_threshold``, as explained below.

.. rubric:: Syntax and Parameters

::

    TemporalSoften (clip clip, int radius, int luma_threshold, int chroma_threshold,
                    int "scenechange", int "mode")

.. describe:: clip

    Source clip. All color formats are supported except RGB24 and RGB48.

.. describe:: radius

    | Filter radius. All frames no more than ``radius`` from the current frame
      are examined.
    | (for ``radius=2``, FIVE frames are processed: the current frame, two ahead
      and two behind)
    | Range 0-7; ``radius=0`` results in no smoothing.

.. describe:: luma_threshold

    When smoothing a given luma pixel Y, the corresponding pixel in neighboring
    frame Y\ :sub:`n` is ignored where Y\ :sub:`n` differs from Y by more than
    ``luma_threshold``.

.. describe:: chroma_threshold

    When smoothing a given chroma pixel C, the corresponding pixel in neighboring
    frame C\ :sub:`n` is ignored where C\ :sub:`n` differs from C by more than
    ``chroma_threshold``.

    * Good starting values are around 1 or 2 times ``luma_threshold``.

.. describe:: scenechange

    Defines the maximum average pixel change between frames; set properly, this
    will avoid blending across scene changes.

    * Good values are between 5 and 30, somewhat higher than ``luma_threshold``.
    * ``scenechange`` not supported in RGB32 and RGB64 colorspaces.

    Default: 0

.. describe:: mode

    **Deprecated** - this parameter is simply ignored.

.. note::
    Note that arguments are `autoscaling`_ – they are always 0-255 at all bit depths.


Examples
--------

Good initial values:

::

    TemporalSoften(4, 4, 8, scenechange=15)


Changelog
---------

+-----------------+---------------------------------------------------------------------------+
| Version         | Changes                                                                   |
+=================+===========================================================================+
| AviSynth+ 3.5.0 | Fix: TemporalSoften possible access violation after SeparateFields        |
|                 | (in general: after filters that only change frame pitch).                 |
+-----------------+---------------------------------------------------------------------------+
| AviSynth+ r2580 | Fix: TemporalSoften 10-14 bits: an SSE 4.1 instruction was used for       |
|                 | SSE2-only CPUs.                                                           |
+-----------------+---------------------------------------------------------------------------+
| AviSynth+ r2508 | Fix: TemporalSoften threshold < 255 (probably since r1576).               |
+-----------------+---------------------------------------------------------------------------+
| AviSynth+ r2487 || TemporalSoften: 10-12-14 bit support for planar RGB(A)/YUV(A) and        |
|                 |  RGB32/64 formats.                                                        |
|                 || TemporalSoften: much faster average mode (thres=255).                    |
+-----------------+---------------------------------------------------------------------------+
| AviSynth+ r2397 || TemporalSoften: Planar RGB(A) support.                                   |
|                 || TemporalSoften: much faster average mode (thres=255).                    |
+-----------------+---------------------------------------------------------------------------+
| AviSynth+ r2290 | TemporalSoften: added 16/32 bit support.                                  |
+-----------------+---------------------------------------------------------------------------+
| AviSynth+ r1841 | TemporalSoften: frame leak fix.                                           |
+-----------------+---------------------------------------------------------------------------+
| Avisynth+ r1576 || TemporalSoften: mode 1 removed, ``mode`` parameter ignored.              |
|                 || TemporalSoften: now has C and SSE2 versions.                             |
+-----------------+---------------------------------------------------------------------------+
| AviSynth 2.6.0  | TemporalSoften: added support for Y8, YV411, YV16 and YV24 color formats. |
+-----------------+---------------------------------------------------------------------------+
| AviSynth 2.5.6  | TemporalSoften working also with RGB32 input (as well as YV12, YUY2).     |
+-----------------+---------------------------------------------------------------------------+

$Date: 2022/02/28 18:06:23 $

.. _YUY2:
    http://avisynth.nl/index.php/YUY2
.. _autoscaling:
    http://avisynth.nl/index.php/Autoscale_parameter
