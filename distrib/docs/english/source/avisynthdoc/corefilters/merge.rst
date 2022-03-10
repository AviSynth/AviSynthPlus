=============
Merge Filters
=============

Set of filters to merge (blend) two clips together:

* **Merge** merges all channels (RGB(A) or YUV(A)) from one video clip into another.
* **MergeChroma** merges only the chroma (U/V channels) from one video clip
  into another.
* **MergeLuma** merges only the luma (Y channel) from one video clip into
  another.

There is an optional weighting, so a percentage between the two clips can be
specified.

Syntax and Parameters
---------------------

::

    Merge (clip clip1, clip clip2, float "weight")
    MergeChroma (clip clip1, clip clip2, float "weight")
    MergeLuma (clip clip1, clip  clip2, float "weight")

    MergeChroma (clip clip1, clip clip2, float "chromaweight")
    MergeLuma (clip clip1, clip clip2, float "lumaweight")

.. describe:: clip1, clip2

    Source clips:

    * ``clip1``; the clip that has the pixels merged into (the base clip).
    * ``clip2``; the clip from which the pixel data is taken (the overlay clip).
    * **Merge** supports all RGB(A)/YUV(A) color formats.
    * **MergeChroma** and **MergeLuma**, only YUV(A) color formats supported.

    | Clips must have the same color format and dimensions.
    | Audio, FrameRate and FrameCount are taken from the first clip.
    | If clips contain an alpha channel, it is also processed.

.. describe:: weight

    Defines how much influence the new clip should have. Range is 0.0–1.0.

    * At 0.0, ``clip2`` has no influence on the output.
    * At 0.5, the output is the average of ``clip1`` and ``clip2``.
    * At 1.0, ``clip2`` replaces ``clip1`` completely.

      * For **MergeChroma**, output chroma taken only from ``clip2``.
      * For **MergeLuma**,  output luma taken only from ``clip2``.

    | Default: 0.5 (Merge)
    | Default: 1.0 (MergeChroma, MergeLuma)

    Note that the alternate parameter names ``chromaweight`` and ``lumaweight``
    are considered deprecated.

Examples
--------

::

    # Blur the Luma channel.
    MPEG2Source("main.d2v")
    clipY = Blur(1.0)
    MergeLuma(clipY)

::

    # Do a spatial smooth on the chroma channel
    # that will be mixed 50/50 with the original image.
    MPEG2Source("main.d2v")
    clipC = SpatialSoften(2,3)
    MergeChroma(clipC, weight=0.5)

::

    # Run a temporal smoother and a soft spatial
    # smoother on the luma channel, and a more aggressive
    # spatial smoother on the chroma channel.
    # The original luma channel is then added with the
    # smoothed version at 75%. The chroma channel is
    # fully replaced with the blurred version.
    MPEG2Source("main.d2v")
    clipY = TemporalSoften(2,3).SpatialSoften(3,10,10)
    clipC = SpatialSoften(3,40,40)
    MergeLuma(clipY, weight=0.75)
    MergeChroma(clipC)

::

    # Average two video sources.
    vid1 = AviSource("main.avi")
    vid2 = AviSource("main2.avi")
    Merge(vid1, vid2)


Changelog
---------

+-----------------+-----------------------------------------------------------------+
| Version         | Changes                                                         |
+=================+=================================================================+
| AviSynth+ r2487 || Merge: added planar RGB(A) and YUV(A) support.                 |
|                 || Merge: SSE2 for 10-14 bits (10-16 for SSE4.1 still work).      |
|                 || Merge (Merge,MergeChroma/Luma): add AVX2.                      |
|                 || Merge: float to sse2 (weighted and average).                   |
+-----------------+-----------------------------------------------------------------+
| AviSynth+ r2290 | Merge filters: added 16/32 bit support.                         |
+-----------------+-----------------------------------------------------------------+
| AviSynth 2.6.0  || MergeChroma and MergeLuma: Added alias ``weight`` for          |
|                 |  ``chromaweight`` and ``lumaweight``.                           |
|                 || Added support for Y8, YV16, YV24 and YV411 color formats.      |
+-----------------+-----------------------------------------------------------------+
| AviSynth 2.5.6  | Added Merge filter.                                             |
+-----------------+-----------------------------------------------------------------+

$Date: 2022/03/10 16:46:19 $
