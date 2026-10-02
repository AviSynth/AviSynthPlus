============
Swap Filters
============

Set of filters to combine, extract and manipulate the order of channels in
YUV(A) clips:

* `SwapUV`_ swaps the order of the U/V channels.
* `UToY / VToY`_ copies the U/V channel onto the Y (luma) channel and keeps the
  same color format.
* `UToY8 / VToY8`_ extract the U/V chroma channel to a Y-only greyscale clip.
* `YToUV`_ combines individual YUV(A) planes into a new YUV(A) clip.

These filters provide similar functionality to the :doc:`CombinePlanes <combineplanes>`,
:doc:`Extract <extract>` and :doc:`ShowU/V <showalpha>` filters.

.. _SwapUV:

SwapUV
------

Swaps the U and V (chroma) channels. Corrects certain decoding errors – faces
blue instead of red, etc.


.. rubric:: Syntax and Parameters

::

    SwapUV (clip)

.. describe:: clip

    Source clip. All YUV(A) color formats supported.

.. _UToY:
.. _VToY:

UToY / VToY
-----------

**UToY** and **VToY** copy the U or V chroma plane to the Y luma plane and
keeps the same color format as the source clip. All color (chroma) information
is removed and set to neutral (greyscale). Depending on the color format, the
image resolution can be changed – i.e.,

* with a YUV444 source, the output clip will be the same width and height as
  input clip, but
* with a YUV420 source, the output clip will be half the input clip's width and
  height.

.. rubric:: Syntax and Parameters

::

    UToY (clip)
    VToY (clip)

.. describe:: clip

    Source clip; all YUV(A) color formats supported.

.. _UToY8:
.. _VToY8:

UToY8 / VToY8
-------------

**UToY8** and **VToY8** extract the U or V chroma channel to a Y-only greyscale
clip. Despite the names, all bit depths are supported. Resulting clip will be Y8,
Y10 etc. as appropriate.

.. rubric:: Syntax and Parameters

::

    UToY8 (clip)
    VToY8 (clip)

.. describe:: clip

    Source clip; all YUV(A) color formats supported.

.. _YToUV:

YToUV
-----

**YToUV** combines up to 4 independent clips to create a new YUV(A) clip.
The Y channel of each of the supplied clips are then copied onto the respective
channel of the output clip. Note that all of the parameters are unnamed, however,
only the first two clips are mandatory. Only Y, YA or YUV(A) color formats are
accepted (only the Y plane of the clips is used; alpha of ``clipA`` if it has one).

.. rubric:: Syntax and Parameters

::

    YToUV (clip clipU, clip clipV, clip clipY, clip clipA)

.. describe:: clipU, clipV

    | Source clips; dimensions of both clips must be identical.
    | The first clip is used for the U channel and the second clip for the V
      channel.

    | ``clipU`` determines the color format of the output clip unless ``clipY``
      is defined.

.. describe:: clipY

    Source clip; If ``clipY`` is given, the Y channel is copied onto the Y
    channel of the output clip. The dimensions of this clip determines the
    color format of the output clip, for example:

    * If the width and height of ``clipY`` are the same as the U/V channels, the
      output clip will be YUV444.
    * If the width and height of ``clipY`` are double the size of the U/V
      channels, the output clip will be YUV420.
    * Width x height ratios of ``clipY`` to the U/V clips: 1x1 (YUV444), 2x1 (YUV422),
      2x2 (YUV420), 4x1 (YUV411), 1x2 (YUV440) and 4x4 (YUV410); other ratios are
      rejected.
    * Due to `chroma subsampling`_ restrictions, some dimensions are not
      compatible with YUV420 and YUV422 color formats.

    If ``clipY`` is not given, the Y channel of the output clip will be set to
    grey (0x7e).

.. describe:: clipA

    Source clip; if ``clipA`` is given, the Y channel is copied onto the A
    channel of the output clip. Dimensions must be identical to ``clipY``.


Examples
--------

Blur the U and V chroma channels different amounts::

    video = ColorBars(512, 512, pixel_type="YUV420P8")
    u = UToY8(video).Blur(1.5)
    v = VToY8(video).Blur(0.5)
    YtoUV(u, v, video)

Build a YUVA clip from greyscale U/V clips and a YA (Y plus alpha) clip, which gives
both the Y and the alpha plane::

    video = ColorBars(512, 512, pixel_type="YUV444P8")
    u = ExtractU(video)
    v = ExtractV(video)
    ya = CombinePlanes(ExtractY(video), ExtractU(video), planes="YA") # YA8, alpha from U
    YToUV(u, v, ya, ya) # YUVA444P8: Y from ya's Y plane, alpha from ya's alpha plane

Show *U* and V channels stacked side by side for illustration purposes.

* Note that with a YUV420 source (like the image below), the *U* and *V* images
  will be half the size of the original.
* In the *U* and *V* images, grey will be "neutral" (for example, 128 for 8-bit)
  and saturated colors will appear brighter or darker.

 .. list-table::

    * - .. figure:: pictures/swap-peppers.jpg

           *swap-peppers.jpg*

    * - .. figure:: pictures/swap-peppers-uv.jpg

        .. code::

            src   = FFImageSource("swap-peppers.jpg")
            srcU  = src.UToY().Subtitle("UtoY", align=2)
            srcV  = src.VToY().Subtitle("VtoY", align=2)
            srcUV = StackHorizontal(srcU, srcV)

            StackVertical(src, srcUV)


Frame properties
----------------

* **SwapUV**: all frame properties are kept unchanged (swapping U and V does not
  change the chroma placement).
* **UToY / VToY**: all frame properties are copied from the source clip, including
  ``_ChromaLocation`` (the chroma planes are neutral grey, so it has no effect).
* **UToY8 / VToY8**: same as :doc:`ExtractU / ExtractV <extract>`: ``_ChromaLocation``
  is deleted, since a Y-only clip has no chroma planes; other properties, such as
  ``_ColorRange`` and ``_Matrix``, are kept.
* **YToUV**: all frame properties are copied from ``clipU``; the properties of
  ``clipV``, ``clipY`` and ``clipA`` are ignored. ``_ChromaLocation`` is not adjusted:
  when the U and V clips are greyscale (e.g. ``UToY8``/``VToY8`` or ``ExtractU``/``ExtractV``
  output), the result has no ``_ChromaLocation``, so the format's default placement
  applies (see ``ChromaInPlacement`` in :doc:`Convert <convert>`; e.g. "left" for 4:2:0
  and 4:2:2). Use ``propSet`` to restore it, or :doc:`CombinePlanes <combineplanes>`,
  which keeps the first clip's ``_ChromaLocation`` in this case::

      u = UToY8(video).Blur(1.5)
      v = VToY8(video).Blur(0.5)
      CombinePlanes(video, u, v, planes="YUV", source_planes="YYY") # keeps video's _ChromaLocation

Changelog
---------

.. table::
    :widths: auto

    +-----------------+----------------------------------------------+
    | Version         | Changes                                      |
    +=================+==============================================+
    | AviSynth+ r2487 || Added parameter ``clipA`` to YToUV.         |
    |                 || Added YUVA support to SwapUV.               |
    |                 || Added support for 10-16 bits and float.     |
    +-----------------+----------------------------------------------+
    | AviSynth 2.6.0  || Added UToY8 and VToY8.                      |
    |                 || Added support for Y8, YV411, YV16, YV24.    |
    +-----------------+----------------------------------------------+
    | AviSynth 2.5.3  | Added support for YUY2.                      |
    +-----------------+----------------------------------------------+
    | AviSynth 2.5.1  | Added parameter ``clipY`` to YToUV.          |
    +-----------------+----------------------------------------------+
    | AviSynth 2.5.0  | Added UToY, VToY, YToUV.                     |
    +-----------------+----------------------------------------------+

$Date: 2026/10/02 10:00:00 $

.. _chroma subsampling:
    https://en.wikipedia.org/wiki/Chroma_subsampling
