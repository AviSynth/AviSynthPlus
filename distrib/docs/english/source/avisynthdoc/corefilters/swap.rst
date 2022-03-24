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
only the first two clips are mandatory. Only Y or YUV(A) color formats are
accepted.

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
    * Due to `chroma subsampling`_ restrictions, some dimensions are not
      compatible with YUV420 and YUV422 color formats.

    If a ``clipY`` is not given, the Y channel of the output clip will be set to
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

$Date: 2022/03/24 14:12:41 $

.. _chroma subsampling:
    https://en.wikipedia.org/wiki/Chroma_subsampling
