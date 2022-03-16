====
Crop
====

AviSynth has two dedicated cropping filters:

* `Crop`_ is an all-purpose cropping filter.
* `CropBottom`_ only crops the bottom of the frame.

For subpixel cropping see the cropping section of the
:ref:`Resize <resize-cropping>` filters.

*Cropping* refers to the removal of the outer parts of an image to improve
framing, changing the framesize (also called image or storage aspect ratio).
See Wikipedia: `Cropping`_.

.. _Crop:

Crop
----

Crops excess pixels off of each frame. Cropping syntax can be specified in two
different ways as explained below.

.. rubric:: Syntax and Parameters

::

    Crop (clip clip, int left, int top, int width, int height, bool "align")
    Crop (clip clip, int left, int top, int -right, int -bottom, bool "align")

.. describe:: clip

    Input clip; all color formats supported.

.. describe:: left, top

        Cropping of the left and top edges respectively, in pixels.

        * See :ref:`crop modulo restrictions <crop-restrictions>` below.

.. describe:: width, height
.. describe:: -right, -bottom

    The third and fourth arguments have different names or aliases, depending on
    their value:

    * If  > zero, these set the ``width`` and ``height`` of the resulting clip.
    * If <= zero, they set the cropping of the ``right`` and ``bottom`` edges
      respectively.

    Note, there are certain limits:

    * clip.Width must be >= (``left`` + ``width``)
    * clip.Width must be >  (``left`` + ``right``)
    * clip.Height must be >= (``top`` + ``height``)
    * clip.Height must be >  (``top`` + ``bottom``)

    ...otherwise it would enlarge ("un-crop") the clip, or reduce width or height
    to 0, which is not allowed.

    * See :ref:`crop modulo restrictions <crop-restrictions>` below.

.. describe:: align

        Cropping an YUY2/RGB32 image is always `mod4`_ (four bytes). However, when
        reading x bytes (an int), it is faster when the read is aligned to a modx
        placement in memory. MMX/SSE likes 8-byte alignment and SSE2 likes 16-byte
        alignment. If the data is NOT aligned, each read/write operation will be
        delayed at least 4 cycles. So images are always aligned to mod16 when they
        are created by AviSynth.

        If an image has been cropped, they will sometimes be placed unaligned in
        memory; ``align=true`` will copy the entire frame from the unaligned
        memory placement to an aligned one.

        **So if the penalty of the following filter is larger than the penalty
        of a complete image copy, using** ``align=true`` **will be faster** –
        especially when it is followed by smoothers.

        Default: true

.. _CropBottom:

CropBottom
----------

**CropBottom** was created to crop garbage off the bottom of a clip captured
from VHS tape. It removes ``count`` lines from the bottom of the frame.

.. rubric:: Syntax and Parameters

::

    CropBottom (clip clip, int count)

.. describe:: clip

    Input clip; all color formats supported.

.. describe:: count

    | How many lines to crop from the bottom.
    | See :ref:`crop modulo restrictions <crop-restrictions>` below.

.. _crop-restrictions:

Crop Restrictions
-----------------

In order to preserve the data structure of the different colorspaces, the
following `mods`_ should be used. You will not get an error message if they are
not obeyed, but it may create strange artifacts. For a complete discussion on
this, see :doc:`../FilterSDK/DataStorageInAviSynth`. Also see the Doom9 Forum
thread `"Crop Restrictions"`_.


+----------------+------------------+--------------------------------------+
| **Colorspace** | **Width**        | **Height**                           |
+----------------+------------------+-------------------+------------------+
|                |                  | progressive video | interlaced video |
+================+==================+===================+==================+
| RGB            | *no restriction* | *no restriction*  | mod-2            |
+----------------+------------------+-------------------+------------------+
+ Y              | *no restriction* | *no restriction*  | mod-2            |
+----------------+------------------+-------------------+------------------+
| YUV411         | mod-4            | *no restriction*  | mod-2            |
+----------------+------------------+-------------------+------------------+
| YUV420         | mod-2            | mod-2             | mod-4            |
+----------------+------------------+-------------------+------------------+
| YUV422         | mod-2            | *no restriction*  | mod-2            |
+----------------+------------------+-------------------+------------------+
| YUV444         | *no restriction* | *no restriction*  | mod-2            |
+----------------+------------------+-------------------+------------------+

.. note:: The :doc:`resize functions <resize>` optionally allow fractional pixel
    cropping of the input frame, this results in a weighting being applied to the
    edge pixels being resized.  These options may be used if the mod-n format
    dimension restriction of crop are inconvenient.

    In summary: *for cropping off hard artifacts like VHS head noise or letterbox
    borders always use Crop. For extracting a portion of an image and to maintain
    accurate edge resampling use the resize cropping parameters.* (`Doom9 thread`_)


Examples
--------

If your source video has 720x480 resolution, and you want to reduce it to 352x240
for `VideoCD`_, here's the correct way to do it::

    # Converts CCIR601 to VCD, preserving the correct aspect ratio
    ReduceBy2
    Crop(4, 0, 352, 240)

Using the alternative syntax with negative (or zero) values in the last two
parameters, they become offsets, `VirtualDub`_-style::

    # Crops 16 pixels all the way around the picture, regardless of image size
    Crop(16, 16, -16, -16)

    # crop 8 off the left, 2 off the top, 9 off the right, and 4 off the bottom
    Crop(8, 2, -9, -4)


Changelog
----------

.. table::
    :widths: auto

    +------------------+----------------------------------------------------+
    | Version          | Changes                                            |
    +==================+====================================================+
    | AviSynth+ r1576  | Fix: a fix for bad alignment in the Crop filter.   |
    +------------------+----------------------------------------------------+
    | AviSynth+ <r1576 | Crop: ``align`` now defaults to true. (2013-11-24) |
    +------------------+----------------------------------------------------+
    | AviSynth 2.5.3   | Crop: added ``align`` option (false by default).   |
    +------------------+----------------------------------------------------+
    | AviSynth 2.0.1   | Crop: added VDub style crop syntax.                |
    +------------------+----------------------------------------------------+

$Date: 2022/03/16 15:10:22 $

.. _Cropping:
    http://en.wikipedia.org/wiki/Cropping_(image)
.. _mod4:
    http://avisynth.nl/index.php/Modulo
.. _mods:
    http://avisynth.nl/index.php/Modulo
.. _"Crop Restrictions":
    https://forum.doom9.org/showthread.php?t=51923
.. _Doom9 thread:
    http://forum.doom9.org/showthread.php?s=&threadid=91630
.. _VideoCD:
    http://en.wikipedia.org/wiki/Video_CD
.. _VirtualDub:
    http://avisynth.nl/index.php/VirtualDub2
