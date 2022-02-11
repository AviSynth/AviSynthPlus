
SelectRangeEvery
================

**SelectRangeEvery** selects *length* number of frames *every* n frames, 
starting from frame *offset*.


Syntax and Parameters
----------------------

::

    SelectRangeEvery (clip, int "every", int "length", int "offset", bool "audio")

.. describe:: clip

    Source clip.

.. describe:: every

    Frame selection interval.

    Default: 1500

.. describe:: length

    Frame selection length; how many frames to select.

    Default: 50

.. describe:: offset

    Frame selection offset.

    Default: 0

.. describe:: audio

    Audio processing; set it to false to keep the original audio.

    Default: true


Examples
--------

**SelectRangeEvery** with default settings selects 50 frames every 1500 frames, 
starting with first selection at frame 0. ::

    SelectRangeEvery(clip, every=1500, length=50, offset=0)

Selects the frames 0 to 13, 280 to 293, 560 to 573, etc. ::

    SelectRangeEvery(clip, every=280, length=14, offset=0)

Selects the frames 2 to 15, 282 to 295, 562 to 575, etc. ::

    SelectRangeEvery(clip, every=280, length=14, offset=2)

$Date: 2022/02/05 18:54:28 $

