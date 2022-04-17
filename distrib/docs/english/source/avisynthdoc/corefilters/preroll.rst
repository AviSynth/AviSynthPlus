=======
Preroll
=======

**Preroll** will seek *video* frames and/or *audio* seconds before the target
point on any non-linear access.

This may help avoid decoding problems with source filters that do not handle
random access correctly.

“Preroll works by detecting any out of order access in the audio or video track,
and seeking the specified amount earlier in the stream and then taking a
contiguous run up to the desired frame or audio sample. Skipping forward less
than the preroll values results in linear access behaviour, i.e. all the
intervening samples are accessed and discarded.” [`IanB`_]

Also see this `Doom9 thread`_.


Syntax and Parameters
----------------------

::

    Preroll (clip, int "video", float "audio")

.. describe:: clip

    Source clip; all color formats supported.

.. describe:: video

    Frames to preroll.

    Default: 0

.. describe:: audio

    Audio to preroll, in seconds.

    Default: 0.0


Examples
---------

25 frame video preroll, 10.0 seconds audio preroll::

    xxxSource("video.mp4")
    Preroll(video=25, audio=10.0)


Changelog
----------

+----------------+-----------------+
| Version        | Changes         |
+================+=================+
| AviSynth 2.6.0 | Initial release |
+----------------+-----------------+

$Date: 2022/04/17 13:38:34 $

.. _IanB:
    https://forum.doom9.org/showthread.php?p=1629239#post1629239
.. _Doom9 thread:
    https://forum.doom9.org/showthread.php?t=164457
