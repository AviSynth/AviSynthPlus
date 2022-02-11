
EnsureVBRMP3Sync
================

**EnsureVBRMP3Sync** will ensure synchronization of video with variable
bitrate audio (MP3-AVI's for example) during seeking or trimming. It does so
by buffering the audio. The name of the filter is a bit misleading, since it
is useful for every stream with variable bitrate audio and not just for MP3.

It will slow seeking down considerably, but is very useful when using
:doc:`Trim <trim>` for instance. Always use it before trimming.


Syntax and Parameters
----------------------

::

    EnsureVBRMP3Sync (clip)

.. describe:: clip

    Source clip.


Examples
--------

Ensures that soundtrack is in sync after trimming::

    AviSource("movie.avi")
    EnsureVBRMP3Sync()
    Trim(250,2500)

$Date: 2022/02/05 09:39:34 $
