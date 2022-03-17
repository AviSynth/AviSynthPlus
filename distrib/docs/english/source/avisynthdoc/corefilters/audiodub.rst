=====================
AudioDub / AudioDubEx
=====================

**AudioDub** takes the video stream from the first argument and the audio stream
from the second argument and combines them into a single clip. If either track
isn't available, it tries it the other way around, and if that doesn't work
it returns an error.

**AudioDubEx** takes the video stream from the first argument if present, the
audio stream from the second argument if present and combines them into a single
clip. Thus, if you feed it with two video clips, and the second one has no audio,
the resulting clip will have the video of the first clip and no audio. If you
feed it with two audio clips, the resulting clip will have the audio of the
second clip and no video.


Syntax and Parameters
----------------------

::

    AudioDub (clip video, clip audio)
    AudioDubEx (clip video, clip audio)

.. describe:: clip

    Video clip; all color formats supported.

.. describe:: clip

    Audio clip; all audio sample types supported.

Examples
--------

::

    # Load capture segments from patched AVICAP32 which puts
    # video in multiple AVI segments and audio in a WAV file
    video = AVISource("capture1.avi") + AVISource("capture2.avi")
    audio = WavSource("capture.wav")
    # combine them into a single clip
    AudioDub(video, audio)


Changelog
----------

.. table::
    :widths: auto

    +----------------+--------------------------+
    | Version        | Changes                  |
    +================+==========================+
    | AviSynth 2.5.6 | Added AudioDubEx filter. |
    +----------------+--------------------------+

$Date: 2022/03/17 12:37:33 $
