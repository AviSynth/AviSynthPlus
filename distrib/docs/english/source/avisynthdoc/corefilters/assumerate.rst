================
AssumeSampleRate
================

Changes the sample rate of the current clip without changing the number of samples.

* This changes the sample rate, pitch, playback speed and running time of the audio.
* It will also affect synchronization with the video.


Syntax and Parameters
---------------------

::

    AssumeSampleRate (clip, int samplerate)

.. describe:: clip

    Source clip; all color formats and audio sample types supported.

.. describe:: samplerate

    New sample rate.


Examples
--------

Fix an audio sync problem::

    AviSource("video_audio.avi")
    # Let's say that this clip loses audio sync:
    # by the end, the audio is behind by 200 milliseconds.

    Ar  = AudioRate()     # original audio sample rate
    dur = AudioDuration() # total duration in seconds
    adj = -0.200          # correction needed (decrease audio duration by 200 msec)

    # Adjust audio duration:
    AssumeSampleRate(Round(dur / (dur+adj) * Ar))

    # Video and audio are now in sync.

    # (optional) restore original sample rate.
    #ResampleAudio(Ar)

Play video and audio at half speed::

    AviSource("video_audio.avi")

    # Play video at half speed: audio will be out of sync
    AssumeFPS(FrameRate/2)

    # Play audio at half speed: (pitch will be lowered by an octave)
    AssumeSampleRate(AudioRate/2)

    # Video and audio are now in sync.
    # Equivalent to calling AssumeFPS(..., sync_audio=true) (see below)


See also
--------

:doc:`AssumeFPS <fps>`: *sync_audio*

    If *true*, the audio sample rate is changed by the same amount; the pitch of
    the resulting audio is shifted.

    If *false* (the default), the audio is unchanged; this means the audio will
    lose synchronization over time.


Changelog
----------

+-----------------+--------------+
| Version         | Changes      |
+=================+==============+
| AviSynth 2.0.7  | First added  |
+-----------------+--------------+

$Date: 2022/03/17 15:28:44 $
