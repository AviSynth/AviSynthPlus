========
MixAudio
========

Mixes audio from two clips. A volume for the two clips can be given, but is
optional.

Syntax and Parameters
----------------------

::

    MixAudio (clip1, clip2, float "clip1_factor", float "clip2_factor")

.. describe:: clip1, clip2

    Source clips; all audio sample types supported. Output video is copied from
    ``clip1``.

    * The sample rate of the two clips needs to be the same – use
      :doc:`ResampleAudio <resampleaudio>` if necessary.
    * The source clips also need to have the same number of channels – use
      :doc:`ConvertToMono <converttomono>`, :doc:`MonoToStereo <monotostereo>`
      or :doc:`MergeChannels <mergechannels>` if necessary.
.. describe:: clip1_factor, clip1_factor

    Set the volume of each clip's audio, where 0.0 is no audio from that clip,
    and 1.0 is 100% audio.

    * If ``clip1_factor`` + ``clip2_factor`` > 1.0, you risk `clipping`_ the
      audio if the sample type is not `Float`_.
    * If only one the first factor is given, the second factor will be
      (1.0-``clip1_factor``).

    Default: 0.5, (1.0 - ``track1_factor``)


Examples
--------

Mixes two sources, with one source slightly lower than the other::

    video = AviSource("c:\movie.avi")
    Soundtrack = WavSource("c:\soundtrack.wav")
    Speak = WavSource("c:\speak.wav")
    audio = MixAudio(Soundtrack, Speak, 0.75, 0.25)
    return AudioDub(video, audio)

For this particular example, setting ``clip2_factor`` to 0.25 is redundant. When
the second factor is not given, it defaults to ``(1.0 - clip1_factor)``, which
ends up being 0.25.


Changelog
---------

+-----------------+------------------------+
| Version         | Changes                |
+=================+========================+
| AviSynth 2.0.3  | Added MixAudio filter. |
+-----------------+------------------------+

$Date: 2022/03/24 14:45:27 $

.. _clipping:
    https://en.wikipedia.org/wiki/Clipping_(audio)
.. _Float:
    http://avisynth.nl/index.php/Float
