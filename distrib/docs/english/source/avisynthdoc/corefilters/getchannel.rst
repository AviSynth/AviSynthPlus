==========
GetChannel
==========

Set of filters to extract audio channels:

* **GetChannel** returns one or more channels of a multichannel signal.
* **GetLeftChannel** returns the left channel from a stereo signal, and
  **GetRightChannel** returns the right.
* **GetChannels** is an alias for **GetChannel** and can be used interchangeably.


Syntax and Parameters
----------------------

::

    GetChannel (clip, int ch1 [, int ch2, ...])
    GetChannels (clip, int ch1 [, int ch2, ...])

    GetLeftChannel(clip clip)
    GetRightChannel(clip clip)

.. describe:: clip

    Source clip; all audio sample types supported.

.. describe:: ch1, ch2, ...

    Specify what channel(s) to return.

    The ordering of the channels is determined by the ordering of the input
    file, because AviSynth doesn't assume any ordering (see `Remarks`_).

    In case of WAV files, the ordering should be as follows:

    +-----------------------------------+
    | WAV, 2 channels (stereo)          |
    +===+===============================+
    | 1 | left channel                  |
    +---+-------------------------------+
    | 2 | right channel                 |
    +---+-------------------------------+

    +-----------------------------------+
    | WAV, 5.1 channels                 |
    +===+===============================+
    | 1 | front left channel            |
    +---+-------------------------------+
    | 2 | front right channel           |
    +---+-------------------------------+
    | 3 | front center channel          |
    +---+-------------------------------+
    | 4 | LFE (Subwoofer)               |
    +---+-------------------------------+
    | 5 | rear left channel             |
    +---+-------------------------------+
    | 6 | rear right channel            |
    +---+-------------------------------+

Remarks
^^^^^^^

Every file format has a different internal channel ordering. The following table
gives this internal ordering for some formats (useful for plugin writers), but
it is the decoder's task to return the expected channel order. If you use
decoders like `NicAudio`_/\ `BassAudio`_ or `ffdshow`_/\ `AC3Filter`_ you don't
need to worry about this.

    +-------------+--------------+--------------+--------------+-------------+------------+------------+
    | Format      | Channel 1    | Channel 2    | Channel 3    | Channel 4   | Channel 5  | Channel 6  |
    +=============+==============+==============+==============+=============+============+============+
    | `5.1 WAV`_  | front left   | front right  | front center | LFE         | rear left  | rear right |
    +-------------+--------------+--------------+--------------+-------------+------------+------------+
    | `5.1 AC3`_  | front left   | front center | front right  | rear left   | rear right | LFE        |
    +-------------+--------------+--------------+--------------+-------------+------------+------------+
    | `5.1 DTS`_\*| front center | front left   | front right  | rear left   | rear right | LFE        |
    +-------------+--------------+--------------+--------------+-------------+------------+------------+
    | `5.1 AAC`_\*| front center | front left   | front right  | rear left   | rear right | LFE        |
    +-------------+--------------+--------------+--------------+-------------+------------+------------+
    | `5.1 AIFF`_ | front left   | rear left    | front center | front right | rear right | LFE        |
    +-------------+--------------+--------------+--------------+-------------+------------+------------+
    | `5.1 FLAC`_ | front left   | front right  | front center | LFE         | rear left  | rear right |
    +-------------+--------------+--------------+--------------+-------------+------------+------------+
    | `5.1 WMA`_  | front left   | front right  | front center | LFE         | rear left  | rear right |
    +-------------+--------------+--------------+--------------+-------------+------------+------------+
    || [1] 5.1 DTS: the LFE is on a separate stream (much like on multichannel MPEG2).                 |
    || [2] There is no free version of the AAC specification available online.                         |
    +--------------------------------------------------------------------------------------------------+


Examples
--------

* Get mono audio from left channel of a WAV file, and dub to video from AVI::

    video = AviSource("c:\video.avi")
    stereo = WavSource("c:\audio2ch.wav")
    mono = GetLeftChannel(stereo)
    return AudioDub(video, mono)

* Do the same, alternate syntax::

    video = AviSource("c:\video.avi")
    stereo = WavSource("c:\audio2ch.wav")
    mono = GetChannel(stereo, 1)
    return AudioDub(video, mono)

* Do the same with the audio channels from the AVI file itself::

    video = AviSource("c:\video+audio2ch.avi")
    return GetChannel(video, 1)

* | Convert AVI with 5.1 audio to stereo (copy *front left* and *front right*)
  | (But see `Remarks`_ for channel ordering, and examples below and `here`_ for
    more complex downmix functions.)

  ::

    video = AviSource("c:\video+audio6ch.avi")
    stereo = GetChannel(video, 1, 2)
    return AudioDub(video, stereo)

* Get *front left* and *front right* audio from a 5.1 AVI::

    video = AviSource("c:\video+audio6ch.avi")
    audio = WavSource("c:\video+audio6ch.avi")
    stereo = GetChannel(audio, 1, 2)
    return AudioDub(video, stereo)

* | Mix 5.1 to stereo - see `discussion here`_.
  | Note returned audio has sample type `Float`_.
  | :doc:`Normalize <normalize>` is recommended before
    :doc:`converting to 16-bit <convertaudio>` to avoid possible overload.

  ::

    AviSource("c:\video+audio6ch.avi")
    DownMix()
    Normalize()
    ConvertAudioTo16bit()
    return Last

    function DownMix(clip a, float "centergain", float "surroundgain")
    {
        a.ConvertAudioToFloat()

        ## 5.1 WAV channel layout:
        fl = GetChannel(1)
        fr = GetChannel(2)
        fc = GetChannel(3)
        lf = GetChannel(4) ## (LFE not used)
        sl = GetChannel(5)
        sr = GetChannel(6)

        ## add center
        gc = Default(centergain, 1.0) * 0.7071
        fl = MixAudio(fl, fc, 1.0, gc)
        fr = MixAudio(fr, fc, 1.0, gc)

        ## add surround
        gs = Default(surroundgain, 1.0) * 0.7071
        fl = MixAudio(fl, sl, 1.0, gs)
        fr = MixAudio(fr, sr, 1.0, gs)

        return AudioDub(a, MergeChannels(fl, fr))
    }


Changelog
----------

+-----------------+---------------------------------------------------+
| Version         | Changes                                           |
+=================+===================================================+
| AviSynth 2.5.0  | Added GetChannel and GetChannels filters.         |
+-----------------+---------------------------------------------------+
| AviSynth 2.0.3  | Added GetLeftChannel and GetRightChannel filters. |
+-----------------+---------------------------------------------------+

$Date: 2022/03/05 18:47:07 $

.. _5.1 WAV:
    https://web.archive.org/web/20210117100754/http://www.cs.bath.ac.uk/~jpff/NOS-DREAM/researchdev/wave-ex/wave_ex.html
.. _5.1 AC3:
    https://web.archive.org/web/20060212130915/http://www.atsc.org:80/standards/a_52a.pdf
.. _5.1 DTS:
    https://web.archive.org/web/20060909033736/http://webapp.etsi.org:80/action/PU/20020827/ts_102114v010101p.pdf
.. _5.1 AAC:
    https://web.archive.org/web/20080213040722/http://www.hydrogenaudio.org/forums/index.php?showtopic=10986
.. _5.1 AIFF:
    https://web.archive.org/web/20030817071619/http://preserve.harvard.edu/standards/Audio%20IFF%20Specification%201%203.pdf/
.. _5.1 FLAC:
    http://flac.sourceforge.net/format.html
.. _5.1 WMA:
    http://lists.mplayerhq.hu/pipermail/mplayer-users/2006-October/063511.html
.. _NicAudio:
    http://avisynth.nl/index.php/NicAudio
.. _BassAudio:
    http://avisynth.nl/index.php/BassAudio
.. _ffdshow:
    http://avisynth.nl/index.php/Ffdshow
.. _AC3Filter:
    https://web.archive.org/web/20200128173805/http://www.ac3filter.net/wiki/AC3Filter
.. _here:
    https://forum.doom9.org/showthread.php?p=1243880#post1243880
.. _discussion here:
    https://forum.doom9.org/showthread.php?p=1735072#post1735072
.. _Float:
    http://avisynth.nl/index.php/Float
