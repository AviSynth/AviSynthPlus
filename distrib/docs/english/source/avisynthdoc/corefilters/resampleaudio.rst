
ResampleAudio
=============

High-quality audio sample rate converter:

* Accepts any number of channels.
* The conversion is skipped if the sample rate is already at the given rate.
* Supports fractional resampling (where *new_rate_denominator* ≠ 1).
* Note, internal rounding may affect 
  :doc:`AudioDuration <../syntax/syntax_clip_properties>` slightly – see 
  `Examples`_ below. 

See Wikipedia: `Sample-rate conversion`_

Syntax and Parameters
----------------------

::

    ResampleAudio (clip, int new_rate_numerator, int "new_rate_denominator")

.. describe:: clip

    | Source clip. Supported audio sample types: 16-bit integer and 32-bit float. 
    | Other sample types (8-, 24- and 32-bit integer) are automatically 
      :doc:`converted <convertaudio>` to 32-bit float.

.. describe:: new_rate_numerator

    Set the numerator for the new sample rate. 

.. describe:: new_rate_denominator

    Set the denominator for the new sample rate.

    Default: 1 


Examples
--------

* Resample audio to 48 kHz::

    source = AviSource("c:\audio.wav")
    return ResampleAudio(source, 48000)

* Exact 4% speed up for Pal telecine::

    nfr_num = 25
    nfr_den = 1
    AviSource("C:\Film.avi") # 23.976 fps, 44100Hz
    ar = Audiorate()
    # intermediate sample rate:
    ResampleAudio(ar*FramerateNumerator*nfr_den, FramerateDenominator*nfr_num)
    # final sample rate:
    AssumeSampleRate(ar)
    AssumeFPS(nfr_num, nfr_den, sync_audio=False)

 In the example above, the intermediate sample rate needs to be:

    | ``(AudioRate*FramerateNumerator*nfr_den=1)/(FramerateDenominator*nfr_num=25)``
    | or (44100 * 24000 * 1) / (1001 * 25) = 42293.706294... 

 But because audio sample rates are always integers, 42293.706294 must be 
 rounded to 42294, which results in a time slippage of about 30ms per hour.


Changelog
---------

+----------------+----------------------------------------------------------+
| Version        | Changes                                                  |
+================+==========================================================+
| AviSynth 2.5.6 || Added Float support in ResampleAudio().                 |
|                || Added Fractional resampling support in ResampleAudio(). |
+----------------+----------------------------------------------------------+
| AviSynth 2.5.3 | ResampleAudio now accepts any number of channels.        |
+----------------+----------------------------------------------------------+

$Date: 2022/02/13 11:10:51 $

.. _Sample-rate conversion:
    https://en.wikipedia.org/wiki/Sample-rate_conversion
