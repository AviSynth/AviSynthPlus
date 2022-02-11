
ConvertAudio
============

Set of filters to convert the audio sample type to 8, 16, 24, 32 integer bits, 
and Float. 

See Wikipedia: `audio bit depth`_


Syntax and Parameters
----------------------

::

    ConvertAudioTo8bit (clip)
    ConvertAudioTo16bit (clip)
    ConvertAudioTo24bit (clip)
    ConvertAudioTo32bit (clip)
    ConvertAudioToFloat (clip)

.. describe:: clip

    Source clip to convert; all sample types supported.

.. note::
    The sample types supported by the AviSynth audio filters are listed `here`_, 
    in the **Sample type** column. If a filter doesn't support the type of 
    sample it is given, it will throw an error. Use **ConvertAudio** to convert 
    the audio clip to the supported format.


Changelog
----------

.. table::
    :widths: 20 80

    +------------------+------------------------------------------------------------+
    | Version          | Changes                                                    |
    +==================+============================================================+
    | AviSynth+ 3.7.1  || Fix: ConvertAudio integer 32-to-8 bits C code garbage     |
    |                  |  (regression in 3.7.0).                                    |
    |                  || Fix: ConvertAudio: float to 32 bit integer conversion max |
    |                  |  value glitch (regression in 3.7.0).                       |
    |                  || Add direct Float from/to 8/16 conversions (C,SSE2,AVX2)   |
    +------------------+------------------------------------------------------------+
    | AviSynth+ 3.7.0  |  Internally refactored ConvertAudio.                       |
    +------------------+------------------------------------------------------------+

$Date: 2022/02/08 22:00:52 $

.. _audio bit depth:
    https://en.wikipedia.org/wiki/Audio_bit_depth
.. _here:
    http://avisynth.nl/index.php/Internal_filters#Audio_processing_filters
