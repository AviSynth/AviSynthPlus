
SSRC
====

**SSRC** (Shibata Sample Rate Converter) is a High quality audio sample rate 
converter by `Naoki Shibata`_. This filter should result in better audio quality 
than :doc:`ResampleAudio <resampleaudio>` when converting between 44100↔48000 Hz.

    “44.1kHz sampling rate is used for a CD, and 48kHz is used for a DVD... 
    Converting between these frequencies is hard, because the ratio between 
    these two frequencies is 147:160, which are not small numbers. Accordingly, 
    we need a very long FIR filter in order not to degrade the sound quality 
    during conversion. This program utilizes FFTs to apply the FIR filter in 
    order to reduce the amount of computation.” `[1]`_

Unlike :doc:`ResampleAudio <resampleaudio>`, **SSRC** doesn't work for arbitrary 
sample rate ratios. If it cannot handle resampling between the two sample rates, 
an error is raised.


Syntax and Parameters
----------------------

::

    SSRC (clip, int samplerate, bool "fast")

.. describe:: clip

    Source clip. Supported audio sample types: 32-bit float. Use 
    :doc:`ConvertToFloat <convertaudio>` if necessary.

.. describe:: samplerate

    Target sample rate.

.. describe:: fast

    | Enable faster processing at slightly lower quality. 
    | Set this to *false* if you are doing large-ratio rate conversions (more than 
      a factor 2). 

    Default: True


Examples
--------

Downsampling to 44,1 kHz::

    AviSource("c:\file.avi") # Has 48000 audio
    SSRC(44100)


Changelog
---------

+------------------+----------------------------------------------------+
| Version          | Changes                                            |
+==================+====================================================+
| AviSynth+ <r1555 || Convert SSRC into a proper plugin (Shibatch.dll). |
|                  || SSRC no longer automatically converts input clip  |
|                  |  to Float. (2013/09/15)                            |
+------------------+----------------------------------------------------+
| AviSynth 2.5.4   | Initial Release                                    |
+------------------+----------------------------------------------------+

| Some parts of SSRC are:
| Copyright © 2001-2003, Peter Pawlowski. All rights reserved.

$Date: 2022/02/13 14:59:42 $

.. _Naoki Shibata:
    http://shibatch.sourceforge.net/
.. _[1]:
    http://shibatch.sourceforge.net/
