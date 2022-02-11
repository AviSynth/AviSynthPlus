
DelayAudio
==========

**DelayAudio** delays the audio track by seconds.

Syntax and Parameters
----------------------

::
    
    DelayAudio (clip, float seconds)

.. describe:: clip

    Source clip.
    
.. describe:: seconds

    Delay time, in seconds. Seconds can be negative and/or have a fractional part.


Examples
---------

* Play audio half a second earlier::

    DelayAudio(-0.5)

 PS: if audio is loaded from separate file (for example :doc:`WavSource <avisource>`), 
 it's better do the negative delay after AudioDub, or else it may cause problems.

* When demuxing audio streams with `DGIndex`_ or `DGDecNV`_ , the delay 
  (actually how the delay should be corrected) is written into the name of the 
  demuxed audio stream. For example::

    vid = MPEG2Source("movie.d2v")
    aud = FFAudioSource("movie T01 2_0ch 448Kbps DELAY -218ms.ac3")
    AudioDub(vid, aud)
    DelayAudio(-0.218)

$Date: 2022/02/04 22:44:06 $

.. _DGIndex:
    http://avisynth.nl/index.php/DGDecode
.. _DGDecNV:
    http://avisynth.nl/index.php/DGDecNV
