
KillAudio / KillVideo
=====================

Removes the audio or video from a clip completely. Can be used, if the
destination does not accept an audio or video source, or if AviSynth crashes
when processing audio or video from a clip.


Syntax and Parameters
----------------------

::

    KillAudio (clip)
    KillVideo (clip)

.. describe:: clip

    Source clip.


Examples
--------

``KillAudio()`` can be called at any time to remove audio, and may be chained::

    DeleteFrame(KillAudio(DirectShowSource(clip)), 100)
    
$Date: 2022/02/05 15:10:22 $
