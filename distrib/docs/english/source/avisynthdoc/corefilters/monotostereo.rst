
MonoToStereo
============
.. note::
    Starting from AviSynth v2.5 ``MonoToStereo`` is replaced by 
    :doc:`MergeChannels <mergechannels>`. ``MonoToStereo`` is simply mapped to 
    ``MergeChannels``.

**MonoToStereo** converts two mono signals to one stereo signal. This can be used, 
if one or both channels has been modified separately, and then has to be recombined.

The sample rate of the two clips need to be the same (use 
:doc:`ResampleAudio <resampleaudio>` if this is a problem). If either of the 
sources are in stereo, the signal will be taken from the corresponding channel 
(left channel from the *left_channel_clip*, and vice versa for the right channel). 


Syntax and Parameters
----------------------

::

    MonoToStereo (left_channel_clip, right_channel_clip)

.. describe:: clip

    Source clip; the left channel will be taken from the first clip.

.. describe:: clip

    Source clip; the right channel will be taken from the second clip.


Example
-------

Combines two separate wav sources to a stereo signal::

    left_channel = WavSource("c:\left_channel.wav")
    right_channel = WavSource("c:\right_channel.wav")
    return MonoToStereo(left_channel, right_channel)


$Date: 2022/02/05 21:28:07 $
