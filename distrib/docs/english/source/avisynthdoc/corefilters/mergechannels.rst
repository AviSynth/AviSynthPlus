
MergeChannels
=============

Merge the audio channels of two or more clips.

There is no *mixing* of channels – :doc:`MixAudio <mixaudio>` and 
:doc:`ConvertToMono <converttomono>` do this. The channels are added to the 
new clip unchanged.


Syntax and Parameters
----------------------

::

    MergeChannels (clip1 , clip2 [, clip3, ...])

.. describe:: clip1, clip2, ...

    | Source clips; a minimum of 2 are required.
    | Output video, framerate and running time are taken from clip1.
    | All audio is :doc:`converted <convertaudio>` to the sample type of clip1. 


Examples
--------

For example, given the following source clips:

    | A (mono) 
    | B (mono) 

...and merging them::

    MergeChannels(A, B)

Results in a clip with the following :doc:`properties <../syntax/syntax_clip_properties>`:

    | video = A 
    | audio channel 1 = A 
    | audio channel 2 = B 

This is equivalent to using :doc:`MonoToStereo <monotostereo>`.

-------------

For a more complex example, given the following source clips:

    | A and B (stereo) 
    | C and D (mono) 

...and merging them::

    MergeChannels(A, B, C, D)

Results in a clip with the following :doc:`properties <../syntax/syntax_clip_properties>`:

    | video = A 
    | audio channel 1 = A channel 1 
    | audio channel 2 = A channel 2 
    | audio channel 3 = B channel 1 
    | audio channel 4 = B channel 2 
    | audio channel 5 = C 
    | audio channel 6 = D 

$Date: 2022/02/08 21:28:07 $
