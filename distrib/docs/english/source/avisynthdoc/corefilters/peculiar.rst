
PeculiarBlend
=============

This filter blends each frame with the following frame in a peculiar way.


Syntax and Parameters
----------------------

::

    PeculiarBlend (clip, int cutoff)

.. describe:: clip

    Source clip; only YUY2 is supported.

.. describe:: cutoff

    The portion of the frame below scanline number ``cutoff`` is unchanged. 
    The portion above scanline number (``cutoff``-30) is replaced with the 
    corresponding portion of the following frame. The 30 scan lines in 
    between are blended incrementally to disguise the switchover.
    
    You're probably wondering why anyone would use this filter. Well, it's 
    like this. Most videos which were originally shot on film use the 
    3:2 pulldown technique which is described in the description of the 
    :doc:`Pulldown <pulldown>` filter. But some use a much nastier system 
    (`involving a camera pointed at a movie screen`_) in which the crossover 
    to the  next frame occurs in the middle of a field--in other words, 
    individual  fields look like one movie frame on the top, and another on 
    the bottom. This filter partially undoes this peculiar effect. It should 
    be used before :doc:`Pulldown <pulldown>`. To determine ``cutoff``, 
    examine a frame which is blended in this way and set it to the number of 
    the first scanline in which you notice a blend.


$Date: 2022/02/08 21:28:07 $

.. _involving a camera pointed at a movie screen:
    http://forum.doom9.org/showthread.php?p=455092&highlight=PeculiarBlend#post455092
