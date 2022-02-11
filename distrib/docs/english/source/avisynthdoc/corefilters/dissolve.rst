
Dissolve
========

**Dissolve** is like :doc:`AlignedSplice <splice>`, except that the clips are 
combined with some overlap. The last *overlap* frames of the first video stream 
are blended progressively with the first *overlap* frames of the second video
stream so that the streams fade into each other. The audio streams are
blended similarly.

The term "dissolve" is sometimes used for a different effect in which the
transition is pointwise rather than gradated. This filter won't do that.


Syntax and Parameters
----------------------

::

    Dissolve (clip1, clip2 [, ...], int overlap, float "fps")

.. describe:: clip1, clip2, ...

    Source clips; the dimensions of the clips and their color formats must be 
    the same. If the clips contain audio, the audio properties must also be the 
    same. The framerate is taken from the first clip, - see :ref:`here <multiclip>` 
    for the resulting :doc:`clip properties <../syntax/syntax_clip_properties>`.

.. describe:: overlap

    How many frames to overlap.

.. describe:: fps

    The ``fps`` parameter is optional and provides a reference for ``overlap`` 
    in audio only clips. It is ignored if a video stream is present. Set 
    ``fps=AudioRate()`` if sample exact audio positioning is required.

    Default: 24.0


Examples
--------

::

     # Load sources
     a = AVISource("clipA.avi")
     b = AVISource("clipB.avi")
     
     # 30 frame Dissolve between clipA & clipB; last & clipA; last & clipB
     # Resulting output: clipA (Dissolve) clipB (Dissolve) clipA (Dissolve) clipB.
     Dissolve(a, b, 30) 
     Dissolve(last, a, 30)
     Dissolve(last, b, 30)


Changelog
----------

.. table::
    :widths: 45 55
    
    +----------------+--------------------------+
    | Version        | Changes                  |
    +================+==========================+
    | AviSynth 2.5.6 | Added ``fps`` parameter. |
    +----------------+--------------------------+


$Date: 2022/02/06 12:22:43 $
