
FlipHorizontal / FlipVertical
=============================

**FlipHorizontal** flips (or mirrors) the video left to right.

**FlipVertical** flips the video top to bottom.

| It is useful for dealing with some video codecs which return an upside-down image.
| It doesn't modify the interlaced `parity flags`_. 


Syntax and Parameters
----------------------

::

    FlipHorizontal (clip)
    FlipVertical (clip)

.. describe:: clip

    Source clip.

.. note::
    Using ``FlipHorizontal()`` followed by ``FlipVertical()`` is equivalent to 
    :doc:`Turn180() <turn>`.


Changelog
----------

+----------------+----------------------------------------------------+
| Version        | Changes                                            |
+================+====================================================+
| AviSynth 2.5.6 | RGB32 FlipHorizontal() code tweaked.               |
+----------------+----------------------------------------------------+
| AviSynth 2.5.3 | Fixed YUY2 FlipHorizontal giving garbage/crashing. |
+----------------+----------------------------------------------------+
| AviSynth 2.5.0 | Fliphorizontal implemented.                        |
+----------------+----------------------------------------------------+

$Date: 2022/02/05 22:44:06 $

.. _parity flags:
    http://avisynth.nl/index.php/Interlaced_fieldbased#The_parity_.28.3D_order.29_of_the_fields_in_AviSynth
