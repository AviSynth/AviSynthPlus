==============
Parity Filters
==============

AviSynth includes these filters to change the parity state of a clip:

* `AssumeFrameBased / AssumeFieldBased`_ forces frame-based or field-based material.
* `AssumeTFF / AssumeBFF`_ forces field order.
* `ComplementParity`_ changes top fields to bottom fields and vice-versa.

.. _AssumeFrameBased:
.. _AssumeFieldBased:
.. _AssumeFrameField:

AssumeFrameBased / AssumeFieldBased
-----------------------------------

AviSynth keeps track of whether a given clip is *field-based* or *frame-based*.
If the clip is field-based it also keeps track of the *parity* of each field –
that is, whether it's the top or the bottom field. If the clip is frame-based
it keeps track of the *dominant field* in each frame – that is, which field in
the frame comes first when they are separated.

However, this information isn't necessarily correct, because field information
usually isn't stored in video files and AviSynth's source filters just guess at
it. **AssumeFrameBased** and **AssumeFieldBased** let you tell AviSynth the
correct type of a clip. See this `Doom9 thread`_ for a bit more information.

**AssumeFrameBased** throws away the existing information and assumes that the
clip is frame-based, with the bottom (even) field dominant in each frame. This
happens to be what the source filters guess. If you want the top field dominant,
use `ComplementParity`_ afterwards.

**AssumeFieldBased** throws away the existing information and assumes that the
clip is field-based, with the even-numbered fields being bottom fields and the
odd-numbered fields being top fields. If you want it the other way around,
use `ComplementParity`_ afterwards.

.. rubric:: Syntax and Parameters

::

    AssumeFrameBased (clip)
    AssumeFieldBased (clip)

.. describe:: clip

    Source clip; all color formats supported.

.. _AssumeBFF:
.. _AssumeTFF:
.. _AssumeFieldFirst:

AssumeTFF / AssumeBFF
---------------------
AviSynth keeps track of whether a given clip is *field-based* or *frame-based*.
If the clip is field-based it also keeps track of the *parity* of each field –
that is, whether it's the top or the bottom field. If the clip is frame-based
it keeps track of the *dominant field* in each frame – that is, which field in
the frame comes first when they're separated.

However, this information isn't necessarily correct, because field information
usually isn't stored in video files and AviSynth's source filters just normally
default to assuming bottom field first (with the exception of the `MPEG2Source`_
plugin, which gets it right!). **AssumeTFF** and **AssumeBFF** let you tell
AviSynth what you believe the field order a clip has.

**AssumeTFF** sets the field order to Top Field First and **AssumeBFF** to
Bottom Field First. They do not change the actual field order, just the internal
state flags in AviSynth relating to the source clip used.

.. rubric:: Syntax and Parameters

::

    AssumeTFF (clip)
    AssumeBFF (clip)

.. describe:: clip

    Source clip; all color formats supported.

--------------

.. rubric:: Examples

::

    AviSource("test.avi")
    AssumeTFF()
    SeparateFields()

Will always return the top field of the first frame followed by the bottom
field of the first frame and so on.

::

    AviSource("test.avi")
    AssumeBFF()
    SeparateFields()

Will always return the bottom field of the first frame followed by the top
field of the first frame and so on.

.. _ComplementParity:

ComplementParity
----------------

If the input clip is field-based, **ComplementParity** changes top fields to
bottom fields and vice-versa. If the input clip is frame-based, it changes
each frame's dominant field (bottom-dominant to top-dominant and vice-versa).

It does not change the actual field order, just the internal state flags in
AviSynth relating to the source clip used.

.. rubric:: Syntax and Parameters

::

    ComplementParity (clip)

.. describe:: clip

    Source clip; all color formats supported.


Changelog
----------

.. table::
    :widths: auto

    +------------------+----------------------------------------------------+
    | Version          | Changes                                            |
    +==================+====================================================+
    | AviSynth 2.0.3   | Added AssumeBFF / AssumeTFF filters.               |
    +------------------+----------------------------------------------------+
    
$Date: 2022/03/17 21:41:11 $

.. _Doom9 thread:
    https://forum.doom9.org/showthread.php?t=150472
.. _MPEG2Source:
    http://avisynth.nl/index.php/DGDecode/MPEG2Source
