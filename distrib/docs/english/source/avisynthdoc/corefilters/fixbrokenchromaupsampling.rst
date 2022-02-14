
FixBrokenChromaUpsampling
=========================

The free Canopus DV Codec v1.00 upsamples the chroma channels incorrectly
(although newer non-free versions appear to work fine). Chroma is `duplicated
from the other field`_, resulting in the famous :doc:`Chroma Upsampling Error <../advancedtopics/sampling>`.

**FixBrokenChromaUpsampling** filter compensates for it. You should put this
after :doc:`AviSource <avisource>` if you're using the above Canopus DV codec. 
Old versions of the DirectShow based MS DV codec also might have this problem 
(the one that comes with DirectX7 (but i need to check this), the one that comes 
with DirectX8/9 works fine).

The Canopus DV codec swaps the chroma of the middle 2 for each group of 4
lines:

+---------------+---------------+
| frame_correct | frame_Canopus |
+===============+===============+
| line 1        | line 1        |
+---------------+---------------+
| line 2        | line 3        |
+---------------+---------------+
| line 3        | line 2        |
+---------------+---------------+
| line 4        | line 4        |
+---------------+---------------+
| line 5        | line 5        |
+---------------+---------------+
| line 6        | line 7        |
+---------------+---------------+
| line 7        | line 6        |
+---------------+---------------+
| line 8        | line 8        |
+---------------+---------------+

For each group of 4 lines **FixBrokenChromaUpsampling** corrects this by
swapping the chroma of the middle 2 back.


Syntax and Parameters
----------------------

::

    FixBrokenChromaUpsampling (clip)

.. describe:: clip

    Source clip; only YUY2 color format supported.

$Date: 2022/02/13 12:37:33 $

.. _duplicated from the other field:
    http://forum.doom9.org/showthread.php?p=180052#post180052
