
AlignedSplice / UnalignedSplice
===============================

**AlignedSplice** and **UnalignedSplice** join two or more video clips end to
end. The difference between the filters lies in the way they treat the sound
track. 

* **AlignedSplice** cuts off the first sound track or inserts silence as 
  necessary to ensure that the second sound track remains synchronized with the 
  video.

* **UnalignedSplice** simply concatenates the sound tracks without regard to 
  synchronization with the video.  

You should use **AlignedSplice** for most situations.

You should use **UnalignedSplice** when the soundtracks being joined were 
originally contiguous—for example, when you're joining files captured with 
`AVI-IO`_. Using **AlignedSplice** in these situations may lead to glitches in 
the sound.

AviSynth provides ``++`` and ``+`` :doc:`operators <../syntax/syntax_operators>` 
as synonyms for **AlignedSplice** and **UnalignedSplice** respectively. 


Syntax and Parameters
----------------------

::

    AlignedSplice (clip1, clip2 [, ...])
    UnAlignedSplice (clip1, clip2 [, ...])

.. describe:: clip1, clip2, ...

    Source clips; the :doc:`media properties <../syntax/syntax_clip_properties>` 
    must be compatible, meaning all clips must have:
    
    #. the same height and width,
    #. the same :doc:`color format <convert>`,
    #. the same frame rate, and
    #. the same audio sample rate, bit depth and number of channels.
   
    See :ref:`here <multiclip>` for the resulting clip properties.

Examples
--------

Join segmented capture files (with UnalignedSplice) to produce a single clip::

    UnalignedSplice(AVISource("cap1.avi"),AVISource("cap2.avi"),AVISource("cap3.avi"))
    # or:
    AVISource("cap1.avi") + AVISource("cap2.avi") + AVISource("cap3.avi")

Extract three scenes from a clip and join them together in a new order with 
AlignedSplice::

    AVISource("video.avi")
    Trim(2000,2500) ++ Trim(3000,3500) ++ Trim(1000,1500)


$Date: 2022/02/06 21:28:07 $

.. _AVI-IO:
    http://www.avi-io.com/
