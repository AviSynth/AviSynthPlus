===================
Amplify / AmplifydB
===================

`Amplify`_ and `AmplifydB`_ amplify the audio of the clip by the prescribed
amount. You can specify different *amount* arguments for each channel. If there
are more arguments than there are channels, the extra ones are ignored. If there
are fewer arguments than channels, the last one is applied to the rest of the
channels.

.. _Amplify:

Amplify
-------

Amplify the audio in linear values. See the `Examples`_ section for more
information.

.. rubric:: Syntax and Parameters

::

    Amplify (clip, float amount [, float amount ]...)

.. describe:: clip

    | Source clip. Supported audio sample types: 16/32-bit integer, and 32-bit
      float.
    | Other sample types (8 and 24-bit integer) are automatically
      :doc:`converted <convertaudio>` to 32-bit float.

.. describe:: amount

    Multiply (scale) the audio by ``amount``; values are linear:

    * ``amount`` > 1.0 increases volume
    * ``amount`` < 1.0 decreases volume
    * ``amount`` = 1.0 retains the same volume
    * Negative scale factors will shift the `phase`_ by 180 degrees (i.e. invert
      the samples).

.. _AmplifydB:

AmplifydB
---------

Amplify the audio in `decibels`_. See the `Examples`_ section for more
information.

.. rubric:: Syntax and Parameters

::

    AmplifydB (clip, float amount [, float amount ]...)

.. describe:: clip

    | Source clip. Supported audio sample types: 16/32-bit integer, and 32-bit
      float.
    | Other sample types (8 and 24-bit integer) are automatically
      :doc:`converted <convertaudio>` to 32-bit float.

.. describe:: amount

    Multiply (scale) the audio by ``amount``; values are in dB (decibels):

    * ``amount`` > 0.0 increases volume
    * ``amount`` < 0.0 decreases volume
    * ``amount`` = 0.0 retains the same volume


Examples
--------

The relation between *linear* and *decibel* gain control is:

    **lin** = 10 :sup:`(dB/20)`

    **dB** = 20·log\ :sub:`10` \(**lin**)

For example,

* ``AmplifydB(  0.0)`` is equivalent to  ``Amplify(1.0)``  (no change)
* ``AmplifydB(  6.0)`` is equivalent to  ``Amplify(2.0)``  (about twice as loud)
* ``AmplifydB( -6.0)`` is equivalent to  ``Amplify(0.5)``  (about half as loud)
* ``AmplifydB( 20.0)`` is equivalent to  ``Amplify(10)``   (much louder)
* ``AmplifydB(-20.0)`` is equivalent to  ``Amplify(0.1)``  (much softer)


More examples:

* ``AmplifydB(+3, 0)`` increases the volume of the left channel by a small amount.
* ``Amplify(-1, 1)`` inverts the phase of the left channel. See
  Wikipedia: `Out Of Phase Stereo`_.

**See Also**

* :doc:`Normalize <normalize>` - with *show=true*, can show the maximum
  amplification possible without `clipping`_.
* How the multichannels are mapped can be found in the description of
  :doc:`GetChannel <getchannel>`.


Changelog
----------

+-----------------+-----------------------------------------+
| Version         | Changes                                 |
+=================+=========================================+
| AviSynth 2.5.7  | Fixed a small memory leak in Amplify(). |
+-----------------+-----------------------------------------+
| AviSynth 1.0.0  | Added Amplify and AmplifydB filters.    |
+-----------------+-----------------------------------------+

$Date: 2022/03/05 15:10:22 $

.. _phase:
    http://www.soundonsound.com/sos/apr08/articles/phasedemystified.htm
.. _decibels:
    http://en.wikipedia.org/wiki/Decibel
.. _Out Of Phase Stereo:
    http://en.wikipedia.org/wiki/Out_Of_Phase_Stereo
.. _clipping:
    http://en.wikipedia.org/wiki/Clipping_(audio)
