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


AmplifydB converts the dB value to a linear factor in double precision, rounded to float once. A dB
value from :doc:`AudioStats <audiostats>` gives back the exact factor: these two are the same::

    s = AudioStats(a)
    a.AmplifydB(-s["peak_db"])     # peak to 0 dBFS
    a.Amplify(1.0 / s["peak"])

More examples:

* ``AmplifydB(+3, 0)`` increases the volume of the left channel by a small amount.
* ``Amplify(-1, 1)`` inverts the phase of the left channel. See
  Wikipedia: `Out Of Phase Stereo`_.

**See Also**

* :doc:`Normalize <normalize>` - with *show=true*, can show the maximum
  amplification possible without `clipping`_.
* :doc:`AudioStats <audiostats>` - peak, RMS and other levels as numbers, also per channel.
* How the multichannels are mapped can be found in the description of
  :doc:`GetChannel <getchannel>`.


Changelog
----------

+-----------------+-----------------------------------------+
| Version         | Changes                                 |
+=================+=========================================+
| AviSynth+ 3.7.7 | AmplifydB: the dB to factor conversion  |
|                 | is calculated in double, see the table  |
|                 | below.                                  |
+-----------------+-----------------------------------------+
| AviSynth 2.5.7  | Fixed a small memory leak in Amplify(). |
+-----------------+-----------------------------------------+
| AviSynth 1.0.0  | Added Amplify and AmplifydB filters.    |
+-----------------+-----------------------------------------+

AmplifydB precision, 3.7.7. A sine with the given peak, then ``AmplifydB(-AudioStats(a)["peak_db"])``,
the resulting peak (should be 1.0):

+--------------+---------------------------+--------------------------+
| Sine peak    | 3.7.6 (float pow)         | 3.7.7 (double pow)       |
+==============+===========================+==========================+
| 0.5          | 0.99999994                | 1.0                      |
+--------------+---------------------------+--------------------------+
| 0.3          | 1.00000012 (above full    | 1.0                      |
|              | scale: clipped when       |                          |
|              | converted to integer)     |                          |
+--------------+---------------------------+--------------------------+
| 0.123        | 1.0                       | 1.0                      |
+--------------+---------------------------+--------------------------+

How it works: before, both the dB value and 10\ :sup:`dB/20` were calculated in 32 bit float; the
float ``pow`` result can be one float step off. Now the script value (64 bit) goes into a double
``pow``, and only the factor is rounded to float. Output changes by at most one float step of the
factor.

$Date: 2026/11/14 13:30:00 $

.. _phase:
    http://www.soundonsound.com/sos/apr08/articles/phasedemystified.htm
.. _decibels:
    http://en.wikipedia.org/wiki/Decibel
.. _Out Of Phase Stereo:
    http://en.wikipedia.org/wiki/Out_Of_Phase_Stereo
.. _clipping:
    http://en.wikipedia.org/wiki/Clipping_(audio)
