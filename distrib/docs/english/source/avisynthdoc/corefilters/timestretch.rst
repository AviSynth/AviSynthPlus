===========
TimeStretch
===========

**TimeStretch** allows changing the sound tempo, `pitch`_ and playback rate
parameters independently from each other, i.e.:

* ``tempo`` adjusts speed while maintaining the original pitch.
* ``pitch`` adjusts speed while maintaining the original tempo.
* ``rate`` adjusts playback rate that affects both tempo and pitch at the same time.

You can use these parameters in any combination – for example, 104% ``tempo``
with 95% ``pitch``.

Syntax and Parameters
---------------------

::

    TimeStretch (clip, float "tempo", float "rate", float "pitch",
                 int "sequence", int "seekwindow", int "overlap", bool "quickseek", int "aa")

    TimeStretch (clip, int "tempo_n", int "tempo_d", int "rate_n", int "rate_d", int "pitch_n", int "pitch_d",
                 int "sequence", int "seekwindow", int "overlap", bool "quickseek", int "aa")

.. describe:: clip

    Source clip; only 32-bit `float`_ audio is supported. Use
    :doc:`ConvertToFloat <convertaudio>` if necessary.

.. _TimeStretch-percentage-float:

.. rubric:: Tempo, Rate and Pitch

.. describe:: tempo

    | Changes speed while maintaining the original `pitch`_.
    | If ``tempo=200``, the audio will play twice (200%) as fast; if ``tempo=50``,
      the audio will play half (50%) as fast.
    | The effect is also known as `time-stretching`_.

    Default: 100.0

.. describe:: rate

    | Changes speed while allowing `pitch`_ to rise or fall, like the traditional
      analog `vari-speed`_ effect.
    | If ``rate=200``, the audio will play twice (200%) as fast; if ``rate=50``,
      the audio will play half (50%) as fast.

    | Rate control is implemented purely by sample rate transposing.
    | If ``rate`` is adjusted by itself, no `time-stretching or pitch-shifting`_
      is performed, and the :ref:`Advanced Parameters <timestretch-advanced>`
      will have no effect.

    Default: 100.0

.. describe:: pitch

    | Changes `pitch`_ while maintaining the original speed (within a small
      tolerance–see `Notes`_ below).
    | If ``pitch=200``, the audio will sound an `octave`_ higher; if ``pitch=50``,
      the audio will sound an octave lower.
    | The effect is also known as `pitch-shifting`_.

    Default: 100.0

``tempo``, ``rate`` and ``pitch`` can all be adjusted independently, in which
case their effects are added together. See section `3.3. About algorithms`_ in
the SoundTouch readme for more information.

.. _TimeStretch-rational-pair:

.. rubric:: Tempo_n, Tempo_d, Rate_n, Rate_d, Pitch_n and Pitch_d

When needing more accuracy you can use the rational pair parameters ``tempo_n``,
``tempo_d``, ``rate_n``, ``rate_d``, ``pitch_n`` and ``pitch_d`` instead. All
parameters are integers and have default value 1. Internally ``tempo`` is
calculated as double(``tempo_n``/``tempo_d``) (``rate`` and ``pitch`` likewise),
before further processing. Seeking should be sample exact.

.. _TimeStretch-advanced:

.. rubric:: Advanced Parameters

The time-stretch algorithm has a few parameters that can be tuned to optimize
sound quality for certain applications. The current default parameters have been
chosen by iterative if-then analysis (read: "trial and error") to obtain the
best subjective sound quality in pop/rock music processing, but in applications
processing different kind of sound the default parameter set may return a
sub-optimal result. See section `3.4. Tuning the algorithm parameters`_ in the
SoundTouch readme for more information.

These parameters affect the time-stretch algorithm as follows:

.. describe:: sequence

    This is the length of a single *processing sequence* in milliseconds, which
    determines how the original sound is chopped in the time-stretch algorithm.
    Larger values mean fewer, and longer, sequences are used. In general,

    * a larger ``sequence`` value sounds better with a lower ``tempo`` and/or
      ``pitch``;
    * a smaller ``sequence`` value sounds better with a higher ``tempo`` and/or
      ``pitch``.

    Default: 100 *

.. describe:: seekwindow

    The length in milliseconds for the algorithm that searches for the best
    possible overlap location. For larger ``seekwindow`` values, the possibility
    of finding a better mixing position increases, but an overly large
    ``seekwindow`` may cause **drifting** (a disturbing artifact where audio
    pitch seems unsteady) because neighboring sequences may be chosen at more
    uneven intervals.

    Default: 22 *

.. describe:: overlap

    The overlap length in milliseconds. When the sound sequences are mixed back
    together to form a continuous sound stream again, ``overlap`` defines how
    much of the ends of the consecutive sequences will be overlapped. This
    shouldn't be a critical parameter. If you reduce the ``sequence`` by a large
    amount, you might wish to try a smaller ``overlap``.

    Default: 8

.. describe:: quickseek

    The time-stretch routine has a 'quick' mode that substantially speeds up the
    algorithm but may degrade the sound quality when ``quickseek`` is set to true.

    * Try ``quickseek=false`` if you hear artifacts like warbling, clicking etc.

    Default: false

.. describe:: aa

    Controls the number of taps the `anti-alias filter`_ uses. Set to 0 to
    disable the filter. Must be a multiple of 4.

    Default: 64

-------------

The table below summarizes how the parameters can be adjusted for different
applications:

.. table::
    :widths: auto

    +----------------+--------------------------+---------------------------+---------------------------+---------------+-----------------+-----------------------------+
    | Parameter      | Default value            | If larger...              | If smaller...             | Music         | Speech          | CPU burden                  |
    +================+==========================+===========================+===========================+===============+=================+=============================+
    | ``Sequence``   | Relatively large, chosen | Usually better for        | Accelerates "echoing"     | Default value | A smaller value | Smaller value increases     |
    |                | for slowing down music   | slowing down tempo.       | artifact when slowing     | usually good. | might be        | CPU burden.                 |
    |                | tempo.                   | tempo. You might need     | down the tempo.           |               | better.         |                             |
    |                |                          | less ``overlap``.         |                           |               |                 |                             |
    |                |                          |                           |                           |               |                 |                             |
    |                |                          |                           |                           |               |                 |                             |
    +----------------+--------------------------+---------------------------+---------------------------+---------------+-----------------+-----------------------------+
    | ``SeekWindow`` | Relatively large, chosen | Eases finding a good      | Makes finding a good      | Default       | Default value   | Larger value increases CPU  |
    |                | for slowing down music   | mixing position, but may  | mixing position more      | usually good, | usually good.   | burden.                     |
    |                | tempo.                   | cause "drifting"          | difficult.                | unless        |                 |                             |
    |                |                          | artifact.                 |                           | "drifting"    |                 |                             |
    |                |                          |                           |                           | is a problem. |                 |                             |
    |                |                          |                           |                           |               |                 |                             |
    +----------------+--------------------------+---------------------------+---------------------------+---------------+-----------------+-----------------------------+
    | ``Overlap``    | Relatively large, chosen |                           |                           |               |                 | Larger value increases CPU  |
    |                | to suit with above       |                           |                           |               |                 | burden.                     |
    |                | parameters.              |                           |                           |               |                 |                             |
    |                |                          |                           |                           |               |                 |                             |
    +----------------+--------------------------+---------------------------+---------------------------+---------------+-----------------+-----------------------------+

\* ``sequence`` and ``seekwindow`` have default values 100 and 22. However they
are updated if the calculated tempo is different from the default value (100).
The calculated tempo depends on the specified tempo or pitch in your script. It
will be different from 100 if tempo or pitch in your script is different from
100. The update of the default values happens in TDStretch::calcSeqParameters().


Notes
-----

* Since ``tempo``, ``rate`` and ``pitch`` are floating-point values, but sample
  rates are integers, rounding effects in calculations are unavoidable; the
  resulting audio track duration may be off by up to several 10's of milliseconds
  (less than one video frame) per hour.

  Pitch is also rounded for the same reason, but the amount is so small that the
  effect is inaudible: according to Wikipedia, the `just-noticeable pitch
  difference`_ is 0.1%–0.6%, while the rounding error is about 0.002%.

  Use the :ref:`rational pair parameters <TimeStretch-rational-pair>` if greater
  accuracy is required.

* In AviSynth+ r2003 and greater, an updated SoundTouch library is used which
  supports multichannel audio (added in v1.8.0). Versions prior to AviSynth+
  r2003 or AviSynth v2.6.0 supports stereo only. If the source clip has two audio
  channels, special processing is used to preserve `stereo imaging`_. Otherwise,
  channels are processed independently. Independent processing works well for
  unrelated audio tracks, but not very well for `surround sound`_. See the thread
  `TimeStretch in AviSynth 2.5.5 Alpha - Strange stereo effects?`_ for details.


Examples
--------

* Raise pitch one `octave`_, without changing speed::

    TimeStretch(pitch=200.0)
    # TimeStretch(pitch_n=2, pitch_d=1) # more accurate processing

* Raise pitch one `semitone`_, without changing speed::

    delta_pitch=1.0 ## (semitones)
    TimeStretch(pitch=100.0*pow(2.0, delta_pitch/12.0))

* Raise playback tempo from NTSC Film speed (23.97 fps) to PAL speed (25 fps)
  without changing pitch::

    TimeStretch(tempo=100.0*25.0/(24000.0/1001.0))

* Increase speed to 105%, allowing pitch to rise::

    TimeStretch(rate=105)

 ...which is equivalent to::

    ar=AudioRate
    AssumeSampleRate(Round(ar*1.05))
    ResampleAudio(ar)


Credits
-------

**TimeStretch** uses the *SoundTouch Audio Processing Library*

    | Copyright © Olli Parviainen
    | SoundTouch home page: http://www.surina.net/soundtouch/


Changelog
---------

+------------------+---------------------------------------------------------+
| Version          | Changes                                                 |
+==================+=========================================================+
| AviSynth+ 3.7.3  || Update SoundTouch library to v2.3.1.                   |
|                  || Add TimeStretch overload with rational pair arguments. |
+------------------+---------------------------------------------------------+
| AviSynth+ 3.4.0  | Update SoundTouch library to  v2.1.3 (Jan 07, 2019).    |
+------------------+---------------------------------------------------------+
| AviSynth+ r2003  || Merge TimeStretch changes from AviSynth 2.6.1          |
|                  || Update SoundTouch library to v1.9.2 - fixes            |
|                  |  multichannel issues.                                   |
+------------------+---------------------------------------------------------+
| AviSynth+ <r1555 || Sep 15, 2013                                           |
|                  || Put TimeStretch function into its own plugin.          |
|                  || TimeStretch no longer automatically converts input     |
|                  |  clip to Float.                                         |
|                  || Update SoundTouch library to v1.7.1, results in        |
|                  |  audible quality improvement for TimeStretch.           |
+------------------+---------------------------------------------------------+
| AviSynth 2.5.7   || Expose SoundTouch adavanced parameters.                |
|                  || Update SoundTouch library to v1.3.1.                   |
+------------------+---------------------------------------------------------+
| AviSynth 2.5.5   | Initial Release (based on SoundTouch library v1.2.1).   |
+------------------+---------------------------------------------------------+


$Date: 2022/03/22 16:46:19 $

.. _pitch:
    https://en.wikipedia.org/wiki/Pitch_(music)
.. _float:
    http://avisynth.nl/index.php/Float
.. _time-stretching:
    https://en.wikipedia.org/wiki/Audio_time_stretching_and_pitch_scaling
.. _vari-speed:
    https://en.wikipedia.org/wiki/Pitch_control
.. _time-stretching or pitch-shifting:
    https://en.wikipedia.org/wiki/Audio_time_stretching_and_pitch_scaling
.. _octave:
    https://en.wikipedia.org/wiki/Octave
.. _3.3. About algorithms:
    http://www.surina.net/soundtouch/README.html
.. _3.4. Tuning the algorithm parameters:
    http://www.surina.net/soundtouch/README.html
.. _anti-alias filter:
    https://en.wikipedia.org/wiki/Finite_impulse_response
.. _octave:
    https://en.wikipedia.org/wiki/Octave
.. _pitch-shifting:
    https://en.wikipedia.org/wiki/Audio_time_stretching_and_pitch_scaling
.. _just-noticeable pitch difference:
    https://en.wikipedia.org/wiki/Pitch_(music)#Just-noticeable_difference
.. _stereo imaging:
    https://en.wikipedia.org/wiki/Stereo_imaging
.. _surround sound:
    https://en.wikipedia.org/wiki/5.1_surround_sound
.. _TimeStretch in AviSynth 2.5.5 Alpha - Strange stereo effects?:
    http://forum.doom9.org/showthread.php?t=71632
.. _semitone:
    https://en.wikipedia.org/wiki/Semitone
