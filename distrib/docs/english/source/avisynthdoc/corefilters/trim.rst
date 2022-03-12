================
Trim / AudioTrim
================

Set of filters to trim audio and video:

* `Trim`_ can trim video only clips and also clips with both audio and video.
* `AudioTrim`_ can trim audio only clips and also clips with both audio and video.

.. _Trim:

Trim
----

*Trims* a clip – removes frames from one or both ends.

**Trim** cannot trim a clip if there is no video. Use `AudioTrim`_ for that case.

The resulting clip starts with frame number 0, and this must be taken into
account when making additional edits using that clip. To view a clip's frame
number at any point in your script, temporarily insert an :doc:`Info <info>`
or :ref:`ShowFrameNumber` statement.

.. rubric:: Syntax and Parameters

For convenience, **Trim** can be called in four slightly different ways:

::

    Trim (clip, int first_frame, int last_frame , bool "pad", bool "cache")

.. describe:: clip

    Source clip; all color formats and audio sample types supported.

.. describe:: first_frame, last_frame

    Returns a clip starting at ``first_frame`` and running up to and including
    ``last_frame``.

    * For example, ``Trim(3, 5)`` returns source clip frames 3, 4 and 5.
    * Remember, AviSynth starts counting at frame 0.
    * If you set ``last_frame`` to 0, you will get a clip starting at
      ``first_frame`` and running to the end of the clip.

.. describe:: pad

    ``pad`` causes the audio stream to be padded to align with the video stream.
    Otherwise the tail of a short audio stream is left so. You should use
    ``pad=false`` when the soundtracks being joined were originally contiguous –
    compare to :doc:`UnalignedSplice <splice>`.

    Default: true

.. describe:: cache

    When set to false: lower memory consumption but may be slower. Benefits
    heavily depend on how trimmed clips are used later. See `issue #274`_ for
    more information.

    Default: true

------------------

::

    Trim (clip, int first_frame, int -num_frames , bool "pad", bool "cache")

.. describe:: first_frame, -num_frames

    With a negative value for the second argument, you get a clip starting at
    ``first_frame`` and running for ``num_frames`` frames.

    * For example, ``Trim(0, -4)`` returns source clip frames 0, 1, 2 and 3.

------------------

::

    Trim (clip, int first_frame, int "end" , bool "pad", bool "cache")

.. describe:: first_frame, end

    Returns a clip starting at ``first_frame`` and running up to and including
    frame ``end``.

    * For example, ``Trim(3, end=7)`` is equivalent to ``Trim(3, 7)``; both
      return frames 3, 4, 5, 6 and 7.
    * ``end`` default = 0; must be >= ``first_frame``.
    * Unlike the ``last_frame`` syntax, the ``end`` syntax has no discontinuous
      boundary values: ``end=0`` means end at frame 0. This feature is useful
      in avoiding unexpected boundary conditions in your user functions.

------------------

::

    Trim (clip, int first_frame, int "length" , bool "pad", bool "cache")

.. describe:: first_frame, length

    Returns a clip starting at ``first_frame`` and running for ``length`` frames.

    * For example, ``Trim(3, length=5)`` is equivalent to ``Trim(3, -5)``;
      both return frames 3, 4, 5, 6 and 7.
    * ``length`` default = 0; must be >= 0.
    * Unlike the ``num_frames`` syntax, the ``length`` syntax has no
      discontinuous boundary values: ``length=0`` means return a zero length
      clip. This feature is useful in avoiding unexpected boundary conditions
      in your user functions.

.. _AudioTrim:

AudioTrim
---------

**AudioTrim** trims a clip based on time, not on frames. This is most useful
for audio-only clips, where "frames" have no meaning anyway, and you may want
to edit with finer precision than whole frames (at 30fps, 1 frame=33.3ms).

All time arguments are in seconds, floating-point.

* Trims on audio-only clips are accurate to the nearest audio sample.
* Trims on clips with video are accurate to the nearest whole video frame.

**AudioTrim** cannot trim a clip if there is no audio. Use `Trim`_ for that case.

The resulting clip starts with time = 0.0, and this must be taken into account
when making additional edits to that clip. To view a clip's time at any point
in your script, temporarily insert an :doc:`Info <info>` or :ref:`ShowTime`
statement.

.. rubric:: Syntax and Parameters

For convenience, **AudioTrim** can be called in four slightly different ways:

::

    AudioTrim (clip, float start_time, float end_time, bool "cache")

.. describe:: clip

    Source clip; all color formats and audio sample types supported.

.. describe:: start_time, end_time

    Returns a clip starting at ``start_time`` and running up to and including
    time ``end_time``.

    * For example, ``AudioTrim(3.0, 5.0)`` returns source clip from time
      00:00:03.000 to 00:00:05.000.
    * If you set ``end_time`` to 0.0, you will get a clip starting at
      ``start_time`` seconds and running to the end of the clip.

.. describe:: cache

  When set to false: lower memory consumption but may be slower. Benefits
  heavily depend on how trimmed clips are used later. See `issue #274`_ for
  more information.

  Default: true

------------------

::

    AudioTrim (clip, float start_time, float -duration, bool "cache")

.. describe:: start_time, -duration

    With a negative value for the second argument, you will get a clip
    starting at ``start_time`` and running for ``duration`` seconds.

    * For example, ``AudioTrim(0.0, -4.0)`` returns the source clip from time
      00:00:00.000 to 00:00:04.000.

------------------

::

    AudioTrim (clip, float start_time, float "end", bool "cache")

.. describe:: start_time, end

    Returns a clip starting at ``start_time`` and running up to and including
    time ``end``.

    * For example, ``AudioTrim(3.0, end=7.0)`` is equivalent to
      ``AudioTrim(3.0, 7.0)``
    * ``end`` default = 0.0; must be >= ``start_time``.
    * Unlike the ``end_time`` syntax, the ``end`` syntax has no discontinuous
      boundary values: ``end=0.0`` means return a zero-length clip. This
      feature is useful in avoiding unexpected boundary conditions in your
      user functions.

------------------

::

    AudioTrim (clip, float start_time, float "length", bool "cache")

.. describe:: start_time, length

    Returns a clip starting at ``start_time`` and running for ``length`` seconds.

    * For example, ``AudioTrim(3.0, length=4.0)`` is equivalent to
      ``AudioTrim(3.0, -4.0)``
    * ``length`` default = 0.0; must be >= 0.
    * Unlike the duration syntax, the ``length`` syntax has no discontinuous
      boundary values: ``length=0.0`` means return a zero-length clip. This
      feature is useful in avoiding unexpected boundary conditions in your
      user functions.


Examples
--------

* Return only the first frame (frame 0)::

    Trim(0, -1)
    Trim(0, end=0)
    Trim(0, length=1)

* Return only the last frame::

    Trim(FrameCount, -1)
    Trim(FrameCount, end=Framecount)
    Trim(FrameCount, length=1)

* Return frames 100-199 (duration=100)::

    Trim(100, 199)
    Trim(100, -100)

* Delete the first 100 frames; audio padded or trimmed to match the video
  length. ::

    Trim(100, 0) # (pad=true by default)

* Delete the first 100 frames of audio and video; the resulting stream lengths
  remain independent. ::

    Trim(100, 0, pad=false)

* Trim audio if longer than video (pad does affect this action)::

    Trim(0, FrameCount-1)

* Audio will be trimmed if longer, but not padded if shorter to frame 199::

    Trim(100, 199, pad=false)

* AudioTrim: trim video if longer than audio::

    AudioTrim(0, AudioDuration)

* AudioTrim: keep the audio between 1.0 and 6.5 seconds (ie, delete the first
  second, keep the following 5.5 seconds)::

    AudioTrim(1, 6.5)
    AudioTrim(1, -5.5)
    AudioTrim(1, length=5.5)


Changelog
----------

+-----------------+----------------------------------------------------+
| Version         | Changes                                            |
+=================+====================================================+
| AviSynth+ 3.7.2 | Added ``cache`` parameter.                         |
+-----------------+----------------------------------------------------+
| AviSynth 2.6.0  || Added AudioTrim.                                  |
|                 || Added explicit ``length`` and ``end`` parameters. |
+-----------------+----------------------------------------------------+
| AviSynth 2.5.6  | Added ``pad`` audio parameter.                     |
+-----------------+----------------------------------------------------+

$Date: 2022/03/12 14:59:42 $

.. _issue #274:
    https://github.com/AviSynth/AviSynthPlus/issues/274
