===========
FPS Filters
===========

A set of functions to change the frame rate of a clip:

* `AssumeFPS`_ changes the frame rate without changing the frame count.
* `AssumeScaledFPS`_ is similar to `AssumeFPS`_, but works by directly scaling
  the frame rate numerator and denominator.
* `ChangeFPS`_ changes the frame rate by deleting or duplicating frames.
* `ConvertFPS`_ attempts to convert the frame rate without dropping or inserting
  frames.


AssumeFPS
---------
Changes the frame rate without changing the frame count, causing the video to
play faster or slower. **AssumeFPS** can be called 4 different ways, as
documented below.


.. rubric:: Syntax and Parameters

::

    AssumeFPS (clip1, clip2, bool "sync_audio")
    AssumeFPS (clip, float fps, bool "sync_audio")
    AssumeFPS (clip, int numerator, int denominator, bool "sync_audio")
    AssumeFPS (clip, string preset, bool "sync_audio")


.. describe:: clip, clip1

   Source clip; all color formats supported.

.. describe:: clip2

    If supplied, the new framerate is taken from this clip.

.. describe:: fps

    | If supplied, the new framerate is set to ``fps``.
    | Due to the way frame rates are stored, ``fps`` must be converted to the
      closest ratio of 32-bit signed integers, eg, 33.3333 → 329833/9895
      (33.3332996)

.. describe:: numerator, denominator

    If supplied, the new framerate is set to ``numerator`` divided by
    ``denominator``. If only the ``numerator`` is given, ``denominator`` defaults
    to 1.

.. describe:: preset

    If supplied, the new framerate is taken from a set of standard rates,
    as shown below:

    +---------------------+-----------+-------------+-----------------+
    |   preset            | numerator | denominator | approximate FPS |
    +=====================+===========+=============+=================+
    | "ntsc_film"         | 24000     | 1001        | 23.976          |
    +---------------------+-----------+-------------+-----------------+
    | "ntsc_video"        | 30000     | 1001        | 29.970          |
    +---------------------+-----------+-------------+-----------------+
    | "ntsc_double"       | 60000     | 1001        | 59.940          |
    +---------------------+-----------+-------------+-----------------+
    | "ntsc_quad"         | 120000    | 1001        | 119.880         |
    +---------------------+-----------+-------------+-----------------+
    | "ntsc_round_film"   | 2997      | 125         | 23.976          |
    +---------------------+-----------+-------------+-----------------+
    | "ntsc_round_video"  | 2997      | 100         | 29.97           |
    +---------------------+-----------+-------------+-----------------+
    | "ntsc_round_double" | 2997      | 50          | 59.94           |
    +---------------------+-----------+-------------+-----------------+
    | "ntsc_round_quad"   | 2997      | 25          | 119.88          |
    +---------------------+-----------+-------------+-----------------+
    | "film"              | 24        | 1           | 24.0            |
    +---------------------+-----------+-------------+-----------------+
    | "pal_film"          | 25        | 1           | 25.0            |
    +---------------------+-----------+-------------+-----------------+
    | "pal_video"         | 25        | 1           | 25.0            |
    +---------------------+-----------+-------------+-----------------+
    | "pal_double"        | 50        | 1           | 50.0            |
    +---------------------+-----------+-------------+-----------------+
    | "pal_quad"          | 100       | 1           | 100.0           |
    +---------------------+-----------+-------------+-----------------+

.. describe:: sync_audio

    | If *true*, the audio sample rate is changed by the same amount; the pitch
      of the resulting audio is shifted.
    | If *false*, the audio is unchanged; this means the audio will lose
      synchronization over time.

    Default: false

Examples
^^^^^^^^

PAL +4% conversion::

    AVISource("FILM_clip.avi") # Get 24fps clip
    LanczosResize(768, 576)    # Resize to PAL square-pixel frame size.
    AssumeFPS(25, 1, true)     # Convert frame rate to PAL, also adjust audio.
    SSRC(44100)                # Restore audio sample rate to a standard rate.

* The +4% speed up is conventionally used for displaying 24fps film on PAL
  television. The slight increase in pitch and tempo is readily accepted by
  viewers of PAL material.

Slowing down of video (framerate of original video is 30 frames a second)::

    AviSource("video.avi")
    AssumeFPS(10, true) # Slows the video down to a third of its speed

Speeding up of video (framerate of original video is 30 frames a second):

::

    AviSource("video.avi")
    AssumeFPS(60, true) # Double speed

sync_audio experiments::

    ColorBars() ## 29.97 fps
    Trim(0, 299) ## vdur (video duration) = 10.01 sec
    A = Tone().AudioTrim(0, -9.0) ## adur (audio duration) = 9.00 sec, sample rate=48000
    AudioDub(A) ## vdur=10.01, adur=9.00
    #return AssumeFPS(24, sync_audio=false) ## vdur=12.50, adur=9.00 (audio unchanged)
    #return AssumeFPS(24, sync_audio=true)  ## vdur=12.50, adur=11.238, sample rate=38438
    ## (audio sample rate is changed by the same amount - in this case, by 24.00/29.97)


.. _AssumeScaledFPS:

AssumeScaledFPS
---------------

Scales the frame rate without changing the frame count, causing the video to
play faster or slower.

It is similar to `AssumeFPS`_, but works by directly scaling
:doc:`FrameRateNumerator <../syntax/syntax_clip_properties>` and/or
:doc:`FrameRateDenominator <../syntax/syntax_clip_properties>`.

The ``AssumeScaledFPS`` filter scales the frame rate without changing the
frame count. The numerator is multiplied by the multiplier, the denominator
is multiplied by the divisor, the resulting rational FPS fraction is
normalized, if either the resulting numerator or denominator exceed 31 bits
the result is rounded and scaled. This allows exact rational scaling to be
applied to the FPS property of a clip.

.. rubric:: Syntax and Parameters

::

    AssumeScaledFPS (clip, int "multiplier", int "divisor", bool "sync_audio")

.. describe:: clip

    Source clip; all color formats supported.

.. describe:: multiplier

    clip.FrameRateNumerator is multiplied by multiplier.

.. describe:: divisor

    clip.FrameRateDenominator is multiplied by divisor.

.. describe:: sync_audio

    | If *true*, the audio sample rate is changed by the same amount; the pitch
      of the resulting audio is shifted.
    | If *false*, the audio is unchanged; this means the audio will lose
      synchronization over time.

    Default: false

.. _ChangeFPS:

ChangeFPS
---------
Changes the frame rate by deleting or duplicating frames.

The audio data are not touched by this filter. Audio will remain synchronized,
although the length of the audio data may slightly differ from that of the video
data after the conversion. This is because the output can only contain an
integer number of frames. This effect will be more pronounced for shorter clips.
The difference in length can be ignored. **ChangeFPS** can be called 4 different
ways, as documented below.

.. rubric:: Syntax and Parameters

::

    ChangeFPS (clip1, clip2, bool "linear")
    ChangeFPS (clip, float fps, bool "linear")
    ChangeFPS (clip, int numerator, int denominator, bool "linear")
    ChangeFPS (clip, string preset, bool "linear")

.. describe:: clip, clip1

   Source clip; all color formats supported.

.. describe:: clip2

    If supplied, the new framerate is taken from this clip.

.. describe:: fps

    | If supplied, the new framerate is set to ``fps``.
    | Due to the way frame rates are stored, ``fps`` must be converted to the
      closest ratio of 32-bit signed integers, eg, 33.3333 → 329833/9895
      (33.3332996)

.. describe:: numerator, denominator

    If supplied, the new framerate is set to ``numerator`` divided by
    ``denominator``.  If only the ``numerator`` is given, ``denominator`` defaults
    to 1.

.. describe:: preset

    If supplied, the new framerate is taken from a set of standard rates, as
    listed under `AssumeFPS`_.

.. describe:: linear

    If *true* (the default), this will make AviSynth request frames in a linear
    fashion when skipping frames.

    Default: true

Examples
^^^^^^^^

PAL->NTSC conversion::

    AVISource("PAL_clip.avi")              # Get clip
    Bob(height=480)                        # Separate fields and interpolate them to full height.
    BicubicResize(640,480)                 # Resize to NTSC square-pixel frame size.
    ChangeFPS(60000, 1001)                 # Convert field rate to NTSC, by duplicating fields.
    SeparateFields.SelectEvery(4,0,3)      # Undo Bob, even field first. Use SelectEvery(4,1,2) for odd field first.
    Weave()                                # Finish undoing Bob.

The effect is similar to 3-2 telecine pull down. Regular viewers of PAL
material may notice a motion stutter that viewers of NTSC material readily
ignore as for telecined film.


.. _ConvertFPS:

ConvertFPS
----------

Attempts to convert the frame rate without dropping or inserting frames,
providing a smooth conversion with results similar to those of standalone
converter boxes. The output will have (almost) the same duration as clip, but
the number of frames will change proportional to the ratio of target and source
frame rates.

Frame rate conversion is inherently difficult. This filter implements two common
methods used by commercial Prosumer-level converter systems. The results are
typically quite good. More sophisticated systems employ motion interpolation
algorithms, which are difficult to get right, but, if done right, do yield
superior results.

Footage converted with this filter should not be converted again. Blurriness
builds up quickly in subsequent generations.

The audio data is not touched by this filter. Audio will remain synchronized,
although the length of the audio data may slightly differ from that of the video
data after the conversion. This is because the output can only contain an integer
number of frames. This effect will be more pronounced for shorter clips. The
difference in length should be ignored.

**ConvertFPS** can be called 4 different ways, as documented below. Not all
parameter values are checked for sanity.

.. rubric:: Syntax and Parameters

::

    ConvertFPS (clip1, clip2, int "zone", int "vbi")
    ConvertFPS (clip, float new_rate, int "zone", int "vbi")
    ConvertFPS (clip, int numerator, int denominator, int "zone", int "vbi")
    ConvertFPS (clip, string preset, int "zone", int "vbi")

.. describe:: clip, clip1

   Source clip; all color formats supported.

.. describe:: clip2

    If supplied, the new framerate is taken from this clip.

.. describe:: new_rate

    | Target frame rate. Can be integer or floating point number.
    | In *Blend* mode, ``new_rate`` must be at least 2/3 (66.7%) of the source
      frame rate, or an error will occur. This is to prevent frame skipping. If
      you need to slow down the frame rate more than that, use Switch mode.
    | Due to the way frame rates are stored, ``new_rate`` must be converted to
      the closest ratio of 32-bit signed integers, eg, 33.3333 → 329833/9895
      (33.3332996)

.. describe:: numerator, denominator

    If supplied, the new framerate is set to ``numerator`` divided by
    ``denominator``. The same restrictions apply as for ``new_rate``.  If only
    the ``numerator`` is given, ``denominator`` defaults to 1.

.. describe:: preset

    If supplied, the new framerate is taken from a set of standard rates, as
    listed under `AssumeFPS`_.

.. describe:: zone

    If specified, puts the filter into *Switch* mode. Must be greater or equal
    to zero. If zero, the filter will perform a hard switch, that is, it will
    immediately display the next frame below the switch line. If greater than
    zero, specifies the height (in lines) of the transition zone, where one
    frame is gradually blended into the next. ``zone=80`` yields good results for
    full-size video (480/576 active lines). The transition is done in the same
    way as in :doc:`PeculiarBlend <peculiar>`. ``Zone`` must be less or equal
    than the number of lines of the target frame that correspond to the duration
    of the source frame. This is typically 5/6 or 6/5 of the target frame height,
    that is, a few hundred lines. An error occurs if a larger value is chosen.

.. describe:: vbi

    If specified in *Switch* mode, specifies that the filter should apply a
    timing correction for the vertical blanking interval (VBI). Integer number
    greater than zero, indicating the height of the VBI of the target frames, in
    lines. Typically ``vbi=49`` for PAL and ``vbi=45`` for NTSC, but these
    values are not critical. Ignored in *Blend* mode.

Modes
^^^^^

The filter has two operating modes. If the optional argument zone is not present,
it will blend adjacent video frames, weighted by a blend factor proportional to
the frames' relative timing (Blend mode). If zone is present, it will switch
from one video frame to the next (Switch mode) whenever a new source frame begins
– usually somewhere in the middle of a target frame. Switch mode assumes that
the output will be shown on a TV where each frame is scanned from top to bottom.
The parameter zone specifies the height of the transition region in which the
current frame will be blended into the next.

* **Blend** mode will cause visible, although slight, blurring of motion. This is a
  typical artifact of frame rate conversion and can be seen on commercial video
  tapes and TV programs as well. When working with interlaced video, it is
  important to let the filter operate on individual fields, not on the interlaced
  frames. (see Examples below.)
* **Switch** mode is an attempt to avoid motion blurring, but comes at the expense
  of slight flicker and motion artifacts. Horizontal and vertical pans may show
  a slight wobble. Still frames from this conversion show "broken" or "bent"
  vertical lines in moving scenes. Scene transitions may occur in the middle of
  a frame. Nevertheless, the results do look less blurry than in Blend mode.

Neither mode is perfect. Which one to choose depends on personal preference and
on the footage to be converted. Switch mode is probably only suitable if the
output will be shown on a TV, not on a computer screen.

Examples
^^^^^^^^

NTSC->PAL conversion::

    AviSource("NTSC_clip.avi")             # Get clip
    Bob(height=576)                        # Separate fields and interpolate them to full height.
    BicubicResize(768,576)                 # Resize to PAL square-pixel frame size.
    ConvertFPS(50)                         # Convert field rate to PAL, using Blend mode.
    SeparateFields.SelectEvery(4,0,3)      # Undo Bob, even field first. Use SelectEvery(4,1,2) for odd field first.
    Weave()                                # Finish undoing Bob.

This example will also work with frame-based NTSC material, even with telecined
film (movies).

For film material, however, you will get better results by using an inverse-telecine
filter and speeding up the frame rate from 23.976 to 25fps.

Changelog
---------

+-----------------+------------------------------------------------------------+
| Version         | Changes                                                    |
+=================+============================================================+
| AviSynth+ r2632 | ConvertFPS: added 10-32bits support.                       |
+-----------------+------------------------------------------------------------+
| AviSynth 2.5.7  || AssumeFPS, ChangeFPS, ConvertFPS: added preset option.    |
|                 || ConvertFPS: YV12 and RGB support and fixed blending ratio.|
|                 || AssumeFPS, ChangeFPS, ConvertFPS: changed framerate       |
|                 |  behaviour, see below.                                     |
|                 |                                                            |
|                 || From v2.57, the behaviour with respect to the framerateis |
|                 |  a bit changed. The main issue is that users are allowed to|
|                 |  specify the framerate as float, but the NTSC (FILM and    |
|                 |  Video) and PAL standards require ratios as framerate.     |
|                 |  Besides this AviSynth exports the framerate as a ratio, so|
|                 |  when specifying a float, it will be converted to a ratio. |
|                 |  The ratios of the standards are given by 24000/1001 for   |
|                 |  23.976 (FILM) and 30000/1001 for 29.97(Video). When       |
|                 |  specifying these floats, they are exported by AviSynth as |
|                 |  ratios, but not as the standard ratios. One of the reasons|
|                 |  for this is, that those floats are approximations         |
|                 |  (remember that 24000/1001 = 23.9760239760...), so how     |
|                 |  should AviSynth know how to choose the correct ratio? In  |
|                 |  order to overcome this issue, the user can use            |
|                 |  AssumeFPS(24000,1001) or simply AssumeFPS("ntsc_film").   |
|                 |                                                            |
|                 || Another problem is that in v2.56 and older, the converted |
|                 |  floats were exported with 64 bit precision, resulting in  |
|                 |  very large numerators and denominators, making some       |
|                 |  players crash. To overcome this, a smart float-ratio is   |
|                 |  added internally, and the framerates are approximated     |
|                 |  accurately by ratios of small numbers. For example,       |
|                 |  AssumeFPS(23.976) is converted to AssumeFPS(2997, 125) as |
|                 |  can be checked with :doc:`Info <info>`.                   |
+-----------------+------------------------------------------------------------+
| AviSynth 2.5.6  || ChangeFPS and ConvertFPS: added clip2 option.             |
|                 || Added AssumeScaledFPS filter.                             |
+-----------------+------------------------------------------------------------+
| AviSynth 2.5.5  | AssumeFPS: added clip2 option.                             |
+-----------------+------------------------------------------------------------+
| AviSynth 2.5.4  | ChangeFPS: added linear=true/false option.                 |
+-----------------+------------------------------------------------------------+
| AviSynth 2.0.5  | ChangeFPS: Up to v2.05, the video gets truncated or filled |
|                 | up to preserve playback speed and play time (the number of |
|                 | frames was not changed). In later versions, the behaviour  |
|                 | has been changed and the number of frames is increased or  |
|                 | decreased as in ConvertFPS.                                |
+-----------------+------------------------------------------------------------+

$Date: 2022/07/30 14:45:27 $
