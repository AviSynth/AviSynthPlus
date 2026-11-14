==========
AudioStats
==========

Reads the whole audio track once and returns its levels: peak, RMS, mean, the number of clipped
samples, a "loose" peak which ignores rare spikes, for all channels together and optionally per
channel. The result is an array of named values (a dictionary-style array, see
:doc:`Arrays <../script_ref/script_ref_arrays>`), available while the script is loaded, so it can be
used directly as a parameter, e.g. of :doc:`Amplify <amplify>`.

AudioStats is a function, not a filter: it returns values, not a clip.


Syntax and Parameters
----------------------

::

    AudioStats (clip, float "threshold", bool "per_channel")

.. describe:: clip

    | Source clip with audio. All sample types are supported. Values are normalized: full scale is 1.0
      for every sample type, as after :doc:`ConvertAudioToFloat <convertaudio>`
      (an integer sample is divided by 2\ :sup:`bits-1`).
    | To measure a part of the track only, use :doc:`AudioTrim <trim>` before.

.. describe:: threshold

    Percentage of the samples ignored at the top when calculating ``loose_peak``, between 0 and 100.
    With 0, ``loose_peak`` is the same as ``peak``. E.g. 0.01: one sample of 10000 may be above
    ``loose_peak``; a few clicks or crackles then do not set the level.

    Default: 0

.. describe:: per_channel

    If *true*, the result gets one more element, ``channels``: the same values for each channel
    separately (see `Per-channel values`_). All channels are measured in the same single pass, there is
    no need to read the track once per channel.

    Default: false


Returned values
---------------

An array of ``[name, value]`` pairs. The values are reached by name, ``s["peak"]``, or by position:
``s[2]`` is the pair ``["peak", value]``, ``s[2][1]`` is the value. Names are case-insensitive.

+---+-------------------+--------+-------------------------------------------------------------------+
| # | Name              | Type   | Meaning                                                           |
+===+===================+========+===================================================================+
| 0 | ``min``           | float  | the lowest sample value (signed)                                  |
+---+-------------------+--------+-------------------------------------------------------------------+
| 1 | ``max``           | float  | the highest sample value (signed)                                 |
+---+-------------------+--------+-------------------------------------------------------------------+
| 2 | ``peak``          | float  | the highest absolute sample value: max(\|min\|, \|max\|)          |
+---+-------------------+--------+-------------------------------------------------------------------+
| 3 | ``peak_db``       | float  | ``peak`` in dBFS: 20 * log10(``peak``)                            |
+---+-------------------+--------+-------------------------------------------------------------------+
| 4 | ``peak_sample``   | int    | position of the first sample with the ``peak`` value, counted in  |
|   |                   | (64    | samples per channel from 0, as ``GetAudio`` positions. Seconds:   |
|   |                   | bit)   | ``peak_sample / Float(AudioRate)``                                |
+---+-------------------+--------+-------------------------------------------------------------------+
| 5 | ``rms``           | float  | root mean square of the samples                                   |
+---+-------------------+--------+-------------------------------------------------------------------+
| 6 | ``rms_db``        | float  | ``rms`` in dBFS                                                   |
+---+-------------------+--------+-------------------------------------------------------------------+
| 7 | ``mean``          | float  | average of the samples: the DC offset, near 0 for normal audio    |
+---+-------------------+--------+-------------------------------------------------------------------+
| 8 | ``clipped``       | int    | number of samples at or beyond full scale, see `Clipped samples`_ |
|   |                   | (64    |                                                                   |
|   |                   | bit)   |                                                                   |
+---+-------------------+--------+-------------------------------------------------------------------+
| 9 | ``loose_peak``    | float  | the level which at most ``threshold`` percent of the samples      |
|   |                   |        | exceed                                                            |
+---+-------------------+--------+-------------------------------------------------------------------+
| 10| ``loose_peak_db`` | float  | ``loose_peak`` in dBFS                                            |
+---+-------------------+--------+-------------------------------------------------------------------+
| 11| ``channels``      | array  | only with ``per_channel=true``, see `Per-channel values`_         |
+---+-------------------+--------+-------------------------------------------------------------------+

The float values are 64 bit (double). dB values are not rounded; silence (0) gives ``-inf``.
Infinite float samples count in ``min``, ``max``, ``peak`` and ``clipped`` (``rms`` and ``mean`` become
infinite); NaN samples are skipped by ``min``, ``max`` and ``peak``, but make ``rms`` and ``mean`` NaN.
dB values use the same formula as :doc:`AmplifydB <amplify>`, so ``AmplifydB(-s["peak_db"])`` gives
the same factor as ``Amplify(1.0 / s["peak"])``.

Without ``per_channel``, all values are calculated over all channels together: ``peak`` is the
highest of the channel peaks, ``rms`` and ``mean`` are taken over all samples of all channels.


Per-channel values
------------------

With ``per_channel=true``, ``s["channels"]`` is an array of ``[name, values]`` pairs, one for each
channel in the order of the audio stream. ``values`` has the same names as the overall result
(``min`` ... ``loose_peak_db``).

The name of a channel comes from the channel mask (see :doc:`SetChannelMask <setchannelmask>`):
``FL``, ``FR``, ``FC``, ``LFE``, ``BL``, ``BR``, ``FLC``, ``FRC``, ``BC``, ``SL``, ``SR``, ``TC``,
``TFL``, ``TFC``, ``TFR``, ``TBL``, ``TBC``, ``TBR``. A 5.1 stream is usually ``FL FR FC LFE BL BR``
(or ``SL SR`` instead of ``BL BR``). If the mask is not known, or it does not match the channel count,
the names are the channel numbers as strings: ``"0"``, ``"1"``, ...

A channel can be reached in two ways:

* by name, when the layout is known: ``s["channels"]["FC"]["peak"]``
* by index, e.g. in a loop of a general script: ``s["channels"][2]`` is the pair ``["FC", values]``,
  ``s["channels"][2][0]`` the name, ``s["channels"][2][1]["peak"]`` the value.
  ``ArrayIndexOf(s["channels"], "FC")`` gives the index of a name, -1 if there is no such channel.

The index here starts at 0, while :doc:`GetChannel <getchannel>` counts the channels from 1:
``s["channels"][1]`` belongs to ``GetChannel(clip, 2)``.


Clipped samples
---------------

``clipped`` counts the samples which reached full scale of the original sample type: an integer
sample at its lowest or highest value (e.g. -32768 or 32767 at 16 bit), a float sample at or beyond
-1.0 or 1.0. Hitting full scale on a few samples can be a properly normalized track; many of them
in a row usually mean the source was clipped. Float audio can be above 1.0 without loss, it is only
clipped when it is converted to an integer type.

At 32 bit integer the highest value, 2147483647, and the values just below it are the same as 1.0 in
float precision, they are counted as clipped as well.


How the loose peak works
------------------------

The absolute sample values are counted in a histogram with bins of about 0.03 dB (the float value
cut to 8 mantissa bits). From the top, the bins are added until the count passes ``threshold``
percent of all samples; ``loose_peak`` is the upper edge of that bin, but not above ``peak``. So at
most ``threshold`` percent of the samples are above ``loose_peak``, and the value can be up to one
bin higher than the exact limit. If that bin is the one of zero (e.g. silence with a few clicks, all
of them within ``threshold``), ``loose_peak`` is 0.

With ``per_channel=true``, each channel has its own histogram; the overall ``loose_peak`` is
calculated from all of them together, not from the channel results.


Speed
-----

AudioStats reads the whole audio track when the script is loaded, so the source filter decodes all
of it. This takes the most time: the level calculation itself is a small part (about 0.05 s for 2
minutes of 32 kHz stereo audio). :doc:`Normalize <normalize>` reads the track the same way, at its
first audio request. The values of a file do not change: they can be calculated once and written into
the script as numbers.


Examples
--------

Peak normalization to 0.98, the same as ``Normalize(0.98)``. A silent track has peak 0: it is left as
it is, as Normalize does::

    a = WavSource("audio.wav")
    s = AudioStats(a)
    a.Amplify(s["peak"] > 0 ? 0.98 / s["peak"] : 1.0)

Peak at -1 dBFS (silence: ``peak_db`` is ``-inf``, 0 dB is used)::

    a = WavSource("audio.wav")
    s = AudioStats(a)
    a.AmplifydB(s["peak"] > 0 ? -1.0 - s["peak_db"] : 0.0)

Show the values (a clip without video gets a blank one)::

    a = WavSource("audio.wav")
    s = AudioStats(a, threshold=0.01)
    BlankClip(a, width=640, height=200, pixel_type="YV12")
    Subtitle("peak " + String(s["peak_db"], "%.2f") + " dB at " + \
      String(s["peak_sample"] / Float(AudioRate(a)), "%.2f") + " s" + \
      "\nloose peak (0.01%) " + String(s["loose_peak_db"], "%.2f") + " dB" + \
      "\nrms " + String(s["rms_db"], "%.2f") + " dB, clipped samples: " + String(s["clipped"]), lsp=0)

The frame of the peak, for a clip with video::

    s = AudioStats(last)
    peak_frame = Int(s["peak_sample"] * FrameRate(last) / AudioRate(last))

A single channel. These give the same peak for the second channel (front right of a 5.1 stream)::

    a = WavSource("audio_5.1.wav")
    p1 = AudioStats(GetChannel(a, 2))["peak"]        # GetChannel counts from 1; reads the track once more
    s = AudioStats(a, per_channel=true)              # all channels in one pass
    p2 = s["channels"]["FR"]["peak"]                 # by name
    p3 = s["channels"][1][1]["peak"]                 # by index, from 0

All channels by index, e.g. in a function which works with any layout. This one lists the channel
peaks, and amplifies each channel to 0.98 on its own (only for a faulty source, e.g. one channel
recorded too low: normal multichannel audio needs the same gain on all channels, or the mix changes).
Silent channels are left as they are::

    a = WavSource("audio.wav")
    ch = AudioStats(a, per_channel=true)["channels"]
    factors = []
    info = ""
    for (i = 0, ArraySize(ch) - 1) {
      peak = ch[i][1]["peak"]
      factors = ArrayAdd(factors, peak > 0 ? 0.98 / peak : 1.0)
      info = info + ch[i][0] + ": " + String(ch[i][1]["peak_db"], "%.1f") + " dB  "
    }
    a.Amplify(factors) # Amplify takes one factor per channel

Quiet dialogue in a 5.1 track: how much the centre channel can be raised before it reaches -1 dBFS::

    a = LWLibavAudioSource("movie.mkv")
    fc = AudioStats(a, per_channel=true)["channels"]["FC"]
    boost = Min(6.0, -1.0 - fc["peak_db"])
    a.AmplifydB(0, 0, boost, 0, 0, 0)                # FL FR FC LFE BL BR

Check for a silent channel (e.g. an empty LFE or fake surround)::

    s = AudioStats(a, per_channel=true)
    i = ArrayIndexOf(s["channels"], "LFE")
    lfe_silent = i >= 0 ? s["channels"][i][1]["peak"] == 0 : false

Ignore a few clicks: the loose peak sets the gain. The clicks go above full scale; they are clipped
when the audio is converted to an integer type (float audio keeps them)::

    s = AudioStats(a, threshold=0.001)
    a.Amplify(s["loose_peak"] > 0 ? 0.98 / s["loose_peak"] : 1.0).ConvertAudioTo16bit()


Changelog
----------

+------------------+------------------------+
| Version          | Changes                |
+==================+========================+
| AviSynth+ 3.7.7  | Added AudioStats.      |
+------------------+------------------------+
