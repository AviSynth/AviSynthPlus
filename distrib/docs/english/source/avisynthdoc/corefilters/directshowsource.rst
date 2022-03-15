================
DirectShowSource
================

**DirectShowSource** reads media files using Microsoft `DirectShow`_, the same
multimedia playback system that WMP (*Windows Media Player*) uses. It can read
most formats that WMP can, including MP4, MP3, most MOV (QuickTime) files, as
well as AVI files that AVISource doesn't support (like DV type 1, or files using
DirectShow-only codecs). There is also support for `GraphEdit`_ (grf) files.

There are some caveats:

* Some decoders (notably MS MPEG-4) will produce upside-down video. You'll have
  to use :doc:`FlipVertical <flip>`.
* DirectShow video decoders are not required to support frame-accurate seeking.
  In most cases seeking will work, but on some it might not.
* DirectShow video decoders are not even required to tell you the frame rate of
  the incoming video. Most do, but the ASF decoder doesn't. You have to specify
  the frame rate using the ``fps`` parameter, like this:
  ``DirectShowSource("video.asf", fps=15)``.
* This version automatically detects the Microsoft DV codec and sets it to
  decode at full (instead of half) resolution. I guess this isn't a caveat. :-)
* Also this version attempts to disable any decoder based deinterlacing.

Try reading AVI files with :doc:`AviSource <avisource>` first. For non-AVI files,
try `FFmpegSource`_ or `LSMASHSource`_. If that doesn't work then try this filter
instead.


Syntax and Parameters
----------------------

::

    DirectShowSource (string filename, float "fps", bool "seek", bool "audio", bool "video",
                      bool "convertfps", bool "seekzero", int "timeout", string "pixel_type",
                      int "framecount", string "logfile", int "logmask")


.. describe:: filename

    The path of the source file; path can be omitted if the source file is in
    the same directory as the AviSynth script (\*.avs).

.. describe:: fps

    Frames Per Second of the resulting clip. This is sometimes needed to specify
    the framerate. If the framerate or the number of frames is incorrect (this
    can happen with ASF or MOV clips for example), use this option to force the
    correct framerate. For live sources, this is like "max fps" that will be
    displayed.

    Default: auto

.. describe:: seek

    There is full seeking support available on most file formats. If problems
    occur, try setting ``seekzero=true`` first. If seeking still causes problems,
    disable seeking completely with ``seek=false``. With seeking disabled and
    trying to seek backwards, the audio stream returns silence, and the video
    stream returns the most recently rendered frame. **Note** the AviSynth cache
    *may* provide limited access to the previous few frames, but beyond that the
    most recently frame rendered will be returned.

    Default: true

.. describe:: audio

    Enable audio on the resulting clip. The channel ordering is the same as in
    the `wave-format-extensible format`_, because the input is always decompressed
    to WAV. For more information, see also :doc:`GetChannel <getchannel>`.
    AviSynth loads 8, 16, 24 and 32 bit int PCM samples, and float PCM format,
    and any number of channels.

    Default: true

.. describe:: video

    Enable video on the resulting clip.

    Default: true

.. describe:: convertfps

    If true, it turns `VFR`_ (variable framerate) video into CFR (constant framerate)
    video by adding frames. This allows you to open VFR video in AviSynth. It is
    most useful when fps is set to the least common multiple of the component
    frame rates, e.g. 120 or 119.880.

    Default: false

.. describe:: seekzero

    If true, restrict backwards seeking only to the beginning, and seeking
    forwards is done the hard way (by reading all samples). Limited backwards
    seeking is allowed with non-indexed `ASF`_.

    Default: false

.. describe:: timeout

    For positive values DirectShowSource waits for up to ``timeout`` milliseconds
    for the DirectShow graph to start. ``timeout`` is clamped between [5000,300000]
    milliseconds. If the graph fails to start a compile time exception is thrown.
    Once the graph starts, each GetFrame/GetAudio call will wait for up to the
    timeout value and then return a grey frame or silence for the audio. No
    runtime exceptions are ever thrown because of time-outs.

    For negative values DirectShowSource waits for up to 2000 milliseconds for
    the DirectShow graph to start. If the graph fails to start it is ignored at
    that point and the initial graph start wait is deferred until the first
    GetFrame/GetAudio call. If any GetFrame/GetAudio call experiences a timeout
    a runtime exception is then thrown.

.. describe:: pixel_type

    Request a color format from the decompressor. Valid values are:

    YV24, YV16, YV12, I420, NV12, YUY2, AYUV, Y41P, Y411, ARGB, RGB64, RGB48,
    RGB32, RGB24,  YUV, YUVex, RGB, AUTO, FULL

        By default, upstream DirectShow filters are free to bid all of their
        supported media types in the order of their choice. A few DirectShow filters
        get this wrong. The ``pixel_type`` argument limits the acceptable video
        stream subformats for the `IPin negotiation`_. Note the graph builder may
        add a format converter to satisfy your request, so make sure the codec in
        use can actually decode to your chosen format. The MS format converter is
        just adequate. The "YUV" and "RGB" pseudo-types restrict the negotiation to
        all official supported YUV or RGB formats respectively. The "YUVex" also
        includes YV24, YV16, I420 and NV12 non-standard pixel types. The "AUTO"
        pseudo-type permits the negotiation to use all relevant official formats,
        YUV plus RGB. The "FULL" pseudo-type includes the non-standard pixel types
        in addition to those supported by "AUTO". The full order of preference is
        YV24, YV16, YV12, I420, NV12, YUY2, AYUV, Y41P, Y411, ARGB, RGB32, RGB24,
        RGB64, RGB48. Many DirectShow filters get this wrong, which is why it is
        not enabled by default. The option exists so you have enough control to
        encourage the  maximum range of filters to serve your media.
        (See `discussion`_.)

        The non-standard pixel types use the following GUID's respectively :- ::

            MEDIASUBTYPE_I420 = {'024I', 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71};
            MEDIASUBTYPE_YV24 = {'42VY', 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71};
            MEDIASUBTYPE_YV16 = {'61VY', 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71};
            MEDIASUBTYPE_NV12 = {'21VN', 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71};

    In other words, if ``pixel_type="AUTO"``, it will try to output YV24; if
    that isn't possible it tries YV16, and if that isn't possible it tries
    YV12, etc...

    For planar color formats, adding a '+' prefix, e.g.
    ``DirectShowSource(..., pixel_type="+YV12")``, tells AviSynth the video rows
    are DWORD aligned in memory instead of packed. This can fix skew or tearing
    of the decoded video when the width of the picture is not divisible by 4.

.. describe:: framecount

    Sometimes needed to specify the :doc:`frame count <../syntax/syntax_clip_properties>`
    of the video. If the framerate or the number of frames is incorrect (this
    can happen with ASF or MOV clips for example), use this option to force the
    correct number of frames. If ``fps`` is also specified, the length of the
    audio stream is adjusted. For live sources, specify a very large number.

    Default: auto

.. describe:: logfile

    Use this option to specify the name of a log file for debugging.

.. describe:: logmask

    When a ``logfile`` is specified, use this option to select which information
    is logged.

    +-------+-------------------------+
    | Value | Data                    |
    +=======+=========================+
    | 1     | Format Negotiation      |
    +-------+-------------------------+
    | 2     | Receive samples         |
    +-------+-------------------------+
    | 4     | GetFrame/GetAudio calls |
    +-------+-------------------------+
    | 8     | Directshow callbacks    |
    +-------+-------------------------+
    | 16    | Requests to Directshow  |
    +-------+-------------------------+
    | 32    | Errors                  |
    +-------+-------------------------+
    | 64    | COM object use count    |
    +-------+-------------------------+
    | 128   | New objects             |
    +-------+-------------------------+
    | 256   | Extra info              |
    +-------+-------------------------+
    | 512   | Wait events             |
    +-------+-------------------------+

    Add the values together of the data you need logged. Specify -1 to log
    everything. The default, 35, logs 1+2+32, or Format Negotiation, Received
    samples and Errors.

    Default: 35


Examples
--------

Opens an avi with the first available RGB format (without audio)::

    DirectShowSource("F:\xvid.avi", fps=25, audio=false, pixel_type="RGB")

Opens a DV clip with the MS DV decoder::

    DirectShowSource("F:\DVCodecs\Ced_dv.avi") # MS-DV

Opens a variable framerate mkv as 119.88 by adding frames (ensuring sync)::

    DirectShowSource("F:\vfr_startrek.mkv", fps=119.88, convertfps=true)

Opens a realmedia (\*.rmvb) clip::

    DirectShowSource("F:\test.rmvb", fps=24, convertfps=true)

Opens a `GraphEdit`_ file::

    V=DirectShowSource("F:\vid_graph.grf", audio=False) # video only (audio renderer removed)
    A=DirectShowSource("F:\aud_graph.grf", video=False) # audio only (video renderer removed)
    AudioDub(V, A)

See :ref:`below <dss-downmixingac3>` for some audio examples.


Troubleshooting video and audio problems
----------------------------------------

AviSynth will by default try to open only the media it can open without any
problems. If one component cannot be opened it will simply not be added to the
output. This will also mean that if there is a problem, you will not see the
error. To get the error message to the missing component, use ``audio=false`` or
``video=false`` and disable the component that is actually working. This way
AviSynth will print out the error message of the component that doesn't work.


RenderFile, the filter graph manager won't talk to me
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is a common error that occurs when DirectShow isn't able to deliver any
format that is readable to AviSynth. Try creating a filter graph manually and
see if you are able to construct a filter graph that delivers any output
AviSynth can open. If not, you might need to download additional DirectShow
filters that can deliver correct material.

The picture is skewed or torn and the colors are wrong
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some DirectShow components incorrectly pad the lines of planar data to be DWORD
aligned as is done for RGB24 DIB format. This is incorrect, but it is a fairly
common mistake. By adding a '+' to the start of the pixel_type string you can
inform DirectShowSource to treat planar data formats as padded DWORD aligned.
This problem shows up when the width of the picture is not divisible by 4.

::

    DirectShowSource("NonMod4Video.mp4", pixel_type="+YV12") # Bad DWORD aligned planar

The samplerate is wrong
~~~~~~~~~~~~~~~~~~~~~~~

Some filters might have problems reporting the right samplerate, and then
correct this when the file is actually playing. Unfortunately there is no way
for AviSynth to correct this once the file has been opened. Use
:doc:`AssumeSampleRate <assumerate>` and set the correct samplerate to fix this
problem.

My sound is choppy
~~~~~~~~~~~~~~~~~~

Unfortunately Directshow is not required to support sample exact seeking.
Open the sound another way, or demux your video file and serve it to AviSynth
another way. Otherwise you can specify ``seekzero=true`` or ``seek=false`` as
parameters or use the :doc:`EnsureVBRMP3Sync <ensuresync>` filter to enforce
linear access to the Directshow audio stream.


My sound is out of sync
~~~~~~~~~~~~~~~~~~~~~~~

This can happen especially with WMV, apparently due to variable frame rate
video being returned. Determine what the fps should be and set it explicitly,
and also :doc:`ConvertFPS <fps>` to force it to remain constant. And
:doc:`EnsureVBRMP3Sync <ensuresync>` reduces problems with variable rate audio.

::

    DirectShowSource("video.wmv", fps=25, ConvertFPS=True)
    EnsureVBRMP3Sync()

My ASF renders start fast and finish slow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Microsoft in their infinite wisdom chose to implement ASF stream timing in
the ASF demuxer. As a result it is not possible to strip ASF format files any
faster than realtime. This is most apparent when you first start to process
the streams, usually after opening the Avisynth script it takes you a while
to configure your video editor, all this time the muxer is accumulating
*credit* time. When you then start to process your stream it races away at
maximum speed until you catch up to realtime at which point it slows down to
the realtime rate of the source material. This feature makes it impossible to
use Avisynth to reclock 24fps ASF material up to 25fps for direct PAL
playback.

Windows7 users
~~~~~~~~~~~~~~

Windows 7 forces its own DirectShow filters for decoding several audio and video
formats. Changing their merits or physically removing those filters doesn't help.
clsid made the tool `"Win7DSFilterTweaker"`_ to change the preferred filters.
However new decoders need to be added each time so it's not the perfect solution.


Common tasks
------------

This section will describe various tasks that might not be 100% obvious. :)


Opening GRF files
~~~~~~~~~~~~~~~~~

`GraphEdit`_ GRF-files are automatically detected by a .grf filename extension
and directly loaded by DirectShowSource. For AviSynth to be able to connect
to it, you must leave a pin open in GraphEdit of a media types that AviSynth
is able to connect to. AviSynth will not attempt to disconnect any filters,
so it is important that the output type is correct. DirectShowSource only
accepts YV24, YV16, YV12, YUY2, AYUV, Y41P, Y411, ARGB, RGB32 and RGB24 video
formats and 32, 24, 16 and 8 bit PCM and IEEE FLOAT audio formats.

A given GRF-file should only target one of an audio or video stream to avoid
confusion when directshowsource attempts the connection to your open pin(s).
This single stream restriction is enforced.

.. _dss-downmixingac3:

Downmixing AC3 to stereo
~~~~~~~~~~~~~~~~~~~~~~~~

The following section covers how to downmix a 5.1 AC3 file to stereo using
`AC3Filter`_ and then load the result with **DirectShowSource**.

#. Install AC3Filter (see warning below).

    a.) Open **AC3Filter Config**. In the "Main" tab on the "Output format"
    sectionand select "Stereo" and set the format to "PCM Float". [Nothing else
    is needed.]

    **-OR-**

    b.) Open the AC3 file in a media player. For example, in `MPC-HC`_ (v1.9.19)
    go to the "Play" tab and then scroll down to "Filters" and select AC3Filter.
    The **AC3Filter Config** window will open, from there set the output format
    to **Stereo** and for maximum quality set the format to **PCM Float**.

    .. list-table::

        * - .. figure:: pictures/ac3filter-downmix.png

               **AC3Filter Config**

    Any changes made within AC3Filter will also be applied to the AC3 file when
    decoded via **DirectShowSource**. For example, if Gain and/or DRC are used
    or any other settings in the Mixer, Gains and Equalizer tabs.

    .. warning::

        The `lastest AC3Filter`_ version is 2.6.0b and x64 is only available in
        the "full" version. Note that the full version comes bundled with the
        (now-defunct) OpenCandy adware module that included unwanted third-party
        offers during the setup process. More information in the archived
        `AC3Filter wiki`_ and the `"Is AC3Filter Safe?"`_ VideoHelp thread. The
        "lite" version does not contain OpenCandy but does not include a 64-bit
        AC3Filter.

#. Write the following script::

    v = Mpeg2Source("e:\movie.d2v")
    a = DirectShowSource("e:\test.ac3")
    AudioDub(v,a)

#. Finally, load the script in VDub or FFmpeg and save the audio stream to the
   desired format.

Note that this method is not only limited to AC3 files but since AC3Filter is no
longer developed, some *modern* formats may not be compatible. However, there are
other alternatives. For example, ``LWLibavAudioSource("test.ac3", layout="DL+DR")``
from `LSMASHSource`_ will also downmix to stereo. And of course, for more control
there are a handful of AviSynth scripts that use the core filters for downmixing.
See the examples section in the :doc:`GetChannels <getchannel>` filter page.

See also
--------

* Haali media splitter also comes with an (unrelated) DirectShow input plugin
  `DirectShowSource2`_, aka DSS2.

* Another (unrelated) alternative is `DSS2mod`_.


Changelog
---------

+-----------------+------------------------------------------------------------+
| Version         | Changes                                                    |
+=================+============================================================+
| AviSynth 2.6.0  | Added pixel_types "YV24", "YV16", "AYUV", "Y41P", "Y411".  |
+-----------------+------------------------------------------------------------+
| AviSynth 2.5.7  || framecount overrides the length of the streams.           |
|                 || logfile and logmask specify debug logging.                |
+-----------------+------------------------------------------------------------+
| AviSynth 2.5.6  || convertfps turns vfr into constant cfr by adding frames.  |
|                 || seekzero restricts seeking to begining only.              |
|                 || timeout controls response to recalcitrant graphs.         |
|                 || pixel_type specifies/restricts output video pixel format. |
+-----------------+------------------------------------------------------------+

$Date: 2022/03/14 02:45:53 $

.. _DirectShow:
    https://en.wikipedia.org/wiki/DirectShow
.. _GraphEdit:
    http://avisynth.nl/index.php/GraphEdit
.. _FFmpegSource:
    http://avisynth.nl/index.php/FFmpegSource
.. _LSMASHSource:
    http://avisynth.nl/index.php/LSMASHSource
.. _wave-format-extensible format:
    https://web.archive.org/web/20190905063051/http://www.cs.bath.ac.uk/~jpff/NOS-DREAM/researchdev/wave-ex/wave_ex.html
.. _VFR:
    http://avisynth.nl/index.php/VFR
.. _ASF:
    https://en.wikipedia.org/wiki/Advanced_Systems_Format
.. _IPin negotiation:
    https://en.wikipedia.org/wiki/DirectShow#Architecture
.. _discussion:
    https://forum.doom9.org/showthread.php?t=143321
.. _planar:
    http://avisynth.nl/index.php/Planar
.. _"Win7DSFilterTweaker":
    https://forum.doom9.org/showthread.php?t=146910
.. _AC3Filter:
    https://web.archive.org/web/20191212120549/http://www.ac3filter.net/wiki/AC3Filter
.. _MPC-HC:
    https://github.com/clsid2/mpc-hc
.. _lastest AC3Filter:
    https://code.google.com/archive/p/ac3filter/downloads
.. _AC3Filter wiki:
    https://web.archive.org/web/20200428203226/http://ac3filter.net/wiki/OpenCandy
.. _"Is AC3Filter Safe?":
    https://forum.videohelp.com/threads/379482-Is-AC3Filter-Safe
.. _Channel Downmixer by Trombettworks:
    https://web.archive.org/web/20190907051617/http://www.trombettworks.com/directshow.php
.. _DirectShowSource2:
    http://avisynth.nl/index.php/DSS2#Source_Filters
.. _DSS2mod:
    http://avisynth.nl/index.php/DSS2mod
