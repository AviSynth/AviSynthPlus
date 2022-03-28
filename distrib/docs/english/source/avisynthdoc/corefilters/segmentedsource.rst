==============================================
SegmentedAviSource / SegmentedDirectShowSource
==============================================

These filters automatically load and join up to 100 clips each:

* `SegmentedAviSource`_ loads AVI files using AviSource and joins them together
  using :doc:`UnalignedSplice <splice>`.

* `SegmentedDirectShowSource`_ works the same way, but calling DirectShowSource
  in place of AviSource.

Parameters are the same as :doc:`AviSource <avisource>` and
:doc:`DirectShowSource <directshowsource>`. See respective pages for full
documentation.


SegmentedAVISource
------------------

**SegmentedAviSource** loads up to 100 AVI files per *base_filename* using
:doc:`AviSource <avisource>` and joins them using :doc:`UnalignedSplice <splice>`.
As with AviSource there is built-in `Audio Compression Manager`_ support
for decoding compressed audio tracks (MP3, AAC, AC3, etc).

.. rubric:: Syntax and Parameters

::

    SegmentedAVISource (string base_filename [, ...], bool "audio", string "pixel_type"
                        string "fourCC", int "vtrack", int "atrack", bool "utf8")

.. describe:: base_filename

    Full or relative path to one or more filenames. Each given filename will
    serve as the "base" filename for loading clips. For example,
    ``SegmentedDirectShowSource("d:\filename.ext")`` will load the files
    d:\filename.\ **00**\ .ext, d:\filename.\ **01**\ .ext and so on, through
    d:\filename.\ **99**\ .ext. Any files in this sequence that don't exist or
    do not have the same media properties as the first clip will be skipped.
    See `Notes`_ below for information on loading clips in multiple directories.
    `UTF-8`_ filenames are supported when ``utf8=true``.

.. describe:: audio

    If true, load the first audio stream, or the stream specified by ``atrack``
    if present. If false, audio is disabled.

    Default: true

.. describe:: pixel_type

    Chooses the output color format of the decompressor. Valid values are listed
    in the table below. This argument has no effect if the video is uncompressed,
    as no decompressor will be used in that case. See the
    :ref:`10+ bit inputs <AVISource-hbd>` section for a full list of pixel types.

    * If omitted or "FULL", AviSynth will use the first format supported by the
      decompressor, in the order shown in the table below.
    * If "AUTO", AviSynth will use the alternate (older) order as shown.

      .. table::
          :widths: auto

          +------------+-------+-------+-------+--------+-------+--------+--------+
          | pixel_type | Color formats, listed by decoding priority (high to low) |
          +============+=======+=======+=======+========+=======+========+========+
          | FULL       | YV24  | YV16  | YV12  | YV411  | YUY2  | RGB32  | RGB24  |
          +------------+-------+-------+-------+--------+-------+--------+--------+
          | AUTO       |       |       | YV12  |        | YUY2  | RGB32  | RGB24  |
          +------------+-------+-------+-------+--------+-------+--------+--------+

    In other words, if you don't specify anything, it will try to output YV24;
    if that isn't possible it tries YV16, and if that isn't possible it tries
    YV12, etc ...

    For `planar`_ color formats, adding a '+' prefix, e.g.
    ``SegmentedAVISource(..., pixel_type="+YV12")``, tells AviSynth the video
    rows are DWORD aligned in memory instead of packed. **This can fix skew or
    tearing of the decoded video** with bad codecs when the width of the picture
    is not divisible by 4.

    Default: "FULL"

.. describe:: fourCC

    Forces AviSynth to use a specific decoder instead of the one specified in
    the source file. See `FourCC`_ for more information.

    Default: auto from source

.. describe:: vtrack

    Specifies a numbered video track. Track numbers start from zero, and are
    guaranteed to be continuous (i.e. there must be a track 1 if there is a
    track 0 and a track 2). If no video stream numbered ``vtrack`` exists, an
    error will be raised.

    Default: 0

.. describe:: atrack

    Specifies a numbered audio track. Track numbers start from zero, and are
    guaranteed to be continuous (i.e. there must be a track 1 if there is a
    track 0 and a track 2). If no audio stream numbered ``atrack`` exists, no
    error will be raised, and no audio will be returned.

    Default: 0

.. describe:: utf8

    If true, file name is treated as `UTF-8`_.

    Default: false

SegmentedDirectShowSource
-------------------------

**SegmentedDirectShowSource** loads up to 100 files per *base_filename* using
:doc:`DirectShowSource <directshowsource>` and joins them using :doc:`UnalignedSplice <splice>`.

.. rubric:: Syntax and Parameters

::

    SegmentedDirectShowSource (string base_filename [, ...], float "fps", bool "seek",
                               bool "audio", bool "video", bool "convertfps", bool "seekzero",
                               int "timeout", string "pixel_type")

.. describe:: base_filename

    Full or relative path to one or more filenames. Each given filename will
    serve as the "base" filename for loading clips. For example,
    ``SegmentedDirectShowSource("d:\filename.ext")`` will load the files
    d:\filename.\ **00**\ .ext, d:\filename.\ **01**\ .ext and so on, through
    d:\filename.\ **99**\ .ext. Any files in this sequence that don't exist or
    do not have the same media properties as the first clip will be skipped.
    See `Notes`_ below for information on loading clips in multiple directories.

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
    ``SegmentedDirectShowSource(..., pixel_type="+YV12")``, tells AviSynth the
    video rows are DWORD aligned in memory instead of packed. This can fix skew
    or tearing of the decoded video when the width of the picture is not
    divisible by 4.

Notes
-----

Helpful hints
^^^^^^^^^^^^^

* | If you get an *Unrecognized Exception* while reading a VirtualDub-generated
    segmented AVI, delete the small final .avi file.

* If segments are spanned across multiple drives/folders, they can be loaded
  provided the folders are given in the correct order. For example, if you have
  capture files arrange across several folders like this:

 .. image:: pictures/segmentedavisource-img1.png

 To load all segments in order, call this::

    SegmentedAviSource("F:\t1\cap.avi", "F:\t2\cap.avi", "F:\t3\cap.avi")


$Date: 2022/03/28 13:57:17 $

.. _Audio Compression Manager:
    https://en.wikipedia.org/wiki/Windows_legacy_audio_components#Audio_Compression_Manager
.. _UTF-8:
    https://en.wikipedia.org/wiki/UTF-8
.. _planar:
    http://avisynth.nl/index.php/Planar
.. _FourCC:
    http://avisynth.nl/index.php/FourCC
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
