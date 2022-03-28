=====================================================
AVISource / OpenDMLSource / AVIFileSource / WAVSource
=====================================================

**AviSource** takes one or more files - not only `AVI`_ but also `WAV`_ (audio),
`AVS`_ (AviSynth scripts), and `VDR`_ (`VirtualDub`_ frameserver) files. There
is built-in `Audio Compression Manager`_ support for decoding compressed audio
tracks (MP3, AAC, AC3, etc). If more than one file name is given, **AviSource**
returns a clip with all files joined end to end. For the files to be joined, the
media properties must be compatible - see `Notes`_ below.

    **AviSource** tries to read the file(s) using AviSynth's built-in `OpenDML`_
    interface (derived from VirtualDub code) where possible; if the file(s) are
    not in OpenDML format, it uses the Video for Windows `AVIFile`_ interface.

    If **AviSource** has trouble with one or the other interface, **OpenDMLSource**
    and **AviFileSource** will force the use of *OpenDML* or *AVIFile*,
    respectively. Only *OpenDML* can read files larger than 2 GB, but only
    *AVIFile* can read non-AVI files (odd, given its name) like the ones listed
    in the opening paragraph. It can read any file (under 2 GB) for which there
    exist appropriate *stream handlers*.

**WavSource** will open a `WAV`_ file, or an audio-only AVI file. It will also
return the audio stream from a normal video+audio AVI. This might be useful if
your video stream is damaged or unreadable, or simply as a shortcut for
``AviSource(...).KillVideo()`` (:doc:`KillVideo <killaudio>`).


Syntax and Parameters
----------------------

::

    AVISource (string filename [, ...], bool "audio", string "pixel_type", string "fourCC", int "vtrack", int "atrack", bool "utf8")

    OpenDMLSource (string filename [, ...], bool "audio", string "pixel_type", string "fourCC", int "vtrack", int "atrack", bool "utf8")

    AVIFileSource (string filename [, ...], bool "audio", string "pixel_type", string "fourCC", int "vtrack", int "atrack", bool "utf8")

    WAVSource (string filename [, ...], bool "utf8")

.. describe:: filename

    One or more file names. The files will be joined into a single clip with
    :doc:`UnalignedSplice <splice>`. To join files, the media properties must
    be compatible. See `Notes`_ below. `UTF-8`_ file names are supported when
    ``utf8=true``.

.. describe:: audio

    If true, load the first audio stream, or the stream specified by ``atrack``
    if present. If false, audio is disabled.

    Default: true

.. describe:: pixel_type

    Chooses the output color format of the decompressor. Valid values are listed
    in the table below. This argument has no effect if the video is uncompressed,
    as no decompressor will be used in that case. See the `10+ bit inputs`_
    section for a full list of pixel types.

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
    ``AviSource(..., pixel_type="+YV12")``, tells AviSynth the video rows are
    DWORD aligned in memory instead of packed. **This can fix skew or tearing of
    the decoded video** with bad codecs when the width of the picture is not
    divisible by 4.

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


Notes
-----

Joining clips
^^^^^^^^^^^^^

* There is a limit (of about 50, sometimes fewer) **AviSource** calls in script
  - see `discussion`_. If the limit is exceeded, you will see the error message:

    *AVISource: couldn't locate a decompressor for fourcc ....*

 If you need to join more AVIs, try `VirtualDub`_ (limit > 700).

* :doc:`Media properties <../syntax/syntax_clip_properties>` must be
  compatible, meaning they must have:

 #. the same height and width;
 #. the same color format (as presented by the decoder);
 #. the same frame rate (precisely the same, not approximately); and
 #. the same audio sample rate, bit depth and number of channels.

* See the VirtualDub blog post `Appending streams and mismatch errors`_ for a
  more in-depth explanation.

Windows 7 users
^^^^^^^^^^^^^^^

**WavSource** under Windows 7 is unable to load WAV files with 32-bit IEEE Float
samples having the `WAVEFORMAT structure`_. You can use `FFmpeg`_ to rewrite the
header to an extensible format (just do a stream copy; it always writes
extensible headers) ::

    ffmpeg -i "bad.wav" -c copy "good.wav"

Helpful hints
^^^^^^^^^^^^^

* Sometimes the colors will be distorted when loading a `DivX`_ clip in AviSynth
  v2.5 (the chroma channels U and V are swapped), due to a bug in DivX (5.02 and
  older). You can use :ref:`SwapUV` to correct it.

* **AVISource** can also open DV type 1 video input (only video, not audio).

* Some video files get decoded with the wrong color standard ('Rec601'/'Rec709')
  or luma range ('Full'/'TV'). This problem can arise if the input and output
  color formats are different, forcing a :doc:`conversion <convert>`. To avoid
  this conversion, try to specify another, compatible output format - for example:

   * If the video was encoded as RGB, try ``pixel_type="RGB24"`` or "RGB32";
   * If the video was encoded as YUV, try ``pixel_type="YV12"``, "YUY2" or "YV24".

 If that does not work, try `FFmpegSource`_, `LSMASHSource`_ or (if absolutely
 necessary) :doc:`DirectShowSource <directshowsource>`.

.. _AVISource-hbd:

10+ bit inputs
^^^^^^^^^^^^^^

When a classic 'pixel_type' shares more internal formats (such as YUV422P10
first tries to request the v210 then P210 format) you can specify one of the
specific format directly. Note that high bit-depth RGBP (Planar RGB) is
prioritized against packed RGB48/64.

The 'FourCCs for ICDecompressQuery' column means that when a codec supports the
format, it will serve the frame in that one, AviSource then will convert it to
the proper colorspace.

::

    Full support list.
    Non *-marked formats (FourCC column) are supported since r2724.

    'pixel_type' Avs+ Format   FourCC(s) for ICDecompressQuery
    YV24         YV24          *YV24
    YV16         YV16          *YV16
    YV12         YV12          *YV12
    YV411        YV411         *Y41B
    YUY2         YUY2          *YUY2
    RGBP10       RGBP10        G3[0][10]  r210  R10k
    r210         RGBP10        r210
    R10k         RGBP10        R10k
    RGBP         RGBP10        G3[0][10]  r210  R10k
                 RGBP12        G3[0][12]
                 RGBP14        G3[0][14]
                 RGBP16        G3[0][16]
                 RGBAP10       G4[0][10]
                 RGBAP12       G4[0][12]
                 RGBAP14       G4[0][14]
                 RGBAP16       G4[0][16]
    RGB32        RGB32         *BI_RGB internal constant (0) with bitcount=32
    RGB24        RGB24         *BI_RGB internal constant (0) with bitcount=24
    RGB48        RGB48         BGR[48]    b48r
    RGB64        RGB64         *BRA[64]   b64a
    Y8           Y8            Y800       Y8[32][32]   GREY
    Y            Y8            Y800       Y8[32][32]   GREY
                 Y10           Y1[0][10]
                 Y12           Y1[0][12]
                 Y14           Y1[0][14]
                 Y16           Y1[0][16]
    YUV422P10    YUV422P10     v210       P210
    v210         YUV422P10     v210
    P210         YUV422P10     P210
    YUV422P16    YUV422P16     P216
    P216         YUV422P16     P216
    YUV420P10    YUV420P10     P010
    P010         YUV422P10     P010
    YUV420P16    YUV420P16     P016
    P016         YUV422P16     P016
    YUV444P10    YUV444P10     v410
    v410         YUV444P10     v410

More on codecs
^^^^^^^^^^^^^^

Some reference threads:

* `MJPEG codecs`_
* `DV codecs`_


Examples
--------

* C programmers note: backslashes are not doubled; forward slashes work too::

    AVISource("d:\capture.avi")
    WAVSource("f:/soundtrack.wav")

* Splice two clips together; the following statements do the same thing::

    AviSource("cap1.avi") + AviSource("cap2.avi")
    AVISource("cap1.avi", "cap2.avi")

* Splice two clips together where frame rates do not match::

    A = AviSource("FileA.avi") # "29.97" fps (30000/1001)
    B = AviSource("FileB.avi") # 30.0000 fps
    A ++ B.AssumeFPS(A)

* Splice two clips together where one of them contains no audio::

    A = AviSource("FileA.avi") # with audio
    B = AviSource("FileB.avi") # no audio stream
    A ++ AudioDub(B, BlankClip(A)) # insert silent audio with same format

* Disable audio and request RGB32 decompression::

    AVISource("cap.avi", audio=false, pixel_type="RGB32")

* Open a DV, forcing the Canopus DV Codec::

    AviSource("cap.avi", fourCC="CDVC")

* Open a file, forcing the `XviD`_ Codec::

    AviSource("cap.avi", fourCC="XVID")

* Open a YV12 video with a bad codec where the width is not a multiple of four::

    AviSource("test.avi", pixel_type="+YV12")

* Opens the first video and second audio stream of a clip::

    AviSource("test_multi10.avi", vtrack=0, atrack=1)


Changelog
---------

+-----------------+----------------------------------------------------+
| Version         | Changes                                            |
+=================+====================================================+
| AviSynth+ r2768 | Added utf8 filename support.                       |
+-----------------+----------------------------------------------------+
| AviSynth+ r2724 || Added 10+ bits new color formats.                 |
+-----------------+----------------------------------------------------+
| AviSynth 2.6.0  || Added new color formats, "AUTO" and "FULL".       |
|                 || Added multiple video and audio stream support.    |
|                 || Add '+' to pixel_type for padded planar support.  |
+-----------------+----------------------------------------------------+
| AviSynth 2.5.5  | Added fourCC option.                               |
+-----------------+----------------------------------------------------+

$Date: 2022/03/14 07:32:20 $

.. _AVI:
    http://avisynth.nl/index.php/AVI
.. _WAV:
    http://avisynth.nl/index.php/WAV
.. _AVS:
    http://avisynth.nl/index.php/AVS
.. _VDR:
    https://www.virtualdub.org/docs_frameserver.html
.. _VirtualDub:
    http://avisynth.nl/index.php/VirtualDub
.. _Audio Compression Manager:
    https://en.wikipedia.org/wiki/Windows_legacy_audio_components#Audio_Compression_Manager
.. _OpenDML:
    http://www.jmcgowan.com/avitech.html#OpenDML
.. _AVIFile:
    http://www.jmcgowan.com/avitech.html#VFW
.. _planar:
    http://avisynth.nl/index.php/Planar
.. _FourCC:
    http://avisynth.nl/index.php/FourCC
.. _UTF-8:
    https://en.wikipedia.org/wiki/UTF-8
.. _discussion:
    https://forum.doom9.org/showthread.php?t=131687
.. _Appending streams and mismatch errors:
    https://www.virtualdub.org/blog2/entry_073.html
.. _WAVEFORMAT structure:
    https://forum.doom9.org/showthread.php?t=170444
.. _FFmpeg:
    https://ffmpeg.org/
.. _DivX:
    http://avisynth.nl/index.php/DivX
.. _FFmpegSource:
    http://avisynth.nl/index.php/FFmpegSource
.. _LSMASHSource:
    http://avisynth.nl/index.php/LSMASHSource
.. _MJPEG codecs:
    https://forum.doom9.org/showthread.php?s=&postid=330657
.. _DV codecs:
    https://forum.doom9.org/showthread.php?s=&threadid=58110
.. _XviD:
    http://avisynth.nl/index.php/Xvid
