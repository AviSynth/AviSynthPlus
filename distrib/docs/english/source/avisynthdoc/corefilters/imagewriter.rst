===========
ImageWriter
===========

**ImageWriter** writes frames from a clip as a sequence of image files using the
`DevIL library`_ (except when you choose the internal "ebmp" format).

Note that frames are not written to the output file until they are actually
rendered by this filter.


Syntax and Parameters
----------------------

::

    ImageWriter (clip, string "file", int "start", int "end", string "type", bool "info")

.. describe:: clip

    | Source clip; only 8 and 16-bit RGB(A) or greyscale (Y) formats supported.
    | EBMP type only supports 8-bit RGB, YUV or greyscale (Y) formats.

.. describe:: file

    | The path + filename prefix of the saved images.
    | The images will have file names with the pattern: ``file``\ 000000.\ ``type``,
      ``file``\ 000001.\ ``type``, etc.
    | `Sprintf`_ formatting of the filenames is supported.
    | By default, images will be saved as *C:\\000000.ebmp, C:\\000001.ebmp*, etc.

    Default: "c:\\"

.. describe:: start, end

    The start and end of the frame range that will be written.
    They both default to 0 (where ``end=0`` means last frame).
    If ``end`` is negative, it specifies the number of frames that will be written.

    Default: 0, 0

.. describe:: type

    | Set the filename extension that defines the format of the saved image.
    | The supported values for ``type`` are:

    .. table::
        :widths: auto

        +-----------------+------------------------------+
        | Type            | Format                       |
        +=================+==============================+
        | bmp             | `BMP file format`_           |
        +-----------------+------------------------------+
        | ebmp            | `EBMP`_                      |
        +-----------------+------------------------------+
        | dds             | `DirectDraw Surface`_        |
        +-----------------+------------------------------+
        | jpg/jpe/jpeg    | `JPEG`_                      |
        +-----------------+------------------------------+
        | pcx             | `PiCture eXchange`_          |
        +-----------------+------------------------------+
        | png             | `Portable Network Graphics`_ |
        +-----------------+------------------------------+
        | pbm/pgm/ppm     | `Portable pixmap`_           |
        +-----------------+------------------------------+
        | tga             | `Truevision TGA`_            |
        +-----------------+------------------------------+
        | tif/tiff        | `Tag Image File Format`_     |
        +-----------------+------------------------------+
        | raw             | `Raw image format`_          |
        +-----------------+------------------------------+
        | sgi/bw/rgb/rgba | `Silicon Graphics Image`_    |
        +-----------------+------------------------------+

    Default: "ebmp"

.. describe:: info

    When true, overlay progress information on the video clip, showing whether
    a file is being written, and if so, the filename.

    Default: false


Notes
-----

* JPGs are saved with the quality setting set to 99. For RGB input, the colorspace
  is converted to YUV with a 4:2:0 chroma subsampling. When the input is greyscale
  (Y), JPGs are saved in the same Y format.

* Greyscale BMPs are not written correctly by DevIL (the luma is written to all
  three channels instead of a single channel). They should be written as ebmp or
  better yet as PNG or TIFF.


EBMP
~~~~

The internal image format "ebmp" supports all 8-bit color spaces. The "ebmp"
files written from the RGB or Y8 color spaces are standard BMP files, but those
produced from YUV spaces are special and can probably only be read by AviSynth's
:doc:`ImageSource <imagesource>`. This special format allows you to save and
reload raw video in any 8-bit color space.

"EBMP" is an AviSynth extension of the standard Microsoft `RIFF`_ image format
that allows you to save raw image data.


Examples
--------

Export the entire clip in the current native AviSynth format::

    ImageWriter("D:\backup-stills\myvideo")

Write frame 5 to "C:\000005.png" ::

    ImageWriter(start=5, end=5, type="png")

Write frame 5 to "000005.png" into the current directory::

    ImageWriter("", start=5, end=5, type="png")

Write frames 100 till the end to *"F:\pic-000100.jpeg", "F:\pic-000101.jpeg",*
etc. and display progress info::

    ImageWriter(file="F:\pic-", start=100, type="jpeg", info=true)

Write all frames to "F:\00.png", "F:\01.png", ..., "F:\10.png", "F:\11.png", ...,
"F:\100.png", ... (thus adding zeros filling two digits)::

    ImageWriter(file="F:\%02d.png")

Load a jpg image and extract the Y (luma) plane and save it as a greyscale png::

    FFImageSource("GoldPetals.jpg")
    ExtractY()
    ImageWriter("GoldPetals-luma", type="png")

Load a YUV420 jpg image and convert it to YUV444 and save it as an ebmp::

    FFImageSource("GoldPetals.jpg")
    ConvertToYUV444(ChromaInPlacement="MPEG1") # "MPEG2" for YUV422 JPGs
    ImageWriter("GoldPetals-YUV444", type="ebmp")

See the :doc:`ImageReader <imagesource>` page for more information on why the
JPGs in the last two examples were loaded with `FFImageSource`_ instead of the
internal ImageReader filter.

Changelog
----------

+-----------------+------------------------------------------------------------------+
| Version         |                                                                  |
+=================+==================================================================+
| AviSynth+ r2768 | Fix: ImageReader/Writer: path "" means current directory.        |
+-----------------+------------------------------------------------------------------+
| AviSynth+ r2502 | Fix: ImageWriter crash when no '.' in filename.                  |
+-----------------+------------------------------------------------------------------+
| AviSynth+ r2487 || Added support for support RGB48/64 and Y16 formats.             |
|                 || Fix: flip greyscale except when raw.                            |
+-----------------+------------------------------------------------------------------+
| AviSynth 2.6.1  || DevIL library updated to 1.7.8.                                 |
|                 || DevIL.dll is now delay loaded, so Avisynth.dll can be used      |
|                 |  without it, in which case ImageReader/Writer would support only |
|                 |  ebmp mode.                                                      |
+-----------------+------------------------------------------------------------------+
| AviSynth 2.6.0  || ebmp supports all formats; greyscale added for all formats.     |
|                 || Add support for printf formating of filename string, default is |
|                 |  ("%06d.%s", n, ext).                                            |
+-----------------+------------------------------------------------------------------+
| AviSynth 2.5.8  | Added end=-num_frames (supports negative values as -count for    |
|                 | ``end`` argument).                                               |
+-----------------+------------------------------------------------------------------+
| AviSynth 2.5.3  | Added ``info`` parameter.                                        |
+-----------------+------------------------------------------------------------------+
| AviSynth 2.5.2  | Added ``start``, ``end`` and ``type`` parameters.                |
+-----------------+------------------------------------------------------------------+
| AviSynth 2.5.1  | Added ImageWriter filter in limited form.                        |
+-----------------+------------------------------------------------------------------+

$Date: 2022/03/24 12:22:43 $

.. _DevIL library:
    https://github.com/DentonW/DevIL
.. _Sprintf:
    http://www.cplusplus.com/reference/cstdio/sprintf/
.. _BMP file format:
    https://en.wikipedia.org/wiki/BMP_file_format
.. _BMP file format:
    https://en.wikipedia.org/wiki/BMP_file_format
.. _DirectDraw Surface:
    https://en.wikipedia.org/wiki/DirectDraw_Surface
.. _JPEG:
    https://en.wikipedia.org/wiki/JPEG
.. _PiCture eXchange:
    https://en.wikipedia.org/wiki/PCX
.. _Portable Network Graphics:
    https://en.wikipedia.org/wiki/Portable_Network_Graphics
.. _Portable pixmap:
    https://en.wikipedia.org/wiki/Netpbm#File_formats
.. _Truevision TGA:
    https://en.wikipedia.org/wiki/Truevision_TGA
.. _Tag Image File Format:
    https://en.wikipedia.org/wiki/TIFF
.. _Raw image format:
    https://en.wikipedia.org/wiki/Raw_image_format
.. _Silicon Graphics Image:
    https://en.wikipedia.org/wiki/Silicon_Graphics_Image
.. _RIFF:
    https://en.wikipedia.org/wiki/Resource_Interchange_File_Format
.. _FFImageSource:
    http://avisynth.nl/index.php/FFmpegSource
