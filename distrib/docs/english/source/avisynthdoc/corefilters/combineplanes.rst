CombinePlanes
=============

Merges planes of source clip(s) into a target clip.
It is similar to ShufflePlanes in Vapoursynth. Performs the functionality of :doc:`SwapUV <swap>`,
:doc:`YToUV <swap>`, :doc:`MergeChroma <merge>`, :doc:`MergeRGB <mergergb>` and more.

See also :doc:`Extract <extract>`, :doc:`AddAlphaPlane <mask>`, :doc:`RemoveAlphaPlane <mask>`,
and :doc:`ShowU/V <showalpha>` filters.

.. rubric:: Syntax and Parameters

::

    CombinePlanes(clip,
        [string planes, string source_planes, string pixel_type, clip sample_clip ] )

    CombinePlanes(clip, clip,
        [string planes, string source_planes, string pixel_type, clip sample_clip ] )

    CombinePlanes(clip, clip, clip,
        [string planes, string source_planes, string pixel_type, clip sample_clip ] )

    CombinePlanes(clip, clip, clip, clip,
        [string planes, string source_planes, string pixel_type, clip sample_clip ] ) 

.. describe:: clip

    | Source clip(s). At least one is required. Up to four clips are accepted.
    | Each clip defines a color plane in the output, as defined by the ``planes`` and 
      ``source_planes`` arguments. 
    | If the clip count is less than the given ``planes`` defined, then the last available 
      clip is used as a source for all later planes. 

.. describe:: planes = ""

    The target plane order (e.g. "YVU", "YYY", "RGB"). Target planes not listed keep the
    first clip's plane at the same position, or are filled with a neutral value if it has none (see Note 3).

.. describe:: source_planes = "YUVA" or "RGBA" or "YA"

    The source plane order, defaulting to "YUVA", "RGBA" or "YA" depending on the video format
    (default is "YYYY" when all source clips are greyscale). 

    Source clips can even be mixed from greyscale, YA, YUV, YUVA or planar RGB(A) — the only rule being that 
    the relevant source plane character should match with the source clip format, respectively. 

.. describe:: pixel_type

    Set color format of the returned clip. Supports all AVS+ color formats.

    If not given (and no ``sample_clip``), the format of the first clip is used; but when all
    clips are greyscale and ``source_planes`` is not given either, it follows ``planes``:
    YUV(A) 4:4:4, planar RGB(A), or Y+alpha (YA) for ``planes="YA"``. 

.. describe:: sample_clip

        If supplied, output pixel_type will match that of sample_clip. 


Examples
--------

Combine greyscale clips into YUVA clip::

    U8 = source.UToY8()
    V8 = source.VToY8()
    Y8 = source.ConvertToY()
    A8 = source.AddAlphaPlane(128).AToY8()
    CombinePlanes(Y8, U8, V8, A8, planes="YUVA", source_planes="YYYY", 
    \               sample_clip=source) #pixel_type="YUV444P8"

Copy planes between planar RGB(A) and YUV(A) without any conversion
yuv 4:4:4 <-> planar rgb::

    source = last.ConvertBits(32) # 4:4:4
    cast_to_planarrgb = CombinePlanes(source, planes="RGB", source_planes="YUV", 
    \               pixel_type="RGBPS")
    # get back a clip identical with "source"
    cast_to_yuv = CombinePlanes(cast_to_planarrgb, planes="YUV", source_planes="RGB", 
    \               pixel_type="YUV444PS")

Create a black and white planar RGB clip using Y channel.
Source is a YUV clip.::

    grey = CombinePlanes(source, planes="RGB", source_planes="YYY", 
    \               pixel_type="RGBP8")

Copy luma from one clip, U and V from another::

    #Source is the template
    #SourceY is a Y or YUV clip
    #SourceUV is a YUV clip
    grey = CombinePlanes(sourceY, sourceUV, planes="YUV",
    \               source_planes="YUV", sample_clip = source)

Initialize target planes which have no source (see Note 3)::

    # Y only clip to YUV 4:2:0. U and V are not listed and the greyscale
    # first clip has no U/V plane: they are filled with chroma center (128).
    grey = source.ConvertToY()
    CombinePlanes(grey, planes="Y", source_planes="Y", pixel_type="YUV420P8")

    # YUV to YUVA: alpha is not listed and source has no alpha,
    # so the alpha plane is filled with fully opaque (255).
    CombinePlanes(source, planes="YUV", source_planes="YUV", pixel_type="YUVA420P8")

``_ChromaLocation`` taken from the U/V source clip (see Frame properties)::

    a = source.ConvertToYUV420(ChromaOutPlacement="left")     # _ChromaLocation = 0 (left)
    b = source.ConvertToYUV420(ChromaOutPlacement="top_left") # _ChromaLocation = 2 (top_left)
    # Y from a, U and V from b: result gets b's _ChromaLocation (top_left),
    # all other frame properties come from a
    CombinePlanes(a, b, planes="YUV", source_planes="YUV")
    # U from a, V from b: sitings differ, _ChromaLocation is removed
    CombinePlanes(a, a, b, planes="YUV", source_planes="YUV")

Notes
-----

One optimization in CombinePlanes is aimed to have one less memory (plane) copy.

Theory behind: when a frame has exactly one 'user' (no other frames are yet referencing it) then it can 
directly be grabbed and made writable without any frame plane content copying.
When this "I'm the only one" condition is fulfilled and the below-written conditions are set 
then it can be a bit quicker than using the ordinary "make a new frame and copy the referenced input
frames into that" logic.

* Source clip has the same format as the target, and the first plane ID is the same.
  Y comes from first clip (no Y plane copy, the input frame containing Y is reused), U and V are copied

* Second clip has the same format as the target, and the 2nd and 3rd plane ID is the same
  U and V comes from 2nd clip (no UV copy, frame containing U and V is reused), 
  while Y (or the given first plane ID) is copied from first clip. When there is a 

Example::

    Colorbars(pixel_type="YV12")
    ConvertBits(16)
    a=last # UV is kept
    Blur(1)
    #luma comes from LAST, a's UV is copied to last
    x=MergeLuma(a,last)
    y=CombinePlanes(last,a,planes="YUV",pixel_type="YUV420P16")
    y  # or x
    Prefetch(4)

Comparison: new CombinePlanes and the usual MergeLuma showed ~4600 fps while old CombinePlanes run at only 3540 fps

Note 2
------

Non-planar formats such as packed RGB or YUY2 inputs will automatically converted to planar RGB or YV16 before CombinePlanes.

Note 3
------

When there is only one input clip, a zero-cost (BitBlt-less, using "subframes") method is used, which is much faster.
It requires that the target has the same dimensions and `chroma subsampling`_ as the source clip (and
alpha only if the source has alpha); otherwise the planes are copied.

Such cases are:

* casting YUV to RGB

* shuffle RGBA to ABGR

* U to Y

* etc..

Target planes that are not listed in ``planes`` keep the first clip's plane from the same plane slot.
Plane slots follow the internal plane order: 1st slot is Y or G, 2nd is U or B, 3rd is V or R, 4th is A.
This matters when the target and the first clip differ in color family (cast between YUV and planar RGB):
e.g. for a YUV 4:4:4 ``clipYUV``, ``CombinePlanes(clipYUV, planes="R", source_planes="Y", pixel_type="RGBP8")`` sets R from Y,
while the unlisted G and B are copied unchanged from Y and U (2nd slot), without any conversion.
If the first clip has no such plane of the same size (e.g. U/V for a greyscale first clip, or alpha), they 
are filled with a neutral value: U/V chroma center, alpha fully opaque, Y/R/G/B black (0, or 16 scaled to 
the bit depth for limited range: by the first clip's ``_ColorRange``, without it RGB is full, YUV/Y limited).

Examples::

    combineplanes(clipRGBP, planes="RGB",source_planes="BGR") # swap R and B
    combineplanes(clipYUV, planes="GBRA",source_planes="YUVA",pixel_type="RGBAP8") # cast YUVA to planar RGBA
    combineplanes(clipYUV, planes="Y",source_planes="U",pixel_type="Y8") # extract U

Frame properties
----------------

Frame properties are copied from the first clip, except ``_ChromaLocation``, which describes
the chroma planes and so follows the clips the target U and V planes come from
(chroma placement is explained in :doc:`Sampling <../advancedtopics/sampling>`, the property values
in :doc:`frame properties <../syntax/syntax_internal_functions_frame_properties>`; they follow
``chroma_sample_loc_type`` of `ITU-T H.264`_ Annex E):

* Target without subsampled chroma (Y, YA, 4:4:4, planar RGB): ``_ChromaLocation`` is removed.
* U and/or V taken from the U/V plane of a subsampled clip with the same subsampling as the
  target: that clip's ``_ChromaLocation`` is used (removed if it has none). If U and V come
  from such clips with different (or missing vs. present) ``_ChromaLocation``, it is removed,
  since no single siting describes the result.
  A U or V plane not listed in ``planes`` keeps the first clip's plane, so it counts as taken
  from the first clip.
* Otherwise, e.g. U and V taken from greyscale clips (``ExtractU``/``ExtractV`` output, which
  carries no siting): the first clip's ``_ChromaLocation`` is kept. This keeps the usual
  "extract, process, recombine" workflow intact::

      u = src.ExtractU().Blur(1.0)
      v = src.ExtractV().Blur(1.0)
      CombinePlanes(src, u, v, planes="YUV", source_planes="YYY") # keeps src's _ChromaLocation

Other frame properties, e.g. ``_ColorRange`` or ``_Matrix``, are simply taken from the first clip:
CombinePlanes does not check whether the clips agree. Combining e.g. a limited range luma with
full range chroma, or putting a limited range Y into a full range first clip, is the user's
responsibility; set the properties of the result accordingly (``propSet``).

Changelog
---------

.. table::
    :widths: auto

    +-----------------+----------------------------------------------+
    | Version         | Changes                                      |
    +=================+==============================================+
    | AviSynth 3.7.6  | YA: ``planes="YA"`` from greyscale clips     |
    |                 | gives YA, default source planes "YA" for a   |
    |                 | YA target                                    |
    |                 |                                              |
    |                 | ``_ChromaLocation`` follows the U/V source   |
    |                 | clips; removed for targets without           |
    |                 | subsampled chroma                            |
    |                 |                                              |
    |                 | Unlisted target planes keep the first clip's |
    |                 | plane at the same position or get a neutral  |
    |                 | value (they were undefined before)           |
    +-----------------+----------------------------------------------+
    | AviSynth 3.7.1  | a bit optimized MergeLuma-like cases         |
    +-----------------+----------------------------------------------+
    | 20161110        | First added                                  |
    +-----------------+----------------------------------------------+

$Date: 2026/10/02 09:00:00 $

.. _chroma subsampling:
    https://en.wikipedia.org/wiki/Chroma_subsampling
.. _ITU-T H.264:
    https://www.itu.int/rec/T-REC-H.264
