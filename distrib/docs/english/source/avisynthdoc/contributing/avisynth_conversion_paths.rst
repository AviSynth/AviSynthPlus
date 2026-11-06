AviSynth+ conversion paths
==========================

Which classes and pixel functions run for a format conversion, for those working on the
source code or looking for the reason of a speed or quality difference. It follows
``convert.cpp``, ``convert_planar.cpp`` and ``convert_rgb.cpp`` of ``avs_core/convert``.
For the script syntax see :doc:`Convert <../corefilters/convert>`.

Pixel functions are listed from the fastest to the slowest, the first one the CPU supports is used.
A ``_mmx`` function exists in 32 bit x86 builds only, a ``_neon`` function on aarch64.
Where no NEON function is listed, aarch64 uses the C function, compiled for its CPU.

.. contents:: Table of contents
    :local:

YUV and YUY2 to RGB
-------------------

Entry point: ``CreateConvertToRGB``. All planar YUV sources go to 4:4:4 first.

.. table::
    :widths: auto

    +-------------------------+---------------+----------------------------------------+---------------------------------------+
    | Source                  | Target        | Steps                                  | Pixel functions                       |
    +=========================+===============+========================================+=======================================+
    | YUY2                    | any RGB       | ConvertYUY2ToYV16_or_Y, then as YV16   | convert_yuy2_to_yv16_sse2             |
    |                         |               |                                        | convert_yuy2_to_yv16_mmx              |
    |                         |               |                                        | convert_yuy2_to_yv16_c                |
    +-------------------------+---------------+----------------------------------------+---------------------------------------+
    | YUV 4:1:0, 4:1:1,       | any RGB       | ConvertToPlanarGeneric to 4:4:4,       | Chroma resizer, see `YUV to YUV`_     |
    | 4:2:0, 4:2:2, 4:4:0     |               | then as 4:4:4                          |                                       |
    +-------------------------+---------------+----------------------------------------+---------------------------------------+
    | 8 bit 4:4:4             | RGB24, RGB32  | ConvertYUV444ToRGB, packed output.     | convert_yv24_to_rgb_avx2              |
    |                         |               | Not with ``quality=true``.             | convert_yv24_to_rgb_ssse3 (RGB24)     |
    |                         |               |                                        | convert_yv24_to_rgb_sse2              |
    |                         |               |                                        | convert_yv24_to_rgb_mmx (no alpha)    |
    |                         |               |                                        | convert_yv24_to_rgb_neon              |
    |                         |               |                                        | convert_yv24_to_rgb_c                 |
    +-------------------------+---------------+----------------------------------------+---------------------------------------+
    | 4:4:4, other bit depth, | RGB24, RGB32  | ConvertYUV444ToRGB, planar output,     | See the planar row, and `Planar RGB`_ |
    | or ``quality=true``     |               | then PlanarRGBtoPackedRGB              |                                       |
    +-------------------------+---------------+----------------------------------------+---------------------------------------+
    | 4:4:4, any bit depth    | RGB48, RGB64  | ConvertYUV444ToRGB, planar output,     | See the planar row, and `Planar RGB`_ |
    |                         |               | then PlanarRGBtoPackedRGB              |                                       |
    +-------------------------+---------------+----------------------------------------+---------------------------------------+
    | 4:4:4, any bit depth    | Planar RGB(A) | ConvertYUV444ToRGB, with the bit depth | convert_yuv_to_planarrgb_avx2         |
    |                         |               | change in the same step                | convert_yuv_to_planarrgb_sse2         |
    |                         |               |                                        | convert_yuv_to_planarrgb_c            |
    +-------------------------+---------------+----------------------------------------+---------------------------------------+

The alpha plane is handled in ``ConvertYUV444ToRGB::GetFrame``: copied, converted to the new bit
depth by the function from ``get_convert_any_bits_functions``, or filled.

RGB to YUV and Y
----------------

Entry point: ``ConvertToPlanarGeneric::Create``. Packed RGB always goes through planar RGB, this is
faster on AVX2 than a direct conversion.

.. table::
    :widths: auto

    +---------------+-----------------------+---------------------------------------------+----------------------------------------+
    | Source        | Target                | Steps                                       | Pixel functions                        |
    +===============+=======================+=============================================+========================================+
    | Packed RGB    | any YUV, Y            | CreateConvertToRGB to planar RGB, RGBA when | See `Packed RGB to planar RGB`_        |
    |               |                       | the target keeps alpha, then as planar      |                                        |
    +---------------+-----------------------+---------------------------------------------+----------------------------------------+
    | Planar RGB(A) | YUV 4:4:4, YUVA 4:4:4 | ConvertRGBToYUV444, with the bit depth      | convert_yuv_to_planarrgb_avx2          |
    |               |                       | change in the same step                     | convert_yuv_to_planarrgb_sse2          |
    |               |                       |                                             | convert_yuv_to_planarrgb_c             |
    +---------------+-----------------------+---------------------------------------------+----------------------------------------+
    | Planar RGB(A) | Y, YA                 | ConvertRGBToYUV444 with ``to_y=true``,      | The same functions, RGB_TO_Y direction |
    |               |                       | only Y is written                           |                                        |
    +---------------+-----------------------+---------------------------------------------+----------------------------------------+
    | Any RGB       | YUV 4:1:0, 4:1:1,     | As above to 4:4:4, then                     | Chroma resizer, see `YUV to YUV`_      |
    |               | 4:2:0, 4:2:2, 4:4:0   | ConvertToPlanarGeneric                      |                                        |
    +---------------+-----------------------+---------------------------------------------+----------------------------------------+

YUV to Y
--------

.. table::
    :widths: auto

    +------------+--------------------+-------------------------------------------+-------------------------------+
    | Source     | Target             | Steps                                     | Pixel functions               |
    +============+====================+===========================================+===============================+
    | YUY2       | Y8                 | ConvertYUY2ToYV16_or_Y with ``to_y=true`` | convert_yuy2_to_y8_sse2       |
    |            |                    |                                           | convert_yuy2_to_y8_c          |
    +------------+--------------------+-------------------------------------------+-------------------------------+
    | Planar YUV | Y, same bit depth  | ConvertToPlanarGeneric                    | None, Subframe of the Y plane |
    +------------+--------------------+-------------------------------------------+-------------------------------+
    | Planar YUV | Y, other bit depth | ConvertToPlanarGeneric, then ConvertBits  | Subframe, then ConvertBits    |
    +------------+--------------------+-------------------------------------------+-------------------------------+

Planar RGB
----------

.. table::
    :widths: auto

    +----------------------+----------------------+---------------------------------------+---------------------------+
    | Source               | Target               | Steps                                 | Pixel functions           |
    +======================+======================+=======================================+===========================+
    | Planar RGB           | Planar RGBA          | AddAlphaPlane                         | fill_plane                |
    +----------------------+----------------------+---------------------------------------+---------------------------+
    | Planar RGBA          | Planar RGB           | RemoveAlphaPlane                      | None, SubframePlanar      |
    +----------------------+----------------------+---------------------------------------+---------------------------+
    | Planar RGB(A), n bit | Planar RGB(A), m bit | Alpha add or remove, then ConvertBits | See `ConvertBits`_        |
    +----------------------+----------------------+---------------------------------------+---------------------------+
    | 8 or 16 bit          | RGB32, RGB64         | PlanarRGBtoPackedRGB                  | convert_rgbp_to_rgba_avx2 |
    |                      |                      |                                       | convert_rgbp_to_rgba_sse2 |
    |                      |                      |                                       | convert_rgbp_to_rgb_c     |
    +----------------------+----------------------+---------------------------------------+---------------------------+
    | 8 or 16 bit          | RGB24, RGB48         | PlanarRGBtoPackedRGB                  | convert_rgbp_to_rgb_c     |
    +----------------------+----------------------+---------------------------------------+---------------------------+

Packed RGB to planar RGB
------------------------

.. table::
    :widths: auto

    +--------------+---------------+----------------------------------------+--------------------------------------------------+
    | Source       | Target        | Steps                                  | Pixel functions                                  |
    +==============+===============+========================================+==================================================+
    | RGB24, RGB48 | Planar RGB(A) | PackedRGBtoPlanarRGB, then ConvertBits | convert_rgb_to_rgbp_avx2 (width 32, 16 at RGB48) |
    |              |               | if the bit depth changes               | convert_rgb_to_rgbp_ssse3 (width 16, 8 at RGB48) |
    |              |               |                                        | convert_rgb_to_rgbp_sse2 (width 16, 8 at RGB48)  |
    |              |               |                                        | convert_rgb_to_rgbp_c                            |
    +--------------+---------------+----------------------------------------+--------------------------------------------------+
    | RGB32, RGB64 | Planar RGB(A) | PackedRGBtoPlanarRGB, then ConvertBits | convert_rgba_to_rgbp_avx512vbmi                  |
    |              |               | if the bit depth changes               | convert_rgba_to_rgbp_avx2                        |
    |              |               |                                        | convert_rgba_to_rgbp_ssse3                       |
    |              |               |                                        | convert_rgba_to_rgbp_sse2                        |
    |              |               |                                        | convert_rgb_to_rgbp_c                            |
    +--------------+---------------+----------------------------------------+--------------------------------------------------+

Packed RGB to packed RGB
------------------------

A bit depth change between 8 and 16 bit comes first, by ConvertBits.

.. table::
    :widths: auto

    +--------+--------+-----------+------------------------------+
    | Source | Target | Steps     | Pixel functions              |
    +========+========+===========+==============================+
    | RGB24  | RGB32  | RGBtoRGBA | convert_rgb24_to_rgb32_ssse3 |
    |        |        |           | convert_rgb24_to_rgb32_neon  |
    |        |        |           | convert_rgb24_to_rgb32_c     |
    +--------+--------+-----------+------------------------------+
    | RGB48  | RGB64  | RGBtoRGBA | convert_rgb48_to_rgb64_ssse3 |
    |        |        |           | convert_rgb48_to_rgb64_neon  |
    |        |        |           | convert_rgb48_to_rgb64_c     |
    +--------+--------+-----------+------------------------------+
    | RGB32  | RGB24  | RGBAtoRGB | convert_rgb32_to_rgb24_ssse3 |
    |        |        |           | convert_rgb32_to_rgb24_sse2  |
    |        |        |           | convert_rgb32_to_rgb24_neon  |
    |        |        |           | convert_rgb32_to_rgb24_c     |
    +--------+--------+-----------+------------------------------+
    | RGB64  | RGB48  | RGBAtoRGB | convert_rgb64_to_rgb48_ssse3 |
    |        |        |           | convert_rgb64_to_rgb48_neon  |
    |        |        |           | convert_rgb64_to_rgb48_c     |
    +--------+--------+-----------+------------------------------+

YUY2 and YV16
-------------

.. table::
    :widths: auto

    +--------+--------+------------------------------------+---------------------------------------+
    | Source | Target | Steps                              | Pixel functions                       |
    +========+========+====================================+=======================================+
    | YUY2   | YV16   | ConvertYUY2ToYV16_or_Y             | convert_yuy2_to_yv16_sse2             |
    |        |        |                                    | convert_yuy2_to_yv16_mmx              |
    |        |        |                                    | convert_yuy2_to_yv16_c                |
    +--------+--------+------------------------------------+---------------------------------------+
    | YV16   | YUY2   | ConvertYV16ToYUY2                  | convert_yv16_to_yuy2_sse2             |
    |        |        |                                    | convert_yv16_to_yuy2_mmx              |
    |        |        |                                    | convert_yv16_to_yuy2_c                |
    +--------+--------+------------------------------------+---------------------------------------+
    | Any    | YUY2   | CreateYUV422 with ``bits=8``, then | The path to 4:2:2, then the row above |
    |        |        | ConvertYV16ToYUY2                  |                                       |
    +--------+--------+------------------------------------+---------------------------------------+

YUV to YUV
----------

.. table::
    :widths: auto

    +------------+----------------------+------------------------------------------+----------------------------------------------+
    | Source     | Target               | Steps                                    | Pixel functions                              |
    +============+======================+==========================================+==============================================+
    | Planar YUV | Other subsampling    | ConvertToPlanarGeneric                   | Y: BitBlt. U, V: SwapUVToY and               |
    |            |                      |                                          | FilteredResize::CreateResize, per field      |
    |            |                      |                                          | when interlaced. Alpha: BitBlt or fill_plane |
    +------------+----------------------+------------------------------------------+----------------------------------------------+
    | Planar YUV | Same subsampling and | No conversion, the clip is returned,     | None                                         |
    |            | chroma placement,    | only alpha is added or removed           |                                              |
    |            | same bit depth       |                                          |                                              |
    +------------+----------------------+------------------------------------------+----------------------------------------------+
    | Planar YUV | Other bit depth      | ConvertToPlanarGeneric, then ConvertBits | See `ConvertBits`_                           |
    +------------+----------------------+------------------------------------------+----------------------------------------------+
    | Y, YA      | YUV                  | ConvertToPlanarGeneric                   | Y: BitBlt. U, V: fill_chroma with neutral    |
    +------------+----------------------+------------------------------------------+----------------------------------------------+

ConvertBits
-----------

The pixel functions come from ``get_convert_any_bits_functions`` (``convert_bits.cpp``), for luma, chroma
and alpha separately. Each has variants for limited and full range, and for dither.

.. table::
    :widths: auto

    +---------------------------------------+-----------------------------------+
    | From, to                              | Pixel functions                   |
    +=======================================+===================================+
    | 8-16 bit to 8-16 bit                  | convert_uint_avx2                 |
    |                                       | convert_uint_sse41                |
    |                                       | convert_uint_limited_avx2         |
    |                                       | convert_uint_limited_sse41        |
    |                                       | convert_uint_c                    |
    |                                       | convert_uint_limited_c            |
    +---------------------------------------+-----------------------------------+
    | 8-16 bit to 8-16 bit, ordered dither  | convert_ordered_dither_uint_avx2  |
    |                                       | convert_ordered_dither_uint_sse41 |
    |                                       | convert_ordered_dither_uint_neon  |
    |                                       | convert_ordered_dither_uint_c     |
    +---------------------------------------+-----------------------------------+
    | 8-16 bit to 8-16 bit, Floyd-Steinberg | convert_uint_floyd_c              |
    +---------------------------------------+-----------------------------------+
    | 8-16 bit to float                     | convert_uintN_to_float_avx2       |
    |                                       | convert_uintN_to_float_c          |
    +---------------------------------------+-----------------------------------+
    | Float to 8-16 bit                     | convert_32_to_uintN_avx2          |
    |                                       | convert_32_to_uintN_sse41         |
    |                                       | convert_32_to_uintN_c             |
    +---------------------------------------+-----------------------------------+
    | Float to float                        | convert_float_to_float_c          |
    +---------------------------------------+-----------------------------------+

The matrix function
-------------------

``convert_yuv_to_planarrgb_avx2``, ``_sse2`` and ``_c`` (``convert_planar.cpp``) are one template for all the matrix
conversions. The direction is a template parameter, alpha is not handled here but in the calling ``GetFrame``.

.. table::
    :widths: auto

    +---------------------+---------------------------------------+---------------+---------------+
    | ConversionDirection | Used by                               | Source planes | Target planes |
    +=====================+=======================================+===============+===============+
    | YUV_TO_RGB          | ConvertYUV444ToRGB, planar output     | Y, U, V       | G, B, R       |
    +---------------------+---------------------------------------+---------------+---------------+
    | RGB_TO_YUV          | ConvertRGBToYUV444                    | G, B, R       | Y, U, V       |
    +---------------------+---------------------------------------+---------------+---------------+
    | RGB_TO_Y            | ConvertRGBToYUV444 with ``to_y=true`` | G, B, R       | Y             |
    +---------------------+---------------------------------------+---------------+---------------+

``YUV_TO_YUV`` and ``RGB_TO_RGB`` are in the enum as well, but they are unused.

Inside it, ``YuvRgbConversionType`` selects the arithmetic:

.. table::
    :widths: auto

    +----------------------+-------------------------------------------------------+
    | YuvRgbConversionType | Used for                                              |
    +======================+=======================================================+
    | NATIVE_INT           | Same bit depth, integer matrix                        |
    +----------------------+-------------------------------------------------------+
    | FORCE_FLOAT          | Float matrix: float input, and ``quality=true``       |
    +----------------------+-------------------------------------------------------+
    | FLOAT_OUTPUT         | Integer matrix, float output                          |
    +----------------------+-------------------------------------------------------+
    | BITCONV_INT_LIMITED  | Bit depth change, limited range, by shifting          |
    +----------------------+-------------------------------------------------------+
    | BITCONV_INT_FULL     | Bit depth change, full range, with float coefficients |
    +----------------------+-------------------------------------------------------+

