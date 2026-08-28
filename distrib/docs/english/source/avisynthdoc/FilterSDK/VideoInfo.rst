
VideoInfo
=========

The VideoInfo structure holds global information about a clip (i.e.
information that does not depend on the frame number). The GetVideoInfo
method in IClip returns this structure. Below is a description of it
(for AVISYNTH_INTERFACE_VERSION=6 and above).

.. toctree::
    :maxdepth: 3

.. contents:: Table of contents


Properties and constants
------------------------

**General properties:**
::

    int width, height; // width=0 means no video
    unsigned fps_numerator, fps_denominator;
    int num_frames; // max. num_frames = 2,147,483,647 (signed int32)
    int audio_samples_per_second; // audio_samples_per_second=0 means no audio
    int sample_type; // samples types are defined in avisynth.h
    uint64_t num_audio_samples;
    int nchannels;


**Colorspace properties and constants:**
::

    int pixel_type;
    
    // Colorspace properties.
    /*
      Planar match mask  1111.1000.0000.0111.0000.0111.0000.0111
      Planar signature   10xx.1000.0000.00xx.0000.00xx.00xx.00xx ?
      Planar signature   10xx.1000.0000.0xxx.0000.00xx.000x.x0xx ? *new
      Planar filter mask 1111.1111.1111.1111.1111.1111.1110.0111 (typo from old header fixed)
      
      pixel_type mapping
      ==================
      pixel_type bit-map PIYB.Z000.0???.0SSS.0000.0???.????.????
              planar YUV            CCC            HHH.000u.vWWW
           planar RGB(A)            CCC                       AR
               nonplanar            CCC            000.00wx xyAR
      Legend
      ======
      Planar YUV:
        Code Bits Remark
        W    0-2  Planar Width Subsampling bits
                  Use (X+1) & 3 for GetPlaneWidthSubsampling
                    000 => 1        YV12, YV16, YUV420, YUV422
                    001 => 2        YV411, YUV9
                    010 => reserved
                    011 => 0        YV24, YUV444, RGBP
                    1xx => reserved
        v    3    VPlaneFirst YV12, YV16, YV24, YV411, YUV9
        u    4    UPlaneFirst I420
        H    7-9  Planar Height Subsampling bits
                  Use ((X>>8)+1) & 3 for GetPlaneHeightSubsampling
                    000 => 1        YV12, YUV420
                    001 => 2        YUV9
                    010 => reserved
                    011 => 0        YV16, YV24, YV411, YUV422, YUV444, RGBP
                    1xx => reserved
      
      Planar RGB
       Code Bits Remark
         R   0   BGR,  (with SSS bits for 8/16 bit/sample or float)
         A   1   BGRA, (with SSS bits for 8/16 bit/sample or float)
      
      
      Not Planar, Interleaved (I flag)
      Code Bits Remark
        R   0   BGR24, and BGRx in future (with SSS bits for 8/16 bit/sample or float)
        A   1   BGR32, and BGRAx in future (with SSS bits for 8/16 bit/sample or float)
        y   2   YUY2
        x   3-4 reserved
        w   5   Raw32
      
      General
      Code Bits Remark
        S 16-18 Sample resolution bits
                000 => 8
                001 => 16
                010 => 32 (float)
                011,100 => reserved
                101 => 10 bits
                110 => 12 bits
                111 => 14 bits
      for packed RGB(A): only 8 and 16 bits are valid
      
      Other YV12 specific (will never be used)
        C  20-23 Chroma Placement values 0-4; see CS_xxx_CHROMA_PLACEMENT
      
      Color family and layout
                             Packed      Planar               Planar  Planar
      Code Bits Remark       RGB/RGBA     YUV  YUY2  Y_Grey  RGB/RGBA  YUVA
        R   0                  1/0         -    0      -       1/0       -
        A   1                  0/1         -    0      -       0/1       -
        y   2                   -          -    1      -        0        -
        Z  27   YUVA            0          0    0      0        1        1
        B  28   BGR             1          0    0      0        1*       0
        Y  29   YUV             0          1    1      1        0        0
        I  30   Interleaved     1          0    1      1        0        0
        P  31   Planar          0          1    0      1        1        1
      * Planar RGB plane order: G,B,R(,A)
    */
    enum AvsColorFormat {
      CS_YUVA        = 1 << 27,
      CS_BGR         = 1 << 28,
      CS_YUV         = 1 << 29,
      CS_INTERLEAVED = 1 << 30,
      CS_PLANAR      = 1 << 31,
    
      CS_Shift_Sub_Width   =  0,
      CS_Shift_Sub_Height  =  8,
      CS_Shift_Sample_Bits = 16,
    
      CS_Sub_Width_Mask    = 7 << CS_Shift_Sub_Width,
      CS_Sub_Width_1       = 3 << CS_Shift_Sub_Width, // YV24
      CS_Sub_Width_2       = 0 << CS_Shift_Sub_Width, // YV12, I420, YV16
      CS_Sub_Width_4       = 1 << CS_Shift_Sub_Width, // YUV9, YV411
    
      CS_VPlaneFirst       = 1 << 3, // YV12, YV16, YV24, YV411, YUV9
      CS_UPlaneFirst       = 1 << 4, // I420
    
      CS_Sub_Height_Mask   = 7 << CS_Shift_Sub_Height,
      CS_Sub_Height_1      = 3 << CS_Shift_Sub_Height, // YV16, YV24, YV411
      CS_Sub_Height_2      = 0 << CS_Shift_Sub_Height, // YV12, I420
      CS_Sub_Height_4      = 1 << CS_Shift_Sub_Height, // YUV9
    
      CS_Sample_Bits_Mask  = 7 << CS_Shift_Sample_Bits,
      CS_Sample_Bits_8     = 0 << CS_Shift_Sample_Bits,
      CS_Sample_Bits_10    = 5 << CS_Shift_Sample_Bits,
      CS_Sample_Bits_12    = 6 << CS_Shift_Sample_Bits,
      CS_Sample_Bits_14    = 7 << CS_Shift_Sample_Bits,
      CS_Sample_Bits_16    = 1 << CS_Shift_Sample_Bits,
      CS_Sample_Bits_32    = 2 << CS_Shift_Sample_Bits,
    
      CS_PLANAR_MASK       = CS_PLANAR | CS_INTERLEAVED | CS_YUV | CS_BGR | CS_YUVA
                             | CS_Sample_Bits_Mask | CS_Sub_Width_Mask | CS_Sub_Height_Mask,
      CS_PLANAR_FILTER     = ~(CS_VPlaneFirst | CS_UPlaneFirst),
    
      CS_RGB_TYPE  = 1 << 0,
      CS_RGBA_TYPE = 1 << 1,
    
      // Specific colorformats
      CS_UNKNOWN = 0,
    
      CS_BGR24 = CS_RGB_TYPE  | CS_BGR | CS_INTERLEAVED,
      CS_BGR32 = CS_RGBA_TYPE | CS_BGR | CS_INTERLEAVED,
      CS_YUY2  = 1 << 2 | CS_YUV | CS_INTERLEAVED,
      //  CS_YV12  = 1 << 3  Reserved
      //  CS_I420  = 1 << 4  Reserved
      CS_RAW32 = 1 << 5 | CS_INTERLEAVED,
    
      //  YV12 must be 0xA0000008. v2.5 Baked API will see all new planar as YV12.
      //  I420 must be 0xA0000010.
    
      CS_GENERIC_YUV444  = CS_PLANAR | CS_YUV | CS_VPlaneFirst | CS_Sub_Width_1 | CS_Sub_Height_1,  // 4:4:4 planar
      CS_GENERIC_YUV422  = CS_PLANAR | CS_YUV | CS_VPlaneFirst | CS_Sub_Width_2 | CS_Sub_Height_1,  // 4:2:2 planar
      CS_GENERIC_YUV420  = CS_PLANAR | CS_YUV | CS_VPlaneFirst | CS_Sub_Width_2 | CS_Sub_Height_2,  // 4:2:0 planar
      CS_GENERIC_Y       = CS_PLANAR | CS_INTERLEAVED | CS_YUV,                                     // Y only (4:0:0)
      CS_GENERIC_RGBP    = CS_PLANAR | CS_BGR | CS_RGB_TYPE,                                        // planar RGB. Though name is RGB but plane order G,B,R
      CS_GENERIC_RGBAP   = CS_PLANAR | CS_BGR | CS_RGBA_TYPE,                                       // planar RGBA
      CS_GENERIC_YUVA444 = CS_PLANAR | CS_YUVA | CS_VPlaneFirst | CS_Sub_Width_1 | CS_Sub_Height_1, // 4:4:4:A planar
      CS_GENERIC_YUVA422 = CS_PLANAR | CS_YUVA | CS_VPlaneFirst | CS_Sub_Width_2 | CS_Sub_Height_1, // 4:2:2:A planar
      CS_GENERIC_YUVA420 = CS_PLANAR | CS_YUVA | CS_VPlaneFirst | CS_Sub_Width_2 | CS_Sub_Height_2, // 4:2:0:A planar
    
      CS_YV24  = CS_GENERIC_YUV444 | CS_Sample_Bits_8,  // YVU 4:4:4 planar
      CS_YV16  = CS_GENERIC_YUV422 | CS_Sample_Bits_8,  // YVU 4:2:2 planar
      CS_YV12  = CS_GENERIC_YUV420 | CS_Sample_Bits_8,  // YVU 4:2:0 planar
      CS_I420  = CS_PLANAR | CS_YUV | CS_Sample_Bits_8 | CS_UPlaneFirst | CS_Sub_Width_2 | CS_Sub_Height_2,  // YUV 4:2:0 planar
      CS_IYUV  = CS_I420,
      CS_YUV9  = CS_PLANAR | CS_YUV | CS_Sample_Bits_8 | CS_VPlaneFirst | CS_Sub_Width_4 | CS_Sub_Height_4,  // YUV 4:1:0 planar
      CS_YV411 = CS_PLANAR | CS_YUV | CS_Sample_Bits_8 | CS_VPlaneFirst | CS_Sub_Width_4 | CS_Sub_Height_1,  // YUV 4:1:1 planar
    
      CS_Y8    = CS_GENERIC_Y | CS_Sample_Bits_8,                                                            // Y   4:0:0 planar
    
      //-------------------------
      // AVS16: new planar constants go live! Experimental PF 160613
      // 10-12-14 bit + planar RGB + BRG48/64 160725
    
      CS_YUV444P10 = CS_GENERIC_YUV444 | CS_Sample_Bits_10, // YUV 4:4:4 10bit samples
      CS_YUV422P10 = CS_GENERIC_YUV422 | CS_Sample_Bits_10, // YUV 4:2:2 10bit samples
      CS_YUV420P10 = CS_GENERIC_YUV420 | CS_Sample_Bits_10, // YUV 4:2:0 10bit samples
      CS_Y10 = CS_GENERIC_Y | CS_Sample_Bits_10,            // Y   4:0:0 10bit samples
    
      CS_YUV444P12 = CS_GENERIC_YUV444 | CS_Sample_Bits_12, // YUV 4:4:4 12bit samples
      CS_YUV422P12 = CS_GENERIC_YUV422 | CS_Sample_Bits_12, // YUV 4:2:2 12bit samples
      CS_YUV420P12 = CS_GENERIC_YUV420 | CS_Sample_Bits_12, // YUV 4:2:0 12bit samples
      CS_Y12 = CS_GENERIC_Y | CS_Sample_Bits_12,            // Y   4:0:0 12bit samples
    
      CS_YUV444P14 = CS_GENERIC_YUV444 | CS_Sample_Bits_14, // YUV 4:4:4 14bit samples
      CS_YUV422P14 = CS_GENERIC_YUV422 | CS_Sample_Bits_14, // YUV 4:2:2 14bit samples
      CS_YUV420P14 = CS_GENERIC_YUV420 | CS_Sample_Bits_14, // YUV 4:2:0 14bit samples
      CS_Y14 = CS_GENERIC_Y | CS_Sample_Bits_14,            // Y   4:0:0 14bit samples
    
      CS_YUV444P16 = CS_GENERIC_YUV444 | CS_Sample_Bits_16, // YUV 4:4:4 16bit samples
      CS_YUV422P16 = CS_GENERIC_YUV422 | CS_Sample_Bits_16, // YUV 4:2:2 16bit samples
      CS_YUV420P16 = CS_GENERIC_YUV420 | CS_Sample_Bits_16, // YUV 4:2:0 16bit samples
      CS_Y16 = CS_GENERIC_Y | CS_Sample_Bits_16,            // Y   4:0:0 16bit samples
    
      // 32 bit samples (float)
      CS_YUV444PS = CS_GENERIC_YUV444 | CS_Sample_Bits_32,  // YUV 4:4:4 32bit samples
      CS_YUV422PS = CS_GENERIC_YUV422 | CS_Sample_Bits_32,  // YUV 4:2:2 32bit samples
      CS_YUV420PS = CS_GENERIC_YUV420 | CS_Sample_Bits_32,  // YUV 4:2:0 32bit samples
      CS_Y32 = CS_GENERIC_Y | CS_Sample_Bits_32,            // Y   4:0:0 32bit samples
    
      // RGB packed
      CS_BGR48 = CS_RGB_TYPE  | CS_BGR | CS_INTERLEAVED | CS_Sample_Bits_16, // BGR 3x16 bit
      CS_BGR64 = CS_RGBA_TYPE | CS_BGR | CS_INTERLEAVED | CS_Sample_Bits_16, // BGR 4x16 bit
      // no packed 32 bit (float) support for these legacy types
    
      // RGB planar
      CS_RGBP   = CS_GENERIC_RGBP | CS_Sample_Bits_8,  // Planar RGB 8 bit samples
      CS_RGBP8  = CS_GENERIC_RGBP | CS_Sample_Bits_8,  // Planar RGB 8 bit samples
      CS_RGBP10 = CS_GENERIC_RGBP | CS_Sample_Bits_10, // Planar RGB 10bit samples
      CS_RGBP12 = CS_GENERIC_RGBP | CS_Sample_Bits_12, // Planar RGB 12bit samples
      CS_RGBP14 = CS_GENERIC_RGBP | CS_Sample_Bits_14, // Planar RGB 14bit samples
      CS_RGBP16 = CS_GENERIC_RGBP | CS_Sample_Bits_16, // Planar RGB 16bit samples
      CS_RGBPS  = CS_GENERIC_RGBP | CS_Sample_Bits_32, // Planar RGB 32bit samples
    
      // RGBA planar
      CS_RGBAP   = CS_GENERIC_RGBAP | CS_Sample_Bits_8,  // Planar RGBA 8 bit samples
      CS_RGBAP8  = CS_GENERIC_RGBAP | CS_Sample_Bits_8,  // Planar RGBA 8 bit samples
      CS_RGBAP10 = CS_GENERIC_RGBAP | CS_Sample_Bits_10, // Planar RGBA 10bit samples
      CS_RGBAP12 = CS_GENERIC_RGBAP | CS_Sample_Bits_12, // Planar RGBA 12bit samples
      CS_RGBAP14 = CS_GENERIC_RGBAP | CS_Sample_Bits_14, // Planar RGBA 14bit samples
      CS_RGBAP16 = CS_GENERIC_RGBAP | CS_Sample_Bits_16, // Planar RGBA 16bit samples
      CS_RGBAPS  = CS_GENERIC_RGBAP | CS_Sample_Bits_32, // Planar RGBA 32bit samples
    
      // Planar YUVA
      CS_YUVA444    = CS_GENERIC_YUVA444 | CS_Sample_Bits_8,  // YUVA 4:4:4 8bit samples
      CS_YUVA422    = CS_GENERIC_YUVA422 | CS_Sample_Bits_8,  // YUVA 4:2:2 8bit samples
      CS_YUVA420    = CS_GENERIC_YUVA420 | CS_Sample_Bits_8,  // YUVA 4:2:0 8bit samples
    
      CS_YUVA444P10 = CS_GENERIC_YUVA444 | CS_Sample_Bits_10, // YUVA 4:4:4 10bit samples
      CS_YUVA422P10 = CS_GENERIC_YUVA422 | CS_Sample_Bits_10, // YUVA 4:2:2 10bit samples
      CS_YUVA420P10 = CS_GENERIC_YUVA420 | CS_Sample_Bits_10, // YUVA 4:2:0 10bit samples
    
      CS_YUVA444P12 = CS_GENERIC_YUVA444 | CS_Sample_Bits_12, // YUVA 4:4:4 12bit samples
      CS_YUVA422P12 = CS_GENERIC_YUVA422 | CS_Sample_Bits_12, // YUVA 4:2:2 12bit samples
      CS_YUVA420P12 = CS_GENERIC_YUVA420 | CS_Sample_Bits_12, // YUVA 4:2:0 12bit samples
    
      CS_YUVA444P14 = CS_GENERIC_YUVA444 | CS_Sample_Bits_14, // YUVA 4:4:4 14bit samples
      CS_YUVA422P14 = CS_GENERIC_YUVA422 | CS_Sample_Bits_14, // YUVA 4:2:2 14bit samples
      CS_YUVA420P14 = CS_GENERIC_YUVA420 | CS_Sample_Bits_14, // YUVA 4:2:0 14bit samples
    
      CS_YUVA444P16 = CS_GENERIC_YUVA444 | CS_Sample_Bits_16, // YUVA 4:4:4 16bit samples
      CS_YUVA422P16 = CS_GENERIC_YUVA422 | CS_Sample_Bits_16, // YUVA 4:2:2 16bit samples
      CS_YUVA420P16 = CS_GENERIC_YUVA420 | CS_Sample_Bits_16, // YUVA 4:2:0 16bit samples
    
      CS_YUVA444PS  = CS_GENERIC_YUVA444 | CS_Sample_Bits_32,  // YUVA 4:4:4 32bit samples
      CS_YUVA422PS  = CS_GENERIC_YUVA422 | CS_Sample_Bits_32,  // YUVA 4:2:2 32bit samples
      CS_YUVA420PS  = CS_GENERIC_YUVA420 | CS_Sample_Bits_32,  // YUVA 4:2:0 32bit samples
    };


**Image_type and audio channel layout properties and constants:**
::

    int image_type;

    enum AvsImageTypeFlags {
      IT_BFF        = 1 << 0,
      IT_TFF        = 1 << 1,
      IT_FIELDBASED = 1 << 2
      
      // Audio channel mask support
      IT_HAS_CHANNELMASK = 1 << 3,
      // shifted by 4 bits compared to WAVEFORMATEXTENSIBLE dwChannelMask
      // otherwise same as AvsChannelMask
      IT_SPEAKER_FRONT_LEFT = 0x1 << 4,
      IT_SPEAKER_FRONT_RIGHT = 0x2 << 4,
      IT_SPEAKER_FRONT_CENTER = 0x4 << 4,
      IT_SPEAKER_LOW_FREQUENCY = 0x8 << 4,
      IT_SPEAKER_BACK_LEFT = 0x10 << 4,
      IT_SPEAKER_BACK_RIGHT = 0x20 << 4,
      IT_SPEAKER_FRONT_LEFT_OF_CENTER = 0x40 << 4,
      IT_SPEAKER_FRONT_RIGHT_OF_CENTER = 0x80 << 4,
      IT_SPEAKER_BACK_CENTER = 0x100 << 4,
      IT_SPEAKER_SIDE_LEFT = 0x200 << 4,
      IT_SPEAKER_SIDE_RIGHT = 0x400 << 4,
      IT_SPEAKER_TOP_CENTER = 0x800 << 4,
      IT_SPEAKER_TOP_FRONT_LEFT = 0x1000 << 4,
      IT_SPEAKER_TOP_FRONT_CENTER = 0x2000 << 4,
      IT_SPEAKER_TOP_FRONT_RIGHT = 0x4000 << 4,
      IT_SPEAKER_TOP_BACK_LEFT = 0x8000 << 4,
      IT_SPEAKER_TOP_BACK_CENTER = 0x10000 << 4,
      IT_SPEAKER_TOP_BACK_RIGHT = 0x20000 << 4,
      // End of officially defined speaker bits
      // The next one is special, since cannot shift SPEAKER_ALL 0x80000000 further.
      // Set mask and get mask handles it.
      IT_SPEAKER_ALL = 0x40000 << 4,
      // Mask for the defined 18 bits + SPEAKER_ALL
      IT_SPEAKER_BITS_MASK = (AvsChannelMask::MASK_SPEAKER_DEFINED << 4) | IT_SPEAKER_ALL,
      IT_NEXT_AVAILABLE = 1 << 23
    };


**Chroma placement constants (bits 20 -> 23):**
::

    enum AvsChromaPlacement {
      CS_UNKNOWN_CHROMA_PLACEMENT = 0 << 20,
      CS_MPEG1_CHROMA_PLACEMENT   = 1 << 20,
      CS_MPEG2_CHROMA_PLACEMENT   = 2 << 20,
      CS_YUY2_CHROMA_PLACEMENT    = 3 << 20,
      CS_TOPLEFT_CHROMA_PLACEMENT = 4 << 20
    };


Functions [need to add examples]
--------------------------------

HasVideo
~~~~~~~~

::

    bool HasVideo();


This will return true if there is any video in the given clip.


IsRGB / IsRGB24 / IsRGB32
~~~~~~~~~~~~~~~~~~~~~~~~~

::

    bool IsRGB();
    bool IsRGB24();
    bool IsRGB32();


This will return true if the colorspace is `RGB`_ (in any way). The first two
return true if the clip has the specific RGB colorspace (:doc:`RGB24 <ColorSpaces>` and
:doc:`RGB32 <ColorSpaces>`). The third returns true for any RGB colorspace; future formats
could also apply.


IsRGB48 / IsRGB64 (AviSynth+)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    bool IsRGB48() const;
    bool IsRGB64() const;

These functions are for 16 bit packed RGB formats, similar to
IsRGB24 and IsRGB32 for 8 bits.


IsYUV / IsYUY2 / IsYV24 / IsYV16 / IsYV12 / IsYV411 / IsY8
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    bool IsYUV() const;

This will return true if the colorspace is :doc:`YUV <ColorSpaces>` (in any way).
Note: Y8-Y32 returns true as well. (Y formats are set both as YUV and Interleaved)

::

    bool IsYUY2() const;
    bool IsYV24() const;  // v5
    bool IsYV16() const;  // v5
    bool IsYV12() const;
    bool IsYV411() const; // v5
    bool IsY8() const;    // v5

They will return true if the clip has the specific YUV colorspace (e.g. :doc:`YUY2 <ColorSpaces>` and
:doc:`YV12 <ColorSpaces>`). Note that I420 is also reported as YV12, because planes are
automatically swapped.


Is444 / Is422 / Is420 / IsY (AviSynth+)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    bool Is444() const;
    bool Is422() const;
    bool Is420() const;
    bool IsY() const;

These functions are the bit depth independent versions of
IsYV24, IsYV16, IsYV12 and IsY8.


Is440 / Is410 / Is411 / IsYA (AviSynth+)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Added in v13:

::

    bool Is440() const;
    bool Is410() const;
    bool Is411() const;
    bool IsYA() const;

These functions will return true if the clip has the specific YUV colorspace.
Is440, Is410, and Is411 also return true for alpha-bearing variants.  IsYA
is the alpha-bearing version.


IsYUVA / IsPlanarRGB / IsPlanarRGBA (AviSynth+)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

  bool    (VideoInfo::*IsYUVA)() const;
  bool    (VideoInfo::*IsPlanarRGB)() const;
  bool    (VideoInfo::*IsPlanarRGBA)() const;


For checking further Avisynth+ specific formats:
YUV with alpha plane, planar RGB with and without an alpha plane.


IsColorSpace / IsSameColorspace / IsPlanar
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    bool IsColorSpace(int c_space);


This function will check if the colorspace (VideoInfo.pixel_type) is the same
as given c_space (or more general it checks for a :doc:`Colorspace property <ColorspaceProperties>` (see
avisynth.h)).


::

    bool IsSameColorspace(const VideoInfo& vi2);


This function will compare two VideoInfos, and check if the colorspace is the
same. Note: It does not check imagesize or similar properties.


::

    bool IsPlanar();


This will return true if the video is planar: Y, YUV(A) and planar RGBP(A).
See the :doc:`Planar <PlanarImageFormat>` image format.


Is / IsFieldBased / IsParityKnown / IsBFF / IsTFF
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    bool Is(int property);


From v6 this will return true if the image type (VideoInfo.image_type)
is the same as the given property (being IT_BFF, IT_TFF or
IT_FIELDBASED).


::

    bool IsFieldBased();


This will return true if the video has been through a :doc:`SeparateFields <../corefilters/separatefields>`, and
the video has not been :doc:`weaved <../corefilters/weave>` yet. Otherwise it will return false.

::

    bool IsParityKnown();


This will return true if the video parity is known.

::

    bool IsBFF();
    bool IsTFF();


This will return true if the video is bottom-field-first or top-field-first
respectively.


SetFieldBased / Set / Clear
~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    void SetFieldBased(bool isfieldbased);


This will set the field-based property to true (respectively false) if
isfieldbased=true (respectively false).

::

    void Set(int property);
    void Clear(int property);


This sets respectively clears an image_type property like: IT_BFF, IT_TFF or
IT_FIELDBASED. See field.h for examples.


BitsPerPixel
~~~~~~~~~~~~

::

    int BitsPerPixel();


This will return the number of bits per pixel. This can be:

+------------------+------------+
| pixel_type       | nr of bits |
+==================+============+
| CS_BGR24         | 24         |
+------------------+------------+
| CS_BGR32         | 32         |
+------------------+------------+
| CS_YUY2          | 16         |
+------------------+------------+
| CS_YV12, CS_I420 | 12         |
+------------------+------------+

Note: This is not the BitsPerComponent which is the bit depth of the format.
The calculation returns an average storage size of a pixel.


SetFPS / MulDivFPS
~~~~~~~~~~~~~~~~~~

::

    void SetFPS(unsigned numerator, unsigned denominator);


This will set the framerate.

::

    void MulDivFPS(unsigned multiplier, unsigned divisor);


This will multiply the denominator by *multiplier* and scale the numerator
and modified denominator.

There is some other useful information in VideoInfo structure (width, height,
fps_numerator, fps_denominator, num_frames, pixel_type and image_type). See
'avisynth.h' header file.


BytesFromPixels
~~~~~~~~~~~~~~~

::

    int BytesFromPixels(int pixels) const;


For interleaved formats it will return the number of bytes from the
specified number of pixels. For planar formats it will do the same
except it operates on the first plane.


RowSize / BMPSize
~~~~~~~~~~~~~~~~~

::

    int RowSize(int plane=0) const;


For interleaved formats it will return the width of the frame in bytes.
For planar formats it will return the width of the specified plane in
bytes.

examples:

| 640x480 RGB24: RowSize() = 3*640 = 1920

| 640x480 YV12: RowSize(PLANAR_Y) = 640
| 640x480 YV12: RowSize(PLANAR_U) = 320

::

    int BMPSize() const;

For interleaved formats it will return the size of the frame in bytes
where the width is rounded up to a multiple of 4 bytes. For planar
formats it will do the same for the luma plane then add the two chroma
planes scaled by the subsampling. So, it's the number of bytes of a
frame as if it was a `BMP frame`_.

examples:

| 640x480 RGB24: BMPSize() = 480 * 3*640 = 921600
| 643x480 RGB24: BMPSize() = 480 * 3*644 = 927360

| 640x480 YV12: BMPSize() = 480 * 640 * (1+2*0.25) = 460800
| 640x480 YV16: BMPSize() = 480 * 640 * (1+2*0.5) = 614400
| 640x480 YV24: BMPSize() = 480 * 640 * (1+2*1) = 921600
| 643x480 YV24: BMPSize() = 480 * 644 * (1+2*1) = 927360


GetPlaneWidthSubsampling / GetPlaneHeightSubsampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    int GetPlaneWidthSubsampling(int plane) const; // v5

This will return the subsampling of the width in bitshifts.

examples:

| YV24: GetPlaneWidthSubsampling(PLANAR_U) = 0 // since there is no
  horizontal subsampling on a chroma plane
| YV16: GetPlaneWidthSubsampling(PLANAR_U) = 0 // since there is no
  horizontal subsampling on a chroma plane

::

    int GetPlaneHeightSubsampling(int plane) const; // v5

This will return the subsampling of the height in bitshifts.

examples:

| YV24: GetPlaneHeightSubsampling(PLANAR_U) = 0 // since there is no
  vertical subsampling on a chroma plane
| YV16: GetPlaneHeightSubsampling(PLANAR_U) = 1 // since vertically there
  are two times less samples on a chroma plane compared to a plane which
  is not subsampled

+----------------------+------------------------------------+-------------------------------------+
| color format         | GetPlaneWidthSubsampling(PLANAR_U) | GetPlaneHeightSubsampling(PLANAR_U) |
+======================+====================================+=====================================+
| YV24, YUV(A)444      | 0                                  | 0                                   |
+----------------------+------------------------------------+-------------------------------------+
| YV16/YUY2, YUV(A)422 | 1                                  | 0                                   |
+----------------------+------------------------------------+-------------------------------------+
| YV12, YUV(A)420      | 1                                  | 1                                   |
+----------------------+------------------------------------+-------------------------------------+
| YV411                | 2                                  | 1                                   |
+----------------------+------------------------------------+-------------------------------------+
| Y8-Y32               | Error thrown                       | Error thrown                        |
+----------------------+------------------------------------+-------------------------------------+


HasAudio
~~~~~~~~

::

    bool HasAudio();


This will return true if there is any audio in the given clip.


AudioChannels / SampleType
~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    int AudioChannels();


This will return the number of audio channels.


::

    int SampleType();


This will return the sampletype. This can be:

+--------------+------------+
| pixel_type   | nr of bits |
+==============+============+
| SAMPLE_INT8  | 1<<0       |
+--------------+------------+
| SAMPLE_INT16 | 1<<1       |
+--------------+------------+
| SAMPLE_INT24 | 1<<2       |
+--------------+------------+
| SAMPLE_INT32 | 1<<3       |
+--------------+------------+
| SAMPLE_FLOAT | 1<<4       |
+--------------+------------+


IsSampleType
~~~~~~~~~~~~

::

    bool IsSampleType(int testtype);


This function will check if the sampletype (VideoInfo.sample_type) is the
same as testtype.


SamplesPerSecond / BytesPerAudioSample / BytesPerChannelSample
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    int SamplesPerSecond();


This will return the number of bytes per second.

::

    int BytesPerAudioSample();


This will return the number of bytes per sample:

::

    int BytesPerChannelSample()


This will return the number of bytes per channel-sample. This can be:

+--------------+----------------------+
| sample       | nr of bytes          |
+==============+======================+
| SAMPLE_INT8  | sizeof(signed char)  |
+--------------+----------------------+
| SAMPLE_INT16 | sizeof(signed short) |
+--------------+----------------------+
| SAMPLE_INT24 | 3                    |
+--------------+----------------------+
| SAMPLE_INT32 | sizeof(signed int)   |
+--------------+----------------------+
| SAMPLE_FLOAT | sizeof(SFLOAT)       |
+--------------+----------------------+


AudioSamplesFromFrames / FramesFromAudioSamples / AudioSamplesFromBytes / BytesFromAudioSamples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    int64_t AudioSamplesFromFrames(int64_t frames);


This returns the number of audiosamples from the first *frames* frames.

::

    int FramesFromAudioSamples(int64_t samples);


This returns the number of frames from the first *samples* audiosamples.

::

    int64_t AudioSamplesFromBytes(int64_t bytes);


This returns the number of audiosamples from the first *bytes* bytes.

::

    int64_t BytesFromAudioSamples(int64_t samples);


This returns the number of bytes from the first *samples* audiosamples.

There is some other useful information in VideoInfo structure
(audio_samples_per_second, sample_type, num_audio_samples and nchannels). See
'avisynth.h' header file.

.. _videoinfo_getchannelmask:
.. _videoinfo_setchannelmask:
.. _videoinfo_ischannelmaskknown:

IsChannelMaskKnown / SetChannelMask / GetChannelMask (Avisynth+ 3.7.3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    bool IsChannelMaskKnown();

This returns if channel layout info is set or not.


::

    void SetChannelMask(bool isChannelMaskKnown, unsigned int dwChannelMask);

This sets the channel layout validity and the layout mask.

Validity bit occupies the 3rd bit of image_type (starting from 0).
Actual mask is stored on 4-21 bits, "speaker all" case is on bit 22.

Note that this mask is stuffed into 'image_type', which had already some
bits used in it.

The given value thus will be stored differently: shifted by 4 bits, while special 
0x80000000 "speaker all" case is handled on a separate bit. Valid dwChannelMask values are 
using 0-17 lsb bits, or is set to 0x80000000 to mask "speaker all".

Input bit mask is following the specification of WAVE_FORMAT_EXTENSIBLE speaker bits.
See https://learn.microsoft.com/en-us/windows/win32/api/mmreg/ns-mmreg-waveformatextensible)


::

    unsigned int GetChannelMask();

Returns the channel mask (same as it was set by SetChannelMask) if mask is valid, 0 otherwise.

**Channel layout mask constants**

avisynth.h and avisynth_c.h contain the mask bit descriptions. In the actual layout they are or'd together.
The AvsChannelMask enum values are the channel mask constants similar to  
WAVE_FORMAT_EXTENSIBLE speaker bits. 18 bits starting from MASK_SPEAKER_FRONT_LEFT to MASK_SPEAKER_TOP_BACK_RIGHT.

+------------------------------------+----------------------+
| MASK_SPEAKER_FRONT_LEFT            | 0x1                  |
+------------------------------------+----------------------+
| MASK_SPEAKER_FRONT_RIGHT           | 0x2                  |
+------------------------------------+----------------------+
| MASK_SPEAKER_FRONT_CENTER          | 0x4                  |
+------------------------------------+----------------------+
| MASK_SPEAKER_LOW_FREQUENCY         | 0x8                  |
+------------------------------------+----------------------+
| MASK_SPEAKER_BACK_LEFT             | 0x10                 |
+------------------------------------+----------------------+
| MASK_SPEAKER_BACK_RIGHT            | 0x20                 |
+------------------------------------+----------------------+
| MASK_SPEAKER_FRONT_LEFT_OF_CENTER  | 0x40                 |
+------------------------------------+----------------------+
| MASK_SPEAKER_FRONT_RIGHT_OF_CENTER | 0x80                 |
+------------------------------------+----------------------+
| MASK_SPEAKER_BACK_CENTER           | 0x100                |
+------------------------------------+----------------------+
| MASK_SPEAKER_SIDE_LEFT             | 0x200                |
+------------------------------------+----------------------+
| MASK_SPEAKER_SIDE_RIGHT            | 0x400                |
+------------------------------------+----------------------+
| MASK_SPEAKER_TOP_CENTER            | 0x800                |
+------------------------------------+----------------------+
| MASK_SPEAKER_TOP_FRONT_LEFT        | 0x1000               |
+------------------------------------+----------------------+
| MASK_SPEAKER_TOP_FRONT_CENTER      | 0x2000               |
+------------------------------------+----------------------+
| MASK_SPEAKER_TOP_FRONT_RIGHT       | 0x4000               |
+------------------------------------+----------------------+
| MASK_SPEAKER_TOP_BACK_LEFT         | 0x8000               |
+------------------------------------+----------------------+
| MASK_SPEAKER_TOP_BACK_CENTER       | 0x10000              |
+------------------------------------+----------------------+
| MASK_SPEAKER_TOP_BACK_RIGHT        | 0x20000              |
+------------------------------------+----------------------+
| MASK_SPEAKER_DEFINED               | 0x0003FFFF           |
+------------------------------------+----------------------+
| MASK_SPEAKER_RESERVED              | 0x7FFC0000           |
+------------------------------------+----------------------+
| MASK_SPEAKER_ALL                   | 0x80000000           |
+------------------------------------+----------------------+

In the actual storage, in VideoInfo::image_type, they are shifted by 4 bits, except MASK_SPEAKER_ALL,
which is handled specially, due to lack of available bits.
See VideoInfo::AvsImageTypeFlags enum in Avisynth.h .


NumComponents (AviSynth+)
~~~~~~~~~~~~~~~~~~~~~~~~~

::

    int NumComponents() const;

This will return the number of logical pixel components.
For Y and planar formats this is the actual number of valid planes.

+--------------------+------------------+
| color format       | nr of components |
+====================+==================+
| Y                  | 1                |
+--------------------+------------------+
| YUY2               | 3                |
+--------------------+------------------+
| planar YUV, RGBP   | 3                |
+--------------------+------------------+
| planar YUVA, RGBPA | 4                |
+--------------------+------------------+
| RGB24, RGB48       | 3                |
+--------------------+------------------+
| RGB32, RGB64       | 4                |
+--------------------+------------------+


ComponentSize (AviSynth+)
~~~~~~~~~~~~~~~~~~~~~~~~~

::

    int ComponentSize() const;

This will return the number bytes occupied for a component.
Basically this depends on the bit depth.

+----------------------+---------------+
| bit depth            | ComponentSize |
+======================+===============+
| 8                    | 1             |
+----------------------+---------------+
| 10-16                | 2             |
+----------------------+---------------+
| 32 bit float         | 4             |
+----------------------+---------------+


BitsPerComponent (AviSynth+)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    int BitsPerComponent() const;

This will return the bit depth: 8-16 or 32.

Note: Avisynth+ does not support all bit depths: 
only 8, 10, 12, 14, 16 and 32.



----

Back to :doc:`FilterSDK`

$Date: 2025/08/25 22:52:15 $

.. _RGB: http://avisynth.nl/index.php/RGB
.. _BMP frame: http://en.wikipedia.org/wiki/BMP_file_format
