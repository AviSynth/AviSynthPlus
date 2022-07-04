Avisynth Plus change log
------------------------
Source: https://github.com/AviSynth/AviSynthPlus

This file contains all change log, with detailed examples and explanations.
The "rst" version of the documentation just lists changes in brief.

20220627 3.7.3 WIP
------------------
- Fix: C interface crash when using avs_new_video_frame_p(_a)
- Fix (#283): broken runtime functions min/max/minmaxdifference when threshold is not 0 (returned -1). Regression in 3.7.2
- New: add a sixth array element to PlaneMinMaxStats: average. Defines variable "PlaneStats_average" as well if setting variables is required.
- Fix (#282): ConvertToRGB
  - do check for exact 8 or 16 bit input, because packed RGB formats exist only for 8 and 16 bits
  - keep alpha for RGBA planar - convert RGBAP8/16 to RGB32/64, while RGBP8/16 is still RGB24/48

20220317 3.7.2
--------------
- C interface Win32 access: fix issue by adding V8 interface function names to avisynth.def
  or else names are decorated (Issue #276) 
  e.g. DLL published _avs_get_frame_props_ro@8 instead of avs_get_frame_props_ro
- ShowRed/Green/Blue/Alpha/Y/U/V: addition to earlier fixes:
  When clips are planar and both source and destination format have alpha plane, 
  then it will be copied instead of filled with 255d.
  Additional checking is done for alpha plane size when ShowU/V, because when
  source is subsampled the original alpha plane cannot be copied (larger).
- ConvertBits: 
  Does not get frame 0 in constructor for frame properties if 'fulls' is directly specified. (magiblot)
  May make script initialization much quicker (Issue #275)
  https://github.com/AviSynth/AviSynthPlus/issues/275
- Trim, AudioTrim: bool 'cache' (default true) parameter. 
  Workaround for Issue #274, lower memory consumption but may be slower. 
  Benefits heavily depend on how trimmed clips are used later.
- Expr: scale_inputs to case insensitive and add floatUV to error message as an allowed value.
- propCopy: able to specify that the property list is negative.
    bool "exclude" = false # default: "props" is positive list
    
    propCopy(org,true,props=["_Matrix", "_ColorRange"], exclude=false) # merge only two properties
    propCopy(org,true,props=["_Matrix", "_ColorRange"], exclude=true) # merge all, except listed ones
    propCopy(org,props=["_Matrix", "_ColorRange"]) # erase all then copy only selected
    propCopy(org,props=["_Matrix", "_ColorRange"], exclude = true) # erase all, then copy all, except listed ones

- Version()
  New optional parameters 

    int length, int width, int height, string pixel_type, clip c

  Version clip defaults: 
    length=240, width = -1, height = -1 (-1: automatically sized to fit for font size 24)
    pixel_type = "RGB24"

  When 'clip' (a format template) is specified then pixel_type, length, 
  fps data, width and height are defined from it.
  If any additional 'length', 'width', 'height', 'pixel_type' parameter is given, it overrides defaults.
  When width and height is given and is <= 0 then it is treated as 'automatic'

  Covers feature request Issue #261

- BlankClip: allow 'colors' with array size more than the number of actual planes.
  If an array is larger, further values are simply ignored.
- BlankClip, AddBorders, LetterBox: no A=0 check for non-YUVA
- Fade filter family new parameters
    int 'color_yuv' 
    array of float 'colors'
  similar to BlankClip
- MergeRGB, MergeARGB
  - add MergeARGB parameter "pixel_type", similar to MergeRGB
  - accept pixel_type other than packed RGB formats, plus a special one is "rgb"
  - output format is planar rgb(a) (MergeRGB/MergeARGB) when
    - pixel_type = "rgb" or
    - pixel_type is empty and 
      - either input is planar RGB
      - either input is different from 8 or 16 bits (no packed RGB formats there)
    - pixel_type is explicitely set to a valid planar rgb constant e.g. "RGBP10"
  - Accept planar RGB clip in place of input clips and the appropriate color plane is copied from them
  - Fill alpha channel with zero when MergeRGB output pixel_type format is specified to have an alpha plane
  - frame property source is the R clip; _Matrix and _ChromaLocation are removed if R is not an RGB clip
- PropDelete: accept a non-empty array string as list of property names to remove
  Parameter is not optional, and has no name. It can be either a string (as before) or an array of strings
    propDelete("_Matrix") # old syntax, still accepted
    propDelete(["_Matrix", "_ColorRange"])
- PropCopy: new string parameter "props" as list of property names to remove
    "props": a non-empty array of strings
    
    old syntax, still accepted:
      propCopy(org,true) # merge from all org's properties
      propCopy(org,false) # erase all then copy all org's properties (exact copy)
    new syntax
      propCopy(org,true,props=["_Matrix", "_ColorRange"]) # merge
      propCopy(org,props=["_Matrix", "_ColorRange"]) # erase all then copy only selected
- Histogram Levels: stop using shades of grey on top of bars.
- Histogram Levels: use bar color 255 for RGB instead of Y's 235. (and scaled eqivivalents)
- Fix: Histogram "Levels": prevent crash when factor=0.0
- Fix: Histogram "Levels": fix regression incorrect "factor" applied for U/V part drawing when format was subsampled (non-444)
  Regression since 20160916 r2666 (commit 986e2756)
- Histogram "Audiolevels" and StereoOverlay to deny planar RGB
- Histogram "Luma": support 10-16 and 32 bits
- Histogram: give parameter name "factor" and type 'float' for Histogram's unnamed optional parameter used in "Level" mode.
  Other modes just ignore this parameter if given.
- Fix: Histogram "color" may crash on certain dimensions for subsampled formats. 
  Regression since 20180301 r2632.
- Fix: Histogram "color" and "color2" mode check and give error on Planar RGB
- Fix: missing Histogram "color2" CCIR rectangle top and bottom line (black on black) 
  Regression since 3.6.2-test1 (commit 1fc82f03)
- Fix: Compare to support 10-14 bits 
  was: factor was always using 65535 (2^16-1) instead of (2^bit depth - 1)
  was: 16 bit luma/rgb color values were used for drawing graph
- Fix: Compare
  'channels' parameter default to "Y" when input is greyscale;
  instead of "YUV" which was giving error because of U and V does not exist for this format.
- ShowRed/Green/Blue/Alpha/Y/U/V
  - support YUY2 input
  - support YV411 output
  - (not changed: ShowU/ShowV may give error for 420, 422 or 411 format outputs when clip dimensions are
    not eligible for a given output subsampling (check for appropriate mod2 or mod4 width or height)
  - Copy alpha from source when target is alpha-capable
  - Fill alpha with maximum pixel value when target is alpha-capable but source ha no alpha component
  - Delete _Matrix and _ChromaLocation frame properties when needed.
  - More consistent behaviour for YUV and planar RGB sources.
    
    Default pixel_type is adaptive. If none or empty ("") is given for pixel_type then target format is
    - YUV444 when source is Y, YUV or YUVA
    - RGB32/64 (packed RGB) when source is RGB24/32/48/64
    - RGBP (planar RGB) when source is RGBP or RGBAP

    When 'rgb' is given for pixel_type then then target format is

    - RGB32/64 (packed) when source is RGB24/32/48/64 - old, compatible way
    - RGB planar when source is planar RGB(A) or YUV(A) or Y - changed from rgb32/64 because all bit depth must be supported

    When 'yuv' is given (new option!) for pixel_type then then target format is

    - YUV444 for all sources

    Also there is a new option when pixel_type is still not exact, and is given w/o bit depth.
    pixel_type which describes the format without bit depth is automatically extended to a valid video string constant:

      y, yuv420, yuv422, yuv444, yuva420, yuva422, yuva444, rgbp, rgbap

    Examples:

      32 bit video and pixel_type 'y' will result in "Y32"
      16 bit video and pixel_type 'yuv444' will result in "YUV444P16"
      8 bit video and pixel_type 'rgbap' will result in "RGBAP8"

- Fix #263. Escaping double-quotes results in error
- Allow top_left (2) and bottom_left (4) chroma placements for 422 in colorspace conversions, they act as "left" (0, "mpeg2")
  in order not to give error with video sources which have _ChromaLocation set to other than "mpeg2"
  See https://trac.ffmpeg.org/ticket/9598#comment:5
- Fix: Expr LUT operation Access Violation on x86 + AVX2 due to an unaligned internal buffer (<32 bytes)
- Fix: Chroma full scale as ITU Rec H.273 (e.g +/-127.5 and not +/-127) in internal converters, ColorYUV and Histogram
- Fix #257: regression in 3.7.1:  GreyScale to not convert to limited range when input is RGB. Regression in 3.7.1
  Accepts only matrix names of limited range as it is put in the documentation.
- Fix #256: ColorYUV(analyse=true) to not set _ColorRange property to "full" if input has no such 
  property and range cannot be 100% sure established. In general: when no _ColorRange for input and 
  no parameter which would rely on a supposed default (such as full range for gamma), then an
  output frame property is not added.
  When no _ColorRange for input and no other parameters to hint color range then 
  - gamma<>0 sets full range
  - opt="coring" sets limited range
  - otherwise no _ColorRange for output would be set
- Overlay (#255): "blend": using accurate formula using float calculation. 8 bit basic case is slower now when opacity=1.0.
  Higher bit depths and opacity<1.0 cases are quicker.
  Mask processing suffered from inaccuracy. For speed reasons mask value 0 to 255 were handled
  as mask/256 instead of mask/255. Since with such calculation maximum value was not the expected 1.0 but rather 255/256 (0.996)
  this case was specially treated as 1.0 to give Overlay proper results at least the the two extremes.
  But for example applying mask=129 to pixel=255 resulted in result_pixel=128 instead of 129. This was valid on higher bit depths as well.
  Note 3.7.2 Test2 has a regression of broken maskless mode for 0<opacity<1 which was fixed in 3.7.2 test 3
- Fix: Attempt to resolve deadlock when an Eval'd (Prefetch inside) Clip result is 
  used in Invoke which calls a filter with GetFrame in its constructor.
  (AvsPMod use case which Invokes frame prop read / ConvertToRGB32 after having the AVS script evaluated)
  Remark: problem emerged in 3.7.1test22 which is trying to read frame properties of the 0th frame in its constructor.
  A similar deadlock situation was already fixed earlier in Neo branch and had been backported but it did not cover this use case.
  Note: Prefetch(1) case was fixed in 3.7.2 Test3

20220122 3.7.1a
---------------
(no binaries)
- Update some rst docs
- Linux: avs_core/CMakeLists: provide version.h and arch.h with installed headers
- Fix: Histogram AudioLevels half character upshift (regression since v3.6)
- Bump Copyright year to 2022

20211231 3.7.1
--------------
- Bump AVISYNTH_INTERFACE_VERSION to 9, AVISYNTHPLUS_INTERFACE_BUGFIX_VERSION to 0
- Fix: Debug build does not crash with stack overflow when some 1000+ clips are in filter chain.
- Fix memory and speed issues when Prefetch was not the last position or there are multiple Prefetch.
  Thanks to magiblot for reporting the issue and pointing out the possible resolution https://github.com/AviSynth/AviSynthPlus/issues/244
  Phenomenon: 
  - MT_MULTI_INSTANCE filters were kept multithreaded and instantiated after the Prefetch line.
    When another Prefetch appeared later, its intended value was not set properly.
  - MT_MULTI_INSTANCE filters were instantiated in a larger count that the actual Prefetch value.
    E.g. Prefetch 3 created 4 filter instances, values 4 to 7 were resulting in 8 filter instances
    and Prefetch 8 to 15 created 16 filter instances, etc.
    When the filter was memory-allocation heavy (like DepanEstimate which allocates large fftw3 buffers in its constructor)
    it caused excession memory consumption because unnecessary filter instances were created.
    For example Prefetch(8) created 16 instances of which 8 was not even used.
  
  Fixes:
  - limit and set proper thread count for 2nd, 3rd.. or no further Prefetch
  - MT_MULTI_INSTANCE filter instances limited to N (Prefetch value) and not the next power of two N

- CombinePlanes: a bit optimized MergeLuma-like cases. (Dogway's finding)
  Theory behind: when a frame has exactly one 'user' (no other frames are yet referencing it) then it can 
  directly be grabbed and made writable without any frame plane content copying.
  When this "I'm the only one" condition is fulfilled and the below-written conditions are set then it can be a bit quicker
  then the ordinary "make a new frame and copy the referenced input frames into that" logic works.
  1.) Source clip has the same format as the target, and the first plane ID is the same.
      Y comes from first clip (no Y plane copy, the input frame containing Y is reused), U and V are copied
  2.) Second clip has the same format as the target, and the 2nd and 3rd plane ID is the same
      U and V comes from 2nd clip (no UV copy, frame containing U and V is reused), 
      while Y (or the given first plane ID) is copied from first clip. When there is a 

Example:

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

- Linux: Show more information when dlopen fails
- Fix: "Text" filter would crash when y coord is odd and format has vertical subsampling
- Working on traditional rst documentation. Filter SDK, Changelog, etc.

- New array modifier function: ArraySet

  For memo here is the list of avaliable array manipulator functions
  - ArrayAdd - append
  - ArrayDel - delete at position
  - ArrayIns - insert before position
  - ArraySet - replace at position

  ArrayIns
  ^^^^^^^^

  ArrayIns(array_to_mod, value_to_insert, index1 [, index2, index3...])

      Insert a value into an array or into its subarray.
      Returns a new array with value_to_insert inserted into array_to_mod (1D array) or array_to_mod[index1 (, index2, index3...)] (multi-dimensional array)
      The indexes point to the insertion point. Index 0 will insert at the beginning of the array.
      Index (ArraySize) will insert after the last element (same as ArrayAdd - append)
      Original array (as with the other functions) remains untouched.

  ArrayAdd
  ^^^^^^^^

  ArrayAdd(array_to_mod, value_to_append [, index1, index2, index3...])

      Appends value to the end of an array or its subarray
      Returns a new array with value_to_append appended to array_to_mod (1D array) or array_to_mod[index1 (, index2, index3...)] (multi-dimensional array).
      Original array (as with the other functions) remains untouched.

  ArrayDel
  ^^^^^^^^

  ArrayDel(array_to_mod, index1 (, index2, index3...])

      Returns a new array in which the requested position was deleted.
      Original array (as with the other functions) remains untouched.

  ArraySet
  ^^^^^^^^

  ArraySet(array_to_mod, replacement_value, index1 [, index2, index3...])

      Returns a new array with array_to_mod[index1 (, index2, index3...)] = replacement_value
      Original array (as with the other functions) remains untouched.

- Array modifier functions to allow multidimensional subarray indexes

Example:

    ColorbarsHD()
    # array indexes are zero based
    a = []
    a=ArrayAdd(a,[1,2]) # [[1,2]]
    a=ArrayIns(a,3,0) # [3,[1,2]]
    a=ArrayAdd(a,"s1") # [3,[1,2],"s1"]
    a=ArrayAdd(a,"s2") # [3,[1,2],"s1","s2"]
    a=ArrayDel(a,2) # [3,[1,2],"s2"]
    a=ArraySet(a,"g",1,0) # [3,["g",2],"s2"]
    a=ArrayAdd(a,"h",1) # [3,["g",2,"h"],"s2"]
    a=ArrayAdd(a,[10,11,12],1) # append to (1) -> [3,["g",2,"h",[10,11,12]],"s2"]
    a=ArrayDel(a,1,3,0) # del from (1,3,0) -> [3,["g",2,"h",[11,12]],"s2"]
    a=ArrayAdd(a,"added") # [3,["g",2,"h",[11,12]],"s2","added"]
    a=ArrayAdd(a,["yet","another","sub"]) # [3,["g",2,"h",[11,12]],"s2","added",["yet","another","sub"]]
    x=a[0] #3
    x=a[1,0] #g
    x=a[1,2] #h
    x=a[1,3,1] #12
    x=a[3] #"added"
    x=a[4,1] #"another"
    SubTitle("x = " + String(x) + " Size=" + String(a.ArraySize()))

- Expr: allow auto scaling effect on pixels obtained from relative addressing
- ConvertBits: ordered dither: possible to dither down with more than 8 bits difference like in
    Clip16.ConvertBits(8, dither=0, dither_bits=4)
  Such conversion is made in two phases. First the clip is converted to (dither_bits+8) bits; in the above example it is 12.
  If the temporary bit depth would be odd (no 9 or 11 bit support in Avisynth+) then it is made even.
  bit depth that differs in only 8 bits for the target. Then this intermediate clip is converted to the required end target.
- Quicker ClearProperties and CopyProperties filters (by using MakePropertyWritable instead of MakeWritable).
- Fix: MinMax runtime filter family: check plane existance (e.g. error when requesting RPlaneMinMaxDifference on YV12)
- Fix: prevent x64 debug AviSynth builds from crashing in VirtualDub2 (opened through CAVIStreamSynth)
- ExtractY/U/V/R/G/B/A, PlaneToY: delete _ChromaLocation property. Set _ColorRange property to "full" if source is Alpha plane
- AviSynth interface additions: extend queryable internal environment properties.
  Since Interface version 8 IScriptEnvironment::GetEnvProperty (Avisynth.h) and avs_get_env_property (avisynth_c.h)
  interface functions can query some specific internal properties of AviSynth core. Thread count, etc..
  These are mainly for internal use but some can be useful for plugins and external applications.
  Each requested property has an identification number, they are found in avisynth.h and avisynth_c.h

  This addition brought new properties to query: host system's endianness, interface version and bugfix subversion.
  Relevant enum names start with AEP_ (cpp) or AVS_AEP_ (c) (AEP stands for Avisynth Environment Property)
  
  Details:
  
  AEP_HOST_SYSTEM_ENDIANNESS (c++) AVS_AEP_HOST_SYSTEM_ENDIANNESS (c)
    Populated by 'little', 'big', or 'middle' based on what GCC and/or Clang report at compile time.

  AEP_INTERFACE_VERSION (c++) AVS_AEP_INTERFACE_VERSION (c)
    for requesting actual interface (main) version. An long awaited function. 
    So far the actual interface version could be queried only indirectly, with trial and error, by starting from e.g. 10 then
    going back one by one until CheckVersion() did not report an exception/error code. 

    Even for V8 interface this was a bit tricky, the only way to detect was the infamous
      has_at_least_v8 = true;
      try { env->CheckVersion(8); } catch (const AvisynthError&) { has_at_least_v8 = false; }
    method.
    
    Now (starting from interface version 8.1) a direct version query is supported as well.
    Of course this (one or two direct call only) is the future.
    Programs or plugins which would like to identify older systems still must rely partially on the CheckVersion method.

    CPP interface (through avisynth.h).

      IScriptEnvironment *env = ...
      int avisynth_if_ver = 6;
      int avisynth_bugfix_ver = 0;
      try { 
        avisynth_if_ver = env->GetEnvProperty(AEP_INTERFACE_VERSION); 
        avisynth_bugfix_ver = env->GetEnvProperty(AEP_INTERFACE_BUGFIX);      
      } 
      catch (const AvisynthError&) { 
        try { env->CheckVersion(8); avisynth_if_ver = 8; } catch (const AvisynthError&) { }
      }
      has_at_least_v8 = avisynth_if_ver >= 8; // frame properties, NewVideoFrameP, other V8 environment functions
      has_at_least_v8_1 = avisynth_if_ver > 8 || (avisynth_if_ver == 8 && avisynth_bugfix_ver >= 1);
      // 8.1: C interface frameprop access fixed, IsPropertyWritable/MakePropertyWritable support, extended GetEnvProperty queries
      has_at_least_v9 = avisynth_if_ver >= 9; // future

    C interface (through avisynth_c.h)

      AVS_ScriptEnvironment *env = ...
      int avisynth_if_ver = 6; // guessed minimum
      int avisynth_bugfix_ver = 0;
      int retval = avs_check_version(env, 8);
      if (retval == 0) {
        avisynth_if_ver = 8;
        // V8 at least, we have avs_get_env_property but AVS_AEP_INTERFACE_VERSION query may not be supported
        int retval = avs_get_env_property(env, AVS_AEP_INTERFACE_VERSION);
        if(env->error == 0) {
          avisynth_if_ver = retval;
          retval = avs_get_env_property(env, AVS_AEP_INTERFACE_BUGFIX);
          if(env->error == 0)
            avisynth_bugfix_ver = retval;
        }
      }
      has_at_least_v8 = avisynth_if_ver >= 8; // frame properties, NewVideoFrameP, other V8 environment functions
      has_at_least_v8_1 = avisynth_if_ver > 8 || (avisynth_if_ver == 8 && avisynth_bugfix_ver >= 1);
      // 8.1: C interface frameprop access fixed, IsPropertyWritable/MakePropertyWritable support, extended GetEnvProperty queries
      has_at_least_v9 = avisynth_if_ver >= 9; // future
 

  AEP_INTERFACE_BUGFIX (c++) AVS_AEP_INTERFACE_BUGFIX (c)
    Denotes situations where there isn't a breaking change to the API,
    but we need to identify when a particular change, fix or addition
    to various API-adjacent bits might have occurred.  Could also be
    used when any new functions get added.

    Since the number is modelled as 'changes since API bump' and
    intended to be used in conjunction with checking the main
    AVISYNTH_INTERFACE_VERSION, whenever the main INTERFACE_VERSION
    gets raised, the value of INTERFACE_BUGFIX should be reset to zero.

    The BUGFIX version is added here with already incremented once,
    both because the addition of AVISYNTH_INTERFACE_BUGFIX_VERSION
    itself would require it, but also because it's intended to signify
    the fix to the C interface allowing frame properties to be read
    back (which was the situation that spurred this define to exist
    in the first place).

- CMake build environment:
  While we do need the compiler to support C++17 features, we can 
  get by on older GCC using CMake 3.6 and -std=c++-1z with some other fixes.
  CMAKE_CXX_STANDARD can be raised intelligently to 17 based on whether we detect CMake 3.8 or higher.
- Add AVISYNTHPLUS_INTERFACE_BUGFIX_VERSION

- Avisynth programming interface V8.1 or V9(?)):
  Add 'MakePropertyWritable' to the IScriptEnvironment (CPP interface), avs_make_property_writable (C interface)
  Add 'VideoFrame::IsPropertyWritable' (CPP interface), avs_is_property_writable (C interface)
  (AviSynth interface version will be stepped to V9 in the release version?)

    bool env->MakePropertyWritable(PVideoFrame *);
    bool VideoFrame::IsPropertyWritable();
  
  'MakePropertyWritable' is similar to 'MakeWritable' but it does not copy all bytes of the frame content in order to have a writable property set.
  
  Reason: 'propSet' is a filter which does not alter frame content, but sets the given frame property in its each GetFrame.
  So far it used MakeWritable to obtain a safely modifiable copy of frame properties, however - as a side-effect - full copy of frame content was performed.
  (env->getFramePropsRW alone does not ensure a uniquely modifiable property set, it just obtains a pointer which can be used in the property setter functions)
  (Note: frame properties of frames obtained by NewVideoFrame, MakeWritable and SubFrame are still safe to modify)
- Expr: when actual bit depth is too large for building LUT table, fallback to realtime mode.
  lut_x 1D (realtime when 32 bit) 
  lut_xy 2D (realtime when 16 or 32 bits)
- Expr: allow 'f32' as internal autoscale target (was: i8, i10, i12, i14, i16 were accepted, only integers)
  affects: 'scale_inputs' when "int", "intf", "all", "allf"
  more on that (todo: refresh docs) http://avisynth.nl/index.php/Expr
- Expr: fix conversion factor (+correct chroma scaling) when integer-to-integer full-scale automatic range scaling was required
- New: Expr: new parameter integer 'lut'
  integer 'lut' (default 0)
    0: realtime expression
    1: expression is converted to 1D lut (lut_x)
    2: expression is converted to 2D lut (lut_xy)
   Valid bit depths: lut=1 : 8-16 bits. lut=2 : 8-14 bits. Note: a 14 bit 2D lut needs (2^14)*(2^14)*2 bytes buffer in memory per plane (~1GByte)
   In lut mode some keywords are forbidden in the expression: sx, sy, sxr, syr, frameno, time, relative pixel addressing
- New: ArrayAdd(a, b): appends b to the end of a (a is array, b any value to append)
  Example:
    a = []
    a=ArrayAdd(a,[1,2]) # [[1,2]]
    a=ArrayIns(a,3,0) # [3,[1,2]]
    a=ArrayAdd(a,"s1") # [3,[1,2],"s1"]
    a=ArrayAdd(a,"s2") # [3,[1,2],"s1","s2"]
    a=ArrayDel(a,2) # [3,[1,2],"s2"]
- New: ArrayIns(a, b, n): inserts b into a to position n (a is array, b is the value to insert, n is a zero based index. 0: inserts at the beginning, array_size: inserts after the last element)
- New: ArrayDel(a, n): removes the n-th element from a (a is array, n is a zero based index, must be a valid index between 0 and arraysize-1)
- Enhancement: xPlaneMin/Max/Median/MinMaxDifference runtime functions to accept old packed formats (RGB24/32/48/64 and YUY2)
  (By autoconverting them to Planar RGB or YV16)
- New runtime function: PlaneMinMaxStats(clip, float "threshold", int "offset", int "plane", bool "setvar")
  Returns an 5-element array with [min,max,thresholded minimum,thresholded maximum,median]
  Parameters:
    float 'threshold': a percent number between 0.0 and 100.0%
    int 'offset': if not 0, they can be used for pulling statistics from a frame number relative to the actual one
    int 'plane' (default 0): 
         0, 1, 2 or 3 
         for YUV inputs they mean Y=0,U=1,V=2,A=3 planes
         for RGB inputs R=0,G=1,B=2 and A=3 planes
    bool 'setvar' (default false): 
         when true then it writes a global variables named 
        "PlaneStats_min" "PlaneStats_max" "PlaneStats_thmin" "PlaneStats_thmax" "PlaneStats_median"

  Note: using global variables are thread safe from ScriptClip only when used with 'function'-syntax call with its default 'local'=true

  Example:

  # function-syntax ScriptClip + runtime function call + dedicated global var demo
  # Here 'local'=true (for the sake of the demo; this is the default for this mode).
  # 'local'=true makes a dedicated global variable area, in which 'last' and 'current frame'
  # A function can see only global variables. 'last' and 'current_frame' are available here - they are global variables which were
  # set by ScriptClip after creating a safe global variable stack.
  # PlaneMinMaxStats writes five global variables "PlaneStats_min", "PlaneStats_max", "PlaneStats_thmin", "PlaneStats_thmax", "PlaneStats_median"
  ScriptClip( function [] () {
    x=PlaneMinMaxStats(threshold=30, offset=0, plane=1, setvar=true)
    subtitle("min=" + string(PlaneStats_min) + " thmax" + String(PlaneStats_thmax) + " median = " + String(PlaneStats_Median) + " median_too=" + String(x[4]))
    } , local = true) 

- Language syntax: accept arrays in the place of "val" script function parameter type regardless of being named or unnamed. 
  (Note: "val" is "." in internal function signatures)
  Example:
    BlankClip(pixel_type="yv12")
    r([1, 2, 3])
    r(n=[10,11,[12,13]])
    r("hello")
    function r(clip c, val "n")
    {
      if (IsArray(n)) {
       if (IsArray(n[2])) {
         return Subtitle(c, String(n[2,1]), align=8) #13 at the top
       } else {
         return Subtitle(c, String(n[2]), align=2) #3 at the bottom
       }
      } else {
        return Subtitle(c, String(n), align=5) #hello in the center
      }
    }

- Histogram "Levels": more precise drawing when bit depth is different from histogram's resolution bit depth
- Expr: no more banker's rounding when converting back float result to integer pixels. Using the usual truncate(x+0.5) rounding method
- ColorYUV: fix 32 bit float output
- ColorYUV: More consistent and accurate output across different color spaces, match with ConvertBits fulls-fulld conversions
- ColorYUV: set _ColorRange frame property
  levels = "TV->PC" -> full
  levels = "PC->TV" or "PC->TV.Y" or "TV" -> limited
  levels = (not given) and _ColorRange property exists -> keeps _ColorRange 
  levels = (not given) and no _ColorRange property  -> full range (old default behaviour)
- ColorYUV: when no hint is given by parameter "levels" then use _ColorRange (limited/full) frame property for establishing source range 
  If _ColorRange does not exist, it treats input as full range (old default behaviour)
  Why: when there is no limited<->full conversion, but gamma is provided then this info is still used in gamma calculation.
- ColorYUV: fixes for showyuv=true:
  - fix display when bits=32
  - "showyuv_fullrange"=true case: U and V range is chroma center +/- span (1..max) for integer bit depths instead of 0..max
    Shown ranges:
    For bits=8: 128 +/- 127 (range 1..255 is shown) (UV size is 255x255 -> 510x510 image YV12)
    bits=10: range 512 +/- 511 (UV size is 1023x1023 -> 2046x2046 image YUV420P10)
    bits=12: range 2048 +/- 2047 (UV size is same as 10 bits 1023x1023 -> 2046x2046 image YUV420P12)
    bits=14: range 8192 +/- 8191 (UV size is same as 10 bits 1023x1023 -> 2046x2046 image YUV420P14)
    bits=16: range 32768 +/- 32767 (UV size is same as 10 bits 1023x1023 -> 2046x2046 image YUV420P16)
    bits=32: range 0.0 +/- 0.5 (UV size is same as 10 bits 1023x1023 -> 2046x2046 image YUV420PS)
    In general: chroma center is 2^(N-1); span is (2^(N-1))-1 where N is the bit depth
- propShow: display _Matrix, _ColorRange and _ChromaLocation constants with friendly names
- Info on Wincows XP compatibility (Microsoft side)
  Avisynth+ can be build to be XP compatible (VS2019): v141_xp toolset and -Z-threadSafeInit flag.
  But in order to work, a _compatible_ (=not latest) Visual C++ runtime is still needed (XP support has been stopped by MS meanwhile)
  As experienced here: https://github.com/AviSynth/AviSynthPlus/issues/241
  The latest XP compatible version is probably 14.28.29213.0.
  Links to official installers for last XP compatible Microsoft Visual C++ 2015-2019 Redistributable (version 14.28.29213):
  x64 - https://download.visualstudio.microsoft.com/download/pr/566435ac-4e1c-434b-b93f-aecc71e8cffc/B75590149FA14B37997C35724BC93776F67E08BFF9BD5A69FACBF41B3846D084/VC_redist.x64.exe
  x86 - https://download.visualstudio.microsoft.com/download/pr/566435ac-4e1c-434b-b93f-aecc71e8cffc/0D59EC7FDBF05DE813736BF875CEA5C894FFF4769F60E32E87BD48406BBF0A3A/VC_redist.x86.exe 

- Expr: new function "sgn". Returns -1 when x is negative; 0 if zero; 1 when x is positive
- Expr: atan2 to SSE2 and AVX2. Up to 20x speed.
  Reference: https://stackoverflow.com/questions/46210708/atan2-approximation-with-11bits-in-mantissa-on-x86with-sse2-and-armwith-vfpv4
  max relative error = 3.53486939e-5
  Its only protection that atan2(0,0) returns 0 instead on NaN.

    ColorbarsHD(pixel_type="YUV444PS")
    x=ExtractU
    y=ExtractV
    clip_C = Expr(x,y,"y x atan2", optSSE2=false) # 52 fps
    clip_SSE2 = Expr(x,y,"y x atan2", optAVX2=false) # 484 fps
    clip_AVX2 = Expr(x,y,"y x atan2") # 1000 fps
    clip_AVX2

  (Hint: in actual AvsPMod you can check actual 16 bit or even float Y-U-V values with its color-picker)
  
- Expr: add "neg": negates stack top: a = -a
- Floyd dither ("dither"=1)
  - add native fulls-fulld support, add special chroma handling when full-range = true involved
  - valid "dither_bits" parameter 1 to 16 (similar to ordered dither)
  - special handling of low (1-7 bits) "dither_bits" => result looks nice, same as at ordered dither
  - more optimized to frequently used source and dither target bits differences: 2,4,6 and 8
    (covers typical 16->8, 10->8, 16->10 bit conversions; others have ~-10% speed, less than 8 bit targets are -20-25% )
- (fix YV411 to and from conversion - regression since recent chroma placement addition)
- ConvertBits: Support YUY2 (by autoconverting to and from YV16), support YV411
- ConvertBits: "bits" parameter is not compulsory, since the dit depths can stay as it was before. One can call like ConvertBits(fulld=true)
- ConvertBits: "dither" parameter: type changed to integer. Why was it float? :) valid values were 0 and 1
- ConvertBits: source: dither almost full refactor
- ConvertBits: allow dithering down from 8 bit sources (use case: specify parameter "dither_bits" less than 8)
  Example: My8bitVideo.ConvertBits(8, fulls=true, fulld=true, dither = 0, dither_bits=1)
- ConvertBits: ordered dither (dither_type=0) new features
  - add AVX2
  - allow odd dither_bits values, 1-16 bits (was: 2,4,6,8,..). The difference is still maximum 8, so dither_bits=1 is available
    only for 8 bit sources. (memo: for Floyd (dither=1) the minimum remained 0, allowed range is 0-16)
  - correct conversion of full-range chroma at 8-16 bits, keeping center
  - fulls-fulld mix support (conversion - if any - happens before dithering)
  - when dither target bitdepth is less than 8, then special measures are taken in order to show 'nice' output;
    using dither_bits=1 would be especially ugly without this. (dither table is treated as signed float, autocorrect levels)
    Why autocorrect? Ordered dither produces (2^dither_bits) different pixel values.
    e.g. dither_bits=1 results in pixel values 0 and 1; dither_bits=2 => 0 to 3, and so on, dither_bits=7 => 0 to 127
    When these dithered pixel values are scaled back to 8 bits, Avisynth stretches the upper extremes to 255 (8 bit case).
    At dither_bits=1 instead of 0, 128 we get 0 and 255. Or at dither_bits=2 the values 0, 64, 128, 192 are translated to 0, 85, 170, 255.
    Note: for low dither targets RGB definitely looks better.

- Use _Matrix name "bt470m" for value=4 ("fcc" is still kept)
  Source: Rename AVS_MATRIX_FCC to AVS_MATRIX_BT470_M 
- ConvertBits: Correct conversion of full-range chroma at 8-16 bits, keeping center (32 bit float was O.K.) (ditherless case)
- ConvertBits: Direct, much quicker conversions between 8-16 bit formats when either source or target is full range, avx2 support (ditherless case)
               Special even quicker case: 8->16 bit fulls=true, fulld=true (simply *257)
- ConvertBits: Fix: fulls=true->fulld=true 16->8 bit missing rounding
- CMake/source: Intel C++ Compiler 2021 and Intel C++ Compiler 19.2 support
  With the help of CMake GUI:
  - Generator: "Visual Studio 16 2019"
  - Optional toolset to use (-T option): (type to the editbox)
    For LLVM based icx: Intel C++ Compiler 2021
    For classic 19.2 icl: Intel C++ Compiler 19.2
  - Specify native compilers (choose radiobutton),
    then browse for the appropriate compiler executable path. For example:
    icx: C:\Program Files (x86)\Intel\oneAPI\compiler\latest\windows\bin\icx.exe
    icl: C:\Program Files (x86)\Intel\oneAPI\compiler\latest\windows\bin\intel64\icl.exe
  There are some bugs in the Intel-VS integration: 
  If you have errors like "xilink: : error : Assertion failed (shared/driver/drvutils.c, line 312" then
  as a workaround you must copy clang.exe (by default it is located in C:\Program Files (x86)\Intel\oneAPI\compiler\latest\windows\bin)
  to the folder beside xilink (for x64 configuration it is in C:\Program Files (x86)\Intel\oneAPI\compiler\latest\windows\bin\intel64).
- CMake/source: Intel C++ Compiler 2021 and Intel C++ Compiler 19.2 support
- ConvertBits: allow dither from 32 bits to 8-16 bits (through an internal 16 bit intermediate clip)
- ConvertBits: allow different fulls fulld when converting between integer bit depths
- ConvertBits: allow 32 bit to 32 bit conversion

    ColorbarsHD()
    # another method for converting to full range
    ConvertToRGB(matrix="709:f")
    ConvertToYUV444(matrix="709:l")
    # 8 to 32 bits
    ConvertBits(32, fulls=true, fulld=false)
    ConvertBits(32, fulld=true) # fulls=false: auto from frame prop _ColorRange
    ConvertBits(8, fulld=false, dither = 1, dither_bits=1) # low dither_bits just for fun :)
    Histogram("levels")

- Expr: consume less bytes on stack. 48x Expr call in sequence caused stack overflow
- frame property support: _ChromaLocation in various filters (e.g. ConvertToYUV422)
  New location parameter values: "top", "bottom_left", "bottom", "auto"
  "ChromaInLocation" rules:
  - if source has _ChromaLocation frame property it will be used else the default is "mpeg2" ("left")
  - if parameter is "auto" or not given at all, ChromaInLocation will be set to the above mentioned default value
  - if parameter is explicitely given, it will be used
  "ChromaOutLocation" rules:
  - default is "mpeg2" ("left")
  - if parameter is "auto" or not given at all, ChromaOutLocation will be set to the above mentioned default value
  - if parameter is explicitely given, it will be used
  
  Accepted values for "ChromaInLocation" and "ChromaOutLocation" (when source/target is a chroma subsampled format)
  (full list):
  - "left" or "mpeg2"
  - "center" or "jpeg" or "mpeg1"
  - "top_left"
  - "dv"
  - "top" 
  - "bottom_left"
  - "bottom"

  _ChromaLocation constants - as seen in propShow()
  
  AVS_CHROMA_LEFT        = 0
  AVS_CHROMA_CENTER      = 1
  AVS_CHROMA_TOP_LEFT    = 2 (4:2:0 only)
  AVS_CHROMA_TOP         = 3 (4:2:0 only)
  AVS_CHROMA_BOTTOM_LEFT = 4 (4:2:0 only)
  AVS_CHROMA_BOTTOM      = 5 (4:2:0 only)
  AVS_CHROMA_DV          = 6  Special to Avisynth

  _ChromaLocation property will be cleared when the result clip is not a chroma subsampled format (4:4:4 or RGB)

- frame propery support: preliminary _Matrix and _ColorRange in various filters
 
  Summary:

  _Matrix constants - as seen in propShow()

  AVS_MATRIX_RGB            0
  AVS_MATRIX_BT709          1
  AVS_MATRIX_UNSPECIFIED    2
  AVS_MATRIX_BT470_M        4 (instead of AVS_MATRIX_FCC)
  AVS_MATRIX_BT470_BG       5 (BT601)
  AVS_MATRIX_ST170_M        6 (practically same as 5)
  AVS_MATRIX_ST240_M        7
  AVS_MATRIX_YCGCO          8 (not supported by internal converters)
  AVS_MATRIX_BT2020_NCL     9
  AVS_MATRIX_BT2020_CL      10 (same as 9)
  AVS_MATRIX_CHROMATICITY_DERIVED_NCL 12 (not supported by internal converters)
  AVS_MATRIX_CHROMATICITY_DERIVED_CL  13 (not supported by internal converters)
  AVS_MATRIX_ICTCP          14 (not supported by internal converters)

  _ColorRange constants:

  AVS_RANGE_FULL    = 0
  AVS_RANGE_LIMITED = 1

  string "matrix" parameter possible values and their mapping (used in YUV-RGB converters)

  "rgb"          AVS_MATRIX_RGB
  "709"          AVS_MATRIX_BT709
  "unspec"       AVS_MATRIX_UNSPECIFIED
  "170m"         AVS_MATRIX_ST170_M
  "240m"         AVS_MATRIX_ST240_M
  "470bg"        AVS_MATRIX_BT470_BG
  "fcc"          AVS_MATRIX_BT470_M
  "bt470m"       AVS_MATRIX_BT470_M
  "ycgco"        AVS_MATRIX_YCGCO      not supported
  "2020ncl"      AVS_MATRIX_BT2020_NCL
  "2020cl"       AVS_MATRIX_BT2020_CL  same as 2020ncl
  "chromacl"     AVS_MATRIX_CHROMATICITY_DERIVED_CL  not supported
  "chromancl"    AVS_MATRIX_CHROMATICITY_DERIVED_NCL not supported
  "ictcp"        AVS_MATRIX_ICTCP      not supported
  "601"          AVS_MATRIX_BT470_BG   compatibility alias
  "2020"         AVS_MATRIX_BT2020_NCL compatibility alias

  the above "matrix" parameters can be followed by a "full" or "f" and "limited" or "l" or "auto" marker after a ":"
  e.g. "709:f" means the same as the old "PC.709"
  When there is no limited-ness marker, or is set to "auto" then value of _ColorRange frame property is used

  old-style "matrix" parameters are kept, their name indicate the full/limited
  For memo and the similar new string
  "rec601" same as         "170m:l"
  "rec709"                 "709:l" 
  "pc.601" and "pc601"     "170m:f"
  "pc.709" and "pc709"     "709:f" 
  "average"                - kept for compatibility, really it has no standard _Matrix equivalent
  "rec2020"                "2020cl:l"
  "pc.2020" and "pc2020"   "2020cl:f"
  
- RGB<->YUV (YUY2) conversions: frame property support _Matrix and _ColorRange
  Unlike smart external plugins, in Avisynth there is a single "matrix" parameter,
  since the function names explicitely tell whether we are converting from RGB or to RGB.

  New: additional "matrix" parameter values: see table above.
  With a new syntax "170m", "240m", "bt470m" are newly available matrixes.
  New-style matrix name can be: 
    matrix name
  or
    matrix name : full_or_limited_marker
  "auto" can appear on before and after the ":" character, e.g. "auto:full" will take matrix from frame property or default
  When converting to RGB the _Matrix parameter is set to 0 ("rgb")
- ConvertBits: frame property support: _ColorRange 
  When parameter "fulls" is not specified, whether the source clip is full or limited is decided on _ColorRange frame property.
  When no property available, then RGB clips are treated as fulls=true, while YUV are fulls=false.
  If not specified, "fulld" parameter will inherit the value of the established fulls
- ColorBars: frame property support: writes _Matrix and _ColorRange. 
  RGB: _ColorRange = 1 ("limited") - ColorBars is using studio RGB values
       _Matrix = 0 ("rgb")
- ColorBarsHD: frame property support:
        _ColorRange = 1 ("limited"), _Matrix = 1 ("709")
- BlankClip: frame property support:
   RGB: _ColorRange = 0 ("full"), _Matrix = 0 ("RGB")
   YUV: _ColorRange = 1 ("limited"), _Matrix = 6 ("170m") aka old Rec601

- Fix: Planar RGB 32 bit -> YUV matrix="PC.709"/"PC.601"/"PC.2020" resulted in greyscale image
- New function: propCopy(clip, clip [,bool 'merge'])
  Copies the frame properties of the second clip to the first.
  Parameter 'merge' (default false):
    when false: exact copy (original target properties will be lost)
    when true: keeps original properties, appends all parameters from source but overwrite if a parameter with the same name already exists.
- xxxPlaneMin xxxPlaneMax, xxxPlaneMinMaxDifference:
  - 32 bit float formats: when threshold is 0 then return real values instead of 0..1 (chroma -0.5..0.5) clamped histogram-based result
  - for threshold 0 they also became a bit quicker for 8-16 bit formats (~10% on i7-7700)
- Allow propGetXXX property getter functions called as normal functions, outside runtime
  By default frame property values are read from frame#0 which index can be overridden by the offset parameter

  Example:
    Colorbars()
    PropSet(last, "hello", 1) # Set to 1 for all frames
    # Override to 2 with runtime function except for frameNo=1
    ScriptClip("""if(current_frame!=1) {propSet("hello",2)}""")
    n0 = propGetInt("hello") # same as propGetInt("hello",offset=0)
    # or get the frame property from the Nth frame
    n1 = propGetInt("hello",offset=1)
    n2 = propGetInt("hello",offset=2)
    # n0 and n2 is 2 (overridden in runtime)
    # n1 will be 1 (keeps global setting)
    SubTitle("n0/n1/n2=" + "{n0}/{n1}/{n2}".Format)

- Add parameter string "ChromaOutPlacement" in ConvertToYV16 and ConvertToYUV422 similar to YV12/420 conversions
  4:2:2 conversions now allow ChromaInPlacement and ChromaOutPlacement parameters
  "left" ("mpeg2") and "center" ("mpeg1", "jpeg").
  Note 1: "top_left" and "dv" is still valid only for 4:2:0
  Note 2: "mpeg2" (sale as "left") was so far the default for 4:2:0 and 4:2:2 sources as well.
- Source code: use common YUV-RGB conversion matrix values and generation throughout the project
  Was: constants and calculations and inline code here and there.
  New: YUY2 RGB conversions now allow matrix "PC.2020" and "Rec2020" (as a side effect)
- 4:2:0 conversions: ChromaInPlacement and ChromaOutPlacement parameters: (see http://avisynth.nl/index.php/Convert)
  add "top_left" (new)
  add "center" and "jpeg" (as an alternative to "mpeg1"), "left" (as an alternative to "mpeg2")
- Expr: sin and cos SIMD acceleration (SSE2 and AVX2) port from VapourSynth (Akarin et al.)
- SelectRangeEvery: experimental fix on getting audio part (TomArrow; https://github.com/AviSynth/AviSynthPlus/issues/232)
- Expr: implement atan2(y,x). "yvalue xvalue atan2"
  Similarly to other trigonometric functions, function is C-only at the moment, SIMD acceleration is disabled when used in an expression.
  Returns values in the range -PI .. +PI (see C++ std::atan2 rules)
- Expr: allow x.framePropName syntax (Akarin's idea)
  Where x is the usual clip identifier letter, and after the . is the name of the frame property.
  Nonexistent or non-number frame properties return with 0.0 value
  Example (increasing brightness until frame nunmber 255)
    ColorbarsHD()
    ScriptClip("""propset("medi", current_frame)""")
    expr("x.medi","","")

- More checks on array parameters in user defined functions.
  Array-typed parameters with "name" have the value "Undefined" when they are not passed.
  Note: but the value is defined and is a zero-sized array if the parameter is unnamed, like in other Avisynth functions.

  Warning for resolving parameter handling for array of anything parameter when array(s) would be passed directly.
  Memo:
  - Avisynth signature: .+/.*
  - Script function specifier val_array or val_array_nz

  When parameter signature is array of anything (.+/.*) and the
  parameter is passed unnamed (even if it is a named parameter) then
  there is an ambiguos situation.
  Example:
    1,2,3 will be detected as [1,2,3] (compatibility)
    1 will be detected as [1] (compatibility)
    (nothing) will be detected as [], but marked in order to override it later directly by name
  Following the rule:
    Passing there a direct script array [1,2,3] will be detected as [[1,2,3]], because unnamed and untyped parameters are
    put together into an array, which has the size of the list. This is a list of 1 element which happens to be an array.
    Avisynth cannot 'guess' whether we want to define a single array directly or this array is the only one part of the list.
    [1,2,3] or [ [1,2,3] ]
  Syntax hint:
    When someone would like to pass a directly specified array (e.g. [1,2,3] instead of 1,2,3) to a .+ or .* parameter
    the parameter must be passed by name!
    Because of the existing avisynth syntax rule: arguments given as unnamed in the place of an array-of-anything parameter
    are considered to be list elements from which Avisynth creates an array

    function foo(val_array "n")
      Call                          n
      foo()                   O.K.  Undefined
      foo(1)                  O.K.  [1] (compatible Avisynth way)
      foo(1,2,3)              O.K.  [1,2,3] (compatible Avisynth way)
      foo([1,2,3])            !     [[1,2,3]] (compatible Avisynth way)
      foo([1,2,3],[4,5])      !     [[1,2,3],[4,5]] (compatible Avisynth way)
      foo(n=[1,2,3])          O.K.  [1,2,3]
      foo(n=[[1,2,3],[4,5]])  O.K.  [[1,2,3],[4,5]]
      foo(n=[])               O.K.  []
      foo(n="hello")          Syntax error, "hello" is not an array

      // unnamed signature
    function foo(val_array n)
      Call                          n
      foo()                   O.K.  [] (defined and array size is zero) Avisynth compatible behaviour

- New: script functions now supports avisynth function array signature '+' (one or more) with _nz type suffix.
  Previously only '*' style (zero or more) was supported by the original naming.
  E.g.: val_array -> .* val_array_nz -> .+, int_array -> i* int_array_nz -> i+
  Others: bool_array_nz, float_array_nz, string_array_nz, clip_array_nz, func_array_nz.
- Fix: Overlay "blend" 10+ bit clips and "opacity"<1 would leave rightmost non-mod8 (10-16 bit format) or non-mod4 (32 bit format) pixels unprocessed.
- Fix: Overlay "blend" with exactly 16 bit clips and "opacity"<1 would treat large mask values as zero (when proc>=SSE4.1)
- Parser: proper error message when a script array is passed to a non-array named function argument
  (e.g. foo(sigma=[1.1,1.1]) to [foo]f parameter signature)
- Expr: allow arbitrary variable names (instead of single letters A..Z), up to 256 different one. 
  Do not use existing keywords.
  Variable names must start with '_' or alpha, continued with '_' or alphanumeric characters.
- Expr: add 'round', 'floor', 'ceil', 'trunc' operators (nearest integer, round down, round up, round to zero)
  Acceleration requires at least SSE4.1 capable processor or else the whole expression is running in C mode.
- Fix: Expr: wrong constant folding optimization when ternary operator and Store-Only (like M^) operator is used together.
  For the expression "sxr 0.5 < 128 192 M^ 255 ?" the expected result must be the same as for "sxr 0.5 < 128 255 ?"
  Since 192 M^ is no-op regarding the expression parts.
- Recognize \' and \b and \v in escaped (e"somethg") string literals (see http://avisynth.nl/index.php/The_full_AviSynth_grammar#Literals)
- Expr: allow TAB, CR and LF characters as whitespace in expression strings
- Enable propSet to accept Clips from function type input, add propSetClip for direct parameter
  Enable propGet to return Clip, add propGetClip for direct parameter
  Clip content support for propGetAsArray propSetArray and propGetAll
- Add "PC.2020" to YUV-RGB conversion matrix set
- ColorBarsHD: use BT.709-2 for +I (Pattern 2), not BT.601
  These are from the SMPTE RP 219-1:2014, but those are also on Wikipedia now: https://en.wikipedia.org/wiki/SMPTE_color_bars
  Former values used BT.601 matrix coeff., which is wrong.
  Also fixed Pattern 1 Green.Y to conform to SMPTE RP 219-1:2014 (133, not 134).
  ColorBars: fixed studio RGB values for -I and +Q for rgb pixel types
- Speedup: Overlay mode "multiply": overlay clip is not converted to 4:4:4 internally when 420 or 422 subsampled format 
  (since only Y is used from that clip)
- Speedup: Overlay mode "multiply": SSE4.1 and AVX2 code (was: C only)
  Proper rounding in internal calculations

  SSE4.1: ~1.2-2.5X speed, AVX2: ~2-3.5X speed (i7700 x64 single thread, depending on opacity full/not, mask clip yes/no)
  # results for opacity:0.2 8/16 bit opacity:1.0 8/16 bit
  Overlay(last, ovr, mask=mask, mode = "multiply", opacity = 0.2 /*or 1.0*/)
  # 962/562/952/571 # new C
  # 934/490/1080/596 # old c
  # 1700/1370/1770/1350 # SSE4.1
  # 3050/1850/3111/1876 # AVX2

  # with no mask
  Overlay(last, ovr, mode = "multiply", opacity = 0.2 /*or 1.0*/)
  # 1330/800/2000/830 # new C
  # 1200/830/2030/1200 # old C
  # 2400/2055/2400/2060 # SSE4.1
  # 3900/3000/4100/3220 # AVX2

- Fix: ConvertAudio integer 32-to-8 bits C code garbage (regression in 3.7)
- ConvertAudio: Add direct Float from/to 8/16 conversions (C,SSE2,AVX2)
- Fix: ConvertAudio: float to 32 bit integer conversion max value glitch (regression in 3.7)
- Fix: Crash in ColorBars very first frame when followed by ResampleAudio
  Colorbars did not tolerate negative audio sample index requests which can happen at the very first frame of ResampleAudio.
- Fix: frame property access from C interface
  Unfortunately you cannot access frame properties with version <= 3.7, it will crash in those releases.

  Test sequence:
  
  AVS_VideoFrame* AVSC_CC demoCfilter_get_frame(AVS_FilterInfo* p, int n)
  {
    AVS_VideoFrame* src = avs_get_frame(p->child, n);

    avs_make_writable(p->env, &src);

    // src is an AVS_VideoFrame* 
    AVS_Map* avsmap;
    avsmap = avs_get_frame_props_rw(p->env, src);
    int error; // needed for error report, for now we'll ignore it
    // read existing propery. For test, we set it by to 77 by propSet("TestIntKey", 77) in Avisynth script
    int64_t testvalue = avs_prop_get_int(p->env, avsmap, "TestIntKey", 0, &error);

    // set new frame properties (by using the just-read integer property)
    avs_prop_set_int(p->env, avsmap, "TestIntKey2", testvalue * 2, AVS_PROPAPPENDMODE_REPLACE);
    avs_prop_set_float(p->env, avsmap, "TestFloatKey2", (testvalue * 2.0f), AVS_PROPAPPENDMODE_REPLACE);
    // store string (in general: any data), by specifying length = -1 we'll notify set_data to have strlen for getting its real size
    avs_prop_set_data(p->env, avsmap, "TestStringKey2", "testStringvalue", -1, AVS_PROPAPPENDMODE_REPLACE);

    // array test (double). Note: we can have double here, contrary to the fact that Avisynth scripts can handle only floats.
    const double test_d_array[] = { 0.5, 1.1 };
    avs_prop_set_float_array(p->env, avsmap, "TestFloatArray", test_d_array, 2);

    // array test (integer). Note: we can have int64 here, contrary to the fact that Avisynth scripts can handle only 32 bit integers.
    const int64_t test_i_array[] = { -1, 0, 1 };
    avs_prop_set_int_array(p->env, avsmap, "TestIntArray", test_i_array, 3);

    // read back the array size of a property (single properties are an 1-element arrays)
    int numElementsOfIntArray = avs_prop_num_elements(p->env, avsmap, "TestIntArray");
    avs_prop_set_int(p->env, avsmap, "TestNumElementsOfIntArray", numElementsOfIntArray, AVS_PROPAPPENDMODE_REPLACE);

    // Int property array: read back one-by-one, mul by 2, and put into another array
    int64_t test_i_array_clone[3];
    for (auto i = 0; i < numElementsOfIntArray; i++) {
      test_i_array_clone[i] = avs_prop_get_int(p->env, avsmap, "TestIntArray", i, &error) * 2;
    }
    avs_prop_set_int_array(p->env, avsmap, "TestIntArrayCloneMul2", test_i_array_clone, 3);
    
    // double property array read back as a whole, div by 3, and put into another array
    int numElementsOfFloatArray = avs_prop_num_elements(p->env, avsmap, "TestFloatArray");
    const double * tmpdarray = avs_prop_get_float_array(p->env, avsmap, "TestFloatArray", &error);
    for (auto i = 0; i < numElementsOfFloatArray; i++) {
      double tmp_d = tmpdarray[i] / 3.0;
      // first element: replace frameprop data, next ones: append new element one by one
      avs_prop_set_float(p->env, avsmap, "TestFloatArrayCloneDiv3", tmp_d, i == 0 ? AVS_PROPAPPENDMODE_REPLACE : AVS_PROPAPPENDMODE_APPEND);
    }

    // delete the key, we defined in Avisynth script
    avs_prop_delete_key(p->env, avsmap, "TestIntKey");

    // count all keys (frame property count) and put it into another frame property
    int numOfKeys = avs_prop_num_keys(p->env, avsmap);
    avs_prop_set_int(p->env, avsmap, "TestNumKeysWithoutThisOne", numOfKeys, AVS_PROPAPPENDMODE_REPLACE);

    return src;

  }


- Fix: StackVertical and packed RGB formats: get audio and parity from the first and not the last clip
- RGBAdjust: analyse=true 32 bit float support
- experimental! Fix CUDA support on specific builds (apply lost-during-merge differences from Nekopanda branch), add CMake support for the option.
- Fixes for building the core as a static library
- RGBAdjust: analyse=true 32 bit float support
- experimental! Fix CUDA support on specific builds (apply lost-during-merge differences from Nekopanda branch), add CMake support for the option.
- Fixes for building the core as a static library

20210111 3.7.0
--------------
- Haiku support
- PowerPC support
- Support for building the core as a static library (mcmtroffaes)
- Fixes for MinGW-w64 compilation
- Shibatch, TimeStretch, and ImageSeq GCC build support
- Shibatch, TimeStretch, and ImageSeq non-Windows support

20201214 3.6.2-WIP
- Fix: AddBorders did not pass frame properties

20201211 3.6.2-test7
--------------------
- Fix: propSet, propDelete and propClearAll not to ruin visibility of variables
  (property read functions are still kept being runtime only)

20201210 3.6.2-test6
-----------------------
- AviSource: fix test5 regression which refused handling old formats like YV24

20201208 3.6.2-test5
-----------------------
- Resizers: throw error on too small dimensions vs. taps
- Add ShowCRC32 debug filter. Parameters are the same as in ShowFrameNumber
- Overlay: allow 4:1:1 input
- Overlay: fix crash when mask is YUV411 and greymask=false
- Overlay: may work quicker, most input/overlay/mask/output clip format conversions moved to filter constructor
- RemoveAlphaPlane: do nothing on YUY2 instead of throwing an error message
- AviSource: support non-printing characters in fourCC code: allow [number] style, e.g. G3[0][16]
- AviSource: add Y410 (YUVA444P10) format support. Allow 'Y410' pixel_type hints.
- AviSource: decode b64a, b48r, v210, P210, P010, P016, P216, v410, Y416, r210, R10k, v308, v408, Y410 fourCCs natively.
- Fix: Average...: check for valid colorspace (e.g. no AverageB for a YUV clip)
- Add: AverageA
- New: Average...: allow YUY2, RGB24/32/48/64 inputs

20201112 3.6.2-test4
--------------------
- Fix: Overlay: Actual frame of a mask clip would be freed up too early in MT environment
- Fix: ConvertBits to ignore dither parameter instead of throwing error, in a 8 to 8 bit case

20201020 3.6.2-test3
--------------------
- Fix: GeneralConvolution missing internal rounding on 8-16 bit formats

20200831 3.6.2-test2
--------------------
- support for Win10 long file path option
- project: Improve inclusion of the ghc filesystem helper library
- project: Add a GitHub action workflow
- posix: fix crash when autoloading imports
- internally refactored ConvertAudio
- ConvertBits(8): fix dither=1 (floyd) for RGB48/RGB64
- Fix: Blur right side garbage: 16 bit+AVX2+non mod32 width
- Fix: check fn signature with implicite "last" first (3.6 regression)
- Fix: function parameters provided as arrays (e.g. GrunT callback of WriteFileIf)

20200624 3.6.2-test1
------------------
- Fix: ConvertBits (YUV): proper rounding when bit depth is reduced and origin is 10-16 bits
  (added rounder before bit-shift)
- New: Histogram("color2") to support 10+ bits.
  Allow bits=x (x=8,9,10,11,12) parameter for this kind of histogram as well.

20200619 3.6.1
--------------

Use the installer or copy files directly
- 64 bit OS:
  copy Avisynth.dll from x86 folder to the windows SysWOW64 folder
  copy Avisynth.dll from x64 folder to the windows System32 folder
- 32 bit OS
  copy Avisynth.dll from x86 folder to the windows System32 folder
- All OS: Copy appropriate files from the plugins+/plugins64+ to the appropriate Avisynth+ folder

Useful links:
-------------
Source (from 3.4): https://github.com/AviSynth/AviSynthPlus
Source (before 3.4): https://github.com/pinterf/AviSynthPlus/tree/MT
Forum: https://forum.doom9.org/showthread.php?t=168856
Forum on some avs+ filters: https://forum.doom9.org/showthread.php?t=169832
Avisynth+ info page: http://avisynth.nl/index.php/AviSynth%2B
Info on Avisynth+ new color spaces: https://forum.doom9.org/showthread.php?p=1783714#post1783714
Avisynth Universal Installer by Groucho2004: https://forum.doom9.org/showthread.php?t=172124

Short info for plugin writers
-----------------------------
  Avisynth+ header (and helper headers) are available here, or choose SDK during install:
  https://github.com/AviSynth/AviSynthPlus/tree/master/avs_core/include

  Use these headers for building x86/x64 plugins, and to use Avisynth+'s high-bitdepth related VideoInfo functions
  without any concern. When a VideoInfo function is non-existant on a system that uses your plugin,
  it won't crash, just fall back to a default behaviour. E.g. VideoInfo.BitsPerComponent() will always return 8
  when your plugin calls it on a Classic Avisynth, or pre-high bit depth Avisynth+ host.

(see readme_history.txt for details, syntax element, etc. They also appear on avisynth.nl)

20200619 3.6.1
--------------
(summary of test versions since 3.6.0)

- Fix: proper handling of autoload directories precedence:
    PluginDir+ in Software/Avisynth in HKEY_CURRENT_USER (highest priority)
    PluginDir+ in Software/Avisynth in HKEY_LOCAL_MACHINE
    PluginDir2_5 in Software/Avisynth in HKEY_CURRENT_USER
    PluginDir2_5 in Software/Avisynth in HKEY_LOCAL_MACHINE (lowest priority)
  Plugin (dll file name) found in a lower priority folder will not load if a similarly named plugin 
  already existed earlier.
- fix: GeneralConvolution: incorrect parse of negative integer coefficient (added +1)
  Regression since r2772.
- Fix: GeneralConvolution: possible crash when chroma=true for 420 and 422 formats
- Fix: ScriptClip + Runtime function object (which are new in 3.6) under heavy multithreading
- New: Histogram("levels") to allow greyscale
- Fix 3.6 regressions
  - when explicit "return last" was needed when followed by legacy function definition.
  - Windows XP is supported again (thread local storage workaround)
  - Stabilize CPP 2.5 plugins
  - allow forced named arrays usage again from plugins (MP_PipeLine)
- Frame property related constants to match existing enum style in avisynth.h.
  Plus they are not colliding now with VapourSynth's definitions.


20200607 3.6.1-test8
--------------------
- fix: GeneralConvolution: incorrect parse of negative integer coefficient (+1)
  regression since r2772
- fix: GeneralConvolution: possible crash when chroma=true for 420 and 422 formats
- Frame property related constant to match existing enum style in avisynth.h

20200605 3.6.1-test7 (internal)
- pass V3 (2.5) IScriptEnvironment for CPP plugins which directly load avisynth.dll and
  request CreateScriptEnvironment with version <= 3 

20200604 3.6.1-test6
- Fix a 3.6 regression when explicit "return last" was needed when followed by legacy function definition.
- Further Cpp 2.5 plugin compatibility fix: when a v2.5 filter is calling GetFrame from its constructor
  (hdragc)

20200603 3.6.1-test5
- Histogram("levels") to allow greyscale
- Fix 3.6 regression: CPP 2.5 interface runtime functions
- Fix 3.6 regression: forced named arrays usage from plugins (MP_PipeLine)

20200531 3.6.1-test4
- Avisynth CPP 2.5 plugin compatibility patch.
- Windows XP patch (thread local storage workaround)
- Fix: proper handling of autoload directories precedence:
    PluginDir+ in Software/Avisynth in HKEY_CURRENT_USER (highest priority)
    PluginDir+ in Software/Avisynth in HKEY_LOCAL_MACHINE
    PluginDir2_5 in Software/Avisynth in HKEY_CURRENT_USER
    PluginDir2_5 in Software/Avisynth in HKEY_LOCAL_MACHINE (lowest priority)
  Plugin (dll file name) found in a lower priority folder will not load if a similarly named plugin already existed earlier.
- Fix: ScriptClip + Runtime function object crash under multithreading
  e.g. script was crashing with high confidence:
    ColorBars(width=640, height=480, pixel_type="yv12")
    ScriptClip(function[](clip c) { propSetInt("s",function[](clip c) { return current_frame }) })
    Prefetch(100)

20200520 3.6.0
--------------
- Added predefined macros for ARM processors. Tested on Raspberry Pi 4B with the aarch64 image of Ubuntu 20.04.
- Added support for disabling the Intel SIMD intrinsics. Gets automatically disabled on non-x86 targets.
- Added submodule to allow macOS 10.13 and 10.14 to build AviSynth+ with the native Clang compiler
- Fixed some warnings on GCC (wangqr)
- Implemented GetNumPhysicalCPUs on Linux and macOS (wangqr)
- Scripts arrays (Info for plugin writers)
  Plugin parameter definition: allow type+'.' and type+'*' syntax for named array parameters

  Example:
    BlankClip has a named "colors" parameter which accepts one or more "float".
    Definition in function signature:
      [colors]f+
    Usage on the caller side: pass [235.0, 128, 128].
    Checking the proper array size is the plugin's task.

  Note:
    Plugins with named array parameter definition syntax will not work for non-array-aware AviSynth versions.

  Note2:
    Type-free unnamed arrays ".+" or ".*" cannot be followed by additional parameters

  Note3:
    A backward compatible way (AVS 2.6 and non-script-array AviSynth+ versions) of using named or unnamed arrays is to specify a single type as "." and
    the plugin would check the argument type for IsArray

- User defined functions:
  Add array parameter types:

  "array" or "val_array": array of any type.
    When unnamed, then this kind of parameter must be the very last one.
    Unnamed free-typed parametes cannot be followed by any other parameter.
    Translates to ".*" in a plugin parameter definition rule.

  "bool_array" "int_array", "float_array", "string_array", "clip_array", "func_array"
    Translates to "b*", "i*", "f*", "s*", "c*", "f*" in a plugin parameter definition rule.

  Example:

    '''
    a = [1.0, 2.0, 4.2]
    b = [3, 4, 5]
    multi = [a,b]

    sum = Summa(multi[0], multi[1], 2)
    SubTitle(Format({sum}))

    Function Summa(array "x", array "y", int "N")
    {
      sum = 0.0
      FOR(i=0,N-1) {
        sum = sum + x[i] * y[i]
      }
      return sum
    }

    or

    Function Summa(float_array x, float_array y, int "N")
    {
      sum = 0.0
      FOR(i=0,N-1) {
        sum = sum + x[i] * y[i]
      }
      return sum
    }
    '''

- IScriptEnvironment new additions
  Ex-IScriptEnvironment2: no-throw version of Invoke name change to -> InvokeTry
  Ex-INeo Invoke versions to -> Invoke2Try, Invoke3, Invoke3Try
  New (was not implemented in any former Interface): Invoke2 (Exception Thrower version of Invoke2Try)

- Test build published on (20200428)

20200428 3.6.0
--------------
- New: Format function

  Syntax:
    string Format(string format, [value1, value2, ...])

  Parameters
    - string format
      unnamed parameter, the format string
    - zero or more values for format replacement
      Single values are converted to string like in AviSynth String() function.
      If a value is a two dimensional array (only with AviSynth+ arrays enabled) then it can be
      used by a named lookup in the format expression.

  Returns
    formatted string

  Description:

    The format string consists of

        ordinary characters (except { and }), which are copied unchanged to the output,
        escape sequences {{ and }}, which are replaced with { and } respectively in the output, and
        replacement fields.

    Each replacement field has the following format:

        introductory { character;

        (optional)
        arg-id, a non-negative number;
        or:
        identifier which is used for lookup named parameters. (["name", value] construction)
        or:
        valid AviSynth variable name

        final } character.

    If arg-id is a number it specifies the index of the argument in args whose value is to be used for formatting;
    Index is zero based.

    If arg-id is string then it serves as a lookup key from the parameters list given as an array ["name",value] pair.
    If not found, then arg-id is searched among Avisynth variables.

    If arg-id is omitted, the arguments are used in order.
    Mixing manual and automatic indexing is not an error.

    Notes

    It is not an error to provide more arguments than the format string requires:

    Format("{} {}!", "Hello", "world", "something"); // OK, produces "Hello world!"

  Examples:
    By Avisynth variable
      max_pixel_value = 255
      SubTitle(Format("max={max_pixel_value}!"))

    By index:
      SubTitle(Format("{0} {1} {0}", "Home", "sweet"))

    In order:
      SubTitle(Format("{} {} {}", "AviSynth", "+", 2020))

    Array name-value pairs
      SubTitle(Format("maximum={max} minimum={min} max again {max}!", ["max",255], ["min",0]))

- frame properties framework, IScriptEnvironment extension
  - Core and concept ported from VapourSynth - thank you
  - Frame properties are really per-frame data which can only be read and written within runtime functions.
    Source filters can feed into frame properties useful data such as information on colorimetry.
  - frame properties are key-value(s) pairs
    - Key is an alphanumeric identifier.
    - Values can be of single value or array of type
      - 64 bit integers (32 bit integer when converted to AVSValue inside Avisynth+, AVSValue limitation)
      - 64 bit doubles (32 bit float when converted to AVSValue inside Avisynth+, AVSValue limitation)
      - strings (or data) with given length. strings are null terminated
      - frame references (PVideoFrame)
      - clip references (PClip)
    - property setting has 3 modes: 0-replace 1-append 2-touch (see AVSPropAppendMode in avisynth.h)
      0 - single value for that key is replaced
      1 - property is appended (make arrays by calling with mode=1 consecutively)
      2 - touch
  - from plugin writer's point of view:
    - new field in internal VideoFrame struct:
      properties, type is AVSMap* which is basically a reference counting struct for holding array of variants
    - frame property passing along the filter chain
      If your filter is not an in-place one (works with NewVideoFrame instead of MakeWritable) then
      you have to pass properties programatically by using a new version of env->NewVideoFrame: NewVideoFrameP.
      This alternative version of NewVideoFrame has a second PVideoFrame* parameter from which properties will be
      copied upon creating an empty VideoFrame.

      Unfortunately if even one filter in the chain is not passing frame properties then it is a dead end on the info.

      Whether you can (should) use it or not, the Avisynth interface version will tell you. (e.g. >= 8) which should be
      queried (best place: filter constructor)
      Unless you drop support for older avs+ or classic Avisynth 2.6, there will be a transient time when you have to support both world.
      Example:
        Check it:
          has_at_least_v8 = true;
          try { env->CheckVersion(8); } catch (const AvisynthError&) { has_at_least_v8 = false; }
        and use it:
          if (has_at_least_v8) dst = env->NewVideoFrameP(vi, &src); else dst = env->NewVideoFrame(vi);

      Another approach VideoFrame is created as before and later 'copyFrameProps' is used.

    - support functions
    In C interface:
      use avs_new_video_frame_p instead of avs_new_video_frame
      use avs_new_video_frame_p_a instead of avs_new_video_frame_a

      AVS_VideoFrame * AVSC_CC avs_new_video_frame_p(AVS_ScriptEnvironment * p, const AVS_VideoInfo * vi, AVS_VideoFrame * propSrc)
      AVS_VideoFrame * AVSC_CC avs_new_video_frame_p_a(AVS_ScriptEnvironment * p, const AVS_VideoInfo * vi, AVS_VideoFrame *propSrc, int align)

    In avisynth.h:
      enums AVSPropTypes, AVSGetPropErrors and AVSPropAppendMode
      AVSMap* is just a pointer which holds the actual list and values of frame properties.
      You don't need to know what's inside. Access its content through IScriptEnvironment or C interface calls

  - IScriptEnvironment interface is extended! For the first time since classic Avisynth 2.6 moved to IF version 6.
    AVISYNTH_INTERFACE_VERION supporting frame properties is: 8.

    [CPP interface]
    - Summary:
      New: frame propery support with NewVideoFrameP and property getter/setter/info helpers
      Old-New: moved from IScriptEnvironment2:
        GetEnvProperty (note: this is for system properties)
        Allocate, Free (buffer pools)
        GetVar versions distinctly named: GetVarTry, GetVarBool, GetVarInt, GetVarDouble, GetVarString, GetVarLong
      Old-New: moved from IScriptEnvironment2 and INeoEnv
        InvokeTry, Invoke2, Invoke2Try, Invoke3, Invoke3Try
        The "xxxTry" versions are returning bool instead of throwing NotFound if specified function does not exist

        IScriptEnvironment V6 contained only:
          AVSValue Invoke(const char* name, const AVSValue args, const char* const* arg_names = 0)
        New content, different parameter list, non-exception thrower versions
          These versions of Invoke will throw NotFound() exception like "Invoke" does
            AVSValue Invoke2(const AVSValue& implicit_last, const char* name, const AVSValue args, const char* const* arg_names = 0)
            AVSValue Invoke3(const AVSValue& implicit_last, const PFunction& func, const AVSValue args, const char* const* arg_names = 0)
          These versions of Invoke will return false instead of throwing NotFound().
            bool InvokeTry(AVSValue* result, const char* name, const AVSValue& args, const char* const* arg_names = 0)
            bool Invoke2Try(AVSValue* result, const AVSValue& implicit_last, const char* name, const AVSValue args, const char* const* arg_names = 0)
            bool Invoke3Try(AVSValue* result, const AVSValue& implicit_last, const PFunction& func, const AVSValue args, const char* const* arg_names = 0)

    - NewVideoFrame with frame property source:
        PVideoFrame NewVideoFrameP(const VideoInfo& vi, PVideoFrame* propSrc, int align = FRAME_ALIGN);

      Instead of using
        PVideoFrame src = child->GetFrame(n, env);
        PVideoFrame frame = env->NewVideoFrame(vi)

      create new video frame with passing properties by giving a source video frame
        PVideoFrame src = child->GetFrame(n, env);
        PVideoFrame dst = env->NewVideoFrameP(vi, &src);

      All core functions in Avisynth support passing frame properties as a first step.
      (Second and further steps for Avisynth core: use "standardized" frame properties for color matrix and primaries info,
      for field based flag, etc...)

      Note: MakeWritable preserves frame properties

    - copy frame properties from one frame to another: copyFrameProps
    - get property pointer for readonly access: getFramePropsRO
    - get property pointer for read/write access: getFramePropsRW
    - property count helper functions
    - clear of individual properties by key or clear everything
    - property getter and setter functions (by key, key+index, by index). Indexes are starting with 0.
    - examples in conditional.cpp, conditional_functions.cpp

      void copyFrameProps(const PVideoFrame& src, PVideoFrame& dst);
        copies frame properties from source to destination frame

      const AVSMap* getFramePropsRO(const PVideoFrame& frame);
      AVSMap* getFramePropsRW(PVideoFrame& frame);
        get pointer to the property struct for readonly or read/write purposes.
        A single frame can have one set of frame properties, but the set can be shared by multiple frames.

      int propNumKeys(const AVSMap* map);
        returns the number of frame properties for a given AVSMap. Each entry have a unique alphanumeric key.
        Key is the identifier name of the frame property.

      const char* propGetKey(const AVSMap* map, int index);
        Retrieves the key (property identifier / property name) by index. 0 <= index < propNumKeys

      int propDeleteKey(AVSMap* map, const char* key);
        Delete an entry. Returns 0 if O.K.

      int propNumElements(const AVSMap* map, const char* key);
        Queries the internal counter for a given property. Properties can be arrays,
        returned value means the array size. Non array properties return 1.
        Returns -1 if property does not exists.

      char propGetType(const AVSMap* map, const char* key);
        returns the data type.
        Type Enums are avaliable here:
          avisynth.h CPP interface:
            '''
            typedef enum AVSPropTypes {
              ptUnset = 'u',
              ptInt = 'i',
              ptFloat = 'f',
              ptData = 's',
              ptClip = 'c',
              ptFrame = 'v',
            } AVSPropTypes;
            '''
          avisynth_c.h C interface:
            '''
            enum {
              AVS_PROPTYPE_UNSET = 'u',
              AVS_PROPTYPE_INT = 'i',
              AVS_PROPTYPE_FLOAT = 'f',
              AVS_PROPTYPE_DATA = 's',
              AVS_PROPTYPE_CLIP = 'c',
              AVS_PROPTYPE_FRAME = 'v'
            };
            '''

      int64_t propGetInt(const AVSMap* map, const char* key, int index, int* error);
      double propGetFloat(const AVSMap* map, const char* key, int index, int* error);
      const char* propGetData(const AVSMap* map, const char* key, int index, int* error);
      PClip propGetClip(const AVSMap* map, const char* key, int index, int* error);
      const PVideoFrame propGetFrame(const AVSMap* map, const char* key, int index, int* error);

        Property read functions.
        Since properties can be arrays, index should be specified.
        Use index = 0 for getting the value for a non-array property, in reality these
        properties are a size=1 arrays.
        Funtion 'propGetData' can be used for both strings and for generic byte arrays.
        Variable 'error' returns an error code of failure (0=success):
          peUnset: key does not exists
          peType: e.g read integer from a float property,
          peIndex: property array index out of range)
        Error enums are defined here:
          avisynth.h CPP interface:
            typedef enum AVSGetPropErrors {
              peUnset = 1,
              peType = 2,
              peIndex = 4
            } AVSGetPropErrors;
          avisynth_c.h C interface:
            enum {
              AVS_GETPROPERROR_UNSET = 1,
              AVS_GETPROPERROR_TYPE = 2,
              AVS_GETPROPERROR_INDEX = 4
            };

      int propGetDataSize(const AVSMap* map, const char* key, int index, int* error);
        returns the string length or data size of a ptData typed element

      int propSetInt(AVSMap* map, const char* key, int64_t i, int append);
      int propSetFloat(AVSMap* map, const char* key, double d, int append);
      int propSetData(AVSMap* map, const char* key, const char* d, int length, int append);
      int propSetClip(AVSMap* map, const char* key, PClip& clip, int append);
      int propSetFrame(AVSMap* map, const char* key, const PVideoFrame& frame, int append);
        Property setter functions.
        propSetData is used for strings or for real generic byte data.
        propSetData has an extra length parameter, specify -1 for automatic string length.
        Data bytes will be copied into the property (no need to keep data for the given pointer)
        Three propery set modes supported.
        - paReplace: adds if not exists, replaces value is it already exists, losing its old value.
        - paAppend: appends to the existing property (note: they are basically arrays)
        - paTouch: do nothing if the key exists. Otherwise, the key is added to the map, with no values associated.
        See mode enums in:
          avisynth.h CPP interface:
            '''
            typedef enum AVSPropAppendMode {
              paReplace = 0,
              paAppend = 1,
              paTouch = 2
            } AVSPropAppendMode;
            '''
          or
          avisynth_c.h C interface:
            '''
            enum {
              AVS_PROPAPPENDMODE_REPLACE = 0,
              AVS_PROPAPPENDMODE_APPEND = 1,
              AVS_PROPAPPENDMODE_TOUCH = 2
            };
            '''

      const int64_t *propGetIntArray(const AVSMap* map, const char* key, int* error);
      const double *propGetFloatArray(const AVSMap* map, const char* key, int* error);
        Get the pointer to the first element of array of the given key.
        Use 'propNumElements' for establishing array size and valid index ranges.

      int propSetIntArray(AVSMap* map, const char* key, const int64_t* i, int size);
      int propSetFloatArray(AVSMap* map, const char* key, const double* d, int size);
        If an array is already available in the application, these functions provide
        a quicker way for setting the properties than to insert them one by one.
        No append mode can be specified, if property already exists it will be overwritten.
        Return value: 0 on success, 1 otherwise

      void clearMap(AVSMap* map);
        Removes all content from the property set.

      AVSMap* createMap();
      void freeMap(AVSMap* map);
        these are for internal use only

    [C interface]
    - Functions and enums are defined avisynth_c.h
    - New: frame propery support with NewVideoFrameP and property getter/setter/info helpers
    - Old-New: moved from IScriptEnvironment2:
        avs_get_env_property (note: this is for system properties)
        avs_pool_allocate, avs_pool_free (buffer pools)
        avs_get_var versions distinctly named: avs_get_var_try, avs_get_var_bool,
        avs_get_var_int, avs_get_var_double, avs_get_var_string, avs_get_var_long

    - new video frame creator functions:
        avs_new_video_frame_p
        avs_new_video_frame_p_a

      They return and empty video frame but with frame properties copied from the source parameter.

        AVS_VideoFrame * AVSC_CC avs_new_video_frame_p(const AVS_VideoInfo* vi, AVS_VideoFrame *propSrc)
        AVS_VideoFrame * AVSC_CC avs_new_video_frame_p_a(AVS_ScriptEnvironment * p, const AVS_VideoInfo * vi, AVS_VideoFrame *propSrc, int align)

      NULL as propSrc will behave like old avs_new_video_frame and avs_new_video_frame_a

    - AVS_VideoFrame struct extended with a placeholder field for 'properties' pointer
    - avs_copyFrameProps
    - avs_getFramePropsRO, avs_getFramePropsRW
    - avs_propNumKeys, avs_propGetKey, avs_propNumElements, avs_propGetType, avs_propGetDataSize
    - avs_propGetInt, avs_propGetFloat, avs_propGetData, avs_propGetClip, avs_propGetFrame, avs_propGetIntArray, avs_propGetFloatArray
    - avs_propSetInt, avs_propSetFloat, avs_propSetData, avs_propSetClip, avs_propSetFrame, avs_propSetIntArray, avs_propSetFloatArray
    - avs_propDeleteKey, avs_clearMap

    See their descriptions above, names are similar to the CPP ones.

    [AviSynth scripting language]

    Frame properties read-write possible only within runtime functions.
    Input value of setter functions are to be come from the return value of "function objects"

    Property setter function names begin with propSet
      - property value is given by the return value of a function object
        Parameters:
          clip c,
          string key_name,
          function object,
          int "mode"
            0=replace (default), 1=append, 2=touch
            There is no append mode for inserting a full array into the property.

          "propSet" csn[mode]i
            generic property setter, automatic type recognition
          "propSetInt" csn[mode]i
            accepts only integer results from function
          "propSetFloat" csn[mode]i
            accepts only float results from function
          "propSetString" csn[mode]i
            accepts only string results from function
          "propSetArray" csn
            accepts only array type results from function

      - Sets a property, its value is given directly

        "propSet" csi[mode]i
        "propSet" csf[mode]i
        "propSet" css[mode]i
        "propSet" csa
          note: array must contain only the similarly typed values, e.g. cannot mix strings with integers.

        Parameters:
          clip c,
          string key_name,
          value (type of integer, float, string and array respectively),
          int "mode"
            0=replace (default), 1=append, 2=touch
            There is no append mode for inserting a full array into the property.

    Delete a specific property entry
      "propDelete" cs

       parameters:
         clip c,
         string key_name,

    Clear all properties for a given video frame
      "propClearAll" c

       parameters:
         clip c

    Reading properties
       Common parameters:
         clip c,
         string key_name,
         integer "index", (default 0): for zero based indexing array access
         integer "offset" (default 0), similar to the other runtime functions: frame offset (e.g. -1: previous, 2: next next)

      "propGetAny" cs[index]i[offset]i
        returns the automatically detected type
      "propGetInt" cs[index]i[offset]i
        returns only if value is integer, throws an error otherwise
      "propGetFloat" cs[index]i[offset]i
        returns only if value is float, throws an error otherwise
      "propGetString"cs[index]i[offset]i
        returns only if value is string, throws an error otherwise
      "propGetAsArray" cs[offset]i
        returns an array. For a single property array size will be 1. (only in array-aware AviSynth+ versions)


    Reading all properties
      "propGetAll" c[offset]i

      (Only in array-aware AviSynth+ versions)
      Returns all frame properties in an array of [key-value] pairs. Array size will be 'numProps'
      Each key-value pair is contained in a two dimensional subarray.
      If the property value for a given key is an array again then "value" will be an array as well.
      Once you have the array with all properties you can access them with the "associative" feature of AviSynth array access

      ScriptClip("""last.propSet("cica","hello"+String(current_frame)).\
        propSetInt("test_i1",function[](clip c) { return current_frame*3 }).\
        propSet("test_i2", current_frame * 2) """)
      ScriptClip("""p = propGetAll() \
        SubTitle("size:" + String(p.ArraySize()) + " " + \
                           String(p["test_i1"]) + " " + \
                           String(p["cica"]) + " " + \
                           String(p["test_i2"]))""")
      ScriptClip("""p = propGetAll() \
        SubTitle("size:" + String(p.ArraySize()) + " " + \
        String(p[0,1]) + " " + \
        String(p[1,1]) + " " + \
        String(p[2,1]), x=0, y=20)""")

    Lists all properties to screen (a debug filter)
      "propShow" c[size]i[showtype]b

      integer "size" default(16)
        font size to use (the "Text" filter is used for display, sizes are of limited set)

      bool "showtype" default (false)
        if true, the data type in parenthesis appears next to the property key name

      Listing appears as a name = value list. Arrays values are put between [ and ]
      Top line contains number is properties. If no properties found, nothing is displayed.

    Other helper function
      "propGetDataSize" cs[index]i[offset]i
        returns the size of the string or underlying data array

      "propNumElements" cs[offset]i
        returns the array size of a given property. 1=single value

      "propNumKeys" c[offset]i
        returns number of entries (keys) for a frame

      "propGetKeyByIndex" c[index]i[offset]i
        returns the key name for the Nth property (zero based, 0<=index<propNumKeys)

      "propGetType" cs[offset]i
        returns the type of the given key
            unset: 0
            integer: 1
            float: 2
            string: 3
            clip: 4
            frame: 5

  Example 1:
  '''
    ColorBars()

    # just practicing with function objects
    ScriptClip(function[](clip c) { c.Subtitle(String(current_frame)) })

    # write frame properties with function object
    ScriptClip("""propSetInt("frameprop_from_str",func(YPlaneMax))""")
    # write frame properties with traditional script string
    ScriptClip(function[](clip c) { propSetInt("frameluma_sc_func",func(AverageLuma)) })

    # read frame properties (function object, string)
    ScriptClip(function[](clip c) { SubTitle(string(propGetInt("frameprop_from_str")), y=20) })
    ScriptClip("""SubTitle(string(propGetInt("frameluma_sc_func")), y=40)""")

    return last
  '''

  Example 2: (long - require NEW_AVSVALUE build - default on POSIX - becasue of arrays)
      '''
      ColorBars(width=640, height=480, pixel_type="yv12", staticframes=true)

      ScriptClip(function[](clip c) { propSetString("s",function[](clip c) { return "Hello " + string(current_frame) }) })
      ScriptClip(function[](clip c) { propSetString("s",function[](clip c) { return "Hello array element #2 " }, mode=1) })
      ScriptClip(function[](clip c) { propSetString("s",function[](clip c) { return "Hello array element #3 "}, mode=1 ) })

      ScriptClip(function[](clip c) { propSetString("s2",function[](clip c) { return "Another property "} ) })

      ScriptClip(function[](clip c) { propSetInt("s_int",function[](clip c) { return current_frame*1 }) })
      ScriptClip(function[](clip c) { propSetInt("s_int",function[](clip c) { return current_frame*2 }, mode=1) })
      ScriptClip(function[](clip c) { propSetInt("s_int",function[](clip c) { return current_frame*4 }, mode=1 ) })

      ScriptClip(function[](clip c) { propSetFloat("s_float",function[](clip c) { return current_frame*1*3.14 }) })
      ScriptClip(function[](clip c) { propSetFloat("s_float",function[](clip c) { return current_frame*2*3.14 }, mode=1) })
      ScriptClip(function[](clip c) { propSetFloat("s_float",function[](clip c) { return current_frame*3*3.14 }, mode=1 ) })

      ScriptClip(function[](clip c) { propSetArray("s_float_arr",function[](clip c) { return [1.1, 2.2] } ) })
      ScriptClip(function[](clip c) { propSetArray("s_int_arr",function[](clip c) { return [-1,-2,-5] } ) })
      ScriptClip(function[](clip c) { propSetArray("s_string",function[](clip c) { return ["ArrayElementS_1", "ArrayElementS_2"] } ) })
      #ScriptClip("""propDelete("s")""")
      ScriptClip(function[](clip c) {
        y = 0
        SubTitle("Prop Key count =" + String(propNumKeys), y=y)
        y = y + 15
        numKeys = propNumKeys() - 1
        for ( i = 0 , numKeys) {
          propName = propGetKeyByIndex(index = i)
          propType = propGetType(propName)
          SubTitle("#"+String(i) + " property: '" + propName + "', Type = " + String(propType) , y=y)
          y = y + 15

          for(j=0, propNumElements(propName) - 1) {
            SubTitle("element #" + String(j) + ", size = " + String(propType == 3 ? propGetDataSize(propName, index=j) : 0) + ", Value = " + String(propGetAny(propName, index=j)), y = y)
            #SubTitle("element #" + String(j) + " size = " + String(propType == 3 ? propGetDataSize(propName, index=j) : 0) + ", Value = " + String(propGetAny(propName, index=j)), y = y)
            y = y + 15
          }

        }
        return last
      })

      ScriptClip(function[](clip c) {
        a = propGetAsArray("s")
        y = 100
        x = 400
        SubTitle(string(a.ArraySize()), x=x, y=y)
        for(i=0, a.ArraySize()-1) {
          SubTitle("["+String(i)+"]="+ String(a[i]),x=x,y=y)
          y = y + 15
        }
      return last
      })

      # get int array one pass
      ScriptClip(function[](clip c) {
        a = propGetAsArray("s_int")
        y = 440
        x = 400
        SubTitle("Array size=" + string(a.ArraySize()), x=x, y=y)
        y = y + 15
        for(i=0, a.ArraySize()-1) {
          SubTitle("["+String(i)+"]="+ String(a[i]),x=x,y=y)
          y = y + 15
        }
      return last
      })

      # get float array one pass
      ScriptClip(function[](clip c) {
        a = propGetAsArray("s_float")
        y = 440
        x = 200
        SubTitle("Array size=" + string(a.ArraySize()), x=x, y=y)
        y = y + 15
        for(i=0, a.ArraySize()-1) {
          SubTitle("["+String(i)+"]="+ String(a[i]),x=x,y=y)
          y = y + 15
        }
      return last
      })

      # get string array
      ScriptClip(function[](clip c) {
        a = propGetAsArray("s_stringa")
        y = 440
        x = 000
        SubTitle("Array size=" + string(a.ArraySize()), x=x, y=y)
        y = y + 15
        for(i=0, a.ArraySize()-1) {
          SubTitle("["+String(i)+"]="+ String(a[i]),x=x,y=y)
          y = y + 15
        }
      return last
      })
    '''

- New function:
  SetMaxCPU(string feature)

  string "feature"

    "" or "none" for zero SIMD support, no processor flags are reported
    "mmx", "sse", "sse2", "sse3", "ssse3", "sse4" or "sse4.1", "sse4.2", "avx, "avx2"

    parameter is case insensitive.
    Note: "avx2" triggers FMA3 flag as well.

  Processor options w/o any modifier will limit the CPU flag report to at most the processor level.
  When "feature" is ended by '+', relevant processor feature flag will be switched on
  When "feature" is ended by '-', relevant processor feature flag will be removed
  Multiple options can be put in a comma separated list. They will evaluated in that order.

  Examples:
    SetMaxCPU("SSE2") reports at most SSE2 processor (even if AVX2 is available)
    SetMaxCPU("avx,sse4.1-") limits to avx2 but explicitely removes reporting sse4.1 support
    SetMaxCPU("none,avx2+") limits to plain C, then switches on AVX2-only support

- Script array for NEW_AVSVALUE define are working again. (default in Linux build - experimental)
  Memo:
    Compiler define NEW_AVSVALUE means script arrays
    - AVSValue deep copy for arrays (arrays in arrays in ...)
    - untyped and unconstrained element number
    - access with indexes or in a dictionary-like associative way
      array_variable = [[1, 2, 3], [4, 5, 8], "hello"]
      dictionary = [["one", 1], ["two", 2]]
      empty = []
      subarray = array_variable[0]
      val = subarray[2]
      val2 = array_variable[1, 3]
      str = array_variable[2]
      n = ArraySize(array_variable) #3
      n2 = ArraySize(empty) #0
      val3 = dictionary["two"]
    - arrays as filter parameters(named and unnamed)
      new 'a' type or use '.+' or '.*' and check AVSValue IsArray()
    - Concept is incompatible with avs 2.5 plugins due to their 'baked' interface code
    - Demo script (to be cleaned up)
      '''
      ColorBars()
      clip=last
      a = [[1,2],[3,4]]
      aa = [1]
      b = a[1,1] + ArrayGet(a, 1,0) + aa[0]

      empty_array = []
      empty_array_2 = empty_array
      #n3 = empty_array_2.ArrayGet(0) # array index out out range error!

      black_yuv_16 = [0,32768,32768]
      grey_yuv_16 = [32768,32768,32768]
      white_yuv_16 = [65535,32768,32768]
      aSelectColors = [\
        ["black", black_yuv_16],\
        ["grey", grey_yuv_16],\
        ["white",white_yuv_16],\
        ["empty",empty_array]\
      ]
      test_array = [99, 1.0, "this is a string"] # mixed types
      test_array2 = [199, 2.0, "This is a string"]

      n = ArraySize(test_array) # 3
      n2 = ArraySize(empty_array_2) # 0
      sum = FirstNSum(grey_yuv_16,2)
      b = b

      clip = clip.Text(e"Array size = " + String(n) +\
       e"\n Empty array size = " + String(n2) +\
       e"\n sum = " + String(sum) +\
       e"\n b = " + String(b) +\
       e"\n white_yuv_16[1]=" + String(aSelectColors["white"][1]) + \
       e"\n [0]=" + String(ArrayGet(test_array,0)) + \
       e"\n [1]=" + String(ArrayGet(test_array,1)) + \
       e"\n [2]=" + ArrayGet(test_array,2), lsp=0, bold=true, font="info_h")

      return clip

      function FirstNSum(array x, int n)
      {
        a = 0
        for (i=0, x.ArraySize()-1) {
          a = a + x[i]
        }
        return a
      }
      '''
- Fix: Mix/Max Runtime function 32bit float chroma: return -0.5..0.5 range (was: 0..1 range)
- AviSynth+ enhancements by Nekopanda (Neo fork)
  - Allow multiple prefetchers (MT) (mentioned earlier)
  - Multithreading and deadlock fixes for ScriptClip
    (originally I intended to pull only Neo changes which were fixing an old AVS+ bug,
     namely ScriptClip and multithreading. But I was not able to do that without pulling
     nearly everything from Neo)
  - Caching enhancements.
    SetCacheMode(0) or SetCacheMode(CACHE_FAST_START) start up time and size balanced mode
    SetCacheMode(1) or SetCacheMode(CACHE_OPTIMAL_SIZE) slow start up but optimal speed and cache size
    Latter can do wonders especially at really low memory environment
  - ScriptClip and variable stability in multithreading.
    Compared to Neo, I extended it a bit.
    Regarding visibility of variables: I had to change back from Neo's new default behaviour,
    which was similar to GRunT's local=true and yielded incompatibility with the past.
    So scripts written on the assumption of local = false did not work.
    Neo's (valid) point: they would not allow the following script to show the behavior to print "3".
    '''
      # prints 3 - Avisynth default but seems incorrect
      global foo=2
      function PrintFoo(clip c)
      { c.ScriptClip("Subtitle(string(foo))", local = false) }
      Version()
      PrintFoo()
      foo = 3
      last

      # prints 2 - Neo's default behaviour
      global foo=2
      function PrintFoo(clip c)
      { c.ScriptClip("Subtitle(string(foo))", local = true) }
      Version()
      PrintFoo()
      foo = 3
      last
    '''

    So all runtime filters* got a bool "local" parameter which acts same as in GRunT
    * ConditionalSelect, ConditionalFilter, ScriptClip, ConditionalReader, FrameEvaluate,
      WriteFile, WriteFileIf, WriteFileStart, WriteFileEnd

    If local=true the filter will evaluate its run-time script in a new variable scope,
    avoiding unintended sharing of variables between run-time scripts.
    (As of current state (20200328): Neo's WriteFileStart and WriteFileEnd works like
    local=false, in Avisynth compatible mode; all other runtime filters work in local=true mode in Neo)

    In our present Avisynth+ all runtime filters are compatible with the usual behaviour,
    but one can set the other mode by "local=true" parameter. This is when script is passed as string
    Except: when the script parameter is not a string but a function (yes! Nekopanda made it available!)
    then we in classic Avs! (haha, Classic+) kept Neo's new recommendation
    and the default value of "local" is true.

    (as a memo, ConditionalFilter still has the GRunT-like "shortcut" style parameter for bool results which implies "=" and "true"
     and parameter "condvarsuffix" for ConditionalReader)
  - UseVar, special filter, opens a clean variable environment in which only the
    variables in the parameter list can be seen.

  - "escaped" string constants: with e prefix right before the quotation mask
    n"Hello \n" will store actual LF (10) control character into the string
    \n \r \t \0 \a \f \\ and \" are converted
  - Introduce function objects into scripts
    Functions can appear as standard Avisynth variables and parameters (AVSValue type='n')
    https://github.com/nekopanda/AviSynthPlus/wiki/Language-New-Features
    Even with variable capture [] (like in GRuntT args)

  - Filter graph. Switch it on by putting SetGraphAnalysis(true) at the beginning of the script.
    Dump to text file with DumpFilterGraph. E.g. DumpFilterGraph("graph.txt", mode=2)
    Output is in "dot" format, it can be converted to an image with Graphviz as follows:
    '''
    dot -Tsvg graph.txt -o graph.svg
    '''

    SetGraphAnalysis (bool)

      Enables (True) or disables (False) graph node insertion into the instantiated filter.
      To output a filter graph, a graph node must be inserted in the filter.
      When a graph node is inserted, performance may decrease slightly due to the increase of internal function calls.
      (In most cases, there is no observable performance degradation.)

    DumpFilterGraph (clip, string "outfile", int "mode", int "nframes", bool "repeat")

      Outputs a filter graph.

      clip
          Clip to output filter graph
      string outfile = ""
          Output file path
      int mode = 0 *
      int nframes = -1
          Outputs the filter graph when processing the specified frame. The cache size and memory usage of each filter at that time are output together. This is effective when you want to know the memory usage of each filter. If -1, output when DumpFilterGraph is called (before the frame is processed).
      bool repeat = false
          Valid only when nframes> 0. Outputs a filter graph repeatedly at nframes intervals.

  - Frame properties (still from Neo!)
    (experimental, we have planned it in Avs+, probably we'll try to follow the
    VapourSynth methods(?))

20200322 3.6.0
--------------
- Fix: Multithreading enhancements and fixes (Nekopanda, from Neo fork)
  - Fix old ScriptClip (runtime filters) issue
    In this example "current_frame" variable was not seen by YDifferenceFromPrevious scripted within SubTitle
    resulting in "ERROR: Plane Difference: This filter can only be used within run-time filters" message
    Now this script finally works:
      SetLogParams("log.txt", LOG_DEBUG)
      ColorBars(width=640, height=480, pixel_type="yv12")
      ScriptClip(last, "Subtitle(String(YDifferenceFromPrevious))")
      Prefetch(4)
  - Fix deadlock of ScriptClip on MT
  - MT improvement
    - Allow multiple Prefetchers
    - Add argument to Prefetch to change # of prefetch frames without changing # of threads

    Prefetch (clip c, int threads, int "frames")

    In the original Plus, you could use only one Prefetch, but you can use any number of CUDA versions.
    Also, an argument has been added to specify the number of frames to prefetch.
    Prefetch (1,4) # Make 1 thread stand and prefetch 4 frames
    By doing so, flexible parallelization configuration is possible, such as pipeline parallelization.

    threads
    Number of threads. If it is 0, it passes without doing anything.

    frames
    Number of frames to prefetch.
    Again, if it is 0, it passes without doing anything.

- Fix: BuildPixelType: chroma subsampling of sample clip was ignored.
- POSIX: better behaviour under non-Windows because of having multiple sized fixed fonts, not only a single size=20 one.
  e.g. MessageClip(), Info(), Version(), ColorYUV "show", internal ApplyMessage
- Text filter:
  - font types with
    - "Terminus" fixed fonts added (12-14-16-18-20-22-24-28-32, regular + bold)
    - "Info_h" good old 10x20 fixed font kept under this name
  - much more international unicode characters (1354), use utf8=true under Windows
  - use fontname parameter (default "Terminus", other choice is "info_h")
  - use font_filename parameter (accepts BDF fonts at the moment - import is probably not too smart but worked for Terminus)
  - use size parameter (12 to 32, if no size is available, a smaller one is chosen but at least the smallest one)
  - new parameter: bold (default false)
- Info() filter: when parameter "size" < 0, font is automatically enlarged over 640x480
  (POSIX limit: minimum size is 12, maximum size is 32 - limited by available fixed fonts)")
- SIL OPEN FONT LICENSE added because of usage of Terminus fonts)
- able to build w/o GDI and font rendering engine under Windows, so that text-overlay filters
  work like in POSIX version of AviSynth+ (mainly for my development test)
  Use with NO_WIN_GDI define.
- Fix: ReplaceStr when the pattern string to be replaced is empty
- New:
  Exist() to have bool utf8 parameter
  This is another function to have utf8 option:
  Usage: b = Exist("Здравствуй.mkv",utf8=true). Avs file is saved as utf8 w/o BOM
- Fix: broken Exist for directories (regression appeared in 3.5.0)
- Fix: ColorYUV: really disable variable search when parameter "conditional" is false
- Development:
  ScriptEnvironment::VSprintf: parameter (void *) is changed back to va_list.
  May affect C interface (avs_vsprintf) and CPP interface (ScriptEnvironment::VSprintf)
- Enhanced: Planar RGB to YUV 444 10-14 bits: more precision (32 bit float internally)
- Enhanced: Planar RGB to YUV 444 10-16 bits: AVX2 (speed improvement)

20200302 3.5.0
--------------
- New: Native Linux, macOS, and BSD support.
- Fix: ConvertBits 32->8 for extremely out of range float pixel values.
  When pixel value in a 32 bit float format video was way out of range and greater than 128 (e.g. instead of 0 to 1.0 for Y plane) then the ConvertBits(8) had artifacts.
- Fix potential crash on exit or cache shrink (linux/gcc only?)
- Layer: support RGB24 and RGB48 (internally processed as Planar RGB - lossless pre and post conversion)
- Fix: RGBP to 444 8-14bit right side artifacts at specific widths
- Fix: "scalef" and "scaleb" for 32 bit input, when scale_inputs="floatf" produced wrong result
- Fix: missing rounder in V channel calculation of PlanarRGB->YUV 8-14bits SSE2 code
- Overlay: show error when Overlay is fed with clips with different bit depths
- New function:
  bool IsVersionOrGreater(int majorVersion, int minorVersion [,int bugfixVersion]) function
  Returns true if the current version is equal or greater than the required one in the parameters.
  The function is checking the current version against the given parameters (similar to a Windows API function)
  e.g. IsVersionOrGreater(3, 4) or IsVersionOrGreater(3, 5, 8)
- New: "Expr" helpers:
  - Constants "yrange_min", "yrange_half", "yrange_max"
    Unlike the luma/chroma plane adaptive "range_min", "range_half", "range_max" these constants always report the luma (Y) values
  - new parameter: bool clamp_float_UV (default false)
    this parameter affects clamping of chroma planes: chroma is clamped between 0..1.0 instead of -0.5..0.5s
  - "clamp_float" is not ignored (and set to true) when parameter "scale_inputs" auto-scales 32 bit float type pixels
  - New "yscalef" and "yscaleb" keywords similar to "scalef" and "scaleb" but scaling is forced to use rules for Y (non-UV) planes
  - new allowed value "floatUV" for scale_inputs.
    In short: chroma pre-shift by 0.5 for 32 bit float pixels
    Affects the chroma plane expressions of 32 bit float formats.
    Shifts the input up by 0.5 before processing it in the expression, thus values from -0.5..0.5 (zero centered) range are converted to the 0..1 (0.5 centered) one.
    Since the expression result internally has 0..1.0 range, this then is shifted back to the original -0.5..0.5 range. (since 2.2.20)
    Note: predefined constants such as cmin, cmax, range_min, range_max and range_half will be shifted as well, e.g. the expression will see range_half = 0.5
  - These modifications are similar to Masktools2 2.2.20+
- Fix: TemporalSoften possible access violation after SeparateFields (in general: after filters that only change frame pitch)
- Fix: Shibatch.DLL Access Violation crash when exit when target rate is the same as vi.audio_samples_per_second or audio_samples_per_second is 0
- Build system: Cmake: use platform toolset "ClangCL" for using built-in Clang support of Visual Studio.
  Since VS2019 v16.4 there is LLVM 9.0 support.
  Use: Tools|Get Tools and Features|Add Individual Components|Compilers, build tools, and runtimes
       [X] C++ Clang compiler for Windows
       [X] C++ Clang-cl for v142 build tools (x64/x86)
  (for LLVM Clang installed separately, please use llvm or LLVM - no change in its usage)
- New: AddBorders, LetterBox: new color_yuv parameter like in BlankClip
- Fix: Resizers to really resize alpha channel (YUVA, RGBPA)
- Fix: crash when outputting VfW (e.g. VirtualDub) for YUV444P16, other fixes for r210 and R10k formats
- WavSource: really use "utf8" parameter, fix some debug asserts
- TimeStrech: pass internal errors as Avisynth exception text (e.g. proper "Excessive sample rate!" instead of "unhandled C++ error")

20191021 3.4.0
--------------

- Merges in the MT branch, the current state of pinterf/MT, and packaging fixes
- Development HEAD is the master repo again in https://github.com/AviSynth/AviSynthPlus
- Bumps version to 3.4

20190829 r2915
--------------
- Changed: Trim, FreezeFrame, DeleteFrame, DuplicateFrame, Reverse and Loop are using frame cache again (similar to classic Avs 2.6)
- Enhanced: Expr: faster exp, log, pow for AVX2 (sekrit-twc)
- ConditionalReader: allow empty value in text file when TYPE string
- Fix: Expr: fix non-mod-8 issues for forced RGB output and YUV inputs
- New: AviSource support v308 and v408 format (packed 8 bit 444 and 4444)
- Fix: AviSource v410 source garbage (YUV444P10)
- Fix: Expr: when using parameter "scale_inputs" and the source bit depth conversion occured, predefined constants
  (ymin/max, cmin/max, range_min/max/half) would not follow the new bit depth
- Fix: ConvertToRGB from 32bit float YUV w/ full scale matrixes (pc.601, pc.709, average)
- Fix: FlipHorizontal RGB48/64 artifacts
- Enhanced: a bit quicker FlipHorizontal
- Fix: RGB64 Blur leftmost column artifact
- Enhanced: quicker RGB24/48 to Planar RGB for AVX2 capable processors
- Fix: Strip alpha channel when origin is YUVA and using ConvertToYV12/ConvertToYV16/ConvertToYV24 funtions
- Fix: garbage with ConvertToYUY2 from 8 bit YUVA colorspaces
- Enhanced: Colorbars to accept RGB24, RGB48, YV411 and 4:2:2 formats for pixel_type (now all colorspaces are supported)
- Fix: shifted chroma in ColorBars and ColorBarsHD for YUV444PS
- Fix: ConvertToY8, ConvertToYV12, ConvertToYV16, ConvertToYV24 are now allowed only for 8 bit inputs.
  Formerly these functions were allowed for 10+ bit colorspaces but were not converted to real 8 bit Y8/YV12/16/24.
  Use ConvertToY, ConvertToYUV420, ConvertToYUV422, ConvertToYUV444 instead which are bit depth independent
- New parameter in ColorYUV, RGBAdjust, Overlay, ConditionalReader.
    string "condvarsuffix"
  Allows multiple filter instances to use differently named conditional parameters.
  Prevents collision and overwrite of variables which are used by different ConditionalReader instances.
  See Conditional Variables section of http://avisynth.nl/index.php/ColorYUV.
  How does it work: when reading the global variables, the "condvarsuffix" parameter is appended to the variable name.
  E.g. ColorYUV will read "coloryuv_gain_y_a" instead of "coloryuv_gain_y" when condvarsuffix = "_a" is provided for ColorYUV.
  In the matching ConditionalReader one have to use the modified name as well:
      ConditionalReader("coloryuvoffset.txt", "coloryuv_gain_y", false, CondVarSuffix = "_a") # "_a" is added here by parameter
  or specify the suffixed name directly:
      ConditionalReader("coloryuvoffset.txt", "coloryuv_gain_y_a", false) # "_a" is added here manually

  Example:
    Colorbars(512,256).ConvertToYV12.Trim(0,299)
    ConditionalReader("coloryuvoffset.txt", "coloryuv_gain_y", false)
    a=ColorYUV(cont_y=10, conditional=true)

    Colorbars(512,256).ConvertToYV12.Trim(0,299)
    ConditionalReader("coloryuvoffset2.txt", "coloryuv_gamma_y", false, condvarsuffix="_a")
    b=ColorYUV(cont_y=10, conditional=true, condvarsuffix="_a") # will read coloryuv_gain_y_a, coloryuv_gamma_y_a, etc.

    Stackvertical(a,b)

- Fix: ColorBars: pixel_type planar RGB will set alpha to 0 instead of 255, consistent with RGB32 Alpha channel
- Fix: text colors for YUV422PS - regression since r2728 (zero-centered chroma)
- New: VirtualDub2 to display 8 bit planar RGB (needs up-to-date VirtualDub2 as well)
  VfW interface to negotiate 8 bit Planar RGB(A) with FourCCs: G3[0][8] and G4[0][8], similar to the 10-16 bit logic
  (When you would still stick with RGB24/RGB32/RGB64 export, use OPT_Enable_PlanarToPackedRGB = true)
- Fix: planar RGBA Alpha on VfW was uninitialized because it wasn't filled.
- Layer: big update
  Previously Layer was working only for RGB32 and YUY2. Overlay was used primarily for YUV. Now Layer accepts practically all formats (no RGB24).
  Note that some modes can be similar to Overlay, but the two filters are still different.
  Overlay accepts mask clip, Layer would use existing A plane.
  Overlay "blend" is Layer "add", Overlay "add" is different.
  Lighten and darken is a bit different in Overlay.
  Layer has "placement" parameter for proper mask positioning over chroma.

  - Support for all 8-32 bit Y and planar YUV/YUVA and planar RGB/RGBA formats
    When overlay clip is YUVA and RGBA, then alpha channels of overlay clip are used (similarly to RGB32 and RGB64 formats)
    Non-alpha plane YUV/planar RGB color spaces act as having a fully transparent alpha channel (like the former YUY2 only working mode)

    Note: now if destination is YUVA/RGBA, the overlay clip also has to be Alpha-aware type.
    Now A channel is not updated for YUVA targets, but RGBA targets do get the Alpha updated (like the old RGB32 mode did)
    Todo: allow non-Alpha destination and Alpha-Overlay
  - New parameter: float "opacity" (0.0 .. 1.0) optionally replaces the previous "level". Similar to "opacity" in "Overlay"
    For usage of "level" see http://avisynth.nl/index.php/Layer
    "opacity" parameter is bit depth independent, one does not have to fiddle with it like had to with level (which was maxed with level=257 when RGB32 but level=256 for YUY2/YUV)
  - threshold parameter (used for lighten/darken) is autoscaled.
    Keep it between 0 and 255, same as it was used for 8 bit videos.
  - new parameter: string "placement" default "mpeg2".
    Possible values: "mpeg2" (default), "mpeg1".
    Used in "mul", "darken" and "lighten", "add" and "subtract" modes with planar YUV 4:2:0 or 4:2:2 color spaces (not available for YUY2)
    in order to properly apply luma/overlay mask on U and V chroma channels.
  - Fix some out-of-frame memory access in YUY2 C code
  - Fix: Add proper rounding for add/subtract/lighten/darken calculations. (YUY2, RGB32, 8 bit YUV and 8 bit Planar RGB)
  - Fix: "lighten" and "darken" gave different results between yuy2 and rgb32 when Threshold<>0
    Fixed "darken" for RGB32 when Threshold<>0
    Fixed "lighten" and "darken" for YUY2 when Threshold<>0
    All the above was done by specification:
    Add: "Where overlay is brigher by threshold" => e.g. Where overlay is brigther by 10 => Where overlay > src + 10
    Calculation: alpha_mask = ovr > (src + thresh) ? level : 0;
    Add: "Where overlay is darker by threshold" => e.g. Where overlay is darker by 10 => Where overlay < src - 10
    Calculation: alpha_mask = ovr < (src - thresh) ? level : 0;
    The only correct case of the above was "lighten" for RGB32, even in Classic Avisynth. Note: Threshold=0 was O.K.
  - (Just an info: existing lighten/darken code for YUY2 is still not correct, messing up chroma a bit,
     since it uses weights from even luma positions (0,2,4,...) for U, and odd luma positions (1,3,5,...) for V)
- Avisynth C interface header (avisynth_c.h):
  - cosmetics: functions regrouped to mix less AVSC_API and AVSC_INLINE, put together Avisynth+ specific stuff
  - cosmetics: remove unused form of avs_get_rowsize and avs_get_height (kept earlier for reference)
  - use #ifndef AVSC_NO_DECLSPEC for AVSC_INLINE functions which are calling API functions
  - define alias AVS_FRAME_ALIGN as FRAME_ALIGN (keep the AVS_xxxx naming convention)
  - dynamic loader (avs_load_library) uses fallback mechanism for non-existant Avisynth+ specific functions,
    in order to have Avisynth+ specific API functions safely callable even when connected to classic Avisynth DLL
    The smart fallback mechanism for Avisynth+ specific functions ensures that if the functions are not loadable,
    they will work in a classic Avisynth compatible mode:
    Example#1: e.g. avs_is_444 will call the existing avs_is_yv24 instead
    Example#2: avs_bits_per_component will return 8 for all colorspaces (Classic Avisynth supports only 8 bits/pixel)
    Affected functions and mapping:
      avs_is_rgb48, avs_is_rgb64, avs_is_yuv444p16, avs_is_yuv422p16, avs_is_yuv420p16, avs_is_y16: returns false (0)
      avs_is_yuv444ps, avs_is_yuv422ps, avs_is_yuv420ps, avs_is_y32: returns false (0)
      avs_is_yuva, avs_is_planar_rgb, avs_is_planar_rgba: returns false (0)
      avs_is_444: calls avs_is_yv24 instead
      avs_is_422: calls avs_is_yv16 instead
      avs_is_420: calls avs_is_yv12 instead
      avs_is_y: calls avs_is_y8 instead
      avs_num_components: returns 1 for y8, 4 for RGB32, 3 otherwise
      avs_component_size: returns 1 (1 bytes)
      avs_bits_per_component: returns 8 (8 bits)
- filter "Version": update year, removed avs-plus.net link
- Updated: TimeStretch plugin with SoundTouch 2.1.3 (as of 07.Jan 2019)
- Source/Build system
  - rst documentation update (qyot27) in distrib\docs\english\source\avisynthdoc\contributing\compiling_avsplus.rst
  - GCC-MinGW build (huge thanks to qyot27)
    - various source fixes for MinGW builds,
      Avisynth core compiles fine, except external plugins Shibatch and DirectShowSource
    - CMakeLists.txt updates (user selecteble CPU arch, MinGW things)
    - PluginManager.cpp: [MSVC|GCC] should only load [MSVC|GCC] plugins (qyot27)
      This commit simply tells GCC builds of AviSynth to use the value
      of the PluginDir+GCC registry entry to find plugins, and ignore
      PluginDir and PluginDir+.  Vice-versa for MSVC builds.
      C plugins are an exception to this, since those can be loaded with
      either MSVC- or GCC-built AviSynth+.
    - GCC 8.3 support
      (see in distrib\docs\english\source\avisynthdoc\contributing\compiling_avsplus.rst)
      - Howto: MSYS2/GCC: for windows based build environment see step-by-step instructions:
        https://github.com/orlp/dev-on-windows/wiki/Installing-GCC--&-MSYS2
      - CMake: choose generator "MinGW Makefiles" from CMakeGUI or
          del CMakeCache.txt
          "c:\Program Files\CMake\bin\cmake.exe" -G "MinGW Makefiles" .
        then build with
          mingw32-make -f makefile
      - todo: fix plugin DLL names
  - CMake: Visual Studio 2019 generator support, please update to CMake to 3.14.1
  - Clang (LLVM) support with Visual Studio 2019 (16.0) / 2017 (15.9.9) (Latest Clang is 8.0 at the moment)
    - CMakeLists.txt update
    - Additional source fixes.
    - Source lines for different processor targets (SSSE3 and SSE4.1) are now separated and are compiled using function attributes.
    Clang howto:
    - Install LLVM 8.0 or later (http://releases.llvm.org/download.html, Windows pre-built libraries)
    - Install Clang Power Tools & LLVM Compiler Toolchain (thx fuchanghao)
      - https://marketplace.visualstudio.com/items?itemName=caphyon.ClangPowerTools
      - https://marketplace.visualstudio.com/items?itemName=LLVMExtensions.llvm-toolchain
    - (When using CMakeGUI) After Configure/Specify generator for this project, type LLVM for "Optional Toolset to use (-T option)"
    - Known issues:
      - compiling Avisynth only .lib is generated (used for C api), .exp is missing
      - When generating assembler output linking or other errors can happen
      - LLVM 8.0 has sub-optimal avg_epu8 intrinsic "optimization", which is fixed in LLVM 9.0 (e.g. in snapshot build 21 June 2019)

20181220 r2772
--------------
- Fix: Expr: possible Expr x64 crash under specific memory circumstances
- Fix: Expr: safer code for internal variables "Store and pop from stack" (see: Internal variables at http://avisynth.nl/index.php/Expr)

20181218 r2768
--------------
- New: Expr: allow input clips to have more planes than an implicitely specified output format
  Expr(aYV12Clip, "x 255.0 /", format="Y32") # target is Y only which needs only Y plane from YV12 -> no error
- New: Expr: Y-plane-only clip(s) can be used as source planes when a non-subsampled (rgb or 444) output format implicitely specified
  Expr(Y, "x", "x 2.0 /", "x 3.0 /", format="RGBPS") # r, g and b expression uses Y plane
  Expr(Grey_r, Grey_g, Grey_b, "x", "y 2.0 /", "z 3.0 /", format="RGBPS") # r, g and b expression uses Y plane
- Fix: ConvertToYUY2() error message for non-8 bit sources.
- Fix: Y32 source to 32 bit 420,422,444 (introduced in big the zero-chroma-center transition)
- Fix: ShowY, ShowU, ShowV crash for YUV (non-YUVA) sources
- Speedup: ConvertToY12/16... for RGB or YUY2 sources where 4:4:4 or YV16 intermediate clip was used internally
  (~1.5-2x speed, was a regression in Avs+, use intermediate cache again)
- Fix: Allow ExtractY on greyscale clips
- ImageReader/ImageSource: use cache before FreezeFrame when result is a multiframe clip (fast again, regression since an early AVS+ version)
- Resizers: don't use crop at special edge cases to avoid inconsistent results across different parameters/color spaces
- Fix: Histogram 'classic': rare incomplete histogram shown in multithreading environment
- Fix: ImageReader and ImageWriter: if path is "" then it works from/to the current directory.
- GeneralConvolution: Allow 7x7 and 9x9 matrices (was: 3x3 and 5x5)
- GeneralConvolution: All 8-32 bit formats (was: RGB32 only): YUY2 is converted to/from YV16, RGB24/32/48/64 are treated as planar RGB internally
  Since 32 bit float input is now possible, matrix elements and bias parameter now is of float type.
  For 8-16 bit clips the matrix is converted to integer before use.
- GeneralConvolution: Allow chroma subsampled formats to have their luma _or_ chroma processed. E.g. set chroma=false for a YV12 input.
- GeneralConvolution: new parameters: boolean luma (true), boolean chroma(true), boolean alpha(true)
    Default: process all planes. For RGB: luma and chroma parameters are ignored.
    Unprocessed planes are copied. Using alpha=false makes RGB32 processing faster, usually A channel is not needed.
- GeneralConvolution: MT friendly parameter parsing
- New: UTF8 filename support in AviSource, AVIFileSource, WAVSource, OpenDMLSource and SegmentedAVISource
  All functions above have a new bool utf8 parameter. Default value is false.
- Experimental: new syntax element (by addewyd): assignment operator ":=" which returns the assigned value itself.
  (Assignment within an expression)
  Examples:
    w := h := 256

    b := blankclip(width=w * 2, height = h * 3, length=40, pixel_type="yv12")
    bm = blankclip(width=w, height = w).letterbox(2,0,2,0, color=$ff)
    b

    for(j = 0, 1, 1) {
        for(i = 0, 1, 1) {
            e = 0 + i * 16 + j * 16 * 4
            ce = string(e)
            c = bm.subtitle("Y = 0x" + hex(e) + " " + ce)
            eval("c" + string(i) + string(j) + " := c")
            b := b.overlay(c, x = i * w, y = j * h)
        }
    }

    cx = c00.trim(0, 9) + c01.trim(0, 9) + c10.trim(0, 9) + c11.trim(0, 9)

    b := overlay(cx, x = 0, y = w * 2)

     /* defined NEW_AVSVALUE at build */
    array = [99, 101, "303", cnt := 4]

    for(j = 0, cnt - 1, 1) {
        b := subtitle(string(array[ind := j]), x = 100, y=(j+1) * 20)
    }

    g := b.tstfunc(kf := "first", ks := "second")

    g := subtitle((s := 4) > 0 ? t := "left" : t := "right", y = 100)
    g := subtitle(string(s) + " " + t + " " + ks, y = 150)

    eval("""h := g.subtitle("G", x=200, y = 20)""")

    h.subtitle("H " + string(ind), x = 300, y = 20)


    function tstfunc(c, d, e) {
        if (f := 1 < 2) {
            c.subtitle(string(f) + e, y = 50)
        } else {
            c.subtitle(d, y = 50)
        }
    }



20180702 r2728
--------------
- New: Expr: implement 'clip' three operand operator like in masktools2
  Description: clips (clamps) value: x minvalue maxvalue clip -> max(min(x, maxvalue), minvalue)
- New: Expr: Parameter "clamp_float"
    True: clamps 32 bit float to valid ranges, which is 0..1 for Luma or for RGB color space and -0.5..0.5 for YUV chroma UV channels
    Default false, ignored (treated as true) when scale_inputs scales float
- New: Expr: parameter "scale_inputs" (default "none")

    Autoscale any input bit depths to 8-16 bit for internal expression use, the conversion method is either full range or limited YUV range.

    Feature is similar to the one in masktools2 v2.2.15

    The primary reason of this feature is the "easy" usage of formerly written expressions optimized for 8 bits.

    Use
    - "int" : scales limited range videos, only integer formats (8-16bits) to 8 (or bit depth specified by 'i8'..'i16')
    - "intf": scales full range videos, only integer formats (8-16bits) to 8 (or bit depth specified by 'i8'..'i16')
    - "float" or "floatf" : only scales 32 bit float format to 8 bit range (or bit depth specified by 'i8'..'i16')
    - "all": scales videos to 8 (or bit depth specified by 'i8'..'i16') - conversion uses limited_range logic (mul/div by two's power)
    - "allf": scales videos to 8 (or bit depth specified by 'i8'..'i16') - conversion uses full scale logic (stretch)
    - "none": no magic

    Usually limited range is for normal YUV videos, full scale is for RGB or known-to-be-fullscale YUV

    By default the internal conversion target is 8 bits, so old expressions written for 8 bit videos will probably work.
    This internal working bit-depth can be overwritten by the i8, i10, i12, i14, i16 specifiers.

    When using autoscale mode, scaleb and scalef keywords are meaningless, because there is nothing to scale.

    How it works:
    - This option scales all 8-32 bit inputs to a common bit depth value, which bit depth is 8 by default and can be
      set to 10, 12, 14 and 16 bits by the 'i10'..'i16' keywords
      For example: scale_inputs="all" converts any inputs to 8 bit range. No truncation occurs however (no precision loss),
      because even a 16 bit data is converted to 8 bit in floating point precision, using division by 256.0 (2^16/2^8).
      So the conversion is _not_ a simple shift-right-8 in the integer domain, which would lose precision.
    - Calculates expression (lut, lut_xy, lut_xyz, lut_xyza)
    - Scales the result back to the original video bit depth.
      Clamping (clipping to valid range) and converting to integer occurs here.

    The predefined constants such as 'range_max', etc. will behave according to the internal working bit depth

    Warning#1
    This feature was created for easy porting earlier 8-bit-video-only lut expressions.
    You have to understand how it works internally.

    Let's see a 16bit input in "all" and "allf" mode (target is the default 8 bits)

    Limited range 16->8 bits conversion has a factor of 1/256.0 (Instead of shift right 8 in integer domain, float-division is used or else it would lose presision)

    Full range 16->8 bits conversion has a factor of 255.0/65535

    Using bit shifts (really it's division and multiplication by 2^8=256.0):
      result = calculate_lut_value(input / 256.0) * 256.0
    Full scale 16-8-16 bit mode ('intf', 'allf')
      result = calculate_lut_value(input / 65535.0 * 255.0 ) / 255.0 * 65535.0

    Use scale_inputs = "all" ("int", "float") for YUV videos with 'limited' range e.g. in 8 bits: Y=16..235, UV=16..240).
    Use scale_inputs = "allf" (intf, floatf) for RGB or YUV videos with 'full' range e.g. in 8 bits: channels 0..255.

    When input is 32bit float, the 0..1.0 (luma) and -0.5..0.5 (chroma) channel is scaled
    to 0..255 (8 bits), 0..1023 (i10 mode), 0..4095 (i12 mode), 0..16383(i14 mode), 0..65535(i16 mode) then back.

    Warning#2
    One cannot specify different conversion methods for converting before and after the expression.
    Neither can you specify different methods for different input clips (e.g. x is full, y is limited is not supported).

- Fix: Expr: expression string order for planar RGB is properly r-g-b like in original VapourSynth version, instead of counter-intuitive g-b-r.
- Fix: Expr: check subsampling when a different output pixel format is given
- Fix: ColorYUV: round to avoid green cast on consecutive TV<>PC
- Enhanced: Limiter to work with 32 bit float clips
- Enhanced: Limiter new parameter bool 'autoscale' default false.
  If set, minimum/maximum luma/chroma values are treated as they were in 8 bit range (but non-integer values are allowed), limiter will autoscale it.
  Default: does not scale at all, parameters are used as-is. Parameters now are of float type to handle 32 bit float values.
- New: function bool VarExist(String variable_name)
  Checks if a variable exists
  Returns true if variable exists even if it holds 'Undefined', false otherwise
- Fix: RGBAdjust memory leak when used in ScriptClip
- Enhanced: RGBAdjust new parameter: conditional (like in ColorYUV)
  The global variables "rgbadjust_xxx" with xxx = r, g, b, a, rb, gb, bb, ab, rg, gg, bg, ag are read each frame, and applied.
  It is possible to modify these variables using FrameEvaluate or ConditionalReader.
- Enhanced: RGBAdjust: support 32 bit float ('analyze' not supported, 'dither' silently ignored)
- Enhanced: AviSource to support more formats with 10+ bit depth.
  http://avisynth.nl/index.php/AviSource

  When pixel_type is not specified or set to "FULL", AviSource will try to request the formats one-by-one in the order shown in the table below.

  When a classic 'pixel_type' shares more internal formats (such as YUV422P10 first tries to request the v210 then P210 format)
  you can specify one of the specific format directly. Note that high bit-depth RGBP is prioritized against packed RGB48/64.

  The 'FourCCs for ICDecompressQuery' column means that when a codec supports the format, it will serve the frame in that one, Avisource then will convert it to the proper colorspace.

  Full support list (* = already supported):
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
- Changed (finally): 32bit float YUV colorspaces: zero centered chroma channels.
  U and V channels are now -0.5..+0.5 (if converted to full scale before) instead of 0..1
  Note: filters that relied on having the U and V channel center as 0.5 will fail.
  Why: the old UV 0..1 range was a very-very early decision in the high-bitdepth transition project. Also it is now
  compatible with z_XXXXX resizers (zimg image library, external plugin at the moment).
- New function: bool IsFloatUvZeroBased()
  For plugin or script writers who want to be compatible with pre r2672 Avisynth+ float YUV format:
    Check function availablity with FunctionExists("IsFloatUvZeroBased").
    When the function does not exists, the center value of 32 bit float U and V channel is 0.5
    When IsFloatUvZeroBased function exists, it will return true (always for official releases) if U and V is 0 based (+/-0.5)
- Fix: RGB64 Turnleft/Turnright (which are also used in RGB64 Resizers)
- Fix: Rare crash in FrameRegistry
- Enhanced: Allow ConvertToRGB24-32-48-64 functions for any source bit depths
- Enhanced: ConvertBits: allow fulls-fulld combinations when either clip is 32bits
  E.g. after a 8->32 bit fulls=false fulld=true:
  Y: 16..235 -> 0..1
  U/V: 16..240 -> -0.5..+0.5
  Note: now ConvertBits does not assume full range for YUV 32 bit float.
  Default values of fulls and fulld are now true only for RGB colorspaces.
- Fix: couldn't see variables in avsi before plugin autoloads (colors_rgb.avsi issue)
- Fix: LoadVirtualdubPlugin: Fix crash on exit when more than one instances of a filter was used in a script
- New: LoadVirtualdubPlugin update:
  - Update from interface V6 to V20, and Filtermod version 6 (partial)
  - VirtualDub2 support with extended colorspaces
    Allow RGB24, RGB48, RGB64 besides RGB32
    AutoConvert 8 bit Planar RGB to/from RGB24, RGBPA to/from RGB32 (lossless)
    AutoConvert RGB48 and 16 bit Planar RGB(A) to/from RGB64 (lossless)
    Support YUV(A) 8 bits: YV12, YV16, YV24, YV411, YUVA420P8, YUVA422P8, YUVA444P8
    Support YUV(A) 10-16 bits (properly set "ref_x" maximum levels, no autoconvert)
  - Supports prefetchProc2 callback (API >= V14 and prefetchProc2 is defined) for multiple input frames from one input clip
    PrefetchFrameDirect and PrefetchFrame are supported. PrefetchFrameSymbolic not supported
  - Supports prefetchProc callback (API >= V12 and prefetchProc is defined)
  - Supports when filter changes frame count of the output clip
  - Extra filter parameter added at the end of filter's (unnamed) parameter list
    Imported Virtualdub filters are getting and extra named parameter to the end:
      String [rangehint]
    This parameter can tell the filter about a YUV-type clip colorspace info
    Allowed values:
      "rec601": limited range + 601
      "rec709": limited range + 709
      "PC.601": full range + 601
      "PC.709": full range + 709
      ""      : not defined (same as not given)
    Parameter will be ignored when clip is non-YUV
    How it works: the hint will _not_ change the internal VirtualDub colorspace
    constant (e.g. kPixFormat_YUV420_Planar -> kPixFormat_YUV420_Planar_709 will not happen).
    Instead the base color space is kept and colorSpaceMode and colorRangeMode will set in PixmapLayout.formatEx.
    Filter can either use this information or not, depending on supported API version and its implementation.
    E.g. Crossfade(20,30) -> Crossfade(20,30,"rec601") though this specific filter won't use it.
- New function: BuildPixelType

  Creates a video format (pixel_type) string by giving a colorspace family, bit depth, optional chroma subsampling and/or a
  template clip, from which the undefined format elements are inherited.
  "[family]s[bits]i[chroma]i[compat]b[oldnames]b[sample_clip]c"

  string family: YUV, YUVA, RGB, RGBA, Y
  int bits: 8, 10, 12, 14, 16, 32
  string chroma: for YUV(A) 420,422,444,411. Ignored for RGB(A) and Y
  bool compat (default false): returns packed rgb formats for 8/16 bits (RGB default: planar RGB)
  bool oldnames (default false): returns YV12/YV16/YV24 instead of YUV420P8/YUV422P8/YUV444P8
  clip sample_clip: when supported, its format is overridden by specified parameters (e.g. only change bits=10)

  Example#1: define YUV 444 P 10

    family = "YUV"
    bits = 10
    chroma = 444
    compat = false
    oldformat = false
    s = BuildPixelType(family, bits, chroma, compat, oldformat)
    BlankClip(width=320,height=200,length=len,pixel_type=s,color=$008080).Info()

  Example#2: Change only the bit depth of the format to 16

    newbits = 16
    c = last
    s = BuildPixelType(bits=newbits, sample_clip=c)
    BlankClip(width=320,height=200,length=len,pixel_type=s,color=$008080).Info()

- Source: move to c++17, 'if constexpr' requires. Use Visual Studio 2017 (or GCC 7?). CMakeLists.txt changed.
- Source: C api: AVSC_EXPORT to dllexport in capi.h for avisynth_c_plugin_init
- Source: C api: avs_is_same_colorspace VideoInfo parameters to const
- Project struct: changelog to git.
- Include current avisynth header files and def/exp file in installer, when SDK is chosen

20180328 r2664
--------------
-   Fix: YUY2 Sharpen overflow artifacts - e.g. Sharpen(0.6)
-   Fix: Levels: 32 bit float shift in luma
-   Fix: Merge sse2 for 10-14bits (regression)
-   Fix: AVX2 resizer possible access violation in extreme resizes (e.g. 600->20)
-   Fix: 32bit float PlanarRGB<->YUV conversion matrix
-   Fix: VfW: fix b64a output for OPT_Enable_b64a=true
-   Enhanced: VfW output P010 and P016 conversion to SSE2 (VfW output is used by VirtualDub for example)
-   Enhanced: ColorYUV: recalculate 8-16 bit LUT in GetFrame only when changed frame-by-frame (e.g. in autowhite)
-   Enhanced: ConvertBits 32->8 sse2/avx2 and 32->10..16 sse41/avx2 (8-15x speed)

Not included, preliminary for the near future:
-   Big change: 32 bit float YUV formats, U and V are now zero based.
    Internally YUV 32 bit float chroma center became 0.0 (the neutral value which is 128 in the 8-bit world)
    Like in VapourSynth or in avsresizer using z.lib image library.
    'Expr' changes are affecting built-in constants/operators when used in chroma plane of a 32bit clip.
    - 'cmin', 'cmax' return the zero-based shifted versions of the 16 and 240 (8 bit) values
    - For U and V planes, constant 'range_half' results in 0.0 instead of the old 0.5
    - 'scaleb' will also give zero-based result when found in an expression for chroma plane
      (e.g. for a 32 bit float clip the '128 scaleb' will result in 0.0 instead of 128/255 for U and V planes)
      But 'scalef' when the target or source of the constant conversion is 32bits, remains independent from the plane type.
    - 'range_max' is 0.5 for 32 bit float chroma
    - new constant 'range_min', which is -0.5 for 32 bit float chroma, (0 otherwise)

    Additional warning: when you move 32bit float U or V plane to Y using CombinePlane, you have to be sure
    that your filters do not rely on this new Y plane being in 0..1 range. Or else convert it by using Expr("x 0.5 +") to the 0..1 range
    Similarly: ExtractU and ExtractV will simply return the unaltered chroma planes, which are now zero-centered


20180302 r2636
--------------
-   Fix: Blur/Sharpen crashed when YUY2.width<8, RGB32.width<4, RGB64.width<2
-   ColorYUV: don't apply TV range gamma for opt="coring" when explicit "PC->TV" is given
-   ColorbarsHD: 32bit float properly zero (0.5)-centered chroma

20180301 r2632
--------------
  - Fix: IsInterleaved returned false for RGB48 and RGB64 (raffriff42)
  - Fix: SubTitle for Planar RGB/RGBA: wrong text colors (raffriff42)
  - Fix: Packed->Planar RGB conversion failed on SSE2-only computers (SSSE3 instruction used)
  - Enhanced: Blur, Sharpen
    AVX2 for 8-16 bit planar colorspaces (>1.35x speed on i7-7770)
    SSE2 for 32 bit float formats (>1.5x speed on i7-7770)
  - Fix: Resizers for 32 bit float rare random garbage on right pixels (simd code NaN issue)
  - Enhanced: Completely rewritten 16bit and float resizers, much faster (and not only with AVX2)
  - Enhanced: 8 bit resizers: AVX2 support
  - Enhanced: Speed up converting from RGB24/RGB48 to Planar RGB(A) - SSSE3, approx. doubled fps
  - New: ConvertFPS supports 10-32 bits, planar RGB(A), YUV(A)
  - New script function: int BitSetCount(int[, int, int, ...])
    Function accepts one or more integer parameters
    Returns the number of bits set to 1 in the number or the total number of '1' bits in the supplied integers.
  - Cherry-picking from StainlessS' great RT_xxxx collection/and raffriff42 utils
  - Modded script function: Hex(int , int "width"=0)
    - New "width" parameter
    - result is in uppercase
    Width is 0 to 8, the _minimum_ width of the returned string. (8 hex digit is the max of Avisynth32 bit integer)
    When width is 0 or not supplied then string length is a minimum needed.
    Function now returns hex string in uppercase, instead of lowercase.
    Example: Hex(255,4) returns "00FF".
  - Modded script function: HexValue(String, "pos"=1)
    - new pos parameter
    Returns an int conversion of the supplied hexadecimal string.
    Conversion will cease at the first non legal number base digit, without producing an error
    Added optional pos arg default=1, start position in string of the HexString, 1 denotes the string beginning.
    Will return 0 if error in 'pos' ie if pos is less than 1 or greater than string length.
  - Modded script function: ReplaceStr(String, String, String[, Boolean "sig"=false])
    - New parameter: sig for case insensitive search (Default false: exact search)
      The uppercase/lowercase rules come from the current active code page of the OS.
  - New script functions: TrimLeft, TrimRight, TrimAll for removing beginning/trailing whitespaces from a string.
    Whitespaces: Tab (9), space (32), nbsp (160)
  - New in ColorYUV:
    New parameter: bool f2c="false".
    When f2c=true, the function accepts the Tweak-like parameters for gain, gamma and contrast
    E.g. use 0/0.5/1.0/1.5/2.0/3.0 instead of -256/-128/0/128/256/512
  - New/Fixed in ColorYUV:
    Parameter "levels" accepts "TV". (can be "TV->PC", "PC->TV", "PC->TV.Y")
    Now gamma calculation is TV-range aware when either
    - levels is "TV->PC" or
    - coring = true or
    - levels is "TV" (new - no level conversion but gamma will know proper handling)
    Previously gamma was properly calculated only for PC range.
  - New in ColorYUV:
    32 bit float support.
    - 32 bit float uses the Expr filter (8-16 bits is LUT-based). The expression is dynamically assembled for each plane, internal precision is float.
    - One can specify bits=32 when showyuv=true -> test clip in YUV420PS format
    For 32 bit clips "loose min" and "loose_max" (omitting the extreme 1/256 population from dark and bright pixels) statistics are computed
    by splitting the 0..1 into 65536 uniform ranges.
 - Modded: remove "scale" parameter from ConvertBits.
   It was introduced at the very beginning of the 10+bit development, for 32bit float conversion - never used
 - Enhanced: VfW: exporting Y416 (YUV444P16) to SSE2.
 - 8-16 bit YUV chroma to 32 bit float: keep middle chroma level (e.g. 128 in 8 bits) at 0.5.
   Calculate chroma as (x-128)/255.0 + 0.5 and not x/255.0 (Note: 32 bit float chroma center will be 0.0 in the future)
 - New: Histogram parameter "keepsource"=true (raffriff42)
   keepsource = false returns only the Histogram w/o the original picture.
   Affects "classic", "levels" and "color", "color2", ignored (n/a) for the other modes
 - New: Histogram type "color" to accept 8-32bit input and "bits"=8,9,..12 display range
 - New: Histogram parameter "markers"=true
   When markers = false:
   For "classic": no "half" level line and no invalid luma zone coloring
   For "levels":  no "half" dotted line, no coloring (neither for YUV nor for RGB)
   Ignored for the others at the moment.

20171226 r2580
----------------
  - Fix (workaround): Merge: Visual Studio 2017 15.5.1/2 generated invalid AVX2 code
    The x86 path of Merge Average was crashing due to bad code generation in the 8 bit version. Seems that thought
    Visual Studio Team was proudly announcing optimizations in their compiler (15.5.x line), it had victims.
  - Fix: Temporalsoften 10-14 bits: an SSE 4.1 instruction was used for SSE2-only CPU-s (Illegal Instruction on Athlon XP)

20171219 r2574
----------------
  - Fix: MaskHS created inverse mask. Regression after r2173
  - Build: changed avisynth.h, strict C++ conformity with Visual Studio 2017 /permissive- flag
  - Installer (finally)
  - Expr:
    - new: Indexable source clip pixels by relative x,y positions like x[-1,1]
    - new functions: sin cos tan asin acos atan
    - new operator: % (modulo)
    - new: Variables: uppercase letters A..Z for storing and reuse temporary results, frequently used computations.
    - new: predefined expr variables 'frameno', 'time', 'width', 'height'
    - fix: jitasm code generation at specific circumstances

20171115 r2544
  - Expr: fix "scalef" for 10-16 bits
  - Expr optimization: eliminate ^1 +0 -0 *1 /1

20171114 r2542
  - New filter: Expr
    Ported from Vapoursynth, with a lot of additional tweaks
    e.g. AVX2, recognize constant fill, spatial absolute and relative coordinates as input, recognize plane copy,
         no GetFrame for unused clips, converts ^2, ^3, ^4 into faster multiplication,
         converts x^0.5 into sqrt(x), borrow specific syntax elements from masktools2 (scaling, operators - but not all of them) to have partial
         compatibility in widely used existing expression strings (see readme.txt)

    clip Expr(clip c[,clip c2, ...], string expr [, string expr2[, string expr3[, string expr4]]] [, string format]
        [, bool optSSE2][, bool optAVX2][, bool optSingleMode])

  Clip and Expr parameters are unnamed
  'format' overrides the output video format
  'optSSE2' to disable simd optimizations (use C code)
  'optAVX2' to disable AVX2 optimizations (use SSE2 code)
  'optSingleMode' default false, to generate simd instructions for one XMM/YMM wide data instead of two. Experimental.
     One simd cycle processes 8 pixels (SSE2) or 16 pixels (AVX2) at a time by using two XMM/YMM registers as working set.
     Very-very complex expressions would use too many XMM/YMM registers which are then "swapped" to memory slots, that can be slow.
     Using optSingleMode = true may result in using less registers with no need for swapping them to memory slots.

  Expr accepts 1 to 26 clips as inputs and up to four expression strings, an optional video format overrider, and some debug parameters.
  Output video format is inherited from the first clip, when no format override.
  All clips have to match their dimensions and plane subsamplings.

  Expressions are evaluated on each plane, Y, U, V (and A) or R, G, B (,A).
  When an expression string is not specified, the previous expression is used for that plane. Except for plane A (alpha) which is copied by default.
  When an expression is an empty string ("") then the relevant plane will be copied (if the output clip bit depth is similar).
  When an expression is a single clip reference letter ("x") and the source/target bit depth is similar, then the relevant plane will be copied.
  When an expression is constant, then the relevant plane will be filled with an optimized memory fill method.
  Expressions are written in Reverse Polish Notation (RPN).

  Expressions use 32 bit float precision internally

  For 8..16 bit formats output is rounded and clamped from the internal 32 bit float representation to valid 8, 10, ... 16 bits range.
  32 bit float output is not clamped at all.

  - Clips: letters x, y, z, a, ... w. x is the first clip parameter, y is the second one, etc.
  - Math: * / + -
  - Math constant: pi
  - Functions: min, max, sqrt, abs, neg, exp, log, pow ^ (synonyms: "pow" and "^")
  - Logical: > < = >= <= and or xor not == & | != (synonyms: "==" and "=", "&" and "and", "|" and "or")
  - Ternary operator: ?
  - Duplicate stack: dup, dupN (dup1, dup2, ...)
  - Swap stack elements: swap, swapN (swap1, swap2, ...)
  - Scale by bit shift: scaleb (operand is treated as being a number in 8 bit range unless i8..i16 or f32 is specified)

  - Scale by full scale stretch: scalef (operand is treated as being a number in 8 bit range unless i8..i16 or f32 is specified)

  - Bit-depth aware constants
    ymin, ymax (ymin_a .. ymin_z for individual clips) - the usual luma limits (16..235 or scaled equivalents)

    cmin, cmax (cmin_a .. cmin_z) - chroma limits (16..240 or scaled equivalents)

    range_half (range_half_a .. range_half_z) - half of the range, (128 or scaled equivalents)

    range_size, range_half, range_max (range_size_a .. range_size_z , etc..)

  - Keywords for modifying base bit depth for scaleb and scalef: i8, i10, i12, i14, i16, f32

  - Spatial input variables in expr syntax:
    sx, sy (absolute x and y coordinates, 0 to width-1 and 0 to height-1)
    sxr, syr (relative x and y coordinates, from 0 to 1.0)

  Additions and differences to VS r39 version:

  ------------------------------
--------------
  (similar features to the masktools mt_lut family syntax)

  Aliases:

    introduced "^", "==", "&", "|"
  New operator: != (not equal)

  Built-in constants

    ymin, ymax (ymin_a .. ymin_z for individual clips) - the usual luma limits (16..235 or scaled equivalents)

    cmin, cmax (cmin_a .. cmin_z) - chroma limits (16..240 or scaled equivalents)

    range_half (range_half_a .. range_half_z) - half of the range, (128 or scaled equivalents)

    range_size, range_half, range_max (range_size_a .. range_size_z , etc..)

  Autoscale helper functions (operand is treated as being a number in 8 bit range unless i8..i16 or f32 is specified)

    scaleb (scale by bit shift - mul or div by 2, 4, 6, 8...)

    scalef (scale by stretch full scale - mul or div by source_max/target_max

  Keywords for modifying base bit depth for scaleb and scalef
: i8, i10, i12, i14, i16, f32

  Built-in math constant

    pi

  Alpha plane handling
. When no separate expression is supplied for alpha, plane is copied instead of reusing last expression parameter.
  Proper clamping when storing 10, 12 or 14 bit outputs

  (Faster storing of results for 8 and 10-16 bit outputs, fixed in VS r40)
  16 pixels/cycle instead of 8 when avx2, with fallback to 8-pixel case on the right edge. Thus no need for 64 byte alignment for 32 bit float.
  (Load zeros for nonvisible pixels, when simd block size goes beyond image width, to prevent garbage input for simd calculation)


  Optimizations: x^0.5 is sqrt, ^1 +0 -0 *1 /1 to nothing, ^2, ^3, ^4 is done by faster and more precise multiplication
  Spatial input variables in expr syntax:
    sx, sy (absolute x and y coordinates, 0 to width-1 and 0 to height-1)
    sxr, syr (relative x and y coordinates, from 0 to 1.0)
  Optimize: recognize constant plane expression: use fast memset instead of generic simd process. Approx. 3-4x (32 bits) to 10-12x (8 bits) speedup
  Optimize: Recognize single clip letter in expression: use fast plane copy (BitBlt)
    (e.g. for 8-16 bits: instead of load-convert_to_float-clamp-convert_to_int-store). Approx. 1.4x (32 bits), 3x (16 bits), 8-9x (8 bits) speedup

  Optimize: do not call GetFrame for input clips that are not referenced or plane-copied

  Recognize constant expression: use fast memset instead of generic simd process. Approx. 3-4x (32 bits) to 10-12x (8 bits) speedup
    Example: Expr(clip,"128","128,"128")

  Differences from masktools 2.2.10
  --------------------------------
-
  Up to 26 clips are allowed (x,y,z,a,b,...w). Masktools handles only up to 4 clips with its mt_lut, my_lutxy, mt_lutxyz, mt_lutxyza

  Clips with different bit depths are allowed

  Works with 32 bit floats instead of 64 bit double internally

  Less functions (e.g. no bit shifts)

  No float clamping and float-to-8bit-and-back load/store autoscale magic

  Logical 'false' is 0 instead of -1

  The ymin, ymax, etc built-in constants can have a _X suffix, where X is the corresponding clip designator letter. E.g. cmax_z, range_half_x

  mt_lutspa-like functionality is available through "sx", "sy", "sxr", "syr"

  No y= u= v= parameters with negative values for filling plane with constant value, constant expressions are changed into optimized "fill" mode

  Sample:

  Average three clips:
  c = Expr(clip1, clip2, clip3, "x y + z + 3 /")
  using spatial feature:
  c = Expr(clip1, clip2, clip3, "sxr syr 1 sxr - 1 syr - * * * 4096 scaleb *", "", "")

  - Add: Levels: 32 bit float format support
  - Fix: RGB (full scale) conversion: 10-16 bits to 8 bits rounding issue; pic got darker in repeated 16<->8 bit conversion chain
  - Fix: ConvertToY: remove unnecessary clamp for Planar RGB 32 bit float
  - Fix: RGB ConvertToY when rec601, rec709 (limited range) matrix. Regression since r2266
  - Optimized: Faster RGB (full scale) 10-16 bits to 8 bits conversion when dithering
  - Other: Default frame alignment is 64 bytes (was: 32 bytes). (independently of AVX512 support)
  - Build:
    Built with Visual Studio 2017, v141_xp toolset
    Note that DLL will work When you have VS2015 Update 3 redistributable.

    Download Visual Studio 2017 Redistributable from here (replaces and compatible with VS2015 redist)

    X64:
    https://go.microsoft.com/fwlink/?LinkId=746572

    x86:
    https://go.microsoft.com/fwlink/?LinkId=746571

  - Experimental x64 builds for test (internal offsets from int to size_t)
    (later note: idea was dropped because of incompatibility; too many x64 plugins out there)
  - Source: avisynth_c.h (C interface header file) changed:
    Optional define SIZETMOD. Experimental. Offsets are size_t instead of int (x64 is different!)
    Fix: avs_get_row_size calls into avs_get_row_size_p, instead of direct field access
    Fix: avs_get_height calls into avs_get_row_size_p, instead of direct field access.

20170629 r2508
  - Fix TemporalSoften: threshold < 255 (probably since r1576)

20170608 r2506
  - Fix CombinePlanes: feeding YV16 or YV411 target with Y8 sources

20170603 r2504
  - fix XP support broken in r2502

20170602 r2502
  - fix: (Important!) MT_SERIALIZED mode did not always protect filters (regression since r2069)
    Such filters sometimes were called in a reentrant way (like being MT_NICE_FILTER), which
    possibly resulted in using their internal buffers parallel.
  - Fix: ImageWriter crash when no '.' in provided filename
  - Fix: Overlay: correct masked blend: keep exact clip1 or clip2 pixel values for mask extremes 255 or 0.
    Previously 0 became 1 for zero mask, similarly 255 changed into 254 for full transparency (255) mask
  - New: script functions: StrToUtf8, StrFromUtf8: Converting a 8 bit (Ansi) string to UTF8 and back.
  - New: PluginManager always throws error on finding wrong bitness DLL in the autoload directories
  - Modified: increased x64 default MemoryMax from 1GB to 4GB, but physicalRAM/4 is still limiting
  - Modified: allow conversions between RGB24/32/48/64 (8<->16 bits) w/o ConvertBits
  - Added VS2017 and v141_xp to CMakeList.txt

20170529 r2489
  - fix: memory leak in CAVIStreamSynth (e.g. feeding vdub)
  - fix: ConvertToY for RGB64 and RGB48

20170528 r2487
  - Blur, Sharpen 10-16 bits planar and RGB64: SSE2/SSE4 (2x-4x speed)
  - New script function: int GetProcessInfo([int type = 0])
    Without parameter or type==0 the current bitness of Avisynth DLL is returned (32 or 64)
    With type=1 the function can return a bit more detailed info:
    -1: error, can't establish
    0: 32 bit DLL on 32 bit OS
    1: 32 bit DLL on 64 bit OS (WoW64 process)
    2: 64 bit DLL
  - Fix: Blur width=16 (YV12 width=32)
  - Fix: Overlay Lighten: artifacts when base clip and overlay clip have different widths (regression since r2290)
  - Fix: YUY2 HorizontalReduceBy2 did nothing if target width was not mod4
  - ImageReader: 16 bit support; "pixel_type" parameter new formats "RGB48", "RGB64" and "Y16"
  - ImageWriter: 16 bit support; save RGB48, RGB64, Y16, planar RGB(A) 8 and 16 bit formats
    (note: greyscale through devIL can be corrupt with some formats, use png)
  - ImageWriter: flip greyscale images vertically (except "raw" format)
  - SubTitle: new parameter "font_filename" allows using non-installed fonts
  - (project can be compiled using gcc)
  - Allows opening unicode filenames through VfW interface (virtualdub, MPC-HC)
  - Script function Import: new parameter bool "utf8" to treat the filenames as UTF8 encoded
    (not the script text!)
  - SubTitle: new parameter bool "utf8" for drawing strings encoded in UTF8.
      Title="Cherry blossom "+CHR($E6)+CHR($A1)+CHR($9C)+CHR($E3)+CHR($81)+CHR($AE)+CHR($E8)+CHR($8A)+CHR($B1)
      SubTitle(Title,utf8=true)
  - New script functions: ScriptNameUtf8(), ScriptFileUtf8(), ScriptDirUtf8(),
    they return variables $ScriptNameUtf8$, $ScriptFileUtf8$ and $ScriptDirUtf8$ respectively

  Known issues:
  - Filters with MT_SERIALIZED sometimes can get called in a reentrant way
  - Runtime Script functions under MT


20170316 r2455
  - Fix: IsY() script function returned IsY8() (VideoInfo::IsY was not affected)
  - New: ConvertBits, dither=1 (Floyd-Steinberg): allow any dither_bits value between 0 and 8 (0=b/w)

20170310 r2440
  - Fix Merge for float formats
  - Fix error text formatting under wine (_vsnprintf_l issue)
  - Fix Regression: YUY2 UToY copied V instead of U, since August, 2016 (v2150)

  - faster Merge: float to sse2 (both weighted and average)
  - faster ordered dither to 8bit: SSE2 (10x speed)

  - ColorBars allows any 4:2:0, 4:4:4 formats, RGB64 and all planar RGB formats
  - ColorBarsHD accepts any 4:4:4 formats
  - Dithering: Floyd-Steinberg
    Use convertBits parameter dither=1: Floyd-Steinberg (was: dither=0 for ordered dither)
  - Dithering: parameter "dither_bits"
    For dithering to lower bit depths than the target clip format
    Usage: ConvertBits(x, dither=n [, dither_bits=y])
    - ordered dither: dither_bits 2, 4, 6, ... but maximum difference between target bitdepth and dither_bits is 8
    - Floyd-Steinberg: dither_bits 1, 2, 4, 6, ... up to target bitdepth - 2
    (Avisynth+ low bitdepth, Windows 3.1 16 bit feeling I was astonished that dither_bits=6 still resulted in a quite usable image)
  - Dithering is allowed from 10-16 -> 10-16 bits (was: only 8 bit targets)
  - Dithering is allowed while keeping original bit-depth. clip10 = clip10.ConvertBits(10, dither=0, dither_bits=8)
    (you still cannot dither from 8 or 32 bit source)
  - ConditionalFilter syntax extension like Gavino's GConditional: no "=" "true" needed
  - Revert: don't give error for interlaced=true for non 4:2:0 sources (compatibility, YATTA)
  - CombinePlanes: silently autoconvert packed RGB/YUY2 inputs to planar
  - ConvertBits: show error message on YV411 conversion attempt: 8 bit only
  - ConvertBits: Don't give error message if dither=-1 (no dithering) is given for currently non-ditherable target formats
  - Script function: IsVideoFloat. returns True if clip format is 32 bit float. For convenience, same as BitsPerComponent()==32
  - ConvertToDoubleWidth and ConvertFromDoubleWidth: RGB24<->RGB48, RGB32<->RGB64
  - New MT mode: MT_SPECIAL_MT. Specify it for MP_Pipeline like filters, even if no Prefetch is used (MP_Pipeline issue, 2 fps instead of 20)


20170202 r2420
  - CombinePlanes:
    When there is only one input clip, zero-cost BitBlt-less subframes are used, which is much faster.

    e.g.: casting YUV to RGB, shuffle RGBA to ABGR, U to Y, etc..
    Target planes that are not specified, preserve their content.

    Examples:
    combineplanes(clipRGBP, planes="RGB",source_planes="BGR") # swap R and B
    combineplanes(clipYUV, planes="GBRA",source_planes="YUVA",pixel_type="RGBAP8") # cast YUVA to planar RGBA
    combineplanes(clipYUV, planes="Y",source_planes="U",pixel_type="Y8") # extract U

  - fix: SubframePlanarA (available in IScriptEnvironment2)
  - faster: Difference-type conditional functions: Simd for 10-16 bits
  - Fix: MinMax-type conditional functions (min, max, median): return float value for float clips
  - ConvertToPlanarRGB(A):
    PlanarRGB <-> PlanarRGBA is now allowed
  - ConvertToPlanarRGB(A):
    YUY2 source is now allowed (through automatic ConvertToRGB proxy)
  - faster: RemoveAlphaPlane (subframe instead of BitBlt copy)
  - Overlay: "Blend" native greyscale mode: process y plane only w/o conversion
  - Overlay: automatic use444=false for "blend"/"luma"/"chroma"
    for inputs: 420/422/444 and any RGB, lossless Planar RGB intermediate for PackedRGB
    Overlay/mask auto-follows input clip format.
    For compatibility: when greymask=true (default) and mask is RGB then mask source is the B channel
  - faster: RGB48->RGB64 SSSE3 (1,6x), RGB64->RGB48 SSSE3 (1.5x speed)
  - faster: RGB24,RGB48->PlanarRGB: uses RGB32/64 intermediate clip
  - Histogram "levels": allow RGB24/32/48/64 input.
    Display R, G and B channels instead of Y, U and V
    Reminder 1: "levels" for Planar RGB was working already
    Reminder 2: Histogram "levels" and "Classic" allows bits=xx parameter, xx=8..12
                If "bits" is specified then Histogram is drawn with 9..12 bits precision. Get a wide monitor though :)
  - ConvertBits: new parameters, partially for the future.
    bool fulls, bool fulld

    For YUV and greyscale clips the bit-depth conversion uses simple bit-shifts by default.
    YUV default is fulls=false

    RGB is converted as full-stretch (e.g. 0..255->0..65535)
    RGB default is fulls=true

    If fulld is not specified, it takes the value of fulls.
    Use case: override greyscale conversion to fullscale instead of bit-shifts

    Note 1: conversion from and to float is always full-scale
    Note 2: alpha plane is always treated as full scale
    Note 3: At the moment you cannot specify fulld to be different from fulls.


20170119 r2397
  - TemporalSoften: Planar RGB support
  - TemporalSoften: much faster average mode (thres=255)
    radius=1 +70%, radius=2 +45%,
    16bit: generally 7-8x speed (SSE2/4 instead of C)
  - SeparateColumns: 10-16bit,float,RGB48/64
  - WeaveColumns: 10-16bit,float,RGB48/64,PlanarRGB(A)
  - AddAlphaPlane: fix function parameter type list, clip type did not work
  - Internals: add SubframePlanarA to IScriptEnvirontment2 for frames with alpha plane
    General note: unlike IScriptEnvironment (that is rock solid for the time beeing), IScriptEnvironment2 is still not final.
    It is used within Avisynth+ core, but is also published in avisynth.h.
    It contains avs+ specific functions, that could not be stuffed into IScriptEnvironment without killing compatibility.
    Although it changes rarely, your plugin may not work with Avisynth+ versions after a change

  - SwapUV: YUVA support
  - ConvertToRGB32/64: copy alpha from YUVA
  - SeparateRows,SeparateFields: PlanarRGB(A),YUVA support
  - WeaveRows: PlanarRGB(A), YUVA
  - Weave (fields,frames): YUVA,PlanarRGB(A)
  - Crop: Fast crop possible for frames with alpha plane (subframe)
  - AddBorders missing l/r/top/bottom vs. subsampling check for YUVA
  - Fix: YUVA->PlanarRGBA and YUVA42x->444 missing alpha plane copy
  - YUV444->RGB48/64: fast intermediate PlanarRGB(A) then RGB48/64 (not C path)
  - RGB48/64->YUV4xx target: Planar RGB intermediate (instead of C, 10x faster)

20170111 r2380
  - Overlay: Merge the source of Chroma and Luma modes into Blend mode
             Modes Blend,Luma and Chroma now support all bit depths and 444 conversionless (use444=false) mode
  - Overlay: fix SSE2 Blend for mask+opacity for 10-16 bits
  - VfW: (vdubmod14 VfW test)
    Fix: YUV444P16 or YUVA444P16 to fourcc Y416
         if alpha channel is present, it will we copied, else filled with FFFF
    Fix: VDubPlanarHack is checked only for 8 bit YUV planar sources
20170110 r2372  (vdubmod14 VfW test)
  - New: SSE2/SSE4 for 10-16 bit <-> 10-16 bit Planar RGB (and Alpha plane) full scale conversions
    (needed for automatic planar RGB -> packed RGB VfW conversions)
  - VfW:
    Fixed: Y3[10][10],Y3[10][16] fourcc's byte order
    New: Planar RGB(A) (MagicYUV)
         10,12,14,16 bits: G3[0][10], G4[0][10], G3[0][12], G4[0][12], G3[0][14], G4[0][14], G3[0][16], G4[0][16]
    New: YUV444P16 to fourcc Y416
    New: Automatic conversion of 12, 14 and float YUV formats to 16 bit for 4:2:0 and 4:2:2
         Note: OPT_Enable_Y3_10_16 is still valid as if format was originally 16 bits
    New: Automatic conversion of 10, 12, 14 and float YUV formats to 16 bit for 4:4:4
    New: Conversion of 10, 12, 14 and float planar RGB formats to RGB64
         when global Avisynth variable Enable_PlanarToPackedRGB is true
    New: Conversion of 8 bit planar RGB formats to RGB24
         when global Avisynth variable Enable_PlanarToPackedRGB is true
    New: Conversion of 8 bit planar RGBA formats to RGB32
         when global Avisynth variable Enable_PlanarToPackedRGB is true

    Note: use OPT_VDubPlanarHack=true for YV16 and YV24 for old VirtualDub

    Supported formats:
    BRA[64],b64a,BGR[48],P010,P016,P210,P216,Y3[10][10],Y3[10][16],v210,Y416
    G3[0][10], G4[0][10], G3[0][12], G4[0][12], G3[0][14], G4[0][14], G3[0][16], G4[0][16]

    Default format FourCCs:
    Avisynth+ will report these FourCC-s, override them with defining OPT_xxx global variables
    RGB64: BRA[64]
    RGB48: BGR[48]
    YUV420P10: P010
    YUV420P16: P016
    YUV422P10: P210
    YUV422P16: P216
    YUV444P16 and YUVA444P16: Y416
    Planar RGB  10,12,14,16 bits: G3[0][10], G3[0][12], G3[0][14], G3[0][16]
    Planar RGBA 10,12,14,16 bits: G4[0][10], G4[0][12], G4[0][14], G4[0][16]

    Global variables to override default formats:
    Put them at the beginning of avs script.
    OPT_Enable_V210 = true --> v210 for YUV422P10
    OPT_Enable_Y3_10_10 = true --> Y3[10][10] for YUV422P10
    OPT_Enable_Y3_10_16 = true --> Y3[10][16] for YUV422P16
    OPT_Enable_b64a = true --> b64a for RGB64
    Enable_PlanarToPackedRGB = true --> RGBP8->RGB24, RGBAP8->RGB32, all other bit depths to RGB64

20170109 r2367dev
  - VfW: BRA[64],b64a,BGR[48],P010,P016,P210,P216,Y3[10][10],Y3[10][16],v210
    experimental!
    Default format FourCCs:
    RGB64: BRA[64]
    RGB48: BGR[48]
    YUV420P10: P010
    YUV420P16: P016
    YUV422P10: P210
    YUV422P16: P216
    Global variables to override default formats:
    OPT_Enable_V210 = true --> v210 for YUV422P10
    OPT_Enable_Y3_10_10 = true --> Y3[10][10] for YUV422P10
    OPT_Enable_Y3_10_16 = true --> Y3[10][16] for YUV422P16
    OPT_Enable_b64a = true --> b64a for RGB64
  - Overlay: blend for float format
  - Overlay: blend: SSE4 for 10-16 bit, SSE2 for float
  - AddAlphaPlane: also accepts clip with Y-only or alpha (YUVA/PRGBA/RGB32/64) for alpha source
    (was: optional int/float mask value)

20170104 r2359dev
  - Overlay:
    new parameter: bool use444 (default true for compatibility)

    Parameter is valid only for "blend" at the moment

    When set to false, and base clip/overlay clip is 420/422/Planar RGB, the overlay core
    does not convert to and back from YV24 (4:4:4 in general)

    mask can be either greyscale or can be any planar YUV when greymask=true (default)

    Works for Planar RGB, so blending is basically lossless for this format (no YUV conversion)

    todo: support for other modes, convert packed RGB to planar RGB internally instead of YUV
  - Overlay:
    Add fast 4:2:0<->4:4:4 conversion, e.g. YV16<->YV24 (only fast YUY2 and YV12 existed so far)
    instead of invoking generic core conversion
  - Overlay:
    10-16bit SSE2/SSE4 for 420/422<->444 conversions
  - Info() made a bit more compact.
    Bit depth info moved after color space info
    Does not display pre-SSE2 CPU flags when at least AVX is available
    Display AVX512 flags in separate line (would be too long)

    Reminder: Info() now has more parameters than is classic Avisynth:
      Info(): c[font]s[size]f[text_color]i[halo_color]i
    Added font (default Courier New), size (default 18), text_color and halo_color parameters, similar to (but less than) e.g. in ShowFrameNumber.

  - new CPU feature constants (see cpuid.h and avisynth_c.h)
    Detect FMA4 and AVX512F,DQ,PF,ER,CD,BW,VL,IFMA,VBMI
  - new script function:
    string ReplaceStr(string s, string pattern, string replacement)
    Function is case sensitive, parameters are unnamed
  - new script function
    int NumComponents(clip)
    returns 1 for grayscale, 3 for YUVxxx, YUY2, planar RGB or RGB24/RGB48, 4 for YUVAxxx, Planar RGBA or RGB32/64
  - new script function:
    bool HasAlpha(clip)
    returns true when clip is YUVA, Planar RGBA, or packed RGB32 or RGB64

20161222 r2347dev
  - CombinePlanes addition: setting target to default RGBP(A)/YUV(A), when inputs are greyscale and no source planes are given
    Decision is made by the target plane characters, if they are like R,G,B then target video format will be planar RGB
    Same logic applies for YUV.
    Example:
    Y1, Y2 and Y3 are greyscale clips
    Old, still valid: combineplanes(Y1, Y2, Y3, planes="RGB", source_planes="YYY", pixel_type="RGBP8")
    New:              combineplanes(Y1, Y2, Y3, planes="RGB") # result: Planar RGB
  - Fix: ScriptClip would show garbage text when internal exception occurs instead of the error message

20161211 r2343dev
  - Overlay: use y offset when greymask=true (fix)
  - Fix DV chroma positioning (UV swapped), interlaced parameter check for 4:2:0
    (fix by IanB in classic Avisynth)
  - BitBlt in 32 bit Avisynth:
    for processors with AVX or better ignore tricky isse memcpy replacement, trust in memcpy (test)
    (x64 is O.K., it always used memcpy)
  - Merge: use stream_load for AVX2
  - VDubFilter.dll:
    convert 'd' double and 'l' long typed parameters to 'f' float and 'i' int for poor AviSynth
    thus allowing the usage of such virtualdub filters
  - remove script array (new AVSValue schema) feature, cannot make it compatible with Invoke from v2.5 plugins
    until I figure out a workaround or such plugins would slowly distinct.

20161208 r2337dev
  - [feature temporarily removed, script arrays are incompatible until 2.5 plugins are supported]
    C interface array compatibility vol#2 (zero size arrays)
    (new AVSValue schema problem)
  - Merge, MergeChroma, MergeLuma: AVX2 (planar)
  - Possibly a bit faster text overlay

20161207 r2333dev
  - Overlay fix

20161206 r2331dev
  - YUY2 PlaneToY finally works
  - C interface compatible array-type AVSValue handling (new AVSValue schema problem)

20161205 r2327dev
  - [feature temporarily removed, script arrays are incompatible until 2.5 plugins are supported]
    BlankClip parameter "colors" accepts exact color values to use
    Color order: Y,U,V,A or R,G,B(,A)
    These color values are used as-is, not scaled or converted in any way.
    Reason: old colors parameter is int (32 bit) cannot hold three or four 16 bit or float values
    Example: BlankClip(width=1920,height=1080,length=1000,pixel_type="RGB64", colors=[64000,32768,1231,65535])
  - ExtractY, PlaneToY("Y") accepts YUY2 clip
  - ExtractR, ExtractG, ExtractB, ExtractA,
    and
    PlaneToY("R"), PlaneToY("G"), PlaneToY("B"), PlaneToY("A")
    functions are accepting packed RGB input (RGB24/32/48/64)
    They are converted to planar RGB on-the-fly before plane extraction
  - Histogram "levels" works from Planar RGB.
    Color legends show R, G and B.
    bits=8..12 parameter is still available for finer ultra-wide histogram display

20161201 r2322dev
  - [feature temporarily removed, script arrays are incompatible until 2.5 plugins are supported]
    constant script arrays
    (note: feature was temporarily removed in r2443)
    array_variable = [[1,2,3],[4,5,8],"hello"]
    dictionary = [["one",1],["two",2]]
    empty = []
    subarray = array_variable[0]
    val = subarray[2]
    val2 = array_variable[1,3]
    str = array_variable[2]
    n = ArraySize(array_variable) #3
    n2 = ArraySize(empty) #0
    val3 = dictionary["two"]
  - arrays as filter parameters (named and unnamed):
    new 'a' type or use '.' (any) and check AVSValue IsArray()
    todo: maybe .+ or *+ syntax?
    (note: feature was temporarily removed in r2443)
  - Planar RGB <-> YUV: SSE2 (SSE4)
  - Planar RGB <-> Packed RGB32/64: SSE2

20161120:
  make PlanarRGB TurnLeft, TurnRight work again.
  (stricter check in PlaneToY)

20161119
  Fix: PlaneToY("Y")

20161116

  Replaced AToY8, GToY8, BToY8, RToY8
  We have ExtractX, (X = R,G,B,Y,U,V,A) instead.
  That Y8 ending was too confusing.
  See new test build

20161110 Avisynth plus additions
New functions

  AToY8 , same as PlaneToY("A") for planar RGBA or YUVA
  RToY8 , same as PlaneToY("R") for planar RGB
  GToY8 , same as PlaneToY("G") for planar RGB
  BToY8 , same as PlaneToY("B") for planar RGB

  They work the same way as UToY8 and VToY8 and ConvertToY did.
  They convert to greyscale keeping the original bit-depth, not only 8 bit!
  Y8 naming was kept, because UToY and VToY already existed.

Extended function
  old: YToUV(clip clipU, clip clipV [, clip clipY ] )
  new: YToUV(clip clipU, clip clipV [, clip clipY [, clip clipA] ] )

  YToUV accepts optional alpha clip after Y clip

  Example
    U = source.UToY8()
    V = source.VToY8()
    Y = source.ConvertToY()
    A = source.AddAlphaPlane(128).AToY8()
    # swaps V, U and A, Y
    YToUV(V,U,A,Y).Histogram("levels").Info().RemoveAlphaPlane()

New function
  CombinePlanes(clip1 [,clip2, clip3, clip4], string planes [, string source_planes, string pixel_type, string sample_clip])

  Combines planes of source clip(s) into a target clip

  If sample_clip is given, target clip properties are copied from that clip
  If no sample_clip is provided, then clip1 provides the template for target clip
  An optional pixel_type string (e.g."YV24", "YUV420PS", "RGBP8") can override the base video format.

  If the source clip count is less than the given planes defined, then the last available clip is
  used as a source.

  string planes
    the target plane order (e.g. "YVU", "YYY", "RGB")
    missing target planes will be undefined in the target

  string source_planes (optional)
    the source plane order, defaulting to "YUVA" or "RGBA" depending on the video format

  Examples#1
    #combine greyscale clips into YUVA clip
    U8 = source.UToY8()
    V8 = source.VToY8()
    Y8 = source.ConvertToY()
    A8 = source.AddAlphaPlane(128).AToY8()
    CombinePlanes(Y8, U8, V8, A8, planes="YUVA", source_planes="YYYY", sample_clip=source) #pixel_type="YUV444P8"

  Examples#2
    # Copy planes between planar RGB(A) and YUV(A) without any conversion
    # yuv 4:4:4 <-> planar rgb
    source = last.ConvertBits(32) # 4:4:4
    cast_to_planarrgb = CombinePlanes(source, planes="RGB", source_planes="YUV", pixel_type="RGBPS")
    # get back a clip identical with "source"
    cast_to_yuv = CombinePlanes(cast_to_planarrgb, planes="YUV", source_planes="RGB", pixel_type="YUV444PS")

  Examples#3
    #create a black and white planar RGB clip using Y channel
    #source is a YUV clip
    grey = CombinePlanes(source, planes="RGB", source_planes="YYY", pixel_type="RGBP8")

  Examples#4
    #copy luma from one clip, U and V from another
    #source is the template
    #sourceY is a Y or YUV clip
    #sourceUV is a YUV clip
    grey = CombinePlanes(sourceY, sourceUV, planes="YUV", source_planes="YUV", sample_clip = source)

  Remark:
    When there is only one input clip, zero-cost BitBlt-less subframes are used, which is much faster.

    e.g.: casting YUV to RGB, shuffle RGBA to ABGR, U to Y, etc..
    Target planes that are not specified, preserve their content.

    Examples:
    combineplanes(clipRGBP, planes="RGB",source_planes="BGR") # swap R and B
    combineplanes(clipYUV, planes="GBRA",source_planes="YUVA",pixel_type="RGBAP8") # cast YUVA to planar RGBA
    combineplanes(clipYUV, planes="Y",source_planes="U",pixel_type="Y8") # extract U


Earlier (pre v2294) modifications and informations on behaviour of existing filter
----------------------------------------------------------------------------------
[ColorSpaceNameToPixelType]
New script function: ColorSpaceNameToPixelType()
Parameter: video colorspace string
Returns: Integer

Returns a VideoInfo::pixel_type integer from a valid colorspace string

In Avisynth+ we have way too many pixel_type's now, this function can be useful for plugins for parsing a target colorspace string parameter.

Earlier I made this function available from within avisynth core, as I made one function from the previous 3-4 different places where colorspace name parameters were parsed in a copy-paste code.

In Avisynth the existing PixelType script function returns the pixeltype name of the current clip.
This function reverses this.

It has the advantage that it returns the same (for example) YV24 constant from "YV24" or "YUV444" or "Yuv444p8", so it recognizes some possible naming variants.

csp_name = "YUV422P8"
csp_name2 = "YV16"
SubTitle("PixelType value of " + csp_name + " = " + String(ColorSpaceNameToPixelType(csp_name))\
+ " and " + csp_name2 + " = " + String(ColorSpaceNameToPixelType(csp_name2)) )

[New conditional functions]

Conditional runtime functions have 10-16 bit/float support for YUV, PlanarRGB and 16 bit packed RGB formats.

Since RGB is also available as a planar colorspace, the plane statistics functions logically were expanded.

New functions
• AverageR, AverageG AverageB like AverageLuma
• RDifference, GDifference, BDifference like LumaDifference(clip1, clip2)
• RDifferenceFromPrevious, GDifferenceFromPrevious, BDifferenceFromPrevious
• RDifferenceToNext, GDifferenceToNext, BDifferenceToNext
• RPlaneMin, GPlaneMin BPlaneMin like YPlaneMin(clip [, float threshold = 0, int offset = 0])
• RPlaneMax, GPlaneMax BPlaneMax like YPlaneMax(clip [, float threshold = 0, int offset = 0])
• RPlaneMinMaxDifference, GPlaneMinMaxDifference BPlaneMinMaxDifference like YPlaneMinMaxDifference(clip [, float threshold = 0, int offset = 0])
• RPlaneMedian, GPlaneMedian, BPlaneMedian like YPlaneMedian(clip [, int offset = 0])

For float colorspaces the Min, Max, MinMaxDifference and Median functions populate pixel counts for the internal statistics at a 16 bit resolution internally.

[Tweak]
See original doc: http://avisynth.nl/index.php/Tweak
The original 8 bit tweak worked with internal LUT both for luma and chroma conversion.
Chroma LUT requires 2D LUT table, thus only implemented for 10 bit clips for memory reasons.
Luma LUT is working at 16 bits (1D table)

Above these limits the calculations are realtime, and done pixel by pixel.
You can use a new parameter to force ignoring LUT usage (calculate each pixel on-the-fly)
For this purpose use realcalc=true.

[MaskHS]
Works for 10-16bit,float.

MaskHS uses LUT for 10/12 bits. Above this (no memory for fast LUTs) the calculation is done realtime for each.
To override LUT for 10-12 bits use new parameter realcalc=true

[ColorKeyMask]:
Works for RGB64, Planar RGBA 8-16,float.
ColorKeyMask color and tolerance parameters are the same as for 8 bit RGB32.
Internally they are automatically scaled to the current bit-depth

[ResetMask]
New extension.
Accepts parameter (Mask=xxx) which is used for setting the alpha plane for a given value.
Default value for Mask is 255 for RGB32, 65535 for RGB64 and full 16 bit, 1.0 for float.
For 10-12-14 bit it is set to 1023, 4095 and 16383 respectively.

Parameter type is float, it can be applied to the alpha plane of a float-type YUVA or Planar RGBA clip.

[Layer]
Layer() now works for RGB64.
Original documentation: http://avisynth.nl/index.php/Layer

By avisynth documentation: for full strength Level=257 (default) should be given.
For RGB64 this magic number is Level=65537 (this is the default when RGB64 is used)

Sample:
lsmashvideosource("test.mp4", format="YUV420P8")
x=last
x = x.Spline16Resize(800,250).ColorYUV(levels="TV->PC")
x = x.ConvertToRGB32()

transparency0_255 = 128 # ResetMask's new parameter. Also helps testing :)
x2 = ColorBars().ConvertToRGB32().ResetMask(transparency0_255)

x_64 = x.ConvertToRGB32().ConvertBits(16)
x2_64 = ColorBars().ConvertToRGB32().ConvertBits(16).ResetMask(transparency0_255 / 255.0 * 65535.0 )

#For pixel-wise transparency information the alpha channel of an RGB32 overlay_clip is used as a mask.

op = "darken" # subtract lighten darken mul fast
level=257         # 0..257
level64=65537     # 0..65537
threshold=128                   # 0..255   Changes the transition point of op = "darken", "lighten."
threshold64=threshold*65535/255 # 0..65535 Changes the transition point of op = "darken", "lighten."
use_chroma = true
rgb32=Layer(x,x2,op=op,level=level,x=0,y=0,threshold=threshold,use_chroma=use_chroma )
rgb64=Layer(x_64,x2_64,op=op,level=level64,x=0,y=0,threshold=threshold64,use_chroma=use_chroma ).ConvertBits(8)
StackVertical(rgb32, rgb64)

[Levels]
Levels: 10-16 bit support for YUV(A), PlanarRGB(A), 16 bits for RGB48/64
No float support yet

Level values are not scaled, they are accepted as-is for 8+ bit depths

Test scripts for Levels
# Gamma, ranges (YUV):
x=ConvertToYUV420()
dither=true
coring=true
gamma=2.2
output_low = 55
output_high = 198
clip8 = x.Levels(0, gamma, 255, output_low, output_high , dither=dither, coring=coring)
clip10 = x.ConvertBits(10).Levels(0,gamma,1023,output_low *4,(output_high +1)*4 - 1, dither=dither, coring=coring)
clip16 = x.ConvertBits(16).Levels(0,gamma,65535,output_low *256,(output_high+1) *256 -1,dither=dither, coring=coring)
stackvertical(clip8.Histogram("levels"), clip10.ConvertBits(8).Histogram("levels"), Clip16.ConvertBits(8).Histogram("levels"))

# packed RGB 32/64
xx = ConvertToRGB32()
dither=false
coring=false
gamma=1.4
clip8 = xx.Levels(0, gamma, 255, 0, 255, dither=dither, coring=coring)
clip16 = xx.ConvertBits(16).Levels(0,gamma,65535,0,65535,dither=dither, coring=coring)
stackvertical(clip8.ConvertToYUV444().Histogram("levels"), Clip16.ConvertBits(8).ConvertToYUV444().Histogram("levels"))

[ColorYUV]
Now it works for 10-16 bit clips

• Slightly modified "demo" mode when using ColorYUV(showyuv=true)

#old: draws YV12 with 16-239 U/V image (448x448)
#new: draws YV12 with 16-240 U/V image (450x450)

• New options for "demo" mode when using ColorYUV(showyuv=true)
New parameter: bool showyuv_fullrange.
Description: Draws YV12 with 0-255 U/V image (512x512)
Usage: ColorYUV(showyuv=true, showyuv_fullrange=true)

New parameter: bits=10,12,14,16
Result clip is the given bit depth for YUV420Pxx format.
As image size is limited (for 10 bits the range 64-963 requires big image size), color resolution is 10 bits maximum.
#This sample draws YUV420P10 with 64-963 U/V image
ColorYUV(showyuv=true, bits=10).Info()

Luma steps are 16-235-16../0-255-0.. up to 0-65535-0... when bits=16

• Additional infos for ColorYUV

- Fixed an uninitialized internal variable regarding pc<->tv conversion,
  resulting in clips sometimes were expanding to pc range when it wasn't asked.
- No parameter scaling needed for high bit depth clips.
  For 8+ bit clips parameter ranges are the same as for the 8 bit clips.
  They will be scaled properly for 10-16 bitdepths.
  e.g. off_u=-20 will be converted to -204 for 10 bits, -20256 for 16 bits
- ColorYUV uses 8-10-12-14-16 bit lut.
- ColorYUV is not available for 32 bit (float) clips at the moment

[Other things you may have not known]
Source filters are automatically detected, specifying MT_SERIALIZED is not necessary for them.

[Known issues/things]
GRunT in MT modes (Avs+ specific)
[done: v2502] Overlay blend with fully transparent mask is incorrect, overlaying pixel=0 becomes 1, overlaying pixel=255 becomes 254.
[done: v2676-] Float-type clips: chroma should be zero based: +/-0.5 instead of 0..1
