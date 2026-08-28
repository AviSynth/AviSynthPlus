Changes from 3.7.5 to 3.7.6
---------------------------

Additions, changes
~~~~~~~~~~~~~~~~~~
- Added new pixel format constants

  * YUV(A) 4:4:0(:α) 8~32bits
  * YV411 aliased to more standard YUV411 naming
  * YUV 4:1:1 10~32bits
  * YUVA 4:1:1:α 8~32bits
  * Y plus alpha 4:0:0:α 8~32bits
  * YUV9 (added in 2.6 but not used except in the constants list) aliased to more standard YUV410 naming
  * YUV 4:1:0 10~32bits
  * YUVA 4:1:0:α 8~32bits

- New array syntax: 

  * dictionary-style ArraySet (replaces or appends).
  * dictionary-style ArrayDel (deletes by key, no-op if not found).
  * dictionary-style ArrayIndexOf (returns the actual index of the dictionary-style array search).

  See :doc:`Arrays <./script_ref/script_ref_arrays>`.
- Added ``SetFilterProp``, ``GetFilterProps``, and ``SetFilterPropPassthrough``:

  * ``SetFilterProp`` — automatically inject a frame property on the output of a named filter
    every time it is instantiated, without modifying the script.  Rules are typically placed in
    an auto-loaded ``.avsi`` file.  The property value may be a static int/float/string, a
    function object evaluated per frame (e.g. to derive ``_Matrix`` from output resolution), or
    captured from the filter's own call argument (``undefined()`` capture mode).
  * ``SetFilterProp`` conditional form — inject a property only when a specific named call
    argument equals a given match value or any element of an alias array.  String comparison is
    case-insensitive.  Both positional and named call sites are matched.
  * ``SetFilterPropPassthrough`` — compatibility shim for old filters that predate frame-property
    support (e.g. ``ColorMatrix``).  Copies all input frame properties to the filter's output;
    self-healing: becomes a no-op per frame once the plugin is updated to use ``NewVideoFrameP``.
  * ``GetFilterProps`` — returns the registered rule table as a JSON string for diagnostics.

  Addresses `issue #393 <https://github.com/AviSynth/AviSynthPlus/issues/393>`_ and more.
  See :doc:`SetFilterProp <./corefilters/setfilterprop>`.
- Bump year to 2026
- Overlay: "add" and "subtract" direct RGB mode
- Overlay: "add" and "subtract" supports 32-bit float.
- Added utf8 parameter to AddAutoLoadDir
- Added utf8 parameter to ListAutoLoadDirs
- Added utf8 parameter to LoadPlugin
- Added utf8 parameter to DumpFilterGraph
- "Info": Optimize AVX512 features display, group features, make a bit more compact
- "Info": add L2 cache size display
- "SetMaxCPU": add "avx512base" and "avx512fast" options to enable/disable AVX512 grouped features.
  see :ref:`SetMaxCPU <setmaxcpu>` . Users of base-only AVX512 CPUs can enable Avisynth-optimizations
  with SetMaxCPU("avx512base+").
- ARM64 (aarch64) area:

  * "Info": add ARMV8-A features display (NEON, DOTPROD, SVE2)

  * Add ArmV8-A cpu feature detection (NEON, DOTPROD, SVE2) on ARM64 Windows/Linux/macOS builds.
    On Windows, only up-to DOTPROD can be detected due to OS limitations.
  * New CPU flags in ``cpuid.h`` and ``avisynth_c.h``: CPUF_ARM_NEON, CPUF_ARM_DOTPROD, CPUF_ARM_SVE2
  * "SetMaxCPU": add "neon", "dotprod", "sve2" options to enable/disable ARM64 (aarch64) features.
- "ConvertToPlanarRGB(A)": ``bits`` parameter: on-the-fly bit-depth conversions to YUV->RGB conversion.
  See :doc:`ConvertToPlanarRGB <./corefilters/convert>`.
- ``ConvertToPlanarRGB(A)``: added ``quality`` parameter: forces 32-bit float
  internal processing instead of S18.13 fixed-point arithmetic when converting
  from YUV, regardless of source or target bit-depth. See :doc:`ConvertToPlanarRGB <./corefilters/convert>`.
- "ConvertToYUV(A)xxx" and legacy 8 bit name versions: ``bits`` parameter: on-the-fly bit-depth conversions to RGB->YUV conversion.
  See :doc:`ConvertToYUV444 <./corefilters/convert>`.
- ``ConvertToYUV(A)xxx`` and legacy 8 bit name versions: added ``quality`` parameter: forces 32-bit float
  internal processing instead of S18.13 fixed-point arithmetic when converting
  to YUV, regardless of source or target bit-depth. See :doc:`ConvertToYUV444 <./corefilters/convert>`.
- "ResetMask": add parameter float "opacity"
- "AddAlphaPlane": add parameter float "opacity"
- "Layer": YUY2 is handled as YV16 (lessen source code bloat)
- "Histogram" Color and Color2 mode additions and fixes:

  * Added ``matrix`` parameter (Color and Color2): specifies the YUV matrix for
    chroma interpretation (BT.601, BT.709, BT.2020, etc.), following the same
    convention as the ``ConvertToYUV`` family. If not set, the matrix is
    read from the clip's ``_Matrix`` and ``_ColorRange`` frame properties.
  * Added ``graticule`` string parameter (Color and Color2): controls the
    danger zone shading (Color) or valid chroma boundary square (Color2).
    ``"on"`` (default) always draws it, preserving pre-3.7.6 behavior;
    ``"off"`` never draws it; ``"auto"`` draws it only for limited-range
    clips and suppresses it for full-range clips.
  * Added ``circle`` parameter (Color and Color2, default false for Color,
    default true for Color2): draws the hue circle with 15° tick marks.
    In Color2 mode this was previously always drawn; it can now be disabled.
  * Added ``targets`` parameter (Color and Color2, default false): draws
    target boxes at the six 75%-amplitude ColorBars Cb/Cr positions
    (Yellow, Cyan, Green, Magenta, Red, Blue). Positions are computed from
    ground-truth linear RGB values through the active matrix, giving
    accurate coordinates at all bit depths.
  * Added ``targets100`` parameter (Color and Color2, default false): draws
    target boxes at the same six ColorBars Cb/Cr positions as ``targets``,
    but placed at 100% luma level instead of 75%.
  * Added ``axes`` parameter (Color and Color2, default false): draws
    horizontal and vertical crosshair lines through the vectorscope center.
  * Added ``iq`` parameter (Color and Color2, default false): draws target
    boxes for the NTSC −I, +I and +Q chroma-phase references, using the
    luma-corrected broadcast convention (Y raised until the most negative
    RGB component reaches Code 0, avoiding illegal RGB values).
  * Added ``iq_lines`` parameter (Color and Color2, default false): draws
    radial lines at the fixed NTSC subcarrier phase angles of 33°, 123°,
    213° and 303°, marking the I and Q chroma axes.
  * Fix: "Color" and "Color2" modes: copy alpha channel from source for
    alpha-carrying formats (YUVA, RGBPA, RGB32, RGB64); initialize alpha
    to zero in the histogram panel area.
  * Fix: accurate pixel positioning and scaling throughout the histogram
    panel, limited/full range aware. Marker positions use the same
    conversion path as the vectorscope signal dots, ensuring exact
    alignment at any bit depth and color range.
  * Vectorscope is now fully matrix-aware: all overlay marker positions are
    computed from ground-truth linear RGB values converted through the active
    YUV matrix, giving accurate Cb/Cr coordinates at all bit depths including
    32-bit float.
- "ConvertToYUY2": rewritten to route all conversions through YV16 as
  intermediate format, using the full ``ConvertToPlanarGeneric`` infrastructure.
  ``ChromaOutPlacement`` parameter added (was missing, present in ConvertToYV16).
  All source formats (YV12, YUV420/422/444, planar/packed RGB, high bit depth)
  are now handled correctly. ``bits`` parameter accepted; must be 8 if specified.
  See :doc:`Convert <./corefilters/convert>`.
- "ConvertBackToYUY2": kept for backward compatibility; now forwards to
  ``ConvertToYUY2``. The pre-2.5 left-pixel-only chroma hack is no longer
  needed or applied; the YV16 lossless repack path avoids chroma resampling
  loss entirely for roundtrip workflows.
- "ConvertToYV12" (legacy 8-bit name): the YUY2 fast-path shortcut is removed;
  now routes directly through ``ConvertToPlanarGeneric::CreateYUV420``.
  ``bits=`` parameter semantics corrected: the legacy 8-bit-named functions
  (``ConvertToYV12``, ``ConvertToYV16``, ``ConvertToYV24``) now check the
  *target* bit depth rather than the source bit depth, allowing high-depth
  sources when ``bits=8`` is explicitly specified.
- 8 bit packed RGB formats are converted to planar RGB before 444 conversion.
  Stop using direct rgb-yv24 conversions (16 bits were already converted for long time).
- Add :doc:`ColorBarsUHD <./corefilters/colorbarsuhd>` ITU-R BT.2111-3 (05/2025) with three signal variants:
  HLG narrow range (mode=0), PQ narrow range (mode=1) and PQ full range (mode=2)
- "Layer": add ``"mulovr"`` op — Overlay-style multiply for YUV(A) formats only (RGB raises an error).
  The overlay luma (Y) plane drives darkening of all base planes: dark overlay Y pulls base luma toward
  black and simultaneously desaturates base chroma toward neutral (128 for 8-bit integer, 0.0 for float).
  Bright overlay Y (max value) leaves the base unchanged.  Matches ``Overlay(mode="multiply")`` within
  ±1 LSB for all bit depths and all chroma placements.  Supports 8–16-bit integer and 32-bit float; all
  YUV subsampling (Y, 4:1:1, 4:2:0, 4:2:2, 4:4:4); alpha-aware (YUVA) and non-alpha sources.
  Uses a two-pass structure like "lighten"/"darken": pass 1 processes chroma using spatially-averaged
  overlay Y; pass 2 processes luma at full resolution.  The ``"placement"`` parameter is respected for
  correct 4:2:0 / 4:2:2 mask downsampling.
  See :doc:`Layer <./corefilters/layer>`.
- ``ShowFrameNumber``, ``ShowSMPTE``, ``ShowTime``, ``ShowCRC32``, ``Info``, ``Compare``: add ``gdi`` bool parameter.

  * When ``gdi=true`` (default on Windows with GDI), the Windows GDI Antialiaser is used for antialiased
    text rendering (existing behavior). When ``gdi=false``, the built-in bitmap font (Terminus) is used
    instead: faster, cross-platform compatible, and the only option on non-Windows builds.
  * ``font`` default is now ``gdi``-dependent: "Arial" / "Courier New" for GDI, "Terminus" otherwise.
  * ``bold`` default is now ``gdi``-dependent: ``true`` for GDI, ``false`` for the bitmap font path.

  See :doc:`showframes <./corefilters/showframes>`, :doc:`info <./corefilters/info>`, and :doc:`compare <./corefilters/compare>`.

- ``ShowCRC32``: add ``channels``, ``mode``, and ``showmode`` parameters.

  * ``channels`` — selects which planes to checksum using their initial letters (Y, U, V, A for YUV;
    R, G, B, A for RGB), defaulting to all planes.  Unrecognised letters are silently ignored.
  * ``mode`` — ``0`` (default): one combined CRC32 over all selected planes; ``1``: separate CRC32
    per plane displayed as ``"Y:XXXXXXXX U:XXXXXXXX ..."``.
  * ``showmode`` — ``0`` (default): display text only; ``1``: display text and store result as frame
    property ``"ShowCRC32"`` (``int64`` array — one element for ``mode=0``, one per active plane for
    ``mode=1``); ``2``: store frame property only, no text drawn.  Values are unsigned 32-bit integers
    stored as ``int64``.
  * For packed formats (YUY2, RGB24/32/48/64) with default parameters the raw interleaved buffer
    is hashed directly, preserving backward compatibility.
  * Planar RGB(A): planes are always processed in R, G, B, A logical order.

  See :doc:`showframes <./corefilters/showframes>`.
- "Layer": add ``"top_left"`` option for the ``"placement"`` parameter — HEVC/AV1 left+top co-sited
  chroma (point-sample, fastest).  Affects ``"mul"``, ``"add"``, ``"subtract"``, ``"lighten"``,
  ``"darken"``, and ``"mulovr"`` modes with 4:2:0 / 4:2:2 sources.
- "Overlay" ``"blend"`` mode: add ``"placement"`` parameter for correct luma-mask-to-chroma
  downsampling in 4:2:0 and 4:2:2 clips.  Values: ``"mpeg2"`` (default), ``"mpeg1"``, ``"top_left"``.
- ``Subtitle``: add ``placement`` string parameter — chroma location hint for subsampled
  YUV formats (4:2:0, 4:2:2, 4:1:1).  When ``gdi=true`` all three siting modes are
  supported: ``"MPEG2"`` / ``"left"`` (default), ``"MPEG1"`` / ``"center"``, and
  ``"top_left"`` (UHD 4:2:0 / 4:2:2 co-sited chroma).  The default is read from the
  ``_ChromaLocation`` frame property, falling back to ``"left"``.
  When ``gdi=false`` only ``"left"`` and ``"center"`` are implemented (same as ``Text``).
  See :doc:`Subtitle / Text <./corefilters/subtitle>`.
- ``Subtitle``: add ``gdi`` bool parameter.  When ``false``, Subtitle uses the built-in
  bitmap font (Terminus) instead of Windows GDI rendering — the same path as the
  ``Text`` filter.  Faster and cross-platform compatible; ``placement`` is then limited
  to ``"left"`` / ``"center"``.  Default: ``true``.
  See :doc:`Subtitle / Text <./corefilters/subtitle>`.
- ``Text``: add ``gdi`` bool parameter (accepted, has no effect; present for API
  compatibility with ``Subtitle`` — on non-Windows, ``Subtitle`` is aliased to ``Text``
  so every ``Subtitle`` parameter must exist in ``Text``).  Default: ``false``.
  See :doc:`Subtitle / Text <./corefilters/subtitle>`.


Build environment, Interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Bump to interface version 13 for new color constants
- introduce ``AVS_RESTRICT`` to ``avs/config.h`` (compiler invariant c++ ``__restrict``)
- AVX512: CMake to recognize ``*_avx512b.*`` and ``*_avx512.*`` file pattern, add compiler specific AVX512 
  compile flags accordingly (AVX512 Base and Ice Lake extensions)
- AVX512 support by conditional define.
  Define `INTEL_INTRINSICS_AVX512` if avx512 modules are enabled and compiler supports it.
  For MSVC,AVX512 support enabled only from MSVC 2019 16.2 (19.22) or newer.
- add ``.editorconfig``, update .gitignore to include the new .slnx format of Visual Studio 2026
- v12 interface: Global Lock support (https://github.com/AviSynth/AviSynthPlus/issues/444), 
  mainly for plugins using common fftw3 library:

  * ``env->AcquireGlobalLock``, ``env->ReleaseGlobalLock`` (C++),
  * ``avs_acquire_global_lock``, ``avs_release_global_lock`` (C)

  see :ref:`global lock support<cplusplus_acquiregloballock>`
- v12 interface: ApplyMessageEx supporting utf8 parameter.
  see :ref:`ApplyMessageEx<cplusplus_applymessage>`
- v12 interface: inform plugins about the effective thread count after Prefetch()
  via cache hints:

  * ``CachePolicyHint::CACHE_INFORM_NUM_THREADS`` (C++)
  * ``AVS_CACHE_INFORM_NUM_THREADS`` (C)

  See :ref:`SetCacheHints<cplusplus_setcachehints>` .
- the internal IScriptEnvironment2 methods AddAutoLoadDir and ListAutoLoadDirs explicitely
  work in UTF-8.
- New CPU flags: ``cpuid.h and ``avisynth_c.h``
  - added AVX512 group feature flags CPUF_AVX512_BASE and CPUF_AVX512_FAST (Ice Lake, usable AVX-512 since that point).
  - added many new AVX512 individual feature flags
  - added ARM64 feature flags CPUF_ARM_NEON, CPUF_ARM_DOTPROD, CPUF_ARM_SVE2
  - CPUF_xxxxx flags are now 64 bit, replace enum with constexpr.
- CMakeLists.txt: avx512 compile flag support for gcc/clang ("base" and "fast", latter is Ice Lake-like feature set).
- V12 interface: ``GetCPUFlagsEx`` returning 64 bit flags (too many AVX512 subfeatures to fit in 32 bit).
  C interface: ``avs_get_cpu_flags_ex``.
  see :ref:`GetCPUFlagsEx<cplusplus_getcpuflagsex>` and :ref:`GetCPUFlags<cplusplus_getcpuflags>`
- V12 interface: L2 cache size query support. New entry in ``AvsEnvProperty``: ``AEP_CACHESIZE_L2`` (C++), 
  ``AVS_AEP_CACHESIZE_L2`` (C) to query L2 cache size in bytes with ``IScriptEnvironment->GetEnvProperty()``. 
  x86/x64 architecture only for now. See :ref:`AvsEnvProperty<cplusplus_getenvproperty>` .

- Refactor CMakeLists.txt: 

  * Correct default of ``ENABLE_INTEL_SIMD`` for cross-compiling scenarios (e.g. ``ARM64`` target on ``x86_64`` host)
    Old logic relied on the host processor: ``${CMAKE_SYSTEM_PROCESSOR}``

  * Add back option to compile ``ARM64`` builds with Visual Studio on Windows. On VS2026 even clangcl (LLVM) is supported 
    out-of-box for ARM64 platform, in an easily cross-compilable way from an x64 machine.
  * VDubFilter: allow building on Windows only x86/x64 targets (and not for ARM64).
  * Fix LLVM/clangcl/Intel ICX compile warning: ``'WIN32' macro redefined as "#define WIN32 /D_WINDOWS /W3 /GR /EHsc 1 "``,
    when CMake injects a command-line macro wrongly and thus redefines WIN32 .
    The fix: converts global ``add_definitions("/D ...")`` and other option string magics into per-target ``target_compile_definitions()``
    and ``target_compile_options()``. Thus removing the accidental injection of ``${CMAKE_CXX_FLAGS}`` 
    into ``add_compile_options()``, and prevents the WIN32 macro redefinition.
- CMakeLists: add option to pre-supply external DevIL library path and include directory
  (``-DDEVIL_LIBRARY`` / ``-DDEVIL_INCLUDE_DIR``).
- CMakeLists: apply compiler parameter settings (warnings, compile flags) to all plugin
  sub-projects in addition to the core library.
- Fix/optimize: ``avs/minmax.h`` content now is force-inlined, since MSVC placed a function call for them! 
  (checked with actual disassembly).
  Plus: wrap min/max/clamp in an anonymous namespace to prevent a theoretical translation unit mismatch across archs
  (e.g. when SSE2/AVX2/AVX512 specific ``.cpp`` files are built with different per-file compiler flags). 
  Part of our public headers, but no interface change was done.


Bugfixes
~~~~~~~~
- Fix: "ConvertToYUY2" / "ConvertToYV12": YV12<->YUY2 interlaced conversion
  used asymmetric 0.75/0.25 chroma interpolation coefficients instead of the
  MPEG-2 specified 7/8,1/8 and 3/8,5/8 coefficients, introducing opposite-
  direction vertical chroma shifts in top and bottom fields. First diagnosed
  by Gavino in 2009; now resolved by routing through ``ConvertToPlanarGeneric``.
- Fix: "ConvertToYUY2" / "ConvertToYV12": progressive YV12<->YUY2 conversion
  used asymmetric 0.75/0.25 averaging (quarter-pixel offset) instead of the
  correct 0.5/0.5 midpoint average for left-sited chroma, introducing a
  1/4-pixel vertical chroma shift.
- Fix: "ConvertToYUY2": ``_ChromaLocation``, ``_Matrix`` and ``_ColorRange``
  frame properties were not read from YV12 source frames and not written to
  YUY2 output frames in the legacy direct conversion path.
- Fix: "ConvertToYUY2": SSE2 interlaced upsampling used wrong weighting
  direction for the lower line of each field pair (75%/25% toward current
  instead of 25%/75% toward next), differing from the C reference implementation.
- Fix: memory leak and stability issue in frame recycling for static-frame sources (ColorBars, BlankClip):
  subframes created by Subframe/MakePropertyWritable accumulated without bound in permanently-live
  VFB buckets, and a race in frame property cleanup could cause access violations under multi-threaded use.
- Fix: LruCache self-tuning cache resize: raised the ghost-hit threshold from ``ghosted > 0`` to
  ``ghosted > 1``, preventing unbounded cache growth during backward seeking (Issue #379) and
  Bob()/SeparateFields access patterns (Issue #270). Frames evicted only once no longer trigger
  a resize; an undersized cache still grows once the same frame has been evicted twice.
- Fix: "Histogram" Color2 mode to copy alpha channel from source for alpha-carrying formats
  (YUVA, RGBPA, RGB32, RGB64); initialize alpha to zero in the histogram panel area.
  (Was: garbage)
- Fix: C-only vertical resampling code added more rounding than needed
  (regression since pre-3.7.5 20250427).
- Fix: "Invert": corrected chroma inversion to pivot around signed 0 instead
  of XORing with max_pixel_value.
- Fix: YUV->RGB limited range matrix accuracy for 10-16 bits, plus use a symmetric rounding in matrix 
  coefficient's integer approximation.
- Fix: inaccurate ColorBars 10+ bit values. Now they are derived from the 32-bit float 
  RGB definitions instead of upscaling a 8 bit precalculated YUV value. -I and +Q are still kept at
  legacy Avisynth values.
- Fix: inaccurate ColorBarsHD 10+ bit values. Now they are derived from the 32-bit float 
  RGB definitions instead of upscaling a 8 bit precalculated YUV value. Add 100% White after 
  Ramp section.
- Fix: GreyScale + SSE2 + RGB32 + matrix="RGB" overflow. 
  Rare usage; "RGB" matrix (Identity) uses a 1.0 coefficient which exceeds the signed 16-bit 
  SIMD limit of 32767 at 15-bit precision. Added bounds checking to fallback to C-code for any 
  coefficients >= 1.0 or < −1.0.
- Fix #448: Resolved an issue where MT_MULTI_INSTANCE filters using relative paths 
  (e.g. "video.mp4" or "../image.png") failed under Prefetch() when used in imported 
  scripts from different directories. The problem occurred because new thread instances did 
  not inherit the original working directory, causing path resolution to fail.
  Now, the current directory is captured at filter instantiation and passed to worker threads, 
  ensuring consistent path resolution.
- Fix #456: "Reverse" corrupts 24-bit audio (https://github.com/AviSynth/AviSynthPlus/issues/456)
- Fix BDF font rendering when it contains variable width characters like mixed Latin and CJK. 
  Preparing feature request #446 (https://github.com/AviSynth/AviSynthPlus/issues/446)
- Fix #462: Report: "AviSynth scripts don't work in a folder with a Unicode name."
  Plugin autoload folders are internally stored in UTF-8, regardless of which Windows ANSI codepage is set.

  * Folder names used in macros in AddAutoLoadDir (SCRIPTDIR, MAINSCRIPTDIR, PROGRAMDIR) and no longer 
    restricted to contain ANSI-only characters
  * Registry-backed macros that can contain plugin folder paths (USER_PLUS_PLUGINS, MACHINE_PLUS_PLUGINS, 
    USER_CLASSIC_PLUGINS, MACHINE_CLASSIC_PLUGINS) are read in Unicode friendly way as well. 
- Fix: Not existing registry entries won't appear as a macro string in auto-load path.
  E.g. Avisynth would automatically add ``USER_CLASSIC_PLUGINS`` at the beginning, but if no such entry
  exist, it kept being in the folder list as ``<current_directory>\USER_CLASSIC_PLUGINS\``. 
  Now this false entry is removed.
- Fix: Overlay give proper error message if 32-bit float is not supported in that mode.
- Change video-framebuffer over-allocation from 16 to 64 bytes. Allocate 64 bytes more than needed for 
  video frame buffer in order to be able to read 64 bytes safely with AVX512 without risking access violation 
  on the last pixels of the frame.
- Fix: The `Animate()` function now explicitly clamps interpolated values to ensure they remain
  strictly between the start and end range. Due to the high precision of 64-bit `double` introduced
  in v3.7.5, intermediate calculations could slightly exceed the boundary (e.g., 360.00000000000006
  when interpolating from 0 to 360.0 in 564 steps), requiring this clamp to prevent out-of-range errors.
- Fix: ``Text`` filter crash when input contains a zero-length line (e.g. ``Text("\n")``).
- Fix: "Layer"/"Overlay" subsampled YUV chroma alignment:

  * "Layer": x/y offsets that are not aligned to the chroma grid (e.g. odd x for 4:2:0/4:2:2)
    are now accepted as-is, matching "Overlay" behaviour.  Previously they were silently snapped
    to the nearest aligned value; that snap also incorrectly shifted the luma start position,
    causing a one-pixel luma displacement that did not occur in Overlay.
  * "Layer"/"Overlay": the last chroma column/row of the blend region is now correctly processed
    when the overlay position is not chroma-grid-aligned.  A ceiling formula replaces the old
    floor division, which caused the rightmost column or bottom row of affected chroma pixels
    to be skipped.
- Fix: RGB->YUV matrix Y offset: missing unity correction when converting narrow range RGB 
  to full range Y.
- Fix #499: "RGBAdjust": an uninitialized conditional parameter caused the existing
  ``rgbadjust_`` branch to always run regardless of settings. Also fixed a memory leak
  when freeing LUT tables.
- Fix: "TemporalSoften" 16-bit scalar (C) rounding.
- Fix: AVX2 YV24 (4:4:4) source alpha packing.
- Fix: "Expr": a typo checked the output clip's ``IsYUVA()`` instead of the source
  clip's when selecting per-source plane enums, which could read the wrong plane for
  certain mixed colorspace expressions.
- Fix: static-analysis findings (memory leaks / buffer overruns, not user visible), 
  Tagged as AVSP001-AVSP018: ``ColorBars`` audio array leak,
  ``FastReadStream`` ``pHeaders`` array leak, ``ConvertAudio`` temp buffer allocation
  check, ``AVIReadHandler`` ``streamBuffer`` leak, the script parser's ``Spline`` work
  buffer (now RAII-managed), the ``TimeStretch`` plugin's buffer offset/``memcpy``
  undefined behavior, "Levels" ``use_lut``+dither buffer overrun, the script parser's
  ``arg_names`` array overrun, and a ``PluginManager`` constructor memory leak.
- Fix #505: "Expr" Vector-C relative-row addressing (``x[dx,dy]``) ignored the current row
  entirely, and additionally mis-scaled the row byte offset for 16-bit/float sources.
- Fix #506: "Compare" 32-bit SAD/SD accumulators (8-bit packed/planar, C and SSE2/ISSE paths)
  could overflow on large frames (e.g. 4K RGB32); now 64-bit unconditionally.
- Fix #507: "Compare" RGB24/RGB48 MAD/MD denominator divided by a hardcoded 4 (assumed
  4 channels/pixel) instead of the actual per-pixel component count (3).
- Fix #508: "Compare" uninitialized internal mask could give non-deterministic MAD/MD
  results for RGB48/RGB64.
- Fix #509: "ConvertBits" automatic preconversion for large ``dither_bits`` gaps (e.g.
  16-bit source with ``dither_bits=1``) silently dropped the dithering, falling back to
  a plain bit-depth expansion.
- Fix #510: "Layer" "Mul" raw multiply product was rounded down incorrectly, so
  near-max products came out one unit too low (e.g. 255*255 -> 254).
- Fix #511: "ConvertYUV444ToRGB"/"ConvertRGBToYUV444" left the synthesized alpha plane
  uninitialized when converting an alpha-less source to an alpha-carrying target.
- Fix #512: "Overlay" masked "add"/"subtract"/"darken"/"lighten"/"difference"/"exclusion"/
  "softlight"/"hardlight" a fully-opaque mask did not reproduce the same result as omitting the mask.
  As a side effect, also added finer opacity-granularity over 8 bits at integer formats.
- Fix: ``ArrayIns``/``ArraySet``/``ArrayDel``: bounds check the index parameter(s)
  (preventing Access Violation).


Optimizations
~~~~~~~~~~~~~
- "Layer" YUVA/YUV/RGBP "add", "subtract", "mul", "darken", "lighten": refactor
  chroma placement calculation and function dispatchers; main algorithm SIMD
  vectorization is now possible for non-444 chroma placements by precalculating
  a mask for the actual row. Add AVX2 path (LLVM/clangcl recommended).
- "Overlay" Blend: improved speed while keeping accuracy; use float arithmetic
  only where strictly needed.
- "Invert": planar formats no longer pre-copy all planes from source before
  conversion; only planes that are unchanged are copied, avoiding unnecessary work.
- TurnLeft, TurnRight: AVX2 support (1,5-3x speed on i7-11700 compared to SSE2 version)
- Turn180 AVX2 support (very slight speed gain)
- Resamplers: 

  * introduce a SIMD-like C header (avs_simd_c.h) for smart auto-vectorizing compilers.
  * restore vertical float performance (3.7.4 was slower than 3.7.3) + SSE2 special optimization
  * further optimize verticals, use ``AVS_RESTRICT``
  * (quicker RGB32/64 horizontal on AVX2 since TurnRigh/Left was optimized - packed RGB H-resize = TurnLeft-V-Resize-TurnRight)
  * optimize SSSE3 and AVX2 horizontal resampler for 32-bit float for small (<=4) kernel sizes
  * optimize 32-bit float vertical avx2
  * add AVX512 code path 
  
    - 32-bit float resamplers, verticals; horizontals up to kernel size 16.
    - 8-16-bit horizontal resamplers, for kernel size <= 16 and specific ratios; speed gain up to 300%+ (DTL2020)!!
    - 8-16-bit vertical resamplers
  * (Work In Progress) unify horizontal and vertical plane processing flow

- add NEON optimizations for ARM64 (aarch64) for TurnLeft/TurnRight/Turn180.
  (First aarch64 code in Avisynth)
- add NEON optimizatons for Overlay blend
- (filter graph) avoid MTGuard and CacheGuard creation if a filter returns one of its clip parameter unaltered.
- Add some avx2 stuff to Invert (no really gain, filter is too simple) and some Layer subfilter.
- "Layer", "Overlay", "Merge": unify three separate masked-merge kernel families into shared
  ``masked_merge_{c,sse41,avx2,neon}`` templates parameterized on ``MaskMode`` and pixel type.
  SIMD precalculation of chroma-placement-corrected mask rows (SSE4.1, AVX2, NEON) for all
  subsampled formats (4:2:0, 4:2:2, 4:1:1) enables full SIMD throughput in the main blend loop
  for non-444 sources.  Accuracy: integer masked merge uses magic-number division for exact
  per-element results at all bit depths; plain weighted merge uses the same 15-bit arithmetic
  in C as in SIMD, eliminating C/SIMD parity issues.  AVX2 flat-weight float weighted merge added.
- "Merge" ``AveragePlane`` / ``MaskedMerge``: add AVX2 float weighted-merge path; remove obsolete
  iSSE code; make memory loads unaligned so the routine is safely reusable from Overlay/Layer.
- "Layer": remove legacy MMX/iSSE (x86-only) code paths; replace hand-rolled min/max with
  ``std::min`` / ``std::max``.
- "Invert": add proper AVX2 and SSE2 SIMD paths for planar luma and chroma.
- ``ExtractX`` (``ExtractR/G/B/A/Y/U/V``): handle packed RGB formats directly, avoiding a
  PlanarRGB(A) conversion round-trip.
- "ConvertBits": restructure integer-to-integer depth-reducing C loop to be more
  auto-vectorization-friendly.
- ``PlanarRGB(A)`` → ``RGB32`` / ``RGB64``: add AVX2 conversion path.
- ``Subtitle`` Antialiaser (Windows GDI path): ``GetAlphaRect()`` — the per-pixel
  alpha/color weight computation from the 8×-supersampled GDI DIB — now has SSE4.1,
  AVX2, and AVX512 SIMD implementations in addition to the scalar fallback; best tier
  is selected once at construction time from cpu flags.
- ``Subtitle`` Antialiaser: internal mask buffer refactored from AoS (interleaved
  alpha/Y/U/V per pixel) to row-interleaved SoA layout, enabling correct
  chroma-placement-aware UV compositing.  Each UV output pixel is derived by spatially
  downsampling the luma-resolution mask row(s) via ``prepare_effective_mask_for_row``
  (same family as Overlay/Layer); eight MaskModes cover all subsampling ratios
  (4:4:4, 4:2:2, 4:2:0, 4:1:1) × siting variants (MPEG2/MPEG1/top_left); SIMD rowprep
  dispatch (AVX2 / SSE4.1 / scalar) is selected at construction.

Documentation
~~~~~~~~~~~~~
- Build on Raspbian, Raspberry Pi 5 and llvm/gcc :doc:`This page <./contributing/posix>` 
  describes linux builds process.
- Add to :ref:`SIMD in ARM64 (aarch64) section <aarch64_simd_tiers>`
- Extend ``env->Allocate/Free`` see at :ref:`Allocate <cplusplus_allocate>`
- Interface V12 changes: see :ref:`API v12 changes<api_v12_whats_new>` for more details.
- Add folder macro description to AddAutoLoadPlugins
- Update :doc:`Overlay <./corefilters/overlay>`
- Update :ref:`SetCacheHints<cplusplus_setcachehints>` with ``CACHE_INFORM_NUM_THREADS``
- Update :ref:`GetCPUFlags<cplusplus_getcpuflags>`, add :ref:`GetCPUFlagsEx<cplusplus_getcpuflagsex>`
- Update :ref:`CPU Feature Flags<cplusplus_cpufeatureflags>` with AVX512 and ARM64 features
- Update :ref:`SetMaxCPU <setmaxcpu>` with AVX512 and ARM64 features
- Update :ref:`AvsEnvProperty<cplusplus_getenvproperty>` with L2 cache size entry
- Update Russian GPL notice in UTF-8 format
- Update :doc:`ConvertToPlanarRGB <./corefilters/convert>` with `"bits"` and "matrix" syntax `":same"`
- Update :doc:`ResetMask <./corefilters/mask>` with `"opacity"` and additional insights
- Update :doc:`AddAlphaPlane <./corefilters/mask>` with `"opacity"` and additional insights
- Update :doc:`Overlay <./corefilters/overlay>` with "add" and "subtract"
  direct RGB mode and 32-bit float support.
- Update :doc:`Layer <./corefilters/layer>` with "use_chroma" and opacity
  details, and YUY2/YV16 internal handling.
- Update :doc:`Histogram <./corefilters/histogram>` with new vectorscope parameters
- Update :doc:`ColorBars <./corefilters/colorbars>`
- Update :ref:`matrix syntax <matrix_parameter_syntax>`
- Update :doc:`Convert <./corefilters/convert>` with ``ConvertToYUY2``
  ``ChromaOutPlacement`` parameter and corrected chroma placement behavior.
- Update :doc:`Convert <./corefilters/convert>` with ``bits`` and ``quality`` parameters
- Update :doc:`Sampling <./advancedtopics/sampling>` with historical content notes
  on legacy ``YUY2`` handling.
- Add another Ubuntu->Windows DLL cross-compilation guide:
  See :ref:`Ubuntu->Windows mingw crosscompilation<compiling_avsplus_crosscompiling2>`
- Add :doc:`ColorBarsUHD <./corefilters/colorbarsuhd>`
- Add :doc:`SetFilterProp / SetFilterPropPassthrough <./corefilters/setfilterprop>`
- Add ``ShowCRC32`` documentation to :doc:`showframes <./corefilters/showframes>`
  (filter exists since 3.7.0; rst page was not updated at the time).
  Document new ``channels`` and ``mode`` parameters added in 3.7.6.
- Update :doc:`Layer <./corefilters/layer>` with ``"mulovr"`` mode, ``"top_left"`` placement
  option, and related chroma-placement refactoring notes.
- Update :doc:`Overlay <./corefilters/overlay>` with ``"placement"`` parameter for ``"blend"`` mode.
- Update :doc:`Arrays <./script_ref/script_ref_arrays>` with dictionary-style key lookup/set/delete
  (``ArrayGet``, ``ArraySet``, ``ArrayDel``) and the new ``ArrayIndexOf``.


Please report bugs at `github AviSynthPlus page`_ - or - `Doom9's AviSynth+
forum`_

$Date: 2026/08/27 14:15:00 $

.. _github AviSynthPlus page:
    https://github.com/AviSynth/AviSynthPlus
.. _Doom9's AviSynth+ forum:
    https://forum.doom9.org/showthread.php?t=181351
