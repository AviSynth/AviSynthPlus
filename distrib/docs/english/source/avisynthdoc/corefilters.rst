
Core Filters
============

Introduction
------------

The available (internal) filters are listed here and divided into categories.
A short description is added, including the supported color formats (and
sample types for the audio filters). There are some functions which combine
two or more clips in different ways. How the video content is calculated is
described for each function, but :doc:`here is a summary which explains which
properties that the resulting clip will have <filters_mult_input_clips>`.

.. _Media file filters:

Media file filters
------------------

These filters can be used to read or write media files. Usually they produce
source clips for processing. See debug filters for non-file source filters.

- :doc:`AVISource / OpenDMLSource / AVIFileSource <corefilters/avisource>` Opens an AVI file.
- :doc:`DirectShowSource <corefilters/directshowsource>` DirectShowSource reads filename using DirectShow
- :doc:`ImageReader / ImageSource / ImageSourceAnim <corefilters/imagesource>` These filters produce a video
  clip by reading in still images or an animated image.
- :doc:`Imagewriter <corefilters/imagewriter>` Writes frames as images to your hard disk.
- :doc:`Import <corefilters/import>` Import an AviSynth script into the current script
- :doc:`SegmentedAVISource / SegmentedDirectShowSource <corefilters/segmentedsource>` The SegmentedAVISource
  filter automatically loads up to 100 avi files per argument
- :doc:`SoundOut <corefilters/soundout>` SoundOut is a GUI driven sound output module for AviSynth (it
  exports audio to several compressors).
- :doc:`WAVSource <corefilters/avisource>` Opens a WAV file or the audio of an AVI file.

.. _Color conversion and adjustment filters:

Color conversion and adjustment filters
---------------------------------------

These filters can be used to change the color format or adjust the colors of
a clip.

- :doc:`BuildPixelType <corefilters/buildpixeltype>` Builds a ``pixel_type`` format name
  string from a colorspace family, bit depth, chroma subsampling and/or a template clip.
- :doc:`ColorYUV <corefilters/coloryuv>` Adjusts colors and luma independently.
- :doc:`ConvertBackToYUY2 / ConvertToRGB / ConvertToRGB24 / ConvertToRGB32 /
  ConvertToYUY2 / ConvertToY8 / ConvertToYV411 / ConvertToYV12 / ConvertToYV16 /
  ConvertToYV24 <corefilters/convert>` AviSynth can deal internally with the
  color formats, RGB24, RGB32, YUY2, Y8, YV411, YV12, YV16 and YV24. These filters
  convert between them.
- :doc:`FixLuminance <corefilters/fixluminance>` Correct shifting vertical luma offset
- :doc:`Greyscale <corefilters/greyscale>` Converts a video to greyscale.
- :doc:`Invert <corefilters/invert>` Inverts selected color channels of a video.
- :doc:`Levels <corefilters/levels>` The Levels filter scales and clamps the blacklevel and whitelevel
  and adjusts the gamma.
- :doc:`Limiter <corefilters/limiter>` A filter for clipping levels to within CCIR-601 range.
- :doc:`MergeARGB / MergeRGB <corefilters/mergergb>` This filter makes it possible to select and combine a
  color channel from each of the input videoclips.
- :doc:`Merge / MergeChroma / MergeLuma <corefilters/merge>` This filter makes it possible to merge
  luma, chroma or both from a videoclip into another. There is an optional
  weighting, so a percentage between the two clips can be specified.
- :doc:`RGBAdjust <corefilters/adjust>` Adjust each color channel seperately.
- :doc:`ShowAlpha / ShowRed / ShowGreen / ShowBlue <corefilters/showalpha>` Shows the selected channel of
  an (A)RGB clip.
- :doc:`SwapUV / UToY / UToY8 / VToY / VToY8 / YToUV <corefilters/swap>` Swaps/copies chroma channels
  of a clip.
- :doc:`Subtract <corefilters/subtract>` Subtract produces an output clip in which every pixel is set
  according to the difference between the corresponding pixels
- :doc:`Tweak <corefilters/tweak>` Adjust the hue, saturation, brightness, and contrast.

.. _Overlay and Mask filters:

Overlay and Mask filters
------------------------

These filters can be used to layer clips with or without using masks and to
create masks.

- :doc:`ColorKeyMask <corefilters/layer>` Sets the alpha-channel (similar as Mask does) but generates
  it by comparing the color.
- :doc:`Layer <corefilters/layer>` Layering two videos.
- :doc:`Mask <corefilters/layer>` Applies an alpha-mask to a clip.
- :doc:`MaskHS <corefilters/maskhs>` Returns a mask (as Y8) of clip using a given hue and saturation
  range.
- :doc:`Overlay <corefilters/overlay>` Overlay puts two clips on top of each other with an optional
  displacement of the overlaying image, and using different overlay methods.
  Furthermore opacity can be adjusted for the overlay clip.
- :doc:`ResetMask <corefilters/layer>` Applies an "all-opaque" alpha-mask to clip.

.. _Geometric deformation filters:

Geometric deformation filters
-----------------------------

These filters can be used to change image size, process borders or make other
deformations of a clip.

- :doc:`AddBorders <corefilters/addborders>` AddBorders adds black borders around the image.
- :doc:`Crop / CropBottom <corefilters/crop>` Crop crops excess pixels off of each frame.
- :doc:`FlipHorizontal / FlipVertical <corefilters/flip>` Flips the video upside-down or left-to-right
- :doc:`Letterbox <corefilters/letterbox>` Letterbox simply blackens out the top and the bottom and
  optionally left and right side of each frame.
- :doc:`Overlay <corefilters/overlay>` Overlay puts two clips on top of each other with an optional
  displacement of the overlaying image, and using different overlay methods.
  Furthermore opacity can be adjusted for the overlay clip.
- :doc:`ReduceBy2 / HorizontalReduceBy2 / VerticalReduceBy2 <corefilters/reduceby2>` ReduceBy2 reduces the
  size of each frame by half.
- :doc:`BilinearResize / BicubicResize / BlackmanResize / GaussResize / LanczosResize /
  Lanczos4Resize / PointResize / SincResize / Spline16Resize / Spline36Resize /
  Spline64Resize <corefilters/resize>` The Resize filters rescale the input video
  frames to an arbitrary new resolution, using different sampling algorithms.
- :doc:`SkewRows <corefilters/skewrows>` SkewRows skews the rows of a clip.
- :doc:`TurnLeft / TurnRight / Turn180 <corefilters/turn>` Rotates the clip 90 degrees counterclock
  wise / 90 degrees clock wise / 180 degrees.

.. _Pixel restoration filters:

Pixel restoration filters
-------------------------

These filters can be used for image detail (pixel) restoration (like
denoising, sharpening) of a clip.

- :doc:`Blur / Sharpen <corefilters/blur>` These are simple 3x3-kernel blurring and sharpening
  filters.
- :doc:`GeneralConvolution <corefilters/convolution>` General 3x3 or 5x5 convolution matrix.
- :doc:`SpatialSoften / TemporalSoften <corefilters/soften>` The SpatialSoften and TemporalSoften
  filters remove noise from a video clip by selectively blending pixels.
- :doc:`FixBrokenChromaUpsampling <corefilters/fixbrokenchromaupsampling>` I noticed that the MS DV codec upsamples the
  chroma channels incorrectly, and I added a FixBrokenChromaUpsampling filter
  to compensate for it.

.. _Timeline editing filters:

Timeline editing filters
------------------------

These filters can be used to arrange frames in time (clip cutting, splicing
and other editing).

- :doc:`AlignedSplice / UnalignedSplice <corefilters/splice>` AlignedSplice and UnalignedSplice join two
  or more video clips end to end.
- :doc:`AssumeFPS / AssumeScaledFPS / ChangeFPS / ConvertFPS <corefilters/fps>` Changes framerates in
  different ways.
- :doc:`DeleteFrame <corefilters/deleteframe>` DeleteFrame deletes a set of single frames, given as a number
  of arguments.
- :doc:`Dissolve <corefilters/dissolve>` Dissolve is like AlignedSplice, except that the clips are
  combined with some overlap.
- :doc:`DuplicateFrame <corefilters/duplicateframe>` DuplicateFrame duplicates a set of single frames, given as
  a number of arguments.
- :doc:`FadeIn0 / FadeOut0 / FadeIn / FadeOut / FadeIn2 / FadeOut2 / FadeIO0 / FadeIO / FadeIO2 <corefilters/fade>` FadeIn and FadeOut cause the video stream to fade linearly
  to black at the start or end.
- :doc:`FreezeFrame <corefilters/freezeframe>` The FreezeFrame filter replaces all the frames between first-
  frame and last-frame with a selected frame.
- :doc:`Interleave <corefilters/interleave>` Interleave interleaves frames from several clips on a frame-by-
  frame basis.
- :doc:`Loop <corefilters/loop>` Loops the segment from start frame to end frame a given number of
  times.
- :doc:`Reverse <corefilters/reverse>` This filter makes a clip play in reverse.
- :doc:`SelectEven / SelectOdd <corefilters/select>` SelectEven makes an output video stream using only
  the even-numbered frames
- :doc:`SelectEvery <corefilters/selectevery>` SelectEvery is a generalization of filters like SelectEven and
  Pulldown.
- :doc:`SelectRangeEvery <corefilters/selectrangeevery>` This filters selects a range of frames with a certain
  period.
- :doc:`Trim <corefilters/trim>` Trims a video clip so that it includes only the frames first-frame
  through last-frame.

.. _Interlace filters:

Interlace filters
-----------------

These filters can be used for creating and processing field-based material
(which is frame-based material separated into fields). AviSynth is capable of
dealing with both progressive and interlaced material. The main problem is,
that it often doesn't know what it receives from source filters. This is the
reason that the field-based flag exists and can be used when dealing with
interlaced material. More information about field-based video can be found
:doc:`here <advancedtopics/interlaced_fieldbased>`.

- :doc:`AssumeFrameBased / AssumeFieldBased <corefilters/parity>` Forces frame-based or field-based
  material.
- :doc:`AssumeTFF / AssumeBFF <corefilters/parity>` Forces field order.
- :doc:`Bob <corefilters/bob>` Bob takes a clip and bob-deinterlaces it
- :doc:`ComplementParity <corefilters/parity>` Changes top fields to bottom fields and vice-versa.
- :doc:`DoubleWeave <corefilters/doubleweave>` The DoubleWeave filter operates like Weave, except that it
  produces double the number of frames by combining both the odd and even pairs
  of fields.
- :doc:`PeculiarBlend <corefilters/peculiar>` This filter blends each frame with the following frame in a
  peculiar way.
- :doc:`Pulldown <corefilters/pulldown>` The Pulldown filter simply selects two out of every five frames
  of the source video.
- :doc:`SeparateColumns / SeparateRows <corefilters/separatefields>` Takes a clip and separates the columns or
  rows of each frame into new frames.
- :doc:`SeparateFields <corefilters/separatefields>` SeparateFields takes a frame-based clip and splits each
  frame into its component top and bottom fields.
- :doc:`SwapFields <corefilters/swapfields>` The SwapFields filter swaps the two fields in an interlaced
  frame
- :doc:`Weave <corefilters/weave>` Weave takes even pairs of fields from a Fields Separated input video
  clip and combines them together to produce interlaced frames.
- :doc:`WeaveColumns / WeaveRows <corefilters/weave>` Takes a clip and weaves sets of columns or rows
  together to produce a composite frames.

.. _Audio processing filters:

Audio processing filters
------------------------

These filters can be used to process audio. Audio samples from a clip will be
automatically converted if any filters requires a special type of sample.
This means that if a filter doesn't support the type of sample it is given,
it will automatically convert the samples to something it supports. The
internal formats supported in each filter is listed in the sample type
column. A specific sample type can be forced by using the :doc:`ConvertAudio <corefilters/convertaudio>`
functions.

If the sample type is float, when AviSynth has to output the data, it will be
converted to 16 bit, since float cannot be passed as valid AVI data.

- :doc:`Amplify / AmplifydB <corefilters/amplify>` Amplify multiply audio samples by amount.
- :doc:`AssumeSampleRate <corefilters/assumerate>` Adjusts the playback speed of the audio.
- :doc:`AudioDub / AudioDubEx <corefilters/audiodub>` AudioDub takes the video stream from the first
  argument and the audio stream from the second argument and combines them.
  AudioDubEx is similar, but it doesn't throw an exception if both clips don't
  have a video or audio stream.
- :doc:`AudioStats <corefilters/audiostats>` Returns the levels of the audio track (peak, RMS, clipped samples,
  loose peak), for all channels and per channel, as an array of named values.
- :doc:`AudioTrim <corefilters/trim>` Trims an audio clip so that it includes only the start_time
  through end_time.
- :doc:`ConvertToMono <corefilters/converttomono>` Merges all audio channels.
- :doc:`ConvertAudioTo8bit / ConvertAudioTo16bit / ConvertAudioTo24bit / ConvertAudioTo32bit /
  ConvertAudioToFloat <corefilters/convertaudio>` Converts audio samples to 8, 16,
  24, 32 bits or float.
- :doc:`DelayAudio <corefilters/delayaudio>` DelayAudio delays the audio track by seconds seconds.
- :doc:`EnsureVBRMP3Sync <corefilters/ensuresync>` Corrects out-of-sync mp3-AVI's, when seeking ot trimming.
- :doc:`GetChannel <corefilters/getchannel>` Returns a channel from an audio signal.
- :doc:`KillAudio <corefilters/killaudio>` Removes the audio from a clip completely.
- :doc:`KillAudio <corefilters/killaudio>` Removes the video from a clip completely.
- :doc:`MergeChannels <corefilters/mergechannels>` Merges channels of two or more audio clips.
- :doc:`MixAudio <corefilters/mixaudio>` Mixes audio from two clips.
- :doc:`Normalize <corefilters/normalize>` Amplifies the entire waveform as much as possible, without
  clipping.
- :doc:`ResampleAudio <corefilters/resampleaudio>` Performs a change of the audio sample rate.
- :doc:`SSRC <corefilters/ssrc>` Performs a high-quality change of the audio sample rate. It uses SSRC
  by Naoki Shibata, which offers the best resample quality available.
- :doc:`SuperEQ <corefilters/supereq>` High quality 16 band sound equalizer.
- :doc:`TimeStretch <corefilters/timestretch>` This filter can change speed of the sound without changing the
  pitch, and change the pitch of a sound without changing the length of a
  sound.
- :doc:`SetChannelMask, GetChannelMask, IsChannelMaskKnown <corefilters/setchannelmask>` 
  This filter help you with setting and getting the audio channel layout of the audio stream.

.. _Meta filters:

Meta filters
------------

These special filters can be used to control other filters execution.

- :doc:`Animate / ApplyRange <corefilters/animate>` Animate (ApplyRange) is a meta-filter which evaluates
  its parameter filter with continuously varying (the same) arguments.
- :doc:`TCPDeliver <corefilters/tcpdeliver>` This filter will enable you to send clips over your network.
  You can connect several clients to the same machine.

.. _Conditional filters:

Conditional filters
-------------------

The basic characteristic of conditional filters is that 'their scripts' are
evaluated (executed) at every frame instead of the whole clip. This allows
for complex video processing that would be difficult or impossible to be
performed by a normal AviSynth script.

- :doc:`ConditionalFilter / FrameEvaluate / ScriptClip / ConditionalSelect <corefilters/conditionalfilter>`
  ConditionalFilter returns source1 if some condition is met, otherwise it
  returns source2. ScriptClip/FrameEvaluate returns the clip which is returned
  by the function evaluated on every frame. ConditionalSelect returns one frame
  from several sources based on an integer evaluator.
- :doc:`ConditionalReader <corefilters/conditionalreader>` ConditionalReader allows you to import information from
  a text file, with different values for each frame - or a range of frames.
- :doc:`WriteFile / WriteFileIf / WriteFileStart / WriteFileEnd <corefilters/write>` These filters
  evaluate expressions and output the results to a text-file.

.. _Debug filters:

Debug filters
-------------

- :doc:`BlankClip / Blackness <corefilters/blankclip>` The BlankClip filter produces a solid color, silent
  video clip of the specified length (in frames).
- :doc:`ColorBars / ColorBarsHD <corefilters/colorbars>` The ColorBars filters produce a video clip
  containing SMPTE color bars scaled to any image size.
- :doc:`Compare <corefilters/compare>` Compares two clips and prints out information about the
  differences.
- :doc:`Echo <corefilters/echo>` Forces getframe calls to all input clips. Returns only first clip
  result.
- :doc:`Histogram <corefilters/histogram>` Adds a Histogram.
- :doc:`Info <corefilters/info>` Prints out image and sound information.
- :doc:`Preroll <corefilters/preroll>` Preroll the audio or video on non linear access.
- :doc:`MessageClip <corefilters/message>` MessageClip produces a clip containing a text message
- :doc:`ShowFiveVersions <corefilters/showfive>` ShowFiveVersions takes five video streams and combines
  them in a staggered arrangement from left to right.
- :doc:`ShowFrameNumber / ShowSMPTE / ShowTime <corefilters/showframes>` ShowFrameNumber draws text on every
  frame indicating what number Avisynth thinks it is.
  ShowSMPTE displays the SMPTE timecode. **hh:mm:ss:ff**
  ShowTime displays the duration with millisecond resolution. **hh:mm:ss.sss**
- :doc:`StackHorizontal / StackVertical <corefilters/stack>` StackHorizontal takes two or more video
  clips and displays them together in left-to-right order.
- :doc:`Subtitle <corefilters/subtitle>` The Subtitle filter adds a single line of anti-aliased text to a
  range of frames.
- :doc:`Tone <corefilters/tone>` This will generate sound.
- :doc:`Version <corefilters/version>` The Version filter generates a video clip with a short version and
  copyright statement

$Date: 2026/11/14 14:15:00 $
