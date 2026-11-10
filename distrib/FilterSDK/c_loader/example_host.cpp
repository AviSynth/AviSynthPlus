// Example host for avisynth_c_loader.hpp (C++17): loads AviSynth, prints its version, evaluates a script.
// Build, e.g.:  g++ -std=c++17 -I<avisynth include dir> example_host.cpp -o example_host  (Linux: add -ldl)
//               cl /std:c++17 /EHsc /I<avisynth include dir> example_host.cpp
// Run:          example_host [path of AviSynth.dll / libavisynth.so]

#include <cstdio>
#include "avisynth_c_loader.hpp"

int main(int argc, char** argv)
{
  avsc::Library lib;
  if (!lib.load(argc > 1 ? argv[1] : nullptr)) { // nullptr: the default name from the library search path
    std::printf("AviSynth could not be loaded: %s\n", lib.error());
    return 1;
  }

  AVS_ScriptEnvironment* env = lib.create_script_environment();
  if (env == nullptr) {
    std::printf("no script environment\n");
    return 1;
  }
  std::printf("interface version %d.%d, 64 bit values: %s\n",
    lib.interface_version, lib.interface_bugfix, lib.has_64bit_values ? "yes" : "no");

  // AVS_Value through the API (avs_set_to_xxx, avs_val_is_xxx, avs_get_as_xxx, V11) rather than the
  // inline helpers of avisynth_c.h (avs_new_value_xxx, avs_is_xxx, avs_as_xxx) which read the struct
  // directly. On an AviSynth older than V11 the loader gives fallbacks for them, so they work everywhere.
  AVS_Value arg;
  lib.avs_set_to_string(&arg, "Version().Info()"); // the string is not copied: it must live during the call
  // One argument. For no arguments pass an empty array (avs_new_value_array(nullptr, 0)):
  // classic AviSynth 2.6 does not accept avs_void there.
  AVS_Value result = lib.avs_invoke(env, "Eval", arg, nullptr);
  if (lib.avs_val_is_error(result))
    std::printf("error: %s\n", lib.avs_get_as_error(result));
  else if (lib.avs_val_is_clip(result)) {
    AVS_Clip* clip = lib.avs_take_clip(result, env); // a new reference: release it as well
    const AVS_VideoInfo* vi = lib.avs_get_video_info(clip);
    std::printf("clip: %dx%d, %d frames\n", vi->width, vi->height, vi->num_frames);
    lib.avs_release_clip(clip);
  }
  // The result of avs_invoke is owned by the caller: release it in every case (error, clip or other
  // value), otherwise e.g. the clip and its whole filter chain is never freed.
  lib.avs_release_value(result);
  lib.avs_release_value(arg); // nothing to free for a string, kept for symmetry

  if (lib.has_at_least_v14_1)
    std::printf("_Range is the stored range property, _ColorRange its alias\n");

  lib.avs_delete_script_environment(env); // before lib goes out of scope
  return 0;
}
