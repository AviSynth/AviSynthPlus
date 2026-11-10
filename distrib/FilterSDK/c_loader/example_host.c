/* Example host for avisynth_c_loader.h (C11): loads AviSynth, prints its version, evaluates a script.
   Build, e.g.:  gcc -std=c11 -I<avisynth include dir> example_host.c -o example_host  (Linux: add -ldl)
                 cl /std:c11 /I<avisynth include dir> example_host.c
   Run:          example_host [path of AviSynth.dll / libavisynth.so] */

#include <stdio.h>
#include "avisynth_c_loader.h"

int main(int argc, char** argv)
{
  char error[AVSC_LOADER_ERROR_SIZE];
  AVS_Library* lib = avs_load_library_ex(argc > 1 ? argv[1] : NULL, error, sizeof(error));
  AVS_ScriptEnvironment* env;
  AVS_Value arg, result;

  if (lib == NULL) {
    printf("AviSynth could not be loaded: %s\n", error);
    return 1;
  }

  env = avs_library_create_script_environment(lib);
  if (env == NULL) {
    printf("no script environment\n");
    avs_free_library(lib);
    return 1;
  }
  printf("interface version %d.%d, 64 bit values: %s\n",
    lib->interface_version, lib->interface_bugfix, lib->has_64bit_values ? "yes" : "no");

  /* AVS_Value through the API (avs_set_to_xxx, avs_val_is_xxx, avs_get_as_xxx, V11) rather than the
     inline helpers of avisynth_c.h (avs_new_value_xxx, avs_is_xxx, avs_as_xxx) which read the struct
     directly. On an AviSynth older than V11 the loader gives fallbacks for them, so they work everywhere. */
  lib->avs_set_to_string(&arg, "Version().Info()"); /* not copied: the string must live during the call */
  /* One argument. For no arguments pass an empty array (avs_new_value_array(NULL, 0)):
     classic AviSynth 2.6 does not accept avs_void there. */
  result = lib->avs_invoke(env, "Eval", arg, NULL);
  if (lib->avs_val_is_error(result))
    printf("error: %s\n", lib->avs_get_as_error(result));
  else if (lib->avs_val_is_clip(result)) {
    AVS_Clip* clip = lib->avs_take_clip(result, env); /* a new reference: release it as well */
    const AVS_VideoInfo* vi = lib->avs_get_video_info(clip);
    printf("clip: %dx%d, %d frames\n", vi->width, vi->height, vi->num_frames);
    lib->avs_release_clip(clip);
  }
  /* The result of avs_invoke is owned by the caller: release it in every case (error, clip or other
     value), otherwise e.g. the clip and its whole filter chain is never freed. */
  lib->avs_release_value(result);
  lib->avs_release_value(arg); /* nothing to free for a string, kept for symmetry */

  if (lib->has_at_least_v14_1)
    printf("_Range is the stored range property, _ColorRange its alias\n");

  lib->avs_delete_script_environment(env);
  avs_free_library(lib);
  return 0;
}
