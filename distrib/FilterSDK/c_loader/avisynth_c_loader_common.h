// AviSynth+ C API dynamic loader: the part shared by the two loader headers. Not for direct inclusion.
// Copyright 2026 AviSynth+ project
// Same license and linking exception as avisynth_c.h.
//
// The two loaders share loading, binding and compatibility handling, with different front ends:
// - avisynth_c_loader.hpp: C++ (C++17). A class, avsc::Library, which releases any owned reference on destruction.
//   For C++ hosts and plugins which use the C API, e.g. to stay independent of the compiler AviSynth was
//   built with: this is the usual case.
// - avisynth_c_loader.h: C11, also compiles as C++. A struct, AVS_Library, and free functions.
//
// This file holds what must be the same in both, so it is written once:
// - AVSC_LOADER_FUNCTIONS: the list of all C API functions
// - the fallbacks for functions missing from the loaded AviSynth, and their table (AVSC_LOADER_FALLBACKS)
// - the interface version query, the failed environment check, the environment request order
// - the platform calls: open the library, get a function address, close the library
//
// It needs avisynth_c.h included with AVSC_NO_DECLSPEC (both loaders do so): then each API function
// name has a function pointer type name_func.
// A sample like the two loaders, not an installed AviSynth header: copy it together with them.

#ifndef AVISYNTH_C_LOADER_COMMON_H
#define AVISYNTH_C_LOADER_COMMON_H

#if !defined(__AVISYNTH_C__) || !defined(AVSC_NO_DECLSPEC)
#error "include avisynth_c_loader.hpp or avisynth_c_loader.h, not avisynth_c_loader_common.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef AVS_WINDOWS
// C++ users of std::min/std::max: define NOMINMAX before including the loader (or before windows.h).
#include <windows.h>
#else
#include <dlfcn.h>
#if defined(AVS_MACOS)
#include <mach-o/dyld.h>
#elif !defined(AVS_HAIKU)
#include <limits.h> // PATH_MAX: loaded-library path buffer and bounds check (Linux, BSD)
#ifndef PATH_MAX
// glibc defines it only with POSIX features enabled, e.g. not with gcc -std=c11 (gnu11 has it)
#define PATH_MAX 4096
#endif
#include <link.h> // dl_iterate_phdr: Linux, BSD
#if defined(__GLIBC__) && !defined(__USE_GNU)
// glibc declares dl_iterate_phdr only with _GNU_SOURCE (g++ defines it, plain C does not, and it is too late
// to define it here): declare the part we use. The first two members of the struct are the same in glibc.
struct dl_phdr_info { ElfW(Addr) dlpi_addr; const char* dlpi_name; };
extern int dl_iterate_phdr(int (*callback)(struct dl_phdr_info*, size_t, void*), void* data);
#endif
#endif
#endif

// Linkage of the functions in the loader headers.
// - C++: inline. One definition for the whole program, the linker merges the copies.
// - C: static inline, a private copy in each source file which uses it. C has no portable
//   header-only alternative: a C inline function needs one non-inline definition in some source file.
#ifdef __cplusplus
#define AVSC_LOADER_INLINE inline
#else
#define AVSC_LOADER_INLINE AVSC_INLINE
#endif

// The list of all C API functions: X(name), grouped by the interface version they appeared in.
#define AVSC_LOADER_FUNCTIONS(X) \
  /* V3: AviSynth 2.5 */ \
  X(avs_add_function) \
  X(avs_at_exit) \
  X(avs_bit_blt) \
  X(avs_check_version) \
  X(avs_clip_get_error) \
  X(avs_copy_clip) \
  X(avs_copy_value) \
  X(avs_copy_video_frame) \
  X(avs_create_script_environment) \
  X(avs_function_exists) \
  X(avs_get_audio) \
  X(avs_get_cpu_flags) \
  X(avs_get_frame) \
  X(avs_get_parity) \
  X(avs_get_var) \
  X(avs_get_version) \
  X(avs_get_video_info) \
  X(avs_invoke) \
  X(avs_make_writable) \
  X(avs_new_c_filter) /* AVS_MT_NICE_FILTER C filters: error reporting reliable from V14 */ \
  X(avs_new_video_frame_a) \
  X(avs_release_clip) \
  X(avs_release_value) \
  X(avs_release_video_frame) \
  X(avs_save_string) \
  X(avs_set_cache_hints) \
  X(avs_set_global_var) \
  X(avs_set_memory_max) \
  X(avs_set_to_clip) \
  X(avs_set_var) \
  X(avs_set_working_dir) \
  X(avs_subframe) \
  X(avs_take_clip) \
  X(avs_vsprintf) \
  X(avs_sprintf) \
  /* V6: AviSynth 2.6 */ \
  X(avs_delete_script_environment) \
  X(avs_subframe_planar) \
  X(avs_get_error) \
  X(avs_is_yv24) \
  X(avs_is_yv16) \
  X(avs_is_yv12) \
  X(avs_is_yv411) \
  X(avs_is_y8) \
  X(avs_is_color_space) \
  X(avs_get_plane_width_subsampling) \
  X(avs_get_plane_height_subsampling) \
  X(avs_bits_per_pixel) \
  X(avs_bytes_from_pixels) \
  X(avs_row_size) \
  X(avs_bmp_size) \
  X(avs_get_pitch_p) \
  X(avs_get_row_size_p) \
  X(avs_get_height_p) \
  X(avs_get_read_ptr_p) \
  X(avs_is_writable) \
  X(avs_get_write_ptr_p) \
  /* AVS+ */ \
  X(avs_num_components) \
  X(avs_component_size) \
  X(avs_bits_per_component) \
  X(avs_is_444) \
  X(avs_is_422) \
  X(avs_is_420) \
  X(avs_is_y) \
  X(avs_is_yuva) \
  X(avs_is_planar_rgb) \
  X(avs_is_planar_rgba) \
  X(avs_is_rgb48) \
  X(avs_is_rgb64) \
  /* AVS+, deprecated */ \
  X(avs_is_yuv444p16) \
  X(avs_is_yuv422p16) \
  X(avs_is_yuv420p16) \
  X(avs_is_y16) \
  X(avs_is_yuv444ps) \
  X(avs_is_yuv422ps) \
  X(avs_is_yuv420ps) \
  X(avs_is_y32) \
  /* V8: frame properties. Frame property access is usable from V9 (crashed in V8) */ \
  X(avs_subframe_planar_a) \
  X(avs_copy_frame_props) /* usable from V9 */ \
  X(avs_get_frame_props_ro) /* usable from V9 */ \
  X(avs_get_frame_props_rw) /* usable from V9 */ \
  X(avs_prop_num_keys) \
  X(avs_prop_get_key) \
  X(avs_prop_num_elements) \
  X(avs_prop_get_type) \
  /* avs_prop_get_xxx: before V14 pass a non-NULL error pointer (NULL: process terminated on a failed read) */ \
  X(avs_prop_get_int) \
  X(avs_prop_get_float) \
  X(avs_prop_get_data) /* usable from V10 */ \
  X(avs_prop_get_data_size) \
  X(avs_prop_get_clip) /* usable from V14 (returned an invalid AVS_Clip) */ \
  X(avs_prop_get_frame) /* V11: *error was always 0, check the result for NULL */ \
  X(avs_prop_delete_key) \
  X(avs_prop_set_int) \
  X(avs_prop_set_float) \
  X(avs_prop_set_data) \
  X(avs_prop_set_clip) \
  X(avs_prop_set_frame) /* usable from V9 */ \
  X(avs_prop_get_int_array) \
  X(avs_prop_get_float_array) \
  X(avs_prop_set_int_array) \
  X(avs_prop_set_float_array) \
  X(avs_clear_map) \
  X(avs_new_video_frame_p) /* non-NULL prop_src usable from V10 */ \
  X(avs_new_video_frame_p_a) /* non-NULL prop_src usable from V10 */ \
  X(avs_get_env_property) /* AVS_AEP_INTERFACE_VERSION and AVS_AEP_INTERFACE_BUGFIX from V9 */ \
  X(avs_pool_allocate) \
  X(avs_pool_free) \
  X(avs_get_var_try) \
  X(avs_get_var_bool) \
  X(avs_get_var_int) \
  X(avs_get_var_double) \
  X(avs_get_var_string) \
  X(avs_get_var_long) \
  /* V9 */ \
  X(avs_is_property_writable) \
  X(avs_make_property_writable) \
  /* V10 */ \
  X(avs_video_frame_get_pixel_type) \
  X(avs_video_frame_amend_pixel_type) \
  /* V10.1 */ \
  X(avs_is_channel_mask_known) \
  X(avs_set_channel_mask) \
  X(avs_get_channel_mask) \
  /* V11: 64 bit AVS_Value types */ \
  X(avs_set_to_void) \
  X(avs_set_to_error) \
  X(avs_set_to_bool) \
  X(avs_set_to_int) \
  X(avs_set_to_string) \
  X(avs_set_to_float) \
  X(avs_set_to_double) \
  X(avs_set_to_long) \
  X(avs_set_to_array) /* NULL src with nonzero size: usable from V14 */ \
  X(avs_get_as_bool) \
  X(avs_get_as_clip) \
  X(avs_get_as_int) \
  X(avs_get_as_long) \
  X(avs_get_as_string) \
  X(avs_get_as_float) \
  X(avs_get_as_error) \
  X(avs_get_as_array) \
  X(avs_get_array_elt) \
  X(avs_get_array_size) \
  X(avs_val_defined) \
  X(avs_val_is_clip) \
  X(avs_val_is_bool) \
  X(avs_val_is_int) \
  X(avs_val_is_long_strict) \
  X(avs_val_is_float) \
  X(avs_val_is_floatf_strict) \
  X(avs_val_is_string) \
  X(avs_val_is_array) \
  X(avs_val_is_error) \
  X(avs_prop_get_int_saturated) \
  X(avs_prop_get_float_saturated) \
  X(avs_prop_get_data_type_hint) \
  X(avs_prop_set_data_h) \
  X(avs_add_function_r) \
  /* V12 */ \
  X(avs_get_cpu_flags_ex) \
  X(avs_acquire_global_lock) \
  X(avs_release_global_lock) \
  /* V13: new pixel formats */ \
  X(avs_is_440) \
  X(avs_is_410) \
  X(avs_is_411) \
  X(avs_is_ya) \
  /* V14 */ \
  X(avs_val_is_function) \
  X(avs_val_get_type) \
  X(avs_invoke_try) \
  X(avs_invoke2) \
  X(avs_invoke2_try) \
  X(avs_invoke3) \
  X(avs_invoke3_try) \
  X(avs_create_map) \
  X(avs_free_map) \
  X(avs_apply_message)

// The V8/V9 frame property functions: has_frame_props requires V9 and all of these (none has a fallback).
// The version alone is not enough: Win32 (x86) 3.7.1 is V9, but exports them only with decorated names
// (_avs_get_frame_props_ro@8), so they are NULL there; 3.6.x/3.7.0 (V8) the same. Fixed in 3.7.2.
// If you trim AVSC_LOADER_FUNCTIONS, trim this list too.
#define AVSC_LOADER_FRAME_PROP_FUNCTIONS(X) \
  X(avs_copy_frame_props) \
  X(avs_get_frame_props_ro) \
  X(avs_get_frame_props_rw) \
  X(avs_prop_num_keys) \
  X(avs_prop_get_key) \
  X(avs_prop_num_elements) \
  X(avs_prop_get_type) \
  X(avs_prop_get_int) \
  X(avs_prop_get_float) \
  X(avs_prop_get_data) \
  X(avs_prop_get_data_size) \
  X(avs_prop_get_clip) \
  X(avs_prop_get_frame) \
  X(avs_prop_delete_key) \
  X(avs_prop_set_int) \
  X(avs_prop_set_float) \
  X(avs_prop_set_data) \
  X(avs_prop_set_clip) \
  X(avs_prop_set_frame) \
  X(avs_prop_get_int_array) \
  X(avs_prop_get_float_array) \
  X(avs_prop_set_int_array) \
  X(avs_prop_set_float_array) \
  X(avs_clear_map) \
  X(avs_is_property_writable) \
  X(avs_make_property_writable)

/////////////////////////////////////////////////////////////////////////////////////////////////////
// Platform part

// Suggested diagnostic buffer size; longer messages are truncated.
#define AVSC_LOADER_ERROR_SIZE 512

// The library name used for NULL
AVSC_LOADER_INLINE const char* avs_loader_default_name(void)
{
#if defined(AVS_WINDOWS)
  return "avisynth";
#elif defined(AVS_MACOS)
  return "libavisynth.dylib";
#else
  return "libavisynth.so";
#endif
}

// Append to an initialized diagnostic buffer; truncation is intentional.
AVSC_LOADER_INLINE void avs_loader_append_error(char* error, size_t error_size, const char* text)
{
  size_t used, length;
  if (error == NULL || error_size == 0)
    return;
  used = strlen(error);
  length = strlen(text);
  if (length > error_size - used - 1)
    length = error_size - used - 1;
  memcpy(error + used, text, length);
  error[used + length] = '\0';
}

// Discard an incomplete trailing character after bounded formatting of valid UTF-8.
AVSC_LOADER_INLINE void avs_loader_trim_incomplete_utf8(char* text)
{
  size_t end = strlen(text);
  size_t start = end;
  while (start > 0 && ((unsigned char)text[start - 1] & 0xC0) == 0x80)
    start--;
  if (start > 0) {
    unsigned char lead = (unsigned char)text[--start];
    size_t width = lead < 0x80 ? 1 : lead < 0xE0 ? 2 : lead < 0xF0 ? 3 : 4;
    if (end - start < width)
      text[start] = '\0';
  }
}

// Loads the library.
// library_path: UTF-8 file name or path; NULL: the default name, see avs_loader_default_name.
// Returns NULL on failure.
// error (optional): receives a NUL-terminated diagnostic in at most error_size bytes; empty on success.
// Windows: both the path and system message are UTF-8; truncation preserves complete characters.
// Other systems: the dlerror() text in its original encoding. AVSC_LOADER_ERROR_SIZE is a suggested
// buffer size, not a limit on the input path.
// Windows: explains common failures, keeping ambiguous causes (e.g. an invalid image versus an
// architecture mismatch) as alternatives. Other systems: the dlerror() text.
AVSC_LOADER_INLINE void* avs_loader_open(const char* library_path, char* error, size_t error_size)
{
  void* handle;
  if (library_path == NULL)
    library_path = avs_loader_default_name();
  if (error != NULL && error_size > 0)
    error[0] = '\0';
#ifdef AVS_WINDOWS
  {
    // We avoid MAX_PATH (260), allow arbitrary-length path.
    // Query the UTF-16 size including NUL, then allocate without a MAX_PATH limit.
    // Windows long-path rules still apply; the caller supplies any required \\?\ prefix.
    int wpath_length = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, library_path, -1, NULL, 0);
    wchar_t* wpath;
    DWORD code;
    DWORD attributes;
    if (wpath_length == 0) {
      avs_loader_append_error(error, error_size, "file name is not valid UTF-8");
      return NULL;
    }
    wpath = (wchar_t*)malloc((size_t)wpath_length * sizeof(wchar_t));
    if (wpath == NULL) {
      avs_loader_append_error(error, error_size, "out of memory");
      return NULL;
    }
    // Reject invalid UTF-8 rather than substituting characters in the file name.
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, library_path, -1, wpath, wpath_length) == 0) {
      free(wpath);
      avs_loader_append_error(error, error_size, "file name conversion failed");
      return NULL;
    }
    handle = (void*)LoadLibraryW(wpath);
    code = handle == NULL ? GetLastError() : ERROR_SUCCESS;
    if (handle != NULL || error == NULL || error_size == 0) {
      free(wpath);
      return handle;
    }
    avs_loader_append_error(error, error_size, library_path);
    avs_loader_append_error(error, error_size, ": ");
    attributes = GetFileAttributesW(wpath);
    if (attributes != INVALID_FILE_ATTRIBUTES && (attributes & FILE_ATTRIBUTE_DIRECTORY)) {
      char reason[64];
      snprintf(reason, sizeof(reason), "path is a directory (error %lu)", (unsigned long)code);
      avs_loader_append_error(error, error_size, reason);
    }
    else switch (code) {
    case ERROR_MOD_NOT_FOUND: // 126: the file itself, or a DLL it depends on
      if (wcschr(wpath, L'\\') == NULL && wcschr(wpath, L'/') == NULL)
        avs_loader_append_error(error, error_size, "not found in the DLL search path, or a DLL it needs is missing "
          "(e.g. the Visual C++ runtime) (error 126)");
      else if (attributes == INVALID_FILE_ATTRIBUTES)
        avs_loader_append_error(error, error_size, "file or a DLL it needs was not found (error 126)");
      else
        avs_loader_append_error(error, error_size,
          "a DLL it needs is missing (e.g. the Visual C++ runtime) (error 126)");
      break;
    case ERROR_BAD_EXE_FORMAT: { // 193: malformed images and incompatible architectures share this code
      char reason[128];
      snprintf(reason, sizeof(reason), "invalid library image or incompatible architecture for this %d bit process "
        "(error 193)", (int)(sizeof(void*) * 8));
      avs_loader_append_error(error, error_size, reason);
      break;
    }
    case ERROR_PROC_NOT_FOUND: // 127
      avs_loader_append_error(error, error_size,
        "a required function was not found in the library or a dependency (error 127)");
      break;
    case ERROR_DLL_INIT_FAILED: // 1114
      avs_loader_append_error(error, error_size, "library or dependency initialization failed (error 1114)");
      break;
    default: {
      wchar_t* wtext = NULL;
      char* text = NULL;
      char code_text[32];
      DWORD len = FormatMessageW(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS |
        FORMAT_MESSAGE_ALLOCATE_BUFFER, NULL, code, 0, (LPWSTR)&wtext, 0, NULL);
      while (len > 0 && (wtext[len - 1] == L'\r' || wtext[len - 1] == L'\n' ||
        wtext[len - 1] == L' ' || wtext[len - 1] == L'.'))
        len--;
      if (len > 0) {
        // we promised to return the error in UTF-8
        int text_length;
        wtext[len] = L'\0';
        text_length = WideCharToMultiByte(CP_UTF8, 0, wtext, -1, NULL, 0, NULL, NULL);
        if (text_length > 0) {
          text = (char*)malloc((size_t)text_length);
          if (text != NULL && WideCharToMultiByte(CP_UTF8, 0, wtext, -1,
            text, text_length, NULL, NULL) == 0) {
            free(text);
            text = NULL;
          }
        }
      }
      if (wtext != NULL)
        LocalFree(wtext); // free up the buffer allocated by FormatMessageW
      avs_loader_append_error(error, error_size, "cannot be loaded: ");
      avs_loader_append_error(error, error_size, text != NULL ? text : "unknown reason");
      snprintf(code_text, sizeof(code_text), " (error %lu)", (unsigned long)code);
      avs_loader_append_error(error, error_size, code_text);
      free(text);
      break;
    }
    }
    free(wpath);
    avs_loader_trim_incomplete_utf8(error);
    return NULL;
  }
#else
  handle = dlopen(library_path, RTLD_NOW | RTLD_LOCAL);
  if (handle == NULL && error != NULL && error_size > 0) {
    const char* text = dlerror(); // already names the file
    avs_loader_append_error(error, error_size, text != NULL ? text : "cannot be loaded");
  }
  return handle;
#endif
}

// For the loaders: the library was loaded, but it does not have the C API entry every AviSynth with a
// C interface has (since 2.5): not AviSynth, or a very old one. Fills the error text.
AVSC_LOADER_INLINE void avs_loader_not_avisynth_error(const char* library_path, char* error, size_t error_size)
{
  if (error == NULL || error_size == 0)
    return;
  if (library_path == NULL)
    library_path = avs_loader_default_name();
  error[0] = '\0';
  avs_loader_append_error(error, error_size, library_path);
  avs_loader_append_error(error, error_size, ": loaded, but it is not AviSynth with a C interface "
    "(avs_create_script_environment is missing)");
  avs_loader_trim_incomplete_utf8(error);
}

// Match prefix[.N[.N...]]suffix exactly. Prefix-only checks also accept unrelated plugins
// and backup files; version components must be nonempty ASCII digit sequences.
// Linux/BSD (prefix "libavisynth.so", suffix ""):
//   Accept: libavisynth.so, libavisynth.so.14, libavisynth.so.3.7.6
//   Reject: libavisynth.something, libavisynth.so.backup, libavisynth.so.14a, libavisynth.so.14.
// macOS (prefix "libavisynth", suffix ".dylib"):
//   Accept: libavisynth.dylib, libavisynth.14.dylib, libavisynth.3.7.6.dylib
//   Reject: libavisynth_plugin.dylib, libavisynth.dylib.backup, libavisynth..dylib
AVSC_LOADER_INLINE int avs_loader_matches_library_name(const char* name, const char* prefix, const char* suffix)
{
  size_t prefix_length = strlen(prefix);
  if (strncmp(name, prefix, prefix_length) != 0)
    return 0;
  name += prefix_length;
  if (strcmp(name, suffix) == 0)
    return 1;
  while (*name == '.') {
    name++;
    if (*name < '0' || *name > '9')
      return 0;
    do {
      name++;
    } while (*name >= '0' && *name <= '9');
    if (strcmp(name, suffix) == 0)
      return 1;
  }
  return 0;
}

#if defined(AVS_POSIX) && !defined(AVS_MACOS) && !defined(AVS_HAIKU)
// dl_iterate_phdr callback: the first loaded object named libavisynth.so[.N] (path in data)
AVSC_LOADER_INLINE int avs_loader_find_loaded_cb(struct dl_phdr_info* info, size_t size, void* data)
{
  const char* name = info->dlpi_name;
  const char* base;
  (void)size;
  if (name == NULL || name[0] == '\0') // the main program
    return 0;
  base = strrchr(name, '/');
  base = base != NULL ? base + 1 : name;
  if (!avs_loader_matches_library_name(base, "libavisynth.so", "") || strlen(name) >= PATH_MAX)
    return 0;
  strcpy((char*)data, name);
  return 1;
}
#endif

// Acquires a reference to the already-loaded AviSynth for plugins; release with avs_loader_close.
// Returns NULL on failure and optionally writes a diagnostic to error.
// - Windows: the module named "avisynth" (AviSynth.dll). A renamed AviSynth.dll is not found.
// - Linux, BSD: dl_iterate_phdr locates libavisynth.so[.N]; dlopen acquires another reference.
// - macOS: dyld image enumeration locates libavisynth[.N].dylib; dlopen acquires another reference.
// - Haiku: attachment is not implemented; native image enumeration uses get_next_image_info.
AVSC_LOADER_INLINE void* avs_loader_attach(char* error, size_t error_size)
{
  void* handle = NULL;
  if (error != NULL && error_size > 0)
    error[0] = '\0';
#if defined(AVS_WINDOWS)
  {
    HMODULE module = NULL;
    // flags 0: the reference count is incremented, pair it with FreeLibrary
    if (GetModuleHandleExW(0, L"avisynth", &module))
      handle = (void*)module;
  }
#elif defined(AVS_MACOS)
  {
    // Not thread-safe: concurrent image loading/unloading can change indices or invalidate dyld-owned
    // names. Call only when no other thread can load/unload libraries, including unrelated ones.
    // Plugin init runs after dlopen returns, so it does not itself guarantee this exclusion.
    // Add-image callbacks have no public unregister API (unsafe for unloadable plugins);
    // dladdr is thread-safe but requires an address in AviSynth, which is not yet available here.
    uint32_t i;
    uint32_t count = _dyld_image_count();
    for (i = 0; i < count && handle == NULL; i++) {
      const char* name = _dyld_get_image_name(i);
      const char* base = name != NULL ? strrchr(name, '/') : NULL;
      base = base != NULL ? base + 1 : name;
      if (base != NULL && avs_loader_matches_library_name(base, "libavisynth", ".dylib"))
        handle = dlopen(name, RTLD_NOW | RTLD_LOCAL); // already loaded: the same instance, a new reference
    }
  }
#elif !defined(AVS_HAIKU)
  {
    char path[PATH_MAX];
    path[0] = '\0';
    // Enumerate loaded objects to reuse AviSynth's actual path. A bare-name dlopen could load
    // another copy, or fail if only the versioned library (not the development symlink) exists.
    if (dl_iterate_phdr(avs_loader_find_loaded_cb, path) != 0)
      handle = dlopen(path, RTLD_NOW | RTLD_LOCAL); // already loaded: the same instance, a new reference
  }
#endif
  if (handle == NULL && error != NULL && error_size > 0)
    avs_loader_append_error(error, error_size, "AviSynth is not loaded in this process (or its file name is not the "
      "usual one, or this system is not supported): use load() with a path");
  return handle;
}

// Address of an exported function, NULL if missing.
// Returned through void*: the POSIX way of converting dlsym's result to a function pointer.
AVSC_LOADER_INLINE void* avs_loader_symbol(void* handle, const char* name)
{
#ifdef AVS_WINDOWS
  FARPROC proc = GetProcAddress((HMODULE)handle, name);
  return *(void**)&proc;
#else
  return dlsym(handle, name);
#endif
}

AVSC_LOADER_INLINE void avs_loader_close(void* handle)
{
  if (handle == NULL)
    return;
#ifdef AVS_WINDOWS
  FreeLibrary((HMODULE)handle);
#else
  dlclose(handle);
#endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
// Fallbacks for functions missing from the loaded AviSynth, see AVSC_LOADER_FALLBACKS below.
// They have the API calling convention (AVSC_CC), their address goes into the function pointers.
// Self-contained: they read the struct fields directly, no API call and no AVSC_INLINE helper.

// VideoInfo: formats which do not exist in the loaded AviSynth, false. The same as the C++ ABI
// (avisynth.h AVS_LinkCallOptDefault) gives.
AVSC_LOADER_INLINE int AVSC_CC avs_loader_fallback_false(const AVS_VideoInfo* p) { (void)p; return 0; }
AVSC_LOADER_INLINE bool AVSC_CC avs_loader_fallback_is_channel_mask_known(const AVS_VideoInfo* p) { (void)p; return 0; }
AVSC_LOADER_INLINE unsigned int AVSC_CC avs_loader_fallback_get_channel_mask(const AVS_VideoInfo* p) { (void)p; return 0; }
AVSC_LOADER_INLINE void AVSC_CC avs_loader_fallback_set_channel_mask(const AVS_VideoInfo* p, bool isChannelMaskKnown, unsigned int dwChannelMask)
{ (void)p; (void)isChannelMaskKnown; (void)dwChannelMask; }
// classic AviSynth 2.6: 8 bit formats only
AVSC_LOADER_INLINE int AVSC_CC avs_loader_fallback_component_size(const AVS_VideoInfo* p) { (void)p; return 1; }
AVSC_LOADER_INLINE int AVSC_CC avs_loader_fallback_bits_per_component(const AVS_VideoInfo* p) { (void)p; return 8; }
AVSC_LOADER_INLINE int AVSC_CC avs_loader_fallback_num_components(const AVS_VideoInfo* p)
{
  switch (p->pixel_type) {
  case AVS_CS_UNKNOWN: return 0;
  case AVS_CS_Y8: return 1;
  case AVS_CS_BGR32: return 4;
  default: return 3; // RGB24, YUY2, YV24, YV16, YV12, I420, YV411
  }
}

// AVS_Value (V11 API) on V3..V10: these have no 64 bit types ('l', 'd'), the values are 32 bit,
// so the V11 functions are exact on them.
AVSC_LOADER_INLINE int AVSC_CC avs_loader_fallback_val_defined(AVS_Value v) { return v.type != 'v'; }
AVSC_LOADER_INLINE int AVSC_CC avs_loader_fallback_val_is_clip(AVS_Value v) { return v.type == 'c'; }
AVSC_LOADER_INLINE int AVSC_CC avs_loader_fallback_val_is_bool(AVS_Value v) { return v.type == 'b'; }
AVSC_LOADER_INLINE int AVSC_CC avs_loader_fallback_val_is_int(AVS_Value v) { return v.type == 'i' || v.type == 'l'; }
AVSC_LOADER_INLINE int AVSC_CC avs_loader_fallback_val_is_long_strict(AVS_Value v) { return v.type == 'l'; }
AVSC_LOADER_INLINE int AVSC_CC avs_loader_fallback_val_is_float(AVS_Value v) { return v.type == 'f' || v.type == 'd' || v.type == 'i' || v.type == 'l'; }
AVSC_LOADER_INLINE int AVSC_CC avs_loader_fallback_val_is_floatf_strict(AVS_Value v) { return v.type == 'f'; }
AVSC_LOADER_INLINE int AVSC_CC avs_loader_fallback_val_is_string(AVS_Value v) { return v.type == 's'; }
AVSC_LOADER_INLINE int AVSC_CC avs_loader_fallback_val_is_array(AVS_Value v) { return v.type == 'a'; }
AVSC_LOADER_INLINE int AVSC_CC avs_loader_fallback_val_is_error(AVS_Value v) { return v.type == 'e'; }
AVSC_LOADER_INLINE int AVSC_CC avs_loader_fallback_get_as_bool(AVS_Value v) { return v.d.boolean ? 1 : 0; }
AVSC_LOADER_INLINE int AVSC_CC avs_loader_fallback_get_as_int(AVS_Value v) { return v.d.integer; }
AVSC_LOADER_INLINE int64_t AVSC_CC avs_loader_fallback_get_as_long(AVS_Value v) { return v.d.integer; } // no 64 bit support yet if fallback needed
AVSC_LOADER_INLINE const char* AVSC_CC avs_loader_fallback_get_as_string(AVS_Value v)
{ return v.type == 's' || v.type == 'e' ? v.d.string : NULL; }
AVSC_LOADER_INLINE double AVSC_CC avs_loader_fallback_get_as_float(AVS_Value v) // no 64 bit 'd' support yet if fallback needed
{ return v.type == 'i' ? (double)v.d.integer : (double)v.d.floating_pt; }
AVSC_LOADER_INLINE const char* AVSC_CC avs_loader_fallback_get_as_error(AVS_Value v) { return v.type == 'e' ? v.d.string : NULL; }
AVSC_LOADER_INLINE const AVS_Value* AVSC_CC avs_loader_fallback_get_as_array(AVS_Value v) { return v.d.array; }
AVSC_LOADER_INLINE AVS_Value AVSC_CC avs_loader_fallback_get_array_elt(AVS_Value v, int index)
{ return v.type == 'a' ? v.d.array[index] : v; }
AVSC_LOADER_INLINE int AVSC_CC avs_loader_fallback_get_array_size(AVS_Value v) { return v.type == 'a' ? v.array_size : 1; }
AVSC_LOADER_INLINE void AVSC_CC avs_loader_fallback_set_to_void(AVS_Value* v) { *v = avs_void; }
AVSC_LOADER_INLINE void AVSC_CC avs_loader_fallback_set_to_bool(AVS_Value* v, int v0)
{ *v = avs_void; v->type = 'b'; v->d.boolean = v0 != 0; }
AVSC_LOADER_INLINE void AVSC_CC avs_loader_fallback_set_to_int(AVS_Value* v, int v0)
{ *v = avs_void; v->type = 'i'; v->d.integer = v0; }
AVSC_LOADER_INLINE void AVSC_CC avs_loader_fallback_set_to_float(AVS_Value* v, float v0)
{ *v = avs_void; v->type = 'f'; v->d.floating_pt = v0; }
AVSC_LOADER_INLINE void AVSC_CC avs_loader_fallback_set_to_string(AVS_Value* v, const char* v0)
{ *v = avs_void; v->type = 's'; v->d.string = v0; }
AVSC_LOADER_INLINE void AVSC_CC avs_loader_fallback_set_to_error(AVS_Value* v, const char* v0)
{ *v = avs_void; v->type = 'e'; v->d.string = v0; }
// no 64 bit types: converted as a V11+ AviSynth converts them for a pre-V11 plugin
AVSC_LOADER_INLINE void AVSC_CC avs_loader_fallback_set_to_double(AVS_Value* v, double d)
{ *v = avs_void; v->type = 'f'; v->d.floating_pt = (float)d; }
AVSC_LOADER_INLINE void AVSC_CC avs_loader_fallback_set_to_long(AVS_Value* v, int64_t l)
{ *v = avs_void; v->type = 'i'; v->d.integer = (int)l; }

// The fallback table. The missing (NULL) entries which can be served without the loaded AviSynth are
// filled, the same way as the C++ ABI (avisynth.h) does. Existing functions are never replaced.
// - VideoInfo, AviSynth+ functions on classic AviSynth 2.6, V13 functions on older AviSynth+:
//   avs_is_444, avs_is_422, avs_is_420, avs_is_y, avs_is_411: avs_is_yv24, avs_is_yv16,
//   avs_is_yv12, avs_is_y8, avs_is_yv411 (the 8 bit formats are the only ones there).
//   avs_component_size: 1, avs_bits_per_component: 8, avs_num_components: from the pixel type.
//   avs_is_yuva, avs_is_planar_rgb(a), avs_is_rgb48/64, avs_is_440, avs_is_410, avs_is_ya, the
//   deprecated avs_is_yuv4xxp16, avs_is_yuv4xxps, avs_is_y16, avs_is_y32: false.
//   V10.1 channel mask: avs_is_channel_mask_known false, avs_get_channel_mask 0,
//   avs_set_channel_mask does nothing.
// - AVS_Value, V11 functions on V3..V10: avs_val_defined, avs_val_is_xxx, avs_get_as_xxx,
//   avs_get_array_elt, avs_get_array_size, avs_set_to_xxx: exact, there are no 64 bit types there.
//   avs_get_as_clip: avs_take_clip. avs_set_to_double and avs_set_to_long give float and 32 bit
//   int, as a V11+ AviSynth converts them for a pre-V11 plugin.
// Not filled, they need the loaded AviSynth: check the version fields (has_at_least_xxx) or the pointer:
// - avs_set_to_array (deep copy owned by the core), frame properties, avs_new_video_frame_p(_a),
//   avs_video_frame_get/amend_pixel_type, avs_pool_xxx, avs_get_var_xxx, the global lock,
//   avs_invoke_try, avs_invoke2/3, function values, avs_create_map.
// - avs_get_cpu_flags_ex: call avs_get_cpu_flags instead when it is NULL (the flags above 32 bits
//   are new in V12 anyway).
// - Functions which exist but work reliably only from a later version (see the list above) are
//   not affected: fallbacks change nothing for them.
//
// The loader supplies the three macros:
// - FALLBACK(name, fn): fn, if name is missing
// - ALIAS(name, older): the older function if name is missing, or avs_loader_fallback_false where
//   the older one is missing too
// - ALIAS_IF_EXISTS(name, older): the older function if name is missing and the older one exists
#define AVSC_LOADER_FALLBACKS(FALLBACK, ALIAS, ALIAS_IF_EXISTS) \
  ALIAS(avs_is_444, avs_is_yv24) \
  ALIAS(avs_is_422, avs_is_yv16) \
  ALIAS(avs_is_420, avs_is_yv12) \
  ALIAS(avs_is_y, avs_is_y8) \
  ALIAS(avs_is_411, avs_is_yv411) \
  FALLBACK(avs_num_components, avs_loader_fallback_num_components) \
  FALLBACK(avs_component_size, avs_loader_fallback_component_size) \
  FALLBACK(avs_bits_per_component, avs_loader_fallback_bits_per_component) \
  FALLBACK(avs_is_yuva, avs_loader_fallback_false) \
  FALLBACK(avs_is_planar_rgb, avs_loader_fallback_false) \
  FALLBACK(avs_is_planar_rgba, avs_loader_fallback_false) \
  FALLBACK(avs_is_rgb48, avs_loader_fallback_false) \
  FALLBACK(avs_is_rgb64, avs_loader_fallback_false) \
  FALLBACK(avs_is_yuv444p16, avs_loader_fallback_false) \
  FALLBACK(avs_is_yuv422p16, avs_loader_fallback_false) \
  FALLBACK(avs_is_yuv420p16, avs_loader_fallback_false) \
  FALLBACK(avs_is_y16, avs_loader_fallback_false) \
  FALLBACK(avs_is_yuv444ps, avs_loader_fallback_false) \
  FALLBACK(avs_is_yuv422ps, avs_loader_fallback_false) \
  FALLBACK(avs_is_yuv420ps, avs_loader_fallback_false) \
  FALLBACK(avs_is_y32, avs_loader_fallback_false) \
  FALLBACK(avs_is_440, avs_loader_fallback_false) \
  FALLBACK(avs_is_410, avs_loader_fallback_false) \
  FALLBACK(avs_is_ya, avs_loader_fallback_false) \
  FALLBACK(avs_is_channel_mask_known, avs_loader_fallback_is_channel_mask_known) \
  FALLBACK(avs_get_channel_mask, avs_loader_fallback_get_channel_mask) \
  FALLBACK(avs_set_channel_mask, avs_loader_fallback_set_channel_mask) \
  FALLBACK(avs_val_defined, avs_loader_fallback_val_defined) \
  FALLBACK(avs_val_is_clip, avs_loader_fallback_val_is_clip) \
  FALLBACK(avs_val_is_bool, avs_loader_fallback_val_is_bool) \
  FALLBACK(avs_val_is_int, avs_loader_fallback_val_is_int) \
  FALLBACK(avs_val_is_long_strict, avs_loader_fallback_val_is_long_strict) \
  FALLBACK(avs_val_is_float, avs_loader_fallback_val_is_float) \
  FALLBACK(avs_val_is_floatf_strict, avs_loader_fallback_val_is_floatf_strict) \
  FALLBACK(avs_val_is_string, avs_loader_fallback_val_is_string) \
  FALLBACK(avs_val_is_array, avs_loader_fallback_val_is_array) \
  FALLBACK(avs_val_is_error, avs_loader_fallback_val_is_error) \
  FALLBACK(avs_get_as_bool, avs_loader_fallback_get_as_bool) \
  ALIAS_IF_EXISTS(avs_get_as_clip, avs_take_clip) \
  FALLBACK(avs_get_as_int, avs_loader_fallback_get_as_int) \
  FALLBACK(avs_get_as_long, avs_loader_fallback_get_as_long) \
  FALLBACK(avs_get_as_string, avs_loader_fallback_get_as_string) \
  FALLBACK(avs_get_as_float, avs_loader_fallback_get_as_float) \
  FALLBACK(avs_get_as_error, avs_loader_fallback_get_as_error) \
  FALLBACK(avs_get_as_array, avs_loader_fallback_get_as_array) \
  FALLBACK(avs_get_array_elt, avs_loader_fallback_get_array_elt) \
  FALLBACK(avs_get_array_size, avs_loader_fallback_get_array_size) \
  FALLBACK(avs_set_to_void, avs_loader_fallback_set_to_void) \
  FALLBACK(avs_set_to_bool, avs_loader_fallback_set_to_bool) \
  FALLBACK(avs_set_to_int, avs_loader_fallback_set_to_int) \
  FALLBACK(avs_set_to_float, avs_loader_fallback_set_to_float) \
  FALLBACK(avs_set_to_string, avs_loader_fallback_set_to_string) \
  FALLBACK(avs_set_to_error, avs_loader_fallback_set_to_error) \
  FALLBACK(avs_set_to_double, avs_loader_fallback_set_to_double) \
  FALLBACK(avs_set_to_long, avs_loader_fallback_set_to_long)

/////////////////////////////////////////////////////////////////////////////////////////////////////
// Version and environment

// Returns the interface version of the AviSynth behind env (e.g. 6, 8, 11, 14), 0 if unknown.
// bugfix (may be NULL) receives the bugfix version (V9 and above, otherwise 0).
// E.g. interface 11 with bugfix 1 is "V11.1" (3.7.5), 14 with bugfix 1 is "V14.1" (3.7.7).
// avs_get_env_property with AVS_AEP_INTERFACE_VERSION is only asked from V9, earlier versions did not know it.
AVSC_LOADER_INLINE int avs_loader_query_version(avs_check_version_func check_version,
  avs_get_env_property_func get_env_property, AVS_ScriptEnvironment* env, int* bugfix)
{
  int version = 0;
  int fix = 0;
  if (check_version != NULL) {
    if (get_env_property != NULL && check_version(env, 9) == 0) {
      version = (int)get_env_property(env, AVS_AEP_INTERFACE_VERSION);
      fix = (int)get_env_property(env, AVS_AEP_INTERFACE_BUGFIX);
    }
    else {
      // Below V9 we can only poll. So also on Win32 3.7.1: it is V9, but avs_get_env_property is
      // exported only with a decorated name there (NULL), so polling starts at 9 (3.7.2 and later export
      // it). No bugfix version.
      int version_probe;
      for (version_probe = 9; version_probe >= 1; version_probe--) {
        if (check_version(env, version_probe) == 0) {
          version = version_probe;
          break;
        }
      }
    }
  }
  if (bugfix != NULL)
    *bugfix = fix;
  return version;
}

// Nonzero if version.bugfix is at least v.b
AVSC_LOADER_INLINE int avs_loader_at_least(int version, int bugfix, int v, int b)
{
  return version > v || (version == v && bugfix >= b);
}

// Nonzero if environment creation failed. Refused version requests can return a non-NULL wrapper
// with a NULL core pointer; before V14, avs_get_error did not report this failure.
// has_env_creation_error_reporting comes from a direct lookup of the V14-only avs_create_map export,
// not the function table or a version query (which requires a valid core). This assumes future cores
// retain that export and the creation-error reporting introduced in V14.
AVSC_LOADER_INLINE int avs_loader_env_failed(avs_get_error_func get_error, AVS_ScriptEnvironment* env,
  int has_env_creation_error_reporting)
{
  if (env == NULL)
    return 1;
  if (get_error != NULL && get_error(env) != NULL)
    return 1;
  // Modern wrappers may change layout. Without the error accessor, they cannot be validated.
  if (has_env_creation_error_reporting)
    return get_error == NULL;
  {
    // AVS_ScriptEnvironment is opaque outside BUILDING_AVSCORE, so env->env is unavailable.
    // Legacy cores only: rely on their historical layout to detect silent NULL-core failures.
    typedef struct { const void* core; const char* error; } AVS_ScriptEnvironment_Layout;
    return ((const AVS_ScriptEnvironment_Layout*)env)->core == NULL;
  }
}

// Hosts: the interface versions requested when creating the script environment, the first accepted
// one is kept. Two different versions are involved:
// - the header's AVISYNTH_INTERFACE_VERSION (compile time): what this caller understands, the first
//   request. Rebuilding with a newer avisynth_c.h raises it automatically; then extend the
//   has_at_least_xxx steps and the function list if needed. An avisynth_c.h older than V14 does not
//   compile with this loader (missing function types).
// - the loaded DLL's interface version (run time): requests above it are refused.
// The requested version matters: it tells AviSynth what the caller understands.
// Below 11 AviSynth converts double values to float and long (int64) values to 32 bit int,
// even if it supports them itself.
// The request list follows historical API milestones, not distinct environment implementations.
// A refused request is cheap (no core environment is built):
// - AVISYNTH_INTERFACE_VERSION: what this caller understands
// - 11: a V11..V13 AviSynth: 64 bit AVS_Value types. Any version from 11 up to the DLL's interface version
//   gives the same environment.
// - 9: a V9/V10 AviSynth: working C-interface frame property access.
// - 8: a V8 AviSynth: frame property API entries exist but are not reliably usable.
//   The environment is the same as with 6; the step only follows the has_at_least_v8 feature level.
// - 6 (classic): older AviSynth. Any version below 11 gives the same environment; bugfix versions
//   cannot be requested at all.
// - 3: AviSynth 2.5
// The exact version and the has_at_least_xxx steps are then queried through the environment.

// FIXME Question: how coarse is this handled inside Avisynth? V11 (64 bit AVSValue support) is surely a milestone
// Also V8, as the first non-classic-Avisynth IF version. But do we need to request 9 in between?
// If 9 does not yield different internal environment inside avisynth, we can remove the 9
#define AVSC_LOADER_ENV_REQUESTS { AVISYNTH_INTERFACE_VERSION, 11, 9, 8, AVISYNTH_INTERFACE_CLASSIC_VERSION, 3 }
// 64 bit AVS_Value types (double, long) reach a C caller only from interface version 11:
// - hosts: the environment requested with version 11 or above (up to the DLL's version) gets them;
//   below 11 the core gives a pre-V11 environment, which converts them to float and 32 bit int.
//   The requested version decides, not the version of the DLL.
// - plugins: the requested version plays no part. AviSynth calls avisynth_c_plugin_init2 if the plugin
//   exports it, then the plugin's functions get them; through avisynth_c_plugin_init they are converted.
//   Only a V11+ AviSynth (3.7.4+) knows init2, the version check is just a guard there.
#define AVSC_LOADER_64BIT_VALUES_VERSION 11

/////////////////////////////////////////////////////////////////////////////////////////////////////
// avs_add_function(_r) proxy for AviSynth+ 3.7.4 and 3.7.5 (V11, V11.1)
//
// The bug: these cores store each registration ({user_data, apply, apply_r}) with SaveString, whose
// cache compared entries with strncmp. strncmp stops at the first zero byte, so the comparison ended
// inside user_data: NULL, or the same pointer for all functions on x64 (its top bytes are zero), made
// the registrations of different functions equal. When two of them fell into the same hash bucket,
// the later function got the earlier one's callback: a script call ran another function of the
// plugin. Fixed in 3.7.6 (V14, commit 8a4edc62); reported from the masktools C port.
// Only the builds made with MSVC or clang-cl (the official Windows ones) are affected: the MSVC STL
// compares the keys of a bucket with the equality alone; libstdc++ and libc++ (MinGW, Linux, macOS)
// compare the stored full hash first, which covers all bytes. The loader cannot tell how a core was
// built, so it works around it on every V11/V11.1 core; it costs nothing on the others.
//
// The proxy: on a V11/V11.1 core the loader's avs_add_function and avs_add_function_r entries point
// here (set with the version, avs_loader_set_add_function_proxy). Each distinct (apply, user_data)
// gets a slot and is registered through a loader trampoline (a small function in between which only
// forwards the call) with a tag pointer as user_data. The tag pointers have distinct, nonzero low
// bytes, so strncmp differs at the first byte. The trampoline calls the plugin's function with its
// own user_data: the plugin sees no difference.
// - Up to 255 slots; a registration over that goes to the core directly.
// - Other cores: the entries are the core's functions, nothing changes.
// - One core per process: the proxy keeps one core function pointer (a plugin has one core anyway).
// - C: each source file has its own copy of the state (static inline); a library always uses the
//   copy of the file which set its version, so this works with the loader in several files too.
// - Two plugins using the loader on one core: their slot k tags have the same low byte. strncmp then
//   goes on to the second byte, which differs unless the two tag blocks happen to match there too, and
//   the two entries must also share a hash bucket: very unlikely, not worth a bigger tag search.
// - Little endian assumed: the low byte comes first in memory. A big-endian core would compare the
//   zero high byte first, the tags would not help, but no MSVC build is big-endian (not affected).

typedef struct avs_loader_apply_slot {
  AVS_ApplyFunc apply;
  AVS_ApplyFuncR apply_r;
  void* user_data;
} avs_loader_apply_slot;

#define AVSC_LOADER_APPLY_SLOTS 255

typedef struct avs_loader_proxy_state {
  avs_add_function_func core_add_function;
  avs_add_function_r_func core_add_function_r;
  volatile long lock;
  int count;
  unsigned char* tag_base; // tag of slot k: tag_base + k + 1 (low byte k + 1)
  avs_loader_apply_slot slots[AVSC_LOADER_APPLY_SLOTS];
  unsigned char tag_buffer[768]; // contains a 256 byte block for the tags, see avs_loader_proxy_slot
} avs_loader_proxy_state;

// Zero-initialized static (no constructor): C++ has one per program, C one per source file.
AVSC_LOADER_INLINE avs_loader_proxy_state* avs_loader_proxy(void)
{
  static avs_loader_proxy_state state;
  return &state;
}

AVSC_LOADER_INLINE void avs_loader_proxy_lock(avs_loader_proxy_state* s)
{
#ifdef AVS_WINDOWS
  while (InterlockedExchange((volatile LONG*)&s->lock, 1) != 0) {}
#else
  while (__atomic_exchange_n(&s->lock, 1, __ATOMIC_ACQUIRE) != 0) {}
#endif
}

AVSC_LOADER_INLINE void avs_loader_proxy_unlock(avs_loader_proxy_state* s)
{
#ifdef AVS_WINDOWS
  InterlockedExchange((volatile LONG*)&s->lock, 0);
#else
  __atomic_store_n(&s->lock, 0, __ATOMIC_RELEASE);
#endif
}

// The tag (user_data for the core) of the slot of (apply or apply_r, user_data), NULL if all slots are used.
AVSC_LOADER_INLINE void* avs_loader_proxy_slot(AVS_ApplyFunc apply, AVS_ApplyFuncR apply_r, void* user_data)
{
  avs_loader_proxy_state* s = avs_loader_proxy();
  void* tag = NULL;
  int i;
  avs_loader_proxy_lock(s);
  if (s->tag_base == NULL) {
    // A 256 byte aligned address whose second byte is nonzero too, so that strncmp also gets past
    // it when two plugins use the same low bytes. Of two neighbouring 256 byte boundaries at most
    // one has a zero second byte; in 768 bytes there are two, each followed by 255 bytes for tags.
    unsigned char* p = s->tag_buffer;
    while (((uintptr_t)p & 0xFF) != 0 || (((uintptr_t)p >> 8) & 0xFF) == 0)
      ++p;
    s->tag_base = p;
  }
  for (i = 0; i < s->count; i++) {
    if (s->slots[i].apply == apply && s->slots[i].apply_r == apply_r && s->slots[i].user_data == user_data)
      break;
  }
  if (i == s->count && s->count < AVSC_LOADER_APPLY_SLOTS) {
    s->slots[i].apply = apply;
    s->slots[i].apply_r = apply_r;
    s->slots[i].user_data = user_data;
    s->count++;
  }
  if (i < s->count)
    tag = s->tag_base + i + 1;
  avs_loader_proxy_unlock(s);
  return tag;
}

AVSC_LOADER_INLINE const avs_loader_apply_slot* avs_loader_proxy_slot_of(void* tag)
{
  avs_loader_proxy_state* s = avs_loader_proxy();
  return &s->slots[(unsigned char*)tag - s->tag_base - 1];
}

AVSC_LOADER_INLINE AVS_Value AVSC_CC avs_loader_apply_trampoline(AVS_ScriptEnvironment* env, AVS_Value args, void* tag)
{
  const avs_loader_apply_slot* slot = avs_loader_proxy_slot_of(tag);
  return slot->apply(env, args, slot->user_data);
}

AVSC_LOADER_INLINE void AVSC_CC avs_loader_apply_r_trampoline(AVS_ScriptEnvironment* env, AVS_Value* ret, AVS_Value args, void* tag)
{
  const avs_loader_apply_slot* slot = avs_loader_proxy_slot_of(tag);
  slot->apply_r(env, ret, args, slot->user_data);
}

AVSC_LOADER_INLINE int AVSC_CC avs_loader_add_function_proxy(AVS_ScriptEnvironment* env, const char* name,
  const char* params, AVS_ApplyFunc apply, void* user_data)
{
  void* tag = avs_loader_proxy_slot(apply, NULL, user_data);
  avs_loader_proxy_state* s = avs_loader_proxy();
  if (tag == NULL)
    return s->core_add_function(env, name, params, apply, user_data);
  return s->core_add_function(env, name, params, avs_loader_apply_trampoline, tag);
}

AVSC_LOADER_INLINE int AVSC_CC avs_loader_add_function_r_proxy(AVS_ScriptEnvironment* env, const char* name,
  const char* params, AVS_ApplyFuncR apply, void* user_data)
{
  void* tag = avs_loader_proxy_slot(NULL, apply, user_data);
  avs_loader_proxy_state* s = avs_loader_proxy();
  if (tag == NULL)
    return s->core_add_function_r(env, name, params, apply, user_data);
  return s->core_add_function_r(env, name, params, avs_loader_apply_r_trampoline, tag);
}

// Called with the detected version: points the two entries to the proxy on V11/V11.1, back to the
// core's functions otherwise.
AVSC_LOADER_INLINE void avs_loader_set_add_function_proxy(avs_add_function_func* add_function,
  avs_add_function_r_func* add_function_r, int version)
{
  avs_loader_proxy_state* s = avs_loader_proxy();
  if (version == 11) {
    if (*add_function != NULL && *add_function != avs_loader_add_function_proxy) {
      s->core_add_function = *add_function;
      *add_function = avs_loader_add_function_proxy;
    }
    if (*add_function_r != NULL && *add_function_r != avs_loader_add_function_r_proxy) {
      s->core_add_function_r = *add_function_r;
      *add_function_r = avs_loader_add_function_r_proxy;
    }
  }
  else {
    if (*add_function == avs_loader_add_function_proxy)
      *add_function = s->core_add_function;
    if (*add_function_r == avs_loader_add_function_r_proxy)
      *add_function_r = s->core_add_function_r;
  }
}

#endif // AVISYNTH_C_LOADER_COMMON_H
