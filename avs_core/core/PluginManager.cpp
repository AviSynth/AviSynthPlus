#include "PluginManager.h"
#include <avisynth.h>
#include <unordered_set>
#include <avisynth_c.h>
#include "strings.h"
#include "InternalEnvironment.h"
#include <cassert>
#include "function.h"
#include <avs/filesystem.h>

#ifdef AVS_WINDOWS
  #include <avs/win.h>
#else
  #include <avs/posix.h>
#endif

#ifdef AVS_WINDOWS
    #include <imagehlp.h>
#endif
#include "parser/script.h"
#include "parser/expression.h" // TODO we only need FunctionInstance from here

typedef const char* (__stdcall *AvisynthPluginInit3Func)(IScriptEnvironment* env, const AVS_Linkage* const vectors);
typedef const char* (__stdcall *AvisynthPluginInit2Func)(IScriptEnvironment_Avs25* env);
typedef const char* (AVSC_CC *AvisynthCPluginInitFunc)(AVS_ScriptEnvironment* env);

#ifdef AVS_WINDOWS // only Windows has a registry we care about
const char RegAvisynthKey[] = "Software\\Avisynth";
#if defined (__GNUC__)
const char RegPluginDirPlus_GCC[] = "PluginDir+GCC";
#if defined(X86_32)
  #define GCC_WIN32
#endif
#else
const char RegPluginDirClassic[] = "PluginDir2_5";
const char RegPluginDirPlus[] = "PluginDir+";
#endif
#endif // AVS_WINDOWS

#ifdef AVS_POSIX
#include <dlfcn.h>
// Redifining these is easier than adding several ifdefs.
#define HMODULE void*
#define FreeLibrary dlclose
#if defined(AVS_MACOS) || defined(AVS_BSD)
#include <sys/syslimits.h>
#endif
#endif

#ifdef AVS_MACOS
#include <mach-o/dyld.h>
#endif

/*
---------------------------------------------------------------------------------
---------------------------------------------------------------------------------
                                 Static helpers
---------------------------------------------------------------------------------
---------------------------------------------------------------------------------
*/

void IFunction::AddRef() {
  InterlockedIncrement(&refcnt);
}

void IFunction::Release() {
  if (InterlockedDecrement(&refcnt) <= 0)
    delete this;
}

#ifdef AVS_WINDOWS // translate to Linux error handling
// Translates a Windows error code to a human-readable text message.
static std::string GetLastErrorText(DWORD nErrorCode)
{
  char* msg;
  // Ask Windows to prepare a standard message for a GetLastError() code:
  if (0 == FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL, nErrorCode, 0, (LPSTR)&msg, 0, NULL))
    return("Unknown error");
  else
  {
    std::string ret(msg);
    LocalFree(msg);
    return ret;
  }
}

static bool GetRegString(HKEY rootKey, const char path[], const char entry[], std::string *result) {
    HKEY AvisynthKey;

    if (RegOpenKeyEx(rootKey, path, 0, KEY_READ, &AvisynthKey))
      return false;

    DWORD size;
    if (ERROR_SUCCESS != RegQueryValueEx(AvisynthKey, entry, 0, 0, 0, &size)) {
      RegCloseKey(AvisynthKey); // Dave Brueck - Dec 2005
      return false;
    }

    char* retStr = new(std::nothrow) char[size];
    if ((retStr == NULL) || (ERROR_SUCCESS != RegQueryValueEx(AvisynthKey, entry, 0, 0, (LPBYTE)retStr, &size))) {
      delete[] retStr;
      RegCloseKey(AvisynthKey); // Dave Brueck - Dec 2005
      return false;
    }
    RegCloseKey(AvisynthKey); // Dave Brueck - Dec 2005

    *result = std::string(retStr);
    delete[] retStr;
    return true;
}

#endif // AVS_WINDOWS

static std::string GetFullPathNameWrap(const std::string &f)
{
  return fs::absolute(fs::path(f).lexically_normal()).generic_string();
}

static bool IsParameterTypeSpecifier(char c) {
  switch (c) {
  case 'b': case 'i': case 'f': case 's': case 'c': case '.':
  case 'n':
  case 'a': // Arrays as function parameters
      return true;
    default:
      return false;
  }
}

static bool IsParameterTypeModifier(char c) {
  switch (c) {
    case '+': case '*':
      return true;
    default:
      return false;
  }
}

static bool IsValidParameterString(const char* p) {
  // does not check for logical errors such as
  // when unnamed untyped array (.+) is followed by additional parameters
  int state = 0;
  char c;
  while ((c = *p++) != '\0' && state != -1) {
    switch (state) {
      case 0:
        if (IsParameterTypeSpecifier(c)) {
          state = 1;
        }
        else if (c == '[') {
          state = 2;
        }
        else {
          state = -1;
        }
        break;

      case 1:
        if (IsParameterTypeSpecifier(c)) {
          // do nothing; stay in the current state
        }
        else if (c == '[') {
          state = 2;
        }
        else if (IsParameterTypeModifier(c)) {
          state = 0;
        }
        else {
          state = -1;
        }
        break;

      case 2:
        if (c == ']') {
          state = 3;
        }
        else {
          // do nothing; stay in the current state
        }
        break;

      case 3:
        if (IsParameterTypeSpecifier(c)) {
          state = 1;
        }
        else {
          state = -1;
        }
        break;
    }
  }

  // states 0, 1 are the only ending states we accept
  return state == 0 || state == 1;
}

/*
---------------------------------------------------------------------------------
---------------------------------------------------------------------------------
                                 AVSFunction
---------------------------------------------------------------------------------
---------------------------------------------------------------------------------
*/

AVSFunction::AVSFunction(void*) :
    AVSFunction(NULL, NULL, NULL, NULL, NULL, NULL, false)
{}

AVSFunction::AVSFunction(const char* _name, const char* _plugin_basename, const char* _param_types, apply_func_t _apply) :
    AVSFunction(_name, _plugin_basename, _param_types, _apply, NULL, NULL, false)
{}

AVSFunction::AVSFunction(const char* _name, const char* _plugin_basename, const char* _param_types, apply_func_t _apply, void *_user_data) :
    AVSFunction(_name, _plugin_basename, _param_types, _apply, _user_data, NULL, false)
{}

AVSFunction::AVSFunction(const char* _name, const char* _plugin_basename, const char* _param_types, apply_func_t _apply, void *_user_data, const char* _dll_path, bool _isAvs25) :
    Function()
{
  apply = _apply;
  user_data = _user_data;
  isAvs25 = _isAvs25;

    if (NULL != _dll_path)
    {
        size_t len = strlen(_dll_path);
        auto tmp = new char[len + 1];
        memcpy(tmp, _dll_path, len);
        tmp[len] = 0;
        dll_path = tmp;
    }

    if (NULL != _name)
    {
        size_t len = strlen(_name);
        auto tmp = new char[len + 1];
        memcpy(tmp, _name, len);
        tmp[len] = 0;
        name = tmp;
    }

    if ( NULL != _param_types )
    {
        size_t len = strlen(_param_types);
        auto tmp = new char[len+1];
        memcpy(tmp, _param_types, len);
        tmp[len] = 0;
        param_types = tmp;
    }

    if ( NULL != _name )
    {
        std::string cn(NULL != _plugin_basename ? _plugin_basename : "");
        cn.append("_").append(_name);
        auto tmp = new char[cn.size()+1];
        memcpy(tmp, cn.c_str(), cn.size());
        tmp[cn.size()] = 0;
        canon_name = tmp;
    }
}

AVSFunction::~AVSFunction()
{
    delete [] canon_name;
    delete [] name;
    delete [] param_types;
    delete [] dll_path;
}

bool AVSFunction::empty() const
{
    return NULL == name;
}

bool AVSFunction::IsScriptFunction(const Function* func)
{
  return ( (func->apply == &(FunctionInstance::Execute_))
          || (func->apply == &(ScriptFunction::Execute))
          || (func->apply == &Eval)
          || (func->apply == &EvalOop)
          || (func->apply == &Import)
        );
}

bool AVSFunction::SingleTypeMatch(char type, const AVSValue& arg, bool strict) {
  switch (type) {
    case '.': return true;
    case 'b': return arg.IsBool();
    case 'i': return arg.IsInt();
    case 'f': return arg.IsFloat() && (!strict || !arg.IsInt());
    case 's': return arg.IsString();
    case 'c': return arg.IsClip();
    case 'n': return arg.IsFunction();
    case 'a': return arg.IsArray(); // PF 161028 AVS+ script arrays
    default:  return false;
  }
}

bool AVSFunction::SingleTypeMatchArray(char type, const AVSValue& arg, bool strict) {
  if (!arg.IsArray())
    return false;

  for (int i = 0; i < arg.ArraySize(); i++)
  {
    if (!SingleTypeMatch(type, arg[i], strict))
      return false;
  }

  return true;
}


bool AVSFunction::TypeMatch(const char* param_types, const AVSValue* args, size_t num_args, bool strict, IScriptEnvironment* env) {

  bool optional = false;

  /* examples
  { "StackHorizontal", BUILTIN_FUNC_PREFIX, "cc+", StackHorizontal::Create },
  { "Spline", BUILTIN_FUNC_PREFIX, "[x]ff+[cubic]b", Spline },
  { "Select",   BUILTIN_FUNC_PREFIX, "i.+", Select },
  { "Array", BUILTIN_FUNC_PREFIX, ".*", ArrayCreate },
  { "IsArray",   BUILTIN_FUNC_PREFIX, ".", IsArray },
  { "ArrayGet",  BUILTIN_FUNC_PREFIX, ".s", ArrayGet },
  { "ArrayGet",  BUILTIN_FUNC_PREFIX, ".i+", ArrayGet }, // .+i+ syntax is not possible.
  { "ArraySize", BUILTIN_FUNC_PREFIX, ".", ArraySize },
  */

  // arguments are provided in a flattened way (flattened=array elements extracted)
  // e.g.    string array is provided here string,string,string

  // '*' or '+' to indicate "zero or more" or "one or more"
  // '.' matches a single argument of any type. To match multiple arguments of any type, use ".*" or ".+".

  size_t i = 0;
  while (i < num_args) {

    if (*param_types == '\0') {
      // more args than params
      return false;
    }

    if (*param_types == '[') {
      // named arg: skip over the name
      param_types = strchr(param_types+1, ']');
      if (param_types == NULL) {
        env->ThrowError("TypeMatch: unterminated parameter name (bug in filter)");
      }

      ++param_types;
      optional = true;

      if (*param_types == '\0') {
        env->ThrowError("TypeMatch: no type specified for optional parameter (bug in filter)");
      }
    }

    if (param_types[1] == '*') {
      // skip over initial test of type for '*' (since zero matches is ok)
      ++param_types;
    }

    switch (*param_types) {
      case 'b': case 'i': case 'f': case 's': case 'c':
      case 'n':
      case 'a':
        // PF 2016: 'a' is special letter for script arrays, but if possible we are using .* and .+ (legacy Avisynth style) instead
        // Note (2021): 'a' is still not used
        // cons: no z or nz (+ or *) possibility
        //       no type check (array of int)
        //       cannot be used in plugins which are intended to work for Avisynth 2.6 Classic. ("a" is invalid in function signature -> plugin load error)
        // pros: clean syntax, accept _only_ arrays when required, no comma-delimited-list-to-array option (like in old Avisynth syntax)
        // array arguments are not necessarily "flattened" when TypeMatch is called.
        if (param_types[1] == '+' // parameter indicates an array-type args[i]
          && args[i].IsArray() // allow single e.g. 'c' parameter in place of a 'c+' requirement
          && *param_types != 'a'
          )
        {
          ++param_types; // will be found in case '+' section
          break;
        }

        if (   (!optional || args[i].Defined())
            && !SingleTypeMatch(*param_types, args[i], strict))
          return false;

        ++param_types;
        ++i;
        break;

      case '.': // any type
        // This allows even an array in the place of a "."
        // Use cases: IsArray "." can be fed with any AvsValue. ArrayGet ".i+" requires an array in the place of "." as well.
        // Array-ness of such AVSValue parameters can be checked in the function itself.
        ++param_types;
        ++i;
        break;
      case '+': case '*':
        // check array content type if required
        if (args[i].IsArray() && param_types[-1] != '.') {
          // A script can provide an array argument in an direct array-type variable.
          // e.g. a user defined script function function Summa(int_array "x") will translate to "[x]i*"
          // parameter list. Passing an integer array directly e.g. [1,2,3] will be handled here.
          // All elements in the array should match with the type character preceding '+' or '*'
          // (There was another option in legacy AviSynth: the comma separated values e.g. 1,2,3
          // could be recognized and moved to an unnamed array, this is check later)
          if (!SingleTypeMatchArray(param_types[-1], args[i], strict))
            return false;
          ++param_types;
          ++i;
        }
        else
        // Legacy Avisynth array check.
        // Array of arguments of known types last until an argument of another type is found.
        // This is the reason why an .+ or .* (array of anything) must only appear at the end
        // of the parameter list since we cannot detect type-change in an any-type argument sequence.
        if (!SingleTypeMatch(param_types[-1], args[i], strict)) {
          // we're done with the + or *, parameter type has been changed
          ++param_types;
        }
        else {
          // parameter type matched, step parameter pointer but leave type pointer
          ++i;
        }
        break;
      default:
        env->ThrowError("TypeMatch: invalid character in parameter list (bug in filter)");
    }
  }

  // We're out of args.  We have a match if one of the following is true:
  // (a) we're out of params.
  // (b) remaining params are named i.e. optional.
  // (c) we're at a '+' or '*' and any remaining params are optional.

  if (*param_types == '+'  || *param_types == '*')
    param_types += 1;

  if (*param_types == '\0' || *param_types == '[')
    return true;

  while (param_types[1] == '*') {
    param_types += 2;
    if (*param_types == '\0' || *param_types == '[')
      return true;
  }

  return false;
}

bool AVSFunction::ArgNameMatch(const char* param_types, size_t args_names_count, const char* const* arg_names) {

  for (size_t i=0; i<args_names_count; ++i) {
    if (arg_names[i]) {
      bool found = false;
      size_t len = strlen(arg_names[i]);
      for (const char* p = param_types; *p; ++p) {
        if (*p == '[') {
          p += 1;
          const char* q = strchr(p, ']');
          if (!q) return false;
          if (len == q-p && !_strnicmp(arg_names[i], p, q-p)) {
            found = true;
            break;
          }
          p = q+1;
        }
      }
      if (!found) return false;
    }
  }
  return true;
}

/*
---------------------------------------------------------------------------------
---------------------------------------------------------------------------------
                                 PluginFile
---------------------------------------------------------------------------------
---------------------------------------------------------------------------------
*/


struct PluginFile
{
  std::string FilePath;             // Fully qualified, canonical file path
  std::string BaseName;             // Only file name, without extension
  HMODULE Library;                  // LoadLibrary handle
  bool isAvs25;

  PluginFile(const std::string &filePath);
};

PluginFile::PluginFile(const std::string &filePath) :
  FilePath(GetFullPathNameWrap(filePath)), BaseName(), Library(NULL), isAvs25(false)
{
  // Turn all '\' into '/'
  replace(FilePath, '\\', '/');

  // Find position of dot in extension
  size_t dot_pos = FilePath.rfind('.');

  // Find position of last directory slash
  size_t slash_pos = FilePath.rfind('/');

  // Extract basename
  if ((dot_pos != std::string::npos) && (slash_pos != std::string::npos))
  {// we have both a slash and a dot
    if (dot_pos > slash_pos)
      BaseName = FilePath.substr(slash_pos+1, dot_pos - slash_pos - 1);
    else
      BaseName = FilePath.substr(slash_pos+1, std::string::npos);
  }
  else if ((dot_pos == std::string::npos) && (slash_pos != std::string::npos))
  {// we have a slash but no dot
    // Extract basename
    BaseName = FilePath.substr(slash_pos+1, std::string::npos);
  }
  else
  {// everything else
    // Because we have used GetFullPathName, FilePath should contain an absolute path,
    // meaning that this case should be unreachable, but the devil never sleeps.
    assert(0);
    BaseName = FilePath;
  }
}

/*
---------------------------------------------------------------------------------
---------------------------------------------------------------------------------
                                 PluginManager
---------------------------------------------------------------------------------
---------------------------------------------------------------------------------
*/

PluginManager::PluginManager(InternalEnvironment* env) :
  Env(env), PluginInLoad(NULL), AutoloadExecuted(false), Autoloading(false)
{
  env->SetGlobalVar("$PluginFunctions$", AVSValue(""));
}

void PluginManager::ClearAutoloadDirs()
{
  if (AutoloadExecuted)
    Env->ThrowError("Cannot modify directory list after the autoload procedure has already executed.");

  AutoloadDirs.clear();
}

void PluginManager::AddAutoloadDir(const std::string &dirPath, bool toFront)
{
  if (AutoloadExecuted)
    Env->ThrowError("Cannot modify directory list after the autoload procedure has already executed.");

  std::string dir(dirPath);

#if !defined(AVS_BSD)
// Any use of /proc should be avoided on BSD, since
// most of them have removed it or discourage its use.
// Thankfully, it actually looks like the need for it
// is to simply populate the PROGRAMDIR variable for
// AddAutoloadDirs, but on POSIX systems this variable
// should probably not be expected to be as flexible
// as it is on Windows, negating the need for pulling
// it out programmatically.  Since the macOS and Linux
// forms of the code still function, leave those alone.
#ifdef AVS_WINDOWS
  // get folder of our executable
  TCHAR ExeFilePath[AVS_MAX_PATH];
  memset(ExeFilePath, 0, sizeof(ExeFilePath[0])*AVS_MAX_PATH);  // WinXP does not terminate the result of GetModuleFileName with a zero, so me must zero our buffer
  GetModuleFileName(NULL, ExeFilePath, AVS_MAX_PATH);
#else // AVS_POSIX
  std::string ExeFilePath;
  char buf[PATH_MAX + 1] {};
#ifdef AVS_LINUX
  if (readlink("/proc/self/exe", buf, sizeof(buf) - 1) != -1)
#elif defined(AVS_MACOS)
  uint32_t size = sizeof(buf) - 1;
  if (_NSGetExecutablePath(buf, &size) == 0)
#endif // AVS_LINUX
  {
    ExeFilePath = buf;
  }
#endif
  std::string ExeFileDir(ExeFilePath);
  replace(ExeFileDir, '\\', '/');
#ifndef AVS_HAIKU
// Haiku's exe path stuff differs enough from the *nix OSes
// that it fails spectacularly when loading the library in a client
// like avs2yuv or FFmpeg.  Try to skip this for now and hope
// this doesn't cause more errors.
  ExeFileDir = ExeFileDir.erase(ExeFileDir.rfind('/'), std::string::npos);
#endif
#endif // !AVS_BSD

  // variable expansion
  replace_beginning(dir, "SCRIPTDIR", Env->GetVarString("$ScriptDir$", ""));
  replace_beginning(dir, "MAINSCRIPTDIR", Env->GetVarString("$MainScriptDir$", ""));
#if !defined(AVS_BSD)
  replace_beginning(dir, "PROGRAMDIR", ExeFileDir);
#endif

  std::string plugin_dir;
#ifdef AVS_WINDOWS
#if defined (__GNUC__)
  if (GetRegString(HKEY_CURRENT_USER, RegAvisynthKey, RegPluginDirPlus_GCC, &plugin_dir))
    replace_beginning(dir, "USER_PLUS_PLUGINS", plugin_dir);
  if (GetRegString(HKEY_LOCAL_MACHINE, RegAvisynthKey, RegPluginDirPlus_GCC, &plugin_dir))
    replace_beginning(dir, "MACHINE_PLUS_PLUGINS", plugin_dir);
#else
  // note: if e.g HKCU/PluginDir+ does not exist, USER_PLUS_PLUGINS as a string remain in search path
  if (GetRegString(HKEY_CURRENT_USER, RegAvisynthKey, RegPluginDirPlus, &plugin_dir))
    replace_beginning(dir, "USER_PLUS_PLUGINS", plugin_dir);
  if (GetRegString(HKEY_LOCAL_MACHINE, RegAvisynthKey, RegPluginDirPlus, &plugin_dir))
    replace_beginning(dir, "MACHINE_PLUS_PLUGINS", plugin_dir);
  if (GetRegString(HKEY_CURRENT_USER, RegAvisynthKey, RegPluginDirClassic, &plugin_dir))
    replace_beginning(dir, "USER_CLASSIC_PLUGINS", plugin_dir);
  if (GetRegString(HKEY_LOCAL_MACHINE, RegAvisynthKey, RegPluginDirClassic, &plugin_dir))
    replace_beginning(dir, "MACHINE_CLASSIC_PLUGINS", plugin_dir);
#endif
#endif

  // replace backslashes with forward slashes
  replace(dir, '\\', '/');

  // append terminating slash if needed
  if (dir.size() > 0 && dir[dir.size()-1] != '/')
    dir.append("/");

  // remove double slashes
  while(replace(dir, "//", "/"));

  if (toFront)
    AutoloadDirs.insert(AutoloadDirs.begin(), GetFullPathNameWrap(dir));
  else
    AutoloadDirs.push_back(GetFullPathNameWrap(dir));
}

void PluginManager::AutoloadPlugins()
{
  if (AutoloadExecuted)
    return;

  AutoloadExecuted = true;
  Autoloading = true;

  // Load binary plugins
  for (const std::string& dir : AutoloadDirs)
  {
    std::error_code ec;

#ifdef AVS_POSIX
#ifdef AVS_MACOS
    const char* binaryFilter = ".dylib";
#else
    const char* binaryFilter = ".so";
#endif
#else
    const char* binaryFilter = ".dll";
#endif
    for (auto& file : fs::directory_iterator(dir, fs::directory_options::skip_permission_denied | fs::directory_options::follow_directory_symlink, ec))
    {
      const bool extensionsMatch =
#ifdef AVS_POSIX
      file.path().extension() == binaryFilter; // case sensitive
#else
      streqi(file.path().extension().generic_string().c_str(), binaryFilter);  // case insensitive
#endif

      if (extensionsMatch)
      {
        PluginFile p(concat(dir, file.path().filename().generic_string()));

        // Search for loaded plugins with the same base name.
        bool same_found = false;
        for (size_t i = 0; i < AutoLoadedPlugins.size(); ++i)
        {
#ifdef AVS_POSIX
          if (AutoLoadedPlugins[i].BaseName == p.BaseName) // case insentitive
#else
          if (streqi(AutoLoadedPlugins[i].BaseName.c_str(), p.BaseName.c_str()))
#endif
          {
            // Prevent loading a plugin with a basename that is
            // already loaded (from another autoload folder).
            same_found = true;
            break;
          }
        }

        if (same_found)
          continue;

        // Try to load plugin
        AVSValue dummy;
        LoadPlugin(p, false, &dummy);
      }
    }

    const char* scriptFilter = ".avsi";
    for (auto& file : fs::directory_iterator(dir, fs::directory_options::skip_permission_denied | fs::directory_options::follow_directory_symlink, ec)) // and not recursive_directory_iterator
    {
      const bool extensionsMatch =
#ifdef AVS_POSIX
        file.path().extension() == scriptFilter; // case sensitive
#else
        streqi(file.path().extension().generic_string().c_str(), scriptFilter);  // case insensitive
#endif

      if (extensionsMatch)
      {
        CWDChanger cwdchange(dir.c_str());

        PluginFile p(concat(dir, file.path().filename().generic_string()));

        // Search for loaded avsi scripts with the same base name.
        bool same_found = false;
        for (size_t i = 0; i < AutoLoadedImports.size(); ++i)
        {
#ifdef AVS_POSIX
          if (AutoLoadedImports[i].BaseName == p.BaseName) // case insensitive
#else
          if (streqi(AutoLoadedImports[i].BaseName.c_str(), p.BaseName.c_str()))
#endif
          {
            // Prevent loading an avsi script with a basename that is
            // already loaded (from another autoload folder).
            same_found = true;
            break;
          }
        }

        if (same_found)
          continue;

        // Try to load script
        Env->Invoke("Import", p.FilePath.c_str()); // FIXME: utf8?
        AutoLoadedImports.push_back(p);
      }
    }
  }

#if 0
  // pre c++17 plugin enumeration methods
  const char *binaryFilter = "*.dll";
  const char *scriptFilter = "*.avsi";

  // Load binary plugins
  for (const std::string& dir : AutoloadDirs)
  {
    // Append file search filter to directory path
    std::string filePattern = concat(dir, binaryFilter);

  // Iterate through all files in directory
    _finddata_t fileData;
    intptr_t hFind = _findfirst(filePattern.c_str(), &fileData);
    for (bool bContinue = hFind;
      bContinue;
      bContinue = (_findnext(hFind, &fileData) != -1)
      )
    {
      if ((fileData.attrib & _A_SUBDIR) == 0)  // do not add directories
      {
        PluginFile p(concat(dir, fileData.name));

        // Search for loaded plugins with the same base name.
        for (size_t i = 0; i < AutoLoadedPlugins.size(); ++i)
        {
          if (streqi(AutoLoadedPlugins[i].BaseName.c_str(), p.BaseName.c_str()))
          {
            // Prevent loading a plugin with a basename that is
            // already loaded (from another autoload folder).
            continue;
          }
        }

        // Try to load plugin
        AVSValue dummy;
        LoadPlugin(p, false, &dummy);
      }
    } // for bContinue
    _findclose(hFind);
  }

  // Load script imports
  for (const std::string& dir : AutoloadDirs)
  {
    CWDChanger cwdchange(dir.c_str());

    // Append file search filter to directory path
    std::string filePattern = concat(dir, scriptFilter);

    // Iterate through all files in directory
    _finddata_t fileData;
    intptr_t hFind = _findfirst(filePattern.c_str(), &fileData);
    for (bool bContinue = hFind;
      bContinue;
      bContinue = (_findnext(hFind, &fileData) != -1)
      )
    {
      if ((fileData.attrib & _A_SUBDIR) == 0)  // do not add directories
      {
        PluginFile p(concat(dir, fileData.name));

        // Search for loaded imports with the same base name.
        for (size_t i = 0; i < AutoLoadedImports.size(); ++i)
        {
          if (streqi(AutoLoadedImports[i].BaseName.c_str(), p.BaseName.c_str()))
          {
            // Prevent loading a plugin with a basename that is
            // already loaded (from another autoload folder).
            continue;
          }
        }

        // Try to load script
        Env->Invoke("Import", p.FilePath.c_str());
        AutoLoadedImports.push_back(p);
      }
    } // for bContinue
    _findclose(hFind);
  }
#endif

  Autoloading = false;
}

PluginManager::~PluginManager()
{
  // Delete all AVSFunction objects that we created
  std::unordered_set<const AVSFunction*> function_set;
  for (const auto& lists : ExternalFunctions)
  {
      const FunctionList& funcList = lists.second;
      for (const auto& func : funcList)
        function_set.insert(func);
  }
  for (const auto& lists : AutoloadedFunctions)
  {
      const FunctionList& funcList = lists.second;
      for (const auto& func : funcList)
        function_set.insert(func);
  }
  for (const auto& func : function_set)
  {
      delete func;
  }


  // Unload plugin binaries
  for (size_t i = 0; i < LoadedPlugins.size(); ++i)
  {
    assert(LoadedPlugins[i].Library);
    FreeLibrary(LoadedPlugins[i].Library);
    LoadedPlugins[i].Library = NULL;
  }
  for (size_t i = 0; i < AutoLoadedPlugins.size(); ++i)
  {
    assert(AutoLoadedPlugins[i].Library);
    FreeLibrary(AutoLoadedPlugins[i].Library);
    AutoLoadedPlugins[i].Library = NULL;
  }

  Env = NULL;
  PluginInLoad = NULL;
}

void PluginManager::UpdateFunctionExports(const char* funcName, const char* funcParams, const char *exportVar)
{
  if (exportVar == NULL)
    exportVar = "$PluginFunctions$";

  // Update $PluginFunctions$
  const char *oldFnList = Env->GetVarString(exportVar, "");
  std::string FnList(oldFnList);
  if (FnList.size() > 0)    // if the list is not empty...
    FnList.push_back(' ');  // ...add a delimiting whitespace
  FnList.append(funcName);
  Env->SetGlobalVar(exportVar, AVSValue( Env->SaveString(FnList.c_str(), (int)FnList.size()) ));

  // Update $Plugin!...!Param$
  std::string param_id;
  param_id.reserve(128);
  param_id.append("$Plugin!");
  param_id.append(funcName);
  param_id.append("!Param$");
  Env->SetGlobalVar(Env->SaveString(param_id.c_str(), (int)param_id.size()), AVSValue(Env->SaveString(funcParams)));
}

bool PluginManager::LoadPlugin(const char* path, bool throwOnError, AVSValue *result)
{
  auto pf = PluginFile { path };
  return LoadPlugin(pf, throwOnError, result);
}
#ifdef AVS_WINDOWS
static bool Is64BitDLL(std::string sDLL, bool &bIs64BitDLL)
{
  bIs64BitDLL = false;
  LOADED_IMAGE li;

  if (!MapAndLoad((LPSTR)sDLL.c_str(), NULL, &li, TRUE, TRUE))
  {
    //error handling (check GetLastError())
    return false;
  }

  if (li.FileHeader->FileHeader.Machine != IMAGE_FILE_MACHINE_I386) //64 bit image
    bIs64BitDLL = true;

  UnMapAndLoad(&li);

  return true;
}
#endif //AVS_WINDOWS
bool PluginManager::LoadPlugin(PluginFile &plugin, bool throwOnError, AVSValue *result)
{
  std::vector<PluginFile>& PluginList = Autoloading ? AutoLoadedPlugins : LoadedPlugins;

  for (size_t i = 0; i < PluginList.size(); ++i)
  {
    if (streqi(PluginList[i].FilePath.c_str(), plugin.FilePath.c_str()))
    {
      // Imitate successful loading if the plugin is already loaded
      plugin = PluginList[i];
      return true;
    }
  }

  plugin.isAvs25 = false;

#ifdef AVS_WINDOWS
  // Search for dependent DLLs in the plugin's directory too
  size_t slash_pos = plugin.FilePath.rfind('/');
  std::string plugin_dir = plugin.FilePath.substr(0, slash_pos);;
  DllDirChanger dllchange(plugin_dir.c_str());

  // Load the dll into memory
  plugin.Library = LoadLibraryEx(plugin.FilePath.c_str(), 0, LOAD_WITH_ALTERED_SEARCH_PATH);
  if (plugin.Library == NULL)
  {
    DWORD errCode = GetLastError();

    // Bitness mixing always throws an error, regardless of throwOnError state
    // By this new behaviour even plugin auto-load will fail
    bool bIs64BitDLL;
    bool succ = Is64BitDLL(plugin.FilePath, bIs64BitDLL);
    if (succ) {
      const bool selfIs32 = sizeof(void *) == 4;
      if (selfIs32 && bIs64BitDLL)
        Env->ThrowError("Cannot load a 64 bit DLL in 32 bit Avisynth: '%s'.\n", plugin.FilePath.c_str());
      if (!selfIs32 && !bIs64BitDLL)
        Env->ThrowError("Cannot load a 32 bit DLL in 64 bit Avisynth: '%s'.\n", plugin.FilePath.c_str());
    }
    if (throwOnError)
    {
      Env->ThrowError("Cannot load file '%s'. Platform returned code %d:\n%s", plugin.FilePath.c_str(), errCode, GetLastErrorText(errCode).c_str());
    }
    else
      return false;
  }
#else // AVS_POSIX
  plugin.Library = dlopen(plugin.FilePath.c_str(), RTLD_LAZY);
  if (plugin.Library == NULL)
    Env->ThrowError("Cannot load file '%s'. Reason: %s", plugin.FilePath.c_str(), dlerror());
#endif

  // Try to load various plugin interfaces
  if (!TryAsAvs26(plugin, result))
  {
    if (!TryAsAvsC(plugin, result))
    {
      if (!TryAsAvs25(plugin, result))
      {
        FreeLibrary(plugin.Library);
        plugin.Library = NULL;

        if (throwOnError)
          Env->ThrowError("'%s' cannot be used as a plugin for AviSynth.", plugin.FilePath.c_str());
        else
          return false;
      }
    }
  }

  PluginList.push_back(plugin);
  return true;
}

const AVSFunction* PluginManager::Lookup(const FunctionMap& map, const char* search_name, const AVSValue* args, size_t num_args,
                    bool strict, size_t args_names_count, const char* const* arg_names) const
{
    FunctionMap::const_iterator list_it = map.find(search_name);
    if (list_it == map.end())
      return NULL;

    for ( FunctionList::const_reverse_iterator func_it = list_it->second.rbegin();
          func_it != list_it->second.rend();
          ++func_it)
    {
      const AVSFunction *func = *func_it;
      if (AVSFunction::TypeMatch(func->param_types, args, num_args, strict, Env) &&
          AVSFunction::ArgNameMatch(func->param_types, args_names_count, arg_names)
         )
      {
        return func;
      }
    }

    return NULL;
}

const AVSFunction* PluginManager::Lookup(const char* search_name, const AVSValue* args, size_t num_args,
                    bool strict, size_t args_names_count, const char* const* arg_names) const
{
  /* Lookup in non-autoloaded functions first, so that they take priority */
  const AVSFunction* func = Lookup(ExternalFunctions, search_name, args, num_args, strict, args_names_count, arg_names);
  if (func != NULL)
    return func;

  /* If not found, look amongst the autoloaded */
  return Lookup(AutoloadedFunctions, search_name, args, num_args, strict, args_names_count, arg_names);
}

bool PluginManager::FunctionExists(const char* name) const
{
    bool autoloaded = (AutoloadedFunctions.find(name) != AutoloadedFunctions.end());
    return autoloaded || (ExternalFunctions.find(name) != ExternalFunctions.end());
}

// A minor helper function
static bool FunctionListHasDll(const FunctionList &list, const char *dll_path)
{
    for (const auto &f : list) {
        if ( (nullptr == f->dll_path) || (nullptr == dll_path) ) {
            if (f->dll_path == dll_path) {
                return true;
            }
        } else if (streqi(f->dll_path, dll_path)) {
            return true;
        }
    }
    return false;
}

void PluginManager::AddFunction(const char* name, const char* params, IScriptEnvironment::ApplyFunc apply, void* user_data, const char *exportVar, bool isAvs25)
{
  if (!IsValidParameterString(params))
    Env->ThrowError("%s has an invalid parameter string (bug in filter)", name);

  FunctionMap& functions = Autoloading ? AutoloadedFunctions : ExternalFunctions;

  AVSFunction *newFunc = NULL;
  if (PluginInLoad != NULL)
  {
    // either called using IScriptEnvironment_Avs25 or we are inside of a CPPv2.5 plugin load
    const bool isAvs25like = isAvs25 || PluginInLoad->isAvs25;
    newFunc = new AVSFunction(name, PluginInLoad->BaseName.c_str(), params, apply, user_data, PluginInLoad->FilePath.c_str(), isAvs25like);
  }
  else
  {
    // isAvs25: called using an IScriptEnvironment_Avs25->AddFunction
    newFunc = new AVSFunction(name, NULL, params, apply, user_data, NULL, isAvs25);
      /*
         // Comment out but kept for reference.
         // This assert is false when AddFunction is called from a cpp non-plugin.
         // e.g. a "master" that directly loads avisynth
         // The extemption could handle only situations when function was loaded by C interface avs_add_function.
      if(apply != &create_c_video_filter)
        assert(AVSFunction::IsScriptFunction(newFunc));
      */
  }

  // Warn user if a function with the same name is already registered by another plugin
  {
      const auto &it = functions.find(newFunc->name);
      if ( (functions.end() != it) && !FunctionListHasDll(it->second, newFunc->dll_path) )
      {
          OneTimeLogTicket ticket(LOGTICKET_W1008, newFunc->name);
          Env->LogMsgOnce(ticket, LOGLEVEL_WARNING, "%s() is defined by multiple plugins. Calls to this filter might be ambiguous and could result in the wrong function being called.", newFunc->name);
      }
  }

  functions[newFunc->name].push_back(newFunc);
  UpdateFunctionExports(newFunc->name, newFunc->param_types, exportVar);

  if (NULL != newFunc->canon_name)
  {
      // Warn user if a function with the same name is already registered by another plugin
      {
          const auto &it = functions.find(newFunc->canon_name);
          if ((functions.end() != it) && !FunctionListHasDll(it->second, newFunc->dll_path))
          {
              OneTimeLogTicket ticket(LOGTICKET_W1008, newFunc->canon_name);
              Env->LogMsgOnce(ticket, LOGLEVEL_WARNING, "%s() is defined by multiple plugins. Calls to this filter might be ambiguous and could result in the wrong function being called.", newFunc->name);
          }
      }

      functions[newFunc->canon_name].push_back(newFunc);
      UpdateFunctionExports(newFunc->canon_name, newFunc->param_types, exportVar);
  }
}

std::string PluginManager::PluginLoading() const
{
    if (NULL == PluginInLoad)
        return std::string();
    else
        return PluginInLoad->BaseName;
}

bool PluginManager::TryAsAvs26(PluginFile &plugin, AVSValue *result)
{
  extern const AVS_Linkage* const AVS_linkage; // In interface.cpp
#ifdef AVS_POSIX
  AvisynthPluginInit3Func AvisynthPluginInit3 = (AvisynthPluginInit3Func)dlsym(plugin.Library, "AvisynthPluginInit3");
#elif defined(GCC_WIN32)
  AvisynthPluginInit3Func AvisynthPluginInit3 = (AvisynthPluginInit3Func)GetProcAddress(plugin.Library, "_AvisynthPluginInit3");
  if (!AvisynthPluginInit3)
    AvisynthPluginInit3 = (AvisynthPluginInit3Func)GetProcAddress(plugin.Library, "AvisynthPluginInit3@8");
#else
  AvisynthPluginInit3Func AvisynthPluginInit3 = (AvisynthPluginInit3Func)GetProcAddress(plugin.Library, "AvisynthPluginInit3");
  if (!AvisynthPluginInit3)
    AvisynthPluginInit3 = (AvisynthPluginInit3Func)GetProcAddress(plugin.Library, "_AvisynthPluginInit3@8");
#endif

  bool success = true;
  if (AvisynthPluginInit3 == NULL)
    return false;
  else
  {
    PluginInLoad = &plugin;
    // a bad plugin can kill everything if it uses e.g. an old IScriptEnvironment2
    try {
      *result = AvisynthPluginInit3(Env, AVS_linkage);
    }
    catch (...)
    {
      success = false;
    }
    PluginInLoad = NULL;
  }

  return success;
}

bool PluginManager::TryAsAvs25(PluginFile &plugin, AVSValue *result)
{
#ifdef AVS_POSIX
  AvisynthPluginInit2Func AvisynthPluginInit2 = (AvisynthPluginInit2Func)dlsym(plugin.Library, "AvisynthPluginInit2");
#elif defined(GCC_WIN32)
  AvisynthPluginInit2Func AvisynthPluginInit2 = (AvisynthPluginInit2Func)GetProcAddress(plugin.Library, "_AvisynthPluginInit2");
  if (!AvisynthPluginInit2)
    AvisynthPluginInit2 = (AvisynthPluginInit2Func)GetProcAddress(plugin.Library, "AvisynthPluginInit2@4");
#else
  AvisynthPluginInit2Func AvisynthPluginInit2 = (AvisynthPluginInit2Func)GetProcAddress(plugin.Library, "AvisynthPluginInit2");
  if (!AvisynthPluginInit2)
    AvisynthPluginInit2 = (AvisynthPluginInit2Func)GetProcAddress(plugin.Library, "_AvisynthPluginInit2@4");
#endif

  bool success = true;
  if (AvisynthPluginInit2 == NULL)
    return false;
  else
  {
    PluginInLoad = &plugin;
    // in case of a crash in init2
    try {
      // Pass the 2.5 variant IScriptEnvironment, which has different Invoke
      // and AddFunction method to avoid array copy/free problems.
      // (NEW_AVSVALUE compatibility: "baked code" strikes back)
      *result = AvisynthPluginInit2(Env->GetEnv25());
      plugin.isAvs25 = true;
    }
    catch (...)
    {
      success = false;
    }
    PluginInLoad = NULL;
  }

  return success;
}

bool PluginManager::TryAsAvsC(PluginFile &plugin, AVSValue *result)
{
#ifdef AVS_POSIX
  AvisynthCPluginInitFunc AvisynthCPluginInit = (AvisynthCPluginInitFunc)dlsym(plugin.Library, "avisynth_c_plugin_init");
#else
#ifdef _WIN64
  AvisynthCPluginInitFunc AvisynthCPluginInit = (AvisynthCPluginInitFunc)GetProcAddress(plugin.Library, "avisynth_c_plugin_init");
  if (!AvisynthCPluginInit)
    AvisynthCPluginInit = (AvisynthCPluginInitFunc)GetProcAddress(plugin.Library, "_avisynth_c_plugin_init@4");
#else // _WIN32
  AvisynthCPluginInitFunc AvisynthCPluginInit = (AvisynthCPluginInitFunc)GetProcAddress(plugin.Library, "_avisynth_c_plugin_init@4");
  if (!AvisynthCPluginInit)
    AvisynthCPluginInit = (AvisynthCPluginInitFunc)GetProcAddress(plugin.Library, "avisynth_c_plugin_init@4");
#endif
#endif // AVS_POSIX

  if (AvisynthCPluginInit == NULL)
    return false;
  else
  {
    PluginInLoad = &plugin;
    {
        AVS_ScriptEnvironment e;
        e.env = Env;
        AVS_ScriptEnvironment *pe;
        pe = &e;
        const char *s = NULL;
#if defined(X86_32) && defined(MSVC)
        int callok = 1; // (stdcall)
        __asm // Tritical - Jan 2006
        {
            push eax
            push edx

            push 0x12345678		// Stash a known value

            mov eax, pe			// Env pointer
            push eax			// Arg1
            call AvisynthCPluginInit			// avisynth_c_plugin_init

            lea edx, s			// return value is in eax
            mov DWORD PTR[edx], eax

            pop eax				// Get top of stack
            cmp eax, 0x12345678	// Was it our known value?
            je end				// Yes! Stack was cleaned up, was a stdcall

            lea edx, callok
            mov BYTE PTR[edx], 0 // Set callok to 0 (_cdecl)

            pop eax				// Get 2nd top of stack
            cmp eax, 0x12345678	// Was this our known value?
            je end				// Yes! Stack is now correctly cleaned up, was a _cdecl

            mov BYTE PTR[edx], 2 // Set callok to 2 (bad stack)
    end:
            pop edx
            pop eax
        }
      switch(callok)
      {
      case 0:   // cdecl
#ifdef AVSC_USE_STDCALL
            Env->ThrowError("Avisynth 2 C Plugin '%s' has wrong calling convention! Must be _stdcall.", plugin.BaseName.c_str());
#endif
        break;
      case 1:   // stdcall
#ifndef AVSC_USE_STDCALL
            Env->ThrowError("Avisynth 2 C Plugin '%s' has wrong calling convention! Must be _cdecl.", plugin.BaseName.c_str());
#endif
        break;
      case 2:
        Env->ThrowError("Avisynth 2 C Plugin '%s' has corrupted the stack.", plugin.BaseName.c_str());
      }
#else
      s = AvisynthCPluginInit(pe);
#endif
//	    if (s == 0)
    //	    Env->ThrowError("Avisynth 2 C Plugin '%s' returned a NULL pointer.", plugin.BaseName.c_str());

      *result = AVSValue(s);
      plugin.isAvs25 = true; // no array deep copy/free when NEW_AVSVALUE
    }
    PluginInLoad = NULL;
  }

  return true;
}

/*
---------------------------------------------------------------------------------
---------------------------------------------------------------------------------
                                 LoadPlugin
---------------------------------------------------------------------------------
---------------------------------------------------------------------------------
*/

AVSValue LoadPlugin(AVSValue args, void*, IScriptEnvironment* env)
{
  IScriptEnvironment2 *env2 = static_cast<IScriptEnvironment2*>(env);

  bool success = true;
  for (int i = 0; i < args[0].ArraySize(); ++i)
  {
    AVSValue dummy;
    success &= env2->LoadPlugin(args[0][i].AsString(), true, &dummy);
  }

  return AVSValue(success);
}

extern const AVSFunction Plugin_functions[] = {
  {"LoadPlugin", BUILTIN_FUNC_PREFIX, "s+", LoadPlugin},
  {"LoadCPlugin", BUILTIN_FUNC_PREFIX, "s+", LoadPlugin },          // for compatibility with older scripts
  {"Load_Stdcall_Plugin", BUILTIN_FUNC_PREFIX, "s+", LoadPlugin },  // for compatibility with older scripts
  { 0 }
};
