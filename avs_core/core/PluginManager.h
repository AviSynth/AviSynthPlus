#ifndef AVSCORE_PLUGINS_H
#define AVSCORE_PLUGINS_H

#include <string>
#include <map>
#include <vector>
#include "internal.h"

class InternalEnvironment;
struct PluginFile;

struct StdStriComparer
{
  bool operator() (const std::string& lhs, const std::string& rhs) const
  {
#if defined(MSVC)
    return (_strcmpi(lhs.c_str(), rhs.c_str()) < 0);
#else
    return (strcasecmp(lhs.c_str(), rhs.c_str()) < 0);
#endif
  }
};

typedef std::vector<const AVSFunction*> FunctionList;
typedef std::map<std::string,FunctionList,StdStriComparer> FunctionMap;
class PluginManager
{
private:
  InternalEnvironment *Env;
  PluginFile *PluginInLoad;
  std::vector<std::string> AutoloadDirs;
  std::vector<PluginFile> AutoLoadedImports;
  std::vector<PluginFile> AutoLoadedPlugins;
  std::vector<PluginFile> LoadedPlugins;
  FunctionMap ExternalFunctions;
  FunctionMap AutoloadedFunctions;
  bool AutoloadExecuted;
  bool Autoloading;
  bool ExportBatch;                  // collect plugin function names during AutoloadPlugins
  std::string PendingFunctionList;   // space separated names, published by FlushPendingExports

  int TryAsAvs26(PluginFile &plugin, AVSValue *result, std::string& avsexception_message);
  bool TryAsAvs25(PluginFile &plugin, AVSValue *result);
  bool TryAsAvsPreV11C(PluginFile& plugin, AVSValue* result);
  bool TryAsAvsC(PluginFile &plugin, AVSValue *result);

  const AVSFunction* Lookup(const FunctionMap& map,
    const char* search_name,
    const AVSValue* args,
    size_t num_args,
    bool strict,
    size_t args_names_count,
    const char* const* arg_names) const;


public:
  PluginManager(InternalEnvironment* env);
  ~PluginManager();

  void ClearAutoloadDirs();
  void AddAutoloadDir(const std::string &dir_utf8, bool toFront);

  bool LoadPlugin(PluginFile &plugin, bool throwOnError, AVSValue *result);
  bool LoadPlugin(const char* path, bool throwOnError, AVSValue *result);

  bool HasAutoloadExecuted() const { return AutoloadExecuted; }

  void UpdateFunctionExports(const char* funcName, const char* funcParams, const char *exportVar);

  std::string ListAutoloadDirs();
  std::string ListAutoloadedFunctions();
  void FlushPendingExports();

  bool FunctionExists(const char* name) const;
  std::string PluginLoading() const;    // Returns the basename of the plugin DLL that is currently being loaded, or NULL if no plugin is being loaded
  void AutoloadPlugins();
  void AddFunction(const char* name, const char* params, IScriptEnvironment::ApplyFunc apply, void* user_data, const char *exportVar,
    bool isCalledFromAvs25Interface,
    bool isCalledFromPreV11CInterface);
  const AVSFunction* Lookup(const char* search_name,
    const AVSValue* args,
    size_t num_args,
    bool strict,
    size_t args_names_count,
    const char* const* arg_names) const;
};

#endif  // AVSCORE_PLUGINS_H
