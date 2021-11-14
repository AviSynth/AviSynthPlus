#include "avs/config.h"
#include "FilterGraph.h"
#include "DeviceManager.h"
#include "InternalEnvironment.h"

#ifdef AVS_WINDOWS
  #include <avs/win.h>
#else
  #include <avs/posix.h>
#endif

#include <map>
#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>

#include <avs/filesystem.h>

static AVSValue DeepCopyValue(std::vector<std::unique_ptr<AVSValue[]>>& arrays, const AVSValue& src) {
  if (src.IsArray()) {
    AVSValue* copy = new AVSValue[src.ArraySize()];
    for (int i = 0; i < src.ArraySize(); ++i) {
      copy[i] = src[i]; // NEW_AVSVALUE is already doing deep copy
      // copy[i] = DeepCopyValue(arrays, src[i]);
    }
    arrays.emplace_back(std::unique_ptr<AVSValue[]>(copy));
    return AVSValue(copy, src.ArraySize());
  }
  return src;
}

FilterGraphNode::FilterGraphNode(PClip child, const char* name,
  const AVSValue& last_, const AVSValue& args_, const char* const* argnames_,
	IScriptEnvironment* env)
  : Env(env)
	, child(child)
  , name(name)
  , memory(new GraphMemoryNode())
{
  if (last_.Defined()) {
    std::vector<AVSValue> argstmp;
    argstmp.push_back(last_);
    if (argnames_) {
      argnames.push_back(std::string());
    }
    for (int i = 0; i < args_.ArraySize(); ++i) {
      argstmp.push_back(args_[i]);
    }
    args = DeepCopyValue(arrays, AVSValue(argstmp.data(), (int)argstmp.size()));
  }
  else {
    args = DeepCopyValue(arrays, args_.IsArray() ? args_ : AVSValue(args_, 1));
  }

  if (argnames_) {
    for (int i = 0; i < args_.ArraySize(); ++i) {
      argnames.push_back(argnames_[i] ? std::string(argnames_[i]) : std::string());
    }
  }

	Env->ManageCache(MC_RegisterGraphNode, this);
}

FilterGraphNode::~FilterGraphNode()
{
	Env->ManageCache(MC_UnRegisterGraphNode, this);
}

struct ScopedGraphNode {
	FilterGraphNode*& target;
  FilterGraphNode* prev;
  ScopedGraphNode(FilterGraphNode*& target, FilterGraphNode* node) : target(target) {
    prev = target;
		target = node;
  }
  ~ScopedGraphNode() {
		target = prev;
  }
};

PVideoFrame __stdcall FilterGraphNode::GetFrame(int n, IScriptEnvironment* env)
{
  ScopedGraphNode scope(static_cast<InternalEnvironment*>(env)->GetCurrentGraphNode(), this);
  return child->GetFrame(n, env);
}

void GraphMemoryNode::OnAllocate(size_t bytes, Device* dev)
{
  auto it = memory.find(dev);
  if (it == memory.end()) {
    memory[dev] = MemoryInfo();
    it = memory.find(dev);
  }
  it->second.numAllocation++;
  it->second.totalBytes += bytes;
}

void GraphMemoryNode::OnFree(size_t bytes, Device* dev)
{
  auto it = memory.find(dev);
  if (it == memory.end()) {
    printf("Unexpected behavior ...\n");
    return;
  }
  it->second.numAllocation--;
  it->second.totalBytes -= bytes;
}

class FilterGraph
{
protected:
  IScriptEnvironment * env;

  struct NodeInfo {
    bool isFunction;
    int number;
    std::string name;
    std::string args;
    std::vector<void*> refNodes;

    int cacheSize;
    int cacheCapacity;
    std::map<Device*, GraphMemoryNode::MemoryInfo> memory;

    NodeInfo() { }
    NodeInfo(int number) : number(number) { }
  };

  std::map<void*, NodeInfo> nodeMap;

  int DoClip(IClip* pclip) {
    if (nodeMap.find(pclip) == nodeMap.end()) {
      nodeMap.insert(std::make_pair(pclip, (int)nodeMap.size()));
      FilterGraphNode* node = dynamic_cast<FilterGraphNode*>(pclip);
      if (node != nullptr) {
        NodeInfo& info = nodeMap[node];
        info.isFunction = false;
        info.name = node->name;
        info.args = "(" + DoArray(info, nullptr, node->argnames.data(), node->args) + ")";
        info.cacheSize = node->SetCacheHints(CACHE_GET_SIZE, 0);
        info.cacheCapacity = node->SetCacheHints(CACHE_GET_CAPACITY, 0);
        info.memory = node->memory->memory;
      }
      OutClip(nodeMap[node]);
    }
    return nodeMap[pclip].number;
  }

  int DoFunc(IFunction* pfunc) {
    if (nodeMap.find(pfunc) == nodeMap.end()) {
      nodeMap.insert(std::make_pair(pfunc, (int)nodeMap.size()));
      NodeInfo& info = nodeMap[pfunc];
      info.isFunction = true;
      auto captures = pfunc->GetCaptures();
      info.name = pfunc->ToString(env);
      info.args = "[" + DoArray(info, captures.var_names, nullptr, AVSValue(captures.var_data, captures.count)) + "]";
      info.cacheSize = 0;
      info.cacheCapacity = 0;
      OutFunc(info);
    }
    return nodeMap[pfunc].number;
  }

  std::string DoArray(NodeInfo& info, const char** argnames_c, std::string* argnames_s, const AVSValue& arr) {
    std::stringstream ss;
    int breakpos = 0;
    int maxlen = 60;

    for (int i = 0; i < arr.ArraySize(); ++i) {
      if (i != 0) {
        ss << ",";
      }
      if (argnames_c && argnames_c[i]) {
        ss << argnames_c[i] << "=";
      }
      if (argnames_s && argnames_s[i].size() > 0) {
        ss << argnames_s[i] << "=";
      }
      const AVSValue& v = arr[i];
      if (!v.Defined()) {
        ss << "default";
      }
      else if (v.IsClip()) {
        IClip* pclip = (IClip*)(void*)v.AsClip();
        int clipnum = DoClip(pclip);
        ss << "clip" << (clipnum + 1);
        info.refNodes.push_back(pclip);
      }
      else if (v.IsFunction()) {
        IFunction* pfunc = (IFunction*)(void*)v.AsFunction();
        int funcnum = DoFunc(pfunc);
        ss << "func" << (funcnum + 1);
        info.refNodes.push_back(pfunc);
      }
      else if (v.IsArray()) {
        ss << OutArray("(" + DoArray(info, nullptr, nullptr, v) + ")");
      }
      else if (v.IsBool()) {
        ss << (v.AsBool() ? "True" : "False");
      }
      else if (v.IsInt()) {
        ss << v.AsInt();
      }
      else if (v.IsFloat()) {
        ss << std::setprecision(8) << v.AsFloat();
      }
      else if (v.IsString()) {
        ss << "\"" << v.AsString() << "\"";
      }
      else {
        ss << "<error>";
      }
      if ((int)ss.tellp() - breakpos > maxlen) {
        ss << "\n";
        breakpos = (int)ss.tellp();
      }
    }
    return ss.str();
  }

  virtual void OutClip(const NodeInfo& info) = 0;
  virtual void OutFunc(const NodeInfo& info) = 0;
  virtual std::string OutArray(const std::string& args) = 0;

public:
  int Construct(FilterGraphNode* root, IScriptEnvironment* env_)
  {
    env = env_;
    nodeMap.clear();
    return DoClip(root);
  }
	void Construct(const std::vector<FilterGraphNode*>& roots, IScriptEnvironment* env_)
	{
		env = env_;
		nodeMap.clear();
		for (auto node : roots) {
			if (node != nullptr) {
				DoClip(node);
			}
		}
	}
};

static void ReplaceAll(std::string& str, const std::string& from, const std::string& to) {
  size_t start_pos = 0;
  while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
    str.replace(start_pos, from.length(), to);
    start_pos += to.length();
  }
}

class AvsScriptFilterGraph : private FilterGraph
{
  std::stringstream ss;

protected:
  virtual void OutClip(const NodeInfo& info) {
    int num = info.number + 1;
    if (info.name.size() == 0) {
      ss << "clip" << num << ": Failed to get information" << std::endl;
    }
    else {
      auto args = info.args;
      ReplaceAll(args, "\n", "");
      ss << "clip" << num << " = " << info.name << args << std::endl;
    }
  }
  virtual void OutFunc(const NodeInfo& info) {
    int num = info.number + 1;
    auto args = info.args;
    ReplaceAll(args, "\n", "");
    ss << "func" << num << " = function" << args << "(){ " << info.name << " }" << std::endl;
  }

  int nextArrayNumber = 0;
  virtual std::string OutArray(const std::string& args) {
    std::string name = std::string("array") + std::to_string(++nextArrayNumber);
    ss << name;
    ss << " = ArrayCreate" << args << std::endl;
    return name;
  }
public:

  void Construct(FilterGraphNode* root, IScriptEnvironment* env) {
    int last = FilterGraph::Construct(root, env);
    ss << "return clip" << (last + 1) << std::endl;
  }

  std::string GetOutput() {
    return ss.str();
  }
};

class DotFilterGraph : private FilterGraph
{
  bool enableArgs;
  bool enableMemory;
  std::stringstream ss;

protected:

  void printfcomma(size_t n) {
    if (n < 1000) {
      ss << n;
      return;
    }
    printfcomma(n / 1000);
    ss << ',' << std::setfill('0') << std::setw(3) << (n % 1000);
  }

  virtual void OutClip(const NodeInfo& info) {
    int num = info.number + 1;
    ss << "clip" << num;
    if (info.name.size() == 0) {
      ss << " [label = \"...\"];" << std::endl;
    }
    else {
      if (enableArgs) {
        std::string label = info.name + info.args;
        ReplaceAll(label, "\\", "\\\\");
        ReplaceAll(label, "\"", "\\\"");
        ReplaceAll(label, "\n", "\\n");
        ss << " [label = \"" << label << "\"];" << std::endl;
      }
      else {
        if (info.cacheCapacity != 0) {
          ss << " [label = \"" << info.name << "(caching " << info.cacheSize << " frames with capacity " << info.cacheCapacity << ")";
        }
        else {
          ss << " [label = \"" << info.name;
        }
        if (enableMemory) {
          for (auto entry : info.memory) {
            ss << "\\n" << entry.first->GetName() << ": " << entry.second.numAllocation << " frames, ";
            printfcomma(entry.second.totalBytes);
            ss << " bytes";
          }
        }
        ss << "\"];" << std::endl;;
      }
    }
    for (void* pclip : info.refNodes) {
      auto& node = nodeMap[pclip];
      int refnum = node.number + 1;
      if (node.isFunction) {
        ss << "func" << refnum << " -> " << "clip" << num << ";" << std::endl;
      }
      else {
        ss << "clip" << refnum << " -> " << "clip" << num << ";" << std::endl;
      }
    }
  }
  virtual void OutFunc(const NodeInfo& info) {
    int num = info.number + 1;
    ss << "func" << num;
    if (enableArgs) {
      std::string label = info.name + "\n" + info.args;
      ReplaceAll(label, "\\", "\\\\");
      ReplaceAll(label, "\"", "\\\"");
      ReplaceAll(label, "\n", "\\n");
      ss << " [label = \"" << label << "\"];" << std::endl;
    }
    else {
      ss << " [label = \"" << info.name << "\"];" << std::endl;
    }
    for (void* pclip : info.refNodes) {
      auto& node = nodeMap[pclip];
      int refnum = node.number + 1;
      if (node.isFunction) {
        ss << "func" << refnum << " -> " << "func" << num << ";" << std::endl;
      }
      else {
        ss << "clip" << refnum << " -> " << "func" << num << ";" << std::endl;
      }
    }
  }

  int nextArrayNumber = 0;
  virtual std::string OutArray(const std::string& args) {
    std::string name = std::string("array") + std::to_string(++nextArrayNumber);
    //ss << name;
    //ss << " = ArrayCreate" << args << std::endl;
    return name;
  }
public:

  void Construct(FilterGraphNode* root, bool enableArgs, bool enableMemory, IScriptEnvironment* env) {
    this->enableArgs = enableArgs;
    this->enableMemory = enableMemory;
    ss << "digraph avs_filter_graph {" << std::endl;
    ss << "node [ shape = box ];" << std::endl;
    int last = FilterGraph::Construct(root, env);
    ss << "GOAL;" << std::endl;
    ss << "clip" << (last + 1) << " -> GOAL" << std::endl;
    ss << "}" << std::endl;
  }

	void Construct(const std::vector<FilterGraphNode*>& roots, bool enableArgs, bool enableMemory, IScriptEnvironment* env) {
		this->enableArgs = enableArgs;
		this->enableMemory = enableMemory;
		ss << "digraph avs_filter_graph {" << std::endl;
		ss << "node [ shape = box ];" << std::endl;
		FilterGraph::Construct(roots, env);
		ss << "}" << std::endl;
	}

  std::string GetOutput() {
    return ss.str();
  }
};

void DoDumpGraph(const std::vector<FilterGraphNode*>& roots, const char* path, IScriptEnvironment* env)
{
	DotFilterGraph graph;
	graph.Construct(roots, true, true, env);
	std::string ret = graph.GetOutput();

	FILE* fp = fopen(path, "w");
	if (fp == nullptr) {
		env->ThrowError("Could not open output file ...");
	}
	fwrite(ret.data(), ret.size(), 1, fp);
	fclose(fp);
}

static void DoDumpGraph(PClip clip, int mode, const char* path, IScriptEnvironment* env)
{
  FilterGraphNode* root = dynamic_cast<FilterGraphNode*>((IClip*)(void*)clip);

  std::string ret;

  if (mode == 0) {
    AvsScriptFilterGraph graph;
    graph.Construct(root, env);
    ret = graph.GetOutput();
  }
  else if (mode == 1 || mode == 2) {
    DotFilterGraph graph;
    graph.Construct(root, mode == 1, true, env);
    ret = graph.GetOutput();
  }
  else {
    env->ThrowError("Unknown mode (%d)", mode);
  }

  FILE* fp = fopen(path, "w");
  if (fp == nullptr) {
    env->ThrowError("Could not open output file ...");
  }
  fwrite(ret.data(), ret.size(), 1, fp);
  fclose(fp);
}

class DelayedDump : public GenericVideoFilter
{
  std::string outpath;
  int mode;
  int nframes;
  bool repeat;
  std::vector<bool> fired;
public:
  DelayedDump(PClip clip, const std::string& outpath, int mode, int nframes, bool repeat)
    : GenericVideoFilter(clip)
    , outpath(outpath)
    , mode(mode)
    , nframes(nframes)
    , repeat(repeat)
  {
    if (repeat) {
      fired.resize((clip->GetVideoInfo().num_frames + nframes - 1) / nframes);
    }
    else {
      fired.resize(1);
    }
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env)
  {
    if (repeat) {
      int slot = std::max(0, std::min(n / nframes, (int)fired.size() - 1));
      if (fired[slot] == false) {
        fired[slot] = true;
        char basename[260];
        strcpy(basename, outpath.c_str());
        char* extension = strrchr(basename, '.');
        if (extension) {
          *extension++ = 0;
        }
        std::string path(basename);
        path = path + "-" + std::to_string(n);
        if (extension) {
          path = path + "." + extension;
        }
        DoDumpGraph(child, mode, path.c_str(), env);
      }
    }
    else {
      if (n == nframes && fired[0] == false) {
        fired[0] = true;
        DoDumpGraph(child, mode, outpath.c_str(), env);
      }
    }
    return child->GetFrame(n, env);
  }
};

static std::string GetFullPathNameWrap(const std::string& f)
{
  return fs::absolute(fs::path(f).lexically_normal()).generic_string();
}

static AVSValue DumpFilterGraph(AVSValue args, void* user_data, IScriptEnvironment* env) {
  PClip clip = args[0].AsClip();
  FilterGraphNode* root = dynamic_cast<FilterGraphNode*>((IClip*)(void*)clip);
  if (root == nullptr) {
    env->ThrowError("clip is not a FilterChainNode. Ensure you have enabled the chain analysis by SetGraphAnalysis(true).");
  }

  int mode = args[2].AsInt(0);
  const char* path = args[1].AsString("");
  int nframes = args[3].AsInt(-1);
  bool repeat = args[4].AsBool(false);

  if (nframes >= 0) {
    return new DelayedDump(clip, GetFullPathNameWrap(path), mode, nframes, repeat);
  }

  DoDumpGraph(clip, mode, path, env);

  return clip;
}

static AVSValue __cdecl SetGraphAnalysis(AVSValue args, void* user_data, IScriptEnvironment* env) {
  static_cast<InternalEnvironment*>(env)->SetGraphAnalysis(args[0].AsBool());
  return AVSValue();
}

extern const AVSFunction FilterGraph_filters[] = {
  { "SetGraphAnalysis", BUILTIN_FUNC_PREFIX, "b", SetGraphAnalysis, nullptr },
  { "DumpFilterGraph", BUILTIN_FUNC_PREFIX, "c[outfile]s[mode]i[nframes]i[repeat]b", DumpFilterGraph, nullptr },
  { 0 }
};
