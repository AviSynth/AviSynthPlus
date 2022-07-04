/*
  ConditionalReader  (c) 2004 by Klaus Post

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

  The author can be contacted at:
  sh0dan[at]stofanet.dk
*/

#include <avisynth.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <mutex>

#ifdef AVS_POSIX
#include <limits.h>
#endif

enum {
  MODE_UNKNOWN = -1,
  MODE_INT = 1,
  MODE_FLOAT = 2,
  MODE_BOOL = 3,
  MODE_STRING = 4
};

struct StringCache {
  char* string;
  StringCache *next;
};

class ConditionalReader : public GenericVideoFilter
{
private:
  const bool show;
  std::string variableName;
  const char* variableNameFixed;
  int mode;
  int offset;
  const bool local;
  StringCache* stringcache; // fixme: to vector
  union {
    int* intVal;
    bool* boolVal;
    float* floatVal;
    const char* *stringVal;
  };

  AVSValue ConvertType(const char* content, int line, IScriptEnvironment* env);
  void SetRange(int start_frame, int stop_frame, AVSValue v);
  void SetFrame(int framenumber, AVSValue v);
  void ThrowLine(const char* err, int line, IScriptEnvironment* env);
  AVSValue GetFrameValue(int framenumber);
  void CleanUp(void);

public:
  ConditionalReader(PClip _child, const char* filename, const char _varname[], bool _show, const char *_condVarSuffix, bool _local, IScriptEnvironment* env);
  ~ConditionalReader(void);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
  int __stdcall SetCacheHints(int cachehints, int frame_range);
  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env);
};


/* ------------------------------------------------------------------------------
** Write function to evaluate expressions per frame and write the results to file
** Ernst Peché, 2004
*/

class Write : public GenericVideoFilter
{
private:
	FILE * fout;
	int linecheck;	// 0=write each line, 1=write only if first expression == true, -1 = write at start, -2 = write at end
	bool flush;
	bool append;
  bool local;

#ifdef AVS_WINDOWS
	char filename[_MAX_PATH];
#else
	char filename[PATH_MAX];
#endif
	int arrsize;
	struct exp_res {
		AVSValue expression;
		const char* string;
	};
	exp_res* arglist;

	bool DoEval(IScriptEnvironment* env);
	void FileOut(IScriptEnvironment* env, const char* mode);

public:
    Write(PClip _child, const char* _filename, AVSValue args, int _linecheck, bool _flush, bool _append, bool _local, IScriptEnvironment* env);
	~Write(void);
	PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
  int __stdcall SetCacheHints(int cachehints, int frame_range);
	static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env);
	static AVSValue __cdecl Create_If(AVSValue args, void* user_data, IScriptEnvironment* env);
	static AVSValue __cdecl Create_Start(AVSValue args, void* user_data, IScriptEnvironment* env);
    static AVSValue __cdecl Create_End(AVSValue args, void* user_data, IScriptEnvironment* env);
};


class UseVar : public GenericVideoFilter
{
private:
   struct Var {
      const char* name;
      AVSValue val;
   };

   std::vector<Var> vars_;

public:
   UseVar(PClip _child, AVSValue vars, IScriptEnvironment* env);
   ~UseVar();
   PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
   int __stdcall SetCacheHints(int cachehints, int frame_range);
   static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env);
};

// avs+ (vs) style frame properties
class SetProperty : public GenericVideoFilter
{
private:
  const char* name;
  AVSValue value;
  const int kind;
  const int append_mode; // AVSPropAppendMode

public:
  SetProperty(PClip _child, const char* name, const AVSValue& value, const int kind, const int mode, IScriptEnvironment* env);
  ~SetProperty();
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
  int __stdcall SetCacheHints(int cachehints, int frame_range);
  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env);
};

class DeleteProperty : public GenericVideoFilter
{
private:
  bool propNames_defined;
  std::vector<std::string> propNames;

public:
  DeleteProperty(PClip _child, AVSValue _propNames, IScriptEnvironment* env);
  ~DeleteProperty();
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
  int __stdcall SetCacheHints(int cachehints, int frame_range);
  static AVSValue __cdecl Create(AVSValue args, void* , IScriptEnvironment* env);
};

class ClearProperties : public GenericVideoFilter
{
private:

public:
  ClearProperties(PClip _child, IScriptEnvironment* env);
  ~ClearProperties();
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
  int __stdcall SetCacheHints(int cachehints, int frame_range);
  static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env);
};

class CopyProperties : public GenericVideoFilter
{
private:
  PClip child2;
  bool merge;
  bool propNames_defined;
  std::vector<std::string> propNames;
  bool exclude;

public:
  CopyProperties(PClip _child, PClip _child2, bool _merge, AVSValue _propNames, bool _exclude, IScriptEnvironment* env);
  ~CopyProperties();
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
  int __stdcall SetCacheHints(int cachehints, int frame_range);
  static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env);
};

class ShowProperties : public GenericVideoFilter
{
private:
  int size;
  bool showtype;

public:
  ShowProperties(PClip _child, int size, bool showtype, IScriptEnvironment* env);
  ~ShowProperties();
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
  int __stdcall SetCacheHints(int cachehints, int frame_range);
  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env);
};
