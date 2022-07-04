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

#include "conditional_reader.h"
#include "../core/internal.h"
#include "../core/parser/scriptparser.h"
#include <cstdlib>

#ifdef AVS_WINDOWS
    #include <avs/win.h>
#else
    #include <avs/posix.h>
#endif

#include <avs/minmax.h>
#include "../core/parser/scriptparser.h"
#include "../core/AVSMap.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>
#include <cstring>
#include "../convert/convert_helper.h"


/*****************************************************************************
 *  Helper code from XviD (http://www.xvid.org)
 *
 *  Copyright (C)      2002 Foxer <email?>
 *                     2002 Dirk Knop <dknop@gwdg.de>
 *                2002-2003 Edouard Gomez <ed.gomez@free.fr>
 *                     2003 Pete Ross <pross@xvid.org>
 *****************************************************************************/

/* Default buffer size for reading lines */
#define BUF_SZ   1024


/* This function returns an allocated string containing a complete line read
 * from the file starting at the current position */
static char *
readline(FILE *f)
{
	char *buffer = NULL;
	int buffer_size = 0;
	int pos = 0;

	for (;;) {
		int c;

		/* Read a character from the stream */
		c = fgetc(f);

		/* Is that EOF or new line ? */
		if(c == EOF || c == '\n')
			break;

		/* Do we have to update buffer ? */
		if(pos >= buffer_size - 1) {
			buffer_size += BUF_SZ;
			char *tmpbuffer = (char*)realloc(buffer, buffer_size);
			if (tmpbuffer == NULL) {
				free(buffer);
				return(NULL);
		    }
			buffer = tmpbuffer;
		}

		buffer[pos] = (char)c;
		pos++;
	}

	/* Read \n or EOF */
	if (buffer == NULL) {
		/* EOF, so we reached the end of the file, return NULL */
		if(feof(f))
			return(NULL);

		/* Just an empty line with just a newline, allocate a 1 byte buffer to
		 * store a zero length string */
		buffer = (char*)malloc(1);
		if(buffer == NULL)
			return(NULL);
	}

	/* Zero terminated string */
	if (pos && buffer[pos-1] == '\r')
		buffer[pos-1] = '\0';
	else
		buffer[pos] = '\0';

	return(buffer);
}

/* This function returns a pointer to the first non space char in the given
 * string or the end of the string */

static char *
skipspaces(char *string)
{
	if (string == NULL) return(NULL);

	while (*string != '\0') {
		/* Test against space chars */
		if (!isspace(*string)) return(string);
		string++;
	}
	return(string);
}

/* This function returns a pointer to the first space char in the given
 * string or the end of the string */

static char *
findspace(char *string)
{
	if (string == NULL) return(NULL);

	while (*string != '\0') {
		/* Test against space chars */
		if (isspace(*string)) return(string);
		string++;
	}
	return(string);
}

/* This function returns a boolean that tells if the string is only a
 * comment */
static int
iscomment(char *string)
{
	const char comments[] =
		{
			'#',';', '%', '\0'
		};
	const char *cmtchar = comments;
	int iscomment = 0;

	if (string == NULL) return(1);

	string = skipspaces(string);

	while(*cmtchar != '\0') {
		if(*string == *cmtchar) {
			iscomment = 1;
			break;
		}
		cmtchar++;
	}

	return(iscomment);
}


// Reader ------------------------------------------------


ConditionalReader::ConditionalReader(PClip _child, const char* filename, const char _varname[],
  bool _show, const char *_condVarSuffix, bool _local, IScriptEnvironment* env)
 : GenericVideoFilter(_child), show(_show), mode(MODE_UNKNOWN), offset(0), local(_local), stringcache(0)
{
  FILE * f;
  char *line = 0;
  int lines;

  variableName = _varname; // std::string
  if (_condVarSuffix[0])
    variableName += _condVarSuffix; // append if parameter exists
  variableNameFixed = env->SaveString(variableName.c_str());

  if ((f = fopen(filename, "rb")) == NULL)
    env->ThrowError("ConditionalReader: Could not open file '%s'.", filename);

  lines = 0;

  try {
    while ((line = readline(f)) != NULL) {
      char *ptr;
      int fields;

      lines++;

      /* We skip spaces */
      ptr = skipspaces(line);

      /* Skip coment lines or empty lines */
      if(iscomment(ptr) || *ptr == '\0') {
        free(line);
        line = 0;
        continue;
      }

      if (mode == MODE_UNKNOWN) {
        // We have not recieved a mode - We expect type.
        char* keyword = ptr;

        ptr = findspace(ptr);
        if (*ptr) {
          *ptr++ = '\0';
          if (!lstrcmpi(keyword, "type")) {
            /* We skip spaces */
            char* type = skipspaces(ptr);

            ptr = findspace(type);
            *ptr = '\0';

            if (!lstrcmpi(type, "int")) {
              mode = MODE_INT;
              intVal = new int[vi.num_frames];
            } else if (!lstrcmpi(type, "float")) {
              mode = MODE_FLOAT;
              floatVal = new float[vi.num_frames];
            } else if (!lstrcmpi(type, "bool")) {
              mode = MODE_BOOL;
              boolVal = new bool[vi.num_frames];
            } else if (!lstrcmpi(type, "string")) {
              mode = MODE_STRING;
              stringVal = new const char*[vi.num_frames];
            } else {
              ThrowLine("ConditionalReader: Unknown 'Type' specified in line %d", lines, env);
            }// end if compare type
            SetRange(0, vi.num_frames-1, AVSValue());
          }// end if compare keyword
        }// end if fields

      } else { // We have a defined mode and allocated the values.

        char* keyword = ptr;
        char* type = findspace(keyword);

        if (*type) *type++ = '\0';

        if (!lstrcmpi(keyword, "default")) {
          AVSValue def = ConvertType(type, lines, env);
          SetRange(0, vi.num_frames-1, def);

        } else if (!lstrcmpi(keyword, "offset")) {
          fields = sscanf(type, "%d", &offset);
          if (fields != 1)
            ThrowLine("ConditionalReader: Could not read Offset in line %d", lines, env);

        } else if (keyword[0] == 'R' || keyword[0] == 'r') {  // Range
          int start;
          int stop;

          type = skipspaces(type);
          fields = sscanf(type, "%d", &start);

          type = findspace(type);
          type = skipspaces(type);
          fields += sscanf(type, "%d", &stop);

          type = findspace(type);
          if (!*type || fields != 2)
            ThrowLine("ConditionalReader: Could not read Range in line %d", lines, env);

          if (start > stop)
            ThrowLine("ConditionalReader: The Range start frame is after the end frame in line %d", lines, env);

          AVSValue set = ConvertType(type+1, lines, env);
          SetRange(start, stop, set);

        } else if (keyword[0] == 'I' || keyword[0] == 'i') {  // Interpolate
          if (mode == MODE_BOOL)
            ThrowLine("ConditionalReader: Cannot Interpolate booleans in line %d", lines, env);

          if (mode == MODE_STRING)
            ThrowLine("ConditionalReader: Cannot Interpolate strings in line %d", lines, env);

          type = skipspaces(type);
          int start;
          int stop;
          char start_value[64];
          char stop_value[64];
          fields = sscanf(type, "%d %d %63s %63s", &start, &stop, start_value, stop_value);

          if (fields != 4)
            ThrowLine("ConditionalReader: Could not read Interpolation range in line %d", lines, env);
          if (start > stop)
            ThrowLine("ConditionalReader: The Interpolation start frame is after the end frame in line %d", lines, env);

          start_value[63] = '\0';
          AVSValue set_start = ConvertType(start_value, lines, env);

          stop_value[63] = '\0';
          AVSValue set_stop = ConvertType(stop_value, lines, env);

          const int range = stop-start;
          const double diff = (set_stop.AsFloat() - set_start.AsFloat()) / range;
          for (int i = 0; i<=range; i++) {
            const double n = i * diff + set_start.AsFloat();
            SetFrame(i+start, (mode == MODE_FLOAT)
                    ? AVSValue(n)
                    : AVSValue((int)(n+0.5)));
          }
        } else {
          int cframe;
          fields = sscanf(keyword, "%d", &cframe);
          if ((*type || mode == MODE_STRING) && fields == 1) { // allow empty string
            AVSValue set = ConvertType(type, lines, env);
            SetFrame(cframe, set);
          } else {
            ThrowLine("ConditionalReader: Do not understand line %d", lines, env);
          }
        }

      } // End we have defined type
      free(line);
      line = 0;
    }// end while still some file left to read.
  }
  catch (...) {
    free(line);
    fclose(f);
    CleanUp();
    throw;
  }

  /* We are done with the file */
  fclose(f);

  if (mode == MODE_UNKNOWN)
    env->ThrowError("ConditionalReader: Type was not defined!");

}



// Converts from the char array given to the type specified.

AVSValue ConditionalReader::ConvertType(const char* content, int line, IScriptEnvironment* env)
{
  if (mode == MODE_UNKNOWN)
    ThrowLine("ConditionalReader: Type has not been defined. Line %d", line, env);

  int fields;
  switch (mode) {
    case MODE_INT:
      int ival;
      fields = sscanf(content, "%d", &ival);
      if (fields != 1)
        ThrowLine("ConditionalReader: Could not find an expected integer at line %d!", line, env);

      return AVSValue(ival);

    case MODE_FLOAT:
      float fval;
      fields = sscanf(content, "%e", &fval);
      if (fields != 1)
        ThrowLine("ConditionalReader: Could not find an expected float at line %d!", line, env);

      return AVSValue(fval);

    case MODE_BOOL:
      char bval[8];
      bval[0] = '\0';
      fields = sscanf(content, "%7s", bval);
      bval[7] = '\0';
      if (!lstrcmpi(bval, "true")) {
        return AVSValue(true);
      }
      else if (!lstrcmpi(bval, "t")) {
        return AVSValue(true);
      }
      else if (!lstrcmpi(bval, "yes")) {
        return AVSValue(true);
      }
      else if (!lstrcmp(bval, "1")) {
        return AVSValue(true);
      }
      else if (!lstrcmpi(bval, "false")) {
        return AVSValue(false);
      }
      else if (!lstrcmpi(bval, "f")) {
        return AVSValue(false);
      }
      else if (!lstrcmpi(bval, "no")) {
        return AVSValue(false);
      }
      else if (!lstrcmp(bval, "0")) {
        return AVSValue(false);
      }
      ThrowLine("ConditionalReader: Boolean value was not true or false in line %d", line, env);

    case MODE_STRING:
      StringCache *str;

      // Look for an existing duplicate
      for (str = stringcache; str; str = str->next ) {
        if (!lstrcmp(str->string, content)) break;
      }
      // Could not find one, add it
      if (!str) {
        str = new StringCache;
        str->string = _strdup(content);
        str->next   = stringcache;
        stringcache = str;
      }
      return AVSValue(str->string);
  }
  return AVSValue();
}


// Sets range with both start and stopframe inclusive.

void ConditionalReader::SetRange(int start_frame, int stop_frame, AVSValue v) {
  int i;
  start_frame = max(start_frame+offset, 0);
  stop_frame = min(stop_frame+offset, vi.num_frames-1);
  int p;
  float q;
  bool r;
  const char* s;

  switch (mode) {
    case MODE_INT:
      p = v.Defined() ? v.AsInt() : 0;
      for (i = start_frame; i <= stop_frame; i++) {
        intVal[i] = p;
      }
      break;
    case MODE_FLOAT:
      q = v.Defined() ? v.AsFloatf() : 0.0f;
      for (i = start_frame; i <= stop_frame; i++) {
        floatVal[i] = q;
      }
      break;
    case MODE_BOOL:
      r = v.Defined() ? v.AsBool() : false;
      for (i = start_frame; i <= stop_frame; i++) {
        boolVal[i] = r;
      }
      break;
    case MODE_STRING:
      s = v.AsString("");
      for (i = start_frame; i <= stop_frame; i++) {
        stringVal[i] = s;
      }
      break;
  }
}

// Sets the value of one frame.

void ConditionalReader::SetFrame(int framenumber, AVSValue v) {

  if ((framenumber+offset) < 0 || (framenumber+offset) > vi.num_frames-1 )
    return;

  switch (mode) {
    case MODE_INT:
      intVal[framenumber+offset] = v.AsInt();
      break;
    case MODE_FLOAT:
      floatVal[framenumber+offset] = v.AsFloatf();
      break;
    case MODE_BOOL:
      boolVal[framenumber+offset] = v.AsBool();
      break;
    case MODE_STRING:
      stringVal[framenumber+offset] = v.AsString("");
      break;
  }
}

// Get the value of a frame.
AVSValue ConditionalReader::GetFrameValue(int framenumber) {
  framenumber = clamp(framenumber, 0, vi.num_frames-1);

  switch (mode) {
    case MODE_INT:
      return AVSValue(intVal[framenumber]);

    case MODE_FLOAT:
      return AVSValue(floatVal[framenumber]);

    case MODE_BOOL:
      return AVSValue(boolVal[framenumber]);

    case MODE_STRING:
      return AVSValue(stringVal[framenumber]);

  }
  return AVSValue(0);
}

// Destructor
ConditionalReader::~ConditionalReader(void)
{
  CleanUp();
}


void ConditionalReader::CleanUp(void)
{
  switch (mode) {
    case MODE_INT:
      delete[] intVal;
      break;
    case MODE_FLOAT:
      delete[] floatVal;
      break;
    case MODE_BOOL:
      delete[] boolVal;
      break;
    case MODE_STRING:
      delete[] stringVal;

      //free the cached strings
      for (StringCache* str = stringcache; str; ) {
        StringCache* curr = str;
        free(str->string);
        str = str->next;
        delete curr;
      }
      stringcache = 0;

      break;
  }
  mode = MODE_UNKNOWN;
}


void ConditionalReader::ThrowLine(const char* err, int line, IScriptEnvironment* env) {
  env->ThrowError(err, line);
}


PVideoFrame __stdcall ConditionalReader::GetFrame(int n, IScriptEnvironment* env)
{
  AVSValue v = GetFrameValue(n);

  InternalEnvironment* envI = static_cast<InternalEnvironment*>(env);

  std::unique_ptr<GlobalVarFrame> var_frame;

  AVSValue child_val = child;

  if (!local) {
    env->SetGlobalVar(variableNameFixed, v);
  }
  else {
    // Neo's default, correct but incompatible with previous Avisynth versions
    var_frame = std::unique_ptr<GlobalVarFrame>(new GlobalVarFrame(envI)); // allocate new frame
    env->SetGlobalVar(variableNameFixed, v);
  }


  PVideoFrame src = child->GetFrame(n,env);

  if (show) {
    AVSValue v2 = env->Invoke("String", v);
    env->MakeWritable(&src);
    env->ApplyMessage(&src, vi, v2.AsString(""), vi.width/2, 0xa0a0a0, 0, 0);
  }
  return src;
}

int __stdcall ConditionalReader::SetCacheHints(int cachehints, int frame_range)
{
  switch (cachehints)
  {
  case CACHE_GET_MTMODE:
    return MT_NICE_FILTER;
  case CACHE_GET_DEV_TYPE:
    return (child->GetVersion() >= 5) ? child->SetCacheHints(CACHE_GET_DEV_TYPE, 0) : 0;
  }
  return 0;  // We do not pass cache requests upwards.
}



AVSValue __cdecl ConditionalReader::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  const bool runtime_local_default = false; // Avisynth compatibility: false, Neo: true.

  return new ConditionalReader(args[0].AsClip(), args[1].AsString(""), args[2].AsString("Conditional") , args[3].AsBool(false), args[4].AsString(""), args[5].AsBool(runtime_local_default), env);
}


// Write ------------------------------------------------


static const char EMPTY[]  = "";
static const char AplusT[] = "a+t";
static const char WplusT[] = "w+t";


Write::Write(PClip _child, const char* _filename, AVSValue args, int _linecheck, bool _append, bool _flush, bool _local, IScriptEnvironment* env) :
  GenericVideoFilter(_child), linecheck(_linecheck), flush(_flush), append(_append), local(_local), arglist(0)
{
#ifdef AVS_WINDOWS
  _fullpath(filename, _filename, _MAX_PATH);
#else
  realpath(_filename, filename);
#endif

  fout = fopen(filename, append ? AplusT : WplusT);	//append or purge file
  if (!fout) env->ThrowError("Write: File '%s' cannot be opened.", filename);

  if (flush) fclose(fout);	//will be reopened in FileOut

  arrsize = args.ArraySize();

  arglist = new exp_res[arrsize];

  for (int i = 0; i < arrsize; i++) {
    arglist[i].expression = args[i];
    arglist[i].string = EMPTY;
  }

  if (linecheck != -1 && linecheck != -2)
    return;

  InternalEnvironment* envI = static_cast<InternalEnvironment*>(env);

  AVSValue prev_last;
  AVSValue prev_current_frame;
  std::unique_ptr<GlobalVarFrame> var_frame;

  AVSValue child_val = child;

  if (!local) {
    prev_last = env->GetVarDef("last");  // Store previous last
    prev_current_frame = env->GetVarDef("current_frame");  // Store previous current_frame
    env->SetVar("last", child_val);       // Set implicit last
    env->SetVar("current_frame", (AVSValue)linecheck);  // special -1 or -2
  }
  else {
    // Neo's default, correct but incompatible with previous Avisynth versions
    var_frame = std::unique_ptr<GlobalVarFrame>(new GlobalVarFrame(envI)); // allocate new frame
    env->SetGlobalVar("last", child_val);       // Set explicit last
    env->SetGlobalVar("current_frame", (AVSValue)linecheck);  // special -1 or -2
  }

  Write::DoEval(env); // at both write at start and write at end

  if (linecheck == -1) { //write at start
    Write::FileOut(env, AplusT);
  }

  if (!local) {
    env->SetVar("last", prev_last);       // Restore implicit last
    env->SetVar("current_frame", prev_current_frame);       // Restore current_frame
  }
}

PVideoFrame __stdcall Write::GetFrame(int n, IScriptEnvironment* env) {

  //changed to call write AFTER the child->GetFrame


  PVideoFrame tmpframe = child->GetFrame(n, env);

  if (linecheck < 0) return tmpframe;	//do nothing here when writing only start or end

  InternalEnvironment* envI = static_cast<InternalEnvironment*>(env);

  AVSValue prev_last;
  AVSValue prev_current_frame;
  std::unique_ptr<GlobalVarFrame> var_frame;

  AVSValue child_val = child;

  if (!local) {
    prev_last = env->GetVarDef("last");  // Store previous last
    prev_current_frame = env->GetVarDef("current_frame");  // Store previous current_frame
    env->SetVar("last", (AVSValue)child_val);       // Set implicit last
    env->SetVar("current_frame", (AVSValue)n);  // Set frame to be tested by the conditional filters.
  }
  else {
    // Neo's default, correct but incompatible with previous Avisynth versions
    var_frame = std::unique_ptr<GlobalVarFrame>(new GlobalVarFrame(envI)); // allocate new frame
    env->SetGlobalVar("last", child_val);       // Set implicit last (to avoid recursive stack calls?)
    env->SetGlobalVar("current_frame", (AVSValue)n);  // Set frame to be tested by the conditional filters.
  }

  if (Write::DoEval(env)) {
    Write::FileOut(env, AplusT);
  }

  if (!local) {
    env->SetVar("last", prev_last);       // Restore implicit last
    env->SetVar("current_frame", prev_current_frame);       // Restore current_frame
  }

  return tmpframe;

};

Write::~Write(void) {
	if (linecheck == -2) {	//write at end
		Write::FileOut(0, append ? AplusT : WplusT); // Allow for retruncating at actual end
	}
	if (!flush) fclose(fout);

	delete[] arglist;
};

void Write::FileOut(IScriptEnvironment* env, const char* mode) {
	int i;
	if (flush) {
		fout = fopen(filename, mode);
		if (!fout) {
			if (env) env->ThrowError("Write: File '%s' cannot be opened.", filename);
			return;
		}
	}
	for (i= ( (linecheck==1) ? 1 : 0) ; i<arrsize; i++ ) {
		fputs(arglist[i].string, fout);
	}
	fputs("\n", fout);
	if (flush) {
		fclose(fout);
	}
}

bool Write::DoEval( IScriptEnvironment* env) {
	bool keep_this_line = true;
	int i;
	AVSValue expr;
	AVSValue result;

	for (i=0; i<arrsize; i++) {
		expr = arglist[i].expression;

		if ( (linecheck==1) && (i==0)) {
			try {
        if (expr.IsFunction()) {
          result = env->Invoke3(child, expr.AsFunction(), AVSValue(nullptr, 0));
        }
        else {
          expr = expr.AsString(EMPTY);
          result = env->Invoke("Eval", expr);
        }
				if (!result.AsBool(true)) {
					keep_this_line = false;
					break;
				}
			} catch (const AvisynthError&) {
//				env->ThrowError("Write: Can't eval linecheck expression!"); // results in KEEPING the line
			}
		} else {
			try {
        if (expr.IsFunction()) {
          result = env->Invoke3(child, expr.AsFunction(), AVSValue(nullptr, 0));
        }
        else {
          expr = expr.AsString(EMPTY);
          result = env->Invoke("Eval", expr);
        }
				result = env->Invoke("string",result);	//convert all results to a string
				arglist[i].string = result.AsString(EMPTY);
			} catch (const AvisynthError &error) {
				arglist[i].string = env->SaveString(error.msg);
			}
		}
	}
	return keep_this_line;
}

int __stdcall Write::SetCacheHints(int cachehints, int frame_range)
{
  switch (cachehints)
  {
  case CACHE_GET_MTMODE:
    return MT_SERIALIZED;
  }
  return 0;  // We do not pass cache requests upwards.
}

AVSValue __cdecl Write::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  bool runtime_local_default = false;
  // Param 2: string/function or array of strings/functions
  if (args[2].IsFunction() || (args[2].IsArray() && args[2].ArraySize() > 0 && args[2][0].IsFunction()))
    runtime_local_default = true;
  // Avisynth compatibility: false, Neo: true. functions are legacy Neo

	return new Write(args[0].AsClip(), args[1].AsString(EMPTY), args[2], 0, args[3].AsBool(true),args[4].AsBool(true), args[5].AsBool(runtime_local_default), env);
}

AVSValue __cdecl Write::Create_If(AVSValue args, void*, IScriptEnvironment* env)
{
  bool runtime_local_default = false;
  // Param 2: string/function or array of strings/functions
  if (args[2].IsFunction() || (args[2].IsArray() && args[2].ArraySize() > 0 && args[2][0].IsFunction()))
    runtime_local_default = true;
  // Avisynth compatibility: false, Neo: true. functions are legacy Neo

	return new Write(args[0].AsClip(), args[1].AsString(EMPTY), args[2], 1, args[3].AsBool(true),args[4].AsBool(true), args[5].AsBool(runtime_local_default), env);
}

AVSValue __cdecl Write::Create_Start(AVSValue args, void*, IScriptEnvironment* env)
{
  bool runtime_local_default = false;
  // Param 2: string/function or array of strings/functions
  /*if (args[2].IsFunction() || (args[2].IsArray() && args[2].ArraySize() > 0 && args[2][0].IsFunction()))
    runtime_local_default = true;
  */
  // Avisynth compatibility: false, Neo: also false as of 2020.03.28

	return new Write(args[0].AsClip(), args[1].AsString(EMPTY), args[2], -1, args[3].AsBool(false), true, args[4].AsBool(runtime_local_default), env);
}

AVSValue __cdecl Write::Create_End(AVSValue args, void*, IScriptEnvironment* env)
{
  bool runtime_local_default = false;
  // Param 2: string/function or array of strings/functions
  /*
  if (args[2].IsFunction() || (args[2].IsArray() && args[2].ArraySize() > 0 && args[2][0].IsFunction()))
    runtime_local_default = true;
  */
  // Avisynth compatibility: false, Neo: also false as of 2020.03.28

	return new Write(args[0].AsClip(), args[1].AsString(EMPTY), args[2], -2, args[3].AsBool(true), args[4].AsBool(runtime_local_default), true, env);
}


UseVar::UseVar(PClip _child, AVSValue vars, IScriptEnvironment* env)
   : GenericVideoFilter(_child)
{
   vars_.resize(vars.ArraySize());
   for (int i = 0; i < vars.ArraySize(); ++i) {
      auto name = vars_[i].name = vars[i].AsString();
      if (!env->GetVarTry(name, &vars_[i].val)) {
        env->ThrowError("UseVar: No variable named %s", name);
      }
   }
}

UseVar::~UseVar() { }

PVideoFrame __stdcall UseVar::GetFrame(int n, IScriptEnvironment* env)
{
   GlobalVarFrame var_frame(static_cast<InternalEnvironment*>(env)); // allocate new frame

   // set variables
   for (int i = 0; i < (int)vars_.size(); ++i) {
      env->SetGlobalVar(vars_[i].name, vars_[i].val);
   }

   return child->GetFrame(n, env);
}

int __stdcall UseVar::SetCacheHints(int cachehints, int frame_range) {
  switch (cachehints)
  {
  case CACHE_GET_MTMODE:
    return MT_NICE_FILTER;
  case CACHE_GET_DEV_TYPE:
    return (child->GetVersion() >= 5) ? child->SetCacheHints(CACHE_GET_DEV_TYPE, 0) : 0;
  }
  return 0;  // We do not pass cache requests upwards.
}

AVSValue __cdecl UseVar::Create(AVSValue args, void* user_data, IScriptEnvironment* env)
{
   return new UseVar(args[0].AsClip(), args[1], env);
}

#define W_DIVISOR 5  // Width divisor for onscreen messages


// Avisynth+ frame property support
//**************************************************
// propSet, propSetInt, propSetFloat, propSetString

SetProperty::SetProperty(PClip _child, const char* name, const AVSValue& value, const int kind,
  const int mode, IScriptEnvironment* env)
  : GenericVideoFilter(_child)
  , name(name)
  , value(value)
  , kind(kind)
  , append_mode(mode)
{ }

SetProperty::~SetProperty() { }

PVideoFrame __stdcall SetProperty::GetFrame(int n, IScriptEnvironment* env)
{
  // parameter type to set, comes different for each script function version
  int propType = kind;

  // 0: auto by result type
  // 1: integer from function
  // 2: float from function
  // 3: char (null terminated data) from function
  // 4: array from function (all elements have the same type)
  // 5: clip from function
  // 10: integer from direct
  // 11: float from direct
  // 12: string from direct
  // 13: array from direct (all elements have the same type)
  // 14: clip from direct

  AVSValue result;
  const char* error_msg = nullptr;
  if (value.IsFunction()) {
    PFunction func = value.AsFunction();
    try {
      const AVSValue empty_args_array = AVSValue(nullptr, 0); // invoke's parameter is const AVSValue&, don't do it inline.
      result = env->Invoke3(child, func, empty_args_array);
    }
    catch (IScriptEnvironment::NotFound) {
      error_msg = env->Sprintf("AddProperties: Invalid function parameter type '%s'(%s)\n"
        "Function should have no argument",
        func->GetDefinition()->param_types, func->ToString(env));
    }
    catch (const AvisynthError& error) {
      error_msg = env->Sprintf("%s\nAddProperties: Error in %s",
        error.msg, func->ToString(env));
    }
  }
  else {
    // direct input
    result = value;
    propType = 0; // autodetect type
  }

  PVideoFrame frame = child->GetFrame(n, env);

  if (error_msg) {
    env->MakeWritable(&frame);
    env->ApplyMessage(&frame, vi, error_msg, vi.width / W_DIVISOR, 0xa0a0a0, 0, 0);
    return frame;
  }

/*
  usage:
   propSet("hello",1)
   ScriptClip("""propSetInt("frameluma",func(AverageLuma))""")
   ScriptClip("""propSet("frameluma2",AverageLuma)""")
   ScriptClip("""SubTitle(string(propGetInt("frameluma")))""")
  or
   ps = func(propSetterFunc) # make function object from function
   ScriptClip(ps) # pass function object to scriptclip
   ScriptClip(function[](clip c) { SubTitle(string(propGetInt("frameprop_demo")), y=20) })
   ScriptClip(function[](clip c) { SubTitle(string(propGetInt("frameprop_demo2")), y=40) })
   function propSetterFunc(Clip x) {
    x
    propSetInt("frameprop_demo", func(AverageLuma))
    propSetInt("frameprop_demo2", function[]() { current_frame })
  }
*/

  try {
    // check auto
    if (propType == 0) {
      // 'u'nset, 'i'nteger, 'f'loat, 's'string, 'c'lip, 'v'ideoframe, 'm'ethod };
      if (result.IsInt())
        propType = 1;
      else if (result.IsFloat())
        propType = 2;
      else if (result.IsString())
        propType = 3;
      else if (result.IsArray())
        propType = 4;
      else if (result.IsClip())
        propType = 5;
      else
        env->ThrowError("Invalid return type (Was a %s)", GetAVSTypeName(result));
    }

    // env->MakeWritable(&frame);
    // do we need this? Yes! better: MakePropertyWritable since V9 interface

    // Check setting a constant value to the same as it was before (int, float, string case)
    // We'd avoid even MakePropertyWritable and return a fully unaltered frame
    do {
      if (append_mode != AVSPropAppendMode::PROPAPPENDMODE_REPLACE) break; // only optimize replace
      if (propType != 1 && propType != 2 && propType != 3) break; // only optimize traditional types

      const AVSMap* avsmap_r = env->getFramePropsRO(frame);
      auto size = env->propNumElements(avsmap_r, name); // 1 if single item, >1 size of array, 0 if not exists
      if(size != 1) break; // no such property or property is an array

      char x = env->propGetType(avsmap_r, name);
      if (propType == 1 && x != 'i') break; // int does not match
      if (propType == 2 && x != 'f') break; // float does not match
      if (propType == 3 && x != 's') break; // string does not match
      
      if (propType == 1) {
        if (!result.IsInt()) break;
        auto val = env->propGetInt(avsmap_r, name, 0, nullptr);
        if (val == result.AsInt()) return frame; // value match -> return unaltered
      } 
      else if (propType == 2) {
        if (!result.IsFloat()) break;
        auto val = env->propGetFloat(avsmap_r, name, 0, nullptr);
        if (val == result.AsFloat()) return frame; // value match -> return unaltered
      }
      else if (propType == 3) {
        if (!result.IsString()) break;
        const char* val_to_set = result.AsString();
        if (val_to_set == nullptr) break;
        auto length_to_set = strlen(val_to_set);
        auto length = env->propGetDataSize(avsmap_r, name, 0, nullptr);
        if (length != length_to_set) break; // different size
        const char* val_storage = env->propGetData(avsmap_r, name, 0, nullptr);
        if(std::memcmp(val_to_set, val_storage, length) == 0) return frame; // value match -> return unaltered
      }
      break;
    } while (0);

    env->MakePropertyWritable(&frame);
    AVSMap* avsmap = env->getFramePropsRW(frame);

    int res = 0;

    // special case: zero sized array -> entry deleted
    if (result.IsArray() && result.ArraySize() == 0)
      res = env->propDeleteKey(avsmap, name); // 0 is success
    else if (propType == 1 && result.IsInt())
      res = env->propSetInt(avsmap, name, result.AsInt(), append_mode);
    else if (propType == 2 && result.IsFloat())
      res = env->propSetFloat(avsmap, name, result.AsFloat(), append_mode);
    else if (propType == 3 && result.IsString())
    {
      const char* s = result.AsString(); // no need for SaveString, it has its own storage
      res = env->propSetData(avsmap, name, s, -1, append_mode); // -1: auto string length
    }
    else if (propType == 4 && result[0].IsInt())
    {
      int size = result.ArraySize();
      std::vector<int64_t> int64array(size); // avs can do int only, temporary array needed
      for (int i = 0; i < size; i++) {
        if (!result[i].IsInt())
          env->ThrowError("Wrong data type in property '%s': all array elements should be the same (integer) type", name);
        int64array[i] = result[i].AsInt(); // all elements should be int
      }
      res = env->propSetIntArray(avsmap, name, int64array.data(), size);
    }
    else if (propType == 4 && result[0].IsFloat())
    {
      int size = result.ArraySize();
      std::vector<double> d_array(size); // avs can do float only, temporary array needed
      for (int i = 0; i < size; i++) {
        if (!result[i].IsFloat())
          env->ThrowError("Wrong data type in property '%s': all array elements should be the same (float) type", name);
        d_array[i] = result[i].AsFloat(); // all elements should be float or int
      }
      res = env->propSetFloatArray(avsmap, name, d_array.data(), size);
    }
    else if (propType == 4 && result[0].IsString())
    {
      const int size = result.ArraySize();
      // no such api like propSetDataArray
      env->propDeleteKey(avsmap, name);
      for (int i = 0; i < size; i++) {
        if (!result[i].IsString())
          env->ThrowError("Wrong data type in property '%s': all array elements should be the same (string) type", name);
        res = env->propSetData(avsmap, name, result[i].AsString(), -1, AVSPropAppendMode::PROPAPPENDMODE_APPEND); // all elements should be string
        if (res)
          break;
      }
    }
    else if (propType == 4 && result[0].IsClip())
    {
      int size = result.ArraySize();
      // no such api like propSetClipArray
      env->propDeleteKey(avsmap, name);
      for (int i = 0; i < size; i++) {
        if (!result[i].IsClip())
          env->ThrowError("Wrong data type in property '%s': all array elements should be the same (Clip) type", name);
        PClip clip = result[i].AsClip();
        res = env->propSetClip(avsmap, name, clip, AVSPropAppendMode::PROPAPPENDMODE_APPEND); // all elements should be Clip
        if (res)
          break;
      }
    }
    else if (propType == 5 && result.IsClip()) {
      PClip clp = result.AsClip();
      res = env->propSetClip(avsmap, name, clp, append_mode);
    }
    else
    {
      env->ThrowError("Wrong data type, property '%s' type is not %s", name, GetAVSTypeName(result));
    }

    if (res)
      env->ThrowError("error setting property '%s', error = %d", name, res); // fixme: res reasons
  }
  catch (const AvisynthError& error) {
    error_msg = env->Sprintf("propAdd: %s", error.msg);
  }

  if (error_msg) {
    env->MakeWritable(&frame);
    env->ApplyMessage(&frame, vi, error_msg, vi.width / W_DIVISOR, 0xa0a0a0, 0, 0);
  }

  return frame;
}

int __stdcall SetProperty::SetCacheHints(int cachehints, int frame_range)
{
  AVS_UNUSED(frame_range);
  switch (cachehints)
  {
  case CACHE_GET_MTMODE:
    return MT_NICE_FILTER;
  }
  return 0;  // We do not pass cache requests upwards.
}

AVSValue __cdecl SetProperty::Create(AVSValue args, void* user_data, IScriptEnvironment* env)
{
  const int kind = (int)(intptr_t)user_data;
  const int defaultMode = AVSPropAppendMode::PROPAPPENDMODE_REPLACE;

  int mode = AVSPropAppendMode::PROPAPPENDMODE_REPLACE;
  if(kind != 4) // at propSetArray there is no mode parameter
    mode = args[3].AsInt(defaultMode);

  /*
    paReplace = 0,
    paAppend = 1,
    paTouch = 2
  */
  return new SetProperty(args[0].AsClip(), args[1].AsString(), args[2], kind, mode, env);
}

//**************************************************
// helper for property name list

static bool PropNamesToArray(const char* filtername, AVSValue& _propNames, std::vector<std::string>& propNames, IScriptEnvironment* env)
{
  if (!_propNames.Defined())
    return false;
  const int size = _propNames.ArraySize();
  propNames.resize(size);
  for (int i = 0; i < size; i++) {
    const char* name = _propNames[i].AsString();
    if (name == nullptr || !*name)
      env->ThrowError("%s error: list contains empty name", filtername);
    propNames[i] = name;
  }
  return true;
}

//**************************************************
// propDelete

DeleteProperty::DeleteProperty(PClip _child, AVSValue _propNames, IScriptEnvironment* env)
  : GenericVideoFilter(_child)
{ 
  propNames_defined = PropNamesToArray("propDelete", _propNames, propNames, env);
}

DeleteProperty::~DeleteProperty() { }

PVideoFrame __stdcall DeleteProperty::GetFrame(int n, IScriptEnvironment* env)
{
  PVideoFrame frame = child->GetFrame(n, env);

  // Erase only if needed
  const AVSMap* avsmap_test = env->getFramePropsRO(frame);
  if (0 == env->propNumKeys(avsmap_test))
    return frame;

  /*
    usage:
      propDelete("frameluma")
      propDelete(["_Matrix","prop2"])
  */

  env->MakePropertyWritable(&frame);

  AVSMap* avsmap = env->getFramePropsRW(frame);

  /*
  // ignore when property does not exist
  if (!res) {
    const char *error_msg = env->Sprintf("propDelete: error deleting property '%s'", name);
    env->ApplyMessage(&frame, vi, error_msg, vi.width / W_DIVISOR, 0xa0a0a0, 0, 0);
  }
  */
  // delete selected names
  AVSMap* mapv = env->getFramePropsRW(frame);
  for (auto &s : propNames) {
    const char* key = s.c_str();
    env->propDeleteKey(avsmap, key); // 0 is success, ignored
  }

  return frame;
}

int __stdcall DeleteProperty::SetCacheHints(int cachehints, int frame_range)
{
  AVS_UNUSED(frame_range);
  switch (cachehints)
  {
  case CACHE_GET_MTMODE:
    return MT_NICE_FILTER;
  }
  return 0;  // We do not pass cache requests upwards.
}

AVSValue __cdecl DeleteProperty::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  return new DeleteProperty(args[0].AsClip(), args[1], env);
}

//**************************************************
// propDelete

ClearProperties::ClearProperties(PClip _child, IScriptEnvironment* env)
  : GenericVideoFilter(_child)
{ }

ClearProperties::~ClearProperties() { }

PVideoFrame __stdcall ClearProperties::GetFrame(int n, IScriptEnvironment* env)
{
  PVideoFrame frame = child->GetFrame(n, env);

  /*
    usage:
      ScriptClip("""propClear()""")
  */

  // Erase only if needed
  const AVSMap* avsmap_test = env->getFramePropsRO(frame);
  if (0 != env->propNumKeys(avsmap_test)) {
    env->MakePropertyWritable(&frame);
    AVSMap* avsmap = env->getFramePropsRW(frame);
    env->clearMap(avsmap);
  }

  return frame;
}

int __stdcall ClearProperties::SetCacheHints(int cachehints, int frame_range)
{
  AVS_UNUSED(frame_range);
  switch (cachehints)
  {
  case CACHE_GET_MTMODE:
    return MT_NICE_FILTER;
  }
  return 0;  // We do not pass cache requests upwards.
}

AVSValue __cdecl ClearProperties::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  return new ClearProperties(args[0].AsClip(), env);
}

//**************************************************
// propCopy

CopyProperties::CopyProperties(PClip _child, PClip _child2, bool _merge, AVSValue _propNames, bool _exclude, IScriptEnvironment* env)
  : GenericVideoFilter(_child), child2(_child2), merge(_merge), exclude(_exclude)
{ 
  propNames_defined = PropNamesToArray("propCopy", _propNames, propNames, env);
}

CopyProperties::~CopyProperties() { }

AVSValue __cdecl CopyProperties::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  const bool merge = args[2].AsBool(false);
  const bool exclude = args[4].AsBool(false);
  return new CopyProperties(args[0].AsClip(), args[1].AsClip(), merge, args[3], exclude, env);
}

int __stdcall CopyProperties::SetCacheHints(int cachehints, int frame_range)
{
  AVS_UNUSED(frame_range);
  switch (cachehints)
  {
  case CACHE_GET_MTMODE:
    return MT_NICE_FILTER;
  }
  return 0;  // We do not pass cache requests upwards.
}

static void CopyOneFrameProp(const char* key, AVSMap* mapv, const AVSMap* avsmap_from, IScriptEnvironment* env)
{
  const int numElements = env->propNumElements(avsmap_from, key);
  char propType = env->propGetType(avsmap_from, key);
  int error;

  // int and float (int64 and double) supports array get set
  if (propType == AVSPropTypes::PROPTYPE_INT) {
    auto srcval = env->propGetIntArray(avsmap_from, key, &error);
    env->propSetIntArray(mapv, key, srcval, numElements);
  }
  else if (propType == AVSPropTypes::PROPTYPE_FLOAT) {
    auto srcval = env->propGetFloatArray(avsmap_from, key, &error);
    env->propSetFloatArray(mapv, key, srcval, numElements);
  }
  else {
    // assumes the property is cleared beforehand

    // frame, clip and data (string)
    if (propType == AVSPropTypes::PROPTYPE_FRAME) {
      for (int index = 0; index < numElements; index++) {
        auto src = env->propGetFrame(avsmap_from, key, index, &error);
        env->propSetFrame(mapv, key, src, AVSPropAppendMode::PROPAPPENDMODE_APPEND);
      }
    }
    else if (propType == AVSPropTypes::PROPTYPE_CLIP) {
      for (int index = 0; index < numElements; index++) {
        auto src = env->propGetClip(avsmap_from, key, index, &error);
        env->propSetClip(mapv, key, src, AVSPropAppendMode::PROPAPPENDMODE_APPEND);
      }
    }
    else if (propType == AVSPropTypes::PROPTYPE_DATA) {
      for (int index = 0; index < numElements; index++) {
        // string, byte array in general
        auto src = env->propGetData(avsmap_from, key, index, &error);
        auto size = env->propGetDataSize(avsmap_from, key, index, &error);
        env->propSetData(mapv, key, src, size, AVSPropAppendMode::PROPAPPENDMODE_APPEND);
      }
    }
  }
}

PVideoFrame __stdcall CopyProperties::GetFrame(int n, IScriptEnvironment* env)
{
  PVideoFrame frame = child->GetFrame(n, env);
  PVideoFrame frame2 = child2->GetFrame(n, env);

  const AVSMap* avsmap_from = env->getFramePropsRO(frame2);
  const int propNum = env->propNumKeys(avsmap_from);

  if (0 == propNum) {
    // source has no properties at all
    if (!merge) {
      // exact copy: make target empty as well.
      // Erase only if needed
      const AVSMap* avsmap_to_test = env->getFramePropsRO(frame);
      if (0 != env->propNumKeys(avsmap_to_test)) {
        env->MakePropertyWritable(&frame);
        AVSMap* avsmap_to = env->getFramePropsRW(frame);
        env->clearMap(avsmap_to);
      }
    }
    // source has no properties
    return frame;
  }

  env->MakePropertyWritable(&frame);

  if (!merge) {
    // exact copy
    if (!propNames_defined)
      env->copyFrameProps(frame2, frame);
    else {
      // copy only selected / not excluded names
      AVSMap* mapv = env->getFramePropsRW(frame);
      env->clearMap(mapv); // clear all
      if (!exclude) {
        // positive list
        for (auto& s : propNames) {
          const char* key = s.c_str();
          CopyOneFrameProp(key, mapv, avsmap_from, env);
        }
      }
      else {
        // negative list
        const int numKeys = env->propNumKeys(avsmap_from);
        for (int i = 0; i < numKeys; i++) {
          const char* key = env->propGetKey(avsmap_from, i);
          if (std::find(propNames.begin(), propNames.end(), key) == propNames.end()) {
            // not in negative list -> copy
            CopyOneFrameProp(key, mapv, avsmap_from, env);
          }
        }
      }
    }
    return frame;
  }

  // merge
  AVSMap* mapv = env->getFramePropsRW(frame);

  // add from source, replace if needed
  const int numKeys = env->propNumKeys(avsmap_from);
  for (int i = 0; i < numKeys; i++) {
    const char* key = env->propGetKey(avsmap_from, i);
    // merge if no list specified or name is in the list
    bool need_copy;
    if (propNames_defined) {
      need_copy = std::find(propNames.begin(), propNames.end(), key) != propNames.end();
      if (exclude)
        need_copy = !need_copy;
    }
    else
      need_copy = true;

    if(need_copy) {
      env->propDeleteKey(mapv, key); // delete if existed
      CopyOneFrameProp(key, mapv, avsmap_from, env);
    }
  }

  return frame;
}

//**************************************************
// propShow

ShowProperties::ShowProperties(PClip _child, int size, bool showtype, IScriptEnvironment* env)
  : GenericVideoFilter(_child), size(size), showtype(showtype)
{ }

ShowProperties::~ShowProperties() { }

PVideoFrame __stdcall ShowProperties::GetFrame(int n, IScriptEnvironment* env)
{
  PVideoFrame frame = child->GetFrame(n, env);

  const AVSMap* avsmap = env->getFramePropsRO(frame);

  const int propNum = env->propNumKeys(avsmap);

  if (0 == propNum)
    return frame;

  std::stringstream ss;
  ss << "Number of keys = " << std::to_string(propNum) << std::endl;

  for (int index = 0; index < propNum; index++) {
    const char* propName = env->propGetKey(avsmap, index);
    std::string propName_s = propName;

    const char propType = env->propGetType(avsmap, propName);
    ss << propName;
    if (showtype)
      ss << " (" << propType << ")";
    ss << " = ";

    const int propNumElements = env->propNumElements(avsmap, propName);

    int error;

    if (propType == 'u') {
      // unSet: undefined value
      ss << "<unset!>";
    }
    else if (propType == 'i') {
      if (propNumElements > 1)
        ss << "[";
      const int64_t* arr = env->propGetIntArray(avsmap, propName, &error);
      for (int i = 0; i < propNumElements; ++i) {
        ss << std::to_string(arr[i]);
        if (i < propNumElements - 1)
          ss << ", ";
      }
      if (propNumElements > 1)
        ss << "]";

      if (propName_s == "_ColorRange") {
        ss << " = ";
        switch (arr[0]) {
        case ColorRange_e::AVS_RANGE_LIMITED: ss << "limited"; break;
        case ColorRange_e::AVS_RANGE_FULL: ss << "full"; break;
        }
      }
      else if (propName_s == "_Matrix") {
        ss << " = ";
        switch (arr[0]) {
          case Matrix_e::AVS_MATRIX_RGB: ss << "rgb"; break;
          case Matrix_e::AVS_MATRIX_BT709: ss << "709"; break;
          case Matrix_e::AVS_MATRIX_UNSPECIFIED: ss << "unspec"; break;
          case Matrix_e::AVS_MATRIX_ST170_M: ss << "170m"; break;
          case Matrix_e::AVS_MATRIX_ST240_M: ss << "240m"; break;
          case Matrix_e::AVS_MATRIX_BT470_BG: ss << "470bg (601)"; break;
          case Matrix_e::AVS_MATRIX_BT470_M: ss << "470m (fcc)"; break;
          case Matrix_e::AVS_MATRIX_YCGCO: ss << "YCgCo"; break;
          case Matrix_e::AVS_MATRIX_BT2020_NCL: ss << "2020ncl (2020)"; break;
          case Matrix_e::AVS_MATRIX_BT2020_CL: ss << "2020cl"; break;
          case Matrix_e::AVS_MATRIX_CHROMATICITY_DERIVED_CL: ss << "chromacl"; break;
          case Matrix_e::AVS_MATRIX_CHROMATICITY_DERIVED_NCL: ss << "chromancl"; break;
          case Matrix_e::AVS_MATRIX_ICTCP: ss << "ictcp"; break;
          case Matrix_e::AVS_MATRIX_AVERAGE: ss << "AVERAGE-Legacy Avisynth"; break;
        }
      }
      else if (propName_s == "_ChromaLocation") {
        ss << " = ";
        switch (arr[0]) {
        case ChromaLocation_e::AVS_CHROMA_LEFT: ss << "left (mpeg2)"; break;
        case ChromaLocation_e::AVS_CHROMA_CENTER: ss << "center (mpeg1, jpeg)"; break;
        case ChromaLocation_e::AVS_CHROMA_TOP_LEFT: ss << "top_left"; break;
        case ChromaLocation_e::AVS_CHROMA_TOP: ss << "top"; break;
        case ChromaLocation_e::AVS_CHROMA_BOTTOM_LEFT: ss << "bottom_left"; break;
        case ChromaLocation_e::AVS_CHROMA_BOTTOM: ss << "bottom"; break;
        case ChromaLocation_e::AVS_CHROMA_DV: ss << "dv-Legacy Avisynth"; break;
        }
      }
    }
    else if (propType == 'f') {
      if (propNumElements > 1)
        ss << "[";
      const double* arr = env->propGetFloatArray(avsmap, propName, &error);
      for (int i = 0; i < propNumElements; ++i) {
        ss << std::to_string(arr[i]);
        if (i < propNumElements - 1)
          ss << ", ";
      }
      if (propNumElements > 1)
        ss << "]";
    }
    else if (propType == 's') {
      if (propNumElements > 1)
        ss << "[";
      for (int i = 0; i < propNumElements; ++i) {
        const char* s = env->propGetData(avsmap, propName, i, &error);
        ss << "\"" << s << "\"";
        if (i < propNumElements - 1)
          ss << ", ";
      }
      if (propNumElements > 1)
        ss << "]";
    }
    else {
      ss << "<cannot display!>";
    }
    ss << std::endl;
  }

  std::string t = ss.str();
  const char* text = t.c_str();
  AVSValue args[3] = { child, text, size };
  const char * argnames[3] = { 0, 0, "size" };
  PClip child2 = env->Invoke("Text", AVSValue(args, 3), argnames).AsClip();
  frame = child2->GetFrame(n, env);

  return frame;
}

int __stdcall ShowProperties::SetCacheHints(int cachehints, int frame_range)
{
  AVS_UNUSED(frame_range);
  switch (cachehints)
  {
  case CACHE_GET_MTMODE:
    return MT_NICE_FILTER;
  }
  return 0;  // We do not pass cache requests upwards.
}

AVSValue __cdecl ShowProperties::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  const int size = args[1].AsInt(16);
  const bool showtype = args[2].AsBool(false);
  return new ShowProperties(args[0].AsClip(), size, showtype, env);
}
