// Avisynth v2.5.  Copyright 2002 Ben Rudiak-Gould et al.
// http://www.avisynth.org

// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA, or visit
// http://www.gnu.org/copyleft/gpl.html .
//
// Linking Avisynth statically or dynamically with other modules is making a
// combined work based on Avisynth.  Thus, the terms and conditions of the GNU
// General Public License cover the whole combination.
//
// As a special exception, the copyright holders of Avisynth give you
// permission to link Avisynth with independent modules that communicate with
// Avisynth solely through the interfaces defined in avisynth.h, regardless of the license
// terms of these independent modules, and to copy and distribute the
// resulting combined work under terms of your choice, provided that
// every copy of the combined work is accompanied by a complete copy of
// the source code of Avisynth (the version of Avisynth used to produce the
// combined work), being distributed under the terms of the GNU General
// Public License plus this exception.  An independent module is a module
// which is not derived from or based on Avisynth, such as 3rd-party filters,
// import and export plugins, or graphical user interfaces.


#include "tokenizer.h"
#include "../InternalEnvironment.h"
#ifdef AVS_WINDOWS
#include <avs/win.h>
#else
#include <avs/posix.h>
#endif

#include <cfloat>
#include <climits>



/****************************
 *******   Tokenizer   ******
 ***************************/

Tokenizer::Tokenizer(const char* pc, IScriptEnvironment* _env)
  : env(_env)
{
  this->pc = pc;
  this->line = 1;
  this->type = 0;
  NextToken();
}

Tokenizer::Tokenizer(Tokenizer* old)
  : env(old->env)
{
  pc = old->pc;
  line = old->line;
  type = old->type;
  NextToken();
}

bool Tokenizer::IsIdentifier(const char* id) const
{
  return IsIdentifier() && !lstrcmpi(id, identifier);
}


void Tokenizer::NextToken() {

  if (IsNewline())
    line++;

  // skip whitespace, comments, and escaped newlines

  for (;;) {
    SkipWhitespace();
    if (*pc == '\\') {
      // eat backslash followed by newline
      const char* const old_pc = pc;
      pc++;
      SkipWhitespace();
      if (*pc == '\n' || *pc == '\r') {
        SkipNewline();
        continue;
      } else {
        token_start = old_pc;
        env->ThrowError("Script error: `\\' can only appear at the beginning or end of a line");
      }
    } else if (*pc == '\n' || *pc == '\r') {
      // skip newline if it's followed by backslash or `{'     }
      const char* const old_pc = pc;
      const int old_line = line;
      do {
        SkipNewline();
        SkipWhitespace();
      } while (*pc == '\n' || *pc == '\r');
      if (*pc == '\\') {
        pc++;
        continue;
      } else if (*pc == '{') {
        break;
      } else {
        pc = old_pc;
        line = old_line;
        break;
      }
    } else if (*pc == '#') {
      // skip from # to end of line (comment)
      while (*pc != 0 && *pc != '\n' && *pc != '\r')
        pc++;
      break;
    } else if (pc[0] == '/' && pc[1] == '*') {    // Block comment /* */
      const char *end = strstr(pc+2, "*/");
      if (!end)
        env->ThrowError("Parse error: block comment missing closing */");

      for (const char *cp = pc+2; cp < end; cp++) {
        if (*cp == '\n') { line++; }
      }
      pc = end+2;
      continue;
    } else if (pc[0] == '*' && pc[1] == '/') {
      env->ThrowError("Parse error: orphan block comment closing */");

    } else if (pc[0] == '[' && pc[1] == '*') {    // Nestable block comment [* *]
      const char *end = strstr(pc+2, "*]");
      const char *nest = strstr(pc+2, "[*");

      while (nest && nest+1 < end) {
        end = strstr(end+2, "*]");
        nest = strstr(nest+2, "[*");
      }
      if (!end)
        env->ThrowError("Parse error: nestable block comment missing closing *]");

      for (const char *cp = pc+2; cp < end; cp++) {
        if (*cp == '\n') { line++; }
      }
      pc = end+2;
      continue;
    } else if (pc[0] == '*' && pc[1] == ']') {
      env->ThrowError("Parse error: orphan nestable block comment closing *]");

    } else {
      break;
    }
  }

  token_start = pc;

  switch (*pc) {

    case 0:
      type = 0;
      break;

    case '\n': case '\r':
      SkipNewline();
      type = 'n';
      line--;
      break;

    case '.':
      // a '.' followed by a digit is a number, otherwise it's an operator
      if (isdigit(pc[1])) {
        GetNumber();
      } else {
        ++pc;
        SetToOperator('.');
      }
      break;

    case '<':    // these operators have versions followed by '='
      if ((pc[1] == '=') || (pc[1] == '>')) {
        SetToOperator(pc[0] * 256 + pc[1]);
        pc += 2;
      } else {
        SetToOperator(*pc++);
      }
      break;

    case '>':    // these operators have versions followed by '='
    case '!':
    case '=':
      if (pc[1] != '=') {
        SetToOperator(*pc++);
      } else {
        SetToOperator(pc[0] * 256 + pc[1]);
        pc += 2;
      }
      break;

    case '+':    // these operators have single and double (++, &&, ||, ==) versions
    case '&':
    case '|':
      if (pc[1] != pc[0]) {
        SetToOperator(*pc++);
      } else {
        SetToOperator(pc[0] * 256 + pc[1]);
        pc += 2;
      }
      break;

    case '{':    // these operators are always lone characters
    case '}':
    case '(':
    case ')':
    case '[': // variable capture for function definition
    case ']':
    case ',':
    case '?':
    case ':':
    case '-':
    case '*':
    case '/':
    case '%':
      SetToOperator(*pc++);
      break;

    case '$':    // hexadecimal number
      type = 'i';
      integer = 0;
      ++pc;
      do {
        if (*pc >= '0' && *pc <= '9')
          integer = integer*16 + (*pc - '0');
        else if (*pc >= 'a' && *pc <= 'f')
          integer = integer*16 + (*pc - 'a' + 10);
        else if (*pc >= 'A' && *pc <= 'F')
          integer = integer*16 + (*pc - 'A' + 10);
        else
          env->ThrowError("$ must be followed by a hexadecimal number");
      } while (isalnum(*++pc));
      break;

    default:
      if(*pc == '"' || (*pc == 'e' && pc[1] == '"')) { // string
        bool escape = false;
        if (*pc == 'e') {
          escape = true;
          ++pc;
        }
        const char *start, *end;
        if (pc[1] == '"' && pc[2] == '"') {
          // """..."""
          start = pc+3;
          bool escaped_quotation = false;
          auto tmp_start = start;
          do {
            end = strstr(tmp_start, "\"\"\"");
            if (!end)
              env->ThrowError("Parse error: string missing closing quotation marks");
            escaped_quotation = escape && (*(end - 1) == '\\');
            if (escaped_quotation)
              tmp_start = end + 1;
          } while (escaped_quotation);

          while (end[3] == '"')
            end++;
          pc = end+3;
        } else {
          // "..."
          start = pc+1;
          bool escaped_quotation = false;
          auto tmp_start = start;
          do {
            end = strchr(tmp_start, '"');
            if (!end)
              env->ThrowError("Parse error: string missing closing quotation mark");
            escaped_quotation = escape && (*(end - 1) == '\\');
            if (escaped_quotation)
              tmp_start = end + 1;
          } while (escaped_quotation);

          /* I like the ability to have newlines in strings, thanks */
          // const char *cr = strchr(start, '\r'), *lf = strchr(start, '\n');
          // if ((cr && cr < end) || (lf && lf < end))
          //   env->ThrowError("Parse error: newline found in string");

          pc = end+1;
        }
        for (const char *cp = start; cp < end; cp++) {
          if (*cp == '\n') { line++; }
        }        type = 's';
        string = static_cast<InternalEnvironment*>(env)->SaveString(start, int(end-start), escape);
      }
      else if (isdigit(*pc)) {
        // number
        GetNumber();
      } else if (*pc == '_' || isalpha(*pc)) {
        // identifier
        do {
          pc++;
        } while (*pc == '_' || isalnum(*pc));
        type = 'd';
        identifier = env->SaveString(token_start, int(pc - token_start));
        if (!lstrcmpi(identifier, "__END__")) {
          type = 0;
        }
      } else {
        env->ThrowError("unexpected character \"%c\"", *pc);
      }
      break;
  }
}


int Tokenizer::GetColumn(const char* start_of_string) const
{
    const char* x = pc;
    while (x > start_of_string && x[-1] != '\n' && x[-1] != '\r')
      x--;
    return int(pc-x);
}


void Tokenizer::SkipWhitespace()
{
  while (*pc == ' ' || *pc == '\t')
    pc++;
}

void Tokenizer::SkipNewline()
{
  if (*pc == '\n' || *pc == '\r')
  {
    pc++;
    line++;
    if ((*pc == '\n' || *pc == '\r') && *pc != *(pc-1))
      pc++;
  }
}

void Tokenizer::AssertType(char expected_type) const
{
  if (type != expected_type)
    ThrowTypeMismatch(expected_type, type, env);
}

void Tokenizer::GetNumber()
{
  // start by assuming an int and switch to float if necessary
  type = 'i';
  integer = 0;
  double dtemp = 0;
  double place = 1;

  do {
    if (*pc == '.') {
      type = 'f';
      ++pc;
      while (isdigit(*pc)) {
        place *= 10;
        dtemp = dtemp*10 + (*pc - '0');
        ++pc;
      }
      break;
    } else {
      integer = integer*10 + (*pc - '0');
	  dtemp   = dtemp*10   + (*pc - '0');
    }
    ++pc;
  } while (isdigit(*pc) || *pc == '.');

  dtemp /= place;
  if (dtemp > FLT_MAX)
	env->ThrowError("Tokenizer: Number is to big.");

  if (type == 'f') {
	floating_pt = (float)dtemp;
  }
  else if (dtemp > INT_MAX) {
	type = 'f';
	floating_pt = (float)dtemp;
  }
}


void Tokenizer::SetToOperator(int o)
{
  type = 'o';
  op = o;
}



/**** Helper Functions ****/

static const char* GetTypeName(char type)
{
  switch (type)
  {
    case 0:   return "undefined";
    case 'a': return "array";
    case 'b': return "boolean";
    case 'c': return "clip";
    case 'd': return "identifier";
    case 'f': return "floating-point";
    case 'i': return "integer";
    case 'o': return "operator";
    case 's': return "string";
    default: return "unknown";
  }
}


void ThrowTypeMismatch(char expected, char actual, IScriptEnvironment* env)
{
  env->ThrowError("Tokenizer: expected type '%s' doesn't match actual type '%s' (this is a bug)",
      GetTypeName(expected), GetTypeName(actual));
}

