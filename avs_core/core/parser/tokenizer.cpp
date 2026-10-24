// Avisynth v2.5.  Copyright 2002 Ben Rudiak-Gould et al.
// http://avisynth.nl

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
#include "../strings.h"
#ifdef AVS_WINDOWS
#include <avs/win.h>
#else
#include <avs/posix.h>
#endif

#include <cfloat>
#include <climits>
#include <cstdlib>
#include <string>



/****************************
 *******   Tokenizer   ******
 ***************************/

// In an e"..." string a quotation mark is escaped only when an odd number of backslashes
// precede it: in e"\\" the backslash is escaped, the quotation mark closes the string.
static bool IsEscapedQuotation(const char* string_start, const char* quotation)
{
  int backslashes = 0;
  for (const char* p = quotation - 1; p >= string_start && *p == '\\'; --p)
    ++backslashes;
  return (backslashes & 1) != 0;
}

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
      if (isAsciiDigit(pc[1])) {
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

    case '$':
      // hexadecimal constant $FFFFFFFF remains -1 int;
      // or suffixed with L for 64 bit data: 0xFFFFFFFFL becomes 0x00000000FFFFFFFF
    {
      bool long64bit = false;
      type = 'l';
      longlong = 0;
      ++pc;
      do {
        if (*pc >= '0' && *pc <= '9')
          longlong = longlong * 16 + (*pc - '0');
        else if (*pc >= 'a' && *pc <= 'f')
          longlong = longlong * 16 + (*pc - 'a' + 10);
        else if (*pc >= 'A' && *pc <= 'F')
          longlong = longlong * 16 + (*pc - 'A' + 10);
        else if (*pc == 'L' || *pc == 'l') {
          long64bit = true;
          ++pc;
          break; // nothing allowed after L cast
        }
        else
          env->ThrowError("$ must be followed by a hexadecimal number");
      } while (isAsciiAlnum(*++pc));
      if (!long64bit) {
        // Default is integer.
        // Thus we can avoid the color (like $FFFFFFFF) constants
        // to be casted to 64 bit.
        type = 'i';
        integer = (int)longlong;
      }
    }
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
            escaped_quotation = escape && IsEscapedQuotation(start, end);
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
            escaped_quotation = escape && IsEscapedQuotation(start, end);
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
      else if (isAsciiDigit(*pc)) {
        // number
        GetNumber();
      } else if (isAlphaUnderscore(*pc)) {
        // identifier
        do {
          pc++;
        } while (isAlphaNumUnderscore(*pc));
        type = 'I';
        identifier = env->SaveString(token_start, int(pc - token_start));
        if (!lstrcmpi(identifier, "__END__")) {
          type = 0;
        }
      } else if ((unsigned char)*pc >= 0x80) {
        // e.g. a part of an UTF-8 sequence: printing a single byte of it would be garbage
        env->ThrowError("unexpected non-ASCII character (byte 0x%02X), allowed only in strings and comments", (unsigned char)*pc);
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
  // old: start by assuming an int and switch to float if necessary
  // v11: default precision is 64 bit
  //      start by assuming long and switch to double if necessary
  constexpr int64_t INT64_MAX_DIV_10 = INT64_MAX / 10;
  constexpr int64_t INT64_MAX_MOD_10 = INT64_MAX % 10;
  const char* const start = pc;
  type = 'l';
  longlong = 0;
  bool overflow = false;
  while (isAsciiDigit(*pc)) {
    // Check for potential overflow before performing the operation, but go on anyway
    if (overflow || longlong > INT64_MAX_DIV_10 || (longlong == INT64_MAX_DIV_10 && (*pc - '0') > INT64_MAX_MOD_10))
      overflow = true;
    else
      longlong = longlong * 10 + (*pc - '0');
    ++pc;
  }

  // Floating point: a decimal point, an exponent (since 3.7.6: e or E, optional sign, at least one
  // digit), or an integer too big for 64 bits. scan_decimal reads the whole number again, correctly
  // rounded and independent of the locale decimal point, which the host application may have changed.
  double d;
  const char* const end = scan_decimal(start, &d);
  if (end != pc || overflow) {
    type = 'f';
    pc = end;
    if (d > DBL_MAX) // inf
      env->ThrowError("Tokenizer: Number is to big.");
    double_pt = d; // v11: type name remained 'f' but content is double
  }
  else if (longlong <= INT_MAX) {
    // store into a more simple type
    type = 'i';
    integer = (int)longlong;
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
    case 'I': return "identifier";
    case 'f': return "floating-point"; // v11: double
    case 'i': return "integer";
    case 'l': return "long";
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

