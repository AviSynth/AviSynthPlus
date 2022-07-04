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


#include "scriptparser.h"
#include "../InternalEnvironment.h"


/********************************
 *******   Script Parser   ******
 *******************************/


ScriptParser::ScriptParser(IScriptEnvironment* _env, const char* _code, const char* _filename)
   : env(static_cast<IScriptEnvironment2*>(_env)), tokenizer(_code, _env), code(_code), filename(_filename), loopDepth(0) {}

PExpression ScriptParser::Parse(void)
{
  try {
    return new ExpRootBlock(ParseBlock(false, NULL));
  }
  catch (const AvisynthError &ae) {
    env->ThrowError("%s\n(%s, line %d, column %d)", ae.msg, filename, tokenizer.GetLine(), tokenizer.GetColumn(code));
  }
#ifndef _DEBUG
  catch (...) {
    env->ThrowError("Parse: Unrecognized exception!");
  }
#endif
  return 0; // To make VC++ happy.  Why isn't the __declspec(noreturn) on ThrowError good enough?
}


void ScriptParser::Expect(int op, const char* msg=0)
{
  if (tokenizer.IsOperator(op))
    tokenizer.NextToken();
  else {
    if (msg)
      env->ThrowError(msg);
    else {
      if (op < 256)
        env->ThrowError("Script error: expected `%c'", op);
      else
        env->ThrowError("Script error: expected `%c%c'", (op>>8), (op&255));
    }
  }
}


PExpression ScriptParser::ParseFunctionDefinition(void)
{
  /*
  bool global_spcified = false;
  if (tokenizer.IsIdentifier("global")) {
    tokenizer.NextToken();

  }
  */
  const char* name = nullptr;
  if (tokenizer.IsIdentifier()) {
    name = tokenizer.AsIdentifier();
    tokenizer.NextToken();
  }
  const char* var_names[max_args];
  int var_count = 0;
  char param_types[4096];
  int param_chars=0;
  bool param_floats[max_args];
  const char* param_names[max_args];
  int param_count=0;

  // variable capture
  if (tokenizer.IsOperator('[')) {
    if (name != nullptr) {
      env->ThrowError("Script error: variable capture is not supported on legacy function definition.");
    }
    tokenizer.NextToken();
    bool need_comma = false;
    for (;;) {
      if (tokenizer.IsOperator(']')) {
        tokenizer.NextToken();
        break;
      }
      if (need_comma) {
        Expect(',', "Script error: expected a , or ]");
      }

      if (tokenizer.IsIdentifier()) {
        var_names[var_count++] = tokenizer.AsIdentifier();
      }
      else {
        env->ThrowError("Script error: expected a parameter name");
      }

      tokenizer.NextToken();
      need_comma = true;
    }
  }

  if (!tokenizer.IsOperator('{')) {
    Expect('(', "Script error: expected ( or { after function name");
    bool need_comma = false;
    bool named_arg_found = false;
    for (;;) {
      if (tokenizer.IsOperator(')')) {
         tokenizer.NextToken();
        break;
      }
      if (need_comma) {
        Expect(',', "Script error: expected a , or )");
      }
      if (param_count == max_args) {
        env->ThrowError("Script error: parameter list too long");
      }

      param_floats[param_count] = false;
      char type = '.';
      char array_kind = ' '; // can be * (zero_or_more) or + (one_or_more)
      Tokenizer lookahead(&tokenizer);
      if (lookahead.IsIdentifier() || lookahead.IsString()) {
        // we have a variable type preceding its name
        if (tokenizer.IsIdentifier("val")) type = '.';
        else if (tokenizer.IsIdentifier("bool")) type = 'b';
        else if (tokenizer.IsIdentifier("int")) type = 'i';
        else if (tokenizer.IsIdentifier("float")) {
          param_floats[param_count] = true;
          type = 'f';
        }
        else if (tokenizer.IsIdentifier("string")) type = 's';
        else if (tokenizer.IsIdentifier("clip")) type = 'c';
        else if (tokenizer.IsIdentifier("func")) type = 'n';
        // AVS+ 161028 array type in user defined functions
        else if (tokenizer.IsIdentifier("array") || tokenizer.IsIdentifier("val_array")) {
          array_kind = '*';  type = '.';
          // or: isArray = false;  type = 'a'; ? No. Keeping the old syntax
          // but .+ must be the very last parameter if parameter is unnamed
        }
        else if (tokenizer.IsIdentifier("array_nz") || tokenizer.IsIdentifier("val_array_nz")) {
          array_kind = '+';  type = '.'; // _nz: non-zero size
        }
        else if (tokenizer.IsIdentifier("bool_array")) {
          array_kind = '*';  type = 'b';
        }
        else if (tokenizer.IsIdentifier("bool_array_nz")) {
          array_kind = '+';  type = 'b';
        }
        else if (tokenizer.IsIdentifier("int_array")) {
          array_kind = '*';  type = 'i';
        }
        else if (tokenizer.IsIdentifier("int_array_nz")) {
          array_kind = '+';  type = 'i';
        }
        else if (tokenizer.IsIdentifier("float_array")) {
          array_kind = '*';  type = 'f';
        }
        else if (tokenizer.IsIdentifier("float_array_nz")) {
          array_kind = '+';  type = 'f';
        }
        else if (tokenizer.IsIdentifier("string_array")) {
          array_kind = '*';  type = 's';
        }
        else if (tokenizer.IsIdentifier("string_array_nz")) {
          array_kind = '+';  type = 's';
        }
        else if (tokenizer.IsIdentifier("clip_array")) {
          array_kind = '*';  type = 'c';
        }
        else if (tokenizer.IsIdentifier("clip_array_nz")) {
          array_kind = '+';  type = 'c';
        }
        else if (tokenizer.IsIdentifier("func_array")) {
          array_kind = '*';  type = 'n';
        }
        else if (tokenizer.IsIdentifier("func_array_nz")) {
          array_kind = '+';  type = 'n';
        }
        else env->ThrowError("Script error: expected \"val\", \"bool\", \"int\", \"float\", \"string\", \"array\", or \"clip\" (or their \"_array\" or \"_array_nz\" versions");
        tokenizer.NextToken();
      }

      if (tokenizer.IsIdentifier()) {
        if (named_arg_found)
          env->ThrowError("Script error: can't have a named (quoted) parameter followed by an ordinary parameter");
        param_names[param_count++] = tokenizer.AsIdentifier();
      } else if (tokenizer.IsString()) {
        named_arg_found = true;
        const char* param_name = param_names[param_count++] = tokenizer.AsString();
        int len = lstrlen(param_name);
        if (param_chars + lstrlen(param_name) >= 4000)
          env->ThrowError("Script error: parameter list too long");
        param_types[param_chars] = '[';
        memcpy(&param_types[param_chars+1], param_name, len);
        param_types[param_chars+len+1] = ']';
        param_chars += len+2;
      } else {
        env->ThrowError("Script error: expected a parameter name");
      }
      param_types[param_chars++] = type;
      if(array_kind == '*' || array_kind == '+')
        param_types[param_chars++] = array_kind; // zero or more / one or more
      tokenizer.NextToken();

      need_comma = true;
    }
  }

  int line = tokenizer.GetLine();

  param_types[param_chars] = 0;
  PExpression body = new ExpRootBlock(ParseBlock(true, NULL));

  const char* saved_param_signature = env->SaveString(param_types);

  if(name != nullptr) {
    // legacy function definition
    ScriptFunction* sf = new ScriptFunction(body, param_floats, param_names, param_count);
    env->AtExit(ScriptFunction::Delete, sf);
    env->AddFunction(name, saved_param_signature, ScriptFunction::Execute, sf, "$UserFunctions$");
    return new ExpLegacyFunctionDefinition();
  }

  if (name) {
    auto envi = static_cast<InternalEnvironment*>(env);
    envi->UpdateFunctionExports(name, saved_param_signature, "$UserFunctions$");
  }

  return new ExpFunctionDefinition(body, name, saved_param_signature,
    param_floats, param_names, param_count, var_names, var_count,
    filename, line);
  // was before 20200324:
  // bool is_global = global_spcified || (var_count == 0);
  // return new ExpFunctionDefinition(name, sf, is_global);
}


PExpression ScriptParser::ParseBlock(bool braced, bool *empty)
{
  if (braced) {
    // allow newlines (and hence comments) before '{' -- Gavino 7 Dec 2009
    while (tokenizer.IsNewline())
      tokenizer.NextToken();

    Expect('{');
  }

  // the purpose of this array and the accompanying code is to produce
  // a nice balanced binary tree of ExpSequence objects, so that the
  // maximum call depth in Evaluate grows logarithmically instead of
  // linearly.
  // For every i, either trees[i]==0 or it's a balanced tree of (1<<i) elts.
  PExpression trees[20];

  bool ignore_remainder = false;
  for (;;) {
    if (tokenizer.IsNewline()) {
      tokenizer.NextToken();
    } else if (tokenizer.IsOperator('}')) {
      if (braced) {
        tokenizer.NextToken();
        break;
      } else {
        env->ThrowError("Script error: found } without a matching {");
      }
    } else if (tokenizer.IsEOF()) {
      if (braced) {
        env->ThrowError("Script error: end of file reached without matching }");
      } else {
        break;
      }
    } else {
      bool stop;
      PExpression exp = ParseStatement(&stop);
      if (!braced && exp) {
        auto ep = dynamic_cast<ExpLegacyFunctionDefinition*>((Expression*)(void*)exp);
        if (ep != nullptr) {
          // Inhibit legacy function definition to get into expression tree.
          // Or else "return last" would be needed before them.
          // We check actual type runtime with dynamic cast.
          exp = nullptr;
        }
      }
      if (exp && !ignore_remainder) {
        if (filename)
          exp = new ExpLine(exp, filename, tokenizer.GetLine());
        for (int i=0; i<20; ++i) {
          if (trees[i]) {
            exp = new ExpSequence(trees[i], exp);
            trees[i] = 0;
          } else {
            trees[i] = exp;
            break;
          }
        }
      }
      ignore_remainder |= stop;
    }
  }

  PExpression result = trees[0];
  for (int i=1; i<20; ++i) {
    if (trees[i])
      result = result ? PExpression(new ExpSequence(trees[i], result)) : trees[i];
  }

  if (result)
  {
    if (empty) *empty = false;
    return result;
  }
  else
  {
    if (empty) *empty = true;
    return PExpression(new ExpConstant(AVSValue()));
  }
}



PExpression ScriptParser::ParseStatement(bool* stop)
{
  *stop = false;
  // null statement
  if (tokenizer.IsNewline() || tokenizer.IsEOF()) {
    return 0;
  }
  /*
  // original AVS2.6/old Avs+: function definition returns 0
  // Function declaration moved more inner
  // Since Neo/3.6.0 we have function objects besides script functions.
  // Thus return 0 is conditional and handled outside.
  else if (tokenizer.IsIdentifier("function")) {
    tokenizer.NextToken();
    ParseFunctionDefinition();
    return 0;
  }
  */
  // exception handling
  else if (tokenizer.IsIdentifier("try")) {
    tokenizer.NextToken();
    PExpression try_block = ParseBlock(true, NULL);
    while (tokenizer.IsNewline())
      tokenizer.NextToken();
    if (!tokenizer.IsIdentifier("catch"))
      env->ThrowError("Script error: expected `catch'");
    tokenizer.NextToken();
    Expect('(');
    if (!tokenizer.IsIdentifier())
      env->ThrowError("Script error: expected identifier");
    const char* id = tokenizer.AsIdentifier();
    tokenizer.NextToken();
    Expect(')');
    return new ExpTryCatch(try_block, id, ParseBlock(true, NULL));
  }
  // 'if', 'while', 'for':
  else if (tokenizer.IsIdentifier("if")) {
    return ParseIf();
  }
  else if (tokenizer.IsIdentifier("while")) {
    return ParseWhile();
  }
  else if (tokenizer.IsIdentifier("for")) {
    return ParseFor();
  }
  // return statement
  else if (tokenizer.IsIdentifier("return")) {
    *stop = true;
    tokenizer.NextToken();
    return new ExpReturn(ParseAssignmentWithRet());
  }
  // break statement
  else if (tokenizer.IsIdentifier("break")) {
    if (loopDepth <= 0)
      throw AvisynthError("'Break' statement outside of loop.");
    tokenizer.NextToken();
    return new ExpBreak();
  }
  else {
    return ParseAssignment();
  }
}

PExpression ScriptParser::ParseIf(void)
{
  bool blockEmpty;

  PExpression If, Then, Else = 0;
  tokenizer.NextToken();
  Expect('(');
  If = ParseAssignmentWithRet();
  Expect(')');

  Then = ParseBlock(true, &blockEmpty);
  if (blockEmpty)
    Then = NULL;

  while (tokenizer.IsNewline())
    tokenizer.NextToken();
  if (tokenizer.IsIdentifier("else")) {
    tokenizer.NextToken();

    if (tokenizer.IsIdentifier("if"))
    {
      Else = ParseIf();
    }
    else
    {
      Else = ParseBlock(true, &blockEmpty);
      if (blockEmpty)
        Else = NULL;
    }
  }
  return new ExpBlockConditional(If, Then, Else);
}

PExpression ScriptParser::ParseWhile(void)
{
  tokenizer.NextToken();
  Expect('(');
  const PExpression cond = ParseAssignmentWithRet();
  Expect(')');

  ++loopDepth;
  bool blockEmpty;
  PExpression body = ParseBlock(true, &blockEmpty);
  if (blockEmpty)
    body = NULL;
  --loopDepth;

  return new ExpWhileLoop(cond, body);
}

PExpression ScriptParser::ParseFor(void)
{
  tokenizer.NextToken();
  Expect('(');
  if (!tokenizer.IsIdentifier())
    env->ThrowError("Script error: expected a variable name");
  const char* id = tokenizer.AsIdentifier();
  tokenizer.NextToken();
  Expect('=');
  const PExpression init = ParseAssignmentWithRet();
  Expect(',');
  const PExpression limit = ParseAssignmentWithRet();
  PExpression step = NULL;
  if (tokenizer.IsOperator(',')) {
    tokenizer.NextToken();
    step = ParseAssignmentWithRet();
  } else {
    step = PExpression(new ExpConstant(AVSValue(1)));
  }

  Expect(')');

  ++loopDepth;
  bool blockEmpty;
  PExpression body = ParseBlock(true, &blockEmpty);
  if (blockEmpty)
    body = NULL;
  --loopDepth;

  return new ExpForLoop(id, init, limit, step, body);
}

PExpression ScriptParser::ParseAssignment(void)
{
  if (tokenizer.IsIdentifier("global")) {
    tokenizer.NextToken();
    if (!tokenizer.IsIdentifier())
      env->ThrowError("Script error: `global' must be followed by a variable name");
    const char* name = tokenizer.AsIdentifier();
    tokenizer.NextToken();
    Expect('=');
    PExpression exp = ParseConditional();
	return new ExpGlobalAssignment(name, exp);
  }
  PExpression exp = ParseAssignmentWithRet();
  if (tokenizer.IsOperator('=')) {
    const char* name = exp->GetLvalue();
    if (!name)
      env->ThrowError("Script error: left operand of `=' must be a variable name");
    tokenizer.NextToken();
	exp = ParseAssignmentWithRet();
    return new ExpAssignment(name, exp);
  }

  return exp;
}

PExpression ScriptParser::ParseAssignmentWithRet(void)
{
	PExpression exp = ParseConditional();
	if (tokenizer.IsOperator(":="_i)) {
		const char* name = exp->GetLvalue();
		if (!name)
			env->ThrowError("Script error: left operand of `:=' must be a variable name");
		tokenizer.NextToken();
		exp = ParseAssignmentWithRet();
		return new ExpAssignment(name, exp, true);
	}
	return exp;
}

PExpression ScriptParser::ParseConditional(void)
{
  PExpression a = ParseOr();
  if (tokenizer.IsOperator('?')) {
    tokenizer.NextToken();
    PExpression b = ParseAssignmentWithRet();
    Expect(':');
    PExpression c = ParseAssignmentWithRet();
    return new ExpConditional(a, b, c);
  }
  return a;
}

PExpression ScriptParser::ParseOr(void)
{
  PExpression left = ParseAnd();
  if (tokenizer.IsOperator("||"_i)) {
    tokenizer.NextToken();
    PExpression right = ParseOr();
    return new ExpOr(left, right);
  }
  return left;
}

PExpression ScriptParser::ParseAnd(void)
{
  PExpression left = ParseComparison();
  if (tokenizer.IsOperator("&&"_i)) {
    tokenizer.NextToken();
    PExpression right = ParseAnd();
    return new ExpAnd(left, right);
  }
  return left;
}


PExpression ScriptParser::ParseComparison(void)
{
  PExpression left = ParseAddition(false);
  PExpression result;
  int op;
  while ((op = GetTokenAsComparisonOperator()) != 0) {
    tokenizer.NextToken();
    PExpression right = ParseAddition(false);
    PExpression term;
    switch (op) {
      case "=="_i: term = new ExpEqual(left, right); break;
      case "!="_i: term = new ExpNot(new ExpEqual(left, right)); break;
      case "<>"_i: term = new ExpNot(new ExpEqual(left, right)); break;
      case '<': term = new ExpLess(left, right); break;
      case ">="_i: term = new ExpNot(new ExpLess(left, right)); break;
      case '>': term = new ExpLess(right, left); break;
      case "<="_i: term = new ExpNot(new ExpLess(right, left)); break;
    }
    result = !result ? term : PExpression(new ExpAnd(result, term));
    left = right;
  }
  return result ? result : left;
}



PExpression ScriptParser::ParseAddition(bool negationOnHold) //update exterior calls to ParseAddition(false)
{
  PExpression left = ParseMultiplication(negationOnHold);
  bool plus = tokenizer.IsOperator('+');
  bool minus = tokenizer.IsOperator('-');
  bool doubleplus = tokenizer.IsOperator("++"_i);
  if (plus || minus || doubleplus) {
    tokenizer.NextToken();
    PExpression right = ParseAddition(minus);
    if (doubleplus) {
      return new ExpDoublePlus(left, right);
    }
    return new ExpPlus(left, right);   //no longer ExpMinus  'right' will be negated when needed
  }
  return left;
}

PExpression ScriptParser::ParseMultiplication(bool negationOnHold)
{
  PExpression left = ParseUnary();

  for (;;) {
    bool mult = tokenizer.IsOperator('*');
    bool div = tokenizer.IsOperator('/');
    bool mod = tokenizer.IsOperator('%');

    if (mult || div || mod)
      tokenizer.NextToken();
    else break;                                 //exits the while if not a mult op

    PExpression right = ParseUnary();
    if (mult)
      left = new ExpMult(left, right);
    else if (div)
      left = new ExpDiv(left, right);
    else
      left = new ExpMod(left, right);
  }

  if (negationOnHold)   //negate the factorised result if needed
    left = new ExpNegate(left);
  return left;
}


PExpression ScriptParser::ParseUnary(void) {
  // accept '+' with anything
  while (tokenizer.IsOperator('+'))
    tokenizer.NextToken();

  if (tokenizer.IsOperator('-')) {
    tokenizer.NextToken();
    return new ExpNegate(ParseUnary());
  }
  else if (tokenizer.IsOperator('!')) {
    tokenizer.NextToken();
    return new ExpNot(ParseUnary());
  }
  else {
    return ParseOOP();
  }
}

PExpression ScriptParser::ParseOOP(void)
{
  // check if need convert [] brackets to function calls: Array() and ArrayGet()
  // no context and [: e.g. x = [23,2] --> x = Array(23,2)
  PExpression left;
  if (tokenizer.IsOperator('['))
    left = ParseArray(nullptr);
  else
    left = ParseFunction(nullptr);
  while (tokenizer.IsOperator('.') || tokenizer.IsOperator('[')) {
    // OOP '.' or array indexing
    // somename.function( ...
    // somename[ ... : e.g. x = a[23,2] --> x = a.ArrayGet(23,2)
    // context is 'somename'
    const bool isArraySpecifier = tokenizer.IsOperator('[');
    tokenizer.NextToken();
    if(isArraySpecifier)
      left = ParseArray(left);
    else
      left = ParseFunction(left);
  }
  return left;
}

PExpression ScriptParser::ParseArray(PExpression context)
{
  PExpression left;
  if (context)
    left = new ExpVariableReference("ArrayGet"); // x = a[2,1] -> x = a.ArrayGet(2,1)
  else
    left = new ExpVariableReference("Array"); // x = [23,3] --> x = Array(23,3)
  PExpression result = ParseCall(left, context, true); // true: array syntax
  return result;
}

PExpression ScriptParser::ParseFunction(PExpression context)
{
  PExpression left = ParseAtom();
  if (context || tokenizer.IsOperator('(')) {
    return ParseCall(left, context, false); // false: function syntax
  }
  return left;
}

/*
  -NEW_AVSVALUE means arrays
  - AVSValue deep copy for arrays (arrays in arrays in ...)
    constant script arrays
    array_variable = [[1, 2, 3], [4, 5, 8], "hello"]
    dictionary = [["one", 1], ["two", 2]]
    empty = []
    subarray = array_variable[0]
    val = subarray[2]
    val2 = array_variable[1, 3]
    str = array_variable[2]
    n = ArraySize(array_variable) #3
    n2 = ArraySize(empty) #0
    val3 = dictionary["two"]

    - arrays as filter parameters(named and unnamed) :
    new 'a' type or use '.+' or '.*' and check AVSValue IsArray()
    Concept is incompatible with avs 2.5 plugins due to their 'baked' interface code
*/
PExpression ScriptParser::ParseCall(PExpression left, PExpression context, bool isArraySpecifier)
{
  // though parser can do here arrays in general, isArraySpecifier is always false for non AVS_NEWVALUE
  // array references are using [] brackets which are here
  // converted to Array() and ArrayGet functions depending on the position
  // x = a[23] --> x = a.ArrayGet(23)  has a variable reference (context is not null)
  // x = [2,3] -> x = Array(2,3)       no context

  // function
  PExpression args[max_args];
  const char* arg_names[max_args];
  memset(arg_names, 0, sizeof(arg_names));
  int params_count = 0;
  int i=0;
  if (context)
    args[i++] = context; // first arg is the object before '.' or '['

  if (
    isArraySpecifier ||
    tokenizer.IsOperator('(')) {

    if(!(context != nullptr && isArraySpecifier)) // ParseOOP already had [ and also the next token
      tokenizer.NextToken();

    const char end_bracket = isArraySpecifier ? ']' : ')';
    // arrays are delimited by ], functions by )

    bool need_comma = false;
    for (;;) {
      if(tokenizer.IsOperator(end_bracket))
      {
        tokenizer.NextToken();
        break;
      }
      if (need_comma) {
        if(isArraySpecifier)
          Expect(',', "Script error: expected a , or ]");
        else
          Expect(',', "Script error: expected a , or )");
      }
      // check for named argument syntax (name=val) for functions
      if (!isArraySpecifier && tokenizer.IsIdentifier()) {
        Tokenizer lookahead(&tokenizer);
        if (lookahead.IsOperator('=')) {
          arg_names[i] = tokenizer.AsIdentifier();
          tokenizer.NextToken();
          tokenizer.NextToken();
        }
      }
      if (i == max_args) {
        env->ThrowError("Script error: argument list too long");
      }
      args[i++] = ParseAssignmentWithRet();
      params_count++;
      need_comma = true;
    }
  }

  if (isArraySpecifier && params_count == 0)
  {
    if (context != nullptr)
      env->ThrowError("Script error: array indexing must have at least one index");
    // contextless case is empty array definition: x = []
  }

  const char* name = left->GetLvalue();
  if (name && i == 1 && args[0]->GetLvalue() && lstrcmpi(name, "func") == 0) {
    // special case the parser should deal with
    // "func(LegacyFunctionName)"
    left = new ExpFunctionWrapper(args[0]->GetLvalue());
  }
  else {
    // we have to set this true for array definitions x = [2,3] or else an
    // implicite "last" will be passed on by ExpFunctionCall::Evaluate
    const bool oop_notation_aka_disable_implicite_last = !!context || isArraySpecifier;
    left = new ExpFunctionCall(name, left, args, arg_names, i, oop_notation_aka_disable_implicite_last);
  }

  if (tokenizer.IsOperator('(')) {
    return ParseCall(left, nullptr, false);
  }
  return left;

}

PExpression ScriptParser::ParseAtom(void)
{
  if (tokenizer.IsIdentifier("function")) {
    tokenizer.NextToken();
    return ParseFunctionDefinition();
  }
  else if (tokenizer.IsIdentifier()) {
    const char* name = tokenizer.AsIdentifier();
    tokenizer.NextToken();
    return new ExpVariableReference(name);
  }
  else if (tokenizer.IsInt()) {
    int result = tokenizer.AsInt();
    tokenizer.NextToken();
    return new ExpConstant(result);
  }
  else if (tokenizer.IsFloat()) {
    float result = tokenizer.AsFloat();
    tokenizer.NextToken();
    return new ExpConstant(result);
  }
  else if (tokenizer.IsString()) {
    const char* result = tokenizer.AsString();
    tokenizer.NextToken();
    return new ExpConstant(result);
  }
#ifdef ARRAYS_AT_TOKENIZER_LEVEL
  else if (tokenizer.IsArray()) {
    std::vector<AVSValue>* result = tokenizer.AsArray(); // PF tokenizer returns new array
    tokenizer.NextToken();
    return new ExpConstant(result);
  }
#endif
  else if (tokenizer.IsOperator('(')) {
    tokenizer.NextToken();
    PExpression result = ParseAssignmentWithRet();
    Expect(')');
    return result;
  }
  else if (tokenizer.IsOperator('[')) { // array
    // with NEW_AVSVALUE this case is unreachable
    env->ThrowError("Script error: array is not supported on this version of avisynth");
    return 0;
  }
  else {
    env->ThrowError("Script error: syntax error");
    return 0;
  }
}

int ScriptParser::GetTokenAsComparisonOperator()
{
  if (!tokenizer.IsOperator())
    return 0;
  int op = tokenizer.AsOperator();
  if (op == "=="_i || op == "!="_i || op == "<>"_i || op == '<' || op == '>' || op == "<="_i || op == ">="_i)
    return op;
  else
    return 0;
}

