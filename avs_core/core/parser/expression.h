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

#ifndef __Expression_H__
#define __Expression_H__

#include <avisynth.h>
#include "../function.h"

#include <vector>
#include <atomic>

/********************************************************************
********************************************************************/


struct ReturnExprException
{
	AVSValue value;
};

/**** Base Classes ****/

class Expression {
public:
  Expression() : refcnt(0) {}
  virtual AVSValue Evaluate(IScriptEnvironment* env) = 0;
  virtual const char* GetLvalue() { return 0; }
  virtual ~Expression() {}

private:
  friend class PExpression;
  std::atomic<int> refcnt;
  void AddRef() { ++refcnt; }
  void Release() { if (--refcnt <= 0) delete this; }
};

class PExpression
{
public:
  PExpression() { Init(0); }
  PExpression(Expression* p) { Init(p); }
  PExpression(const PExpression& p) { Init(p.e); }
  PExpression& operator=(Expression* p) { Set(p); return *this; }
  PExpression& operator=(const PExpression& p) { Set(p.e); return *this; }
  int operator!() const { return !e; }
  operator void*() const { return e; }
  Expression* operator->() const { return e; }
  ~PExpression() { Release(); }

private:
  Expression* e;
  void Init(Expression* p) { e=p; if (e) e->AddRef(); }
  void Set(Expression* p) { if (p) p->AddRef(); if (e) e->Release(); e=p; }
  void Release() { if (e) e->Release(); }
};







/**** Object classes ****/

class ExpRootBlock : public Expression
{
public:
  ExpRootBlock(const PExpression& e) : exp(e) {}
  virtual AVSValue Evaluate(IScriptEnvironment* env);

private:
  const PExpression exp;
};

class ExpConstant : public Expression
{
public:
  ExpConstant(std::vector<AVSValue>* v) : val(v->data(), (int)(v->size())) {} // array of AVSValue*
  ExpConstant(AVSValue v) : val(v) {}
  ExpConstant(int i) : val(i) {}
  ExpConstant(float f) : val(f) {}
  ExpConstant(const char* s) : val(s) {}
  virtual AVSValue Evaluate(IScriptEnvironment* env) {
    AVS_UNUSED(env);
    return val; }

private:
  friend class ExpNegative;
  const AVSValue val;
};


class ExpSequence : public Expression
{
public:
  ExpSequence(const PExpression& _a, const PExpression& _b) : a(_a), b(_b) {}
  virtual AVSValue Evaluate(IScriptEnvironment* env);
private:
  const PExpression a, b;
};


class ExpExceptionTranslator : public Expression
{
public:
  ExpExceptionTranslator(const PExpression& _exp) : exp(_exp) {}
  AVSValue Evaluate(IScriptEnvironment* env);

private:
  const PExpression exp;
  void TrapEval(AVSValue&, unsigned &excode, IScriptEnvironment*);
};


class ExpTryCatch : public ExpExceptionTranslator
{
public:
  ExpTryCatch(const PExpression& _try_block, const char* _id, const PExpression& _catch_block)
    : ExpExceptionTranslator(_try_block), id(_id), catch_block(_catch_block) {}
  AVSValue Evaluate(IScriptEnvironment* env);

private:
  const char* const id;
  const PExpression catch_block;
};

class ExpLine : public ExpExceptionTranslator
{
public:
  ExpLine(const PExpression& _exp, const char* _filename, int _line)
    : ExpExceptionTranslator(_exp), filename(_filename), line(_line) {}
  AVSValue Evaluate(IScriptEnvironment* env);

private:
  const char* const filename;
  const int line;
};


class ExpBlockConditional : public Expression
{
public:
  ExpBlockConditional(const PExpression& _If, const PExpression& _Then, const PExpression& _Else)
   : If(_If), Then(_Then), Else(_Else) {}
  virtual AVSValue Evaluate(IScriptEnvironment* env);

private:
  const PExpression If, Then, Else;
};


class ExpWhileLoop : public Expression
{
public:
  ExpWhileLoop(const PExpression& _condition, const PExpression& _body)
   : condition(_condition), body(_body) {}
  virtual AVSValue Evaluate(IScriptEnvironment* env);

private:
  const PExpression condition, body;
};


class ExpForLoop : public Expression
{
public:
  ExpForLoop(const char* const _id, const PExpression& _init, const PExpression& _limit,
             const PExpression& _step, const PExpression& _body)
   : id(_id), init(_init), limit(_limit), step(_step), body(_body) {}
  virtual AVSValue Evaluate(IScriptEnvironment* env);

private:
  const char* const id;
  const PExpression init, limit, step, body;
};

class ExpBreak : public Expression
{
public:
  ExpBreak() {}
  virtual AVSValue Evaluate(IScriptEnvironment* env);
};

class ExpConditional : public Expression
{
public:
  ExpConditional(const PExpression& _If, const PExpression& _Then, const PExpression& _Else)
   : If(_If), Then(_Then), Else(_Else) {}
  virtual AVSValue Evaluate(IScriptEnvironment* env);

private:
  const PExpression If, Then, Else;
};

class ExpReturn : public Expression
{
public:
	ExpReturn(PExpression value) : value(value) {}
	virtual AVSValue Evaluate(IScriptEnvironment* env);

private:
	const PExpression value;
};



/**** Operator classes ****/

class ExpOr : public Expression
{
public:
  ExpOr(const PExpression& _a, const PExpression& _b) : a(_a), b(_b) {}
  virtual AVSValue Evaluate(IScriptEnvironment* env);

private:
  const PExpression a, b;
};


class ExpAnd : public Expression
{
public:
  ExpAnd(const PExpression& _a, const PExpression& _b) : a(_a), b(_b) {}
  virtual AVSValue Evaluate(IScriptEnvironment* env);

private:
  const PExpression a, b;
};


class ExpEqual : public Expression
{
public:
  ExpEqual(const PExpression& _a, const PExpression& _b) : a(_a), b(_b) {}
  virtual AVSValue Evaluate(IScriptEnvironment* env);

private:
  const PExpression a, b;
};


class ExpLess : public Expression
{
public:
  ExpLess(const PExpression& _a, const PExpression& _b) : a(_a), b(_b) {}
  virtual AVSValue Evaluate(IScriptEnvironment* env);

private:
  const PExpression a, b;
};


class ExpPlus : public Expression
{
public:
  ExpPlus(const PExpression& _a, const PExpression& _b) : a(_a), b(_b) {}
  virtual AVSValue Evaluate(IScriptEnvironment* env);

private:
  const PExpression a, b;
};


class ExpDoublePlus : public Expression
{
public:
  ExpDoublePlus(const PExpression& _a, const PExpression& _b) : a(_a), b(_b) {}
  virtual AVSValue Evaluate(IScriptEnvironment* env);

private:
  const PExpression a, b;
};


class ExpMinus : public Expression
{
public:
  ExpMinus(const PExpression& _a, const PExpression& _b) : a(_a), b(_b) {}
  virtual AVSValue Evaluate(IScriptEnvironment* env);

private:
  const PExpression a, b;
};


class ExpMult : public Expression
{
public:
  ExpMult(const PExpression& _a, const PExpression& _b) : a(_a), b(_b) {}
  virtual AVSValue Evaluate(IScriptEnvironment* env);

private:
  const PExpression a, b;
};


class ExpDiv : public Expression
{
public:
  ExpDiv(const PExpression& _a, const PExpression& _b) : a(_a), b(_b) {}
  virtual AVSValue Evaluate(IScriptEnvironment* env);

private:
  const PExpression a, b;
};


class ExpMod : public Expression
{
public:
  ExpMod(const PExpression& _a, const PExpression& _b) : a(_a), b(_b) {}
  virtual AVSValue Evaluate(IScriptEnvironment* env);

private:
  const PExpression a, b;
};


class ExpNegate : public Expression
{
public:
  ExpNegate(const PExpression& _e) : e(_e) {}
virtual AVSValue Evaluate(IScriptEnvironment* env);

private:
  const PExpression e;
};


class ExpNot : public Expression
{
public:
  ExpNot(const PExpression& _e) : e(_e) {}
  virtual AVSValue Evaluate(IScriptEnvironment* env);

private:
  const PExpression e;
};


class ExpVariableReference : public Expression
{
public:
  ExpVariableReference(const char* _name) : name(_name) {}
  virtual AVSValue Evaluate(IScriptEnvironment* env);

  virtual const char* GetLvalue() { return name; }

private:
  const char* const name;
};


class ExpAssignment : public Expression
{
public:
  ExpAssignment(const char* _lhs, const PExpression& _rhs) : lhs(_lhs), rhs(_rhs), withret(false) {}
  ExpAssignment(const char* _lhs, const PExpression& _rhs, bool wr) : lhs(_lhs), rhs(_rhs), withret(wr) {}

  virtual AVSValue Evaluate(IScriptEnvironment* env);

private:
  const char* const lhs;
  PExpression rhs;
  bool withret;
};


class ExpGlobalAssignment : public Expression
{
public:
  ExpGlobalAssignment(const char* _lhs, const PExpression& _rhs) : lhs(_lhs), rhs(_rhs) {}
  virtual AVSValue Evaluate(IScriptEnvironment* env);

private:
  const char* const lhs;
  PExpression rhs;
};


class ExpFunctionCall : public Expression
{
public:
  ExpFunctionCall( const char* _name, const PExpression& _func, PExpression* _arg_exprs,
                   const char** _arg_expr_names, int _arg_expr_count, bool _oop_notation );
  ~ExpFunctionCall(void);

  virtual AVSValue Evaluate(IScriptEnvironment* env);

private:
  const char* const name;
  PExpression func;
  PExpression* arg_exprs;
  const char** arg_expr_names;
  const int arg_expr_count;
  const bool oop_notation;
};


class ExpLegacyFunctionDefinition : public Expression {
public:
  virtual AVSValue Evaluate(IScriptEnvironment* env) { return AVSValue(); }
};


class ExpFunctionWrapper : public Expression
{
public:
  ExpFunctionWrapper(const char* name);
  virtual AVSValue Evaluate(IScriptEnvironment* env);
private:
  PFunction func;
  const char* const name;
};


class ExpFunctionDefinition : public Expression
{
public:
  ExpFunctionDefinition(const PExpression& _body,
    const char* _name, const char* _param_types,
    const bool* _param_floats, const char** _param_names, int param_count,
    const char** _var_names, int _var_count,
    const char* filename, int line);
  ~ExpFunctionDefinition() {
    delete[]param_floats;
    delete[]param_names;
    delete[]var_names;
  }

  virtual AVSValue Evaluate(IScriptEnvironment* env);

//private:
  const PExpression body;
  const char* name;
  const char* param_types;
  bool *param_floats;
  const char** param_names;
  int var_count;
  const char** var_names;

  const char* filename;
  int line;
};


class FunctionInstance : public IFunction
{
public:
  FunctionInstance(ExpFunctionDefinition* pdef, IScriptEnvironment* env);
  virtual ~FunctionInstance();
  virtual const char* ToString(IScriptEnvironment* env);
  virtual const char* GetLegacyName() { return nullptr; }
  virtual const Function* GetDefinition() { return &data; }
  virtual CaptureVars GetCaptures() {
    CaptureVars ret = { pdef->var_count, pdef->var_names, var_data };
    return ret;
  }
  AVSValue Execute(const AVSValue& args, IScriptEnvironment* env);
  static AVSValue Execute_(AVSValue args, void* user_data, IScriptEnvironment* env);

private:
  Function data;
  ExpFunctionDefinition* pdef;
  PExpression pdef_ref;
  AVSValue *var_data;
};


#endif  // __Expression_H_
