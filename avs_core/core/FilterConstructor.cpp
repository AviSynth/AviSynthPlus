#include "FilterConstructor.h"
#include "avisynth.h"

FilterConstructor::FilterConstructor(IScriptEnvironment2 * env, IScriptEnvironment_Avs25* env25, const Function *func, std::vector<AVSValue>* argStorage, std::vector<AVSValue>* ctorArgs) :
  Env(env),
  Env25(env25),
  Func(func),
#if 1
  ArgStorage(std::move(*argStorage)),
  CtorArgs(std::move(*ctorArgs))
#else
  // no need to move, subarrays could not be returned
  ArgStorage(*argStorage),
  CtorArgs(*ctorArgs)
#endif
{
}


AVSValue FilterConstructor::InstantiateFilter() const
{
  // funcArgs is passed to the function as an array.
  // Calls the plugin's instance creator, which is usually Filter_Create(AVSValue args, void *user_data, IScriptEnvironment *env)
  AVSValue funcArgs(CtorArgs.data(), (int)CtorArgs.size());
  AVSValue retval = Func->apply(funcArgs, Func->user_data, 
    Func->isAvs25 ? (IScriptEnvironment *)Env25 : Env);
  // pass back proper ScriptEnvironment, because a 2.5 plugin e.g. GRunT can back-Invoke ScriptClip
  if (Func->isAvs25)
  {
    // After instantiate a v2.5 "baked code" filter
    // manually release Clips or else clip variables get stuck in memory on exit
    if (funcArgs.IsArray())
    {
      for (int i = 0; i < funcArgs.ArraySize(); i++)
      {
        if (funcArgs[i].IsClip()) {
          IClip* rawClip = (IClip*)(void*)funcArgs[i].AsClip();
          rawClip->Release();
        }
      }
    }
    // Problem: AVS 2.5 plugins have "baked code", their interface contains direct manipulation, refcount, free of AVSValue resources.
    // So they do not know about array deep copy and deep free mechanism.
    // Thus array sub-elements of the parameter AVSValue won't be freed up on exiting of such functions.
    // Example: CtorArgs contains clip(s) with reference count N
    // Upon creating funcArgs the NEW_AVSVALUE "smart" array creation will copy the subarray as well.
    // Because of the deep copy the reference count of these PClip values are increased by 1.
    // Then funcArgs will be passed to the CPP 2.5 function which won't release the Array content, only destroys the
    // 'holder' array-typed AVSValue, array elements will be untouched.
    // So if there is a PClip parameters the Clip's reference count remains N+1, causing memory leak on 
    // This is why we have to release them manually, deferencing the Clip by one, instead of the plugin's code.
    // If this reference count decrease is not done then env->DeleteScriptEnvironment will not free up everything.
    // ScriptEnvironment destroy is freeing up variables as well.
    // If a variable contains such a clip (e.g. "last"), it will fail to be released, because such Clip will not reached refcount==0.
    // e.g. VirtualDub consecutively press "script reload"s. Memory will leak even 50-80Mbytes so after a while
    // memory gets full.
  }
  return retval;
}
