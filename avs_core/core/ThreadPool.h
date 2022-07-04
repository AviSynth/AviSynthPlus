#ifndef _AVS_THREADPOOL_H
#define _AVS_THREADPOOL_H

#include <avisynth.h>
#include <future>
#include <vector>

typedef std::future<AVSValue> AVSFuture;
typedef std::promise<AVSValue> AVSPromise;

class InternalEnvironment;

class JobCompletion : public IJobCompletion
{
private:
  const size_t max_jobs;
  size_t nJobs;

public:
  typedef std::pair<AVSPromise, AVSFuture> PromFutPair;
  PromFutPair *pairs;

  JobCompletion(size_t _max_jobs) :
    max_jobs(_max_jobs),
    nJobs(0),
    pairs(NULL)
  {
    pairs = new PromFutPair[max_jobs];

    // Initialize for first use
    nJobs = max_jobs;
    Reset();
  }

  AVSPromise* Add()
  {
    if (nJobs == max_jobs)
      throw AvisynthError("The completion object is already full.");

    AVSPromise* ret = &(pairs[nJobs].first);
    ++nJobs;
    return ret;
  }

  virtual ~JobCompletion()
  {
    Wait();
    delete [] pairs;
  }

  void __stdcall Wait()
  {
    for (size_t i = 0; i < nJobs; ++i)
      pairs[i].second.wait();
  }
  size_t __stdcall Size() const
  {
    return nJobs;
  }
  size_t __stdcall Capacity() const
  {
    return max_jobs;
  }
  AVSValue __stdcall Get(size_t i)
  {
    return pairs[i].second.get();
  }
  void __stdcall Reset()
  {
    for (size_t i = 0; i < nJobs; ++i)
    {
      pairs[i].first = std::move(AVSPromise());
      pairs[i].second = std::move(pairs[i].first.get_future());
    }
    nJobs = 0;
  }
  void __stdcall Destroy()
  {
    delete this;
  }
};

class ThreadPoolPimpl;
class ThreadPool
{
private:
  ThreadPoolPimpl* const _pimpl;

  static void ThreadFunc(size_t thread_id, ThreadPoolPimpl* const _pimpl, InternalEnvironment* env);
public:
  ThreadPool(size_t nThreads, size_t nStartId, InternalEnvironment* env);
  ~ThreadPool();

  void QueueJob(ThreadWorkerFuncPtr clb, void* params, InternalEnvironment* env, JobCompletion* tc);
  size_t NumThreads() const;

  std::vector<void*> Finish();
  void Join();
};


#endif  // _AVS_THREADPOOL_H
