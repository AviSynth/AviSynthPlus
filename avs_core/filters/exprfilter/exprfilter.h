#ifndef __Exprfilter_h
#define __Exprfilter_h

#include <avisynth.h>
#include <mutex>
#ifdef AVS_POSIX
#include <sys/mman.h>
#endif

#define MAX_EXPR_INPUTS 26
#define INTERNAL_VARIABLES 6
#define MAX_FRAMEPROP_VARIABLES 64
#define MAX_USER_VARIABLES 128

// indexing RWPTR array (pointer sized elements)
#define RWPTR_START_OF_OUTPUT 0   // 1
#define RWPTR_START_OF_XCOUNTER 1 // 1
#define RWPTR_START_OF_INPUTS 2   // count = 26
#define RWPTR_START_OF_PADDING 4  // padding to have 32 pointers (rfu for 8 ptr/cycle ymm simd)
#define RWPTR_START_OF_STRIDES 32 // count = 26 for relative_y
#define RWPTR_START_OF_INTERNAL_VARIABLES 58 // count = 6

// special frame-by-frame variables (incl. frame properties) occupy only float size
#define INTERNAL_VAR_CURRENT_FRAME 0
#define INTERNAL_VAR_RELTIME 1
#define INTERNAL_VAR_RFU2 2
#define INTERNAL_VAR_RFU3 3
#define INTERNAL_VAR_RFU4 4
#define INTERNAL_VAR_RFU5 5
#define INTERNAL_VAR_FRAMEPROP_VARIABLES_START 6
#define RWPTR_START_OF_INTERNAL_FRAMEPROP_VARIABLES (RWPTR_START_OF_INTERNAL_VARIABLES + INTERNAL_VAR_FRAMEPROP_VARIABLES_START) // count = 256

// pad to 32 bytes boundary in x86: 64 * sizeof(pointer) is 32 byte aligned
#define RWPTR_START_OF_USERVARIABLES (RWPTR_START_OF_INTERNAL_FRAMEPROP_VARIABLES + MAX_FRAMEPROP_VARIABLES) // count = max.256 (for 2*ymm sized variables)
#define RWPTR_SIZE (RWPTR_START_OF_USERVARIABLES + MAX_USER_VARIABLES * (2*32 / sizeof(void *)))

struct split1 {
  enum empties_t { empties_ok, no_empties };
};

template <typename Container>
Container& split(
  Container& result,
  const typename Container::value_type& s,
  const typename Container::value_type& delimiters,
  split1::empties_t empties = split1::empties_ok)
{
  result.clear();
  size_t current;
  size_t next = (size_t)-1;
  do {
    if (empties == split1::no_empties) {
      next = s.find_first_not_of(delimiters, next + 1);
      if (next == Container::value_type::npos) break;
      next -= 1;
    }
    current = next + 1;
    next = s.find_first_of(delimiters, current);
    result.push_back(s.substr(current, next - current));
  } while (next != Container::value_type::npos);
  return result;
}

typedef enum {
  opLoadSrc8, opLoadSrc16, opLoadSrcF32, opLoadSrcF16,
  opLoadRelSrc8, opLoadRelSrc16, opLoadRelSrcF32,
  opLoadConst,
  opLoadSpatialX, opLoadSpatialY,
  opLoadInternalVar,
  opStore8, opStore10, opStore12, opStore14, opStore16, opStoreF32, opStoreF16, // avs+: 10,12,14 bit store
  opDup, opSwap,
  opAdd, opSub, opMul, opDiv, opMax, opMin, opSqrt, opAbs, opSgn,
  opFmod,
  opGt, opLt, opEq, opNotEq, opLE, opGE, opTernary,
  opAnd, opOr, opXor, opNeg, opNegSign,
  opExp, opLog, opPow,
  opSin, opCos, opTan, opAsin, opAcos, opAtan, opAtan2,
  opClip, opRound, opFloor, opCeil, opTrunc,
  opStoreVar, opLoadVar, opLoadFramePropVar, opStoreVarAndDrop1
} SOperation;

union ExprUnion {
  float fval;
  int32_t ival;
  uint32_t uval;
  constexpr ExprUnion() : uval{} {}
  constexpr ExprUnion(int32_t _i) : ival(_i) {}
  constexpr ExprUnion(uint32_t _u) : uval(_u) {}
  constexpr ExprUnion(float _f) : fval(_f) {}
} ;

struct ExprOp {
  ExprUnion e;
  uint32_t op;
  int dx, dy;
  ExprOp(SOperation op, float val) : op(op), dx(0), dy(0) {
    e.fval = val;
  }
  ExprOp(SOperation op, int32_t val = 0) : op(op), dx(0), dy(0) {
    e.ival = val;
  }
  ExprOp(SOperation op, int32_t val, int dx, int dy) : op(op), dx(dx), dy(dy) {
    e.ival = val;
  }
};

struct ExprFramePropData {
  int srcIndex;
  std::string name;
  int var_index;
  float value;
};

enum PlaneOp {
  poProcess, poCopy, poUndefined, poFill
};

struct ExprData {
#ifdef __VAPOURSYNTH__
  VSNodeRef *node[MAX_EXPR_INPUTS];
  VSVideoInfo vi;
#else
  PClip clips[MAX_EXPR_INPUTS];
  VideoInfo vi;
#endif
  bool clipsUsed[MAX_EXPR_INPUTS]; // not doing GetFrame unreferenced input clips
  std::vector<ExprOp> ops[4]; // 4th: alpha
  std::vector<ExprFramePropData> frameprops[4];
  int plane[4];
  float planeFillValue[4]; // optimize: fill plane with const
  int planeCopySourceClip[4]; // optimize: copy plane from which clip
  bool planeOptAvx2[4]; // instruction set constraints
  bool planeOptSSE2[4];

  int lutmode; // 0: no, 1:1D (lutx), 2:2D (lutxy)
  uint8_t* luts[4]; // different lut tables, reusable by multiple planes
  // int planeLutIndex[4]; // which luts is used by the plane. todo: when luts are the same for different planes
  
  size_t maxStackSize;
  int numInputs;
#ifdef VS_TARGET_CPU_X86
  typedef void(*ProcessLineProc)(void *rwptrs, intptr_t ptroff[RWPTR_SIZE], intptr_t niter, uint32_t spatialY);
  ProcessLineProc proc[4]; // 4th: alpha
  ExprData() : clips(), vi(), proc() {}
#else
  ExprData() : clips(), vi() {}
#endif
  ~ExprData() {
#ifdef VS_TARGET_CPU_X86
    for (int i = 0; i < 4; i++) // 4th: alpha
#ifdef VS_TARGET_OS_WINDOWS
      VirtualFree((LPVOID)proc[i], 0, MEM_RELEASE);
#else
      munmap((void *)proc[i], 0);
#endif
#endif
  }
};

class Exprfilter : public IClip
/**
  *
**/
{
private:
  std::vector<PClip> children;
  std::vector<std::string> expressions;
  VideoInfo vi;
  ExprData d;
  const bool optAvx2; // disable avx2 path
  const bool optSingleMode; // generate asm code using only one XMM/YMM register set instead of two
  const bool optSSE2; // disable simd path

  // scale_inputs related settings
  const std::string scale_inputs;
  bool autoconv_full_scale;
  bool autoconv_conv_int;
  bool autoconv_conv_float;
  const int clamp_float_i;
  int lutmode;

  // special internal flag since v2.2.20: set when scale_inputs is "floatUV"
  // like in masktools 2.2.20+, preshifts chroma -0.5..+0.5 to 0..1.0 range and then shifts the result back.
  bool shift_float;

  void preReadFrameProps(int plane, std::vector<PVideoFrame>& src, IScriptEnvironment* env);
  void calculate_lut(IScriptEnvironment *env);
public:
  Exprfilter(const std::vector<PClip>& _child_array, const std::vector<std::string>& _expr_array, const char *_newformat, const bool _optAvx2,
    const bool _optSingleMode2, const bool _optSSE2, const std::string _scale_inputs, const int _clamp_float, const int _lutmode, IScriptEnvironment *env);
  void processFrame(int plane, int w, int h, int pixels_per_iter, float framecount, float relative_time, int numInputs,
    uint8_t*& dstp, int dst_stride,
    std::vector<const uint8_t*>& srcp, std::vector<int>& src_stride, std::vector<intptr_t>& ptroffsets, std::vector<const uint8_t*>& srcp_orig);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment *env);
  ~Exprfilter();
  static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env);

  inline void __stdcall GetAudio(void* buf, int64_t start, int64_t count, IScriptEnvironment* env) {
    children[0]->GetAudio(buf, start, count, env);
  }

  inline const VideoInfo& __stdcall GetVideoInfo() {
    return vi;
  }

  inline bool __stdcall GetParity(int n) {
    return children[0]->GetParity(n);
  }

  int __stdcall SetCacheHints(int cachehints, int frame_range) {
    AVS_UNUSED(frame_range);
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

};

#endif //__Exprfilter_h
