// AviSynth+.  Copyright 2026- AviSynth+ Project
// https://avs-plus.net
//
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

#ifndef __Scratch_Buffer_H__
#define __Scratch_Buffer_H__

// Heap scratch buffer for per-call temporaries (row buffers) in code that is also compiled
// in the ISA-specific translation units (*_sse41.cpp, *_avx2.cpp, *_avx512*.cpp ...).
//
// Why not std::vector there: its members (resize, the reallocation and length-error/bad_alloc paths)
// are inline templates with external linkage. A TU built with -mavx2 or /arch:AVX2 may compile them
// with AVX2 (VEX) instructions, and the linker keeps a single COMDAT copy for all TUs of the DLL:
// possibly the AVX2 one for a generic caller, running on a CPU without AVX2.
// This class is in an anonymous namespace: every TU has its own copy, built with its own flags,
// nothing is shared between TUs. Allocation goes through the out-of-line operator new[]/delete[].
//
// Minimal on purpose: resize() (zero-filled, like std::vector, old content not kept), data(),
// operator[], size(). Trivial element types only. Not copyable.
// Use it only inside functions with internal linkage (static or in an anonymous namespace), never
// as a member or parameter type of anything shared between TUs.

#include <cstddef>
#include <type_traits>

namespace {

template<typename T>
class ScratchBuffer {
  static_assert(std::is_trivial<T>::value, "ScratchBuffer: trivial element types only");
  T* p_ = nullptr;
  size_t n_ = 0;

public:
  ScratchBuffer() noexcept = default;
  explicit ScratchBuffer(size_t count) { resize(count); }
  ~ScratchBuffer() { delete[] p_; }
  ScratchBuffer(const ScratchBuffer&) = delete;
  ScratchBuffer& operator=(const ScratchBuffer&) = delete;

  // Allocates count zero-initialized elements. Old content is not kept; same size is a no-op.
  void resize(size_t count) {
    if (count == n_) return;
    delete[] p_;
    p_ = nullptr;
    n_ = 0;
    if (count) {
      p_ = new T[count]();
      n_ = count;
    }
  }

  T* data() noexcept { return p_; }
  const T* data() const noexcept { return p_; }
  T& operator[](size_t i) noexcept { return p_[i]; }
  const T& operator[](size_t i) const noexcept { return p_[i]; }
  size_t size() const noexcept { return n_; }
};

} // namespace

#endif // __Scratch_Buffer_H__
