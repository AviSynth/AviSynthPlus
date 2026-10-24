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


#ifndef AVSCORE_VARTABLE_H
#define AVSCORE_VARTABLE_H

#include "strings.h"
#include "avs/alignment.h"
#include "avs/minmax.h"
#include <avisynth.h>
#include <unordered_map>
#include <mutex>
#include <iostream>
#include <cstring>

struct iequal_to_ascii
{
  bool operator()(const char* str1, const char* str2) const
  {
    return streqi(str1, str2);
  }
};

struct ihash_ascii
{
  std::size_t operator()(const char* s) const
  {
	  // NOTE the connection between the hash() and equals() functions!
	  // In order for the hash table to work correctly, if two strings compare
	  // equal, they MUST have the same hash.

    size_t hash = 0;
    while (*s)
      hash = hash * 101  +  asciiToLower(*s++); // case folding must match iequal_to_ascii (streqi): ASCII only

    return hash;
  }
};

// Custom hash function for composite keys (const char* pointer + size_t length)
struct CompositeKeyDjb2Hash {
  size_t operator()(const std::pair<const char*, size_t>& key) const {
    const char* str = key.first;
    size_t len = key.second;
    size_t hash = 5381;

    while (len--) {
      hash = ((hash << 5) + hash) + *str;
      ++str;
    }
    return hash;
  }
};

// Custom equality function for composite keys
struct CompositeKeyEqual {
  bool operator()(const std::pair<const char*, size_t>& lhs, const std::pair<const char*, size_t>& rhs) const {
    // memcmp, not strncmp: a string saved with a length may contain embedded zeros
    return (lhs.second == rhs.second) && (std::memcmp(lhs.first, rhs.first, lhs.second) == 0);
  }
};

// This doles out storage space for strings.  No space is ever freed
// until the class instance is destroyed (which happens when a script
// file is closed).
class StringDump {
   enum { BLOCK_SIZE = 32768 };
   char* current_block;
   size_t block_pos, block_size;

   // string cache
   std::unordered_map<std::pair<const char*, size_t>, const char*, CompositeKeyDjb2Hash, CompositeKeyEqual> cache;

   // Get the pointer to the string (if it exists)
   const char* get_string(const char* keyStr, size_t keyLen) {
     auto it = cache.find({ keyStr, keyLen });
     if (it != cache.end()) {
       return it->second; // Return existing pointer
     }
     else {
       return nullptr;
     }
   }

   void add_string(const char* keyStr, size_t keyLen) {
     cache[{keyStr, keyLen}] = keyStr;
   }

   void ensure_length(size_t len)
   {
     if (block_pos + len + 1 > block_size) {
       char* new_block = new char[block_size = max(block_size, len + 1 + sizeof(char*))];
       //_RPT0(0, "StringDump: Allocating new stringblock.\r\n");
       *(char**)new_block = current_block;   // beginning of block holds pointer to previous block
       current_block = new_block;
       block_pos = sizeof(char*);
     }
   }

public:
   StringDump() : current_block(0), block_pos(BLOCK_SIZE), block_size(BLOCK_SIZE) {}

   ~StringDump() {
      _RPT0(0, "StringDump: DeAllocating all stringblocks.\r\n");
      char* p = current_block;
      while (p) {
         char* next = *(char**)p;
         delete[] p;
         p = next;
      }
   }

   // SaveString with len==-1 can store more than 'int' sized data
   // up to size_t theoretically even on 32 bits.
   // Public interface defines 'len' as int, this can we live with.
   char* SaveString(const char* s, int _len = -1, bool escape = false) {
     size_t srclen = (_len == -1) ? strlen(s) : _len >= 0 ? static_cast<size_t>(_len) : 0;
     size_t len;

     std::string ss;
     if (escape) {
       ss.reserve(srclen); // worst case. Note: input string may not be closed by 0x00
       // PF: no need for WideString conversion, utf8 lower 128 ascii characters are freely searchable w/o conversion
       len = 0;
       for (size_t i = 0; i < srclen && s[i]; ++i, ++len) {
         if (s[i] == '\\' && i + 1 < srclen) {
           switch (s[i + 1]) {
           case 'n': ss += '\n'; ++i; continue;
           case 'r': ss += '\r'; ++i; continue;
           case 't': ss += '\t'; ++i; continue;
           case '0': ss += '\0'; ++i; continue;
           case 'a': ss += '\a'; ++i; continue;
           case 'f': ss += '\f'; ++i; continue;
           case '\\': ss += '\\'; ++i; continue;
           case '\"': ss += '\"'; ++i; continue;
           case '\'': ss += '\''; ++i; continue;
           case 'b': ss += '\b'; ++i; continue;
           case 'v': ss += '\v'; ++i; continue;
           }
         }
         ss += s[i];
       }
       len = ss.size();
       s = ss.c_str();
     }
     else {
       len = srclen;
     }

    // check in cache content
    const char* test_ptr = get_string(s, len);
    if(test_ptr != nullptr) // same string found in cache
      return const_cast<char *>(test_ptr);

    ensure_length(len);
    char* result = current_block + block_pos;
    memcpy(result, s, len);
    result[len] = 0;
    block_pos += AlignNumber(len + 1, sizeof(char*)); // Keep word-aligned

    add_string(result, len); // update cache

    return result;
   }

   void Clear() {
     cache.clear(); // its keys point into the freed or reused blocks
     if (current_block) {
         // deallocate string blocks except the first one
         while (char* p = *(char**)current_block) {
            delete[] current_block;
            current_block = p;
         }
         block_pos = sizeof(char*);
         block_size = BLOCK_SIZE;
      }
   }
};

class VarFrame
{
   typedef std::unordered_map<const char*, AVSValue, ihash_ascii, iequal_to_ascii> ValueMap;
   ValueMap variables;

public:
   VarFrame() {
      variables.max_load_factor(0.8f);
   }

   // This method will not modify the *val argument if it returns false.
   bool Get(const char* name, AVSValue *val) const
   {
      ValueMap::const_iterator v = variables.find(name);
      if (v != variables.end())
      {
         *val = v->second;
         return true;
      }
      return false;
   }

   bool Set(const char* name, const AVSValue& val)
   {
      std::pair<ValueMap::iterator, bool> ret = variables.insert(ValueMap::value_type(name, val));
      ret.first->second = val;
      return ret.second;
   }

   void Clear()
   {
      variables.clear();
   }
};

class VarStringFrame : public VarFrame
{
   StringDump string_dump;
public:
   char* SaveString(const char* s, int len = -1, bool escape = false) {
      return string_dump.SaveString(s, len, escape);
   }

   void Clear()
   {
      //string_dump.Clear(); // do not destroy string (it is destroyed at destructor)
      VarFrame::Clear();
   }
};

class ConcurrentVarStringFrame : protected VarStringFrame
{
   // avoid write/read concurrency of global variables in runtime scripts in MT
   mutable std::mutex var_mutex;

public:
   // This method will not modify the *val argument if it returns false.
   bool Get(const char* name, AVSValue *val) const
   {
      std::lock_guard<std::mutex> lock(var_mutex); // avoid concurrency for global variables
      return VarFrame::Get(name, val);
   }

   bool Set(const char* name, const AVSValue& val)
   {
      std::lock_guard<std::mutex> lock(var_mutex); // avoid concurrency for global variables
      return VarFrame::Set(name, val);
   }

   char* SaveString(const char* s, int len = -1, bool escape = false) {
      std::lock_guard<std::mutex> lock(var_mutex); // avoid concurrency for global variables
      return VarStringFrame::SaveString(s, len, escape);
   }

   void Clear()
   {
      std::lock_guard<std::mutex> lock(var_mutex); // avoid concurrency for global variables
      VarStringFrame::Clear();
   }
};

class VarTable
{
private:
   ConcurrentVarStringFrame* topFrame;

   std::vector<std::unique_ptr<VarFrame>> stackFrames;
   std::vector<std::unique_ptr<VarStringFrame>> globalFrames;

   std::vector<std::unique_ptr<VarFrame>> stackPool;
   std::vector<std::unique_ptr<VarStringFrame>> globalPool;

public:
   VarTable(ConcurrentVarStringFrame* topFrame) : topFrame(topFrame)
   {
      Push();
   }

   void Clear()
   {
      stackFrames.clear();
      globalFrames.clear();
      stackPool.clear();
      globalPool.clear();
   }

   void Push()
   {
      if (stackPool.size() > 0) {
         stackFrames.emplace_back(std::move(stackPool.back()));
         stackPool.pop_back();
      }
      else {
         stackFrames.emplace_back(new VarFrame());
      }
   }

   void Pop()
   {
      assert(stackFrames.size() > 0);
      stackFrames.back()->Clear();
      stackPool.emplace_back(std::move(stackFrames.back()));
      stackFrames.pop_back();
   }

   void PushGlobal()
   {
      Push();
      if (globalPool.size() > 0) {
         globalFrames.emplace_back(std::move(globalPool.back()));
         globalPool.pop_back();
      }
      else {
         globalFrames.emplace_back(new VarStringFrame());
      }
   }

   void PopGlobal()
   {
      Pop();
      assert(globalFrames.size() > 0);
      globalFrames.back()->Clear();
      globalPool.emplace_back(std::move(globalFrames.back()));
      globalFrames.pop_back();
   }

   bool Set(const char* name, const AVSValue& val)
   {
      return stackFrames.back()->Set(name, val);
   }

   bool SetGlobal(const char* name, const AVSValue& val)
   {
      if (globalFrames.size() > 0) {
         return globalFrames.back()->Set(name, val);
      }
      return topFrame->Set(name, val);
   }

   bool Get(const char* name, AVSValue *val) const
   {
      if (stackFrames.size() > 0 && stackFrames.back()->Get(name, val)) {
         return true;
      }
      for (auto it = globalFrames.rbegin(); it != globalFrames.rend(); ++it) {
         if ((**it).Get(name, val)) {
            return true;
         }
      }
      return topFrame->Get(name, val);
   }

   char* SaveString(const char* s, int len = -1, bool escape = false)
   {
      if (globalFrames.size() > 0) {
         return globalFrames.back()->SaveString(s, len, escape);
      }
      return topFrame->SaveString(s, len, escape);
   }
};

#endif // AVSCORE_VARTABLE_H
