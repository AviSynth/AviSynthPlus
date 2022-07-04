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


#include <avisynth.h>


class AveragePlane {

public:
  static AVSValue Create(AVSValue args, void* user_data, IScriptEnvironment* env);
  static AVSValue AvgPlane(AVSValue clip, void* user_data, int plane, int offset, IScriptEnvironment* env);
};

class ComparePlane {

public:
  static AVSValue CmpPlane(AVSValue clip, AVSValue clip2, void* user_data, int plane, IScriptEnvironment* env);
  static AVSValue CmpPlaneSame(AVSValue clip, void* user_data, int offset, int plane, IScriptEnvironment* env);

  static AVSValue Create(AVSValue args, void* user_data, IScriptEnvironment* env);
  static AVSValue Create_prev(AVSValue args, void* user_data, IScriptEnvironment* env);
  static AVSValue Create_next(AVSValue args, void* user_data, IScriptEnvironment* env);
};


class MinMaxPlane {

public:
  static AVSValue MinMax(AVSValue clip, void* user_data, double threshold, int offset, int plane, int mode, bool setvar, IScriptEnvironment* env);

  static AVSValue Create_max(AVSValue args, void* user_data, IScriptEnvironment* env);
  static AVSValue Create_min(AVSValue args, void* user_data, IScriptEnvironment* env);
  static AVSValue Create_median(AVSValue args, void* user_data, IScriptEnvironment* env);
  static AVSValue Create_minmax(AVSValue args, void* user_data, IScriptEnvironment* env);
  static AVSValue Create_minmax_stats(AVSValue args, void* user_data, IScriptEnvironment* env);

private:
  enum { MIN = 1, MAX = 2, MEDIAN = 3, MINMAX_DIFFERENCE = 4, STATS = 5 };

};


class GetProperty {
public:
  static AVSValue Create(AVSValue args, void* user_data, IScriptEnvironment* env);
};

class GetPropertyAsArray {
public:
  static AVSValue Create(AVSValue args, void* user_data, IScriptEnvironment* env);
};

class GetAllProperties {
public:
  static AVSValue Create(AVSValue args, void* user_data, IScriptEnvironment* env);
};

class GetPropertyDataSize {
public:
  static AVSValue Create(AVSValue args, void* , IScriptEnvironment* env);
};

class GetPropertyType {
public:
  static AVSValue Create(AVSValue args, void*, IScriptEnvironment* env);
};

class GetPropertyNumElements {
public:
  static AVSValue Create(AVSValue args, void* , IScriptEnvironment* env);
};

class GetPropertyNumKeys {
public:
  static AVSValue Create(AVSValue args, void*, IScriptEnvironment* env);
};

class GetPropertyKeyByIndex {
public:
  static AVSValue Create(AVSValue args, void*, IScriptEnvironment* env);
};

