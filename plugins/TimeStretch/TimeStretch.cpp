// Avisynth v2.5.
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

// AviSynth -> SoundTouch interface (c) 2004, Klaus Post.



#include <vector>
#include <avisynth.h>
#include <avs/minmax.h>
#include "SoundTouch/SoundTouch.h"


#define BUFFERSIZE 8192

using namespace soundtouch;


class AVSsoundtouch : public GenericVideoFilter
{
private:
  SoundTouch* sampler;

  int dst_samples_filled;

  SFLOAT* dstbuffer;
  int64_t next_sample;
  int64_t inputReadOffset;
  long double sample_multiplier; // 64bit mantissa!

public:
static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env);
static AVSValue __cdecl CreateRational(AVSValue args, void*, IScriptEnvironment* env);


AVSsoundtouch(PClip _child, float _tempo, float _rate, float _pitch, const AVSValue* args, IScriptEnvironment* env)
: GenericVideoFilter(_child)
{
  dstbuffer = new SFLOAT[BUFFERSIZE * vi.AudioChannels()];

  sample_multiplier  = (long double)_tempo * (long double)_rate;

  sampler = new SoundTouch();

  sampler->setRate(_rate);
  sampler->setTempo(_tempo);
  sampler->setPitch(_pitch);
  sampler->setChannels(vi.AudioChannels());
  sampler->setSampleRate(vi.audio_samples_per_second);
  setSettings(sampler, args, env);

  vi.num_audio_samples = (int64_t)(vi.num_audio_samples / sample_multiplier);

  next_sample = 0;  // Next output sample
  inputReadOffset = 0;  // Next input sample
  dst_samples_filled = 0;

}

static void setSettings(SoundTouch* sampler, const AVSValue* args, IScriptEnvironment* env)
{

  if (args[0].Defined()) sampler->setSetting(SETTING_SEQUENCE_MS,   args[0].AsInt());
  if (args[1].Defined()) sampler->setSetting(SETTING_SEEKWINDOW_MS, args[1].AsInt());
  if (args[2].Defined()) sampler->setSetting(SETTING_OVERLAP_MS,    args[2].AsInt());

  if (args[3].Defined()) sampler->setSetting(SETTING_USE_QUICKSEEK, args[3].AsBool() ? 1 : 0);

  if (args[4].Defined()) {
	int i = args[4].AsInt();
	if (i<0 || i%4 != 0)
	  env->ThrowError("TimeStretch: AntiAliaser filter length must divisible by 4.");

	if (i)
	  sampler->setSetting(SETTING_AA_FILTER_LENGTH, i);
	else
	  sampler->setSetting(SETTING_USE_AA_FILTER,    0);
  }

}

void __stdcall GetAudio(void* buf, int64_t start, int64_t count, IScriptEnvironment* env)
{

  if (start != next_sample) {  // Reset on seek
    sampler->clear();
    next_sample = start;
    inputReadOffset = (int64_t)(sample_multiplier * start);  // Reset at new read position (NOT sample exact :( ).
    dst_samples_filled=0;
  }

  bool buffer_full = false;
  int samples_filled = 0;

  do {
    // Empty buffer if something is still left.
    if (dst_samples_filled) {
      int copysamples = min((int)count-samples_filled, dst_samples_filled);
      // Copy finished samples
      if (copysamples) {
        memcpy((BYTE*)buf+vi.BytesFromAudioSamples(samples_filled), (BYTE*)dstbuffer, (size_t)vi.BytesFromAudioSamples(copysamples));
        samples_filled += copysamples;

        dst_samples_filled -= copysamples;
        // Move non-used samples
        memcpy(dstbuffer, &dstbuffer[copysamples*vi.AudioChannels()], (size_t)vi.BytesFromAudioSamples(dst_samples_filled));
      }
      if (samples_filled >= count)
        buffer_full = true;
    }

    // If buffer empty - refill
    if (dst_samples_filled==0) {
      // Read back samples from filter
      int samples_out = 0;
      int gotsamples = 0;
      do {
        gotsamples = sampler->receiveSamples(&dstbuffer[vi.BytesFromAudioSamples(samples_out)], BUFFERSIZE - samples_out);
        samples_out += gotsamples;
      } while (gotsamples > 0);

      dst_samples_filled = samples_out;

      if (!dst_samples_filled) {  // We didn't get any samples
          // Feed new samples to filter
        child->GetAudio(dstbuffer, inputReadOffset, BUFFERSIZE, env);
        inputReadOffset += BUFFERSIZE;
        sampler->putSamples(dstbuffer, BUFFERSIZE);
      } // End if no samples
    } // end if empty buffer
  } while (!buffer_full);
  next_sample += count;
}

~AVSsoundtouch()
{
    delete[] dstbuffer;
    delete sampler;
}


};

// Percentage float arguments.
AVSValue __cdecl Create_SoundTouch(AVSValue args, void*, IScriptEnvironment* env)
{
	try {	// HIDE DAMN SEH COMPILER BUG!!!
    // 2021?? Probably for VS2005 or so

		PClip clip = args[0].AsClip();

		if (!clip->GetVideoInfo().HasAudio())
			env->ThrowError("Input clip does not have audio.");

		if (!(clip->GetVideoInfo().SampleType()&SAMPLE_FLOAT))
			env->ThrowError("Input audio sample format to TimeStretch must be float.");

		return new AVSsoundtouch(
			args[0].AsClip(),
                        args[1].AsFloat(100.0) / 100.0,
                        args[2].AsFloat(100.0) / 100.0,
                        args[3].AsFloat(100.0) / 100.0,
			&args[4],
			env
		);

	}
  catch (const std::runtime_error &error)
  {
    env->ThrowError("TimeStretch: %s ",error.what());
  }
	catch (...) { throw; }
  return AVSValue(); // n/a
}

// Rational pair arguments
AVSValue __cdecl CreateRational_SoundTouch(AVSValue args, void*, IScriptEnvironment* env)
{
	try {	// HIDE DAMN SEH COMPILER BUG!!!
    // 2021?? Probably for VS2005 or so

		PClip clip = args[0].AsClip();

		if (!clip->GetVideoInfo().HasAudio())
			env->ThrowError("Input clip does not have audio.");

		if (!(clip->GetVideoInfo().SampleType()&SAMPLE_FLOAT))
			env->ThrowError("Input audio sample format to TimeStretch must be float.");

		return new AVSsoundtouch(
                        args[0].AsClip(),
                        (double)args[1].AsInt(1) / args[2].AsInt(1),
                        (double)args[3].AsInt(1) / args[4].AsInt(1),
                        (double)args[5].AsInt(1) / args[6].AsInt(1),
                        &args[7],
			env
		);

	}
  catch (const std::runtime_error &error)
  {
    env->ThrowError("TimeStretch: %s ",error.what());
  }
	catch (...) { throw; }
  return AVSValue(); // n/a
}

const AVS_Linkage * AVS_linkage = 0;
extern "C" __declspec(dllexport) const char* __stdcall AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors) {
	AVS_linkage = vectors;

  // clip, base filename, start, end, image format/extension, info
  env->AddFunction("TimeStretch", "c[tempo]f[rate]f[pitch]f[sequence]i[seekwindow]i[overlap]i[quickseek]b[aa]i", Create_SoundTouch, 0);
  env->AddFunction("TimeStretch", "c[tempo_n]i[tempo_d]i[rate_n]i[rate_d]i[pitch_n]i[pitch_d]i[sequence]i[seekwindow]i[overlap]i[quickseek]b[aa]i", CreateRational_SoundTouch, 0);

  return "`TimeStretch' Changes tempo, pitch, and/or playback rate of audio.";
}
