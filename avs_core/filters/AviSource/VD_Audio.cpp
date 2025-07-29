//	VirtualDub - Video processing and capture application
//	Copyright (C) 1998-2001 Avery Lee
//
//	This program is free software; you can redistribute it and/or modify
//	it under the terms of the GNU General Public License as published by
//	the Free Software Foundation; either version 2 of the License, or
//	(at your option) any later version.
//
//	This program is distributed in the hope that it will be useful,
//	but WITHOUT ANY WARRANTY; without even the implied warranty of
//	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//	GNU General Public License for more details.
//
//	You should have received a copy of the GNU General Public License
//	along with this program; if not, write to the Free Software
//	Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

//#include "VirtualDub.h"

#undef _DEBUG
#include <crtdbg.h>
#define _DEBUG 1
#include <cassert>
#include <limits>
#include <new>

//#include "gui.h"
//#include "crash.h"

#include "Error.h"
#include <avs/minmax.h>
#include "AudioSource.h"

#include "VD_Audio.h"


#pragma warning(disable: 4706)    // assignment within conditional expression


AudioFormatConverter AudioPickConverter(WAVEFORMATEX *src, BOOL to_16bit, BOOL to_stereo);

//////////////// no change converters /////////////////////////////////////

static void convert_audio_nochange8(void *dest, void *src, long count) {
	memcpy(dest, src, count);
}

static void convert_audio_nochange16(void *dest, void *src, long count) {
	memcpy(dest, src, count*2);
}

static void convert_audio_nochange32(void *dest, void *src, long count) {
	memcpy(dest, src, count*4);
}

//////////////// regular converters /////////////////////////////////////

static void convert_audio_mono8_to_mono16(void *dest, void *src, long count) {
	unsigned char *s = (unsigned char *)src;
	signed short *d = (signed short *)dest;

	do {
		*d++ = (signed short)((unsigned long)(*s++-0x80)<<8);
	} while(--count);
}

static void convert_audio_mono8_to_stereo8(void *dest, void *src, long count) {
	unsigned char c,*s = (unsigned char *)src;
	unsigned char *d = (unsigned char *)dest;

	do {
		c = *s++;
		*d++ = c;
		*d++ = c;
	} while(--count);
}

static void convert_audio_mono8_to_stereo16(void *dest, void *src, long count) {
	unsigned char *s = (unsigned char *)src;
	unsigned long c, *d = (unsigned long *)dest;

	do {
		c = ((*s++-0x80)&0xff) << 8;
		*d++ = c | (c<<16);
	} while(--count);
}

static void convert_audio_mono16_to_mono8(void *dest, void *src, long count) {
	signed short *s = (signed short *)src;
	unsigned char *d = (unsigned char *)dest;

	do {
		*d++ = (unsigned char)((((unsigned long)*s++)+0x8000)>>8);
	} while(--count);
}

static void convert_audio_mono16_to_stereo8(void *dest, void *src, long count) {
	signed short *s = (signed short *)src;
	unsigned char c, *d = (unsigned char *)dest;

	do {
		c = (unsigned char)((((unsigned long)*s++)+0x8000)>>8);
		*d++ = c;
		*d++ = c;
	} while(--count);
}

static void convert_audio_mono16_to_stereo16(void *dest, void *src, long count) {
	signed short *s = (signed short *)src;
	unsigned long *d = (unsigned long *)dest, c;

	do {
		c = 0xffff & *s++;
		*d++ = (c | (c<<16));
	} while(--count);
}

static void convert_audio_stereo8_to_mono8(void *dest, void *src, long count) {
	unsigned short *s = (unsigned short *)src;
	unsigned char *d = (unsigned char *)dest;
	unsigned long c;

	do {
		c = *s++;
		*d++ = (unsigned char)(((c&0xff) + (c>>8))/2);
	} while(--count);
}

static void convert_audio_stereo8_to_mono16(void *dest, void *src, long count) {
	unsigned short *s = (unsigned short *)src;
	signed short *d = (signed short *)dest;
	unsigned long c;

	do {
		c = *s++;
		*d++ = (signed short)((((c&0xff) + (c>>8))<<7)-0x8000);
	} while(--count);
}

static void convert_audio_stereo8_to_stereo16(void *dest, void *src, long count) {
	unsigned short c,*s = (unsigned short *)src;
	unsigned long *d = (unsigned long *)dest;

	do {
		c = *s++;
		*d++ = ((unsigned long)((c-0x80)&0xff)<<8) | ((unsigned long)((c&0xff00)-0x8000)<<16);
	} while(--count);
}

static void convert_audio_stereo16_to_mono8(void *dest, void *src, long count) {
	unsigned long c, *s = (unsigned long *)src;
	unsigned char *d = (unsigned char *)dest;

	do {
		c = *s++;
		*d++ = (unsigned char)(((((c&0xffff)+0xffff8000)^0xffff8000) + ((signed long)c>>16) + 0x10000)>>9);
	} while(--count);
}

static void convert_audio_stereo16_to_mono16(void *dest, void *src, long count) {
	unsigned long c, *s = (unsigned long *)src;
	signed short *d = (signed short *)dest;

	do {
		c = *s++;
		*d++ = (signed short)(((((c&0xffff)+0xffff8000)^0xffff8000) + ((signed long)c>>16))/2);
	} while(--count);
}

static void convert_audio_stereo16_to_stereo8(void *dest, void *src, long count) {
	unsigned long c,*s = (unsigned long *)src;
	unsigned char *d = (unsigned char *)dest;

	do {
		c = *s++;
		*d++ = (unsigned char)((((unsigned long)(c & 0xffff))+0x8000)>>8);
		*d++ = (unsigned char)((((unsigned long)(c>>16))+0x8000)>>8);
	} while(--count);
}

static void convert_audio_dual8_to_mono8(void *dest, void *src, long count) {
	const unsigned char *s = (unsigned char *)src;
	unsigned char *d = (unsigned char *)dest;

	do {
		*d++ = *s;
		s+=2;
	} while(--count);
}

static void convert_audio_dual8_to_mono16(void *dest, void *src, long count) {
	unsigned char *s = (unsigned char *)src;
	signed short *d = (signed short *)dest;

	do {
		*d++ = (signed short)((unsigned long)(*s-0x80)<<8);
		s += 2;
	} while(--count);
}

static void convert_audio_dual16_to_mono8(void *dest, void *src, long count) {
	signed short *s = (signed short *)src;
	unsigned char *d = (unsigned char *)dest;

	do {
		*d++ = (unsigned char)((((unsigned long)*s)+0x8000)>>8);
		s += 2;
	} while(--count);
}

static void convert_audio_dual16_to_mono16(void *dest, void *src, long count) {
	const signed short *s = (signed short *)src;
	signed short *d = (signed short *)dest;

	do {
		*d++ = *s;
		s+=2;
	} while(--count);
}

////////////////////////////////////////////

static const AudioFormatConverter acv[]={
	convert_audio_nochange8,
	convert_audio_mono8_to_mono16,
	convert_audio_mono8_to_stereo8,
	convert_audio_mono8_to_stereo16,
	convert_audio_mono16_to_mono8,
	convert_audio_nochange16,
	convert_audio_mono16_to_stereo8,
	convert_audio_mono16_to_stereo16,
	convert_audio_stereo8_to_mono8,
	convert_audio_stereo8_to_mono16,
	convert_audio_nochange16,
	convert_audio_stereo8_to_stereo16,
	convert_audio_stereo16_to_mono8,
	convert_audio_stereo16_to_mono16,
	convert_audio_stereo16_to_stereo8,
	convert_audio_nochange32,
};

static const AudioFormatConverter acv2[]={
	convert_audio_nochange8,
	convert_audio_mono8_to_mono16,
	convert_audio_mono16_to_mono8,
	convert_audio_nochange16,
	convert_audio_dual8_to_mono8,
	convert_audio_dual8_to_mono16,
	convert_audio_dual16_to_mono8,
	convert_audio_dual16_to_mono16,
};

AudioFormatConverter AudioPickConverter(WAVEFORMATEX *src, BOOL to_16bit, BOOL to_stereo) {
	return acv[
			  (src->nChannels>1 ? 8 : 0)
			 +(src->wBitsPerSample>8 ? 4 : 0)
			 +(to_stereo ? 2 : 0)
			 +(to_16bit ? 1 : 0)
		];
}

AudioFormatConverter AudioPickConverterSingleChannel(WAVEFORMATEX *src, bool to_16bit) {
	return acv2[
			  (src->nChannels>1 ? 4 : 0)
			 +(src->wBitsPerSample>8 ? 2 : 0)
			 +(to_16bit ? 1 : 0)
		];
}

///////////////////////////////////

AudioStream::AudioStream() {
	format = NULL;
	format_len = 0;
	samples_read = 0;
	stream_limit = 0x7FFFFFFF;
}

AudioStream::~AudioStream() {
	free(format);
}

WAVEFORMATEX *AudioStream::AllocFormat(long len) {
	if (format) { free(format); format = 0; }

	if (!(format = (WAVEFORMATEX *)malloc(len)))
		throw MyError("AudioStream: Out of memory");

    format_len = len;

	return format;
}

WAVEFORMATEX *AudioStream::GetFormat() {
	return format;
}

long AudioStream::GetFormatLen() {
	return format_len;
}

long AudioStream::GetSampleCount() {
	return samples_read;
}

long AudioStream::GetLength() {
	return stream_limit < stream_len ? stream_limit : stream_len;
}

long AudioStream::Read(void *buffer, long max_samples, long *lplBytes) {
	long actual;

	if (max_samples <= 0) {
		*lplBytes = 0;
		return 0;
	}

	if (samples_read >= stream_limit) {
		*lplBytes = 0;
		return 0;
	}

    if (samples_read + max_samples > stream_limit)
		max_samples = stream_limit - samples_read;

	actual = _Read(buffer, max_samples, lplBytes);

	_ASSERT(actual >= 0 && actual <= max_samples);

	samples_read += actual;

	return actual;
}

bool AudioStream::Skip(long samples) {
	return false;
}

void AudioStream::SetLimit(long limit) {
	_RPT1(0,"AudioStream: limit set to %ld\n",limit);
	stream_limit = limit;
}

void AudioStream::SetSource(AudioStream *src) {
	source = src;
	stream_len = src->GetLength();
}

BOOL AudioStream::isEnd() {
	return samples_read >= stream_limit || _isEnd();
}

long AudioStream::_Read(void *buffer, long max_samples, long *lplBytes) {
	*lplBytes = 0;

	return 0;
}

BOOL AudioStream::_isEnd() {
	return FALSE;
}

////////////////////

AudioStreamSource::AudioStreamSource(AudioSource *src, long first_samp, long max_samples, BOOL allow_decompression) : AudioStream() {
	WAVEFORMATEX *iFormat = src->getWaveFormat();
	WAVEFORMATEX *oFormat;
	WAVEFORMATEX  tFormat;
	MMRESULT res = 0;

	hACStream = NULL;
	inputBuffer = outputBuffer = NULL;
	fZeroRead = false;
	fStart = true;
	pwfexTempInput = NULL;
	lPreskip = 0;

	if (max_samples < 0)
		max_samples = 0;

    // Undo any Wave Format Extensible back to standard Wave FormatEx
	if (iFormat->wFormatTag == WAVE_FORMAT_EXTENSIBLE &&
	    iFormat->cbSize == sizeof(WAVEFORMATEXTENSIBLE)-sizeof(WAVEFORMATEX))
    {
      memcpy(&tFormat, iFormat, sizeof(WAVEFORMATEX));
      tFormat.cbSize = 0;
      tFormat.wFormatTag = (WORD)((WAVEFORMATEXTENSIBLE*)iFormat)->SubFormat.Data1;
      iFormat = &tFormat;
	}

	if (iFormat->wFormatTag != WAVE_FORMAT_PCM &&
		iFormat->wFormatTag != WAVE_FORMAT_IEEE_FLOAT && allow_decompression) {
		DWORD dwOutputBufferSize;
		DWORD dwOutputFormatSize;
		DWORD dwSuggest=0;
		int i;

		if (!AllocFormat(sizeof(WAVEFORMATEX)))
			throw MyMemoryError();

		if (acmMetrics(NULL, ACM_METRIC_MAX_SIZE_FORMAT, (LPVOID)&dwOutputFormatSize))
			throw MyError("Couldn't get ACM's max format size");

		oFormat = (WAVEFORMATEX *)malloc(dwOutputFormatSize); //AllocFormat(dwOutputFormatSize);
		if (!oFormat) throw MyMemoryError();

#define MAX_TRIES 10
		for (i=0; i<=MAX_TRIES; i++) {
			switch (i) {
			case 0: oFormat->wFormatTag = WAVE_FORMAT_IEEE_FLOAT;
					oFormat->nChannels = iFormat->nChannels;
					oFormat->nSamplesPerSec = iFormat->nSamplesPerSec;
					dwSuggest = ACM_FORMATSUGGESTF_WFORMATTAG | ACM_FORMATSUGGESTF_NCHANNELS |
						ACM_FORMATSUGGESTF_NSAMPLESPERSEC;
					break;
			case 1: oFormat->nSamplesPerSec = 0;
					dwSuggest = ACM_FORMATSUGGESTF_WFORMATTAG | ACM_FORMATSUGGESTF_NCHANNELS;
					break;
			case 2: oFormat->nChannels = 0;
					dwSuggest = ACM_FORMATSUGGESTF_WFORMATTAG;
					break;
			case 3: oFormat->wFormatTag = WAVE_FORMAT_PCM;
					oFormat->nChannels = iFormat->nChannels;
					oFormat->wBitsPerSample = 32;
					oFormat->nSamplesPerSec = iFormat->nSamplesPerSec;
					dwSuggest = ACM_FORMATSUGGESTF_WFORMATTAG | ACM_FORMATSUGGESTF_NCHANNELS |
						ACM_FORMATSUGGESTF_WBITSPERSAMPLE | ACM_FORMATSUGGESTF_NSAMPLESPERSEC;
					break;
			case 4: oFormat->nSamplesPerSec = 0;
					dwSuggest = ACM_FORMATSUGGESTF_WFORMATTAG | ACM_FORMATSUGGESTF_NCHANNELS |
						ACM_FORMATSUGGESTF_WBITSPERSAMPLE;
					break;
			case 5: oFormat->wBitsPerSample = 24;
					break;
			case 6: oFormat->wBitsPerSample = 0;
					dwSuggest = ACM_FORMATSUGGESTF_WFORMATTAG | ACM_FORMATSUGGESTF_NCHANNELS;
					break;
			case 7: oFormat->nChannels = 0;
					oFormat->wBitsPerSample = 32;
					dwSuggest = ACM_FORMATSUGGESTF_WFORMATTAG | ACM_FORMATSUGGESTF_WBITSPERSAMPLE;
					break;
			case 8: oFormat->wBitsPerSample = 24;
					break;
			case 9: oFormat->wBitsPerSample = 0;
					dwSuggest = ACM_FORMATSUGGESTF_WFORMATTAG;
					break;
					// Hack to get Fraunhoffer MP3 codec to accept data when wBitsPerSample==16
			case 10: if(iFormat->wFormatTag == 0x0055 && iFormat->wBitsPerSample != 0) {
						iFormat->wBitsPerSample = 0;
						break;
					}
					continue;
			default:
					break;
			}
			if (acmFormatSuggest(NULL, iFormat, oFormat, dwOutputFormatSize, dwSuggest)) continue;

			if (oFormat->wBitsPerSample!=8  && oFormat->wBitsPerSample!=16 &&
				oFormat->wBitsPerSample!=24 && oFormat->wBitsPerSample!=32)
					oFormat->wBitsPerSample=16;

			if (oFormat->nChannels==0)
				oFormat->nChannels = 2;

			oFormat->nBlockAlign		= (oFormat->wBitsPerSample/8) * oFormat->nChannels;
			oFormat->nAvgBytesPerSec	= oFormat->nBlockAlign * oFormat->nSamplesPerSec;
			oFormat->cbSize				= 0;


			if (!(res = acmStreamOpen(&hACStream, NULL, iFormat, oFormat, NULL, 0, 0, ACM_STREAMOPENF_NONREALTIME)))
				break;
		}
		if (res) {
			free(oFormat);
			if (res == ACMERR_NOTPOSSIBLE) {
				throw MyError(
							"Error initializing audio stream decompression:\n"
							"The requested conversion is not possible.\n"
							"\n"
							"Check to make sure you have the required codec%s."
							,
							(iFormat->wFormatTag&~1)==0x160 ? " (Microsoft Audio Codec)" : ""
						);
			} else
				throw MyError("Error initializing audio stream decompression.");
		}
		if (i > MAX_TRIES) {
			free(oFormat);
			throw MyError("No compatible ACM codec to decode 0x%04X audio stream to PCM.", iFormat->wFormatTag);
		}

		memcpy(GetFormat(), oFormat, sizeof(WAVEFORMATEX));
		free(oFormat);

		if (acmStreamSize(hACStream, INPUT_BUFFER_SIZE, &dwOutputBufferSize, ACM_STREAMSIZEF_SOURCE))
			throw MyError("Error initializing audio stream output size.");

		if (!(inputBuffer = malloc(INPUT_BUFFER_SIZE))
			|| !(outputBuffer = malloc(dwOutputBufferSize)))

			throw MyMemoryError();

		memset(&ashBuffer, 0, sizeof ashBuffer);
		ashBuffer.cbStruct		= sizeof(ACMSTREAMHEADER);
		ashBuffer.pbSrc			= (LPBYTE)inputBuffer;
		ashBuffer.cbSrcLength	= INPUT_BUFFER_SIZE;
		ashBuffer.pbDst			= (LPBYTE)outputBuffer;
		ashBuffer.cbDstLength	= dwOutputBufferSize;

		if (acmStreamPrepareHeader(hACStream, &ashBuffer, 0))
			throw MyError("Error preparing audio decompression buffers.");

		ashBuffer.cbSrcLength = 0;
		ashBuffer.cbDstLengthUsed = 0;
	} else {

		// FIX: If we have a PCMWAVEFORMAT stream, artificially cut the format size
		//		to sizeof(PCMWAVEFORMAT).  LSX-MPEG Encoder doesn't like large PCM
		//		formats!

		if (iFormat->wFormatTag == WAVE_FORMAT_PCM) {
			oFormat = AllocFormat(sizeof(PCMWAVEFORMAT));
			memcpy(oFormat, iFormat, sizeof(PCMWAVEFORMAT));
		} else {
			oFormat = AllocFormat(src->getFormatLen());
			memcpy(oFormat, iFormat, GetFormatLen());
		}
	}

	aSrc = src;
	stream_len = min(max_samples, aSrc->lSampleLast - first_samp);

	if (hACStream) {
		stream_len = MulDiv(stream_len, GetFormat()->nSamplesPerSec * aSrc->getWaveFormat()->nBlockAlign, aSrc->getWaveFormat()->nAvgBytesPerSec);
	}

	cur_samp = first_samp;
	end_samp = first_samp + max_samples;

}

AudioStreamSource::~AudioStreamSource() {
	if (hACStream) {
		if (ashBuffer.fdwStatus & ACMSTREAMHEADER_STATUSF_PREPARED) {
			ashBuffer.cbSrcLength = INPUT_BUFFER_SIZE;
			acmStreamUnprepareHeader(hACStream, &ashBuffer, 0);
		}
		acmStreamClose(hACStream, 0);
	}
	if (inputBuffer)	free(inputBuffer);
	if (outputBuffer)	free(outputBuffer);
	delete[] pwfexTempInput;
}

long AudioStreamSource::_Read(void *buffer, long max_samples, long *lplBytes) {
	LONG lSamples=0;
	LONG lAddedBytes=0;
	LONG lAddedSamples=0;
	int err;
	MMRESULT res;

	// add filler samples as necessary

	if (cur_samp < 0) {
		long tc = -cur_samp;
		const int nBlockAlign = aSrc->getWaveFormat()->nBlockAlign;

		if (tc > max_samples)
			tc = max_samples;

		if (GetFormat()->nChannels > 1)
			memset(buffer, 0, nBlockAlign*tc);
		else
			memset(buffer, 0x80, nBlockAlign*tc);

		buffer = (char *)buffer + nBlockAlign*tc;

		max_samples -= tc;
		lAddedBytes = tc*nBlockAlign;
		lAddedSamples = tc;
		cur_samp += tc;
	}

	// read actual samples

	if (hACStream) {
		LONG ltActualBytes, ltActualSamples;
		LONG lBytesLeft = max_samples * GetFormat()->nBlockAlign;
		LONG lTotalBytes = lBytesLeft;
		const int nBlockAlign = aSrc->getWaveFormat()->nBlockAlign;

		while(lBytesLeft > 0) {
			// hmm... data still in the output buffer?

			if (ashBuffer.cbDstLengthUsed>0) {
				long tc = min(lBytesLeft, long(ashBuffer.cbDstLengthUsed));


				if (lPreskip) {
					if (tc > lPreskip)
						tc = lPreskip;

					lPreskip -= tc;
				} else {
					memcpy(buffer, outputBufferPtr, tc);
					buffer = (void *)((char *)buffer + tc);
					lBytesLeft -= tc;
				}

				outputBufferPtr += tc;
				ashBuffer.cbDstLengthUsed -= tc;
				continue;
			}

			// fill the input buffer up... if we haven't gotten a zero yet.

			if (ashBuffer.cbSrcLength < INPUT_BUFFER_SIZE && !fZeroRead) {
				LONG lBytes=0;

				do {
					LONG to_read = (INPUT_BUFFER_SIZE - ashBuffer.cbSrcLength)/nBlockAlign;

					if (to_read > end_samp - cur_samp)
						to_read = end_samp - cur_samp;

					err = aSrc->read(cur_samp, to_read, (char *)inputBuffer + ashBuffer.cbSrcLength, INPUT_BUFFER_SIZE - ashBuffer.cbSrcLength, &ltActualBytes, &ltActualSamples);

					if (err != AVIERR_OK && err != AVIERR_BUFFERTOOSMALL) {
						if (err == AVIERR_FILEREAD)
							throw MyError("Audio samples %lu-%lu could not be read in the source.  The file may be corrupted.", cur_samp, cur_samp+to_read-1);
						else
							throw MyWin32Error("AudioStreamSource", err);
					}

					cur_samp += ltActualSamples;

//					_RPT1(0,"Read to %ld\n", cur_samp);

					ashBuffer.cbSrcLength += ltActualBytes;

					lBytes += ltActualBytes;

				} while(err != AVIERR_BUFFERTOOSMALL && ashBuffer.cbSrcLength < INPUT_BUFFER_SIZE && ltActualBytes && cur_samp < end_samp);

				if (!lBytes) fZeroRead = true;
			}

			// ask ACM to convert for us

			ashBuffer.cbSrcLengthUsed = 0;
			ashBuffer.cbDstLengthUsed = 0;

//	VDCHECKPOINT;
      if (ashBuffer.cbSrcLength) {
        res = acmStreamConvert(hACStream, &ashBuffer, (fStart ? ACM_STREAMCONVERTF_START : 0) | ACM_STREAMCONVERTF_BLOCKALIGN);
        if (res)
          throw MyError("ACM reported error on audio decompress (%lx)", res);
      }
//	VDCHECKPOINT;

			fStart = false;

			_RPT2(0,"Converted %ld bytes to %ld\n", ashBuffer.cbSrcLengthUsed, ashBuffer.cbDstLengthUsed);

			if (!ashBuffer.cbSrcLengthUsed && fZeroRead)
				break;

			// if ACM didn't use all the source data, copy the remainder down

			if (ashBuffer.cbSrcLengthUsed < ashBuffer.cbSrcLength) {
				long left = ashBuffer.cbSrcLength - ashBuffer.cbSrcLengthUsed;

				memmove(inputBuffer, (char *)inputBuffer + ashBuffer.cbSrcLengthUsed, left);

				ashBuffer.cbSrcLength = left;
			} else
				ashBuffer.cbSrcLength = 0;

			outputBufferPtr = (char *)outputBuffer;
		};

		*lplBytes = (lTotalBytes - lBytesLeft) + lAddedBytes;

		return *lplBytes / GetFormat()->nBlockAlign + lAddedSamples;
	} else {
		if (max_samples > end_samp - cur_samp)
			max_samples = end_samp - cur_samp;

		if (max_samples) {
			if (AVIERR_OK != (err = aSrc->read(cur_samp, max_samples, buffer, 0x7FFFFFFFL, lplBytes, &lSamples))) {
				if (err == AVIERR_FILEREAD)
					throw MyError("Audio samples %lu-%lu could not be read in the source.  The file may be corrupted.", cur_samp, cur_samp+max_samples-1);
				else
					throw MyWin32Error("AudioStreamSource", err);
			}

		} else
			lSamples = *lplBytes = 0;

		*lplBytes += lAddedBytes;

		cur_samp += lSamples;

		return lSamples + lAddedSamples;
	}
}

bool AudioStreamSource::Skip(long samples) {

	// nBlockAlign = bytes per block.
	//
	// nAvgBytesPerSec / nBlockAlign = blocks per second.
	// nSamplesPerSec * nBlockAlign / nAvgBytesPerSec = samples per block.

	if (hACStream) {
		const WAVEFORMATEX *pwfex = aSrc->getWaveFormat();

		if (samples < MulDiv(4*pwfex->nBlockAlign, pwfex->nSamplesPerSec, pwfex->nAvgBytesPerSec)) {
			lPreskip += samples*GetFormat()->nBlockAlign;
			return true;
		}

		// Flush input and output buffers.

		ashBuffer.cbSrcLength = 0;
		ashBuffer.cbDstLengthUsed = 0;

		// Trigger a reseek.

		long new_pos = long(((samples_read + samples) * (__int64)pwfex->nAvgBytesPerSec) / ((__int64)pwfex->nBlockAlign*pwfex->nSamplesPerSec));

		if (new_pos > cur_samp)
			cur_samp = new_pos;

		fStart = true;

		// Skip fractional samples.

		long samp_start = long((new_pos * (__int64)pwfex->nSamplesPerSec*pwfex->nBlockAlign) / pwfex->nAvgBytesPerSec);

		lPreskip = ((samples_read + samples) - samp_start)*GetFormat()->nBlockAlign;

		samples_read = samp_start;

		return true;

	} else {
		cur_samp += samples;
		samples_read += samples;

		return true;
	}
}

bool AudioStreamSource::Seek(long samples) {

	// nBlockAlign = bytes per block.
	//
	// nAvgBytesPerSec / nBlockAlign = blocks per second.
	// nSamplesPerSec * nBlockAlign / nAvgBytesPerSec = samples per block.

	if (samples < end_samp)
		fZeroRead = false;

	if (hACStream) {
		const WAVEFORMATEX *pwfex = aSrc->getWaveFormat();

		if (samples > samples_read && samples - samples_read < MulDiv(4*pwfex->nBlockAlign, pwfex->nSamplesPerSec, pwfex->nAvgBytesPerSec)) {
			lPreskip += (samples - samples_read)*GetFormat()->nBlockAlign;
			return true;
		}

		// Flush input and output buffers.

		ashBuffer.cbSrcLength = 0;
		ashBuffer.cbDstLengthUsed = 0;

		// Trigger a reseek.

		long new_pos = long((samples * (__int64)pwfex->nAvgBytesPerSec) / ((__int64)pwfex->nBlockAlign*pwfex->nSamplesPerSec));

		cur_samp = new_pos;

		fStart = true;

		// Skip fractional samples.

		long samp_start = long((new_pos * (__int64)pwfex->nSamplesPerSec*pwfex->nBlockAlign) / pwfex->nAvgBytesPerSec);

		lPreskip = (samples - samp_start)*GetFormat()->nBlockAlign;

		samples_read = samp_start;

		return true;

	} else {
		cur_samp = samples;
		samples_read = samples;

		return true;
	}
}

BOOL AudioStreamSource::_isEnd() {
	return (cur_samp >= end_samp || fZeroRead) && (!hACStream || !ashBuffer.cbDstLengthUsed);
}



///////////////////////////////////////////////////////////////////////////
//
//		AudioStreamConverter
//
//		This audio filter handles changes in format between 8/16-bit
//		and mono/stereo.
//
///////////////////////////////////////////////////////////////////////////



AudioStreamConverter::AudioStreamConverter(AudioStream *src, bool to_16bit, bool to_stereo_or_right, bool single_only) {
	WAVEFORMATEX *iFormat = src->GetFormat();
	WAVEFORMATEX *oFormat;
	bool to_stereo = single_only ? false : to_stereo_or_right;

	memcpy(oFormat = AllocFormat(src->GetFormatLen()), iFormat, src->GetFormatLen());

	oFormat->nChannels = to_stereo ? 2 : 1;
	oFormat->wBitsPerSample = to_16bit ? 16 : 8;

	bytesPerInputSample = (iFormat->nChannels>1 ? 2 : 1)
						* (iFormat->wBitsPerSample>8 ? 2 : 1);

	bytesPerOutputSample = (to_stereo ? 2 : 1)
						 * (to_16bit ? 2 : 1);

	offset = 0;

	if (single_only) {
		convRout = AudioPickConverterSingleChannel(iFormat, to_16bit);

		if (to_stereo_or_right && iFormat->nChannels>1) {
			offset = 1;

			if (iFormat->wBitsPerSample>8)
				offset = 2;
		}
	} else
		convRout = AudioPickConverter(iFormat, to_16bit, to_stereo);
	SetSource(src);

	oFormat->nAvgBytesPerSec = oFormat->nSamplesPerSec * bytesPerOutputSample;
	oFormat->nBlockAlign = (WORD)bytesPerOutputSample;


	if (!(cbuffer = malloc(bytesPerInputSample * BUFFER_SIZE)))
		throw MyError("AudioStreamConverter: out of memory");
}

AudioStreamConverter::~AudioStreamConverter() {
	free(cbuffer);
}

long AudioStreamConverter::_Read(void *buffer, long samples, long *lplBytes) {
	long lActualSamples=0;

	while(samples>0) {
		long srcSamples;
		long lBytes;

		// figure out how many source samples we need

		srcSamples = samples;

		if (srcSamples > BUFFER_SIZE) srcSamples = BUFFER_SIZE;

		srcSamples = source->Read(cbuffer, srcSamples, &lBytes);

		if (!srcSamples) break;

		convRout(buffer, (char *)cbuffer + offset, srcSamples);

		buffer = (void *)((char *)buffer + bytesPerOutputSample * srcSamples);
		lActualSamples += srcSamples;
		samples -= srcSamples;

	}

	*lplBytes = lActualSamples * bytesPerOutputSample;

	return lActualSamples;
}

BOOL AudioStreamConverter::_isEnd() {
	return source->isEnd();
}

bool AudioStreamConverter::Skip(long samples) {
	return source->Skip(samples);
}



///////////////////////////////////////////////////////////////////////////
//
//		AudioStreamResampler
//
//		This audio filter handles changes in sampling rate.
//
///////////////////////////////////////////////////////////////////////////

static long audio_pointsample_8(void *dst, void *src, long accum, long samp_frac, long cnt) {
	unsigned char *d = (unsigned char *)dst;
	unsigned char *s = (unsigned char *)src;

	do {
		*d++ = s[accum>>19];
		accum += samp_frac;
	} while(--cnt);

	return accum;
}

static long audio_pointsample_16(void *dst, void *src, long accum, long samp_frac, long cnt) {
	unsigned short *d = (unsigned short *)dst;
	unsigned short *s = (unsigned short *)src;

	do {
		*d++ = s[accum>>19];
		accum += samp_frac;
	} while(--cnt);

	return accum;
}

static long audio_pointsample_32(void *dst, void *src, long accum, long samp_frac, long cnt) {
	unsigned long *d = (unsigned long *)dst;
	unsigned long *s = (unsigned long *)src;

	do {
		*d++ = s[accum>>19];
		accum += samp_frac;
	} while(--cnt);

	return accum;
}

static long audio_downsample_mono8(void *dst, void *src, long *filter_bank, int filter_width, long accum, long samp_frac, long cnt) {
	unsigned char *d = (unsigned char *)dst;
	unsigned char *s = (unsigned char *)src;

	do {
		long sum = 0;
		int w;
		long *fb_ptr;
		unsigned char *s_ptr;

		w = filter_width;
		fb_ptr = filter_bank + filter_width * ((accum>>11)&0xff);
		s_ptr = s + (accum>>19);
		do {
			sum += *fb_ptr++ * (int)*s_ptr++;
		} while(--w);

		if (sum < 0)
			*d++ = 0;
		else if (sum > 0x3fffff)
			*d++ = 0xff;
		else
			*d++ = (unsigned char)((sum + 0x2000)>>14);

		accum += samp_frac;
	} while(--cnt);

	return accum;
}

static long audio_downsample_mono16(void *dst, void *src, long *filter_bank, int filter_width, long accum, long samp_frac, long cnt) {
	signed short *d = (signed short *)dst;
	signed short *s = (signed short *)src;

	do {
		long sum = 0;
		int w;
		long *fb_ptr;
		signed short *s_ptr;

		w = filter_width;
		fb_ptr = filter_bank + filter_width * ((accum>>11)&0xff);
		s_ptr = s + (accum>>19);
		do {
			sum += *fb_ptr++ * (int)*s_ptr++;
		} while(--w);

		if (sum < -0x20000000)
			*d++ = -0x8000;
		else if (sum > 0x1fffffff)
			*d++ = 0x7fff;
		else
			*d++ = (short)((sum + 0x2000)>>14);

		accum += samp_frac;
	} while(--cnt);

	return accum;
}

static long audio_downsample_stereo8(void *dst, void *src, long *filter_bank, int filter_width, long accum, long samp_frac, long cnt) {
	unsigned char *d = (unsigned char *)dst;
	unsigned char *s = (unsigned char *)src;

	do {
		long sum_l = 0, sum_r = 0;
		int w;
		long *fb_ptr;
		unsigned char *s_ptr;

		w = filter_width;
		fb_ptr = filter_bank + filter_width * ((accum>>11)&0xff);
		s_ptr = s + (accum>>19)*2;
		do {
			long f = *fb_ptr++;

			sum_l += f * (int)*s_ptr++;
			sum_r += f * (int)*s_ptr++;
		} while(--w);

		if (sum_l < 0)
			*d++ = 0;
		else if (sum_l > 0x3fffff)
			*d++ = 0xff;
		else
			*d++ = (unsigned char)((sum_l + 0x2000)>>14);

		if (sum_r < 0)
			*d++ = 0;
		else if (sum_r > 0x3fffff)
			*d++ = 0xff;
		else
			*d++ = (unsigned char)((sum_r + 0x2000)>>14);

		accum += samp_frac;
	} while(--cnt);

	return accum;
}

static long audio_downsample_stereo16(void *dst, void *src, long *filter_bank, int filter_width, long accum, long samp_frac, long cnt) {
	signed short *d = (signed short *)dst;
	signed short *s = (signed short *)src;

	do {
		long sum_l = 0, sum_r = 0;
		int w;
		long *fb_ptr;
		signed short *s_ptr;

		w = filter_width;
		fb_ptr = filter_bank + filter_width * ((accum>>11)&0xff);
		s_ptr = s + (accum>>19)*2;
		do {
			long f = *fb_ptr++;

			sum_l += f * (int)*s_ptr++;
			sum_r += f * (int)*s_ptr++;
		} while(--w);

		if (sum_l < -0x20000000)
			*d++ = -0x8000;
		else if (sum_l > 0x1fffffff)
			*d++ = 0x7fff;
		else
			*d++ = (short)((sum_l + 0x2000)>>14);

		if (sum_r < -0x20000000)
			*d++ = -0x8000;
		else if (sum_r > 0x1fffffff)
			*d++ = 0x7fff;
		else
			*d++ = (short)((sum_r + 0x2000)>>14);

		accum += samp_frac;
	} while(--cnt);

	return accum;
}

static long audio_upsample_mono8(void *dst, void *src, long accum, long samp_frac, long cnt) {
	unsigned char *d = (unsigned char *)dst;
	unsigned char *s = (unsigned char *)src;

	do {
		unsigned char *s_ptr = s + (accum>>19);
		long frac = (accum>>3) & 0xffff;

		*d++ = (unsigned char)(((int)s_ptr[0] * (0x10000 - frac) + (int)s_ptr[1] * frac) >> 16);
		accum += samp_frac;
	} while(--cnt);

	return accum;
}

static long audio_upsample_mono16(void *dst, void *src, long accum, long samp_frac, long cnt) {
	signed short *d = (signed short *)dst;
	signed short *s = (signed short *)src;

	do {
		signed short *s_ptr = s + (accum>>19);
		long frac = (accum>>3) & 0xffff;

		*d++ = short(((int)s_ptr[0] * (0x10000 - frac) + (int)s_ptr[1] * frac) >> 16);
		accum += samp_frac;
	} while(--cnt);

	return accum;
}

static long audio_upsample_stereo8(void *dst, void *src, long accum, long samp_frac, long cnt) {
	unsigned char *d = (unsigned char *)dst;
	unsigned char *s = (unsigned char *)src;

	do {
		unsigned char *s_ptr = s + (accum>>19)*2;
		long frac = (accum>>3) & 0xffff;

		*d++ = (unsigned char)(((int)s_ptr[0] * (0x10000 - frac) + (int)s_ptr[2] * frac) >> 16);
		*d++ = (unsigned char)(((int)s_ptr[1] * (0x10000 - frac) + (int)s_ptr[3] * frac) >> 16);
		accum += samp_frac;
	} while(--cnt);

	return accum;
}

static long audio_upsample_stereo16(void *dst, void *src, long accum, long samp_frac, long cnt) {
	signed short *d = (signed short *)dst;
	signed short *s = (signed short *)src;

	do {
		signed short *s_ptr = s + (accum>>19)*2;
		long frac = (accum>>3) & 0xffff;

		*d++ = short(((int)s_ptr[0] * (0x10000 - frac) + (int)s_ptr[2] * frac) >> 16);
		*d++ = short(((int)s_ptr[1] * (0x10000 - frac) + (int)s_ptr[3] * frac) >> 16);
		accum += samp_frac;
	} while(--cnt);

	return accum;
}

static int permute_index(int a, int b) {
	return (b-(a>>8)-1) + (a&255)*b;
}

static void make_downsample_filter(long *filter_bank, int filter_width, long samp_frac) {
	int i, j, v;
	double filt_max;
	double filtwidth_frac;

	filtwidth_frac = samp_frac/2048.0;

	filter_bank[filter_width-1] = 0;

	filt_max = (16384.0 * 524288.0) / samp_frac;

	for(i=0; i<128*filter_width; i++) {
		int y = 0;
		double d = i / filtwidth_frac;

		if (d<1.0)
			y = (int)(0.5 + filt_max*(1.0 - d));

		filter_bank[permute_index(128*filter_width + i, filter_width)]
			= filter_bank[permute_index(128*filter_width - i, filter_width)]
			= y;
	}

	// Normalize the filter to correct for integer roundoff errors

	for(i=0; i<256*filter_width; i+=filter_width) {
		v=0;
		for(j=0; j<filter_width; j++)
			v += filter_bank[i+j];

		_RPT2(0,"error[%02x] = %04x\n", i/filter_width, 0x4000 - v);

		v = (0x4000 - v)/filter_width;
		for(j=0; j<filter_width; j++)
			filter_bank[i+j] += v;
	}

	_CrtCheckMemory();
}

AudioStreamResampler::AudioStreamResampler(AudioStream *src, long new_rate, bool integral_conversion, bool hi_quality) : AudioStream() {
	WAVEFORMATEX *iFormat = src->GetFormat();
	WAVEFORMATEX *oFormat;

	memcpy(oFormat = AllocFormat(src->GetFormatLen()), iFormat, src->GetFormatLen());

	if (oFormat->nChannels>1)
		if (oFormat->wBitsPerSample>8) {
			ptsampleRout = audio_pointsample_32;
			upsampleRout = audio_upsample_stereo16;
			dnsampleRout = audio_downsample_stereo16;
		} else {
			ptsampleRout = audio_pointsample_16;
			upsampleRout = audio_upsample_stereo8;
			dnsampleRout = audio_downsample_stereo8;
		}
	else
		if (oFormat->wBitsPerSample>8) {
			ptsampleRout = audio_pointsample_16;
			upsampleRout = audio_upsample_mono16;
			dnsampleRout = audio_downsample_mono16;
		} else {
			ptsampleRout = audio_pointsample_8;
			upsampleRout = audio_upsample_mono8;
			dnsampleRout = audio_downsample_mono8;
		}

	SetSource(src);

	bytesPerSample = (iFormat->nChannels>1 ? 2 : 1)
						* (iFormat->wBitsPerSample>8 ? 2 : 1);

	_RPT2(0,"AudioStreamResampler: converting from %ldHz to %ldHz\n", iFormat->nSamplesPerSec, new_rate);

	if (integral_conversion)
		if (new_rate > long(iFormat->nSamplesPerSec))
//			samp_frac = MulDiv(0x10000, new_rate + iFormat->nSamplesPerSec/2, iFormat->nSamplesPerSec);
			samp_frac = 0x80000 / ((new_rate + iFormat->nSamplesPerSec/2) / iFormat->nSamplesPerSec);
		else
			samp_frac = 0x80000 * ((iFormat->nSamplesPerSec + new_rate/2) / new_rate);
	else
		samp_frac = MulDiv(iFormat->nSamplesPerSec, 0x80000L, new_rate);

	stream_len = MulDiv(stream_len, 0x80000L, samp_frac);

	oFormat->nSamplesPerSec = MulDiv(iFormat->nSamplesPerSec, 0x80000L, samp_frac);
	oFormat->nAvgBytesPerSec = oFormat->nSamplesPerSec * bytesPerSample;
	oFormat->nBlockAlign = (WORD)bytesPerSample;

	holdover = 0;
	filter_bank = NULL;
	filter_width = 1;
	accum=0;
	fHighQuality = hi_quality;

	if (!(cbuffer = malloc(bytesPerSample * BUFFER_SIZE)))
		throw MyMemoryError();

	// Initialize the buffer.

	if (oFormat->wBitsPerSample>8)
		memset(cbuffer, 0x00, bytesPerSample * BUFFER_SIZE);
	else
		memset(cbuffer, 0x80, bytesPerSample * BUFFER_SIZE);

	// If this is a high-quality downsample, allocate memory for the filter bank

	if (hi_quality) {
		if (samp_frac>0x80000) {

			// HQ downsample: allocate filter bank

			filter_width = ((samp_frac + 0x7ffff)>>19)<<1;

			if (!(filter_bank = new(std::nothrow) long[filter_width * 256])) {
				free(cbuffer);
				throw MyMemoryError();
			}

			make_downsample_filter(filter_bank, filter_width, samp_frac);

			// Clear lower samples

			if (oFormat->wBitsPerSample>8)
				memset(cbuffer, 0, bytesPerSample*filter_width);
			else
				memset(cbuffer, 0x80, bytesPerSample*filter_width);

			holdover = filter_width/2;
		}
	}
}

AudioStreamResampler::~AudioStreamResampler() {
	free(cbuffer);
	delete filter_bank;
}

long AudioStreamResampler::_Read(void *buffer, long samples, long *lplBytes) {

	if (samp_frac == 0x80000)
		return source->Read(buffer, samples, lplBytes);

	if (samp_frac < 0x80000)
		return Upsample(buffer, samples, lplBytes);
	else
		return Downsample(buffer, samples, lplBytes);
}


long AudioStreamResampler::Upsample(void *buffer, long samples, long *lplBytes) {
	long lActualSamples=0;

	// Upsampling: producing more output samples than input
	//
	// There are two issues we need to watch here:
	//
	//	o  An input sample can be read more than once.  In particular, even
	//	   when point sampling, we may need the last input sample again.
	//
	//	o  When interpolating (HQ), we need one additional sample.

	while(samples>0) {
		long srcSamples, dstSamples;
		long lBytes;
		int holdover = 0;

		// A negative accum value indicates that we need to reprocess a sample.
		// The last iteration should have left it at the bottom of the buffer
		// for us.  In interpolation mode, we'll always have at least a 1
		// sample overlap.

		if (accum<0) {
			holdover = 1;
			accum += 0x80000;
		}

		if (fHighQuality)
			++holdover;

		// figure out how many source samples we need

		srcSamples = (long)(((__int64)samp_frac*(samples-1) + accum) >> 19) + 1 - holdover;

		if (fHighQuality)
			++srcSamples;

		if (srcSamples > BUFFER_SIZE-holdover) srcSamples = BUFFER_SIZE-holdover;

		srcSamples = source->Read((char *)cbuffer + holdover * bytesPerSample, srcSamples, &lBytes);

		if (!srcSamples) break;

		srcSamples += holdover;

		// figure out how many destination samples we'll get out of what we read

		if (fHighQuality)
			dstSamples = ((srcSamples<<19) - accum - 0x80001)/samp_frac + 1;
		else
			dstSamples = ((srcSamples<<19) - accum - 1)/samp_frac + 1;

		if (dstSamples > samples)
			dstSamples = samples;

		if (dstSamples>=1) {

			if (fHighQuality)
				accum = upsampleRout(buffer, cbuffer, accum, samp_frac, dstSamples);
			else
				accum = ptsampleRout(buffer, cbuffer, accum, samp_frac, dstSamples);

			buffer = (void *)((char *)buffer + bytesPerSample * dstSamples);
			lActualSamples += dstSamples;
			samples -= dstSamples;
		}

		if (fHighQuality)
			accum -= ((srcSamples-1)<<19);
		else
			accum -= (srcSamples<<19);

		// do we need to hold a sample over?

		if (fHighQuality)
			if (accum<0)
				memcpy(cbuffer, (char *)cbuffer + (srcSamples-2)*bytesPerSample, bytesPerSample*2);
			else
				memcpy(cbuffer, (char *)cbuffer + (srcSamples-1)*bytesPerSample, bytesPerSample);
		else if (accum<0)
			memcpy(cbuffer, (char *)cbuffer + (srcSamples-1)*bytesPerSample, bytesPerSample);
	}

	*lplBytes = lActualSamples * bytesPerSample;

//	_RPT2(0,"Converter: %ld samples, %ld bytes\n", lActualSamples, *lplBytes);

	return lActualSamples;
}

long AudioStreamResampler::Downsample(void *buffer, long samples, long *lplBytes) {
	long lActualSamples=0;

	// Downsampling is even worse because we have overlap to the left and to the
	// right of the interpolated point.
	//
	// We need (n/2) points to the left and (n/2-1) points to the right.

	while(samples>0) {
		long srcSamples, dstSamples;
		long lBytes;
		int nhold;

		// Figure out how many source samples we need.
		//
		// To do this, compute the highest fixed-point accumulator we'll reach.
		// Truncate that, and add the filter width.  Then subtract however many
		// samples are sitting at the bottom of the buffer.

		srcSamples = (long)(((__int64)samp_frac*(samples-1) + accum) >> 19) + filter_width - holdover;

		// Don't exceed the buffer (BUFFER_SIZE - holdover).

		if (srcSamples > BUFFER_SIZE - holdover)
			srcSamples = BUFFER_SIZE - holdover;

		// Read into buffer.

		srcSamples = source->Read((char *)cbuffer + holdover*bytesPerSample, srcSamples, &lBytes);

		if (!srcSamples) break;

		// Figure out how many destination samples we'll get out of what we
		// read.  We'll have (srcSamples+holdover) bytes, so the maximum
		// fixed-pt accumulator we can hit is
		// (srcSamples+holdover-filter_width)<<16 + 0xffff.

		dstSamples = (((srcSamples+holdover-filter_width)<<19) + 0x7ffff - accum) / samp_frac + 1;

		if (dstSamples > samples)
			dstSamples = samples;

		if (dstSamples>=1) {
			if (filter_bank)
				accum = dnsampleRout(buffer, cbuffer, filter_bank, filter_width, accum, samp_frac, dstSamples);
			else
				accum = ptsampleRout(buffer, cbuffer, accum, samp_frac, dstSamples);

			buffer = (void *)((char *)buffer + bytesPerSample * dstSamples);
			lActualSamples += dstSamples;
			samples -= dstSamples;
		}

		// We're "shifting" the new samples down to the bottom by discarding
		// all the samples in the buffer, so adjust the fixed-pt accum
		// accordingly.

		accum -= ((srcSamples+holdover)<<19);

		// Oops, did we need some of those?
		//
		// If accum=0, we need (n/2) samples back.  accum>=0x10000 is fewer,
		// accum<0 is more.

		nhold = - (accum>>19);

//		_ASSERT(nhold<=(filter_width/2));

		if (nhold>0) {
			memmove(cbuffer, (char *)cbuffer+bytesPerSample*(srcSamples+holdover-nhold), bytesPerSample*nhold);
			holdover = nhold;
			accum += nhold<<19;
		} else
			holdover = 0;

		_ASSERT(accum>=0);
	}

	*lplBytes = lActualSamples * bytesPerSample;

	return lActualSamples;
}

BOOL AudioStreamResampler::_isEnd() {
	return accum>=0 && source->isEnd();
}



///////////////////////////////////////////////////////////////////////////
//
//		AudioCompressor
//
//		This audio filter handles audio compression.
//
///////////////////////////////////////////////////////////////////////////


struct FrustratedACMOpenData {
	HACMSTREAM *pp;
	HACMDRIVER hdrv;
	WAVEFORMATEX *iFormat;
	WAVEFORMATEX *oFormat;
	bool success;
};


BOOL CALLBACK ACMStreamOpenCallback(HACMDRIVERID hadid, DWORD_PTR dwInstance, DWORD fdwSupport)
{
	FrustratedACMOpenData *pfad = (FrustratedACMOpenData *)dwInstance;

	// Ignore drivers that don't do format conversion.

	if (fdwSupport & ACMDRIVERDETAILS_SUPPORTF_CODEC) {
		MMRESULT res;
		ACMDRIVERDETAILS add;

		memset(&add, 0, sizeof add);
		add.cbStruct = sizeof add;

		// Attempt to open driver.

		res = acmDriverOpen(&pfad->hdrv, hadid, 0);

		if (!res) {

			res = acmDriverDetails(hadid, &add, 0);
      if (!res) {
				_RPT1(0,"Trying driver: [%s]\n", add.szLongName);
      }

			// Attempt our stream open!

			res = acmStreamOpen(pfad->pp, pfad->hdrv, pfad->iFormat, pfad->oFormat, NULL, 0, 0, ACM_STREAMOPENF_NONREALTIME);
			if (!res)
				pfad->success = true;
			else {
				res = acmStreamOpen(pfad->pp, pfad->hdrv, pfad->iFormat, pfad->oFormat, NULL, 0, 0, 0);
				if (!res)
					pfad->success = true;
				else
					acmDriverClose(pfad->hdrv, 0);
			}
		}
	}

	return !pfad->success;
}


AudioCompressor::AudioCompressor(AudioStream *src, WAVEFORMATEX *dst_format, long dst_format_len) : AudioStream() {
	WAVEFORMATEX *iFormat = src->GetFormat();
	WAVEFORMATEX *oFormat;
	DWORD dwOutputBufferSize;
	MMRESULT err;

	hADriver = NULL;
	hACStream = NULL;
	pwfexTempOutput = NULL;
	inputBuffer = outputBuffer = NULL;
	holdBuffer = NULL;
	holdBufferSize = 0;
	holdBufferOffset = 0;

	// Stupid Microsoft Audio Codec.

	oFormat = AllocFormat(dst_format_len);
	memcpy(oFormat, dst_format, dst_format_len);

	SetSource(src);

	memset(&ashBuffer, 0, sizeof ashBuffer);

	for (;;) {
		// Try opening with ACM_STREAMOPENF_NONREALTIME.

		if (!(err = acmStreamOpen(&hACStream, NULL, iFormat, oFormat, NULL, 0, 0, ACM_STREAMOPENF_NONREALTIME)))
			break;

		// Still didn't work, try every f*cking driver.

		struct FrustratedACMOpenData fad;

		fad.pp		= &hACStream;
		fad.hdrv	= NULL;
		fad.iFormat = iFormat;
		fad.oFormat = oFormat;
		fad.success = false;

		if (!acmDriverEnum(ACMStreamOpenCallback, (DWORD_PTR)&fad, 0) && fad.success) {
			hADriver = fad.hdrv;
			break;
		}

		// Damn!

		if (err == ACMERR_NOTPOSSIBLE)
			throw MyError("Error initializing audio stream compression:\nThe requested conversion is not possible.");
		else
			throw MyError("Error initializing audio stream compression.");
	}

	if (acmStreamSize(hACStream, INPUT_BUFFER_SIZE, &dwOutputBufferSize, ACM_STREAMSIZEF_SOURCE))
		throw MyError("Error querying audio compression.");

	if (!(inputBuffer = malloc(INPUT_BUFFER_SIZE))
		|| !(outputBuffer = malloc(dwOutputBufferSize)))

		throw MyMemoryError();

	ashBuffer.cbStruct		= sizeof(ACMSTREAMHEADER);
	ashBuffer.pbSrc			= (LPBYTE)inputBuffer;
	ashBuffer.cbSrcLength	= INPUT_BUFFER_SIZE;
	ashBuffer.pbDst			= (LPBYTE)outputBuffer;
	ashBuffer.cbDstLength	= dwOutputBufferSize;

	if (acmStreamPrepareHeader(hACStream, &ashBuffer, 0))
		throw MyError("Error preparing audio compression buffers.");

	ashBuffer.cbSrcLength = 0;

	bytesPerInputSample = iFormat->nBlockAlign;
	bytesPerOutputSample = oFormat->nBlockAlign;

	fStreamEnded = FALSE;
}

AudioCompressor::~AudioCompressor() {
	if (hACStream) {
		if (ashBuffer.fdwStatus & ACMSTREAMHEADER_STATUSF_PREPARED) acmStreamUnprepareHeader(hACStream, &ashBuffer, 0);
		acmStreamClose(hACStream, 0);
	}
	if (hADriver)
		acmDriverClose(hADriver, 0);

	if (inputBuffer)	free(inputBuffer);
	if (outputBuffer)	free(outputBuffer);
	if (holdBuffer)		free(holdBuffer);

	delete[] pwfexTempOutput;
}

void AudioCompressor::CompensateForMP3() {

	// Fraunhofer-IIS's MP3 codec has a compression delay that we need to
	// compensate for.  Comparison of PCM input, F-IIS output, and
	// WinAmp's Nitrane output reveals that the decompressor half of the
	// ACM codec is fine, but the compressor inserts a delay of 1373
	// (0x571) samples at the start.  This is a lag of 2 frames at
	// 30fps and 22KHz, so it's significant enough to be noticed.  At
	// 11KHz, this becomes a tenth of a second.  Needless to say, the
	// F-IIS MP3 codec is a royal piece of sh*t.
	//
	// By coincidence, the MPEGLAYER3WAVEFORMAT struct has a field
	// called nCodecDelay which is set to this value...

	if (GetFormat()->wFormatTag == WAVE_FORMAT_MPEGLAYER3) {
		long lSamplesToRead = ((MPEGLAYER3WAVEFORMAT *)GetFormat())->nCodecDelay;

		// Note: LameACM does not have a codec delay!

		if (lSamplesToRead) {
			long ltActualBytes, ltActualSamples;
			int nBlockAlign = source->GetFormat()->nBlockAlign;

			do {
				long tc;

				tc = lSamplesToRead;
				if (tc > INPUT_BUFFER_SIZE / nBlockAlign)
					tc = INPUT_BUFFER_SIZE / nBlockAlign;

				ltActualSamples = source->Read((char *)inputBuffer, tc,
							&ltActualBytes);

				lSamplesToRead -= ltActualSamples;
			} while(lSamplesToRead>0 && ltActualBytes);

			if (!ltActualBytes || source->isEnd())
				fStreamEnded = TRUE;
		}
	}
}

void AudioCompressor::ResizeHoldBuffer(long lNewSize) {
	void *holdBufferTemp;

	if (holdBufferSize >= lNewSize) return;

	if (!(holdBufferTemp = malloc(holdBufferSize = ((lNewSize + 65535) & -65536))))
		throw MyError("AudioCompressor: Unable to resize hold buffer");

	memcpy(holdBufferTemp, holdBuffer, holdBufferOffset);
	free(holdBuffer);

	holdBuffer = holdBufferTemp;
}

void AudioCompressor::WriteToHoldBuffer(void *data, long lBytes) {
	if (lBytes + holdBufferOffset > holdBufferSize)
		ResizeHoldBuffer(lBytes + holdBufferOffset);

	memcpy((char *)holdBuffer + holdBufferOffset, data, lBytes);
	holdBufferOffset += lBytes;
}

void *AudioCompressor::Compress(long lInputSamples, long *lplSrcInputSamples, long *lplOutputBytes, long *lplOutputSamples) {
	LONG ltActualSamples, ltActualBytes;
	LONG lActualSrcSamples = 0;

//	_RPT1(0,"Compressor: we want to convert %ld input samples\n", lInputSamples);

	holdBufferOffset = 0;

	while(lInputSamples > 0 && !fStreamEnded) {

		// fill the input buffer up!

		if (ashBuffer.cbSrcLength < INPUT_BUFFER_SIZE) {
			//LONG lBytes=0;
			long lSamplesToRead;

			do {
				lSamplesToRead = min(lInputSamples, (INPUT_BUFFER_SIZE - (long)ashBuffer.cbSrcLength)/bytesPerInputSample);

				ltActualSamples = source->Read((char *)inputBuffer + ashBuffer.cbSrcLength, lSamplesToRead,
							&ltActualBytes);

				ashBuffer.cbSrcLength += ltActualBytes;

				//lBytes += ltActualBytes;

				lInputSamples -= ltActualSamples;
				lActualSrcSamples += ltActualSamples;

				_ASSERT(lInputSamples >= 0);
			} while(ashBuffer.cbSrcLength < INPUT_BUFFER_SIZE && lInputSamples>0 && ltActualBytes);

			if (!ltActualBytes || source->isEnd())
				fStreamEnded = TRUE;
		}

		// ask ACM to convert for us

		ashBuffer.cbSrcLengthUsed = 0;
		ashBuffer.cbDstLengthUsed = 0;

//		_RPT2(0,"Converting %ld bytes to %ld\n", ashBuffer.cbSrcLength, ashBuffer.cbDstLength);

//	VDCHECKPOINT;
		if (acmStreamConvert(hACStream, &ashBuffer, fStreamEnded ? ACM_STREAMCONVERTF_END : ACM_STREAMCONVERTF_BLOCKALIGN))
			throw MyError("Audio Compression Manager (ACM) failure on compress");
//	VDCHECKPOINT;

//		_RPT2(0,"Converted %ld bytes to %ld\n", ashBuffer.cbSrcLengthUsed, ashBuffer.cbDstLengthUsed);

		// if ACM didn't use all the source data, copy the remainder down

		if (ashBuffer.cbSrcLengthUsed < ashBuffer.cbSrcLength) {
			long left = ashBuffer.cbSrcLength - ashBuffer.cbSrcLengthUsed;

			memmove(inputBuffer, (char *)inputBuffer + ashBuffer.cbSrcLengthUsed, left);

			ashBuffer.cbSrcLength = left;
		} else
			ashBuffer.cbSrcLength = 0;

		// chuck all pending data to the hold buffer

		WriteToHoldBuffer(outputBuffer, ashBuffer.cbDstLengthUsed);
	}

	*lplOutputBytes = holdBufferOffset;
	*lplOutputSamples = (holdBufferOffset + bytesPerOutputSample - 1) / bytesPerOutputSample;
	*lplSrcInputSamples = lActualSrcSamples;

//	_RPT2(0,"Compressor: %ld bytes, %ld samples\n", *lplOutputBytes, *lplOutputSamples);

	return holdBuffer;
}

BOOL AudioCompressor::isEnd() {
	return fStreamEnded;
}

///////////////////////////////////////////////////////////////////////////
//
//	Corrects the nAvgBytesPerFrame for that stupid Fraunhofer-IIS
//	codec.


AudioL3Corrector::AudioL3Corrector() {
	samples = frame_bytes = 0;
	read_left = 4;
	header_mode = true;
}

long AudioL3Corrector::ComputeByterate(long sample_rate) {
	return MulDiv(frame_bytes, sample_rate, samples);
}

void AudioL3Corrector::Process(void *buffer, long bytes) {
	static const int bitrates[2][16]={
		{0, 8,16,24,32,40,48,56, 64, 80, 96,112,128,144,160,0},
		{0,32,40,48,56,64,80,96,112,128,160,192,224,256,320,0}
	};
	static const long samp_freq[4] = {44100, 48000, 32000, 0};

	int cnt=0;
	int tc;

	while(cnt < bytes) {
		tc = bytes - cnt;
		if (tc > read_left)
			tc = read_left;

		if (header_mode)
			memcpy(&hdr_buffer[4-read_left], buffer, tc);

		buffer = (char *)buffer + tc;
		cnt += tc;
		read_left -= tc;

    if (read_left <= 0) {
      if (header_mode) {
        // We've got a header!  Process it...

        long hdr = *(long *)hdr_buffer;
        long samp_rate, framelen;

        if ((hdr & 0xE0FF) != 0xE0FF)
          throw MyError("MPEG audio sync error: try disabling MPEG audio time correction");

        samp_rate = samp_freq[(hdr >> 18) & 3];

        if (!((hdr >> 11) & 1)) {
          samp_rate /= 2;
          samples += 576;
        }
        else {
          samples += 1152;
        }

        if (!(hdr & 0x1000))
          samp_rate /= 2;

        framelen = (bitrates[(hdr >> 11) & 1][(hdr >> 20) & 15] * (((hdr >> 11) & 1) ? 144000 : 72000)) / samp_rate;

        if (hdr & 0x20000) ++framelen;

        // update statistics

        frame_bytes += framelen;

        // start skipping the remainder

        read_left = framelen - 4;
        header_mode = false;

      }
      else {

     // Done skipping frame data; collect the next header

        read_left = 4;
        header_mode = true;
      }
    }
	}
}

///////////////////////////////////////////////////////////////////////////
/*
long AudioTranslateVideoSubset(FrameSubset& dst, FrameSubset& src, long usPerFrame, WAVEFORMATEX *pwfex) {
	FrameSubsetNode *fsn = src.getFirstFrame();
	const __int64 i64usPerFrame = usPerFrame;
	const long lSampPerSec = pwfex->nSamplesPerSec;
	const long nBytesPerSec = pwfex->nAvgBytesPerSec;
	const int nBlockAlign = pwfex->nBlockAlign;
	long total = 0;

	// I like accuracy, so let's strive for accuracy.  Accumulate errors as we go;
	// use them to offset the starting points of subsequent segments, never being
	// more than 1/2 segment off.
	//
	// The conversion equation is in units of (1000000*nBlockAlign).

	__int64 nError = 0;
	__int64 nMultiplier = i64usPerFrame * nBytesPerSec;
	__int64 nDivisor = 1000000i64*nBlockAlign;
	__int64 nRound = nDivisor/2;
	long nTotalFramesAccumulated = 0;

	while(fsn) {
		long start, end;

		// Compute error.
		//
		// Ideally, we want the audio and video streams to be of the exact length.
		//
		// Audiolen = (videolen * usPerFrame * nBytesPerSec) / (1000000*nBlockAlign);

		nError = total*nDivisor - (nTotalFramesAccumulated * nMultiplier);

_RPT1(0,"nError = %I64d\n", nError);

		// Add a block.

		start = ((__int64)fsn->start * nMultiplier + nRound + nError) / nDivisor;
		end = ((__int64)(fsn->start + fsn->len) * nMultiplier + nRound) / nDivisor;

		nTotalFramesAccumulated += fsn->len;

		dst.addRange(start, end-start, false);

		total += end-start;

		fsn = src.getNextFrame(fsn);
	}

	return total;
}

AudioSubset::AudioSubset(AudioStream *src, FrameSubset *pfs, long usPerFrame, long preskew) : AudioStream() {
	memcpy(AllocFormat(src->GetFormatLen()), src->GetFormat(), src->GetFormatLen());

	SetSource(src);

#if 1

	long total = AudioTranslateVideoSubset(subset, *pfs, usPerFrame, src->GetFormat());

#else
	while(fsn) {
		long start, end;

//		start = ((__int64)fsn->start * i64usPerFrame * lSampPerSec) / 1000000;
//		end = ((__int64)(fsn->start + fsn->len) * i64usPerFrame * lSampPerSec) / 1000000;

		start = ((__int64)fsn->start * i64usPerFrame * nBytesPerSec + 500000*nBlockAlign) / (1000000*nBlockAlign);
		end = ((__int64)(fsn->start + fsn->len) * i64usPerFrame * nBytesPerSec + 500000*nBlockAlign) / (1000000*nBlockAlign);

		subset.addRange(start, end-start);

		total += end-start;

		fsn = pfs->getNextFrame(fsn);
	}
#endif

	subset.deleteRange(0, MulDiv(preskew, src->GetFormat()->nSamplesPerSec, 1000));

	stream_len = total;

	SetLimit(total);

	pfsnCur = subset.getFirstFrame();
	iOffset = 0;
	lSrcPos = 0;
	lSkipSize = sizeof skipBuffer / src->GetFormat()->nBlockAlign;
}

AudioSubset::~AudioSubset() {
}

long AudioSubset::_Read(void *buffer, long samples, long *lplBytes) {
	int offset, actual;

	if (!pfsnCur) {
		*lplBytes = 0;
		return 0;
	}

	while ((offset = pfsnCur->start - lSrcPos) > 0) {
		long t;

		if (source->Skip(offset)) {
			lSrcPos += offset;
			break;
		}

		if (offset > lSkipSize) offset = lSkipSize;

		actual = source->Read(skipBuffer, offset, &t);

		if (!actual) {
			*lplBytes = 0;
			return 0;
		}

		lSrcPos += actual;
	}

	if (samples > pfsnCur->len - iOffset)
		samples = pfsnCur->len - iOffset;

	samples = source->Read(buffer, samples, lplBytes);

	iOffset += samples;
	lSrcPos += samples;
	while (pfsnCur && iOffset >= pfsnCur->len) {
		iOffset -= pfsnCur->len;
		pfsnCur = subset.getNextFrame(pfsnCur);
	}

	return samples;
}

BOOL AudioSubset::_isEnd() {
	return !pfsnCur || source->isEnd();
}
*/

///////////////////////////////////////////////////////////////////////////
//
//	AudioAmplifier
//
///////////////////////////////////////////////////////////////////////////

static void amplify8(unsigned char *dst, int count, long lFactor) {
	long lBias = 0x8080 - 0x80*lFactor;

	if (count)
		do {
			int y = ((long)*dst++ * lFactor + lBias) >> 8;

			if (y<0) y=0; else if (y>255) y=255;

			dst[-1] = (unsigned char)y;
		} while(--count);
}

static void amplify16(signed short *dst, int count, long lFactor) {
	if (count)
		do {
			int y = ((long)*dst++ * lFactor + 0x80) >> 8;

			if (y<-0x7FFF) y=-0x7FFF; else if (y>0x7FFF) y=0x7FFF;

			dst[-1] = (signed short)y;
		} while(--count);
}

AudioStreamAmplifier::AudioStreamAmplifier(AudioStream *src, long _lFactor)
: lFactor(_lFactor) {

	WAVEFORMATEX *iFormat = src->GetFormat();
	WAVEFORMATEX *oFormat;

	memcpy(oFormat = AllocFormat(src->GetFormatLen()), iFormat, src->GetFormatLen());

	SetSource(src);
}

AudioStreamAmplifier::~AudioStreamAmplifier() {
}

long AudioStreamAmplifier::_Read(void *buffer, long samples, long *lplBytes) {
	long lActualSamples=0;
	long lBytes;

	lActualSamples = source->Read(buffer, samples, &lBytes);

	if (lActualSamples) {
		if (GetFormat()->wBitsPerSample > 8)
			amplify16((signed short *)buffer, lBytes/2, lFactor);
		else
			amplify8((unsigned char *)buffer, lBytes, lFactor);
	}

	if (lplBytes)
		*lplBytes = lBytes;

	return lActualSamples;
}

BOOL AudioStreamAmplifier::_isEnd() {
	return source->isEnd();
}

bool AudioStreamAmplifier::Skip(long samples) {
	return source->Skip(samples);
}
