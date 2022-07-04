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

#include <crtdbg.h>
#include <new>

#include "AVIReadHandler.h"
//#include "FastReadStream.h"
//#include "ProgressDialog.h"
#include "AVIIndex.h"
#include "Error.h"
#include "list.h"
#include "Fixes.h"
#include "File64.h"

#include "clip_info.h"
#include "../core/internal.h"
#include <cmath>
#include <cstdint>
#include "../core/strings.h"


#pragma warning(disable: 4706)    // assignment within conditional expression

// These two functions translate VirtualDub exceptions to Avisynth exceptions.

char exception_conversion_buffer[2048];

AvisynthError MyError(const char* fmt, ...) {
  va_list val;
  va_start(val, fmt);
  wvsprintf(exception_conversion_buffer, fmt, val);
  va_end(val);
  return AvisynthError(exception_conversion_buffer);
}

AvisynthError MyWin32Error(const char *format, DWORD err, ...) {
  char szError[128];
  char szTemp[256];

  FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
    0, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), szError, sizeof szError, NULL);

  if (szError[0]) {
    size_t l = strlen(szError);
    if (l>1 && szError[l-2] == '\r')
      szError[l-2] = 0;
    else if (szError[l-1] == '\n')
      szError[l-1] = 0;
  }

  va_list val;
  va_start(val, err);
  wvsprintf(szTemp, format, val);
  va_end(val);

  wsprintf(exception_conversion_buffer, szTemp, szError);
  return AvisynthError(exception_conversion_buffer);
}

// Copied over from VD_misc.cpp, as the only function being used from there
static bool isValidFOURCC(FOURCC fcc) {
	return isprint((unsigned char)(fcc>>24))
		&& isprint((unsigned char)(fcc>>16))
		&& isprint((unsigned char)(fcc>> 8))
		&& isprint((unsigned char)(fcc    ));
}

//#define STREAMING_DEBUG

// HACK!!!!

CRITICAL_SECTION g_diskcs;
bool g_disklockinited=false;


///////////////////////////////////////////

typedef __int64 QUADWORD;

// The following comes from the OpenDML 1.0 spec for extended AVI files

// bIndexType codes
//
#define AVI_INDEX_OF_INDEXES 0x00	// when each entry in aIndex
									// array points to an index chunk

#define AVI_INDEX_OF_CHUNKS 0x01	// when each entry in aIndex
									// array points to a chunk in the file

#define AVI_INDEX_IS_DATA	0x80	// when each entry is aIndex is
									// really the data

// bIndexSubtype codes for INDEX_OF_CHUNKS

#define AVI_INDEX_2FIELD	0x01	// when fields within frames
									// are also indexed
	struct _avisuperindex_entry {
		QUADWORD qwOffset;		// absolute file offset, offset 0 is
								// unused entry??
		DWORD dwSize;			// size of index chunk at this offset
		DWORD dwDuration;		// time span in stream ticks
	};
	struct _avistdindex_entry {
		DWORD dwOffset;			// qwBaseOffset + this is absolute file offset
		DWORD dwSize;			// bit 31 is set if this is NOT a keyframe
	};
	struct _avifieldindex_entry {
		DWORD	dwOffset;
		DWORD	dwSize;
		DWORD	dwOffsetField2;
	};

#pragma pack(push)
#pragma pack(2)

#pragma warning( push )
#pragma warning (disable: 4200) // nonstandard extension used : zero-sized array in struct/union

	typedef struct _avisuperindex_chunk {
	FOURCC fcc;					// ’ix##’
	DWORD cb;					// size of this structure
	WORD wLongsPerEntry;		// must be 4 (size of each entry in aIndex array)
	BYTE bIndexSubType;			// must be 0 or AVI_INDEX_2FIELD
	BYTE bIndexType;			// must be AVI_INDEX_OF_INDEXES
	DWORD nEntriesInUse;		// number of entries in aIndex array that
								// are used
	DWORD dwChunkId;			// ’##dc’ or ’##db’ or ’##wb’, etc
	DWORD dwReserved[3];		// must be 0
	struct _avisuperindex_entry aIndex[];
} AVISUPERINDEX, * PAVISUPERINDEX;

typedef struct _avistdindex_chunk {
	FOURCC fcc;					// ’ix##’
	DWORD cb;
	WORD wLongsPerEntry;		// must be sizeof(aIndex[0])/sizeof(DWORD)
	BYTE bIndexSubType;			// must be 0
	BYTE bIndexType;			// must be AVI_INDEX_OF_CHUNKS
	DWORD nEntriesInUse;		//
	DWORD dwChunkId;			// ’##dc’ or ’##db’ or ’##wb’ etc..
	QUADWORD qwBaseOffset;		// all dwOffsets in aIndex array are
								// relative to this
	DWORD dwReserved3;			// must be 0
	struct _avistdindex_entry aIndex[];
} AVISTDINDEX, * PAVISTDINDEX;

typedef struct _avifieldindex_chunk {
	FOURCC		fcc;
	DWORD		cb;
	WORD		wLongsPerEntry;
	BYTE		bIndexSubType;
	BYTE		bIndexType;
	DWORD		nEntriesInUse;
	DWORD		dwChunkId;
	QUADWORD	qwBaseOffset;
	DWORD		dwReserved3;
	struct	_avifieldindex_entry aIndex[];
} AVIFIELDINDEX, * PAVIFIELDINDEX;

#pragma warning( pop )

#pragma pack(pop)

///////////////////////////////////////////////////////////////////////////

IAVIReadStream::~IAVIReadStream() {
}

///////////////////////////////////////////////////////////////////////////

class AVIStreamNode;
class AVIReadHandler;

class AVIReadCache {
public:
	long cache_hit_bytes, cache_miss_bytes;
	int reads;

	AVIReadCache(int nlines, int nstream, AVIReadHandler *root, AVIStreamNode *psnData);
	~AVIReadCache();

	void ResetStatistics();
	bool WriteBegin(__int64 pos, long len);
	void Write(void *buffer, long len);
	void WriteEnd();
	long Read(void *dest, __int64 chunk_pos, __int64 pos, long len);

	long getMaxRead() {
		return (lines_max - 1) << 4;
	}

private:
	AVIStreamNode *psnData;
	__int64 (*buffer)[2];
	int lines_max, lines;
	long read_tail, write_tail, write_hdr;
	int write_offset;
	int stream;
	AVIReadHandler *source;
};

class AVIStreamNode : public ListNode2<AVIStreamNode> {
public:
	AVIStreamHeader_fixed	hdr;
	char					*pFormat;
	long					lFormatLen;
	AVIIndex				index;
	__int64					bytes;
	bool					keyframe_only;
	bool					was_VBR;
	double					bitrate_mean;
	double					bitrate_stddev;
	double					max_deviation;		// seconds
	int						handler_count;
	class AVIReadCache		*cache;
	int						streaming_count;
	__int64					stream_push_pos;
	__int64					stream_bytes;
	int						stream_pushes;
	long					length;
	long					frames;
	List2<class AVIReadStream>	listHandlers;

	AVIStreamNode();
	~AVIStreamNode();

	void FixVBRAudio();
};

AVIStreamNode::AVIStreamNode() {
	pFormat = NULL;
	bytes = 0;
	handler_count = 0;
	streaming_count = 0;

	stream_bytes = 0;
	stream_pushes = 0;
	cache = NULL;

	was_VBR = false;
}

AVIStreamNode::~AVIStreamNode() {
	delete [] pFormat;
	delete cache;
}

void AVIStreamNode::FixVBRAudio() {
	WAVEFORMATEX *pWaveFormat = (WAVEFORMATEX *)pFormat;

	// If this is an MP3 stream, undo the Nandub 1152 value.

	if (pWaveFormat->wFormatTag == 0x0055) {
		pWaveFormat->nBlockAlign = 1;
	}

	// Determine VBR statistics.

	was_VBR = true;

	const AVIIndexEntry2 *pent = index.index2Ptr();
	__int64 size_accum = 0;
	__int64 max_dev = 0;
	double size_sq_sum = 0.0;

	for(int i=0; i<frames; ++i) {
		unsigned size = pent[i].size & 0x7ffffffful;
		__int64 mean_center = (bytes * (2*i+1)) / (2*frames);
		__int64 dev = mean_center - (size_accum + (size>>1));

		if (dev<0)
			dev = -dev;

		if (dev > max_dev)
			max_dev = dev;

		size_accum += size;
		size_sq_sum += (double)size*size;
	}

	// I hate probability & sadistics.
	//
	// Var(X) = E(X2) - E(X)^2
	//		  = S(X2)/n - (S(x)/n)^2
	//		  = (n*S(X2) - S(X)^2)/n^2
	//
	// SD(x) = sqrt(n*S(X2) - S(X)^2) / n

	double frames_per_second = (double)hdr.dwRate / (double)hdr.dwScale;
	double sum1_bits = bytes * 8.0;
	double sum2_bits = size_sq_sum * 64.0;

	bitrate_mean		= (sum1_bits / frames) * frames_per_second;
	bitrate_stddev		= sqrt(frames * sum2_bits - sum1_bits * sum1_bits) / frames * frames_per_second;
	max_deviation		= (double)max_dev * 8.0 / bitrate_mean;

	// Assume that each audio block is of the same duration.

	hdr.dwRate			= (DWORD)(bitrate_mean/8.0 + 0.5);
	hdr.dwScale			= pWaveFormat->nBlockAlign;
	hdr.dwSampleSize	= pWaveFormat->nBlockAlign;
}

///////////////////////////////////////////////////////////////////////////

class AVIFileDesc : public ListNode2<AVIFileDesc> {
public:
	HANDLE		hFile;
	HANDLE		hFileUnbuffered;
	__int64		i64Size;
};

class AVIStreamNode;

class AVIReadHandler : public IAVIReadHandler, private File64 {
public:
	bool		fDisableFastIO;

	AVIReadHandler(const wchar_t *);
	AVIReadHandler(PAVIFILE);
	~AVIReadHandler();

	void AddRef();
	void Release();
	IAVIReadStream *GetStream(DWORD fccType, LONG lParam);
	void EnableFastIO(bool);
	bool isOptimizedForRealtime();
	bool isStreaming();
	bool isIndexFabricated();
	bool AppendFile(const wchar_t *pszFile);
	bool getSegmentHint(const char **ppszPath);

	void EnableStreaming(int stream);
	void DisableStreaming(int stream);
	void AdjustRealTime(bool fRealTime);
	bool Stream(AVIStreamNode *, int64_t pos);
	int64_t getStreamPtr();
	void FixCacheProblems(class AVIReadStream *);
	long ReadData(int stream, void *buffer, int64_t position, long len);

private:
//	enum { STREAM_SIZE = 65536 };
	enum { STREAM_SIZE = 1048576 };
	enum { STREAM_RT_SIZE = 65536 };
	enum { STREAM_BLOCK_SIZE = 4096 };

	IAvisynthClipInfo *pAvisynthClipInfo;
	PAVIFILE paf;
	int ref_count;
	int64_t		i64StreamPosition;
	int			streams;
	char		*streamBuffer;
	int			sbPosition;
	int			sbSize;
	long		fStreamsActive;
	int			nRealTime;
	int			nActiveStreamers;
	bool		fFakeIndex;
	int64_t		i64Size;
	int			nFiles, nCurrentFile;
	char *		pSegmentHint;

	// Whenever a file is aggressively recovered, do not allow streaming.

	bool		bAggressivelyRecovered;

	List2<AVIStreamNode>		listStreams;
	List2<AVIFileDesc>			listFiles;

	void		_construct(const wchar_t *pszFile_w);
	void		_parseFile(List2<AVIStreamNode>& streams);
	bool		_parseStreamHeader(List2<AVIStreamNode>& streams, DWORD dwLengthLeft, bool& bIndexDamaged);
	bool		_parseIndexBlock(List2<AVIStreamNode>& streams, int count, __int64);
	void		_parseExtendedIndexBlock(List2<AVIStreamNode>& streams, AVIStreamNode *pasn, __int64 fpos, DWORD dwLength);
	void		_destruct();

	char *		_StreamRead(long& bytes);
	void		_SelectFile(int file);

};

IAVIReadHandler *CreateAVIReadHandler(PAVIFILE paf) {
	return new AVIReadHandler(paf);
}

IAVIReadHandler *CreateAVIReadHandler(const wchar_t *pszFile_w) {
	return new AVIReadHandler(pszFile_w);
}

///////////////////////////////////////////////////////////////////////////

AVIReadCache::AVIReadCache(int nlines, int nstream, AVIReadHandler *root, AVIStreamNode *psnData) {
	buffer = new(std::nothrow) __int64[nlines][2];
	if (!buffer) throw MyMemoryError();

	this->psnData	= psnData;
	lines		= 0;
	lines_max	= nlines;
	read_tail	= 0;
	write_tail	= 0;
	stream		= nstream;
	source		= root;
	ResetStatistics();
}

AVIReadCache::~AVIReadCache() {
	delete[] buffer;
}

void AVIReadCache::ResetStatistics() {
	reads		= 0;
	cache_hit_bytes	= cache_miss_bytes = 0;
}

bool AVIReadCache::WriteBegin(__int64 pos, long len) {
	int needed;

	// delete lines as necessary to make room

	needed = 1 + (len+15)/16;

	if (needed > lines_max)
		return false;

	while(lines+needed > lines_max) {
		int siz = (int)((buffer[read_tail][1]+15)/16 + 1);
		read_tail += siz;
		lines -= siz;
		if (read_tail >= lines_max)
			read_tail -= lines_max;
	}

	// write in header

//	_RPT1(0,"\tbeginning write at line %ld\n", write_tail);

	write_hdr = write_tail;
	write_offset = 0;

	buffer[write_tail][0] = pos;
	buffer[write_tail][1] = 0;

	if (++write_tail >= lines_max)
		write_tail = 0;

	return true;
}

void AVIReadCache::Write(void *src, long len) {
	long dest;

	// copy in data

	buffer[write_hdr][1] += len;

	dest = write_tail + (len + write_offset + 15)/16;

	if (dest > lines_max) {
		int tc = (lines_max - write_tail)*16 - write_offset;

		memcpy((char *)&buffer[write_tail][0] + write_offset, src, tc);
		memcpy(&buffer[0][0], (char *)src + tc, len - tc);

		write_tail = (len-tc)/16;
		write_offset = (len-tc)&15;

	} else {
		memcpy((char *)&buffer[write_tail][0] + write_offset, src, len);
		write_tail += (len+write_offset)/16;
		if (write_tail >= lines_max)
			write_tail = 0;

		write_offset = (len+write_offset) & 15;
	}
}

void AVIReadCache::WriteEnd() {
	long cnt = (long)(1 + (buffer[write_hdr][1]+15)/16);
	lines += cnt;
	write_tail = write_hdr + cnt;

	if (write_tail >= lines_max)
		write_tail -= lines_max;

//	_RPT3(0,"\twrite complete -- header at line %d, size %ld, next line %ld\n", write_hdr, (long)buffer[write_hdr][1], write_tail);
}

// #pragma function(memcpy)

long AVIReadCache::Read(void *dest, __int64 chunk_pos, __int64 pos, long len) {
	long ptr;
	__int64 offset;

//	_RPT3(0,"Read request: chunk %16I64, pos %16I64d, %ld bytes\n", chunk_pos, pos, len);

	++reads;

	do {
		// scan buffer looking for a range that contains data

		ptr = read_tail;
		while(ptr != write_tail) {
			offset = pos - buffer[ptr][0];

//		_RPT4(0,"line %8d: pos %16I64d, len %ld bytes (%ld lines)\n", ptr, buffer[ptr][0], (long)buffer[ptr][1], (long)(buffer[ptr][1]+15)/16);

			if (offset>=0 && offset < buffer[ptr][1]) {
				long end;

				// cache hit

//				_RPT1(0, "cache hit (offset %" PRIu64 ")\n", chunk_pos);

				cache_hit_bytes += len;

				while (cache_hit_bytes > 16777216) {
					cache_miss_bytes >>= 1;
					cache_hit_bytes >>= 1;
				}

				if (len > (long)(buffer[ptr][1]*16 - offset))
					len = (long)(buffer[ptr][1]*16 - offset);

				ptr += 1+((long)offset>>4);
				if (ptr >= lines_max)
					ptr -= lines_max;

				end = ptr + ((len+((long)offset&15)+15)>>4);

				if (end > lines_max) {
					long tc = (lines_max - ptr)*16 - ((long)offset&15);
					memcpy(dest, (char *)&buffer[ptr][0] + ((long)offset&15), tc);
					memcpy((char *)dest + tc, (char *)&buffer[0][0], len-tc);
				} else {
					memcpy(dest, (char *)&buffer[ptr][0] + ((long)offset&15), len);
				}

				return len;
			}

//		_RPT4(0,"[x] line %8d: pos %16I64d, len %ld bytes (%ld lines)\n", ptr, buffer[ptr][0], (long)buffer[ptr][1], (long)(buffer[ptr][1]+15)/16);
			ptr += (long)(1+(buffer[ptr][1] + 15)/16);

			if (ptr >= lines_max)
				ptr -= lines_max;
		}

		if (source->getStreamPtr() > chunk_pos)
			break;

	} while(source->Stream(psnData, chunk_pos));

//	OutputDebugString("cache miss\n");
//	_RPT1(0, "cache miss (offset %" PRIu64 ")\n", chunk_pos);

	cache_miss_bytes += len;

	while (cache_miss_bytes > 16777216) {
		cache_miss_bytes >>= 1;
		cache_hit_bytes >>= 1;
	}

	return source->ReadData(stream, dest, pos, len);
}

///////////////////////////////////////////////////////////////////////////

class AVIReadTunnelStream : public IAVIReadStream {
public:
	AVIReadTunnelStream(AVIReadHandler *, PAVISTREAM, IAvisynthClipInfo *pClipInfo);
	~AVIReadTunnelStream();

	HRESULT BeginStreaming(long lStart, long lEnd, long lRate);
	HRESULT EndStreaming();
	HRESULT Info(AVISTREAMINFO *pasi, long lSize);
	bool IsKeyFrame(long lFrame);
	HRESULT Read(long lStart, long lSamples, void *lpBuffer, long cbBuffer, long *plBytes, long *plSamples);
	long Start();
	long End();
	long PrevKeyFrame(long lFrame);
	long NextKeyFrame(long lFrame);
	long NearestKeyFrame(long lFrame);
	HRESULT FormatSize(long lFrame, long *plSize);
	HRESULT ReadFormat(long lFrame, void *pFormat, long *plSize);
	bool isStreaming();
	bool isKeyframeOnly();
	bool getVBRInfo(double& bitrate_mean, double& bitrate_stddev, double& maxdev) { return false; }

private:
	IAvisynthClipInfo *const pAvisynthClipInfo;
	AVIReadHandler *const parent;
	const PAVISTREAM pas;
};

///////////////////////////////////////////////////////////////////////////

AVIReadTunnelStream::AVIReadTunnelStream(AVIReadHandler *_parent, PAVISTREAM _pas, IAvisynthClipInfo *pClipInfo)
: pAvisynthClipInfo(pClipInfo)
, parent(_parent)
, pas(_pas)
{
	parent->AddRef();
}

AVIReadTunnelStream::~AVIReadTunnelStream() {
	pas->Release();
	parent->Release();
}

HRESULT AVIReadTunnelStream::BeginStreaming(long lStart, long lEnd, long lRate) {
	return AVIStreamBeginStreaming(pas, lStart, lEnd, lRate);
}

HRESULT AVIReadTunnelStream::EndStreaming() {
	return AVIStreamEndStreaming(pas);
}

HRESULT AVIReadTunnelStream::Info(AVISTREAMINFO *pasi, long lSize) {
	return AVIStreamInfo(pas, pasi, lSize);
}

bool AVIReadTunnelStream::IsKeyFrame(long lFrame) {
	return !!AVIStreamIsKeyFrame(pas, lFrame);
}

HRESULT AVIReadTunnelStream::Read(long lStart, long lSamples, void *lpBuffer, long cbBuffer, long *plBytes, long *plSamples) {
	HRESULT hr;

	hr = AVIStreamRead(pas, lStart, lSamples, lpBuffer, cbBuffer, plBytes, plSamples);

	if (pAvisynthClipInfo) {
		const char *pszErr;

		if (pAvisynthClipInfo->GetError(&pszErr))
			throw MyError("Avisynth read error:\n%s", pszErr);
	}

	return hr;
}

long AVIReadTunnelStream::Start() {
	return AVIStreamStart(pas);
}

long AVIReadTunnelStream::End() {
	return AVIStreamEnd(pas);
}

long AVIReadTunnelStream::PrevKeyFrame(long lFrame) {
	return AVIStreamPrevKeyFrame(pas, lFrame);
}

long AVIReadTunnelStream::NextKeyFrame(long lFrame) {
	return AVIStreamNextKeyFrame(pas, lFrame);
}

long AVIReadTunnelStream::NearestKeyFrame(long lFrame) {
	return AVIStreamNearestKeyFrame(pas, lFrame);
}

HRESULT AVIReadTunnelStream::FormatSize(long lFrame, long *plSize) {
	return AVIStreamFormatSize(pas, lFrame, plSize);
}

HRESULT AVIReadTunnelStream::ReadFormat(long lFrame, void *pFormat, long *plSize) {
	return AVIStreamReadFormat(pas, lFrame, pFormat, plSize);
}

bool AVIReadTunnelStream::isStreaming() {
	return false;
}

bool AVIReadTunnelStream::isKeyframeOnly() {
   return false;
}

///////////////////////////////////////////////////////////////////////////

class AVIReadStream : public IAVIReadStream, public ListNode2<AVIReadStream> {
	friend AVIReadHandler;

public:
	AVIReadStream(AVIReadHandler *, AVIStreamNode *, int);
	~AVIReadStream();

	HRESULT BeginStreaming(long lStart, long lEnd, long lRate);
	HRESULT EndStreaming();
	HRESULT Info(AVISTREAMINFO *pasi, long lSize);
	bool IsKeyFrame(long lFrame);
	HRESULT Read(long lStart, long lSamples, void *lpBuffer, long cbBuffer, long *plBytes, long *plSamples);
	long Start();
	long End();
	long PrevKeyFrame(long lFrame);
	long NextKeyFrame(long lFrame);
	long NearestKeyFrame(long lFrame);
	HRESULT FormatSize(long lFrame, long *plSize);
	HRESULT ReadFormat(long lFrame, void *pFormat, long *plSize);
	bool isStreaming();
	bool isKeyframeOnly();
	void Reinit();
	bool getVBRInfo(double& bitrate_mean, double& bitrate_stddev, double& maxdev);

private:
	AVIReadHandler *parent;
	AVIStreamNode *psnData;
	AVIIndexEntry2 *pIndex;
	//AVIReadCache *rCache;
	long& length;
	long& frames;
	long sampsize;
	int streamno;
	bool fStreamingEnabled;
	bool fStreamingActive;
	int iStreamTrackCount;
	long lStreamTrackValue;
	long lStreamTrackInterval;
	bool fRealTime;

	__int64		i64CachedPosition;
	AVIIndexEntry2	*pCachedEntry;

};

///////////////////////////////////////////////////////////////////////////

AVIReadStream::AVIReadStream(AVIReadHandler *parent, AVIStreamNode *psnData, int streamno)
:length(psnData->length)
,frames(psnData->frames)
{
	this->parent = parent;
	this->psnData = psnData;
	this->streamno = streamno;

	fStreamingEnabled = false;
	fStreamingActive = false;
	fRealTime = false;

	parent->AddRef();

	pIndex = psnData->index.index2Ptr();
	sampsize = psnData->hdr.dwSampleSize;

	// Hack to imitate Microsoft's parser.  It seems to ignore this value
	// for audio streams.

	if (psnData->hdr.fccType == streamtypeAUDIO)
		sampsize = ((WAVEFORMATEX *)psnData->pFormat)->nBlockAlign;

	if (sampsize) {
		i64CachedPosition = 0;
		pCachedEntry = pIndex;
	}

	psnData->listHandlers.AddTail(this);
}

AVIReadStream::~AVIReadStream() {
	EndStreaming();
	parent->Release();
	Remove();
}

void AVIReadStream::Reinit() {
	pIndex = psnData->index.index2Ptr();
	i64CachedPosition = 0;
	pCachedEntry = pIndex;
}

HRESULT AVIReadStream::BeginStreaming(long lStart, long lEnd, long lRate) {
	if (fStreamingEnabled)
		return 0;

//	OutputDebugString(lRate>1500 ? "starting: fast" : "starting: slow");

	if (lRate <= 1500) {
		parent->AdjustRealTime(true);
		fRealTime = true;
	} else
		fRealTime = false;

	if (parent->fDisableFastIO)
		return 0;

	if (!psnData->streaming_count) {
//		if (!(psnData->cache = new AVIReadCache(psnData->hdr.fccType == 'sdiv' ? 65536 : 16384, streamno, parent, psnData)))
//			return AVIERR_MEMORY;

		psnData->stream_bytes = 0;
		psnData->stream_pushes = 0;
		psnData->stream_push_pos = 0;
	}
	++psnData->streaming_count;

	fStreamingEnabled = true;
	fStreamingActive = false;
	iStreamTrackCount = 0;
	lStreamTrackValue = -1;
	lStreamTrackInterval = -1;
	return 0;
}

HRESULT AVIReadStream::EndStreaming() {
	if (!fStreamingEnabled)
		return 0;

	if (fRealTime)
		parent->AdjustRealTime(false);

	if (fStreamingActive)
		parent->DisableStreaming(streamno);

	fStreamingEnabled = false;
	fStreamingActive = false;

	if (!--psnData->streaming_count) {
		delete psnData->cache;
		psnData->cache = NULL;
	}
	return 0;
}

HRESULT AVIReadStream::Info(AVISTREAMINFO *pasi, long lSize) {
	AVISTREAMINFO asi;

	memset(&asi, 0, sizeof asi);

	asi.fccType				= psnData->hdr.fccType;
	asi.fccHandler			= psnData->hdr.fccHandler;
	asi.dwFlags				= psnData->hdr.dwFlags;
	asi.wPriority			= psnData->hdr.wPriority;
	asi.wLanguage			= psnData->hdr.wLanguage;
	asi.dwScale				= psnData->hdr.dwScale;
	asi.dwRate				= psnData->hdr.dwRate;
	asi.dwStart				= psnData->hdr.dwStart;
	asi.dwLength			= psnData->hdr.dwLength;
	asi.dwInitialFrames		= psnData->hdr.dwInitialFrames;
	asi.dwSuggestedBufferSize = psnData->hdr.dwSuggestedBufferSize;
	asi.dwQuality			= psnData->hdr.dwQuality;
	asi.dwSampleSize		= psnData->hdr.dwSampleSize;
	asi.rcFrame.top			= psnData->hdr.rcFrame.top;
	asi.rcFrame.left		= psnData->hdr.rcFrame.left;
	asi.rcFrame.right		= psnData->hdr.rcFrame.right;
	asi.rcFrame.bottom		= psnData->hdr.rcFrame.bottom;

	if (lSize < sizeof asi)
		memcpy(pasi, &asi, lSize);
	else {
		memcpy(pasi, &asi, sizeof asi);
		memset((char *)pasi + sizeof asi, 0, lSize - sizeof asi);
	}

	return 0;
}

bool AVIReadStream::IsKeyFrame(long lFrame) {
	if (sampsize)
		return true;
	else {
		if (lFrame < 0 || lFrame >= length)
			return false;

		return !(pIndex[lFrame].size & 0x80000000);
	}
}

HRESULT AVIReadStream::Read(long lStart, long lSamples, void *lpBuffer, long cbBuffer, long *plBytes, long *plSamples) {
	long lActual;

	if (lStart < 0 || lStart >= length || (lSamples <= 0 && lSamples != AVISTREAMREAD_CONVENIENT)) {
		// umm... dummy!  can't read outside of stream!

		if (plBytes) *plBytes = 0;
		if (plSamples) *plSamples = 0;

		return 0;
	}

	// blocked or discrete?

	if (sampsize) {
		AVIIndexEntry2 *avie2, *avie2_limit = pIndex+frames;
		__int64 byte_off = (__int64)lStart * sampsize;
		__int64 bytecnt;
		__int64 actual_bytes=0;
		__int64 block_pos;

		// too small to hold a sample?

		if (lpBuffer && cbBuffer < sampsize) {
			if (plBytes) *plBytes = sampsize * lSamples;
			if (plSamples) *plSamples = lSamples;

			return AVIERR_BUFFERTOOSMALL;
		}

		// find the frame that has the starting sample -- try and work
		// from our last position to save time

		if (byte_off >= i64CachedPosition) {
			block_pos = i64CachedPosition;
			avie2 = pCachedEntry;
			byte_off -= block_pos;
		} else {
			block_pos = 0;
			avie2 = pIndex;
		}

		while(byte_off >= (avie2->size & 0x7FFFFFFF)) {
			byte_off -= (avie2->size & 0x7FFFFFFF);
			block_pos += (avie2->size & 0x7FFFFFFF);
			++avie2;
		}

		pCachedEntry = avie2;
		i64CachedPosition = block_pos;

		// Client too lazy to specify a size?

		if (lSamples == AVISTREAMREAD_CONVENIENT) {
			lSamples = ((avie2->size & 0x7FFFFFFF) - (long)byte_off) / sampsize;

			if (!lSamples && avie2+1 < avie2_limit)
				lSamples = ((avie2[0].size & 0x7FFFFFFF) + (avie2[1].size & 0x7FFFFFFF) - (long)byte_off) / sampsize;

			if (lSamples < 0)
				lSamples = 1;
		}

		// trim down sample count

		if (lpBuffer && lSamples > cbBuffer / sampsize)
			lSamples = cbBuffer / sampsize;

		if (lStart+lSamples > length)
			lSamples = length - lStart;

		bytecnt = lSamples * sampsize;

		// begin reading frames from this point on

		if (lpBuffer) {
			// detect streaming

			if (fStreamingEnabled) {

				// We consider the client to be streaming if we detect at least
				// 3 consecutive accesses

				if (lStart == lStreamTrackValue) {
					++iStreamTrackCount;

					if (iStreamTrackCount >= 15) {

						__int64 streamptr = parent->getStreamPtr();
						__int64 fptrdiff = streamptr - avie2->pos;

						if (!parent->isStreaming() || streamptr<0 || (fptrdiff<4194304 && fptrdiff>-4194304)) {
							if (!psnData->cache)
								psnData->cache = new AVIReadCache(psnData->hdr.fccType == MAKEFOURCC('v','i','d','s') ? 131072 : 16384, streamno, parent, psnData);
							else
								psnData->cache->ResetStatistics();

							if (!fStreamingActive) {
								fStreamingActive = true;
								parent->EnableStreaming(streamno);
							}

#ifdef STREAMING_DEBUG
							OutputDebugString("[a] streaming enabled\n");
#endif
						}
					} else {
#ifdef STREAMING_DEBUG
						OutputDebugString("[a] streaming detected\n");
#endif
					}
				} else {
#ifdef STREAMING_DEBUG
					OutputDebugString("[a] streaming disabled\n");
#endif
					iStreamTrackCount = 0;

					if (fStreamingActive) {
						fStreamingActive = false;
						parent->DisableStreaming(streamno);
					}
				}
			}

			while(bytecnt > 0) {
				long tc;

				tc = (avie2->size & 0x7FFFFFFF) - (long)byte_off;
				if (tc > bytecnt)
					tc = (long)bytecnt;

				if (psnData->cache && fStreamingActive && tc < psnData->cache->getMaxRead()) {
//OutputDebugString("[a] attempting cached read\n");
					lActual = psnData->cache->Read(lpBuffer, avie2->pos, avie2->pos + byte_off + 8, tc);
					psnData->stream_bytes += lActual;
				} else
					lActual = parent->ReadData(streamno, lpBuffer, avie2->pos + byte_off + 8, tc);

				if (lActual < 0)
					break;

				actual_bytes += lActual;
				++avie2;
				byte_off = 0;

				if (lActual < tc)
					break;

				bytecnt -= tc;
				lpBuffer = (char *)lpBuffer + tc;
			}

			if (actual_bytes < sampsize) {
				if (plBytes) *plBytes = 0;
				if (plSamples) *plSamples = 0;
				return AVIERR_FILEREAD;
			}

			actual_bytes -= actual_bytes % sampsize;

			if (plBytes) *plBytes = (long)actual_bytes;
			if (plSamples) *plSamples = (long)actual_bytes / sampsize;

			lStreamTrackValue = lStart + (long)actual_bytes / sampsize;

		} else {
			if (plBytes) *plBytes = (long)bytecnt;
			if (plSamples) *plSamples = lSamples;
		}

	} else {
		AVIIndexEntry2 *avie2 = &pIndex[lStart];

		if (lpBuffer && (avie2->size & 0x7FFFFFFF) > cbBuffer) {
			if (plBytes) *plBytes = avie2->size & 0x7FFFFFFF;
			if (plSamples) *plSamples = 1;

			return AVIERR_BUFFERTOOSMALL;
		}

		if (lpBuffer) {

			// detect streaming

			if (fStreamingEnabled && lStart != lStreamTrackValue) {
				if (lStreamTrackValue>=0 && lStart-lStreamTrackValue == lStreamTrackInterval) {
					if (++iStreamTrackCount >= 15) {

						__int64 streamptr = parent->getStreamPtr();
						__int64 fptrdiff = streamptr - avie2->pos;

						if (!parent->isStreaming() || streamptr<0 || (fptrdiff<4194304 && fptrdiff>-4194304)) {
							if (!psnData->cache)
								psnData->cache = new AVIReadCache(psnData->hdr.fccType == MAKEFOURCC('v','i','d','s') ? 131072 : 16384, streamno, parent, psnData);
							else
								psnData->cache->ResetStatistics();

							if (!fStreamingActive) {
								fStreamingActive = true;
								parent->EnableStreaming(streamno);
							}

#ifdef STREAMING_DEBUG
							OutputDebugString("[v] streaming activated\n");
#endif
						}
					} else {
#ifdef STREAMING_DEBUG
						OutputDebugString("[v] streaming detected\n");
#endif
					}
				} else {
					iStreamTrackCount = 0;
#ifdef STREAMING_DEBUG
					OutputDebugString("[v] streaming disabled\n");
#endif
					if (lStreamTrackValue>=0 && lStart > lStreamTrackValue) {
						lStreamTrackInterval = lStart - lStreamTrackValue;
					} else
						lStreamTrackInterval = -1;

					if (fStreamingActive) {
						fStreamingActive = false;
						parent->DisableStreaming(streamno);
					}
				}

				lStreamTrackValue = lStart;
			}

			// read data

			long size = avie2->size & 0x7FFFFFFF;

			if (psnData->cache && fStreamingActive && size < psnData->cache->getMaxRead()) {
//OutputDebugString("[v] attempting cached read\n");
				lActual = psnData->cache->Read(lpBuffer, avie2->pos, avie2->pos + 8, size);
				psnData->stream_bytes += lActual;
			} else
				lActual = parent->ReadData(streamno, lpBuffer, avie2->pos+8, size);

			if (lActual != size) {
				if (plBytes) *plBytes = 0;
				if (plSamples) *plSamples = 0;
				return AVIERR_FILEREAD;
			}
		}

		if (plBytes) *plBytes = avie2->size & 0x7FFFFFFF;
		if (plSamples) *plSamples = 1;
	}

	if (psnData->cache && fStreamingActive) {

		// Are we experiencing a high rate of cache misses?

		if (psnData->cache->cache_miss_bytes*2 > psnData->cache->cache_hit_bytes && psnData->cache->reads > 50) {

			// sh*t, notify the parent that we have cache misses so it can check which stream is
			// screwing up, and disable streaming on feeds that are too far off

			parent->FixCacheProblems(this);
			iStreamTrackCount = 0;
		}
	}/* else if (fStreamingEnabled) {

		// hmm... our cache got killed!

		iStreamTrackCount = 0;

	}*/

	return 0;
}

long AVIReadStream::Start() {
	return 0;
}

long AVIReadStream::End() {
	return length;
}

long AVIReadStream::PrevKeyFrame(long lFrame) {
	if (sampsize)
		return lFrame>0 ? lFrame-1 : -1;

	if (lFrame < 0)
		return -1;

	if (lFrame >= length)
		lFrame = length;

	while(--lFrame > 0)
		if (!(pIndex[lFrame].size & 0x80000000))
			return lFrame;

	return -1;
}

long AVIReadStream::NextKeyFrame(long lFrame) {
	if (sampsize)
		return lFrame<length ? lFrame+1 : -1;

	if (lFrame < 0)
		return 0;

	if (lFrame >= length)
		return -1;

	while(++lFrame < length)
		if (!(pIndex[lFrame].size & 0x80000000))
			return lFrame;

	return -1;
}

long AVIReadStream::NearestKeyFrame(long lFrame) {
	long lprev;

	if (sampsize)
		return lFrame;

	if (IsKeyFrame(lFrame))
		return lFrame;

	lprev = PrevKeyFrame(lFrame);

	if (lprev < 0)
		return 0;
	else
		return lprev;
}

HRESULT AVIReadStream::FormatSize(long lFrame, long *plSize) {
	*plSize = psnData->lFormatLen;
	return 0;
}

HRESULT AVIReadStream::ReadFormat(long lFrame, void *pFormat, long *plSize) {
	if (!pFormat) {
		*plSize = psnData->lFormatLen;
		return 0;
	}

	if (*plSize < psnData->lFormatLen) {
		memcpy(pFormat, psnData->pFormat, *plSize);
	} else {
		memcpy(pFormat, psnData->pFormat, psnData->lFormatLen);
		*plSize = psnData->lFormatLen;
	}

	return 0;
}

bool AVIReadStream::isStreaming() {
	return psnData->cache && fStreamingActive;
}

bool AVIReadStream::isKeyframeOnly() {
   return psnData->keyframe_only;
}

bool AVIReadStream::getVBRInfo(double& bitrate_mean, double& bitrate_stddev, double& maxdev) {
	if (psnData->was_VBR) {
		bitrate_mean = psnData->bitrate_mean;
		bitrate_stddev = psnData->bitrate_stddev;
		maxdev = psnData->max_deviation;
		return true;
	}

	return false;
}

///////////////////////////////////////////////////////////////////////////

AVIReadHandler::AVIReadHandler(const wchar_t *s)
: pAvisynthClipInfo(0)
, bAggressivelyRecovered(false)
{
	this->hFile = INVALID_HANDLE_VALUE;
	this->hFileUnbuffered = INVALID_HANDLE_VALUE;
	this->paf = NULL;
	ref_count = 1;
	streams=0;
	fStreamsActive = 0;
	fDisableFastIO = false;
	streamBuffer = NULL;
	nRealTime = 0;
	nActiveStreamers = 0;
	fFakeIndex = false;
	nFiles = 1;
	nCurrentFile = 0;
	pSegmentHint = NULL;

	if (!g_disklockinited) {
		g_disklockinited=true;
		InitializeCriticalSection(&g_diskcs);
	}

	_construct(s);
}

AVIReadHandler::AVIReadHandler(PAVIFILE paf) {
	this->hFile = INVALID_HANDLE_VALUE;
	this->hFileUnbuffered = INVALID_HANDLE_VALUE;
	this->paf = paf;
	ref_count = 1;
	streams=0;
	streamBuffer = NULL;
	pSegmentHint = NULL;
	fFakeIndex = false;

	if (FAILED(paf->QueryInterface(IID_IAvisynthClipInfo, (void **)&pAvisynthClipInfo)))
		pAvisynthClipInfo = NULL;
	else {
		const char *s;

		if (pAvisynthClipInfo->GetError(&s)) {
			char* msg = _strdup(s);
			pAvisynthClipInfo->Release();
			paf->Release();
			try {
				throw MyError("Avisynth open failure:%s\n", msg);
			} catch (...) {
				free(msg);
				throw;
			}
		}
	}
}

AVIReadHandler::~AVIReadHandler() {
	_destruct();
}

void AVIReadHandler::_construct(const wchar_t *pszFile_w) {

	try {
		AVIFileDesc *pDesc;

		// open file

		hFile = CreateFileW(pszFile_w, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, NULL);

		if (INVALID_HANDLE_VALUE == hFile)
			throw MyWin32Error("Couldn't open %s: %%s", GetLastError(), WideCharToAnsi(pszFile_w).get());

		hFileUnbuffered = CreateFileW(
				pszFile_w,
				GENERIC_READ,
				FILE_SHARE_READ,
				NULL,
				OPEN_EXISTING,
				FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING,
				NULL
			);

		i64FilePosition = 0;

		// recursively parse file

		_parseFile(listStreams);

		// Create first link

		if (!(pDesc = new(std::nothrow) AVIFileDesc))
			throw MyMemoryError();

		pDesc->hFile			= hFile;
		pDesc->hFileUnbuffered	= hFileUnbuffered;
		pDesc->i64Size			= i64Size = _sizeFile();

		listFiles.AddHead(pDesc);

	} catch(...) {
		_destruct();
		throw;
	}
}

bool AVIReadHandler::AppendFile(const wchar_t *pszFile_w) {
	List2<AVIStreamNode> newstreams;
	AVIStreamNode *pasn_old, *pasn_new, *pasn_old_next=NULL, *pasn_new_next=NULL;
	AVIFileDesc *pDesc;

	nCurrentFile = -1;

	// open file

	hFile = CreateFileW(pszFile_w, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, NULL);

	if (INVALID_HANDLE_VALUE == hFile)
		throw MyWin32Error("Couldn't open %s: %%s", GetLastError(), WideCharToAnsi(pszFile_w).get());

	hFileUnbuffered = CreateFileW(
			pszFile_w,
			GENERIC_READ,
			FILE_SHARE_READ,
			NULL,
			OPEN_EXISTING,
			FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING,
			NULL
		);

	try {
		_parseFile(newstreams);

		pasn_old = listStreams.AtHead();
		pasn_new = newstreams.AtHead();

		while(!!(pasn_old_next = pasn_old->NextFromHead()) & !!(pasn_new_next = pasn_new->NextFromHead())) {
			const char *szPrefix;

			switch(pasn_old->hdr.fccType) {
			case streamtypeAUDIO:	szPrefix = "Cannot append segment: The audio streams "; break;
			case streamtypeVIDEO:	szPrefix = "Cannot append segment: The video streams "; break;
			case MAKEFOURCC('i', 'v', 'a', 's'):			szPrefix = "Cannot append segment: The DV streams "; break;
			default:				szPrefix = ""; break;
			}

			// If it's not an audio or video stream, why do we care?

			if (szPrefix) {
				// allow ivas as a synonym for vids

				FOURCC fccOld = pasn_old->hdr.fccType;
				FOURCC fccNew = pasn_new->hdr.fccType;

				if (fccOld != fccNew)
					throw MyError("Cannot append segment: The stream types do not match.");

//				if (pasn_old->hdr.fccHandler != pasn_new->hdr.fccHandler)
//					throw MyError("%suse incompatible compression types.", szPrefix);

				// A/B ?= C/D ==> AD ?= BC

				if ((__int64)pasn_old->hdr.dwScale * pasn_new->hdr.dwRate != (__int64)pasn_new->hdr.dwScale * pasn_old->hdr.dwRate)
					throw MyError("%shave different sampling rates (%.5f vs. %.5f)"
							,szPrefix
							,(double)pasn_old->hdr.dwRate / pasn_old->hdr.dwScale
							,(double)pasn_new->hdr.dwRate / pasn_new->hdr.dwScale
							);

				if (pasn_old->hdr.dwSampleSize != pasn_new->hdr.dwSampleSize)
					throw MyError("%shave different block sizes (%d vs %d).", szPrefix, pasn_old->hdr.dwSampleSize, pasn_new->hdr.dwSampleSize);

				// I hate PCMWAVEFORMAT.

				bool fOk = pasn_old->lFormatLen == pasn_new->lFormatLen && !memcmp(pasn_old->pFormat, pasn_new->pFormat, pasn_old->lFormatLen);

				if (pasn_old->hdr.fccType == streamtypeAUDIO)
					if (((WAVEFORMATEX *)pasn_old->pFormat)->wFormatTag == WAVE_FORMAT_PCM
						&& ((WAVEFORMATEX *)pasn_new->pFormat)->wFormatTag == WAVE_FORMAT_PCM)
							fOk = !memcmp(pasn_old->pFormat, pasn_new->pFormat, sizeof(PCMWAVEFORMAT));

				if (!fOk)
					throw MyError("%shave different data formats.", szPrefix);
			}

			pasn_old = pasn_old_next;
			pasn_new = pasn_new_next;
		}

		if (pasn_old_next || pasn_new_next)
			throw MyError("Cannot append segment: The segment has a different number of streams.");

		if (!(pDesc = new(std::nothrow) AVIFileDesc))
			throw MyMemoryError();

		pDesc->hFile			= hFile;
		pDesc->hFileUnbuffered	= hFileUnbuffered;
		pDesc->i64Size			= _sizeFile();
	} catch(const AvisynthError&) {
		while((pasn_new = newstreams.RemoveHead()))
			delete pasn_new;

		CloseHandle(hFile);
		if (hFileUnbuffered != INVALID_HANDLE_VALUE)
			CloseHandle(hFileUnbuffered);

		throw;
	}

	// Accept segment; begin merging process.

	pasn_old = listStreams.AtHead();

	while((pasn_old_next = pasn_old->NextFromHead())) {
		pasn_new = newstreams.RemoveHead();

		// Fix up header.

		pasn_old->hdr.dwLength	+= pasn_new->hdr.dwLength;

		if (pasn_new->hdr.dwSuggestedBufferSize > pasn_old->hdr.dwSuggestedBufferSize)
			pasn_old->hdr.dwSuggestedBufferSize = pasn_new->hdr.dwSuggestedBufferSize;

		pasn_old->bytes		+= pasn_new->bytes;
		pasn_old->frames	+= pasn_new->frames;
		pasn_old->length	+= pasn_new->length;

		// Merge indices.

		int oldlen = pasn_old->index.indexLen();
		AVIIndexEntry2 *idx_old = pasn_old->index.takeIndex2();
		AVIIndexEntry2 *idx_new = pasn_new->index.index2Ptr();
		int i;

		pasn_old->index.clear();

		for(i=0; i<oldlen; i++) {
			idx_old[i].size ^= 0x80000000;
			pasn_old->index.add(&idx_old[i]);
		}

		delete[] idx_old;

		for(i=pasn_new->index.indexLen(); i; i--) {
			idx_new->size ^= 0x80000000;
			idx_new->pos += (__int64)nFiles << 48;
			pasn_old->index.add(idx_new++);
		}

		pasn_old->index.makeIndex2();

		// Notify all handlers.

		AVIReadStream *pStream, *pStreamNext;

		pStream = pasn_old->listHandlers.AtHead();
		while((pStreamNext = pStream->NextFromHead())) {
			pStream->Reinit();

			pStream = pStreamNext;
		}

		// Next!

		pasn_old = pasn_old_next;
		delete pasn_new;
	}

	++nFiles;
	listFiles.AddTail(pDesc);

	return true;
}

void AVIReadHandler::_parseFile(List2<AVIStreamNode>& streamlist) {
	FOURCC fccType;
	DWORD dwLength;
	bool index_found = false;
	bool fAcceptIndexOnly = true;
	bool hyperindexed = false;
	bool bScanRequired = false;
	AVIStreamNode *pasn, *pasn_next;

	__int64	i64ChunkMoviPos = 0;
	DWORD	dwChunkMoviLength = 0;

	if (!_readChunkHeader(fccType, dwLength))
		throw MyError("Invalid AVI file: File is less than 8 bytes");

	if (fccType != FOURCC_RIFF)
		throw MyError("Invalid AVI file: Not a RIFF file");

	// If the RIFF header is <4 bytes, assume it was an improperly closed
	// file.

	_readFile2(&fccType, 4);

	if (fccType != MAKEFOURCC('A', 'V', 'I', ' '))
		throw MyError("Invalid AVI file: RIFF type is not 'AVI'");

	// Aggressive mode recovery does extensive validation and searching to attempt to
	// recover an AVI.  It is significantly slower, however.  We place the flag here
	// so we can set it if something dreadfully wrong is with the AVI.

	bool bAggressive = false;

	// begin parsing chunks

	while(_readChunkHeader(fccType, dwLength)) {

//		_RPT4(0,"%08I64x %08I64x Chunk '%-4s', length %08lx\n", _posFile()+dwLengthLeft, _posFile(), &fccType, dwLength);

		// Invalid FCCs are bad.  If we find one, set the aggressive flag so if a scan
		// starts, we aggressively search for and validate data.

		if (!isValidFOURCC(fccType)) {
			bAggressive = true;
			break;
		}

		switch(fccType) {
		case FOURCC_LIST:
			_readFile2(&fccType, 4);

			// Some idiot Premiere plugin is writing AVI files with an invalid
			// size field in the LIST/hdrl chunk.

			if (dwLength<4 && fccType != listtypeAVIHEADER)
				throw MyError("Invalid AVI file: LIST chunk <4 bytes");

			if (dwLength < 4)
				dwLength = 0;
			else
				dwLength -= 4;

//			_RPT1(0,"\tList type '%-4s'\n", &fccType);

			switch(fccType) {
      case MAKEFOURCC('m','o','v','i'):

				if (dwLength < 8) {
					i64ChunkMoviPos = _posFile();
					dwChunkMoviLength = 0xFFFFFFF0;
					dwLength = 0;
				} else {
					i64ChunkMoviPos = _posFile();
					dwChunkMoviLength = dwLength;
				}

				if (fAcceptIndexOnly)
					goto terminate_scan;

				break;
			case listtypeAVIHEADER:
				dwLength = 0;	// silently enter the header block
				break;
			case listtypeSTREAMHEADER:
				if (!_parseStreamHeader(streamlist, dwLength, bScanRequired))
					fAcceptIndexOnly = false;
				else
					hyperindexed = true;

				++streams;
				dwLength = 0;
				break;
			}

			break;

		case ckidAVINEWINDEX:	// idx1
			if (!hyperindexed) {
				index_found = _parseIndexBlock(streamlist, dwLength/16, i64ChunkMoviPos);
				dwLength &= 15;
			}
			break;

		case ckidAVIPADDING:	// JUNK
			break;

		case MAKEFOURCC('s', 'e', 'g', 'm'):			// VirtualDub segment hint block
			delete [] pSegmentHint;
			if (!(pSegmentHint = new(std::nothrow) char[dwLength]))
				throw MyMemoryError();

			_readFile2(pSegmentHint, dwLength);

			if (dwLength&1)
				_skipFile2(1);

			dwLength = 0;
			break;

		}

		if (dwLength) {
			if (!_skipFile2(dwLength + (dwLength&1)))
				break;
		}

		// Quit as soon as we see the index block.

		if (fccType == ckidAVINEWINDEX)
			break;
	}

	if (i64ChunkMoviPos == 0)
		throw MyError("This AVI file doesn't have a movie data block (movi)!");

terminate_scan:

	if (!hyperindexed && !index_found)
		bScanRequired = true;

	if (bScanRequired) {
		// It's possible that we were in the middle of reading an index when an error
		// occurred, so we need to clear all of the indices for all streams.

		pasn = streamlist.AtHead();

		while((pasn_next = pasn->NextFromHead())) {
			pasn->index.clear();
			pasn = pasn_next;
		}

		// obtain length of file and limit scanning if so

		__int64 i64FileSize = _sizeFile();

		DWORD dwLengthLeft = dwChunkMoviLength;

//		long short_length = (long)((dwChunkMoviLength + 1023i64) >> 10);
//		long long_length = (long)((i64FileSize - i64ChunkMoviPos + 1023i64) >> 10);

//		__int64 length = (hyperindexed || bAggressive) ? long_length : short_length;
//		ProgressDialog pd(NULL, "AVI Import Filter", bAggressive ? "Reconstructing missing index block (aggressive mode)" : "Reconstructing missing index block", length, true);

//		pd.setValueFormat("%ldK of %ldK");

		fFakeIndex = true;

		_seekFile(i64ChunkMoviPos);

		// For standard AVI files, stop as soon as the movi chunk is exhausted or
		// the end of the AVI file is hit.  For OpenDML files, continue as long as
		// valid chunks are found.

		bool bStopWhenLengthExhausted = !hyperindexed && !bAggressive;

		for(;;) {
//			pd.advance((long)((_posFile() - i64ChunkMoviPos)/1024));
//			pd.check();

			// Exit if we are out of movi chunk -- except for OpenDML files.

			if (!bStopWhenLengthExhausted && dwLengthLeft < 8)
				break;

			// Validate the FOURCC itself but avoid validating the size, since it
			// may be too large for the last LIST/movi.

			if (!_readChunkHeader(fccType, dwLength))
				break;

			bool bValid = isValidFOURCC(fccType) && ((_posFile() + dwLength) <= i64FileSize);

			// In aggressive mode, verify that a valid FOURCC follows this chunk.

			if (bAggressive) {
				int64_t current_pos = _posFile();
				int64_t rounded_length = (dwLength+1LL) & ~1LL;

				if (current_pos + dwLength > i64FileSize)
					bValid = false;
				else if (current_pos + rounded_length <= i64FileSize-8) {
					FOURCC fccNext;
					DWORD dwLengthNext;

					_seekFile(current_pos + rounded_length);
					if (!_readChunkHeader(fccNext, dwLengthNext))
						break;

					bValid &= isValidFOURCC(fccNext) && ((_posFile() + dwLengthNext) <= i64FileSize);

					_seekFile(current_pos);
				}
			}

			if (!bValid) {
				// Notify the user that recovering this file requires stronger measures.

				if (!bAggressive) {
					bAggressive = true;
					bStopWhenLengthExhausted = false;
//					pd.setCaption("Reconstructing missing index block (aggressive mode)");
//					pd.setLimit(long_length);
				}

				// Backup by seven bytes and continue.

				_seekFile(_posFile()-7);
				continue;
			}

			int stream;

//			_RPT2(0,"(stream header) Chunk '%-4s', length %08lx\n", &fccType, dwLength);

			dwLengthLeft -= 8+(dwLength + (dwLength&1));

			// Skip over the chunk.  Don't skip over RIFF chunks of type AVIX, or
			// LIST chunks of type 'movi'.

			if (dwLength) {
				if (fccType == MAKEFOURCC('R','I','F','F') || fccType == MAKEFOURCC('L','I','S','T')) {
					FOURCC fccType2;

					if (!_readFile(&fccType2, 4))
						break;

					if (fccType2 != MAKEFOURCC('A','V','I','X') && fccType2 != MAKEFOURCC('m','o','v','i')) {
						if (!_skipFile2(dwLength + (dwLength&1) - 4))
							break;
					}
				} else {
					if (!_skipFile2(dwLength + (dwLength&1)))
						break;
				}
			}

			if (_posFile() > i64FileSize)
				break;

			if (isxdigit(fccType&0xff) && isxdigit((fccType>>8)&0xff)) {
				stream = StreamFromFOURCC(fccType);

				if (stream >=0 && stream < streams) {

					pasn = streamlist.AtHead();

					while(stream-- && (pasn_next = pasn->NextFromHead()))
						pasn = pasn_next;

					if (pasn && pasn_next) {

						// Set the keyframe flag for the first sample in the stream, or
						// if this is known to be a keyframe-only stream.  Do not set the
						// keyframe flag if the frame has zero bytes (drop frame).

						pasn->index.add(fccType, _posFile()-(dwLength + (dwLength&1))-8, dwLength, (!pasn->bytes || pasn->keyframe_only) && dwLength>0);
						pasn->bytes += dwLength;
					}
				}

			}
		}
	}

	bAggressivelyRecovered |= bAggressive;

	// glue together indices

	pasn = streamlist.AtHead();

	while((pasn_next = pasn->NextFromHead())) {
		if (!pasn->index.makeIndex2())
			throw MyMemoryError();

		pasn->frames = pasn->index.indexLen();

		// Clear sample size for video streams!

		if (pasn->hdr.fccType == streamtypeVIDEO)
			pasn->hdr.dwSampleSize=0;

		// Sample size == nBlockAlign for audio

		if (pasn->hdr.fccType == streamtypeAUDIO) {
			const AVIIndexEntry2 *pIdx = pasn->index.index2Ptr();
			long nBlockAlign = ((WAVEFORMATEX *)pasn->pFormat)->nBlockAlign;

			pasn->hdr.dwSampleSize = nBlockAlign;

			for(int i=0; i<pasn->frames; ++i) {
				long s = pIdx[i].size & 0x7FFFFFFF;

				if (s && s < nBlockAlign) {
					pasn->FixVBRAudio();
					break;
				}
			}
		}

		if (pasn->hdr.dwSampleSize)
			pasn->length = (long)pasn->bytes / pasn->hdr.dwSampleSize;
		else
			pasn->length = pasn->frames;

		pasn = pasn_next;
	}

//	throw MyError("Parse complete.  Aborting.");
}

bool AVIReadHandler::_parseStreamHeader(List2<AVIStreamNode>& streamlist, DWORD dwLengthLeft, bool& bIndexDamaged) {
	AVIStreamNode *pasn;
	FOURCC fccType;
	DWORD dwLength;
	bool hyperindexed = false;

	if (!(pasn = new(std::nothrow) AVIStreamNode()))
		throw MyMemoryError();

	try {
		while(dwLengthLeft >= 8 && _readChunkHeader(fccType, dwLength)) {

//			_RPT2(0,"(stream header) Chunk '%-4s', length %08lx\n", &fccType, dwLength);

			dwLengthLeft -= 8;

			if (dwLength > dwLengthLeft)
				throw MyError("Invalid AVI file: chunk size extends outside of parent");

			dwLengthLeft -= (dwLength + (dwLength&1));

			switch(fccType) {

			case ckidSTREAMHEADER:
				memset(&pasn->hdr, 0, sizeof pasn->hdr);

				if (dwLength < sizeof pasn->hdr) {
					_readFile2(&pasn->hdr, dwLength);
					if (dwLength & 1)
						_skipFile(1);
				} else {
					_readFile2(&pasn->hdr, sizeof pasn->hdr);
					_skipFile(dwLength+(dwLength&1) - sizeof pasn->hdr);
				}
				dwLength = 0;

				pasn->keyframe_only = false;

				break;

			case ckidSTREAMFORMAT:
				if (!(pasn->pFormat = new(std::nothrow) char[pasn->lFormatLen = dwLength]))
					throw MyMemoryError();

				_readFile2(pasn->pFormat, dwLength);

				if (pasn->hdr.fccType == streamtypeVIDEO) {
					switch(((BITMAPINFOHEADER *)pasn->pFormat)->biCompression) {
						case 0:
						case MAKEFOURCC('R', 'A', 'W', ' '): // ' WAR'
						case MAKEFOURCC('D', 'I', 'B', ' '): // ' BID'
            case MAKEFOURCC('d', 'm', 'b', '1'): // '1bmd'
            case MAKEFOURCC('m', 'j', 'p', 'g'): // 'gpjm'
            case MAKEFOURCC('M', 'J', 'P', 'G'): // 'GPJM'
            case MAKEFOURCC('V', 'Y', 'U', 'Y'): // 'YUYV'
            case MAKEFOURCC('Y', 'U', 'Y', '2'): // '2YUY'
            case MAKEFOURCC('U', 'Y', 'V', 'Y'): // 'YVYU'
            case MAKEFOURCC('Y', 'V', 'Y', 'U'): // 'UYVY'
            case MAKEFOURCC('Y', 'V', '1', '2'): // '21VY'
            case MAKEFOURCC('Y', 'V', '1', '6'): // '61VY'
            case MAKEFOURCC('Y', 'V', '2', '4'): // '42VY'
            case MAKEFOURCC('Y', '4', '1', 'B'): // 'B14Y'
            case MAKEFOURCC('Y', '8', '0', '0'): // '008Y'
            case MAKEFOURCC('Y', '8', ' ', ' '): // '  8Y'
            case MAKEFOURCC('I', '4', '2', '0'): // '024I'
            case MAKEFOURCC('Y', '4', '1', 'P'): // 'P14Y'
            case MAKEFOURCC('c', 'y', 'u', 'v'): // 'vuyc'
            case MAKEFOURCC('H', 'F', 'Y', 'U'): // 'UYFH'
            case MAKEFOURCC('b', 't', '2', '0'): // '02tb'
							pasn->keyframe_only = true;
					}
				}

				if (dwLength & 1)
					_skipFile(1);
				dwLength = 0;
				break;

			case MAKEFOURCC('i', 'n', 'd', 'x'):			// OpenDML extended index
				{
					__int64 posFileSave = _posFile();

					try {
						_parseExtendedIndexBlock(streamlist, pasn, -1, dwLength);
					} catch(const AvisynthError&) {
						bIndexDamaged = true;
					}

					_seekFile(posFileSave);
				}
				hyperindexed = true;
				break;

			case ckidAVIPADDING:	// JUNK
				break;
			}

			if (dwLength) {
				if (!_skipFile2(dwLength + (dwLength&1)))
					break;
			}
		}

		if (dwLengthLeft)
			_skipFile2(dwLengthLeft);
	} catch(...) {
		delete pasn;
		throw;
	}

//	_RPT1(0,"Found stream: type %s\n", pasn->hdr.fccType==streamtypeVIDEO ? "video" : pasn->hdr.fccType==streamtypeAUDIO ? "audio" : "unknown");

	streamlist.AddTail(pasn);

	return hyperindexed;
}

bool AVIReadHandler::_parseIndexBlock(List2<AVIStreamNode>& streamlist, int count, __int64 movi_offset) {
	AVIINDEXENTRY avie[32];
	AVIStreamNode *pasn, *pasn_next;
	bool absolute_addr = true;

	// Some AVI files have relative addresses in their AVI index chunks, and some
	// relative.  They're supposed to be relative to the 'movi' chunk; all versions
	// of VirtualDub using fast write routines prior to build 4936 generate absolute
	// addresses (oops). AVIFile and ActiveMovie are both ambivalent.  I guess we'd
	// better be as well.

	while(count > 0) {
		int tc = count;
		int i;

		if (tc>32) tc=32;
		count -= tc;

		if (tc*int(sizeof(AVIINDEXENTRY)) != _readFile(avie, tc*sizeof(AVIINDEXENTRY))) {
			pasn = streamlist.AtHead();

			while((pasn_next = pasn->NextFromHead())) {
				pasn->index.clear();
				pasn->bytes = 0;

				pasn = pasn_next;
			}
			return false;
		}

		for(i=0; i<tc; i++) {
			int stream = StreamFromFOURCC(avie[i].ckid);

			if (absolute_addr && avie[i].dwChunkOffset<movi_offset)
				absolute_addr = false;

			pasn = streamlist.AtHead();

			while((pasn_next = (AVIStreamNode *)pasn->NextFromHead()) && stream--)
				pasn = pasn_next;

			if (pasn && pasn_next) {
				if (absolute_addr)
					pasn->index.add(&avie[i]);
				else
					pasn->index.add(avie[i].ckid, (movi_offset-4) + (__int64)avie[i].dwChunkOffset, avie[i].dwChunkLength, !!(avie[i].dwFlags & AVIIF_KEYFRAME));

				pasn->bytes += avie[i].dwChunkLength;
			}
		}
	}

	return true;

}

void AVIReadHandler::_parseExtendedIndexBlock(List2<AVIStreamNode>& streamlist, AVIStreamNode *pasn, __int64 fpos, DWORD dwLength) {
#pragma warning( push )
#pragma warning (disable: 4815) // zero-sized array in stack object will have no elements
	union {
		AVISUPERINDEX idxsuper;
		AVISTDINDEX idxstd;
	};
#pragma warning( pop )

	union {
		struct	_avisuperindex_entry		superent[64];
		DWORD	dwHeap[256];
	};

	int entries, tp;
	int i;
	__int64 i64FPSave = _posFile();

	if (fpos>=0)
		_seekFile(fpos);
	_readFile2((char *)&idxsuper + 8, sizeof(AVISUPERINDEX) - 8);

	switch(idxsuper.bIndexType) {
	case AVI_INDEX_OF_INDEXES:
		// sanity check

		if (idxsuper.wLongsPerEntry != 4)
			throw MyError("Invalid superindex block in stream");

//		if (idxsuper.bIndexSubType != 0)
//			throw MyError("Field indexes not supported");

		entries = idxsuper.nEntriesInUse;

		while(entries > 0) {
			tp = sizeof superent / sizeof superent[0];
			if (tp>entries) tp=entries;

			_readFile2(superent, tp*sizeof superent[0]);

			for(i=0; i<tp; i++)
				_parseExtendedIndexBlock(streamlist, pasn, superent[i].qwOffset+8, superent[i].dwSize-8);

			entries -= tp;
		}

		break;

	case AVI_INDEX_OF_CHUNKS:

//		if (idxstd.bIndexSubType != 0)
//			throw MyError("Frame indexes not supported");

		entries = idxstd.nEntriesInUse;

		// In theory, if bIndexSubType==AVI_INDEX_2FIELD it's supposed to have
		// wLongsPerEntry=3, and bIndexSubType==0 gives wLongsPerEntry=2.
		// Matrox's MPEG-2 stuff generates bIndexSubType=16 and wLongsPerEntry=6.
		// *sigh*
		//
		// For wLongsPerEntry==2 and ==3, dwOffset is at 0 and dwLength at 1;
		// for wLongsPerEntry==6, dwOffset is at 2 and all are keyframes.

		{
			if (idxstd.wLongsPerEntry!=2 && idxstd.wLongsPerEntry!=3 && idxstd.wLongsPerEntry!=6)
				throw MyError("Invalid OpenDML index block in stream (wLongsPerEntry=%d)", idxstd.wLongsPerEntry);

			while(entries > 0) {
				tp = (sizeof dwHeap / sizeof dwHeap[0]) / idxstd.wLongsPerEntry;
				if (tp>entries) tp=entries;

				_readFile2(dwHeap, tp*idxstd.wLongsPerEntry*sizeof(DWORD));

				if (idxstd.wLongsPerEntry == 6)
					for(i=0; i<tp; i++) {
						DWORD dwOffset = dwHeap[i*idxstd.wLongsPerEntry + 0];
						DWORD dwSize = dwHeap[i*idxstd.wLongsPerEntry + 2];

						pasn->index.add(idxstd.dwChunkId, (idxstd.qwBaseOffset+dwOffset)-8, dwSize, true);

						pasn->bytes += dwSize;
					}
				else
					for(i=0; i<tp; i++) {
						DWORD dwOffset = dwHeap[i*idxstd.wLongsPerEntry + 0];
						DWORD dwSize = dwHeap[i*idxstd.wLongsPerEntry + 1];

						pasn->index.add(idxstd.dwChunkId, (idxstd.qwBaseOffset+dwOffset)-8, dwSize&0x7FFFFFFF, !(dwSize&0x80000000));

						pasn->bytes += dwSize & 0x7FFFFFFF;
					}

				entries -= tp;
			}
		}

		break;

	default:
		throw MyError("Unknown hyperindex type");
	}

	_seekFile(i64FPSave);
}

void AVIReadHandler::_destruct() {
	AVIStreamNode *pasn;
	AVIFileDesc *pDesc;

	if (pAvisynthClipInfo)
		pAvisynthClipInfo->Release();

	if (paf)
		AVIFileRelease(paf);

	while((pasn = listStreams.RemoveTail()))
		delete pasn;

	delete streamBuffer;

	if (listFiles.IsEmpty()) {
		if (hFile != INVALID_HANDLE_VALUE)
			CloseHandle(hFile);
		if (hFileUnbuffered != INVALID_HANDLE_VALUE)
			CloseHandle(hFileUnbuffered);
	} else
		while((pDesc = listFiles.RemoveTail())) {
			CloseHandle(pDesc->hFile);
			CloseHandle(pDesc->hFileUnbuffered);
			delete pDesc;
		}

	delete [] pSegmentHint;
}

///////////////////////////////////////////////////////////////////////////

void AVIReadHandler::Release() {
	if (!--ref_count)
		delete this;
}

void AVIReadHandler::AddRef() {
	++ref_count;
}

IAVIReadStream *AVIReadHandler::GetStream(DWORD fccType, LONG lParam) {
	if (paf) {
		PAVISTREAM pas;
		HRESULT hr;

		hr = AVIFileGetStream(paf, &pas, fccType, lParam);

		if (FAILED(hr))
			return NULL;

		return new AVIReadTunnelStream(this, pas, pAvisynthClipInfo);
	} else {
		AVIStreamNode *pasn, *pasn_next;
		int streamno = 0;

		pasn = listStreams.AtHead();

		while((pasn_next = pasn->NextFromHead())) {
			if (pasn->hdr.fccType == fccType && !lParam--)
				break;

			pasn = pasn_next;
			++streamno;
		}

		if (pasn_next) {
			return new AVIReadStream(this, pasn, streamno);
		}

		return NULL;
	}
}

void AVIReadHandler::EnableFastIO(bool f) {
	fDisableFastIO = !f;
}

bool AVIReadHandler::isOptimizedForRealtime() {
	return nRealTime!=0;
}

bool AVIReadHandler::isStreaming() {
	return nActiveStreamers!=0;
}

bool AVIReadHandler::isIndexFabricated() {
	return fFakeIndex;
}

bool AVIReadHandler::getSegmentHint(const char **ppszPath) {
	if (!pSegmentHint) {
		if (ppszPath)
			*ppszPath = NULL;

		return false;
	}

	if (ppszPath)
		*ppszPath = pSegmentHint+1;

	return !!pSegmentHint[0];
}

///////////////////////////////////////////////////////////////////////////

void AVIReadHandler::EnableStreaming(int stream) {
	if (!fStreamsActive) {
		if (!(streamBuffer = new(std::nothrow) char[STREAM_SIZE]))
			throw MyMemoryError();

		i64StreamPosition = -1;
		sbPosition = sbSize = 0;
	}

	fStreamsActive |= (1<<stream);
	++nActiveStreamers;
}

void AVIReadHandler::DisableStreaming(int stream) {
	fStreamsActive &= ~(1<<stream);

	if (!fStreamsActive) {
		delete streamBuffer;
		streamBuffer = NULL;
	}
	--nActiveStreamers;
}

void AVIReadHandler::AdjustRealTime(bool fInc) {
	if (fInc)
		++nRealTime;
	else
		--nRealTime;
}

char *AVIReadHandler::_StreamRead(long& bytes) {
	if (nCurrentFile<0 || nCurrentFile != (int)(i64StreamPosition>>48))
		_SelectFile((int)(i64StreamPosition>>48));

	if (sbPosition >= sbSize) {
		if (nRealTime || (((i64StreamPosition&0x0000FFFFFFFFFFFFLL)+sbSize) & -STREAM_BLOCK_SIZE)+STREAM_SIZE > i64Size) {
			i64StreamPosition += sbSize;
			sbPosition = 0;
			_seekFile(i64StreamPosition & 0x0000FFFFFFFFFFFFLL);

			sbSize = _readFile(streamBuffer, STREAM_RT_SIZE);

			if (sbSize < 0) {
				sbSize = 0;
				throw MyWin32Error("Failure streaming AVI file: %%s.",GetLastError());
			}
		} else {
			i64StreamPosition += sbSize;
			sbPosition = (int)i64StreamPosition & (STREAM_BLOCK_SIZE-1);
			i64StreamPosition &= -STREAM_BLOCK_SIZE;
			_seekFileUnbuffered(i64StreamPosition & 0x0000FFFFFFFFFFFFLL);

			sbSize = _readFileUnbuffered(streamBuffer, STREAM_SIZE);

			if (sbSize < 0) {
				sbSize = 0;
				throw MyWin32Error("Failure streaming AVI file: %%s.",GetLastError());
			}
		}
	}

	if (sbPosition >= sbSize)
		return NULL;

	if (bytes > sbSize - sbPosition)
		bytes = sbSize - sbPosition;

	sbPosition += bytes;

	return streamBuffer + sbPosition - bytes;
}

bool AVIReadHandler::Stream(AVIStreamNode *pusher, __int64 pos) {

	// Do not stream aggressively recovered files.

	if (bAggressivelyRecovered)
		return false;

	bool read_something = false;

	if (!streamBuffer)
		return false;

	if (i64StreamPosition == -1) {
		i64StreamPosition = pos;
		sbPosition = 0;
	}

	if (pos < i64StreamPosition+sbPosition)
		return false;

	// >4Mb past current position!?

	if (pos > i64StreamPosition+sbPosition+4194304) {
//		OutputDebugString("Resetting streaming position!\n");
		i64StreamPosition = pos;
		sbSize = sbPosition = 0;
	}

/*	if (pusher->hdr.fccType == 'sdiv')
		OutputDebugString("pushed by video\n");
	else
		OutputDebugString("pushed by audio\n");*/

	++pusher->stream_pushes;
	pusher->stream_push_pos = pos;

	while(pos >= i64StreamPosition+sbPosition) {
		long actual, left;
		char *src;
		FOURCC hdr[2];
		int stream;

		// read next header

		left = 8;
		while(left > 0) {
			actual = left;
			src = _StreamRead(actual);

			if (!src)
				return read_something;

			memcpy((char *)hdr + (8-left), src, actual);
			left -= actual;
		}

		stream = StreamFromFOURCC(hdr[0]);

		if (isxdigit(hdr[0]&0xff) && isxdigit((hdr[0]>>8)&0xff) && stream<32 &&
			((1L<<stream) & fStreamsActive)) {

//			_RPT3(0,"\tstream: reading chunk at %I64x, length %6ld, stream %ld\n", i64StreamPosition+sbPosition-8, hdr[1], stream);

			AVIStreamNode *pasn, *pasn_next;
			int streamno = 0;

			pasn = listStreams.AtHead();

			while((pasn_next = pasn->NextFromHead())) {
				if (streamno == stream) {
					long left = hdr[1] + (hdr[1]&1);
					bool fWrite = pasn->cache->WriteBegin(i64StreamPosition + sbPosition, left);
					char *dst;

					while(left > 0) {
						actual = left;

						dst = _StreamRead(actual);

						if (!dst) {
							if (fWrite)
								pasn->cache->WriteEnd();
							return read_something;
						}

						if (fWrite)
							pasn->cache->Write(dst, actual);

						left -= actual;
					}

					if (fWrite)
						pasn->cache->WriteEnd();

					read_something = true;

					break;
				}

				pasn = pasn_next;
				++streamno;
			}
		} else {

			if (hdr[0] != FOURCC_LIST && hdr[0] != FOURCC_RIFF) {
				long actual;

				// skip chunk

				left = hdr[1] + (hdr[1] & 1);

				// Determine if the chunk is overly large.  If the chunk is too large, don't
				// stream through it.

				if (left > 262144)
					return read_something;

				while(left > 0) {
					actual = left;

					if (!_StreamRead(actual))
						return read_something;

					left -= actual;
				}
			} else {
				left = 4;

				while(left > 0) {
					actual = left;

					if (!_StreamRead(actual))
						return read_something;

					left -= actual;
				}
			}

		}
	}

	return true;
}

__int64 AVIReadHandler::getStreamPtr() {
	return i64StreamPosition + sbPosition;
}

void AVIReadHandler::FixCacheProblems(AVIReadStream *arse) {
	AVIStreamNode *pasn, *pasn_next;

	// The simplest fix is simply to disable caching on the stream that's
	// cache-missing.  However, this is a bad idea if the problem is a low-bandwidth
	// audio stream that's outrunning a high-bandwidth video stream behind it.
	// Check the streaming leader, and if the streaming leader is comparatively
	// low bandwidth and running at least 512K ahead of the cache-missing stream,
	// disable its cache.

	AVIStreamNode *stream_leader = NULL;
	int stream_leader_no=0;
	int streamno=0;

	pasn = listStreams.AtHead();

	while((pasn_next = pasn->NextFromHead())) {
		if (pasn->cache)
			if (!stream_leader || pasn->stream_pushes > stream_leader->stream_pushes) {
				stream_leader = pasn;
				stream_leader_no = streamno;
			}

		pasn = pasn_next;
		++streamno;
	}

	if (stream_leader && stream_leader->stream_bytes*2 < arse->psnData->stream_bytes
		&& stream_leader->stream_push_pos >= arse->psnData->stream_push_pos+524288) {
#ifdef STREAMING_DEBUG
		OutputDebugString("caching disabled on fast puny leader\n");
#endif
		delete stream_leader->cache;
		stream_leader->cache = NULL;

		DisableStreaming(stream_leader_no);

		i64StreamPosition = -1;
		sbPosition = sbSize = 0;
	} else {
#ifdef STREAMING_DEBUG
		OutputDebugString("disabling caching at request of client\n");
#endif

		arse->EndStreaming();

		if (arse->psnData == stream_leader) {
			i64StreamPosition = -1;
			sbPosition = sbSize = 0;
		}
	}

	// Reset cache statistics on all caches.

	pasn = listStreams.AtHead();

	while((pasn_next = pasn->NextFromHead())) {
		if (pasn->cache)
			pasn->cache->ResetStatistics();

		pasn = pasn_next;
	}
}

long AVIReadHandler::ReadData(int stream, void *buffer, __int64 position, long len) {
	if (nCurrentFile<0 || nCurrentFile != (int)(position>>48))
		_SelectFile((int)(position>>48));

//	_RPT3(0,"Reading from file %d, position %I64x, size %d\n", nCurrentFile, position, len);

	if (!_seekFile2(position & 0x0000FFFFFFFFFFFFLL))
		return -1;
	return _readFile(buffer, len);
}

void AVIReadHandler::_SelectFile(int file) {
	AVIFileDesc *pDesc, *pDesc_next;

	nCurrentFile = file;

	pDesc = listFiles.AtHead();
	while((pDesc_next = pDesc->NextFromHead()) && file--)
		pDesc = pDesc_next;

	hFile			= pDesc->hFile;
	hFileUnbuffered	= pDesc->hFileUnbuffered;
	i64Size			= pDesc->i64Size;
}
