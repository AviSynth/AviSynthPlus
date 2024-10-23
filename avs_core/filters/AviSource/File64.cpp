// Packaged with Avisynth v1.0 beta.
// http://www.math.berkeley.edu/~benrg/avisynth.html

//	VirtualDub - Video processing and capture application
//	Copyright (C) 1998-2000 Avery Lee
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

#include "Error.h"

#include "File64.h"

// hack...

extern CRITICAL_SECTION g_diskcs;

////////////

File64::File64() {
}

File64::File64(HANDLE _hFile, HANDLE _hFileUnbuffered)
: hFile(_hFile), hFileUnbuffered(_hFileUnbuffered)
{
	i64FilePosition = 0;
}

long File64::_readFile(void *data, long len) {
	DWORD dwActual;

	if (!ReadFile(hFile, data, len, &dwActual, NULL))
		return -1;

	i64FilePosition += dwActual;

	return (long)dwActual;
}

void File64::_readFile2(void *data, long len) {
	long lActual = _readFile(data, len);

	if (lActual < 0)
		throw MyWin32Error("Failure reading file: %%s.",GetLastError());

	if (lActual != len)
		throw MyError("Failure reading file: Unexpected end of file");
}

bool File64::_readChunkHeader(FOURCC& pfcc, DWORD& pdwLen) {
	DWORD dw[2];
	long actual;

	actual = _readFile(dw, 8);

	if (actual != 8)
		return false;

	pfcc = dw[0];
	pdwLen = dw[1];

	return true;
}

void File64::_seekFile(__int64 i64NewPos) {
	LONG lHi = (LONG)(i64NewPos>>32);
	DWORD dwError;

	if (0xFFFFFFFFL == SetFilePointer(hFile, (LONG)i64NewPos, &lHi, FILE_BEGIN))
		if ((dwError = GetLastError()) != NO_ERROR)
			throw MyWin32Error("File64: %%s", dwError);

	i64FilePosition = i64NewPos;
}

bool File64::_seekFile2(__int64 i64NewPos) {
	LONG lHi = (LONG)(i64NewPos>>32);
	DWORD dwError;

//	_RPT1(0,"Seeking to %" PRIu64 "\n", i64NewPos);

	if (0xFFFFFFFFL == SetFilePointer(hFile, (LONG)i64NewPos, &lHi, FILE_BEGIN))
		if ((dwError = GetLastError()) != NO_ERROR)
			return false;

	i64FilePosition = i64NewPos;

	return true;
}

void File64::_skipFile(__int64 bytes) {
	LONG lHi = (LONG)(bytes>>32);
	DWORD dwError;
	LONG lNewLow;

	if (0xFFFFFFFFL == (lNewLow = SetFilePointer(hFile, (LONG)bytes, &lHi, FILE_CURRENT)))
		if ((dwError = GetLastError()) != NO_ERROR)
			throw MyWin32Error("File64: %%s", dwError);

	i64FilePosition = (unsigned long)lNewLow | (((__int64)(unsigned long)lHi)<<32);
}

bool File64::_skipFile2(__int64 bytes) {
	LONG lHi = (LONG)(bytes>>32);
	DWORD dwError;
	LONG lNewLow;

	if (0xFFFFFFFFL == (lNewLow = SetFilePointer(hFile, (LONG)bytes, &lHi, FILE_CURRENT)))
		if ((dwError = GetLastError()) != NO_ERROR)
			return false;

	i64FilePosition = (unsigned long)lNewLow | (((__int64)(unsigned long)lHi)<<32);

	return true;
}

long File64::_readFileUnbuffered(void *data, long len) {
	DWORD dwActual;

	EnterCriticalSection(&g_diskcs);
	if (!ReadFile(hFileUnbuffered, data, len, &dwActual, NULL)) {
		LeaveCriticalSection(&g_diskcs);
		return -1;
	}
	LeaveCriticalSection(&g_diskcs);

	return (long)dwActual;
}

void File64::_seekFileUnbuffered(__int64 i64NewPos) {
	LONG lHi = (LONG)(i64NewPos>>32);
	DWORD dwError;

	if (0xFFFFFFFF == SetFilePointer(hFileUnbuffered, (LONG)i64NewPos, &lHi, FILE_BEGIN))
		if ((dwError = GetLastError()) != NO_ERROR)
			throw MyWin32Error("File64: %%s", dwError);
}

__int64 File64::_posFile() {
	return i64FilePosition;
}

__int64 File64::_sizeFile() {
	DWORD dwLow, dwHigh;
	DWORD dwError;

	dwLow = GetFileSize(hFile, &dwHigh);

	if (dwLow == 0xFFFFFFFF && (dwError = GetLastError()) != NO_ERROR)
		throw MyWin32Error("Cannot determine file size: %%s", dwError);

	return ((__int64)dwHigh << 32) | (unsigned long)dwLow;
}
