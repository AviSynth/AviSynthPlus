/******************************************************
  A fast and high quality sampling rate converter SSRC
                                           written by Naoki Shibata


Homepage : http://shibatch.sourceforge.net/
e-mail   : shibatch@users.sourceforge.net

Some changes are:

Copyright (c) 2001-2003, Peter Pawlowski
All rights reserved.

*******************************************************/

#include "ssrc.h"
#include <avs/alignment.h>
#include <avs/config.h>
#include <cassert>

#pragma warning(disable:4244)

template<class REAL>
class Resampler_i_base : public Resampler_base
{
protected:
	Resampler_i_base(const Resampler_base::CONFIG & c) : Resampler_base(c) {}
	void make_outbuf(int nsmplwrt2, REAL* outbuf, int& delay2);
  void make_inbuf(int nsmplread, int inbuflen, REAL_inout* rawinbuf, REAL* inbuf, int toberead);
};

#define M 15

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795028842
#endif

#define RANDBUFLEN 65536

#define RINT(x) ((x) >= 0 ? ((int)((x) + 0.5)) : ((int)((x) - 0.5)))

extern "C"
{
extern double dbesi0(double);
}


static double alpha(double a)
{
  if (a <= 21) return 0;
  if (a <= 50) return 0.5842*pow(a-21,0.4)+0.07886*(a-21);
  return 0.1102*(a-8.7);
}


static double win(double n,int len,double alp,double iza)
{
  return dbesi0(alp*sqrt(1-4*n*n/(((double)len-1)*((double)len-1))))/iza;
}

static double sinc(double x)
{
  return x == 0 ? 1 : sin(x)/x;
}

static double hn_lpf(int n,double lpf,double fs)
{
  double t = 1/fs;
  double omega = 2*M_PI*lpf;
  return 2*lpf*t*sinc(n*omega*t);
}




static int gcd(int x, int y)
{
    int t;

    while (y != 0) {
        t = x % y;  x = y;  y = t;
    }
    return x;
}

int CanResample(int sfrq,int dfrq)
{
	if (sfrq==dfrq) return 1;
    int frqgcd = gcd(sfrq,dfrq);

	if (dfrq>sfrq)
	{
		int fs1 = sfrq / frqgcd * dfrq;

		if (fs1/dfrq == 1) return 1;
		else if (fs1/dfrq % 2 == 0) return 1;
		else if (fs1/dfrq % 3 == 0) return 1;
		else return 0;
	}
	else
	{
		if (dfrq/frqgcd == 1) return 1;
		else if (dfrq/frqgcd % 2 == 0) return 1;
		else if (dfrq/frqgcd % 3 == 0) return 1;
		else return 0;
	}
}

void Buffer::Read(int size)
{
	if (size)
	{
		if (buf_data==size) buf_data=0;
		else
		{
			mem_ops<REAL_inout>::move(buffer,buffer+size,buf_data-size);
			buf_data-=size;
		}
	}
}

void Buffer::Write(const REAL_inout * ptr,int size)
{
	buffer.check_size(buf_data + size);
	mem_ops<REAL_inout>::copy(buffer+buf_data,ptr,size);
	buf_data+=size;
}

void Resampler_base::bufloop(int finish)
{
	int s;
	REAL_inout * ptr = in.GetBuffer(&s);
	int done=0;
	while(done<s)
	{
		int d=Resample(ptr,s-done,finish);
		if (d==0) break;
		done+=d;
		ptr+=d;
	}
	in.Read(done);
}

void Resampler_base::Write(const REAL_inout * input,int size)
{
	in.Write(input,size);
	bufloop(0);
}

template<class REAL>
void Resampler_i_base<REAL>::make_inbuf(int nsmplread, int inbuflen, REAL_inout* rawinbuf, REAL* inbuf, int toberead)
{
	const int	MaxLoop = nsmplread * nch;
	const int	InbufBase = inbuflen * nch;

	for(int i = 0; i < MaxLoop; i++) {
		inbuf[InbufBase + i] = rawinbuf[i];
	}

	size_t	ClearSize = toberead - nsmplread;

	if(ClearSize) {
		memset(inbuf + InbufBase + MaxLoop, 0, ClearSize * nch * sizeof(REAL));
	}
}

template<class REAL>
void Resampler_i_base<REAL>::make_outbuf(int nsmplwrt2, REAL* outbuf, int& delay2)
{
	const int	MaxLoop = nsmplwrt2 * nch;

	for(int i = 0; i < MaxLoop; i++) {
		REAL	s = outbuf[i] * gain;

		if(s > 1.0) {
			peak = peak < s ? s : peak;
//			s =  1.0;
		} else if(s < -1.0) {
			peak = peak < -s ? -s : peak;
//			s = -1.0;
		}

		__output((REAL_inout)s,delay2);
	}
}


#ifdef _WIN32
#if defined(GCC) || (defined(CLANG) && !defined(MSVC))
#ifndef MulDiv
  #define MulDiv(x,y,z) ((x)*(y)/(z))
#endif
#else
extern "C" {
  _declspec(dllimport) int _stdcall MulDiv(int nNumber,int nNumerator,int nDenominator);
}
#endif
#else
#ifndef MulDiv
  #define MulDiv(x,y,z) ((x)*(y)/(z))
#endif
#endif

unsigned int Resampler_base::GetLatency()
{
	return MulDiv(in.Size(),1000,sfrq*nch) + MulDiv(out.Size(),1000,dfrq*nch);
}

template<class REAL>
class Upsampler : public Resampler_i_base<REAL>
{
  using Resampler_i_base<REAL>::FFTFIRLEN;
  using Resampler_i_base<REAL>::peak;
  using Resampler_i_base<REAL>::sfrq;
  using Resampler_i_base<REAL>::dfrq;
  using Resampler_i_base<REAL>::nch;
  using Resampler_i_base<REAL>::DF;
  using Resampler_i_base<REAL>::AA;
  using Resampler_i_base<REAL>::make_outbuf;
  using Resampler_i_base<REAL>::make_inbuf;

  int64_t fs1;
  int frqgcd,osf,fs2;
  REAL **stage1,*stage2;
  int n1,n1x,n1y,n2,n2b;
  int filter2len;
  int *f1order,*f1inc;
  int *fft_ip;// = NULL;
  REAL *fft_w;// = NULL;
  //unsigned char *rawinbuf,*rawoutbuf;
  REAL *inbuf,*outbuf;
  REAL **buf1,**buf2;
  int spcount;
  int i,j;

		int n2b2;//=n2b/2;
		int rp;        // keep the location of the next samples to read in inbuf at fs1.
		int ds;        // number of samples to dispose next in sfrq.
		int nsmplwrt1; // actually number of samples to send stage2 filters .
		int nsmplwrt2; // actually number of samples to send stage2 filters .
		int s1p;       // the reminder obtained by dividing the samples output from stage1 filter by n1y*osf.
    int init;
		unsigned int sumread,sumwrite;
		int osc;
		REAL *ip,*ip_backup;
		int s1p_backup,osc_backup;
		int p;
		int inbuflen;
		int delay;// = 0;
		int delay2;


public:
  Upsampler(const Resampler_base::CONFIG & c);
  unsigned int Resample(REAL_inout * rawinbuf, unsigned int in_size, int ending);
  ~Upsampler();

};


template<class REAL>
Upsampler<REAL>::Upsampler(const Resampler_base::CONFIG & c) : Resampler_i_base<REAL>(c)
{
  fft_ip = NULL;
  fft_w = NULL;
  peak = 0;
  spcount = 0;


  filter2len = FFTFIRLEN; /* stage 2 filter length */

                          /* Make stage 1 filter */

  {
    double aa = AA; /* stop band attenuation(dB) */
    double lpf, delta, d, df, alp, iza;
    double guard = 2;

    frqgcd = gcd(sfrq, dfrq);

    fs1 = (int64_t)(sfrq / frqgcd) * (int64_t)dfrq;

    if (fs1 / dfrq == 1) osf = 1;
    else if (fs1 / dfrq % 2 == 0) osf = 2;
    else if (fs1 / dfrq % 3 == 0) osf = 3;
    else {
      //		  fprintf(stderr,"Resampling from %dHz to %dHz is not supported.\n",sfrq,dfrq);
      //		  fprintf(stderr,"%d/gcd(%d,%d)=%d must be divided by 2 or 3.\n",sfrq,sfrq,dfrq,fs1/dfrq);
      //		  exit(-1);
      return;
    }

    df = (dfrq*osf / 2 - sfrq / 2) * 2 / guard;
    lpf = sfrq / 2 + (dfrq*osf / 2 - sfrq / 2) / guard;

    delta = pow(10.0, -aa / 20);
    if (aa <= 21) d = 0.9222; else d = (aa - 7.95) / 14.36;

    n1 = fs1 / df * d + 1;
    if (n1 % 2 == 0) n1++;

    alp = alpha(aa);
    iza = dbesi0(alp);
    //printf("iza = %g\n",iza);

    n1y = fs1 / sfrq;
    n1x = n1 / n1y + 1;

    f1order = (int*)avs_malloc(sizeof(int)*n1y*osf, 64);
    for (i = 0; i<n1y*osf; i++) {
      f1order[i] = fs1 / sfrq - (i*(fs1 / (dfrq*osf))) % (fs1 / sfrq);
      if (f1order[i] == fs1 / sfrq) f1order[i] = 0;
    }

    f1inc = (int*)avs_malloc(sizeof(int)*n1y*osf, 64);
    for (i = 0; i<n1y*osf; i++) {
      f1inc[i] = f1order[i] < fs1 / (dfrq*osf) ? nch : 0;
      if (f1order[i] == fs1 / sfrq) f1order[i] = 0;
    }

    stage1 = (REAL**)avs_malloc(sizeof(REAL *)*n1y, 64);
    stage1[0] = (REAL*)avs_malloc(sizeof(REAL)*n1x*n1y, 64);

    for (i = 1; i<n1y; i++) {
      stage1[i] = &(stage1[0][n1x*i]);
      for (j = 0; j<n1x; j++) stage1[i][j] = 0;
    }

    for (i = -(n1 / 2); i <= n1 / 2; i++)
    {
      stage1[(i + n1 / 2) % n1y][(i + n1 / 2) / n1y] = win(i, n1, alp, iza)*hn_lpf(i, lpf, fs1)*fs1 / sfrq;
    }
  }

  /* Make stage 2 filter */

  {
    double aa = AA; /* stop band attenuation(dB) */
    double lpf, delta, d, df, alp, iza;
    int ipsize, wsize;

    delta = pow(10.0, -aa / 20);
    if (aa <= 21) d = 0.9222; else d = (aa - 7.95) / 14.36;

    fs2 = dfrq * osf;

    for (i = 1;; i = i * 2)
    {
      n2 = filter2len * i;
      if (n2 % 2 == 0) n2--;
      df = (fs2*d) / (n2 - 1);
      lpf = sfrq / 2;
      if (df < DF) break;
    }

    alp = alpha(aa);

    iza = dbesi0(alp);

    for (n2b = 1; n2b<n2; n2b *= 2);
    n2b *= 2;

    stage2 = (REAL*)avs_malloc(sizeof(REAL)*n2b, 64);

    for (i = 0; i<n2b; i++) stage2[i] = 0;

    for (i = -(n2 / 2); i <= n2 / 2; i++) {
      stage2[i + n2 / 2] = win(i, n2, alp, iza)*hn_lpf(i, lpf, fs2) / n2b * 2;
    }

    ipsize = 2 + sqrt((double)n2b);
    fft_ip = (int*)avs_malloc(sizeof(int)*ipsize, 64);
    fft_ip[0] = 0;
    wsize = n2b / 2;
    fft_w = (REAL*)avs_malloc(sizeof(REAL)*wsize, 64);

    fft<REAL>::rdft(n2b, 1, stage2, fft_ip, fft_w);
  }

  //	  delay=0;
  n2b2 = n2b / 2;

  buf1 = (REAL**)avs_malloc(sizeof(REAL *)*nch, 64);
  for (i = 0; i<nch; i++)
  {
    buf1[i] = (REAL*)avs_malloc(sizeof(REAL)*(n2b2 / osf + 1), 64);
    for (j = 0; j<(n2b2 / osf + 1); j++) buf1[i][j] = 0;
  }

  buf2 = (REAL**)avs_malloc(sizeof(REAL *)*nch, 64);
  for (i = 0; i<nch; i++) buf2[i] = (REAL*)avs_malloc(sizeof(REAL)*n2b, 64);


  inbuf = (REAL*)avs_malloc(nch*(n2b2 + n1x) * sizeof(REAL), 64);
  outbuf = (REAL*)avs_malloc(sizeof(REAL)*nch*(n2b2 / osf + 1), 64);

  s1p = 0;
  rp = 0;
  ds = 0;
  osc = 0;

  init = 1;
  inbuflen = n1 / 2 / (fs1 / sfrq) + 1;
  delay = (double)n2 / 2 / (fs2 / dfrq);
  delay2 = delay * nch;

  sumread = sumwrite = 0;

}

template<class REAL>
unsigned int Upsampler<REAL>::Resample(REAL_inout * rawinbuf, unsigned int in_size, int ending)
{
  /* Apply filters */

  int nsmplread, toberead, toberead2;
  unsigned int rv = 0;
  int ch;


  toberead2 = toberead = floor((double)n2b2*sfrq / (dfrq*osf)) + 1 + n1x - inbuflen;

  if (!ending)
  {
    rv = nch * toberead;
    if (in_size<rv) return 0;
    nsmplread = toberead;
  }
  else
  {
    nsmplread = in_size / nch;
    rv = nsmplread * nch;
  }

  make_inbuf(nsmplread, inbuflen, rawinbuf, inbuf, toberead);

  inbuflen += toberead2;

  sumread += nsmplread;

  //nsmplwrt1 = ((rp-1)*sfrq/fs1+inbuflen-n1x)*dfrq*osf/sfrq;
  //if (nsmplwrt1 > n2b2) nsmplwrt1 = n2b2;
  nsmplwrt1 = n2b2;


  // apply stage 1 filter

  ip = &inbuf[((sfrq*(rp - 1) + fs1) / fs1)*nch];

  s1p_backup = s1p;
  ip_backup = ip;
  osc_backup = osc;

  for (ch = 0; ch<nch; ch++)
  {
    REAL *op = &outbuf[ch];
    int fdo = fs1 / (dfrq*osf), no = n1y * osf;

    s1p = s1p_backup; ip = ip_backup + ch;

    switch (n1x)
    {
    case 7:
      for (p = 0; p<nsmplwrt1; p++)
      {
        int s1o = f1order[s1p];

        buf2[ch][p] =
          stage1[s1o][0] * *(ip + 0 * nch) +
          stage1[s1o][1] * *(ip + 1 * nch) +
          stage1[s1o][2] * *(ip + 2 * nch) +
          stage1[s1o][3] * *(ip + 3 * nch) +
          stage1[s1o][4] * *(ip + 4 * nch) +
          stage1[s1o][5] * *(ip + 5 * nch) +
          stage1[s1o][6] * *(ip + 6 * nch);

        ip += f1inc[s1p];

        s1p++;
        if (s1p == no) s1p = 0;
      }
      break;

    case 9:
      for (p = 0; p<nsmplwrt1; p++)
      {
        int s1o = f1order[s1p];

        buf2[ch][p] =
          stage1[s1o][0] * *(ip + 0 * nch) +
          stage1[s1o][1] * *(ip + 1 * nch) +
          stage1[s1o][2] * *(ip + 2 * nch) +
          stage1[s1o][3] * *(ip + 3 * nch) +
          stage1[s1o][4] * *(ip + 4 * nch) +
          stage1[s1o][5] * *(ip + 5 * nch) +
          stage1[s1o][6] * *(ip + 6 * nch) +
          stage1[s1o][7] * *(ip + 7 * nch) +
          stage1[s1o][8] * *(ip + 8 * nch);

        ip += f1inc[s1p];

        s1p++;
        if (s1p == no) s1p = 0;
      }
      break;

    default:
      for (p = 0; p<nsmplwrt1; p++)
      {
        REAL tmp = 0;
        REAL *ip2 = ip;

        int s1o = f1order[s1p];

        for (i = 0; i<n1x; i++)
        {
          tmp += stage1[s1o][i] * *ip2;
          ip2 += nch;
        }
        buf2[ch][p] = tmp;

        ip += f1inc[s1p];

        s1p++;
        if (s1p == no) s1p = 0;
      }
      break;
    }

    osc = osc_backup;

    // apply stage 2 filter

    for (p = nsmplwrt1; p<n2b; p++) buf2[ch][p] = 0;

    //for(i=0;i<n2b2;i++) printf("%d:%g ",i,buf2[ch][i]);

    fft<REAL>::rdft(n2b, 1, buf2[ch], fft_ip, fft_w);


    buf2[ch][0] = stage2[0] * buf2[ch][0];
    buf2[ch][1] = stage2[1] * buf2[ch][1];



    for (i = 1; i<n2b / 2; i++)
    {
      REAL re, im;

      re = stage2[i * 2] * buf2[ch][i * 2] - stage2[i * 2 + 1] * buf2[ch][i * 2 + 1];
      im = stage2[i * 2 + 1] * buf2[ch][i * 2] + stage2[i * 2] * buf2[ch][i * 2 + 1];

      //printf("%d : %g %g %g %g %g %g\n",i,stage2[i*2],stage2[i*2+1],buf2[ch][i*2],buf2[ch][i*2+1],re,im);

      buf2[ch][i * 2] = re;
      buf2[ch][i * 2 + 1] = im;
    }

    fft<REAL>::rdft(n2b, -1, buf2[ch], fft_ip, fft_w);

    for (i = osc, j = 0; i<n2b2; i += osf, j++)
    {
      REAL f = (buf1[ch][j] + buf2[ch][i]);
      op[j*nch] = f;
    }

    nsmplwrt2 = j;

    osc = i - n2b2;

    for (j = 0; i<n2b; i += osf, j++)
      buf1[ch][j] = buf2[ch][i];

  }

  rp += nsmplwrt1 * (sfrq / frqgcd) / osf;


  make_outbuf(nsmplwrt2, outbuf, delay2);

  if (!init) {
    if (ending) {
      if ((double)sumread*dfrq / sfrq + 2 > sumwrite + nsmplwrt2) {


        sumwrite += nsmplwrt2;
      }
      else {

      }
    }
    else {
      sumwrite += nsmplwrt2;
    }
  }
  else {
    if (nsmplwrt2 < delay) {
      delay -= nsmplwrt2;
    }
    else {
      if (ending) {
        if ((double)sumread*dfrq / sfrq + 2 > sumwrite + nsmplwrt2 - delay) {
          sumwrite += nsmplwrt2 - delay;
        }
        else {

        }
      }
      else {

        sumwrite += nsmplwrt2 - delay;
        init = 0;
      }
    }
  }

  {
    int ds = (rp - 1) / (fs1 / sfrq);

    assert(inbuflen >= ds);

    mem_ops<REAL>::move(inbuf, inbuf + nch * ds, nch*(inbuflen - ds));
    inbuflen -= ds;
    rp -= ds * (fs1 / sfrq);
  }

  return rv;

}

template<class REAL>
Upsampler<REAL>::~Upsampler()
{
  avs_free(f1order);
  avs_free(f1inc);
  avs_free(stage1[0]);
  avs_free(stage1);
  avs_free(stage2);
  avs_free(fft_ip);
  avs_free(fft_w);
  for (i = 0; i<nch; i++) avs_free(buf1[i]);
  avs_free(buf1);
  for (i = 0; i<nch; i++) avs_free(buf2[i]);
  avs_free(buf2);
  avs_free(inbuf);
  avs_free(outbuf);
  //free(rawoutbuf);
}


template<class REAL>
class Downsampler : public Resampler_i_base<REAL>
{
  //using Resampler_i_base<REAL>::peak; overridden here
  // Strict c++: access protected fields from a templated class (or use this->)
  using Resampler_i_base<REAL>::FFTFIRLEN;
  using Resampler_i_base<REAL>::sfrq;
  using Resampler_i_base<REAL>::dfrq;
  using Resampler_i_base<REAL>::nch;
  using Resampler_i_base<REAL>::DF;
  using Resampler_i_base<REAL>::AA;
  using Resampler_i_base<REAL>::make_outbuf;
  using Resampler_i_base<REAL>::make_inbuf;

private:
  int frqgcd,osf,fs1,fs2;
  REAL *stage1,**stage2;
  int n2,n2x,n2y,n1,n1b;
  int filter1len;
  int *f2order,*f2inc;
  int *fft_ip;// = NULL;
  REAL *fft_w;// = NULL;
  //unsigned char *rawinbuf,*rawoutbuf;
  REAL *inbuf,*outbuf;
  REAL **buf1,**buf2;
  int i,j;
  int spcount;// = 0;
  double peak;//=0;



    int n1b2;// = n1b/2;
    int rp;        // keep the location of the next samples to read in inbuf at fs1.
    int rps;       // the reminder obtained by dividing rp by (fs1/sfrq=osf).
    int rp2;       // keep the location of the next samples to read in buf2 at fs2.
    int ds;        // the number of samples to dispose next in sfrq.
    int nsmplwrt2; // actually number of samples to send stage2 filter .
    int s2p;       // the reminder obtained by dividing the samples output from stage1 filter by n1y*osf.
    int init,ending;
    int osc;
    REAL *bp; // the location of the next samples to read calculated with rp2
    int rps_backup,s2p_backup;
    int k,ch,p;
    int inbuflen;//=0;
    unsigned int sumread,sumwrite;
    int delay;// = 0;
    int delay2;
    REAL *op;

public:
  Downsampler(Resampler_base::CONFIG & c);

  ~Downsampler();
  /*
  {
	avs_free(stage1);
	avs_free(fft_ip);
	avs_free(fft_w);
	avs_free(f2order);
	avs_free(f2inc);
	avs_free(stage2[0]);
	avs_free(stage2);
	for(i=0;i<nch;i++) avs_free(buf1[i]);
	avs_free(buf1);
	for(i=0;i<nch;i++) avs_free(buf2[i]);
	avs_free(buf2);
	avs_free(inbuf);
	avs_free(outbuf);
	//free(rawoutbuf);
  }; // dtor
  */
  unsigned int Resample(REAL_inout * rawinbuf, unsigned int in_size, int ending);

};

template<class REAL>
Downsampler<REAL>::Downsampler(Resampler_base::CONFIG & c) : Resampler_i_base<REAL>(c)
{
  spcount = 0;
  peak = 0;
  fft_ip = 0;
  fft_w = 0;



  filter1len = FFTFIRLEN; /* stage 1 filter length */

                          /* Make stage 1 filter */

  {
    double aa = AA; /* stop band attenuation(dB) */
    double lpf, delta, d, df, alp, iza;
    int ipsize, wsize;

    frqgcd = gcd(sfrq, dfrq);

    if (dfrq / frqgcd == 1) osf = 1;
    else if (dfrq / frqgcd % 2 == 0) osf = 2;
    else if (dfrq / frqgcd % 3 == 0) osf = 3;
    else {
      //      fprintf(stderr,"Resampling from %dHz to %dHz is not supported.\n",sfrq,dfrq);
      //      fprintf(stderr,"%d/gcd(%d,%d)=%d must be divided by 2 or 3.\n",dfrq,sfrq,dfrq,dfrq/frqgcd);
      //      exit(-1);
      return;
    }

    fs1 = sfrq * osf;

    delta = pow(10.0, -aa / 20);
    if (aa <= 21) d = 0.9222; else d = (aa - 7.95) / 14.36;

    n1 = filter1len;
    for (i = 1;; i = i * 2)
    {
      n1 = filter1len * i;
      if (n1 % 2 == 0) n1--;
      df = (fs1*d) / (n1 - 1);
      lpf = (dfrq - df) / 2;
      if (df < DF) break;
    }

    alp = alpha(aa);

    iza = dbesi0(alp);

    for (n1b = 1; n1b<n1; n1b *= 2);
    n1b *= 2;

    stage1 = (REAL*)avs_malloc(sizeof(REAL)*n1b, 64);

    for (i = 0; i<n1b; i++) stage1[i] = 0;

    for (i = -(n1 / 2); i <= n1 / 2; i++) {
      stage1[i + n1 / 2] = win(i, n1, alp, iza)*hn_lpf(i, lpf, fs1)*fs1 / sfrq / n1b * 2;
    }

    ipsize = 2 + sqrt((double)n1b);
    fft_ip = (int*)avs_malloc(sizeof(int)*ipsize, 64);
    fft_ip[0] = 0;
    wsize = n1b / 2;
    fft_w = (REAL*)avs_malloc(sizeof(REAL)*wsize, 64);

    fft<REAL>::rdft(n1b, 1, stage1, fft_ip, fft_w);
  }

  /* Make stage 2 filter */

  if (osf == 1) {
    fs2 = sfrq / frqgcd * dfrq;
    n2 = 1;
    n2y = n2x = 1;
    f2order = (int*)avs_malloc(sizeof(int)*n2y, 64);
    f2order[0] = 0;
    f2inc = (int*)avs_malloc(sizeof(int)*n2y, 64);
    f2inc[0] = sfrq / dfrq;
    stage2 = (REAL**)avs_malloc(sizeof(REAL *)*n2y, 64);
    stage2[0] = (REAL*)avs_malloc(sizeof(REAL)*n2x*n2y, 64);
    stage2[0][0] = 1;
  }
  else {
    double aa = AA; /* stop band attenuation(dB) */
    double lpf, delta, d, df, alp, iza;
    double guard = 2;

    fs2 = sfrq / frqgcd * dfrq;

    df = (fs1 / 2 - sfrq / 2) * 2 / guard;
    lpf = sfrq / 2 + (fs1 / 2 - sfrq / 2) / guard;

    delta = pow(10.0, -aa / 20);
    if (aa <= 21) d = 0.9222; else d = (aa - 7.95) / 14.36;

    n2 = fs2 / df * d + 1;
    if (n2 % 2 == 0) n2++;

    alp = alpha(aa);
    iza = dbesi0(alp);

    n2y = fs2 / fs1; // The interval where the sample which isn't 0 in fs2 exists.
    n2x = n2 / n2y + 1;

    f2order = (int*)avs_malloc(sizeof(int)*n2y, 64);
    for (i = 0; i<n2y; i++) {
      f2order[i] = fs2 / fs1 - (i*(fs2 / dfrq)) % (fs2 / fs1);
      if (f2order[i] == fs2 / fs1) f2order[i] = 0;
    }

    f2inc = (int*)avs_malloc(sizeof(int)*n2y, 64);
    for (i = 0; i<n2y; i++) {
      f2inc[i] = (fs2 / dfrq - f2order[i]) / (fs2 / fs1) + 1;
      if (f2order[i + 1 == n2y ? 0 : i + 1] == 0) f2inc[i]--;
    }

    stage2 = (REAL**)avs_malloc(sizeof(REAL *)*n2y, 64);
    stage2[0] = (REAL*)avs_malloc(sizeof(REAL)*n2x*n2y, 64);

    for (i = 1; i<n2y; i++) {
      stage2[i] = &(stage2[0][n2x*i]);
      for (j = 0; j<n2x; j++) stage2[i][j] = 0;
    }

    for (i = -(n2 / 2); i <= n2 / 2; i++)
    {
      stage2[(i + n2 / 2) % n2y][(i + n2 / 2) / n2y] = win(i, n2, alp, iza)*hn_lpf(i, lpf, fs2)*fs2 / fs1;
    }
  }

  /* Apply filters */

  n1b2 = n1b / 2;
  inbuflen = 0;
  //    delay = 0;

  //    |....B....|....C....|   buf1      n1b2+n1b2
  //|.A.|....D....|             buf2  n2x+n1b2
  //
  // At first, take samples from inbuf and multiplied by osf, then write those into B.
  // Clear C.
  // Apply stage1-filter to B and C.
  // Add B to D.
  // Apply stage2-filter to A and D
  // Move last part of D to A.
  // Copy C to D.

  buf1 = (REAL**)avs_malloc(sizeof(REAL *)*nch, 64);
  for (i = 0; i<nch; i++)
    buf1[i] = (REAL*)avs_malloc(n1b * sizeof(REAL), 64);

  buf2 = (REAL**)avs_malloc(sizeof(REAL *)*nch, 64);
  for (i = 0; i<nch; i++) {
    buf2[i] = (REAL*)avs_malloc(sizeof(REAL)*(n2x + 1 + n1b2), 64);
    for (j = 0; j<n2x + n1b2; j++) buf2[i][j] = 0;
  }

  //rawoutbuf = (unsigned char*)malloc(dbps*nch*((double)n1b2*sfrq/dfrq+1));
  inbuf = (REAL*)avs_malloc(nch*(n1b2 / osf + osf + 1) * sizeof(REAL), 64);
  outbuf = (REAL*)avs_malloc(sizeof(REAL)*nch*((double)n1b2*sfrq / dfrq + 1), 64);

  op = outbuf;

  s2p = 0;
  rp = 0;
  rps = 0;
  ds = 0;
  osc = 0;
  rp2 = 0;

  init = 1;
  ending = 0;
  delay = (double)n1 / 2 / ((double)fs1 / dfrq) + (double)n2 / 2 / ((double)fs2 / dfrq);
  delay2 = delay * nch;

  sumread = sumwrite = 0;



}; // ctor


template<class REAL>
Downsampler<REAL>::~Downsampler()
{
  avs_free(stage1);
  avs_free(fft_ip);
  avs_free(fft_w);
  avs_free(f2order);
  avs_free(f2inc);
  avs_free(stage2[0]);
  avs_free(stage2);
  for (i = 0; i<nch; i++) avs_free(buf1[i]);
  avs_free(buf1);
  for (i = 0; i<nch; i++) avs_free(buf2[i]);
  avs_free(buf2);
  avs_free(inbuf);
  avs_free(outbuf);
  //free(rawoutbuf);
}; // dtor

template<class REAL>
unsigned int Downsampler<REAL>::Resample(REAL_inout * rawinbuf, unsigned int in_size, int ending) {
  unsigned int rv;
  int nsmplread;
  int toberead;

  toberead = (n1b2 - rps - 1) / osf + 1;

  if (!ending)
  {
    rv = nch * toberead;
    if (in_size<rv) return 0;
    nsmplread = toberead;
  }
  else
  {
    nsmplread = in_size / nch;
    rv = nsmplread * nch;
  }

  make_inbuf(nsmplread, inbuflen, rawinbuf, inbuf, toberead);

  sumread += nsmplread;

  rps_backup = rps;
  s2p_backup = s2p;

  for (ch = 0; ch<nch; ch++)
  {
    rps = rps_backup;

    for (k = 0; k<rps; k++) buf1[ch][k] = 0;

    for (i = rps, j = 0; i<n1b2; i += osf, j++)
    {
      assert(j < ((n1b2 - rps - 1) / osf + 1));

      buf1[ch][i] = inbuf[j*nch + ch];

      for (k = i + 1; k<i + osf; k++) buf1[ch][k] = 0;
    }

    assert(j == ((n1b2 - rps - 1) / osf + 1));

    for (k = n1b2; k<n1b; k++) buf1[ch][k] = 0;

    rps = i - n1b2;
    rp += j;

    fft<REAL>::rdft(n1b, 1, buf1[ch], fft_ip, fft_w);

    buf1[ch][0] = stage1[0] * buf1[ch][0];
    buf1[ch][1] = stage1[1] * buf1[ch][1];

    for (i = 1; i<n1b2; i++)
    {
      REAL re, im;

      re = stage1[i * 2] * buf1[ch][i * 2] - stage1[i * 2 + 1] * buf1[ch][i * 2 + 1];
      im = stage1[i * 2 + 1] * buf1[ch][i * 2] + stage1[i * 2] * buf1[ch][i * 2 + 1];

      buf1[ch][i * 2] = re;
      buf1[ch][i * 2 + 1] = im;
    }

    fft<REAL>::rdft(n1b, -1, buf1[ch], fft_ip, fft_w);

    for (i = 0; i<n1b2; i++) {
      buf2[ch][n2x + 1 + i] += buf1[ch][i];
    }

    {
      int t1 = rp2 / (fs2 / fs1);
      if (rp2 % (fs2 / fs1) != 0) t1++;

      bp = &(buf2[ch][t1]);
    }

    s2p = s2p_backup;

    for (p = 0; bp - buf2[ch]<n1b2 + 1; p++)
    {
      REAL tmp = 0;
      REAL *bp2;
      int s2o;

      bp2 = bp;
      s2o = f2order[s2p];
      bp += f2inc[s2p];
      s2p++;

      if (s2p == n2y) s2p = 0;

      assert((bp2 - &(buf2[ch][0]))*(fs2 / fs1) - (rp2 + p * (fs2 / dfrq)) == s2o);

      for (i = 0; i<n2x; i++)
        tmp += stage2[s2o][i] * *bp2++;

      op[p*nch + ch] = tmp;
    }

    nsmplwrt2 = p;
  }

  rp2 += nsmplwrt2 * (fs2 / dfrq);

  make_outbuf(nsmplwrt2, outbuf, delay2);

  if (!init) {
    if (ending) {
      if ((double)sumread*dfrq / this->sfrq + 2 > sumwrite + nsmplwrt2) {

        sumwrite += nsmplwrt2;
      }
      else {

        return rv;
      }
    }
    else {

      sumwrite += nsmplwrt2;
    }
  }
  else {
    if (nsmplwrt2 < delay) {
      delay -= nsmplwrt2;
    }
    else {
      if (ending) {
        if ((double)sumread*this->dfrq / this->sfrq + 2 > sumwrite + nsmplwrt2 - delay) {

          sumwrite += nsmplwrt2 - delay;
        }
        else {

          return rv;
        }
      }
      else {

        sumwrite += nsmplwrt2 - delay;
        init = 0;
      }
    }
  }

  {
    int ds = (rp2 - 1) / (fs2 / fs1);

    if (ds > n1b2) ds = n1b2;

    for (ch = 0; ch<nch; ch++)
      mem_ops<REAL>::move(buf2[ch], buf2[ch] + ds, n2x + 1 + n1b2 - ds);

    rp2 -= ds * (fs2 / fs1);
  }

  for (ch = 0; ch<nch; ch++)
    mem_ops<REAL>::copy(buf2[ch] + n2x + 1, buf1[ch] + n1b2, n1b2);

  return rv;
}

Resampler_base::Resampler_base(const Resampler_base::CONFIG & c)
{
	if (c.fast)
	{
		AA = 96;
		DF = 8000;
		FFTFIRLEN = 1024;
	}
	else
	{
		AA=120;
		DF=100;
		FFTFIRLEN=16384;
	}
/*
#else
	AA=170;
	DF=100;
	FFTFIRLEN=65536;
#endif
	*/

	nch=c.nch;
	sfrq=c.sfrq;
	dfrq=c.dfrq;

	double noiseamp = 0.18;
	//double att=0;
	gain=1;//pow(10.0,-att/20);

}

Resampler_base * Resampler_base::Create(Resampler_base::CONFIG & c)
{
	if (!CanResample(c.sfrq,c.dfrq)) return 0;

/*	if (c.math)
	{
		if (c.sfrq < c.dfrq) return new Upsampler<double>(c);
		else if (c.sfrq > c.dfrq) return new Downsampler<double>(c);
		else return 0;
	}
	else*/
	{
		if (c.sfrq < c.dfrq) return new Upsampler<float>(c);
		else if (c.sfrq > c.dfrq) return new Downsampler<float>(c);
		else return 0;
	}
}

Resampler_base *SSRC_create(int sfrq, int dfrq, int nch, int dither, int pdf, int fast)
{
  Resampler_base::CONFIG c(sfrq, dfrq, nch, dither, pdf, fast);
  return Resampler_base::Create(c);
}

