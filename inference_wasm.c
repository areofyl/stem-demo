/*
 * inference_wasm.c — browser-compatible stem separator
 *
 * same micro GRU models as inference.c, but:
 * - no libsndfile (browser decodes audio)
 * - no OpenBLAS (simple C loops, WASM SIMD friendly)
 * - no pthreads (runs sequentially, 4 calls from JS)
 * - exported via emscripten for JS interop
 *
 * build: emcc inference_wasm.c -O3 -s WASM=1 -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap"]' \
 *        -s ALLOW_MEMORY_GROWTH=1 -s MODULARIZE=1 -s EXPORT_NAME=StemModule \
 *        -o stem.js
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#else
#define EMSCRIPTEN_KEEPALIVE
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---- basic ops (no BLAS) ---- */

static void matvec(float *y, const float *A, const float *x, int M, int N)
{
	for (int i = 0; i < M; i++) {
		float sum = 0;
		for (int j = 0; j < N; j++)
			sum += A[i * N + j] * x[j];
		y[i] = sum;
	}
}

static void vec_add(float *y, const float *x, int n)
{
	for (int i = 0; i < n; i++)
		y[i] += x[i];
}

static void linear(float *y, const float *x, const float *w, const float *b,
                    int M, int K, int N)
{
	/* y = x * w^T, where x is [M,K], w is [N,K], result is [M,N] */
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			float sum = 0;
			for (int k = 0; k < K; k++)
				sum += x[i * K + k] * w[j * K + k];
			y[i * N + j] = sum;
		}
	}
	if (b)
		for (int i = 0; i < M; i++)
			vec_add(y + i * N, b, N);
}

static void relu(float *x, int n)
{
	for (int i = 0; i < n; i++)
		if (x[i] < 0) x[i] = 0;
}

static void sigmoid_inplace(float *x, int n)
{
	for (int i = 0; i < n; i++)
		x[i] = 1.0f / (1.0f + expf(-x[i]));
}

static void gru_step(float *h, const float *x_t, int input_dim, int hidden,
                     const float *w_ih, const float *w_hh,
                     const float *b_ih, const float *b_hh)
{
	int h3 = 3 * hidden;
	float *gates_i = (float *)calloc(h3, sizeof(float));
	float *gates_h = (float *)calloc(h3, sizeof(float));

	matvec(gates_i, w_ih, x_t, h3, input_dim);
	vec_add(gates_i, b_ih, h3);
	matvec(gates_h, w_hh, h, h3, hidden);
	vec_add(gates_h, b_hh, h3);

	for (int i = 0; i < hidden; i++) {
		float r = 1.0f / (1.0f + expf(-(gates_i[i] + gates_h[i])));
		float z = 1.0f / (1.0f + expf(-(gates_i[hidden + i] + gates_h[hidden + i])));
		float n = tanhf(gates_i[2 * hidden + i] + r * gates_h[2 * hidden + i]);
		h[i] = (1.0f - z) * n + z * h[i];
	}

	free(gates_i);
	free(gates_h);
}

/* ---- FFT ---- */

typedef struct { float re, im; } cpx;

static void fft(cpx *buf, int n, int inverse)
{
	for (int i = 1, j = 0; i < n; i++) {
		int bit = n >> 1;
		for (; j & bit; bit >>= 1) j ^= bit;
		j ^= bit;
		if (i < j) { cpx tmp = buf[i]; buf[i] = buf[j]; buf[j] = tmp; }
	}
	for (int len = 2; len <= n; len <<= 1) {
		float ang = 2.0f * (float)M_PI / len * (inverse ? -1 : 1);
		cpx wlen = { cosf(ang), sinf(ang) };
		for (int i = 0; i < n; i += len) {
			cpx w = { 1.0f, 0.0f };
			for (int j = 0; j < len / 2; j++) {
				cpx u = buf[i + j];
				cpx v = {
					buf[i+j+len/2].re * w.re - buf[i+j+len/2].im * w.im,
					buf[i+j+len/2].re * w.im + buf[i+j+len/2].im * w.re
				};
				buf[i+j] = (cpx){ u.re + v.re, u.im + v.im };
				buf[i+j+len/2] = (cpx){ u.re - v.re, u.im - v.im };
				float wr = w.re * wlen.re - w.im * wlen.im;
				w.im = w.re * wlen.im + w.im * wlen.re;
				w.re = wr;
			}
		}
	}
	if (inverse) {
		float s = 1.0f / n;
		for (int i = 0; i < n; i++) { buf[i].re *= s; buf[i].im *= s; }
	}
}

/* ---- model ---- */

typedef struct {
	int n_fft, hop, hidden, n_gru_layers;
	int n_bins;
	float *enc_w1, *enc_b1, *enc_w2, *enc_b2;
	float *gru_wih[2], *gru_whh[2];
	float *gru_bih[2], *gru_bhh[2];
	float *dec_w1, *dec_b1, *dec_w2, *dec_b2;
} MicroModel;

#define MAX_MODELS 4
static MicroModel models[MAX_MODELS];
static int model_loaded[MAX_MODELS] = {0};

static float *read_tensor_from_mem(const unsigned char **ptr)
{
	int name_len;
	memcpy(&name_len, *ptr, 4); *ptr += 4;
	*ptr += name_len; /* skip name */
	int ndim;
	memcpy(&ndim, *ptr, 4); *ptr += 4;
	int size = 1;
	for (int i = 0; i < ndim; i++) {
		int s;
		memcpy(&s, *ptr, 4); *ptr += 4;
		size *= s;
	}
	float *data = (float *)malloc(size * sizeof(float));
	memcpy(data, *ptr, size * sizeof(float));
	*ptr += size * sizeof(float);
	return data;
}

EMSCRIPTEN_KEEPALIVE
int load_model_from_buffer(int idx, const unsigned char *data, int len)
{
	if (idx < 0 || idx >= MAX_MODELS) return -1;

	const unsigned char *ptr = data;

	/* check magic */
	if (memcmp(ptr, "MICR", 4) != 0) return -1;
	ptr += 4;

	MicroModel *m = &models[idx];
	memcpy(&m->n_fft, ptr, 4); ptr += 4;
	memcpy(&m->hop, ptr, 4); ptr += 4;
	memcpy(&m->hidden, ptr, 4); ptr += 4;
	memcpy(&m->n_gru_layers, ptr, 4); ptr += 4;
	m->n_bins = m->n_fft / 2 + 1;

	int n_tensors;
	memcpy(&n_tensors, ptr, 4); ptr += 4;

	m->enc_w1 = read_tensor_from_mem(&ptr); m->enc_b1 = read_tensor_from_mem(&ptr);
	m->enc_w2 = read_tensor_from_mem(&ptr); m->enc_b2 = read_tensor_from_mem(&ptr);
	for (int l = 0; l < 2; l++) {
		m->gru_wih[l] = read_tensor_from_mem(&ptr);
		m->gru_whh[l] = read_tensor_from_mem(&ptr);
		m->gru_bih[l] = read_tensor_from_mem(&ptr);
		m->gru_bhh[l] = read_tensor_from_mem(&ptr);
	}
	m->dec_w1 = read_tensor_from_mem(&ptr); m->dec_b1 = read_tensor_from_mem(&ptr);
	m->dec_w2 = read_tensor_from_mem(&ptr); m->dec_b2 = read_tensor_from_mem(&ptr);

	model_loaded[idx] = 1;
	return 0;
}

/* output buffer — reused between calls */
static float *out_buf = NULL;
static int out_buf_len = 0;

EMSCRIPTEN_KEEPALIVE
float *get_output_ptr(void)
{
	return out_buf;
}

EMSCRIPTEN_KEEPALIVE
int separate(int model_idx, const float *input, int n_samples)
{
	if (model_idx < 0 || model_idx >= MAX_MODELS || !model_loaded[model_idx])
		return -1;

	MicroModel *m = &models[model_idx];
	int n_fft = m->n_fft, hop = m->hop, n_bins = m->n_bins, hidden = m->hidden;

	/* hann window */
	float *win = (float *)malloc(n_fft * sizeof(float));
	for (int i = 0; i < n_fft; i++)
		win[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / n_fft));

	/* STFT */
	int T = (n_samples - n_fft) / hop + 1;
	float *mag = (float *)calloc(n_bins * T, sizeof(float));
	float *ph_re = (float *)calloc(n_bins * T, sizeof(float));
	float *ph_im = (float *)calloc(n_bins * T, sizeof(float));
	cpx *fb = (cpx *)calloc(n_fft, sizeof(cpx));

	for (int t = 0; t < T; t++) {
		for (int i = 0; i < n_fft; i++)
			fb[i] = (cpx){ input[t * hop + i] * win[i], 0 };
		fft(fb, n_fft, 0);
		for (int f = 0; f < n_bins; f++) {
			ph_re[f * T + t] = fb[f].re;
			ph_im[f * T + t] = fb[f].im;
			mag[f * T + t] = sqrtf(fb[f].re * fb[f].re + fb[f].im * fb[f].im);
		}
	}

	/* process frame-by-frame: encoder -> GRU -> decoder */
	float *h0 = (float *)calloc(hidden, sizeof(float));
	float *h1 = (float *)calloc(hidden, sizeof(float));
	float *enc = (float *)calloc(hidden, sizeof(float));
	float *enc2 = (float *)calloc(hidden, sizeof(float));
	float *dec = (float *)calloc(n_bins, sizeof(float));
	float *dec2 = (float *)calloc(hidden, sizeof(float));
	float *fin = (float *)calloc(n_bins, sizeof(float));
	float *mask = (float *)calloc(n_bins * T, sizeof(float));

	for (int t = 0; t < T; t++) {
		for (int f = 0; f < n_bins; f++)
			fin[f] = mag[f * T + t];

		linear(enc, fin, m->enc_w1, m->enc_b1, 1, n_bins, hidden);
		relu(enc, hidden);
		linear(enc2, enc, m->enc_w2, m->enc_b2, 1, hidden, hidden);
		relu(enc2, hidden);

		gru_step(h0, enc2, hidden, hidden,
		         m->gru_wih[0], m->gru_whh[0], m->gru_bih[0], m->gru_bhh[0]);
		gru_step(h1, h0, hidden, hidden,
		         m->gru_wih[1], m->gru_whh[1], m->gru_bih[1], m->gru_bhh[1]);

		linear(dec2, h1, m->dec_w1, m->dec_b1, 1, hidden, hidden);
		relu(dec2, hidden);
		linear(dec, dec2, m->dec_w2, m->dec_b2, 1, hidden, n_bins);
		sigmoid_inplace(dec, n_bins);

		for (int f = 0; f < n_bins; f++)
			mask[f * T + t] = dec[f];
	}

	/* allocate output */
	if (out_buf_len < n_samples) {
		free(out_buf);
		out_buf = (float *)calloc(n_samples, sizeof(float));
		out_buf_len = n_samples;
	} else {
		memset(out_buf, 0, n_samples * sizeof(float));
	}

	/* apply mask and ISTFT */
	float *ws = (float *)calloc(n_samples, sizeof(float));

	for (int t = 0; t < T; t++) {
		for (int f = 0; f < n_bins; f++) {
			float mv = mask[f * T + t];
			fb[f] = (cpx){ ph_re[f * T + t] * mv, ph_im[f * T + t] * mv };
		}
		for (int f = 1; f < n_bins - 1; f++)
			fb[n_fft - f] = (cpx){ fb[f].re, -fb[f].im };

		fft(fb, n_fft, 1);

		int st = t * hop;
		for (int i = 0; i < n_fft && st + i < n_samples; i++) {
			out_buf[st + i] += fb[i].re * win[i];
			ws[st + i] += win[i] * win[i];
		}
	}

	for (int i = 0; i < n_samples; i++)
		if (ws[i] > 1e-8f) out_buf[i] /= ws[i];

	free(win); free(mag); free(ph_re); free(ph_im); free(fb);
	free(h0); free(h1); free(enc); free(enc2);
	free(dec); free(dec2); free(fin); free(mask); free(ws);

	return 0;
}
