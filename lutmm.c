/*
 * Lookup-table matrix multiply for approximate 8-bit multipliers.
 *
 *   out[n][j] = sum_k sgn[k][j] * lut[x[n][k]][mag[k][j]]
 *
 * x:   (N, K) uint8 activations          mag: (K, J) uint8 weight magnitudes
 * sgn: (K, J) int8 weight signs (-1/0/1)  lut: 256 x 256 int32 products of one multiplier
 * out: (N, J) int32 accumulators (K * 65535 fits easily for K < 30,000)
 *
 * When the multiplier gives 0 for every product with a 0 activation (lut row 0 is all
 * zero, skip_zero = 1), zero activations are skipped; the result is identical.
 * Build: gcc -O3 -march=native -fopenmp -shared -fPIC lutmm.c -o _lutmm.so
 */
#include <stdint.h>
#include <string.h>

void lut_matmul(const uint8_t *x, const uint8_t *mag, const int8_t *sgn, const int32_t *lut,
                int32_t *out, int N, int K, int J, int skip_zero)
{
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        int32_t *o = out + (int64_t)n * J;
        memset(o, 0, sizeof(int32_t) * J);
        const uint8_t *xr = x + (int64_t)n * K;
        for (int k = 0; k < K; k++) {
            int a = xr[k];
            if (skip_zero && a == 0) continue;
            const int32_t *row = lut + a * 256;
            const uint8_t *m = mag + (int64_t)k * J;
            const int8_t *s = sgn + (int64_t)k * J;
            for (int j = 0; j < J; j++) o[j] += s[j] * row[m[j]];
        }
    }
}
