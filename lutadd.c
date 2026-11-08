/*
 * Lookup-table matrix multiply with an approximate adder in the accumulator.
 *
 * Compiled once per adder with -DADDER_FILE='"path/add16u_XXX.c"' -DADDER=add16u_XXX.
 * Positive and negative products are summed in two unsigned accumulators (natural for
 * sign-magnitude weights). Each accumulator keeps its low 16 bits in the approximate 16-bit adder
 * and counts the carry-out (bit 16 of the adder's output) exactly in the upper bits:
 *     r = ADDER(lo, p);  lo = r & 0xFFFF;  hi += r >> 16;
 * so only the 16-bit adder is approximate, as in an accelerator whose wide accumulator is built
 * around a 16-bit adder. Every product is added, zeros included, because an approximate adder can
 * change the sum even when adding zero.
 */
#include <stdint.h>
#include <stdlib.h>
#include ADDER_FILE

void lut_matmul_add(const uint8_t *x, const uint8_t *mag, const int8_t *sgn, const int32_t *lut,
                    int64_t *out, int N, int K, int J)
{
    #pragma omp parallel
    {
        uint64_t *plo = malloc(8 * J), *phi = malloc(8 * J), *nlo = malloc(8 * J), *nhi = malloc(8 * J);
        #pragma omp for schedule(static)
        for (int n = 0; n < N; n++) {
            for (int j = 0; j < J; j++) plo[j] = phi[j] = nlo[j] = nhi[j] = 0;
            const uint8_t *xr = x + (int64_t)n * K;
            for (int k = 0; k < K; k++) {
                const int32_t *row = lut + xr[k] * 256;
                const uint8_t *m = mag + (int64_t)k * J;
                const int8_t *s = sgn + (int64_t)k * J;
                for (int j = 0; j < J; j++) {
                    if (s[j] == 0) continue;
                    uint64_t p = (uint64_t)row[m[j]];
                    if (s[j] > 0) { uint64_t r = ADDER(plo[j], p); plo[j] = r & 0xFFFF; phi[j] += r >> 16; }
                    else          { uint64_t r = ADDER(nlo[j], p); nlo[j] = r & 0xFFFF; nhi[j] += r >> 16; }
                }
            }
            int64_t *o = out + (int64_t)n * J;
            for (int j = 0; j < J; j++)
                o[j] = (int64_t)((phi[j] << 16) + plo[j]) - (int64_t)((nhi[j] << 16) + nlo[j]);
        }
        free(plo); free(phi); free(nlo); free(nhi);
    }
}
