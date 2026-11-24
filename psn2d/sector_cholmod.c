/* Optional SuiteSparse bridge for finite-sector diffusion blocks. */
#include <cholmod.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    cholmod_common common;
    cholmod_factor *factor;
    cholmod_dense *x, *work, *extra;
} PsnCholmod;

void psn_cholmod_free(PsnCholmod *handle)
{
    if (handle == NULL) return;
    cholmod_free_dense(&handle->x, &handle->common);
    cholmod_free_dense(&handle->work, &handle->common);
    cholmod_free_dense(&handle->extra, &handle->common);
    cholmod_free_factor(&handle->factor, &handle->common);
    cholmod_finish(&handle->common);
    free(handle);
}

static PsnCholmod *new_handle(void)
{
    PsnCholmod *handle = calloc(1, sizeof(*handle));
    if (handle == NULL) return NULL;
    if (!cholmod_start(&handle->common)) {
        free(handle);
        return NULL;
    }
    handle->common.print = 0;
    handle->common.supernodal = CHOLMOD_SUPERNODAL;
    return handle;
}

static cholmod_sparse sparse(size_t n, size_t nnz, int *p, int *i, double *x)
{
    cholmod_sparse a = {0};
    a.nrow = a.ncol = n;
    a.nzmax = nnz;
    a.p = p;
    a.i = i;
    a.x = x;
    a.stype = -1;
    a.itype = CHOLMOD_INT;
    a.xtype = x == NULL ? CHOLMOD_PATTERN : CHOLMOD_REAL;
    a.dtype = CHOLMOD_DOUBLE;
    a.sorted = a.packed = 1;
    return a;
}

PsnCholmod *psn_cholmod_analyze(size_t n, size_t nnz, int *p, int *i)
{
    PsnCholmod *handle = new_handle();
    if (handle == NULL) return NULL;
    handle->common.nmethods = 1;
    handle->common.method[0].ordering = CHOLMOD_METIS;
    cholmod_sparse a = sparse(n, nnz, p, i, NULL);
    handle->factor = cholmod_analyze(&a, &handle->common);
    if (handle->factor == NULL) {
        psn_cholmod_free(handle);
        return NULL;
    }
    cholmod_free_work(&handle->common);
    return handle;
}

PsnCholmod *psn_cholmod_factor(PsnCholmod *symbolic, size_t n, size_t nnz,
                               int *p, int *i, double *x, size_t *bytes)
{
    PsnCholmod *handle = new_handle();
    if (handle == NULL) return NULL;
    cholmod_sparse a = sparse(n, nnz, p, i, x);
    handle->factor = cholmod_copy_factor(symbolic->factor, &handle->common);
    if (handle->factor == NULL || !cholmod_factorize(&a, handle->factor, &handle->common)
        || handle->factor->minor != n) {
        psn_cholmod_free(handle);
        return NULL;
    }
    cholmod_free_work(&handle->common);
    *bytes = sizeof(*handle) + handle->common.memory_inuse + 4 * n * sizeof(double);
    return handle;
}

int psn_cholmod_solve(PsnCholmod *handle, double *rhs, double *result)
{
    const size_t n = handle->factor->n;
    cholmod_dense b = {0};
    b.nrow = b.nzmax = b.d = n;
    b.ncol = 1;
    b.x = rhs;
    b.xtype = CHOLMOD_REAL;
    b.dtype = CHOLMOD_DOUBLE;
    if (!cholmod_solve2(CHOLMOD_A, handle->factor, &b, NULL, &handle->x, NULL,
                        &handle->work, &handle->extra, &handle->common)) return 0;
    memcpy(result, handle->x->x, n * sizeof(double));
    return 1;
}
