Great idea. DMDA + `Vec` is *the* core PETSc workflow, and learning it by **concrete operations** is exactly the right approach.

Below are **copy-pasteable, minimal, progressively richer examples** showing how to **set/get elements, do algebra, and use DMDA-aware access patterns**. All examples are correct in **serial and MPI**.

I’ll assume a **2D DMDA**, since that’s what you were using before.

---

# Example 0 — Skeleton (DMDA + global Vec)

All later examples reuse this pattern.

```c
#include <petscdm.h>
#include <petscdmda.h>
#include <petscvec.h>

int main(int argc, char **argv)
{
    PetscInitialize(&argc, &argv, NULL, NULL);

    DM da;
    Vec x, y;

    PetscInt mx = 8, my = 6;

    DMDACreate2d(PETSC_COMM_WORLD,
        DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
        DMDA_STENCIL_STAR,
        mx, my,
        PETSC_DECIDE, PETSC_DECIDE,
        1, 1,
        NULL, NULL,
        &da);

    DMSetUp(da);

    DMCreateGlobalVector(da, &x);
    DMCreateGlobalVector(da, &y);

    /* --- insert examples here --- */

    VecDestroy(&x);
    VecDestroy(&y);
    DMDestroy(&da);
    PetscFinalize();
    return 0;
}
```

---

# Example 1 — Set all elements

```c
VecSet(x, 2.0);   /* x_i = 2 for all i */
```

---

# Example 2 — Setting a single element (global index)

⚠️ This is **legal but discouraged** for structured grids.

```c
PetscInt idx = 3;
PetscScalar val = 5.0;

VecSetValue(x, idx, val, INSERT_VALUES);
VecAssemblyBegin(x);
VecAssemblyEnd(x);
```

This uses **global numbering**, which is usually *not* what you want with DMDA.

---

# Example 3 — Correct DMDA way: set values via grid indices

✅ **This is the preferred method**

```c
PetscInt xs, ys, xm, ym, i, j;
PetscScalar **xarr;

DMDAVecGetArray(da, x, &xarr);
DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL);

for (j = ys; j < ys + ym; j++) {
    for (i = xs; i < xs + xm; i++) {
        xarr[j][i] = i + 10.0 * j;
    }
}

DMDAVecRestoreArray(da, x, &xarr);
```

✔ parallel safe
✔ cache friendly
✔ no assembly needed

---

# Example 4 — Getting values (local access)

```c
PetscScalar **xarr;
DMDAVecGetArrayRead(da, x, &xarr);

PetscInt xs, ys, xm, ym;
DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL);

for (PetscInt j = ys; j < ys + ym; j++)
    for (PetscInt i = xs; i < xs + xm; i++)
        PetscPrintf(PETSC_COMM_SELF,
            "Rank local (%d,%d) = %g\n",
            i, j, (double)xarr[j][i]);

DMDAVecRestoreArrayRead(da, x, &xarr);
```

---

# Example 5 — Vec addition

### y = x + y

```c
VecAXPY(y, 1.0, x);
```

### y = 2x + y

```c
VecAXPY(y, 2.0, x);
```

---

# Example 6 — Vec linear combination

### y = αx + βy

```c
PetscScalar alpha = 2.0, beta = -1.0;
VecAXPBY(y, alpha, beta, x);
```

---

# Example 7 — Pointwise multiplication

### yᵢ = xᵢ · yᵢ

```c
VecPointwiseMult(y, x, y);
```

### yᵢ = xᵢ / yᵢ

```c
VecPointwiseDivide(y, x, y);
```

---

# Example 8 — Norms and dot products

```c
PetscReal norm2;
VecNorm(x, NORM_2, &norm2);

PetscScalar dot;
VecDot(x, y, &dot);
```

---

# Example 9 — Scaling and shifting

```c
VecScale(x, 0.5);    /* x = 0.5 x */
VecShift(x, 1.0);   /* x = x + 1 */
```

---

# Example 10 — Copy and swap

```c
VecCopy(x, y);   /* y = x */
VecSwap(x, y);   /* swap contents */
```

---

# Example 11 — Local vector + ghost points

This is **crucial for stencil operations**.

```c
Vec xlocal;
DMCreateLocalVector(da, &xlocal);

/* Fill global x somehow */
VecSet(x, 1.0);

/* Update ghost values */
DMGlobalToLocalBegin(da, x, INSERT_VALUES, xlocal);
DMGlobalToLocalEnd(da, x, INSERT_VALUES, xlocal);

PetscScalar **xl;
DMDAVecGetArrayRead(da, xlocal, &xl);

/* Now xl includes ghost points */
DMDAVecRestoreArrayRead(da, xlocal, &xl);

VecDestroy(&xlocal);
```

---

# Example 12 — Zeroing boundaries (classic PDE task)

```c
PetscInt xs, ys, xm, ym;
PetscScalar **xarr;

DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL);
DMDAVecGetArray(da, x, &xarr);

for (PetscInt j = ys; j < ys + ym; j++) {
    for (PetscInt i = xs; i < xs + xm; i++) {
        if (i == 0 || j == 0 || i == mx-1 || j == my-1)
            xarr[j][i] = 0.0;
    }
}

DMDAVecRestoreArray(da, x, &xarr);
```

---

# Mental model (important)

| Concept              | Think of it as                   |
| -------------------- | -------------------------------- |
| `Vec`                | Distributed 1D storage           |
| `DMDA`               | Mapping between grid ↔ Vec       |
| `DMDAVecGetArray`    | Safe local multidimensional view |
| `VecAXPY`, `VecNorm` | MPI-aware BLAS                   |

If you follow **DM creates Vec, DMDA accesses Vec**, you’ll never hit type/layout bugs.

---

## Want next?

Given your earlier interest in eigenproblems and Hamiltonians, the natural next examples are:

1. 🔁 **Time stepping: x ← Ax**
2. 🧮 **Laplacian stencil using ghost points**
3. 🔍 **Lowest eigenvectors → density**
4. ⏱️ **Profiling Vec operations**
5. 🔄 **Periodic BCs in DMDA Vec**

Tell me which one and I’ll give a **ready-to-run example**.
