This is an **excellent PETSc design question**, and getting it right early will save you a lot of pain later.

Short answer up front:

> **Pass `Vec` by value (i.e. `Vec x`), *not* `Vec *x`, in almost all cases.**

Then I’ll explain **why**, **when you *do* need `Vec*`**, and give **correct function patterns** you can copy.

---

## 1️⃣ What a `Vec` actually is (key insight)

In PETSc:

```c
typedef struct _p_Vec *Vec;
```

So:

* `Vec` is **already a pointer**
* Passing `Vec x` passes a pointer to the PETSc object
* You are *not copying vector data*

Therefore:

```c
void f(Vec x)
```

✔ modifies the same underlying vector
✔ cheap (pointer copy)
✔ PETSc-idiomatic

---

## 2️⃣ The 95% rule (what you should usually do)

### ✅ If the function **operates on an existing vector**

Use:

```c
PetscErrorCode MyVecOp(DM da, Vec x)
```

Example: set values via DMDA

```c
PetscErrorCode SetByCoordinates(DM da, Vec x)
{
    PetscInt xs, ys, xm, ym, i, j;
    PetscScalar **xa;

    DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL);
    DMDAVecGetArray(da, x, &xa);

    for (j = ys; j < ys + ym; j++)
        for (i = xs; i < xs + xm; i++)
            xa[j][i] = i + 10.0*j;

    DMDAVecRestoreArray(da, x, &xa);
    return 0;
}
```

Call site:

```c
SetByCoordinates(da, x);
```

✔ correct
✔ parallel-safe
✔ no pointer confusion

---

## 3️⃣ When you MUST use `Vec *`

Only when the function **creates, destroys, or reassigns** the vector handle.

### Case A — Creating a vector

```c
PetscErrorCode CreateMyVector(DM da, Vec *x)
{
    DMCreateGlobalVector(da, x);
    VecSet(*x, 0.0);
    return 0;
}
```

Call:

```c
Vec x;
CreateMyVector(da, &x);
```

---

### Case B — Destroying a vector

```c
PetscErrorCode DestroyMyVector(Vec *x)
{
    VecDestroy(x);   /* PETSc sets *x = NULL */
    return 0;
}
```

---

### Case C — Replacing the vector

```c
PetscErrorCode ReplaceVector(DM da, Vec *x)
{
    Vec y;
    DMCreateGlobalVector(da, &y);
    VecSet(y, 1.0);

    VecDestroy(x);
    *x = y;
    return 0;
}
```

---

## 4️⃣ What NOT to do (very common mistake)

❌ Don’t do this:

```c
void BadFunc(Vec *x)
{
    VecSet(*x, 1.0);
}
```

Why?

* It **adds indirection for no benefit**
* Encourages accidental misuse
* Non-idiomatic PETSc style

Use:

```c
void GoodFunc(Vec x)
{
    VecSet(x, 1.0);
}
```

---

## 5️⃣ DMDA + Vec function patterns (cheat sheet)

| Operation          | Signature                                 |
| ------------------ | ----------------------------------------- |
| Read/write entries | `f(DM da, Vec x)`                         |
| Read-only          | `f(DM da, Vec x)` + `DMDAVecGetArrayRead` |
| Combine vectors    | `f(Vec x, Vec y)`                         |
| Compute scalar     | `f(Vec x, PetscReal *out)`                |
| Create vector      | `f(DM da, Vec *x)`                        |
| Destroy vector     | `f(Vec *x)`                               |

---

## 6️⃣ Example: library-style Vec ops

### Scale + shift

```c
PetscErrorCode ScaleShift(Vec x, PetscScalar a, PetscScalar b)
{
    VecScale(x, a);
    VecShift(x, b);
    return 0;
}
```

---

### Dot product

```c
PetscErrorCode MyDot(Vec x, Vec y, PetscScalar *dot)
{
    VecDot(x, y, dot);
    return 0;
}
```

---

### Create + fill

```c
PetscErrorCode CreateAndFill(DM da, Vec *x)
{
    DMCreateGlobalVector(da, x);
    VecSet(*x, 1.0);
    return 0;
}
```

---

## 7️⃣ PETSc style tips (important)

1. **Always return `PetscErrorCode`**
2. Use `CHKERRQ(ierr)` in real code
3. Never hide `VecAssembly` inside generic functions unless clearly documented
4. Let **DM own layout**, not utility functions

---

## 8️⃣ Rule of thumb (memorize this)

> **If your function needs to change the *contents* of a Vec → `Vec x`
> If it needs to change the *handle* → `Vec *x`**

This rule applies equally to `Mat`, `KSP`, `DM`, `SNES`, etc.

---

If you want, next I can show:

* how to organize these functions into a **mini PETSc library**
* how PETSc’s **object ownership model** works
* how to pass `Vec` safely between **time-stepping / eigen / nonlinear solvers**

Just say the word.
