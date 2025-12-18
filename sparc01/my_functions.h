#ifndef _MY_FUNCTIONS_H_
#define _MY_FUNCTIONS_H_

#include "petsc.h"
#include "isddft.h"

void my_SddftObjInitialize(SDDFT_OBJ* pSddft);
void my_Read_parameters(SDDFT_OBJ *pSddft);
void my_Read_ion(SDDFT_OBJ *pSddft);
void my_Read_relax(SDDFT_OBJ *pSddft);
void my_Read_pseudopotential(SDDFT_OBJ *pSddft);

#endif