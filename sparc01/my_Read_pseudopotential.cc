////////////////////////////////////////////////////////////////////////////////////////////////
//                   Read_pseudopotential: reads the pseudopotential file                     //
////////////////////////////////////////////////////////////////////////////////////////////////
void my_Read_pseudopotential(SDDFT_OBJ *pSddft) {

  PetscPrintf(PETSC_COMM_WORLD, "\n\n *** ENTER my_Read_pseudopotential\n\n");

  FILE *fRvsVFile;
  int at, i, l;
  PetscScalar value, rc;
  int lineCtr;
  PetscScalar Pi = M_PI;

  /*
   * allocate memory for pseudopotential object in the structure
   */
  pSddft->psd = (PSD_OBJ *)malloc(sizeof(PSD_OBJ) * pSddft->Ntype);
  if (pSddft->psd == NULL) {
    cout << "Memory alocation fail in pSddft->psd";
    exit(1);
  }

  for (at = 0; at < pSddft->Ntype; at++) {
    char RvsVFile[100] = "./pseudopotential/";
    strcat(RvsVFile, pSddft->psdName[at]);

    if ((fRvsVFile = fopen((const char *)RvsVFile, "rb")) == NULL) {
      cout << "Couldn't open pseudopotential file... " << RvsVFile << endl;
      cout << "Exiting for this configuration...\n";
      exit(1);
    }

    int count = 0; // number of elements in the radial grid
    char str[60];
    /*
     * check if pseudopotential file is consistent with input atom type
     */
    fscanf(fRvsVFile, "%s", str);
    if (strcmp(str, pSddft->atomType[at]) != 0) {
      cout << "pseudopotential file does not match with input atom type" << at << endl;
      exit(1);
    }

    /*
     * read the pseudopotential file (must be in Troulier-Martins format)
     */
    do {
      fgets(str, 60, fRvsVFile);
    } while (strcmp(str, " Radial grid follows\n"));
    do {
      fscanf(fRvsVFile, "%s", str);
      count++;
    } while (strcmp(str, "Pseudopotential") != 0);
    count = count - 1;
    fclose(fRvsVFile);
    pSddft->psd[at].size = count;

    pSddft->psd[at].rc_s = 0.0;
    pSddft->psd[at].rc_p = 0.0;
    pSddft->psd[at].rc_d = 0.0;
    pSddft->psd[at].rc_f = 0.0;

    /*
     * allocate memory for arrays storing radial grid, pseudopotentials and
     * pseudowavefunctions
     */
    pSddft->psd[at].RadialGrid = (PetscScalar *)malloc(sizeof(PetscScalar) * count);
    if (pSddft->psd[at].RadialGrid == NULL) {
      cout << "Memory alocation fail in pSddft->psd[at].RadialGrid" << endl;
      exit(1);
    }

    pSddft->psd[at].Vs = (PetscScalar *)malloc(sizeof(PetscScalar) * count);
    if (pSddft->psd[at].Vs == NULL) {
      cout << "Memory alocation fail in pSddft->psd[at].Vs" << endl;
      exit(1);
    }

    pSddft->psd[at].Vp = (PetscScalar *)malloc(sizeof(PetscScalar) * count);
    if (pSddft->psd[at].Vp == NULL) {
      cout << "Memory alocation fail in pSddft->psd[at].Vp" << endl;
      exit(1);
    }

    pSddft->psd[at].Vd = (PetscScalar *)malloc(sizeof(PetscScalar) * count);
    if (pSddft->psd[at].Vd == NULL) {
      cout << "Memory alocation fail in pSddft->psd[at].Vd" << endl;
      exit(1);
    }

    pSddft->psd[at].Vf = (PetscScalar *)malloc(sizeof(PetscScalar) * count);
    if (pSddft->psd[at].Vf == NULL) {
      cout << "Memory alocation fail in pSddft->psd[at].Vf" << endl;
      exit(1);
    }

    pSddft->psd[at].Us = (PetscScalar *)malloc(sizeof(PetscScalar) * count);
    if (pSddft->psd[at].Us == NULL) {
      cout << "Memory alocation fail in pSddft->psd[at].Us" << endl;
      exit(1);
    }

    pSddft->psd[at].Up = (PetscScalar *)malloc(sizeof(PetscScalar) * count);
    if (pSddft->psd[at].Up == NULL) {
      cout << "Memory alocation fail in pSddft->psd[at].Up" << endl;
      exit(1);
    }

    pSddft->psd[at].Ud = (PetscScalar *)malloc(sizeof(PetscScalar) * count);
    if (pSddft->psd[at].Ud == NULL) {
      cout << "Memory alocation fail in pSddft->psd[at].Ud" << endl;
      exit(1);
    }

    pSddft->psd[at].Uf = (PetscScalar *)malloc(sizeof(PetscScalar) * count);
    if (pSddft->psd[at].Uf == NULL) {
      cout << "Memory alocation fail in pSddft->psd[at].Uf" << endl;
      exit(1);
    }

    pSddft->psd[at].uu = (PetscScalar *)malloc(sizeof(PetscScalar) * count);
    if (pSddft->psd[at].uu == NULL) {
      cout << "Memory alocation fail in pSddft->psd[at].uu" << endl;
      exit(1);
    }

    pSddft->psd[at].Vloc = (PetscScalar *)malloc(sizeof(PetscScalar) * count);
    if (pSddft->psd[at].Vloc == NULL) {
      cout << "Memory alocation fail in pSddft->psd[at].Vloc" << endl;
      exit(1);
    }

    /*
     * open file again and read the pseudopotentials and pseudo wave functions
     */
    if ((fRvsVFile = fopen((const char *)RvsVFile, "rb")) == NULL) {
      cout << "Couldn't open pseudopotential file... " << RvsVFile << endl;
      cout << "Exiting for this configuration...\n";
      exit(1);
    }
    do {
      fgets(str, 60, fRvsVFile);
    } while (strcmp(str, " Radial grid follows\n"));

    for (i = 0; i < count; i++) {
      fscanf(fRvsVFile, "%lf", &value);
      pSddft->psd[at].RadialGrid[i] = value;
    }

    lineCtr = 0;
    while (strcmp(str, " Pseudopotential follows (l on next line)\n")) {
      fgets(str, 60, fRvsVFile);
      lineCtr++;
    }
    /*
     * read pseudopotential
     */
    while (strcmp(str, " Pseudopotential follows (l on next line)\n") == 0) {
      fscanf(fRvsVFile, "%d", &l);
      if (l == 0) // s orbital
      {
        pSddft->psd[at].lmax = 0;
        for (i = 0; i < count; i++) {
          fscanf(fRvsVFile, "%lf", &value);
          value = 0.5 * value / (pSddft->psd[at].RadialGrid[i]);
          pSddft->psd[at].Vs[i] = value;

          if (pSddft->localPsd[at] == 0)
            pSddft->psd[at].Vloc[i] = value;
        }
      }
      if (l == 1) // p orbital
      {
        pSddft->psd[at].lmax = 1;
        for (i = 0; i < count; i++) {
          fscanf(fRvsVFile, "%lf", &value);
          value = 0.5 * value / (pSddft->psd[at].RadialGrid[i]);
          pSddft->psd[at].Vp[i] = value;

          if (pSddft->localPsd[at] == 1)
            pSddft->psd[at].Vloc[i] = value;
        }
      }
      if (l == 2) // d orbital
      {
        pSddft->psd[at].lmax = 2;
        for (i = 0; i < count; i++) {
          fscanf(fRvsVFile, "%lf", &value);
          value = 0.5 * value / (pSddft->psd[at].RadialGrid[i]);
          pSddft->psd[at].Vd[i] = value;

          if (pSddft->localPsd[at] == 2)
            pSddft->psd[at].Vloc[i] = value;
        }
      }
      if (l == 3) // f orbital
      {
        pSddft->psd[at].lmax = 3;
        for (i = 0; i < count; i++) {
          fscanf(fRvsVFile, "%lf", &value);
          value = 0.5 * value / (pSddft->psd[at].RadialGrid[i]);
          pSddft->psd[at].Vf[i] = value;

          if (pSddft->localPsd[at] == 3)
            pSddft->psd[at].Vloc[i] = value;
        }
      }

      for (i = 0; i < lineCtr; i++)
        fgets(str, 60, fRvsVFile);
    }
    /*
     * read until valence charge block is found
     */
    while (strcmp(str, " Valence charge follows\n")) {
      fgets(str, 60, fRvsVFile);
    }
    /*
     * read valence charge
     */
    while (strcmp(str, " Valence charge follows\n") == 0) {
      for (i = 0; i < count; i++) {
        fscanf(fRvsVFile, "%lf", &value);
        value = value / (4 * Pi * pSddft->psd[at].RadialGrid[i] * pSddft->psd[at].RadialGrid[i]);
        pSddft->psd[at].uu[i] = value;
      }

      for (i = 0; i < lineCtr; i++)
        fgets(str, 60, fRvsVFile);
    }
    /*
     * read pseudowavefunction
     */
    while (strcmp(str, " Pseudo-wave-function follows (l, zelect, rc)\n") == 0) {
      fscanf(fRvsVFile, "%d", &l);
      fscanf(fRvsVFile, "%lf", &value);
      fscanf(fRvsVFile, "%lf", &rc);
      if (l == 0) // s orbital
      {
        pSddft->psd[at].rc_s = rc;
        for (i = 0; i < count; i++) {
          fscanf(fRvsVFile, "%lf", &value);
          value = value / pSddft->psd[at].RadialGrid[i];
          pSddft->psd[at].Us[i] = value;
        }
      }
      if (l == 1) // p orbital
      {
        pSddft->psd[at].rc_p = rc;
        for (i = 0; i < count; i++) {
          fscanf(fRvsVFile, "%lf", &value);
          value = value / pSddft->psd[at].RadialGrid[i];
          pSddft->psd[at].Up[i] = value;
        }
      }
      if (l == 2) // d orbital
      {
        pSddft->psd[at].rc_d = rc;
        for (i = 0; i < count; i++) {
          fscanf(fRvsVFile, "%lf", &value);
          value = value / pSddft->psd[at].RadialGrid[i];
          pSddft->psd[at].Ud[i] = value;
        }
      }
      if (l == 3) // f orbital
      {
        pSddft->psd[at].rc_f = rc;
        for (i = 0; i < count; i++) {
          fscanf(fRvsVFile, "%lf", &value);
          value = value / pSddft->psd[at].RadialGrid[i];
          pSddft->psd[at].Uf[i] = value;
        }
      }

      for (i = 0; i < lineCtr; i++) {
        if (feof(fRvsVFile))
          break;
        fgets(str, 60, fRvsVFile);
      }
    }
    fclose(fRvsVFile);
  }

  PetscPrintf(PETSC_COMM_WORLD, "\n\n *** EXIT my_Read_pseudopotential\n\n");

  return;
}
