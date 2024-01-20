// Read_relax: reads the .relax file for constraint on movement of atoms
void my_Read_relax(SDDFT_OBJ *pSddft) {
  FILE *fConfFile;
  char ConfigFile[100] = "./";
  strcat(ConfigFile, pSddft->file);
  strcat(ConfigFile, ".relax");
  PetscInt poscnt, index = 0;
  PetscScalar xct, yct, zct;
  char name[10];
  int at = 0;
  PetscInt noa;

  VecDuplicate(pSddft->forces, &pSddft->mvAtmConstraint);
  VecSet(pSddft->mvAtmConstraint, 1);

  if (pSddft->RelaxFlag == 1) {

    if ((fConfFile = fopen((const char *)ConfigFile, "rb")) == NULL) {
      cout << "Couldn't open config file... " << ConfigFile << '\n';
      cout << "Exiting for this configuration...\n";
      exit(1);
    }

    do {
      if (fscanf(fConfFile, "%s", name) == EOF)
        break;
      if (fscanf(fConfFile, "%d", &noa) == EOF)
        break;
      /*
       * check if .relax file is consistent with .ion file
       */
      if (strcmp(name, pSddft->atomType[at]) != 0) {
        cout << "Incorrect atom type in .relax file. Give same type and same order as .ion file\n";
        exit(1);
      }
      for (poscnt = 0; poscnt < noa; poscnt++) {
        fscanf(fConfFile, "%lf", &xct);
        fscanf(fConfFile, "%lf", &yct);
        fscanf(fConfFile, "%lf", &zct);

        VecSetValues(pSddft->mvAtmConstraint, 1, &index, &xct, INSERT_VALUES);
        index++;
        VecSetValues(pSddft->mvAtmConstraint, 1, &index, &yct, INSERT_VALUES);
        index++;
        VecSetValues(pSddft->mvAtmConstraint, 1, &index, &zct, INSERT_VALUES);
        index++;
      }
      at++;
    } while (!feof(fConfFile));
    fclose(fConfFile);

    if (at != pSddft->Ntype) {
      cout << "Number of blocks: " << at << " in .relax file not same as number of atom type: " << pSddft->Ntype
           << " in .inpt file\n";
      exit(1);
    }
  }
  return;
}
