# NSSP-Mini-project-1

The data file containing the original anatomical, functional and event files must first be downloaded from moodle and placed in the data folder.

```code
NSSP-Mini-project-1
    /Project_notebook.ipynb
    /data
        /T1
            /T1w.nii.gz
        /fMRI
            /tfMRI_LANGUAGE_LR
                /tfMRI_LANGUAGE_LR_task_onsets.csv
                /tfMRI_LANGUAGE_LR.nii.gz
        /derivatives          <- created by this notebook (outputs + figures)
```

Requirements: FSL (`bet`, `fast`, `flirt` in the PATH, `FSLDIR` set) and the Python packages `nilearn`, `nibabel`, `antspyx`.

Then the notebook must be run in order to generate all figure and deliverable outputs (this takes a few minutes). The deliverable outputs are placed in the `data/derivatives` folder.

## Repository contents
 
 - **Code:** `T_code.ipynb`

 - **Report:** `T_report.pdf`

 - **Video:** `T_presentation.pdf`
