# NSSP-Mini-project-1

The data file containing the original anatomical, functional and event files must first be downloaded from moodle and placed in a `data` folder next to `T_code.ipynb` (inside `T_miniproject_1`).

```code
T_miniproject_1
    /T_code.ipynb
    /T_report.pdf
    /T_presentation.mp4
    /data
        /T1
            /T1w.nii.gz
        /fMRI
            /tfMRI_LANGUAGE_LR
                /tfMRI_LANGUAGE_LR_task_onsets.csv
                /tfMRI_LANGUAGE_LR.nii.gz
        /derivatives          <- created by this notebook (outputs + figures)
```

Requirements: FSL (`bet`, `fast`, `flirt` in the PATH, `FSLDIR` set) and the Python packages `nilearn`, `nibabel`, `antspyx`, `scikit-learn`.

Then the notebook must be run in order to generate all figure and deliverable outputs (this takes a few minutes). The deliverable outputs are placed in the `data/derivatives` folder (inside `T_miniproject_1`).

## Repository contents
 
 - **Code:** `T_code.ipynb`

 - **Report:** `T_report.pdf`

 - **Video:** `T_presentation.mp4`
