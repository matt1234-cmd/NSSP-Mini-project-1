# %% [markdown]
# # Part 1 - 3. GLM
# Single subject, single run (HCP language task, story vs math).
#
# This file can be run as a script (`python 02_glm.py`) or cell by cell (`# %%`) in VS Code.
#
# Inputs: the outputs of part 1-2 (functional preprocessing, `Project_notebook.ipynb`, sections 4-5),
# in `data/derivatives/func/`. Run the notebook first.
# - `func_mc_brain_smooth6mm.nii.gz`: run without the first 3 volumes, motion-corrected, masked, smoothed;
# - `task_onsets_used.csv`: onsets shifted by 3 x TR to match the trimmed run;
# - `motion_confounds.txt`: the 6 MCFLIRT parameters (+ one spike regressor per volume with FD > 0.5 mm, none here).
#
# - **3a** Design matrix
# - **3b** Correlation between the regressors

# %%
import os
import os.path as op

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix

# Paths relative to this file: works on every machine without editing
try:
    root_dir = op.dirname(op.abspath(__file__))
except NameError:                                   # VS Code interactive window
    root_dir = op.abspath(".")
data_dir = op.join(root_dir, "data")
out_dir = op.join(data_dir, "derivatives")
glm_dir = op.join(out_dir, "glm")
fig_dir = op.join(out_dir, "figures")
for d in (glm_dir, fig_dir):
    os.makedirs(d, exist_ok=True)
FIG_DPI = 300

# Outputs of part 1-2 (functional preprocessing)
FWHM = 6                                            # mm, smoothing chosen in part 1-2
func_dir = op.join(out_dir, "func")
func_img_path = op.join(func_dir, f"func_mc_brain_smooth{FWHM}mm.nii.gz")
events_path = op.join(func_dir, "task_onsets_used.csv")         # onsets in the time frame of the trimmed run
confounds_path = op.join(func_dir, "motion_confounds.txt")      # rx ry rz (rad), tx ty tz (mm), spikes

for path in (func_img_path, events_path, confounds_path):
    assert op.exists(path), f"{path} not found: run part 1-2 of Project_notebook.ipynb first"

# %% [markdown]
# ## Acquisition parameters and events

# %%
func_header = nib.load(func_img_path).header        # header only, the 4D data is not loaded
n_scans = func_header.get_data_shape()[3]
t_r = float(func_header.get_zooms()[3])
frame_times = np.arange(n_scans) * t_r              # same convention as FirstLevelModel (slice_time_ref=0)
print(f"{n_scans} volumes, TR = {t_r} s, run duration = {n_scans * t_r:.1f} s")

events = pd.read_csv(events_path)
print(events.groupby("trial_type")["duration"].agg(["count", "mean", "sum"]).round(2))

# %% [markdown]
# ## 3a. Design matrix
#
# The event file contains two levels of description:
# - the **blocks** `story` and `math` (~12-30 s each), which alternate without rest;
# - the **sub-events** inside each block: `present_*` (listening to the story / the operations),
#   `question_*` (2-alternative question), `response_*` (button press), and the short auditory
#   `cue` before each block.
#
# The two levels overlap in time, so they cannot be put in the same model (a block is almost the sum
# of its sub-events). We build:
# 1. **Block design (main model)**: `story`, `math`. This answers the question of the project (story vs
#    math) with the most power, and is the design used by Binder et al. (2011) and HCP.
# 2. **Event design (detailed model)**: the 7 sub-events. Used to discuss the collinearity in 3b and for the
#    theoretical questions (question vs response, processing vs preparation).
#
# Common settings, following lab 5:
# - **HRF**: canonical SPM (double gamma); each regressor = boxcar of the condition convolved with the HRF.
# - **Drifts**: cosine basis with a 0.01 Hz high-pass cut-off (100 s). The task alternates story/math
#   with a period of ~55-60 s (~0.017 Hz), above the cut-off, so the drift terms do not absorb the task.
# - **Constant**: intercept.
# - **Motion**: the 6 MCFLIRT parameters of part 1-2 as nuisance regressors (no spike regressor: no volume
#   has FD > 0.5 mm). The motion is small (mean FD 0.10 mm), but it costs only 6 degrees of freedom out of 313.

# %%
HRF_MODEL = "spm"
DRIFT_MODEL = "cosine"
HIGH_PASS = 0.01                                    # Hz

BLOCK_CONDITIONS = ["story", "math"]
EVENT_CONDITIONS = ["present_story", "question_story", "response_story",
                    "present_math", "question_math", "response_math", "cue"]

# Motion confounds (part 1-2)
motion = np.loadtxt(confounds_path, ndmin=2)
assert motion.shape[0] == n_scans, "confounds and fMRI run must have the same number of volumes"
motion_names = ["rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"]
motion_names += [f"spike_{k}" for k in range(motion.shape[1] - 6)]


def build_design(conditions):
    """First-level design matrix for a subset of the event types."""
    ev = events[events["trial_type"].isin(conditions)]
    return make_first_level_design_matrix(frame_times, ev, hrf_model=HRF_MODEL,
                                          drift_model=DRIFT_MODEL, high_pass=HIGH_PASS,
                                          add_regs=motion, add_reg_names=motion_names)


dm_block = build_design(BLOCK_CONDITIONS)
dm_event = build_design(EVENT_CONDITIONS)
drift_cols = [c for c in dm_block.columns if c.startswith("drift")]
print("Block design:", dm_block.shape, list(dm_block.columns))
print("Event design:", dm_event.shape, list(dm_event.columns))

# Saved for the GLM (3c-3d) and for part 2 (correlation of the K-means clusters with the regressors)
dm_block.to_csv(op.join(glm_dir, "design_matrix_block.csv"))
dm_event.to_csv(op.join(glm_dir, "design_matrix_event.csv"))

# %%
# Figure: both design matrices side by side
fig, axs = plt.subplots(1, 2, figsize=(11, 6), gridspec_kw={"width_ratios": [dm_block.shape[1], dm_event.shape[1]]})
plot_design_matrix(dm_block, axes=axs[0])
axs[0].set_title("Block design (main model)", fontsize=11)
plot_design_matrix(dm_event, axes=axs[1])
axs[1].set_title("Event design (detailed model)", fontsize=11)
for ax in axs:
    ax.set_ylabel("scan")
fig.tight_layout()
fig.savefig(op.join(fig_dir, "7a_design_matrix.png"), dpi=FIG_DPI)
plt.show()

# %%
# Figure: task regressors of the block design over time, with the blocks shaded
STORY_COLOR, MATH_COLOR = "#2a78d6", "#eb6834"
fig, ax = plt.subplots(figsize=(10, 3))
for _, ev in events[events["trial_type"].isin(BLOCK_CONDITIONS)].iterrows():
    color = STORY_COLOR if ev["trial_type"] == "story" else MATH_COLOR
    ax.axvspan(ev["onset"], ev["onset"] + ev["duration"], color=color, alpha=0.12, lw=0)
ax.plot(frame_times, dm_block["story"], color=STORY_COLOR, lw=2, label="story")
ax.plot(frame_times, dm_block["math"], color=MATH_COLOR, lw=2, label="math")
ax.set_xlabel("time (s)")
ax.set_ylabel("regressor (a.u.)")
ax.set_xlim(0, frame_times[-1])
ax.set_title("Block regressors (block boxcars convolved with the SPM HRF; shading = block timing)", fontsize=10)
ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="0.9")
fig.tight_layout()
fig.savefig(op.join(fig_dir, "7a_block_regressors.png"), dpi=FIG_DPI)
plt.show()

# %% [markdown]
# ## 3b. Correlation between the regressors
#
# Pearson correlation between the columns of each design matrix. The constant is left out (zero variance).
# The variance inflation factor (VIF = 1 / (1 - R²) of each regressor regressed on all the others,
# constant included) summarises how much each beta's variance is inflated by collinearity.

# %%
def vif(dm, columns):
    """Variance inflation factor of each column, regressed on all the other columns of dm."""
    out = {}
    for col in columns:
        y = dm[col].to_numpy()
        X = dm.drop(columns=col).to_numpy()
        resid = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        r2 = 1 - resid.var() / y.var()
        out[col] = 1 / (1 - r2) if r2 < 1 else np.inf
    return pd.Series(out)


corr_block = dm_block.drop(columns="constant").corr()
corr_event = dm_event.drop(columns="constant").corr()
print("Block design correlations (task regressors):\n", corr_block.loc[BLOCK_CONDITIONS, BLOCK_CONDITIONS].round(3))
print("\nVIF block design:\n", vif(dm_block, BLOCK_CONDITIONS).round(2))
print("\nVIF event design:\n", vif(dm_event, EVENT_CONDITIONS).round(2))

# Largest absolute correlations between task regressors of the event design
task_corr = corr_event.loc[EVENT_CONDITIONS, EVENT_CONDITIONS]
pairs = (task_corr.where(np.triu(np.ones(task_corr.shape, bool), k=1)).stack()
         .sort_values(key=np.abs, ascending=False))
print("\nEvent design, strongest task correlations:\n", pairs.head(6).round(3))

# Task-correlated motion: a motion regressor that follows the task would remove task signal
print("\nCorrelation of the motion parameters with the block regressors:\n",
      corr_block.loc[motion_names, BLOCK_CONDITIONS].round(3))
print("\nCorrelation of the motion parameters with the drifts (max |r|):\n",
      corr_block.loc[motion_names, drift_cols].abs().max(axis=1).round(3))

# %%
# Figure: correlation matrices (diverging blue - gray - red, symmetric -1..1)
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#184f95", "#f0efec", "#b3302f"])


def plot_corr(corr, ax, title):
    n = len(corr)
    im = ax.imshow(corr.to_numpy(), cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(n), corr.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(n), corr.index, fontsize=8)
    for i in range(n):
        for j in range(n):
            v = corr.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(v) > 0.6 else "#0b0b0b")
    ax.set_title(title, fontsize=11)
    return im


fig, axs = plt.subplots(1, 2, figsize=(14, 6.5), gridspec_kw={"width_ratios": [len(corr_block), len(corr_event)]})
plot_corr(corr_block, axs[0], "Block design")
im = plot_corr(corr_event, axs[1], "Event design")
fig.colorbar(im, ax=axs, shrink=0.7, label="Pearson correlation")
fig.savefig(op.join(fig_dir, "7b_regressor_correlation.png"), dpi=FIG_DPI, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### Comments on 3b
#
# Values for the preprocessed run of part 1-2: 313 volumes (first 3 removed), onsets shifted by 3 x TR.
#
# **Block design: `story` and `math` are strongly anti-correlated (r = -0.94, VIF 10.5 and 12.1).**
# The blocks alternate with no rest: 94.5% of the run (213 of 225.4 s) is covered by either a story or a math
# block. Only ~12.4 s (cues and gaps) are left as the implicit baseline. So `story + math` ≈ constant, and
# the two regressors are almost collinear with each other and with the intercept.
# - Consequence for **3c**: the individual betas (story alone, math alone) are poorly estimated (variance
#   inflated ~10×). They do not mean "versus rest", only "versus the ~12 s of cues/gaps". Their maps must be
#   read with caution, and they mostly show what is common to both tasks (auditory input, attention, button press).
# - Consequence for **3d**: the **story − math** contrast is well estimated. It lies along the direction where
#   the design varies most (the alternation itself). This is the robust result of the GLM, and the reason the
#   task was designed with matched math blocks (Binder et al., 2011).
#
# The last math block starts at 217.4 s and is cut by the end of the run (8 s of its 12.6 s are acquired);
# its regressor is simply truncated.
#
# **Drifts.** `drift_4` (cosine of period ~113 s) correlates with the task (r = 0.34 with story, -0.41 with
# math). The four consecutive math blocks between 138 and 187 s give the task some low-frequency content, which
# the 0.01 Hz high-pass partly competes with. The correlation stays moderate. A lower cut-off (e.g. 1/128 Hz,
# the SPM default) would reduce it, at the cost of less drift removal.
#
# **Motion.** The 6 motion parameters are only weakly correlated with the task (|r| ≤ 0.25, largest for
# `trans_x` with math), so there is no sign of task-correlated motion, and adding them as confounds removes
# little task signal. They are more correlated with the slow drifts (|r| up to 0.62 between the rotations and
# `drift_1`): part of the "motion" is a slow drift of the head or of the realignment estimate, which the two
# sets of regressors share. Between themselves, `rot_y` and `trans_x` correlate at 0.75 (a rotation around y
# moves the brain's centre of mass along x).
#
# **Event design.** The correlations follow the timeline of each trial:
# - `present_story` vs `present_math` (r = -0.56): they alternate;
# - `response_story` vs `cue` (r = 0.50): the cue starts ~0.2 s after the story response ends, and the HRF blurs
#   the two together;
# - `question_math` vs `response_math` (r = 0.47): adjacent ~3 s events.
#
# `present_story` has the highest VIF (12.6): it covers most of each story block, so it is close to a combination
# of the other regressors. The short story events (`question_story`, 4 × 1.2 s) carry little signal, hence low
# power for the question-vs-response contrasts of the theoretical part.

# %% [markdown]
# ## 3c. GLM and statistical maps of the task regressors
#
# **Level.** First level (single subject, single run): there is only one subject and one run, so there is
# no second level to run. The GLM is fitted voxel by voxel on the preprocessed run of part 1-2, with the
# design matrices of 3a passed directly (same regressors, same drifts, same motion confounds).
#
# Settings, following lab 5:
# - **mask**: the functional brain mask of part 1-2 (BET on the mean fMRI);
# - **no extra smoothing**: the run is already smoothed (FWHM 6 mm, part 1-2);
# - **signal scaling**: each voxel is expressed in % of its mean, so the betas are in % signal change;
# - **noise model**: AR(1), because the fMRI noise is autocorrelated at TR = 0.72 s (OLS would underestimate
#   the variance and inflate the statistics);
# - **thresholding**: FDR q < 0.05 and clusters of at least 10 voxels (`threshold_stats_img`, as in lab 5).
#   The maps are z-scores, two-sided.

# %%
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm import threshold_stats_img
from nilearn.reporting import get_clusters_table
from nilearn import plotting

func_mask = op.join(func_dir, "func_mean_brain_mask.nii.gz")              # part 1-2
anat_mni = op.join(out_dir, "anat", "T1w_brain_MNI_SyN.nii.gz")          # part 1-1, background of the maps
assert op.exists(func_mask) and op.exists(anat_mni), "run parts 1-1 and 1-2 of Project_notebook.ipynb first"

FDR_ALPHA = 0.05
CLUSTER_SIZE = 10                                   # voxels
Z_CUTS = [-24, -12, -4, 4, 14, 26, 40, 52]          # axial slices (MNI z, mm)
Z_VMAX = 10                                         # common colour scale for all the maps


def fit_glm(design):
    glm = FirstLevelModel(t_r=t_r, mask_img=func_mask, smoothing_fwhm=None, noise_model="ar1",
                          signal_scaling=0, minimize_memory=True)
    return glm.fit(func_img_path, design_matrices=design)


def contrast_vector(design, weights):
    """Contrast vector over all the columns of the design matrix, from a {column: weight} dict."""
    vec = np.zeros(design.shape[1])
    for col, w in weights.items():
        vec[design.columns.get_loc(col)] = w
    return vec


def threshold(z_map, control="fdr", alpha=FDR_ALPHA):
    """FDR q < 0.05 (two-sided) + cluster extent, as in lab 5. Returns the thresholded map and the z threshold."""
    return threshold_stats_img(z_map, mask_img=func_mask, alpha=alpha, height_control=control,
                               cluster_threshold=CLUSTER_SIZE, two_sided=True)


def plot_map(z_thr, z_thr_value, title, axes, vmax=Z_VMAX):
    display = plotting.plot_stat_map(z_thr, bg_img=anat_mni, display_mode="z", cut_coords=Z_CUTS, axes=axes,
                                     threshold=z_thr_value, vmax=vmax, cmap="RdBu_r", black_bg=False,
                                     annotate=True, draw_cross=False)
    display.title(title, size=9, color="black", bgcolor="white")
    return display


glm_block = fit_glm(dm_block)
glm_event = fit_glm(dm_event)

# %%
# Statistical map of each task regressor (each regressor vs the implicit baseline)
z_maps = {}
for name, glm, design, conditions in [("block", glm_block, dm_block, BLOCK_CONDITIONS),
                                      ("event", glm_event, dm_event, EVENT_CONDITIONS)]:
    for cond in conditions:
        z = glm.compute_contrast(contrast_vector(design, {cond: 1}), output_type="z_score")
        z.to_filename(op.join(glm_dir, f"zmap_{name}_{cond}.nii.gz"))
        z_maps[(name, cond)] = z

summary = []
for (name, cond), z in z_maps.items():
    z_thr, thr = threshold(z)
    data = z_thr.get_fdata()
    summary.append({"model": name, "regressor": cond, "z threshold": round(thr, 2),
                    "voxels z>0": int((data > 0).sum()), "voxels z<0": int((data < 0).sum()),
                    "max z": round(float(z.get_fdata().max()), 1), "min z": round(float(z.get_fdata().min()), 1)})
summary = pd.DataFrame(summary)
summary.to_csv(op.join(glm_dir, "regressor_maps_summary.csv"), index=False)
print(summary.to_string(index=False))

# %%
# Figure: block design, two rows per task regressor: thresholded map, then the unthresholded map
# (the thresholded maps are almost empty, see the comments; the raw map shows the spatial pattern behind them)
fig, axs = plt.subplots(2 * len(BLOCK_CONDITIONS), 1, figsize=(14, 2.4 * 2 * len(BLOCK_CONDITIONS)))
for k, cond in enumerate(BLOCK_CONDITIONS):
    z_thr, thr = threshold(z_maps[("block", cond)])
    plot_map(z_thr, thr, f"{cond} (vs implicit baseline), z, FDR q < {FDR_ALPHA}, |z| > {thr:.2f}", axs[2 * k])
    plot_map(z_maps[("block", cond)], 0, f"{cond}, unthresholded z", axs[2 * k + 1], vmax=5)
fig.savefig(op.join(fig_dir, "7c_regressor_maps_block.png"), dpi=FIG_DPI, bbox_inches="tight")
plt.show()

# Figure (appendix): event design, one row per sub-event regressor
fig, axs = plt.subplots(len(EVENT_CONDITIONS), 1, figsize=(14, 2.4 * len(EVENT_CONDITIONS)))
for ax, cond in zip(axs, EVENT_CONDITIONS):
    z_thr, thr = threshold(z_maps[("event", cond)])
    plot_map(z_thr, thr, f"{cond}, z, FDR q < {FDR_ALPHA}, |z| > {thr:.2f}", ax)
fig.savefig(op.join(fig_dir, "annex_regressor_maps_event.png"), dpi=FIG_DPI, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3d. Story vs math contrast
#
# Contrast vector on the block design (columns in the order of the design matrix):
# +1 on `story`, -1 on `math`, 0 on the motion, drift and constant columns. It is printed below.
# Positive z = more active during the stories (semantic processing), negative z = more active during math
# (numerical processing). The math - story map is the same map with the sign flipped.

# %%
c_story_math = contrast_vector(dm_block, {"story": 1, "math": -1})
print("Contrast vector story - math:")
print(pd.Series(c_story_math, index=dm_block.columns).astype(int).to_string())

z_sm = glm_block.compute_contrast(c_story_math, output_type="z_score")
z_sm.to_filename(op.join(glm_dir, "zmap_story_minus_math.nii.gz"))
z_sm_thr, thr_sm = threshold(z_sm)
z_sm_thr.to_filename(op.join(glm_dir, "zmap_story_minus_math_thr.nii.gz"))
d = z_sm_thr.get_fdata()
print(f"\nFDR threshold |z| > {thr_sm:.2f}:  story > math {(d > 0).sum()} voxels,  math > story {(d < 0).sum()} voxels")

clusters = get_clusters_table(z_sm, stat_threshold=thr_sm, cluster_threshold=CLUSTER_SIZE, two_sided=True)
clusters.to_csv(op.join(glm_dir, "clusters_story_minus_math.csv"), index=False)
print(clusters.head(25).to_string(index=False))

# %%
# Stricter control for the figure: Bonferroni (family-wise error) at 0.05, which keeps only the core regions
z_sm_bonf, thr_bonf = threshold(z_sm, control="bonferroni", alpha=0.05)
d = z_sm_bonf.get_fdata()
print(f"Bonferroni threshold |z| > {thr_bonf:.2f}:  story > math {(d > 0).sum()} voxels,  math > story {(d < 0).sum()} voxels")

# Figure: story - math contrast, FDR and Bonferroni, axial slices + glass brain (Bonferroni)
fig = plt.figure(figsize=(14, 8.5))
ax_fdr = fig.add_axes([0, 0.68, 1, 0.32])
ax_bonf = fig.add_axes([0, 0.35, 1, 0.32])
ax_glass = fig.add_axes([0.15, 0, 0.7, 0.33])
plot_map(z_sm_thr, thr_sm, f"story - math, FDR q < {FDR_ALPHA}, |z| > {thr_sm:.2f}   (red: story > math, blue: math > story)", ax_fdr)
plot_map(z_sm_bonf, thr_bonf, f"story - math, Bonferroni p < 0.05, |z| > {thr_bonf:.2f}", ax_bonf)
plotting.plot_glass_brain(z_sm_bonf, threshold=thr_bonf, vmax=Z_VMAX, cmap="RdBu_r", plot_abs=False,
                          colorbar=True, display_mode="lzry", axes=ax_glass)
fig.savefig(op.join(fig_dir, "7d_contrast_story_minus_math.png"), dpi=FIG_DPI, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### Comments on 3c
#
# **Block design (`7c_regressor_maps_block.png`).** As predicted in 3b, the maps of `story` alone and `math`
# alone are almost empty after FDR correction (story: 145 voxels, math: 1137 voxels, all negative), and the
# unthresholded maps look like smooth noise. Each regressor is tested against the implicit baseline, i.e. the
# ~12 s of the run that are covered by no block, and `story + math` is almost equal to the constant. The model
# therefore cannot tell "story" from "constant", the betas have a ~10× inflated variance, and their sign is
# driven by how the intercept absorbs the mean signal, not by the task. These maps are not interpretable on their
# own; the information of the block design is in the difference between the two regressors (3d).
#
# **Event design (`annex_regressor_maps_event.png`).** The short sub-events vary independently of the constant,
# so their maps are interpretable and match the expected functional anatomy:
# - `cue`: right (and left) superior temporal gyrus, i.e. auditory cortex, for the short auditory cue;
# - `question_math`: left precentral gyrus (hand motor cortex), consistent with the right-hand button press that
#   follows the question (the HRF blurs question and response together, r = 0.47 in 3b);
# - `present_math`: bilateral intraparietal and precentral regions (positive) with medial prefrontal decreases
#   (negative), the classic arithmetic vs default-mode pattern;
# - `question_story`: bilateral lateral temporal cortex, the story question being a spoken sentence;
# - `present_story`: nothing survives (VIF 12.6, it spans most of each story block and is collinear with the rest).

# %% [markdown]
# ### Comments on 3d
#
# Contrast vector (column order of the block design): `[-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]` for
# `[math, story, rot_x, rot_y, rot_z, trans_x, trans_y, trans_z, drift_1, drift_2, drift_3, drift_4, constant]`.
#
# Unlike the single-regressor maps, the contrast is very strong (peak |z| ≈ 10). With FDR q < 0.05 the threshold
# drops to |z| > 2.42 because so many voxels are significant (~72 000 voxels, about a third of the mask), and the
# map becomes diffuse. The Bonferroni map (|z| > 5.19, ~12 000 voxels) shows the core regions more clearly:
# - **story > math (red)**: medial prefrontal cortex and left superior frontal gyrus (peaks at (-4, 60, 28) and
#   (-20, 40, 50)), bilateral anterior temporal lobes / superior temporal sulcus (peak at (-54, -4, -10)),
#   angular gyrus, posterior cingulate / precuneus, and the right posterior cerebellum (24, -82, -44);
# - **math > story (blue)**: bilateral intraparietal sulcus (peaks at (40, -40, 38), (54, -38, 46),
#   (-52, -46, 56)), right precentral / middle frontal gyrus (46, 4, 20) and (36, 6, 64), right lateral prefrontal
#   cortex (42, 42, 22) and the inferior temporal / occipito-temporal cortex.
#
# These are respectively the semantic / default-mode regions and the fronto-parietal regions of numerical
# processing reported by Binder et al. (2011); their attribution to functional networks is done in 3e.
# Peak coordinates and cluster sizes: `data/derivatives/glm/clusters_story_minus_math.csv`.
