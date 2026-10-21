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
