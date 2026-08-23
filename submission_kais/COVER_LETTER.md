Hoa Dinh Nguyen
Posts and Telecommunications Institute of Technology
Hanoi, Vietnam
Email: hoand@ptit.edu.vn
ORCID: 0009-0004-1708-9248

23 August 2026

Editor-in-Chief
Knowledge and Information Systems

Re: Regular research article
“Region-Specific Incremental Feature Menus for Progressive Feature Revelation”

Dear Editor,

Please find enclosed my manuscript for consideration as a regular research article in Knowledge and Information Systems.

The paper is a knowledge-discovery contribution to feature selection under a measurement budget. When covariates are costly or revealed on demand, the object to be mined is not a single global support but local acquisition knowledge: which features to obtain next in each region of the input space. The manuscript proposes region-specific incremental feature menus. Training data are partitioned; an ordered forward-stepwise menu with Bayesian information criterion stopping is learned in each region; nested prefix least-squares models produce predictions at every budget. The resulting menus can be stored, inspected, and executed as piecewise-constant knowledge.

This work is believed to fit KAIS because it concerns knowledge discovery, data mining, and learning from data, not a neural architecture or a single industrial deployment. The evaluation is an anytime error curve, which is the natural metric when knowledge is used incrementally. Early menu entries are reported as knowledge (low Jaccard overlap across regions). Localized Lasso–LARS menus are re-scored under the same nested protocol so that the comparison with localized sparse regression is fair. Public UCI and OpenML datasets are used throughout, and the implementation is included with the submission.

The main empirical finding is scoped rather than universal. On heterogeneous regression problems, including UCI CT slice localization and superconductivity prediction, region menus reduce progressive root-mean-square error relative to a strong global forward-stepwise policy, and the advantage on CT slices is retained as the sample size grows beyond 50,000 observations. Mixed controls (correlated synthetics and a weak-signal OpenML set) show when a single global menu remains sufficient.

The manuscript is original, has not been published elsewhere, and is not under consideration by another journal. I am the sole author and am responsible for all aspects of the work.

Sincerely,

Hoa Dinh Nguyen
Posts and Telecommunications Institute of Technology, Hanoi, Vietnam
hoand@ptit.edu.vn
