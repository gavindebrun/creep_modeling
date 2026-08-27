# Experimental surface-analysis protocol

This document is the analysis contract for the six datasets in `agent.md`. Core detrending, resampling, PSD, ACF, and height-series calculations are centralized in `utils/calc_utils.py`; `utils/experimental_analysis.py` contains the experimental dataset registry and acquisition I/O. `scripts/audit_experimental_protocol.py` writes the complete map- and section-level audit under `results/analysis/experimental_protocol/`.

## Canonical dataset identities

The interrupted polished tests are 475, 525, and 575 MPa. The uninterrupted polished tests are 500, 530, and 588 MPa. Raw-source provenance paths, standardized directories, analysis tables, figures, captions, and prose all use the proper 530 and 588 MPa identities. The genuine interrupted 525 MPa condition is unchanged.

The 475, 525, 575, 500, and 530 MPa surface datasets have complete baseline maps and are eligible for primary paired change analysis. At 588 MPa, 17 pre-test and 25 post-test maps are present, with 14 matched identifiers. All 17 baseline images contain directional grinding relief, none resolves the intended central Vickers fiducial, and their median Sa is 10.5 and 4.9 times the corresponding 500 and 530 MPa baselines. The maps are structurally readable but do not consistently represent the same prepared surface condition or field location as the other uninterrupted baselines. They are therefore restricted to a within-load scalar sensitivity check and excluded from primary cross-load and paired PSD/ACF inference. The 588 MPa strain history remains eligible for mechanical calibration and strain reporting.

Only the six datasets in the executable registry are in manuscript scope. Other archived experiment directories, including an additional interrupted 525 MPa campaign used during exploratory development, are not pooled into the primary or sensitivity results unless this contract, the registry, and the audit are amended together.

## Raw-map acquisition validity

Every in-scope 10x and 50x map must have the recorded Keyence XY calibration, a 768 by 1024 numerical height grid, a missing-pixel fraction no greater than 0.0001, and no connected missing region larger than one pixel. Valid isolated missing pixels are filled from the nearest finite pixel before filtering. A map failing any of these acquisition checks is rejected with its reason recorded in `map_audit.csv`. No map is rejected because its measured roughness, PSD, or ACF is large or small.

## Absolute roughness sectioning

Each 10x map is divided at native sampling into a fixed 2 by 4 grid. The eight sections are 384 by 256 pixels, approximately 529.9 by 353.3 micrometers. A least-squares plane is removed independently from every section. Section arithmetic mean height is

Sa = mean(abs(h - mean(h))).

The specimen-time statistic is the median of all eight section values. Every acquisition-valid section is included in the primary result; sections are subsamples and the specimen is the replicate.

The domain was selected from 1 by 1, 2 by 2, 2 by 4, 4 by 4, 4 by 8, and 8 by 8 candidates without comparing candidates back to a designated reference grid. Correlation length was calculated independently at the native 1.379951-micrometer spacing as the map median of four plane-leveled 256 by 256 micrometer peripheral patches, excluding the central fiducial. Across 226 maps and 904 patches, the map-level 95th-percentile radial ACF 1/e length is 31.55 micrometers. For candidate area A, the conservative number of circular correlation areas is A divided by pi times the squared 95th-percentile correlation length. A candidate had to supply at least eight subsamples, span at least ten of those conservative correlation lengths on its short side, and contain at least 50 circular correlation areas. The ten-length threshold bounds the leading two-dimensional finite-area scale, proportional to the squared correlation-length-to-domain ratio, to order one percent; 50 correlation areas give a nominal inverse-square-root sampling scale of 14 percent. Map-median bootstrap error is reported as a precision diagnostic but is not a pass/fail rule. The 2 by 4 grid is the only eligible candidate: it contains 59.9 conservative correlation areas, spans 11.2 correlation lengths on its short side, supplies eight subsamples, and has a median relative bootstrap error of 15.7 percent. The 4 by 4 and smaller domains fail both physical representativeness criteria; larger domains supply fewer than eight subsamples. The executable evidence and decision are in `results/analysis/experimental_method_selection/`.

A response-based filter is permitted only as a named sensitivity analysis. Within each dataset and time, modified z scores are calculated across all specimen sections for section Sa and maximum absolute leveled height. A section is excluded from that sensitivity result when either absolute score is at least 3.5. The retained values are aggregated by the same median used primarily; changing the aggregation to a mean would confound rejection with a second estimator change. In the completed study this rule rejected 226 of 1808 sections (12.5 percent), and the rejected median Sa was 2.010 micrometers compared with 0.528 micrometers in retained sections. It left four maps with no retained section and makes one 575 MPa endpoint pair unanalyzable. By contrast, retaining every section and using the map median limited the maximum leave-one-section-out change to 2.8 percent at the median map and 13.6 percent at the 95th percentile. Primary estimates therefore use no response-based deletion: the median bounds individual influence without preferentially suppressing the high-relief response. Every sensitivity exclusion is recorded in `section_audit.csv`, and counts by dataset and time are in `section_rejection_summary.csv`.

## Voxel resolution and minimum synthetic feature size

The corrected `XY` and `YZ` segmentations use grain tolerances of 1.5 and
6 degrees, respectively. Both use a 256-pixel (31.36 square micrometer)
connected-defect threshold. A two-pixel in-plane closing is applied only to
the temporary `YZ` segmentation mask to bridge the diagonal scratch; the
preserved defect mask is restored before grain statistics are calculated.
The subsequent grain-size cleanup uses 16 pixels (1.96 square micrometers).
Three-dimensional size, shape, and orientation are selected by forward
sectioning against complete non-surface features in both orthogonal EBSD
planes. The active parameters are
`mu = 2.00`, `sigma = 0.85`, `B/A = 0.40`, `C/A = 0.15`, and a 55 micrometer
maximum equivalent diameter.

Rasterizing the same four continuous feature sets shows that the 0.5- and
1-micrometer morphology distances differ by only 0.00035 at an
8-cubic-micrometer cleanup volume; the distance increases by 0.035 at
2 micrometers and by 0.139 at 4 micrometers. The active discretization is
therefore 1 micrometer with an eight-cell connected-feature cutoff. The
complete rationale and generated evidence are in
`docs/working/ebsd_microstructure_pipeline_audit.md`.

## Solver domain size

Six realizations at each of five sizes were compared through
microstructure-sampling descriptors, without using a mechanical convergence
criterion. The 96 by 96 by 192 micrometer domain is the smallest candidate
meeting the prescribed Schmid-factor, Hall--Petch size-moment, largest-grain,
and full-domain-bias thresholds. One fixed realization is used throughout
calibration, and independent realizations are reserved for subsequent
validation. The 128 by 128 by 256 micrometer domain remains the full-size
spatial-simulation domain.

## Primary matched spatial operator

After acquisition validation, the full height map is resampled to the configured 1-micrometer grid using bilinear interpolation with pixel-center grid semantics. Gaussian anti-alias filtering is applied only when the target spacing is coarser than the source spacing; upsampling the native 10x maps therefore adds no high-frequency information. The largest centered integer grid of non-overlapping 256 by 128 pixel windows gives 256 by 128 micrometer windows and 44 windows per standard 10x map. Each window is independently plane leveled. No window is rejected after its parent map passes acquisition validity. Curves and scalar descriptors are first reduced to the map median across windows; specimen-level map summaries are the inferential observations.

## Power spectral density

For each leveled window, the mean is removed and a separable Hann taper is applied. The two-dimensional spectrum is

P(k) = delta_x^2 * abs(FFT(h * w))^2 / sum(w^2),

with delta_x equal to `VOXELSIZE`. Radial PSD is the arithmetic mean of the two-dimensional spectral values in 17 fixed, equal-width physical-frequency annuli. The lower edge is the inverse 256-micrometer window dimension. The upper edge is the coarser of the numerical-grid and native-10x Nyquist frequencies, 0.3623 inverse micrometers, so bilinear upsampling is not interpreted as added measurement bandwidth. A 12-micrometer shortest-wavelength sensitivity verifies that the broadband direction is not selected by the upper-frequency bins. Morphology plots use radial PSD divided by its trapezoidal integral over this band. Absolute post/pre gain uses the unnormalized radial PSD.

The spectral-median wavelength is obtained from the cumulative power of all resolved two-dimensional modes, including mode multiplicity. Long-wavelength power fraction is the fraction of resolved two-dimensional power at wavelengths of at least 32 micrometers. The high-frequency exponent is the least-squares slope of log10 radial PSD against log10 frequency for wavelengths no greater than 32 micrometers.

## Autocorrelation

The ACF is a linear, overlap-corrected estimator, not a circular FFT correlation. Full autocovariance is calculated by convolution with the reversed field and divided pointwise by the number of overlapping pixel pairs. It is normalized by the zero-lag value. The ACF is averaged in 1-micrometer radial annuli from 0 through 64 micrometers. The correlation length is the first 1/e crossing, linearly interpolated between adjacent lag samples. Directional ACFs, when reported, average the positive and negative central-axis values and remain secondary unless the acquisition-axis metadata establish direction.

`scripts/audit_spatial_estimators.py` verifies the implementation with an analytic plane-plus-32-micrometer-cosine field. The audit checks plane removal, the two-dimensional periodogram power identity, spectral-median wavelength recovery, overlap-corrected ACF crossing, and constant-field resampling. Its machine-readable record is `results/analysis/experimental_protocol/spatial_estimator_validation.json`.

## Sensitivity operators

Changes to window footprint, detrending order, taper, radial-bin count, shortest retained wavelength, magnification, or field of view are sensitivity analyses. They must call the same centralized detrending, PSD, and ACF implementations with their altered parameters and must never silently replace the primary operator.
