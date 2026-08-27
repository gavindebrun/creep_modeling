# Appendix: experimental roughness bandwidth and synthetic-domain requirement

## A.1. Purpose and resulting bounds

A surface-height map contains undulations of many lateral sizes: micron-scale
relief, broader grain-scale features, and still broader variation that may be
specimen form or measurement tilt rather than roughness. A simulated surface
can be compared with the experiment only over wavelengths that are both
resolved by the numerical grid and supported by the experimental
measurement. This appendix therefore answers three separate questions:

1. What is the shortest wavelength at which the surface changes
   reproducibly between the initial and final creep measurements?
2. What numerical grid spacing represents that shortest wavelength without
   materially changing its measured spectrum?
3. What is the longest wavelength that can be distinguished from the
   large-scale form removed during plane leveling?

The power spectral density (PSD), $C(\lambda)$, measures how much of the
height variation is carried by features of wavelength $\lambda$. For each
specimen, the initial and final PSDs are compared through

$$
g(\lambda)
=
\log_{10}
\left[
\frac{C_f(\lambda)}{C_0(\lambda)}
\right],
$$

where $C_0$ and $C_f$ are the initial and final PSDs. This quantity is called
the *final/initial PSD change* below. It is zero when the surface power at a
given wavelength is unchanged, positive when that wavelength gains power,
and negative when it loses power. For example, $g=1$ denotes a tenfold
increase and $g=0.1$ denotes an increase by a factor of
$10^{0.1}=1.259$. The comparison is made within each specimen before
specimens are combined, so pre-existing differences in absolute roughness do
not by themselves produce a positive change.

The retained experimental band is

$$
4.13 \leq \lambda \leq 60.87~\mu\mathrm{m}.
$$

The recommended surface sampling interval is $1~\mu\mathrm{m}$. A lateral
periodic domain containing two copies of the longest retained wavelength
requires

$$
L \geq 2\lambda_{\max}=121.74~\mu\mathrm{m}.
$$

Rounding upward to an FFT-compatible size gives a minimum candidate surface
of $128\times128~\mu\mathrm{m}^2$. Retaining the current two-to-one
build-direction aspect gives a $128\times128\times256$-cell candidate at
$1~\mu\mathrm{m}$ spacing. The factor of two is a numerical minimum that
prevents $\lambda_{\max}$ from being represented only by the fundamental
periodic mode; it is not an experimental proof that two wavelengths constitute
a representative surface area. The candidate therefore requires numerical
domain-sensitivity validation before final production calculations.

![Experimental roughness-band decision](../../figures/experimental_roughness_bandwidth.png)

**Figure A.1. Experimental determination of the retained roughness band.**
(a,b) Median specimen-level final/initial PSD change,
$g=\log_{10}(C_f/C_0)$, measured with the 10x and 50x objectives. The
horizontal dashed line is no change; positive values mean that the final
surface contains more height variation at that wavelength. Colored shading
is the 95% confidence interval for the specimen median. The 10x data show
the long-wavelength response but do not alone set the upper cutoff. The 50x
data resolve the short-wavelength response and set $\lambda_{\min}$ as the
shortest wavelength whose lower confidence bound is positive in both test
modes. Gray regions are excluded from comparison.
(c) Error caused by replacing the native 50x pixel spacing with each
candidate numerical spacing. The blue curve is the median absolute change in
$g$; the orange curve is the median absolute change in the final PSD
magnitude. Zero would reproduce the native calculation exactly. The dotted
line at $\lambda_{\min}/4=1.03~\mu\mathrm{m}$ is the largest spacing that
places four cells across the shortest retained wavelength; tested spacings
to its right are too coarse under that representation rule.
(d) Difference between the final/initial PSD change calculated from the
smaller $4\times8$ fields and the larger $2\times4$ fields,
$\Delta g=g_{4\times8}-g_{2\times4}$. Zero means that changing the
plane-leveling field did not change the inferred surface evolution.
Horizontal dotted lines mark the prescribed median tolerance of
$\pm0.10$ decades. The black vertical line is the selected upper cutoff; the
gray dash-dot line is the next wavelength bin, which fails the section-scale
test.

The four panels have different roles. Panels (a,b) establish which
wavelengths exhibit measured surface evolution. Panel (c) converts the
short-wave limit into a grid-spacing requirement. Panel (d) identifies where
the inferred long-wave evolution becomes dependent on the area over which
the surface was leveled. The figure is therefore a bandwidth decision, not a
claim that all loads share one PSD-change amplitude.

## A.2. Experimental data and endpoint selection

Only the prescribed polished datasets are used. A specimen contributes to a
paired initial/final comparison when its magnification directory contains a
map named `0.csv` and at least one later map; the largest numeric filename is
the final endpoint. The defined cohort excludes the second interrupted 525 MPa
campaign. The 588 MPa pre-test maps are excluded from change calculations, so
the 588 MPa dataset does not contribute to this bandwidth analysis.

| Test mode | Load (MPa) | Paired 10x specimens | Paired 50x specimens |
|---|---:|---:|---:|
| Interrupted | 475 | 6 | 6 |
| Interrupted | 525 | 6 | 6 |
| Interrupted | 575 | 6 | 6 |
| Uninterrupted | 500 | 25 | 0 |
| Uninterrupted | 530 | 25 | 25 |
| Uninterrupted | 588 | 0 | 0 |
| **Total** |  | **68** | **43** |

Every source map must contain $768\times1024$ numeric heights. Its header
calibration must equal $1.379951~\mu\mathrm{m}$ per pixel at 10x or
$0.278489~\mu\mathrm{m}$ per pixel at 50x within
$5\times10^{-7}~\mu\mathrm{m}$. The allowable missing-data fraction is
$10^{-4}$ and no connected missing component may exceed one pixel. An
isolated missing value is replaced by its nearest finite neighbor before
analysis. Maps failing any acquisition criterion are rejected.

## A.3. Leveling and observation fields

For an observation field $z(x,y)$, a least-squares plane is fitted and
subtracted:

$$
z_\mathrm{L}(x,y)=z(x,y)-(a+bx+cy).
$$

The fitted plane represents the mean height and linear scan tilt, not
roughness. Subtracting it prevents mounting angle or specimen-scale slope
from entering the PSD. Because the plane fitted to a field also removes part
of any wavelength comparable to that field, the field dimensions are an
analysis parameter that must be stated and tested.

Leveling is performed independently for every field before any PSD or ACF
calculation. No numerical-solver quantity enters field selection or
leveling.

The primary 10x operator divides each map into a $2\times4$ grid. Its eight
$384\times256$-pixel fields span
$529.901\times353.267~\mu\mathrm{m}^2$. A section-scale sensitivity operator
divides the same map into a $4\times8$ grid, producing 32
$192\times128$-pixel fields spanning
$264.951\times176.634~\mu\mathrm{m}^2$. The initial/final comparison always uses
spatially corresponding partitions.

The term *fine partition* refers only to the larger number of smaller
leveling fields in the $4\times8$ tiling; it does not mean finer pixel
spacing. The $2\times4$ tiling is the primary experimental operator used for
10x roughness statistics. The $4\times8$ tiling is applied only as a
sensitivity calculation. If a wavelength represents local surface
morphology, its final/initial PSD change should be similar when the same map
is divided and leveled using either tiling. A large difference indicates
that the inferred change is coupled to the leveling-window size.

Before profilometry, a 0.1-kgf Vickers indentation was placed near the center
of the observation area so that approximately the same surface location
could be found at successive times. This deliberately imposed, pyramidal
depression is the *fiducial*. It is useful for relocation but is not
creep-generated roughness. Because the 50x field of view is only
approximately $285\times214~\mu\mathrm{m}^2$, the central indentation
occupies a substantial part of the map and would dominate its PSD.

The 50x calculation therefore uses four square fields taken from the map
corners, as shown in Figure A.2(c). These are called *peripheral fields*
because they lie around the periphery of the map, away from the central
indentation. Each field contains $229\times229$ pixels and spans
$63.774\times63.774~\mu\mathrm{m}^2$. The four fields are plane-leveled
independently, their PSDs are calculated separately, and their median is used
as the specimen result. This retains the high-resolution surface texture
needed to determine $\lambda_{\min}$ without treating the reference
indentation as morphology.

The autocorrelation function (ACF) supplies a separate measure of
long-wavelength support. For a separation distance $r$, the radial ACF
$\rho(r)$ measures how similar surface heights remain, on average, to
heights a distance $r$ away. It begins at $\rho(0)=1$. Two landmarks are
reported:

- the e-folding distance, where $\rho$ first falls to
  $1/e\approx0.368$, is a characteristic decay length of the correlated
  morphology; and
- the first zero crossing, where $\rho$ first reaches zero, is the distance
  beyond which the surface no longer retains positive height correlation.

The e-folding distance describes the typical correlation scale but does not
set $\lambda_{\min}$. The zero crossing is the relevant upper-scale check:
a wavelength much longer than the distance over which heights remain
correlated is not supported as a repeatable morphological scale.

![Sectioning and ACF support](../../figures/experimental_roughness_bandwidth_method.png)

**Figure A.2. Leveling domains and independent correlation-length evidence.**
(a) Primary $2\times4$ partition and (b) sensitivity-only $4\times8$
partition over the same representative 530 MPa final 10x map. The displayed
map is globally leveled only for visualization; every red-bounded field
receives its own plane fit in the calculation. The smaller fields in (b)
remove large-scale variation more aggressively, enabling the leveling-scale
test in Figure A.1(d).
(c) Final 50x map showing the central Vickers indentation and the four blue
$64\times64~\mu\mathrm{m}^2$ corner fields used for the short-wavelength PSD.
The indentation is excluded from these fields.
(d) Empirical cumulative distributions of specimen-level ACF landmarks. At
an abscissa $x$, the ordinate is the fraction of specimens whose landmark
distance is less than or equal to $x$. Color identifies interrupted or
uninterrupted tests. Solid curves are first zero crossings
($\rho=0$); dashed curves are e-folding distances ($\rho=1/e$). Thus there
are two curves of each color because two different ACF landmarks are shown
for each test mode. Each specimen value is the median from four independently
leveled endpoint patches. The black dotted line is
$\lambda_{\max}=60.87~\mu\mathrm{m}$, and the gray dash-dot line is the next
PSD wavelength bin, $77.13~\mu\mathrm{m}$, which fails the section-scale
test.

Panel (d) does not determine the short-wave limit. Its role is to check the
upper limit independently of the initial/final PSD ratio. Across all
specimens, the pooled median e-folding distance is
$19.55~\mu\mathrm{m}$, showing that the characteristic correlation decay is
well inside the proposed domain. The median first zero crossing is
$58.20~\mu\mathrm{m}$, close to the selected
$60.87~\mu\mathrm{m}$ cutoff. Its 95th percentile is
$77.10~\mu\mathrm{m}$, nearly identical to the first longer PSD bin that
fails the section-scale test. The ACF and sectioning calculations therefore
place the loss of reliable long-wave support in the same interval,
$60.87$--$77.13~\mu\mathrm{m}$, despite being calculated from different
quantities.

## A.4. Two-dimensional PSD calculation

The two-dimensional Fourier transform represents the surface using waves
with spatial frequencies $f_x$ and $f_y$. A Fourier coefficient at
$(f_x,f_y)$ describes a wave with a particular direction and wavelength. The
present analysis seeks an orientation-averaged spectrum, so coefficients are
grouped by radial spatial frequency,

$$
f=\sqrt{f_x^2+f_y^2}.
$$

An *annulus* is the ring-shaped region of this two-dimensional frequency
plane between radial frequencies $f_l$ and $f_r$. Averaging the PSD
coefficients in an annulus combines waves of similar spatial size but
different direction. Each annulus is assigned the geometric-center
frequency $\sqrt{f_lf_r}$ and wavelength

$$
\lambda=\frac{1}{\sqrt{f_lf_r}}.
$$

The reported spectrum therefore contains discrete wavelength bins rather
than every possible wavelength. A boundary such as
$\lambda=4.1264~\mu\mathrm{m}$ is the center of one bin; the corresponding
bin edges are reported with the boundary.

For every leveled field, its mean is removed and a separable two-dimensional
Hann taper is applied:

$$
w_{ij}=w_i^{(\mathrm{Hann})}w_j^{(\mathrm{Hann})}.
$$

The discrete two-dimensional PSD is

$$
C(f_x,f_y)
=
\frac{\Delta^2}
{\sum_{ij}w_{ij}^2}
\left|
\mathcal{F}
\left\{
w_{ij}\left[z_{\mathrm{L},ij}-\overline{z_\mathrm{L}}\right]
\right\}
\right|^2,
$$

where $\Delta$ is the lateral pixel spacing. The resulting PSD has units of
$\mu\mathrm{m}^4$. The Hann taper smoothly reduces the field to zero at its
edges and limits spectral leakage caused by treating opposite field edges as
adjacent in the discrete Fourier transform. The radial PSD in an annulus
$[f_l,f_r)$ is the arithmetic mean of every two-dimensional PSD coefficient
in that annulus.

The primary 10x spectrum uses 22 logarithmic annuli with edges from
$0.00283072$ to $0.36~\mu\mathrm{m}^{-1}$. The section-scale comparison uses
16 logarithmic annuli from $0.00566143$ to
$0.25~\mu\mathrm{m}^{-1}$ so that both partitions populate identical
frequency bins. The 50x spectrum uses 23 annuli derived from logarithmic
edges between $0.015$ and $1.5~\mu\mathrm{m}^{-1}$; removal of the third
original edge merges its two adjacent intervals.

For specimen $s$, the median field PSD is calculated independently at the
initial and final endpoints:

$$
\widetilde C_{s,0}(f)
=\operatorname{median}_{k}\left[C_{s,0,k}(f)\right],
\qquad
\widetilde C_{s,f}(f)
=\operatorname{median}_{k}\left[C_{s,f,k}(f)\right].
$$

The specimen-level spectral change is

$$
g_s(f)
=
\log_{10}
\left[
\frac{\widetilde C_{s,f}(f)}
{\widetilde C_{s,0}(f)}
\right].
$$

Every subtraction or ratio is paired. For the final/initial change,
$\widetilde C_{s,0}$ and $\widetilde C_{s,f}$ come from the same specimen,
the same physical field positions, and the same frequency annulus. For the
section-scale calculation, $g_s^{(2\times4)}$ and
$g_s^{(4\times8)}$ are calculated from the same specimen's original initial
and final maps and compared at the same annulus. A *paired difference*
therefore means subtraction of two results derived from the same specimen;
it is not a difference between unrelated specimens.

Specimens, rather than individual sections, are the independent statistical
units. For each test mode and frequency annulus, the reported center is the
median of $g_s$. A percentile 95% confidence interval is obtained from 2000
bootstrap resamples of the specimens with replacement. The fixed random seed
is 20260729. Interrupted and uninterrupted specimens are resampled and tested
separately.

### Interpretation of the confidence bands

The shaded regions in Figure A.1(a,b) are confidence intervals for the
across-specimen median final/initial PSD change. They are not pixel noise,
uncertainty in a single PSD, or the spread among the fields cut from one map.
The interrupted intervals are estimated from 18 specimens distributed among
three loads. The uninterrupted intervals use 50 specimens at 10x and 25
specimens at 50x. The interrupted intervals are consequently wider because
they combine a smaller cohort with strong specimen- and load-dependent
changes in the PSD ratio.

The effect is largest near the short-wave boundary. At
$4.1264~\mu\mathrm{m}$, the interrupted specimen-median final/initial change
is 0.7111 decades with a 95% confidence interval of
$[0.0094,3.8594]$ decades; the uninterrupted value is 0.1110 decades with an interval of
$[0.0285,0.2279]$ decades. The large interrupted upper bound reflects the
observed mixture of weak changes and several multi-decade increases in
short-wavelength power. The log ratio becomes especially large when the
initial PSD in a bin is small and the final PSD is finite. The interval must
not be read as a precise estimate of a common change magnitude.

The bandwidth decision remains useful because it does not require a precise
common magnitude:

1. The short-wave test uses the lower confidence bound, not the median or
   upper bound. Greater heterogeneity therefore makes a wavelength harder,
   rather than easier, to retain. Both modes remain above zero at
   $4.1264~\mu\mathrm{m}$; at the next shorter annulus,
   $3.4060~\mu\mathrm{m}$, the intervals are
   $[-0.0501,3.7490]$ and $[-0.0264,0.1478]$ decades and both include zero.
2. The long-wave test uses the paired difference between two sectioning
   operators applied to the same specimen. This pairing removes the
   specimen's absolute change amplitude from the comparison and tests only
   whether the inferred final/initial PSD change depends on leveling area.
3. The upper bound is also checked against endpoint ACF lengths, which are
   calculated from the final morphology and do not contain the potentially
   unstable division by a small initial PSD.

The uncertainty does limit the strength of the upper-bound statement. At
$60.8714~\mu\mathrm{m}$, the interrupted median section difference is
0.0887 decades but its interval, $[-0.0362,0.2222]$ decades, does not exclude
a practically important positive section effect. Thus the data demonstrate
no statistically detectable directional shift at this annulus; they do not
establish section-size equivalence to within $\pm0.10$ decades with 95%
confidence. The uninterrupted interval is narrower,
$[-0.0381,0.0842]$ decades. At $77.1306~\mu\mathrm{m}$, both mode-specific
intervals exclude zero and the uninterrupted median exceeds 0.10 decades,
so the section dependence is unambiguous there.

Accordingly, $60.87~\mu\mathrm{m}$ is an operational upper cutoff supported
by the intersection of paired PSD and independent ACF evidence, not an exact
material constant. The large confidence bands prohibit a universal
amplitude calibration from these pooled spectra, but they do not erase the
observed transition from a common positive short-wave change to
section-dependent long-wave variation. Any simulation comparison must use
the same magnification-specific sectioning and leveling operators and should
compare mode- or load-resolved distributions rather than only the pooled
median curve.

## A.5. Short-wavelength boundary

A wavelength is useful for validating simulated roughness evolution only if
the experiment resolves a reproducible change at that wavelength. A
wavelength whose final/initial change cannot be distinguished from zero
would ask the simulation to reproduce measurement scatter or unchanged
surface texture rather than creep-associated morphology. The lower boundary
is therefore not the microscope's smallest resolvable wavelength. It is the
shortest wavelength at which both experimental test modes show evidence of
surface evolution.

A frequency annulus is classified as containing reproducible
creep-associated change only when the lower 95% bootstrap confidence bound
of the specimen-median $g=\log_{10}(C_f/C_0)$ is greater than zero
independently for the interrupted and uninterrupted groups. Requiring both
groups prevents a short-wave response unique to one acquisition protocol
from setting the common comparison bandwidth. The shortest annulus
satisfying that requirement is centered at

$$
\lambda_{\min}=4.1264~\mu\mathrm{m},
$$

with wavelength edges of $3.7489$ and $4.5419~\mu\mathrm{m}$. At the next
shorter annulus, centered at $3.4060~\mu\mathrm{m}$, both confidence
intervals include zero. The short-wave boundary therefore follows measured
surface change rather than the native 50x Nyquist limit. It should be
interpreted as the center of the shortest supported wavelength bin, not as
an exact physical feature diameter.

## A.6. Long-wavelength boundary

Plane leveling removes the best-fit tilt from every field. As the wavelength
approaches the field size, the distinction between surface morphology and
the removed plane becomes operator dependent. The long-wave decision
therefore asks whether the measured final/initial PSD change remains similar
when the *same maps* are divided into larger primary fields or smaller fine
fields. It then checks that candidate against the independent ACF
correlation range.

For every specimen and frequency annulus, the change caused by reducing the
leveling field is

$$
\Delta g_s(f)
=g_s^{(4\times8)}(f)-g_s^{(2\times4)}(f).
$$

Here, $\Delta g_s=0$ means that the two field sizes give the same
final/initial PSD change. A positive value means that the smaller fields
produce a larger inferred change; a negative value means that they produce a
smaller inferred change. A section-scale-consistent candidate must satisfy
all of the following:

1. The absolute median $\Delta g_s$ is no greater than 0.10 decades in each
   test mode. The median result may therefore change by no more than a factor
   of $10^{0.10}=1.259$ when the leveling field is changed.
2. The bootstrap 95% confidence interval of $\Delta g_s$ contains zero in
   each test mode. This rejects wavelengths having a statistically
   detectable, consistently signed field-size shift.
3. The wavelength does not exceed the pooled 95th percentile of the
   specimen-level ACF zero crossing, $77.103~\mu\mathrm{m}$. This rejects
   wavelengths longer than the positive-correlation range of nearly all
   measured specimens.

The largest annulus satisfying all three conditions is centered at
$60.8714~\mu\mathrm{m}$ and spans wavelengths from $54.0763$ to
$68.5205~\mu\mathrm{m}$. At this annulus, the median fine-minus-primary
PSD-change difference is 0.0887 decades for interrupted specimens and 0.0374
decades for uninterrupted specimens. Their respective 95% confidence intervals are
$[-0.0362,0.2222]$ and $[-0.0381,0.0842]$ decades.

Thus $60.8714~\mu\mathrm{m}$ satisfies the implemented criteria because
$|0.0887|<0.10$ and $|0.0374|<0.10$, both confidence intervals contain zero,
and $60.8714<77.103~\mu\mathrm{m}$. The relatively wide interrupted
confidence interval is an important limitation: the median passes the
practical tolerance and no consistently signed shift is detected, but the
data do not prove that every plausible interrupted-population shift lies
within $\pm0.10$ decades.

| Requirement | $60.8714~\mu\mathrm{m}$ | $77.1306~\mu\mathrm{m}$ |
|---|---|---|
| $|\operatorname{median}(\Delta g)|\leq0.10$ in both modes | 0.0887 interrupted; 0.0374 uninterrupted: **pass** | 0.0862 interrupted; 0.1185 uninterrupted: **fail** |
| Both 95% intervals contain zero | $[-0.0362,0.2222]$; $[-0.0381,0.0842]$: **pass** | $[0.0079,0.1527]$; $[0.0260,0.1745]$: **fail** |
| $\lambda\leq77.103~\mu\mathrm{m}$ ACF upper-tail limit | **pass** | **fail** |

The next annulus is centered at $77.1306~\mu\mathrm{m}$ and spans
$68.5205$ to $86.8228~\mu\mathrm{m}$. Its median changes are 0.0862 and
0.1185 decades, with confidence intervals
$[0.0079,0.1527]$ and $[0.0260,0.1745]$ decades. Both intervals exclude
zero, and the uninterrupted median exceeds the prescribed bias limit.
Consequently, the apparent spectral response at this and larger wavelengths
depends on the leveling area and is not retained as roughness.

The annulus-by-annulus section differences are not monotonic because tapered
radial PSD bins are noisy and statistically coupled. The procedure therefore
does not fit a sharp physical change point. It selects the largest tested
wavelength below the ACF support limit that satisfies the paired
section-scale criteria. The resulting $\lambda_{\max}$ is an operational
comparison cutoff, and simulations must still be processed with the same
field and leveling operator as the corresponding experiment.

## A.7. Autocorrelation calculation

The ACF is calculated from four peripheral
$256\times256~\mu\mathrm{m}^2$ fields of each valid final 10x map. Each patch
is independently plane-leveled. For a leveled field $z_\mathrm{L}$, the
two-dimensional covariance is evaluated by FFT convolution and divided
pointwise by the number of overlapping pixel pairs. It is normalized by the
zero-lag covariance:

$$
\rho(\Delta x,\Delta y)
=
\frac{
\left\langle
z_\mathrm{L}(x,y)
z_\mathrm{L}(x+\Delta x,y+\Delta y)
\right\rangle
}{
\left\langle z_\mathrm{L}^2\right\rangle
}.
$$

The radial ACF is the mean covariance in annuli of width one native 10x pixel,
$1.379951~\mu\mathrm{m}$. Radial lags extend from zero to less than
$128~\mu\mathrm{m}$. Linear interpolation between adjacent lag samples gives
the first crossing of $\rho=e^{-1}$ and the first crossing of $\rho=0$. The
specimen statistic is the median crossing length from its four patches.

Across the 68 specimens, the median e-folding length is
$19.550~\mu\mathrm{m}$ and its 95th percentile is
$31.153~\mu\mathrm{m}$. The median zero crossing is
$58.199~\mu\mathrm{m}$ and its 95th percentile is
$77.103~\mu\mathrm{m}$. The ACF median agrees with the final
section-scale-consistent PSD annulus, while its upper tail coincides with the
first section-scale-dependent annulus.

The ACF landmarks are distances, not PSD wavelengths, and are not converted
into wavelengths. They are used only as an independent spatial-support
check. In particular, neither ACF landmark determines $\lambda_{\min}$.
The e-folding distance describes the typical decay of local correlation,
whereas the zero-crossing distribution checks whether a proposed
$\lambda_{\max}$ extends beyond the observed positive-correlation range.

## A.8. Resolution sensitivity

Native 50x fields are resampled to candidate spacings of 0.5, 0.75, 1.0,
1.25, 1.5, and $2.0~\mu\mathrm{m}$. For coarsening ratio
$r=\Delta_\mathrm{target}/\Delta_\mathrm{native}>1$, the source field is
Gaussian-filtered with a standard deviation of
$0.5(r-1)$ source pixels and then bilinearly resampled. Only frequencies
below the candidate Nyquist frequency,
$0.5/\Delta_\mathrm{target}$, enter the comparison.

The resampling calculation defines two errors at every retained wavelength:

$$
b_g
=
g_\mathrm{resampled}-g_\mathrm{native}
=
\log_{10}
\left[
\frac{(C_f/C_0)_\mathrm{resampled}}
{(C_f/C_0)_\mathrm{native}}
\right],
$$

and

$$
b_C
=
\log_{10}
\left[
\frac{C_{f,\mathrm{resampled}}}
{C_{f,\mathrm{native}}}
\right].
$$

The absolute values $|b_g|$ and $|b_C|$ are errors because the native 50x
calculation is the reference: zero means that resampling changes neither the
measured evolution nor the final PSD magnitude. Logarithmic error treats
equal multiplicative increases and decreases symmetrically. For example,
0.10 decades corresponds to a factor of 1.259 in either direction. Figure
A.1(c) shows the median absolute error across all specimens and all
wavelength bins satisfying $\lambda\geq\lambda_{\min}$. Its blue curve is
$|b_g|$ and its orange curve is $|b_C|$.

Two requirements determine the selected spacing. First, the grid is required
to contain at least four cells across the shortest retained wavelength:

$$
\frac{\lambda_{\min}}{\Delta}
=
\frac{4.1264}{1.0}
=4.13.
$$

This is stricter than the two-cell Nyquist detection limit and is adopted so
that the shortest retained wave is represented by more than alternating
grid values. It limits the spacing to

$$
\Delta
\leq
\frac{\lambda_{\min}}{4}
=1.0316~\mu\mathrm{m}.
$$

Second, explicit resampling must not introduce a large spectral error.
Among the tested candidates, $1~\mu\mathrm{m}$ is the largest spacing that
satisfies the four-cell condition. It gives a median absolute
final/initial-change error of 0.0066 decades and a median absolute final-PSD
error of 0.0132 decades, corresponding to multiplicative errors of
approximately 1.5% and 3.1%, respectively. The 95th-percentile absolute
final-PSD error is 0.1430 decades, or a factor of 1.39, showing that some
specimen-frequency combinations remain more sensitive than the median.
Coarser tested spacings violate the four-cell condition and show increasing
final-PSD error. The $1~\mu\mathrm{m}$ conclusion therefore follows from
both the experimentally selected $\lambda_{\min}$ and the native-data
resampling test; it is not inferred from microscope pixel pitch alone.

The 10x and 50x absolute PSDs are not stitched because their common-field
spectra do not quantitatively overlap. The 50x data determine
$\lambda_{\min}$ and the 10x data determine $\lambda_{\max}$. Numerical
comparisons must reproduce the field extraction, leveling, tapering, radial
binning, and magnification-specific operator used for the corresponding
experimental objective.

## A.9. Domain conversion and computational consequence

The $128\times128~\mu\mathrm{m}^2$ lateral candidate contains 2.10 copies of
$\lambda_{\max}$ in each in-plane direction. A $64~\mu\mathrm{m}$ periodic
domain would contain the upper wavelength only as an approximately
one-cycle fundamental mode and would provide inadequate directional and
radial sampling near the cutoff.

Cost is projected from the measured 588 MPa
$128\times64\times256$-cell calculation, which required 4729 s on seven
local threads. Scaling comparative FFT work as $N\log N$ gives approximately
2.75 h for $128\times128\times256$ on the same local reference. This is not
an HPC timing prediction. Profilometry constrains the two surface dimensions;
the $256~\mu\mathrm{m}$ depth retains the current build-direction aspect and
must be treated as a separate numerical boundary-sensitivity choice.

## A.10. Reproduction and generated evidence

Run:

```bash
python scripts/analyze_experimental_roughness_bandwidth.py
```

The script writes both figures as PNG and PDF to `figures/` and
`docs/manuscript/figs/`. It writes the following numerical evidence to
`results/analysis/experimental_roughness_bandwidth/`:

- `specimen_spectral_gain.csv`: initial PSD, final PSD, and
  $g=\log_{10}(C_f/C_0)$ for
  every specimen and radial annulus;
- `spectral_gain_summary.csv`: group medians and specimen-bootstrap
  confidence intervals;
- `section_scale_pairs.csv`: paired primary/fine PSD and final/initial
  PSD-change differences for every specimen;
- `section_scale_summary.csv`: section-scale median differences and
  confidence intervals;
- `endpoint_acf_scales.csv`: specimen-level e-folding and zero-crossing
  lengths;
- `resolution_sensitivity.csv` and `resolution_summary.csv`: native-to-grid
  spectral changes;
- `runtime_projection.csv`: comparative domain-cost projection; and
- `decision.json`: the machine-readable selected bounds and grid.

The treatment of finite measurement bandwidth, windowing, scale-dependent
roughness, background-subtraction bias, and representative observation area
is informed by
`docs/papers/jacobs2017_surface_topography_spectral_analysis.pdf`,
`docs/papers/sanner2022_scale_dependent_roughness.pdf`,
`docs/papers/necas2020_roughness_measurement_bias.pdf`, and
`docs/papers/singh2024_surface_roughness_representative_area.pdf`.
