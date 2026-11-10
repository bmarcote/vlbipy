# Quick numbers

- Delay from a phase slope: tau = dphi / (360 deg x dnu). 1 ns gives 92 deg across 256 MHz; 360 deg across 512 MHz is
  ~2 ns.
- Ionosphere at 1.6 GHz: 1 TECU ~ 0.5 ns of delay ~ 300 deg of phase (delay ~ nu^-2, phase ~ nu^-1). Dispersive delay
  and TEC corrections matter below ~8 GHz.
- Beam: theta[mas] ~ 2.06e8 x lambda[m] / B[m]. EVN L band ~4-5 mas (e-MERLIN adds short spacings), C band ~1 mas,
  K band ~0.3 mas.
- difmap cell: <= theta/4.6 or long baselines are dropped; ~theta/10 for imaging.
- Thermal noise per baseline: sigma ~ (1/eta_s) sqrt(SEFD_i SEFD_j / (2 dnu t)), eta_s ~ 0.7-0.9 (quantisation). Use
  the EVN sensitivity calculator for image noise.
- Correlated channels: variance of an n-channel average = sigma^2 (1 + 2 rho1) / n; SFXC rho1 ~ 0.5.
- Position error (thermal): ~ theta_beam / (2 S/N).
- Coherence time: ~4-6 min at 1.4-8 GHz, ~1-2 min at 22 GHz, <~30 s at 43 GHz.
- Smearing: radial ~ (dnu/nu) r; tangential ~ 7.3e-5 rad/s x dt x r.
- Fringe-fit S/N: >~10 for the instrumental delay; 5-7 (faint) to >=10 (bright) for the global fringe fit; false
  detections grow fast below ~5. Inverse phase referencing: target S/N > 7 (down to > 3 if it recovers more baselines).
- Detection: S/N >~ 8 for an unknown position, ~5 known; target self-cal from phase-referenced S/N >~ 20.
- Self-cal interval: phase changes < a few tens of degrees within it, S/N per antenna >~ 3-5.
- A-priori amplitude accuracy ~5-10% (EVN/VLBA); typical self-cal gain factors 0.8-1.2; report cells beyond 10% / 30%.
- Bandpass normalised over the inner portion only: up to ~15% amplitude bias.
- e-MERLIN on 3C286: resolved fraction rho = 0.04 (f/5 GHz x L_shortest/11.237 km)^2, flux factor 1/(1+rho) (eMCP);
  older tutorials used a fixed 0.938.
- Opacity and fast troposphere: >~22 GHz.
