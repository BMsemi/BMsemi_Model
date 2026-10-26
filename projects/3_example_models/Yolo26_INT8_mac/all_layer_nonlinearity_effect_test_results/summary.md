| Effect | Enabled hardware model | Main measured error / stress |
|---|---|---|
| Reliability | 30-day retention drift, 0.02 log-normal state spread, 10 ppm raw BER with ECC | 30,210,165,440 programmed cell reads audited; 29,965,565,094 cell-level reads changed after the retention/state model; 1 raw BER flips after ECC; 808,391,600 effective INT8 weight read entries changed; mean absolute integer branch error 2.687606; max 2,286 |
| Noise | Thermal 4 k T / R, shot 2 q I * fano, flicker/1/f, RTN, readout RMS, 5 kHz bandwidth integration | Mean equivalent branch read-noise sigma 0.026175 conductance levels; max 0.029965 levels |
| IV / ADC | hard_clamp, 4 row groups, full-scale 2,250 product levels before digitization | 29,941,977 upper saturation events; 0 lower clips; 10,293,222 clamped tile-columns; 262,320 saturated tiles; max branch level 14,599 |
| TDC | 9-bit signed-payload SAR-FP TDC, gain_shift=0 | 68,276,800 output values converted; 614,491,200 wire bits; 0 extra SAR range clips after IV limiting; mean absolute integer readout error 23.864615; max 128 |
