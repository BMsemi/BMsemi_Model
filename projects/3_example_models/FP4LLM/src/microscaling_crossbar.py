"""Adapter to the project's CrossbarArray simulator (no board I/O)."""
import numpy as np

from reram_crossbar import CrossbarArray


class MicroscalingCrossbar:
    """One persistent simulated array per weight tile, programmed once.

    Physical columns include both differential banks; the default 32 columns
    accommodate 16 logical outputs. Edge tiles are zero padded. No packing of
    multiple scale groups into one array is attempted.
    """

    def __init__(self, *, rows=32, physical_columns=32, dac_bits=2, adc_bits=10,
                 noise_config=None, reliability_config=None, iv_config=None, seed=42):
        if any(not isinstance(v, (int, np.integer)) for v in (rows, physical_columns, dac_bits, adc_bits, seed)):
            raise ValueError("Array dimensions, converter widths, and seed must be integers")
        if seed < 0:
            raise ValueError("Seed must be nonnegative")
        if rows <= 0 or physical_columns <= 0 or physical_columns % 2:
            raise ValueError("Crossbar rows must be positive and physical columns positive/even")
        if not 1 <= dac_bits <= 6 or not 1 <= adc_bits <= 30:
            raise ValueError("DAC bits must be 1..6 and ADC bits 1..30")
        self.rows, self.physical_columns = int(rows), int(physical_columns)
        self.dac_bits, self.adc_bits = int(dac_bits), int(adc_bits)
        self.config = dict(noise_config=noise_config, reliability_config=reliability_config,
                           iv_config=iv_config)
        self.seed = int(seed)
        self.tiles = {}
        self._active = None

    def prepare_tile(self, key, positive, negative):
        positive, negative = np.asarray(positive), np.asarray(negative)
        if positive.ndim != 2 or positive.shape != negative.shape or not positive.size:
            raise ValueError("Cell banks must be matching nonempty matrices")
        for cells in (positive, negative):
            if not np.issubdtype(cells.dtype, np.integer) or np.any(cells < 0) or np.any(cells > 15):
                raise ValueError("Each physical cell must be an integer in [0,15]")
        if np.any((positive != 0) & (negative != 0)):
            raise ValueError("INT5 adapter expects canonical positive/negative cell pairs")
        nr, nc = positive.shape
        if nr > self.rows or 2 * nc > self.physical_columns:
            raise ValueError("Microscaling tile exceeds physical crossbar capacity; reduce block size or increase array dimensions")
        signed = positive.astype(np.int16) - negative.astype(np.int16)
        if key not in self.tiles:
            tile_seed = int(np.random.SeedSequence([self.seed, *key]).generate_state(1)[0])
            cb = CrossbarArray(input_size=self.rows, output_size=self.physical_columns,
                               cell_bits=4, adc_bits=self.adc_bits,
                               noise_seed=tile_seed, reliability_seed=tile_seed, **self.config)
            padded = np.zeros((self.rows, self.physical_columns // 2), dtype=np.int16)
            padded[:nr, :nc] = signed
            cb.program_int5_weights(padded)
            self.tiles[key] = (cb, signed.copy())
        elif not np.array_equal(self.tiles[key][1], signed):
            raise ValueError("Tile weights changed; create a new adapter to reprogram")
        self._active = self.tiles[key]

    def __call__(self, positive, negative, pulses):
        if self._active is None:
            raise RuntimeError("prepare_tile must be called before a crossbar read")
        cb, signed = self._active
        if not np.array_equal(np.asarray(positive) - np.asarray(negative), signed):
            raise ValueError("Read cell banks do not match the prepared tile")
        x = np.asarray(pulses)
        if x.shape != (signed.shape[0],) or not np.issubdtype(x.dtype, np.integer):
            raise ValueError("Expected one integer pulse per active tile row")
        padded = np.zeros(self.rows, dtype=np.int64)
        padded[:x.size] = x
        return cb.compute_int5_pulses(padded, dac_bits=self.dac_bits)[:signed.shape[1]]

    def report(self):
        infos = [{"tile": list(key), "active_shape": list(w.shape), **cb.get_info()}
                 for key, (cb, w) in self.tiles.items()]
        return {"implementation": "reram_crossbar.CrossbarArray", "physical_hardware": False,
                "programmed_arrays": len(infos),
                "programmed_cells": sum(i["reliability_stats"]["programmed_cells"] for i in infos),
                "read_calls": sum(i["int5_read_calls"] for i in infos),
                "adc_clip_events": sum(i["int5_adc_clip_events"] for i in infos),
                "tiles": infos}
