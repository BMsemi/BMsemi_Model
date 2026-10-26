import numpy as np
import sys
import io
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    # Some environments (e.g., Jupyter) don't support reconfigure
    pass


class CrossbarArray:
    """
    32×32 crossbar performing analog-style vector-matrix multiplication.

    Supported modes
    ---------------
    1) UNSIGNED (legacy)
        - Weight storage: (input_size, output_size) non-negative floats
        - Compute: raw = x @ W  -> output_size outputs

    2) SIGNED (legacy differential, float)
        - Input weights: (input_size, output_size//2) signed floats
        - Internal storage: (input_size, output_size) non-negative floats (even=pos, odd=neg)
        - Compute: diff = raw[0::2] - raw[1::2] -> output_size//2 outputs, padded to output_size

    3) PWM_INT8_ROW_SERIAL (NEW; per INT8xINT8 PWM row-serial PE doc)
        - Assumptions:
            * Activations a are UNSIGNED INT8 in [0,255] (e.g., post-ReLU)
            * Weights w are SIGNED INT8 in [-128,127]
            * Signed weights split into pos/neg banks: w = w_pos - w_neg (separate bank passes)
            * 4-bit cells, 8-bit ADC, radix-16 (nibble) slicing + multi-read recombine
        - Storage per logical weight: 4 x 4-bit values (w_pos0, w_pos1, w_neg0, w_neg1)
        - Compute: row-serial, two PWM passes per row (a0 then a1),
          ADC digitizes per-pass partial products, digital shift-add recombines, INT32 accumulates.
        - Output contract: returns output_size signed INT32 outputs (full crossbar utilization, no padding)
    """

    MODE_UNSIGNED = 'UNSIGNED'
    MODE_SIGNED = 'SIGNED'
    MODE_PWM_INT8_ROW_SERIAL = 'PWM_INT8_ROW_SERIAL'

    def __init__(self, input_size=32, output_size=32, *, adc_bits=8, cell_bits=4, slice_gain0=1.0, slice_gain1=1.0):
        self.input_size = int(input_size)
        self.output_size = int(output_size)

        # legacy storage (float)
        self._weights_internal = np.zeros((self.input_size, self.output_size), dtype=float)
        self._effective_signed = None
        self.mode = None

        # PWM_INT8_ROW_SERIAL storage (4 banks of 4-bit cells, column-differential)
        self.adc_bits = int(adc_bits)
        self.cell_bits = int(cell_bits)
        self.slice_gain0 = float(slice_gain0)  # optional calibration gain for slice 0
        self.slice_gain1 = float(slice_gain1)  # optional calibration gain for slice 1

        # Each nibble bank spans all output_size columns; differential via separate pos/neg bank passes
        self._w_pos0 = np.zeros((self.input_size, self.output_size), dtype=np.uint8)
        self._w_pos1 = np.zeros((self.input_size, self.output_size), dtype=np.uint8)
        self._w_neg0 = np.zeros((self.input_size, self.output_size), dtype=np.uint8)
        self._w_neg1 = np.zeros((self.input_size, self.output_size), dtype=np.uint8)

        print(f"[INIT] CrossbarArray created: {self.input_size}×{self.output_size}")
        print(f"       Mode will be auto-detected from weights\n")

    # -----------------------
    # Legacy properties
    # -----------------------
    @property
    def weights(self):
        return self._weights_internal

    @weights.setter
    def weights(self, val):
        self._weights_internal = val

    # -----------------------
    # Legacy helpers
    # -----------------------
    def _detect_mode(self, weight_matrix_float):
        has_negative = np.any(weight_matrix_float < 0)
        if has_negative:
            print(f"[AUTO-DETECT] Negative weights found → SIGNED mode")
            return self.MODE_SIGNED
        print(f"[AUTO-DETECT] All weights ≥ 0 → UNSIGNED mode")
        return self.MODE_UNSIGNED

    def _convert_to_differential(self, signed_weights):
        differential = np.zeros((self.input_size, self.output_size), dtype=float)
        differential[:, 0::2] = np.maximum(signed_weights, 0)
        differential[:, 1::2] = np.maximum(-signed_weights, 0)
        return differential

    def _print_program_info(self, original_weights):
        if self.mode == self.MODE_UNSIGNED:
            nonzero = np.count_nonzero(self._weights_internal)
            total = self.input_size * self.output_size
            print(f"[PROGRAM] UNSIGNED mode:")
            print(f"  Weight shape: {self._weights_internal.shape}")
            print(f"  Non-zero weights: {nonzero}/{total}")
            print(f"  Output size: {self.output_size}\n")
            return

        if self.mode == self.MODE_SIGNED:
            even_cols = self._weights_internal[:, 0::2]
            odd_cols = self._weights_internal[:, 1::2]
            print(f"[PROGRAM] SIGNED mode (differential):")
            print(f"  Original weight shape: {original_weights.shape}")
            print(f"  Differential storage:  {self._weights_internal.shape}")
            print(f"  Positive columns (even): {np.count_nonzero(even_cols)} non-zero")
            print(f"  Negative columns (odd):  {np.count_nonzero(odd_cols)} non-zero")
            print(f"  Differential outputs: {self.output_size // 2}  (zeros in upper half)")
            print(f"  Total output vector:  {self.output_size}  (fixed, mode-transparent)\n")
            return

        if self.mode == self.MODE_PWM_INT8_ROW_SERIAL:
            print(f"[PROGRAM] PWM_INT8_ROW_SERIAL mode (4-bit cells, 8-bit ADC, row-serial):")
            print(f"  Original signed INT8 weight shape: {original_weights.shape} (expects {self.input_size}×{self.output_size})")
            print(f"  Stored banks per logical weight: (pos0,pos1,neg0,neg1) each 4-bit")
            print(f"  Logical outputs: {self.output_size}  (full crossbar, no padding)")
            print(f"  ADC bits: {self.adc_bits}, Cell bits: {self.cell_bits}\n")
            return

    # -----------------------
    # NEW: nibble decomposition per doc
    # -----------------------
    def _decompose_signed_int8_to_4bit_banks(self, w_signed_int):
        """
        Implements:
            w_pos = max(w,0), w_neg = max(-w,0)
            w_pos = w_pos0 + 16*w_pos1, w_neg = w_neg0 + 16*w_neg1
        where each nibble is in [0,15].
        """
        w = np.asarray(w_signed_int, dtype=np.int16)
        w_pos = np.maximum(w, 0).astype(np.int16)
        w_neg = np.maximum(-w, 0).astype(np.int16)

        pos0 = (w_pos & 0xF).astype(np.uint8)
        pos1 = ((w_pos >> 4) & 0xF).astype(np.uint8)
        neg0 = (w_neg & 0xF).astype(np.uint8)
        neg1 = ((w_neg >> 4) & 0xF).astype(np.uint8)
        return pos0, pos1, neg0, neg1

    # -----------------------
    # Weight programming
    # -----------------------
    def program_weights(self, weight_matrix):
        """
        Program weights with automatic mode detection.

        - If weights are integer and shape is (input_size, output_size): programs PWM_INT8_ROW_SERIAL mode.
        - If weights are float and shape is (input_size, output_size//2): programs legacy SIGNED (differential) mode.
        - If weights are float and shape is (input_size, output_size): programs legacy UNSIGNED mode.
        """
        wm = np.asarray(weight_matrix)

        logical_out = self.output_size // 2

        # NEW: INT8 PWM row-serial mode detection (full crossbar, separate pos/neg banks)
        if wm.shape == (self.input_size, self.output_size) and np.issubdtype(wm.dtype, np.integer):
            self.mode = self.MODE_PWM_INT8_ROW_SERIAL
            # Store the effective signed weights for reference
            self._effective_signed = wm.astype(np.int16, copy=True)

            pos0, pos1, neg0, neg1 = self._decompose_signed_int8_to_4bit_banks(wm)
            self._w_pos0, self._w_pos1, self._w_neg0, self._w_neg1 = pos0, pos1, neg0, neg1

            # Keep legacy matrix cleared (not used in this mode)
            self._weights_internal.fill(0.0)
            self._print_program_info(wm)
            return

        # -----------------------
        # Legacy paths (float)
        # -----------------------
        weight_matrix = np.asarray(weight_matrix, dtype=float)

        if weight_matrix.shape == (self.input_size, logical_out):
            # Explicit SIGNED mode
            print(f"[AUTO-DETECT] Input shape {weight_matrix.shape} → SIGNED mode")
            self.mode = self.MODE_SIGNED
            self._effective_signed = weight_matrix.copy()
            self._weights_internal = self._convert_to_differential(weight_matrix)

        elif weight_matrix.shape == (self.input_size, self.output_size):
            # Auto-detect from values
            self.mode = self._detect_mode(weight_matrix)
            if self.mode == self.MODE_SIGNED:
                raise ValueError(
                    f"Cannot program a ({self.input_size}, {self.output_size}) signed weight matrix "
                    f"into a {self.output_size}-column crossbar in SIGNED (differential) mode. "
                    f"SIGNED mode requires ({self.input_size}, {logical_out}) weights "
                    f"so each signed column maps to a differential pair (even=pos, odd=neg). "
                    f"Pass a ({self.input_size}, {logical_out}) matrix instead."
                )
            self._weights_internal = weight_matrix.copy()
            self._effective_signed = None
        else:
            raise ValueError(
                f"Invalid weight shape: {weight_matrix.shape}. Expected "
                f"({self.input_size},{self.output_size}) for UNSIGNED or PWM_INT8 (integer dtype) or "
                f"({self.input_size},{logical_out}) for SIGNED (float, differential)."
            )

        self._print_program_info(weight_matrix)

    # -----------------------
    # NEW: ADC model
    # -----------------------
    def _adc_u(self, x_nonneg):
        """
        Unsigned ADC quantization with saturation.
        The doc budgets for 8-bit ADC measuring 4b×4b magnitudes (0..225).
        """
        maxv = (1 << self.adc_bits) - 1
        x = np.asarray(x_nonneg, dtype=np.int32)
        x = np.clip(x, 0, maxv)
        return x.astype(np.int32)

    def _compute_pwm_row_serial_int8(self, a_u8_vec):
        """
        Implements the doc's nibble-slicing + two-pass PWM per row:

            a = a0 + 16*a1, with a0=a&0xF, a1=a>>4
            w0 = (w_pos0 - w_neg0), w1 = (w_pos1 - w_neg1)
            p00=a0*w0, p01=a0*w1, p10=a1*w0, p11=a1*w1
            a*w = p00 + (p01<<4) + (p10<<4) + (p11<<8)

        Accumulates across rows into INT32.
        """
        logical_out = self.output_size

        a = np.asarray(a_u8_vec)
        if a.shape[0] != self.input_size:
            raise ValueError(f"Input size mismatch: expected {self.input_size}, got {a.shape[0]}")

        # Force to 0..255
        a_u = (a.astype(np.int32) & 0xFF).astype(np.int32)
        a0 = (a_u & 0xF).astype(np.int32)     # 0..15
        a1 = (a_u >> 4).astype(np.int32)      # 0..15

        acc = np.zeros((logical_out,), dtype=np.int32)

        # Row-serial: one row contributes per "integration window"
        for i in range(self.input_size):
            ai0 = int(a0[i])
            ai1 = int(a1[i])
            if ai0 == 0 and ai1 == 0:
                continue

            # Slice 0 (LS nibble) per polarity
            wpos0 = self._w_pos0[i, :].astype(np.int32)
            wneg0 = self._w_neg0[i, :].astype(np.int32)
            # Slice 1 (MS nibble) per polarity
            wpos1 = self._w_pos1[i, :].astype(np.int32)
            wneg1 = self._w_neg1[i, :].astype(np.int32)

            # Pass for a0
            if ai0 != 0:
                adc_pos00 = self._adc_u(ai0 * wpos0)
                adc_neg00 = self._adc_u(ai0 * wneg0)
                p00 = (adc_pos00 - adc_neg00).astype(np.int32)
                if self.slice_gain0 != 1.0:
                    p00 = np.rint(p00 * self.slice_gain0).astype(np.int32)

                adc_pos01 = self._adc_u(ai0 * wpos1)
                adc_neg01 = self._adc_u(ai0 * wneg1)
                p01 = (adc_pos01 - adc_neg01).astype(np.int32)
                if self.slice_gain1 != 1.0:
                    p01 = np.rint(p01 * self.slice_gain1).astype(np.int32)
            else:
                p00 = 0
                p01 = 0

            # Pass for a1
            if ai1 != 0:
                adc_pos10 = self._adc_u(ai1 * wpos0)
                adc_neg10 = self._adc_u(ai1 * wneg0)
                p10 = (adc_pos10 - adc_neg10).astype(np.int32)
                if self.slice_gain0 != 1.0:
                    p10 = np.rint(p10 * self.slice_gain0).astype(np.int32)

                adc_pos11 = self._adc_u(ai1 * wpos1)
                adc_neg11 = self._adc_u(ai1 * wneg1)
                p11 = (adc_pos11 - adc_neg11).astype(np.int32)
                if self.slice_gain1 != 1.0:
                    p11 = np.rint(p11 * self.slice_gain1).astype(np.int32)
            else:
                p10 = 0
                p11 = 0

            # Exact digital recombination (shift-add)
            # a*w = p00 + (p01<<4) + (p10<<4) + (p11<<8)
            acc += (p00 + (p01 << 4) + (p10 << 4) + (p11 << 8)).astype(np.int32)

        return acc.copy()

    # -----------------------
    # Compute
    # -----------------------
    def compute(self, input_vector):
        """
        Mode-aware MVM / MAC.
        Output is always length output_size.
        """
        if self.mode is None:
            raise RuntimeError("Weights not programmed yet. Call program_weights() first.")

        if self.mode == self.MODE_PWM_INT8_ROW_SERIAL:
            return self._compute_pwm_row_serial_int8(input_vector)

        # legacy float modes
        if len(input_vector) != self.input_size:
            raise ValueError(f"Input size mismatch: expected {self.input_size}, got {len(input_vector)}")

        raw = np.dot(np.asarray(input_vector, dtype=float), self._weights_internal)  # (output_size,)

        if self.mode == self.MODE_UNSIGNED:
            return raw

        # SIGNED
        diff = raw[0::2] - raw[1::2]  # (output_size//2,)
        result = np.zeros(self.output_size, dtype=float)
        result[: len(diff)] = diff
        return result

    # -----------------------
    # Introspection helpers
    # -----------------------
    def get_mode(self):
        return self.mode if self.mode else "NOT_PROGRAMMED"

    def get_output_size(self):
        if self.mode is None:
            raise RuntimeError("Weights not programmed yet.")
        return self.output_size

    def get_effective_weights(self):
        """
        Logical weight matrix:
          - UNSIGNED: (input_size, output_size)
          - SIGNED  : (input_size, output_size//2) signed float
          - PWM_INT8_ROW_SERIAL: (input_size, output_size) signed int16 (reference)
        """
        if self.mode is None:
            raise RuntimeError("Weights not programmed yet.")

        if self.mode == self.MODE_UNSIGNED:
            return self._weights_internal.copy()

        if self.mode == self.MODE_PWM_INT8_ROW_SERIAL:
            return self._effective_signed.copy()

        # SIGNED float: reconstruct signed from differential
        even = self._weights_internal[:, 0::2]
        odd = self._weights_internal[:, 1::2]
        return even - odd

    def get_info(self):
        info = {
            'mode': self.get_mode(),
            'input_size': self.input_size,
            'physical_output_size': self.output_size,
            'weight_shape': getattr(self._weights_internal, "shape", None),
        }
        if self.mode == self.MODE_PWM_INT8_ROW_SERIAL:
            info.update({
                'logical_outputs': self.output_size,
                'adc_bits': self.adc_bits,
                'cell_bits': self.cell_bits,
                'slice_gain0': self.slice_gain0,
                'slice_gain1': self.slice_gain1,
            })
        return info
