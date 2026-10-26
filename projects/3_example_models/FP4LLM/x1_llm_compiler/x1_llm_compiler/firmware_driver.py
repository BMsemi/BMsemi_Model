from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from .fabric import DescriptorOp, StreamPlan, verify_descriptor_words


FIRMWARE_YAML_TEMPLATE = """# SPDX-License-Identifier: Apache-2.0

Tests:
    - {name: __TEST_NAME__, sim: RTL}
"""


FIRMWARE_C_TEMPLATE = r'''#define USER_ADDR_SPACE_C_HEADER_FILE

#include <firmware_apis.h>
#include <custom_user_space.h>
#include <stdint.h>

#define CSR_VERSION 0u
#define CSR_CONFIG  1u
#define CSR_STATUS  2u
#define CSR_FLIT0   8u
#define CSR_FLIT1   9u
#define CSR_FLIT2   10u
#define CSR_FLIT3   11u
#define CSR_DESC_INDEX  13u
#define CSR_DESC_DATA   14u
#define CSR_DESC_CTRL   15u
#define CSR_DESC_STATUS 16u
#define CSR_DESC_RESULT 17u
#define CSR_WINDOW_INDEX 19u
#define CSR_WINDOW_DATA  20u

#define STATUS_BUSY   0x01u
#define STATUS_INIT   0x02u
#define STATUS_NOC    0x08u
#define STATUS_ERROR  0x10u

#define DESC_DONE        0x02u
#define DESC_ERROR       0x04u
#define DESC_CHECKSUM_OK 0x08u

#define DESC_CTRL_START_CSR 0x01u
#define DESC_CTRL_CLEAR     0x02u

#define DESCRIPTOR_WORDS 16u
#define DESCRIPTOR_COUNT __DESCRIPTOR_COUNT__u
#define WINDOW_WORDS __WINDOW_WORD_COUNT__u
#define EXPECTED_NUM_MACROS __EXPECTED_NUM_MACROS__u
#define EXPECTED_ROLE_ID __EXPECTED_ROLE_ID__u
#define EXPECTED_BITS_PER_VALUE __EXPECTED_BITS_PER_VALUE__u
#define EXPECTED_TILE_GROUP __EXPECTED_TILE_GROUP__u
#define EXPECTED_DST __EXPECTED_DST__u

static const uint32_t descriptor_queue[DESCRIPTOR_COUNT][DESCRIPTOR_WORDS] = {
__DESCRIPTOR_QUEUE__
};

static const uint32_t descriptor_op_ids[DESCRIPTOR_COUNT] = {
__DESCRIPTOR_OP_IDS__
};

static const uint32_t packed_window_words[WINDOW_WORDS] = {
__WINDOW_WORDS__
};

static const uint32_t expected_payload_values[EXPECTED_NUM_MACROS] = {
__EXPECTED_PAYLOAD_VALUES__
};

static inline void wait_cycles(uint32_t cycles)
{
    for (uint32_t i = 0u; i < cycles; i++) {
        __asm__ volatile ("nop");
    }
}

static uint32_t descriptor_checksum(const uint32_t *words)
{
    uint32_t value = 0x58315550u;
    for (uint32_t i = 0u; i < 14u; i++) {
        value = (value << 5) | (value >> 27);
        value ^= words[i];
    }
    return value;
}

static uint32_t wait_init(void)
{
    uint32_t status = 0u;
    for (uint32_t i = 0u; i < 300000u; i++) {
        status = USER_readWord(CSR_STATUS);
        if ((status & STATUS_INIT) && !(status & STATUS_BUSY)) {
            return status;
        }
        wait_cycles(4u);
    }
    return status | STATUS_ERROR;
}

static uint32_t wait_descriptor_done(void)
{
    uint32_t status = 0u;
    for (uint32_t i = 0u; i < 500000u; i++) {
        status = USER_readWord(CSR_DESC_STATUS);
        if ((status & DESC_DONE) != 0u) {
            return status;
        }
        wait_cycles(4u);
    }
    return status | DESC_ERROR;
}

static void fail(void)
{
    ManagmentGpio_write(0);
    while (1) { }
}

static void require_true(uint32_t condition)
{
    if (!condition) {
        fail();
    }
}

static void write_packed_window(void)
{
    USER_writeWord(0u, CSR_WINDOW_INDEX);
    for (uint32_t i = 0u; i < WINDOW_WORDS; i++) {
        USER_writeWord(packed_window_words[i], CSR_WINDOW_DATA);
    }
    for (uint32_t i = 0u; i < WINDOW_WORDS; i++) {
        USER_writeWord(i, CSR_WINDOW_INDEX);
        require_true(USER_readWord(CSR_WINDOW_DATA) == packed_window_words[i]);
    }
    USER_writeWord(0u, CSR_WINDOW_INDEX);
}

static void load_descriptor(uint32_t queue_index)
{
    USER_writeWord(0u, CSR_DESC_INDEX);
    for (uint32_t i = 0u; i < DESCRIPTOR_WORDS; i++) {
        USER_writeWord(descriptor_queue[queue_index][i], CSR_DESC_DATA);
    }
    for (uint32_t i = 0u; i < DESCRIPTOR_WORDS; i++) {
        USER_writeWord(i, CSR_DESC_INDEX);
        require_true(USER_readWord(CSR_DESC_DATA) == descriptor_queue[queue_index][i]);
    }
    USER_writeWord(0u, CSR_DESC_INDEX);
}

static uint32_t run_loaded_descriptor(uint32_t expected_op)
{
    USER_writeWord(DESC_CTRL_CLEAR, CSR_DESC_CTRL);
    USER_writeWord(DESC_CTRL_START_CSR, CSR_DESC_CTRL);
    uint32_t status = wait_descriptor_done();
    require_true((status & DESC_ERROR) == 0u);
    require_true((status & DESC_CHECKSUM_OK) != 0u);
    require_true(((status >> 8) & 0xFFu) == expected_op);
    return USER_readWord(CSR_DESC_RESULT);
}

static uint32_t payload_value(uint32_t flit2, uint32_t flit3, uint32_t idx, uint32_t bits)
{
    uint32_t mask = (1u << bits) - 1u;
    uint32_t bit_pos = idx * bits;
    uint32_t value;
    if (bit_pos < 32u) {
        value = flit2 >> bit_pos;
        if ((bit_pos + bits) > 32u) {
            value |= flit3 << (32u - bit_pos);
        }
    } else {
        value = flit3 >> (bit_pos - 32u);
    }
    return value & mask;
}

static void drain_and_check_noc_result(void)
{
    uint32_t status = USER_readWord(CSR_STATUS);
    uint32_t flit0 = USER_readWord(CSR_FLIT0);
    uint32_t flit1 = USER_readWord(CSR_FLIT1);
    uint32_t flit2 = USER_readWord(CSR_FLIT2);
    uint32_t flit3 = USER_readWord(CSR_FLIT3);

    uint32_t fmt_id = (flit1 >> 28) & 0xFu;
    uint32_t role_id = (flit1 >> 24) & 0xFu;
    uint32_t bits_per_value = (flit1 >> 19) & 0x1Fu;
    uint32_t coarse_bits = (flit1 >> 16) & 0x7u;
    uint32_t fine_bits = (flit1 >> 13) & 0x7u;
    uint32_t value_count = ((flit1 & 0xFFu) << 2) | ((flit0 >> 30) & 0x3u);
    uint32_t tile_group = (flit0 >> 18) & 0xFFFu;

    require_true((status & STATUS_NOC) != 0u);
    require_true(fmt_id == 1u);
    require_true(role_id == EXPECTED_ROLE_ID);
    require_true(bits_per_value == EXPECTED_BITS_PER_VALUE);
    require_true(coarse_bits == __EXPECTED_COARSE_BITS__u);
    require_true(fine_bits == __EXPECTED_FINE_BITS__u);
    require_true(value_count == EXPECTED_NUM_MACROS);
    require_true(tile_group == EXPECTED_TILE_GROUP);
    for (uint32_t i = 0u; i < EXPECTED_NUM_MACROS; i++) {
        require_true(payload_value(flit2, flit3, i, EXPECTED_BITS_PER_VALUE) == expected_payload_values[i]);
    }
    (void)EXPECTED_DST;
}

static void run_descriptor_queue(void)
{
    for (uint32_t i = 0u; i < DESCRIPTOR_COUNT; i++) {
        uint32_t op = descriptor_op_ids[i];
        require_true(descriptor_checksum(descriptor_queue[i]) == descriptor_queue[i][14]);
        if (op == 3u) {
            write_packed_window();
        }
        load_descriptor(i);
        uint32_t result = run_loaded_descriptor(op);
        if (op == 4u) {
            require_true((result & 0xFFFu) == EXPECTED_TILE_GROUP);
            drain_and_check_noc_result();
        }
    }
}

void main()
{
    ManagmentGpio_outputEnable();
    ManagmentGpio_write(0);
    enableHkSpi(0);
    User_enableIF();

    require_true((wait_init() & STATUS_ERROR) == 0u);
    require_true(USER_readWord(CSR_VERSION) == 0x58314E43u);
    require_true(((USER_readWord(CSR_CONFIG) >> 16) & 0xFFu) == EXPECTED_NUM_MACROS);

    run_descriptor_queue();

    ManagmentGpio_write(1);
    while (1) { }
}
'''


def _c_word(value: int) -> str:
    return f"0x{int(value) & 0xFFFF_FFFF:08X}u"


def _array_initializer(values: list[int], indent: str = "    ") -> str:
    return indent + ", ".join(_c_word(value) for value in values) + ","


def _descriptor_queue_initializer(queue: list[list[int]]) -> str:
    rows = []
    for words in queue:
        rows.append("    {\n" + _array_initializer(words, "        ") + "\n    },")
    return "\n".join(rows)


def _role_bits(role_id: int) -> tuple[int, int]:
    if role_id == 1:
        return 4, 3
    if role_id == 8:
        return 5, 5
    return 5, 3


def _role_bits_per_value(role_id: int) -> int:
    if int(role_id) == 1:
        return 8
    if int(role_id) == 8:
        return 11
    return 9


def _make_firmware_vector(plan: StreamPlan) -> dict[str, object]:
    execute = next((descriptor for descriptor in plan.descriptors if descriptor.op == "execute_tile"), None)
    if execute is None:
        raise ValueError("Stream plan has no execute_tile descriptor.")

    program = next(
        (
            descriptor
            for descriptor in plan.descriptors
            if descriptor.op == "program_x1_window"
            and descriptor.layer_id == execute.layer_id
            and descriptor.window_id == execute.window_id
        ),
        None,
    )
    if program is None:
        raise ValueError("Stream plan has no program_x1_window descriptor for the execute window.")

    window_entries = [
        ((0xFF & 0xFF) << 24)
        | ((macro & 0xFF) << 16)
        | ((int(execute.col_block) & 0x1F) << 8)
        | (int(execute.row_block) & 0x1F)
        for macro in (0, 1)
    ]
    program_words = replace(program, byte_count=len(window_entries) * 4).pack_words()
    execute_words = execute.pack_words()
    if not verify_descriptor_words(program_words):
        raise ValueError("Internal error: generated program descriptor checksum failed.")
    if not verify_descriptor_words(execute_words):
        raise ValueError("Internal error: generated execute descriptor checksum failed.")

    return {
        "execution_mode": plan.execution_mode,
        "program_descriptor_words": program_words,
        "execute_descriptor_words": execute_words,
        "role": execute.role,
        "role_id": execute.role_id(),
        "bits_per_value": _role_bits_per_value(execute.role_id()),
        "gain_shift": 0,
        "vc": execute.vc,
        "dst": execute.route_dst,
        "tile_group": execute.tile_group,
        "num_macros": int(plan.fabric.get("x1_macros_per_endpoint", 4)),
        "program_row": int(execute.row_block) & 0x1F,
        "program_col": int(execute.col_block) & 0x1F,
        "packed_window_words": window_entries,
        "expected_payload_values": [4, 4, 0, 0],
        "program_descriptor": replace(program, byte_count=len(window_entries) * 4).as_dict(),
        "descriptor": execute.as_dict(),
    }


def make_firmware_driver(plan: StreamPlan, test_name: str = "x1_streaming_descriptor_firmware") -> dict[str, object]:
    vector = _make_firmware_vector(plan)
    queue = [vector["program_descriptor_words"], vector["execute_descriptor_words"]]
    op_ids = [DescriptorOp.PROGRAM_X1_WINDOW, DescriptorOp.EXECUTE_TILE]
    coarse_bits, fine_bits = _role_bits(int(vector["role_id"]))
    return {
        "test_name": test_name,
        "descriptor_queue": queue,
        "descriptor_op_ids": [int(op) for op in op_ids],
        "packed_window_words": vector["packed_window_words"],
        "expected_payload_values": vector["expected_payload_values"],
        "expected": {
            "num_macros": int(vector["num_macros"]),
            "role_id": int(vector["role_id"]),
            "bits_per_value": int(vector["bits_per_value"]),
            "coarse_bits": int(coarse_bits),
            "fine_bits": int(fine_bits),
            "tile_group": int(vector["tile_group"]),
            "dst": int(vector["dst"]),
        },
        "vector": vector,
    }


def render_firmware_c(driver: dict[str, object]) -> str:
    expected = dict(driver["expected"])
    queue = [list(map(int, item)) for item in driver["descriptor_queue"]]
    op_ids = list(map(int, driver["descriptor_op_ids"]))
    window_words = list(map(int, driver["packed_window_words"]))
    payload_values = list(map(int, driver["expected_payload_values"]))
    source = FIRMWARE_C_TEMPLATE
    replacements = {
        "__DESCRIPTOR_COUNT__": str(len(queue)),
        "__WINDOW_WORD_COUNT__": str(len(window_words)),
        "__EXPECTED_NUM_MACROS__": str(expected["num_macros"]),
        "__EXPECTED_ROLE_ID__": str(expected["role_id"]),
        "__EXPECTED_BITS_PER_VALUE__": str(expected["bits_per_value"]),
        "__EXPECTED_TILE_GROUP__": str(expected["tile_group"]),
        "__EXPECTED_DST__": str(expected["dst"]),
        "__EXPECTED_COARSE_BITS__": str(expected["coarse_bits"]),
        "__EXPECTED_FINE_BITS__": str(expected["fine_bits"]),
        "__DESCRIPTOR_QUEUE__": _descriptor_queue_initializer(queue),
        "__DESCRIPTOR_OP_IDS__": _array_initializer(op_ids),
        "__WINDOW_WORDS__": _array_initializer(window_words),
        "__EXPECTED_PAYLOAD_VALUES__": _array_initializer(payload_values),
    }
    for key, value in replacements.items():
        source = source.replace(key, value)
    return source


def write_firmware_driver(
    plan: StreamPlan,
    out_c: str | Path,
    out_json: str | Path | None = None,
    out_yaml: str | Path | None = None,
    *,
    test_name: str = "x1_streaming_descriptor_firmware",
) -> dict[str, object]:
    driver = make_firmware_driver(plan, test_name=test_name)
    c_path = Path(out_c)
    c_path.parent.mkdir(parents=True, exist_ok=True)
    c_path.write_text(render_firmware_c(driver), encoding="utf-8")
    if out_json is not None:
        json_path = Path(out_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(driver, indent=2) + "\n", encoding="utf-8")
    if out_yaml is not None:
        yaml_path = Path(out_yaml)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_path.write_text(FIRMWARE_YAML_TEMPLATE.replace("__TEST_NAME__", test_name), encoding="utf-8")
    return driver
