from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum

from .noc import role_policy


class Csr(IntEnum):
    VERSION = 0
    CONFIG = 1
    STATUS = 2
    COMMAND = 3
    ADDR = 4
    DATA = 5
    ROLE = 6
    RESULT = 7
    FLIT0 = 8
    FLIT1 = 9
    FLIT2 = 10
    FLIT3 = 11
    CLEAR = 12
    DESC_INDEX = 13
    DESC_DATA = 14
    DESC_CTRL = 15
    DESC_STATUS = 16
    DESC_RESULT = 17
    DESC_FETCH_ADDR = 18
    WINDOW_INDEX = 19
    WINDOW_DATA = 20


class StatusBit(IntEnum):
    BUSY = 0x01
    INIT_DONE = 0x02
    RESULT_VALID = 0x04
    NOC_VALID = 0x08
    ERROR = 0x10


class DescStatusBit(IntEnum):
    BUSY = 0x01
    DONE = 0x02
    ERROR = 0x04
    CHECKSUM_OK = 0x08


class DescCtrlBit(IntEnum):
    START_CSR_LOADED = 0x01
    CLEAR = 0x02
    START_FETCHED = 0x04


class Command(IntEnum):
    NOP = 0
    PROGRAM = 1
    RESET = 2
    READ = 3
    COMPUTE = 4


CMD_START = 0x80000000
CMD_FULL_ROW = 1 << 9


@dataclass(frozen=True)
class CsrOp:
    kind: str
    csr: str
    offset: int
    value: int | None = None
    mask: int | None = None
    description: str = ""

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def addr_value(row: int, col: int) -> int:
    return ((int(col) & 0x1F) << 8) | (int(row) & 0x1F)


def role_value(role: str | int, gain_shift: int, vc: int, dst: int, tile_group: int) -> int:
    policy = role_policy(role)
    return (
        ((int(tile_group) & 0xFFF) << 16)
        | ((int(dst) & 0xF) << 12)
        | ((int(vc) & 0x7) << 9)
        | ((int(gain_shift) & 0x1F) << 4)
        | (policy.role_id & 0xF)
    )


def command_value(command: Command | int, macro: int = 0, *, full_row: bool = False) -> int:
    value = CMD_START | (int(command) & 0xF) | ((int(macro) & 0xF) << 4)
    if full_row:
        value |= CMD_FULL_ROW
    return value


def csr_write(csr: Csr, value: int, description: str = "") -> CsrOp:
    return CsrOp("write", csr.name, int(csr), int(value), description=description)


def csr_read(csr: Csr, description: str = "") -> CsrOp:
    return CsrOp("read", csr.name, int(csr), description=description)


def poll_status(mask: int, description: str = "") -> CsrOp:
    return CsrOp("poll", Csr.STATUS.name, int(Csr.STATUS), mask=int(mask), description=description)


def wait_init() -> list[CsrOp]:
    return [poll_status(StatusBit.INIT_DONE, "wait for reset-time X1 macro configuration")]


def clear(result: bool = True, noc: bool = True, error: bool = True) -> CsrOp:
    value = (1 if result else 0) | (2 if noc else 0) | (4 if error else 0)
    return csr_write(Csr.CLEAR, value, "clear result/noc/error sticky bits")


def program_cell(macro: int, row: int, col: int, data: int) -> list[CsrOp]:
    return [
        csr_write(Csr.ADDR, addr_value(row, col), f"select X1 row={row} col={col}"),
        csr_write(Csr.DATA, int(data) & 0xFF, "load program threshold/PWM data"),
        csr_write(Csr.COMMAND, command_value(Command.PROGRAM, macro), f"program macro {macro}"),
        poll_status(StatusBit.RESULT_VALID, "wait for program completion"),
    ]


def read_cell(macro: int, row: int, col: int) -> list[CsrOp]:
    return [
        csr_write(Csr.ADDR, addr_value(row, col), f"select X1 row={row} col={col}"),
        csr_write(Csr.COMMAND, command_value(Command.READ, macro), f"read macro {macro}"),
        poll_status(StatusBit.RESULT_VALID, "wait for read result"),
        csr_read(Csr.RESULT, "read raw X1 result"),
    ]


def compute_tile(
    role: str | int,
    *,
    row: int = 0,
    col: int = 0,
    gain_shift: int = 0,
    vc: int = 1,
    dst: int = 0,
    tile_group: int = 0,
    full_row: bool = False,
) -> list[CsrOp]:
    return [
        csr_write(Csr.ADDR, addr_value(row, col), f"select compute row={row} col={col}"),
        csr_write(Csr.ROLE, role_value(role, gain_shift, vc, dst, tile_group), f"configure role {role}"),
        csr_write(Csr.COMMAND, command_value(Command.COMPUTE, 0, full_row=full_row), "compute active macro group"),
        poll_status(StatusBit.RESULT_VALID | StatusBit.NOC_VALID, "wait for NoC flit"),
        csr_read(Csr.FLIT0, "read NoC flit word 0"),
        csr_read(Csr.FLIT1, "read NoC flit word 1"),
        csr_read(Csr.FLIT2, "read NoC flit word 2"),
        csr_read(Csr.FLIT3, "read NoC flit word 3"),
    ]
