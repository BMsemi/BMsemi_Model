# Read the live X2 AXI-Lite registers and the smoke-test DDR regions without
# resetting or reprogramming the board.

set ACCEL_BASE 0xA0000000
set TX_BASE 0x10000000
set RX_BASE 0x11000000

connect
after 1000
targets -set -filter {name =~ "PSU"}

foreach {name offset} {
    CTRL 0x00 STATUS 0x04 CONFIG 0x08 EXPECTED 0x0C
    RX_WORDS 0x10 TX_WORDS 0x14 CYCLES 0x18 ERROR 0x1C
    MAGIC 0x20 VERSION 0x24
    DBG_SHAPE 0x28 DBG_STREAM0 0x2C DBG_STREAM1 0x30
    DBG_STREAM2 0x34 DBG_STREAM3 0x38
    DBG_WEIGHT0 0x3C DBG_WEIGHT1 0x40
    DBG_VECTOR0 0x44 DBG_VECTOR1 0x48
    DBG_RESULT0 0x4C DBG_RESULT1 0x50 DBG_RESULT2 0x54 DBG_RESULT3 0x58
    DBG_AXIS0 0x5C DBG_AXIS1 0x60 DBG_AXIS2 0x64 DBG_AXIS3 0x68
    DBG_AXIS_META 0x6C
    WR_ADDR_LO 0x70 WR_ADDR_HI 0x74 WR_COUNT 0x78 WR_STATUS 0x7C
    RD_ADDR_LO 0x80 RD_ADDR_HI 0x84 RD_COUNT 0x88 RD_STATUS 0x8C
    TDC_ORDER 0x90 TDC_MIN 0x94 TDC_MAX 0x98 TDC_SPREAD 0x9C
    TDC_COMPLETION 0xA0 TDC_CONFIG 0xA4
} {
    puts "$name [mrd -force [expr {$ACCEL_BASE + $offset}] 1]"
}

puts "TX0 [mrd -force $TX_BASE 4]"
puts "TX1 [mrd -force [expr {$TX_BASE + 16}] 4]"
puts "TX2 [mrd -force [expr {$TX_BASE + 32}] 4]"
puts "RX [mrd -force $RX_BASE 4]"

disconnect
exit
