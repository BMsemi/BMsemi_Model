# Read the accelerator identification register through the Zynq debug interface.
connect

set apu_targets [targets -filter {name =~ "*APU*"}]
if {[llength $apu_targets] == 0} {
    error "No APU debug target found. The processing system may be held in reset."
}

targets -set [lindex $apu_targets 0]
configparams force-mem-access 1
memmap -addr 0xA0000000 -size 0x10000 -flags 3
set id_value [mrd -value 0xA0000000]
puts [format "INFO: Accelerator ID register = 0x%08X" $id_value]
set pl0_ref_ctrl [mrd -value 0xFF5E00C0]
set divisor0 [expr {($pl0_ref_ctrl >> 8) & 0x3F}]
set divisor1 [expr {($pl0_ref_ctrl >> 16) & 0x3F}]
puts [format "INFO: PL0_REF_CTRL = 0x%08X (DIV0=%d, DIV1=%d)" \
    $pl0_ref_ctrl $divisor0 $divisor1]
disconnect
