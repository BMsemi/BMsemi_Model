# Set Zynq UltraScale+ PL_CLK0 to approximately 50 MHz from the 1499.85 MHz IOPLL.
set pl0_ref_ctrl_addr 0xFF5E00C0

connect
set apu_targets [targets -filter {name =~ "*APU*"}]
if {[llength $apu_targets] == 0} {
    error "No APU debug target found."
}
targets -set [lindex $apu_targets 0]
configparams force-mem-access 1

set current [mrd -value $pl0_ref_ctrl_addr]
set stopped [expr {$current & ~(1 << 24)}]
set divided [expr {($stopped & ~(0x3F << 8)) | (30 << 8)}]
mwr $pl0_ref_ctrl_addr $stopped
mwr $pl0_ref_ctrl_addr $divided
mwr $pl0_ref_ctrl_addr [expr {$divided | (1 << 24)}]

set updated [mrd -value $pl0_ref_ctrl_addr]
set divisor0 [expr {($updated >> 8) & 0x3F}]
set divisor1 [expr {($updated >> 16) & 0x3F}]
if {$divisor0 != 30 || $divisor1 != 1 || !(($updated >> 24) & 1)} {
    error [format "PL_CLK0 update failed: 0x%08X" $updated]
}
puts [format "PASS: PL0_REF_CTRL 0x%08X -> 0x%08X (DIV0=%d, DIV1=%d)" \
    $current $updated $divisor0 $divisor1]
disconnect
