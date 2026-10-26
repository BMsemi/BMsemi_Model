# Fast physical validation ECO for the v0x00020021 routed checkpoint.
#
# The source-of-truth RTL is fixed and versioned as v0x00020022. This script
# installs explicit constants on the X2 SmartConnect input ARSIZE[2:1] pins so the
# already-routed v0x00020021 image can validate the 16-byte read hypothesis
# without waiting for a full clean synthesis.

if {$argc != 3} {
    puts "Usage: vivado -mode batch -source eco_fix_reader_arsize.tcl -tclargs <routed.dcp> <eco.dcp> <eco.bit>"
    exit 2
}

set input_dcp [file normalize [lindex $argv 0]]
set output_dcp [file normalize [lindex $argv 1]]
set output_bit [file normalize [lindex $argv 2]]

open_checkpoint $input_dcp

set pin1 [get_pins -quiet {zcu106_axi_x2_mvm_i/axi_smc_mem/inst/S01_AXI_arsize[1]}]
set pin2 [get_pins -quiet {zcu106_axi_x2_mvm_i/axi_smc_mem/inst/S01_AXI_arsize[2]}]
if {[llength $pin1] != 1 || [llength $pin2] != 1} {
    puts "ERROR: Could not resolve SmartConnect S01 ARSIZE pins: bit1=$pin1 bit2=$pin2"
    exit 3
}

set net1 [get_nets -quiet -of_objects $pin1]
set net2 [get_nets -quiet -of_objects $pin2]
if {[llength $net1] != 1 || [llength $net2] > 1} {
    puts "ERROR: Unexpected ARSIZE constant nets: bit1=$net1 bit2=$net2"
    exit 4
}

puts "ECO_BEFORE: ARSIZE[1]=$net1 ARSIZE[2]=$net2"
disconnect_net -net $net1 -objects $pin1
if {[llength $net2] == 1} {
    disconnect_net -net $net2 -objects $pin2
}

# The original constants were optimized at the hierarchy boundary, so moving
# their surviving scalar net leaves it driverless. Explicit primitive sources
# make the ECO electrically complete: 3'b100 selects a 16-byte AXI beat.
create_cell -reference GND eco_reader_arsize_gnd
create_cell -reference VCC eco_reader_arsize_vcc
create_net eco_reader_arsize_zero
create_net eco_reader_arsize_one
connect_net -hierarchical -net eco_reader_arsize_zero -objects \
    [list [get_pins eco_reader_arsize_gnd/G] $pin1]
connect_net -hierarchical -net eco_reader_arsize_one -objects \
    [list [get_pins eco_reader_arsize_vcc/P] $pin2]
puts "ECO_AFTER: ARSIZE[1]=[get_nets -of_objects $pin1] ARSIZE[2]=[get_nets -of_objects $pin2]"

route_design -preserve
report_timing_summary -file "${output_bit}.timing.rpt"
write_checkpoint -force $output_dcp
write_bitstream -force $output_bit
puts "ECO_READER_ARSIZE_OK: $output_bit"
exit
