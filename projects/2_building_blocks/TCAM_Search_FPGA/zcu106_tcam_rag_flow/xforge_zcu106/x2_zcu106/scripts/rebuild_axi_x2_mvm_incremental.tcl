# Rebuild the corrected AXI X2 MVM design from an existing Vivado project.
#
# Usage:
#   vivado -mode batch -source rebuild_axi_x2_mvm_incremental.tcl -tclargs \
#       <project.xpr> <corrected-rtl.sv> <output-dir>

if {$argc != 3} {
    puts "Usage: rebuild_axi_x2_mvm_incremental.tcl <project.xpr> <corrected-rtl.sv> <output-dir>"
    exit 2
}

set project_xpr [file normalize [lindex $argv 0]]
set source_rtl [file normalize [lindex $argv 1]]
set output_dir [file normalize [lindex $argv 2]]
set project_dir [file dirname $project_xpr]
set project_rtl [file join $project_dir src axi_x2_mvm_accel.sv]

if {![file exists $project_xpr]} {
    puts "ERROR: Missing project: $project_xpr"
    exit 3
}
if {![file exists $source_rtl]} {
    puts "ERROR: Missing corrected RTL: $source_rtl"
    exit 4
}

file mkdir $output_dir
file copy -force $source_rtl $project_rtl
puts "Updated project RTL: $project_rtl"

open_project $project_xpr

# The accelerator is packaged as a block-design module reference. Vivado can
# otherwise reuse an OOC checkpoint whose cache key predates an edited RTL
# source, leaving a clean-looking bitstream with the old hardware behavior.
config_ip_cache -disable_cache
set_property IP_CACHE_PERMISSIONS disabled [current_project]
set accelerator_ips [get_ips -quiet {zcu106_axi_x2_mvm_axi_x2_mvm_0 zcu106_axi_x2_mvm_axi_x2_mvm_1_0}]
foreach accelerator_ip $accelerator_ips {
    config_ip_cache -disable_for_ip $accelerator_ip
}
puts "IP_CACHE_PERMISSIONS=[get_property IP_CACHE_PERMISSIONS [current_project]]"

update_compile_order -fileset sources_1

set accelerator_runs [list \
    zcu106_axi_x2_mvm_axi_x2_mvm_0_synth_1 \
    zcu106_axi_x2_mvm_axi_x2_mvm_1_0_synth_1 \
]
foreach required_run [concat $accelerator_runs [list synth_1 impl_1]] {
    if {[llength [get_runs -quiet $required_run]] != 1} {
        puts "ERROR: Missing required Vivado run: $required_run"
        exit 5
    }
}

set jobs 4
if {[info exists ::env(AXI_X2_MVM_JOBS)] && $::env(AXI_X2_MVM_JOBS) ne ""} {
    set jobs $::env(AXI_X2_MVM_JOBS)
}

proc require_success {run_name} {
    set run [get_runs $run_name]
    set status [get_property STATUS $run]
    set progress [get_property PROGRESS $run]
    puts "RUN_RESULT: $run_name status={$status} progress={$progress}"
    if {$progress ne "100%" || [string match -nocase "*error*" $status]} {
        puts "ERROR: Vivado run did not complete successfully: $run_name"
        exit 6
    }
}

# Only the custom accelerator OOC modules and their consumers need rebuilding.
foreach accelerator_run $accelerator_runs {
    reset_run $accelerator_run
}
reset_run synth_1
reset_run impl_1

foreach accelerator_run $accelerator_runs {
    launch_runs $accelerator_run -jobs $jobs
}
foreach accelerator_run $accelerator_runs {
    wait_on_run $accelerator_run
    require_success $accelerator_run
}

launch_runs synth_1 -jobs $jobs
wait_on_run synth_1
require_success synth_1

launch_runs impl_1 -to_step write_bitstream -jobs $jobs
wait_on_run impl_1
require_success impl_1

open_run impl_1
report_timing_summary -file [file join $output_dir timing_summary.rpt]
report_utilization -file [file join $output_dir utilization.rpt]
report_route_status -file [file join $output_dir route_status.rpt]
report_drc -file [file join $output_dir drc.rpt]

set generated_bits [glob -nocomplain -directory [get_property DIRECTORY [get_runs impl_1]] *.bit]
if {[llength $generated_bits] != 1} {
    puts "ERROR: Expected one generated bitstream, found: $generated_bits"
    exit 7
}
set output_bit [file join $output_dir zcu106_axi_x2_mvm_v22.bit]
file copy -force [lindex $generated_bits 0] $output_bit
puts "CLEAN_REBUILD_OK: $output_bit"
exit
