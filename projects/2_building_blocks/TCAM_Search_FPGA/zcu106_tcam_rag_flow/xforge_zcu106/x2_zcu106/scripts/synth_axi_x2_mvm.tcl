set script_dir [file dirname [file normalize [info script]]]
set root_dir [file dirname $script_dir]

if {[info exists ::env(AXI_X2_MVM_SYNTH_BUILD_DIR)] && $::env(AXI_X2_MVM_SYNTH_BUILD_DIR) ne ""} {
    set build_dir [file normalize $::env(AXI_X2_MVM_SYNTH_BUILD_DIR)]
} else {
    set build_dir [file normalize "C:/fpga_build/yolo26n_axi_x2_mvm_synth"]
}
set reports_dir [file join $build_dir reports]
set src_dir [file join $build_dir src]
file mkdir $build_dir
file mkdir $reports_dir
file mkdir $src_dir

set part_name xczu7ev-ffvc1156-2-e
set top_name axi_x2_mvm_accel
set rtl_file [file join $root_dir rtl axi_x2_mvm_accel.sv]
set rtl_build_file [file join $src_dir axi_x2_mvm_accel.sv]

if {![file exists $rtl_file]} {
    puts "ERROR: Missing RTL source: $rtl_file"
    exit 2
}

create_project -force ${top_name}_synth $build_dir -part $part_name
set_property target_language Verilog [current_project]

file copy -force $rtl_file $rtl_build_file
add_files -norecurse [list $rtl_build_file]
set_property file_type SystemVerilog [get_files $rtl_build_file]
update_compile_order -fileset sources_1

synth_design -top $top_name -part $part_name -mode out_of_context
create_clock -period 10.000 -name aclk [get_ports aclk]

report_utilization -file [file join $reports_dir utilization_synth.rpt]
report_timing_summary -file [file join $reports_dir timing_summary_synth.rpt]
write_checkpoint -force [file join $build_dir ${top_name}_synth.dcp]

puts "SYNTH_DCP: [file normalize [file join $build_dir ${top_name}_synth.dcp]]"
puts "UTIL_REPORT: [file normalize [file join $reports_dir utilization_synth.rpt]]"
puts "TIMING_REPORT: [file normalize [file join $reports_dir timing_summary_synth.rpt]]"
exit
