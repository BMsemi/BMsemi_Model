set script_dir [file dirname [file normalize [info script]]]
set root_dir [file dirname $script_dir]
set workspace_dir [file dirname $root_dir]
set build_dir [file join $root_dir build synth_axis_x2]
set reports_dir [file join $build_dir reports]

file mkdir $build_dir
file mkdir $reports_dir

set part_name xczu7ev-ffvc1156-2-e
set board_part xilinx.com:zcu106:part0:2.6
set top_name AXIS_X2_SIMD_Wrapper

set ref_hdl_dir [file join $workspace_dir reference_code happy_family-main hardware hdl AXIS_X2_SIMD srcs]
set sv_files [list \
    [file join $ref_hdl_dir PipelineReg.sv] \
    [file join $ref_hdl_dir BRAM.sv] \
    [file join $ref_hdl_dir AdderTree_32x8bit.sv] \
    [file join $ref_hdl_dir X2_Emu.sv] \
    [file join $ref_hdl_dir X2_Writer.sv] \
    [file join $ref_hdl_dir AXIS_X2_Controller.sv] \
    [file join $ref_hdl_dir AXIS_X2_SIMD.sv] \
    [file join $ref_hdl_dir AXIS_X2_SIMD_Wrapper.v] \
]

foreach f $sv_files {
    if {![file exists $f]} {
        puts "ERROR: Missing HDL source: $f"
        exit 2
    }
}

create_project -force axis_x2_synth $build_dir -part $part_name
catch {set_property board_part $board_part [current_project]}

read_verilog -sv $sv_files
update_compile_order -fileset sources_1

synth_design -top $top_name -part $part_name

report_utilization -file [file join $reports_dir utilization_synth.rpt]
report_timing_summary -file [file join $reports_dir timing_summary_synth.rpt]
write_checkpoint -force [file join $build_dir axis_x2_synth.dcp]

puts "SYNTH_DCP: [file normalize [file join $build_dir axis_x2_synth.dcp]]"
puts "UTIL_REPORT: [file normalize [file join $reports_dir utilization_synth.rpt]]"
puts "TIMING_REPORT: [file normalize [file join $reports_dir timing_summary_synth.rpt]]"
exit
