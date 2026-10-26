set script_dir [file dirname [file normalize [info script]]]
set root_dir [file dirname $script_dir]
set workspace_dir [file dirname $root_dir]
# Keep the implementation build out of OneDrive and under a short path.
# Vivado IP generation can lock or exceed path limits under synced folders.
set build_dir [file normalize "C:/fpga_build/yolo26n_x2_dma_bd"]
set reports_dir [file join $build_dir reports]

file mkdir $build_dir
file mkdir $reports_dir

set part_name xczu7ev-ffvc1156-2-e
set board_part xilinx.com:zcu106:part0:2.6
set design_name zcu106_x2_dma

set ref_hdl_dir [file join $workspace_dir reference_code happy_family-main hardware hdl AXIS_X2_SIMD srcs]
set rtl_files [list \
    [file join $ref_hdl_dir PipelineReg.sv] \
    [file join $ref_hdl_dir BRAM.sv] \
    [file join $ref_hdl_dir AdderTree_32x8bit.sv] \
    [file join $ref_hdl_dir X2_Emu.sv] \
    [file join $ref_hdl_dir X2_Writer.sv] \
    [file join $ref_hdl_dir AXIS_X2_Controller.sv] \
    [file join $ref_hdl_dir AXIS_X2_SIMD.sv] \
    [file join $ref_hdl_dir AXIS_X2_SIMD_Wrapper.v] \
]

foreach f $rtl_files {
    if {![file exists $f]} {
        puts "ERROR: Missing HDL source: $f"
        exit 2
    }
}

proc set_config_if_present {cell key value} {
    set prop "CONFIG.$key"
    if {[lsearch -exact [list_property $cell] $prop] >= 0} {
        set_property $prop $value $cell
    } else {
        puts "INFO: Property not present on $cell: $prop"
    }
}

create_project -force $design_name $build_dir -part $part_name
set_property target_language Verilog [current_project]
catch {set_property board_part $board_part [current_project]}

add_files -norecurse $rtl_files
foreach f $rtl_files {
    if {[file extension $f] eq ".sv"} {
        set file_obj [get_files -quiet -all [list $f]]
        if {[llength $file_obj] > 0} {
            set_property file_type SystemVerilog $file_obj
        } else {
            puts "WARNING: Could not find project file object for: $f"
        }
    }
}
update_compile_order -fileset sources_1

create_bd_design $design_name

# ZCU106 Processing System.
create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:* psu
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e \
    -config {apply_board_preset "1"} [get_bd_cells psu]

# Enable one high-performance PS slave port for DMA access to DDR.
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP1 {0} \
    CONFIG.PSU__USE__S_AXI_GP0 {1} \
    CONFIG.PSU__SAXIGP0__DATA_WIDTH {64} \
] [get_bd_cells psu]

# Reset bridge for PL logic.
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:* rst_pl

# AXI DMA in simple/direct-register mode.
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:* axi_dma_x2_simd
set dma [get_bd_cells axi_dma_x2_simd]
set_config_if_present $dma c_include_sg 0
set_config_if_present $dma c_include_mm2s 1
set_config_if_present $dma c_include_s2mm 1
set_config_if_present $dma c_m_axis_mm2s_tdata_width 32
set_config_if_present $dma c_s_axis_s2mm_tdata_width 32
set_config_if_present $dma c_m_axi_mm2s_data_width 64
set_config_if_present $dma c_m_axi_s2mm_data_width 64
set_config_if_present $dma c_sg_include_stscntrl_strm 0

# Import the reference X2 AXI-Stream core as a BD module.
create_bd_cell -type module -reference AXIS_X2_SIMD_Wrapper AXIS_X2_SIMD

# SmartConnect: PS master controls DMA registers.
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:* axi_smc_lite
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] [get_bd_cells axi_smc_lite]

# SmartConnect: DMA masters read/write PS DDR through HP0.
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:* axi_smc_mem
set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {1}] [get_bd_cells axi_smc_mem]

# Clocking: use PS PL clock for the accelerator shell.
set pl_clk [get_bd_pins psu/pl_clk0]
set pl_rstn [get_bd_pins rst_pl/peripheral_aresetn]

connect_bd_net $pl_clk [get_bd_pins psu/maxihpm0_fpd_aclk]
connect_bd_net $pl_clk [get_bd_pins psu/saxihpc0_fpd_aclk]
connect_bd_net $pl_clk [get_bd_pins rst_pl/slowest_sync_clk]
connect_bd_net $pl_clk [get_bd_pins axi_dma_x2_simd/s_axi_lite_aclk]
connect_bd_net $pl_clk [get_bd_pins axi_dma_x2_simd/m_axi_mm2s_aclk]
connect_bd_net $pl_clk [get_bd_pins axi_dma_x2_simd/m_axi_s2mm_aclk]
connect_bd_net $pl_clk [get_bd_pins AXIS_X2_SIMD/aclk]
connect_bd_net $pl_clk [get_bd_pins axi_smc_lite/aclk]
connect_bd_net $pl_clk [get_bd_pins axi_smc_mem/aclk]

connect_bd_net [get_bd_pins psu/pl_resetn0] [get_bd_pins rst_pl/ext_reset_in]
connect_bd_net $pl_rstn [get_bd_pins AXIS_X2_SIMD/aresetn]
connect_bd_net $pl_rstn [get_bd_pins axi_dma_x2_simd/axi_resetn]
connect_bd_net $pl_rstn [get_bd_pins axi_smc_lite/aresetn]
connect_bd_net $pl_rstn [get_bd_pins axi_smc_mem/aresetn]

# AXI-Lite control path.
connect_bd_intf_net [get_bd_intf_pins psu/M_AXI_HPM0_FPD] [get_bd_intf_pins axi_smc_lite/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_smc_lite/M00_AXI] [get_bd_intf_pins axi_dma_x2_simd/S_AXI_LITE]

# AXI memory path to PS DDR.
connect_bd_intf_net [get_bd_intf_pins axi_dma_x2_simd/M_AXI_MM2S] [get_bd_intf_pins axi_smc_mem/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_dma_x2_simd/M_AXI_S2MM] [get_bd_intf_pins axi_smc_mem/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_smc_mem/M00_AXI] [get_bd_intf_pins psu/S_AXI_HPC0_FPD]

# AXI-Stream datapath: DMA -> X2 -> DMA.
connect_bd_intf_net [get_bd_intf_pins axi_dma_x2_simd/M_AXIS_MM2S] [get_bd_intf_pins AXIS_X2_SIMD/s_axis]
connect_bd_intf_net [get_bd_intf_pins AXIS_X2_SIMD/m_axis] [get_bd_intf_pins axi_dma_x2_simd/S_AXIS_S2MM]

assign_bd_address
validate_bd_design
save_bd_design

set bd_path [file join $build_dir $design_name.srcs sources_1 bd $design_name $design_name.bd]
set bd_file [get_files -quiet -all [list $bd_path]]
if {[llength $bd_file] == 0} {
    set bd_file [get_files -quiet -all *.bd]
}
if {[llength $bd_file] == 0} {
    puts "ERROR: Could not find generated block design file: $bd_path"
    exit 3
}
generate_target all $bd_file
set wrapper [make_wrapper -files $bd_file -top]
set wrapper_file $wrapper
add_files -norecurse [list $wrapper_file]
set wrapper_top [file rootname [file tail $wrapper_file]]
update_compile_order -fileset sources_1
set_property top $wrapper_top [get_filesets sources_1]
set_property top_file $wrapper_file [get_filesets sources_1]
update_compile_order -fileset sources_1

launch_runs synth_1 -jobs 2
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
    puts "ERROR: synth_1 did not complete."
    exit 3
}

launch_runs impl_1 -to_step write_bitstream -jobs 2
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {
    puts "ERROR: impl_1 did not complete."
    exit 4
}

open_run impl_1
report_utilization -file [file join $reports_dir utilization_impl.rpt]
report_timing_summary -file [file join $reports_dir timing_summary_impl.rpt]

set bit_file [file normalize [lindex [glob -nocomplain [file join $build_dir $design_name.runs impl_1 *.bit]] 0]]
set xsa_file [file normalize [file join $build_dir ${design_name}.xsa]]
write_hw_platform -force -fixed -include_bit -file $xsa_file

puts "BITSTREAM: $bit_file"
puts "XSA: $xsa_file"
puts "UTIL_REPORT: [file normalize [file join $reports_dir utilization_impl.rpt]]"
puts "TIMING_REPORT: [file normalize [file join $reports_dir timing_summary_impl.rpt]]"
exit
