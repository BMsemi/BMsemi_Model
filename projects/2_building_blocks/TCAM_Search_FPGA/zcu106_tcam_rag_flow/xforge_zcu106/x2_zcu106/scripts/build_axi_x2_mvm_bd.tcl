set script_dir [file dirname [file normalize [info script]]]
set root_dir [file dirname $script_dir]
set default_build_root "C:/fpga_build"
if {[file exists "D:/"]} {
    set default_build_root "D:/fpga_build"
}
set build_dir [file normalize [file join $default_build_root yolo26n_axi_x2_mvm_bd]]
if {[info exists ::env(AXI_X2_MVM_BUILD_DIR)] && $::env(AXI_X2_MVM_BUILD_DIR) ne ""} {
    set build_dir [file normalize $::env(AXI_X2_MVM_BUILD_DIR)]
}
set reports_dir [file join $build_dir reports]
set deploy_dir [file join $root_dir deploy]
set src_dir [file join $build_dir src]

file mkdir $build_dir
file mkdir $reports_dir
file mkdir $deploy_dir
file mkdir $src_dir

set part_name xczu7ev-ffvc1156-2-e
set board_part xilinx.com:zcu106:part0:2.6
set design_name zcu106_axi_x2_mvm
set top_name axi_x2_mvm_accel
set bd_top_name axi_x2_mvm_accel_bd
set rtl_file [file join $root_dir rtl axi_x2_mvm_accel.sv]
set wrapper_file [file join $root_dir rtl axi_x2_mvm_accel_bd.v]
set rram_core_file [file join $root_dir rtl rram_int4_programming_accuracy_lut.sv]
set rram_axi_file [file join $root_dir rtl axi_rram_int4_accuracy.sv]
set rram_wrapper_file [file join $root_dir rtl axi_rram_int4_accuracy_bd.v]
set rtl_build_file [file join $src_dir axi_x2_mvm_accel.sv]
set wrapper_build_file [file join $src_dir axi_x2_mvm_accel_bd.v]
set rram_core_build_file [file join $src_dir rram_int4_programming_accuracy_lut.sv]
set rram_axi_build_file [file join $src_dir axi_rram_int4_accuracy.sv]
set rram_wrapper_build_file [file join $src_dir axi_rram_int4_accuracy_bd.v]

set validate_only 0
if {[info exists ::env(AXI_X2_MVM_VALIDATE_ONLY)] && $::env(AXI_X2_MVM_VALIDATE_ONLY) eq "1"} {
    set validate_only 1
}
set build_jobs 1
if {[info exists ::env(AXI_X2_MVM_JOBS)] && $::env(AXI_X2_MVM_JOBS) ne ""} {
    set build_jobs $::env(AXI_X2_MVM_JOBS)
}
set ps_mem_port S_AXI_HP0_FPD
if {[info exists ::env(AXI_X2_MVM_PS_MEM_PORT)] && $::env(AXI_X2_MVM_PS_MEM_PORT) ne ""} {
    set ps_mem_port $::env(AXI_X2_MVM_PS_MEM_PORT)
}
set ps_mem_data_width 64
if {[info exists ::env(AXI_X2_MVM_PS_MEM_WIDTH)] && $::env(AXI_X2_MVM_PS_MEM_WIDTH) ne ""} {
    set ps_mem_data_width $::env(AXI_X2_MVM_PS_MEM_WIDTH)
}
if {$ps_mem_port eq "S_AXI_HP0_FPD"} {
    set ps_mem_clk_pin saxihp0_fpd_aclk
    set ps_mem_use_prop PSU__USE__S_AXI_GP2
    set ps_mem_width_prop PSU__SAXIGP2__DATA_WIDTH
} elseif {$ps_mem_port eq "S_AXI_HPC0_FPD"} {
    set ps_mem_clk_pin saxihpc0_fpd_aclk
    set ps_mem_use_prop PSU__USE__S_AXI_GP0
    set ps_mem_width_prop PSU__SAXIGP0__DATA_WIDTH
} else {
    error "Unsupported AXI_X2_MVM_PS_MEM_PORT=$ps_mem_port; use S_AXI_HP0_FPD or S_AXI_HPC0_FPD"
}

if {![file exists $rtl_file]} {
    puts "ERROR: Missing RTL source: $rtl_file"
    exit 2
}
if {![file exists $wrapper_file]} {
    puts "ERROR: Missing BD wrapper source: $wrapper_file"
    exit 2
}
if {![file exists $rram_core_file]} {
    puts "ERROR: Missing RRAM INT4 PL model source: $rram_core_file"
    exit 2
}
if {![file exists $rram_axi_file]} {
    puts "ERROR: Missing RRAM INT4 AXI wrapper source: $rram_axi_file"
    exit 2
}
if {![file exists $rram_wrapper_file]} {
    puts "ERROR: Missing RRAM INT4 BD wrapper source: $rram_wrapper_file"
    exit 2
}

create_project -force $design_name $build_dir -part $part_name
set_property target_language Verilog [current_project]
catch {set_property board_part $board_part [current_project]}

file copy -force $rtl_file $rtl_build_file
file copy -force $wrapper_file $wrapper_build_file
file copy -force $rram_core_file $rram_core_build_file
file copy -force $rram_axi_file $rram_axi_build_file
file copy -force $rram_wrapper_file $rram_wrapper_build_file
add_files -norecurse [list $rtl_build_file $wrapper_build_file $rram_core_build_file $rram_axi_build_file $rram_wrapper_build_file]
set_property file_type SystemVerilog [get_files [list $rtl_build_file $rram_core_build_file $rram_axi_build_file]]
update_compile_order -fileset sources_1

create_bd_design $design_name

# ZCU106 Processing System. The PS provides Linux/PYNQ or bare-metal control,
# DDR, AXI-Lite MMIO, and accelerator memory access.
create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:* psu
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e \
    -config {apply_board_preset "1"} [get_bd_cells psu]

set ps_props [list \
    CONFIG.PSU__USE__M_AXI_GP1 {0} \
    CONFIG.PSU__USE__S_AXI_GP0 {0} \
    CONFIG.PSU__USE__S_AXI_GP2 {0} \
]
lappend ps_props "CONFIG.$ps_mem_use_prop" {1}
lappend ps_props "CONFIG.$ps_mem_width_prop" $ps_mem_data_width
set_property -dict $ps_props [get_bd_cells psu]

create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:* rst_pl

puts "PS_MEM_PORT: $ps_mem_port"
puts "PS_MEM_WIDTH: $ps_mem_data_width"
puts "PE_ARRAY_COUNT: 2"
puts "PE0_MMIO_BASE: 0xA0000000"
puts "PE1_MMIO_BASE: 0xA0010000"
puts "RRAM_INT4_ACCURACY_MMIO_BASE: 0xA0060000"

# Two independent 32x32 PL PE arrays. Each instance has its own register bank,
# weight/vector state, TDC model, and AXI reader/writer.
create_bd_cell -type module -reference $bd_top_name axi_x2_mvm
create_bd_cell -type module -reference $bd_top_name axi_x2_mvm_1
create_bd_cell -type module -reference axi_rram_int4_accuracy_bd rram_int4_accuracy_0

# PS AXI master controls both PE register banks and the RRAM INT4 model.
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:* axi_smc_lite
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {3}] [get_bd_cells axi_smc_lite]

# Both PE AXI memory masters reach PS DDR through the selected PS slave port.
# HP0 is the default for non-coherent Linux/PYNQ DMA buffers; HPC0 can be
# re-enabled with AXI_X2_MVM_PS_MEM_PORT=S_AXI_HPC0_FPD for comparison builds.
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:* axi_smc_mem
set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {1}] [get_bd_cells axi_smc_mem]

set pl_clk [get_bd_pins psu/pl_clk0]
set pl_rstn [get_bd_pins rst_pl/peripheral_aresetn]

connect_bd_net $pl_clk [get_bd_pins psu/maxihpm0_fpd_aclk]
connect_bd_net $pl_clk [get_bd_pins psu/$ps_mem_clk_pin]
connect_bd_net $pl_clk [get_bd_pins rst_pl/slowest_sync_clk]
connect_bd_net $pl_clk [get_bd_pins axi_x2_mvm/aclk]
connect_bd_net $pl_clk [get_bd_pins axi_x2_mvm_1/aclk]
connect_bd_net $pl_clk [get_bd_pins rram_int4_accuracy_0/aclk]
connect_bd_net $pl_clk [get_bd_pins axi_smc_lite/aclk]
connect_bd_net $pl_clk [get_bd_pins axi_smc_mem/aclk]

connect_bd_net [get_bd_pins psu/pl_resetn0] [get_bd_pins rst_pl/ext_reset_in]
connect_bd_net $pl_rstn [get_bd_pins axi_x2_mvm/aresetn]
connect_bd_net $pl_rstn [get_bd_pins axi_x2_mvm_1/aresetn]
connect_bd_net $pl_rstn [get_bd_pins rram_int4_accuracy_0/aresetn]
connect_bd_net $pl_rstn [get_bd_pins axi_smc_lite/aresetn]
connect_bd_net $pl_rstn [get_bd_pins axi_smc_mem/aresetn]

# AXI-Lite MMIO path.
connect_bd_intf_net [get_bd_intf_pins psu/M_AXI_HPM0_FPD] [get_bd_intf_pins axi_smc_lite/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_smc_lite/M00_AXI] [get_bd_intf_pins axi_x2_mvm/S_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_smc_lite/M01_AXI] [get_bd_intf_pins axi_x2_mvm_1/S_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_smc_lite/M02_AXI] [get_bd_intf_pins rram_int4_accuracy_0/S_AXI]

# AXI memory path to PS DDR.
connect_bd_intf_net [get_bd_intf_pins axi_x2_mvm/M_AXI] [get_bd_intf_pins axi_smc_mem/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_x2_mvm_1/M_AXI] [get_bd_intf_pins axi_smc_mem/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_smc_mem/M00_AXI] [get_bd_intf_pins psu/$ps_mem_port]

assign_bd_address -offset 0xA0000000 -range 64K -target_address_space \
    [get_bd_addr_spaces psu/Data] [get_bd_addr_segs axi_x2_mvm/S_AXI/reg0] -force
assign_bd_address -offset 0xA0010000 -range 64K -target_address_space \
    [get_bd_addr_spaces psu/Data] [get_bd_addr_segs axi_x2_mvm_1/S_AXI/reg0] -force
assign_bd_address -offset 0xA0060000 -range 64K -target_address_space \
    [get_bd_addr_spaces psu/Data] [get_bd_addr_segs rram_int4_accuracy_0/S_AXI/reg0] -force
assign_bd_address
validate_bd_design
save_bd_design

puts "BD_VALIDATE_OK: $design_name"
if {$validate_only} {
    exit
}

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

launch_runs synth_1 -jobs $build_jobs
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
    puts "ERROR: synth_1 did not complete."
    exit 4
}

launch_runs impl_1 -to_step write_bitstream -jobs $build_jobs
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {
    puts "ERROR: impl_1 did not complete."
    exit 5
}

open_run impl_1
report_utilization -file [file join $reports_dir utilization_impl.rpt]
report_timing_summary -file [file join $reports_dir timing_summary_impl.rpt]

set bit_file [file normalize [lindex [glob -nocomplain [file join $build_dir $design_name.runs impl_1 *.bit]] 0]]
set xsa_file [file normalize [file join $build_dir ${design_name}.xsa]]
set hwh_files [glob -nocomplain [file join $build_dir $design_name.gen sources_1 bd $design_name hw_handoff *.hwh]]
set hwh_file ""
if {[llength $hwh_files] > 0} {
    set hwh_file [file normalize [lindex $hwh_files 0]]
}

write_hw_platform -force -fixed -include_bit -file $xsa_file

if {$bit_file ne ""} {
    file copy -force $bit_file [file join $deploy_dir ${design_name}.bit]
}
if {$xsa_file ne ""} {
    file copy -force $xsa_file [file join $deploy_dir ${design_name}.xsa]
}
if {$hwh_file ne ""} {
    file copy -force $hwh_file [file join $deploy_dir ${design_name}.hwh]
}

puts "BITSTREAM: $bit_file"
puts "XSA: $xsa_file"
puts "HWH: $hwh_file"
puts "DEPLOY_DIR: [file normalize $deploy_dir]"
puts "UTIL_REPORT: [file normalize [file join $reports_dir utilization_impl.rpt]]"
puts "TIMING_REPORT: [file normalize [file join $reports_dir timing_summary_impl.rpt]]"
exit
