# Build the ZCU106 FPGA design for the TCAM RAG accelerator.
#
# Run from the repository root with:
#   vivado -mode batch -source scripts/build_zcu106.tcl
#
# Outputs:
#   build/vivado/tcam_rag_zcu106/tcam_rag_zcu106.runs/impl_1/design_1_wrapper.bit
#   deploy/tcam_rag_zcu106.bit
#   deploy/tcam_rag_zcu106.xsa

set script_dir [file dirname [file normalize [info script]]]
set repo_dir   [file normalize [file join $script_dir ".."]]
set build_dir  [file normalize [file join $repo_dir "build" "vivado"]]
set proj_name  "tcam_rag_zcu106"
set proj_dir   [file join $build_dir $proj_name]
set ip_proj_dir [file join $build_dir "ip_packager_project"]
set ip_repo_dir [file join $build_dir "ip_repo" "axi_tcam_rag_accel_1_0"]
set deploy_dir [file normalize [file join $repo_dir "deploy"]]
set reports_dir [file normalize [file join $repo_dir "reports"]]

file mkdir $build_dir
file mkdir $deploy_dir
file mkdir $reports_dir

# Vivado 2022.2 does not accept a SystemVerilog top as a block-design module
# reference. Package the RTL wrapper as a local IP so its AXI metadata is
# available to IP Integrator.
create_project ip_packager $ip_proj_dir -part xczu7ev-ffvc1156-2-e -force
add_files -norecurse [list \
    [file join $repo_dir "rtl" "tcam_rag_core.sv"] \
    [file join $repo_dir "rtl" "axi_tcam_rag_accel.sv"] \
]
set_property file_type SystemVerilog [get_files *.sv]
set_property top axi_tcam_rag_accel [current_fileset]
update_compile_order -fileset sources_1
ipx::package_project -root_dir $ip_repo_dir -vendor user.org -library user \
    -taxonomy /UserIP -import_files -force
set ip_core [ipx::current_core]
set_property name axi_tcam_rag_accel $ip_core
set_property display_name {TCAM RAG AXI Accelerator} $ip_core
set_property description {AXI4-Lite TCAM retrieval accelerator} $ip_core
set_property version 1.0 $ip_core
ipx::save_core $ip_core
close_project

create_project $proj_name $proj_dir -part xczu7ev-ffvc1156-2-e -force
set_property target_language Verilog [current_project]
set_property simulator_language Mixed [current_project]
set_property ip_repo_paths [list [file dirname $ip_repo_dir]] [current_project]
update_ip_catalog

# Use the ZCU106 board preset when the board files are installed. The part
# remains sufficient for synthesis/implementation if the board files are absent.
foreach board_part {
    xilinx.com:zcu106:part0:2.6
    xilinx.com:zcu106:part0:2.5
    xilinx.com:zcu106:part0:2.4
    xilinx.com:zcu106:part0:2.3
    xilinx.com:zcu106:part0:2.2
    xilinx.com:zcu106:part0:2.1
    xilinx.com:zcu106:part0:2.0
} {
    if {[catch {set_property board_part $board_part [current_project]}] == 0} {
        puts "INFO: Using board_part $board_part"
        break
    }
}

create_bd_design "design_1"

create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:* zynq_ultra_ps_e_0

# Prefer a board preset if available, then explicitly enable the PL interfaces
# used by this design. These properties are stable across common Vivado releases.
catch {
    apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e \
        -config {apply_board_preset "1"} [get_bd_cells zynq_ultra_ps_e_0]
}
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} \
    CONFIG.PSU__USE__M_AXI_GP1 {0} \
    CONFIG.PSU__USE__IRQ0 {1} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {50} \
] [get_bd_cells zynq_ultra_ps_e_0]

create_bd_cell -type ip -vlnv user.org:user:axi_tcam_rag_accel:1.0 axi_tcam_rag_accel_0

connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] \
               [get_bd_pins axi_tcam_rag_accel_0/aclk]

# Let Vivado insert the interconnect/smartconnect between the PS AXI master and
# the accelerator's AXI4-Lite slave, then assign an address range.
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config { \
        Master {/zynq_ultra_ps_e_0/M_AXI_HPM0_FPD} \
        Slave {/axi_tcam_rag_accel_0/S_AXI} \
        Clk_master {/zynq_ultra_ps_e_0/pl_clk0} \
        Clk_slave {/zynq_ultra_ps_e_0/pl_clk0} \
        Clk_xbar {/zynq_ultra_ps_e_0/pl_clk0} \
        intc_ip {New AXI SmartConnect} \
        master_apm {0} \
    } [get_bd_intf_pins axi_tcam_rag_accel_0/S_AXI]

assign_bd_address
set addr_seg [get_bd_addr_segs -quiet zynq_ultra_ps_e_0/Data/SEG_axi_tcam_rag_accel_0_reg0]
if {[llength $addr_seg] > 0} {
    set_property range 64K $addr_seg
    set_property offset 0xA0000000 $addr_seg
}

validate_bd_design
save_bd_design

set bd_file [file join $proj_dir "$proj_name.srcs" "sources_1" "bd" "design_1" "design_1.bd"]
make_wrapper -files [get_files [list $bd_file]] -top
set wrapper_file [file join $proj_dir "$proj_name.gen" "sources_1" "bd" "design_1" "hdl" "design_1_wrapper.v"]
add_files -norecurse [list $wrapper_file]
set_property top design_1_wrapper [current_fileset]
update_compile_order -fileset sources_1

launch_runs synth_1 -jobs 4
wait_on_run synth_1
open_run synth_1
report_utilization -file [file join $reports_dir "post_synth_utilization.rpt"]
report_timing_summary -file [file join $reports_dir "post_synth_timing_summary.rpt"]

launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
open_run impl_1
report_utilization -file [file join $reports_dir "post_impl_utilization.rpt"]
report_timing_summary -file [file join $reports_dir "post_impl_timing_summary.rpt"]

set bit_src [file join $proj_dir "$proj_name.runs" "impl_1" "design_1_wrapper.bit"]
set bit_dst [file join $deploy_dir "tcam_rag_zcu106.bit"]
file copy -force $bit_src $bit_dst

write_hw_platform -fixed -include_bit -force -file [file join $deploy_dir "tcam_rag_zcu106.xsa"]

puts "INFO: Bitstream written to $bit_dst"
puts "INFO: Hardware platform written to [file join $deploy_dir tcam_rag_zcu106.xsa]"
