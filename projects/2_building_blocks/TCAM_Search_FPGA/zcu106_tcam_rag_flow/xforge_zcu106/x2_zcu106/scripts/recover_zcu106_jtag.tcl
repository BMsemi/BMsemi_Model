# Recover a ZCU106 whose DAP is wedged by an outstanding PL AXI transaction.
# Programming the PL target does not require the DAP; the subsequent PS-TAP
# system reset allows XSCT to enumerate the PSU/APU targets again.

if {$argc != 1} {
    puts "Usage: xsct recover_zcu106_jtag.tcl <bitstream.bit>"
    exit 2
}

set bit_path [file normalize [lindex $argv 0]]
if {![file exists $bit_path]} {
    puts "RECOVER_FAIL: Missing bitstream: $bit_path"
    exit 3
}

connect
after 1000
targets -set -filter {name =~ "PL"}
puts "RECOVER: programming PL through independent JTAG target"
fpga -file $bit_path
after 1000

# A system reset intentionally preserves debug state, so clear the Arm DAP's
# sticky transaction fault explicitly before resetting the processing system.
set dap_selected 0
set dap_reset_ok 0
if {![catch {targets -set -filter {name =~ "DAP"}} dap_target_message]} {
    set dap_selected 1
} else {
    puts "RECOVER: DAP filter selection failed: $dap_target_message"
    # On the degraded target tree used by XSCT 2022.2 the errored DAP is still
    # addressable by its displayed target id even though name filtering omits it.
    if {![catch {targets -set 4} dap_id_message]} {
        set dap_selected 1
    } else {
        puts "RECOVER: DAP id selection failed: $dap_id_message"
    }
}
if {$dap_selected} {
    if {[catch {rst -dap} dap_reset_message]} {
        puts "RECOVER: DAP reset was unavailable: $dap_reset_message"
    } else {
        puts "RECOVER: Arm DAP reset command completed: $dap_reset_message"
        set dap_reset_ok 1
    }
    after 1000
}

targets -set -filter {name =~ "PS TAP"}
puts "RECOVER: resetting processing system through PS TAP"
if {[catch {rst -system} reset_message]} {
    puts "RECOVER: PS TAP reset was unavailable: $reset_message"
} else {
    puts "RECOVER: PS TAP reset command completed: $reset_message"
}
after 2000

# Some XSCT versions expose system reset only through the PMU processor when
# the ARM DAP is in a sticky-error state.
if {![catch {targets -set -filter {name =~ "PMU"}} pmu_target_message]} {
    if {[catch {rst -system} pmu_reset_message]} {
        puts "RECOVER: PMU system reset was unavailable: $pmu_reset_message"
    } else {
        puts "RECOVER: PMU system reset command completed: $pmu_reset_message"
    }
    after 2000

    if {!$dap_reset_ok} {
        if {[catch {rst -por} por_reset_message]} {
            puts "RECOVER: JTAG power-on reset was unavailable: $por_reset_message"
        } else {
            puts "RECOVER: JTAG power-on reset command completed: $por_reset_message"
            after 5000
            targets -set -filter {name =~ "PL"}
            puts "RECOVER: reprogramming PL after power-on reset"
            fpga -file $bit_path
            after 2000
        }
    }
}

disconnect
after 1000
connect
after 2000

set final_dap_ok 0
if {![catch {targets -set -filter {name =~ "DAP"}} final_dap_select]} {
    if {![catch {rst -dap} final_dap_reset]} {
        set final_dap_ok 1
    }
} elseif {![catch {targets -set 4} final_dap_id_select]} {
    if {![catch {rst -dap} final_dap_reset]} {
        set final_dap_ok 1
    }
}

puts "RECOVER_TARGETS:"
targets
disconnect
if {!$final_dap_ok} {
    puts "RECOVER_FAIL: DAP remains unavailable after PMU/POR recovery"
    exit 4
}
puts "RECOVER_OK: DAP reset completed after PMU/POR recovery"
exit 0
