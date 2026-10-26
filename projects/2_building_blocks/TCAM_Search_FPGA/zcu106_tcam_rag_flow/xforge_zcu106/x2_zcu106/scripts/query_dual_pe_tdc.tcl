# Read the live TDC metadata registers for both AXI X2 MVM PE instances.
#
# Usage:
#   xsct.bat query_dual_pe_tdc.tcl ?result.txt?

proc fail {message code} {
    puts "JTAG_TDC_FAIL: $message"
    catch {disconnect}
    exit $code
}

proc read_u32 {address} {
    set raw [mrd -force $address 1]
    if {![regexp -nocase {:\s+([0-9a-f]{8})} $raw _ value]} {
        error "Could not parse mrd result for [format 0x%08X $address]: $raw"
    }
    return [expr {wide("0x$value") & 0xFFFFFFFF}]
}

proc read_pe_tdc {label base} {
    set REG_STATUS           0x04
    set REG_CYCLES           0x18
    set REG_ERROR            0x1C
    set REG_MAGIC            0x20
    set REG_VERSION          0x24
    set REG_TDC_ORDER        0x90
    set REG_TDC_ARRIVAL_MIN  0x94
    set REG_TDC_ARRIVAL_MAX  0x98
    set REG_TDC_SPREAD       0x9C
    set REG_TDC_COMPLETION   0xA0
    set REG_TDC_CONFIG       0xA4
    set REG_TDC_TIMING_MIN   0xA8
    set REG_TDC_TIMING_MAX   0xAC

    set magic [read_u32 [expr {$base + $REG_MAGIC}]]
    set version [read_u32 [expr {$base + $REG_VERSION}]]
    set status [read_u32 [expr {$base + $REG_STATUS}]]
    set error [read_u32 [expr {$base + $REG_ERROR}]]
    set cycles [read_u32 [expr {$base + $REG_CYCLES}]]
    set order [read_u32 [expr {$base + $REG_TDC_ORDER}]]
    set config [read_u32 [expr {$base + $REG_TDC_CONFIG}]]
    set arrival_min [read_u32 [expr {$base + $REG_TDC_ARRIVAL_MIN}]]
    set arrival_max [read_u32 [expr {$base + $REG_TDC_ARRIVAL_MAX}]]
    set spread [read_u32 [expr {$base + $REG_TDC_SPREAD}]]
    set completion [read_u32 [expr {$base + $REG_TDC_COMPLETION}]]
    set timing_min [read_u32 [expr {$base + $REG_TDC_TIMING_MIN}]]
    set timing_max [read_u32 [expr {$base + $REG_TDC_TIMING_MAX}]]

    set timing_valid [expr {($order >> 31) & 1}]
    set first_col [expr {$order & 0xFF}]
    set last_col [expr {($order >> 8) & 0xFF}]
    set tdc_enabled [expr {($config >> 31) & 1}]
    set reference_levels [expr {$config & 0xFFFF}]

    return [format \
        "%s_magic=0x%08X\n%s_version=0x%08X\n%s_status=0x%08X\n%s_error=%u\n%s_cycles=%u\n%s_tdc_config=0x%08X\n%s_tdc_enabled=%u\n%s_tdc_reference_levels=%u\n%s_tdc_timing_min_ps=%u\n%s_tdc_timing_max_ps=%u\n%s_tdc_timing_valid=%u\n%s_tdc_first_ready_column=%u\n%s_tdc_last_ready_column=%u\n%s_tdc_arrival_min_ps=%u\n%s_tdc_arrival_max_ps=%u\n%s_tdc_arrival_spread_ps=%u\n%s_tdc_pe_completion_time_ps=%u\n" \
        $label $magic $label $version $label $status $label $error $label $cycles \
        $label $config $label $tdc_enabled $label $reference_levels \
        $label $timing_min $label $timing_max $label $timing_valid \
        $label $first_col $label $last_col $label $arrival_min $label $arrival_max \
        $label $spread $label $completion]
}

set result_path ""
if {$argc >= 1 && [lindex $argv 0] ne ""} {
    set result_path [file normalize [lindex $argv 0]]
}

if {[catch {connect} message]} {
    fail "connect failed: $message" 2
}
after 1000
if {[catch {targets -set -filter {name =~ "PSU"}} message]} {
    fail "Could not select PSU target: $message" 3
}

set report ""
append report [read_pe_tdc "pe0" 0xA0000000]
append report [read_pe_tdc "pe1" 0xA0010000]

puts "JTAG_DUAL_PE_TDC_RESULT:"
puts $report

if {$result_path ne ""} {
    if {[catch {
        set result_file [open $result_path w]
        puts -nonewline $result_file $report
        close $result_file
    } message]} {
        fail "Could not write result file: $message" 4
    }
}

disconnect
exit 0
