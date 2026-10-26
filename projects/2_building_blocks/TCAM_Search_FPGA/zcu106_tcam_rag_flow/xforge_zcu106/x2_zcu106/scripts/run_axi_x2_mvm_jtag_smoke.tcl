# Run one complete X2 MVM transaction on a ZCU106 over JTAG.
#
# This path initializes the processing system and DDR, programs the PL, writes
# the packed accelerator stream into DDR, starts the accelerator through its
# AXI-Lite registers, and checks the output in DDR.  It does not require Linux
# or PYNQ on the board.
#
# Usage:
#   xsct.bat run_axi_x2_mvm_jtag_smoke.tcl \
#       <psu_init.tcl> <zcu106_axi_x2_mvm.bit> ?result.txt? ?case.tcl?

proc fail {message code} {
    puts "JTAG_MVM_FAIL: $message"
    if {[info exists ::result_path] && $::result_path ne ""} {
        catch {
            set result_file [open $::result_path w]
            puts $result_file "status=FAIL"
            puts $result_file "exit_code=$code"
            puts $result_file "error=$message"
            close $result_file
        }
    }
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

proc read_u32_range {address count} {
    set values {}
    set offset 0
    while {$offset < $count} {
        set chunk [expr {min(64, $count - $offset)}]
        set chunk_address [expr {$address + 4 * $offset}]
        set raw [mrd -force $chunk_address $chunk]
        set chunk_values {}
        foreach line [split $raw "\n"] {
            if {[regexp {^[^:]+:\s*(.*)$} $line _ payload]} {
                foreach value [regexp -all -inline -nocase {[0-9a-f]{8}} $payload] {
                    lappend chunk_values [expr {wide("0x$value") & 0xFFFFFFFF}]
                }
            }
        }
        if {[llength $chunk_values] != $chunk} {
            error [format \
                "Could not parse %u words from mrd block at 0x%08X (parsed %u): %s" \
                $chunk $chunk_address [llength $chunk_values] $raw]
        }
        set values [concat $values $chunk_values]
        incr offset $chunk
    }
    return $values
}

proc read_u32_range_retry {address count retries} {
    for {set attempt 1} {$attempt <= $retries} {incr attempt} {
        if {![catch {set values [read_u32_range $address $count]} message]} {
            return $values
        }
        puts [format \
            "JTAG_MVM: DDR read retry %u/%u at 0x%08X: %s" \
            $attempt $retries $address $message]
        after 100
    }
    error "DDR read failed after $retries attempts at [format 0x%08X $address]"
}

proc scattered_word_mismatches {base_addr expected_values} {
    set line_image [read_u32_range $base_addr [expr {4 * [llength $expected_values]}]]
    set mismatches {}
    for {set index 0} {$index < [llength $expected_values]} {incr index} {
        set got [lindex $line_image [expr {4 * $index}]]
        set expected [expr {wide([lindex $expected_values $index]) & 0xFFFFFFFF}]
        if {$got != $expected} {
            lappend mismatches $index
        }
    }
    return $mismatches
}

proc write_u32 {address value} {
    mwr -force $address [format 0x%08X [expr {wide($value) & 0xFFFFFFFF}]]
}

if {$argc < 2 || $argc > 4} {
    puts "Usage: xsct run_axi_x2_mvm_jtag_smoke.tcl <psu_init.tcl> <bitstream.bit> ?result.txt? ?case.tcl?"
    exit 2
}

set psu_init_path [file normalize [lindex $argv 0]]
set bit_path [file normalize [lindex $argv 1]]
set result_path ""
if {$argc >= 3 && [lindex $argv 2] ne ""} {
    set result_path [file normalize [lindex $argv 2]]
}
set case_path ""
if {$argc == 4} {
    set case_path [file normalize [lindex $argv 3]]
}

if {![file exists $psu_init_path]} {
    fail "Missing PS initialization script: $psu_init_path" 3
}
if {![file exists $bit_path]} {
    fail "Missing bitstream: $bit_path" 4
}
if {$case_path ne "" && ![file exists $case_path]} {
    fail "Missing MVM case file: $case_path" 22
}

set ACCEL_BASE 0xA0000000
set TX_BASE    0x10000000
set RX_BASE    0x11000000

if {[info exists ::env(X2_JTAG_TX_BASE)] && $::env(X2_JTAG_TX_BASE) ne ""} {
    set TX_BASE [expr {$::env(X2_JTAG_TX_BASE)}]
}
if {[info exists ::env(X2_JTAG_RX_BASE)] && $::env(X2_JTAG_RX_BASE) ne ""} {
    set RX_BASE [expr {$::env(X2_JTAG_RX_BASE)}]
}

set REG_CTRL       0x00
set REG_STATUS     0x04
set REG_CONFIG     0x08
set REG_EXPECTED   0x0C
set REG_RX_WORDS   0x10
set REG_TX_WORDS   0x14
set REG_CYCLES     0x18
set REG_ERROR      0x1C
set REG_MAGIC      0x20
set REG_VERSION    0x24
set REG_WR_ADDR_LO 0x70
set REG_WR_ADDR_HI 0x74
set REG_WR_COUNT   0x78
set REG_RD_ADDR_LO 0x80
set REG_RD_ADDR_HI 0x84
set REG_RD_COUNT   0x88

set MAGIC_X2MV 0x58324D56
set MIN_VERSION 0x00020022
set DUAL_PE_MODE 0
if {[info exists ::env(X2_JTAG_DUAL_PE)] && $::env(X2_JTAG_DUAL_PE) eq "1"} {
    set DUAL_PE_MODE 1
    set MIN_VERSION 0x00020024
}
if {[info exists ::env(X2_JTAG_MIN_VERSION)] && $::env(X2_JTAG_MIN_VERSION) ne ""} {
    set MIN_VERSION [expr {$::env(X2_JTAG_MIN_VERSION)}]
    puts [format "Using explicit test-only minimum version override: 0x%08X" $MIN_VERSION]
}

# The 3x2 weights [[4,5],[6,7],[8,9]] followed by vector [1,2,3].
# Each input word occupies the low 32 bits of a 128-bit DDR beat.
set case_name "builtin_3x2"
set rows 3
set cols 2
set batch 1
set packed_words {0x07060504 0x00000908 0x00030201}
set expected_values {40 46}

if {$case_path ne ""} {
    puts "JTAG_MVM: loading case $case_path"
    if {[catch {source $case_path} message]} {
        fail "Could not source MVM case: $message" 23
    }
    if {[info exists ::env(X2_JTAG_DISABLE_INPUT_BINARY)] &&
        $::env(X2_JTAG_DISABLE_INPUT_BINARY) eq "1"} {
        unset -nocomplain input_binary
        puts "JTAG_MVM: using scalar DDR writes for case input"
    }
}

if {$rows < 1 || $rows > 32 || $cols < 1 || $cols > 32 ||
    $batch < 1 || $batch > 32} {
    fail "Invalid case dimensions rows=$rows cols=$cols batch=$batch" 24
}
set required_input_words [expr {(($rows * $cols) + 3) / 4 +
                                $batch * (($rows + 3) / 4)}]
set required_output_words [expr {$batch * $cols}]
set output_storage_words [expr {(($required_output_words + 3) / 4) * 4}]
if {[llength $packed_words] != $required_input_words} {
    fail "Case has [llength $packed_words] packed words; expected $required_input_words" 25
}
if {[llength $expected_values] != $required_output_words} {
    fail "Case has [llength $expected_values] outputs; expected $required_output_words" 26
}
set normalized_expected {}
foreach value $expected_values {
    lappend normalized_expected [expr {wide($value) & 0xFFFFFFFF}]
}
set expected_values $normalized_expected
set config [expr {$rows | ($cols << 8) | ($batch << 16)}]

if {[catch {
puts "JTAG_MVM: connecting"
if {[catch {connect} message]} {
    fail "connect failed: $message" 5
}
after 2000

if {[catch {targets -set -filter {name =~ "PSU"}} message]} {
    fail "Could not select PSU target: $message" 6
}

puts "JTAG_MVM: resetting and initializing PS/DDR"
catch {rst -system} reset_message
after 1000
if {[catch {source $psu_init_path} message]} {
    fail "Could not source psu_init.tcl: $message" 7
}
if {[catch {psu_init} message]} {
    fail "psu_init failed: $message" 8
}
if {[catch {psu_ps_pl_isolation_removal} message]} {
    fail "PS-PL isolation removal failed: $message" 9
}
if {[catch {psu_ps_pl_reset_config} message]} {
    fail "PS-PL reset configuration failed: $message" 10
}
after 1000

if {[catch {targets -set -filter {name =~ "PL"}} message]} {
    fail "Could not select PL target: $message" 11
}
puts "JTAG_MVM: programming $bit_path"
if {[catch {fpga -file $bit_path} message]} {
    fail "PL programming failed: $message" 12
}
after 1000

if {[catch {targets -set -filter {name =~ "PSU"}} message]} {
    fail "Could not reselect PSU target: $message" 13
}

if {[catch {set magic [read_u32 [expr {$ACCEL_BASE + $REG_MAGIC}]]} message]} {
    fail "AXI-Lite magic read failed: $message" 14
}
set version [read_u32 [expr {$ACCEL_BASE + $REG_VERSION}]]
puts [format "JTAG_MVM: magic=0x%08X version=0x%08X" $magic $version]
if {$magic != $MAGIC_X2MV} {
    fail [format "Bad accelerator magic 0x%08X" $magic] 15
}
if {$version < $MIN_VERSION} {
    fail [format "Accelerator version 0x%08X is older than required 0x%08X" \
        $version $MIN_VERSION] 21
}

puts "JTAG_MVM: loading packed input"
set used_input_binary 0
set skip_input_verify 0
if {[info exists ::env(X2_JTAG_SKIP_INPUT_VERIFY)] &&
    $::env(X2_JTAG_SKIP_INPUT_VERIFY) eq "1"} {
    set skip_input_verify 1
}
if {[info exists input_binary] && [file exists $input_binary]} {
    puts "JTAG_MVM: bulk-loading custom-layout DDR image $input_binary"
    if {[catch {targets -set -filter {name =~ "Cortex-A53 #0"}} target_message]} {
        fail "Could not select A53 #0 for binary DDR transfer: $target_message" 28
    }
    if {[catch {dow -data $input_binary $TX_BASE} download_message]} {
        fail "Bulk DDR input download failed: $download_message" 29
    }
    if {$skip_input_verify} {
        puts "JTAG_MVM: skipping binary DDR verify; output and transfer counts remain strict"
    } else {
        set binary_verified 0
        for {set repair_attempt 0} {$repair_attempt < 32} {incr repair_attempt} {
            if {![catch {verify -data $input_binary $TX_BASE} verify_message]} {
                set binary_verified 1
                break
            }
            if {![regexp -nocase {address 0x([0-9a-f]+)} $verify_message _ bad_hex]} {
                fail "Bulk DDR input verification failed: $verify_message" 27
            }
            set bad_address [expr {wide("0x$bad_hex")}]
            set aligned_address [expr {$bad_address & ~3}]
            set byte_offset [expr {$aligned_address - $TX_BASE}]
            if {$byte_offset < 0 || $byte_offset >= 16 * [llength $packed_words]} {
                fail "Verifier returned an out-of-range address: $verify_message" 27
            }
            set line_index [expr {$byte_offset / 16}]
            set line_lane [expr {($byte_offset % 16) / 4}]
            set repair_value [expr {$line_lane == 0 ? [lindex $packed_words $line_index] : 0}]
            puts [format \
                "JTAG_MVM: repairing binary mismatch at 0x%08X (line %u lane %u)" \
                $aligned_address $line_index $line_lane]
            write_u32 $aligned_address $repair_value
        }
        if {!$binary_verified} {
            fail "Bulk DDR input did not verify after targeted repairs" 27
        }
    }
    if {[catch {targets -set -filter {name =~ "PSU"}} target_message]} {
        fail "Could not reselect PSU after binary DDR transfer: $target_message" 30
    }
    set used_input_binary 1
} else {
    set word_index 0
    foreach value $packed_words {
        set line_base [expr {$TX_BASE + 16 * $word_index}]
        write_u32 $line_base $value
        write_u32 [expr {$line_base + 4}] 0
        write_u32 [expr {$line_base + 8}] 0
        write_u32 [expr {$line_base + 12}] 0
        incr word_index
    }
}

# XSCT/DAP scalar writes are a debug transport, not a production loader. Read
# every 16-byte input line back and repair any lane-zero word that did not stick.
if {!$used_input_binary && !$skip_input_verify} {
    set input_mismatches {}
    for {set verify_attempt 0} {$verify_attempt < 3} {incr verify_attempt} {
        set input_mismatches [scattered_word_mismatches $TX_BASE $packed_words]
        if {[llength $input_mismatches] == 0} {
            break
        }
        puts [format "JTAG_MVM: repairing %u input words after DDR read-back attempt %u" \
            [llength $input_mismatches] [expr {$verify_attempt + 1}]]
        foreach word_index $input_mismatches {
            write_u32 [expr {$TX_BASE + 16 * $word_index}] \
                [lindex $packed_words $word_index]
        }
    }
    if {[llength $input_mismatches] != 0} {
        fail "DDR input read-back still differs at word indices $input_mismatches" 27
    }
} elseif {$skip_input_verify} {
    puts "JTAG_MVM: skipping DAP input read-back; output and transfer counts remain strict"
}
puts [format "JTAG_MVM: verified %u packed input words in DDR" [llength $packed_words]]

# Abort/clear any previous operation, configure DDR buffers, then start.
write_u32 [expr {$ACCEL_BASE + $REG_CTRL}] 0x00000102
write_u32 [expr {$ACCEL_BASE + $REG_CONFIG}] $config
write_u32 [expr {$ACCEL_BASE + $REG_RD_ADDR_LO}] $TX_BASE
write_u32 [expr {$ACCEL_BASE + $REG_RD_ADDR_HI}] 0
write_u32 [expr {$ACCEL_BASE + $REG_WR_ADDR_LO}] $RX_BASE
write_u32 [expr {$ACCEL_BASE + $REG_WR_ADDR_HI}] 0
write_u32 [expr {$ACCEL_BASE + $REG_CTRL}] 1

set status 0
set done 0
for {set poll 0} {$poll < 10000} {incr poll} {
    set status [read_u32 [expr {$ACCEL_BASE + $REG_STATUS}]]
    if {$status & 0x4} {
        break
    }
    if {$status & 0x2} {
        set poll_rd_count [read_u32 [expr {$ACCEL_BASE + $REG_RD_COUNT}]]
        set poll_wr_count [read_u32 [expr {$ACCEL_BASE + $REG_WR_COUNT}]]
        if {$poll_rd_count == $required_input_words &&
            $poll_wr_count == $required_output_words} {
            set done 1
            break
        }
    }
    after 1
}

set error_code [read_u32 [expr {$ACCEL_BASE + $REG_ERROR}]]
set expected_words [read_u32 [expr {$ACCEL_BASE + $REG_EXPECTED}]]
set rx_words [read_u32 [expr {$ACCEL_BASE + $REG_RX_WORDS}]]
set tx_words [read_u32 [expr {$ACCEL_BASE + $REG_TX_WORDS}]]
set cycles [read_u32 [expr {$ACCEL_BASE + $REG_CYCLES}]]
set rd_count [read_u32 [expr {$ACCEL_BASE + $REG_RD_COUNT}]]
set wr_count [read_u32 [expr {$ACCEL_BASE + $REG_WR_COUNT}]]

if {!$done} {
    fail [format "Accelerator did not complete, status=0x%08X error=0x%08X" $status $error_code] 17
}
if {$error_code != 0} {
    fail [format "Accelerator reported error 0x%08X" $error_code] 18
}
if {$rd_count != $required_input_words || $wr_count != $required_output_words} {
    fail "Unexpected AXI transfer counts: read=$rd_count write=$wr_count" 19
}

set got_values [read_u32_range $RX_BASE $required_output_words]

set report [format \
    "case=%s\nrows=%u\ncols=%u\nbatch=%u\nmagic=0x%08X\nversion=0x%08X\nstatus=0x%08X\nerror=0x%08X\nexpected_words=%u\nrx_words=%u\ntx_words=%u\nrd_count=%u\nwr_count=%u\ncycles=%u\noutput=%s\nexpected=%s\n" \
    $case_name $rows $cols $batch $magic $version $status $error_code $expected_words $rx_words $tx_words \
    $rd_count $wr_count $cycles $got_values $expected_values]
puts "JTAG_MVM_RESULT:"
puts $report

if {$result_path ne ""} {
    if {[catch {
        set result_file [open $result_path w]
        puts -nonewline $result_file $report
        close $result_file
    } message]} {
        fail "Could not write result file: $message" 16
    }
}

if {$got_values ne $expected_values} {
    fail "Output mismatch: got $got_values expected $expected_values" 20
}

if {$DUAL_PE_MODE} {
    set ACCEL_BASE_1 0xA0010000
    set RX_BASE_1    0x12000000
    if {[info exists ::env(X2_JTAG_RX_BASE_1)] && $::env(X2_JTAG_RX_BASE_1) ne ""} {
        set RX_BASE_1 [expr {$::env(X2_JTAG_RX_BASE_1)}]
    }
    set DUAL_PE_SEQUENTIAL 0
    if {[info exists ::env(X2_JTAG_DUAL_PE_SEQUENTIAL)] && $::env(X2_JTAG_DUAL_PE_SEQUENTIAL) ne ""} {
        set DUAL_PE_SEQUENTIAL [expr {$::env(X2_JTAG_DUAL_PE_SEQUENTIAL) != 0}]
    }

    set magic_1 [read_u32 [expr {$ACCEL_BASE_1 + $REG_MAGIC}]]
    set version_1 [read_u32 [expr {$ACCEL_BASE_1 + $REG_VERSION}]]
    puts [format "JTAG_DUAL_PE: pe1 magic=0x%08X version=0x%08X" $magic_1 $version_1]
    if {$magic_1 != $MAGIC_X2MV || $version_1 < $MIN_VERSION} {
        fail [format "PE1 identity mismatch: magic=0x%08X version=0x%08X" \
            $magic_1 $version_1] 31
    }

    if {$DUAL_PE_SEQUENTIAL} {
        puts "JTAG_DUAL_PE: sequential mode; starting PE1 after verified PE0 pass"
        write_u32 [expr {$ACCEL_BASE_1 + $REG_CTRL}] 0x00000102
        write_u32 [expr {$ACCEL_BASE_1 + $REG_CONFIG}] $config
        write_u32 [expr {$ACCEL_BASE_1 + $REG_RD_ADDR_LO}] $TX_BASE
        write_u32 [expr {$ACCEL_BASE_1 + $REG_RD_ADDR_HI}] 0
        write_u32 [expr {$ACCEL_BASE_1 + $REG_WR_ADDR_LO}] $RX_BASE_1
        write_u32 [expr {$ACCEL_BASE_1 + $REG_WR_ADDR_HI}] 0
        write_u32 [expr {$ACCEL_BASE_1 + $REG_CTRL}] 1

        set pe1_done 0
        for {set poll 0} {$poll < 10000} {incr poll} {
            set status_1 [read_u32 [expr {$ACCEL_BASE_1 + $REG_STATUS}]]
            if {$status_1 & 0x4} {
                break
            }
            if {$status_1 & 0x2} {
                set pe1_done 1
                break
            }
            after 1
        }

        set status_0 $status
        set error_0 $error_code
        set error_1 [read_u32 [expr {$ACCEL_BASE_1 + $REG_ERROR}]]
        set rd_count_0 $rd_count
        set rd_count_1 [read_u32 [expr {$ACCEL_BASE_1 + $REG_RD_COUNT}]]
        set wr_count_0 $wr_count
        set wr_count_1 [read_u32 [expr {$ACCEL_BASE_1 + $REG_WR_COUNT}]]
        set cycles_0 $cycles
        set cycles_1 [read_u32 [expr {$ACCEL_BASE_1 + $REG_CYCLES}]]
        if {!$pe1_done || $error_1 != 0} {
            fail [format \
                "Sequential PE1 run failed: done=%u status1=0x%08X error1=%u rd1=%u wr1=%u" \
                $pe1_done $status_1 $error_1 $rd_count_1 $wr_count_1] 36
        }
        if {$rd_count_1 != $required_input_words || $wr_count_1 != $required_output_words} {
            fail [format "Sequential PE1 counts differ: rd1=%u wr1=%u" $rd_count_1 $wr_count_1] 37
        }

        set dual_got_0 $got_values
        set dual_got_1 [read_u32_range_retry $RX_BASE_1 $required_output_words 5]
        if {$dual_got_1 ne $expected_values} {
            fail "Sequential PE1 output mismatch: pe1=$dual_got_1 expected=$expected_values" 38
        }
        append report [format \
            "dual_pe_mode=sequential\npe1_magic=0x%08X\npe1_version=0x%08X\npe0_status=0x%08X\npe1_status=0x%08X\npe0_error=%u\npe1_error=%u\npe0_rd_count=%u\npe1_rd_count=%u\npe0_wr_count=%u\npe1_wr_count=%u\npe0_cycles=%u\npe1_cycles=%u\npe0_output=%s\npe1_output=%s\n" \
            $magic_1 $version_1 $status_0 $status_1 $error_0 $error_1 \
            $rd_count_0 $rd_count_1 $wr_count_0 $wr_count_1 $cycles_0 $cycles_1 \
            $dual_got_0 $dual_got_1]
        if {$result_path ne ""} {
            set result_file [open $result_path w]
            puts -nonewline $result_file $report
            close $result_file
        }
        puts [format \
            "JTAG_DUAL_PE_SEQUENTIAL_PASS: PE0 then PE1 each completed %u inputs and %u outputs" \
            $required_input_words $required_output_words]
    } else {
        # Configure both complete 32x32 engines before either is started. Both
        # readers may share the immutable input image; each writer has its own DDR
        # output buffer.
        write_u32 [expr {$ACCEL_BASE + $REG_CTRL}] 0x00000102
        write_u32 [expr {$ACCEL_BASE_1 + $REG_CTRL}] 0x00000102
        foreach {base rx_base} [list $ACCEL_BASE $RX_BASE $ACCEL_BASE_1 $RX_BASE_1] {
        write_u32 [expr {$base + $REG_CONFIG}] $config
        write_u32 [expr {$base + $REG_RD_ADDR_LO}] $TX_BASE
        write_u32 [expr {$base + $REG_RD_ADDR_HI}] 0
        write_u32 [expr {$base + $REG_WR_ADDR_LO}] $rx_base
        write_u32 [expr {$base + $REG_WR_ADDR_HI}] 0
        }

        puts "JTAG_DUAL_PE: starting PE0 and PE1"
        write_u32 [expr {$ACCEL_BASE + $REG_CTRL}] 1
        write_u32 [expr {$ACCEL_BASE_1 + $REG_CTRL}] 1

        set dual_done 0
        for {set poll 0} {$poll < 10000} {incr poll} {
            set status_0 [read_u32 [expr {$ACCEL_BASE + $REG_STATUS}]]
            set status_1 [read_u32 [expr {$ACCEL_BASE_1 + $REG_STATUS}]]
            if {($status_0 & 0x4) || ($status_1 & 0x4)} {
                break
            }
            if {($status_0 & 0x2) && ($status_1 & 0x2)} {
                set dual_done 1
                break
            }
            after 1
        }

        set error_0 [read_u32 [expr {$ACCEL_BASE + $REG_ERROR}]]
        set error_1 [read_u32 [expr {$ACCEL_BASE_1 + $REG_ERROR}]]
        set rd_count_0 [read_u32 [expr {$ACCEL_BASE + $REG_RD_COUNT}]]
        set rd_count_1 [read_u32 [expr {$ACCEL_BASE_1 + $REG_RD_COUNT}]]
        set wr_count_0 [read_u32 [expr {$ACCEL_BASE + $REG_WR_COUNT}]]
        set wr_count_1 [read_u32 [expr {$ACCEL_BASE_1 + $REG_WR_COUNT}]]
        set cycles_0 [read_u32 [expr {$ACCEL_BASE + $REG_CYCLES}]]
        set cycles_1 [read_u32 [expr {$ACCEL_BASE_1 + $REG_CYCLES}]]
        if {!$dual_done || $error_0 != 0 || $error_1 != 0} {
            fail [format \
                "Dual PE run failed: done=%u status0=0x%08X status1=0x%08X error0=%u error1=%u" \
                $dual_done $status_0 $status_1 $error_0 $error_1] 32
        }
        if {$rd_count_0 != $required_input_words ||
            $rd_count_1 != $required_input_words ||
            $wr_count_0 != $required_output_words ||
            $wr_count_1 != $required_output_words} {
            fail [format \
                "Dual PE counts differ: rd0=%u rd1=%u wr0=%u wr1=%u" \
                $rd_count_0 $rd_count_1 $wr_count_0 $wr_count_1] 33
        }

        set dual_got_0 [read_u32_range_retry $RX_BASE $required_output_words 5]
        set dual_got_1 [read_u32_range_retry $RX_BASE_1 $required_output_words 5]
        if {$dual_got_0 ne $expected_values || $dual_got_1 ne $expected_values} {
            fail "Dual PE output mismatch: pe0=$dual_got_0 pe1=$dual_got_1 expected=$expected_values" 34
        }
        append report [format \
            "dual_pe_mode=1\npe1_magic=0x%08X\npe1_version=0x%08X\npe0_status=0x%08X\npe1_status=0x%08X\npe0_error=%u\npe1_error=%u\npe0_rd_count=%u\npe1_rd_count=%u\npe0_wr_count=%u\npe1_wr_count=%u\npe0_cycles=%u\npe1_cycles=%u\npe0_output=%s\npe1_output=%s\n" \
            $magic_1 $version_1 $status_0 $status_1 $error_0 $error_1 \
            $rd_count_0 $rd_count_1 $wr_count_0 $wr_count_1 $cycles_0 $cycles_1 \
            $dual_got_0 $dual_got_1]
        if {$result_path ne ""} {
            set result_file [open $result_path w]
            puts -nonewline $result_file $report
            close $result_file
        }
        puts [format \
            "JTAG_DUAL_PE_PASS: PE0 and PE1 each completed %u inputs and %u outputs" \
            $required_input_words $required_output_words]
    }
}

puts "JTAG_MVM_PASS: physical ZCU106 case $case_name matched $required_output_words outputs"
disconnect
} message]} {
    fail "Unhandled JTAG operation failed: $message" 35
}
exit 0
