# End-to-end TCAM accelerator smoke test over the Zynq debug interface.
set base 0xA0000000

proc reg_write {offset value} {
    global base
    mwr [expr {$base + $offset}] $value
}

proc reg_read {offset} {
    global base
    return [mrd -value [expr {$base + $offset}]]
}

connect
set apu_targets [targets -filter {name =~ "*APU*"}]
if {[llength $apu_targets] == 0} {
    error "No APU debug target found."
}
targets -set [lindex $apu_targets 0]
configparams force-mem-access 1
memmap -addr $base -size 0x10000 -flags 3

set id_value [reg_read 0x00]
if {$id_value != 0x5443414D} {
    error [format "Unexpected accelerator ID 0x%08X" $id_value]
}
set version_value [reg_read 0x04]
if {$version_value != 0x00020000} {
    error [format "Unexpected differential TCAM version 0x%08X" $version_value]
}

# Reset the core, then install entries 0x1234 through 0x1243.
reg_write 0x04 0x4
after 10
for {set index 0} {$index < 16} {incr index} {
    reg_write 0x18 $index
    reg_write 0x1C [expr {0x1234 + $index}]
    reg_write 0x20 0xFFFF
    reg_write 0x24 [expr {0x1000 + $index}]
    reg_write 0x28 0x1
}

reg_write 0x0C 0x55AA
reg_write 0x10 0x1234
reg_write 0x14 0xFFFF
reg_write 0x04 0x1

set ready 0
for {set attempt 0} {$attempt < 100} {incr attempt} {
    if {[reg_read 0x08] & 0x2} {
        set ready 1
        break
    }
    after 10
}
if {!$ready} {
    error "Timed out waiting for accelerator results."
}

set expected_chunk {0x1000 0x1001 0x1002 0x1008}
set expected_hd {0 1 1 1}
set expected_score {32 30 30 30}
set expected_a {0 1 1 1}
set expected_b {16 15 15 15}
set expected_diff {16 14 14 14}
for {set expected_rank 0} {$expected_rank < 4} {incr expected_rank} {
    set query_id [reg_read 0x2C]
    set meta [reg_read 0x30]
    set score_word [reg_read 0x34]
    set diff_word [reg_read 0x3C]
    set rank [expr {($meta >> 17) & 0x3}]
    set chunk [expr {$meta & 0xFFFF}]
    set last [expr {($meta >> 31) & 0x1}]
    set hd [expr {($score_word >> 8) & 0x1F}]
    set score [expr {$score_word & 0x3F}]
    set a_current [expr {$diff_word & 0x1F}]
    set b_current [expr {($diff_word >> 8) & 0x1F}]
    set diff_raw [expr {($diff_word >> 16) & 0x3F}]
    set b_minus_a [expr {$diff_raw >= 32 ? $diff_raw - 64 : $diff_raw}]

    puts [format "RESULT rank=%d query=0x%04X chunk=0x%04X A=%d B=%d B-A=%d TDC=%d HD=%d last=%d" \
        $rank $query_id $chunk $a_current $b_current $b_minus_a $score $hd $last]

    if {$query_id != 0x55AA || $rank != $expected_rank ||
        $chunk != [lindex $expected_chunk $expected_rank] ||
        $hd != [lindex $expected_hd $expected_rank] ||
        $score != [lindex $expected_score $expected_rank] ||
        $a_current != [lindex $expected_a $expected_rank] ||
        $b_current != [lindex $expected_b $expected_rank] ||
        $b_minus_a != [lindex $expected_diff $expected_rank] ||
        $b_minus_a != ($b_current - $a_current) ||
        $score != ($b_minus_a + $a_current + $b_current) ||
        $hd != (($a_current + $b_current - $b_minus_a) / 2) ||
        $last != ($expected_rank == 3)} {
        error "Result verification failed at rank $expected_rank."
    }
    reg_write 0x04 0x2
}

set completed [reg_read 0x38]
if {$completed != 1} {
    error "Expected one completed query, got $completed."
}

# Recreate the attached 6x6 test in the low three logical features. The
# original keys "100", "000", and "011" become 0x1, 0x0, and 0x6 when the
# first string symbol maps to logical feature 0. Query "10X" uses care 0x3.
set demo_keys {0x1 0x0 0x6}
for {set index 0} {$index < 16} {incr index} {
    reg_write 0x18 $index
    if {$index < 3} {
        reg_write 0x1C [lindex $demo_keys $index]
        reg_write 0x20 0x0007
    } else {
        reg_write 0x1C 0x0000
        reg_write 0x20 0x0000
    }
    reg_write 0x24 [expr {0x2000 + $index}]
    reg_write 0x28 0x1
}

reg_write 0x0C 0x6006
reg_write 0x10 0x0001
reg_write 0x14 0x0003
reg_write 0x04 0x1

set ready 0
for {set attempt 0} {$attempt < 100} {incr attempt} {
    if {[reg_read 0x08] & 0x2} {
        set ready 1
        break
    }
    after 10
}
if {!$ready} {
    error "Timed out waiting for scaled 6x6 results."
}

set demo_chunk {0x2000 0x2001 0x2002}
set demo_a {0 1 2}
set demo_b {2 1 0}
set demo_diff {2 0 -2}
set demo_tdc {4 2 0}
set demo_hd {0 1 2}
for {set expected_rank 0} {$expected_rank < 4} {incr expected_rank} {
    set query_id [reg_read 0x2C]
    set meta [reg_read 0x30]
    set score_word [reg_read 0x34]
    set diff_word [reg_read 0x3C]
    set rank [expr {($meta >> 17) & 0x3}]
    set chunk [expr {$meta & 0xFFFF}]
    set hd [expr {($score_word >> 8) & 0x1F}]
    set score [expr {$score_word & 0x3F}]
    set a_current [expr {$diff_word & 0x1F}]
    set b_current [expr {($diff_word >> 8) & 0x1F}]
    set diff_raw [expr {($diff_word >> 16) & 0x3F}]
    set b_minus_a [expr {$diff_raw >= 32 ? $diff_raw - 64 : $diff_raw}]

    puts [format "DEMO6_SCALED rank=%d query=0x%04X chunk=0x%04X A=%d B=%d B-A=%d TDC=%d HD=%d" \
        $rank $query_id $chunk $a_current $b_current $b_minus_a $score $hd]

    if {$query_id != 0x6006 || $rank != $expected_rank} {
        error "Scaled 6x6 query/rank verification failed at rank $expected_rank."
    }
    if {$expected_rank < 3 &&
        ($chunk != [lindex $demo_chunk $expected_rank] ||
         $a_current != [lindex $demo_a $expected_rank] ||
         $b_current != [lindex $demo_b $expected_rank] ||
         $b_minus_a != [lindex $demo_diff $expected_rank] ||
         $score != [lindex $demo_tdc $expected_rank] ||
         $hd != [lindex $demo_hd $expected_rank])} {
        error "Scaled 6x6 result verification failed at rank $expected_rank."
    }
    reg_write 0x04 0x2
}

set completed [reg_read 0x38]
if {$completed != 2} {
    error "Expected two completed queries, got $completed."
}
puts [format "PASS: ID=0x%08X, version=0x%08X, physical_array=32x32, queries_completed=%d" \
    $id_value $version_value $completed]
disconnect
