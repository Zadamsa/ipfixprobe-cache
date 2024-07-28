#!/bin/sh

test -z "$srcdir" && export srcdir=.

. $srcdir/pcapreader.sh

run_pcap_reader_test "$pcap_dir/invalid_ipv6_ext_len.pcap" invalid_ipv6_ext_len

