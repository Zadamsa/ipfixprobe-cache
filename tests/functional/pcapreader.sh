#!/bin/bash

export LC_ALL=C
export LANG=C

test -z "$srcdir" && export srcdir=.

ipfixprobe_bin=../../ipfixprobe
if [ -x ../../../logger/logger ]; then
   logger_bin=../../../logger/logger
elif [ -x /usr/local/bin/logger ]; then
   logger_bin=/usr/local/bin/logger
else
   logger_bin=/usr/bin/nemea/logger
fi

pcap_dir=$srcdir/../../pcaps
ref_dir=$srcdir/reference
output_dir=./output
file_out="$$.data"

# Usage: run_pcap_reader_test <data file> <reference fie>
run_pcap_reader_test() {
   if ! [ -f "$ipfixprobe_bin" ]; then
      echo "ipfixprobe not compiled"
      return 77
   fi

   if ! `"$ipfixprobe_bin" -h unirec |head -1 | grep -q '^unirec'`; then
      echo "compiled without NEMEA"
      return 77
   fi

   if ! [ -f "$logger_bin" ]; then
      echo "logger not compiled"
      return 77
   fi

   if ! [ -d "$output_dir" ]; then
      mkdir "$output_dir"
   fi

   "$ipfixprobe_bin" -i "pcap;file=$1" -o "unirec;ifc=f:${output_dir}/${file_out}:buffer=off:timeout=WAIT;id=0" >/dev/null
   "$logger_bin"     -i f:"$output_dir/$file_out" -t | sort > "$output_dir/$2"
   rm "$output_dir/$file_out"

   if sort "$ref_dir/$2" | diff -u "$output_dir/$2" -s - ; then
      echo "$2 pcap reader test OK"
   else
      echo "$2 pcap reader test FAILED"
      return 1
   fi
}

