#!/usr/bin/env bash
# Optimize and map one multiplier with ABC, then check the result is still the same function.
#   bash hardware/synth.sh evoapproxlib mul8u_1AGV path/to/abc
set -euo pipefail
LIB=${1:-evoapproxlib}; NAME=${2:-mul8u_1AGV}; ABC=${3:-abc}
HERE=$(cd "$(dirname "$0")" && pwd)
python3 -c "
import sys; sys.path.insert(0, '$HERE')
from netlist import Netlist, find_netlist
Netlist(find_netlist('$LIB', '$NAME')).to_blif('/tmp/$NAME.blif')"
"$ABC" -c "read_blif /tmp/$NAME.blif; strash; dc2; dc2; balance; rewrite; refactor; balance; rewrite; rewrite -z; balance; refactor -z; rewrite -z; balance; print_stats; read_library $HERE/cells.genlib; map -a; print_stats; cec /tmp/$NAME.blif"
