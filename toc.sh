#!/bin/bash

# Script to create a markdown version of the 'tree' command.

tree=$(tree -tf --noreport -L 2 -I '*~' -I '__*__' --charset ascii $1 |
       sed -e 's/| \+/  /g' -e 's/[|`]-\+/ */g' -e 's:\(* \)\(\(.*/\)\([^/]\+\)\):\1[\4](\2):g')

printf "# Project tree\n\n${tree}"

