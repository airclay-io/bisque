#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Airclay LLC
#
# Checks tracked source files for the accepted SPDX license header.
set -euo pipefail

# Accepted header (exact match at end of line).
allowed_re="Apache-2[.]0"
allowed="Apache-2.0"
status=0

while IFS= read -r -d '' file; do
  case "$file" in
    *.rs | *.sh) ;;
    *) continue ;;
  esac
  header="$(head -n 5 "$file")"
  if ! grep -qE "SPDX-License-Identifier: ($allowed_re)\$" <<<"$header"; then
    if grep -q "SPDX-License-Identifier:" <<<"$header"; then
      echo "::error file=$file::SPDX license is not accepted (want: $allowed)"
    else
      echo "::error file=$file::missing SPDX license header ('SPDX-License-Identifier: Apache-2.0')"
    fi
    status=1
  fi
done < <(git ls-files -z)

if [ "$status" -eq 0 ]; then
  echo "All source files have an accepted SPDX license header."
fi
exit "$status"
