#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Installs the gungraun gate's Valgrind in its pinned image (ci.yml's gungraun
# job, gungraun-bless.yml, benches/gungraun/README.md), from a fixed Debian
# snapshot, never the live archive.
#
# Valgrind depends on libc6-dbg, which must equal the installed libc6 exactly.
# From the live archive that upgrades the image's glibc to whatever Debian
# ships that day (2.41-12+deb13u4 to deb13u5 in Nov 2026, so the environment
# no longer matched benches/gungraun/environment.toml). The snapshot below is
# the last one with the image's glibc, deb13u4, and Valgrind 1:3.24.0-3.
# Moving it is a deliberate change, made with a reviewed re-bless.
#
# POSIX sh: the image's /bin/sh is dash. Runs as root in the container.
set -eu

SNAPSHOT=20261113T000000Z
VALGRIND=1:3.24.0-3
GLIBC=2.41-12+deb13u4

fail() { echo "gungraun_apt: $*" >&2; exit 1; }

# The only source: trixie main at the snapshot. A snapshot's Release file is
# past its Valid-Until, so that check is off; apt still verifies the archive
# signature with the image's keyring.
rm -f /etc/apt/sources.list.d/*.sources /etc/apt/sources.list.d/*.list
echo "deb [check-valid-until=no] https://snapshot.debian.org/archive/debian/$SNAPSHOT trixie main" > /etc/apt/sources.list

apt-get -o Acquire::Retries=5 update ||
    fail "apt-get update from snapshot.debian.org ($SNAPSHOT) failed: the snapshot service is unreachable or slow; re-run the job"
apt-get -o Acquire::Retries=5 install -y --no-upgrade "valgrind=$VALGRIND" "libc6-dbg=$GLIBC" ||
    fail "apt-get install of valgrind=$VALGRIND libc6-dbg=$GLIBC from the snapshot failed"

# --no-upgrade covers only the packages named above, not libc6 as a
# dependency: check what is installed, and stop if it is not the pin.
for pkg in libc6 libc6-dbg libc-bin; do
    got="$(dpkg-query -W -f '${Version}' "$pkg")"
    [ "$got" = "$GLIBC" ] || fail "$pkg is $got after the install, not $GLIBC"
done
got="$(dpkg-query -W -f '${Version}' valgrind)"
[ "$got" = "$VALGRIND" ] || fail "valgrind is $got after the install, not $VALGRIND"
ldd_line="$(ldd --version | head -1)"
case "$ldd_line" in
    *"GLIBC $GLIBC)"*) ;;
    *) fail "ldd reports '$ldd_line', not glibc $GLIBC" ;;
esac
echo "gungraun_apt: $ldd_line; valgrind $VALGRIND; snapshot $SNAPSHOT"
