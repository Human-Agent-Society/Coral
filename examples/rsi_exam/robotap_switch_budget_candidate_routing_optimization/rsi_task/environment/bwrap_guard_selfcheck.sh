#!/bin/sh
set -eu

GUARD=/usr/bin/bwrap

expect_reject() {
    expected=$1
    shift
    set +e
    output=$("$GUARD" --unshare-pid --unshare-net --die-with-parent --new-session "$@" -- /bin/true 2>&1)
    return_code=$?
    set -e
    if [ "$return_code" -ne 126 ]; then
        echo "guard unexpectedly accepted arguments: $*" >&2
        exit 1
    fi
    case "$output" in
        *"$expected"*) ;;
        *)
            echo "guard rejection did not contain '$expected': $output" >&2
            exit 1
            ;;
    esac
}

expect_raw_reject() {
    expected=$1
    shift
    set +e
    output=$("$GUARD" "$@" -- /bin/true 2>&1)
    return_code=$?
    set -e
    if [ "$return_code" -ne 126 ]; then
        echo "guard unexpectedly accepted raw arguments: $*" >&2
        exit 1
    fi
    case "$output" in
        *"$expected"*) ;;
        *)
            echo "raw guard rejection did not contain '$expected': $output" >&2
            exit 1
            ;;
    esac
}

cd /
ln -s /proc /tmp/ara-guard-proc-link

expect_reject "absolute paths" --ro-bind proc /mnt
expect_reject "fixed sandbox allowlist" --ro-bind /proc /mnt
expect_reject "fixed sandbox allowlist" --ro-bind /tmp/ara-guard-proc-link /mnt
expect_reject "fixed sandbox allowlist" --ro-bind / /mnt
expect_reject "writable binds" --bind / /
expect_reject "safe pseudo-device" --dev-bind / /
expect_reject "safe pseudo-device" --dev-bind /dev/full /dev/full
expect_reject "safe pseudo-device" --dev-bind /logs /logs
expect_reject "writable binds" --bind /logs /logs
expect_reject "best-effort bind" --bind-try /app /app
expect_reject "namespace, privilege" --unshare-all
expect_reject "namespace, privilege" --userns 3
expect_reject "namespace, privilege" --userns2 3
expect_reject "namespace, privilege" --pidns 3
expect_reject "namespace, privilege" --uid 0
expect_reject "namespace, privilege" --gid 0
expect_reject "namespace, privilege" --cap-add ALL
expect_reject "namespace, privilege" --bind-fd 3 /mnt
expect_reject "argument-file" --args 3
expect_reject "outside the fixed sandbox allowlist" --file 3 /sandbox/file
expect_reject "outside the fixed sandbox allowlist" --bind-data 3 /sandbox/file
expect_reject "inline proc and dev" --proc=/proc
expect_reject "inline proc and dev" --dev=/dev
expect_reject "missing a required operand" --setenv

# These option-looking strings are values, not isolation flags. A token scan
# would incorrectly count them and allow Bubblewrap to consume the real flags.
expect_raw_reject "pid, network" \
    --setenv X --unshare-pid \
    --setenv Y --unshare-net \
    --die-with-parent --new-session
expect_raw_reject "pid, network" \
    --setenv --unshare-pid X \
    --setenv --unshare-net Y \
    --die-with-parent --new-session

rm -f /tmp/ara-guard-proc-link
"$GUARD" --version >/dev/null
echo "bwrap guard parser selfcheck ok"
