#!/usr/bin/env bash

# Preserve Harbor's host-side ownership across root-run atomic verifier writes.

capture_reward_log_owner() {
    local directory="$1"
    if [[ -L "$directory" || ! -d "$directory" ]]; then
        echo "unsafe verifier log mount: $directory" >&2
        return 2
    fi
    REWARD_LOG_UID="$(stat -c '%u' -- "$directory")"
    REWARD_LOG_GID="$(stat -c '%g' -- "$directory")"
    [[ "$REWARD_LOG_UID" =~ ^[0-9]+$ && "$REWARD_LOG_GID" =~ ^[0-9]+$ ]]
}

restore_reward_log_owner() {
    local directory="$1"
    local name
    local path
    if [[ -L "$directory" || ! -d "$directory" ]]; then
        echo "refusing to restore unsafe verifier log mount: $directory" >&2
        return 2
    fi
    for name in reward.json reward.txt grade_debug.json score_details.json; do
        path="$directory/$name"
        if [[ -e "$path" || -L "$path" ]]; then
            if [[ -L "$path" || ! -f "$path" ]]; then
                echo "refusing to restore unsafe verifier output: $path" >&2
                return 2
            fi
            if [[ "$(id -u)" -eq 0 ]]; then
                chown --no-dereference "${REWARD_LOG_UID}:${REWARD_LOG_GID}" "$path"
            fi
        fi
    done
    if [[ "$(id -u)" -eq 0 ]]; then
        chown --no-dereference "${REWARD_LOG_UID}:${REWARD_LOG_GID}" "$directory"
    fi
    chmod 0700 "$directory"
}
