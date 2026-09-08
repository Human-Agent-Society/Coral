#!/usr/bin/env bash
set -euo pipefail

source /tests/reward_log_owner.sh

if [[ "$(id -u)" -ne 0 ]]; then
    echo "reward log owner regression requires root; skipped"
    exit 0
fi

temporary="$(mktemp -d)"
trap 'rm -rf "$temporary"' EXIT
reward_dir="$temporary/verifier"
mkdir "$reward_dir"
chown 61224:61224 "$reward_dir"
chmod 0700 "$reward_dir"

capture_reward_log_owner "$reward_dir"
chown 0:0 "$reward_dir"
printf '%s\n' '{"reward":0.25}' > "$reward_dir/reward.json"
printf '%s\n' '{"correctness":true,"raw_metric":0.5}' > "$reward_dir/score_details.json"
restore_reward_log_owner "$reward_dir"

[[ "$(stat -c '%u:%g' "$reward_dir")" == "61224:61224" ]]
[[ "$(stat -c '%u:%g' "$reward_dir/reward.json")" == "61224:61224" ]]
[[ "$(stat -c '%u:%g' "$reward_dir/score_details.json")" == "61224:61224" ]]
[[ "$(stat -c '%a' "$reward_dir")" == "700" ]]

rm "$reward_dir/reward.json"
ln -s /etc/passwd "$reward_dir/reward.json"
if restore_reward_log_owner "$reward_dir"; then
    echo "symlink verifier output was accepted" >&2
    exit 1
fi
rm "$reward_dir/reward.json"

mkdir "$temporary/real"
ln -s "$temporary/real" "$temporary/link"
if capture_reward_log_owner "$temporary/link"; then
    echo "symlink log path was accepted" >&2
    exit 1
fi

echo "reward log ownership regression ok"
