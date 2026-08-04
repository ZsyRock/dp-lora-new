#!/usr/bin/env bash

# Classify one or more child-process return codes.  A genuine failure always
# dominates a checkpoint stop, including when a failed concurrent lane asks
# its sibling to stop cleanly with code 75.
classify_full_slaclip_exit_codes() {
    local return_code
    local saw_checkpoint_stop=0
    if [[ "$#" -eq 0 ]]; then
        echo "INVALID"
        return 2
    fi
    for return_code in "$@"; do
        if [[ ! "$return_code" =~ ^[0-9]+$ || "$return_code" -gt 255 ]]; then
            echo "INVALID"
            return 2
        fi
        if [[ "$return_code" -ne 0 && "$return_code" -ne 75 ]]; then
            echo "HARD_FAILURE"
            return 0
        fi
        if [[ "$return_code" -eq 75 ]]; then
            saw_checkpoint_stop=1
        fi
    done
    if [[ "$saw_checkpoint_stop" -eq 1 ]]; then
        echo "CHECKPOINTED_STOP"
    else
        echo "SUCCESS"
    fi
}

# Bash interrupts ``wait`` with 128+signal after running a trap, even though
# the named child is still alive.  Retry only when a stop request arrived
# during that particular wait; a repeated wait returns the child's real status
# (including a genuine signal exit) once no newer request interrupted it.
wait_for_full_slaclip_child() {
    local child_pid="${1:-}"
    local observed_generation
    local current_generation
    local return_code
    if [[ "$#" -ne 1 || ! "$child_pid" =~ ^[1-9][0-9]*$ ]]; then
        return 2
    fi
    while true; do
        observed_generation="${full_slaclip_stop_generation:-0}"
        wait "$child_pid"
        return_code=$?
        current_generation="${full_slaclip_stop_generation:-0}"
        if [[ ! "$observed_generation" =~ ^[0-9]+$ || ! "$current_generation" =~ ^[0-9]+$ ]]; then
            return 2
        fi
        if [[ "$return_code" -gt 128 && "$current_generation" -gt "$observed_generation" ]]; then
            continue
        fi
        return "$return_code"
    done
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    set -euo pipefail
    classify_full_slaclip_exit_codes "$@"
fi
