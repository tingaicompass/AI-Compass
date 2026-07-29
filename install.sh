#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

if [[ ! -f "$repo_dir/SKILL.md" ]]; then
  printf 'SKILL.md was not found in the repository root: %s\n' "$repo_dir" >&2
  exit 1
fi

for skills_dir in "$HOME/.codex/skills" "$HOME/.claude/skills"; do
  link_path="$skills_dir/ai-compass"
  mkdir -p "$skills_dir"

  if [[ -e "$link_path" && ! -L "$link_path" ]]; then
    printf 'Refusing to replace a non-symlink path: %s\n' "$link_path" >&2
    exit 1
  fi

  ln -sfn "$repo_dir" "$link_path"
  printf 'Linked %s -> %s\n' "$link_path" "$repo_dir"
done

printf 'AI-Compass is installed. Restart Codex and Claude Code to load the Skill.\n'
