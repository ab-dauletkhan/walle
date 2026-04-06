#!/usr/bin/env bash

function _xdg_env_completions() {

  local curr
  curr="${COMP_WORDS[COMP_CWORD]}"

  local prev
  prev="${COMP_WORDS[COMP_CWORD-1]}"

  local options
  options=""
  if [[ "${prev}" == "-p" || "${prev}" == "--package" ]] ; then
    # Tab complete package names.
    local dirs
    dirs=""
    dirs+="$(find ~/.cache/ -maxdepth 1 -type d -printf '%f\n')"
    dirs+="$(find ~/.config/ -maxdepth 1 -type d -printf '%f\n')"
    dirs+="$(find ~/.local/share/ -maxdepth 1 -type d -printf '%f\n')"
    options+="$(echo "${dirs}" | sort -u)"
  else
    # Tab complete command options.
    options+=" -h --help"
    options+=" -V --version"
    options+=" -p --package"
    options+=" -d --defaults-only"
    options+=" -a --all"
  fi

  local suggestions
  suggestions=("$(compgen -W "${options}" -- "${curr}")")

  if [ "${#suggestions[@]}" == "1" ]; then
    # If there's only one match, we remove the command literal to proceed with
    # the automatic completion of the suggestion.
    COMPREPLY=("${suggestions[0]/%\ */}")
  else
    # More than one suggestions resolved, respond with the suggestions intact.
    COMPREPLY=("${suggestions[@]}")
  fi
}

complete -F _xdg_env_completions xdg-env
