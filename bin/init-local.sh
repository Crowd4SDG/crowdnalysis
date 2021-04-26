#!/usr/bin/env bash
if [[ ! -d .venv ]]; then
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:${PWD}/src

CMDSTAN_DIR="$HOME/.cmdstan"
if [[ "$OSTYPE" == "linux-gnu"*  ||  "$OSTYPE" == "darwin"* ]]; then
  if [[ ! -d "$CMDSTAN_DIR" ]]; then
    echo "Installing CmdStan library into '${CMDSTAN_DIR}', please wait..."
    install_cmdstan
    echo "CmdStan library installed."
  else
    echo "CmdStan library already installed in '${CMDSTAN_DIR}'."
  fi
fi