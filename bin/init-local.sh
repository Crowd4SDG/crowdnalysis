#!/usr/bin/env bash
if [[ ! -d .venv ]]; then
    python3 -m venv .venv
    echo "Created venv"
fi

source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:${PWD}/src

CMDSTAN_DIR="$HOME/.cmdstan"
CMDSTAN_VER="2.26.1"
if [[ "$OSTYPE" == "linux-gnu"*  ||  "$OSTYPE" == "darwin"* ]]; then
  if [[ ! -d "$CMDSTAN_DIR" ]]; then
    echo "Installing CmdStan library into '${CMDSTAN_DIR}', please wait..."
    install_cmdstan -d $CMDSTAN_DIR  -v $CMDSTAN_VER
    echo "CmdStan library installed (v${CMDSTAN_VER})."
  else
    echo "CmdStan library (v${CMDSTAN_VER}) already installed in '${CMDSTAN_DIR}'."
  fi
fi