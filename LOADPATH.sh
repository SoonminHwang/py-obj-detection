#!/usr/bin/env bash
WORKSPACE_DIR=$(pwd)
echo "Working directory: ${WORKSPACE_DIR}"
export PYTHONPATH="$WORKSPACE_DIR/caffe-latest/python:$WORKSPACE_DIR/libs"