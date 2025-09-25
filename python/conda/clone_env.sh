#!/bin/bash

# Name of the environment to create
ENV_NAME="cmb_py"

# YAML file containing the environment (make sure it is in the same folder)
YAML_FILE="cmb_py_environment.yml"

# Create the environment
conda env create -f $YAML_FILE -n $ENV_NAME

# Final message
echo "Environment '$ENV_NAME' created."
echo "To activate it, run: conda activate $ENV_NAME"
