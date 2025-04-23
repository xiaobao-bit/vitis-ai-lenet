#!/bin/sh

XMODEL_DIR="/workspace/xmodel_generator/quantize_result/LeNet_int.xmodel"
ARCH_DIR="/workspace/xmodel_generator/arch/arch.json"
OUTPUT="/workspace/xmodel_generator/xmodel/LeNet"

#!/bin/bash

vai_c_xir \
  -x $XMODEL_DIR \
  -a $ARCH_DIR \
  -o $OUTPUT \
  -n LeNet
