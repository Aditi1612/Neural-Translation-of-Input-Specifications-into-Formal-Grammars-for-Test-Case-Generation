#!/bin/bash

rsync -zvhPrtL --update \
  "node07:projects/test-case-generation-arr/$(basename ${DATA_DIR})/{grammar,generation-result,parsing-result,testcase,execution-summary}" \
  ${DATA_DIR}
