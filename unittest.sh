#!/bin/bash
NUM_TESTS=10

cd ./GeoCodeBench

for project in [0-9]*_*/; do
  for unittest in "${project}"unittest*/; do
    if [[ -f "${unittest}test_runner.py" ]]; then
      echo "=== $unittest ==="
      cd "$unittest"
      python test_runner.py --num-tests "$NUM_TESTS" --quiet || true
      cd - > /dev/null
    fi
  done
done

# bash unittest.sh 2>&1 | tee log.txt
