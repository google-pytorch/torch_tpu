Follow-up on the v6e-4, same branch wheel: `pytest tests/ops_unit_test.py -k "piecewise or smooth_l1 or huber or logaddexp or nansum or std_mean or trace or addbmm"` against the installed wheel: 53 passed. Both the bazel target and the pytest run pass on the 4-chip host.

Log: https://github.com/joinalahmed/captum-on-torchtpu/blob/main/results/v6e-4/run2/results_missing-ops-batch.txt
