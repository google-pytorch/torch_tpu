Negative confirmation on the same v6e-1, same script, caches off:

- Previous revision (c32405eea, `contiguous_format`): your four cases fail exactly as you described (f1 and f2 return the transposed values, f3 and f4 fail with the quoted `view()` errors); the original repro passes. 4/8.
- Base main (f7172418b): the original repro fails (`x * cum[1]` gives zeros, `x + cum[1]` gives `x`); your four cases pass. 5/8.
- This branch (de1abb50c): 8/8.

Log: https://github.com/joinalahmed/captum-on-torchtpu/blob/main/results/cpuview_negative_v6e1.txt
