# End-to-end results — `artifacts\devGLM` + `artifacts\devGLM_r2`

| method | n | Acc | mean score | Pass^2 | Pass@2 | steps | max-steps | refetch/ep | blocked/ep | peak prompt tok | compactions/ep | compressor tok/ep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| compass_v2 | 30 | 70.0% | 0.91 | 60.0% | 76.7% | 16.6 | 0% | 2.2 | 0.4 | 6705 | 1.9 | 14304 |
| full | 30 | 80.0% | 0.96 | 73.3% | 93.3% | 15.4 | 0% | 1.2 | 0.6 | 10311 | 0.0 | 0 |
| openclaw | 30 | 73.3% | 0.93 | 60.0% | 83.3% | 15.7 | 0% | 1.8 | 0.7 | 6382 | 2.0 | 14810 |