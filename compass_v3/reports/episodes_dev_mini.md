# End-to-end results — `artifacts\dev`

| method | n | Acc | mean score | Pass^2 | Pass@2 | steps | max-steps | refetch/ep | blocked/ep | peak prompt tok | compactions/ep | compressor tok/ep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| compass_v2 | 30 | 30.0% | 0.64 | - | - | 20.7 | 0% | 4.2 | 5.1 | 5996 | 1.1 | 2924 |
| full | 30 | 26.7% | 0.63 | - | - | 17.8 | 0% | 3.4 | 3.6 | 8102 | 0.0 | 0 |
| openclaw | 30 | 26.7% | 0.62 | - | - | 22.8 | 3% | 4.8 | 5.2 | 6037 | 1.5 | 8311 |