# End-to-end results — `artifacts\devO`

| method | n | Acc | mean score | Pass^2 | Pass@2 | steps | max-steps | refetch/ep | blocked/ep | peak prompt tok | compactions/ep | compressor tok/ep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| compass_v2 | 30 | 66.7% | 0.88 | - | - | 21.3 | 3% | 2.4 | 0.7 | 6616 | 2.8 | 27644 |
| full | 30 | 83.3% | 0.94 | - | - | 20.0 | 0% | 1.7 | 0.9 | 11205 | 0.0 | 0 |
| openclaw | 30 | 73.3% | 0.93 | - | - | 24.0 | 7% | 4.8 | 1.2 | 6959 | 3.6 | 29661 |