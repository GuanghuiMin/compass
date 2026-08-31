# End-to-end results — `artifacts\fullO_r1`

| method | n | Acc | mean score | Pass^2 | Pass@2 | steps | max-steps | refetch/ep | blocked/ep | peak prompt tok | compactions/ep | compressor tok/ep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| compass_v2 | 168 | 65.5% | 0.86 | - | - | 25.3 | 12% | 3.5 | 1.3 | 6860 | 3.4 | 26218 |
| full | 168 | 81.0% | 0.95 | - | - | 19.9 | 0% | 1.0 | 0.8 | 12058 | 0.0 | 0 |
| openclaw | 168 | 69.0% | 0.89 | - | - | 26.1 | 8% | 5.2 | 1.6 | 6986 | 4.1 | 33619 |