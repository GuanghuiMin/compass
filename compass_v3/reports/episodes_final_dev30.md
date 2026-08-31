# End-to-end results — `artifacts\final_r1` + `artifacts\final_r2`

| method | n | Acc | mean score | Pass^2 | Pass@2 | steps | max-steps | refetch/ep | blocked/ep | peak prompt tok | compactions/ep | compressor tok/ep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| compass_v2 | 30 | 43.3% | 0.84 | 33.3% | 66.7% | 18.3 | 0% | 1.9 | 2.3 | 7141 | 1.9 | 4737 |
| full | 30 | 66.7% | 0.90 | 50.0% | 86.7% | 17.8 | 0% | 1.9 | 1.8 | 9707 | 0.0 | 0 |
| openclaw | 30 | 43.3% | 0.82 | 23.3% | 60.0% | 21.1 | 0% | 2.8 | 2.9 | 6880 | 2.1 | 12993 |