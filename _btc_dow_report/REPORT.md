# BTC DOW × Sessions Report (UTC)

Source CSV: `/home/jupiter/PYTHON/MARKET_DATA/_crypto_data/BTCUSDT/BTCUSDT_1h.csv`

Sessions (UTC): Asia 00–07, EU 08–15, US 16–23

## Daily by DOW

|   dow | dow_name   |   days |   mean_day_ret_pct |   median_day_ret_pct |   winrate_day_pct |   std_day_ret_pct |   mean_day_range_pct |   median_day_range_pct |
|------:|:-----------|-------:|-------------------:|---------------------:|------------------:|------------------:|---------------------:|-----------------------:|
|     0 | Mon        |    314 |           0.410692 |            0.18821   |           52.2293 |           3.91166 |              5.81044 |                4.68906 |
|     1 | Tue        |    313 |           0.131207 |            0.0887381 |           51.7572 |           2.97729 |              5.068   |                4.30914 |
|     2 | Wed        |    314 |           0.514592 |            0.191752  |           52.5478 |           3.51553 |              5.42931 |                4.57125 |
|     3 | Thu        |    314 |          -0.133678 |           -0.173213  |           46.1783 |           4.01422 |              5.46546 |                4.22476 |
|     4 | Fri        |    314 |           0.105812 |            0.04933   |           50.6369 |           3.32378 |              5.36827 |                4.31883 |

## Sessions by DOW

|   dow | dow_name   | session   |   days |   mean_session_ret_pct |   median_session_ret_pct |   winrate_session_pct |   std_session_ret_pct |
|------:|:-----------|:----------|-------:|-----------------------:|-------------------------:|----------------------:|----------------------:|
|     0 | Mon        | Asia      |    314 |            0.0608455   |               0.093983   |               53.5032 |               2.1848  |
|     0 | Mon        | EU        |    314 |            0.124261    |               0.00581843 |               50.3185 |               2.08833 |
|     0 | Mon        | US        |    314 |            0.217168    |               0.0734298  |               52.8662 |               2.08882 |
|     1 | Tue        | Asia      |    313 |            0.00482802  |              -0.00396795 |               49.5208 |               1.61055 |
|     1 | Tue        | EU        |    313 |           -0.0179312   |               0.0224132  |               51.1182 |               1.97522 |
|     1 | Tue        | US        |    313 |            0.146393    |               0.148524   |               56.869  |               1.69023 |
|     2 | Wed        | Asia      |    314 |            0.10457     |               0.0266911  |               51.2739 |               1.55209 |
|     2 | Wed        | EU        |    314 |            0.236341    |               0.0963469  |               51.5924 |               2.00021 |
|     2 | Wed        | US        |    314 |            0.168021    |               0.10026    |               54.4586 |               2.21866 |
|     3 | Thu        | Asia      |    314 |           -0.00310022  |              -0.0426924  |               47.7707 |               1.5768  |
|     3 | Thu        | EU        |    314 |            0.000440895 |              -0.143496   |               46.1783 |               2.2197  |
|     3 | Thu        | US        |    314 |           -0.158162    |              -0.0260065  |               47.7707 |               2.18145 |
|     4 | Fri        | Asia      |    314 |           -0.10455     |              -0.098124   |               45.2229 |               1.76257 |
|     4 | Fri        | EU        |    314 |            0.151158    |               0.0559187  |               51.9108 |               2.12708 |
|     4 | Fri        | US        |    314 |            0.057887    |               0.0853242  |               54.1401 |               1.77229 |

## Session conditionals by DOW

|   dow | dow_name   | relation               |   prob_pct |   N |
|------:|:-----------|:-----------------------|-----------:|----:|
|     0 | Mon        | P(EU bull | Asia bull) |    50.5952 | 168 |
|     0 | Mon        | P(EU bull | Asia bear) |    50      | 146 |
|     0 | Mon        | P(US bull | EU bull)   |    53.7975 | 158 |
|     0 | Mon        | P(US bull | EU bear)   |    51.9231 | 156 |
|     1 | Tue        | P(EU bull | Asia bull) |    46.4516 | 155 |
|     1 | Tue        | P(EU bull | Asia bear) |    55.6962 | 158 |
|     1 | Tue        | P(US bull | EU bull)   |    60      | 160 |
|     1 | Tue        | P(US bull | EU bear)   |    53.5948 | 153 |
|     2 | Wed        | P(EU bull | Asia bull) |    54.0373 | 161 |
|     2 | Wed        | P(EU bull | Asia bear) |    49.0196 | 153 |
|     2 | Wed        | P(US bull | EU bull)   |    48.1481 | 162 |
|     2 | Wed        | P(US bull | EU bear)   |    61.1842 | 152 |
|     3 | Thu        | P(EU bull | Asia bull) |    49.3333 | 150 |
|     3 | Thu        | P(EU bull | Asia bear) |    43.2927 | 164 |
|     3 | Thu        | P(US bull | EU bull)   |    49.6552 | 145 |
|     3 | Thu        | P(US bull | EU bear)   |    46.1538 | 169 |
|     4 | Fri        | P(EU bull | Asia bull) |    52.1127 | 142 |
|     4 | Fri        | P(EU bull | Asia bear) |    51.7442 | 172 |
|     4 | Fri        | P(US bull | EU bull)   |    50.9202 | 163 |
|     4 | Fri        | P(US bull | EU bear)   |    57.6159 | 151 |

## Day-to-day conditionals by DOW

|   dow | dow_name   | relation                       |   prob_pct |   N |
|------:|:-----------|:-------------------------------|-----------:|----:|
|     0 | Mon        | P(today bull | yesterday bull) |    51.5723 | 159 |
|     0 | Mon        | P(today bull | yesterday bear) |    52.9032 | 155 |
|     1 | Tue        | P(today bull | yesterday bull) |    44.1718 | 163 |
|     1 | Tue        | P(today bull | yesterday bear) |    60      | 150 |
|     2 | Wed        | P(today bull | yesterday bull) |    45.0617 | 162 |
|     2 | Wed        | P(today bull | yesterday bear) |    60.2649 | 151 |
|     3 | Thu        | P(today bull | yesterday bull) |    41.8182 | 165 |
|     3 | Thu        | P(today bull | yesterday bear) |    51.0067 | 149 |
|     4 | Fri        | P(today bull | yesterday bull) |    46.8966 | 145 |
|     4 | Fri        | P(today bull | yesterday bear) |    53.8462 | 169 |

## Top 5 hours by mean hourly return (per DOW)

|   dow | dow_name   |   rank |   hour_utc |   mean_ret_pct |
|------:|:-----------|-------:|-----------:|---------------:|
|     0 | Mon        |      1 |         22 |      0.115064  |
|     0 | Mon        |      2 |          7 |      0.0827751 |
|     0 | Mon        |      3 |         12 |      0.0715566 |
|     0 | Mon        |      4 |         20 |      0.0644755 |
|     0 | Mon        |      5 |         15 |      0.0608073 |
|     1 | Tue        |      1 |         22 |      0.0511434 |
|     1 | Tue        |      2 |         16 |      0.049056  |
|     1 | Tue        |      3 |         10 |      0.0396616 |
|     1 | Tue        |      4 |         20 |      0.035519  |
|     1 | Tue        |      5 |          6 |      0.0329164 |
|     2 | Wed        |      1 |         13 |      0.108896  |
|     2 | Wed        |      2 |         16 |      0.0595761 |
|     2 | Wed        |      3 |         10 |      0.0570103 |
|     2 | Wed        |      4 |          8 |      0.0553344 |
|     2 | Wed        |      5 |          6 |      0.0506672 |
|     3 | Thu        |      1 |          0 |      0.0799086 |
|     3 | Thu        |      2 |         11 |      0.0483778 |
|     3 | Thu        |      3 |         21 |      0.0441818 |
|     3 | Thu        |      4 |         14 |      0.0372323 |
|     3 | Thu        |      5 |         22 |      0.0369299 |
|     4 | Fri        |      1 |         21 |      0.0745393 |
|     4 | Fri        |      2 |          8 |      0.0741874 |
|     4 | Fri        |      3 |          9 |      0.0656722 |
|     4 | Fri        |      4 |         19 |      0.0452486 |
|     4 | Fri        |      5 |          3 |      0.0452414 |
