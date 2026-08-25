# Softmask Fake-Factor Validation

## S1 - Validate the Fake Factor Inputs
The signed fake-factor weights assign positive weights to data and negative weights to simulation so that summing them subtracts the simulated non-fake contribution, while the fake factor is the ratio of ID to anti-ID fake yields used to map the observed anti-ID fake yield into a prediction of the fake-tau yield in the ID region.
inclusive FF : 0.0753  published 0.0754   deviation -0.07%

## S2 - Softmask GNN Benchmark
The softmask GNN benchmark comparisons for folds 0, 1, and 2 can be seen below:

**Fold 0:**
 --- mbL ---
                   bin    actual      binned         GNN
         20-97             4,667       1.004       1.038
         97-116            4,669       0.924       0.949
        116-136            4,137       0.955       0.980
        136-191            2,885       1.040       1.009
        191-2923             940       1.296       1.022
         RMS deviation                 0.139       0.032

  --- T1 ---
                   bin    actual      binned         GNN
        -25--12            4,917       0.935       0.985
        -12--9             4,528       0.979       1.034
         -9-9              4,136       0.961       0.949
          9-13             2,798       1.072       1.012
         13-25               919       1.269       1.002
         RMS deviation                 0.130       0.029

  --- mTtau0 ---
                   bin    actual      binned         GNN
          0-20             3,695       0.898       1.037
         20-39             3,902       0.920       0.956
         39-60             4,007       0.987       0.953
         60-90             3,646       1.102       1.021
         90-1943           2,048       1.117       1.027
         RMS deviation                 0.091       0.037

  --- MET ---
                   bin    actual      binned         GNN
          0-29             4,080       0.975       0.967
         29-45             4,251       0.968       0.959
         45-63             4,008       0.971       0.978
         63-90             3,176       1.000       1.026
         90-2281           1,784       1.130       1.123
         RMS deviation                 0.062       0.062

  --- HT ---
                   bin    actual      binned         GNN
        128-301            4,044       1.091       1.014
        301-379            4,870       0.965       0.945
        379-477            4,026       0.958       0.994
        477-649            2,882       0.979       1.055
        649-7133           1,476       0.937       0.990
         RMS deviation                 0.056       0.036

  --- mBB ---
                   bin    actual      binned         GNN
        150-175            3,998       0.996       0.965
        175-208            3,675       1.051       1.034
        208-254            3,769       0.970       0.969
        254-338            3,300       1.008       1.043
        338-7354           2,556       0.917       0.959
         RMS deviation                 0.046       0.037

  --- dRTauLep ---
                   bin    actual      binned         GNN
          0-2              2,573       0.940       0.998
          2-2              3,762       0.988       1.015
          2-3              3,624       1.002       1.009
          3-3              3,480       1.000       0.991
          3-6              3,859       1.018       0.963
         RMS deviation                 0.029       0.019

  --- mTauTauVis ---
                   bin    actual      binned         GNN
         30-70             4,243       0.962       0.991
         70-94             4,419       0.974       0.979
         94-125            3,766       1.033       1.026
        125-180            3,122       0.966       0.949
        180-3584           1,749       1.075       1.057
         RMS deviation                 0.045       0.038

========================================================================
VERDICT over 8 unused variables: GNN flatter in 8, binned flatter in 0
========================================================================

**Fold 1:**
  --- mbL ---
                   bin    actual      binned         GNN
         19-97             4,776       0.959       0.967
         97-116            4,580       0.953       0.973
        116-136            4,195       0.948       0.984
        136-192            2,818       1.056       1.039
        192-3526           1,078       1.101       0.880
         RMS deviation                 0.063       0.060

  --- T1 ---
                   bin    actual      binned         GNN
        -27--12            4,814       0.957       0.980
        -12--9             4,760       0.929       0.967
         -9-9              3,890       1.023       1.026
          9-13             2,982       0.986       0.960
         13-26             1,003       1.135       0.904
         RMS deviation                 0.072       0.051

  --- mTtau0 ---
                   bin    actual      binned         GNN
          0-20             3,680       0.888       1.020
         20-40             3,863       0.919       0.968
         40-61             4,069       0.966       0.944
         61-90             3,779       1.058       0.972
         90-1843           2,057       1.136       1.009
         RMS deviation                 0.092       0.033

  --- MET ---
                   bin    actual      binned         GNN
          0-29             3,949       1.003       1.000
         29-45             4,267       0.963       0.950
         45-63             3,977       0.979       0.979
         63-90             3,360       0.931       0.958
         90-1566           1,895       1.051       1.036
         RMS deviation                 0.043       0.035

  --- HT ---
                   bin    actual      binned         GNN
        130-301            4,273       1.036       0.932
        301-379            4,653       1.011       0.972
        379-477            4,155       0.919       0.956
        477-649            2,948       0.944       1.051
        649-9411           1,419       0.953       1.060
         RMS deviation                 0.052       0.052

  --- mBB ---
                   bin    actual      binned         GNN
        150-175            3,821       1.030       0.972
        175-208            3,871       0.998       0.969
        208-254            3,661       1.000       1.001
        254-338            3,511       0.937       0.972
        338-6655           2,585       0.903       0.982
         RMS deviation                 0.054       0.024

  --- dRTauLep ---
                   bin    actual      binned         GNN
          0-2              2,784       0.870       0.908
          2-2              3,907       0.954       0.964
          2-3              3,570       1.010       1.002
          3-3              3,327       1.040       1.027
          3-6              3,861       1.003       0.983
         RMS deviation                 0.064       0.046

  --- mTauTauVis ---
                   bin    actual      binned         GNN
         30-70             4,445       0.899       0.913
         70-94             4,467       0.963       0.949
         94-126            3,969       0.973       0.957
        126-181            2,862       1.063       1.057
        181-3637           1,705       1.105       1.151
         RMS deviation                 0.074       0.087

========================================================================
VERDICT over 8 unused variables: GNN flatter in 6, binned flatter in 2
========================================================================

**Fold 2:**
--- mbL ---
                   bin    actual      binned         GNN
         18-97             4,529       1.049       1.055
         97-116            4,426       1.004       1.036
        116-136            4,306       0.946       0.983
        136-192            2,876       1.069       1.038
        192-4234             956       1.300       1.019
         RMS deviation                 0.141       0.036

  --- T1 ---
                   bin    actual      binned         GNN
        -25--12            4,701       1.000       1.044
        -12--9             4,507       1.006       1.046
         -9-9              4,098       0.997       0.993
          9-13             2,891       1.064       1.015
         13-27               897       1.329       1.038
         RMS deviation                 0.150       0.034

  --- mTtau0 ---
                   bin    actual      binned         GNN
          0-20             3,840       0.873       0.982
         20-40             3,696       0.994       1.030
         40-60             3,941       1.026       1.004
         60-90             3,661       1.120       1.049
         90-1762           1,956       1.237       1.116
         RMS deviation                 0.132       0.059

  --- MET ---
                   bin    actual      binned         GNN
          0-29             4,041       1.014       0.977
         29-45             4,201       0.992       0.987
         45-63             3,958       1.010       1.019
         63-90             3,082       1.055       1.100
         90-1388           1,812       1.145       1.125
         RMS deviation                 0.070       0.073

  --- HT ---
                   bin    actual      binned         GNN
        129-301            4,126       1.096       1.019
        301-379            4,689       1.027       1.015
        379-477            3,911       1.011       1.055
        477-649            2,969       0.969       1.031
        649-6484           1,398       1.016       1.003
         RMS deviation                 0.048       0.030

  --- mBB ---
                   bin    actual      binned         GNN
        150-175            3,868       1.054       1.013
        175-208            3,708       1.064       1.043
        208-254            3,480       1.082       1.089
        254-338            3,486       0.979       1.008
        338-6424           2,551       0.934       0.967
         RMS deviation                 0.061       0.047

  --- dRTauLep ---
                   bin    actual      binned         GNN
          0-2              2,630       0.944       0.979
          2-2              3,921       0.965       0.979
          2-3              3,467       1.070       1.077
          3-3              3,300       1.077       1.074
          3-6              3,775       1.074       1.023
         RMS deviation                 0.064       0.050

  --- mTauTauVis ---
                   bin    actual      binned         GNN
         30-70             4,204       0.974       0.993
         70-94             4,318       1.027       1.027
         94-126            3,740       1.061       1.054
        126-181            2,933       1.075       1.065
        181-2810           1,898       1.019       0.991
         RMS deviation                 0.047       0.040

========================================================================
VERDICT over 8 unused variables: GNN flatter in 7, binned flatter in 1
========================================================================

## S3 - Softmask Closure Check
This step compares the Softmask subtraction to the Signed Data-MC subtraction in bins of mTtau0, mbL, and MET. We need to answer the question:
"Does the new postive-weight softmask reproduce the fake distribution that the original signed subtraction gave us?"
For this, a ratio of soft-to-signed means perfect closure, or perfect recreation. Ideally, the ration should be consistent with 1.0 within statistical uncertainty


## S4 - Validation-Loss Stability

## S5 - Random-Seed Stability
