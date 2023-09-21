# project alpha-discovery-workflow

    Attempts to disprove the trading strategy through introducing counter examples.
    Needs some improvements on realistic backtesting.
    Features and results doesn't work but the workflow in general might be useful.


# Steps to install some dependencies

    virtualenv -p python3.9 env && source env/bin/activate
    pip install -r requirements.txt
    pip install "git+https://github.com/richmanbtc/crypto_data_fetcher.git@v0.0.18#egg=crypto_data_fetcher"
    pip install TA-lib --no-binary TA-lib
    

# Feature generation

    python generate_features.py btcusdt


# Resources and references

    - https://twitter.com/lopezdeprado/status/1148534576874184704
    - https://twitter.com/lopezdeprado/status/1270131167140945924?lang=en
    - https://artifact-research.com/prob-prog/probabilistic-programming-in-finance-a-robust-sharpe-ratio-estimate/
    - https://artifact-research.com/edu/preview/probabilistic-sharpe-ratio/
    - https://forum.numer.ai/t/probabilistic-sharpe-ratio/446
    - https://portfoliooptimizer.io/blog/the-probabilistic-sharpe-ratio-bias-adjustment-confidence-intervals-hypothesis-testing-and-minimum-track-record-length/
    - https://portfoliooptimizer.io/blog/the-probabilistic-sharpe-ratio-hypothesis-testing-and-minimum-track-record-length-for-the-difference-of-sharpe-ratios/
    - http://datagrid.lbl.gov/backtest/backtest.php
    - https://stefan-jansen.github.io/machine-learning-for-trading/08_ml4t_workflow/
    - https://github.com/stefan-jansen/machine-learning-for-trading/tree/main/08_ml4t_workflow
    - https://quantdare.com/probabilistic-sharpe-ratio/
    - https://quantdare.com/deflated-sharpe-ratio-how-to-avoid-been-fooled-by-randomness/
    - https://gmarti.gitlab.io/qfin/2018/05/30/deflated-sharpe-ratio.html
    - https://github.com/rubenbriones/Probabilistic-Sharpe-Ratio
    - https://mathinvestor.org/2022/01/how-backtest-overfitting-in-finance-leads-to-false-discoveries/
    - https://mathinvestor.org/2022/02/backtest-overfitting-and-the-post-hoc-probability-fallacy/
    - https://github.com/esvhd/pypbo
