# univ4-fee-oracle
This code can be used by an oracle or keeper (or other design for offchain computation) to calculate the fees owed by a swap based on the current liquidity profile of the pool. Note that the liquidity profile is composed of liquidity at different tick ranges and with different custom fee tiers for each LP position.

# Setup
`pip install -r requirements.txt`

# Run
`python swap.py`
