from math import log, floor
from typing import Tuple
import pandas as pd
from collections import defaultdict


def get_liquidity_data(path='sample_data/all_pos.csv'):
    df = pd.read_csv(path)
    df['fee_tier'] = df['fee_tier']/1000000
    return df


def agg_liquidity_data(df):
    df = df.groupby(['lower_tick', 'upper_tick', 'fee_tier']).sum('liquidity')
    df.reset_index(inplace=True)
    return df


def get_liquidity_at_each_tick(df):
    # Create a dictionary to accumulate liquidity at each tick
    tick_liquidity = defaultdict(int)

    # Loop through each row in the dataframe
    for _, row in df.iterrows():
        for tick in range(int(row['lower_tick']), int(row['upper_tick']) + 1):
            tick_liquidity[tick] += row['liquidity']

    # Convert the dictionary into a sorted DataFrame
    liquidity_df = pd.DataFrame(
        list(tick_liquidity.items()),
        columns=['tick', 'total_liquidity']
    ).sort_values(by='tick')

    return liquidity_df


def get_liquidity_for_each_fee_tier(df):
    tick_fee_liquidity = defaultdict(lambda: defaultdict(int))

    # Loop through each row in the dataframe
    for _, row in df.iterrows():
        for tick in range(int(row['lower_tick']), int(row['upper_tick']) + 1): # upper_tick is inclusive
            tick_fee_liquidity[tick][row['fee_tier']] += row['liquidity']

    # Convert the nested dictionaries into a DataFrame
    rows = [(tick, fee_tier, liquidity) for tick, fees in tick_fee_liquidity.items() for fee_tier, liquidity in fees.items()]
    liquidity_df = pd.DataFrame(rows, columns=['tick', 'fee_tier', 'total_liquidity']).sort_values(by=['tick', 'fee_tier'])
    return liquidity_df.sort_values(['tick', 'fee_tier'])


def tick_by_tick_spacing(precise_tick: float, tick_spacing: int = 1) -> int:
    """
    Rounds down the precise tick to the nearest tick that is a multiple
    of the tick spacing.

    Parameters
    ----------
    precise_tick : float
        The precise tick to be rounded down.

    Returns
    -------
    int
        The corresponding tick, as per the pool's tick spacing.
    """
    tick_step = int(precise_tick/tick_spacing)
    if precise_tick >= 0:
        tick = tick_step * tick_spacing
    else:
        tick = (tick_step - 1) * tick_spacing
    return tick


def tick_to_sqrt_price(tick: int, tick_spacing: int) -> Tuple[float, float]:
    """
    Determines nearest lower and upper ticks as per tick spacing and
    returns the corresponding sqrt prices.

    Parameters
    ----------
    tick : int
        The tick to convert.

    Returns
    -------
    float
        The sqrt price at the lower bound of the nearest lower tick
    float
        The sqrt price at upper bound of the nearest upper tick
    """
    lower_bound_tick = tick_by_tick_spacing(tick, tick_spacing)
    upper_bound_tick = lower_bound_tick + tick_spacing
    # Lower bound of lower tick
    lower_sqrt_price = 1.0001 ** (lower_bound_tick/2)
    # Upper bound of upper tick
    upper_sqrt_price = 1.0001 ** ((upper_bound_tick+1)/2)
    return lower_sqrt_price, upper_sqrt_price


def sqrt_price_to_tick(sqrt_price: float) -> Tuple[int, int]:
    """
    Returns closest tick and closest possible tick as per the pool's
    defined tick spacing.

    Parameters
    ----------
    price : float
        The price to convert.

    Returns
    -------
    int
        The closest precise associated tick.
    """
    log_sqrt_price = log(sqrt_price, 1.0001**(1/2))
    precise_tick = floor(log_sqrt_price)
    tick = tick_by_tick_spacing(precise_tick)
    return precise_tick, tick


def swap_x_for_y(
    liquidity_data,
    curr_price,
    tick_spacing,
    amount_in: float
) -> float:
    """
    Swaps token X for token Y in the Uniswap pool.

    Parameters
    ----------
    amount_in : float
        The amount of token X to be swapped in (after deducting fees).
    sqrt_price_limit : float
        The limit of the square root price for the swap.

    Returns
    -------
    float
        The amount of token Y swapped out.
    """

    amount_remaining = amount_in
    fee = 0
    curr_tick = sqrt_price_to_tick(curr_price)[0]
    lower_tick_sqrt_price, _ = tick_to_sqrt_price(32190, 1)

    while amount_remaining > 0:
        tiers_within_tick = liquidity_data[liquidity_data['tick'] == curr_tick]
        len_tiers_within_tick = len(tiers_within_tick)
        for i in range(len_tiers_within_tick):
            liquidity = tiers_within_tick['total_liquidity'].values[i]
            fee_tier = tiers_within_tick['fee_tier'].values[i]

            # delta(1/sqrt(P)) = delta(x) / L
            delta_inv_sqrt_price = (1/lower_tick_sqrt_price) - (1/curr_price)
            amount_used = delta_inv_sqrt_price * liquidity
            amount_used_w_fee = amount_used/(1-fee_tier)
            if amount_used_w_fee > amount_remaining:
                fee += amount_remaining * fee_tier
                amount_remaining = 0
                lower_tick_sqrt_price
                return fee
            else:
                fee += amount_used_w_fee - amount_used
                amount_remaining = amount_remaining - amount_used_w_fee
        curr_price = lower_tick_sqrt_price
        curr_tick = curr_tick - 1
        lower_tick_sqrt_price, _ = tick_to_sqrt_price(curr_tick, 1)


if __name__ == "__main__":
    df = get_liquidity_data()
    df = agg_liquidity_data(df)
    liq_df = get_liquidity_at_each_tick(df)
    liq_tick_df = get_liquidity_for_each_fee_tier(df)
    fee = swap_x_for_y(liq_tick_df, 5, 1, 100)
    print(fee)
