from datetime import datetime
import math
from .utils import (
    get_web3,
    ERC20_ABI,
    assert_token_address,
    get_token_holders,
    is_contract_address,
)


def calculate_gini_coefficient(holders: list) -> float:
    """
    Calculate the Gini coefficient for token distribution
    A measure of inequality where 0 = perfect equality and 1 = perfect inequality

    Args:
        holders: List of dictionaries containing holder balances

    Returns:
        float: Gini coefficient between 0 and 1
    """
    if not holders:
        return 0

    # Sort balances in ascending order
    balances = sorted([holder["balance"] for holder in holders])
    n = len(balances)

    # Calculate cumulative sum
    cumsum = [sum(balances[0 : i + 1]) for i in range(n)]
    total = cumsum[-1]

    # Calculate Gini coefficient using the formula
    gini = (
        n
        + 1
        - 2 * sum([(n - i) * balance / total for i, balance in enumerate(balances)])
    ) / n

    return max(0, min(1, gini))  # Ensure result is between 0 and 1


def calculate_nakamoto_coefficient(holders: list, total_supply: float) -> int:
    """
    Calculate the Nakamoto Coefficient - the minimum number of entities required to reach 51% of token supply

    Args:
        holders: List of dictionaries containing holder balances
        total_supply: Total token supply

    Returns:
        int: Nakamoto coefficient (number of holders needed for 51% control)
    """
    if not holders:
        return 0

    cumulative_percentage = 0
    for i, holder in enumerate(
        sorted(holders, key=lambda x: x["balance"], reverse=True)
    ):
        cumulative_percentage += (holder["balance"] / total_supply) * 100
        if cumulative_percentage >= 51:
            return i + 1
    return len(holders)


def calculate_theil_index(holders: list) -> float:
    """
    Calculate the Theil Index - a measure of economic inequality
    The index ranges from 0 (perfect equality) to ln(n) (perfect inequality)

    Args:
        holders: List of dictionaries containing holder balances

    Returns:
        float: Theil index value
    """
    if not holders:
        return 0

    balances = [holder["balance"] for holder in holders]
    n = len(balances)

    if n == 1:
        return 0

    # Calculate mean balance
    mean_balance = sum(balances) / n

    # Calculate Theil index
    theil = 0
    for balance in balances:
        if balance > 0:  # Avoid log(0)
            ratio = balance / mean_balance
            theil += (ratio * math.log(ratio)) / n

    return max(0, theil)  # Ensure non-negative


# potential method to use later. Not using it now since its too slow
def get_new_holders_24h(token_address: str, max_transfers: int = 100) -> int:
    """
    Get the number of new token holders in the last 24 hours using web3.py events

    Args:
        token_address (str): The token contract address
        max_transfers (int, optional): Maximum number of recent transfers to analyze. Defaults to 100.

    Returns:
        int: Number of new holders in last 24 hours
    """
    assert_token_address(token_address)

    web3 = get_web3()

    # Create contract instance
    token_contract = web3.eth.contract(address=token_address, abi=ERC20_ABI)

    # Calculate block from 24 hours ago (assuming 12 seconds per block)
    current_block = web3.eth.block_number
    blocks_per_day = 24 * 60 * 60 // 12  # Approximately 7200 blocks per day
    from_block = current_block - blocks_per_day

    try:
        # Create filter for Transfer events
        transfer_filter = token_contract.events.Transfer().createFilter(
            fromBlock=from_block, toBlock="latest"
        )

        # Get all transfer events
        transfers = transfer_filter.get_all_entries()

        # Track potential new holders and checked addresses
        potential_new_holders = set()
        checked_addresses = set()  # Track addresses we've already checked

        # Get last N transfers (most recent first)
        for transfer in sorted(transfers, key=lambda x: x["blockNumber"], reverse=True)[
            :max_transfers
        ]:
            receiver = transfer["args"]["to"]

            # Skip if we've already checked this address
            if receiver.lower() in checked_addresses:
                continue

            # Skip if receiver is a contract
            if is_contract_address(receiver):
                checked_addresses.add(receiver.lower())  # Mark as checked
                continue

            # Check if this address had any balance before this transfer
            try:
                # Get block number just before this transfer
                prev_block = transfer["blockNumber"] - 1

                # Check balance at previous block
                prev_balance = token_contract.functions.balanceOf(receiver).call(
                    block_identifier=prev_block
                )

                # Mark address as checked
                checked_addresses.add(receiver.lower())

                # If balance was 0 before this transfer, it's a new holder
                if prev_balance == 0:
                    potential_new_holders.add(receiver.lower())

            except Exception as e:
                print(f"Error checking previous balance: {str(e)}")
                continue

        return len(potential_new_holders)

    except Exception as e:
        print(f"Error fetching new holders: {str(e)}")
        return 0


def analyze_token_distribution(token_address: str) -> dict:
    """
    Analyze the token distribution metrics including Gini coefficient and holder concentration

    Args:
        token_address (str): The token contract address to analyze

    Returns:
        dict: A dictionary containing detailed token distribution analysis with the following structure:
            {
                'token_info': {
                    'address': str,  # The token's contract address
                    'symbol': str,   # The token's symbol (e.g., 'SHIB', 'UNI')
                    'total_supply': float  # Total token supply adjusted for decimals
                },
                'distribution_metrics': {
                    'gini_coefficient': float,  # Value between 0-1, where:
                                              # 0 = perfect equality
                                              # 1 = perfect inequality
                                              # Higher values indicate more concentrated ownership

                    'top_holder_concentration': {
                        'top_1_percentage': float,  # Percentage of supply held by largest holder
                        'top_5_percentage': float,  # Percentage of supply held by top 5 holders
                        'top_10_percentage': float  # Percentage of supply held by top 10 holders
                    },

                    'concentration_metrics': {
                        'nakamoto_interpretation': str,  # Interpretation of Nakamoto coefficient
                        'theil_interpretation': str      # Interpretation of Theil index
                    },

                    'holder_statistics': {
                        'total_holders_analyzed': int,  # Number of holders analyzed (max 100)
                        'non_contract_holders': int,    # Number of holders that are EOAs (not contracts)
                    }
                },
                'timestamp': str  # ISO format timestamp of when analysis was performed
            }

            If there's an error fetching data:
            {
                'error': str,  # Error message
                'token_address': str  # The token address that was analyzed
            }

    Example:
        >>> data = analyze_token_distribution("0x95aD61b0a150d79219dCF64E1E6Cc01f0B64C4cE")
        >>> print(data['distribution_metrics']['gini_coefficient'])
        0.8411298486981609
        >>> print(data['distribution_metrics']['top_holder_concentration']['top_1_percentage'])
        41.04368089149804

    Notes:
        - Gini coefficient interpretation:
          * 0.0-0.3: Low concentration
          * 0.3-0.6: Moderate concentration
          * 0.6-0.8: High concentration
          * 0.8-1.0: Very high concentration

        - Top holder concentration interpretation:
          * >50% by top 1: Potential centralization risk
          * >70% by top 10: High centralization

        - Non-contract holders ratio can indicate institutional vs retail ownership
    """
    assert_token_address(token_address)

    web3 = get_web3()

    # Create contract instance
    token_contract = web3.eth.contract(address=token_address, abi=ERC20_ABI)

    # Get token metadata
    decimals = token_contract.functions.decimals().call()
    total_supply = token_contract.functions.totalSupply().call() / (10**decimals)
    token_symbol = token_contract.functions.symbol().call()

    # Get all holders (limited to top 100 for practical purposes)
    holders = get_token_holders(token_address, 100)
    if not holders:
        return {"error": "Failed to fetch holder data", "token_address": token_address}

    # Calculate Gini coefficient
    gini = calculate_gini_coefficient(holders)

    # Calculate top holder concentrations
    top_10_holders = holders[:10] if len(holders) >= 10 else holders
    top_10_balance = sum(holder["balance"] for holder in top_10_holders)
    top_10_percentage = (top_10_balance / total_supply) * 100

    # Calculate other concentration metrics
    top_1_percentage = (holders[0]["balance"] / total_supply * 100) if holders else 0
    top_5_percentage = (
        (sum(h["balance"] for h in holders[:5]) / total_supply * 100)
        if len(holders) >= 5
        else 0
    )

    # Count non-contract holders
    non_contract_holders = sum(
        1 for holder in holders if not is_contract_address(holder["address"])
    )

    # Calculate additional concentration metrics
    nakamoto_coef = calculate_nakamoto_coefficient(holders, total_supply)
    theil_index = calculate_theil_index(holders)

    return {
        "token_info": {
            "address": token_address,
            "symbol": token_symbol,
            "total_supply": total_supply,
        },
        "distribution_metrics": {
            "gini_coefficient": gini,
            "top_holder_concentration": {
                "top_1_percentage": top_1_percentage,
                "top_5_percentage": top_5_percentage,
                "top_10_percentage": top_10_percentage,
            },
            "concentration_metrics": {
                "nakamoto_interpretation": (
                    "High centralization"
                    if nakamoto_coef < 4
                    else (
                        "Moderate centralization"
                        if nakamoto_coef < 10
                        else "Decentralized"
                    )
                ),
                "theil_interpretation": (
                    "High inequality"
                    if theil_index > 1
                    else (
                        "Moderate inequality" if theil_index > 0.5 else "Low inequality"
                    )
                ),
            },
            "holder_statistics": {
                "total_holders_analyzed": len(holders),
                "non_contract_holders": non_contract_holders,
            },
        },
        "timestamp": datetime.now().isoformat(),
    }
