from .utils import (
    get_web3,
    ERC20_ABI,
    blacklisted_wallets,
    assert_token_address,
    get_token_holders,
    is_contract_address,
    SHIB_INU,
)


def get_recent_transfers(token_address: str, wallet_address: str, days: int = 7):
    """
    Get the transfers of a wallet for a given token in the last n days

    Args:
        token_address: The token contract address
        wallet_address: The wallet address to get transfers for
        days: The number of days to get transfers for

    Returns:
        list: A list of transfers
    """
    assert_token_address(token_address)
    assert_token_address(wallet_address)

    web3 = get_web3()

    # Create contract instance
    token_contract = web3.eth.contract(address=token_address, abi=ERC20_ABI)

    # Get current block
    current_block = web3.eth.block_number

    # Calculate block from n days ago (assuming 12.07 seconds per block)
    blocks_per_day = 24 * 60 * 60 // 12.07
    from_block = current_block - int(blocks_per_day * days)

    try:
        # Get transfers FROM the wallet
        outgoing_filter = token_contract.events.Transfer.createFilter(
            fromBlock=from_block,
            toBlock="latest",
            argument_filters={"from": wallet_address},
        )
        outgoing_events = outgoing_filter.get_all_entries()

        # Get transfers TO the wallet
        incoming_filter = token_contract.events.Transfer.createFilter(
            fromBlock=from_block,
            toBlock="latest",
            argument_filters={"to": wallet_address},
        )
        incoming_events = incoming_filter.get_all_entries()

        # Combine and format transfers
        all_transfers = []

        for event in outgoing_events + incoming_events:
            transfer = {
                "blockNumber": event["blockNumber"],
                "transactionHash": event["transactionHash"].hex(),
                "from": event["args"]["from"],
                "to": event["args"]["to"],
                "value": event["args"]["value"],
                "type": (
                    "out"
                    if event["args"]["from"].lower() == wallet_address.lower()
                    else "in"
                ),
            }
            all_transfers.append(transfer)

        # Sort by block number (most recent first)
        all_transfers.sort(key=lambda x: x["blockNumber"], reverse=True)

        return all_transfers

    except Exception as e:
        print(f"Error fetching transfers: {str(e)}")
        return []


def monitor_whale_wallets(token_address: str, days: int = 7):
    """
    Monitor whale wallets for a given token, analyzing their balances, transfers, and activity patterns

    Args:
        token_address (str): The token contract address to analyze
        days (int, optional): Number of days to look back for transfer history. Defaults to 7

    Returns:
        dict: A dictionary containing detailed whale monitoring data with the following structure:
            {
                'token_info': {
                    'address': str,         # The token's contract address
                    'symbol': str,          # The token's symbol (e.g., 'SHIB', 'UNI')
                    'total_supply': float,  # Total token supply adjusted for decimals
                    'whale_threshold': float # Minimum balance to be considered a whale (1% of supply)
                },

                'analysis_timeframe': str,  # Description of analysis period (e.g., "Last 7 days")

                'whales': [                 # List of whale wallet data
                    {
                        'address': str,     # Whale's wallet address
                        'current_balance': float,  # Current token balance
                        'percentage_of_supply': float,  # Percentage of total supply held

                        'transfer_activity': {
                            'total_transfers': int,    # Total number of transfers
                            'outgoing_transfers': int, # Number of outgoing transfers
                            'incoming_transfers': int, # Number of incoming transfers
                            'total_outgoing_amount': float,  # Total tokens sent
                            'total_incoming_amount': float,  # Total tokens received
                            'net_flow': float,        # Net token flow (incoming - outgoing)
                        },
                    },
                    ...
                ]
            }

            If there's an error:
            {
                'error': str,  # Error message
                'token_address': str  # The token address that was analyzed
            }

    Example:
        >>> data = monitor_whale_wallets("0x95aD61b0a150d79219dCF64E1E6Cc01f0B64C4cE")
        >>> print(f"Number of whales: {len(data['whales'])}")
        >>> print(f"Largest whale balance: {data['whales'][0]['current_balance']}")
        >>> print(f"Recent activity: {data['whales'][0]['transfer_activity']}")

    Notes:
        - Whale Definition:
          * Minimum 1% of total token supply
          * Must be an EOA (externally owned account), not a contract
          * Not in blacklisted addresses

        - Transfer Activity Analysis:
          * Tracks both incoming and outgoing transfers
          * Calculates net token flow
          * Identifies patterns of accumulation or distribution

        - Important Metrics:
          * percentage_of_supply: Individual whale's market control
          * net_flow: Whether whale is accumulating or distributing
          * transfer_activity: Overall trading behavior

        - Blacklisted addresses include:
          * Dead addresses (0xdead...)
          * Zero address (0x0000...)
          * Known burn addresses
    """
    web3 = get_web3()
    assert_token_address(token_address)
    token_contract = web3.eth.contract(address=token_address, abi=ERC20_ABI)

    # Get token metadata
    decimals = token_contract.functions.decimals().call()
    total_supply = token_contract.functions.totalSupply().call() / (10**decimals)
    token_symbol = token_contract.functions.symbol().call()

    # Define whale threshold (>1% of supply)
    MIN_WHALE_HOLDINGS = total_supply * 0.01

    # Get all token holders
    holders = get_token_holders(token_address, 25)

    # Filter holders and get their data
    whales_data = {
        "token_info": {
            "address": token_address,
            "symbol": token_symbol,
            "total_supply": total_supply,
            "whale_threshold": MIN_WHALE_HOLDINGS,
        },
        "analysis_timeframe": f"Last {days} days",
        "whales": [],
    }

    # Filter and process whale data
    for holder in holders:
        if (
            holder["balance"] > MIN_WHALE_HOLDINGS
            and not is_contract_address(holder["address"])
            and holder["address"] not in blacklisted_wallets
        ):

            # Get recent transfers for this whale
            recent_transfers = get_recent_transfers(
                token_address, holder["address"], days
            )

            # Calculate transfer statistics
            outgoing_transfers = [t for t in recent_transfers if t["type"] == "out"]
            incoming_transfers = [t for t in recent_transfers if t["type"] == "in"]

            total_outgoing = sum(t["value"] for t in outgoing_transfers) / (
                10**decimals
            )
            total_incoming = sum(t["value"] for t in incoming_transfers) / (
                10**decimals
            )

            whale_data = {
                "address": holder["address"],
                "current_balance": holder["balance"],
                "percentage_of_supply": (holder["balance"] / total_supply) * 100,
                "transfer_activity": {
                    "total_transfers": len(recent_transfers),
                    "outgoing_transfers": len(outgoing_transfers),
                    "incoming_transfers": len(incoming_transfers),
                    "total_outgoing_amount": total_outgoing,
                    "total_incoming_amount": total_incoming,
                    "net_flow": total_incoming - total_outgoing,
                },
            }

            whales_data["whales"].append(whale_data)

    # Sort whales by balance
    whales_data["whales"].sort(key=lambda x: x["current_balance"], reverse=True)

    return whales_data


if __name__ == "__main__":
    # Monitor SHIB whale wallets
    shib_whales = monitor_whale_wallets(SHIB_INU)

    # Print summary
    print(shib_whales)
