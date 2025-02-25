from web3 import Web3
from dotenv import load_dotenv
import os
import json
import requests
import json
from datetime import datetime, timedelta
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.gridspec import GridSpec

# Load environment variables from .env file
load_dotenv()

blacklisted_wallets = [
    "0xdead000000000000000042069420694206942069",
    "0x0000000000000000000000000000000000000000",
]


with open("abis/ERC20.json", "r") as f:
    ERC20_ABI = json.load(f)

SHIB_INU = "0x95aD61b0a150d79219dCF64E1E6Cc01f0B64C4cE"


def get_web3() -> Web3:
    """Returns a Web3 instance connected to the Ethereum network via Alchemy"""
    return Web3(Web3.HTTPProvider(os.getenv("ALCHEMY_ETHEREUM_RPC_URL")))


def is_contract_address(address: str) -> bool:
    """
    Check if an address is a contract address

    Args:
        address: Ethereum address to check

    Returns:
        bool: True if contract address, False if EOA (externally owned account)
    """
    web3 = get_web3()
    # First verify it's a valid address
    if not web3.isAddress(address):
        return False

    # Get the code at the address
    code = web3.eth.get_code(web3.toChecksumAddress(address))

    # If there's code at the address, it's a contract
    # If no code (b'0x' or empty bytes), it's an EOA
    return code != b"" and code != b"0x"


def get_bitquery_access_token() -> str:
    url = "https://oauth2.bitquery.io/oauth2/token"

    payload = f'grant_type=client_credentials&client_id={os.getenv("BITQUERY_CLIENT_ID")}&client_secret={os.getenv("BITQUERY_CLIENT_SECRET")}&scope=api'

    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    response = requests.request("POST", url, headers=headers, data=payload)
    resp = json.loads(response.text)

    return resp["access_token"]


def get_token_holders(token_address: str, limit: int = 10) -> dict:
    """
    Get top token holders for a given token address using BitQuery API

    Args:
        token_address: Ethereum token contract address
        limit: Number of top holders to return (default 10)

    Returns:
        dict: Response containing token holder data
    """

    access_token = get_bitquery_access_token()
    url = "https://streaming.bitquery.io/graphql"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}",
    }

    payload = json.dumps(
        {
            "query": """
        {
      EVM(network: eth, dataset: archive) {
        TokenHolders(
          date: "%s"
          tokenSmartContract: "%s"
          limit: {count: %d}
          orderBy: {descendingByField: "Balance_Amount"}
        ) {
          Balance {
            Amount
          }
          Holder {
            Address
          }
        }
      }
    }
        """
            % (datetime.now().strftime("%Y-%m-%d"), token_address, limit)
        }
    )

    try:
        response = requests.post(url, headers=headers, data=payload)
        holders = response.json()["data"]["EVM"]["TokenHolders"]
        return [
            {
                "balance": float(h["Balance"]["Amount"]),
                "address": h["Holder"]["Address"],
            }
            for h in holders
        ]
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None


def assert_token_address(token_address: str):
    web3 = get_web3()
    assert web3.isAddress(token_address), "Invalid token address"
