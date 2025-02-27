from web3 import Web3
from dotenv import load_dotenv
import os
import json
import requests
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

blacklisted_wallets = [
    "0xdead000000000000000042069420694206942069",
    "0x0000000000000000000000000000000000000000",
]


with open("onchain/abis/ERC20.json", "r") as f:
    ERC20_ABI = json.load(f)

SHIB_INU = "0x95aD61b0a150d79219dCF64E1E6Cc01f0B64C4cE"


def get_web3() -> Web3:
    """Returns a Web3 instance connected to the Ethereum network via Alchemy"""
    return Web3(Web3.HTTPProvider(os.getenv("ALCHEMY_ETHEREUM_RPC_URL")))


def get_token_name(token_address: str) -> str:
    """Returns the name of a token from its address"""
    web3 = get_web3()
    contract = web3.eth.contract(address=token_address, abi=ERC20_ABI)
    return contract.functions.name().call()


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
    Get top token holders for a given token address using BitQuery API,
    excluding contract addresses

    Args:
        token_address: Ethereum token contract address
        limit: Number of top holders to return (default 10)

    Returns:
        dict: Response containing token holder data for non-contract addresses
    """
    access_token = get_bitquery_access_token()
    url = "https://streaming.bitquery.io/graphql"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}",
    }

    # Request more holders than needed since we'll filter out contracts
    adjusted_limit = limit * 2

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
            % (datetime.now().strftime("%Y-%m-%d"), token_address, adjusted_limit)
        }
    )

    try:
        response = requests.post(url, headers=headers, data=payload)
        holders = response.json()["data"]["EVM"]["TokenHolders"]
        filtered_holders = [
            {
                "balance": float(h["Balance"]["Amount"]),
                "address": h["Holder"]["Address"],
            }
            for h in holders
            if not is_contract_address(h["Holder"]["Address"])
        ]
        return filtered_holders[:limit]  # Return only up to the requested limit
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None


def assert_token_address(token_address: str):
    web3 = get_web3()
    assert web3.isAddress(token_address), "Invalid token address"


def send_telegram_message(message: str) -> None:
    """Send a message via Telegram bot"""
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        raise ValueError(
            "Telegram bot token or chat ID not found in environment variables"
        )

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    # Split message into chunks if it's too long (Telegram has a 4096 character limit)
    max_length = 4096
    messages = [message[i : i + max_length] for i in range(0, len(message), max_length)]

    for msg in messages:
        try:
            payload = {"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}
            response = requests.post(url, json=payload)
            response.raise_for_status()
        except Exception as e:
            print(f"Failed to send Telegram message: {e}")
