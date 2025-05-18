import os
from dotenv import load_dotenv
from alpaca_trade_api.rest import REST, APIError
from alpaca_trade_api.common import URL

# Load environment variables from .env file
load_dotenv()

# Get credentials from environment variables
API_KEY = os.environ.get('APCA_API_KEY_ID')
API_SECRET = os.environ.get('APCA_API_SECRET_KEY')
BASE_URL = os.environ.get('APCA_PAPER_URL') # This should be your paper trading URL

# Validate that credentials and URL are loaded
if not API_KEY:
    print("Error: APCA_API_KEY_ID not found in .env file or environment.")
    exit()
if not API_SECRET:
    print("Error: APCA_API_SECRET_KEY not found in .env file or environment.")
    exit()
if not BASE_URL:
    print("Error: APCA_PAPER_URL not found in .env file or environment. Should be 'https://paper-api.alpaca.markets'")
    exit()

# Ensure the BASE_URL is correct for paper trading
if "paper-api.alpaca.markets" not in BASE_URL:
    print(f"Warning: Your BASE_URL is '{BASE_URL}'. Are you sure this is your PAPER trading URL?")
    print("Expected paper URL: 'https://paper-api.alpaca.markets'")
    # You might want to exit if it's not the paper URL to be safe
    # exit()

print("Connecting to Alpaca Paper Trading...")
try:
    # Initialize the Alpaca API client
    # The URL object helps ensure the URL is formatted correctly if it doesn't have http/https
    api = REST(API_KEY, API_SECRET, base_url=URL(BASE_URL), api_version='v2')

    # Check account status (optional, but good for verifying connection)
    account = api.get_account()
    print(f"Account Status: {account.status}")
    print(f"Paper Buying Power: ${account.buying_power}")
    if float(account.buying_power) < 500: # NVDA is expensive, make sure there's some buffer
        print("Warning: Low paper buying power. NVDA is an expensive stock.")

except APIError as e:
    print(f"Error connecting to Alpaca or getting account info: {e}")
    print(f"Full error response: {e._error}") # More detailed error from Alpaca
    exit()
except Exception as e:
    print(f"An unexpected error occurred during API initialization: {e}")
    exit()


# --- Define Order Parameters ---
symbol_to_buy = "NVDA"
quantity_to_buy = 1 # Buying 1 share
order_side = "buy"
order_type = "market" # Market order to get it filled at the current market price
time_in_force = "day" # 'day' means the order is good for the current trading day

print(f"\nAttempting to place a {order_side} {order_type} order for {quantity_to_buy} share(s) of {symbol_to_buy}...")

try:
    # Submit the order
    order = api.submit_order(
        symbol=symbol_to_buy,
        qty=quantity_to_buy,
        side=order_side,
        type=order_type,
        time_in_force=time_in_force
    )

    # Print order details if successful
    print("\n--- Order Submitted Successfully ---")
    print(f"Order ID: {order.id}")
    print(f"Client Order ID: {order.client_order_id}")
    print(f"Symbol: {order.symbol}")
    print(f"Quantity: {order.qty}")
    print(f"Side: {order.side}")
    print(f"Type: {order.type}")
    print(f"Status: {order.status}") # Will likely be 'accepted' or 'new' initially
    print(f"Submitted at: {order.submitted_at}")
    print("\nNote: This is the initial status. Check your Alpaca paper account to see if/when it fills.")

except APIError as e:
    print(f"\n--- Alpaca API Error During Order Submission ---")
    print(f"Error: {e}")
    # The e._error attribute often contains a more specific JSON response from Alpaca
    if hasattr(e, '_error') and e._error:
        print(f"Alpaca Error Details: {e._error}")
    print("Order failed. Please check the error message and your Alpaca paper account.")

except Exception as e:
    print(f"\n--- An Unexpected Error Occurred During Order Submission ---")
    print(f"Error: {e}")
    print("Order failed. Please check the error message.")