import os
import json
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

def load_config(config_file="config.json"):
    """
    Loads the Slack configuration from a JSON file.

    Args:
        config_file (str): The path to the configuration file.

    Returns:
        dict: A dictionary containing the Slack configuration.
    """
    try:
        with open(config_file, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Configuration file '{config_file}' not found.")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error decoding JSON in the configuration file '{config_file}'.")
        exit(1)

# Load the configuration
config = load_config()

# Extract the Slack token from the configuration
slack_token = config["slack_token"]

def send_slack_message(token, channel, message):
    """
    Sends a message to a Slack channel.

    Args:
        token (str): The Slack OAuth token.
        channel (str): The ID or name of the Slack channel.
        message (str): The message to send.

    Raises:
        SlackApiError: If the Slack API call fails.
    """
    client = WebClient(token=token)
    
    try:
        # Send the message to the specified Slack channel
        response = client.chat_postMessage(
            channel=channel,
            text=message,
            username="SaaS Alerts"
        )
        print("Message sent successfully.")
    except SlackApiError as e:
        # Handle errors from the Slack API
        print(f"Error sending message: {e.response['error']}")

if __name__ == '__main__':
    """
    Main entry point for the script.

    Sends a predefined message to a specified Slack channel.
    """
    # Ensure the Slack OAuth token is set
    if not slack_token:
        print("Slack token is not set in the configuration file.")
        exit(1)
    
    # Define the Slack channel and message
    channel = "#eventlog"  # Change to your desired channel
    message = """🚨 Unauthorized File Access Detected 🚨

File: **Marketing_Strategy_Q2.xlsx**
File ID: **12tJ8uPlR9dP1dO0sZ8pX3i7N5o4J9k5l**
Access Attempted By: **intruder@unknown.com**
Access Timestamp: **2025-03-25 15:45:59 UTC**"""
    
    # Send the message to Slack
    send_slack_message(slack_token, channel, message)


