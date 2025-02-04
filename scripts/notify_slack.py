import os
import logging
import sys
from slack_sdk import WebClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def send_slack_message(message):
    logger.info(f"Sending Slack message: {message}")
    client = WebClient(token=os.environ["SLACK_TOKEN"])
    client.chat_postMessage(channel="#devops", text=message)

if __name__ == "__main__":
    message = sys.argv[1]
    send_slack_message(message)
