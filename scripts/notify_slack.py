import os
import logging
from slack_sdk import WebClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def send_slack_message(message):
    logger.info("Sending Slack message...")
    client = WebClient(token=os.environ["SLACK_TOKEN"])
    client.chat_postMessage(channel="#devops", text=message)

if __name__ == "__main__":
    import sys
    message = sys.argv[1]  # Get the message from the command line
    send_slack_message(message)
