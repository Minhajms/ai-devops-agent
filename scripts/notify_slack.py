import os
import logging
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def send_slack_message(message):
    logger.info("Sending Slack message...")
    client = WebClient(token=os.environ.get("SLACK_TOKEN"))
    
    try:
        response = client.chat_postMessage(channel="#devops", text=message)
        logger.info(f"Message sent: {response['message']['text']}")
    except SlackApiError as e:
        logger.error(f"Error sending message to Slack: {e.response['error']}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    send_slack_message("Build failed! Check the logs.")

