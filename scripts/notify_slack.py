import os
import logging
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# Enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SlackNotifier:
    def __init__(self):
        self.token = os.environ.get("SLACK_TOKEN")
        if not self.token:
            raise ValueError("SLACK_TOKEN environment variable not set")
        self.client = WebClient(token=self.token)
        self.channel = "#devops"

    def send_message(self, message: str, blocks: list = None):
        """Send a message to Slack with optional blocks."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat_postMessage(
                    channel=self.channel,
                    text=message,
                    blocks=blocks
                )
                logger.info(f"Message sent: {message}")
                logger.info(f"Response: {response.data}")
                return response
            except SlackApiError as e:
                logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise

    def send_build_status(self, status: str, build_info: dict):
        """Send formatted build status message."""
        emoji = "✅" if status == "success" else "❌"
        blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"Build {status.capitalize()} {emoji}"
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Build Time:*\n{build_info.get('duration', 'N/A')}s"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Tests:*\n{build_info.get('test_count', 'N/A')} tests"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Failures:*\n{build_info.get('failure_count', 'N/A')} failures"
                    }
                ]
            }
        ]
        return self.send_message(f"Build {status.capitalize()}", blocks=blocks)
