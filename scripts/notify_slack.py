import os
import logging
import sys
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from datetime import datetime
import json

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
        
    def send_message(self, message: str, attachments: list = None):
        """Send a message to Slack with optional attachments."""
        try:
            response = self.client.chat_postMessage(
                channel=self.channel,
                text=message,
                attachments=attachments
            )
            logger.info(f"Message sent: {message}")
            return response
        except SlackApiError as e:
            logger.error(f"Error sending message: {str(e)}")
            raise
            
    def send_build_status(self, status: str, build_info: dict):
        """Send formatted build status message."""
        emoji = "✅" if status == "success" else "❌"
        
        attachments = [{
            "color": "#36a64f" if status == "success" else "#ff0000",
            "fields": [
                {
                    "title": "Build Status",
                    "value": f"{emoji} {status.capitalize()}",
                    "short": True
                },
                {
                    "title": "Build Time",
                    "value": f"{build_info.get('duration', 'N/A')}s",
                    "short": True
                },
                {
                    "title": "Tests",
                    "value": f"{build_info.get('test_count', 'N/A')} tests",
                    "short": True
                }
            ],
            "footer": f"Build ID: {build_info.get('build_id', 'N/A')}"
        }]
        
        return self.send_message(
            f"Build {status.capitalize()}",
            attachments=attachments
        )
