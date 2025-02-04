import os
import logging
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SlackNotifier:
    def __init__(self):
        # Get the Slack token from environment variables
        self.token = os.environ.get("SLACK_TOKEN")
        if not self.token:
            raise ValueError("SLACK_TOKEN environment variable not set")
        self.client = WebClient(token=self.token)
        self.channel = "#devops"  # Change this to your desired Slack channel
        
    def send_message(self, message: str, blocks: list = None):
        """Send a message to Slack with optional blocks."""
        try:
            response = self.client.chat_postMessage(
                channel=self.channel,
                text=message,
                blocks=blocks
            )
            logger.info(f"Message sent: {message}")
            logger.info(f"Response: {response.data}")  # Log the response for debugging
            return response
        except SlackApiError as e:
            logger.error(f"Error sending message: {str(e)}")
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
                    }
                ]
            },
            {
                "type": "actions",
                "elements": self.add_actions(build_info.get('run_id'))
            }
        ]
        
        return self.send_message(
            f"Build {status.capitalize()}",
            blocks=blocks
        )

    def add_actions(self, run_id):
        """Add action buttons for Slack message."""
        return [
            {
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": "Retry Build"
                },
                "url": f"https://api.github.com/repos/Minhajms/ai-devops-agent/actions/runs/{run_id}/rerun"
            }
        ]
