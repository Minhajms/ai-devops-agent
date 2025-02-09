import os
import sys
import logging
import time
from defusedxml import ElementTree as safe_ET
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import pandas as pd
from typing import Optional, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SlackManager:
    def __init__(self):
        self.token = os.getenv("SLACK_TOKEN")
        if not self.token:
            raise ValueError("SLACK_TOKEN environment variable not set")
            
        self.client = WebClient(token=self.token)
        self.channel = "#devops-alerts"
        self.max_retries = 3

    def _ensure_channel_exists(self):
        """Create channel if it doesn't exist"""
        try:
            # Try to find the channel
            channel_name = self.channel.lstrip('#')
            
            try:
                # First try to join the channel
                self.client.conversations_join(channel=self.channel)
                return True
            except SlackApiError as e:
                if e.response['error'] == 'name_taken':
                    logger.info(f"Channel {self.channel} already exists.")
                    return True
                elif e.response['error'] == 'channel_not_found':
                    # Create the channel if it doesn't exist
                    self.client.conversations_create(
                        name=channel_name,
                        is_private=False
                    )
                    return True
                else:
                    raise e
                    
        except SlackApiError as e:
            logger.error(f"Channel setup failed: {str(e)}")
            # Fall back to general channel
            self.channel = "#general"
            return False

    def send_notification(self, status: str, data: Optional[Dict[str, Any]] = None) -> bool:
        try:
            # Ensure channel exists before sending
            if not self._ensure_channel_exists():
                logger.error("Failed to set up Slack channel")
                return False
            
            blocks = self._create_message_blocks(status, data)
            
            # Add risk prediction if available
            try:
                if os.path.exists('data/risk_predictions.csv'):
                    risk_data = pd.read_csv('data/risk_predictions.csv')
                    if not risk_data.empty:
                        latest_risk = risk_data.iloc[-1]
                        if latest_risk['failure_probability'] > 0.5:
                            blocks.append({
                                "type": "section",
                                "text": {
                                    "type": "mrkdwn",
                                    "text": f"*⚠️ High Risk Alert*\nFailure Probability: {latest_risk['failure_probability']:.1%}\nSuggested Action: {latest_risk['ai_suggestion']}"
                                }
                            })
            except Exception as e:
                logger.warning(f"Risk prediction inclusion failed: {str(e)}")
            
            # Send message with retries
            for attempt in range(self.max_retries):
                try:
                    response = self.client.chat_postMessage(
                        channel=self.channel,
                        blocks=blocks,
                        text=f"Pipeline {status}"  # Fallback text
                    )
                    logger.info(f"Notification sent successfully: {status}")
                    return True
                except SlackApiError as e:
                    if e.response['error'] == 'channel_not_found':
                        logger.error(f"Channel not found: {self.channel}")
                        break
                    if attempt == self.max_retries - 1:
                        raise e
                    logger.warning(f"Retry {attempt + 1}/{self.max_retries} failed: {str(e)}")
                    time.sleep(2 ** attempt)
            
        except SlackApiError as e:
            logger.error(f"Slack API error: {str(e)}")
            return False

    def _create_message_blocks(self, status: str, data: Optional[Dict[str, Any]] = None) -> list:
        status = status.lower()
        color = ":white_check_mark:" if status == "success" else ":x:"
        
        base_blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"Pipeline {status.upper()} {color}",
                    "emoji": True
                }
            }
        ]
        
        if data:
            metrics_block = {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Duration:*\n{data.get('duration', 'N/A')}s"},
                    {"type": "mrkdwn", "text": f"*Tests Run:*\n{data.get('tests', 0)}"},
                    {"type": "mrkdwn", "text": f"*Failures:*\n{data.get('failures', 0)}"},
                    {"type": "mrkdwn", "text": f"*Coverage:*\n{data.get('coverage', 'N/A')}%"}
                ]
            }
            base_blocks.append(metrics_block)
            
        return base_blocks

def main():
    try:
        status = sys.argv[1] if len(sys.argv) > 1 else "unknown"
        
        # Parse actual test results using secure XML parsing
        data = {}
        try:
            if os.path.exists('test-results.xml'):
                tree = safe_ET.parse('test-results.xml')
                root = tree.getroot()
                data = {
                    'tests': root.attrib.get('tests', 0),
                    'failures': root.attrib.get('failures', 0),
                    'errors': root.attrib.get('errors', 0),
                    'duration': root.attrib.get('time', 0)
                }
        except Exception as e:
            logger.warning(f"Test metrics gathering failed: {str(e)}")
        
        notifier = SlackManager()
        success = notifier.send_notification(status, data)
        
        sys.exit(0 if success else 1)
            
    except Exception as e:
        logger.error(f"Notification failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
