import os
import logging
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import pandas as pd

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

    def _create_message_blocks(self, status, data=None):
        base_blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"Pipeline {status.upper()}",
                    "emoji": True
                }
            }
        ]
        
        if data is not None:
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
            
        if status.lower() != 'success':
            base_blocks.append({
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "View Artifacts"},
                        "url": os.getenv("GITHUB_ARTIFACT_URL")
                    }
                ]
            })
            
        return base_blocks

def send_pipeline_report(self, status, report_data=None):
    try:
        blocks = self._create_message_blocks(status, report_data)
        emoji = ":white_check_mark:" if status.lower() == "success" else ":x:"
        
        # Include AI suggestion in the report data if available
        if report_data and 'ai_suggestion' in report_data:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*AI Suggestion:*\n{report_data['ai_suggestion']}"
                }
            })
        
        response = self.client.chat_postMessage(
            channel=self.channel,
            text=f"Pipeline {status} {emoji}",
            blocks=blocks
        )
        
        logger.info(f"Sent {status} report to Slack")
        return response
        
    except SlackApiError as e:
        logger.error(f"Slack API error: {str(e)}")
        return None

    def handle_chatops_command(self, command):
        try:
            if "status" in command.lower():
                # Get latest predictions
                df = pd.read_csv('data/risk_predictions.csv')
                latest = df.iloc[-1]
                
                message = (
                    f"*Current Pipeline Risk Status*\n"
                    f"Failure Probability: {latest['failure_probability']:.2%}\n"
                    f"Last Suggestion:\n{latest['ai_suggestion']}"
                )
                
                self.client.chat_postMessage(
                    channel=self.channel,
                    text=message,
                    mrkdwn=True
                )
                
            elif "help" in command.lower():
                self.send_help_message()
                
            else:
                self.client.chat_postMessage(
                    channel=self.channel,
                    text="Unknown command. Available commands: `status`, `help`"
                )
                
        except Exception as e:
            logger.error(f"ChatOps handling failed: {str(e)}")

    def send_help_message(self):
        help_text = (
            "*Pipeline Guardian Help*\n"
            "• `status`: Get current pipeline risk status\n"
            "• `help`: Show this help message\n"
            "• `details [failure_type]`: Get detailed failure analysis"
        )
        
        self.client.chat_postMessage(
            channel=self.channel,
            text=help_text,
            mrkdwn=True
        )

if __name__ == "__main__":
    import sys
    command = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "status"
    
    try:
        notifier = SlackManager()
        notifier.handle_chatops_command(command)
    except Exception as e:
        logger.error(f"Slack notification failed: {str(e)}")
        exit(1)
