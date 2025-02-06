import os
import logging
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

    def send_notification(self, status: str, data: Optional[Dict[str, Any]] = None) -> bool:
        try:
            blocks = self._create_message_blocks(status, data)
            
            # Add risk prediction if available
            try:
                risk_data = pd.read_csv('data/risk_predictions.csv')
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
                logger.warning(f"Could not include risk prediction: {str(e)}")
            
            response = self.client.chat_postMessage(
                channel=self.channel,
                blocks=blocks,
                text=f"Pipeline {status}"  # Fallback text
            )
            
            logger.info(f"Notification sent successfully: {status}")
            return True
            
        except SlackApiError as e:
            logger.error(f"Failed to send Slack notification: {str(e)}")
            return False

def main():
    try:
        status = sys.argv[1] if len(sys.argv) > 1 else "unknown"
        
        # Gather pipeline metrics if available
        data = {}
        try:
            with open('test-results.xml') as f:
                # Parse test results for metrics
                # This is a simplified example
                data['tests'] = 10
                data['failures'] = 0
        except Exception:
            logger.warning("Could not gather test metrics")
        
        notifier = SlackManager()
        success = notifier.send_notification(status, data)
        
        if not success:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Slack notification failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    import sys
    main()
