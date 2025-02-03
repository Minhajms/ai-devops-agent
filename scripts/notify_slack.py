import os
from slack_sdk import WebClient

def send_slack_message(message):
    client = WebClient(token=os.environ["SLACK_TOKEN"])
    client.chat_postMessage(channel="#devops", text=message)

if __name__ == "__main__":
    send_slack_message("Build failed! Check the logs.")
