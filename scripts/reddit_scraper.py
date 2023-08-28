import praw
import json

# Replace the following placeholders with your Reddit API credentials
client_id = ##
client_secret = ##
user_agent = ##
username = ##
password = ##

# Initialize Reddit instance
reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent,
    username=username,
    password=password
)

# Scrape conversation from the Reddit post
url = "https://www.reddit.com/r/wallstreetbets/comments/11vy6we/the_great_financial_collapse_of_2023_comparison/"
submission = reddit.submission(url=url)

print("tree")
# Flatten the conversation tree
submission.comments.replace_more(limit=100)
print("This is everything")
print(submission.comments.list())
comment_list = []
for comment in submission.comments.list():
    comment_list.append(comment.body + ". ")

# Export the conversation to a JSON file
with open("wsb_creditsuissecollapse.json", "w") as json_file:
    json.dump(comment_list, json_file, indent=4)
