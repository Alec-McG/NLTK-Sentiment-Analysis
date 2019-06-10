import json
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener

import sent_mod as s

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import time


open('twitter-out.txt', 'w').close()

# consumer key, consumer secret, access token, access secret.

ckey = "yrmIwajSJirPJPItxuVomEuGg"
csecret = "fJ4RSBjP7XOptdAPZ7WpJFObUeQrHSQ8zH4GJBTqutLKy1TRB1"
atoken = "2739349071-kxXX7xthcpsMfy5Q0dDb6IgfGXHw5r3mfI7fLAK"
asecret = "enPUWtSmA2qmmaRKSxrAzh5zSlYEwZptlTUYPdVrpBU6W"


class listener(StreamListener):

    def on_data(self, data):

        all_data = json.loads(data)

        all_data = json.loads(data)
        tweet = ascii(all_data["text"])
        # tweet = all_data["text"]

        sentiment_value, confidence = s.sentiment(tweet)

        print(tweet, sentiment_value, confidence)

        if confidence*100 >= 80:
            output = open("twitter-out.txt", "a")
            output.write(sentiment_value)
            output.write('\n')
            output.close()

        return True

    def on_error(self, status):
        print(status)


auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
# twitterStream.filter(track=["happy"])

tweet = twitterStream.filter(track=["Australia"])


