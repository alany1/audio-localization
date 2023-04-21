import flickrapi
import urllib.request
import os

# Your Flickr API credentials
api_key = "c708a2e3c40eacbe024aed9ebf8c5e24"
api_secret = "4429c633a5fd4b03"

# Initialize Flickr API client
flickr = flickrapi.FlickrAPI(api_key, api_secret, format='parsed-json')

# Specify your query parameters
query = 'car'
extras = 'url_o'

# Search for sounds that match your query
results = flickr.sounds.search(text=query, extras=extras, per_page=1)

# Get the URL of the sound file
sound_url = results['sounds']['sound'][0]['url_o']

# Download the sound file
import urllib.request

urllib.request.urlretrieve(sound_url, 'car_sound.mp3')
