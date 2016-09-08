from TwitterAPI import TwitterAPI

# Consumer Key (API Key)
con_secret = ''
# Consumer Secret (API Secret)
con_secret_key = ''
# Access Token
token = ''
# Access Token Secret
token_key = ''



def tweet_chart(img_file, msg_text):
    api = TwitterAPI(con_secret, con_secret_key,token, token_key)
    file = open(img_file, 'rb')
    data = file.read()
    r = api.request('statuses/update_with_media', {'status': msg_text}, {'media[]': data})
    print(r.status_code)