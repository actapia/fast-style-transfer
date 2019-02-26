import os
import base64
import mimetypes
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart

import pickle
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/gmail.send']

message_txt = \
"""
Hi! This is William from E-Day this past Saturday, and you participated in my \
demo named "The Painting Art of Deep Learning". I converted your image into \
different styles, all six of which are attached. If you are interested in more \
details about the concept of image style transfer, please visit the original \
creator's website.

Thank you very much for visiting my booth!

Regards,
Weilian Song
"""

def auth():
  creds = None
  # The file token.pickle stores the user's access and refresh tokens, and is
  # created automatically when the authorization flow completes for the first
  # time.
  if os.path.exists('token.pickle'):
      with open('token.pickle', 'rb') as token:
          creds = pickle.load(token)
  # If there are no (valid) credentials available, let the user log in.
  if not creds or not creds.valid:
      if creds and creds.expired and creds.refresh_token:
          creds.refresh(Request())
      else:
          flow = InstalledAppFlow.from_client_secrets_file(
              'credentials.json', SCOPES)
          creds = flow.run_local_server()
      # Save the credentials for the next run
      with open('token.pickle', 'wb') as token:
          pickle.dump(creds, token)

  service = build('gmail', 'v1', credentials=creds)

  return service

def create_msg(sender, to, subject, message_txt, img_files):
  message = MIMEMultipart()
  message['to'] = to
  message['from'] = sender
  message['subject'] = subject

  msg = MIMEText(message_txt)
  message.attach(msg)

  for img_f in img_files:
    content_type, encoding = mimetypes.guess_type(img_f)
    main_type, sub_type = content_type.split('/', 1)

    with open(img_f, 'rb') as fp:
      msg = MIMEImage(fp.read(), _subtype=sub_type)

    filename = os.path.basename(img_f)
    msg.add_header('Content-Disposition', 'attachment', filename=filename)
    message.attach(msg)

  return {'raw': base64.urlsafe_b64encode(message.as_string())}

def send_msg(service, user_id, message):
  try:
    message = (service.users().messages().send(userId=user_id, body=message)
               .execute())
    print 'Message Id: %s' % message['id']
    return message

  except Exception, error:
    print 'An error occurred: %s' % error

def main():
  service = auth()

  sender = ''
  to = 'weilian.song@uky.edu'
  subject = 'Test Email'
  message_txt = 'This is a test.\nBest Regards,\nWeilian Song'
  img_files = ['./examples/eday/weilian.song@uky.edu.jpg',]

  message = create_msg(sender, to, subject, message_txt, img_files)
  message = send_msg(service, 'me', message)

if __name__ == '__main__':
  main()
