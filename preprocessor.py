import pandas as pd
import re
def preprocess(data):
    pattern = r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{1,2}(?:\s[APap][Mm])?\s-\s'
    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    # Create the dataframe
    df = pd.DataFrame({'user_message': messages, 'message_date': dates})

    # Convert message_date to datetime, handle both 24-hour and AM/PM formats
    def parse_date(date):
        try:
            # Try parsing as 24-hour format
            return pd.to_datetime(date, format='%m/%d/%y, %H:%M - ')
        except ValueError:
            # Fallback to 12-hour (AM/PM) format
            return pd.to_datetime(date, format='%m/%d/%y, %I:%M %p - ', errors='coerce')

    df['message_date'] = df['message_date'].apply(parse_date)
    df.rename(columns={'message_date': 'date'}, inplace=True)
    users = []
    messages = []
    for message in df['user_message']:
        entry = re.split('([\w\W]+?):\s', message)
        if entry[1:]:
            # user name
            users.append(entry[1])
            messages.append(entry[2])
        else:
            users.append('Notification')
            messages.append(entry[0])
    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month_name()
    df['month_num']=df['date'].dt.month
    df['only_date']=df['date'].dt.date
    df['day'] = df['date'].dt.day
    df['day_name']=df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    # Filter out rows with 'Notification' user
    df = df[df['user'] != 'Notification']
    return df
