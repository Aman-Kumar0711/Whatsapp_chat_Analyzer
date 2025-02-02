from urlextract import URLExtract
import pandas as pd
from collections import Counter
import emoji
import os
from dotenv import load_dotenv
from groq import Groq
extractor = URLExtract()
import matplotlib.pyplot as plt
load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)

from wordcloud import WordCloud
def fetch_stats(selected_user,df):
    if selected_user !='Overall Analysis':
        df=df[df['user']==selected_user]
    num_messages=df.shape[0]
    words=[]
    for message in df['message']:
        words.extend(message.split())
    #media
    #num_media=df[df['message']=='null\n'].shape[0]
    num_media = df[df['message'].str.strip().isin(['null', '<Media omitted>'])].shape[0]
    links=[]
    for link in df['message']:
        links.extend(extractor.find_urls(link))
    return num_messages,len(words), num_media, len(links)
#message export karte time agar do not include media karoge to message mei <media omitted> ya null likha aata hai
#hence agar main message column mei media omitted ya null naam ke messages count kar loon wahi mera count hoga total media shared ka
def most_busy_users(df):
    x=df['user'].value_counts().head()
    df_percentage=round((df['user'].value_counts()/df.shape[0])*100,2).reset_index()
    df_percentage.rename(columns={'index':'name','user':'percent'})
    return x,df_percentage
def create_wordcloud(selected_user,df):
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()
    if selected_user !='Overall Analysis':
        df=df[df['user']==selected_user]
    temp = df[df['message'] != 'Notification']
    #temp = temp[temp['message'] != 'null\n']
    temp = temp[~temp['message'].str.strip().isin(['null\n', '<Media omitted>\n'])]
    def remove_stop_words(message):
        y=[]
        for word in message.lower().split():
            if word not in stop_words and word not in ['<Media omitted>','null']:
                y.append(word)
        return ' '.join(y)
    wc=WordCloud(width=500,height=500,min_font_size=10,background_color='black')
    #generate fxn hota hai iske andar woh ek word cloud generate karta hai as an image
    temp['message']=temp['message'].apply(remove_stop_words)
    df_wc=wc.generate(df['message'].str.cat(sep=' '))
    return df_wc
#ab main bataunga ki kaunse words sabse zyada used hain, per uske liye mujhe grp notification waala part hatana padega
#mujhe stop words bhi hatane padenge
#agar english mei chats hoti to fir nltk use karke filter out kar deta
#per main hinglish use kar rha hoon to woh use nahin kar sakta main, iske liye I will use ek file jo ki online available hai usme generally used stop words hain woh sab hain...

#chalo karte hain fir
def most_common_words(selected_user,df):
    if selected_user !='Overall Analysis':
        df=df[df['user']==selected_user]
    f=open('stop_hinglish.txt','r')
    stop_words=f.read()
    temp=df[df['message']!='Notification']
    #temp=temp[temp['message']!='null\n']
    temp = temp[~temp['message'].str.strip().isin(['null\n', '<Media omitted>\n'])]
    words=[]
    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)
    final_df=pd.DataFrame(Counter(words).most_common(20))
    return final_df


def emoji_helper(selected_user,df):
    if selected_user !='Overall Analysis':
        df=df[df['user']==selected_user]
    emojis=[]
    for message in df['message']:
        emojis.extend([c for c in message if emoji.is_emoji(c)])
    emoji_df=pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))),columns=['emoji', 'count'])
    return emoji_df
#reset_index karke dataframe ban jaata hai




def monthly_timeline(selected_user,df):
    if selected_user !='Overall Analysis':
        df=df[df['user']==selected_user]
    timeline=df.groupby(['year','month','month_num']).count()['message'].reset_index()
    time=[]
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i]+"-"+str(timeline['year'][i]))
    timeline['time']=time
    return timeline





def daily_timeline(selected_user,df):
    if selected_user !='Overall Analysis':
        df=df[df['user']==selected_user]
    dtimeline=df.groupby(['only_date']).count()['message'].reset_index()
    return dtimeline




def week_activity_map(selected_user,df):
    if selected_user !='Overall Analysis':
        df=df[df['user']==selected_user]
    return df['day_name'].value_counts()




def month_activity_map(selected_user,df):
    if selected_user !='Overall Analysis':
        df=df[df['user']==selected_user]
    return df['month'].value_counts()


def split_text_into_chunks(text, chunk_size=3000):
    """Split text into chunks of approximately equal size"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0

    for word in words:
        current_size += len(word) + 1  # +1 for space
        if current_size > chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_size = len(word)
        else:
            current_chunk.append(word)

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def summarize_chat(selected_user, df):
    if selected_user != "Overall Analysis":
        df = df[df['user'] == selected_user]

    chat_content = df['message'].str.cat(sep=' ')

    # Split the chat content into chunks
    chunks = split_text_into_chunks(chat_content)
    chunk_summaries = []

    # Get summary for each chunk
    for chunk in chunks:
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"Summarize this part of a chat conversation concisely: {chunk}",
                    }
                ],
                model="llama-3.3-70b-versatile",
            )
            chunk_summary = chat_completion.choices[0].message.content
            chunk_summaries.append(chunk_summary)
        except Exception as e:
            print(f"Error processing chunk: {str(e)}")
            continue

    # If we have multiple summaries, get a final summary of summaries
    if len(chunk_summaries) > 1:
        combined_summaries = " ".join(chunk_summaries)
        try:
            final_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"Create a coherent final summary from these partial summaries: {combined_summaries}",
                    }
                ],
                model="llama-3.3-70b-versatile",
            )
            final_summary = final_completion.choices[0].message.content
            return final_summary
        except Exception as e:
            print(f"Error creating final summary: {str(e)}")
            return "Error creating final summary. " + " ".join(chunk_summaries)

    # If we only had one chunk, return its summary
    elif len(chunk_summaries) == 1:
        return chunk_summaries[0]
    else:
        return "No summary could be generated."
