import streamlit as st
from matplotlib import pyplot as plt
import preprocessor
import helper
st.sidebar.title("WhatsApp Chat Analyzer")
uploaded_file=st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data=uploaded_file.getvalue()
    data=bytes_data.decode("utf-8") #iss se stream jo aayegi usko string mei convert kar denge
    #st.text(data) #iss se right side mei sari chats dikhne lagengi
    df=preprocessor.preprocess(data)
    #st.dataframe(df)
    #Fetch unique users
    user_list=df['user'].unique().tolist()
    if 'Notification' in user_list:
        user_list.remove('Notification')
    if 'Group_notification' in user_list:
        user_list.remove('Group_notification')
    user_list.sort()
    user_list.insert(0,'Overall Analysis')

    selected_user=st.sidebar.selectbox("Show Chat Analysis WRT",user_list)
    if st.sidebar.button("Show Analysis"):
        num_messages,words,num_media,links=helper.fetch_stats(selected_user,df)
        st.title("Top Statistics")
        col1, col2, col3, col4=st.columns(4)
        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(words)
        with col3:
            st.header("Total Media")
            st.title(num_media)
        with col4:
            st.header("Total Links")
            st.title(links)
        #monthly_timeline
        st.title("Monthly Timeline")
        timeline=helper.monthly_timeline(selected_user,df)
        fig,ax=plt.subplots()
        ax.plot(timeline['time'],timeline['message'])
        plt.xticks(rotation='vertical')
        st.pyplot(fig)
        #daily_timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'])
        plt.xticks(rotation='vertical')
        st.pyplot(fig)
        #activity_map
        st.title("Activity Map")
        col1,col2=st.columns(2)
        with col1:
            st.header("Most Busy Day")
            busy_day=helper.week_activity_map(selected_user,df)
            fig,ax=plt.subplots()
            ax.bar(busy_day.index,busy_day.values)
            st.pyplot(fig)
        with col2:
            st.header("Most Busy Month")
            busy_month=helper.month_activity_map(selected_user,df)
            fig,ax=plt.subplots()
            ax.bar(busy_month.index,busy_month.values)
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        #Finding the busiest user in the group
        if selected_user=='Overall Analysis':
            st.title("Most Busy Users")
            x,new_df=helper.most_busy_users(df)
            fig,ax=plt.subplots()
            col1,col2=st.columns(2)
            with col1:
                ax.bar(x.index, x.values)
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)
        st.title("WordCloud")
        df_wc=helper.create_wordcloud(selected_user,df)
        fig,ax=plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)
        st.title("Most Common 20 Words")
        most_common_df=helper.most_common_words(selected_user,df)
        fig,ax=plt.subplots()
        ax.bar(most_common_df[0], most_common_df[1])
        plt.xticks(rotation='vertical')
        #st.dataframe(most_common_df)
        st.pyplot(fig)
        #Emoji Analysis
        st.title("Emoji Analysis")
        emoji_df=helper.emoji_helper(selected_user,df)
        st.dataframe(emoji_df)
        #fig,ax=plt.subplots()
    if st.sidebar.button("Summarize Chat"):
        st.title("Chat Summary")
        summary = helper.summarize_chat(selected_user, df)
        st.write(summary)




