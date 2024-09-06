import streamlit as st
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from urlextract import URLExtract
from wordcloud import WordCloud
from collections import Counter
from setuptools import setup
import emoji
import os
from scipy.stats import chi2_contingency

extract = URLExtract()

# Function to preprocess data
def preprocess(data):
    pattern = '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s[APap][Mm]\s-\s'
    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)
    clean_dates = [date.replace('\u202f', ' ') for date in dates]
    df = pd.DataFrame({'user_message': messages, 'message_date': clean_dates})
    df['message_date'] = pd.to_datetime(df['message_date'], format='%d/%m/%y, %I:%M %p - ', errors='coerce')
    df.rename(columns={'message_date': 'date'}, inplace=True)
    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df['formatted_date'] = df['date'].dt.strftime('%d/%m/%y, %I:%M %p')

    users, messages = [], []
    for message in df['user_message']:
        entry = re.split('([\w\W]+?):\s', message)
        if entry[1:]:
            users.append(entry[1])
            messages.append(' '.join(entry[2:]))
        else:
            users.append('group_notification')
            messages.append(entry[0])

    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)
    df = df[['formatted_date', 'user', 'message', 'year', 'month_num', 'month', 'day', 'day_name', 'hour', 'minute']]
    
    period = []
    for hour in df['hour']:
        if hour == 23:
            period.append(f"{hour}-00")
        elif hour == 0:
            period.append("00-01")
        else:
            period.append(f"{hour}-{hour+1}")
    df['period'] = period

    return df

# Sidebar and file upload
st.sidebar.title("Whatsapp Chat Analyzer")
uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocess(data)
    st.dataframe(df)

    # Fetch unique users
    user_list = df['user'].unique().tolist()
    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")
    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

    if st.sidebar.button("Show Analysis"):
        # Fetch stats
        def fetch_stats(selected_user, df):
            if selected_user != 'Overall':
                df = df[df['user'] == selected_user]

            num_messages = df.shape[0]
            words = [word for message in df['message'] for word in message.split()]
            num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]
            links = [link for message in df['message'] for link in extract.find_urls(message)]
            
            return num_messages, len(words), num_media_messages, len(links)
        
        num_messages, words, num_media_messages, links = fetch_stats(selected_user, df)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.header('Total Messages')
            st.title(num_messages)
        with col2:
            st.header('Total Words')
            st.title(words)
        with col3:
            st.header('Media Shared')
            st.title(num_media_messages)
        with col4:
            st.header('Links Shared')
            st.title(links)

        # Most Busy Users
        if selected_user == 'Overall':
            st.title('Most Busy Users')
            def most_busy_users(df):
                x = df['user'].value_counts().head()
                df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(columns={'user': 'name', 'count': 'percentage'})
                return x, df
            most_busy, new_df = most_busy_users(df)  
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots()
                ax.bar(most_busy.index, most_busy.values)
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)
        # Stop words
        # Load all stop words from the stopwords directory and store in a set
        stopwords_dir = "D:\Data Science\Project\Realtime Projects\Whatsapp chat analysis\stop_words"
        stop_words = []
        if os.path.exists(stopwords_dir):
            for file in os.listdir(stopwords_dir):
                if file.endswith(".txt"):
                    with open(os.path.join(stopwords_dir, file), 'r', encoding='ISO-8859-1') as f:
                        stop_words.extend(set(f.read().splitlines()))
        stop_words = set([x.lower() for x in stop_words])

        # WordCloud
        def create_wordcloud(selected_user, df):
            if selected_user != 'Overall':
                df = df[df['user'] == selected_user]
            temp = df[df['user'] != 'group_notification']
            temp = temp[temp['message'] != '<Media omitted>\n']
            
            def remove_stop_words(message):
                y = [word for word in message.lower().split() if word not in stop_words]
                return " ".join(y)

            wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
            temp['message'] = temp['message'].apply(remove_stop_words)
            df_wc = wc.generate(temp['message'].str.cat(sep=" "))
            return df_wc
        
        st.title("Wordcloud")
        df_wc = create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        # Most Common Words
        def most_common_words(selected_user, df):
            if selected_user != 'Overall':
                df = df[df['user'] == selected_user]
            temp = df[df['user'] != 'group_notification']
            temp = temp[temp['message'] != '<Media omitted>\n']
            words = [word for message in temp['message'] for word in message.lower().split() if word not in stop_words]
            most_common_df = pd.DataFrame(Counter(words).most_common(20))
            return most_common_df
        
        most_common_df = most_common_words(selected_user, df)
        st.title('Most Common Words')
        st.dataframe(most_common_df)

        fig, ax = plt.subplots()
        ax.barh(most_common_df[0], most_common_df[1])
        plt.xticks(rotation='vertical')
        st.title('Most Common Words - Visual Representation')
        st.pyplot(fig)

        # Emoji Analysis
        def emoji_helper(selected_user, df):
            if selected_user != 'Overall':
                df = df[df['user'] == selected_user]
            emojis = [char for message in df['message'] for char in message if char in emoji.EMOJI_DATA]
            emoji_counts = Counter(emojis)
            emoji_df = pd.DataFrame(emoji_counts.most_common(5), columns=['emoji', 'count'])
            return emoji_df

        emoji_df = emoji_helper(selected_user, df)
        st.title("Emoji Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(emoji_df)
        with col2:
            fig, ax = plt.subplots()
            ax.pie(emoji_df['count'], labels=emoji_df['emoji'], autopct='%0.2f%%', startangle=140)
            ax.axis('equal')
            st.pyplot(fig)

        # Monthly Timeline
        def monthly_timeline(selected_user, df):
            if selected_user != 'Overall':
                df = df[df['user'] == selected_user]
            timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
            timeline['message'] = pd.to_numeric(timeline['message'], errors='coerce')
            timeline['time'] = pd.to_datetime(timeline['month'] + "-" + timeline['year'].astype(str), format='%B-%Y')
            return timeline

        st.title("Monthly Timeline")
        timeline = monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'].values, timeline['message'].values, color='green')
        plt.xticks(rotation='vertical')
        ax.set_xlabel("Time")
        ax.set_ylabel("Number of Messages")
        ax.set_title("Monthly Message Counts")
        st.pyplot(fig)

        # Daily Timeline
        def daily_timeline(selected_user,df):

            if selected_user != 'Overall':
                df = df[df['user'] == selected_user]

            # Ensure 'only_date' column exists or rename as necessary
            if 'formatted_date' not in df.columns:
                raise KeyError("Column 'only_date' does not exist in DataFrame")

            df['formatted_date'] = pd.to_datetime(df['formatted_date'], format='%m/%d/%y, %I:%M %p', errors='coerce')

            daily_timeline_dt = df.groupby('formatted_date').count()['message'].reset_index()
            daily_timeline_dt['message'] = pd.to_numeric(daily_timeline_dt['message'], errors='coerce') 
            return daily_timeline_dt
        
        # daily timeline
        st.title("Daily Timeline")
        daily_timeline_dt = daily_timeline(selected_user, df)
        time_values = daily_timeline_dt['formatted_date'].values
        message_values = daily_timeline_dt['message'].values

        fig,ax = plt.subplots()
        ax.plot(time_values, message_values, color='green')
        plt.xticks(rotation='vertical')

        ax.set_xlabel("Time")
        ax.set_ylabel("Number of Messages")
        ax.set_title("Daily Message Counts")
        st.pyplot(fig)

        def week_activity_map(selected_user,df):

            if selected_user != 'Overall':
                df = df[df['user'] == selected_user]

            return df['day_name'].value_counts()

        def month_activity_map(selected_user,df):

            if selected_user != 'Overall':
                df = df[df['user'] == selected_user]

            return df['month'].value_counts()

        def activity_heatmap(selected_user,df):

            if selected_user != 'Overall':
                df = df[df['user'] == selected_user]

            user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

            return user_heatmap
        # activity map
        st.title('Activity Map')
        col1,col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = week_activity_map(selected_user,df)
            fig,ax = plt.subplots()
            ax.bar(busy_day.index,busy_day.values,color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values,color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        user_heatmap = activity_heatmap(selected_user,df)
        fig,ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)

    # User-Day Association Test
    st.sidebar.header("comparing user Activity Patterns Across Days of the Week")

    def user_day_association_test(df, user1, user2, year, selected_month):
        # Filter DataFrame based on the time period and users
        df_filtered = df[(df['year'] == year) & (df['month'] == selected_month)]
        
        # Ensure both users are included in the filtered DataFrame
        if user1 != 'Overall' and user2 != 'Overall':
            df_filtered = df_filtered[df_filtered['user'].isin([user1, user2])]
        elif user1 != 'Overall':
            df_filtered = df_filtered[df_filtered['user'] == user1]
        elif user2 != 'Overall':
            df_filtered = df_filtered[df_filtered['user'] == user2]
        
        # Check if there's enough data to perform the test
        if df_filtered.empty:
            return None, None, None, None, None, None
        
        # Create a contingency table of days for the two users
        contingency_table = pd.crosstab(df_filtered['user'], df_filtered['day_name'])
        
        # Perform Chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # Optionally, return the difference between observed and expected frequencies
        observed_minus_expected = contingency_table - pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns)
        
        return chi2, p_value, dof, expected, contingency_table, observed_minus_expected

    # Filter out 'group_notification' for both dropdowns
    users_excluding_group_notification = [user for user in df['user'].unique() if user != 'group_notification']

    # Select two users and a year
    user1 = st.sidebar.selectbox("Select First User", users_excluding_group_notification, key='user1')
    users_excluding_user1 = [user for user in users_excluding_group_notification if user != user1]
    user2 = st.sidebar.selectbox("Select Second User", users_excluding_user1, key='user2')
  
    year = st.sidebar.slider("Select Year", min_value=df['year'].min(), max_value=df['year'].max(), value=df['year'].max(), key='year')
    selected_month = st.sidebar.selectbox("Select Month for Test", df['month'].unique(), key='month')

    if st.sidebar.button("Run User-Day Association Test"):
        chi2, p_value, dof, expected, contingency_table, observed_minus_expected = user_day_association_test(df, user1, user2, year, selected_month)

        if chi2 is not None:
            st.header(f"Statistical analysis of {user1} vs {user2} in {selected_month} (Year: {year})")
            #st.write("Chi-square statistic:", chi2)
            #st.write("P-value:", p_value)
            #st.write("Degrees of Freedom:", dof)

            st.write("Expected Frequencies Table:")
            st.dataframe(pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns))

            st.write("Contingency Table:")
            st.dataframe(contingency_table)

            if p_value < 0.05:
                st.write(f"{user1} and {user2} were very interactive on different days in {selected_month} (Year: {year}).")
            else:
                st.write(f"{user1} and {user2} were not very interactive on different days in {selected_month} (Year: {year}).")
        else:
            st.write("No data available for the selected criteria.")



            
