from utils.packages import *
#...............................................................................................    

#...............................................................................................    
def csvtotext(df):
    text = df.to_csv(sep=' ', index=False, header=False).lower()
    return(text)
#...............................................................................................        


#...............................................................................................    
def cleantext(text):
    full_tokens = word_tokenize(text)
    tokens = [word for word in full_tokens if word.isalpha()]
    tokens = [word for word in tokens if len(word)>1]        
    return(full_tokens, tokens)
#...............................................................................................    


#...............................................................................................    
def word_freq_analyser(tokens):
    freq = nltk.FreqDist(tokens)
    freqdf = pd.DataFrame.from_dict(freq,orient='index')
    freqdf = freqdf.reset_index()
    freqdf = freqdf.rename(columns={'index' : 'word', 0:'count'})
    freqdf = freqdf.sort_values(by = ['count'], ascending = False)
    freqdf = freqdf.reset_index(drop=True)
    freqdf = freqdf.reset_index()
    freqdf = freqdf.rename(columns={'index' : 'sno'})
    freqdf['sno'] = freqdf['sno'] + 1
    freqdf['distinct_word_%'] = freqdf['sno']/len(freqdf['sno']) * 100
    freqdf['cumsum'] = freqdf['count'].cumsum()
    freqdf['corpus_%'] = freqdf['cumsum'] / freqdf['count'].sum() * 100
    
    freqdf['distinct_word_%'] = np.round(freqdf['distinct_word_%'], 0)
    freqdf['corpus_%'] = np.round(freqdf['corpus_%'], 0)
    
    small_freq_df = freqdf.drop_duplicates(subset=['corpus_%'], keep = 'first')
    
    return(freqdf, small_freq_df)    
#...............................................................................................    


#...............................................................................................    
def plot_reverse_elbow_curve(small_freq_df):
    x = small_freq_df['distinct_word_%']
    y = small_freq_df['corpus_%']

    fig, ax = plt.subplots()
    fig.set_size_inches(15, 5)

    ax.plot(x, y, **{'color': 'lightsteelblue', 'marker': 'o'})
    plt.xticks(np.arange(min(x), 100, 2))
    plt.yticks(np.arange(min(y), 100, 10))
    plt.xlabel ('distinct_word_%', fontsize = 15)
    plt.ylabel ('corpus_%', fontsize = 15)
    return(plt)
#...............................................................................................    


#...............................................................................................    
def create_report_df(small_freq_df, freqdf):
    df_report = pd.DataFrame()
    imp_deciles = list(np.arange(75,101,5))
    num_of_words = []
    perc_of_dist_words = []

    for dec in imp_deciles:
        perc_of_dist_words.append(small_freq_df[small_freq_df['corpus_%'] == dec].reset_index()['distinct_word_%'][0])
        num_of_words.append(small_freq_df[small_freq_df['corpus_%'] == dec].reset_index()['sno'][0])

    df_report['num_of_words'] = num_of_words
    df_report['dist_words%'] = perc_of_dist_words
    df_report['corpus%'] = imp_deciles
    report1 = f'1. Most frequent {df_report["num_of_words"][0]} unique words, which are {df_report["dist_words%"][0]}% of the total distinct words, are contributing to the {df_report["corpus%"][0]}% of the total corpus.'
    
    dist_word_count_90 = small_freq_df[small_freq_df['corpus_%'] == 90].reset_index()['sno'][0]
    dist_word_count_100 = freqdf['sno'].iloc[-1]
    dist_word_90_100 = dist_word_count_100 - dist_word_count_90
    dist_word_bottom_perc = np.round((dist_word_90_100 / dist_word_count_100)*100,1)
    report2 = f'2. Least frequent {dist_word_90_100} unique words, which are {dist_word_bottom_perc}% of the total distinct words, are contributing only to the bottom 10% of the total corpus.'

    print("---------------------------------- Analysis Results ----------------------------------")
    print(report1)
    print(report2)
    print("--------------------------------------------")
    print(df_report)
    print("--------------------")
    print('num_of_words : Distinct count of Top-most frequent words')
    print('dist_words%  : num_of_words / total distinct words in corpus')
    print('corpus%      : Total count of num_of_words / Total number of words in corpus')
    print("--------------------------------------------")
    return(df_report)            
#...............................................................................................    


#...............................................................................................    
def dynamic_stop_word_analyzer(tokens):
    freqdf, small_freq_df = word_freq_analyser(tokens)
    plt = plot_reverse_elbow_curve(small_freq_df) 
    df_report = create_report_df(small_freq_df, freqdf)
    return(freqdf, small_freq_df, df_report)
#...............................................................................................    


#...............................................................................................    
def get_dynamic_stop_words_list(freqdf, corpus_perc_threshold, stop_words_skip_list):
    stop_words_list = list(freqdf[freqdf['corpus_%'] > corpus_perc_threshold]['word'])
    stop_words_list = list(set(stop_words_list) - set(stop_words_skip_list))
    
    non_stop_words_list = list(freqdf[freqdf['corpus_%'] < corpus_perc_threshold]['word'])
    non_stop_words_list = list(set(non_stop_words_list + stop_words_skip_list))
    
    print('-----------------------------')
    print(f'non_stop_words_list has {len(non_stop_words_list)} words e.g.({non_stop_words_list[1:5]})')      
    print(f'stop_words_list has {len(stop_words_list)} words e.g.({stop_words_list[1:5]})')
    print('-----------------------------')

    return(stop_words_list, non_stop_words_list)
#...............................................................................................    
       


#...............................................................................................    
def create_tokens_list_of_list(tokens):   
        
    sentence = []
    message = []
    for word in tokens:
        if word != '§':
            sentence.append(word)
        if word == '§':
            message.append(sentence)
            sentence = []
    
    joined_message = [' '.join(x) for x in message]    
    return(message, joined_message)
#...............................................................................................    