#Εισαγωγή απαραίτητων Βιβλιοθηκών
import pandas as pd
import requests
from bs4 import BeautifulSoup as soup
from lxml import html
from warnings import warn
import os
#Εισαγωγή Λίστών Διατάξεων και Νομικών Συμβούλων
from classification.data_list import provision_list, lawyer_list
# Στήλες : Τίτλος , Γνωμοδότηση, Κατηγορία
columns=['Title','Concultatory','Category']
'''
0:μη αποδεκτή, 1:αποδεκτή,2:εν μερει αποδεκτή, 3:ανακληθηκε το ερώτημα,-1: Εκκρεμεί η Αποδοχή
4:Για την αποδοχή ή μη επικοινωνήστε με τον Σχ. Επιστ. Δραστηριοτήτων κ Δημοσίων Σχέσεων, 
'''
df_all = pd.DataFrame(columns=columns)

for status in ['1','-1','4']:
    for year in range(1980,2020):
        nsk_url = 'http://www.nsk.gr/web/nsk/anazitisi-gnomodoteseon?p_p_' \
                  'id=nskconsulatories_WAR_nskplatformportlet&p_p_lifecycle=0&p_p_state=normal&p_p_' \
                  'mode=view&p_p_col_id=column-4&p_p_col_pos=2&p_p_col_count=3'
        post_data = {"_nskconsulatories_WAR_nskplatformportlet_isSearch": "1",
                     "_nskconsulatories_WAR_nskplatformportlet_inputDatefrom": year,
                     "_nskconsulatories_WAR_nskplatformportlet_consulState": status}
        # Εδώ γίνεται το request της σελίδας του νομικού συμβουλίου
        response = requests.post(nsk_url , post_data)
        ##########################################################
        #parsing και scraping της σελίδας
        if response.status_code != 200:
            warn('Request: {}; Status code: {}'.format(requests , response.status_code))

        page_soup = soup(response.text, 'html.parser')

        #Εξαγωγή γνωμοδοτήσεων!
        concul_containers = page_soup.find_all("div", {"class" : "article_text"})

        #λίστα με της γνωμοδοτήσεις(τίτλοι)
        concultatories= []
        for container in concul_containers:
            article = container.strong.text.strip()
            concultatories.append(article)

        #Εξαγωγή αποφάσεων
        articles = html.fromstring (response.content.decode ('utf-8'))
        decisions = articles.xpath('//*[@id="resdiv"]/div/div[1]/div[2]/text()')

        #Καθαρισμός από " και \n\t
        punctuation = '"'
        for index, item in enumerate(decisions):
            decisions[index] = item.strip()
        for char in decisions[:]:
            if char in punctuation:
                decisions.remove(char)

        # Εξαγωγή κατηγοριών από τα λήμματα-keywords
        keywords = articles.xpath('//div[@class="article_text"]/p/text()')
        keywords = [x.strip() for x in keywords]
        keywords = [x for x in keywords if len(x) > 0]
        keywords = [x for x in keywords if 'Εκκρεμεί αποδοχή' not in x]
        keywords = [x for x in keywords if 'Αποδεκτή' not in x]
        keywords = [x for x in keywords if 'Για την αποδοχή ή μη επικοινωνήστε με τον Σχ. Επιστ. Δραστηριοτήτων κ Δημοσίων Σχέσεων'
                    not in x]
        keywords = [x for x in keywords if x not in lawyer_list]
        keywords = [x for x in keywords if x not in provision_list]
        keywords = [x for x in keywords if '/' not in x]

        #Αφαίρεση '"','-','--','---'
        punctuation = ['"','-','--','---']
        for i, word in enumerate(keywords):
            keywords[i] = word.strip()
        for chr in keywords[:]:
            if chr in punctuation:
                keywords.remove(chr)
        #Λίστα Λημμάτων
        keywords_list = [x.split(',') for x in keywords]
        s1 = pd.Series(concultatories, name='Title')
        s2 = pd.Series(decisions, name='Concultatory')
        s3 = pd.Series(keywords_list, name='Category')

        df = pd.concat([s1,s2,s3], axis=1)
        #DATA-FRAME και export σε xlsx

        df_all = pd.concat([df_all, df],ignore_index=True,sort=True)

    #αφαιρεση τυχόν κενών στους τίτλους και στις αποφάσεις των γνωμοδοτήσεων

    df_all['Title'] = df_all['Title'].apply(lambda x: x.rstrip())
    df_all['Concultatory'] = df_all['Concultatory'].apply(lambda x: x.rstrip())

    #Αποθήκευση df σε xlsx
    fname_path = os.path.join('data','nsk_multilabel.xlsx')
    df_all.to_excel(fname_path, index=False)