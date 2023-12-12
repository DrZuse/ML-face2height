import requests, csv, os
from bs4 import BeautifulSoup

# Setting the folder name where data will be stored
data_folder = './celebheights'
# Setting the alphabet for iterating over the website's pages
alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
# Base URL of the website to be scraped
url = 'https://www.celebheights.com/'

# Checking if the data folder exists, if not, creating it
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

# Function to save a photo from a given URL
def save_photo(photo_url, thumbnail_name):
    try:
        # Sending a GET request to the photo URL
        response = requests.get(photo_url)
    except Exception as e:
        # Printing the exception if any occurs
        print(str(e))
        return False

    # If the response status code is 200, save the photo
    if response.status_code == 200:
        with open('celebheights/' + thumbnail_name, 'wb') as f:
            f.write(response.content)
        return True
    else:
        # If the status code is not 200, print it and return False
        print('photo_url status code: ' + response.status_code)
        return False

# Creating a CSV file with headers
with open(data_folder+'/metadata.csv', 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['name', 'photo_url', 'thumbnail_name', 'height_cm', 'link'])

# Iterating over each letter in the alphabet
for a in alphabet:
    try:
        # Constructing the URL to scrape
        scrap = url + 's/all' + a + '.html'
        print(scrap)
        # Sending a GET request to the URL
        response = requests.get(scrap)
        # Parsing the HTML content of the page
        soup = BeautifulSoup(response.text, 'html.parser')
        # Finding all the blocks with class sAZ2'
        blocks = soup.findAll(attrs={'class':'sAZ2'})

        # Iterating over each block
        for b in blocks:
            # Extracting the name, link, height, and photo URL
            name = b.find('a').text.strip().lower()
            link = b.find('a')['href']
            thumbnail_name = ''.join(filter(str.isalpha, link[31:-5])).lower()+'.jpg'
            height = b.text.strip()
            height_cm = height[height.index('(')+1 : height.index(')')-2]
            photo_url = url + 'tr/' + name[0] + '/' + thumbnail_name
            print(name, link, height_cm, photo_url)
            # Saving the photo
            saved = save_photo(photo_url, thumbnail_name)

            # If the photo is saved, write the data to the CSV file
            if saved:
                with open(data_folder+'/metadata.csv', 'a') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow([name, photo_url, thumbnail_name, height_cm, link])

    except Exception as e:
        # Printing the exception if any occurs
        print(str(e))