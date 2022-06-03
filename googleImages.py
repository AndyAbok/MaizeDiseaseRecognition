from selenium import webdriver
from selenium.webdriver.common.by import By
import requests
import io
from PIL import Image
import time
import os

pathChrome = r'C:\Users\andrewa\.nuget\packages\selenium.webdriver.chromedriver\100.0.4896.2000\driver\win32\chromedriver.exe'
pathEdge = r'D:\Projects\Machine Learning Exploration\Analytics and MachineLearning\Maize disease Recognition\webdrivers\edgedriver_win64\msedgedriver.exe'

chromeDriver = webdriver.Chrome(pathChrome)
edgeDriver = webdriver.Edge(pathEdge)

#URL getter function.Obtains and stores the urls for the queried images.
def getImagesUrl(webDriver, delay, maxImages,query):
	def scroll_down(webDriver):
		webDriver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
		time.sleep(delay)

	url = f"https://www.google.com/search?q={query}&tbm=isch&ved=2ahUKEwjykJ779tbzAhXhgnIEHSVQBksQ2-cCegQIABAA&oq={query}&gs_lcp=CgNpbWcQAzIHCAAQsQMQQzIHCAAQsQMQQzIECAAQQzIECAAQQzIECAAQQzIECAAQQzIECAAQQzIECAAQQzIECAAQQzIECAAQQzoHCCMQ7wMQJ1C_31NYvOJTYPbjU2gCcAB4AIABa4gBzQSSAQMzLjOYAQCgAQGqAQtnd3Mtd2l6LWltZ8ABAQ&sclient=img&ei=7vZuYfLhOeGFytMPpaCZ2AQ&bih=817&biw=1707&rlz=1C1CHBF_enCA918CA918"
	webDriver.get(url)

	image_urls = set()
	skips = 0

	while len(image_urls) + skips < maxImages:
		scroll_down(webDriver)

		thumbnails = webDriver.find_elements(By.CLASS_NAME, "Q4LuWd")

		for img in thumbnails[len(image_urls) + skips:maxImages]:
			try:
				img.click()
				time.sleep(delay)
			except:
				continue

			images = webDriver.find_elements(By.CLASS_NAME, "n3VNCb")
			for image in images:
				if image.get_attribute('src') in image_urls:
					maxImages += 1
					skips += 1
					break

				if image.get_attribute('src') and 'http' in image.get_attribute('src'):
					image_urls.add(image.get_attribute('src'))
					print(f"Found {len(image_urls)}")

	return image_urls

#Image downloader function - downloads the images from the urls obtained by url search function.
def downloadImage(downloadPath, url, fileName):
	try:
		imageContent = requests.get(url).content
		imageFile = io.BytesIO(imageContent)
		image = Image.open(imageFile)
		filePath = downloadPath + fileName

		with open(filePath, "wb") as f:
			image.save(f, "JPEG")

		print("Success")
	except Exception as e:
		print('FAILED -', e)

#Images downloader helper function to assign lables to the downloaded images.
def imageName(strInput):
    res1 = strInput.rstrip("/")
    res = res1.rsplit('/', 1)[-1]    
    return res

query = 'Nothern+Leaf+Blight+Maize'
folderName = 'imageData/disease1/'

if not os.path.isdir(folderName):
    os.makedirs(folderName)

urls = getImagesUrl(chromeDriver,3,5,query)
imgName = imageName(folderName)

for i, url in enumerate(urls):
	downloadImage(folderName, url, f'{imgName}'+ str(i) + ".jpg")

chromeDriver.quit() 
edgeDriver.quit()

