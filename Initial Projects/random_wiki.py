import requests

def main():
    get_articles()
    get_more()

url = "https://en.wikipedia.org/w/api.php"

def get_articles():
    parameters = {
        "action": "query",
        "prop": "info",
        "inprop": "displaytitle|url",
        "list": "random",
        "rnnamespace": 0,
        "rnlimit": 3,
        "format": "json"
    }

    data = requests.get(url, params=parameters).json()

    articles = data["query"]["random"]

    for article in articles:
        print(f"Title: {article["title"]}")

def get_more():
    while True:
        user = input("Would you like more articles? (Y/N): ").lower().strip()
        if user == "y":
            get_articles()
        else:
            break

if __name__ == "__main__":
    main()