from searchEngine import SearchEngine
import time

def main():
    searchEngine = SearchEngine()
    '''
    searchQueries = ["cristina lopes", "machine learning", "ACM", "master software engineering"]
    for query in searchQueries:
        start = time.time()
        results = searchEngine.search(query)
        end = time.time()
        print(end-start, query)
        for i in range(5):
            print(results[i])
    '''
    query = input("\nType \"quit\" to exit program\nSearch: ")
    while query != "quit":
        start = time.time()
        results = searchEngine.search(query)
        end = time.time()
        if len(results) == 0:
            print("No results")
        else:
            print("Retrieved in", round(end-start, 4), "seconds")
            for i in range(5):
                try:
                    print(results[i])
                except IndexError:
                    break
   
        query = input("\nSearch: ")
    
if __name__ == "__main__":
    main()
