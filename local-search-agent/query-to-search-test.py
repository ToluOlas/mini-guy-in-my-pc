from ddgs import DDGS
import trafilatura

def run_search(query: str):
    raw = DDGS().text(query, max_results=5)
    results = [
        {"title": r["title"], "url": r["href"], "description": r["body"]}
        for r in (raw or [])
    ]
    return results

def run_extract(url: str):
    html = trafilatura.fetch_url(url)
    if not html:
        return None
    text = trafilatura.extract(html, include_comments=False, include_tables=False)
    return text

def main():
    while True:
        query = input("Search: ")
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        results = run_search(query)

        if not results:
            print("No results found.\n")
            continue

        print(f"\n--- {len(results)} results for: '{query}' ---\n")
        for i, r in enumerate(results, 1):
            print(f"[{i}] {r['title']}")
            print(f"    URL: {r['url']}")
            print(f"    Snippet: {r['description']}")
            print()

        for i, r in enumerate(results, 1):
            print(f"--- Extracting [{i}] {r['title']} ---\n")
            content = run_extract(r["url"])
            if content:
                print(content)
            else:
                print("Could not extract content from this page.")
            print()

if __name__ == "__main__":
    main()
