import json
import re
try:
    from duckduckgo_search import DDGS
except ImportError:
    print("⚠️ duckduckgo_search not installed. Web search disabled.")
    print("   Install with: pip install duckduckgo-search")
    DDGS = None
from config import conf

def get_knowledge_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "consult_internet_for_facts",
                "description": "Search the internet for current facts. Use for queries about people, news, or events.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query. Use 'current' keyword for living people."
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]

class KnowledgeToolExecutor:
    def execute(self, fn_name: str, args: dict) -> str:
        if fn_name == "consult_internet_for_facts":
            return self._search_web(args.get("query"))
        return f"❌ Unknown knowledge tool: {fn_name}"

    def _is_english_result(self, result: dict) -> bool:
        """
        Reliably filters out non-English results without blocking short facts.
        """
        text = (result.get('title', '') + ' ' + result.get('body', '')).lower()
        if not text: return False

        # 1. CJK (Asian) Character Density Check
        # This catches Chinese/Japanese/Korean results immediately
        cjk_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff' or 
                       '\u3040' <= char <= '\u309f' or 
                       '\uac00' <= char <= '\ud7af')
        if len(text) > 0 and (cjk_chars / len(text)) > 0.05:
            return False 

        # 2. Cyrillic (Russian/Kazakh) Character Check
        # If you are in Kazakhstan, DuckDuckGo sends Russian results even with "english" keyword.
        cyrillic_chars = sum(1 for char in text if '\u0400' <= char <= '\u04FF')
        if len(text) > 0 and (cyrillic_chars / len(text)) > 0.05:
            return False

        # 3. Relaxed English Check
        # Just one common word is enough to confirm it's not gibberish
        common_words = ['the', 'is', 'and', 'to', 'in', 'of', 'for', 'with', 'president', 'he', 'she']
        if not any(f" {w} " in text for w in common_words):
            return False 
            
        return True

    def _search_web(self, query: str) -> str:
        if DDGS is None:
            return "❌ SYSTEM: Web search not available. Install duckduckgo-search package."

        # Logic: Only append 'english' if the query is long enough to likely return foreign results.
        # Short financial/weather queries usually work better raw.

        clean_query = query.replace("2025", "").strip()
        
        # Heuristic: If it's about 'price', 'weather', or 'time', DON'T add 'english'
        if any(x in clean_query.lower() for x in ['price', 'weather', 'time', 'usd', 'bitcoin']):
            search_query = clean_query
        else:
            search_query = f"{clean_query} english" if "english" not in clean_query.lower() else clean_query
            
        print(f"   🌍 Searching: '{search_query}'...")

        best_results = []

        try:
            with DDGS() as ddgs:
                # Multi-Region Fallback
                for region in conf.SEARCH_REGIONS:
                    try:
                        raw_results = list(ddgs.text(
                            search_query, 
                            region=region, 
                            safesearch='moderate', 
                            max_results=8
                        ))
                        
                        valid = [r for r in raw_results if self._is_english_result(r)]
                        
                        if valid:
                            best_results = valid[:conf.MAX_SEARCH_RESULTS]
                            break 
                            
                    except Exception as e:
                        print(f"   ⚠️ Region {region} failed: {e}")
                        continue
            
            if not best_results:
                return "❌ SYSTEM: No relevant English results found."

            formatted = f"🔍 SEARCH RESULTS for '{clean_query}':\n"
            for i, res in enumerate(best_results, 1):
                formatted += f"\n[{i}] {res['title']}\n    Source: {res.get('href', 'N/A')}\n    Summary: {res['body']}\n"
            
            formatted += "\n[INSTRUCTION: Answer the user using these facts. Use [1] for citations.]"
            return formatted
            
        except Exception as e:
            print(f"❌ Search Error: {e}")
            return f"❌ Search failed: {str(e)}"