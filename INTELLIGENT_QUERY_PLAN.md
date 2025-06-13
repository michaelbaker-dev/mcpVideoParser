# 🧠 Intelligent Query Understanding System - Implementation Plan

## Current Problems
1. **Rigid pattern matching** - Only recognizes exact phrases
2. **No understanding of time expressions** - "last day", "past week", "since Monday" fail  
3. **No query intent understanding** - Can't infer what user wants
4. **Limited natural language processing** - Misses variations of the same request

## Proposed Architecture: Hybrid Intelligence Layer

```
User Query → LLM Intent Parser → Structured Query → MCP Tools → Results → LLM Response Formatter
```

## 📋 Implementation Plan

### Phase 1: Query Intent Classification
Create an intelligent query parser that uses the chat LLM to understand intent:

```python
class QueryIntentParser:
    """Uses LLM to understand user intent and extract parameters."""
    
    async def parse_query(self, query: str) -> Dict[str, Any]:
        prompt = """You are a query parser for a video analysis system. 
        Extract the intent and parameters from the user's query.
        
        Available intents:
        - search_videos: Search for specific content
        - query_by_time: Query videos by time/date
        - query_by_location: Query videos by location
        - summarize_single: Summarize one video
        - summarize_multiple: Summarize multiple videos
        - general_chat: General conversation
        
        Time expressions to parse:
        - Relative: "last day", "past week", "last 3 hours"
        - Specific: "December 6th", "yesterday", "this morning"
        - Ranges: "between Monday and Friday", "from 9am to 5pm"
        
        Return JSON with:
        {
            "intent": "...",
            "parameters": {
                "time_expression": "original time phrase",
                "location": "location if mentioned",
                "content_query": "what to search for",
                "limit": number or null
            }
        }
        
        User query: {query}
        """
        # Use LLM to parse intent
```

### Phase 2: Enhanced Time Parser
Expand the DateParser to handle more expressions:

```python
class EnhancedDateParser:
    """Parse complex time expressions."""
    
    patterns = {
        "last_n_units": r"(?:last|past)\s+(\d+)\s+(hour|day|week|month)s?",
        "since_day": r"since\s+(monday|tuesday|...)",
        "between_dates": r"between\s+(.+)\s+and\s+(.+)",
        "specific_date": r"on\s+([A-Za-z]+ \d+)",
        # Add more patterns
    }
```

### Phase 3: Intelligent Query Router
Replace rigid pattern matching with LLM-assisted routing:

```python
async def process_query(self, query: str) -> str:
    # Step 1: Use LLM to understand intent
    intent_data = await self.parse_intent(query)
    
    # Step 2: Route based on intent
    if intent_data['intent'] == 'query_by_time':
        # Convert natural language to structured query
        structured_query = await self.convert_to_structured(intent_data)
        result = await self.call_tool('query_location_time', structured_query)
        
    # Step 3: Format response naturally
    return await self.format_response(result, query, intent_data)
```

### Phase 4: Natural Language Query Builder
Help users build complex queries:

```python
class QueryAssistant:
    """Helps users build effective queries."""
    
    async def suggest_query_refinements(self, query: str, results: Any) -> str:
        """Suggest how to refine the query."""
        if not results:
            return "No results found. Try: 'videos from the last week' or 'videos at the shed yesterday'"
```

### Phase 5: Multi-Step Query Handling
Handle complex queries that need multiple steps:

```python
async def handle_complex_query(self, query: str):
    # Example: "Compare what happened at the shed yesterday vs last week"
    # This needs:
    # 1. Query yesterday's shed videos
    # 2. Query last week's shed videos  
    # 3. Compare and summarize differences
```

## 🚀 Immediate Improvements (Quick Wins)

### 1. Expand Time Query Patterns
Add more time patterns to the existing system:

```python
time_patterns = [
    # Existing
    'latest', 'recent', 'new',
    # Add these
    'last day', 'past day', 'last 24 hours',
    'last week', 'past week', 'this week',
    'yesterday', 'today', 'this morning',
    'last month', 'this month',
    # Relative
    'last \d+ days?', 'past \d+ hours?',
    # Specific
    'on \w+', 'since \w+', 'before \w+', 'after \w+'
]
```

### 2. Pre-Process Query with LLM
Before routing, use LLM to normalize the query:

```python
async def normalize_query(self, query: str) -> str:
    """Use LLM to convert natural language to standard form."""
    prompt = f"""Convert this query to a standard form:
    
    User query: "{query}"
    
    Examples:
    - "videos from the last day" → "show videos from last 24 hours"
    - "what happened yesterday" → "show videos from yesterday"
    - "summary of this week's footage" → "summarize videos from this week"
    
    Standardized query:"""
```

### 3. Fallback to LLM Interpretation
When pattern matching fails, use LLM to interpret and execute:

```python
async def llm_interpret_and_execute(self, query: str):
    """Use LLM to interpret query and call appropriate tools."""
    prompt = f"""You have access to these tools:
    - query_location_time(location, time_query)
    - search_videos(query)
    - get_video_summary(video_id)
    
    User wants: "{query}"
    
    What tool should we call and with what parameters?
    Return JSON: {"tool": "...", "params": {...}}"""
```

## 📝 Example Implementation Flow

User: "give me a short summary of the videos the last day"

1. **Intent Parser** → `{"intent": "summarize_multiple", "time_expression": "last day"}`
2. **Time Parser** → Convert "last day" to actual datetime range
3. **Query Execution** → Fetch videos from that time range
4. **Summary Generation** → Use LLM to create natural summary
5. **Response** → "Here's what happened in the last 24 hours..."

## 🎯 Priority Implementation Order

1. **Quick Fix** (Today): Add more patterns to existing matcher
2. **Smart Router** (This Week): Add LLM-based query interpretation 
3. **Time Parser** (Next Week): Enhance date parsing capabilities
4. **Full System** (Month): Implement complete intent-based architecture

## 💡 Key Design Principles

1. **Graceful Degradation**: If LLM interpretation fails, fall back to pattern matching
2. **User Feedback**: Learn from failed queries to improve the system
3. **Transparency**: Show users how their query was interpreted
4. **Flexibility**: Support multiple ways of asking the same thing
5. **Context Awareness**: Remember previous queries in the conversation

## 🔧 Technical Implementation Notes

### Required Components:
- Enhanced DateParser with more regex patterns
- QueryIntentParser class using LLM
- Conversation context manager
- Query suggestion engine
- Response formatter using LLM

### Integration Points:
- Modify `process_query()` in mcp_http_client.py
- Enhance DateParser in date_parser.py
- Add new intent parsing module
- Update MCP tools to accept more flexible parameters

### Testing Strategy:
- Unit tests for each time expression pattern
- Integration tests for intent parsing
- End-to-end tests for complex queries
- User acceptance testing with real queries

## 📊 Success Metrics

1. **Query Success Rate**: % of queries that return relevant results
2. **Intent Accuracy**: % of correctly identified user intents
3. **Time Parsing Coverage**: Number of time expressions supported
4. **User Satisfaction**: Reduction in query reformulation attempts
5. **Response Time**: Maintain sub-2s response for most queries