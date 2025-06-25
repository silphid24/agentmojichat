"""Agent prompt templates"""

from langchain.prompts import PromptTemplate, ChatPromptTemplate
from typing import Dict, Any


# System prompts for different agent types
SYSTEM_PROMPTS = {
    "general": """You are MOJI, a helpful AI assistant. You provide clear, accurate, and helpful responses while being friendly and professional.""",
    
    "task": """You are MOJI Task Agent, specialized in helping users manage and complete tasks. You:
- Break down complex tasks into manageable steps
- Provide clear action items
- Track progress and provide updates
- Suggest best practices and efficient approaches""",
    
    "knowledge": """You are MOJI Knowledge Agent, specialized in information retrieval and analysis. You:
- Provide accurate and well-researched information
- Cite sources when available
- Explain complex topics clearly
- Acknowledge limitations in your knowledge""",
    
    "creative": """You are MOJI Creative Agent, specialized in creative tasks. You:
- Generate creative content and ideas
- Help with writing, brainstorming, and design concepts
- Provide multiple creative options
- Encourage innovation and originality""",
    
    "technical": """You are MOJI Technical Agent, specialized in technical assistance. You:
- Provide detailed technical explanations
- Help with coding and technical problem-solving
- Follow best practices and industry standards
- Explain technical concepts clearly"""
}


# Response format templates
RESPONSE_FORMATS = {
    "structured": ChatPromptTemplate.from_template("""
Please provide a structured response with the following format:
1. Summary: Brief overview of the response
2. Details: Detailed explanation or steps
3. Next Steps: Recommended actions or follow-up

User Query: {query}
"""),
    
    "conversational": ChatPromptTemplate.from_template("""
Respond in a natural, conversational manner to: {query}
"""),
    
    "technical": ChatPromptTemplate.from_template("""
Provide a technical response including:
- Problem Analysis
- Solution Approach
- Implementation Details
- Potential Considerations

Technical Query: {query}
""")
}


# Chain of thought prompts
COT_PROMPTS = {
    "reasoning": PromptTemplate.from_template("""
Let me think through this step by step:

Question: {question}

Step 1: Understanding the question
{understanding}

Step 2: Breaking down the problem
{breakdown}

Step 3: Analysis
{analysis}

Step 4: Solution
{solution}

Final Answer: {answer}
"""),
    
    "planning": PromptTemplate.from_template("""
To accomplish this task, I'll create a plan:

Task: {task}

Objectives:
{objectives}

Steps:
{steps}

Resources Needed:
{resources}

Timeline:
{timeline}

Success Criteria:
{criteria}
""")
}


def get_agent_prompt(agent_type: str, include_context: bool = True) -> str:
    """Get appropriate prompt for agent type"""
    base_prompt = SYSTEM_PROMPTS.get(agent_type, SYSTEM_PROMPTS["general"])
    
    if include_context:
        context_addon = """

Current Context:
- Time: {current_time}
- User: {user_name}
- Session: {session_id}
- Previous Interactions: {interaction_count}
"""
        return base_prompt + context_addon
    
    return base_prompt


def format_tool_response(tool_name: str, tool_output: Any) -> str:
    """Format tool output for inclusion in agent response"""
    return f"""
Tool Used: {tool_name}
Result: {tool_output}

Based on this information, here's my response:
"""