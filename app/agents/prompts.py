"""Agent prompt templates"""

from langchain.prompts import PromptTemplate, ChatPromptTemplate
from typing import Any


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
- Explain technical concepts clearly""",
    "rag": """You are MOJI RAG Agent, specialized in retrieving and synthesizing information from documents. You:
- Base your answers strictly on the provided context
- Cite specific sources and passages
- Acknowledge when information is not available in the context
- Provide complete and accurate answers without speculation
- Use a structured reasoning approach to ensure accuracy""",
}


# Response format templates
RESPONSE_FORMATS = {
    "structured": ChatPromptTemplate.from_template(
        """
Please provide a structured response with the following format:
1. Summary: Brief overview of the response
2. Details: Detailed explanation or steps
3. Next Steps: Recommended actions or follow-up

User Query: {query}
"""
    ),
    "conversational": ChatPromptTemplate.from_template(
        """
Respond in a natural, conversational manner to: {query}
"""
    ),
    "technical": ChatPromptTemplate.from_template(
        """
Provide a technical response including:
- Problem Analysis
- Solution Approach
- Implementation Details
- Potential Considerations

Technical Query: {query}
"""
    ),
    "rag_structured": ChatPromptTemplate.from_template(
        """
주어진 컨텍스트를 바탕으로 다음 형식에 따라 답변하세요:

질문: {query}

컨텍스트:
{context}

답변 형식:
1. 핵심 답변: 질문에 대한 직접적인 답변
2. 상세 설명: 컨텍스트에서 찾은 구체적인 정보
3. 근거 출처: 답변의 근거가 된 문서와 위치
4. 추가 정보: 관련된 부가 정보 (있는 경우)

중요: 컨텍스트에 없는 정보는 추측하지 말고, 명확히 "제공된 정보에 없음"이라고 표시하세요.
"""
    ),
}


# Chain of thought prompts
COT_PROMPTS = {
    "reasoning": PromptTemplate.from_template(
        """
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
"""
    ),
    "rag_reasoning": PromptTemplate.from_template(
        """
주어진 컨텍스트를 체계적으로 분석하여 답변하겠습니다.

질문: {question}

단계 1: 질문 분석
- 핵심 요구사항: {key_requirements}
- 필요한 정보 유형: {info_type}

단계 2: 컨텍스트 검토
- 관련 정보 발견: {relevant_info}
- 정보의 위치: {info_location}

단계 3: 답변 구성
- 주요 내용: {main_content}
- 추가 세부사항: {details}

단계 4: 검증
- 답변 완전성: {completeness_check}
- 정확성 확인: {accuracy_check}

최종 답변: {final_answer}
출처: {sources}
"""
    ),
    "planning": PromptTemplate.from_template(
        """
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
"""
    ),
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
