import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser, PydanticOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.runnables.config import RunnableConfig

import json
import asyncio
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field
import re
from datetime import datetime
import logging
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for structured output
class Severity(str, Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"

class IssueCategory(str, Enum):
    SECURITY = "Security"
    PERFORMANCE = "Performance"
    BUG = "Bug"
    STYLE = "Style"
    MAINTAINABILITY = "Maintainability"

class CodeIssue(BaseModel):
    """Structured code issue model for LangChain output parsing"""
    severity: Severity
    category: IssueCategory
    line_number: int = Field(description="Line number where issue occurs")
    title: str = Field(description="Brief issue title")
    description: str = Field(description="Detailed issue description")
    suggestion: str = Field(description="How to fix the issue")
    code_snippet: Optional[str] = Field(default="", description="Relevant code snippet")
    confidence: float = Field(ge=0.0, le=1.0, description="AI confidence in this issue")

class AnalysisResult(BaseModel):
    """Complete analysis result structure"""
    issues: List[CodeIssue]
    summary: str
    quality_score: float = Field(ge=0.0, le=100.0)
    model_used: str
    analysis_time: float
    recommendations: List[str]
    token_usage: Dict[str, Union[int, float]] = Field(default_factory=dict)

# OpenRouter model configurations
OPENROUTER_MODELS = {
    # Anthropic Models
    "Claude 3.5 Sonnet": {
        "id": "anthropic/claude-3.5-sonnet",
        "context": "200K tokens",
        "cost": "High quality, excellent for code analysis",
        "provider": "Anthropic"
    },
    "Claude 3 Opus": {
        "id": "anthropic/claude-3-opus",
        "context": "200K tokens", 
        "cost": "Premium, best reasoning",
        "provider": "Anthropic"
    },
    "Claude 3 Haiku": {
        "id": "anthropic/claude-3-haiku",
        "context": "200K tokens",
        "cost": "Fast and economical",
        "provider": "Anthropic"
    },
    
    # OpenAI Models
    "GPT-4 Turbo": {
        "id": "openai/gpt-4-turbo",
        "context": "128K tokens",
        "cost": "High quality, good for complex analysis",
        "provider": "OpenAI"
    },
    "GPT-4": {
        "id": "openai/gpt-4",
        "context": "8K tokens",
        "cost": "Premium quality, shorter context",
        "provider": "OpenAI"
    },
    "GPT-3.5 Turbo": {
        "id": "openai/gpt-3.5-turbo",
        "context": "16K tokens",
        "cost": "Economical, good for basic analysis",
        "provider": "OpenAI"
    },
    
    # Google Models
    "Gemini Pro": {
        "id": "google/gemini-pro",
        "context": "128K tokens",
        "cost": "Good balance of quality and cost",
        "provider": "Google"
    },
    "Gemini 2.0 Flash": {
        "id": "google/gemini-2.0-flash-exp",
        "context": "1M tokens",
        "cost": "Experimental, very large context",
        "provider": "Google"
    },
    
    # Meta Models
    "Llama 3.1 70B": {
        "id": "meta-llama/llama-3.1-70b-instruct",
        "context": "128K tokens",
        "cost": "Open source, good performance",
        "provider": "Meta"
    },
    "Llama 3.1 405B": {
        "id": "meta-llama/llama-3.1-405b-instruct",
        "context": "128K tokens",
        "cost": "Largest open model, premium performance",
        "provider": "Meta"
    },
    
    # Specialized Models
    "DeepSeek Coder": {
        "id": "deepseek/deepseek-coder",
        "context": "16K tokens",  
        "cost": "Specialized for code analysis",
        "provider": "DeepSeek"
    },
    "Codestral": {
        "id": "mistralai/codestral-mamba",
        "context": "256K tokens",
        "cost": "Code-focused model",
        "provider": "Mistral"
    }
}

class OpenRouterCallback(BaseCallbackHandler):
    """Custom callback for tracking OpenRouter usage"""
    def __init__(self):
        self.model_calls = []
        self.total_tokens = 0
        self.total_cost = 0.0
        self.current_model = None
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs):
        self.model_calls.append({
            'timestamp': datetime.now(),
            'model': serialized.get('model_name', 'unknown'),
            'prompt_length': sum(len(p) for p in prompts)
        })
        self.current_model = serialized.get('model_name', 'unknown')
    
    def on_llm_end(self, response, **kwargs):
        if hasattr(response, 'llm_output') and response.llm_output:
            usage = response.llm_output.get('token_usage', {})
            tokens = usage.get('total_tokens', 0)
            self.total_tokens += tokens
            
            # Estimate cost (rough estimates)
            cost_estimates = {
                'gpt-4': 0.03 * tokens / 1000,
                'gpt-3.5-turbo': 0.002 * tokens / 1000,
                'claude-3': 0.015 * tokens / 1000,
                'gemini': 0.001 * tokens / 1000
            }
            
            model_family = self.current_model.split('/')[0] if '/' in self.current_model else self.current_model
            estimated_cost = cost_estimates.get(model_family, 0.01 * tokens / 1000)
            self.total_cost += estimated_cost

class OpenRouterLangChainReviewer:
    """LangChain-based code reviewer using OpenRouter API"""
    
    def __init__(self):
        self.llm = None
        self.current_model = None
        self.api_key = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200,
            separators=["\n\n", "\n", "class ", "def ", "function ", "//", "#", "```"]
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.callback = OpenRouterCallback()
        self._setup_parsers()
        self._setup_prompts()
    
    def _setup_parsers(self):
        """Initialize LangChain output parsers"""
        self.json_parser = JsonOutputParser()
        self.pydantic_parser = PydanticOutputParser(pydantic_object=AnalysisResult)
        self.issue_parser = PydanticOutputParser(pydantic_object=CodeIssue)
        self.str_parser = StrOutputParser()
    
    def _setup_prompts(self):
        """Setup LangChain prompt templates"""
        
        # System message for all analyses
        self.system_message = SystemMessage(content="""
        You are an expert senior software engineer and security auditor with 15+ years of experience.
        You specialize in code review, security analysis, and performance optimization across multiple languages.
        
        Your analysis should be:
        - Thorough but practical and actionable
        - Security-focused when vulnerabilities are present
        - Performance-aware for optimization opportunities
        - Maintainability-oriented for long-term code health
        
        Always provide specific, actionable suggestions with concrete examples when possible.
        Focus on high-impact issues that improve code quality, security, and performance.
        """)
        
        # Main comprehensive analysis prompt
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            self.system_message,
            HumanMessage(content="""
            Analyze this {language} code for issues. You MUST return a valid JSON array only, no other text.

            CODE TO ANALYZE:
            ```{language}
            {code}
            ```

            Return ONLY a JSON array of issues in this exact format:
            [
                {{
                    "severity": "Critical",
                    "category": "Security",
                    "line_number": 1,
                    "title": "Brief issue title",
                    "description": "Detailed description",
                    "suggestion": "How to fix",
                    "confidence": 0.9
                }}
            ]

            If no issues found, return: []

            Focus on:
            - Security vulnerabilities
            - Performance problems  
            - Logic bugs
            - Code quality issues

            Return ONLY valid JSON, no explanatory text before or after.
            """)
        ])
        
        # Specialized security analysis prompt
        self.security_prompt = ChatPromptTemplate.from_messages([
            self.system_message,
            HumanMessage(content="""
            Perform security analysis on this {language} code. Return ONLY valid JSON array.

            CODE:
            ```{language}
            {code}
            ```

            Look for:
            - SQL injection, XSS, CSRF
            - Authentication bypass
            - Input validation issues
            - Cryptographic problems
            - Data exposure

            Return ONLY JSON array format:
            [
                {{
                    "severity": "Critical",
                    "category": "Security",
                    "line_number": 1,
                    "title": "Security issue title",
                    "description": "Detailed security description",
                    "suggestion": "Security fix suggestion",
                    "confidence": 0.95
                }}
            ]

            If no security issues: []
            """)
        ])
        
        # Performance-focused analysis prompt
        self.performance_prompt = ChatPromptTemplate.from_messages([
            self.system_message,
            HumanMessage(content="""
            Analyze performance of this {language} code. Return ONLY valid JSON array.

            CODE:
            ```{language}
            {code}
            ```

            Look for:
            - Inefficient algorithms
            - Memory leaks
            - I/O bottlenecks
            - Caching opportunities

            Return ONLY JSON array:
            [
                {{
                    "severity": "Medium",
                    "category": "Performance",
                    "line_number": 1,
                    "title": "Performance issue",
                    "description": "Performance problem description", 
                    "suggestion": "Optimization suggestion",
                    "confidence": 0.8
                }}
            ]

            If no performance issues: []
            """)
        ])
        
        # Summary generation prompt
        self.summary_prompt = ChatPromptTemplate.from_messages([
            self.system_message,
            HumanMessage(content="""
            Generate a comprehensive executive summary based on this code analysis:
            
            **Analysis Results:**
            {issues_summary}
            
            **Code Statistics:**
            - Language: {language}
            - Lines of code: {loc}
            - Total issues found: {total_issues}
            - Model used: {model_name}
            
            Provide a professional summary including:
            1. **Overall Assessment** (2-3 sentences on code quality)
            2. **Critical Issues** (immediate security/stability concerns)
            3. **Priority Actions** (top 3 most important fixes)
            4. **Quality Score Justification** (why this score out of 100)
            5. **Improvement Roadmap** (strategic recommendations)
            
            Be specific, actionable, and business-focused in your recommendations.
            """)
        ])
    
    def setup_model(self, model_name: str, api_key: str, temperature: float = 0.1) -> bool:
        """Setup OpenRouter model with LangChain"""
        try:
            self.api_key = api_key
            model_id = OPENROUTER_MODELS[model_name]["id"]
            
            # Create ChatOpenAI instance configured for OpenRouter
            self.llm = ChatOpenAI(
                model=model_id,
                temperature=temperature,
                openai_api_key=api_key,
                openai_api_base="https://openrouter.ai/api/v1",
                max_tokens=4000,
                model_kwargs={
                    "extra_headers": {
                        "HTTP-Referer": "https://streamlit-langchain-code-review.com",
                        "X-Title": "LangChain Code Review Agent"
                    }
                },
                callbacks=[self.callback]
            )
            
            # Test the model
            test_chain = self.llm | self.str_parser
            test_response = test_chain.invoke("Respond with 'Model ready' if you can analyze code.")
            
            if len(test_response) > 5:
                self.current_model = model_name
                logger.info(f"Successfully configured {model_name}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to setup {model_name}: {e}")
            return False
    
    def get_model_info(self, model_name: str) -> Dict[str, str]:
        """Get information about a specific model"""
        return OPENROUTER_MODELS.get(model_name, {})
    
    def analyze_code_comprehensive(self, code: str, language: str) -> AnalysisResult:
        """Comprehensive code analysis using LangChain chains"""
        if not self.llm:
            raise ValueError("No model configured")
        
        start_time = datetime.now()
        
        try:
            # Check if code needs chunking
            if len(code) > 10000:
                return self._analyze_large_code(code, language)
            
            # Create analysis chain using LCEL
            analysis_chain = (
                self.analysis_prompt
                | self.llm 
                | self.str_parser  # Use string parser first, then parse JSON manually
            )
            
            # Run analysis with better error handling
            try:
                raw_result = analysis_chain.invoke({
                    "code": code,
                    "language": language
                })
                
                # Try to parse as JSON
                try:
                    import json
                    result = json.loads(raw_result)
                except json.JSONDecodeError:
                    # Extract JSON from response if wrapped in text
                    json_match = re.search(r'\[.*\]', raw_result, re.DOTALL)
                    if json_match:
                        result = json.loads(json_match.group(0))
                    else:
                        # Fallback to text parsing
                        result = self._parse_text_to_issues(raw_result, code)
                        
            except Exception as e:
                logger.warning(f"Chain execution failed: {e}")
                result = []
            
            # Parse issues with error handling
            issues = []
            if isinstance(result, list):
                for issue_data in result:
                    try:
                        # Add code snippet if line number is valid
                        if isinstance(issue_data, dict):
                            line_num = issue_data.get('line_number', 0)
                            if line_num > 0:
                                code_lines = code.split('\n')
                                if line_num <= len(code_lines):
                                    issue_data['code_snippet'] = code_lines[line_num - 1].strip()
                        
                        issue = CodeIssue(**issue_data)
                        issues.append(issue)
                    except Exception as e:
                        logger.warning(f"Failed to parse issue: {e}")
                        continue
            elif isinstance(result, str):
                # Handle case where result is still a string
                issues = self._parse_text_to_issues(result, code)
            
            # Generate summary using another chain
            summary = self._generate_summary(issues, code, language)
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(issues, len(code.split('\n')))
            
            # Generate recommendations
            recommendations = self._generate_recommendations(issues)
            
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            return AnalysisResult(
                issues=issues,
                summary=summary,
                quality_score=quality_score,
                model_used=f"{self.current_model} (via OpenRouter)",
                analysis_time=analysis_time,
                recommendations=recommendations,
                token_usage={
                    "total_tokens": self.callback.total_tokens,
                    "estimated_cost": round(self.callback.total_cost, 4)
                }
            )
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            raise
    
    def analyze_security_focused(self, code: str, language: str) -> List[CodeIssue]:
        """Security-focused analysis using specialized chain"""
        if not self.llm:
            raise ValueError("No model configured")
        
        try:
            # Create security analysis chain
            security_chain = (
                self.security_prompt
                | self.llm
                | self.str_parser
            )
            
            raw_result = security_chain.invoke({
                "code": code,
                "language": language
            })
            
            # Parse JSON from response
            try:
                import json
                result = json.loads(raw_result)
            except json.JSONDecodeError:
                json_match = re.search(r'\[.*\]', raw_result, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(0))
                else:
                    result = self._parse_text_to_issues(raw_result, code)
            
            issues = []
            if isinstance(result, list):
                for issue_data in result:
                    try:
                        if isinstance(issue_data, dict):
                            line_num = issue_data.get('line_number', 0)
                            if line_num > 0:
                                code_lines = code.split('\n')
                                if line_num <= len(code_lines):
                                    issue_data['code_snippet'] = code_lines[line_num - 1].strip()
                        
                        issue = CodeIssue(**issue_data)
                        if issue.category == IssueCategory.SECURITY:
                            issues.append(issue)
                    except Exception as e:
                        logger.warning(f"Failed to parse security issue: {e}")
            
            return issues
            
        except Exception as e:
            logger.error(f"Security analysis failed: {e}")
            return []
    
    def analyze_performance_focused(self, code: str, language: str) -> List[CodeIssue]:
        """Performance-focused analysis using specialized chain"""
        if not self.llm:
            raise ValueError("No model configured")
        
        try:
            performance_chain = (
                self.performance_prompt
                | self.llm
                | self.str_parser
            )
            
            raw_result = performance_chain.invoke({
                "code": code,
                "language": language
            })
            
            # Parse JSON from response
            try:
                import json
                result = json.loads(raw_result)
            except json.JSONDecodeError:
                json_match = re.search(r'\[.*\]', raw_result, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(0))
                else:
                    result = self._parse_text_to_issues(raw_result, code)
            
            issues = []
            if isinstance(result, list):
                for issue_data in result:
                    try:
                        if isinstance(issue_data, dict):
                            line_num = issue_data.get('line_number', 0)
                            if line_num > 0:
                                code_lines = code.split('\n')
                                if line_num <= len(code_lines):
                                    issue_data['code_snippet'] = code_lines[line_num - 1].strip()
                        
                        issue = CodeIssue(**issue_data)
                        if issue.category == IssueCategory.PERFORMANCE:
                            issues.append(issue)
                    except Exception as e:
                        logger.warning(f"Failed to parse performance issue: {e}")
            
            return issues
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            return []
    
    def _analyze_large_code(self, code: str, language: str) -> AnalysisResult:
        """Handle large code files using LangChain text splitter"""
        start_time = datetime.now()
        
        # Split code into manageable chunks
        documents = [Document(page_content=code)]
        chunks = self.text_splitter.split_documents(documents)
        
        st.info(f"📄 Large codebase detected! Processing {len(chunks)} chunks with {self.current_model}")
        
        all_issues = []
        progress_bar = st.progress(0)
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            try:
                analysis_chain = (
                    self.analysis_prompt
                    | self.llm
                    | self.json_parser
                )
                
                result = analysis_chain.invoke({
                    "code": chunk.page_content,
                    "language": language
                })
                
                if isinstance(result, list):
                    for issue_data in result:
                        try:
                            if isinstance(issue_data, dict):
                                line_num = issue_data.get('line_number', 0)
                                if line_num > 0:
                                    chunk_lines = chunk.page_content.split('\n')
                                    if line_num <= len(chunk_lines):
                                        issue_data['code_snippet'] = chunk_lines[line_num - 1].strip()
                            
                            issue = CodeIssue(**issue_data)
                            all_issues.append(issue)
                        except Exception as e:
                            logger.warning(f"Failed to parse issue in chunk {i+1}: {e}")
                
                progress_bar.progress((i + 1) / len(chunks))
                
            except Exception as e:
                logger.error(f"Chunk {i+1} analysis failed: {e}")
                continue
        
        progress_bar.empty()
        
        # Generate comprehensive summary
        summary = self._generate_summary(all_issues, code, language)
        quality_score = self._calculate_quality_score(all_issues, len(code.split('\n')))
        recommendations = self._generate_recommendations(all_issues)
        
        analysis_time = (datetime.now() - start_time).total_seconds()
        
        return AnalysisResult(
            issues=all_issues,
            summary=summary,
            quality_score=quality_score,
            model_used=f"{self.current_model} (via OpenRouter)",
            analysis_time=analysis_time,
            recommendations=recommendations,
            token_usage={
                "total_tokens": self.callback.total_tokens,
                "estimated_cost": round(self.callback.total_cost, 4)
            }
        )
    
    def _generate_summary(self, issues: List[CodeIssue], code: str, language: str) -> str:
        """Generate analysis summary using LangChain chain"""
        try:
            issues_summary = "\n".join([
                f"- {issue.severity.value}: {issue.title} (Line {issue.line_number}) - {issue.category.value}"
                for issue in issues[:20]  # Limit for prompt size
            ])
            
            summary_chain = self.summary_prompt | self.llm | self.str_parser
            
            summary = summary_chain.invoke({
                "issues_summary": issues_summary if issues_summary else "No issues found",
                "language": language,
                "loc": len(code.split('\n')),
                "total_issues": len(issues),
                "model_name": self.current_model
            })
            
            return summary
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return f"Analysis completed using {self.current_model}. Found {len(issues)} issues across multiple categories. Review the detailed findings below for specific recommendations."
    
    def _calculate_quality_score(self, issues: List[CodeIssue], lines_of_code: int) -> float:
        """Calculate code quality score based on issues found"""
        if not issues:
            return 95.0
        
        # Severity-based penalties
        severity_weights = {
            Severity.CRITICAL: 25,
            Severity.HIGH: 15,
            Severity.MEDIUM: 8,
            Severity.LOW: 3
        }
        
        total_penalty = sum(severity_weights.get(issue.severity, 0) for issue in issues)
        
        # Adjust for code size (penalty per line)
        penalty_per_line = total_penalty / max(lines_of_code, 1)
        
        # Calculate final score (0-100)
        base_score = 100
        final_score = max(0, base_score - (penalty_per_line * 100))
        
        return round(final_score, 1)
    
    def _parse_text_to_issues(self, text: str, code: str) -> List[CodeIssue]:
        """Parse text response into CodeIssue objects when JSON parsing fails"""
        issues = []
        lines = text.split('\n')
        code_lines = code.split('\n')
        
        current_issue = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for severity indicators
            if any(sev in line.upper() for sev in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']):
                # Save previous issue if exists
                if current_issue:
                    try:
                        issue = self._create_issue_from_dict(current_issue, code_lines)
                        if issue:
                            issues.append(issue)
                    except:
                        pass
                    current_issue = {}
                
                # Start new issue
                if 'CRITICAL' in line.upper():
                    current_issue['severity'] = 'Critical'
                elif 'HIGH' in line.upper():
                    current_issue['severity'] = 'High'
                elif 'MEDIUM' in line.upper():
                    current_issue['severity'] = 'Medium'
                else:
                    current_issue['severity'] = 'Low'
                
                # Extract line number
                line_match = re.search(r'line\s+(\d+)', line, re.IGNORECASE)
                if line_match:
                    current_issue['line_number'] = int(line_match.group(1))
                else:
                    current_issue['line_number'] = 0
                
                # Extract title/description
                current_issue['title'] = line[:50] + "..." if len(line) > 50 else line
                current_issue['description'] = line
                current_issue['suggestion'] = "Review and fix this issue"
                current_issue['confidence'] = 0.8
                
                # Determine category
                if any(word in line.upper() for word in ['SECURITY', 'INJECTION', 'XSS', 'VULNERABILITY']):
                    current_issue['category'] = 'Security'
                elif any(word in line.upper() for word in ['PERFORMANCE', 'SLOW', 'OPTIMIZE', 'MEMORY']):
                    current_issue['category'] = 'Performance'
                elif any(word in line.upper() for word in ['BUG', 'ERROR', 'EXCEPTION', 'NULL']):
                    current_issue['category'] = 'Bug'
                elif any(word in line.upper() for word in ['STYLE', 'FORMAT', 'NAMING']):
                    current_issue['category'] = 'Style'
                else:
                    current_issue['category'] = 'Maintainability'
        
        # Add last issue
        if current_issue:
            try:
                issue = self._create_issue_from_dict(current_issue, code_lines)
                if issue:
                    issues.append(issue)
            except:
                pass
        
        return issues
    
    def _create_issue_from_dict(self, issue_dict: dict, code_lines: list) -> Optional[CodeIssue]:
        """Create CodeIssue from dictionary with validation"""
        try:
            # Add code snippet if line number is valid
            line_num = issue_dict.get('line_number', 0)
            if line_num > 0 and line_num <= len(code_lines):
                issue_dict['code_snippet'] = code_lines[line_num - 1].strip()
            else:
                issue_dict['code_snippet'] = ""
            
            return CodeIssue(**issue_dict)
        except Exception as e:
            logger.warning(f"Failed to create issue: {e}")
            return None
        """Generate prioritized recommendations based on issues"""
        recommendations = []
        
        # Categorize issues
        critical_issues = [i for i in issues if i.severity == Severity.CRITICAL]
        high_issues = [i for i in issues if i.severity == Severity.HIGH]
        security_issues = [i for i in issues if i.category == IssueCategory.SECURITY]
        performance_issues = [i for i in issues if i.category == IssueCategory.PERFORMANCE]
        
        # Generate contextual recommendations
        if critical_issues:
            recommendations.append(f"🚨 **URGENT**: Address {len(critical_issues)} critical issues immediately - potential security/stability risks")
        
        if security_issues:
            recommendations.append(f"🔒 **SECURITY**: Review {len(security_issues)} security vulnerabilities before deployment")
        
        if high_issues:
            recommendations.append(f"⚠️ **HIGH PRIORITY**: Fix {len(high_issues)} high-severity issues in next sprint")
        
        if performance_issues:
            recommendations.append(f"⚡ **PERFORMANCE**: Optimize {len(performance_issues)} performance bottlenecks for better user experience")
        
        if len(issues) > 30:
            recommendations.append("🔧 **REFACTORING**: Consider architectural review - high issue density suggests design problems")
        
        if not recommendations:
            recommendations.append("✅ **EXCELLENT**: Code quality is high, continue following current practices")
        
        return recommendations

    def _generate_recommendations(self, issues: List[CodeIssue]) -> List[str]:
        """Generate prioritized recommendations based on issues"""
        recommendations = []
        
        # Categorize issues
        critical_issues = [i for i in issues if i.severity == Severity.CRITICAL]
        high_issues = [i for i in issues if i.severity == Severity.HIGH]
        security_issues = [i for i in issues if i.category == IssueCategory.SECURITY]
        performance_issues = [i for i in issues if i.category == IssueCategory.PERFORMANCE]
        
        # Generate contextual recommendations
        if critical_issues:
            recommendations.append(f"🚨 **URGENT**: Address {len(critical_issues)} critical issues immediately - potential security/stability risks")
        
        if security_issues:
            recommendations.append(f"🔒 **SECURITY**: Review {len(security_issues)} security vulnerabilities before deployment")
        
        if high_issues:
            recommendations.append(f"⚠️ **HIGH PRIORITY**: Fix {len(high_issues)} high-severity issues in next sprint")
        
        if performance_issues:
            recommendations.append(f"⚡ **PERFORMANCE**: Optimize {len(performance_issues)} performance bottlenecks for better user experience")
        
        if len(issues) > 30:
            recommendations.append("🔧 **REFACTORING**: Consider architectural review - high issue density suggests design problems")
        
        if not recommendations:
            recommendations.append("✅ **EXCELLENT**: Code quality is high, continue following current practices")
        
        return recommendations

def main():
    st.set_page_config(
        page_title="LangChain + OpenRouter Code Review",
        page_icon="🔗",
        layout="wide"
    )
    
    st.title("🔗 LangChain Multi-Model Code Review")
    st.caption("**Powered by OpenRouter API** - Access 20+ AI models through LangChain")
    
    # Model showcase info
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.info("🤖 **12+ Models** - Claude, GPT-4, Gemini, Llama")
    with col_info2:
        st.info("🔗 **LangChain Powered** - Structured chains & parsing")
    with col_info3:
        st.info("💰 **Cost Tracking** - Monitor usage across models")
    
    # Initialize session state
    if 'reviewer' not in st.session_state:
        st.session_state.reviewer = OpenRouterLangChainReviewer()
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    
    reviewer = st.session_state.reviewer
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔑 OpenRouter Configuration")
        
        # API Key input
        api_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            help="Get your API key from https://openrouter.ai/keys"
        )
        
        if api_key:
            st.success("✅ API Key configured")
        else:
            st.warning("⚠️ Enter OpenRouter API key to continue")
        
        st.divider()
        
        # Model selection with detailed info
        st.header("🤖 Model Selection")
        
        # Group models by provider
        providers = {}
        for model_name, info in OPENROUTER_MODELS.items():
            provider = info["provider"]
            if provider not in providers:
                providers[provider] = []
            providers[provider].append(model_name)
        
        # Provider selection
        selected_provider = st.selectbox(
            "AI Provider",
            list(providers.keys()),
            help="Choose your preferred AI provider"
        )
        
        # Model selection within provider
        available_models = providers[selected_provider]
        selected_model = st.selectbox(
            "Model",
            available_models,
            help="Select specific model variant"
        )
        
        # Show model details
        if selected_model:
            model_info = reviewer.get_model_info(selected_model)
            st.info(f"**Context:** {model_info.get('context', 'Unknown')}")
            st.info(f"**Notes:** {model_info.get('cost', 'Standard pricing')}")
        
        # Model setup
        if api_key and selected_model:
            if st.button("🚀 Initialize Model"):
                with st.spinner(f"Setting up {selected_model}..."):
                    success = reviewer.setup_model(
                        model_name=selected_model,
                        api_key=api_key,
                        temperature=0.1
                    )
                    
                    if success:
                        st.success(f"✅ {selected_model} ready!")
                        st.session_state.model_ready = True
                    else:
                        st.error("❌ Model setup failed")
                        st.session_state.model_ready = False
        
        # Show active model
        if reviewer.current_model:
            st.success(f"🤖 **Active:** {reviewer.current_model}")
            
            # Usage statistics
            if reviewer.callback.model_calls:
                st.metric("API Calls", len(reviewer.callback.model_calls))
                st.metric("Total Tokens", f"{reviewer.callback.total_tokens:,}")
                if reviewer.callback.total_cost > 0:
                    st.metric("Est. Cost", f"${reviewer.callback.total_cost:.4f}")
        
        st.divider()
        
        # Analysis configuration
        st.header("📊 Analysis Settings")
        
        language = st.selectbox(
            "Programming Language",
            ["Python", "JavaScript", "TypeScript", "Java", "C++", "Go", "Rust", 
             "PHP", "C#", "Swift", "Kotlin", "Ruby", "Scala", "SQL"],
            help="Select the primary language of your code"
        )
        
        analysis_mode = st.selectbox(
            "Analysis Mode",
            ["🔍 Comprehensive", "🔒 Security Focused", "⚡ Performance Focused"],
            help="Choose analysis depth and focus area"
        )
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            temperature = st.slider(
                "Model Temperature", 
                0.0, 1.0, 0.1, 0.1,
                help="Lower = more focused, Higher = more creative"
            )
            
            confidence_threshold = st.slider(
                "Confidence Threshold",
                0.0, 1.0, 0.6, 0.1,
                help="Filter out low-confidence issues"
            )
            
            max_issues_display = st.selectbox(
                "Max Issues to Display",
                [10, 25, 50, 100, "All"],
                index=2
            )
            
            chunk_size = st.slider(
                "Chunk Size (for large files)",
                2000, 8000, 4000, 500,
                help="Smaller chunks = more detailed analysis"
            )
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Code Input")
        
        # Input method tabs
        tab1, tab2, tab3 = st.tabs(["📝 Text Editor", "📁 File Upload", "📂 Multiple Files"])
        
        code_content = ""
        file_info = {}
        
        with tab1:
            code_content = st.text_area(
                "Enter your code:",
                height=400,
                placeholder=f"""// Paste your {language} code here
// LangChain + OpenRouter will analyze it with advanced AI models

def example_function():
    # Your code here
    pass""",
                help="Paste code directly for immediate analysis"
            )
        
        with tab2:
            uploaded_file = st.file_uploader(
                "Upload single code file",
                type=['py', 'js', 'ts', 'java', 'cpp', 'go', 'rs', 'php', 'cs', 'rb', 'scala', 'kt', 'swift', 'txt'],
                help="Upload a code file for analysis"
            )
            
            if uploaded_file:
                code_content = str(uploaded_file.read(), "utf-8")
                lines = len(code_content.split('\n'))
                chars = len(code_content)
                
                file_info = {
                    'name': uploaded_file.name,
                    'lines': lines,
                    'size': chars,
                    'estimated_tokens': chars // 4
                }
                
                st.success(f"📁 **{uploaded_file.name}** loaded successfully!")
                
                col_f1, col_f2, col_f3 = st.columns(3)
                with col_f1:
                    st.metric("Lines", f"{lines:,}")
                with col_f2:
                    st.metric("Characters", f"{chars:,}")
                with col_f3:
                    st.metric("Est. Tokens", f"{chars//4:,}")
                
                with st.expander("📄 File Preview"):
                    preview_length = min(2000, len(code_content))
                    st.code(
                        code_content[:preview_length] + ("..." if len(code_content) > preview_length else ""),
                        language=language.lower()
                    )
        
        with tab3:
            uploaded_files = st.file_uploader(
                "Upload multiple code files",
                type=['py', 'js', 'ts', 'java', 'cpp', 'go', 'rs', 'php', 'cs', 'rb', 'scala', 'kt', 'swift', 'txt'],
                accept_multiple_files=True,
                help="Upload multiple files to analyze as a combined codebase"
            )
            
            if uploaded_files:
                combined_code = []
                total_lines = 0
                total_chars = 0
                
                for file in uploaded_files:
                    file_content = str(file.read(), "utf-8")
                    file_lines = len(file_content.split('\n'))
                    
                    combined_code.append(f"""
# ===== FILE: {file.name} =====
# Lines: {file_lines}
{file_content}
""")
                    total_lines += file_lines
                    total_chars += len(file_content)
                
                code_content = "\n".join(combined_code)
                
                file_info = {
                    'files': len(uploaded_files),
                    'total_lines': total_lines,
                    'total_chars': total_chars,
                    'estimated_tokens': total_chars // 4
                }
                
                st.success(f"📂 **{len(uploaded_files)} files** combined successfully!")
                
                col_mf1, col_mf2, col_mf3 = st.columns(3)
                with col_mf1:
                    st.metric("Files", len(uploaded_files))
                with col_mf2:
                    st.metric("Total Lines", f"{total_lines:,}")
                with col_mf3:
                    st.metric("Est. Tokens", f"{total_chars//4:,}")
                
                with st.expander("📋 Files Included"):
                    for file in uploaded_files:
                        st.write(f"• {file.name}")
        
        # Analysis execution
        can_analyze = (code_content and 
                      reviewer.current_model and 
                      api_key)
        
        # Dynamic button text
        if not api_key:
            button_text = "🔑 Enter OpenRouter API Key First"
        elif not reviewer.current_model:
            button_text = "🤖 Initialize Model First"
        elif not code_content:
            button_text = "📝 Add Code to Analyze"
        else:
            estimated_tokens = len(code_content) // 4
            if estimated_tokens > 50000:
                button_text = f"🚀 Analyze Large Codebase ({estimated_tokens:,} tokens)"
            else:
                button_text = f"🚀 Start {analysis_mode.split()[1]} Analysis"
        
        if st.button(button_text, type="primary", disabled=not can_analyze):
            with st.spinner(f"🔍 Running {analysis_mode} with {reviewer.current_model}..."):
                try:
                    # Update model temperature if changed
                    if hasattr(reviewer.llm, 'temperature'):
                        reviewer.llm.temperature = temperature
                    
                    # Run appropriate analysis
                    if analysis_mode == "🔍 Comprehensive":
                        result = reviewer.analyze_code_comprehensive(code_content, language)
                        
                    elif analysis_mode == "🔒 Security Focused":
                        issues = reviewer.analyze_security_focused(code_content, language)
                        
                        # Create result object for security analysis
                        result = AnalysisResult(
                            issues=issues,
                            summary=f"Security analysis completed with {reviewer.current_model}. Found {len(issues)} potential security vulnerabilities requiring attention.",
                            quality_score=max(50, 100 - len([i for i in issues if i.severity == Severity.CRITICAL]) * 15 - len([i for i in issues if i.severity == Severity.HIGH]) * 8),
                            model_used=f"{reviewer.current_model} (Security Focus)",
                            analysis_time=1.0,
                            recommendations=[f"🔒 Review and fix {len(issues)} security issues"],
                            token_usage={
                                "total_tokens": reviewer.callback.total_tokens,
                                "estimated_cost": round(reviewer.callback.total_cost, 4)
                            }
                        )
                        
                    else:  # Performance Focused
                        issues = reviewer.analyze_performance_focused(code_content, language)
                        
                        result = AnalysisResult(
                            issues=issues,
                            summary=f"Performance analysis completed with {reviewer.current_model}. Identified {len(issues)} optimization opportunities to improve application performance.",
                            quality_score=max(60, 100 - len([i for i in issues if i.severity == Severity.HIGH]) * 10 - len([i for i in issues if i.severity == Severity.MEDIUM]) * 5),
                            model_used=f"{reviewer.current_model} (Performance Focus)",
                            analysis_time=1.0,
                            recommendations=[f"⚡ Optimize {len(issues)} performance bottlenecks"],
                            token_usage={
                                "total_tokens": reviewer.callback.total_tokens,
                                "estimated_cost": round(reviewer.callback.total_cost, 4)
                            }
                        )
                    
                    # Apply confidence filtering
                    if confidence_threshold > 0:
                        original_count = len(result.issues)
                        result.issues = [i for i in result.issues if i.confidence >= confidence_threshold]
                        filtered_count = len(result.issues)
                        
                        if filtered_count < original_count:
                            st.info(f"🎯 Filtered {original_count - filtered_count} low-confidence issues (< {confidence_threshold})")
                    
                    # Limit display results
                    if max_issues_display != "All" and len(result.issues) > int(max_issues_display):
                        result.issues = result.issues[:int(max_issues_display)]
                        st.info(f"📊 Showing top {max_issues_display} issues (use sidebar to show more)")
                    
                    st.session_state.analysis_results = result
                    st.session_state.original_code = code_content
                    st.session_state.file_info = file_info
                    
                    # Success message with metrics
                    success_msg = f"✅ **Analysis Complete!** Found {len(result.issues)} issues"
                    if result.token_usage.get('total_tokens'):
                        success_msg += f" • Used {result.token_usage['total_tokens']:,} tokens"
                    if result.token_usage.get('estimated_cost'):
                        success_msg += f" • Est. cost: ${result.token_usage['estimated_cost']:.4f}"
                    
                    st.success(success_msg)
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    st.info("💡 Try a different model or check your API key")
    
    with col2:
        st.header("📊 Analysis Results")
        
        if st.session_state.analysis_results:
            result = st.session_state.analysis_results
            
            # Quality score with color coding
            quality_score = result.quality_score
            if quality_score >= 85:
                score_color = "🟢"
                score_status = "Excellent"
            elif quality_score >= 70:
                score_color = "🟡"
                score_status = "Good"
            elif quality_score >= 50:
                score_color = "🟠"
                score_status = "Needs Improvement"
            else:
                score_color = "🔴"
                score_status = "Critical Issues"
            
            # Metrics dashboard
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            
            with col_m1:
                st.metric(
                    "Quality Score", 
                    f"{score_color} {quality_score:.1f}/100",
                    help=f"Overall code quality: {score_status}"
                )
            
            with col_m2:
                critical_count = len([i for i in result.issues if i.severity == Severity.CRITICAL])
                st.metric(
                    "Critical Issues", 
                    critical_count,
                    delta=f"-{critical_count}" if critical_count > 0 else "0",
                    delta_color="inverse"
                )
            
            with col_m3:
                high_count = len([i for i in result.issues if i.severity == Severity.HIGH])
                st.metric(
                    "High Priority", 
                    high_count,
                    delta=f"-{high_count}" if high_count > 0 else "0",
                    delta_color="inverse"
                )
            
            with col_m4:
                st.metric("Total Issues", len(result.issues))
            
            # Model and performance info
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.info(f"🤖 **Model:** {result.model_used}")
            with col_info2:
                st.info(f"⏱️ **Time:** {result.analysis_time:.1f}s")
            
            # Token usage info
            if result.token_usage:
                col_usage1, col_usage2 = st.columns(2)
                with col_usage1:
                    if result.token_usage.get('total_tokens'):
                        st.info(f"🔢 **Tokens:** {result.token_usage['total_tokens']:,}")
                with col_usage2:
                    if result.token_usage.get('estimated_cost'):
                        st.info(f"💰 **Est. Cost:** ${result.token_usage['estimated_cost']:.4f}")
            
            # Executive summary
            st.subheader("📋 Executive Summary")
            st.write(result.summary)
            
            # Key recommendations
            if result.recommendations:
                st.subheader("💡 Priority Recommendations")
                for i, rec in enumerate(result.recommendations):
                    if i < 5:  # Limit to top 5 recommendations
                        st.info(rec)
            
            # Issues visualization
            if result.issues:
                st.subheader("📊 Issues Breakdown")
                
                # Create severity and category charts
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    severity_counts = {}
                    for issue in result.issues:
                        severity = issue.severity.value
                        severity_counts[severity] = severity_counts.get(severity, 0) + 1
                    
                    if severity_counts:
                        st.bar_chart(
                            severity_counts,
                            use_container_width=True,
                            height=200
                        )
                        st.caption("Issues by Severity")
                
                with col_chart2:
                    category_counts = {}
                    for issue in result.issues:
                        category = issue.category.value
                        category_counts[category] = category_counts.get(category, 0) + 1
                    
                    if category_counts:
                        st.bar_chart(
                            category_counts,
                            use_container_width=True,
                            height=200
                        )
                        st.caption("Issues by Category")
        
        else:
            st.info("👆 Configure OpenRouter, select a model, and analyze code to see results")
            
            # Feature showcase
            with st.expander("🌟 LangChain + OpenRouter Features"):
                st.markdown("""
                **🔗 LangChain Integration:**
                - Structured prompt templates with ChatPromptTemplate
                - Type-safe parsing with Pydantic models
                - Chain composition using LCEL (LangChain Expression Language)
                - Automatic text splitting for large codebases
                - Memory and conversation tracking
                - Custom callbacks for usage monitoring
                
                **🤖 Multi-Model Support via OpenRouter:**
                - **Anthropic:** Claude 3.5 Sonnet, Opus, Haiku
                - **OpenAI:** GPT-4 Turbo, GPT-4, GPT-3.5 Turbo
                - **Google:** Gemini Pro, Gemini 2.0 Flash
                - **Meta:** Llama 3.1 70B, 405B
                - **Specialized:** DeepSeek Coder, Codestral
                
                **💰 Cost & Usage Tracking:**
                - Real-time token usage monitoring
                - Cost estimation across different models
                - API call tracking and performance metrics
                """)
    
    # Detailed issues section
    if st.session_state.analysis_results and st.session_state.analysis_results.issues:
        st.header("🔍 Detailed Issues Analysis")
        
        issues = st.session_state.analysis_results.issues
        
        # Filtering options
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        
        with col_filter1:
            severity_filter = st.multiselect(
                "Filter by Severity",
                [s.value for s in Severity],
                default=[s.value for s in Severity],
                key="severity_filter"
            )
        
        with col_filter2:
            category_filter = st.multiselect(
                "Filter by Category",
                [c.value for c in IssueCategory],
                default=[c.value for c in IssueCategory],
                key="category_filter"
            )
        
        with col_filter3:
            sort_by = st.selectbox(
                "Sort by",
                ["Severity", "Confidence", "Line Number", "Category"],
                key="sort_by"
            )
        
        # Apply filters
        filtered_issues = [
            issue for issue in issues
            if (issue.severity.value in severity_filter and 
                issue.category.value in category_filter)
        ]
        
        # Sort issues
        if sort_by == "Severity":
            severity_order = {Severity.CRITICAL: 0, Severity.HIGH: 1, Severity.MEDIUM: 2, Severity.LOW: 3}
            filtered_issues.sort(key=lambda x: severity_order.get(x.severity, 4))
        elif sort_by == "Confidence":
            filtered_issues.sort(key=lambda x: x.confidence, reverse=True)
        elif sort_by == "Line Number":
            filtered_issues.sort(key=lambda x: x.line_number)
        else:  # Category
            filtered_issues.sort(key=lambda x: x.category.value)
        
        st.write(f"**Showing {len(filtered_issues)} of {len(issues)} issues**")
        
        # Display issues
        for i, issue in enumerate(filtered_issues):
            severity_emoji = {
                Severity.CRITICAL: "🔴",
                Severity.HIGH: "🟠",
                Severity.MEDIUM: "🟡",
                Severity.LOW: "🔵"
            }[issue.severity]
            
            # Create expandable issue card
            with st.expander(
                f"{severity_emoji} **{issue.title}** | Line {issue.line_number} | {issue.category.value} | Confidence: {issue.confidence:.1f}",
                expanded=(i < 3 and issue.severity in [Severity.CRITICAL, Severity.HIGH])
            ):
                col_issue1, col_issue2 = st.columns([3, 1])
                
                with col_issue1:
                    st.markdown("**📝 Description:**")
                    st.write(issue.description)
                    
                    st.markdown("**💡 Suggested Fix:**")
                    st.write(issue.suggestion)
                    
                    if issue.code_snippet:
                        st.markdown("**📄 Code Snippet:**")
                        st.code(issue.code_snippet, language=language.lower())
                
                with col_issue2:
                    st.markdown("**📊 Issue Details:**")
                    st.write(f"**Severity:** {issue.severity.value}")
                    st.write(f"**Category:** {issue.category.value}")
                    st.write(f"**Line:** {issue.line_number}")
                    st.write(f"**Confidence:** {issue.confidence:.1f}")
                    
                    # Priority indicator
                    if issue.severity == Severity.CRITICAL:
                        st.error("🚨 **CRITICAL** - Fix immediately")
                    elif issue.severity == Severity.HIGH:
                        st.warning("⚠️ **HIGH** - Address soon")
                    elif issue.severity == Severity.MEDIUM:
                        st.info("📋 **MEDIUM** - Plan to fix")
                    else:
                        st.success("✓ **LOW** - Optional improvement")
    
    # Export and sharing section
    if st.session_state.analysis_results:
        st.header("📤 Export & Share Results")
        
        col_export1, col_export2, col_export3, col_export4 = st.columns(4)
        
        with col_export1:
            # JSON export
            if st.button("📋 Export JSON"):
                export_data = {
                    "analysis_result": st.session_state.analysis_results.dict(),
                    "model_info": {
                        "model_used": reviewer.current_model,
                        "provider": "OpenRouter",
                        "analysis_time": st.session_state.analysis_results.analysis_time
                    },
                    "code_info": st.session_state.get('file_info', {}),
                    "langchain_info": {
                        "framework": "LangChain",
                        "parsers_used": ["JsonOutputParser", "PydanticOutputParser"],
                        "chains_used": ["analysis_chain", "summary_chain"]
                    },
                    "timestamp": datetime.now().isoformat()
                }
                
                st.download_button(
                    "📥 Download Full Report",
                    json.dumps(export_data, indent=2, default=str),
                    file_name=f"langchain_openrouter_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    help="Complete analysis results with metadata"
                )
        
        with col_export2:
            # Summary export
            if st.button("📝 Export Summary"):
                summary_text = f"""# Code Review Summary
Generated by: {st.session_state.analysis_results.model_used}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Framework: LangChain + OpenRouter

## Quality Score
{st.session_state.analysis_results.quality_score}/100

## Issues Found
- Total: {len(st.session_state.analysis_results.issues)}
- Critical: {len([i for i in st.session_state.analysis_results.issues if i.severity == Severity.CRITICAL])}
- High: {len([i for i in st.session_state.analysis_results.issues if i.severity == Severity.HIGH])}
- Medium: {len([i for i in st.session_state.analysis_results.issues if i.severity == Severity.MEDIUM])}
- Low: {len([i for i in st.session_state.analysis_results.issues if i.severity == Severity.LOW])}

## Executive Summary
{st.session_state.analysis_results.summary}

## Key Recommendations
{chr(10).join(f"- {rec}" for rec in st.session_state.analysis_results.recommendations)}

## Usage Statistics
- Analysis Time: {st.session_state.analysis_results.analysis_time:.1f}s
- Tokens Used: {st.session_state.analysis_results.token_usage.get('total_tokens', 'N/A')}
- Estimated Cost: ${st.session_state.analysis_results.token_usage.get('estimated_cost', 0):.4f}
"""
                
                st.download_button(
                    "📥 Download Summary",
                    summary_text,
                    file_name=f"code_review_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
        
        with col_export3:
            # Usage stats
            if st.button("📊 Usage Stats"):
                st.json({
                    "model_calls": len(reviewer.callback.model_calls),
                    "total_tokens": reviewer.callback.total_tokens,
                    "estimated_cost": round(reviewer.callback.total_cost, 4),
                    "current_model": reviewer.current_model,
                    "analysis_time": st.session_state.analysis_results.analysis_time,
                    "issues_found": len(st.session_state.analysis_results.issues)
                })
        
        with col_export4:
            # Share link (placeholder)
            if st.button("🔗 Share Results"):
                st.info("🚧 Share functionality coming soon!")
                st.write("For now, use the export options to share your analysis results.")

# Advanced features demonstration
def show_advanced_langchain_patterns():
    """Demonstrate advanced LangChain patterns"""
    st.header("🚀 Advanced LangChain Patterns")
    
    with st.expander("🔗 Chain Composition Examples"):
        st.code("""
# LangChain Expression Language (LCEL) examples

# Basic chain composition
analysis_chain = prompt | llm | json_parser

# Parallel processing multiple aspects
parallel_analysis = RunnableParallel({
    "security": security_prompt | llm | json_parser,
    "performance": performance_prompt | llm | json_parser,
    "maintainability": maintainability_prompt | llm | json_parser
})

# Conditional routing based on code complexity
def route_by_complexity(input_data):
    lines = len(input_data["code"].split('\n'))
    if lines > 1000:
        return detailed_analysis_chain
    else:
        return quick_analysis_chain

routing_chain = RunnableLambda(route_by_complexity)

# Sequential chain with memory
chain_with_context = (
    RunnablePassthrough.assign(
        history=lambda x: memory.chat_memory.messages
    )
    | contextualized_prompt
    | llm
    | json_parser
)
        """, language="python")
    
    with st.expander("🎯 Structured Output with Pydantic"):
        st.code("""
# Type-safe output models
class SecurityVulnerability(BaseModel):
    cve_id: Optional[str] = None
    severity: Literal["Critical", "High", "Medium", "Low"]
    vulnerability_type: str
    affected_lines: List[int]
    exploitability_score: float = Field(ge=0.0, le=10.0)
    remediation_steps: List[str]
    references: List[str] = Field(default_factory=list)

class PerformanceIssue(BaseModel):
    bottleneck_type: str
    current_complexity: str  # e.g., "O(n²)"
    optimized_complexity: str  # e.g., "O(n log n)"
    performance_impact: Literal["High", "Medium", "Low"]
    optimization_suggestion: str
    code_example: str

# Parser integration
security_parser = PydanticOutputParser(pydantic_object=SecurityVulnerability)
performance_parser = PydanticOutputParser(pydantic_object=PerformanceIssue)

# Chain with structured output
security_chain = (
    security_prompt.partial(
        format_instructions=security_parser.get_format_instructions()
    )
    | llm
    | security_parser
)
        """, language="python")
    
    with st.expander("📊 Custom Callbacks and Monitoring"):
        st.code("""
class ComprehensiveCallback(BaseCallbackHandler):
    def __init__(self):
        self.metrics = {
            "start_time": None,
            "end_time": None,
            "token_usage": {},
            "model_switches": [],
            "error_count": 0,
            "chain_executions": []
        }
    
    def on_chain_start(self, serialized, inputs, **kwargs):
        self.metrics["start_time"] = datetime.now()
        self.metrics["chain_executions"].append({
            "chain": serialized.get("id", ["unknown"])[-1],
            "start_time": datetime.now(),
            "inputs_size": len(str(inputs))
        })
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        model_name = serialized.get("id", ["unknown"])[-1]
        self.metrics["model_switches"].append({
            "model": model_name,
            "timestamp": datetime.now(),
            "prompt_tokens": sum(len(p) for p in prompts)
        })
    
    def on_llm_end(self, response, **kwargs):
        if hasattr(response, 'llm_output'):
            usage = response.llm_output.get('token_usage', {})
            for key, value in usage.items():
                if key in self.metrics["token_usage"]:
                    self.metrics["token_usage"][key] += value
                else:
                    self.metrics["token_usage"][key] = value
    
    def on_chain_error(self, error, **kwargs):
        self.metrics["error_count"] += 1
        logger.error(f"Chain error: {error}")
    
    def get_performance_report(self):
        total_time = (self.metrics["end_time"] - self.metrics["start_time"]).total_seconds()
        return {
            "total_execution_time": total_time,
            "chains_executed": len(self.metrics["chain_executions"]),
            "models_used": len(set(m["model"] for m in self.metrics["model_switches"])),
            "total_tokens": self.metrics["token_usage"].get("total_tokens", 0),
            "errors_encountered": self.metrics["error_count"]
        }

# Usage
callback = ComprehensiveCallback()
chain = prompt | llm.with_config(callbacks=[callback]) | parser
result = chain.invoke({"code": code_content})
performance_report = callback.get_performance_report()
        """, language="python")

if __name__ == "__main__":
    main()
    
    # Advanced features toggle
    with st.sidebar:
        st.divider()
        if st.checkbox("🚀 Show Advanced Patterns", help="Display advanced LangChain usage examples"):
            show_advanced_langchain_patterns()
