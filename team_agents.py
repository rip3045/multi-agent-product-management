# team_agents.py

"""
Module: team_agents
Description: This module defines various agent classes representing team members
             in an automated coding system. Each agent encapsulates specific
             responsibilities and interacts with a language model backend
             (Ollama) via LangChain v0.2.

Agents Included:
- SoftwareArchitectAgent
- CoderAgent
- CodeReviewerAgent
- ProductManagerAgent
- TesterAgent
- DevOpsAgent
- SecurityAnalystAgent
- UXUIDesignerAgent
- BusinessAnalystAgent
- ProjectManagerAgent
- PerformanceEngineerAgent
- AccessibilityExpertAgent
- CustomerFeedbackAgent
- DocumentationSpecialistAgent

Usage:
Import this module and instantiate the agents as needed in your workflow.

Prerequisites:
- LangChain v0.2
- Ollama LLM backend
"""

import logging
import sys
import io
from langchain_community.llms.ollama import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain


# Initialize the Ollama LLM once to be shared among agents
llm = Ollama(model="llama3.1:8b")


# Configure logging
def setup_logger(name, log_file, level=logging.INFO):
    """Function to configure logging for each agent."""
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


class SoftwareArchitectAgent:
    """Designs software modules based on a product description."""

    def __init__(self, llm_instance=llm):
        self.llm = llm_instance
        self.logger = setup_logger('SoftwareArchitectAgent', 'software_architect_agent.log')
        self.prompt = PromptTemplate(
            input_variables=["product_description"],
            template=(
                "As a software architect, design the modules for the following product:\n\n"
                "{product_description}\n\nModules with descriptions:"
            ),
        )

    def design_modules(self, product_description):
        self.logger.info(f"Received product description: {product_description}")
        chain = LLMChain(llm=self.llm, prompt=self.prompt)
        response = chain.run(product_description)
        self.logger.info(f"Architect response: {response}")

        # Parse the response into modules and descriptions
        modules = []
        lines = response.strip().split('\n')
        for line in lines:
            if line.strip() and ": " in line:
                name, desc = line.split(": ", 1)
                modules.append({"name": name.strip(), "description": desc.strip()})

        self.logger.info(f"Parsed modules: {modules}")
        return modules


class CoderAgent:
    """Implements code for specified modules."""

    def __init__(self, llm_instance=llm):
        self.llm = llm_instance
        self.logger = setup_logger('CoderAgent', 'coder_agent.log')
        self.prompt = PromptTemplate(
            input_variables=["module_name", "module_description"],
            template=(
                "As a coder, implement the following module in Python:\n\n"
                "Module: {module_name}\n\nDescription: {module_description}\n\nCode:"
            ),
        )

    def code_module(self, module_name, module_description):
        self.logger.info(f"Received module name: {module_name}, description: {module_description}")
        chain = LLMChain(llm=self.llm, prompt=self.prompt)
        code = chain.run({"module_name": module_name, "module_description": module_description})
        self.logger.info(f"Generated code for module {module_name}: {code}")
        return code


class CodeReviewerAgent:
    """Reviews and documents code for errors and best practices."""

    def __init__(self, llm_instance=llm):
        self.llm = llm_instance
        self.logger = setup_logger('CodeReviewerAgent', 'code_reviewer_agent.log')
        self.review_prompt = PromptTemplate(
            input_variables=["code"],
            template=(
                "As a code reviewer, review the following Python code for errors and best practices:\n\n"
                "{code}\n\nReview:"
            ),
        )
        self.documentation_prompt = PromptTemplate(
            input_variables=["code"],
            template=(
                "As a documenter, provide documentation for the following Python code:\n\n"
                "{code}\n\nDocumentation:"
            ),
        )

    def review_code(self, code):
        self.logger.info(f"Received code for review: {code}")
        chain = LLMChain(llm=self.llm, prompt=self.review_prompt)
        review = chain.run(code)
        self.logger.info(f"Review response: {review}")
        return review

    def document_code(self, code):
        self.logger.info(f"Received code for documentation: {code}")
        chain = LLMChain(llm=self.llm, prompt=self.documentation_prompt)
        documentation = chain.run(code)
        self.logger.info(f"Documentation response: {documentation}")
        return documentation


class ProductManagerAgent:
    """Attempts to run the code and reviews the output."""

    def __init__(self):
        self.logger = setup_logger('ProductManagerAgent', 'product_manager_agent.log')
        self.errors = []

    def run_and_review(self, code):
        self.logger.info(f"Received code for execution:\n{code}")

        # Dictionary to hold the global execution environment
        exec_globals = {}

        # Capture standard output
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout

        try:
            # Execute the code
            exec(code, exec_globals)

            # Capture the output (anything printed)
            output = new_stdout.getvalue()
            self.logger.info(f"Execution output:\n{output}")

            # Check for specific variables or function outputs in exec_globals
            if 'result' in exec_globals:
                self.logger.info(f"Result of execution: {exec_globals['result']}")
            else:
                self.logger.info("No 'result' variable found in executed code.")

            # Return successful output for further processing
            return True, output

        except Exception as e:
            # Capture the error and log it
            self.logger.error(f"Execution failed with error: {e}")
            self.errors.append(str(e))
            return False, str(e)

        finally:
            # Restore the original stdout
            sys.stdout = old_stdout

    def is_satisfactory(self):
        """Checks if there are any errors in the execution process."""
        return len(self.errors) == 0

class TesterAgent:
    """Writes and runs tests for the code."""

    def __init__(self, llm_instance=llm):
        self.llm = llm_instance
        self.test_prompt = PromptTemplate(
            input_variables=["code"],
            template=(
                "As a tester, write unit tests for the following Python code:\n\n"
                "{code}\n\nUnit Tests:"
            ),
        )

    def write_tests(self, code):
        chain = LLMChain(llm=self.llm, prompt=self.test_prompt)
        test_code = chain.run(code)
        return test_code

    def run_tests(self, code, test_code):
        # Placeholder for test execution logic
        pass


class DevOpsAgent:
    """Manages deployment pipelines and ensures smooth deployment."""

    def __init__(self, llm_instance=llm):
        self.llm = llm_instance
        self.deployment_prompt = PromptTemplate(
            input_variables=["codebase"],
            template=(
                "As a DevOps engineer, write a Dockerfile for the following Python codebase:\n\n"
                "{codebase}\n\nDockerfile:"
            ),
        )

    def create_dockerfile(self, codebase):
        chain = LLMChain(llm=self.llm, prompt=self.deployment_prompt)
        dockerfile = chain.run(codebase)
        return dockerfile

    def simulate_deployment(self, dockerfile):
        # Placeholder for deployment simulation logic
        pass


class SecurityAnalystAgent:
    """Analyzes code for security vulnerabilities."""

    def __init__(self, llm_instance=llm):
        self.llm = llm_instance
        self.security_prompt = PromptTemplate(
            input_variables=["code"],
            template=(
                "As a security analyst, review the following Python code for security vulnerabilities:\n\n"
                "{code}\n\nSecurity Analysis:"
            ),
        )

    def analyze_security(self, code):
        chain = LLMChain(llm=self.llm, prompt=self.security_prompt)
        analysis = chain.run(code)
        return analysis


class UXUIDesignerAgent:
    """Provides design guidelines for user interface and experience."""

    def __init__(self, llm_instance=llm):
        self.llm = llm_instance
        self.design_prompt = PromptTemplate(
            input_variables=["product_description"],
            template=(
                "As a UX/UI designer, provide design guidelines for the following product:\n\n"
                "{product_description}\n\nDesign Guidelines:"
            ),
        )

    def create_design_guidelines(self, product_description):
        chain = LLMChain(llm=self.llm, prompt=self.design_prompt)
        guidelines = chain.run(product_description)
        return guidelines


class BusinessAnalystAgent:
    """Analyzes product descriptions to provide detailed requirements."""

    def __init__(self, llm_instance=llm):
        self.llm = llm_instance
        self.analysis_prompt = PromptTemplate(
            input_variables=["product_description"],
            template=(
                "As a business analyst, analyze the following product description and provide detailed requirements:\n\n"
                "{product_description}\n\nRequirements:"
            ),
        )

    def analyze_requirements(self, product_description):
        chain = LLMChain(llm=self.llm, prompt=self.analysis_prompt)
        requirements = chain.run(product_description)
        return requirements


class ProjectManagerAgent:
    """Oversees the project, manages timelines, and coordinates agents."""

    def __init__(self):
        self.task_queue = []

    def assign_tasks(self, agents, tasks):
        for agent, task in zip(agents, tasks):
            print(f"Assigning task '{task}' to {agent.__class__.__name__}")
            self.task_queue.append((agent, task))

    def monitor_progress(self):
        # Placeholder for monitoring logic
        pass


class PerformanceEngineerAgent:
    """Optimizes code for performance and scalability."""

    def __init__(self, llm_instance=llm):
        self.llm = llm_instance
        self.performance_prompt = PromptTemplate(
            input_variables=["code"],
            template=(
                "As a performance engineer, analyze the following Python code for performance issues and suggest optimizations:\n\n"
                "{code}\n\nPerformance Analysis:"
            ),
        )

    def analyze_performance(self, code):
        chain = LLMChain(llm=self.llm, prompt=self.performance_prompt)
        analysis = chain.run(code)
        return analysis


class AccessibilityExpertAgent:
    """Ensures the product is accessible to users with disabilities."""

    def __init__(self, llm_instance=llm):
        self.llm = llm_instance
        self.accessibility_prompt = PromptTemplate(
            input_variables=["ui_code"],
            template=(
                "As an accessibility expert, review the following UI code for accessibility issues:\n\n"
                "{ui_code}\n\nAccessibility Analysis:"
            ),
        )

    def analyze_accessibility(self, ui_code):
        chain = LLMChain(llm=self.llm, prompt=self.accessibility_prompt)
        analysis = chain.run(ui_code)
        return analysis


class CustomerFeedbackAgent:
    """Simulates user feedback and collects bug reports."""

    def __init__(self, llm_instance=llm):
        self.llm = llm_instance
        self.feedback_prompt = PromptTemplate(
            input_variables=["product_demo"],
            template=(
                "As a customer, provide feedback on the following product demo:\n\n"
                "{product_demo}\n\nFeedback:"
            ),
        )

    def provide_feedback(self, product_demo):
        chain = LLMChain(llm=self.llm, prompt=self.feedback_prompt)
        feedback = chain.run(product_demo)
        return feedback


class DocumentationSpecialistAgent:
    """Creates comprehensive documentation for users and developers."""

    def __init__(self, llm_instance=llm):
        self.llm = llm_instance
        self.documentation_prompt = PromptTemplate(
            input_variables=["codebase"],
            template=(
                "As a documentation specialist, write comprehensive documentation for the following codebase:\n\n"
                "{codebase}\n\nDocumentation:"
            ),
        )

    def write_documentation(self, codebase):
        chain = LLMChain(llm=self.llm, prompt=self.documentation_prompt)
        documentation = chain.run(codebase)
        return documentation