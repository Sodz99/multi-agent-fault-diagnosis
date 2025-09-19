"""Simple CrewAI agent definitions for fault diagnosis MVP."""
from __future__ import annotations

import os
import boto3
import botocore
from typing import Dict, Any, Optional
from crewai import Agent, LLM
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Amazon Titan model configuration for MVP
SIMPLE_MODEL_CONFIG = {
    "planner": "bedrock/amazon.titan-text-express-v1",     # Fast orchestration
    "retriever": "bedrock/amazon.titan-text-express-v1",  # Quick information extraction
    "reasoner": "bedrock/amazon.titan-text-premier-v1:0", # Complex analysis
    "reporter": "bedrock/amazon.titan-text-express-v1"    # Documentation
}


def validate_aws_credentials() -> Dict[str, Any]:
    """Validate AWS credentials and environment setup."""
    validation_result = {
        "credentials_valid": False,
        "region_configured": False,
        "bedrock_accessible": False,
        "error_message": None,
        "region": None
    }

    try:
        # Check environment variables
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_region = os.getenv("AWS_DEFAULT_REGION") or os.getenv("BEDROCK_REGION", "us-east-1")

        if not aws_access_key or not aws_secret_key:
            validation_result["error_message"] = "Missing AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY in environment"
            return validation_result

        validation_result["credentials_valid"] = True
        validation_result["region_configured"] = bool(aws_region)
        validation_result["region"] = aws_region

        # Test AWS session and Bedrock access
        session = boto3.Session(
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region
        )

        # Try to create Bedrock client (use 'bedrock' not 'bedrock-runtime' for listing models)
        bedrock_client = session.client('bedrock')

        # Test basic connectivity by listing models (this requires minimal permissions)
        try:
            # This is a lightweight call to test connectivity
            models = bedrock_client.list_foundation_models()
            validation_result["bedrock_accessible"] = True
        except botocore.exceptions.ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code in ['UnauthorizedOperation', 'AccessDenied']:
                # Client works but may need permissions - this is still a successful connection
                validation_result["bedrock_accessible"] = True
                validation_result["error_message"] = f"Bedrock accessible but may need additional permissions: {error_code}"
            else:
                validation_result["error_message"] = f"Bedrock connection failed: {error_code}"
        except Exception as e:
            validation_result["error_message"] = f"Bedrock client creation failed: {str(e)}"

    except Exception as e:
        validation_result["error_message"] = f"AWS validation failed: {str(e)}"

    return validation_result


def test_bedrock_model_access(model_id: str, region: str = None) -> bool:
    """Test if a specific Bedrock model is accessible."""
    try:
        if region is None:
            region = os.getenv("AWS_DEFAULT_REGION") or os.getenv("BEDROCK_REGION", "us-east-1")

        session = boto3.Session(region_name=region)
        bedrock_client = session.client('bedrock')

        # Convert LiteLLM format to pure model ID
        clean_model_id = model_id.replace("bedrock/", "")

        # Try to get model details
        bedrock_client.get_foundation_model(modelIdentifier=clean_model_id)
        return True

    except botocore.exceptions.ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        if error_code == 'ValidationException':
            # Model might not exist in this region
            return False
        elif error_code in ['UnauthorizedOperation', 'AccessDenied']:
            # Model exists but we don't have permissions to describe it - assume it's accessible for inference
            return True
        return False
    except Exception:
        return False


def initialize_bedrock_for_litellm() -> bool:
    """Initialize AWS Bedrock configuration for LiteLLM."""
    try:
        # Validate AWS setup first
        validation = validate_aws_credentials()

        if not validation["credentials_valid"]:
            print(f"[Agents] AWS credentials validation failed: {validation['error_message']}")
            return False

        if not validation["region_configured"]:
            print(f"[Agents] AWS region not configured")
            return False

        # Set environment variables that LiteLLM expects
        region = validation["region"]
        os.environ["AWS_DEFAULT_REGION"] = region
        os.environ["AWS_REGION"] = region

        # Optionally set up a default session for boto3
        # This can help with cases where LiteLLM doesn't pick up credentials properly
        boto3.setup_default_session(region_name=region)

        if validation["bedrock_accessible"]:
            print(f"[Agents] Bedrock initialized for LiteLLM in region: {region}")
            return True
        else:
            print(f"[Agents] Bedrock may not be accessible: {validation['error_message']}")
            return False

    except Exception as e:
        print(f"[Agents] Failed to initialize Bedrock for LiteLLM: {e}")
        return False


def create_bedrock_llm(model_id: Optional[str] = None, agent_role: Optional[str] = None) -> LLM:
    """Create a simple Bedrock LLM instance using CrewAI's format.

    Args:
        model_id: Specific model ID (optional)
        agent_role: Agent role for model selection (optional)

    Returns:
        LLM: CrewAI LLM instance configured for Bedrock
    """
    if model_id is None:
        if agent_role and agent_role in SIMPLE_MODEL_CONFIG:
            model_string = SIMPLE_MODEL_CONFIG[agent_role]
        else:
            # Default to Titan Premier for general use
            model_string = "bedrock/amazon.titan-text-premier-v1:0"
    else:
        # Ensure proper bedrock/ prefix for LiteLLM format
        if not model_id.startswith("bedrock/"):
            model_string = f"bedrock/{model_id}"
        else:
            model_string = model_id

    return LLM(model=model_string)


def validate_llm_client(llm: LLM, role: str = "unknown", verbose: bool = True) -> bool:
    """Validate that an LLM object has a properly initialized client.

    Args:
        llm: The LLM object to validate
        role: Role name for logging purposes
        verbose: Enable detailed logging

    Returns:
        bool: True if client is properly initialized, False otherwise
    """
    if llm is None:
        if verbose:
            print(f"[Agents] ERROR: {role} LLM is None")
        return False

    try:
        # Check if the LLM object has the expected attributes
        if not hasattr(llm, 'model'):
            if verbose:
                print(f"[Agents] ERROR: {role} LLM missing 'model' attribute")
            return False

        # Try to access internal client-related attributes that CrewAI might use
        # This is a lightweight check to ensure the LLM is properly constructed
        model_name = llm.model
        if not model_name or not isinstance(model_name, str):
            if verbose:
                print(f"[Agents] ERROR: {role} LLM has invalid model name: {model_name}")
            return False

        # For Bedrock models, ensure proper format and test accessibility
        if model_name.startswith("bedrock/"):
            model_id = model_name.replace("bedrock/", "")
            if not model_id:
                if verbose:
                    print(f"[Agents] ERROR: {role} LLM has empty Bedrock model ID")
                return False

            # Test actual Bedrock model accessibility
            region = os.getenv("AWS_DEFAULT_REGION") or os.getenv("BEDROCK_REGION", "us-east-1")
            if not test_bedrock_model_access(model_name, region):
                if verbose:
                    print(f"[Agents] WARNING: {role} LLM model {model_id} may not be accessible in region {region}")
                # Don't fail validation for this, just warn - the model might still work for inference

        # Perform a lightweight validation to check if LLM can be used
        try:
            # Try to access any internal properties that CrewAI might use
            # This is to catch cases where the LLM object exists but has None internal clients
            if hasattr(llm, '_client') and llm._client is None:
                if verbose:
                    print(f"[Agents] ERROR: {role} LLM has None internal client")
                return False
        except AttributeError:
            # It's okay if _client doesn't exist - it might be initialized differently
            pass

        if verbose:
            print(f"[Agents] SUCCESS: {role} LLM client validation passed")
        return True

    except Exception as e:
        if verbose:
            print(f"[Agents] ERROR: {role} LLM client validation failed: {e}")
        return False


class FaultDiagnosisAgents:
    """Simple factory for creating fault diagnosis crew agents."""

    def __init__(self, custom_models: Optional[Dict[str, str]] = None, verbose: bool = True):
        """Initialize with simple model configuration.

        Args:
            custom_models: Override default model assignments per role
            verbose: Enable detailed logging
        """
        self.verbose = verbose

        # Initialize Bedrock configuration for LiteLLM first
        if self.verbose:
            print(f"[Agents] Initializing AWS Bedrock for LiteLLM...")

        bedrock_ready = initialize_bedrock_for_litellm()
        if not bedrock_ready:
            if self.verbose:
                print(f"[Agents] WARNING: Bedrock initialization failed, but continuing with LLM creation...")

        # Use custom models or simple defaults
        self.model_config = custom_models or SIMPLE_MODEL_CONFIG.copy()

        # Validate environment setup
        validation = validate_aws_credentials()
        if self.verbose:
            if validation["credentials_valid"]:
                print(f"[Agents] SUCCESS: AWS credentials validated for region: {validation['region']}")
            else:
                print(f"[Agents] WARNING: AWS validation issue: {validation['error_message']}")

        # Create role-specific LLM instances
        self.llms = {}
        successful_llms = 0

        for role, model_string in self.model_config.items():
            try:
                # Test model access before creating LLM
                if validation["bedrock_accessible"]:
                    model_accessible = test_bedrock_model_access(model_string, validation["region"])
                    if not model_accessible and self.verbose:
                        print(f"[Agents] WARNING: Model {model_string} may not be accessible in {validation['region']}")

                self.llms[role] = LLM(model=model_string)
                successful_llms += 1
                if self.verbose:
                    print(f"[Agents] SUCCESS: Created {role} LLM: {model_string}")

            except Exception as e:
                if self.verbose:
                    print(f"[Agents] ERROR: Failed to create {role} LLM: {e}")
                    print(f"[Agents] Attempting fallback for {role}...")

                # Try fallback to default Titan Premier
                try:
                    self.llms[role] = LLM(model="bedrock/amazon.titan-text-premier-v1:0")
                    successful_llms += 1
                    if self.verbose:
                        print(f"[Agents] SUCCESS: Created {role} LLM with fallback model")
                except Exception as fallback_error:
                    if self.verbose:
                        print(f"[Agents] ERROR: Fallback also failed for {role}: {fallback_error}")
                    # Create a placeholder - this will likely fail at runtime but allows initialization to complete
                    self.llms[role] = None

        # Validate all created LLMs
        if self.verbose:
            print(f"[Agents] Validating {successful_llms} created LLMs...")

        validated_llms = 0
        for role, llm in self.llms.items():
            if llm is not None and validate_llm_client(llm, role, self.verbose):
                validated_llms += 1

        if self.verbose:
            print(f"[Agents] Simple FaultDiagnosisAgents initialized with {successful_llms}/{len(self.model_config)} LLMs ({validated_llms} validated)")

        # Store validation status for diagnostics
        self.validation_status = {
            "total_roles": len(self.model_config),
            "successful_llms": successful_llms,
            "validated_llms": validated_llms,
            "aws_validation": validation
        }

    def create_planner_agent(self) -> Agent:
        """Create the NOC Sentinel Planner agent."""
        if self.verbose:
            print(f"[Agents] Creating planner agent")

        planner_llm = self.llms.get("planner")
        if planner_llm is None:
            raise RuntimeError("Failed to create planner agent: LLM is not available. Check AWS Bedrock configuration.")

        # Validate LLM client before agent creation
        if not validate_llm_client(planner_llm, "planner", self.verbose):
            raise RuntimeError("Failed to create planner agent: LLM client validation failed. Check AWS Bedrock configuration and connectivity.")

        agent = Agent(
            role="NOC Sentinel Planner",
            goal=(
                "Analyze incoming telecom alerts, classify severity, prioritize actions, "
                "and coordinate the fault diagnosis workflow across specialist teams."
            ),
            backstory=(
                "You are an experienced NOC (Network Operations Center) sentinel with 15+ years "
                "monitoring telecom infrastructure. You excel at rapid alert triage, understanding "
                "the cascading effects of network issues, and orchestrating response teams. Your "
                "specialty is quickly identifying the most critical issues and ensuring the right "
                "specialists are engaged with proper context and priority."
            ),
            verbose=True,
            allow_delegation=False,  # Fix: No delegation tools available
            llm=planner_llm,
            max_iter=3,
            max_execution_time=120,  # 2-minute timeout for planning
            memory=True,
        )
        if self.verbose:
            print(f"[Agents] SUCCESS: Planner agent created")
        return agent

    def create_retriever_agent(self) -> Agent:
        """Create the Core Network Analyst Retriever agent."""
        if self.verbose:
            print(f"[Agents] Creating retriever agent")

        retriever_llm = self.llms.get("retriever")
        if retriever_llm is None:
            raise RuntimeError("Failed to create retriever agent: LLM is not available. Check AWS Bedrock configuration.")

        # Validate LLM client before agent creation
        if not validate_llm_client(retriever_llm, "retriever", self.verbose):
            raise RuntimeError("Failed to create retriever agent: LLM client validation failed. Check AWS Bedrock configuration and connectivity.")

        agent = Agent(
            role="Core Network Analyst",
            goal=(
                "Retrieve relevant historical incidents, runbooks, topology data, and technical "
                "documentation to support fault diagnosis decisions. Ensure all evidence is "
                "properly cited and grounded in authoritative sources."
            ),
            backstory=(
                "You are a core network analyst with deep expertise in telecom infrastructure "
                "and extensive knowledge of network topology, protocols, and failure patterns. "
                "You maintain the organization's knowledge base of incidents, runbooks, and "
                "technical documentation. Your strength is quickly finding the most relevant "
                "historical context and technical references for any network issue."
            ),
            verbose=True,
            allow_delegation=False,  # Specialist shouldn't delegate
            llm=retriever_llm,
            max_iter=2,
            max_execution_time=60,  # 1-minute timeout for retrieval
            memory=True,
        )
        if self.verbose:
            print(f"[Agents] SUCCESS: Retriever agent created")
        return agent

    def create_reasoner_agent(self) -> Agent:
        """Create the Hypothesis Chair Reasoner agent."""
        if self.verbose:
            print(f"[Agents] Creating reasoner agent")

        reasoner_llm = self.llms.get("reasoner")
        if reasoner_llm is None:
            raise RuntimeError("Failed to create reasoner agent: LLM is not available. Check AWS Bedrock configuration.")

        # Validate LLM client before agent creation
        if not validate_llm_client(reasoner_llm, "reasoner", self.verbose):
            raise RuntimeError("Failed to create reasoner agent: LLM client validation failed. Check AWS Bedrock configuration and connectivity.")

        agent = Agent(
            role="Hypothesis Chair",
            goal=(
                "Synthesize evidence into ranked hypotheses about root causes, validate each "
                "hypothesis through technical analysis, and coordinate with validator systems "
                "to ensure all conclusions are grounded and actionable."
            ),
            backstory=(
                "You are a senior network engineer and incident response specialist who excels "
                "at root cause analysis. With 20+ years of experience diagnosing complex telecom "
                "failures, you can synthesize multiple data sources into clear, testable hypotheses. "
                "You're known for your methodical approach, attention to technical detail, and "
                "ability to avoid false conclusions through rigorous validation."
            ),
            verbose=True,
            allow_delegation=False,   # Fix: No delegation tools available
            llm=reasoner_llm,
            max_iter=4,
            max_execution_time=300,  # 5-minute timeout for complex reasoning
            memory=True,
        )
        if self.verbose:
            print(f"[Agents] SUCCESS: Reasoner agent created")
        return agent

    def create_reporter_agent(self) -> Agent:
        """Create the Postmortem Writer Reporter agent."""
        if self.verbose:
            print(f"[Agents] Creating reporter agent")

        reporter_llm = self.llms.get("reporter")
        if reporter_llm is None:
            raise RuntimeError("Failed to create reporter agent: LLM is not available. Check AWS Bedrock configuration.")

        # Validate LLM client before agent creation
        if not validate_llm_client(reporter_llm, "reporter", self.verbose):
            raise RuntimeError("Failed to create reporter agent: LLM client validation failed. Check AWS Bedrock configuration and connectivity.")

        agent = Agent(
            role="Postmortem Writer",
            goal=(
                "Document the fault diagnosis process, create executive summaries, generate "
                "remediation plans, and produce stakeholder-ready reports with proper citations "
                "and actionable recommendations."
            ),
            backstory=(
                "You are a technical communication specialist with expertise in incident response "
                "documentation. You excel at translating complex technical findings into clear, "
                "actionable reports for different audiences - from engineers to executives. Your "
                "reports are known for their clarity, completeness, and practical remediation "
                "guidance that helps prevent similar incidents."
            ),
            verbose=True,
            allow_delegation=False,  # Specialist shouldn't delegate
            llm=reporter_llm,
            max_iter=2,
            max_execution_time=180,  # 3-minute timeout for reporting
            memory=True,
        )
        if self.verbose:
            print(f"[Agents] SUCCESS: Reporter agent created")
        return agent


# Keep the existing data classes for backwards compatibility
from .crew import Hypothesis, CrewMessage, CrewTranscriptBuilder

__all__ = [
    "FaultDiagnosisAgents",
    "create_bedrock_llm",
    "Hypothesis",
    "CrewMessage",
    "CrewTranscriptBuilder",
    "SIMPLE_MODEL_CONFIG",
]