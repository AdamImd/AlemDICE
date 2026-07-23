"""
Robust Naive Agent for Alem LLM Evaluation.

Based on BALROG's RobustNaiveAgent scaffolding with additional robustness:
  - Structured output tags (<action>...</action>) for reliable parsing
  - Multi-strategy fallback extraction: tag → ACTION: prefix → substring → fuzzy
  - Retry with error feedback when extraction fails (up to MAX_RETRIES)
  - Action validation against the valid action list before returning
  - Retry statistics tracking for analysis
"""

import copy
import logging

from alem_action_parser import (
    CANONICAL_ACTIONS,
)
from alem_action_parser import (
    extract_action_multistrategy as _extract_action_multistrategy,
)
from alem_action_parser import (
    fuzzy_match_action as _fuzzy_match_action,
)
from alem_action_parser import (
    validate_action as _validate_action,
)

try:
    from .base import BaseAgent
except ImportError:
    from eval_utils.agents.base import BaseAgent

logger = logging.getLogger(__name__)

# ============================================================================
# Valid actions & helpers (shared with robust_cot.py)
# ============================================================================

VALID_ACTIONS = list(CANONICAL_ACTIONS)


def validate_action(candidate):
    """Compatibility wrapper around the shared parser's exact matcher."""

    return _validate_action(candidate, VALID_ACTIONS)


def fuzzy_match_action(candidate, cutoff=0.6):
    """Compatibility wrapper around the shared parser's fuzzy matcher."""

    return _fuzzy_match_action(candidate, VALID_ACTIONS, cutoff=cutoff)


def extract_action_multistrategy(completion_text):
    """Compatibility wrapper around the shared multi-strategy parser."""

    return _extract_action_multistrategy(completion_text, VALID_ACTIONS)


# ============================================================================
# RobustNaiveAgent
# ============================================================================


class RobustNaiveAgent(BaseAgent):
    """An agent that generates actions based on observations without complex reasoning.

    Uses <action>...</action> XML tags for reliable action extraction,
    with multi-strategy fallback parsing and retry-with-feedback on failure.
    """

    MAX_RETRIES = 0  # Additional attempts after first failure

    def __init__(self, client_factory, prompt_builder):
        """Initialize the RobustNaiveAgent with a client and prompt builder."""
        super().__init__(client_factory, prompt_builder)
        self.step_count = 0
        self.total_retries = 0
        self.total_parse_failures = 0

    NAIVE_INSTRUCTION = (
        "You must choose exactly one action from the action list and output it in the following format:\n"
        "<action>YOUR_CHOSEN_ACTION</action>\n"
        "Output no other text, explanation, or reasoning."
    )

    def build_prompt(self, obs, prev_action=None):
        """Build the full prompt messages for this step (without calling the LLM)."""
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)
        messages = self.prompt_builder.get_prompt()
        if messages and messages[-1].role == "user":
            messages[-1].content += "\n\n" + self.NAIVE_INSTRUCTION
        return messages

    def act(self, obs, prev_action=None):
        """Generate the next action based on the observation and previous action.

        Args:
            obs (dict): The current observation in the environment.
            prev_action (str, optional): The previous action taken.

        Returns:
            LLMResponse: The response with the extracted action in `completion`.
        """
        if prev_action:
            self.prompt_builder.update_action(prev_action)

        self.prompt_builder.update_observation(obs)

        messages = self.prompt_builder.get_prompt()

        naive_instruction = self.NAIVE_INSTRUCTION

        if messages and messages[-1].role == "user":
            messages[-1].content += "\n\n" + naive_instruction

        _, last_response, extracted, retries = self.client.generate_with_validation(
            messages,
            validate_fn=lambda r: extract_action_multistrategy(r.completion),
            error_message=(
                "Your output did not contain a valid action. "
                "You must output exactly one action from the game's action list "
                "using the format: <action>YOUR_CHOSEN_ACTION</action>\n"
                "Try again."
            ),
            max_parse_retries=self.MAX_RETRIES,
        )
        self.step_count += 1
        self.total_retries += retries

        if extracted is None:
            self.total_parse_failures += 1
            extracted = "Noop"
            logger.warning(
                f"RobustNaiveAgent step {self.step_count}: failed to parse after "
                f"{retries + 1} attempts. Raw: '{last_response.completion[:200]}'. "
                f"Defaulting to Noop."
            )

        # Save raw outputs before extraction/cleaning (for debug logging)
        self._last_raw_completion = last_response.completion
        self._last_raw_reasoning = last_response.reasoning

        final_answer = self._extract_final_answer(last_response, extracted)

        if self.step_count % 50 == 0:
            logger.info(
                f"RobustNaiveAgent step {self.step_count}: action='{final_answer.completion}', "
                f"retries_total={self.total_retries}, failures={self.total_parse_failures}"
            )

        return final_answer

    def _extract_final_answer(self, answer, extracted_action):
        """Build the final LLMResponse with the extracted action.

        Args:
            answer (LLMResponse): The raw response from the LLM.
            extracted_action (str): The validated action name.

        Returns:
            LLMResponse: A copy of answer with `completion` set to extracted_action
                and `reasoning` set to the raw LLM output (for evaluator diagnostics).
        """
        final_answer = copy.deepcopy(answer)
        final_answer = final_answer._replace(
            reasoning=None,  # robust_naive doesn't reason; raw API data in _last_raw_reasoning
            completion=extracted_action,
        )
        return final_answer

    def reset(self):
        """Reset the agent state for a new episode."""
        super().reset()
        self.step_count = 0
        self.total_retries = 0
        self.total_parse_failures = 0

    def get_retry_stats(self):
        """Return retry/parse statistics for logging and analysis."""
        return {
            "total_steps": self.step_count,
            "total_retries": self.total_retries,
            "total_parse_failures": self.total_parse_failures,
            "retry_rate": round(self.total_retries / max(self.step_count, 1), 4),
            "parse_failure_rate": round(self.total_parse_failures / max(self.step_count, 1), 4),
        }
